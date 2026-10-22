from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from Dynamixel_win_pro_hand_book.spacer_virtual_tcp import (
    SpacerVirtualTcpGeometry,
    SpacerVirtualTcpMechanismState,
    calculate_right_tilted_desired_tip_position,
    calculate_sp_lin_displacement_mm,
    calculate_spacer_tip_position_robot,
    calculate_virtual_tcp_solution,
)

from xarm7.control.xarm_monitor import assert_xarm_normal

from storage.config import (
    RIGHT_TILTED_SYNC_STEP_DEG,
    RIGHT_TILTED_SIMULTANEOUS_VELOCITY_MM_S,
    RIGHT_TILTED_SIMULTANEOUS_ACCELERATION_MM_S2,
    RIGHT_TILTED_SIMULTANEOUS_Y_POSITION_TOLERANCE_MM,
    RIGHT_TILTED_SIMULTANEOUS_Z_POSITION_TOLERANCE_MM,
    RIGHT_TILTED_SIMULTANEOUS_YAW_TOLERANCE_DEG,
    SPACER_ROTATION_START_DELAY_SEC,
    SPACER_ROTATION_RESET_ADVANCE_SEC,
    APPROACH_BACKOFF_MM,
    STORAGE_TILTED_LOWER_MM,
    ENABLE_SPACER_VIRTUAL_TCP,
    SPACER_VIRTUAL_TCP_GEOMETRY,
    SPACER_VIRTUAL_TCP_STATE_READER,
    SPACER_VIRTUAL_TCP_UNAVAILABLE_REASON,
)

from storage.models import RightTiltedSupportPlan

from storage.planner import plan_right_tilted_rotation_steps


def reset_spacer_rotation_before_linear_retraction(
    hand_worker: Any,
) -> None:
    """Synchronously restore SP_ROT_0 before any linear retraction."""

    print(
        "[STORAGE SPACER] "
        "reset rotation before linear retraction"
    )
    print("[STORAGE SPACER] reset_rot start")

    hand_worker.reset_rot(
        asynchronous=False
    )

    print("[STORAGE SPACER] reset_rot done")


def open_right_tilted_with_y(
    *,
    hand_worker: Any,
    xarm7: Any,
    command_angle_deg: float,
    travel_y_mm: float,
    yaw_offset_deg: float = 6.0,
) -> None:
    """Start positive SP_ROT opening and base -Y/+Yaw translation concurrently.

    No angle/position step synchronization. Both operations must finish
    successfully before the insertion phase may begin.
    """
    angle = float(command_angle_deg)
    travel_y = float(travel_y_mm)
    yaw_offset = float(yaw_offset_deg)
    if not math.isfinite(yaw_offset) or yaw_offset <= 0.0:
        raise RuntimeError(f"invalid RIGHT tilted +Yaw offset: {yaw_offset}")
    if not math.isfinite(angle) or angle <= 0.0:
        raise RuntimeError(f"RIGHT tilted SP_ROT must be positive: {angle}")
    if not math.isfinite(travel_y) or travel_y >= 0.0:
        raise RuntimeError(f"RIGHT tilted Y travel must be negative: {travel_y}")

    before_pose = np.asarray(xarm7.get_tcp_pose(is_radian=True), dtype=float).reshape(6)
    rotation_errors: list[Exception] = []

    def rotate() -> None:
        try:
            hand_worker.rotate_spacer(angle)
        except Exception as exc:
            rotation_errors.append(exc)

    print("\n[RIGHT-TILTED SIMULTANEOUS OPEN + Y/YAW]")
    print(f"SP_ROT opening target = {angle:+.2f} deg")
    print(f"xArm base Y travel = {travel_y:+.2f} mm")
    print(f"xArm Yaw offset = {yaw_offset:+.2f} deg")
    thread = threading.Thread(target=rotate, name="right_tilted_open", daemon=False)
    thread.start()
    arm_error: Exception | None = None
    ret: Any = None
    try:
        ret = xarm7.moveL_relative(
            [0.0, travel_y, 0.0, 0.0, 0.0, math.radians(yaw_offset)],
            velocity=RIGHT_TILTED_SIMULTANEOUS_VELOCITY_MM_S,
            acceleration=RIGHT_TILTED_SIMULTANEOUS_ACCELERATION_MM_S2,
            asynchronous=False,
        )
    except Exception as exc:
        arm_error = exc
    finally:
        thread.join()

    if rotation_errors or arm_error is not None or ret != 0:
        raise RuntimeError(
            "RIGHT tilted simultaneous opening failed; abort before insertion: "
            f"SP_ROT={rotation_errors!r}, xArm_exception={arm_error!r}, "
            f"xArm_ret={ret!r}"
        )
    state = xarm7.get_state()
    error = int(xarm7.arm.error_code)
    if state in (4, 5) or error != 0:
        raise RuntimeError(
            f"xArm abnormal after RIGHT opening: state={state}, error={error}"
        )
    after_pose = np.asarray(xarm7.get_tcp_pose(is_radian=True), dtype=float).reshape(6)
    actual_delta_y = float(after_pose[1] - before_pose[1])
    actual_delta_z = float(after_pose[2] - before_pose[2])
    actual_delta_yaw_deg = math.degrees(math.atan2(
        math.sin(after_pose[5] - before_pose[5]),
        math.cos(after_pose[5] - before_pose[5]),
    ))
    print(
        "[RIGHT-TILTED OPEN COMPLETE] "
        f"actual xArm dY={actual_delta_y:+.3f} mm, "
        f"dZ={actual_delta_z:+.3f} mm, "
        f"dYaw={actual_delta_yaw_deg:+.3f} deg"
    )
    if (abs(actual_delta_y - travel_y) > RIGHT_TILTED_SIMULTANEOUS_Y_POSITION_TOLERANCE_MM
            or abs(actual_delta_z) > RIGHT_TILTED_SIMULTANEOUS_Z_POSITION_TOLERANCE_MM
            or abs(actual_delta_yaw_deg - yaw_offset) > RIGHT_TILTED_SIMULTANEOUS_YAW_TOLERANCE_DEG):
        raise RuntimeError(
            "RIGHT tilted Y/Yaw target not reached; abort before insertion: "
            f"commanded dY={travel_y:+.3f} mm, dYaw={yaw_offset:+.3f} deg, "
            f"measured dY={actual_delta_y:+.3f} mm, dZ={actual_delta_z:+.3f} mm, "
            f"dYaw={actual_delta_yaw_deg:+.3f} deg"
        )


def execute_right_tilted_support_plan(
    *,
    hand_worker: Any,
    xarm7: Any,
    plan: RightTiltedSupportPlan,
    step_deg: float = RIGHT_TILTED_SYNC_STEP_DEG,
    sp_lin_target_steps: float | None = None,
    records: list[dict[str, Any]] | None = None,
    log_path: Path | None = None,
) -> float:
    """Synchronize SP_ROT with legacy TCP-Y or calibrated tip motion."""
    virtual_tcp_enabled = bool(ENABLE_SPACER_VIRTUAL_TCP)
    if virtual_tcp_enabled:
        validate_spacer_virtual_tcp_runtime_configuration()
        if sp_lin_target_steps is None or records is None or log_path is None:
            raise RuntimeError(
                "virtual TCP RIGHT sync requires SP_LIN target and motion log"
            )
        if SPACER_VIRTUAL_TCP_GEOMETRY is None:
            raise RuntimeError("spacer virtual TCP geometry disappeared at runtime")

    steps = plan_right_tilted_rotation_steps(
        plan.spacer_angle_deg,
        plan.arm_right_total_mm,
        step_deg=step_deg,
    )

    initial_virtual_state: SpacerVirtualTcpMechanismState | None = None
    initial_virtual_tip: np.ndarray | None = None
    if virtual_tcp_enabled:
        initial_virtual_state = _read_spacer_virtual_tcp_state()
        initial_virtual_pose = np.asarray(
            xarm7.get_tcp_pose(is_radian=True),
            dtype=np.float64,
        ).reshape(6)
        initial_virtual_tip = calculate_spacer_tip_position_robot(
            initial_virtual_pose,
            _require_spacer_virtual_tcp_geometry(),
            sp_lin_position_steps=initial_virtual_state.sp_lin_position_steps,
            sp_rot_angle_deg=initial_virtual_state.sp_rot_angle_deg,
        )

    print("\n[RIGHT-TILTED CONTROL PLAN]")
    print("classification = triangular_left_tilted")
    print("real-world tilted side = RIGHT")
    print(f"boundary tilt alpha = {plan.boundary_tilt_deg:.2f} deg")
    print(f"contact height h = {plan.contact_height_mm:.2f} mm")
    movement_subject = "tip" if virtual_tcp_enabled else "arm"
    print(
        f"{movement_subject} total RIGHT movement = h*tan(abs(alpha)) = "
        f"{plan.arm_right_total_mm:.2f} mm"
    )
    print("arm direction = real-world RIGHT / robot -Y")
    print(f"G_max = {plan.max_space_width_mm:.2f} mm")
    print(f"spacer effective length L = {plan.spacer_effective_length_mm:.2f} mm")
    print("SP_ROT angle method = fixed +90 deg")
    print(f"SP_ROT target = {plan.spacer_angle_deg:+.2f} deg")
    print("SP_ROT direction = same as LEFT tilted / positive")
    print(f"high reach offset = +{plan.high_reach_offset_mm:.2f} mm")
    print(f"synchronized step = {float(step_deg):.2f} deg")
    print("\n[STORAGE SPACER TIMING]")
    print(
        "rotation start delay = "
        f"{SPACER_ROTATION_START_DELAY_SEC:.3f} sec"
    )
    print(
        "rotation reset advance = "
        f"{SPACER_ROTATION_RESET_ADVANCE_SEC:.3f} sec"
    )

    for index, rotation_step in enumerate(steps, start=1):
        print("\n[RIGHT-TILTED ROTATE + ARM STEP]")
        print(f"step = {index}/{len(steps)}")
        print(f"angle prev = {rotation_step.angle_prev_deg:.2f} deg")
        print(f"angle next = {rotation_step.angle_next_deg:.2f} deg")
        print(f"final angle = {plan.spacer_angle_deg:.2f} deg")
        print(f"rotation progress = {rotation_step.rotation_progress:.6f}")
        print(
            "SP_ROT command = "
            f"{rotation_step.spacer_command_angle_deg:+.2f} deg"
        )
        print(
            f"{movement_subject} total RIGHT target = "
            f"{plan.arm_right_total_mm:.2f} mm"
        )
        print(
            "arm cumulative RIGHT = "
            f"{rotation_step.arm_cumulative_right_mm:.2f} mm"
        )
        print(
            "robot cumulative dY = "
            f"{rotation_step.robot_cumulative_y_mm:+.2f} mm"
        )
        print(f"robot step dY = {rotation_step.robot_delta_y_mm:+.2f} mm")

        rotation_errors: list[Exception] = []

        virtual_solution = None
        virtual_before_pose: np.ndarray | None = None
        virtual_before_state: SpacerVirtualTcpMechanismState | None = None
        virtual_before_tip: np.ndarray | None = None
        virtual_desired_tip: np.ndarray | None = None
        if virtual_tcp_enabled:
            if initial_virtual_tip is None:
                raise RuntimeError("virtual TCP initial tip is missing")
            virtual_before_state = _read_spacer_virtual_tcp_state()
            virtual_before_pose = np.asarray(
                xarm7.get_tcp_pose(is_radian=True),
                dtype=np.float64,
            ).reshape(6)
            virtual_before_tip = calculate_spacer_tip_position_robot(
                virtual_before_pose,
                _require_spacer_virtual_tcp_geometry(),
                sp_lin_position_steps=(
                    virtual_before_state.sp_lin_position_steps
                ),
                sp_rot_angle_deg=virtual_before_state.sp_rot_angle_deg,
            )
            virtual_desired_tip = calculate_right_tilted_desired_tip_position(
                initial_virtual_tip,
                contact_height_mm=plan.contact_height_mm,
                boundary_tilt_deg=plan.boundary_tilt_deg,
                rotation_progress=rotation_step.rotation_progress,
            )
            virtual_solution = calculate_virtual_tcp_solution(
                virtual_before_pose,
                virtual_desired_tip,
                _require_spacer_virtual_tcp_geometry(),
                sp_lin_position_steps=(
                    virtual_before_state.sp_lin_position_steps
                ),
                sp_rot_angle_deg=rotation_step.angle_next_deg,
            )

        rotation_delay_sec = (
            SPACER_ROTATION_START_DELAY_SEC if index == 1 else 0.0
        )

        def rotate_at_this_step() -> None:
            try:
                if rotation_delay_sec > 0.0:
                    time.sleep(rotation_delay_sec)
                hand_worker.rotate_spacer(
                    float(rotation_step.spacer_command_angle_deg)
                )
            except Exception as exc:
                rotation_errors.append(exc)

        rotation_thread = threading.Thread(
            target=rotate_at_this_step,
            name=f"right_tilted_sp_rot_step_{index}",
            daemon=False,
        )
        xarm_exception: Exception | None = None
        xarm_ret: Any = None
        rotation_thread.start()
        try:
            if virtual_tcp_enabled:
                if virtual_solution is None:
                    raise RuntimeError("virtual TCP solution is missing")
                compensation = list(
                    virtual_solution.tcp_compensation_robot_mm
                )
                xarm_ret = xarm7.moveL_relative(
                    compensation + [0.0, 0.0, 0.0],
                    asynchronous=False,
                )
            else:
                xarm_ret = xarm7.moveL_y_offset(
                    y_offset=rotation_step.robot_delta_y_mm,
                    asynchronous=False,
                )
        except Exception as exc:
            xarm_exception = exc
        finally:
            rotation_thread.join()

        if rotation_errors and xarm_exception is not None:
            raise RuntimeError(
                "SP_ROT and xArm both failed during RIGHT tilted sync: "
                f"SP_ROT={rotation_errors[0]!r}, xArm={xarm_exception!r}"
            ) from rotation_errors[0]
        if rotation_errors:
            raise RuntimeError(
                "SP_ROT failed during RIGHT tilted sync: "
                f"{rotation_errors[0]}"
            ) from rotation_errors[0]
        if xarm_exception is not None:
            raise RuntimeError(
                "xArm MoveL failed during RIGHT tilted sync"
            ) from xarm_exception
        if xarm_ret != 0:
            raise RuntimeError(
                "xArm MoveL returned an error during RIGHT tilted sync: "
                f"ret={xarm_ret}"
            )
        xarm_state = xarm7.get_state()
        xarm_error = int(xarm7.arm.error_code)
        xarm_warn = int(xarm7.arm.warn_code)
        print(
            "[RIGHT-TILTED ROTATE + ARM STEP] completed: "
            f"state={xarm_state}, error={xarm_error}, warn={xarm_warn}"
        )
        if xarm_state in (4, 5) or xarm_error != 0:
            raise RuntimeError(
                "xArm abnormal after RIGHT tilted synchronized step: "
                f"state={xarm_state}, error={xarm_error}, warn={xarm_warn}"
            )

        if virtual_tcp_enabled:
            if (
                records is None
                or log_path is None
                or sp_lin_target_steps is None
                or initial_virtual_tip is None
                or virtual_before_pose is None
                or virtual_before_state is None
                or virtual_before_tip is None
                or virtual_desired_tip is None
                or virtual_solution is None
            ):
                raise RuntimeError("virtual TCP logging state is incomplete")
            after_state = _read_spacer_virtual_tcp_state()
            after_pose = np.asarray(
                xarm7.get_tcp_pose(is_radian=True),
                dtype=np.float64,
            ).reshape(6)
            after_tip = calculate_spacer_tip_position_robot(
                after_pose,
                _require_spacer_virtual_tcp_geometry(),
                sp_lin_position_steps=after_state.sp_lin_position_steps,
                sp_rot_angle_deg=after_state.sp_rot_angle_deg,
            )
            tip_error = after_tip - virtual_desired_tip
            desired_cumulative_right = float(
                initial_virtual_tip[1] - virtual_desired_tip[1]
            )
            actual_cumulative_right = float(
                initial_virtual_tip[1] - after_tip[1]
            )
            records.append(
                {
                    "type": "spacer_virtual_tcp",
                    "phase": "right_tilted_synchronized_step",
                    "classification": "triangular_left_tilted",
                    "calibration_id": (
                        _require_spacer_virtual_tcp_geometry().calibration_id
                    ),
                    "step": index,
                    "step_count": len(steps),
                    "rotation_progress": rotation_step.rotation_progress,
                    "sp_lin_position_steps": (
                        after_state.sp_lin_position_steps
                    ),
                    "sp_lin_target_steps": float(sp_lin_target_steps),
                    "sp_lin_displacement_mm": (
                        calculate_sp_lin_displacement_mm(
                            _require_spacer_virtual_tcp_geometry(),
                            after_state.sp_lin_position_steps,
                        )
                    ),
                    "sp_rot_command_angle_deg": (
                        rotation_step.spacer_command_angle_deg
                    ),
                    "sp_rot_measured_angle_before_deg": (
                        virtual_before_state.sp_rot_angle_deg
                    ),
                    "sp_rot_measured_angle_deg": (
                        after_state.sp_rot_angle_deg
                    ),
                    "tcp_before_mm_rad": virtual_before_pose.tolist(),
                    "tcp_after_mm_rad": after_pose.tolist(),
                    "tip_offset_tcp_mm": list(
                        virtual_solution.tip_offset_tcp_mm
                    ),
                    "tip_before_robot_mm": virtual_before_tip.tolist(),
                    "predicted_tip_without_compensation_robot_mm": list(
                        virtual_solution.current_tip_robot_mm
                    ),
                    "desired_tip_robot_mm": virtual_desired_tip.tolist(),
                    "required_tcp_robot_mm": list(
                        virtual_solution.required_tcp_robot_mm
                    ),
                    "tcp_compensation_dxyz_mm": list(
                        virtual_solution.tcp_compensation_robot_mm
                    ),
                    "tip_after_robot_mm": after_tip.tolist(),
                    "tip_error_after_mm": tip_error.tolist(),
                    "tip_error_after_norm_mm": float(
                        np.linalg.norm(tip_error)
                    ),
                    "desired_cumulative_tip_right_mm": (
                        desired_cumulative_right
                    ),
                    "actual_cumulative_tip_right_mm": (
                        actual_cumulative_right
                    ),
                }
            )
            _save_storage_0911_motion_log(log_path, records)
            print(
                "[SPACER VIRTUAL TCP] RIGHT step "
                f"{index}/{len(steps)} desired/actual cumulative RIGHT = "
                f"{desired_cumulative_right:.3f}/"
                f"{actual_cumulative_right:.3f} mm, "
                f"tip error norm={np.linalg.norm(tip_error):.3f} mm"
            )

    robot_delta_y = -plan.arm_right_total_mm
    print("[RIGHT-TILTED CONTROL DONE]")
    print(f"final SP_ROT = {plan.spacer_angle_deg:+.2f} deg")
    if virtual_tcp_enabled:
        print(
            "tip target method = -Y h*tan(abs(alpha)); xArm dXYZ includes "
            "SP_ROT virtual-TCP compensation"
        )
    else:
        print(f"final robot Y shift = {robot_delta_y:+.2f} mm")
        print("xArm total method = -h*tan(abs(alpha)); L*sin(theta) not used")
    print("continue insertion without Y recenter = True")
    return robot_delta_y




def _save_storage_0911_motion_log(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Atomically persist the plan and every measured TCP checkpoint."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f".{path.name}.tmp"
    )

    temporary_path.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(path)


def _record_storage_0911_tcp(
    *,
    xarm7: Any,
    label: str,
    phase: str,
    records: list[dict[str, Any]],
    log_path: Path,
) -> np.ndarray:

    pose = np.asarray(
        xarm7.get_tcp_pose(
            is_radian=True
        ),
        dtype=np.float64,
    ).reshape(6)

    if not np.isfinite(pose).all():
        raise RuntimeError(
            f"non-finite TCP pose at "
            f"{label}/{phase}"
        )

    records.append(
        {
            "type": "tcp_checkpoint",
            "label": label,
            "phase": phase,
            "tcp_pose_mm_rad": pose.tolist(),
        }
    )

    _save_storage_0911_motion_log(
        log_path,
        records,
    )

    print(
        f"[{label}] {phase} TCP XYZ = "
        f"[{pose[0]:.3f}, "
        f"{pose[1]:.3f}, "
        f"{pose[2]:.3f}] mm"
    )

    return pose

def move_storage_0911_absolute_approach(
    *,
    xarm7: Any,
    reach_target_xyz: Any,
    roll_offset_rad: float,
    apply_roll: bool,
    records: list[dict[str, Any]],
    log_path: Path,
    velocity: float,
    acceleration: float,
    rpy_offsets_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
    backoff_mm: float = APPROACH_BACKOFF_MM,
) -> np.ndarray:
    """書架右側専用：X/Y/Roll補正付きアプローチ。"""

    label = "0911 APPROACH"
    side = "right"

    before = _record_storage_0911_tcp(
        xarm7=xarm7,
        label=label,
        phase="before",
        records=records,
        log_path=log_path,
    )

    original_target_xyz = np.asarray(
        reach_target_xyz,
        dtype=np.float64,
    ).reshape(3)

    if not np.isfinite(original_target_xyz).all():
        raise RuntimeError("non-finite reach target")

    if not math.isfinite(roll_offset_rad):
        raise RuntimeError("invalid roll offset")

    # Set backoff_mm=0 to reach the recognition target in one MoveL.
    if not math.isfinite(backoff_mm) or backoff_mm < 0.0:
        raise RuntimeError(f"invalid approach backoff: {backoff_mm}")
    approach_target_xyz = original_target_xyz.copy()
    approach_target_xyz[0] -= float(backoff_mm)

    # 補正量を初期化
    x_offset_mm = 0.0
    y_offset_mm = 0.0
    calib_rad = 0.0

    if apply_roll and abs(roll_offset_rad) > 1e-8:
        d_roll_rad = float(roll_offset_rad)

        # ===== 書架右側の既存補正式 =====
        roll_micro_calib_mm = (
            1.5 * max(
                0.7,
                abs(d_roll_rad) / np.deg2rad(60)
            )
        )

        roll_calib_right = (
            -2.6618 * d_roll_rad**2
            - 1.3218 * d_roll_rad
            + 2.5725
            - roll_micro_calib_mm
        )

        target_position_calib_right = 0.0

        calib_rad_roll = (
            0.0016 * d_roll_rad**2
            + 0.0036 * d_roll_rad
            - 0.0294
        )

        calib_rad_y = 0.0

        u_offset_mm = (
            roll_calib_right
            + target_position_calib_right
            - 2.0
        )

        calib_rad = calib_rad_roll + calib_rad_y

        y_offset_mm = (
            u_offset_mm / np.cos(d_roll_rad)
        )

    # ===== 目標TCP姿勢 =====
    target_pose = before.copy()

    target_pose[0] = approach_target_xyz[0] + x_offset_mm
    target_pose[1] = approach_target_xyz[1] + y_offset_mm
    target_pose[2] = approach_target_xyz[2]

    if apply_roll:
        target_pose[3] = (
            before[3] + roll_offset_rad + calib_rad
        )
    extra_rpy = np.asarray(rpy_offsets_rad, dtype=np.float64).reshape(3)
    if not np.isfinite(extra_rpy).all():
        raise RuntimeError("non-finite approach RPY offsets")
    target_pose[3:6] += extra_rpy

    if not np.isfinite(target_pose).all():
        raise RuntimeError("non-finite calibrated target pose")

    print(f"\n[{label}]")
    print(f"side = {side}")
    print(f"apply_roll = {apply_roll}")

    print(
        f"recognition Roll offset = "
        f"{np.degrees(roll_offset_rad):+.3f} deg"
    )

    print(f"X correction = {x_offset_mm:+.3f} mm")
    print(f"Y correction = {y_offset_mm:+.3f} mm")

    print(
        f"Roll calibration = "
        f"{np.degrees(calib_rad):+.3f} deg"
    )

    print(
        f"approach backoff = "
        f"{backoff_mm:.3f} mm"
    )

    print(
        f"original target XYZ = "
        f"{original_target_xyz.tolist()}"
    )

    print(
        f"target TCP pose = "
        f"{target_pose.tolist()}"
    )

    records.append({
        "type": "motion_command",
        "label": label,
        "side": side,
        "apply_roll": bool(apply_roll),
        "original_reach_target_xyz_mm":
            original_target_xyz.tolist(),
        "approach_backoff_mm":
            float(backoff_mm),
        "roll_offset_rad":
            float(roll_offset_rad),
        "roll_calibration_rad":
            float(calib_rad),
        "x_offset_mm":
            float(x_offset_mm),
        "y_offset_mm":
            float(y_offset_mm),
        "rpy_offsets_rad": extra_rpy.tolist(),
        "target_tcp_pose_mm_rad":
            target_pose.tolist(),
        "velocity_mm_s":
            float(velocity),
        "acceleration_mm_s2":
            float(acceleration),
    })

    _save_storage_0911_motion_log(
        log_path,
        records,
    )

    # ===== xArm移動 =====
    public_move_l = getattr(xarm7, "moveL", None)

    if callable(public_move_l):
        ret = public_move_l(
            target_pose.tolist(),
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=False,
        )
    else:
        ret = xarm7._moveL(
            target_pose.tolist(),
            velocity,
            acceleration,
            False,
        )

    print(f"[{label}] MoveL ret = {ret}")

    if ret != 0:
        raise RuntimeError(
            f"0911 absolute approach failed: ret={ret}"
        )

    after = _record_storage_0911_tcp(
        xarm7=xarm7,
        label=label,
        phase="after",
        records=records,
        log_path=log_path,
    )

    position_error = target_pose[:3] - after[:3]
    error_norm = float(np.linalg.norm(position_error))

    print(
        f"[{label}] position error = "
        f"{position_error.tolist()} mm"
    )

    print(
        f"[{label}] position error norm = "
        f"{error_norm:.3f} mm"
    )

    print(
        f"[{label}] actual Roll = "
        f"{np.degrees(after[3]):+.3f} deg"
    )

    return after

def move_storage_post_release(
    *,
    xarm7: Any,
    retreat_x_mm: float,
    lower_z_mm: float = 0.0,
    velocity: float,
    acceleration: float,
) -> int:
    """書籍開放後のX方向退避・Z方向下降。"""

    curr_pose = np.asarray(
        xarm7.get_tcp_pose(is_radian=True),
        dtype=np.float64,
    ).reshape(6)

    target_pose = curr_pose.copy()

    target_pose[0] -= retreat_x_mm
    target_pose[2] -= lower_z_mm

    print("[POST STORAGE] before =", curr_pose.tolist())
    print("[POST STORAGE] target =", target_pose.tolist())
    print(f"[POST STORAGE] retreat X = {-retreat_x_mm:+.1f} mm")
    print(f"[POST STORAGE] lower Z = {-lower_z_mm:+.1f} mm")

    ret = xarm7._moveL(
        target_pose.tolist(),
        velocity,
        acceleration,
        False,
    )

    if ret != 0:
        raise RuntimeError(f"post-storage motion failed: ret={ret}")

    return ret


def move_storage_0911_relative_x(
    *,
    xarm7: Any,
    distance_mm: float,
    label: str,
    records: list[dict[str, Any]],
    log_path: Path,
    velocity: float,
    acceleration: float,
) -> np.ndarray:

    before = _record_storage_0911_tcp(
        xarm7=xarm7,
        label=label,
        phase="before",
        records=records,
        log_path=log_path,
    )

    delta = [
        float(distance_mm),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    print(
        f"[{label}] command dX = "
        f"{float(distance_mm):+.1f} mm"
    )

    print(
        f"[{label}] relative MoveL = "
        f"{delta}"
    )

    records.append(
        {
            "type": "motion_command",
            "label": label,
            "delta_pose_mm_rad": delta,
            "velocity_mm_s": float(
                velocity
            ),
            "acceleration_mm_s2": float(
                acceleration
            ),
        }
    )

    _save_storage_0911_motion_log(
        log_path,
        records,
    )

    ret = xarm7.moveL_relative(
        delta,
        velocity=velocity,
        acceleration=acceleration,
        asynchronous=False,
    )

    after = _record_storage_0911_tcp(
        xarm7=xarm7,
        label=label,
        phase="after",
        records=records,
        log_path=log_path,
    )

    print(
        f"[{label}] MoveL ret = {ret}"
    )

    print(
        f"[{label}] actual X movement = "
        f"{after[0] - before[0]:+.3f} mm"
    )

    if ret != 0:
        raise RuntimeError(
            f"{label} failed: ret={ret}"
        )

    return after

def move_storage_0911_tilted_lower(
    *,
    xarm7: Any,
    classification: str,
    normal_target_z_mm: float,
    high_reach_target_z_mm: float,
    records: list[dict[str, Any]],
    log_path: Path,
    velocity: float,
    acceleration: float,
) -> np.ndarray:
    """55mm挿入後、設定した量だけZ方向へ下降する。"""

    label = "0911 TILTED LOWER"

    if classification not in {
        "triangular_left_tilted",
        "triangular_right_tilted",
    }:
        raise RuntimeError(
            f"tilted lowering requested for {classification}"
        )

    lowering_amount = float(STORAGE_TILTED_LOWER_MM)

    if not math.isfinite(lowering_amount) or lowering_amount < 0.0:
        raise RuntimeError(
            f"invalid lowering amount: {lowering_amount}"
        )

    normal_target_z = float(normal_target_z_mm)
    high_reach_target_z = float(high_reach_target_z_mm)

    before = _record_storage_0911_tcp(
        xarm7=xarm7,
        label=label,
        phase="before",
        records=records,
        log_path=log_path,
    )

    # 現在位置から指定量だけ下げる
    final_target_z = float(before[2]) - lowering_amount

    print(f"\n[{label}]")
    print(f"[{label}] classification = {classification}")
    print(f"[{label}] normal target Z = {normal_target_z:.3f} mm")
    print(f"[{label}] high reach target Z = {high_reach_target_z:.3f} mm")
    print(f"[{label}] configured lowering = {lowering_amount:.3f} mm")
    print(f"[{label}] current Z = {before[2]:.3f} mm")
    print(f"[{label}] final target Z = {final_target_z:.3f} mm")

    records.append({
        "type": "motion_command",
        "label": label,
        "classification": classification,
        "normal_target_z_mm": normal_target_z,
        "high_reach_target_z_mm": high_reach_target_z,
        "lowering_amount_mm": lowering_amount,
        "final_target_z_mm": final_target_z,
        "delta_z_mm": -lowering_amount,
        "velocity_mm_s": float(velocity),
        "acceleration_mm_s2": float(acceleration),
        "skipped": lowering_amount == 0.0,
    })

    _save_storage_0911_motion_log(log_path, records)

    # 0mmなら移動命令を出さない
    if lowering_amount == 0.0:
        print(f"[{label}] skipped: lowering = 0 mm")
        return before

    # 設定量だけZ方向へ下降
    ret = xarm7.moveL_z_offset(
        z_offset=-lowering_amount,
        velocity=velocity,
        acceleration=acceleration,
        asynchronous=False,
    )

    print(f"[{label}] MoveL ret = {ret}")

    if ret != 0:
        raise RuntimeError(
            f"{label} failed: ret={ret}"
        )

    after = _record_storage_0911_tcp(
        xarm7=xarm7,
        label=label,
        phase="after",
        records=records,
        log_path=log_path,
    )

    print(
        f"[{label}] actual Z movement = "
        f"{after[2] - before[2]:+.3f} mm"
    )

    return after


def _require_spacer_virtual_tcp_geometry() -> SpacerVirtualTcpGeometry:
    """Return measured spacer geometry or fail closed."""
    geometry = SPACER_VIRTUAL_TCP_GEOMETRY

    if geometry is None:
        raise RuntimeError(
            "spacer virtual TCP geometry is unavailable: "
            f"{SPACER_VIRTUAL_TCP_UNAVAILABLE_REASON}"
        )

    return geometry


def validate_spacer_virtual_tcp_runtime_configuration() -> None:
    """Fail closed before motion when virtual-TCP inputs are incomplete."""
    if not ENABLE_SPACER_VIRTUAL_TCP:
        return
    if SPACER_VIRTUAL_TCP_GEOMETRY is None:
        raise RuntimeError(
            "spacer virtual TCP is enabled without measured geometry: "
            f"{SPACER_VIRTUAL_TCP_UNAVAILABLE_REASON}"
        )
    if not callable(SPACER_VIRTUAL_TCP_STATE_READER):
        raise RuntimeError(
            "spacer virtual TCP is enabled without a measured SP_LIN/SP_ROT "
            "state reader; commanded targets must not be used as actual state"
        )


def _read_spacer_virtual_tcp_state() -> SpacerVirtualTcpMechanismState:
    validate_spacer_virtual_tcp_runtime_configuration()
    state = SPACER_VIRTUAL_TCP_STATE_READER()
    if not isinstance(state, SpacerVirtualTcpMechanismState):
        raise RuntimeError(
            "SPACER_VIRTUAL_TCP_STATE_READER must return "
            "SpacerVirtualTcpMechanismState"
        )
    return state


def _spacer_virtual_tcp_status_record(
    classification: str,
) -> dict[str, Any]:
    """Describe why reach/control is active or safely left unchanged."""
    enabled = bool(ENABLE_SPACER_VIRTUAL_TCP)
    calibrated = SPACER_VIRTUAL_TCP_GEOMETRY is not None
    state_reader_available = callable(SPACER_VIRTUAL_TCP_STATE_READER)
    return {
        "type": "spacer_virtual_tcp",
        "phase": "configuration",
        "classification": classification,
        "enabled": enabled,
        "calibrated": calibrated,
        "measured_state_reader_available": state_reader_available,
        "initial_reach_virtual_tcp_applied": False,
        "initial_reach_reason": (
            "the recognition-derived reach target is not proven to be a "
            "spacer-tip contact target"
        ),
        "reason": (
            None
            if enabled and calibrated and state_reader_available
            else SPACER_VIRTUAL_TCP_UNAVAILABLE_REASON
        ),
    }


def _record_left_tilted_virtual_tcp_rotation(
    *,
    xarm7: Any,
    initial_tcp_pose: np.ndarray,
    initial_state: SpacerVirtualTcpMechanismState,
    initial_tip_robot_mm: np.ndarray,
    sp_lin_target_steps: float,
    records: list[dict[str, Any]],
    log_path: Path,
) -> None:
    """Log LEFT-tilted tip drift without introducing a lateral trajectory."""
    if SPACER_VIRTUAL_TCP_GEOMETRY is None:
        raise RuntimeError("spacer virtual TCP geometry disappeared at runtime")
    after_state = _read_spacer_virtual_tcp_state()
    after_tcp_pose = np.asarray(
        xarm7.get_tcp_pose(is_radian=True),
        dtype=np.float64,
    ).reshape(6)
    after_tip = calculate_spacer_tip_position_robot(
        after_tcp_pose,
        _require_spacer_virtual_tcp_geometry(),
        sp_lin_position_steps=after_state.sp_lin_position_steps,
        sp_rot_angle_deg=after_state.sp_rot_angle_deg,
    )
    hold_solution = calculate_virtual_tcp_solution(
        after_tcp_pose,
        initial_tip_robot_mm,
        _require_spacer_virtual_tcp_geometry(),
        sp_lin_position_steps=after_state.sp_lin_position_steps,
        sp_rot_angle_deg=after_state.sp_rot_angle_deg,
    )
    tip_error = after_tip - initial_tip_robot_mm
    records.append(
        {
            "type": "spacer_virtual_tcp",
            "phase": "left_tilted_rotation_diagnostic",
            "classification": "triangular_right_tilted",
            "calibration_id": _require_spacer_virtual_tcp_geometry().calibration_id,
            "lateral_trajectory_applied": False,
            "lateral_trajectory_reason": (
                "a unique LEFT-tilted desired lateral tip path is not defined"
            ),
            "sp_lin_position_steps": after_state.sp_lin_position_steps,
            "sp_lin_target_steps": float(sp_lin_target_steps),
            "sp_lin_displacement_mm": calculate_sp_lin_displacement_mm(
                _require_spacer_virtual_tcp_geometry(),
                after_state.sp_lin_position_steps,
            ),
            "sp_rot_angle_before_deg": initial_state.sp_rot_angle_deg,
            "sp_rot_angle_after_deg": after_state.sp_rot_angle_deg,
            "tcp_before_mm_rad": initial_tcp_pose.tolist(),
            "tcp_after_mm_rad": after_tcp_pose.tolist(),
            "tip_before_robot_mm": initial_tip_robot_mm.tolist(),
            "tip_offset_tcp_mm": list(hold_solution.tip_offset_tcp_mm),
            "tip_after_robot_mm": after_tip.tolist(),
            "diagnostic_hold_tip_robot_mm": initial_tip_robot_mm.tolist(),
            "diagnostic_required_tcp_robot_mm": list(
                hold_solution.required_tcp_robot_mm
            ),
            "diagnostic_compensation_dxyz_mm": list(
                hold_solution.tcp_compensation_robot_mm
            ),
            "tip_error_after_mm": tip_error.tolist(),
            "tip_error_after_norm_mm": float(np.linalg.norm(tip_error)),
        }
    )
    _save_storage_0911_motion_log(log_path, records)
    print(
        "[SPACER VIRTUAL TCP] LEFT-tilted diagnostic only; "
        f"tip error={tip_error.tolist()} mm"
    )
