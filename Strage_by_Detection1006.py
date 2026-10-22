from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import math
import time
import numpy as np
import sys
import traceback
import threading
from contextlib import redirect_stdout, redirect_stderr
from Dynamixel_win_pro_hand_book.spacer_virtual_tcp import (
    calculate_spacer_tip_position_robot,
)

from storage.motion import (
    validate_spacer_virtual_tcp_runtime_configuration,
    _require_spacer_virtual_tcp_geometry,
    _read_spacer_virtual_tcp_state,
    _spacer_virtual_tcp_status_record,
    _record_left_tilted_virtual_tcp_rotation,
    _save_storage_0911_motion_log,
    move_storage_0911_absolute_approach,
    move_storage_0911_relative_x,
    move_storage_0911_tilted_lower,
    move_storage_post_release,
)

from storage.planner import (
    validate_storage_0911_depth_constants,
    build_storage_0911_depth_plan,
    apply_tilted_reach_y_correction,
)

from xarm7.control.xarm_monitor import assert_xarm_normal
from xarm7.control.robot_base_coordinate import cam_mm_to_robot_mm
from xarm7.control.xarm7 import (
    XArm7,
    INSERT_BOOK_FULL_DX,
    INSERT_BOOK_TIP_DX,
    TCP_ACC_2,
    TCP_VEL_2,
)
from storage.motion import (
    reset_spacer_rotation_before_linear_retraction,
    execute_right_tilted_support_plan,
    open_right_tilted_with_y,
)
from storage.planner import (
    finite_float,
    should_use_right_tilted_sp_lin_partial_expansion,
    should_use_right_tilted_support_sync,
    calculate_storage_final_tip_forward_mm,
    plan_triangular_forward_split,
    plan_storage_spacer,
    build_tilted_book_high_reach_target_robot,
    apply_left_tilted_reach_correction,
    build_right_and_forward_move_delta,
    classify_boundary_tilts,
    resolve_ambiguous_storage_space,
    classify_storage_space,
    evaluate_storage_capture_candidate,
    select_best_storage_capture,
)

from storage.models import (
    TriangularSpacerGeometry,
    TriangularForwardPlan,
    AmbiguousStorageResolution,
    StorageSpacerPlan,
    StorageSpaceClassification,
    StorageCaptureCandidate,
)

from storage.analysis import (
    estimate_contact_height_opening_mm,
    calculate_bottleneck_percentile_mm,
    estimate_selected_space_row_widths_mm,
    evaluate_physical_bottom_narrowing_rule,
    evaluate_selected_space_vertical_width_check,
    estimate_rectangular_bottleneck_mm,
    build_real_world_right_side_target_camera,
    build_real_world_left_boundary_target_camera,
    build_physical_shelf_frame,
    build_left_tilted_book_right_edge_target_camera,
    save_left_tilted_book_target_geometry_overlay,
)

from storage.config import (
    # 書籍・スペーサ基本設定
    TEST_BOOK_WIDTH_MM,
    STORAGE_CLEARANCE_MM,
    SPACER_EFFECTIVE_LENGTH_MM,
    STORAGE_SPACER_FINAL_TIP_ADDITIONAL_MM,
    SPACER_CONTACT_HEIGHT_MM,
    RIGHT_TILTED_SPACER_CONTACT_HEIGHT_MM,
    # 収納スペース認識・分類
    RECTANGULAR_BOTTLENECK_PERCENTILE,
    CONTACT_OPENING_MIN_VALID_ROWS,
    TWO_VIEW_BOTTLENECK_TIE_TOLERANCE_MM,
    VERTICAL_ANGLE_TOLERANCE_DEG,
    BOUNDARY_TILT_DIFFERENCE_THRESHOLD_DEG,
    AMBIGUOUS_TILT_EQUAL_EPSILON_DEG,
    MIN_SUPPORTED_SPACE_WIDTH_PX,
    # アーム位置・姿勢補正
    LEFT_BOOK_REACH_Z_OFFSET_MM,
    UPRIGHT_XARM_D_ROLL_RAD,
    LEFT_TILTED_REACH_RIGHT_OFFSET_MM,
    LEFT_TILTED_REACH_BACKOFF_MM,
    STORAGE_CAPTURE_Z_OFFSET_MM,
    BOOK_PUSH_HEIGHT_MM,
    LEFT_BOOK_LATERAL_SHIFT_MM,
    # 撮影・認識
    STORAGE_CAPTURE_ROOT,
    AUTO_RUN,
    STORAGE_TARGET_3D_MODE,
    SECOND_CAPTURE_DY_MM,
    CAPTURE_SHIFT_VELOCITY_MM_S,
    CAPTURE_SHIFT_ACCELERATION_MM_S2,
    THIRD_CAPTURE_VELOCITY_MM_S,
    THIRD_CAPTURE_ACCELERATION_MM_S2,
    RECOGNITION_TO_REACH_WAIT_SEC,
    STORAGE_TERMINAL_LOG_FILENAME,
    # ハンド・スペーサ制御
    SPACER_ROTATION_START_DELAY_SEC,
    SPACER_ROTATION_RESET_ADVANCE_SEC,
    SPACER_ROTATION_RESET_BASE_WAIT_SEC,
    SPACER_RESET_OVERSHOOT_DEG,
    SPACER_ROTATION_RESET_WAIT_SEC,
    SPACER_EXPANSION_WAIT_SEC,
    GRIPPER_OPEN_TO_GRASP_WAIT_SEC,
    ONE_SHOT_SP_LIN_BACK_TIMEOUT_SEC,
    STORAGE_PRE_INSERT_YAW_DEG,
    LEFT_BOOK_TILTED_REACH_PITCH_DEG,
    RIGHT_TILTED_INSERT_Y_MM,
    STORAGE_INSERT_XYZ_TOLERANCE_MM,
    STORAGE_INSERT_YAW_TOLERANCE_DEG,
    LEFT_TILTED_PRE_INSERT_YAW_VELOCITY_MM_S,
    LEFT_TILTED_PRE_INSERT_YAW_ACCELERATION_MM_S2,
    MAX_REACH_ROLL_DEG,
    RIGHT_TILTED_RETURN_Y_MM,
    # リーチング・入庫深さ
    TCP_TO_BOOK_REAR_MM,
    APPROACH_BACKOFF_MM,
    INSERT_WITH_SPACER_MM,
    FINAL_BOOK_PUSH_MM,
    STORAGE_POST_RELEASE_RETREAT_X_MM,
    STORAGE_POST_RELEASE_LOWER_Z_MM,
    STORAGE_RELEASE_LOWER_Z_MM,
    # 入庫時の速度・加速度
    TILTED_SPACER_INSERT_VELOCITY_MM_S,
    TILTED_SPACER_INSERT_ACCELERATION_MM_S2,
    FINAL_BOOK_PUSH_VELOCITY_MM_S,
    FINAL_BOOK_PUSH_ACCELERATION_MM_S2,
    FINAL_PUSH_TO_RELEASE_WAIT_SEC,
    # 動作完了監視
    XARM_MOTION_DONE_TIMEOUT_SEC,
    XARM_MOTION_IDLE_CONFIRM_SEC,
    XARM_MOTION_POLL_INTERVAL_SEC,
    # 実行・安全設定
    ENABLE_XARM_LATERAL_CORRECTION,
    ENABLE_SPACER_VIRTUAL_TCP,
    SPACER_VIRTUAL_TCP_GEOMETRY,
    XARM_HOST,
    SHUTDOWN_SETTLE_WAIT_SEC,
)


class FrozenTcpArm:
    """Use the saved capture-time TCP for camera-to-robot conversion."""

    def __init__(self, arm: Any, tcp_pose: Any) -> None:
        self._arm = arm
        self._tcp_pose = np.asarray(
            tcp_pose,
            dtype=np.float64,
        ).reshape(6).copy()

    def get_tcp_pose(self, is_radian: bool = True) -> list[float]:
        pose = self._tcp_pose.copy()
        if not is_radian:
            pose[3:6] = np.degrees(pose[3:6])
        return pose.tolist()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._arm, name)


def log_storage_space_classification(
    decision: StorageSpaceClassification,
    capture_label: str,
) -> None:
    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    print("\n[STORAGE SPACE CLASSIFICATION]")
    print(f"capture = {capture_label}")
    print(f"selected space ID = {decision.selected_space_id}")
    print(
        "left boundary angle from vertical = "
        f"{fmt(decision.left_boundary_tilt_deg)} deg"
    )
    print(
        "right boundary angle from vertical = "
        f"{fmt(decision.right_boundary_tilt_deg)} deg"
    )
    print(
        "tilt difference = "
        f"{fmt(decision.tilt_difference_deg)} deg"
    )
    print(
        "vertical tolerance = "
        f"{VERTICAL_ANGLE_TOLERANCE_DEG:.2f} deg"
    )
    print(
        "difference threshold = "
        f"{decision.difference_threshold_deg:.2f} deg"
    )
    print(f"classification = {decision.classification}")
    print(f"classification source = {decision.classification_source}")
    print(f"tilted side = {decision.tilted_side}")
    print(
        "image guide angle = "
        f"{fmt(decision.image_guide_angle_deg)} deg"
    )
    print(
        "xArm roll command = "
        f"{np.degrees(decision.xarm_d_roll_rad):.2f} deg "
        "(guide angle is NOT applied; keep upright)"
    )
    print(
        "spacer command angle = "
        f"{fmt(decision.spacer_command_angle_deg)} deg"
    )
    print(
        "book-bottom-side opening width = "
        f"{fmt(decision.book_bottom_side_width_px)} px"
    )
    print(f"reasons = {list(decision.reasons)}")
    for reason in decision.reasons:
        if reason.startswith("guide_side_mismatch"):
            print(f"guide-side diagnostic mismatch = {reason}")

    if decision.classification_source == "ambiguous_physical_bottleneck_fallback":
        print("\n[STORAGE AMBIGUOUS FALLBACK]")
        print(
            "left tilt = "
            f"{fmt(decision.left_boundary_tilt_deg)} deg"
        )
        print(
            "right tilt = "
            f"{fmt(decision.right_boundary_tilt_deg)} deg"
        )
        print(
            "difference = "
            f"{fmt(decision.tilt_difference_deg)} deg"
        )
        print(
            "physical bottleneck = "
            f"{fmt(decision.ambiguous_bottleneck_opening_mm)} mm"
        )
        print(f"held book width = {TEST_BOOK_WIDTH_MM:.2f} mm")
        print("clearance used for fallback = False")
        print("direct insert condition = bottleneck > held book width")
        print(f"fallback result = {decision.classification}")
        if decision.classification == "direct_insert":
            print("target side = real-world right")
            print("spacer required = False")
        elif decision.classification.startswith("triangular_"):
            print(
                "selected by larger boundary tilt = "
                f"{decision.fallback_selected_by_larger_boundary}"
            )


def log_triangular_spacer_geometry(
    geometry: TriangularSpacerGeometry,
    *,
    selected_space_width_info: dict[str, Any] | None = None,
) -> None:
    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    print("\n[STORAGE TRIANGULAR SPACER GEOMETRY]")
    print(f"classification = {geometry.classification}")
    print(f"real-world tilted side = {geometry.real_world_tilted_side}")
    print(f"book width = {geometry.book_width_mm:.2f} mm")
    print(f"storage clearance = {geometry.clearance_mm:.2f} mm")
    print(
        "base required opening = "
        f"{geometry.base_required_opening_mm:.2f} mm"
    )
    print(
        "triangular extra opening calibration = "
        f"{geometry.triangular_extra_opening_mm:.2f} mm"
    )
    print(
        "calibrated required opening = "
        f"{geometry.calibrated_required_opening_mm:.2f} mm"
    )
    print(f"book height = {geometry.book_height_mm:.2f} mm")
    print(
        "spacer contact height = "
        f"{geometry.spacer_contact_height_mm:.2f} mm"
    )
    print(
        "boundary tilt from vertical = "
        f"{geometry.boundary_tilt_deg:.2f} deg"
    )
    print(
        "spacer axis offset from held-book right edge = "
        f"{geometry.spacer_axis_right_of_held_book_right_edge_mm:.2f} mm"
    )
    print(
        "geometry offset angle = "
        f"{geometry.geometry_offset_angle_deg:.2f} deg"
    )
    print(f"contact angle = {geometry.contact_angle_deg:.2f} deg")
    displacement_label = (
        "selected-space max physical width used for angle"
        if geometry.classification == "triangular_left_tilted"
        else "selected-space max physical width used for angle"
    )
    print(f"{displacement_label} = {geometry.push_displacement_mm:.2f} mm")
    print(
        "spacer effective length = "
        f"{geometry.spacer_effective_length_mm:.2f} mm"
    )
    print(
        "maximum remaining horizontal capacity = "
        f"{geometry.maximum_remaining_horizontal_capacity_mm:.2f} mm"
    )
    print(f"asin argument = {geometry.asin_argument:.6f}")
    print(
        "command angle magnitude = "
        f"{fmt(geometry.command_angle_magnitude_deg)} deg"
    )
    signed = geometry.signed_command_angle_deg
    signed_text = "n/a" if signed is None else f"{signed:+.2f}"
    print(f"signed spacer command = {signed_text} deg")
    print(
        "xArm lateral correction enabled = "
        f"{ENABLE_XARM_LATERAL_CORRECTION}"
    )
    print(f"feasible = {geometry.feasible}")
    print(f"reason = {geometry.reason}")
    if geometry.classification == "triangular_left_tilted":
        plan = geometry.right_tilted_support_plan
        if plan is None:
            raise RuntimeError("RIGHT tilted support plan is missing")
        print("\n[RIGHT-TILTED CONTROL PLAN]")
        print("classification = triangular_left_tilted")
        print("real-world tilted side = RIGHT")
        print(f"alpha from image-left boundary = {plan.boundary_tilt_deg:.2f} deg")
        print(f"contact height h = {plan.contact_height_mm:.2f} mm")
        print(
            "arm total RIGHT movement = 210*sin(abs(alpha)) = "
            f"{plan.arm_right_total_mm:.2f} mm"
        )
        print("arm direction = real-world RIGHT / robot -Y")
        print(f"G_max = {plan.max_space_width_mm:.2f} mm")
        print(f"spacer effective length = {plan.spacer_effective_length_mm:.2f} mm")
        print("SP_ROT angle method = fixed +90 deg")
        print(f"SP_ROT angle = {plan.spacer_angle_deg:+.2f} deg")
        print("SP_ROT direction = same as LEFT tilted / positive")
        print(f"high reach offset = +{plan.high_reach_offset_mm:.2f} mm")
        print("book width used for angle = False")
        print("clearance used for angle = False")
        print("boundary tilt used for SP_ROT angle = True")
    elif geometry.classification == "triangular_right_tilted":
        if selected_space_width_info is None:
            raise RuntimeError(
                "selected-space physical row widths are required for "
                f"{geometry.classification} logging"
            )
        print("\n[STORAGE SPACER GMAX ANGLE]")
        print(f"classification = {geometry.classification}")
        print(
            "real-world tilted side = "
            f"{geometry.real_world_tilted_side}"
        )
        print(
            "selected-space valid row count = "
            f"{selected_space_width_info['valid_row_count']}"
        )
        print(
            "selected-space min width = "
            f"{selected_space_width_info['row_width_min_mm']:.2f} mm"
        )
        print(
            "selected-space median width = "
            f"{selected_space_width_info['row_width_median_mm']:.2f} mm"
        )
        print(
            "G_max = "
            f"{selected_space_width_info['row_width_max_mm']:.2f} mm"
        )
        print(
            "spacer effective length = "
            f"{geometry.spacer_effective_length_mm:.2f} mm"
        )
        print("angle method = fixed +90 deg (G_max is diagnostic)")
        print("angle model = fixed +90 deg; G_max remains diagnostic")
        print(f"raw ratio = G_max / L = {geometry.asin_argument:.6f}")
        print(
            "90-degree saturation = "
            f"{geometry.push_displacement_mm >= geometry.spacer_effective_length_mm}"
        )
        print(
            "angle magnitude = "
            f"{fmt(geometry.command_angle_magnitude_deg)} deg"
        )
        signed = geometry.signed_command_angle_deg
        signed_text = "n/a" if signed is None else f"{signed:+.2f}"
        rotation_sign = (
            "negative" if signed is not None and signed < 0.0 else "positive"
        )
        print(f"rotation sign = {rotation_sign}")
        print(f"command angle = {signed_text} deg")
        print(f"boundary tilt = {geometry.boundary_tilt_deg:.2f} deg")
        print("boundary tilt used for final angle = False")
        print("book width used for final angle = False")
        print("clearance used for final angle = False")
        print("extra opening used for final angle = False")


def log_triangular_forward_plan(plan: TriangularForwardPlan) -> None:
    print("\n[STORAGE PRE-ROTATION FORWARD]")
    print(
        "pre-rotation forward = "
        f"{plan.pre_rotation_forward_mm:.2f} mm"
    )
    print(
        "original total forward = "
        f"{plan.original_total_forward_mm:.2f} mm"
    )
    print(
        "remaining forward after spacer rotation = "
        f"{plan.remaining_forward_mm:.2f} mm"
    )
    print("robot direction = base +X")


def log_storage_spacer_plan(
    plan: StorageSpacerPlan,
    *,
    rectangular_bottleneck_info: dict[str, Any] | None = None,
    selected_space_width_info: dict[str, Any] | None = None,
    diagnostic_current_opening_mm: float | None = None,
) -> None:
    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    print("\n[STORAGE SPACER PLAN]")
    print(f"test book width = {plan.book_width_mm:.2f} mm")
    print(f"clearance = {plan.clearance_mm:.2f} mm")
    print(f"required opening = {plan.required_opening_mm:.2f} mm")
    print(f"classification = {plan.classification}")
    print(f"bottleneck percentile = {fmt(plan.bottleneck_percentile)}")
    print(f"valid row count = {plan.valid_row_count}")
    print(f"bottleneck opening = {fmt(plan.bottleneck_opening_mm)} mm")
    if rectangular_bottleneck_info is not None:
        print(
            "row width stats [mm] = "
            f"min={rectangular_bottleneck_info['row_width_min_mm']:.2f}, "
            f"median={rectangular_bottleneck_info['row_width_median_mm']:.2f}, "
            f"max={rectangular_bottleneck_info['row_width_max_mm']:.2f}"
        )
    if plan.triangular_geometry is None:
        print(
            "required additional opening = "
            f"{fmt(plan.required_additional_opening_mm)} mm"
        )
    else:
        print(
            "selected-space max physical width used for angle = "
            f"{fmt(plan.required_additional_opening_mm)} mm"
        )
    print(f"spacer effective length = {SPACER_EFFECTIVE_LENGTH_MM:.2f} mm")
    print(f"angle magnitude = {fmt(plan.spacer_angle_magnitude_deg)} deg")
    print(f"signed spacer command = {fmt(plan.spacer_command_angle_deg)} deg")
    print(f"spacer required = {plan.spacer_required}")
    print(
        "diagnostic current opening at 100 mm = "
        f"{fmt(diagnostic_current_opening_mm)} mm"
    )
    print(
        "current opening used for angle calculation = "
        f"{plan.current_opening_used_for_angle_calculation}"
    )
    print(
        "xArm lateral correction enabled = "
        f"{ENABLE_XARM_LATERAL_CORRECTION}"
    )
    print(f"feasible = {plan.feasible}")
    print(f"decision = {plan.decision}")
    print(f"reason = {plan.reason}")
    if plan.triangular_geometry is not None:
        log_triangular_spacer_geometry(
            plan.triangular_geometry,
            selected_space_width_info=selected_space_width_info,
        )


def _is_no_space_error(exc: RuntimeError) -> bool:
    return "収納スペース候補が見つかりませんでした" in str(exc)


def _is_retryable_storage_recognition_error(exc: RuntimeError) -> bool:
    """Keep only known image/geometry failures nonfatal between two views."""
    message = str(exc)
    return _is_no_space_error(exc) or "geometry calculation failed" in message


def _storage_recognition_error_type(message: str | None) -> str:
    if message is None:
        return "none"
    if "geometry calculation failed" in message:
        return "geometry"
    if "収納スペース候補が見つかりませんでした" in message:
        return "no_space"
    return "recognition"


def _latest_capture_shot_dir(capture_root: Path) -> str | None:
    """Recover the per-shot directory even when recognition raises later."""
    if not capture_root.is_dir():
        return None
    shot_dirs = [path for path in capture_root.iterdir() if path.is_dir()]
    if not shot_dirs:
        return None
    return str(max(shot_dirs, key=lambda path: path.name))


def log_storage_capture_candidate(
    candidate: StorageCaptureCandidate,
    *,
    continue_to_second_capture: bool = False,
) -> None:
    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    print("\n[STORAGE TWO-VIEW CANDIDATE]")
    print(f"capture = {candidate.label}")
    print(f"shot_dir = {candidate.shot_dir or 'n/a'}")
    print(f"recognition success = {candidate.recognition_success}")
    print(
        "classification = "
        f"{candidate.decision.classification if candidate.decision else 'n/a'}"
    )
    print(f"supported = {candidate.supported}")
    print(f"plan feasible = {candidate.plan_feasible}")
    print(
        "physical bottleneck = "
        f"{fmt(candidate.physical_bottleneck_mm)} mm"
    )
    print(
        "selected-space height = "
        f"{candidate.selected_space_height_px} px"
    )
    print(
        "valid physical rows = "
        f"{candidate.valid_physical_row_count}"
    )
    if candidate.physical_vertical_width_check is not None:
        width_check = candidate.physical_vertical_width_check
        print("\n[STORAGE SPACE WIDTH CHECK]")
        print(
            "physical top width = "
            f"{float(width_check['physical_top_width_mm']):.2f} mm"
        )
        print(
            "physical bottom width = "
            f"{float(width_check['physical_bottom_width_mm']):.2f} mm"
        )
        print(
            "minimum width = "
            f"{float(width_check['minimum_space_width_mm']):.2f} mm"
        )
        print(
            "held book width = "
            f"{float(width_check['held_book_width_mm']):.2f} mm"
        )
        print(f"bottom narrowing = {width_check['bottom_narrowing']}")
        print(
            "reject = "
            f"{width_check['bottom_narrowing_width_reject']}"
        )
    if candidate.recognition_error is not None:
        print(
            "error type = "
            f"{_storage_recognition_error_type(candidate.recognition_error)}"
        )
        print(f"error = {candidate.recognition_error}")
    elif candidate.planning_error is not None:
        print("error type = classification/planning")
        print(f"error = {candidate.planning_error}")
    elif not candidate.usable:
        print(f"error = {candidate.failure_reason}")
    print(f"candidate usable = {candidate.usable}")
    if continue_to_second_capture:
        print("continue to 2nd capture = True")


def log_two_view_selection(
    candidates: list[StorageCaptureCandidate],
    selected: StorageCaptureCandidate,
    reason: str,
) -> None:
    ordered = sorted(candidates, key=lambda candidate: candidate.capture_index)
    print("\n[STORAGE TWO-VIEW SELECTION]")
    print(f"selected capture = {selected.label}")
    print(f"reason = {reason}")
    for candidate in ordered:
        bottleneck = candidate.physical_bottleneck_mm
        bottleneck_text = "n/a" if bottleneck is None else f"{bottleneck:.2f}"
        print(
            f"{candidate.label} bottleneck = {bottleneck_text} mm, "
            f"space height = {candidate.selected_space_height_px} px, "
            f"valid physical rows = {candidate.valid_physical_row_count}, "
            f"usable = {candidate.usable}"
        )


def _format_timing_seconds(value: Any, unavailable: str = "n/a") -> str:
    return unavailable if value is None else f"{float(value):.3f} s"


def _print_storage_timing_summary(
    camera_session: Any,
    view_timings: dict[int, dict[str, float]],
    storage_recognition_total_s: float,
) -> None:
    executed_timings = [
        view_timings[index]
        for index in sorted(view_timings)
        if "recognition_compute_time_s" in view_timings[index]
    ]
    recognition_compute_times = [
        timing["recognition_compute_time_s"] for timing in executed_timings
    ]
    camera_inclusive_times = [
        timing["camera_inclusive_view_time_s"] for timing in executed_timings
    ]
    print("\n" + "=" * 60)
    print("[TIMING SUMMARY]\n")
    print("RealSense session")
    print(
        "  Pipeline start              : "
        f"{_format_timing_seconds(camera_session.pipeline_start_time_s)}"
    )
    print(
        "  Initial warm-up             : "
        f"{_format_timing_seconds(camera_session.initial_warmup_time_s)}"
    )
    print(
        "  Pipeline stop               : "
        f"{_format_timing_seconds(camera_session.pipeline_stop_time_s)}"
    )
    print(
        "  Lifecycle counts            : "
        f"start={camera_session.start_count}, "
        "warm-up="
        f"{1 if camera_session.initial_warmup_time_s is not None else 0}, "
        f"stop={camera_session.stop_count}"
    )

    for view_number in range(1, 4):
        print(f"\nVIEW {view_number}")
        timing = view_timings.get(view_number)
        if timing is None or "recognition_compute_time_s" not in timing:
            print("  not executed")
            continue
        print(
            "  Capture                     : "
            f"{_format_timing_seconds(timing.get('capture_time_s'))}"
        )
        print(
            "  Recognition compute         : "
            f"{_format_timing_seconds(timing.get('recognition_compute_time_s'))}"
        )
        print(
            "  Camera-inclusive total      : "
            f"{_format_timing_seconds(timing.get('camera_inclusive_view_time_s'))}"
        )

    recognition_count = len(recognition_compute_times)
    print("\nSUMMARY")
    print(f"  Recognition count           : {recognition_count}")
    print(
        "  Average recognition compute : "
        f"{_format_timing_seconds(sum(recognition_compute_times) / recognition_count)}"
    )
    print(
        "  Average camera-inclusive    : "
        f"{_format_timing_seconds(sum(camera_inclusive_times) / recognition_count)}"
    )
    print(
        "  Total recognition compute   : "
        f"{_format_timing_seconds(sum(recognition_compute_times))}"
    )
    print(
        "  Storage recognition total  : "
        f"{_format_timing_seconds(storage_recognition_total_s)}"
    )
    print("  (RealSense session start through session stop)")
    print("=" * 60)


class Tee:
    """ターミナルへの表示とファイル保存を同時に行う。"""

    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self):
        return self.terminal.isatty()

def run_one_storage_cycle(
    *,
    hand_worker: Any,
    xarm7: Any,
    cycle_index: int,
    capture_cycle_root: Path,
    camera_session: Any,
) -> bool:
    """Run one complete grasp/recognition/storage/return cycle.

    Hardware objects are owned by :func:`main` and reused here.  Returning
    ``False`` preserves the existing Ctrl+D meaning: end the whole program
    normally after returning toward the capture side.
    """
    from Dynamixel_win_pro_hand_book.HandBook_Storage import (
        RIGHT_TILTED_SP_LIN_REDUCTION_STEPS,
        SP_LIN_FRONT,
        SP_LIN_RIGHT_TILTED_FRONT,
    )
    from detection.pro_handbook.sam_py_demo.Storage_SAM3 import (
        apply_selected_triangular_space_mask_correction,
        run_capture_and_pca_depth_space,
    )
    from xarm7.control.robot_base_coordinate import cam_mm_to_robot_mm
    from xarm7.control.xarm7 import (
        TCP_ACC_1,
        TCP_ACC_2,
        TCP_VEL_1,
        TCP_VEL_2,
    )

    try:
        # This check intentionally runs before the first cycle motion.
        validate_storage_0911_depth_constants()
        validate_spacer_virtual_tcp_runtime_configuration()
        print(
            "\n"
            "==================================================\n"
            f" STORAGE CYCLE {cycle_index} START\n"
            "=================================================="
        )

        # A new function frame is created for every cycle.  Keep all mutable
        # recognition, planning, target, and spacer state explicit here so no
        # result or capture TCP pose can leak from the previous book.
        capture_tcp_pose_1 = None
        capture_tcp_pose_2 = None
        capture_tcp_pose_3 = None
        decision = None
        res = None
        recognition_source = None
        rectangular_bottleneck_info = None
        selected_space_width_info = None
        diagnostic_current_opening_mm = None
        motion_target_cam = None
        right_side_target_info = None
        left_boundary_target_info = None
        spacer_required = False
        spacer_rotation_reset = False

        ret = xarm7.moveJ_to_capture_right_strage(asynchronous=False)
        print("[STORAGE] capture pose ret =", ret)
        if ret != 0:
            raise RuntimeError(f"capture pose move failed: ret={ret}")

        capture_offset_before_pose = np.asarray(
            xarm7.get_tcp_pose(is_radian=True),
            dtype=np.float64,
        ).reshape(6)
        print("\n[STORAGE CAPTURE HEIGHT OFFSET]")
        print(
            "capture Z offset = "
            f"{STORAGE_CAPTURE_Z_OFFSET_MM:+.2f} mm"
        )
        if STORAGE_CAPTURE_Z_OFFSET_MM < 0.0:
            print("direction = robot -Z")
        elif STORAGE_CAPTURE_Z_OFFSET_MM > 0.0:
            print("direction = robot +Z")
        else:
            print("direction = none (0 mm offset)")
        print("applied before grasp = True")
        print("before offset TCP =", capture_offset_before_pose.tolist())
        capture_z_ret = xarm7.moveL_z_offset(
            z_offset=STORAGE_CAPTURE_Z_OFFSET_MM,
            asynchronous=False,
        )
        print("capture Z MoveL ret =", capture_z_ret)
        if capture_z_ret != 0:
            raise RuntimeError(
                "storage capture Z offset failed after grasp: "
                f"ret={capture_z_ret}"
            )
        capture_offset_after_pose = np.asarray(
            xarm7.get_tcp_pose(is_radian=True),
            dtype=np.float64,
        ).reshape(6)
        print("after offset TCP =", capture_offset_after_pose.tolist())
        capture_offset_state = xarm7.get_state()
        capture_offset_error = int(xarm7.arm.error_code)
        capture_offset_warn = int(xarm7.arm.warn_code)
        print(
            "[STORAGE CAPTURE HEIGHT OFFSET] xArm check: "
            f"state={capture_offset_state}, "
            f"error={capture_offset_error}, warn={capture_offset_warn}"
        )
        if capture_offset_state in (4, 5) or capture_offset_error != 0:
            raise RuntimeError(
                "xArm abnormal after storage capture Z offset: "
                f"state={capture_offset_state}, "
                f"error={capture_offset_error}, warn={capture_offset_warn}"
            )
        # Z降下完了後に書籍を把持
        hand_worker.open_until_full(asynchronous=False)
        time.sleep(GRIPPER_OPEN_TO_GRASP_WAIT_SEC)
        hand_worker.grasp()
        
        print(f"1st capture Z offset applied = {STORAGE_CAPTURE_Z_OFFSET_MM != 0.0}")
        print(f"2nd capture keeps same Z offset = {STORAGE_CAPTURE_Z_OFFSET_MM:+.2f} mm")

        def move_to_second_capture() -> None:
            nonlocal capture_tcp_pose_1
            capture_tcp_pose_1 = xarm7.get_tcp_pose(is_radian=True)
            print("[STORAGE] 1st capture TCP pose =", capture_tcp_pose_1)
            second_capture_move_ret = xarm7.moveL_y_offset(
                SECOND_CAPTURE_DY_MM,
                velocity=CAPTURE_SHIFT_VELOCITY_MM_S * 1.5,
                acceleration=CAPTURE_SHIFT_ACCELERATION_MM_S2 * 1.5,
                asynchronous=True,
            )
            print(
                "[STORAGE] move to 2nd capture ret =",
                second_capture_move_ret,
            )
            if second_capture_move_ret != 0:
                raise RuntimeError(
                    "move to second storage capture failed: "
                    f"ret={second_capture_move_ret}"
                )

        def wait_xarm_motion_done(timeout: float = XARM_MOTION_DONE_TIMEOUT_SEC) -> None:
            start_time = time.monotonic()
            moving_detected = False
            while True:
                result = xarm7.arm.get_is_moving()
                if isinstance(result, bool):
                    moving = result
                elif isinstance(result, (tuple, list)) and len(result) >= 2:
                    if int(result[0]) != 0:
                        raise RuntimeError(f"get_is_moving failed: {result}")
                    moving = bool(result[1])
                else:
                    raise RuntimeError(
                        f"unexpected get_is_moving result: {result}"
                    )
                if moving:
                    moving_detected = True
                if moving_detected and not moving:
                    return
                elapsed = time.monotonic() - start_time
                if elapsed >= XARM_MOTION_IDLE_CONFIRM_SEC and not moving_detected and not moving:
                    return
                if elapsed >= timeout:
                    raise RuntimeError(
                        f"xArm motion timeout: {elapsed:.1f} sec"
                    )
                time.sleep(XARM_MOTION_POLL_INTERVAL_SEC)

        first_capture_root = capture_cycle_root / "1st"
        second_capture_root = capture_cycle_root / "2nd"
        third_capture_root = capture_cycle_root / "3rd"
        view_timings: dict[int, dict[str, float]] = {}
        storage_recognition_started = time.perf_counter()
        camera_session.start()

        first_result = None
        first_recognition_error: str | None = None
        view_timings[1] = {}
        print("\n========== STORAGE 1st CAPTURE ==========")
        try:
            first_result = run_capture_and_pca_depth_space(
                out_dir=first_capture_root,
                after_capture_callback=move_to_second_capture,
                show_debug_images=not AUTO_RUN,
                target_3d_mode=STORAGE_TARGET_3D_MODE,
                camera_session=camera_session,
                end_to_end_timing_sink=view_timings[1],
            )
        except RuntimeError as exc:
            if not _is_retryable_storage_recognition_error(exc):
                raise
            first_recognition_error = str(exc)
            print(
                "[STORAGE] 1st recognition/geometry failure recorded; "
                "continue to 2nd capture"
            )

        if capture_tcp_pose_1 is None:
            raise RuntimeError(
                "1st capture completed without a saved capture-time TCP pose"
            )
        first_shot_dir = (
            None
            if first_result is None
            else str(first_result[2].get("shot_dir") or "") or None
        )
        if first_shot_dir is None:
            first_shot_dir = _latest_capture_shot_dir(first_capture_root)
        print(f"[STORAGE] 1st shot_dir = {first_shot_dir or 'n/a'}")

        wait_xarm_motion_done()
        second_capture_state = xarm7.get_state()
        second_capture_error = int(xarm7.arm.error_code)
        second_capture_warn = int(xarm7.arm.warn_code)
        print(
            "[STORAGE] arrived at 2nd capture position: "
            f"state={second_capture_state}, error={second_capture_error}, "
            f"warn={second_capture_warn}"
        )
        if second_capture_state in (4, 5) or second_capture_error != 0:
            raise RuntimeError(
                "xArm abnormal before second storage capture: "
                f"state={second_capture_state}, error={second_capture_error}, "
                f"warn={second_capture_warn}"
            )
        capture_tcp_pose_2 = xarm7.get_tcp_pose(is_radian=True)
        print("[STORAGE] 2nd capture TCP pose =", capture_tcp_pose_2)

        second_result = None
        second_recognition_error: str | None = None
        view_timings[2] = {}
        print("\n========== STORAGE 2nd CAPTURE ==========")
        try:
            second_result = run_capture_and_pca_depth_space(
                out_dir=second_capture_root,
                after_capture_callback=None,
                show_debug_images=not AUTO_RUN,
                target_3d_mode=STORAGE_TARGET_3D_MODE,
                camera_session=camera_session,
                end_to_end_timing_sink=view_timings[2],
            )
        except RuntimeError as exc:
            if not _is_retryable_storage_recognition_error(exc):
                raise
            second_recognition_error = str(exc)
            print("[STORAGE] 2nd recognition/geometry failure recorded")

        second_shot_dir = (
            None
            if second_result is None
            else str(second_result[2].get("shot_dir") or "") or None
        )
        if second_shot_dir is None:
            second_shot_dir = _latest_capture_shot_dir(second_capture_root)
        print(f"[STORAGE] 2nd shot_dir = {second_shot_dir or 'n/a'}")
        if (
            first_shot_dir is not None
            and second_shot_dir is not None
            and Path(first_shot_dir) == Path(second_shot_dir)
        ):
            raise RuntimeError(
                "two-view captures resolved to the same shot_dir; refusing "
                "to continue after possible overwrite"
            )

        if first_result is None:
            first_candidate = StorageCaptureCandidate(
                label="1st",
                capture_index=1,
                capture_tcp_pose=tuple(
                    np.asarray(capture_tcp_pose_1, dtype=np.float64)
                    .reshape(6)
                    .tolist()
                ),
                shot_dir=first_shot_dir,
                recognition_error=(
                    first_recognition_error or "recognition result is missing"
                ),
            )
        else:
            first_candidate = evaluate_storage_capture_candidate(
                label="1st",
                capture_index=1,
                result=first_result,
                capture_tcp_pose=capture_tcp_pose_1,
            )

        if second_result is None:
            second_candidate = StorageCaptureCandidate(
                label="2nd",
                capture_index=2,
                capture_tcp_pose=tuple(
                    np.asarray(capture_tcp_pose_2, dtype=np.float64)
                    .reshape(6)
                    .tolist()
                ),
                shot_dir=second_shot_dir,
                recognition_error=(
                    second_recognition_error or "recognition result is missing"
                ),
            )
        else:
            second_candidate = evaluate_storage_capture_candidate(
                label="2nd",
                capture_index=2,
                result=second_result,
                capture_tcp_pose=capture_tcp_pose_2,
            )

        candidates = [first_candidate, second_candidate]
        for candidate in candidates:
            if candidate.decision is not None:
                log_storage_space_classification(
                    candidate.decision,
                    f"{candidate.label} capture",
                )
            log_storage_capture_candidate(
                candidate,
                continue_to_second_capture=(candidate.capture_index == 1),
            )

        if any(candidate.usable for candidate in candidates):
            selected_candidate, selection_reason = select_best_storage_capture(
                candidates
            )
            log_two_view_selection(
                candidates,
                selected_candidate,
                selection_reason,
            )
        else:
            print(
                "[STORAGE] 1st and 2nd candidates are unusable -> "
                "try 3rd capture"
            )
            first_pose = np.asarray(
                capture_tcp_pose_1,
                dtype=np.float64,
            ).reshape(6)
            current_pose = np.asarray(
                xarm7.get_tcp_pose(is_radian=True),
                dtype=np.float64,
            ).reshape(6)
            third_target_y_mm = (
                float(capture_tcp_pose_1[1]) - SECOND_CAPTURE_DY_MM
            )
            third_move_dy_mm = third_target_y_mm - float(current_pose[1])

            print("[STORAGE] 3rd target is symmetric about 1st capture")
            print(f"[STORAGE] 1st capture Y = {float(first_pose[1]):.3f} mm")
            print(
                "[STORAGE] 2nd/current Y = "
                f"{float(current_pose[1]):.3f} mm"
            )
            print(f"[STORAGE] 3rd target Y = {third_target_y_mm:.3f} mm")
            print(f"[STORAGE] move to 3rd capture dY = {third_move_dy_mm:.3f} mm")

            third_capture_move_ret = xarm7.moveL_y_offset(
                third_move_dy_mm,
                velocity=THIRD_CAPTURE_VELOCITY_MM_S,
                acceleration=THIRD_CAPTURE_ACCELERATION_MM_S2,
                asynchronous=True,
            )
            if third_capture_move_ret != 0:
                raise RuntimeError(
                    "move to third storage capture failed: "
                    f"ret={third_capture_move_ret}"
                )

            wait_xarm_motion_done()
            third_capture_state = xarm7.get_state()
            third_capture_error = int(xarm7.arm.error_code)
            third_capture_warn = int(xarm7.arm.warn_code)
            print(
                "[STORAGE] arrived at 3rd capture position: "
                f"state={third_capture_state}, error={third_capture_error}, "
                f"warn={third_capture_warn}"
            )
            if third_capture_state in (4, 5) or third_capture_error != 0:
                raise RuntimeError(
                    "xArm abnormal before third storage capture: "
                    f"state={third_capture_state}, "
                    f"error={third_capture_error}, "
                    f"warn={third_capture_warn}"
                )

            capture_tcp_pose_3 = xarm7.get_tcp_pose(is_radian=True)
            print("[STORAGE] 3rd capture TCP pose =", capture_tcp_pose_3)

            third_result = None
            third_recognition_error: str | None = None
            view_timings[3] = {}
            print("\n========== STORAGE 3rd CAPTURE ==========")
            try:
                third_result = run_capture_and_pca_depth_space(
                    out_dir=third_capture_root,
                    after_capture_callback=None,
                    show_debug_images=not AUTO_RUN,
                    target_3d_mode=STORAGE_TARGET_3D_MODE,
                    camera_session=camera_session,
                    end_to_end_timing_sink=view_timings[3],
                )
            except RuntimeError as exc:
                if not _is_retryable_storage_recognition_error(exc):
                    raise
                third_recognition_error = str(exc)
                print("[STORAGE] 3rd recognition/geometry failure recorded")

            third_shot_dir = (
                None
                if third_result is None
                else str(third_result[2].get("shot_dir") or "") or None
            )
            if third_shot_dir is None:
                third_shot_dir = _latest_capture_shot_dir(third_capture_root)
            print(f"[STORAGE] 3rd shot_dir = {third_shot_dir or 'n/a'}")
            if third_shot_dir is not None and any(
                previous_shot_dir is not None
                and Path(third_shot_dir) == Path(previous_shot_dir)
                for previous_shot_dir in (first_shot_dir, second_shot_dir)
            ):
                raise RuntimeError(
                    "three-view captures resolved to the same shot_dir; "
                    "refusing to continue after possible overwrite"
                )

            if third_result is None:
                third_candidate = StorageCaptureCandidate(
                    label="3rd",
                    capture_index=3,
                    capture_tcp_pose=tuple(
                        np.asarray(capture_tcp_pose_3, dtype=np.float64)
                        .reshape(6)
                        .tolist()
                    ),
                    shot_dir=third_shot_dir,
                    recognition_error=(
                        third_recognition_error
                        or "recognition result is missing"
                    ),
                )
            else:
                third_candidate = evaluate_storage_capture_candidate(
                    label="3rd",
                    capture_index=3,
                    result=third_result,
                    capture_tcp_pose=capture_tcp_pose_3,
                )

            if third_candidate.decision is not None:
                log_storage_space_classification(
                    third_candidate.decision,
                    "3rd capture",
                )
            log_storage_capture_candidate(third_candidate)

            if not third_candidate.usable:
                raise RuntimeError(
                    "[STORAGE THREE-VIEW SELECTION FAILED]\n"
                    f"1st: {first_candidate.failure_reason}\n"
                    f"2nd: {second_candidate.failure_reason}\n"
                    f"3rd: {third_candidate.failure_reason}\n"
                    "no usable storage capture"
                )

            selected_candidate = third_candidate
            selection_reason = "1st and 2nd unusable; 3rd usable"
            candidates.append(third_candidate)
            print(
                "[STORAGE] selected 3rd capture because 1st and 2nd "
                "were unusable"
            )
            log_two_view_selection(
                candidates,
                selected_candidate,
                selection_reason,
            )

        camera_session.stop()
        storage_recognition_total_s = (
            time.perf_counter() - storage_recognition_started
        )
        _print_storage_timing_summary(
            camera_session,
            view_timings,
            storage_recognition_total_s,
        )

        if (
            selected_candidate.result is None
            or selected_candidate.decision is None
            or selected_candidate.spacer_plan is None
            or selected_candidate.capture_tcp_pose is None
        ):
            raise RuntimeError(
                "selected storage candidate is missing required state"
            )
        angle_rad, first_target_cam, res = selected_candidate.result
        decision = selected_candidate.decision
        # Freeze the original classification/capture choice, then correct only
        # the final selected mask used by target selection and visualization.
        first_target_cam, res = apply_selected_triangular_space_mask_correction(
            res,
            first_target_cam,
            space_type=decision.classification,
        )
        spacer_plan = selected_candidate.spacer_plan
        rectangular_bottleneck_info = (
            selected_candidate.rectangular_bottleneck_info
        )
        selected_space_width_info = (
            selected_candidate.selected_space_width_info
        )
        diagnostic_current_opening_mm = (
            selected_candidate.diagnostic_current_opening_mm
        )
        coordinate_transform_arm = FrozenTcpArm(
            xarm7,
            selected_candidate.capture_tcp_pose,
        )
        recognition_source = selected_candidate.label
        print(
            "[STORAGE] coordinate transform uses saved "
            f"{recognition_source} capture TCP pose"
        )

        log_storage_spacer_plan(
            spacer_plan,
            rectangular_bottleneck_info=rectangular_bottleneck_info,
            selected_space_width_info=selected_space_width_info,
            diagnostic_current_opening_mm=diagnostic_current_opening_mm,
        )
        if ENABLE_XARM_LATERAL_CORRECTION:
            raise RuntimeError(
                "ENABLE_XARM_LATERAL_CORRECTION must remain False for this "
                "standalone geometry-based spacer experiment"
            )
        if not spacer_plan.feasible:
            raise RuntimeError(
                "storage spacer plan aborted before reaching/spacer motion: "
                f"decision={spacer_plan.decision}, "
                f"reason={spacer_plan.reason}"
            )
        spacer_angle_magnitude_deg = (
            spacer_plan.spacer_angle_magnitude_deg
        )
        spacer_command_angle_deg = spacer_plan.spacer_command_angle_deg

        # 左右の傾斜時に、210mmの書籍長さから横方向変位を算出し、
        # arcsin(変位/50 mm)でSP_ROTの回転角度を決める。
        # asinの入力は[0, 1]に制限し、角度の符号は従来どおり維持する。
        if decision.classification in {
            "triangular_left_tilted", "triangular_right_tilted"
        } and spacer_plan.spacer_required:
            guide_angle = decision.image_guide_angle_deg
            if guide_angle is None or not np.isfinite(guide_angle):
                raise RuntimeError("SP_ROT arcsin: invalid image guide angle")
            if spacer_command_angle_deg is None:
                raise RuntimeError("SP_ROT arcsin: original command angle missing")
            signed_roll_offset_deg = 90.0 - float(guide_angle)
            if abs(signed_roll_offset_deg) > MAX_REACH_ROLL_DEG:
                raise RuntimeError(
                    f"SP_ROT arcsin: excessive tilt {signed_roll_offset_deg:+.2f} deg"
                )
            lateral_displacement_mm = 210.0* abs(
                np.sin(np.deg2rad(signed_roll_offset_deg))
            )
            ratio = min(1.0, lateral_displacement_mm /35.0)
            original_spacer_angle_deg = float(spacer_command_angle_deg)
            angle_magnitude_deg = float(np.rad2deg(np.arcsin(ratio)))
            spacer_command_angle_deg = float(np.copysign(
                angle_magnitude_deg, original_spacer_angle_deg
            ))
            print(
                "[TILTED SP_ROT ARCSIN 200MM] "
                f"classification={decision.classification}, "
                f"roll_offset={signed_roll_offset_deg:+.2f} deg, "
                f"displacement={lateral_displacement_mm:.2f} mm, "
                f"ratio={ratio:.3f}, "
                f"angle={original_spacer_angle_deg:+.2f} "
                f"-> {spacer_command_angle_deg:+.2f} deg"
            )

        # 認識画像 final_space_target_point.png の点をそのまま使用
        # 分類ごとの左右境界への目標点変更は行わない
        motion_target_cam = np.asarray(
            first_target_cam,
            dtype=np.float64,
        ).reshape(3).copy()

        right_side_target_info = None
        left_boundary_target_info = None

        print("\n[STORAGE RAW RECOGNITION TARGET]")
        print("target method = raw recognition point")
        print(f"recognition camera XYZ [m] = {first_target_cam.tolist()}")
        print(f"motion camera XYZ [m] = {motion_target_cam.tolist()}")
        print("additional camera target shift = 0 mm")

        angle_deg = float(np.degrees(angle_rad))
        oblique_line = float(res.get("guide_edge_length_mm", 0.0))
        dy = -oblique_line * np.cos(angle_rad) / 2.0

        print("========== run_capture_and_pca result ==========")
        print(f"[RESULT] angle_rad = {angle_rad:.6f} rad")
        print(f"[RESULT] angle_deg = {angle_deg:.2f} deg")
        print(f"[RESULT] recognition_source = {recognition_source}")
        print(f"[RESULT] pair_indices = {res.get('pair_indices')}")
        print(f"[RESULT] line_p0 = {res.get('line_p0')}")
        print(f"[RESULT] line_p1 = {res.get('line_p1')}")
        print(f"[RESULT] guide_side = {res.get('guide_side')}")
        print(f"[RESULT] target pixel = {res.get('first_target_px_selected')}")
        print(f"[RESULT] depth_ref_m = {res.get('depth_ref_m')}")
        print(f"[RESULT] target 3D mode = {res.get('target_3d_mode')}")
        print(f"[RESULT] target camera XYZ [m] = {first_target_cam.tolist()}")
        print(f"[RESULT] target camera Z [m] = {float(first_target_cam[2])}")
        print("================================================")

        if (
            left_boundary_target_info is not None
            and decision.classification == "triangular_left_tilted"
        ):
            print("\n[RIGHT-TILTED REACH TARGET]")
            print("classification = triangular_left_tilted")
            print("image tilted side = left boundary")
            print("real-world tilted side = RIGHT")
            print("reach boundary = real-world LEFT")
            print("image boundary = right")
            print("target method = image-right selected-space boundary")
            print(
                "original target px = "
                f"{left_boundary_target_info['original_target_px']}"
            )
            print(
                "modified boundary target px = "
                f"{left_boundary_target_info['modified_target_px']}"
            )
            print(
                "selected mask row run = "
                f"{left_boundary_target_info['selected_row_run_xy']}, "
                "inside_mask="
                f"{left_boundary_target_info['target_inside_selected_space_mask']}"
            )
            print(
                "original target cam [m] = "
                f"{left_boundary_target_info['original_target_cam_m']}"
            )
            print(
                "modified target cam [m] = "
                f"{left_boundary_target_info['modified_target_cam_m']}"
            )
        elif (
            right_side_target_info is not None
            and decision.classification == "triangular_right_tilted"
        ):
            print("\n[LEFT-BOOK-TILTED STORAGE]")
            print(f"classification = {decision.classification}")
            print("image tilted side = right boundary")
            print("real-world tilted side = left book")
            print("target side = real-world right")
            print("target method = existing right-side target")
            print(
                "selected right-side target = image-left side of selected "
                "space (real-world right-book side)"
            )
            print(
                "original target px = "
                f"{right_side_target_info['original_target_px']}"
            )
            print(
                "modified target px = "
                f"{right_side_target_info['modified_target_px']}"
            )
            print(
                "selected mask row run = "
                f"{right_side_target_info['selected_row_run_xy']}, "
                f"inset={right_side_target_info['target_inset_px']} px, "
                "inside_mask="
                f"{right_side_target_info['target_inside_selected_space_mask']}"
            )
            print(
                "original target cam [m] = "
                f"{right_side_target_info['original_target_cam_m']}"
            )
            print(
                "modified target cam [m] = "
                f"{right_side_target_info['modified_target_cam_m']}"
            )
            print(
                "spacer command angle = "
                f"{spacer_command_angle_deg:+.2f} deg (absolute)"
            )
            print(f"xArm d_roll = {decision.xarm_d_roll_rad:+.6f} rad")
        elif (
            right_side_target_info is not None
            and decision.classification == "direct_insert"
        ):
            print("\n[STORAGE DIRECT INSERT]")
            print("classification = direct_insert")
            print("source = ambiguous physical bottleneck fallback")
            print("target side = real-world right")
            print("target method = existing image-left right-side target")
            print(
                "original target px = "
                f"{right_side_target_info['original_target_px']}"
            )
            print(
                "modified target px = "
                f"{right_side_target_info['modified_target_px']}"
            )
            print(
                "target inside selected-space mask = "
                f"{right_side_target_info['target_inside_selected_space_mask']}"
            )
            print("normal target height = True")
            print("high reach +60 mm = False")
            print("spacer required = False")
            print(f"xArm d_roll = {decision.xarm_d_roll_rad:+.6f} rad")

        first_target_cam_mm = 1000.0 * np.asarray(
            motion_target_cam,
            dtype=np.float64,
        ).reshape(3)
        first_target_robot_mm = cam_mm_to_robot_mm(
            coordinate_transform_arm,
            first_target_cam_mm,
        )
        surface_robot_xyz = np.asarray(
            first_target_robot_mm,
            dtype=np.float64,
        ).reshape(3).copy()
        print("[STORAGE 0911] surface_robot_xyz =", surface_robot_xyz)
        depth_plan = build_storage_0911_depth_plan(surface_robot_xyz)
        normal_reach_target_xyz = np.asarray(
            depth_plan["reach_target_xyz_mm"],
            dtype=np.float64,
        ).reshape(3)
        before_reach_y_mm = float(normal_reach_target_xyz[1])
        normal_reach_target_xyz = apply_tilted_reach_y_correction(
            normal_reach_target_xyz,
            decision.classification,
        )
        if decision.classification in {
            "triangular_left_tilted",
            "triangular_right_tilted",
        }:
            reach_y_correction_mm = (
                float(normal_reach_target_xyz[1]) - before_reach_y_mm
            )
            real_world_direction = (
                "LEFT" if reach_y_correction_mm > 0.0 else "RIGHT"
            )
            print("\n[TILTED REACH Y CORRECTION]")
            print(f"space_type = {decision.classification}")
            print(f"before Y = {before_reach_y_mm:.3f} mm")
            print(
                f"correction = {reach_y_correction_mm:+.1f} mm "
                f"(real-world {real_world_direction})"
            )
            print(
                f"after Y = {float(normal_reach_target_xyz[1]):.3f} mm"
            )
        reach_target_xyz = normal_reach_target_xyz.copy()
        tilted_reach_z_offset_mm: float | None = None
        if decision.classification in {
            "triangular_left_tilted",
            "triangular_right_tilted",
        }:
            reach_target_xyz, tilted_reach_z_offset_mm = (
                build_tilted_book_high_reach_target_robot(
                    normal_reach_target_xyz,
                )
            )
            print("\n[0911 TILTED HIGH REACH]")
            print(f"classification = {decision.classification}")
            print("normal reach target XYZ [mm] =", normal_reach_target_xyz)
            print(
                "high reach Z offset = "
                f"{tilted_reach_z_offset_mm:+.2f} mm"
            )
            print("high reach target XYZ [mm] =", reach_target_xyz)
        else:
            print(
                "[STORAGE 0911] normal-height reach retained for "
                f"classification={decision.classification}"
            )
        depth_plan["classification"] = decision.classification
        depth_plan["normal_reach_target_xyz_mm"] = (
            normal_reach_target_xyz.tolist()
        )
        depth_plan["reach_target_xyz_mm"] = reach_target_xyz.tolist()
        depth_plan["tilted_high_reach_offset_mm"] = (
            tilted_reach_z_offset_mm
        )
        motion_records: list[dict[str, Any]] = [
            {"type": "depth_plan", **depth_plan},
            _spacer_virtual_tcp_status_record(decision.classification),
        ]
        motion_log_path = (
            capture_cycle_root / "storage_0911_motion_log.json"
        )
        _save_storage_0911_motion_log(motion_log_path, motion_records)
        print(f"[STORAGE 0911] motion log = {motion_log_path}")

        time.sleep(RECOGNITION_TO_REACH_WAIT_SEC)
        if AUTO_RUN:
            print("[AUTO RUN] recognition accepted -> proceed to reaching")
        else:
            try:
                input(
                    "Enter: execute upright reaching / "
                    "Ctrl+D: retract and return : "
                )
            except EOFError:
                print("[INFO] Ctrl+D detected before reaching")
                print(
                    "[STORAGE SPACER] no spacer motion was started; "
                    "skip unnecessary linear retraction"
                )
                try:
                    xarm7.moveJ_to_capture_right(asynchronous=False)
                except Exception as exc:
                    print("[WARN] moveJ_to_capture_right failed:", exc)
                return False

        print(
            f"xArm7 approaches the 0911 point derived from "
            f"{recognition_source}; left-book tilt RPY is applied during approach"
        )
        # Save the original RPY before applying the image-derived Roll in reaching.
        left_tilt_original_pose = None
        left_tilt_approach_rpy_offsets = (0.0, 0.0, 0.0)

        # Derive book tilt from the selected SAM3 image (vertical = 90 deg).
        roll_offset_rad = 0.0
        roll_offset_deg = 0.0
        if decision.classification in {
            "triangular_left_tilted", "triangular_right_tilted"
        }:
            guide_angle = decision.image_guide_angle_deg
            if guide_angle is None or not np.isfinite(guide_angle):
                raise RuntimeError("書籍の傾き角度を取得できません")
            # Positive xArm Roll tilts the book left; image angle increases rightward.
            roll_offset_deg = 90.0 - float(guide_angle)
            if abs(roll_offset_deg) > MAX_REACH_ROLL_DEG:
                raise RuntimeError(
                    f"Roll補正角度が大きすぎます: {roll_offset_deg:.2f} deg"
                )
            roll_offset_rad = float(np.deg2rad(roll_offset_deg))
            print(
                f"[TILTED ARM REACH] image angle={guide_angle:.2f} deg, "
                f"roll correction={roll_offset_deg:+.2f} deg"
            )


        # 実世界の左側の書籍が傾斜
        if (
            decision.classification == "triangular_right_tilted"
            and spacer_plan.spacer_required
        ):
            left_tilt_original_pose = np.asarray(
                xarm7.get_tcp_pose(is_radian=True),
                dtype=np.float64,
            ).reshape(6).copy()

            left_tilt_approach_rpy_offsets = tuple(np.deg2rad([
                roll_offset_deg,
                LEFT_BOOK_TILTED_REACH_PITCH_DEG,
                STORAGE_PRE_INSERT_YAW_DEG,
            ]))

        # 実世界の右側の書籍が傾斜
        elif (
            decision.classification == "triangular_left_tilted"
            and spacer_plan.spacer_required
        ):
            left_tilt_approach_rpy_offsets = tuple(np.deg2rad([
                0.0,
                LEFT_BOOK_TILTED_REACH_PITCH_DEG,
                0.0,
            ]))

        print(
            "[STORAGE REACH RPY] offsets [deg] =",
            np.rad2deg(left_tilt_approach_rpy_offsets).tolist(),
        )


        # Rotation is provided via rpy_offsets_rad, never apply it twice.
        ENABLE_TILTED_ARM_REACH = False

        # AUTO_RUN時はリーチングとSP_LIN伸長を並行して開始する。
        # SP_LINの既存APIはasynchronous=Trueなので、指令後すぐに
        # xArmのリーチングへ移る（SP_ROTは従来どおり伸長待機後）。
        spacer_required = spacer_plan.spacer_required
        sp_lin_started_at = None
        if AUTO_RUN and spacer_required:
            if spacer_command_angle_deg is None:
                raise RuntimeError("geometry-based spacer command is missing")
            print("[PARALLEL REACH/SP_LIN] partial extension start")
            hand_worker.expand_sp_lin_right_tilted(asynchronous=True)
            sp_lin_started_at = time.monotonic()
            print("[PARALLEL REACH/SP_LIN] SP_LIN command sent; start xArm reach")


        approach_pose = move_storage_0911_absolute_approach(
            xarm7=xarm7,
            reach_target_xyz=depth_plan["reach_target_xyz_mm"],
            roll_offset_rad=roll_offset_rad,
            apply_roll=ENABLE_TILTED_ARM_REACH,
            rpy_offsets_rad=left_tilt_approach_rpy_offsets,
            records=motion_records,
            log_path=motion_log_path,
            velocity=TCP_VEL_1,
            acceleration=TCP_ACC_1,
            backoff_mm=APPROACH_BACKOFF_MM,
        )

        if left_tilt_original_pose is not None:
            expected_rpy = left_tilt_original_pose[3:6] + np.asarray(left_tilt_approach_rpy_offsets)
            actual_rpy = np.asarray(approach_pose, dtype=np.float64).reshape(6)[3:6]
            rpy_error = np.rad2deg(np.arctan2(
                np.sin(actual_rpy - expected_rpy), np.cos(actual_rpy - expected_rpy)
            ))
            print("[LEFT-BOOK REACH RPY] approach errors [deg] =", rpy_error.tolist())
            if np.any(np.abs(rpy_error) > 0.3):
                raise RuntimeError("left-book reach RPY not achieved; stop before forward motion")

        # Already at the final reach point; do not make a second +X 50 mm move.
        reach_pose = approach_pose

        if AUTO_RUN:
            print("[AUTO RUN] reaching completed -> proceed to spacer")
        else:
            try:
                input(
                    "After reaching: Enter: operate spacer and insert / "
                    "Ctrl+D: return : "
                )
            except EOFError:
                print("[INFO] Ctrl+D detected after reaching")
                try:
                    xarm7.moveJ_to_capture_right(asynchronous=False)
                except Exception as exc:
                    print("[WARN] moveJ_to_capture_right failed:", exc)
                return False

        pre_insert_original_pose = None
        if spacer_required:
            if spacer_command_angle_deg is None:
                raise RuntimeError("geometry-based spacer command is missing")

            if sp_lin_started_at is None:
                # 手動モードも左右共通の短縮伸長
                print("[SP_LIN] partial expansion after reach (manual mode)")
                hand_worker.expand_sp_lin_right_tilted(asynchronous=True)
                sp_lin_started_at = time.monotonic()

            # 14秒は既存の推定待機値。並列動作による経過時間を差し引く。
            elapsed = time.monotonic() - sp_lin_started_at
            remaining = max(0.0, SPACER_EXPANSION_WAIT_SEC - elapsed)
            print(
                f"[SP_LIN] elapsed={elapsed:.2f}s, "
                f"remaining_wait={remaining:.2f}s "
                f"(configured={SPACER_EXPANSION_WAIT_SEC:.2f}s)"
            )
            if remaining > 0.0:
                time.sleep(remaining)
            # Note: 時間待機は実位置による伸長完了確認ではない。
            print("[SP_LIN] expansion wait elapsed; continue existing SP_ROT sequence")

            print(
                "[STORAGE SPACER] correct tilted neighbor: "
                f"side={decision.tilted_side}, "
                f"angle={spacer_command_angle_deg:.2f} deg"
            )
            if decision.classification == "triangular_left_tilted":
                print(
                    "[STORAGE SIDE MAPPING] image-left tilted boundary = "
                    "real-world right tilted book"
                )
                if spacer_command_angle_deg is None:
                    raise RuntimeError("RIGHT tilted SP_ROT angle is missing")
                # Open positive SP_ROT and move -Y simultaneously, no step sync.
                # The right-book tilt plan uses A5 210 mm * sin(boundary tilt).
                geometry = spacer_plan.triangular_geometry
                support_plan = (
                    geometry.right_tilted_support_plan
                    if geometry is not None else None
                )
                if support_plan is None:
                    raise RuntimeError("RIGHT tilted support plan missing")
                y_travel_mm = -float(support_plan.arm_right_total_mm)
                pre_insert_original_pose = np.asarray(
                    xarm7.get_tcp_pose(is_radian=True), dtype=np.float64,
                ).reshape(6).copy()
                open_right_tilted_with_y(
                    hand_worker=hand_worker,
                    xarm7=xarm7,
                    command_angle_deg=float(spacer_command_angle_deg),
                    travel_y_mm=y_travel_mm,
                    yaw_offset_deg=-STORAGE_PRE_INSERT_YAW_DEG,
                )
                # Keep SP_ROT open for insertion, like the LEFT-book case.
                # The existing common reset runs after insertion.
            elif decision.classification == "triangular_right_tilted":
                print(
                    "[STORAGE SIDE MAPPING] image-right tilted boundary = "
                    "real-world left tilted book"
                )
                print("\n[STORAGE SPACER TIMING]")
                print(
                    "rotation start delay = "
                    f"{SPACER_ROTATION_START_DELAY_SEC:.3f} sec"
                )
                print(
                    "rotation reset advance = "
                    f"{SPACER_ROTATION_RESET_ADVANCE_SEC:.3f} sec"
                )
                left_virtual_initial_state = None
                left_virtual_initial_pose = None
                left_virtual_initial_tip = None
                if ENABLE_SPACER_VIRTUAL_TCP:
                    if SPACER_VIRTUAL_TCP_GEOMETRY is None:
                        raise RuntimeError(
                            "spacer virtual TCP geometry disappeared at runtime"
                        )
                    left_virtual_initial_state = (
                        _read_spacer_virtual_tcp_state()
                    )
                    left_virtual_initial_pose = np.asarray(
                        xarm7.get_tcp_pose(is_radian=True),
                        dtype=np.float64,
                    ).reshape(6)
                    left_virtual_initial_tip = (
                        calculate_spacer_tip_position_robot(
                            left_virtual_initial_pose,
                            _require_spacer_virtual_tcp_geometry(),
                            sp_lin_position_steps=(
                                left_virtual_initial_state
                                .sp_lin_position_steps
                            ),
                            sp_rot_angle_deg=(
                                left_virtual_initial_state.sp_rot_angle_deg
                            ),
                        )
                    )
                time.sleep(SPACER_ROTATION_START_DELAY_SEC)

                # 左の本が傾いている場合：SP_ROTだけ回転する。
                # xArmの+Y 10 mm移動は一時的に無効化。
                hand_worker.rotate_spacer(float(spacer_command_angle_deg))
                assert_xarm_normal(xarm7)
                print(
                    "[LEFT-TILTED SPACER] "
                    f"SP_ROT rotation {float(spacer_command_angle_deg):+.2f} deg "
                    "completed; xArm +Y movement skipped"
                )
                if ENABLE_SPACER_VIRTUAL_TCP:
                    if (
                        left_virtual_initial_state is None
                        or left_virtual_initial_pose is None
                        or left_virtual_initial_tip is None
                    ):
                        raise RuntimeError(
                            "LEFT virtual TCP diagnostic state is incomplete"
                        )
                    _record_left_tilted_virtual_tcp_rotation(
                        xarm7=xarm7,
                        initial_tcp_pose=left_virtual_initial_pose,
                        initial_state=left_virtual_initial_state,
                        initial_tip_robot_mm=left_virtual_initial_tip,
                        sp_lin_target_steps=SP_LIN_FRONT,
                        records=motion_records,
                        log_path=motion_log_path,
                    )
            print(
                "[STORAGE SPACER] SP_LIN stays extended during insertion; "
                "SP_ROT remains deployed during insertion"
                if spacer_rotation_reset else
                "[STORAGE SPACER] SP_LIN/SP_ROT remain deployed during insertion"
            )
        else:
            print(
                "[STORAGE SPACER] opening is sufficient: skip spacer "
                "expansion/rotation; rotate_spacer(0) is not called"
            )

        if decision.classification == "triangular_right_tilted":
            if ENABLE_XARM_LATERAL_CORRECTION:
                lateral_start_pose = np.asarray(
                    xarm7.get_tcp_pose(is_radian=True),
                    dtype=np.float64,
                ).reshape(6)
                lateral_target_pose = lateral_start_pose.copy()
                lateral_target_pose[1] += LEFT_BOOK_LATERAL_SHIFT_MM
                print("\n[LEFT-BOOK-TILTED LATERAL PUSH]")
                print("real-world direction = left")
                print(
                    "robot-base delta = "
                    f"dX=+0.00 mm, dY={LEFT_BOOK_LATERAL_SHIFT_MM:+.2f} mm, "
                    "dZ=+0.00 mm, dRoll=dPitch=dYaw=0"
                )
                print("start pose =", lateral_start_pose.tolist())
                print("target pose =", lateral_target_pose.tolist())
                lateral_ret = xarm7.moveL_y_offset(
                    y_offset=LEFT_BOOK_LATERAL_SHIFT_MM,
                    velocity=TCP_VEL_2,
                    acceleration=TCP_ACC_2,
                    asynchronous=False,
                )
                print(
                    "[LEFT-BOOK-TILTED LATERAL PUSH] MoveL ret =",
                    lateral_ret,
                )
                if lateral_ret != 0:
                    raise RuntimeError(
                        f"left-book lateral push failed: ret={lateral_ret}"
                    )
                lateral_after_pose = np.asarray(
                    xarm7.get_tcp_pose(is_radian=True),
                    dtype=np.float64,
                ).reshape(6)
                lateral_position_error_mm = (
                    lateral_after_pose[:3] - lateral_target_pose[:3]
                )
                lateral_position_error_norm_mm = float(
                    np.linalg.norm(lateral_position_error_mm)
                )
                print("after pose =", lateral_after_pose.tolist())
                print(
                    "position error [mm] = "
                    f"{lateral_position_error_mm.tolist()}, "
                    f"norm={lateral_position_error_norm_mm:.3f}"
                )
                lateral_xarm_state = xarm7.get_state()
                lateral_xarm_error = int(xarm7.arm.error_code)
                lateral_xarm_warn = int(xarm7.arm.warn_code)
                print(
                    "[LEFT-BOOK-TILTED LATERAL PUSH] xArm check: "
                    f"state={lateral_xarm_state}, "
                    f"error={lateral_xarm_error}, warn={lateral_xarm_warn}"
                )
                if (
                    lateral_xarm_state in (4, 5)
                    or lateral_xarm_error != 0
                ):
                    raise RuntimeError(
                        "xArm abnormal after left-book lateral push: "
                        f"state={lateral_xarm_state}, "
                        f"error={lateral_xarm_error}, "
                        f"warn={lateral_xarm_warn}"
                    )
            else:
                print(
                    "[STORAGE SPACER ANGLE EXPERIMENT] xArm +Y 5 mm "
                    "lateral correction disabled; start +55 mm insertion"
                )

        # 実世界の右側書籍が傾斜する場合は、SP_ROT/−Yと同時に
        # Yaw +6度を実行済み。ここで二度目のYawは絶対に加えない。
        # pre_insert_original_pose は開く直前に保存した元のYawを保持する。
        if spacer_required and decision.classification == "triangular_left_tilted":
            if pre_insert_original_pose is None:
                raise RuntimeError("RIGHT tilted original Yaw pose was not saved")
            current_yaw_pose = np.asarray(
                xarm7.get_tcp_pose(is_radian=True), dtype=np.float64,
            ).reshape(6)
            yaw_opened_deg = np.degrees(np.arctan2(
                np.sin(current_yaw_pose[5] - pre_insert_original_pose[5]),
                np.cos(current_yaw_pose[5] - pre_insert_original_pose[5]),
            ))
            expected_yaw_deg = -STORAGE_PRE_INSERT_YAW_DEG
            print("[RIGHT-TILTED PRE-INSERT] additional Yaw = 0 deg")
            print(f"[RIGHT-TILTED PRE-INSERT] already rotated Yaw = {yaw_opened_deg:+.3f} deg")
            if abs(yaw_opened_deg - expected_yaw_deg) > STORAGE_INSERT_YAW_TOLERANCE_DEG:
                raise RuntimeError(
                    "RIGHT tilted Yaw mismatch before insertion: "
                    f"actual={yaw_opened_deg:+.3f} deg, "
                    f"expected={expected_yaw_deg:+.3f} deg"
                )

        # 実世界の左側書籍が傾斜する場合
        elif spacer_required and decision.classification == "triangular_right_tilted":
            # RPY was already applied in the 0911 approach. No second rotation.
            if left_tilt_original_pose is None:
                raise RuntimeError("left-tilted original reach pose missing")
            pre_insert_original_pose = left_tilt_original_pose.copy()
            actual_tilted_pose = np.asarray(
                xarm7.get_tcp_pose(is_radian=True), dtype=np.float64,
            ).reshape(6)
            target_rpy = pre_insert_original_pose[3:6] + np.asarray(
                left_tilt_approach_rpy_offsets, dtype=np.float64
            )
            orientation_error_deg = np.rad2deg(np.arctan2(
                np.sin(actual_tilted_pose[3:6] - target_rpy),
                np.cos(actual_tilted_pose[3:6] - target_rpy),
            ))
            print("[LEFT-BOOK PRE-INSERT] RPY already applied at reaching")
            print("[LEFT-BOOK PRE-INSERT] orientation errors [deg] =", orientation_error_deg.tolist())
            if abs(orientation_error_deg[0]) > 0.3 or abs(orientation_error_deg[1]) > 0.3 or abs(orientation_error_deg[2]) > STORAGE_INSERT_YAW_TOLERANCE_DEG:
                raise RuntimeError("left-tilted RPY does not match reaching target; abort insertion")
        spacer_insert_velocity = (
            TILTED_SPACER_INSERT_VELOCITY_MM_S
            if decision.classification == "triangular_right_tilted"
            else TCP_VEL_2
        )
        spacer_insert_acceleration = (
            TILTED_SPACER_INSERT_ACCELERATION_MM_S2
            if decision.classification == "triangular_right_tilted"
            else TCP_ACC_2
        )


        # Left-book tilt: complete the Roll/Pitch/Yaw restoration BEFORE +X insertion.
        # The spacer has already rotated while the xArm moved +Y by 10 mm.
        if spacer_required and decision.classification == "triangular_right_tilted":
            if pre_insert_original_pose is None:
                raise RuntimeError("LEFT tilted original pose is missing")

            restore_start_pose = np.asarray(
                xarm7.get_tcp_pose(is_radian=True), dtype=np.float64,
            ).reshape(6)
            restore_target_pose = restore_start_pose.copy()  # hold current XYZ
            # SP_ROTにも使用した傾斜角の1/2をRollに適用する。
            # 同一の値を使うことで回転角と挿入姿勢の補正を揃える。
            left_book_tilt_deg = abs(
                90.0 - float(decision.image_guide_angle_deg)
            )
            restore_roll_offset_deg = left_book_tilt_deg / 2.0
            restore_target_pose[3] = (
                float(pre_insert_original_pose[3])
                - np.deg2rad(restore_roll_offset_deg)
            )
            restore_target_pose[4] = float(pre_insert_original_pose[4])
            restore_target_pose[5] = float(pre_insert_original_pose[5])

            print("\n[LEFT-BOOK PRE-INSERT RPY RESTORE]")
            print("start pose =", restore_start_pose.tolist())
            print("target pose =", restore_target_pose.tolist())
            motion_records.append({
                "type": "motion_command",
                "label": "LEFT-BOOK PRE-INSERT RPY RESTORE (XYZ fixed)",
                "start_tcp_pose_mm_rad": restore_start_pose.tolist(),
                "target_tcp_pose_mm_rad": restore_target_pose.tolist(),
                "velocity_mm_s": float(LEFT_TILTED_PRE_INSERT_YAW_VELOCITY_MM_S),
                "acceleration_mm_s2": float(LEFT_TILTED_PRE_INSERT_YAW_ACCELERATION_MM_S2),
            })
            _save_storage_0911_motion_log(motion_log_path, motion_records)

            restore_ret = xarm7._moveL(
                restore_target_pose.tolist(),
                float(LEFT_TILTED_PRE_INSERT_YAW_VELOCITY_MM_S),
                float(LEFT_TILTED_PRE_INSERT_YAW_ACCELERATION_MM_S2),
                False,
            )
            if restore_ret != 0:
                raise RuntimeError(
                    f"LEFT tilted pre-insert RPY restore failed: ret={restore_ret}"
                )
            wait_xarm_motion_done()
            assert_xarm_normal(xarm7)
            restored_pose = np.asarray(
                xarm7.get_tcp_pose(is_radian=True), dtype=np.float64,
            ).reshape(6)
            xyz_error_mm = restored_pose[:3] - restore_target_pose[:3]
            rpy_error_deg = np.rad2deg(np.arctan2(
                np.sin(restored_pose[3:6] - restore_target_pose[3:6]),
                np.cos(restored_pose[3:6] - restore_target_pose[3:6]),
            ))
            print("XYZ restoration error [mm] =", xyz_error_mm.tolist())
            print("RPY restoration error [deg] =", rpy_error_deg.tolist())
            if np.any(np.abs(xyz_error_mm) > STORAGE_INSERT_XYZ_TOLERANCE_MM):
                raise RuntimeError("LEFT tilted XYZ changed during RPY restore")
            if np.any(np.abs(rpy_error_deg) > STORAGE_INSERT_YAW_TOLERANCE_DEG):
                raise RuntimeError("LEFT tilted RPY restore incomplete")
            print("[LEFT-BOOK] RPY restored; ready for +X insertion")

        # Start SP_ROT reverse 1 second after the xArm insertion command starts.
        insertion_command_started = threading.Event()
        insertion_reverse_cancel = threading.Event()
        insertion_reverse_done = threading.Event()
        insertion_reverse_errors: list[Exception] = []
        insertion_reverse_thread = None

        def reverse_sp_rot_one_second_after_insert() -> None:
            try:
                insertion_command_started.wait()
                if insertion_reverse_cancel.wait(1.0):
                    return
                angle = float(SPACER_RESET_OVERSHOOT_DEG)
                if not (np.isfinite(angle) and 0.0 < angle <= 15.0):
                    raise ValueError(f"invalid SP_ROT reverse angle: {angle}")
                print(f"[INSERT +1s] SP_ROT reverse to {-angle:+.2f} deg")
                hand_worker.rotate_spacer(-angle)
                print("[INSERT +1s] SP_ROT reverse complete")
            except Exception as exc:
                insertion_reverse_errors.append(exc)
            finally:
                insertion_reverse_done.set()

        if spacer_required:
            insertion_reverse_thread = threading.Thread(
                target=reverse_sp_rot_one_second_after_insert,
                name="storage_sp_rot_reverse_after_1s",
                daemon=False,
            )
            insertion_reverse_thread.start()

        try:
            if pre_insert_original_pose is not None:
                start_insert_pose = np.asarray(
                    xarm7.get_tcp_pose(is_radian=True), dtype=np.float64,
                ).reshape(6)
                insert_end_pose = start_insert_pose.copy()
                insert_end_pose[0] += float(INSERT_WITH_SPACER_MM)
                # 2回目の挿入では、高いリーチ姿勢からのZ下降を+X挿入と同じMoveLに統合する。
                if decision.classification in {
                    "triangular_left_tilted",
                    "triangular_right_tilted",
                }:
                    if tilted_reach_z_offset_mm is None:
                        raise RuntimeError("tilted high-reach Z offset is missing")
                    insert_end_pose[2] += float(
                        normal_reach_target_xyz[2] - reach_target_xyz[2]
                    )
                # Right-side tilted book only: advance +X and +Y simultaneously.
                # LEFT-book tilt has already restored RPY before this MoveL.
                if decision.classification == "triangular_left_tilted":
                    insert_end_pose[1] += RIGHT_TILTED_INSERT_Y_MM
                # LEFT-book tilt: hold restored Roll/Pitch/Yaw during +X.
                # RIGHT-book tilt: restore Yaw only (Pitch was not modified).

                if decision.classification == "triangular_right_tilted":
                    insert_end_pose[3] = (
                        float(pre_insert_original_pose[3])
                        - np.deg2rad(restore_roll_offset_deg)
                    )

                # 左右とも挿入時にPitchを元の姿勢に戻す
                if decision.classification in {
                    "triangular_left_tilted",
                    "triangular_right_tilted",
                }:
                    if decision.classification == "triangular_left_tilted":
                        insert_end_pose[4] = (
                            float(pre_insert_original_pose[4])
                            - np.deg2rad(LEFT_BOOK_TILTED_REACH_PITCH_DEG)
                        )
                    else:
                        insert_end_pose[4] = float(pre_insert_original_pose[4])

                insert_end_pose[5] = float(pre_insert_original_pose[5])

                print("\n[0911 SPACER INSERT WITH YAW RESTORE]")
                print("start pose =", start_insert_pose.tolist())
                print("target pose =", insert_end_pose.tolist())
                print("robot +X distance [mm] =", float(INSERT_WITH_SPACER_MM))
                print("robot +Y distance [mm] =",
                      RIGHT_TILTED_INSERT_Y_MM if decision.classification == "triangular_left_tilted" else 0.0)
                print("Roll/Pitch/Yaw restored before insertion; holding orientation" if decision.classification == "triangular_right_tilted" else "Yaw restores during the same MoveL")
                motion_records.append({
                    "type": "motion_command",
                    "label": "0911 SPACER INSERT WITH YAW RESTORE",
                    "start_tcp_pose_mm_rad": start_insert_pose.tolist(),
                    "target_tcp_pose_mm_rad": insert_end_pose.tolist(),
                    "velocity_mm_s": float(spacer_insert_velocity),
                    "acceleration_mm_s2": float(spacer_insert_acceleration),
                })
                _save_storage_0911_motion_log(motion_log_path, motion_records)
                if spacer_required:
                    insertion_command_started.set()
                ret = xarm7._moveL(
                    insert_end_pose.tolist(),
                    float(spacer_insert_velocity),
                    float(spacer_insert_acceleration),
                    False,
                )
                if ret != 0:
                    raise RuntimeError(f"Yaw-restoring insertion failed: ret={ret}")
                spacer_insert_pose = np.asarray(
                    xarm7.get_tcp_pose(is_radian=True), dtype=np.float64,
                ).reshape(6)
                final_yaw_error = np.arctan2(
                    np.sin(spacer_insert_pose[5] - pre_insert_original_pose[5]),
                    np.cos(spacer_insert_pose[5] - pre_insert_original_pose[5]),
                )
                print("after insert TCP =", spacer_insert_pose.tolist())
                position_error_xyz = insert_end_pose[:3] - spacer_insert_pose[:3]
                print("insertion XYZ target error [mm] =", position_error_xyz.tolist())
                xarm_state = xarm7.get_state()
                xarm_error = int(xarm7.arm.error_code)
                if xarm_state in (4, 5) or xarm_error != 0:
                    raise RuntimeError(
                        f"xArm abnormal after insertion: state={xarm_state}, error={xarm_error}"
                    )
                if np.any(np.abs(position_error_xyz) > STORAGE_INSERT_XYZ_TOLERANCE_MM):
                    raise RuntimeError(
                        "Insertion XYZ target not reached; stop before spacer reset/retraction: "
                        f"error_xyz_mm={position_error_xyz.tolist()}"
                    )
                if decision.classification == "triangular_right_tilted":
                    final_pitch_error = np.arctan2(
                        np.sin(spacer_insert_pose[4] - pre_insert_original_pose[4]),
                        np.cos(spacer_insert_pose[4] - pre_insert_original_pose[4]),
                    )
                    print(f"Pitch restoration error = {np.rad2deg(final_pitch_error):+.3f} deg")
                if decision.classification == "triangular_left_tilted":
                    expected_pitch = (
                        float(pre_insert_original_pose[4])
                        - np.deg2rad(LEFT_BOOK_TILTED_REACH_PITCH_DEG)
                    )

                    final_pitch_error = np.arctan2(
                        np.sin(spacer_insert_pose[4] - expected_pitch),
                        np.cos(spacer_insert_pose[4] - expected_pitch),
                    )

                    print(
                        "[RIGHT-BOOK INSERT] Pitch restoration error = "
                        f"{np.rad2deg(final_pitch_error):+.3f} deg"
                    )

                    if abs(np.rad2deg(final_pitch_error)) > STORAGE_INSERT_YAW_TOLERANCE_DEG:
                        raise RuntimeError(
                            "RIGHT-book Pitch did not return to the original angle"
                        )

                print(f"Yaw restoration error = {np.rad2deg(final_yaw_error):+.3f} deg")
                if abs(np.rad2deg(final_yaw_error)) > STORAGE_INSERT_YAW_TOLERANCE_DEG:
                    raise RuntimeError("Yaw did not return to the pre-tilt angle")
            else:
                if spacer_required:
                    insertion_command_started.set()
                spacer_insert_pose = move_storage_0911_relative_x(
                    xarm7=xarm7,
                    distance_mm=INSERT_WITH_SPACER_MM,
                    label="0911 SPACER INSERT",
                    records=motion_records,
                    log_path=motion_log_path,
                    velocity=spacer_insert_velocity,
                    acceleration=spacer_insert_acceleration,
                )

        except BaseException:
            insertion_reverse_cancel.set()
            insertion_command_started.set()  # release thread if insertion never began
            if insertion_reverse_thread is not None:
                insertion_reverse_thread.join()
            raise

        # 傾斜時のZ下降は上の挿入MoveLに統合済み。後続の到達位置検証との互換性を維持。
        tilted_lower_pose: np.ndarray | None = (
            spacer_insert_pose.copy()
            if decision.classification in {
                "triangular_left_tilted", "triangular_right_tilted"
            }
            else None
        )

        # Final push and release-height lowering use ONE Cartesian motion.
        # Do not issue an additional Z move after this function.
        def final_push_with_z_lowering() -> np.ndarray:
            start_pose = np.asarray(
                xarm7.get_tcp_pose(is_radian=True), dtype=float,
            ).reshape(6)
            target_pose = start_pose.copy()
            target_pose[0] += float(FINAL_BOOK_PUSH_MM)
            target_pose[2] -= float(STORAGE_RELEASE_LOWER_Z_MM)
            print("\n[0911 FINAL PUSH + RELEASE Z SIMULTANEOUS]")
            print("start pose =", start_pose.tolist())
            print("target pose =", target_pose.tolist())
            print(f"command dX={FINAL_BOOK_PUSH_MM:+.2f} mm, "
                  f"dZ={-STORAGE_RELEASE_LOWER_Z_MM:+.2f} mm")
            motion_records.append({
                "type": "motion_command",
                "label": "0911 FINAL PUSH + RELEASE Z SIMULTANEOUS",
                "start_tcp_pose_mm_rad": start_pose.tolist(),
                "target_tcp_pose_mm_rad": target_pose.tolist(),
                "velocity_mm_s": float(FINAL_BOOK_PUSH_VELOCITY_MM_S),
                "acceleration_mm_s2": float(FINAL_BOOK_PUSH_ACCELERATION_MM_S2),
            })
            _save_storage_0911_motion_log(motion_log_path, motion_records)
            ret = xarm7._moveL(
                target_pose.tolist(),
                float(FINAL_BOOK_PUSH_VELOCITY_MM_S),
                float(FINAL_BOOK_PUSH_ACCELERATION_MM_S2),
                False,
            )
            if ret != 0:
                raise RuntimeError(f"final simultaneous XZ MoveL failed: ret={ret}")
            actual = np.asarray(
                xarm7.get_tcp_pose(is_radian=True), dtype=float,
            ).reshape(6)
            xyz_error = target_pose[:3] - actual[:3]
            state = xarm7.get_state()
            error = int(xarm7.arm.error_code)
            print("[FINAL XZ] actual pose =", actual.tolist())
            print("[FINAL XZ] XYZ target error [mm] =", xyz_error.tolist())
            if (state in (4, 5) or error != 0
                    or np.any(np.abs(xyz_error) > STORAGE_INSERT_XYZ_TOLERANCE_MM)):
                raise RuntimeError("final XZ pose was not reached: "
                                   f"state={state}, error={error}, "
                                   f"xyz_error={xyz_error.tolist()}")
            return actual

        if spacer_required:
            print("\n[STORAGE SPACER TIMING]")
            print(
                "rotation reset advance = "
                f"{SPACER_ROTATION_RESET_ADVANCE_SEC:.3f} sec"
            )
            print(
                "reset wait = "
                f"{SPACER_ROTATION_RESET_BASE_WAIT_SEC:.3f} -> "
                f"{SPACER_ROTATION_RESET_WAIT_SEC:.3f} sec"
            )
            if spacer_rotation_reset:
                raise RuntimeError(
                    "SP_ROT is already reset before parallel reset/retract/insertion"
                )
            # SP_ROT +90 -> -15 (reverse past home) -> SP_LIN BACK starts
            # -> 1s after confirmed background start, SP_ROT returns to 0.
            # The xArm final push retains its existing 2s delay after -15.
            print("\n[STORAGE SPACER OVERSHOOT -> RETRACT || FINAL PUSH]")
            print("contract_sp_lin_1 used = False")
            print(f"SP_LIN BACK timeout = {ONE_SHOT_SP_LIN_BACK_TIMEOUT_SEC:.1f} sec")
            mechanism_errors: list[Exception] = []
            mechanism_status: dict[str, bool] = {
                "rotation_overtravel": False,
                "rotation_reset": False,
                "linear_retracted": False,
                "linear_command_sent": False,
            }
            start_gate = threading.Event()
            rotation_overtravel_done = threading.Event()
            rotation_home_done = threading.Event()

            def reset_then_retract_during_final_push() -> None:
                try:
                    start_gate.wait()
                    angle = float(SPACER_RESET_OVERSHOOT_DEG)
                    hold_sec = 1.0  # seconds after SP_LIN start acknowledgement
                    if not (np.isfinite(angle) and 0.0 < angle <= 15.0):
                        raise ValueError(f"invalid SP_ROT reverse overshoot: {angle}")
                    if not (np.isfinite(hold_sec) and hold_sec >= 0.0):
                        raise ValueError(f"invalid SP_ROT overshoot hold: {hold_sec}")

                    # Reversal was initiated during insertion; do not command it twice.
                    if not insertion_reverse_done.wait(timeout=ONE_SHOT_SP_LIN_BACK_TIMEOUT_SEC):
                        raise RuntimeError("timed out waiting for insertion-time SP_ROT reverse")
                    if insertion_reverse_errors:
                        raise RuntimeError(
                            f"insertion-time SP_ROT reverse failed: {insertion_reverse_errors[0]!r}"
                        )
                    if insertion_reverse_cancel.is_set():
                        raise RuntimeError("insertion-time SP_ROT reverse was cancelled")
                    mechanism_status["rotation_overtravel"] = True
                    rotation_overtravel_done.set()

                    # SP_ROT -15°の完了を確認した直後にSP_LIN収縮を開始する。
                    print("[PARALLEL DXL] SP_ROT -15 done -> start SP_LIN BACK")
                    print("[SP_LIN BACK] starting retraction and PWM detection")

                    started = hand_worker.contract_sp_lin_2_start(
                        timeout_sec=20.0,
                    )

                    if started is not True:
                        raise RuntimeError("SP_LIN background start failed")

                    mechanism_status["linear_command_sent"] = True
                    lin_start_ack_time = time.monotonic()
                    print("[SP_LIN BACK] background start acknowledged")

                    # 正常開始を確認できた場合だけ、1秒後にSP_ROTを0°へ戻す。
                    # SP_LINの収縮/PWM監視はworker内で並列実行される。
                    remaining = hold_sec - (time.monotonic() - lin_start_ack_time)
                    if remaining > 0:
                        time.sleep(remaining)

                    status = hand_worker.sp_lin_task_status()
                    if status["failed"]:
                        raise RuntimeError(
                            f"SP_LIN failed before SP_ROT reset: {status['error']}"
                        )
                    if not status["running"] and not status["completed"]:
                        raise RuntimeError(
                            f"SP_LIN state unknown before SP_ROT reset: {status}"
                        )

                    print("[PARALLEL DXL] SP_LIN start +1s -> SP_ROT reset to 0")
                    hand_worker.reset_rot(asynchronous=False)
                    mechanism_status["rotation_reset"] = True
                    rotation_home_done.set()
                    print("[PARALLEL DXL] SP_ROT zero restored")

                except Exception as exc:
                    mechanism_errors.append(exc)
                    rotation_overtravel_done.set()
                    rotation_home_done.set()
                    print(f"[PARALLEL DXL] ERROR: {exc!r}")

            mechanism_thread = threading.Thread(
                target=reset_then_retract_during_final_push,
                name="storage_parallel_reset_retract",
                daemon=False,
            )
            mechanism_thread.start()
            final_push_error: BaseException | None = None
            final_tcp_pose: np.ndarray | None = None
            try:
                start_gate.set()

                # SP_ROTの逆回転が完了するまで待機
                if not rotation_overtravel_done.wait(timeout=ONE_SHOT_SP_LIN_BACK_TIMEOUT_SEC):
                    raise RuntimeError("SP_ROT -15 deg timeout")

                if mechanism_errors:
                    raise RuntimeError(f"SP_ROT reverse failed: {mechanism_errors[0]!r}")

                if not mechanism_status["rotation_overtravel"]:
                    raise RuntimeError("SP_ROT -15 deg not confirmed")

                print("[FINAL PUSH] SP_ROT -15 deg complete -> wait 2 seconds")

                if mechanism_errors:
                    raise RuntimeError(f"SP_LIN command failed: {mechanism_errors[0]!r}")

                print("[FINAL PUSH] starting combined +X / -Z move")

                final_tcp_pose = final_push_with_z_lowering()
            except BaseException as exc:
                final_push_error = exc
            finally:
                # On xArm failure, wait for the already-running DXL worker to
                # finish before propagating the error; never release the book.
                if final_push_error is not None:
                    mechanism_thread.join()

            if final_push_error is not None:
                raise RuntimeError(
                    "xArm final push failed during SP_ROT reset/retraction: "
                    f"{final_push_error!r}"
                ) from final_push_error

            # SP_ROTのゼロ復帰を確認
            # SP_LINの収縮完了はZ下降後に確認する
            if not rotation_home_done.wait(
                timeout=ONE_SHOT_SP_LIN_BACK_TIMEOUT_SEC
            ):
                raise RuntimeError("Timed out waiting for SP_ROT zero restoration")

            spacer_rotation_reset = mechanism_status["rotation_reset"]
            if mechanism_errors:
                raise RuntimeError(
                    "SP_ROT reset or SP_LIN retraction failed during final push: "
                    f"{mechanism_errors[0]!r}; xArm_result={final_push_error!r}"
                ) from mechanism_errors[0]
            if not spacer_rotation_reset:
                raise RuntimeError("SP_ROT reset failed; prohibit book release")
            print("[PARALLEL RELEASE] Final push completed; "
                  "SP_LIN retraction may still be in progress")
            if final_tcp_pose is None:
                raise RuntimeError("xArm final TCP pose missing after parallel operation")
        else:
            time.sleep(SPACER_ROTATION_RESET_BASE_WAIT_SEC)
            print(
                "[STORAGE SPACER] no-spacer classification: "
                "skip reset_rot and SP_LIN retraction"
            )
            final_tcp_pose = final_push_with_z_lowering()

        # SDK ret=0 is NOT enough: prior C31 runs reported very short X
        # displacement with large unintended Y/Z motion. Fail closed before
        # gripper release whenever the actual final pose misses the target.
        intended_final_xyz = np.asarray(spacer_insert_pose[:3], dtype=float).copy()
        if tilted_lower_pose is not None:
            intended_final_xyz = np.asarray(tilted_lower_pose[:3], dtype=float).copy()
        intended_final_xyz[0] += float(FINAL_BOOK_PUSH_MM)
        intended_final_xyz[2] -= float(STORAGE_RELEASE_LOWER_Z_MM)
        final_xyz_error = intended_final_xyz - final_tcp_pose[:3]
        final_state = xarm7.get_state()
        final_error = int(xarm7.arm.error_code)
        print("[0911 FINAL PUSH] target XYZ =", intended_final_xyz.tolist())
        print("[0911 FINAL PUSH] actual XYZ =", final_tcp_pose[:3].tolist())
        print("[0911 FINAL PUSH] XYZ error [mm] =", final_xyz_error.tolist())
        print(f"[0911 FINAL PUSH] xArm state={final_state}, error={final_error}")
        if final_state in (4, 5) or final_error != 0:
            raise RuntimeError(
                "xArm abnormal after final push: "
                f"state={final_state}, error={final_error}"
            )
        if np.any(np.abs(final_xyz_error) > STORAGE_INSERT_XYZ_TOLERANCE_MM):
            raise RuntimeError(
                "FINAL PUSH target not reached; prohibit book release: "
                f"XYZ error [mm]={final_xyz_error.tolist()}"
            )
        actual_book_rear_x = final_tcp_pose[0] + TCP_TO_BOOK_REAR_MM
        actual_insertion_depth = actual_book_rear_x - surface_robot_xyz[0]
        print("\n[STORAGE 0911 ACTUAL DEPTH]")
        print(
            "final actual TCP XYZ = "
            f"[{final_tcp_pose[0]:.3f}, {final_tcp_pose[1]:.3f}, "
            f"{final_tcp_pose[2]:.3f}] mm"
        )
        print(f"actual book rear X = {actual_book_rear_x:.3f} mm")
        print(
            "actual insertion depth = "
            f"{actual_insertion_depth:.3f} mm"
        )
        motion_records.append(
            {
                "type": "actual_depth_summary",
                "approach_actual_tcp_pose_mm_rad": approach_pose.tolist(),
                "reach_actual_tcp_pose_mm_rad": reach_pose.tolist(),
                "spacer_insert_actual_tcp_pose_mm_rad": (
                    spacer_insert_pose.tolist()
                ),
                "tilted_lower_actual_tcp_pose_mm_rad": (
                    None
                    if tilted_lower_pose is None
                    else tilted_lower_pose.tolist()
                ),
                "final_actual_tcp_pose_mm_rad": final_tcp_pose.tolist(),
                "actual_book_rear_x_mm": float(actual_book_rear_x),
                "actual_insertion_depth_mm": float(actual_insertion_depth),
            }
        )
        _save_storage_0911_motion_log(motion_log_path, motion_records)

        # Z lowering was completed as part of the final XZ MoveL above.
        # Final-pose XYZ tolerance and xArm state were checked before release.
        print("\n[STORAGE RELEASE] final XZ combined move verified; "
              "no separate Z lowering command")

        # バックグラウンド開始要求が終了したことだけ確認する
        if spacer_required:
            mechanism_thread.join()

            if mechanism_errors:
                raise RuntimeError(
                    "SP_LIN background start failed: "
                    f"{mechanism_errors[0]!r}"
                )

            if not mechanism_status["linear_command_sent"]:
                raise RuntimeError("SP_LIN start was not confirmed")

            status = hand_worker.sp_lin_task_status()
            if status["failed"]:
                raise RuntimeError(
                    f"SP_LIN background error: {status['error']}"
                )

            print("[PARALLEL] SP_LIN may still be retracting")


        print("[STORAGE RELEASE] combined XZ final position confirmed -> opening gripper")
        # Existing worker IPC command; HandBook_Storage delegates it to partial release.
        gripper_ok = hand_worker.open_for_storage_release()
        if gripper_ok is not True:
            raise RuntimeError(
                f"gripper release failed after Z lowering: result={gripper_ok!r}"
            )
        print("[STORAGE RELEASE] gripper released")

        ret = move_storage_post_release(
            xarm7=xarm7,
            retreat_x_mm=STORAGE_POST_RELEASE_RETREAT_X_MM,
            lower_z_mm=STORAGE_POST_RELEASE_LOWER_Z_MM,
            velocity=TCP_VEL_1,
            acceleration=TCP_ACC_1,
        )

        print("[DEBUG] move_storage_post_release ret =", ret)
        if ret != 0:
            raise RuntimeError(
                f"post-storage retreat failed: ret={ret}"
            )

        ret = xarm7.moveJ_to_capture_right(asynchronous=False)
        print("[DEBUG] moveJ_to_capture_right ret =", ret)
        if ret != 0:
            raise RuntimeError(
                f"capture-side return failed at cycle end: ret={ret}"
            )

        # 撮影姿勢への復帰完了後、SP_LIN収縮完了を待たずにグリッパを閉じる
        if spacer_required:
            status = hand_worker.sp_lin_task_status()
            if status["failed"]:
                raise RuntimeError(
                    f"SP_LIN background failed: {status['error']}"
                )

        print("[PARALLEL] Capture pose reached -> closing gripper")
        hand_worker.grasp()
        print("[PARALLEL] Gripper close command completed")

        # グリッパ閉鎖後もSP_LIN収縮完了は必ず確認する
        if spacer_required:
            print("[SP_LIN] waiting for background completion")

            deadline = time.monotonic() + 45.0

            while True:
                status = hand_worker.sp_lin_task_status()

                if status["failed"]:
                    raise RuntimeError(
                        f"SP_LIN background failed: {status['error']}"
                    )

                if status["running"] is False:
                    if status["failed"] is True:
                        raise RuntimeError(
                            f"SP_LIN failed: {status['error']}"
                        )

                    if status["completed"] is not True:
                        raise RuntimeError(
                            f"SP_LIN completion not confirmed: {status}"
                        )

                    if status.get("thread_alive") is False:
                        break

                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "SP_LIN background completion timeout"
                    )

                time.sleep(0.1)

            print("[SP_LIN] background task completed")

        print(
            "\n"
            "==================================================\n"
            f" STORAGE CYCLE {cycle_index} COMPLETED\n"
            "=================================================="
        )
        print("[STORAGE] 1シーケンス完了")
        return True

    except BaseException:
        # Do not convert Ctrl+C or runtime/hardware failures into a next-cycle
        # request.  The process-level handler in main performs the established
        # emergency stop and one-time cleanup.
        raise


def main() -> None:
    print(f"[STORAGE TARGET 3D MODE] {STORAGE_TARGET_3D_MODE}")

    # Pure validation precedes worker/client construction so an accidentally
    # enabled but uncalibrated virtual TCP cannot even connect to hardware.
    validate_spacer_virtual_tcp_runtime_configuration()

    # Lazy imports keep pure classification importable without starting or
    # requiring any hardware connection.
    from Dynamixel_win_pro_hand_book.dynamixel_worker_client import (
        DynamixelWorkerClient,
    )
    from xarm7.control.xarm7 import XArm7
    from detection.pro_handbook.sam_py_demo.modules.realsense_persistent_session import (
        RealSensePersistentSession,
    )
    import rclpy

    hand_worker = None
    xarm7 = None
    node = None
    camera_session = None

    run_retrieval_after_cycle = False

    try:
        print("start upright storage loop")
        camera_session = RealSensePersistentSession(
            width=1280,
            height=720,
            fps=6,
        )
        print("[DXL] starting Dynamixel worker")
        hand_worker = DynamixelWorkerClient()
        print("[DXL] worker initialized")

        rclpy.init()
        node = rclpy.create_node("upright_storage_node")
        xarm7 = XArm7(node=node, host=XARM_HOST)
        print(
            "[AUTO RUN] interactive confirmation = "
            f"{'disabled' if AUTO_RUN else 'enabled'}"
        )
        print(
            "[AUTO RUN] recognition GUI = "
            f"{'disabled' if AUTO_RUN else 'enabled'}"
        )

        cycle_index = 1

        while True:
            # 1冊ごとに保存先フォルダを生成
            capture_cycle_name = (
                f"cycle_{cycle_index:04d}_"
                f"{time.strftime('%Y%m%d_%H%M%S')}_"
                f"{time.time_ns()}"
            )

            capture_cycle_root = STORAGE_CAPTURE_ROOT / capture_cycle_name

            # 存在する場合は上書きせず停止
            capture_cycle_root.mkdir(
                parents=True,
                exist_ok=False,
            )

            log_path = (
                capture_cycle_root
                / STORAGE_TERMINAL_LOG_FILENAME
            )

            # ターミナル表示とTXT保存を同時に実行
            with open(log_path, "x", encoding="utf-8", buffering=1) as logfile:
                tee_out = Tee(sys.stdout, logfile)
                tee_err = Tee(sys.stderr, logfile)

                with redirect_stdout(tee_out), redirect_stderr(tee_err):
                    print(f"[STORAGE LOG] {log_path}")

                    try:
                        continue_running = run_one_storage_cycle(
                            hand_worker=hand_worker,
                            xarm7=xarm7,
                            cycle_index=cycle_index,
                            capture_cycle_root=capture_cycle_root,
                            camera_session=camera_session,
                        )
                    except BaseException:
                        # 失敗・Ctrl+C時の例外情報もTXTに残す
                        traceback.print_exc()
                        try:
                            camera_session.stop()
                        except Exception as stop_exc:
                            print(
                                "[WARN] persistent RealSense stop failed: ",
                                stop_exc,
                            )
                        raise

            if not continue_running:
                print(
                    "[STORAGE LOOP] user-requested normal exit after "
                    f"cycle {cycle_index}"
                )
                break
            print(
                f"[STORAGE] cycle {cycle_index} completed"
                " -> PWM homing completed; Retrieval disabled"
            )
            run_retrieval_after_cycle = True
            break

    except KeyboardInterrupt:
        print("\nCtrl+C detected")
        if hand_worker is not None:
            try:
                hand_worker.emergency_stop_sp_lin()
                print("[EMERGENCY] SP_LIN Torque Enable=0 verified")
            except Exception as stop_exc:
                print(f"[DANGER] SP_LIN STOP NOT VERIFIED: {stop_exc}")
        try:
            if xarm7 is not None:
                xarm7.emergency_stop()
        except Exception as exc:
            print("[WARN] xArm emergency_stop failed:", exc)
        print("storage sequence interrupted")

    except Exception as exc:
        print(
            "[STORAGE ABORT] "
            f"{type(exc).__name__}: {exc}"
        )
        if hand_worker is not None:
            try:
                hand_worker.emergency_stop_sp_lin()
                print("[EMERGENCY] SP_LIN Torque Enable=0 verified")
            except Exception as stop_exc:
                print(f"[DANGER] SP_LIN STOP NOT VERIFIED: {stop_exc}")
        try:
            if xarm7 is not None:
                xarm7.emergency_stop()
        except Exception as stop_exc:
            print("[WARN] xArm emergency_stop failed:", stop_exc)
        raise

    finally:
        try:
            if camera_session is not None:
                camera_session.stop()
        except Exception as exc:
            print("[WARN] persistent RealSense stop failed:", exc)
        try:
            if hand_worker is not None:
                hand_worker.close()
        except Exception as exc:
            print("[WARN] Dynamixel worker close failed:", exc)
        time.sleep(SHUTDOWN_SETTLE_WAIT_SEC)
        try:
            if xarm7 is not None:
                xarm7.disconnect()
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                
                rclpy.shutdown()
        except Exception as exc:
            print("[WARN] cleanup failed:", exc)
        print("dynamixel worker deactivated")
        
    if run_retrieval_after_cycle:
        print("[STORAGE] Retrieval automatic launch disabled")
        return True

    return False


if __name__ == "__main__":
    while True:
        completed = main()

        if not completed:
            print("[STORAGE] 終了します")
            break

        try:
            input("[STORAGE] Enterで次の入庫シーケンスへ ")
        except EOFError:
            print("[STORAGE] 終了します")
            break