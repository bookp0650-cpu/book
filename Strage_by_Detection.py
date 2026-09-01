"""Storage sequence that keeps the grasped book upright.

This is a hardware entry point derived from ``Storage_by_Detection2.py``.
Importing this module is hardware-free; hardware-dependent imports and object
construction are intentionally confined to :func:`main`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import time

import numpy as np


STORAGE_CAPTURE_ROOT = Path(
    "/home/book/pro_book_SAM3/pro_hand_book_python/captures/strage"
)

# Storage_rev/Storage_SAM3 has no classification threshold.  The value below
# separates the current saved upright-boundary group (about 1.1--7.1 deg)
# from the clearly tilted group (about 10.3--19.5 deg).  It must be confirmed
# again on hardware before unattended operation.
VERTICAL_ANGLE_TOLERANCE_DEG = 10.0

# This is the same current threshold as MIN_STORAGE_SPACE_WIDTH_PX in
# offline_100test_storage_space_sam3.py.  It is repeated here only as a safety
# check on the recognition result; the recognition core remains authoritative.
MIN_SUPPORTED_SPACE_WIDTH_PX = 30.0

# move_to_storage_target_xyz_and_roll() adds this value to the current TCP
# roll.  The capture pose and all pre-reaching Y moves preserve RPY, therefore
# zero keeps the grasped book at that existing upright reference.
UPRIGHT_XARM_D_ROLL_RAD = 0.0

SECOND_CAPTURE_DY_MM = -140.0
BOOK_PUSH_HEIGHT_MM = 70.0
LEFT_BOOK_LATERAL_SHIFT_MM = 5.0

# rotate_spacer(theta_deg) converts theta_deg from the configured zero
# position to one absolute Dynamixel target position.  For the image-left
# tilted case only, the new motion intentionally ignores the guide angle.
LEFT_TILTED_SPACER_ABSOLUTE_ANGLE_DEG = -90.0

# The opposite physical correction uses the opposite absolute pose.  This is
# not a relative increment: rotate_spacer() converts an angle from SP_ROT_0 to
# one absolute Dynamixel goal position.
RIGHT_TILTED_SPACER_ABSOLUTE_ANGLE_DEG = 90.0

# Reuse the existing target.target_inset_px value from
# config/shelf_storage_detection_sam3.json.  In this camera arrangement,
# image-left is the real-world right-book side.  Keeping this inset inside the
# selected-space mask prevents the target from landing on the book boundary.
REAL_WORLD_RIGHT_TARGET_INSET_PX = 6

# No existing Z value describes the extra high reach needed for the short
# spacer.  INSERT_DZ is a different X/Z insertion trajectory and PLACE_BOOK_DZ
# is the final 5 mm placement, so neither is reused here.  Set this positive
# robot-base +Z distance only after clearance has been confirmed on hardware.
LEFT_BOOK_REACH_Z_OFFSET_MM: float | None = 60.0


@dataclass(frozen=True)
class StorageSpaceClassification:
    selected_space_id: int | None
    left_boundary_tilt_deg: float | None
    right_boundary_tilt_deg: float | None
    classification: str
    tilted_side: str | None
    image_guide_angle_deg: float | None
    xarm_d_roll_rad: float
    spacer_command_angle_deg: float | None
    book_bottom_side_width_px: float | None
    reasons: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return self.classification != "unsupported"


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


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def determine_spacer_command_angle(
    classification: str,
    image_guide_angle_deg: float | None,
) -> float | None:
    """Return the spacer's absolute command angle for each motion case."""
    if classification == "rectangular":
        return None
    if image_guide_angle_deg is None:
        return None
    if classification == "triangular_left_tilted":
        # Image-left tilted means the real-world right book is tilted.
        # This one case now uses the configured absolute 90-degree pose.
        return LEFT_TILTED_SPACER_ABSOLUTE_ANGLE_DEG
    if classification == "triangular_right_tilted":
        # Image-right tilted means the real-world left book is tilted.  Use
        # the direction opposite to the established image-left correction.
        return RIGHT_TILTED_SPACER_ABSOLUTE_ANGLE_DEG
    return None


def reset_spacer_rotation_before_linear_retraction(hand_worker: Any) -> None:
    """Synchronously restore SP_ROT_0 before any linear retraction."""
    print("[STORAGE SPACER] reset rotation before linear retraction")
    print("[STORAGE SPACER] reset_rot start")
    hand_worker.reset_rot(asynchronous=False)
    print("[STORAGE SPACER] reset_rot done")


def _contiguous_x_runs(row_mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive x-runs for one binary mask row."""
    xs = np.flatnonzero(np.asarray(row_mask, dtype=bool))
    if xs.size == 0:
        return []
    split_points = np.flatnonzero(np.diff(xs) > 1) + 1
    groups = np.split(xs, split_points)
    return [(int(group[0]), int(group[-1])) for group in groups]


def select_real_world_right_side_target_pixel(
    result_metadata: dict[str, Any],
    *,
    inset_px: int = REAL_WORLD_RIGHT_TARGET_INSET_PX,
) -> tuple[tuple[int, int], dict[str, Any]]:
    """Select an image-left, mask-interior target at the existing target y.

    The camera reverses the relevant physical left/right relationship, so the
    real-world right-book side is the image-left boundary of the selected
    space.  The original target's y coordinate is retained.  The contiguous
    mask run nearest the original target is used, which avoids crossing into a
    separate component if a row contains more than one run.
    """
    selected_mask = np.asarray(result_metadata.get("selected_space_mask"))
    if selected_mask.ndim != 2:
        raise RuntimeError(
            "selected_space_mask is required for right-side target selection"
        )
    selected_mask = selected_mask.astype(bool, copy=False)
    height, width = selected_mask.shape

    original_value = result_metadata.get("first_target_px_selected")
    if original_value is None:
        original_value = result_metadata.get("first_target_px")
    if original_value is None or len(original_value) != 2:
        raise RuntimeError("original target pixel is missing")
    original_x = int(round(float(original_value[0])))
    target_y = int(round(float(original_value[1])))
    if not 0 <= target_y < height:
        raise RuntimeError(
            f"original target y is outside selected mask: y={target_y}"
        )

    safe_inset = int(inset_px)
    if safe_inset < 0:
        raise ValueError(f"target inset must be nonnegative: {safe_inset}")
    runs = _contiguous_x_runs(selected_mask[target_y])
    if not runs:
        raise RuntimeError(
            f"selected space has no mask pixels at target y={target_y}"
        )

    def distance_to_original(run: tuple[int, int]) -> tuple[int, int]:
        run_left, run_right = run
        if original_x < run_left:
            distance = run_left - original_x
        elif original_x > run_right:
            distance = original_x - run_right
        else:
            distance = 0
        return distance, -(run_right - run_left + 1)

    run_left, run_right = min(runs, key=distance_to_original)
    run_width = run_right - run_left + 1
    minimum_run_width = 2 * safe_inset + 1
    if run_width < minimum_run_width:
        raise RuntimeError(
            "selected-space row is too narrow for the existing safety inset: "
            f"width={run_width}px, required={minimum_run_width}px"
        )

    target_x = int(np.clip(run_left + safe_inset, 0, width - 1))
    if not selected_mask[target_y, target_x]:
        raise AssertionError("modified right-side target left selected-space mask")

    info = {
        "physical_mapping": "image_left_equals_real_world_right",
        "original_target_px": [original_x, target_y],
        "modified_target_px": [target_x, target_y],
        "selected_row_run_xy": [run_left, run_right],
        "target_inset_px": safe_inset,
        "left_clearance_px": target_x - run_left,
        "right_clearance_px": run_right - target_x,
        "target_inside_selected_space_mask": True,
    }
    return (target_x, target_y), info


def build_real_world_right_side_target_camera(
    result_metadata: dict[str, Any],
    original_target_cam: Any,
    *,
    inset_px: int = REAL_WORLD_RIGHT_TARGET_INSET_PX,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Move only camera X to the image-left safe target at the same Y/Z.

    Current captures use zero-distortion aligned intrinsics.  Reusing the
    recognized target's camera Y and Z preserves its height/depth while the
    selected pixel changes only horizontally within the existing space mask.
    """
    modified_target_px, info = select_real_world_right_side_target_pixel(
        result_metadata,
        inset_px=inset_px,
    )
    original_cam = np.asarray(original_target_cam, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(original_cam)) or original_cam[2] <= 0.0:
        raise RuntimeError(f"invalid original camera target: {original_cam}")

    shot_dir_value = result_metadata.get("shot_dir")
    if not shot_dir_value:
        raise RuntimeError("shot_dir is missing; camera intrinsics unavailable")
    intrinsics_path = Path(shot_dir_value) / "camera_intrinsics.json"
    try:
        intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"failed to load camera intrinsics: {intrinsics_path}: {exc}"
        ) from exc

    fx = _finite_float(intrinsics.get("fx"))
    ppx = _finite_float(intrinsics.get("ppx"))
    coeffs = intrinsics.get("coeffs", [])
    if fx is None or fx <= 0.0 or ppx is None:
        raise RuntimeError(f"invalid camera intrinsics: {intrinsics_path}")
    if coeffs and any(abs(float(value)) > 1.0e-9 for value in coeffs):
        raise RuntimeError(
            "right-side target horizontal shift currently requires the "
            "zero-distortion aligned intrinsics used by saved captures"
        )

    modified_cam = original_cam.copy()
    modified_cam[0] = (
        (float(modified_target_px[0]) - ppx) / fx * original_cam[2]
    )
    if not np.all(np.isfinite(modified_cam)):
        raise RuntimeError(f"invalid modified camera target: {modified_cam}")

    info.update(
        {
            "camera_target_method": "horizontal_pixel_shift_same_camera_y_z",
            "camera_intrinsics_path": str(intrinsics_path),
            "original_target_cam_m": original_cam.tolist(),
            "modified_target_cam_m": modified_cam.tolist(),
        }
    )
    return modified_cam, info


def build_left_book_high_reach_target_robot(
    right_side_target_robot_mm: Any,
    *,
    z_offset_mm: float | None = LEFT_BOOK_REACH_Z_OFFSET_MM,
) -> tuple[np.ndarray, float]:
    """Raise the right-side-shifted target along robot-base +Z.

    ``moveL_relative`` and the existing storage Z helpers use robot-base +Z
    for the upward/retract direction and -Z for lowering/insertion.  Refuse to
    run until a positive hardware-validated offset has been configured.
    """
    target = np.asarray(
        right_side_target_robot_mm,
        dtype=np.float64,
    ).reshape(3).copy()
    if not np.all(np.isfinite(target)):
        raise RuntimeError(f"invalid right-side robot target: {target}")
    if z_offset_mm is None:
        raise RuntimeError(
            "LEFT_BOOK_REACH_Z_OFFSET_MM is not configured; determine a safe "
            "positive robot +Z offset by hardware clearance testing before "
            "running triangular_right_tilted storage"
        )
    offset = float(z_offset_mm)
    if not math.isfinite(offset) or offset <= 0.0:
        raise RuntimeError(
            "LEFT_BOOK_REACH_Z_OFFSET_MM must be finite and positive: "
            f"{z_offset_mm}"
        )
    high_target = target.copy()
    high_target[2] += offset
    return high_target, offset


def build_right_and_forward_move_delta(
    *,
    real_world_right_robot_y_mm: float,
    shelf_forward_robot_x_mm: float,
) -> list[float]:
    """Build one robot-base MoveL delta for real-right (-Y) and forward (+X)."""
    right_y = float(real_world_right_robot_y_mm)
    forward_x = float(shelf_forward_robot_x_mm)
    if not math.isfinite(right_y) or right_y >= 0.0:
        raise RuntimeError(
            "real-world right correction must be finite robot -Y: "
            f"dy={right_y}"
        )
    if not math.isfinite(forward_x) or forward_x <= 0.0:
        raise RuntimeError(
            "shelf forward correction must be finite robot +X: "
            f"dx={forward_x}"
        )
    return [forward_x, right_y, 0.0, 0.0, 0.0, 0.0]


def classify_storage_space(
    result_metadata: dict[str, Any],
    *,
    vertical_tolerance_deg: float = VERTICAL_ANGLE_TOLERANCE_DEG,
    minimum_width_px: float = MIN_SUPPORTED_SPACE_WIDTH_PX,
) -> StorageSpaceClassification:
    """Classify one selected space using Storage_SAM3 boundary metadata."""
    selected_space_id = result_metadata.get("selected_space_id")
    if selected_space_id is not None:
        selected_space_id = int(selected_space_id)

    left_tilt = _finite_float(
        result_metadata.get("left_tilt_from_vertical_deg")
    )
    right_tilt = _finite_float(
        result_metadata.get("right_tilt_from_vertical_deg")
    )
    image_guide_angle_deg = _finite_float(
        result_metadata.get("angle_deg")
    )
    if image_guide_angle_deg is None:
        angle_rad = _finite_float(result_metadata.get("angle_rad"))
        if angle_rad is not None:
            image_guide_angle_deg = float(np.degrees(angle_rad))

    book_bottom_side_width_px = _finite_float(
        result_metadata.get("book_bottom_side_width_px")
    )
    guide_side = result_metadata.get("guide_side")
    reasons: list[str] = []

    if selected_space_id is None:
        reasons.append("selected_space_id_missing")
    if left_tilt is None:
        reasons.append("left_boundary_tilt_missing_or_nonfinite")
    if right_tilt is None:
        reasons.append("right_boundary_tilt_missing_or_nonfinite")
    if book_bottom_side_width_px is None:
        reasons.append("book_bottom_side_width_missing_or_nonfinite")
    elif book_bottom_side_width_px < float(minimum_width_px):
        reasons.append(
            "book_bottom_side_width_below_minimum:"
            f"{book_bottom_side_width_px:.1f}<{minimum_width_px:.1f}"
        )

    classification = "unsupported"
    tilted_side: str | None = None
    if left_tilt is not None and right_tilt is not None:
        left_is_tilted = left_tilt > float(vertical_tolerance_deg)
        right_is_tilted = right_tilt > float(vertical_tolerance_deg)
        if not left_is_tilted and not right_is_tilted:
            classification = "rectangular"
        elif left_is_tilted and not right_is_tilted:
            classification = "triangular_left_tilted"
            tilted_side = "left"
        elif not left_is_tilted and right_is_tilted:
            classification = "triangular_right_tilted"
            tilted_side = "right"
        else:
            reasons.append("both_boundaries_are_tilted")

    if classification.startswith("triangular_"):
        if image_guide_angle_deg is None:
            reasons.append("image_guide_angle_missing_or_nonfinite")
        if guide_side != tilted_side:
            reasons.append(
                f"guide_side_mismatch:guide={guide_side},tilted={tilted_side}"
            )
        if (
            tilted_side == "left"
            and image_guide_angle_deg is not None
            and image_guide_angle_deg >= 90.0
        ):
            reasons.append("left_tilted_guide_angle_not_below_90_deg")
        if (
            tilted_side == "right"
            and image_guide_angle_deg is not None
            and image_guide_angle_deg <= 90.0
        ):
            reasons.append("right_tilted_guide_angle_not_above_90_deg")

    if reasons:
        classification = "unsupported"
        tilted_side = None

    spacer_angle = determine_spacer_command_angle(
        classification,
        image_guide_angle_deg,
    )
    return StorageSpaceClassification(
        selected_space_id=selected_space_id,
        left_boundary_tilt_deg=left_tilt,
        right_boundary_tilt_deg=right_tilt,
        classification=classification,
        tilted_side=tilted_side,
        image_guide_angle_deg=image_guide_angle_deg,
        xarm_d_roll_rad=UPRIGHT_XARM_D_ROLL_RAD,
        spacer_command_angle_deg=spacer_angle,
        book_bottom_side_width_px=book_bottom_side_width_px,
        reasons=tuple(reasons),
    )


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
    print(f"classification = {decision.classification}")
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


def _is_no_space_error(exc: RuntimeError) -> bool:
    return "収納スペース候補が見つかりませんでした" in str(exc)


def main() -> None:
    # Lazy imports keep pure classification importable without starting or
    # requiring any hardware connection.
    from Dynamixel_win_pro_hand_book.dynamixel_worker_client import (
        DynamixelWorkerClient,
    )
    from detection.pro_handbook.sam_py_demo.Storage_SAM3 import (
        run_capture_and_pca_depth_space,
    )
    from xarm7.control.robot_base_coordinate import cam_mm_to_robot_mm
    from xarm7.control.xarm7 import (
        INSERT_BOOK_FULL_DX,
        TCP_ACC_2,
        TCP_VEL_2,
        XArm7,
    )
    import rclpy

    hand_worker = None
    xarm7 = None
    node = None

    try:
        print("start upright storage sequence")
        print("[DXL] starting Dynamixel worker")
        hand_worker = DynamixelWorkerClient()
        print("[DXL] worker initialized")

        rclpy.init()
        node = rclpy.create_node("upright_storage_node")
        xarm7 = XArm7(node=node, host="192.168.2.197")

        ret = xarm7.moveJ_to_capture_right_strage(asynchronous=False)
        print("[STORAGE] capture pose ret =", ret)
        hand_worker.open_until_full(asynchronous=False)
        time.sleep(5.0)
        hand_worker.grasp()

        capture_tcp_pose_1 = None

        def move_to_second_capture() -> None:
            nonlocal capture_tcp_pose_1
            capture_tcp_pose_1 = xarm7.get_tcp_pose(is_radian=True)
            print("[STORAGE] 1st capture TCP pose =", capture_tcp_pose_1)
            xarm7.moveL_y_offset(
                SECOND_CAPTURE_DY_MM,
                velocity=40,
                acceleration=40,
                asynchronous=True,
            )

        def wait_xarm_motion_done(timeout: float = 10.0) -> None:
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
                if elapsed >= 1.0 and not moving_detected and not moving:
                    return
                if elapsed >= timeout:
                    raise RuntimeError(
                        f"xArm motion timeout: {elapsed:.1f} sec"
                    )
                time.sleep(0.02)

        first_result = None
        first_decision = None
        try:
            print("\n========== STORAGE 1st CAPTURE ==========")
            candidate_result = run_capture_and_pca_depth_space(
                out_dir=STORAGE_CAPTURE_ROOT,
                after_capture_callback=move_to_second_capture,
            )
            _, _, candidate_metadata = candidate_result
            first_decision = classify_storage_space(candidate_metadata)
            log_storage_space_classification(first_decision, "first capture")
            if first_decision.supported:
                first_result = candidate_result
                print("[STORAGE] 1st recognition SUPPORTED")
            else:
                print(
                    "[STORAGE] 1st recognition UNSUPPORTED -> "
                    "retry with second capture"
                )
        except RuntimeError as exc:
            if not _is_no_space_error(exc):
                raise
            print("\n[STORAGE] 1st recognition: NO SPACE -> retry")

        wait_xarm_motion_done()
        print("[STORAGE] arrived at 2nd capture position")

        if first_result is not None:
            if capture_tcp_pose_1 is None:
                raise RuntimeError(
                    "1st recognition succeeded but capture_tcp_pose_1 was not saved"
                )
            angle_rad, first_target_cam, res = first_result
            decision = first_decision
            coordinate_transform_arm = FrozenTcpArm(
                xarm7,
                capture_tcp_pose_1,
            )
            recognition_source = "1st"
            print(
                "[STORAGE] use 1st supported result while staying at "
                "2nd capture position"
            )
        else:
            print("\n========== STORAGE 2nd CAPTURE ==========")
            try:
                angle_rad, first_target_cam, res = (
                    run_capture_and_pca_depth_space(
                        out_dir=STORAGE_CAPTURE_ROOT,
                        after_capture_callback=None,
                    )
                )
            except RuntimeError as exc:
                if _is_no_space_error(exc):
                    raise RuntimeError(
                        "2nd storage capture also found no usable space; "
                        "insertion aborted"
                    ) from exc
                raise
            decision = classify_storage_space(res)
            log_storage_space_classification(decision, "second capture")
            if not decision.supported:
                raise RuntimeError(
                    "2nd storage result is unsupported; insertion aborted: "
                    f"{list(decision.reasons)}"
                )
            coordinate_transform_arm = xarm7
            recognition_source = "2nd"
            print("[STORAGE] 2nd recognition SUPPORTED")

        if decision is None:
            raise RuntimeError("storage classification result is missing")

        motion_target_cam = np.asarray(
            first_target_cam,
            dtype=np.float64,
        ).reshape(3).copy()
        right_side_target_info: dict[str, Any] | None = None
        if decision.classification == "triangular_right_tilted":
            motion_target_cam, right_side_target_info = (
                build_real_world_right_side_target_camera(
                    res,
                    first_target_cam,
                )
            )

        angle_deg = float(np.degrees(angle_rad))
        oblique_line = float(res.get("guide_edge_length_mm", 0.0))
        dy = -oblique_line * np.cos(angle_rad) / 2.0
        left_tilted_correction_delta = None
        if decision.classification == "triangular_left_tilted":
            # Approximate the neighboring book as a rigid body rotating about
            # its nonslipping bottom edge.  The spacer contacts it 70 mm above
            # that pivot; image-left tilted is real-world right in this setup.
            left_tilted_push_dy = (
                -BOOK_PUSH_HEIGHT_MM * np.cos(angle_rad)
            )
            vertical_tilt_angle_deg = 90.0 - angle_deg
            print(
                "[LEFT-TILTED PUSH GEOMETRY] "
                f"push_height={BOOK_PUSH_HEIGHT_MM:.2f} mm, "
                f"angle={angle_deg:.2f} deg, "
                f"vertical_tilt={vertical_tilt_angle_deg:.2f} deg, "
                "calculated real-world-right displacement="
                f"{abs(left_tilted_push_dy):.2f} mm "
                f"(robot dY={left_tilted_push_dy:+.2f} mm)"
            )
            left_tilted_correction_delta = build_right_and_forward_move_delta(
                real_world_right_robot_y_mm=left_tilted_push_dy,
                shelf_forward_robot_x_mm=INSERT_BOOK_FULL_DX,
            )

        print("========== run_capture_and_pca result ==========")
        print(f"[RESULT] angle_rad = {angle_rad:.6f} rad")
        print(f"[RESULT] angle_deg = {angle_deg:.2f} deg")
        print(f"[RESULT] recognition_source = {recognition_source}")
        print(f"[RESULT] pair_indices = {res.get('pair_indices')}")
        print(f"[RESULT] line_p0 = {res.get('line_p0')}")
        print(f"[RESULT] line_p1 = {res.get('line_p1')}")
        print(f"[RESULT] guide_side = {res.get('guide_side')}")
        print("================================================")

        if right_side_target_info is not None:
            print("\n[LEFT-BOOK-TILTED STORAGE]")
            print(f"classification = {decision.classification}")
            print("image tilted side = right boundary")
            print("real-world tilted side = left book")
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
                f"{decision.spacer_command_angle_deg:+.2f} deg (absolute)"
            )
            print(f"xArm d_roll = {decision.xarm_d_roll_rad:+.6f} rad")
            print(
                "forward command = moveL_to_insert_book_full: "
                f"robot dX={INSERT_BOOK_FULL_DX:+.2f} mm, "
                "velocity=15, acceleration=15, asynchronous=False"
            )

        first_target_cam_mm = 1000.0 * np.asarray(
            motion_target_cam,
            dtype=np.float64,
        ).reshape(3)
        first_target_robot_mm = cam_mm_to_robot_mm(
            coordinate_transform_arm,
            first_target_cam_mm,
        )
        left_book_reach_z_offset_mm: float | None = None
        left_book_normal_target_robot_mm: np.ndarray | None = None
        left_book_right_side_target_robot_mm: np.ndarray | None = None
        print("[DEBUG] first_target_cam_mm =", first_target_cam_mm)
        print("[DEBUG] first_target_robot_mm =", first_target_robot_mm)
        if right_side_target_info is not None:
            left_book_normal_target_robot_mm = cam_mm_to_robot_mm(
                coordinate_transform_arm,
                1000.0 * np.asarray(first_target_cam, dtype=np.float64).reshape(3),
            )
            left_book_right_side_target_robot_mm = np.asarray(
                first_target_robot_mm,
                dtype=np.float64,
            ).reshape(3).copy()
            first_target_robot_mm, left_book_reach_z_offset_mm = (
                build_left_book_high_reach_target_robot(
                    left_book_right_side_target_robot_mm,
                )
            )
            print("\n[LEFT-BOOK-TILTED HIGH REACH]")
            print(
                "normal target robot [mm] =",
                left_book_normal_target_robot_mm,
            )
            print(
                "right-side shifted target robot [mm] =",
                left_book_right_side_target_robot_mm,
            )
            print(
                "Z offset [mm] = "
                f"{left_book_reach_z_offset_mm:+.2f} (robot-base +Z/up)"
            )
            print("high reaching target robot [mm] =", first_target_robot_mm)
            print(
                "spacer angle [deg] = "
                f"{decision.spacer_command_angle_deg:+.2f} (absolute)"
            )
            print(
                "insertion step = high reach -> spacer -> "
                "moveL_to_insert_book_full -> lower -> "
                "move_L_to_insert_book_tip -> release"
            )
            print(
                "Z lowering amount [mm] = "
                f"{-left_book_reach_z_offset_mm:+.2f}"
            )
            print(
                "final target Z [mm] = "
                f"{left_book_right_side_target_robot_mm[2]:.2f}"
            )
            print(f"xArm roll command [rad] = {decision.xarm_d_roll_rad:+.6f}")
            print(
                "[LEFT-BOOK-TILTED STORAGE] original target robot [mm] =",
                left_book_normal_target_robot_mm,
            )
            print(
                "[LEFT-BOOK-TILTED STORAGE] modified target robot [mm] =",
                left_book_right_side_target_robot_mm,
            )

        time.sleep(5.0)
        try:
            input("Enter: execute upright reaching / Ctrl+D: retract and return : ")
        except EOFError:
            print("[INFO] Ctrl+D detected before reaching")
            try:
                hand_worker.contract_sp_lin_2(asynchronous=False)
            except Exception as exc:
                print("[WARN] contract_sp_lin_2 failed:", exc)
            try:
                xarm7.moveJ_to_capture_right(asynchronous=False)
            except Exception as exc:
                print("[WARN] moveJ_to_capture_right failed:", exc)
            return

        print(
            f"xArm7 reaches {recognition_source} recognition target "
            "with upright roll"
        )
        print(
            "[STORAGE UPRIGHT] image guide angle is not applied to xArm; "
            "d_roll_rad=0.0"
        )
        ret = xarm7.move_to_storage_target_xyz_and_roll(
            p_robot_mm=first_target_robot_mm,
            d_roll_rad=decision.xarm_d_roll_rad,
            side="right",
        )
        print("[DEBUG] upright reaching ret =", ret)
        if decision.classification == "triangular_right_tilted" and ret != 0:
            raise RuntimeError(f"left-book high reaching failed: ret={ret}")

        try:
            input(
                "After reaching: Enter: correct neighbor and insert / "
                "Ctrl+D: return : "
            )
        except EOFError:
            print("[INFO] Ctrl+D detected after reaching")
            try:
                xarm7.moveJ_to_capture_right(asynchronous=False)
            except Exception as exc:
                print("[WARN] moveJ_to_capture_right failed:", exc)
            return

        triangular = decision.classification.startswith("triangular_")
        if triangular:
            print("[DEBUG] before expand_sp_lin")
            hand_worker.expand_sp_lin(asynchronous=True)
            print("[DEBUG] after expand_sp_lin")
            time.sleep(14.0)
            print(
                "[STORAGE SPACER] correct tilted neighbor: "
                f"side={decision.tilted_side}, "
                f"angle={decision.spacer_command_angle_deg:.2f} deg"
            )
            if decision.classification == "triangular_left_tilted":
                print(
                    "[STORAGE SIDE MAPPING] image-left tilted boundary = "
                    "real-world right tilted book"
                )
            elif decision.classification == "triangular_right_tilted":
                print(
                    "[STORAGE SIDE MAPPING] image-right tilted boundary = "
                    "real-world left tilted book"
                )
            hand_worker.rotate_spacer(decision.spacer_command_angle_deg)
            print(
                "[STORAGE SPACER] do not issue reset while inserting; "
                "the existing rotate_spacer torque policy is unchanged"
            )
        else:
            print(
                "[STORAGE SPACER] rectangular space: "
                "skip spacer expansion/rotation"
            )

        spacer_rotation_reset = False
        if decision.classification == "triangular_left_tilted":
            if left_tilted_correction_delta is None:
                raise RuntimeError("left-tilted correction delta is missing")
            print(
                "[STORAGE UPRIGHT CORRECTION] simultaneous robot-base MoveL: "
                f"dX(forward)={left_tilted_correction_delta[0]:+.2f} mm, "
                f"dY(real-world right)="
                f"{left_tilted_correction_delta[1]:+.2f} mm"
            )
            ret = xarm7.moveL_relative(
                left_tilted_correction_delta,
                velocity=TCP_VEL_2,
                acceleration=TCP_ACC_2,
                asynchronous=False,
            )
            reset_spacer_rotation_before_linear_retraction(hand_worker)
            spacer_rotation_reset = True
            print(
                "[STORAGE SPACER] contract_sp_lin_1 start after rotation reset "
                "(existing asynchronous SP_LIN_KEEP command)"
            )
            hand_worker.contract_sp_lin_1(asynchronous=True)
        elif decision.classification == "triangular_right_tilted":
            if (
                left_book_reach_z_offset_mm is None
                or left_book_right_side_target_robot_mm is None
            ):
                raise RuntimeError(
                    "left-book high-reach target or Z offset is missing"
                )
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
                asynchronous=False,
            )
            print("[LEFT-BOOK-TILTED LATERAL PUSH] MoveL ret =", lateral_ret)
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
            if lateral_xarm_state in (4, 5) or lateral_xarm_error != 0:
                raise RuntimeError(
                    "xArm abnormal after left-book lateral push: "
                    f"state={lateral_xarm_state}, "
                    f"error={lateral_xarm_error}, "
                    f"warn={lateral_xarm_warn}"
                )
            print(
                "[LEFT-BOOK-TILTED LATERAL PUSH] completed; "
                "start forward insertion"
            )
            print(
                "[LEFT-BOOK-TILTED HIGH REACH] step 1/2: advance the upright "
                "book while the spacer opens the space at the high position"
            )
            print(
                "[LEFT-BOOK-TILTED HIGH REACH] forward command: "
                "moveL_to_insert_book_full, "
                f"robot dX={INSERT_BOOK_FULL_DX:+.2f} mm"
            )
            forward_ret = xarm7.moveL_to_insert_book_full(
                velocity=15,
                acceleration=15,
                asynchronous=False,
            )
            print(
                "[LEFT-BOOK-TILTED HIGH REACH] high-position forward ret =",
                forward_ret,
            )
            if forward_ret != 0:
                raise RuntimeError(
                    "left-book high-position forward insertion failed: "
                    f"ret={forward_ret}"
                )

            print(
                "[LEFT-BOOK-TILTED HIGH REACH] step 2/2: lower the grasped "
                "book to the normal target height"
            )
            lowering_ret = xarm7.moveL_z_offset(
                z_offset=-left_book_reach_z_offset_mm,
                asynchronous=False,
            )
            print(
                "[LEFT-BOOK-TILTED HIGH REACH] lowering command: "
                f"robot dZ={-left_book_reach_z_offset_mm:+.2f} mm, "
                "final target Z="
                f"{left_book_right_side_target_robot_mm[2]:.2f} mm, "
                f"ret={lowering_ret}"
            )
            if lowering_ret != 0:
                raise RuntimeError(
                    f"left-book lowering failed: ret={lowering_ret}"
                )
            ret = forward_ret
        else:
            ret = xarm7.moveL_to_insert_book_full(asynchronous=False)
        print("[DEBUG] full upright insertion ret =", ret)

        time.sleep(2.0)
        if triangular and not spacer_rotation_reset:
            reset_spacer_rotation_before_linear_retraction(hand_worker)
            spacer_rotation_reset = True
        print("[STORAGE SPACER] contract_sp_lin_1 start")
        hand_worker.contract_sp_lin_1(asynchronous=False)
        print("[DEBUG] after contract_sp_lin_1")

        ret = xarm7.move_L_to_insert_book_tip(
            velocity=15,
            acceleration=15,
            asynchronous=False,
        )
        print("[DEBUG] move_L_to_insert_book_tip ret =", ret)
        if ret != 0:
            raise RuntimeError(f"move_L_to_insert_book_tip failed: ret={ret}")

        time.sleep(0.5)
        if triangular and not spacer_rotation_reset:
            raise RuntimeError(
                "spacer linear retraction completed without rotation reset"
            )
        print("[STORAGE SPACER] contract_sp_lin_2 start")
        hand_worker.contract_sp_lin_2(asynchronous=False)
        if decision.classification == "triangular_right_tilted":
            print(
                "[LEFT-BOOK-TILTED HIGH REACH] final insertion completed at "
                "normal target height; release the book now"
            )
        print("[STORAGE RELEASE] open gripper until full")
        hand_worker.open_until_full(asynchronous=False)
        print("[STORAGE RELEASE] gripper open command completed")
        ret = xarm7.moveL_to_post_storage(asynchronous=True)
        print("[DEBUG] moveL_to_post_storage ret =", ret)
        time.sleep(4.0)
        ret = xarm7.moveJ_to_capture_right(asynchronous=False)
        print("[DEBUG] moveJ_to_capture_right ret =", ret)
        hand_worker.grasp()
        print("sequence done")

    except KeyboardInterrupt:
        print("\nCtrl+C detected")
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
        try:
            if xarm7 is not None:
                xarm7.emergency_stop()
        except Exception as stop_exc:
            print("[WARN] xArm emergency_stop failed:", stop_exc)
        raise

    finally:
        try:
            if hand_worker is not None:
                hand_worker.close()
        except Exception as exc:
            print("[WARN] Dynamixel worker close failed:", exc)
        time.sleep(0.2)
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


if __name__ == "__main__":
    main()
