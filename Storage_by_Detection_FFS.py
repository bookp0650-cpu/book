"""FFS-corrected storage experiment with the 0911 insertion sequence.

This hardware entry point is derived from ``Storage_by_Detection0911.py``.
Importing this module is hardware-free; hardware-dependent imports and object
construction are intentionally confined to :func:`main`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import os
import threading
import time

import numpy as np


STORAGE_CAPTURE_ROOT = Path(
    "/home/book/pro_book_SAM3/pro_hand_book_python/captures/strage"
)

# Default experiment mode: no Enter/Ctrl+D confirmation and no recognition
# GUI.  Set False only when the existing manual checkpoints are required for
# debugging; file-based recognition/debug outputs are saved in either mode.
AUTO_RUN = True

_REQUESTED_STORAGE_TARGET_3D_MODE = os.environ.get(
    "STORAGE_TARGET_3D_MODE",
    "depth_ref",
).strip().lower()
if _REQUESTED_STORAGE_TARGET_3D_MODE != "depth_ref":
    raise RuntimeError(
        "Storage_by_Detection_FFS.py requires "
        "STORAGE_TARGET_3D_MODE=depth_ref so FFS corrected depth is always "
        f"used, got {_REQUESTED_STORAGE_TARGET_3D_MODE!r}"
    )
STORAGE_TARGET_3D_MODE = "depth_ref"

# Storage_rev/Storage_SAM3 has no classification threshold.  The value below
# separates the current saved upright-boundary group (about 1.1--7.1 deg)
# from the clearly tilted group (about 10.3--19.5 deg).  It must be confirmed
# again on hardware before unattended operation.
VERTICAL_ANGLE_TOLERANCE_DEG = 10.0
BOUNDARY_TILT_DIFFERENCE_THRESHOLD_DEG = 5.0
AMBIGUOUS_TILT_EQUAL_EPSILON_DEG = 1.0e-9

# This is the same current threshold as MIN_STORAGE_SPACE_WIDTH_PX in
# offline_100test_storage_space_sam3.py.  It is repeated here only as a safety
# check on the recognition result; the recognition core remains authoritative.
MIN_SUPPORTED_SPACE_WIDTH_PX = 30.0

# move_to_storage_target_xyz_and_roll() adds this value to the current TCP
# roll.  The capture pose and all pre-reaching Y moves preserve RPY, therefore
# zero keeps the grasped book at that existing upright reference.
UPRIGHT_XARM_D_ROLL_RAD = 0.0

SECOND_CAPTURE_DY_MM = -60.0
STORAGE_CAPTURE_Z_OFFSET_MM = -90.0
BOOK_PUSH_HEIGHT_MM = 70.0

# 0911 post-recognition depth plan.  The recognized surface is robot X=0 in
# the relative explanation below.  Reach places the TCP at -75 mm; the two
# and only two insertion moves then add +50 mm and +100 mm.  The grasped
# book's rear face is +65 mm from the TCP, so it finishes at +140 mm:
#
#   TCP       : -75 -> -25 -> +75 mm
#   book rear : -10 -> +40 -> +140 mm
TCP_TO_BOOK_REAR_MM = 65.0
REACH_MARGIN_MM = 10.0
INSERT_WITH_SPACER_MM = 50.0
FINAL_BOOK_PUSH_MM = 100.0
REACH_TCP_BACKOFF_MM = TCP_TO_BOOK_REAR_MM + REACH_MARGIN_MM
EXPECTED_FINAL_TCP_OFFSET_MM = (
    -REACH_TCP_BACKOFF_MM
    + INSERT_WITH_SPACER_MM
    + FINAL_BOOK_PUSH_MM
)
EXPECTED_INSERT_DEPTH_MM = (
    -TCP_TO_BOOK_REAR_MM
    - REACH_MARGIN_MM
    + INSERT_WITH_SPACER_MM
    + FINAL_BOOK_PUSH_MM
    + TCP_TO_BOOK_REAR_MM
)

LEFT_BOOK_LATERAL_SHIFT_MM = 5.0
# Only for image-right boundary tilt = real-world LEFT tilted book.  In the
# established robot-base convention, real-world RIGHT is -Y and shelf backoff
# is -X.  These offsets modify only the initial pre-spacer reach target.
LEFT_TILTED_REACH_RIGHT_OFFSET_MM = 5.0
LEFT_TILTED_REACH_BACKOFF_MM = 10.0

# Experimental held-book and spacer geometry for this standalone test.
TEST_BOOK_WIDTH_MM = 15.0
STORAGE_CLEARANCE_MM = 1.0
# Empirical linear opening calibration for triangular spaces.  This is an
# additional millimetre displacement after contact, not an angle and not the
# separate hand-geometry reach offset below.
TRIANGULAR_EXTRA_OPENING_MM = 10.0
SPACER_EFFECTIVE_LENGTH_MM = 50.0
SPACER_PRE_ROTATION_FORWARD_MM = 30.0
RIGHT_TILTED_SYNC_STEP_DEG = 5.0
SPACER_ROTATION_START_DELAY_SEC = 0.3
SPACER_ROTATION_RESET_ADVANCE_SEC = 0.3
SPACER_ROTATION_RESET_BASE_WAIT_SEC = 2.0
SPACER_ROTATION_RESET_WAIT_SEC = (
    SPACER_ROTATION_RESET_BASE_WAIT_SEC
    - SPACER_ROTATION_RESET_ADVANCE_SEC
)
STORAGE_SPACER_FINAL_TIP_ADDITIONAL_MM = 33.5
STORAGE_TILTED_LOWER_ADDITIONAL_MM = 49.5
ONE_SHOT_SP_LIN_BACK_TIMEOUT_SEC = 20.0
BOOK_HEIGHT_MM = 210.0
# Both tilted-book paths use the same effective contact height: the existing
# 40 mm normal spacer height plus the existing 60 mm high-reach offset.
SPACER_CONTACT_HEIGHT_MM = 100.0
RIGHT_TILTED_SPACER_CONTACT_HEIGHT_MM = SPACER_CONTACT_HEIGHT_MM
LEFT_TILTED_SPACER_CONTACT_HEIGHT_MM = SPACER_CONTACT_HEIGHT_MM
RECTANGULAR_BOTTLENECK_PERCENTILE = 5.0
# Treat sub-millimetre physical-width differences as effectively tied before
# comparing mask coverage.  The primary quality metric remains the existing
# RANSAC-plane physical bottleneck, not a pixel width.
TWO_VIEW_BOTTLENECK_TIE_TOLERANCE_MM = 0.5

# This experiment isolates spacer-angle control.  Keep the established xArm
# lateral corrections available for comparison, but disable them by default.
ENABLE_XARM_LATERAL_CORRECTION = False

# Match the existing boundary-fit minimum in
# detection/pro_handbook/sam_py_demo/config/shelf_storage_detection_sam3.json.
CONTACT_OPENING_MIN_VALID_ROWS = 20

# Preserve the established LEFT-book active direction as the source of truth;
# RIGHT-book tilt intentionally uses the same positive SP_ROT direction.
RIGHT_TILTED_SPACER_DIRECTION_SIGN = 1.0
LEFT_TILTED_SPACER_DIRECTION_SIGN = RIGHT_TILTED_SPACER_DIRECTION_SIGN

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

if SPACER_ROTATION_RESET_WAIT_SEC < 0.0:
    raise RuntimeError(
        "SPACER_ROTATION_RESET_WAIT_SEC must be nonnegative: "
        f"{SPACER_ROTATION_RESET_WAIT_SEC}"
    )


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
    tilt_difference_deg: float | None = None
    difference_threshold_deg: float = BOUNDARY_TILT_DIFFERENCE_THRESHOLD_DEG
    classification_source: str = "normal"
    ambiguous_bottleneck_opening_mm: float | None = None
    ambiguous_bottleneck_valid_row_count: int | None = None
    fallback_selected_by_larger_boundary: str | None = None

    @property
    def supported(self) -> bool:
        return self.classification != "unsupported"


@dataclass(frozen=True)
class RightTiltedSupportPlan:
    boundary_tilt_deg: float
    contact_height_mm: float
    arm_right_total_mm: float
    max_space_width_mm: float
    spacer_effective_length_mm: float
    spacer_angle_deg: float
    high_reach_offset_mm: float


@dataclass(frozen=True)
class TriangularSpacerGeometry:
    classification: str
    real_world_tilted_side: str
    book_width_mm: float
    clearance_mm: float
    base_required_opening_mm: float
    triangular_extra_opening_mm: float
    calibrated_required_opening_mm: float
    book_height_mm: float
    spacer_contact_height_mm: float
    boundary_tilt_deg: float
    spacer_axis_right_of_held_book_right_edge_mm: float
    geometry_offset_angle_deg: float
    contact_angle_deg: float
    push_displacement_mm: float
    spacer_effective_length_mm: float
    maximum_remaining_horizontal_capacity_mm: float
    asin_argument: float
    command_angle_magnitude_deg: float | None
    signed_command_angle_deg: float | None
    feasible: bool
    reason: str
    right_tilted_support_plan: RightTiltedSupportPlan | None = None


@dataclass(frozen=True)
class TriangularForwardPlan:
    original_total_forward_mm: float
    pre_rotation_forward_mm: float
    remaining_forward_mm: float


@dataclass(frozen=True)
class SpacerAngleFromMaxWidth:
    max_space_width_mm: float
    effective_length_mm: float
    raw_ratio: float
    angle_magnitude_deg: float
    saturated_at_90_deg: bool


@dataclass(frozen=True)
class RightTiltedRotationStep:
    angle_prev_deg: float
    angle_next_deg: float
    spacer_command_angle_deg: float
    rotation_progress: float
    arm_cumulative_right_mm: float
    robot_cumulative_y_mm: float
    robot_delta_y_mm: float


@dataclass(frozen=True)
class AmbiguousStorageResolution:
    classification: str
    tilted_side: str | None
    selected_by_larger_boundary: str | None
    reason: str


@dataclass(frozen=True)
class StorageSpacerPlan:
    classification: str
    book_width_mm: float
    clearance_mm: float
    required_opening_mm: float
    bottleneck_opening_mm: float | None
    required_additional_opening_mm: float | None
    spacer_required: bool
    spacer_angle_magnitude_deg: float | None
    spacer_command_angle_deg: float | None
    feasible: bool
    decision: str
    reason: str
    bottleneck_percentile: float | None = None
    valid_row_count: int | None = None
    current_opening_used_for_angle_calculation: bool = False
    triangular_geometry: TriangularSpacerGeometry | None = None


@dataclass(frozen=True)
class StorageCaptureCandidate:
    """One capture and all pure recognition/planning results derived from it."""

    label: str
    capture_index: int
    result: tuple[float, np.ndarray, dict[str, Any]] | None = None
    capture_tcp_pose: tuple[float, ...] | None = None
    shot_dir: str | None = None
    recognition_error: str | None = None
    decision: StorageSpaceClassification | None = None
    spacer_plan: StorageSpacerPlan | None = None
    rectangular_bottleneck_info: dict[str, Any] | None = None
    selected_space_width_info: dict[str, Any] | None = None
    diagnostic_current_opening_mm: float | None = None
    physical_bottleneck_mm: float | None = None
    selected_space_height_px: int = 0
    valid_physical_row_count: int = 0
    physical_vertical_width_check: dict[str, Any] | None = None
    planning_error: str | None = None

    @property
    def recognition_success(self) -> bool:
        return self.result is not None and self.recognition_error is None

    @property
    def supported(self) -> bool:
        return self.decision is not None and self.decision.supported

    @property
    def plan_feasible(self) -> bool:
        return self.spacer_plan is not None and self.spacer_plan.feasible

    @property
    def usable(self) -> bool:
        return (
            self.recognition_success
            and self.supported
            and self.plan_feasible
            and self.capture_tcp_pose is not None
            and self.physical_bottleneck_mm is not None
        )

    @property
    def failure_reason(self) -> str:
        if self.recognition_error is not None:
            return f"recognition/geometry failure: {self.recognition_error}"
        if self.result is None:
            return "recognition result is missing"
        if self.planning_error is not None:
            return f"classification/planning failure: {self.planning_error}"
        if self.decision is None:
            return "classification result is missing"
        if not self.decision.supported:
            return (
                f"unsupported classification: {self.decision.classification}; "
                f"reasons={list(self.decision.reasons)}"
            )
        if self.spacer_plan is None:
            return "spacer plan is missing"
        if not self.spacer_plan.feasible:
            return (
                f"spacer plan infeasible: decision={self.spacer_plan.decision}; "
                f"reason={self.spacer_plan.reason}"
            )
        if self.capture_tcp_pose is None:
            return "capture-time TCP pose is missing"
        if self.physical_bottleneck_mm is None:
            return "physical bottleneck is missing"
        return "candidate is not usable"


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


def calculate_storage_final_tip_forward_mm(
    original_forward_mm: float,
    *,
    spacer_required: bool,
) -> float:
    """Add the configured distance only after spacer use and retraction."""
    original = _finite_float(original_forward_mm)
    if original is None or original <= 0.0:
        raise RuntimeError(
            "original final tip forward must be finite and positive: "
            f"{original_forward_mm}"
        )
    if spacer_required:
        return original + STORAGE_SPACER_FINAL_TIP_ADDITIONAL_MM
    return original


def should_use_right_tilted_sp_lin_partial_expansion(
    classification: str,
) -> bool:
    """Use partial SP_LIN only for image-left/real-world-right tilt."""
    return classification == "triangular_left_tilted"


def should_use_right_tilted_support_sync(classification: str) -> bool:
    """Enable G_max SP_ROT plus alpha-based xArm sync for RIGHT-book tilt."""
    return classification == "triangular_left_tilted"


def plan_triangular_forward_split(
    total_forward_mm: float,
    *,
    pre_rotation_forward_mm: float = SPACER_PRE_ROTATION_FORWARD_MM,
) -> TriangularForwardPlan:
    """Split one existing +X insertion into pre/post-rotation MoveL steps."""
    total = _finite_float(total_forward_mm)
    pre_rotation = _finite_float(pre_rotation_forward_mm)
    if total is None or total <= 0.0:
        raise RuntimeError(
            f"total forward must be finite and positive: {total_forward_mm}"
        )
    if pre_rotation is None or pre_rotation <= 0.0:
        raise RuntimeError(
            "pre-rotation forward must be finite and positive: "
            f"{pre_rotation_forward_mm}"
        )
    remaining = total - pre_rotation
    if remaining <= 0.0:
        raise RuntimeError(
            "remaining forward after spacer rotation must be positive: "
            f"total={total:.2f} mm, pre_rotation={pre_rotation:.2f} mm, "
            f"remaining={remaining:.2f} mm"
        )
    return TriangularForwardPlan(
        original_total_forward_mm=total,
        pre_rotation_forward_mm=pre_rotation,
        remaining_forward_mm=remaining,
    )


def calculate_spacer_angle_deg(
    required_additional_opening_mm: float,
    *,
    effective_length_mm: float = SPACER_EFFECTIVE_LENGTH_MM,
) -> float:
    """Return the legacy direct-displacement angle used by rectangular plans."""
    additional = _finite_float(required_additional_opening_mm)
    if additional is None or additional < 0.0:
        raise RuntimeError(
            "required additional opening must be finite and nonnegative: "
            f"{required_additional_opening_mm}"
        )
    length = _finite_float(effective_length_mm)
    if length is None or length <= 0.0:
        raise RuntimeError(
            f"spacer effective length must be finite and positive: "
            f"{effective_length_mm}"
        )
    if additional > length:
        raise RuntimeError(
            "required additional opening exceeds spacer effective length: "
            f"{additional:.6f}>{length:.6f} mm"
        )
    ratio = additional / length
    return float(math.degrees(math.asin(ratio)))


def calculate_spacer_angle_from_gmax(
    max_space_width_mm: float,
    effective_length_mm: float = SPACER_EFFECTIVE_LENGTH_MM,
) -> SpacerAngleFromMaxWidth:
    """Convert the widest physical selected-space row to a spacer angle."""
    max_width = _finite_float(max_space_width_mm)
    length = _finite_float(effective_length_mm)
    if max_width is None or max_width < 0.0:
        raise RuntimeError(
            "maximum selected-space width must be finite and nonnegative: "
            f"{max_space_width_mm}"
        )
    if length is None or length <= 0.0:
        raise RuntimeError(
            "spacer effective length must be finite and positive: "
            f"{effective_length_mm}"
        )

    raw_ratio = max_width / length
    saturated = max_width >= length
    if saturated:
        angle_magnitude_deg = 90.0
    else:
        angle_magnitude_deg = math.degrees(math.asin(raw_ratio))
    return SpacerAngleFromMaxWidth(
        max_space_width_mm=max_width,
        effective_length_mm=length,
        raw_ratio=raw_ratio,
        angle_magnitude_deg=float(angle_magnitude_deg),
        saturated_at_90_deg=saturated,
    )


def spacer_lateral_projection_mm(
    angle_deg: float,
    length_mm: float = SPACER_EFFECTIVE_LENGTH_MM,
) -> float:
    """Return the lateral projection ``L*sin(abs(theta))`` in millimetres."""
    angle = _finite_float(angle_deg)
    length = _finite_float(length_mm)
    if angle is None or not 0.0 <= abs(angle) <= 90.0:
        raise RuntimeError(
            f"spacer angle must be finite and within [-90, 90]: {angle_deg}"
        )
    if length is None or length <= 0.0:
        raise RuntimeError(
            f"spacer effective length must be finite and positive: {length_mm}"
        )
    return float(length * math.sin(math.radians(abs(angle))))


def calculate_right_tilted_support_plan(
    boundary_tilt_deg: float,
    max_space_width_mm: float,
    *,
    contact_height_mm: float = RIGHT_TILTED_SPACER_CONTACT_HEIGHT_MM,
    spacer_effective_length_mm: float = SPACER_EFFECTIVE_LENGTH_MM,
    high_reach_offset_mm: float | None = LEFT_BOOK_REACH_Z_OFFSET_MM,
) -> RightTiltedSupportPlan:
    """Plan independent RIGHT-book xArm travel and G_max SP_ROT angle."""
    boundary_tilt = _finite_float(boundary_tilt_deg)
    max_width = _finite_float(max_space_width_mm)
    contact_height = _finite_float(contact_height_mm)
    spacer_length = _finite_float(spacer_effective_length_mm)
    high_reach_offset = _finite_float(high_reach_offset_mm)
    if boundary_tilt is None or not 0.0 <= abs(boundary_tilt) < 90.0:
        raise RuntimeError(
            "RIGHT-book boundary tilt must be finite and within (-90, 90): "
            f"{boundary_tilt_deg}"
        )
    if contact_height is None or contact_height <= 0.0:
        raise RuntimeError(
            "RIGHT-book spacer contact height must be finite and positive: "
            f"{contact_height_mm}"
        )
    if spacer_length is None or spacer_length <= 0.0:
        raise RuntimeError(
            "spacer effective length must be finite and positive: "
            f"{spacer_effective_length_mm}"
        )
    if max_width is None or max_width < 0.0:
        raise RuntimeError(
            "RIGHT-book G_max must be finite and nonnegative: "
            f"{max_space_width_mm}"
        )
    if high_reach_offset is None or high_reach_offset <= 0.0:
        raise RuntimeError(
            "RIGHT-book high-reach offset must be finite and positive: "
            f"{high_reach_offset_mm}"
        )

    arm_right_total = contact_height * math.tan(
        math.radians(abs(boundary_tilt))
    )
    spacer_angle = calculate_spacer_angle_from_gmax(
        max_width,
        effective_length_mm=spacer_length,
    )
    return RightTiltedSupportPlan(
        boundary_tilt_deg=float(boundary_tilt),
        contact_height_mm=float(contact_height),
        arm_right_total_mm=float(arm_right_total),
        max_space_width_mm=float(max_width),
        spacer_effective_length_mm=float(spacer_length),
        spacer_angle_deg=float(spacer_angle.angle_magnitude_deg),
        high_reach_offset_mm=float(high_reach_offset),
    )


def plan_right_tilted_rotation_steps(
    final_angle_magnitude_deg: float,
    arm_right_total_mm: float,
    *,
    step_deg: float = RIGHT_TILTED_SYNC_STEP_DEG,
) -> tuple[RightTiltedRotationStep, ...]:
    """Split SP_ROT and an independent h*tan(alpha) arm path into steps."""
    final_angle = _finite_float(final_angle_magnitude_deg)
    arm_right_total = _finite_float(arm_right_total_mm)
    step = _finite_float(step_deg)
    if final_angle is None or not 0.0 <= final_angle <= 90.0:
        raise RuntimeError(
            "final spacer angle magnitude must be within [0, 90]: "
            f"{final_angle_magnitude_deg}"
        )
    if step is None or step <= 0.0:
        raise RuntimeError(f"sync step must be finite and positive: {step_deg}")
    if arm_right_total is None or arm_right_total < 0.0:
        raise RuntimeError(
            "arm RIGHT total must be finite and nonnegative: "
            f"{arm_right_total_mm}"
        )
    if final_angle <= 1.0e-12:
        if arm_right_total > 1.0e-9:
            raise RuntimeError(
                "nonzero RIGHT arm travel cannot be synchronized with a "
                "zero SP_ROT angle"
            )
        return ()

    planned_steps: list[RightTiltedRotationStep] = []
    angle_prev = 0.0
    previous_cumulative_y = 0.0
    while angle_prev + 1.0e-12 < final_angle:
        angle_next = min(final_angle, angle_prev + step)
        progress = angle_next / final_angle
        cumulative_right = arm_right_total * progress
        cumulative_y = -cumulative_right
        planned_steps.append(
            RightTiltedRotationStep(
                angle_prev_deg=angle_prev,
                angle_next_deg=angle_next,
                spacer_command_angle_deg=angle_next,
                rotation_progress=progress,
                arm_cumulative_right_mm=cumulative_right,
                robot_cumulative_y_mm=cumulative_y,
                robot_delta_y_mm=cumulative_y - previous_cumulative_y,
            )
        )
        angle_prev = angle_next
        previous_cumulative_y = cumulative_y
    if not math.isclose(
        sum(item.robot_delta_y_mm for item in planned_steps),
        -arm_right_total,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise AssertionError("RIGHT tilted stepped robot Y does not sum to total")
    return tuple(planned_steps)


def calculate_triangular_spacer_geometry(
    *,
    classification: str,
    book_width_mm: float,
    clearance_mm: float,
    boundary_tilt_deg: float,
    triangular_extra_opening_mm: float = TRIANGULAR_EXTRA_OPENING_MM,
    book_height_mm: float = BOOK_HEIGHT_MM,
    spacer_effective_length_mm: float = SPACER_EFFECTIVE_LENGTH_MM,
    right_tilted_contact_height_mm: float = (
        RIGHT_TILTED_SPACER_CONTACT_HEIGHT_MM
    ),
    left_tilted_contact_height_mm: float = (
        LEFT_TILTED_SPACER_CONTACT_HEIGHT_MM
    ),
    triangular_max_space_width_mm: float | None = None,
) -> TriangularSpacerGeometry:
    """Plan a triangular-space command without accessing any hardware.

    Both tilted sides use the established physical G_max SP_ROT model.
    Image-left/real-world-RIGHT tilt additionally uses its boundary tilt and
    100 mm contact height to plan the synchronized xArm -Y travel.
    """
    width = _finite_float(book_width_mm)
    clearance = _finite_float(clearance_mm)
    opening_calibration = _finite_float(triangular_extra_opening_mm)
    boundary_tilt = _finite_float(boundary_tilt_deg)
    book_height = _finite_float(book_height_mm)
    spacer_length = _finite_float(spacer_effective_length_mm)
    right_contact_height = _finite_float(right_tilted_contact_height_mm)
    left_contact_height = _finite_float(left_tilted_contact_height_mm)
    if width is None or width <= 0.0:
        raise RuntimeError(f"book width must be finite and positive: {book_width_mm}")
    if clearance is None or clearance < 0.0:
        raise RuntimeError(
            f"storage clearance must be finite and nonnegative: {clearance_mm}"
        )
    if opening_calibration is None or opening_calibration < 0.0:
        raise RuntimeError(
            "triangular extra opening must be finite and nonnegative: "
            f"{triangular_extra_opening_mm}"
        )
    if (
        boundary_tilt is None
        or boundary_tilt < 0.0
        or boundary_tilt > 90.0
    ):
        raise RuntimeError(
            "boundary tilt must be finite and within [0, 90] degrees: "
            f"{boundary_tilt_deg}"
        )
    if book_height is None or book_height <= 0.0:
        raise RuntimeError(
            f"book height must be finite and positive: {book_height_mm}"
        )
    if spacer_length is None or spacer_length <= 0.0:
        raise RuntimeError(
            "spacer effective length must be finite and positive: "
            f"{spacer_effective_length_mm}"
        )
    if right_contact_height is None or right_contact_height <= 0.0:
        raise RuntimeError(
            "right-tilted spacer contact height must be finite and positive: "
            f"{right_tilted_contact_height_mm}"
        )
    if left_contact_height is None or left_contact_height <= 0.0:
        raise RuntimeError(
            "left-tilted spacer contact height must be finite and positive: "
            f"{left_tilted_contact_height_mm}"
        )
    if classification not in {
        "triangular_left_tilted",
        "triangular_right_tilted",
    }:
        raise RuntimeError(
            "triangular spacer geometry requested for unsupported "
            f"classification: {classification}"
        )

    if classification == "triangular_left_tilted":
        support = calculate_right_tilted_support_plan(
            boundary_tilt,
            triangular_max_space_width_mm,
            contact_height_mm=right_contact_height,
            spacer_effective_length_mm=spacer_length,
            high_reach_offset_mm=LEFT_BOOK_REACH_Z_OFFSET_MM,
        )
        signed_command = determine_spacer_command_angle(
            classification,
            support.spacer_angle_deg,
        )
        return TriangularSpacerGeometry(
            classification=classification,
            real_world_tilted_side="right",
            book_width_mm=width,
            clearance_mm=clearance,
            base_required_opening_mm=support.max_space_width_mm,
            triangular_extra_opening_mm=0.0,
            calibrated_required_opening_mm=(
                support.max_space_width_mm
            ),
            book_height_mm=book_height,
            spacer_contact_height_mm=right_contact_height,
            boundary_tilt_deg=boundary_tilt,
            spacer_axis_right_of_held_book_right_edge_mm=0.0,
            geometry_offset_angle_deg=0.0,
            contact_angle_deg=boundary_tilt,
            push_displacement_mm=support.max_space_width_mm,
            spacer_effective_length_mm=spacer_length,
            maximum_remaining_horizontal_capacity_mm=spacer_length,
            asin_argument=(
                support.max_space_width_mm / spacer_length
            ),
            command_angle_magnitude_deg=support.spacer_angle_deg,
            signed_command_angle_deg=signed_command,
            feasible=True,
            reason=(
                "right-book support model: SP_ROT uses G_max; xArm total "
                "RIGHT travel independently uses h*tan(abs(alpha))"
            ),
            right_tilted_support_plan=support,
        )

    # Keep the real-world LEFT-book path on its established G_max model.
    angle = calculate_spacer_angle_from_gmax(
        triangular_max_space_width_mm,
        effective_length_mm=spacer_length,
    )
    signed_command = determine_spacer_command_angle(
        classification,
        angle.angle_magnitude_deg,
    )
    real_world_tilted_side = "left"
    contact_height = left_contact_height
    return TriangularSpacerGeometry(
        classification=classification,
        real_world_tilted_side=real_world_tilted_side,
        book_width_mm=width,
        clearance_mm=clearance,
        base_required_opening_mm=angle.max_space_width_mm,
        triangular_extra_opening_mm=0.0,
        calibrated_required_opening_mm=angle.max_space_width_mm,
        book_height_mm=book_height,
        spacer_contact_height_mm=contact_height,
        boundary_tilt_deg=boundary_tilt,
        spacer_axis_right_of_held_book_right_edge_mm=0.0,
        geometry_offset_angle_deg=0.0,
        contact_angle_deg=0.0,
        push_displacement_mm=angle.max_space_width_mm,
        spacer_effective_length_mm=spacer_length,
        maximum_remaining_horizontal_capacity_mm=spacer_length,
        asin_argument=angle.raw_ratio,
        command_angle_magnitude_deg=angle.angle_magnitude_deg,
        signed_command_angle_deg=signed_command,
        feasible=True,
        reason=(
            f"{real_world_tilted_side}-book max-width model: "
            "L*sin(theta)=G_max"
            + (
                "; explicit 90-degree saturation"
                if angle.saturated_at_90_deg
                else ""
            )
        ),
    )


def plan_storage_spacer(
    *,
    classification: str,
    book_width_mm: float = TEST_BOOK_WIDTH_MM,
    clearance_mm: float = STORAGE_CLEARANCE_MM,
    triangular_boundary_tilt_deg: float | None = None,
    triangular_max_space_width_mm: float | None = None,
    rectangular_bottleneck_mm: float | None = None,
    rectangular_valid_row_count: int | None = None,
    direct_insert_bottleneck_mm: float | None = None,
    direct_insert_valid_row_count: int | None = None,
    effective_length_mm: float = SPACER_EFFECTIVE_LENGTH_MM,
) -> StorageSpacerPlan:
    """Create the standalone experiment's spacer plan without hardware."""
    width = _finite_float(book_width_mm)
    clearance = _finite_float(clearance_mm)
    length = _finite_float(effective_length_mm)
    if width is None or width <= 0.0:
        raise RuntimeError(f"book width must be finite and positive: {book_width_mm}")
    if clearance is None or clearance < 0.0:
        raise RuntimeError(
            f"storage clearance must be finite and nonnegative: {clearance_mm}"
        )
    if length is None or length <= 0.0:
        raise RuntimeError(
            f"spacer effective length must be finite and positive: "
            f"{effective_length_mm}"
        )
    required_opening = width + clearance

    if classification == "direct_insert":
        bottleneck = _finite_float(direct_insert_bottleneck_mm)
        if bottleneck is None or bottleneck < 0.0:
            raise RuntimeError(
                "direct-insert bottleneck must be finite and nonnegative: "
                f"{direct_insert_bottleneck_mm}"
            )
        if (
            direct_insert_valid_row_count is None
            or int(direct_insert_valid_row_count)
            < CONTACT_OPENING_MIN_VALID_ROWS
        ):
            raise RuntimeError(
                "too few valid rows for direct-insert bottleneck plan: "
                f"{direct_insert_valid_row_count}"
                f"<{CONTACT_OPENING_MIN_VALID_ROWS}"
            )
        # The ambiguous fallback intentionally compares against the held-book
        # width only.  STORAGE_CLEARANCE_MM is not part of this direct-insert
        # experiment's acceptance criterion.
        if bottleneck <= width:
            raise RuntimeError(
                "direct insert requires physical bottleneck > held book "
                f"width: {bottleneck:.6f}<={width:.6f} mm"
            )
        return StorageSpacerPlan(
            classification=classification,
            book_width_mm=width,
            clearance_mm=clearance,
            required_opening_mm=width,
            bottleneck_opening_mm=bottleneck,
            required_additional_opening_mm=0.0,
            spacer_required=False,
            spacer_angle_magnitude_deg=None,
            spacer_command_angle_deg=None,
            feasible=True,
            decision="direct_insert_without_spacer",
            reason=(
                "ambiguous boundaries but physical bottleneck exceeds held "
                "book width"
            ),
            bottleneck_percentile=RECTANGULAR_BOTTLENECK_PERCENTILE,
            valid_row_count=int(direct_insert_valid_row_count),
        )

    if classification in {
        "triangular_left_tilted",
        "triangular_right_tilted",
    }:
        geometry = calculate_triangular_spacer_geometry(
            classification=classification,
            book_width_mm=width,
            clearance_mm=clearance,
            boundary_tilt_deg=triangular_boundary_tilt_deg,
            triangular_max_space_width_mm=triangular_max_space_width_mm,
            book_height_mm=BOOK_HEIGHT_MM,
            spacer_effective_length_mm=length,
        )
        return StorageSpacerPlan(
            classification=classification,
            book_width_mm=width,
            clearance_mm=clearance,
            required_opening_mm=(
                geometry.calibrated_required_opening_mm
            ),
            bottleneck_opening_mm=None,
            required_additional_opening_mm=geometry.push_displacement_mm,
            spacer_required=True,
            spacer_angle_magnitude_deg=(
                geometry.command_angle_magnitude_deg
            ),
            spacer_command_angle_deg=geometry.signed_command_angle_deg,
            feasible=geometry.feasible,
            decision=(
                "spacer_required"
                if geometry.feasible
                else "mechanism_capacity_exceeded"
            ),
            reason=geometry.reason,
            triangular_geometry=geometry,
        )

    if classification == "rectangular":
        bottleneck = _finite_float(rectangular_bottleneck_mm)
        if bottleneck is None or bottleneck < 0.0:
            raise RuntimeError(
                "rectangular bottleneck must be finite and nonnegative: "
                f"{rectangular_bottleneck_mm}"
            )
        if (
            rectangular_valid_row_count is None
            or int(rectangular_valid_row_count) < CONTACT_OPENING_MIN_VALID_ROWS
        ):
            raise RuntimeError(
                "too few valid rows for rectangular bottleneck plan: "
                f"{rectangular_valid_row_count}"
                f"<{CONTACT_OPENING_MIN_VALID_ROWS}"
            )
        percentile = RECTANGULAR_BOTTLENECK_PERCENTILE
        valid_row_count = int(rectangular_valid_row_count)
    else:
        return StorageSpacerPlan(
            classification=classification,
            book_width_mm=width,
            clearance_mm=clearance,
            required_opening_mm=required_opening,
            bottleneck_opening_mm=None,
            required_additional_opening_mm=None,
            spacer_required=False,
            spacer_angle_magnitude_deg=None,
            spacer_command_angle_deg=None,
            feasible=False,
            decision="unsupported",
            reason=f"unsupported storage classification: {classification}",
        )

    additional = max(0.0, required_opening - bottleneck)
    spacer_required = additional > 0.0
    if not spacer_required:
        return StorageSpacerPlan(
            classification=classification,
            book_width_mm=width,
            clearance_mm=clearance,
            required_opening_mm=required_opening,
            bottleneck_opening_mm=bottleneck,
            required_additional_opening_mm=0.0,
            spacer_required=False,
            spacer_angle_magnitude_deg=0.0,
            spacer_command_angle_deg=None,
            feasible=True,
            decision="spacer_not_required",
            reason="bottleneck opening already satisfies required opening",
            bottleneck_percentile=percentile,
            valid_row_count=valid_row_count,
        )

    if additional > length:
        return StorageSpacerPlan(
            classification=classification,
            book_width_mm=width,
            clearance_mm=clearance,
            required_opening_mm=required_opening,
            bottleneck_opening_mm=bottleneck,
            required_additional_opening_mm=additional,
            spacer_required=True,
            spacer_angle_magnitude_deg=None,
            spacer_command_angle_deg=None,
            feasible=False,
            decision="mechanism_capacity_exceeded",
            reason=(
                "required additional opening exceeds spacer effective length; "
                "request is not clipped to 90 degrees"
            ),
            bottleneck_percentile=percentile,
            valid_row_count=valid_row_count,
        )

    angle_magnitude = calculate_spacer_angle_deg(
        additional,
        effective_length_mm=length,
    )
    return StorageSpacerPlan(
        classification=classification,
        book_width_mm=width,
        clearance_mm=clearance,
        required_opening_mm=required_opening,
        bottleneck_opening_mm=bottleneck,
        required_additional_opening_mm=additional,
        spacer_required=True,
        spacer_angle_magnitude_deg=angle_magnitude,
        spacer_command_angle_deg=None,
        feasible=False,
        decision="escape_capacity_unavailable",
        reason=(
            "rectangular_space_requires_expansion_but_escape_capacity_"
            "is_unavailable"
        ),
        bottleneck_percentile=percentile,
        valid_row_count=valid_row_count,
    )


def determine_spacer_command_angle(
    classification: str,
    angle_magnitude_deg: float,
) -> float | None:
    """Apply only the established classification-dependent direction sign."""
    if classification == "rectangular":
        return None
    magnitude = _finite_float(angle_magnitude_deg)
    if magnitude is None or not 0.0 <= magnitude <= 90.0:
        raise RuntimeError(
            f"invalid spacer angle magnitude: {angle_magnitude_deg}"
        )
    if classification == "triangular_left_tilted":
        return LEFT_TILTED_SPACER_DIRECTION_SIGN * magnitude
    if classification == "triangular_right_tilted":
        return RIGHT_TILTED_SPACER_DIRECTION_SIGN * magnitude
    raise RuntimeError(
        f"spacer direction requested for unsupported classification: "
        f"{classification}"
    )


def reset_spacer_rotation_before_linear_retraction(hand_worker: Any) -> None:
    """Synchronously restore SP_ROT_0 before any linear retraction."""
    print("[STORAGE SPACER] reset rotation before linear retraction")
    print("[STORAGE SPACER] reset_rot start")
    hand_worker.reset_rot(asynchronous=False)
    print("[STORAGE SPACER] reset_rot done")


def execute_right_tilted_support_plan(
    *,
    hand_worker: Any,
    xarm7: Any,
    plan: RightTiltedSupportPlan,
    step_deg: float = RIGHT_TILTED_SYNC_STEP_DEG,
) -> float:
    """Synchronize G_max SP_ROT with h*tan(alpha) robot -Y movement."""
    steps = plan_right_tilted_rotation_steps(
        plan.spacer_angle_deg,
        plan.arm_right_total_mm,
        step_deg=step_deg,
    )

    print("\n[RIGHT-TILTED CONTROL PLAN]")
    print("classification = triangular_left_tilted")
    print("real-world tilted side = RIGHT")
    print(f"boundary tilt alpha = {plan.boundary_tilt_deg:.2f} deg")
    print(f"contact height h = {plan.contact_height_mm:.2f} mm")
    print(
        "arm total RIGHT movement = h*tan(abs(alpha)) = "
        f"{plan.arm_right_total_mm:.2f} mm"
    )
    print("arm direction = real-world RIGHT / robot -Y")
    print(f"G_max = {plan.max_space_width_mm:.2f} mm")
    print(f"spacer effective length L = {plan.spacer_effective_length_mm:.2f} mm")
    print("SP_ROT angle method = asin(G_max/L), saturated at 90 deg")
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
        print(f"arm total RIGHT target = {plan.arm_right_total_mm:.2f} mm")
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

        rotation_delay_sec = (
            SPACER_ROTATION_START_DELAY_SEC if index == 1 else 0.0
        )

        def rotate_at_this_step() -> None:
            try:
                if rotation_delay_sec > 0.0:
                    time.sleep(rotation_delay_sec)
                hand_worker.rotate_spacer(
                    rotation_step.spacer_command_angle_deg
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

    robot_delta_y = -plan.arm_right_total_mm
    print("[RIGHT-TILTED CONTROL DONE]")
    print(f"final SP_ROT = {plan.spacer_angle_deg:+.2f} deg")
    print(f"final robot Y shift = {robot_delta_y:+.2f} mm")
    print("xArm total method = -h*tan(abs(alpha)); L*sin(theta) not used")
    print("continue insertion without Y recenter = True")
    return robot_delta_y


def _contiguous_x_runs(row_mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive x-runs for one binary mask row."""
    xs = np.flatnonzero(np.asarray(row_mask, dtype=bool))
    if xs.size == 0:
        return []
    split_points = np.flatnonzero(np.diff(xs) > 1) + 1
    groups = np.split(xs, split_points)
    return [(int(group[0]), int(group[-1])) for group in groups]


def estimate_contact_height_opening_mm(
    result_metadata: dict[str, Any],
    *,
    contact_height_mm: float = SPACER_CONTACT_HEIGHT_MM,
) -> dict[str, Any]:
    """Estimate selected-space width at a physical height on the shelf plane.

    The selected mask supplies the actual (possibly sloped) boundaries. Pixel
    rays are intersected with the existing RANSAC shelf-front plane, so no
    fixed px/mm scale is assumed. The less tilted boundary supplies the local
    physical vertical direction; the first valid mask row is the shelf-bottom
    reference because Storage_SAM3 explicitly records image-top of the space
    as the physical book-bottom side.
    """
    requested_height = _finite_float(contact_height_mm)
    if requested_height is None or requested_height < 0.0:
        raise RuntimeError(
            f"invalid spacer contact height: {contact_height_mm}"
        )
    if result_metadata.get("target_y_reference") != (
        "image_top_of_selected_space_equals_physical_book_bottom_side"
    ):
        raise RuntimeError(
            "physical shelf-bottom reference is missing or unsupported"
        )

    selected_mask = np.asarray(result_metadata.get("selected_space_mask"))
    if selected_mask.ndim != 2 or selected_mask.size == 0:
        raise RuntimeError(
            "selected_space_mask is required for contact-height opening"
        )
    selected_mask = selected_mask.astype(bool, copy=False)

    plane = np.asarray(result_metadata.get("plane"), dtype=np.float64)
    if plane.shape != (4,) or not np.all(np.isfinite(plane)):
        raise RuntimeError("valid RANSAC shelf-front plane is required")
    normal_scale = float(np.linalg.norm(plane[:3]))
    if normal_scale <= 1.0e-12:
        raise RuntimeError("RANSAC shelf-front plane has zero normal")
    plane_normal = plane[:3] / normal_scale
    plane_d = float(plane[3] / normal_scale)

    shot_dir_value = result_metadata.get("shot_dir")
    if not shot_dir_value:
        raise RuntimeError("shot_dir is required to load camera intrinsics")
    intrinsics_path = Path(shot_dir_value) / "camera_intrinsics.json"
    try:
        intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"failed to load camera intrinsics: {intrinsics_path}"
        ) from exc

    fx = _finite_float(intrinsics.get("fx"))
    fy = _finite_float(intrinsics.get("fy"))
    ppx = _finite_float(intrinsics.get("ppx"))
    ppy = _finite_float(intrinsics.get("ppy"))
    if (
        fx is None
        or fy is None
        or ppx is None
        or ppy is None
        or fx <= 0.0
        or fy <= 0.0
    ):
        raise RuntimeError(f"invalid camera intrinsics: {intrinsics_path}")
    coeffs = np.asarray(intrinsics.get("coeffs", []), dtype=np.float64)
    if coeffs.size and (
        not np.all(np.isfinite(coeffs))
        or float(np.max(np.abs(coeffs))) > 1.0e-9
    ):
        raise RuntimeError(
            "nonzero/invalid distortion coefficients are unsupported for "
            "contact-height opening estimation"
        )

    def pixel_ray_plane_point(x_px: int, y_px: int) -> np.ndarray:
        ray = np.asarray(
            [(float(x_px) - ppx) / fx, (float(y_px) - ppy) / fy, 1.0],
            dtype=np.float64,
        )
        denominator = float(np.dot(plane_normal, ray))
        if abs(denominator) <= 1.0e-12:
            raise RuntimeError("pixel ray is parallel to the shelf-front plane")
        distance = -plane_d / denominator
        if not math.isfinite(distance) or distance <= 0.0:
            raise RuntimeError("pixel ray/plane intersection is behind camera")
        return distance * ray

    boundary_rows: list[dict[str, Any]] = []
    for y_px in range(selected_mask.shape[0]):
        runs = _contiguous_x_runs(selected_mask[y_px])
        if not runs:
            continue
        left_x, right_x = max(
            runs,
            key=lambda run: (run[1] - run[0] + 1, -run[0]),
        )
        if right_x <= left_x:
            continue
        boundary_rows.append(
            {
                "y": y_px,
                "left_x": left_x,
                "right_x": right_x,
                "left_point": pixel_ray_plane_point(left_x, y_px),
                "right_point": pixel_ray_plane_point(right_x, y_px),
            }
        )
    if len(boundary_rows) < CONTACT_OPENING_MIN_VALID_ROWS:
        raise RuntimeError(
            "too few valid selected-mask rows for contact-height opening: "
            f"{len(boundary_rows)}<{CONTACT_OPENING_MIN_VALID_ROWS}"
        )

    left_tilt = _finite_float(
        result_metadata.get("left_tilt_from_vertical_deg")
    )
    right_tilt = _finite_float(
        result_metadata.get("right_tilt_from_vertical_deg")
    )
    if left_tilt is None or right_tilt is None:
        raise RuntimeError("both boundary tilt values are required")
    upright_side = "left" if left_tilt <= right_tilt else "right"
    image_rows = np.asarray(
        [row["y"] for row in boundary_rows], dtype=np.float64
    )
    upright_points = np.asarray(
        [row[f"{upright_side}_point"] for row in boundary_rows],
        dtype=np.float64,
    )
    design = np.column_stack([image_rows, np.ones_like(image_rows)])
    boundary_fit, _, fit_rank, _ = np.linalg.lstsq(
        design, upright_points, rcond=None
    )
    if fit_rank < 2:
        raise RuntimeError("upright boundary 3D fit is rank deficient")
    vertical_slope = boundary_fit[0]
    vertical_slope -= plane_normal * float(
        np.dot(vertical_slope, plane_normal)
    )
    vertical_slope_norm = float(np.linalg.norm(vertical_slope))
    if vertical_slope_norm <= 1.0e-12:
        raise RuntimeError("physical vertical direction could not be estimated")
    physical_vertical = vertical_slope / vertical_slope_norm
    if float(np.dot(upright_points[-1] - upright_points[0], physical_vertical)) < 0:
        physical_vertical *= -1.0

    vertical_mm_per_image_row = float(
        np.dot(boundary_fit[0], physical_vertical) * 1000.0
    )
    if (
        not math.isfinite(vertical_mm_per_image_row)
        or vertical_mm_per_image_row <= 0.0
    ):
        raise RuntimeError(
            "invalid physical height scale from the upright boundary: "
            f"{vertical_mm_per_image_row} mm/row"
        )

    shelf_bottom_row = float(image_rows[0])
    requested_row = shelf_bottom_row + (
        requested_height / vertical_mm_per_image_row
    )
    if requested_row < image_rows[0] or requested_row > image_rows[-1]:
        raise RuntimeError(
            "selected-space mask has no section at spacer contact height: "
            f"requested_y={requested_row:.2f}, "
            f"valid_y=[{image_rows[0]:.0f},{image_rows[-1]:.0f}]"
        )
    selected_index = int(np.argmin(np.abs(image_rows - requested_row)))
    selected_row = boundary_rows[selected_index]

    physical_horizontal = np.cross(plane_normal, physical_vertical)
    horizontal_norm = float(np.linalg.norm(physical_horizontal))
    if horizontal_norm <= 1.0e-12:
        raise RuntimeError("physical horizontal direction could not be estimated")
    physical_horizontal /= horizontal_norm
    left_point = selected_row["left_point"]
    right_point = selected_row["right_point"]
    current_opening_mm = abs(
        float(np.dot(right_point - left_point, physical_horizontal))
    ) * 1000.0
    if not math.isfinite(current_opening_mm) or current_opening_mm < 0.0:
        raise RuntimeError(
            f"invalid contact-height opening width: {current_opening_mm}"
        )

    actual_height_mm = (
        float(selected_row["y"]) - shelf_bottom_row
    ) * vertical_mm_per_image_row
    representative_depth_m = float(
        0.5 * (left_point[2] + right_point[2])
    )
    return {
        "current_opening_mm": current_opening_mm,
        "contact_height_requested_mm": requested_height,
        "contact_height_actual_mm": actual_height_mm,
        "contact_height_error_mm": actual_height_mm - requested_height,
        "mask_row_y": int(selected_row["y"]),
        "mask_row_run_xy": [
            int(selected_row["left_x"]),
            int(selected_row["right_x"]),
        ],
        "pixel_opening_width_px": int(
            selected_row["right_x"] - selected_row["left_x"] + 1
        ),
        "representative_depth_m": representative_depth_m,
        "fx_px": fx,
        "fy_px": fy,
        "upright_boundary_side": upright_side,
        "vertical_mm_per_image_row": vertical_mm_per_image_row,
        "plane_source": "non_held_book_spine_ransac_plane",
        "camera_intrinsics_path": str(intrinsics_path),
        "valid_mask_row_count": len(boundary_rows),
    }


def calculate_bottleneck_percentile_mm(
    valid_row_widths_mm: list[float] | np.ndarray,
    *,
    percentile: float = RECTANGULAR_BOTTLENECK_PERCENTILE,
    minimum_valid_rows: int = CONTACT_OPENING_MIN_VALID_ROWS,
) -> float:
    """Return the configured lower-tail percentile of valid physical widths."""
    widths = np.asarray(valid_row_widths_mm, dtype=np.float64).reshape(-1)
    widths = widths[np.isfinite(widths) & (widths > 0.0)]
    if widths.size < int(minimum_valid_rows):
        raise RuntimeError(
            "too few valid rows for rectangular bottleneck: "
            f"{widths.size}<{int(minimum_valid_rows)}"
        )
    pct = _finite_float(percentile)
    if pct is None or not 0.0 <= pct <= 100.0:
        raise RuntimeError(f"invalid rectangular bottleneck percentile: {percentile}")
    return float(np.percentile(widths, pct))


def estimate_selected_space_row_widths_mm(
    result_metadata: dict[str, Any],
    *,
    minimum_valid_rows: int = CONTACT_OPENING_MIN_VALID_ROWS,
) -> dict[str, Any]:
    """Measure valid selected-mask row widths on the RANSAC shelf plane."""
    selected_mask = np.asarray(result_metadata.get("selected_space_mask"))
    if selected_mask.ndim != 2 or selected_mask.size == 0:
        raise RuntimeError(
            "selected_space_mask is required for physical row widths"
        )
    selected_mask = selected_mask.astype(bool, copy=False)

    plane = np.asarray(result_metadata.get("plane"), dtype=np.float64)
    if plane.shape != (4,) or not np.all(np.isfinite(plane)):
        raise RuntimeError("valid RANSAC shelf-front plane is required")
    normal_scale = float(np.linalg.norm(plane[:3]))
    if normal_scale <= 1.0e-12:
        raise RuntimeError("RANSAC shelf-front plane has zero normal")
    plane_normal = plane[:3] / normal_scale
    plane_d = float(plane[3] / normal_scale)

    shot_dir_value = result_metadata.get("shot_dir")
    if not shot_dir_value:
        raise RuntimeError("shot_dir is required to load camera intrinsics")
    intrinsics_path = Path(shot_dir_value) / "camera_intrinsics.json"
    try:
        intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"failed to load camera intrinsics: {intrinsics_path}"
        ) from exc

    fx = _finite_float(intrinsics.get("fx"))
    fy = _finite_float(intrinsics.get("fy"))
    ppx = _finite_float(intrinsics.get("ppx"))
    ppy = _finite_float(intrinsics.get("ppy"))
    if (
        fx is None
        or fy is None
        or ppx is None
        or ppy is None
        or fx <= 0.0
        or fy <= 0.0
    ):
        raise RuntimeError(f"invalid camera intrinsics: {intrinsics_path}")
    coeffs = np.asarray(intrinsics.get("coeffs", []), dtype=np.float64)
    if coeffs.size and (
        not np.all(np.isfinite(coeffs))
        or float(np.max(np.abs(coeffs))) > 1.0e-9
    ):
        raise RuntimeError(
            "nonzero/invalid distortion coefficients are unsupported for "
            "selected-space row-width estimation"
        )

    def pixel_ray_plane_point(x_px: int, y_px: int) -> np.ndarray:
        ray = np.asarray(
            [(float(x_px) - ppx) / fx, (float(y_px) - ppy) / fy, 1.0],
            dtype=np.float64,
        )
        denominator = float(np.dot(plane_normal, ray))
        if abs(denominator) <= 1.0e-12:
            raise RuntimeError("pixel ray is parallel to shelf-front plane")
        distance = -plane_d / denominator
        if not math.isfinite(distance) or distance <= 0.0:
            raise RuntimeError("pixel ray/plane intersection is behind camera")
        point = distance * ray
        if not np.all(np.isfinite(point)):
            raise RuntimeError("pixel ray/plane intersection is nonfinite")
        return point

    boundary_rows: list[dict[str, Any]] = []
    invalid_row_count = 0
    for y_px in range(selected_mask.shape[0]):
        runs = _contiguous_x_runs(selected_mask[y_px])
        if not runs:
            continue
        left_x, right_x = max(
            runs,
            key=lambda run: (run[1] - run[0] + 1, -run[0]),
        )
        if right_x <= left_x:
            invalid_row_count += 1
            continue
        try:
            left_point = pixel_ray_plane_point(left_x, y_px)
            right_point = pixel_ray_plane_point(right_x, y_px)
        except RuntimeError:
            invalid_row_count += 1
            continue
        boundary_rows.append(
            {
                "y": int(y_px),
                "left_x": int(left_x),
                "right_x": int(right_x),
                "left_point": left_point,
                "right_point": right_point,
            }
        )

    if len(boundary_rows) < int(minimum_valid_rows):
        raise RuntimeError(
            "too few valid selected-mask rows for physical width estimation: "
            f"{len(boundary_rows)}<{int(minimum_valid_rows)}"
        )

    left_tilt = _finite_float(
        result_metadata.get("left_tilt_from_vertical_deg")
    )
    right_tilt = _finite_float(
        result_metadata.get("right_tilt_from_vertical_deg")
    )
    if left_tilt is None or right_tilt is None:
        raise RuntimeError("both boundary tilt values are required")
    upright_side = "left" if left_tilt <= right_tilt else "right"
    image_rows = np.asarray(
        [row["y"] for row in boundary_rows], dtype=np.float64
    )
    upright_points = np.asarray(
        [row[f"{upright_side}_point"] for row in boundary_rows],
        dtype=np.float64,
    )
    design = np.column_stack([image_rows, np.ones_like(image_rows)])
    boundary_fit, _, fit_rank, _ = np.linalg.lstsq(
        design, upright_points, rcond=None
    )
    if fit_rank < 2:
        raise RuntimeError("upright boundary 3D fit is rank deficient")
    physical_vertical = boundary_fit[0]
    physical_vertical -= plane_normal * float(
        np.dot(physical_vertical, plane_normal)
    )
    vertical_norm = float(np.linalg.norm(physical_vertical))
    if vertical_norm <= 1.0e-12:
        raise RuntimeError("physical vertical direction could not be estimated")
    physical_vertical /= vertical_norm

    physical_horizontal = np.cross(plane_normal, physical_vertical)
    horizontal_norm = float(np.linalg.norm(physical_horizontal))
    if horizontal_norm <= 1.0e-12:
        raise RuntimeError("physical horizontal direction could not be estimated")
    physical_horizontal /= horizontal_norm

    valid_widths_mm: list[float] = []
    valid_rows: list[int] = []
    for row in boundary_rows:
        width_mm = abs(
            float(
                np.dot(
                    row["right_point"] - row["left_point"],
                    physical_horizontal,
                )
            )
        ) * 1000.0
        if not math.isfinite(width_mm) or width_mm <= 0.0:
            invalid_row_count += 1
            continue
        valid_widths_mm.append(width_mm)
        valid_rows.append(int(row["y"]))

    if len(valid_widths_mm) < int(minimum_valid_rows):
        raise RuntimeError(
            "too few valid physical selected-space row widths: "
            f"{len(valid_widths_mm)}<{int(minimum_valid_rows)}"
        )

    return {
        "valid_row_count": len(valid_widths_mm),
        "invalid_row_count": int(invalid_row_count),
        "valid_row_y_min": min(valid_rows),
        "valid_row_y_max": max(valid_rows),
        "valid_row_y_px": valid_rows,
        "row_width_min_mm": float(np.min(valid_widths_mm)),
        "row_width_median_mm": float(np.median(valid_widths_mm)),
        "row_width_max_mm": float(np.max(valid_widths_mm)),
        "row_widths_mm": valid_widths_mm,
        "upright_boundary_side": upright_side,
        "camera_intrinsics_path": str(intrinsics_path),
        "width_method": (
            "largest continuous selected-mask run per row; endpoint rays "
            "intersected with RANSAC shelf-front plane; physical-horizontal "
            "3D separation"
        ),
    }


def evaluate_physical_bottom_narrowing_rule(
    *,
    physical_top_width_mm: float,
    physical_bottom_width_mm: float,
    minimum_space_width_mm: float,
    held_book_width_mm: float,
    width_trend_tolerance_mm: float = TWO_VIEW_BOTTLENECK_TIE_TOLERANCE_MM,
) -> dict[str, Any]:
    """Reject only a physically bottom-narrowing space that is too narrow."""
    physical_top = _finite_float(physical_top_width_mm)
    physical_bottom = _finite_float(physical_bottom_width_mm)
    minimum_space = _finite_float(minimum_space_width_mm)
    held_book = _finite_float(held_book_width_mm)
    trend_tolerance = _finite_float(width_trend_tolerance_mm)
    if physical_top is None or physical_top <= 0.0:
        raise RuntimeError(
            f"physical top width must be finite and positive: {physical_top_width_mm}"
        )
    if physical_bottom is None or physical_bottom <= 0.0:
        raise RuntimeError(
            "physical bottom width must be finite and positive: "
            f"{physical_bottom_width_mm}"
        )
    if minimum_space is None or minimum_space <= 0.0:
        raise RuntimeError(
            "minimum space width must be finite and positive: "
            f"{minimum_space_width_mm}"
        )
    if held_book is None or held_book <= 0.0:
        raise RuntimeError(
            f"held book width must be finite and positive: {held_book_width_mm}"
        )
    if trend_tolerance is None or trend_tolerance < 0.0:
        raise RuntimeError(
            "width trend tolerance must be finite and nonnegative: "
            f"{width_trend_tolerance_mm}"
        )

    bottom_narrowing = physical_bottom < physical_top - trend_tolerance
    reject = bottom_narrowing and minimum_space < held_book
    return {
        "physical_top_width_mm": physical_top,
        "physical_bottom_width_mm": physical_bottom,
        "minimum_space_width_mm": minimum_space,
        "minimum_space_width_method": (
            "configured lower-tail percentile of valid physical row widths"
        ),
        "minimum_space_width_percentile": float(
            RECTANGULAR_BOTTLENECK_PERCENTILE
        ),
        "held_book_width_mm": held_book,
        "held_book_width_source": "TEST_BOOK_WIDTH_MM",
        "physical_width_trend_tolerance_mm": trend_tolerance,
        "bottom_narrowing": bottom_narrowing,
        "bottom_narrowing_width_reject": reject,
    }


def evaluate_selected_space_vertical_width_check(
    result_metadata: dict[str, Any],
    selected_space_width_info: dict[str, Any],
    *,
    minimum_space_width_mm: float,
    held_book_width_mm: float = TEST_BOOK_WIDTH_MM,
) -> dict[str, Any]:
    """Map image bands to physical top/bottom and apply the width rule."""
    expected_reference = (
        "image_top_of_selected_space_equals_physical_book_bottom_side"
    )
    if result_metadata.get("target_y_reference") != expected_reference:
        raise RuntimeError(
            "physical top/bottom mapping is missing or unsupported"
        )

    selected_space_id = result_metadata.get("selected_space_id")
    spaces = result_metadata.get("space_candidate_metadata")
    if not isinstance(spaces, list):
        raise RuntimeError("space candidate metadata is required for width bands")
    selected_space = next(
        (
            item
            for item in spaces
            if isinstance(item, dict)
            and item.get("space_id") == selected_space_id
        ),
        None,
    )
    if selected_space is None:
        raise RuntimeError(
            "selected space metadata is missing for physical width trend"
        )

    vertical_mapping = selected_space.get("image_to_physical_vertical_mapping")
    if (
        not isinstance(vertical_mapping, dict)
        or vertical_mapping.get("image_top_small_y")
        != "physical_book_bottom_side"
        or vertical_mapping.get("image_bottom_large_y")
        != "physical_book_top_side"
    ):
        raise RuntimeError(
            "selected space has an unsupported image-to-physical mapping"
        )

    def band_range(name: str) -> tuple[int, int]:
        value = selected_space.get(name)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise RuntimeError(f"selected space is missing {name}")
        start, end = int(value[0]), int(value[1])
        if end <= start:
            raise RuntimeError(f"selected space has invalid {name}: {value}")
        return start, end

    image_top_y_start, image_top_y_end = band_range(
        "image_top_band_y_range"
    )
    image_bottom_y_start, image_bottom_y_end = band_range(
        "image_bottom_band_y_range"
    )
    row_y_px = np.asarray(
        selected_space_width_info.get("valid_row_y_px"),
        dtype=np.int64,
    ).reshape(-1)
    row_widths_mm = np.asarray(
        selected_space_width_info.get("row_widths_mm"),
        dtype=np.float64,
    ).reshape(-1)
    if row_y_px.size == 0 or row_y_px.size != row_widths_mm.size:
        raise RuntimeError(
            "physical row positions and widths are missing or inconsistent"
        )

    physical_bottom_band = row_widths_mm[
        (row_y_px >= image_top_y_start) & (row_y_px < image_top_y_end)
    ]
    physical_top_band = row_widths_mm[
        (row_y_px >= image_bottom_y_start) & (row_y_px < image_bottom_y_end)
    ]
    if physical_bottom_band.size == 0 or physical_top_band.size == 0:
        raise RuntimeError(
            "physical top/bottom width bands contain no valid rows"
        )

    check = evaluate_physical_bottom_narrowing_rule(
        physical_top_width_mm=float(np.median(physical_top_band)),
        physical_bottom_width_mm=float(np.median(physical_bottom_band)),
        minimum_space_width_mm=minimum_space_width_mm,
        held_book_width_mm=held_book_width_mm,
    )
    check.update(
        {
            "physical_bottom_width_source": (
                "median physical row width in image-top/physical-bottom band"
            ),
            "physical_top_width_source": (
                "median physical row width in image-bottom/physical-top band"
            ),
            "physical_bottom_image_y_range": [
                image_top_y_start,
                image_top_y_end,
            ],
            "physical_top_image_y_range": [
                image_bottom_y_start,
                image_bottom_y_end,
            ],
        }
    )
    return check


def estimate_rectangular_bottleneck_mm(
    result_metadata: dict[str, Any],
    *,
    percentile: float = RECTANGULAR_BOTTLENECK_PERCENTILE,
    minimum_valid_rows: int = CONTACT_OPENING_MIN_VALID_ROWS,
) -> dict[str, Any]:
    """Preserve the existing rectangular 5-percentile bottleneck result."""
    width_info = estimate_selected_space_row_widths_mm(
        result_metadata,
        minimum_valid_rows=minimum_valid_rows,
    )
    bottleneck_mm = calculate_bottleneck_percentile_mm(
        width_info["row_widths_mm"],
        percentile=percentile,
        minimum_valid_rows=minimum_valid_rows,
    )
    return {
        **width_info,
        "bottleneck_opening_mm": bottleneck_mm,
        "bottleneck_percentile": float(percentile),
    }


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


def select_real_world_left_boundary_target_pixel(
    result_metadata: dict[str, Any],
) -> tuple[tuple[int, int], dict[str, Any]]:
    """Select the image-right boundary at the existing target y coordinate."""
    selected_mask = np.asarray(result_metadata.get("selected_space_mask"))
    if selected_mask.ndim != 2:
        raise RuntimeError(
            "selected_space_mask is required for real-world-left boundary target"
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
    target_x = int(np.clip(run_right, 0, width - 1))
    if not selected_mask[target_y, target_x]:
        raise AssertionError(
            "real-world-left boundary target left selected-space mask"
        )

    info = {
        "physical_mapping": "image_right_equals_real_world_left",
        "original_target_px": [original_x, target_y],
        "modified_target_px": [target_x, target_y],
        "selected_row_run_xy": [run_left, run_right],
        "target_boundary": "image_right_real_world_left",
        "new_target_offset_px": 0,
        "target_inside_selected_space_mask": True,
    }
    return (target_x, target_y), info


def build_real_world_left_boundary_target_camera(
    result_metadata: dict[str, Any],
    original_target_cam: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Move only camera X to the image-right/real-world-left boundary."""
    modified_target_px, info = select_real_world_left_boundary_target_pixel(
        result_metadata
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
            "left-boundary target horizontal shift currently requires the "
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
            "camera_target_method": (
                "horizontal_pixel_shift_same_camera_y_z_to_image_right_boundary"
            ),
            "camera_intrinsics_path": str(intrinsics_path),
            "original_target_cam_m": original_cam.tolist(),
            "modified_target_cam_m": modified_cam.tolist(),
        }
    )
    return modified_cam, info


def _load_target_geometry_intrinsics(
    result_metadata: dict[str, Any],
    intrinsics_override: dict[str, Any] | None = None,
) -> tuple[dict[str, float], str]:
    """Load the aligned pinhole intrinsics without accessing any hardware."""
    if intrinsics_override is None:
        shot_dir_value = result_metadata.get("shot_dir")
        if not shot_dir_value:
            raise RuntimeError("shot_dir is missing; camera intrinsics unavailable")
        intrinsics_path = Path(shot_dir_value) / "camera_intrinsics.json"
        try:
            raw = json.loads(intrinsics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"failed to load camera intrinsics: {intrinsics_path}: {exc}"
            ) from exc
        source = str(intrinsics_path)
    else:
        raw = dict(intrinsics_override)
        source = "explicit_intrinsics_override"

    values = {key: _finite_float(raw.get(key)) for key in ("fx", "fy", "ppx", "ppy")}
    if (
        values["fx"] is None
        or values["fy"] is None
        or values["ppx"] is None
        or values["ppy"] is None
        or values["fx"] <= 0.0
        or values["fy"] <= 0.0
    ):
        raise RuntimeError(f"invalid camera intrinsics: {source}")
    coeffs = np.asarray(raw.get("coeffs", []), dtype=np.float64)
    if coeffs.size and (
        not np.all(np.isfinite(coeffs))
        or float(np.max(np.abs(coeffs))) > 1.0e-9
    ):
        raise RuntimeError(
            "B-based target requires the zero-distortion aligned intrinsics "
            "used by saved storage captures"
        )
    return {key: float(value) for key, value in values.items()}, source


def _pixel_ray_plane_point(
    x_px: float,
    y_px: float,
    intrinsics: dict[str, float],
    plane_normal: np.ndarray,
    plane_d: float,
) -> np.ndarray:
    ray = np.asarray(
        [
            (float(x_px) - intrinsics["ppx"]) / intrinsics["fx"],
            (float(y_px) - intrinsics["ppy"]) / intrinsics["fy"],
            1.0,
        ],
        dtype=np.float64,
    )
    denominator = float(np.dot(plane_normal, ray))
    if abs(denominator) <= 1.0e-12:
        raise RuntimeError("pixel ray is parallel to the RANSAC shelf plane")
    scale = -float(plane_d) / denominator
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("pixel ray/RANSAC plane intersection is behind camera")
    point = scale * ray
    if not np.all(np.isfinite(point)):
        raise RuntimeError("pixel ray/RANSAC plane intersection is nonfinite")
    return point


def build_physical_shelf_frame(
    result_metadata: dict[str, Any],
    *,
    intrinsics_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build vertical/normal/real-right axes on the saved RANSAC plane.

    Cross-product order is not trusted for horizontal sign.  The sign is
    explicitly aligned with the projected image-left direction because the
    established camera mapping is image-left == real-world RIGHT.
    """
    selected_mask = np.asarray(result_metadata.get("selected_space_mask"))
    if selected_mask.ndim != 2 or selected_mask.size == 0:
        raise RuntimeError("selected_space_mask is required for shelf axes")
    selected_mask = selected_mask.astype(bool, copy=False)
    height, width = selected_mask.shape

    plane = np.asarray(result_metadata.get("plane"), dtype=np.float64)
    if plane.shape != (4,) or not np.all(np.isfinite(plane)):
        raise RuntimeError("valid non-held-book RANSAC shelf plane is required")
    normal_norm = float(np.linalg.norm(plane[:3]))
    if normal_norm <= 1.0e-12:
        raise RuntimeError("RANSAC shelf plane has zero normal")
    plane_normal = plane[:3] / normal_norm
    plane_d = float(plane[3] / normal_norm)
    intrinsics, intrinsics_source = _load_target_geometry_intrinsics(
        result_metadata,
        intrinsics_override,
    )

    left_tilt = _finite_float(result_metadata.get("left_tilt_from_vertical_deg"))
    right_tilt = _finite_float(result_metadata.get("right_tilt_from_vertical_deg"))
    if left_tilt is None or right_tilt is None:
        raise RuntimeError("both selected-space boundary tilts are required")
    upright_side = "left" if left_tilt <= right_tilt else "right"

    image_rows: list[int] = []
    upright_points: list[np.ndarray] = []
    for y_px in range(height):
        runs = _contiguous_x_runs(selected_mask[y_px])
        if not runs:
            continue
        run_left, run_right = max(
            runs,
            key=lambda run: (run[1] - run[0] + 1, -run[0]),
        )
        boundary_x = run_left if upright_side == "left" else run_right
        image_rows.append(y_px)
        upright_points.append(
            _pixel_ray_plane_point(
                boundary_x,
                y_px,
                intrinsics,
                plane_normal,
                plane_d,
            )
        )
    if len(image_rows) < CONTACT_OPENING_MIN_VALID_ROWS:
        raise RuntimeError(
            "too few selected-mask rows for physical shelf frame: "
            f"{len(image_rows)}<{CONTACT_OPENING_MIN_VALID_ROWS}"
        )

    row_values = np.asarray(image_rows, dtype=np.float64)
    point_values = np.asarray(upright_points, dtype=np.float64)
    design = np.column_stack([row_values, np.ones_like(row_values)])
    fit, _, fit_rank, _ = np.linalg.lstsq(design, point_values, rcond=None)
    if fit_rank < 2:
        raise RuntimeError("upright boundary 3D fit is rank deficient")
    physical_vertical = fit[0]
    physical_vertical -= plane_normal * float(
        np.dot(physical_vertical, plane_normal)
    )
    vertical_norm = float(np.linalg.norm(physical_vertical))
    if vertical_norm <= 1.0e-12:
        raise RuntimeError("physical vertical direction could not be estimated")
    physical_vertical /= vertical_norm
    if float(np.dot(point_values[-1] - point_values[0], physical_vertical)) < 0.0:
        physical_vertical *= -1.0

    physical_horizontal_right = np.cross(plane_normal, physical_vertical)
    horizontal_norm = float(np.linalg.norm(physical_horizontal_right))
    if horizontal_norm <= 1.0e-12:
        raise RuntimeError("physical horizontal direction could not be estimated")
    physical_horizontal_right /= horizontal_norm

    mapping_y = int(round(float(np.median(row_values))))
    image_left_x = max(0, int(round(0.25 * (width - 1))))
    image_right_x = min(width - 1, int(round(0.75 * (width - 1))))
    image_left_point = _pixel_ray_plane_point(
        image_left_x,
        mapping_y,
        intrinsics,
        plane_normal,
        plane_d,
    )
    image_right_point = _pixel_ray_plane_point(
        image_right_x,
        mapping_y,
        intrinsics,
        plane_normal,
        plane_d,
    )
    image_left_direction = image_left_point - image_right_point
    if float(np.dot(physical_horizontal_right, image_left_direction)) < 0.0:
        physical_horizontal_right *= -1.0
    if float(np.dot(physical_horizontal_right, image_left_direction)) <= 0.0:
        raise RuntimeError("failed to align physical horizontal with image-left")

    return {
        "physical_vertical": physical_vertical,
        "plane_normal": plane_normal,
        "physical_horizontal_right": physical_horizontal_right,
        "physical_horizontal_mapping": "image-left -> real-world right",
        "horizontal_sign_reference_row_y": mapping_y,
        "horizontal_sign_reference_pixels": [
            [image_left_x, mapping_y],
            [image_right_x, mapping_y],
        ],
        "upright_boundary_side": upright_side,
        "intrinsics": intrinsics,
        "intrinsics_source": intrinsics_source,
    }


def build_left_tilted_book_right_edge_target_camera(
    result_metadata: dict[str, Any],
    original_target_cam: Any,
    *,
    book_width_mm: float = TEST_BOOK_WIDTH_MM,
    intrinsics_override: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replace only target horizontal position with X_ref = B + w right."""
    width_mm = _finite_float(book_width_mm)
    if width_mm is None or width_mm <= 0.0:
        raise RuntimeError(f"book width must be finite and positive: {book_width_mm}")
    b_px_value = result_metadata.get(
        "real_world_left_neighbor_book_inner_bottom_px"
    )
    b_cam_value = result_metadata.get(
        "real_world_left_neighbor_book_inner_bottom_cam_m"
    )
    if b_px_value is None or len(b_px_value) != 2:
        raise RuntimeError("real-world LEFT neighbor B pixel is unavailable")
    if b_cam_value is None:
        raise RuntimeError("real-world LEFT neighbor B camera point is unavailable")
    b_pixel = [int(round(float(value))) for value in b_px_value]
    b_cam = np.asarray(b_cam_value, dtype=np.float64).reshape(3)
    original_cam = np.asarray(original_target_cam, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(b_cam)) or not np.all(np.isfinite(original_cam)):
        raise RuntimeError("B/original camera target contains nonfinite values")

    shelf_frame = build_physical_shelf_frame(
        result_metadata,
        intrinsics_override=intrinsics_override,
    )
    horizontal_right = shelf_frame["physical_horizontal_right"]
    physical_vertical = shelf_frame["physical_vertical"]
    plane_normal = shelf_frame["plane_normal"]
    x_reference_cam = b_cam + (width_mm / 1000.0) * horizontal_right
    horizontal_delta_m = float(
        np.dot(x_reference_cam - original_cam, horizontal_right)
    )
    modified_cam = original_cam + horizontal_delta_m * horizontal_right

    vertical_delta_m = float(np.dot(modified_cam - original_cam, physical_vertical))
    normal_delta_m = float(np.dot(modified_cam - original_cam, plane_normal))
    horizontal_error_m = float(
        np.dot(modified_cam - x_reference_cam, horizontal_right)
    )
    tolerance_m = 1.0e-9
    if (
        abs(vertical_delta_m) > tolerance_m
        or abs(normal_delta_m) > tolerance_m
        or abs(horizontal_error_m) > tolerance_m
    ):
        raise AssertionError(
            "B-based target component preservation failed: "
            f"vertical={vertical_delta_m}, normal={normal_delta_m}, "
            f"horizontal_error={horizontal_error_m}"
        )

    intrinsics = shelf_frame["intrinsics"]
    if x_reference_cam[2] <= 0.0:
        raise RuntimeError("X reference lies behind the camera")
    x_reference_pixel = [
        int(round(intrinsics["fx"] * x_reference_cam[0] / x_reference_cam[2] + intrinsics["ppx"])),
        int(round(intrinsics["fy"] * x_reference_cam[1] / x_reference_cam[2] + intrinsics["ppy"])),
    ]
    info = {
        "neighbor_book_mask_index": result_metadata.get(
            "real_world_left_neighbor_book_mask_index"
        ),
        "B_pixel": b_pixel,
        "B_camera_m": b_cam.tolist(),
        "physical_horizontal_direction_camera": horizontal_right.tolist(),
        "physical_vertical_direction_camera": physical_vertical.tolist(),
        "shelf_plane_normal_camera": plane_normal.tolist(),
        "physical_horizontal_mapping": shelf_frame[
            "physical_horizontal_mapping"
        ],
        "book_width_mm": width_mm,
        "X_reference_definition": "B + book_width toward real-world right",
        "X_reference_camera_m": x_reference_cam.tolist(),
        "X_reference_pixel": x_reference_pixel,
        "original_target_camera_m": original_cam.tolist(),
        "modified_target_camera_m": modified_cam.tolist(),
        "horizontal_target_shift_mm": horizontal_delta_m * 1000.0,
        "vertical_component_delta_mm": vertical_delta_m * 1000.0,
        "normal_component_delta_mm": normal_delta_m * 1000.0,
        "horizontal_component_error_mm": horizontal_error_m * 1000.0,
        "vertical_component_preserved": True,
        "normal_depth_component_preserved": True,
        "upright_boundary_side": shelf_frame["upright_boundary_side"],
        "intrinsics_source": shelf_frame["intrinsics_source"],
        "target_semantics": (
            "X is the desired held-book right-edge horizontal reference; "
            "the existing recognition-target-to-storage calibration is reused"
        ),
    }
    return modified_cam, info


def save_left_tilted_book_target_geometry_overlay(
    result_metadata: dict[str, Any],
    target_info: dict[str, Any],
) -> str:
    """Save selected space, exact neighbor mask, B, and X as a new image."""
    import cv2

    shot_dir_value = result_metadata.get("shot_dir")
    if not shot_dir_value:
        raise RuntimeError("shot_dir is required for B/X debug overlay")
    shot_dir = Path(shot_dir_value)
    input_path = shot_dir / "input.png"
    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"failed to load debug input image: {input_path}")
    selected_mask = np.asarray(
        result_metadata.get("selected_space_mask"), dtype=bool
    )
    neighbor_mask = np.asarray(
        result_metadata.get("real_world_left_neighbor_book_mask"), dtype=bool
    )
    if selected_mask.shape != image_bgr.shape[:2]:
        raise RuntimeError("selected-space mask/image shape mismatch for overlay")
    if neighbor_mask.shape != image_bgr.shape[:2]:
        raise RuntimeError("neighbor-book mask/image shape mismatch for overlay")

    overlay = image_bgr.copy()
    for mask, color in (
        (selected_mask, np.asarray([255, 200, 0], dtype=np.float32)),
        (neighbor_mask, np.asarray([0, 165, 255], dtype=np.float32)),
    ):
        overlay[mask] = (
            0.65 * overlay[mask].astype(np.float32) + 0.35 * color
        ).astype(np.uint8)

    b_pixel = tuple(int(value) for value in target_info["B_pixel"])
    x_pixel = tuple(int(value) for value in target_info["X_reference_pixel"])
    cv2.line(overlay, b_pixel, x_pixel, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(overlay, b_pixel, 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(overlay, x_pixel, 6, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.putText(
        overlay,
        "B",
        (b_pixel[0] + 7, b_pixel[1] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "X",
        (x_pixel[0] + 7, x_pixel[1] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    label_position = (
        max(5, min(b_pixel[0], x_pixel[0])),
        max(22, min(b_pixel[1], x_pixel[1]) - 18),
    )
    cv2.putText(
        overlay,
        f"w={float(target_info['book_width_mm']):.0f}mm",
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    output_path = shot_dir / "left_tilted_book_target_geometry_overlay.png"
    suffix = 1
    while output_path.exists():
        output_path = shot_dir / (
            f"left_tilted_book_target_geometry_overlay_{suffix:03d}.png"
        )
        suffix += 1
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"failed to save B/X debug overlay: {output_path}")
    return str(output_path)


def build_tilted_book_high_reach_target_robot(
    target_robot_mm: Any,
    *,
    z_offset_mm: float | None = LEFT_BOOK_REACH_Z_OFFSET_MM,
) -> tuple[np.ndarray, float]:
    """Raise either tilted-book reach target along robot-base +Z.

    ``moveL_relative`` and the existing storage Z helpers use robot-base +Z
    for the upward/retract direction and -Z for lowering/insertion.  Refuse to
    run until a positive hardware-validated offset has been configured.
    """
    target = np.asarray(
        target_robot_mm,
        dtype=np.float64,
    ).reshape(3).copy()
    if not np.all(np.isfinite(target)):
        raise RuntimeError(f"invalid tilted-book robot target: {target}")
    if z_offset_mm is None:
        raise RuntimeError(
            "LEFT_BOOK_REACH_Z_OFFSET_MM is not configured; determine a safe "
            "positive robot +Z offset by hardware clearance testing before "
            "running tilted-book storage"
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


def apply_left_tilted_reach_correction(
    right_side_target_robot_mm: Any,
    *,
    right_offset_mm: float = LEFT_TILTED_REACH_RIGHT_OFFSET_MM,
    backoff_mm: float = LEFT_TILTED_REACH_BACKOFF_MM,
) -> np.ndarray:
    """Apply real-world-right (-Y) and shelf-front backoff (-X) in robot mm."""
    target = np.asarray(
        right_side_target_robot_mm,
        dtype=np.float64,
    ).reshape(3).copy()
    if not np.all(np.isfinite(target)):
        raise RuntimeError(f"invalid base right-side robot target: {target}")
    right_offset = _finite_float(right_offset_mm)
    backoff = _finite_float(backoff_mm)
    if right_offset is None or right_offset < 0.0:
        raise RuntimeError(
            "left-tilted real-world-right offset must be finite and "
            f"nonnegative: {right_offset_mm}"
        )
    if backoff is None or backoff < 0.0:
        raise RuntimeError(
            "left-tilted shelf-front backoff must be finite and nonnegative: "
            f"{backoff_mm}"
        )

    corrected = target.copy()
    corrected[0] -= backoff
    corrected[1] -= right_offset
    return corrected


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


def classify_boundary_tilts(
    left_tilt_deg: float,
    right_tilt_deg: float,
    *,
    vertical_tolerance_deg: float = VERTICAL_ANGLE_TOLERANCE_DEG,
    difference_threshold_deg: float = (
        BOUNDARY_TILT_DIFFERENCE_THRESHOLD_DEG
    ),
) -> tuple[str, str | None, float]:
    """Apply the absolute-plus-relative boundary shape rules."""
    left = _finite_float(left_tilt_deg)
    right = _finite_float(right_tilt_deg)
    vertical_tolerance = _finite_float(vertical_tolerance_deg)
    difference_threshold = _finite_float(difference_threshold_deg)
    if left is None or left < 0.0:
        raise RuntimeError(f"invalid left boundary tilt: {left_tilt_deg}")
    if right is None or right < 0.0:
        raise RuntimeError(f"invalid right boundary tilt: {right_tilt_deg}")
    if vertical_tolerance is None or vertical_tolerance < 0.0:
        raise RuntimeError(
            f"invalid vertical tolerance: {vertical_tolerance_deg}"
        )
    if difference_threshold is None or difference_threshold <= 0.0:
        raise RuntimeError(
            "invalid boundary tilt difference threshold: "
            f"{difference_threshold_deg}"
        )

    difference = abs(left - right)
    if (
        left <= vertical_tolerance
        and right <= vertical_tolerance
        and difference < difference_threshold
    ):
        return "rectangular", None, difference
    if left - right >= difference_threshold:
        return "triangular_left_tilted", "left", difference
    if right - left >= difference_threshold:
        return "triangular_right_tilted", "right", difference
    return "ambiguous", None, difference


def resolve_ambiguous_storage_space(
    *,
    left_tilt_deg: float,
    right_tilt_deg: float,
    bottleneck_opening_mm: float,
    book_width_mm: float = TEST_BOOK_WIDTH_MM,
    equal_tilt_epsilon_deg: float = AMBIGUOUS_TILT_EQUAL_EPSILON_DEG,
) -> AmbiguousStorageResolution:
    """Resolve an ambiguous shape using strict width, then larger tilt."""
    left = _finite_float(left_tilt_deg)
    right = _finite_float(right_tilt_deg)
    bottleneck = _finite_float(bottleneck_opening_mm)
    book_width = _finite_float(book_width_mm)
    epsilon = _finite_float(equal_tilt_epsilon_deg)
    if left is None or right is None or left < 0.0 or right < 0.0:
        raise RuntimeError("ambiguous fallback requires finite boundary tilts")
    if bottleneck is None or bottleneck < 0.0:
        raise RuntimeError(
            "ambiguous fallback requires a finite nonnegative bottleneck: "
            f"{bottleneck_opening_mm}"
        )
    if book_width is None or book_width <= 0.0:
        raise RuntimeError(
            f"ambiguous fallback requires positive book width: {book_width_mm}"
        )
    if epsilon is None or epsilon < 0.0:
        raise RuntimeError(f"invalid equal-tilt epsilon: {equal_tilt_epsilon_deg}")

    # Deliberately strict and deliberately excludes STORAGE_CLEARANCE_MM.
    if bottleneck > book_width:
        return AmbiguousStorageResolution(
            classification="direct_insert",
            tilted_side=None,
            selected_by_larger_boundary=None,
            reason=(
                "ambiguous boundaries but physical bottleneck exceeds held "
                "book width"
            ),
        )

    delta = left - right
    if abs(delta) <= epsilon:
        return AmbiguousStorageResolution(
            classification="unsupported",
            tilted_side=None,
            selected_by_larger_boundary=None,
            reason="cannot determine tilted side from equal boundary angles",
        )
    if delta > 0.0:
        return AmbiguousStorageResolution(
            classification="triangular_left_tilted",
            tilted_side="left",
            selected_by_larger_boundary="left",
            reason=(
                "physical bottleneck does not exceed held book width; "
                "selected larger left boundary tilt"
            ),
        )
    return AmbiguousStorageResolution(
        classification="triangular_right_tilted",
        tilted_side="right",
        selected_by_larger_boundary="right",
        reason=(
            "physical bottleneck does not exceed held book width; "
            "selected larger right boundary tilt"
        ),
    )


def classify_storage_space(
    result_metadata: dict[str, Any],
    *,
    vertical_tolerance_deg: float = VERTICAL_ANGLE_TOLERANCE_DEG,
    difference_threshold_deg: float = (
        BOUNDARY_TILT_DIFFERENCE_THRESHOLD_DEG
    ),
    minimum_width_px: float = MIN_SUPPORTED_SPACE_WIDTH_PX,
) -> StorageSpaceClassification:
    """Classify and fully resolve one selected space before capture choice."""
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
    fatal_reasons: list[str] = []
    diagnostic_reasons: list[str] = []

    if selected_space_id is None:
        fatal_reasons.append("selected_space_id_missing")
    if left_tilt is None:
        fatal_reasons.append("left_boundary_tilt_missing_or_nonfinite")
    if right_tilt is None:
        fatal_reasons.append("right_boundary_tilt_missing_or_nonfinite")
    if book_bottom_side_width_px is None:
        fatal_reasons.append("book_bottom_side_width_missing_or_nonfinite")
    elif book_bottom_side_width_px < float(minimum_width_px):
        fatal_reasons.append(
            "book_bottom_side_width_below_minimum:"
            f"{book_bottom_side_width_px:.1f}<{minimum_width_px:.1f}"
        )

    classification = "unsupported"
    tilted_side: str | None = None
    tilt_difference: float | None = None
    classification_source = "invalid_metadata"
    fallback_bottleneck: float | None = None
    fallback_valid_row_count: int | None = None
    fallback_selected_side: str | None = None
    if not fatal_reasons:
        assert left_tilt is not None
        assert right_tilt is not None
        classification, tilted_side, tilt_difference = classify_boundary_tilts(
            left_tilt,
            right_tilt,
            vertical_tolerance_deg=vertical_tolerance_deg,
            difference_threshold_deg=difference_threshold_deg,
        )
        classification_source = "absolute_and_relative_boundary_tilts"
        if classification == "ambiguous":
            bottleneck_info = estimate_rectangular_bottleneck_mm(
                result_metadata
            )
            fallback_bottleneck = float(
                bottleneck_info["bottleneck_opening_mm"]
            )
            fallback_valid_row_count = int(
                bottleneck_info["valid_row_count"]
            )
            resolution = resolve_ambiguous_storage_space(
                left_tilt_deg=left_tilt,
                right_tilt_deg=right_tilt,
                bottleneck_opening_mm=fallback_bottleneck,
                book_width_mm=TEST_BOOK_WIDTH_MM,
            )
            classification = resolution.classification
            tilted_side = resolution.tilted_side
            fallback_selected_side = resolution.selected_by_larger_boundary
            classification_source = "ambiguous_physical_bottleneck_fallback"
            if classification == "unsupported":
                fatal_reasons.append(resolution.reason)
            else:
                diagnostic_reasons.append(resolution.reason)

    if classification.startswith("triangular_"):
        if image_guide_angle_deg is None:
            diagnostic_reasons.append(
                "image_guide_angle_missing_or_nonfinite"
            )
        if guide_side != tilted_side:
            diagnostic_reasons.append(
                f"guide_side_mismatch:guide={guide_side},tilted={tilted_side}"
            )
        if (
            tilted_side == "left"
            and image_guide_angle_deg is not None
            and image_guide_angle_deg >= 90.0
        ):
            diagnostic_reasons.append(
                "left_tilted_guide_angle_not_below_90_deg"
            )
        if (
            tilted_side == "right"
            and image_guide_angle_deg is not None
            and image_guide_angle_deg <= 90.0
        ):
            diagnostic_reasons.append(
                "right_tilted_guide_angle_not_above_90_deg"
            )

    if fatal_reasons:
        classification = "unsupported"
        tilted_side = None

    # Spacer planning is intentionally deferred until the supported boundary
    # tilt is known.  Rectangular planning additionally needs the selected
    # mask, RANSAC plane, and intrinsics for a physical width in mm.
    spacer_angle = None
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
        reasons=tuple(fatal_reasons + diagnostic_reasons),
        tilt_difference_deg=tilt_difference,
        difference_threshold_deg=float(difference_threshold_deg),
        classification_source=classification_source,
        ambiguous_bottleneck_opening_mm=fallback_bottleneck,
        ambiguous_bottleneck_valid_row_count=fallback_valid_row_count,
        fallback_selected_by_larger_boundary=fallback_selected_side,
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
            "arm total RIGHT movement = h*tan(abs(alpha)) = "
            f"{plan.arm_right_total_mm:.2f} mm"
        )
        print("arm direction = real-world RIGHT / robot -Y")
        print(f"G_max = {plan.max_space_width_mm:.2f} mm")
        print(f"spacer effective length = {plan.spacer_effective_length_mm:.2f} mm")
        print("SP_ROT angle method = asin(G_max/L), saturated at 90 deg")
        print(f"SP_ROT angle = {plan.spacer_angle_deg:+.2f} deg")
        print("SP_ROT direction = same as LEFT tilted / positive")
        print(f"high reach offset = +{plan.high_reach_offset_mm:.2f} mm")
        print("book width used for angle = False")
        print("clearance used for angle = False")
        print("boundary tilt used for SP_ROT angle = False")
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
        print("angle method = G_max / spacer_length")
        print("angle model = L * sin(theta) = G_max")
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


def _selected_space_mask_height_px(result_metadata: dict[str, Any]) -> int:
    selected_mask = np.asarray(result_metadata.get("selected_space_mask"))
    if selected_mask.ndim != 2 or selected_mask.size == 0:
        raise RuntimeError(
            "selected_space_mask is required for two-view mask height"
        )
    valid_y = np.flatnonzero(np.any(selected_mask.astype(bool), axis=1))
    if valid_y.size == 0:
        raise RuntimeError("selected_space_mask has no valid pixels")
    return int(valid_y[-1] - valid_y[0] + 1)


def _latest_capture_shot_dir(capture_root: Path) -> str | None:
    """Recover the per-shot directory even when recognition raises later."""
    if not capture_root.is_dir():
        return None
    shot_dirs = [path for path in capture_root.iterdir() if path.is_dir()]
    if not shot_dirs:
        return None
    return str(max(shot_dirs, key=lambda path: path.name))


def evaluate_storage_capture_candidate(
    *,
    label: str,
    capture_index: int,
    result: tuple[float, np.ndarray, dict[str, Any]],
    capture_tcp_pose: Any,
) -> StorageCaptureCandidate:
    """Run existing classification and spacer planning without hardware motion."""
    _, _, result_metadata = result
    shot_dir_value = result_metadata.get("shot_dir")
    shot_dir = None if shot_dir_value is None else str(shot_dir_value)
    pose = tuple(
        np.asarray(capture_tcp_pose, dtype=np.float64).reshape(6).tolist()
    )

    decision: StorageSpaceClassification | None = None
    spacer_plan: StorageSpacerPlan | None = None
    rectangular_bottleneck_info: dict[str, Any] | None = None
    selected_space_width_info: dict[str, Any] | None = None
    diagnostic_current_opening_mm: float | None = None
    physical_bottleneck_mm: float | None = None
    selected_space_height_px = 0
    valid_physical_row_count = 0
    physical_vertical_width_check: dict[str, Any] | None = None
    planning_error: str | None = None

    try:
        selected_space_height_px = _selected_space_mask_height_px(
            result_metadata
        )
        decision = classify_storage_space(result_metadata)
        if not decision.supported:
            return StorageCaptureCandidate(
                label=label,
                capture_index=capture_index,
                result=result,
                capture_tcp_pose=pose,
                shot_dir=shot_dir,
                decision=decision,
                selected_space_height_px=selected_space_height_px,
            )

        # Use the same selected mask, camera intrinsics, and RANSAC shelf-front
        # plane for every classification so two viewpoints are compared in mm.
        selected_space_width_info = estimate_selected_space_row_widths_mm(
            result_metadata
        )
        physical_bottleneck_mm = calculate_bottleneck_percentile_mm(
            selected_space_width_info["row_widths_mm"]
        )
        valid_physical_row_count = int(
            selected_space_width_info["valid_row_count"]
        )
        physical_vertical_width_check = (
            evaluate_selected_space_vertical_width_check(
                result_metadata,
                selected_space_width_info,
                minimum_space_width_mm=physical_bottleneck_mm,
                held_book_width_mm=TEST_BOOK_WIDTH_MM,
            )
        )
        result_metadata.update(physical_vertical_width_check)
        if physical_vertical_width_check[
            "bottom_narrowing_width_reject"
        ]:
            raise RuntimeError(
                "physical-bottom-narrowing space bottleneck is smaller than "
                "held book width: "
                f"{physical_bottleneck_mm:.6f}<"
                f"{TEST_BOOK_WIDTH_MM:.6f} mm"
            )

        if decision.classification.startswith("triangular_"):
            try:
                spacer_opening_info = estimate_contact_height_opening_mm(
                    result_metadata
                )
                diagnostic_current_opening_mm = float(
                    spacer_opening_info["current_opening_mm"]
                )
            except RuntimeError as exc:
                print(
                    f"[STORAGE SPACER DIAGNOSTIC][{label}] "
                    f"current opening unavailable: {exc}"
                )

            if decision.classification == "triangular_left_tilted":
                triangular_boundary_tilt_deg = decision.left_boundary_tilt_deg
            elif decision.classification == "triangular_right_tilted":
                triangular_boundary_tilt_deg = decision.right_boundary_tilt_deg
            else:
                raise RuntimeError(
                    "unsupported triangular classification: "
                    f"{decision.classification}"
                )

            result_metadata.update(
                {
                    "selected_space_row_widths_mm": list(
                        selected_space_width_info["row_widths_mm"]
                    ),
                    "selected_space_width_min_mm": float(
                        selected_space_width_info["row_width_min_mm"]
                    ),
                    "selected_space_width_median_mm": float(
                        selected_space_width_info["row_width_median_mm"]
                    ),
                    "selected_space_width_max_mm": float(
                        selected_space_width_info["row_width_max_mm"]
                    ),
                    "selected_space_width_valid_row_count": (
                        valid_physical_row_count
                    ),
                }
            )
            spacer_plan = plan_storage_spacer(
                classification=decision.classification,
                triangular_boundary_tilt_deg=triangular_boundary_tilt_deg,
                triangular_max_space_width_mm=float(
                    selected_space_width_info["row_width_max_mm"]
                ),
            )
        elif decision.classification == "rectangular":
            rectangular_bottleneck_info = {
                **selected_space_width_info,
                "bottleneck_opening_mm": physical_bottleneck_mm,
                "bottleneck_percentile": float(
                    RECTANGULAR_BOTTLENECK_PERCENTILE
                ),
            }
            spacer_plan = plan_storage_spacer(
                classification=decision.classification,
                rectangular_bottleneck_mm=physical_bottleneck_mm,
                rectangular_valid_row_count=valid_physical_row_count,
            )
        elif decision.classification == "direct_insert":
            if (
                decision.ambiguous_bottleneck_opening_mm is None
                or decision.ambiguous_bottleneck_valid_row_count is None
            ):
                raise RuntimeError(
                    "direct insert is missing ambiguous bottleneck metadata"
                )
            if not math.isclose(
                physical_bottleneck_mm,
                decision.ambiguous_bottleneck_opening_mm,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise RuntimeError(
                    "direct-insert bottleneck changed between classification "
                    "and planning: "
                    f"{decision.ambiguous_bottleneck_opening_mm:.9f} vs "
                    f"{physical_bottleneck_mm:.9f} mm"
                )
            rectangular_bottleneck_info = {
                **selected_space_width_info,
                "bottleneck_opening_mm": physical_bottleneck_mm,
                "bottleneck_percentile": float(
                    RECTANGULAR_BOTTLENECK_PERCENTILE
                ),
            }
            spacer_plan = plan_storage_spacer(
                classification=decision.classification,
                direct_insert_bottleneck_mm=physical_bottleneck_mm,
                direct_insert_valid_row_count=valid_physical_row_count,
            )
        else:
            raise RuntimeError(
                "unsupported final storage classification: "
                f"{decision.classification}"
            )
    except RuntimeError as exc:
        planning_error = str(exc)

    return StorageCaptureCandidate(
        label=label,
        capture_index=capture_index,
        result=result,
        capture_tcp_pose=pose,
        shot_dir=shot_dir,
        decision=decision,
        spacer_plan=spacer_plan,
        rectangular_bottleneck_info=rectangular_bottleneck_info,
        selected_space_width_info=selected_space_width_info,
        diagnostic_current_opening_mm=diagnostic_current_opening_mm,
        physical_bottleneck_mm=physical_bottleneck_mm,
        selected_space_height_px=selected_space_height_px,
        valid_physical_row_count=valid_physical_row_count,
        physical_vertical_width_check=physical_vertical_width_check,
        planning_error=planning_error,
    )


def select_best_storage_capture(
    candidates: list[StorageCaptureCandidate],
) -> tuple[StorageCaptureCandidate, str]:
    """Choose a usable view by physical width, coverage, then stable order."""
    usable = sorted(
        (candidate for candidate in candidates if candidate.usable),
        key=lambda candidate: candidate.capture_index,
    )
    if not usable:
        details = "\n".join(
            f"{candidate.label}:\n{candidate.failure_reason}"
            for candidate in sorted(
                candidates,
                key=lambda candidate: candidate.capture_index,
            )
        )
        raise RuntimeError(
            "[STORAGE TWO-VIEW SELECTION FAILED]\n"
            f"{details}\n"
            "no usable capture result"
        )
    if len(usable) == 1:
        return usable[0], "only usable capture"

    first, second = usable[0], usable[1]
    assert first.physical_bottleneck_mm is not None
    assert second.physical_bottleneck_mm is not None
    width_delta = (
        second.physical_bottleneck_mm - first.physical_bottleneck_mm
    )
    if abs(width_delta) > TWO_VIEW_BOTTLENECK_TIE_TOLERANCE_MM:
        if width_delta > 0.0:
            return second, "larger physical bottleneck"
        return first, "larger physical bottleneck"
    if second.selected_space_height_px != first.selected_space_height_px:
        if second.selected_space_height_px > first.selected_space_height_px:
            return second, "larger selected-space mask height"
        return first, "larger selected-space mask height"
    if second.valid_physical_row_count != first.valid_physical_row_count:
        if second.valid_physical_row_count > first.valid_physical_row_count:
            return second, "more valid physical rows"
        return first, "more valid physical rows"
    return first, "deterministic first-capture tie-break"


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


def validate_storage_0911_depth_constants() -> None:
    """Stop before hardware motion if the fixed 0911 depth plan drifts."""
    named_values = {
        "TCP_TO_BOOK_REAR_MM": TCP_TO_BOOK_REAR_MM,
        "REACH_MARGIN_MM": REACH_MARGIN_MM,
        "INSERT_WITH_SPACER_MM": INSERT_WITH_SPACER_MM,
        "FINAL_BOOK_PUSH_MM": FINAL_BOOK_PUSH_MM,
        "EXPECTED_INSERT_DEPTH_MM": EXPECTED_INSERT_DEPTH_MM,
    }
    for name, value in named_values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(
                f"invalid 0911 storage depth constant: {name}={value}"
            )
    if not math.isclose(
        EXPECTED_FINAL_TCP_OFFSET_MM,
        75.0,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise RuntimeError(
            "0911 final TCP offset must be 75.0 mm: "
            f"{EXPECTED_FINAL_TCP_OFFSET_MM}"
        )
    if not math.isclose(
        EXPECTED_INSERT_DEPTH_MM,
        140.0,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise RuntimeError(
            "0911 expected book insertion depth must be 140.0 mm: "
            f"{EXPECTED_INSERT_DEPTH_MM}"
        )


def build_storage_0911_depth_plan(
    surface_robot_xyz: Any,
) -> dict[str, Any]:
    """Build and display the fixed post-recognition XYZ/depth plan."""
    validate_storage_0911_depth_constants()
    surface = np.asarray(surface_robot_xyz, dtype=np.float64).reshape(3)
    if not np.isfinite(surface).all():
        raise RuntimeError(
            f"recognized surface robot XYZ is not finite: {surface.tolist()}"
        )
    reach_target = surface.copy()
    reach_target[0] -= REACH_TCP_BACKOFF_MM
    expected_final_tcp_x = surface[0] + EXPECTED_FINAL_TCP_OFFSET_MM
    expected_final_book_rear_x = expected_final_tcp_x + TCP_TO_BOOK_REAR_MM

    print("\n[STORAGE 0911 DEPTH PLAN]")
    print(
        "recognized surface XYZ = "
        f"[{surface[0]:.3f}, {surface[1]:.3f}, {surface[2]:.3f}] mm"
    )
    print(f"TCP to book rear = {TCP_TO_BOOK_REAR_MM:.1f} mm")
    print(f"reach margin = {REACH_MARGIN_MM:.1f} mm")
    print(
        "reach TCP X = surface X - "
        f"{REACH_TCP_BACKOFF_MM:.1f} mm = {reach_target[0]:.3f} mm"
    )
    print(f"insert with spacer = +{INSERT_WITH_SPACER_MM:.1f} mm")
    print(f"final push = +{FINAL_BOOK_PUSH_MM:.1f} mm")
    print(f"expected final TCP X = {expected_final_tcp_x:.3f} mm")
    print(
        "expected final book rear X = "
        f"{expected_final_book_rear_x:.3f} mm"
    )
    print(
        "expected book insertion depth = "
        f"{EXPECTED_INSERT_DEPTH_MM:.1f} mm"
    )
    return {
        "surface_robot_xyz_mm": surface.tolist(),
        "reach_target_xyz_mm": reach_target.tolist(),
        "expected_final_tcp_x_mm": float(expected_final_tcp_x),
        "expected_final_book_rear_x_mm": float(
            expected_final_book_rear_x
        ),
        "expected_book_insertion_depth_mm": EXPECTED_INSERT_DEPTH_MM,
    }


def _save_storage_0911_motion_log(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Atomically persist the plan and every measured TCP checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
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
        xarm7.get_tcp_pose(is_radian=True),
        dtype=np.float64,
    ).reshape(6)
    if not np.isfinite(pose).all():
        raise RuntimeError(f"non-finite TCP pose at {label}/{phase}")
    records.append(
        {
            "type": "tcp_checkpoint",
            "label": label,
            "phase": phase,
            "tcp_pose_mm_rad": pose.tolist(),
        }
    )
    _save_storage_0911_motion_log(log_path, records)
    print(
        f"[{label}] {phase} TCP XYZ = "
        f"[{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] mm"
    )
    return pose


def move_storage_0911_absolute_reach(
    *,
    xarm7: Any,
    reach_target_xyz: Any,
    records: list[dict[str, Any]],
    log_path: Path,
    velocity: float,
    acceleration: float,
) -> np.ndarray:
    """Move to surface XYZ with only the specified -75 mm X backoff."""
    label = "STORAGE 0911 REACH"
    before = _record_storage_0911_tcp(
        xarm7=xarm7,
        label=label,
        phase="before",
        records=records,
        log_path=log_path,
    )
    target_xyz = np.asarray(reach_target_xyz, dtype=np.float64).reshape(3)
    target_pose = before.copy()
    target_pose[:3] = target_xyz
    print(f"[{label}] target TCP pose = {target_pose.tolist()}")
    # The current XArm7 exposes only the internal absolute MoveL primitive;
    # keep that compatibility detail inside this 0911-side wrapper.
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
    after = _record_storage_0911_tcp(
        xarm7=xarm7,
        label=label,
        phase="after",
        records=records,
        log_path=log_path,
    )
    print(f"[{label}] MoveL ret = {ret}")
    print(f"[{label}] actual X movement = {after[0] - before[0]:+.3f} mm")
    if ret != 0:
        raise RuntimeError(f"0911 absolute reaching failed: ret={ret}")
    return after


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
    """Issue one pure robot-base +X MoveL and log measured before/after TCP."""
    before = _record_storage_0911_tcp(
        xarm7=xarm7,
        label=label,
        phase="before",
        records=records,
        log_path=log_path,
    )
    delta = [float(distance_mm), 0.0, 0.0, 0.0, 0.0, 0.0]
    print(f"[{label}] relative MoveL = {delta}")
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
    print(f"[{label}] MoveL ret = {ret}")
    print(f"[{label}] actual X movement = {after[0] - before[0]:+.3f} mm")
    if ret != 0:
        raise RuntimeError(f"{label} failed: ret={ret}")
    return after


def run_one_storage_cycle(
    *,
    hand_worker: Any,
    xarm7: Any,
    cycle_index: int,
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
    from storage_ffs_depth import (
        record_ffs_robot_target_trace,
        run_capture_and_pca_depth_space,
    )
    from xarm7.control.robot_base_coordinate import cam_mm_to_robot_mm
    from xarm7.control.xarm7 import (
        TCP_ACC_2,
        TCP_VEL_2,
    )

    try:
        # This check intentionally runs before the first cycle motion.
        validate_storage_0911_depth_constants()
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
        hand_worker.open_until_full(asynchronous=False)
        time.sleep(5.0)
        hand_worker.grasp()

        capture_offset_before_pose = np.asarray(
            xarm7.get_tcp_pose(is_radian=True),
            dtype=np.float64,
        ).reshape(6)
        print("\n[STORAGE CAPTURE HEIGHT OFFSET]")
        print(
            "capture Z offset = "
            f"{STORAGE_CAPTURE_Z_OFFSET_MM:+.2f} mm"
        )
        print("direction = robot -Z")
        print("applied after grasp = True")
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
        print("1st capture uses lowered pose = True")
        print("2nd capture keeps same lowered Z = True")

        def move_to_second_capture() -> None:
            nonlocal capture_tcp_pose_1
            capture_tcp_pose_1 = xarm7.get_tcp_pose(is_radian=True)
            print("[STORAGE] 1st capture TCP pose =", capture_tcp_pose_1)
            second_capture_move_ret = xarm7.moveL_y_offset(
                SECOND_CAPTURE_DY_MM,
                velocity=40,
                acceleration=40,
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

        capture_cycle_name = (
            f"cycle_{cycle_index:04d}_{time.strftime('%Y%m%d_%H%M%S')}_"
            f"{time.time_ns()}"
        )
        capture_cycle_root = STORAGE_CAPTURE_ROOT / capture_cycle_name
        if capture_cycle_root.exists():
            raise RuntimeError(
                "two-view capture cycle directory already exists; refusing "
                f"to overwrite: {capture_cycle_root}"
            )
        first_capture_root = capture_cycle_root / "1st"
        second_capture_root = capture_cycle_root / "2nd"
        third_capture_root = capture_cycle_root / "3rd"

        first_result = None
        first_recognition_error: str | None = None
        print("\n========== STORAGE 1st CAPTURE ==========")
        try:
            first_result = run_capture_and_pca_depth_space(
                out_dir=first_capture_root,
                after_capture_callback=move_to_second_capture,
                show_debug_images=not AUTO_RUN,
                target_3d_mode=STORAGE_TARGET_3D_MODE,
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
        print("\n========== STORAGE 2nd CAPTURE ==========")
        try:
            second_result = run_capture_and_pca_depth_space(
                out_dir=second_capture_root,
                after_capture_callback=None,
                show_debug_images=not AUTO_RUN,
                target_3d_mode=STORAGE_TARGET_3D_MODE,
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
                velocity=40,
                acceleration=40,
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
            print("\n========== STORAGE 3rd CAPTURE ==========")
            try:
                third_result = run_capture_and_pca_depth_space(
                    out_dir=third_capture_root,
                    after_capture_callback=None,
                    show_debug_images=not AUTO_RUN,
                    target_3d_mode=STORAGE_TARGET_3D_MODE,
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
        if not res.get("ffs_depth_correction_valid"):
            raise RuntimeError(
                "FFS depth correction invalid: selected capture has no valid "
                "FFS correction; abort storage before reaching"
            )
        decision = selected_candidate.decision
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

        motion_target_cam = np.asarray(
            first_target_cam,
            dtype=np.float64,
        ).reshape(3).copy()
        right_side_target_info: dict[str, Any] | None = None
        left_boundary_target_info: dict[str, Any] | None = None
        if should_use_right_tilted_support_sync(decision.classification):
            # For image-left/real-world-RIGHT tilt, reach the opposite
            # real-world LEFT boundary: image-right in the current camera.
            motion_target_cam, left_boundary_target_info = (
                build_real_world_left_boundary_target_camera(
                    res,
                    first_target_cam,
                )
            )
        elif decision.classification in {
            "triangular_right_tilted",
            "direct_insert",
        }:
            motion_target_cam, right_side_target_info = (
                build_real_world_right_side_target_camera(
                    res,
                    first_target_cam,
                )
            )

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
        print(f"[RESULT] depth_ref_rs_m = {res.get('depth_ref_rs_m')}")
        print(
            "[RESULT] depth_correction_ffs_m = "
            f"{res.get('depth_correction_ffs_m')}"
        )
        print(
            "[RESULT] depth_ref_corrected_m = "
            f"{res.get('depth_ref_corrected_m')}"
        )
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
        depth_ref_rs_m = float(res["depth_ref_rs_m"])
        depth_ref_corrected_m = float(res["depth_ref_corrected_m"])
        if not math.isclose(
            float(motion_target_cam[2]),
            depth_ref_corrected_m,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise RuntimeError(
                "FFS depth correction invalid: corrected depth was not "
                "propagated to the selected camera motion target"
            )
        motion_target_cam_rs = (
            np.asarray(motion_target_cam, dtype=np.float64)
            * (depth_ref_rs_m / depth_ref_corrected_m)
        )
        first_target_robot_rs_mm = cam_mm_to_robot_mm(
            coordinate_transform_arm,
            1000.0 * motion_target_cam_rs,
        )
        record_ffs_robot_target_trace(
            res,
            camera_target_rs_m=motion_target_cam_rs,
            camera_target_corrected_m=motion_target_cam,
            robot_target_rs_mm=first_target_robot_rs_mm,
            robot_target_corrected_mm=first_target_robot_mm,
        )
        surface_robot_xyz = np.asarray(
            first_target_robot_mm,
            dtype=np.float64,
        ).reshape(3).copy()
        print("[STORAGE 0911] surface_robot_xyz =", surface_robot_xyz)
        depth_plan = build_storage_0911_depth_plan(surface_robot_xyz)
        motion_records: list[dict[str, Any]] = [
            {"type": "depth_plan", **depth_plan}
        ]
        motion_log_path = (
            capture_cycle_root / "storage_0911_motion_log.json"
        )
        _save_storage_0911_motion_log(motion_log_path, motion_records)
        print(f"[STORAGE 0911] motion log = {motion_log_path}")

        time.sleep(5.0)
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
            f"xArm7 reaches the 0911 point derived from "
            f"{recognition_source} with unchanged RPY"
        )
        reach_pose = move_storage_0911_absolute_reach(
            xarm7=xarm7,
            reach_target_xyz=depth_plan["reach_target_xyz_mm"],
            records=motion_records,
            log_path=motion_log_path,
            velocity=TCP_VEL_2,
            acceleration=TCP_ACC_2,
        )

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

        spacer_required = spacer_plan.spacer_required
        if spacer_required:
            if spacer_command_angle_deg is None:
                raise RuntimeError("geometry-based spacer command is missing")
            if should_use_right_tilted_sp_lin_partial_expansion(
                decision.classification
            ):
                print("\n[RIGHT-TILTED SP_LIN PARTIAL EXPANSION]")
                print("classification = triangular_left_tilted")
                print("real-world tilted side = right")
                print(f"normal SP_LIN target = {SP_LIN_FRONT}")
                print(
                    "reduction = "
                    f"{RIGHT_TILTED_SP_LIN_REDUCTION_STEPS} step"
                )
                print(
                    "partial SP_LIN target = "
                    f"{SP_LIN_RIGHT_TILTED_FRONT}"
                )
                print("estimated normal stroke = approx. 240 mm")
                print("estimated reduction = approx. 30 mm")
                print("SP_LIN partial expansion start")
                hand_worker.expand_sp_lin_right_tilted(
                    asynchronous=True,
                )
            else:
                print(
                    "[SP_LIN] use normal expansion target = "
                    f"{SP_LIN_FRONT}"
                )
                print("[DEBUG] before expand_sp_lin")
                hand_worker.expand_sp_lin(asynchronous=True)
                print("[DEBUG] after expand_sp_lin")
            time.sleep(14.0)
            if should_use_right_tilted_sp_lin_partial_expansion(
                decision.classification
            ):
                print("SP_LIN partial expansion completed")

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
                if spacer_angle_magnitude_deg is None:
                    raise RuntimeError(
                        "RIGHT tilted G_max angle magnitude is missing"
                    )
                if spacer_command_angle_deg <= 0.0:
                    raise RuntimeError(
                        "RIGHT tilted SP_ROT command must be positive: "
                        f"{spacer_command_angle_deg}"
                    )
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
            time.sleep(SPACER_ROTATION_START_DELAY_SEC)
            hand_worker.rotate_spacer(spacer_command_angle_deg)
            print(
                "[STORAGE 0911] SP_ROT completed without synchronized "
                "xArm Y motion"
            )
        else:
            print(
                "[STORAGE SPACER] opening is sufficient: skip spacer "
                "expansion/rotation; rotate_spacer(0) is not called"
            )

        spacer_insert_pose = move_storage_0911_relative_x(
            xarm7=xarm7,
            distance_mm=INSERT_WITH_SPACER_MM,
            label="STORAGE 0911 INSERT WITH SPACER",
            records=motion_records,
            log_path=motion_log_path,
            velocity=15,
            acceleration=15,
        )

        spacer_rotation_reset = False
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
            time.sleep(SPACER_ROTATION_RESET_WAIT_SEC)
            reset_spacer_rotation_before_linear_retraction(hand_worker)
            spacer_rotation_reset = True
            if not spacer_rotation_reset:
                raise RuntimeError(
                    "one-shot SP_LIN retraction requested without rotation "
                    "reset"
                )
            print("\n[STORAGE SPACER ONE-SHOT RETRACT]")
            print("rotation reset completed = True")
            print("retract sequence = current -> BACK")
            print("contract_sp_lin_1 used = False")
            print(
                "SP_LIN BACK timeout = "
                f"{ONE_SHOT_SP_LIN_BACK_TIMEOUT_SEC:.1f} sec"
            )
            print("contract_sp_lin_2 start")
            hand_worker.contract_sp_lin_2(
                asynchronous=False,
                timeout_sec=ONE_SHOT_SP_LIN_BACK_TIMEOUT_SEC,
            )
            print("[DEBUG] after one-shot contract_sp_lin_2")
        else:
            time.sleep(SPACER_ROTATION_RESET_BASE_WAIT_SEC)
            print(
                "[STORAGE SPACER] no-spacer classification: "
                "skip reset_rot and SP_LIN retraction"
            )

        final_tcp_pose = move_storage_0911_relative_x(
            xarm7=xarm7,
            distance_mm=FINAL_BOOK_PUSH_MM,
            label="STORAGE 0911 FINAL BOOK PUSH",
            records=motion_records,
            log_path=motion_log_path,
            velocity=15,
            acceleration=15,
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
                "reach_actual_tcp_pose_mm_rad": reach_pose.tolist(),
                "spacer_insert_actual_tcp_pose_mm_rad": (
                    spacer_insert_pose.tolist()
                ),
                "final_actual_tcp_pose_mm_rad": final_tcp_pose.tolist(),
                "actual_book_rear_x_mm": float(actual_book_rear_x),
                "actual_insertion_depth_mm": float(actual_insertion_depth),
            }
        )
        _save_storage_0911_motion_log(motion_log_path, motion_records)

        time.sleep(0.5)
        print("[STORAGE RELEASE] open gripper until full")
        hand_worker.open_until_full(asynchronous=False)
        print("[STORAGE RELEASE] gripper open command completed")
        ret = xarm7.moveL_to_post_storage(asynchronous=True)
        print("[DEBUG] moveL_to_post_storage ret =", ret)
        time.sleep(4.0)
        ret = xarm7.moveJ_to_capture_right(asynchronous=False)
        print("[DEBUG] moveJ_to_capture_right ret =", ret)
        if ret != 0:
            raise RuntimeError(
                f"capture-side return failed at cycle end: ret={ret}"
            )
        print(
            "\n"
            "==================================================\n"
            f" STORAGE CYCLE {cycle_index} COMPLETED\n"
            "=================================================="
        )
        return True

    except BaseException:
        # Do not convert Ctrl+C or runtime/hardware failures into a next-cycle
        # request.  The process-level handler in main performs the established
        # emergency stop and one-time cleanup.
        raise


def main() -> None:
    print(f"[STORAGE TARGET 3D MODE] {STORAGE_TARGET_3D_MODE}")
    # Validate the isolated runner/repository/weight before creating any
    # xArm, Dynamixel, ROS, or RealSense object.
    from storage_ffs_depth import load_ffs_runtime_config

    ffs_runtime = load_ffs_runtime_config()
    print("[FFS] isolated runner configuration validated")
    print(f"[FFS] model weights = {ffs_runtime.weights_path}")

    # Lazy imports keep pure classification importable without starting or
    # requiring any hardware connection.
    from Dynamixel_win_pro_hand_book.dynamixel_worker_client import (
        DynamixelWorkerClient,
    )
    from xarm7.control.xarm7 import XArm7
    import rclpy

    hand_worker = None
    xarm7 = None
    node = None

    try:
        print("start upright storage loop")
        print("[DXL] starting Dynamixel worker")
        hand_worker = DynamixelWorkerClient()
        print("[DXL] worker initialized")

        rclpy.init()
        node = rclpy.create_node("upright_storage_node")
        xarm7 = XArm7(node=node, host="192.168.2.197")
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
            continue_running = run_one_storage_cycle(
                hand_worker=hand_worker,
                xarm7=xarm7,
                cycle_index=cycle_index,
            )
            if not continue_running:
                print(
                    "[STORAGE LOOP] user-requested normal exit after "
                    f"cycle {cycle_index}"
                )
                break
            if AUTO_RUN:
                print(
                    f"[AUTO RUN] cycle {cycle_index} completed -> next cycle"
                )
            cycle_index += 1

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
