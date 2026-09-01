"""Experimental 0831 storage sequence with geometry-based spacer angles.

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

# Experimental held-book and spacer geometry for this standalone test.
TEST_BOOK_WIDTH_MM = 15.0
STORAGE_CLEARANCE_MM = 1.0
# Empirical linear opening calibration for triangular spaces.  This is an
# additional millimetre displacement after contact, not an angle and not the
# separate hand-geometry reach offset below.
TRIANGULAR_EXTRA_OPENING_MM = 10.0
SPACER_EFFECTIVE_LENGTH_MM = 50.0
SPACER_PRE_ROTATION_FORWARD_MM = 30.0
BOOK_HEIGHT_MM = 210.0
RIGHT_TILTED_SPACER_CONTACT_HEIGHT_MM = 40.0
# Real-world left-book correction uses the existing 40 mm normal spacer
# height plus LEFT_BOOK_REACH_Z_OFFSET_MM=60 mm high reach.
LEFT_TILTED_SPACER_CONTACT_HEIGHT_MM = 100.0
# Mechanical reach across the held book; this is distinct from the 1 mm
# storage clearance used to admit the book.
LEFT_TILTED_HAND_GEOMETRY_EXTRA_MM = 10.0
# The existing 100 mm selected-mask measurement remains diagnostic only.
SPACER_CONTACT_HEIGHT_MM = LEFT_TILTED_SPACER_CONTACT_HEIGHT_MM
RECTANGULAR_BOTTLENECK_PERCENTILE = 5.0

# This experiment isolates spacer-angle control.  Keep the established xArm
# lateral corrections available for comparison, but disable them by default.
ENABLE_XARM_LATERAL_CORRECTION = False

# Match the existing boundary-fit minimum in
# detection/pro_handbook/sam_py_demo/config/shelf_storage_detection_sam3.json.
CONTACT_OPENING_MIN_VALID_ROWS = 20

# Preserve the established physical direction without fixing the magnitude.
LEFT_TILTED_SPACER_DIRECTION_SIGN = -1.0
RIGHT_TILTED_SPACER_DIRECTION_SIGN = 1.0

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
    hand_geometry_horizontal_offset_mm: float
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


@dataclass(frozen=True)
class TriangularForwardPlan:
    original_total_forward_mm: float
    pre_rotation_forward_mm: float
    remaining_forward_mm: float


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
    left_tilted_hand_geometry_extra_mm: float = (
        LEFT_TILTED_HAND_GEOMETRY_EXTRA_MM
    ),
) -> TriangularSpacerGeometry:
    """Plan a triangular-space command without accessing any hardware.

    Image-left tilt means the real-world right book is tilted; image-right
    tilt means the real-world left book is tilted.  For this conservative
    standalone experiment, a triangular bottleneck is treated as zero and the
    book width, storage clearance, and triangular opening calibration are all
    generated after contact.  That linear displacement is added in spacer-tip
    displacement (sin) space rather than angle space; it is intentionally not
    reduced by contact_height/book_height.
    """
    width = _finite_float(book_width_mm)
    clearance = _finite_float(clearance_mm)
    opening_calibration = _finite_float(triangular_extra_opening_mm)
    boundary_tilt = _finite_float(boundary_tilt_deg)
    book_height = _finite_float(book_height_mm)
    spacer_length = _finite_float(spacer_effective_length_mm)
    right_contact_height = _finite_float(right_tilted_contact_height_mm)
    left_contact_height = _finite_float(left_tilted_contact_height_mm)
    hand_extra = _finite_float(left_tilted_hand_geometry_extra_mm)
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
    if hand_extra is None or hand_extra < 0.0:
        raise RuntimeError(
            "left-tilted hand geometry extra must be finite and nonnegative: "
            f"{left_tilted_hand_geometry_extra_mm}"
        )

    base_required_opening = width + clearance
    calibrated_required_opening = (
        base_required_opening + opening_calibration
    )
    if classification == "triangular_left_tilted":
        # Image-left boundary tilt is the real-world RIGHT tilted book.  The
        # spacer is already on that side, so no hand-crossing free rotation is
        # added before contact.
        real_world_tilted_side = "right"
        contact_height = right_contact_height
        hand_geometry_offset = 0.0
        geometry_offset_angle = 0.0
    elif classification == "triangular_right_tilted":
        # Image-right boundary tilt is the real-world LEFT tilted book.  The
        # spacer must reach across the held-book width plus a distinct 10 mm
        # mechanical hand offset; this is not storage clearance.
        real_world_tilted_side = "left"
        contact_height = left_contact_height
        hand_geometry_offset = width + hand_extra
        geometry_offset_angle = math.degrees(
            math.atan(hand_geometry_offset / contact_height)
        )
    else:
        raise RuntimeError(
            "triangular spacer geometry requested for unsupported "
            f"classification: {classification}"
        )

    contact_angle = boundary_tilt + geometry_offset_angle
    # Do not apply the former rigid-body h/H scaling.  Real shelf books may
    # slide, flex, or move as a group, so generate the calibrated full
    # 0 -> (w+clearance+opening_calibration) bottleneck opening after contact.
    # The calibration is a linear millimetre value, never an angle offset.
    push_displacement = calibrated_required_opening
    maximum_remaining_horizontal_capacity = spacer_length * (
        1.0 - math.sin(math.radians(contact_angle))
    )
    asin_argument = (
        math.sin(math.radians(contact_angle))
        + push_displacement / spacer_length
    )
    if not math.isfinite(asin_argument):
        return TriangularSpacerGeometry(
            classification=classification,
            real_world_tilted_side=real_world_tilted_side,
            book_width_mm=width,
            clearance_mm=clearance,
            base_required_opening_mm=base_required_opening,
            triangular_extra_opening_mm=opening_calibration,
            calibrated_required_opening_mm=calibrated_required_opening,
            book_height_mm=book_height,
            spacer_contact_height_mm=contact_height,
            boundary_tilt_deg=boundary_tilt,
            hand_geometry_horizontal_offset_mm=hand_geometry_offset,
            geometry_offset_angle_deg=geometry_offset_angle,
            contact_angle_deg=contact_angle,
            push_displacement_mm=push_displacement,
            spacer_effective_length_mm=spacer_length,
            maximum_remaining_horizontal_capacity_mm=(
                maximum_remaining_horizontal_capacity
            ),
            asin_argument=asin_argument,
            command_angle_magnitude_deg=None,
            signed_command_angle_deg=None,
            feasible=False,
            reason="triangular spacer asin argument is nonfinite",
        )
    if not -1.0 <= asin_argument <= 1.0:
        return TriangularSpacerGeometry(
            classification=classification,
            real_world_tilted_side=real_world_tilted_side,
            book_width_mm=width,
            clearance_mm=clearance,
            base_required_opening_mm=base_required_opening,
            triangular_extra_opening_mm=opening_calibration,
            calibrated_required_opening_mm=calibrated_required_opening,
            book_height_mm=book_height,
            spacer_contact_height_mm=contact_height,
            boundary_tilt_deg=boundary_tilt,
            hand_geometry_horizontal_offset_mm=hand_geometry_offset,
            geometry_offset_angle_deg=geometry_offset_angle,
            contact_angle_deg=contact_angle,
            push_displacement_mm=push_displacement,
            spacer_effective_length_mm=spacer_length,
            maximum_remaining_horizontal_capacity_mm=(
                maximum_remaining_horizontal_capacity
            ),
            asin_argument=asin_argument,
            command_angle_magnitude_deg=None,
            signed_command_angle_deg=None,
            feasible=False,
            reason=(
                "triangular spacer asin argument is outside [-1, 1]; "
                "command is not clipped to 90 degrees"
            ),
        )

    command_magnitude = math.degrees(math.asin(asin_argument))
    signed_command = determine_spacer_command_angle(
        classification,
        command_magnitude,
    )
    return TriangularSpacerGeometry(
        classification=classification,
        real_world_tilted_side=real_world_tilted_side,
        book_width_mm=width,
        clearance_mm=clearance,
        base_required_opening_mm=base_required_opening,
        triangular_extra_opening_mm=opening_calibration,
        calibrated_required_opening_mm=calibrated_required_opening,
        book_height_mm=book_height,
        spacer_contact_height_mm=contact_height,
        boundary_tilt_deg=boundary_tilt,
        hand_geometry_horizontal_offset_mm=hand_geometry_offset,
        geometry_offset_angle_deg=geometry_offset_angle,
        contact_angle_deg=contact_angle,
        push_displacement_mm=push_displacement,
        spacer_effective_length_mm=spacer_length,
        maximum_remaining_horizontal_capacity_mm=(
            maximum_remaining_horizontal_capacity
        ),
        asin_argument=asin_argument,
        command_angle_magnitude_deg=command_magnitude,
        signed_command_angle_deg=signed_command,
        feasible=True,
        reason="triangular zero-bottleneck conservative geometry is feasible",
    )


def plan_storage_spacer(
    *,
    classification: str,
    book_width_mm: float = TEST_BOOK_WIDTH_MM,
    clearance_mm: float = STORAGE_CLEARANCE_MM,
    triangular_boundary_tilt_deg: float | None = None,
    rectangular_bottleneck_mm: float | None = None,
    rectangular_valid_row_count: int | None = None,
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

    if classification in {
        "triangular_left_tilted",
        "triangular_right_tilted",
    }:
        geometry = calculate_triangular_spacer_geometry(
            classification=classification,
            book_width_mm=width,
            clearance_mm=clearance,
            boundary_tilt_deg=triangular_boundary_tilt_deg,
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


def estimate_rectangular_bottleneck_mm(
    result_metadata: dict[str, Any],
    *,
    percentile: float = RECTANGULAR_BOTTLENECK_PERCENTILE,
    minimum_valid_rows: int = CONTACT_OPENING_MIN_VALID_ROWS,
) -> dict[str, Any]:
    """Measure all valid selected-mask row widths on the RANSAC shelf plane."""
    selected_mask = np.asarray(result_metadata.get("selected_space_mask"))
    if selected_mask.ndim != 2 or selected_mask.size == 0:
        raise RuntimeError(
            "selected_space_mask is required for rectangular bottleneck"
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
            "rectangular bottleneck estimation"
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
            "too few valid selected-mask rows for rectangular bottleneck: "
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

    bottleneck_mm = calculate_bottleneck_percentile_mm(
        valid_widths_mm,
        percentile=percentile,
        minimum_valid_rows=minimum_valid_rows,
    )
    return {
        "bottleneck_opening_mm": bottleneck_mm,
        "bottleneck_percentile": float(percentile),
        "valid_row_count": len(valid_widths_mm),
        "invalid_row_count": int(invalid_row_count),
        "valid_row_y_min": min(valid_rows),
        "valid_row_y_max": max(valid_rows),
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


def log_triangular_spacer_geometry(
    geometry: TriangularSpacerGeometry,
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
        "hand geometry horizontal offset = "
        f"{geometry.hand_geometry_horizontal_offset_mm:.2f} mm"
    )
    print(
        "geometry offset angle = "
        f"{geometry.geometry_offset_angle_deg:.2f} deg"
    )
    print(f"contact angle = {geometry.contact_angle_deg:.2f} deg")
    print(
        "push displacement at contact height = "
        f"{geometry.push_displacement_mm:.2f} mm"
    )
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
            "push displacement at contact height = "
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
        log_triangular_spacer_geometry(plan.triangular_geometry)


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

        rectangular_bottleneck_info: dict[str, Any] | None = None
        diagnostic_current_opening_mm: float | None = None
        if decision.classification.startswith("triangular_"):
            # The 100 mm opening remains diagnostic only.  Triangular control
            # instead uses the matching image-boundary tilt, book geometry,
            # side-specific contact height, and hand geometry.
            try:
                spacer_opening_info = estimate_contact_height_opening_mm(res)
                diagnostic_current_opening_mm = float(
                    spacer_opening_info["current_opening_mm"]
                )
            except RuntimeError as exc:
                print(
                    "[STORAGE SPACER DIAGNOSTIC] current opening unavailable: "
                    f"{exc}"
                )
            if decision.classification == "triangular_left_tilted":
                # Image-left boundary = real-world RIGHT tilted book.
                triangular_boundary_tilt_deg = (
                    decision.left_boundary_tilt_deg
                )
            elif decision.classification == "triangular_right_tilted":
                # Image-right boundary = real-world LEFT tilted book.
                triangular_boundary_tilt_deg = (
                    decision.right_boundary_tilt_deg
                )
            else:
                raise RuntimeError(
                    "unsupported triangular classification: "
                    f"{decision.classification}"
                )
            spacer_plan = plan_storage_spacer(
                classification=decision.classification,
                triangular_boundary_tilt_deg=(
                    triangular_boundary_tilt_deg
                ),
            )
        else:
            rectangular_bottleneck_info = (
                estimate_rectangular_bottleneck_mm(res)
            )
            spacer_plan = plan_storage_spacer(
                classification=decision.classification,
                rectangular_bottleneck_mm=float(
                    rectangular_bottleneck_info["bottleneck_opening_mm"]
                ),
                rectangular_valid_row_count=int(
                    rectangular_bottleneck_info["valid_row_count"]
                ),
            )

        log_storage_spacer_plan(
            spacer_plan,
            rectangular_bottleneck_info=rectangular_bottleneck_info,
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
        triangular_forward_plan: TriangularForwardPlan | None = None
        if decision.classification.startswith("triangular_"):
            triangular_forward_plan = plan_triangular_forward_split(
                INSERT_BOOK_FULL_DX,
            )
            log_triangular_forward_plan(triangular_forward_plan)
        spacer_angle_magnitude_deg = (
            spacer_plan.spacer_angle_magnitude_deg
        )
        spacer_command_angle_deg = spacer_plan.spacer_command_angle_deg

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
        if (
            decision.classification == "triangular_left_tilted"
            and ENABLE_XARM_LATERAL_CORRECTION
        ):
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
                shelf_forward_robot_x_mm=(
                    triangular_forward_plan.remaining_forward_mm
                ),
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
                f"{spacer_command_angle_deg:+.2f} deg (absolute)"
            )
            print(f"xArm d_roll = {decision.xarm_d_roll_rad:+.6f} rad")
            print(
                "forward split = pre-rotation "
                f"{triangular_forward_plan.pre_rotation_forward_mm:+.2f} mm "
                "+ post-rotation "
                f"{triangular_forward_plan.remaining_forward_mm:+.2f} mm "
                "= original total "
                f"{triangular_forward_plan.original_total_forward_mm:+.2f} mm, "
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
                f"{spacer_command_angle_deg:+.2f} (absolute)"
            )
            print(
                "insertion step = high reach -> pre-forward -> spacer -> "
                "remaining forward -> lower -> "
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
            print(
                "[STORAGE SPACER] no spacer motion was started; "
                "skip unnecessary linear retraction"
            )
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
        if ret != 0:
            reach_label = (
                "left-book high reaching"
                if decision.classification == "triangular_right_tilted"
                else "upright reaching"
            )
            raise RuntimeError(f"{reach_label} failed: ret={ret}")

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

        spacer_required = spacer_plan.spacer_required
        if spacer_required:
            if spacer_command_angle_deg is None:
                raise RuntimeError("geometry-based spacer command is missing")
            if triangular_forward_plan is None:
                raise RuntimeError(
                    "spacer motion requires a triangular forward plan"
                )
            if decision.classification == "triangular_right_tilted":
                pre_forward_velocity = 15
                pre_forward_acceleration = 15
            else:
                pre_forward_velocity = TCP_VEL_2
                pre_forward_acceleration = TCP_ACC_2
            pre_forward_delta = [
                triangular_forward_plan.pre_rotation_forward_mm,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            print("\n[STORAGE PRE-ROTATION FORWARD] start")
            print(
                "robot-base relative MoveL = "
                f"dX={pre_forward_delta[0]:+.2f} mm, "
                "dY=+0.00 mm, dZ=+0.00 mm, "
                "dRoll=dPitch=dYaw=0"
            )
            pre_forward_ret = xarm7.moveL_relative(
                pre_forward_delta,
                velocity=pre_forward_velocity,
                acceleration=pre_forward_acceleration,
                asynchronous=False,
            )
            print(
                "[STORAGE PRE-ROTATION FORWARD] MoveL ret =",
                pre_forward_ret,
            )
            if pre_forward_ret != 0:
                raise RuntimeError(
                    "pre-rotation forward failed before spacer motion: "
                    f"ret={pre_forward_ret}"
                )
            pre_forward_xarm_state = xarm7.get_state()
            pre_forward_xarm_error = int(xarm7.arm.error_code)
            pre_forward_xarm_warn = int(xarm7.arm.warn_code)
            print(
                "[STORAGE PRE-ROTATION FORWARD] xArm check: "
                f"state={pre_forward_xarm_state}, "
                f"error={pre_forward_xarm_error}, "
                f"warn={pre_forward_xarm_warn}"
            )
            if (
                pre_forward_xarm_state in (4, 5)
                or pre_forward_xarm_error != 0
            ):
                raise RuntimeError(
                    "xArm abnormal after pre-rotation forward: "
                    f"state={pre_forward_xarm_state}, "
                    f"error={pre_forward_xarm_error}, "
                    f"warn={pre_forward_xarm_warn}"
                )
            print(
                "[STORAGE PRE-ROTATION FORWARD] completed; "
                "start spacer expansion/rotation"
            )
            print("[DEBUG] before expand_sp_lin")
            hand_worker.expand_sp_lin(asynchronous=True)
            print("[DEBUG] after expand_sp_lin")
            time.sleep(14.0)
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
            elif decision.classification == "triangular_right_tilted":
                print(
                    "[STORAGE SIDE MAPPING] image-right tilted boundary = "
                    "real-world left tilted book"
                )
            hand_worker.rotate_spacer(spacer_command_angle_deg)
            print(
                "[STORAGE SPACER] do not issue reset while inserting; "
                "the existing rotate_spacer torque policy is unchanged"
            )
        else:
            print(
                "[STORAGE SPACER] opening is sufficient: skip spacer "
                "expansion/rotation; rotate_spacer(0) is not called"
            )

        spacer_rotation_reset = False
        if decision.classification == "triangular_left_tilted":
            if ENABLE_XARM_LATERAL_CORRECTION:
                if left_tilted_correction_delta is None:
                    raise RuntimeError("left-tilted correction delta is missing")
                print(
                    "[STORAGE UPRIGHT CORRECTION] simultaneous robot-base "
                    "MoveL: "
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
            else:
                print(
                    "[STORAGE SPACER ANGLE EXPERIMENT] xArm lateral "
                    "correction disabled; use remaining forward-only insertion"
                )
                if triangular_forward_plan is None:
                    raise RuntimeError("triangular forward plan is missing")
                remaining_forward_delta = [
                    triangular_forward_plan.remaining_forward_mm,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                ret = xarm7.moveL_relative(
                    remaining_forward_delta,
                    velocity=TCP_VEL_2,
                    acceleration=TCP_ACC_2,
                    asynchronous=False,
                )
            if ret != 0:
                raise RuntimeError(
                    f"right-book forward insertion failed: ret={ret}"
                )
        elif decision.classification == "triangular_right_tilted":
            if (
                left_book_reach_z_offset_mm is None
                or left_book_right_side_target_robot_mm is None
            ):
                raise RuntimeError(
                    "left-book high-reach target or Z offset is missing"
                )
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
            else:
                print(
                    "[STORAGE SPACER ANGLE EXPERIMENT] xArm +Y 5 mm "
                    "lateral correction disabled; start forward insertion"
                )
            print(
                "[LEFT-BOOK-TILTED HIGH REACH] step 1/2: advance the upright "
                "book while the spacer opens the space at the high position"
            )
            if triangular_forward_plan is None:
                raise RuntimeError("triangular forward plan is missing")
            print(
                "[LEFT-BOOK-TILTED HIGH REACH] forward command: "
                "remaining relative MoveL after spacer rotation, "
                "robot dX="
                f"{triangular_forward_plan.remaining_forward_mm:+.2f} mm"
            )
            remaining_forward_delta = [
                triangular_forward_plan.remaining_forward_mm,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            forward_ret = xarm7.moveL_relative(
                remaining_forward_delta,
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
        if ret != 0:
            raise RuntimeError(f"full upright insertion failed: ret={ret}")

        time.sleep(2.0)
        if spacer_required:
            reset_spacer_rotation_before_linear_retraction(hand_worker)
            spacer_rotation_reset = True
            print("[STORAGE SPACER] contract_sp_lin_1 start")
            hand_worker.contract_sp_lin_1(asynchronous=False)
            print("[DEBUG] after contract_sp_lin_1")
        else:
            print(
                "[STORAGE SPACER] unused in rectangular space: "
                "skip reset_rot and contract_sp_lin_1"
            )

        ret = xarm7.move_L_to_insert_book_tip(
            velocity=15,
            acceleration=15,
            asynchronous=False,
        )
        print("[DEBUG] move_L_to_insert_book_tip ret =", ret)
        if ret != 0:
            raise RuntimeError(f"move_L_to_insert_book_tip failed: ret={ret}")

        time.sleep(0.5)
        if spacer_required:
            if not spacer_rotation_reset:
                raise RuntimeError(
                    "spacer linear retraction requested without rotation reset"
                )
            print("[STORAGE SPACER] contract_sp_lin_2 start")
            hand_worker.contract_sp_lin_2(asynchronous=False)
        else:
            print(
                "[STORAGE SPACER] unused in rectangular space: "
                "skip contract_sp_lin_2"
            )
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
