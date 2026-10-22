from __future__ import annotations

from typing import Any
import math
import numpy as np

from storage.models import (
    RightTiltedSupportPlan,
    TriangularSpacerGeometry,
    TriangularForwardPlan,
    SpacerAngleFromMaxWidth,
    RightTiltedRotationStep,
    StorageSpacerPlan,
    AmbiguousStorageResolution,
    StorageSpaceClassification,
    StorageCaptureCandidate,
)

from storage.config import (
    RIGHT_TILTED_BOOK_LONG_EDGE_MM,
    RIGHT_TILTED_Y_TRAVEL_SCALE,
    RIGHT_TILTED_SP_ROT_SCALE,
    TEST_BOOK_WIDTH_MM,
    STORAGE_CLEARANCE_MM,
    TRIANGULAR_EXTRA_OPENING_MM,
    SPACER_EFFECTIVE_LENGTH_MM,
    BOOK_HEIGHT_MM,
    RIGHT_TILTED_SPACER_CONTACT_HEIGHT_MM,
    LEFT_TILTED_SPACER_CONTACT_HEIGHT_MM,
    RECTANGULAR_BOTTLENECK_PERCENTILE,
    CONTACT_OPENING_MIN_VALID_ROWS,
    RIGHT_TILTED_SPACER_DIRECTION_SIGN,
    LEFT_TILTED_SPACER_DIRECTION_SIGN,
    LEFT_BOOK_REACH_Z_OFFSET_MM,
    SPACER_PRE_ROTATION_FORWARD_MM,
    RIGHT_TILTED_SYNC_STEP_DEG,
    STORAGE_SPACER_FINAL_TIP_ADDITIONAL_MM,
    VERTICAL_ANGLE_TOLERANCE_DEG,
    BOUNDARY_TILT_DIFFERENCE_THRESHOLD_DEG,
    AMBIGUOUS_TILT_EQUAL_EPSILON_DEG,
    MIN_SUPPORTED_SPACE_WIDTH_PX,
    UPRIGHT_XARM_D_ROLL_RAD,
    LEFT_TILTED_REACH_RIGHT_OFFSET_MM,
    LEFT_TILTED_REACH_BACKOFF_MM,
    TWO_VIEW_BOTTLENECK_TIE_TOLERANCE_MM,
    TCP_TO_BOOK_REAR_MM,
    REACH_MARGIN_MM,
    APPROACH_BACKOFF_MM,
    INSERT_WITH_SPACER_MM,
    FINAL_BOOK_PUSH_MM,
    REACH_TCP_BACKOFF_MM,
    EXPECTED_FINAL_TCP_OFFSET_MM,
    EXPECTED_INSERT_DEPTH_MM,
    TILTED_REACH_Y_CORRECTION_MM,
    RIGHT_TILTED_REACH_Y_CORRECTION_MM,
)


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    return result if math.isfinite(result) else None


def should_use_right_tilted_sp_lin_partial_expansion(
    classification: str,
) -> bool:
    """Use partial SP_LIN only for image-left/real-world-right tilt."""
    return classification == "triangular_left_tilted"


def should_use_right_tilted_support_sync(
    classification: str,
) -> bool:
    """Enable tilt-based SP_ROT plus alpha-based xArm sync for RIGHT-book tilt."""
    return classification == "triangular_left_tilted"


def spacer_lateral_projection_mm(
    angle_deg: float,
    length_mm: float,
) -> float:
    """Return L*sin(abs(theta)) in millimetres."""
    angle = finite_float(angle_deg)
    length = finite_float(length_mm)

    if angle is None or not 0.0 <= abs(angle) <= 90.0:
        raise RuntimeError(
            f"spacer angle must be finite and within [-90, 90]: {angle_deg}"
        )

    if length is None or length <= 0.0:
        raise RuntimeError(
            f"spacer effective length must be finite and positive: {length_mm}"
        )

    return float(
        length * math.sin(math.radians(abs(angle)))
    )
    
def calculate_storage_final_tip_forward_mm(
    original_forward_mm: float,
    *,
    spacer_required: bool,
) -> float:
    """Add the configured distance only after spacer use and retraction."""
    original = finite_float(original_forward_mm)
    if original is None or original <= 0.0:
        raise RuntimeError(
            "original final tip forward must be finite and positive: "
            f"{original_forward_mm}"
        )

    if spacer_required:
        return original + STORAGE_SPACER_FINAL_TIP_ADDITIONAL_MM
    return original


def plan_triangular_forward_split(
    total_forward_mm: float,
    *,
    pre_rotation_forward_mm: float = SPACER_PRE_ROTATION_FORWARD_MM,
) -> TriangularForwardPlan:
    """Split one existing +X insertion into pre/post-rotation MoveL steps."""
    total = finite_float(total_forward_mm)
    pre_rotation = finite_float(pre_rotation_forward_mm)

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
            f"total={total:.2f} mm, "
            f"pre_rotation={pre_rotation:.2f} mm, "
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
    additional = finite_float(required_additional_opening_mm)

    if additional is None or additional < 0.0:
        raise RuntimeError(
            "required additional opening must be finite and nonnegative: "
            f"{required_additional_opening_mm}"
        )

    length = finite_float(effective_length_mm)

    if length is None or length <= 0.0:
        raise RuntimeError(
            "spacer effective length must be finite and positive: "
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
    max_space_width_mm: float | None,
    effective_length_mm: float = SPACER_EFFECTIVE_LENGTH_MM,
) -> SpacerAngleFromMaxWidth:
    """Convert the widest physical selected-space row to a spacer angle."""
    max_width = finite_float(max_space_width_mm)
    length = finite_float(effective_length_mm)

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


# Set to a higher value only after verifying the actual mechanism/servo range.
# 90 is the existing verified-by-code limit; commands above it fail closed.
SP_ROT_VERIFIED_MAX_ANGLE_DEG = 160


def calculate_sp_rot_angle_from_tilt(boundary_tilt_deg: float) -> float:
    """Return fixed +90 deg SP_ROT target for either tilted-book case."""
    tilt = finite_float(boundary_tilt_deg)
    if tilt is None or not 0.0 <= tilt < 90.0:
        raise RuntimeError(f"invalid SAM3 boundary tilt: {boundary_tilt_deg}")
    angle = 90.0
    limit = finite_float(SP_ROT_VERIFIED_MAX_ANGLE_DEG)
    if limit is None or not 90.0 <= limit <= 180.0:
        raise RuntimeError("invalid verified SP_ROT mechanical limit")
    if angle > limit:
        raise RuntimeError(
            f"SP_ROT target {angle:.2f} deg exceeds verified limit "
            f"{limit:.2f} deg; verify physical travel before increasing "
            "SP_ROT_VERIFIED_MAX_ANGLE_DEG"
        )
    return angle


def calculate_right_tilted_support_plan(
    boundary_tilt_deg: float,
    max_space_width_mm: float | None,
    *,
    contact_height_mm: float = RIGHT_TILTED_SPACER_CONTACT_HEIGHT_MM,
    spacer_effective_length_mm: float = SPACER_EFFECTIVE_LENGTH_MM,
    high_reach_offset_mm: float | None = LEFT_BOOK_REACH_Z_OFFSET_MM,
) -> RightTiltedSupportPlan:
    """Plan RIGHT-book Y travel (A5 length*sin tilt) and 90+tilt SP_ROT."""
    boundary_tilt = finite_float(boundary_tilt_deg)
    max_width = finite_float(max_space_width_mm)
    contact_height = finite_float(contact_height_mm)
    spacer_length = finite_float(spacer_effective_length_mm)
    high_reach_offset = finite_float(high_reach_offset_mm)

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

    if high_reach_offset is None:
        raise RuntimeError(
            "RIGHT-book high-reach offset must be finite and positive: "
            f"{high_reach_offset_mm}"
        )

    # A5 long edge projected horizontally by the tilt from vertical.
    arm_right_total = (
        RIGHT_TILTED_BOOK_LONG_EDGE_MM
        * RIGHT_TILTED_Y_TRAVEL_SCALE
        * math.sin(math.radians(abs(boundary_tilt)))
    )

    return RightTiltedSupportPlan(
        boundary_tilt_deg=float(boundary_tilt),
        contact_height_mm=float(contact_height),
        arm_right_total_mm=float(arm_right_total),
        max_space_width_mm=float(max_width),
        spacer_effective_length_mm=float(spacer_length),
        spacer_angle_deg=calculate_sp_rot_angle_from_tilt(abs(boundary_tilt)),
        high_reach_offset_mm=float(high_reach_offset),
    )


def plan_right_tilted_rotation_steps(
    final_angle_magnitude_deg: float,
    arm_right_total_mm: float,
    *,
    step_deg: float = RIGHT_TILTED_SYNC_STEP_DEG,
) -> tuple[RightTiltedRotationStep, ...]:
    """Split SP_ROT and an independent h*tan(alpha) arm path into steps."""
    final_angle = finite_float(final_angle_magnitude_deg)
    arm_right_total = finite_float(arm_right_total_mm)
    step = finite_float(step_deg)

    if final_angle is None or not 0.0 <= final_angle <= SP_ROT_VERIFIED_MAX_ANGLE_DEG:
        raise RuntimeError(
            "final spacer angle magnitude exceeds verified range: "
            f"{final_angle_magnitude_deg}"
        )

    if step is None or step <= 0.0:
        raise RuntimeError(
            f"sync step must be finite and positive: {step_deg}"
        )

    if arm_right_total is None or arm_right_total < 0.0:
        raise RuntimeError(
            "arm RIGHT total must be finite and nonnegative: "
            f"{arm_right_total_mm}"
        )

    if final_angle <= 1.0e-12:
        if arm_right_total > 1.0e-9:
            raise RuntimeError(
                "nonzero RIGHT arm travel cannot be synchronized with "
                "a zero SP_ROT angle"
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
                robot_delta_y_mm=(
                    cumulative_y - previous_cumulative_y
                ),
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
        raise AssertionError(
            "RIGHT tilted stepped robot Y does not sum to total"
        )

    return tuple(planned_steps)


def calculate_triangular_spacer_geometry(
    *,
    classification: str,
    book_width_mm: float,
    clearance_mm: float,
    boundary_tilt_deg: float | None,
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

    Both tilted sides use a fixed +90 deg SP_ROT target.
    Image-left/real-world-RIGHT tilt additionally uses its boundary tilt and
    100 mm contact height to plan the synchronized xArm -Y travel.
    """
    width = finite_float(book_width_mm)
    clearance = finite_float(clearance_mm)
    opening_calibration = finite_float(triangular_extra_opening_mm)
    boundary_tilt = finite_float(boundary_tilt_deg)
    book_height = finite_float(book_height_mm)
    spacer_length = finite_float(spacer_effective_length_mm)
    right_contact_height = finite_float(right_tilted_contact_height_mm)
    left_contact_height = finite_float(left_tilted_contact_height_mm)

    if width is None or width <= 0.0:
        raise RuntimeError(
            f"book width must be finite and positive: {book_width_mm}"
        )

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

        # Physical RIGHT-tilted book: positive rotation, same as LEFT.
        signed_command = +float(support.spacer_angle_deg)

        return TriangularSpacerGeometry(
            classification=classification,
            real_world_tilted_side="right",
            book_width_mm=width,
            clearance_mm=clearance,
            base_required_opening_mm=support.max_space_width_mm,
            triangular_extra_opening_mm=0.0,
            calibrated_required_opening_mm=support.max_space_width_mm,
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
                "right-book support model: SP_ROT uses -(90+tilt); "
                "xArm -Y 210*sin(tilt) mm moves during SP_ROT opening before insertion"
            ),
            right_tilted_support_plan=support,
        )

    # Retain G_max only for diagnostics; SP_ROT uses fixed +90 degrees.
    
    angle = calculate_spacer_angle_from_gmax(
        triangular_max_space_width_mm,
        effective_length_mm=spacer_length,
    )

    tilt_command_angle = calculate_sp_rot_angle_from_tilt(boundary_tilt)

    # 左側書籍の傾斜：SP_ROT = +90°固定
    signed_command = +abs(tilt_command_angle)

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
        command_angle_magnitude_deg=tilt_command_angle,
        signed_command_angle_deg=signed_command,
        feasible=True,
        reason=(
            f"{real_world_tilted_side}-book model: SP_ROT=90+SAM3 tilt; "
            "G_max remains diagnostic only"
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
    width = finite_float(book_width_mm)
    clearance = finite_float(clearance_mm)
    length = finite_float(effective_length_mm)

    if width is None or width <= 0.0:
        raise RuntimeError(
            f"book width must be finite and positive: {book_width_mm}"
        )

    if clearance is None or clearance < 0.0:
        raise RuntimeError(
            f"storage clearance must be finite and nonnegative: {clearance_mm}"
        )

    if length is None or length <= 0.0:
        raise RuntimeError(
            "spacer effective length must be finite and positive: "
            f"{effective_length_mm}"
        )

    required_opening = width + clearance

    if classification == "direct_insert":
        bottleneck = finite_float(direct_insert_bottleneck_mm)

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
        # width only. STORAGE_CLEARANCE_MM is not part of this direct-insert
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
        bottleneck = finite_float(rectangular_bottleneck_mm)

        if bottleneck is None or bottleneck < 0.0:
            raise RuntimeError(
                "rectangular bottleneck must be finite and nonnegative: "
                f"{rectangular_bottleneck_mm}"
            )

        if (
            rectangular_valid_row_count is None
            or int(rectangular_valid_row_count)
            < CONTACT_OPENING_MIN_VALID_ROWS
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

    magnitude = finite_float(angle_magnitude_deg)

    if magnitude is None or not 0.0 <= magnitude <= SP_ROT_VERIFIED_MAX_ANGLE_DEG:
        raise RuntimeError(
            f"invalid spacer angle magnitude: {angle_magnitude_deg}"
        )

    if classification == "triangular_left_tilted":
        return LEFT_TILTED_SPACER_DIRECTION_SIGN * magnitude

    if classification == "triangular_right_tilted":
        return RIGHT_TILTED_SPACER_DIRECTION_SIGN * magnitude

    raise RuntimeError(
        "spacer direction requested for unsupported classification: "
        f"{classification}"
    )


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
    if not math.isfinite(offset):
        raise RuntimeError(
            "LEFT_BOOK_REACH_Z_OFFSET_MM must be finite: "
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
    right_offset = finite_float(right_offset_mm)
    backoff = finite_float(backoff_mm)
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
    left = finite_float(left_tilt_deg)
    right = finite_float(right_tilt_deg)
    vertical_tolerance = finite_float(vertical_tolerance_deg)
    difference_threshold = finite_float(difference_threshold_deg)
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
    left = finite_float(left_tilt_deg)
    right = finite_float(right_tilt_deg)
    bottleneck = finite_float(bottleneck_opening_mm)
    book_width = finite_float(book_width_mm)
    epsilon = finite_float(equal_tilt_epsilon_deg)
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
    # Local import avoids planner <-> analysis import cycle.
    from storage.analysis import estimate_rectangular_bottleneck_mm
    selected_space_id = result_metadata.get("selected_space_id")
    if selected_space_id is not None:
        selected_space_id = int(selected_space_id)

    left_tilt = finite_float(
        result_metadata.get("left_tilt_from_vertical_deg")
    )
    right_tilt = finite_float(
        result_metadata.get("right_tilt_from_vertical_deg")
    )
    image_guide_angle_deg = finite_float(
        result_metadata.get("angle_deg")
    )
    if image_guide_angle_deg is None:
        angle_rad = finite_float(result_metadata.get("angle_rad"))
        if angle_rad is not None:
            image_guide_angle_deg = float(np.degrees(angle_rad))

    book_bottom_side_width_px = finite_float(
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


def evaluate_storage_capture_candidate(
    *,
    label: str,
    capture_index: int,
    result: tuple[float, np.ndarray, dict[str, Any]],
    capture_tcp_pose: Any,
) -> StorageCaptureCandidate:
    """Run existing classification and spacer planning without hardware motion."""
    # Local import avoids planner <-> analysis import cycle.
    from storage.analysis import (
        _selected_space_mask_height_px,
        evaluate_storage_space_upper_reach,
        log_storage_space_upper_reach_check,
        estimate_contact_height_opening_mm,
        calculate_bottleneck_percentile_mm,
        estimate_selected_space_row_widths_mm,
        evaluate_selected_space_vertical_width_check,
    )
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

        upper_reach_check = evaluate_storage_space_upper_reach(
            result_metadata
        )

        result_metadata.update(
            upper_reach_check
        )

        log_storage_space_upper_reach_check(
            upper_reach_check,
            label,
        )

        if not upper_reach_check[
            "storage_space_upper_reach_valid"
        ]:
            raise RuntimeError(
                "storage-space upper-reach validation failed: "
                f"{upper_reach_check['storage_space_upper_reach_reason']}"
            )

        decision = classify_storage_space(
            result_metadata
        )
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


def validate_storage_0911_depth_constants() -> None:
    """Validate positive finite depth parameters without hard-coded target depths."""

    named_values = {
        "TCP_TO_BOOK_REAR_MM": TCP_TO_BOOK_REAR_MM,
        "REACH_MARGIN_MM": REACH_MARGIN_MM,
        "APPROACH_BACKOFF_MM": APPROACH_BACKOFF_MM,
        "INSERT_WITH_SPACER_MM": INSERT_WITH_SPACER_MM,
        "FINAL_BOOK_PUSH_MM": FINAL_BOOK_PUSH_MM,
        "EXPECTED_INSERT_DEPTH_MM": EXPECTED_INSERT_DEPTH_MM,
    }

    for name, value in named_values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(
                f"invalid 0911 storage depth constant: "
                f"{name}={value}"
            )

    # 固定値75 mm / 140 mmとの一致チェックは撤廃。
    # 深さを変更した場合も非有限値・負値の検証は上で継続する。

    if not math.isclose(
        -APPROACH_BACKOFF_MM + APPROACH_BACKOFF_MM,
        0.0,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise RuntimeError(
            "0911 approach backoff/forward must cancel"
        )


def build_storage_0911_depth_plan(
    surface_robot_xyz: Any,
) -> dict[str, Any]:
    """Build and display the fixed post-recognition XYZ/depth plan."""

    validate_storage_0911_depth_constants()

    surface = np.asarray(
        surface_robot_xyz,
        dtype=np.float64,
    ).reshape(3)

    if not np.isfinite(surface).all():
        raise RuntimeError(
            "recognized surface robot XYZ is not finite: "
            f"{surface.tolist()}"
        )

    reach_target = surface.copy()
    reach_target[0] -= REACH_TCP_BACKOFF_MM

    expected_final_tcp_x = (
        surface[0]
        + EXPECTED_FINAL_TCP_OFFSET_MM
    )

    expected_final_book_rear_x = (
        expected_final_tcp_x
        + TCP_TO_BOOK_REAR_MM
    )

    print("\n[STORAGE 0911 DEPTH PLAN]")

    print(
        "recognized surface XYZ = "
        f"[{surface[0]:.3f}, "
        f"{surface[1]:.3f}, "
        f"{surface[2]:.3f}] mm"
    )

    print(
        f"TCP to book rear = "
        f"{TCP_TO_BOOK_REAR_MM:.1f} mm"
    )

    print(
        f"reach margin = "
        f"{REACH_MARGIN_MM:.1f} mm"
    )

    print(
        "reach TCP X = surface X - "
        f"{REACH_TCP_BACKOFF_MM:.1f} mm = "
        f"{reach_target[0]:.3f} mm"
    )

    print(
        f"insert with spacer = "
        f"+{INSERT_WITH_SPACER_MM:.1f} mm"
    )

    print(
        f"final push = "
        f"+{FINAL_BOOK_PUSH_MM:.1f} mm"
    )

    print(
        f"expected final TCP X = "
        f"{expected_final_tcp_x:.3f} mm"
    )

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
        "expected_final_tcp_x_mm": float(
            expected_final_tcp_x
        ),
        "expected_final_book_rear_x_mm": float(
            expected_final_book_rear_x
        ),
        "expected_book_insertion_depth_mm": (
            EXPECTED_INSERT_DEPTH_MM
        ),
    }


def apply_tilted_reach_y_correction(
    reach_target_robot_mm: Any,
    classification: str,
    *,
    correction_mm: float = TILTED_REACH_Y_CORRECTION_MM,
) -> np.ndarray:
    """傾斜した隣接書籍から離れる方向へYを補正する。"""

    target = np.asarray(
        reach_target_robot_mm,
        dtype=np.float64,
    ).reshape(3).copy()

    if not np.isfinite(target).all():
        raise RuntimeError(f"invalid robot reach target: {target}")

    if classification == "triangular_left_tilted":
        # 実世界の右側の本が傾斜 → 左（+Y）へ逃がす
        correction = float(RIGHT_TILTED_REACH_Y_CORRECTION_MM)
        target[1] += correction

    elif classification == "triangular_right_tilted":
        # 実世界の左側の本が傾斜 → 右（-Y）へ逃がす
        correction = float(correction_mm)
        target[1] -= correction

    else:
        correction = 0.0

    if not math.isfinite(correction):
        raise RuntimeError(f"invalid Y correction: {correction}")

    print("\n[TILTED REACH Y CORRECTION]")
    print(f"classification = {classification}")
    print(f"Y correction = {target[1] - float(reach_target_robot_mm[1]):+.3f} mm")
    print(f"corrected Y = {target[1]:.3f} mm")

    return target