from __future__ import annotations

from typing import Any
import numpy as np

from storage.config import BOUNDARY_TILT_DIFFERENCE_THRESHOLD_DEG

from dataclasses import dataclass


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
