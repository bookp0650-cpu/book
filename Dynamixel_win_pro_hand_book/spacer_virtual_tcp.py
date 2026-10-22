"""Pure geometry for a Python-side virtual spacer-tip TCP.

This module deliberately has no xArm, Dynamixel, ROS, or RealSense imports.
All vectors are expressed in millimetres.  TCP roll/pitch/yaw values are in
radians and follow the existing xArm code's ``Rotation.from_euler("xyz")``
convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
from scipy.spatial.transform import Rotation


class SpacerVirtualTcpCalibrationError(ValueError):
    """Raised when virtual-TCP geometry is absent or physically incomplete."""


def _finite_vector3(name: str, value: Any) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError) as exc:
        raise SpacerVirtualTcpCalibrationError(
            f"{name} must contain exactly three numeric values"
        ) from exc
    if not np.isfinite(vector).all():
        raise SpacerVirtualTcpCalibrationError(
            f"{name} must contain three finite values: {vector.tolist()}"
        )
    return vector


def _unit_vector3(name: str, value: Any) -> np.ndarray:
    vector = _finite_vector3(name, value)
    norm = float(np.linalg.norm(vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise SpacerVirtualTcpCalibrationError(
            f"{name} must be a calibrated unit vector, norm={norm}"
        )
    return vector


@dataclass(frozen=True)
class SpacerVirtualTcpGeometry:
    """Measured spacer geometry expressed in the configured xArm TCP frame.

    ``sp_lin_axis_tcp_unit`` points in the physical direction produced by an
    increase in the raw SP_LIN position.  ``sp_rot_axis_tcp_unit`` is oriented
    so that a positive application angle obeys the right-hand rule.  Neither
    direction may be inferred from motor command signs.
    """

    tcp_to_rot_axis_reference_mm: tuple[float, float, float]
    sp_lin_axis_tcp_unit: tuple[float, float, float]
    sp_lin_reference_steps: float
    sp_lin_mm_per_step: float
    sp_rot_axis_tcp_unit: tuple[float, float, float]
    rot_axis_to_tip_zero_mm: tuple[float, float, float]
    calibration_id: str

    def __post_init__(self) -> None:
        _finite_vector3(
            "tcp_to_rot_axis_reference_mm",
            self.tcp_to_rot_axis_reference_mm,
        )
        _unit_vector3("sp_lin_axis_tcp_unit", self.sp_lin_axis_tcp_unit)
        _unit_vector3("sp_rot_axis_tcp_unit", self.sp_rot_axis_tcp_unit)
        tip_vector = _finite_vector3(
            "rot_axis_to_tip_zero_mm",
            self.rot_axis_to_tip_zero_mm,
        )
        if float(np.linalg.norm(tip_vector)) <= 0.0:
            raise SpacerVirtualTcpCalibrationError(
                "rot_axis_to_tip_zero_mm must be a measured non-zero vector"
            )
        reference_steps = float(self.sp_lin_reference_steps)
        millimetres_per_step = float(self.sp_lin_mm_per_step)
        if not math.isfinite(reference_steps):
            raise SpacerVirtualTcpCalibrationError(
                "sp_lin_reference_steps must be finite"
            )
        if not math.isfinite(millimetres_per_step) or millimetres_per_step <= 0.0:
            raise SpacerVirtualTcpCalibrationError(
                "sp_lin_mm_per_step must be a measured positive value"
            )
        if not str(self.calibration_id).strip():
            raise SpacerVirtualTcpCalibrationError(
                "calibration_id must identify the measured geometry"
            )


@dataclass(frozen=True)
class SpacerVirtualTcpMechanismState:
    """Measured mechanism state used for one virtual-TCP calculation."""

    sp_lin_position_steps: float
    sp_rot_angle_deg: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.sp_lin_position_steps)):
            raise ValueError("sp_lin_position_steps must be finite")
        if not math.isfinite(float(self.sp_rot_angle_deg)):
            raise ValueError("sp_rot_angle_deg must be finite")


@dataclass(frozen=True)
class SpacerVirtualTcpSolution:
    """Forward/inverse result for a desired spacer-tip position."""

    tip_offset_tcp_mm: tuple[float, float, float]
    current_tip_robot_mm: tuple[float, float, float]
    desired_tip_robot_mm: tuple[float, float, float]
    required_tcp_robot_mm: tuple[float, float, float]
    tcp_compensation_robot_mm: tuple[float, float, float]


def rotation_matrix_about_axis(
    axis_unit: Any,
    angle_rad: float,
) -> np.ndarray:
    """Return an arbitrary-axis Rodrigues rotation matrix."""
    axis = _unit_vector3("rotation_axis_unit", axis_unit)
    angle = float(angle_rad)
    if not math.isfinite(angle):
        raise ValueError(f"angle_rad must be finite: {angle_rad}")
    x, y, z = axis
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    identity = np.eye(3, dtype=np.float64)
    return identity + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (
        skew @ skew
    )


def calculate_sp_lin_displacement_mm(
    geometry: SpacerVirtualTcpGeometry,
    sp_lin_position_steps: float,
) -> float:
    """Convert measured SP_LIN position to signed calibrated displacement."""
    position = float(sp_lin_position_steps)
    if not math.isfinite(position):
        raise ValueError("sp_lin_position_steps must be finite")
    return (
        position - float(geometry.sp_lin_reference_steps)
    ) * float(geometry.sp_lin_mm_per_step)


def calculate_spacer_tip_offset_tcp(
    geometry: SpacerVirtualTcpGeometry,
    *,
    sp_lin_position_steps: float,
    sp_rot_angle_deg: float,
) -> np.ndarray:
    """Return ``p_tip_tcp(q, theta)`` for measured mechanism state."""
    angle_deg = float(sp_rot_angle_deg)
    if not math.isfinite(angle_deg):
        raise ValueError("sp_rot_angle_deg must be finite")
    pivot = _finite_vector3(
        "tcp_to_rot_axis_reference_mm",
        geometry.tcp_to_rot_axis_reference_mm,
    )
    pivot = pivot + _unit_vector3(
        "sp_lin_axis_tcp_unit",
        geometry.sp_lin_axis_tcp_unit,
    ) * calculate_sp_lin_displacement_mm(geometry, sp_lin_position_steps)
    rotation = rotation_matrix_about_axis(
        geometry.sp_rot_axis_tcp_unit,
        math.radians(angle_deg),
    )
    radius = _finite_vector3(
        "rot_axis_to_tip_zero_mm",
        geometry.rot_axis_to_tip_zero_mm,
    )
    return pivot + rotation @ radius


def tcp_rpy_rotation_matrix(tcp_rpy_rad: Any) -> np.ndarray:
    """Match the repository's base-from-TCP xyz Euler convention."""
    rpy = _finite_vector3("tcp_rpy_rad", tcp_rpy_rad)
    return Rotation.from_euler("xyz", rpy, degrees=False).as_matrix()


def calculate_spacer_tip_position_robot(
    tcp_pose_mm_rad: Any,
    geometry: SpacerVirtualTcpGeometry,
    *,
    sp_lin_position_steps: float,
    sp_rot_angle_deg: float,
) -> np.ndarray:
    """Apply ``p_tip_base = p_tcp_base + R_base_tcp p_tip_tcp``."""
    pose = np.asarray(tcp_pose_mm_rad, dtype=np.float64).reshape(6)
    if not np.isfinite(pose).all():
        raise ValueError("tcp_pose_mm_rad must contain six finite values")
    offset_tcp = calculate_spacer_tip_offset_tcp(
        geometry,
        sp_lin_position_steps=sp_lin_position_steps,
        sp_rot_angle_deg=sp_rot_angle_deg,
    )
    return pose[:3] + tcp_rpy_rotation_matrix(pose[3:]) @ offset_tcp


def calculate_required_tcp_position_robot(
    desired_tip_robot_mm: Any,
    tcp_rpy_rad: Any,
    geometry: SpacerVirtualTcpGeometry,
    *,
    sp_lin_position_steps: float,
    sp_rot_angle_deg: float,
) -> np.ndarray:
    """Inverse virtual TCP: desired tip XYZ to required physical TCP XYZ."""
    desired_tip = _finite_vector3(
        "desired_tip_robot_mm",
        desired_tip_robot_mm,
    )
    offset_tcp = calculate_spacer_tip_offset_tcp(
        geometry,
        sp_lin_position_steps=sp_lin_position_steps,
        sp_rot_angle_deg=sp_rot_angle_deg,
    )
    return desired_tip - tcp_rpy_rotation_matrix(tcp_rpy_rad) @ offset_tcp


def calculate_virtual_tcp_solution(
    tcp_pose_mm_rad: Any,
    desired_tip_robot_mm: Any,
    geometry: SpacerVirtualTcpGeometry,
    *,
    sp_lin_position_steps: float,
    sp_rot_angle_deg: float,
) -> SpacerVirtualTcpSolution:
    """Compute current tip, inverse TCP target, and base-frame dXYZ."""
    pose = np.asarray(tcp_pose_mm_rad, dtype=np.float64).reshape(6)
    if not np.isfinite(pose).all():
        raise ValueError("tcp_pose_mm_rad must contain six finite values")
    desired_tip = _finite_vector3(
        "desired_tip_robot_mm",
        desired_tip_robot_mm,
    )
    offset_tcp = calculate_spacer_tip_offset_tcp(
        geometry,
        sp_lin_position_steps=sp_lin_position_steps,
        sp_rot_angle_deg=sp_rot_angle_deg,
    )
    current_tip = pose[:3] + tcp_rpy_rotation_matrix(pose[3:]) @ offset_tcp
    required_tcp = calculate_required_tcp_position_robot(
        desired_tip,
        pose[3:],
        geometry,
        sp_lin_position_steps=sp_lin_position_steps,
        sp_rot_angle_deg=sp_rot_angle_deg,
    )
    compensation = required_tcp - pose[:3]
    return SpacerVirtualTcpSolution(
        tip_offset_tcp_mm=tuple(offset_tcp.tolist()),
        current_tip_robot_mm=tuple(current_tip.tolist()),
        desired_tip_robot_mm=tuple(desired_tip.tolist()),
        required_tcp_robot_mm=tuple(required_tcp.tolist()),
        tcp_compensation_robot_mm=tuple(compensation.tolist()),
    )


def calculate_right_tilted_desired_tip_position(
    initial_tip_robot_mm: Any,
    *,
    contact_height_mm: float,
    boundary_tilt_deg: float,
    rotation_progress: float,
) -> np.ndarray:
    """Return cumulative RIGHT-tip target (robot-base negative Y)."""
    initial_tip = _finite_vector3(
        "initial_tip_robot_mm",
        initial_tip_robot_mm,
    )
    height = float(contact_height_mm)
    tilt = float(boundary_tilt_deg)
    progress = float(rotation_progress)
    if not math.isfinite(height) or height < 0.0:
        raise ValueError("contact_height_mm must be finite and nonnegative")
    if not math.isfinite(tilt) or abs(tilt) >= 90.0:
        raise ValueError("boundary_tilt_deg must be finite with abs < 90")
    if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError("rotation_progress must be within [0, 1]")
    total_right_mm = height * math.tan(math.radians(abs(tilt)))
    return initial_tip + np.array(
        [0.0, -total_right_mm * progress, 0.0],
        dtype=np.float64,
    )
