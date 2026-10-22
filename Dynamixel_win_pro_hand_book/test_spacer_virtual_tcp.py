"""Pure, hardware-free tests for spacer_virtual_tcp."""

from __future__ import annotations

import math
import unittest

import numpy as np

from Dynamixel_win_pro_hand_book.spacer_virtual_tcp import (
    SpacerVirtualTcpCalibrationError,
    SpacerVirtualTcpGeometry,
    calculate_required_tcp_position_robot,
    calculate_right_tilted_desired_tip_position,
    calculate_sp_lin_displacement_mm,
    calculate_spacer_tip_offset_tcp,
    calculate_spacer_tip_position_robot,
    calculate_virtual_tcp_solution,
)


def calibrated_geometry() -> SpacerVirtualTcpGeometry:
    return SpacerVirtualTcpGeometry(
        tcp_to_rot_axis_reference_mm=(10.0, 20.0, 30.0),
        sp_lin_axis_tcp_unit=(0.0, 0.0, 1.0),
        sp_lin_reference_steps=1000.0,
        sp_lin_mm_per_step=0.01,
        sp_rot_axis_tcp_unit=(2.0**-0.5, 2.0**-0.5, 0.0),
        rot_axis_to_tip_zero_mm=(0.0, 0.0, 50.0),
        calibration_id="pure-test-calibration",
    )


class SpacerVirtualTcpTests(unittest.TestCase):
    def test_zero_angle_and_reference_linear_position(self) -> None:
        geometry = calibrated_geometry()
        actual = calculate_spacer_tip_offset_tcp(
            geometry,
            sp_lin_position_steps=1000.0,
            sp_rot_angle_deg=0.0,
        )
        np.testing.assert_allclose(actual, [10.0, 20.0, 80.0])

    def test_rotation_preserves_tip_radius(self) -> None:
        geometry = calibrated_geometry()
        reference_pivot = np.asarray(
            geometry.tcp_to_rot_axis_reference_mm,
            dtype=np.float64,
        )
        for angle_deg in (-80.0, -15.0, 37.0, 90.0):
            tip = calculate_spacer_tip_offset_tcp(
                geometry,
                sp_lin_position_steps=1000.0,
                sp_rot_angle_deg=angle_deg,
            )
            self.assertAlmostEqual(
                float(np.linalg.norm(tip - reference_pivot)),
                50.0,
                places=9,
            )

    def test_sp_lin_moves_only_on_calibrated_axis(self) -> None:
        geometry = calibrated_geometry()
        self.assertAlmostEqual(
            calculate_sp_lin_displacement_mm(geometry, 1250.0),
            2.5,
        )
        at_reference = calculate_spacer_tip_offset_tcp(
            geometry,
            sp_lin_position_steps=1000.0,
            sp_rot_angle_deg=20.0,
        )
        extended = calculate_spacer_tip_offset_tcp(
            geometry,
            sp_lin_position_steps=1250.0,
            sp_rot_angle_deg=20.0,
        )
        np.testing.assert_allclose(extended - at_reference, [0.0, 0.0, 2.5])

    def test_forward_inverse_round_trip(self) -> None:
        geometry = calibrated_geometry()
        tcp_pose = np.array([300.0, -25.0, 410.0, 0.2, -0.3, 0.4])
        tip = calculate_spacer_tip_position_robot(
            tcp_pose,
            geometry,
            sp_lin_position_steps=1234.0,
            sp_rot_angle_deg=28.0,
        )
        required_tcp = calculate_required_tcp_position_robot(
            tip,
            tcp_pose[3:],
            geometry,
            sp_lin_position_steps=1234.0,
            sp_rot_angle_deg=28.0,
        )
        np.testing.assert_allclose(required_tcp, tcp_pose[:3], atol=1.0e-10)
        reconstructed_pose = np.concatenate([required_tcp, tcp_pose[3:]])
        reconstructed_tip = calculate_spacer_tip_position_robot(
            reconstructed_pose,
            geometry,
            sp_lin_position_steps=1234.0,
            sp_rot_angle_deg=28.0,
        )
        np.testing.assert_allclose(reconstructed_tip, tip, atol=1.0e-10)

    def test_right_target_uses_height_tangent_and_negative_robot_y(self) -> None:
        desired = calculate_right_tilted_desired_tip_position(
            [1.0, 2.0, 3.0],
            contact_height_mm=100.0,
            boundary_tilt_deg=-10.0,
            rotation_progress=0.25,
        )
        expected_y = 2.0 - 100.0 * math.tan(math.radians(10.0)) * 0.25
        np.testing.assert_allclose(desired, [1.0, expected_y, 3.0])

    def test_rotation_can_require_xyz_tcp_compensation(self) -> None:
        geometry = SpacerVirtualTcpGeometry(
            tcp_to_rot_axis_reference_mm=(8.0, -4.0, 12.0),
            sp_lin_axis_tcp_unit=(0.0, 0.0, 1.0),
            sp_lin_reference_steps=0.0,
            sp_lin_mm_per_step=0.02,
            sp_rot_axis_tcp_unit=(
                3.0**-0.5,
                3.0**-0.5,
                3.0**-0.5,
            ),
            rot_axis_to_tip_zero_mm=(15.0, -20.0, 35.0),
            calibration_id="xyz-compensation-test",
        )
        pose = np.array([100.0, 200.0, 300.0, 0.3, -0.2, 0.5])
        initial_tip = calculate_spacer_tip_position_robot(
            pose,
            geometry,
            sp_lin_position_steps=0.0,
            sp_rot_angle_deg=0.0,
        )
        solution = calculate_virtual_tcp_solution(
            pose,
            initial_tip,
            geometry,
            sp_lin_position_steps=0.0,
            sp_rot_angle_deg=35.0,
        )
        compensation = np.asarray(solution.tcp_compensation_robot_mm)
        self.assertTrue(np.all(np.abs(compensation) > 1.0e-6))

    def test_uncalibrated_geometry_fails_closed(self) -> None:
        with self.assertRaises(SpacerVirtualTcpCalibrationError):
            SpacerVirtualTcpGeometry(
                tcp_to_rot_axis_reference_mm=(0.0, 0.0, 0.0),
                sp_lin_axis_tcp_unit=(0.0, 0.0, 0.0),
                sp_lin_reference_steps=0.0,
                sp_lin_mm_per_step=0.0,
                sp_rot_axis_tcp_unit=(0.0, 0.0, 0.0),
                rot_axis_to_tip_zero_mm=(0.0, 0.0, 0.0),
                calibration_id="",
            )


if __name__ == "__main__":
    unittest.main()
