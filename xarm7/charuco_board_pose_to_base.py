#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ChArUcoボードをRealSenseカラー画像から検出し、PnPで T_cam_board を求め、
既存の cam_mm_to_robot_mm() を使って T_base_board に変換する。

対象ボード:
    dictionary      : DICT_6X6_250
    squares         : 7 x 5
    square length   : 40 mm
    marker length   : 20 mm
    board size      : 280 x 200 mm

座標変換:
    T_A_B はB座標系の点をA座標系へ変換する4x4行列。
    T_base_board = T_base_cam @ T_cam_board

操作:
    Enter : 直近の有効フレームを平均してベース座標姿勢を確定・JSON保存
    r     : 時系列平均バッファをリセット
    ESC/q : 終了
"""

from __future__ import annotations

import json
import math
import os
import signal
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
import yaml

from rclpy.executors import MultiThreadedExecutor
from xarm7.control.xarm7 import XArm7
from xarm7.control.robot_base_coordinate import cam_mm_to_robot_mm

CONFIG_PATH = "Retrieval_integration.yaml"
RESULT_JSON_PATH = "charuco_board_pose_base.json"

CHARUCO_DICT_NAME = "DICT_6X6_250"
CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_LENGTH_M = 0.040
CHARUCO_MARKER_LENGTH_M = 0.020
CHARUCO_LEGACY_PATTERN = False
BOARD_WIDTH_M = CHARUCO_SQUARES_X * CHARUCO_SQUARE_LENGTH_M
BOARD_HEIGHT_M = CHARUCO_SQUARES_Y * CHARUCO_SQUARE_LENGTH_M

MIN_CHARUCO_CORNERS = 8
MAX_MEAN_REPROJECTION_ERROR_PX = 2.0
MAX_POINT_REPROJECTION_ERROR_PX = 4.0

POSE_AVERAGE_WINDOW = 15
MIN_POSES_FOR_AVERAGE = 5
BUFFER_RESET_TRANSLATION_MM = 30.0
BUFFER_RESET_ROTATION_DEG = 10.0
CAMERA_AXIS_PROBE_LENGTH_MM = 100.0

COLOR_WIDTH = 640
COLOR_HEIGHT = 480
COLOR_FPS = 30


def sigint_handler(sig, frame):
    print("Ctrl+C detected -> FORCE KILL")
    try:
        arm = globals().get("arm", None)
        if arm:
            arm.emergency_stop()
    except Exception:
        pass
    os._exit(1)


signal.signal(signal.SIGINT, sigint_handler)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def try_call(obj, name, *args, **kwargs):
    fn = getattr(obj, name, None)
    if fn is None:
        return None
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"[WARN] {name} failed: {exc}")
        return None


def recover_xarm_if_possible(arm):
    print("Try xArm recovery...")
    try_call(arm, "clean_error")
    try_call(arm, "clean_warn")
    try_call(arm, "motion_enable", enable=True)
    try_call(arm, "set_mode", 0)
    try_call(arm, "set_state", 0)

    inner = getattr(arm, "arm", None)
    if inner is not None:
        try_call(inner, "clean_error")
        try_call(inner, "clean_warn")
        try_call(inner, "motion_enable", enable=True)
        try_call(inner, "set_mode", 0)
        try_call(inner, "set_state", 0)
    time.sleep(0.5)


def get_camera_matrix_and_dist(intr):
    camera_matrix = np.array([
        [intr.fx, 0.0, intr.ppx],
        [0.0, intr.fy, intr.ppy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.asarray(intr.coeffs, dtype=np.float64).reshape(-1, 1)
    return camera_matrix, dist_coeffs


def get_aruco_dict(name: str):
    aruco = cv2.aruco
    table = {
        "DICT_4X4_50": aruco.DICT_4X4_50,
        "DICT_4X4_100": aruco.DICT_4X4_100,
        "DICT_4X4_250": aruco.DICT_4X4_250,
        "DICT_4X4_1000": aruco.DICT_4X4_1000,
        "DICT_5X5_50": aruco.DICT_5X5_50,
        "DICT_5X5_100": aruco.DICT_5X5_100,
        "DICT_5X5_250": aruco.DICT_5X5_250,
        "DICT_5X5_1000": aruco.DICT_5X5_1000,
        "DICT_6X6_50": aruco.DICT_6X6_50,
        "DICT_6X6_100": aruco.DICT_6X6_100,
        "DICT_6X6_250": aruco.DICT_6X6_250,
        "DICT_6X6_1000": aruco.DICT_6X6_1000,
    }
    if name not in table:
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return aruco.getPredefinedDictionary(table[name])


def create_charuco_board_and_detector(camera_matrix, dist_coeffs):
    dictionary = get_aruco_dict(CHARUCO_DICT_NAME)

    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
            CHARUCO_SQUARE_LENGTH_M,
            CHARUCO_MARKER_LENGTH_M,
            dictionary,
        )
        if hasattr(board, "setLegacyPattern"):
            board.setLegacyPattern(CHARUCO_LEGACY_PATTERN)
    elif hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(
            CHARUCO_SQUARES_X,
            CHARUCO_SQUARES_Y,
            CHARUCO_SQUARE_LENGTH_M,
            CHARUCO_MARKER_LENGTH_M,
            dictionary,
        )
    else:
        raise RuntimeError("ChArUco APIがありません。opencv-contrib-pythonを使用してください。")

    if hasattr(cv2.aruco, "DetectorParameters"):
        detector_params = cv2.aruco.DetectorParameters()
    else:
        detector_params = cv2.aruco.DetectorParameters_create()
    # ChArUco交点は補間後にサブピクセル補正される。
    # 内部ArUco角のSUBPIX補正はチェス模様の影響を受け得るため無効。
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

    if hasattr(cv2.aruco, "CharucoDetector"):
        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.cameraMatrix = camera_matrix
        charuco_params.distCoeffs = dist_coeffs
        detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    else:
        detector = None

    return board, dictionary, detector_params, detector


def detect_charuco(
    gray, board, dictionary, detector_params, detector,
    camera_matrix, dist_coeffs,
):
    if detector is not None:
        return detector.detectBoard(gray)

    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
        gray, dictionary, parameters=detector_params
    )
    if marker_ids is None or len(marker_ids) == 0:
        return None, None, marker_corners, marker_ids

    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
    )
    return charuco_corners, charuco_ids, marker_corners, marker_ids


def match_charuco_image_points(board, charuco_corners, charuco_ids):
    if hasattr(board, "matchImagePoints"):
        object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
        return (
            np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3),
            np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2),
        )

    if hasattr(board, "getChessboardCorners"):
        all_object_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    elif hasattr(board, "chessboardCorners"):
        all_object_points = np.asarray(board.chessboardCorners, dtype=np.float64)
    else:
        raise RuntimeError("ChArUcoの3D交点座標を取得できません。")

    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    object_points = all_object_points[ids].reshape(-1, 1, 3)
    image_points = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 1, 2)
    return object_points, image_points


def make_transform(rotation_matrix, translation):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def project_errors(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    return np.linalg.norm(
        projected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1
    )


def estimate_charuco_pose_pnp(board, charuco_corners, charuco_ids, camera_matrix, dist_coeffs):
    if charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS:
        return None

    object_points, image_points = match_charuco_image_points(
        board, charuco_corners, charuco_ids
    )
    if len(object_points) < MIN_CHARUCO_CORNERS:
        return None

    first_flag = cv2.SOLVEPNP_IPPE if hasattr(cv2, "SOLVEPNP_IPPE") else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs, flags=first_flag
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not ok:
        return None

    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec
        )

    errors = project_errors(
        object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    median_error = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median_error)))
    robust_threshold = median_error + 3.0 * max(1.4826 * mad, 0.25)
    inlier_threshold = min(
        MAX_POINT_REPROJECTION_ERROR_PX, max(1.5, robust_threshold)
    )
    inlier_mask = errors <= inlier_threshold
    inlier_count = int(np.count_nonzero(inlier_mask))

    if inlier_count >= MIN_CHARUCO_CORNERS and inlier_count < len(errors):
        obj_in = object_points[inlier_mask]
        img_in = image_points[inlier_mask]
        ok2, rvec2, tvec2 = cv2.solvePnP(
            obj_in, img_in, camera_matrix, dist_coeffs,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok2:
            rvec, tvec = rvec2, tvec2
            if hasattr(cv2, "solvePnPRefineLM"):
                rvec, tvec = cv2.solvePnPRefineLM(
                    obj_in, img_in, camera_matrix, dist_coeffs, rvec, tvec
                )
            object_points, image_points = obj_in, img_in

    final_errors = project_errors(
        object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    return {
        "T_cam_board_m": make_transform(rotation_matrix, tvec.reshape(3)),
        "rvec": rvec.reshape(3),
        "tvec_m": tvec.reshape(3),
        "mean_reprojection_error_px": float(np.mean(final_errors)),
        "max_reprojection_error_px": float(np.max(final_errors)),
        "detected_corner_count": int(len(charuco_ids)),
        "used_corner_count": int(len(final_errors)),
        "charuco_ids": np.asarray(charuco_ids, dtype=np.int32).reshape(-1),
    }


def rotation_angle_deg(rotation_a, rotation_b):
    relative = rotation_a.T @ rotation_b
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def orthonormalize_rotation(rotation):
    u, _, vt = np.linalg.svd(rotation)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def average_transforms(transforms):
    translations = np.asarray([T[:3, 3] for T in transforms], dtype=np.float64)
    translation = np.median(translations, axis=0)
    rotation_sum = sum((T[:3, :3] for T in transforms), np.zeros((3, 3)))
    rotation = orthonormalize_rotation(rotation_sum)
    return make_transform(rotation, translation)


def rotation_matrix_to_rpy_deg(rotation):
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-8
    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def map_camera_point_to_base_mm(arm, point_camera_mm):
    result = cam_mm_to_robot_mm(
        arm, np.asarray(point_camera_mm, dtype=np.float64).reshape(3)
    )
    result = np.asarray(result, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("cam_mm_to_robot_mm() の返り値にNaN/Infがあります。")
    return result


def estimate_T_base_cam_from_existing_mapper(arm):
    """
    cam_mm_to_robot_mm() にカメラ原点とX/Y/Z基準点を入力し、
    既存の点変換から T_base_cam を復元する。
    """
    axis = CAMERA_AXIS_PROBE_LENGTH_MM
    p0 = map_camera_point_to_base_mm(arm, [0.0, 0.0, 0.0])
    px = map_camera_point_to_base_mm(arm, [axis, 0.0, 0.0])
    py = map_camera_point_to_base_mm(arm, [0.0, axis, 0.0])
    pz = map_camera_point_to_base_mm(arm, [0.0, 0.0, axis])

    raw_x = (px - p0) / axis
    raw_y = (py - p0) / axis
    raw_z = (pz - p0) / axis
    raw_rotation = np.column_stack([raw_x, raw_y, raw_z])
    rotation = orthonormalize_rotation(raw_rotation)

    diagnostics = {
        "raw_x_axis_norm": float(np.linalg.norm(raw_x)),
        "raw_y_axis_norm": float(np.linalg.norm(raw_y)),
        "raw_z_axis_norm": float(np.linalg.norm(raw_z)),
        "raw_rotation_det": float(np.linalg.det(raw_rotation)),
        "orthonormalized_rotation_det": float(np.linalg.det(rotation)),
    }
    return make_transform(rotation, p0), diagnostics


def convert_T_cam_board_to_base(arm, T_cam_board_m):
    T_cam_board_mm = np.asarray(T_cam_board_m, dtype=np.float64).copy()
    T_cam_board_mm[:3, 3] *= 1000.0
    T_base_cam_mm, diagnostics = estimate_T_base_cam_from_existing_mapper(arm)
    T_base_board_mm = T_base_cam_mm @ T_cam_board_mm

    # OpenCVのボード原点から、幾何中心への固定変換。
    T_board_center_mm = np.eye(4, dtype=np.float64)
    T_board_center_mm[:3, 3] = [
        BOARD_WIDTH_M * 500.0,
        BOARD_HEIGHT_M * 500.0,
        0.0,
    ]
    T_base_board_center_mm = T_base_board_mm @ T_board_center_mm

    return (
        T_cam_board_mm,
        T_base_cam_mm,
        T_base_board_mm,
        T_base_board_center_mm,
        diagnostics,
    )


def put_text(image, text, y, color=(0, 255, 0), scale=0.60):
    cv2.putText(
        image, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 2, cv2.LINE_AA,
    )


def save_result_json(result, path=RESULT_JSON_PATH):
    data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "board": {
            "dictionary": CHARUCO_DICT_NAME,
            "squares_x": CHARUCO_SQUARES_X,
            "squares_y": CHARUCO_SQUARES_Y,
            "square_length_m": CHARUCO_SQUARE_LENGTH_M,
            "marker_length_m": CHARUCO_MARKER_LENGTH_M,
            "legacy_pattern": CHARUCO_LEGACY_PATTERN,
        },
        "camera_pose": {
            "T_cam_board_mm": result["T_cam_board_mm"].tolist(),
            "position_mm": result["T_cam_board_mm"][:3, 3].tolist(),
            "rpy_deg": rotation_matrix_to_rpy_deg(
                result["T_cam_board_mm"][:3, :3]
            ).tolist(),
        },
        "base_pose": {
            "T_base_cam_mm": result["T_base_cam_mm"].tolist(),
            "T_base_board_mm": result["T_base_board_mm"].tolist(),
            "T_base_board_center_mm": result["T_base_board_center_mm"].tolist(),
            "position_mm": result["T_base_board_mm"][:3, 3].tolist(),
            "center_position_mm": result["T_base_board_center_mm"][:3, 3].tolist(),
            "rpy_deg": rotation_matrix_to_rpy_deg(
                result["T_base_board_mm"][:3, :3]
            ).tolist(),
            "rpy_convention": "R = Rz(yaw) @ Ry(pitch) @ Rx(roll)",
        },
        "detection": {
            "mean_reprojection_error_px": result["mean_reprojection_error_px"],
            "max_reprojection_error_px": result["max_reprojection_error_px"],
            "detected_corner_count": result["detected_corner_count"],
            "used_corner_count": result["used_corner_count"],
            "charuco_ids": result["charuco_ids"].tolist(),
            "averaged_frame_count": result["averaged_frame_count"],
        },
        "base_transform_diagnostics": result["base_transform_diagnostics"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[RESULT SAVED] {path}")


def print_pose_result(result):
    T_cam = result["T_cam_board_mm"]
    T_base_cam = result["T_base_cam_mm"]
    T_base_board = result["T_base_board_mm"]
    T_base_board_center = result["T_base_board_center_mm"]

    print("\n============ CHARUCO POSE ============")
    print(f"Board: {CHARUCO_SQUARES_X}x{CHARUCO_SQUARES_Y}, "
          f"square={CHARUCO_SQUARE_LENGTH_M*1000:.1f} mm, "
          f"marker={CHARUCO_MARKER_LENGTH_M*1000:.1f} mm")
    print("--------------------------------------")
    print("T_cam_board [mm]:\n", np.array2string(T_cam, precision=6, suppress_small=True))
    print("camera position [mm] =", T_cam[:3, 3])
    print("camera rpy [deg]     =", rotation_matrix_to_rpy_deg(T_cam[:3, :3]))
    print("--------------------------------------")
    print("T_base_cam [mm]:\n", np.array2string(T_base_cam, precision=6, suppress_small=True))
    print("--------------------------------------")
    print("T_base_board [mm]:\n", np.array2string(T_base_board, precision=6, suppress_small=True))
    print("base position [mm] =", T_base_board[:3, 3])
    print("base rpy [deg]     =", rotation_matrix_to_rpy_deg(T_base_board[:3, :3]))
    print("base board center [mm] =", T_base_board_center[:3, 3])
    print("--------------------------------------")
    print(f"reprojection mean/max [px] = "
          f"{result['mean_reprojection_error_px']:.3f} / "
          f"{result['max_reprojection_error_px']:.3f}")
    print(f"corners detected/used = "
          f"{result['detected_corner_count']} / {result['used_corner_count']}")
    print("averaged frames =", result["averaged_frame_count"])
    print("======================================\n")


def run_charuco_capture_to_base(arm):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, COLOR_FPS
    )
    profile = pipeline.start(config)

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_matrix, dist_coeffs = get_camera_matrix_and_dist(intr)
    board, dictionary, detector_params, detector = create_charuco_board_and_detector(
        camera_matrix, dist_coeffs
    )

    pose_buffer = deque(maxlen=POSE_AVERAGE_WINDOW)
    metadata_buffer = deque(maxlen=POSE_AVERAGE_WINDOW)

    print("======================================")
    print("RealSense started")
    print("Position source : ChArUco + solvePnP")
    print("Depth           : NOT USED")
    print("Dictionary      :", CHARUCO_DICT_NAME)
    print("Board           :", f"{CHARUCO_SQUARES_X} x {CHARUCO_SQUARES_Y}")
    print("Square / marker :",
          f"{CHARUCO_SQUARE_LENGTH_M*1000:.1f} / "
          f"{CHARUCO_MARKER_LENGTH_M*1000:.1f} mm")
    print("ENTER : camera pose -> base pose")
    print("r     : reset temporal average")
    print("ESC/q : exit")
    print("======================================")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            debug = color.copy()
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco(
                gray, board, dictionary, detector_params, detector,
                camera_matrix, dist_coeffs,
            )

            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(debug, marker_corners, marker_ids)

            pose_result = None
            if charuco_ids is not None and len(charuco_ids) > 0:
                cv2.aruco.drawDetectedCornersCharuco(
                    debug, charuco_corners, charuco_ids, (0, 0, 255)
                )
                pose_result = estimate_charuco_pose_pnp(
                    board, charuco_corners, charuco_ids,
                    camera_matrix, dist_coeffs,
                )

            if pose_result is not None:
                mean_error = pose_result["mean_reprojection_error_px"]
                accepted = mean_error <= MAX_MEAN_REPROJECTION_ERROR_PX
                T_now = pose_result["T_cam_board_m"]

                if accepted:
                    if pose_buffer:
                        T_prev = pose_buffer[-1]
                        translation_jump_mm = np.linalg.norm(
                            T_now[:3, 3] - T_prev[:3, 3]
                        ) * 1000.0
                        rotation_jump_deg = rotation_angle_deg(
                            T_prev[:3, :3], T_now[:3, :3]
                        )
                        if (translation_jump_mm > BUFFER_RESET_TRANSLATION_MM or
                                rotation_jump_deg > BUFFER_RESET_ROTATION_DEG):
                            pose_buffer.clear()
                            metadata_buffer.clear()
                    pose_buffer.append(T_now.copy())
                    metadata_buffer.append(pose_result.copy())

                try:
                    cv2.drawFrameAxes(
                        debug, camera_matrix, dist_coeffs,
                        pose_result["rvec"], pose_result["tvec_m"], 0.08, 3
                    )
                except Exception:
                    pass

                tvec_mm = pose_result["tvec_m"] * 1000.0
                color_status = (0, 255, 0) if accepted else (0, 0, 255)
                put_text(debug,
                         f"corners {pose_result['used_corner_count']}/"
                         f"{pose_result['detected_corner_count']}",
                         30, color_status)
                put_text(debug,
                         f"reproj mean/max {mean_error:.2f}/"
                         f"{pose_result['max_reprojection_error_px']:.2f} px",
                         58, color_status)
                put_text(debug,
                         f"cam X={tvec_mm[0]:.1f} Y={tvec_mm[1]:.1f} "
                         f"Z={tvec_mm[2]:.1f} mm",
                         86, color_status)
                put_text(debug,
                         f"average buffer {len(pose_buffer)}/{POSE_AVERAGE_WINDOW}",
                         114, color_status)
            else:
                visible_count = 0 if charuco_ids is None else len(charuco_ids)
                put_text(debug,
                         f"Need >= {MIN_CHARUCO_CORNERS} corners; now {visible_count}",
                         35, (0, 0, 255))

            cv2.imshow("ChArUco board pose to base", debug)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                print("Canceled")
                return None
            if key == ord("r"):
                pose_buffer.clear()
                metadata_buffer.clear()
                print("[BUFFER RESET]")
                continue
            if key in (10, 13):
                if not pose_buffer:
                    print("有効なChArUco姿勢がありません。")
                    continue
                if len(pose_buffer) < MIN_POSES_FOR_AVERAGE:
                    print(f"[WARN] 平均フレーム数が少ないです: {len(pose_buffer)}")

                T_cam_board_average_m = average_transforms(list(pose_buffer))
                latest = metadata_buffer[-1]
                (
                    T_cam_mm,
                    T_base_cam_mm,
                    T_base_board_mm,
                    T_base_board_center_mm,
                    diagnostics,
                ) = convert_T_cam_board_to_base(arm, T_cam_board_average_m)

                result = {
                    "T_cam_board_mm": T_cam_mm,
                    "T_base_cam_mm": T_base_cam_mm,
                    "T_base_board_mm": T_base_board_mm,
                    "T_base_board_center_mm": T_base_board_center_mm,
                    "mean_reprojection_error_px": float(np.mean([
                        item["mean_reprojection_error_px"] for item in metadata_buffer
                    ])),
                    "max_reprojection_error_px": float(np.max([
                        item["max_reprojection_error_px"] for item in metadata_buffer
                    ])),
                    "detected_corner_count": latest["detected_corner_count"],
                    "used_corner_count": latest["used_corner_count"],
                    "charuco_ids": latest["charuco_ids"],
                    "averaged_frame_count": len(pose_buffer),
                    "base_transform_diagnostics": diagnostics,
                }
                print_pose_result(result)
                save_result_json(result)
                return result
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def main():
    config = load_config(CONFIG_PATH)
    rclpy.init()
    node = rclpy.create_node("charuco_board_pose_to_base")
    xarm_host = config["robot"]["xarm"]["host"]
    arm = XArm7(node=node, host=xarm_host)
    globals()["arm"] = arm

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        print("xArm ready")
        recover_xarm_if_possible(arm)
        result = run_charuco_capture_to_base(arm)
        if result is not None:
            print("Base board origin [mm] =", result["T_base_board_mm"][:3, 3])
            print("Base board center [mm] =", result["T_base_board_center_mm"][:3, 3])
            print("Base board RPY [deg] =", rotation_matrix_to_rpy_deg(
                result["T_base_board_mm"][:3, :3]
            ))
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as exc:
        print("Abort due to exception:")
        print(exc)
        try:
            arm.emergency_stop()
        except Exception:
            pass
        raise
    finally:
        print("Shutting down...")
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

