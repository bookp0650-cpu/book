#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
7x5 ChArUcoボード1枚だけを使うリーチング・到達後評価コード。

処理:
    1. 7x5 ChArUcoボードの位置姿勢 T_cam_board を多点PnPで推定
    2. 中央のマス目の左上角をリーチング対象点にする
       7x5、square=40 mm のため、board座標では [120, 80, 0] mm
    3. 対象点を camera -> robot/base 座標へ変換
    4. move_to_target_xyz_and_roll() でリーチング
    5. 移動後、同じChArUcoボードを再検出
    6. 2列目3行目のマスの左上ChArUco交点を直接検出
    7. その画像座標・傾き・基準誤差・base座標をCSVへ記録

対象ボード:
    dictionary      : DICT_6X6_250
    squares         : 7 x 5
    square length   : 40 mm
    marker length   : 20 mm
    board size      : 280 x 200 mm

座標規約:
    T_A_B は、B座標系の点をA座標系へ変換する4x4行列。
    p_cam = T_cam_board @ p_board

操作:
    検出画面:
        Enter : 直近の有効姿勢を平均して確定
        r     : 平均バッファをリセット
        ESC/q : キャンセル
"""

from __future__ import annotations

import csv
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
from xarm7.control.robot_base_coordinate import (
    cam_mm_to_robot_mm,
    print_camera_debug_info,
)


# =============================================================================
# user settings
# =============================================================================

CONFIG_PATH = "Retrieval_integration.yaml"
RESULT_CSV_PATH = "charuco_single_board_reaching_eval_live_base_compact_uv_mm.csv"

HANDEYE_JSON_PATH = (
    "/home/book/pro_book/pro_hand_book_python/xarm7/handeye_pairs/handeye_T_tcp_cam_20260717_223007 copy.json"
)

SIDE = "right"
ASK_BEFORE_MOVE = True

RETURN_JOINT_DEG = [
    88.2,
    -55.2,
    171.3,
    65.1,
    40.1,
    11.6,
    44.4,
]

# 以前作成した7x5 ChArUcoボード
CHARUCO_DICT_NAME = "DICT_6X6_250"
CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_LENGTH_M = 0.040
CHARUCO_MARKER_LENGTH_M = 0.020
CHARUCO_LEGACY_PATTERN = False

BOARD_WIDTH_M = CHARUCO_SQUARES_X * CHARUCO_SQUARE_LENGTH_M
BOARD_HEIGHT_M = CHARUCO_SQUARES_Y * CHARUCO_SQUARE_LENGTH_M

# リーチング対象点。現在の設定値は元コードを維持する。
# 0始まりのマス番号で、そのマスの左上角をboard座標として使う。
TARGET_SQUARE_COL = 3
TARGET_SQUARE_ROW = 4
TARGET_POINT_BOARD_M = np.array(
    [
        TARGET_SQUARE_COL * CHARUCO_SQUARE_LENGTH_M,
        TARGET_SQUARE_ROW * CHARUCO_SQUARE_LENGTH_M,
        0.0,
    ],
    dtype=np.float64,
)

# 到達後に評価する「2列目・3行目のマス」の左上角。
# こちらは人が数える1始まりで指定する。
# 7x5、square=40 mmなのでboard座標は [40, 80, 0] mm。
EVAL_SQUARE_COL_1BASED = 4
EVAL_SQUARE_ROW_1BASED = 3
EVAL_POINT_BOARD_M = np.array(
    [
        (EVAL_SQUARE_COL_1BASED - 1) * CHARUCO_SQUARE_LENGTH_M,
        (EVAL_SQUARE_ROW_1BASED - 1) * CHARUCO_SQUARE_LENGTH_M,
        0.0,
    ],
    dtype=np.float64,
)

# ChArUco交点ID。7x5 squaresでは内部交点は6x4個。
# 2列目3行目のマス左上角はID=6、その右隣はID=7。
EVAL_CHARUCO_ID = (
    (EVAL_SQUARE_ROW_1BASED - 2) * (CHARUCO_SQUARES_X - 1)
    + (EVAL_SQUARE_COL_1BASED - 2)
)
EVAL_RIGHT_CHARUCO_ID = EVAL_CHARUCO_ID + 1

# board +X方向の画像上角度から、従来と同じroll補正量を作る。
ROLL_SIGN = -1.0
ROLL_OFFSET_RAD = 0.0
ROLL_PROBE_LENGTH_M = CHARUCO_SQUARE_LENGTH_M

# ChArUco/PnP品質条件
MIN_CHARUCO_CORNERS = 6
MAX_MEAN_REPROJECTION_ERROR_PX = 2.0
MAX_POINT_REPROJECTION_ERROR_PX = 4.0

# リーチング前後とも同じ1280x720で撮影・表示する。
PRE_COLOR_WIDTH = 1280
PRE_COLOR_HEIGHT = 720

POST_COLOR_WIDTH = 1280
POST_COLOR_HEIGHT = 720

# D435iの接続環境によって利用可能FPSが異なるため、上から順に試す。
# リーチング前後で同じ候補を使用する。
COLOR_FPS_CANDIDATES = (6, 15, 30)

# 高解像度・部分可視のリーチング後は、より多くのフレームを平均する。
PRE_POSE_AVERAGE_WINDOW = 15
PRE_MIN_POSES_FOR_AVERAGE = 5
POST_POSE_AVERAGE_WINDOW = 30
POST_MIN_POSES_FOR_AVERAGE = 12

BUFFER_RESET_TRANSLATION_MM = 30.0
BUFFER_RESET_ROTATION_DEG = 10.0

CAMERA_WARMUP_SECONDS = 2.0

# 再投影誤差の閾値は640 px幅を基準に、解像度に比例させる。
# 1280幅では同じ角度誤差が約2倍のpixel値になるため。
REPROJECTION_REFERENCE_WIDTH_PX = 640.0

# -----------------------------------------------------------------------------
# ChArUco検出前の画像処理
#
# pキーで実行中に切り替え可能:
#   gray     : 元のグレースケール
#   clahe    : 局所コントラスト補正（初期設定・推奨）
#   adaptive : 適応的二値化
#
# リーチング後は1280x720なので、u,v基準値も高解像度用にする。
# -----------------------------------------------------------------------------
IMAGE_PREPROCESS_MODE = "clahe"
IMAGE_PREPROCESS_MODES = ("gray", "clahe", "adaptive")

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# adaptiveThresholdのblock sizeは3以上の奇数である必要がある。
ADAPTIVE_THRESH_BLOCK_SIZE = 31
ADAPTIVE_THRESH_C = 7

# Trueにすると、検出へ入力している処理画像を別ウィンドウで表示する。
SHOW_PREPROCESSED_WINDOW = False

# OpenCV表示は、カメラ映像を180度回転してから検出結果・文字を描画する。
# ChArUco検出、PnP、座標変換、CSV値は回転前画像で計算する。
ROTATE_DISPLAY_180 = True

# -----------------------------------------------------------------------------
# 到達後評価の仮基準値。
# 完璧に合った状態を複数回計測した平均値に後で置き換える。
# -----------------------------------------------------------------------------
# 旧640x480基準 (344.8, 183.4) を画素比で暫定変換した値。
# 1280x720では画角・クロップが完全一致しない可能性があるため、
# 完璧に合った状態を再計測して必ず置き換える。
EVAL_REF_U_PX = 670.17
EVAL_REF_V_PX = 263.73
EVAL_REF_ROLL_DEG = -0.48


# =============================================================================
# general helpers
# =============================================================================


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


def moveJ_to_return_pose_direct(
    arm,
    joint_deg=None,
    speed=20,
    mvacc=200,
    wait=True,
):
    if joint_deg is None:
        joint_deg = RETURN_JOINT_DEG

    print("\n========== RETURN JOINT MOVE ==========")
    print("target joint deg =", joint_deg)
    print("speed =", speed)
    print("mvacc =", mvacc)
    print("=======================================\n")

    sdk_arm = getattr(arm, "arm", None)
    if sdk_arm is None:
        sdk_arm = getattr(arm, "_arm", None)
    if sdk_arm is None:
        sdk_arm = arm

    if not hasattr(sdk_arm, "set_servo_angle"):
        raise RuntimeError(
            "set_servo_angle が見つからない。XArm7内のSDK本体の変数名を確認して。"
        )

    ret = sdk_arm.set_servo_angle(
        angle=joint_deg,
        speed=speed,
        mvacc=mvacc,
        is_radian=False,
        wait=wait,
    )
    print("[return pose ret] =", ret)
    return ret


def normalize_angle_rad(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def normalize_roll_for_board(image_angle_rad):
    angle = normalize_angle_rad(image_angle_rad)

    # 従来コードと同じく -90〜90 degへ寄せる。
    if angle > np.pi / 2:
        angle -= np.pi
    elif angle < -np.pi / 2:
        angle += np.pi

    return float(ROLL_SIGN * angle + ROLL_OFFSET_RAD)


def wrap_angle_deg(angle_deg):
    return float((angle_deg + 180.0) % 360.0 - 180.0)


def make_transform(rotation_matrix, translation):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def transform_point(T, point_xyz):
    point_xyz = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    return T[:3, :3] @ point_xyz + T[:3, 3]


def rotation_matrix_to_rpy_deg(rotation):
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
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
    if not transforms:
        raise ValueError("transforms is empty")

    translations = np.asarray([T[:3, 3] for T in transforms], dtype=np.float64)
    translation = np.median(translations, axis=0)

    rotation_sum = sum(
        (T[:3, :3] for T in transforms),
        np.zeros((3, 3), dtype=np.float64),
    )
    rotation = orthonormalize_rotation(rotation_sum)
    return make_transform(rotation, translation)


def put_text(image, text, y, color=(0, 255, 0), scale=0.58):
    cv2.putText(
        image,
        text,
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def rotate_uv_180_for_display(point_uv, image_width, image_height):
    """計算用画像上の点を、180度回転した表示画像上の点へ変換する。"""
    point_uv = np.asarray(point_uv, dtype=np.float64).reshape(2)
    return np.array(
        [
            (float(image_width) - 1.0) - point_uv[0],
            (float(image_height) - 1.0) - point_uv[1],
        ],
        dtype=np.float64,
    )


def rotate_corner_array_180_for_display(corners, image_width, image_height):
    """ArUco/ChArUco角座標を、180度回転した表示座標へ変換する。"""
    if corners is None:
        return None

    if isinstance(corners, (list, tuple)):
        return [
            rotate_corner_array_180_for_display(
                item, image_width, image_height
            )
            for item in corners
        ]

    rotated = np.asarray(corners, dtype=np.float32).copy()
    rotated[..., 0] = (float(image_width) - 1.0) - rotated[..., 0]
    rotated[..., 1] = (float(image_height) - 1.0) - rotated[..., 1]
    return rotated


# =============================================================================
# ChArUco detection and PnP
# =============================================================================


def preprocess_charuco_gray(gray, mode, clahe):
    """ChArUco検出へ入力する8-bitグレースケール画像を作る。"""
    if mode == "gray":
        return gray

    if mode == "clahe":
        return clahe.apply(gray)

    if mode == "adaptive":
        block_size = int(ADAPTIVE_THRESH_BLOCK_SIZE)
        if block_size < 3:
            block_size = 3
        if block_size % 2 == 0:
            block_size += 1

        # 二値化前にもCLAHEをかけ、局所的な照明むらを軽減する。
        enhanced = clahe.apply(gray)
        return cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            ADAPTIVE_THRESH_C,
        )

    raise ValueError(
        f"Unknown IMAGE_PREPROCESS_MODE={mode!r}; "
        f"choose from {IMAGE_PREPROCESS_MODES}"
    )


def get_camera_matrix_and_dist(intr):
    camera_matrix = np.array(
        [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.asarray(intr.coeffs, dtype=np.float64).reshape(-1, 1)
    return camera_matrix, dist_coeffs


def pixel_delta_to_image_plane_mm(
    current_uv_px,
    reference_uv_px,
    depth_m,
    camera_matrix,
    dist_coeffs,
):
    """
    現在点と基準点の画素差を、現在点の深さZにおける画像平面相当mmへ変換する。

    戻り値の符号は元のカメラ画像座標に従う:
        +du_mm : 画像のu正方向（右）
        +dv_mm : 画像のv正方向（下）

    180度回転は表示だけなので、この符号規約には影響しない。
    奥行き方向の差は含めず、両点が同じ深さ平面にあるとみなす。
    """
    depth_m = float(depth_m)
    if not np.isfinite(depth_m) or depth_m <= 0.0:
        return None

    pixels = np.asarray(
        [current_uv_px, reference_uv_px],
        dtype=np.float64,
    ).reshape(-1, 1, 2)

    # 歪みを補正し、正規化画像座標(x, y)へ変換する。
    normalized = cv2.undistortPoints(
        pixels,
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
    ).reshape(-1, 2)

    delta_normalized = normalized[0] - normalized[1]
    delta_uv_mm = delta_normalized * depth_m * 1000.0
    error_mm = float(np.linalg.norm(delta_uv_mm))

    return {
        "delta_uv_mm": delta_uv_mm,
        "error_mm": error_mm,
        "depth_mm": depth_m * 1000.0,
    }


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
        raise RuntimeError(
            "ChArUco APIがありません。opencv-contrib-pythonを使用してください。"
        )

    if hasattr(cv2.aruco, "DetectorParameters"):
        detector_params = cv2.aruco.DetectorParameters()
    else:
        detector_params = cv2.aruco.DetectorParameters_create()

    # ArUcoマーカー角をサブピクセル精度で補正する。
    # その角を使ってChArUco交点を補間するため、PnPの再投影誤差低減を狙う。
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector_params.cornerRefinementWinSize = 5
    detector_params.cornerRefinementMaxIterations = 50
    detector_params.cornerRefinementMinAccuracy = 0.01

    if hasattr(cv2.aruco, "CharucoDetector"):
        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.cameraMatrix = camera_matrix
        charuco_params.distCoeffs = dist_coeffs
        detector = cv2.aruco.CharucoDetector(
            board,
            charuco_params,
            detector_params,
        )
    else:
        detector = None

    return board, dictionary, detector_params, detector


def detect_charuco(
    gray,
    board,
    dictionary,
    detector_params,
    detector,
    camera_matrix,
    dist_coeffs,
):
    if detector is not None:
        return detector.detectBoard(gray)

    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
        gray,
        dictionary,
        parameters=detector_params,
    )
    if marker_ids is None or len(marker_ids) == 0:
        return None, None, marker_corners, marker_ids

    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
    )
    return charuco_corners, charuco_ids, marker_corners, marker_ids


def match_charuco_image_points(board, charuco_corners, charuco_ids):
    if hasattr(board, "matchImagePoints"):
        object_points, image_points = board.matchImagePoints(
            charuco_corners,
            charuco_ids,
        )
        return (
            np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3),
            np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2),
        )

    if hasattr(board, "getChessboardCorners"):
        all_object_points = np.asarray(
            board.getChessboardCorners(),
            dtype=np.float64,
        )
    elif hasattr(board, "chessboardCorners"):
        all_object_points = np.asarray(
            board.chessboardCorners,
            dtype=np.float64,
        )
    else:
        raise RuntimeError("ChArUcoの3D交点座標を取得できません。")

    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    object_points = all_object_points[ids].reshape(-1, 1, 3)
    image_points = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 1, 2)
    return object_points, image_points


def project_errors(
    object_points,
    image_points,
    rvec,
    tvec,
    camera_matrix,
    dist_coeffs,
):
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    return np.linalg.norm(
        projected.reshape(-1, 2) - image_points.reshape(-1, 2),
        axis=1,
    )


def estimate_charuco_pose_pnp(
    board,
    charuco_corners,
    charuco_ids,
    camera_matrix,
    dist_coeffs,
    reprojection_scale: float = 1.0,
):
    if charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS:
        return None

    object_points, image_points = match_charuco_image_points(
        board,
        charuco_corners,
        charuco_ids,
    )
    if len(object_points) < MIN_CHARUCO_CORNERS:
        return None

    first_flag = (
        cv2.SOLVEPNP_IPPE
        if hasattr(cv2, "SOLVEPNP_IPPE")
        else cv2.SOLVEPNP_ITERATIVE
    )

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=first_flag,
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not ok:
        return None

    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
        )

    errors = project_errors(
        object_points,
        image_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    median_error = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median_error)))
    robust_threshold = median_error + 3.0 * max(1.4826 * mad, 0.25)
    max_point_error_px = (
        MAX_POINT_REPROJECTION_ERROR_PX * float(reprojection_scale)
    )
    minimum_inlier_threshold_px = 1.5 * float(reprojection_scale)

    inlier_threshold = min(
        max_point_error_px,
        max(minimum_inlier_threshold_px, robust_threshold),
    )
    inlier_mask = errors <= inlier_threshold
    inlier_count = int(np.count_nonzero(inlier_mask))

    if inlier_count >= MIN_CHARUCO_CORNERS and inlier_count < len(errors):
        object_points_inlier = object_points[inlier_mask]
        image_points_inlier = image_points[inlier_mask]

        ok2, rvec2, tvec2 = cv2.solvePnP(
            object_points_inlier,
            image_points_inlier,
            camera_matrix,
            dist_coeffs,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok2:
            rvec, tvec = rvec2, tvec2
            if hasattr(cv2, "solvePnPRefineLM"):
                rvec, tvec = cv2.solvePnPRefineLM(
                    object_points_inlier,
                    image_points_inlier,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                )
            object_points = object_points_inlier
            image_points = image_points_inlier

    final_errors = project_errors(
        object_points,
        image_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
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


def project_board_points(points_board_m, rvec, tvec, camera_matrix, dist_coeffs):
    points = np.asarray(points_board_m, dtype=np.float64).reshape(-1, 1, 3)
    pixels, _ = cv2.projectPoints(
        points,
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        camera_matrix,
        dist_coeffs,
    )
    return pixels.reshape(-1, 2)


def compute_board_point_and_roll(
    pose_result,
    camera_matrix,
    dist_coeffs,
    point_board_m,
):
    """board上の任意点についてcamera座標・画像座標・board +X傾きを計算。"""
    point_board_m = np.asarray(point_board_m, dtype=np.float64).reshape(3)
    T_cam_board_m = pose_result["T_cam_board_m"]

    point_camera_m = transform_point(T_cam_board_m, point_board_m)
    point_x_probe_board_m = point_board_m + np.array(
        [ROLL_PROBE_LENGTH_M, 0.0, 0.0],
        dtype=np.float64,
    )

    point_uv, point_x_uv = project_board_points(
        [point_board_m, point_x_probe_board_m],
        pose_result["rvec"],
        pose_result["tvec_m"],
        camera_matrix,
        dist_coeffs,
    )

    image_angle_rad = math.atan2(
        float(point_x_uv[1] - point_uv[1]),
        float(point_x_uv[0] - point_uv[0]),
    )

    return {
        "point_camera_m": point_camera_m,
        "point_uv": point_uv,
        "point_x_uv": point_x_uv,
        "image_angle_rad": float(image_angle_rad),
        "d_roll_rad": float(normalize_roll_for_board(image_angle_rad)),
    }


def get_charuco_corner_uv(charuco_corners, charuco_ids, corner_id):
    """指定したChArUco交点IDの実測画像座標を返す。未検出ならNone。"""
    if charuco_corners is None or charuco_ids is None:
        return None

    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    corners = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
    indices = np.where(ids == int(corner_id))[0]
    if len(indices) == 0:
        return None
    return corners[int(indices[0])].copy()


def get_eval_pixel_measurement(charuco_corners, charuco_ids):
    """
    2列目3行目のマス左上角と右隣交点を直接使い、
    実測u,vと上辺の画像上傾きを返す。
    """
    uv = get_charuco_corner_uv(
        charuco_corners, charuco_ids, EVAL_CHARUCO_ID
    )
    right_uv = get_charuco_corner_uv(
        charuco_corners, charuco_ids, EVAL_RIGHT_CHARUCO_ID
    )
    if uv is None or right_uv is None:
        return None

    angle_rad = math.atan2(
        float(right_uv[1] - uv[1]),
        float(right_uv[0] - uv[0]),
    )
    return {
        "uv": uv,
        "right_uv": right_uv,
        "angle_rad": float(angle_rad),
    }


def average_eval_pixel_measurements(measurements):
    if not measurements:
        raise ValueError("measurements is empty")

    uv_values = np.asarray([m["uv"] for m in measurements], dtype=np.float64)
    uv = np.median(uv_values, axis=0)

    angles = np.asarray([m["angle_rad"] for m in measurements], dtype=np.float64)
    angle_rad = math.atan2(
        float(np.mean(np.sin(angles))),
        float(np.mean(np.cos(angles))),
    )

    return {
        "uv": uv,
        "angle_rad": float(angle_rad),
        "frame_count": int(len(measurements)),
    }


# =============================================================================
# ChArUco capture
# =============================================================================


def start_color_pipeline_with_fallback(color_profiles):
    """
    指定順にRealSense color profileを試して起動する。

    color_profiles:
        [(width, height, fps), ...]
    """
    errors = []

    for width, height, fps in color_profiles:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            int(width),
            int(height),
            rs.format.bgr8,
            int(fps),
        )

        try:
            profile = pipeline.start(config)
            print(
                f"[CAMERA] color stream started: "
                f"{width}x{height} @ {fps} fps"
            )
            return pipeline, profile, int(width), int(height), int(fps)
        except Exception as exc:
            errors.append(f"{width}x{height}@{fps}: {exc}")
            try:
                pipeline.stop()
            except Exception:
                pass

    raise RuntimeError(
        "RealSense color streamを開始できませんでした。\n"
        + "\n".join(errors)
    )


def capture_charuco_pose(
    arm: XArm7,
    stage_name: str,
    *,
    point_board_m,
    point_label: str,
    require_eval_pixel: bool = False,
):
    # リーチング前後とも1280x720を使用する。
    # D435iで利用可能なFPSを上から順に試す。
    color_profiles = [
        (POST_COLOR_WIDTH, POST_COLOR_HEIGHT, fps)
        for fps in COLOR_FPS_CANDIDATES
    ]

    if require_eval_pixel:
        pose_average_window = POST_POSE_AVERAGE_WINDOW
        min_poses_for_average = POST_MIN_POSES_FOR_AVERAGE
    else:
        pose_average_window = PRE_POSE_AVERAGE_WINDOW
        min_poses_for_average = PRE_MIN_POSES_FOR_AVERAGE

    (
        pipeline,
        profile,
        stream_width,
        stream_height,
        stream_fps,
    ) = start_color_pipeline_with_fallback(color_profiles)

    intr = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .get_intrinsics()
    )

    reprojection_scale = (
        float(stream_width) / REPROJECTION_REFERENCE_WIDTH_PX
    )
    max_mean_reprojection_error_px = (
        MAX_MEAN_REPROJECTION_ERROR_PX * reprojection_scale
    )
    camera_matrix, dist_coeffs = get_camera_matrix_and_dist(intr)
    board, dictionary, detector_params, detector = create_charuco_board_and_detector(
        camera_matrix,
        dist_coeffs,
    )

    point_board_m = np.asarray(point_board_m, dtype=np.float64).reshape(3)
    pose_buffer = deque(maxlen=pose_average_window)
    metadata_buffer = deque(maxlen=pose_average_window)
    eval_pixel_buffer = deque(maxlen=pose_average_window)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    preprocess_mode = IMAGE_PREPROCESS_MODE
    if preprocess_mode not in IMAGE_PREPROCESS_MODES:
        raise ValueError(
            f"IMAGE_PREPROCESS_MODE={preprocess_mode!r} is invalid; "
            f"choose from {IMAGE_PREPROCESS_MODES}"
        )

    # base座標表示は毎フレーム変換すると重いため、一定周期で更新する。
    board_base_mm_live = None
    last_base_update_time = 0.0
    BASE_DISPLAY_UPDATE_INTERVAL_S = 0.20

    print("\n======================================")
    print(stage_name)
    print("Position source : ChArUco + solvePnP")
    print("Depth           : NOT USED")
    print(
        "Color stream    :",
        f"{stream_width} x {stream_height} @ {stream_fps} fps",
    )
    print(
        "Reproj threshold:",
        f"mean <= {max_mean_reprojection_error_px:.2f} px "
        f"(scale={reprojection_scale:.2f})",
    )
    print("Dictionary      :", CHARUCO_DICT_NAME)
    print("Board           :", f"{CHARUCO_SQUARES_X} x {CHARUCO_SQUARES_Y}")
    print(
        "Square / marker :",
        f"{CHARUCO_SQUARE_LENGTH_M*1000:.1f} / "
        f"{CHARUCO_MARKER_LENGTH_M*1000:.1f} mm",
    )
    print(f"{point_label} board mm :", point_board_m * 1000.0)
    if require_eval_pixel:
        print("Required ChArUco IDs:", EVAL_CHARUCO_ID, EVAL_RIGHT_CHARUCO_ID)
    print("ENTER : average pose and confirm")
    print("r     : reset average buffer")
    print("p     : gray / clahe / adaptive を切り替える")
    print("ESC/q : cancel")
    print("======================================")

    # リーチング前後でOpenCVウィンドウの表示サイズを統一する。
    cv2.namedWindow(stage_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(stage_name, POST_COLOR_WIDTH, POST_COLOR_HEIGHT)
    if SHOW_PREPROCESSED_WINDOW:
        processed_window_name = f"{stage_name} - processed"
        cv2.namedWindow(processed_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            processed_window_name,
            POST_COLOR_WIDTH,
            POST_COLOR_HEIGHT,
        )

    try:
        warmup_frames = max(
            5,
            int(round(CAMERA_WARMUP_SECONDS * stream_fps)),
        )
        for _ in range(warmup_frames):
            pipeline.wait_for_frames()

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())

            # 計算はRealSenseの元画像で行う。表示用背景だけ先に180度回転し、
            # この後の枠・点・文字は回転後の座標で描く。
            debug = cv2.rotate(color, cv2.ROTATE_180)
            display_height, display_width = debug.shape[:2]

            gray_raw = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            gray_processed = preprocess_charuco_gray(
                gray_raw,
                preprocess_mode,
                clahe,
            )

            charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco(
                gray_processed,
                board,
                dictionary,
                detector_params,
                detector,
                camera_matrix,
                dist_coeffs,
            )

            if marker_ids is not None and len(marker_ids) > 0:
                marker_corners_display = rotate_corner_array_180_for_display(
                    marker_corners, display_width, display_height
                )
                cv2.aruco.drawDetectedMarkers(
                    debug, marker_corners_display, marker_ids
                )

            eval_pixel_measurement = get_eval_pixel_measurement(
                charuco_corners, charuco_ids
            )

            pose_result = None
            if charuco_ids is not None and len(charuco_ids) > 0:
                charuco_corners_display = rotate_corner_array_180_for_display(
                    charuco_corners, display_width, display_height
                )
                cv2.aruco.drawDetectedCornersCharuco(
                    debug,
                    charuco_corners_display,
                    charuco_ids,
                    (0, 0, 255),
                )
                pose_result = estimate_charuco_pose_pnp(
                    board,
                    charuco_corners,
                    charuco_ids,
                    camera_matrix,
                    dist_coeffs,
                    reprojection_scale=reprojection_scale,
                )

            if pose_result is not None:
                mean_error = pose_result["mean_reprojection_error_px"]
                accepted = mean_error <= max_mean_reprojection_error_px
                T_now = pose_result["T_cam_board_m"]

                if accepted:
                    if pose_buffer:
                        T_previous = pose_buffer[-1]
                        translation_jump_mm = np.linalg.norm(
                            T_now[:3, 3] - T_previous[:3, 3]
                        ) * 1000.0
                        rotation_jump_deg = rotation_angle_deg(
                            T_previous[:3, :3],
                            T_now[:3, :3],
                        )

                        if (
                            translation_jump_mm > BUFFER_RESET_TRANSLATION_MM
                            or rotation_jump_deg > BUFFER_RESET_ROTATION_DEG
                        ):
                            pose_buffer.clear()
                            metadata_buffer.clear()
                            eval_pixel_buffer.clear()

                    pose_buffer.append(T_now.copy())
                    metadata_buffer.append(pose_result.copy())
                    if eval_pixel_measurement is not None:
                        eval_pixel_buffer.append(eval_pixel_measurement)

                # 座標軸描画は無効化する。
                # drawFrameAxes() は軸端点が画像外に出ると毎フレーム警告を出すが、
                # PnP・リーチング・CSV記録には不要。

                point_info = compute_board_point_and_roll(
                    pose_result,
                    camera_matrix,
                    dist_coeffs,
                    point_board_m,
                )
                point_uv = point_info["point_uv"]
                point_x_uv = point_info["point_x_uv"]

                point_uv_display = rotate_uv_180_for_display(
                    point_uv, display_width, display_height
                )
                point_x_uv_display = rotate_uv_180_for_display(
                    point_x_uv, display_width, display_height
                )

                cv2.circle(
                    debug,
                    tuple(np.round(point_uv_display).astype(int)),
                    7,
                    (0, 255, 255),
                    -1,
                )
                cv2.line(
                    debug,
                    tuple(np.round(point_uv_display).astype(int)),
                    tuple(np.round(point_x_uv_display).astype(int)),
                    (255, 0, 255),
                    3,
                )

                if eval_pixel_measurement is not None:
                    direct_uv = eval_pixel_measurement["uv"]
                    direct_right_uv = eval_pixel_measurement["right_uv"]
                    direct_uv_display = rotate_uv_180_for_display(
                        direct_uv, display_width, display_height
                    )
                    direct_right_uv_display = rotate_uv_180_for_display(
                        direct_right_uv, display_width, display_height
                    )
                    cv2.circle(
                        debug,
                        tuple(np.round(direct_uv_display).astype(int)),
                        7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.line(
                        debug,
                        tuple(np.round(direct_uv_display).astype(int)),
                        tuple(np.round(direct_right_uv_display).astype(int)),
                        (0, 255, 0),
                        2,
                    )

                point_cam_mm = point_info["point_camera_m"] * 1000.0
                status_color = (0, 255, 0) if accepted else (0, 0, 255)

                # 検出したボード原点のbase座標を定期更新して画面表示する。
                now_monotonic = time.monotonic()
                if (
                    board_base_mm_live is None
                    or now_monotonic - last_base_update_time
                    >= BASE_DISPLAY_UPDATE_INTERVAL_S
                ):
                    try:
                        board_camera_mm = pose_result["tvec_m"] * 1000.0
                        board_base_mm_live = map_camera_point_to_base_mm(
                            arm,
                            board_camera_mm,
                        )
                        last_base_update_time = now_monotonic
                    except Exception as exc:
                        board_base_mm_live = None
                        last_base_update_time = now_monotonic
                        print(f"[WARN] live base coordinate failed: {exc}")

                # 表示を簡潔にするため、corners / projected uv / pose bufferは表示しない。
                put_text(
                    debug,
                    f"reproj {mean_error:.2f}/"
                    f"{pose_result['max_reprojection_error_px']:.2f} px",
                    28,
                    status_color,
                )
                put_text(
                    debug,
                    f"{point_label} cam X={point_cam_mm[0]:.1f} "
                    f"Y={point_cam_mm[1]:.1f} Z={point_cam_mm[2]:.1f} mm",
                    55,
                    status_color,
                )

                if board_base_mm_live is not None:
                    put_text(
                        debug,
                        f"board base X={board_base_mm_live[0]:.1f} "
                        f"Y={board_base_mm_live[1]:.1f} "
                        f"Z={board_base_mm_live[2]:.1f} mm",
                        82,
                        (255, 255, 0),
                    )

                # 到達後評価では、ArUco版と同じroll/droll計算を表示する。
                # roll = ROLL_SIGN * [-90, 90]へ正規化した画像上角度 + offset
                # droll = roll - 基準roll
                if require_eval_pixel and eval_pixel_measurement is not None:
                    u_now = float(eval_pixel_measurement["uv"][0])
                    v_now = float(eval_pixel_measurement["uv"][1])
                    roll_now_deg = math.degrees(
                        normalize_roll_for_board(
                            eval_pixel_measurement["angle_rad"]
                        )
                    )
                    du_px_live = u_now - EVAL_REF_U_PX
                    dv_px_live = v_now - EVAL_REF_V_PX
                    droll_deg_live = roll_now_deg - EVAL_REF_ROLL_DEG

                    uv_mm_live = pixel_delta_to_image_plane_mm(
                        current_uv_px=(u_now, v_now),
                        reference_uv_px=(EVAL_REF_U_PX, EVAL_REF_V_PX),
                        depth_m=point_info["point_camera_m"][2],
                        camera_matrix=camera_matrix,
                        dist_coeffs=dist_coeffs,
                    )

                    put_text(
                        debug,
                        f"u={u_now:.2f} v={v_now:.2f} "
                        f"roll={roll_now_deg:+.2f} deg",
                        109,
                        (0, 255, 255),
                    )
                    put_text(
                        debug,
                        f"du={du_px_live:+.2f} dv={dv_px_live:+.2f} px "
                        f"droll={droll_deg_live:+.2f} deg",
                        136,
                        (0, 255, 255),
                    )

                    if uv_mm_live is not None:
                        du_mm_live, dv_mm_live = uv_mm_live["delta_uv_mm"]
                        put_text(
                            debug,
                            f"du_mm={du_mm_live:+.2f} "
                            f"dv_mm={dv_mm_live:+.2f} mm",
                            163,
                            (0, 255, 255),
                        )
                        put_text(
                            debug,
                            f"uv_error={uv_mm_live['error_mm']:.2f} mm "
                            f"(Z={uv_mm_live['depth_mm']:.1f} mm)",
                            190,
                            (0, 255, 255),
                        )
            else:
                visible_count = 0 if charuco_ids is None else len(charuco_ids)
                put_text(
                    debug,
                    f"Need >= {MIN_CHARUCO_CORNERS} corners; now {visible_count}",
                    35,
                    (0, 0, 255),
                )

            put_text(
                debug,
                f"preprocess={preprocess_mode}  (p: change)  "
                f"{stream_width}x{stream_height}@{stream_fps}",
                244,
                (255, 255, 255),
                scale=0.50,
            )

            # debugは、背景を180度回転した後に枠・点・文字を描いた画像。
            # ここでは再回転しない。
            cv2.imshow(stage_name, debug)
            if SHOW_PREPROCESSED_WINDOW:
                display_processed = cv2.rotate(
                    gray_processed, cv2.ROTATE_180
                )
                cv2.imshow(f"{stage_name} - processed", display_processed)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                print("Canceled")
                return None

            if key == ord("r"):
                pose_buffer.clear()
                metadata_buffer.clear()
                eval_pixel_buffer.clear()
                print("[BUFFER RESET]")
                continue

            if key == ord("p"):
                current_index = IMAGE_PREPROCESS_MODES.index(preprocess_mode)
                preprocess_mode = IMAGE_PREPROCESS_MODES[
                    (current_index + 1) % len(IMAGE_PREPROCESS_MODES)
                ]
                pose_buffer.clear()
                metadata_buffer.clear()
                eval_pixel_buffer.clear()
                board_base_mm_live = None
                print(f"[PREPROCESS] switched to {preprocess_mode}")
                print("[BUFFER RESET] preprocessing mode changed")
                continue

            if key in (10, 13):
                if not pose_buffer:
                    print("有効なChArUco姿勢がありません。")
                    continue

                if require_eval_pixel and not eval_pixel_buffer:
                    print(
                        f"評価交点ID={EVAL_CHARUCO_ID}と右隣ID="
                        f"{EVAL_RIGHT_CHARUCO_ID}が同時に検出されていません。"
                    )
                    continue

                if len(pose_buffer) < min_poses_for_average:
                    print(
                        f"[WARN] 平均フレーム数が少ないです: "
                        f"{len(pose_buffer)}/{min_poses_for_average}"
                    )

                T_cam_board_average_m = average_transforms(list(pose_buffer))
                latest = metadata_buffer[-1]

                average_rvec, _ = cv2.Rodrigues(
                    T_cam_board_average_m[:3, :3]
                )
                average_pose_result = latest.copy()
                average_pose_result["T_cam_board_m"] = T_cam_board_average_m
                average_pose_result["rvec"] = average_rvec.reshape(3)
                average_pose_result["tvec_m"] = T_cam_board_average_m[:3, 3].copy()

                point_info = compute_board_point_and_roll(
                    average_pose_result,
                    camera_matrix,
                    dist_coeffs,
                    point_board_m,
                )

                if eval_pixel_buffer:
                    direct_eval = average_eval_pixel_measurements(
                        list(eval_pixel_buffer)
                    )
                else:
                    direct_eval = {
                        "uv": point_info["point_uv"].copy(),
                        "angle_rad": float(point_info["image_angle_rad"]),
                        "frame_count": 0,
                    }

                return {
                    "T_cam_board_m": T_cam_board_average_m,
                    "rvec": average_rvec.reshape(3),
                    "tvec_m": T_cam_board_average_m[:3, 3].copy(),
                    "point_board_m": point_board_m.copy(),
                    "point_camera_m": point_info["point_camera_m"],
                    "point_uv_projected": point_info["point_uv"],
                    "point_uv": direct_eval["uv"],
                    "image_angle_rad_projected": point_info["image_angle_rad"],
                    "image_angle_rad": direct_eval["angle_rad"],
                    "d_roll_rad": point_info["d_roll_rad"],
                    "direct_pixel_frame_count": direct_eval["frame_count"],
                    "mean_reprojection_error_px": float(
                        np.mean(
                            [
                                item["mean_reprojection_error_px"]
                                for item in metadata_buffer
                            ]
                        )
                    ),
                    "max_reprojection_error_px": float(
                        np.max(
                            [
                                item["max_reprojection_error_px"]
                                for item in metadata_buffer
                            ]
                        )
                    ),
                    "detected_corner_count": int(
                        latest["detected_corner_count"]
                    ),
                    "used_corner_count": int(latest["used_corner_count"]),
                    "charuco_ids": latest["charuco_ids"].copy(),
                    "averaged_frame_count": int(len(pose_buffer)),
                    "stream_width": int(stream_width),
                    "stream_height": int(stream_height),
                    "stream_fps": int(stream_fps),
                    "reprojection_scale": float(reprojection_scale),
                    "camera_matrix": camera_matrix.copy(),
                    "dist_coeffs": dist_coeffs.copy(),
                }
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


# =============================================================================
# base conversion and evaluation
# =============================================================================


def map_camera_point_to_base_mm(arm, point_camera_mm):
    result = cam_mm_to_robot_mm(
        arm,
        np.asarray(point_camera_mm, dtype=np.float64).reshape(3),
        handeye_json_path=HANDEYE_JSON_PATH,
    )
    result = np.asarray(result, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("cam_mm_to_robot_mm() の返り値にNaN/Infがあります。")
    return result


def estimate_T_base_cam_from_existing_mapper(arm, axis_probe_mm=100.0):
    p0 = map_camera_point_to_base_mm(arm, [0.0, 0.0, 0.0])
    px = map_camera_point_to_base_mm(arm, [axis_probe_mm, 0.0, 0.0])
    py = map_camera_point_to_base_mm(arm, [0.0, axis_probe_mm, 0.0])
    pz = map_camera_point_to_base_mm(arm, [0.0, 0.0, axis_probe_mm])

    raw_x = (px - p0) / axis_probe_mm
    raw_y = (py - p0) / axis_probe_mm
    raw_z = (pz - p0) / axis_probe_mm
    raw_rotation = np.column_stack([raw_x, raw_y, raw_z])
    rotation = orthonormalize_rotation(raw_rotation)

    return make_transform(rotation, p0)


def convert_board_pose_to_base_mm(arm, T_cam_board_m):
    T_cam_board_mm = np.asarray(T_cam_board_m, dtype=np.float64).copy()
    T_cam_board_mm[:3, 3] *= 1000.0

    T_base_cam_mm = estimate_T_base_cam_from_existing_mapper(arm)
    T_base_board_mm = T_base_cam_mm @ T_cam_board_mm

    return T_cam_board_mm, T_base_cam_mm, T_base_board_mm


def build_post_reaching_evaluation(arm, post_capture):
    eval_camera_mm = np.asarray(
        post_capture["point_camera_m"], dtype=np.float64
    ).reshape(3) * 1000.0
    eval_base_mm = map_camera_point_to_base_mm(arm, eval_camera_mm)

    eval_uv = np.asarray(post_capture["point_uv"], dtype=np.float64).reshape(2)

    # ArUco版と同じ方法:
    # 画像上の右向き辺角度を[-90, 90]へ寄せ、
    # ROLL_SIGNとROLL_OFFSET_RADを適用した値をrollとする。
    eval_roll_deg = math.degrees(
        normalize_roll_for_board(post_capture["image_angle_rad"])
    )

    ref_uv = np.array([EVAL_REF_U_PX, EVAL_REF_V_PX], dtype=np.float64)
    delta_uv = eval_uv - ref_uv
    pixel_error = float(np.linalg.norm(delta_uv))
    droll_deg = eval_roll_deg - EVAL_REF_ROLL_DEG

    uv_mm = pixel_delta_to_image_plane_mm(
        current_uv_px=eval_uv,
        reference_uv_px=ref_uv,
        depth_m=eval_camera_mm[2] / 1000.0,
        camera_matrix=post_capture["camera_matrix"],
        dist_coeffs=post_capture["dist_coeffs"],
    )

    return {
        "eval_point_board_mm": EVAL_POINT_BOARD_M * 1000.0,
        "eval_charuco_id": int(EVAL_CHARUCO_ID),
        "eval_point_camera_mm": eval_camera_mm,
        "eval_point_base_mm": eval_base_mm,
        "eval_uv_px": eval_uv,
        "eval_roll_deg": float(eval_roll_deg),
        "ref_uv_px": ref_uv,
        "ref_roll_deg": float(EVAL_REF_ROLL_DEG),
        "delta_uv_px": delta_uv,
        "pixel_error_px": pixel_error,
        "delta_uv_mm": (
            None if uv_mm is None else uv_mm["delta_uv_mm"]
        ),
        "image_plane_error_mm": (
            None if uv_mm is None else uv_mm["error_mm"]
        ),
        "evaluation_depth_mm": (
            None if uv_mm is None else uv_mm["depth_mm"]
        ),
        "droll_deg": float(droll_deg),
        "direct_pixel_frame_count": int(
            post_capture["direct_pixel_frame_count"]
        ),
        "mean_reprojection_error_px": post_capture[
            "mean_reprojection_error_px"
        ],
        "max_reprojection_error_px": post_capture[
            "max_reprojection_error_px"
        ],
        "detected_corner_count": post_capture["detected_corner_count"],
        "used_corner_count": post_capture["used_corner_count"],
        "averaged_frame_count": post_capture["averaged_frame_count"],
    }


def print_post_reaching_evaluation(evaluation):
    print("\n========== POST-REACH CHARUCO EVALUATION ==========")
    print(
        f"evaluation square = col {EVAL_SQUARE_COL_1BASED}, "
        f"row {EVAL_SQUARE_ROW_1BASED}, top-left corner"
    )
    print("evaluation board point [mm] =", evaluation["eval_point_board_mm"])
    print("ChArUco corner ID =", evaluation["eval_charuco_id"])
    print("---------------------------------------------------")
    print("[detected image coordinate]")
    print("u, v [px] =", evaluation["eval_uv_px"])
    print("roll [deg] =", evaluation["eval_roll_deg"])
    print("---------------------------------------------------")
    print("[temporary reference]")
    print("u, v [px] =", evaluation["ref_uv_px"])
    print("roll [deg] =", evaluation["ref_roll_deg"])
    print("---------------------------------------------------")
    print("[error]")
    print("du, dv [px] =", evaluation["delta_uv_px"])
    print("2D pixel error [px] =", evaluation["pixel_error_px"])
    if evaluation["delta_uv_mm"] is not None:
        print("du, dv image-plane [mm] =", evaluation["delta_uv_mm"])
        print("2D image-plane error [mm] =", evaluation["image_plane_error_mm"])
        print("evaluation depth [mm] =", evaluation["evaluation_depth_mm"])
    print("droll [deg] =", evaluation["droll_deg"])
    print("---------------------------------------------------")
    print("[evaluation point: camera frame]")
    print("position [mm] =", evaluation["eval_point_camera_mm"])
    print("[evaluation point: robot/base frame]")
    print("position [mm] =", evaluation["eval_point_base_mm"])
    print("---------------------------------------------------")
    print(
        "reprojection mean/max [px] = "
        f"{evaluation['mean_reprojection_error_px']:.3f} / "
        f"{evaluation['max_reprojection_error_px']:.3f}"
    )
    print(
        "corners detected/used = "
        f"{evaluation['detected_corner_count']} / "
        f"{evaluation['used_corner_count']}"
    )
    print("pose averaged frames =", evaluation["averaged_frame_count"])
    print("direct pixel averaged frames =", evaluation["direct_pixel_frame_count"])
    print("===================================================\n")


def append_result_csv(pre_capture, target_base_mm, evaluation):
    csv_path = Path(RESULT_CSV_PATH)
    file_exists = csv_path.exists()

    pre_target_camera_mm = np.asarray(
        pre_capture["point_camera_m"], dtype=np.float64
    ).reshape(3) * 1000.0
    target_base_mm = np.asarray(target_base_mm, dtype=np.float64).reshape(3)

    header = [
        "timestamp",
        "pre_target_cam_x_mm",
        "pre_target_cam_y_mm",
        "pre_target_cam_z_mm",
        "command_target_base_x_mm",
        "command_target_base_y_mm",
        "command_target_base_z_mm",
        "command_droll_deg",
        "du_px",
        "dv_px",
        "pixel_error_px",
        "du_mm",
        "dv_mm",
        "uv_error_mm",
        "droll_deg",
        "post_reprojection_mean_px",
        "post_detected_corner_count",
    ]

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        *[f"{value:.3f}" for value in pre_target_camera_mm],
        *[f"{value:.3f}" for value in target_base_mm],
        f"{math.degrees(pre_capture['d_roll_rad']):.3f}",
        f"{evaluation['delta_uv_px'][0]:.3f}",
        f"{evaluation['delta_uv_px'][1]:.3f}",
        f"{evaluation['pixel_error_px']:.3f}",
        (
            ""
            if evaluation["delta_uv_mm"] is None
            else f"{evaluation['delta_uv_mm'][0]:.3f}"
        ),
        (
            ""
            if evaluation["delta_uv_mm"] is None
            else f"{evaluation['delta_uv_mm'][1]:.3f}"
        ),
        (
            ""
            if evaluation["image_plane_error_mm"] is None
            else f"{evaluation['image_plane_error_mm']:.3f}"
        ),
        f"{evaluation['droll_deg']:.3f}",
        f"{evaluation['mean_reprojection_error_px']:.6f}",
        str(evaluation["detected_corner_count"]),
    ]

    if len(header) != len(row):
        raise RuntimeError(
            f"CSV column mismatch: header={len(header)}, row={len(row)}"
        )

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

    print(f"[LOG SAVED] {csv_path}")


# =============================================================================
# reaching sequence
# =============================================================================


def reach_charuco_center_square_top_left(arm: XArm7, side: str = "right"):
    pre_capture = capture_charuco_pose(
        arm,
        "BEFORE REACH: detect 7x5 ChArUco target",
        point_board_m=TARGET_POINT_BOARD_M,
        point_label="target",
        require_eval_pixel=False,
    )
    if pre_capture is None:
        raise RuntimeError("ChArUco target capture canceled or failed")

    target_camera_mm = pre_capture["point_camera_m"] * 1000.0
    d_roll_rad = float(pre_capture["d_roll_rad"])

    print("\n========== CAMERA TARGET ==========")
    print("target board [mm]  =", TARGET_POINT_BOARD_M * 1000.0)
    print("target camera [mm] =", target_camera_mm)
    print("target pixel [px]  =", pre_capture["point_uv"])
    print("board image angle [deg] =", math.degrees(pre_capture["image_angle_rad"]))
    print("d_roll [deg] =", math.degrees(d_roll_rad))
    print("===================================\n")

    # デバッグ表示と実際の変換で同じhand-eye JSONを必ず使用する。
    print_camera_debug_info(
        arm,
        target_camera_mm,
        handeye_json_path=HANDEYE_JSON_PATH,
    )

    target_base_mm = cam_mm_to_robot_mm(
        arm,
        target_camera_mm,
        handeye_json_path=HANDEYE_JSON_PATH,
    )
    target_base_mm = np.asarray(target_base_mm, dtype=np.float64).reshape(3)

    print("========== ROBOT TARGET ==========")
    print("p_robot_mm =", target_base_mm)
    print("d_roll_rad =", d_roll_rad)
    print("d_roll_deg =", math.degrees(d_roll_rad))
    print("side       =", side)
    print("==================================")



    ret = arm.move_to_target_xyz_and_roll(
        p_robot_mm=target_base_mm,
        d_roll_rad=d_roll_rad,
        side=side,
    )
    print("move_to_target_xyz_and_roll returned:", ret)
    print("ChArUco target reaching done")

    post_capture = capture_charuco_pose(
        arm,
        "AFTER REACH: evaluate ChArUco corner",
        point_board_m=EVAL_POINT_BOARD_M,
        point_label="eval_2x3",
        require_eval_pixel=True,
    )
    if post_capture is None:
        print("Post-reaching ChArUco evaluation canceled")
        evaluation = None
    else:
        evaluation = build_post_reaching_evaluation(arm, post_capture)
        print_post_reaching_evaluation(evaluation)
        append_result_csv(pre_capture, target_base_mm, evaluation)

    ret2 = moveJ_to_return_pose_direct(
        arm,
        joint_deg=RETURN_JOINT_DEG,
        speed=20,
        mvacc=200,
        wait=True,
    )
    print("return pose returned:", ret2)
    print("returned to calibration return pose")

    return {
        "pre_capture": pre_capture,
        "target_base_mm": target_base_mm,
        "move_return": ret,
        "post_evaluation": evaluation,
        "return_move_return": ret2,
    }


# =============================================================================
# main
# =============================================================================


def main():
    config = load_config(CONFIG_PATH)

    rclpy.init()
    node = rclpy.create_node("charuco_7x5_single_board_reaching")

    xarm_host = config["robot"]["xarm"]["host"]
    arm = XArm7(node=node, host=xarm_host)
    globals()["arm"] = arm

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        print("xArm ready")
        print("side =", SIDE)
        print("handeye JSON =", HANDEYE_JSON_PATH)
        print("target board point [mm] =", TARGET_POINT_BOARD_M * 1000.0)
        print("evaluation board point [mm] =", EVAL_POINT_BOARD_M * 1000.0)
        print("evaluation ChArUco ID =", EVAL_CHARUCO_ID)

        recover_xarm_if_possible(arm)
        reach_charuco_center_square_top_left(
            arm=arm,
            side=SIDE,
        )

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


if __name__ == "__main__":
    main()