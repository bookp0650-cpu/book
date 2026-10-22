#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
簡易ChArUco自動サンプリング 完全修正版 v2

動作:
  1. 実行開始時の現在TCP位置姿勢を基準姿勢とする。
  2. 基準姿勢でChArUcoを計測し、ベース座標系のボード姿勢をCSVに保存する。
  3. 基準TCP位置からX±150 mm、Y/Z±100 mmの楕円体内にランダムな位置を20個作る。
  4. 各位置で、基準RPYから各軸±15 deg以内の姿勢を5個計測する。
  5. 合計100姿勢をCSVへ逐次保存する。
  6. 各位置の計測後は基準姿勢へ戻る。

必要ファイル:
  - このファイル
  - 同じディレクトリの charuco_board_pose_to_base.py

結果:
  charuco_variation_plots/sampling_csv/
    charuco_handeye_samples_YYYYMMDD_HHMMSS.csv

各実行につき、日時付きCSVを1ファイルだけ生成する。
xarm7直下にはCSVを生成せず、既存CSVのバックアップ・リネームも行わない。

実行:
  VS Codeの「Pythonファイルを実行」ボタンを押す。
  コマンドライン引数は不要で、実行直後に実機サンプリングを開始する。

注意:
  - MoveItによる衝突回避は行わない。
  - 基準姿勢からX±150 mm・Y/Z±100 mm・各RPY±15 degが安全であることを事前確認する。
  - 最初は下記定数を NUM_POSITIONS=2, POSITION_X_RADIUS_MM=20, POSITION_Y_RADIUS_MM=20,
    POSITION_Z_RADIUS_MM=20,
    ORIENTATION_LIMIT_DEG=3 に落として確認することを推奨する。
"""

from __future__ import annotations

import csv
import math
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy

# xarm7/ または xarm7/charuco_variation_plots/ のどちらに置いても動作する。
SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "charuco_variation_plots":
    XARM7_DIR = SCRIPT_DIR.parent
else:
    XARM7_DIR = SCRIPT_DIR

PROJECT_ROOT = XARM7_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(XARM7_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import charuco_board_pose_to_base as cb
from xarm7.control.xarm7 import XArm7


# =========================================================
# 設定: JSONは使わず、ここだけ変更する
# =========================================================

SAMPLER_VERSION = "2.3.0"

NUM_POSITIONS = 40
ORIENTATIONS_PER_POSITION = 5
POSITION_X_RADIUS_MM = 150.0
POSITION_Y_RADIUS_MM = 100.0
POSITION_Z_RADIUS_MM = 100.0
ORIENTATION_LIMIT_DEG = 15.0
RANDOM_SEED = 20260714

TCP_VELOCITY_MM_S = 30.0
TCP_ACCELERATION_MM_S2 = 40.0
SETTLE_TIME_S = 1.0
MAX_TRANSLATION_PER_STEP_MM = 25.0
MAX_ROTATION_PER_STEP_DEG = 5.0

AVERAGE_FRAMES = 10
CAPTURE_TIMEOUT_S = 8.0
MIN_CHARUCO_CORNERS = 8
MAX_MEAN_REPROJECTION_ERROR_PX = 1.5
MAX_POINT_REPROJECTION_ERROR_PX = 4.0

MAX_POSITION_ATTEMPTS = 100
MAX_ORIENTATION_ATTEMPTS_PER_POSITION = 30
MIN_POSITION_SEPARATION_MM = 20.0

OUTPUT_ROOT_DIR = (
    SCRIPT_DIR
    if SCRIPT_DIR.name == "charuco_variation_plots"
    else XARM7_DIR / "charuco_variation_plots"
)
SAMPLING_CSV_DIR = OUTPUT_ROOT_DIR / "sampling_csv"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = (
    SAMPLING_CSV_DIR
    / f"charuco_handeye_samples_{RUN_TIMESTAMP}.csv"
)

COLOR_WIDTH = 640
COLOR_HEIGHT = 480
COLOR_FPS = 30
WARMUP_FRAMES = 30


# =========================================================
# global / exception
# =========================================================

ARM: XArm7 | None = None
MOTION_STARTED = False


class UserAbort(RuntimeError):
    pass


def signal_handler(sig, frame):
    print("\nCtrl+C detected")
    if ARM is not None:
        try:
            ARM.emergency_stop()
        except Exception:
            pass
    os._exit(1)


signal.signal(signal.SIGINT, signal_handler)


# =========================================================
# robot helper
# =========================================================


def get_sdk_arm(arm: XArm7):
    return getattr(arm, "arm", arm)


def get_tcp_pose(arm: XArm7) -> np.ndarray:
    pose = arm.get_tcp_pose(is_radian=True)
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.size < 6:
        raise RuntimeError(f"TCP pose must have 6 elements: {pose}")
    return pose[:6].copy()


def get_joint_deg(arm: XArm7) -> np.ndarray:
    joints = arm.get_joint_angle(is_radian=False)
    joints = np.asarray(joints, dtype=np.float64).reshape(-1)
    if joints.size < 7:
        raise RuntimeError(f"joint angles must have 7 elements: {joints}")
    return joints[:7].copy()


def check_robot(arm: XArm7) -> None:
    sdk = get_sdk_arm(arm)

    code, state = sdk.get_state()
    if code != 0 or int(state) >= 4:
        raise RuntimeError(f"xArm state abnormal: code={code}, state={state}")

    result = sdk.get_err_warn_code()
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        code = int(result[0])
        values = np.asarray(result[1]).reshape(-1)
        error = int(values[0]) if values.size >= 1 else 0
        warn = int(values[1]) if values.size >= 2 else 0
        if code != 0 or error != 0:
            raise RuntimeError(
                f"xArm error: api_code={code}, error={error}, warn={warn}"
            )


def enable_robot(arm: XArm7) -> None:
    sdk = get_sdk_arm(arm)
    sdk.clean_warn()
    sdk.clean_error()
    sdk.motion_enable(enable=True)
    sdk.set_mode(0)
    sdk.set_state(0)
    time.sleep(0.5)
    check_robot(arm)


def moveL_absolute(arm: XArm7, target_pose: np.ndarray) -> None:
    """target_pose = [x,y,z,roll,pitch,yaw], xyz[mm], rpy[rad]"""
    global MOTION_STARTED
    check_robot(arm)
    target = np.asarray(target_pose, dtype=np.float64).reshape(6)

    # この時点以降の例外は、移動中または移動後に起きた可能性がある。
    MOTION_STARTED = True
    ret = arm._moveL(
        target.tolist(),
        velocity=TCP_VELOCITY_MM_S,
        acceleration=TCP_ACCELERATION_MM_S2,
        asynchronous=False,
    )
    code = int(ret[0]) if isinstance(ret, (tuple, list)) else int(ret or 0)
    if code != 0:
        raise RuntimeError(f"moveL failed: {ret}")
    check_robot(arm)


def wrap_angle_delta_rad(delta: np.ndarray | float):
    """角度差を[-pi, pi]へ正規化し、無駄な大回りを防ぐ。"""
    value = np.asarray(delta, dtype=np.float64)
    wrapped = np.arctan2(np.sin(value), np.cos(value))
    if np.ndim(delta) == 0:
        return float(wrapped)
    return wrapped


def rotation_difference_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    delta = wrap_angle_delta_rad(
        np.asarray(pose_b, dtype=np.float64)[3:6]
        - np.asarray(pose_a, dtype=np.float64)[3:6]
    )
    return float(np.max(np.abs(np.degrees(delta))))


def move_interpolated(arm: XArm7, target_pose: np.ndarray) -> None:
    """
    大きい移動を小分けにする。

    RPYは±pi境界をまたいでも、現在姿勢から最短角度で補間する。
    例: +179 deg -> -179 deg を358 deg回転ではなく2 deg回転にする。
    """
    start = get_tcp_pose(arm)
    requested_target = np.asarray(target_pose, dtype=np.float64).reshape(6)

    xyz_delta = requested_target[:3] - start[:3]
    rpy_delta = wrap_angle_delta_rad(requested_target[3:6] - start[3:6])

    translation_mm = float(np.linalg.norm(xyz_delta))
    rotation_deg = float(np.max(np.abs(np.degrees(rpy_delta))))

    steps = max(
        1,
        math.ceil(translation_mm / MAX_TRANSLATION_PER_STEP_MM),
        math.ceil(rotation_deg / MAX_ROTATION_PER_STEP_DEG),
    )

    for step in range(1, steps + 1):
        alpha = step / steps
        intermediate = np.empty(6, dtype=np.float64)
        intermediate[:3] = start[:3] + alpha * xyz_delta
        intermediate[3:6] = start[3:6] + alpha * rpy_delta
        moveL_absolute(arm, intermediate)

    time.sleep(SETTLE_TIME_S)


# =========================================================
# transform helper
# =========================================================


def rpy_to_rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64).reshape(3)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def pose_to_transform_mm(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_rotation(pose[3:6])
    T[:3, 3] = pose[:3]
    return T


def matrix_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{r}{c}" for r in range(4) for c in range(4)]


def flatten_matrix(T: np.ndarray) -> list[float]:
    return np.asarray(T, dtype=np.float64).reshape(16).tolist()


# =========================================================
# ChArUco camera
# =========================================================


class CharucoCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            COLOR_WIDTH,
            COLOR_HEIGHT,
            rs.format.bgr8,
            COLOR_FPS,
        )
        profile = self.pipeline.start(config)
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intr = stream.get_intrinsics()
        self.camera_matrix, self.dist_coeffs = cb.get_camera_matrix_and_dist(
            self.intr
        )
        (
            self.board,
            self.dictionary,
            self.detector_params,
            self.detector,
        ) = cb.create_charuco_board_and_detector(
            self.camera_matrix,
            self.dist_coeffs,
        )

        for _ in range(WARMUP_FRAMES):
            self.pipeline.wait_for_frames()

    def close(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()

    def detect_one(self, label: str):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None

        image = np.asanyarray(color_frame.get_data())
        debug = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, marker_corners, marker_ids = cb.detect_charuco(
            gray,
            self.board,
            self.dictionary,
            self.detector_params,
            self.detector,
            self.camera_matrix,
            self.dist_coeffs,
        )

        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(debug, marker_corners, marker_ids)

        result = None
        if charuco_ids is not None and len(charuco_ids) >= MIN_CHARUCO_CORNERS:
            cv2.aruco.drawDetectedCornersCharuco(
                debug, charuco_corners, charuco_ids, (0, 0, 255)
            )
            result = cb.estimate_charuco_pose_pnp(
                self.board,
                charuco_corners,
                charuco_ids,
                self.camera_matrix,
                self.dist_coeffs,
            )

        visible = 0 if charuco_ids is None else len(charuco_ids)
        cb.put_text(debug, label, 28)
        cb.put_text(debug, f"corners={visible}", 56)

        if result is not None:
            try:
                cv2.drawFrameAxes(
                    debug,
                    self.camera_matrix,
                    self.dist_coeffs,
                    result["rvec"],
                    result["tvec_m"],
                    0.08,
                    3,
                )
            except Exception:
                pass
            cb.put_text(
                debug,
                (
                    f"reproj={result['mean_reprojection_error_px']:.2f}/"
                    f"{result['max_reprojection_error_px']:.2f}px"
                ),
                84,
            )

        cv2.imshow("ChArUco simple sampler", debug)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            raise UserAbort("camera window canceled")

        return result

    def capture_average(self, label: str):
        transforms: list[np.ndarray] = []
        metadata: list[dict] = []
        deadline = time.monotonic() + CAPTURE_TIMEOUT_S

        while time.monotonic() < deadline and len(transforms) < AVERAGE_FRAMES:
            result = self.detect_one(
                f"{label} {len(transforms)}/{AVERAGE_FRAMES}"
            )
            if result is None:
                continue

            accepted = bool(
                result["used_corner_count"] >= MIN_CHARUCO_CORNERS
                and result["mean_reprojection_error_px"]
                <= MAX_MEAN_REPROJECTION_ERROR_PX
                and result["max_reprojection_error_px"]
                <= MAX_POINT_REPROJECTION_ERROR_PX
            )
            if accepted:
                transforms.append(result["T_cam_board_m"].copy())
                metadata.append(result.copy())

        if len(transforms) < AVERAGE_FRAMES:
            return None

        latest = metadata[-1]
        return {
            "T_cam_board_m": cb.average_transforms(transforms),
            "mean_reprojection_error_px": float(
                np.mean([m["mean_reprojection_error_px"] for m in metadata])
            ),
            "max_reprojection_error_px": float(
                np.max([m["max_reprojection_error_px"] for m in metadata])
            ),
            "used_corner_count": int(latest["used_corner_count"]),
            "detected_corner_count": int(latest["detected_corner_count"]),
            "averaged_frames": len(transforms),
        }


# =========================================================
# sampling / CSV
# =========================================================


def sample_position_offset(rng: np.random.Generator) -> np.ndarray:
    """
    基準TCPを中心とする楕円体内部で体積一様にサンプリングする。

    X: ±150 mm
    Y: ±100 mm
    Z: ±100 mm

    条件:
        (x/150)^2 + (y/100)^2 + (z/100)^2 <= 1
    """
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)

    # 単位球内部で体積一様にする半径
    radius = rng.random() ** (1.0 / 3.0)
    unit_ball_offset = direction * radius

    axis_radii_mm = np.array(
        [
            POSITION_X_RADIUS_MM,
            POSITION_Y_RADIUS_MM,
            POSITION_Z_RADIUS_MM,
        ],
        dtype=np.float64,
    )
    return unit_ball_offset * axis_radii_mm


def position_is_distinct(offset: np.ndarray, previous: list[np.ndarray]) -> bool:
    return all(
        np.linalg.norm(offset - existing) >= MIN_POSITION_SEPARATION_MM
        for existing in previous
    )


def csv_header() -> list[str]:
    return [
        "timestamp",
        "sample_id",
        "sample_type",
        "position_index",
        "orientation_index",
        "offset_x_mm",
        "offset_y_mm",
        "offset_z_mm",
        "offset_roll_deg",
        "offset_pitch_deg",
        "offset_yaw_deg",
        "tcp_x_mm",
        "tcp_y_mm",
        "tcp_z_mm",
        "tcp_roll_deg",
        "tcp_pitch_deg",
        "tcp_yaw_deg",
        "marker_base_x_mm",
        "marker_base_y_mm",
        "marker_base_z_mm",
        "marker_base_roll_deg",
        "marker_base_pitch_deg",
        "marker_base_yaw_deg",
        "mean_reprojection_error_px",
        "max_reprojection_error_px",
        "used_corner_count",
        "detected_corner_count",
        "averaged_frames",
        *matrix_columns("T_base_tcp"),
        *matrix_columns("T_cam_board_m"),
        *matrix_columns("T_base_board_mm"),
    ]


def unpack_base_conversion(converted):
    """
    charuco_board_pose_to_base.py の複数バージョンに対応する。

    対応する戻り値:
      5個: T_cam_board_mm, T_base_cam_mm, T_base_board_mm,
           T_base_board_center_mm, diagnostics
      4個: T_cam_board_mm, T_base_cam_mm, T_base_board_mm, diagnostics
      dict: 同名キーを持つ辞書
    """
    if isinstance(converted, dict):
        required = ("T_cam_board_mm", "T_base_cam_mm", "T_base_board_mm")
        missing = [key for key in required if key not in converted]
        if missing:
            raise RuntimeError(
                "convert_T_cam_board_to_base() result is missing keys: "
                + ", ".join(missing)
            )
        return (
            np.asarray(converted["T_cam_board_mm"], dtype=np.float64),
            np.asarray(converted["T_base_cam_mm"], dtype=np.float64),
            np.asarray(converted["T_base_board_mm"], dtype=np.float64),
            (
                None
                if converted.get("T_base_board_center_mm") is None
                else np.asarray(
                    converted["T_base_board_center_mm"], dtype=np.float64
                )
            ),
            converted.get("diagnostics", converted.get("base_transform_diagnostics")),
        )

    if not isinstance(converted, (tuple, list)):
        raise RuntimeError(
            "convert_T_cam_board_to_base() must return tuple/list/dict; "
            f"got {type(converted).__name__}"
        )

    if len(converted) == 5:
        T_cam_board_mm, T_base_cam_mm, T_base_board_mm, center, diagnostics = converted
    elif len(converted) == 4:
        T_cam_board_mm, T_base_cam_mm, T_base_board_mm, diagnostics = converted
        center = None
    else:
        raise RuntimeError(
            "convert_T_cam_board_to_base() returned "
            f"{len(converted)} values; expected 4 or 5"
        )

    return (
        np.asarray(T_cam_board_mm, dtype=np.float64),
        np.asarray(T_base_cam_mm, dtype=np.float64),
        np.asarray(T_base_board_mm, dtype=np.float64),
        None if center is None else np.asarray(center, dtype=np.float64),
        diagnostics,
    )


def prepare_sample(
    *,
    sample_type: str,
    position_index: int,
    orientation_index: int,
    position_offset_mm: np.ndarray,
    orientation_offset_deg: np.ndarray,
    actual_tcp_pose: np.ndarray,
    capture: dict,
    arm: XArm7,
) -> dict:
    """その姿勢にいる間にベース座標変換まで確定して保持する。"""
    T_base_tcp = pose_to_transform_mm(actual_tcp_pose)
    converted = cb.convert_T_cam_board_to_base(
        arm,
        capture["T_cam_board_m"],
    )
    (
        _T_cam_board_mm,
        _T_base_cam_mm,
        T_base_board_mm,
        _T_base_board_center_mm,
        _diagnostics,
    ) = unpack_base_conversion(converted)

    return {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "sample_type": sample_type,
        "position_index": position_index,
        "orientation_index": orientation_index,
        "position_offset_mm": np.asarray(position_offset_mm, dtype=np.float64).copy(),
        "orientation_offset_deg": np.asarray(
            orientation_offset_deg, dtype=np.float64
        ).copy(),
        "actual_tcp_pose": np.asarray(actual_tcp_pose, dtype=np.float64).copy(),
        "capture": capture,
        "T_base_tcp": T_base_tcp,
        "T_base_board_mm": T_base_board_mm,
    }


def append_prepared_sample_csv(sample_id: int, prepared: dict) -> None:
    SAMPLING_CSV_DIR.mkdir(parents=True, exist_ok=True)

    actual_tcp_pose = prepared["actual_tcp_pose"]
    capture = prepared["capture"]
    T_base_board_mm = prepared["T_base_board_mm"]
    marker_rpy_deg = cb.rotation_matrix_to_rpy_deg(T_base_board_mm[:3, :3])

    row = [
        prepared["timestamp"],
        sample_id,
        prepared["sample_type"],
        prepared["position_index"],
        prepared["orientation_index"],
        *prepared["position_offset_mm"].tolist(),
        *prepared["orientation_offset_deg"].tolist(),
        *actual_tcp_pose[:3].tolist(),
        *np.degrees(actual_tcp_pose[3:6]).tolist(),
        *T_base_board_mm[:3, 3].tolist(),
        *marker_rpy_deg.tolist(),
        capture["mean_reprojection_error_px"],
        capture["max_reprojection_error_px"],
        capture["used_corner_count"],
        capture["detected_corner_count"],
        capture["averaged_frames"],
        *flatten_matrix(prepared["T_base_tcp"]),
        *flatten_matrix(capture["T_cam_board_m"]),
        *flatten_matrix(T_base_board_mm),
    ]

    file_exists = OUTPUT_CSV.exists()
    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(csv_header())
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())

    print(
        f"[SAVED] sample={sample_id}, type={prepared['sample_type']}, "
        f"position={prepared['position_index']}, "
        f"orientation={prepared['orientation_index']}, "
        f"marker_base={T_base_board_mm[:3, 3]}"
    )


# =========================================================
# main sampling
# =========================================================


def return_to_reference_safely(arm: XArm7, reference_pose: np.ndarray) -> None:
    """復帰失敗で元の例外を隠さないための安全な基準姿勢復帰。"""
    print("[RETURN] reference pose")
    try:
        move_interpolated(arm, reference_pose)
    except Exception as exc:
        print(f"[RETURN FAILED] {exc}")
        try:
            arm.emergency_stop()
        except Exception:
            pass
        raise


def run_sampling(arm: XArm7, camera: CharucoCamera):
    rng = np.random.default_rng(RANDOM_SEED)

    SAMPLING_CSV_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[CSV OUTPUT] {OUTPUT_CSV}")

    reference_pose = get_tcp_pose(arm)
    print("\n========== REFERENCE TCP ==========")
    print("xyz [mm] =", reference_pose[:3])
    print("rpy [deg] =", np.degrees(reference_pose[3:6]))
    print("===================================\n")

    # まず基準姿勢で記録。
    reference_capture = camera.capture_average("reference")
    if reference_capture is None:
        raise RuntimeError(
            "基準姿勢でChArUcoを安定検出できません。"
            "ボードを見える位置へ置いてから再実行してください。"
        )

    actual_reference_pose = get_tcp_pose(arm)
    reference_prepared = prepare_sample(
        sample_type="reference",
        position_index=0,
        orientation_index=0,
        position_offset_mm=np.zeros(3),
        orientation_offset_deg=np.zeros(3),
        actual_tcp_pose=actual_reference_pose,
        capture=reference_capture,
        arm=arm,
    )
    append_prepared_sample_csv(0, reference_prepared)

    valid_random_samples = 0
    position_index = 0
    accepted_positions: list[np.ndarray] = []
    position_attempts = 0

    while position_index < NUM_POSITIONS:
        if position_attempts >= MAX_POSITION_ATTEMPTS:
            raise RuntimeError(
                f"位置候補を{MAX_POSITION_ATTEMPTS}回試しても、"
                f"{NUM_POSITIONS}位置を収集できませんでした。"
            )
        position_attempts += 1

        position_offset = sample_position_offset(rng)
        if not position_is_distinct(position_offset, accepted_positions):
            continue

        target_xyz = reference_pose[:3] + position_offset
        position_base_pose = reference_pose.copy()
        position_base_pose[:3] = target_xyz

        print("\n=======================================")
        print(f"POSITION {position_index + 1}/{NUM_POSITIONS}")
        print("offset [mm] =", position_offset)
        print("target xyz [mm] =", target_xyz)
        print("=======================================")

        try:
            # まず基準姿勢角のまま、その位置へ平行移動。
            move_interpolated(arm, position_base_pose)

            # この位置でマーカーが最低限見えるか確認。
            visibility = camera.capture_average(
                f"position {position_index + 1} visibility"
            )
            if visibility is None:
                print("[REJECT POSITION] marker not visible")
                # finally節で基準姿勢へ1回だけ戻る。
                continue

            valid_at_position = 0
            orientation_attempts = 0
            pending_samples: list[dict] = []

            while valid_at_position < ORIENTATIONS_PER_POSITION:
                if (
                    orientation_attempts
                    >= MAX_ORIENTATION_ATTEMPTS_PER_POSITION
                ):
                    print(
                        f"[REJECT POSITION] only {valid_at_position}/"
                        f"{ORIENTATIONS_PER_POSITION} orientations accepted"
                    )
                    break

                orientation_attempts += 1
                orientation_offset_deg = rng.uniform(
                    -ORIENTATION_LIMIT_DEG,
                    ORIENTATION_LIMIT_DEG,
                    size=3,
                )

                target_pose = position_base_pose.copy()
                target_pose[3:6] = (
                    reference_pose[3:6]
                    + np.radians(orientation_offset_deg)
                )

                print(
                    f"[TRY] position={position_index + 1}, "
                    f"orientation={valid_at_position + 1}/"
                    f"{ORIENTATIONS_PER_POSITION}, "
                    f"offset_rpy_deg={orientation_offset_deg}"
                )

                move_interpolated(arm, target_pose)
                capture = camera.capture_average(
                    f"P{position_index + 1} "
                    f"R{valid_at_position + 1}"
                )

                if capture is None:
                    print("[REJECT ORIENTATION] ChArUco quality failed")
                    # 次の候補は同一位置・基準姿勢から始める。
                    move_interpolated(arm, position_base_pose)
                    continue

                actual_pose = get_tcp_pose(arm)
                valid_at_position += 1
                pending_samples.append(
                    prepare_sample(
                        sample_type="random",
                        position_index=position_index + 1,
                        orientation_index=valid_at_position,
                        position_offset_mm=position_offset,
                        orientation_offset_deg=orientation_offset_deg,
                        actual_tcp_pose=actual_pose,
                        capture=capture,
                        arm=arm,
                    )
                )

            # 5姿勢すべて揃った位置だけCSVへ書き込む。
            # 途中までしか取れなかった位置は丸ごと破棄するため、
            # 最終的なランダムサンプル数は必ず20 x 5 = 100になる。
            if valid_at_position == ORIENTATIONS_PER_POSITION:
                for pending in pending_samples:
                    valid_random_samples += 1
                    append_prepared_sample_csv(valid_random_samples, pending)
                accepted_positions.append(position_offset.copy())
                position_index += 1
            else:
                print(
                    "[DISCARD POSITION] 5姿勢揃わなかったため、"
                    "この位置の一時データはCSVへ保存しません。"
                )

        finally:
            # 毎位置、必ず基準姿勢へ戻る。
            return_to_reference_safely(arm, reference_pose)

    print("\n========== FINISHED ==========")
    print(f"Reference samples : 1")
    print(f"Random samples    : {valid_random_samples}")
    print(f"CSV               : {OUTPUT_CSV}")
    print("==============================")


# =========================================================
# main
# =========================================================


def main() -> int:
    global ARM

    print(f"ChArUco sampler version: {SAMPLER_VERSION}")
    print("[START] VS Code direct execution mode")
    print("[WARNING] The robot will move immediately after initialization.")

    rclpy.init()
    node = rclpy.create_node("charuco_simple_random_sampler")
    camera: CharucoCamera | None = None

    try:
        ARM = XArm7(node)
        enable_robot(ARM)
        camera = CharucoCamera()
        run_sampling(ARM, camera)
        return 0

    except UserAbort as exc:
        print(f"Canceled: {exc}")
        return 1

    except Exception as exc:
        print(f"ABORT: {exc}")
        traceback.print_exc()
        # 移動開始後の例外だけ非常停止する。
        # 基準姿勢の画像処理など、移動前のソフトウェアエラーでは
        # アームを不要にdisableしない。
        if ARM is not None and MOTION_STARTED:
            try:
                ARM.emergency_stop()
            except Exception:
                pass
        else:
            print("[INFO] Robot motion had not started; emergency stop was not issued.")
        return 2

    finally:
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())