#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ID=1 ArUco marker reference detector: 5-sample terminal version

Purpose:
  完璧にリーチングが合っている状態で，評価用マーカー ID=1 の
  画像中心座標 u, v [px] と画像上の傾き roll [deg] を5回取得し，
  平均値と標準偏差をターミナルに表示する。

Usage:
  python3 detect_id1_reference_pose_5samples_terminal.py

Controls:
  Enter : 現在検出しているID=1を1サンプルとして記録
          5回記録すると平均値を計算してターミナルに表示して終了
  r     : 記録済みサンプルをリセット
  ESC/q : 終了

Output:
  ターミナル表示のみ
  画像保存なし
"""

import math
import time

import cv2
import numpy as np
import pyrealsense2 as rs


# =========================================================
# User settings
# =========================================================

ARUCO_DICT_NAME = "DICT_4X4_1000"
TARGET_MARKER_ID = 1
MARKER_LENGTH_M = 0.050  # ID=1 marker size: 20 mm

# RealSense color stream
COLOR_WIDTH = 640
COLOR_HEIGHT = 480
COLOR_FPS = 30

# 5回計測
NUM_SAMPLES = 1

# roll direction
# 画像上の上辺角度をそのまま使うなら +1.0
# 既存のリーチングコードの d_roll と符号を合わせたいなら -1.0 にする
ROLL_SIGN = -1.0
ROLL_OFFSET_RAD = 0.0



# =========================================================
# ArUco dictionary
# =========================================================

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


def create_aruco_detector():
    aruco_dict = get_aruco_dict(ARUCO_DICT_NAME)

    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return detector, aruco_dict

    params = cv2.aruco.DetectorParameters_create()
    return None, aruco_dict


# =========================================================
# Angle helpers
# =========================================================

def normalize_angle_rad(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def normalize_marker_angle_rad(image_angle_rad: float) -> float:
    """
    ArUco上辺の画像上角度を -90〜+90 deg に寄せる。
    PnP姿勢ではなく，画像上で見た傾きのみを使う。
    """
    a = normalize_angle_rad(image_angle_rad)

    if a > math.pi / 2:
        a -= math.pi
    elif a < -math.pi / 2:
        a += math.pi

    return float(ROLL_SIGN * a + ROLL_OFFSET_RAD)


def angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """a-b を -180〜+180 deg に正規化"""
    d = math.radians(a_deg - b_deg)
    return math.degrees(math.atan2(math.sin(d), math.cos(d)))


# =========================================================
# Detection
# =========================================================

def detect_target_marker(color, detector, aruco_dict, sample_count: int):
    debug = color.copy()
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    if ids is None or len(ids) == 0:
        cv2.putText(debug, "No ArUco detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(debug, f"samples: {sample_count}/{NUM_SAMPLES}", (20, COLOR_HEIGHT - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return None, debug

    ids_flat = ids.flatten()
    cv2.aruco.drawDetectedMarkers(debug, corners, ids)

    target_index = None
    for i, marker_id in enumerate(ids_flat):
        if int(marker_id) == TARGET_MARKER_ID:
            target_index = i
            break

    if target_index is None:
        cv2.putText(debug, f"Target ID {TARGET_MARKER_ID} not found", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(debug, f"samples: {sample_count}/{NUM_SAMPLES}", (20, COLOR_HEIGHT - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return None, debug

    marker_corners = corners[target_index].reshape(4, 2).astype(np.float64)
    center_uv = marker_corners.mean(axis=0)
    u, v = float(center_uv[0]), float(center_uv[1])

    # OpenCV ArUco corner order:
    # 0: top-left, 1: top-right, 2: bottom-right, 3: bottom-left
    p0 = marker_corners[0]
    p1 = marker_corners[1]
    image_angle_rad = math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))
    roll_rad = normalize_marker_angle_rad(image_angle_rad)
    roll_deg = math.degrees(roll_rad)

    side_lengths_px = []
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        side_lengths_px.append(float(np.linalg.norm(marker_corners[b] - marker_corners[a])))
    marker_size_px = float(np.mean(side_lengths_px))
    mm_per_px = (MARKER_LENGTH_M * 1000.0) / marker_size_px if marker_size_px > 1e-9 else None

    cv2.circle(debug, (int(round(u)), int(round(v))), 6, (0, 255, 0), -1)
    cv2.line(debug, tuple(marker_corners[0].astype(int)), tuple(marker_corners[1].astype(int)), (255, 0, 0), 3)

    cv2.putText(debug, f"ID={TARGET_MARKER_ID}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(debug, f"center u={u:.3f}, v={v:.3f} px", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(debug, f"roll={roll_deg:.4f} deg", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(debug, f"size={marker_size_px:.3f} px, {mm_per_px:.6f} mm/px", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(debug, f"samples: {sample_count}/{NUM_SAMPLES}", (20, COLOR_HEIGHT - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(debug, "Enter: add sample / r: reset / ESC,q: quit", (20, COLOR_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    result = {
        "marker_id": TARGET_MARKER_ID,
        "u_px": u,
        "v_px": v,
        "roll_rad": roll_rad,
        "roll_deg": roll_deg,
        "image_angle_rad_raw": float(image_angle_rad),
        "image_angle_deg_raw": math.degrees(image_angle_rad),
        "marker_size_px": marker_size_px,
        "mm_per_px_reference": mm_per_px,
        "marker_length_m": MARKER_LENGTH_M,
        "corners_px": marker_corners.tolist(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return result, debug


# =========================================================
# Save helpers
# =========================================================

def circular_mean_deg(deg_values):
    rad = np.radians(np.asarray(deg_values, dtype=np.float64))
    s = np.mean(np.sin(rad))
    c = np.mean(np.cos(rad))
    return float(np.degrees(np.arctan2(s, c)))


def circular_std_deg(deg_values, mean_deg):
    diffs = [angle_diff_deg(v, mean_deg) for v in deg_values]
    return float(np.std(diffs, ddof=1)) if len(diffs) >= 2 else 0.0


def make_summary(samples):
    u_values = np.array([s["u_px"] for s in samples], dtype=np.float64)
    v_values = np.array([s["v_px"] for s in samples], dtype=np.float64)
    roll_values = [s["roll_deg"] for s in samples]
    size_values = np.array([s["marker_size_px"] for s in samples], dtype=np.float64)
    mm_per_px_values = np.array([s["mm_per_px_reference"] for s in samples], dtype=np.float64)

    roll_mean = circular_mean_deg(roll_values)

    return {
        "u_mean_px": float(np.mean(u_values)),
        "v_mean_px": float(np.mean(v_values)),
        "roll_mean_deg": roll_mean,
        "u_std_px": float(np.std(u_values, ddof=1)) if len(samples) >= 2 else 0.0,
        "v_std_px": float(np.std(v_values, ddof=1)) if len(samples) >= 2 else 0.0,
        "roll_std_deg": circular_std_deg(roll_values, roll_mean),
        "marker_size_mean_px": float(np.mean(size_values)),
        "marker_size_std_px": float(np.std(size_values, ddof=1)) if len(samples) >= 2 else 0.0,
        "mm_per_px_mean": float(np.mean(mm_per_px_values)),
        "mm_per_px_std": float(np.std(mm_per_px_values, ddof=1)) if len(samples) >= 2 else 0.0,
    }


def print_terminal_summary(samples, summary):
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 60)
    print("ID=1 REFERENCE POSE 5-SAMPLE RESULT")
    print(f"calculated_at          : {now}")
    print(f"target_marker_id       : {TARGET_MARKER_ID}")
    print(f"marker_length_m        : {MARKER_LENGTH_M}")
    print(f"aruco_dict             : {ARUCO_DICT_NAME}")
    print(f"color_stream           : {COLOR_WIDTH}x{COLOR_HEIGHT} @ {COLOR_FPS}fps")
    print(f"roll_sign              : {ROLL_SIGN}")
    print(f"roll_offset_rad        : {ROLL_OFFSET_RAD}")
    print("-" * 60)
    print("samples")
    for i, s in enumerate(samples, start=1):
        print(
            f"sample {i}: "
            f"u_px={s['u_px']:.6f}, "
            f"v_px={s['v_px']:.6f}, "
            f"roll_deg={s['roll_deg']:.6f}, "
            f"marker_size_px={s['marker_size_px']:.6f}, "
            f"mm_per_px={s['mm_per_px_reference']:.8f}"
        )
    print("-" * 60)
    print("summary")
    print(f"u_mean_px              : {summary['u_mean_px']:.6f}")
    print(f"v_mean_px              : {summary['v_mean_px']:.6f}")
    print(f"roll_mean_deg          : {summary['roll_mean_deg']:.6f}")
    print(f"u_std_px               : {summary['u_std_px']:.6f}")
    print(f"v_std_px               : {summary['v_std_px']:.6f}")
    print(f"roll_std_deg           : {summary['roll_std_deg']:.6f}")
    print(f"marker_size_mean_px    : {summary['marker_size_mean_px']:.6f}")
    print(f"marker_size_std_px     : {summary['marker_size_std_px']:.6f}")
    print(f"mm_per_px_mean         : {summary['mm_per_px_mean']:.8f}")
    print(f"mm_per_px_std          : {summary['mm_per_px_std']:.8f}")
    print("-" * 60)
    print("values for evaluation program")
    print(f"EVAL_REF_U_PX = {summary['u_mean_px']:.6f}")
    print(f"EVAL_REF_V_PX = {summary['v_mean_px']:.6f}")
    print(f"EVAL_REF_ROLL_DEG = {summary['roll_mean_deg']:.6f}")
    print("=" * 60 + "\n")


# =========================================================
# Main loop
# =========================================================

def main():
    detector, aruco_dict = create_aruco_detector()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, COLOR_FPS)

    profile = pipeline.start(config)
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()

    print("==========================================")
    print("ID=1 Reference Pose Detector: 5 samples terminal only")
    print("Robot motion      : NOT USED")
    print("Depth             : NOT USED")
    print("Target marker ID  :", TARGET_MARKER_ID)
    print("Marker length [m] :", MARKER_LENGTH_M)
    print("ArUco dictionary  :", ARUCO_DICT_NAME)
    print("Color stream      :", COLOR_WIDTH, COLOR_HEIGHT, COLOR_FPS)
    print("fx, fy            :", intr.fx, intr.fy)
    print("ppx, ppy          :", intr.ppx, intr.ppy)
    print("==========================================")
    print(f"Enter : add current ID=1 sample ({NUM_SAMPLES} times)")
    print("r     : reset samples")
    print("ESC/q : quit")
    print("==========================================")

    samples = []
    last_result = None
    last_debug = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            result, debug = detect_target_marker(color, detector, aruco_dict, len(samples))

            if result is not None:
                last_result = result
                last_debug = debug

            cv2.imshow("ID=1 reference detector 5 samples", debug)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                print("Quit")
                break

            if key == ord("r"):
                samples.clear()
                print("samples reset")
                continue

            if key in (10, 13):
                if last_result is None:
                    print("ID=1が検出できていないのでサンプル追加できない")
                    continue

                sample_index = len(samples) + 1
                sample = dict(last_result)
                samples.append(sample)

                print("\n========== SAMPLE ADDED ==========")
                print(f"sample {sample_index}/{NUM_SAMPLES}")
                print(f"u_px     = {sample['u_px']:.6f}")
                print(f"v_px     = {sample['v_px']:.6f}")
                print(f"roll_deg = {sample['roll_deg']:.6f}")
                print("==================================\n")

                if len(samples) >= NUM_SAMPLES:
                    summary = make_summary(samples)
                    print_terminal_summary(samples, summary)

                    break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
