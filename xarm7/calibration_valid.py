#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

# =========================================================
# import path fix
# /home/book/pro_book/pro_hand_book_python を import パスに追加
# このファイル:
# /home/book/pro_book/pro_hand_book_python/xarm7/calibration_valid.py
# =========================================================
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os
import signal
import time
import cv2
import yaml
import rclpy
import numpy as np
import pyrealsense2 as rs

from rclpy.executors import MultiThreadedExecutor

from xarm7.control.xarm7 import XArm7
from xarm7.control.robot_base_coordinate import (
    cam_mm_to_robot_mm,
    print_camera_debug_info,
)
import csv
from datetime import datetime


# =========================================================
# user settings
# =========================================================

RESULT_CSV_PATH = "reaching_result_log.csv"
CONFIG_PATH = "Retrieval_integration.yaml"

ARUCO_DICT_NAME = "DICT_4X4_1000"
TARGET_MARKER_ID = 0

# 実際に印刷したマーカーの一辺 [m]
# ただし今回はPnPを使わないので，主に表示・確認用
# Depth座標計算には直接使わない
MARKER_LENGTH_M = 0.150

# 右棚なら right，左棚なら left
SIDE = "right"

# roll方向が逆なら -1.0 にする
ROLL_SIGN = -1.0

# rollに固定オフセットを足したい場合
# 例: 90度足すなら np.deg2rad(90.0)
ROLL_OFFSET_RAD = 0.0

# Depth中央値を取る範囲
# window=5なら 11x11 pixel
DEPTH_WINDOW = 5

# Depthの許容範囲 [m]
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 2.0

# 移動前確認
ASK_BEFORE_MOVE = True


# =========================================================
# ID=1 reaching evaluation settings
# 完璧に合っている状態で5回計測した平均値
# =========================================================

EVAL_MARKER_ID = 1
EVAL_MARKER_LENGTH_M = 0.020  # 20 mm

EVAL_REF_U_PX = 349.100000
EVAL_REF_V_PX = 266.650000
EVAL_REF_ROLL_DEG = -0.344628



def append_reaching_result_csv(
    id0_camera_mm,
    id0_roll_deg,
    eval_result,
    csv_path=RESULT_CSV_PATH,
):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "id0_camera_x_mm",
                "id0_camera_y_mm",
                "id0_camera_z_mm",
                "id0_roll_deg",
                "id1_du_px",
                "id1_dv_px",
                "id1_du_mm",
                "id1_dv_mm",
                "id1_d_mm",
                "id1_droll_deg",
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{id0_camera_mm[0]:.3f}",
            f"{id0_camera_mm[1]:.3f}",
            f"{id0_camera_mm[2]:.3f}",
            f"{id0_roll_deg:.3f}",
            f"{eval_result['du_px']:.3f}",
            f"{eval_result['dv_px']:.3f}",
            f"{eval_result['du_mm']:.3f}",
            f"{eval_result['dv_mm']:.3f}",
            f"{eval_result['d_mm']:.3f}",
            f"{eval_result['droll_deg']:.3f}",
        ])

    print(f"[LOG SAVED] {csv_path}")


# =========================================================
# config
# =========================================================

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =========================================================
# emergency stop
# =========================================================

def sigint_handler(sig, frame):
    print("Ctrl+C detected → FORCE KILL")
    try:
        arm = globals().get("arm", None)
        if arm:
            arm.emergency_stop()
    except Exception:
        pass
    os._exit(1)


signal.signal(signal.SIGINT, sigint_handler)


# =========================================================
# xArm recovery helper
# =========================================================

def try_call(obj, name, *args, **kwargs):
    """
    XArm7の実装差を吸収するための安全呼び出し。
    メソッドが無ければ何もしない。
    """
    fn = getattr(obj, name, None)
    if fn is None:
        return None

    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[WARN] {name} failed: {e}")
        return None


def recover_xarm_if_possible(arm):
    """
    前回 Ctrl+C / emergency_stop 後の状態をできるだけ復帰する。
    XArm7クラス側に存在するメソッドだけ実行される。
    """
    print("Try xArm recovery...")

    # wrapper名の違いに備えて複数候補
    try_call(arm, "clean_error")
    try_call(arm, "clean_warn")
    try_call(arm, "motion_enable", enable=True)
    try_call(arm, "set_mode", 0)
    try_call(arm, "set_state", 0)

    # XArm7内部にarmなどを持っている場合の保険
    inner = getattr(arm, "arm", None)
    if inner is not None:
        try_call(inner, "clean_error")
        try_call(inner, "clean_warn")
        try_call(inner, "motion_enable", enable=True)
        try_call(inner, "set_mode", 0)
        try_call(inner, "set_state", 0)

    time.sleep(0.5)


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

    # OpenCV 4.7以降
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return detector, aruco_dict

    # 古いOpenCV
    params = cv2.aruco.DetectorParameters_create()
    return None, aruco_dict


# =========================================================
# angle
# =========================================================

def normalize_angle_rad(a):
    return np.arctan2(np.sin(a), np.cos(a))


def normalize_roll_for_marker(image_angle_rad):
    """
    画像上のArUco上辺角度をroll補正量にする。

    注意:
    これはPnP姿勢ではなく，画像上の2D傾きだけを使う。
    """
    a = normalize_angle_rad(image_angle_rad)

    # -90〜90 degに寄せる
    if a > np.pi / 2:
        a -= np.pi
    elif a < -np.pi / 2:
        a += np.pi

    d_roll = ROLL_SIGN * a + ROLL_OFFSET_RAD
    return float(d_roll)


# =========================================================
# RealSense depth -> camera coordinate
# =========================================================

def pixel_to_camera_depth(u, v, depth_frame, intr, window=5):
    """
    OpenCV pixel座標(u, v) + RealSense Depthから
    camera座標 [X, Y, Z] [m] を返す。

    PnPは使わない。
    ZはRealSense depth_frame.get_distance()。
    X,YはRealSense color intrinsicsから計算。

    camera座標:
        X: 画像右
        Y: 画像下
        Z: カメラ前方
    """
    depths = []

    h = depth_frame.get_height()
    w = depth_frame.get_width()

    for dy in range(-window, window + 1):
        for dx in range(-window, window + 1):
            uu = int(round(u + dx))
            vv = int(round(v + dy))

            if uu < 0 or uu >= w or vv < 0 or vv >= h:
                continue

            d = depth_frame.get_distance(uu, vv)

            if DEPTH_MIN_M < d < DEPTH_MAX_M:
                depths.append(float(d))

    if len(depths) == 0:
        return None

    Z = float(np.median(depths))
    X = (float(u) - intr.ppx) / intr.fx * Z
    Y = (float(v) - intr.ppy) / intr.fy * Z

    return np.array([X, Y, Z], dtype=np.float64)


# =========================================================
# ArUco detection only
# =========================================================

def detect_aruco_marker_2d(color, detector, aruco_dict):
    """
    ArUcoを2D検出するだけ。
    PnPなし。
    Depthなし。

    return:
        {
            center_uv,
            image_angle_rad,
            d_roll_rad,
            debug
        }
    """
    debug = color.copy()
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    if ids is None or len(ids) == 0:
        cv2.putText(
            debug,
            "No ArUco detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        return None, debug

    ids_flat = ids.flatten()

    target_index = None
    for i, marker_id in enumerate(ids_flat):
        if int(marker_id) == TARGET_MARKER_ID:
            target_index = i
            break

    cv2.aruco.drawDetectedMarkers(debug, corners, ids)

    if target_index is None:
        cv2.putText(
            debug,
            f"Target ID {TARGET_MARKER_ID} not found",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )
        return None, debug

    marker_corners = corners[target_index].reshape(4, 2)
    marker_id = int(ids_flat[target_index])

    # 中心pixel
    center_uv = marker_corners.mean(axis=0)
    u, v = center_uv

    # ArUcoのcorner順:
    # 0: top-left, 1: top-right, 2: bottom-right, 3: bottom-left
    # 上辺の画像上角度だけ使う
    p0 = marker_corners[0]
    p1 = marker_corners[1]
    image_angle_rad = np.arctan2(
        float(p1[1] - p0[1]),
        float(p1[0] - p0[0]),
    )

    d_roll_rad = normalize_roll_for_marker(image_angle_rad)

    # debug draw
    cv2.circle(debug, (int(u), int(v)), 6, (0, 255, 0), -1)

    cv2.line(
        debug,
        tuple(marker_corners[0].astype(int)),
        tuple(marker_corners[1].astype(int)),
        (255, 0, 0),
        3,
    )

    cv2.putText(
        debug,
        f"ID={marker_id}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        debug,
        f"center u={u:.1f}, v={v:.1f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        debug,
        f"img angle={np.degrees(image_angle_rad):.1f} deg",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        debug,
        f"d_roll={np.degrees(d_roll_rad):.1f} deg",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    result = {
        "center_uv": center_uv.astype(np.float64),
        "image_angle_rad": float(image_angle_rad),
        "d_roll_rad": float(d_roll_rad),
    }

    return result, debug


# =========================================================
# RealSense loop
# =========================================================

def run_capture_and_aruco_center_depth():
    """
    RealSenseを起動してOpenCV画面を出す。
    Enterを押した瞬間に、
      1. ArUco中心pixel
      2. RealSense Depth
      3. camera座標 target_m
      4. 画像上傾き d_roll_rad
    を返す。

    PnPは完全に使わない。
    """
    detector, aruco_dict = create_aruco_detector()

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    # depthをcolorにalign
    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()

    print("===================================")
    print("RealSense started")
    print("ArUco dictionary:", ARUCO_DICT_NAME)
    print("Target marker ID:", TARGET_MARKER_ID)
    print("Marker length [m]:", MARKER_LENGTH_M)
    print("fx :", intr.fx)
    print("fy :", intr.fy)
    print("ppx:", intr.ppx)
    print("ppy:", intr.ppy)
    print("===================================")
    print("ENTER : calculate target from RealSense Depth")
    print("ESC   : cancel")
    print("===================================")

    last_result = None
    last_depth_frame = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())

            result, debug = detect_aruco_marker_2d(
                color=color,
                detector=detector,
                aruco_dict=aruco_dict,
            )

            if result is not None:
                last_result = result
                last_depth_frame = depth_frame

                u, v = result["center_uv"]
                d = depth_frame.get_distance(int(round(u)), int(round(v)))

                cv2.putText(
                    debug,
                    f"center depth={d:.3f} m",
                    (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("aruco depth detection", debug)

            key = cv2.waitKey(1) & 0xFF

            # ESC
            if key == 27:
                print("Canceled by ESC")
                return None, None

            # ENTER
            if key in (10, 13):
                if last_result is None or last_depth_frame is None:
                    print("ArUcoが検出できていない")
                    continue

                u, v = last_result["center_uv"]

                target_m = pixel_to_camera_depth(
                    u,
                    v,
                    last_depth_frame,
                    intr,
                    window=DEPTH_WINDOW,
                )

                if target_m is None:
                    print("Depthが取得できない")
                    continue

                d_roll_rad = last_result["d_roll_rad"]

                print("")
                print("========== ARUCO TARGET ==========")
                print("POSITION SOURCE : RealSense Depth")
                print("PnP             : NOT USED")
                print(f"center pixel u={u:.2f}, v={v:.2f}")
                print(f"image angle = {np.degrees(last_result['image_angle_rad']):.2f} deg")
                print(f"d_roll      = {np.degrees(d_roll_rad):.2f} deg")
                print("target camera [m]  =", target_m)
                print("target camera [mm] =", target_m * 1000.0)
                print("==================================")
                print("")

                return d_roll_rad, target_m

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def run_id1_reaching_evaluation():
    """
    リーチング後にID=1マーカーを検出して，
    完璧位置の基準値との差分を画像座標[px]，mm，roll[deg]で評価する。
    ロボットは動かさない。

    mm換算はPnPではなくRealSense Depthを使う。
    """

    global TARGET_MARKER_ID
    global MARKER_LENGTH_M

    old_target_marker_id = TARGET_MARKER_ID
    old_marker_length_m = MARKER_LENGTH_M

    TARGET_MARKER_ID = EVAL_MARKER_ID
    MARKER_LENGTH_M = EVAL_MARKER_LENGTH_M

    detector, aruco_dict = create_aruco_detector()

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    fx = intr.fx
    fy = intr.fy

    print("")
    print("===================================")
    print("ID=1 REACHING EVALUATION START")
    print("POSITION SOURCE : RealSense Depth")
    print("PnP             : NOT USED")
    print("Target marker ID:", EVAL_MARKER_ID)
    print("Marker length [m]:", EVAL_MARKER_LENGTH_M)
    print("-----------------------------------")
    print("Reference values")
    print(f"EVAL_REF_U_PX    = {EVAL_REF_U_PX:.6f}")
    print(f"EVAL_REF_V_PX    = {EVAL_REF_V_PX:.6f}")
    print(f"EVAL_REF_ROLL_DEG= {EVAL_REF_ROLL_DEG:.6f}")
    print("-----------------------------------")
    print("ENTER : evaluate current ID=1 pose")
    print("ESC/q : cancel")
    print("===================================")

    last_result = None
    last_depth_frame = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())

            result, debug = detect_aruco_marker_2d(
                color=color,
                detector=detector,
                aruco_dict=aruco_dict,
            )

            if result is not None:
                last_result = result
                last_depth_frame = depth_frame

                u_now, v_now = result["center_uv"]
                roll_now_deg = np.degrees(result["d_roll_rad"])

                du_px_live = float(u_now - EVAL_REF_U_PX)
                dv_px_live = float(v_now - EVAL_REF_V_PX)
                droll_deg_live = float(roll_now_deg - EVAL_REF_ROLL_DEG)

                target_m_live = pixel_to_camera_depth(
                    u_now,
                    v_now,
                    depth_frame,
                    intr,
                    window=DEPTH_WINDOW,
                )

                cv2.putText(
                    debug,
                    f"du={du_px_live:+.2f}px dv={dv_px_live:+.2f}px droll={droll_deg_live:+.2f}deg",
                    (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )

                if target_m_live is not None:
                    id1_z_mm_live = float(target_m_live[2] * 1000.0)

                    du_mm_live = du_px_live * id1_z_mm_live / fx
                    dv_mm_live = dv_px_live * id1_z_mm_live / fy
                    d_mm_live = np.hypot(du_mm_live, dv_mm_live)

                    cv2.putText(
                        debug,
                        f"du_mm={du_mm_live:+.2f} dv_mm={dv_mm_live:+.2f} err={d_mm_live:.2f}mm",
                        (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                    )

                    cv2.putText(
                        debug,
                        f"ID1 Z={id1_z_mm_live:.1f} mm",
                        (20, 270),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                    )

            cv2.imshow("ID=1 reaching evaluation", debug)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                print("ID=1 evaluation canceled")
                return None

            if key in (10, 13):
                if last_result is None or last_depth_frame is None:
                    print("ID=1 ArUcoが検出できていない")
                    continue

                u_now, v_now = last_result["center_uv"]
                roll_now_deg = np.degrees(last_result["d_roll_rad"])

                du_px = float(u_now - EVAL_REF_U_PX)
                dv_px = float(v_now - EVAL_REF_V_PX)
                droll_deg = float(roll_now_deg - EVAL_REF_ROLL_DEG)

                target_m = pixel_to_camera_depth(
                    u_now,
                    v_now,
                    last_depth_frame,
                    intr,
                    window=DEPTH_WINDOW,
                )

                if target_m is None:
                    print("ID=1 Depthが取得できない")
                    continue

                id1_z_mm = float(target_m[2] * 1000.0)

                du_mm = du_px * id1_z_mm / fx
                dv_mm = dv_px * id1_z_mm / fy
                d_mm = np.hypot(du_mm, dv_mm)

                print("")
                print("========== REACHING EVALUATION ==========")
                print("[reference]")
                print(f"ref u      = {EVAL_REF_U_PX:.3f} px")
                print(f"ref v      = {EVAL_REF_V_PX:.3f} px")
                print(f"ref roll   = {EVAL_REF_ROLL_DEG:.3f} deg")
                print("-----------------------------------------")
                print("[current]")
                print(f"now u      = {u_now:.3f} px")
                print(f"now v      = {v_now:.3f} px")
                print(f"now roll   = {roll_now_deg:.3f} deg")
                print(f"du_mm      = {du_mm:+.3f} mm")
                print(f"dv_mm      = {dv_mm:+.3f} mm")
                print(f"2D error   = {d_mm:.3f} mm")
                print(f"id1_z_mm   = {id1_z_mm:.3f} mm")
                print("-----------------------------------------")
                print("[error]")
                print(f"du         = {du_px:+.3f} px")
                print(f"dv         = {dv_px:+.3f} px")
                print(f"droll      = {droll_deg:+.3f} deg")
                print("=========================================")
                print("")

                return {
                    "u_now": float(u_now),
                    "v_now": float(v_now),
                    "roll_now_deg": float(roll_now_deg),
                    "du_px": du_px,
                    "dv_px": dv_px,
                    "droll_deg": droll_deg,
                    "du_mm": du_mm,
                    "dv_mm": dv_mm,
                    "d_mm": d_mm,
                }

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

        TARGET_MARKER_ID = old_target_marker_id
        MARKER_LENGTH_M = old_marker_length_m


# =========================================================
# reaching like box
# =========================================================

RETURN_JOINT_DEG = [
    85.5,
    -55.1,
    174.1,
    64.7,
    39.7,
    8.8,
    43.6
]
def moveJ_to_return_pose_direct(
    arm,
    joint_deg=None,
    speed=20,
    mvacc=200,
    wait=True,
):
    """
    calibration_valid.py 内だけで使う退避姿勢移動。
    xarm7.py は変更しない。
    joint_deg 単位: deg
    """

    if joint_deg is None:
        joint_deg = RETURN_JOINT_DEG

    print("\n========== RETURN JOINT MOVE ==========")
    print("target joint deg =", joint_deg)
    print("speed =", speed)
    print("mvacc =", mvacc)
    print("=======================================\n")

    # XArm7クラスの中にSDK本体が arm.arm として入っている場合
    sdk_arm = getattr(arm, "arm", None)

    # もし arm.arm が無ければ arm._arm も見る
    if sdk_arm is None:
        sdk_arm = getattr(arm, "_arm", None)

    # それでも無ければ、XArm7自体が set_servo_angle を持っているか見る
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

def reach_aruco_center_like_box_depth(arm: XArm7, side: str = "right"):
    """
    リーチング箱と同じ処理:
      target_m
      -> target_mm
      -> cam_mm_to_robot_mm()
      -> move_to_target_xyz_and_roll()
    """
    d_roll_rad, target_m = run_capture_and_aruco_center_depth()

    if target_m is None:
        raise RuntimeError("ArUco recognition canceled or failed")

    target_mm = 1000.0 * target_m

    # ログ保存用
    id0_camera_mm = target_mm.copy()
    id0_roll_deg = float(np.degrees(d_roll_rad))

    print("========== CAMERA DEBUG ==========")
    print_camera_debug_info(
        arm,
        target_mm,
    )

    p_robot_mm = cam_mm_to_robot_mm(
        arm,
        target_mm,
    )

    print("========== ROBOT TARGET ==========")
    print("p_robot_mm =", p_robot_mm)
    print("d_roll_rad =", d_roll_rad)
    print("d_roll_deg =", np.degrees(d_roll_rad))
    print("side       =", side)
    print("==================================")

    if ASK_BEFORE_MOVE:
        ans = input("Move robot? [y/N]: ")
        if ans.lower() != "y":
            print("移動キャンセル")
            return

    # 重要:
    # XArmMonitorは認識待機中に作らない。
    # 待機中に state=5 を異常判定して emergency_stop するため。
    # move直前だけ作る。
    print("========== DIRECT MOVE CALL ==========")
    print("Calling arm.move_to_target_xyz_and_roll() directly...")
    print("p_robot_mm =", p_robot_mm)
    print("d_roll_rad =", d_roll_rad)
    print("d_roll_deg =", np.degrees(d_roll_rad))
    print("side =", side)
    print("======================================")

    ret = arm.move_to_target_xyz_and_roll(
        p_robot_mm=p_robot_mm,
        d_roll_rad=d_roll_rad,
        side=side,
    )

    print("move_to_target_xyz_and_roll returned:", ret)
    print("aruco center reaching done")

    input("EnterでID=1評価を開始 / Ctrl+Cで終了: ")

    eval_result = run_id1_reaching_evaluation()

    if eval_result is not None:
        print("ID=1 reaching evaluation finished")

        append_reaching_result_csv(
            id0_camera_mm=id0_camera_mm,
            id0_roll_deg=id0_roll_deg,
            eval_result=eval_result,
        )
    else:
        print("ID=1 reaching evaluation skipped or canceled")


    ret2 = moveJ_to_return_pose_direct(
        arm,
        joint_deg=RETURN_JOINT_DEG,
        speed=20,
        mvacc=200,
        wait=True,
    )

    print("return pose returned:", ret2)
    print("returned to calibration return pose")
# =========================================================
# main
# =========================================================

def main():
    config = load_config(CONFIG_PATH)

    rclpy.init()

    node = rclpy.create_node("aruco_center_depth_reaching_test")

    XARM_HOST = config["robot"]["xarm"]["host"]

    arm = XArm7(
        node=node,
        host=XARM_HOST,
    )
    globals()["arm"] = arm

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        print("xArm ready")
        print("side =", SIDE)

        recover_xarm_if_possible(arm)

        reach_aruco_center_like_box_depth(
            arm=arm,
            side=SIDE,
        )

    except KeyboardInterrupt:
        print("Interrupted by user")

    except Exception as e:
        print("Abort due to exception:")
        print(e)
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