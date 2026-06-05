#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

# =========================================================
# import path fix
# /home/book/pro_book/pro_hand_book_python を import パスに追加
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


# =========================================================
# user settings
# =========================================================

CONFIG_PATH = "Retrieval_integration.yaml"

ARUCO_DICT_NAME = "DICT_4X4_1000"
TARGET_MARKER_ID = 0

# 必ず実測値を入れる
# 印刷したArUcoの黒枠ではなく「マーカー一辺」の実寸[m]
# 150 mmなら 0.150
# 149 mmなら 0.149
MARKER_LENGTH_M = 0.150

SIDE = "right"

# roll方向が逆なら -1.0
ROLL_SIGN = -1.0

# rollに固定オフセットを入れたい場合
ROLL_OFFSET_RAD = 0.0

ASK_BEFORE_MOVE = True

# リーチング後に戻る関節角[deg]
RETURN_JOINT_DEG = [
    106.8,
    -28.0,
    149.7,
    52.4,
    32.2,
    28.8,
    51.6,
]


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
# xArm helper
# =========================================================

def try_call(obj, name, *args, **kwargs):
    fn = getattr(obj, name, None)
    if fn is None:
        return None

    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[WARN] {name} failed: {e}")
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
    """
    calibration_valid_pnp.py 内だけで使う退避姿勢移動。
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
# angle
# =========================================================

def normalize_angle_rad(a):
    return np.arctan2(np.sin(a), np.cos(a))


def normalize_roll_for_marker(image_angle_rad):
    """
    ArUco上辺の画像上角度をroll補正量にする。
    PnP姿勢のrollではなく，画像上の傾きだけを使う。
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
# camera intrinsics for PnP
# =========================================================

def get_camera_matrix_and_dist(intr):
    camera_matrix = np.array([
        [intr.fx, 0.0, intr.ppx],
        [0.0, intr.fy, intr.ppy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    dist_coeffs = np.array(intr.coeffs, dtype=np.float64).reshape(-1, 1)

    return camera_matrix, dist_coeffs


# =========================================================
# PnP
# =========================================================

def estimate_aruco_pose_pnp(marker_corners, intr):
    """
    ArUco 4隅からsolvePnPでマーカー中心のcamera座標を推定する。
    Depthは一切使わない。

    marker_corners:
        shape = (4, 2)
        corner order:
            0: top-left
            1: top-right
            2: bottom-right
            3: bottom-left

    return:
        rvec: marker姿勢
        tvec: marker中心のcamera座標[m]
    """

    L = MARKER_LENGTH_M
    h = L / 2.0

    # OpenCV ArUco corners の順序に合わせる
    obj_points = np.array([
        [-h,  h, 0.0],
        [ h,  h, 0.0],
        [ h, -h, 0.0],
        [-h, -h, 0.0],
    ], dtype=np.float64)

    img_points = marker_corners.astype(np.float64)

    camera_matrix, dist_coeffs = get_camera_matrix_and_dist(intr)

    # 正方形マーカーなのでIPPE_SQUAREを使用
    ok, rvec, tvec = cv2.solvePnP(
        obj_points,
        img_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not ok:
        return None, None

    return rvec.reshape(3), tvec.reshape(3)


# =========================================================
# ArUco 2D detection
# =========================================================

def detect_aruco_marker_2d(color, detector, aruco_dict):
    """
    ArUcoを2D検出する。
    PnP用に4隅を返す。
    Depthは使わない。

    return:
        result, debug
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

    center_uv = marker_corners.mean(axis=0)
    u, v = center_uv

    # 上辺の画像上角度
    p0 = marker_corners[0]
    p1 = marker_corners[1]
    image_angle_rad = np.arctan2(
        float(p1[1] - p0[1]),
        float(p1[0] - p0[0]),
    )

    d_roll_rad = normalize_roll_for_marker(image_angle_rad)

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
        "marker_id": marker_id,
        "center_uv": center_uv.astype(np.float64),
        "marker_corners": marker_corners.astype(np.float64),
        "image_angle_rad": float(image_angle_rad),
        "d_roll_rad": float(d_roll_rad),
    }

    return result, debug


# =========================================================
# RealSense loop PnP only
# =========================================================

def run_capture_and_aruco_center_pnp():
    """
    RealSense color画像からArUcoを検出し，
    ArUco 4隅 + カメラ内部パラメータ + 実寸マーカーサイズからPnPでcamera座標を出す。

    Depthは一切使わない。
    """
    detector, aruco_dict = create_aruco_detector()

    pipeline = rs.pipeline()
    config = rs.config()

    # PnP検証ではcolorだけ使う
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()

    print("===================================")
    print("RealSense started")
    print("POSITION SOURCE : PnP")
    print("Depth           : NOT USED")
    print("ArUco dictionary:", ARUCO_DICT_NAME)
    print("Target marker ID:", TARGET_MARKER_ID)
    print("Marker length [m]:", MARKER_LENGTH_M)
    print("fx :", intr.fx)
    print("fy :", intr.fy)
    print("ppx:", intr.ppx)
    print("ppy:", intr.ppy)
    print("coeffs:", intr.coeffs)
    print("===================================")
    print("ENTER : calculate target from PnP")
    print("ESC   : cancel")
    print("===================================")

    last_result = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())

            result, debug = detect_aruco_marker_2d(
                color=color,
                detector=detector,
                aruco_dict=aruco_dict,
            )

            if result is not None:
                marker_corners = result["marker_corners"]

                rvec, tvec = estimate_aruco_pose_pnp(
                    marker_corners=marker_corners,
                    intr=intr,
                )

                if rvec is not None and tvec is not None:
                    camera_matrix, dist_coeffs = get_camera_matrix_and_dist(intr)

                    try:
                        cv2.drawFrameAxes(
                            debug,
                            camera_matrix,
                            dist_coeffs,
                            rvec,
                            tvec,
                            MARKER_LENGTH_M * 0.5,
                        )
                    except Exception:
                        pass

                    cv2.putText(
                        debug,
                        f"PnP X={tvec[0]*1000:.1f} Y={tvec[1]*1000:.1f} Z={tvec[2]*1000:.1f} mm",
                        (20, 175),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )

                    result["pnp_rvec"] = rvec
                    result["pnp_tvec_m"] = tvec

                last_result = result

            cv2.imshow("aruco pnp detection", debug)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                print("Canceled by ESC")
                return None, None

            if key in (10, 13):
                if last_result is None:
                    print("ArUcoが検出できていない")
                    continue

                if "pnp_tvec_m" not in last_result:
                    print("PnPが失敗している")
                    continue

                target_m = last_result["pnp_tvec_m"]
                d_roll_rad = last_result["d_roll_rad"]

                u, v = last_result["center_uv"]

                print("")
                print("========== ARUCO TARGET ==========")
                print("POSITION SOURCE : PnP")
                print("Depth           : NOT USED")
                print(f"center pixel u={u:.2f}, v={v:.2f}")
                print(f"image angle = {np.degrees(last_result['image_angle_rad']):.2f} deg")
                print(f"d_roll      = {np.degrees(d_roll_rad):.2f} deg")
                print("target camera PnP [m]  =", target_m)
                print("target camera PnP [mm] =", target_m * 1000.0)
                print("==================================")
                print("")

                return d_roll_rad, target_m

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


# =========================================================
# reaching like box
# =========================================================

def reach_aruco_center_like_box_pnp(arm: XArm7, side: str = "right"):
    """
    PnP版:
      ArUco 4隅
      -> solvePnP
      -> target_m
      -> target_mm
      -> cam_mm_to_robot_mm()
      -> move_to_target_xyz_and_roll()
    """

    d_roll_rad, target_m = run_capture_and_aruco_center_pnp()

    if target_m is None:
        raise RuntimeError("ArUco PnP recognition canceled or failed")

    target_mm = 1000.0 * target_m

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
        ans = input("Move robot? [y/N]: ").strip()
        if ans.lower() != "y":
            print("移動キャンセル")
            return

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
    print("aruco PnP center reaching done")

    input("Enterで退避姿勢に戻る / Ctrl+Cで終了: ")

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

    node = rclpy.create_node("aruco_center_pnp_reaching_test")

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

        reach_aruco_center_like_box_pnp(
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