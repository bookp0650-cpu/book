#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

# /home/book/pro_book/pro_hand_book_python を import パスに追加
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os
import csv
import json
import time
import signal
import yaml
import rclpy
import numpy as np

from scipy.spatial.transform import Rotation as R
from rclpy.executors import MultiThreadedExecutor

from xarm7.control.xarm7 import XArm7


# =========================================================
# settings
# =========================================================

CONFIG_PATH = "Retrieval_integration.yaml"

SAVE_DIR = Path("/home/book/pro_book/pro_hand_book_python/xarm7/tcp_calib_logs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = SAVE_DIR / "tcp_pivot_samples.csv"
RESULT_JSON_PATH = SAVE_DIR / "tcp_pivot_result.json"

# 最低でも6姿勢以上，できれば10〜20姿勢
MIN_SAMPLES = 6

# xArmのRPY順序
# 既存コードで xyz を使っているなら xyz のままでOK
EULER_ORDER = "xyz"


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
# xArm helpers
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


def get_current_pose_rad(arm):
    """
    現在姿勢を [x_mm, y_mm, z_mm, roll_rad, pitch_rad, yaw_rad] で取得。

    注意:
    ここで取得する姿勢は，必ず「フランジ姿勢」または
    「TCPオフセットをゼロにした状態の姿勢」である必要がある。

    もし現在のTCP設定が入ったまま get_tcp_pose() を使うと，
    推定したいTCPを使って測ってしまうので意味が崩れる。
    """

    # 既存XArm7クラスに get_tcp_pose がある想定
    if hasattr(arm, "get_tcp_pose"):
        pose = arm.get_tcp_pose(is_radian=True)
        return np.array(pose, dtype=np.float64)

    # SDK本体を直接持っている場合
    sdk_arm = getattr(arm, "arm", None)
    if sdk_arm is None:
        sdk_arm = getattr(arm, "_arm", None)

    if sdk_arm is not None and hasattr(sdk_arm, "get_position"):
        code, pose = sdk_arm.get_position(is_radian=True)
        if code != 0:
            raise RuntimeError(f"get_position failed: code={code}")
        return np.array(pose, dtype=np.float64)

    raise RuntimeError("現在姿勢を取得できない。get_tcp_pose または SDK get_position を確認して。")


# =========================================================
# math
# =========================================================

def pose_to_R_t(pose):
    """
    pose:
        [x_mm, y_mm, z_mm, roll_rad, pitch_rad, yaw_rad]

    return:
        R_base_flange
        t_base_flange
    """
    x, y, z, roll, pitch, yaw = pose

    t = np.array([x, y, z], dtype=np.float64)
    R_mat = R.from_euler(
        EULER_ORDER,
        [roll, pitch, yaw],
        degrees=False
    ).as_matrix()

    return R_mat, t


def estimate_flange_to_tcp_translation(poses):
    """
    Pivot calibration.

    各姿勢でTCP先端が同じ固定点P_baseにあると仮定する。

        P_base = R_i @ p_flange_tcp + t_i

    未知:
        p_flange_tcp = [px, py, pz]
        P_base       = [X, Y, Z]

    線形最小二乗:
        [R_i  -I] [p_flange_tcp] = -t_i
                  [P_base      ]

    poses:
        list of [x_mm, y_mm, z_mm, roll_rad, pitch_rad, yaw_rad]

    return:
        p_flange_tcp_mm
        fixed_point_base_mm
        residuals_mm
        T_flange_tcp
    """

    if len(poses) < MIN_SAMPLES:
        raise RuntimeError(f"Need at least {MIN_SAMPLES} samples, got {len(poses)}")

    A_list = []
    b_list = []

    for pose in poses:
        R_i, t_i = pose_to_R_t(pose)

        A_i = np.hstack([R_i, -np.eye(3)])
        b_i = -t_i

        A_list.append(A_i)
        b_list.append(b_i)

    A = np.vstack(A_list)
    b = np.hstack(b_list)

    sol, residual_sum, rank, svals = np.linalg.lstsq(A, b, rcond=None)

    p_flange_tcp_mm = sol[0:3]
    fixed_point_base_mm = sol[3:6]

    residuals = []

    for pose in poses:
        R_i, t_i = pose_to_R_t(pose)

        pred_fixed = R_i @ p_flange_tcp_mm + t_i
        err = pred_fixed - fixed_point_base_mm

        residuals.append(err)

    residuals = np.array(residuals, dtype=np.float64)

    T_flange_tcp = np.eye(4, dtype=np.float64)
    T_flange_tcp[:3, :3] = np.eye(3)
    T_flange_tcp[:3, 3] = p_flange_tcp_mm

    return {
        "p_flange_tcp_mm": p_flange_tcp_mm,
        "fixed_point_base_mm": fixed_point_base_mm,
        "residuals_mm": residuals,
        "T_flange_tcp": T_flange_tcp,
        "rank": int(rank),
        "singular_values": svals,
    }


# =========================================================
# save / load
# =========================================================

def init_csv_if_needed(csv_path):
    if csv_path.exists():
        return

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "x_mm",
            "y_mm",
            "z_mm",
            "roll_rad",
            "pitch_rad",
            "yaw_rad",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
        ])


def append_pose_csv(csv_path, index, pose):
    x, y, z, roll, pitch, yaw = pose

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            index,
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
            np.degrees(roll),
            np.degrees(pitch),
            np.degrees(yaw),
        ])


def load_poses_from_csv(csv_path):
    poses = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            pose = [
                float(row["x_mm"]),
                float(row["y_mm"]),
                float(row["z_mm"]),
                float(row["roll_rad"]),
                float(row["pitch_rad"]),
                float(row["yaw_rad"]),
            ]
            poses.append(pose)

    return poses


def save_result_json(result, path):
    residuals = result["residuals_mm"]
    err_norm = np.linalg.norm(residuals, axis=1)

    data = {
        "description": "Pivot calibration result. Rotation part is identity because fixed-point pivot calibration estimates translation only.",
        "euler_order": EULER_ORDER,
        "p_flange_tcp_mm": result["p_flange_tcp_mm"].tolist(),
        "fixed_point_base_mm": result["fixed_point_base_mm"].tolist(),
        "T_flange_tcp": result["T_flange_tcp"].tolist(),
        "residuals_mm": residuals.tolist(),
        "residual_norm_mm": err_norm.tolist(),
        "mean_error_mm": float(np.mean(err_norm)),
        "max_error_mm": float(np.max(err_norm)),
        "std_error_mm": float(np.std(err_norm)),
        "rank": result["rank"],
        "singular_values": result["singular_values"].tolist(),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved result: {path}")


# =========================================================
# main calibration flow
# =========================================================

def print_pose(index, pose):
    x, y, z, roll, pitch, yaw = pose
    print("")
    print(f"========== SAMPLE {index} ==========")
    print(f"X={x:.3f} mm, Y={y:.3f} mm, Z={z:.3f} mm")
    print(f"roll ={roll:.6f} rad  ({np.degrees(roll):.3f} deg)")
    print(f"pitch={pitch:.6f} rad  ({np.degrees(pitch):.3f} deg)")
    print(f"yaw  ={yaw:.6f} rad  ({np.degrees(yaw):.3f} deg)")
    print("====================================")
    print("")


def collect_samples(arm):
    """
    手動でTCP先端を固定点に合わせてEnterで記録する。
    """

    init_csv_if_needed(CSV_PATH)

    poses = []

    print("")
    print("======================================================")
    print(" TCP Pivot Calibration")
    print("======================================================")
    print("やること:")
    print("1. TCP先端を固定点に合わせる")
    print("2. 姿勢だけ変える")
    print("3. 先端が同じ固定点に合ったら Enter")
    print("4. 10〜20姿勢くらい記録")
    print("")
    print("重要:")
    print("・取得姿勢はフランジ姿勢であること")
    print("・既存TCPオフセットを入れたまま測ると推定が崩れる")
    print("・可能ならTCP設定をゼロにして実行")
    print("======================================================")
    print("")
    print("Enter : 現在姿勢を記録")
    print("s     : 推定実行")
    print("q     : 終了")
    print("")

    index = 0

    while True:
        cmd = input("Command [Enter=sample / s=solve / q=quit]: ").strip().lower()

        if cmd == "q":
            print("quit")
            return poses

        if cmd == "s":
            if len(poses) < MIN_SAMPLES:
                print(f"まだ少ない。最低 {MIN_SAMPLES} 姿勢必要。現在 {len(poses)}")
                continue
            return poses

        # Enterなら記録
        pose = get_current_pose_rad(arm)

        index += 1
        poses.append(pose)

        print_pose(index, pose)
        append_pose_csv(CSV_PATH, index, pose)

        print(f"sample count = {len(poses)}")


def solve_and_print(poses):
    result = estimate_flange_to_tcp_translation(poses)

    p_tcp = result["p_flange_tcp_mm"]
    fixed = result["fixed_point_base_mm"]
    T = result["T_flange_tcp"]
    residuals = result["residuals_mm"]
    err_norm = np.linalg.norm(residuals, axis=1)

    print("")
    print("======================================================")
    print(" TCP PIVOT CALIBRATION RESULT")
    print("======================================================")
    print("[p_flange_tcp_mm]")
    print(f"x = {p_tcp[0]:.6f} mm")
    print(f"y = {p_tcp[1]:.6f} mm")
    print(f"z = {p_tcp[2]:.6f} mm")
    print("")
    print("[fixed_point_base_mm]")
    print(f"X = {fixed[0]:.6f} mm")
    print(f"Y = {fixed[1]:.6f} mm")
    print(f"Z = {fixed[2]:.6f} mm")
    print("")
    print("[T_flange_tcp]")
    np.set_printoptions(precision=6, suppress=True)
    print(T)
    print("")
    print("[residual error]")
    for i, e in enumerate(err_norm, start=1):
        print(f"sample {i:02d}: {e:.6f} mm   residual={residuals[i-1]}")
    print("")
    print(f"mean error = {np.mean(err_norm):.6f} mm")
    print(f"max  error = {np.max(err_norm):.6f} mm")
    print(f"std  error = {np.std(err_norm):.6f} mm")
    print("======================================================")
    print("")

    save_result_json(result, RESULT_JSON_PATH)

    return result


# =========================================================
# main
# =========================================================

def main():
    config = load_config(CONFIG_PATH)

    rclpy.init()

    node = rclpy.create_node("tcp_pivot_calibration_xarm7")

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
        recover_xarm_if_possible(arm)

        poses = collect_samples(arm)

        if len(poses) >= MIN_SAMPLES:
            solve_and_print(poses)
        else:
            print("samples are not enough. no solve.")

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