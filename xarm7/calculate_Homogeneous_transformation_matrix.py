#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def T_from_Rt(Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm.astype(np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    Rm = T[:3, :3]
    tv = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = Rm.T
    Ti[:3, 3] = -Rm.T @ tv
    return Ti


def _Rx(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa, ca]], dtype=np.float64)


def _Ry(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0, sa],
                     [0, 1, 0],
                     [-sa, 0, ca]], dtype=np.float64)


def _Rz(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0],
                     [sa,  ca, 0],
                     [0,   0,  1]], dtype=np.float64)


def euler_to_R(order: str, angles_rad: np.ndarray) -> np.ndarray:
    """
    order: 'xyz' or 'zyx' etc (len=3)
    angles_rad: [a0,a1,a2] corresponds to axes in 'order'
    Returns: R = R_axis0(a0) @ R_axis1(a1) @ R_axis2(a2)
    """
    order = order.lower()
    if len(order) != 3:
        raise ValueError("order must be length 3, e.g. 'xyz' or 'zyx'")

    Rm = np.eye(3, dtype=np.float64)
    for ax, ang in zip(order, angles_rad.tolist()):
        if ax == "x":
            Rm = Rm @ _Rx(ang)
        elif ax == "y":
            Rm = Rm @ _Ry(ang)
        elif ax == "z":
            Rm = Rm @ _Rz(ang)
        else:
            raise ValueError(f"unsupported axis in order: {ax}")
    return Rm

def T_from_Rt(Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm.astype(np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", type=str, default="/home/book/pro_book/pro_hand_book_python/xarm7/handeye_pairs/handeye_pairs_20260604_202044_238.json", help="handeye_pairs_*.json")
    ap.add_argument("--out-json", type=str, default="", help="output json path (optional)")
    ap.add_argument("--euler-order", type=str, default="zyx",
                    help="robot euler order for [roll,pitch,yaw]. try xyz first; if bad try zyx")
    ap.add_argument("--method", type=str, default="DANIILIDIS",
                    choices=["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"],
                    help="OpenCV calibrateHandEye method")
    args = ap.parse_args()

    in_path = Path(args.in_json)
    data = json.loads(in_path.read_text(encoding="utf-8"))

    # output path
    if args.out_json:
        out_path = Path(args.out_json)
    else:
        out_path = in_path.with_name(f"handeye_T_tcp_cam_{now_stamp()}.json")

    # method mapping
    method_map = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    method = method_map[args.method]

    # angle unit
    angle_unit = data.get("meta", {}).get("xarm", {}).get("angle_unit", "deg")
    is_deg = (angle_unit.lower() == "deg")

    def to_rad(a: float) -> float:
        return np.deg2rad(a) if is_deg else float(a)

    # build lists
    R_gripper2base, t_gripper2base = [], []
    R_target2cam,   t_target2cam   = [], []

    for s in data["samples"]:
        # --- target->cam from ArUco ---
        rvec = np.array(s["marker"]["rvec_target2cam"], dtype=np.float64).reshape(3, 1)
        tvec = np.array(s["marker"]["tvec_target2cam_m"], dtype=np.float64).reshape(3, 1)
        R_ct, _ = cv2.Rodrigues(rvec)
        # T_cam2target = T_from_Rt(R_ct, tvec)
        # T_target2cam = inv_T(T_cam2target)
        # R_tc = T_target2cam[:3, :3]
        # tvec = T_target2cam[:3, 3].reshape(3, 1)
        R_target2cam.append(R_ct)
        t_target2cam.append(tvec)
        # # print("target->cam")
        # # print(R_tc)
        # # print(tvec)
        # T_cam2target = T_from_Rt(R_tc, tvec)
        # print("T_cam2target:")
        # print(T_cam2target)
        # T_target2cam = inv_T(T_cam2target)
        # print("T_target2cam:")
        # print(T_target2cam)

        # --- base->tcp from xArm ---
        x_mm, y_mm, z_mm, roll, pitch, yaw = s["robot"]["tcp_pose_base"] # x_mm, y_mm, z_mm roll, pitch, yaw
        t_bt_m = np.array([x_mm, y_mm, z_mm], dtype=np.float64) * 1e-3  # mm -> m

        # angles_rad = np.array([to_rad(roll), to_rad(pitch), to_rad(yaw)], dtype=np.float64)
        angles_rad = np.array([to_rad(yaw), to_rad(pitch), to_rad(roll)], dtype=np.float64)
        R_bt = euler_to_R(args.euler_order, angles_rad)

        T_base_tcp = T_from_Rt(R_bt, t_bt_m)
        R_gripper2base.append(T_base_tcp[:3, :3])
        t_gripper2base.append(T_base_tcp[:3, 3].reshape(3, 1))

        # OpenCV calibrateHandEye wants gripper->base
        # T_tcp_base = inv_T(T_base_tcp)
        # print("gripper->base")
        # print(T_tcp_base[:3, :3])
        # print(T_tcp_base[:3, 3].reshape(3, 1))
        # R_gripper2base.append(T_tcp_base[:3, :3])
        # t_gripper2base.append(T_tcp_base[:3, 3].reshape(3, 1))
    # print("R_gripper2base, t_gripper2base, R_target2cam, t_target2cam lengths =",R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
    # calibrate
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam,   t_target2cam,
        method=method
    )

    print(f"R_cam2gripper: {R_cam2gripper}")
    print(f"t_cam2gripper: {t_cam2gripper}")
    T_cam_tcp = T_from_Rt(R_cam2gripper, np.asarray(t_cam2gripper).reshape(3))
    T_tcp_cam = inv_T(T_cam_tcp)  # <-- これが欲しい T_tcp_camera逆になっとるやんけ！！
    print("T_tcp_cam:", T_tcp_cam)

    # save json
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_json": str(in_path.resolve()),
        "settings": {
            "method": args.method,
            "robot_euler_order": args.euler_order,
            "robot_angle_unit": angle_unit,
            "robot_pos_unit_in_input": "mm",
            "target_tvec_unit_in_input": "m",
        },
        "T_cam_tcp": T_cam_tcp.tolist(),
        "T_tcp_cam": T_tcp_cam.tolist(),
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # print("[OK] saved:", out_path)
    # print("T_tcp_camera (= T_tcp_cam):")
    # np.set_printoptions(precision=6, suppress=True)
    # print(np.array(T_tcp_cam))

    print(out_path)


if __name__ == "__main__":
    main()
