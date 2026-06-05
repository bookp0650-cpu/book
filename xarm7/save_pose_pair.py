#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# あなたのモジュール（同じフォルダに置く想定）
from control.ar_marker_pose import rs_color_K_dist, aruco_marker_pose_target2cam


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def json_dump(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xarm-host", type=str, default="192.168.2.197", help="xArm IP (e.g. 192.168.2.197)")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="xarm7/handeye_pairs",
        help="output directory"
    )
    ap.add_argument("--num-samples", type=int, default=40, help="number of samples to save")
    ap.add_argument("--marker-len-m", type=float, default=0.15, help="marker side length [m] (e.g. 0.040)")
    ap.add_argument("--marker-id", type=int, default=0, help="use specific marker id (optional)")
    ap.add_argument("--aruco-dict", type=str, default="DICT_4X4_1000", help="ArUco dict name")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--is-radian", action="store_true", help="xArm pose angles are radians (default: degrees)")
    ap.add_argument("--save-images", action="store_true", help="also save RGB images")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    if args.save_images:
        img_dir.mkdir(parents=True, exist_ok=True)

    # --- RealSense start ---
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipe.start(cfg)

    intr, K, dist = rs_color_K_dist(profile)

    # --- xArm connect ---
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(args.xarm_host)
    arm.connect()
    # 読み取りだけでも繋がればOKだが、念のため一般的な初期化
    try:
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)
    except Exception:
        pass

    session_id = now_stamp()
    json_path = out_dir / f"handeye_pairs_{session_id}.json"

    meta = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "marker_len_m": float(args.marker_len_m),
        "aruco_dict": args.aruco_dict,
        "marker_id_target": args.marker_id,
        "realsense": {
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "ppx": float(intr.ppx),
            "ppy": float(intr.ppy),
            "model": str(intr.model),
            "coeffs": [float(x) for x in intr.coeffs],
            "K": K.tolist(),
            "dist5": dist.tolist(),
        },
        "xarm": {
            "host": args.xarm_host,
            "is_radian": bool(args.is_radian),
            "pos_unit": "mm",
            "angle_unit": "rad" if args.is_radian else "deg",
            "pose_order": ["x", "y", "z", "roll", "pitch", "yaw"],
        },
        "note": "Each sample has (tcp_pose in base frame) and (marker pose target->camera: rvec,tvec).",
    }

    samples = []

    print("[INFO] --- Capture Hand-Eye pairs ---")
    print("[INFO] 's' : save one sample (when marker detected)")
    print("[INFO] 'q' : quit")
    print(f"[INFO] will save {args.num_samples} samples to: {json_path}")

    try:
        while True:
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            bgr = np.asanyarray(color.get_data())
            vis = bgr.copy()

            # marker pose (target->camera) 取得
            ret = aruco_marker_pose_target2cam(
                bgr=bgr,
                K=K,
                dist=dist,
                marker_len_m=args.marker_len_m,
                dict_name=args.aruco_dict,
                target_id=args.marker_id,
            )

            if ret is None:
                cv2.putText(vis, "Marker: not detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                mid, rvec, tvec = ret
                cv2.putText(vis, f"Marker: id={mid} t[m]={tvec[0]:.3f},{tvec[1]:.3f},{tvec[2]:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # 軸描画（見やすさ用）
                try:
                    cv2.drawFrameAxes(vis, K, dist, rvec.reshape(3, 1), tvec.reshape(3, 1), 0.05)
                except Exception:
                    pass

            cv2.putText(vis, f"Saved: {len(samples)}/{args.num_samples}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            scale = 1.0
            vis_big = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("handeye_capture (s=save, q=quit)", vis_big)
            key = cv2.waitKey(10) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                if ret is None:
                    print("[WARN] marker not detected -> not saved")
                    continue

                # xArm TCP pose（base基準）取得
                code, pose = arm.get_position(is_radian=args.is_radian)
                if code != 0 or pose is None:
                    print(f"[WARN] xArm get_position failed: code={code}, pose={pose} -> not saved")
                    continue

                mid, rvec, tvec = ret

                sid = now_stamp()
                img_path = None
                if args.save_images:
                    img_path = str((img_dir / f"rgb_{len(samples):02d}_{sid}.png").resolve())
                    cv2.imwrite(img_path, bgr)

                sample = {
                    "sample_id": sid,
                    "marker": {
                        "id": int(mid),
                        # OpenCV ArUcoの結果: target(marker) -> camera
                        "rvec_target2cam": [float(x) for x in rvec.reshape(3)],
                        "tvec_target2cam_m": [float(x) for x in tvec.reshape(3)],
                    },
                    "robot": {
                        # xArm SDKの get_position 戻り: [x,y,z,roll,pitch,yaw]
                        "tcp_pose_base": [float(x) for x in pose],
                    },
                    "rgb_path": img_path,
                }

                samples.append(sample)

                # 逐次JSON保存（途中で落ちてもデータ残る）
                json_dump(json_path, {"meta": meta, "samples": samples})

                print(f"[SAVE] {len(samples)}/{args.num_samples}  id={mid}  json={json_path.name}")

                if len(samples) >= args.num_samples:
                    print("[INFO] reached num_samples, done.")
                    break
                print("[DEBUG] json full path =", json_path.resolve())

    finally:
        try:
            if len(samples) > 0:
                json_dump(json_path, {"meta": meta, "samples": samples})
                print("[INFO] final json saved:", json_path.resolve())
        except Exception as e:
            print("[ERROR] final json save failed:", e)

        try:
            pipe.stop()
        except Exception:
            pass
        try:
            arm.disconnect()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
