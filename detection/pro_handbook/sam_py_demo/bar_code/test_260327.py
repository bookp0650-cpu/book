#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code_1_pic_ros2.py

フロー（統合コード想定）:
- /navigation_goal == True を受けて処理開始
- 連続フレームで「バーコードらしさ（BBox）」を検出し続ける
- 画像中心からのオフセットを /barcode_offset_u_px に publish（デバッグ/監視用）
- 画像中心から ±align_thresh_px 以内に入るまで /cmd_vel を publish（AMRを中央寄せ：並進のみ）
- しきい値内が stable_frames 連続したら /cmd_vel を止める（必要なら1回だけゼロを送る）
- 停止後に post_stop_settle_sec だけ待つ
- 再撮影2回＝合計 decode_attempts 回（デフォルト3回）
  各回：frame取得 → bbox再検出（取れなければlast_bbox）→ ROI → 既存の barcode_detect_unified でdecode
- デコードできたら /error_x に "<formatted_code>/<X_m>" を publish
"""

import time
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, String, Float32
from geometry_msgs.msg import Twist

from pyzbar.pyzbar import decode  # ★従来通り

class WallYawWatcher:
    def __init__(self, node: Node, topic_name: str = "/wall_yaw_deg"):
        self._node = node
        self._yaw_deg: Optional[float] = None
        self._sub = node.create_subscription(Float32, topic_name, self._callback, 10)

    def _callback(self, msg: Float32):
        self._yaw_deg = float(msg.data)

    def get_yaw_deg(self) -> Optional[float]:
        return self._yaw_deg

    def destroy(self):
        try:
            self._node.destroy_subscription(self._sub)
        except Exception:
            pass

class BoolPulseWatcher:
    """msg.data==True を受け取った回数をカウントして、1回ずつ消費できる watcher"""
    def __init__(self, node: Node, topic_name: str):
        self._node = node
        self._pending = 0
        self._sub = node.create_subscription(Bool, topic_name, self._cb, 10)

    def _cb(self, msg: Bool):
        if msg.data:
            self._pending += 1

    def consume(self) -> bool:
        if self._pending > 0:
            self._pending -= 1
            return True
        return False

    def destroy(self):
        try:
            self._node.destroy_subscription(self._sub)
        except Exception:
            pass


class BoolLatchWatcher:
    """msg.data==True が一度でも来たら True のまま保持"""
    def __init__(self, node: Node, topic_name: str):
        self._node = node
        self._value = False
        self._sub = node.create_subscription(Bool, topic_name, self._cb, 10)

    def _cb(self, msg: Bool):
        if msg.data:
            self._value = True

    def is_true(self) -> bool:
        return self._value

    def destroy(self):
        try:
            self._node.destroy_subscription(self._sub)
        except Exception:
            pass

from detection.pro_handbook.sam_py_demo.bar_code.web_camera_capture import (
    capture_one_depstech,
)


# =========================================================
# /navigation_goal watcher
# =========================================================
class NavigationGoalWatcher:
    def __init__(self, node: Node, topic_name: str = "/navigation_goal"):
        self._node = node
        self._received_true = False
        self._sub = node.create_subscription(Bool, topic_name, self._callback, 10)

    def _callback(self, msg: Bool):
        if msg.data:
            self._received_true = True
            self._node.get_logger().info("Received navigation_goal = True")

    def wait_until_true(self, executor, timeout_sec: Optional[float] = None) -> bool:
        start = time.time()
        while rclpy.ok() and not self._received_true:
            executor.spin_once(timeout_sec=0.1)
            if timeout_sec is not None and (time.time() - start) > timeout_sec:
                return False
        return self._received_true

    def destroy(self):
        try:
            self._node.destroy_subscription(self._sub)
        except Exception:
            pass


# =========================================================
# /wall_distance watcher（統合コードで import される前提）
# =========================================================
class WallDistanceWatcher:
    def __init__(self, node: Node, topic_name: str = "/wall_distance"):
        self._node = node
        self._distance: Optional[float] = None
        self._sub = node.create_subscription(Float32, topic_name, self._callback, 10)

    def _callback(self, msg: Float32):
        self._distance = float(msg.data)

    def get_distance(self) -> Optional[float]:
        return self._distance

    def destroy(self):
        try:
            self._node.destroy_subscription(self._sub)
        except Exception:
            pass


# =========================================================
# Utils
# =========================================================
def save_image(img, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(save_path), img)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {save_path}")


def format_barcode_code(raw_code: str) -> str:
    """
    例: 030104070500 -> 1-4-7-5
    """
    try:
        parts = [raw_code[i:i+2] for i in range(0, len(raw_code), 2)]
        if len(parts) < 5:
            return raw_code
        selected = parts[1:5]
        selected_int = [str(int(p)) for p in selected]
        return "-".join(selected_int)
    except Exception:
        return raw_code


def draw_bbox(img, data, color=(0, 255, 0), thickness: int = 2, label: Optional[str] = None):
    out = img.copy()

    if isinstance(data, (tuple, list)) and len(data) == 4:
        x, y, w, h = data
    elif isinstance(data, dict):
        if all(k in data for k in ("x", "y", "w", "h")):
            x, y, w, h = data["x"], data["y"], data["w"], data["h"]
        elif all(k in data for k in ("left", "top", "width", "height")):
            x, y, w, h = data["left"], data["top"], data["width"], data["height"]
        else:
            raise ValueError("dict形式の data は {x,y,w,h} か {left,top,width,height} にしてください")
    else:
        raise ValueError("data は (x,y,w,h) か dict を渡してください")

    x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
    H, W = out.shape[:2]

    x1 = max(0, min(W - 1, x))
    y1 = max(0, min(H - 1, y))
    x2 = max(0, min(W - 1, x + w))
    y2 = max(0, min(H - 1, y + h))

    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.5
        t = max(1, thickness)
        (tw, th), _ = cv2.getTextSize(label, font, fs, t)

        by1 = y1 - th - 6
        if by1 < 0:
            by1 = y1 + 2
        bx1 = x1
        bx2 = min(W - 1, x1 + tw + 6)
        by2 = min(H - 1, by1 + th + 6)

        cv2.rectangle(out, (bx1, by1), (bx2, by2), color, -1)
        cv2.putText(out, label, (bx1 + 3, by2 - 3), font, fs, (0, 0, 0), t, cv2.LINE_AA)

    return out


# =========================================================
# Barcode decode helpers（従来コードを維持）
# =========================================================
def _select_barcode(decode_data: List[Any], frame, target_code: Optional[str] = None):
    if len(decode_data) == 0:
        return None

    if target_code is not None and target_code != "":
        for d in decode_data:
            code_str = d.data.decode("utf-8", errors="ignore")
            if code_str == str(target_code):
                return d

    H, W = frame.shape[:2]
    u_c = W / 2.0
    v_c = H / 2.0

    best_barcode = None
    min_dist2 = float("inf")

    for d in decode_data:
        rect = d.rect
        u_b = rect.left + rect.width / 2.0
        v_b = rect.top + rect.height / 2.0
        dist2 = (u_b - u_c) ** 2 + (v_b - v_c) ** 2
        if dist2 < min_dist2:
            min_dist2 = dist2
            best_barcode = d

    return best_barcode


def preprocess_for_barcode(gray: "cv2.Mat") -> "cv2.Mat":
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(blur)

    binary = cv2.adaptiveThreshold(
        clahe_img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        25,
        10,
    )

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return binary


def barcode_detect_unified(
    img,
    barcode_number: Optional[str] = None,
    save_dir: Optional[Path] = None,
) -> Tuple[bool, Optional[Any], List[Any]]:
    if img is None:
        return False, None, []

    # カラー → グレー
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 前処理
    pre = preprocess_for_barcode(gray)

    # 保存（指定があれば）
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_image(gray, save_dir / "bookshelf_barcode_gray_proc.png")
        save_image(pre, save_dir / "bookshelf_barcode_preproc.png")

    candidates = [gray, pre]

    last_decoded: List[Any] = []
    selected = None

    for cand in candidates:
        decode_data = decode(cand)
        last_decoded = decode_data
        if len(decode_data) == 0:
            continue

        selected = _select_barcode(decode_data, img, target_code=barcode_number)
        if selected is not None:
            break

    if selected is None:
        return False, None, last_decoded

    return True, selected, last_decoded


# =========================================================
# BBox detection (barcode-likeness)
# =========================================================
def _open_depstech_cap(width=3840, height=2160, fps=30, fourcc="MJPG"):
    dev = None
    try:
        from detection.pro_handbook.sam_py_demo.bar_code.web_camera_capture import (
            find_first_openable_video_device,
        )
        dev = find_first_openable_video_device("Depstech")
    except Exception:
        dev = "/dev/video10"

    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {dev}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap, dev


def _detect_barcode_bbox_opencv(frame_bgr: np.ndarray):
    if not (hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector")):
        return None

    det = cv2.barcode.BarcodeDetector()
    ok, corners = det.detect(frame_bgr)  # corners: Nx4x2
    if (not ok) or corners is None:
        return None

    H, W = frame_bgr.shape[:2]
    cx = W / 2.0
    cy = H / 2.0

    best = None
    best_d2 = 1e18
    for c in corners:
        c = np.array(c, dtype=np.float32).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(c.astype(np.int32))
        bx = x + w / 2.0
        by = y + h / 2.0
        d2 = (bx - cx) ** 2 + (by - cy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = (float(x), float(y), float(x + w), float(y + h))
    return best


def _detect_barcode_bbox_fallback(frame_bgr: np.ndarray):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gx = np.abs(gx)
    gx = cv2.normalize(gx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    gx = cv2.GaussianBlur(gx, (9, 9), 0)
    _, bw = cv2.threshold(gx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 7)),
        iterations=1,
    )
    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    H, W = gray.shape[:2]
    cx = W / 2.0
    cy = H / 2.0

    best = None
    best_score = -1e18

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < (W * H) * 0.002:
            continue
        ar = w / max(1, h)
        if ar < 1.2:
            continue

        bx = x + w / 2.0
        by = y + h / 2.0
        dist = (bx - cx) ** 2 + (by - cy) ** 2

        score = area - 0.5 * dist
        if score > best_score:
            best_score = score
            best = (float(x), float(y), float(x + w), float(y + h))

    return best


def detect_barcode_bbox(frame_bgr: np.ndarray):
    b = _detect_barcode_bbox_opencv(frame_bgr)
    if b is not None:
        return b
    return _detect_barcode_bbox_fallback(frame_bgr)


def crop_bbox(frame_bgr: np.ndarray, bbox, margin: float = 0.15):
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    mx = w * margin
    my = h * margin
    X1 = int(max(0, np.floor(x1 - mx)))
    Y1 = int(max(0, np.floor(y1 - my)))
    X2 = int(min(W, np.ceil(x2 + mx)))
    Y2 = int(min(H, np.ceil(y2 + my)))
    return frame_bgr[Y1:Y2, X1:X2]


# =========================================================
# Compute x error
# =========================================================
def compute_X_m_from_bbox_center(frame_bgr: np.ndarray, bbox, fx_px: float, depth_m: float):
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    u_b = (x1 + x2) / 2.0
    v_b = (y1 + y2) / 2.0
    offset_u_px = float(u_b - (W / 2.0))
    X_m = float((offset_u_px / fx_px) * depth_m)
    return offset_u_px, u_b, v_b, X_m


# =========================================================
# Main function
# =========================================================
# def capture_barcode_and_x_offset(
#     node: Node,
#     executor,
#     shot_dir: Path,
#     barcode_number: Optional[str] = None,
#     fx_px: float = 2500.0,
#     depth_m: float = 0.40,

#     # alignment
#     align_before_decode: bool = True,
#     align_thresh_px: float = 40.0,
#     stable_frames: int = 5,
#     align_timeout_sec: float = 25.0,

#     # publish debug topics
#     publish_align_topics: bool = True,
#     topic_offset_u: str = "/barcode_offset_u_px",
#     topic_stop_align: str = "/amr_stop_align",
#     topic_bbox_debug: str = "/barcode_bbox_xyxy",

#     # cmd_vel（並進のみ）
#     cmd_vel_topic: str = "/cmd_vel",
#     v_x: float = 0.03,
#     cmd_sign_x: float = 1.0,
#     send_stop_once: bool = True,

#     # decode ROI
#     decode_margin: float = 0.15,

#     # ---- NEW: stop後の待ち + 3回撮影decode ----
#     post_stop_settle_sec: float = 0.7,
#     decode_attempts: int = 3,              # 3回（=再撮影2回）
#     attempt_interval_sec: float = 0.4,
#     post_stop_warmup_frames: int = 8,

#     # ---- Compatibility: ignore old keywords to avoid crash ----
#     kp_w: Optional[float] = None,
#     max_w: Optional[float] = None,
#     cmd_sign: Optional[float] = None,
#     **kwargs,
# ) -> Tuple[bool, Optional[str], Optional[Dict[str, float]]]:

#     shot_dir = Path(shot_dir)
#     shot_dir.mkdir(parents=True, exist_ok=True)

#     # 1) wait /navigation_goal
#     node.get_logger().info("[bookshelf barcode] Waiting /navigation_goal == True ...")
#     watcher = NavigationGoalWatcher(node)
#     ok = watcher.wait_until_true(executor, timeout_sec=None)
#     watcher.destroy()
#     if not ok:
#         node.get_logger().warn("[bookshelf barcode] /navigation_goal wait aborted → skip")
#         return False, None, None

#     # 2) publishers
#     if publish_align_topics:
#         if not hasattr(capture_barcode_and_x_offset, "_pub_offset_u"):
#             capture_barcode_and_x_offset._pub_offset_u = node.create_publisher(Float32, topic_offset_u, 10)
#         if not hasattr(capture_barcode_and_x_offset, "_pub_stop_align"):
#             capture_barcode_and_x_offset._pub_stop_align = node.create_publisher(Bool, topic_stop_align, 10)
#         if not hasattr(capture_barcode_and_x_offset, "_pub_bbox_dbg"):
#             capture_barcode_and_x_offset._pub_bbox_dbg = node.create_publisher(String, topic_bbox_debug, 10)

#         pub_off = capture_barcode_and_x_offset._pub_offset_u
#         pub_stop = capture_barcode_and_x_offset._pub_stop_align
#         pub_bbox = capture_barcode_and_x_offset._pub_bbox_dbg
#     else:
#         pub_off = pub_stop = pub_bbox = None

#     if not hasattr(capture_barcode_and_x_offset, "_pub_cmd_vel"):
#         capture_barcode_and_x_offset._pub_cmd_vel = node.create_publisher(Twist, cmd_vel_topic, 10)
#     pub_cmd = capture_barcode_and_x_offset._pub_cmd_vel

#     def publish_cmd(vx: float):
#         msg = Twist()
#         msg.linear.x = float(vx)
#         msg.angular.z = 0.0
#         pub_cmd.publish(msg)

#     # /error_x publisher
#     if not hasattr(capture_barcode_and_x_offset, "_error_x_pub"):
#         capture_barcode_and_x_offset._error_x_pub = node.create_publisher(String, "/error_x", 10)
#     pub_err = capture_barcode_and_x_offset._error_x_pub

#     # 3) alignment loop（capは decode完了まで閉じない）
#     last_bbox = None
#     last_frame = None

#     cap = None
#     dev = None
#     try:
#         if align_before_decode:
#             cap, dev = _open_depstech_cap(width=3840, height=2160, fps=30, fourcc="MJPG")
#             node.get_logger().info(f"[bookshelf barcode] stream opened: {dev}")

#             for _ in range(10):
#                 cap.read()

#             t0 = time.time()
#             stable = 0
#             stop_sent = False

#             while rclpy.ok() and (time.time() - t0) < align_timeout_sec:
#                 try:
#                     executor.spin_once(timeout_sec=0.0)
#                 except Exception:
#                     pass

#                 ok2, frame = cap.read()
#                 if not ok2 or frame is None:
#                     stable = 0
#                     continue

#                 last_frame = frame
#                 H, W = frame.shape[:2]

#                 bbox = detect_barcode_bbox(frame)

#                 if bbox is None:
#                     stable = 0
#                     stop_sent = False
#                     if send_stop_once:
#                         publish_cmd(0.0)
#                     if pub_stop:
#                         pub_stop.publish(Bool(data=False))
#                     if pub_off:
#                         pub_off.publish(Float32(data=0.0))
#                     if pub_bbox:
#                         pub_bbox.publish(String(data=""))
#                     continue

#                 last_bbox = bbox
#                 x1, y1, x2, y2 = bbox
#                 bx = (x1 + x2) / 2.0
#                 offset_u = float(bx - (W / 2.0))

#                 if pub_off:
#                     pub_off.publish(Float32(data=offset_u))
#                 if pub_bbox:
#                     pub_bbox.publish(String(data=f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}"))

#                 if abs(offset_u) <= align_thresh_px:
#                     stable += 1
#                 else:
#                     stable = 0
#                     stop_sent = False

#                 centered = (stable >= stable_frames)
#                 if pub_stop:
#                     pub_stop.publish(Bool(data=centered))

#                 if not centered:
#                     direction = 1.0 if offset_u > 0 else (-1.0 if offset_u < 0 else 0.0)
#                     vx_cmd = cmd_sign_x * v_x * direction
#                     publish_cmd(vx_cmd)
#                 else:
#                     if send_stop_once and not stop_sent:
#                         publish_cmd(0.0)
#                         stop_sent = True
#                     node.get_logger().info("[bookshelf barcode] centered -> stop cmd_vel & start decode")
#                     break

#             # debug
#             if last_frame is not None:
#                 save_image(last_frame, shot_dir / "bookshelf_barcode_align_last.png")

#             if last_bbox is None:
#                 node.get_logger().warn("[bookshelf barcode] alignment failed: bbox not found")
#                 return False, None, None
#         else:
#             # alignmentしない場合は 1shot
#             last_frame = capture_one_depstech(shot_dir / "bookshelf_barcode_capture_selfloc.png")
#             last_bbox = detect_barcode_bbox(last_frame)

#         # 4) STOP後の settle 待ち
#         t_settle0 = time.time()
#         while rclpy.ok() and (time.time() - t_settle0) < post_stop_settle_sec:
#             try:
#                 executor.spin_once(timeout_sec=0.05)
#             except Exception:
#                 pass

#         # 5) decode retry（合計 decode_attempts 回）
#         decoded_code_str = None
#         decoded_frame = None
#         decoded_bbox = None
#         decoded_selected = None

#         # capが無い場合でも動くように（最後の保険）
#         if cap is None:
#             cap, dev = _open_depstech_cap(width=3840, height=2160, fps=30, fourcc="MJPG")
#             for _ in range(10):
#                 cap.read()

#         for i in range(max(1, int(decode_attempts))):
#             # 停止直後の微振動を避ける：捨てフレーム
#             for _ in range(max(0, int(post_stop_warmup_frames))):
#                 cap.read()

#             ok3, frame3 = cap.read()
#             if not ok3 or frame3 is None:
#                 node.get_logger().warn(f"[decode try {i+1}] capture failed")
#             else:
#                 save_image(frame3, shot_dir / f"decode_try_{i+1}.png")

#                 bbox3 = detect_barcode_bbox(frame3)
#                 if bbox3 is None:
#                     bbox3 = last_bbox  # 取れなければ最後のbboxを使う

#                 if bbox3 is not None:
#                     roi3 = crop_bbox(frame3, bbox3, margin=decode_margin)
#                 else:
#                     roi3 = frame3

#                 save_image(roi3, shot_dir / f"decode_try_{i+1}_roi.png")

#                 try_dir = shot_dir / f"try_{i+1}"
#                 detected3, selected3, _ = barcode_detect_unified(
#                     roi3,
#                     barcode_number,
#                     save_dir=try_dir,
#                 )

#                 if detected3 and selected3 is not None:
#                     decoded_code_str = selected3.data.decode("utf-8", errors="ignore")
#                     decoded_selected = selected3
#                     decoded_frame = frame3
#                     decoded_bbox = bbox3
#                     node.get_logger().info(f"[decode try {i+1}] SUCCESS: {decoded_code_str}")
#                     break
#                 else:
#                     node.get_logger().warn(f"[decode try {i+1}] failed")

#             # 次試行まで少し待つ
#             t_wait0 = time.time()
#             while rclpy.ok() and (time.time() - t_wait0) < attempt_interval_sec:
#                 try:
#                     executor.spin_once(timeout_sec=0.05)
#                 except Exception:
#                     pass

#         if decoded_code_str is None:
#             node.get_logger().warn("[capture_barcode_and_x_offset] decode failed after retries.")
#             return False, None, None

#         # 成功した回で以降を計算
#         frame = decoded_frame
#         bbox = decoded_bbox
#         code_str = decoded_code_str
#         formatted_code = format_barcode_code(code_str)

#         if bbox is None:
#             node.get_logger().warn("[capture_barcode_and_x_offset] bbox missing at success (unexpected)")
#             return False, None, None

#         offset_u_px, u_b, v_b, X_m = compute_X_m_from_bbox_center(frame, bbox, fx_px, depth_m)
#         H, W = frame.shape[:2]

#         info = {
#             "image_width": int(W),
#             "image_height": int(H),
#             "u_center": float(u_b),
#             "v_center": float(v_b),
#             "offset_u_px": float(offset_u_px),
#             "X_m": float(X_m),
#         }

#         # bbox visualize
#         x1, y1, x2, y2 = bbox
#         vis = draw_bbox(frame, (x1, y1, x2 - x1, y2 - y1), label=code_str)
#         save_image(vis, shot_dir / "bookshelf_barcode_capture_selfloc_bbox.png")

#         # /error_x publish
#         msg = String()
#         msg.data = f"{formatted_code}/{X_m:.6f}"
#         pub_err.publish(msg)
#         node.get_logger().info(f"[error_x published] {msg.data}")

#         return True, code_str, info

#     finally:
#         try:
#             if cap is not None:
#                 cap.release()
#         except Exception:
#             pass

def capture_barcode_and_x_offset(
    node: Node,
    executor,
    shot_dir: Path,
    barcode_number: Optional[str] = None,
    fx_px: float = 2500.0,
    depth_m: float = 0.40,

    # alignment
    align_before_decode: bool = True,
    align_thresh_px: float = 1000.0,
    stable_frames: int = 5,
    align_timeout_sec: float = 25.0,

    # publish debug topics
    publish_align_topics: bool = True,
    topic_offset_u: str = "/barcode_offset_u_px",
    topic_stop_align: str = "/amr_stop_align",
    topic_bbox_debug: str = "/barcode_bbox_xyxy",

    # cmd_vel（並進 + yaw補正）
    cmd_vel_topic: str = "/cmd_vel",
    v_x: float = 0.03,
    cmd_sign_x: float = 1.0,
    send_stop_once: bool = True,

    # decode ROI
    decode_margin: float = 0.15,

    # stop後の待ち + 複数回撮影decode
    post_stop_settle_sec: float = 0.7,
    decode_attempts: int = 10,
    attempt_interval_sec: float = 0.4,
    post_stop_warmup_frames: int = 8,

    # /navigation_goal 待ちを関数内でやるか
    wait_navigation_goal: bool = True,
    navigation_goal_topic: str = "/navigation_goal",
    navigation_goal_timeout_sec: Optional[float] = None,

    # /error_x topic
    error_x_topic: str = "/error_x",

    # --- yaw align by /wall_yaw_deg ---
    use_wall_yaw: bool = True,
    wall_yaw_topic: str = "/wall_yaw_deg",
    yaw_target_abs_deg: float = 90.0,
    yaw_tol_deg: float = 3.0,
    yaw_stable_frames: int = 5,

    # P制御（rad/s per deg）
    yaw_kp: float = 0.00,
    yaw_max_omega: float = 0.0,

    # Compatibility
    kp_w: Optional[float] = None,
    max_w: Optional[float] = None,
    cmd_sign: Optional[float] = None,
    **kwargs,
) -> Tuple[bool, Optional[str], Optional[Dict[str, float]]]:

    shot_dir = Path(shot_dir)
    shot_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # publisher cache helper（topicごとに保持）
    # -------------------------------------------------
    def _get_cached_pub(attr_name: str, topic: str, msg_type, qos_depth: int = 10):
        cache = getattr(capture_barcode_and_x_offset, attr_name, None)
        if cache is None:
            cache = {}
            setattr(capture_barcode_and_x_offset, attr_name, cache)

        pub = cache.get(topic, None)
        if pub is None:
            pub = node.create_publisher(msg_type, topic, qos_depth)
            cache[topic] = pub
        return pub

    # -------------------------------------------------
    # 1) wait /navigation_goal（必要なときだけ）
    # -------------------------------------------------
    if wait_navigation_goal:
        node.get_logger().info(f"[bookshelf barcode] Waiting {navigation_goal_topic} == True ...")
        watcher = NavigationGoalWatcher(node, topic_name=navigation_goal_topic)
        ok = watcher.wait_until_true(executor, timeout_sec=navigation_goal_timeout_sec)
        watcher.destroy()
        if not ok:
            node.get_logger().warn("[bookshelf barcode] navigation_goal wait aborted → skip")
            return False, None, None

    # -------------------------------------------------
    # 2) publishers
    # -------------------------------------------------
    if publish_align_topics:
        pub_off  = _get_cached_pub("_pub_cache_offset_u", topic_offset_u, Float32, 10)
        pub_stop = _get_cached_pub("_pub_cache_stop_align", topic_stop_align, Bool, 10)
        pub_bbox = _get_cached_pub("_pub_cache_bbox_dbg", topic_bbox_debug, String, 10)
    else:
        pub_off = pub_stop = pub_bbox = None

    pub_cmd = _get_cached_pub("_pub_cache_cmd_vel", cmd_vel_topic, Twist, 10)
    pub_err = _get_cached_pub("_pub_cache_error_x", error_x_topic, String, 10)

    yaw_watcher = WallYawWatcher(node, topic_name=wall_yaw_topic) if use_wall_yaw else None
    yaw_stable = 0

    # 関数の先頭付近（yaw_watcher 作った後）に追加してOK
    if not hasattr(capture_barcode_and_x_offset, "_yaw_sign_hold"):
        capture_barcode_and_x_offset._yaw_sign_hold = 1.0  # 初期は +側へ寄せる

    yaw_sign_deadband_deg = 5.0  # 0付近で符号反転しないための帯域（必要に応じて調整）

    def _compute_omega_from_wall_yaw() -> Tuple[float, bool]:
        """
        /wall_yaw_deg を使って |yaw| を 90±tol に寄せる（符号はラッチ）
        仕様：左回転で yaw が正に増える / 右回転で負に増える
        """
        if yaw_watcher is None:
            return 0.0, True

        yaw_deg = yaw_watcher.get_yaw_deg()
        if yaw_deg is None:
            return 0.0, False

        # --- sign latch（0付近では符号を保持）---
        if abs(yaw_deg) >= yaw_sign_deadband_deg:
            capture_barcode_and_x_offset._yaw_sign_hold = 1.0 if yaw_deg >= 0.0 else -1.0
        yaw_sign = capture_barcode_and_x_offset._yaw_sign_hold

        # 目標は ±90（符号は保持した yaw_sign を採用）
        target_yaw = yaw_sign * yaw_target_abs_deg

        # 目標との差（deg）
        err_deg = target_yaw - yaw_deg
        yaw_ok_now = (abs(err_deg) <= yaw_tol_deg)
        if yaw_ok_now:
            return 0.0, True

        omega = 0.0  # yaw_kp は「rad/s per deg」のつもりでOK
        omega = 0.0        
        return omega, False

    def publish_cmd(vx: float):
        omega, _ = _compute_omega_from_wall_yaw()
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(omega)
        pub_cmd.publish(msg)

    # -------------------------------------------------
    # 3) alignment loop（capは decode完了まで閉じない）
    # -------------------------------------------------
    last_bbox = None
    last_frame = None

    cap = None
    dev = None
    try:
        if align_before_decode:
            cap, dev = _open_depstech_cap(width=3840, height=2160, fps=30, fourcc="MJPG")
            node.get_logger().info(f"[bookshelf barcode] stream opened: {dev}")

            for _ in range(10):
                cap.read()

            t0 = time.time()
            stable = 0
            stop_sent = False

            while rclpy.ok() and (time.time() - t0) < align_timeout_sec:
                try:
                    executor.spin_once(timeout_sec=0.0)
                except Exception:
                    pass

                ok2, frame = cap.read()
                if not ok2 or frame is None:
                    stable = 0
                    yaw_stable = 0
                    continue

                last_frame = frame
                H, W = frame.shape[:2]

                # --- yaw stable update ---
                _, yaw_ok_now = _compute_omega_from_wall_yaw()
                if yaw_ok_now:
                    yaw_stable += 1
                else:
                    yaw_stable = 0
                yaw_centered = (yaw_stable >= yaw_stable_frames)

                bbox = detect_barcode_bbox(frame)

                if bbox is None:
                    # バーコードが見えない → 並進は止める（yawだけは合わせたいので publish_cmd(0)）
                    stable = 0
                    stop_sent = False

                    publish_cmd(0.0)

                    if pub_stop:
                        pub_stop.publish(Bool(data=False))
                    if pub_off:
                        pub_off.publish(Float32(data=0.0))
                    if pub_bbox:
                        pub_bbox.publish(String(data=""))
                    continue

                last_bbox = bbox
                x1, y1, x2, y2 = bbox
                bx = (x1 + x2) / 2.0
                offset_u = float(bx - (W / 2.0))

                if pub_off:
                    pub_off.publish(Float32(data=offset_u))
                if pub_bbox:
                    pub_bbox.publish(String(data=f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}"))

                if abs(offset_u) <= align_thresh_px:
                    stable += 1
                else:
                    stable = 0
                    stop_sent = False

                centered = (stable >= stable_frames)

                # ★ stop_align は「バーコード中心 AND yaw OK」のときだけ True
                if pub_stop:
                    pub_stop.publish(Bool(data=(centered and yaw_centered)))

                if centered and yaw_centered:
                    if send_stop_once and not stop_sent:
                        publish_cmd(0.0)  # yaw_okなので omega=0 になる
                        stop_sent = True
                    node.get_logger().info("[bookshelf barcode] centered + yaw_ok -> stop & start decode")
                    break

                # バーコード中心はOKだが yaw がまだ → 回転だけ（vx=0）
                if centered and (not yaw_centered):
                    publish_cmd(0.0)
                    continue

                # まだ中心でない → 並進（yaw は publish_cmd 内で自動付与）
                direction = 1.0 if offset_u > 0 else (-1.0 if offset_u < 0 else 0.0)
                vx_cmd = cmd_sign_x * v_x * direction
                publish_cmd(vx_cmd)

            if last_frame is not None:
                save_image(last_frame, shot_dir / "bookshelf_barcode_align_last.png")

            if last_bbox is None:
                node.get_logger().warn("[bookshelf barcode] alignment failed: bbox not found")
                return False, None, None

        else:
            last_frame = capture_one_depstech(shot_dir / "bookshelf_barcode_capture_selfloc.png")
            last_bbox = detect_barcode_bbox(last_frame)

        # -------------------------------------------------
        # 4) STOP後の settle 待ち
        # -------------------------------------------------
        t_settle0 = time.time()
        while rclpy.ok() and (time.time() - t_settle0) < post_stop_settle_sec:
            try:
                executor.spin_once(timeout_sec=0.05)
            except Exception:
                pass

        # -------------------------------------------------
        # 5) decode retry
        # -------------------------------------------------
        decoded_code_str = None
        decoded_frame = None
        decoded_bbox = None

        if cap is None:
            cap, dev = _open_depstech_cap(width=3840, height=2160, fps=30, fourcc="MJPG")
            for _ in range(10):
                cap.read()

        for i in range(max(1, int(decode_attempts))):
            for _ in range(max(0, int(post_stop_warmup_frames))):
                cap.read()

            ok3, frame3 = cap.read()
            if not ok3 or frame3 is None:
                node.get_logger().warn(f"[decode try {i+1}] capture failed")
            else:
                save_image(frame3, shot_dir / f"decode_try_{i+1}.png")

                bbox3 = detect_barcode_bbox(frame3)
                if bbox3 is None:
                    bbox3 = last_bbox

                roi3 = crop_bbox(frame3, bbox3, margin=decode_margin) if bbox3 is not None else frame3
                save_image(roi3, shot_dir / f"decode_try_{i+1}_roi.png")

                try_dir = shot_dir / f"try_{i+1}"
                detected3, selected3, _ = barcode_detect_unified(
                    roi3,
                    barcode_number,
                    save_dir=try_dir,
                )

                if detected3 and selected3 is not None:
                    decoded_code_str = selected3.data.decode("utf-8", errors="ignore")
                    decoded_frame = frame3
                    decoded_bbox = bbox3
                    node.get_logger().info(f"[decode try {i+1}] SUCCESS: {decoded_code_str}")
                    break
                else:
                    node.get_logger().warn(f"[decode try {i+1}] failed")

            t_wait0 = time.time()
            while rclpy.ok() and (time.time() - t_wait0) < attempt_interval_sec:
                try:
                    executor.spin_once(timeout_sec=0.05)
                except Exception:
                    pass

        if decoded_code_str is None:
            node.get_logger().warn("[capture_barcode_and_x_offset] decode failed after retries.")
            return False, None, None

        frame = decoded_frame
        bbox = decoded_bbox
        code_str = decoded_code_str
        formatted_code = format_barcode_code(code_str)

        if frame is None or bbox is None:
            node.get_logger().warn("[capture_barcode_and_x_offset] frame/bbox missing at success (unexpected)")
            return False, None, None

        offset_u_px, u_b, v_b, X_m = compute_X_m_from_bbox_center(frame, bbox, fx_px, depth_m)
        H, W = frame.shape[:2]

        info = {
            "image_width": int(W),
            "image_height": int(H),
            "u_center": float(u_b),
            "v_center": float(v_b),
            "offset_u_px": float(offset_u_px),
            "X_m": float(X_m),
        }

        x1, y1, x2, y2 = bbox
        vis = draw_bbox(frame, (x1, y1, x2 - x1, y2 - y1), label=code_str)
        save_image(vis, shot_dir / "bookshelf_barcode_capture_selfloc_bbox.png")

        msg = String()
        msg.data = f"{formatted_code}/{X_m:.6f}"
        pub_err.publish(msg)
        node.get_logger().info(f"[error_x published] {msg.data}")

        return True, code_str, info

    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if yaw_watcher is not None:
                yaw_watcher.destroy()
        except Exception:
            pass
# =========================================================
# Standalone test main (optional)
# =========================================================
def main():
    try:
        print("=== BookShelf Barcode Align+Decode (ROS2) ===")
        rclpy.init()
        from rclpy.executors import SingleThreadedExecutor
        node = rclpy.create_node("bookshelf_barcode_align_decode_test")
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        shot_dir = Path("/home/book/pro_book_SAM3/pro_hand_book_python/captures/bookshelf_barcode_test")
        shot_dir.mkdir(parents=True, exist_ok=True)

        # trigger
        nav_pub = node.create_publisher(Bool, "/navigation_goal", 10)
        nav_pub.publish(Bool(data=True))
        node.get_logger().info("[test] published /navigation_goal=True")

        detected, code_str, info = capture_barcode_and_x_offset(
            node=node,
            executor=executor,
            shot_dir=shot_dir,
            barcode_number=None,
            fx_px=2500.0,
            depth_m=0.40,
            cmd_vel_topic="/cmd_vel_debug",  # 誤動作防止
            post_stop_settle_sec=0.7,
            decode_attempts=3,
            attempt_interval_sec=0.4,
        )

        print("detected:", detected)
        print("code_str:", code_str)
        print("info:", info)

    except Exception:
        traceback.print_exc()
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()