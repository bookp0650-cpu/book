#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code_1_pic_ros2.py

目的:
- /navigation_goal=True を受信したら処理開始
- 連続フレームでバーコードBBoxを検出
- まずは停止状態でOCR(番号: ab-07-0c)を試す
- 読めなければ、撮影＆OCRを回しながら /cmd_vel (linear.xのみ, angular.z=0) でバーコードを画像中央に寄せる
- OCR成功した瞬間に /cmd_vel を停止し、/error_x に "<整形ID>/<error_x>" をpublishして終了

今回の制約:
- 1ブロック目: 01～14
- 2ブロック目: 07 固定
- 3ブロック目: 01～05

改善内容:
- OCR結果がこの制約を満たすときだけ採用
- 満たさない場合、ROIの左側を少しずつ削って再認識
"""

import re
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Bool, String, Float32
from geometry_msgs.msg import Twist

import easyocr


# =========================================================
# QoS
# =========================================================
CMDVEL_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


# =========================================================
# Utility
# =========================================================
def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def save_image(img, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")


def append_log(log_path: Path, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


# =========================================================
# Watchers
# =========================================================
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


class NavigationGoalWatcher:
    def __init__(self, node: Node, topic_name: str = "/navigation_goal"):
        self._node = node
        self._received_true = False
        self._sub = node.create_subscription(Bool, topic_name, self._cb, 10)

    def _cb(self, msg: Bool):
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


class WallDistanceWatcher:
    def __init__(self, node: Node, topic_name: str = "/wall_distance"):
        self._node = node
        self._distance: Optional[float] = None
        self._sub = node.create_subscription(Float32, topic_name, self._cb, 10)

    def _cb(self, msg: Float32):
        self._distance = float(msg.data)

    def get_distance(self) -> Optional[float]:
        return self._distance

    def destroy(self):
        try:
            self._node.destroy_subscription(self._sub)
        except Exception:
            pass


# =========================================================
# /cmd_vel repeater
# =========================================================
# =========================================================
# /cmd_vel repeater
# =========================================================
class CmdVelRepeater:
    def __init__(self, node: Node, topic: str = "/cmd_vel", rate_hz: float = 10.0):
        self._node = node
        self._pub = node.create_publisher(Twist, topic, CMDVEL_QOS)
        self._vx = 0.0
        self._timer = node.create_timer(1.0 / rate_hz, self._on_timer)
        self._vx_dbg_pub = node.create_publisher(Float32, "/cmd_vel_vx_debug", 10)

    def _publish_now(self):
        msg = Twist()
        msg.linear.x = float(self._vx)
        msg.angular.z = 0.0
        self._pub.publish(msg)
        self._vx_dbg_pub.publish(Float32(data=float(self._vx)))

    def set(self, vx: float):
        self._vx = float(vx)
        self._publish_now()   # 変更直後に即 publish

    def stop(self):
        self._vx = 0.0
        self._publish_now()   # 停止も即 publish

    def publish_zero_once(self):
        self._vx = 0.0
        self._publish_now()

    def _on_timer(self):
        # keep-alive 用
        self._publish_now()

    def destroy(self):
        try:
            self._timer.cancel()
        except Exception:
            pass


# =========================================================
# Camera open
# =========================================================
def find_video_devices_by_name(keyword="Depstech") -> List[str]:
    import glob
    import os

    key = keyword.lower()
    devs = []
    for v in sorted(glob.glob("/sys/class/video4linux/video*")):
        name_path = os.path.join(v, "name")
        try:
            name = open(name_path, "r").read().strip()
        except Exception:
            continue
        if key in name.lower():
            devs.append("/dev/" + os.path.basename(v))
    return devs


def find_first_openable_video_device(keyword="Depstech") -> Optional[str]:
    for dev in find_video_devices_by_name(keyword):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        ok = cap.isOpened()
        cap.release()
        if ok:
            return dev
    return None


def open_depstech_cap(width=3840, height=2160, fps=30, fourcc="MJPG") -> Tuple[cv2.VideoCapture, str]:
    dev = find_first_openable_video_device("Depstech")
    if dev is None:
        dev = "/dev/video0"

    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {dev}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    afps = cap.get(cv2.CAP_PROP_FPS)
    return cap, f"{dev} ({aw}x{ah}@{afps:.1f}fps)"


# =========================================================
# Barcode detection
# =========================================================
def _detect_barcode_bbox_opencv(frame_bgr: np.ndarray):
    if not (hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector")):
        return None

    det = cv2.barcode.BarcodeDetector()
    ok, corners = det.detect(frame_bgr)
    if (not ok) or corners is None:
        return None

    best = None
    best_x = 1e18

    for c in corners:
        c = np.array(c, dtype=np.float32).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(c.astype(np.int32))

        # 左にあるバーコードを優先
        if x < best_x:
            best_x = x
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

    best = None
    best_x = 1e18

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < (W * H) * 0.002:
            continue
        ar = w / max(1, h)
        if ar < 1.2:
            continue

        # 左にある候補を優先
        if x < best_x:
            best_x = x
            best = (float(x), float(y), float(x + w), float(y + h))

    return best


def detect_barcode_bbox(frame_bgr: np.ndarray):
    b = _detect_barcode_bbox_opencv(frame_bgr)
    if b is not None:
        return b
    return _detect_barcode_bbox_fallback(frame_bgr)


def draw_bbox(img: np.ndarray, bbox_xyxy, label: Optional[str] = None):
    x1, y1, x2, y2 = bbox_xyxy
    out = img.copy()
    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
    cv2.rectangle(out, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
    if label:
        cv2.putText(out, label, (x1i, max(0, y1i - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


# =========================================================
# Geometry / ROI
# =========================================================
def compute_error_x_from_bbox(frame_bgr: np.ndarray, bbox_xyxy, fx_px: float, depth_m: float):
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    u_b = (x1 + x2) / 2.0
    v_b = (y1 + y2) / 2.0
    offset_u_px = float(u_b - (W / 2.0))
    X_m = float((offset_u_px / fx_px) * depth_m)
    return offset_u_px, u_b, v_b, X_m


def crop_band(frame_rot):
    H, W = frame_rot.shape[:2]
    y1 = int(H * 0.15)
    y2 = int(H * 0.45)
    return frame_rot[y1:y2, :]


def make_label_roi_strict(frame_rot, bbox):
    band = crop_band(frame_rot)

    x1, y1, x2, y2 = bbox
    bw = x2 - x1

    X1 = int(x2 - bw * 0.35)
    X2 = int(x2 + bw * 3.30)

    H_band, W_band = band.shape[:2]
    X1 = max(0, min(W_band - 1, X1))
    X2 = max(0, min(W_band, X2))

    if X2 <= X1:
        return band

    return band[:, X1:X2]


# =========================================================
# OCR
# =========================================================
reader = easyocr.Reader(['en'], gpu=True)


def run_ocr_easy(roi_bgr):
    results = reader.readtext(
        roi_bgr,
        detail=1,
        paragraph=False,
        allowlist='0123456789-',
        decoder='greedy',
        width_ths=0.7,
        ycenter_ths=0.5,
        height_ths=0.5,
    )

    texts = []
    for bbox, txt, conf in results:
        texts.append((txt, float(conf)))
    return texts


def normalize_ocr_text(txt: str) -> str:
    table = str.maketrans({
        "O": "0", "o": "0",
        "I": "1", "l": "1",
        "S": "5", "s": "5",
        "B": "8",
        "Z": "2", "z": "2",
    })
    return txt.translate(table)


def validate_label_07_format(label: str) -> bool:
    if label is None:
        return False

    m = re.fullmatch(r"(\d{2})-(\d{2})-(\d{2})", label)
    if not m:
        return False

    p1 = int(m.group(1))
    p2 = m.group(2)
    p3 = int(m.group(3))

    if not (1 <= p1 <= 14):
        return False
    if p2 != "07":
        return False
    if not (1 <= p3 <= 5):
        return False

    return True

def looks_like_label_fully_visible(results) -> bool:
    """
    左クロップをかける前に、
    「番号全体がすでにROI内に写っていそうか」を緩く判定する。

    True:
      - 6桁以上読めている
      - またはハイフン込みで xx-xx-xx に近い文字数が見えている
    False:
      - まだ桁が足りず、番号全体が写り切っていない可能性が高い
    """
    if results is None or len(results) == 0:
        return False

    digit_counts = []
    hyphen_like_found = False

    for txt, conf in results:
        txt_n = normalize_ocr_text(txt)
        digits = re.sub(r"\D", "", txt_n)

        digit_counts.append(len(digits))

        # 例: 12-07-03, 120703, 12 07 03 っぽいものを広めに許容
        if re.search(r"\d{2}\D*\d{2}\D*\d{2}", txt_n):
            hyphen_like_found = True

    if hyphen_like_found:
        return True

    if len(digit_counts) == 0:
        return False

    # どれかで6桁以上見えていれば「全体は写っている」とみなす
    if max(digit_counts) >= 6:
        return True

    return False

def extract_label_easy_strict(results):
    best_label = None
    best_conf = -1.0

    for txt, conf in results:
        txt = normalize_ocr_text(txt)
        digits = re.sub(r"\D", "", txt)

        if len(digits) != 6:
            continue

        label = f"{digits[:2]}-{digits[2:4]}-{digits[4:6]}"

        if validate_label_07_format(label):
            if float(conf) > best_conf:
                best_label = label
                best_conf = float(conf)

    return best_label


def crop_left_progressively(img: np.ndarray, left_ratios=(0.0, 0.05, 0.10, 0.15, 0.20, 0.25)):
    H, W = img.shape[:2]
    crops = []

    for r in left_ratios:
        x1 = int(W * r)
        if x1 >= W - 5:
            continue
        crops.append((r, img[:, x1:]))

    return crops

def run_ocr_without_left_crop(
    roi_bgr,
    log_path=None,
    shot_dir: Optional[Path] = None,
    tag: Optional[str] = None
):
    """
    左クロップなしで1回だけOCRする。
    戻り値:
      label, conf, raw_texts, fully_visible
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None, -1.0, [], False

    if shot_dir is not None and tag is not None:
        save_image(roi_bgr, shot_dir / f"dbg_ocr_roi_nocrop_{tag}.png")

    results = run_ocr_easy(roi_bgr)

    raw_texts = []
    confs = []
    for txt, conf in results:
        raw_texts.append(txt)
        confs.append(float(conf))

    label = extract_label_easy_strict(results)
    fully_visible = looks_like_label_fully_visible(results)
    conf_avg = float(np.mean(confs)) if confs else 0.0

    if log_path is not None:
        append_log(
            log_path,
            f"[OCR nocrop] raw={' | '.join(raw_texts)}, "
            f"label={label}, fully_visible={fully_visible}, conf={conf_avg:.3f}"
        )

    return label, conf_avg, raw_texts, fully_visible

def run_ocr_with_left_crop_retry(
    roi_bgr,
    log_path=None,
    shot_dir: Optional[Path] = None,
    tag: Optional[str] = None
):
    crop_candidates = crop_left_progressively(
        roi_bgr,
        left_ratios=(0.05, 0.10, 0.15, 0.20, 0.25)
    )

    best_label = None
    best_conf = -1.0
    best_crop_ratio = None
    best_raw_texts = []

    for crop_ratio, roi in crop_candidates:
        if roi is None or roi.size == 0:
            continue

        if shot_dir is not None and tag is not None:
            ratio_str = f"{int(crop_ratio * 100):02d}"
            save_image(roi, shot_dir / f"dbg_ocr_roi_leftcut_{ratio_str}_{tag}.png")

        results = run_ocr_easy(roi)

        raw_texts = []
        confs = []
        for txt, conf in results:
            raw_texts.append(txt)
            confs.append(float(conf))

        label = extract_label_easy_strict(results)

        if log_path is not None:
            append_log(
                log_path,
                f"[OCR retry] crop_left={crop_ratio:.2f}, raw={' | '.join(raw_texts)}, label={label}"
            )

        if label is not None:
            conf_avg = float(np.mean(confs)) if confs else 0.0
            if conf_avg > best_conf:
                best_label = label
                best_conf = conf_avg
                best_crop_ratio = crop_ratio
                best_raw_texts = raw_texts

    return best_label, best_conf, best_crop_ratio, best_raw_texts


# =========================================================
# ID formatting
# =========================================================
def label_to_publish_id(label_00_00_00: str, prefix: str = "02") -> Optional[str]:
    if label_00_00_00 is None:
        return None
    digits = re.sub(r"[^0-9]", "", label_00_00_00)
    if len(digits) != 6:
        return None
    return prefix + digits


def format_id(id_str):
    if len(id_str) != 8:
        return id_str

    parts = [id_str[i:i+2] for i in range(0, 8, 2)]
    parts = [str(int(p)) for p in parts]
    return "-".join(parts)


# =========================================================
# Main
# =========================================================
def capture_barcode_and_x_offset(
    node: Node,
    executor,
    shot_dir: Path,
    barcode_number: Optional[str] = None,
    fx_px: float = 2500.0,
    depth_m: float = 0.40,

    # 制御
    align_thresh_px: float = 20.0,
    v_x: float = 0.05,
    min_search_vx: float = 0.02,
    max_search_vx: float = 0.08,
    k_p_search: float = 0.00008,
    cmd_sign_x: float = 1.0,
    cmd_rate_hz: float = 10.0,

    # 時間
    total_timeout_sec: float = 25.0,
    ocr_interval_sec: float = 0.25,
    bbox_lost_grace_sec: float = 0.50,

    # navigation_goal
    wait_navigation_goal: bool = True,
    navigation_goal_topic: str = "/navigation_goal",
    navigation_goal_timeout_sec: Optional[float] = None,

    # publish
    cmd_vel_topic: str = "/cmd_vel",
    error_x_topic: str = "/error_x",
    id_prefix: str = "02",
    refind_timeout_sec: float = 2.0,
    refind_interval_sec: float = 0.10,

    # 停止後の再撮影用
    settle_wait_sec: float = 0.8,
    flush_frame_count: int = 5,
    final_retry_count: int = 10,
    final_retry_interval_sec: float = 0.1,
) -> Tuple[bool, Optional[str], Optional[Dict[str, float]]]:
    """
    仕様:
      1) 最初に1回だけ停止状態でOCR
      2) 有効ラベルでなければ、バーコードを中央へ寄せるために移動開始
      3) 移動中もOCRは回し続ける
      4) 中央に入ったら停止し、そのままOCR継続
      5) 有効ラベルを読めた瞬間に停止して publish
    """
    shot_dir = Path(shot_dir)
    shot_dir.mkdir(parents=True, exist_ok=True)
    log_path = shot_dir / "selfloc.log"

    pub_phase = node.create_publisher(String, "/ocr_phase", 10)
    pub_raw   = node.create_publisher(String, "/ocr_raw_text", 10)
    pub_label = node.create_publisher(String, "/ocr_label", 10)
    pub_off   = node.create_publisher(Float32, "/barcode_offset_u_px", 10)
    pub_bbox  = node.create_publisher(String, "/barcode_bbox_xyxy", 10)
    pub_err   = node.create_publisher(String, error_x_topic, 10)

    if wait_navigation_goal:
        node.get_logger().info(f"[selfloc] Waiting {navigation_goal_topic}==True ...")
        append_log(log_path, f"[{_ts()}] wait {navigation_goal_topic}")
        watcher = NavigationGoalWatcher(node, topic_name=navigation_goal_topic)
        ok = watcher.wait_until_true(executor, timeout_sec=navigation_goal_timeout_sec)
        watcher.destroy()
        if not ok:
            node.get_logger().warn("[selfloc] navigation_goal wait aborted")
            append_log(log_path, f"[{_ts()}] navigation_goal aborted")
            return False, None, None

    cmd = CmdVelRepeater(node, topic=cmd_vel_topic, rate_hz=cmd_rate_hz)
    cmd.stop()

    cap, devinfo = open_depstech_cap(width=3840, height=2160, fps=30, fourcc="MJPG")
    node.get_logger().info(f"[selfloc] stream opened: {devinfo}")
    append_log(log_path, f"[{_ts()}] stream opened: {devinfo}")

    for _ in range(10):
        cap.read()

    t0 = time.time()
    last_ocr_t = 0.0
    last_bbox_time = 0.0
    last_offset_u_px = 0.0

    phase = "FIRST_OCR"

    def calc_search_vx(offset_u_px: float) -> float:
        mag = abs(offset_u_px) * k_p_search
        mag = max(min_search_vx, mag)
        mag = min(max_search_vx, mag)
        direction = 1.0 if offset_u_px > 0 else -1.0
        return cmd_sign_x * mag * direction

    try:
        while rclpy.ok() and (time.time() - t0) < total_timeout_sec:
            try:
                executor.spin_once(timeout_sec=0.01)
            except Exception:
                pass

            ok, frame = cap.read()
            if not ok or frame is None:
                if phase in ("SEARCHING", "CENTER_HOLD"):
                    # 画像取得失敗でも即停止せず、timer keep-alive に任せる
                    pass
                else:
                    cmd.stop()
                continue

            frame_rot = cv2.rotate(frame, cv2.ROTATE_180)

            bbox = detect_barcode_bbox(frame_rot)
            now = time.time()

            if bbox is None:
                pub_phase.publish(String(data="NO_BBOX"))
                pub_off.publish(Float32(data=float(last_offset_u_px)))
                pub_bbox.publish(String(data=""))

                # SEARCHING中だけは短時間の見失いを許容
                if phase == "SEARCHING":
                    if (now - last_bbox_time) <= bbox_lost_grace_sec:
                        # 直前の速度を維持
                        pass
                    else:
                        node.get_logger().info("[selfloc] bbox lost for too long -> stop")
                        append_log(log_path, f"[{_ts()}] bbox lost too long -> stop")
                        cmd.stop()

                elif phase == "CENTER_HOLD":
                    if (now - last_bbox_time) > bbox_lost_grace_sec:
                        cmd.stop()

                else:
                    cmd.stop()

                continue

            last_bbox_time = now

            x1, y1, x2, y2 = bbox
            pub_bbox.publish(String(data=f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}"))

            offset_u_px, u_b, v_b, X_m = compute_error_x_from_bbox(
                frame_rot, bbox, fx_px, depth_m
            )
            last_offset_u_px = float(offset_u_px)
            pub_off.publish(Float32(data=float(offset_u_px)))

            # =====================================================
            # 移動制御
            # =====================================================
            if phase == "FIRST_OCR":
                cmd.stop()

            elif phase == "SEARCHING":
                if abs(offset_u_px) <= align_thresh_px:
                    cmd.stop()
                    node.get_logger().info(
                        f"[selfloc] barcode centered (offset_u_px={offset_u_px:.1f}) -> hold and continue OCR"
                    )
                    append_log(log_path, f"[{_ts()}] centered -> phase=CENTER_HOLD offset_u_px={offset_u_px:.1f}")
                    phase = "CENTER_HOLD"
                else:
                    vx_cmd = calc_search_vx(offset_u_px)
                    cmd.set(vx_cmd)

            elif phase == "CENTER_HOLD":
                if abs(offset_u_px) <= align_thresh_px:
                    cmd.stop()
                else:
                    node.get_logger().info(
                        f"[selfloc] barcode left center (offset_u_px={offset_u_px:.1f}) -> resume moving"
                    )
                    append_log(log_path, f"[{_ts()}] left center -> phase=SEARCHING offset_u_px={offset_u_px:.1f}")
                    phase = "SEARCHING"
                    vx_cmd = calc_search_vx(offset_u_px)
                    cmd.set(vx_cmd)

            # =====================================================
            # OCR実行
            # =====================================================
            now = time.time()
            if (now - last_ocr_t) < ocr_interval_sec:
                continue
            last_ocr_t = now

            pub_phase.publish(String(data=phase))

            roi = make_label_roi_strict(frame_rot, bbox)

            tag = _ts()
            save_image(frame_rot, shot_dir / f"dbg_frame_rot180_{tag}.png")
            save_image(draw_bbox(frame_rot, bbox, label="BARCODE"), shot_dir / f"dbg_barcode_bbox_{tag}.png")
            save_image(roi, shot_dir / f"dbg_ocr_roi_strict_{tag}.png")

            # 1) まずは左クロップなしでOCR
            label, label_conf, raw_texts, fully_visible = run_ocr_without_left_crop(
                roi_bgr=roi,
                log_path=log_path,
                shot_dir=shot_dir,
                tag=tag,
            )

            used_crop_ratio = None

            # 2) 有効ラベルならそのまま採用
            # 3) 有効でなくても、まだ全体が写っていないなら左クロップしない
            # 4) 全体は写っているのに不適なら、その時だけ左クロップ再試行
            if label is None and fully_visible:
                label, label_conf_retry, used_crop_ratio, raw_texts_retry = run_ocr_with_left_crop_retry(
                    roi_bgr=roi,
                    log_path=log_path,
                    shot_dir=shot_dir,
                    tag=tag,
                )

                if label is not None:
                    label_conf = label_conf_retry
                    raw_texts = raw_texts_retry

            pub_raw.publish(String(data=" ".join(raw_texts)))
            append_log(
                log_path,
                f"[{tag}] phase={phase} OCR_final={' '.join(raw_texts)} / "
                f"label={label} / crop_left={used_crop_ratio} / conf={label_conf:.3f} / "
                f"offset_u_px={offset_u_px:.1f}"
            )
            pub_label.publish(String(data=str(label) if label else ""))

            if label is not None:
                # -------------------------------------------------
                # 移動中に読めても、その場の値は使わない
                # 必ず stop -> wait -> 再撮影 -> 再認識した結果だけ publish
                # -------------------------------------------------
                append_log(
                    log_path,
                    f"[{tag}] OCR success detected during phase={phase} -> finalize after stop"
                )
                node.get_logger().info(
                    "[selfloc] OCR success detected -> stop, wait, and recapture for final publish"
                )

                ok_final, final_label, final_info = recapture_and_finalize_after_stop_simple(
                    node=node,
                    executor=executor,
                    cap=cap,
                    cmd=cmd,
                    shot_dir=shot_dir,
                    log_path=log_path,
                    fx_px=fx_px,
                    depth_m=depth_m,
                    pub_phase=pub_phase,
                    pub_raw=pub_raw,
                    pub_label=pub_label,
                    pub_off=pub_off,
                    pub_bbox=pub_bbox,
                    settle_wait_sec=settle_wait_sec,
                    flush_frame_count=flush_frame_count,
                    retry_count=final_retry_count,
                    retry_interval_sec=final_retry_interval_sec,
                )

                if not ok_final or final_label is None or final_info is None:
                    node.get_logger().warn("[selfloc] final recapture after stop failed")
                    append_log(log_path, f"[{_ts()}] final recapture failed")
                    continue

                publish_id = label_to_publish_id(final_label, prefix=id_prefix)
                if publish_id is None:
                    node.get_logger().warn(f"[selfloc] final OCR label format unexpected: {final_label}")
                    append_log(log_path, f"[{_ts()}] final bad label: {final_label}")
                    continue

                msg = String()
                formatted_id = format_id(publish_id)
                msg.data = f"{formatted_id}/{final_info['X_m']:.6f}"
                pub_err.publish(msg)

                node.get_logger().info(
                    f"[error_x published after full stop] {msg.data} "
                    f"(phase={final_info.get('phase')}, conf={final_info.get('ocr_conf', -1.0):.3f}, "
                    f"offset_u_px={final_info.get('offset_u_px', 0.0):.1f})"
                )
                append_log(log_path, f"[{_ts()}] SUCCESS publish after full stop {msg.data}")

                pub_phase.publish(String(data="SUCCESS"))
                final_info["publish_id"] = publish_id
                return True, final_label, final_info

            if phase == "FIRST_OCR":
                node.get_logger().info("[selfloc] first OCR failed -> start moving while continuing OCR")
                append_log(log_path, f"[{tag}] first OCR failed -> phase=SEARCHING")
                phase = "SEARCHING"

                # 状態遷移したその場で移動開始
                if abs(offset_u_px) > align_thresh_px:
                    vx_cmd = calc_search_vx(offset_u_px)
                    cmd.set(vx_cmd)
                else:
                    cmd.stop()
                    phase = "CENTER_HOLD"

        node.get_logger().warn("[selfloc] timeout -> failed")
        append_log(log_path, f"[{_ts()}] TIMEOUT")
        pub_phase.publish(String(data="TIMEOUT"))
        cmd.stop()
        return False, None, None

    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cmd.stop()
            cmd.destroy()
        except Exception:
            pass
def recapture_and_finalize_after_stop(
    node: Node,
    executor,
    cap,
    cmd: CmdVelRepeater,
    shot_dir: Path,
    log_path: Path,
    fx_px: float,
    depth_m: float,
    pub_phase,
    pub_raw,
    pub_label,
    pub_off,
    pub_bbox,
    wait_sec: float = 0.60,
    timeout_sec: float = 2.0,
    interval_sec: float = 0.10,
):
    """
    OCR成功を検知した後に、
    1) 必ず停止
    2) 少し待って静止
    3) 再撮影
    4) 停止後フレームで bbox / OCR / X_m を再計算
    して、publish 用の確定値を返す。

    戻り値:
      (ok, label, info_dict)
    """
    cmd.stop()
    t_wait_start = time.time()
    while rclpy.ok() and (time.time() - t_wait_start) < wait_sec:
        try:
            executor.spin_once(timeout_sec=0.01)
        except Exception:
            pass
        time.sleep(0.01)

    pub_phase.publish(String(data="REFIND_AFTER_STOP"))
    append_log(log_path, f"[{_ts()}] enter REFIND_AFTER_STOP")

    t0 = time.time()
    last_reason = "unknown"

    while rclpy.ok() and (time.time() - t0) < timeout_sec:
        try:
            executor.spin_once(timeout_sec=0.01)
        except Exception:
            pass

        ok, frame = cap.read()
        if not ok or frame is None:
            last_reason = "frame_read_failed"
            time.sleep(interval_sec)
            continue

        frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
        bbox = detect_barcode_bbox(frame_rot)

        tag = _ts()
        save_image(frame_rot, shot_dir / f"dbg_refind_frame_rot180_{tag}.png")

        if bbox is None:
            pub_bbox.publish(String(data=""))
            pub_phase.publish(String(data="REFIND_NO_BBOX"))
            append_log(log_path, f"[{tag}] REFIND no bbox")
            last_reason = "bbox_not_found"
            time.sleep(interval_sec)
            continue

        x1, y1, x2, y2 = bbox
        pub_bbox.publish(String(data=f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}"))

        offset_u_px, u_b, v_b, X_m = compute_error_x_from_bbox(
            frame_rot, bbox, fx_px, depth_m
        )
        pub_off.publish(Float32(data=float(offset_u_px)))

        dbg = draw_bbox(frame_rot, bbox, label="BARCODE")
        H, W = dbg.shape[:2]
        cx = int(W / 2)
        ub = int(u_b)
        cv2.line(dbg, (cx, 0), (cx, H - 1), (255, 0, 0), 2)  # camera center
        cv2.line(dbg, (ub, 0), (ub, H - 1), (0, 0, 255), 2)  # barcode center
        save_image(dbg, shot_dir / f"dbg_refind_barcode_bbox_{tag}.png")

        roi = make_label_roi_strict(frame_rot, bbox)
        save_image(roi, shot_dir / f"dbg_refind_ocr_roi_{tag}.png")

        # 停止後の確定OCR
        label, label_conf, used_crop_ratio, raw_texts = run_ocr_with_left_crop_retry(
            roi_bgr=roi,
            log_path=log_path,
            shot_dir=shot_dir,
            tag=f"refind_{tag}",
        )

        pub_raw.publish(String(data=" ".join(raw_texts)))
        pub_label.publish(String(data=str(label) if label else ""))

        append_log(
            log_path,
            f"[{tag}] REFIND OCR_final={' '.join(raw_texts)} / "
            f"label={label} / crop_left={used_crop_ratio} / conf={label_conf:.3f} / "
            f"offset_u_px={offset_u_px:.1f} / X_m={X_m:.6f}"
        )

        if label is not None:
            pub_phase.publish(String(data="REFIND_SUCCESS"))
            append_log(log_path, f"[{tag}] REFIND success")
            return True, label, {
                "label": label,
                "image_width": int(frame_rot.shape[1]),
                "image_height": int(frame_rot.shape[0]),
                "u_center": float(u_b),
                "v_center": float(v_b),
                "offset_u_px": float(offset_u_px),
                "X_m": float(X_m),
                "crop_left_ratio": float(used_crop_ratio) if used_crop_ratio is not None else None,
                "ocr_conf": float(label_conf),
                "phase": "REFIND_AFTER_STOP",
            }

        last_reason = "ocr_failed_after_stop"
        time.sleep(interval_sec)

    pub_phase.publish(String(data="REFIND_TIMEOUT"))
    append_log(log_path, f"[{_ts()}] REFIND timeout reason={last_reason}")
    return False, None, None

def recapture_and_finalize_after_stop_simple(
    node: Node,
    executor,
    cap,
    cmd: CmdVelRepeater,
    shot_dir: Path,
    log_path: Path,
    fx_px: float,
    depth_m: float,
    pub_phase,
    pub_raw,
    pub_label,
    pub_off,
    pub_bbox,
    settle_wait_sec: float = 0.8,
    flush_frame_count: int = 5,
    retry_count: int = 10,
    retry_interval_sec: float = 0.1,
):
    """
    移動中にOCR成功を検知した後、
    必ず stop -> 少し待つ -> 古いフレームを捨てる -> 最新フレームで再認識
    を行い、その停止後画像だけで最終結果を確定する。
    """
    cmd.stop()
    pub_phase.publish(String(data="STOP_AND_WAIT"))
    append_log(log_path, f"[{_ts()}] STOP_AND_WAIT for {settle_wait_sec:.2f}s")

    # 停止待ち
    time.sleep(settle_wait_sec)

    # 念のため executor を少し回す
    for _ in range(3):
        try:
            executor.spin_once(timeout_sec=0.01)
        except Exception:
            pass

    # カメラバッファを少し捨てる
    for _ in range(flush_frame_count):
        cap.read()

    pub_phase.publish(String(data="RECAPTURE_AFTER_STOP"))
    append_log(log_path, f"[{_ts()}] RECAPTURE_AFTER_STOP flush={flush_frame_count}")

    for i in range(retry_count):
        try:
            executor.spin_once(timeout_sec=0.01)
        except Exception:
            pass

        ok, frame = cap.read()
        if not ok or frame is None:
            append_log(log_path, f"[{_ts()}] recapture failed ({i+1}/{retry_count})")
            time.sleep(retry_interval_sec)
            continue

        frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
        bbox = detect_barcode_bbox(frame_rot)

        tag = _ts()
        save_image(frame_rot, shot_dir / f"dbg_final_frame_rot180_{tag}.png")

        if bbox is None:
            pub_bbox.publish(String(data=""))
            pub_phase.publish(String(data="FINAL_NO_BBOX"))
            append_log(log_path, f"[{tag}] FINAL no bbox")
            time.sleep(retry_interval_sec)
            continue

        x1, y1, x2, y2 = bbox
        pub_bbox.publish(String(data=f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}"))

        offset_u_px, u_b, v_b, X_m = compute_error_x_from_bbox(
            frame_rot, bbox, fx_px, depth_m
        )
        pub_off.publish(Float32(data=float(offset_u_px)))

        dbg = draw_bbox(frame_rot, bbox, label="FINAL_BARCODE")
        H, W = dbg.shape[:2]
        cx = int(W / 2)
        ub = int(u_b)
        cv2.line(dbg, (cx, 0), (cx, H - 1), (255, 0, 0), 2)   # camera center
        cv2.line(dbg, (ub, 0), (ub, H - 1), (0, 0, 255), 2)   # barcode center
        save_image(dbg, shot_dir / f"dbg_final_bbox_{tag}.png")

        roi = make_label_roi_strict(frame_rot, bbox)
        save_image(roi, shot_dir / f"dbg_final_roi_{tag}.png")

        label, label_conf, used_crop_ratio, raw_texts = run_ocr_with_left_crop_retry(
            roi_bgr=roi,
            log_path=log_path,
            shot_dir=shot_dir,
            tag=f"final_{tag}",
        )

        pub_raw.publish(String(data=" ".join(raw_texts)))
        pub_label.publish(String(data=str(label) if label else ""))

        append_log(
            log_path,
            f"[{tag}] FINAL OCR={' '.join(raw_texts)} / "
            f"label={label} / crop_left={used_crop_ratio} / conf={label_conf:.3f} / "
            f"offset_u_px={offset_u_px:.1f} / X_m={X_m:.6f}"
        )

        if label is not None:
            pub_phase.publish(String(data="FINAL_SUCCESS"))
            return True, label, {
                "label": label,
                "image_width": int(frame_rot.shape[1]),
                "image_height": int(frame_rot.shape[0]),
                "u_center": float(u_b),
                "v_center": float(v_b),
                "offset_u_px": float(offset_u_px),
                "X_m": float(X_m),
                "crop_left_ratio": float(used_crop_ratio) if used_crop_ratio is not None else None,
                "ocr_conf": float(label_conf),
                "phase": "FINAL_AFTER_STOP",
            }

        time.sleep(retry_interval_sec)

    pub_phase.publish(String(data="FINAL_TIMEOUT"))
    append_log(log_path, f"[{_ts()}] FINAL_TIMEOUT")
    return False, None, None
# =========================================================
# Standalone test
# =========================================================
def main():
    try:
        rclpy.init()
        from rclpy.executors import SingleThreadedExecutor

        node = rclpy.create_node("bookshelf_selfloc_test")
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        outdir = Path("/home/book/pro_book_SAM3/pro_hand_book_python/captures/bookshelf_barcode/_selfloc_test")
        outdir.mkdir(parents=True, exist_ok=True)

        nav_pub = node.create_publisher(Bool, "/navigation_goal", 10)
        nav_pub.publish(Bool(data=True))
        node.get_logger().info("[test] published /navigation_goal=True")

        ok, label, info = capture_barcode_and_x_offset(
            node=node,
            executor=executor,
            shot_dir=outdir,
            fx_px=2500.0,
            depth_m=0.40,
            v_x=0.01,
            cmd_sign_x=1.0,
            align_thresh_px=40.0,
            total_timeout_sec=20.0,
            cmd_vel_topic="/cmd_vel",
        )
        print("ok:", ok)
        print("label:", label)
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