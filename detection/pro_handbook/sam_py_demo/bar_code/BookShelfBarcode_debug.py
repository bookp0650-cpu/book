#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BookShelfBarcode_debug.py

目的:
- 既存の ROS2 統合コードや code_1_pic_ros2.py を大きく改変せずに、
  「保存画像」からバーコード検出〜Xオフセット算出〜/error_x publish までを
  1ファイルで完結させてデバッグできるようにする。

使い方（例）:
  python -m detection.pro_handbook.sam_py_demo.bar_code.BookShelfBarcode_debug \
    --img /home/book/pro_book_SAM3/pro_hand_book_python/captures/bookshelf_barcode/bookshelf_barcode_capture_selfloc_20260303_161654.png \
    --shot-dir /home/book/pro_book_SAM3/pro_hand_book_python/captures/bookshelf_barcode \
    --fx 2500 --depth 0.40 --no-wait-nav --no-capture

注意:
- 本ファイルは「デバッグ専用」です（統合コード側は触らない前提）。
- 画像前処理の途中結果を shot_dir に保存します。
- /error_x の publish も行います（不要なら --no-publish）。
"""

import argparse
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
from pyzbar.pyzbar import decode

import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Bool, String, Float32

# （任意）カメラ撮影を使いたい場合だけ import
try:
    from detection.pro_handbook.sam_py_demo.bar_code.web_camera_capture import (
        capture_one_depstech,
    )
except Exception:
    capture_one_depstech = None


# =========================================================
#  ROS2 Watchers
# =========================================================
class NavigationGoalWatcher:
    def __init__(self, node, topic_name: str = "/navigation_goal"):
        self._node = node
        self._received_true = False
        self._sub = node.create_subscription(Bool, topic_name, self._callback, 10)

    def _callback(self, msg: Bool):
        if msg.data:
            self._received_true = True
            self._node.get_logger().info("Received navigation_goal = True")

    def wait_until_true(self, executor, timeout_sec: float | None = None) -> bool:
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
    def __init__(self, node, topic_name: str = "/wall_distance"):
        self._node = node
        self._distance: Optional[float] = None
        self._sub = node.create_subscription(Float32, topic_name, self._callback, 10)

    def _callback(self, msg: Float32):
        self._distance = float(msg.data)

    def get_distance(self) -> Optional[float]:
        return self._distance

    def wait_latest(self, executor, timeout_sec: float = 2.0) -> Optional[float]:
        start = time.time()
        while rclpy.ok() and self._distance is None:
            executor.spin_once(timeout_sec=0.1)
            if (time.time() - start) > timeout_sec:
                break
        return self._distance

    def destroy(self):
        try:
            self._node.destroy_subscription(self._sub)
        except Exception:
            pass


# =========================================================
#  Utils
# =========================================================
def save_image(img, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(save_path), img)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {save_path}")


def format_barcode_code(raw_code: str) -> str:
    """
    例:
      030104070500 -> 1-4-7-5
    """
    parts = [raw_code[i : i + 2] for i in range(0, len(raw_code), 2)]
    if len(parts) < 5:
        return raw_code
    selected = parts[1:5]
    selected_int = [str(int(p)) for p in selected]
    return "-".join(selected_int)


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
            raise ValueError("dict data must be {x,y,w,h} or {left,top,width,height}")
    else:
        raise ValueError("data must be (x,y,w,h) or dict")

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
#  Barcode selection
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

    best = None
    min_dist2 = float("inf")
    for d in decode_data:
        rect = d.rect
        u_b = rect.left + rect.width / 2.0
        v_b = rect.top + rect.height / 2.0
        dist2 = (u_b - u_c) ** 2 + (v_b - v_c) ** 2
        if dist2 < min_dist2:
            min_dist2 = dist2
            best = d
    return best


# =========================================================
#  Preprocess (ノイズ抑制 + 垂直線補完の候補を追加)
# =========================================================
def preprocess_for_barcode(gray: "cv2.Mat", save_dir: Optional[Path] = None) -> Dict[str, "cv2.Mat"]:
    """
    いくつか前処理候補を生成し、decode しやすいものを探すための辞書を返す。
    - ノイズが強い/細線が消える/線が癒着する等に対して、
      複数候補を出して「どれが読むか」を試す方針。

    返り値: {"name": image, ...}
    """

    outs: Dict[str, cv2.Mat] = {}

    # 0) そのまま
    outs["gray"] = gray

    # 1) 軽い平滑化 + CLAHE
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(blur)
    outs["clahe"] = clahe_img

    # 2) 適応二値化（現状）
    bin_adapt = cv2.adaptiveThreshold(
        clahe_img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        25,
        10,
    )
    outs["bin_adapt"] = bin_adapt

    # 3) median でゴマノイズ抑制 → Otsu（二値化）
    med = cv2.medianBlur(clahe_img, 3)
    _, bin_otsu = cv2.threshold(med, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    outs["bin_otsu"] = bin_otsu

    # 4) 「垂直線を優先的につなぐ」: 縦長カーネルで close（※前景=黒線想定なので反転して処理）
    #    - 二値化画像を反転して黒線を白(前景)として扱う
    inv = cv2.bitwise_not(bin_adapt)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))  # 調整ポイント
    inv_close_v = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, vert_kernel, iterations=1)
    bin_adapt_vertclose = cv2.bitwise_not(inv_close_v)
    outs["bin_adapt_vertclose"] = bin_adapt_vertclose

    # 5) 癒着を軽減するため「細線を守る」方向: close しすぎない（small close のみ）
    inv2 = cv2.bitwise_not(bin_adapt)
    small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    inv2 = cv2.morphologyEx(inv2, cv2.MORPH_CLOSE, small, iterations=1)
    outs["bin_adapt_smallclose"] = cv2.bitwise_not(inv2)

    # 6) もし背景のムラが強いなら top-hat も候補に（文字/バーを強調）
    #    ただしやりすぎると線が太るので候補として保存して試す
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, tophat_kernel)
    # tophat→Otsu
    _, bin_tophat = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    outs["bin_tophat_otsu"] = bin_tophat

    # 保存
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for k, im in outs.items():
            # 8bit 画像想定。もし違うなら変換が必要。
            save_image(im, save_dir / f"pre_{k}.png")

    return outs


def barcode_detect_unified(
    img,
    barcode_number: Optional[str] = None,
    save_dir: Optional[Path] = None,
) -> Tuple[bool, Optional[Any], List[Any], Optional[str]]:
    """
    画像中のバーコードを検出する共通関数（デバッグ版）
    - preprocess の複数候補を作り、順に decode を試す
    - どの候補でヒットしたか (hit_name) も返す
    """

    if img is None:
        return False, None, [], None

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 前処理候補を生成（全部保存も可能）
    pre_dict = preprocess_for_barcode(gray, save_dir=save_dir)

    # decode を試す順番（おすすめ順に並べる）
    candidates: List[Tuple[str, np.ndarray]] = [
        ("gray", pre_dict["gray"]),
        ("clahe", pre_dict["clahe"]),
        ("bin_adapt", pre_dict["bin_adapt"]),
        ("bin_adapt_smallclose", pre_dict["bin_adapt_smallclose"]),
        ("bin_adapt_vertclose", pre_dict["bin_adapt_vertclose"]),
        ("bin_otsu", pre_dict["bin_otsu"]),
        ("bin_tophat_otsu", pre_dict["bin_tophat_otsu"]),
    ]

    last_decoded: List[Any] = []
    selected = None
    hit_name: Optional[str] = None

    for name, cand in candidates:
        decode_data = decode(cand)
        last_decoded = decode_data

        print(f"[decode] candidate={name} len={len(decode_data)}")

        if len(decode_data) == 0:
            continue

        selected = _select_barcode(decode_data, img, target_code=barcode_number)
        if selected is not None:
            hit_name = name
            break

    if selected is None:
        return False, None, last_decoded, None

    return True, selected, last_decoded, hit_name


# =========================================================
#  Compute offset
# =========================================================
def compute_barcode_center_and_x_offset(
    barcode_data,
    frame,
    fx_px: float,
    depth_m: float,
) -> Dict[str, float]:
    H, W = frame.shape[:2]
    u_c = W / 2.0
    v_c = H / 2.0

    rect = barcode_data.rect
    u_b = rect.left + rect.width / 2.0
    v_b = rect.top + rect.height / 2.0

    offset_u_px = u_b - u_c
    X_m = (offset_u_px / fx_px) * depth_m

    return {
        "image_width": W,
        "image_height": H,
        "u_center": u_b,
        "v_center": v_b,
        "offset_u_px": offset_u_px,
        "X_m": X_m,
    }


# =========================================================
#  Main flow (デバッグ版 capture_barcode_and_x_offset)
# =========================================================
def capture_barcode_and_x_offset_debug(
    node,
    executor,
    shot_dir: Path,
    barcode_number: Optional[str] = None,
    fx_px: float = 2500.0,
    depth_m: float = 0.40,
    frame_path: Optional[Path] = None,
    wait_for_navigation: bool = True,
    do_capture: bool = True,
    publish_error_x: bool = True,
) -> Tuple[bool, Optional[str], Optional[Dict[str, float]]]:
    """
    デバッグ用:
    - wait_for_navigation=False にすれば /navigation_goal を待たない
    - do_capture=False なら frame_path の画像を読む
    - 前処理画像を shot_dir に全保存する（pre_*.png）
    - /error_x publish も可能
    """

    shot_dir = Path(shot_dir)
    shot_dir.mkdir(parents=True, exist_ok=True)

    # /navigation_goal を待つ（必要なら）
    if wait_for_navigation:
        node.get_logger().info("[debug] Waiting /navigation_goal == True ...")
        watcher = NavigationGoalWatcher(node)
        ok = watcher.wait_until_true(executor, timeout_sec=None)
        watcher.destroy()
        if not ok:
            node.get_logger().warn("[debug] /navigation_goal wait aborted → skip")
            return False, None, None
        node.get_logger().info("[debug] /navigation_goal received")

    # 画像取得
    frame = None
    if do_capture:
        if capture_one_depstech is None:
            node.get_logger().error("capture_one_depstech import failed. Use --no-capture with --img.")
            return False, None, None
        frame = capture_one_depstech(shot_dir / "bookshelf_barcode_capture_selfloc.png")
    else:
        if frame_path is None:
            node.get_logger().error("frame_path is None but do_capture=False")
            return False, None, None
        frame = cv2.imread(str(frame_path))
        if frame is None:
            node.get_logger().error(f"Failed to load image: {frame_path}")
            return False, None, None
        # 元画像のコピーも shot_dir に置いておく（追跡しやすい）
        save_image(frame, shot_dir / "bookshelf_barcode_loaded.png")

    # 検出（前処理画像は shot_dir に保存される）
    detected, selected, all_decoded, hit_name = barcode_detect_unified(
        frame,
        barcode_number=barcode_number,
        save_dir=shot_dir,  # pre_*.png を全部吐く
    )

    if not detected or selected is None:
        node.get_logger().warn("[debug] no barcode selected.")
        return False, None, None

    code_str = selected.data.decode("utf-8", errors="ignore")
    node.get_logger().info(f"[debug] selected barcode: {code_str} (hit={hit_name})")

    info = compute_barcode_center_and_x_offset(
        barcode_data=selected,
        frame=frame,
        fx_px=fx_px,
        depth_m=depth_m,
    )
    x_error = float(info["X_m"])

    # bbox 可視化
    vis = draw_bbox(frame, selected.rect, label=code_str)
    save_image(vis, shot_dir / "bookshelf_barcode_bbox.png")

    # publish /error_x
    if publish_error_x:
        if not hasattr(capture_barcode_and_x_offset_debug, "_error_x_pub"):
            capture_barcode_and_x_offset_debug._error_x_pub = node.create_publisher(String, "/error_x", 10)
        pub = capture_barcode_and_x_offset_debug._error_x_pub

        formatted_code = format_barcode_code(code_str)
        msg = String()
        msg.data = f"{formatted_code}/{x_error:.6f}"
        pub.publish(msg)
        node.get_logger().info(f"[debug] /error_x published: {msg.data}")

    return True, code_str, info


# =========================================================
#  CLI main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img",
        type=str,
        default="",
        help="保存画像パス（--no-capture 時に使用）",
    )
    parser.add_argument(
        "--shot-dir",
        type=str,
        default="",
        help="出力先ディレクトリ（省略時は img の親 or デフォルト）",
    )
    parser.add_argument("--fx", type=float, default=2500.0)
    parser.add_argument("--depth", type=float, default=0.40)
    parser.add_argument("--barcode", type=str, default="", help="ターゲットバーコード（空なら中央選択）")

    parser.add_argument("--no-wait-nav", action="store_true", help="/navigation_goal を待たない")
    parser.add_argument("--no-capture", action="store_true", help="カメラ撮影しない（保存画像から読む）")
    parser.add_argument("--no-publish", action="store_true", help="/error_x を publish しない")

    parser.add_argument("--use-wall-distance", action="store_true", help="/wall_distance を読んで depth を上書き（取れなければ --depth を使用）")

    args = parser.parse_args()

    img_path = Path(args.img) if args.img else None
    barcode_number = args.barcode.strip() or None

    # shot_dir の決定
    if args.shot_dir:
        shot_dir = Path(args.shot_dir)
    else:
        if img_path is not None:
            shot_dir = img_path.parent
        else:
            shot_dir = Path("/home/book/pro_book_SAM3/pro_hand_book_python/captures/bookshelf_barcode")

    # ROS2
    rclpy.init()
    node = rclpy.create_node("bookshelf_barcode_debug")
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        depth_m = float(args.depth)

        # /wall_distance を使うなら取得
        if args.use_wall_distance:
            w = WallDistanceWatcher(node)
            d = w.wait_latest(executor, timeout_sec=2.0)
            w.destroy()
            if d is not None:
                depth_m = float(d)
                node.get_logger().info(f"[debug] depth from /wall_distance = {depth_m:.3f} m")
            else:
                node.get_logger().warn("[debug] /wall_distance not received. Use --depth")

        detected, code_str, info = capture_barcode_and_x_offset_debug(
            node=node,
            executor=executor,
            shot_dir=shot_dir,
            barcode_number=barcode_number,
            fx_px=float(args.fx),
            depth_m=depth_m,
            frame_path=img_path,
            wait_for_navigation=(not args.no_wait_nav),
            do_capture=(not args.no_capture),
            publish_error_x=(not args.no_publish),
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