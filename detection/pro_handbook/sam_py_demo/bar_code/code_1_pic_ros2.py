#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import cv2
from pyzbar.pyzbar import decode

from detection.pro_handbook.sam_py_demo.bar_code.web_camera_capture import (
    capture_one_depstech,
)
import time  # 追加
import rclpy  # 追加
from std_msgs.msg import Bool, String  # 追加
from rclpy.executors import SingleThreadedExecutor


# =========================================================
#  ユーティリティ：画像保存
# =========================================================

# =========================================================
#  /navigation_goal を待つヘルパークラス
# =========================================================
from std_msgs.msg import Bool, String

class NavigationGoalWatcher:
    def __init__(self, node, topic_name: str = "/navigation_goal"):
        self._node = node
        self._received_true = False

        # Subscriber（受信用）
        self._sub = node.create_subscription(
            Bool,
            topic_name,
            self._callback,
            10,
        )

        # Publisher（確認用）
        self._pub = node.create_publisher(
            String,
            "/error_x",
            10,
        )

    def _callback(self, msg: Bool):
        if msg.data:
            self._received_true = True
            self._node.get_logger().info("Received navigation_goal = True")

            # # debug publish
            # debug_msg = String()
            # debug_msg.data = "Navigation goal received!"
            # self._pub.publish(debug_msg)

    def wait_until_true(self, executor, timeout_sec: float | None = None) -> bool:
        """
        executor.spin_once() を回しながら /navigation_goal==True を待つ。
        timeout_sec=None の場合はタイムアウトなし。
        戻り値: True を受け取れたら True, タイムアウト/中断で False。
        """
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

from std_msgs.msg import Float32

class WallDistanceWatcher:
    def __init__(self, node, topic_name: str = "/wall_distance"):
        self._node = node
        self._distance = None

        self._sub = node.create_subscription(
            Float32,
            topic_name,
            self._callback,
            10,
        )

    def _callback(self, msg: Float32):
        self._distance = float(msg.data)

    def get_distance(self):
        return self._distance

    def destroy(self):
        try:
            self._node.destroy_subscription(self._sub)
        except Exception:
            pass

def save_image(img_bgr, save_path: str | Path) -> None:
    """
    img_bgr: OpenCV画像 (BGR)
    save_path: 保存先パス
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(save_path), img_bgr)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {save_path}")

def format_barcode_code(raw_code: str) -> str:
    """
    例:
    030104070500
      ↓
    1-4-7-5
    """

    # 2桁ずつに分割
    parts = [raw_code[i:i+2] for i in range(0, len(raw_code), 2)]

    # 安全チェック
    if len(parts) < 5:
        return raw_code  # 想定外ならそのまま返す

    # 2〜5番目を使う（index 1〜4）
    selected = parts[1:5]

    # int化して先頭ゼロを消す
    selected_int = [str(int(p)) for p in selected]

    # "-"で連結
    return "-".join(selected_int)

# =========================================================
#  ユーティリティ：バウンディングボックス描画
# =========================================================
def draw_bbox(
    img,
    data,
    color=(0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None,
):
    """
    img : 入力画像 (BGR, cv2画像)
    data: (x, y, w, h) もしくは dict{ x,y,w,h } or { left,top,width,height }
    """
    out = img.copy()

    # ---- data の取り出し ----
    if isinstance(data, (tuple, list)) and len(data) == 4:
        x, y, w, h = data
    elif isinstance(data, dict):
        if all(k in data for k in ("x", "y", "w", "h")):
            x, y, w, h = data["x"], data["y"], data["w"], data["h"]
        elif all(k in data for k in ("left", "top", "width", "height")):
            x, y, w, h = data["left"], data["top"], data["width"], data["height"]
        else:
            raise ValueError(
                "dict形式の data は {x,y,w,h} か {left,top,width,height} にしてください"
            )
    else:
        raise ValueError("data は (x,y,w,h) か dict を渡してください")

    # ---- 数値化 & 画像範囲にクリップ ----
    x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
    H, W = out.shape[:2]

    x1 = max(0, min(W - 1, x))
    y1 = max(0, min(H - 1, y))
    x2 = max(0, min(W - 1, x + w))
    y2 = max(0, min(H - 1, y + h))

    # ---- 枠線 ----
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

    # ---- ラベル描画（必要なら）----
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
#  コア：バーコード検出（中央に最も近いものを選ぶ）
# =========================================================
def _select_barcode(
    decode_data: List[Any],
    frame,
    target_code: Optional[str] = None,
):
    """
    decode() の結果リストから 1つだけ選ぶヘルパー。

    優先順位:
      1) target_code が指定されていれば、それと一致するもの
      2) なければ画像中心に最も近いバーコード
    """
    if len(decode_data) == 0:
        return None

    # 1) ターゲット番号指定がある場合は、まず一致を探す
    if target_code is not None and target_code != "":
        for d in decode_data:
            code_str = d.data.decode("utf-8", errors="ignore")
            if code_str == str(target_code):
                return d

    # 2) 一致が見つからない or target_code なし → 画像中心に最も近いもの
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

import numpy as np  # まだ上の方に無ければ追加

def preprocess_for_barcode(gray: "cv2.Mat") -> "cv2.Mat":
    """
    バーコード認識用の前処理。
    入力: グレースケール画像
    出力: 前処理後のグレースケール or 2値画像
    """

    # 1. ノイズを軽く落とす
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. 局所コントラストを上げる（暗いところでもバーが見えるように）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(blur)

    # 3. 適応的二値化で明るさムラに対応
    binary = cv2.adaptiveThreshold(
        clahe_img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,   # or cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        cv2.THRESH_BINARY,
        25,   # blockSize（奇数）
        10    # C（閾値から引くオフセット）
    )

    # 4. 線が切れているのを繋ぐためのモルフォロジー処理
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return binary

# def preprocess_for_barcode(gray: "cv2.Mat") -> "cv2.Mat":
#     """
#     バーコード認識用の前処理。
#     入力: グレースケール画像
#     出力: 前処理後のグレースケール or 2値画像

#     追加:
#       - バーコードの垂直線を優先的につなぐための
#         縦長カーネルでのモルフォロジー処理を入れている。
#     """

#     # 1. ノイズを軽く落とす
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)

#     # 2. 局所コントラストを上げる（暗いところでもバーが見えるように）
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     clahe_img = clahe.apply(blur)

#     # 3. 適応的二値化で明るさムラに対応
#     binary = cv2.adaptiveThreshold(
#         clahe_img,
#         255,
#         cv2.ADAPTIVE_THRESH_MEAN_C,   # or cv2.ADAPTIVE_THRESH_GAUSSIAN_C
#         cv2.THRESH_BINARY,
#         25,   # blockSize（奇数）
#         10    # C（閾値から引くオフセット）
#     )

#     # 4. まず軽く全方向の穴埋め（小さなノイズ用）
#     kernel_small = np.ones((3, 3), np.uint8)
#     binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=1)

#     # 5. 縦線優先で「ぶつ切り」をつなぐ処理
#     #    - バーコードの縦線は「ほぼ垂直」とわかっているので、
#     #      縦に長いカーネルで縦方向だけ強めに閉じる。
#     #    - OpenCV のモルフォロジーは「白(255)」を前景とみなすので、
#     #      いったん反転してから処理し、最後に戻す。
#     inv = cv2.bitwise_not(binary)

#     # 高さ 15px, 幅 3px の縦長カーネル（ここは調整パラメータ）
#     vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
#     inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, vert_kernel, iterations=1)

#     # 反転して元の極性に戻す
#     binary = cv2.bitwise_not(inv)

#     # （必要なら）横方向ノイズを軽く削ることもできる
#     # horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
#     # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=1)

#     return binary

def barcode_detect_unified(
    img,
    barcode_number: Optional[str] = None,
    save_dir: Optional[Path] = None,   # ★ 追加
) -> Tuple[bool, Optional[Any], List[Any]]:
    """
    画像中のバーコードを検出する共通関数。

    barcode_number が指定されている場合:
      → その番号のバーコードを優先的に探す。
         見つからなければ、中央に最も近いバーコードを返す。
    指定されていない場合（None or ""）:
      → 画像に写っているバーコードのうち、
         画像中心に最も近いものを1つ返す。

    戻り値:
      detected     : bool（何か1つは選べたか）
      selected_one : 選択された pyzbar Decoded オブジェクト or None
      all_decoded  : 最後に decode した結果リスト（デバッグ用など）
    """

    if img is None:
        return False, None, []

    # --- カラー → グレースケール変換 ---
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # --- 前処理画像の生成 ---
    pre = preprocess_for_barcode(gray)

    # ★★★ 前処理結果の保存（指定があれば） ★★★
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 元のグレースケール
        save_image(gray, save_dir / "bookshelf_barcode_gray_proc.png")
        # 前処理後（CLAHE + 二値化 + モルフォロジー）の画像
        save_image(pre, save_dir / "bookshelf_barcode_preproc.png")

    # この2種類で順番に decode を試す
    candidates = [
        gray,  # まずは素のグレースケール
        pre,   # ダメなら前処理済み画像
    ]

    last_decoded: List[Any] = []
    selected = None

    for cand in candidates:
        decode_data = decode(cand)
        last_decoded = decode_data  # 最後に試したものを保存（デバッグ用）

        print(f"decode_data (len={len(decode_data)}) on candidate")

        if len(decode_data) == 0:
            continue  # 次の候補画像にトライ

        # 候補が見つかったら、ターゲット優先 + 中央に近いやつを選ぶ
        selected = _select_barcode(decode_data, img, target_code=barcode_number)
        if selected is not None:
            break

    if selected is None:
        return False, None, last_decoded

    return True, selected, last_decoded

# =========================================================
#  カメラ座標系 X 方向オフセット計算
# =========================================================
def compute_barcode_center_and_x_offset(
    barcode_data,
    frame,
    fx_px: float,
    depth_m: float,
) -> Dict[str, float]:
    """
    barcode_data: pyzbar の Decoded オブジェクト
    frame      : OpenCVのBGR画像（numpy配列）
    fx_px      : カメラの焦点距離[px]
    depth_m    : バーコードまでの奥行き[m]（仮定もOK）

    戻り値 dict:
      image_width, image_height
      u_center, v_center
      offset_u_px
      X_m   ← カメラ座標系 X[m]
    """

    H, W = frame.shape[:2]

    # 画像中心（ピクセル）
    u_c = W / 2.0
    v_c = H / 2.0

    # バーコードの外接矩形
    rect = barcode_data.rect  # Rect(left, top, width, height)

    # バーコード中心のピクセル座標
    u_b = rect.left + rect.width / 2.0
    v_b = rect.top + rect.height / 2.0

    # x方向のピクセルずれ（画像中心からの差）
    offset_u_px = u_b - u_c

    # カメラ座標系 X[m]（Zを depth_m と仮定して近似）
    #   X = (u - c_x) / f_x * Z
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
#  カメラで1枚撮って → バーコード検出 → Xオフセット
# =========================================================
def capture_barcode_and_x_offset(
    node,
    executor,
    shot_dir: Path,
    barcode_number: Optional[str] = None,
    fx_px: float = 2500.0,
    depth_m: float = 0.40,
) -> Tuple[bool, Optional[str], Optional[Dict[str, float]]]:
    

    """
    /navigation_goal が True になるのを待ってから:

      1. Depstechで1枚撮影
      2. バーコードを1つ選ぶ（target指定あり/なし両対応）
      3. カメラ座標系X方向のずれを計算
      4. /error_x (String) に 「バーコード番号/x誤差」を publish

    barcode_number:
      - None or "" の場合: 中央に最も近いバーコードを使用
      - 文字列の場合    : そのコードを優先的に探し、なければ中央に最も近いもの

    戻り値:
      detected   : bool（最終的に1つ選べたか）
      code_str   : 選んだバーコード文字列（見つからなければ None）
      info       : compute_barcode_center_and_x_offset の戻り値 dict（見つからなければ None）
    """

    shot_dir = Path(shot_dir)
    shot_dir.mkdir(parents=True, exist_ok=True)

    # ==============================
    # /navigation_goal を待つ
    # ==============================
    node.get_logger().info("[bookshelf barcode] Waiting /navigation_goal == True ...")
    watcher = NavigationGoalWatcher(node)

    ok = watcher.wait_until_true(executor, timeout_sec=None)
    watcher.destroy()

    if not ok:
        node.get_logger().warn(
            "[bookshelf barcode] /navigation_goal wait aborted → skip capture"
        )
        return False, None, None

    node.get_logger().info(
        "[bookshelf barcode] /navigation_goal received → start capture & localization"
    )

    # ==============================
    # 1枚撮影
    # ==============================
    frame = capture_one_depstech(shot_dir / "bookshelf_barcode_capture_selfloc.png")

    # ==============================
    # バーコード検出（target付き or なし）
    # ==============================
    detected, selected, all_decoded = barcode_detect_unified(
        frame,
        barcode_number,
        save_dir=shot_dir,   # ★ 追加：ここに処理後画像が保存される
    )

    if not detected or selected is None:
        node.get_logger().warn(
            "[capture_barcode_and_x_offset] no barcode selected."
        )
        return False, None, None

    # どのバーコードを使ったかログ
    code_str = selected.data.decode("utf-8", errors="ignore")
    node.get_logger().info(
        f"[capture_barcode_and_x_offset] selected barcode: {code_str}"
    )

    # ==============================
    # カメラ座標系Xずれの計算
    # ==============================
    info = compute_barcode_center_and_x_offset(
        barcode_data=selected,
        frame=frame,
        fx_px=fx_px,
        depth_m=depth_m,
    )
    x_error = float(info["X_m"])

    # デバッグ用に bbox 付き画像を保存
    vis = draw_bbox(frame, selected.rect, label=code_str)
    save_image(vis, shot_dir / "bookshelf_barcode_capture_selfloc_bbox.png")

    # ==============================
    # /error_x (String) publish
    #  フォーマット: "<barcode>/<x_error>"
    #  x_error は 将来 Float32 別トピックに差し替えやすいように分離
    # ==============================
    if not hasattr(capture_barcode_and_x_offset, "_error_x_pub"):
        capture_barcode_and_x_offset._error_x_pub = node.create_publisher(
            String,
            "/error_x",
            10,
        )
    error_x_pub = capture_barcode_and_x_offset._error_x_pub

    formatted_code = format_barcode_code(code_str)

    msg = String()
    msg.data = f"{formatted_code}/{x_error:.6f}"
    error_x_pub.publish(msg)

    node.get_logger().info(f"[error_x published] {msg.data}")

    # 将来的に x_error を Float32 で別トピックに出したくなったら、
    # ここに Float32 publisher を追加して publish するだけでOK。

    return True, code_str, info


# =========================================================
#  単体テスト用 main（呼び出し側の例）
# =========================================================
def main():
    """
    端末から実行すると:
      - ROS2 を初期化
      - テスト用ノードを立ち上げ
      - /navigation_goal に True を1回だけ publish
      - 1枚撮影して、選んだバーコードと X[m] を表示
    """

    try:
        print("=== Barcode Central-Select Test (ROS2) ===")

        # ========== ROS2 初期化 ==========
        rclpy.init()
        node = rclpy.create_node("barcode_central_select_test")
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        # 保存ディレクトリ
        shot_dir = Path(
            "/home/book/pro_book_SAM3/pro_hand_book_python/captures/bookshelf_barcode_test"
        )

        # ターゲットバーコードを指定したい場合はここで入力
        target = input("ターゲットバーコード番号（任意・空で自動選択）: ").strip()
        if target == "":
            target = None

        # 焦点距離と距離は仮値（あとでキャリブして更新）
        fx_px = 2500.0
        depth_m = 0.40

        # ========== /navigation_goal を自分で True にする ==========
        nav_pub = node.create_publisher(Bool, "/navigation_goal", 10)
        # 少し待ってから publish してもいいが、ここでは即 publish
        nav_msg = Bool()
        nav_msg.data = True
        nav_pub.publish(nav_msg)
        node.get_logger().info("[test main] published /navigation_goal=True")

        # ========== 実際の処理 ==========
        detected, code_str, info = capture_barcode_and_x_offset(
            node=node,
            executor=executor,
            shot_dir=shot_dir,
            barcode_number=target,
            fx_px=fx_px,
            depth_m=depth_m,
        )

        if not detected:
            print("バーコードを検出できませんでした。")
        else:
            print("\n=== 結果 ===")
            print("選択されたバーコード:", code_str)
            print(f"画像サイズ: W={info['image_width']}, H={info['image_height']}")
            print(
                f"バーコード中心ピクセル: u={info['u_center']:.1f}, v={info['v_center']:.1f}"
            )
            print(
                f"画像中心からのx方向ピクセルずれ: {info['offset_u_px']:.1f} [px]"
            )
            print(
                f"カメラ座標系X方向の位置 (Z={depth_m}m 仮定): {info['X_m']:.3f} [m]"
            )
            print("\n※ /error_x トピックにも 'コード/x誤差' を publish 済みです。")

    except Exception:
        print("エラーが発生しました")
        traceback.print_exc()

    finally:
        # 後片付け
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()