import cv2
from pyzbar.pyzbar import decode
from pathlib import Path
from .rs_435i_rgb_only import Deproject
from pathlib import Path
from datetime import datetime
import traceback
from detection.pro_handbook.sam_py_demo.bar_code.web_camera_capture import capture_one_depstech
import rclpy
from std_msgs.msg import Bool, String
import time

class NavigationGoalWatcher:
    """
    /navigation_goal (std_msgs/Bool) を監視するヘルパークラス。
    True が一度来たら received=True にする。
    """
    def __init__(self, node: rclpy.node.Node):
        self._node = node
        self._received = False
        self._last_value = False

        # /navigation_goal を購読
        self._sub = node.create_subscription(
            Bool,
            "/navigation_goal",
            self._callback,
            10,
        )

    def _callback(self, msg: Bool):
        self._last_value = bool(msg.data)
        if self._last_value:
            self._received = True
            self._node.get_logger().info(
                "[NavigationGoalWatcher] /navigation_goal True received."
            )

    def is_received(self) -> bool:
        return self._received

    def wait_until_true(self, executor, timeout_sec: float | None = None) -> bool:
        """
        executor.spin_once() を使って /navigation_goal==True を待つ。
        timeout_sec を None にすると「無限待ち」。
        戻り値 True: True を受信して終了
               False: timeout などで終了
        """
        start_t = time.time()

        while rclpy.ok() and not self._received:
            executor.spin_once(timeout_sec=0.1)

            if timeout_sec is not None:
                if time.time() - start_t > timeout_sec:
                    self._node.get_logger().warn(
                        "[NavigationGoalWatcher] timeout while waiting /navigation_goal."
                    )
                    return False
        return self._received
    
#camera_bookshelf_barcode = Deproject()
def save_image(img_bgr, save_path: str | Path) -> None:
    """
    img_bgr: OpenCV画像(BGR)
    save_path: 保存先パス（親ディレクトリが無ければ作る）
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(save_path), img_bgr)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {save_path}")

# 例: bboxを描画して保存
# data = (x, y, w, h

def draw_bbox(img, data, color=(0, 255, 0), thickness=2, label=None):
    """
    img: 入力画像 (BGR, cv2画像)
    data: (x, y, w, h) もしくは {"x":..,"y":..,"w":..,"h":..} / {"left":..,"top":..,"width":..,"height":..}
          ここで x,y は左上座標、w,h は幅/高さ
    color: 枠の色 (BGR)
    thickness: 線の太さ
    label: 任意の文字列（枠の左上に表示）
    return: 描画後の画像（コピー）
    """
    out = img.copy()

    # ---- data の取り出し（タプル/リスト or dict どちらも対応）----
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

    # ---- 数値化 & 画像範囲にクリップ ----
    x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
    H, W = out.shape[:2]

    x1 = max(0, min(W - 1, x))
    y1 = max(0, min(H - 1, y))
    x2 = max(0, min(W - 1, x + w))
    y2 = max(0, min(H - 1, y + h))

    # ---- 描画 ----
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.0
        t = max(1, thickness)
        (tw, th), _ = cv2.getTextSize(label, font, fs, t)

        # ラベル背景（枠の上に出したいが、はみ出るなら下へ）
        by1 = y1 - th - 6
        if by1 < 0:
            by1 = y1 + 2
        bx1 = x1
        bx2 = min(W - 1, x1 + tw + 6)
        by2 = min(H - 1, by1 + th + 6)

        cv2.rectangle(out, (bx1, by1), (bx2, by2), color, -1)
        cv2.putText(out, label, (bx1 + 3, by2 - 3), font, fs, (0, 0, 0), t, cv2.LINE_AA)

    return out

def barcode_inference(barcode_number, image_path): #写真読み込む&バーコード認識をする関数

    #img_path = Path(shot_dir) / "before_init_rgb.png"
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    # 读取图像　写真を読み取る(画像処理)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast_img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    blur = cv2.GaussianBlur(contrast_img, (3, 3), 0)
    sharpened = cv2.addWeighted(blur, 1.5, cv2.GaussianBlur(img, (0, 0), 3), -0.5, 0)
    _, binary = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #clahe_img = clahe.apply(img)
    #laplacian_raw = cv2.Laplacian(img, cv2.CV_64F)
    #laplacian = cv2.convertScaleAbs(laplacian_raw)
    #sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    #sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    #sobel_combined = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))
    #canny = cv2.Canny(img, 100, 200)
    _path = "/home/book/pro_book_SAM3/pro_hand_book_python/captures/20260119_134439/bbox.png"
    # 解码图像　写真を解析（元コード同様 binary を使う）
    decode_data = decode(img)
    print("decode_data list", decode_data)
    data = decode_data[0]#TODO とりあえず
    rect = data.rect
    vis = draw_bbox(img, rect, label="bbox")
    save_image(vis, _path)
    # 入力の barcode_number と一致するものがあれば True
    for barcode_data in decode_data:
        bar_code_script = barcode_data.data.decode("utf-8")
        if bar_code_script == str(barcode_number):
            rect = barcode_data.rect
            left_point = barcode_data.rect.left
            right_point = left_point + barcode_data.rect.width
            mid_point = (left_point + right_point) / 2
            print("mid_point:", mid_point)
            return True, rect
        pass 
    print("no barcode matched")   
    return False, None

def barcode_perception(barcode_number, img): #写真読み込む&バーコード認識をする関数

    #img_path = Path(shot_dir) / "before_init_rgb.png"
    #img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    #if img is None:
    #    return False

    # 读取图像　写真を読み取る(画像処理)
    contrast_img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    blur = cv2.GaussianBlur(contrast_img, (3, 3), 0)
    sharpened = cv2.addWeighted(blur, 1.5, cv2.GaussianBlur(img, (0, 0), 3), -0.5, 0)
    _, binary = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #clahe_img = clahe.apply(img)
    #laplacian_raw = cv2.Laplacian(img, cv2.CV_64F)
    #laplacian = cv2.convertScaleAbs(laplacian_raw)
    #sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    #sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    #sobel_combined = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))
    #canny = cv2.Canny(img, 100, 200)

    # 解码图像　写真を解析（元コード同様 binary を使う）
    decode_data = decode(img)
    print("decode_data list", decode_data)
    # 入力の barcode_number と一致するものがあれば True
    for barcode_data in decode_data:
        bar_code_script = barcode_data.data.decode("utf-8")
        if bar_code_script == str(barcode_number):
            return True, barcode_data
        pass    
    return False, decode_data
# def detect_barcode(self, barcode_number, shot_dir):
#     camera_bookshelf_barcode = Deproject()
#     image_path = camera_bookshelf_barcode.get_frames()
#     bar_code_number = input("barcode number?")
#     self.color_frame

# if __name__ == '__main__':
#     image_path = Path("/home/book/pro_book_SAM3/pro_hand_book_python/captures/20260119_134439/book_barcode_capture_20260119_134520.png")
#     #image_path = "/home/book/pro_book_SAM3/pro_hand_book_python/captures/20251224_155932/before_init_rgb.png"
#     bar_code = "15 14"
#     barcode_inference(bar_code, image_path)

def compute_barcode_center_and_x_offset(
    barcode_data,
    frame,
    fx_px: float,
    depth_m: float,
):
    """
    barcode_data: pyzbar の Decoded オブジェクト
    frame      : OpenCVのBGR画像（numpy配列）
    fx_px      : カメラの焦点距離[px]（とりあえず仮値でOK）
    depth_m    : バーコードまでの奥行き[m]（ここでは 0.4 など仮定）
    """

    # 画像サイズ
    H, W = frame.shape[:2]

    # 画像中心（ピクセル）
    u_c = W / 2.0
    v_c = H / 2.0

    # バーコードの外接矩形
    rect = barcode_data.rect  # Rect(left, top, width, height)

    # バーコード中心のピクセル座標
    u_b = rect.left + rect.width  / 2.0
    v_b = rect.top  + rect.height / 2.0

    # x方向のピクセルずれ（画像中心からの差）
    offset_u_px = u_b - u_c

    # カメラ座標系 X[m]（Zを depth_m と仮定して近似）
    #   X = (u - c_x) / f_x * Z
    X_m = (offset_u_px / fx_px) * depth_m

    return {
        "image_width":  W,
        "image_height": H,
        "u_center": u_b,
        "v_center": v_b,
        "offset_u_px": offset_u_px,
        "X_m": X_m,
    }

def capture_barcode_and_x_offset(
    barcode_number: str,
    shot_dir: Path,
    node,              # 追加: rclpy.Node
    executor,          # 追加: executor (MultiThreadedExecutor など)
    fx_px: float = 2500.0,
    depth_m: float = 0.40,
):
    """
    navigation_goal=True を受信してから:
      Depstechで1枚撮影 → 指定バーコードを探す → 
      カメラ座標系X方向のずれを計算し、/error_x に publish する関数。

    戻り値:
      detected   : bool（指定バーコードを見つけたか）
      code_str   : 見つけたバーコード文字列（見つからなければ None）
      info       : compute_barcode_center_and_x_offset の戻り値 dict（見つからなければ None）
    """

    # ==============================
    # 1. /navigation_goal を待つ
    # ==============================
    node.get_logger().info(
        "[capture_barcode_and_x_offset] Waiting /navigation_goal..."
    )
    watcher = NavigationGoalWatcher(node)
    ok = watcher.wait_until_true(executor, timeout_sec=None)

    if not ok:
        node.get_logger().warn(
            "[capture_barcode_and_x_offset] /navigation_goal wait aborted."
        )
        return False, None, None

    node.get_logger().info(
        "[capture_barcode_and_x_offset] /navigation_goal received -> start capture."
    )

    # ==============================
    # 2. 1枚撮影
    # ==============================
    frame = capture_one_depstech(shot_dir / "bookshelf_barcode_capture_selfloc.png")

    # ==============================
    # 3. バーコード認識
    # ==============================
    detected, barcode_data = barcode_perception(barcode_number, frame)

    if not detected:
        node.get_logger().warn(
            "[capture_barcode_and_x_offset] target barcode NOT found."
        )
        return False, None, None

    # 読み取ったバーコード文字列
    code_str = barcode_data.data.decode("utf-8", errors="ignore")

    # ==============================
    # 4. カメラ座標系Xずれの計算
    # ==============================
    info = compute_barcode_center_and_x_offset(
        barcode_data=barcode_data,
        frame=frame,
        fx_px=fx_px,
        depth_m=depth_m,
    )
    x_error = float(info["X_m"])

    # ==============================
    # 5. /error_x に publish（String 型）
    #    形式: "バーコード番号/xの誤差"
    #    ※あとで Float32 に変更しやすいように、
    #      Publisher はこの関数の static 属性にしておく
    # ==============================
    if not hasattr(capture_barcode_and_x_offset, "_error_x_pub"):
        capture_barcode_and_x_offset._error_x_pub = node.create_publisher(
            String,
            "/error_x",
            10,
        )
    error_x_pub = capture_barcode_and_x_offset._error_x_pub

    msg = String()
    msg.data = f"{code_str}/{x_error:.6f}"

    error_x_pub.publish(msg)
    node.get_logger().info(
        f"[capture_barcode_and_x_offset] /error_x published: {msg.data}"
    )

    # 将来的に Float32 で誤差だけ publish したくなったら、
    # ここに Float32 Publisher を追加すればOK。

    return True, code_str, info

def main():
    try:
        print("=== Bookshelf Barcode Test (capture mode) ===")

        # 1. 読み取りたいバーコード番号を入力
        barcode_number = input("読み取りたいバーコード番号を入力してください: ").strip()

        # 2. 撮影した画像の保存ディレクトリ
        shot_dir = Path("/home/book/pro_book_SAM3/pro_hand_book_python/captures/bookshelf_barcode_test")
        shot_dir.mkdir(parents=True, exist_ok=True)

        # 3. Depstech で1枚撮影
        print("カメラ撮影中...")
        frame = capture_one_depstech(shot_dir / "bookshelf_barcode_capture.png")
        print("撮影完了")

        # 4. バーコード解析
        print("バーコード解析中...")
        detected, barcode_data = barcode_perception(barcode_number, frame)

        print("=== 結果 ===")
        print("検出成功:", detected)

        if not detected:
            # barcode_data には「検出されたバーコード一覧」が入っている
            print("指定バーコードは見つかりませんでした。")
            print("検出されたバーコード一覧:")
            for d in barcode_data:
                print(" -", d.data.decode("utf-8", errors="ignore"))
            return

        # ここから先は「指定バーコードが見つかった」場合
        print("一致したバーコード情報:")
        print("データ:", barcode_data.data.decode("utf-8", errors="ignore"))
        print("タイプ:", barcode_data.type)
        print("位置(rect):", barcode_data.rect)

        # 5. バーコード中心と x方向のずれを計算
        #   fx_px と depth_m は仮値（あとでちゃんとキャリブすればOK）
        fx_px   = 2500.0   # 仮の焦点距離[px]
        depth_m = 0.40     # バーコードまでの奥行き[m] を 40cm と仮定

        info = compute_barcode_center_and_x_offset(
            barcode_data=barcode_data,
            frame=frame,
            fx_px=fx_px,
            depth_m=depth_m,
        )

        print("\n--- バーコード中心と横ずれ ---")
        print(f"画像サイズ: W={info['image_width']}, H={info['image_height']}")
        print(f"バーコード中心ピクセル: u={info['u_center']:.1f}, v={info['v_center']:.1f}")
        print(f"画像中心からのx方向ピクセルずれ: {info['offset_u_px']:.1f} [px]")
        print(f"カメラ座標系X方向の位置 (Z={depth_m}m 仮定): {info['X_m']:.3f} [m]")

    except Exception:
        print("エラーが発生しました")
        traceback.print_exc()


if __name__ == "__main__":
    main()