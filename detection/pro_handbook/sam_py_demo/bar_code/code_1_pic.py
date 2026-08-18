import cv2
from pyzbar.pyzbar import decode
from pathlib import Path
from .rs_435i_rgb_only import Deproject
from detection.pro_handbook.sam_py_demo.bar_code.web_camera_capture import capture_one_depstech

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
    _path = "/home/book/pro_book/pro_hand_book_python/captures/20260119_134439/bbox.png"
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

import cv2
from pyzbar.pyzbar import decode


def barcode_perception(barcode_number, img):
    """
    画像からバーコードを検出し、指定番号と一致するか確認する。

    Returns
    -------
    tuple[str, object | list | None]

    status:
        "success"
            指定されたバーコードと一致

        "wrong_barcode"
            バーコードは検出されたが、指定番号とは不一致

        "no_barcode"
            バーコード自体が1件も検出されなかった

        "error"
            画像またはデコード処理でエラー

    data:
        success:
            一致したDecodedオブジェクト

        wrong_barcode:
            検出されたDecodedオブジェクトのリスト

        no_barcode / error:
            None
    """

    # =====================================
    # 入力画像確認
    # =====================================
    if img is None:
        print("[ERROR] barcode_perception(): img is None")
        return "error", None

    if not hasattr(img, "size") or img.size == 0:
        print("[ERROR] barcode_perception(): img is empty")
        return "error", None

    target = str(barcode_number).strip()

    # =====================================
    # バーコード検出
    # =====================================
    try:
        blur = cv2.GaussianBlur(
            img,
            (3, 3),
            0,
        )

        decode_data = decode(blur)

    except Exception as e:
        print(
            "[ERROR] Barcode decode failed:",
            repr(e),
        )
        return "error", None

    print("\n===== BARCODE CHECK =====")
    print("TARGET       :", repr(target))
    print("NUM DETECTED :", len(decode_data))
    print("-------------------------")

    # =====================================
    # バーコードが1件も検出されなかった
    # → 右面確認へ進むため no_barcode
    # =====================================
    if not decode_data:
        print(">>> BARCODE NOT DETECTED <<<\n")
        return "no_barcode", None

    # 正常に文字列化できたバーコード数
    valid_decode_count = 0

    # =====================================
    # 1件以上のバーコードを確認
    # =====================================
    for i, barcode_data in enumerate(decode_data):
        try:
            raw_data = barcode_data.data

            if isinstance(raw_data, (bytes, bytearray)):
                detected = raw_data.decode(
                    "utf-8",
                    errors="replace",
                ).strip()
            else:
                detected = str(raw_data).strip()

            valid_decode_count += 1

        except Exception as e:
            print(
                f"[{i}] DECODE ERROR:",
                repr(e),
            )
            print("-------------------------")
            continue

        match = detected == target

        print(f"[{i}] DETECTED :", repr(detected))
        print(f"[{i}] MATCH    :", match)
        print(
            f"[{i}] TYPE     :",
            getattr(barcode_data, "type", None),
        )
        print(
            f"[{i}] POS      :",
            getattr(barcode_data, "rect", None),
        )
        print("-------------------------")

        # =================================
        # 指定バーコードと一致
        # =================================
        if match:
            print(">>> BARCODE SUCCESS <<<\n")
            return "success", barcode_data

    # decode_dataは存在したが、
    # 全データの処理に失敗した場合
    if valid_decode_count == 0:
        print(
            ">>> BARCODE DATA PROCESSING ERROR <<<\n"
        )
        return "error", None

    # =====================================
    # バーコードは検出したが番号が違う
    # =====================================
    print(
        ">>> BARCODE DETECTED, "
        "BUT WRONG BARCODE <<<\n"
    )

    return "wrong_barcode", decode_data

if __name__ == '__main__':
    image_path = Path("/home/book/pro_book/pro_hand_book_python/captures/20260119_134439/book_barcode_capture_20260119_134520.png")
    #image_path = "/home/book/pro_book/pro_hand_book_python/captures/20251224_155932/before_init_rgb.png"
    bar_code = "15 14"
    barcode_inference(bar_code, image_path)