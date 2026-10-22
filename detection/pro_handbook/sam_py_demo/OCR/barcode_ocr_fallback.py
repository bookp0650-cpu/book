#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import re

import cv2


# ==========================================
# 既存PP-OCRv5
# ==========================================
OCR_DETECTION_MODEL_DIR = Path(
    "/home/book/.paddlex/official_models/"
    "PP-OCRv5_server_det"
)

OCR_RECOGNITION_MODEL_DIR = Path(
    "/home/book/.paddlex/official_models/"
    "PP-OCRv5_server_rec"
)

OCR_VISUALIZATION_FONT = Path(
    "/home/book/.paddlex/fonts/simfang.ttf"
)


# PaddleOCR import前に設定
os.environ[
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"
] = "True"

os.environ[
    "PADDLE_PDX_LOCAL_FONT_FILE_PATH"
] = str(
    OCR_VISUALIZATION_FONT
)


from paddleocr import PaddleOCR


def normalize_digits(
    text,
):
    """
    数字以外をすべて除去する。
    """

    return re.sub(
        r"[^0-9]",
        "",
        str(text),
    )


def collect_recognized_texts(
    obj,
):
    """
    PaddleOCR JSONから認識文字列を回収する。
    """

    out = []

    def walk(x):

        if isinstance(x, dict):

            for key, value in x.items():

                key_lower = str(
                    key
                ).lower()

                if (
                    key_lower
                    in {
                        "rec_texts",
                        "texts",
                    }
                    and isinstance(
                        value,
                        list,
                    )
                ):

                    for item in value:

                        if (
                            isinstance(
                                item,
                                str,
                            )
                            and item.strip()
                        ):
                            out.append(
                                item.strip()
                            )

                    continue

                if (
                    key_lower
                    in {
                        "rec_text",
                        "text",
                    }
                    and isinstance(
                        value,
                        str,
                    )
                ):

                    if value.strip():

                        out.append(
                            value.strip()
                        )

                    continue

                walk(
                    value
                )

        elif isinstance(
            x,
            list,
        ):

            for item in x:
                walk(
                    item
                )

    walk(
        obj
    )

    # 重複除去
    unique = []
    seen = set()

    for text in out:

        if text not in seen:

            seen.add(
                text
            )

            unique.append(
                text
            )

    return unique


def _iter_ocr_items(obj):
    """
    PaddleOCR の JSON から
    text / bbox / score を取り出す。
    """

    if obj is None:
        return

    if isinstance(obj, dict):
        text_keys = (
            "rec_texts",
            "texts",
            "text",
        )
        poly_keys = (
            "dt_polys",
            "dt_boxes",
            "boxes",
            "polys",
            "points",
            "bbox",
        )
        score_keys = (
            "rec_scores",
            "scores",
            "confidences",
        )

        rec_texts = None
        for k in text_keys:
            v = obj.get(k)
            if isinstance(v, list) and (
                len(v) == 0
                or isinstance(v[0], str)
            ):
                rec_texts = v
                break

        rec_polys = None
        for k in poly_keys:
            v = obj.get(k)
            if isinstance(v, list):
                if rec_texts is not None:
                    if len(v) == len(rec_texts):
                        rec_polys = v
                        break
                else:
                    rec_polys = v
                    break

        rec_scores = None
        for k in score_keys:
            v = obj.get(k)
            if isinstance(v, list):
                rec_scores = v
                break

        if rec_texts is not None and rec_polys is not None:
            for i, text in enumerate(rec_texts):
                score = None
                if (
                    rec_scores is not None
                    and i < len(rec_scores)
                ):
                    try:
                        score = float(rec_scores[i])
                    except Exception:
                        score = None

                yield {
                    "text": str(text),
                    "bbox": rec_polys[i],
                    "score": score,
                }

        # 再帰
        for v in obj.values():
            yield from _iter_ocr_items(v)

    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_ocr_items(item)


def _bbox_to_xyxy(bbox, image_shape):
    """
    bbox/poly を x1,y1,x2,y2 に変換する。
    """
    H, W = image_shape[:2]

    try:
        # 例: [[x,y],[x,y],[x,y],[x,y]]
        if (
            isinstance(bbox, list)
            and len(bbox) >= 1
            and isinstance(bbox[0], (list, tuple))
        ):
            xs = []
            ys = []

            for p in bbox:
                if len(p) >= 2:
                    xs.append(float(p[0]))
                    ys.append(float(p[1]))

            if xs and ys:
                x1 = max(0, min(W - 1, int(min(xs))))
                y1 = max(0, min(H - 1, int(min(ys))))
                x2 = max(0, min(W - 1, int(max(xs))))
                y2 = max(0, min(H - 1, int(max(ys))))

                if x2 > x1 and y2 > y1:
                    return x1, y1, x2, y2

        # 例: {"left":..,"top":..,"width":..,"height":..}
        if isinstance(bbox, dict):
            if all(
                k in bbox
                for k in (
                    "left",
                    "top",
                    "width",
                    "height",
                )
            ):
                x1 = int(bbox["left"])
                y1 = int(bbox["top"])
                x2 = x1 + int(bbox["width"])
                y2 = y1 + int(bbox["height"])

                x1 = max(0, min(W - 1, x1))
                y1 = max(0, min(H - 1, y1))
                x2 = max(0, min(W - 1, x2))
                y2 = max(0, min(H - 1, y2))

                if x2 > x1 and y2 > y1:
                    return x1, y1, x2, y2

    except Exception:
        pass

    return None


def _safe_name(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    return text[:40] if text else "empty"

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        type=Path,
    )

    parser.add_argument(
        "--query",
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    image_path = (
        args.image
        .expanduser()
        .resolve()
    )

    out_dir = (
        args.out_dir
        .expanduser()
        .resolve()
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==========================================
    # 入力画像を読み込み
    # ==========================================
    img = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if img is None:
        raise RuntimeError(
            f"cv2.imread failed: "
            f"{image_path}"
        )

    # ==========================================
    # OCR前に180度回転
    # ==========================================
    ocr_img = cv2.rotate(
        img,
        cv2.ROTATE_180,
    )

    # ==========================================
    # 実際にOCRへ渡す180度回転画像を保存
    # ==========================================
    rotated_input_path = (
        out_dir
        / "ocr_input_rot180.png"
    )

    ok = cv2.imwrite(
        str(rotated_input_path),
        ocr_img,
    )

    if not ok:
        raise RuntimeError(
            f"cv2.imwrite failed: "
            f"{rotated_input_path}"
        )

    print(
        "[BARCODE OCR] "
        "input rotated 180 deg"
    )

    # ==========================================
    # 検索対象バーコード番号
    # ==========================================
    target_digits = normalize_digits(
        args.query
    )

    print(
        "[BARCODE OCR] "
        f"target_digits={target_digits}"
    )

    # ==========================================
    # OCRモデル
    # ==========================================
    ocr = PaddleOCR(
        text_detection_model_dir=str(
            OCR_DETECTION_MODEL_DIR
        ),
        text_recognition_model_dir=str(
            OCR_RECOGNITION_MODEL_DIR
        ),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    # ==========================================
    # OCRはこの1回だけ
    # ==========================================
    result = ocr.predict(
        ocr_img
    )
    texts = []

    raw_json_path = (
        out_dir
        / "ocr_raw.json"
    )

    overlay_path = (
        out_dir
        / "ocr_overlay.png"
    )

    # ==========================================
    # OCR可視化結果の保存先
    # ==========================================
    ocr_region_dir = (
        out_dir
        / "ocr_regions"
    )

    labeled_path = (
        out_dir
        / "ocr_regions_labeled.png"
    )

    regions_json_path = (
        out_dir
        / "ocr_regions.json"
    )

    if result:
        res0 = result[0]

        res0.save_to_json(
            str(
                raw_json_path
            )
        )

        res0.save_to_img(
            str(
                overlay_path
            )
        )

        try:

            raw_json = json.loads(
                raw_json_path.read_text(
                    encoding="utf-8"
                )
            )

            texts = (
                collect_recognized_texts(
                    raw_json
                )
            )

            # ==========================================
            # OCR領域の可視化と各領域の切り出し保存
            # ==========================================
            labeled_vis = ocr_img.copy()

            ocr_region_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            ocr_regions_info = []

            for idx, item in enumerate(
                _iter_ocr_items(raw_json),
                start=1,
            ):
                text = str(
                    item.get("text", "")
                ).strip()

                bbox = item.get("bbox")
                score = item.get("score", None)

                xyxy = _bbox_to_xyxy(
                    bbox,
                    ocr_img.shape,
                )
                if xyxy is None:
                    continue

                x1, y1, x2, y2 = xyxy

                # 全体可視化画像
                cv2.rectangle(
                    labeled_vis,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                label = (
                    f"{idx}: {text}"
                    if text
                    else f"{idx}: <empty>"
                )

                cv2.putText(
                    labeled_vis,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                # 各領域切り出し
                crop = ocr_img[
                    y1:y2,
                    x1:x2,
                ].copy()

                crop_path = (
                    ocr_region_dir
                    / f"region_{idx:02d}_{_safe_name(text)}.png"
                )

                if crop.size > 0:
                    cv2.imwrite(
                        str(crop_path),
                        crop,
                    )

                ocr_regions_info.append(
                    {
                        "index": idx,
                        "text": text,
                        "score": score,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "crop_path": str(crop_path),
                    }
                )



            cv2.imwrite(
                str(labeled_path),
                labeled_vis,
            )


            regions_json_path.write_text(
                json.dumps(
                    ocr_regions_info,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )


        except Exception as e:

            print(
                "[BARCODE OCR] "
                "JSON parse error:",
                repr(e),
            )

    print("")
    print(
        "===== BARCODE OCR TEXTS ====="
    )

    matched = False
    matched_text = None
    matched_digits = None

    for text in texts:

        digits = normalize_digits(
            text
        )

        print(
            f"text={text!r} "
            f"digits={digits!r}"
        )

        # ======================================
        # クエリ数字列がOCR結果に含まれていれば成功
        # ======================================
        if (
            target_digits
            and target_digits in digits
        ):

            matched = True
            matched_text = text
            matched_digits = digits

            print(
                ">>> BARCODE OCR SUCCESS <<<"
            )

            break

    payload = {
        "matched": bool(matched),
        "target": str(args.query),
        "target_digits": target_digits,
        "matched_text": matched_text,
        "matched_digits": matched_digits,
        "texts": texts,
        "input_image": str(image_path),
        "method": "ocr",
        "ocr_overlay_path": str(overlay_path),
        "ocr_regions_labeled_path": str(labeled_path),
        "ocr_regions_json_path": str(regions_json_path),
        "ocr_region_dir": str(ocr_region_dir),
    }

    result_path = (
        out_dir
        / "ocr_match.json"
    )

    result_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "[BARCODE OCR] "
        f"saved={result_path}"
    )


if __name__ == "__main__":
    main()
