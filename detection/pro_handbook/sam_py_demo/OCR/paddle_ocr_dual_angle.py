#!/usr/bin/env python3
from __future__ import annotations

"""
paddle_ocr_dual_angle.py

既存の start_ocr_subprocess() を変更せず、
_resolve_ocr_subprocess_paths() の ocr_script だけこのファイルへ差し替えて使う。

入力:
    python paddle_ocr_dual_angle.py <shot_dir>

処理:
    after_init_rgb.png
      -> 既存仕様の CW90
      -> +30 deg OCR
      -> -30 deg OCR
      -> それぞれのOCR polygonを「CW90だけの座標系」へ逆変換
      -> 近い重複候補を統合
      -> 従来互換の ocr_result.json を出力

重要:
    後段の match_text_to_mask_main(... forced_angle=90) は変更不要。
    SAM3 mask は元画像座標、OCR polygon は従来と同じ CW90 座標へ戻してあるため、
    既存の SAM3/OCR 対応付けを維持する。
"""

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 同じOCRディレクトリにある既存ファイルを利用
import paddle_ocr_test as base_ocr


EXTRA_ANGLES_DEG = (0.0, +30.0, -30.0)

# 同じ文字行を 0 / +30 / -30 で複数回拾った場合の重複除去
DUPLICATE_AABB_IOU_THRESHOLD = 0.6
DUPLICATE_CENTER_DISTANCE_PX = 6.0

def save_json(path: Path, obj) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def rotate_bound_with_matrix(
    image: np.ndarray,
    angle_deg: float,
    border_value=(255, 255, 255),
):
    """
    画像を欠けないように回転。
    戻り値:
        rotated
        M: 入力画像座標 -> rotated画像座標 の 2x3 affine
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(
        center,
        float(angle_deg),
        1.0,
    ).astype(np.float64)

    c = abs(M[0, 0])
    s = abs(M[0, 1])

    new_w = int(math.ceil(h * s + w * c))
    new_h = int(math.ceil(h * c + w * s))

    M[0, 2] += new_w / 2.0 - center[0]
    M[1, 2] += new_h / 2.0 - center[1]

    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return rotated, M


def transform_points(points, affine_2x3):
    arr = np.asarray(points, dtype=np.float64)

    if arr.ndim < 2 or arr.shape[-1] != 2:
        raise ValueError(
            f"polygon shape must end in 2 coordinates: {arr.shape}"
        )

    original_shape = arr.shape
    flat = arr.reshape(-1, 2)

    hom = np.concatenate(
        [
            flat,
            np.ones((flat.shape[0], 1), dtype=np.float64),
        ],
        axis=1,
    )

    out = hom @ np.asarray(
        affine_2x3,
        dtype=np.float64,
    ).T

    return out.reshape(original_shape).tolist()


def find_ocr_arrays_container(obj):
    """
    PaddleOCR/PaddleXのsave_to_json形式差を吸収。
    dt_polys と rec_texts を同じdictに持つ場所を再帰探索する。
    """
    if isinstance(obj, dict):
        dt_polys = obj.get("dt_polys")
        rec_texts = obj.get("rec_texts")

        if (
            isinstance(dt_polys, list)
            and isinstance(rec_texts, list)
        ):
            return obj

        for value in obj.values():
            found = find_ocr_arrays_container(value)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for value in obj:
            found = find_ocr_arrays_container(value)
            if found is not None:
                return found

    return None


def polygon_aabb(poly):
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)

    return np.array(
        [
            pts[:, 0].min(),
            pts[:, 1].min(),
            pts[:, 0].max(),
            pts[:, 1].max(),
        ],
        dtype=np.float64,
    )


def aabb_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 1e-9:
        return 0.0

    return float(inter / union)


def polygon_center(poly):
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    return pts.mean(axis=0)


def normalize_text(text):
    return "".join(
        str(text)
        .replace("　", "")
        .replace(" ", "")
        .split()
    ).lower()


def candidate_quality(candidate):
    """
    同じ位置の候補が重複した場合にどちらを残すか。
    基本はOCR confidence優先。
    confidenceが近い場合は長く読めた文字列を優先。
    """
    score = candidate.get("score")
    if score is None:
        score = 0.0

    text_len = len(
        normalize_text(candidate.get("text", ""))
    )

    return (
        float(score),
        int(text_len),
    )


def is_duplicate_candidate(a, b):
    a_box = polygon_aabb(a["poly"])
    b_box = polygon_aabb(b["poly"])

    iou = aabb_iou(a_box, b_box)

    a_center = polygon_center(a["poly"])
    b_center = polygon_center(b["poly"])

    center_distance = float(
        np.linalg.norm(a_center - b_center)
    )

    # 同一文字列なら少し緩く重複扱い
    same_text = (
        normalize_text(a["text"])
        == normalize_text(b["text"])
        and normalize_text(a["text"]) != ""
    )

    if iou >= DUPLICATE_AABB_IOU_THRESHOLD:
        return True

    if (
        same_text
        and center_distance
        <= DUPLICATE_CENTER_DISTANCE_PX * 2.0
    ):
        return True

    if center_distance <= DUPLICATE_CENTER_DISTANCE_PX:
        return True

    return False


def merge_candidates(candidates):
    """
    0 / +30 / -30 の候補を座標ベースで重複除去。
    別の文字行はそのまま両方残す。
    """
    merged = []

    for cand in candidates:
        duplicate_index = None

        for i, old in enumerate(merged):
            if is_duplicate_candidate(cand, old):
                duplicate_index = i
                break

        if duplicate_index is None:
            merged.append(cand)
            continue

        old = merged[duplicate_index]

        if candidate_quality(cand) > candidate_quality(old):
            merged[duplicate_index] = cand

    return merged


def run_one_angle(
    ocr,
    base_img,
    shot_dir: Path,
    extra_angle_deg: float,
):
    """
    1角度分OCRし、polygonをCW90基準座標へ戻して候補listを返す。
    """
    rotated, base_to_rotated = rotate_bound_with_matrix(
        base_img,
        extra_angle_deg,
    )

    rotated_to_base = cv2.invertAffineTransform(
        base_to_rotated
    )

    # 角度ごとに必ず別ファイル名にする。
    # 旧実装では 0 deg も "m030" になり、-30 deg のログで上書きされていた。
    if abs(float(extra_angle_deg)) < 1e-9:
        tag = "0deg"
    elif extra_angle_deg > 0:
        tag = "p030"
    else:
        tag = "m030"

    input_path = shot_dir / f"ocr_input_{tag}.png"
    tmp_json_path = shot_dir / f"ocr_result_{tag}.json"
    overlay_path = shot_dir / f"ocr_overlay_{tag}.png"

    # デバッグ確認用。不要になったら消してもよい。
    cv2.imwrite(
        str(input_path),
        rotated,
    )

    print(
        f"[OCR DUAL] predict extra={extra_angle_deg:+.1f} deg",
        flush=True,
    )

    t0 = time.perf_counter()
    result = ocr.predict(rotated)
    elapsed = time.perf_counter() - t0

    if not result:
        raise RuntimeError(
            f"PaddleOCR returned no result: angle={extra_angle_deg}"
        )

    res0 = result[0]

    res0.save_to_json(
        str(tmp_json_path)
    )

    res0.save_to_img(
        str(overlay_path)
    )

    obj = json.loads(
        tmp_json_path.read_text(
            encoding="utf-8"
        )
    )

    container = find_ocr_arrays_container(obj)

    if container is None:
        raise RuntimeError(
            f"dt_polys/rec_texts not found in {tmp_json_path}"
        )

    dt_polys = container.get(
        "dt_polys",
        [],
    )

    rec_texts = container.get(
        "rec_texts",
        [],
    )

    rec_scores = container.get(
        "rec_scores",
        [],
    )

    if len(dt_polys) != len(rec_texts):
        raise RuntimeError(
            "OCR JSON array length mismatch: "
            f"dt_polys={len(dt_polys)}, "
            f"rec_texts={len(rec_texts)}"
        )

    candidates = []

    for i, (poly, text) in enumerate(
        zip(dt_polys, rec_texts)
    ):
        corrected_poly = transform_points(
            poly,
            rotated_to_base,
        )

        score = None

        if (
            isinstance(rec_scores, list)
            and i < len(rec_scores)
        ):
            try:
                score = float(rec_scores[i])
            except Exception:
                score = None

        candidates.append(
            {
                "poly": corrected_poly,
                "text": str(text),
                "score": score,
                "source_extra_angle_deg": float(
                    extra_angle_deg
                ),
            }
        )

    print(
        f"[OCR DUAL] angle={extra_angle_deg:+.1f} "
        f"detected={len(candidates)} "
        f"time={elapsed:.3f} sec",
        flush=True,
    )

    return candidates, elapsed


def candidates_to_legacy_json(
    candidates,
    *,
    shot_dir: Path,
    runtimes,
):
    """
    既存の load_text_data() / _iter_ocr_items() がそのまま読める
    root-level parallel arraysを生成。
    """
    dt_polys = []
    rec_texts = []
    rec_scores = []
    rec_boxes = []
    source_angles = []

    for cand in candidates:
        poly = np.asarray(
            cand["poly"],
            dtype=np.float64,
        ).reshape(-1, 2)

        dt_polys.append(
            poly.tolist()
        )

        rec_texts.append(
            str(cand["text"])
        )

        score = cand.get("score")
        rec_scores.append(
            0.0
            if score is None
            else float(score)
        )

        rec_boxes.append(
            [
                float(poly[:, 0].min()),
                float(poly[:, 1].min()),
                float(poly[:, 0].max()),
                float(poly[:, 1].max()),
            ]
        )

        source_angles.append(
            float(
                cand[
                    "source_extra_angle_deg"
                ]
            )
        )

    return {
        # 既存コードが直接読む主要フィールド
        "dt_polys": dt_polys,
        "rec_texts": rec_texts,
        "rec_scores": rec_scores,

        # 後段debug/互換用
        "rec_polys": dt_polys,
        "rec_boxes": rec_boxes,

        # 追加情報。既存処理は無視する。
        "dual_angle_ocr": {
            "base_rotation": "CW90",
            "extra_angles_deg": [
                float(v)
                for v in EXTRA_ANGLES_DEG
            ],
            "polygon_coordinate_frame": (
                "CW90_base_frame"
            ),
            "source_extra_angle_deg": (
                source_angles
            ),
            "candidate_count": int(
                len(candidates)
            ),
            "predict_seconds": {
                str(angle): float(sec)
                for angle, sec in runtimes.items()
            },
            "shot_dir": str(shot_dir),
        },
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python paddle_ocr_dual_angle.py <shot_dir>"
        )

    shot_dir = Path(
        sys.argv[1]
    ).expanduser().resolve()

    img_path = (
        shot_dir
        / "after_init_rgb.png"
    )

    original = cv2.imread(
        str(img_path),
        cv2.IMREAD_COLOR,
    )

    if original is None:
        raise FileNotFoundError(
            img_path
        )

    # 既存仕様:
    # OCRはまず時計回り90度
    base_img = base_ocr.rotate270_cw(
        original
    )

    # 既存モデル生成/キャッシュ機構をそのまま使用
    ocr = base_ocr.get_ocr_model()

    all_candidates = []
    runtimes = {}

    for angle in EXTRA_ANGLES_DEG:
        candidates, elapsed = run_one_angle(
            ocr,
            base_img,
            shot_dir,
            angle,
        )

        all_candidates.extend(
            candidates
        )

        runtimes[angle] = elapsed

    merged_candidates = merge_candidates(
        all_candidates
    )

    final_json = candidates_to_legacy_json(
        merged_candidates,
        shot_dir=shot_dir,
        runtimes=runtimes,
    )

    final_path = (
        shot_dir
        / "ocr_result.json"
    )

    save_json(
        final_path,
        final_json,
    )

    save_json(
        shot_dir
        / "ocr_runtime_info.json",
        {
            "dual_angle": True,
            "base_rotation": "CW90",
            "extra_angles_deg": [
                float(v)
                for v in EXTRA_ANGLES_DEG
            ],
            "raw_candidate_count": int(
                len(all_candidates)
            ),
            "merged_candidate_count": int(
                len(merged_candidates)
            ),
            "predict_seconds": {
                str(angle): float(sec)
                for angle, sec in runtimes.items()
            },
            "ocr_result_json": str(
                final_path
            ),
            "sam_compatibility": (
                "OCR polygons are inverse-transformed "
                "to the legacy CW90 frame. "
                "Existing forced_angle=90 matching remains unchanged."
            ),
        },
    )

    print(
        "[OCR DUAL] done: "
        f"raw={len(all_candidates)}, "
        f"merged={len(merged_candidates)}",
        flush=True,
    )

    print(
        f"[OCR DUAL] saved: {final_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()