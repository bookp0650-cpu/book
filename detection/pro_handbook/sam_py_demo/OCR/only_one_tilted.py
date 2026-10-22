# import json
# from rapidfuzz import fuzz

# from pathlib import Path
# import json
# from datetime import datetime
# import numpy as np
# import cv2
# def unrotate_poly_to_original(poly, angle: int, w: int, h: int):
#     """
#     poly: [[x,y]x4]  (OCRが返す座標: 回転後画像座標)
#     angle: doc_preprocessor_res["angle"] (0/90/180/270)
#            ※ここでは「時計回り(angle)だけ回転された画像上の座標」を元に戻す前提
#     w,h: 元画像サイズ（SAMと同じ画像の width/height）
#     return: 元画像座標系に戻したpoly
#     """
#     pts = np.asarray(poly, dtype=np.float32)  # (4,2)
#     x = pts[:, 0]
#     y = pts[:, 1]

#     a = int(angle) % 360
#     if a == 0:
#         xo, yo = x, y
#     elif a == 90:
#         # 90° clockwise の逆変換
#         xo = y
#         yo = (h - 1) - x
#     elif a == 180:
#         xo = (w - 1) - x
#         yo = (h - 1) - y
#     elif a == 270:
#         # 270° clockwise (= 90° CCW) の逆変換
#         xo = (w - 1) - y
#         yo = x
#     else:
#         # 想定外はそのまま返す
#         xo, yo = x, y

#     return np.stack([xo, yo], axis=1).tolist()

# def compute_min_arearect_list(masks, min_area_px=50):
#     """
#     masks: バイナリマスクのリスト/iterable
#     return: 各マスクの minAreaRect（取れなければ None）のリスト
#             rect = ((cx,cy),(w,h),angle)
#     """
#     min_arearect = []
#     for m in masks:
#         b = (np.asarray(m) > 0).astype(np.uint8)
#         if b.sum() < min_area_px:
#             min_arearect.append(None); continue
#         cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not cnts:
#             min_arearect.append(None); continue
#         rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
#         min_arearect.append(rect if rect[1][0] * rect[1][1] > 1e-6 else None)
#     return min_arearect

# def rects_to_pts(rects):
#     """rects: list of minAreaRect or None -> list of quad(4,2)"""
#     quads = []
#     for r in rects:
#         if r is None:
#             continue
#         quads.append(cv2.boxPoints(r).astype(np.float32))
#     return quads


# def load_text_data(shot_dir):
#     with open(shot_dir, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     all_min_arearect = data.get("dt_polys", [])
#     all_mojis = data.get("rec_texts", [])
#     return all_min_arearect, all_mojis

# def match_text_to_mask(all_boxes, all_mojis, sam_quads, rgb_path, forced_angle=None):
#     img = cv2.imread(str(rgb_path))
#     h, w = img.shape[:2]

#     if forced_angle in (90, 180, 270):
#         all_boxes = [unrotate_poly_to_original(p, forced_angle, w=w, h=h) for p in all_boxes]

#     names = [f"mask_{i+1}" for i in range(len(sam_quads))]
#     book_texts = {n: [] for n in names}

#     # 返り値用に AABB も作る（必要なければ消してOK）
#     book_boxes = {}
#     for n, q in zip(names, sam_quads):
#         q = np.asarray(q, np.float32)
#         book_boxes[n] = {"x1": float(q[:,0].min()), "y1": float(q[:,1].min()),
#                          "x2": float(q[:,0].max()), "y2": float(q[:,1].max())}

#     for moji, quad in zip(all_mojis, all_boxes):
#         pts = np.asarray(quad, np.float32)
#         c = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))  # 重心

#         matched = None
#         for n, q in zip(names, sam_quads):
#             poly = cv2.convexHull(np.asarray(q, np.float32).reshape(-1, 1, 2))
#             if cv2.pointPolygonTest(poly, c, False) >= 0:  # 内部 or 辺上
#                 matched = n
#                 break

#         if matched is not None:
#             book_texts[matched].append(moji)  # どこにも入らなければ捨てる

#     return book_texts, book_boxes

# def save_json(path: str | Path, obj) -> None:
#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

# def match_book_name(query, book_texts, shot_dir, threshold=40.0):
#     matches = []
#     all_scores = []

#     for name, text in book_texts.items():
#         combined = " ".join(text) if isinstance(text, (list, tuple)) else str(text)
#         score = int(fuzz.partial_ratio(query, combined))
#         all_scores.append({"name": name, "score": score, "text": combined})

#         if score > threshold:
#             matches.append((name, score))

#     matches.sort(key=lambda x: x[1], reverse=True)
#     all_scores.sort(key=lambda d: d["score"], reverse=True)
#     # 全スコアを保存（可視化=見れる形）
#     save_json(Path(shot_dir) / "similarity_scores.json", {
#         "query": query,
#         "threshold": threshold,
#         "scores": all_scores, # [{"name","score","text"}, ...]
#     })


#     return matches, all_scores



# def match_text_to_mask_main(query, masks, shot_dir, threshold = 40):
#     ocr_json_path = shot_dir / "ocr_result.json"
#     rgb_path = shot_dir / "after_init_rgb.png"
#     sam_min_arearect = compute_min_arearect_list(masks)
#     sam_rect_points = rects_to_pts(sam_min_arearect)
#     all_text_min_arearects, all_texts = load_text_data(ocr_json_path)
#     book_texts, book_boxes = match_text_to_mask(all_text_min_arearects, all_texts, sam_rect_points, 
#                                                 rgb_path, forced_angle=90)                                               
#     matches, all_scores = match_book_name(query, book_texts, shot_dir, threshold)  # ★ここ

#     results = []
#     for name, score in matches:
#         results.append({
#             "name": name,
#             "score": score,
#             "box": book_boxes.get(name),
#             "forced_angle": 90
#         })
#     return results
 
import json
import time
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz


def unrotate_poly_to_original(poly, angle: int, w: int, h: int):
    """
    poly: [[x,y]x4]  (OCRが返す座標: 回転後画像座標)
    angle: doc_preprocessor_res["angle"] (0/90/180/270)
           ※ここでは「時計回り(angle)だけ回転された画像上の座標」を元に戻す前提
    w,h: 元画像サイズ（SAMと同じ画像の width/height）
    return: 元画像座標系に戻したpoly
    """
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]

    a = int(angle) % 360
    if a == 0:
        xo, yo = x, y
    elif a == 90:
        # 90° clockwise の逆変換
        xo = y
        yo = (h - 1) - x
    elif a == 180:
        xo = (w - 1) - x
        yo = (h - 1) - y
    elif a == 270:
        # 270° clockwise (= 90° CCW) の逆変換
        xo = (w - 1) - y
        yo = x
    else:
        xo, yo = x, y

    return np.stack([xo, yo], axis=1).tolist()


def compute_min_arearect_list(masks, min_area_px=50):
    """
    互換性のため残している関数。
    今回の IoU ベース対応付けでは未使用。
    """
    min_arearect = []
    for m in masks:
        b = (np.asarray(m) > 0).astype(np.uint8)
        if b.sum() < min_area_px:
            min_arearect.append(None)
            continue

        cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            min_arearect.append(None)
            continue

        rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        min_arearect.append(rect if rect[1][0] * rect[1][1] > 1e-6 else None)
    return min_arearect


def rects_to_pts(rects):
    """
    互換性のため残している関数。
    今回の IoU ベース対応付けでは未使用。
    """
    quads = []
    for r in rects:
        if r is None:
            continue
        quads.append(cv2.boxPoints(r).astype(np.float32))
    return quads


def load_text_data(shot_dir):
    with open(shot_dir, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_min_arearect = data.get("dt_polys", [])
    all_mojis = data.get("rec_texts", [])
    return all_min_arearect, all_mojis


def load_per_angle_text_data(shot_dir):
    with open(shot_dir, "r", encoding="utf-8") as f:
        data = json.load(f)

    angles = data.get("per_angle_ocr", {}).get("angles", {})
    required = ("0deg", "+30deg", "-30deg")
    if not all(label in angles for label in required):
        return None

    return {
        label: (
            angles[label].get("dt_polys", []),
            angles[label].get("rec_texts", []),
        )
        for label in required
    }


def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _mask_to_binary(mask, h: int, w: int) -> np.ndarray:
    """
    任意の mask を 0/1 の uint8 2次元配列にそろえる
    """
    b = (np.asarray(mask) > 0).astype(np.uint8)

    if b.ndim > 2:
        b = np.squeeze(b)

    if b.ndim != 2:
        raise ValueError(f"mask must be 2D after squeeze, but got shape={b.shape}")

    if b.shape != (h, w):
        raise ValueError(f"mask shape mismatch: expected {(h, w)}, got {b.shape}")

    return b


def _poly_to_mask(poly, h: int, w: int) -> np.ndarray:
    """
    OCR polygon を塗りつぶした 0/1 mask に変換
    """
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)

    if pts.shape[0] < 3:
        return np.zeros((h, w), dtype=np.uint8)

    pts = np.round(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(canvas, [pts], 1)
    return canvas


def _mask_bbox(mask_bin: np.ndarray):
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return {
        "x1": float(xs.min()),
        "y1": float(ys.min()),
        "x2": float(xs.max()),
        "y2": float(ys.max()),
    }


def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray):
    """
    0/1 mask 同士の IoU
    """
    inter = int(np.logical_and(mask_a > 0, mask_b > 0).sum())
    if inter == 0:
        return 0.0, 0, int(np.logical_or(mask_a > 0, mask_b > 0).sum())

    union = int(np.logical_or(mask_a > 0, mask_b > 0).sum())
    if union <= 0:
        return 0.0, inter, 0

    return float(inter / union), inter, union


def match_text_to_mask(
    all_boxes,
    all_mojis,
    sam_quads,   # ← 互換性のため名前はそのまま。中身は「SAM masks」を渡す
    rgb_path,
    forced_angle=None,
    return_debug=False,
):
    """
    OCR polygon と SAM マスクの IoU に基づいて文字列を本マスクへ割り当てる。

    Parameters
    ----------
    all_boxes : list
        OCR の polygon 一覧（dt_polys）
    all_mojis : list
        OCR の認識文字列一覧（rec_texts）
    sam_quads : list
        実際には SAM masks の list を渡す
    rgb_path : Path or str
        元画像パス
    forced_angle : int or None
        OCR polygon を元画像座標へ戻す回転角
    return_debug : bool
        True のとき debug 情報も返す

    Returns
    -------
    book_texts, book_boxes
    または
    book_texts, book_boxes, debug_rows
    """
    img = cv2.imread(str(rgb_path))
    if img is None:
        raise FileNotFoundError(f"画像が読めませんでした: {rgb_path}")

    h, w = img.shape[:2]

    if forced_angle in (90, 180, 270):
        all_boxes = [unrotate_poly_to_original(p, forced_angle, w=w, h=h) for p in all_boxes]

    # sam_quads という名前だが、中身は SAM の元マスクを受ける
    sam_masks = [_mask_to_binary(m, h, w) for m in sam_quads]

    names = [f"mask_{i+1}" for i in range(len(sam_masks))]
    book_texts = {n: [] for n in names}

    book_boxes = {}
    for n, m in zip(names, sam_masks):
        book_boxes[n] = _mask_bbox(m)

    debug_rows = []

    for idx, (moji, quad) in enumerate(zip(all_mojis, all_boxes), start=1):
        text_mask = _poly_to_mask(quad, h, w)
        text_area = int(text_mask.sum())

        if text_area <= 0:
            debug_rows.append({
                "ocr_index": idx,
                "text": moji,
                "matched": None,
                "best_iou": 0.0,
                "text_area": 0,
                "note": "text polygon area is zero",
            })
            continue

        best_name = None
        best_iou = 0.0
        best_inter = 0
        best_union = 0
        best_text_cover = 0.0

        per_mask_scores = []

        for name, sam_mask in zip(names, sam_masks):
            iou, inter, union = _compute_iou(text_mask, sam_mask)
            text_cover = float(inter / text_area) if text_area > 0 else 0.0

            per_mask_scores.append({
                "name": name,
                "iou": float(iou),
                "intersection": int(inter),
                "union": int(union),
                "text_cover": float(text_cover),
            })

            # まず IoU 最大を優先
            # 同点なら「OCR領域の何割がその本に含まれるか」で比較
            if (
                iou > best_iou
                or (
                    np.isclose(iou, best_iou)
                    and text_cover > best_text_cover
                )
            ):
                best_name = name
                best_iou = float(iou)
                best_inter = int(inter)
                best_union = int(union)
                best_text_cover = float(text_cover)

        # 少なくとも 1px でも重なっているものだけ採用
        matched = best_name if best_inter > 0 else None

        if matched is not None:
            book_texts[matched].append(moji)

        debug_rows.append({
            "ocr_index": idx,
            "text": moji,
            "matched": matched,
            "best_iou": float(best_iou),
            "best_intersection": int(best_inter),
            "best_union": int(best_union),
            "best_text_cover": float(best_text_cover),
            "text_area": int(text_area),
            "per_mask_scores": per_mask_scores,
        })

    if return_debug:
        return book_texts, book_boxes, debug_rows
    return book_texts, book_boxes


def match_book_name(query, book_texts, shot_dir, threshold=40.0):
    matches = []
    all_scores = []

    for name, text in book_texts.items():
        combined = " ".join(text) if isinstance(text, (list, tuple)) else str(text)
        score = int(fuzz.partial_ratio(query, combined))
        all_scores.append({"name": name, "score": score, "text": combined})

        if score > threshold:
            matches.append((name, score))

    matches.sort(key=lambda x: x[1], reverse=True)
    all_scores.sort(key=lambda d: d["score"], reverse=True)

    save_json(
        Path(shot_dir) / "similarity_scores.json",
        {
            "query": query,
            "threshold": threshold,
            "scores": all_scores,
        },
    )

    return matches, all_scores


def match_book_name_by_angle(
    query,
    book_texts_by_angle,
    shot_dir,
    threshold=40.0,
):
    """Score each OCR angle independently and retain its maximum per mask."""
    angle_fields = (
        ("0deg", "text_0deg", "score_0deg"),
        ("+30deg", "text_plus30", "score_plus30"),
        ("-30deg", "text_minus30", "score_minus30"),
    )
    names = list(book_texts_by_angle["0deg"].keys())
    matches = []
    all_scores = []

    for name in names:
        row = {"name": name}
        angle_scores = []

        for angle, text_field, score_field in angle_fields:
            parts = book_texts_by_angle[angle].get(name, [])
            text = " ".join(parts) if isinstance(parts, (list, tuple)) else str(parts)
            score = int(fuzz.partial_ratio(query, text))
            row[text_field] = text
            row[score_field] = score
            angle_scores.append((score, angle, text))

        # Tie order intentionally prefers 0deg, then +30deg, then -30deg.
        # Adding rescue angles therefore cannot replace an equally good 0deg result.
        final_score, source_angle, final_text = max(
            angle_scores,
            key=lambda item: item[0],
        )
        row.update(
            {
                # Backward-compatible fields used by existing artifacts/readers.
                "score": final_score,
                "text": final_text,
                "final_score": final_score,
                "final_score_source_angle": source_angle,
            }
        )
        all_scores.append(row)

        if final_score > threshold:
            matches.append((name, final_score))

    matches.sort(key=lambda item: item[1], reverse=True)
    all_scores.sort(key=lambda row: row["final_score"], reverse=True)

    save_json(
        Path(shot_dir) / "similarity_scores.json",
        {
            "query": query,
            "threshold": threshold,
            "scoring_mode": "max_independent_angle_scores",
            "source_priority_for_ties": ["0deg", "+30deg", "-30deg"],
            "scores": all_scores,
        },
    )

    return matches, all_scores


def match_text_to_mask_main(query, masks, shot_dir, threshold=40):
    """
    既存コードからそのまま呼べる入口関数。
    run_capture_and_pca 側の修正は不要。
    """
    ocr_json_path = Path(shot_dir) / "ocr_result.json"
    rgb_path = Path(shot_dir) / "after_init_rgb.png"

    per_angle_text_data = load_per_angle_text_data(ocr_json_path)

    # OCR 完了後にだけ GPU 実装を読み込み、CUDA 不可/OOM 時は内部で本ファイルの
    # match_text_to_mask へフォールバックする。
    from .only_one_tilted_gpu import match_text_to_mask as match_text_to_mask_gpu

    if per_angle_text_data is None:
        # Older saved OCR results have only the legacy merged arrays.
        all_text_polys, all_texts = load_text_data(ocr_json_path)
        assignment_start = time.perf_counter()
        book_texts, book_boxes, debug_rows = match_text_to_mask_gpu(
            all_text_polys,
            all_texts,
            masks,
            rgb_path,
            forced_angle=90,
            return_debug=True,
            ocr_batch_size=4,
        )
        assignment_seconds = time.perf_counter() - assignment_start
        similarity_start = time.perf_counter()
        matches, all_scores = match_book_name(
            query,
            book_texts,
            shot_dir,
            threshold,
        )
        similarity_seconds = time.perf_counter() - similarity_start
        debug_payload = {
            "forced_angle": 90,
            "num_masks": len(masks),
            "num_ocr_boxes": len(all_texts),
            "assignments": debug_rows,
            "scoring_mode": "legacy_combined_fallback",
            "timing_seconds": {
                "ocr_to_mask_assignment": float(assignment_seconds),
                "similarity_and_scores_json": float(similarity_seconds),
            },
        }
    else:
        book_texts_by_angle = {}
        debug_rows_by_angle = {}
        book_boxes = None

        assignment_seconds_by_angle = {}
        for angle in ("0deg", "+30deg", "-30deg"):
            all_text_polys, all_texts = per_angle_text_data[angle]
            assignment_start = time.perf_counter()
            angle_book_texts, angle_book_boxes, angle_debug_rows = (
                match_text_to_mask_gpu(
                    all_text_polys,
                    all_texts,
                    masks,
                    rgb_path,
                    forced_angle=90,
                    return_debug=True,
                    ocr_batch_size=4,
                )
            )
            assignment_seconds_by_angle[angle] = float(
                time.perf_counter() - assignment_start
            )
            book_texts_by_angle[angle] = angle_book_texts
            debug_rows_by_angle[angle] = angle_debug_rows
            if book_boxes is None:
                book_boxes = angle_book_boxes

        similarity_start = time.perf_counter()
        matches, all_scores = match_book_name_by_angle(
            query,
            book_texts_by_angle,
            shot_dir,
            threshold,
        )
        similarity_seconds = time.perf_counter() - similarity_start
        debug_payload = {
            "forced_angle": 90,
            "num_masks": len(masks),
            "num_ocr_boxes": sum(
                len(per_angle_text_data[angle][1])
                for angle in ("0deg", "+30deg", "-30deg")
            ),
            "num_ocr_boxes_by_angle": {
                angle: len(per_angle_text_data[angle][1])
                for angle in ("0deg", "+30deg", "-30deg")
            },
            "assignments": [
                {**row, "source_angle": angle}
                for angle in ("0deg", "+30deg", "-30deg")
                for row in debug_rows_by_angle[angle]
            ],
            "per_angle_assignments": debug_rows_by_angle,
            "scoring_mode": "max_independent_angle_scores",
            "timing_seconds": {
                "ocr_to_mask_assignment_by_angle": assignment_seconds_by_angle,
                "ocr_to_mask_assignment": float(
                    sum(assignment_seconds_by_angle.values())
                ),
                "similarity_and_scores_json": float(similarity_seconds),
            },
        }

    save_json(
        Path(shot_dir) / "text_mask_iou_debug.json",
        debug_payload,
    )

    score_rows = {row["name"]: row for row in all_scores}

    results = []
    for name, score in matches:
        results.append(
            {
                "name": name,
                "score": score,
                "box": book_boxes.get(name),
                "forced_angle": 90,
                "final_score_source_angle": score_rows[name].get(
                    "final_score_source_angle",
                    "combined",
                ),
            }
        )
    return results
