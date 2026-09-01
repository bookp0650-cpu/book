from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Tuple, Callable

import cv2
import numpy as np
import pyrealsense2 as rs


# ============================================================
# RealSense Depth filter
# ============================================================

def depth_filter_like_viewer(raw_depth_frame: rs.depth_frame) -> rs.depth_frame:
    """RealSense Viewerに近いDepthフィルタ．decimationは使わない．"""
    spat = rs.spatial_filter()
    spat.set_option(rs.option.filter_magnitude, 2.0)
    spat.set_option(rs.option.filter_smooth_alpha, 0.5)
    spat.set_option(rs.option.filter_smooth_delta, 20.0)

    hole_fill = rs.hole_filling_filter()
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    filtered = depth_to_disparity.process(raw_depth_frame)
    filtered = spat.process(filtered)
    filtered = disparity_to_depth.process(filtered)
    filtered = hole_fill.process(filtered)
    return filtered.as_depth_frame()


# ============================================================
# Crop-aware depth mask generation
# ============================================================

def estimate_depth_mode(
    depth_u16: np.ndarray,
    depth_scale: float,
    z_min_m: float = 0.2,
    z_max_m: float = 1.2,
    bins: int = 100,
) -> float:
    """指定Depth画像内で最も多いDepth帯を代表書籍面Depthとして推定する．"""
    depth_m = depth_u16.astype(np.float32) * float(depth_scale)
    valid = (depth_m > z_min_m) & (depth_m < z_max_m)
    if not np.any(valid):
        raise RuntimeError("有効なDepthがありません．")

    vals = depth_m[valid]
    hist, edges = np.histogram(vals, bins=bins, range=(z_min_m, z_max_m))
    idx = int(np.argmax(hist))
    return float(0.5 * (edges[idx] + edges[idx + 1]))


def estimate_depth_mode_in_y_range(
    depth_u16: np.ndarray,
    depth_scale: float,
    *,
    crop_y_min: int,
    crop_y_max: int,
    z_min_m: float = 0.2,
    z_max_m: float = 1.2,
    bins: int = 100,
) -> float:
    """書籍が存在する高さ帯だけで代表Depthを推定する．"""
    H = depth_u16.shape[0]
    y0 = int(np.clip(crop_y_min, 0, H - 1))
    y1 = int(np.clip(crop_y_max, 0, H))
    if y1 <= y0:
        raise ValueError(f"invalid crop range: crop_y_min={crop_y_min}, crop_y_max={crop_y_max}")

    return estimate_depth_mode(
        depth_u16[y0:y1, :],
        depth_scale,
        z_min_m=z_min_m,
        z_max_m=z_max_m,
        bins=bins,
    )


def make_far_space_mask(
    depth_u16: np.ndarray,
    depth_scale: float,
    depth_ref_m: float,
    delta_depth_m: float = 0.02,
) -> np.ndarray:
    """代表Depthより奥にある領域を収納スペース候補として二値化する．"""
    depth_m = depth_u16.astype(np.float32) * float(depth_scale)
    valid = depth_u16 > 0
    far = valid & (depth_m > float(depth_ref_m) + float(delta_depth_m))
    mask = far.astype(np.uint8)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask


def apply_vertical_crop_mask(
    mask: np.ndarray,
    *,
    crop_y_min: int,
    crop_y_max: int,
) -> np.ndarray:
    """
    画像全体サイズのmaskに対し，書籍高さ帯以外を0にする．
    サイズを変えないため，後段の画像座標を戻す必要がない．
    """
    out = np.zeros_like(mask, dtype=np.uint8)
    H = mask.shape[0]
    y0 = int(np.clip(crop_y_min, 0, H - 1))
    y1 = int(np.clip(crop_y_max, 0, H))
    if y1 <= y0:
        raise ValueError(f"invalid crop range: crop_y_min={crop_y_min}, crop_y_max={crop_y_max}")
    out[y0:y1, :] = (mask[y0:y1, :] > 0).astype(np.uint8)
    return out


def remove_horizontal_structures(
    mask: np.ndarray,
    *,
    horizontal_kernel_w: int = 300,
    horizontal_kernel_h: int = 30,
) -> np.ndarray:
    """横長構造を除去する．収納候補が削れすぎる場合はkernelを大きくする．"""
    mask_u8 = (mask > 0).astype(np.uint8)
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (int(horizontal_kernel_w), int(horizontal_kernel_h)),
    )
    horizontal = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, horizontal_kernel)
    cleaned = cv2.subtract(mask_u8, horizontal)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
    return cleaned

def merge_nearby_space_regions(
    mask: np.ndarray,
    *,
    close_kernel_w: int = 35,
    close_kernel_h: int = 35,
    dilate_kernel_w: int = 9,
    dilate_kernel_h: int = 9,
    dilate_iterations: int = 1,
) -> np.ndarray:
    """
    同じ収納スペース由来なのにDepth欠損やエッジで分断された白領域を統合する．

    注意：
      - 大きくしすぎると別スペース同士も結合する
      - まずは close_kernel_w/h = 35 程度から試す
    """
    mask_u8 = (mask > 0).astype(np.uint8)

    # 小さな隙間を埋めて，同じスペースの分断をつなぐ
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (int(close_kernel_w), int(close_kernel_h)),
    )
    merged = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)

    # 少しだけ膨張して，境界付近の途切れを補う
    if dilate_iterations > 0:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (int(dilate_kernel_w), int(dilate_kernel_h)),
        )
        merged = cv2.dilate(
            merged,
            dilate_kernel,
            iterations=int(dilate_iterations),
        )

    return (merged > 0).astype(np.uint8)

# ============================================================
# Candidate selection
# ============================================================

def compute_space_shape_features(component_mask: np.ndarray) -> dict[str, float]:
    """
    収納スペース候補の形状特徴を計算する．

    特に，
      - 上側の幅
      - 下側の幅
      - 上側が下側より広いか
    を見る．

    上部に長辺を持つ三角形・台形なら，
      top_width が大きく，bottom_width が小さくなりやすい．
    """
    mask = (component_mask > 0).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return {
            "top_width": 0.0,
            "middle_width": 0.0,
            "bottom_width": 0.0,
            "top_bottom_width_diff": 0.0,
            "top_bottom_width_ratio": 0.0,
            "max_row_width": 0.0,
            "median_row_width": 0.0,
        }

    y_min = int(ys.min())
    y_max = int(ys.max())
    h = y_max - y_min + 1

    row_widths = []
    row_info = []

    for y in range(y_min, y_max + 1):
        xs_y = np.where(mask[y] > 0)[0]
        if xs_y.size == 0:
            continue

        width = int(xs_y.max() - xs_y.min() + 1)
        row_widths.append(width)
        row_info.append((y, width))

    if len(row_widths) == 0:
        return {
            "top_width": 0.0,
            "middle_width": 0.0,
            "bottom_width": 0.0,
            "top_bottom_width_diff": 0.0,
            "top_bottom_width_ratio": 0.0,
            "max_row_width": 0.0,
            "median_row_width": 0.0,
        }

    def band_width(y0_ratio: float, y1_ratio: float) -> float:
        y0 = y_min + int(h * y0_ratio)
        y1 = y_min + int(h * y1_ratio)

        widths = [
            width for y, width in row_info
            if y0 <= y <= y1
        ]

        if len(widths) == 0:
            return 0.0

        # maxだとノイズに引っ張られやすいので，75パーセンタイルを使う
        return float(np.percentile(widths, 75))

    top_width = band_width(0.00, 0.30)
    middle_width = band_width(0.35, 0.65)
    bottom_width = band_width(0.70, 1.00)

    top_bottom_width_diff = top_width - bottom_width
    top_bottom_width_ratio = top_width / max(bottom_width, 1.0)

    return {
        "top_width": float(top_width),
        "middle_width": float(middle_width),
        "bottom_width": float(bottom_width),
        "top_bottom_width_diff": float(top_bottom_width_diff),
        "top_bottom_width_ratio": float(top_bottom_width_ratio),
        "max_row_width": float(np.max(row_widths)),
        "median_row_width": float(np.median(row_widths)),
    }

def select_space_component(
    candidate_mask: np.ndarray,
    *,
    min_area_px: int = 500,
    min_height_px: int = 100,
    horizontal_ratio_thr: float = 4.0,
    max_width_px: int = 520,
    max_area_px: int = 160000,
    min_band_width_px: int = 25,
    prefer_top_open_shape: bool = True,
) -> Optional[dict[str, Any]]:
    """
    Depthから得た候補二値画像に対して連結成分解析を行い，
    三角形・台形・四角形の収納スペース候補を1つ選ぶ．

    従来の「上部に長辺がある候補のみ優先」では，
    縦長四角形スペースが落ちやすいため，以下の2種類を許容する．

      1. 上部が広く，下部が狭い三角形・台形
      2. 上下の幅が近い縦長四角形

    Parameters
    ----------
    min_band_width_px:
        top / middle / bottom のいずれかの幅がこの値以上なら，
        スペース候補として最低限の幅があるとみなす．

    prefer_top_open_shape:
        Trueの場合，上部に長辺がある三角形・台形を少し優先する．
        ただし四角形も落とさない．
    """
    mask_u8 = (candidate_mask > 0).astype(np.uint8)
    H, W = mask_u8.shape[:2]

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )

    candidates: list[dict[str, Any]] = []

    print(f"[DEBUG] connected components: {num_labels - 1}")

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        cx, cy = centroids[label]

        aspect_w_h = float(w / max(h, 1))
        aspect_h_w = float(h / max(w, 1))

        component_mask = (labels == label).astype(np.uint8)
        shape_feat = compute_space_shape_features(component_mask)

        top_width = shape_feat["top_width"]
        middle_width = shape_feat["middle_width"]
        bottom_width = shape_feat["bottom_width"]
        max_row_width = shape_feat["max_row_width"]
        median_row_width = shape_feat["median_row_width"]

        top_bottom_diff = top_width - bottom_width
        top_bottom_ratio = top_width / max(bottom_width, 1.0)

        max_band_width = max(top_width, middle_width, bottom_width)
        min_nonzero_band_width = min(
            [v for v in [top_width, middle_width, bottom_width] if v > 0.0],
            default=0.0,
        )

        # 形状タイプの簡易分類
        is_top_open_triangle_or_trapezoid = (
            top_width >= float(min_band_width_px)
            and top_bottom_diff >= float(min_band_width_px)
        )

        is_vertical_rectangle = (
            max_band_width >= float(min_band_width_px)
            and abs(top_width - bottom_width) <= max(40.0, 0.6 * max_band_width)
            and middle_width >= float(min_band_width_px)
        )

        is_valid_shape = is_top_open_triangle_or_trapezoid or is_vertical_rectangle

        print(
            f"[DEBUG] label={label}, bbox=({x},{y},{w},{h}), "
            f"area={area}, w/h={aspect_w_h:.2f}, h/w={aspect_h_w:.2f}, "
            f"top={top_width:.1f}, mid={middle_width:.1f}, bottom={bottom_width:.1f}, "
            f"top-bottom={top_bottom_diff:.1f}, ratio={top_bottom_ratio:.2f}, "
            f"max_band={max_band_width:.1f}, "
            f"is_top_open={is_top_open_triangle_or_trapezoid}, "
            f"is_rect={is_vertical_rectangle}"
        )

        if area < int(min_area_px):
            print("  -> reject: area too small")
            continue

        if h < int(min_height_px):
            print("  -> reject: height too small")
            continue

        if area > int(max_area_px):
            print("  -> reject: area too large")
            continue

        if h <= 0 or w <= 0:
            print("  -> reject: invalid bbox")
            continue

        if aspect_w_h > float(horizontal_ratio_thr):
            print("  -> reject: horizontal")
            continue

        if w > int(max_width_px):
            print("  -> reject: bbox too wide")
            continue

        # ここが重要：
        # top_width だけで落とさず，top / middle / bottom のどれかが十分なら通す．
        if max_band_width < float(min_band_width_px):
            print("  -> reject: all band widths too small")
            continue

        # 三角形・台形・四角形のどれにも見えないものは落とす
        if not is_valid_shape:
            print("  -> reject: not triangle/trapezoid/rectangle-like")
            continue

        # row幅の最大値は緩めに見る．四角形も許容するため厳しすぎないようにする．
        if max_row_width > float(max_width_px) * 1.3:
            print("  -> reject: row too wide")
            continue

        # スコア設計：
        # - 高さがある
        # - 面積がある
        # - 幅がある
        # - 四角形も許容
        # - ただし，上部に長辺がある候補は少し優先
        score = 0.0
        score += 6.0 * float(h)
        score += 0.04 * float(area)
        score += 2.0 * float(max_band_width)
        score += 1.0 * float(median_row_width)
        score += 80.0 * float(aspect_h_w)

        if is_vertical_rectangle:
            score += 300.0

        if prefer_top_open_shape and is_top_open_triangle_or_trapezoid:
            score += 300.0
            score += 3.0 * max(0.0, float(top_bottom_diff))
            score += 80.0 * min(float(top_bottom_ratio), 4.0)

        # 画像のかなり下だけにあるノイズは少し減点
        score -= 0.2 * float(y)

        candidates.append({
            "label": int(label),
            "bbox": (int(x), int(y), int(w), int(h)),
            "center": (float(cx), float(cy)),
            "area": int(area),
            "width": int(w),
            "height": int(h),
            "aspect_w_h": aspect_w_h,
            "aspect_h_w": aspect_h_w,
            "top_width": float(top_width),
            "middle_width": float(middle_width),
            "bottom_width": float(bottom_width),
            "top_bottom_width_diff": float(top_bottom_diff),
            "top_bottom_width_ratio": float(top_bottom_ratio),
            "max_band_width": float(max_band_width),
            "min_nonzero_band_width": float(min_nonzero_band_width),
            "median_row_width": float(median_row_width),
            "max_row_width": float(max_row_width),
            "is_top_open_triangle_or_trapezoid": bool(is_top_open_triangle_or_trapezoid),
            "is_vertical_rectangle": bool(is_vertical_rectangle),
            "score": float(score),
            "mask": component_mask,
        })

        print(f"  -> keep: score={score:.1f}")

    if not candidates:
        return None

    candidates.sort(key=lambda d: d["score"], reverse=True)

    print("[DEBUG] selected component:", {
        "bbox": candidates[0]["bbox"],
        "area": candidates[0]["area"],
        "score": candidates[0]["score"],
        "top_width": candidates[0]["top_width"],
        "middle_width": candidates[0]["middle_width"],
        "bottom_width": candidates[0]["bottom_width"],
        "is_top_open": candidates[0]["is_top_open_triangle_or_trapezoid"],
        "is_rect": candidates[0]["is_vertical_rectangle"],
    })

    return candidates[0]

def trim_bottom_excess_from_space_mask(
    component_mask: np.ndarray,
    *,
    max_row_width_px: int = 180,
    row_width_ratio_thr: float = 2.0,
    bottom_search_ratio: float = 0.45,
    min_keep_height_px: int = 60,
) -> np.ndarray:
    """選択された収納スペース候補から，下側に連結した余分領域を削除する．"""
    mask = (component_mask > 0).astype(np.uint8)
    ys_all, _ = np.where(mask > 0)
    if ys_all.size == 0:
        return mask

    y_min = int(ys_all.min())
    y_max = int(ys_all.max())
    comp_h = y_max - y_min + 1
    if comp_h < min_keep_height_px:
        return mask

    row_infos = []
    for y in range(y_min, y_max + 1):
        xs = np.where(mask[y] > 0)[0]
        if xs.size == 0:
            continue
        x_left = int(xs.min())
        x_right = int(xs.max())
        width = x_right - x_left + 1
        row_infos.append((y, x_left, x_right, width))

    if len(row_infos) < 5:
        return mask

    widths = np.array([r[3] for r in row_infos], dtype=np.float32)
    ref_width = float(np.median(widths))
    width_thr = min(
        float(max_row_width_px),
        max(float(max_row_width_px) * 0.5, ref_width * float(row_width_ratio_thr)),
    )

    search_start_y = int(y_min + comp_h * (1.0 - bottom_search_ratio))
    cut_y = None
    for y, _, _, width in reversed(row_infos):
        if y < search_start_y:
            break
        if width > width_thr:
            cut_y = y
        elif cut_y is not None:
            break

    cleaned = mask.copy()
    if cut_y is not None:
        cut_y = max(cut_y, y_min + min_keep_height_px)
        cleaned[cut_y:, :] = 0
        print(
            f"[INFO] trim_bottom_excess: y_min={y_min}, y_max={y_max}, "
            f"ref_width={ref_width:.1f}, width_thr={width_thr:.1f}, cut_y={cut_y}"
        )
    else:
        print(f"[INFO] trim_bottom_excess: no trim, ref_width={ref_width:.1f}, width_thr={width_thr:.1f}")

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels <= 1:
        return cleaned

    best_label = None
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label
    if best_label is None:
        return cleaned
    return (labels == best_label).astype(np.uint8)


# ============================================================
# Guide line and target pixel
# ============================================================

def extract_side_boundary_points(component_mask: np.ndarray, side: str) -> np.ndarray:
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side}")
    mask = component_mask.astype(bool)
    pts = []
    ys = np.where(mask.any(axis=1))[0]
    for y in ys:
        xs = np.where(mask[y])[0]
        if xs.size == 0:
            continue
        x = int(xs.min()) if side == "left" else int(xs.max())
        pts.append([x, int(y)])
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def fit_line_pca_2d(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 2:
        raise RuntimeError("直線フィットに必要な点が不足しています．")

    mean = pts.mean(axis=0)
    X = pts - mean
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    axis = vh[0].astype(np.float32)
    if axis[1] < 0:
        axis = -axis
    proj = X @ axis
    p0 = mean + axis * float(proj.min())
    p1 = mean + axis * float(proj.max())
    if p0[1] > p1[1]:
        p0, p1 = p1, p0
    return p0.astype(np.float32), p1.astype(np.float32)


def line_tilt_from_vertical_deg(p0: np.ndarray, p1: np.ndarray) -> float:
    p0 = np.asarray(p0, dtype=np.float32).reshape(2)
    p1 = np.asarray(p1, dtype=np.float32).reshape(2)
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    if angle_deg < 0.0:
        angle_deg += 180.0
    return float(abs(angle_deg - 90.0))


def extract_tilted_boundary_line(
    component_mask: np.ndarray,
    *,
    default_side: str = "left",
    min_points: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict[str, Any]]:
    """左右境界のうち，垂直からより傾いた側をガイドラインにする．"""
    left_pts = extract_side_boundary_points(component_mask, "left")
    right_pts = extract_side_boundary_points(component_mask, "right")

    candidates = []
    if left_pts.shape[0] >= min_points:
        lp0, lp1 = fit_line_pca_2d(left_pts)
        candidates.append({"side": "left", "pts": left_pts, "p0": lp0, "p1": lp1, "tilt": line_tilt_from_vertical_deg(lp0, lp1)})
    if right_pts.shape[0] >= min_points:
        rp0, rp1 = fit_line_pca_2d(right_pts)
        candidates.append({"side": "right", "pts": right_pts, "p0": rp0, "p1": rp1, "tilt": line_tilt_from_vertical_deg(rp0, rp1)})

    if len(candidates) == 0:
        raise RuntimeError("左右境界点が不足しています．")

    candidates.sort(key=lambda d: d["tilt"], reverse=True)
    selected = candidates[0]
    if len(candidates) >= 2 and abs(candidates[0]["tilt"] - candidates[1]["tilt"]) < 1.0:
        selected = next((c for c in candidates if c["side"] == default_side), selected)

    info = {
        "left_points": int(left_pts.shape[0]),
        "right_points": int(right_pts.shape[0]),
        "left_tilt_from_vertical_deg": None,
        "right_tilt_from_vertical_deg": None,
    }
    for c in candidates:
        if c["side"] == "left":
            info["left_tilt_from_vertical_deg"] = c["tilt"]
        elif c["side"] == "right":
            info["right_tilt_from_vertical_deg"] = c["tilt"]

    return (
        selected["p0"].astype(np.float32),
        selected["p1"].astype(np.float32),
        selected["pts"].astype(np.float32),
        selected["side"],
        info,
    )


def select_target_pixel_on_line_by_y(
    p0: tuple[int, int],
    p1: tuple[int, int],
    *,
    target_y_px: int,
    width: int,
    height: int,
) -> tuple[int, int]:
    """画像上のガイドラインp0-p1に対し，指定y座標上の点を求める．"""
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])
    target_y_px = int(np.clip(target_y_px, 0, height - 1))
    if abs(y1 - y0) < 1:
        x = int(round(0.5 * (x0 + x1)))
        return int(np.clip(x, 0, width - 1)), target_y_px
    t = (float(target_y_px) - float(y0)) / float(y1 - y0)
    t = float(np.clip(t, 0.0, 1.0))
    x = float(x0) + t * float(x1 - x0)
    return int(np.clip(round(x), 0, width - 1)), target_y_px


# ============================================================
# 3D geometry
# ============================================================

def deproject_single_pixel_to_3d(
    pixel_xy: tuple[int, int],
    depth_u16: np.ndarray,
    intr: rs.intrinsics,
    depth_scale: float,
    *,
    search_radius_px: int = 5,
    z_min_m: float = 0.1,
    z_max_m: float = 1.5,
) -> np.ndarray:
    H, W = depth_u16.shape[:2]
    u, v = int(pixel_xy[0]), int(pixel_xy[1])
    u = int(np.clip(u, 0, W - 1))
    v = int(np.clip(v, 0, H - 1))
    d = int(depth_u16[v, u])
    z = float(d) * float(depth_scale)

    # 0 / 65535 / 範囲外Depthなら周囲から有効Depthを探す
    if d == 0 or d == 65535 or z < z_min_m or z > z_max_m:

        x0 = max(0, u - search_radius_px)
        x1 = min(W, u + search_radius_px + 1)
        y0 = max(0, v - search_radius_px)
        y1 = min(H, v + search_radius_px + 1)

        patch = depth_u16[y0:y1, x0:x1]
        patch_z = patch.astype(np.float32) * float(depth_scale)

        valid_mask = (
            (patch > 0)
            & (patch < 65535)
            & (patch_z >= z_min_m)
            & (patch_z <= z_max_m)
        )

        valid = patch[valid_mask]

        if valid.size == 0:
            raise RuntimeError(
                f"target pixel depth is invalid and no valid depth nearby: "
                f"pixel={pixel_xy}, raw_depth={d}, z={z:.4f}"
            )

        d = int(np.median(valid))
        z = float(d) * float(depth_scale)

        print(
            f"[WARN] invalid target depth -> neighborhood fallback: "
            f"pixel={pixel_xy}, depth={z:.4f} m"
        )

    X, Y, Z = rs.rs2_deproject_pixel_to_point(
        intr,
        [float(u), float(v)],
        z
    )
    return np.asarray([X, Y, Z], dtype=np.float32)

    z = float(d) * float(depth_scale)
    if z < z_min_m or z > z_max_m:
        raise RuntimeError(f"target depth out of range: pixel={pixel_xy}, z={z:.4f}")
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], z)
    return np.asarray([X, Y, Z], dtype=np.float32)


def collect_book_surface_points_around_space(
    depth_u16: np.ndarray,
    intr: rs.intrinsics,
    depth_scale: float,
    bbox: tuple[int, int, int, int],
    depth_ref_m: float,
    *,
    depth_tol_m: float = 0.05,
    side_margin_px: int = 140,
    gap_px: int = 5,
    stride: int = 2,
    z_min_m: float = 0.1,
    z_max_m: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """収納スペースbbox左右から，書籍面らしいDepth点を集める．"""
    x, y, w, h = [int(v) for v in bbox]
    H, W = depth_u16.shape[:2]
    y0 = max(0, y)
    y1 = min(H, y + h)
    rois = [
        (max(0, x - int(side_margin_px)), max(0, x - int(gap_px)), y0, y1),
        (min(W, x + w + int(gap_px)), min(W, x + w + int(side_margin_px)), y0, y1),
    ]

    pts = []
    pixels = []
    for rx0, rx1, ry0, ry1 in rois:
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        for v in range(ry0, ry1, max(1, int(stride))):
            for u in range(rx0, rx1, max(1, int(stride))):
                d = int(depth_u16[v, u])
                if d == 0:
                    continue
                z = float(d) * float(depth_scale)
                if z < z_min_m or z > z_max_m:
                    continue
                if abs(z - float(depth_ref_m)) > float(depth_tol_m):
                    continue
                X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], z)
                pts.append([X, Y, Z])
                pixels.append([u, v])

    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)
    return np.asarray(pts, dtype=np.float32), np.asarray(pixels, dtype=np.int32)


def fit_plane_ransac(
    points: np.ndarray,
    *,
    distance_threshold_m: float = 0.01,
    max_iter: int = 300,
    min_inliers: int = 30,
    random_seed: int = 0,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = pts.shape[0]
    if n < 3:
        return None, np.zeros((n,), dtype=bool)

    rng = np.random.default_rng(random_seed)
    best_plane = None
    best_inlier_mask = np.zeros((n,), dtype=bool)
    best_count = 0

    for _ in range(int(max_iter)):
        ids = rng.choice(n, size=3, replace=False)
        p1, p2, p3 = pts[ids]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal = normal / norm
        d = -float(np.dot(normal, p1))
        plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)
        distances = np.abs(pts @ plane[:3] + plane[3])
        inlier_mask = distances < float(distance_threshold_m)
        count = int(np.sum(inlier_mask))
        if count > best_count:
            best_count = count
            best_plane = plane
            best_inlier_mask = inlier_mask

    if best_plane is None or best_count < int(min_inliers):
        return None, np.zeros((n,), dtype=bool)

    inlier_pts = pts[best_inlier_mask]
    centroid = inlier_pts.mean(axis=0)
    X = inlier_pts - centroid
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    normal = vh[-1].astype(np.float32)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
    d = -float(np.dot(normal, centroid))
    plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)
    return plane, best_inlier_mask


def intersect_pixel_ray_with_plane(
    pixel_xy: tuple[int, int],
    intr: rs.intrinsics,
    plane: np.ndarray,
    *,
    surface_clearance_m: float = 0.0,
) -> np.ndarray:
    """画像上の画素レイとRANSAC平面の交点を求める．"""
    u, v = int(pixel_xy[0]), int(pixel_xy[1])
    a, b, c, d = [float(x) for x in np.asarray(plane).reshape(4)]
    ray = np.asarray(rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], 1.0), dtype=np.float32)
    normal = np.asarray([a, b, c], dtype=np.float32)
    denom = float(np.dot(normal, ray))
    if abs(denom) < 1e-8:
        raise RuntimeError("画素レイと平面がほぼ平行なため，交点を計算できません．")
    scale = -d / denom
    if scale <= 0:
        raise RuntimeError(f"平面交点がカメラ後方になりました．pixel={pixel_xy}, scale={scale:.6f}")
    point = ray * float(scale)
    point[2] -= float(surface_clearance_m)
    return point.astype(np.float32)


def fallback_correct_point_z_by_median_surface(
    point_cam: np.ndarray,
    surface_points: np.ndarray,
    *,
    surface_clearance_m: float = 0.0,
) -> np.ndarray:
    pts = np.asarray(surface_points, dtype=np.float32).reshape(-1, 3)
    p = np.asarray(point_cam, dtype=np.float32).reshape(3)
    if pts.shape[0] == 0:
        return p.astype(np.float32)
    corrected = p.copy()
    corrected[2] = float(np.median(pts[:, 2])) - float(surface_clearance_m)
    return corrected.astype(np.float32)


def project_cam_point_to_pixel(point_cam: np.ndarray, intr: rs.intrinsics) -> tuple[int, int]:
    p = np.asarray(point_cam, dtype=np.float32).reshape(3)
    u, v = rs.rs2_project_point_to_pixel(intr, [float(p[0]), float(p[1]), float(p[2])])
    return int(round(u)), int(round(v))


def deproject_pixel_with_neighborhood(
    pixel_xy: tuple[int, int],
    depth_u16: np.ndarray,
    intr: rs.intrinsics,
    depth_scale: float,
    *,
    search_radius_px: int = 5,
    z_min_m: float = 0.1,
    z_max_m: float = 1.5,
) -> np.ndarray:
    return deproject_single_pixel_to_3d(
        pixel_xy,
        depth_u16,
        intr,
        depth_scale,
        search_radius_px=search_radius_px,
        z_min_m=z_min_m,
        z_max_m=z_max_m,
    )


def compute_guide_edge_length_3d(
    p0_int: tuple[int, int],
    p1_int: tuple[int, int],
    intr: rs.intrinsics,
    depth_u16: np.ndarray,
    depth_scale: float,
    *,
    plane: Optional[np.ndarray] = None,
    search_radius_px: int = 5,
) -> tuple[float, np.ndarray, np.ndarray, str]:
    """ガイドラインとして選択された斜辺の3D長さを計算する．"""
    if plane is not None:
        try:
            p0_cam = intersect_pixel_ray_with_plane(p0_int, intr, plane)
            p1_cam = intersect_pixel_ray_with_plane(p1_int, intr, plane)
            return float(np.linalg.norm(p1_cam - p0_cam)), p0_cam, p1_cam, "ransac_plane"
        except Exception as e:
            print(f"[WARN] guide edge length by RANSAC plane failed: {e}")
            print("[WARN] fallback to endpoint depth.")

    p0_cam = deproject_pixel_with_neighborhood(p0_int, depth_u16, intr, depth_scale, search_radius_px=search_radius_px)
    p1_cam = deproject_pixel_with_neighborhood(p1_int, depth_u16, intr, depth_scale, search_radius_px=search_radius_px)
    return float(np.linalg.norm(p1_cam - p0_cam)), p0_cam, p1_cam, "depth_endpoint"


# ============================================================
# Visualization
# ============================================================

def save_binary_target_overlay(
    binary_mask: np.ndarray,
    *,
    line_p0: tuple[int, int] | None = None,
    line_p1: tuple[int, int] | None = None,
    target_px: tuple[int, int] | None = None,
    save_path: Path,
) -> None:
    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    vis = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    if line_p0 is not None and line_p1 is not None:
        cv2.line(vis, line_p0, line_p1, (0, 0, 255), 3)
    if target_px is not None:
        cv2.circle(vis, target_px, 8, (0, 0, 255), -1)
        cv2.circle(vis, target_px, 13, (0, 0, 255), 2)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)
    print(f"[SAVE] binary target overlay: {save_path}")


def save_debug_overlay(
    rgb_bgr: np.ndarray,
    candidate_mask: np.ndarray,
    selected_mask: np.ndarray,
    line_p0: tuple[int, int],
    line_p1: tuple[int, int],
    target_px: tuple[int, int] | None,
    save_path: Path,
) -> None:
    vis = rgb_bgr.copy()
    cand = candidate_mask.astype(bool)
    vis[cand] = (0.5 * vis[cand] + 0.5 * np.array([0, 255, 255], dtype=np.float32)).astype(np.uint8)
    sel = selected_mask.astype(bool)
    vis[sel] = (0.4 * vis[sel] + 0.6 * np.array([0, 0, 255], dtype=np.float32)).astype(np.uint8)
    cv2.line(vis, line_p0, line_p1, (255, 0, 255), 4)
    cv2.circle(vis, line_p0, 5, (0, 255, 255), -1)
    cv2.circle(vis, line_p1, 5, (255, 255, 255), -1)
    if target_px is not None:
        cv2.circle(vis, target_px, 8, (0, 255, 0), -1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)
    print(f"[SAVE] RGB debug overlay: {save_path}")


def save_surface_points_overlay(
    rgb_bgr: np.ndarray,
    surface_pixels: np.ndarray,
    inlier_mask: Optional[np.ndarray],
    save_path: Path,
) -> None:
    vis = rgb_bgr.copy()
    pixels = np.asarray(surface_pixels, dtype=np.int32).reshape(-1, 2)
    for k, (u, v) in enumerate(pixels):
        if v < 0 or v >= vis.shape[0] or u < 0 or u >= vis.shape[1]:
            continue
        color = (0, 255, 0) if (inlier_mask is not None and k < len(inlier_mask) and bool(inlier_mask[k])) else (255, 0, 0)
        cv2.circle(vis, (int(u), int(v)), 1, color, -1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)
    print(f"[SAVE] surface points overlay: {save_path}")


def show_image(title: str, img: np.ndarray, wait: bool = True, delay_ms: int = 30) -> None:
    if img is None:
        print(f"[WARN] show_image: {title} is None")
        return
    cv2.imshow(title, img)
    if not wait:
        cv2.waitKey(delay_ms)
        return
    print(f"[INFO] showing {title}. Press any key or close the window.")
    while True:
        key = cv2.waitKey(delay_ms)
        if key != -1:
            break
        visible = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)
        if visible < 1:
            break
    cv2.destroyWindow(title)


def show_image_from_path(title: str, path: str | Path, wait: bool = True, delay_ms: int = 30) -> None:
    path = Path(path)
    if not path.exists():
        print(f"[WARN] image not found: {path}")
        return
    img = cv2.imread(str(path))
    if img is None:
        print(f"[WARN] failed to read image: {path}")
        return
    show_image(title, img, wait=wait, delay_ms=delay_ms)


# ============================================================
# Main function: same name as current code
# ============================================================

def run_capture_and_pca_depth_space(
    out_dir: str | Path = "captures_depth_space",
    width: int = 1280,
    height: int = 720,
    fps: int = 6,
    delta_depth_m: float = 0.03,
    # selection
    min_top_width_px: int = 80,
    min_area_px: int = 500,
    min_height_px: int = 100,
    horizontal_ratio_thr: float = 4.0,
    max_space_width_px: int = 520,
    max_space_area_px: int = 160000,
    min_band_width_px: int = 25,
    # vertical crop: 書籍が存在する高さ帯だけを候補にする
    crop_y_min: int = 120,
    crop_y_max: int = 400,
    # trim selected component
    trim_max_row_width_px: int = 180,
    trim_row_width_ratio_thr: float = 2.0,
    trim_bottom_search_ratio: float = 0.45,
    trim_min_keep_height_px: int = 60,
    # target
    target_y_from_space_top_px: int = 240,
    rotate_180: bool = False,
    # horizontal removal
    horizontal_kernel_w: int = 300,
    horizontal_kernel_h: int = 30,
    # surface plane
    surface_depth_tol_m: float = 0.05,
    surface_side_margin_px: int = 140,
    surface_gap_px: int = 5,
    surface_stride: int = 2,
    use_plane_correction: bool = True,
    plane_ransac_threshold_m: float = 0.01,
    plane_ransac_max_iter: int = 300,
    plane_min_inliers: int = 30,
    surface_clearance_m: float = 0.0,
    show_debug_images: bool = True,
    after_capture_callback: Optional[Callable[[], None]] = None,
):
    """
    Depth画像から収納スペースを推定する新版．

    特徴：
      - 書籍高さ帯だけを使うvertical cropを追加
      - 小さすぎる候補，高さ不足候補を除外
      - 斜辺側の境界をガイドライン化
      - 目標点は画像yで固定
      - 目標3D点は画素レイとRANSAC平面の交点で求める

    Returns
    -------
    angle_rad, first_target_cam, res
    """
    out_dir = Path(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    shot_dir = out_dir / ts
    shot_dir.mkdir(parents=True, exist_ok=True)

    # 1. RealSense capture
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    try:
        profile = pipe.start(cfg)
        align = rs.align(rs.stream.color)
        for _ in range(30):
            pipe.wait_for_frames()
        frames = pipe.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense frame取得に失敗しました．")

        depth_frame = depth_filter_like_viewer(depth_frame)
        rgb_bgr = np.asanyarray(color_frame.get_data())
        depth_u16 = np.asanyarray(depth_frame.get_data())
        color_prof = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intr = color_prof.get_intrinsics()
        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

        # ============================================================
        # 回転前RGB-D画像を横4分割し、
        # 左から3番目の領域をDepth認識対象外にする
        # ============================================================
        H_before, W_before = depth_u16.shape[:2]

        ignore_x_min = W_before // 2
        ignore_x_max = 3 * W_before // 4

        depth_u16 = depth_u16.copy()
        depth_u16[:, ignore_x_min:ignore_x_max] = 0

        print(
            f"[INFO] pre-rotation Depth ignore region: "
            f"x={ignore_x_min} ~ {ignore_x_max - 1}"
        )


    finally:
        try:
            pipe.stop()
        except Exception:
            pass

    # ============================================================
    # 撮影完了後コールバック
    # 画像データはすでに rgb_bgr / depth_u16 にコピー済みなので、
    # ここからロボットを動かしても1回目画像には影響しない
    # ============================================================
    if after_capture_callback is not None:
        print("[INFO] capture completed -> start robot motion")
        after_capture_callback()

    if rotate_180:
        rgb_bgr = cv2.rotate(rgb_bgr, cv2.ROTATE_180)
        depth_u16 = cv2.rotate(depth_u16, cv2.ROTATE_180)
        print("[WARN] rotate_180=True の場合，3D復元にはintrinsics補正が必要です．")

    H, W = depth_u16.shape[:2]
    cv2.imwrite(str(shot_dir / "rgb.png"), rgb_bgr)
    np.save(shot_dir / "depth_u16.npy", depth_u16)
    # クロップ範囲のRGB画像を保存
    rgb_crop = rgb_bgr[crop_y_min:crop_y_max, :]
    cv2.imwrite(str(shot_dir / "rgb_cropped.png"), rgb_crop)

    # クロップ範囲を元RGB上に可視化
    rgb_crop_vis = rgb_bgr.copy()
    cv2.line(rgb_crop_vis, (0, crop_y_min), (W - 1, crop_y_min), (0, 0, 255), 3)
    cv2.line(rgb_crop_vis, (0, crop_y_max), (W - 1, crop_y_max), (0, 0, 255), 3)
    cv2.imwrite(str(shot_dir / "rgb_crop_range_overlay.png"), rgb_crop_vis)


    # 2. Candidate mask with vertical crop
    depth_ref_m = estimate_depth_mode_in_y_range(
        depth_u16,
        depth_scale,
        crop_y_min=crop_y_min,
        crop_y_max=crop_y_max,
        z_min_m=0.2,
        z_max_m=1.2,
        bins=100,
    )
    print(f"[INFO] depth_ref_m = {depth_ref_m:.4f} m")
    print(f"[INFO] crop_y_min={crop_y_min}, crop_y_max={crop_y_max}")

    candidate_mask_raw = make_far_space_mask(
        depth_u16,
        depth_scale,
        depth_ref_m,
        delta_depth_m=delta_depth_m,
    )
    candidate_mask_raw = apply_vertical_crop_mask(
        candidate_mask_raw,
        crop_y_min=crop_y_min,
        crop_y_max=crop_y_max,
    )
    cv2.imwrite(str(shot_dir / "candidate_far_mask_raw_cropped.png"), candidate_mask_raw * 255)

    candidate_mask = remove_horizontal_structures(
        candidate_mask_raw,
        horizontal_kernel_w=horizontal_kernel_w,
        horizontal_kernel_h=horizontal_kernel_h,
    )

    cv2.imwrite(str(shot_dir / "candidate_far_mask_before_merge.png"), candidate_mask * 255)

    candidate_mask = merge_nearby_space_regions(
        candidate_mask,
        close_kernel_w=35,
        close_kernel_h=35,
        dilate_kernel_w=7,
        dilate_kernel_h=7,
        dilate_iterations=1,
    )

    cv2.imwrite(str(shot_dir / "candidate_far_mask.png"), candidate_mask * 255)
    candidate_mask = apply_vertical_crop_mask(
        candidate_mask,
        crop_y_min=crop_y_min,
        crop_y_max=crop_y_max,
    )
    cv2.imwrite(str(shot_dir / "candidate_far_mask.png"), candidate_mask * 255)

    # 3. Select storage space component
    selected = select_space_component(
        candidate_mask,
        min_area_px=min_area_px,
        min_height_px=min_height_px,
        horizontal_ratio_thr=horizontal_ratio_thr,
        max_width_px=max_space_width_px,
        max_area_px=max_space_area_px,
        min_band_width_px=min_band_width_px,
        prefer_top_open_shape=True,
    )
    if selected is None:
        save_binary_target_overlay(candidate_mask, save_path=shot_dir / "candidate_mask_no_target.png")
        raise RuntimeError("収納スペース候補が見つかりませんでした．")

    selected_mask_raw = selected["mask"]
    bbox_raw = selected["bbox"]
    cv2.imwrite(str(shot_dir / "selected_space_mask_raw.png"), selected_mask_raw * 255)
    print(
        f"[INFO] selected raw bbox={bbox_raw}, area={selected['area']}, "
        f"aspect_w_h={selected['aspect_w_h']:.3f}"
    )

    # 3.5 Trim selected component
    selected_mask = trim_bottom_excess_from_space_mask(
        selected_mask_raw,
        max_row_width_px=trim_max_row_width_px,
        row_width_ratio_thr=trim_row_width_ratio_thr,
        bottom_search_ratio=trim_bottom_search_ratio,
        min_keep_height_px=trim_min_keep_height_px,
    )
    selected_mask = apply_vertical_crop_mask(
        selected_mask,
        crop_y_min=crop_y_min,
        crop_y_max=crop_y_max,
    )
    cv2.imwrite(str(shot_dir / "selected_space_mask_cleaned.png"), selected_mask * 255)

    ys_clean, xs_clean = np.where(selected_mask > 0)
    if ys_clean.size == 0:
        raise RuntimeError("下側余分領域削除後，収納スペース候補が空になりました．")
    x0, x1 = int(xs_clean.min()), int(xs_clean.max())
    y0, y1 = int(ys_clean.min()), int(ys_clean.max())
    bbox = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
    print(f"[INFO] cleaned selected bbox={bbox}")

    selected.update({
        "bbox_raw": bbox_raw,
        "bbox": bbox,
        "mask_raw": selected_mask_raw,
        "mask": selected_mask,
        "area_raw": int(np.sum(selected_mask_raw > 0)),
        "area": int(np.sum(selected_mask > 0)),
        "center": (float(np.mean(xs_clean)), float(np.mean(ys_clean))),
        "width": int(bbox[2]),
        "height": int(bbox[3]),
        "aspect_w_h": float(bbox[2] / max(bbox[3], 1)),
    })
    cv2.imwrite(str(shot_dir / "selected_space_mask.png"), selected_mask * 255)

    # 4. Guide line from tilted boundary
    p0, p1, guide_boundary_pts, guide_side, guide_info = extract_tilted_boundary_line(
        selected_mask,
        default_side="left",
    )
    p0_int = (int(round(float(p0[0]))), int(round(float(p0[1]))))
    p1_int = (int(round(float(p1[0]))), int(round(float(p1[1]))))

    print(f"[INFO] guide_side = {guide_side}")
    print(f"[INFO] left_tilt_from_vertical_deg = {guide_info.get('left_tilt_from_vertical_deg')}")
    print(f"[INFO] right_tilt_from_vertical_deg = {guide_info.get('right_tilt_from_vertical_deg')}")

    dx = float(p1_int[0] - p0_int[0])
    dy = float(p1_int[1] - p0_int[1])
    angle_rad = float(np.arctan2(dy, dx))
    angle_deg = float(np.degrees(angle_rad))
    print(f"[INFO] selected boundary angle = {angle_deg:.2f} deg")

    # 5. Target pixel by fixed image height
    space_top_y = int(ys_clean.min())
    space_bottom_y = int(ys_clean.max())
    space_height_px = space_bottom_y - space_top_y + 1

    # 指定位置まで高さがない候補は使用しない
    if space_height_px <= target_y_from_space_top_px:
        raise RuntimeError(
            "収納スペース候補が見つかりませんでした．"
            f" target位置まで高さ不足: "
            f"height={space_height_px}px, "
            f"required>{target_y_from_space_top_px}px"
        )

    target_y_px = space_top_y + int(target_y_from_space_top_px)

    first_target_px_selected = select_target_pixel_on_line_by_y(
        p0_int,
        p1_int,
        target_y_px=target_y_px,
        width=W,
        height=H,
    )
    first_target_cam_raw = deproject_single_pixel_to_3d(
        first_target_px_selected,
        depth_u16,
        intr,
        depth_scale,
        search_radius_px=5,
        z_min_m=0.1,
        z_max_m=1.5,
    )
    print(f"[INFO] space_top_y={space_top_y}, space_bottom_y={space_bottom_y}")
    print(f"[INFO] target_y_from_space_top_px={target_y_from_space_top_px}")
    print(f"[INFO] first_target_px_selected={first_target_px_selected}")
    print("[INFO] first_target_cam_raw =", first_target_cam_raw)

    # 6. Surface plane and target 3D
    surface_points, surface_pixels = collect_book_surface_points_around_space(
        depth_u16,
        intr,
        depth_scale,
        bbox,
        depth_ref_m,
        depth_tol_m=surface_depth_tol_m,
        side_margin_px=surface_side_margin_px,
        gap_px=surface_gap_px,
        stride=surface_stride,
        z_min_m=0.1,
        z_max_m=1.5,
    )
    print(f"[INFO] surface_points = {surface_points.shape[0]}")

    plane = None
    inlier_mask = np.zeros((surface_points.shape[0],), dtype=bool)
    plane_correction_used = False
    first_target_cam = first_target_cam_raw.copy()

    if use_plane_correction and surface_points.shape[0] >= 3:
        plane, inlier_mask = fit_plane_ransac(
            surface_points,
            distance_threshold_m=plane_ransac_threshold_m,
            max_iter=plane_ransac_max_iter,
            min_inliers=plane_min_inliers,
            random_seed=0,
        )
        if plane is not None:
            try:
                first_target_cam = intersect_pixel_ray_with_plane(
                    first_target_px_selected,
                    intr,
                    plane,
                    surface_clearance_m=surface_clearance_m,
                )
                plane_correction_used = True
                print("[INFO] RANSAC plane =", plane)
                print("[INFO] first_target_cam by ray-plane intersection =", first_target_cam)
            except Exception as e:
                print(f"[WARN] plane correction failed: {e}")
                first_target_cam = fallback_correct_point_z_by_median_surface(
                    first_target_cam_raw,
                    surface_points,
                    surface_clearance_m=surface_clearance_m,
                )
                print("[WARN] fallback median Z correction used.")
        else:
            first_target_cam = fallback_correct_point_z_by_median_surface(
                first_target_cam_raw,
                surface_points,
                surface_clearance_m=surface_clearance_m,
            )
            print("[WARN] RANSAC plane not found. fallback median Z correction used.")
    elif use_plane_correction:
        print("[WARN] surface points are insufficient. raw target is used.")

    try:
        first_target_px_projected = project_cam_point_to_pixel(first_target_cam, intr)
    except Exception as e:
        print(f"[WARN] failed to project first_target_cam: {e}")
        first_target_px_projected = first_target_px_selected

    print("[INFO] first_target_px selected  =", first_target_px_selected)
    print("[INFO] first_target_px projected =", first_target_px_projected)

    # 6.5 Edge length
    guide_edge_length_m = None
    guide_edge_length_mm = None
    guide_edge_p0_cam = None
    guide_edge_p1_cam = None
    guide_edge_length_source = None
    try:
        guide_edge_length_m, guide_edge_p0_cam, guide_edge_p1_cam, guide_edge_length_source = compute_guide_edge_length_3d(
            p0_int,
            p1_int,
            intr,
            depth_u16,
            depth_scale,
            plane=plane if plane_correction_used else None,
            search_radius_px=5,
        )
        guide_edge_length_mm = float(guide_edge_length_m * 1000.0)
        print(
            f"[INFO] guide_edge_length = {guide_edge_length_m:.4f} m "
            f"({guide_edge_length_mm:.1f} mm), source={guide_edge_length_source}"
        )
    except Exception as e:
        print(f"[WARN] guide edge length estimation failed: {e}")

    # 7. Debug images
    selected_mask_target_overlay_path = shot_dir / "selected_mask_target_overlay.png"
    candidate_mask_target_overlay_path = shot_dir / "candidate_mask_target_overlay.png"
    overlay_path = shot_dir / "depth_space_overlay.png"
    surface_overlay_path = shot_dir / "surface_points_overlay.png"

    save_binary_target_overlay(
        selected_mask,
        line_p0=p0_int,
        line_p1=p1_int,
        target_px=first_target_px_projected,
        save_path=selected_mask_target_overlay_path,
    )
    save_binary_target_overlay(
        candidate_mask,
        line_p0=p0_int,
        line_p1=p1_int,
        target_px=first_target_px_projected,
        save_path=candidate_mask_target_overlay_path,
    )
    save_debug_overlay(
        rgb_bgr,
        candidate_mask,
        selected_mask,
        p0_int,
        p1_int,
        first_target_px_projected,
        overlay_path,
    )
    save_surface_points_overlay(
        rgb_bgr,
        surface_pixels,
        inlier_mask if surface_points.shape[0] == inlier_mask.shape[0] else None,
        surface_overlay_path,
    )

    if show_debug_images:
        show_image_from_path("selected_mask_target_overlay", selected_mask_target_overlay_path, wait=True)
        show_image_from_path("candidate_mask_target_overlay", candidate_mask_target_overlay_path, wait=True)
        show_image_from_path("depth_space_overlay", overlay_path, wait=True)
        show_image_from_path("surface_points_overlay", surface_overlay_path, wait=True)
        cv2.destroyAllWindows()

    # 8. res
    res = {
        "line_p0": p0_int,
        "line_p1": p1_int,
        "sub_line_p0": p0_int,
        "sub_line_p1": p1_int,
        "right_is_tilted": False,
        "is_right_half": False,
        "pair_indices": None,
        "centers": None,

        "guide_side": guide_side,
        "guide_info": guide_info,
        "left_tilt_from_vertical_deg": guide_info.get("left_tilt_from_vertical_deg"),
        "right_tilt_from_vertical_deg": guide_info.get("right_tilt_from_vertical_deg"),

        "space_bbox": bbox,
        "space_center_px": selected["center"],
        "space_area": selected["area"],
        "space_aspect_w_h": selected["aspect_w_h"],

        "first_target_px": first_target_px_projected,
        "first_target_px_selected": first_target_px_selected,
        "first_target_px_projected": first_target_px_projected,
        "first_target_cam_raw": first_target_cam_raw,
        "first_target_cam": first_target_cam,
        "final_target_cam": first_target_cam.copy(),

        "depth_ref_m": depth_ref_m,
        "delta_depth_m": delta_depth_m,
        "crop_y_min": crop_y_min,
        "crop_y_max": crop_y_max,
        "target_y_from_space_top_px": target_y_from_space_top_px,

        "use_plane_correction": use_plane_correction,
        "plane_correction_used": plane_correction_used,
        "plane": plane.tolist() if plane is not None else None,
        "surface_points_count": int(surface_points.shape[0]),
        "surface_inliers_count": int(np.sum(inlier_mask)) if inlier_mask is not None else 0,
        "surface_clearance_m": surface_clearance_m,

        "guide_edge_length_m": guide_edge_length_m,
        "guide_edge_length_mm": guide_edge_length_mm,
        "guide_edge_length_source": guide_edge_length_source,
        "guide_edge_p0_cam": guide_edge_p0_cam.tolist() if guide_edge_p0_cam is not None else None,
        "guide_edge_p1_cam": guide_edge_p1_cam.tolist() if guide_edge_p1_cam is not None else None,

        "overlay_path": str(overlay_path),
        "selected_mask_target_overlay_path": str(selected_mask_target_overlay_path),
        "candidate_mask_target_overlay_path": str(candidate_mask_target_overlay_path),
        "surface_points_overlay_path": str(surface_overlay_path),
        "shot_dir": str(shot_dir),
    }

    print("[INFO] first_target_cam_raw  =", first_target_cam_raw)
    print("[INFO] first_target_cam      =", first_target_cam)
    print("[INFO] plane_correction_used =", plane_correction_used)
    print("[INFO] files saved under     =", shot_dir)

    return angle_rad, first_target_cam, res
