from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List
import numpy as np


def find_target_point_longitudinal(
    pts: np.ndarray,
    pc1: np.ndarray,
    target_height_m: float = 0.075,
    candidate_tolerance_m: float = 0.003,
    bottom_percentile: float = 0.5,
    fallback_tolerances_m: Tuple[float, ...] = (0.005, 0.008),
    nearest_max_distance_m: float = 0.012,
) -> Dict[str, Any]:
    """書籍底面から長手方向へ一定距離の実点を目標点として選ぶ。

    ``pc1`` を書籍長手方向として正規化し、倒立設置されたカメラで
    実空間の書籍上方向に対応する camera ``+Y`` へ符号をそろえる。
    RANSAC後点群の長手方向射影の下位パーセンタイルを堅牢な底面とし、
    底面から ``target_height_m`` の候補帯にある実点のうち camera ``X``
    が最小の点を返す。座標系は camera、単位は meter。
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"pts shape must be (N,3), got {pts.shape}")
    if pts.shape[0] == 0:
        return {"ok": False, "reason": "no_points", "target_m": None}
    if not np.all(np.isfinite(pts)):
        return {"ok": False, "reason": "non_finite_points", "target_m": None}

    pc1_raw = np.asarray(pc1, dtype=np.float64).reshape(3)
    pc1_norm = float(np.linalg.norm(pc1_raw))
    if not np.isfinite(pc1_norm) or pc1_norm < 1e-12:
        return {"ok": False, "reason": "invalid_pc1", "target_m": None}
    book_up = pc1_raw / pc1_norm
    if book_up[1] < 0.0:
        book_up = -book_up

    height = float(target_height_m)
    tolerance = float(candidate_tolerance_m)
    percentile = float(bottom_percentile)
    nearest_limit = float(nearest_max_distance_m)
    if height < 0.0:
        raise ValueError(f"target_height_m must be non-negative, got {height}")
    if tolerance <= 0.0:
        raise ValueError(
            f"candidate_tolerance_m must be positive, got {tolerance}"
        )
    if not 0.0 <= percentile < 50.0:
        raise ValueError(
            f"bottom_percentile must be in [0, 50), got {percentile}"
        )
    if nearest_limit <= 0.0:
        raise ValueError(
            f"nearest_max_distance_m must be positive, got {nearest_limit}"
        )

    origin = pts.mean(axis=0)
    longitudinal = (pts - origin) @ book_up
    bottom = float(np.percentile(longitudinal, percentile))
    top = float(np.percentile(longitudinal, 100.0 - percentile))
    target_longitudinal = bottom + height
    if target_longitudinal > top:
        return {
            "ok": False,
            "reason": "target_height_exceeds_book_extent",
            "target_m": None,
            "pc1_raw": pc1_raw,
            "book_up_vector": book_up,
            "origin_m": origin,
            "bottom_longitudinal_m": bottom,
            "top_longitudinal_m": top,
            "target_longitudinal_m": target_longitudinal,
            "target_height_m": height,
        }

    requested_tolerances = [tolerance]
    requested_tolerances.extend(float(value) for value in fallback_tolerances_m)
    tolerances = []
    for value in requested_tolerances:
        if value > 0.0 and value not in tolerances:
            tolerances.append(value)
    tolerances.sort()

    distance = np.abs(longitudinal - target_longitudinal)
    candidate_idx = np.empty(0, dtype=np.int64)
    used_tolerance = None
    fallback_method = "none"
    for band in tolerances:
        candidate_idx = np.flatnonzero(distance <= band)
        if candidate_idx.size:
            used_tolerance = float(band)
            fallback_method = (
                "primary_band"
                if np.isclose(band, tolerance)
                else "expanded_band"
            )
            break

    nearest_distance = float(np.min(distance))
    if candidate_idx.size == 0 and nearest_distance <= nearest_limit:
        # Include points effectively on the same nearest longitudinal slice,
        # then preserve the existing camera-X minimum edge selection.
        nearest_band = min(nearest_distance + 0.0005, nearest_limit)
        candidate_idx = np.flatnonzero(distance <= nearest_band)
        fallback_method = "nearest_longitudinal_point"

    if candidate_idx.size == 0:
        return {
            "ok": False,
            "reason": "no_longitudinal_target_candidates",
            "target_m": None,
            "pc1_raw": pc1_raw,
            "book_up_vector": book_up,
            "origin_m": origin,
            "bottom_longitudinal_m": bottom,
            "top_longitudinal_m": top,
            "target_longitudinal_m": target_longitudinal,
            "target_height_m": height,
            "candidate_tolerance_m": tolerance,
            "fallback_tolerances_m": tolerances[1:],
            "nearest_distance_m": nearest_distance,
            "nearest_max_distance_m": nearest_limit,
        }

    local_x = pts[candidate_idx, 0]
    target_index = int(candidate_idx[int(np.argmin(local_x))])
    selected_longitudinal = float(longitudinal[target_index])
    target = pts[target_index].copy()
    return {
        "ok": True,
        "reason": "ok",
        "target_m": target,
        "target_index": target_index,
        "pc1_raw": pc1_raw,
        "book_up_vector": book_up,
        "origin_m": origin,
        "bottom_method": "longitudinal_percentile",
        "bottom_percentile": percentile,
        "bottom_longitudinal_m": bottom,
        "top_longitudinal_m": top,
        "target_longitudinal_m": target_longitudinal,
        "selected_longitudinal_m": selected_longitudinal,
        "selected_height_from_bottom_m": selected_longitudinal - bottom,
        "target_height_m": height,
        "candidate_tolerance_m": tolerance,
        "used_tolerance_m": used_tolerance,
        "candidate_count": int(candidate_idx.size),
        "fallback_method": fallback_method,
        "nearest_distance_m": nearest_distance,
        "nearest_max_distance_m": nearest_limit,
    }


def find_target_point(
    pts: np.ndarray,                  # (N,3) [m]
    y_offset_m: float = 0.1,           # 100 mm（旧方式・比較用）
    y_band_half_m: float = 0.003,      # ±3 mm
) -> Dict[str, Any]:
    """
    2)
    - y最大点A, y最小点C を見つける
    - y=y_min+100 mm 近傍（±3 mm）にある点のうち x が最小の点を
      target として返す（旧方式・比較用）

    返り値:
      {
        "ok": bool,
        "reason": str,
        "h_m": float | None,
        "y_min_m": float | None,
        "y_max_m": float | None,
        "target_m": np.ndarray(3,) | None,
        "num_candidates": int
      }
    """
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"pts shape must be (N,3), got {pts.shape}")
    if pts.shape[0] == 0:
        return {"ok": False, "reason": "no_points", "target_m": None, "h_m": None}

    ys = pts[:, 1]
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))

    y0 = y_min + float(y_offset_m)
    band = float(y_band_half_m)

    cand_idx = np.where(np.abs(ys - y0) <= band)[0]
    if cand_idx.size == 0:
        return {
            "ok": False,
            "target_m": None,
            "num_candidates": 0,
        }

    # x が最小の点
    xs = pts[cand_idx, 0]
    i = cand_idx[int(np.argmin(xs))]
    target = pts[i].copy()

    return {
        "ok": True,
        "target_m": target,
        "num_candidates": int(cand_idx.size),
    }
