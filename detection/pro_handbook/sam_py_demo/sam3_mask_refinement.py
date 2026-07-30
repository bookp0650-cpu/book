"""Conservative connected-component refinement for a selected SAM3 book mask.

The policy is deliberately asymmetric: keeping a questionable component is
preferred to deleting part of a book.  The anchor is selected from OCR overlap,
not merely by component area, and every safety-check failure returns the raw
mask.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


DEFAULT_PARAMETERS = {
    "max_removable_area_ratio": 0.003,
    "max_removable_anchor_area_ratio": 0.012,
    "min_distance_anchor_diag_ratio": 0.08,
    "min_b_gap_anchor_span_ratio": 0.15,
    "max_a_extension_anchor_span_ratio": 0.20,
    "min_depth_difference_raw": 80.0,
    "min_removal_evidence_count": 4,
    "max_total_removed_ratio": 0.025,
    "max_axis_change_deg": 5.0,
    "min_final_area_px": 200,
    "min_valid_depth_count": 30,
}


def _json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _polygon_mask(shape: tuple[int, int], polygon) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    if polygon is None:
        return out.astype(bool)
    pts = np.asarray(polygon, np.float64).reshape(-1, 2)
    if len(pts) >= 3 and np.isfinite(pts).all():
        cv2.fillPoly(out, [np.rint(pts).astype(np.int32)], 1)
    return out.astype(bool)


def _axis_from_points(points: np.ndarray):
    points = np.asarray(points, np.float64).reshape(-1, 2)
    if len(points) < 2 or not np.isfinite(points).all():
        return None
    centered = points - points.mean(axis=0)
    _, vectors, _ = cv2.PCACompute2(centered, mean=None)
    axis = vectors[0].astype(np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    return axis


def _axis_angle(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) < 2:
        return None
    axis = _axis_from_points(np.column_stack([xs, ys]))
    if axis is None:
        return None
    return float(math.atan2(axis[1], axis[0]))


def _undirected_angle_difference_deg(left, right) -> float | None:
    if left is None or right is None:
        return None
    delta = abs(math.degrees(left - right)) % 180.0
    return float(min(delta, 180.0 - delta))


def _component_distance(component: np.ndarray, anchor: np.ndarray) -> float:
    # Distance transforms give exact pixel distance and avoid O(N*M) searches.
    distance = cv2.distanceTransform((~anchor).astype(np.uint8), cv2.DIST_L2, 5)
    values = distance[component]
    return float(values.min()) if values.size else float("inf")


def _projection_range(points: np.ndarray, center: np.ndarray, axis: np.ndarray):
    projected = (points - center) @ axis
    return float(projected.min()), float(projected.max())


def _depth_stats(depth: np.ndarray | None, mask: np.ndarray):
    if depth is None:
        return 0, None, None
    values = np.asarray(depth)[mask]
    valid = np.isfinite(values) & (values > 0)
    values = values[valid].astype(np.float64)
    if not values.size:
        return 0, None, None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return int(values.size), median, mad


def _overlay(rgb_bgr: np.ndarray, mask: np.ndarray, color, alpha=0.42):
    out = rgb_bgr.copy()
    colored = out.copy()
    colored[mask] = color
    return cv2.addWeighted(out, 1.0 - alpha, colored, alpha, 0)


def refine_selected_sam3_mask(
    raw_mask,
    *,
    ocr_polygon,
    depth_raw=None,
    rgb_bgr=None,
    output_dir: str | Path | None = None,
    mode: str = "refine_and_median_flatten",
    parameters: dict | None = None,
):
    """Return ``(mask, result, metrics)`` using a conservative delete policy."""
    params = {**DEFAULT_PARAMETERS, **(parameters or {})}
    raw = np.asarray(raw_mask, dtype=bool)
    if raw.ndim != 2 or not raw.any():
        raise ValueError("raw selected SAM3 mask must be a non-empty 2D mask")
    depth = None if depth_raw is None else np.asarray(depth_raw)
    if depth is not None and depth.shape != raw.shape:
        raise ValueError(f"depth/mask shape mismatch: {depth.shape}/{raw.shape}")

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        raw.astype(np.uint8), connectivity=8
    )
    component_ids = list(range(1, n_labels))
    raw_area = int(raw.sum())
    ocr_mask = _polygon_mask(raw.shape, ocr_polygon)
    ocr_area = int(ocr_mask.sum())
    ocr_pts = np.asarray(ocr_polygon if ocr_polygon is not None else [], np.float64)
    if ocr_pts.size:
        ocr_pts = ocr_pts.reshape(-1, 2)
        ocr_center = ocr_pts.mean(axis=0)
    else:
        ocr_center = None

    # OCR polygon PCA is the primary direction; its long edge is the first
    # fallback. Anchor-mask PCA is resolved after anchor selection.
    axis = _axis_from_points(ocr_pts) if len(ocr_pts) >= 3 else None
    axis_source = "ocr_polygon_pca" if axis is not None else None
    if axis is None and len(ocr_pts) >= 2:
        edges = np.roll(ocr_pts, -1, axis=0) - ocr_pts
        edge = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
        if np.linalg.norm(edge) > 0:
            axis = edge / np.linalg.norm(edge)
            axis_source = "ocr_polygon_long_edge"

    provisional = []
    for cid in component_ids:
        cmask = labels == cid
        overlap = int(np.count_nonzero(cmask & ocr_mask))
        center_inside = False
        center_distance = None
        if ocr_center is not None:
            u, v = np.rint(ocr_center).astype(int)
            center_inside = bool(
                0 <= v < raw.shape[0] and 0 <= u < raw.shape[1] and cmask[v, u]
            )
            center_distance = float(np.linalg.norm(centroids[cid] - ocr_center))
        provisional.append(
            {
                "id": cid,
                "overlap": overlap,
                "overlap_ratio": overlap / max(ocr_area, 1),
                "center_inside": center_inside,
                "center_distance": center_distance,
                "area": int(stats[cid, cv2.CC_STAT_AREA]),
            }
        )

    # Lexicographic OCR evidence, with area only as the last tie breaker.
    candidates = sorted(
        provisional,
        key=lambda item: (
            item["overlap"] > 0,
            item["center_inside"],
            item["overlap_ratio"],
            -(item["center_distance"] if item["center_distance"] is not None else 1e12),
            item["area"],
        ),
        reverse=True,
    )
    anchor_id = candidates[0]["id"] if candidates else None
    anchor_confident = bool(
        candidates
        and (candidates[0]["overlap"] > 0 or candidates[0]["center_inside"])
    )
    fallback_reason = None
    if not anchor_confident:
        fallback_reason = "anchor_not_supported_by_ocr_overlap_or_center"
        # Keep processing metrics, but never delete on this uncertain anchor.
        anchor_id = max(component_ids, key=lambda cid: stats[cid, cv2.CC_STAT_AREA])

    anchor = labels == anchor_id
    if axis is None:
        ys, xs = np.where(anchor)
        axis = _axis_from_points(np.column_stack([xs, ys]))
        axis_source = "anchor_mask_pca" if axis is not None else "disabled"
    b_axis = None if axis is None else np.asarray([-axis[1], axis[0]])
    anchor_center = centroids[anchor_id].astype(np.float64)
    anchor_points = np.column_stack(np.where(anchor)[::-1]).astype(np.float64)
    anchor_diag = float(
        math.hypot(
            stats[anchor_id, cv2.CC_STAT_WIDTH],
            stats[anchor_id, cv2.CC_STAT_HEIGHT],
        )
    )
    if axis is not None:
        anchor_a = _projection_range(anchor_points, anchor_center, axis)
        anchor_b = _projection_range(anchor_points, anchor_center, b_axis)
    else:
        anchor_a = anchor_b = None
    _, anchor_depth, _ = _depth_stats(depth, anchor)

    kept = [anchor_id]
    removed = []
    metrics = []
    for cid in component_ids:
        cmask = labels == cid
        area = int(stats[cid, cv2.CC_STAT_AREA])
        x, y, w, h = [int(value) for value in stats[cid, :4]]
        overlap = int(np.count_nonzero(cmask & ocr_mask))
        valid_count, depth_median, depth_mad = _depth_stats(depth, cmask)
        distance = 0.0 if cid == anchor_id else _component_distance(cmask, anchor)
        points = np.column_stack(np.where(cmask)[::-1]).astype(np.float64)
        a_gap = b_distance = None
        outside_a_extension = off_axis_b = False
        if axis is not None:
            crange_a = _projection_range(points, anchor_center, axis)
            crange_b = _projection_range(points, anchor_center, b_axis)
            a_gap = max(
                0.0,
                anchor_a[0] - crange_a[1],
                crange_a[0] - anchor_a[1],
            )
            b_distance = max(
                0.0,
                anchor_b[0] - crange_b[1],
                crange_b[0] - anchor_b[1],
            )
            a_span = max(anchor_a[1] - anchor_a[0], 1.0)
            b_span = max(anchor_b[1] - anchor_b[0], 1.0)
            outside_a_extension = a_gap > params[
                "max_a_extension_anchor_span_ratio"
            ] * a_span
            off_axis_b = b_distance > params["min_b_gap_anchor_span_ratio"] * b_span
        area_ratio = area / max(raw_area, 1)
        anchor_area_ratio = area / max(int(anchor.sum()), 1)
        small = (
            area_ratio <= params["max_removable_area_ratio"]
            and anchor_area_ratio <= params["max_removable_anchor_area_ratio"]
        )
        separated = distance > max(
            3.0, params["min_distance_anchor_diag_ratio"] * anchor_diag
        )
        depth_different = bool(
            anchor_depth is not None
            and depth_median is not None
            and abs(depth_median - anchor_depth) >= params["min_depth_difference_raw"]
        )
        evidence = {
            "very_small": small,
            "clearly_separated": separated,
            "outside_a_extension": outside_a_extension,
            "large_b_axis_distance": off_axis_b,
            "depth_different": depth_different,
            "outside_anchor_bbox": bool(
                x + w - 1 < stats[anchor_id, cv2.CC_STAT_LEFT]
                or x > stats[anchor_id, cv2.CC_STAT_LEFT]
                + stats[anchor_id, cv2.CC_STAT_WIDTH]
                - 1
                or y + h - 1 < stats[anchor_id, cv2.CC_STAT_TOP]
                or y > stats[anchor_id, cv2.CC_STAT_TOP]
                + stats[anchor_id, cv2.CC_STAT_HEIGHT]
                - 1
            ),
        }
        evidence_count = sum(evidence.values())
        remove = bool(
            cid != anchor_id
            and anchor_confident
            and small
            and separated
            and overlap == 0
            and not provisional[cid - 1]["center_inside"]
            and (outside_a_extension or off_axis_b)
            and evidence_count >= params["min_removal_evidence_count"]
        )
        if remove:
            removed.append(cid)
            decision = "remove"
            reason = "multiple_high_confidence_isolation_signals"
        else:
            if cid not in kept:
                kept.append(cid)
            decision = "keep"
            reason = (
                "anchor_component"
                if cid == anchor_id
                else "insufficient_evidence_for_safe_removal"
            )
        metrics.append(
            {
                "component_id": cid,
                "area_px": area,
                "raw_mask_area_ratio": area_ratio,
                "anchor_area_ratio": anchor_area_ratio,
                "bbox_xyxy": [x, y, x + w - 1, y + h - 1],
                "centroid_uv": centroids[cid].astype(float).tolist(),
                "ocr_overlap_px": overlap,
                "ocr_overlap_ratio": overlap / max(ocr_area, 1),
                "contains_ocr_center": provisional[cid - 1]["center_inside"],
                "anchor_min_distance_px": distance,
                "a_axis_gap_px": a_gap,
                "b_axis_distance_px": b_distance,
                "valid_depth_count": valid_count,
                "depth_median_raw": depth_median,
                "depth_mad_raw": depth_mad,
                "touches_image_boundary": bool(
                    x == 0 or y == 0 or x + w == raw.shape[1] or y + h == raw.shape[0]
                ),
                "evidence": evidence,
                "evidence_count": evidence_count,
                "decision": decision,
                "reason": reason,
            }
        )

    candidate_refined = np.isin(labels, kept)
    warnings = []
    raw_overlap = int(np.count_nonzero(raw & ocr_mask))
    refined_overlap = int(np.count_nonzero(candidate_refined & ocr_mask))
    removed_area = int(raw_area - candidate_refined.sum())
    axis_change = _undirected_angle_difference_deg(
        _axis_angle(raw), _axis_angle(candidate_refined)
    )
    valid_final = (
        int(np.count_nonzero(candidate_refined & (depth > 0)))
        if depth is not None
        else int(candidate_refined.sum())
    )
    safety_failures = []
    if not np.all(candidate_refined[anchor]):
        safety_failures.append("anchor_not_fully_preserved")
    if refined_overlap != raw_overlap:
        safety_failures.append("ocr_overlap_would_be_removed")
    if not candidate_refined.any():
        safety_failures.append("refined_mask_empty")
    if removed_area / max(raw_area, 1) > params["max_total_removed_ratio"]:
        safety_failures.append("removed_area_ratio_too_large")
    if axis_change is not None and axis_change > params["max_axis_change_deg"]:
        safety_failures.append("long_axis_changed_too_much")
    if int(candidate_refined.sum()) < params["min_final_area_px"]:
        safety_failures.append("refined_mask_too_small")
    if depth is not None and valid_final < params["min_valid_depth_count"]:
        safety_failures.append("insufficient_valid_depth")
    if fallback_reason:
        safety_failures.append(fallback_reason)

    fallback = bool(safety_failures)
    if fallback:
        refined = raw.copy()
        fallback_reason = ";".join(safety_failures)
        kept = component_ids
        removed = []
        for metric in metrics:
            metric["decision"] = "keep"
            metric["reason"] = "raw_mask_fallback"
    else:
        refined = candidate_refined
    no_op = bool(np.array_equal(raw, refined))

    result = {
        "mode": mode,
        "raw_component_count": len(component_ids),
        "anchor_component_id": int(anchor_id),
        "anchor_confident": anchor_confident,
        "axis_source": axis_source,
        "axis_uv": None if axis is None else axis.astype(float).tolist(),
        "kept_component_ids": [int(value) for value in kept],
        "removed_component_ids": [int(value) for value in removed],
        "raw_mask_area_px": raw_area,
        "refined_mask_area_px": int(refined.sum()),
        "removed_area_px": int(raw_area - refined.sum()),
        "no_op": no_op,
        "raw_and_refined_mask_equal": no_op,
        "fallback_to_raw_mask": fallback,
        "fallback_reason": fallback_reason,
        "ocr_overlap_before_px": raw_overlap,
        "ocr_overlap_after_px": int(np.count_nonzero(refined & ocr_mask)),
        "axis_change_deg": axis_change,
        "parameters": params,
        "warnings": warnings,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "selected_mask_raw.png"), raw.astype(np.uint8) * 255)
        cv2.imwrite(
            str(out / "selected_mask_refined.png"), refined.astype(np.uint8) * 255
        )
        cv2.imwrite(str(out / "anchor_component.png"), anchor.astype(np.uint8) * 255)
        cv2.imwrite(
            str(out / "components_kept.png"), np.isin(labels, kept).astype(np.uint8) * 255
        )
        cv2.imwrite(
            str(out / "components_removed.png"),
            (raw & ~refined).astype(np.uint8) * 255,
        )
        # 16-bit IDs preserve exact labels; a colored PNG makes them inspectable.
        cv2.imwrite(str(out / "component_ids.png"), labels.astype(np.uint16))
        colored = np.zeros((*raw.shape, 3), np.uint8)
        for cid in component_ids:
            color = (
                int((37 * cid) % 255),
                int((97 * cid) % 255),
                int((173 * cid) % 255),
            )
            colored[labels == cid] = color
        if rgb_bgr is not None:
            rgb = np.asarray(rgb_bgr)
            cv2.imwrite(
                str(out / "selected_mask_raw_overlay.png"),
                _overlay(rgb, raw, (0, 255, 0)),
            )
            cv2.imwrite(
                str(out / "selected_mask_refined_overlay.png"),
                _overlay(rgb, refined, (0, 255, 0)),
            )
            cv2.imwrite(str(out / "selected_mask_components.png"), _overlay(rgb, raw, (0, 255, 255)))
            axis_vis = rgb.copy()
            if axis is not None:
                center = (
                    np.asarray(ocr_center)
                    if ocr_center is not None
                    else anchor_center
                )
                span = max(raw.shape)
                p0 = np.rint(center - axis * span).astype(int)
                p1 = np.rint(center + axis * span).astype(int)
                cv2.line(axis_vis, tuple(p0), tuple(p1), (255, 0, 255), 2)
            if ocr_pts.size:
                cv2.polylines(
                    axis_vis,
                    [np.rint(ocr_pts).astype(np.int32)],
                    True,
                    (0, 255, 255),
                    2,
                )
            cv2.imwrite(str(out / "ocr_axis_overlay.png"), axis_vis)
        else:
            cv2.imwrite(str(out / "selected_mask_components.png"), colored)
        _json(out / "component_metrics.json", metrics)
        _json(out / "mask_refinement_result.json", result)
    return refined, result, metrics
