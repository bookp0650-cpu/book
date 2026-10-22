#!/usr/bin/env python3
"""Deletion-only post-processing for an OCR-selected book mask.

This module deliberately knows nothing about GT masks.  It receives the mask
that has already been selected by the existing OCR/SAM matcher and only removes
pixels from it.  Every safety rejection returns the original mask.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.ndimage import median_filter


METHODS = (
    "baseline",
    "remove_islands",
    "morphology",
    "ocr_axis_ransac",
    "ocr_axis_smooth",
)

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "method": "baseline",
    "ocr_aspect_ratio_min": 1.5,
    "orientation_outlier_deg": 15.0,
    "boundary_margin_ratio": 0.10,
    "median_window": 31,
    "min_area_retention": 0.80,
    "min_width_retention": 0.85,
    "save_debug_images": True,
}


@dataclass(frozen=True)
class PostprocessResult:
    mask: np.ndarray
    info: dict[str, Any]


def normalize_postprocess_config(value: dict[str, Any] | None) -> dict[str, Any]:
    """Validate one scalar post-processing configuration."""
    cfg = dict(DEFAULT_CONFIG)
    if value:
        cfg.update(value)

    cfg["enabled"] = bool(cfg["enabled"])
    cfg["method"] = str(cfg["method"])
    if cfg["method"] not in METHODS:
        raise ValueError(f"unknown postprocess method: {cfg['method']!r}")

    for key in (
        "ocr_aspect_ratio_min",
        "orientation_outlier_deg",
        "boundary_margin_ratio",
        "min_area_retention",
        "min_width_retention",
    ):
        cfg[key] = float(cfg[key])
        if not math.isfinite(cfg[key]):
            raise ValueError(f"postprocess.{key} must be finite")

    if cfg["ocr_aspect_ratio_min"] <= 1.0:
        raise ValueError("postprocess.ocr_aspect_ratio_min must be > 1")
    if not 0.0 < cfg["orientation_outlier_deg"] <= 90.0:
        raise ValueError("postprocess.orientation_outlier_deg must be in (0, 90]")
    if cfg["boundary_margin_ratio"] < 0.0:
        raise ValueError("postprocess.boundary_margin_ratio must be >= 0")
    for key in ("min_area_retention", "min_width_retention"):
        if not 0.0 < cfg[key] <= 1.0:
            raise ValueError(f"postprocess.{key} must be in (0, 1]")

    cfg["median_window"] = int(cfg["median_window"])
    if cfg["median_window"] < 3 or cfg["median_window"] % 2 == 0:
        raise ValueError("postprocess.median_window must be an odd integer >= 3")
    cfg["save_debug_images"] = bool(cfg["save_debug_images"])
    return cfg


def _unrotate_poly(poly: Any, angle: int, width: int, height: int) -> np.ndarray:
    """Match OCR.only_one_tilted.unrotate_poly_to_original exactly."""
    points = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    angle = int(angle) % 360
    if angle == 90:
        xo, yo = y, (height - 1) - x
    elif angle == 180:
        xo, yo = (width - 1) - x, (height - 1) - y
    elif angle == 270:
        xo, yo = (width - 1) - y, x
    else:
        xo, yo = x, y
    return np.stack([xo, yo], axis=1).astype(np.float64)


def load_linked_ocr_polygons(
    shot_dir: str | Path,
    selected_mask_index: int,
    image_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    """Load only OCR polygons assigned by the unchanged matcher to one mask."""
    shot_dir = Path(shot_dir)
    debug_path = shot_dir / "text_mask_iou_debug.json"
    if not debug_path.is_file():
        # case_artifacts uses this descriptive name for the byte-copied matcher
        # debug JSON.  This is an input filename alias, not a matching rerun.
        debug_path = shot_dir / "ocr_sam3_matching.json"
    ocr_path = shot_dir / "ocr_result.json"
    if not debug_path.is_file() or not ocr_path.is_file():
        return []

    debug = json.loads(debug_path.read_text(encoding="utf-8"))
    ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
    target_name = f"mask_{int(selected_mask_index)}"
    height, width = int(image_shape[0]), int(image_shape[1])
    forced_angle = int(debug.get("forced_angle", 0) or 0)
    result: list[dict[str, Any]] = []

    assignments = debug.get("assignments") or []
    if not assignments:
        # Early three-angle artifacts stored only the angle-separated matcher
        # rows.  Flatten them in the same order as the production matcher and
        # retain the angle key so the corresponding OCR polygon is selected.
        assignments = [
            {**assignment, "source_angle": source_angle}
            for source_angle, angle_assignments in (
                debug.get("per_angle_assignments") or {}
            ).items()
            for assignment in (angle_assignments or [])
        ]

    for assignment in assignments:
        if assignment.get("matched") != target_name:
            continue
        try:
            index = int(assignment["ocr_index"]) - 1
        except (KeyError, TypeError, ValueError):
            continue
        if index < 0:
            continue

        source_angle = assignment.get("source_angle")
        payload = ocr
        if source_angle is not None:
            payload = (
                (ocr.get("per_angle_ocr") or {}).get("angles", {}).get(source_angle)
                or payload
            )
        polygons = payload.get("dt_polys") or []
        texts = payload.get("rec_texts") or []
        if index >= len(polygons):
            continue
        try:
            polygon = _unrotate_poly(
                polygons[index], forced_angle, width=width, height=height
            )
        except Exception:
            continue
        if polygon.shape[0] < 3 or not np.all(np.isfinite(polygon)):
            continue
        result.append(
            {
                "polygon": polygon,
                "center": polygon.mean(axis=0),
                "text": str(texts[index]) if index < len(texts) else str(
                    assignment.get("text", "")
                ),
                "source_angle": source_angle,
                "ocr_index": index + 1,
            }
        )
    return result


def _polygon_axis(item: dict[str, Any]) -> dict[str, Any] | None:
    polygon = np.asarray(item["polygon"], dtype=np.float32).reshape(-1, 2)
    if polygon.shape[0] < 3:
        return None
    (_, _), (side_a, side_b), _ = cv2.minAreaRect(polygon)
    short_side = float(min(side_a, side_b))
    long_side = float(max(side_a, side_b))
    if short_side <= 1e-6 or long_side <= 1e-6:
        return None

    box = cv2.boxPoints(cv2.minAreaRect(polygon)).astype(np.float64)
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)
    axis = edges[int(np.argmax(lengths))]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        return None
    axis = axis / norm
    if axis[1] < 0.0 or (abs(float(axis[1])) <= 1e-12 and axis[0] < 0.0):
        axis = -axis
    angle_deg = float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)
    return {
        **item,
        "axis": axis,
        "angle_deg_180": angle_deg,
        "aspect_ratio": long_side / short_side,
        "long_side_px": long_side,
        "short_side_px": short_side,
    }


def _axial_angle_distance_deg(a: float, b: float) -> float:
    delta = abs(float(a) - float(b)) % 180.0
    return float(min(delta, 180.0 - delta))


def estimate_ocr_axis(
    linked_ocr: Iterable[dict[str, Any]],
    aspect_ratio_min: float,
    orientation_outlier_deg: float,
) -> tuple[np.ndarray | None, np.ndarray | None, list[dict[str, Any]]]:
    """Estimate an axial (180-degree periodic) long direction robustly."""
    candidates = []
    for item in linked_ocr:
        parsed = _polygon_axis(item)
        if parsed is not None and parsed["aspect_ratio"] >= float(aspect_ratio_min):
            candidates.append(parsed)
    if not candidates:
        return None, None, []

    angles = np.asarray([item["angle_deg_180"] for item in candidates])
    distance_sums = np.asarray(
        [sum(_axial_angle_distance_deg(a, b) for b in angles) for a in angles]
    )
    medoid_angle = float(angles[int(np.argmin(distance_sums))])
    valid = [
        item
        for item in candidates
        if _axial_angle_distance_deg(item["angle_deg_180"], medoid_angle)
        <= float(orientation_outlier_deg)
    ]
    if not valid:
        return None, None, []

    weights = np.asarray([max(item["long_side_px"], 1.0) for item in valid])
    radians2 = np.deg2rad(
        np.asarray([item["angle_deg_180"] for item in valid]) * 2.0
    )
    mean2 = math.atan2(
        float(np.sum(weights * np.sin(radians2))),
        float(np.sum(weights * np.cos(radians2))),
    )
    angle = 0.5 * mean2
    axis = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
    if axis[1] < 0.0 or (abs(float(axis[1])) <= 1e-12 and axis[0] < 0.0):
        axis = -axis
    centers = np.asarray([item["center"] for item in valid], dtype=np.float64)
    origin = np.average(centers, axis=0, weights=weights)
    return axis, np.asarray(origin, dtype=np.float64), valid


def _boundary_profile(
    mask: np.ndarray, axis: np.ndarray, origin: np.ndarray
) -> dict[str, np.ndarray | float] | None:
    ys, xs = np.where(mask)
    if xs.size < 20:
        return None
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    relative = points - origin
    u_values = relative @ axis
    v_values = relative @ normal
    u_pixels = np.rint(u_values).astype(np.int32)
    u_min = int(u_pixels.min())
    u_max = int(u_pixels.max())
    size = u_max - u_min + 1
    left = np.full(size, np.inf, dtype=np.float64)
    right = np.full(size, -np.inf, dtype=np.float64)
    indices = u_pixels - u_min
    np.minimum.at(left, indices, v_values)
    np.maximum.at(right, indices, v_values)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return None
    u_grid = np.arange(u_min, u_max + 1, dtype=np.float64)
    if not np.all(valid):
        left[~valid] = np.interp(u_grid[~valid], u_grid[valid], left[valid])
        right[~valid] = np.interp(u_grid[~valid], u_grid[valid], right[valid])
    widths = right[valid] - left[valid]
    robust_width = float(np.median(widths[widths >= 0.0]))
    return {
        "u": u_grid,
        "left": left,
        "right": right,
        "width_px": robust_width,
        "normal": normal,
        "u_values": u_values,
        "v_values": v_values,
        "xs": xs,
        "ys": ys,
        "u_min": float(u_min),
    }


def _fit_line_ransac(
    x: np.ndarray, y: np.ndarray, residual_threshold: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or float(np.ptp(x)) <= 1e-9:
        return np.asarray([0.0, float(np.median(y))]), np.ones(x.size, dtype=bool)

    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(x.size, dtype=bool)
    best_key = (-1, -float("inf"))
    for _ in range(min(256, max(64, x.size * 2))):
        i, j = rng.choice(x.size, size=2, replace=False)
        dx = float(x[j] - x[i])
        if abs(dx) <= 1e-9:
            continue
        slope = float((y[j] - y[i]) / dx)
        intercept = float(y[i] - slope * x[i])
        residual = np.abs(y - (slope * x + intercept))
        inliers = residual <= float(residual_threshold)
        count = int(inliers.sum())
        median_residual = float(np.median(residual[inliers])) if count else float("inf")
        key = (count, -median_residual)
        if key > best_key:
            best_key = key
            best_inliers = inliers

    if int(best_inliers.sum()) < 2:
        best_inliers[:] = True
    coefficient = np.polyfit(x[best_inliers], y[best_inliers], 1)
    return np.asarray(coefficient, dtype=np.float64), best_inliers


def _apply_boundary_limits(
    original: np.ndarray,
    profile: dict[str, Any],
    left_limit: np.ndarray,
    right_limit: np.ndarray,
    margin_px: float,
) -> np.ndarray:
    indices = np.clip(
        np.rint(profile["u_values"]).astype(np.int32) - int(profile["u_min"]),
        0,
        len(left_limit) - 1,
    )
    keep = (
        (profile["v_values"] >= left_limit[indices] - float(margin_px))
        & (profile["v_values"] <= right_limit[indices] + float(margin_px))
    )
    output = np.zeros_like(original, dtype=bool)
    output[profile["ys"][keep], profile["xs"][keep]] = True
    return output & original


def _morphology(original: np.ndarray) -> np.ndarray:
    binary = original.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return original.copy()
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == largest).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(component, cv2.MORPH_OPEN, kernel)
    return (opened > 0) & original


def _remove_islands(
    original: np.ndarray, linked_ocr: Iterable[dict[str, Any]]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove only components disconnected from the OCR-selected main component.

    This method is deliberately not a GT-shape correction.  It splits the
    selected mask with 8-connectivity, chooses one main component from saved
    OCR linkage, and removes every other component.  Pixels inside the main
    component are copied bit-for-bit, so connected protrusions, indentations,
    and boundary roughness remain unchanged.  No GT, morphology, smoothing,
    contour fitting, or boundary model is used here.
    """
    binary = np.asarray(original, dtype=bool).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    # OpenCV includes background label 0.
    component_ids = list(range(1, int(component_count)))
    if not component_ids:
        return original.copy(), {
            "component_count": 0,
            "main_component_label": None,
            "main_component_source": "none",
            "removed_component_count": 0,
            "removed_pixel_count": 0,
            "components": [],
        }

    linked = list(linked_ocr)
    overlap_by_label = {label: 0 for label in component_ids}
    center_count_by_label = {label: 0 for label in component_ids}
    height, width = original.shape

    # Sum the intersection area of every linked OCR polygon with each component.
    # Each polygon is handled independently; GT is never consulted here.
    for item in linked:
        try:
            polygon = np.asarray(item["polygon"], dtype=np.float64).reshape(-1, 2)
        except Exception:
            continue
        if polygon.shape[0] < 3 or not np.all(np.isfinite(polygon)):
            continue
        points = np.round(polygon).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        polygon_mask = np.zeros_like(binary)
        cv2.fillPoly(polygon_mask, [points.reshape(-1, 1, 2)], 1)
        intersecting_labels, counts = np.unique(
            labels[polygon_mask > 0], return_counts=True
        )
        for label, count in zip(intersecting_labels, counts):
            label = int(label)
            if label in overlap_by_label:
                overlap_by_label[label] += int(count)

        center = item.get("center")
        if center is not None:
            try:
                x = int(round(float(center[0])))
                y = int(round(float(center[1])))
            except (TypeError, ValueError, IndexError):
                continue
            if 0 <= x < width and 0 <= y < height:
                label = int(labels[y, x])
                if label in center_count_by_label:
                    center_count_by_label[label] += 1

    areas = {
        label: int(stats[label, cv2.CC_STAT_AREA]) for label in component_ids
    }
    maximum_overlap = max(overlap_by_label.values(), default=0)
    if maximum_overlap > 0:
        # Area and label are deterministic tie-breakers only.
        main_label = max(
            component_ids,
            key=lambda label: (
                overlap_by_label[label],
                areas[label],
                -label,
            ),
        )
        source = "maximum_ocr_overlap"
    else:
        maximum_center_count = max(center_count_by_label.values(), default=0)
        if maximum_center_count > 0:
            main_label = max(
                component_ids,
                key=lambda label: (
                    center_count_by_label[label],
                    areas[label],
                    -label,
                ),
            )
            source = "ocr_center_component"
        else:
            main_label = max(
                component_ids, key=lambda label: (areas[label], -label)
            )
            source = "largest_area_component"

    # This label selection is the whole shape operation: keep the main
    # component exactly and delete only components that are fully disconnected
    # from it under 8-connectivity.
    cleaned = labels == int(main_label)
    removed_component_count = len(component_ids) - 1
    removed_pixel_count = int(original.sum()) - int(cleaned.sum())
    components = [
        {
            "label": label,
            "area_px": areas[label],
            "ocr_overlap_px": overlap_by_label[label],
            "ocr_center_count": center_count_by_label[label],
            "main": label == main_label,
        }
        for label in component_ids
    ]
    return cleaned, {
        "component_count": len(component_ids),
        "main_component_label": int(main_label),
        "main_component_source": source,
        "removed_component_count": int(removed_component_count),
        "removed_pixel_count": int(removed_pixel_count),
        "components": components,
    }


def _pca_axis(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    ys, xs = np.where(mask)
    if xs.size < 20:
        return None
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        return None
    return axis / norm, center


def _mask_width(mask: np.ndarray, frame: tuple[np.ndarray, np.ndarray] | None) -> float | None:
    if frame is None:
        return None
    profile = _boundary_profile(mask, frame[0], frame[1])
    if profile is None:
        return None
    width = float(profile["width_px"])
    return width if math.isfinite(width) and width > 0.0 else None


def _center_survives(mask: np.ndarray, center: np.ndarray) -> bool:
    x = int(round(float(center[0])))
    y = int(round(float(center[1])))
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def _fallback(
    original: np.ndarray, info: dict[str, Any], reason: str, started: float
) -> PostprocessResult:
    info.update(
        {
            "fallback": True,
            "fallback_reason": str(reason),
            "mask_area_after": int(original.sum()),
            "area_ratio": 1.0,
            "processing_time_ms": (time.perf_counter() - started) * 1000.0,
        }
    )
    return PostprocessResult(original.copy(), info)


def postprocess_selected_mask(
    mask: np.ndarray,
    linked_ocr: Iterable[dict[str, Any]],
    config: dict[str, Any] | None,
) -> PostprocessResult:
    """Apply one deletion-only method and its four mandatory safety checks."""
    started = time.perf_counter()
    original = np.asarray(mask, dtype=bool)
    cfg = normalize_postprocess_config(config)
    linked = list(linked_ocr)
    area_before = int(original.sum())
    info: dict[str, Any] = {
        "method": cfg["method"],
        "params": cfg,
        "mask_area_before": area_before,
        "mask_area_after": area_before,
        "area_ratio": 1.0,
        "num_linked_ocr": len(linked),
        "num_valid_ocr": 0,
        "estimated_axis_deg": None,
        "axis": None,
        "axis_origin": None,
        "boundary_u": [],
        "boundary_left": [],
        "boundary_right": [],
        "boundary_margin_px": None,
        "width_before_px": None,
        "width_after_px": None,
        "width_ratio": None,
        "component_count": None,
        "removed_component_count": 0,
        "removed_pixel_count": 0,
        "fallback": False,
        "fallback_reason": "",
    }

    if area_before <= 0:
        return _fallback(original, info, "empty_original_mask", started)
    if not cfg["enabled"] or cfg["method"] == "baseline":
        info["processing_time_ms"] = (time.perf_counter() - started) * 1000.0
        return PostprocessResult(original.copy(), info)

    try:
        axis = None
        origin = None
        if cfg["method"] == "remove_islands":
            candidate, component_info = _remove_islands(original, linked)
            candidate = np.asarray(candidate, dtype=bool) & original
            # remove_islands is component cleanup, not GT-shape correction.
            # labels == main_label copies the main component bit-for-bit; this
            # branch intentionally bypasses morphology, smoothing, contour
            # fitting, boundary correction, and the generic shape guards below.
            info.update(component_info)
            info.update(
                {
                    "mask_area_after": int(candidate.sum()),
                    "area_ratio": float(candidate.sum() / max(area_before, 1)),
                    "processing_time_ms": (
                        time.perf_counter() - started
                    ) * 1000.0,
                }
            )
            return PostprocessResult(candidate, info)
        if cfg["method"] == "morphology":
            candidate = _morphology(original)
        else:
            axis, origin, valid_ocr = estimate_ocr_axis(
                linked,
                cfg["ocr_aspect_ratio_min"],
                cfg["orientation_outlier_deg"],
            )
            info["num_valid_ocr"] = len(valid_ocr)
            if axis is None or origin is None:
                return _fallback(original, info, "no_valid_ocr", started)
            normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
            profile = _boundary_profile(original, axis, origin)
            if profile is None or float(profile["width_px"]) <= 0.0:
                return _fallback(original, info, "invalid_boundary_profile", started)
            margin_px = max(
                1.0, float(cfg["boundary_margin_ratio"]) * float(profile["width_px"])
            )
            if cfg["method"] == "ocr_axis_ransac":
                residual_threshold = max(1.0, margin_px)
                left_fit, left_inliers = _fit_line_ransac(
                    profile["u"], profile["left"], residual_threshold, seed=1729
                )
                right_fit, right_inliers = _fit_line_ransac(
                    profile["u"], profile["right"], residual_threshold, seed=2718
                )
                left_limit = np.polyval(left_fit, profile["u"])
                right_limit = np.polyval(right_fit, profile["u"])
                info["ransac"] = {
                    "left_coefficients": left_fit.tolist(),
                    "right_coefficients": right_fit.tolist(),
                    "left_inlier_count": int(left_inliers.sum()),
                    "right_inlier_count": int(right_inliers.sum()),
                }
            else:
                window = int(cfg["median_window"])
                left_limit = median_filter(profile["left"], size=window, mode="nearest")
                right_limit = median_filter(profile["right"], size=window, mode="nearest")
            candidate = _apply_boundary_limits(
                original, profile, left_limit, right_limit, margin_px
            )
            angle_deg = float(np.degrees(np.arctan2(axis[1], axis[0])))
            if angle_deg >= 90.0:
                angle_deg -= 180.0
            info.update(
                {
                    "estimated_axis_deg": angle_deg,
                    "axis": axis.tolist(),
                    "axis_origin": origin.tolist(),
                    "boundary_u": np.asarray(profile["u"]).tolist(),
                    "boundary_left": np.asarray(left_limit).tolist(),
                    "boundary_right": np.asarray(right_limit).tolist(),
                    "boundary_margin_px": float(margin_px),
                }
            )

        # An explicit final intersection makes pixel addition impossible even if
        # a future morphology implementation uses a dilating operation.
        candidate = np.asarray(candidate, dtype=bool) & original
        area_after = int(candidate.sum())
        area_ratio = float(area_after / max(area_before, 1))
        if area_after <= 0:
            return _fallback(original, info, "empty_postprocessed_mask", started)
        if area_ratio < float(cfg["min_area_retention"]):
            return _fallback(
                original,
                info,
                f"area_retention_below_threshold:{area_ratio:.6f}",
                started,
            )

        # Only centers that existed in the original mask can be protected.
        protected_centers = [
            np.asarray(item["center"], dtype=np.float64)
            for item in linked
            if _center_survives(original, np.asarray(item["center"]))
        ]
        if any(not _center_survives(candidate, center) for center in protected_centers):
            return _fallback(original, info, "linked_ocr_center_removed", started)

        width_frame = (axis, origin) if axis is not None else _pca_axis(original)
        width_before = _mask_width(original, width_frame)
        width_after = _mask_width(candidate, width_frame)
        if width_before is None or width_after is None:
            return _fallback(original, info, "invalid_width_estimate", started)
        width_ratio = float(width_after / width_before)
        info.update(
            {
                "width_before_px": width_before,
                "width_after_px": width_after,
                "width_ratio": width_ratio,
            }
        )
        if width_ratio < float(cfg["min_width_retention"]):
            return _fallback(
                original,
                info,
                f"width_retention_below_threshold:{width_ratio:.6f}",
                started,
            )

        info.update(
            {
                "mask_area_after": area_after,
                "area_ratio": area_ratio,
                "processing_time_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
        return PostprocessResult(candidate, info)
    except Exception as exc:
        return _fallback(
            original,
            info,
            f"processing_error:{type(exc).__name__}:{exc}",
            started,
        )


def render_debug_overlay(
    rgb_bgr: np.ndarray,
    original_mask: np.ndarray,
    postprocessed_mask: np.ndarray,
    gt_mask: np.ndarray,
    linked_ocr: Iterable[dict[str, Any]],
    info: dict[str, Any],
) -> np.ndarray:
    """Render all requested experiment evidence into one overlay image."""
    image = np.asarray(rgb_bgr, dtype=np.uint8).copy()
    original = np.asarray(original_mask, dtype=bool)
    processed = np.asarray(postprocessed_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    if image.shape[:2] != original.shape or gt.shape != original.shape:
        raise ValueError("debug image/mask shape mismatch")

    tint = image.copy()
    tint[gt] = (0, 150, 0)
    tint[original] = (0, 215, 255)
    tint[processed] = (0, 0, 255)
    removed = original & ~processed
    tint[removed] = (255, 0, 255)
    output = cv2.addWeighted(image, 0.62, tint, 0.38, 0)

    def contours(mask: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
        found, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, found, -1, color, thickness, cv2.LINE_AA)

    contours(gt, (0, 255, 0), 2)
    contours(original, (0, 255, 255), 2)
    contours(processed, (0, 0, 255), 2)
    for item in linked_ocr:
        polygon = np.round(np.asarray(item["polygon"])).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [polygon], True, (255, 120, 0), 2, cv2.LINE_AA)
        center = tuple(np.round(np.asarray(item["center"])).astype(int))
        cv2.circle(output, center, 3, (255, 255, 255), -1)

    axis = info.get("axis")
    origin = info.get("axis_origin")
    if axis is not None and origin is not None:
        axis = np.asarray(axis, dtype=np.float64)
        origin = np.asarray(origin, dtype=np.float64)
        length = float(max(image.shape[:2]) * 1.5)
        p0 = tuple(np.round(origin - length * axis).astype(int))
        p1 = tuple(np.round(origin + length * axis).astype(int))
        cv2.line(output, p0, p1, (255, 255, 255), 2, cv2.LINE_AA)

        u = np.asarray(info.get("boundary_u") or [], dtype=np.float64)
        left = np.asarray(info.get("boundary_left") or [], dtype=np.float64)
        right = np.asarray(info.get("boundary_right") or [], dtype=np.float64)
        if u.size and u.size == left.size == right.size:
            normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
            left_xy = origin + u[:, None] * axis + left[:, None] * normal
            right_xy = origin + u[:, None] * axis + right[:, None] * normal
            cv2.polylines(
                output,
                [np.round(left_xy).astype(np.int32).reshape(-1, 1, 2)],
                False,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.polylines(
                output,
                [np.round(right_xy).astype(np.int32).reshape(-1, 1, 2)],
                False,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

    lines = [
        f"method={info.get('method')} fallback={info.get('fallback')}",
        f"reason={info.get('fallback_reason') or '-'}",
        "GT=green original=yellow post=red removed=magenta OCR=blue boundary=blue",
    ]
    y = 25
    for line in lines:
        cv2.putText(output, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(output, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 23
    return output
