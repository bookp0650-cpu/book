"""RANSAC plane estimation used only for per-pixel Depth completion.

RANSAC outliers are never deleted. Valid measurements within the 8 mm plane
distance are preserved; invalid or plane-outlier pixels are replaced by the
pixel ray/plane intersection.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import get_book_points as current
from .modules.pointcloud_utils import save_ply_ascii


DEFAULT_PARAMETERS = {
    "ransac_distance_threshold_m": 0.008,
    "ransac_n": 3,
    "num_iterations": 1200,
    "min_input_points": 100,
    "min_inlier_ratio": 0.50,
    "max_residual_median_m": 0.008,
    "min_abs_plane_c_normalized": 0.20,
    "min_depth_m": 0.10,
    "max_depth_m": 3.00,
}


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _depth_visualization(depth, mask=None):
    values = np.asarray(depth, np.float64)
    valid = np.isfinite(values) & (values > 0)
    if mask is not None:
        valid &= np.asarray(mask, bool)
    gray = np.zeros(values.shape, np.uint8)
    if valid.any():
        low, high = np.percentile(values[valid], [2, 98])
        if high <= low:
            high = low + 1.0
        gray[valid] = np.clip(
            (values[valid] - low) * 255.0 / (high - low), 0, 255
        )
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def _select_input_region(anchor_mask, final_mask, depth_raw, min_points):
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(anchor_mask.astype(np.uint8), kernel, iterations=1) > 0
    candidates = [
        ("anchor_eroded_3x3", eroded),
        ("anchor_component", np.asarray(anchor_mask, bool)),
        ("final_mask", np.asarray(final_mask, bool)),
    ]
    for name, region in candidates:
        valid = region & np.isfinite(depth_raw) & (depth_raw > 0)
        if int(valid.sum()) >= int(min_points):
            return name, region, valid
    return None, None, None


def _predict_plane_depth_m(mask, intr, plane):
    ys, xs = np.where(mask)
    a, b, c, d = np.asarray(plane, np.float64).reshape(4)
    denom = (
        a * (xs.astype(np.float64) - float(intr.ppx)) / float(intr.fx)
        + b * (ys.astype(np.float64) - float(intr.ppy)) / float(intr.fy)
        + c
    )
    z = np.full(xs.shape, np.nan, np.float64)
    good = np.isfinite(denom) & (np.abs(denom) > 1e-9)
    z[good] = -d / denom[good]
    prediction = np.full(mask.shape, np.nan, np.float64)
    prediction[ys, xs] = z
    return prediction


def complete_depth_with_ransac_plane(
    *,
    final_mask,
    anchor_mask,
    depth_raw,
    intr,
    depth_scale,
    rgb_bgr,
    output_dir: str | Path,
    parameters: dict | None = None,
):
    """Return corrected Depth, source UVs, plane JSON, and completion JSON."""
    params = {**DEFAULT_PARAMETERS, **(parameters or {})}
    final_mask = np.asarray(final_mask, bool)
    anchor_mask = np.asarray(anchor_mask, bool)
    depth_raw = np.asarray(depth_raw)
    rgb = np.asarray(rgb_bgr)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if final_mask.shape != depth_raw.shape or rgb.shape[:2] != final_mask.shape:
        raise ValueError("RGB, Depth, and mask shapes must match")

    input_name, input_region, input_valid = _select_input_region(
        anchor_mask,
        final_mask,
        depth_raw,
        params["min_input_points"],
    )
    plane_result = {
        "success": False,
        "input_region": input_name,
        "input_point_count": 0,
        "plane_coefficients": None,
        "plane_normal": None,
        "ransac_distance_threshold_m": params["ransac_distance_threshold_m"],
        "ransac_n": params["ransac_n"],
        "num_iterations": params["num_iterations"],
        "random_seed": 0,
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "residual_median_m": None,
        "residual_mean_m": None,
        "residual_p90_m": None,
        "residual_max_m": None,
        "predicted_depth_min_m": None,
        "predicted_depth_max_m": None,
        "quality_passed": False,
        "quality_fail_reasons": [],
        "fallback_used": False,
        "fallback_reason": None,
    }

    if input_valid is None:
        plane_result["quality_fail_reasons"].append(
            "insufficient_valid_depth_for_ransac"
        )
        plane = None
        input_points = np.empty((0, 3), np.float64)
        input_uv = np.empty((0, 2), np.int32)
        inlier_mask = np.zeros(0, bool)
    else:
        input_points, input_uv = current._mask_depth_to_points_uv_for_plane_filter(
            input_valid.astype(np.uint8), depth_raw, intr, depth_scale
        )
        plane_result["input_point_count"] = int(input_points.shape[0])
        try:
            # Open3D exposes a process-local RANSAC RNG. This does not affect
            # the protected stable function or its callers.
            current.o3d.utility.random.seed(0)
        except Exception:
            plane_result["random_seed"] = None
        plane, inlier_mask, base_info = current._fit_plane_ransac_open3d_for_spine(
            input_points,
            distance_threshold_m=params["ransac_distance_threshold_m"],
            ransac_n=params["ransac_n"],
            num_iterations=params["num_iterations"],
        )
        if plane is None or not base_info.get("used"):
            plane_result["quality_fail_reasons"].append(
                f"ransac_failed:{base_info.get('reason')}"
            )

    colors_input = (
        rgb[input_uv[:, 1], input_uv[:, 0], ::-1].astype(np.uint8)
        if input_uv.size
        else np.empty((0, 3), np.uint8)
    )
    save_ply_ascii(out / "ransac_input_points.ply", input_points, colors_input)
    if input_region is None:
        input_region = np.zeros_like(final_mask)
    cv2.imwrite(
        str(out / "ransac_input_region.png"), input_region.astype(np.uint8) * 255
    )

    prediction_m = np.full(final_mask.shape, np.nan, np.float64)
    residuals = np.empty(0, np.float64)
    if plane is not None:
        normal_norm = float(np.linalg.norm(plane[:3]))
        normalized_plane = np.asarray(plane, np.float64) / max(normal_norm, 1e-12)
        plane = normalized_plane
        residuals = current._point_plane_distance_for_spine(input_points, plane)
        prediction_m = _predict_plane_depth_m(final_mask, intr, plane)
        predicted_values = prediction_m[final_mask]
        predicted_valid = (
            np.isfinite(predicted_values)
            & (predicted_values >= params["min_depth_m"])
            & (predicted_values <= params["max_depth_m"])
        )
        inlier_count = int(np.count_nonzero(inlier_mask))
        ratio = inlier_count / max(int(input_points.shape[0]), 1)
        plane_result.update(
            {
                "plane_coefficients": plane.astype(float).tolist(),
                "plane_normal": plane[:3].astype(float).tolist(),
                "inlier_count": inlier_count,
                "inlier_ratio": float(ratio),
                "residual_median_m": float(np.median(residuals)),
                "residual_mean_m": float(np.mean(residuals)),
                "residual_p90_m": float(np.percentile(residuals, 90)),
                "residual_max_m": float(np.max(residuals)),
                "predicted_depth_min_m": (
                    float(np.min(predicted_values[predicted_valid]))
                    if predicted_valid.any()
                    else None
                ),
                "predicted_depth_max_m": (
                    float(np.max(predicted_values[predicted_valid]))
                    if predicted_valid.any()
                    else None
                ),
            }
        )
        if input_points.shape[0] < params["min_input_points"]:
            plane_result["quality_fail_reasons"].append("too_few_input_points")
        if ratio < params["min_inlier_ratio"]:
            plane_result["quality_fail_reasons"].append("low_inlier_ratio")
        if float(np.median(residuals)) > params["max_residual_median_m"]:
            plane_result["quality_fail_reasons"].append("high_median_residual")
        if abs(float(plane[2])) < params["min_abs_plane_c_normalized"]:
            plane_result["quality_fail_reasons"].append(
                "plane_nearly_parallel_to_camera_rays"
            )
        if not bool(np.all(predicted_valid)):
            plane_result["quality_fail_reasons"].append(
                "invalid_predicted_depth_inside_final_mask"
            )

    quality = plane is not None and not plane_result["quality_fail_reasons"]
    plane_result["success"] = bool(plane is not None)
    plane_result["quality_passed"] = bool(quality)

    input_inliers = input_points[inlier_mask] if input_points.size else input_points
    input_outliers = input_points[~inlier_mask] if input_points.size else input_points
    colors_inliers = colors_input[inlier_mask] if colors_input.size else colors_input
    colors_outliers = colors_input[~inlier_mask] if colors_input.size else colors_input
    save_ply_ascii(out / "ransac_plane_inliers.ply", input_inliers, colors_inliers)
    save_ply_ascii(out / "ransac_plane_outliers.ply", input_outliers, colors_outliers)

    corrected = np.zeros(depth_raw.shape, np.float64)
    classification = np.zeros(depth_raw.shape, np.uint8)
    raw_valid_mask = final_mask & np.isfinite(depth_raw) & (depth_raw > 0)
    fallback = not quality
    if quality:
        ys, xs = np.where(final_mask)
        raw_z_m = depth_raw[ys, xs].astype(np.float64) * float(depth_scale)
        raw_valid = np.isfinite(raw_z_m) & (raw_z_m > 0)
        raw_points = np.column_stack(
            [
                (xs - float(intr.ppx)) / float(intr.fx) * raw_z_m,
                (ys - float(intr.ppy)) / float(intr.fy) * raw_z_m,
                raw_z_m,
            ]
        )
        distances = current._point_plane_distance_for_spine(raw_points, plane)
        predicted = prediction_m[ys, xs]
        keep = raw_valid & (
            distances <= float(params["ransac_distance_threshold_m"])
        )
        invalid_replace = ~raw_valid
        outlier_replace = raw_valid & ~keep
        completed_z_m = raw_z_m.copy()
        completed_z_m[invalid_replace | outlier_replace] = predicted[
            invalid_replace | outlier_replace
        ]
        corrected[ys, xs] = completed_z_m / float(depth_scale)
        classification[ys[keep], xs[keep]] = 1
        classification[ys[invalid_replace], xs[invalid_replace]] = 2
        classification[ys[outlier_replace], xs[outlier_replace]] = 3
    else:
        corrected[raw_valid_mask] = depth_raw[raw_valid_mask]
        classification[raw_valid_mask] = 1
        plane_result["fallback_used"] = True
        plane_result["fallback_reason"] = ";".join(
            plane_result["quality_fail_reasons"]
        )

    cv2.imwrite(
        str(out / "depth_raw_visualization.png"),
        _depth_visualization(depth_raw, final_mask),
    )
    cv2.imwrite(
        str(out / "depth_plane_prediction_visualization.png"),
        _depth_visualization(
            prediction_m / float(depth_scale), final_mask & np.isfinite(prediction_m)
        ),
    )
    cv2.imwrite(
        str(out / "depth_ransac_completed_visualization.png"),
        _depth_visualization(corrected, final_mask),
    )
    class_vis = np.zeros((*final_mask.shape, 3), np.uint8)
    class_vis[classification == 1] = (0, 255, 0)
    class_vis[classification == 2] = (255, 0, 0)
    class_vis[classification == 3] = (0, 0, 255)
    cv2.imwrite(str(out / "depth_replacement_classification.png"), class_vis)
    plane_overlay = rgb.copy()
    tint = rgb.copy()
    tint[final_mask] = class_vis[final_mask]
    plane_overlay = cv2.addWeighted(plane_overlay, 0.65, tint, 0.35, 0)
    cv2.imwrite(str(out / "ransac_plane_overlay.png"), plane_overlay)
    np.save(out / "target_mask_depth_ransac_completed.npy", corrected)

    kept = int(np.count_nonzero(classification == 1))
    invalid_replaced = int(np.count_nonzero(classification == 2))
    outlier_replaced = int(np.count_nonzero(classification == 3))
    replacement = invalid_replaced + outlier_replaced
    ratio = replacement / max(int(final_mask.sum()), 1)
    if ratio < 0.10:
        category = "under_10_percent"
    elif ratio < 0.30:
        category = "10_to_30_percent"
    elif ratio < 0.50:
        category = "30_to_50_percent"
    else:
        category = "50_percent_or_more"
    warnings = []
    if ratio >= 0.50:
        warnings.append(
            "replacement ratio is at least 50%; mask, Depth, or plane may be unstable"
        )
    completion_result = {
        "enabled": True,
        "method": "ransac_plane",
        "final_mask_area_px": int(final_mask.sum()),
        "raw_valid_depth_count": int(raw_valid_mask.sum()),
        "raw_depth_kept": kept,
        "invalid_depth_replaced": invalid_replaced,
        "plane_outlier_replaced": outlier_replaced,
        "replacement_count": replacement,
        "replacement_ratio": float(ratio),
        "replacement_ratio_category": category,
        "completed_depth_count": int(np.count_nonzero(corrected[final_mask] > 0)),
        "pointcloud_point_count": None,
        "pca_input_point_count": None,
        "pointcloud_equals_pca_input": None,
        "median_used_for_replacement": False,
        "ransac_outliers_deleted": False,
        "fallback_used": fallback,
        "fallback_reason": plane_result["fallback_reason"],
        "warnings": warnings,
    }
    _write_json(out / "ransac_plane_result.json", plane_result)
    _write_json(out / "depth_completion_result.json", completion_result)
    return corrected, classification, plane_result, completion_result

