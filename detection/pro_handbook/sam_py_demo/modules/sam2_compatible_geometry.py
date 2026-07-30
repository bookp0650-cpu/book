"""SAM2-compatible terminal geometry for a selected/refined SAM3 mask.

The formulas are a narrow, traceable port of the terminal geometry in the
legacy get_book_points.py.  Recognition and mask selection are intentionally
outside this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .grip_point import find_target_point
from .pca_vector import pca_axes_fix_dir


GEOMETRY_MODES = {
    "sam3_current_geometry",
    "sam2_roll_only",
    "sam2_width_only",
    "sam2_target_only",
    "sam2_all_geometry",
}


def _axis_and_center(mask, axis=None, center=None):
    mask = np.asarray(mask) > 0
    ys, xs = np.where(mask)
    if len(xs) < 20:
        raise ValueError("too few mask pixels")
    if axis is None:
        uv = np.column_stack([xs, ys]).astype(np.float64)
        covariance = np.cov((uv - uv.mean(axis=0)).T)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        source = "mask_pca_axis_fallback"
    else:
        axis = np.asarray(axis, np.float64).reshape(2)
        source = "selected_ocr_axis"
    axis /= np.linalg.norm(axis)
    if center is None:
        center = np.array([xs.mean(), ys.mean()], np.float64)
    return axis, np.asarray(center, np.float64).reshape(2), source


def _project_points(points, intrinsics):
    points = np.asarray(points, np.float64)
    z = points[:, 2]
    u = intrinsics["fx"] * points[:, 0] / z + intrinsics["ppx"]
    v = intrinsics["fy"] * points[:, 1] / z + intrinsics["ppy"]
    return np.column_stack([u, v])


def estimate_sam2_compatible_geometry(
    mask,
    rgb_image,
    depth_image,
    points,
    camera_intrinsics,
    *,
    selected_axis=None,
    selected_center=None,
    current_geometry=None,
    geometry_mode="sam2_all_geometry",
    debug_dir=None,
):
    """Apply the legacy terminal roll/width/target formulas.

    ``mask`` corresponds to legacy ``mask01`` immediately before
    ``calculate_yaw``. ``depth_image`` corresponds to legacy ``depth_masked``.
    ``points`` is the final point cloud produced from that mask/depth stage.
    """
    if geometry_mode not in GEOMETRY_MODES:
        raise ValueError(f"unknown geometry_mode: {geometry_mode}")
    mask = np.asarray(mask) > 0
    depth = np.asarray(depth_image)
    points = np.asarray(points, np.float64)
    rgb = np.asarray(rgb_image)
    if mask.shape != depth.shape or rgb.shape[:2] != mask.shape:
        raise ValueError("mask/rgb/depth shape mismatch")
    if len(points) < 3:
        raise ValueError("too few final geometry points")

    axis, center, axis_source = _axis_and_center(
        mask, selected_axis, selected_center
    )
    normal = np.array([-axis[1], axis[0]], np.float64)
    valid = mask & np.isfinite(depth) & (depth > 0)
    ys, xs = np.where(valid)
    if len(xs) < 20:
        raise ValueError("too few valid Depth pixels for SAM2 width")
    uv = np.column_stack([xs, ys]).astype(np.float64)
    transverse = (uv - center) @ normal
    low, high = np.percentile(transverse, [2.0, 98.0])
    width_px = float(max(0.0, high - low))
    depth_scale = float(camera_intrinsics["depth_scale"])
    representative_depth_m = float(np.median(depth[valid]) * depth_scale)
    scale_m_per_px = representative_depth_m * float(np.sqrt(
        (normal[0] / float(camera_intrinsics["fx"])) ** 2
        + (normal[1] / float(camera_intrinsics["fy"])) ** 2
    ))
    width_mm = width_px * scale_m_per_px * 1000.0
    width_fallback = not np.isfinite(width_mm) or not 2.0 <= width_mm <= 150.0

    mean, pc1, pc2 = pca_axes_fix_dir(points)
    norm_xy = float(np.hypot(pc1[0], pc1[1]))
    roll = 0.0 if norm_xy < 1e-8 else float(np.arctan2(pc1[1], pc1[0]))
    target_info = find_target_point(points)
    target = target_info.get("target_m")
    if target is None:
        raise RuntimeError(f"legacy find_target_point failed: {target_info}")
    target = np.asarray(target, np.float64)
    target_uv_float = _project_points(target.reshape(1, 3), camera_intrinsics)[0]
    target_pixel = np.rint(target_uv_float).astype(int)

    current = current_geometry or {}
    use_roll = geometry_mode in {"sam2_roll_only", "sam2_all_geometry"}
    use_width = geometry_mode in {"sam2_width_only", "sam2_all_geometry"}
    use_target = geometry_mode in {"sam2_target_only", "sam2_all_geometry"}
    if geometry_mode == "sam3_current_geometry":
        use_roll = use_width = use_target = False
    final_roll = roll if use_roll else current.get("roll_rad")
    final_width = width_mm if use_width else current.get("width_mm")
    final_target = target if use_target else np.asarray(
        current.get("target_point_m"), np.float64
    )

    result = {
        "geometry_mode": geometry_mode,
        "mask_stage": "sam3_conservative_refinement_output_mapped_to_sam2_final_mask01",
        "roll": {
            "method": "sam2_compatible_pca_pc1_atan2" if use_roll else "sam3_current",
            "value_rad": None if final_roll is None else float(final_roll),
            "value_deg": None if final_roll is None else float(np.degrees(final_roll)),
            "sam2_candidate_rad": roll,
            "axis_source": "final_pointcloud_pca_pc1",
            "pc1": pc1.tolist(),
            "fallback": False,
            "fallback_reason": None,
        },
        "width": {
            "method": "sam2_compatible_filtered_mask_axis_pixel_width_to_metric"
            if use_width else "sam3_current_3d_pca_pc2_slices",
            "pixel_width": width_px,
            "representative_depth_m": representative_depth_m,
            "scale_m_per_px": scale_m_per_px,
            "width_mm": None if final_width is None else float(final_width),
            "sam2_candidate_width_mm": width_mm,
            "axis": axis.tolist(),
            "normal": normal.tolist(),
            "axis_source": axis_source,
            "percentiles": [2.0, 98.0],
            "fallback": width_fallback,
            "fallback_reason": "estimated width out of [2,150] mm"
            if width_fallback else None,
        },
        "target": {
            "method": "sam2_compatible_find_target_point"
            if use_target else "sam3_current",
            "target_pixel": target_pixel.tolist(),
            "target_pixel_float": target_uv_float.tolist(),
            "target_depth_m": float(target[2]),
            "target_point_m": final_target.tolist(),
            "sam2_candidate_target_point_m": target.tolist(),
            "pointcloud_source": "final_geometry_pointcloud",
            "y_offset_m": 0.1,
            "y_band_half_m": 0.003,
            "fallback": False,
            "fallback_reason": None,
        },
        "geometry_methods": {
            "roll_method": "sam2_compatible" if use_roll else "sam3_current",
            "width_method": "sam2_compatible" if use_width else "sam3_current",
            "target_method": "sam2_compatible" if use_target else "sam3_current",
        },
        "debug": {
            "valid_depth_pixels": int(valid.sum()),
            "point_count": int(len(points)),
            "pca_mean": mean.tolist(),
            "pc2": pc2.tolist(),
        },
    }
    if debug_dir is not None:
        _save_debug(
            Path(debug_dir), rgb, mask, axis, center, normal, low, high,
            target_pixel, current, result
        )
    return result


def _save_debug(out, rgb, mask, axis, center, normal, low, high,
                target_pixel, current, result):
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "geometry_input_mask.png"), mask.astype(np.uint8) * 255)
    base = rgb.copy()
    green = base.copy(); green[mask] = (0, 255, 0)
    overlay = cv2.addWeighted(base, 0.65, green, 0.35, 0)
    cv2.imwrite(str(out / "geometry_input_mask_overlay.png"), overlay)
    scale = 220
    def axis_image(vector, color, name):
        image = overlay.copy()
        p0 = tuple(np.rint(center - vector * scale).astype(int))
        p1 = tuple(np.rint(center + vector * scale).astype(int))
        cv2.line(image, p0, p1, color, 3)
        cv2.imwrite(str(out / name), image)
    axis_image(axis, (255, 0, 0), "sam2_long_axis_overlay.png")
    axis_image(normal, (0, 255, 255), "sam2_width_axis_overlay.png")
    axis_image(axis, (255, 0, 0), "sam2_roll_overlay.png")
    width_image = overlay.copy()
    for value, color in [(low, (255, 255, 0)), (high, (0, 255, 255))]:
        point = center + normal * value
        p0 = tuple(np.rint(point - axis * scale).astype(int))
        p1 = tuple(np.rint(point + axis * scale).astype(int))
        cv2.line(width_image, p0, p1, color, 2)
    cv2.imwrite(str(out / "sam2_width_measurement_overlay.png"), width_image)
    target_image = overlay.copy()
    cv2.circle(target_image, tuple(target_pixel), 6, (0, 0, 255), -1)
    cv2.imwrite(str(out / "sam2_target_pixel_overlay.png"), target_image)
    depth_region = np.zeros(mask.shape, np.uint8)
    depth_region[mask] = 255
    cv2.imwrite(str(out / "sam2_target_depth_region.png"), depth_region)
    final_image = overlay.copy()
    cv2.circle(final_image, tuple(target_pixel), 6, (0, 0, 255), -1)
    cv2.imwrite(str(out / "final.png"), final_image)
    comparison = overlay.copy()
    current_target = current.get("target_pixel")
    if current_target is not None:
        cv2.circle(comparison, tuple(map(int, current_target)), 7, (255, 0, 0), -1)
    cv2.circle(comparison, tuple(target_pixel), 5, (0, 0, 255), -1)
    cv2.imwrite(str(out / "target_point_comparison_overlay.png"), comparison)
    (out / "sam2_geometry_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
