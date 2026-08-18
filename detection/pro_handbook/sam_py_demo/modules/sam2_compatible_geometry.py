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


MASK_PCA_MIN_PIXELS = 20
MASK_PCA_EPSILON = 1e-12
MASK_PCA_WIDTH_MODE_GLOBAL = "global_p2p98"
MASK_PCA_WIDTH_MODE_SLICE_P2P98 = "slice_p2p98_median"
MASK_PCA_WIDTH_MODE_SLICE_MINMAX = "slice_minmax_median"
MASK_PCA_WIDTH_MODES = (
    MASK_PCA_WIDTH_MODE_GLOBAL,
    MASK_PCA_WIDTH_MODE_SLICE_P2P98,
    MASK_PCA_WIDTH_MODE_SLICE_MINMAX,
)
DEFAULT_MASK_PCA_WIDTH_MODE = MASK_PCA_WIDTH_MODE_SLICE_MINMAX
MASK_SLICE_BIN_SIZE_PX = 8.0
MASK_SLICE_MIN_PIXELS = 12
MASK_SLICE_MIN_VALID_COUNT = 5


def _axis_angle_deg(axis):
    if axis is None:
        return None
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)
    return 0.0 if abs(angle - 180.0) < 1e-12 else angle


def _axial_angle_difference_deg(first, second):
    if first is None or second is None:
        return None
    difference = abs(_axis_angle_deg(first) - _axis_angle_deg(second))
    return float(min(difference, 180.0 - difference))


def _optional_diagnostic_axis(axis):
    if axis is None:
        return None
    try:
        value = np.asarray(axis, np.float64).reshape(2)
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(value))
    if not np.all(np.isfinite(value)) or not np.isfinite(norm) or norm <= 0.0:
        return None
    return value / norm


def _mask_pca_axis(mask):
    mask = np.asarray(mask) > 0
    if mask.ndim != 2:
        raise ValueError("mask_pca_axis_unavailable: mask must be 2D")
    ys, xs = np.where(mask)
    if len(xs) < MASK_PCA_MIN_PIXELS:
        raise ValueError(
            "mask_pca_too_few_points: "
            f"need at least {MASK_PCA_MIN_PIXELS}, got {len(xs)}"
        )
    uv = np.column_stack([xs, ys]).astype(np.float64)
    center = uv.mean(axis=0)
    covariance = np.cov((uv - center).T)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError("mask_pca_axis_unavailable: non-finite covariance")
    values, vectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
        raise ValueError("mask_pca_axis_unavailable: non-finite eigensystem")
    principal_index = int(np.argmax(values))
    principal_value = float(values[principal_index])
    if principal_value <= MASK_PCA_EPSILON:
        raise ValueError("mask_pca_degenerate: both eigenvalues are near zero")
    axis = vectors[:, principal_index]
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= MASK_PCA_EPSILON:
        raise ValueError("mask_pca_axis_unavailable: zero-norm principal axis")
    axis = axis / norm
    # The line is unoriented, so this only stabilizes diagnostic angles/images.
    if axis[1] < 0.0 or (abs(axis[1]) <= MASK_PCA_EPSILON and axis[0] < 0.0):
        axis = -axis
    ordered_values = np.sort(values.astype(np.float64))
    eigenvalue_ratio = float(
        ordered_values[-1] / max(float(ordered_values[0]), MASK_PCA_EPSILON)
    )
    return axis, center, {
        "mask_pca_axis_uv": axis.tolist(),
        "mask_pca_axis_angle_deg": _axis_angle_deg(axis),
        "mask_pca_eigenvalues": ordered_values.tolist(),
        "mask_pca_eigenvalue_ratio": eigenvalue_ratio,
        "mask_pca_pixel_count": int(len(xs)),
    }


def _slice_statistics(widths):
    values = np.asarray(widths, np.float64)
    if values.size == 0:
        return {
            "valid_slice_count": 0,
            "slice_width_mean_px": None,
            "slice_width_std_px": None,
            "slice_width_cv": None,
            "slice_width_min_px": None,
            "slice_width_max_px": None,
        }
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        "valid_slice_count": int(values.size),
        "slice_width_mean_px": mean,
        "slice_width_std_px": std,
        "slice_width_cv": None if mean <= MASK_PCA_EPSILON else float(std / mean),
        "slice_width_min_px": float(np.min(values)),
        "slice_width_max_px": float(np.max(values)),
    }


def _mask_width_pixel_modes(mask, axis, center):
    """Measure all three pixel widths from one mask and one PCA frame."""
    mask = np.asarray(mask) > 0
    ys, xs = np.where(mask)
    points_xy = np.column_stack([xs, ys]).astype(np.float64)
    centered_xy = points_xy - np.asarray(center, np.float64).reshape(2)
    long_axis = np.asarray(axis, np.float64).reshape(2)
    width_axis = np.array([-long_axis[1], long_axis[0]], np.float64)
    long_coordinate = centered_xy @ long_axis
    width_coordinate = centered_xy @ width_axis

    global_low, global_high = np.percentile(width_coordinate, [2.0, 98.0])
    global_width = float(max(0.0, global_high - global_low))
    modes = {
        MASK_PCA_WIDTH_MODE_GLOBAL: {
            "ok": bool(np.isfinite(global_width)),
            "reason": None if np.isfinite(global_width) else "width_non_finite",
            "pixel_width": global_width,
            "projection_low_px": float(global_low),
            "projection_high_px": float(global_high),
            "percentiles": [2.0, 98.0],
            "aggregation": "global_projection_span",
            **_slice_statistics([]),
            "slices": [],
        }
    }

    long_min = float(np.min(long_coordinate))
    long_max = float(np.max(long_coordinate))
    span = max(0.0, long_max - long_min)
    bin_size = float(MASK_SLICE_BIN_SIZE_PX)
    bin_count = max(1, int(np.ceil(span / bin_size)))
    edges = long_min + np.arange(bin_count + 1, dtype=np.float64) * bin_size
    if edges[-1] < long_max:
        edges = np.append(edges, long_max)
        bin_count += 1
    else:
        edges[-1] = max(edges[-1], long_max)

    p2p98_widths = []
    minmax_widths = []
    p2p98_slices = []
    minmax_slices = []
    for index in range(bin_count):
        start = float(edges[index])
        end = float(edges[index + 1])
        if index == bin_count - 1:
            in_slice = (long_coordinate >= start) & (long_coordinate <= end)
        else:
            in_slice = (long_coordinate >= start) & (long_coordinate < end)
        local_width = width_coordinate[in_slice]
        pixel_count = int(local_width.size)
        common = {
            "slice_index": int(index),
            "long_start_px": start,
            "long_end_px": end,
            "long_center_px": float(0.5 * (start + end)),
            "pixel_count": pixel_count,
        }
        if pixel_count < MASK_SLICE_MIN_PIXELS:
            excluded = {
                **common,
                "valid": False,
                "reason": "too_few_points_in_slice",
                "width_px": None,
                "width_low_px": None,
                "width_high_px": None,
            }
            p2p98_slices.append(dict(excluded))
            minmax_slices.append(dict(excluded))
            continue

        p2, p98 = np.percentile(local_width, [2.0, 98.0])
        local_min = float(np.min(local_width))
        local_max = float(np.max(local_width))
        p_width = float(max(0.0, p98 - p2))
        m_width = float(max(0.0, local_max - local_min))
        finite = bool(np.isfinite(p_width) and np.isfinite(m_width))
        reason = None if finite else "slice_width_non_finite"
        p2p98_slices.append({
            **common,
            "valid": finite,
            "reason": reason,
            "width_px": p_width if finite else None,
            "width_low_px": float(p2) if finite else None,
            "width_high_px": float(p98) if finite else None,
        })
        minmax_slices.append({
            **common,
            "valid": finite,
            "reason": reason,
            "width_px": m_width if finite else None,
            "width_low_px": local_min if finite else None,
            "width_high_px": local_max if finite else None,
        })
        if finite:
            p2p98_widths.append(p_width)
            minmax_widths.append(m_width)

    def slice_mode(widths, slices, aggregation):
        stats = _slice_statistics(widths)
        enough = stats["valid_slice_count"] >= MASK_SLICE_MIN_VALID_COUNT
        return {
            "ok": bool(enough),
            "reason": None if enough else "no_valid_mask_slices",
            "pixel_width": None if not enough else float(np.median(widths)),
            "aggregation": aggregation,
            "slice_bin_size_px": bin_size,
            "min_pixels_in_slice": int(MASK_SLICE_MIN_PIXELS),
            "min_valid_slice_count": int(MASK_SLICE_MIN_VALID_COUNT),
            "end_exclusion_px": 0.0,
            "long_coordinate_min_px": long_min,
            "long_coordinate_max_px": long_max,
            "total_slice_count": int(bin_count),
            **stats,
            "slices": slices,
        }

    modes[MASK_PCA_WIDTH_MODE_SLICE_P2P98] = slice_mode(
        p2p98_widths,
        p2p98_slices,
        "median_of_per_slice_p2p98",
    )
    modes[MASK_PCA_WIDTH_MODE_SLICE_MINMAX] = slice_mode(
        minmax_widths,
        minmax_slices,
        "median_of_per_slice_minmax",
    )
    return {
        "long_axis": long_axis.tolist(),
        "width_axis": width_axis.tolist(),
        "center_xy": np.asarray(center, np.float64).reshape(2).tolist(),
        "mask_pixel_count": int(points_xy.shape[0]),
        "modes": modes,
    }


def _axis_and_center(mask, axis=None, center=None, *, use_mask_pca_axis=False):
    mask = np.asarray(mask) > 0
    ys, xs = np.where(mask)
    if use_mask_pca_axis:
        if axis is not None:
            raise ValueError("mask_pca mode does not accept selected_axis")
        axis, center, mask_pca = _mask_pca_axis(mask)
        return axis, center, "mask_pca", mask_pca
    if len(xs) < MASK_PCA_MIN_PIXELS:
        raise ValueError("too few mask pixels")
    if axis is None:
        axis, pca_center, mask_pca = _mask_pca_axis(mask)
        if center is None:
            center = pca_center
        return axis, np.asarray(center, np.float64).reshape(2), "mask_pca", mask_pca
    axis = np.asarray(axis, np.float64).reshape(2)
    norm = float(np.linalg.norm(axis))
    if not np.all(np.isfinite(axis)) or not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("selected_axis is non-finite or has zero norm")
    axis = axis / norm
    if center is None:
        center = np.array([xs.mean(), ys.mean()], np.float64)
    return axis, np.asarray(center, np.float64).reshape(2), "selected_ocr_axis", None


def _camera_value(camera_intrinsics, name, *, positive=False):
    if name not in camera_intrinsics:
        raise ValueError(f"camera_intrinsics_missing_or_invalid: {name}")
    try:
        value = float(camera_intrinsics[name])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"camera_intrinsics_missing_or_invalid: {name}"
        ) from exc
    if not np.isfinite(value) or (positive and value <= 0.0):
        raise ValueError(f"camera_intrinsics_missing_or_invalid: {name}")
    return value


def estimate_mask_pca_width_modes(mask, depth_image, camera_intrinsics):
    """Return all width modes from one mask PCA and one metric scale."""
    mask = np.asarray(mask) > 0
    depth = np.asarray(depth_image)
    if mask.ndim != 2 or depth.shape != mask.shape:
        raise ValueError("mask/depth shape mismatch")
    axis, center, mask_pca = _mask_pca_axis(mask)
    pixel_result = _mask_width_pixel_modes(mask, axis, center)
    normal = np.asarray(pixel_result["width_axis"], np.float64)

    valid = mask & np.isfinite(depth) & (depth > 0)
    if int(valid.sum()) < 20:
        raise ValueError("too few valid Depth pixels for SAM2 width")
    depth_scale = _camera_value(camera_intrinsics, "depth_scale", positive=True)
    representative_depth_m = float(np.median(depth[valid]) * depth_scale)
    if not np.isfinite(representative_depth_m) or representative_depth_m <= 0.0:
        raise ValueError("invalid_representative_depth")
    fx = _camera_value(camera_intrinsics, "fx", positive=True)
    fy = _camera_value(camera_intrinsics, "fy", positive=True)
    scale_m_per_px = representative_depth_m * float(np.sqrt(
        (normal[0] / fx) ** 2 + (normal[1] / fy) ** 2
    ))

    modes = {}
    for mode, value in pixel_result["modes"].items():
        item = dict(value)
        pixel_width = item.get("pixel_width")
        item["width_mm"] = (
            None
            if pixel_width is None
            else float(pixel_width) * scale_m_per_px * 1000.0
        )
        modes[mode] = item
    return {
        "selected_width_axis_source": "mask_pca",
        **mask_pca,
        "mask_center_xy": pixel_result["center_xy"],
        "width_axis_uv": pixel_result["width_axis"],
        "representative_depth_m": representative_depth_m,
        "median_depth_raw": float(np.median(depth[valid])),
        "valid_depth_pixel_count": int(valid.sum()),
        "scale_m_per_px": scale_m_per_px,
        "modes": modes,
    }


def _project_points(points, intrinsics):
    points = np.asarray(points, np.float64)
    z = points[:, 2]
    fx = _camera_value(intrinsics, "fx", positive=True)
    fy = _camera_value(intrinsics, "fy", positive=True)
    ppx = _camera_value(intrinsics, "ppx")
    ppy = _camera_value(intrinsics, "ppy")
    u = fx * points[:, 0] / z + ppx
    v = fy * points[:, 1] / z + ppy
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
    use_mask_pca_axis=False,
    diagnostic_ocr_axis=None,
    mask_width_mode=DEFAULT_MASK_PCA_WIDTH_MODE,
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
    if mask_width_mode not in MASK_PCA_WIDTH_MODES:
        raise ValueError(f"unknown mask_width_mode: {mask_width_mode}")
    mask = np.asarray(mask) > 0
    depth = np.asarray(depth_image)
    points = np.asarray(points, np.float64)
    rgb = np.asarray(rgb_image)
    if mask.shape != depth.shape or rgb.shape[:2] != mask.shape:
        raise ValueError("mask/rgb/depth shape mismatch")
    if len(points) < 3:
        raise ValueError("too few final geometry points")

    axis, center, axis_source, mask_pca = _axis_and_center(
        mask,
        selected_axis,
        selected_center,
        use_mask_pca_axis=use_mask_pca_axis,
    )
    ocr_axis = _optional_diagnostic_axis(diagnostic_ocr_axis)
    mask_axis_angle_deg = None if mask_pca is None else mask_pca[
        "mask_pca_axis_angle_deg"
    ]
    ocr_axis_angle_deg = _axis_angle_deg(ocr_axis)
    ocr_mask_axis_angle_diff_deg = _axial_angle_difference_deg(
        ocr_axis,
        None if mask_pca is None else np.asarray(mask_pca["mask_pca_axis_uv"]),
    )
    normal = np.array([-axis[1], axis[0]], np.float64)
    if mask_width_mode != MASK_PCA_WIDTH_MODE_GLOBAL and axis_source != "mask_pca":
        raise ValueError("slice mask width modes require mask_pca axis")
    pixel_width_modes = _mask_width_pixel_modes(mask, axis, center)
    selected_pixel_width = pixel_width_modes["modes"][mask_width_mode]
    if not selected_pixel_width["ok"]:
        raise ValueError(str(selected_pixel_width["reason"]))
    width_px = float(selected_pixel_width["pixel_width"])
    global_pixel_width = pixel_width_modes["modes"][MASK_PCA_WIDTH_MODE_GLOBAL]
    low = float(global_pixel_width["projection_low_px"])
    high = float(global_pixel_width["projection_high_px"])

    valid = mask & np.isfinite(depth) & (depth > 0)
    ys, xs = np.where(valid)
    if len(xs) < 20:
        raise ValueError("too few valid Depth pixels for SAM2 width")
    depth_scale = _camera_value(camera_intrinsics, "depth_scale", positive=True)
    representative_depth_m = float(np.median(depth[valid]) * depth_scale)
    if not np.isfinite(representative_depth_m) or representative_depth_m <= 0.0:
        raise ValueError("invalid_representative_depth")
    fx = _camera_value(camera_intrinsics, "fx", positive=True)
    fy = _camera_value(camera_intrinsics, "fy", positive=True)
    scale_m_per_px = representative_depth_m * float(np.sqrt(
        (normal[0] / fx) ** 2
        + (normal[1] / fy) ** 2
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
    if not use_width:
        width_method = "sam3_current_3d_pca_pc2_slices"
    elif axis_source == "mask_pca":
        width_method = f"mask_pca_{mask_width_mode}_pixel_width_to_metric"
    else:
        width_method = "sam2_compatible_filtered_mask_axis_pixel_width_to_metric"

    result = {
        "geometry_mode": geometry_mode,
        "mask_stage": "sam3_conservative_refinement_output_mapped_to_sam2_final_mask01",
        "selected_width_axis_source": axis_source,
        "mask_pca_axis_uv": None if mask_pca is None else mask_pca[
            "mask_pca_axis_uv"
        ],
        "mask_pca_axis_angle_deg": mask_axis_angle_deg,
        "mask_pca_eigenvalues": None if mask_pca is None else mask_pca[
            "mask_pca_eigenvalues"
        ],
        "mask_pca_eigenvalue_ratio": None if mask_pca is None else mask_pca[
            "mask_pca_eigenvalue_ratio"
        ],
        "mask_pca_pixel_count": None if mask_pca is None else mask_pca[
            "mask_pca_pixel_count"
        ],
        "ocr_axis_uv": None if ocr_axis is None else ocr_axis.tolist(),
        "ocr_axis_angle_deg": ocr_axis_angle_deg,
        "ocr_mask_axis_angle_diff_deg": ocr_mask_axis_angle_diff_deg,
        "pixel_width": width_px,
        "width_mm": None if final_width is None else float(final_width),
        "mask_width_mode": mask_width_mode,
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
            "method": width_method,
            "pixel_width": width_px,
            "representative_depth_m": representative_depth_m,
            "scale_m_per_px": scale_m_per_px,
            "width_mm": None if final_width is None else float(final_width),
            "sam2_candidate_width_mm": width_mm,
            "axis": axis.tolist(),
            "normal": normal.tolist(),
            "axis_source": axis_source,
            "mask_width_mode": mask_width_mode,
            "selected_width_axis_source": axis_source,
            "mask_pca_axis_uv": None if mask_pca is None else mask_pca[
                "mask_pca_axis_uv"
            ],
            "mask_pca_axis_angle_deg": mask_axis_angle_deg,
            "mask_pca_eigenvalues": None if mask_pca is None else mask_pca[
                "mask_pca_eigenvalues"
            ],
            "mask_pca_eigenvalue_ratio": None if mask_pca is None else mask_pca[
                "mask_pca_eigenvalue_ratio"
            ],
            "mask_pca_pixel_count": None if mask_pca is None else mask_pca[
                "mask_pca_pixel_count"
            ],
            "ocr_axis_uv": None if ocr_axis is None else ocr_axis.tolist(),
            "ocr_axis_angle_deg": ocr_axis_angle_deg,
            "ocr_mask_axis_angle_diff_deg": ocr_mask_axis_angle_diff_deg,
            "valid_slice_count": selected_pixel_width.get("valid_slice_count"),
            "slice_width_mean_px": selected_pixel_width.get("slice_width_mean_px"),
            "slice_width_std_px": selected_pixel_width.get("slice_width_std_px"),
            "slice_width_cv": selected_pixel_width.get("slice_width_cv"),
            "slice_width_min_px": selected_pixel_width.get("slice_width_min_px"),
            "slice_width_max_px": selected_pixel_width.get("slice_width_max_px"),
            "slice_bin_size_px": selected_pixel_width.get("slice_bin_size_px"),
            "min_pixels_in_slice": selected_pixel_width.get("min_pixels_in_slice"),
            "end_exclusion_px": selected_pixel_width.get("end_exclusion_px"),
            "slices": selected_pixel_width.get("slices", []),
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
            "mask_pca_width_modes": pixel_width_modes["modes"],
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
    def add_lines(image, lines):
        for index, line in enumerate(lines):
            cv2.putText(
                image, line, (12, 26 + 24 * index), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 3, cv2.LINE_AA
            )
            cv2.putText(
                image, line, (12, 26 + 24 * index), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (0, 0, 0), 1, cv2.LINE_AA
            )

    width = result["width"]
    diagnostic_lines = [
        f"axis source: {result['selected_width_axis_source']}",
        f"width mode: {width['mask_width_mode']}",
        "mask PCA angle: n/a" if result["mask_pca_axis_angle_deg"] is None
        else f"mask PCA angle: {result['mask_pca_axis_angle_deg']:.3f} deg",
        "OCR angle: n/a" if result["ocr_axis_angle_deg"] is None
        else f"OCR angle: {result['ocr_axis_angle_deg']:.3f} deg",
        "angle difference: n/a" if result["ocr_mask_axis_angle_diff_deg"] is None
        else f"angle difference: {result['ocr_mask_axis_angle_diff_deg']:.3f} deg",
        f"width: {width['pixel_width']:.3f} px / {width['sam2_candidate_width_mm']:.3f} mm",
    ]

    def axis_image(vector, color, name, lines=()):
        image = overlay.copy()
        p0 = tuple(np.rint(center - vector * scale).astype(int))
        p1 = tuple(np.rint(center + vector * scale).astype(int))
        cv2.line(image, p0, p1, color, 3)
        add_lines(image, lines)
        cv2.imwrite(str(out / name), image)
    axis_image(axis, (255, 0, 0), "sam2_long_axis_overlay.png", diagnostic_lines)
    axis_image(axis, (255, 0, 0), "mask_pca_axis_overlay.png", diagnostic_lines)
    axis_image(axis, (255, 0, 0), "selected_width_axis_overlay.png", diagnostic_lines)
    axis_image(normal, (0, 255, 255), "sam2_width_axis_overlay.png")
    axis_image(axis, (255, 0, 0), "sam2_roll_overlay.png", diagnostic_lines)
    if result["ocr_axis_uv"] is not None:
        axis_image(
            np.asarray(result["ocr_axis_uv"], np.float64),
            (0, 165, 255),
            "ocr_axis_overlay.png",
            diagnostic_lines,
        )
    width_image = overlay.copy()
    slices = width.get("slices") or []
    if slices:
        boundary_values = [item["long_start_px"] for item in slices]
        boundary_values.append(slices[-1]["long_end_px"])
        for value in boundary_values:
            point = center + axis * float(value)
            p0 = tuple(np.rint(point - normal * 70.0).astype(int))
            p1 = tuple(np.rint(point + normal * 70.0).astype(int))
            cv2.line(width_image, p0, p1, (128, 128, 128), 1, cv2.LINE_AA)
        for item in slices:
            long_value = float(item["long_center_px"])
            midpoint = center + axis * long_value
            if not item["valid"]:
                point = tuple(np.rint(midpoint).astype(int))
                cv2.drawMarker(
                    width_image, point, (0, 0, 255), cv2.MARKER_TILTED_CROSS,
                    8, 1, cv2.LINE_AA
                )
                continue
            low_value = float(item["width_low_px"])
            high_value = float(item["width_high_px"])
            p0 = tuple(np.rint(midpoint + normal * low_value).astype(int))
            p1 = tuple(np.rint(midpoint + normal * high_value).astype(int))
            cv2.line(width_image, p0, p1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(
                width_image, f"{float(item['width_px']):.1f}", p1,
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA
            )
    else:
        for value, color in [(low, (255, 255, 0)), (high, (0, 255, 255))]:
            point = center + normal * value
            p0 = tuple(np.rint(point - axis * scale).astype(int))
            p1 = tuple(np.rint(point + axis * scale).astype(int))
            cv2.line(width_image, p0, p1, color, 2)
    add_lines(width_image, diagnostic_lines)
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
