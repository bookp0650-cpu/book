#!/usr/bin/env python3
"""Comparison variant: conservative SAM3 mask refinement + flat median depth.

This file is intentionally isolated from the stable and integration paths.
The default mode never applies a depth-range filter or RANSAC to PCA input.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import get_book_points as current
from . import get_book_points_no_mask_merge_no_side_filter as stable
from .modules.book_width import estimate_book_width
from .modules.grip_point import (
    find_target_point,
    find_target_point_longitudinal,
)
from .modules.pca_vector import pca_axes_fix_dir
from .modules.pointcloud_utils import save_ply_ascii
from .sam3_mask_refinement import refine_selected_sam3_mask


VARIANT = "sam3_refined_median_depth"
MODES = {
    "baseline",
    "refine_only",
    "median_flatten_only",
    "refine_and_median_flatten",
}
DEFAULT_MODE = "refine_and_median_flatten"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CAPTURES_DIR = PROJECT_ROOT / "captures"


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _camera_point_to_uv(point_m, intr):
    """Project one camera-frame point with the intrinsics used for 3D recovery."""
    point = np.asarray(point_m, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(point)) or point[2] <= 0.0:
        return None
    u = float(intr.fx) * point[0] / point[2] + float(intr.ppx)
    v = float(intr.fy) * point[1] / point[2] + float(intr.ppy)
    return int(round(u)), int(round(v))


def _save_target_point_before_after(
    output_path,
    color_np,
    mask01,
    old_target_uv,
    new_target_uv,
    new_target_m,
    book_up_vector,
    intr,
):
    """Save a separate old/new target comparison without changing final.png."""
    color = np.asarray(color_np)
    mask = np.asarray(mask01) > 0
    green = color.copy()
    green[mask] = (0, 255, 0)
    image = cv2.addWeighted(color, 0.75, green, 0.25, 0)

    new_uv = tuple(int(value) for value in new_target_uv)
    old_uv = (
        None
        if old_target_uv is None
        else tuple(int(value) for value in old_target_uv)
    )
    if old_uv is not None:
        cv2.line(image, old_uv, new_uv, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(image, old_uv, 7, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            "OLD camera-Y +100mm",
            (old_uv[0] + 10, old_uv[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.circle(image, new_uv, 8, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(image, new_uv, 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(
        image,
        "NEW bottom +75mm",
        (new_uv[0] + 10, new_uv[1] + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    arrow_end_m = np.asarray(new_target_m, dtype=np.float64).reshape(3) + (
        0.050 * np.asarray(book_up_vector, dtype=np.float64).reshape(3)
    )
    arrow_end_uv = _camera_point_to_uv(arrow_end_m, intr)
    if arrow_end_uv is not None:
        cv2.arrowedLine(
            image,
            new_uv,
            arrow_end_uv,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
            tipLength=0.2,
        )
        cv2.putText(
            image,
            "BOOK UP",
            (arrow_end_uv[0] + 8, arrow_end_uv[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(str(Path(output_path)), image):
        raise RuntimeError(f"failed to save target comparison: {output_path}")


def _target_longitudinal_log(
    target_info,
    target_uv,
    old_target,
    old_target_uv,
):
    """Build and print the auditable camera-frame target-point log."""
    origin = np.asarray(target_info["origin_m"], dtype=np.float64).reshape(3)
    book_up = np.asarray(
        target_info["book_up_vector"], dtype=np.float64
    ).reshape(3)
    target = np.asarray(target_info["target_m"], dtype=np.float64).reshape(3)
    bottom = float(target_info["bottom_longitudinal_m"])
    old_height = None
    old_distance_mm = None
    if old_target is not None:
        old = np.asarray(old_target, dtype=np.float64).reshape(3)
        old_height = float(np.dot(old - origin, book_up) - bottom)
        old_distance_mm = float(np.linalg.norm(target - old) * 1000.0)

    tilt_xy_deg = float(np.degrees(np.arctan2(book_up[0], book_up[1])))
    payload = {
        "coordinate_frame": "camera",
        "unit": "meter",
        "pc1_raw": np.asarray(target_info["pc1_raw"], float).tolist(),
        "book_up_vector": book_up.tolist(),
        "book_tilt_camera_xy_deg": tilt_xy_deg,
        "bottom_method": target_info["bottom_method"],
        "bottom_percentile": float(target_info["bottom_percentile"]),
        "origin_m": origin.tolist(),
        "bottom_longitudinal_m": bottom,
        "target_height_m": float(target_info["target_height_m"]),
        "selected_height_from_bottom_m": float(
            target_info["selected_height_from_bottom_m"]
        ),
        "candidate_tolerance_m": float(
            target_info["candidate_tolerance_m"]
        ),
        "used_tolerance_m": target_info["used_tolerance_m"],
        "candidate_count": int(target_info["candidate_count"]),
        "fallback_method": target_info["fallback_method"],
        "target_xyz_m": target.tolist(),
        "target_uv": [int(value) for value in target_uv],
        "old_target_xyz_m": (
            None
            if old_target is None
            else np.asarray(old_target, dtype=float).tolist()
        ),
        "old_target_uv": (
            None
            if old_target_uv is None
            else [int(value) for value in old_target_uv]
        ),
        "old_longitudinal_height_m": old_height,
        "new_longitudinal_height_m": float(
            target_info["selected_height_from_bottom_m"]
        ),
        "old_to_new_distance_mm": old_distance_mm,
    }
    print("[TARGET POINT LONGITUDINAL]")
    for key, value in payload.items():
        print(f"{key:<25}: {value}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _polygon_mask(shape, polygon):
    out = np.zeros(shape, np.uint8)
    pts = np.asarray(polygon if polygon is not None else [], np.float64)
    if pts.size:
        pts = pts.reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(out, [np.rint(pts).astype(np.int32)], 1)
    return out.astype(bool)


def _save_depth_visualization(path: Path, depth, mask=None) -> None:
    array = np.asarray(depth, np.float64)
    valid = np.isfinite(array) & (array > 0)
    if mask is not None:
        valid &= np.asarray(mask, bool)
    vis = np.zeros(array.shape, np.uint8)
    if valid.any():
        low, high = np.percentile(array[valid], [2, 98])
        if high <= low:
            high = low + 1.0
        vis[valid] = np.clip((array[valid] - low) * 255.0 / (high - low), 0, 255)
    colored = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    cv2.imwrite(str(path), colored)


def _representative_depth(
    depth_raw: np.ndarray,
    final_mask: np.ndarray,
    anchor_mask: np.ndarray,
    ocr_polygon,
):
    ocr_mask = _polygon_mask(final_mask.shape, ocr_polygon)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(anchor_mask.astype(np.uint8), kernel, iterations=1) > 0
    candidates = [
        ("anchor_eroded_3x3", eroded),
        ("anchor_ocr_intersection", anchor_mask & ocr_mask),
        ("anchor_component", anchor_mask),
        ("refined_mask", final_mask),
    ]
    selected = None
    selected_values = None
    for name, region in candidates:
        values = np.asarray(depth_raw)[region]
        values = values[np.isfinite(values) & (values > 0)].astype(np.float64)
        if values.size >= 30:
            selected = (name, region)
            selected_values = values
            break
    if selected is None:
        raise RuntimeError(
            "No representative median depth: fewer than 30 valid pixels in all "
            "conservative source regions."
        )
    median = float(np.median(selected_values))
    return {
        "source": selected[0],
        "region": selected[1],
        "values": selected_values,
        "median_raw": median,
        "mad_raw": float(np.median(np.abs(selected_values - median))),
        "p10_raw": float(np.percentile(selected_values, 10)),
        "p90_raw": float(np.percentile(selected_values, 90)),
    }


def _selected_ocr_geometry(mask, merged, image_shape, shot_dir, query):
    # This existing helper is used only to obtain the OCR polygon and its axis.
    # Its returned refined mask is deliberately ignored.
    _, info = current.refine_mask_by_ocr_axis_band(
        mask01=mask.astype(np.uint8),
        merged=merged,
        image_shape=image_shape,
        shot_dir=shot_dir,
        query=query,
        mask_width_ratio=1.05,
        min_keep_ratio=0.85,
        use_ocr_short_width=False,
        ocr_short_to_half_width_scale=1.35,
        min_ocr_half_width_px=28.0,
        suppress_side_protrusions=False,
        profile_bin_size_px=8.0,
        profile_width_margin=1.15,
        min_profile_half_width_px=16.0,
        max_profile_shrink_ratio=0.35,
        return_info=True,
    )
    selected = info.get("selected_ocr_polygon") if isinstance(info, dict) else None
    polygon = selected.get("points") if isinstance(selected, dict) else None
    if polygon is None and isinstance(selected, dict):
        polygon = selected.get("polygon")
    if polygon is None and isinstance(selected, dict):
        polygon = selected.get("poly")
    if polygon is None:
        raise RuntimeError("selected OCR polygon is unavailable; conservative refinement cannot anchor")
    return polygon, info


def _depth_and_pca(
    *,
    mode,
    shot_dir,
    color_np,
    depth_raw,
    raw_mask,
    used_mask,
    anchor_mask,
    ocr_polygon,
    intr,
    depth_scale,
    depth_merge_tolerance_raw,
):
    flatten = mode in {"median_flatten_only", "refine_and_median_flatten"}
    if flatten:
        median = _representative_depth(
            depth_raw, used_mask, anchor_mask, ocr_polygon
        )
        source_region = median["region"]
        cv2.imwrite(
            str(shot_dir / "depth_median_source_region.png"),
            source_region.astype(np.uint8) * 255,
        )
        flattened = np.zeros_like(depth_raw)
        flattened[used_mask] = np.asarray(median["median_raw"], dtype=depth_raw.dtype)
        np.save(shot_dir / "target_mask_depth_flattened.npy", flattened)
        _save_depth_visualization(
            shot_dir / "depth_flattened_visualization.png", flattened, used_mask
        )
        points, uv = current._mask_depth_to_points_uv_for_plane_filter(
            used_mask.astype(np.uint8), flattened, intr, depth_scale
        )
        colors = color_np[uv[:, 1], uv[:, 0], ::-1].astype(np.uint8)
        save_ply_ascii(
            shot_dir / "pointcloud_from_flattened_mask.ply", points, colors
        )
        # Exact same arrays, without filtering, are passed to PCA.
        points_for_pca = points
        uv_for_pca = uv
        colors_for_pca = colors
        save_ply_ascii(shot_dir / "pointcloud_sent_to_pca.ply", points_for_pca, colors_for_pca)
        flatten_info = {
            "enabled": True,
            "median_source": median["source"],
            "valid_depth_count": int(median["values"].size),
            "valid_depth_ratio": float(
                median["values"].size / max(int(source_region.sum()), 1)
            ),
            "median_depth_raw": median["median_raw"],
            "median_depth_m": float(median["median_raw"] * depth_scale),
            "mad": median["mad_raw"],
            "p10": median["p10_raw"],
            "p90": median["p90_raw"],
            "raw_mask_area_px": int(raw_mask.sum()),
            "final_mask_area_px": int(used_mask.sum()),
            "flattened_pixel_count": int(np.count_nonzero(flattened)),
            "pointcloud_point_count": int(points.shape[0]),
            "pca_input_point_count": int(points_for_pca.shape[0]),
            "all_final_mask_pixels_assigned_median": bool(
                np.all(flattened[used_mask] == flattened[used_mask][0])
            ),
            "pointcloud_equals_pca_input": bool(
                np.array_equal(points, points_for_pca)
            ),
            "pointcloud_ply_equals_pca_ply": bool(
                _sha256(shot_dir / "pointcloud_from_flattened_mask.ply")
                == _sha256(shot_dir / "pointcloud_sent_to_pca.ply")
            ),
            "fallback": False,
            "fallback_reason": None,
        }
        final_depth = flattened
    else:
        # refine_only deliberately retains the stable depth-range and one
        # normal-RANSAC path so that refinement can be isolated.
        depth_median, depth_info = current.save_masked_and_cropped(
            color_np,
            depth_raw,
            used_mask.astype(np.uint8),
            shot_dir,
            "refine_only_median_depth_filter",
            z_tolerance_raw=int(depth_merge_tolerance_raw),
            return_info=True,
        )
        depth_mask = used_mask & (depth_median > 0)
        points, uv = current._mask_depth_to_points_uv_for_plane_filter(
            depth_mask.astype(np.uint8), depth_median, intr, depth_scale
        )
        colors = color_np[uv[:, 1], uv[:, 0], ::-1].astype(np.uint8)
        _, inliers, ransac_info = current._fit_plane_ransac_open3d_for_spine(
            points, distance_threshold_m=0.008, ransac_n=3, num_iterations=1200
        )
        if not ransac_info.get("used"):
            raise RuntimeError(f"refine_only normal RANSAC failed: {ransac_info}")
        points_for_pca = points[inliers]
        uv_for_pca = uv[inliers]
        colors_for_pca = colors[inliers]
        save_ply_ascii(shot_dir / "pointcloud_sent_to_pca.ply", points_for_pca, colors_for_pca)
        final_depth = depth_median
        flatten_info = {
            "enabled": False,
            "median_source": depth_info.get("depth_reference_source_used"),
            "valid_depth_count": int(points.shape[0]),
            "valid_depth_ratio": float(points.shape[0] / max(int(used_mask.sum()), 1)),
            "median_depth_raw": depth_info.get("median_depth_raw"),
            "median_depth_m": None,
            "mad": None,
            "p10": None,
            "p90": None,
            "raw_mask_area_px": int(raw_mask.sum()),
            "final_mask_area_px": int(used_mask.sum()),
            "flattened_pixel_count": None,
            "pointcloud_point_count": int(points.shape[0]),
            "pca_input_point_count": int(points_for_pca.shape[0]),
            "all_final_mask_pixels_assigned_median": False,
            "pointcloud_equals_pca_input": False,
            "fallback": False,
            "fallback_reason": None,
            "normal_ransac": ransac_info,
        }

    if points_for_pca.shape[0] < 3:
        raise RuntimeError(f"insufficient PCA points: {points_for_pca.shape[0]}")
    mean, pc1, pc2 = pca_axes_fix_dir(points_for_pca)
    norm_xy = float(np.hypot(float(pc1[0]), float(pc1[1])))
    roll = 0.0 if norm_xy < 1e-8 else float(np.arctan2(pc1[1], pc1[0]))
    width_info = estimate_book_width(points_for_pca, mean, pc1, pc2)
    width_m = width_info.get("av_book_width_m")
    if width_m is None:
        raise RuntimeError(f"book width estimation failed: {width_info}")
    # Keep the old camera-Y target only for comparison. The formal target is
    # selected directly at 75 mm from the robust book bottom along pc1.
    old_target_info = find_target_point(points_for_pca)
    old_target = old_target_info.get("target_m")
    target_info = find_target_point_longitudinal(points_for_pca, pc1)
    target = target_info.get("target_m")
    if target is None:
        raise RuntimeError(f"target point estimation failed: {target_info}")
    target_index = int(target_info["target_index"])
    target_uv_direct = tuple(int(value) for value in uv_for_pca[target_index])
    old_target_uv = None
    if old_target is not None:
        old_target_index = int(
            np.argmin(
                np.linalg.norm(
                    points_for_pca - np.asarray(old_target).reshape(3), axis=1
                )
            )
        )
        old_target_uv = tuple(
            int(value) for value in uv_for_pca[old_target_index]
        )
    # The final display mask is the actual component-refined/raw mask, not a
    # RANSAC visualization. Yellow contours are not drawn.
    target_uv = stable._save_final_png_with_target(
        shot_dir / "final.png",
        color_np,
        used_mask,
        np.asarray(target),
        points_for_pca,
        uv_for_pca,
    )
    if tuple(target_uv) != target_uv_direct:
        raise RuntimeError(
            "target point/UV correspondence mismatch: "
            f"selected={target_uv_direct}, visualized={target_uv}"
        )
    _save_target_point_before_after(
        shot_dir / "target_point_before_after.png",
        color_np,
        used_mask,
        old_target_uv,
        target_uv,
        target,
        target_info["book_up_vector"],
        intr,
    )
    target_longitudinal = _target_longitudinal_log(
        target_info,
        target_uv,
        old_target,
        old_target_uv,
    )
    _write_json(shot_dir / "depth_flattening_result.json", flatten_info)
    return {
        "roll_rad": roll,
        "point_3d": np.asarray(target, float).tolist(),
        "pred_book_width_mm": float(width_m * 1000.0),
        "width_info": width_info,
        "target_uv": list(target_uv),
        "target_longitudinal": target_longitudinal,
        "point_counts": {
            "pointcloud_from_mask": int(points.shape[0]),
            "sent_to_pca": int(points_for_pca.shape[0]),
        },
        "depth_flattening": flatten_info,
        "final_depth": final_depth,
    }


def run_capture_and_pca_offline_sam3_refined_median_depth(
    query,
    shot_dir,
    sam_device="gpu",
    *,
    mode=DEFAULT_MODE,
    intr=None,
    depth_scale=None,
    depth_merge_tolerance_raw=30,
):
    """Run one comparison mode on an existing RGB-D directory."""
    if mode not in MODES:
        raise ValueError(f"unsupported mode {mode!r}; expected one of {sorted(MODES)}")
    if mode == "baseline":
        return stable.run_capture_and_pca_offline_no_mask_merge_no_side_filter(
            query,
            shot_dir,
            sam_device=sam_device,
            intr=intr,
            depth_scale=depth_scale,
            depth_merge_tolerance_raw=depth_merge_tolerance_raw,
        )

    started = time.perf_counter()
    shot_dir = Path(shot_dir).expanduser().resolve()
    color_np = cv2.imread(str(shot_dir / "after_init_rgb.png"), cv2.IMREAD_COLOR)
    depth_raw = np.load(shot_dir / "after_init_depth.npy", allow_pickle=False)
    if color_np is None:
        raise FileNotFoundError(shot_dir / "after_init_rgb.png")
    if depth_raw.shape != color_np.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch: {color_np.shape}/{depth_raw.shape}")
    if intr is None or depth_scale is None:
        intr, depth_scale = stable._intrinsics()
    current._save_camera_params_json(shot_dir, intr, depth_scale)
    np.save(shot_dir / "after_init_depth_raw.npy", depth_raw)
    _save_depth_visualization(shot_dir / "depth_raw_visualization.png", depth_raw)

    ocr_proc = current.start_ocr_subprocess(shot_dir)
    print(f"[PARALLEL][{VARIANT}] OCR subprocess started")
    runner = current._get_sam_runner_compat(
        encoder_path="unused-by-sam3",
        decoder_path="unused-by-sam3",
        sam_device=sam_device,
        use_cache=True,
    )
    rgb_pil = Image.fromarray(cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))
    masks, sam_data = current._infer_masks_compat(
        runner,
        rgb_pil,
        current._make_stage_save_cfg_compat(shot_dir),
        depth_np_u16=depth_raw,
        depth_merge_tolerance_raw=depth_merge_tolerance_raw,
    )
    ocr_stdout = current.wait_ocr_subprocess(ocr_proc, timeout=120.0)
    if ocr_stdout.strip():
        print(ocr_stdout, end="" if ocr_stdout.endswith("\n") else "\n")
    merged = current.merge_ocr_and_masks(
        query=query, masks=masks, shot_dir=shot_dir, interactive=False, threshold=40
    )
    if not merged.get("results"):
        raise stable.TargetMaskSelectionError(
            "target OCR similarity did not exceed threshold 40"
        )
    selected_index = int(merged["sel_idx"])
    raw_mask = np.asarray(merged["mask01"], bool)
    score = float(sam_data[selected_index - 1]["score"])
    polygon, ocr_info = _selected_ocr_geometry(
        raw_mask, merged, color_np.shape[:2], shot_dir, query
    )

    use_refinement = mode in {"refine_only", "refine_and_median_flatten"}
    refined, refinement, metrics = refine_selected_sam3_mask(
        raw_mask,
        ocr_polygon=polygon,
        depth_raw=depth_raw,
        rgb_bgr=color_np,
        output_dir=shot_dir,
        mode=mode,
    )
    used_mask = refined if use_refinement else raw_mask.copy()
    if not use_refinement:
        refinement["mode"] = mode
        refinement["refinement_enabled"] = False
        refinement["no_op"] = True
        refinement["raw_and_refined_mask_equal"] = True
        refinement["refined_mask_area_px"] = int(raw_mask.sum())
        refinement["removed_area_px"] = 0
        refinement["kept_component_ids"] = list(
            range(1, refinement["raw_component_count"] + 1)
        )
        refinement["removed_component_ids"] = []
        cv2.imwrite(
            str(shot_dir / "selected_mask_refined.png"),
            raw_mask.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(shot_dir / "components_kept.png"),
            raw_mask.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(shot_dir / "components_removed.png"),
            np.zeros_like(raw_mask, dtype=np.uint8),
        )
        overlay = color_np.copy()
        green = color_np.copy()
        green[raw_mask] = (0, 255, 0)
        overlay = cv2.addWeighted(overlay, 0.65, green, 0.35, 0)
        cv2.imwrite(str(shot_dir / "selected_mask_refined_overlay.png"), overlay)
        _write_json(shot_dir / "mask_refinement_result.json", refinement)

    labels_count, labels = cv2.connectedComponents(
        raw_mask.astype(np.uint8), connectivity=8
    )
    anchor_mask = labels == int(refinement["anchor_component_id"])
    compute = _depth_and_pca(
        mode=mode,
        shot_dir=shot_dir,
        color_np=color_np,
        depth_raw=depth_raw,
        raw_mask=raw_mask,
        used_mask=used_mask,
        anchor_mask=anchor_mask,
        ocr_polygon=polygon,
        intr=intr,
        depth_scale=float(depth_scale),
        depth_merge_tolerance_raw=depth_merge_tolerance_raw,
    )
    ocr_result = json.loads((shot_dir / "ocr_result.json").read_text(encoding="utf-8"))
    selected_ocr = (ocr_info.get("selected_ocr_polygon") or {}).get("text")
    result = {
        "success": True,
        "variant": VARIANT,
        "mode": mode,
        "query": query,
        "selected_mask_index": selected_index,
        "selected_mask_score": score,
        "selected_ocr_text": selected_ocr,
        "selected_ocr_confidence": stable._selected_ocr_confidence(
            ocr_result, selected_ocr
        ),
        "raw_mask_area_px": int(raw_mask.sum()),
        "final_mask_area_px": int(used_mask.sum()),
        "raw_component_count": refinement["raw_component_count"],
        "anchor_component_id": refinement["anchor_component_id"],
        "kept_component_ids": refinement["kept_component_ids"],
        "removed_component_ids": refinement["removed_component_ids"],
        "raw_and_refined_mask_equal": bool(np.array_equal(raw_mask, used_mask)),
        "fallback": refinement["fallback_to_raw_mask"],
        "mask_refinement": refinement,
        **{key: value for key, value in compute.items() if key != "final_depth"},
        "processing_seconds": float(time.perf_counter() - started),
        "returned_shot_dir": str(shot_dir),
    }
    _write_json(shot_dir / "offline_recognition_result.json", result)
    return result


def run_capture_and_pca_sam3_refined_median_depth(
    query,
    sam_device="gpu",
    *,
    mode=DEFAULT_MODE,
    shot_dir=None,
):
    """Live-compatible public API. This function is not used by offline tests."""
    if shot_dir is None:
        stem = datetime.now().astimezone().strftime(
            "%Y%m%d_%H%M%S_live_sam3_refined_median_depth"
        )
        shot_dir = CAPTURES_DIR / stem
    shot_dir = Path(shot_dir).expanduser().resolve()
    shot_dir.mkdir(parents=True, exist_ok=False)
    _, _, intr, depth_scale, _ = stable.capture_rgbd_once_no_mask_merge_no_side_filter(
        shot_dir
    )
    result = run_capture_and_pca_offline_sam3_refined_median_depth(
        query,
        shot_dir,
        sam_device=sam_device,
        mode=mode,
        intr=intr,
        depth_scale=depth_scale,
    )
    return (
        float(result["roll_rad"]),
        np.asarray(result["point_3d"], dtype=float),
        float(result["pred_book_width_mm"]),
        shot_dir,
    )
