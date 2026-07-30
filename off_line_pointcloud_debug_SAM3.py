#!/usr/bin/env python3
"""100-case SAM3 refine_only point-cloud removal audit.

Derived from detection/pro_handbook/sam_py_demo/offline_pointcloud_debug.py.
The stable modules are imported, never modified. Every filtering stage calls
the exact helper used by the current refine_only candidate.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from detection.pro_handbook.sam_py_demo import get_book_points as current
from detection.pro_handbook.sam_py_demo import (
    get_book_points_no_mask_merge_no_side_filter as stable,
)
from detection.pro_handbook.sam_py_demo import (
    get_book_points_sam3_refined_median_depth as refined_variant,
)
from detection.pro_handbook.sam_py_demo.modules.book_width import estimate_book_width
from detection.pro_handbook.sam_py_demo.modules.grip_point import find_target_point
from detection.pro_handbook.sam_py_demo.modules.pca_vector import pca_axes_fix_dir
from detection.pro_handbook.sam_py_demo.modules.pointcloud_utils import save_ply_ascii
from detection.pro_handbook.sam_py_demo.sam3_mask_refinement import (
    refine_selected_sam3_mask,
)


BASE_DIR = Path("/home/book/pro_book_SAM3/pro_hand_book_python").resolve()
TEST_BASE_DIR = BASE_DIR / "captures" / "100test"
MASTER_JSON = BASE_DIR / "master_20260216.json"
OUTPUT_PARENT = BASE_DIR / "captures"
SAM_DEVICE = "gpu"
START_INDEX = 1
END_INDEX = 100
REPEATS_PER_BOOK = 5
ERROR_THRESHOLDS_MM = (1.0, 1.5, 2.0)
INPUT_FILES = ("after_init_rgb.png", "after_init_depth.npy")
SERVICE_URL = "http://127.0.0.1:8765/health"
SERVICE_PYTHON = (
    BASE_DIR / "detection/pro_handbook/sam3_runtime/.venv/bin/python"
)
SERVICE_MODULE = "detection.pro_handbook.sam3_runtime.service.service"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_master():
    data = json.loads(MASTER_JSON.read_text(encoding="utf-8"))
    return [
        {
            "book_name": item["book_name"],
            "book_width_mm": float(item["book_width"]),
            "raw": item,
        }
        for item in data
    ]


def book_for_case(master, case_index):
    master_index = (case_index - 1) // REPEATS_PER_BOOK
    repeat_index = (case_index - 1) % REPEATS_PER_BOOK + 1
    return master_index, repeat_index, master[master_index]


def make_run_root():
    stem = datetime.now().strftime("100test_offline_SAM3_debug_%Y%m%d_%H%M%S")
    candidate = OUTPUT_PARENT / stem
    suffix = 1
    while candidate.exists():
        candidate = OUTPUT_PARENT / f"{stem}_{suffix:03d}"
        suffix += 1
    candidate.mkdir(parents=False)
    return candidate


def copy_inputs(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    missing = []
    for name in INPUT_FILES:
        src = source / name
        dst = destination / name
        if not src.is_file():
            missing.append(str(src))
            continue
        shutil.copy2(src, dst)
        entries.append(
            {
                "name": name,
                "source": str(src),
                "destination": str(dst),
                "size_bytes": dst.stat().st_size,
                "sha256": sha256(dst),
            }
        )
    save_json(
        destination / "offline_input_manifest.json",
        {
            "source_shot_dir": str(source),
            "run_shot_dir": str(destination),
            "input_files": entries,
            "missing": missing,
        },
    )
    if missing:
        raise FileNotFoundError("offline input is missing: " + ", ".join(missing))


def service_health(timeout=3.0):
    try:
        with urllib.request.urlopen(SERVICE_URL, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def ensure_service(run_root):
    health = service_health()
    if health and health.get("ready"):
        return None, {"borrowed": True, "health": health}
    log_path = run_root / "sam3_service.log"
    log = log_path.open("w", encoding="utf-8")
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(BASE_DIR)
    process = subprocess.Popen(
        [str(SERVICE_PYTHON), "-m", SERVICE_MODULE],
        cwd=str(BASE_DIR),
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    deadline = time.monotonic() + 90.0
    last = None
    while time.monotonic() < deadline and process.poll() is None:
        time.sleep(1.0)
        last = service_health()
        if last and last.get("ready"):
            return process, {
                "borrowed": False,
                "health": last,
                "log": str(log_path),
                "_log_handle": log,
            }
        if last and last.get("error"):
            break
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    log.close()
    raise RuntimeError(
        "SAM3 service failed to become ready; CUDA/model/environment were not "
        f"changed. health={last}, log={log_path}"
    )


def stop_owned_service(process, info):
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    handle = info.get("_log_handle")
    if handle:
        handle.close()


def colors_for_uv(rgb_bgr, uv):
    uv = np.asarray(uv, np.int32).reshape(-1, 2)
    return rgb_bgr[uv[:, 1], uv[:, 0], ::-1].astype(np.uint8)


def save_mask_overlay(path, rgb, keep, remove=None):
    keep = np.asarray(keep, bool)
    remove = np.zeros_like(keep) if remove is None else np.asarray(remove, bool)
    tint = rgb.copy()
    tint[keep] = (0, 255, 0)
    tint[remove] = (0, 0, 255)
    cv2.imwrite(str(path), cv2.addWeighted(rgb, 0.65, tint, 0.35, 0))


def save_binary(path, mask):
    cv2.imwrite(str(path), np.asarray(mask, bool).astype(np.uint8) * 255)


def uv_mask(shape, uv):
    mask = np.zeros(shape, bool)
    uv = np.asarray(uv, np.int32).reshape(-1, 2)
    if uv.size:
        mask[uv[:, 1], uv[:, 0]] = True
    return mask


def region_distribution(removed, reference):
    removed = np.asarray(removed, bool)
    ys, xs = np.where(reference)
    if not len(xs):
        return {key: 0 for key in ("left_10pct", "right_10pct", "top_10pct", "bottom_10pct", "center")}
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    dx = max(1, int(math.ceil((x1 - x0 + 1) * 0.10)))
    dy = max(1, int(math.ceil((y1 - y0 + 1) * 0.10)))
    left = removed & (np.indices(removed.shape)[1] < x0 + dx)
    right = removed & (np.indices(removed.shape)[1] > x1 - dx)
    top = removed & (np.indices(removed.shape)[0] < y0 + dy)
    bottom = removed & (np.indices(removed.shape)[0] > y1 - dy)
    edge = left | right | top | bottom
    return {
        "left_10pct": int(left.sum()),
        "right_10pct": int(right.sum()),
        "top_10pct": int(top.sum()),
        "bottom_10pct": int(bottom.sum()),
        "center": int((removed & ~edge).sum()),
    }


def stage_panel(rgb, masks_and_labels):
    panels = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for mask, label, color in masks_and_labels:
        tint = rgb.copy()
        tint[np.asarray(mask, bool)] = color
        panel = cv2.addWeighted(rgb, 0.65, tint, 0.35, 0)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(panel, label, (12, 28), font, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(cv2.resize(panel, (384, 216), interpolation=cv2.INTER_AREA))
    top = np.hstack(panels[:4])
    bottom = np.hstack(
        panels[4:]
        + [np.zeros_like(panels[0]) for _ in range(4 - len(panels[4:]))]
    )
    return np.vstack([top, bottom])


def run_debug_case(query, shot_dir, gt_width_mm, case_index):
    shot_dir = Path(shot_dir)
    rgb = cv2.imread(str(shot_dir / "after_init_rgb.png"), cv2.IMREAD_COLOR)
    depth = np.load(shot_dir / "after_init_depth.npy", allow_pickle=False)
    if rgb is None or depth.shape != rgb.shape[:2]:
        raise ValueError(f"invalid RGB-D: {None if rgb is None else rgb.shape}/{depth.shape}")
    intr, depth_scale = stable._intrinsics()
    current._save_camera_params_json(shot_dir, intr, depth_scale)

    # Preserve production parallelism.
    ocr_proc = current.start_ocr_subprocess(shot_dir)
    print("[PARALLEL][100CASE DEBUG] OCR subprocess started")
    runner = current._get_sam_runner_compat(
        "unused-by-sam3", "unused-by-sam3", SAM_DEVICE, use_cache=True
    )
    rgb_pil = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    masks, sam_data = current._infer_masks_compat(
        runner,
        rgb_pil,
        current._make_stage_save_cfg_compat(shot_dir),
        depth_np_u16=depth,
        depth_merge_tolerance_raw=30,
    )
    ocr_stdout = current.wait_ocr_subprocess(ocr_proc, timeout=120)
    if ocr_stdout.strip():
        print(ocr_stdout, end="" if ocr_stdout.endswith("\n") else "\n")
    merged = current.merge_ocr_and_masks(
        query=query, masks=masks, shot_dir=shot_dir, interactive=False, threshold=40
    )
    if not merged.get("results"):
        raise stable.TargetMaskSelectionError("OCR/mask score did not exceed 40")
    selected_index = int(merged["sel_idx"])
    raw = np.asarray(merged["mask01"], bool)
    polygon, ocr_info = refined_variant._selected_ocr_geometry(
        raw, merged, rgb.shape[:2], shot_dir, query
    )
    refined, refinement, components = refine_selected_sam3_mask(
        raw,
        ocr_polygon=polygon,
        depth_raw=depth,
        rgb_bgr=rgb,
        output_dir=shot_dir,
        mode="refine_only",
    )
    removed_refinement = raw & ~refined
    save_binary(shot_dir / "mask_refinement_removed_pixels.png", removed_refinement)
    save_mask_overlay(
        shot_dir / "mask_refinement_removed_overlay.png",
        rgb,
        refined,
        removed_refinement,
    )

    # Stage 2: raw valid Depth before any range filtering.
    valid_depth = refined & np.isfinite(depth) & (depth > 0)
    invalid_depth = refined & ~valid_depth
    raw_points, raw_uv = current._mask_depth_to_points_uv_for_plane_filter(
        valid_depth.astype(np.uint8), depth, intr, depth_scale
    )
    raw_colors = colors_for_uv(rgb, raw_uv)
    save_ply_ascii(shot_dir / "pointcloud_mask_valid_depth_raw.ply", raw_points, raw_colors)
    save_ply_ascii(shot_dir / "pointcloud_before_median_depth_filter.ply", raw_points, raw_colors)
    save_binary(shot_dir / "valid_depth_pixels.png", valid_depth)
    save_binary(shot_dir / "invalid_depth_pixels.png", invalid_depth)
    save_mask_overlay(shot_dir / "invalid_depth_overlay.png", rgb, valid_depth, invalid_depth)

    # Stage 3: exact refine_only median-range helper and tolerance.
    filtered_depth, depth_info = current.save_masked_and_cropped(
        rgb,
        depth,
        refined.astype(np.uint8),
        shot_dir,
        "debug_refine_only_median_depth_filter",
        z_tolerance_raw=30,
        return_info=True,
    )
    median_kept = refined & (filtered_depth > 0)
    median_removed = valid_depth & ~median_kept
    median_points, median_uv = current._mask_depth_to_points_uv_for_plane_filter(
        median_kept.astype(np.uint8), filtered_depth, intr, depth_scale
    )
    median_colors = colors_for_uv(rgb, median_uv)
    removed_median_points, removed_median_uv = current._mask_depth_to_points_uv_for_plane_filter(
        median_removed.astype(np.uint8), depth, intr, depth_scale
    )
    save_ply_ascii(shot_dir / "pointcloud_after_median_depth_filter.ply", median_points, median_colors)
    save_ply_ascii(
        shot_dir / "removed_by_median_depth_filter.ply",
        removed_median_points,
        colors_for_uv(rgb, removed_median_uv),
    )
    save_binary(shot_dir / "median_depth_filter_kept_pixels.png", median_kept)
    save_binary(shot_dir / "median_depth_filter_removed_pixels.png", median_removed)
    save_mask_overlay(
        shot_dir / "median_depth_filter_overlay.png", rgb, median_kept, median_removed
    )
    median_value = depth_info.get("median_depth_raw")
    lower = None if median_value is None else float(median_value) - 30
    upper = None if median_value is None else float(median_value) + 30
    median_result = {
        **depth_info,
        "median_depth_raw": median_value,
        "median_depth_m": None if median_value is None else float(median_value) * depth_scale,
        "depth_lower_limit": lower,
        "depth_upper_limit": upper,
        "before_median_filter_count": int(raw_points.shape[0]),
        "after_median_filter_count": int(median_points.shape[0]),
        "removed_by_median_filter_count": int(removed_median_points.shape[0]),
        "removed_by_median_filter_ratio": float(
            removed_median_points.shape[0] / max(raw_points.shape[0], 1)
        ),
        "removed_region_distribution": region_distribution(median_removed, refined),
    }
    save_json(shot_dir / "median_depth_filter_result.json", median_result)

    # Stage 4: exact normal RANSAC used by refine_only.
    plane, inliers, ransac_info = current._fit_plane_ransac_open3d_for_spine(
        median_points,
        distance_threshold_m=0.008,
        ransac_n=3,
        num_iterations=1200,
    )
    if not ransac_info.get("used"):
        raise RuntimeError(f"normal RANSAC failed: {ransac_info}")
    ransac_points = median_points[inliers]
    ransac_uv = median_uv[inliers]
    ransac_colors = median_colors[inliers]
    ransac_removed_points = median_points[~inliers]
    ransac_removed_uv = median_uv[~inliers]
    ransac_removed_colors = median_colors[~inliers]
    save_ply_ascii(shot_dir / "pointcloud_before_normal_ransac.ply", median_points, median_colors)
    save_ply_ascii(shot_dir / "pointcloud_after_normal_ransac.ply", ransac_points, ransac_colors)
    save_ply_ascii(shot_dir / "removed_by_normal_ransac.ply", ransac_removed_points, ransac_removed_colors)
    ransac_kept = uv_mask(refined.shape, ransac_uv)
    ransac_removed = uv_mask(refined.shape, ransac_removed_uv)
    save_binary(shot_dir / "normal_ransac_kept_pixels.png", ransac_kept)
    save_binary(shot_dir / "normal_ransac_removed_pixels.png", ransac_removed)
    save_mask_overlay(
        shot_dir / "normal_ransac_overlay.png", rgb, ransac_kept, ransac_removed
    )
    residual = current._point_plane_distance_for_spine(median_points, plane)
    normal_result = {
        **ransac_info,
        "before_ransac_count": int(median_points.shape[0]),
        "after_ransac_count": int(ransac_points.shape[0]),
        "removed_by_ransac_count": int(ransac_removed_points.shape[0]),
        "removed_by_ransac_ratio": float(
            ransac_removed_points.shape[0] / max(median_points.shape[0], 1)
        ),
        "plane_coefficients": np.asarray(plane, float).tolist(),
        "residual_median_m": float(np.median(residual)),
        "residual_p90_m": float(np.percentile(residual, 90)),
        "removed_region_distribution": region_distribution(ransac_removed, refined),
    }
    save_json(shot_dir / "normal_ransac_result.json", normal_result)

    # Stage 5: the exact RANSAC array is the PCA input; no hidden processing.
    points_for_pca = ransac_points
    uv_for_pca = ransac_uv
    save_ply_ascii(shot_dir / "pointcloud_sent_to_pca.ply", points_for_pca, ransac_colors)
    save_binary(shot_dir / "pca_input_pixels.png", uv_mask(refined.shape, uv_for_pca))
    ransac_to_pca_equal = bool(np.array_equal(ransac_points, points_for_pca))
    if not ransac_to_pca_equal:
        raise RuntimeError("unexpected removal between normal RANSAC and PCA")
    mean, pc1, pc2 = pca_axes_fix_dir(points_for_pca)
    norm_xy = float(np.hypot(pc1[0], pc1[1]))
    roll = 0.0 if norm_xy < 1e-8 else float(np.arctan2(pc1[1], pc1[0]))
    width_info = estimate_book_width(points_for_pca, mean, pc1, pc2)
    width_m = width_info.get("av_book_width_m")
    target_info = find_target_point(points_for_pca)
    target = target_info.get("target_m")
    if width_m is None or target is None:
        raise RuntimeError(f"PCA result failed: width={width_info}, target={target_info}")
    pred_width = float(width_m * 1000)
    stable._save_final_png_with_target(
        shot_dir / "final.png",
        rgb,
        refined,
        np.asarray(target),
        points_for_pca,
        uv_for_pca,
    )
    pca_result = {
        "theta_rad": roll,
        "theta_deg": float(np.degrees(roll)),
        "p_min_m": np.asarray(target, float).tolist(),
        "book_width_mm": pred_width,
        "book_width_info": width_info,
        "pca_input_count": int(points_for_pca.shape[0]),
        "normal_ransac_output_equals_pca_input": ransac_to_pca_equal,
    }
    save_json(shot_dir / "pca_result_offline.json", pca_result)

    panel = stage_panel(
        rgb,
        [
            (raw, f"0 raw {raw.sum()}", (0, 255, 0)),
            (refined, f"1 refined {refined.sum()} (-{removed_refinement.sum()})", (0, 255, 0)),
            (valid_depth, f"2 valid depth {valid_depth.sum()} (-{invalid_depth.sum()})", (0, 255, 0)),
            (median_kept, f"3 median {median_kept.sum()} (-{median_removed.sum()})", (0, 255, 0)),
            (ransac_kept, f"4 RANSAC {ransac_kept.sum()} (-{ransac_removed.sum()})", (0, 255, 0)),
            (uv_mask(refined.shape, uv_for_pca), f"5 PCA {len(points_for_pca)}", (0, 255, 0)),
            (refined, f"6 final width {pred_width:.2f} mm", (0, 255, 0)),
        ],
    )
    cv2.imwrite(str(shot_dir / "pointcloud_filter_stages.png"), panel)

    raw_components = cv2.connectedComponents(raw.astype(np.uint8), connectivity=8)[0] - 1
    result = {
        "case_index": case_index,
        "book_name": query,
        "gt_book_width_mm": gt_width_mm,
        "success": True,
        "failure_stage": None,
        "selected_mask_index": selected_index,
        "mask": {
            "raw_area_px": int(raw.sum()),
            "refined_area_px": int(refined.sum()),
            "refinement_removed_px": int(removed_refinement.sum()),
            "refinement_no_op": bool(np.array_equal(raw, refined)),
            "raw_refined_sha256_equal": sha256(shot_dir / "selected_mask_raw.png")
            == sha256(shot_dir / "selected_mask_refined.png"),
            "fallback_to_raw_mask": refinement["fallback_to_raw_mask"],
            "raw_component_count": int(raw_components),
            "anchor_component_id": refinement["anchor_component_id"],
            "kept_component_ids": refinement["kept_component_ids"],
            "removed_component_ids": refinement["removed_component_ids"],
        },
        "depth": {
            "final_mask_area_px": int(refined.sum()),
            "valid_depth_count": int(valid_depth.sum()),
            "invalid_depth_count": int(invalid_depth.sum()),
            "valid_depth_ratio": float(valid_depth.sum() / max(refined.sum(), 1)),
            "median_depth": median_value,
            "lower_limit": lower,
            "upper_limit": upper,
        },
        "point_counts": {
            "raw_valid_depth": int(raw_points.shape[0]),
            "before_median_filter": int(raw_points.shape[0]),
            "after_median_filter": int(median_points.shape[0]),
            "removed_by_median_filter": int(removed_median_points.shape[0]),
            "before_ransac": int(median_points.shape[0]),
            "after_ransac": int(ransac_points.shape[0]),
            "removed_by_ransac": int(ransac_removed_points.shape[0]),
            "sent_to_pca": int(points_for_pca.shape[0]),
            "removed_between_ransac_and_pca": 0,
        },
        "removal_ratios": {
            "refinement": float(removed_refinement.sum() / max(raw.sum(), 1)),
            "invalid_depth": float(invalid_depth.sum() / max(refined.sum(), 1)),
            "median_filter": median_result["removed_by_median_filter_ratio"],
            "ransac": normal_result["removed_by_ransac_ratio"],
            "ransac_to_pca": 0.0,
            "total_mask_to_pca": float(
                (refined.sum() - len(points_for_pca)) / max(refined.sum(), 1)
            ),
        },
        "removed_region_distribution": {
            "median_filter": median_result["removed_region_distribution"],
            "ransac": normal_result["removed_region_distribution"],
        },
        "result": {
            "roll_rad": roll,
            "roll_deg": float(np.degrees(roll)),
            "pred_width_mm": pred_width,
            "abs_error_mm": abs(pred_width - gt_width_mm),
            "target_point_m": np.asarray(target, float).tolist(),
        },
        "outputs": {},
        "error": None,
    }
    result["outputs"] = {
        path.name: str(path.resolve())
        for path in shot_dir.iterdir()
        if path.is_file()
    }
    save_json(shot_dir / "pointcloud_debug_result.json", result)
    return result


CSV_FIELDS = [
    "case_index", "book_name", "gt_width_mm", "success", "failure_stage",
    "raw_mask_area_px", "refined_mask_area_px", "refinement_removed_px",
    "valid_depth_count", "invalid_depth_count", "before_median_filter",
    "after_median_filter", "removed_by_median_filter",
    "median_filter_removal_ratio", "before_ransac", "after_ransac",
    "removed_by_ransac", "ransac_removal_ratio", "pca_input_count",
    "removed_between_ransac_and_pca", "total_mask_to_pca_removal_ratio",
    "roll_rad", "pred_width_mm", "abs_error_mm", "target_x_m", "target_y_m",
    "target_z_m", "output_dir",
]


def csv_row(result, output_dir):
    point = ((result.get("result") or {}).get("target_point_m") or [None] * 3)
    return {
        "case_index": result.get("case_index"),
        "book_name": result.get("book_name"),
        "gt_width_mm": result.get("gt_book_width_mm"),
        "success": result.get("success"),
        "failure_stage": result.get("failure_stage"),
        "raw_mask_area_px": (result.get("mask") or {}).get("raw_area_px"),
        "refined_mask_area_px": (result.get("mask") or {}).get("refined_area_px"),
        "refinement_removed_px": (result.get("mask") or {}).get("refinement_removed_px"),
        "valid_depth_count": (result.get("depth") or {}).get("valid_depth_count"),
        "invalid_depth_count": (result.get("depth") or {}).get("invalid_depth_count"),
        "before_median_filter": (result.get("point_counts") or {}).get("before_median_filter"),
        "after_median_filter": (result.get("point_counts") or {}).get("after_median_filter"),
        "removed_by_median_filter": (result.get("point_counts") or {}).get("removed_by_median_filter"),
        "median_filter_removal_ratio": (result.get("removal_ratios") or {}).get("median_filter"),
        "before_ransac": (result.get("point_counts") or {}).get("before_ransac"),
        "after_ransac": (result.get("point_counts") or {}).get("after_ransac"),
        "removed_by_ransac": (result.get("point_counts") or {}).get("removed_by_ransac"),
        "ransac_removal_ratio": (result.get("removal_ratios") or {}).get("ransac"),
        "pca_input_count": (result.get("point_counts") or {}).get("sent_to_pca"),
        "removed_between_ransac_and_pca": (result.get("point_counts") or {}).get("removed_between_ransac_and_pca"),
        "total_mask_to_pca_removal_ratio": (result.get("removal_ratios") or {}).get("total_mask_to_pca"),
        "roll_rad": (result.get("result") or {}).get("roll_rad"),
        "pred_width_mm": (result.get("result") or {}).get("pred_width_mm"),
        "abs_error_mm": (result.get("result") or {}).get("abs_error_mm"),
        "target_x_m": point[0], "target_y_m": point[1], "target_z_m": point[2],
        "output_dir": str(output_dir),
    }


def write_csv(path, results, run_root):
    with path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(csv_row(result, run_root / str(result["case_index"])))


def stats(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return {
        "count": len(values),
        "mean": None if not values else float(np.mean(values)),
        "median": None if not values else float(np.median(values)),
        "max": None if not values else float(np.max(values)),
    }


def correlation(left, right):
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right)
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)
    ]
    if len(pairs) < 3:
        return None
    a, b = np.asarray(pairs).T
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def make_summary(results):
    success = [r for r in results if r.get("success")]
    errors = [(r.get("result") or {}).get("abs_error_mm") for r in success]
    median_rates = [(r.get("removal_ratios") or {}).get("median_filter") for r in success]
    ransac_rates = [(r.get("removal_ratios") or {}).get("ransac") for r in success]
    total_rates = [(r.get("removal_ratios") or {}).get("total_mask_to_pca") for r in success]
    invalid_rates = [(r.get("removal_ratios") or {}).get("invalid_depth") for r in success]
    valid_rates = [
        None if value is None else 1.0 - value for value in invalid_rates
    ]
    total_removed_counts = [
        (r.get("mask") or {}).get("refined_area_px", 0)
        - (r.get("point_counts") or {}).get("sent_to_pca", 0)
        for r in success
    ]
    def top(key, count=10):
        return [
            {
                "case_index": r["case_index"],
                "value": key(r),
                "output_dir": str(Path(r["_output_dir"]).resolve()),
            }
            for r in sorted(success, key=lambda x: key(x) or -1, reverse=True)[:count]
        ]
    def case_refs(candidates, count=3):
        return [
            {
                "case_index": r["case_index"],
                "output_dir": str(Path(r["_output_dir"]).resolve()),
            }
            for r in candidates[:count]
        ]
    clean = [r for r in success if (r.get("mask") or {}).get("refinement_no_op")]
    clean_by_removal = sorted(
        clean,
        key=lambda r: (r.get("removal_ratios") or {}).get("total_mask_to_pca", 0),
        reverse=True,
    )
    clean_by_error = sorted(
        clean,
        key=lambda r: (r.get("result") or {}).get("abs_error_mm", 0),
        reverse=True,
    )
    def edge_removed(r, stage):
        values = (r.get("removed_region_distribution") or {}).get(stage, {})
        return sum(values.get(name, 0) for name in (
            "left_10pct", "right_10pct", "top_10pct", "bottom_10pct"
        ))
    median_edge = sorted(success, key=lambda r: edge_removed(r, "median_filter"), reverse=True)
    ransac_edge = sorted(success, key=lambda r: edge_removed(r, "ransac"), reverse=True)
    ransac_to_pca = [
        r for r in success
        if (r.get("point_counts") or {}).get("removed_between_ransac_and_pca", 0) > 0
    ]
    underestimated = sorted(
        [
            r for r in success
            if (r.get("result") or {}).get("pred_width_mm", 0) < r["gt_book_width_mm"]
        ],
        key=lambda r: r["gt_book_width_mm"] - (r.get("result") or {}).get("pred_width_mm", 0),
        reverse=True,
    )
    overestimated = sorted(
        [
            r for r in success
            if (r.get("result") or {}).get("pred_width_mm", 0) > r["gt_book_width_mm"]
        ],
        key=lambda r: (r.get("result") or {}).get("pred_width_mm", 0) - r["gt_book_width_mm"],
        reverse=True,
    )
    return {
        "total_cases": 100,
        "success_count": len(success),
        "failure_count": 100 - len(success),
        "within_error_mm": {
            str(limit): sum(e is not None and e <= limit for e in errors)
            for limit in ERROR_THRESHOLDS_MM
        },
        "absolute_error_mm": stats(errors),
        "mask_refinement": {
            "no_op_count": sum((r.get("mask") or {}).get("refinement_no_op", False) for r in success),
            "component_removal_case_count": sum(bool((r.get("mask") or {}).get("removed_component_ids")) for r in success),
            "fallback_count": sum((r.get("mask") or {}).get("fallback_to_raw_mask", False) for r in success),
            "total_removed_pixels": sum((r.get("mask") or {}).get("refinement_removed_px", 0) for r in success),
            "single_component_changed_count": sum(
                (r.get("mask") or {}).get("raw_component_count") == 1
                and not (r.get("mask") or {}).get("refinement_no_op")
                for r in success
            ),
        },
        "invalid_depth_ratio": stats(invalid_rates),
        "valid_depth_ratio": stats(valid_rates),
        "median_filter": {
            "removed_count": stats([(r.get("point_counts") or {}).get("removed_by_median_filter") for r in success]),
            "removal_ratio": stats(median_rates),
            "zero_removal_cases": sum((r.get("point_counts") or {}).get("removed_by_median_filter") == 0 for r in success),
            "ratio_ge_10pct": sum(v is not None and v >= .10 for v in median_rates),
            "ratio_ge_20pct": sum(v is not None and v >= .20 for v in median_rates),
        },
        "ransac": {
            "removed_count": stats([(r.get("point_counts") or {}).get("removed_by_ransac") for r in success]),
            "removal_ratio": stats(ransac_rates),
            "zero_removal_cases": sum((r.get("point_counts") or {}).get("removed_by_ransac") == 0 for r in success),
            "ratio_ge_10pct": sum(v is not None and v >= .10 for v in ransac_rates),
            "ratio_ge_20pct": sum(v is not None and v >= .20 for v in ransac_rates),
        },
        "total_mask_to_pca": {
            "removed_count": stats(total_removed_counts),
            "removal_ratio": stats(total_rates),
            "top_10": top(lambda r: (r.get("removal_ratios") or {}).get("total_mask_to_pca")),
        },
        "width_error_top_10": top(lambda r: (r.get("result") or {}).get("abs_error_mm")),
        "correlations_reference_only": {
            "total_removal_vs_width_error": correlation(total_rates, errors),
            "median_removal_vs_width_error": correlation(median_rates, errors),
            "ransac_removal_vs_width_error": correlation(ransac_rates, errors),
            "note": "Correlation is descriptive and does not establish causation.",
        },
        "focus_cases": {
            "clean_mask_but_large_point_removal": case_refs(clean_by_removal),
            "refinement_no_op_but_large_width_error": case_refs(clean_by_error),
            "median_filter_removed_book_edges": case_refs(median_edge),
            "ransac_removed_book_edges": case_refs(ransac_edge),
            "removed_between_ransac_and_pca": case_refs(ransac_to_pca),
            "high_total_removal": case_refs(
                sorted(
                    success,
                    key=lambda r: (r.get("removal_ratios") or {}).get(
                        "total_mask_to_pca", 0
                    ),
                    reverse=True,
                )
            ),
            "width_underestimated": case_refs(underestimated),
            "width_overestimated": case_refs(overestimated),
        },
        "results": [{k: v for k, v in r.items() if k != "_output_dir"} for r in results],
    }


def write_report(path, summary, run_root):
    def representatives(stage, region, n=3):
        candidates = []
        for r in summary["results"]:
            if not r.get("success"):
                continue
            value = (
                (r.get("removed_region_distribution") or {})
                .get(stage, {})
                .get(region, 0)
            )
            candidates.append((value, r["case_index"]))
        return [
            f"`{idx}` ({value} px; `{run_root / str(idx)}`)"
            for value, idx in sorted(candidates, reverse=True)[:n]
        ]
    lines = [
        "# SAM3 refine_only point-cloud removal report",
        "",
        f"Output root: `{run_root}`",
        "",
        "## Recognition",
        "",
        f"- Success: {summary['success_count']}/100",
        f"- Failure: {summary['failure_count']}/100",
        f"- Within 1.0/1.5/2.0 mm: {summary['within_error_mm']}",
        f"- Absolute error [mm]: {summary['absolute_error_mm']}",
        "",
        "## Mask refinement",
        "",
        f"`{summary['mask_refinement']}`",
        "",
        "## Invalid Depth",
        "",
        f"Invalid ratio: `{summary['invalid_depth_ratio']}`",
        f"Valid ratio: `{summary['valid_depth_ratio']}`",
        "",
        "## Median-range filter",
        "",
        f"`{summary['median_filter']}`",
        f"- Left-edge representatives: {', '.join(representatives('median_filter', 'left_10pct'))}",
        f"- Right-edge representatives: {', '.join(representatives('median_filter', 'right_10pct'))}",
        f"- Top-edge representatives: {', '.join(representatives('median_filter', 'top_10pct'))}",
        f"- Bottom-edge representatives: {', '.join(representatives('median_filter', 'bottom_10pct'))}",
        f"- Center representatives: {', '.join(representatives('median_filter', 'center'))}",
        "",
        "## Normal RANSAC",
        "",
        f"`{summary['ransac']}`",
        f"- Left-edge representatives: {', '.join(representatives('ransac', 'left_10pct'))}",
        f"- Right-edge representatives: {', '.join(representatives('ransac', 'right_10pct'))}",
        f"- Top-edge representatives: {', '.join(representatives('ransac', 'top_10pct'))}",
        f"- Bottom-edge representatives: {', '.join(representatives('ransac', 'bottom_10pct'))}",
        f"- Center representatives: {', '.join(representatives('ransac', 'center'))}",
        "",
        "## PCA and combined removal",
        "",
        "- RANSAC output is assigned directly to PCA input; additional removal is zero by construction and checked per case.",
        f"- Combined removal: `{summary['total_mask_to_pca']['removal_ratio']}`",
        f"- Top 10 combined removal: `{summary['total_mask_to_pca']['top_10']}`",
        f"- Top 10 width error: `{summary['width_error_top_10']}`",
        f"- Reference correlations: `{summary['correlations_reference_only']}`",
        "",
        "## Focus cases",
        "",
        *[
            f"- {name}: "
            + (
                ", ".join(
                    f"`{item['case_index']}` (`{item['output_dir']}`)"
                    for item in cases
                )
                if cases else "none"
            )
            for name, cases in summary["focus_cases"].items()
        ],
        "",
        "## Interpretation",
        "",
        "The stage with the larger aggregate and edge-localized removal is the primary over-removal candidate. "
        "This is a diagnostic association, not proof of causation. Inspect each representative directory's overlays and `pointcloud_filter_stages.png`.",
        "",
        "## Safety",
        "",
        "No robot, ROS node/topic, Height Controller, manipulator, hand, or RealSense live capture was used. "
        "Inputs were copied; `/home/book/pro_book` and protected source/model/environment files were not modified.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    run_root = make_run_root()
    run_log = run_root / "run_console.log"
    master = load_master()
    results = []
    owned_service = None
    service_info = None
    with run_log.open("w", encoding="utf-8") as root_log:
        tee_out, tee_err = Tee(sys.__stdout__, root_log), Tee(sys.__stderr__, root_log)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"RUN_ROOT={run_root}")
            try:
                owned_service, service_info = ensure_service(run_root)
                public_info = {k: v for k, v in service_info.items() if not k.startswith("_")}
                save_json(run_root / "sam3_service_session.json", public_info)
                print(f"SAM3 service ready: {public_info}")
            except Exception as exc:
                save_json(
                    run_root / "fatal_error.json",
                    {"stage": "sam3_service", "error": f"{type(exc).__name__}: {exc}"},
                )
                traceback.print_exc()
                print("Fatal shared-service error; no case inference was attempted.")
                return 2
            try:
                for case_index in range(START_INDEX, END_INDEX + 1):
                    source = TEST_BASE_DIR / str(case_index)
                    output = run_root / str(case_index)
                    output.mkdir()
                    master_index, repeat_index, book = book_for_case(master, case_index)
                    result = {
                        "case_index": case_index,
                        "book_name": book["book_name"],
                        "gt_book_width_mm": book["book_width_mm"],
                        "success": False,
                        "failure_stage": "input",
                        "error": None,
                        "_output_dir": str(output),
                    }
                    with (output / "offline_run_console.log").open("w", encoding="utf-8") as case_log:
                        case_out, case_err = Tee(sys.__stdout__, root_log, case_log), Tee(sys.__stderr__, root_log, case_log)
                        with redirect_stdout(case_out), redirect_stderr(case_err):
                            print(f"CASE {case_index}/100 master={master_index} repeat={repeat_index} book={book['book_name']}")
                            try:
                                copy_inputs(source, output)
                                result["failure_stage"] = "recognition"
                                result = run_debug_case(
                                    book["book_name"], output, book["book_width_mm"], case_index
                                )
                                result["_output_dir"] = str(output)
                                print(f"CASE_SUCCESS width={result['result']['pred_width_mm']:.3f}")
                            except Exception as exc:
                                result["error"] = f"{type(exc).__name__}: {exc}"
                                traceback.print_exc()
                                save_json(output / "pointcloud_debug_result.json", {k: v for k, v in result.items() if k != "_output_dir"})
                                print(f"CASE_FAILED {result['error']}")
                    results.append(result)
                    write_csv(run_root / "summary.csv", results, run_root)
                    save_json(run_root / "summary.json", make_summary(results + [
                        {
                            "case_index": i,
                            "success": False,
                            "failure_stage": "not_run_yet",
                            "error": None,
                            "_output_dir": str(run_root / str(i)),
                        }
                        for i in range(case_index + 1, 101)
                    ]))
            finally:
                stop_owned_service(owned_service, service_info or {})
    summary = make_summary(results)
    write_csv(run_root / "summary.csv", results, run_root)
    save_json(run_root / "summary.json", summary)
    write_report(run_root / "POINTCLOUD_REMOVAL_REPORT.md", summary, run_root)
    print(f"RUN_ROOT={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
