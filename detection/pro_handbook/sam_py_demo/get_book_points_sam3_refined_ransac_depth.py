#!/usr/bin/env python3
"""Comparison variant: conservative mask refinement + RANSAC Depth completion."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import get_book_points as current
from . import get_book_points_no_mask_merge_no_side_filter as stable
from . import get_book_points_sam3_refined_median_depth as median_variant
from .modules.book_width import estimate_book_width
from .modules.grip_point import find_target_point
from .modules.pca_vector import pca_axes_fix_dir
from .modules.pointcloud_utils import save_ply_ascii
from .sam3_mask_refinement import refine_selected_sam3_mask
from .sam3_ransac_depth_completion import complete_depth_with_ransac_plane


VARIANT = "sam3_refined_ransac_depth"
MODES = {
    "baseline",
    "refine_only",
    "ransac_complete_only",
    "refine_and_ransac_complete",
}
DEFAULT_MODE = "refine_and_ransac_complete"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CAPTURES_DIR = PROJECT_ROOT / "captures"


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _ransac_complete_and_pca(
    *,
    shot_dir,
    color_np,
    depth_raw,
    raw_mask,
    used_mask,
    anchor_mask,
    intr,
    depth_scale,
):
    corrected, classes, plane_result, completion = (
        complete_depth_with_ransac_plane(
            final_mask=used_mask,
            anchor_mask=anchor_mask,
            depth_raw=depth_raw,
            intr=intr,
            depth_scale=depth_scale,
            rgb_bgr=color_np,
            output_dir=shot_dir,
        )
    )
    completed_mask = used_mask & np.isfinite(corrected) & (corrected > 0)
    points, uv = current._mask_depth_to_points_uv_for_plane_filter(
        completed_mask.astype(np.uint8), corrected, intr, depth_scale
    )
    colors = color_np[uv[:, 1], uv[:, 0], ::-1].astype(np.uint8)
    save_ply_ascii(
        shot_dir / "pointcloud_from_ransac_completed_depth.ply", points, colors
    )
    # No RANSAC, statistical, clustering, or side-surface deletion is allowed
    # between these assignments.
    points_for_pca = points
    uv_for_pca = uv
    colors_for_pca = colors
    save_ply_ascii(
        shot_dir / "pointcloud_sent_to_pca.ply", points_for_pca, colors_for_pca
    )
    if points_for_pca.shape[0] < 3:
        raise RuntimeError(
            f"insufficient PCA points after safe RANSAC fallback: {points_for_pca.shape[0]}"
        )
    mean, pc1, pc2 = pca_axes_fix_dir(points_for_pca)
    norm_xy = float(np.hypot(float(pc1[0]), float(pc1[1])))
    roll = 0.0 if norm_xy < 1e-8 else float(np.arctan2(pc1[1], pc1[0]))
    width_info = estimate_book_width(points_for_pca, mean, pc1, pc2)
    width_m = width_info.get("av_book_width_m")
    if width_m is None:
        raise RuntimeError(f"book width estimation failed: {width_info}")
    target_info = find_target_point(points_for_pca)
    target = target_info.get("target_m")
    if target is None:
        raise RuntimeError(f"target point estimation failed: {target_info}")
    target_uv = stable._save_final_png_with_target(
        shot_dir / "final.png",
        color_np,
        used_mask,
        np.asarray(target),
        points_for_pca,
        uv_for_pca,
    )
    completion.update(
        {
            "pointcloud_point_count": int(points.shape[0]),
            "pca_input_point_count": int(points_for_pca.shape[0]),
            "pointcloud_equals_pca_input": bool(
                np.array_equal(points, points_for_pca)
            ),
            "pointcloud_ply_equals_pca_ply": bool(
                stable._sha256(
                    shot_dir / "pointcloud_from_ransac_completed_depth.ply"
                )
                == stable._sha256(shot_dir / "pointcloud_sent_to_pca.ply")
            ),
        }
    )
    _write_json(shot_dir / "depth_completion_result.json", completion)
    return {
        "roll_rad": roll,
        "point_3d": np.asarray(target, float).tolist(),
        "pred_book_width_mm": float(width_m * 1000.0),
        "width_info": width_info,
        "target_uv": list(target_uv),
        "point_counts": {
            "pointcloud_from_ransac_completed_depth": int(points.shape[0]),
            "sent_to_pca": int(points_for_pca.shape[0]),
        },
        "ransac_plane": plane_result,
        "depth_completion": completion,
    }


def run_capture_and_pca_offline_sam3_refined_ransac_depth(
    query,
    shot_dir,
    sam_device="gpu",
    *,
    mode=DEFAULT_MODE,
    intr=None,
    depth_scale=None,
    depth_merge_tolerance_raw=30,
):
    """Run fresh OCR/SAM3 selection then the requested comparison Depth path."""
    if mode not in MODES:
        raise ValueError(f"unsupported mode {mode!r}; expected {sorted(MODES)}")
    if mode == "baseline":
        return stable.run_capture_and_pca_offline_no_mask_merge_no_side_filter(
            query,
            shot_dir,
            sam_device=sam_device,
            intr=intr,
            depth_scale=depth_scale,
            depth_merge_tolerance_raw=depth_merge_tolerance_raw,
        )
    if mode == "refine_only":
        return median_variant.run_capture_and_pca_offline_sam3_refined_median_depth(
            query,
            shot_dir,
            sam_device=sam_device,
            mode="refine_only",
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
    polygon, ocr_info = median_variant._selected_ocr_geometry(
        raw_mask, merged, color_np.shape[:2], shot_dir, query
    )
    refined, refinement, _ = refine_selected_sam3_mask(
        raw_mask,
        ocr_polygon=polygon,
        depth_raw=depth_raw,
        rgb_bgr=color_np,
        output_dir=shot_dir,
        mode=mode,
    )
    use_refinement = mode == "refine_and_ransac_complete"
    used_mask = refined if use_refinement else raw_mask.copy()
    if not use_refinement:
        refinement.update(
            {
                "mode": mode,
                "refinement_enabled": False,
                "kept_component_ids": list(
                    range(1, refinement["raw_component_count"] + 1)
                ),
                "removed_component_ids": [],
                "refined_mask_area_px": int(raw_mask.sum()),
                "removed_area_px": 0,
                "no_op": True,
                "raw_and_refined_mask_equal": True,
                "fallback_to_raw_mask": False,
                "fallback_reason": None,
            }
        )
        cv2.imwrite(
            str(shot_dir / "selected_mask_refined.png"),
            raw_mask.astype(np.uint8) * 255,
        )
        green = color_np.copy()
        green[raw_mask] = (0, 255, 0)
        cv2.imwrite(
            str(shot_dir / "selected_mask_refined_overlay.png"),
            cv2.addWeighted(color_np, 0.65, green, 0.35, 0),
        )
        _write_json(shot_dir / "mask_refinement_result.json", refinement)

    _, labels = cv2.connectedComponents(raw_mask.astype(np.uint8), connectivity=8)
    anchor = labels == int(refinement["anchor_component_id"])
    compute = _ransac_complete_and_pca(
        shot_dir=shot_dir,
        color_np=color_np,
        depth_raw=depth_raw,
        raw_mask=raw_mask,
        used_mask=used_mask,
        anchor_mask=anchor,
        intr=intr,
        depth_scale=float(depth_scale),
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
        **compute,
        "processing_seconds": float(time.perf_counter() - started),
        "returned_shot_dir": str(shot_dir),
    }
    _write_json(shot_dir / "offline_recognition_result.json", result)
    return result


def run_capture_and_pca_sam3_refined_ransac_depth(
    query,
    sam_device="gpu",
    *,
    mode=DEFAULT_MODE,
    shot_dir=None,
):
    """Live-compatible API. Offline evaluation never calls this function."""
    if shot_dir is None:
        stem = datetime.now().astimezone().strftime(
            "%Y%m%d_%H%M%S_live_sam3_refined_ransac_depth"
        )
        shot_dir = CAPTURES_DIR / stem
    shot_dir = Path(shot_dir).expanduser().resolve()
    shot_dir.mkdir(parents=True, exist_ok=False)
    _, _, intr, depth_scale, _ = stable.capture_rgbd_once_no_mask_merge_no_side_filter(
        shot_dir
    )
    result = run_capture_and_pca_offline_sam3_refined_ransac_depth(
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

