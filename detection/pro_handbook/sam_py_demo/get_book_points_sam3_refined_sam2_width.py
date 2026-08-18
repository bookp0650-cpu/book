#!/usr/bin/env python3
"""SAM3 refined recognition with only the terminal width changed to SAM2 style."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import get_book_points_no_mask_merge_no_side_filter as stable
from .get_book_points_sam3_refined_median_depth import (
    CAPTURES_DIR,
    run_capture_and_pca_offline_sam3_refined_median_depth,
)
from .modules.sam2_compatible_geometry import (
    DEFAULT_MASK_PCA_WIDTH_MODE,
    estimate_sam2_compatible_geometry,
)


VARIANT = "sam3_refined_sam2_width"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ply_xyz(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="ascii").splitlines()
    header_end = lines.index("end_header")
    points = np.loadtxt(lines[header_end + 1 :], dtype=np.float64)
    return np.atleast_2d(points)[:, :3]


def run_capture_and_pca_offline_sam3_refined_sam2_width(
    query,
    shot_dir,
    sam_device="gpu",
    *,
    intr=None,
    depth_scale=None,
    depth_merge_tolerance_raw=30,
    mask_width_mode=DEFAULT_MASK_PCA_WIDTH_MODE,
):
    """Run current refine-only recognition, replacing only its returned width."""
    shot_dir = Path(shot_dir).expanduser().resolve()
    base = run_capture_and_pca_offline_sam3_refined_median_depth(
        query,
        shot_dir,
        sam_device=sam_device,
        mode="refine_only",
        intr=intr,
        depth_scale=depth_scale,
        depth_merge_tolerance_raw=depth_merge_tolerance_raw,
    )

    mask_path = shot_dir / "selected_mask_refined.png"
    pca_path = shot_dir / "pointcloud_sent_to_pca.ply"
    mask_sha = _sha256(mask_path)
    pca_sha = _sha256(pca_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    rgb = cv2.imread(str(shot_dir / "after_init_rgb.png"), cv2.IMREAD_COLOR)
    if mask is None or rgb is None:
        raise FileNotFoundError("refined mask or RGB output is missing")
    depth = np.load(
        shot_dir / "refine_only_median_depth_filter_depth_masked.npy",
        allow_pickle=False,
    )
    points = _read_ply_xyz(pca_path)
    camera = _load_json(shot_dir / "camera_params.json")
    refinement = _load_json(shot_dir / "mask_refinement_result.json")
    # OCR geometry remains available for mask refinement and diagnostics, but
    # final image-width measurement must use all pixels of the refined mask.
    ocr_axis = refinement.get("axis_uv")
    current = {
        "roll_rad": float(base["roll_rad"]),
        "width_mm": float(base["pred_book_width_mm"]),
        "target_point_m": list(base["point_3d"]),
        "target_pixel": list(base["target_uv"]),
    }
    width_geometry = estimate_sam2_compatible_geometry(
        mask > 0,
        rgb,
        depth,
        points,
        camera,
        selected_axis=None,
        use_mask_pca_axis=True,
        diagnostic_ocr_axis=ocr_axis,
        mask_width_mode=mask_width_mode,
        current_geometry=current,
        geometry_mode="sam2_width_only",
        debug_dir=shot_dir / "sam2_width_debug",
    )
    width_mm = float(width_geometry["width"]["width_mm"])

    # Width-only contract: these values are copied directly from the current
    # refine-only path and are never replaced by compatibility candidates.
    result = dict(base)
    result.update(
        {
            "variant": VARIANT,
            "pred_book_width_mm": width_mm,
            "width_info": width_geometry["width"],
            "width_method": "sam2_compatible_width_only",
            "mask_width_mode": mask_width_mode,
            "current_sam3_width_mm": float(base["pred_book_width_mm"]),
            "roll_rad": float(base["roll_rad"]),
            "point_3d": list(base["point_3d"]),
            "target_uv": list(base["target_uv"]),
            "width_input_mask": "selected_mask_refined.png",
            "selected_width_axis_source": width_geometry[
                "selected_width_axis_source"
            ],
            "mask_pca_axis_uv": width_geometry["mask_pca_axis_uv"],
            "mask_pca_axis_angle_deg": width_geometry[
                "mask_pca_axis_angle_deg"
            ],
            "mask_pca_eigenvalues": width_geometry["mask_pca_eigenvalues"],
            "mask_pca_eigenvalue_ratio": width_geometry[
                "mask_pca_eigenvalue_ratio"
            ],
            "mask_pca_pixel_count": width_geometry["mask_pca_pixel_count"],
            "ocr_axis_uv": width_geometry["ocr_axis_uv"],
            "ocr_axis_angle_deg": width_geometry["ocr_axis_angle_deg"],
            "ocr_mask_axis_angle_diff_deg": width_geometry[
                "ocr_mask_axis_angle_diff_deg"
            ],
            "pixel_width": width_geometry["pixel_width"],
            "width_mm": width_geometry["width_mm"],
            "invariance": {
                "refined_mask_sha256_before_width": mask_sha,
                "refined_mask_sha256_after_width": _sha256(mask_path),
                "pca_pointcloud_sha256_before_width": pca_sha,
                "pca_pointcloud_sha256_after_width": _sha256(pca_path),
                "roll_difference_rad": 0.0,
                "target_point_difference_mm": 0.0,
            },
        }
    )
    if (
        result["invariance"]["refined_mask_sha256_before_width"]
        != result["invariance"]["refined_mask_sha256_after_width"]
        or result["invariance"]["pca_pointcloud_sha256_before_width"]
        != result["invariance"]["pca_pointcloud_sha256_after_width"]
    ):
        raise RuntimeError("width calculation modified a protected recognition artifact")
    _save_json(shot_dir / "sam3_refined_sam2_width_result.json", result)
    _save_json(shot_dir / "offline_recognition_result.json", result)
    return result


def run_capture_and_pca_sam3_refined_sam2_width(
    query,
    sam_device="gpu",
    *,
    shot_dir=None,
    mask_width_mode=DEFAULT_MASK_PCA_WIDTH_MODE,
):
    """Live-compatible API; capture once, then run the offline width-only path."""
    if shot_dir is None:
        stem = datetime.now().astimezone().strftime(
            "%Y%m%d_%H%M%S_live_sam3_refined_sam2_width"
        )
        shot_dir = CAPTURES_DIR / stem
    shot_dir = Path(shot_dir).expanduser().resolve()
    shot_dir.mkdir(parents=True, exist_ok=False)
    _, _, intr, depth_scale, _ = (
        stable.capture_rgbd_once_no_mask_merge_no_side_filter(shot_dir)
    )
    result = run_capture_and_pca_offline_sam3_refined_sam2_width(
        query,
        shot_dir,
        sam_device=sam_device,
        intr=intr,
        depth_scale=depth_scale,
        mask_width_mode=mask_width_mode,
    )
    return (
        float(result["roll_rad"]),
        np.asarray(result["point_3d"], dtype=float),
        float(result["pred_book_width_mm"]),
        shot_dir,
    )
