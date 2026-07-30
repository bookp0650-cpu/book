"""SAM3 conservative-refinement variant with legacy SAM2 terminal geometry."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .get_book_points_sam3_refined_median_depth import (
    run_capture_and_pca_offline_sam3_refined_median_depth,
)
from .modules.sam2_compatible_geometry import (
    GEOMETRY_MODES,
    estimate_sam2_compatible_geometry,
)


def _read_ply_xyz(path):
    lines = Path(path).read_text(encoding="ascii").splitlines()
    end = lines.index("end_header")
    return np.loadtxt(lines[end + 1:], dtype=np.float64)[:, :3]


def run_capture_and_pca_offline_sam3_refined_sam2_geometry(
    query,
    shot_dir,
    sam_device="gpu",
    *,
    geometry_mode="sam2_all_geometry",
):
    if geometry_mode not in GEOMETRY_MODES:
        raise ValueError(f"unsupported mode: {geometry_mode}")
    shot_dir = Path(shot_dir).resolve()
    current = run_capture_and_pca_offline_sam3_refined_median_depth(
        query, shot_dir, sam_device=sam_device, mode="refine_only"
    )
    rgb = cv2.imread(str(shot_dir / "after_init_rgb.png"))
    mask = cv2.imread(str(shot_dir / "selected_mask_refined.png"), 0) > 0
    depth = np.load(
        shot_dir / "refine_only_median_depth_filter_depth_masked.npy"
    )
    points = _read_ply_xyz(shot_dir / "pointcloud_sent_to_pca.ply")
    camera = json.loads((shot_dir / "camera_params.json").read_text())
    refinement = json.loads(
        (shot_dir / "mask_refinement_result.json").read_text()
    )
    geometry = estimate_sam2_compatible_geometry(
        mask, rgb, depth, points, camera,
        selected_axis=refinement.get("axis_uv"),
        current_geometry={
            "roll_rad": current["roll_rad"],
            "width_mm": current["pred_book_width_mm"],
            "target_point_m": current["point_3d"],
            "target_pixel": current.get("target_uv"),
        },
        geometry_mode=geometry_mode,
        debug_dir=shot_dir / f"sam2_geometry_{geometry_mode}",
    )
    result = {
        **current,
        "geometry_mode": geometry_mode,
        "roll_rad": geometry["roll"]["value_rad"],
        "pred_book_width_mm": geometry["width"]["width_mm"],
        "point_3d": geometry["target"]["target_point_m"],
        "sam2_compatible_geometry": geometry,
    }
    (shot_dir / "sam2_compatible_geometry_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def run_capture_and_pca_sam3_refined_sam2_geometry(*args, **kwargs):
    raise RuntimeError(
        "Live RealSense entry is intentionally disabled in this comparison variant; "
        "use the offline entry or integrate only after offline acceptance."
    )
