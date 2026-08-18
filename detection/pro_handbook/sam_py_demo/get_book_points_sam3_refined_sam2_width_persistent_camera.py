#!/usr/bin/env python3
"""SAM2-compatible-width recognition using an externally owned camera."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import get_book_points_no_mask_merge_no_side_filter as stable
from .get_book_points_sam3_refined_sam2_width import (
    CAPTURES_DIR,
    run_capture_and_pca_offline_sam3_refined_sam2_width,
)
from .modules.sam2_compatible_geometry import DEFAULT_MASK_PCA_WIDTH_MODE

if TYPE_CHECKING:
    from .modules.realsense_persistent_session import (
        RealSensePersistentSession,
    )


def _result_tuple(result, shot_dir: Path):
    return (
        float(result["roll_rad"]),
        np.asarray(result["point_3d"], dtype=float),
        float(result["pred_book_width_mm"]),
        shot_dir,
    )


def run_capture_and_pca_sam3_refined_sam2_width_persistent_camera(
    query,
    sam_device="gpu",
    *,
    camera_session: "RealSensePersistentSession | None" = None,
    shot_dir=None,
    mask_width_mode=DEFAULT_MASK_PCA_WIDTH_MODE,
):
    """Capture through a persistent session, or process an offline directory.

    Supplying ``shot_dir`` selects the offline path and never accesses the
    RealSense session. Live calls leave ``shot_dir`` unset and must provide an
    already-started ``camera_session``.
    """
    if shot_dir is not None:
        offline_dir = Path(shot_dir).expanduser().resolve()
        result = run_capture_and_pca_offline_sam3_refined_sam2_width(
            query,
            offline_dir,
            sam_device=sam_device,
            mask_width_mode=mask_width_mode,
        )
        return _result_tuple(result, offline_dir)

    if camera_session is None:
        raise RuntimeError(
            "live persistent-camera recognition requires camera_session"
        )
    if not camera_session.is_started:
        raise RuntimeError(
            "live persistent-camera recognition requires a started session"
        )

    stem = datetime.now().astimezone().strftime(
        "%Y%m%d_%H%M%S_live_sam3_refined_sam2_width"
    )
    live_dir = (CAPTURES_DIR / stem).expanduser().resolve()
    live_dir.mkdir(parents=True, exist_ok=False)

    capture_start = time.perf_counter()
    _, _, intr, depth_scale, _ = camera_session.capture(
        live_dir,
        depth_filter=stable.current.depth_filter_like_viewer,
    )
    capture_seconds = time.perf_counter() - capture_start

    result = run_capture_and_pca_offline_sam3_refined_sam2_width(
        query,
        live_dir,
        sam_device=sam_device,
        intr=intr,
        depth_scale=depth_scale,
        mask_width_mode=mask_width_mode,
    )
    result["capture_seconds"] = capture_seconds
    return _result_tuple(result, live_dir)
