#!/usr/bin/env python3
"""Measure one saved-RGB-D SAM3/OCR run without importing robot code."""
from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
from pathlib import Path

from detection.pro_handbook.sam_py_demo import get_book_points as current
from detection.pro_handbook.sam_py_demo.get_book_points_no_mask_merge_no_side_filter import (
    MASTER_JSON,
    SOURCE_SHOT_DIR,
    run_capture_and_pca_offline_no_mask_merge_no_side_filter,
)


def main() -> int:
    stamps: dict[str, float] = {}
    original_start_ocr = current.start_ocr_subprocess
    original_wait_ocr = current.wait_ocr_subprocess
    original_infer = current._infer_masks_compat

    def measured_start_ocr(shot_dir):
        stamps["ocr_start"] = time.perf_counter()
        proc = original_start_ocr(shot_dir)

        def observe_exit():
            proc.wait()
            stamps["ocr_end"] = time.perf_counter()

        threading.Thread(target=observe_exit, daemon=True).start()
        return proc

    def measured_wait_ocr(proc, *, timeout=None):
        output = original_wait_ocr(proc, timeout=timeout)
        stamps["join_or_wait_end"] = time.perf_counter()
        return output

    def measured_infer(*args, **kwargs):
        stamps["sam3_start"] = time.perf_counter()
        try:
            return original_infer(*args, **kwargs)
        finally:
            stamps["sam3_end"] = time.perf_counter()

    current.start_ocr_subprocess = measured_start_ocr
    current.wait_ocr_subprocess = measured_wait_ocr
    current._infer_masks_compat = measured_infer

    with tempfile.TemporaryDirectory(prefix="sam3_ocr_parallelism_") as temp:
        shot_dir = Path(temp)
        for name in ("after_init_rgb.png", "after_init_depth.npy"):
            shutil.copy2(SOURCE_SHOT_DIR / name, shot_dir / name)
        query = json.loads(MASTER_JSON.read_text(encoding="utf-8"))[0]["book_name"]
        run_capture_and_pca_offline_no_mask_merge_no_side_filter(
            query=query,
            shot_dir=shot_dir,
            sam_device="gpu",
        )

    required = (
        "sam3_start",
        "sam3_end",
        "ocr_start",
        "ocr_end",
        "join_or_wait_end",
    )
    missing = [name for name in required if name not in stamps]
    if missing:
        raise RuntimeError(f"missing timestamps: {missing}")
    overlap = max(
        0.0,
        min(stamps["sam3_end"], stamps["ocr_end"])
        - max(stamps["sam3_start"], stamps["ocr_start"]),
    )
    payload = {
        **stamps,
        "overlap_seconds": overlap,
        "parallel": (
            stamps["sam3_start"] < stamps["ocr_end"]
            and stamps["ocr_start"] < stamps["sam3_end"]
        ),
    }
    print("PARALLELISM_JSON=" + json.dumps(payload, sort_keys=True))
    return 0 if payload["parallel"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
