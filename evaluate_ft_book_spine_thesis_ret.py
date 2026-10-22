#!/usr/bin/env python3
"""Evaluate one alternate book-spine checkpoint on the existing thesis/ret set.

The existing baseline output is deliberately outside this runner's output
directory and is never read as an inference input or overwritten.
"""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
EVALUATION_ROOT = PROJECT_ROOT / "captures" / "thesis" / "ret"
OUTPUT_ROOT = EVALUATION_ROOT / "retrieval_results_ft_expE_ck16"
CHECKPOINT = (
    PROJECT_ROOT
    / "detection"
    / "pro_handbook"
    / "sam3_runtime"
    / "models"
    / "candidates"
    / "sam3_bookspine_expE_ck16"
    / "sam3_bookspine_expE_ck16_inference.pt"
)
CHECKPOINT_SHA256 = (
    "1b43d2906da3f11d9e49c056b169f567762fb96cf91579df0110f66fa27afcdd"
)


def main() -> int:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise RuntimeError(f"FT output directory is not empty: {OUTPUT_ROOT}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    # These are consumed by the separately launched SAM3 service.  Every
    # threshold remains at the service baseline default; only checkpoint and
    # its integrity hash differ.
    os.environ["SAM3_MODEL_PATH"] = str(CHECKPOINT.resolve())
    os.environ["SAM3_BOOK_SPINE_EXPECTED_SHA256"] = CHECKPOINT_SHA256

    from detection.pro_handbook.thesis import case_artifacts
    from detection.pro_handbook.thesis import thesis_ret

    thesis_ret.EVALUATION_ROOT = EVALUATION_ROOT
    thesis_ret.INTRINSICS_JSON = EVALUATION_ROOT / "camera_intrinsics.json"
    thesis_ret.GT_ROOT = EVALUATION_ROOT / "thesis_ret" / "SegmentationClass"
    thesis_ret.LABELMAP_PATH = EVALUATION_ROOT / "thesis_ret" / "labelmap.txt"
    thesis_ret.OUTPUT_ROOT = OUTPUT_ROOT
    case_artifacts.DEFAULT_ARTIFACT_ROOT = OUTPUT_ROOT / "case_artifacts"

    return thesis_ret.main(
        [
            "--iou-threshold",
            "0.5",
            "--sam-device",
            "gpu",
            "--ocr-resident",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
