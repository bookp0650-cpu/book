#!/usr/bin/env python3
"""Run the existing fine-tuned SAM3 on a saved RGB image for ``book end``.

This script is intentionally limited to offline visual inspection.  It does not
import or start RealSense, ROS, robot control, ROI cropping, book-spine removal,
or free-space extraction code.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from detection.pro_handbook.sam3_runtime.integration_service_manager import (
    Sam3ServiceSession,
)
from detection.pro_handbook.sam3_runtime.service.client import Sam3BatchInfer
from detection.pro_handbook.sam3_runtime.service.settings import MODEL_PATH


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "captures" / "book_end_check"
PROMPT = "book end"
OVERLAY_ALPHA = 0.45
INSTANCE_COLORS_RGB = np.asarray(
    [
        [255, 64, 64],
        [64, 220, 96],
        [64, 128, 255],
        [255, 192, 64],
        [192, 64, 255],
        [64, 224, 224],
    ],
    dtype=np.float32,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use the existing fine-tuned SAM3 service to visualize all "
            "'book end' masks in a saved RGB image."
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to an existing saved shelf image.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Parent output directory (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    return parser.parse_args()


def _make_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        run_dir.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(
            f"output directory already exists; wait one second and retry: {run_dir}"
        ) from exc
    return run_dir


def _validate_service_checkpoint(health: dict[str, Any]) -> Path:
    expected = MODEL_PATH.expanduser().resolve()
    reported = health.get("model")
    if not reported:
        raise RuntimeError("SAM3 service health response did not report its checkpoint")
    actual = Path(str(reported)).expanduser().resolve()
    if actual != expected:
        raise RuntimeError(
            "SAM3 service checkpoint mismatch: "
            f"expected {expected}, service reported {actual}"
        )
    return actual


def _normalize_masks(
    masks_list: list[np.ndarray], image_shape: tuple[int, int]
) -> np.ndarray:
    if not masks_list:
        return np.zeros((0, *image_shape), dtype=bool)
    masks = np.asarray(masks_list, dtype=bool)
    expected_shape = (len(masks_list), *image_shape)
    if masks.shape != expected_shape:
        raise RuntimeError(
            f"unexpected SAM3 mask shape: expected {expected_shape}, got {masks.shape}"
        )
    return masks


def _make_overlay(rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    for index, mask in enumerate(masks):
        color = INSTANCE_COLORS_RGB[index % len(INSTANCE_COLORS_RGB)]
        overlay[mask] = (
            (1.0 - OVERLAY_ALPHA) * overlay[mask] + OVERLAY_ALPHA * color
        ).astype(np.uint8)
    return overlay


def _instance_metadata(
    sam_data: list[dict[str, Any]], masks: np.ndarray
) -> list[dict[str, Any]]:
    instances = []
    for index, (item, mask) in enumerate(zip(sam_data, masks, strict=True)):
        instances.append(
            {
                "index": index,
                "score": float(item["score"]),
                "box_xyxy": [
                    float(item["box"][key]) for key in ("x1", "y1", "x2", "y2")
                ],
                "area_px": int(mask.sum()),
                "overlay_color_rgb": INSTANCE_COLORS_RGB[
                    index % len(INSTANCE_COLORS_RGB)
                ].astype(int).tolist(),
            }
        )
    return instances


def main() -> None:
    args = _parse_args()
    source = args.image.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input image not found: {source}")

    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    run_dir = _make_run_dir(args.output_root.expanduser().resolve())
    Image.fromarray(rgb, mode="RGB").save(run_dir / "input.png")

    session = Sam3ServiceSession()
    with session:
        checkpoint = _validate_service_checkpoint(session.health_payload or {})
        runner = Sam3BatchInfer(endpoint=session.endpoint, prompt=PROMPT)
        masks_list, sam_data = runner.infer_masks(Image.fromarray(rgb, mode="RGB"))

    masks = _normalize_masks(masks_list, rgb.shape[:2])
    if len(sam_data) != len(masks):
        raise RuntimeError(
            "SAM3 masks and instance metadata counts do not match: "
            f"{len(masks)} masks, {len(sam_data)} metadata entries"
        )

    combined_mask = (
        np.any(masks, axis=0) if len(masks) else np.zeros(rgb.shape[:2], bool)
    )
    overlay = _make_overlay(rgb, masks)
    Image.fromarray(combined_mask.astype(np.uint8) * 255, mode="L").save(
        run_dir / "book_end_mask.png"
    )
    Image.fromarray(overlay, mode="RGB").save(run_dir / "book_end_overlay.png")

    metadata = {
        "source_image": str(source),
        "prompt": PROMPT,
        "checkpoint": str(checkpoint),
        "image_width": int(rgb.shape[1]),
        "image_height": int(rgb.shape[0]),
        "mask_count": int(len(masks)),
        "combined_mask_area_px": int(combined_mask.sum()),
        "overlay_alpha": OVERLAY_ALPHA,
        "instances": _instance_metadata(sam_data, masks),
        "service_inference": runner.last_metadata,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"output_dir={run_dir}")
    print(f"checkpoint={checkpoint}")
    print(f"prompt={PROMPT!r}")
    print(f"mask_count={len(masks)}")


if __name__ == "__main__":
    main()
