from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from detection.pro_handbook.sam3_runtime.service.model_loader import load_model
from detection.pro_handbook.sam3_runtime.service.predictor import infer_array

PROMPT = "book spine"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, out = args.image.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=out / "inference.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    bgr = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError(f"expected uint8 HxWx3 PNG, got {None if bgr is None else (bgr.shape, bgr.dtype)}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    shutil.copy2(source, out / "input_copy.png")
    write_json(out / "input_info.json", {
        "source_absolute_path": str(source), "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
        "array_shape": list(bgr.shape), "source_dtype": str(bgr.dtype), "source_channel_order": "BGR (OpenCV)",
        "sam3_channel_order": "RGB", "preprocessing": "OpenCV IMREAD_UNCHANGED; cv2.COLOR_BGR2RGB; official Sam3Processor.set_image; no caller resize or normalization",
        "text_prompt": PROMPT,
    })
    logging.info("loading model checkpoint with strict=True")
    loaded = load_model()
    logging.info("strict load ok; missing=%s unexpected=%s", loaded.missing_keys, loaded.unexpected_keys)
    if loaded.device == "cuda": torch.cuda.reset_peak_memory_stats()
    warm_started = time.perf_counter(); infer_array(loaded, rgb, PROMPT); warm_seconds = time.perf_counter() - warm_started
    if loaded.device == "cuda": torch.cuda.synchronize()
    masks, boxes, scores, infer_meta = infer_array(loaded, rgb, PROMPT)
    if loaded.device == "cuda": torch.cuda.synchronize()
    np.savez_compressed(out / "raw_masks.npz", masks=masks)
    selected = int(np.argmax(scores)) if len(scores) else None
    selected_mask = masks[selected] if selected is not None else np.zeros(rgb.shape[:2], dtype=bool)
    Image.fromarray(selected_mask.astype(np.uint8) * 255).save(out / "selected_mask.png")
    overlay = rgb.copy(); overlay[selected_mask] = (0.45 * overlay[selected_mask] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
    Image.fromarray(overlay).save(out / "overlay.png")
    write_json(out / "boxes.json", boxes.tolist())
    write_json(out / "scores.json", scores.tolist())
    write_json(out / "metadata.json", {
        "route": "standalone", "prompt": PROMPT, "mask_count": int(len(masks)), "mask_shape": list(masks.shape),
        "mask_dtype": str(masks.dtype), "mask_areas": [int(m.sum()) for m in masks], "selected_index": selected,
        "selected_area": int(selected_mask.sum()), "selected_score": float(scores[selected]) if selected is not None else None,
        "selected_box_xyxy": boxes[selected].tolist() if selected is not None else None,
        "strict_load": loaded.strict_load, "missing_keys": loaded.missing_keys, "unexpected_keys": loaded.unexpected_keys,
    })
    runtime = {
        "python": __import__("sys").version, "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "device": loaded.device, "device_name": torch.cuda.get_device_name(0) if loaded.device == "cuda" else None,
        "model_load_seconds": loaded.model_load_seconds, "warmup_seconds": warm_seconds,
        "inference_seconds": infer_meta["inference_seconds"], "gpu_max_memory_mb": infer_meta["gpu_memory_mb"],
    }
    write_json(out / "runtime.json", runtime)
    logging.info("complete %s", {**runtime, "mask_count": len(masks), "selected_index": selected})
    print(json.dumps({**runtime, "mask_count": len(masks), "selected_index": selected}, indent=2))


if __name__ == "__main__":
    main()
