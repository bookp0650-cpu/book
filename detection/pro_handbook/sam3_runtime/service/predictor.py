from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from .settings import (
    MIN_AREA,
    NMS_IOU_THRESHOLD,
    PROCESSOR_CONFIDENCE_THRESHOLD,
    PROMPT,
    SCORE_THRESHOLD,
)


def infer_array(loaded, image: np.ndarray, prompt: str = PROMPT):
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected RGB uint8 HxWx3 array, got {image.shape} {image.dtype}")
    if loaded.device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    # Sam3Adapter accepts ndarray as OpenCV BGR, while the service contract is
    # RGB. Passing RGB as PIL-equivalent is achieved by reversing to BGR here;
    # the adapter then performs its canonical BGR->RGB conversion exactly once.
    raw_instances = loaded.adapter.predict(
        np.ascontiguousarray(image[:, :, ::-1]),
        prompt=prompt,
        score_threshold=SCORE_THRESHOLD,
        min_area=MIN_AREA,
    )
    from core.mask_nms import apply_mask_nms
    instances = apply_mask_nms(
        raw_instances, iou_thresh=NMS_IOU_THRESHOLD, metric="iou", mode="suppress"
    ).instances
    if loaded.device == "cuda":
        torch.cuda.synchronize()
    masks = instances.masks.astype(bool, copy=False)
    scores = instances.scores.astype(np.float32, copy=False)
    xywh = instances.bboxes.astype(np.float32, copy=False)
    boxes = xywh.copy()
    if len(boxes):
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2] - 1
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3] - 1
    gpu_mb = 0.0
    if loaded.device == "cuda":
        gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    meta = {
        "count": int(len(masks)),
        "raw_count": int(raw_instances.count),
        "nms_count": int(instances.count),
        "inference_seconds": time.perf_counter() - started,
        "gpu_memory_mb": gpu_mb,
        "score_threshold": SCORE_THRESHOLD,
        "processor_confidence_threshold": PROCESSOR_CONFIDENCE_THRESHOLD,
        "min_area": MIN_AREA,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
    }
    return masks, boxes, scores, meta


def infer_to_npz(loaded, input_path: Path, output_path: Path, prompt: str = PROMPT) -> dict:
    image = np.load(input_path, allow_pickle=False)
    masks, boxes, scores, meta = infer_array(loaded, image, prompt)
    np.savez_compressed(output_path, masks=masks, boxes=boxes, scores=scores)
    return meta


def infer_storage_to_npz(
    loaded_models,
    input_path: Path,
    spine_output_path: Path,
    book_end_output_path: Path,
) -> dict:
    """Run both resident models sequentially against one loaded RGB array."""
    image = np.load(input_path, allow_pickle=False)
    spine_masks, spine_boxes, spine_scores, spine_meta = infer_array(
        loaded_models.book_spine,
        image,
        "book spine",
    )
    np.savez_compressed(
        spine_output_path,
        masks=spine_masks,
        boxes=spine_boxes,
        scores=spine_scores,
    )
    end_masks, end_boxes, end_scores, end_meta = infer_array(
        loaded_models.book_end,
        image,
        "book end",
    )
    np.savez_compressed(
        book_end_output_path,
        masks=end_masks,
        boxes=end_boxes,
        scores=end_scores,
    )
    return {
        "inference_order": ["book_spine", "book_end"],
        "parallel_inference": False,
        "book_spine": spine_meta,
        "book_end": end_meta,
        "total_inference_seconds": float(
            spine_meta["inference_seconds"] + end_meta["inference_seconds"]
        ),
    }
