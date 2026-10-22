"""GPU-batched OCR polygon to SAM mask matching.

This module is an additive GPU counterpart of :mod:`only_one_tilted`.  It keeps
the original assignment rules and public entry point while falling back to the
original CPU implementation when CUDA is unavailable or runs out of memory.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from . import only_one_tilted as _cpu


DEFAULT_OCR_BATCH_SIZE = 4


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "cublas" in message)


def _cpu_fallback(
    all_boxes,
    all_mojis,
    sam_quads,
    rgb_path,
    forced_angle,
    return_debug,
    return_runtime,
    reason: str,
):
    started = time.perf_counter()
    result = _cpu.match_text_to_mask(
        all_boxes,
        all_mojis,
        sam_quads,
        rgb_path,
        forced_angle=forced_angle,
        return_debug=return_debug,
    )
    runtime = {
        "backend": "cpu",
        "fallback_reason": reason,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_compute_seconds": None,
    }
    if not return_runtime:
        return result
    if return_debug:
        book_texts, book_boxes, debug_rows = result
        return book_texts, book_boxes, debug_rows, runtime
    book_texts, book_boxes = result
    return book_texts, book_boxes, runtime


def _gpu_intersections(
    text_masks: list[np.ndarray],
    sam_masks: list[np.ndarray],
    ocr_batch_size: int,
    device: str,
):
    """Calculate batched pairwise IoUs on CUDA and return exact intersections."""
    import torch

    if ocr_batch_size < 1:
        raise ValueError(f"ocr_batch_size must be >= 1, got {ocr_batch_size}")

    # float32 matmul is integer-exact while the flattened mask has <= 2**24
    # pixels. Larger images use the CPU implementation to preserve decisions.
    if text_masks[0].size > 2**24:
        raise ValueError("mask has more than 2**24 pixels")

    gpu_started = time.perf_counter()
    sam_cpu = torch.from_numpy(np.stack(sam_masks, axis=0).astype(np.bool_, copy=False))
    sam_flat = sam_cpu.to(device=device, non_blocking=False).flatten(1).float()
    sam_areas = sam_flat.sum(dim=1)

    intersection_parts = []
    for start in range(0, len(text_masks), ocr_batch_size):
        batch_np = np.stack(text_masks[start : start + ocr_batch_size], axis=0)
        batch_cpu = torch.from_numpy(batch_np.astype(np.bool_, copy=False))
        text_flat = batch_cpu.to(device=device, non_blocking=False).flatten(1).float()

        intersections = torch.mm(text_flat, sam_flat.t())
        text_areas = text_flat.sum(dim=1)
        unions = text_areas[:, None] + sam_areas[None, :] - intersections

        # Keep the IoU operation on CUDA, following sam3/perflib/masks_ops.py.
        # Selection below uses integer counts to reproduce Python float division
        # and np.isclose behavior of the existing CPU implementation exactly.
        ious = intersections / unions.clamp(min=1)
        intersection_parts.append(intersections.to(dtype=torch.int64).cpu())
        _ = ious.cpu()

    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    gpu_seconds = time.perf_counter() - gpu_started
    return torch.cat(intersection_parts, dim=0).numpy(), gpu_seconds


def _match_cuda(
    all_boxes,
    all_mojis,
    sam_quads,
    rgb_path,
    forced_angle,
    return_debug,
    return_runtime,
    ocr_batch_size,
    device,
):
    total_started = time.perf_counter()
    image = cv2.imread(str(rgb_path))
    if image is None:
        raise FileNotFoundError(f"画像が読めませんでした: {rgb_path}")
    h, w = image.shape[:2]

    if forced_angle in (90, 180, 270):
        all_boxes = [
            _cpu.unrotate_poly_to_original(poly, forced_angle, w=w, h=h)
            for poly in all_boxes
        ]

    sam_masks = [_cpu._mask_to_binary(mask, h, w) for mask in sam_quads]
    names = [f"mask_{index + 1}" for index in range(len(sam_masks))]
    book_texts = {name: [] for name in names}
    book_boxes = {
        name: _cpu._mask_bbox(mask) for name, mask in zip(names, sam_masks)
    }

    pairs = list(zip(all_mojis, all_boxes))
    text_masks = [_cpu._poly_to_mask(quad, h, w) for _, quad in pairs]
    text_areas = [int(mask.sum()) for mask in text_masks]
    sam_areas = [int(mask.sum()) for mask in sam_masks]

    if not sam_masks or not text_masks:
        return _cpu_fallback(
            all_boxes,
            all_mojis,
            sam_quads,
            rgb_path,
            None,
            return_debug,
            return_runtime,
            "empty OCR or SAM mask input",
        )

    intersections, gpu_seconds = _gpu_intersections(
        text_masks, sam_masks, ocr_batch_size, device
    )

    debug_rows = []
    for row_index, ((moji, _), text_area) in enumerate(zip(pairs, text_areas)):
        ocr_index = row_index + 1
        if text_area <= 0:
            debug_rows.append(
                {
                    "ocr_index": ocr_index,
                    "text": moji,
                    "matched": None,
                    "best_iou": 0.0,
                    "text_area": 0,
                    "note": "text polygon area is zero",
                }
            )
            continue

        best_name = None
        best_iou = 0.0
        best_inter = 0
        best_union = 0
        best_text_cover = 0.0
        per_mask_scores = []

        for mask_index, name in enumerate(names):
            inter = int(intersections[row_index, mask_index])
            union = int(text_area + sam_areas[mask_index] - inter)
            iou = float(inter / union) if inter > 0 and union > 0 else 0.0
            text_cover = float(inter / text_area) if text_area > 0 else 0.0

            per_mask_scores.append(
                {
                    "name": name,
                    "iou": float(iou),
                    "intersection": int(inter),
                    "union": int(union),
                    "text_cover": float(text_cover),
                }
            )

            # Preserve the current rule and iteration order exactly: maximum
            # IoU, then OCR coverage when np.isclose says the IoUs are tied.
            # A complete tie keeps the first SAM mask.
            if iou > best_iou or (
                np.isclose(iou, best_iou) and text_cover > best_text_cover
            ):
                best_name = name
                best_iou = float(iou)
                best_inter = int(inter)
                best_union = int(union)
                best_text_cover = float(text_cover)

        matched = best_name if best_inter > 0 else None
        if matched is not None:
            book_texts[matched].append(moji)

        debug_rows.append(
            {
                "ocr_index": ocr_index,
                "text": moji,
                "matched": matched,
                "best_iou": float(best_iou),
                "best_intersection": int(best_inter),
                "best_union": int(best_union),
                "best_text_cover": float(best_text_cover),
                "text_area": int(text_area),
                "per_mask_scores": per_mask_scores,
            }
        )

    runtime = {
        "backend": "cuda",
        "device": device,
        "ocr_batch_size": int(ocr_batch_size),
        "num_ocr_masks": len(text_masks),
        "num_sam_masks": len(sam_masks),
        "elapsed_seconds": time.perf_counter() - total_started,
        "gpu_compute_seconds": gpu_seconds,
        "fallback_reason": None,
    }
    if return_runtime:
        if return_debug:
            return book_texts, book_boxes, debug_rows, runtime
        return book_texts, book_boxes, runtime
    if return_debug:
        return book_texts, book_boxes, debug_rows
    return book_texts, book_boxes


def match_text_to_mask(
    all_boxes,
    all_mojis,
    sam_quads,
    rgb_path,
    forced_angle=None,
    return_debug=False,
    *,
    ocr_batch_size: int = DEFAULT_OCR_BATCH_SIZE,
    return_runtime: bool = False,
):
    """Match OCR polygons to SAM masks using CUDA, with safe CPU fallback."""
    if ocr_batch_size < 1:
        raise ValueError(f"ocr_batch_size must be >= 1, got {ocr_batch_size}")

    try:
        import torch
    except (ImportError, OSError) as exc:
        return _cpu_fallback(
            all_boxes,
            all_mojis,
            sam_quads,
            rgb_path,
            forced_angle,
            return_debug,
            return_runtime,
            f"PyTorch unavailable: {exc}",
        )

    if not torch.cuda.is_available():
        return _cpu_fallback(
            all_boxes,
            all_mojis,
            sam_quads,
            rgb_path,
            forced_angle,
            return_debug,
            return_runtime,
            "CUDA unavailable",
        )

    device = "cuda"
    try:
        result = _match_cuda(
            all_boxes,
            all_mojis,
            sam_quads,
            rgb_path,
            forced_angle,
            return_debug,
            return_runtime,
            ocr_batch_size,
            device,
        )
        # The next case runs SAM3 and OCR before matching again. Returning the
        # unused allocator cache avoids accumulating cross-process VRAM pressure.
        torch.cuda.empty_cache()
        return result
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if not isinstance(exc, torch.cuda.OutOfMemoryError) and not _is_cuda_oom(exc):
            raise
        torch.cuda.empty_cache()
        return _cpu_fallback(
            all_boxes,
            all_mojis,
            sam_quads,
            rgb_path,
            forced_angle,
            return_debug,
            return_runtime,
            f"CUDA out of memory: {exc}",
        )
    except ValueError as exc:
        if "more than 2**24 pixels" not in str(exc):
            raise
        return _cpu_fallback(
            all_boxes,
            all_mojis,
            sam_quads,
            rgb_path,
            forced_angle,
            return_debug,
            return_runtime,
            str(exc),
        )


def match_book_name(query, book_texts, shot_dir, threshold=40.0):
    """Keep the existing text-score threshold and ordering unchanged."""
    return _cpu.match_book_name(query, book_texts, shot_dir, threshold=threshold)


def match_text_to_mask_main(
    query,
    masks,
    shot_dir,
    threshold=40,
    *,
    ocr_batch_size: int = DEFAULT_OCR_BATCH_SIZE,
):
    """Drop-in entry point; CUDA work begins only after OCR results are loaded."""
    ocr_json_path = Path(shot_dir) / "ocr_result.json"
    rgb_path = Path(shot_dir) / "after_init_rgb.png"

    # This read is deliberately before match_text_to_mask(), which is the first
    # place CUDA availability is checked or any CUDA tensor is allocated.
    all_text_polys, all_texts = _cpu.load_text_data(ocr_json_path)
    book_texts, book_boxes, debug_rows, runtime = match_text_to_mask(
        all_text_polys,
        all_texts,
        masks,
        rgb_path,
        forced_angle=90,
        return_debug=True,
        ocr_batch_size=ocr_batch_size,
        return_runtime=True,
    )

    _cpu.save_json(
        Path(shot_dir) / "text_mask_iou_debug.json",
        {
            "forced_angle": 90,
            "num_masks": len(masks),
            "num_ocr_boxes": len(all_texts),
            "assignments": debug_rows,
            "runtime": runtime,
        },
    )

    matches, _ = match_book_name(query, book_texts, shot_dir, threshold)
    return [
        {
            "name": name,
            "score": score,
            "box": book_boxes.get(name),
            "forced_angle": 90,
        }
        for name, score in matches
    ]


__all__ = [
    "DEFAULT_OCR_BATCH_SIZE",
    "match_book_name",
    "match_text_to_mask",
    "match_text_to_mask_main",
]
