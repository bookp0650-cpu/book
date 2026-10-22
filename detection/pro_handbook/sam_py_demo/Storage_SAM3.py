"""Live-compatible SAM3 storage recognition plus legacy Storage_rev geometry.

The completed recognition policy lives in ``offline_100test_storage_space_sam3``.
This module keeps the recognition-time selected-space mask unchanged through
classification.  After classification, the selected triangular mask can be
corrected for held-book occlusion before final target use and visualization.
"""
from __future__ import annotations

import atexit
import importlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import math

import offline_100test_storage_space_sam3 as storage_recognition
from core.npz_io import InstanceSet
from detection.pro_handbook.sam3_runtime.integration_service_manager import (
    Sam3ServiceSession,
)
from detection.pro_handbook.sam3_runtime.service.client import Sam3StorageClient


class _LazyLegacyGeometry:
    """Keep RealSense/ROS-side dependencies out of the SAM3 service venv."""

    def __getattr__(self, name: str) -> Any:
        module = importlib.import_module(
            "detection.pro_handbook.sam_py_demo.Storage_rev"
        )
        return getattr(module, name)


legacy_geometry = _LazyLegacyGeometry()


HELD_BOOK_EXPECTED_CENTER_X_RATIO = 0.58
HELD_BOOK_CENTER_X_RATIO_MIN = 0.52
HELD_BOOK_CENTER_X_RATIO_MAX = 0.63
HELD_BOOK_MIN_HEIGHT_RATIO = 0.75


@dataclass(frozen=True)
class HeldBookDetectionConfig:
    """Select one grasped-book spine by its stable horizontal image position.

    The X defaults come from deduplicated October 2026 saved captures: known
    held-book center-X ratios span 0.5266--0.6246 with median 0.5789.  The
    minimum height is only an auxiliary false-positive guard; top and bottom
    margins remain diagnostic fields for compatibility and are not gates.
    """

    enabled: bool = True
    expected_center_x_ratio: float = HELD_BOOK_EXPECTED_CENTER_X_RATIO
    min_center_x_ratio: float = HELD_BOOK_CENTER_X_RATIO_MIN
    max_center_x_ratio: float = HELD_BOOK_CENTER_X_RATIO_MAX
    min_height_ratio: float = HELD_BOOK_MIN_HEIGHT_RATIO
    max_top_margin_ratio: float = 0.02
    max_bottom_margin_ratio: float = 0.10


DEFAULT_HELD_BOOK_CONFIG = HeldBookDetectionConfig()

# Use several physical-bottom-side rows instead of one bbox corner or one
# potentially noisy mask pixel.  In the saved storage images, smaller image y
# is the physical shelf/book-bottom side.
LEFT_NEIGHBOR_BOTTOM_CORNER_FIT_ROWS = 10


def _add_timing_ms(
    timing_sink: dict[str, float] | None,
    name: str,
    elapsed_seconds: float,
) -> None:
    if timing_sink is not None:
        timing_sink[name] = (
            timing_sink.get(name, 0.0) + elapsed_seconds * 1000.0
        )


def _merge_timing_ms(
    timing_sink: dict[str, float] | None,
    values: dict[str, float],
) -> None:
    if timing_sink is None:
        return
    for name, value in values.items():
        timing_sink[name] = timing_sink.get(name, 0.0) + float(value)


def detect_held_book_spines(
    spine_instances: InstanceSet,
    config: HeldBookDetectionConfig = DEFAULT_HELD_BOOK_CONFIG,
) -> list[dict[str, Any]]:
    """Return per-instance evidence and one position-ranked held-book decision."""
    masks = np.asarray(spine_instances.masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"expected spine masks NxHxW, got {masks.shape}")
    image_height = int(masks.shape[1]) if masks.ndim == 3 else 0
    image_width = int(masks.shape[2]) if masks.ndim == 3 else 0
    records: list[dict[str, Any]] = []
    for index, mask in enumerate(masks):
        ys, xs = np.where(mask)
        if not len(xs):
            records.append(
                {
                    "instance_index": index,
                    "is_held_book": False,
                    "reason": "empty_mask",
                    "bbox_xyxy": None,
                }
            )
            continue
        y_min = int(ys.min())
        y_max = int(ys.max())
        x_min = int(xs.min())
        x_max = int(xs.max())
        height_px = y_max - y_min + 1
        bbox_width_px = x_max - x_min + 1
        center_x = 0.5 * (x_min + x_max + 1)
        center_x_ratio = float(center_x / max(image_width, 1))
        height_ratio = float(height_px / max(image_height, 1))
        top_margin_ratio = float(y_min / max(image_height, 1))
        bottom_margin_px = image_height - 1 - y_max
        bottom_margin_ratio = float(bottom_margin_px / max(image_height, 1))
        bbox_x_min_ratio = float(x_min / max(image_width, 1))
        bbox_x_max_ratio = float((x_max + 1) / max(image_width, 1))
        x_band_overlap_ratio = max(
            0.0,
            min(bbox_x_max_ratio, config.max_center_x_ratio)
            - max(bbox_x_min_ratio, config.min_center_x_ratio),
        )
        checks = {
            "center_x_ratio_in_expected_band": (
                config.min_center_x_ratio
                <= center_x_ratio
                <= config.max_center_x_ratio
            ),
            "height_ratio_at_least_minimum": (
                height_ratio >= config.min_height_ratio
            ),
        }
        is_candidate = bool(config.enabled and all(checks.values()))
        records.append(
            {
                "instance_index": index,
                "instance_id": int(spine_instances.instance_ids[index]),
                "score": float(spine_instances.scores[index]),
                "is_held_book": False,
                "is_held_book_candidate": is_candidate,
                "reason": "held_book_candidate_not_ranked",
                "bbox_xyxy": [x_min, y_min, x_max + 1, y_max + 1],
                "bbox_width_px": bbox_width_px,
                "image_width_px": image_width,
                "center_x": center_x,
                "center_x_ratio": center_x_ratio,
                "expected_center_x_ratio": config.expected_center_x_ratio,
                "center_x_distance_ratio": abs(
                    center_x_ratio - config.expected_center_x_ratio
                ),
                "expected_center_x_band_ratio": [
                    config.min_center_x_ratio,
                    config.max_center_x_ratio,
                ],
                "x_band_overlap_ratio": x_band_overlap_ratio,
                "mask_height_px": height_px,
                "image_height_px": image_height,
                "height_ratio": height_ratio,
                "top_margin_px": y_min,
                "top_margin_ratio": top_margin_ratio,
                "bottom_margin_px": bottom_margin_px,
                "bottom_margin_ratio": bottom_margin_ratio,
                "checks": checks,
            }
        )

    candidate_records = [
        record for record in records if record.get("is_held_book_candidate")
    ]
    selected_record = (
        max(
            candidate_records,
            key=lambda record: (
                float(record["x_band_overlap_ratio"]),
                -float(record["center_x_distance_ratio"]),
                float(record["height_ratio"]),
                float(record["score"]),
                -int(record["instance_index"]),
            ),
        )
        if candidate_records
        else None
    )
    for record in records:
        if record is selected_record:
            record["is_held_book"] = True
            record["reason"] = "x_position_candidate_selected"
        elif record.get("is_held_book_candidate"):
            record["reason"] = "x_position_candidate_not_selected"
        elif not config.enabled:
            record["reason"] = "held_book_detection_disabled"
        else:
            record["reason"] = "x_position_or_minimum_height_rule_not_matched"
    return records


def _save_held_book_debug(
    output_dir: Path,
    rgb: np.ndarray,
    spine_instances: InstanceSet,
    detections: list[dict[str, Any]],
) -> dict[str, str]:
    held_indices = [
        int(item["instance_index"])
        for item in detections
        if item.get("is_held_book")
    ]
    shape = rgb.shape[:2]
    held_union = (
        np.any(spine_instances.masks[held_indices], axis=0)
        if held_indices
        else np.zeros(shape, dtype=bool)
    )
    held_overlay_path = output_dir / "held_book_spine_overlay.png"
    storage_recognition.save_png(
        held_overlay_path,
        storage_recognition.overlay_masks(rgb, held_union[None, ...]),
    )
    return {
        "held_book_spine_overlay_path": str(held_overlay_path),
    }


def _get_realsense_module() -> Any:
    try:
        return importlib.import_module("pyrealsense2")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RealSense処理にはpyrealsense2が必要です．"
            "SAM3 service専用venvではなく，RealSense側の環境で実行してください．"
        ) from exc


class StorageSam3Runtime:
    """Own or reuse one dual-model SAM3 service for repeated recognition."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        timeout: float = 120.0,
        service_session: Sam3ServiceSession | None = None,
        client: Sam3StorageClient | None = None,
    ) -> None:
        self.session = service_session or Sam3ServiceSession(
            endpoint=endpoint,
            required_capability="storage_dual_model",
        )
        self.client = client or Sam3StorageClient(
            endpoint=self.session.endpoint,
            timeout=timeout,
        )
        self.started = False
        self.inference_requests = 0
        self.start_health: dict[str, Any] | None = None
        self.last_prepare_seconds = 0.0

    def start(self) -> dict[str, Any]:
        started = time.perf_counter()
        if not self.started:
            self.start_health = self.session.ensure_ready()
            self.started = True
        self.last_prepare_seconds = time.perf_counter() - started
        return self.start_health or {}

    def infer(
        self,
        rgb: np.ndarray,
    ) -> tuple[InstanceSet, InstanceSet, dict[str, Any]]:
        first_prepare = not self.started
        self.start()
        spine_instances, end_instances, metadata = self.client.infer_instances(rgb)
        metadata["service_prepare_seconds"] = self.last_prepare_seconds
        initialization_observed = bool(
            first_prepare
            and not getattr(
                self.session,
                "used_preexisting_ready_service",
                False,
            )
        )
        model_load_seconds = (
            (self.start_health or {}).get("model_load_seconds", {})
            if initialization_observed
            else {}
        )
        metadata["model_initialization"] = {
            "observed_during_request": initialization_observed,
            "total_seconds": (
                self.last_prepare_seconds if initialization_observed else 0.0
            ),
            "sam3_model_load_seconds": model_load_seconds.get("total"),
            "book_spine_model_load_seconds": model_load_seconds.get("book_spine"),
            "book_end_model_load_seconds": model_load_seconds.get("book_end"),
            "ocr_model_load_seconds": None,
            "ocr_used_by_storage_recognition": False,
        }
        self.inference_requests += 1
        return spine_instances, end_instances, metadata

    def health(self) -> dict[str, Any]:
        self.start()
        return self.client.health()

    def close(self) -> bool:
        stopped = self.session.stop_if_owned()
        self.started = False
        return stopped


_DEFAULT_RUNTIME: StorageSam3Runtime | None = None


def get_default_runtime() -> StorageSam3Runtime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = StorageSam3Runtime()
    return _DEFAULT_RUNTIME


def _close_default_runtime() -> None:
    if _DEFAULT_RUNTIME is not None:
        _DEFAULT_RUNTIME.close()


atexit.register(_close_default_runtime)


def recognize_storage_space(
    rgb: np.ndarray,
    *,
    runtime: StorageSam3Runtime | None = None,
    spine_instances: InstanceSet | None = None,
    end_instances: InstanceSet | None = None,
    output_dir: str | Path | None = None,
    held_book_config: HeldBookDetectionConfig = DEFAULT_HELD_BOOK_CONFIG,
    timing_sink: dict[str, float] | None = None,
) -> tuple[
    storage_recognition.StorageRecognitionResult,
    InstanceSet,
    InstanceSet,
    dict[str, Any],
]:
    """Run offline recognition with fixed image-left held-book occlusion."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"expected RGB uint8 HxWx3 array, got {image.shape} {image.dtype}"
        )
    if (spine_instances is None) != (end_instances is None):
        raise ValueError(
            "spine_instances and end_instances must be supplied together"
        )
    if spine_instances is None or end_instances is None:
        active_runtime = runtime or get_default_runtime()
        sam3_started = time.perf_counter()
        spine_instances, end_instances, inference_metadata = active_runtime.infer(
            image
        )
        _add_timing_ms(
            timing_sink,
            "sam3_client_wall_ms",
            time.perf_counter() - sam3_started,
        )
        spine_meta = inference_metadata.get("book_spine", {})
        end_meta = inference_metadata.get("book_end", {})
        client_meta = inference_metadata.get("client_timing", {})
        _add_timing_ms(
            timing_sink,
            "sam3_service_model_prepare_ms",
            float(inference_metadata.get("service_prepare_seconds", 0.0)),
        )
        _add_timing_ms(
            timing_sink,
            "sam3_book_spine_ms",
            float(
                spine_meta.get(
                    "model_predict_seconds",
                    spine_meta.get("inference_seconds", 0.0),
                )
            ),
        )
        _add_timing_ms(
            timing_sink,
            "sam3_book_end_ms",
            float(
                end_meta.get(
                    "model_predict_seconds",
                    end_meta.get("inference_seconds", 0.0),
                )
            ),
        )
        _add_timing_ms(
            timing_sink,
            "mask_postprocess_ms",
            float(spine_meta.get("mask_postprocess_seconds", 0.0))
            + float(end_meta.get("mask_postprocess_seconds", 0.0)),
        )
        _add_timing_ms(
            timing_sink,
            "sam3_communication_wait_ms",
            float(client_meta.get("communication_overhead_seconds", 0.0)),
        )
        sam3_other_seconds = sum(
            float(value)
            for value in (
                client_meta.get("input_save_seconds", 0.0),
                inference_metadata.get("input_load_seconds", 0.0),
                inference_metadata.get("book_spine_output_save_seconds", 0.0),
                inference_metadata.get("book_end_output_save_seconds", 0.0),
                client_meta.get("output_load_seconds", 0.0),
            )
        )
        _add_timing_ms(timing_sink, "sam3_other_ms", sam3_other_seconds)
        initialization_metadata = inference_metadata.get(
            "model_initialization", {}
        )
        if timing_sink is not None:
            timing_sink["model_initialization_total_ms"] = 1000.0 * float(
                initialization_metadata.get("total_seconds", 0.0) or 0.0
            )
            sam3_model_load_seconds = initialization_metadata.get(
                "sam3_model_load_seconds"
            )
            if sam3_model_load_seconds is not None:
                timing_sink["sam3_model_load_ms"] = (
                    1000.0 * float(sam3_model_load_seconds)
                )
    else:
        inference_metadata = {
            "mode": "precomputed_instances",
            "inference_order": ["book_spine", "book_end"],
            "parallel_inference": False,
        }

    held_started = time.perf_counter()
    held_book_detections = detect_held_book_spines(
        spine_instances,
        held_book_config,
    )
    _add_timing_ms(
        timing_sink,
        "held_book_detection_ms",
        time.perf_counter() - held_started,
    )
    held_book_indices = {
        int(item["instance_index"])
        for item in held_book_detections
        if item.get("is_held_book")
    }
    inference_metadata["held_book_detection"] = {
        "config": asdict(held_book_config),
        "held_book_spine_indices": sorted(held_book_indices),
        "instances": held_book_detections,
    }

    result = storage_recognition.recognize_storage_space(
        image,
        spine_instances,
        end_instances,
        blocked_book_spine_indices=held_book_indices,
    )
    _merge_timing_ms(timing_sink, result.timings_ms)
    inference_metadata["held_book_occlusion"] = (
        result.held_book_occlusion_metadata
    )
    if output_dir is not None:
        recognition_dir = Path(output_dir)
        debug_save_before = (
            timing_sink.get("png_jpg_save_ms", 0.0)
            if timing_sink is not None
            else 0.0
        )
        debug_started = time.perf_counter()
        storage_recognition.save_storage_recognition_debug(
            recognition_dir,
            image,
            spine_instances,
            end_instances,
            result,
        )
        debug_elapsed_ms = (time.perf_counter() - debug_started) * 1000.0
        debug_save_after = (
            timing_sink.get("png_jpg_save_ms", 0.0)
            if timing_sink is not None
            else debug_save_before
        )
        if timing_sink is not None:
            timing_sink["debug_image_generation_ms"] = (
                timing_sink.get("debug_image_generation_ms", 0.0)
                + max(0.0, debug_elapsed_ms - (debug_save_after - debug_save_before))
            )
        storage_recognition.save_npz(
            recognition_dir / storage_recognition.SPINE_MODEL.npz_name,
            spine_instances,
        )
        storage_recognition.save_npz(
            recognition_dir / storage_recognition.BOOK_END_MODEL.npz_name,
            end_instances,
        )
        storage_masks_path = storage_recognition.save_storage_recognition_masks(
            recognition_dir,
            result,
            spine_instances,
            held_book_indices,
        )
        debug_save_before = (
            timing_sink.get("png_jpg_save_ms", 0.0)
            if timing_sink is not None
            else 0.0
        )
        debug_started = time.perf_counter()
        held_debug_paths = _save_held_book_debug(
            recognition_dir,
            image,
            spine_instances,
            held_book_detections,
        )
        debug_elapsed_ms = (time.perf_counter() - debug_started) * 1000.0
        debug_save_after = (
            timing_sink.get("png_jpg_save_ms", 0.0)
            if timing_sink is not None
            else debug_save_before
        )
        if timing_sink is not None:
            timing_sink["debug_image_generation_ms"] = (
                timing_sink.get("debug_image_generation_ms", 0.0)
                + max(0.0, debug_elapsed_ms - (debug_save_after - debug_save_before))
            )
        matched_held_books = [
            item for item in held_book_detections if item.get("is_held_book")
        ]
        primary_held_book = matched_held_books[0] if matched_held_books else None
        storage_recognition.write_json(
            recognition_dir / "storage_recognition_metadata.json",
            {
                "book_spine_mask_count": spine_instances.count,
                "book_end_mask_count": end_instances.count,
                "book_end_instances": [
                    {
                        "instance_index": index,
                        "instance_id": int(end_instances.instance_ids[index]),
                        "score": float(end_instances.scores[index]),
                        "bbox_xyxy": [
                            int(value) for value in end_instances.bboxes[index]
                        ],
                    }
                    for index in range(end_instances.count)
                ],
                "selected_left_book_end": result.left_book_end,
                "selected_right_book_end": result.right_book_end,
                "book_end_selection": result.book_end_selection,
                "roi_xyxy": result.roi,
                "roi_error": result.roi_error,
                "roi_determination": result.roi_determination,
                "obstacle_order": result.obstacle_order,
                "adjacent_obstacle_pairs": result.obstacle_pairs,
                "space_candidate_count": len(result.spaces),
                "spaces": result.spaces,
                "rejected_spaces": result.rejected_spaces,
                "final_space_selection": result.final_space_selection,
                "selected_space_id": result.selected_space_id,
                "selected_space": result.selected_space,
                "candidate_pixels_outside_residual": int(
                    np.count_nonzero((result.final_labels > 0) & ~result.residual)
                ),
                "held_book_detection_config": asdict(held_book_config),
                "held_book_spine_indices": sorted(held_book_indices),
                "held_book_spine_index": (
                    min(held_book_indices) if held_book_indices else None
                ),
                "held_book_detection": held_book_detections,
                "held_book_detection_reason": (
                    "x_position_band_with_minimum_height_rule"
                    if held_book_indices
                    else "no_instance_matched"
                ),
                "held_book_bbox": (
                    primary_held_book.get("bbox_xyxy")
                    if primary_held_book
                    else None
                ),
                "held_book_height_ratio": (
                    primary_held_book.get("height_ratio")
                    if primary_held_book
                    else None
                ),
                "held_book_occlusion": (
                    result.held_book_occlusion_metadata
                ),
                "held_book_occlusion_pixel_count": int(
                    np.count_nonzero(result.held_book_occlusion_mask)
                ),
                "held_pair_space_before_occlusion_area_px": int(
                    np.count_nonzero(
                        result.held_pair_space_before_occlusion
                    )
                ),
                "held_pair_space_after_occlusion_area_px": int(
                    np.count_nonzero(
                        result.held_pair_space_after_occlusion
                    )
                ),
                "held_pair_space_removed_by_occlusion_area_px": int(
                    np.count_nonzero(
                        result.held_pair_space_before_occlusion
                        & ~result.held_pair_space_after_occlusion
                    )
                ),
                "storage_masks_npz_path": str(storage_masks_path),
                "storage_masks_npz_keys": [
                    "residual_mask",
                    "roi_mask",
                    "held_book_spine_mask",
                    "held_book_front_mask",
                    "held_book_left_occlusion_band",
                    "held_book_occlusion_mask",
                    "held_pair_space_before_occlusion",
                    "held_pair_space_after_occlusion",
                    "final_labels",
                    "candidate_ids",
                    "selected_space_mask",
                ],
                "candidate_mask_reconstruction": (
                    "storage_masks.npz: final_labels == space_id"
                ),
                **held_debug_paths,
                "inference_metadata": inference_metadata,
            },
        )
    if result.selected_space_id is None:
        raise RuntimeError("収納スペース候補が見つかりませんでした")
    return result, spine_instances, end_instances, inference_metadata


def _selected_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise RuntimeError("収納スペース候補が見つかりませんでした")
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _fit_mask_boundary_x_per_y(
    mask: np.ndarray,
    side: str,
) -> tuple[float, float, np.ndarray]:
    """Fit x = slope * y + intercept to one actual mask boundary."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or side not in {"left", "right"}:
        raise ValueError(
            "invalid mask boundary request: "
            f"shape={binary.shape}, side={side}"
        )
    rows = np.flatnonzero(binary.any(axis=1))
    if rows.size < 2:
        raise RuntimeError("mask boundary has fewer than two occupied rows")
    boundary_x = np.asarray(
        [
            int(np.flatnonzero(binary[y])[0 if side == "left" else -1])
            for y in rows
        ],
        dtype=np.float64,
    )
    fit_rows = rows.astype(np.float64)
    keep = np.ones(fit_rows.shape, dtype=bool)
    for _ in range(3):
        slope, intercept = np.polyfit(fit_rows[keep], boundary_x[keep], 1)
        residual = boundary_x - (slope * fit_rows + intercept)
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        if mad <= 1.0e-9:
            break
        next_keep = np.abs(residual - median) <= 3.5 * 1.4826 * mad
        if np.count_nonzero(next_keep) < 2 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    return float(slope), float(intercept), rows


def build_corrected_triangular_space_mask(
    selected_mask: np.ndarray,
    held_book_mask: np.ndarray,
    *,
    space_type: str,
    tilted_line_p0: tuple[float, float],
    tilted_line_p1: tuple[float, float],
    clip_roi: list[int] | tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extend one selected triangular mask with one held-line bend.

    The visible mask is never removed or rewritten.  The extension direction
    is the image side on which the two fitted lines intersect, so this also
    handles the current camera's physical-bottom == smaller-image-y mapping.
    Until that intersection the visible tilt is continued; after it, the
    boundary follows the fitted held-book side line.  The opposite selected-
    space boundary is continued from its existing mask fit.
    """
    triangular_types = {
        "triangular_left_tilted",
        "triangular_right_tilted",
    }
    original = np.asarray(selected_mask, dtype=bool)
    held = np.asarray(held_book_mask, dtype=bool)
    info: dict[str, Any] = {
        "space_type": str(space_type),
        "mask_correction_applied": False,
        "using_hold_book_side_line": False,
        "bend_point": None,
        "target_point_uses_corrected_mask": False,
    }
    if space_type not in triangular_types:
        info["reason"] = "non-triangular space"
        return original.copy(), info
    if original.ndim != 2 or held.shape != original.shape:
        raise ValueError(
            "selected/held mask shape mismatch: "
            f"selected={original.shape}, held={held.shape}"
        )
    if not np.any(original):
        raise RuntimeError("selected triangular space mask is empty")
    if not np.any(held):
        info["reason"] = "held-book mask is empty"
        return original.copy(), info

    clip_x0, clip_y0, clip_x1, clip_y1 = (
        0,
        0,
        original.shape[1],
        original.shape[0],
    )
    if clip_roi is not None:
        if len(clip_roi) != 4:
            raise ValueError(f"clip_roi must be xyxy: {clip_roi}")
        clip_x0, clip_y0, clip_x1, clip_y1 = (
            int(value) for value in clip_roi
        )
        clip_x0 = int(np.clip(clip_x0, 0, original.shape[1]))
        clip_x1 = int(np.clip(clip_x1, clip_x0, original.shape[1]))
        clip_y0 = int(np.clip(clip_y0, 0, original.shape[0]))
        clip_y1 = int(np.clip(clip_y1, clip_y0, original.shape[0]))
        if clip_x0 >= clip_x1 or clip_y0 >= clip_y1:
            raise ValueError(f"clip_roi is empty after clipping: {clip_roi}")

    selected_y, selected_x = np.where(original)
    held_y, held_x = np.where(held)
    selected_is_right_of_held = float(np.median(selected_x)) > float(
        np.median(held_x)
    )
    held_facing_side = "right" if selected_is_right_of_held else "left"
    held_slope, held_intercept, held_rows = _fit_mask_boundary_x_per_y(
        held,
        held_facing_side,
    )
    held_boundary_x = np.asarray(
        [
            int(
                np.flatnonzero(held[y])[
                    -1 if held_facing_side == "right" else 0
                ]
            )
            for y in held_rows
        ],
        dtype=np.float64,
    )
    held_fit_residual = held_boundary_x - (
        held_slope * held_rows.astype(np.float64) + held_intercept
    )
    # Keep the lower parallel segment on the space side of every actual held
    # mask row.  The one-pixel step is only an exclusive raster boundary; its
    # angle and position otherwise come entirely from the fitted SAM3 mask.
    if held_facing_side == "right":
        held_intercept += float(np.max(held_fit_residual)) + 1.0
    else:
        held_intercept += float(np.min(held_fit_residual)) - 1.0

    p0_x, p0_y = (float(value) for value in tilted_line_p0)
    p1_x, p1_y = (float(value) for value in tilted_line_p1)
    if abs(p1_y - p0_y) <= 1.0e-9:
        raise RuntimeError("visible tilted boundary is horizontal")
    tilted_slope = (p1_x - p0_x) / (p1_y - p0_y)
    tilted_intercept = p0_x - tilted_slope * p0_y
    denominator = tilted_slope - held_slope
    if abs(denominator) <= 1.0e-9:
        info["reason"] = "visible tilt and held-book side line are parallel"
        return original.copy(), info
    bend_y = (held_intercept - tilted_intercept) / denominator
    bend_x = tilted_slope * bend_y + tilted_intercept
    if not math.isfinite(bend_x) or not math.isfinite(bend_y):
        raise RuntimeError("tilted/held-book line intersection is nonfinite")

    tilted_side = "left" if space_type == "triangular_left_tilted" else "right"
    opposite_side = "right" if tilted_side == "left" else "left"
    opposite_slope, opposite_intercept, _ = _fit_mask_boundary_x_per_y(
        original,
        opposite_side,
    )
    selected_y_min = int(selected_y.min())
    selected_y_max = int(selected_y.max())
    if bend_y < selected_y_min:
        extension_direction = -1
        extension_start_y = selected_y_min - 1
        extension_end_y = max(int(held_rows.min()), clip_y0)
    elif bend_y > selected_y_max:
        extension_direction = 1
        extension_start_y = selected_y_max + 1
        extension_end_y = min(int(held_rows.max()), clip_y1 - 1)
    elif bend_y - selected_y_min <= selected_y_max - bend_y:
        extension_direction = -1
        extension_start_y = selected_y_min - 1
        extension_end_y = max(int(held_rows.min()), clip_y0)
    else:
        extension_direction = 1
        extension_start_y = selected_y_max + 1
        extension_end_y = min(int(held_rows.max()), clip_y1 - 1)
    if not 0 <= extension_start_y < original.shape[0]:
        info.update(
            {
                "using_hold_book_side_line": True,
                "bend_point": [float(bend_x), float(bend_y)],
                "reason": "selected mask already reaches the extension image edge",
            }
        )
        return original.copy(), info

    visible_row_widths = []
    for y in np.flatnonzero(original.any(axis=1)):
        xs = np.flatnonzero(original[y])
        visible_row_widths.append(int(xs[-1] - xs[0] + 1))
    maximum_visible_width = max(visible_row_widths)

    corrected = original.copy()
    added_rows = 0
    extension_rows = range(
        extension_start_y,
        extension_end_y + extension_direction,
        extension_direction,
    )
    for y in extension_rows:
        before_bend = (
            y >= bend_y if extension_direction < 0 else y <= bend_y
        )
        boundary_x = (
            tilted_slope * y + tilted_intercept
            if before_bend
            else held_slope * y + held_intercept
        )
        opposite_x = opposite_slope * y + opposite_intercept
        x0 = max(clip_x0, int(math.ceil(min(boundary_x, opposite_x))))
        x1 = min(
            clip_x1 - 1,
            int(math.floor(max(boundary_x, opposite_x))),
        )
        if x1 < x0:
            continue
        row_fill = np.zeros(original.shape[1], dtype=bool)
        row_fill[x0 : x1 + 1] = True
        row_fill &= ~held[y]
        if np.any(row_fill):
            corrected[y] |= row_fill
            added_rows += 1

    added = corrected & ~original
    added_area = int(np.count_nonzero(added))
    if added_area == 0:
        info.update(
            {
                "using_hold_book_side_line": True,
                "bend_point": [float(bend_x), float(bend_y)],
                "reason": "piecewise boundary added no non-held pixels",
            }
        )
        return original.copy(), info

    info.update(
        {
            "mask_correction_applied": True,
            "using_hold_book_side_line": True,
            "bend_point": [float(bend_x), float(bend_y)],
            "reason": "triangular selected space with held-book lower occlusion",
            "tilted_boundary_side": tilted_side,
            "hold_book_space_facing_side": held_facing_side,
            "hold_book_side_line_x_per_y": float(held_slope),
            "hold_book_side_line_intercept_px": float(held_intercept),
            "hold_book_side_line_raster_clearance_px": 1,
            "hold_book_side_line_method": (
                "robust fitted held-book boundary shifted to the space-side "
                "outer envelope of actual mask rows"
            ),
            "visible_tilt_line_x_per_y": float(tilted_slope),
            "visible_tilt_line_intercept_px": float(tilted_intercept),
            "opposite_boundary_side": opposite_side,
            "opposite_boundary_line_x_per_y": float(opposite_slope),
            "opposite_boundary_line_intercept_px": float(opposite_intercept),
            "visible_mask_area_px": int(np.count_nonzero(original)),
            "corrected_mask_area_px": int(np.count_nonzero(corrected)),
            "added_area_px": added_area,
            "extension_y_range": [
                min(extension_start_y, extension_end_y),
                max(extension_start_y, extension_end_y),
            ],
            "extension_image_y_direction": extension_direction,
            "correction_clip_roi_xyxy": [
                clip_x0,
                clip_y0,
                clip_x1,
                clip_y1,
            ],
            "correction_clipped_to_recognition_roi": clip_roi is not None,
            "extension_row_count": added_rows,
            "visible_maximum_row_width_px": maximum_visible_width,
            "maximum_added_row_width_px": int(
                max(
                    np.count_nonzero(added[y])
                    for y in np.flatnonzero(added.any(axis=1))
                )
            ),
            "visible_tilt_preserved": bool(np.all(corrected[original])),
            "bend_count": 1,
        }
    )
    return corrected, info


def load_saved_capture_rgb(shot_dir: str | Path) -> np.ndarray:
    """Load either the current lossless NPY RGB or a legacy input PNG."""
    capture_dir = Path(shot_dir)
    for name in ("input_rgb.npy", "input.png"):
        path = capture_dir / name
        if path.is_file():
            return storage_recognition.read_rgb(path)
    raise FileNotFoundError(f"saved capture RGB is missing: {capture_dir}")


def load_saved_selected_space_mask(shot_dir: str | Path) -> np.ndarray:
    """Load the final selected mask from current bundles or a legacy PNG."""
    capture_dir = Path(shot_dir)
    correction_path = capture_dir / "triangular_space_masks.npz"
    if correction_path.is_file():
        with np.load(correction_path, allow_pickle=False) as archive:
            return np.asarray(archive["selected_space_mask_after"], dtype=bool)
    masks_path = capture_dir / "storage_masks.npz"
    if masks_path.is_file():
        with np.load(masks_path, allow_pickle=False) as archive:
            return np.asarray(archive["selected_space_mask"], dtype=bool)
    legacy_path = capture_dir / "final_space_mask.png"
    legacy = cv2.imread(str(legacy_path), cv2.IMREAD_GRAYSCALE)
    if legacy is None:
        raise FileNotFoundError(f"saved selected-space mask is missing: {capture_dir}")
    return legacy > 0


def _save_triangular_correction_status(
    result_metadata: dict[str, Any],
    correction: dict[str, Any],
) -> None:
    """Persist the applied/not-applied correction decision without images."""
    shot_dir_value = result_metadata.get("shot_dir")
    if not shot_dir_value:
        return
    shot_dir = Path(shot_dir_value)
    storage_recognition.write_json(
        shot_dir / "triangular_space_mask_correction.json",
        correction,
    )
    geometry_metadata_path = shot_dir / "geometry_metadata.json"
    if geometry_metadata_path.is_file():
        geometry_metadata = json.loads(
            geometry_metadata_path.read_text(encoding="utf-8")
        )
        geometry_metadata["triangular_space_mask_correction"] = correction
        storage_recognition.write_json(geometry_metadata_path, geometry_metadata)


def apply_selected_triangular_space_mask_correction(
    result_metadata: dict[str, Any],
    first_target_cam: np.ndarray,
    *,
    space_type: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply and save the post-classification selected-mask correction."""
    print("\n[TRIANGULAR SPACE MASK CORRECTION]")
    print(f"space_type = {space_type}")
    if space_type not in {
        "triangular_left_tilted",
        "triangular_right_tilted",
    }:
        print("mask correction applied = False")
        print("reason = non-triangular space")
        updated = dict(result_metadata)
        correction = {
            "space_type": space_type,
            "mask_correction_applied": False,
            "reason": "non-triangular space",
        }
        updated["triangular_space_mask_correction"] = correction
        _save_triangular_correction_status(updated, correction)
        return (
            np.asarray(first_target_cam, dtype=np.float64).reshape(3),
            updated,
        )

    updated = dict(result_metadata)
    selected_mask = np.asarray(updated.get("selected_space_mask"), dtype=bool)
    shot_dir_value = updated.get("shot_dir")
    held_indices = [
        int(value) for value in updated.get("held_book_spine_indices", [])
    ]
    if selected_mask.ndim != 2 or not shot_dir_value or not held_indices:
        reason = "selected mask, shot_dir, or held-book index is unavailable"
        print("mask correction applied = False")
        print(f"reason = {reason}")
        updated["triangular_space_mask_correction"] = {
            "space_type": space_type,
            "mask_correction_applied": False,
            "reason": reason,
        }
        _save_triangular_correction_status(
            updated,
            updated["triangular_space_mask_correction"],
        )
        return np.asarray(first_target_cam, dtype=np.float64).reshape(3), updated

    shot_dir = Path(shot_dir_value)
    spine_path = shot_dir / storage_recognition.SPINE_MODEL.npz_name
    with np.load(spine_path, allow_pickle=False) as archive:
        spine_masks = np.asarray(archive["masks"], dtype=bool)
    valid_held_indices = [
        index for index in held_indices if 0 <= index < spine_masks.shape[0]
    ]
    held_mask = (
        np.any(spine_masks[valid_held_indices], axis=0)
        if valid_held_indices
        else np.zeros_like(selected_mask)
    )
    corrected_mask, correction = build_corrected_triangular_space_mask(
        selected_mask,
        held_mask,
        space_type=space_type,
        tilted_line_p0=tuple(updated["line_p0"]),
        tilted_line_p1=tuple(updated["line_p1"]),
        clip_roi=updated.get("roi"),
    )
    print(
        "mask correction applied = "
        f"{correction['mask_correction_applied']}"
    )
    print(
        "using hold-book side line = "
        f"{correction['using_hold_book_side_line']}"
    )
    bend = correction.get("bend_point")
    print(
        "bend point = "
        + ("n/a" if bend is None else f"({bend[0]:.3f}, {bend[1]:.3f})")
    )
    if not correction["mask_correction_applied"]:
        print(f"reason = {correction['reason']}")
        print("target point uses corrected mask = False")
        updated["triangular_space_mask_correction"] = correction
        _save_triangular_correction_status(updated, correction)
        return np.asarray(first_target_cam, dtype=np.float64).reshape(3), updated

    target_value = updated.get("first_target_px_selected")
    if target_value is None or len(target_value) != 2:
        raise RuntimeError("selected target pixel is unavailable for corrected mask")
    visible_y = np.flatnonzero(selected_mask.any(axis=1))
    corrected_y_rows = np.flatnonzero(corrected_mask.any(axis=1))
    target_y_offset = int(
        updated.get(
            "target_y_from_space_top_px",
            int(round(float(target_value[1]))) - int(visible_y.min()),
        )
    )
    target_y = int(corrected_y_rows.min()) + target_y_offset
    if target_y > int(corrected_y_rows.max()):
        raise RuntimeError(
            "corrected mask is too short for the existing target-y rule: "
            f"height={corrected_y_rows.size}, offset={target_y_offset}"
        )
    target_row_x = np.flatnonzero(corrected_mask[target_y])
    if target_row_x.size == 0:
        raise RuntimeError("corrected mask has no pixels at the target row")
    before_bend = (
        target_y >= float(correction["bend_point"][1])
        if int(correction["extension_image_y_direction"]) < 0
        else target_y <= float(correction["bend_point"][1])
    )
    if before_bend:
        target_boundary_x = (
            float(correction["visible_tilt_line_x_per_y"]) * target_y
            + float(correction["visible_tilt_line_intercept_px"])
        )
    else:
        target_boundary_x = (
            float(correction["hold_book_side_line_x_per_y"]) * target_y
            + float(correction["hold_book_side_line_intercept_px"])
        )
    target_x = int(
        target_row_x[np.argmin(np.abs(target_row_x - target_boundary_x))]
    )

    intrinsics_path = shot_dir / "camera_intrinsics.json"
    intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    target_cam = np.asarray(first_target_cam, dtype=np.float64).reshape(3).copy()
    target_cam[0] = (
        (target_x - float(intrinsics["ppx"]))
        / float(intrinsics["fx"])
        * target_cam[2]
    )
    target_cam[1] = (
        (target_y - float(intrinsics["ppy"]))
        / float(intrinsics["fy"])
        * target_cam[2]
    )
    updated["selected_space_mask"] = corrected_mask
    updated["space_bbox"] = _selected_mask_bbox(corrected_mask)
    corrected_y, corrected_x = np.where(corrected_mask)
    updated["space_center_px"] = (
        float(np.mean(corrected_x)),
        float(np.mean(corrected_y)),
    )
    updated["space_area"] = int(np.count_nonzero(corrected_mask))
    updated["first_target_px"] = (target_x, target_y)
    updated["first_target_px_selected"] = (target_x, target_y)
    updated["first_target_px_projected"] = (target_x, target_y)
    updated["target_y_reference"] = (
        "image_top_of_corrected_selected_space_equals_physical_book_bottom_side"
    )
    updated["first_target_cam_raw"] = target_cam.astype(np.float32)
    updated["first_target_cam"] = target_cam.astype(np.float32)
    updated["final_target_cam"] = target_cam.astype(np.float32)
    correction["target_point_uses_corrected_mask"] = True
    correction["corrected_target_pixel"] = [target_x, target_y]
    correction["target_y_recomputed_from_corrected_mask"] = True
    correction["target_y_from_space_top_px"] = target_y_offset
    correction["space_type_classification_preserved"] = True
    correction["visible_space_center_px"] = [
        float(value) for value in result_metadata.get("space_center_px", ())
    ]
    correction["corrected_space_center_px"] = [
        float(np.mean(corrected_x)),
        float(np.mean(corrected_y)),
    ]
    updated["triangular_space_mask_correction"] = correction

    image_rgb = load_saved_capture_rgb(shot_dir)
    if image_rgb.shape[:2] != corrected_mask.shape:
        raise RuntimeError(
            "correction visualization input shape mismatch: "
            f"rgb={image_rgb.shape[:2]} mask={corrected_mask.shape}"
        )
    correction_masks_path = shot_dir / "triangular_space_masks.npz"
    np.savez_compressed(
        correction_masks_path,
        selected_space_mask_before=selected_mask,
        selected_space_mask_after=corrected_mask,
    )
    correction["visible_space_bbox_xywh"] = list(_selected_mask_bbox(selected_mask))
    correction["corrected_space_bbox_xywh"] = list(
        _selected_mask_bbox(corrected_mask)
    )
    correction["visible_target_pixel"] = [
        int(round(float(target_value[0]))),
        int(round(float(target_value[1]))),
    ]
    correction["visible_target_camera_m"] = (
        np.asarray(first_target_cam, dtype=np.float64).reshape(3).tolist()
    )
    correction["corrected_target_camera_m"] = target_cam.tolist()
    correction["mask_npz_path"] = str(correction_masks_path)
    correction["mask_npz_keys"] = [
        "selected_space_mask_before",
        "selected_space_mask_after",
    ]
    final_overlay = image_rgb.copy()
    final_color = np.asarray(
        storage_recognition.FINAL_SPACE_COLOR_RGB,
        dtype=np.float32,
    )
    final_overlay[corrected_mask] = (
        (1.0 - storage_recognition.SPACE_ALPHA)
        * final_overlay[corrected_mask].astype(np.float32)
        + storage_recognition.SPACE_ALPHA * final_color
    ).astype(np.uint8)
    roi_value = updated.get("roi")
    if roi_value is not None and len(roi_value) == 4:
        roi_x0, roi_y0, roi_x1, roi_y1 = (
            int(value) for value in roi_value
        )
        cv2.rectangle(
            final_overlay,
            (roi_x0, roi_y0),
            (roi_x1 - 1, roi_y1 - 1),
            tuple(int(value) for value in storage_recognition.ROI_COLOR_RGB),
            2,
        )
    storage_recognition.save_png(
        shot_dir / "final_space_overlay.png",
        final_overlay,
    )
    storage_recognition.write_json(
        shot_dir / "triangular_space_mask_correction.json",
        correction,
    )
    updated["triangular_space_masks_npz_path"] = str(correction_masks_path)
    geometry_metadata_path = shot_dir / "geometry_metadata.json"
    if geometry_metadata_path.is_file():
        geometry_metadata = json.loads(
            geometry_metadata_path.read_text(encoding="utf-8")
        )
        geometry_metadata.update(
            {
                "space_bbox": list(updated["space_bbox"]),
                "space_center_px": list(updated["space_center_px"]),
                "space_area": int(updated["space_area"]),
                "first_target_px": [target_x, target_y],
                "first_target_px_selected": [target_x, target_y],
                "first_target_px_projected": [target_x, target_y],
                "target_y_reference": updated["target_y_reference"],
                "first_target_cam_raw": target_cam.tolist(),
                "first_target_cam": target_cam.tolist(),
                "final_target_cam": target_cam.tolist(),
                "triangular_space_mask_correction": correction,
                "triangular_space_masks_npz_path": str(correction_masks_path),
            }
        )
        storage_recognition.write_json(
            geometry_metadata_path,
            geometry_metadata,
        )
    print("target point uses corrected mask = True")
    return target_cam, updated


def extract_real_world_left_neighbor_inner_bottom(
    recognition: storage_recognition.StorageRecognitionResult,
    spine_instances: InstanceSet,
    *,
    fit_rows: int = LEFT_NEIGHBOR_BOTTOM_CORNER_FIT_ROWS,
) -> tuple[dict[str, Any], np.ndarray]:
    """Locate B on the exact book-spine paired with the selected space.

    Image-right is real-world left in the current camera arrangement.  The
    selected space's right obstacle is therefore the real-world LEFT neighbor,
    and the obstacle's image-left boundary faces the selected space.  Physical
    bottom corresponds to the first (smallest-y) mask rows.  A Theil-Sen line
    fit over those rows makes the boundary x robust to a single noisy pixel;
    no bbox corner or nearest-center fallback is used.
    """
    selected_space = recognition.selected_space or {}
    pair_id_value = selected_space.get("obstacle_pair_id")
    if pair_id_value is None:
        raise RuntimeError("selected space has no obstacle_pair_id")
    pair_id = int(pair_id_value)
    pair = next(
        (
            item
            for item in recognition.obstacle_pairs
            if int(item.get("pair_id", -1)) == pair_id
        ),
        None,
    )
    if pair is None:
        raise RuntimeError(f"selected obstacle pair was not found: pair_id={pair_id}")
    if pair.get("right_obstacle_type") != "book_spine":
        raise RuntimeError(
            "selected space image-right obstacle is not a book_spine: "
            f"pair_id={pair_id}, type={pair.get('right_obstacle_type')}"
        )

    instance_index = int(pair["right_instance_index"])
    masks = np.asarray(spine_instances.masks, dtype=bool)
    if masks.ndim != 3 or not 0 <= instance_index < masks.shape[0]:
        raise RuntimeError(
            "real-world LEFT neighbor book-spine index is invalid: "
            f"index={instance_index}, mask_shape={masks.shape}"
        )
    neighbor_mask = masks[instance_index]
    occupied_rows = np.flatnonzero(neighbor_mask.any(axis=1))
    requested_fit_rows = int(fit_rows)
    if requested_fit_rows < 2:
        raise ValueError(f"fit_rows must be at least 2: {fit_rows}")
    if occupied_rows.size < 2:
        raise RuntimeError(
            "real-world LEFT neighbor mask has fewer than two occupied rows"
        )

    bottom_rows = occupied_rows[: min(requested_fit_rows, occupied_rows.size)]
    boundary_x = np.asarray(
        [int(np.flatnonzero(neighbor_mask[y])[0]) for y in bottom_rows],
        dtype=np.float64,
    )
    row_y = bottom_rows.astype(np.float64)
    slopes = [
        float((boundary_x[j] - boundary_x[i]) / (row_y[j] - row_y[i]))
        for i in range(len(row_y))
        for j in range(i + 1, len(row_y))
        if row_y[j] != row_y[i]
    ]
    if not slopes:
        raise RuntimeError("neighbor bottom-boundary fit has no valid row pairs")
    slope = float(np.median(np.asarray(slopes, dtype=np.float64)))
    intercept = float(np.median(boundary_x - slope * row_y))

    # A middle row of the physical-bottom band avoids assigning the 3D point
    # to a single possibly spurious topmost pixel while remaining within the
    # configured bottom-side band.
    reference_y = int(round(float(np.median(row_y))))
    reference_x = int(round(slope * reference_y + intercept))
    reference_x = int(np.clip(reference_x, 0, neighbor_mask.shape[1] - 1))
    reference_y = int(np.clip(reference_y, 0, neighbor_mask.shape[0] - 1))

    info = {
        "real_world_left_neighbor_book_mask_index": instance_index,
        "real_world_left_neighbor_book_inner_bottom_px": [
            reference_x,
            reference_y,
        ],
        "real_world_left_neighbor_book_reference_method": (
            "selected_space_obstacle_pair_image_right_book_spine; "
            "selected-space-facing image-left mask boundary; Theil-Sen fit "
            "over physical-bottom (smallest image-y) rows"
        ),
        "real_world_left_neighbor_book_obstacle_pair_id": pair_id,
        "real_world_left_neighbor_book_pair_side": "right",
        "real_world_left_neighbor_book_selected_space_facing_image_side": (
            "left"
        ),
        "real_world_left_neighbor_book_physical_bottom_image_side": "top",
        "real_world_left_neighbor_book_bottom_fit_rows_requested": (
            requested_fit_rows
        ),
        "real_world_left_neighbor_book_bottom_fit_rows_used": int(len(row_y)),
        "real_world_left_neighbor_book_bottom_fit_y_range": [
            int(bottom_rows[0]),
            int(bottom_rows[-1]),
        ],
        "real_world_left_neighbor_book_boundary_fit_x_per_y": slope,
        "real_world_left_neighbor_book_boundary_fit_intercept_px": intercept,
    }
    return info, neighbor_mask.copy()


def _save_geometry_debug(
    output_dir: Path,
    rgb: np.ndarray,
    recognition: storage_recognition.StorageRecognitionResult,
    line_p0: tuple[int, int],
    line_p1: tuple[int, int],
    target_px: tuple[int, int],
) -> dict[str, str]:
    base = storage_recognition.draw_selected_space(
        rgb,
        recognition.selected_space,
        recognition.final_labels,
        recognition.roi,
    )
    guide = base.copy()
    cv2.line(guide, line_p0, line_p1, (255, 255, 0), 3, cv2.LINE_AA)
    cv2.circle(guide, line_p0, 5, (0, 255, 255), -1)
    cv2.circle(guide, line_p1, 5, (0, 255, 255), -1)

    target = guide.copy()
    cv2.circle(target, target_px, 8, (255, 0, 255), -1)
    target_path = output_dir / "final_space_target_point.png"
    storage_recognition.save_png(target_path, target)
    return {"final_space_target_point_path": str(target_path)}


def collect_non_held_book_spine_surface_points(
    depth_u16: np.ndarray,
    intr: Any,
    depth_scale: float,
    spine_instances: InstanceSet,
    held_book_spine_indices: set[int],
    *,
    crop_y_min: int,
    crop_y_max: int,
    stride: int = 2,
    z_min_m: float = 0.1,
    z_max_m: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Deproject valid Depth only inside non-held SAM3 spine masks."""
    depth = np.asarray(depth_u16)
    masks = np.asarray(spine_instances.masks, dtype=bool)
    if masks.ndim != 3 or masks.shape[1:] != depth.shape:
        raise ValueError(
            "book spine mask/depth shape mismatch: "
            f"masks={masks.shape} depth={depth.shape}"
        )
    held_indices = sorted(
        index
        for index in held_book_spine_indices
        if 0 <= index < spine_instances.count
    )
    held_set = set(held_indices)
    shelf_indices = [
        index for index in range(spine_instances.count) if index not in held_set
    ]
    source_mask = (
        np.any(masks[shelf_indices], axis=0)
        if shelf_indices
        else np.zeros(depth.shape, dtype=bool)
    )
    held_union = (
        np.any(masks[held_indices], axis=0)
        if held_indices
        else np.zeros(depth.shape, dtype=bool)
    )
    source_mask &= ~held_union
    y0 = int(np.clip(crop_y_min, 0, depth.shape[0]))
    y1 = int(np.clip(crop_y_max, y0, depth.shape[0]))
    y_range_mask = np.zeros(depth.shape, dtype=bool)
    y_range_mask[y0:y1] = True
    sample_mask = np.zeros(depth.shape, dtype=bool)
    step = max(1, int(stride))
    sample_mask[y0:y1:step, ::step] = True
    z = depth.astype(np.float32) * float(depth_scale)
    valid_depth = (
        (depth > 0)
        & (depth < 65535)
        & (z >= float(z_min_m))
        & (z <= float(z_max_m))
    )
    selected = source_mask & y_range_mask & sample_mask & valid_depth
    pixels_y, pixels_x = np.where(selected)
    pixels = np.column_stack((pixels_x, pixels_y)).astype(np.int32)
    rs = _get_realsense_module()
    points = np.empty((len(pixels), 3), dtype=np.float32)
    for index, (u, v) in enumerate(pixels):
        points[index] = rs.rs2_deproject_pixel_to_point(
            intr,
            [float(u), float(v)],
            float(z[v, u]),
        )
    source_depth = depth[source_mask & y_range_mask]
    source_z = source_depth.astype(np.float32) * float(depth_scale)
    stats = {
        "source": "non_held_book_spine_masks",
        "held_book_spine_indices": held_indices,
        "shelf_book_spine_indices": shelf_indices,
        "crop_y_range_half_open": [y0, y1],
        "stride": step,
        "z_min_m": float(z_min_m),
        "z_max_m": float(z_max_m),
        "source_mask_pixel_count": int(np.count_nonzero(source_mask)),
        "held_overlap_removed_pixel_count": int(
            np.count_nonzero(
                held_union
                & (
                    np.any(masks[shelf_indices], axis=0)
                    if shelf_indices
                    else np.zeros(depth.shape, dtype=bool)
                )
            )
        ),
        "sampled_source_pixel_count": int(
            np.count_nonzero(source_mask & y_range_mask & sample_mask)
        ),
        "valid_point_count": int(len(points)),
        "invalid_zero_count_before_stride": int(
            np.count_nonzero(source_depth == 0)
        ),
        "invalid_65535_count_before_stride": int(
            np.count_nonzero(source_depth == 65535)
        ),
        "invalid_out_of_range_count_before_stride": int(
            np.count_nonzero(
                (source_depth > 0)
                & (source_depth < 65535)
                & (
                    (source_z < float(z_min_m))
                    | (source_z > float(z_max_m))
                )
            )
        ),
    }
    return points, pixels, stats


def _point_on_target_ray_at_z(
    target_px: tuple[int, int],
    intr: Any,
    z_m: float,
) -> np.ndarray:
    rs = _get_realsense_module()
    ray = np.asarray(
        rs.rs2_deproject_pixel_to_point(
            intr,
            [float(target_px[0]), float(target_px[1])],
            1.0,
        ),
        dtype=np.float32,
    )
    if abs(float(ray[2])) < 1e-8:
        raise RuntimeError("target pixel ray has zero Z component")
    return (ray * (float(z_m) / float(ray[2]))).astype(np.float32)


TARGET_3D_MODES = frozenset({"ransac_plane", "depth_ref"})
DEPTH_REF_METHOD = "roi_median_0p25m_to_0p5m"
DEPTH_REF_MIN_M = 0.25
DEPTH_REF_MAX_M = 0.5


def _normalize_target_3d_mode(target_3d_mode: str) -> str:
    mode = str(target_3d_mode).strip().lower()
    if mode not in TARGET_3D_MODES:
        raise ValueError(
            "target_3d_mode must be one of "
            f"{sorted(TARGET_3D_MODES)}, got {target_3d_mode!r}"
        )
    return mode


def calculate_depth_ref_from_roi(
    depth_u16: np.ndarray,
    depth_scale: float,
    roi: list[int] | None,
) -> tuple[float, int, list[int]]:
    """Return the median valid Depth in the recognition ROI."""
    depth = np.asarray(depth_u16)
    if depth.ndim != 2:
        raise RuntimeError(
            "geometry calculation failed: depth_ref requires 2D Depth, "
            f"got shape={depth.shape}"
        )
    if roi is None or len(roi) != 4:
        raise RuntimeError(
            "geometry calculation failed: depth_ref ROI is unavailable"
        )

    x1, y1, x2, y2 = (int(value) for value in roi)
    height, width = depth.shape
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise RuntimeError(
            "geometry calculation failed: invalid depth_ref ROI: "
            f"roi={[x1, y1, x2, y2]}, depth_shape={depth.shape}"
        )

    roi_depth_m = (
        depth[y1:y2, x1:x2].astype(np.float32) * float(depth_scale)
    )
    valid = (
        np.isfinite(roi_depth_m)
        & (roi_depth_m > DEPTH_REF_MIN_M)
        & (roi_depth_m < DEPTH_REF_MAX_M)
    )
    valid_depth_m = roi_depth_m[valid]
    valid_pixel_count = int(valid_depth_m.size)
    if valid_pixel_count == 0:
        raise RuntimeError(
            "geometry calculation failed: depth_ref ROI has no valid Depth "
            f"in ({DEPTH_REF_MIN_M}, {DEPTH_REF_MAX_M}) m: "
            f"roi={[x1, y1, x2, y2]}"
        )

    return (
        float(np.median(valid_depth_m)),
        valid_pixel_count,
        [x1, y1, x2, y2],
    )


def calculate_geometry_from_selected_space(
    rgb: np.ndarray,
    depth_u16: np.ndarray,
    intr: rs.intrinsics,
    depth_scale: float,
    recognition: storage_recognition.StorageRecognitionResult,
    *,
    output_dir: str | Path,
    crop_y_min: int = 120,
    crop_y_max: int = 400,
    target_y_from_space_top_px: int = 240,
    delta_depth_m: float = 0.03,
    surface_depth_tol_m: float = 0.05,
    surface_side_margin_px: int = 140,
    surface_gap_px: int = 5,
    surface_stride: int = 2,
    use_plane_correction: bool = True,
    target_3d_mode: str = "ransac_plane",
    plane_ransac_threshold_m: float = 0.01,
    plane_ransac_max_iter: int = 300,
    plane_min_inliers: int = 30,
    surface_clearance_m: float = 0.0,
    inference_metadata: dict[str, Any] | None = None,
    spine_instances: InstanceSet | None = None,
    end_instances: InstanceSet | None = None,
    held_book_config: HeldBookDetectionConfig = DEFAULT_HELD_BOOK_CONFIG,
    timing_sink: dict[str, float] | None = None,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Apply Storage_rev geometry to the immutable selected SAM3 space mask."""
    geometry_total_started = time.perf_counter()
    geometry_tracked_keys = (
        "depth_ref_calculation_ms",
        "ransac_plane_calculation_ms",
        "target_point_calculation_ms",
        "debug_image_generation_ms",
        "png_jpg_save_ms",
        "binary_file_save_ms",
        "json_metadata_save_ms",
    )
    geometry_before = {
        key: timing_sink.get(key, 0.0) if timing_sink is not None else 0.0
        for key in geometry_tracked_keys
    }
    target_3d_mode = _normalize_target_3d_mode(target_3d_mode)
    image = np.asarray(rgb)
    depth = np.asarray(depth_u16)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError(f"expected uint16 depth HxW, got {depth.shape} {depth.dtype}")
    if image.shape[:2] != depth.shape:
        raise ValueError(
            f"RGB/depth shape mismatch: rgb={image.shape[:2]} depth={depth.shape}"
        )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    selected_mask = recognition.selected_space_mask.astype(np.uint8, copy=False)
    selected_mask_before_geometry = selected_mask.copy()
    bbox = _selected_mask_bbox(selected_mask)
    ys, xs = np.where(selected_mask > 0)

    try:
        p0, p1, guide_boundary_points, guide_side, guide_info = (
            legacy_geometry.extract_tilted_boundary_line(
                selected_mask,
                default_side="left",
            )
        )
    except Exception as exc:
        raise RuntimeError(f"geometry calculation failed: guide boundary: {exc}") from exc
    line_p0 = (int(round(float(p0[0]))), int(round(float(p0[1]))))
    line_p1 = (int(round(float(p1[0]))), int(round(float(p1[1]))))
    dx = float(line_p1[0] - line_p0[0])
    dy = float(line_p1[1] - line_p0[1])
    angle_rad = float(np.arctan2(dy, dx))
    angle_deg = float(np.degrees(angle_rad))

    space_image_top_y = int(ys.min())
    space_image_bottom_y = int(ys.max())
    space_height_px = space_image_bottom_y - space_image_top_y + 1
    if space_height_px <= int(target_y_from_space_top_px):
        raise RuntimeError(
            "geometry calculation failed: target位置まで高さ不足: "
            f"height={space_height_px}px, "
            f"required>{target_y_from_space_top_px}px"
        )
    target_y_px = space_image_top_y + int(target_y_from_space_top_px)
    target_started = time.perf_counter()
    target_px_selected = legacy_geometry.select_target_pixel_on_line_by_y(
        line_p0,
        line_p1,
        target_y_px=target_y_px,
        width=image.shape[1],
        height=image.shape[0],
    )
    _add_timing_ms(
        timing_sink,
        "target_point_calculation_ms",
        time.perf_counter() - target_started,
    )
    held_detection = (inference_metadata or {}).get(
        "held_book_detection",
        {},
    )
    if held_detection:
        held_book_spine_indices = {
            int(index)
            for index in held_detection.get("held_book_spine_indices", [])
        }
        held_book_detection_records = held_detection.get("instances", [])
    elif spine_instances is not None:
        held_book_detection_records = detect_held_book_spines(
            spine_instances,
            held_book_config,
        )
        held_book_spine_indices = {
            int(item["instance_index"])
            for item in held_book_detection_records
            if item.get("is_held_book")
        }
    else:
        held_book_detection_records = []
        held_book_spine_indices = set()

    surface_points = np.zeros((0, 3), dtype=np.float32)
    surface_pixels = np.zeros((0, 2), dtype=np.int32)
    ransac_point_stats: dict[str, Any] = {
        "source": "unavailable_without_spine_instances",
        "valid_point_count": 0,
    }
    if spine_instances is not None:
        surface_points, surface_pixels, ransac_point_stats = (
            collect_non_held_book_spine_surface_points(
                depth,
                intr,
                depth_scale,
                spine_instances,
                held_book_spine_indices,
                crop_y_min=crop_y_min,
                crop_y_max=crop_y_max,
                stride=surface_stride,
                z_min_m=0.1,
                z_max_m=1.5,
            )
        )
    depth_ref_started = time.perf_counter()
    (
        depth_ref_m,
        depth_ref_valid_pixel_count,
        depth_ref_roi,
    ) = calculate_depth_ref_from_roi(
        depth,
        depth_scale,
        recognition.roi,
    )
    _add_timing_ms(
        timing_sink,
        "depth_ref_calculation_ms",
        time.perf_counter() - depth_ref_started,
    )
    print(
        "[STORAGE DEPTH REF] "
        "method=roi_median, range=(0.25,0.5)m, "
        f"depth={depth_ref_m:.6f}m, "
        f"valid_pixels={depth_ref_valid_pixel_count}"
    )
    plane = None
    inlier_mask = np.zeros((surface_points.shape[0],), dtype=bool)
    plane_correction_used = False
    first_target_cam_raw: np.ndarray | None = None
    first_target_cam: np.ndarray | None = None
    target_3d_source: str | None = None
    fallback_used = False
    fallback_reason: str | None = None
    if use_plane_correction and surface_points.shape[0] >= 3:
        ransac_started = time.perf_counter()
        plane, inlier_mask = legacy_geometry.fit_plane_ransac(
            surface_points,
            distance_threshold_m=plane_ransac_threshold_m,
            max_iter=plane_ransac_max_iter,
            min_inliers=plane_min_inliers,
            random_seed=0,
        )
        _add_timing_ms(
            timing_sink,
            "ransac_plane_calculation_ms",
            time.perf_counter() - ransac_started,
        )
    target_started = time.perf_counter()
    if target_3d_mode == "depth_ref":
        if (
            depth_ref_m is None
            or not math.isfinite(depth_ref_m)
            or depth_ref_m <= 0.0
        ):
            raise RuntimeError(
                "geometry calculation failed: target_3d_mode=depth_ref "
                f"requires finite positive depth_ref_m, got {depth_ref_m!r}"
            )
        first_target_cam_raw = _point_on_target_ray_at_z(
            target_px_selected,
            intr,
            depth_ref_m,
        )
        first_target_cam = first_target_cam_raw.copy()
        target_3d_source = "depth_ref"
    else:
        if plane is not None:
            try:
                first_target_cam_raw = (
                    legacy_geometry.intersect_pixel_ray_with_plane(
                        target_px_selected,
                        intr,
                        plane,
                        surface_clearance_m=0.0,
                    )
                )
                first_target_cam = legacy_geometry.intersect_pixel_ray_with_plane(
                    target_px_selected,
                    intr,
                    plane,
                    surface_clearance_m=surface_clearance_m,
                )
                plane_correction_used = True
                target_3d_source = "non_held_book_spine_ransac_plane"
            except Exception as exc:
                fallback_reason = f"book_spine_plane_intersection_failed:{exc}"
        elif use_plane_correction and surface_points.shape[0] >= 3:
            fallback_reason = "book_spine_plane_ransac_failed"
        elif not use_plane_correction:
            fallback_reason = "book_spine_plane_disabled"
        else:
            fallback_reason = "insufficient_book_spine_depth_points_for_ransac"

        if first_target_cam is None:
            fallback_used = True
            try:
                target_depth_point = legacy_geometry.deproject_single_pixel_to_3d(
                    target_px_selected,
                    depth,
                    intr,
                    depth_scale,
                    search_radius_px=5,
                    z_min_m=0.1,
                    z_max_m=1.5,
                )
                first_target_cam_raw = target_depth_point.copy()
                if len(surface_points):
                    first_target_cam = (
                        legacy_geometry.fallback_correct_point_z_by_median_surface(
                            target_depth_point,
                            surface_points,
                            surface_clearance_m=surface_clearance_m,
                        )
                    )
                    target_3d_source = (
                        "target_depth_neighborhood_with_book_spine_median_z"
                    )
                else:
                    first_target_cam = target_depth_point
                    target_3d_source = "target_depth_neighborhood"
            except Exception as target_depth_exc:
                if len(surface_points):
                    median_z = float(np.median(surface_points[:, 2]))
                    first_target_cam_raw = _point_on_target_ray_at_z(
                        target_px_selected,
                        intr,
                        median_z,
                    )
                    first_target_cam = first_target_cam_raw.copy()
                    first_target_cam[2] -= float(surface_clearance_m)
                    target_3d_source = "non_held_book_spine_median_depth"
                    fallback_reason = (
                        f"{fallback_reason};target_depth_invalid:{target_depth_exc}"
                    )
                else:
                    raise RuntimeError(
                        "geometry calculation failed: no valid non-held book-spine "
                        "Depth plane/representative Depth and target Depth is invalid: "
                        f"{target_depth_exc}"
                    ) from target_depth_exc

    _add_timing_ms(
        timing_sink,
        "target_point_calculation_ms",
        time.perf_counter() - target_started,
    )

    assert first_target_cam is not None
    assert first_target_cam_raw is not None

    real_world_left_neighbor_info: dict[str, Any] = {
        "real_world_left_neighbor_book_mask_index": None,
        "real_world_left_neighbor_book_inner_bottom_px": None,
        "real_world_left_neighbor_book_inner_bottom_cam_m": None,
        "real_world_left_neighbor_book_reference_method": (
            "unavailable_until_selected-pair book mask and RANSAC plane exist"
        ),
        "real_world_left_neighbor_book_geometry_error": None,
    }
    real_world_left_neighbor_mask: np.ndarray | None = None
    if spine_instances is not None:
        try:
            (
                real_world_left_neighbor_info,
                real_world_left_neighbor_mask,
            ) = extract_real_world_left_neighbor_inner_bottom(
                recognition,
                spine_instances,
            )
            real_world_left_neighbor_info[
                "real_world_left_neighbor_book_inner_bottom_cam_m"
            ] = None
            real_world_left_neighbor_info[
                "real_world_left_neighbor_book_geometry_error"
            ] = None
            if plane is None:
                raise RuntimeError(
                    "RANSAC shelf-front plane is unavailable for B 3D projection"
                )
            bottom_px_value = real_world_left_neighbor_info[
                "real_world_left_neighbor_book_inner_bottom_px"
            ]
            bottom_px = (int(bottom_px_value[0]), int(bottom_px_value[1]))
            bottom_cam = legacy_geometry.intersect_pixel_ray_with_plane(
                bottom_px,
                intr,
                plane,
                surface_clearance_m=0.0,
            )
            real_world_left_neighbor_info[
                "real_world_left_neighbor_book_inner_bottom_cam_m"
            ] = bottom_cam.astype(np.float64).tolist()
            real_world_left_neighbor_info[
                "real_world_left_neighbor_book_3d_method"
            ] = "B pixel ray intersection with non-held-book RANSAC shelf-front plane"
        except Exception as exc:
            real_world_left_neighbor_info[
                "real_world_left_neighbor_book_geometry_error"
            ] = f"{type(exc).__name__}: {exc}"

    try:
        target_px_projected = legacy_geometry.project_cam_point_to_pixel(
            first_target_cam,
            intr,
        )
    except Exception:
        target_px_projected = target_px_selected

    guide_length_plane = (
        plane
        if target_3d_mode == "depth_ref"
        else plane if plane_correction_used else None
    )
    guide_length_plane_source = (
        "book_spine_ransac_plane" if guide_length_plane is not None else None
    )
    if guide_length_plane is None and len(surface_points):
        representative_z = float(np.median(surface_points[:, 2]))
        guide_length_plane = np.asarray(
            [0.0, 0.0, 1.0, -representative_z],
            dtype=np.float32,
        )
        guide_length_plane_source = "book_spine_median_depth_plane"
    try:
        (
            guide_edge_length_m,
            guide_edge_p0_cam,
            guide_edge_p1_cam,
            guide_edge_length_source,
        ) = legacy_geometry.compute_guide_edge_length_3d(
            line_p0,
            line_p1,
            intr,
            depth,
            depth_scale,
            plane=guide_length_plane,
            search_radius_px=5,
        )
    except Exception as exc:
        raise RuntimeError(
            f"geometry calculation failed: guide edge length: {exc}"
        ) from exc
    guide_edge_length_mm = float(guide_edge_length_m * 1000.0)

    debug_save_before = (
        timing_sink.get("png_jpg_save_ms", 0.0)
        if timing_sink is not None
        else 0.0
    )
    debug_started = time.perf_counter()
    debug_paths = _save_geometry_debug(
        output_path,
        image,
        recognition,
        line_p0,
        line_p1,
        target_px_projected,
    )
    debug_elapsed_ms = (time.perf_counter() - debug_started) * 1000.0
    debug_save_after = (
        timing_sink.get("png_jpg_save_ms", 0.0)
        if timing_sink is not None
        else debug_save_before
    )
    if timing_sink is not None:
        timing_sink["debug_image_generation_ms"] = (
            timing_sink.get("debug_image_generation_ms", 0.0)
            + max(0.0, debug_elapsed_ms - (debug_save_after - debug_save_before))
        )
    ransac_npz_path = output_path / "book_spine_ransac_points.npz"
    binary_save_started = time.perf_counter()
    np.savez_compressed(
        ransac_npz_path,
        pixels_xy=surface_pixels.astype(np.int32, copy=False),
        points_camera_m=surface_points.astype(np.float32, copy=False),
        inlier_mask=inlier_mask.astype(bool, copy=False),
        plane=(
            plane.astype(np.float32, copy=False)
            if plane is not None
            else np.zeros((0,), dtype=np.float32)
        ),
    )
    _add_timing_ms(
        timing_sink,
        "binary_file_save_ms",
        time.perf_counter() - binary_save_started,
    )
    if not np.array_equal(selected_mask, selected_mask_before_geometry):
        raise AssertionError("geometry processing modified selected_space_mask")

    selected_space = recognition.selected_space or {}
    matched_held_books = [
        item for item in held_book_detection_records if item.get("is_held_book")
    ]
    primary_held_book = matched_held_books[0] if matched_held_books else None
    res: dict[str, Any] = {
        "line_p0": line_p0,
        "line_p1": line_p1,
        "sub_line_p0": line_p0,
        "sub_line_p1": line_p1,
        "right_is_tilted": False,
        "is_right_half": False,
        "pair_indices": None,
        "centers": None,
        "guide_side": guide_side,
        "guide_info": guide_info,
        "guide_boundary_points": guide_boundary_points,
        "left_tilt_from_vertical_deg": guide_info.get(
            "left_tilt_from_vertical_deg"
        ),
        "right_tilt_from_vertical_deg": guide_info.get(
            "right_tilt_from_vertical_deg"
        ),
        "angle_rad": angle_rad,
        "angle_deg": angle_deg,
        "space_bbox": bbox,
        "space_center_px": (float(np.mean(xs)), float(np.mean(ys))),
        "space_area": int(np.count_nonzero(selected_mask)),
        "space_aspect_w_h": float(bbox[2] / max(bbox[3], 1)),
        "first_target_px": target_px_projected,
        "first_target_px_selected": target_px_selected,
        "first_target_px_projected": target_px_projected,
        "first_target_cam_raw": first_target_cam_raw,
        "first_target_cam": first_target_cam,
        "final_target_cam": first_target_cam.copy(),
        "depth_ref_method": DEPTH_REF_METHOD,
        "depth_ref_m": depth_ref_m,
        "depth_ref_valid_pixel_count": depth_ref_valid_pixel_count,
        "depth_ref_roi": depth_ref_roi,
        "target_depth_m": float(first_target_cam[2]),
        "target_3d_mode": target_3d_mode,
        "delta_depth_m": delta_depth_m,
        "crop_y_min": crop_y_min,
        "crop_y_max": crop_y_max,
        "target_y_from_space_top_px": target_y_from_space_top_px,
        "target_y_reference": (
            "image_top_of_selected_space_equals_physical_book_bottom_side"
        ),
        "use_plane_correction": use_plane_correction,
        "plane_correction_used": plane_correction_used,
        "plane": plane.tolist() if plane is not None else None,
        "ransac_source": "non_held_book_spine_masks",
        "ransac_point_count": int(surface_points.shape[0]),
        "ransac_inlier_count": int(np.sum(inlier_mask)),
        "ransac_point_stats": ransac_point_stats,
        "target_3d_source": target_3d_source,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "held_book_spine_indices": sorted(held_book_spine_indices),
        "held_book_spine_index": (
            min(held_book_spine_indices)
            if held_book_spine_indices
            else None
        ),
        "held_book_detection_reason": (
            "x_position_band_with_minimum_height_rule"
            if held_book_spine_indices
            else "no_instance_matched"
        ),
        "held_book_detection": held_book_detection_records,
        "held_book_detection_config": asdict(held_book_config),
        "held_book_bbox": (
            primary_held_book.get("bbox_xyxy") if primary_held_book else None
        ),
        "held_book_height_ratio": (
            primary_held_book.get("height_ratio") if primary_held_book else None
        ),
        "surface_points_count": int(surface_points.shape[0]),
        "surface_inliers_count": int(np.sum(inlier_mask)),
        "surface_clearance_m": surface_clearance_m,
        "guide_edge_length_m": float(guide_edge_length_m),
        "guide_edge_length_mm": guide_edge_length_mm,
        "guide_edge_length_source": guide_edge_length_source,
        "guide_edge_length_plane_source": guide_length_plane_source,
        "guide_edge_p0_cam": guide_edge_p0_cam.tolist(),
        "guide_edge_p1_cam": guide_edge_p1_cam.tolist(),
        "selected_space_id": recognition.selected_space_id,
        "selected_space_mask": recognition.selected_space_mask,
        "book_spine_mask_count": (
            spine_instances.count if spine_instances is not None else None
        ),
        "book_end_mask_count": (
            end_instances.count if end_instances is not None else None
        ),
        "roi": recognition.roi,
        "residual": recognition.residual,
        "space_candidate_count": len(recognition.spaces),
        "space_candidate_metadata": recognition.spaces,
        "image_top_width_px": selected_space.get("image_top_width_px"),
        "image_bottom_width_px": selected_space.get("image_bottom_width_px"),
        "book_bottom_side_width_px": selected_space.get(
            "book_bottom_side_width_px"
        ),
        "book_top_side_width_px": selected_space.get("book_top_side_width_px"),
        **real_world_left_neighbor_info,
        "real_world_left_neighbor_book_mask": real_world_left_neighbor_mask,
        "sam3_inference_metadata": inference_metadata,
        "shot_dir": str(output_path),
        "book_spine_ransac_points_npz_path": str(ransac_npz_path),
        **debug_paths,
    }
    storage_recognition.write_json(
        output_path / "geometry_metadata.json",
        {
            key: value
            for key, value in res.items()
            if key
            not in {
                "selected_space_mask",
                "residual",
                "space_candidate_metadata",
                "guide_boundary_points",
                "real_world_left_neighbor_book_mask",
            }
        },
    )
    geometry_total_ms = (time.perf_counter() - geometry_total_started) * 1000.0
    geometry_tracked_delta_ms = sum(
        (
            timing_sink.get(key, 0.0) - geometry_before[key]
            if timing_sink is not None
            else 0.0
        )
        for key in geometry_tracked_keys
    )
    if timing_sink is not None:
        timing_sink["geometry_calculation_ms"] = (
            timing_sink.get("geometry_calculation_ms", 0.0)
            + max(0.0, geometry_total_ms - geometry_tracked_delta_ms)
        )
        res["recognition_timing_ms"] = dict(timing_sink)
    return angle_rad, first_target_cam, res


def _capture_rgbd_once(
    *,
    width: int,
    height: int,
    fps: int,
    shot_dir: Path,
    camera_session: Any = None,
    timing_sink: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, rs.intrinsics, float, dict[str, Any] | None]:
    if camera_session is not None:
        if not camera_session.is_started:
            raise RuntimeError(
                "persistent RealSense session is not started"
            )
        requested_stream = (int(width), int(height), int(fps))
        session_stream = (
            int(camera_session.width),
            int(camera_session.height),
            int(camera_session.fps),
        )
        if session_stream != requested_stream:
            raise RuntimeError(
                "persistent RealSense stream does not match storage capture: "
                f"session={session_stream}, requested={requested_stream}"
            )

        # The retrieval path retries the whole capture/recognition request up
        # to three times while keeping the same pipeline active.  Storage uses
        # the same count here, limited to frame acquisition so 1st/2nd/3rd
        # viewpoint selection semantics remain unchanged.
        max_capture_attempts = 3
        for capture_attempt in range(1, max_capture_attempts + 1):
            try:
                rgb_bgr, depth_u16, intr, depth_scale, metadata = (
                    camera_session.capture(
                        shot_dir,
                        depth_filter=legacy_geometry.depth_filter_like_viewer,
                        save_capture_artifacts=False,
                    )
                )
                return (
                    np.asarray(rgb_bgr, dtype=np.uint8).copy(),
                    np.asarray(depth_u16, dtype=np.uint16).copy(),
                    intr,
                    float(depth_scale),
                    metadata,
                )
            except RuntimeError as exc:
                if capture_attempt >= max_capture_attempts:
                    raise
                print(
                    "[CAMERA] storage frame acquisition failed; retrying "
                    "with the active pipeline: "
                    f"attempt={capture_attempt}/{max_capture_attempts}, "
                    f"error={exc}"
                )

    rs = _get_realsense_module()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    camera_inclusive_started = time.perf_counter()
    if timing_sink is not None:
        timing_sink["_camera_inclusive_started_at_perf_counter"] = (
            camera_inclusive_started
        )
    try:
        pipeline_start_started = time.perf_counter()
        profile = pipeline.start(config)
        if timing_sink is not None:
            timing_sink["pipeline_start_time_s"] = (
                time.perf_counter() - pipeline_start_started
            )
        align = rs.align(rs.stream.color)
        warmup_started = time.perf_counter()
        for _ in range(30):
            pipeline.wait_for_frames()
        if timing_sink is not None:
            timing_sink["warmup_time_s"] = (
                time.perf_counter() - warmup_started
            )
        capture_started = time.perf_counter()
        aligned = align.process(pipeline.wait_for_frames())
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense frame取得に失敗しました．")
        filtered_depth = legacy_geometry.depth_filter_like_viewer(depth_frame)
        rgb_bgr = np.asanyarray(color_frame.get_data()).copy()
        depth_u16 = np.asanyarray(filtered_depth.get_data()).copy()
        color_profile = rs.video_stream_profile(
            profile.get_stream(rs.stream.color)
        )
        intr = color_profile.get_intrinsics()
        depth_scale = float(
            profile.get_device().first_depth_sensor().get_depth_scale()
        )
        if timing_sink is not None:
            timing_sink["capture_time_s"] = (
                time.perf_counter() - capture_started
            )
    finally:
        pipeline_stop_started = time.perf_counter()
        try:
            pipeline.stop()
        except Exception:
            pass
        finally:
            if timing_sink is not None:
                timing_sink["pipeline_stop_time_s"] = (
                    time.perf_counter() - pipeline_stop_started
                )
    return rgb_bgr, depth_u16, intr, depth_scale, None


def _apply_legacy_depth_ignore(depth_u16: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_u16).copy()
    width = depth.shape[1]
    depth[:, width // 2 : 3 * width // 4] = 0
    return depth


def run_from_rgbd(
    rgb: np.ndarray,
    depth_u16: np.ndarray,
    intr: rs.intrinsics,
    depth_scale: float,
    *,
    shot_dir: str | Path,
    runtime: StorageSam3Runtime | None = None,
    spine_instances: InstanceSet | None = None,
    end_instances: InstanceSet | None = None,
    apply_legacy_depth_ignore: bool = True,
    held_book_config: HeldBookDetectionConfig = DEFAULT_HELD_BOOK_CONFIG,
    crop_y_min: int = 120,
    crop_y_max: int = 400,
    target_y_from_space_top_px: int = 240,
    delta_depth_m: float = 0.03,
    surface_depth_tol_m: float = 0.05,
    surface_side_margin_px: int = 140,
    surface_gap_px: int = 5,
    surface_stride: int = 2,
    use_plane_correction: bool = True,
    target_3d_mode: str = "ransac_plane",
    plane_ransac_threshold_m: float = 0.01,
    plane_ransac_max_iter: int = 300,
    plane_min_inliers: int = 30,
    surface_clearance_m: float = 0.0,
    timing_sink: dict[str, float] | None = None,
    selected_space_mask_override: np.ndarray | None = None,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Offline-safe entry point for one already captured and aligned RGB-D set."""
    timings = timing_sink if timing_sink is not None else {}
    recognition_total_started = time.perf_counter()
    result: tuple[float, np.ndarray, dict[str, Any]] | None = None
    try:
        with storage_recognition.collect_recognition_file_timings(timings):
            target_3d_mode = _normalize_target_3d_mode(target_3d_mode)
            output_dir = Path(shot_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            image = np.asarray(rgb, dtype=np.uint8)
            depth_raw = np.asarray(depth_u16, dtype=np.uint16)
            depth = depth_raw.copy()
            if apply_legacy_depth_ignore:
                depth = _apply_legacy_depth_ignore(depth)
            binary_save_started = time.perf_counter()
            np.save(
                output_dir / "input_rgb.npy",
                image,
                allow_pickle=False,
            )
            np.save(
                output_dir / "depth_u16_raw.npy",
                depth_raw,
                allow_pickle=False,
            )
            np.save(
                output_dir / "depth_u16.npy",
                depth,
                allow_pickle=False,
            )
            _add_timing_ms(
                timings,
                "binary_file_save_ms",
                time.perf_counter() - binary_save_started,
            )
            storage_recognition.write_json(
                output_dir / "camera_intrinsics.json",
                {
                    "width": int(intr.width),
                    "height": int(intr.height),
                    "fx": float(intr.fx),
                    "fy": float(intr.fy),
                    "ppx": float(intr.ppx),
                    "ppy": float(intr.ppy),
                    "model": str(intr.model),
                    "coeffs": [float(value) for value in intr.coeffs],
                    "depth_scale": float(depth_scale),
                    "legacy_depth_ignore_applied": bool(
                        apply_legacy_depth_ignore
                    ),
                },
            )
            recognition, spine, ends, inference_metadata = recognize_storage_space(
                image,
                runtime=runtime,
                spine_instances=spine_instances,
                end_instances=end_instances,
                output_dir=output_dir,
                held_book_config=held_book_config,
                timing_sink=timings,
            )
            original_prediction_mask = np.asarray(
                recognition.selected_space_mask, dtype=bool
            )
            storage_recognition.save_png(
                output_dir / "original_prediction_mask.png",
                original_prediction_mask.astype(np.uint8) * 255,
            )
            if selected_space_mask_override is not None:
                override = np.asarray(selected_space_mask_override, dtype=bool)
                if override.shape != original_prediction_mask.shape:
                    raise ValueError(
                        "selected_space_mask_override shape mismatch: "
                        f"{override.shape} / {original_prediction_mask.shape}"
                    )
                if not np.any(override):
                    raise ValueError("selected_space_mask_override is empty")
                recognition = replace(
                    recognition,
                    selected_space_mask=override.copy(),
                )
                storage_recognition.save_png(
                    output_dir / "artificial_mask_used_for_geometry.png",
                    override.astype(np.uint8) * 255,
                )
                inference_metadata = dict(inference_metadata)
                inference_metadata["roc_artificial_mask_override"] = {
                    "enabled": True,
                    "original_area_px": int(original_prediction_mask.sum()),
                    "artificial_area_px": int(override.sum()),
                }
            result = calculate_geometry_from_selected_space(
                image,
                depth,
                intr,
                depth_scale,
                recognition,
                output_dir=output_dir,
                crop_y_min=crop_y_min,
                crop_y_max=crop_y_max,
                target_y_from_space_top_px=target_y_from_space_top_px,
                delta_depth_m=delta_depth_m,
                surface_depth_tol_m=surface_depth_tol_m,
                surface_side_margin_px=surface_side_margin_px,
                surface_gap_px=surface_gap_px,
                surface_stride=surface_stride,
                use_plane_correction=use_plane_correction,
                target_3d_mode=target_3d_mode,
                plane_ransac_threshold_m=plane_ransac_threshold_m,
                plane_ransac_max_iter=plane_ransac_max_iter,
                plane_min_inliers=plane_min_inliers,
                surface_clearance_m=surface_clearance_m,
                inference_metadata=inference_metadata,
                spine_instances=spine,
                end_instances=ends,
                held_book_config=held_book_config,
                timing_sink=timings,
            )
    finally:
        timings["recognition_total_time_ms"] = (
            time.perf_counter() - recognition_total_started
        ) * 1000.0
        timings["file_save_ms"] = (
            timings.get("png_jpg_save_ms", 0.0)
            + timings.get("binary_file_save_ms", 0.0)
        )
        timings["roi_residual_ms"] = (
            timings.get("roi_generation_ms", 0.0)
            + timings.get("residual_generation_ms", 0.0)
        )
        timings["space_candidate_ms"] = (
            timings.get("obstacle_sequence_ms", 0.0)
            + timings.get("space_candidate_generation_ms", 0.0)
            + timings.get("candidate_selection_ms", 0.0)
        )
        timings["depth_geometry_ms"] = (
            timings.get("depth_ref_calculation_ms", 0.0)
            + timings.get("ransac_plane_calculation_ms", 0.0)
            + timings.get("geometry_calculation_ms", 0.0)
        )
        timings["target_ms"] = timings.get(
            "target_point_calculation_ms", 0.0
        )
        timings["debug_image_ms"] = timings.get(
            "debug_image_generation_ms", 0.0
        )
        accounted_keys = (
            "sam3_service_model_prepare_ms",
            "sam3_book_spine_ms",
            "sam3_book_end_ms",
            "sam3_communication_wait_ms",
            "sam3_other_ms",
            "mask_postprocess_ms",
            "held_book_detection_ms",
            "book_end_selection_ms",
            "roi_generation_ms",
            "residual_generation_ms",
            "held_book_occlusion_mask_ms",
            "obstacle_sequence_ms",
            "space_candidate_generation_ms",
            "candidate_selection_ms",
            "depth_ref_calculation_ms",
            "ransac_plane_calculation_ms",
            "geometry_calculation_ms",
            "target_point_calculation_ms",
            "debug_image_generation_ms",
            "png_jpg_save_ms",
            "binary_file_save_ms",
            "json_metadata_save_ms",
        )
        accounted_ms = sum(timings.get(key, 0.0) for key in accounted_keys)
        timings["unaccounted_ms"] = (
            timings["recognition_total_time_ms"] - accounted_ms
        )
        timings["other_ms"] = 0.0
    if result is None:
        raise AssertionError("recognition completed without a result")
    result[2]["recognition_timing_ms"] = dict(timings)
    return result


def intrinsics_from_camera_params(params: dict[str, Any]) -> rs.intrinsics:
    """Build RealSense intrinsics from saved 100test camera_params.json."""
    rs = _get_realsense_module()
    intr = rs.intrinsics()
    intr.width = int(params["width"])
    intr.height = int(params["height"])
    intr.fx = float(params["fx"])
    intr.fy = float(params["fy"])
    intr.ppx = float(params["ppx"])
    intr.ppy = float(params["ppy"])
    intr.model = rs.distortion.none
    intr.coeffs = [0.0] * 5
    return intr


def run_saved_rgbd(
    rgb_path: str | Path,
    depth_path: str | Path,
    camera_params_path: str | Path,
    *,
    shot_dir: str | Path,
    runtime: StorageSam3Runtime | None = None,
    spine_instances: InstanceSet | None = None,
    end_instances: InstanceSet | None = None,
    **kwargs: Any,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Offline validation wrapper; never starts RealSense, ROS, or a robot."""
    rgb = storage_recognition.read_rgb(Path(rgb_path))
    depth = np.load(Path(depth_path), allow_pickle=False).astype(np.uint16)
    params = json.loads(Path(camera_params_path).read_text(encoding="utf-8"))
    intr = intrinsics_from_camera_params(params)
    return run_from_rgbd(
        rgb,
        depth,
        intr,
        float(params["depth_scale"]),
        shot_dir=shot_dir,
        runtime=runtime,
        spine_instances=spine_instances,
        end_instances=end_instances,
        **kwargs,
    )


def run_capture_and_pca_depth_space(
    out_dir: str | Path = "captures_depth_space",
    width: int = 1280,
    height: int = 720,
    fps: int = 6,
    delta_depth_m: float = 0.03,
    min_top_width_px: int = 80,
    min_area_px: int = 500,
    min_height_px: int = 100,
    horizontal_ratio_thr: float = 4.0,
    max_space_width_px: int = 520,
    max_space_area_px: int = 160000,
    min_band_width_px: int = 25,
    crop_y_min: int = 120,
    crop_y_max: int = 400,
    trim_max_row_width_px: int = 180,
    trim_row_width_ratio_thr: float = 2.0,
    trim_bottom_search_ratio: float = 0.45,
    trim_min_keep_height_px: int = 60,
    target_y_from_space_top_px: int = 240,
    rotate_180: bool = False,
    horizontal_kernel_w: int = 300,
    horizontal_kernel_h: int = 30,
    surface_depth_tol_m: float = 0.05,
    surface_side_margin_px: int = 140,
    surface_gap_px: int = 5,
    surface_stride: int = 2,
    use_plane_correction: bool = True,
    target_3d_mode: str = "ransac_plane",
    plane_ransac_threshold_m: float = 0.01,
    plane_ransac_max_iter: int = 300,
    plane_min_inliers: int = 30,
    surface_clearance_m: float = 0.0,
    apply_legacy_depth_ignore: bool = False,
    held_book_detection_enabled: bool = True,
    held_book_min_height_ratio: float = (
        DEFAULT_HELD_BOOK_CONFIG.min_height_ratio
    ),
    held_book_max_top_margin_ratio: float = (
        DEFAULT_HELD_BOOK_CONFIG.max_top_margin_ratio
    ),
    held_book_max_bottom_margin_ratio: float = (
        DEFAULT_HELD_BOOK_CONFIG.max_bottom_margin_ratio
    ),
    show_debug_images: bool = True,
    after_capture_callback: Optional[Callable[[], None]] = None,
    sam3_runtime: StorageSam3Runtime | None = None,
    camera_session: Any = None,
    end_to_end_timing_sink: dict[str, float] | None = None,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Capture once, call the motion callback once, then recognize and calculate."""
    target_3d_mode = _normalize_target_3d_mode(target_3d_mode)
    shot_dir = Path(out_dir) / time.strftime("%Y%m%d_%H%M%S")
    shot_dir.mkdir(parents=True, exist_ok=True)
    rgbd_capture_started = time.perf_counter()
    try:
        rgb_bgr, depth_u16, intr, depth_scale, capture_metadata = (
            _capture_rgbd_once(
                width=width,
                height=height,
                fps=fps,
                shot_dir=shot_dir,
                camera_session=camera_session,
                timing_sink=end_to_end_timing_sink,
            )
        )
    finally:
        rgbd_capture_time_s = time.perf_counter() - rgbd_capture_started
        if end_to_end_timing_sink is not None:
            end_to_end_timing_sink.setdefault(
                "capture_time_s", rgbd_capture_time_s
            )
    rgbd_capture_time_ms = rgbd_capture_time_s * 1000.0
    if after_capture_callback is not None:
        after_capture_callback()

    if rotate_180:
        rgb_bgr = cv2.rotate(rgb_bgr, cv2.ROTATE_180)
        depth_u16 = cv2.rotate(depth_u16, cv2.ROTATE_180)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    held_book_config = HeldBookDetectionConfig(
        enabled=bool(held_book_detection_enabled),
        min_height_ratio=float(held_book_min_height_ratio),
        max_top_margin_ratio=float(held_book_max_top_margin_ratio),
        max_bottom_margin_ratio=float(held_book_max_bottom_margin_ratio),
    )
    recognition_detail_timings: dict[str, float] = {}
    recognition_started = time.perf_counter()
    try:
        angle_rad, first_target_cam, res = run_from_rgbd(
            rgb,
            depth_u16,
            intr,
            depth_scale,
            shot_dir=shot_dir,
            runtime=sam3_runtime,
            apply_legacy_depth_ignore=apply_legacy_depth_ignore,
            held_book_config=held_book_config,
            crop_y_min=crop_y_min,
            crop_y_max=crop_y_max,
            target_y_from_space_top_px=target_y_from_space_top_px,
            delta_depth_m=delta_depth_m,
            surface_depth_tol_m=surface_depth_tol_m,
            surface_side_margin_px=surface_side_margin_px,
            surface_gap_px=surface_gap_px,
            surface_stride=surface_stride,
            use_plane_correction=use_plane_correction,
            target_3d_mode=target_3d_mode,
            plane_ransac_threshold_m=plane_ransac_threshold_m,
            plane_ransac_max_iter=plane_ransac_max_iter,
            plane_min_inliers=plane_min_inliers,
            surface_clearance_m=surface_clearance_m,
            timing_sink=recognition_detail_timings,
        )
    finally:
        recognition_finished = time.perf_counter()
        if end_to_end_timing_sink is not None:
            end_to_end_timing_sink[
                "_recognition_finished_at_perf_counter"
            ] = recognition_finished
            end_to_end_timing_sink["recognition_time_s"] = (
                recognition_finished - recognition_started
            )
            end_to_end_timing_sink["capture_to_recognition_end_time_s"] = (
                recognition_finished - rgbd_capture_started
            )
            model_initialization_s = 0.001 * float(
                recognition_detail_timings.get(
                    "model_initialization_total_ms", 0.0
                )
            )
            end_to_end_timing_sink["model_initialization_total_s"] = (
                model_initialization_s
            )
            sam3_model_load_ms = recognition_detail_timings.get(
                "sam3_model_load_ms"
            )
            if sam3_model_load_ms is not None:
                end_to_end_timing_sink["sam3_model_load_time_s"] = (
                    0.001 * float(sam3_model_load_ms)
                )
            end_to_end_timing_sink["recognition_compute_time_s"] = max(
                0.0,
                end_to_end_timing_sink["recognition_time_s"]
                - model_initialization_s,
            )
            camera_started = end_to_end_timing_sink.get(
                "_camera_inclusive_started_at_perf_counter",
                rgbd_capture_started,
            )
            end_to_end_timing_sink["camera_inclusive_view_time_s"] = max(
                0.0,
                recognition_finished
                - camera_started
                - model_initialization_s,
            )
    res["rgbd_capture_time_ms"] = rgbd_capture_time_ms
    if capture_metadata is not None:
        res["realsense_capture_metadata"] = capture_metadata
    res["legacy_sam2_selection_arguments_accepted_but_unused"] = {
        "min_top_width_px": min_top_width_px,
        "min_area_px": min_area_px,
        "min_height_px": min_height_px,
        "horizontal_ratio_thr": horizontal_ratio_thr,
        "max_space_width_px": max_space_width_px,
        "max_space_area_px": max_space_area_px,
        "min_band_width_px": min_band_width_px,
        "trim_max_row_width_px": trim_max_row_width_px,
        "trim_row_width_ratio_thr": trim_row_width_ratio_thr,
        "trim_bottom_search_ratio": trim_bottom_search_ratio,
        "trim_min_keep_height_px": trim_min_keep_height_px,
        "horizontal_kernel_w": horizontal_kernel_w,
        "horizontal_kernel_h": horizontal_kernel_h,
    }
    res["capture_callback_order"] = [
        "capture_and_copy_rgb_depth_intrinsics",
        "after_capture_callback_once",
        "book_spine_inference",
        "book_end_inference",
        "storage_space_recognition",
        "geometry_calculation",
    ]
    if show_debug_images:
        legacy_geometry.show_image_from_path(
            "final_space_target_point",
            res["final_space_target_point_path"],
            wait=True,
        )
        cv2.destroyAllWindows()
    return angle_rad, first_target_cam, res
