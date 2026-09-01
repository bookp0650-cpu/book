"""Live-compatible SAM3 storage recognition plus legacy Storage_rev geometry.

The completed recognition policy lives in ``offline_100test_storage_space_sam3``.
This module treats its selected-space mask as immutable and only adds the guide
boundary, angle, target pixel, depth deprojection, plane correction, and guide
edge length required by the existing storage caller.
"""
from __future__ import annotations

import atexit
import importlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

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


@dataclass(frozen=True)
class HeldBookDetectionConfig:
    """Configurable full-image occupancy rule for the grasped book spine.

    The defaults separate the two saved real captures in ``captures/strage``:
    their held-book height ratios are 0.925 and 0.990, while shelf-book ratios
    are at most 0.618; the held masks touch y=0 and end within 7/54 pixels of
    the image bottom.  They remain public parameters for hardware tuning.
    """

    enabled: bool = True
    min_height_ratio: float = 0.90
    max_top_margin_ratio: float = 0.02
    max_bottom_margin_ratio: float = 0.10


DEFAULT_HELD_BOOK_CONFIG = HeldBookDetectionConfig()


def detect_held_book_spines(
    spine_instances: InstanceSet,
    config: HeldBookDetectionConfig = DEFAULT_HELD_BOOK_CONFIG,
) -> list[dict[str, Any]]:
    """Return per-instance evidence and held-book decisions from actual masks."""
    masks = np.asarray(spine_instances.masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"expected spine masks NxHxW, got {masks.shape}")
    image_height = int(masks.shape[1]) if masks.ndim == 3 else 0
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
        height_ratio = float(height_px / max(image_height, 1))
        top_margin_ratio = float(y_min / max(image_height, 1))
        bottom_margin_px = image_height - 1 - y_max
        bottom_margin_ratio = float(bottom_margin_px / max(image_height, 1))
        checks = {
            "height_ratio_at_least_minimum": (
                height_ratio >= config.min_height_ratio
            ),
            "top_margin_at_most_maximum": (
                top_margin_ratio <= config.max_top_margin_ratio
            ),
            "bottom_margin_at_most_maximum": (
                bottom_margin_ratio <= config.max_bottom_margin_ratio
            ),
        }
        is_held = bool(config.enabled and all(checks.values()))
        records.append(
            {
                "instance_index": index,
                "instance_id": int(spine_instances.instance_ids[index]),
                "score": float(spine_instances.scores[index]),
                "is_held_book": is_held,
                "reason": (
                    "full_image_vertical_occupancy_rule_matched"
                    if is_held
                    else "full_image_vertical_occupancy_rule_not_matched"
                ),
                "bbox_xyxy": [x_min, y_min, x_max + 1, y_max + 1],
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
    held_index_set = set(held_indices)
    shelf_indices = [
        index
        for index in range(spine_instances.count)
        if index not in held_index_set
    ]
    shape = rgb.shape[:2]
    held_union = (
        np.any(spine_instances.masks[held_indices], axis=0)
        if held_indices
        else np.zeros(shape, dtype=bool)
    )
    shelf_union = (
        np.any(spine_instances.masks[shelf_indices], axis=0)
        if shelf_indices
        else np.zeros(shape, dtype=bool)
    )
    shelf_union &= ~held_union
    held_mask_path = output_dir / "held_book_spine_mask.png"
    shelf_mask_path = output_dir / "shelf_book_spine_mask.png"
    held_overlay_path = output_dir / "held_book_spine_overlay.png"
    shelf_overlay_path = output_dir / "shelf_book_spine_overlay.png"
    storage_recognition.save_png(held_mask_path, held_union.astype(np.uint8) * 255)
    storage_recognition.save_png(shelf_mask_path, shelf_union.astype(np.uint8) * 255)
    storage_recognition.save_png(
        held_overlay_path,
        storage_recognition.overlay_masks(rgb, held_union[None, ...]),
    )
    storage_recognition.save_png(
        shelf_overlay_path,
        storage_recognition.overlay_masks(rgb, shelf_union[None, ...]),
    )
    return {
        "held_book_spine_mask_path": str(held_mask_path),
        "held_book_spine_overlay_path": str(held_overlay_path),
        "shelf_book_spine_mask_path": str(shelf_mask_path),
        "shelf_book_spine_overlay_path": str(shelf_overlay_path),
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

    def start(self) -> dict[str, Any]:
        if not self.started:
            self.start_health = self.session.ensure_ready()
            self.started = True
        return self.start_health or {}

    def infer(
        self,
        rgb: np.ndarray,
    ) -> tuple[InstanceSet, InstanceSet, dict[str, Any]]:
        self.start()
        spine_instances, end_instances, metadata = self.client.infer_instances(rgb)
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
        spine_instances, end_instances, inference_metadata = active_runtime.infer(
            image
        )
    else:
        inference_metadata = {
            "mode": "precomputed_instances",
            "inference_order": ["book_spine", "book_end"],
            "parallel_inference": False,
        }

    held_book_detections = detect_held_book_spines(
        spine_instances,
        held_book_config,
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
    inference_metadata["held_book_occlusion"] = (
        result.held_book_occlusion_metadata
    )
    if output_dir is not None:
        recognition_dir = Path(output_dir)
        storage_recognition.save_storage_recognition_debug(
            recognition_dir,
            image,
            spine_instances,
            end_instances,
            result,
        )
        storage_recognition.save_npz(
            recognition_dir / storage_recognition.SPINE_MODEL.npz_name,
            spine_instances,
        )
        storage_recognition.save_npz(
            recognition_dir / storage_recognition.BOOK_END_MODEL.npz_name,
            end_instances,
        )
        held_debug_paths = _save_held_book_debug(
            recognition_dir,
            image,
            spine_instances,
            held_book_detections,
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
                    "full_image_vertical_occupancy_rule"
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
    guide_path = output_dir / "final_space_guide_boundary.png"
    storage_recognition.save_png(guide_path, guide)

    target = guide.copy()
    cv2.circle(target, target_px, 8, (255, 0, 255), -1)
    target_path = output_dir / "final_space_target_point.png"
    storage_recognition.save_png(target_path, target)

    angle = rgb.copy()
    cv2.line(angle, line_p0, line_p1, (255, 0, 255), 4, cv2.LINE_AA)
    cv2.circle(angle, target_px, 7, (255, 255, 0), -1)
    angle_path = output_dir / "angle_calculation_line.png"
    storage_recognition.save_png(angle_path, angle)
    return {
        "final_space_guide_boundary_path": str(guide_path),
        "final_space_target_point_path": str(target_path),
        "angle_calculation_line_path": str(angle_path),
    }


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
    plane_ransac_threshold_m: float = 0.01,
    plane_ransac_max_iter: int = 300,
    plane_min_inliers: int = 30,
    surface_clearance_m: float = 0.0,
    inference_metadata: dict[str, Any] | None = None,
    spine_instances: InstanceSet | None = None,
    end_instances: InstanceSet | None = None,
    held_book_config: HeldBookDetectionConfig = DEFAULT_HELD_BOOK_CONFIG,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Apply Storage_rev geometry to the immutable selected SAM3 space mask."""
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
    target_px_selected = legacy_geometry.select_target_pixel_on_line_by_y(
        line_p0,
        line_p1,
        target_y_px=target_y_px,
        width=image.shape[1],
        height=image.shape[0],
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
    depth_ref_m = (
        float(np.median(surface_points[:, 2]))
        if len(surface_points)
        else None
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
        plane, inlier_mask = legacy_geometry.fit_plane_ransac(
            surface_points,
            distance_threshold_m=plane_ransac_threshold_m,
            max_iter=plane_ransac_max_iter,
            min_inliers=plane_min_inliers,
            random_seed=0,
        )
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
        else:
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

    assert first_target_cam is not None
    assert first_target_cam_raw is not None

    try:
        target_px_projected = legacy_geometry.project_cam_point_to_pixel(
            first_target_cam,
            intr,
        )
    except Exception:
        target_px_projected = target_px_selected

    guide_length_plane = plane if plane_correction_used else None
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

    debug_paths = _save_geometry_debug(
        output_path,
        image,
        recognition,
        line_p0,
        line_p1,
        target_px_projected,
    )
    legacy_geometry.save_binary_target_overlay(
        selected_mask,
        line_p0=line_p0,
        line_p1=line_p1,
        target_px=target_px_projected,
        save_path=output_path / "selected_mask_target_overlay.png",
    )
    legacy_geometry.save_surface_points_overlay(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        surface_pixels,
        inlier_mask if surface_points.shape[0] == inlier_mask.shape[0] else None,
        output_path / "surface_points_overlay.png",
    )
    ransac_overlay_path = output_path / "book_spine_ransac_points_overlay.png"
    legacy_geometry.save_surface_points_overlay(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        surface_pixels,
        inlier_mask if surface_points.shape[0] == inlier_mask.shape[0] else None,
        ransac_overlay_path,
    )
    ransac_npz_path = output_path / "book_spine_ransac_points.npz"
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
        "depth_ref_m": depth_ref_m,
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
            "full_image_vertical_occupancy_rule"
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
        "sam3_inference_metadata": inference_metadata,
        "shot_dir": str(output_path),
        "selected_mask_target_overlay_path": str(
            output_path / "selected_mask_target_overlay.png"
        ),
        "surface_points_overlay_path": str(
            output_path / "surface_points_overlay.png"
        ),
        "book_spine_ransac_points_overlay_path": str(ransac_overlay_path),
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
            }
        },
    )
    return angle_rad, first_target_cam, res


def _capture_rgbd_once(
    *,
    width: int,
    height: int,
    fps: int,
) -> tuple[np.ndarray, np.ndarray, rs.intrinsics, float]:
    rs = _get_realsense_module()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    try:
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        for _ in range(30):
            pipeline.wait_for_frames()
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
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
    return rgb_bgr, depth_u16, intr, depth_scale


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
    plane_ransac_threshold_m: float = 0.01,
    plane_ransac_max_iter: int = 300,
    plane_min_inliers: int = 30,
    surface_clearance_m: float = 0.0,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Offline-safe entry point for one already captured and aligned RGB-D set."""
    output_dir = Path(shot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = np.asarray(rgb, dtype=np.uint8)
    depth_raw = np.asarray(depth_u16, dtype=np.uint16)
    depth = depth_raw.copy()
    if apply_legacy_depth_ignore:
        depth = _apply_legacy_depth_ignore(depth)
    storage_recognition.save_png(output_dir / "input.png", image)
    np.save(output_dir / "depth_u16_raw.npy", depth_raw, allow_pickle=False)
    np.save(output_dir / "depth_u16.npy", depth, allow_pickle=False)
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
            "legacy_depth_ignore_applied": bool(apply_legacy_depth_ignore),
        },
    )
    recognition, spine, ends, inference_metadata = recognize_storage_space(
        image,
        runtime=runtime,
        spine_instances=spine_instances,
        end_instances=end_instances,
        output_dir=output_dir,
        held_book_config=held_book_config,
    )
    return calculate_geometry_from_selected_space(
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
        plane_ransac_threshold_m=plane_ransac_threshold_m,
        plane_ransac_max_iter=plane_ransac_max_iter,
        plane_min_inliers=plane_min_inliers,
        surface_clearance_m=surface_clearance_m,
        inference_metadata=inference_metadata,
        spine_instances=spine,
        end_instances=ends,
        held_book_config=held_book_config,
    )


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
    target_y_from_space_top_px: int = 200,#240
    rotate_180: bool = False,
    horizontal_kernel_w: int = 300,
    horizontal_kernel_h: int = 30,
    surface_depth_tol_m: float = 0.05,
    surface_side_margin_px: int = 140,
    surface_gap_px: int = 5,
    surface_stride: int = 2,
    use_plane_correction: bool = True,
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
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Capture once, call the motion callback once, then recognize and calculate."""
    shot_dir = Path(out_dir) / time.strftime("%Y%m%d_%H%M%S")
    shot_dir.mkdir(parents=True, exist_ok=True)
    rgb_bgr, depth_u16, intr, depth_scale = _capture_rgbd_once(
        width=width,
        height=height,
        fps=fps,
    )
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
        plane_ransac_threshold_m=plane_ransac_threshold_m,
        plane_ransac_max_iter=plane_ransac_max_iter,
        plane_min_inliers=plane_min_inliers,
        surface_clearance_m=surface_clearance_m,
    )
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
