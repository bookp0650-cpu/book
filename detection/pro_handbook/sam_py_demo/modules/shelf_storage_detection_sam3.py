"""Offline SAM3 shelf-storage perception.

This module only reads saved RGB, aligned Depth, intrinsics, and optionally a
saved SAM3 NPZ.  It deliberately has no RealSense, ROS, or robot imports.

ROI conventions are ``[x1, y1, x2, y2]`` with exclusive ``x2``/``y2``.
Camera points use metres in the optical camera frame: +X right, +Y down, +Z
forward.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional

import cv2
import numpy as np
from PIL import Image

from detection.pro_handbook.sam_py_demo.modules.island import (
    remove_islands_from_masks,
)


class PipelineFailure(RuntimeError):
    """Expected, user-facing recognition failure."""


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float


class RunLog:
    def __init__(self, path: Path):
        self.path = path
        self.lines: list[str] = []

    def write(self, message: str) -> None:
        stamp = datetime.now().isoformat(timespec="milliseconds")
        self.lines.append(f"{stamp} {message}")
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def load_config(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config root must be a JSON object")
    return config


def resolve_shot_inputs(
    *,
    shot_dir: Optional[Path] = None,
    rgb_path: Optional[Path] = None,
    depth_path: Optional[Path] = None,
    intrinsics_path: Optional[Path] = None,
    masks_path: Optional[Path] = None,
    prefer_saved_masks: bool = True,
) -> dict[str, Optional[Path]]:
    """Resolve the repository's established saved-capture filenames."""
    shot = Path(shot_dir).expanduser().resolve() if shot_dir else None

    def choose(explicit: Optional[Path], names: Iterable[str]) -> Optional[Path]:
        if explicit:
            return Path(explicit).expanduser().resolve()
        if shot:
            for name in names:
                candidate = shot / name
                if candidate.is_file():
                    return candidate
        return None

    rgb = choose(rgb_path, ("after_init_rgb.png", "rgb.png", "color.png"))
    depth = choose(depth_path, ("after_init_depth.npy", "depth.npy"))
    intrinsics = choose(
        intrinsics_path, ("camera_params.json", "camera_intrinsics.json")
    )
    saved_masks = (
        choose(masks_path, ("sam3_service_masks.npz", "sam3_instances.npz"))
        if prefer_saved_masks or masks_path
        else None
    )
    return {
        "rgb_path": rgb,
        "depth_path": depth,
        "intrinsics_path": intrinsics,
        "sam3_masks_path": saved_masks,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save image: {path}")


def _new_output_dir(config: dict[str, Any]) -> Path:
    root = Path(config["output"]["root_dir"]).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    root.mkdir(parents=True, exist_ok=True)
    for suffix in range(100):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"{stamp}_offline_shelf_storage_sam3"
        if suffix:
            name += f"_{suffix:02d}"
        output = root / name
        try:
            output.mkdir()
            return output.resolve()
        except FileExistsError:
            continue
    raise RuntimeError("Could not allocate a unique timestamped output directory")


def _load_intrinsics(path: Path, image_shape: tuple[int, int]) -> CameraIntrinsics:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    height, width = image_shape
    intr = CameraIntrinsics(
        width=int(data.get("width", width)),
        height=int(data.get("height", height)),
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data.get("ppx", data.get("cx"))),
        cy=float(data.get("ppy", data.get("cy"))),
        depth_scale=float(data["depth_scale"]),
    )
    if intr.fx <= 0 or intr.fy <= 0 or intr.depth_scale <= 0:
        raise ValueError("fx, fy, and depth_scale must be positive")
    if (intr.width, intr.height) != (width, height):
        raise ValueError(
            f"Intrinsics size {(intr.width, intr.height)} != RGB size {(width, height)}"
        )
    return intr


def _valid_depth_mask(
    depth_raw: np.ndarray, intr: CameraIntrinsics, config: dict[str, Any]
) -> np.ndarray:
    cfg = config["depth_validation"]
    depth_m = depth_raw.astype(np.float32) * intr.depth_scale
    abnormal = depth_raw == int(cfg["sensor_abnormal_max_raw"])
    return (
        (depth_raw > 0)
        & (~abnormal)
        & (depth_m >= float(cfg["min_depth_m"]))
        & (depth_m <= float(cfg["max_depth_m"]))
    )


def _mask_bbox(mask: np.ndarray) -> Optional[list[int]]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _bbox_intersection_ratio(box: list[int], roi: list[int]) -> float:
    x1 = max(box[0], roi[0])
    y1 = max(box[1], roi[1])
    x2 = min(box[2], roi[2])
    y2 = min(box[3], roi[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    return float(intersection / area)


def _bbox_iou(a: list[int], b: list[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - intersection
    )
    return float(intersection / max(1, union))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return float(intersection / max(1, union))


def _acquire_sam3_masks(
    rgb_bgr: np.ndarray,
    saved_npz: Optional[Path],
    output_dir: Path,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    if saved_npz is not None:
        if not saved_npz.is_file():
            raise PipelineFailure(f"SAM3 masks NPZ not found: {saved_npz}")
        with np.load(saved_npz, allow_pickle=False) as data:
            if "masks" not in data:
                raise PipelineFailure(f"'masks' key is absent in {saved_npz}")
            masks = data["masks"].astype(bool)
            boxes = (
                data["boxes"].astype(np.float32)
                if "boxes" in data
                else np.zeros((len(masks), 4), np.float32)
            )
            scores = (
                data["scores"].astype(np.float32)
                if "scores" in data
                else np.ones((len(masks),), np.float32)
            )
        metadata = {
            "source": "saved_sam3_npz",
            "path": str(saved_npz),
            "inference_time_sec": None,
            "load_time_sec": time.perf_counter() - started,
        }
    else:
        from detection.pro_handbook.sam3_runtime.service.client import Sam3BatchInfer

        runner = Sam3BatchInfer(prompt=config["sam3"].get("prompt", "book spine"))
        pil_rgb = Image.fromarray(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))
        masks_list, sam_data = runner.infer_masks(
            pil_rgb, stage_save=SimpleNamespace(out_dir=output_dir)
        )
        masks = np.asarray(masks_list, dtype=bool)
        boxes = np.asarray(
            [
                [
                    item["box"]["x1"],
                    item["box"]["y1"],
                    item["box"]["x2"],
                    item["box"]["y2"],
                ]
                for item in sam_data
            ],
            dtype=np.float32,
        )
        scores = np.asarray(
            [float(item.get("score", 1.0)) for item in sam_data], dtype=np.float32
        )
        metadata = {
            "source": "sam3_service_client",
            "path": None,
            "inference_time_sec": time.perf_counter() - started,
            "service_metadata": runner.last_metadata,
        }
    if masks.ndim != 3 or masks.shape[1:] != rgb_bgr.shape[:2]:
        raise PipelineFailure(
            f"SAM3 masks shape {masks.shape} does not match RGB {rgb_bgr.shape[:2]}"
        )
    if len(boxes) != len(masks) or len(scores) != len(masks):
        raise PipelineFailure("SAM3 masks/boxes/scores counts do not match")
    cache_path = output_dir / "sam3_service_masks.npz"
    np.savez_compressed(
        cache_path,
        masks=masks.astype(bool, copy=False),
        boxes=boxes.astype(np.float32, copy=False),
        scores=scores.astype(np.float32, copy=False),
    )
    metadata["cache_path"] = str(cache_path)
    return masks, boxes, scores, metadata


def _instance_metadata(instance: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _jsonable(value)
        for key, value in instance.items()
        if key not in {"mask", "contour"}
    }


def _candidate_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _jsonable(value)
        for key, value in candidate.items()
        if key
        not in {
            "candidate_mask",
            "candidate_depth_mask",
            "boundary_band_mask",
            "candidate_core_mask",
            "deep_space_mask",
            "boundary_surface_mask",
            "interior_obstacle_mask",
            "significant_obstacle_mask",
            "unknown_depth_mask",
        }
    }


def _build_common_target_roi_mask(
    image_shape: tuple[int, int], config: dict[str, Any]
) -> tuple[np.ndarray, list[int]]:
    height, width = image_shape
    cfg = config["target_shelf_roi"]
    if cfg.get("mode") != "common_fixed_rectangle":
        raise PipelineFailure(
            "target_shelf_roi.mode must be 'common_fixed_rectangle'"
        )
    expected = (int(cfg["image_height"]), int(cfg["image_width"]))
    if (height, width) != expected:
        raise PipelineFailure(
            f"RGB size {(width, height)} does not match configured common ROI "
            f"image size {(expected[1], expected[0])}"
        )
    roi = _clip_roi(
        [cfg["x1"], cfg["y1"], cfg["x2"], cfg["y2"]], width, height
    )
    if roi[2] <= roi[0] or roi[3] <= roi[1]:
        raise PipelineFailure(f"common target shelf ROI is empty: {roi}")
    mask = np.zeros(image_shape, dtype=bool)
    mask[roi[1] : roi[3], roi[0] : roi[2]] = True
    return mask, roi


def _prepare_instances(
    raw_masks: np.ndarray,
    supplied_boxes: np.ndarray,
    scores: np.ndarray,
    depth_raw: np.ndarray,
    intr: CameraIntrinsics,
    config: dict[str, Any],
    target_roi_mask: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    h, w = raw_masks.shape[1:]
    cfg = config["mask_filter"]
    valid_depth = _valid_depth_mask(depth_raw, intr, config)
    prepared: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    order = sorted(range(len(raw_masks)), key=lambda index: float(scores[index]), reverse=True)

    for index in order:
        raw = raw_masks[index].astype(bool)
        reasons: list[str] = []
        raw_area = int(raw.sum())
        if raw_area == 0:
            reasons.append("empty_mask")
            cleaned = raw
        else:
            cleaned = remove_islands_from_masks([raw])[0]
        preclip_area = int(cleaned.sum())
        roi_intersection_ratio = float(
            np.count_nonzero(cleaned & target_roi_mask) / max(1, preclip_area)
        )
        if roi_intersection_ratio < float(cfg["min_roi_intersection_ratio"]):
            reasons.append("mostly_outside_common_target_roi")
        cleaned = cleaned & target_roi_mask
        if np.any(cleaned):
            cleaned = remove_islands_from_masks([cleaned])[0]
        box = _mask_bbox(cleaned)
        if box is None:
            reasons.append("no_connected_component")
            box = [0, 0, 0, 0]
        area = int(cleaned.sum())
        if area < int(cfg["min_mask_area_px"]):
            reasons.append("area_below_minimum")

        supplied = supplied_boxes[index].astype(float).tolist()
        supplied_area = max(1.0, (supplied[2] - supplied[0]) * (supplied[3] - supplied[1]))
        clipped_area = max(0.0, min(w, supplied[2]) - max(0.0, supplied[0])) * max(
            0.0, min(h, supplied[3]) - max(0.0, supplied[1])
        )
        if 1.0 - clipped_area / supplied_area > float(
            cfg["max_bbox_outside_image_ratio"]
        ):
            reasons.append("supplied_bbox_outside_image")

        duplicate_of = None
        if not reasons:
            for kept in prepared:
                if _bbox_iou(box, kept["bbox"]) > 0.05 and _mask_iou(
                    cleaned, kept["mask"]
                ) >= float(cfg["max_pair_iou"]):
                    duplicate_of = kept["instance_id"]
                    reasons.append(f"duplicate_of_{duplicate_of}")
                    break

        ys, xs = np.nonzero(cleaned)
        contour_data = cv2.findContours(
            cleaned.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contours = contour_data[-2]
        contour = (
            max(contours, key=cv2.contourArea).reshape(-1, 2)
            if contours
            else np.empty((0, 2), dtype=np.int32)
        )
        valid_ratio = (
            float(np.count_nonzero(valid_depth & cleaned) / max(1, area))
            if area
            else 0.0
        )
        edge_margin = int(cfg["image_edge_margin_px"])
        edge_touch = bool(
            box[0] <= edge_margin
            or box[1] <= edge_margin
            or box[2] >= w - edge_margin
            or box[3] >= h - edge_margin
        )
        item = {
            "instance_id": int(index),
            "mask": cleaned,
            "contour": contour,
            "bbox": box,
            "sam3_bbox": supplied,
            "score": float(scores[index]),
            "raw_area_px": raw_area,
            "preclip_largest_component_area_px": preclip_area,
            "area_px": area,
            "common_roi_intersection_ratio": roi_intersection_ratio,
            "top_px": int(ys.min()) if ys.size else None,
            "bottom_px": int(ys.max()) if ys.size else None,
            "height_px": int(ys.max() - ys.min() + 1) if ys.size else 0,
            "valid_depth_ratio": valid_ratio,
            "touches_image_edge": edge_touch,
            "largest_component_removed_px": raw_area - area,
            "rejection_reasons": reasons,
        }
        if reasons:
            rejected.append(_instance_metadata(item))
        else:
            prepared.append(item)
    prepared.sort(key=lambda item: item["bbox"][0])
    return prepared, rejected


def _clip_roi(roi: Iterable[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in roi]
    return [
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    ]


def _estimate_shelf_roi(
    instances: list[dict[str, Any]],
    image_shape: tuple[int, int],
    config: dict[str, Any],
    target_roi: list[int],
) -> dict[str, Any]:
    h, w = image_shape
    cfg = config["shelf_roi_detection"]
    fallback = list(target_roi)
    reasons: list[str] = []
    metrics: dict[str, Any] = {"valid_instance_count": len(instances)}
    detected: Optional[list[int]] = None
    robust_bottom: Optional[int] = None

    if not instances:
        reasons.append("no_valid_instances")
    else:
        bottoms = np.asarray([item["bottom_px"] for item in instances], dtype=np.float32)
        heights = np.asarray([item["height_px"] for item in instances], dtype=np.float32)
        median_bottom = float(np.median(bottoms))
        bottom_mad = float(np.median(np.abs(bottoms - median_bottom)))
        tolerance = max(1.0, float(cfg["bottom_mad_multiplier"]) * max(bottom_mad, 1.0))
        bottom_inliers = bottoms[np.abs(bottoms - median_bottom) <= tolerance]
        robust_bottom = int(round(float(np.median(bottom_inliers))))
        robust_height = int(
            round(float(np.percentile(heights, float(cfg["maximum_height_percentile"]))))
        )
        left = min(item["bbox"][0] for item in instances)
        right = max(item["bbox"][2] for item in instances)
        detected_unclipped = [
            left - int(cfg["roi_margin_left_px"]),
            robust_bottom - robust_height,
            right + int(cfg["roi_margin_right_px"]),
            robust_bottom + 1,
        ]
        detected = _clip_roi(detected_unclipped, w, h)
        unclipped_area = max(1, (detected_unclipped[2] - detected_unclipped[0])
                             * (detected_unclipped[3] - detected_unclipped[1]))
        clipped_area = max(0, detected[2] - detected[0]) * max(
            0, detected[3] - detected[1]
        )
        outside_ratio = 1.0 - clipped_area / unclipped_area
        reference_width = max(1, target_roi[2] - target_roi[0])
        occupancy_ratio = float((right - left) / reference_width)
        edge_ratio = float(
            sum(bool(item["touches_image_edge"]) for item in instances) / len(instances)
        )
        depth_instance_ratio = float(
            sum(
                item["valid_depth_ratio"]
                >= float(cfg["min_instance_valid_depth_ratio"])
                for item in instances
            )
            / len(instances)
        )
        metrics.update(
            {
                "bottom_values_px": bottoms.astype(int).tolist(),
                "bottom_median_px": median_bottom,
                "bottom_mad_px": bottom_mad,
                "bottom_inlier_count": int(len(bottom_inliers)),
                "robust_bottom_px": robust_bottom,
                "height_values_px": heights.astype(int).tolist(),
                "robust_maximum_method": (
                    f"percentile_{float(cfg['maximum_height_percentile']):g}"
                ),
                "robust_maximum_height_px": robust_height,
                "horizontal_occupancy_ratio": occupancy_ratio,
                "detected_roi_width_px": detected[2] - detected[0],
                "detected_roi_height_px": detected[3] - detected[1],
                "detected_roi_outside_image_ratio": outside_ratio,
                "edge_touch_instance_ratio": edge_ratio,
                "instances_with_valid_depth_ratio": depth_instance_ratio,
            }
        )
        checks = (
            (
                len(instances) < int(cfg["min_valid_instances"]),
                "too_few_valid_instances",
            ),
            (
                occupancy_ratio < float(cfg["min_horizontal_occupancy_ratio"]),
                "horizontal_occupancy_too_small",
            ),
            (
                detected[2] - detected[0] < int(cfg["min_roi_width_px"]),
                "detected_roi_width_too_small",
            ),
            (
                detected[3] - detected[1] < int(cfg["min_roi_height_px"]),
                "detected_roi_height_too_small",
            ),
            (
                bottom_mad > float(cfg["max_bottom_mad_px"]),
                "bottom_dispersion_too_large",
            ),
            (
                outside_ratio > float(cfg["max_outside_image_ratio"]),
                "detected_roi_too_far_outside_image",
            ),
            (
                edge_ratio > float(cfg["max_edge_touch_instance_ratio"]),
                "too_many_instances_touch_image_edge",
            ),
            (
                depth_instance_ratio
                < float(cfg["min_instances_with_valid_depth_ratio"]),
                "too_few_instances_have_valid_depth",
            ),
        )
        reasons.extend(reason for failed, reason in checks if failed)

    reliable = detected is not None and not reasons
    adopted = list(target_roi)
    source = "common_fixed_rectangle"
    decision = (
        ["common_fixed_rectangle_with_reliable_book_bottom"]
        if reliable
        else ["common_fixed_rectangle_auto_roi_unreliable", *reasons]
    )
    if robust_bottom is None:
        raise PipelineFailure("shelf bottom could not be estimated from book masks")
    robust_bottom = max(adopted[1], min(adopted[3] - 1, int(robust_bottom)))
    return {
        "detected_shelf_roi": detected,
        "fallback_shelf_roi": fallback,
        "adopted_shelf_roi": adopted,
        "roi_source": source,
        "shelf_bottom_px": int(robust_bottom),
        "detected_roi_is_reliable": reliable,
        "decision_reasons": decision,
        "reliability_metrics": metrics,
    }


def _build_occupancy(
    instances: list[dict[str, Any]],
    image_shape: tuple[int, int],
    target_roi_mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.zeros(image_shape, dtype=bool)
    for item in instances:
        raw |= item["mask"]
    raw &= target_roi_mask
    cfg = config["occupied_mask_refinement"]
    safe_u8 = raw.astype(np.uint8)
    hole_kernel = int(cfg["hole_closing_kernel_px"])
    if hole_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (hole_kernel, hole_kernel)
        )
        safe_u8 = cv2.morphologyEx(safe_u8, cv2.MORPH_CLOSE, kernel)
    close_x = int(cfg["horizontal_closing_kernel_px"])
    if close_x > 1:
        safe_u8 = cv2.morphologyEx(
            safe_u8, cv2.MORPH_CLOSE, np.ones((1, close_x), np.uint8)
        )
    margin = int(cfg["horizontal_safety_margin_px"])
    if margin > 0:
        safe_u8 = cv2.dilate(
            safe_u8, np.ones((1, margin * 2 + 1), np.uint8), iterations=1
        )
    safe = safe_u8.astype(bool) & target_roi_mask
    free_raw = target_roi_mask & (~raw)
    free_safe = target_roi_mask & (~safe)
    return raw, safe, safe ^ raw, free_raw, free_safe


def _bottom_clearance(
    free_mask: np.ndarray, roi: list[int], shelf_bottom: int
) -> np.ndarray:
    x1, y1, x2, y2 = roi
    bottom = max(y1, min(y2 - 1, int(shelf_bottom)))
    profile = np.zeros(free_mask.shape[1], dtype=np.int32)
    for x in range(x1, x2):
        column = free_mask[y1 : bottom + 1, x]
        blocked = np.flatnonzero(~column)
        profile[x] = (
            len(column) if blocked.size == 0 else len(column) - int(blocked[-1]) - 1
        )
    return profile


def _extract_bottom_connected_free_mask(
    free_mask: np.ndarray,
    roi: list[int],
    shelf_bottom: int,
    bottom_contact_band_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only vertical free runs connected to a bottom tolerance band."""
    x1, y1, x2, y2 = roi
    bottom = max(y1, min(y2 - 1, int(shelf_bottom)))
    band_top = max(y1, bottom - max(1, int(bottom_contact_band_px)) + 1)
    source = np.zeros_like(free_mask, dtype=np.uint8)
    source[y1 : bottom + 1, x1:x2] = free_mask[
        y1 : bottom + 1, x1:x2
    ].astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(source, connectivity=8)
    kept_labels = {
        int(label)
        for label in np.unique(labels[band_top : bottom + 1, x1:x2])
        if int(label) != 0
    }
    bottom_connected = np.zeros_like(free_mask, dtype=bool)
    component_labels = np.zeros_like(labels, dtype=np.int32)
    profile = np.zeros(free_mask.shape[1], dtype=np.int32)
    next_label = 1
    for label in range(1, count):
        if label not in kept_labels:
            continue
        component = labels == label
        component_output = np.zeros_like(component)
        xs = np.flatnonzero(component[y1 : bottom + 1].any(axis=0))
        for x in xs:
            seed_rows = np.flatnonzero(component[band_top : bottom + 1, x])
            if seed_rows.size == 0:
                continue
            seed_y = band_top + int(seed_rows[-1])
            top_y = seed_y
            while top_y > y1 and component[top_y - 1, x]:
                top_y -= 1
            component_output[top_y : seed_y + 1, x] = component[
                top_y : seed_y + 1, x
            ]
            profile[x] = max(profile[x], seed_y - top_y + 1)
        if np.any(component_output):
            bottom_connected |= component_output
            component_labels[component_output] = next_label
            next_label += 1
    return bottom_connected, component_labels, profile


def _build_candidate_shape_masks(
    bottom_connected_mask: np.ndarray,
    component_labels: np.ndarray,
    profile: np.ndarray,
    required_height: Optional[int],
    roi: list[int],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    gap_cfg = config["gap_detection"]
    threshold = (
        int(required_height)
        if required_height is not None
        else int(gap_cfg["minimum_clearance_height_px_without_book_size"])
    )
    minimum_width = int(gap_cfg["minimum_candidate_width_px"])
    support_threshold = max(
        1,
        int(
            math.ceil(
                threshold
                * float(gap_cfg["minimum_shape_support_height_ratio"])
            )
        ),
    )
    shapes: list[dict[str, Any]] = []
    for component_label in range(1, int(component_labels.max()) + 1):
        component = component_labels == component_label
        valid_columns = (profile >= threshold) & component.any(axis=0)
        support_columns = (
            (profile >= support_threshold) & component.any(axis=0)
        )
        emitted_ranges: set[tuple[int, int]] = set()
        for run_x1, run_x2 in _contiguous_runs(valid_columns, roi[0], roi[2]):
            if run_x2 - run_x1 < minimum_width:
                continue
            shape_x1 = run_x1
            while shape_x1 > roi[0] and support_columns[shape_x1 - 1]:
                shape_x1 -= 1
            shape_x2 = run_x2
            while shape_x2 < roi[2] and support_columns[shape_x2]:
                shape_x2 += 1
            shape_range = (shape_x1, shape_x2)
            if shape_range in emitted_ranges:
                continue
            emitted_ranges.add(shape_range)
            interval = np.zeros_like(component)
            interval[:, shape_x1:shape_x2] = True
            candidate_mask = component & bottom_connected_mask & interval
            if not np.any(candidate_mask):
                continue
            shapes.append(
                {
                    "component_label": component_label,
                    "valid_height_x_range": [run_x1, run_x2],
                    "x_range": [shape_x1, shape_x2],
                    "candidate_mask": candidate_mask,
                }
            )
    return shapes


def _compute_candidate_width_profile(
    candidate_mask: np.ndarray,
    shelf_bottom: int,
    required_height: Optional[int],
    roi: list[int],
    safe_percentile: float,
) -> dict[str, Any]:
    bbox = _mask_bbox(candidate_mask)
    if bbox is None:
        return {
            "profile": [],
            "minimum": 0,
            "safe_percentile": 0.0,
            "maximum": 0,
            "minimum_row": None,
            "height": 0,
        }
    bottom = max(roi[1], min(roi[3] - 1, int(shelf_bottom)))
    top = (
        max(roi[1], bottom - int(required_height) + 1)
        if required_height is not None
        else bbox[1]
    )
    profile: list[dict[str, int]] = []
    widths: list[int] = []
    for y in range(top, bottom + 1):
        xs = np.flatnonzero(candidate_mask[y])
        if xs.size:
            left_x, right_x = int(xs.min()), int(xs.max())
            width = right_x - left_x + 1
        else:
            left_x = right_x = -1
            width = 0
        profile.append(
            {"y": y, "left_x": left_x, "right_x": right_x, "width_px": width}
        )
        widths.append(width)
    width_array = np.asarray(widths, dtype=np.float32)
    minimum_index = int(np.argmin(width_array)) if width_array.size else 0
    return {
        "profile": profile,
        "minimum": int(width_array.min()) if width_array.size else 0,
        "safe_percentile": (
            float(np.percentile(width_array, float(safe_percentile)))
            if width_array.size
            else 0.0
        ),
        "maximum": int(width_array.max()) if width_array.size else 0,
        "minimum_row": profile[minimum_index] if profile else None,
        "height": int(bbox[3] - bbox[1]),
    }


def _largest_contour_points(mask: np.ndarray) -> list[list[int]]:
    contours = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    if not contours:
        return []
    return (
        max(contours, key=cv2.contourArea)
        .reshape(-1, 2)
        .astype(int)
        .tolist()
    )


def _median_depth_m(
    depth_raw: np.ndarray,
    mask: np.ndarray,
    intr: CameraIntrinsics,
    config: dict[str, Any],
) -> Optional[float]:
    valid = _valid_depth_mask(depth_raw, intr, config) & mask
    if not np.any(valid):
        return None
    return float(np.median(depth_raw[valid].astype(np.float64) * intr.depth_scale))


def _representative_front_depth(
    instances: list[dict[str, Any]],
    depth_raw: np.ndarray,
    intr: CameraIntrinsics,
    config: dict[str, Any],
) -> Optional[float]:
    values = [
        value
        for value in (
            _median_depth_m(depth_raw, item["mask"], intr, config)
            for item in instances
        )
        if value is not None
    ]
    return float(np.median(values)) if values else None


def _required_height_px(
    config: dict[str, Any], intr: CameraIntrinsics, front_depth_m: Optional[float]
) -> tuple[Optional[int], str]:
    direct = config["gap_detection"].get("required_height_px")
    if direct is not None:
        return max(1, int(round(float(direct)))), "config_required_height_px"
    height_mm = config.get("book_dimensions_mm", {}).get("book_height_mm")
    if height_mm is not None and front_depth_m and front_depth_m > 0:
        value = float(height_mm) / 1000.0 * intr.fy / front_depth_m
        return max(1, int(math.ceil(value))), "book_height_mm_at_front_plane"
    return None, "not_available"


def _contiguous_runs(values: np.ndarray, start: int, stop: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    run_start: Optional[int] = None
    for x in range(start, stop):
        if bool(values[x]) and run_start is None:
            run_start = x
        elif not bool(values[x]) and run_start is not None:
            runs.append((run_start, x))
            run_start = None
    if run_start is not None:
        runs.append((run_start, stop))
    return runs


def _neighbor_instances(
    x1: int,
    x2: int,
    instances: list[dict[str, Any]],
    roi: list[int],
    max_distance: int,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], str, str]:
    left_options = [
        item
        for item in instances
        if item["bbox"][2] <= x1 and x1 - item["bbox"][2] <= max_distance
    ]
    right_options = [
        item
        for item in instances
        if item["bbox"][0] >= x2 and item["bbox"][0] - x2 <= max_distance
    ]
    left = (
        max(left_options, key=lambda item: item["bbox"][2])
        if left_options
        else None
    )
    right = (
        min(right_options, key=lambda item: item["bbox"][0])
        if right_options
        else None
    )
    left_type = "book" if left is not None else ("shelf_side" if x1 <= roi[0] else "unknown")
    right_type = (
        "book" if right is not None else ("shelf_side" if x2 >= roi[2] else "unknown")
    )
    return left, right, left_type, right_type


def _neighbor_instances_for_mask(
    candidate_mask: np.ndarray,
    instances: list[dict[str, Any]],
    roi: list[int],
    max_distance: int,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], str, str]:
    bbox = _mask_bbox(candidate_mask)
    if bbox is None:
        return None, None, "unknown", "unknown"
    ys, xs = np.nonzero(candidate_mask)
    center_x = float(np.median(xs))
    options: list[tuple[float, str, dict[str, Any]]] = []
    for item in instances:
        book_mask = item["mask"].astype(bool)
        if not np.any(book_mask):
            continue
        distance = cv2.distanceTransform(
            (~book_mask).astype(np.uint8), cv2.DIST_L2, 3
        )
        minimum = float(distance[candidate_mask].min())
        if minimum > float(max_distance):
            continue
        book_x = float(np.median(np.nonzero(book_mask)[1]))
        side = "left" if book_x < center_x else "right"
        options.append((minimum, side, item))
    left_options = [entry for entry in options if entry[1] == "left"]
    right_options = [entry for entry in options if entry[1] == "right"]
    left = min(left_options, default=None, key=lambda entry: entry[0])
    right = min(right_options, default=None, key=lambda entry: entry[0])
    left_item = left[2] if left is not None else None
    right_item = right[2] if right is not None else None
    touches_left = bbox[0] <= roi[0]
    touches_right = bbox[2] >= roi[2]
    left_type = "book" if left_item is not None else ("roi_edge" if touches_left else "unknown")
    right_type = (
        "book" if right_item is not None else ("roi_edge" if touches_right else "unknown")
    )
    return left_item, right_item, left_type, right_type


def _deproject(pixel: tuple[float, float], depth_m: float, intr: CameraIntrinsics) -> np.ndarray:
    u, v = pixel
    return np.asarray(
        [
            (float(u) - intr.cx) / intr.fx * depth_m,
            (float(v) - intr.cy) / intr.fy * depth_m,
            depth_m,
        ],
        dtype=np.float64,
    )


def _points_from_mask(
    mask: np.ndarray,
    depth_raw: np.ndarray,
    intr: CameraIntrinsics,
    config: dict[str, Any],
) -> np.ndarray:
    valid = mask & _valid_depth_mask(depth_raw, intr, config)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    stride = max(1, int(config["front_plane"]["point_sample_stride_px"]))
    xs, ys = xs[::stride], ys[::stride]
    depths = depth_raw[ys, xs].astype(np.float64) * intr.depth_scale
    median = float(np.median(depths))
    tolerance = float(config["front_plane"]["depth_median_tolerance_m"])
    keep = np.abs(depths - median) <= tolerance
    xs, ys, depths = xs[keep], ys[keep], depths[keep]
    points = np.empty((len(xs), 3), dtype=np.float64)
    points[:, 0] = (xs - intr.cx) / intr.fx * depths
    points[:, 1] = (ys - intr.cy) / intr.fy * depths
    points[:, 2] = depths
    return points


def fit_plane_ransac(
    points: np.ndarray,
    *,
    distance_threshold_m: float,
    max_iterations: int,
    min_inliers: int,
    random_seed: int,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """Independent, deterministic storage-plane RANSAC followed by SVD refinement."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 3:
        return None, np.zeros(len(pts), dtype=bool)
    rng = np.random.default_rng(int(random_seed))
    best = np.zeros(len(pts), dtype=bool)
    for _ in range(int(max_iterations)):
        ids = rng.choice(len(pts), 3, replace=False)
        normal = np.cross(pts[ids[1]] - pts[ids[0]], pts[ids[2]] - pts[ids[0]])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-10:
            continue
        normal /= norm
        d = -float(np.dot(normal, pts[ids[0]]))
        inliers = np.abs(pts @ normal + d) <= float(distance_threshold_m)
        if int(inliers.sum()) > int(best.sum()):
            best = inliers
    if int(best.sum()) < int(min_inliers):
        return None, np.zeros(len(pts), dtype=bool)
    inlier_points = pts[best]
    centroid = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    if normal[2] < 0:
        normal = -normal
    plane = np.r_[normal, -float(np.dot(normal, centroid))]
    return plane.astype(np.float64), best


def _fit_instance_plane(
    instance: Optional[dict[str, Any]],
    depth_raw: np.ndarray,
    intr: CameraIntrinsics,
    config: dict[str, Any],
) -> dict[str, Any]:
    if instance is None:
        return {
            "plane": None,
            "median_depth_m": None,
            "point_count": 0,
            "inlier_count": 0,
            "ransac_ok": False,
            "front_region_source": "unavailable",
        }
    front_mask = instance["mask"].astype(bool)
    front_region_source = "book_mask"
    if "instance_id" in instance:
        radius = max(
            0, int(config.get("book_front_plane", {}).get("front_region_erode_px", 0))
        )
        if radius > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
            )
            eroded = cv2.erode(front_mask.astype(np.uint8), kernel).astype(bool)
            if np.count_nonzero(eroded) >= int(
                config["front_plane"]["ransac_min_inliers"]
            ):
                front_mask = eroded
                front_region_source = "eroded_book_center"
            else:
                front_region_source = "book_mask_erosion_fallback"
    points = _points_from_mask(front_mask, depth_raw, intr, config)
    cfg = config["front_plane"]
    plane, inliers = fit_plane_ransac(
        points,
        distance_threshold_m=float(cfg["ransac_distance_threshold_m"]),
        max_iterations=int(cfg["ransac_max_iterations"]),
        min_inliers=int(cfg["ransac_min_inliers"]),
        random_seed=int(cfg["ransac_random_seed"]),
    )
    return {
        "plane": plane,
        "median_depth_m": float(np.median(points[:, 2])) if len(points) else None,
        "point_count": int(len(points)),
        "inlier_count": int(inliers.sum()),
        "ransac_ok": plane is not None,
        "front_region_source": front_region_source,
    }


def _fit_mask_plane(
    mask: np.ndarray,
    depth_raw: np.ndarray,
    intr: CameraIntrinsics,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit a plane to a non-book support mask, such as a shelf-side strip."""
    return _fit_instance_plane(
        {"mask": np.asarray(mask, dtype=bool)}, depth_raw, intr, config
    )


def intersect_pixel_ray_with_plane(
    pixel: tuple[float, float], plane: np.ndarray, intr: CameraIntrinsics
) -> np.ndarray:
    ray = _deproject(pixel, 1.0, intr)
    normal = np.asarray(plane[:3], dtype=np.float64)
    denominator = float(np.dot(normal, ray))
    if abs(denominator) < 1e-10:
        raise PipelineFailure("pixel ray is parallel to the estimated front plane")
    scale = -float(plane[3]) / denominator
    if scale <= 0:
        raise PipelineFailure("pixel ray/front plane intersection is behind camera")
    return ray * scale


def _point_on_front(
    pixel: tuple[float, float],
    plane_info: dict[str, Any],
    intr: CameraIntrinsics,
    fallback_depth_m: Optional[float],
) -> tuple[Optional[np.ndarray], str]:
    if plane_info.get("plane") is not None:
        return (
            intersect_pixel_ray_with_plane(pixel, plane_info["plane"], intr),
            "ransac_plane",
        )
    depth = plane_info.get("median_depth_m") or fallback_depth_m
    if depth is None or depth <= 0:
        return None, "unavailable"
    return _deproject(pixel, float(depth), intr), "median_depth_fallback"


def _fit_left_boundary_line(
    instance: Optional[dict[str, Any]],
    y_top: int,
    y_bottom: int,
    min_points: int,
) -> Optional[dict[str, Any]]:
    if instance is None:
        return None
    mask = instance["mask"]
    points = []
    for y in range(max(0, y_top), min(mask.shape[0], y_bottom + 1)):
        xs = np.flatnonzero(mask[y])
        if xs.size:
            points.append([float(xs.min()), float(y)])
    if len(points) < min_points:
        return None
    values = np.asarray(points, dtype=np.float64)
    center = values.mean(axis=0)
    _, _, vh = np.linalg.svd(values - center, full_matrices=False)
    direction = vh[0]
    if abs(float(direction[1])) < 1e-8:
        return None
    return {
        "point": center,
        "direction": direction,
        "point_count": len(points),
    }


def _evaluate_line_x(line: dict[str, Any], y: float) -> float:
    point = line["point"]
    direction = line["direction"]
    return float(point[0] + (float(y) - point[1]) * direction[0] / direction[1])


def _build_candidate_depth_regions(
    candidate_shape_mask: np.ndarray,
    left: Optional[dict[str, Any]],
    right: Optional[dict[str, Any]],
    shelf_bottom: int,
    required_height: Optional[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    cfg = config["candidate_depth_regions"]
    radius = max(0, int(cfg["candidate_depth_erode_px"]))
    if radius:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        depth_mask = cv2.erode(
            candidate_shape_mask.astype(np.uint8), kernel
        ).astype(bool)
    else:
        depth_mask = candidate_shape_mask.copy()
    erosion_fallback = False
    if not np.any(depth_mask) and bool(cfg["fallback_when_eroded_empty"]):
        depth_mask = candidate_shape_mask.copy()
        erosion_fallback = True

    evaluation_top = 0
    evaluation_bottom = candidate_shape_mask.shape[0] - 1
    if required_height is not None:
        evaluation_bottom = max(
            0, min(candidate_shape_mask.shape[0] - 1, int(shelf_bottom))
        )
        evaluation_top = max(
            0, evaluation_bottom - int(required_height) + 1
        )
        insertion_height_mask = np.zeros_like(candidate_shape_mask, dtype=bool)
        insertion_height_mask[
            evaluation_top : evaluation_bottom + 1
        ] = True
        depth_mask &= insertion_height_mask

    band = np.zeros_like(candidate_shape_mask, dtype=bool)
    band_px = float(cfg["boundary_band_px"])
    for neighbor in (left, right):
        if neighbor is None:
            continue
        distance = cv2.distanceTransform(
            (~neighbor["mask"].astype(bool)).astype(np.uint8), cv2.DIST_L2, 3
        )
        band |= depth_mask & (distance <= band_px)
    core = depth_mask & (~band)
    return {
        "candidate_depth_mask": depth_mask,
        "boundary_band_mask": band,
        "candidate_core_mask": core,
        "candidate_depth_erosion_fallback": erosion_fallback,
        "depth_evaluation_row_range": [
            int(evaluation_top),
            int(evaluation_bottom),
        ],
    }


def _front_depth_map(
    mask: np.ndarray,
    plane_infos: list[dict[str, Any]],
    intr: CameraIntrinsics,
    fallback_depth_m: Optional[float],
) -> tuple[np.ndarray, str]:
    expected = np.full(mask.shape, np.nan, dtype=np.float64)
    ys, xs = np.nonzero(mask)
    sources: list[str] = []
    maps: list[np.ndarray] = []
    for info in plane_infos:
        plane = info.get("plane")
        if plane is not None:
            a, b, c, d = [float(value) for value in plane]
            denominator = (
                a * (xs - intr.cx) / intr.fx
                + b * (ys - intr.cy) / intr.fy
                + c
            )
            values = np.full(xs.shape, np.nan, dtype=np.float64)
            usable = np.abs(denominator) > 1e-10
            values[usable] = -d / denominator[usable]
            values[values <= 0] = np.nan
            maps.append(values)
            sources.append("plane")
        elif info.get("median_depth_m") is not None:
            maps.append(
                np.full(xs.shape, float(info["median_depth_m"]), dtype=np.float64)
            )
            sources.append("median")
    if maps:
        stacked = np.stack(maps, axis=0)
        with np.errstate(all="ignore"):
            expected[ys, xs] = np.nanmin(stacked, axis=0)
    elif fallback_depth_m is not None:
        expected[ys, xs] = float(fallback_depth_m)
        sources.append("fallback")

    plane_count = sum(source == "plane" for source in sources)
    if plane_count >= 2:
        source = "both_book_planes"
    elif plane_count == 1:
        source = "book_plane"
    elif "median" in sources:
        source = "book_median_depth"
    elif "fallback" in sources:
        source = "fallback_depth"
    else:
        source = "unavailable"
    return expected, source


def _analyze_core_obstacles(
    obstacle_mask: np.ndarray,
    candidate_core_mask: np.ndarray,
    required_height: Optional[int],
    required_width_px: Optional[float],
    config: dict[str, Any],
) -> dict[str, Any]:
    cfg = config["core_obstacle"]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        obstacle_mask.astype(np.uint8), connectivity=8
    )
    components: list[dict[str, Any]] = []
    core_area = max(1, int(candidate_core_mask.sum()))
    min_area = max(
        int(cfg["minimum_component_area_px"]),
        int(math.ceil(float(cfg["minimum_component_area_ratio"]) * core_area)),
    )
    for label in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[label]]
        if area >= min_area:
            components.append(
                {
                    "label": label,
                    "area": area,
                    "bbox": [x, y, x + width, y + height],
                    "height": height,
                }
            )
    largest = max(components, default=None, key=lambda item: item["area"])
    significant = np.zeros_like(obstacle_mask, dtype=bool)
    for item in components:
        significant |= labels == item["label"]

    crosses_height = False
    if largest is not None and required_height is not None:
        crosses_height = largest["height"] >= int(
            math.ceil(
                float(cfg["required_height_crossing_ratio"]) * required_height
            )
        )

    reduces_width = False
    if required_width_px is not None and np.any(significant):
        ys = np.flatnonzero(candidate_core_mask.any(axis=1))
        for y in ys:
            free_row = candidate_core_mask[y] & (~significant[y])
            runs = _contiguous_runs(free_row, 0, free_row.size)
            maximum = max((x2 - x1 for x1, x2 in runs), default=0)
            if maximum < float(required_width_px):
                reduces_width = True
                break

    base_count = cv2.connectedComponents(
        candidate_core_mask.astype(np.uint8), connectivity=8
    )[0]
    remaining = candidate_core_mask & (~significant)
    remaining_count = cv2.connectedComponents(
        remaining.astype(np.uint8), connectivity=8
    )[0]
    splits = bool(np.any(significant) and remaining_count > base_count)
    return {
        "significant_obstacle_mask": significant,
        "largest_core_obstacle_area_px": largest["area"] if largest else 0,
        "largest_core_obstacle_bbox": largest["bbox"] if largest else None,
        "obstacle_crosses_required_height": crosses_height,
        "obstacle_reduces_required_width": reduces_width,
        "obstacle_splits_candidate": splits,
        "significant_obstacle_count": len(components),
    }


def _classify_candidate_depth(
    regions: dict[str, Any],
    left_plane: dict[str, Any],
    right_plane: dict[str, Any],
    fallback_front_depth_m: Optional[float],
    depth_raw: np.ndarray,
    intr: CameraIntrinsics,
    required_height: Optional[int],
    required_width_px: Optional[float],
    config: dict[str, Any],
) -> dict[str, Any]:
    depth_mask = regions["candidate_depth_mask"]
    boundary = regions["boundary_band_mask"]
    core = regions["candidate_core_mask"]
    valid = _valid_depth_mask(depth_raw, intr, config)
    expected, reference_source = _front_depth_map(
        depth_mask, [left_plane, right_plane], intr, fallback_front_depth_m
    )
    observed = depth_raw.astype(np.float64) * intr.depth_scale
    comparable = depth_mask & valid & np.isfinite(expected)
    deep = comparable & (
        observed >= expected + float(config["depth_validation"]["min_gap_behind_front_m"])
    )
    boundary_surface = boundary & comparable & (~deep)
    interior_obstacle = core & comparable & (~deep)
    unknown = depth_mask & (~comparable)
    deep_space = deep

    obstacle = _analyze_core_obstacles(
        interior_obstacle,
        core,
        required_height,
        required_width_px,
        config,
    )

    def ratio(mask: np.ndarray, region: np.ndarray) -> float:
        return float(np.count_nonzero(mask & region) / max(1, np.count_nonzero(region)))

    core_valid_ratio = ratio(valid & np.isfinite(expected), core)
    core_deep_ratio = ratio(deep_space, core)
    core_obstacle_ratio = ratio(interior_obstacle, core)
    core_unknown_ratio = ratio(unknown, core)
    boundary_valid_ratio = ratio(valid & np.isfinite(expected), boundary)
    boundary_near_ratio = ratio(boundary_surface, boundary)
    boundary_deep_ratio = ratio(deep_space, boundary)
    boundary_unknown_ratio = ratio(unknown, boundary)

    rejection: list[str] = []
    uncertain: list[str] = []
    if np.count_nonzero(core) < int(
        config["candidate_depth_regions"]["minimum_core_area_px"]
    ):
        uncertain.append("candidate_core_too_small")
    if core_valid_ratio < float(config["depth_validation"]["min_valid_depth_ratio"]):
        uncertain.append("core_depth_valid_ratio_too_low")
    if core_deep_ratio < float(
        config["candidate_depth_regions"]["minimum_core_deep_ratio"]
    ):
        uncertain.append("insufficient_deep_space_confirmation")
    if reference_source == "unavailable":
        uncertain.append("front_reference_unavailable")
    if regions["candidate_depth_erosion_fallback"]:
        uncertain.append("candidate_depth_erosion_fallback")
    if obstacle["significant_obstacle_count"]:
        rejection.append("core_interior_obstacle_blocks_insertion")

    if rejection:
        depth_status = "rejected"
    elif uncertain:
        depth_status = "uncertain"
    else:
        depth_status = "accepted"
    return {
        "depth_status": depth_status,
        "front_reference_source": reference_source,
        "core_valid_ratio": core_valid_ratio,
        "core_deep_ratio": core_deep_ratio,
        "core_obstacle_ratio": core_obstacle_ratio,
        "core_unknown_ratio": core_unknown_ratio,
        "boundary_valid_ratio": boundary_valid_ratio,
        "boundary_near_ratio": boundary_near_ratio,
        "boundary_deep_ratio": boundary_deep_ratio,
        "boundary_unknown_ratio": boundary_unknown_ratio,
        "depth_reasons": rejection,
        "depth_uncertain_reasons": uncertain,
        "deep_space_mask": deep_space,
        "boundary_surface_mask": boundary_surface,
        "interior_obstacle_mask": interior_obstacle,
        "unknown_depth_mask": unknown,
        **obstacle,
    }


def _candidate_confidence(candidate: dict[str, Any]) -> float:
    valid = float(candidate.get("core_valid_ratio") or 0.0)
    deep = float(candidate.get("core_deep_ratio") or 0.0)
    obstacle = float(candidate.get("core_obstacle_ratio") or 0.0)
    return valid * max(0.0, deep) * max(0.0, 1.0 - obstacle)


def _select_safe_target_pixel(
    nominal_target_pixel: Optional[list[int]],
    candidate_depth_mask: np.ndarray,
    interior_obstacle_mask: np.ndarray,
) -> tuple[Optional[list[int]], Optional[float], str]:
    if nominal_target_pixel is None:
        return None, None, "nominal_target_unavailable"
    x, y = [int(value) for value in nominal_target_pixel]
    if not (0 <= y < candidate_depth_mask.shape[0]):
        return None, None, "nominal_target_outside_image"
    safe = candidate_depth_mask & (~interior_obstacle_mask)
    xs = np.flatnonzero(safe[y])
    if xs.size == 0:
        return None, None, "no_safe_pixel_at_target_height"
    selected_x = int(xs[np.argmin(np.abs(xs - x))])
    adjustment = float(abs(selected_x - x))
    reason = "nominal_target_is_safe" if adjustment == 0 else "adjusted_inside_safe_candidate"
    return [selected_x, y], adjustment, reason


def _extract_candidates_legacy(
    candidate_shapes: list[dict[str, Any]],
    required_height: Optional[int],
    roi: list[int],
    shelf_bottom: int,
    instances: list[dict[str, Any]],
    depth_raw: np.ndarray,
    intr: CameraIntrinsics,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    gap_cfg = config["gap_detection"]
    threshold = (
        required_height
        if required_height is not None
        else int(gap_cfg["minimum_clearance_height_px_without_book_size"])
    )
    valid_columns = profile >= threshold
    runs = _contiguous_runs(valid_columns, roi[0], roi[2])
    candidates: list[dict[str, Any]] = []
    front_cfg = config["front_plane"]
    dims = config.get("book_dimensions_mm", {})

    for candidate_id, (x1, x2) in enumerate(runs):
        safe_x1 = x1 + int(gap_cfg["left_clearance_px"])
        safe_x2 = x2 - int(gap_cfg["right_clearance_px"])
        clearances = profile[x1:x2]
        minimum_clearance = int(clearances.min()) if clearances.size else 0
        maximum_clearance = int(clearances.max()) if clearances.size else 0
        y_top = max(roi[1], shelf_bottom - minimum_clearance + 1)
        left, right, left_type, right_type = _neighbor_instances(
            x1,
            x2,
            instances,
            roi,
            int(gap_cfg["boundary_search_max_distance_px"]),
        )
        rejection: list[str] = []
        if safe_x2 <= safe_x1:
            rejection.append("clearance_margins_consume_candidate")
        if safe_x2 - safe_x1 < int(gap_cfg["minimum_candidate_width_px"]):
            rejection.append("pixel_width_below_minimum")
        if right_type == "unknown":
            rejection.append("right_boundary_unknown")

        height_for_region = required_height or minimum_clearance
        region_top = max(roi[1], shelf_bottom - height_for_region + 1)
        candidate_mask = np.zeros(depth_raw.shape, dtype=bool)
        if safe_x2 > safe_x1:
            candidate_mask[region_top : shelf_bottom + 1, safe_x1:safe_x2] = True

        left_plane = _fit_instance_plane(left, depth_raw, intr, config)
        right_plane = _fit_instance_plane(right, depth_raw, intr, config)
        neighbor_depths = [
            info["median_depth_m"]
            for info in (left_plane, right_plane)
            if info["median_depth_m"] is not None
        ]
        neighbor_front = float(np.median(neighbor_depths)) if neighbor_depths else None
        depth_result = _depth_check(
            candidate_mask, neighbor_front, depth_raw, intr, config
        )

        if required_height is not None:
            target_y = int(
                round(
                    shelf_bottom
                    - float(config["target"]["target_height_ratio"]) * required_height
                )
            )
            target_y_rule = "shelf_bottom_minus_required_height_ratio"
        else:
            target_y = int(round((y_top + shelf_bottom) / 2))
            target_y_rule = "candidate_vertical_midpoint"
        target_y = max(roi[1], min(shelf_bottom, target_y))

        boundary_line = None
        if right is not None:
            boundary_line = _fit_left_boundary_line(
                right,
                region_top,
                shelf_bottom,
                int(config["target"]["boundary_fit_min_points"]),
            )
            if boundary_line is not None:
                right_boundary_x = _evaluate_line_x(boundary_line, target_y)
                right_boundary_method = "right_book_left_contour_svd"
            else:
                right_boundary_x = float(right["bbox"][0])
                right_boundary_method = "right_book_bbox_left_fallback"
        elif right_type == "shelf_side":
            right_boundary_x = float(roi[2] - 1)
            right_boundary_method = "adopted_shelf_roi_right_side"
        else:
            right_boundary_x = None
            right_boundary_method = "unavailable"

        target_x_unclamped = (
            int(round(right_boundary_x - int(config["target"]["target_inset_px"])))
            if right_boundary_x is not None
            else None
        )
        target_x = target_x_unclamped
        if target_x is not None and safe_x2 > safe_x1:
            target_x = max(safe_x1, min(max(safe_x1, safe_x2 - 1), target_x))
        elif safe_x2 <= safe_x1:
            target_x = None
        target_clamped = (
            target_x is not None
            and target_x_unclamped is not None
            and target_x != target_x_unclamped
        )
        target_pixel = [target_x, target_y] if target_x is not None else None

        if right is not None:
            target_plane_info, target_plane_source = right_plane, "right_book"
        else:
            target_plane_info, target_plane_source = {
                "plane": None,
                "median_depth_m": None,
            }, "fallback_depth"
            if right_type == "shelf_side":
                side_width = max(
                    1, int(config["target"]["shelf_side_strip_width_px"])
                )
                side_mask = np.zeros(depth_raw.shape, dtype=bool)
                side_x1 = max(roi[0], roi[2] - side_width)
                side_mask[region_top : shelf_bottom + 1, side_x1 : roi[2]] = True
                shelf_side_plane = _fit_mask_plane(
                    side_mask, depth_raw, intr, config
                )
                if shelf_side_plane["plane"] is not None:
                    target_plane_info = shelf_side_plane
                    target_plane_source = "shelf_side"
                elif left is not None and (
                    left_plane["plane"] is not None
                    or left_plane["median_depth_m"] is not None
                ):
                    target_plane_info = left_plane
                    target_plane_source = "left_book"
                else:
                    neighboring_mask = np.zeros(depth_raw.shape, dtype=bool)
                    for neighbor in instances:
                        neighboring_mask |= neighbor["mask"]
                    neighboring_plane = _fit_mask_plane(
                        neighboring_mask, depth_raw, intr, config
                    )
                    if neighboring_plane["plane"] is not None:
                        target_plane_info = neighboring_plane
                        target_plane_source = "neighboring_books"
        fallback_depth = (
            neighbor_front
            if neighbor_front is not None
            else float(front_cfg["fallback_front_depth_m"])
        )
        target_point = None
        target_3d_method = "unavailable"
        if target_pixel is not None:
            target_point, target_3d_method = _point_on_front(
                (target_pixel[0], target_pixel[1]),
                target_plane_info,
                intr,
                fallback_depth,
            )
            if target_point is not None:
                target_point = target_point.copy()
                target_point[2] -= float(config["target"]["surface_clearance_m"])

        width_plane_info = (
            right_plane
            if right_plane["plane"] is not None
            else left_plane
            if left_plane["plane"] is not None
            else {"plane": None, "median_depth_m": neighbor_front}
        )
        if safe_x2 > safe_x1:
            left_point, width_left_method = _point_on_front(
                (safe_x1, target_y), width_plane_info, intr, neighbor_front
            )
            right_point, width_right_method = _point_on_front(
                (safe_x2 - 1, target_y),
                width_plane_info,
                intr,
                neighbor_front,
            )
        else:
            left_point = right_point = None
            width_left_method = width_right_method = "unavailable"
        estimated_width = (
            float(np.linalg.norm(right_point - left_point) * 1000.0)
            if left_point is not None and right_point is not None
            else None
        )
        estimated_height = (
            float(minimum_clearance / intr.fy * neighbor_front * 1000.0)
            if neighbor_front is not None
            else None
        )
        width_mm = dims.get("book_width_mm")
        height_mm = dims.get("book_height_mm")
        width_fit = (
            None
            if width_mm is None
            else estimated_width is not None and estimated_width >= float(width_mm)
        )
        height_fit = (
            None
            if height_mm is None
            else estimated_height is not None and estimated_height >= float(height_mm)
        )
        if width_fit is False:
            rejection.append("required_book_width_not_met")
        if height_fit is False:
            rejection.append("required_book_height_not_met")
        rejection.extend(depth_result["depth_reasons"])

        if depth_result["depth_status"] == "rejected" or any(
            reason
            in {
                "clearance_margins_consume_candidate",
                "pixel_width_below_minimum",
                "right_boundary_unknown",
                "required_book_width_not_met",
                "required_book_height_not_met",
            }
            for reason in rejection
        ):
            status = "rejected"
        elif depth_result["depth_status"] == "uncertain":
            status = "uncertain"
        else:
            status = "accepted"

        candidate = {
            "candidate_id": candidate_id,
            "bbox": [x1, y_top, x2, shelf_bottom + 1],
            "x_range_raw": [x1, x2],
            "x_range_safe": [safe_x1, safe_x2],
            "pixel_width_raw": x2 - x1,
            "pixel_width_safe": max(0, safe_x2 - safe_x1),
            "estimated_width_mm": estimated_width,
            "minimum_bottom_clearance_px": minimum_clearance,
            "maximum_bottom_clearance_px": maximum_clearance,
            "estimated_height_mm": estimated_height,
            "left_boundary_type": left_type,
            "right_boundary_type": right_type,
            "left_instance_id": left["instance_id"] if left else None,
            "right_instance_id": right["instance_id"] if right else None,
            "right_boundary_x_at_target": right_boundary_x,
            "right_boundary_method": right_boundary_method,
            "right_boundary_line": (
                {
                    "point": boundary_line["point"],
                    "direction": boundary_line["direction"],
                    "point_count": boundary_line["point_count"],
                }
                if boundary_line is not None
                else None
            ),
            "target_y_rule": target_y_rule,
            "target_x_unclamped": target_x_unclamped,
            "target_clamped_to_safe_range": target_clamped,
            "target_pixel": target_pixel,
            "target_point_camera_m": (
                target_point.astype(float).tolist() if target_point is not None else None
            ),
            "target_plane_source": target_plane_source,
            "target_3d_method": target_3d_method,
            "left_front_plane": {
                "ransac_ok": left_plane["ransac_ok"],
                "point_count": left_plane["point_count"],
                "inlier_count": left_plane["inlier_count"],
                "median_depth_m": left_plane["median_depth_m"],
            },
            "right_front_plane": {
                "ransac_ok": right_plane["ransac_ok"],
                "point_count": right_plane["point_count"],
                "inlier_count": right_plane["inlier_count"],
                "median_depth_m": right_plane["median_depth_m"],
            },
            "width_3d_method": (
                f"{width_left_method}+{width_right_method}"
                if estimated_width is not None
                else "unavailable"
            ),
            "width_fit": width_fit,
            "height_fit": height_fit,
            "status": status,
            "rejection_reasons": list(dict.fromkeys(rejection)),
            **depth_result,
        }
        candidate["depth_confidence"] = _candidate_confidence(candidate)
        candidates.append(candidate)
    return candidates


def _extract_candidates(
    candidate_shapes: list[dict[str, Any]],
    required_height: Optional[int],
    roi: list[int],
    shelf_bottom: int,
    instances: list[dict[str, Any]],
    depth_raw: np.ndarray,
    intr: CameraIntrinsics,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    gap_cfg = config["gap_detection"]
    dims = config.get("book_dimensions_mm", {})
    front_cfg = config["front_plane"]
    width_percentile = float(config["candidate_width"]["safe_percentile"])
    candidates: list[dict[str, Any]] = []

    for candidate_id, shape in enumerate(candidate_shapes):
        candidate_mask = shape["candidate_mask"].astype(bool)
        bbox = _mask_bbox(candidate_mask)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        touches_left = x1 <= roi[0]
        touches_right = x2 >= roi[2]
        left, right, left_type, right_type = _neighbor_instances_for_mask(
            candidate_mask,
            instances,
            roi,
            int(gap_cfg["boundary_search_max_distance_px"]),
        )
        left_plane = _fit_instance_plane(left, depth_raw, intr, config)
        right_plane = _fit_instance_plane(right, depth_raw, intr, config)
        neighbor_depths = [
            info["median_depth_m"]
            for info in (left_plane, right_plane)
            if info["median_depth_m"] is not None
        ]
        neighbor_front = (
            float(min(neighbor_depths)) if neighbor_depths else None
        )
        fallback_front = (
            neighbor_front
            if neighbor_front is not None
            else float(front_cfg["fallback_front_depth_m"])
        )

        width_info = _compute_candidate_width_profile(
            candidate_mask,
            shelf_bottom,
            required_height,
            roi,
            width_percentile,
        )
        contour = _largest_contour_points(candidate_mask)
        bottom_band_top = max(
            roi[1],
            int(shelf_bottom) - int(gap_cfg["bottom_contact_band_px"]) + 1,
        )
        bottom_contact_width = int(
            np.count_nonzero(
                candidate_mask[bottom_band_top : int(shelf_bottom) + 1].any(axis=0)
            )
        )

        required_width_px = None
        if dims.get("book_width_mm") is not None and neighbor_front is not None:
            required_width_px = (
                float(dims["book_width_mm"]) / 1000.0 * intr.fx / neighbor_front
            )
        width_fit = (
            None
            if required_width_px is None
            else width_info["minimum"] >= required_width_px
            and width_info["safe_percentile"] >= required_width_px
        )
        height_fit = (
            None
            if required_height is None
            else width_info["height"] >= int(required_height)
        )

        regions = _build_candidate_depth_regions(
            candidate_mask,
            left,
            right,
            shelf_bottom,
            required_height,
            config,
        )
        depth_result = _classify_candidate_depth(
            regions,
            left_plane,
            right_plane,
            fallback_front,
            depth_raw,
            intr,
            required_height,
            required_width_px,
            config,
        )
        if left_plane["ransac_ok"] and right_plane["ransac_ok"]:
            depth_result["front_reference_source"] = "both_book_planes"
        elif left_plane["ransac_ok"]:
            depth_result["front_reference_source"] = "left_book_plane"
        elif right_plane["ransac_ok"]:
            depth_result["front_reference_source"] = "right_book_plane"
        elif neighbor_depths:
            depth_result["front_reference_source"] = "book_median_depth"
        else:
            depth_result["front_reference_source"] = "fallback_depth"

        if required_height is not None:
            target_y = int(
                round(
                    shelf_bottom
                    - float(config["target"]["target_height_ratio"])
                    * required_height
                )
            )
            target_y_rule = "shelf_bottom_minus_required_height_ratio"
        else:
            target_y = int(round((y1 + min(y2 - 1, shelf_bottom)) / 2))
            target_y_rule = "candidate_vertical_midpoint"
        target_y = max(roi[1], min(min(roi[3] - 1, shelf_bottom), target_y))

        candidate_row_xs = np.flatnonzero(candidate_mask[target_y])
        candidate_right_x = (
            int(candidate_row_xs.max()) if candidate_row_xs.size else None
        )
        boundary_line = None
        if right is not None:
            boundary_line = _fit_left_boundary_line(
                right,
                max(roi[1], shelf_bottom - (required_height or width_info["height"]) + 1),
                shelf_bottom,
                int(config["target"]["boundary_fit_min_points"]),
            )
            if boundary_line is not None:
                right_boundary_x = _evaluate_line_x(boundary_line, target_y)
                right_boundary_method = "right_book_left_contour_svd"
            else:
                right_boundary_x = float(right["bbox"][0])
                right_boundary_method = "right_book_bbox_left_fallback"
        elif right_type == "roi_edge" and candidate_right_x is not None:
            right_boundary_x = float(candidate_right_x)
            right_boundary_method = "candidate_right_contour_at_roi_edge"
        else:
            right_boundary_x = None
            right_boundary_method = "unavailable"

        right_boundary_pixel = (
            [int(round(right_boundary_x)), target_y]
            if right_boundary_x is not None
            else None
        )
        nominal_target_pixel = (
            [
                int(round(right_boundary_x - int(config["target"]["target_inset_px"]))),
                target_y,
            ]
            if right_boundary_x is not None
            else None
        )
        safe_target_pixel, target_adjustment, adjustment_reason = (
            _select_safe_target_pixel(
                nominal_target_pixel,
                regions["candidate_depth_mask"],
                depth_result["significant_obstacle_mask"],
            )
        )
        boundary_disagreement = (
            float(abs(right_boundary_x - candidate_right_x))
            if right_boundary_x is not None and candidate_right_x is not None
            else None
        )

        if right is not None:
            target_plane_info = right_plane
            target_plane_source = "right_book"
        elif left is not None:
            target_plane_info = left_plane
            target_plane_source = "left_book"
        else:
            target_plane_info = {"plane": None, "median_depth_m": None}
            target_plane_source = "fallback_depth"
        target_point = None
        target_3d_method = "unavailable"
        if safe_target_pixel is not None:
            target_point, target_3d_method = _point_on_front(
                (safe_target_pixel[0], safe_target_pixel[1]),
                target_plane_info,
                intr,
                fallback_front,
            )
            if target_point is not None:
                target_point = target_point.copy()
                target_point[2] -= float(config["target"]["surface_clearance_m"])

        estimated_width = (
            float(width_info["safe_percentile"] / intr.fx * neighbor_front * 1000.0)
            if neighbor_front is not None
            else None
        )
        estimated_height = (
            float(width_info["height"] / intr.fy * neighbor_front * 1000.0)
            if neighbor_front is not None
            else None
        )

        rejection: list[str] = []
        uncertain: list[str] = list(depth_result["depth_uncertain_reasons"])
        if width_info["minimum"] < int(gap_cfg["minimum_candidate_width_px"]):
            rejection.append("pixel_width_below_minimum")
        if width_fit is False:
            rejection.append("required_book_width_not_met")
        if height_fit is False:
            rejection.append("required_book_height_not_met")
        if safe_target_pixel is None:
            rejection.append("safe_target_unavailable")
        if left is None and right is None:
            uncertain.append("candidate_has_no_book_boundary")
        if boundary_disagreement is not None:
            if boundary_disagreement > float(
                config["target"]["reject_boundary_disagreement_px"]
            ):
                rejection.append("boundary_disagreement_too_large")
            elif boundary_disagreement > float(
                config["target"]["max_boundary_disagreement_px"]
            ):
                uncertain.append("boundary_disagreement_large")
        if target_adjustment is not None:
            if target_adjustment > float(
                config["target"]["reject_safe_target_adjustment_px"]
            ):
                rejection.append("safe_target_adjustment_too_large")
            elif target_adjustment > float(
                config["target"]["max_safe_target_adjustment_px"]
            ):
                uncertain.append("safe_target_adjustment_large")
        rejection.extend(depth_result["depth_reasons"])

        if rejection:
            status = "rejected"
        elif uncertain or depth_result["depth_status"] == "uncertain":
            status = "uncertain"
        else:
            status = "accepted"

        candidate = {
            "candidate_id": candidate_id,
            "candidate_mask": candidate_mask,
            "candidate_depth_mask": regions["candidate_depth_mask"],
            "boundary_band_mask": regions["boundary_band_mask"],
            "candidate_core_mask": regions["candidate_core_mask"],
            "deep_space_mask": depth_result["deep_space_mask"],
            "boundary_surface_mask": depth_result["boundary_surface_mask"],
            "interior_obstacle_mask": depth_result["interior_obstacle_mask"],
            "significant_obstacle_mask": depth_result[
                "significant_obstacle_mask"
            ],
            "unknown_depth_mask": depth_result["unknown_depth_mask"],
            "candidate_mask_npz_key": f"candidate_{candidate_id}",
            "candidate_depth_mask_npz_key": f"candidate_{candidate_id}",
            "candidate_bbox": bbox,
            "bbox": bbox,
            "candidate_contour_px": contour,
            "candidate_area_px": int(candidate_mask.sum()),
            "candidate_height_px": width_info["height"],
            "candidate_bottom_contact_width_px": bottom_contact_width,
            "candidate_width_profile_px": width_info["profile"],
            "candidate_min_width_px": width_info["minimum"],
            "candidate_safe_percentile_width_px": width_info["safe_percentile"],
            "candidate_max_width_px": width_info["maximum"],
            "candidate_min_width_row": width_info["minimum_row"],
            "touches_roi_left": touches_left,
            "touches_roi_right": touches_right,
            "left_boundary_type": left_type,
            "right_boundary_type": right_type,
            "left_instance_id": left["instance_id"] if left else None,
            "right_instance_id": right["instance_id"] if right else None,
            "right_boundary_method": right_boundary_method,
            "right_boundary_line": (
                {
                    "point": boundary_line["point"],
                    "direction": boundary_line["direction"],
                    "point_count": boundary_line["point_count"],
                }
                if boundary_line is not None
                else None
            ),
            "right_boundary_pixel": right_boundary_pixel,
            "candidate_right_pixel_at_target": (
                [candidate_right_x, target_y]
                if candidate_right_x is not None
                else None
            ),
            "boundary_disagreement_px": boundary_disagreement,
            "nominal_target_pixel": nominal_target_pixel,
            "safe_target_pixel": safe_target_pixel,
            "safe_target_adjustment_px": target_adjustment,
            "safe_target_adjustment_reason": adjustment_reason,
            "target_pixel": safe_target_pixel,
            "target_y_rule": target_y_rule,
            "target_point_camera_m": (
                target_point.astype(float).tolist() if target_point is not None else None
            ),
            "target_plane_source": target_plane_source,
            "target_3d_method": target_3d_method,
            "estimated_width_mm": estimated_width,
            "estimated_height_mm": estimated_height,
            "width_fit": width_fit,
            "height_fit": height_fit,
            "left_front_plane": {
                "ransac_ok": left_plane["ransac_ok"],
                "point_count": left_plane["point_count"],
                "inlier_count": left_plane["inlier_count"],
                "median_depth_m": left_plane["median_depth_m"],
                "front_region_source": left_plane["front_region_source"],
            },
            "right_front_plane": {
                "ransac_ok": right_plane["ransac_ok"],
                "point_count": right_plane["point_count"],
                "inlier_count": right_plane["inlier_count"],
                "median_depth_m": right_plane["median_depth_m"],
                "front_region_source": right_plane["front_region_source"],
            },
            "left_front_plane_ok": left_plane["ransac_ok"],
            "right_front_plane_ok": right_plane["ransac_ok"],
            "candidate_depth_erosion_fallback": regions[
                "candidate_depth_erosion_fallback"
            ],
            "depth_evaluation_row_range": regions[
                "depth_evaluation_row_range"
            ],
            "status": status,
            "rejection_reasons": list(dict.fromkeys(rejection)),
            "uncertain_reasons": list(dict.fromkeys(uncertain)),
            **{
                key: value
                for key, value in depth_result.items()
                if key
                not in {
                    "deep_space_mask",
                    "boundary_surface_mask",
                    "interior_obstacle_mask",
                    "unknown_depth_mask",
                    "significant_obstacle_mask",
                    "depth_reasons",
                    "depth_uncertain_reasons",
                }
            },
        }
        candidate["depth_confidence"] = _candidate_confidence(candidate)
        candidates.append(candidate)
    return candidates


def _select_candidate(
    candidates: list[dict[str, Any]], strategy: str
) -> Optional[dict[str, Any]]:
    eligible = [
        item
        for item in candidates
        if item["status"] == "accepted"
        and item.get("target_point_camera_m") is not None
        and item.get("right_boundary_type") in {"book", "roi_edge"}
    ]
    if not eligible:
        return None
    if strategy == "rightmost":
        return max(eligible, key=lambda item: item["candidate_bbox"][2])
    if strategy == "leftmost":
        return min(eligible, key=lambda item: item["candidate_bbox"][0])
    if strategy == "widest":
        return max(
            eligible,
            key=lambda item: (
                item["estimated_width_mm"]
                if item["estimated_width_mm"] is not None
                else item["candidate_safe_percentile_width_px"]
            ),
        )
    if strategy == "highest_confidence":
        return max(
            eligible,
            key=lambda item: (
                item["depth_confidence"],
                item["candidate_safe_percentile_width_px"],
            ),
        )
    raise PipelineFailure(f"Unknown candidate selection strategy: {strategy}")


def _overlay_instances(
    rgb: np.ndarray, masks: Iterable[np.ndarray], alpha: float = 0.38
) -> np.ndarray:
    output = rgb.copy()
    colors = (
        (32, 80, 255),
        (64, 220, 64),
        (255, 128, 32),
        (220, 64, 220),
        (64, 220, 220),
    )
    for index, mask in enumerate(masks):
        color = np.asarray(colors[index % len(colors)], dtype=np.float32)
        output[mask] = (
            (1.0 - alpha) * output[mask].astype(np.float32) + alpha * color
        ).astype(np.uint8)
    return output


def _draw_roi(image: np.ndarray, roi: Optional[list[int]], label: str, color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    if roi is not None:
        cv2.rectangle(output, (roi[0], roi[1]), (roi[2] - 1, roi[3] - 1), color, 2)
        cv2.putText(
            output,
            label,
            (roi[0] + 4, max(18, roi[1] + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def _depth_colormap(
    depth_raw: np.ndarray, intr: CameraIntrinsics, config: dict[str, Any]
) -> np.ndarray:
    valid = _valid_depth_mask(depth_raw, intr, config)
    output = np.zeros((*depth_raw.shape, 3), dtype=np.uint8)
    if np.any(valid):
        depth_m = depth_raw.astype(np.float32) * intr.depth_scale
        low, high = np.percentile(depth_m[valid], [2, 98])
        normalized = np.clip((depth_m - low) / max(1e-6, high - low), 0, 1)
        colored = cv2.applyColorMap(
            (normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
        )
        output[valid] = colored[valid]
    return output


def _clearance_visual(
    rgb: np.ndarray, profile: np.ndarray, roi: list[int], required: Optional[int]
) -> np.ndarray:
    output = rgb.copy()
    x1, y1, x2, y2 = roi
    maximum = max(1, y2 - y1)
    for x in range(x1, x2):
        height = int(min(maximum, profile[x]))
        color = (0, 220, 0) if required is not None and height >= required else (0, 180, 255)
        output[max(y1, y2 - height) : y2, x] = (
            0.70 * output[max(y1, y2 - height) : y2, x]
            + 0.30 * np.asarray(color)
        ).astype(np.uint8)
    return output


def _draw_candidates(
    rgb: np.ndarray,
    candidates: list[dict[str, Any]],
    after_depth: bool,
    *,
    draw_bbox: bool = True,
    draw_minimum_width: bool = True,
) -> np.ndarray:
    output = rgb.copy()
    status_colors = {
        "accepted": (0, 220, 0),
        "rejected": (0, 0, 255),
        "uncertain": (0, 180, 255),
    }
    for item in candidates:
        color = (
            status_colors[item["status"]] if after_depth else (255, 180, 0)
        )
        mask = item.get("candidate_mask")
        if mask is not None:
            color_array = np.asarray(color, dtype=np.float32)
            output[mask] = (
                0.65 * output[mask].astype(np.float32) + 0.35 * color_array
            ).astype(np.uint8)
        x1, y1, x2, y2 = item["bbox"]
        if draw_bbox:
            cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), color, 2)
        contour = np.asarray(item.get("candidate_contour_px", []), dtype=np.int32)
        if contour.size:
            cv2.polylines(output, [contour.reshape(-1, 1, 2)], True, color, 2)
        minimum_row = item.get("candidate_min_width_row")
        if (
            draw_minimum_width
            and minimum_row
            and minimum_row["left_x"] >= 0
        ):
            cv2.line(
                output,
                (int(minimum_row["left_x"]), int(minimum_row["y"])),
                (int(minimum_row["right_x"]), int(minimum_row["y"])),
                (255, 255, 255),
                3,
            )
        cv2.putText(
            output,
            f"C{item['candidate_id']} {item['status'] if after_depth else ''}",
            (x1 + 2, max(18, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def _draw_final_safe_target(
    rgb: np.ndarray,
    selected: Optional[dict[str, Any]],
    *,
    include_boundary_line: bool,
) -> np.ndarray:
    """Draw the final-use target as one small red point."""
    output = rgb.copy()
    if selected is None:
        cv2.putText(
            output,
            "NO ACCEPTED TARGET",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    line = selected.get("right_boundary_line")
    if include_boundary_line and line is not None:
        point = np.asarray(line["point"], dtype=float)
        direction = np.asarray(line["direction"], dtype=float)
        if abs(direction[1]) > 1e-8:
            ys = [selected["bbox"][1], selected["bbox"][3] - 1]
            xs = [
                int(
                    round(
                        point[0]
                        + (y - point[1]) * direction[0] / direction[1]
                    )
                )
                for y in ys
            ]
            cv2.line(
                output,
                (xs[0], ys[0]),
                (xs[1], ys[1]),
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

    safe_target = selected.get("safe_target_pixel")
    if safe_target is not None:
        cv2.circle(
            output,
            (int(safe_target[0]), int(safe_target[1])),
            4,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )
    return output


def _draw_target(
    rgb: np.ndarray, selected: Optional[dict[str, Any]], include_boundary: bool
) -> np.ndarray:
    output = rgb.copy()
    if selected is None:
        cv2.putText(
            output,
            "NO ACCEPTED TARGET",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output
    line = selected.get("right_boundary_line")
    if include_boundary and line is not None:
        point = np.asarray(line["point"], dtype=float)
        direction = np.asarray(line["direction"], dtype=float)
        if abs(direction[1]) > 1e-8:
            ys = [selected["bbox"][1], selected["bbox"][3] - 1]
            xs = [
                int(round(point[0] + (y - point[1]) * direction[0] / direction[1]))
                for y in ys
            ]
            cv2.line(output, (xs[0], ys[0]), (xs[1], ys[1]), (255, 0, 255), 3)
    for key, color, marker in (
        ("right_boundary_pixel", (255, 0, 255), cv2.MARKER_DIAMOND),
        ("nominal_target_pixel", (0, 255, 255), cv2.MARKER_TILTED_CROSS),
        ("safe_target_pixel", (0, 255, 0), cv2.MARKER_CROSS),
    ):
        pixel = selected.get(key)
        if pixel is None:
            continue
        cv2.drawMarker(
            output,
            (int(pixel[0]), int(pixel[1])),
            color,
            marker,
            24,
            3,
        )
    return output


def _draw_mask_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.35,
) -> np.ndarray:
    output = rgb.copy()
    output[mask] = (
        (1.0 - alpha) * output[mask].astype(np.float32)
        + alpha * np.asarray(color, dtype=np.float32)
    ).astype(np.uint8)
    return output


def _component_label_visual(labels: np.ndarray) -> np.ndarray:
    output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    colors = (
        (255, 96, 64),
        (64, 220, 64),
        (64, 160, 255),
        (220, 64, 220),
        (64, 220, 220),
        (220, 180, 64),
    )
    for label in range(1, int(labels.max()) + 1):
        output[labels == label] = colors[(label - 1) % len(colors)]
    return output


def _candidate_depth_class_visual(
    rgb: np.ndarray, candidate: Optional[dict[str, Any]]
) -> np.ndarray:
    output = rgb.copy()
    if candidate is None:
        return output
    classes = (
        ("deep_space_mask", (255, 128, 0)),
        ("boundary_surface_mask", (0, 255, 255)),
        ("interior_obstacle_mask", (0, 0, 255)),
        ("unknown_depth_mask", (128, 128, 128)),
    )
    for key, color in classes:
        mask = candidate[key]
        output[mask] = (
            0.45 * output[mask].astype(np.float32)
            + 0.55 * np.asarray(color, dtype=np.float32)
        ).astype(np.uint8)
    return output


def run_offline_shelf_storage_detection(
    *,
    rgb_path: Optional[Path],
    depth_path: Optional[Path],
    intrinsics_path: Optional[Path],
    sam3_masks_path: Optional[Path],
    config: dict[str, Any],
    output_dir: Optional[Path] = None,
    run_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run the complete saved-data pipeline and always return a compact result."""
    started = time.perf_counter()
    if output_dir is None:
        output_dir = _new_output_dir(config)
    else:
        output_dir = Path(output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        output_dir.mkdir(parents=True, exist_ok=False)
        output_dir = output_dir.resolve()
    log = RunLog(output_dir / "run.log")
    failure_stage = "input_validation"
    result: dict[str, Any] = {
        "ok": False,
        "pipeline_ok": False,
        "failure_stage": failure_stage,
        "failure_reason": None,
        "reason": None,
        "roi_source": None,
        "detected_shelf_roi": None,
        "fallback_shelf_roi": config.get("fallback_shelf_roi"),
        "adopted_shelf_roi": None,
        "target_shelf_roi": None,
        "shelf_bottom_px": None,
        "book_instances": [],
        "excluded_instances": [],
        "candidates": [],
        "selected_candidate_id": None,
        "target_pixel": None,
        "right_boundary_pixel": None,
        "nominal_target_pixel": None,
        "safe_target_pixel": None,
        "target_point_camera_m": None,
        "right_boundary_type": None,
        "right_boundary_instance_id": None,
        "target_plane_source": None,
        "estimated_gap_width_mm": None,
        "processing_time_sec": None,
        "output_dir": str(output_dir),
        "artifacts": {},
    }
    if run_metadata:
        result.update(_jsonable(run_metadata))
    candidates: list[dict[str, Any]] = []
    try:
        input_paths = {
            "rgb": str(rgb_path) if rgb_path else None,
            "depth": str(depth_path) if depth_path else None,
            "intrinsics": str(intrinsics_path) if intrinsics_path else None,
            "sam3_masks": str(sam3_masks_path) if sam3_masks_path else None,
        }
        if run_metadata:
            log.write(
                "run_metadata="
                + json.dumps(_jsonable(run_metadata), ensure_ascii=False)
            )
        log.write(f"inputs={json.dumps(input_paths, ensure_ascii=False)}")
        if rgb_path is None or not Path(rgb_path).is_file():
            raise PipelineFailure(f"RGB image not found: {rgb_path}")
        if depth_path is None or not Path(depth_path).is_file():
            raise PipelineFailure(f"aligned Depth NPY not found: {depth_path}")
        if intrinsics_path is None or not Path(intrinsics_path).is_file():
            raise PipelineFailure(f"camera intrinsics JSON not found: {intrinsics_path}")

        failure_stage = "input_loading"
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise PipelineFailure(f"RGB image could not be decoded: {rgb_path}")
        try:
            depth_raw = np.load(depth_path, allow_pickle=False)
        except Exception as exc:
            raise PipelineFailure(f"Depth NPY could not be read: {exc}") from exc
        if depth_raw.ndim != 2 or depth_raw.shape != rgb.shape[:2]:
            raise PipelineFailure(
                f"Depth shape {depth_raw.shape} != RGB shape {rgb.shape[:2]}"
            )
        intr = _load_intrinsics(Path(intrinsics_path), rgb.shape[:2])
        target_roi_mask, target_roi = _build_common_target_roi_mask(
            rgb.shape[:2], config
        )
        result["target_shelf_roi"] = target_roi
        _save_image(output_dir / "01_rgb.png", rgb)
        _save_image(output_dir / "02_depth_colormap.png", _depth_colormap(depth_raw, intr, config))
        _save_image(
            output_dir / "target_shelf_roi_mask.png",
            target_roi_mask.astype(np.uint8) * 255,
        )
        target_overlay = _draw_mask_overlay(
            rgb, target_roi_mask, (0, 255, 255), alpha=0.20
        )
        target_overlay = _draw_roi(
            target_overlay, target_roi, "common target ROI", (0, 255, 255)
        )
        _save_image(output_dir / "target_shelf_roi_overlay.png", target_overlay)

        failure_stage = "sam3_inference"
        raw_masks, supplied_boxes, scores, sam_meta = _acquire_sam3_masks(
            rgb, sam3_masks_path, output_dir, config
        )
        result["sam3"] = sam_meta
        log.write(
            f"SAM3 source={sam_meta['source']} mask_count={len(raw_masks)} "
            f"inference_time_sec={sam_meta.get('inference_time_sec')}"
        )
        _save_image(
            output_dir / "03_sam3_all_instances_raw.png",
            _overlay_instances(rgb, raw_masks),
        )
        failure_stage = "instance_filtering"
        instances, excluded = _prepare_instances(
            raw_masks,
            supplied_boxes,
            scores,
            depth_raw,
            intr,
            config,
            target_roi_mask,
        )
        result["book_instances"] = [_instance_metadata(item) for item in instances]
        result["excluded_instances"] = excluded
        log.write(
            f"mask_filter kept={len(instances)} excluded={len(excluded)} "
            f"reasons={[item['rejection_reasons'] for item in excluded]}"
        )
        if not instances:
            raise PipelineFailure("No valid book masks remain after filtering")
        filtered_masks = np.asarray([item["mask"] for item in instances], dtype=bool)
        _save_image(
            output_dir / "04_sam3_all_instances_filtered.png",
            _overlay_instances(rgb, filtered_masks),
        )
        np.savez_compressed(
            output_dir / "sam3_instances.npz",
            raw_masks=raw_masks,
            raw_boxes=supplied_boxes.astype(np.float32),
            raw_scores=scores.astype(np.float32),
            filtered_masks=filtered_masks,
            filtered_instance_ids=np.asarray(
                [item["instance_id"] for item in instances], dtype=np.int32
            ),
        )

        failure_stage = "shelf_roi"
        roi_result = _estimate_shelf_roi(
            instances, rgb.shape[:2], config, target_roi
        )
        result.update(roi_result)
        log.write(
            f"shelf_roi detected={roi_result['detected_shelf_roi']} "
            f"adopted={roi_result['adopted_shelf_roi']} "
            f"source={roi_result['roi_source']} reasons={roi_result['decision_reasons']}"
        )
        detected_visual = _draw_roi(
            rgb, roi_result["detected_shelf_roi"], "detected shelf ROI", (255, 0, 255)
        )
        _save_image(output_dir / "05_detected_shelf_roi.png", detected_visual)
        adopted_visual = _draw_roi(
            rgb,
            roi_result["adopted_shelf_roi"],
            f"adopted ROI ({roi_result['roi_source']})",
            (0, 255, 255),
        )
        _save_image(output_dir / "06_adopted_shelf_roi.png", adopted_visual)

        failure_stage = "occupancy_and_free_space"
        raw_occ, safe_occ, occ_diff, free_raw, free_safe = _build_occupancy(
            instances, rgb.shape[:2], target_roi_mask, config
        )
        _save_image(output_dir / "07_occupied_mask_raw.png", raw_occ.astype(np.uint8) * 255)
        _save_image(output_dir / "08_occupied_mask_safe.png", safe_occ.astype(np.uint8) * 255)
        _save_image(output_dir / "09_occupied_mask_difference.png", occ_diff.astype(np.uint8) * 255)
        _save_image(output_dir / "10_free_mask_raw.png", free_raw.astype(np.uint8) * 255)
        _save_image(output_dir / "11_free_mask_safe.png", free_safe.astype(np.uint8) * 255)
        np.savez_compressed(
            output_dir / "occupied_masks.npz",
            occupied_mask_raw=raw_occ,
            occupied_mask_safe=safe_occ,
            difference_mask=occ_diff,
        )
        np.savez_compressed(
            output_dir / "free_masks.npz",
            free_mask_raw=free_raw,
            free_mask_safe=free_safe,
        )
        log.write(
            "occupied_refinement="
            + json.dumps(config["occupied_mask_refinement"], ensure_ascii=False)
        )

        front_depth = _representative_front_depth(
            instances, depth_raw, intr, config
        )
        required_height, required_rule = _required_height_px(
            config, intr, front_depth
        )
        result["required_height_px"] = required_height
        result["required_height_rule"] = required_rule
        result["representative_front_depth_m"] = front_depth
        bottom_connected, component_labels, profile = (
            _extract_bottom_connected_free_mask(
            free_safe,
            roi_result["adopted_shelf_roi"],
            roi_result["shelf_bottom_px"],
                int(config["gap_detection"]["bottom_contact_band_px"]),
            )
        )
        _save_image(
            output_dir / "bottom_connected_free_mask.png",
            bottom_connected.astype(np.uint8) * 255,
        )
        _save_image(
            output_dir / "bottom_connected_component_labels.png",
            _component_label_visual(component_labels),
        )
        _save_image(
            output_dir / "12_bottom_clearance_profile.png",
            _clearance_visual(
                rgb, profile, roi_result["adopted_shelf_roi"], required_height
            ),
        )

        failure_stage = "candidate_extraction"
        candidate_shapes = _build_candidate_shape_masks(
            bottom_connected,
            component_labels,
            profile,
            required_height,
            roi_result["adopted_shelf_roi"],
            config,
        )
        failure_stage = "candidate_depth_evaluation"
        candidates = _extract_candidates(
            candidate_shapes,
            required_height,
            roi_result["adopted_shelf_roi"],
            roi_result["shelf_bottom_px"],
            instances,
            depth_raw,
            intr,
            config,
        )
        result["candidates"] = [_candidate_metadata(item) for item in candidates]
        candidate_mask_arrays = {
            item["candidate_mask_npz_key"]: item["candidate_mask"]
            for item in candidates
        }
        candidate_depth_arrays: dict[str, np.ndarray] = {}
        candidate_class_arrays: dict[str, np.ndarray] = {}
        for item in candidates:
            key = item["candidate_depth_mask_npz_key"]
            candidate_depth_arrays[f"{key}_depth"] = item["candidate_depth_mask"]
            candidate_depth_arrays[f"{key}_boundary"] = item["boundary_band_mask"]
            candidate_depth_arrays[f"{key}_core"] = item["candidate_core_mask"]
            candidate_class_arrays[f"{key}_deep_space"] = item["deep_space_mask"]
            candidate_class_arrays[f"{key}_boundary_surface"] = item[
                "boundary_surface_mask"
            ]
            candidate_class_arrays[f"{key}_interior_obstacle"] = item[
                "interior_obstacle_mask"
            ]
            candidate_class_arrays[f"{key}_unknown"] = item["unknown_depth_mask"]
        np.savez_compressed(
            output_dir / "candidate_masks.npz", **candidate_mask_arrays
        )
        np.savez_compressed(
            output_dir / "candidate_depth_masks.npz", **candidate_depth_arrays
        )
        np.savez_compressed(
            output_dir / "candidate_depth_classes.npz", **candidate_class_arrays
        )
        _save_image(
            output_dir / "13_gap_candidates_before_depth.png",
            _draw_candidates(rgb, candidates, after_depth=False),
        )
        _save_image(
            output_dir / "14_gap_candidates_after_depth.png",
            _draw_candidates(rgb, candidates, after_depth=True),
        )
        _save_image(
            output_dir / "candidate_masks_overlay.png",
            _draw_candidates(rgb, candidates, after_depth=True),
        )
        _save_image(
            output_dir / "candidate_width_profile_overlay.png",
            _draw_candidates(rgb, candidates, after_depth=True),
        )
        log.write(f"candidate_count={len(candidates)}")
        for item in candidates:
            log.write(
                f"candidate={item['candidate_id']} status={item['status']} "
                f"depth={item['depth_status']} reasons={item['rejection_reasons']} "
                f"RANSAC_left={item['left_front_plane']['ransac_ok']} "
                f"RANSAC_right={item['right_front_plane']['ransac_ok']}"
            )
        if not candidates:
            raise PipelineFailure("No bottom-connected gap candidates were found")

        failure_stage = "candidate_selection"
        strategy = config["candidate_selection"]["strategy"]
        selected = _select_candidate(candidates, strategy)
        debug_candidate = selected or (
            max(candidates, key=lambda item: item["depth_confidence"])
            if candidates
            else None
        )
        empty_mask = np.zeros(rgb.shape[:2], dtype=bool)
        selected_mask = (
            debug_candidate["candidate_mask"]
            if debug_candidate is not None
            else empty_mask
        )
        _save_image(
            output_dir / "selected_candidate_mask.png",
            selected_mask.astype(np.uint8) * 255,
        )
        _save_image(
            output_dir / "candidate_boundary_band.png",
            (
                debug_candidate["boundary_band_mask"]
                if debug_candidate is not None
                else empty_mask
            ).astype(np.uint8)
            * 255,
        )
        _save_image(
            output_dir / "candidate_core_mask.png",
            (
                debug_candidate["candidate_core_mask"]
                if debug_candidate is not None
                else empty_mask
            ).astype(np.uint8)
            * 255,
        )
        _save_image(
            output_dir / "candidate_depth_classification.png",
            _candidate_depth_class_visual(rgb, debug_candidate),
        )
        _save_image(
            output_dir / "candidate_core_obstacles.png",
            (
                debug_candidate["significant_obstacle_mask"]
                if debug_candidate is not None
                else empty_mask
            ).astype(np.uint8)
            * 255,
        )
        _save_image(
            output_dir / "15_right_boundary_line.png",
            _draw_target(rgb, selected, include_boundary=True),
        )
        _save_image(
            output_dir / "16_target_pixel.png",
            _draw_target(rgb, selected, include_boundary=False),
        )
        _save_image(
            output_dir / "right_boundary_nominal_safe_target.png",
            _draw_target(
                _draw_candidates(rgb, candidates, after_depth=True),
                debug_candidate,
                include_boundary=True,
            ),
        )
        final = _draw_candidates(
            rgb,
            candidates,
            after_depth=True,
            draw_bbox=False,
            draw_minimum_width=False,
        )
        final = _draw_final_safe_target(
            final,
            selected,
            include_boundary_line=True,
        )
        _save_image(output_dir / "17_final_result.png", final)
        _save_image(output_dir / "final_result_candidate_mask.png", final)
        if selected is None:
            statuses = {item["status"] for item in candidates}
            if statuses == {"uncertain"} or "uncertain" in statuses:
                raise PipelineFailure(
                    "No accepted candidate: Depth confidence is insufficient"
                )
            if all(item.get("width_fit") is False for item in candidates):
                raise PipelineFailure("No candidate satisfies the required book width")
            raise PipelineFailure("No candidate passed Depth and boundary checks")

        result.update(
            {
                "ok": True,
                "reason": None,
                "selected_candidate_id": selected["candidate_id"],
                "target_pixel": selected["target_pixel"],
                "right_boundary_pixel": selected["right_boundary_pixel"],
                "nominal_target_pixel": selected["nominal_target_pixel"],
                "safe_target_pixel": selected["safe_target_pixel"],
                "target_point_camera_m": selected["target_point_camera_m"],
                "right_boundary_type": selected["right_boundary_type"],
                "right_boundary_instance_id": selected["right_instance_id"],
                "target_plane_source": selected["target_plane_source"],
                "estimated_gap_width_mm": selected["estimated_width_mm"],
            }
        )
        failure_stage = "complete"
        log.write(
            f"selected={selected['candidate_id']} strategy={strategy} "
            f"right_boundary={selected['right_boundary_method']} "
            f"target_pixel={selected['target_pixel']} "
            f"target_point_camera_m={selected['target_point_camera_m']} "
            f"target_plane_source={selected['target_plane_source']}"
        )
    except PipelineFailure as exc:
        result["ok"] = False
        result["reason"] = str(exc)
        log.write(f"pipeline_failure={exc}")
    except Exception as exc:
        result["ok"] = False
        result["reason"] = f"unexpected_error: {type(exc).__name__}: {exc}"
        log.write(result["reason"])
    finally:
        result["processing_time_sec"] = time.perf_counter() - started
        result["pipeline_ok"] = bool(result["ok"])
        result["failure_stage"] = None if result["ok"] else failure_stage
        result["failure_reason"] = None if result["ok"] else result["reason"]
        artifact_names = (
            "01_rgb.png",
            "02_depth_colormap.png",
            "03_sam3_all_instances_raw.png",
            "04_sam3_all_instances_filtered.png",
            "05_detected_shelf_roi.png",
            "06_adopted_shelf_roi.png",
            "07_occupied_mask_raw.png",
            "08_occupied_mask_safe.png",
            "09_occupied_mask_difference.png",
            "10_free_mask_raw.png",
            "11_free_mask_safe.png",
            "12_bottom_clearance_profile.png",
            "13_gap_candidates_before_depth.png",
            "14_gap_candidates_after_depth.png",
            "15_right_boundary_line.png",
            "16_target_pixel.png",
            "17_final_result.png",
            "target_shelf_roi_mask.png",
            "target_shelf_roi_overlay.png",
            "bottom_connected_free_mask.png",
            "bottom_connected_component_labels.png",
            "candidate_masks_overlay.png",
            "selected_candidate_mask.png",
            "candidate_width_profile_overlay.png",
            "candidate_boundary_band.png",
            "candidate_core_mask.png",
            "candidate_depth_classification.png",
            "candidate_core_obstacles.png",
            "right_boundary_nominal_safe_target.png",
            "final_result_candidate_mask.png",
            "sam3_instances.npz",
            "sam3_service_masks.npz",
            "sam3_all_masks_overlay.png",
            "sam3_service_inference.json",
            "occupied_masks.npz",
            "free_masks.npz",
            "candidate_masks.npz",
            "candidate_depth_masks.npz",
            "candidate_depth_classes.npz",
            "candidates.json",
            "summary.json",
            "run.log",
        )
        result["artifacts"] = {
            name: str(output_dir / name)
            for name in artifact_names
            if (output_dir / name).exists() or name in {"candidates.json", "summary.json"}
        }
        candidate_records = [_candidate_metadata(item) for item in candidates]
        result["candidates"] = candidate_records
        _save_json(output_dir / "candidates.json", candidate_records)
        _save_json(output_dir / "summary.json", result)
        log.write(
            f"complete ok={result['ok']} reason={result['reason']} "
            f"processing_time_sec={result['processing_time_sec']:.6f}"
        )
        # Rewrite summary once so the final run.log path and elapsed time are reflected.
        result["artifacts"]["run.log"] = str(output_dir / "run.log")
        result["artifacts"]["summary.json"] = str(output_dir / "summary.json")
        result["artifacts"]["candidates.json"] = str(output_dir / "candidates.json")
        _save_json(output_dir / "summary.json", result)
    return _jsonable(result)


__all__ = [
    "CameraIntrinsics",
    "fit_plane_ransac",
    "intersect_pixel_ray_with_plane",
    "load_config",
    "resolve_shot_inputs",
    "run_offline_shelf_storage_detection",
]
