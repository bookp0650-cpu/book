#!/usr/bin/env python3
"""Estimate 2D book-storage-space candidates on saved 100test RGB images.

The pipeline intentionally implements only the first, inspectable baseline:

1. infer all ``book spine`` masks with the retrieval checkpoint;
2. infer all ``book end`` masks with the separate model-only checkpoint;
3. choose one book end on each side of the detected spine group;
4. define an inner-edge horizontal ROI and an IQR-filtered spine-height ROI;
5. subtract the actual spine masks from that ROI; and
6. split residuals by adjacent obstacle pairs, then retain one local component
   per pair in its original mask shape.

There are no RealSense, ROS, robot, depth, point-cloud, insertion-target, or
multi-view imports in this offline-only script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = PROJECT_ROOT / "captures" / "100test"
OUTPUT_ROOT = PROJECT_ROOT / "captures" / "100test_storage_space_offline"
SAM3_RUNTIME = PROJECT_ROOT / "detection" / "pro_handbook" / "sam3_runtime"
SAM3_SOURCE = SAM3_RUNTIME / "sam3_source"
OWNER_REPO = SAM3_RUNTIME / "vendor" / "owner_repo"
SPINE_CHECKPOINT = SAM3_RUNTIME / "models" / "inference_best.pt"
BOOK_END_CHECKPOINT = SAM3_RUNTIME / "models" / "checkpoint_50_modelonly.pt"

SPINE_PROMPT = "book spine"
BOOK_END_PROMPT = "book end"
EXPECTED_SPINE_SHA256 = (
    "d8b297b0a9a8a81c7926541a0f8fb08f7a15ee7d53d210b9827190aa21b16bce"
)
EXPECTED_BOOK_END_SHA256 = (
    "a1a0c32fc6f2e2f9d10612ac55d6e851566b0a26c8dc2d83b9e6ae0ac1a3e83c"
)

# Existing SAM3 inference defaults.  Keep both checkpoint comparisons aligned.
PROCESSOR_CONFIDENCE_THRESHOLD = 0.05
SAM3_SCORE_THRESHOLD = 0.30
SAM3_MIN_MASK_AREA_PX = 200
SAM3_NMS_IOU_THRESHOLD = 0.50

# A spine contributes to vertical ROI bounds only when at least this fraction of
# its actual mask pixels lies inside the horizontal book-end ROI.
SPINE_MIN_MASK_AREA_FRACTION_INSIDE_HORIZONTAL_ROI = 0.50

# Tukey's IQR fences are applied independently to spine y_top and y_bottom.
VERTICAL_ROI_OUTLIER_IQR_MULTIPLIER = 2.00

# Intentionally permissive first-pass space filtering.  Bottom reach is not used
# to reject extracted candidates, but is required by final-target selection.
SPACE_MIN_AREA_PX = 500
SPACE_MIN_WIDTH_PX = 5
SPACE_MIN_HEIGHT_PX = 5
ROI_BOTTOM_TOLERANCE_PX = 20
ROI_BOTTOM_TOLERANCE_RATIO = 0.05
CONNECTED_COMPONENT_CONNECTIVITY = 8

# The grasped book's real-world right side appears on the image-left side of
# its SAM3 book-spine mask.  Keep the front mask occupied and reserve exactly
# this many pixels to the left of each real per-row mask boundary.
HELD_BOOK_LEFT_OCCLUSION_WIDTH_PX = 45

# Evaluate the actual continuous opening at the physical book-bottom side
# (the image-top band), independently from the held-book occlusion width.
MIN_STORAGE_SPACE_WIDTH_PX = 30

# In saved images, smaller y is the physical bottom side of a book and larger y
# is the physical top side.  Final selection evaluates each candidate's own
# image-top/image-bottom bands with that mapping made explicit.
CANDIDATE_SHAPE_BAND_RATIO = 0.20
FINAL_SELECTION_MIN_HEIGHT_PX = 30
FINAL_SELECTION_MIN_HEIGHT_RATIO = 0.10

# A boundary may be extended only through a short, continuous missing tail at
# the ROI bottom.  The estimate is fitted from real mask-boundary points; bbox
# edges and center_x are deliberately never used as fallbacks.
BOUNDARY_EXTRAPOLATION_FIT_ROWS = 40
BOUNDARY_EXTRAPOLATION_MIN_FIT_ROWS = 20
BOUNDARY_EXTRAPOLATION_MAX_DISTANCE_RATIO = 0.15
BOUNDARY_EXTRAPOLATION_MAD_MULTIPLIER = 3.5
BOUNDARY_EXTRAPOLATION_COLLISION_MARGIN_PX = 1

MASK_ALPHA = 0.40
SPACE_ALPHA = 0.45
FINAL_SPACE_COLOR_RGB = (255, 0, 0)
MASK_COLORS_RGB = (
    (255, 64, 64),
    (64, 220, 96),
    (64, 128, 255),
    (255, 192, 64),
    (192, 64, 255),
    (64, 224, 224),
)
SPACE_COLORS_RGB = (
    (255, 80, 80),
    (80, 220, 100),
    (80, 140, 255),
    (255, 200, 70),
    (210, 80, 255),
    (70, 225, 225),
    (255, 120, 200),
    (160, 230, 70),
)
ROI_COLOR_RGB = (255, 255, 0)


for import_path in (OWNER_REPO, SAM3_SOURCE):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from core.checkpoint_export import _fresh_model  # noqa: E402
from core.mask_nms import apply_mask_nms  # noqa: E402
from core.npz_io import InstanceSet, empty_instances, load_npz, save_npz  # noqa: E402
from core.sam3_adapter import Sam3Adapter  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    role: str
    checkpoint: Path
    expected_sha256: str
    prompt: str
    checkpoint_format: str
    npz_name: str
    overlay_name: str
    inference_json_name: str


@dataclass
class StorageRecognitionResult:
    """Exact post-inference result shared by offline and live integrations."""

    left_book_end: dict[str, Any] | None
    right_book_end: dict[str, Any] | None
    book_end_selection: dict[str, Any]
    roi: list[int] | None
    roi_error: str | None
    roi_determination: dict[str, Any]
    residual: np.ndarray
    obstacles: list[dict[str, Any]]
    obstacle_order: list[dict[str, Any]]
    spaces: list[dict[str, Any]]
    rejected_spaces: list[dict[str, Any]]
    final_labels: np.ndarray
    obstacle_pairs: list[dict[str, Any]]
    selected_space: dict[str, Any] | None
    final_space_selection: dict[str, Any]
    selected_space_id: int | None
    selected_space_mask: np.ndarray
    held_book_front_mask: np.ndarray
    held_book_left_occlusion_band: np.ndarray
    held_book_occlusion_mask: np.ndarray
    held_pair_space_before_occlusion: np.ndarray
    held_pair_space_after_occlusion: np.ndarray
    held_book_occlusion_metadata: dict[str, Any]


SPINE_MODEL = ModelSpec(
    role="book_spine",
    checkpoint=SPINE_CHECKPOINT,
    expected_sha256=EXPECTED_SPINE_SHA256,
    prompt=SPINE_PROMPT,
    checkpoint_format="sam3_inference",
    npz_name="book_spine_masks.npz",
    overlay_name="book_spine_overlay.png",
    inference_json_name="book_spine_inference.json",
)
BOOK_END_MODEL = ModelSpec(
    role="book_end",
    checkpoint=BOOK_END_CHECKPOINT,
    expected_sha256=EXPECTED_BOOK_END_SHA256,
    prompt=BOOK_END_PROMPT,
    checkpoint_format="model_only_epoch_wrapper",
    npz_name="book_end_masks.npz",
    overlay_name="book_end_overlay.png",
    inference_json_name="book_end_inference.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline 2D storage-space candidate extraction for captures/100test."
    )
    parser.add_argument(
        "--case",
        type=int,
        action="append",
        help="Process only this case; repeat for multiple cases. Default: cases 1..100.",
    )
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Default auto uses CUDA when available.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def make_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_root / stamp
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{stamp}_{suffix:03d}"
        suffix += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate.resolve()


def setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("offline-storage-space")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def discover_cases(input_root: Path, selected: list[int] | None) -> list[tuple[int, Path]]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root not found: {input_root}")
    wanted = sorted(set(selected if selected else range(1, 101)))
    if any(case_id < 1 or case_id > 100 for case_id in wanted):
        raise ValueError("case numbers must be in 1..100")
    cases: list[tuple[int, Path]] = []
    missing: list[str] = []
    for case_id in wanted:
        image_path = input_root / str(case_id) / "after_init_rgb.png"
        if not image_path.is_file():
            missing.append(str(image_path))
        else:
            cases.append((case_id, image_path.resolve()))
    if missing:
        raise FileNotFoundError("required RGB images are missing:\n" + "\n".join(missing))
    return cases


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        return np.asarray(image, dtype=np.uint8)


def verify_checkpoint(spec: ModelSpec) -> dict[str, Any]:
    path = spec.checkpoint.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{spec.role} checkpoint not found: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != spec.expected_sha256:
        raise RuntimeError(
            f"{spec.role} checkpoint SHA-256 mismatch: expected "
            f"{spec.expected_sha256}, got {actual_sha256} ({path})"
        )
    return {
        "role": spec.role,
        "checkpoint": str(path),
        "sha256": actual_sha256,
        "expected_format": spec.checkpoint_format,
        "prompt": spec.prompt,
    }


def load_adapter(
    spec: ModelSpec, device: str, verified: dict[str, Any]
) -> tuple[Sam3Adapter, dict[str, Any]]:
    started = time.perf_counter()
    if spec is SPINE_MODEL:
        adapter = Sam3Adapter(
            checkpoint=spec.checkpoint,
            sam3_root=SAM3_SOURCE,
            confidence_threshold=PROCESSOR_CONFIDENCE_THRESHOLD,
            dtype_mode="bf16",
            device=device,
        )
        if getattr(adapter, "checkpoint_type", None) != "inference":
            raise RuntimeError(
                "book spine checkpoint was not identified as an inference checkpoint: "
                f"{getattr(adapter, 'checkpoint_type', None)!r}"
            )
        load_info = {
            **verified,
            "actual_loaded_format": "sam3_inference",
            "strict_load": True,
            "checkpoint_metadata": getattr(adapter, "checkpoint_metadata", None),
        }
    else:
        # Trusted local checkpoint named explicitly by this offline script.
        payload = torch.load(
            str(spec.checkpoint), map_location="cpu", weights_only=True
        )
        if not isinstance(payload, dict) or set(payload) != {"model", "epoch"}:
            raise RuntimeError(
                "book end checkpoint is not the expected {'model', 'epoch'} "
                f"model-only wrapper: keys={list(payload) if isinstance(payload, dict) else type(payload).__name__}"
            )
        state_dict = payload["model"]
        if not isinstance(state_dict, dict) or not state_dict:
            raise RuntimeError("book end checkpoint has an empty/non-dict model state")
        if not all(torch.is_tensor(value) for value in state_dict.values()):
            raise RuntimeError("book end model state contains non-tensor values")
        model = _fresh_model(device=device, sam301_root=SAM3_SOURCE)
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"strict load of book end model-only checkpoint failed: {exc}"
            ) from exc
        if missing or unexpected:
            raise RuntimeError(
                f"strict load reported missing={missing}, unexpected={unexpected}"
            )
        epoch = payload.get("epoch")
        tensor_count = len(state_dict)
        parameter_count = sum(value.numel() for value in state_dict.values())
        del state_dict, payload
        gc.collect()
        adapter = Sam3Adapter.from_model(
            model,
            sam3_root=SAM3_SOURCE,
            confidence_threshold=PROCESSOR_CONFIDENCE_THRESHOLD,
            dtype_mode="bf16",
            device=device,
            checkpoint_label=str(spec.checkpoint),
        )
        load_info = {
            **verified,
            "actual_loaded_format": "model_only_epoch_wrapper",
            "strict_load": True,
            "epoch": epoch,
            "tensor_count": tensor_count,
            "parameter_count": parameter_count,
            "missing_keys": [],
            "unexpected_keys": [],
        }
    load_info["device"] = device
    load_info["load_seconds"] = time.perf_counter() - started
    return adapter, load_info


def release_adapter(adapter: Sam3Adapter | None) -> None:
    if adapter is not None:
        if getattr(adapter, "model", None) is not None:
            del adapter.model
        if getattr(adapter, "processor", None) is not None:
            del adapter.processor
    Sam3Adapter._cached_model = None
    Sam3Adapter._cached_metadata = None
    Sam3Adapter._cached_key = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def overlay_masks(
    rgb: np.ndarray,
    masks: Iterable[np.ndarray],
    *,
    alpha: float = MASK_ALPHA,
    colors: tuple[tuple[int, int, int], ...] = MASK_COLORS_RGB,
) -> np.ndarray:
    overlay = rgb.copy()
    for index, mask in enumerate(masks):
        mask_bool = np.asarray(mask, dtype=bool)
        color = np.asarray(colors[index % len(colors)], dtype=np.float32)
        overlay[mask_bool] = (
            (1.0 - alpha) * overlay[mask_bool].astype(np.float32) + alpha * color
        ).astype(np.uint8)
    return overlay


def save_png(path: Path, array: np.ndarray) -> None:
    if array.ndim == 2:
        Image.fromarray(array.astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def predict_instances(
    adapter: Sam3Adapter, rgb: np.ndarray, prompt: str
) -> tuple[InstanceSet, dict[str, Any]]:
    if device_of(adapter) == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    raw = adapter.predict(
        Image.fromarray(rgb, mode="RGB"),
        prompt=prompt,
        score_threshold=SAM3_SCORE_THRESHOLD,
        min_area=SAM3_MIN_MASK_AREA_PX,
    )
    nms_started = time.perf_counter()
    nms = apply_mask_nms(
        raw,
        iou_thresh=SAM3_NMS_IOU_THRESHOLD,
        metric="iou",
        mode="suppress",
    )
    if device_of(adapter) == "cuda":
        torch.cuda.synchronize()
    return nms.instances, {
        "raw_mask_count": raw.count,
        "nms_mask_count": nms.instances.count,
        "nms_removed_count": raw.count - nms.instances.count,
        "nms_seconds": time.perf_counter() - nms_started,
        "total_inference_seconds": time.perf_counter() - started,
        "scores": nms.instances.scores.tolist(),
        "thresholds": {
            "processor_confidence": PROCESSOR_CONFIDENCE_THRESHOLD,
            "score": SAM3_SCORE_THRESHOLD,
            "minimum_mask_area_px": SAM3_MIN_MASK_AREA_PX,
            "nms_iou": SAM3_NMS_IOU_THRESHOLD,
        },
    }


def device_of(adapter: Sam3Adapter) -> str:
    return str(getattr(adapter, "device", "cpu"))


def run_inference_phase(
    spec: ModelSpec,
    verified: dict[str, Any],
    cases: list[tuple[int, Path]],
    run_dir: Path,
    device: str,
    logger: logging.Logger,
) -> tuple[dict[str, Any], dict[int, str]]:
    logger.info(
        "loading role=%s checkpoint=%s sha256=%s prompt=%r",
        spec.role,
        spec.checkpoint,
        verified["sha256"],
        spec.prompt,
    )
    adapter: Sam3Adapter | None = None
    errors: dict[int, str] = {}
    try:
        adapter, load_info = load_adapter(spec, device, verified)
        logger.info(
            "loaded role=%s actual_format=%s strict=%s seconds=%.3f",
            spec.role,
            load_info["actual_loaded_format"],
            load_info["strict_load"],
            load_info["load_seconds"],
        )
        for position, (case_id, image_path) in enumerate(cases, start=1):
            case_dir = run_dir / str(case_id)
            case_dir.mkdir(parents=False, exist_ok=True)
            try:
                rgb = read_rgb(image_path)
                input_path = case_dir / "input.png"
                if not input_path.exists():
                    save_png(input_path, rgb)
                instances, inference = predict_instances(adapter, rgb, spec.prompt)
                save_npz(case_dir / spec.npz_name, instances)
                save_png(case_dir / spec.overlay_name, overlay_masks(rgb, instances.masks))
                write_json(
                    case_dir / spec.inference_json_name,
                    {
                        "status": "success",
                        "input_image": str(image_path),
                        "model": load_info,
                        **inference,
                    },
                )
                logger.info(
                    "%s case=%d progress=%d/%d masks=%d seconds=%.3f",
                    spec.role,
                    case_id,
                    position,
                    len(cases),
                    instances.count,
                    inference["total_inference_seconds"],
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                errors[case_id] = error
                logger.error("%s case=%d failed: %s", spec.role, case_id, error)
                try:
                    rgb = read_rgb(image_path)
                    if not (case_dir / "input.png").exists():
                        save_png(case_dir / "input.png", rgb)
                    empty = empty_instances(*rgb.shape[:2])
                    save_npz(case_dir / spec.npz_name, empty)
                    save_png(case_dir / spec.overlay_name, rgb)
                except Exception:
                    pass
                write_json(
                    case_dir / spec.inference_json_name,
                    {
                        "status": "failed",
                        "input_image": str(image_path),
                        "model": load_info,
                        "error": error,
                        "traceback": traceback.format_exc(),
                    },
                )
        return load_info, errors
    finally:
        release_adapter(adapter)
        logger.info("released role=%s model", spec.role)


def bbox_xyxy_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def instance_records(instances: InstanceSet) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, (mask, score) in enumerate(
        zip(instances.masks, instances.scores, strict=True)
    ):
        bbox = bbox_xyxy_from_mask(mask)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        records.append(
            {
                "index": index,
                "score": float(score),
                "bbox_xyxy": bbox,
                "area_px": int(mask.sum()),
                "center_x": 0.5 * (x1 + x2 - 1),
                "center_y": 0.5 * (y1 + y2 - 1),
            }
        )
    return records


def select_book_end_pair(
    spine_masks: np.ndarray, book_end_instances: InstanceSet
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    """Select adjustable left/right book ends around the aggregate spine group.

    Primary pools require the book-end center to sit outside the spine group's
    horizontal bounds.  If either pool is empty, the transparent fallback splits
    candidates at the spine-group center.  The nearest inner edge is selected in
    each pool, with confidence only as the tie-breaker.
    """
    if len(spine_masks) == 0:
        return None, None, {"status": "failed", "reason": "no_book_spine_masks"}
    spine_union = np.any(spine_masks, axis=0)
    spine_bbox = bbox_xyxy_from_mask(spine_union)
    if spine_bbox is None:
        return None, None, {"status": "failed", "reason": "empty_spine_union"}
    candidates = instance_records(book_end_instances)
    if len(candidates) < 2:
        return None, None, {
            "status": "failed",
            "reason": "fewer_than_two_book_end_masks",
            "candidate_count": len(candidates),
            "spine_group_bbox_xyxy": spine_bbox,
        }

    spine_x1, _, spine_x2, _ = spine_bbox
    spine_center_x = 0.5 * (spine_x1 + spine_x2 - 1)
    left_pool = [item for item in candidates if item["center_x"] < spine_x1]
    right_pool = [item for item in candidates if item["center_x"] >= spine_x2]
    strategy = "centers_outside_spine_group"
    if not left_pool or not right_pool:
        left_pool = [item for item in candidates if item["center_x"] < spine_center_x]
        right_pool = [item for item in candidates if item["center_x"] > spine_center_x]
        strategy = "fallback_split_at_spine_center"
    if not left_pool or not right_pool:
        return None, None, {
            "status": "failed",
            "reason": "book_end_candidates_do_not_cover_both_sides",
            "candidate_count": len(candidates),
            "spine_group_bbox_xyxy": spine_bbox,
            "strategy": strategy,
        }

    left = min(
        left_pool,
        key=lambda item: (
            abs(item["bbox_xyxy"][2] - spine_x1),
            -item["score"],
        ),
    )
    right = min(
        right_pool,
        key=lambda item: (
            abs(item["bbox_xyxy"][0] - spine_x2),
            -item["score"],
        ),
    )
    return left, right, {
        "status": "success",
        "strategy": strategy,
        "spine_group_bbox_xyxy": spine_bbox,
        "candidate_count": len(candidates),
        "left_pool_indices": [item["index"] for item in left_pool],
        "right_pool_indices": [item["index"] for item in right_pool],
    }


def select_spines_for_vertical_roi(
    spine_masks: np.ndarray, x1: int, x2: int
) -> tuple[list[int], list[dict[str, Any]]]:
    """Select spines by actual-mask area fraction inside horizontal ROI."""
    selected_indices: list[int] = []
    overlap_records: list[dict[str, Any]] = []
    for index, mask in enumerate(spine_masks):
        total_area = int(mask.sum())
        inside_area = int(mask[:, x1:x2].sum())
        fraction = float(inside_area / total_area) if total_area else 0.0
        selected = bool(
            fraction >= SPINE_MIN_MASK_AREA_FRACTION_INSIDE_HORIZONTAL_ROI
        )
        overlap_records.append(
            {
                "index": index,
                "total_mask_area_px": total_area,
                "mask_area_inside_horizontal_roi_px": inside_area,
                "area_fraction_inside_horizontal_roi": fraction,
                "selected_for_vertical_roi": selected,
            }
        )
        if selected:
            selected_indices.append(index)
    return selected_indices, overlap_records


def tukey_iqr_inliers(values: list[int]) -> tuple[np.ndarray, dict[str, Any]]:
    """Return an inlier mask and auditable Tukey-IQR statistics."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Tukey-IQR requires at least one value")
    q1, q3 = np.percentile(array, [25.0, 75.0])
    iqr = float(q3 - q1)
    lower_fence = float(q1 - VERTICAL_ROI_OUTLIER_IQR_MULTIPLIER * iqr)
    upper_fence = float(q3 + VERTICAL_ROI_OUTLIER_IQR_MULTIPLIER * iqr)
    inliers = (array >= lower_fence) & (array <= upper_fence)
    # Tukey fences always retain the central observations; keep this explicit so
    # a future threshold change cannot silently produce an empty representative.
    if not np.any(inliers):
        inliers = np.ones(array.shape, dtype=bool)
    kept = array[inliers]
    return inliers, {
        "method": "tukey_iqr",
        "iqr_multiplier": VERTICAL_ROI_OUTLIER_IQR_MULTIPLIER,
        "count_before": int(array.size),
        "values_before": array.astype(np.int64).tolist(),
        "mean_before": float(array.mean()),
        "median_before": float(np.median(array)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "count_after": int(kept.size),
        "values_after": kept.astype(np.int64).tolist(),
        "mean_after": float(kept.mean()),
    }


def create_roi(
    spine_masks: np.ndarray,
    left_book_end: dict[str, Any] | None,
    right_book_end: dict[str, Any] | None,
    image_shape: tuple[int, int],
) -> tuple[list[int] | None, str | None, dict[str, Any]]:
    """Build inner-edge x ROI, then robust y ROI from overlapping spines."""
    details: dict[str, Any] = {
        "horizontal_boundary_method": (
            "left book-end right edge to right book-end left edge"
        ),
        "spine_overlap_method": "actual mask area fraction inside horizontal ROI",
        "spine_overlap_threshold": (
            SPINE_MIN_MASK_AREA_FRACTION_INSIDE_HORIZONTAL_ROI
        ),
        "vertical_boundary_method": (
            "independent Tukey-IQR filtering of y_top and y_bottom, then mean"
        ),
        "coordinate_convention": "xyxy_half_open",
    }
    if left_book_end is None or right_book_end is None:
        return None, "left_or_right_book_end_not_selected", details
    if len(spine_masks) == 0:
        return None, "no_book_spine_masks", details
    height, width = image_shape
    left_index = int(left_book_end["index"])
    right_index = int(right_book_end["index"])
    left_bbox = left_book_end["bbox_xyxy"]
    right_bbox = right_book_end["bbox_xyxy"]
    # bbox_xyxy is half-open: bbox x2 is the boundary immediately after the
    # left book-end's maximum occupied x; bbox x1 is the right book-end minimum x.
    left_x = int(left_bbox[2])
    right_x = int(right_bbox[0])
    details["left_book_end_boundary"] = {
        "book_end_index": left_index,
        "method": "selected_book_end_mask_right_edge",
        "bbox_xyxy": left_bbox,
        "selected_x": left_x,
    }
    details["right_book_end_boundary"] = {
        "book_end_index": right_index,
        "method": "selected_book_end_mask_left_edge",
        "bbox_xyxy": right_bbox,
        "selected_x": right_x,
    }
    x1 = int(np.clip(left_x, 0, width))
    x2 = int(np.clip(right_x, 0, width))
    details["horizontal_roi_x"] = [x1, x2]
    details["x_left"] = x1
    details["x_right"] = x2
    details["left_book_end_x_boundary"] = x1
    details["right_book_end_x_boundary"] = x2
    if x2 <= x1:
        return None, f"invalid_horizontal_extent:{x1}>={x2}", details

    selected_indices, overlap_records = select_spines_for_vertical_roi(
        spine_masks, x1, x2
    )
    details["vertical_roi_spine_mask_count"] = len(selected_indices)
    details["horizontal_roi_spine_candidate_count"] = len(selected_indices)
    details["vertical_roi_spine_indices"] = selected_indices
    details["book_spine_horizontal_overlap"] = overlap_records
    if not selected_indices:
        return None, "no_book_spine_masks_sufficiently_overlap_horizontal_roi", details

    spine_bounds: list[dict[str, Any]] = []
    for index in selected_indices:
        bbox = bbox_xyxy_from_mask(spine_masks[index])
        if bbox is None:
            continue
        spine_bounds.append(
            {
                "index": index,
                "y_top": int(bbox[1]),
                "y_bottom": int(bbox[3]),
            }
        )
    if not spine_bounds:
        return None, "selected_book_spine_masks_have_no_pixels", details

    y_tops = [item["y_top"] for item in spine_bounds]
    y_bottoms = [item["y_bottom"] for item in spine_bounds]
    top_inliers, top_stats = tukey_iqr_inliers(y_tops)
    bottom_inliers, bottom_stats = tukey_iqr_inliers(y_bottoms)
    top_excluded = [
        item["index"] for item, keep in zip(spine_bounds, top_inliers, strict=True)
        if not keep
    ]
    bottom_excluded = [
        item["index"]
        for item, keep in zip(spine_bounds, bottom_inliers, strict=True)
        if not keep
    ]
    excluded_union = sorted(set(top_excluded) | set(bottom_excluded))
    for position, item in enumerate(spine_bounds):
        item["y_top_outlier"] = not bool(top_inliers[position])
        item["y_bottom_outlier"] = not bool(bottom_inliers[position])

    y1 = int(np.clip(np.rint(top_stats["mean_after"]), 0, height))
    y2 = int(np.clip(np.rint(bottom_stats["mean_after"]), 0, height))
    details["book_spine_vertical_bounds"] = spine_bounds
    details["excluded_outlier_instances"] = {
        "y_top": top_excluded,
        "y_bottom": bottom_excluded,
        "union": excluded_union,
    }
    details["outlier_filter_method"] = "tukey_iqr_independent_y_top_y_bottom"
    details["outlier_filter_threshold"] = {
        "iqr_multiplier": VERTICAL_ROI_OUTLIER_IQR_MULTIPLIER,
    }
    details["outlier_filter_spine_count_before"] = len(spine_bounds)
    details["outlier_filter_spine_count_after"] = (
        len(spine_bounds) - len(excluded_union)
    )
    details["y_top_inlier_spine_count"] = int(np.count_nonzero(top_inliers))
    details["y_bottom_inlier_spine_count"] = int(
        np.count_nonzero(bottom_inliers)
    )
    details["outlier_filter_spine_count_after_definition"] = (
        "candidate count minus union of y_top/y_bottom outlier instance indices"
    )
    details["y_top_outlier_analysis"] = top_stats
    details["y_bottom_outlier_analysis"] = bottom_stats
    details["y_top_representative_before"] = top_stats["mean_before"]
    details["y_bottom_representative_before"] = bottom_stats["mean_before"]
    details["y_top_representative_after"] = top_stats["mean_after"]
    details["y_bottom_representative_after"] = bottom_stats["mean_after"]
    details["roi_y_top"] = y1
    details["roi_y_bottom"] = y2
    if y2 <= y1:
        return None, f"invalid_vertical_extent:{y1}>={y2}", details
    roi = [x1, y1, x2, y2]
    details["final_roi_xyxy"] = roi
    return roi, None, details


def residual_from_roi(
    image_shape: tuple[int, int], roi: list[int] | None, spine_masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    roi_mask = np.zeros(image_shape, dtype=bool)
    if roi is None:
        return roi_mask, roi_mask.copy()
    x1, y1, x2, y2 = roi
    roi_mask[y1:y2, x1:x2] = True
    spine_union = (
        np.any(spine_masks, axis=0)
        if len(spine_masks)
        else np.zeros(image_shape, dtype=bool)
    )
    return roi_mask, roi_mask & ~spine_union


def build_obstacle_sequence(
    spine_masks: np.ndarray,
    book_end_masks: np.ndarray,
    spine_indices: list[int],
    left_book_end: dict[str, Any] | None,
    right_book_end: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the left-to-right obstacle sequence used for pairwise splitting."""
    obstacles: list[dict[str, Any]] = []

    def append_obstacle(kind: str, instance_index: int, mask: np.ndarray) -> None:
        bbox = bbox_xyxy_from_mask(mask)
        if bbox is None:
            return
        x1, _, x2, _ = bbox
        obstacles.append(
            {
                "type": kind,
                "instance_index": int(instance_index),
                "mask": mask,
                "bbox_xyxy": bbox,
                "x_left": int(x1),
                "x_right": int(x2),
                "center_x": float(0.5 * (x1 + x2 - 1)),
            }
        )

    if left_book_end is not None:
        index = int(left_book_end["index"])
        if 0 <= index < len(book_end_masks):
            append_obstacle("left_book_end", index, book_end_masks[index])
    for index in spine_indices:
        if 0 <= index < len(spine_masks):
            append_obstacle("book_spine", index, spine_masks[index])
    if right_book_end is not None:
        index = int(right_book_end["index"])
        if 0 <= index < len(book_end_masks):
            append_obstacle("right_book_end", index, book_end_masks[index])

    obstacles.sort(
        key=lambda item: (
            item["center_x"],
            item["x_left"],
            item["instance_index"],
        )
    )
    serializable: list[dict[str, Any]] = []
    for order, obstacle in enumerate(obstacles, start=1):
        obstacle["obstacle_id"] = order
        serializable.append(
            {key: value for key, value in obstacle.items() if key != "mask"}
        )
    return obstacles, serializable


def build_held_book_occlusion_masks(
    image_shape: tuple[int, int],
    spine_masks: np.ndarray,
    held_book_spine_indices: set[int] | None,
    roi: list[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Reserve only the image-left side of each held-book mask row."""
    height, width = image_shape
    front_mask = np.zeros(image_shape, dtype=bool)
    left_band = np.zeros(image_shape, dtype=bool)
    held_indices = sorted(set(held_book_spine_indices or ()))
    instance_records: list[dict[str, Any]] = []

    if roi is None:
        x1_roi, y1_roi, x2_roi, y2_roi = 0, 0, width, height
    else:
        x1_roi, y1_roi, x2_roi, y2_roi = roi

    for index in held_indices:
        if index < 0 or index >= len(spine_masks):
            instance_records.append(
                {
                    "instance_index": int(index),
                    "status": "index_out_of_range",
                    "left_occlusion_band_pixel_count": 0,
                }
            )
            continue
        mask = np.asarray(spine_masks[index], dtype=bool)
        front_mask |= mask
        band_pixel_count_before = int(np.count_nonzero(left_band))
        rows_with_mask = 0
        rows_with_band = 0
        boundary_points: list[list[int]] = []
        for y in range(max(0, y1_roi), min(height, y2_roi)):
            xs = np.flatnonzero(mask[y])
            if not len(xs):
                continue
            rows_with_mask += 1
            held_left_x = int(xs.min())
            boundary_points.append([held_left_x, int(y)])
            band_x1 = max(
                0,
                int(x1_roi),
                held_left_x - HELD_BOOK_LEFT_OCCLUSION_WIDTH_PX,
            )
            band_x2 = min(width, int(x2_roi), held_left_x)
            if band_x2 <= band_x1:
                continue
            left_band[y, band_x1:band_x2] = True
            rows_with_band += 1
        instance_records.append(
            {
                "instance_index": int(index),
                "status": "processed",
                "image_side": "left",
                "configured_width_px": HELD_BOOK_LEFT_OCCLUSION_WIDTH_PX,
                "rows_with_held_mask_in_roi": rows_with_mask,
                "rows_with_left_occlusion_band": rows_with_band,
                "left_boundary_points_xy": boundary_points,
                "left_occlusion_band_pixel_count": int(
                    np.count_nonzero(left_band) - band_pixel_count_before
                ),
            }
        )

    occlusion_mask = front_mask | left_band
    metadata = {
        "enabled": bool(held_indices),
        "held_book_spine_indices": held_indices,
        "side_mapping": (
            "real held-book right side equals image-left side of held mask"
        ),
        "left_occlusion_width_px": HELD_BOOK_LEFT_OCCLUSION_WIDTH_PX,
        "left_occlusion_method": (
            "per-row held-mask minimum x minus fixed width; image-left only"
        ),
        "roi_clip_xyxy": roi,
        "held_book_front_pixel_count": int(np.count_nonzero(front_mask)),
        "held_book_left_occlusion_band_pixel_count": int(
            np.count_nonzero(left_band)
        ),
        "held_book_occlusion_pixel_count": int(
            np.count_nonzero(occlusion_mask)
        ),
        "instances": instance_records,
    }
    return front_mask, left_band, occlusion_mask, metadata


def maximum_contiguous_width(row: np.ndarray) -> int:
    """Return the widest contiguous True run in one mask row."""
    indices = np.flatnonzero(row)
    if not len(indices):
        return 0
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(indices) - 1]
    return int(np.max(indices[ends] - indices[starts] + 1))


def candidate_shape_width_features(
    mask: np.ndarray,
    roi: list[int],
) -> dict[str, Any]:
    """Measure continuous image-top/bottom openings of one candidate mask."""
    _, y1_roi, _, y2_roi = roi
    roi_height = y2_roi - y1_roi
    occupied_rows = np.flatnonzero(mask.any(axis=1))
    if len(occupied_rows):
        candidate_y_min = int(occupied_rows.min())
        candidate_y_max = int(occupied_rows.max())
        height_px = candidate_y_max - candidate_y_min + 1
        image_band_height_px = max(
            1,
            int(np.ceil(height_px * CANDIDATE_SHAPE_BAND_RATIO)),
        )
        image_top_band_y1 = candidate_y_min
        image_top_band_y2 = min(
            candidate_y_max + 1,
            candidate_y_min + image_band_height_px,
        )
        image_bottom_band_y1 = max(
            candidate_y_min,
            candidate_y_max - image_band_height_px + 1,
        )
        image_bottom_band_y2 = candidate_y_max + 1
        image_top_row_widths = [
            maximum_contiguous_width(mask[y])
            for y in range(image_top_band_y1, image_top_band_y2)
        ]
        image_bottom_row_widths = [
            maximum_contiguous_width(mask[y])
            for y in range(image_bottom_band_y1, image_bottom_band_y2)
        ]
        image_top_width_px = float(np.median(image_top_row_widths))
        image_bottom_width_px = float(np.median(image_bottom_row_widths))
        image_bottom_gap_px = int((y2_roi - 1) - candidate_y_max)
    else:
        candidate_y_min = None
        candidate_y_max = None
        height_px = 0
        image_band_height_px = 0
        image_top_band_y1 = None
        image_top_band_y2 = None
        image_bottom_band_y1 = None
        image_bottom_band_y2 = None
        image_top_width_px = 0.0
        image_bottom_width_px = 0.0
        image_bottom_gap_px = roi_height

    book_bottom_side_width_px = image_top_width_px
    book_top_side_width_px = image_bottom_width_px
    return {
        "candidate_y_min": candidate_y_min,
        "candidate_y_max": candidate_y_max,
        "height_px": height_px,
        "candidate_height_px": height_px,
        "candidate_shape_band_ratio": CANDIDATE_SHAPE_BAND_RATIO,
        "image_band_height_px": image_band_height_px,
        "image_top_band_y_range": (
            [image_top_band_y1, image_top_band_y2]
            if image_top_band_y1 is not None
            else None
        ),
        "image_bottom_band_y_range": (
            [image_bottom_band_y1, image_bottom_band_y2]
            if image_bottom_band_y1 is not None
            else None
        ),
        "image_band_width_statistic": (
            "median of per-row maximum contiguous candidate-mask widths"
        ),
        "image_top_width_px": image_top_width_px,
        "image_bottom_width_px": image_bottom_width_px,
        "book_bottom_side_width_px": book_bottom_side_width_px,
        "book_top_side_width_px": book_top_side_width_px,
        "book_bottom_minus_top_width_px": (
            book_bottom_side_width_px - book_top_side_width_px
        ),
        "image_bottom_gap_px": image_bottom_gap_px,
        "image_to_physical_vertical_mapping": {
            "image_top_small_y": "physical_book_bottom_side",
            "image_bottom_large_y": "physical_book_top_side",
        },
    }


def _fit_boundary_tail_robust(
    actual_boundary_x: np.ndarray,
    y1_roi: int,
) -> tuple[tuple[float, float] | None, dict[str, Any]]:
    """Fit x=a*y+b to terminal real boundary rows using MAD inliers."""
    actual_rows_local = np.flatnonzero(np.isfinite(actual_boundary_x))
    selected_rows_local = actual_rows_local[-BOUNDARY_EXTRAPOLATION_FIT_ROWS:]
    selected_y = selected_rows_local.astype(np.float64) + float(y1_roi)
    selected_x = actual_boundary_x[selected_rows_local].astype(np.float64)
    info: dict[str, Any] = {
        "fit_method": "MAD-filtered least-squares x=a*y+b",
        "fit_rows_requested": BOUNDARY_EXTRAPOLATION_FIT_ROWS,
        "fit_rows_available": int(len(selected_rows_local)),
        "fit_rows_used": 0,
        "fit_y_range": (
            [int(selected_y.min()), int(selected_y.max())]
            if len(selected_y)
            else None
        ),
        "fit_line": None,
        "fit_residual_median_px": None,
        "fit_residual_mad_px": None,
        "fit_points_xy": [],
    }
    if len(selected_rows_local) < BOUNDARY_EXTRAPOLATION_MIN_FIT_ROWS:
        info["fit_status"] = "insufficient_real_boundary_rows"
        return None, info

    design = np.column_stack((selected_y, np.ones_like(selected_y)))
    initial_slope, initial_intercept = np.linalg.lstsq(
        design, selected_x, rcond=None
    )[0]
    residuals = selected_x - (initial_slope * selected_y + initial_intercept)
    residual_median = float(np.median(residuals))
    residual_mad = float(np.median(np.abs(residuals - residual_median)))
    if residual_mad > 0.0:
        robust_sigma = 1.4826 * residual_mad
        inliers = (
            np.abs(residuals - residual_median)
            <= BOUNDARY_EXTRAPOLATION_MAD_MULTIPLIER * robust_sigma
        )
    else:
        inliers = np.ones(len(selected_y), dtype=bool)

    if int(np.count_nonzero(inliers)) < BOUNDARY_EXTRAPOLATION_MIN_FIT_ROWS:
        info.update(
            {
                "fit_status": "insufficient_MAD_inliers",
                "fit_residual_median_px": residual_median,
                "fit_residual_mad_px": residual_mad,
            }
        )
        return None, info

    fit_y = selected_y[inliers]
    fit_x = selected_x[inliers]
    fit_design = np.column_stack((fit_y, np.ones_like(fit_y)))
    slope, intercept = np.linalg.lstsq(fit_design, fit_x, rcond=None)[0]
    info.update(
        {
            "fit_status": "fitted",
            "fit_rows_used": int(len(fit_y)),
            "fit_y_range": [int(fit_y.min()), int(fit_y.max())],
            "fit_line": {
                "slope_x_per_y": float(slope),
                "intercept_x": float(intercept),
            },
            "fit_residual_median_px": residual_median,
            "fit_residual_mad_px": residual_mad,
            "fit_points_xy": [
                [int(round(x)), int(round(y))] for x, y in zip(fit_x, fit_y)
            ],
        }
    )
    return (float(slope), float(intercept)), info


def _extrapolate_bottom_boundary_tail(
    actual_boundary_x: np.ndarray,
    other_actual_boundary_x: np.ndarray,
    *,
    side: str,
    y1_roi: int,
    y2_roi: int,
    x1_roi: int,
    x2_roi: int,
    obstacle_union: np.ndarray,
    allow_extrapolation: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extrapolate one boundary only while the opposite real boundary exists."""
    estimated_boundary_x = np.full(actual_boundary_x.shape, np.nan, dtype=float)
    actual_rows_local = np.flatnonzero(np.isfinite(actual_boundary_x))
    roi_height = y2_roi - y1_roi
    max_distance = max(
        1, int(round(roi_height * BOUNDARY_EXTRAPOLATION_MAX_DISTANCE_RATIO))
    )
    actual_y_range = (
        [
            int(y1_roi + actual_rows_local.min()),
            int(y1_roi + actual_rows_local.max()),
        ]
        if len(actual_rows_local)
        else None
    )
    last_actual_y = actual_y_range[1] if actual_y_range is not None else None
    missing_tail_rows = (
        int((y2_roi - 1) - last_actual_y) if last_actual_y is not None else roi_height
    )
    info: dict[str, Any] = {
        "side": side,
        "actual_boundary_y_range": actual_y_range,
        "actual_boundary_row_count": int(len(actual_rows_local)),
        "has_bottom_tail_missing": bool(missing_tail_rows > 0),
        "missing_tail_is_contiguous": bool(missing_tail_rows > 0),
        "bottom_tail_missing_row_count": missing_tail_rows,
        "extrapolation_allowed_for_pair": allow_extrapolation,
        "extrapolation_attempted": False,
        "extrapolated": False,
        "extrapolation_start_y": None,
        "extrapolation_end_y": None,
        "extrapolation_distance_px": 0,
        "estimated_boundary_row_count": 0,
        "estimated_boundary_points_xy": [],
        "collision_stopped": False,
        "stop_reason": None,
        "max_extrapolation_distance_px": max_distance,
        "max_extrapolation_distance_ratio": (
            BOUNDARY_EXTRAPOLATION_MAX_DISTANCE_RATIO
        ),
        "fit_method": "MAD-filtered least-squares x=a*y+b",
        "fit_status": "not_attempted",
        "fit_rows_requested": BOUNDARY_EXTRAPOLATION_FIT_ROWS,
        "fit_rows_available": 0,
        "fit_rows_used": 0,
        "fit_y_range": None,
        "fit_line": None,
        "fit_residual_median_px": None,
        "fit_residual_mad_px": None,
        "fit_points_xy": [],
    }
    if missing_tail_rows <= 0:
        info["stop_reason"] = "no_bottom_tail_missing"
        return estimated_boundary_x, info
    if not allow_extrapolation:
        info["stop_reason"] = "not_exactly_one_bottom_tail_missing"
        return estimated_boundary_x, info
    if last_actual_y is None:
        info["stop_reason"] = "no_real_boundary_rows"
        return estimated_boundary_x, info
    if missing_tail_rows > max_distance:
        info["stop_reason"] = "missing_tail_exceeds_maximum_distance"
        return estimated_boundary_x, info

    fit, fit_info = _fit_boundary_tail_robust(actual_boundary_x, y1_roi)
    info.update(fit_info)
    if fit is None:
        info["stop_reason"] = fit_info["fit_status"]
        return estimated_boundary_x, info

    slope, intercept = fit
    start_y = last_actual_y + 1
    stop_y = min(y2_roi - 1, last_actual_y + max_distance)
    info["extrapolation_attempted"] = True
    info["extrapolation_start_y"] = start_y
    for y in range(start_y, stop_y + 1):
        local_y = y - y1_roi
        other_boundary = other_actual_boundary_x[local_y]
        if not np.isfinite(other_boundary):
            info["stop_reason"] = "opposite_real_boundary_missing"
            break

        predicted_x = int(round(slope * y + intercept))
        if predicted_x < x1_roi or predicted_x >= x2_roi:
            info["stop_reason"] = "predicted_boundary_outside_roi"
            break

        collision_x1 = max(
            x1_roi,
            predicted_x - BOUNDARY_EXTRAPOLATION_COLLISION_MARGIN_PX,
        )
        collision_x2 = min(
            x2_roi,
            predicted_x + BOUNDARY_EXTRAPOLATION_COLLISION_MARGIN_PX + 1,
        )
        if obstacle_union[y, collision_x1:collision_x2].any():
            info["collision_stopped"] = True
            info["stop_reason"] = "predicted_boundary_collides_with_obstacle"
            break

        if side == "left":
            gap_x1 = predicted_x + 1
            gap_x2 = int(round(other_boundary))
        else:
            gap_x1 = int(round(other_boundary)) + 1
            gap_x2 = predicted_x
        if gap_x2 <= gap_x1:
            info["collision_stopped"] = True
            info["stop_reason"] = "extrapolated_boundaries_cross_or_touch"
            break
        if obstacle_union[y, gap_x1:gap_x2].any():
            info["collision_stopped"] = True
            info["stop_reason"] = "extrapolated_gap_crosses_other_obstacle"
            break

        estimated_boundary_x[local_y] = predicted_x
        info["estimated_boundary_points_xy"].append([predicted_x, y])

    estimated_rows = int(np.count_nonzero(np.isfinite(estimated_boundary_x)))
    info["estimated_boundary_row_count"] = estimated_rows
    info["extrapolated"] = bool(estimated_rows)
    info["extrapolation_distance_px"] = estimated_rows
    if estimated_rows:
        estimated_global_rows = y1_roi + np.flatnonzero(
            np.isfinite(estimated_boundary_x)
        )
        info["extrapolation_end_y"] = int(estimated_global_rows.max())
        if info["stop_reason"] is None:
            info["stop_reason"] = "reached_requested_tail_end"
    return estimated_boundary_x, info


def extract_space_candidates_by_obstacle_pairs(
    residual: np.ndarray,
    roi: list[int] | None,
    obstacles: list[dict[str, Any]],
    blocked_book_spine_indices: set[int] | None = None,
    held_book_occlusion_mask: np.ndarray | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    np.ndarray,
    list[dict[str, Any]],
    dict[str, np.ndarray],
]:
    """Split residual space using actual per-row boundaries of adjacent masks."""
    final_labels = np.zeros(residual.shape, dtype=np.int32)
    empty_debug = {
        "pair_space_before_occlusion": np.zeros(residual.shape, dtype=bool),
        "pair_space_after_occlusion": np.zeros(residual.shape, dtype=bool),
    }
    if roi is None or len(obstacles) < 2:
        return [], [], final_labels, [], empty_debug

    x1_roi, y1_roi, x2_roi, y2_roi = roi
    roi_height = y2_roi - y1_roi
    bottom_tolerance = max(
        ROI_BOTTOM_TOLERANCE_PX,
        int(round(ROI_BOTTOM_TOLERANCE_RATIO * roi_height)),
    )
    retained_with_masks: list[tuple[dict[str, Any], np.ndarray]] = []
    rejected: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    obstacle_union = np.zeros(residual.shape, dtype=bool)
    blocked_spine_indices = set(blocked_book_spine_indices or ())
    narrow_space_filter_enabled = bool(blocked_spine_indices)
    occlusion_mask = (
        np.asarray(held_book_occlusion_mask, dtype=bool)
        if held_book_occlusion_mask is not None
        else np.zeros(residual.shape, dtype=bool)
    )
    if occlusion_mask.shape != residual.shape:
        raise ValueError(
            "held_book_occlusion_mask shape mismatch: "
            f"mask={occlusion_mask.shape} residual={residual.shape}"
        )
    held_pair_before_union = np.zeros(residual.shape, dtype=bool)
    held_pair_after_union = np.zeros(residual.shape, dtype=bool)
    for obstacle in obstacles:
        obstacle_union |= obstacle["mask"]

    for pair_id, (left, right) in enumerate(
        zip(obstacles, obstacles[1:]), start=1
    ):
        nominal_x1 = int(left["x_right"])
        nominal_x2 = int(right["x_left"])
        rowwise_gap_mask = np.zeros(residual.shape, dtype=bool)
        rowwise_gap_widths: list[int] = []
        rowwise_gap_starts: list[int] = []
        rowwise_gap_ends: list[int] = []
        left_actual_boundary_x = np.full(roi_height, np.nan, dtype=float)
        right_actual_boundary_x = np.full(roi_height, np.nan, dtype=float)
        rows_missing_left_mask = 0
        rows_missing_right_mask = 0
        for y in range(y1_roi, y2_roi):
            left_x = np.flatnonzero(left["mask"][y, x1_roi:x2_roi])
            right_x = np.flatnonzero(right["mask"][y, x1_roi:x2_roi])
            if not len(left_x):
                rows_missing_left_mask += 1
            else:
                left_actual_boundary_x[y - y1_roi] = (
                    x1_roi + int(left_x.max())
                )
            if not len(right_x):
                rows_missing_right_mask += 1
            else:
                right_actual_boundary_x[y - y1_roi] = (
                    x1_roi + int(right_x.min())
                )

        left_has_bottom_tail = bool(
            np.isfinite(left_actual_boundary_x).any()
            and not np.isfinite(left_actual_boundary_x[-1])
        )
        right_has_bottom_tail = bool(
            np.isfinite(right_actual_boundary_x).any()
            and not np.isfinite(right_actual_boundary_x[-1])
        )
        left_estimated_boundary_x, left_extrapolation = (
            _extrapolate_bottom_boundary_tail(
                left_actual_boundary_x,
                right_actual_boundary_x,
                side="left",
                y1_roi=y1_roi,
                y2_roi=y2_roi,
                x1_roi=x1_roi,
                x2_roi=x2_roi,
                obstacle_union=obstacle_union,
                allow_extrapolation=(
                    left_has_bottom_tail and not right_has_bottom_tail
                ),
            )
        )
        right_estimated_boundary_x, right_extrapolation = (
            _extrapolate_bottom_boundary_tail(
                right_actual_boundary_x,
                left_actual_boundary_x,
                side="right",
                y1_roi=y1_roi,
                y2_roi=y2_roi,
                x1_roi=x1_roi,
                x2_roi=x2_roi,
                obstacle_union=obstacle_union,
                allow_extrapolation=(
                    right_has_bottom_tail and not left_has_bottom_tail
                ),
            )
        )
        left_boundary_x = np.where(
            np.isfinite(left_actual_boundary_x),
            left_actual_boundary_x,
            left_estimated_boundary_x,
        )
        right_boundary_x = np.where(
            np.isfinite(right_actual_boundary_x),
            right_actual_boundary_x,
            right_estimated_boundary_x,
        )
        rows_with_both_masks = int(
            np.count_nonzero(
                np.isfinite(left_actual_boundary_x)
                & np.isfinite(right_actual_boundary_x)
            )
        )
        rows_with_estimated_boundary = 0
        rows_without_positive_gap = 0
        for local_y in range(roi_height):
            y = y1_roi + local_y
            left_boundary = left_boundary_x[local_y]
            right_boundary = right_boundary_x[local_y]
            if not np.isfinite(left_boundary) or not np.isfinite(right_boundary):
                continue
            if (
                np.isfinite(left_estimated_boundary_x[local_y])
                or np.isfinite(right_estimated_boundary_x[local_y])
            ):
                rows_with_estimated_boundary += 1
            left_boundary = int(round(left_boundary))
            right_boundary = int(round(right_boundary))
            gap_x1 = left_boundary + 1
            gap_x2 = right_boundary
            if gap_x2 <= gap_x1:
                rows_without_positive_gap += 1
                continue
            rowwise_gap_mask[y, gap_x1:gap_x2] = True
            rowwise_gap_widths.append(gap_x2 - gap_x1)
            rowwise_gap_starts.append(gap_x1)
            rowwise_gap_ends.append(gap_x2)

        blocked_sides = [
            side
            for side, obstacle in (("left", left), ("right", right))
            if obstacle["type"] == "book_spine"
            and int(obstacle["instance_index"]) in blocked_spine_indices
        ]
        pair_space_before_occlusion = rowwise_gap_mask & residual
        pair_space_mask = pair_space_before_occlusion & ~occlusion_mask
        if blocked_sides:
            held_pair_before_union |= pair_space_before_occlusion
            held_pair_after_union |= pair_space_mask
        before_occlusion_area = int(
            np.count_nonzero(pair_space_before_occlusion)
        )
        after_occlusion_area = int(np.count_nonzero(pair_space_mask))
        pair_record: dict[str, Any] = {
            "pair_id": pair_id,
            "left_obstacle_id": int(left["obstacle_id"]),
            "left_obstacle_type": left["type"],
            "left_instance_index": int(left["instance_index"]),
            "right_obstacle_id": int(right["obstacle_id"]),
            "right_obstacle_type": right["type"],
            "right_instance_index": int(right["instance_index"]),
            "nominal_bbox_edge_interval_x": [nominal_x1, nominal_x2],
            "x_interval": None,
            "x_interval_method": "not_used_rowwise_actual_mask_boundaries",
            "rowwise_boundary_method": (
                "left actual mask row maximum x + 1 to right actual mask row "
                "minimum x; one short bottom-tail boundary may use robust "
                "sloped extrapolation"
            ),
            "missing_mask_row_policy": (
                "skip unless exactly one boundary has a short continuous "
                "bottom tail; then extrapolate that side only while the "
                "opposite real boundary exists; never use bbox or center"
            ),
            "rows_with_both_obstacle_masks": rows_with_both_masks,
            "rows_with_usable_pair_boundaries": (
                len(rowwise_gap_widths) + rows_without_positive_gap
            ),
            "rows_with_extrapolated_boundary": rows_with_estimated_boundary,
            "rows_missing_left_obstacle_mask": rows_missing_left_mask,
            "rows_missing_right_obstacle_mask": rows_missing_right_mask,
            "rows_without_positive_gap": rows_without_positive_gap,
            "rows_with_positive_rowwise_gap": len(rowwise_gap_widths),
            "rowwise_gap_x_extent": (
                [min(rowwise_gap_starts), max(rowwise_gap_ends)]
                if rowwise_gap_widths
                else None
            ),
            "rowwise_gap_width_min_px": (
                int(np.min(rowwise_gap_widths)) if rowwise_gap_widths else None
            ),
            "rowwise_gap_width_median_px": (
                float(np.median(rowwise_gap_widths))
                if rowwise_gap_widths
                else None
            ),
            "rowwise_gap_width_max_px": (
                int(np.max(rowwise_gap_widths)) if rowwise_gap_widths else None
            ),
            "boundary_extrapolation": {
                "fit_tail_rows": BOUNDARY_EXTRAPOLATION_FIT_ROWS,
                "minimum_fit_rows": BOUNDARY_EXTRAPOLATION_MIN_FIT_ROWS,
                "MAD_multiplier": BOUNDARY_EXTRAPOLATION_MAD_MULTIPLIER,
                "collision_margin_px": (
                    BOUNDARY_EXTRAPOLATION_COLLISION_MARGIN_PX
                ),
                "extrapolated_side": (
                    "left"
                    if left_extrapolation["extrapolated"]
                    else "right"
                    if right_extrapolation["extrapolated"]
                    else None
                ),
                "collision_stopped": bool(
                    left_extrapolation["collision_stopped"]
                    or right_extrapolation["collision_stopped"]
                ),
                "left": left_extrapolation,
                "right": right_extrapolation,
            },
            "pair_space_before_occlusion_area_px": before_occlusion_area,
            "pair_space_after_occlusion_area_px": after_occlusion_area,
            "held_book_occlusion_removed_area_px": (
                before_occlusion_area - after_occlusion_area
            ),
            "held_book_occlusion_width_px": (
                HELD_BOOK_LEFT_OCCLUSION_WIDTH_PX
            ),
            "held_book_occlusion_side_in_image": "left",
            "narrow_space_filter_enabled": narrow_space_filter_enabled,
            "minimum_storage_space_width_px": MIN_STORAGE_SPACE_WIDTH_PX,
            "minimum_storage_space_width_metric": (
                "book_bottom_side_width_px"
            ),
            "selected_space_id": None,
        }
        if blocked_spine_indices:
            pair_record["blocked_by_held_book_spine"] = bool(blocked_sides)
            pair_record["held_book_blocked_sides"] = blocked_sides
            pair_record["held_book_spine_indices"] = sorted(
                blocked_spine_indices
            )
        if not rowwise_gap_widths:
            pair_record.update(
                {"status": "no_positive_rowwise_gap", "component_count": 0}
            )
            pair_records.append(pair_record)
            continue

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            pair_space_mask.astype(np.uint8),
            connectivity=CONNECTED_COMPONENT_CONNECTIVITY,
        )
        valid_components: list[tuple[dict[str, Any], np.ndarray]] = []
        pair_rejected: list[dict[str, Any]] = []
        for label in range(1, count):
            x, y, width, height, area = [int(value) for value in stats[label]]
            reasons: list[str] = []
            if area < SPACE_MIN_AREA_PX:
                reasons.append("area_below_minimum")
            if width < SPACE_MIN_WIDTH_PX:
                reasons.append("width_below_minimum")
            if height < SPACE_MIN_HEIGHT_PX:
                reasons.append("height_below_minimum")
            component_mask = labels == label
            shape_width_features = candidate_shape_width_features(
                component_mask,
                roi,
            )
            storage_space_width_px = float(
                shape_width_features["book_bottom_side_width_px"]
            )
            narrow_space_rejected = bool(
                narrow_space_filter_enabled
                and storage_space_width_px < MIN_STORAGE_SPACE_WIDTH_PX
            )
            narrow_space_rejection_reason = (
                "book_bottom_side_width_below_minimum_storage_space_width"
                if narrow_space_rejected
                else None
            )
            if narrow_space_rejection_reason is not None:
                reasons.append(narrow_space_rejection_reason)
            component_gap_widths = [
                maximum_contiguous_width(component_mask[y, x1_roi:x2_roi])
                for y in range(y1_roi, y2_roi)
            ]
            component_gap_widths = [
                value for value in component_gap_widths if value > 0
            ]
            record = {
                "obstacle_pair_id": pair_id,
                "left_obstacle_id": int(left["obstacle_id"]),
                "right_obstacle_id": int(right["obstacle_id"]),
                "x_interval": None,
                "rowwise_gap_x_extent": pair_record["rowwise_gap_x_extent"],
                "rowwise_boundary_method": pair_record["rowwise_boundary_method"],
                "missing_mask_row_policy": pair_record["missing_mask_row_policy"],
                "boundary_extrapolated_side": pair_record[
                    "boundary_extrapolation"
                ]["extrapolated_side"],
                "boundary_estimated_row_count": pair_record[
                    "rows_with_extrapolated_boundary"
                ],
                "source_component_label": label,
                "area_px": area,
                "bbox_xyxy": [x, y, x + width, y + height],
                "mask_height_px": height,
                "mask_width_px": width,
                "gap_width_min_px": (
                    int(np.min(component_gap_widths))
                    if component_gap_widths
                    else None
                ),
                "gap_width_median_px": (
                    float(np.median(component_gap_widths))
                    if component_gap_widths
                    else None
                ),
                "gap_width_max_px": (
                    int(np.max(component_gap_widths))
                    if component_gap_widths
                    else None
                ),
                "gap_width_measured_row_count": len(component_gap_widths),
                "gap_width_definition": (
                    "maximum contiguous candidate-mask run per occupied image row"
                ),
                "height_ratio_to_roi": float(height / max(1, roi_height)),
                "reaches_roi_bottom": bool(
                    y + height - 1 >= y2_roi - 1 - bottom_tolerance
                ),
                "roi_bottom_tolerance_px": bottom_tolerance,
                **shape_width_features,
                "storage_space_width_metric": "book_bottom_side_width_px",
                "storage_space_width_px": storage_space_width_px,
                "minimum_storage_space_width_px": MIN_STORAGE_SPACE_WIDTH_PX,
                "narrow_space_filter_enabled": narrow_space_filter_enabled,
                "narrow_space_rejected": narrow_space_rejected,
                "narrow_space_rejection_reason": (
                    narrow_space_rejection_reason
                ),
                "rejection_reasons": reasons,
            }
            if reasons:
                pair_rejected.append(record)
            else:
                valid_components.append((record, component_mask))

        valid_components.sort(
            key=lambda item: (
                -int(item[0]["area_px"]),
                int(item[0]["source_component_label"]),
            )
        )
        if valid_components:
            retained_with_masks.append(valid_components[0])
            for record, _ in valid_components[1:]:
                record["rejection_reasons"] = [
                    "not_largest_valid_component_for_obstacle_pair"
                ]
                pair_rejected.append(record)
            pair_record["status"] = "candidate_retained"
        else:
            pair_record["status"] = "no_valid_component"
        pair_record["component_count"] = int(count - 1)
        pair_record["valid_component_count"] = len(valid_components)
        pair_record["rejected_component_count"] = len(pair_rejected)
        rejected.extend(pair_rejected)
        pair_records.append(pair_record)

    retained: list[dict[str, Any]] = []
    pair_by_id = {item["pair_id"]: item for item in pair_records}
    for space_id, (record, mask) in enumerate(retained_with_masks, start=1):
        final_labels[mask] = space_id
        candidate = {"space_id": space_id, **record}
        retained.append(candidate)
        pair_by_id[record["obstacle_pair_id"]]["selected_space_id"] = space_id
    return retained, rejected, final_labels, pair_records, {
        "pair_space_before_occlusion": held_pair_before_union,
        "pair_space_after_occlusion": held_pair_after_union,
    }


def select_final_space(
    spaces: list[dict[str, Any]],
    labels: np.ndarray,
    roi: list[int] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Rank by physical book-bottom width, which is image-top width."""
    method = (
        "book_bottom_side_width_then_book_bottom_minus_top_then_area_then_space_id"
    )
    coordinate_mapping = {
        "image_top_small_y": "physical_book_bottom_side",
        "image_bottom_large_y": "physical_book_top_side",
    }
    if not spaces:
        return None, {
            "status": "no_candidates",
            "method": method,
            "coordinate_mapping": coordinate_mapping,
        }
    if roi is None:
        return None, {
            "status": "no_eligible_candidates",
            "method": method,
            "reason": "ROI is unavailable",
            "selected_space_id": None,
            "coordinate_mapping": coordinate_mapping,
        }

    _, y1_roi, _, y2_roi = roi
    roi_height = y2_roi - y1_roi
    minimum_candidate_height = max(
        FINAL_SELECTION_MIN_HEIGHT_PX,
        int(round(roi_height * FINAL_SELECTION_MIN_HEIGHT_RATIO)),
    )

    for space in spaces:
        space_id = int(space["space_id"])
        mask = labels == space_id
        occupied_rows = np.flatnonzero(mask.any(axis=1))
        if len(occupied_rows):
            candidate_y_min = int(occupied_rows.min())
            candidate_y_max = int(occupied_rows.max())
            height_px = candidate_y_max - candidate_y_min + 1
            image_band_height_px = max(
                1,
                int(np.ceil(height_px * CANDIDATE_SHAPE_BAND_RATIO)),
            )
            image_top_band_y1 = candidate_y_min
            image_top_band_y2 = min(
                candidate_y_max + 1,
                candidate_y_min + image_band_height_px,
            )
            image_bottom_band_y1 = max(
                candidate_y_min,
                candidate_y_max - image_band_height_px + 1,
            )
            image_bottom_band_y2 = candidate_y_max + 1
            image_top_row_widths = [
                maximum_contiguous_width(mask[y])
                for y in range(image_top_band_y1, image_top_band_y2)
            ]
            image_bottom_row_widths = [
                maximum_contiguous_width(mask[y])
                for y in range(image_bottom_band_y1, image_bottom_band_y2)
            ]
            image_top_width_px = float(np.median(image_top_row_widths))
            image_bottom_width_px = float(
                np.median(image_bottom_row_widths)
            )
            image_bottom_gap_px = int((y2_roi - 1) - candidate_y_max)
        else:
            candidate_y_min = None
            candidate_y_max = None
            height_px = 0
            image_band_height_px = 0
            image_top_band_y1 = None
            image_top_band_y2 = None
            image_bottom_band_y1 = None
            image_bottom_band_y2 = None
            image_top_width_px = 0.0
            image_bottom_width_px = 0.0
            image_bottom_gap_px = roi_height

        book_bottom_side_width_px = image_top_width_px
        book_top_side_width_px = image_bottom_width_px
        book_bottom_minus_top_width_px = (
            book_bottom_side_width_px - book_top_side_width_px
        )

        ineligibility_reasons: list[str] = []
        if height_px < minimum_candidate_height:
            ineligibility_reasons.append("candidate_height_below_minimum")
        if book_bottom_side_width_px <= 0.0:
            ineligibility_reasons.append("book_bottom_side_width_is_zero")
        eligible = not ineligibility_reasons
        score = {
            "book_bottom_side_width_px": book_bottom_side_width_px,
            "book_bottom_minus_top_width_px": (
                book_bottom_minus_top_width_px
            ),
            "area_px": int(space["area_px"]),
            "space_id_tie_break": -space_id,
        }
        space.update(
            {
                "candidate_y_min": candidate_y_min,
                "candidate_y_max": candidate_y_max,
                "height_px": height_px,
                "candidate_height_px": height_px,
                "candidate_shape_band_ratio": CANDIDATE_SHAPE_BAND_RATIO,
                "image_band_height_px": image_band_height_px,
                "image_top_band_y_range": (
                    [image_top_band_y1, image_top_band_y2]
                    if image_top_band_y1 is not None
                    else None
                ),
                "image_bottom_band_y_range": (
                    [image_bottom_band_y1, image_bottom_band_y2]
                    if image_bottom_band_y1 is not None
                    else None
                ),
                "image_band_width_statistic": (
                    "median of per-row maximum contiguous candidate-mask widths"
                ),
                "image_top_width_px": image_top_width_px,
                "image_bottom_width_px": image_bottom_width_px,
                "book_bottom_side_width_px": book_bottom_side_width_px,
                "book_top_side_width_px": book_top_side_width_px,
                "book_bottom_minus_top_width_px": (
                    book_bottom_minus_top_width_px
                ),
                "image_bottom_gap_px": image_bottom_gap_px,
                "image_to_physical_vertical_mapping": coordinate_mapping,
                "selection_minimum_candidate_height_px": (
                    minimum_candidate_height
                ),
                "selection_eligible": eligible,
                "selection_ineligibility_reasons": ineligibility_reasons,
                "selection_score": score,
            }
        )

    ranked = sorted(
        spaces,
        key=lambda item: (
            bool(item["selection_eligible"]),
            float(item["book_bottom_side_width_px"]),
            float(item["book_bottom_minus_top_width_px"]),
            int(item["area_px"]),
            -int(item["space_id"]),
        ),
        reverse=True,
    )
    for rank, item in enumerate(ranked, start=1):
        item["final_selection_rank"] = rank

    eligible = [item for item in ranked if item["selection_eligible"]]
    ranked_summary = [
        {
            "rank": int(item["final_selection_rank"]),
            "space_id": int(item["space_id"]),
            "eligible": bool(item["selection_eligible"]),
            "ineligibility_reasons": item["selection_ineligibility_reasons"],
            "score": item["selection_score"],
        }
        for item in ranked
    ]
    if not eligible:
        return None, {
            "status": "no_eligible_candidates",
            "method": method,
            "reason": (
                "no candidate met the height condition with a nonzero "
                "physical book-bottom-side width"
            ),
            "selected_space_id": None,
            "coordinate_mapping": coordinate_mapping,
            "candidate_shape_band_ratio": CANDIDATE_SHAPE_BAND_RATIO,
            "minimum_candidate_height_px": minimum_candidate_height,
            "ranked_candidates": ranked_summary,
        }
    selected = eligible[0]
    return selected, {
        "status": "selected",
        "method": method,
        "selected_space_id": int(selected["space_id"]),
        "selection_reason": (
            "highest eligible lexicographic score: book_bottom_side_width_px "
            "(image_top_width_px), book_bottom_minus_top_width_px, area_px, "
            "then lowest space_id"
        ),
        "selected_score": selected["selection_score"],
        "eligibility_requirements": [
            "height_px >= minimum_candidate_height_px",
            "book_bottom_side_width_px > 0",
        ],
        "coordinate_mapping": coordinate_mapping,
        "candidate_shape_band_ratio": CANDIDATE_SHAPE_BAND_RATIO,
        "minimum_candidate_height_px": minimum_candidate_height,
        "ranked_candidates": ranked_summary,
    }


def draw_roi(rgb: np.ndarray, roi: list[int] | None) -> np.ndarray:
    if roi is None:
        output = rgb.copy()
        cv2.putText(
            output,
            "ROI unavailable",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 64, 64),
            2,
            cv2.LINE_AA,
        )
        return output
    x1, y1, x2, y2 = roi
    output = (0.25 * rgb).astype(np.uint8)
    output[y1:y2, x1:x2] = rgb[y1:y2, x1:x2]
    cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), ROI_COLOR_RGB, 3)
    cv2.putText(
        output,
        "ROI",
        (x1 + 5, max(22, y1 + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        ROI_COLOR_RGB,
        2,
        cv2.LINE_AA,
    )
    return output


def draw_book_end_overlay(
    rgb: np.ndarray,
    instances: InstanceSet,
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> np.ndarray:
    output = overlay_masks(rgb, instances.masks)
    for label, item, color in (
        ("LEFT", left, (255, 255, 0)),
        ("RIGHT", right, (255, 0, 255)),
    ):
        if item is None:
            continue
        x1, y1, x2, y2 = item["bbox_xyxy"]
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), color, 3)
        cv2.putText(
            output,
            label,
            (x1 + 3, max(20, y1 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def draw_all_space_candidates(
    rgb: np.ndarray,
    spaces: list[dict[str, Any]],
    labels: np.ndarray,
    roi: list[int] | None,
) -> np.ndarray:
    output = rgb.copy()
    for space in spaces:
        space_id = int(space["space_id"])
        mask = labels == space_id
        color = np.asarray(
            SPACE_COLORS_RGB[(space_id - 1) % len(SPACE_COLORS_RGB)],
            dtype=np.float32,
        )
        output[mask] = (
            (1.0 - SPACE_ALPHA) * output[mask].astype(np.float32)
            + SPACE_ALPHA * color
        ).astype(np.uint8)
        x1, y1, _, _ = space["bbox_xyxy"]
        cv2.putText(
            output,
            (
                f"space {space_id} "
                f"bookB={space.get('book_bottom_side_width_px', 0):.1f} "
                f"bookT={space.get('book_top_side_width_px', 0):.1f}"
            ),
            (x1 + 2, max(18, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            tuple(int(value) for value in color),
            1,
            cv2.LINE_AA,
        )
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), ROI_COLOR_RGB, 2)
    return output


def draw_selected_space(
    rgb: np.ndarray,
    selected_space: dict[str, Any] | None,
    labels: np.ndarray,
    roi: list[int] | None,
) -> np.ndarray:
    """Draw only the selected final space in translucent red."""
    output = rgb.copy()
    if selected_space is not None:
        space_id = int(selected_space["space_id"])
        mask = labels == space_id
        color = np.asarray(FINAL_SPACE_COLOR_RGB, dtype=np.float32)
        output[mask] = (
            (1.0 - SPACE_ALPHA) * output[mask].astype(np.float32)
            + SPACE_ALPHA * color
        ).astype(np.uint8)
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), ROI_COLOR_RGB, 2)
    return output


def draw_boundary_extrapolation_debug(
    rgb: np.ndarray,
    residual: np.ndarray,
    roi: list[int] | None,
    obstacles: list[dict[str, Any]],
    pair_records: list[dict[str, Any]],
    labels: np.ndarray,
) -> np.ndarray:
    """Show real/estimated pair boundaries over residual and candidates."""
    output = rgb.copy()
    residual_color = np.asarray((40, 120, 255), dtype=np.float32)
    output[residual] = (
        0.82 * output[residual].astype(np.float32) + 0.18 * residual_color
    ).astype(np.uint8)
    candidate_union = labels > 0
    candidate_color = np.asarray((255, 210, 0), dtype=np.float32)
    output[candidate_union] = (
        0.72 * output[candidate_union].astype(np.float32)
        + 0.28 * candidate_color
    ).astype(np.uint8)
    if roi is None:
        return output

    x1_roi, y1_roi, x2_roi, y2_roi = roi
    obstacle_by_id = {
        int(obstacle["obstacle_id"]): obstacle for obstacle in obstacles
    }
    boundary_colors = {
        "left_actual": (0, 255, 80),
        "right_actual": (0, 220, 255),
        "left_estimated": (255, 40, 40),
        "right_estimated": (255, 40, 255),
    }
    for pair in pair_records:
        extrapolation = pair.get("boundary_extrapolation", {})
        if not (
            extrapolation.get("left", {}).get("extrapolation_attempted")
            or extrapolation.get("right", {}).get("extrapolation_attempted")
        ):
            continue
        left = obstacle_by_id[int(pair["left_obstacle_id"])]
        right = obstacle_by_id[int(pair["right_obstacle_id"])]
        for y in range(y1_roi, y2_roi):
            left_x = np.flatnonzero(left["mask"][y, x1_roi:x2_roi])
            right_x = np.flatnonzero(right["mask"][y, x1_roi:x2_roi])
            if len(left_x):
                cv2.circle(
                    output,
                    (x1_roi + int(left_x.max()), y),
                    1,
                    boundary_colors["left_actual"],
                    -1,
                )
            if len(right_x):
                cv2.circle(
                    output,
                    (x1_roi + int(right_x.min()), y),
                    1,
                    boundary_colors["right_actual"],
                    -1,
                )
        for side in ("left", "right"):
            side_info = extrapolation.get(side, {})
            for x, y in side_info.get("estimated_boundary_points_xy", []):
                cv2.circle(
                    output,
                    (int(x), int(y)),
                    2,
                    boundary_colors[f"{side}_estimated"],
                    -1,
                )
            if side_info.get("extrapolated"):
                start_y = int(side_info["extrapolation_start_y"])
                points = side_info["estimated_boundary_points_xy"]
                start_x = int(points[0][0])
                cv2.putText(
                    output,
                    f"pair {pair['pair_id']} {side} extrap",
                    (start_x + 4, max(18, start_y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    boundary_colors[f"{side}_estimated"],
                    1,
                    cv2.LINE_AA,
                )
    cv2.rectangle(
        output,
        (x1_roi, y1_roi),
        (x2_roi - 1, y2_roi - 1),
        ROI_COLOR_RGB,
        2,
    )
    cv2.putText(
        output,
        "real: green/cyan  extrapolated: red/magenta  candidates: yellow",
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def recognize_storage_space(
    rgb: np.ndarray,
    spine_instances: InstanceSet,
    end_instances: InstanceSet,
    *,
    blocked_book_spine_indices: set[int] | None = None,
) -> StorageRecognitionResult:
    """Run the completed storage recognition flow without changing its policy."""
    left, right, selection = select_book_end_pair(
        spine_instances.masks,
        end_instances,
    )
    roi, roi_error, roi_determination = create_roi(
        spine_instances.masks,
        left,
        right,
        rgb.shape[:2],
    )
    _, residual = residual_from_roi(
        rgb.shape[:2],
        roi,
        spine_instances.masks,
    )
    (
        held_book_front_mask,
        held_book_left_occlusion_band,
        held_book_occlusion_mask,
        held_book_occlusion_metadata,
    ) = build_held_book_occlusion_masks(
        rgb.shape[:2],
        spine_instances.masks,
        blocked_book_spine_indices,
        roi,
    )
    obstacles, obstacle_order = build_obstacle_sequence(
        spine_instances.masks,
        end_instances.masks,
        roi_determination.get("vertical_roi_spine_indices", []),
        left,
        right,
    )
    (
        spaces,
        rejected_spaces,
        final_labels,
        obstacle_pairs,
        held_pair_debug,
    ) = (
        extract_space_candidates_by_obstacle_pairs(
            residual,
            roi,
            obstacles,
            blocked_book_spine_indices=blocked_book_spine_indices,
            held_book_occlusion_mask=held_book_occlusion_mask,
        )
    )
    selected_space, final_space_selection = select_final_space(
        spaces,
        final_labels,
        roi,
    )
    selected_space_id = (
        int(selected_space["space_id"])
        if selected_space is not None
        else None
    )
    selected_space_mask = (
        final_labels == selected_space_id
        if selected_space_id is not None
        else np.zeros(final_labels.shape, dtype=bool)
    )
    return StorageRecognitionResult(
        left_book_end=left,
        right_book_end=right,
        book_end_selection=selection,
        roi=roi,
        roi_error=roi_error,
        roi_determination=roi_determination,
        residual=residual,
        obstacles=obstacles,
        obstacle_order=obstacle_order,
        spaces=spaces,
        rejected_spaces=rejected_spaces,
        final_labels=final_labels,
        obstacle_pairs=obstacle_pairs,
        selected_space=selected_space,
        final_space_selection=final_space_selection,
        selected_space_id=selected_space_id,
        selected_space_mask=selected_space_mask,
        held_book_front_mask=held_book_front_mask,
        held_book_left_occlusion_band=held_book_left_occlusion_band,
        held_book_occlusion_mask=held_book_occlusion_mask,
        held_pair_space_before_occlusion=held_pair_debug[
            "pair_space_before_occlusion"
        ],
        held_pair_space_after_occlusion=held_pair_debug[
            "pair_space_after_occlusion"
        ],
        held_book_occlusion_metadata=held_book_occlusion_metadata,
    )


def save_storage_recognition_debug(
    output_dir: Path,
    rgb: np.ndarray,
    spine_instances: InstanceSet,
    end_instances: InstanceSet,
    result: StorageRecognitionResult,
) -> None:
    """Save the same recognition debug artifacts used by the offline runner."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_png(output_dir / "input.png", rgb)
    save_png(
        output_dir / "book_spine_overlay.png",
        overlay_masks(rgb, spine_instances.masks),
    )
    save_png(
        output_dir / "book_end_overlay.png",
        draw_book_end_overlay(
            rgb,
            end_instances,
            result.left_book_end,
            result.right_book_end,
        ),
    )
    save_png(output_dir / "roi.png", draw_roi(rgb, result.roi))
    residual_u8 = result.residual.astype(np.uint8) * 255
    save_png(output_dir / "residual_mask.png", residual_u8)
    save_png(output_dir / "residual_mask_before_split.png", residual_u8)
    save_png(
        output_dir / "held_book_front_mask.png",
        result.held_book_front_mask.astype(np.uint8) * 255,
    )
    save_png(
        output_dir / "held_book_left_occlusion_band.png",
        result.held_book_left_occlusion_band.astype(np.uint8) * 255,
    )
    save_png(
        output_dir / "held_book_occlusion_mask.png",
        result.held_book_occlusion_mask.astype(np.uint8) * 255,
    )
    save_png(
        output_dir / "pair_space_before_occlusion.png",
        result.held_pair_space_before_occlusion.astype(np.uint8) * 255,
    )
    save_png(
        output_dir / "pair_space_after_occlusion.png",
        result.held_pair_space_after_occlusion.astype(np.uint8) * 255,
    )
    all_candidate_mask = result.final_labels > 0
    candidate_mask_image = all_candidate_mask.astype(np.uint8) * 255
    candidate_overlay = draw_all_space_candidates(
        rgb,
        result.spaces,
        result.final_labels,
        result.roi,
    )
    save_png(output_dir / "space_candidates_mask.png", candidate_mask_image)
    save_png(output_dir / "space_candidates_overlay.png", candidate_overlay)
    save_png(
        output_dir / "candidate_overlay_after_occlusion.png",
        candidate_overlay,
    )
    save_png(
        output_dir / "boundary_extrapolation_debug.png",
        draw_boundary_extrapolation_debug(
            rgb,
            result.residual,
            result.roi,
            result.obstacles,
            result.obstacle_pairs,
            result.final_labels,
        ),
    )
    save_png(output_dir / "all_space_candidates_mask.png", candidate_mask_image)
    save_png(output_dir / "all_space_candidates_overlay.png", candidate_overlay)
    save_png(
        output_dir / "final_space_mask.png",
        result.selected_space_mask.astype(np.uint8) * 255,
    )
    save_png(
        output_dir / "final_space_overlay.png",
        draw_selected_space(
            rgb,
            result.selected_space,
            result.final_labels,
            result.roi,
        ),
    )
    candidate_mask_dir = output_dir / "candidate_masks"
    candidate_mask_dir.mkdir(parents=True, exist_ok=True)
    for space in result.spaces:
        space_id = int(space["space_id"])
        save_png(
            candidate_mask_dir / f"space_{space_id:02d}.png",
            (result.final_labels == space_id).astype(np.uint8) * 255,
        )


def read_inference_status(case_dir: Path, spec: ModelSpec) -> dict[str, Any]:
    path = case_dir / spec.inference_json_name
    if not path.is_file():
        return {"status": "failed", "error": f"missing {path.name}"}
    return json.loads(path.read_text(encoding="utf-8"))


def process_case(
    case_id: int,
    image_path: Path,
    run_dir: Path,
    model_info: dict[str, dict[str, Any]],
    logger: logging.Logger,
) -> dict[str, Any]:
    case_dir = run_dir / str(case_id)
    rgb = read_rgb(image_path)
    spine_status = read_inference_status(case_dir, SPINE_MODEL)
    end_status = read_inference_status(case_dir, BOOK_END_MODEL)
    spine_instances = load_npz(case_dir / SPINE_MODEL.npz_name)
    end_instances = load_npz(case_dir / BOOK_END_MODEL.npz_name)

    recognition = recognize_storage_space(rgb, spine_instances, end_instances)
    left = recognition.left_book_end
    right = recognition.right_book_end
    selection = recognition.book_end_selection
    roi = recognition.roi
    roi_error = recognition.roi_error
    roi_determination = recognition.roi_determination
    residual = recognition.residual
    obstacle_order = recognition.obstacle_order
    spaces = recognition.spaces
    rejected_spaces = recognition.rejected_spaces
    final_labels = recognition.final_labels
    obstacle_pairs = recognition.obstacle_pairs
    selected_space = recognition.selected_space
    final_space_selection = recognition.final_space_selection
    selected_space_id = recognition.selected_space_id
    save_storage_recognition_debug(
        case_dir,
        rgb,
        spine_instances,
        end_instances,
        recognition,
    )

    errors: list[str] = []
    if spine_status.get("status") != "success":
        errors.append(f"book_spine_inference:{spine_status.get('error')}")
    if end_status.get("status") != "success":
        errors.append(f"book_end_inference:{end_status.get('error')}")
    if selection.get("status") != "success":
        errors.append(f"book_end_selection:{selection.get('reason')}")
    if roi_error:
        errors.append(f"roi:{roi_error}")
    status = "success" if not errors else "failed"

    metadata = {
        "case": case_id,
        "status": status,
        "errors": errors,
        "input_image": str(image_path),
        "models": model_info,
        "book_spine_mask_count": spine_instances.count,
        "book_end_mask_count": end_instances.count,
        "selected_left_book_end": left,
        "selected_right_book_end": right,
        "book_end_selection": selection,
        "left_book_end_x_boundary": roi_determination.get(
            "left_book_end_x_boundary"
        ),
        "right_book_end_x_boundary": roi_determination.get(
            "right_book_end_x_boundary"
        ),
        "x_left": roi_determination.get("x_left"),
        "x_right": roi_determination.get("x_right"),
        "horizontal_roi_x": roi_determination.get("horizontal_roi_x"),
        "horizontal_roi_spine_candidate_count": roi_determination.get(
            "horizontal_roi_spine_candidate_count", 0
        ),
        "vertical_roi_spine_mask_count": roi_determination.get(
            "vertical_roi_spine_mask_count", 0
        ),
        "vertical_roi_spine_indices": roi_determination.get(
            "vertical_roi_spine_indices", []
        ),
        "book_spine_vertical_bounds": roi_determination.get(
            "book_spine_vertical_bounds", []
        ),
        "excluded_outlier_instances": roi_determination.get(
            "excluded_outlier_instances", {"y_top": [], "y_bottom": [], "union": []}
        ),
        "y_top_representative_before": roi_determination.get(
            "y_top_representative_before"
        ),
        "y_bottom_representative_before": roi_determination.get(
            "y_bottom_representative_before"
        ),
        "y_top_representative_after": roi_determination.get(
            "y_top_representative_after"
        ),
        "y_bottom_representative_after": roi_determination.get(
            "y_bottom_representative_after"
        ),
        "roi_y_top": roi_determination.get("roi_y_top"),
        "roi_y_bottom": roi_determination.get("roi_y_bottom"),
        "roi_determination": roi_determination,
        "roi_xyxy": roi,
        "roi_error": roi_error,
        "roi_book_spine_indices": roi_determination.get(
            "vertical_roi_spine_indices", []
        ),
        "obstacle_order": obstacle_order,
        "adjacent_obstacle_pairs": obstacle_pairs,
        "space_candidate_count": len(spaces),
        "space_candidate_gap_width_definition": (
            "maximum contiguous candidate-mask run per occupied image row"
        ),
        "spaces": spaces,
        "final_space_selection": final_space_selection,
        "selected_space_id": selected_space_id,
        "selection_reason": final_space_selection.get(
            "selection_reason", final_space_selection.get("reason")
        ),
        "selection_scores": final_space_selection.get("ranked_candidates", []),
        "final_selected_space_id": selected_space_id,
        "final_selected_space_area_px": (
            int(selected_space["area_px"]) if selected_space is not None else None
        ),
        "final_selected_space_bbox_xyxy": (
            selected_space["bbox_xyxy"] if selected_space is not None else None
        ),
        "candidate_pixels_outside_residual": int(
            np.count_nonzero((final_labels > 0) & ~residual)
        ),
        "outlier_filter_spine_count_before": roi_determination.get(
            "outlier_filter_spine_count_before", 0
        ),
        "outlier_filter_spine_count_after": roi_determination.get(
            "outlier_filter_spine_count_after", 0
        ),
        "outlier_filter_y_top_spine_count_after": roi_determination.get(
            "y_top_inlier_spine_count", 0
        ),
        "outlier_filter_y_bottom_spine_count_after": roi_determination.get(
            "y_bottom_inlier_spine_count", 0
        ),
        "outlier_filter_method": roi_determination.get(
            "outlier_filter_method", "tukey_iqr_independent_y_top_y_bottom"
        ),
        "outlier_filter_threshold": roi_determination.get(
            "outlier_filter_threshold",
            {"iqr_multiplier": VERTICAL_ROI_OUTLIER_IQR_MULTIPLIER},
        ),
        "rejected_space_component_count": len(rejected_spaces),
        "rejected_space_components": rejected_spaces,
        "held_book_occlusion": recognition.held_book_occlusion_metadata,
        "held_book_occlusion_pixel_count": int(
            np.count_nonzero(recognition.held_book_occlusion_mask)
        ),
        "held_pair_space_before_occlusion_area_px": int(
            np.count_nonzero(recognition.held_pair_space_before_occlusion)
        ),
        "held_pair_space_after_occlusion_area_px": int(
            np.count_nonzero(recognition.held_pair_space_after_occlusion)
        ),
        "held_pair_space_removed_by_occlusion_area_px": int(
            np.count_nonzero(
                recognition.held_pair_space_before_occlusion
                & ~recognition.held_pair_space_after_occlusion
            )
        ),
        "thresholds": {
            "space_min_area_px": SPACE_MIN_AREA_PX,
            "space_min_width_px": SPACE_MIN_WIDTH_PX,
            "space_min_height_px": SPACE_MIN_HEIGHT_PX,
            "connected_component_connectivity": CONNECTED_COMPONENT_CONNECTIVITY,
            "held_book_left_occlusion_width_px": (
                HELD_BOOK_LEFT_OCCLUSION_WIDTH_PX
            ),
            "minimum_storage_space_width_px": MIN_STORAGE_SPACE_WIDTH_PX,
            "minimum_storage_space_width_metric": (
                "book_bottom_side_width_px"
            ),
            "minimum_storage_space_width_filter_scope": (
                "enabled only when at least one held-book spine is detected"
            ),
            "roi_bottom_tolerance_px_minimum": ROI_BOTTOM_TOLERANCE_PX,
            "roi_bottom_tolerance_ratio": ROI_BOTTOM_TOLERANCE_RATIO,
            "bottom_reach_is_extraction_filter": False,
            "bottom_reach_is_final_selection_requirement": False,
            "image_to_physical_vertical_mapping": {
                "image_top_small_y": "physical_book_bottom_side",
                "image_bottom_large_y": "physical_book_top_side",
            },
            "candidate_shape_band_ratio": CANDIDATE_SHAPE_BAND_RATIO,
            "final_selection_min_height_px": FINAL_SELECTION_MIN_HEIGHT_PX,
            "final_selection_min_height_ratio": (
                FINAL_SELECTION_MIN_HEIGHT_RATIO
            ),
            "image_band_width_statistic": (
                "median of maximum contiguous candidate-mask width per row"
            ),
            "boundary_extrapolation_fit_rows": (
                BOUNDARY_EXTRAPOLATION_FIT_ROWS
            ),
            "boundary_extrapolation_min_fit_rows": (
                BOUNDARY_EXTRAPOLATION_MIN_FIT_ROWS
            ),
            "boundary_extrapolation_max_distance_ratio": (
                BOUNDARY_EXTRAPOLATION_MAX_DISTANCE_RATIO
            ),
            "boundary_extrapolation_MAD_multiplier": (
                BOUNDARY_EXTRAPOLATION_MAD_MULTIPLIER
            ),
            "boundary_extrapolation_collision_margin_px": (
                BOUNDARY_EXTRAPOLATION_COLLISION_MARGIN_PX
            ),
            "spine_min_mask_area_fraction_inside_horizontal_roi": (
                SPINE_MIN_MASK_AREA_FRACTION_INSIDE_HORIZONTAL_ROI
            ),
            "vertical_roi_outlier_iqr_multiplier": (
                VERTICAL_ROI_OUTLIER_IQR_MULTIPLIER
            ),
        },
    }
    write_json(case_dir / "metadata.json", metadata)
    logger.info(
        "processed case=%d status=%s spine_masks=%d end_masks=%d spaces=%d",
        case_id,
        status,
        spine_instances.count,
        end_instances.count,
        len(spaces),
    )
    return metadata


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    cases = discover_cases(input_root, args.case)
    run_dir = make_run_dir(output_root)
    logger = setup_logging(run_dir)
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    started = datetime.now().astimezone()
    logger.info("run_dir=%s", run_dir)
    logger.info("input_root=%s cases=%d device=%s", input_root, len(cases), device)
    spine_verified = verify_checkpoint(SPINE_MODEL)
    end_verified = verify_checkpoint(BOOK_END_MODEL)
    if spine_verified["sha256"] == end_verified["sha256"]:
        raise RuntimeError("book spine and book end checkpoints unexpectedly have the same SHA-256")

    write_json(
        run_dir / "run_config.json",
        {
            "started_at": started.isoformat(),
            "input_root": str(input_root),
            "case_ids": [case_id for case_id, _ in cases],
            "device": device,
            "models_verified_before_load": {
                "book_spine": spine_verified,
                "book_end": end_verified,
            },
            "algorithm": (
                "horizontal ROI=selected left book-end right edge to selected right "
                "book-end left edge; vertical ROI=independent Tukey-IQR filtering "
                "and post-filter mean of y_top/y_bottom for book-spine masks whose "
                "actual-mask in-range area fraction passes the configured threshold; "
                "residual=ROI minus union of all actual book-spine masks; "
                "candidates=actual per-row gaps between adjacent obstacle masks "
                "AND residual, minus held-book front plus its per-row image-left "
                "fixed occlusion band, followed by local 8-connected-component "
                "selection; held-book scenes reject openings whose physical "
                "book-bottom-side continuous width is below the configured minimum; "
                "rows missing either obstacle mask are skipped without bbox/center "
                "fallback; final space=largest stable bottom-band opening, then "
                "bottom positive-row median gap width, then area, rendered alone "
                "in translucent red"
            ),
        },
    )

    spine_load, spine_errors = run_inference_phase(
        SPINE_MODEL, spine_verified, cases, run_dir, device, logger
    )
    end_load, end_errors = run_inference_phase(
        BOOK_END_MODEL, end_verified, cases, run_dir, device, logger
    )
    model_info = {"book_spine": spine_load, "book_end": end_load}

    results: list[dict[str, Any]] = []
    for case_id, image_path in cases:
        try:
            results.append(
                process_case(case_id, image_path, run_dir, model_info, logger)
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("postprocess case=%d failed", case_id)
            case_dir = run_dir / str(case_id)
            try:
                rgb = read_rgb(image_path)
                zero = np.zeros(rgb.shape[:2], dtype=np.uint8)
                for name in (
                    "residual_mask.png",
                    "residual_mask_before_split.png",
                    "space_candidates_mask.png",
                    "all_space_candidates_mask.png",
                    "final_space_mask.png",
                ):
                    save_png(case_dir / name, zero)
                for name in (
                    "roi.png",
                    "space_candidates_overlay.png",
                    "all_space_candidates_overlay.png",
                    "final_space_overlay.png",
                ):
                    save_png(case_dir / name, rgb)
            except Exception:
                pass
            failure = {
                "case": case_id,
                "status": "failed",
                "errors": [f"postprocess:{error}"],
                "input_image": str(image_path),
                "models": model_info,
                "book_spine_mask_count": None,
                "book_end_mask_count": None,
                "selected_left_book_end": None,
                "selected_right_book_end": None,
                "left_book_end_x_boundary": None,
                "right_book_end_x_boundary": None,
                "x_left": None,
                "x_right": None,
                "horizontal_roi_x": None,
                "horizontal_roi_spine_candidate_count": 0,
                "vertical_roi_spine_mask_count": 0,
                "vertical_roi_spine_indices": [],
                "book_spine_vertical_bounds": [],
                "excluded_outlier_instances": {
                    "y_top": [],
                    "y_bottom": [],
                    "union": [],
                },
                "y_top_representative_before": None,
                "y_bottom_representative_before": None,
                "y_top_representative_after": None,
                "y_bottom_representative_after": None,
                "roi_y_top": None,
                "roi_y_bottom": None,
                "roi_determination": {},
                "roi_xyxy": None,
                "roi_book_spine_indices": [],
                "obstacle_order": [],
                "adjacent_obstacle_pairs": [],
                "space_candidate_count": 0,
                "spaces": [],
                "final_space_selection": {
                    "status": "no_candidates",
                    "method": (
                        "book_bottom_side_width_then_book_bottom_minus_top_"
                        "then_area_then_space_id"
                    ),
                },
                "selected_space_id": None,
                "selection_reason": "postprocess_failed",
                "selection_scores": [],
                "final_selected_space_id": None,
                "final_selected_space_area_px": None,
                "final_selected_space_bbox_xyxy": None,
                "outlier_filter_spine_count_before": 0,
                "outlier_filter_spine_count_after": 0,
                "outlier_filter_y_top_spine_count_after": 0,
                "outlier_filter_y_bottom_spine_count_after": 0,
                "outlier_filter_method": (
                    "tukey_iqr_independent_y_top_y_bottom"
                ),
                "outlier_filter_threshold": {
                    "iqr_multiplier": VERTICAL_ROI_OUTLIER_IQR_MULTIPLIER,
                },
                "traceback": traceback.format_exc(),
            }
            write_json(case_dir / "metadata.json", failure)
            results.append(failure)

    successful = [item["case"] for item in results if item["status"] == "success"]
    failed = [item["case"] for item in results if item["status"] != "success"]
    book_end_failures = [
        item["case"]
        for item in results
        if (item.get("book_end_mask_count") or 0) < 2
        or item.get("selected_left_book_end") is None
        or item.get("selected_right_book_end") is None
    ]
    zero_space = [
        item["case"]
        for item in results
        if item["status"] == "success" and item.get("space_candidate_count") == 0
    ]
    completed = datetime.now().astimezone()
    summary = {
        "status": "completed",
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "run_dir": str(run_dir),
        "requested_case_count": len(cases),
        "successful_case_count": len(successful),
        "failed_case_count": len(failed),
        "successful_cases": successful,
        "failed_cases": failed,
        "book_end_recognition_or_selection_failure_cases": book_end_failures,
        "zero_space_candidate_cases_among_successes": zero_space,
        "phase_inference_error_cases": {
            "book_spine": spine_errors,
            "book_end": end_errors,
        },
        "models": model_info,
        "cases": [
            {
                "case": item["case"],
                "status": item["status"],
                "book_spine_mask_count": item.get("book_spine_mask_count"),
                "book_end_mask_count": item.get("book_end_mask_count"),
                "space_candidate_count": item.get("space_candidate_count"),
                "errors": item.get("errors", []),
            }
            for item in results
        ],
    }
    write_json(run_dir / "summary.json", summary)
    logger.info(
        "completed success=%d failed=%d book_end_failures=%d zero_spaces=%d output=%s",
        len(successful),
        len(failed),
        len(book_end_failures),
        len(zero_space),
        run_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
