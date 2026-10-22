#!/usr/bin/env python3
"""Shared, hardware-free utilities for artificial-mask ROC experiments.

This module deliberately contains no RealSense, ROS, xArm, Dynamixel, OCR, or
SAM3 imports.  Hardware entry points import it only when an ROC experiment is
explicitly requested.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROC_ROOT = PROJECT_ROOT / "captures" / "thesis" / "ROC"
ERROR_TYPES = (
    "expansion",
    "shrink",
    "partial_holes",
    "local_region_addition",
)
IOU_BINS = (
    (1, 0.30, 0.45, False),
    (2, 0.45, 0.60, False),
    (3, 0.60, 0.75, False),
    (4, 0.75, 0.90, False),
    (5, 0.90, 1.00, True),
)

# Ratios are relative to the GT mask width.  They are intentionally data, not
# scattered constants, so preliminary runs can replace them via a JSON file.
DEFAULT_SEVERITY_RANGES = {
    "default": {
        "1": [0.28, 0.65],
        "2": [0.18, 0.48],
        "3": [0.10, 0.34],
        "4": [0.035, 0.20],
        "5": [0.002, 0.07],
    },
    "partial_holes": {
        "1": {
            "hole_count_range": [8, 12],
            "hole_size_ratio_range": [0.45, 0.85],
            "hole_aspect_ratio_range": [0.45, 0.90],
            "min_center_distance_ratio": 0.65,
        },
        "2": {
            "hole_count_range": [6, 9],
            "hole_size_ratio_range": [0.35, 0.70],
            "hole_aspect_ratio_range": [0.45, 0.90],
            "min_center_distance_ratio": 0.55,
        },
        "3": {
            "hole_count_range": [4, 7],
            "hole_size_ratio_range": [0.25, 0.55],
            "hole_aspect_ratio_range": [0.45, 0.90],
            "min_center_distance_ratio": 0.45,
        },
        "4": {
            "hole_count_range": [2, 4],
            "hole_size_ratio_range": [0.15, 0.38],
            "hole_aspect_ratio_range": [0.45, 0.90],
            "min_center_distance_ratio": 0.35,
        },
        "5": {
            "hole_count_range": [1, 2],
            "hole_size_ratio_range": [0.06, 0.20],
            "hole_aspect_ratio_range": [0.45, 0.90],
            "min_center_distance_ratio": 0.20,
        },
    },
    "local_region_addition": {
        "1": {
            "addition_count_range": [7, 10],
            "major_axis_scale_range": [0.65, 1.10],
            "minor_axis_ratio_range": [0.60, 0.95],
            "boundary_offset_ratio_range": [0.30, 0.62],
            "position_candidates": ["top", "bottom", "left", "right"],
            "min_center_distance_ratio": 0.55,
        },
        "2": {
            "addition_count_range": [5, 8],
            "major_axis_scale_range": [0.50, 0.95],
            "minor_axis_ratio_range": [0.60, 0.95],
            "boundary_offset_ratio_range": [0.30, 0.62],
            "position_candidates": ["top", "bottom", "left", "right"],
            "min_center_distance_ratio": 0.50,
        },
        "3": {
            "addition_count_range": [3, 6],
            "major_axis_scale_range": [0.35, 0.75],
            "minor_axis_ratio_range": [0.60, 0.95],
            "boundary_offset_ratio_range": [0.30, 0.62],
            "position_candidates": ["top", "bottom", "left", "right"],
            "min_center_distance_ratio": 0.45,
        },
        "4": {
            "addition_count_range": [2, 4],
            "major_axis_scale_range": [0.20, 0.55],
            "minor_axis_ratio_range": [0.60, 0.95],
            "boundary_offset_ratio_range": [0.30, 0.62],
            "position_candidates": ["top", "bottom", "left", "right"],
            "min_center_distance_ratio": 0.35,
        },
        "5": {
            "addition_count_range": [1, 2],
            "major_axis_scale_range": [0.08, 0.25],
            "minor_axis_ratio_range": [0.60, 0.95],
            "boundary_offset_ratio_range": [0.30, 0.62],
            "position_candidates": ["top", "bottom", "left", "right"],
            "min_center_distance_ratio": 0.20,
        },
    },
}
DEFAULT_LOCAL_ADDITION_SHAPE = {
    "major_axis_scale_range": [0.35, 0.72],
    "minor_axis_ratio_range": [0.60, 0.95],
    "boundary_offset_ratio_range": [0.30, 0.62],
}

PLAN_FIELDS = [
    "trial_id", "task", "book_id", "storage_condition",
    "target_iou_bin", "target_iou_min", "target_iou_max",
    "error_type", "random_seed", "completed", "success",
]
TRIAL_FIELDS = [
    "trial_id", "task", "book_id", "storage_condition",
    "target_iou_bin", "target_iou_min", "target_iou_max", "actual_iou",
    "error_type", "error_parameters", "random_seed", "gt_polygon_points",
    "gt_mask_path", "gt_overlay_path", "artificial_mask_path",
    "artificial_overlay_path", "comparison_overlay_path",
    "original_prediction_mask_path", "rgb_path", "depth_path",
    "calculated_grasp_position", "calculated_storage_position", "success",
    "timestamp", "trial_metadata_path", "downstream_result_path",
]


@dataclass(frozen=True)
class ArtificialMaskResult:
    mask: np.ndarray
    iou: float
    parameters: dict[str, Any]
    attempts: int


class CandidateGenerationError(RuntimeError):
    """IoU-bin generation failure with a non-production debug candidate."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: dict[str, Any],
        closest_mask: np.ndarray,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics
        self.closest_mask = np.asarray(closest_mask, bool).copy()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            row = dict(source)
            for key, value in row.items():
                if isinstance(value, (dict, list, tuple)):
                    row[key] = json.dumps(json_safe(value), ensure_ascii=False)
                elif value is None:
                    row[key] = ""
            writer.writerow(row)
    temporary.replace(path)


def ensure_layout() -> None:
    for path in (
        ROC_ROOT / "ret" / "captures",
        ROC_ROOT / "ret" / "scripts",
        ROC_ROOT / "ret" / "results",
        ROC_ROOT / "stor" / "captures" / "grasp",
        ROC_ROOT / "stor" / "captures" / "non-grasp",
        ROC_ROOT / "stor" / "scripts",
        ROC_ROOT / "stor" / "results",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _bin(bin_id: int) -> tuple[int, float, float, bool]:
    try:
        return next(item for item in IOU_BINS if item[0] == int(bin_id))
    except StopIteration as exc:
        raise ValueError(f"unknown IoU bin: {bin_id}") from exc


def _plan_row(
    trial_id: int, task: str, book_id: str, condition: str,
    bin_id: int, error_type: str, seed: int,
) -> dict[str, Any]:
    _, low, high, _ = _bin(bin_id)
    return {
        "trial_id": f"{trial_id:04d}", "task": task, "book_id": book_id,
        "storage_condition": condition, "target_iou_bin": bin_id,
        "target_iou_min": low, "target_iou_max": high,
        "error_type": error_type, "random_seed": seed,
        "completed": "false", "success": "",
    }


def generate_retrieval_plan(seed: int = 20261021) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    per_bin_errors: dict[int, list[str]] = {}
    for bin_id, *_ in IOU_BINS:
        errors = [kind for kind in ERROR_TYPES for _ in range(5)]
        rng.shuffle(errors)
        per_bin_errors[bin_id] = errors
    rows: list[dict[str, Any]] = []
    trial_id = 1
    for book_number in range(1, 21):
        bins = list(range(1, 6))
        rng.shuffle(bins)
        for bin_id in bins:
            error_type = per_bin_errors[bin_id].pop()
            trial_seed = int(rng.integers(0, 2**32 - 1))
            rows.append(_plan_row(
                trial_id, "retrieval", f"{book_number:02d}", "retrieval",
                bin_id, error_type, trial_seed,
            ))
            trial_id += 1
    return rows


def generate_storage_plan(seed: int = 20261021) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    counts = {
        "with_held_book": [13, 13, 12, 12],
        "without_held_book": [12, 12, 13, 13],
    }
    rows: list[dict[str, Any]] = []
    trial_id = 1
    for condition in ("with_held_book", "without_held_book"):
        errors = [
            kind for kind, count in zip(ERROR_TYPES, counts[condition])
            for _ in range(count)
        ]
        rng.shuffle(errors)
        assignments = [(scene, bin_id) for scene in range(1, 11) for bin_id in range(1, 6)]
        rng.shuffle(assignments)
        for scene, bin_id in assignments:
            trial_seed = int(rng.integers(0, 2**32 - 1))
            rows.append(_plan_row(
                trial_id, "storage", f"{scene:02d}", condition,
                bin_id, errors.pop(), trial_seed,
            ))
            trial_id += 1
    return rows


def create_plan(task: str, seed: int, *, force: bool = False) -> Path:
    ensure_layout()
    results = ROC_ROOT / ("ret" if task == "retrieval" else "stor") / "results"
    path = results / "experiment_plan.csv"
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing plan: {path}")
    rows = generate_retrieval_plan(seed) if task == "retrieval" else generate_storage_plan(seed)
    write_csv_atomic(path, rows, PLAN_FIELDS)
    write_json_atomic(results / "experiment_plan_metadata.json", {
        "task": task, "plan_random_seed": seed, "created_at": now_iso(),
        "iou_bins": [
            {"id": i, "minimum": lo, "maximum": hi, "maximum_inclusive": inc}
            for i, lo, hi, inc in IOU_BINS
        ],
        "error_types": list(ERROR_TYPES), "trial_count": len(rows),
    })
    return path


def resolve_trial(task: str, trial_id: int | None, *, resume: bool) -> dict[str, str]:
    plan = ROC_ROOT / ("ret" if task == "retrieval" else "stor") / "results" / "experiment_plan.csv"
    rows = read_csv(plan)
    if not rows:
        raise FileNotFoundError(f"experiment plan is missing or empty: {plan}")
    if resume:
        for row in rows:
            if row.get("completed", "").strip().lower() != "true":
                return row
        raise RuntimeError("all trials are already completed")
    if trial_id is None:
        raise ValueError("--trial-id is required unless --resume is used")
    wanted = f"{int(trial_id):04d}"
    row = next((item for item in rows if item["trial_id"] == wanted), None)
    if row is None:
        raise ValueError(f"trial not found in plan: {wanted}")
    if row.get("completed", "").strip().lower() == "true":
        raise RuntimeError(f"trial {wanted} is already completed; refusing to rerun")
    return row


def trial_dir(row: dict[str, Any]) -> Path:
    if row["task"] == "retrieval":
        return ROC_ROOT / "ret" / "captures" / str(row["trial_id"])
    condition_dir = "grasp" if row["storage_condition"] == "with_held_book" else "non-grasp"
    # Storage IDs are global in the plan but directories are 0001..0050 per condition.
    global_id = int(row["trial_id"])
    local_id = global_id if global_id <= 50 else global_id - 50
    return ROC_ROOT / "stor" / "captures" / condition_dir / f"{local_id:04d}"


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, bool), np.asarray(right, bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shape mismatch: {a.shape} / {b.shape}")
    union = int(np.count_nonzero(a | b))
    return 1.0 if union == 0 else float(np.count_nonzero(a & b) / union)


def mask_width(mask: np.ndarray) -> int:
    _, xs = np.where(np.asarray(mask, bool))
    if xs.size == 0:
        raise ValueError("GT mask is empty")
    return int(xs.max() - xs.min() + 1)


def _odd_kernel(radius: int) -> np.ndarray:
    size = max(3, int(radius) * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    return cv2.dilate(
        np.asarray(mask, bool).astype(np.uint8),
        _odd_kernel(radius),
        iterations=1,
    ) > 0


def _erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    return cv2.erode(
        np.asarray(mask, bool).astype(np.uint8),
        _odd_kernel(radius),
        iterations=1,
    ) > 0


def _filled_ellipse_mask(
    shape: tuple[int, ...],
    center: tuple[int, int],
    axes: tuple[int, int],
    angle: float,
) -> np.ndarray:
    ellipse = np.zeros(shape, np.uint8)
    cv2.ellipse(ellipse, center, axes, angle, 0, 360, 1, -1)
    return ellipse > 0


def _validated_ratio_range(
    config: dict[str, Any], key: str,
) -> tuple[float, float]:
    raw = config[key]
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{key} must contain exactly two numbers: {raw!r}")
    low, high = (float(value) for value in raw)
    if not (0.0 <= low <= high):
        raise ValueError(f"invalid {key}: {raw!r}")
    return low, high


def _validated_integer_range(
    config: dict[str, Any], key: str,
) -> tuple[int, int]:
    low, high = _validated_ratio_range(config, key)
    if not low.is_integer() or not high.is_integer() or low < 1:
        raise ValueError(f"{key} must contain positive integers: {config[key]!r}")
    return int(low), int(high)


def _random_integer(
    rng: np.random.Generator, bounds: tuple[int, int],
) -> int:
    low, high = bounds
    return int(rng.integers(low, high + 1))


def _spaced_gt_center(
    xs: np.ndarray,
    ys: np.ndarray,
    rng: np.random.Generator,
    existing: list[tuple[int, int]],
    minimum_distance: float,
    *,
    attempts: int = 48,
) -> tuple[int, int]:
    """Prefer a GT pixel separated from earlier centers; use the best fallback."""
    best: tuple[int, int] | None = None
    best_distance = -1.0
    for _ in range(attempts):
        pick = int(rng.integers(0, xs.size))
        candidate = (int(xs[pick]), int(ys[pick]))
        if not existing:
            return candidate
        distance = min(math.dist(candidate, center) for center in existing)
        if distance >= minimum_distance:
            return candidate
        if distance > best_distance:
            best, best_distance = candidate, distance
    assert best is not None
    return best


def _side_sequence(
    sides: list[str], count: int, rng: np.random.Generator,
) -> list[str]:
    """Cycle shuffled side sets so additions do not all concentrate on one side."""
    sequence: list[str] = []
    while len(sequence) < count:
        block = list(sides)
        rng.shuffle(block)
        sequence.extend(block)
    return sequence[:count]


def _outward_ellipse_center(
    boundary_point: tuple[int, int],
    outward: tuple[int, int],
    major: int,
    minor: int,
    angle: float,
    offset_ratio: float,
) -> tuple[int, int]:
    angle_rad = math.radians(angle)
    major_direction = np.asarray([math.cos(angle_rad), math.sin(angle_rad)])
    minor_direction = np.asarray([-math.sin(angle_rad), math.cos(angle_rad)])
    normal = np.asarray(outward, dtype=np.float64)
    normal_radius = math.sqrt(
        (major * float(np.dot(normal, major_direction))) ** 2
        + (minor * float(np.dot(normal, minor_direction))) ** 2
    )
    offset_px = max(1, int(round(normal_radius * offset_ratio)))
    return (
        int(boundary_point[0] + outward[0] * offset_px),
        int(boundary_point[1] + outward[1] * offset_px),
    )


def _sample_contour_point_by_side(
    gt: np.ndarray,
    rng: np.random.Generator,
    *,
    requested_side: str | None = None,
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Uniformly choose a side, then a contour point assigned to that side."""
    contours, _ = cv2.findContours(
        gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )
    points = [contour.reshape(-1, 2) for contour in contours if contour.size]
    if not points:
        raise ValueError("GT mask has no external contour")
    contour_points = np.concatenate(points, axis=0)
    xs = contour_points[:, 0].astype(np.float64)
    ys = contour_points[:, 1].astype(np.float64)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    width = max(1.0, x_max - x_min)
    height = max(1.0, y_max - y_min)
    distances = np.column_stack((
        (ys - y_min) / height,
        (y_max - ys) / height,
        (xs - x_min) / width,
        (x_max - xs) / width,
    ))
    side_names = np.asarray(["top", "bottom", "left", "right"])
    outward_normals = ((0, -1), (0, 1), (-1, 0), (1, 0))
    assignments = np.argmin(distances, axis=1)
    available = np.unique(assignments)
    if requested_side is None:
        side_index = int(rng.choice(available))
    else:
        matches = np.flatnonzero(side_names == requested_side)
        if matches.size != 1:
            raise ValueError(f"unknown local-addition side: {requested_side!r}")
        requested_index = int(matches[0])
        side_index = (
            requested_index
            if requested_index in available
            else int(rng.choice(available))
        )
    candidates = np.flatnonzero(assignments == side_index)
    point = contour_points[int(rng.choice(candidates))]
    return (
        str(side_names[side_index]),
        (int(point[0]), int(point[1])),
        outward_normals[side_index],
    )


def generate_candidate(
    gt_mask: np.ndarray, error_type: str, severity: float, rng: np.random.Generator,
    *, local_addition_shape: dict[str, Any] | None = None,
    partial_holes_shape: dict[str, Any] | None = None,
    requested_addition_side: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    gt = np.asarray(gt_mask, bool)
    if error_type not in ERROR_TYPES:
        raise ValueError(f"unsupported error type: {error_type}")
    width = mask_width(gt)
    amount_px = max(1, int(round(width * float(severity))))
    direct_shape_control = (
        error_type == "partial_holes" and partial_holes_shape is not None
    ) or (
        error_type == "local_region_addition"
        and local_addition_shape is not None
        and "addition_count_range" in local_addition_shape
    )
    parameters: dict[str, Any] = {
        "error_type": error_type,
        "width_ratio": None if direct_shape_control else float(severity),
        "mask_width_px": width, "morphology_amount_px": None,
        "hole_count": 0, "holes": [], "addition_side": None,
        "addition_center": None, "addition_axes": None, "addition_angle": None,
    }
    if error_type == "expansion":
        parameters["morphology_amount_px"] = amount_px
        result = _dilate_mask(gt, amount_px)
    elif error_type == "shrink":
        parameters["morphology_amount_px"] = amount_px
        result = _erode_mask(gt, amount_px)
    elif error_type == "partial_holes":
        result = gt.copy()
        ys, xs = np.where(gt)
        if partial_holes_shape is None:
            count_bounds = (1, 5)
            size_bounds = (0.55 * float(severity), 1.25 * float(severity))
            aspect_bounds = (0.35, 1.0)
            minimum_distance = 0.0
        else:
            count_bounds = _validated_integer_range(
                partial_holes_shape, "hole_count_range",
            )
            size_bounds = _validated_ratio_range(
                partial_holes_shape, "hole_size_ratio_range",
            )
            aspect_bounds = _validated_ratio_range(
                partial_holes_shape, "hole_aspect_ratio_range",
            )
            minimum_distance = width * float(
                partial_holes_shape.get("min_center_distance_ratio", 0.0)
            )
        count = _random_integer(rng, count_bounds)
        holes: list[dict[str, Any]] = []
        centers: list[tuple[int, int]] = []
        for _ in range(count):
            center = _spaced_gt_center(
                xs, ys, rng, centers, minimum_distance,
            )
            major = max(1, int(round(width * rng.uniform(*size_bounds))))
            minor = max(1, int(round(major * rng.uniform(*aspect_bounds))))
            axes = (major, minor)
            angle = float(rng.uniform(0.0, 180.0))
            cut = _filled_ellipse_mask(gt.shape, center, axes, angle)
            result &= ~cut
            holes.append({"center": list(center), "axes": list(axes), "angle": angle})
            centers.append(center)
        parameters.update({"hole_count": count, "holes": holes})
    else:
        shape = dict(DEFAULT_LOCAL_ADDITION_SHAPE)
        if local_addition_shape is not None:
            shape.update(local_addition_shape)
        major_min, major_max = _validated_ratio_range(
            shape, "major_axis_scale_range",
        )
        minor_min, minor_max = _validated_ratio_range(
            shape, "minor_axis_ratio_range",
        )
        offset_min, offset_max = _validated_ratio_range(
            shape, "boundary_offset_ratio_range",
        )
        structured = local_addition_shape is not None and (
            "addition_count_range" in local_addition_shape
        )
        count = (
            _random_integer(
                rng,
                _validated_integer_range(shape, "addition_count_range"),
            )
            if structured else 1
        )
        sides = shape.get("position_candidates", ["top", "bottom", "left", "right"])
        if not isinstance(sides, list) or not sides:
            raise ValueError("position_candidates must be a non-empty list")
        allowed_sides = {"top", "bottom", "left", "right"}
        if not set(sides).issubset(allowed_sides):
            raise ValueError(f"invalid position_candidates: {sides!r}")
        requested_sides = (
            [requested_addition_side]
            if requested_addition_side is not None and count == 1
            else _side_sequence([str(side) for side in sides], count, rng)
        )
        minimum_distance = width * float(shape.get("min_center_distance_ratio", 0.0))
        centers: list[tuple[int, int]] = []
        additions: list[dict[str, Any]] = []
        result = gt.copy()
        for requested_side in requested_sides:
            major_scale = float(rng.uniform(major_min, major_max))
            minor_ratio = float(rng.uniform(minor_min, minor_max))
            offset_ratio = float(rng.uniform(offset_min, offset_max))
            major_base = width if structured else amount_px
            major = max(1, int(round(major_base * major_scale)))
            minor = max(1, int(round(major * minor_ratio)))
            angle = float(rng.uniform(0.0, 180.0))
            best: tuple[str, tuple[int, int], tuple[int, int]] | None = None
            best_center: tuple[int, int] | None = None
            best_distance = -1.0
            for _ in range(48):
                placement = _sample_contour_point_by_side(
                    gt, rng, requested_side=requested_side,
                )
                side, boundary_point, outward = placement
                center = _outward_ellipse_center(
                    boundary_point, outward, major, minor, angle, offset_ratio,
                )
                distance = (
                    min(math.dist(center, prior) for prior in centers)
                    if centers else float("inf")
                )
                if distance >= minimum_distance:
                    best, best_center = placement, center
                    break
                if distance > best_distance:
                    best, best_center, best_distance = placement, center, distance
            assert best is not None and best_center is not None
            side, boundary_point, _ = best
            addition = _filled_ellipse_mask(
                gt.shape, best_center, (major, minor), angle,
            )
            result |= addition
            item = {
                "side": side,
                "center": list(best_center),
                "axes": [major, minor],
                "angle": angle,
                "boundary_point": list(boundary_point),
                "scale": major_scale,
                "offset_ratio": offset_ratio,
                "boundary_offset_ratio": offset_ratio,
                "minor_axis_ratio": minor_ratio,
            }
            additions.append(item)
            centers.append(best_center)
        first = additions[0]
        parameters.update({
            "addition_count": count,
            "additions": additions,
            "addition_side": first["side"],
            "addition_center": first["center"],
            "addition_axes": first["axes"],
            "addition_angle": first["angle"],
            "addition_boundary_point": first["boundary_point"],
            "addition_offset_ratio": first["offset_ratio"],
            "addition_major_axis_scale": first["scale"],
            "addition_minor_axis_ratio": first["minor_axis_ratio"],
        })
    return result, parameters


def generate_for_bin(
    gt_mask: np.ndarray, error_type: str, bin_id: int, seed: int,
    *, severity_ranges: dict[str, Any] | None = None,
    max_attempts: int = 5000,
) -> ArtificialMaskResult:
    _, low, high, high_inclusive = _bin(bin_id)
    gt = np.asarray(gt_mask, bool)
    ranges = severity_ranges or DEFAULT_SEVERITY_RANGES
    error_ranges = ranges.get(error_type, {}) if "default" in ranges else {}
    configured_bin = (
        error_ranges.get(str(bin_id))
        if isinstance(error_ranges, dict)
        else None
    )
    candidate_config = (
        dict(configured_bin) if isinstance(configured_bin, dict) else None
    )
    if candidate_config is None:
        selected_ranges = (
            error_ranges if error_ranges else ranges.get("default", ranges)
        )
        severity_min, severity_max = (
            float(v) for v in selected_ranges[str(bin_id)]
        )
        if not (0 <= severity_min <= severity_max):
            raise ValueError(
                f"invalid severity range for {error_type} bin {bin_id}: "
                f"{selected_ranges[str(bin_id)]}"
            )
        severity_metadata: list[float] | None = [severity_min, severity_max]
    else:
        # Structured hole/addition configs directly define width-relative axes.
        # The legacy scalar severity is not involved in these two generators.
        severity_min = severity_max = 1.0
        severity_metadata = None
    rng = np.random.default_rng(int(seed))
    local_addition_shape = dict(DEFAULT_LOCAL_ADDITION_SHAPE)
    partial_holes_shape = None
    if error_type == "partial_holes" and candidate_config is not None:
        partial_holes_shape = candidate_config
    if error_type == "local_region_addition" and candidate_config is not None:
        local_addition_shape.update(candidate_config)
    elif severity_ranges is not None:
        local_addition_shape.update(
            severity_ranges.get("local_region_addition_shape", {})
        )
    requested_addition_side = None
    if error_type == "local_region_addition" and candidate_config is None:
        # With one deliberately small region, lower-IoU candidates on a
        # tall spine are only feasible along a long side.  Higher-IoU bins use
        # all four sides, keeping top/bottom samples without reintroducing the
        # former oversized top/bottom bulges.
        side_choices = (
            np.asarray(["left", "right"])
            if int(bin_id) <= 3
            else np.asarray(["top", "bottom", "left", "right"])
        )
        requested_addition_side = str(rng.choice(side_choices))
    local_iou_values: list[float] = []
    closest_distance = float("inf")
    closest_mask: np.ndarray | None = None
    closest_parameters: dict[str, Any] | None = None
    for attempt in range(1, int(max_attempts) + 1):
        severity = float(rng.uniform(severity_min, severity_max))
        mask, parameters = generate_candidate(
            gt, error_type, severity, rng,
            local_addition_shape=local_addition_shape,
            partial_holes_shape=partial_holes_shape,
            requested_addition_side=requested_addition_side,
        )
        actual = mask_iou(gt, mask)
        if error_type == "local_region_addition":
            local_iou_values.append(actual)
            parameters.update({
                "random_seed": int(seed), "candidate_attempt": attempt,
                "target_iou_bin": int(bin_id),
                "severity_range": severity_metadata,
                "candidate_config": candidate_config,
                "actual_iou": actual,
            })
            distance = (
                low - actual
                if actual < low
                else actual - high
                if actual > high or (actual == high and not high_inclusive)
                else 0.0
            )
            if distance < closest_distance:
                closest_distance = distance
                closest_mask = mask.copy()
                closest_parameters = dict(parameters)
            if not np.any(mask & ~gt):
                continue
        if low <= actual and (actual <= high if high_inclusive else actual < high):
            parameters.update({
                "random_seed": int(seed), "candidate_attempt": attempt,
                "target_iou_bin": int(bin_id),
                "severity_range": severity_metadata,
                "candidate_config": candidate_config,
                "actual_iou": actual,
            })
            return ArtificialMaskResult(mask, actual, parameters, attempt)
    message = (
        f"no {error_type} candidate reached IoU bin {bin_id} "
        f"[{low}, {high}{']' if high_inclusive else ')'} in {max_attempts} attempts; "
        "adjust severity configuration or retry with a different seed"
    )
    if error_type == "local_region_addition":
        if not local_iou_values or closest_mask is None or closest_parameters is None:
            raise RuntimeError(message)
        diagnostics = {
            "error_type": error_type,
            "candidate_count": len(local_iou_values),
            "actual_iou_min": float(np.min(local_iou_values)),
            "actual_iou_max": float(np.max(local_iou_values)),
            "actual_iou_mean": float(np.mean(local_iou_values)),
            "closest_actual_iou": float(closest_parameters["actual_iou"]),
            "closest_candidate_attempt": int(
                closest_parameters["candidate_attempt"]
            ),
            "closest_error_parameters": closest_parameters,
            "target_bin": {
                "id": int(bin_id),
                "minimum": float(low),
                "maximum": float(high),
                "maximum_inclusive": bool(high_inclusive),
            },
            "max_attempts": int(max_attempts),
            "formal_candidate_adopted": False,
        }
        print("\n[ROC LOCAL-ADDITION CANDIDATE FAILURE]")
        print(json.dumps(json_safe(diagnostics), ensure_ascii=False, indent=2))
        raise CandidateGenerationError(
            message,
            diagnostics=diagnostics,
            closest_mask=closest_mask,
        )
    raise RuntimeError(message)


def annotate_polygon_8(rgb_bgr: np.ndarray, *, title: str = "8-point GT polygon") -> list[list[int]]:
    """Interactive matplotlib UI; returned coordinates are in source pixels."""
    import matplotlib.pyplot as plt
    image = cv2.cvtColor(np.asarray(rgb_bgr, np.uint8), cv2.COLOR_BGR2RGB)
    points: list[list[int]] = []
    cancelled = False
    fig, ax = plt.subplots(num=title)
    ax.imshow(image)
    ax.set_title("Left: add 8 points | Right/Backspace: undo | R: reset | Enter: confirm | Esc: cancel")
    artists: list[Any] = []

    def redraw() -> None:
        nonlocal artists
        for artist in artists:
            artist.remove()
        artists = []
        if points:
            data = np.asarray(points + ([points[0]] if len(points) == 8 else []))
            line, = ax.plot(data[:, 0], data[:, 1], "y-o", linewidth=2)
            artists.append(line)
            if len(points) >= 3:
                artists.append(ax.fill(data[:, 0], data[:, 1], color="lime", alpha=0.25)[0])
            for index, (x, y) in enumerate(points, 1):
                artists.append(ax.text(x, y, str(index), color="white", fontsize=10,
                                       bbox={"facecolor": "black", "alpha": 0.7, "pad": 1}))
        fig.canvas.draw_idle()

    def click(event: Any) -> None:
        if event.inaxes is not ax:
            return
        if event.button == 1 and len(points) < 8 and event.xdata is not None:
            x = int(np.clip(round(event.xdata), 0, image.shape[1] - 1))
            y = int(np.clip(round(event.ydata), 0, image.shape[0] - 1))
            points.append([x, y])
        elif event.button == 3 and points:
            points.pop()
        redraw()

    def key(event: Any) -> None:
        nonlocal cancelled
        if event.key in {"backspace", "delete"} and points:
            points.pop(); redraw()
        elif event.key in {"r", "R"}:
            points.clear(); redraw()
        elif event.key == "escape":
            cancelled = True; plt.close(fig)
        elif event.key in {"enter", "return"} and len(points) == 8:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", click)
    fig.canvas.mpl_connect("key_press_event", key)
    plt.show(block=True)
    if cancelled:
        raise KeyboardInterrupt("trial cancelled during GT annotation")
    if len(points) != 8:
        raise RuntimeError(f"GT annotation closed without 8 points: {len(points)}")
    return points


def polygon_mask(shape: tuple[int, int], points: list[list[int]]) -> np.ndarray:
    if len(points) != 8:
        raise ValueError(f"exactly 8 polygon points are required, got {len(points)}")
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, np.int32)], 1)
    if not np.any(mask):
        raise ValueError("polygon produced an empty mask")
    return mask > 0


def overlay(rgb_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    image = np.asarray(rgb_bgr, np.uint8).copy()
    tint = image.copy(); tint[np.asarray(mask, bool)] = color
    return cv2.addWeighted(image, 0.68, tint, 0.32, 0)


def comparison_overlay(rgb_bgr: np.ndarray, gt: np.ndarray, artificial: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb_bgr, np.uint8).copy()
    only_gt = np.asarray(gt, bool) & ~np.asarray(artificial, bool)
    only_art = np.asarray(artificial, bool) & ~np.asarray(gt, bool)
    overlap = np.asarray(gt, bool) & np.asarray(artificial, bool)
    tint = image.copy()
    tint[only_gt] = (0, 0, 255)       # red: missed GT
    tint[only_art] = (255, 0, 0)      # blue: added region
    tint[overlap] = (0, 255, 0)       # green: intersection
    return cv2.addWeighted(image, 0.58, tint, 0.42, 0)


def save_mask_artifacts(
    directory: Path, rgb_bgr: np.ndarray, gt: np.ndarray, artificial: np.ndarray,
    points: list[list[int]], result: ArtificialMaskResult,
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=False)
    paths = {
        "rgb_path": directory / "input_rgb.png",
        "gt_mask_path": directory / "gt_mask.png",
        "gt_overlay_path": directory / "gt_overlay.png",
        "artificial_mask_path": directory / "artificial_mask.png",
        "artificial_overlay_path": directory / "artificial_mask_overlay.png",
        "comparison_overlay_path": directory / "gt_artificial_comparison_overlay.png",
    }
    cv2.imwrite(str(paths["rgb_path"]), rgb_bgr)
    cv2.imwrite(str(paths["gt_mask_path"]), gt.astype(np.uint8) * 255)
    cv2.imwrite(str(paths["gt_overlay_path"]), overlay(rgb_bgr, gt, (0, 255, 0)))
    cv2.imwrite(str(paths["artificial_mask_path"]), artificial.astype(np.uint8) * 255)
    cv2.imwrite(str(paths["artificial_overlay_path"]), overlay(rgb_bgr, artificial, (255, 128, 0)))
    cv2.imwrite(str(paths["comparison_overlay_path"]), comparison_overlay(rgb_bgr, gt, artificial))
    write_json_atomic(directory / "gt_polygon_points.json", {
        "points_xy_source_pixels": points,
        "source_width": int(rgb_bgr.shape[1]), "source_height": int(rgb_bgr.shape[0]),
        "point_count": 8, "coordinate_origin": "top_left", "polygon_order": "click_order",
    })
    write_json_atomic(directory / "artificial_mask_parameters.json", result.parameters)
    return {key: str(path) for key, path in paths.items()}


def update_trial_records(task: str, row: dict[str, Any], metadata: dict[str, Any]) -> None:
    base = ROC_ROOT / ("ret" if task == "retrieval" else "stor") / "results"
    plan_path = base / "experiment_plan.csv"
    plan = read_csv(plan_path)
    for item in plan:
        if item["trial_id"] == str(row["trial_id"]):
            item["completed"] = "true"
            item["success"] = str(metadata.get("success", "")).lower()
            break
    else:
        raise RuntimeError(f"trial disappeared from plan: {row['trial_id']}")
    write_csv_atomic(plan_path, plan, PLAN_FIELDS)
    trials_path = base / "trials.csv"
    trials = read_csv(trials_path)
    if any(item["trial_id"] == str(row["trial_id"]) for item in trials):
        raise FileExistsError(f"trial already exists in aggregate results: {row['trial_id']}")
    trial_row = {key: metadata.get(key, "") for key in TRIAL_FIELDS}
    if isinstance(trial_row.get("success"), bool):
        trial_row["success"] = str(trial_row["success"]).lower()
    trials.append(trial_row)
    trials.sort(key=lambda item: int(item["trial_id"]))
    write_csv_atomic(trials_path, trials, TRIAL_FIELDS)


def _roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[list[dict[str, float]], float]:
    if scores.size == 0 or len(np.unique(labels)) < 2:
        return [], float("nan")
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    positives, negatives = int(labels.sum()), int((~labels).sum())
    points = []
    for threshold in thresholds:
        prediction = scores >= threshold
        tpr = float(np.count_nonzero(prediction & labels) / positives)
        fpr = float(np.count_nonzero(prediction & ~labels) / negatives)
        points.append({"threshold": float(threshold), "fpr": fpr, "tpr": tpr,
                       "youden_j": tpr - fpr})
    ordered = sorted(points, key=lambda item: (item["fpr"], item["tpr"]))
    auc = float(np.trapz([p["tpr"] for p in ordered], [p["fpr"] for p in ordered]))
    return points, auc


def analyze(task: str) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    base = ROC_ROOT / ("ret" if task == "retrieval" else "stor") / "results"
    rows = read_csv(base / "trials.csv")
    valid = [r for r in rows if r.get("actual_iou", "") != "" and r.get("success", "").lower() in {"true", "false", "1", "0"}]
    if not valid:
        raise RuntimeError("trials.csv has no rows with both actual_iou and success")

    def group_result(name: str, group: list[dict[str, str]]) -> dict[str, Any]:
        scores = np.asarray([float(r["actual_iou"]) for r in group], float)
        labels = np.asarray([r["success"].lower() in {"true", "1"} for r in group], bool)
        points, auc = _roc_points(scores, labels)
        best = max(points, key=lambda p: p["youden_j"]) if points else None
        return {"name": name, "trial_count": len(group), "success_count": int(labels.sum()),
                "failure_count": int((~labels).sum()), "auc": auc, "points": points,
                "youden_reference": best,
                "note": "Youden is diagnostic only; no formal threshold rule is selected."}

    groups = {"all": valid}
    for kind in ERROR_TYPES:
        groups[f"error_type:{kind}"] = [r for r in valid if r["error_type"] == kind]
    if task == "storage":
        for condition in ("with_held_book", "without_held_book"):
            groups[f"storage_condition:{condition}"] = [r for r in valid if r["storage_condition"] == condition]
    analyses = {name: group_result(name, group) for name, group in groups.items() if group}
    point_rows = []
    for name, data in analyses.items():
        for point in data["points"]:
            point_rows.append({"group": name, **point})
    write_csv_atomic(base / "roc_points.csv", point_rows, ["group", "threshold", "fpr", "tpr", "youden_j"])
    all_result = analyses["all"]
    plt.figure(figsize=(6, 6))
    if all_result["points"]:
        ordered = sorted(all_result["points"], key=lambda p: (p["fpr"], p["tpr"]))
        plt.plot([p["fpr"] for p in ordered], [p["tpr"] for p in ordered], marker="o",
                 label=f"all (AUC={all_result['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"{task.title()} ROC: IoU predicts manual success")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(base / "roc_curve.png", dpi=180); plt.close()
    summary = {"task": task, "generated_at": now_iso(), "score": "actual_iou",
               "positive_label": "success", "groups": analyses}
    write_json_atomic(base / "summary.json", summary)
    return summary
