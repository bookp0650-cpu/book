"""Hardware-free visual preview generation for ROC artificial-mask errors."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detection.pro_handbook.thesis import roc_experiment as roc


PATTERN_ORDER = ("expansion", "erosion", "holes", "local_region_addition")


def _range(config: dict[str, Any], key: str) -> tuple[float, float]:
    raw = config.get(key)
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"{key} must be a two-value JSON array: {raw!r}")
    low, high = float(raw[0]), float(raw[1])
    if not (0.0 <= low <= high):
        raise ValueError(f"invalid {key}: {raw!r}")
    return low, high


def _integer_range(config: dict[str, Any], key: str) -> tuple[int, int]:
    low, high = _range(config, key)
    if not low.is_integer() or not high.is_integer():
        raise ValueError(f"{key} must contain integers: {config[key]!r}")
    return int(low), int(high)


def _random_integer(
    rng: np.random.Generator, bounds: tuple[int, int],
) -> int:
    low, high = bounds
    return int(rng.integers(low, high + 1))


def _load_gt(
    rgb: np.ndarray,
    rgb_path: Path,
    *,
    gt_mask_path: Path | None,
    gt_points_path: Path | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if gt_mask_path is not None:
        mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(gt_mask_path)
        if mask.shape != rgb.shape[:2]:
            raise ValueError(
                f"RGB/GT shape mismatch: {rgb.shape[:2]} / {mask.shape}"
            )
        return mask > 0, {"type": "mask", "path": str(gt_mask_path.resolve())}
    if gt_points_path is not None:
        payload = json.loads(gt_points_path.read_text(encoding="utf-8"))
        points = (
            payload.get("points_xy_source_pixels", payload)
            if isinstance(payload, dict)
            else payload
        )
        points = [[int(x), int(y)] for x, y in points]
        return roc.polygon_mask(rgb.shape[:2], points), {
            "type": "polygon_points", "path": str(gt_points_path.resolve()),
            "points_xy_source_pixels": points,
        }

    stem = rgb_path.stem.removesuffix("_rgb")
    for candidate in (
        rgb_path.with_name(f"{stem}_gt_mask.png"),
        rgb_path.parent / "gt_mask.png",
    ):
        if candidate.is_file():
            return _load_gt(
                rgb, rgb_path, gt_mask_path=candidate, gt_points_path=None,
            )
    raise FileNotFoundError(
        "GT mask was not found beside the RGB; pass --gt-mask or --gt-points"
    )


def _generate_expansion(
    gt: np.ndarray, config: dict[str, Any], rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    amount = _random_integer(rng, _integer_range(config, "morphology_amount_px_range"))
    mask = roc._dilate_mask(gt, amount)
    return mask, {"error_type": "expansion", "morphology_amount_px": amount}


def _generate_erosion(
    gt: np.ndarray, config: dict[str, Any], rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    amount = _random_integer(rng, _integer_range(config, "morphology_amount_px_range"))
    mask = roc._erode_mask(gt, amount)
    return mask, {"error_type": "shrink", "morphology_amount_px": amount}


def _generate_holes(
    gt: np.ndarray, config: dict[str, Any], rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    count = _random_integer(rng, _integer_range(config, "hole_count_range"))
    width = roc.mask_width(gt)
    if "hole_size_ratio_range" in config:
        size_min, size_max = _range(config, "hole_size_ratio_range")
        major_bounds = None
    else:
        major_bounds = _integer_range(config, "hole_major_axis_px_range")
        size_min = size_max = 0.0
    aspect_key = (
        "hole_aspect_ratio_range"
        if "hole_aspect_ratio_range" in config
        else "hole_minor_axis_ratio_range"
    )
    minor_min, minor_max = _range(config, aspect_key)
    minimum_distance = width * float(config.get("min_center_distance_ratio", 0.0))
    ys, xs = np.where(gt)
    if xs.size == 0:
        raise ValueError("GT mask is empty")
    result = gt.copy()
    holes: list[dict[str, Any]] = []
    centers: list[tuple[int, int]] = []
    for _ in range(count):
        center = roc._spaced_gt_center(
            xs, ys, rng, centers, minimum_distance,
        )
        major = (
            _random_integer(rng, major_bounds)
            if major_bounds is not None
            else max(1, int(round(width * rng.uniform(size_min, size_max))))
        )
        minor = max(1, int(round(major * rng.uniform(minor_min, minor_max))))
        angle = float(rng.uniform(0.0, 180.0))
        cut = roc._filled_ellipse_mask(gt.shape, center, (major, minor), angle)
        result &= ~cut
        holes.append({
            "center": list(center), "axes": [major, minor], "angle": angle,
        })
        centers.append(center)
    return result, {
        "error_type": "partial_holes", "hole_count": count, "holes": holes,
    }


def _generate_local_addition(
    gt: np.ndarray, config: dict[str, Any], rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    count = _random_integer(rng, _integer_range(config, "addition_count_range"))
    width = roc.mask_width(gt)
    if "major_axis_scale_range" in config:
        scale_min, scale_max = _range(config, "major_axis_scale_range")
        major_bounds = None
    else:
        major_bounds = _integer_range(config, "major_axis_px_range")
        scale_min = scale_max = 0.0
    minor_min, minor_max = _range(config, "minor_axis_ratio_range")
    offset_min, offset_max = _range(config, "boundary_offset_ratio_range")
    sides = config.get("position_candidates")
    if not isinstance(sides, list) or not sides:
        raise ValueError("position_candidates must be a non-empty JSON array")
    allowed = {"top", "bottom", "left", "right"}
    if not set(sides).issubset(allowed):
        raise ValueError(f"invalid position_candidates: {sides!r}")

    result = gt.copy()
    additions: list[dict[str, Any]] = []
    centers: list[tuple[int, int]] = []
    minimum_distance = width * float(config.get("min_center_distance_ratio", 0.0))
    requested_sides = roc._side_sequence([str(side) for side in sides], count, rng)
    for requested_side in requested_sides:
        scale = (
            float(rng.uniform(scale_min, scale_max))
            if major_bounds is None
            else None
        )
        major = (
            _random_integer(rng, major_bounds)
            if major_bounds is not None
            else max(1, int(round(width * float(scale))))
        )
        minor = max(1, int(round(major * rng.uniform(minor_min, minor_max))))
        angle = float(rng.uniform(0.0, 180.0))
        offset_ratio = float(rng.uniform(offset_min, offset_max))
        best = None
        best_center = None
        best_distance = -1.0
        for _ in range(48):
            placement = roc._sample_contour_point_by_side(
                gt, rng, requested_side=requested_side,
            )
            side, boundary_point, outward = placement
            center = roc._outward_ellipse_center(
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
        addition = roc._filled_ellipse_mask(
            gt.shape, best_center, (major, minor), angle,
        )
        result |= addition
        additions.append({
            "side": side, "boundary_point": list(boundary_point),
            "center": list(best_center), "axes": [major, minor], "angle": angle,
            "scale": scale,
            "offset_ratio": offset_ratio,
            "boundary_offset_ratio": offset_ratio,
        })
        centers.append(best_center)
    parameters: dict[str, Any] = {
        "error_type": "local_region_addition",
        "addition_count": count,
        "additions": additions,
    }
    item = additions[0]
    parameters.update({
        "addition_side": item["side"],
        "addition_center": item["center"],
        "addition_axes": item["axes"],
        "addition_angle": item["angle"],
    })
    return result, parameters


GENERATORS = {
    "expansion": _generate_expansion,
    "erosion": _generate_erosion,
    "holes": _generate_holes,
    "local_region_addition": _generate_local_addition,
}


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to save image: {path}")


def _panel(
    gt_overlay: np.ndarray,
    artificial_overlay: np.ndarray,
    comparison: np.ndarray,
) -> np.ndarray:
    panels = []
    for label, image in (
        ("GT", gt_overlay),
        ("ARTIFICIAL", artificial_overlay),
        ("COMPARE", comparison),
    ):
        part = image.copy()
        cv2.rectangle(part, (0, 0), (230, 42), (0, 0, 0), -1)
        cv2.putText(
            part, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        panels.append(part)
    return cv2.hconcat(panels)


def _contact_sheet(images: list[np.ndarray]) -> np.ndarray:
    width = 420
    thumbs = [
        cv2.resize(image, (width, max(1, round(image.shape[0] * width / image.shape[1]))))
        for image in images
    ]
    row_width = 2
    blank = np.zeros_like(thumbs[0])
    rows = []
    for start in range(0, len(thumbs), row_width):
        row = thumbs[start:start + row_width]
        rows.append(cv2.hconcat(row + [blank] * (row_width - len(row))))
    return cv2.vconcat(rows)


def _sample_configs(
    pattern: str,
    config: dict[str, Any],
    default_count: int,
) -> list[tuple[str | None, dict[str, Any]]]:
    """Expand optional visual-strength profiles while preserving legacy count."""
    profiles = config.get("profiles")
    if profiles is None:
        count = int(config.get("count", default_count))
        if count < 1:
            raise ValueError(f"preview count must be positive for {pattern}: {count}")
        return [(None, config)] * count
    if not isinstance(profiles, list) or not profiles:
        raise ValueError(f"profiles must be a non-empty array for {pattern}")
    base = {key: value for key, value in config.items() if key != "profiles"}
    expanded: list[tuple[str | None, dict[str, Any]]] = []
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError(f"preview profile must be an object: {profile!r}")
        name = str(profile.get("name", "profile"))
        samples = int(profile.get("samples", 1))
        if samples < 1:
            raise ValueError(f"profile samples must be positive: {profile!r}")
        merged = dict(base)
        merged.update({
            key: value for key, value in profile.items()
            if key not in {"name", "samples"}
        })
        expanded.extend((name, merged) for _ in range(samples))
    return expanded


def generate_previews(
    *,
    rgb_path: Path,
    depth_path: Path,
    output_dir: Path,
    config_path: Path,
    gt_mask_path: Path | None = None,
    gt_points_path: Path | None = None,
) -> dict[str, Any]:
    """Generate fixed-count previews without IoU-bin acceptance filtering."""
    rgb_path, depth_path = rgb_path.resolve(), depth_path.resolve()
    config_path, output_dir = config_path.resolve(), output_dir.resolve()
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    depth = np.load(depth_path, mmap_mode="r", allow_pickle=False)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch: {rgb.shape[:2]} / {depth.shape}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gt, gt_source = _load_gt(
        rgb, rgb_path, gt_mask_path=gt_mask_path, gt_points_path=gt_points_path,
    )
    if not np.any(gt):
        raise ValueError("GT mask is empty")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite preview directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    master_seed = int(config.get("random_seed", 20261022))
    default_count = int(config.get("samples_per_pattern", 10))
    master_rng = np.random.default_rng(master_seed)
    summary: dict[str, Any] = {
        "mode": "preview_only_no_iou_acceptance_filter",
        "rgb_path": str(rgb_path), "depth_path": str(depth_path),
        "gt_source": gt_source, "config_path": str(config_path),
        "random_seed": master_seed, "patterns": {},
        "hardware_started": False, "sam3_started": False, "ocr_started": False,
    }
    pattern_configs = config.get("patterns", {})
    for pattern in PATTERN_ORDER:
        pattern_config = pattern_configs.get(pattern, {})
        if not bool(pattern_config.get("enabled", True)):
            continue
        samples = _sample_configs(pattern, pattern_config, default_count)
        count = len(samples)
        pattern_dir = output_dir / pattern
        pattern_dir.mkdir(parents=False, exist_ok=False)
        ious: list[float] = []
        panels: list[np.ndarray] = []
        profile_counts: dict[str, int] = {}
        for index, (profile_name, sample_config) in enumerate(samples, 1):
            sample_seed = int(master_rng.integers(0, 2**32 - 1))
            sample_rng = np.random.default_rng(sample_seed)
            artificial, parameters = GENERATORS[pattern](
                gt, sample_config, sample_rng,
            )
            actual_iou = roc.mask_iou(gt, artificial)
            parameters.update({
                "preview_pattern": pattern, "preview_index": index,
                "sample_seed": sample_seed, "actual_iou": actual_iou,
                "iou_bin_filter_applied": False,
                "preview_profile": profile_name,
            })
            if profile_name is not None:
                profile_counts[profile_name] = profile_counts.get(profile_name, 0) + 1
            sample_dir = pattern_dir / f"{index:03d}"
            sample_dir.mkdir(parents=False, exist_ok=False)
            gt_overlay = roc.overlay(rgb, gt, (0, 255, 0))
            artificial_overlay = roc.overlay(rgb, artificial, (255, 128, 0))
            comparison = roc.comparison_overlay(rgb, gt, artificial)
            panel = _panel(gt_overlay, artificial_overlay, comparison)
            for filename, image in (
                ("input_rgb.png", rgb),
                ("gt_mask.png", gt.astype(np.uint8) * 255),
                ("artificial_mask.png", artificial.astype(np.uint8) * 255),
                ("gt_overlay.png", gt_overlay),
                ("artificial_mask_overlay.png", artificial_overlay),
                ("gt_artificial_comparison_overlay.png", comparison),
                ("preview_panel.png", panel),
            ):
                _write_image(sample_dir / filename, image)
            roc.write_json_atomic(sample_dir / "parameters.json", parameters)
            panels.append(panel)
            ious.append(actual_iou)
        contact_sheet = pattern_dir / "contact_sheet.png"
        _write_image(contact_sheet, _contact_sheet(panels))
        summary["patterns"][pattern] = {
            "count": count,
            "iou_min": float(np.min(ious)),
            "iou_max": float(np.max(ious)),
            "iou_mean": float(np.mean(ious)),
            "contact_sheet": str(contact_sheet),
            "profile_counts": profile_counts,
        }
    roc.write_json_atomic(output_dir / "preview_summary.json", summary)
    return summary
