#!/usr/bin/env python3
"""Compare deletion-only selected-book-mask post-processing on saved cases.

SAM inference, OCR inference, OCR/SAM assignment, and target selection are not
rerun.  Every method receives the exact selected mask and depth-filter result
saved by one baseline run, so the only experimental variable is mask deletion.
GT is loaded only after post-processing, for metrics and debug visualization.
In particular, remove_islands is disconnected-component cleanup rather than
GT-shape correction; it never edits pixels inside its 8-connected main component.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml
from PIL import Image

from detection.pro_handbook.sam_py_demo import get_book_points as current
from detection.pro_handbook.sam_py_demo.book_mask_postprocess import (
    DEFAULT_CONFIG,
    METHODS,
    load_linked_ocr_polygons,
    normalize_postprocess_config,
    postprocess_selected_mask,
    render_debug_overlay,
)
from detection.pro_handbook.sam_py_demo.modules.book_width import estimate_book_width
from detection.pro_handbook.sam_py_demo.modules.pca_vector import pca_axes_fix_dir
from detection.pro_handbook.thesis.thesis_ret import (
    _load_gt_mask,
    _load_intrinsics,
    _parse_labelmap,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_ROOT = PROJECT_ROOT / "captures" / "thesis" / "ret"
DEFAULT_BASELINE_RUN = (
    EVALUATION_ROOT
    / "retrieval_results"
    / "logs"
    / "run_20261006_023515_527898"
)
DEFAULT_BASELINE_CSV = EVALUATION_ROOT / "retrieval_results" / "results.csv"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("book_mask_postprocess.yaml")
GT_ROOT = EVALUATION_ROOT / "thesis_ret" / "SegmentationClass"
LABELMAP_PATH = EVALUATION_ROOT / "thesis_ret" / "labelmap.txt"

CASE_FIELDS = [
    "case_id",
    "method",
    "params",
    "iou_before",
    "iou_after",
    "delta_iou",
    "width_before_mm",
    "width_after_mm",
    "width_gt_mm",
    "width_error_before_mm",
    "width_error_after_mm",
    "delta_width_error_mm",
    "mask_area_before",
    "mask_area_after",
    "area_ratio",
    "num_linked_ocr",
    "num_valid_ocr",
    "estimated_axis_deg",
    "component_count",
    "removed_component_count",
    "removed_pixel_count",
    "fallback",
    "fallback_reason",
    "processing_time_ms",
]

SUMMARY_FIELDS = [
    "method",
    "params",
    "case_count",
    "mean_iou",
    "median_iou",
    "mean_delta_iou",
    "iou_improved_count",
    "iou_degraded_count",
    "iou_unchanged_count",
    "width_mae",
    "delta_width_mae",
    "width_improved_count",
    "width_degraded_count",
    "island_removed_case_count",
    "fallback_rate",
    "mean_processing_time_ms",
]


def _scalar_or_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else [value]


def _expand_variants(document: dict[str, Any]) -> list[dict[str, Any]]:
    raw = dict(document.get("postprocess") or {})
    enabled = bool(raw.get("enabled", DEFAULT_CONFIG["enabled"]))
    methods_value = raw.get("method", DEFAULT_CONFIG["method"])
    requested = [str(value) for value in _scalar_or_list(methods_value)]
    if "all" in requested:
        requested = list(METHODS)
    if not enabled:
        requested = ["baseline"]
    if "baseline" not in requested:
        requested.insert(0, "baseline")

    variants: list[dict[str, Any]] = []
    seen: set[str] = set()
    relevant = {
        "baseline": [],
        "remove_islands": [],
        "morphology": ["min_area_retention", "min_width_retention"],
        "ocr_axis_ransac": [
            "ocr_aspect_ratio_min",
            "orientation_outlier_deg",
            "boundary_margin_ratio",
            "min_area_retention",
            "min_width_retention",
        ],
        "ocr_axis_smooth": [
            "ocr_aspect_ratio_min",
            "orientation_outlier_deg",
            "boundary_margin_ratio",
            "median_window",
            "min_area_retention",
            "min_width_retention",
        ],
    }
    for method in requested:
        if method not in METHODS:
            raise ValueError(f"unknown postprocess method in YAML: {method!r}")
        keys = relevant[method]
        values = [
            _scalar_or_list(raw.get(key, DEFAULT_CONFIG[key])) for key in keys
        ]
        combinations = itertools.product(*values) if values else [()]
        for combination in combinations:
            cfg = dict(DEFAULT_CONFIG)
            cfg.update(
                {
                    key: value
                    for key, value in raw.items()
                    if not isinstance(value, list) and key != "method"
                }
            )
            cfg.update(dict(zip(keys, combination)))
            cfg["method"] = method
            cfg["enabled"] = method != "baseline"
            cfg = normalize_postprocess_config(cfg)
            key = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                variants.append(cfg)
                seen.add(key)
    return variants


def _parse_indices(value: str) -> list[int]:
    result: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"invalid case range: {token!r}")
            result.update(range(start, end + 1))
        else:
            result.add(int(token))
    if not result or min(result) < 1 or max(result) > 100:
        raise ValueError("indices must be within 1..100")
    return sorted(result)


def _read_baseline_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {str(row["case_id"]).zfill(4): row for row in rows}
    if len(result) < 100:
        raise ValueError(f"baseline CSV has only {len(result)} distinct cases: {path}")
    return result


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return 1.0 if union == 0 else float(intersection / union)


def _selected_index(case_dir: Path) -> int | None:
    path = case_dir / "similarity_scores.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores = payload.get("scores") or []
    if not scores:
        return None
    name = str(scores[0].get("name", ""))
    if not name.startswith("mask_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return None


def _estimate_width_mm(
    mask: np.ndarray,
    baseline_filtered_depth: np.ndarray,
    intr: Any,
    depth_scale: float,
    *,
    random_seed: int,
) -> float:
    """Replay the unchanged point-cloud/RANSAC/PCA/width path.

    The saved baseline depth filter is reused for every method.  This holds its
    median-depth decision constant and isolates the effect of deleting pixels.
    """
    depth = np.asarray(baseline_filtered_depth).copy()
    depth[~np.asarray(mask, dtype=bool)] = 0
    points, _ = current._mask_depth_to_points_uv_for_plane_filter(
        np.asarray(mask, dtype=np.uint8), depth, intr, depth_scale
    )
    if points.shape[0] < 3:
        raise RuntimeError(f"too few depth points after postprocess: {points.shape[0]}")
    try:
        current.o3d.utility.random.seed(int(random_seed))
    except Exception:
        pass
    _, inliers, ransac_info = current._fit_plane_ransac_open3d_for_spine(
        points,
        distance_threshold_m=0.008,
        ransac_n=3,
        num_iterations=1200,
    )
    if not ransac_info.get("used"):
        raise RuntimeError(f"normal RANSAC failed: {ransac_info}")
    points_for_pca = points[inliers]
    if points_for_pca.shape[0] < 10:
        raise RuntimeError(
            f"too few RANSAC inliers for width: {points_for_pca.shape[0]}"
        )
    mean, pc1, pc2 = pca_axes_fix_dir(points_for_pca)
    width_info = estimate_book_width(points_for_pca, mean, pc1, pc2)
    width_m = width_info.get("av_book_width_m")
    if width_m is None or not math.isfinite(float(width_m)):
        raise RuntimeError(f"book width estimation failed: {width_info}")
    return float(width_m) * 1000.0


def _params_json(config: dict[str, Any]) -> str:
    return json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _variant_slug(config: dict[str, Any]) -> str:
    digest = hashlib.sha1(_params_json(config).encode("utf-8")).hexdigest()[:10]
    return f"{config['method']}__{digest}"


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return statistics.fmean(valid) if valid else None


def _summaries(
    rows: list[dict[str, Any]], variants: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    summaries = []
    for config in variants:
        params = _params_json(config)
        group = [
            row
            for row in rows
            if row["method"] == config["method"] and row["params"] == params
        ]
        iou_after = [row["iou_after"] for row in group if row["iou_after"] is not None]
        before_errors = [
            row["width_error_before_mm"]
            for row in group
            if row["width_error_before_mm"] is not None
        ]
        after_errors = [
            row["width_error_after_mm"]
            for row in group
            if row["width_error_after_mm"] is not None
        ]
        summaries.append(
            {
                "method": config["method"],
                "params": params,
                "case_count": len(group),
                "mean_iou": _mean(iou_after),
                "median_iou": statistics.median(iou_after) if iou_after else None,
                "mean_delta_iou": _mean(row["delta_iou"] for row in group),
                "iou_improved_count": sum(row["delta_iou"] > 1e-12 for row in group),
                "iou_degraded_count": sum(row["delta_iou"] < -1e-12 for row in group),
                "iou_unchanged_count": sum(
                    abs(row["delta_iou"]) <= 1e-12 for row in group
                ),
                "width_mae": _mean(after_errors),
                "delta_width_mae": (
                    _mean(after_errors) - _mean(before_errors)
                    if before_errors and after_errors
                    else None
                ),
                "width_improved_count": sum(
                    row["delta_width_error_mm"] is not None
                    and row["delta_width_error_mm"] < -1e-12
                    for row in group
                ),
                "width_degraded_count": sum(
                    row["delta_width_error_mm"] is not None
                    and row["delta_width_error_mm"] > 1e-12
                    for row in group
                ),
                "island_removed_case_count": sum(
                    int(row.get("removed_pixel_count") or 0) > 0 for row in group
                ),
                "fallback_rate": (
                    sum(bool(row["fallback"]) for row in group) / len(group)
                    if group
                    else None
                ),
                "mean_processing_time_ms": _mean(
                    row["processing_time_ms"] for row in group
                ),
            }
        )
    return summaries


def _debug_needed(row: dict[str, Any], width_degradation_mm: float) -> bool:
    return bool(
        row["delta_iou"] < -0.05
        or (
            row["delta_width_error_mm"] is not None
            and row["delta_width_error_mm"] > float(width_degradation_mm)
        )
        or row["fallback"]
    )


def _render_island_debug(
    rgb_bgr: np.ndarray,
    original_mask: np.ndarray,
    cleaned_mask: np.ndarray,
    gt_mask: np.ndarray,
    component_count: int,
    removed_component_count: int,
    removed_pixel_count: int,
) -> np.ndarray:
    """Create explicit original / cleaned / GT panels for removed-island cases."""
    rgb = np.asarray(rgb_bgr, dtype=np.uint8)
    masks = [
        ("original mask", np.asarray(original_mask, dtype=bool), (0, 215, 255)),
        ("cleaned mask", np.asarray(cleaned_mask, dtype=bool), (0, 0, 255)),
        ("GT", np.asarray(gt_mask, dtype=bool), (0, 255, 0)),
    ]
    panels = []
    for title, mask, color in masks:
        tint = rgb.copy()
        tint[mask] = color
        panel = cv2.addWeighted(rgb, 0.62, tint, 0.38, 0)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(panel, contours, -1, color, 2, cv2.LINE_AA)
        cv2.putText(
            panel,
            title,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            title,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    output = np.concatenate(panels, axis=1)
    footer = (
        f"components={component_count} removed_components={removed_component_count} "
        f"removed_pixels={removed_pixel_count}"
    )
    cv2.putText(
        output,
        footer,
        (12, output.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        footer,
        (12, output.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--indices", default="1-100")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--width-degradation-debug-mm",
        type=float,
        default=2.0,
        help="Save debug when width absolute error worsens by more than this value.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    baseline_run = args.baseline_run.expanduser().resolve()
    baseline_csv = args.baseline_csv.expanduser().resolve()
    document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    variants = _expand_variants(document)
    indices = _parse_indices(args.indices)
    baseline_rows = _read_baseline_rows(baseline_csv)
    intr, depth_scale, _ = _load_intrinsics(EVALUATION_ROOT / "camera_intrinsics.json")
    labels = _parse_labelmap(LABELMAP_PATH)
    if "book spine" not in labels:
        raise KeyError(f"'book spine' is absent from {LABELMAP_PATH}")
    target_rgb = labels["book spine"]

    if args.output_dir is None:
        timestamp = datetime.now().astimezone().strftime("run_%Y%m%d_%H%M%S_%f")
        output_dir = (
            EVALUATION_ROOT / "retrieval_results" / "postprocess_comparison" / timestamp
        )
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config.yaml")
    debug_root = output_dir / "debug"

    rows: list[dict[str, Any]] = []
    total = len(indices)
    for position, case_number in enumerate(indices, start=1):
        case_id = f"{case_number:04d}"
        print(f"[{position}/{total}] case {case_id}", flush=True)
        baseline_row = baseline_rows[case_id]
        case_dir = baseline_run / case_id
        rgb_path = case_dir / "after_init_rgb.png"
        original_path = case_dir / "selected_mask_used_for_depth.png"
        depth_path = case_dir / "comparison_median_depth_filter_depth_masked.npy"
        gt_path = GT_ROOT / f"{case_id}_rgb.png"
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(rgb_path)
        gt = _load_gt_mask(gt_path, target_rgb)
        gt_width = float(baseline_row["gt_width_mm"])

        original = None
        linked: list[dict[str, Any]] = []
        width_before = None
        baseline_width_error = None
        baseline_iou = _optional_float(baseline_row.get("iou")) or 0.0
        baseline_depth = None
        index = _selected_index(case_dir)
        if original_path.is_file():
            original = np.asarray(Image.open(original_path).convert("L")) > 0
            baseline_iou = _mask_iou(original, gt)
            if index is not None:
                linked = load_linked_ocr_polygons(case_dir, index, original.shape)
            if depth_path.is_file():
                baseline_depth = np.load(depth_path, allow_pickle=False)
                try:
                    width_before = _estimate_width_mm(
                        original,
                        baseline_depth,
                        intr,
                        depth_scale,
                        random_seed=case_number,
                    )
                except Exception as exc:
                    print(f"  width-before unavailable: {exc}", flush=True)
            if width_before is not None:
                baseline_width_error = abs(width_before - gt_width)

        for config in variants:
            method = config["method"]
            params = _params_json(config)
            info: dict[str, Any]
            used = original
            width_after = width_before
            processing_ms = 0.0
            fallback = False
            fallback_reason = ""

            if original is None:
                used = np.zeros(gt.shape, dtype=bool)
                info = {
                    "method": method,
                    "axis": None,
                    "axis_origin": None,
                    "fallback": method != "baseline",
                    "fallback_reason": (
                        "selected_mask_unavailable" if method != "baseline" else ""
                    ),
                    "num_linked_ocr": 0,
                    "num_valid_ocr": 0,
                    "estimated_axis_deg": None,
                    "component_count": 0,
                    "removed_component_count": 0,
                    "removed_pixel_count": 0,
                }
                fallback = bool(info["fallback"])
                fallback_reason = str(info["fallback_reason"])
                area_before = 0
                area_after = 0
                area_ratio = 1.0
                iou_before = baseline_iou
                iou_after = baseline_iou
            else:
                result = postprocess_selected_mask(original, linked, config)
                used = result.mask
                info = dict(result.info)
                processing_ms = float(info.get("processing_time_ms", 0.0))
                fallback = bool(info.get("fallback", False))
                fallback_reason = str(info.get("fallback_reason", ""))

                # A downstream failure caused by the candidate is also handled
                # as a safe fallback for the experiment result.
                if (
                    not fallback
                    and method != "baseline"
                    and not np.array_equal(used, original)
                    and baseline_depth is not None
                ):
                    try:
                        width_after = _estimate_width_mm(
                            used,
                            baseline_depth,
                            intr,
                            depth_scale,
                            random_seed=case_number,
                        )
                    except Exception as exc:
                        fallback = True
                        fallback_reason = (
                            f"downstream_width_estimation_failed:{type(exc).__name__}:{exc}"
                        )
                        used = original.copy()
                        width_after = width_before
                        info["fallback"] = True
                        info["fallback_reason"] = fallback_reason
                area_before = int(original.sum())
                area_after = int(used.sum())
                area_ratio = float(area_after / max(area_before, 1))
                iou_before = baseline_iou
                iou_after = _mask_iou(used, gt)

            width_error_after = (
                abs(width_after - gt_width) if width_after is not None else None
            )
            row = {
                "case_id": case_id,
                "method": method,
                "params": params,
                "iou_before": iou_before,
                "iou_after": iou_after,
                "delta_iou": iou_after - iou_before,
                "width_before_mm": width_before,
                "width_after_mm": width_after,
                "width_gt_mm": gt_width,
                "width_error_before_mm": baseline_width_error,
                "width_error_after_mm": width_error_after,
                "delta_width_error_mm": (
                    width_error_after - baseline_width_error
                    if width_error_after is not None and baseline_width_error is not None
                    else None
                ),
                "mask_area_before": area_before,
                "mask_area_after": area_after,
                "area_ratio": area_ratio,
                "num_linked_ocr": int(info.get("num_linked_ocr", len(linked))),
                "num_valid_ocr": int(info.get("num_valid_ocr", 0)),
                "estimated_axis_deg": info.get("estimated_axis_deg"),
                "component_count": info.get("component_count"),
                "removed_component_count": int(
                    info.get("removed_component_count", 0)
                ),
                "removed_pixel_count": int(info.get("removed_pixel_count", 0)),
                "fallback": fallback,
                "fallback_reason": fallback_reason,
                "processing_time_ms": processing_ms,
            }
            rows.append(row)

            save_removed_island_debug = (
                method == "remove_islands" and row["removed_pixel_count"] > 0
            )
            save_debug = (
                save_removed_island_debug
                if method == "remove_islands"
                else (
                    method != "baseline"
                    and bool(config["save_debug_images"])
                    and _debug_needed(row, args.width_degradation_debug_mm)
                )
            )
            if save_debug:
                debug_dir = debug_root / _variant_slug(config)
                debug_dir.mkdir(parents=True, exist_ok=True)
                if save_removed_island_debug:
                    overlay = _render_island_debug(
                        rgb,
                        original,
                        used,
                        gt,
                        int(row["component_count"] or 0),
                        row["removed_component_count"],
                        row["removed_pixel_count"],
                    )
                else:
                    overlay = render_debug_overlay(
                        rgb,
                        np.zeros(gt.shape, dtype=bool) if original is None else original,
                        used,
                        gt,
                        linked,
                        info,
                    )
                path = debug_dir / f"{case_id}.png"
                if not cv2.imwrite(str(path), overlay):
                    raise RuntimeError(f"failed to save debug image: {path}")

        _write_csv(output_dir / "cases.csv", CASE_FIELDS, rows)

    summaries = _summaries(rows, variants)
    _write_csv(output_dir / "summary.csv", SUMMARY_FIELDS, summaries)
    manifest = {
        "baseline_run": str(baseline_run),
        "baseline_csv": str(baseline_csv),
        "config": str(config_path),
        "indices": indices,
        "variant_count": len(variants),
        "case_row_count": len(rows),
        "width_degradation_debug_mm": args.width_degradation_debug_mm,
        "depth_policy": "reuse baseline comparison_median_depth_filter_depth_masked.npy",
        "gt_usage": "metrics and debug visualization only",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"cases:   {output_dir / 'cases.csv'}")
    print(f"summary: {output_dir / 'summary.csv'}")
    print(f"debug:   {debug_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
