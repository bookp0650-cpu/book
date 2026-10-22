#!/usr/bin/env python3
"""Export disconnected-component cleanup from saved case_artifacts only.

remove_islands preserves its OCR-linked 8-connected main component bit-for-bit
and deletes only other components.  GT is passed only to metric and comparison
image code; it is never passed to the postprocessor.

When pixels are removed, downstream width/target geometry is replayed from the
cleaned mask and final_recognition.png is redrawn with the production renderer.
This keeps the target marker on the same cleaned geometry used for measurement;
the geometry replay never edits the cleaned mask.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image

from detection.pro_handbook.sam_py_demo import get_book_points as current
from detection.pro_handbook.sam_py_demo.book_mask_postprocess import (
    load_linked_ocr_polygons,
    postprocess_selected_mask,
)
from detection.pro_handbook.sam_py_demo.get_book_points_no_mask_merge_no_side_filter import (
    _save_final_png_with_target,
)
from detection.pro_handbook.sam_py_demo.modules.book_width import estimate_book_width
from detection.pro_handbook.sam_py_demo.modules.grip_point import find_target_point
from detection.pro_handbook.sam_py_demo.modules.pca_vector import pca_axes_fix_dir
from detection.pro_handbook.thesis.evaluate_book_mask_postprocess import (
    _mask_iou,
    _selected_index,
)
from detection.pro_handbook.thesis.thesis_ret import _load_intrinsics


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "captures"
    / "thesis"
    / "ret"
    / "retrieval_results"
    / "case_artifacts"
    / "run_20261006_023515_527898"
)
DEFAULT_DESTINATION = (
    PROJECT_ROOT / "captures" / "thesis" / "ret" / "refined" / "1~100"
)

RESULT_FIELDS = [
    "case_id",
    "before_iou",
    "after_iou",
    "iou_diff",
    "before_width_error_mm",
    "after_width_error_mm",
    "width_error_diff_mm",
    "removed_pixels",
    "removed_components",
    "before_mask_pixels",
    "after_mask_pixels",
    "recognition_success_before",
    "recognition_success_after",
    "iou_threshold",
    "before_width_mm",
    "after_width_mm",
    "width_gt_mm",
    "component_count",
    "method",
    "fallback",
    "fallback_reason",
]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bbox_dict(mask: np.ndarray) -> dict[str, float] | None:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return None
    return {
        "x1": float(xs.min()),
        "y1": float(ys.min()),
        "x2": float(xs.max()),
        "y2": float(ys.max()),
    }


def _filtered_depth_for_refined_mask(
    source_case_dir: Path,
    refined_mask: np.ndarray,
    depth: np.ndarray,
    query: str,
    *,
    depth_merge_tolerance_raw: int = 30,
) -> np.ndarray:
    """Recreate only the saved pipeline's post-selection median-depth step."""
    box = _bbox_dict(refined_mask)
    merged = {
        "results": [{"box": box, "forced_angle": 90}] if box is not None else [],
        "mask01": np.asarray(refined_mask, dtype=np.uint8),
    }
    _, depth_anchor_info = current.refine_mask_by_ocr_axis_band(
        mask01=np.asarray(refined_mask, dtype=np.uint8),
        merged=merged,
        image_shape=refined_mask.shape,
        shot_dir=source_case_dir,
        query=query,
        mask_width_ratio=1.05,
        min_keep_ratio=0.85,
        use_ocr_short_width=False,
        ocr_short_to_half_width_scale=1.35,
        min_ocr_half_width_px=28.0,
        suppress_side_protrusions=False,
        profile_bin_size_px=8.0,
        profile_width_margin=1.15,
        min_profile_half_width_px=16.0,
        max_profile_shrink_ratio=0.35,
        return_info=True,
    )
    reference_mask, _ = current._make_depth_reference_mask_from_ocr_info(
        depth_anchor_info,
        refined_mask.shape,
        selected_mask01=np.asarray(refined_mask, dtype=np.uint8),
        min_intersection_px=30,
    )

    selected = np.asarray(refined_mask, dtype=bool)
    filtered = np.asarray(depth).copy()
    filtered[~selected] = 0
    selected_values = filtered[filtered > 0]
    reference_values = np.asarray([], dtype=filtered.dtype)
    if reference_mask is not None:
        reference_valid = (
            (np.asarray(reference_mask) > 0) & selected & (np.asarray(depth) > 0)
        )
        reference_values = np.asarray(depth)[reference_valid]
    if reference_values.size < 10:
        reference_values = selected_values
    if reference_values.size == 0:
        raise RuntimeError("no valid depth in refined mask")
    median_raw = int(np.median(reference_values))
    keep = (
        (filtered >= median_raw - int(depth_merge_tolerance_raw))
        & (filtered <= median_raw + int(depth_merge_tolerance_raw))
    )
    filtered[~keep] = 0
    return filtered


def _replay_refined_geometry(
    refined_mask: np.ndarray,
    filtered_depth: np.ndarray,
    intr: Any,
    depth_scale: float,
    *,
    random_seed: int,
) -> dict[str, Any]:
    """Replay unchanged downstream geometry from the remove_islands mask.

    This does not alter the mask.  It keeps the production depth/RANSAC/PCA
    path, then derives width and target from the same retained point set so the
    target drawn in final_recognition.png cannot remain on a removed island.
    """
    mask = np.asarray(refined_mask, dtype=bool)
    depth = np.asarray(filtered_depth).copy()
    depth[~mask] = 0
    points, uv = current._mask_depth_to_points_uv_for_plane_filter(
        mask.astype(np.uint8), depth, intr, depth_scale
    )
    if points.shape[0] < 3:
        raise RuntimeError(f"too few depth points after remove_islands: {points.shape[0]}")
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
        raise RuntimeError(f"normal geometry RANSAC failed: {ransac_info}")
    points_for_pca = points[inliers]
    uv_for_pca = uv[inliers]
    if points_for_pca.shape[0] < 10:
        raise RuntimeError(
            f"too few geometry inliers after remove_islands: {points_for_pca.shape[0]}"
        )

    mean, pc1, pc2 = pca_axes_fix_dir(points_for_pca)
    width_info = estimate_book_width(points_for_pca, mean, pc1, pc2)
    width_m = width_info.get("av_book_width_m")
    if width_m is None or not math.isfinite(float(width_m)):
        raise RuntimeError(f"book width estimation failed: {width_info}")
    target_info = find_target_point(points_for_pca)
    target = target_info.get("target_m")
    if target is None:
        raise RuntimeError(f"target point estimation failed: {target_info}")
    target = np.asarray(target, dtype=np.float64)
    target_index = int(
        np.argmin(np.linalg.norm(points_for_pca.astype(np.float64) - target, axis=1))
    )
    target_pixel = tuple(int(value) for value in uv_for_pca[target_index])
    final_mask = np.zeros_like(mask, dtype=bool)
    final_mask[uv_for_pca[:, 1], uv_for_pca[:, 0]] = True
    return {
        "width_mm": float(width_m) * 1000.0,
        "target_m": target,
        "target_pixel": target_pixel,
        "points_for_pca": points_for_pca,
        "uv_for_pca": uv_for_pca,
        "final_mask": final_mask,
        "ransac_info": ransac_info,
    }


def _mean(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return statistics.fmean(valid) if valid else None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    width_pairs = [
        row
        for row in rows
        if row["before_width_error_mm"] is not None
        and row["after_width_error_mm"] is not None
    ]
    before_mae = _mean(row["before_width_error_mm"] for row in width_pairs)
    after_mae = _mean(row["after_width_error_mm"] for row in width_pairs)
    before_iou = [float(row["before_iou"]) for row in rows]
    after_iou = [float(row["after_iou"]) for row in rows]

    def iou_stats(values: list[float]) -> dict[str, Any]:
        ordered = sorted(values)
        return {
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "minimum": ordered[0] if ordered else None,
            "second_minimum": ordered[1] if len(ordered) >= 2 else None,
            "maximum": ordered[-1] if ordered else None,
            "count_ge_0_90": sum(value >= 0.90 for value in values),
            "count_ge_0_95": sum(value >= 0.95 for value in values),
        }

    def iou_case(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "case_id": row["case_id"],
            "before_iou": row["before_iou"],
            "after_iou": row["after_iou"],
            "iou_diff": row["iou_diff"],
        }

    improved = [row for row in rows if row["iou_diff"] > 1e-12]
    degraded = [row for row in rows if row["iou_diff"] < -1e-12]
    unchanged = [row for row in rows if abs(row["iou_diff"]) <= 1e-12]
    success_to_failure = [
        row["case_id"]
        for row in rows
        if row["recognition_success_before"]
        and not row["recognition_success_after"]
    ]
    failure_to_success = [
        row["case_id"]
        for row in rows
        if not row["recognition_success_before"]
        and row["recognition_success_after"]
    ]
    removed_values = [int(row["removed_pixels"]) for row in rows]
    top_removed = sorted(
        rows, key=lambda row: (-int(row["removed_pixels"]), row["case_id"])
    )[:10]
    total = len(rows)
    success_before = sum(bool(row["recognition_success_before"]) for row in rows)
    success_after = sum(bool(row["recognition_success_after"]) for row in rows)
    return {
        "case_count": len(rows),
        "processed_count": sum(not bool(row["fallback"]) for row in rows),
        "unprocessable_count": sum(bool(row["fallback"]) for row in rows),
        "iou": {
            "before": iou_stats(before_iou),
            "after": iou_stats(after_iou),
            "mean_diff": _mean(row["iou_diff"] for row in rows),
            "improved_count": len(improved),
            "unchanged_count": len(unchanged),
            "degraded_count": len(degraded),
            "maximum_improvement": (
                iou_case(max(improved, key=lambda row: row["iou_diff"]))
                if improved else None
            ),
            "maximum_degradation": (
                iou_case(min(degraded, key=lambda row: row["iou_diff"]))
                if degraded else None
            ),
            "degraded_cases": [
                iou_case(row)
                for row in sorted(degraded, key=lambda row: row["iou_diff"])
            ],
        },
        "recognition": {
            "iou_thresholds_from_artifacts": sorted(
                {float(row["iou_threshold"]) for row in rows}
            ),
            "success_before_count": success_before,
            "success_after_count": success_after,
            "success_before_rate_percent": (
                100.0 * success_before / total if total else 0.0
            ),
            "success_after_rate_percent": (
                100.0 * success_after / total if total else 0.0
            ),
            "success_to_failure_count": len(success_to_failure),
            "success_to_failure_cases": success_to_failure,
            "failure_to_success_count": len(failure_to_success),
            "failure_to_success_cases": failure_to_success,
        },
        "width": {
            "valid_pair_count": len(width_pairs),
            "mae_before_mm": before_mae,
            "mae_after_mm": after_mae,
            "mae_diff_mm": (
                after_mae - before_mae
                if before_mae is not None and after_mae is not None
                else None
            ),
            "improved_count": sum(
                row["width_error_diff_mm"] is not None
                and row["width_error_diff_mm"] < -1e-12
                for row in rows
            ),
            "degraded_count": sum(
                row["width_error_diff_mm"] is not None
                and row["width_error_diff_mm"] > 1e-12
                for row in rows
            ),
            "unchanged_count": sum(
                row["width_error_diff_mm"] is not None
                and abs(row["width_error_diff_mm"]) <= 1e-12
                for row in rows
            ),
        },
        "remove_islands": {
            "changed_case_count": sum(value > 0 for value in removed_values),
            "unchanged_case_count": sum(value == 0 for value in removed_values),
            "removed_pixels_total": sum(removed_values),
            "removed_pixels_mean_all_cases": (
                statistics.fmean(removed_values) if removed_values else None
            ),
            "removed_pixels_median_all_cases": (
                statistics.median(removed_values) if removed_values else None
            ),
            "removed_pixels_maximum": max(removed_values) if removed_values else None,
            "removed_components_total": sum(
                int(row["removed_components"]) for row in rows
            ),
            "top_10_removed_pixels": [
                {
                    "case_id": row["case_id"],
                    "removed_pixels": int(row["removed_pixels"]),
                    "removed_components": int(row["removed_components"]),
                }
                for row in top_removed
            ],
        },
    }


def _validate_source(source: Path) -> list[str]:
    expected = [f"{number:04d}" for number in range(1, 101)]
    actual = sorted(path.name for path in source.iterdir() if path.is_dir())
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"source case directories mismatch: missing={missing}, extra={extra}")
    for case_id in expected:
        case_dir = source / case_id
        for required in ("input_rgb.png", "input_depth.npy", "gt_mask.png", "result.json"):
            if not (case_dir / required).is_file():
                raise FileNotFoundError(case_dir / required)
    return expected


def _render_comparison(
    rgb: np.ndarray,
    original: np.ndarray,
    refined: np.ndarray,
    gt: np.ndarray,
    *,
    component_count: int,
    removed_component_count: int,
    removed_pixel_count: int,
) -> np.ndarray:
    """Render RGB / Before / After / GT; GT is visualization-only."""

    def panel(mask: np.ndarray | None, color: tuple[int, int, int], title: str):
        image = rgb.copy()
        if mask is not None:
            active = np.asarray(mask, dtype=bool)
            colored = image.copy()
            colored[active] = color
            image = cv2.addWeighted(image, 0.65, colored, 0.35, 0)
            contours, _ = cv2.findContours(
                active.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)
        cv2.putText(
            image, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
            0.72, (255, 255, 255), 3, cv2.LINE_AA,
        )
        cv2.putText(
            image, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
            0.72, (0, 0, 0), 1, cv2.LINE_AA,
        )
        return image

    panels = [
        panel(None, (0, 0, 0), "Original RGB"),
        panel(original, (0, 255, 255), "Before: selected mask"),
        panel(refined, (0, 0, 255), "After: remove_islands"),
        panel(gt, (0, 255, 0), "GT (evaluation only)"),
    ]
    comparison = np.concatenate(panels, axis=1)
    cv2.putText(
        comparison,
        (
            f"components={component_count}  "
            f"removed_components={removed_component_count}  "
            f"removed_pixels={removed_pixel_count}"
        ),
        (12, comparison.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        comparison,
        (
            f"components={component_count}  "
            f"removed_components={removed_component_count}  "
            f"removed_pixels={removed_pixel_count}"
        ),
        (12, comparison.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return comparison


def _process_case(source_case: Path, destination_case: Path, case_id: str) -> dict[str, Any]:
    destination_case.mkdir(parents=False, exist_ok=False)
    # Preserve every packaged artifact that exists.  No source file is modified.
    for source_file in sorted(source_case.iterdir()):
        if source_file.is_file():
            shutil.copy2(source_file, destination_case / source_file.name)

    result_json = _read_json(source_case / "result.json")
    gt = np.asarray(Image.open(source_case / "gt_mask.png").convert("L")) > 0
    selected_path = source_case / "selected_mask.png"
    selected_available = selected_path.is_file()
    iou_threshold = float(result_json["iou_threshold"])
    width_gt = float(result_json["gt_width_mm"])
    width_before = _optional_float(result_json.get("pred_width_mm"))
    fallback = False
    fallback_reason = ""
    component_count = 0
    removed_component_count = 0
    removed_pixel_count = 0

    if selected_path.is_file():
        original = np.asarray(Image.open(selected_path).convert("L")) > 0
        selected_index = _selected_index(source_case)
        linked = (
            load_linked_ocr_polygons(source_case, selected_index, original.shape)
            if selected_index is not None
            else []
        )
        # Deliberately pass only the selected mask and saved OCR linkage.  The
        # GT mask loaded above is reserved for IoU and visualization below.
        postprocess = postprocess_selected_mask(
            original,
            linked,
            {
                "enabled": True,
                "method": "remove_islands",
            },
        )
        refined = np.asarray(postprocess.mask, dtype=bool)
        component_count = int(postprocess.info.get("component_count") or 0)
        removed_component_count = int(
            postprocess.info.get("removed_component_count") or 0
        )
        removed_pixel_count = int(postprocess.info.get("removed_pixel_count") or 0)
        fallback = bool(postprocess.info.get("fallback", False))
        fallback_reason = str(postprocess.info.get("fallback_reason", ""))
    else:
        original = np.zeros_like(gt, dtype=bool)
        refined = original.copy()
        fallback = True
        source_reason = str(result_json.get("error_reason") or "").strip()
        fallback_reason = "selected_mask_unavailable"
        if source_reason:
            fallback_reason += f": {source_reason}"

    if original.shape != gt.shape or refined.shape != gt.shape:
        raise ValueError(
            f"mask shape mismatch in {case_id}: original={original.shape}, "
            f"refined={refined.shape}, gt={gt.shape}"
        )
    if np.any(refined & ~original):
        raise AssertionError(f"remove_islands added pixels in {case_id}")

    iou_before = _mask_iou(original, gt)
    iou_after = _mask_iou(refined, gt)
    width_after = width_before
    refined_geometry: dict[str, Any] | None = None
    if not fallback and not np.array_equal(original, refined):
        depth = np.load(source_case / "input_depth.npy", allow_pickle=False)
        intr, depth_scale, _ = _load_intrinsics(source_case / "camera_intrinsics.json")
        filtered_depth = _filtered_depth_for_refined_mask(
            source_case,
            refined,
            depth,
            str(result_json.get("target_title") or ""),
        )
        refined_geometry = _replay_refined_geometry(
            refined,
            filtered_depth,
            intr,
            depth_scale,
            random_seed=int(case_id),
        )
        width_after = float(refined_geometry["width_mm"])

    width_error_before = (
        abs(width_before - width_gt) if width_before is not None else None
    )
    width_error_after = (
        abs(width_after - width_gt) if width_after is not None else None
    )
    delta_width_error = (
        width_error_after - width_error_before
        if width_error_before is not None and width_error_after is not None
        else None
    )

    original_u8 = original.astype(np.uint8) * 255
    refined_u8 = refined.astype(np.uint8) * 255
    Image.fromarray(original_u8, mode="L").save(destination_case / "original_mask.png")
    Image.fromarray(refined_u8, mode="L").save(destination_case / "refined_mask.png")
    # gt_mask.png was copied byte-for-byte above.
    rgb = cv2.imread(str(source_case / "input_rgb.png"), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(source_case / "input_rgb.png")
    comparison = _render_comparison(
        rgb,
        original,
        refined,
        gt,
        component_count=component_count,
        removed_component_count=removed_component_count,
        removed_pixel_count=removed_pixel_count,
    )
    if not cv2.imwrite(str(destination_case / "comparison.png"), comparison):
        raise RuntimeError(f"failed to save comparison image for {case_id}")
    if refined_geometry is not None:
        rendered_target_pixel = _save_final_png_with_target(
            destination_case / "final_recognition.png",
            rgb,
            refined_geometry["final_mask"],
            refined_geometry["target_m"],
            refined_geometry["points_for_pca"],
            refined_geometry["uv_for_pca"],
        )
        if tuple(rendered_target_pixel) != tuple(refined_geometry["target_pixel"]):
            raise AssertionError(
                f"rendered target mismatch in {case_id}: "
                f"{rendered_target_pixel} != {refined_geometry['target_pixel']}"
            )

    recognition_success_before = bool(
        selected_available and iou_before >= iou_threshold
    )
    recognition_success_after = bool(
        selected_available and iou_after >= iou_threshold
    )
    row = {
        "case_id": case_id,
        "before_iou": iou_before,
        "after_iou": iou_after,
        "iou_diff": iou_after - iou_before,
        "before_width_error_mm": width_error_before,
        "after_width_error_mm": width_error_after,
        "width_error_diff_mm": delta_width_error,
        "removed_pixels": removed_pixel_count,
        "removed_components": removed_component_count,
        "before_mask_pixels": int(original.sum()),
        "after_mask_pixels": int(refined.sum()),
        "recognition_success_before": recognition_success_before,
        "recognition_success_after": recognition_success_after,
        "iou_threshold": iou_threshold,
        "before_width_mm": width_before,
        "after_width_mm": width_after,
        "width_gt_mm": width_gt,
        "method": "remove_islands",
        "component_count": component_count,
        "fallback": fallback,
        "fallback_reason": fallback_reason,
        "final_recognition_regenerated": refined_geometry is not None,
        "target_pixel_after": (
            list(refined_geometry["target_pixel"])
            if refined_geometry is not None
            else None
        ),
        "target_point_after_m": (
            np.asarray(refined_geometry["target_m"], dtype=float).tolist()
            if refined_geometry is not None
            else None
        ),
    }
    refinement_info = dict(row)
    # Backward-compatible aliases used by the earlier case0072 review files.
    refinement_info.update(
        {
            "iou_before": iou_before,
            "iou_after": iou_after,
            "delta_iou": iou_after - iou_before,
            "width_before_mm": width_before,
            "width_after_mm": width_after,
            "width_error_before_mm": width_error_before,
            "width_error_after_mm": width_error_after,
            "delta_width_error_mm": delta_width_error,
            "removed_component_count": removed_component_count,
            "removed_pixel_count": removed_pixel_count,
            "mask_area_before": int(original.sum()),
            "mask_area_after": int(refined.sum()),
            "input_run": source_case.parent.name,
            "gt_usage": "evaluation_and_visualization_only_after_postprocess",
        }
    )
    _write_json(destination_case / "refinement_info.json", refinement_info)
    return row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"destination already exists; refusing overwrite: {destination}")
    case_ids = _validate_source(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".1~100_staging_", dir=str(destination.parent))
    )
    rows: list[dict[str, Any]] = []
    try:
        for position, case_id in enumerate(case_ids, start=1):
            print(f"[{position}/100] {case_id}", flush=True)
            rows.append(
                _process_case(source / case_id, staging / case_id, case_id)
            )
        summary = _summary(rows)
        _write_csv(staging / "results.csv", RESULT_FIELDS, rows)
        _write_json(staging / "summary.json", summary)
        run_info = {
            "created_at": datetime.now().astimezone().isoformat(),
            "source": str(source),
            "destination": str(destination),
            "method": "remove_islands",
            "case_range": ["0001", "0100"],
            "case_count": 100,
            "input_policy": (
                "saved case_artifacts only; SAM/OCR inference, OCR-SAM matching, "
                "and target-mask selection were not rerun"
            ),
            "evaluation_only": True,
            "production_default_changed": False,
            "success_definition": (
                "existing thesis_ret definition: IoU >= per-case artifact "
                "iou_threshold and predicted mask exists"
            ),
            "gt_policy": "evaluation and visualization only after postprocess",
            "copy_policy": "all existing per-case source artifacts copied with copy2",
            "missing_selected_mask_policy": (
                "write zero original/refined masks and record selected_mask_unavailable"
            ),
            "summary": summary,
        }
        _write_json(staging / "run_info.json", run_info)

        actual = sorted(path.name for path in staging.iterdir() if path.is_dir())
        if actual != case_ids:
            raise AssertionError("staging case directory verification failed")
        for case_id in case_ids:
            for required in (
                "original_mask.png",
                "refined_mask.png",
                "final_recognition.png",
                "gt_mask.png",
                "comparison.png",
                "refinement_info.json",
            ):
                if not (staging / case_id / required).is_file():
                    raise FileNotFoundError(staging / case_id / required)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"created_case_directories={summary['case_count']}")
    print(f"processed={summary['processed_count']}")
    print(f"unprocessable={summary['unprocessable_count']}")
    print(
        "changed_by_remove_islands="
        f"{summary['remove_islands']['changed_case_count']}"
    )
    print(f"mean_iou_before={summary['iou']['before']['mean']:.12f}")
    print(f"mean_iou_after={summary['iou']['after']['mean']:.12f}")
    print(f"width_mae_before={summary['width']['mae_before_mm']:.12f}")
    print(f"width_mae_after={summary['width']['mae_after_mm']:.12f}")
    print(f"output={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
