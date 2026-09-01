#!/usr/bin/env python3
"""Create offline diagnostics for the storage-space pair-splitting stage.

This script does not run SAM3.  It reloads masks already saved by
``offline_100test_storage_space_sam3.py`` and reproduces ROI, residual, pair
splitting, and final selection while saving auditable intermediate images.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import offline_100test_storage_space_sam3 as pipeline
from core.npz_io import load_npz


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_RUN = (
    PROJECT_ROOT
    / "captures"
    / "100test_storage_space_offline"
    / "20260826_234507"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "captures" / "100test_storage_space_split_debug"
)
DEFAULT_CASES = (1, 17, 35)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose pair-interval storage-space splitting without SAM3 inference."
    )
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case", type=int, action="append")
    return parser.parse_args()


def make_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / stamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{stamp}_{suffix:03d}"
        suffix += 1
    run_dir.mkdir()
    return run_dir.resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(pipeline.jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def draw_pair_intervals(
    rgb: np.ndarray,
    roi: list[int] | None,
    obstacles: list[dict[str, Any]],
    pair_records: list[dict[str, Any]],
) -> np.ndarray:
    output = rgb.copy()
    if roi is None:
        return output
    x1_roi, y1_roi, x2_roi, y2_roi = roi
    colors = pipeline.SPACE_COLORS_RGB

    for pair in pair_records:
        extent = pair.get("rowwise_gap_x_extent") or pair.get("x_interval")
        if extent is None:
            continue
        x1, x2 = extent
        if x2 <= x1:
            continue
        color = np.asarray(colors[(int(pair["pair_id"]) - 1) % len(colors)])
        slab = output[y1_roi:y2_roi, x1:x2].astype(np.float32)
        output[y1_roi:y2_roi, x1:x2] = (
            0.82 * slab + 0.18 * color.astype(np.float32)
        ).astype(np.uint8)
        line_color = tuple(int(value) for value in color)
        cv2.line(output, (x1, y1_roi), (x1, y2_roi - 1), line_color, 2)
        cv2.line(output, (x2 - 1, y1_roi), (x2 - 1, y2_roi - 1), line_color, 2)
        method = pair.get("x_interval_method", "")
        method_tag = "R" if method.startswith("not_used_rowwise") else (
            "C" if method.startswith("center_to_center") else "E"
        )
        label_y = y1_roi + 18 + 18 * ((int(pair["pair_id"]) - 1) % 3)
        cv2.putText(
            output,
            f"P{pair['pair_id']}{method_tag}",
            (x1 + 1, min(y2_roi - 4, label_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            line_color,
            1,
            cv2.LINE_AA,
        )

    for obstacle in obstacles:
        contour_color = (
            (255, 255, 255)
            if obstacle["type"] == "book_spine"
            else (255, 0, 255)
        )
        contours, _ = cv2.findContours(
            obstacle["mask"].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, contours, -1, contour_color, 1)
        center_x = int(round(obstacle["center_x"]))
        cv2.circle(output, (center_x, y2_roi - 6), 3, contour_color, -1)

    cv2.rectangle(
        output,
        (x1_roi, y1_roi),
        (x2_roi - 1, y2_roi - 1),
        pipeline.ROI_COLOR_RGB,
        2,
    )
    return output


def row_gap_metrics(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    roi: list[int],
) -> dict[str, Any]:
    x1_roi, y1_roi, x2_roi, y2_roi = roi
    both_present_rows = 0
    positive_gap_widths: list[int] = []
    for y in range(y1_roi, y2_roi):
        left_x = np.flatnonzero(left_mask[y, x1_roi:x2_roi])
        right_x = np.flatnonzero(right_mask[y, x1_roi:x2_roi])
        if not len(left_x) or not len(right_x):
            continue
        both_present_rows += 1
        left_right_edge = x1_roi + int(left_x.max()) + 1
        right_left_edge = x1_roi + int(right_x.min())
        gap_width = right_left_edge - left_right_edge
        if gap_width > 0:
            positive_gap_widths.append(gap_width)
    return {
        "roi_rows": y2_roi - y1_roi,
        "rows_where_both_obstacles_have_mask_pixels": both_present_rows,
        "rows_with_positive_per_y_gap": len(positive_gap_widths),
        "positive_per_y_gap_width_min_px": (
            min(positive_gap_widths) if positive_gap_widths else None
        ),
        "positive_per_y_gap_width_max_px": (
            max(positive_gap_widths) if positive_gap_widths else None
        ),
        "positive_per_y_gap_width_mean_px": (
            float(np.mean(positive_gap_widths)) if positive_gap_widths else None
        ),
    }


def debug_case(source_run: Path, output_dir: Path, case_id: int) -> dict[str, Any]:
    source_dir = source_run / str(case_id)
    metadata = load_json(source_dir / "metadata.json")
    rgb = pipeline.read_rgb(source_dir / "input.png")
    spine_instances = load_npz(source_dir / pipeline.SPINE_MODEL.npz_name)
    end_instances = load_npz(source_dir / pipeline.BOOK_END_MODEL.npz_name)
    roi = metadata.get("roi_xyxy")
    left = metadata.get("selected_left_book_end")
    right = metadata.get("selected_right_book_end")
    spine_indices = metadata.get("roi_book_spine_indices", [])

    roi_mask, residual = pipeline.residual_from_roi(
        rgb.shape[:2], roi, spine_instances.masks
    )
    spine_union = (
        np.any(spine_instances.masks, axis=0)
        if spine_instances.count
        else np.zeros(rgb.shape[:2], dtype=bool)
    )
    obstacles, obstacle_order = pipeline.build_obstacle_sequence(
        spine_instances.masks,
        end_instances.masks,
        spine_indices,
        left,
        right,
    )
    spaces, rejected, labels, pair_records = (
        pipeline.extract_space_candidates_by_obstacle_pairs(
            residual, roi, obstacles
        )
    )
    selected, selection = pipeline.select_final_space(spaces, labels, roi)

    output_dir.mkdir(parents=True)
    candidate_dir = output_dir / "candidate_masks"
    candidate_dir.mkdir()
    pipeline.save_png(output_dir / "01_input.png", rgb)
    pipeline.save_png(output_dir / "02_roi.png", pipeline.draw_roi(rgb, roi))
    pipeline.save_png(
        output_dir / "03_book_spine_union_mask.png",
        spine_union.astype(np.uint8) * 255,
    )
    pipeline.save_png(
        output_dir / "04_residual_mask_before_split.png",
        residual.astype(np.uint8) * 255,
    )
    pipeline.save_png(
        output_dir / "05_obstacle_or_pair_intervals.png",
        draw_pair_intervals(rgb, roi, obstacles, pair_records),
    )
    pipeline.save_png(
        output_dir / "06_space_candidates_after_split.png",
        pipeline.draw_all_space_candidates(rgb, spaces, labels, roi),
    )
    pipeline.save_png(
        output_dir / "07_final_space_overlay.png",
        pipeline.draw_selected_space(rgb, selected, labels, roi),
    )

    global_component_count, global_labels = cv2.connectedComponents(
        residual.astype(np.uint8),
        connectivity=pipeline.CONNECTED_COMPONENT_CONNECTIVITY,
    )
    pair_by_id = {int(item["pair_id"]): item for item in pair_records}
    obstacle_by_id = {int(item["obstacle_id"]): item for item in obstacles}
    candidate_metrics: list[dict[str, Any]] = []
    for space in spaces:
        space_id = int(space["space_id"])
        mask = labels == space_id
        x1, y1, x2, y2 = space["bbox_xyxy"]
        pair = pair_by_id[int(space["obstacle_pair_id"])]
        extent = pair.get("rowwise_gap_x_extent") or pair.get("x_interval")
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        envelope_area = (
            max(1, (extent[1] - extent[0]) * (roi[3] - roi[1]))
            if extent is not None
            else None
        )
        source_global_labels = sorted(
            int(value) for value in np.unique(global_labels[mask]) if value > 0
        )
        item = {
            **space,
            "x_interval_method": pair["x_interval_method"],
            "bbox_fill_ratio": float(mask.sum() / bbox_area),
            "pair_rowwise_gap_envelope_fill_ratio": (
                float(mask.sum() / envelope_area)
                if envelope_area is not None
                else None
            ),
            "pixels_outside_residual": int(np.count_nonzero(mask & ~residual)),
            "source_global_residual_component_labels": source_global_labels,
        }
        candidate_metrics.append(item)
        pipeline.save_png(
            candidate_dir / f"space_{space_id:02d}.png",
            mask.astype(np.uint8) * 255,
        )

    pair_metrics: list[dict[str, Any]] = []
    if roi is not None:
        for pair in pair_records:
            left_obstacle = obstacle_by_id[int(pair["left_obstacle_id"])]
            right_obstacle = obstacle_by_id[int(pair["right_obstacle_id"])]
            pair_metrics.append(
                {
                    **pair,
                    "per_y_actual_mask_gap": row_gap_metrics(
                        left_obstacle["mask"], right_obstacle["mask"], roi
                    ),
                }
            )

    recomputed_union = labels > 0
    source_candidate_mask = (
        pipeline.read_rgb(source_dir / "space_candidates_mask.png")[:, :, 0] > 0
    )
    report = {
        "case": case_id,
        "source_run": str(source_run.resolve()),
        "source_case_dir": str(source_dir.resolve()),
        "roi_xyxy": roi,
        "spine_mask_count": spine_instances.count,
        "roi_spine_indices": spine_indices,
        "obstacle_order": obstacle_order,
        "pair_metrics": pair_metrics,
        "space_candidates": candidate_metrics,
        "rejected_components": rejected,
        "final_selection": selection,
        "invariants": {
            "residual_equals_roi_minus_spine_union": bool(
                np.array_equal(residual, roi_mask & ~spine_union)
            ),
            "candidate_pixels_outside_residual": int(
                np.count_nonzero(recomputed_union & ~residual)
            ),
            "recomputed_candidates_equal_saved_candidates": bool(
                np.array_equal(recomputed_union, source_candidate_mask)
            ),
            "global_residual_component_count": int(global_component_count - 1),
            "morphology_applied_by_split_function": False,
            "candidate_masks_created_from_bbox_fill": False,
            "candidate_masks_created_from_residual_x_interval_slice": False,
            "candidate_masks_created_from_rowwise_actual_mask_gaps": True,
        },
    }
    save_json(output_dir / "debug_analysis.json", report)
    return report


def main() -> None:
    args = parse_args()
    source_run = args.source_run.resolve()
    if not source_run.is_dir():
        raise FileNotFoundError(f"source run not found: {source_run}")
    case_ids = sorted(set(args.case or DEFAULT_CASES))
    run_dir = make_run_dir(args.output_root.resolve())
    reports = [
        debug_case(source_run, run_dir / str(case_id), case_id)
        for case_id in case_ids
    ]
    save_json(
        run_dir / "summary.json",
        {
            "source_run": str(source_run),
            "case_ids": case_ids,
            "output_dir": str(run_dir),
            "reports": reports,
        },
    )
    print(run_dir)


if __name__ == "__main__":
    main()
