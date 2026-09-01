#!/usr/bin/env python3
"""Audit why a desired storage candidate loses final selection.

The script only reads an existing offline run.  It does not run SAM3 or change
the extraction/selection algorithms.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

import offline_100test_storage_space_sam3 as pipeline
from core.npz_io import load_npz


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_RUN = (
    PROJECT_ROOT / "captures" / "100test_storage_space_offline" / "20260827_123345"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "captures" / "100test_storage_selection_debug"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case", type=int, default=17)
    parser.add_argument("--target-space", type=int, default=11)
    return parser.parse_args()


def make_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / stamp
    suffix = 1
    while path.exists():
        path = root / f"{stamp}_{suffix:03d}"
        suffix += 1
    path.mkdir()
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(pipeline.jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def true_runs(flags: list[tuple[int, bool]]) -> list[list[int]]:
    rows = [y for y, value in flags if value]
    if not rows:
        return []
    runs: list[list[int]] = []
    start = end = rows[0]
    for y in rows[1:]:
        if y == end + 1:
            end = y
        else:
            runs.append([start, end])
            start = end = y
    runs.append([start, end])
    return runs


def blend(output: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    color_array = np.asarray(color, dtype=np.float32)
    output[mask] = (
        (1.0 - alpha) * output[mask].astype(np.float32) + alpha * color_array
    ).astype(np.uint8)


def main() -> None:
    args = parse_args()
    source_run = args.source_run.resolve()
    source_dir = source_run / str(args.case)
    metadata = load_json(source_dir / "metadata.json")
    selected_space_id = int(metadata["selected_space_id"])
    target = next(item for item in metadata["spaces"] if int(item["space_id"]) == args.target_space)
    selected = next(item for item in metadata["spaces"] if int(item["space_id"]) == selected_space_id)
    pair = next(
        item
        for item in metadata["adjacent_obstacle_pairs"]
        if int(item["pair_id"]) == int(target["obstacle_pair_id"])
    )
    obstacle_by_id = {int(item["obstacle_id"]): item for item in metadata["obstacle_order"]}
    left_obstacle = obstacle_by_id[int(pair["left_obstacle_id"])]
    right_obstacle = obstacle_by_id[int(pair["right_obstacle_id"])]

    rgb = pipeline.read_rgb(source_dir / "input.png")
    residual = np.asarray(Image.open(source_dir / "residual_mask.png")) > 0
    spine_instances = load_npz(source_dir / pipeline.SPINE_MODEL.npz_name)
    end_instances = load_npz(source_dir / pipeline.BOOK_END_MODEL.npz_name)

    def obstacle_mask(obstacle: dict[str, Any]) -> np.ndarray:
        instances = spine_instances if obstacle["type"] == "book_spine" else end_instances
        return instances.masks[int(obstacle["instance_index"])]

    left_mask = obstacle_mask(left_obstacle)
    right_mask = obstacle_mask(right_obstacle)
    roi = metadata["roi_xyxy"]
    x1_roi, y1_roi, x2_roi, y2_roi = roi
    target_mask = np.asarray(
        Image.open(source_dir / "candidate_masks" / f"space_{args.target_space:02d}.png")
    ) > 0
    selected_mask = np.asarray(
        Image.open(source_dir / "candidate_masks" / f"space_{selected_space_id:02d}.png")
    ) > 0

    labels = np.zeros(residual.shape, dtype=np.int32)
    candidate_masks: dict[int, np.ndarray] = {}
    for item in metadata["spaces"]:
        space_id = int(item["space_id"])
        mask = np.asarray(
            Image.open(source_dir / "candidate_masks" / f"space_{space_id:02d}.png")
        ) > 0
        labels[mask] = space_id
        candidate_masks[space_id] = mask
    candidate_union = labels > 0

    _, global_labels, stats, _ = cv2.connectedComponentsWithStats(
        residual.astype(np.uint8), connectivity=pipeline.CONNECTED_COMPONENT_CONNECTIVITY
    )
    component_ids = [int(value) for value in np.unique(global_labels[target_mask]) if value > 0]
    if len(component_ids) != 1:
        raise RuntimeError(f"target overlaps unexpected residual components: {component_ids}")
    component_id = component_ids[0]
    component = global_labels == component_id
    shared_candidate_ids = sorted(
        space_id for space_id, mask in candidate_masks.items() if np.any(mask & component)
    )

    band_y1, band_y2 = target["bottom_band_y_range"]
    component_bottom_band = np.zeros(residual.shape, dtype=bool)
    component_bottom_band[band_y1:band_y2] = component[band_y1:band_y2]
    unassigned_component_bottom_band = component_bottom_band & ~candidate_union

    row_records: list[dict[str, Any]] = []
    for y in range(y1_roi, y2_roi):
        left_x = np.flatnonzero(left_mask[y, x1_roi:x2_roi])
        right_x = np.flatnonzero(right_mask[y, x1_roi:x2_roi])
        row_records.append(
            {
                "y": y,
                "left_boundary_available": bool(len(left_x)),
                "right_boundary_available": bool(len(right_x)),
                "left_rightmost_x": x1_roi + int(left_x.max()) if len(left_x) else None,
                "right_leftmost_x": x1_roi + int(right_x.min()) if len(right_x) else None,
                "target_mask_width_px": pipeline.maximum_contiguous_width(target_mask[y]),
                "residual_component_width_px": pipeline.maximum_contiguous_width(component[y]),
            }
        )

    left_runs = true_runs([(item["y"], item["left_boundary_available"]) for item in row_records])
    right_runs = true_runs([(item["y"], item["right_boundary_available"]) for item in row_records])
    both_runs = true_runs(
        [
            (
                item["y"],
                item["left_boundary_available"] and item["right_boundary_available"],
            )
            for item in row_records
        ]
    )
    target_runs = true_runs([(y, bool(target_mask[y].any())) for y in range(y1_roi, y2_roi)])
    target_last_y = int(np.where(target_mask)[0].max())
    component_bbox = [
        int(stats[component_id, cv2.CC_STAT_LEFT]),
        int(stats[component_id, cv2.CC_STAT_TOP]),
        int(stats[component_id, cv2.CC_STAT_LEFT] + stats[component_id, cv2.CC_STAT_WIDTH]),
        int(stats[component_id, cv2.CC_STAT_TOP] + stats[component_id, cv2.CC_STAT_HEIGHT]),
    ]

    report = {
        "case": args.case,
        "source_run": str(source_run),
        "target_space_id": args.target_space,
        "selected_space_id": selected_space_id,
        "target_candidate": target,
        "selected_candidate": selected,
        "target_obstacle_pair": pair,
        "left_obstacle": left_obstacle,
        "right_obstacle": right_obstacle,
        "boundary_presence": {
            "left_present_y_runs_in_roi": left_runs,
            "right_present_y_runs_in_roi": right_runs,
            "both_present_y_runs_in_roi": both_runs,
            "target_mask_y_runs": target_runs,
            "target_last_y": target_last_y,
            "roi_last_y": y2_roi - 1,
            "target_distance_to_roi_bottom_px": (y2_roi - 1) - target_last_y,
            "row_records": row_records,
        },
        "target_global_residual_component": {
            "component_id": component_id,
            "area_px": int(component.sum()),
            "bbox_xyxy": component_bbox,
            "reaches_roi_bottom": bool(component[y2_roi - 1].any()),
            "bottom_band_pixels": int(component_bottom_band.sum()),
            "bottom_band_unassigned_pixels": int(unassigned_component_bottom_band.sum()),
            "candidate_ids_sharing_component": shared_candidate_ids,
        },
        "audit": {
            "target_pixels_outside_residual": int(np.count_nonzero(target_mask & ~residual)),
            "selected_pixels_outside_residual": int(np.count_nonzero(selected_mask & ~residual)),
            "fixed_bbox_or_center_fallback_used": False,
            "first_disadvantage_stage": (
                "rowwise pair-mask generation: missing left obstacle rows are skipped"
            ),
            "reaches_roi_bottom_consistent_with_candidate_mask": bool(
                target["reaches_roi_bottom"]
                == (target_last_y >= y2_roi - 1 - int(target["roi_bottom_tolerance_px"]))
            ),
            "bottom_band_contains_target_pixels": bool(target_mask[band_y1:band_y2].any()),
        },
    }

    run_dir = make_run_dir(args.output_root.resolve())
    output_dir = run_dir / str(args.case)
    output_dir.mkdir()
    for name in (
        "input.png",
        "residual_mask.png",
        "space_candidates_overlay.png",
        "final_space_overlay.png",
        "metadata.json",
    ):
        shutil.copy2(source_dir / name, output_dir / name)
    shutil.copytree(source_dir / "candidate_masks", output_dir / "candidate_masks")
    pipeline.save_png(output_dir / "target_candidate_mask.png", target_mask.astype(np.uint8) * 255)
    pipeline.save_png(output_dir / "selected_candidate_mask.png", selected_mask.astype(np.uint8) * 255)
    pipeline.save_png(
        output_dir / "target_global_residual_component_mask.png",
        component.astype(np.uint8) * 255,
    )
    pipeline.save_png(
        output_dir / "target_component_unassigned_bottom_band_mask.png",
        unassigned_component_bottom_band.astype(np.uint8) * 255,
    )

    comparison = rgb.copy()
    blend(comparison, component & ~candidate_union, (255, 210, 0), 0.22)
    blend(comparison, target_mask, (40, 140, 255), 0.55)
    blend(comparison, selected_mask, (255, 0, 0), 0.60)
    cv2.line(comparison, (x1_roi, band_y1), (x2_roi - 1, band_y1), (0, 255, 255), 2)
    cv2.rectangle(comparison, (x1_roi, y1_roi), (x2_roi - 1, y2_roi - 1), (255, 255, 0), 2)
    cv2.putText(comparison, f"target space {args.target_space}", (target["bbox_xyxy"][0], target["bbox_xyxy"][1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 140, 255), 2, cv2.LINE_AA)
    cv2.putText(comparison, f"selected space {selected_space_id}", (selected["bbox_xyxy"][0], max(20, selected["bbox_xyxy"][1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 0, 0), 2, cv2.LINE_AA)
    pipeline.save_png(output_dir / "target_vs_selected_overlay.png", comparison)

    boundary = rgb.copy()
    blend(boundary, unassigned_component_bottom_band, (255, 210, 0), 0.55)
    blend(boundary, target_mask, (40, 140, 255), 0.50)
    for mask, color in ((left_mask, (255, 255, 255)), (right_mask, (255, 0, 255))):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(boundary, contours, -1, color, 2)
    cv2.line(boundary, (x1_roi, target_last_y), (x2_roi - 1, target_last_y), (255, 128, 0), 2)
    cv2.line(boundary, (x1_roi, band_y1), (x2_roi - 1, band_y1), (0, 255, 255), 2)
    cv2.putText(boundary, f"target last y={target_last_y}", (x1_roi + 8, target_last_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 128, 0), 2, cv2.LINE_AA)
    cv2.putText(boundary, f"bottom band y={band_y1}:{band_y2}", (x1_roi + 8, band_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    pipeline.save_png(output_dir / "target_pair_boundary_debug.png", boundary)

    debug_overlay = pipeline.draw_all_space_candidates(rgb, metadata["spaces"], labels, roi)
    for item in metadata["spaces"]:
        x, y, _, _ = item["bbox_xyxy"]
        cv2.putText(
            debug_overlay,
            f"r={int(bool(item['reaches_roi_bottom']))}",
            (x + 2, max(34, y + 34)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    pipeline.save_png(output_dir / "space_candidates_overlay_debug.png", debug_overlay)
    save_json(output_dir / "debug_selection_analysis.json", report)
    save_json(run_dir / "summary.json", {"output_dir": str(run_dir), **report})
    print(run_dir)


if __name__ == "__main__":
    main()
