#!/usr/bin/env python3
"""Profile current storage recognition on saved RGB-D captures.

This runner never opens RealSense, ROS, xArm, or Dynamixel.  It intentionally
uses the current SAM3 storage service and current Storage_SAM3 recognition
entry point, while keeping all per-capture debug outputs temporary.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import offline_100test_storage_space_sam3 as storage_recognition
from detection.pro_handbook.sam_py_demo.Storage_SAM3 import (
    StorageSam3Runtime,
    intrinsics_from_camera_params,
    load_saved_capture_rgb,
    run_from_rgbd,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "captures" / "入庫認識失敗"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "recognition timing result"

IDENTITY_FIELDS = (
    "capture_index",
    "cycle",
    "view",
    "shot_dir",
    "recognition_success",
    "selected_space_id",
    "error",
)
TIMING_FIELDS = (
    "total_ms",
    "rgbd_ms",
    "recognition_total_time_ms",
    "sam3_service_initial_setup_ms",
    "sam3_service_model_prepare_ms",
    "sam3_client_wall_ms",
    "sam3_book_spine_ms",
    "sam3_book_end_ms",
    "sam3_communication_wait_ms",
    "sam3_other_ms",
    "mask_postprocess_ms",
    "held_book_detection_ms",
    "book_end_selection_ms",
    "roi_generation_ms",
    "residual_generation_ms",
    "roi_residual_ms",
    "held_book_occlusion_mask_ms",
    "obstacle_sequence_ms",
    "space_candidate_generation_ms",
    "candidate_selection_ms",
    "space_candidate_ms",
    "depth_ref_calculation_ms",
    "ransac_plane_calculation_ms",
    "geometry_calculation_ms",
    "depth_geometry_ms",
    "target_point_calculation_ms",
    "target_ms",
    "debug_image_generation_ms",
    "debug_image_ms",
    "png_jpg_save_ms",
    "binary_file_save_ms",
    "file_save_ms",
    "json_metadata_save_ms",
    "other_ms",
    "unaccounted_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile current storage recognition using saved RGB-D."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N captures after deterministic sorting.",
    )
    return parser.parse_args()


def discover_shots(input_root: Path, output_root: Path) -> list[Path]:
    excluded_names = {"improvement result", output_root.name}
    shots = {
        path.parent
        for name in ("input_rgb.npy", "input.png")
        for path in input_root.rglob(name)
        if not any(part in excluded_names for part in path.parts)
    }
    return sorted(shots, key=lambda path: str(path.relative_to(input_root)))


def identity(input_root: Path, shot_dir: Path) -> tuple[str, str]:
    relative = shot_dir.relative_to(input_root)
    cycle = relative.parts[0]
    view = next(
        (part for part in relative.parts if part in {"1st", "2nd", "3rd"}),
        "unknown",
    )
    return cycle, view


def read_saved_rgbd(shot_dir: Path) -> tuple[np.ndarray, np.ndarray, Any, float]:
    rgb = load_saved_capture_rgb(shot_dir)
    depth_path = shot_dir / "depth_u16_raw.npy"
    if not depth_path.is_file():
        depth_path = shot_dir / "depth_u16.npy"
    depth = np.load(depth_path, allow_pickle=False).astype(np.uint16)
    params = json.loads(
        (shot_dir / "camera_intrinsics.json").read_text(encoding="utf-8")
    )
    intr = intrinsics_from_camera_params(params)
    return rgb, depth, intr, float(params["depth_scale"])


def empty_row(index: int, input_root: Path, shot_dir: Path) -> dict[str, Any]:
    cycle, view = identity(input_root, shot_dir)
    row: dict[str, Any] = {
        "capture_index": index,
        "cycle": cycle,
        "view": view,
        "shot_dir": str(shot_dir.resolve()),
        "recognition_success": False,
        "selected_space_id": None,
        "error": "",
    }
    row.update({name: 0.0 for name in TIMING_FIELDS})
    return row


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(*IDENTITY_FIELDS, *TIMING_FIELDS),
        )
        writer.writeheader()
        writer.writerows(rows)


def percentile90(values: list[float]) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), 90))


def write_statistics(path: Path, rows: list[dict[str, Any]]) -> None:
    records: list[dict[str, Any]] = []
    for field in TIMING_FIELDS:
        values = [float(row[field]) for row in rows]
        records.append(
            {
                "metric": field,
                "count": len(values),
                "mean_ms": statistics.fmean(values) if values else None,
                "median_ms": statistics.median(values) if values else None,
                "p90_ms": percentile90(values) if values else None,
                "max_ms": max(values) if values else None,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("metric", "count", "mean_ms", "median_ms", "p90_ms", "max_ms"),
        )
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    shots = discover_shots(input_root, output_root)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        shots = shots[: args.limit]

    runtime = StorageSam3Runtime()
    setup_started = time.perf_counter()
    runtime.start()
    service_setup_ms = (time.perf_counter() - setup_started) * 1000.0
    rows: list[dict[str, Any]] = []
    log_path = output_root / "run.log"
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            log.write(f"input_root={input_root}\n")
            log.write(f"capture_count={len(shots)}\n")
            log.write(f"sam3_service_initial_setup_ms={service_setup_ms:.6f}\n")
            for index, shot_dir in enumerate(shots, start=1):
                print(f"[{index:02d}/{len(shots):02d}] {shot_dir}", flush=True)
                row = empty_row(index, input_root, shot_dir)
                row["sam3_service_initial_setup_ms"] = (
                    service_setup_ms if index == 1 else 0.0
                )
                timing: dict[str, float] = {}
                total_started = time.perf_counter()
                try:
                    rgbd_started = time.perf_counter()
                    rgb, depth, intr, depth_scale = read_saved_rgbd(shot_dir)
                    timing["rgbd_ms"] = (
                        time.perf_counter() - rgbd_started
                    ) * 1000.0
                    with tempfile.TemporaryDirectory(
                        prefix="capture-",
                        dir=output_root,
                    ) as work_name:
                        result = run_from_rgbd(
                            rgb,
                            depth,
                            intr,
                            depth_scale,
                            shot_dir=work_name,
                            runtime=runtime,
                            target_3d_mode="depth_ref",
                            timing_sink=timing,
                        )
                    row["selected_space_id"] = result[2].get(
                        "selected_space_id"
                    )
                    row["recognition_success"] = True
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    log.write(
                        f"\n[FAILED] {shot_dir}\n{traceback.format_exc()}\n"
                    )
                timing["total_ms"] = (
                    time.perf_counter() - total_started
                ) * 1000.0
                for name in TIMING_FIELDS:
                    if name == "sam3_service_initial_setup_ms":
                        continue
                    row[name] = float(timing.get(name, 0.0))
                rows.append(row)
                write_rows(output_root / "per_capture_timings.csv", rows)
                print(
                    f"  success={row['recognition_success']} "
                    f"recognition={row['recognition_total_time_ms']:.1f}ms "
                    f"spine={row['sam3_book_spine_ms']:.1f}ms "
                    f"end={row['sam3_book_end_ms']:.1f}ms",
                    flush=True,
                )
    finally:
        runtime.close()

    write_statistics(output_root / "timing_statistics.csv", rows)
    write_statistics(
        output_root / "timing_statistics_successful.csv",
        [row for row in rows if bool(row["recognition_success"])],
    )
    (output_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "input_root": str(input_root),
                "capture_count": len(rows),
                "success_count": sum(
                    bool(row["recognition_success"]) for row in rows
                ),
                "failure_count": sum(
                    not bool(row["recognition_success"]) for row in rows
                ),
                "sam3_service_initial_setup_ms": service_setup_ms,
                "rgbd_definition": "saved RGB/depth/intrinsics file loading",
                "recognition_total_definition": (
                    "run_from_rgbd wall time after saved RGB-D loading; includes "
                    "input/debug writes, SAM3, 2D recognition, and geometry"
                ),
                "cuda_timing": (
                    "one synchronization before each model timing and one at "
                    "the model-predict/NMS boundary"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
