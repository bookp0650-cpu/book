#!/usr/bin/env python3
"""Evaluate the current retrieval recognition pipeline on saved RGB-D cases.

This module is intentionally a thin evaluation wrapper.  Recognition, OCR,
point-cloud processing, PCA, and book-width estimation are delegated to the
existing retrieval implementation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyrealsense2 as rs
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_ROOT = PROJECT_ROOT / "captures" / "thesis" / "ret"
MASTER_JSON = PROJECT_ROOT / "master_json" / "master_20260216.json"
INTRINSICS_JSON = EVALUATION_ROOT / "camera_intrinsics.json"
GT_ROOT = EVALUATION_ROOT / "thesis_ret" / "SegmentationClass"
LABELMAP_PATH = EVALUATION_ROOT / "thesis_ret" / "labelmap.txt"
OUTPUT_ROOT = EVALUATION_ROOT / "retrieval_results"
TARGET_LABEL = "book spine"
CASE_MIN = 1
CASE_MAX = 100
TIMING_FIELDS = [
    "sam3_time_ms",
    "ocr_time_ms",
    "ocr_initialization_time_ms",
    "ocr_load_time_ms",
    "ocr_inference_time_ms",
    "ocr_total_time_ms",
    "ocr_predict_0deg_time_ms",
    "ocr_predict_plus30_time_ms",
    "ocr_predict_minus30_time_ms",
    "ocr_predict_3angle_total_time_ms",
    "ocr_other_time_ms",
    "matching_time_ms",
    "ocr_sam_assignment_time_ms",
    "similarity_time_ms",
    "mask_refinement_time_ms",
    "depth_pointcloud_time_ms",
    "ransac_time_ms",
    "pca_geometry_time_ms",
    "grasp_point_time_ms",
    "recognition_total_time_ms",
]

CSV_FIELDS = [
    "case_id",
    "book_index",
    "target_title",
    "gt_width_mm",
    "pred_width_mm",
    "width_abs_error_mm",
    "width_valid",
    "iou",
    "iou_threshold",
    "recognition_success",
    *TIMING_FIELDS,
    "error_reason",
    "matched_text",
    "similarity_score",
    "book_angle",
    "book_angle_unit",
    "grasp_x",
    "grasp_y",
    "grasp_z",
    "grasp_unit",
    "case_artifacts_dir",
    "artifact_error_reason",
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_labelmap(path: Path) -> dict[str, tuple[int, int, int]]:
    """Parse CVAT SegmentationClass label colors from labelmap.txt."""
    labels: dict[str, tuple[int, int, int]] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(":", 3)
        if len(fields) < 2:
            raise ValueError(f"invalid labelmap line {line_number}: {raw_line!r}")
        name = fields[0].strip()
        try:
            color = tuple(int(value.strip()) for value in fields[1].split(","))
        except ValueError as exc:
            raise ValueError(
                f"invalid RGB color on labelmap line {line_number}: {raw_line!r}"
            ) from exc
        if len(color) != 3 or any(value < 0 or value > 255 for value in color):
            raise ValueError(
                f"invalid RGB color on labelmap line {line_number}: {color!r}"
            )
        labels[name] = color
    return labels


def _load_gt_mask(path: Path, target_rgb: tuple[int, int, int]) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    color = np.asarray(target_rgb, dtype=np.uint8)
    return np.all(rgb == color, axis=2)


def _load_predicted_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def _mask_iou(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)
    if predicted.shape != ground_truth.shape:
        raise ValueError(
            f"predicted/GT mask shape mismatch: {predicted.shape} / {ground_truth.shape}"
        )
    intersection = int(np.count_nonzero(predicted & ground_truth))
    union = int(np.count_nonzero(predicted | ground_truth))
    if union == 0:
        return 1.0
    return float(intersection / union)


def _distortion_from_name(value: Any):
    name = str(value or "none").strip().lower().split(".")[-1]
    supported = {
        "none": rs.distortion.none,
        "modified_brown_conrady": rs.distortion.modified_brown_conrady,
        "inverse_brown_conrady": rs.distortion.inverse_brown_conrady,
        "ftheta": rs.distortion.ftheta,
        "brown_conrady": rs.distortion.brown_conrady,
        "kannala_brandt4": rs.distortion.kannala_brandt4,
    }
    if name not in supported:
        raise ValueError(f"unsupported RealSense distortion model: {value!r}")
    return supported[name]


def _load_intrinsics(path: Path) -> tuple[Any, float, dict[str, Any]]:
    payload = _read_json(path)
    intr = rs.intrinsics()
    intr.width = int(payload["width"])
    intr.height = int(payload["height"])
    intr.fx = float(payload["fx"])
    intr.fy = float(payload["fy"])
    intr.ppx = float(payload["ppx"])
    intr.ppy = float(payload["ppy"])
    intr.model = _distortion_from_name(payload.get("model"))
    coeffs = [float(value) for value in payload.get("coeffs", [0.0] * 5)]
    if len(coeffs) != 5:
        raise ValueError(f"camera intrinsics coeffs must contain 5 values: {coeffs!r}")
    intr.coeffs = coeffs
    depth_scale = float(payload["depth_scale"])
    if intr.width <= 0 or intr.height <= 0 or intr.fx <= 0 or intr.fy <= 0:
        raise ValueError("camera intrinsics dimensions and focal lengths must be positive")
    if not math.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError(f"invalid depth_scale: {depth_scale!r}")
    return intr, depth_scale, payload


def _new_run_log_dir(logs_root: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    candidate = logs_root / f"run_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = logs_root / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _validate_case_inputs(
    rgb_path: Path,
    depth_path: Path,
    gt_path: Path,
    intrinsics_payload: dict[str, Any],
) -> None:
    missing = [path for path in (rgb_path, depth_path, gt_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    with Image.open(rgb_path) as image:
        rgb_size = image.size
    depth = np.load(depth_path, mmap_mode="r", allow_pickle=False)
    expected_size = (
        int(intrinsics_payload["width"]),
        int(intrinsics_payload["height"]),
    )
    if rgb_size != expected_size:
        raise ValueError(f"RGB/intrinsics size mismatch: {rgb_size} / {expected_size}")
    if depth.shape != (expected_size[1], expected_size[0]):
        raise ValueError(
            f"Depth/intrinsics shape mismatch: {depth.shape} / "
            f"{(expected_size[1], expected_size[0])}"
        )


def _load_match_details(case_dir: Path, core: dict[str, Any]) -> tuple[Any, Any]:
    matched_text = core.get("selected_ocr_text")
    similarity_score = core.get("ocr_matching_score")
    similarity_path = case_dir / "similarity_scores.json"
    if similarity_path.is_file():
        payload = _read_json(similarity_path)
        scores = payload.get("scores") or []
        if scores:
            matched_text = scores[0].get("text", matched_text)
            similarity_score = scores[0].get("score", similarity_score)
    return matched_text, similarity_score


def _append_error(errors: list[str], exc: BaseException) -> None:
    errors.append(f"{type(exc).__name__}: {exc}")


def _extract_timing_ms(
    core: dict[str, Any], partial_timing_ms: dict[str, Any] | None = None
) -> dict[str, float | None]:
    source = dict(partial_timing_ms or {})
    core_timing = core.get("timing_ms")
    if isinstance(core_timing, dict):
        source.update(core_timing)
    result: dict[str, float | None] = {}
    for field in TIMING_FIELDS:
        value = source.get(field)
        if value is None:
            result[field] = None
            continue
        numeric = float(value)
        result[field] = numeric if math.isfinite(numeric) and numeric >= 0 else None
    return result


def _extract_saved_detail_timing_ms(case_dir: Path) -> dict[str, float | None]:
    result = {
        "ocr_predict_0deg_time_ms": None,
        "ocr_predict_plus30_time_ms": None,
        "ocr_predict_minus30_time_ms": None,
        "ocr_predict_3angle_total_time_ms": None,
        "ocr_other_time_ms": None,
        "ocr_sam_assignment_time_ms": None,
        "similarity_time_ms": None,
    }
    runtime_path = case_dir / "ocr_runtime_info.json"
    if runtime_path.is_file():
        runtime = _read_json(runtime_path)
        predict = runtime.get("predict_seconds") or {}
        angle_fields = (
            ("0.0", "ocr_predict_0deg_time_ms"),
            ("30.0", "ocr_predict_plus30_time_ms"),
            ("-30.0", "ocr_predict_minus30_time_ms"),
        )
        values = []
        for key, field in angle_fields:
            value = predict.get(key)
            if value is not None:
                result[field] = float(value) * 1000.0
                values.append(float(value))
        if len(values) == len(angle_fields):
            predict_total = sum(values)
            result["ocr_predict_3angle_total_time_ms"] = predict_total * 1000.0
            worker_total = (
                runtime.get("detailed_timing_seconds") or {}
            ).get("worker_total_before_runtime_json")
            if worker_total is not None:
                result["ocr_other_time_ms"] = max(
                    0.0, (float(worker_total) - predict_total) * 1000.0
                )

    matching_path = case_dir / "text_mask_iou_debug.json"
    if matching_path.is_file():
        matching = (_read_json(matching_path).get("timing_seconds") or {})
        assignment = matching.get("ocr_to_mask_assignment")
        similarity = matching.get("similarity_and_scores_json")
        if assignment is not None:
            result["ocr_sam_assignment_time_ms"] = float(assignment) * 1000.0
        if similarity is not None:
            result["similarity_time_ms"] = float(similarity) * 1000.0
    return result


def _evaluate_case(
    case_number: int,
    master: list[dict[str, Any]],
    intr: Any,
    depth_scale: float,
    intrinsics_payload: dict[str, Any],
    target_rgb: tuple[int, int, int],
    iou_threshold: float,
    predicted_masks_dir: Path,
    run_log_dir: Path,
    sam_device: str,
) -> dict[str, Any]:
    case_id = f"{case_number:04d}"
    book_index = (case_number - 1) // 5
    book = master[book_index]
    target_title = str(book["book_name"])
    gt_width_mm = float(book["book_width"])

    source_case_dir = EVALUATION_ROOT / str(case_number)
    rgb_path = source_case_dir / f"{case_id}_rgb.png"
    depth_path = source_case_dir / f"{case_id}_depth_filtered.npy"
    gt_path = GT_ROOT / f"{case_id}_rgb.png"
    case_dir = run_log_dir / case_id
    case_dir.mkdir(parents=False, exist_ok=False)
    console_log = case_dir / "console.log"
    predicted_output = predicted_masks_dir / f"{case_id}_pred_mask.png"
    predicted_output.unlink(missing_ok=True)

    errors: list[str] = []
    core: dict[str, Any] = {}
    partial_timing_ms: dict[str, Any] = {}
    pred_width_mm: float | None = None
    iou = 0.0

    try:
        _validate_case_inputs(rgb_path, depth_path, gt_path, intrinsics_payload)
        shutil.copy2(rgb_path, case_dir / "after_init_rgb.png")
        shutil.copy2(depth_path, case_dir / "after_init_depth.npy")
        shutil.copy2(INTRINSICS_JSON, case_dir / "source_camera_intrinsics.json")

        from detection.pro_handbook.sam_py_demo.get_book_points_no_mask_merge_no_side_filter import (
            run_capture_and_pca_offline_no_mask_merge_no_side_filter,
        )

        with console_log.open("w", encoding="utf-8") as log_handle:
            try:
                with redirect_stdout(log_handle), redirect_stderr(log_handle):
                    core = run_capture_and_pca_offline_no_mask_merge_no_side_filter(
                        query=target_title,
                        shot_dir=case_dir,
                        sam_device=sam_device,
                        intr=intr,
                        depth_scale=depth_scale,
                        timing_sink=partial_timing_ms,
                    )
            except Exception as exc:
                recognition_start = partial_timing_ms.get(
                    "_recognition_total_start_perf_counter"
                )
                if recognition_start is not None:
                    partial_timing_ms["recognition_total_time_ms"] = (
                        time.perf_counter() - float(recognition_start)
                    ) * 1000.0
                traceback.print_exc(file=log_handle)
                _append_error(errors, exc)
    except Exception as exc:
        _append_error(errors, exc)
        console_log.write_text(traceback.format_exc(), encoding="utf-8")

    if core:
        try:
            value = float(core["pred_book_width_mm"])
            if math.isfinite(value) and value > 0:
                pred_width_mm = value
            else:
                raise ValueError(f"invalid predicted width: {value!r}")
        except Exception as exc:
            _append_error(errors, exc)

    selected_mask_path = case_dir / "selected_mask_used_for_depth.png"
    if selected_mask_path.is_file():
        try:
            predicted_mask = _load_predicted_mask(selected_mask_path)
            gt_mask = _load_gt_mask(gt_path, target_rgb)
            iou = _mask_iou(predicted_mask, gt_mask)
            Image.fromarray(predicted_mask.astype(np.uint8) * 255, mode="L").save(
                predicted_output
            )
        except Exception as exc:
            iou = 0.0
            _append_error(errors, exc)

    matched_text = None
    similarity_score = None
    try:
        matched_text, similarity_score = _load_match_details(case_dir, core)
    except Exception as exc:
        _append_error(errors, exc)

    book_angle = core.get("roll_rad")
    point = core.get("point_3d")
    timing_ms = _extract_timing_ms(core, partial_timing_ms)
    timing_ms.update(_extract_saved_detail_timing_ms(case_dir))
    grasp = [None, None, None]
    if isinstance(point, (list, tuple, np.ndarray)) and len(point) == 3:
        grasp = [float(value) for value in point]

    width_valid = pred_width_mm is not None
    width_abs_error_mm = (
        abs(pred_width_mm - gt_width_mm) if pred_width_mm is not None else None
    )
    recognition_success = bool(iou >= iou_threshold and predicted_output.is_file())
    return {
        "case_id": case_id,
        "book_index": book_index,
        "target_title": target_title,
        "gt_width_mm": gt_width_mm,
        "pred_width_mm": pred_width_mm,
        "width_abs_error_mm": width_abs_error_mm,
        "width_valid": width_valid,
        "iou": iou,
        "iou_threshold": iou_threshold,
        "recognition_success": recognition_success,
        **timing_ms,
        "error_reason": " | ".join(errors),
        "matched_text": matched_text,
        "similarity_score": similarity_score,
        "book_angle": float(book_angle) if book_angle is not None else None,
        "book_angle_unit": "rad" if book_angle is not None else None,
        "grasp_x": grasp[0],
        "grasp_y": grasp[1],
        "grasp_z": grasp[2],
        "grasp_unit": "m" if point is not None else None,
        "case_artifacts_dir": str(case_dir.resolve()),
    }


def _write_results_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_summary(
    rows: list[dict[str, Any]], iou_threshold: float, run_log_dir: Path
) -> dict[str, Any]:
    total_cases = len(rows)
    success_count = sum(bool(row["recognition_success"]) for row in rows)
    valid_width_rows = [row for row in rows if row["width_valid"]]
    absolute_errors = [float(row["width_abs_error_mm"]) for row in valid_width_rows]
    def timing_stats(values: list[float]) -> dict[str, Any]:
        return {
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "p90": float(np.percentile(values, 90)) if values else None,
            "sd": statistics.pstdev(values) if values else None,
            "valid_count": len(values),
        }

    timing_names = {
        "sam3": "sam3_time_ms",
        "ocr": "ocr_time_ms",
        "ocr_initialization": "ocr_initialization_time_ms",
        "ocr_load": "ocr_load_time_ms",
        "ocr_inference": "ocr_inference_time_ms",
        "ocr_total": "ocr_total_time_ms",
        "ocr_predict_0deg": "ocr_predict_0deg_time_ms",
        "ocr_predict_plus30": "ocr_predict_plus30_time_ms",
        "ocr_predict_minus30": "ocr_predict_minus30_time_ms",
        "ocr_predict_3angle_total": "ocr_predict_3angle_total_time_ms",
        "ocr_other": "ocr_other_time_ms",
        "matching": "matching_time_ms",
        "ocr_sam_assignment": "ocr_sam_assignment_time_ms",
        "similarity": "similarity_time_ms",
        "mask_refinement": "mask_refinement_time_ms",
        "depth_pointcloud": "depth_pointcloud_time_ms",
        "ransac": "ransac_time_ms",
        "pca_geometry": "pca_geometry_time_ms",
        "grasp_point": "grasp_point_time_ms",
        "recognition_total": "recognition_total_time_ms",
    }
    timing_summary = {}
    for name, field in timing_names.items():
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        timing_summary[name] = timing_stats(values)

    matching_values = [
        float(row["matching_time_ms"])
        for row in rows
        if row.get("matching_time_ms") is not None
    ]
    timing_summary["target_identification"] = timing_stats(matching_values)
    geometry_fields = (
        "depth_pointcloud_time_ms",
        "ransac_time_ms",
        "pca_geometry_time_ms",
        "grasp_point_time_ms",
    )
    geometry_values = [
        sum(float(row[field]) for field in geometry_fields)
        for row in rows
        if all(row.get(field) is not None for field in geometry_fields)
    ]
    timing_summary["geometry"] = timing_stats(geometry_values)

    return {
        "total_cases": total_cases,
        "recognition_success_count": success_count,
        "recognition_success_rate_percent": (
            100.0 * success_count / total_cases if total_cases else 0.0
        ),
        "mean_iou": (
            sum(float(row["iou"]) for row in rows) / total_cases
            if total_cases
            else 0.0
        ),
        "valid_width_count": len(valid_width_rows),
        "valid_width_rate_percent": (
            100.0 * len(valid_width_rows) / total_cases if total_cases else 0.0
        ),
        "width_mae_mm": (
            statistics.fmean(absolute_errors) if absolute_errors else None
        ),
        "width_error_sd_mm": (
            statistics.pstdev(absolute_errors) if absolute_errors else None
        ),
        "width_error_sd_definition": (
            "population standard deviation of per-case absolute width errors"
        ),
        "timing_unit": "ms",
        "timing_ms": timing_summary,
        "iou_threshold": iou_threshold,
        "evaluated_case_ids": [row["case_id"] for row in rows],
        "width_excluded_case_ids": [
            row["case_id"] for row in rows if not row["width_valid"]
        ],
        "run_artifacts_dir": str(run_log_dir.resolve()),
        "created_at": datetime.now().astimezone().isoformat(),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate current retrieval recognition on saved thesis cases."
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        help="case numbers to evaluate (default: 1 through 100)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        required=True,
        help="IoU threshold used only for recognition success judgement",
    )
    parser.add_argument(
        "--sam-device",
        default="gpu",
        help="existing recognition function's SAM device argument (default: gpu)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=EVALUATION_ROOT,
        help="saved thesis retrieval dataset root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="result directory (default: DATASET_ROOT/retrieval_results)",
    )
    parser.add_argument(
        "--ocr-device",
        choices=("gpu", "cpu"),
        default="gpu",
        help="PaddleOCR device; production default is strict GPU",
    )
    parser.add_argument(
        "--ocr-overlap",
        action="store_true",
        help="start OCR before SAM3 and measure shared-GPU overlap",
    )
    parser.add_argument(
        "--ocr-resident",
        action="store_true",
        help="keep one OCR worker/model resident across all evaluated cases",
    )
    parser.add_argument(
        "--gpu-monitor",
        action="store_true",
        help="record GPU and per-process VRAM samples for this run",
    )
    parser.add_argument(
        "--gpu-monitor-interval",
        type=float,
        default=0.2,
        help="GPU monitor sampling interval in seconds (default: 0.2)",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.iou_threshold <= 1.0:
        parser.error("--iou-threshold must be between 0 and 1")
    indices = args.indices or list(range(CASE_MIN, CASE_MAX + 1))
    if any(index < CASE_MIN or index > CASE_MAX for index in indices):
        parser.error(f"--indices values must be between {CASE_MIN} and {CASE_MAX}")
    if len(indices) != len(set(indices)):
        parser.error("--indices must not contain duplicates")
    if args.gpu_monitor_interval < 0.05:
        parser.error("--gpu-monitor-interval must be at least 0.05")
    args.indices = indices
    return args


def main(argv: list[str] | None = None) -> int:
    global EVALUATION_ROOT, INTRINSICS_JSON, GT_ROOT, LABELMAP_PATH, OUTPUT_ROOT
    args = _parse_args(argv)
    EVALUATION_ROOT = args.dataset_root.expanduser().resolve()
    INTRINSICS_JSON = EVALUATION_ROOT / "camera_intrinsics.json"
    GT_ROOT = EVALUATION_ROOT / "thesis_ret" / "SegmentationClass"
    LABELMAP_PATH = EVALUATION_ROOT / "thesis_ret" / "labelmap.txt"
    OUTPUT_ROOT = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else EVALUATION_ROOT / "retrieval_results"
    )
    master = _read_json(MASTER_JSON)
    if not isinstance(master, list) or len(master) < 20:
        raise ValueError(f"master JSON must contain at least 20 books: {MASTER_JSON}")
    labelmap = _parse_labelmap(LABELMAP_PATH)
    if TARGET_LABEL not in labelmap:
        raise KeyError(f"target label {TARGET_LABEL!r} is absent from {LABELMAP_PATH}")
    intr, depth_scale, intrinsics_payload = _load_intrinsics(INTRINSICS_JSON)

    predicted_masks_dir = OUTPUT_ROOT / "predicted_masks"
    logs_root = OUTPUT_ROOT / "logs"
    predicted_masks_dir.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    run_log_dir = _new_run_log_dir(logs_root)
    results_path = OUTPUT_ROOT / "results.csv"
    summary_path = OUTPUT_ROOT / "summary.json"
    rows: list[dict[str, Any]] = []

    from detection.pro_handbook.thesis.case_artifacts import (
        build_case_artifact,
        finalize_artifact_run,
        prepare_artifact_run,
    )

    artifact_run_root = prepare_artifact_run(
        output_root=OUTPUT_ROOT / "case_artifacts",
        run_id=run_log_dir.name,
        source_run_dir=run_log_dir,
        results_csv=results_path,
    )
    artifact_case_ids: list[str] = []

    execution_order = "overlap" if args.ocr_overlap else "sequential"
    mode_name = (
        ("ocr_resident" if args.ocr_resident else "ocr_reload")
        + f"_{execution_order}_{args.ocr_device}"
    )
    gpu_monitor_proc: subprocess.Popen | None = None
    gpu_run_dir: Path | None = None
    if args.gpu_monitor:
        gpu_run_dir = OUTPUT_ROOT / "gpu_memory_logs" / f"{run_log_dir.name}_{mode_name}"
        gpu_run_dir.mkdir(parents=True, exist_ok=False)
        phase_log = gpu_run_dir / "phases.jsonl"
        phase_log.touch()
        os.environ["BOOK_GPU_PHASE_LOG"] = str(phase_log)
        monitor_script = Path(__file__).with_name("gpu_memory_monitor.py")
        gpu_monitor_proc = subprocess.Popen(
            [
                sys.executable,
                str(monitor_script),
                "--output",
                str(gpu_run_dir / "samples.csv"),
                "--interval",
                str(args.gpu_monitor_interval),
            ]
        )
        time.sleep(max(0.4, args.gpu_monitor_interval * 2.0))
    else:
        os.environ.pop("BOOK_GPU_PHASE_LOG", None)

    from detection.pro_handbook.thesis.gpu_memory_monitor import record_phase

    record_phase("recognition_program_before_models", mode=mode_name)

    os.environ["BOOK_OCR_GPU_SAFE"] = "0" if args.ocr_overlap else "1"
    os.environ["BOOK_OCR_DEVICE"] = args.ocr_device
    os.environ["BOOK_OCR_RELOAD_EACH_SEQUENCE"] = (
        "0" if args.ocr_resident else "1"
    )
    from detection.pro_handbook.sam3_runtime.integration_service_manager import (
        Sam3ServiceSession,
    )
    from detection.pro_handbook.sam_py_demo.ocr_worker_client import (
        initialize_ocr_worker,
        shutdown_ocr_worker,
    )

    ocr_initialization_time_ms: float | None = None
    ocr_runtime: dict[str, Any] | None = None
    try:
        with Sam3ServiceSession() as sam3_session:
            record_phase(
                "sam3_service_ready",
                sam3_pid=sam3_session.owned_pid,
                started_by_this_process=sam3_session.started_by_this_process,
            )
            try:
                if args.ocr_resident:
                    record_phase("ocr_initialization_start")
                    initialization = initialize_ocr_worker(timeout=120.0)
                    ocr_initialization_time_ms = (
                        float(initialization["initialization_seconds"]) * 1000.0
                    )
                    ocr_runtime = initialization.get("runtime")
                    print(
                        "OCR resident initialization: "
                        f"{ocr_initialization_time_ms:.3f} ms",
                        flush=True,
                    )

                for position, case_number in enumerate(args.indices, start=1):
                    book_index = (case_number - 1) // 5
                    target_title = str(master[book_index]["book_name"])
                    print(
                        f"[{position}/{len(args.indices)}] case {case_number:04d}",
                        flush=True,
                    )
                    print(f"  target: {target_title}", flush=True)
                    record_phase("case_start", case_id=f"{case_number:04d}")
                    row = _evaluate_case(
                        case_number=case_number,
                        master=master,
                        intr=intr,
                        depth_scale=depth_scale,
                        intrinsics_payload=intrinsics_payload,
                        target_rgb=labelmap[TARGET_LABEL],
                        iou_threshold=args.iou_threshold,
                        predicted_masks_dir=predicted_masks_dir,
                        run_log_dir=run_log_dir,
                        sam_device=args.sam_device,
                    )
                    row["ocr_initialization_time_ms"] = (
                        ocr_initialization_time_ms if position == 1 else None
                    )
                    raw_case_log_dir = Path(row["case_artifacts_dir"])
                    row["artifact_error_reason"] = ""
                    try:
                        artifact_dir = build_case_artifact(
                            row=row,
                            source_case_dir=raw_case_log_dir,
                            capture_case_dir=EVALUATION_ROOT / str(case_number),
                            predicted_masks_dir=predicted_masks_dir,
                            gt_root=GT_ROOT,
                            artifact_run_root=artifact_run_root,
                            target_rgb=labelmap[TARGET_LABEL],
                        )
                        row["case_artifacts_dir"] = str(artifact_dir.resolve())
                        artifact_case_ids.append(f"{case_number:04d}")
                    except Exception as artifact_exc:
                        row["case_artifacts_dir"] = None
                        row["artifact_error_reason"] = (
                            f"{type(artifact_exc).__name__}: {artifact_exc}"
                        )
                        print(
                            "  artifact generation error: "
                            f"{row['artifact_error_reason']}",
                            file=sys.stderr,
                            flush=True,
                        )
                    rows.append(row)
                    record_phase(
                        "case_end",
                        case_id=f"{case_number:04d}",
                        error_reason=row["error_reason"],
                    )
                    _write_results_csv(results_path, rows)
                    recognition = "success" if row["recognition_success"] else "failure"
                    print(f"  recognition: {recognition}")
                    print(f"  IoU: {row['iou']:.3f}")
                    if row["width_valid"]:
                        print(
                            "  width: "
                            f"pred={row['pred_width_mm']:.3f} mm, "
                            f"gt={row['gt_width_mm']:.3f} mm, "
                            f"error={row['width_abs_error_mm']:.3f} mm"
                        )
                    else:
                        print(f"  width: invalid (gt={row['gt_width_mm']:.3f} mm)")
                    if row["recognition_total_time_ms"] is not None:
                        print(
                            "  recognition time: "
                            f"{row['recognition_total_time_ms']:.3f} ms"
                        )
                    if row["ocr_total_time_ms"] is not None:
                        load_text = (
                            f"{row['ocr_load_time_ms']:.3f}"
                            if row["ocr_load_time_ms"] is not None
                            else "N/A"
                        )
                        inference_text = (
                            f"{row['ocr_inference_time_ms']:.3f}"
                            if row["ocr_inference_time_ms"] is not None
                            else "N/A"
                        )
                        print(
                            "  OCR time: "
                            f"load={load_text} ms, "
                            f"inference={inference_text} ms, "
                            f"total={row['ocr_total_time_ms']:.3f} ms"
                        )
                    if row["error_reason"]:
                        print(f"  error: {row['error_reason']}")
                        lowered_error = row["error_reason"].lower()
                        gpu_failure_markers = (
                            "out of memory",
                            "resourceexhausted",
                            "cuda error",
                            "cudnn",
                            "cublas",
                        )
                        if any(
                            marker in lowered_error for marker in gpu_failure_markers
                        ):
                            print(
                                "  stopping: GPU-related failure detected",
                                file=sys.stderr,
                                flush=True,
                            )
                            break
            finally:
                shutdown_ocr_worker()
    finally:
        record_phase("recognition_program_models_stopped", mode=mode_name)
        if gpu_monitor_proc is not None:
            gpu_monitor_proc.terminate()
            try:
                gpu_monitor_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                gpu_monitor_proc.kill()
                gpu_monitor_proc.wait(timeout=5.0)

    summary = _make_summary(rows, args.iou_threshold, run_log_dir)
    summary["execution_mode"] = mode_name
    summary["execution_order"] = execution_order
    summary["ocr_device_requested"] = args.ocr_device
    summary["ocr_runtime"] = ocr_runtime
    summary["gpu_memory_log_dir"] = (
        str(gpu_run_dir.resolve()) if gpu_run_dir is not None else None
    )
    summary["case_artifacts_run_dir"] = str(artifact_run_root.resolve())
    summary["case_artifacts_valid_count"] = len(artifact_case_ids)
    _write_results_csv(results_path, rows)
    _write_json(summary_path, summary)
    finalize_artifact_run(
        output_root=OUTPUT_ROOT / "case_artifacts",
        run_root=artifact_run_root,
        case_ids=artifact_case_ids,
        results_csv=results_path,
    )
    print(f"results: {results_path}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
