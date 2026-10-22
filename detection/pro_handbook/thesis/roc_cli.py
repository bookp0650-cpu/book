#!/usr/bin/env python3
"""Command-line orchestration shared by retrieval and storage ROC wrappers."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detection.pro_handbook.thesis import roc_experiment as roc


def _copy_exclusive(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    shutil.copy2(source, destination)


def _load_points(path: Path | None, rgb: np.ndarray) -> list[list[int]]:
    if path is None:
        return roc.annotate_polygon_8(rgb)
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = payload.get("points_xy_source_pixels", payload) if isinstance(payload, dict) else payload
    return [[int(x), int(y)] for x, y in points]


def _severity(path: Path | None) -> dict[str, Any] | None:
    return None if path is None else json.loads(path.read_text(encoding="utf-8"))


def _save_failed_candidate_debug(
    directory: Path,
    *,
    rgb: np.ndarray,
    gt: np.ndarray,
    points: list[list[int]],
    depth_path: Path,
    error: roc.CandidateGenerationError,
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    closest = error.closest_mask
    paths = {
        "rgb_path": directory / "input_rgb.png",
        "gt_mask_path": directory / "gt_mask.png",
        "gt_overlay_path": directory / "gt_overlay.png",
        "debug_closest_mask_path": directory / "debug_closest_candidate_mask.png",
        "debug_closest_overlay_path": directory / "debug_closest_candidate_overlay.png",
        "debug_closest_comparison_path": (
            directory / "debug_closest_candidate_comparison.png"
        ),
        "depth_path": directory / "input_depth.npy",
    }
    cv2.imwrite(str(paths["rgb_path"]), rgb)
    cv2.imwrite(str(paths["gt_mask_path"]), gt.astype(np.uint8) * 255)
    cv2.imwrite(str(paths["gt_overlay_path"]), roc.overlay(rgb, gt, (0, 255, 0)))
    cv2.imwrite(
        str(paths["debug_closest_mask_path"]),
        closest.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(paths["debug_closest_overlay_path"]),
        roc.overlay(rgb, closest, (255, 128, 0)),
    )
    cv2.imwrite(
        str(paths["debug_closest_comparison_path"]),
        roc.comparison_overlay(rgb, gt, closest),
    )
    _copy_exclusive(depth_path, paths["depth_path"])
    roc.write_json_atomic(directory / "gt_polygon_points.json", {
        "points_xy_source_pixels": points,
        "source_width": int(rgb.shape[1]),
        "source_height": int(rgb.shape[0]),
        "point_count": len(points),
        "coordinate_origin": "top_left",
        "polygon_order": "click_order",
    })
    roc.write_json_atomic(directory / "debug_candidate_failure.json", {
        **error.diagnostics,
        "debug_paths": {key: str(path) for key, path in paths.items()},
    })
    print(f"[ROC DEBUG] closest rejected candidate saved to: {directory}")


def _run_downstream(
    task: str, directory: Path, rgb_path: Path, depth_path: Path,
    artificial: np.ndarray, args: argparse.Namespace,
) -> tuple[dict[str, Any], Path]:
    if task == "retrieval":
        if not args.query:
            raise ValueError("retrieval --run-downstream requires --query")
        required_rgb = directory / "after_init_rgb.png"
        required_depth = directory / "after_init_depth.npy"
        if rgb_path.resolve() != required_rgb.resolve():
            _copy_exclusive(rgb_path, required_rgb)
        if depth_path.resolve() != required_depth.resolve():
            _copy_exclusive(depth_path, required_depth)
        from detection.pro_handbook.sam_py_demo.get_book_points_sam3_refined_sam2_width import (
            run_capture_and_pca_offline_sam3_refined_sam2_width,
        )
        result = run_capture_and_pca_offline_sam3_refined_sam2_width(
            args.query, directory, sam_device=args.sam_device,
            selected_mask_override=artificial,
        )
        return result, directory / "offline_recognition_result.json"

    if args.camera_params is None:
        raise ValueError("storage --run-downstream requires --camera-params")
    from detection.pro_handbook.sam_py_demo.Storage_SAM3 import run_saved_rgbd
    angle, target, detail = run_saved_rgbd(
        rgb_path, depth_path, args.camera_params, shot_dir=directory,
        selected_space_mask_override=artificial,
    )
    result = {"angle_rad": float(angle), "point_3d": np.asarray(target).tolist(), "detail": detail}
    result_path = directory / "downstream_result.json"
    roc.write_json_atomic(result_path, result)
    return result, result_path


def run_trial(task: str, args: argparse.Namespace) -> int:
    row = roc.resolve_trial(task, args.trial_id, resume=args.resume)
    captures_root = roc.ROC_ROOT / ("ret" if task == "retrieval" else "stor") / "captures"
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    directory = captures_root / "dry-run" / f"trial_{row['trial_id']}_{stamp}"
    if directory.exists():
        raise FileExistsError(f"refusing to reuse existing trial directory: {directory}")
    rgb_path, depth_path = args.rgb.resolve(), args.depth.resolve()
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    depth = np.load(depth_path, mmap_mode="r", allow_pickle=False)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch: {rgb.shape[:2]} / {depth.shape}")
    points = _load_points(args.gt_points, rgb)
    gt = roc.polygon_mask(rgb.shape[:2], points)
    try:
        result = roc.generate_for_bin(
            gt, row["error_type"], int(row["target_iou_bin"]),
            int(row["random_seed"]), severity_ranges=_severity(args.severity_config),
            max_attempts=args.max_attempts,
        )
    except roc.CandidateGenerationError as exc:
        _save_failed_candidate_debug(
            directory, rgb=rgb, gt=gt, points=points, depth_path=depth_path,
            error=exc,
        )
        raise
    paths = roc.save_mask_artifacts(directory, rgb, gt, result.mask, points, result)
    saved_depth = directory / "input_depth.npy"
    _copy_exclusive(depth_path, saved_depth)
    paths["depth_path"] = str(saved_depth)
    original_prediction = ""
    if args.original_prediction is not None:
        destination = directory / "original_prediction_mask.png"
        _copy_exclusive(args.original_prediction.resolve(), destination)
        original_prediction = str(destination)

    downstream: dict[str, Any] = {}
    downstream_path = ""
    if args.run_downstream:
        downstream, result_path = _run_downstream(
            task, directory, rgb_path, depth_path, result.mask, args,
        )
        downstream_path = str(result_path)
        produced_original = directory / "original_prediction_mask.png"
        if produced_original.is_file():
            original_prediction = str(produced_original)

    calculated_grasp = downstream.get("point_3d") if task == "retrieval" else None
    calculated_storage = downstream.get("point_3d") if task == "storage" else None
    metadata = {
        **row, "actual_iou": result.iou, "error_parameters": result.parameters,
        "gt_polygon_points": points, **paths,
        "original_prediction_mask_path": original_prediction,
        "calculated_grasp_position": calculated_grasp,
        "calculated_storage_position": calculated_storage,
        "success": "", "timestamp": roc.now_iso(),
        "dry_run": True, "robot_motion_executed": False,
        "candidate_attempts": result.attempts,
        "downstream_result_path": downstream_path,
    }
    metadata_path = directory / "trial_metadata.json"
    metadata["trial_metadata_path"] = str(metadata_path)
    roc.write_json_atomic(metadata_path, metadata)
    print(json.dumps({
        "trial_id": row["trial_id"], "task": task, "book_id": row["book_id"],
        "storage_condition": row["storage_condition"],
        "target_iou_bin": row["target_iou_bin"], "actual_iou": result.iou,
        "error_type": row["error_type"], "error_parameters": result.parameters,
        "calculated_grasp_position": calculated_grasp,
        "calculated_storage_position": calculated_storage,
        "robot_motion_executed": False, "trial_directory": str(directory),
    }, ensure_ascii=False, indent=2))
    return 0


def build_parser(task: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{task} artificial-mask ROC experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="create the reproducible experiment plan")
    plan.add_argument("--seed", type=int, default=20261021)
    plan.add_argument("--force", action="store_true", help="replace only the plan files")
    trial = commands.add_parser("dry-run", help="annotate and generate from saved RGB-D; no hardware")
    select = trial.add_mutually_exclusive_group(required=True)
    select.add_argument("--trial-id", type=int)
    select.add_argument("--resume", action="store_true")
    trial.add_argument("--rgb", type=Path, required=True)
    trial.add_argument("--depth", type=Path, required=True)
    trial.add_argument("--gt-points", type=Path, help="headless/repeatable 8-point JSON")
    trial.add_argument("--original-prediction", type=Path)
    trial.add_argument("--severity-config", type=Path)
    trial.add_argument("--max-attempts", type=int, default=5000)
    trial.add_argument("--run-downstream", action="store_true",
                       help="run current saved-RGB-D recognition and downstream geometry (may start SAM3/OCR)")
    trial.add_argument("--sam-device", default="gpu")
    trial.add_argument("--query", help="retrieval target title")
    trial.add_argument("--camera-params", type=Path, help="storage camera parameter JSON")
    commands.add_parser("analyze", help="build ROC outputs from trials.csv")
    commands.add_parser("layout", help="create required empty directories")
    if task == "retrieval":
        preview = commands.add_parser(
            "preview-errors",
            help="generate visual artificial-mask examples only; no hardware or models",
        )
        preview.add_argument("--rgb", type=Path, required=True)
        preview.add_argument("--depth", type=Path, required=True)
        preview.add_argument("--output-dir", type=Path, required=True)
        preview.add_argument("--config", type=Path, required=True)
        gt_source = preview.add_mutually_exclusive_group()
        gt_source.add_argument("--gt-mask", type=Path)
        gt_source.add_argument("--gt-points", type=Path)
    live = commands.add_parser(
        "live", help="start the existing hardware entry point with one ROC trial"
    )
    live_select = live.add_mutually_exclusive_group(required=True)
    live_select.add_argument("--trial-id", type=int)
    live_select.add_argument("--resume", action="store_true")
    live.add_argument("--severity-config", type=Path)
    live.add_argument("--max-attempts", type=int, default=5000)
    live.add_argument(
        "--confirm-hardware", action="store_true", required=True,
        help="required acknowledgement that this command initializes and moves hardware",
    )
    return parser


def main_for(task: str, argv: list[str] | None = None) -> int:
    args = build_parser(task).parse_args(argv)
    if args.command == "layout":
        roc.ensure_layout(); print(roc.ROC_ROOT); return 0
    if args.command == "plan":
        print(roc.create_plan(task, args.seed, force=args.force)); return 0
    if args.command == "analyze":
        summary = roc.analyze(task)
        print(json.dumps(summary["groups"]["all"], ensure_ascii=False, indent=2)); return 0
    if args.command == "preview-errors":
        from detection.pro_handbook.thesis.roc_preview import generate_previews
        summary = generate_previews(
            rgb_path=args.rgb,
            depth_path=args.depth,
            output_dir=args.output_dir,
            config_path=args.config,
            gt_mask_path=args.gt_mask,
            gt_points_path=args.gt_points,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "live":
        row = roc.resolve_trial(task, args.trial_id, resume=args.resume)
        if not args.confirm_hardware:
            raise RuntimeError("--confirm-hardware is required")
        entrypoint = (
            roc.PROJECT_ROOT / "Retrieval_integration_SAM3.py"
            if task == "retrieval"
            else roc.PROJECT_ROOT / "Storage_by_Detection0911.py"
        )
        environment = os.environ.copy()
        environment.update({
            "BOOK_ROC_TASK": task,
            "BOOK_ROC_TRIAL_ID": str(int(row["trial_id"])),
            "BOOK_ROC_MAX_ATTEMPTS": str(int(args.max_attempts)),
        })
        environment.pop("BOOK_ROC_COMPLETED", None)
        if args.severity_config is not None:
            environment["BOOK_ROC_SEVERITY_CONFIG"] = str(
                args.severity_config.resolve()
            )
        print(
            f"[ROC LIVE] trial={row['trial_id']} task={task} "
            f"book={row['book_id']} condition={row['storage_condition']}"
        )
        print(f"[ROC LIVE] exec existing entry point: {entrypoint}")
        os.execve(sys.executable, [sys.executable, str(entrypoint)], environment)
        raise AssertionError("os.execve unexpectedly returned")
    return run_trial(task, args)
