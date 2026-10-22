#!/usr/bin/env python3
"""Thesis storage-recognition experiment wrapper.

The recognition and selection policies are intentionally delegated to the
current importable hardware entry point (``Storage_by_Detection0911.py``) and
``Storage_SAM3.py``.  This file owns only experiment orchestration, timing,
the single measured maximum-opening GT, durable result files, and post-CVAT
evaluation.

Live acquisition moves the robot.  ``--dry-run`` and ``evaluate`` never
initialize RealSense, ROS, the xArm, Dynamixel, or SAM3.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "captures" / "thesis" / "stor"
RESULTS_ROOT = OUTPUT_ROOT / "storage_results"
SCENE_RESULTS_PATH = RESULTS_ROOT / "scene_results.csv"
VIEW_RESULTS_PATH = RESULTS_ROOT / "view_results.csv"
SUMMARY_PATH = RESULTS_ROOT / "summary.json"
LOGS_ROOT = RESULTS_ROOT / "logs"
SELECTED_IMAGE_ROOT = OUTPUT_ROOT / "selected_img"
SELECTED_IMAGE_INFO_PATH = OUTPUT_ROOT / "selected_img_info.csv"
ROOT_INTRINSICS_PATH = OUTPUT_ROOT / "camera_intrinsics.json"
DEFAULT_GT_ROOT = OUTPUT_ROOT / "cvat" / "SegmentationClass"
DEFAULT_LABELMAP_PATH = OUTPUT_ROOT / "cvat" / "labelmap.txt"
DEFAULT_TARGET_LABEL = "storage space"
CURRENT_STORAGE_MODULE = "Storage_by_Detection0911"
WIDTH_PREDICTION_METRIC = "selected_space_row_width_max_mm"
GT_WIDTH_METHOD = "single_measured_maximum_physical_opening_width"
# Lower the thesis acquisition pose after grasp.  The following view2/view3
# moves change only robot Y, so all views retain this same capture height.
THESIS_CAPTURE_Z_OFFSET_MM = -90.0
CASE_MIN = 1
CASE_MAX = 100


SCENE_FIELDS = [
    "scene_id",
    "view1_usable",
    "view2_usable",
    "view3_required",
    "view3_used",
    "view3_usable",
    "selected_view",
    "selected_space_id",
    "selection_reason",
    "space_classification",
    "pred_opening_width_mm",
    "physical_bottleneck_mm",
    "width_prediction_metric",
    "gt_opening_width_mm",
    "gt_width_method",
    "abs_width_error_mm",
    "gt_mask_available",
    "iou",
    "iou_threshold",
    "recognition_success",
    "selected_image_path",
    "selected_pred_mask_path",
    "capture_total_time_ms",
    "robot_motion_total_time_ms",
    "candidate_evaluation_time_ms",
    "viewpoint_selection_time_ms",
    "target_position_time_ms",
    "recognition_total_time_ms",
    "error_reason",
    "created_at",
    # Legacy P5-GT columns are retained only so existing CSV rows can be read
    # and rewritten by resume/evaluate without losing their original values.
    # New maximum-width scenes leave all four columns blank, and summary.json
    # never reads them.
    "measurement_count",
    "measurement_percentile",
    "measured_widths",
    "width_abs_error_mm",
]

VIEW_FIELDS = [
    "scene_id",
    "view_id",
    "candidate_found",
    "usable",
    "selected_candidate_id",
    "pred_opening_width_mm",
    "physical_bottleneck_mm",
    "width_prediction_metric",
    "space_classification",
    "is_final_selected_view",
    "rgb_path",
    "depth_path",
    "raw_depth_path",
    "depth_visualization_path",
    "intrinsics_path",
    "shot_dir",
    "capture_tcp_pose",
    "capture_time_ms",
    "robot_motion_time_ms",
    "recognition_time_ms",
    "candidate_evaluation_time_ms",
    "view_total_processing_time_ms",
    "error_reason",
]

SELECTED_IMAGE_INFO_FIELDS = [
    "scene_id",
    "selected_view",
    "selected_space_id",
    "source_image_path",
    "selected_image_path",
]


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass
class ViewState:
    view_id: str
    capture_index: int
    view_dir: Path
    capture_tcp_pose: tuple[float, ...] | None = None
    result: tuple[float, np.ndarray, dict[str, Any]] | None = None
    candidate: Any = None
    shot_dir: Path | None = None
    capture_time_ms: float | None = None
    robot_motion_time_ms: float | None = None
    recognition_time_ms: float | None = None
    candidate_evaluation_time_ms: float | None = None
    error_reason: str = ""


@dataclass
class SceneOutcome:
    scene_row: dict[str, Any]
    view_rows: list[dict[str, Any]]
    selected_image_info: dict[str, Any] | None = None
    fatal_exception: BaseException | None = None


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv_atomic(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            row = dict(source)
            for key, value in row.items():
                if isinstance(value, (list, tuple, dict)):
                    row[key] = json.dumps(_json_safe(value), ensure_ascii=False)
                elif value is None:
                    row[key] = ""
            writer.writerow(row)
    temporary.replace(path)


def _upsert_rows(
    existing: list[dict[str, Any]],
    additions: Iterable[dict[str, Any]],
    key: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> list[dict[str, Any]]:
    merged = {key(row): row for row in existing}
    for row in additions:
        merged[key(row)] = row
    return [merged[item] for item in sorted(merged)]


def _relative(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(OUTPUT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def prompt_maximum_opening_width() -> float:
    """Read one finite positive maximum physical opening measurement."""
    while True:
        raw = input("Maximum physical opening width [mm]: ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a numeric value.")
            continue
        if not math.isfinite(value) or value <= 0.0:
            print("Please enter a finite positive value.")
            continue
        return value


def _initialize_held_book_grasp(hand_worker: Any) -> None:
    """Reuse the established open/wait/grasp sequence once per live run."""
    print("[THESIS STOR] Initializing held-book grasp (once per run).")
    hand_worker.open_until_full(asynchronous=False)
    time.sleep(5.0)
    hand_worker.grasp()


def _format_optional_mm(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.2f} mm" if math.isfinite(number) else "N/A"


def _print_final_selected_storage_space(
    *,
    scene_id: int,
    selected_view: str,
    selected_space_id: Any,
    classification: Any,
    maximum_width_mm: Any,
    physical_bottleneck_mm: Any,
    selected_image_path: Path,
) -> None:
    """Print the selected space before the operator measures its GT width."""
    print("\n========================================")
    print("Final selected storage space")
    print(f"Scene          : {scene_id}")
    print(f"Selected view  : {selected_view or 'N/A'}")
    print(
        "Space ID       : "
        f"{selected_space_id if selected_space_id is not None else 'N/A'}"
    )
    print(f"Classification : {classification or 'N/A'}")
    print(f"W_max          : {_format_optional_mm(maximum_width_mm)}")
    print(f"P5             : {_format_optional_mm(physical_bottleneck_mm)}")
    print(f"Image          : {_relative(selected_image_path)}")
    print("========================================")


def _prepare_final_space_display_image(
    *,
    visualization_path: Path,
    source_image_path: Path,
    selected_mask_path: Path,
) -> tuple[np.ndarray, str]:
    """Load a selected-space visualization and rotate only the display copy."""
    display_image = (
        cv2.imread(str(visualization_path), cv2.IMREAD_COLOR)
        if visualization_path.is_file()
        else None
    )
    display_source = str(visualization_path)
    if display_image is None:
        source_image = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
        selected_mask = cv2.imread(str(selected_mask_path), cv2.IMREAD_GRAYSCALE)
        if source_image is None or selected_mask is None:
            raise RuntimeError(
                "failed to load selected-space display inputs: "
                f"{source_image_path}, {selected_mask_path}"
            )
        if source_image.shape[:2] != selected_mask.shape:
            raise RuntimeError(
                "selected RGB/mask shape mismatch for display: "
                f"{source_image.shape[:2]} vs {selected_mask.shape}"
            )
        display_image = source_image.copy()
        mask = selected_mask > 0
        highlight = np.zeros_like(display_image)
        highlight[:, :] = (0, 255, 0)
        display_image[mask] = cv2.addWeighted(
            source_image[mask], 0.55, highlight[mask], 0.45, 0.0
        )
        display_source = (
            f"generated display overlay from {source_image_path} and "
            f"{selected_mask_path}"
        )
    display_image = _display_image_as_uint8(display_image)
    return cv2.rotate(display_image, cv2.ROTATE_180), display_source


def _display_image_as_uint8(image: np.ndarray) -> np.ndarray:
    """Convert only the human-confirmation copy to the PNG uint8 range."""
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if not np.issubdtype(array.dtype, np.number):
        raise RuntimeError(f"unsupported display image dtype: {array.dtype}")
    values = array.astype(np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError("display image has no finite values")
    finite_min = float(np.min(finite))
    finite_max = float(np.max(finite))
    values = np.nan_to_num(values, nan=0.0, posinf=255.0, neginf=0.0)
    if finite_min >= 0.0 and finite_max <= 1.0:
        values *= 255.0
    return np.clip(np.rint(values), 0.0, 255.0).astype(np.uint8)


def _save_and_open_final_space_display(
    *,
    scene_id: int,
    visualization_path: Path,
    source_image_path: Path,
    selected_mask_path: Path,
) -> Path:
    """Save the rotated display copy and open it asynchronously with xdg-open."""
    display_image, display_source = _prepare_final_space_display_image(
        visualization_path=visualization_path,
        source_image_path=source_image_path,
        selected_mask_path=selected_mask_path,
    )
    temporary_path = Path(
        f"/tmp/thesis_stor_scene_{scene_id:04d}_selected_space.png"
    )
    if not cv2.imwrite(str(temporary_path), display_image):
        raise RuntimeError(
            f"failed to save selected-space confirmation image: {temporary_path}"
        )
    print(f"Selected-space display source: {display_source}")
    print("Display image:")
    print(f"  shape = {display_image.shape}")
    print(f"  dtype = {display_image.dtype}")
    print(f"  min = {int(np.min(display_image))}")
    print(f"  max = {int(np.max(display_image))}")
    print(f"  temp path = {temporary_path}")
    xdg_open = shutil.which("xdg-open")
    if xdg_open is None:
        print("Could not open image viewer.")
        print(f"Check confirmation image:\n {temporary_path}")
        return temporary_path
    try:
        subprocess.Popen(
            [xdg_open, str(temporary_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        print(f"Could not open image viewer: {exc}")
        print(f"Check confirmation image:\n {temporary_path}")
    return temporary_path


def _print_scene_progress(
    *,
    completed_count: int,
    total_count: int,
    scene_id: int,
    first_scene_id: int,
    last_scene_id: int,
) -> None:
    """Report progress for the resolved acquisition range."""
    print("\n========================================")
    print(f"Completed scene: {completed_count} / {total_count}")
    print(f"Last scene ID  : {scene_id}")
    print(f"Run scene range: {first_scene_id} - {last_scene_id}")
    print("========================================")


def _new_log_path(scene_id: int) -> Path:
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return LOGS_ROOT / f"scene_{scene_id:04d}_{stamp}.log"


def _scene_dir(scene_id: int) -> Path:
    return OUTPUT_ROOT / str(scene_id)


def _prepare_scene_dir(scene_id: int, overwrite: bool) -> Path:
    scene_dir = _scene_dir(scene_id)
    selected_path = SELECTED_IMAGE_ROOT / f"{scene_id:04d}.png"
    if scene_dir.exists() or selected_path.exists():
        confirmed = overwrite
        if not confirmed and sys.stdin.isatty():
            answer = input(
                f"Scene {scene_id} already exists. Archive it and re-run? [y/N]: "
            ).strip().lower()
            confirmed = answer in {"y", "yes"}
        if not confirmed:
            raise FileExistsError(
                f"scene {scene_id} already exists; use --overwrite to archive and re-run"
            )
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        archive = RESULTS_ROOT / "replaced_scenes" / f"scene_{scene_id:04d}_{stamp}"
        archive.mkdir(parents=True, exist_ok=False)
        if scene_dir.exists():
            shutil.move(str(scene_dir), str(archive / "scene"))
        if selected_path.exists():
            shutil.move(str(selected_path), str(archive / selected_path.name))
    scene_dir.mkdir(parents=True, exist_ok=False)
    return scene_dir


def _wait_xarm_motion_done(xarm7: Any, timeout: float = 10.0) -> None:
    """Exact wait rule used by the current storage entry point."""
    start_time = time.monotonic()
    moving_detected = False
    while True:
        result = xarm7.arm.get_is_moving()
        if isinstance(result, bool):
            moving = result
        elif isinstance(result, (tuple, list)) and len(result) >= 2:
            if int(result[0]) != 0:
                raise RuntimeError(f"get_is_moving failed: {result}")
            moving = bool(result[1])
        else:
            raise RuntimeError(f"unexpected get_is_moving result: {result}")
        moving_detected |= moving
        if moving_detected and not moving:
            return
        elapsed = time.monotonic() - start_time
        if elapsed >= 1.0 and not moving_detected and not moving:
            return
        if elapsed >= timeout:
            raise RuntimeError(f"xArm motion timeout: {elapsed:.1f} sec")
        time.sleep(0.02)


def _assert_xarm_capture_ready(xarm7: Any, label: str) -> None:
    state = xarm7.get_state()
    error = int(xarm7.arm.error_code)
    warning = int(xarm7.arm.warn_code)
    print(f"[STORAGE] {label}: state={state}, error={error}, warn={warning}")
    if state in (4, 5) or error != 0:
        raise RuntimeError(
            f"xArm abnormal {label}: state={state}, error={error}, warn={warning}"
        )


def _make_depth_visualization(shot_dir: Path) -> Path | None:
    depth_path = shot_dir / "depth_u16.npy"
    if not depth_path.is_file():
        return None
    depth = np.load(depth_path, allow_pickle=False).astype(np.float32)
    valid = depth > 0
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth[valid], [2.0, 98.0])
        if high <= low:
            high = low + 1.0
        normalized[valid] = np.clip(
            (depth[valid] - low) * 255.0 / (high - low), 0, 255
        ).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    output = shot_dir / "depth_visualization.png"
    if not cv2.imwrite(str(output), color):
        raise RuntimeError(f"failed to save depth visualization: {output}")
    return output


def _shot_dir_from_result(
    view_dir: Path,
    result: tuple[float, np.ndarray, dict[str, Any]] | None,
    storage: Any,
) -> Path | None:
    if result is not None:
        value = result[2].get("shot_dir")
        if value:
            return Path(str(value))
    recovered = storage._latest_capture_shot_dir(view_dir)
    return Path(recovered) if recovered else None


def _capture_and_recognize_view(
    state: ViewState,
    *,
    storage: Any,
    run_capture: Callable[..., Any],
    after_capture_action: Callable[[], None] | None = None,
) -> None:
    capture_started = time.perf_counter()
    capture_finished: float | None = None
    recognition_started: float | None = None

    def after_capture() -> None:
        nonlocal capture_finished, recognition_started
        capture_finished = time.perf_counter()
        if after_capture_action is not None:
            after_capture_action()
        recognition_started = time.perf_counter()

    try:
        state.result = run_capture(
            out_dir=state.view_dir,
            after_capture_callback=after_capture,
            show_debug_images=False,
            target_3d_mode=storage.STORAGE_TARGET_3D_MODE,
        )
    except RuntimeError as exc:
        if not storage._is_retryable_storage_recognition_error(exc):
            raise
        state.error_reason = str(exc)
        print(f"[THESIS STOR] {state.view_id} recognition failure: {exc}")
    finally:
        finished = time.perf_counter()
        if capture_finished is not None:
            state.capture_time_ms = (capture_finished - capture_started) * 1000.0
        if recognition_started is not None:
            state.recognition_time_ms = (finished - recognition_started) * 1000.0

    if capture_finished is None or recognition_started is None:
        raise RuntimeError(f"{state.view_id} capture callback was not reached")
    state.shot_dir = _shot_dir_from_result(state.view_dir, state.result, storage)


def _evaluate_candidate(state: ViewState, storage: Any) -> None:
    started = time.perf_counter()
    if state.result is None:
        state.candidate = storage.StorageCaptureCandidate(
            label=state.view_id,
            capture_index=state.capture_index,
            capture_tcp_pose=state.capture_tcp_pose,
            shot_dir=str(state.shot_dir) if state.shot_dir else None,
            recognition_error=state.error_reason or "recognition result is missing",
        )
    else:
        state.candidate = storage.evaluate_storage_capture_candidate(
            label=state.view_id,
            capture_index=state.capture_index,
            result=state.result,
            capture_tcp_pose=state.capture_tcp_pose,
        )
    state.candidate_evaluation_time_ms = (time.perf_counter() - started) * 1000.0
    if state.candidate.decision is not None:
        storage.log_storage_space_classification(
            state.candidate.decision, f"{state.view_id} capture"
        )
    storage.log_storage_capture_candidate(
        state.candidate,
        continue_to_second_capture=(state.capture_index == 1),
    )


def _calculate_final_target(
    selected: Any,
    storage: Any,
    xarm7: Any,
    *,
    first_target_cam: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Reuse current target helpers without issuing an insertion motion."""
    from xarm7.control.robot_base_coordinate import cam_mm_to_robot_mm

    if selected.result is None or selected.decision is None:
        raise RuntimeError("selected candidate lacks recognition/decision state")
    decision = selected.decision
    motion_target_cam = np.asarray(first_target_cam, dtype=np.float64).reshape(3).copy()
    target_adjustment = "original_recognition_target"
    adjustment_info: dict[str, Any] | None = None
    if storage.should_use_right_tilted_support_sync(decision.classification):
        motion_target_cam, adjustment_info = (
            storage.build_real_world_left_boundary_target_camera(
                metadata, first_target_cam
            )
        )
        target_adjustment = "real_world_left_boundary_target"
    elif decision.classification in {"triangular_right_tilted", "direct_insert"}:
        motion_target_cam, adjustment_info = (
            storage.build_real_world_right_side_target_camera(
                metadata, first_target_cam
            )
        )
        target_adjustment = "real_world_right_side_target"

    frozen_arm = storage.FrozenTcpArm(xarm7, selected.capture_tcp_pose)
    robot_target = np.asarray(
        cam_mm_to_robot_mm(frozen_arm, motion_target_cam * 1000.0),
        dtype=np.float64,
    ).reshape(3)
    normal_robot_target = robot_target.copy()
    high_reach_offset_mm: float | None = None
    if decision.classification == "triangular_left_tilted":
        robot_target, high_reach_offset_mm = (
            storage.build_tilted_book_high_reach_target_robot(robot_target)
        )
    elif decision.classification == "triangular_right_tilted":
        robot_target = storage.apply_left_tilted_reach_correction(robot_target)
        normal_robot_target = np.asarray(robot_target, dtype=np.float64).reshape(3)
        robot_target, high_reach_offset_mm = (
            storage.build_tilted_book_high_reach_target_robot(robot_target)
        )

    return {
        "target_adjustment": target_adjustment,
        "adjustment_info": adjustment_info,
        "original_target_camera_m": np.asarray(first_target_cam).reshape(3).tolist(),
        "motion_target_camera_m": motion_target_cam.tolist(),
        "normal_target_robot_mm": normal_robot_target.tolist(),
        "final_target_robot_mm": np.asarray(robot_target).reshape(3).tolist(),
        "high_reach_offset_mm": high_reach_offset_mm,
        "target_3d_mode": metadata.get("target_3d_mode"),
        "target_3d_source": metadata.get("target_3d_source"),
    }


def _copy_root_intrinsics(shot_dir: Path) -> None:
    source = shot_dir / "camera_intrinsics.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    if ROOT_INTRINSICS_PATH.is_file():
        existing = json.loads(ROOT_INTRINSICS_PATH.read_text(encoding="utf-8"))
        current = json.loads(source.read_text(encoding="utf-8"))
        keys = ("width", "height", "fx", "fy", "ppx", "ppy", "model", "coeffs", "depth_scale")
        if any(existing.get(key) != current.get(key) for key in keys):
            raise RuntimeError(
                "camera intrinsics differ from the experiment-root calibration; "
                "refusing to mix camera configurations"
            )
    else:
        ROOT_INTRINSICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, ROOT_INTRINSICS_PATH)


def _candidate_maximum_opening_width_mm(candidate: Any | None) -> float | None:
    """Return the already-computed maximum physical row width for a candidate."""
    if candidate is None:
        return None
    width_info = getattr(candidate, "selected_space_width_info", None)
    if not isinstance(width_info, dict):
        return None
    try:
        value = float(width_info["row_width_max_mm"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def _view_row(scene_id: int, state: ViewState, selected: Any | None) -> dict[str, Any]:
    candidate = state.candidate
    maximum_width = _candidate_maximum_opening_width_mm(candidate)
    metadata = state.result[2] if state.result is not None else {}
    shot = state.shot_dir
    selected_id = metadata.get("selected_space_id")
    candidate_found = selected_id is not None
    errors = [state.error_reason]
    if candidate is not None and not candidate.usable:
        errors.append(candidate.failure_reason)
    recognition = state.recognition_time_ms or 0.0
    evaluation = state.candidate_evaluation_time_ms or 0.0
    return {
        "scene_id": scene_id,
        "view_id": state.view_id,
        "candidate_found": candidate_found,
        "usable": bool(candidate.usable) if candidate is not None else False,
        "selected_candidate_id": selected_id,
        "pred_opening_width_mm": maximum_width,
        "physical_bottleneck_mm": (
            candidate.physical_bottleneck_mm if candidate is not None else None
        ),
        "width_prediction_metric": (
            WIDTH_PREDICTION_METRIC if maximum_width is not None else None
        ),
        "space_classification": (
            candidate.decision.classification
            if candidate is not None and candidate.decision is not None
            else None
        ),
        "is_final_selected_view": candidate is selected,
        "rgb_path": (
            _relative(
                shot
                / ("input_rgb.npy" if (shot / "input_rgb.npy").is_file() else "input.png")
            )
            if shot
            else ""
        ),
        "depth_path": _relative(shot / "depth_u16.npy") if shot else "",
        "raw_depth_path": _relative(shot / "depth_u16_raw.npy") if shot else "",
        "depth_visualization_path": (
            _relative(shot / "depth_visualization.png")
            if shot and (shot / "depth_visualization.png").is_file()
            else ""
        ),
        "intrinsics_path": _relative(shot / "camera_intrinsics.json") if shot else "",
        "shot_dir": _relative(shot),
        "capture_tcp_pose": state.capture_tcp_pose,
        "capture_time_ms": state.capture_time_ms,
        "robot_motion_time_ms": state.robot_motion_time_ms,
        "recognition_time_ms": state.recognition_time_ms,
        "candidate_evaluation_time_ms": state.candidate_evaluation_time_ms,
        "view_total_processing_time_ms": recognition + evaluation,
        "error_reason": " | ".join(item for item in errors if item),
    }


def _empty_scene_row(scene_id: int) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "view1_usable": False,
        "view2_usable": False,
        "view3_required": False,
        "view3_used": False,
        "view3_usable": False,
        "selected_view": None,
        "selected_space_id": None,
        "selection_reason": None,
        "space_classification": None,
        "pred_opening_width_mm": None,
        "physical_bottleneck_mm": None,
        "width_prediction_metric": None,
        "gt_opening_width_mm": None,
        "gt_width_method": None,
        "abs_width_error_mm": None,
        "gt_mask_available": False,
        "iou": None,
        "iou_threshold": None,
        "recognition_success": None,
        "selected_image_path": None,
        "selected_pred_mask_path": None,
        "capture_total_time_ms": None,
        "robot_motion_total_time_ms": None,
        "candidate_evaluation_time_ms": None,
        "viewpoint_selection_time_ms": None,
        "target_position_time_ms": None,
        "recognition_total_time_ms": None,
        "error_reason": "",
        "created_at": _now_iso(),
        "measurement_count": None,
        "measurement_percentile": None,
        "measured_widths": None,
        "width_abs_error_mm": None,
    }


def _run_scene(
    *,
    scene_id: int,
    scene_dir: Path,
    xarm7: Any,
) -> SceneOutcome:
    storage = importlib.import_module(CURRENT_STORAGE_MODULE)
    from detection.pro_handbook.sam_py_demo.Storage_SAM3 import (
        apply_selected_triangular_space_mask_correction,
        load_saved_capture_rgb,
        load_saved_selected_space_mask,
        run_capture_and_pca_depth_space,
    )

    states: list[ViewState] = []
    scene_row = _empty_scene_row(scene_id)
    selected = None
    selection_reason: str | None = None
    selected_info: dict[str, Any] | None = None
    target_info: dict[str, Any] | None = None
    fatal: BaseException | None = None
    view3_required = False

    try:
        print(f"\n========== THESIS STORAGE SCENE {scene_id} ==========")
        # Retain the current entry point's pre-motion configuration gates.
        storage.validate_storage_0911_depth_constants()
        storage.validate_spacer_virtual_tcp_runtime_configuration()
        motion_total_ms = 0.0

        motion_started = time.perf_counter()
        ret = xarm7.moveJ_to_capture_right_strage(asynchronous=False)
        motion_total_ms += (time.perf_counter() - motion_started) * 1000.0
        if ret != 0:
            raise RuntimeError(f"move to view1 capture pose failed: ret={ret}")
        print(
            "[THESIS STOR] capture Z offset after grasp = "
            f"{THESIS_CAPTURE_Z_OFFSET_MM:+.2f} mm"
        )
        motion_started = time.perf_counter()
        ret = xarm7.moveL_z_offset(
            z_offset=THESIS_CAPTURE_Z_OFFSET_MM,
            asynchronous=False,
        )
        z_motion_ms = (time.perf_counter() - motion_started) * 1000.0
        motion_total_ms += z_motion_ms
        if ret != 0:
            raise RuntimeError(f"storage capture Z offset failed: ret={ret}")
        _assert_xarm_capture_ready(xarm7, "before view1 capture")

        view1 = ViewState("view1", 1, scene_dir / "view1")
        states.append(view1)
        view1.robot_motion_time_ms = motion_total_ms
        view2_motion_started: float | None = None

        def move_to_view2() -> None:
            nonlocal view2_motion_started
            view1.capture_tcp_pose = tuple(
                np.asarray(xarm7.get_tcp_pose(is_radian=True), dtype=np.float64)
                .reshape(6)
                .tolist()
            )
            view2_motion_started = time.perf_counter()
            move_ret = xarm7.moveL_y_offset(
                storage.SECOND_CAPTURE_DY_MM,
                velocity=40,
                acceleration=40,
                asynchronous=True,
            )
            if move_ret != 0:
                raise RuntimeError(f"move to view2 capture pose failed: ret={move_ret}")

        _capture_and_recognize_view(
            view1,
            storage=storage,
            run_capture=run_capture_and_pca_depth_space,
            after_capture_action=move_to_view2,
        )
        if view1.shot_dir is None:
            raise RuntimeError("view1 shot directory is missing")
        _copy_root_intrinsics(view1.shot_dir)
        _wait_xarm_motion_done(xarm7)
        if view2_motion_started is None:
            raise RuntimeError("view2 motion start was not recorded")
        view2_motion_ms = (time.perf_counter() - view2_motion_started) * 1000.0
        _assert_xarm_capture_ready(xarm7, "before view2 capture")

        view2 = ViewState("view2", 2, scene_dir / "view2")
        states.append(view2)
        view2.robot_motion_time_ms = view2_motion_ms
        view2.capture_tcp_pose = tuple(
            np.asarray(xarm7.get_tcp_pose(is_radian=True), dtype=np.float64)
            .reshape(6)
            .tolist()
        )
        _capture_and_recognize_view(
            view2,
            storage=storage,
            run_capture=run_capture_and_pca_depth_space,
        )

        _evaluate_candidate(view1, storage)
        _evaluate_candidate(view2, storage)
        candidates = [view1.candidate, view2.candidate]

        if any(candidate.usable for candidate in candidates):
            selection_started = time.perf_counter()
            selected, selection_reason = storage.select_best_storage_capture(candidates)
            viewpoint_selection_ms = (time.perf_counter() - selection_started) * 1000.0
            storage.log_two_view_selection(candidates, selected, selection_reason)
        else:
            view3_required = True
            current_pose = np.asarray(
                xarm7.get_tcp_pose(is_radian=True), dtype=np.float64
            ).reshape(6)
            first_pose = np.asarray(view1.capture_tcp_pose, dtype=np.float64).reshape(6)
            third_target_y_mm = float(first_pose[1]) - storage.SECOND_CAPTURE_DY_MM
            third_move_dy_mm = third_target_y_mm - float(current_pose[1])
            print(
                "[THESIS STOR] view1/view2 unusable; moving to symmetric view3 "
                f"with dY={third_move_dy_mm:.3f} mm"
            )
            third_motion_started = time.perf_counter()
            ret = xarm7.moveL_y_offset(
                third_move_dy_mm,
                velocity=40,
                acceleration=40,
                asynchronous=True,
            )
            if ret != 0:
                raise RuntimeError(f"move to view3 capture pose failed: ret={ret}")
            _wait_xarm_motion_done(xarm7)
            third_motion_ms = (time.perf_counter() - third_motion_started) * 1000.0
            _assert_xarm_capture_ready(xarm7, "before view3 capture")

            view3 = ViewState("view3", 3, scene_dir / "view3")
            states.append(view3)
            view3.robot_motion_time_ms = third_motion_ms
            view3.capture_tcp_pose = tuple(
                np.asarray(xarm7.get_tcp_pose(is_radian=True), dtype=np.float64)
                .reshape(6)
                .tolist()
            )
            _capture_and_recognize_view(
                view3,
                storage=storage,
                run_capture=run_capture_and_pca_depth_space,
            )
            _evaluate_candidate(view3, storage)
            candidates.append(view3.candidate)
            selection_started = time.perf_counter()
            if view3.candidate.usable:
                selected = view3.candidate
                selection_reason = "view1 and view2 unusable; view3 usable"
                storage.log_two_view_selection(candidates, selected, selection_reason)
            else:
                selection_reason = "all three views unusable"
            viewpoint_selection_ms = (time.perf_counter() - selection_started) * 1000.0

        target_position_ms = 0.0
        if selected is not None:
            target_started = time.perf_counter()
            _, first_target_cam, selected_metadata = selected.result
            first_target_cam, selected_metadata = (
                apply_selected_triangular_space_mask_correction(
                    selected_metadata,
                    first_target_cam,
                    space_type=selected.decision.classification,
                )
            )
            target_info = _calculate_final_target(
                selected,
                storage,
                xarm7,
                first_target_cam=first_target_cam,
                metadata=selected_metadata,
            )
            target_position_ms = (time.perf_counter() - target_started) * 1000.0
            selected_state = next(state for state in states if state.candidate is selected)
            if selected_state.shot_dir is None:
                raise RuntimeError("selected view shot directory is missing")
            _copy_root_intrinsics(selected_state.shot_dir)

            source_image = next(
                (
                    selected_state.shot_dir / name
                    for name in ("input_rgb.npy", "input.png")
                    if (selected_state.shot_dir / name).is_file()
                ),
                None,
            )
            if source_image is None:
                raise FileNotFoundError(
                    f"selected view RGB is missing: {selected_state.shot_dir}"
                )

            SELECTED_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
            selected_image = SELECTED_IMAGE_ROOT / f"{scene_id:04d}.png"
            selected_mask = scene_dir / "selected_pred_mask.png"
            Image.fromarray(
                load_saved_capture_rgb(selected_state.shot_dir),
                mode="RGB",
            ).save(selected_image)
            Image.fromarray(
                load_saved_selected_space_mask(selected_state.shot_dir).astype(
                    np.uint8
                )
                * 255,
                mode="L",
            ).save(selected_mask)

            pred_width = _candidate_maximum_opening_width_mm(selected)
            if pred_width is None:
                raise RuntimeError(
                    "selected candidate has no valid row_width_max_mm"
                )
            physical_bottleneck_mm = selected.physical_bottleneck_mm
            selected_space_id = selected_metadata.get("selected_space_id")
            _print_final_selected_storage_space(
                scene_id=scene_id,
                selected_view=selected_state.view_id,
                selected_space_id=selected_space_id,
                classification=selected.decision.classification,
                maximum_width_mm=pred_width,
                physical_bottleneck_mm=physical_bottleneck_mm,
                selected_image_path=selected_image,
            )
            _save_and_open_final_space_display(
                scene_id=scene_id,
                visualization_path=(
                    selected_state.shot_dir / "final_space_overlay.png"
                ),
                source_image_path=selected_image,
                selected_mask_path=selected_mask,
            )
            gt_width = prompt_maximum_opening_width()
            abs_width_error_mm = abs(pred_width - gt_width)
            _write_json_atomic(
                scene_dir / "measurements.json",
                {
                    "scene_id": scene_id,
                    "measurement_type": GT_WIDTH_METHOD,
                    "measured_maximum_opening_width_mm": gt_width,
                    "gt_opening_width_mm": gt_width,
                    "predicted_maximum_opening_width_mm": pred_width,
                    "physical_bottleneck_mm": physical_bottleneck_mm,
                    "abs_width_error_mm": abs_width_error_mm,
                    "created_at": _now_iso(),
                },
            )
            selected_info = {
                "scene_id": scene_id,
                "selected_view": selected_state.view_id,
                "selected_space_id": selected_space_id,
                "source_image_path": _relative(source_image),
                "selected_image_path": _relative(selected_image),
            }
            scene_row.update(
                {
                    "selected_view": selected_state.view_id,
                    "selected_space_id": selected_space_id,
                    "selection_reason": selection_reason,
                    "space_classification": selected.decision.classification,
                    "pred_opening_width_mm": pred_width,
                    "physical_bottleneck_mm": physical_bottleneck_mm,
                    "width_prediction_metric": WIDTH_PREDICTION_METRIC,
                    "gt_opening_width_mm": gt_width,
                    "gt_width_method": GT_WIDTH_METHOD,
                    "abs_width_error_mm": abs_width_error_mm,
                    "selected_image_path": _relative(selected_image),
                    "selected_pred_mask_path": _relative(selected_mask),
                }
            )
        else:
            failure_details = "; ".join(
                f"{state.view_id}={state.candidate.failure_reason}"
                for state in states
            )
            scene_row["error_reason"] = (
                f"all three views unusable: {failure_details}"
            )

        capture_total = sum(state.capture_time_ms or 0.0 for state in states)
        robot_total = sum(state.robot_motion_time_ms or 0.0 for state in states)
        recognition_views = sum(state.recognition_time_ms or 0.0 for state in states)
        candidate_total = sum(
            state.candidate_evaluation_time_ms or 0.0 for state in states
        )
        recognition_total = (
            recognition_views
            + candidate_total
            + viewpoint_selection_ms
            + target_position_ms
        )
        scene_row.update(
            {
                "view1_usable": bool(states[0].candidate.usable),
                "view2_usable": bool(states[1].candidate.usable),
                "view3_required": view3_required,
                "view3_used": len(states) == 3,
                "view3_usable": bool(states[2].candidate.usable) if len(states) == 3 else False,
                "selection_reason": selection_reason,
                "capture_total_time_ms": capture_total,
                "robot_motion_total_time_ms": robot_total,
                "candidate_evaluation_time_ms": candidate_total,
                "viewpoint_selection_time_ms": viewpoint_selection_ms,
                "target_position_time_ms": target_position_ms,
                "recognition_total_time_ms": recognition_total,
            }
        )
    except BaseException as exc:
        fatal = exc
        scene_row["error_reason"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    view_rows = [_view_row(scene_id, state, selected) for state in states]
    measurements_path = scene_dir / "measurements.json"
    if not measurements_path.exists():
        _write_json_atomic(
            measurements_path,
            {
                "scene_id": scene_id,
                "measurement_type": GT_WIDTH_METHOD,
                "measured_maximum_opening_width_mm": None,
                "gt_opening_width_mm": None,
                "not_recorded_reason": (
                    scene_row.get("error_reason") or "no selected usable space"
                ),
                "created_at": _now_iso(),
            },
        )
    scene_metadata = {
        "scene_id": scene_id,
        "storage_source_module": CURRENT_STORAGE_MODULE,
        "storage_target_3d_mode": getattr(storage, "STORAGE_TARGET_3D_MODE", None),
        "capture_z_offset_mm": THESIS_CAPTURE_Z_OFFSET_MM,
        "view3_required": view3_required,
        "selected_view": scene_row.get("selected_view"),
        "selected_space_id": scene_row.get("selected_space_id"),
        "selection_reason": selection_reason,
        "target_position": target_info,
        "scene_result": scene_row,
        "views": view_rows,
        "fatal_error": scene_row.get("error_reason") if fatal else None,
        "updated_at": _now_iso(),
    }
    _write_json_atomic(scene_dir / "scene_metadata.json", scene_metadata)
    return SceneOutcome(scene_row, view_rows, selected_info, fatal)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool_or_none(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _maximum_width_error_or_none(row: dict[str, Any]) -> float | None:
    """Recompute one maximum-width error and exclude legacy P5-GT rows."""
    if row.get("width_prediction_metric") != WIDTH_PREDICTION_METRIC:
        return None
    if row.get("gt_width_method") != GT_WIDTH_METHOD:
        return None
    prediction = _float_or_none(row.get("pred_opening_width_mm"))
    ground_truth = _float_or_none(row.get("gt_opening_width_mm"))
    if prediction is None or prediction <= 0.0:
        return None
    if ground_truth is None or ground_truth <= 0.0:
        return None
    return abs(prediction - ground_truth)


def _make_summary(rows: list[dict[str, Any]], iou_threshold: float) -> dict[str, Any]:
    total = len(rows)
    successes = [row for row in rows if _bool_or_none(row.get("recognition_success")) is True]
    ious = [value for row in rows if (value := _float_or_none(row.get("iou"))) is not None]
    width_errors = [
        value
        for row in rows
        if (value := _maximum_width_error_or_none(row)) is not None
    ]
    third_required = [row for row in rows if _bool_or_none(row.get("view3_required")) is True]
    third_success = [row for row in third_required if _bool_or_none(row.get("view3_usable")) is True]
    recognition_times = [
        value
        for row in rows
        if (value := _float_or_none(row.get("recognition_total_time_ms"))) is not None
    ]
    return {
        "total_cases": total,
        "recognition_success_count": len(successes),
        "recognition_success_rate_percent": 100.0 * len(successes) / total if total else 0.0,
        "gt_evaluated_count": len(ious),
        "mean_iou": statistics.fmean(ious) if ious else None,
        "valid_width_count": len(width_errors),
        "valid_width_rate_percent": 100.0 * len(width_errors) / total if total else 0.0,
        "width_mae_mm": statistics.fmean(width_errors) if width_errors else None,
        "width_error_sd_mm": statistics.pstdev(width_errors) if width_errors else None,
        "width_error_sd_definition": "population standard deviation of per-scene absolute width errors",
        "width_prediction_metric": WIDTH_PREDICTION_METRIC,
        "width_gt_method": GT_WIDTH_METHOD,
        "width_error_definition": (
            "abs(predicted selected-space maximum physical row width - "
            "single measured maximum physical opening width)"
        ),
        "third_view_required_count": len(third_required),
        "third_view_required_rate_percent": 100.0 * len(third_required) / total if total else 0.0,
        "third_view_success_count": len(third_success),
        "recognition_total_time_mean_ms": statistics.fmean(recognition_times) if recognition_times else None,
        "recognition_total_time_sd_ms": statistics.pstdev(recognition_times) if recognition_times else None,
        "recognition_total_time_valid_count": len(recognition_times),
        "recognition_total_time_definition": (
            "sum of per-view post-capture recognition processing, existing candidate evaluation, "
            "existing viewpoint selection, and final target-position calculation; excludes robot "
            "motion, RGB-D capture, human measurement/input wait, CSV/JSON writes, selected-image "
            "copy, CVAT/GT loading, IoU, and evaluation visualizations"
        ),
        "iou_threshold": iou_threshold,
        "created_at": _now_iso(),
    }


def _persist_outcome(outcome: SceneOutcome, iou_threshold: float) -> None:
    scene_id = int(outcome.scene_row["scene_id"])
    scene_rows = _upsert_rows(
        _read_csv(SCENE_RESULTS_PATH),
        [outcome.scene_row],
        lambda row: (int(row["scene_id"]),),
    )
    existing_view_rows = [
        row
        for row in _read_csv(VIEW_RESULTS_PATH)
        if int(row["scene_id"]) != scene_id
    ]
    view_rows = _upsert_rows(
        existing_view_rows,
        outcome.view_rows,
        lambda row: (int(row["scene_id"]), str(row["view_id"])),
    )
    _write_csv_atomic(SCENE_RESULTS_PATH, scene_rows, SCENE_FIELDS)
    _write_csv_atomic(VIEW_RESULTS_PATH, view_rows, VIEW_FIELDS)
    existing_selected_rows = [
        row
        for row in _read_csv(SELECTED_IMAGE_INFO_PATH)
        if int(row["scene_id"]) != scene_id
    ]
    selected_rows = existing_selected_rows
    if outcome.selected_image_info is not None:
        selected_rows = _upsert_rows(
            existing_selected_rows,
            [outcome.selected_image_info],
            lambda row: (int(row["scene_id"]),),
        )
    _write_csv_atomic(
        SELECTED_IMAGE_INFO_PATH,
        selected_rows,
        SELECTED_IMAGE_INFO_FIELDS,
    )
    _write_json_atomic(SUMMARY_PATH, _make_summary(scene_rows, iou_threshold))


def _parse_labelmap(path: Path) -> dict[str, tuple[int, int, int]]:
    labels: dict[str, tuple[int, int, int]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(":", 3)
        if len(fields) < 2:
            continue
        color = tuple(int(value.strip()) for value in fields[1].split(","))
        if len(color) == 3:
            labels[fields[0].strip()] = color
    return labels


def _load_gt_mask(
    path: Path,
    labelmap_path: Path | None,
    target_label: str,
) -> np.ndarray:
    image = Image.open(path)
    if labelmap_path is not None and labelmap_path.is_file():
        labels = _parse_labelmap(labelmap_path)
        if target_label not in labels:
            raise KeyError(f"target label {target_label!r} not found in {labelmap_path}")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return np.all(rgb == np.asarray(labels[target_label], dtype=np.uint8), axis=2)
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.any(array != 0, axis=2)


def _mask_iou(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    pred = np.asarray(predicted, dtype=bool)
    gt = np.asarray(ground_truth, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"predicted/GT shape mismatch: {pred.shape} / {gt.shape}")
    union = int(np.count_nonzero(pred | gt))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(pred & gt) / union)


def _find_gt_path(gt_root: Path, scene_id: int) -> Path | None:
    for name in (f"{scene_id:04d}.png", f"{scene_id:04d}_rgb.png"):
        candidate = gt_root / name
        if candidate.is_file():
            return candidate
    return None


def run_evaluation(
    *,
    gt_root: Path,
    labelmap_path: Path | None,
    target_label: str,
    iou_threshold: float,
) -> int:
    rows: list[dict[str, Any]] = _read_csv(SCENE_RESULTS_PATH)
    if not rows:
        raise FileNotFoundError(f"no acquisition results: {SCENE_RESULTS_PATH}")
    for row in rows:
        scene_id = int(row["scene_id"])
        gt_path = _find_gt_path(gt_root, scene_id)
        predicted_value = row.get("selected_pred_mask_path")
        predicted_path = OUTPUT_ROOT / str(predicted_value) if predicted_value else None
        row["iou_threshold"] = iou_threshold
        if gt_path is None:
            row["gt_mask_available"] = False
            row["iou"] = None
            row["recognition_success"] = None
            continue
        row["gt_mask_available"] = True
        try:
            if predicted_path is None or not predicted_path.is_file():
                raise FileNotFoundError(predicted_path)
            predicted = np.asarray(Image.open(predicted_path).convert("L")) > 0
            ground_truth = _load_gt_mask(gt_path, labelmap_path, target_label)
            iou = _mask_iou(predicted, ground_truth)
            row["iou"] = iou
            row["recognition_success"] = iou >= iou_threshold
        except Exception as exc:
            previous = str(row.get("error_reason") or "")
            evaluation_error = f"evaluation {type(exc).__name__}: {exc}"
            row["error_reason"] = " | ".join(item for item in (previous, evaluation_error) if item)
            row["iou"] = None
            row["recognition_success"] = False
    rows.sort(key=lambda row: int(row["scene_id"]))
    _write_csv_atomic(SCENE_RESULTS_PATH, rows, SCENE_FIELDS)
    _write_json_atomic(SUMMARY_PATH, _make_summary(rows, iou_threshold))
    print(f"Updated {SCENE_RESULTS_PATH}")
    print(f"Updated {SUMMARY_PATH}")
    return 0


def _resolve_scene_ids(args: argparse.Namespace) -> list[int]:
    if args.resume:
        existing = {int(row["scene_id"]) for row in _read_csv(SCENE_RESULTS_PATH)}
        first = next((value for value in range(CASE_MIN, CASE_MAX + 1) if value not in existing), None)
        if first is None:
            raise RuntimeError("all 100 scenes already exist")
    else:
        first = args.scene_id
    assert first is not None
    last = first + args.scene_count - 1
    if last > CASE_MAX:
        raise ValueError(f"scene range exceeds {CASE_MAX}: {first}..{last}")
    return list(range(first, last + 1))


def _run_live(args: argparse.Namespace) -> int:
    scene_ids = _resolve_scene_ids(args)
    if args.dry_run:
        source = PROJECT_ROOT / f"{CURRENT_STORAGE_MODULE}.py"
        recognition = PROJECT_ROOT / "detection/pro_handbook/sam_py_demo/Storage_SAM3.py"
        if not source.is_file() or not recognition.is_file():
            raise FileNotFoundError(f"current storage sources missing: {source}, {recognition}")
        print("Dry run only; no hardware, camera, model, or output directory was initialized.")
        print("Scenes:", scene_ids)
        print("Output root:", OUTPUT_ROOT)
        print("Current storage module:", CURRENT_STORAGE_MODULE)
        print("Capture Z offset:", f"{THESIS_CAPTURE_Z_OFFSET_MM:+.2f} mm")
        return 0

    from Dynamixel_win_pro_hand_book.dynamixel_worker_client import DynamixelWorkerClient
    from xarm7.control.xarm7 import XArm7
    import rclpy

    hand_worker = None
    xarm7 = None
    node = None
    try:
        hand_worker = DynamixelWorkerClient()
        rclpy.init()
        node = rclpy.create_node("thesis_storage_recognition")
        xarm7 = XArm7(node=node, host=args.xarm_host)
        _initialize_held_book_grasp(hand_worker)
        for offset, scene_id in enumerate(scene_ids):
            scene_dir = _prepare_scene_dir(scene_id, args.overwrite)
            log_path = _new_log_path(scene_id)
            with log_path.open("w", encoding="utf-8") as log_handle:
                with redirect_stdout(Tee(sys.stdout, log_handle)), redirect_stderr(
                    Tee(sys.stderr, log_handle)
                ):
                    outcome = _run_scene(
                        scene_id=scene_id,
                        scene_dir=scene_dir,
                        xarm7=xarm7,
                    )
            _persist_outcome(outcome, args.iou_threshold)
            print(f"Scene {scene_id} saved. Log: {log_path}")
            if outcome.fatal_exception is not None:
                try:
                    xarm7.emergency_stop()
                finally:
                    raise outcome.fatal_exception
            _print_scene_progress(
                completed_count=offset + 1,
                total_count=len(scene_ids),
                scene_id=scene_id,
                first_scene_id=scene_ids[0],
                last_scene_id=scene_ids[-1],
            )
            if offset + 1 < len(scene_ids) and not args.non_interactive_next:
                answer = input("Press Enter for the next scene, or q to stop: ").strip().lower()
                if answer in {"q", "quit"}:
                    break
        return 0
    except KeyboardInterrupt:
        if xarm7 is not None:
            xarm7.emergency_stop()
        print("Storage experiment interrupted; completed scene files were retained.")
        return 130
    except Exception:
        if xarm7 is not None:
            try:
                xarm7.emergency_stop()
            except Exception as stop_exc:
                print("[WARN] xArm emergency_stop failed:", stop_exc)
        raise
    finally:
        if hand_worker is not None:
            try:
                hand_worker.close()
            except Exception as exc:
                print("[WARN] Dynamixel worker close failed:", exc)
        time.sleep(0.2)
        try:
            if xarm7 is not None:
                xarm7.disconnect()
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as exc:
            print("[WARN] cleanup failed:", exc)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire or evaluate thesis storage-recognition scenes."
    )
    subparsers = parser.add_subparsers(dest="command")
    acquire = subparsers.add_parser("acquire", help="run live RGB-D acquisition and recognition")
    scene_group = acquire.add_mutually_exclusive_group(required=True)
    scene_group.add_argument("--scene-id", type=int)
    scene_group.add_argument("--resume", action="store_true")
    acquire.add_argument("--scene-count", type=int, default=1)
    acquire.add_argument("--iou-threshold", type=float, default=0.5)
    acquire.add_argument("--xarm-host", default="192.168.2.197")
    acquire.add_argument("--overwrite", action="store_true")
    acquire.add_argument("--non-interactive-next", action="store_true")
    acquire.add_argument(
        "--dry-run",
        action="store_true",
        help="validate configuration without hardware/model startup or filesystem writes",
    )

    evaluate = subparsers.add_parser("evaluate", help="update IoU after CVAT annotation")
    evaluate.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_ROOT)
    evaluate.add_argument("--labelmap", type=Path, default=DEFAULT_LABELMAP_PATH)
    evaluate.add_argument("--target-label", default=DEFAULT_TARGET_LABEL)
    evaluate.add_argument("--iou-threshold", type=float, default=0.5)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.command is None:
        parser.error("choose 'acquire' or 'evaluate'")
    if not 0.0 <= args.iou_threshold <= 1.0:
        parser.error("--iou-threshold must be between 0 and 1")
    if args.command == "acquire":
        if args.scene_id is not None and not CASE_MIN <= args.scene_id <= CASE_MAX:
            parser.error(f"--scene-id must be between {CASE_MIN} and {CASE_MAX}")
        if args.scene_count <= 0:
            parser.error("--scene-count must be positive")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    if args.command == "evaluate":
        labelmap = args.labelmap if args.labelmap.is_file() else None
        return run_evaluation(
            gt_root=args.gt_dir,
            labelmap_path=labelmap,
            target_label=args.target_label,
            iou_threshold=args.iou_threshold,
        )
    return _run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
