"""Explicitly enabled bridges from ROC trials to existing live entry points."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from detection.pro_handbook.thesis import roc_experiment as roc


ENV_TASK = "BOOK_ROC_TASK"
ENV_TRIAL_ID = "BOOK_ROC_TRIAL_ID"
ENV_ACTIVE_DIR = "BOOK_ROC_ACTIVE_TRIAL_DIR"
ENV_MAX_ATTEMPTS = "BOOK_ROC_MAX_ATTEMPTS"
ENV_SEVERITY_CONFIG = "BOOK_ROC_SEVERITY_CONFIG"
ENV_COMPLETED = "BOOK_ROC_COMPLETED"


def active(task: str | None = None) -> bool:
    configured = os.environ.get(ENV_TASK, "").strip()
    return bool(configured) and (task is None or configured == task)


def completed() -> bool:
    return os.environ.get(ENV_COMPLETED, "") == "1"


def storage_condition() -> str | None:
    """Return the active storage ROC condition without initializing hardware."""
    if not active("storage"):
        return None
    condition = _row("storage")["storage_condition"]
    if condition not in {"with_held_book", "without_held_book"}:
        raise RuntimeError(f"unknown storage ROC condition: {condition!r}")
    return condition


def storage_non_grasp_active() -> bool:
    return storage_condition() == "without_held_book"


def _row(task: str) -> dict[str, str]:
    if not active(task):
        raise RuntimeError(f"ROC live context is not active for {task}")
    raw = os.environ.get(ENV_TRIAL_ID, "").strip()
    if not raw:
        raise RuntimeError(f"{ENV_TRIAL_ID} is missing")
    return roc.resolve_trial(task, int(raw), resume=False)


def _severity() -> dict[str, Any] | None:
    raw = os.environ.get(ENV_SEVERITY_CONFIG, "").strip()
    return None if not raw else json.loads(Path(raw).read_text(encoding="utf-8"))


def _prepare(
    task: str, rgb_bgr: np.ndarray, depth: np.ndarray,
    *, original_prediction: Path | None = None,
) -> tuple[np.ndarray, dict[str, str], Path]:
    row = _row(task)
    canonical_directory = roc.trial_dir(row)
    if canonical_directory.exists():
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        directory = canonical_directory / "retry_attempts" / stamp
    else:
        directory = canonical_directory
    points = roc.annotate_polygon_8(rgb_bgr, title=f"ROC {task} trial {row['trial_id']}")
    gt = roc.polygon_mask(rgb_bgr.shape[:2], points)
    generated = roc.generate_for_bin(
        gt, row["error_type"], int(row["target_iou_bin"]), int(row["random_seed"]),
        severity_ranges=_severity(),
        max_attempts=int(os.environ.get(ENV_MAX_ATTEMPTS, "5000")),
    )
    paths = roc.save_mask_artifacts(directory, rgb_bgr, gt, generated.mask, points, generated)
    depth_path = directory / "input_depth.npy"
    np.save(depth_path, np.asarray(depth), allow_pickle=False)
    paths["depth_path"] = str(depth_path)
    original_path = ""
    if original_prediction is not None and original_prediction.is_file():
        destination = directory / "original_prediction_mask.png"
        shutil.copy2(original_prediction, destination)
        original_path = str(destination)
    metadata = {
        **row, **paths, "actual_iou": generated.iou,
        "error_parameters": generated.parameters, "gt_polygon_points": points,
        "original_prediction_mask_path": original_path,
        "calculated_grasp_position": None, "calculated_storage_position": None,
        "success": "", "timestamp": roc.now_iso(), "dry_run": False,
        "robot_motion_executed": False, "candidate_attempts": generated.attempts,
        "downstream_result_path": "",
    }
    metadata_path = directory / "trial_metadata.json"
    metadata["trial_metadata_path"] = str(metadata_path)
    roc.write_json_atomic(metadata_path, metadata)
    os.environ[ENV_ACTIVE_DIR] = str(directory)
    return generated.mask, row, directory


def prepare_retrieval_override(shot_dir: str | Path) -> np.ndarray:
    shot = Path(shot_dir)
    rgb = cv2.imread(str(shot / "after_init_rgb.png"), cv2.IMREAD_COLOR)
    depth = np.load(shot / "after_init_depth.npy", allow_pickle=False)
    if rgb is None:
        raise FileNotFoundError(shot / "after_init_rgb.png")
    mask, _, _ = _prepare("retrieval", rgb, depth)
    return mask


def prepare_storage_override(
    selected_candidate: Any,
    *, evaluate_candidate: Callable[..., Any],
) -> Any:
    """Recalculate only the selected view using one artificial mask."""
    if not active("storage"):
        return selected_candidate
    if not selected_candidate.shot_dir:
        raise RuntimeError("selected storage candidate has no shot_dir")
    shot = Path(selected_candidate.shot_dir)
    rgb_array_path = shot / "input_rgb.npy"
    if rgb_array_path.is_file():
        rgb_rgb = np.load(rgb_array_path, allow_pickle=False)
        rgb_bgr = cv2.cvtColor(np.asarray(rgb_rgb, np.uint8), cv2.COLOR_RGB2BGR)
    else:
        rgb_bgr = cv2.imread(str(shot / "input.png"), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise FileNotFoundError(shot / "input.png")
        rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    depth_raw = np.load(shot / "depth_u16_raw.npy", allow_pickle=False)
    original = shot / "final_space_mask.png"
    artificial, _, directory = _prepare(
        "storage", rgb_bgr, depth_raw, original_prediction=original,
    )
    import offline_100test_storage_space_sam3 as storage_recognition
    from detection.pro_handbook.sam_py_demo import Storage_SAM3
    params = json.loads((shot / "camera_intrinsics.json").read_text(encoding="utf-8"))
    intr = Storage_SAM3.intrinsics_from_camera_params(params)
    spine = storage_recognition.load_npz(shot / "book_spine_masks.npz")
    ends = storage_recognition.load_npz(shot / "book_end_masks.npz")
    downstream_dir = directory / "downstream"
    result = Storage_SAM3.run_from_rgbd(
        rgb_rgb, depth_raw, intr, float(params["depth_scale"]),
        shot_dir=downstream_dir, spine_instances=spine, end_instances=ends,
        apply_legacy_depth_ignore=bool(params.get("legacy_depth_ignore_applied", False)),
        selected_space_mask_override=artificial,
    )
    candidate = evaluate_candidate(
        label=selected_candidate.label,
        capture_index=selected_candidate.capture_index,
        result=result,
        capture_tcp_pose=selected_candidate.capture_tcp_pose,
    )
    _update_metadata(
        downstream_result_path=str(downstream_dir / "geometry_metadata.json"),
        original_prediction_mask_path=str(directory / "original_prediction_mask.png"),
    )
    return candidate


def _metadata_path() -> Path:
    raw = os.environ.get(ENV_ACTIVE_DIR, "").strip()
    if not raw:
        raise RuntimeError("ROC active trial directory is unavailable")
    return Path(raw) / "trial_metadata.json"


def _update_metadata(**updates: Any) -> dict[str, Any]:
    path = _metadata_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(updates)
    roc.write_json_atomic(path, data)
    return data


def _open_review(directory: Path) -> None:
    names = ["input_rgb.png", "gt_overlay.png", "artificial_mask_overlay.png",
             "gt_artificial_comparison_overlay.png"]
    images = [cv2.imread(str(directory / name), cv2.IMREAD_COLOR) for name in names]
    if any(image is None for image in images):
        return
    height, width = images[0].shape[:2]
    resized = [cv2.resize(image, (width // 2, height // 2)) for image in images]
    review = np.vstack((np.hstack(resized[:2]), np.hstack(resized[2:])))
    review_path = directory / "pre_motion_review.png"
    cv2.imwrite(str(review_path), review)
    try:
        subprocess.Popen(["xdg-open", str(review_path)], start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as exc:
        print(f"[ROC] review viewer could not be opened: {exc}")


def confirm_pre_motion(
    task: str, *, calculated_grasp_position: Any = None,
    calculated_storage_position: Any = None, validity: Any = None,
    prompt_for_motion: bool = True,
) -> None:
    if not active(task):
        return
    data = _update_metadata(
        calculated_grasp_position=calculated_grasp_position,
        calculated_storage_position=calculated_storage_position,
        downstream_validity=validity,
    )
    directory = Path(os.environ[ENV_ACTIVE_DIR])
    original_source = None
    if task == "retrieval":
        # The current retrieval recognizer writes this after GT annotation.
        # Find it from its recorded returned shot directory when available.
        shot = data.get("recognition_shot_dir")
        if shot:
            original_source = Path(shot) / "original_prediction_mask.png"
    if original_source is not None and original_source.is_file():
        destination = directory / "original_prediction_mask.png"
        if not destination.exists():
            shutil.copy2(original_source, destination)
        _update_metadata(original_prediction_mask_path=str(destination))
    _open_review(directory)
    print("\n[ROC PRE-MOTION REVIEW]")
    for key in ("trial_id", "task", "book_id", "storage_condition",
                "target_iou_bin", "actual_iou", "error_type", "error_parameters"):
        print(f"{key}: {data.get(key)}")
    print(f"calculated_grasp_position: {calculated_grasp_position}")
    print(f"calculated_storage_position: {calculated_storage_position}")
    print(f"validity: {validity}")
    if not prompt_for_motion:
        return
    while True:
        answer = input("Proceed with robot motion? [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            _update_metadata(robot_motion_confirmed=True)
            return
        if answer in {"n", "no"}:
            _update_metadata(robot_motion_confirmed=False)
            raise KeyboardInterrupt("ROC robot motion declined by operator")


def prepare_non_grasp_insertion_book(
    hand_worker: Any,
    *,
    input_fn: Callable[[str], str] | None = None,
) -> None:
    """Run the two-stage post-recognition setup for one non-grasp trial.

    The physical calls deliberately reuse the same synchronous open and grasp
    worker APIs used at the start of Strage_by_Detection1006.py.  No camera or
    recognition function is called from this transition.
    """
    if not storage_non_grasp_active():
        raise RuntimeError(
            "non-grasp insertion-book setup requires an active "
            "without_held_book storage ROC trial"
        )
    read_input = input if input_fn is None else input_fn

    print("\nNon-grasp image acquisition and recognition completed.")
    read_input("Press Enter to open the gripper for insertion-book setup...")
    hand_worker.open_until_full(asynchronous=False)
    _update_metadata(non_grasp_gripper_opened_after_recognition=True)

    read_input(
        "Place the insertion book in the gripper.\n"
        "Press Enter when the book is correctly positioned..."
    )
    hand_worker.grasp()
    _update_metadata(
        non_grasp_insertion_book_grasped=True,
        robot_motion_confirmed=True,
    )
    print("[ROC NON-GRASP] insertion book grasped; proceeding to storage motion")


def set_retrieval_result_context(shot_dir: str | Path) -> None:
    if active("retrieval"):
        shot = Path(shot_dir)
        directory = Path(os.environ[ENV_ACTIVE_DIR])
        copied: dict[str, str] = {}
        for source_name, destination_name in (
            ("offline_recognition_result.json", "downstream_result.json"),
            ("final.png", "downstream_result.png"),
            ("original_prediction_mask.png", "original_prediction_mask.png"),
            ("artificial_mask_used_for_depth.png", "artificial_mask_used_for_depth.png"),
        ):
            source = shot / source_name
            destination = directory / destination_name
            if source.is_file() and not destination.exists():
                shutil.copy2(source, destination)
                copied[source_name] = str(destination)
        _update_metadata(
            recognition_shot_dir=str(shot),
            downstream_result_path=copied.get(
                "offline_recognition_result.json", ""
            ),
            original_prediction_mask_path=copied.get(
                "original_prediction_mask.png", ""
            ),
        )


def record_manual_success(task: str) -> bool:
    if not active(task):
        return False
    prompt = (
        "Retrieval completed successfully? [y/n]: "
        if task == "retrieval" else "Storage completed successfully? [y/n]: "
    )
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"y", "yes", "n", "no"}:
            success = answer in {"y", "yes"}
            break
    data = _update_metadata(
        success=success,
        completed=True,
        robot_motion_executed=True,
        completed_at=roc.now_iso(),
    )
    row = _row(task)
    roc.update_trial_records(task, row, data)
    os.environ[ENV_COMPLETED] = "1"
    os.environ.pop(ENV_TASK, None)
    os.environ.pop(ENV_TRIAL_ID, None)
    return success
