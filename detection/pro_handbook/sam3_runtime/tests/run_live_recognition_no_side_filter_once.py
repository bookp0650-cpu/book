from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from detection.pro_handbook.sam_py_demo.get_book_points_no_mask_merge_no_side_filter import (
    CAPTURES_DIR,
    MASTER_JSON,
    TargetMaskSelectionError,
    Tee,
    VARIANT,
    _read_json,
    _unique_live_run_dir,
    _write_json,
    run_capture_and_pca_no_mask_merge_no_side_filter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNTIME_DIR = Path(__file__).resolve().parents[1]
SERVICE_PYTHON = RUNTIME_DIR / ".venv" / "bin" / "python"
OCR_PYTHON = (
    PROJECT_ROOT
    / "detection"
    / "pro_handbook"
    / "sam_py_demo"
    / "OCR"
    / ".paadle_ocr"
    / "bin"
    / "python"
)
HEALTH_URL = "http://127.0.0.1:8765/health"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _health(timeout: float = 1.0):
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as response:
            payload = json.load(response)
        return bool(payload.get("ready")), payload
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return False, None


def _start_service_if_needed(log_handle):
    ready, payload = _health()
    if ready:
        print("[SERVICE] using pre-existing ready service")
        return None, False, payload
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": str(PROJECT_ROOT),
        }
    )
    process = subprocess.Popen(
        [
            str(SERVICE_PYTHON),
            "-m",
            "detection.pro_handbook.sam3_runtime.service.service",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SAM3 service exited early with code {process.returncode}")
        ready, payload = _health()
        if ready:
            print(f"[SERVICE] started owned service pid={process.pid}")
            return process, True, payload
        time.sleep(0.5)
    process.terminate()
    process.wait(timeout=10)
    raise TimeoutError("SAM3 service did not become ready within 60 seconds")


def _stop_owned_service(process, owned: bool) -> bool:
    if process is None or not owned:
        return False
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    print(f"[SERVICE] stopped owned service pid={process.pid}")
    return True


def _realsense_preflight() -> dict:
    context = rs.context()
    devices = list(context.query_devices())
    if not devices:
        raise RuntimeError("no RealSense device found")
    device = devices[0]
    sensors = list(device.query_sensors())
    profiles = [profile for sensor in sensors for profile in sensor.get_stream_profiles()]
    has_color = any(profile.stream_type() == rs.stream.color for profile in profiles)
    has_depth = any(profile.stream_type() == rs.stream.depth for profile in profiles)
    if not has_color or not has_depth:
        raise RuntimeError(
            f"required streams unavailable: color={has_color} depth={has_depth}"
        )

    def info(key):
        try:
            return device.get_info(key) if device.supports(key) else None
        except Exception:
            return None

    return {
        "device_count": len(devices),
        "color_stream_available": has_color,
        "depth_stream_available": has_depth,
        "device_name": info(rs.camera_info.name),
        "serial_number": info(rs.camera_info.serial_number),
        "firmware_version": info(rs.camera_info.firmware_version),
    }


def _robot_process_preflight() -> dict:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    patterns = (
        "xarm7/control",
        "xarm_init_to_capture",
        "dynamixel",
        "iai_cylinder",
        "Retrieval_integration.py",
        "Retrieval_integration_editing.py",
    )
    matches = [
        line.strip()
        for line in completed.stdout.splitlines()
        if any(pattern in line for pattern in patterns)
        and "run_live_recognition_no_side_filter_once" not in line
    ]
    if matches:
        raise RuntimeError(f"robot-control-related process is running: {matches}")
    return {"checked_patterns": list(patterns), "matches": [], "safe": True}


def _ocr_preflight() -> dict:
    if not OCR_PYTHON.is_file():
        raise FileNotFoundError(OCR_PYTHON)
    completed = subprocess.run(
        [str(OCR_PYTHON), "-c", "import paddle, paddleocr; print(paddle.__version__)"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        env={key: value for key, value in os.environ.items() if key != "LD_LIBRARY_PATH"},
    )
    return {"python": str(OCR_PYTHON), "probe": completed.stdout.strip()}


def main() -> int:
    started = datetime.now().astimezone()
    total_start = time.perf_counter()
    shot_dir = _unique_live_run_dir(started)
    result_path = shot_dir / "live_recognition_result.json"
    log_path = shot_dir / "live_recognition_console.log"
    service_process = None
    service_owned = False
    service_stopped = False
    failure_stage = "preflight"
    result = {
        "success": False,
        "mode": "live_recognition_only",
        "variant": VARIANT,
        "timestamp": started.isoformat(),
        "run_shot_dir": str(shot_dir),
        "master_json": str(MASTER_JSON.resolve()),
        "master_index": 0,
        "robot_control_executed": False,
        "realsense_capture_count": 0,
        "error": None,
    }

    with log_path.open("w", encoding="utf-8") as log_handle:
        tee, err = Tee(sys.__stdout__, log_handle), Tee(sys.__stderr__, log_handle)
        with redirect_stdout(tee), redirect_stderr(err):
            try:
                master = _read_json(MASTER_JSON)
                query = master[0]["book_name"]
                gt_width = float(master[0]["book_width"])
                result.update({"target_query": query, "gt_book_width_mm": gt_width})
                print(f"run_shot_dir: {shot_dir}")
                print(f"master_index: 0")
                print(f"target_query: {query}")
                print(f"gt_book_width_mm: {gt_width}")

                robot_preflight = _robot_process_preflight()
                camera_preflight = _realsense_preflight()
                ocr_preflight = _ocr_preflight()
                result["preflight"] = {
                    "robot": robot_preflight,
                    "realsense": camera_preflight,
                    "ocr": ocr_preflight,
                }
                print(f"[PREFLIGHT] RealSense: {camera_preflight}")
                print("[PREFLIGHT] robot-control process: none")
                print(f"[PREFLIGHT] PaddleOCR: {ocr_preflight}")

                service_process, service_owned, health = _start_service_if_needed(
                    log_handle
                )
                result["sam3_service"] = {
                    "owned_by_this_script": service_owned,
                    "health_before_capture": health,
                }

                failure_stage = "capture_and_recognition"
                print("live RGB-D capture and recognition invocation: 1 of 1")
                roll, point, width_mm, returned_shot_dir = (
                    run_capture_and_pca_no_mask_merge_no_side_filter(
                        query=query,
                        sam_device="gpu",
                        shot_dir=shot_dir,
                    )
                )
                core = _read_json(shot_dir / "live_core_result.json")
                capture_meta = _read_json(
                    shot_dir / "realsense_capture_metadata.json"
                )
                result["realsense_capture_count"] = 1
                depth = np.load(shot_dir / "after_init_depth.npy", allow_pickle=False)
                selected = (
                    cv2.imread(
                        str(shot_dir / "selected_mask_used_for_depth.png"),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    > 0
                )
                result.update(
                    {
                        "success": True,
                        "pred_book_width_mm": float(width_mm),
                        "abs_error_mm": abs(float(width_mm) - gt_width),
                        "roll_rad": float(roll),
                        "roll_deg": float(np.degrees(roll)),
                        "point_3d": np.asarray(point, dtype=float).tolist(),
                        "sam3": {
                            "raw_mask_count": core["raw_mask_count"],
                            "nms_mask_count": core["nms_mask_count"],
                            "selected_mask_index": core["selected_mask_index"],
                            "selected_mask_index_convention": "1-based",
                            "selected_mask_score": core["selected_mask_score"],
                            "selected_mask_area_px": core[
                                "selected_mask_area_px"
                            ],
                            "mask_derived_bbox_xyxy": core[
                                "mask_derived_bbox_xyxy"
                            ],
                        },
                        "ocr": {
                            "selected_text": core["selected_ocr_text"],
                            "confidence": core["selected_ocr_confidence"],
                            "matching_score": core["ocr_matching_score"],
                            "candidate_count": core["ocr_candidate_count"],
                        },
                        "depth": {
                            "shape": list(depth.shape),
                            "dtype": str(depth.dtype),
                            "scale": capture_meta.get("depth_scale"),
                            "valid_pixel_count": int(np.count_nonzero(depth > 0)),
                            "selected_mask_valid_pixel_count": int(
                                np.count_nonzero(selected & (depth > 0))
                            ),
                        },
                        "point_counts": core["point_counts"],
                        "disabled_processing": {
                            "legacy_non_book_mask_removal": True,
                            "legacy_floating_mask_merge": True,
                            "side_surface_triggered_point_filter": True,
                        },
                        "retained_processing": {
                            "owner_mask_nms": True,
                            "median_depth_filter": True,
                            "normal_ransac": True,
                            "pca": True,
                        },
                        "verification": core["verification"],
                        "timings": {
                            **core["timings"],
                            "capture_seconds": core["capture_seconds"],
                            "sam3_inference_seconds": core[
                                "sam3_inference_seconds"
                            ],
                            "ocr_inference_seconds": core[
                                "ocr_inference_seconds"
                            ],
                            "total_seconds": time.perf_counter() - total_start,
                        },
                        "gpu_peak_memory_mib": core["gpu_max_memory_mb"],
                        "returned_shot_dir": str(Path(returned_shot_dir).resolve()),
                    }
                )
                print(
                    f"selected mask={core['selected_mask_index']} "
                    f"score={core['selected_mask_score']} area={core['selected_mask_area_px']}"
                )
                print(f"point counts={core['point_counts']}")
                print(
                    f"roll={result['roll_rad']} width_mm={width_mm} "
                    f"point_3d={result['point_3d']}"
                )
            except Exception as exc:
                if isinstance(exc, TargetMaskSelectionError):
                    failure_stage = "mask_selection"
                result.update(
                    {
                        "success": False,
                        "failure_stage": failure_stage,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                        "timings": {"total_seconds": time.perf_counter() - total_start},
                    }
                )
                traceback.print_exc()
            finally:
                try:
                    service_stopped = _stop_owned_service(
                        service_process, service_owned
                    )
                except Exception:
                    traceback.print_exc()
                result.setdefault("sam3_service", {})
                result["sam3_service"].update(
                    {
                        "owned_by_this_script": service_owned,
                        "owned_service_stopped": service_stopped,
                    }
                )
                result["outputs"] = {
                    path.name: str(path.resolve())
                    for path in sorted(shot_dir.iterdir())
                    if path.is_file()
                }
                result["outputs"][result_path.name] = str(result_path.resolve())
                _write_json(result_path, result)
                print(f"result_json: {result_path}")

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
