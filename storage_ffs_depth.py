"""Fast-FoundationStereo depth correction for the storage experiment.

This module keeps Fast-FoundationStereo in a separate process so its Python,
CUDA, and PyTorch requirements cannot silently replace the existing SAM3
environment.  The external runner contract is configured with environment
variables and is validated before RealSense capture starts.

Recommended environment variables::

    FFS_PYTHON=/path/to/ffs/environment/bin/python
    FFS_REPOSITORY=/path/to/NVlabs/Fast-FoundationStereo
    FFS_MODEL_WEIGHTS=/path/to/model_weight

Advanced users may instead provide ``FFS_COMMAND_JSON`` with placeholders
``{left}``, ``{right}``, ``{output_disparity}``,
``{output_right_disparity}``, ``{weights}``, and ``{shot_dir}``.

The runner must write an HxW float ``.npy`` disparity in rectified-left pixel
units to ``{output_disparity}``.  It may additionally write positive
right-to-left disparity to ``{output_right_disparity}``; when absent, the log
explicitly records that left-right consistency is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import json
import math
import os
import shutil
import subprocess
import time

import cv2
import numpy as np


BOOK_SPINE_ERODE_PX = 3
MIN_COMMON_VALID_PIXELS = 200
MIN_COMMON_VALID_RATIO = 0.10
FFS_DEPTH_MIN_M = 0.10
FFS_DEPTH_MAX_M = 1.50
CORRECTED_DEPTH_MIN_M = 0.25
CORRECTED_DEPTH_MAX_M = 0.50
LEFT_RIGHT_CONSISTENCY_THRESHOLD_PX = 1.0
FFS_INFERENCE_TIMEOUT_SEC = 180.0


class FfsDepthCorrectionError(RuntimeError):
    """Safety-stop error that must never trigger RealSense-only insertion."""


@dataclass(frozen=True)
class FfsRuntimeConfig:
    command_template: tuple[str, ...]
    weights_path: Path
    work_dir: Path | None
    timeout_sec: float


@dataclass(frozen=True)
class StereoCapture:
    rgb_bgr: np.ndarray
    depth_u16_aligned_rgb: np.ndarray
    color_intrinsics_object: Any
    depth_scale: float
    infrared_left: np.ndarray
    infrared_right: np.ndarray
    calibration: dict[str, Any]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _save_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), np.asarray(image)):
        raise FfsDepthCorrectionError(f"failed to save image: {path}")


def _intrinsics_dict(intr: Any) -> dict[str, Any]:
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "model": str(intr.model),
        "coeffs": [float(value) for value in intr.coeffs],
    }


def _extrinsics_dict(extr: Any) -> dict[str, Any]:
    rotation = np.asarray(extr.rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(extr.translation, dtype=np.float64).reshape(3)
    return {
        "rotation": rotation.tolist(),
        "translation_m": translation.tolist(),
    }


def load_ffs_runtime_config() -> FfsRuntimeConfig:
    """Validate the isolated FFS runner and weight without executing either."""
    raw_command = os.environ.get("FFS_COMMAND_JSON", "").strip()
    raw_weights = os.environ.get("FFS_MODEL_WEIGHTS", "").strip()
    if not raw_weights:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: FFS_MODEL_WEIGHTS is not configured"
        )
    if raw_command:
        try:
            command = json.loads(raw_command)
        except json.JSONDecodeError as exc:
            raise FfsDepthCorrectionError(
                f"FFS depth correction invalid: FFS_COMMAND_JSON is invalid: {exc}"
            ) from exc
    else:
        raw_python = os.environ.get("FFS_PYTHON", "").strip()
        raw_repository = os.environ.get("FFS_REPOSITORY", "").strip()
        if not raw_python or not raw_repository:
            raise FfsDepthCorrectionError(
                "FFS depth correction invalid: configure FFS_PYTHON, "
                "FFS_REPOSITORY, and FFS_MODEL_WEIGHTS"
            )
        repository = Path(raw_repository).expanduser().resolve()
        if not (repository / "core").is_dir() or not (
            repository / "Utils.py"
        ).is_file():
            raise FfsDepthCorrectionError(
                "FFS depth correction invalid: official repository layout "
                f"is unavailable: {repository}"
            )
        runner = Path(__file__).resolve().with_name(
            "fast_foundation_stereo_runner.py"
        )
        scale = os.environ.get("FFS_SCALE", "0.5").strip()
        command = [
            raw_python,
            str(runner),
            "--repository",
            str(repository),
            "--weights",
            "{weights}",
            "--left",
            "{left}",
            "--right",
            "{right}",
            "--output-disparity",
            "{output_disparity}",
            "--output-right-disparity",
            "{output_right_disparity}",
            "--scale",
            scale,
            "--valid-iters",
            os.environ.get("FFS_VALID_ITERS", "8").strip(),
            "--max-disp",
            os.environ.get("FFS_MAX_DISP", "192").strip(),
        ]
    if not isinstance(command, list) or not command or not all(
        isinstance(value, str) and value for value in command
    ):
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: FFS_COMMAND_JSON must be a "
            "non-empty JSON string array"
        )
    executable = command[0]
    if Path(executable).is_absolute():
        executable_available = Path(executable).is_file()
    else:
        executable_available = shutil.which(executable) is not None
    if not executable_available:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: runner executable is unavailable: "
            f"{executable}"
        )
    weights_path = Path(raw_weights).expanduser().resolve()
    if not weights_path.is_file():
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: model weight is unavailable: "
            f"{weights_path}"
        )
    raw_work_dir = os.environ.get("FFS_WORK_DIR", "").strip()
    work_dir = Path(raw_work_dir).expanduser().resolve() if raw_work_dir else None
    if work_dir is not None and not work_dir.is_dir():
        raise FfsDepthCorrectionError(
            f"FFS depth correction invalid: FFS_WORK_DIR is unavailable: {work_dir}"
        )
    return FfsRuntimeConfig(
        command_template=tuple(command),
        weights_path=weights_path,
        work_dir=work_dir,
        timeout_sec=FFS_INFERENCE_TIMEOUT_SEC,
    )


def _opencv_camera(intrinsics: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["ppx"]],
            [0.0, intrinsics["fy"], intrinsics["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    coefficients = np.asarray(intrinsics.get("coeffs", []), dtype=np.float64)
    if coefficients.size == 0:
        coefficients = np.zeros(5, dtype=np.float64)
    if not np.isfinite(matrix).all() or not np.isfinite(coefficients).all():
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: non-finite stereo intrinsics"
        )
    model = str(intrinsics.get("model", "")).lower().split(".")[-1]
    nonzero_distortion = bool(np.max(np.abs(coefficients)) > 1.0e-12)
    supported = model in {
        "none",
        "brown_conrady",
        "modified_brown_conrady",
    }
    if nonzero_distortion and not supported:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: OpenCV rectification is not "
            f"defined for active distortion model {model!r}"
        )
    return matrix, coefficients


def rectify_stereo_pair(
    infrared_left: np.ndarray,
    infrared_right: np.ndarray,
    calibration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    """Rectify the active IR pair; raw-stream rectification is never assumed."""
    left = np.asarray(infrared_left, dtype=np.uint8)
    right = np.asarray(infrared_right, dtype=np.uint8)
    if left.ndim != 2 or right.shape != left.shape:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: IR pair shape mismatch: "
            f"left={left.shape}, right={right.shape}"
        )
    height, width = left.shape
    left_intr = calibration["infrared_left_intrinsics"]
    right_intr = calibration["infrared_right_intrinsics"]
    if (int(left_intr["width"]), int(left_intr["height"])) != (width, height):
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: left IR active profile/image mismatch"
        )
    if (int(right_intr["width"]), int(right_intr["height"])) != (width, height):
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: right IR active profile/image mismatch"
        )
    k_left, d_left = _opencv_camera(left_intr)
    k_right, d_right = _opencv_camera(right_intr)
    left_to_right = calibration["infrared_left_to_right"]
    rotation = np.asarray(left_to_right["rotation"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(
        left_to_right["translation_m"], dtype=np.float64
    ).reshape(3)
    baseline_raw_m = float(np.linalg.norm(translation))
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: non-finite stereo extrinsics"
        )
    if not 0.001 < baseline_raw_m < 0.5:
        raise FfsDepthCorrectionError(
            f"FFS depth correction invalid: implausible baseline {baseline_raw_m} m"
        )
    rectify = cv2.stereoRectify(
        k_left,
        d_left,
        k_right,
        d_right,
        (width, height),
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )
    r_left, r_right, p_left, p_right, q, roi_left, roi_right = rectify
    horizontal_term = abs(float(p_right[0, 3]))
    vertical_term = abs(float(p_right[1, 3]))
    if horizontal_term <= vertical_term or horizontal_term <= 1.0e-12:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: active stereo pair is not a "
            "horizontal-disparity rectified pair"
        )
    rectified_fx = float(p_left[0, 0])
    baseline_m = horizontal_term / abs(float(p_right[0, 0]))
    if rectified_fx <= 0.0 or not 0.001 < baseline_m < 0.5:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: invalid rectified fx/baseline"
        )
    map_lx, map_ly = cv2.initUndistortRectifyMap(
        k_left, d_left, r_left, p_left[:, :3], (width, height), cv2.CV_32FC1
    )
    map_rx, map_ry = cv2.initUndistortRectifyMap(
        k_right, d_right, r_right, p_right[:, :3], (width, height), cv2.CV_32FC1
    )
    left_rectified = cv2.remap(left, map_lx, map_ly, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right, map_rx, map_ry, cv2.INTER_LINEAR)
    source_valid = np.ones((height, width), dtype=np.uint8)
    left_valid = cv2.remap(source_valid, map_lx, map_ly, cv2.INTER_NEAREST) > 0
    right_valid = cv2.remap(source_valid, map_rx, map_ry, cv2.INTER_NEAREST) > 0
    metadata = {
        "rectification_applied": True,
        "rectification_source": "active_realsense_intrinsics_extrinsics",
        "raw_baseline_norm_m": baseline_raw_m,
        "rectified_fx_px": rectified_fx,
        "rectified_baseline_m": baseline_m,
        "left_rotation": r_left.tolist(),
        "right_rotation": r_right.tolist(),
        "left_projection": p_left.tolist(),
        "right_projection": p_right.tolist(),
        "q_matrix": q.tolist(),
        "left_valid_roi_xywh": [int(value) for value in roi_left],
        "right_valid_roi_xywh": [int(value) for value in roi_right],
    }
    return left_rectified, right_rectified, metadata, left_valid, right_valid


def disparity_to_metric_depth(
    disparity_left_px: np.ndarray,
    *,
    fx_px: float,
    baseline_m: float,
    left_rectified_valid: np.ndarray | None = None,
    right_rectified_valid: np.ndarray | None = None,
    disparity_right_px: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Convert pixel disparity to metric Z with bounds and optional LR check."""
    disparity = np.asarray(disparity_left_px, dtype=np.float32)
    if disparity.ndim != 2:
        raise FfsDepthCorrectionError(
            f"FFS depth correction invalid: disparity must be HxW: {disparity.shape}"
        )
    if not math.isfinite(fx_px) or fx_px <= 0.0:
        raise FfsDepthCorrectionError("FFS depth correction invalid: invalid fx")
    if not math.isfinite(baseline_m) or baseline_m <= 0.0:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: invalid stereo baseline"
        )
    height, width = disparity.shape
    yy, xx = np.indices(disparity.shape)
    right_x = xx.astype(np.float32) - disparity
    right_x_rounded = np.rint(right_x).astype(np.int32)
    valid = (
        np.isfinite(disparity)
        & (disparity > 0.0)
        & (right_x >= 0.0)
        & (right_x <= float(width - 1))
    )
    if left_rectified_valid is not None:
        left_valid = np.asarray(left_rectified_valid, dtype=bool)
        if left_valid.shape != disparity.shape:
            raise FfsDepthCorrectionError(
                "FFS depth correction invalid: left rectification mask mismatch"
            )
        valid &= left_valid
    if right_rectified_valid is not None:
        right_valid = np.asarray(right_rectified_valid, dtype=bool)
        if right_valid.shape != disparity.shape:
            raise FfsDepthCorrectionError(
                "FFS depth correction invalid: right rectification mask mismatch"
            )
        clipped_right_x = np.clip(right_x_rounded, 0, width - 1)
        valid &= right_valid[yy, clipped_right_x]
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    positive = np.isfinite(disparity) & (disparity > 0.0)
    depth[positive] = float(fx_px) * float(baseline_m) / disparity[positive]
    valid &= (
        np.isfinite(depth)
        & (depth >= FFS_DEPTH_MIN_M)
        & (depth <= FFS_DEPTH_MAX_M)
    )
    consistency_applied = disparity_right_px is not None
    if disparity_right_px is not None:
        right_disparity = np.asarray(disparity_right_px, dtype=np.float32)
        if right_disparity.shape != disparity.shape:
            raise FfsDepthCorrectionError(
                "FFS depth correction invalid: right disparity shape mismatch"
            )
        clipped_right_x = np.clip(right_x_rounded, 0, width - 1)
        sampled_right = right_disparity[yy, clipped_right_x]
        consistent = (
            np.isfinite(sampled_right)
            & (sampled_right > 0.0)
            & (
                np.abs(disparity - sampled_right)
                <= LEFT_RIGHT_CONSISTENCY_THRESHOLD_PX
            )
        )
        valid &= consistent
    depth[~valid] = np.nan
    metadata = {
        "formula": "Z_m = rectified_fx_px * baseline_m / disparity_px",
        "fx_px": float(fx_px),
        "baseline_m": float(baseline_m),
        "depth_range_m": [FFS_DEPTH_MIN_M, FFS_DEPTH_MAX_M],
        "left_right_consistency_applied": consistency_applied,
        "left_right_consistency_threshold_px": (
            LEFT_RIGHT_CONSISTENCY_THRESHOLD_PX
            if consistency_applied
            else None
        ),
        "left_right_consistency_status": (
            "applied_positive_right_to_left_disparity_contract"
            if consistency_applied
            else "not_implemented_runner_did_not_produce_right_disparity"
        ),
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "total_pixel_count": int(valid.size),
    }
    return depth, valid, metadata


def align_rectified_left_depth_to_rgb(
    depth_left_m: np.ndarray,
    valid_left: np.ndarray,
    *,
    rectification: dict[str, Any],
    left_to_color: dict[str, Any],
    color_intrinsics: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Project rectified-left FFS points into RGB with a nearest-depth z-buffer."""
    depth = np.asarray(depth_left_m, dtype=np.float32)
    valid = np.asarray(valid_left, dtype=bool)
    if depth.shape != valid.shape or depth.ndim != 2:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: FFS depth/valid shape mismatch"
        )
    p_left = np.asarray(rectification["left_projection"], dtype=np.float64)
    r_left = np.asarray(rectification["left_rotation"], dtype=np.float64)
    rotation_lc = np.asarray(left_to_color["rotation"], dtype=np.float64)
    translation_lc = np.asarray(
        left_to_color["translation_m"], dtype=np.float64
    )
    if p_left.shape != (3, 4) or r_left.shape != (3, 3):
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: malformed rectification matrices"
        )
    ys, xs = np.where(valid & np.isfinite(depth) & (depth > 0.0))
    if xs.size == 0:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: no valid rectified-left depth"
        )
    z = depth[ys, xs].astype(np.float64)
    fx = float(p_left[0, 0])
    fy = float(p_left[1, 1])
    cx = float(p_left[0, 2])
    cy = float(p_left[1, 2])
    points_rectified = np.column_stack(
        ((xs - cx) / fx * z, (ys - cy) / fy * z, z)
    )
    points_left = points_rectified @ r_left
    points_color = points_left @ rotation_lc.T + translation_lc
    z_color = points_color[:, 2]
    finite_front = np.isfinite(points_color).all(axis=1) & (z_color > 0.0)
    points_color = points_color[finite_front]
    z_color = z_color[finite_front]
    coefficients = np.asarray(color_intrinsics.get("coeffs", []), dtype=np.float64)
    model = str(color_intrinsics.get("model", "")).lower().split(".")[-1]
    if coefficients.size == 0:
        coefficients = np.zeros(5, dtype=np.float64)
    nonzero_distortion = bool(np.max(np.abs(coefficients)) > 1.0e-12)
    if nonzero_distortion and model not in {
        "brown_conrady",
        "modified_brown_conrady",
    }:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: RGB projection is unsupported for "
            f"active distortion model {model!r}"
        )
    camera_matrix, _ = _opencv_camera(color_intrinsics)
    if nonzero_distortion:
        projected, _ = cv2.projectPoints(
            points_color,
            np.zeros(3),
            np.zeros(3),
            camera_matrix,
            coefficients,
        )
        uv = projected.reshape(-1, 2)
    else:
        uv = np.column_stack(
            (
                float(color_intrinsics["fx"]) * points_color[:, 0] / z_color
                + float(color_intrinsics["ppx"]),
                float(color_intrinsics["fy"]) * points_color[:, 1] / z_color
                + float(color_intrinsics["ppy"]),
            )
        )
    rgb_width = int(color_intrinsics["width"])
    rgb_height = int(color_intrinsics["height"])
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    inside = (
        np.isfinite(uv).all(axis=1)
        & (u >= 0)
        & (u < rgb_width)
        & (v >= 0)
        & (v < rgb_height)
    )
    u = u[inside]
    v = v[inside]
    z_color = z_color[inside]
    flat = np.full(rgb_width * rgb_height, np.inf, dtype=np.float32)
    np.minimum.at(flat, v * rgb_width + u, z_color.astype(np.float32))
    aligned = flat.reshape(rgb_height, rgb_width)
    valid_rgb = np.isfinite(aligned)
    aligned[~valid_rgb] = np.nan
    metadata = {
        "method": (
            "rectified_left_3d_to_original_left_via_R1_transpose_then_"
            "active_left_to_color_extrinsics_then_rgb_projection_z_buffer"
        ),
        "input_valid_count": int(xs.size),
        "projected_inside_rgb_count": int(np.count_nonzero(inside)),
        "aligned_valid_rgb_pixel_count": int(np.count_nonzero(valid_rgb)),
        "rgb_shape": [rgb_height, rgb_width],
    }
    return aligned, valid_rgb, metadata


def _mad(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def estimate_depth_correction_per_book(
    *,
    book_spine_masks: np.ndarray,
    held_book_indices: set[int],
    rs_depth_m: np.ndarray,
    ffs_depth_rgb_m: np.ndarray,
    ffs_valid_rgb: np.ndarray,
) -> tuple[float, list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Median pixels per instance, then median accepted book deltas."""
    masks = np.asarray(book_spine_masks, dtype=bool)
    rs_depth = np.asarray(rs_depth_m, dtype=np.float32)
    ffs_depth = np.asarray(ffs_depth_rgb_m, dtype=np.float32)
    ffs_valid = np.asarray(ffs_valid_rgb, dtype=bool)
    if masks.ndim != 3 or masks.shape[1:] != rs_depth.shape:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: book mask/RS depth shape mismatch"
        )
    if ffs_depth.shape != rs_depth.shape or ffs_valid.shape != rs_depth.shape:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: RGB-aligned FFS shape mismatch"
        )
    held = {int(index) for index in held_book_indices}
    if not held:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: held book was not identified; "
            "safe non-held exclusion cannot be guaranteed"
        )
    non_held_indices = [index for index in range(masks.shape[0]) if index not in held]
    if not non_held_indices:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: no non-held book spine instance"
        )
    rs_valid = (
        np.isfinite(rs_depth)
        & (rs_depth >= FFS_DEPTH_MIN_M)
        & (rs_depth <= FFS_DEPTH_MAX_M)
    )
    kernel_size = 2 * BOOK_SPINE_ERODE_PX + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    records: list[dict[str, Any]] = []
    accepted_deltas: list[float] = []
    non_held_union = np.zeros(rs_depth.shape, dtype=bool)
    common_union = np.zeros(rs_depth.shape, dtype=bool)
    held_union = np.any(masks[sorted(held)], axis=0)
    for index in non_held_indices:
        original_mask = masks[index] & ~held_union
        non_held_union |= original_mask
        eroded = cv2.erode(
            original_mask.astype(np.uint8), kernel, iterations=1
        ).astype(bool)
        common = eroded & rs_valid & ffs_valid & np.isfinite(ffs_depth)
        common_union |= common
        mask_count = int(np.count_nonzero(eroded))
        common_count = int(np.count_nonzero(common))
        common_ratio = float(common_count / mask_count) if mask_count else 0.0
        rejection_reasons: list[str] = []
        if mask_count == 0:
            rejection_reasons.append("empty_after_erosion")
        if common_count < MIN_COMMON_VALID_PIXELS:
            rejection_reasons.append(
                f"common_pixels_below_{MIN_COMMON_VALID_PIXELS}"
            )
        if common_ratio < MIN_COMMON_VALID_RATIO:
            rejection_reasons.append(
                f"common_ratio_below_{MIN_COMMON_VALID_RATIO:.3f}"
            )
        accepted = not rejection_reasons
        record: dict[str, Any] = {
            "book_index": index,
            "held_book": False,
            "original_mask_pixel_count": int(np.count_nonzero(original_mask)),
            "mask_pixel_count": mask_count,
            "common_valid_pixel_count": common_count,
            "common_valid_ratio": common_ratio,
            "accepted": accepted,
            "rejection_reason": ";".join(rejection_reasons) or None,
        }
        if common_count:
            rs_values = rs_depth[common].astype(np.float64)
            ffs_values = ffs_depth[common].astype(np.float64)
            differences = ffs_values - rs_values
            delta = float(np.median(differences))
            record.update(
                {
                    "rs_median_m": float(np.median(rs_values)),
                    "ffs_median_m": float(np.median(ffs_values)),
                    "median_ffs_minus_rs_m": delta,
                    "rs_std_m": float(np.std(rs_values)),
                    "rs_mad_m": _mad(rs_values),
                    "ffs_std_m": float(np.std(ffs_values)),
                    "ffs_mad_m": _mad(ffs_values),
                }
            )
            if accepted:
                accepted_deltas.append(delta)
        records.append(record)
    if not accepted_deltas:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: no accepted non-held book spine "
            "common valid region"
        )
    correction = float(np.median(np.asarray(accepted_deltas, dtype=np.float64)))
    if not math.isfinite(correction):
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: aggregate correction is non-finite"
        )
    return correction, records, non_held_union, common_union


def corrected_target_on_rgb_ray(
    target_pixel: Any,
    corrected_depth_m: float,
    color_intrinsics: dict[str, Any],
) -> np.ndarray:
    """Regenerate XYZ from target pixel and corrected Z, not by replacing Z."""
    pixel = np.asarray(target_pixel, dtype=np.float64).reshape(2)
    coeffs = np.asarray(color_intrinsics.get("coeffs", []), dtype=np.float64)
    if coeffs.size and np.max(np.abs(coeffs)) > 1.0e-9:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: explicit pinhole target-ray formula "
            "requires zero-distortion aligned RGB intrinsics"
        )
    if not math.isfinite(corrected_depth_m) or corrected_depth_m <= 0.0:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: corrected target depth is invalid"
        )
    fx = float(color_intrinsics["fx"])
    fy = float(color_intrinsics["fy"])
    ppx = float(color_intrinsics["ppx"])
    ppy = float(color_intrinsics["ppy"])
    z = float(corrected_depth_m)
    return np.asarray(
        [(pixel[0] - ppx) / fx * z, (pixel[1] - ppy) / fy * z, z],
        dtype=np.float32,
    )


def _run_ffs(
    config: FfsRuntimeConfig,
    *,
    left_path: Path,
    right_path: Path,
    output_disparity: Path,
    output_right_disparity: Path,
    shot_dir: Path,
) -> dict[str, Any]:
    replacements = {
        "left": str(left_path),
        "right": str(right_path),
        "output_disparity": str(output_disparity),
        "output_right_disparity": str(output_right_disparity),
        "weights": str(config.weights_path),
        "shot_dir": str(shot_dir),
    }
    try:
        command = [argument.format(**replacements) for argument in config.command_template]
    except KeyError as exc:
        raise FfsDepthCorrectionError(
            f"FFS depth correction invalid: unknown command placeholder {exc}"
        ) from exc
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=None if config.work_dir is None else str(config.work_dir),
            check=False,
            capture_output=True,
            text=True,
            timeout=config.timeout_sec,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FfsDepthCorrectionError(
            f"FFS depth correction invalid: inference launch failed: {exc}"
        ) from exc
    elapsed = time.perf_counter() - started
    metadata = {
        "integration": "isolated_subprocess",
        "runner_command": command,
        "work_dir": None if config.work_dir is None else str(config.work_dir),
        "model_weights": str(config.weights_path),
        "model_weights_size_bytes": config.weights_path.stat().st_size,
        "elapsed_seconds": elapsed,
        "return_code": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "left_right_output_requested": str(output_right_disparity),
    }
    _write_json(shot_dir / "ffs_inference.json", metadata)
    if completed.returncode != 0:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: inference returned "
            f"{completed.returncode}; see {shot_dir / 'ffs_inference.json'}"
        )
    if not output_disparity.is_file():
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: runner did not create disparity: "
            f"{output_disparity}"
        )
    return metadata


def _capture_rgbd_ir_once(*, width: int, height: int, fps: int) -> StereoCapture:
    """Capture RGB, aligned RS depth, and both IR images from one frameset."""
    from detection.pro_handbook.sam_py_demo import Storage_SAM3 as base

    rs = base._get_realsense_module()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
    try:
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        for _ in range(30):
            pipeline.wait_for_frames()
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        infrared_left_frame = frames.get_infrared_frame(1)
        infrared_right_frame = frames.get_infrared_frame(2)
        if not all(
            (color_frame, depth_frame, infrared_left_frame, infrared_right_frame)
        ):
            raise FfsDepthCorrectionError(
                "FFS depth correction invalid: incomplete RGB/depth/IR frameset"
            )
        aligned = align.process(frames)
        aligned_color = aligned.get_color_frame()
        aligned_depth = aligned.get_depth_frame()
        if not aligned_color or not aligned_depth:
            raise FfsDepthCorrectionError(
                "FFS depth correction invalid: RGB depth alignment failed"
            )
        filtered_depth = base.legacy_geometry.depth_filter_like_viewer(aligned_depth)
        rgb_bgr = np.asanyarray(aligned_color.get_data()).copy()
        depth_u16 = np.asanyarray(filtered_depth.get_data()).copy()
        infrared_left = np.asanyarray(infrared_left_frame.get_data()).copy()
        infrared_right = np.asanyarray(infrared_right_frame.get_data()).copy()

        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        left_profile = rs.video_stream_profile(
            profile.get_stream(rs.stream.infrared, 1)
        )
        right_profile = rs.video_stream_profile(
            profile.get_stream(rs.stream.infrared, 2)
        )
        color_intr = color_profile.get_intrinsics()
        calibration = {
            "source": "active_realsense_stream_profile",
            "same_frameset": True,
            "frame_numbers": {
                "color": int(color_frame.get_frame_number()),
                "depth": int(depth_frame.get_frame_number()),
                "infrared_left": int(infrared_left_frame.get_frame_number()),
                "infrared_right": int(infrared_right_frame.get_frame_number()),
            },
            "timestamps_ms": {
                "color": float(color_frame.get_timestamp()),
                "depth": float(depth_frame.get_timestamp()),
                "infrared_left": float(infrared_left_frame.get_timestamp()),
                "infrared_right": float(infrared_right_frame.get_timestamp()),
            },
            "color_intrinsics": _intrinsics_dict(color_intr),
            "depth_intrinsics": _intrinsics_dict(depth_profile.get_intrinsics()),
            "infrared_left_intrinsics": _intrinsics_dict(left_profile.get_intrinsics()),
            "infrared_right_intrinsics": _intrinsics_dict(right_profile.get_intrinsics()),
            "infrared_left_to_right": _extrinsics_dict(
                left_profile.get_extrinsics_to(right_profile)
            ),
            "infrared_left_to_color": _extrinsics_dict(
                left_profile.get_extrinsics_to(color_profile)
            ),
            "depth_to_color": _extrinsics_dict(
                depth_profile.get_extrinsics_to(color_profile)
            ),
        }
        depth_scale = float(
            profile.get_device().first_depth_sensor().get_depth_scale()
        )
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
    return StereoCapture(
        rgb_bgr=rgb_bgr,
        depth_u16_aligned_rgb=depth_u16,
        color_intrinsics_object=color_intr,
        depth_scale=depth_scale,
        infrared_left=infrared_left,
        infrared_right=infrared_right,
        calibration=calibration,
    )


def save_stereo_capture_inputs(shot_dir: Path, capture: StereoCapture) -> None:
    """Persist same-frameset inputs before callbacks or recognition can fail."""
    _save_png(shot_dir / "infrared_left.png", capture.infrared_left)
    _save_png(shot_dir / "infrared_right.png", capture.infrared_right)
    np.save(
        shot_dir / "depth.npy",
        capture.depth_u16_aligned_rgb,
        allow_pickle=False,
    )
    _save_png(shot_dir / "depth.png", capture.depth_u16_aligned_rgb)
    calibration = dict(capture.calibration)
    calibration["depth_scale"] = float(capture.depth_scale)
    calibration["rectification"] = {
        "status": "pending_until_ffs_processing",
        "raw_ir_assumed_rectified": False,
    }
    _write_json(shot_dir / "stereo_calibration.json", calibration)


def apply_ffs_depth_correction(
    *,
    shot_dir: Path,
    capture: StereoCapture,
    first_target_cam_rs: np.ndarray,
    result_metadata: dict[str, Any],
    runtime_config: FfsRuntimeConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run FFS, robustly aggregate book deltas, and rebuild target camera XYZ."""
    failure_path = shot_dir / "ffs_depth_correction.json"
    try:
        raw_left_path = shot_dir / "infrared_left.png"
        raw_right_path = shot_dir / "infrared_right.png"
        rect_left_path = shot_dir / "infrared_left_rectified.png"
        rect_right_path = shot_dir / "infrared_right_rectified.png"
        disparity_path = shot_dir / "ffs_disparity.npy"
        right_disparity_path = shot_dir / "ffs_disparity_right.npy"
        left_rect, right_rect, rectification, left_valid, right_valid = (
            rectify_stereo_pair(
                capture.infrared_left,
                capture.infrared_right,
                capture.calibration,
            )
        )
        calibration = dict(capture.calibration)
        calibration["rectification"] = rectification
        calibration["metric_depth_formula"] = (
            "Z_m = rectified_fx_px * rectified_baseline_m / disparity_px"
        )
        _write_json(shot_dir / "stereo_calibration.json", calibration)
        _save_png(rect_left_path, left_rect)
        _save_png(rect_right_path, right_rect)

        inference_metadata = _run_ffs(
            runtime_config,
            left_path=rect_left_path,
            right_path=rect_right_path,
            output_disparity=disparity_path,
            output_right_disparity=right_disparity_path,
            shot_dir=shot_dir,
        )
        disparity_left = np.load(disparity_path, allow_pickle=False)
        if disparity_left.shape != left_rect.shape:
            raise FfsDepthCorrectionError(
                "FFS depth correction invalid: FFS disparity/IR shape mismatch: "
                f"{disparity_left.shape}!={left_rect.shape}"
            )
        disparity_right = (
            np.load(right_disparity_path, allow_pickle=False)
            if right_disparity_path.is_file()
            else None
        )
        ffs_depth_left, ffs_valid_left, disparity_metadata = (
            disparity_to_metric_depth(
                disparity_left,
                fx_px=float(rectification["rectified_fx_px"]),
                baseline_m=float(rectification["rectified_baseline_m"]),
                left_rectified_valid=left_valid,
                right_rectified_valid=right_valid,
                disparity_right_px=disparity_right,
            )
        )
        print(
            "[FFS OCCLUSION] "
            f"{disparity_metadata['left_right_consistency_status']}"
        )
        np.save(shot_dir / "ffs_depth_left.npy", ffs_depth_left, allow_pickle=False)
        aligned_ffs, valid_rgb, alignment_metadata = (
            align_rectified_left_depth_to_rgb(
                ffs_depth_left,
                ffs_valid_left,
                rectification=rectification,
                left_to_color=calibration["infrared_left_to_color"],
                color_intrinsics=calibration["color_intrinsics"],
            )
        )
        np.save(shot_dir / "ffs_depth_aligned_rgb.npy", aligned_ffs, allow_pickle=False)
        _save_png(
            shot_dir / "ffs_valid_mask_rgb.png",
            valid_rgb.astype(np.uint8) * 255,
        )

        import offline_100test_storage_space_sam3 as storage_recognition

        spine_instances = storage_recognition.load_npz(
            shot_dir / storage_recognition.SPINE_MODEL.npz_name
        )
        held_indices = {
            int(index)
            for index in result_metadata.get("held_book_spine_indices", [])
        }
        rs_depth_m = (
            capture.depth_u16_aligned_rgb.astype(np.float32)
            * float(capture.depth_scale)
        )
        correction_m, book_records, non_held_union, common_union = (
            estimate_depth_correction_per_book(
                book_spine_masks=spine_instances.masks,
                held_book_indices=held_indices,
                rs_depth_m=rs_depth_m,
                ffs_depth_rgb_m=aligned_ffs,
                ffs_valid_rgb=valid_rgb,
            )
        )
        _save_png(
            shot_dir / "non_held_book_spine_mask.png",
            non_held_union.astype(np.uint8) * 255,
        )
        _save_png(
            shot_dir / "common_valid_mask.png",
            common_union.astype(np.uint8) * 255,
        )
        depth_ref_rs_m = float(result_metadata.get("depth_ref_m"))
        corrected_depth_m = depth_ref_rs_m + correction_m
        if not (
            math.isfinite(corrected_depth_m)
            and CORRECTED_DEPTH_MIN_M
            <= corrected_depth_m
            <= CORRECTED_DEPTH_MAX_M
        ):
            raise FfsDepthCorrectionError(
                "FFS depth correction invalid: corrected depth outside "
                f"[{CORRECTED_DEPTH_MIN_M}, {CORRECTED_DEPTH_MAX_M}] m: "
                f"{corrected_depth_m}"
            )
        target_pixel = result_metadata.get("first_target_px_selected")
        corrected_target = corrected_target_on_rgb_ray(
            target_pixel,
            corrected_depth_m,
            calibration["color_intrinsics"],
        )
        correction_metadata = {
            "status": "valid",
            "depth_ref_rs_m": depth_ref_rs_m,
            "depth_correction_ffs_m": correction_m,
            "depth_ref_corrected_m": corrected_depth_m,
            "number_of_valid_books": int(
                sum(bool(record["accepted"]) for record in book_records)
            ),
            "book_spine_erode_px": BOOK_SPINE_ERODE_PX,
            "minimum_common_valid_pixels": MIN_COMMON_VALID_PIXELS,
            "minimum_common_valid_ratio": MIN_COMMON_VALID_RATIO,
            "corrected_depth_valid_range_m": [
                CORRECTED_DEPTH_MIN_M,
                CORRECTED_DEPTH_MAX_M,
            ],
            "held_book_spine_indices": sorted(held_indices),
            "books": book_records,
            "target_pixel_uv": [float(value) for value in target_pixel],
            "old_target_camera_xyz_m": np.asarray(
                first_target_cam_rs, dtype=np.float64
            ).tolist(),
            "corrected_target_camera_xyz_m": corrected_target.tolist(),
            "camera_target_depth_change_mm": correction_m * 1000.0,
            "ffs_model_inference": inference_metadata,
            "stereo_fx_px": rectification["rectified_fx_px"],
            "baseline_m": rectification["rectified_baseline_m"],
            "disparity_validation": disparity_metadata,
            "alignment": alignment_metadata,
        }
        _write_json(failure_path, correction_metadata)
        print("\n========== FFS DEPTH CORRECTION ==========")
        print(f"depth_ref_rs = {depth_ref_rs_m:.6f} m")
        for record in book_records:
            print(f"book {record['book_index']}")
            print(f"  mask pixels       = {record['mask_pixel_count']}")
            print(f"  common pixels     = {record['common_valid_pixel_count']}")
            print(f"  common ratio      = {record['common_valid_ratio']:.6f}")
            if record["common_valid_pixel_count"]:
                print(f"  RS median         = {record['rs_median_m']:.6f} m")
                print(f"  FFS median        = {record['ffs_median_m']:.6f} m")
                print(
                    "  FFS - RS median   = "
                    f"{record['median_ffs_minus_rs_m'] * 1000.0:+.3f} mm"
                )
            print(f"  accepted          = {record['accepted']}")
            if record["rejection_reason"]:
                print(f"  rejection reason  = {record['rejection_reason']}")
        print(f"valid books = {correction_metadata['number_of_valid_books']}")
        print(f"median delta = {correction_m * 1000.0:+.3f} mm")
        print(f"RS depth        = {depth_ref_rs_m:.6f} m")
        print(f"FFS correction  = {correction_m:+.6f} m")
        print(f"corrected depth = {corrected_depth_m:.6f} m")
        print(f"target pixel = {target_pixel}")
        print(f"old target camera XYZ = {np.asarray(first_target_cam_rs).tolist()}")
        print(f"corrected target camera XYZ = {corrected_target.tolist()}")
        print(f"camera target depth change = {correction_m * 1000.0:+.3f} mm")
        print("robot target trace is emitted after camera-to-robot conversion")
        print("==========================================")

        result_metadata.update(
            {
                "ffs_depth_correction_valid": True,
                "depth_ref_rs_m": depth_ref_rs_m,
                "depth_correction_ffs_m": correction_m,
                "depth_ref_corrected_m": corrected_depth_m,
                "first_target_cam_rs": np.asarray(first_target_cam_rs).copy(),
                "first_target_cam": corrected_target.copy(),
                "final_target_cam": corrected_target.copy(),
                "target_depth_m": corrected_depth_m,
                "target_3d_source": "ffs_corrected_depth_ref_on_rgb_target_ray",
                "ffs_depth_correction_path": str(failure_path),
            }
        )
        geometry_path = shot_dir / "geometry_metadata.json"
        if geometry_path.is_file():
            geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
            geometry.update(
                {
                    "ffs_depth_correction_valid": True,
                    "depth_ref_rs_m": depth_ref_rs_m,
                    "depth_correction_ffs_m": correction_m,
                    "depth_ref_corrected_m": corrected_depth_m,
                    "first_target_cam_rs": np.asarray(
                        first_target_cam_rs
                    ).tolist(),
                    "first_target_cam": corrected_target.tolist(),
                    "final_target_cam": corrected_target.tolist(),
                    "target_depth_m": corrected_depth_m,
                    "target_3d_source": (
                        "ffs_corrected_depth_ref_on_rgb_target_ray"
                    ),
                    "ffs_depth_correction_path": str(failure_path),
                }
            )
            _write_json(geometry_path, geometry)
        return corrected_target, correction_metadata
    except Exception as exc:
        failure = {
            "status": "invalid",
            "error": str(exc),
            "real_sense_only_fallback": False,
            "storage_reaching_allowed": False,
        }
        try:
            _write_json(failure_path, failure)
        except Exception:
            pass
        if isinstance(exc, FfsDepthCorrectionError):
            raise
        raise FfsDepthCorrectionError(
            f"FFS depth correction invalid: {exc}"
        ) from exc


def record_ffs_robot_target_trace(
    result_metadata: dict[str, Any],
    *,
    camera_target_rs_m: Any,
    camera_target_corrected_m: Any,
    robot_target_rs_mm: Any,
    robot_target_corrected_mm: Any,
) -> None:
    """Persist proof that corrected camera depth reached robot coordinates."""
    if not result_metadata.get("ffs_depth_correction_valid"):
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: robot trace requested without "
            "a valid FFS correction"
        )
    robot_rs = np.asarray(robot_target_rs_mm, dtype=np.float64).reshape(3)
    robot_corrected = np.asarray(
        robot_target_corrected_mm, dtype=np.float64
    ).reshape(3)
    camera_rs = np.asarray(camera_target_rs_m, dtype=np.float64).reshape(3)
    camera_corrected = np.asarray(
        camera_target_corrected_m, dtype=np.float64
    ).reshape(3)
    robot_change = robot_corrected - robot_rs
    print("\n[FFS ROBOT TARGET TRACE]")
    print(f"robot target XYZ before correction = {robot_rs.tolist()} mm")
    print(f"robot target XYZ after correction  = {robot_corrected.tolist()} mm")
    print(f"robot X change = {robot_change[0]:+.3f} mm")
    correction_path = Path(result_metadata["ffs_depth_correction_path"])
    metadata = json.loads(correction_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "motion_target_camera_xyz_rs_m": camera_rs.tolist(),
            "motion_target_camera_xyz_corrected_m": camera_corrected.tolist(),
            "robot_target_xyz_before_correction_mm": robot_rs.tolist(),
            "robot_target_xyz_after_correction_mm": robot_corrected.tolist(),
            "robot_target_change_mm": robot_change.tolist(),
            "robot_x_change_mm": float(robot_change[0]),
        }
    )
    _write_json(correction_path, metadata)
    result_metadata.update(
        {
            "robot_target_xyz_before_ffs_mm": robot_rs,
            "robot_target_xyz_after_ffs_mm": robot_corrected,
            "robot_target_change_ffs_mm": robot_change,
            "robot_x_change_ffs_mm": float(robot_change[0]),
        }
    )


def run_capture_and_pca_depth_space(
    out_dir: str | Path = "captures_depth_space",
    width: int = 1280,
    height: int = 720,
    fps: int = 6,
    *,
    rotate_180: bool = False,
    show_debug_images: bool = True,
    after_capture_callback: Optional[Callable[[], None]] = None,
    sam3_runtime: Any = None,
    apply_legacy_depth_ignore: bool = False,
    held_book_detection_enabled: bool = True,
    held_book_min_height_ratio: float = 0.90,
    held_book_max_top_margin_ratio: float = 0.02,
    held_book_max_bottom_margin_ratio: float = 0.10,
    target_3d_mode: str = "depth_ref",
    **geometry_kwargs: Any,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """FFS capture/recognition wrapper with no RealSense-only fallback."""
    from detection.pro_handbook.sam_py_demo import Storage_SAM3 as base

    runtime_config = load_ffs_runtime_config()
    if rotate_180:
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: rotate_180 calibration transform "
            "is intentionally unsupported"
        )
    if target_3d_mode != "depth_ref":
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: FFS experiment requires "
            "target_3d_mode=depth_ref"
        )
    shot_dir = Path(out_dir) / time.strftime("%Y%m%d_%H%M%S")
    shot_dir.mkdir(parents=True, exist_ok=False)
    capture = _capture_rgbd_ir_once(width=width, height=height, fps=fps)
    save_stereo_capture_inputs(shot_dir, capture)
    if after_capture_callback is not None:
        after_capture_callback()
    rgb = cv2.cvtColor(capture.rgb_bgr, cv2.COLOR_BGR2RGB)
    held_config = base.HeldBookDetectionConfig(
        enabled=bool(held_book_detection_enabled),
        min_height_ratio=float(held_book_min_height_ratio),
        max_top_margin_ratio=float(held_book_max_top_margin_ratio),
        max_bottom_margin_ratio=float(held_book_max_bottom_margin_ratio),
    )
    angle_rad, first_target_cam_rs, result = base.run_from_rgbd(
        rgb,
        capture.depth_u16_aligned_rgb,
        capture.color_intrinsics_object,
        capture.depth_scale,
        shot_dir=shot_dir,
        runtime=sam3_runtime,
        apply_legacy_depth_ignore=apply_legacy_depth_ignore,
        held_book_config=held_config,
        target_3d_mode="depth_ref",
        **geometry_kwargs,
    )
    corrected_target, _ = apply_ffs_depth_correction(
        shot_dir=shot_dir,
        capture=capture,
        first_target_cam_rs=first_target_cam_rs,
        result_metadata=result,
        runtime_config=runtime_config,
    )
    if not result.get("ffs_depth_correction_valid"):
        raise FfsDepthCorrectionError(
            "FFS depth correction invalid: correction did not reach result metadata"
        )
    if show_debug_images:
        base.legacy_geometry.show_image_from_path(
            "final_space_target_point",
            result["final_space_target_point_path"],
            wait=True,
        )
        cv2.destroyAllWindows()
    return angle_rad, corrected_target, result
