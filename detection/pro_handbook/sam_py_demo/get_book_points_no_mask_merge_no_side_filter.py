#!/usr/bin/env python3
"""Offline comparison flow without legacy mask merge or side-surface filtering.

The production ``get_book_points.py`` is intentionally not modified.  This module
reuses its OCR matching, median-depth filter, camera conversion, normal plane
RANSAC helper, PCA, width, target-point, and output writers.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image, ImageDraw, ImageFont

from . import get_book_points as current
from .modules.book_width import estimate_book_width
from .modules.grip_point import find_target_point
from .modules.pca_vector import pca_axes_fix_dir
from .modules.pointcloud_utils import save_ply_ascii


VARIANT = "no_mask_merge_no_side_filter"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CAPTURES_DIR = PROJECT_ROOT / "captures"
SOURCE_SHOT_DIR = CAPTURES_DIR / "100test" / "1"
MASTER_JSON = PROJECT_ROOT / "master_20260216.json"
CURRENT_RESULT = CAPTURES_DIR / "20260723_154307" / "offline_recognition_result.json"
REQUIRED_INPUTS = ("after_init_rgb.png", "after_init_depth.npy")


class TargetMaskSelectionError(RuntimeError):
    """Raised when OCR cannot identify the requested book with enough confidence."""


class Tee:
    def __init__(self, *streams):
        self.streams = streams
        self.parts: list[str] = []

    def write(self, value: str):
        self.parts.append(value)
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    @property
    def text(self) -> str:
        return "".join(self.parts)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_run_dir(started: datetime) -> Path:
    stem = started.strftime("20260723_%H%M%S_no_mask_merge_no_side_filter")
    candidate = CAPTURES_DIR / stem
    suffix = 1
    while candidate.exists():
        candidate = CAPTURES_DIR / f"{stem}_{suffix:03d}"
        suffix += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate.resolve()


def _unique_live_run_dir(started: datetime) -> Path:
    stem = started.strftime("%Y%m%d_%H%M%S_live_no_mask_merge_no_side_filter")
    candidate = CAPTURES_DIR / stem
    suffix = 1
    while candidate.exists():
        candidate = CAPTURES_DIR / f"{stem}_{suffix:03d}"
        suffix += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate.resolve()


def _intrinsics():
    intr = rs.intrinsics()
    intr.width = 1280
    intr.height = 720
    intr.fx = 908.1617431640625
    intr.fy = 906.4829711914062
    intr.ppx = 637.79833984375
    intr.ppy = 371.0213928222656
    return intr, 0.0010000000474974513


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if not len(xs):
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _font(size: int):
    for path in (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _save_selected_mask_outputs(
    shot_dir: Path,
    rgb_pil: Image.Image,
    mask: np.ndarray,
    query: str,
    selected_index: int,
    score: float,
) -> None:
    mask = np.asarray(mask, dtype=bool)
    binary = mask.astype(np.uint8) * 255
    # Comparison invariant: these two files must be identical.
    Image.fromarray(binary, mode="L").save(
        shot_dir / "selected_mask_before_legacy_postprocess.png"
    )
    Image.fromarray(binary, mode="L").save(
        shot_dir / "selected_mask_used_for_depth.png"
    )
    Image.fromarray(binary, mode="L").save(shot_dir / "target_book_mask.png")

    rgb = np.asarray(rgb_pil.convert("RGB"))
    only = rgb.copy()
    only[~mask] = 0
    Image.fromarray(only, mode="RGB").save(shot_dir / "target_book_only.png")

    box = _bbox(mask)
    overlay = rgb.copy()
    overlay[mask] = (
        0.55 * overlay[mask] + 0.45 * np.asarray([255, 48, 48])
    ).astype(np.uint8)
    image = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = _font(22)
    text = (
        f"{query}\nmask index: {selected_index} (1-based)  score: {score:.6f}\n"
        f"area: {int(mask.sum())} px  bbox XYXY: {box}"
    )
    tb = draw.multiline_textbbox((12, 10), text, font=font, spacing=5)
    draw.rectangle((tb[0] - 6, tb[1] - 5, tb[2] + 6, tb[3] + 5), fill="black")
    draw.multiline_text((12, 10), text, font=font, fill="white", spacing=5)
    draw.rectangle(tuple(box), outline=(255, 255, 0), width=3)
    image.save(shot_dir / "target_book_mask_overlay.png")


def _selected_ocr_confidence(ocr_result: dict, selected_text: str | None):
    if not selected_text:
        return None
    for text, score in zip(
        ocr_result.get("rec_texts") or [], ocr_result.get("rec_scores") or []
    ):
        if str(text).strip() == str(selected_text).strip():
            return float(score)
    return None


def run_capture_and_pca_offline_no_mask_merge_no_side_filter(
    query: str,
    shot_dir: str | Path,
    sam_device: str = "gpu",
    depth_merge_tolerance_raw: int = 30,
    *,
    intr=None,
    depth_scale: float | None = None,
) -> dict:
    """Run the isolated comparison pipeline on already-copied RGB-D inputs."""
    shot_dir = Path(shot_dir).expanduser().resolve()
    color_np = cv2.imread(str(shot_dir / "after_init_rgb.png"), cv2.IMREAD_COLOR)
    depth_np = np.load(shot_dir / "after_init_depth.npy", allow_pickle=False)
    if color_np is None:
        raise FileNotFoundError(shot_dir / "after_init_rgb.png")
    if depth_np.shape != color_np.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch: {color_np.shape} / {depth_np.shape}")
    if intr is None or depth_scale is None:
        intr, depth_scale = _intrinsics()
    current._save_camera_params_json(shot_dir, intr, depth_scale)

    timings: dict[str, float] = {}
    ocr_start = time.perf_counter()
    ocr_proc = current.start_ocr_subprocess(shot_dir)
    print("[PARALLEL][COMPARISON] OCR subprocess started")

    sam_start = time.perf_counter()
    runner = current._get_sam_runner_compat(
        encoder_path="unused-by-sam3",
        decoder_path="unused-by-sam3",
        sam_device=sam_device,
        use_cache=True,
    )
    rgb_pil = Image.fromarray(cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))
    stage_cfg = current._make_stage_save_cfg_compat(shot_dir)
    masks, sam_data = current._infer_masks_compat(
        runner,
        rgb_pil,
        stage_cfg,
        depth_np_u16=depth_np,
        depth_merge_tolerance_raw=depth_merge_tolerance_raw,
    )
    timings["sam3_offline_call_seconds"] = time.perf_counter() - sam_start

    ocr_stdout = current.wait_ocr_subprocess(ocr_proc, timeout=120.0)
    if ocr_stdout.strip():
        print(ocr_stdout, end="" if ocr_stdout.endswith("\n") else "\n")
    timings["ocr_wall_seconds"] = time.perf_counter() - ocr_start

    selection_start = time.perf_counter()
    merged = current.merge_ocr_and_masks(
        query=query, masks=masks, shot_dir=shot_dir, interactive=False, threshold=40
    )
    timings["ocr_mask_selection_seconds"] = time.perf_counter() - selection_start
    if not merged.get("results"):
        raise TargetMaskSelectionError(
            "target OCR similarity did not exceed threshold 40; "
            "the non-interactive mask-1 fallback is not accepted"
        )
    selected_index = int(merged["sel_idx"])
    selected_mask = np.asarray(merged["mask01"], dtype=bool)
    score = float(sam_data[selected_index - 1]["score"])
    _save_selected_mask_outputs(
        shot_dir, rgb_pil, selected_mask, query, selected_index, score
    )

    # Comparison version:
    # legacy quality_filter / box-NMS / mask de-dup / merge_coaxial_rect_masks
    # and all later mask shape completion/refinement are intentionally skipped.
    mask_used_for_depth = selected_mask.copy()
    masks_equal = bool(np.array_equal(selected_mask, mask_used_for_depth))

    # OCR geometry is used only as the existing median-depth reference.
    _, depth_anchor_info = current.refine_mask_by_ocr_axis_band(
        mask01=mask_used_for_depth.astype(np.uint8),
        merged=merged,
        image_shape=color_np.shape[:2],
        shot_dir=shot_dir,
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
    reference_mask, reference_info = current._make_depth_reference_mask_from_ocr_info(
        depth_anchor_info,
        color_np.shape[:2],
        selected_mask01=mask_used_for_depth.astype(np.uint8),
        min_intersection_px=30,
    )

    depth_start = time.perf_counter()
    before_median_count = int(
        np.count_nonzero(mask_used_for_depth & (depth_np > 0))
    )
    depth_median, depth_info = current.save_masked_and_cropped(
        color_np,
        depth_np,
        mask_used_for_depth.astype(np.uint8),
        shot_dir,
        "comparison_median_depth_filter",
        z_tolerance_raw=int(depth_merge_tolerance_raw),
        depth_reference_mask01=reference_mask,
        depth_reference_name="selected_ocr_polygon_intersection_selected_mask",
        return_info=True,
    )
    mask_after_median = mask_used_for_depth & (depth_median > 0)
    points_median, uv_median = current._mask_depth_to_points_uv_for_plane_filter(
        mask_after_median.astype(np.uint8), depth_median, intr, depth_scale
    )
    colors_rgb = color_np[uv_median[:, 1], uv_median[:, 0], ::-1].astype(np.uint8)
    save_ply_ascii(
        shot_dir / "pointcloud_after_median_depth_filter.ply",
        points_median,
        colors_rgb,
    )
    timings["median_depth_filter_seconds"] = time.perf_counter() - depth_start

    # Comparison version:
    # retain one normal plane RANSAC; do not call column/side filters,
    # post-RANSAC plane-a95, residual policies, reject-release, or completion.
    ransac_start = time.perf_counter()
    plane, inlier_mask, ransac_info = current._fit_plane_ransac_open3d_for_spine(
        points_median,
        distance_threshold_m=0.008,
        ransac_n=3,
        num_iterations=1200,
    )
    if not ransac_info.get("used"):
        raise RuntimeError(f"normal RANSAC failed: {ransac_info}")
    points_ransac = points_median[inlier_mask]
    uv_ransac = uv_median[inlier_mask]
    colors_ransac = colors_rgb[inlier_mask]
    save_ply_ascii(
        shot_dir / "pointcloud_after_normal_ransac.ply",
        points_ransac,
        colors_ransac,
    )
    # Exact same array is sent to PCA and saved separately for auditable equality.
    points_for_pca = points_ransac
    save_ply_ascii(
        shot_dir / "pointcloud_sent_to_pca.ply", points_for_pca, colors_ransac
    )
    timings["normal_ransac_seconds"] = time.perf_counter() - ransac_start
    ransac_equals_pca = bool(np.array_equal(points_ransac, points_for_pca))

    final_mask = np.zeros_like(mask_used_for_depth, dtype=np.uint8)
    final_mask[uv_ransac[:, 1], uv_ransac[:, 0]] = 1
    final_depth = depth_median.copy()
    final_depth[final_mask == 0] = 0
    current.save_final_png_and_reconstruction_bundle(
        shot_dir,
        color_np,
        final_mask,
        final_depth,
        intr,
        depth_scale,
        stem="comparison_no_mask_merge_no_side_filter",
    )

    pca_start = time.perf_counter()
    mean, pc1, pc2 = pca_axes_fix_dir(points_for_pca)
    vx, vy = float(pc1[0]), float(pc1[1])
    norm_xy = float(np.hypot(vx, vy))
    roll = 0.0 if norm_xy < 1e-8 else float(np.arctan2(vy, vx))
    width_info = estimate_book_width(points_for_pca, mean, pc1, pc2)
    width_m = width_info.get("av_book_width_m")
    if width_m is None:
        raise RuntimeError(f"book width estimation failed: {width_info}")
    target_info = find_target_point(points_for_pca)
    target = target_info.get("target_m")
    if target is None:
        raise RuntimeError(f"target point estimation failed: {target_info}")
    timings["pca_width_target_seconds"] = time.perf_counter() - pca_start

    pca_json = {
        "variant": VARIANT,
        "theta_rad": roll,
        "theta_deg": float(np.degrees(roll)),
        "p_min_m": np.asarray(target, dtype=float).tolist(),
        "book_width_mm": float(width_m * 1000.0),
        "book_width_info": width_info,
        "normal_ransac": ransac_info,
        "pca_input_count": int(points_for_pca.shape[0]),
    }
    _write_json(shot_dir / "pca_result_offline.json", pca_json)

    service_info = _read_json(shot_dir / "sam3_service_inference.json")
    similarity = _read_json(shot_dir / "similarity_scores.json")
    ocr_result = _read_json(shot_dir / "ocr_result.json")
    ocr_runtime = _read_json(shot_dir / "ocr_runtime_info.json")
    selected_ocr = (
        (depth_anchor_info.get("selected_ocr_polygon") or {}).get("text")
        if isinstance(depth_anchor_info, dict)
        else None
    )
    return {
        "roll_rad": roll,
        "point_3d": np.asarray(target, dtype=float).tolist(),
        "pred_book_width_mm": float(width_m * 1000.0),
        "selected_mask_index": selected_index,
        "selected_mask_score": score,
        "selected_mask_area_px": int(selected_mask.sum()),
        "mask_derived_bbox_xyxy": _bbox(selected_mask),
        "selected_ocr_text": selected_ocr,
        "selected_ocr_confidence": _selected_ocr_confidence(
            ocr_result, selected_ocr
        ),
        "ocr_matching_score": similarity["scores"][0].get("score"),
        "ocr_candidate_count": len(similarity.get("scores", [])),
        "raw_mask_count": service_info.get("raw_mask_count"),
        "nms_mask_count": service_info.get("nms_mask_count"),
        "gpu_max_memory_mb": (service_info.get("service_metadata") or {}).get(
            "gpu_memory_mb"
        ),
        "sam3_inference_seconds": (
            service_info.get("service_metadata") or {}
        ).get("inference_seconds"),
        "ocr_inference_seconds": ocr_runtime.get("ocr_predict_sec"),
        "depth_info": depth_info,
        "depth_reference_info": reference_info,
        "point_counts": {
            "before_median_depth_filter": before_median_count,
            "after_median_depth_filter": int(points_median.shape[0]),
            "before_normal_ransac": int(points_median.shape[0]),
            "after_normal_ransac": int(points_ransac.shape[0]),
            "sent_to_pca": int(points_for_pca.shape[0]),
        },
        "verification": {
            "selected_mask_unchanged_after_selection": masks_equal,
            "normal_ransac_output_equals_pca_input": ransac_equals_pca,
        },
        "timings": timings,
        "returned_shot_dir": str(shot_dir),
    }


def _device_info(device, key):
    try:
        return device.get_info(key) if device.supports(key) else None
    except Exception:
        return None


def capture_rgbd_once_no_mask_merge_no_side_filter(
    shot_dir: str | Path,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 6,
) -> tuple[np.ndarray, np.ndarray, object, float, dict]:
    """Capture one aligned RGB-D sample using the production stream settings.

    The sequence mirrors current ``run_capture_and_pca()`` and
    ``capture_one_shot()``: color BGR8 + depth Z16, align depth to color, discard
    ten warm-up frames, apply the existing depth filter, then save one sample.
    ``pipeline.stop()`` is guaranteed by ``finally``.
    """
    shot_dir = Path(shot_dir).expanduser().resolve()
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    align = rs.align(rs.stream.color)
    profile = None
    started = False
    capture_timestamp = datetime.now().astimezone()
    try:
        profile = pipe.start(cfg)
        started = True
        for _ in range(10):
            pipe.wait_for_frames()
        frames = pipe.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = current.depth_filter_like_viewer(aligned.get_depth_frame())
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("aligned RealSense color/depth frame is unavailable")

        color_np = np.asanyarray(color_frame.get_data())
        depth_np = np.asanyarray(depth_frame.get_data())
        if color_np.shape != (height, width, 3) or color_np.dtype != np.uint8:
            raise RuntimeError(
                f"unexpected RGB frame: shape={color_np.shape} dtype={color_np.dtype}"
            )
        if depth_np.shape != (height, width) or depth_np.dtype != np.uint16:
            raise RuntimeError(
                f"unexpected Depth frame: shape={depth_np.shape} dtype={depth_np.dtype}"
            )

        depth_profile = rs.video_stream_profile(depth_frame.get_profile())
        intr = depth_profile.get_intrinsics()
        depth_scale = float(
            profile.get_device().first_depth_sensor().get_depth_scale()
        )
        cv2.imwrite(str(shot_dir / "after_init_rgb.png"), color_np)
        np.save(shot_dir / "after_init_depth.npy", depth_np)

        intrinsics_payload = {
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "ppx": float(intr.ppx),
            "ppy": float(intr.ppy),
            "model": str(intr.model),
            "coeffs": [float(value) for value in intr.coeffs],
            "depth_scale": depth_scale,
        }
        _write_json(shot_dir / "camera_intrinsics.json", intrinsics_payload)
        device = profile.get_device()
        metadata = {
            "capture_timestamp": capture_timestamp.isoformat(),
            "capture_count": 1,
            "warmup_frame_count": 10,
            "rgb_shape": list(color_np.shape),
            "rgb_dtype": str(color_np.dtype),
            "depth_shape": list(depth_np.shape),
            "depth_dtype": str(depth_np.dtype),
            "rgb_format": "bgr8",
            "depth_format": "z16",
            "depth_unit": "meter per Z16 count",
            "depth_scale": depth_scale,
            "camera_intrinsics": intrinsics_payload,
            "aligned_depth_to_color": True,
            "realsense_device_name": _device_info(device, rs.camera_info.name),
            "serial_number": _device_info(device, rs.camera_info.serial_number),
            "firmware_version": _device_info(
                device, rs.camera_info.firmware_version
            ),
            "requested_stream": {
                "width": width,
                "height": height,
                "fps": fps,
                "color": "bgr8",
                "depth": "z16",
            },
        }
        _write_json(shot_dir / "realsense_capture_metadata.json", metadata)
        return color_np, depth_np, intr, depth_scale, metadata
    finally:
        if started:
            pipe.stop()
            print("[REALSENSE] pipeline stopped")


def run_capture_and_pca_no_mask_merge_no_side_filter(
    query: str,
    sam_device: str = "gpu",
    *,
    shot_dir: str | Path | None = None,
    out_dir: str | Path = CAPTURES_DIR,
    width: int = 1280,
    height: int = 720,
    fps: int = 6,
) -> tuple[float, np.ndarray, float, Path]:
    """Capture one live RGB-D sample and run the shared comparison core."""
    if shot_dir is None:
        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        shot_dir = _unique_live_run_dir(datetime.now().astimezone())
    else:
        shot_dir = Path(shot_dir).expanduser().resolve()
        shot_dir.mkdir(parents=True, exist_ok=True)
        if (shot_dir / "after_init_rgb.png").exists() or (
            shot_dir / "after_init_depth.npy"
        ).exists():
            raise FileExistsError(f"refusing to overwrite captured RGB-D in {shot_dir}")

    capture_start = time.perf_counter()
    _, _, intr, depth_scale, metadata = (
        capture_rgbd_once_no_mask_merge_no_side_filter(
            shot_dir, width=width, height=height, fps=fps
        )
    )
    capture_seconds = time.perf_counter() - capture_start
    core = run_capture_and_pca_offline_no_mask_merge_no_side_filter(
        query=query,
        shot_dir=shot_dir,
        sam_device=sam_device,
        intr=intr,
        depth_scale=depth_scale,
    )
    core["capture_seconds"] = capture_seconds
    core["capture_metadata"] = metadata
    _write_json(Path(shot_dir) / "live_core_result.json", core)
    return (
        float(core["roll_rad"]),
        np.asarray(core["point_3d"], dtype=float),
        float(core["pred_book_width_mm"]),
        Path(shot_dir),
    )


def _comparison_payload(current_result: dict, variant_result: dict) -> dict:
    fields = {
        "target_query": "target_query",
        "selected_ocr_text": "selected_ocr_text",
        "selected_mask_index": "selected_mask_index",
        "selected_mask_score": "selected_mask_score",
        "selected_mask_area_px": "selected_mask_area_px",
        "mask_derived_bbox_xyxy": "mask_derived_bbox_xyxy",
        "depth_valid_pixel_count": ("depth", "valid_pixel_count"),
        "selected_mask_valid_depth_pixel_count": (
            "depth",
            "selected_mask_valid_pixel_count",
        ),
        "ransac_before_count": ("point_cloud", "before_filter_count"),
        "ransac_after_count": ("point_cloud", "after_ransac_count"),
        "pca_input_count": ("point_cloud", "pca_input_count"),
        "roll_rad": "roll_rad",
        "pred_book_width_mm": "pred_book_width_mm",
        "abs_error_mm": "abs_error_mm",
        "point_3d": "point_3d",
        "total_seconds": ("timings", "total_seconds"),
    }

    def get(data, key):
        if isinstance(key, tuple):
            value = data
            for part in key:
                value = value.get(part) if isinstance(value, dict) else None
            return value
        return data.get(key)

    rows = {}
    for name, key in fields.items():
        if name == "selected_ocr_text":
            variant_value = (variant_result.get("ocr") or {}).get("selected_text")
        elif name in {
            "selected_mask_index",
            "selected_mask_score",
            "selected_mask_area_px",
            "mask_derived_bbox_xyxy",
        }:
            variant_value = (variant_result.get("sam3") or {}).get(name)
        elif name == "ransac_before_count":
            variant_value = variant_result["point_counts"]["before_normal_ransac"]
        elif name == "ransac_after_count":
            variant_value = variant_result["point_counts"]["after_normal_ransac"]
        elif name == "pca_input_count":
            variant_value = variant_result["point_counts"]["sent_to_pca"]
        elif name == "depth_valid_pixel_count":
            variant_value = variant_result["depth"]["valid_pixel_count"]
        elif name == "selected_mask_valid_depth_pixel_count":
            variant_value = variant_result["depth"][
                "selected_mask_valid_pixel_count"
            ]
        else:
            variant_value = get(variant_result, key)
        rows[name] = {
            "current": get(current_result, key),
            "comparison_variant": variant_value,
        }
    return {
        "current_result": str(CURRENT_RESULT.resolve()),
        "comparison_variant": VARIANT,
        "note": "Differences are expected and are not treated as failures.",
        "fields": rows,
    }


def _write_comparison_markdown(path: Path, comparison: dict) -> None:
    lines = [
        "# Comparison with current SAM3 offline result",
        "",
        "Numerical differences are expected because this variant removes legacy mask",
        "post-processing and side-surface-triggered filtering.",
        "",
        "| Field | Current | Comparison variant |",
        "|---|---:|---:|",
    ]
    for name, values in comparison["fields"].items():
        left = json.dumps(values["current"], ensure_ascii=False)
        right = json.dumps(values["comparison_variant"], ensure_ascii=False)
        lines.append(f"| `{name}` | {left} | {right} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    total_start = time.perf_counter()
    started = datetime.now().astimezone()
    shot_dir = _unique_run_dir(started)
    result_path = shot_dir / "offline_recognition_result.json"
    log_path = shot_dir / "offline_run_console.log"
    result = {
        "success": False,
        "variant": VARIANT,
        "timestamp": started.isoformat(),
        "source_shot_dir": str(SOURCE_SHOT_DIR.resolve()),
        "run_shot_dir": str(shot_dir),
        "master_json": str(MASTER_JSON.resolve()),
        "master_index": 0,
        "error": None,
    }
    failure_stage = "input"

    with log_path.open("w", encoding="utf-8") as handle:
        tee, err = Tee(sys.__stdout__, handle), Tee(sys.__stderr__, handle)
        with redirect_stdout(tee), redirect_stderr(err):
            try:
                master = _read_json(MASTER_JSON)
                query = master[0]["book_name"]
                gt_width = float(master[0]["book_width"])
                print(f"source_shot_dir: {SOURCE_SHOT_DIR.resolve()}")
                print(f"run_shot_dir: {shot_dir}")
                print(f"master_index: 0")
                print(f"target_query: {query}")
                print(f"gt_book_width_mm: {gt_width}")

                copy_start = time.perf_counter()
                copied = []
                for name in REQUIRED_INPUTS:
                    source = (SOURCE_SHOT_DIR / name).resolve()
                    destination = shot_dir / name
                    source_hash = _sha256(source)
                    shutil.copy2(source, destination)
                    destination_hash = _sha256(destination)
                    if source_hash != destination_hash:
                        raise RuntimeError(f"copy hash mismatch: {name}")
                    copied.append(
                        {
                            "name": name,
                            "source": str(source),
                            "destination": str(destination.resolve()),
                            "size_bytes": destination.stat().st_size,
                            "sha256": destination_hash,
                        }
                    )
                    print(f"copied: {name} sha256={destination_hash}")
                copy_seconds = time.perf_counter() - copy_start
                _write_json(
                    shot_dir / "offline_input_manifest.json",
                    {
                        "source_shot_dir": str(SOURCE_SHOT_DIR.resolve()),
                        "run_shot_dir": str(shot_dir),
                        "copied_files": copied,
                        "copy_policy": "copy only files required for offline recognition",
                    },
                )

                failure_stage = "recognition"
                print("comparison offline recognition invocation: 1 of 1")
                core = run_capture_and_pca_offline_no_mask_merge_no_side_filter(
                    query=query, shot_dir=shot_dir, sam_device="gpu"
                )
                depth = np.load(shot_dir / "after_init_depth.npy", allow_pickle=False)
                selected = cv2.imread(
                    str(shot_dir / "selected_mask_used_for_depth.png"),
                    cv2.IMREAD_GRAYSCALE,
                ) > 0
                total_seconds = time.perf_counter() - total_start
                result.update(
                    {
                        "success": True,
                        "target_query": query,
                        "gt_book_width_mm": gt_width,
                        "pred_book_width_mm": core["pred_book_width_mm"],
                        "abs_error_mm": abs(core["pred_book_width_mm"] - gt_width),
                        "roll_rad": core["roll_rad"],
                        "roll_deg": math.degrees(core["roll_rad"]),
                        "point_3d": core["point_3d"],
                        "sam3": {
                            "raw_mask_count": core["raw_mask_count"],
                            "nms_mask_count": core["nms_mask_count"],
                            "selected_mask_index": core["selected_mask_index"],
                            "selected_mask_index_convention": "1-based",
                            "selected_mask_score": core["selected_mask_score"],
                            "selected_mask_area_px": core["selected_mask_area_px"],
                            "mask_derived_bbox_xyxy": core[
                                "mask_derived_bbox_xyxy"
                            ],
                            "prompt": "book spine",
                            "processor_confidence_threshold": 0.05,
                            "score_threshold": 0.3,
                            "min_area": 200,
                            "nms_iou_threshold": 0.5,
                        },
                        "ocr": {
                            "selected_text": core["selected_ocr_text"],
                            "confidence": core["selected_ocr_confidence"],
                            "matching_score": core["ocr_matching_score"],
                            "candidate_count": core["ocr_candidate_count"],
                        },
                        "disabled_processing": {
                            "legacy_non_book_mask_removal": True,
                            "legacy_floating_mask_merge": True,
                            "legacy_mask_shape_completion": True,
                            "side_surface_triggered_point_filter": True,
                            "residual_policy": True,
                            "reject_release": True,
                            "post_ransac_plane_a95": True,
                        },
                        "retained_processing": {
                            "owner_mask_nms": True,
                            "median_depth_filter": True,
                            "normal_ransac": True,
                            "pca": True,
                        },
                        "depth": {
                            "shape": list(depth.shape),
                            "dtype": str(depth.dtype),
                            "valid_pixel_count": int(np.count_nonzero(depth > 0)),
                            "selected_mask_valid_pixel_count": int(
                                np.count_nonzero(selected & (depth > 0))
                            ),
                            "median_filter": core["depth_info"],
                        },
                        "point_counts": core["point_counts"],
                        "verification": core["verification"],
                        "timings": {
                            "input_copy_seconds": copy_seconds,
                            "sam3_inference_seconds": core[
                                "sam3_inference_seconds"
                            ],
                            "ocr_inference_seconds": core[
                                "ocr_inference_seconds"
                            ],
                            **core["timings"],
                            "total_seconds": total_seconds,
                        },
                        "gpu_max_memory_mb": core["gpu_max_memory_mb"],
                        "returned_shot_dir": core["returned_shot_dir"],
                        "outputs": {},
                        "error": None,
                    }
                )
                comparison = _comparison_payload(_read_json(CURRENT_RESULT), result)
                _write_json(shot_dir / "comparison_with_current.json", comparison)
                _write_comparison_markdown(
                    shot_dir / "comparison_with_current.md", comparison
                )
                result["outputs"] = {
                    path.name: str(path.resolve())
                    for path in sorted(shot_dir.iterdir())
                    if path.is_file()
                }
                result["outputs"][result_path.name] = str(result_path.resolve())
                _write_json(result_path, result)
                print(
                    f"selected mask={core['selected_mask_index']} "
                    f"score={core['selected_mask_score']} area={core['selected_mask_area_px']}"
                )
                print(f"point counts={core['point_counts']}")
                print(
                    f"roll={result['roll_rad']} rad width={result['pred_book_width_mm']} mm "
                    f"point={result['point_3d']}"
                )
                print(f"total_seconds={total_seconds}")
                print(f"result_json={result_path}")
                return 0
            except Exception as exc:
                tb = traceback.format_exc()
                traceback.print_exc()
                result.update(
                    {
                        "success": False,
                        "failure_stage": failure_stage,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": tb,
                        "timings": {"total_seconds": time.perf_counter() - total_start},
                    }
                )
                _write_json(result_path, result)
                print(f"FAILED stage={failure_stage}: {exc}")
                return 1


if __name__ == "__main__":
    raise SystemExit(main())
