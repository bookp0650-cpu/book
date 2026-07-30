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
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_SHOT_DIR = PROJECT_ROOT / "captures" / "100test" / "1"
CAPTURES_DIR = PROJECT_ROOT / "captures"
MASTER_JSON = PROJECT_ROOT / "master_20260216.json"
MODEL_PATH = (
    PROJECT_ROOT
    / "detection"
    / "pro_handbook"
    / "sam3_runtime"
    / "models"
    / "inference_best.pt"
)
REQUIRED_INPUTS = ("after_init_rgb.png", "after_init_depth.npy")
EXPECTED_MODEL_SHA256 = "d8b297b0a9a8a81c7926541a0f8fb08f7a15ee7d53d210b9827190aa21b16bce"


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def unique_run_dir(started: datetime) -> Path:
    base_name = started.strftime("20260723_%H%M%S")
    candidate = CAPTURES_DIR / base_name
    suffix = 1
    while candidate.exists():
        candidate = CAPTURES_DIR / f"{base_name}_{suffix:03d}"
        suffix += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate.resolve()


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if not len(xs):
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def japanese_font(size: int):
    candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def create_target_outputs(
    run_dir: Path,
    target_query: str,
    selected_index_1based: int,
    selected_score: float,
) -> tuple[np.ndarray, list[int]]:
    data = np.load(run_dir / "sam3_service_masks.npz", allow_pickle=False)
    masks = data["masks"].astype(bool)
    selected = masks[selected_index_1based - 1]
    bbox = bbox_from_mask(selected)
    rgb = np.asarray(Image.open(run_dir / "after_init_rgb.png").convert("RGB"))

    Image.fromarray(selected.astype(np.uint8) * 255, mode="L").save(
        run_dir / "target_book_mask.png"
    )
    target_only = rgb.copy()
    target_only[~selected] = 0
    Image.fromarray(target_only, mode="RGB").save(run_dir / "target_book_only.png")

    overlay = rgb.copy()
    overlay[selected] = (
        0.55 * overlay[selected] + 0.45 * np.asarray([255, 48, 48])
    ).astype(np.uint8)
    image = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = japanese_font(22)
    label = (
        f"{target_query}\n"
        f"mask index: {selected_index_1based} (1-based)  score: {selected_score:.6f}\n"
        f"area: {int(selected.sum())} px  bbox XYXY: {bbox}"
    )
    text_bbox = draw.multiline_textbbox((12, 10), label, font=font, spacing=5)
    draw.rectangle(
        (text_bbox[0] - 6, text_bbox[1] - 5, text_bbox[2] + 6, text_bbox[3] + 5),
        fill=(0, 0, 0),
    )
    draw.multiline_text((12, 10), label, font=font, fill=(255, 255, 255), spacing=5)
    draw.rectangle(tuple(bbox), outline=(255, 255, 0), width=3)
    image.save(run_dir / "target_book_mask_overlay.png")
    return selected, bbox


def match_ocr_confidence(ocr_result: dict, selected_text: str | None):
    if not selected_text:
        return None
    texts = ocr_result.get("rec_texts") or []
    scores = ocr_result.get("rec_scores") or []
    for text, score in zip(texts, scores):
        if str(text).strip() == str(selected_text).strip():
            return float(score)
    return None


def extract_timing(console: str, label: str):
    match = re.search(rf"{re.escape(label)}\s*:\s*([0-9.]+)\s*sec", console)
    return float(match.group(1)) if match else None


def main() -> int:
    total_start = time.perf_counter()
    started = datetime.now().astimezone()
    run_dir = unique_run_dir(started)
    log_path = run_dir / "offline_run_console.log"
    result_path = run_dir / "offline_recognition_result.json"
    failure_stage = "input"
    result: dict = {
        "success": False,
        "timestamp": started.isoformat(),
        "source_shot_dir": str(SOURCE_SHOT_DIR.resolve()),
        "run_shot_dir": str(run_dir),
        "master_json": str(MASTER_JSON.resolve()),
        "master_index": 0,
        "error": None,
    }

    with log_path.open("w", encoding="utf-8") as log_handle:
        tee = Tee(sys.__stdout__, log_handle)
        err_tee = Tee(sys.__stderr__, log_handle)
        with redirect_stdout(tee), redirect_stderr(err_tee):
            try:
                print(f"source_shot_dir: {SOURCE_SHOT_DIR.resolve()}")
                print(f"run_shot_dir: {run_dir}")
                print(f"master_index: 0")
                print(f"master_json: {MASTER_JSON.resolve()}")

                master_data = read_json(MASTER_JSON)
                if not isinstance(master_data, list) or not master_data:
                    raise ValueError("master JSON must be a non-empty array")
                target_item = master_data[0]
                target_query = target_item["book_name"]
                gt_width = float(target_item["book_width"])
                print(f"book_name: {target_query}")
                print(f"gt_book_width_mm: {gt_width}")
                result.update(
                    {
                        "target_query": target_query,
                        "gt_book_width_mm": gt_width,
                    }
                )

                model_hash = sha256(MODEL_PATH)
                if model_hash != EXPECTED_MODEL_SHA256:
                    raise RuntimeError(
                        f"model SHA-256 mismatch: expected={EXPECTED_MODEL_SHA256} actual={model_hash}"
                    )
                result["model"] = {
                    "path": str(MODEL_PATH.resolve()),
                    "sha256": model_hash,
                }

                copy_start = time.perf_counter()
                copied_files = []
                for name in REQUIRED_INPUTS:
                    source = (SOURCE_SHOT_DIR / name).resolve()
                    destination = run_dir / name
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    before_hash = sha256(source)
                    shutil.copy2(source, destination)
                    after_hash = sha256(destination)
                    if before_hash != after_hash:
                        raise RuntimeError(f"copy SHA-256 mismatch: {name}")
                    item = {
                        "name": name,
                        "source": str(source),
                        "destination": str(destination.resolve()),
                        "size_bytes": destination.stat().st_size,
                        "sha256": after_hash,
                    }
                    copied_files.append(item)
                    print(f"copied: {name} sha256={after_hash}")
                input_copy_seconds = time.perf_counter() - copy_start
                write_json(
                    run_dir / "offline_input_manifest.json",
                    {
                        "source_shot_dir": str(SOURCE_SHOT_DIR.resolve()),
                        "run_shot_dir": str(run_dir),
                        "copied_files": copied_files,
                        "copy_policy": "copy only files required for offline recognition",
                    },
                )

                failure_stage = "sam3"
                from detection.pro_handbook.sam_py_demo.get_book_points import (
                    run_capture_and_pca_offline,
                )

                print("offline recognition invocation: 1 of 1")
                theta_rad, point_3d, pred_width, returned_shot_dir = (
                    run_capture_and_pca_offline(
                        query=target_query,
                        shot_dir=run_dir.resolve(),
                        sam_device="gpu",
                        interactive=False,
                        show_pointcloud_gui=False,
                        save_pointcloud_debug=True,
                        save_step_by_step_pointcloud_debug=True,
                    )
                )
                if theta_rad is None or point_3d is None or pred_width is None:
                    raise RuntimeError("offline recognition returned an unsuccessful result")

                failure_stage = "output"
                service_info = read_json(run_dir / "sam3_service_inference.json")
                similarity = read_json(run_dir / "similarity_scores.json")
                selected_candidate = similarity["scores"][0]
                selected_index = int(
                    re.search(r"(\d+)$", selected_candidate["name"]).group(1)
                )
                instances = service_info["instances"]
                selected_instance = instances[selected_index - 1]
                selected_score = float(selected_instance["score"])
                selected_mask, mask_bbox = create_target_outputs(
                    run_dir, target_query, selected_index, selected_score
                )

                depth = np.load(run_dir / "after_init_depth.npy", allow_pickle=False)
                valid_depth = depth > 0
                processing_path = run_dir / f"mask{selected_index}_offline_processing_log.json"
                processing = read_json(processing_path)
                ransac = processing.get("ransac_spine_plane_info") or {}
                reconstruction = processing.get("final_reconstruction_info") or {}
                pca = read_json(run_dir / "pca_result_offline.json")
                depth_reference = (
                    (processing.get("depth_prefilter") or {}).get("ocr_depth_reference")
                    or {}
                )
                selected_ocr_text = depth_reference.get("ocr_text")
                ocr_result = read_json(run_dir / "ocr_result.json")
                ocr_runtime = read_json(run_dir / "ocr_runtime_info.json")
                ocr_confidence = match_ocr_confidence(ocr_result, selected_ocr_text)

                output_write_start = time.perf_counter()
                pred_width = float(pred_width)
                theta_rad = float(theta_rad)
                point = np.asarray(point_3d, dtype=float).reshape(-1).tolist()
                console = tee.text + err_tee.text
                sam_seconds = extract_timing(console, "[TIME][OFFLINE] SAM total")
                ocr_seconds = extract_timing(console, "[TIME][OFFLINE] OCR wall")
                service_meta = service_info.get("service_metadata") or {}
                timings = {
                    "input_copy_seconds": input_copy_seconds,
                    "sam3_inference_seconds": service_meta.get(
                        "inference_seconds", sam_seconds
                    ),
                    "sam3_offline_call_seconds": sam_seconds,
                    "ocr_inference_seconds": ocr_runtime.get("ocr_predict_sec"),
                    "ocr_wall_seconds": ocr_seconds,
                    "depth_processing_seconds": None,
                    "ransac_seconds": None,
                    "pca_seconds": None,
                    "output_write_seconds": None,
                    "total_seconds": None,
                    "unavailable_reason": (
                        "Existing offline core does not expose separate depth, RANSAC, "
                        "or PCA timings; logic was not modified solely for instrumentation."
                    ),
                }
                result.update(
                    {
                        "success": True,
                        "pred_book_width_mm": pred_width,
                        "abs_error_mm": abs(pred_width - gt_width),
                        "roll_rad": theta_rad,
                        "roll_deg": math.degrees(theta_rad),
                        "point_3d": point,
                        "point_3d_implementation_name": "p_min_m / target_point",
                        "sam3_prompt": service_info.get("prompt", "book spine"),
                        "processor_confidence_threshold": service_meta.get(
                            "processor_confidence_threshold", 0.05
                        ),
                        "score_threshold": service_meta.get("score_threshold", 0.3),
                        "raw_mask_count": service_info.get("raw_mask_count"),
                        "nms_mask_count": service_info.get("nms_mask_count"),
                        "selected_mask_index": selected_index,
                        "selected_mask_index_convention": "1-based, matching get_book_points.py",
                        "selected_mask_score": selected_score,
                        "selected_mask_area_px": int(selected_mask.sum()),
                        "model_predicted_bbox_xyxy": None,
                        "model_predicted_bbox_unavailable_reason": (
                            "Owner Sam3Adapter intentionally discards processor boxes and "
                            "recomputes bboxes from thresholded masks."
                        ),
                        "service_returned_mask_derived_bbox_xyxy": [
                            float(selected_instance["box"][key])
                            for key in ("x1", "y1", "x2", "y2")
                        ],
                        "mask_derived_bbox_xyxy": mask_bbox,
                        "selected_ocr_text": selected_ocr_text,
                        "selected_ocr_confidence": ocr_confidence,
                        "ocr_candidates": similarity.get("scores", []),
                        "depth": {
                            "shape": list(depth.shape),
                            "dtype": str(depth.dtype),
                            "valid_pixel_count": int(valid_depth.sum()),
                            "selected_mask_valid_pixel_count": int(
                                np.logical_and(valid_depth, selected_mask).sum()
                            ),
                        },
                        "point_cloud": {
                            "before_filter_count": ransac.get("valid_count_before"),
                            "after_ransac_count": ransac.get("valid_count_after"),
                            "pca_input_count": reconstruction.get("point_count"),
                        },
                        "timings": timings,
                        "gpu_max_memory_mb": service_meta.get("gpu_memory_mb"),
                        "returned_shot_dir": str(Path(returned_shot_dir).resolve()),
                        "outputs": {},
                        "error": None,
                    }
                )
                timings["output_write_seconds"] = (
                    time.perf_counter() - output_write_start
                )
                timings["total_seconds"] = time.perf_counter() - total_start
                result["outputs"] = {
                    path.name: str(path.resolve())
                    for path in sorted(run_dir.iterdir())
                    if path.is_file()
                }
                result["outputs"][result_path.name] = str(result_path.resolve())
                write_json(result_path, result)
                print(f"SAM3 masks: raw={result['raw_mask_count']} nms={result['nms_mask_count']}")
                print(f"OCR candidates: {len(result['ocr_candidates'])}")
                print(f"selected OCR: {selected_ocr_text} confidence={ocr_confidence}")
                print(
                    f"selected mask: index={selected_index} score={selected_score} "
                    f"area={int(selected_mask.sum())} bbox={mask_bbox}"
                )
                print(
                    f"Depth: shape={depth.shape} valid={int(valid_depth.sum())} "
                    f"selected_valid={result['depth']['selected_mask_valid_pixel_count']}"
                )
                print(
                    f"RANSAC points: before={ransac.get('valid_count_before')} "
                    f"after={ransac.get('valid_count_after')}"
                )
                print(f"PCA points: {reconstruction.get('point_count')}")
                print(f"roll: {theta_rad} rad / {math.degrees(theta_rad)} deg")
                print(f"pred_book_width_mm: {pred_width}")
                print(f"point_3d: {point}")
                print(f"total_seconds: {timings['total_seconds']}")
                print(f"result_json: {result_path}")
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
                result["outputs"] = {
                    path.name: str(path.resolve())
                    for path in sorted(run_dir.iterdir())
                    if path.is_file()
                }
                result["outputs"][result_path.name] = str(result_path.resolve())
                write_json(result_path, result)
                print(f"FAILED stage={failure_stage}: {exc}")
                print(f"result_json: {result_path}")
                return 1


if __name__ == "__main__":
    raise SystemExit(main())
