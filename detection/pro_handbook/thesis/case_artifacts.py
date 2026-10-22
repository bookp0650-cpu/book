#!/usr/bin/env python3
"""Build review-friendly thesis artifacts from an already completed retrieval run."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_ROOT = PROJECT_ROOT / "captures" / "evaluation_20261005_153514"
RESULTS_ROOT = EVALUATION_ROOT / "retrieval_results"
GT_ROOT = EVALUATION_ROOT / "thesis_ret" / "SegmentationClass"
LABELMAP_PATH = EVALUATION_ROOT / "thesis_ret" / "labelmap.txt"
DEFAULT_ARTIFACT_ROOT = RESULTS_ROOT / "case_artifacts"
TARGET_LABEL = "book spine"

RUN_README = """# Thesis retrieval case artifacts

Each numbered directory is a review-oriented view of one recognition case.
`results.csv` is the run-level source of truth; each `result.json` is a concise
case view of the same values.

- `input_*`: exact RGB-D input used by recognition and its saved visualization
- `sam3_*`: all-mask overlay, raw masks, scores, and service metadata
- `ocr_*`: three-angle OCR overlays/results and OCR-to-SAM3 assignments
- `similarity_scores.json`: target-title similarity ranking and threshold
- `selected_mask*`: selected target and the mask actually used for depth
- `processed_depth_vis.png`: depth after the recognition depth filter
- `geometry_result.json` / `final_recognition.png`: final geometry and overlay
- `pred_mask.png`, `gt_mask.png`, `pred_gt_comparison.png`: evaluation-only output

`pred_gt_comparison.png` uses blue for prediction-only, red for GT-only, and
green for overlap. These GT products are generated after recognition and never
participate in target selection. Missing post-selection files are expected when
recognition stops before a target mask is selected; see `artifact_manifest.json`.
"""

COPIED_LOG_FILES = {
    "after_init_rgb.png": "input_rgb.png",
    "after_init_depth.npy": "input_depth.npy",
    "source_camera_intrinsics.json": "camera_intrinsics.json",
    "sam3_all_masks_overlay.png": "sam3_masks_overlay.png",
    "sam3_service_masks.npz": "sam3_masks.npz",
    "sam3_service_inference.json": "sam3_inference.json",
    "ocr_overlay_0deg.png": "ocr_overlay_0deg.png",
    "ocr_overlay_p030.png": "ocr_overlay_p030.png",
    "ocr_overlay_m030.png": "ocr_overlay_m030.png",
    "ocr_result.json": "ocr_result.json",
    "ocr_result_0deg.json": "ocr_result_0deg.json",
    "ocr_result_p030.json": "ocr_result_p030.json",
    "ocr_result_m030.json": "ocr_result_m030.json",
    "ocr_runtime_info.json": "ocr_runtime_info.json",
    "text_mask_iou_debug.json": "ocr_sam3_matching.json",
    "similarity_scores.json": "similarity_scores.json",
    "before_remove_islands.png": "before_remove_islands.png",
    "selected_mask_before_legacy_postprocess.png": "selected_mask.png",
    "selected_mask_used_for_depth.png": "selected_mask_used_for_depth.png",
    "target_book_mask_overlay.png": "selected_mask_overlay.png",
    "target_book_only.png": "selected_book_rgb.png",
    "comparison_median_depth_filter_depth_masked_vis.png": "processed_depth_vis.png",
    "final.png": "final_recognition.png",
    "pca_result_offline.json": "geometry_result.json",
    "console.log": "console.log",
}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_labelmap(path: Path) -> dict[str, tuple[int, int, int]]:
    labels: dict[str, tuple[int, int, int]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(":", 3)
        if len(fields) < 2:
            continue
        color = tuple(int(value.strip()) for value in fields[1].split(","))
        if len(color) == 3:
            labels[fields[0].strip()] = color
    return labels


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _copy_if_present(
    source: Path,
    destination: Path,
    copied: dict[str, str],
    missing: dict[str, str],
) -> None:
    if source.is_file():
        shutil.copy2(source, destination)
        copied[destination.name] = str(source.resolve())
    else:
        missing[destination.name] = str(source.resolve())


def _load_binary_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        return np.zeros(shape, dtype=bool)
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0
    if mask.shape != shape:
        raise ValueError(f"mask shape mismatch: {path}: {mask.shape} != {shape}")
    return mask


def _save_gt_review_images(
    *,
    input_rgb_path: Path,
    gt_source_path: Path,
    pred_source_path: Path,
    case_dir: Path,
    target_rgb: tuple[int, int, int],
) -> list[str]:
    rgb = np.asarray(Image.open(input_rgb_path).convert("RGB"), dtype=np.uint8)
    gt_rgb = np.asarray(Image.open(gt_source_path).convert("RGB"), dtype=np.uint8)
    if gt_rgb.shape != rgb.shape:
        raise ValueError(f"GT/RGB shape mismatch: {gt_rgb.shape} != {rgb.shape}")
    gt = np.all(gt_rgb == np.asarray(target_rgb, dtype=np.uint8), axis=2)
    pred = _load_binary_mask(pred_source_path, rgb.shape[:2])

    Image.fromarray(gt.astype(np.uint8) * 255, mode="L").save(case_dir / "gt_mask.png")
    if pred_source_path.is_file():
        shutil.copy2(pred_source_path, case_dir / "pred_mask.png")

    comparison = rgb.astype(np.float32)
    colors = np.zeros_like(rgb, dtype=np.uint8)
    pred_only = pred & ~gt
    gt_only = gt & ~pred
    overlap = pred & gt
    colors[pred_only] = [0, 102, 255]
    colors[gt_only] = [255, 48, 48]
    colors[overlap] = [64, 220, 64]
    active = pred | gt
    comparison[active] = 0.55 * comparison[active] + 0.45 * colors[active]
    Image.fromarray(np.clip(comparison, 0, 255).astype(np.uint8), mode="RGB").save(
        case_dir / "pred_gt_comparison.png"
    )
    generated = ["gt_mask.png", "pred_gt_comparison.png"]
    if pred_source_path.is_file():
        generated.append("pred_mask.png")
    return generated


def _failure_category(row: dict[str, Any]) -> str | None:
    error = str(row.get("error_reason") or "")
    if "TargetMaskSelectionError" in error:
        return "ocr_matching_threshold_not_met"
    if not _as_bool(row.get("recognition_success")):
        iou = _optional_float(row.get("iou"))
        threshold = _optional_float(row.get("iou_threshold"))
        if iou is not None and threshold is not None and iou < threshold:
            return "iou_below_threshold"
        return "recognition_failure"
    return None


def prepare_artifact_run(
    *, output_root: Path, run_id: str, source_run_dir: Path, results_csv: Path
) -> Path:
    run_root = output_root / run_id
    if run_root.exists():
        raise FileExistsError(f"artifact run already exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "README.md").write_text(RUN_README, encoding="utf-8")
    _write_json(
        run_root / "run_info.json",
        {
            "run_id": run_id,
            "status": "building",
            "created_at": datetime.now().astimezone().isoformat(),
            "source_run_dir": str(source_run_dir.resolve()),
            "results_csv_source_of_truth": str(results_csv.resolve()),
            "layout": "one review directory per case",
        },
    )
    return run_root


def build_case_artifact(
    *,
    row: dict[str, Any],
    source_case_dir: Path,
    capture_case_dir: Path,
    predicted_masks_dir: Path,
    gt_root: Path,
    artifact_run_root: Path,
    target_rgb: tuple[int, int, int],
) -> Path:
    case_id = str(row["case_id"]).zfill(4)
    case_dir = artifact_run_root / case_id
    if case_dir.exists():
        raise FileExistsError(f"case artifact already exists: {case_dir}")
    case_dir.mkdir(parents=False, exist_ok=False)

    copied: dict[str, str] = {}
    missing: dict[str, str] = {}
    for source_name, destination_name in COPIED_LOG_FILES.items():
        _copy_if_present(
            source_case_dir / source_name,
            case_dir / destination_name,
            copied,
            missing,
        )

    depth_vis_source = capture_case_dir / f"{case_id}_depth_vis.png"
    _copy_if_present(
        depth_vis_source,
        case_dir / "input_depth_vis.png",
        copied,
        missing,
    )

    gt_source = gt_root / f"{case_id}_rgb.png"
    pred_source = predicted_masks_dir / f"{case_id}_pred_mask.png"
    generated = _save_gt_review_images(
        input_rgb_path=case_dir / "input_rgb.png",
        gt_source_path=gt_source,
        pred_source_path=pred_source,
        case_dir=case_dir,
        target_rgb=target_rgb,
    )
    if not pred_source.is_file():
        missing["pred_mask.png"] = str(pred_source.resolve())

    grasp_values = [
        _optional_float(row.get("grasp_x")),
        _optional_float(row.get("grasp_y")),
        _optional_float(row.get("grasp_z")),
    ]
    grasp_point = None if any(value is None for value in grasp_values) else grasp_values
    timing_ms = {
        key: _optional_float(value)
        for key, value in row.items()
        if key.endswith("_time_ms")
    }
    failure_category = _failure_category(row)
    result = {
        "case_id": case_id,
        "target_title": row.get("target_title"),
        "matched_text": row.get("matched_text") or None,
        "similarity_score": _optional_float(row.get("similarity_score")),
        "iou": _optional_float(row.get("iou")),
        "iou_threshold": _optional_float(row.get("iou_threshold")),
        "recognition_success": _as_bool(row.get("recognition_success")),
        "failure_category": failure_category,
        "gt_width_mm": _optional_float(row.get("gt_width_mm")),
        "pred_width_mm": _optional_float(row.get("pred_width_mm")),
        "width_abs_error_mm": _optional_float(row.get("width_abs_error_mm")),
        "book_angle": _optional_float(row.get("book_angle")),
        "book_angle_unit": row.get("book_angle_unit") or None,
        "grasp_point": grasp_point,
        "grasp_unit": row.get("grasp_unit") or None,
        "error_reason": row.get("error_reason") or "",
        "timing_ms": timing_ms,
        "selected_mask_available": (case_dir / "selected_mask.png").is_file(),
        "results_csv_source_of_truth": str(
            (artifact_run_root / "results.csv").resolve()
        ),
        "source_case_log_dir": str(source_case_dir.resolve()),
        "raw_pointcloud_artifacts_remain_in_source_log": True,
    }
    _write_json(case_dir / "result.json", result)
    generated.append("result.json")

    _write_json(
        case_dir / "artifact_manifest.json",
        {
            "case_id": case_id,
            "copied": copied,
            "generated": generated,
            "missing": missing,
            "missing_note": (
                "Missing post-selection artifacts are expected when recognition stops "
                "before target-mask selection."
            ),
            "pred_gt_comparison_legend": {
                "predicted_only": "blue",
                "ground_truth_only": "red",
                "overlap": "green",
            },
        },
    )
    return case_dir


def finalize_artifact_run(
    *, output_root: Path, run_root: Path, case_ids: Iterable[str], results_csv: Path
) -> None:
    case_ids = [str(case_id).zfill(4) for case_id in case_ids]
    selected_ids = set(case_ids)
    info_path = run_root / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.update(
        {
            "status": "complete",
            "completed_at": datetime.now().astimezone().isoformat(),
            "case_count": len(case_ids),
            "case_ids": case_ids,
        }
    )
    with results_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        snapshot_rows = [
            row
            for row in reader
            if str(row.get("case_id", "")).zfill(4) in selected_ids
        ]
    if "artifact_error_reason" not in fieldnames:
        fieldnames.append("artifact_error_reason")
    for row in snapshot_rows:
        case_id = str(row["case_id"]).zfill(4)
        artifact_dir = run_root / case_id
        row["case_artifacts_dir"] = str(artifact_dir.resolve())
        row["artifact_error_reason"] = "" if artifact_dir.is_dir() else "missing"
    with (run_root / "results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(snapshot_rows)
    _write_json(info_path, info)
    _write_json(
        output_root / "latest_run.json",
        {
            "run_id": run_root.name,
            "path": str(run_root.resolve()),
            "case_count": len(case_ids),
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build case_artifacts from an existing completed thesis run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, default=RESULTS_ROOT / "results.csv")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--indices", nargs="+", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    results_csv = args.results_csv.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8-sig", newline="")))
    if args.indices:
        requested = {f"{index:04d}" for index in args.indices}
        rows = [row for row in rows if str(row["case_id"]).zfill(4) in requested]
        missing_rows = requested - {str(row["case_id"]).zfill(4) for row in rows}
        if missing_rows:
            raise ValueError(f"cases absent from results CSV: {sorted(missing_rows)}")
    if not rows:
        raise ValueError("no result rows selected")

    labels = _parse_labelmap(LABELMAP_PATH)
    target_rgb = labels[TARGET_LABEL]
    output_root.mkdir(parents=True, exist_ok=True)
    run_root = prepare_artifact_run(
        output_root=output_root,
        run_id=run_dir.name,
        source_run_dir=run_dir,
        results_csv=results_csv,
    )
    built: list[str] = []
    for row in rows:
        case_id = str(row["case_id"]).zfill(4)
        build_case_artifact(
            row=row,
            source_case_dir=run_dir / case_id,
            capture_case_dir=EVALUATION_ROOT / str(int(case_id)),
            predicted_masks_dir=RESULTS_ROOT / "predicted_masks",
            gt_root=GT_ROOT,
            artifact_run_root=run_root,
            target_rgb=target_rgb,
        )
        built.append(case_id)
    finalize_artifact_run(
        output_root=output_root,
        run_root=run_root,
        case_ids=built,
        results_csv=results_csv,
    )
    print(f"case artifacts: {run_root}")
    print(f"cases: {len(built)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
