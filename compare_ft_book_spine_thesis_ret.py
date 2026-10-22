#!/usr/bin/env python3
"""Compare the preserved thesis/ret baseline with the expE checkpoint run."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent / "captures" / "thesis" / "ret"
BASELINE_ROOT = ROOT / "retrieval_results"
FT_ROOT = ROOT / "retrieval_results_ft_expE_ck16"
OUTPUT_ROOT = FT_ROOT / "baseline_comparison"
GT_ROOT = ROOT / "thesis_ret" / "SegmentationClass"
IOU_THRESHOLD = 0.5


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def _bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _mask(path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, bool]:
    if not path.is_file():
        return np.zeros(shape, dtype=bool), False
    value = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0
    if value.shape != shape:
        raise ValueError(f"mask shape mismatch: {path}: {value.shape} != {shape}")
    return value, True


def _overlay(rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    result = rgb.astype(np.float32)
    colors = np.zeros_like(rgb)
    colors[pred & ~gt] = [0, 102, 255]
    colors[gt & ~pred] = [255, 48, 48]
    colors[pred & gt] = [64, 220, 64]
    active = pred | gt
    result[active] = result[active] * 0.55 + colors[active] * 0.45
    return np.clip(result, 0, 255).astype(np.uint8)


def _write_overlay(case_id: str, baseline_iou: float, ft_iou: float) -> Path:
    rgb = np.asarray(
        Image.open(ROOT / str(int(case_id)) / f"{case_id}_rgb.png").convert("RGB"),
        dtype=np.uint8,
    )
    gt_rgb = np.asarray(
        Image.open(GT_ROOT / f"{case_id}_rgb.png").convert("RGB"),
        dtype=np.uint8,
    )
    gt = np.all(gt_rgb == np.asarray([189, 233, 64], dtype=np.uint8), axis=2)
    baseline, _ = _mask(
        BASELINE_ROOT / "predicted_masks" / f"{case_id}_pred_mask.png",
        rgb.shape[:2],
    )
    ft, _ = _mask(
        FT_ROOT / "predicted_masks" / f"{case_id}_pred_mask.png",
        rgb.shape[:2],
    )
    panels = [rgb, _overlay(rgb, baseline, gt), _overlay(rgb, ft, gt)]
    labels = [
        f"input {case_id}",
        f"baseline IoU={baseline_iou:.3f}",
        f"FT expE ck16 IoU={ft_iou:.3f}",
    ]
    header_height = 28
    canvas = Image.new(
        "RGB",
        (rgb.shape[1] * 3, rgb.shape[0] + header_height),
        (0, 0, 0),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index * rgb.shape[1]
        canvas.paste(Image.fromarray(panel, mode="RGB"), (x, header_height))
        draw.text((x + 8, 7), label, fill=(255, 255, 255))
    output = OUTPUT_ROOT / "overlays" / f"{case_id}_baseline_vs_ft.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def main() -> None:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise RuntimeError(f"comparison output is not empty: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    baseline = _read_rows(BASELINE_ROOT / "results.csv")
    ft = _read_rows(FT_ROOT / "results.csv")
    if set(baseline) != set(ft):
        raise RuntimeError("baseline and FT case IDs differ")

    rows = []
    for case_id in sorted(baseline):
        base_row = baseline[case_id]
        ft_row = ft[case_id]
        baseline_iou = float(base_row["iou"])
        ft_iou = float(ft_row["iou"])
        baseline_success = _bool(base_row["recognition_success"])
        ft_success = _bool(ft_row["recognition_success"])
        baseline_pred_exists = (
            BASELINE_ROOT / "predicted_masks" / f"{case_id}_pred_mask.png"
        ).is_file()
        ft_pred_exists = (
            FT_ROOT / "predicted_masks" / f"{case_id}_pred_mask.png"
        ).is_file()
        transition = "unchanged"
        if not baseline_success and ft_success:
            transition = "baseline_failure_to_ft_success"
        elif baseline_success and not ft_success:
            transition = "baseline_success_to_ft_failure"
        rows.append(
            {
                "case_id": case_id,
                "target_title": base_row["target_title"],
                "baseline_iou": baseline_iou,
                "ft_iou": ft_iou,
                "iou_delta": ft_iou - baseline_iou,
                "baseline_success": baseline_success,
                "ft_success": ft_success,
                "transition": transition,
                "baseline_predicted_mask_exists": baseline_pred_exists,
                "ft_predicted_mask_exists": ft_pred_exists,
                "baseline_error": base_row["error_reason"],
                "ft_error": ft_row["error_reason"],
            }
        )

    transitions = [row for row in rows if row["transition"] != "unchanged"]
    top_improved = sorted(rows, key=lambda row: row["iou_delta"], reverse=True)[:5]
    top_worsened = sorted(rows, key=lambda row: row["iou_delta"])[:5]
    overlay_ids = sorted(
        {
            row["case_id"]
            for row in [*transitions, *top_improved, *top_worsened]
        }
    )
    overlays = {}
    by_id = {row["case_id"]: row for row in rows}
    for case_id in overlay_ids:
        row = by_id[case_id]
        overlays[case_id] = str(
            _write_overlay(case_id, row["baseline_iou"], row["ft_iou"])
        )

    fields = list(rows[0])
    with (OUTPUT_ROOT / "scene_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    baseline_ious = [row["baseline_iou"] for row in rows]
    ft_ious = [row["ft_iou"] for row in rows]
    baseline_summary = json.loads(
        (BASELINE_ROOT / "summary.json").read_text(encoding="utf-8")
    )
    ft_summary = json.loads((FT_ROOT / "summary.json").read_text(encoding="utf-8"))
    summary = {
        "iou_threshold": IOU_THRESHOLD,
        "case_count": len(rows),
        "baseline": {
            "source": str((BASELINE_ROOT / "summary.json").resolve()),
            "success_count": sum(row["baseline_success"] for row in rows),
            "success_rate_percent": 100.0
            * sum(row["baseline_success"] for row in rows)
            / len(rows),
            "mean_iou": statistics.fmean(baseline_ious),
            "median_iou": statistics.median(baseline_ious),
            "missing_selected_mask_count": sum(
                not row["baseline_predicted_mask_exists"] for row in rows
            ),
            "wrong_selected_mask_below_threshold_count": sum(
                row["baseline_predicted_mask_exists"]
                and row["baseline_iou"] < IOU_THRESHOLD
                for row in rows
            ),
            "sam3_time_mean_ms": baseline_summary["timing_ms"]["sam3"]["mean"],
            "width_mae_mm": baseline_summary["width_mae_mm"],
        },
        "ft": {
            "source": str((FT_ROOT / "summary.json").resolve()),
            "success_count": sum(row["ft_success"] for row in rows),
            "success_rate_percent": 100.0
            * sum(row["ft_success"] for row in rows)
            / len(rows),
            "mean_iou": statistics.fmean(ft_ious),
            "median_iou": statistics.median(ft_ious),
            "missing_selected_mask_count": sum(
                not row["ft_predicted_mask_exists"] for row in rows
            ),
            "wrong_selected_mask_below_threshold_count": sum(
                row["ft_predicted_mask_exists"] and row["ft_iou"] < IOU_THRESHOLD
                for row in rows
            ),
            "sam3_time_mean_ms": ft_summary["timing_ms"]["sam3"]["mean"],
            "width_mae_mm": ft_summary["width_mae_mm"],
        },
        "delta_ft_minus_baseline": {
            "success_count": sum(row["ft_success"] for row in rows)
            - sum(row["baseline_success"] for row in rows),
            "success_rate_percentage_points": 100.0
            * (
                sum(row["ft_success"] for row in rows)
                - sum(row["baseline_success"] for row in rows)
            )
            / len(rows),
            "mean_iou": statistics.fmean(ft_ious) - statistics.fmean(baseline_ious),
            "median_iou": statistics.median(ft_ious) - statistics.median(baseline_ious),
            "sam3_time_mean_ms": ft_summary["timing_ms"]["sam3"]["mean"]
            - baseline_summary["timing_ms"]["sam3"]["mean"],
            "width_mae_mm": ft_summary["width_mae_mm"]
            - baseline_summary["width_mae_mm"],
        },
        "scene_counts": {
            "iou_improved": sum(row["iou_delta"] > 0.0 for row in rows),
            "iou_worsened": sum(row["iou_delta"] < 0.0 for row in rows),
            "iou_improved_by_at_least_0_1": sum(
                row["iou_delta"] >= 0.1 for row in rows
            ),
            "iou_worsened_by_at_least_0_1": sum(
                row["iou_delta"] <= -0.1 for row in rows
            ),
            "baseline_failure_to_ft_success": sum(
                row["transition"] == "baseline_failure_to_ft_success" for row in rows
            ),
            "baseline_success_to_ft_failure": sum(
                row["transition"] == "baseline_success_to_ft_failure" for row in rows
            ),
        },
        "transitions": transitions,
        "top_improved": top_improved,
        "top_worsened": top_worsened,
        "comparison_overlays": overlays,
    }
    (OUTPUT_ROOT / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT_ROOT / "comparison_summary.json")


if __name__ == "__main__":
    main()
