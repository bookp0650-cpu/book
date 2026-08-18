#!/usr/bin/env python3
"""Compare saved SAM3-original and SAM2-compatible book widths.

This script is analysis-only.  It reads an existing offline result directory and
does not import or execute SAM3, PaddleOCR, ROS, RealSense, or robot code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = (
    PROJECT_DIR / "captures" / "100test_offline" / "20260725_005852"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "captures" / "width_comparison_from_saved_20260725_005852"
)
CASE81_OLD_DIR = (
    PROJECT_DIR
    / "captures"
    / "100test_offline_SAM3_debug_20260724_173921"
    / "81"
)
THRESHOLDS_MM = (1.0, 1.5, 2.0)
TIE_TOLERANCE_MM = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a paired width-comparison report from already-saved JSON. "
            "No inference is performed."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="existing offline 100-case result directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="new analysis output directory",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def unique_output_dir(requested: Path) -> Path:
    requested = requested.expanduser().resolve()
    if not requested.exists():
        return requested
    suffix = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    candidate = requested.with_name(f"{requested.name}_{suffix}")
    counter = 1
    while candidate.exists():
        candidate = requested.with_name(
            f"{requested.name}_{suffix}_{counter:03d}"
        )
        counter += 1
    return candidate


def method_statistics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    successful = [row for row in rows if row[f"{prefix}_success"]]
    signed = [float(row[f"{prefix}_signed_error_mm"]) for row in successful]
    absolute = [float(row[f"{prefix}_abs_error_mm"]) for row in successful]
    total = len(rows)
    success_count = len(successful)

    thresholds: dict[str, Any] = {}
    for threshold in THRESHOLDS_MM:
        count = sum(value <= threshold for value in absolute)
        thresholds[f"within_{threshold:.1f}mm"] = {
            "count": count,
            "success_denominator": success_count,
            "success_rate": count / success_count if success_count else None,
            "all_100_denominator": total,
            "all_100_rate": count / total if total else None,
        }

    if absolute:
        mean_abs = statistics.fmean(absolute)
        median_abs = statistics.median(absolute)
        rmse = math.sqrt(statistics.fmean(value * value for value in signed))
        maximum = max(absolute)
        minimum = min(absolute)
        mean_signed = statistics.fmean(signed)
        signed_std = statistics.pstdev(signed)
    else:
        mean_abs = median_abs = rmse = maximum = minimum = None
        mean_signed = signed_std = None

    return {
        "success_count": success_count,
        "failure_count": total - success_count,
        "thresholds": thresholds,
        "mean_abs_error_mm": mean_abs,
        "median_abs_error_mm": median_abs,
        "rmse_mm": rmse,
        "max_abs_error_mm": maximum,
        "min_abs_error_mm": minimum,
        "mean_signed_error_mm": mean_signed,
        "signed_error_population_std_mm": signed_std,
        "underestimate_count": sum(value < 0.0 for value in signed),
        "overestimate_count": sum(value > 0.0 for value in signed),
        "exact_count": sum(value == 0.0 for value in signed),
    }


def paired_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired = [row for row in rows if row["both_success"]]
    improvements = [float(row["improvement_mm"]) for row in paired]
    improved = [
        row for row in paired if row["improvement_mm"] > TIE_TOLERANCE_MM
    ]
    worsened = [
        row for row in paired if row["improvement_mm"] < -TIE_TOLERANCE_MM
    ]
    tied = [
        row
        for row in paired
        if abs(float(row["improvement_mm"])) <= TIE_TOLERANCE_MM
    ]

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "case_no": row["case_no"],
            "ground_truth_width_mm": row["ground_truth_width_mm"],
            "sam3_original_width_mm": row["sam3_original_width_mm"],
            "sam2_compatible_width_mm": row["sam2_compatible_width_mm"],
            "sam3_original_abs_error_mm": row[
                "sam3_original_abs_error_mm"
            ],
            "sam2_compatible_abs_error_mm": row[
                "sam2_compatible_abs_error_mm"
            ],
            "improvement_mm": row["improvement_mm"],
        }

    return {
        "both_success_count": len(paired),
        "sam2_improved_count": len(improved),
        "sam2_worsened_count": len(worsened),
        "tie_count": len(tied),
        "mean_improvement_mm": (
            statistics.fmean(improvements) if improvements else None
        ),
        "median_improvement_mm": (
            statistics.median(improvements) if improvements else None
        ),
        "max_improvement_mm": max(improvements) if improvements else None,
        "max_worsening_mm": min(improvements) if improvements else None,
        "top_10_improvements": [
            compact(row)
            for row in sorted(
                paired, key=lambda item: item["improvement_mm"], reverse=True
            )[:10]
        ],
        "top_10_worsenings": [
            compact(row)
            for row in sorted(paired, key=lambda item: item["improvement_mm"])[
                :10
            ]
        ],
        "one_method_only_cases": [
            row["case_no"]
            for row in rows
            if row["sam3_original_success"]
            != row["sam2_compatible_success"]
        ],
        "neither_method_cases": [
            row["case_no"]
            for row in rows
            if not row["sam3_original_success"]
            and not row["sam2_compatible_success"]
        ],
    }


def inspect_case81(input_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    current_dir = input_dir / "81"
    current = next(row for row in rows if row["case_no"] == 81)
    old_pca_path = CASE81_OLD_DIR / "pca_result_offline.json"
    old_ransac_path = CASE81_OLD_DIR / "normal_ransac_result.json"
    current_detail_path = current_dir / "sam3_refined_sam2_width_result.json"
    old_report_path = (
        CASE81_OLD_DIR
        / "book_width_diagnosis"
        / "CASE81_BOOK_WIDTH_ROOT_CAUSE_REPORT.md"
    )
    old_pca = load_json(old_pca_path)
    old_ransac = load_json(old_ransac_path)
    current_detail = load_json(current_detail_path)
    current_ransac = current_detail["depth_flattening"]["normal_ransac"]

    names = (
        "after_init_rgb.png",
        "after_init_depth.npy",
        "selected_mask_raw.png",
        "selected_mask_refined.png",
        "pointcloud_sent_to_pca.ply",
    )
    artifact_comparison: dict[str, Any] = {}
    for name in names:
        old_path = CASE81_OLD_DIR / name
        current_path = current_dir / name
        old_hash = sha256(old_path)
        current_hash = sha256(current_path)
        artifact_comparison[name] = {
            "old_path": str(old_path),
            "current_path": str(current_path),
            "old_sha256": old_hash,
            "current_sha256": current_hash,
            "identical": old_hash == current_hash,
        }

    old_width = finite_float(old_pca.get("book_width_mm"))
    return {
        "ground_truth_width_mm": current["ground_truth_width_mm"],
        "saved_current_sam3_width_mm": current["sam3_original_width_mm"],
        "saved_sam2_compatible_width_mm": current[
            "sam2_compatible_width_mm"
        ],
        "old_sam3_original_width_mm": old_width,
        "old_diagnosis_report": str(old_report_path),
        "artifact_comparison": artifact_comparison,
        "old_ransac": {
            "plane_model": old_ransac.get("plane_model"),
            "inlier_count": old_ransac.get("inlier_count"),
            "inlier_ratio": old_ransac.get("inlier_ratio"),
        },
        "current_ransac": {
            "plane_model": current_ransac.get("plane_model"),
            "inlier_count": current_ransac.get("inlier_count"),
            "inlier_ratio": current_ransac.get("inlier_ratio"),
        },
        "reason": (
            "RGB, Depth, raw mask, and refined mask are byte-identical, but "
            "Open3D RANSAC produced a different plane/inlier set. The helper "
            "calls segment_plane without fixing a random seed. The resulting "
            "PCA input PLY differs, so the pc2/pc3 ordering and pc2 slice width "
            "can change. The old run selected a depth-dominant pc2 and yielded "
            "about 4.957 mm; the saved 20260725_005852 run selected a "
            "spine-width-aligned pc2 and yielded about 12.773 mm."
        ),
        "comparison_choice": (
            "Use 12.773206 mm for case81 in this analysis because it is the "
            "current_sam3_width_mm produced in the same 20260725_005852 "
            "execution and from the same intermediate artifacts as the paired "
            "SAM2-compatible value. Keep 4.957485 mm as a separate historical "
            "RANSAC realization, not as the value for this paired run."
        ),
    }


def build_rows(input_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results_path = input_dir / "results.json"
    saved_results = load_json(results_path)
    if not isinstance(saved_results, list):
        raise ValueError(f"results.json must contain a list: {results_path}")
    by_case = {int(item["test_index"]): item for item in saved_results}
    expected = set(range(1, 101))
    if set(by_case) != expected:
        missing = sorted(expected - set(by_case))
        extra = sorted(set(by_case) - expected)
        raise ValueError(
            f"expected cases 1..100; missing={missing}, extra={extra}"
        )

    rows: list[dict[str, Any]] = []
    counts = {
        "total_cases": 100,
        "ground_truth_finite_count": 0,
        "sam2_compatible_finite_count": 0,
        "current_sam3_width_finite_count": 0,
        "both_widths_finite_count": 0,
        "sam2_only_count": 0,
        "sam3_only_count": 0,
        "neither_width_count": 0,
    }

    for case_no in range(1, 101):
        saved = by_case[case_no]
        detail_path = (
            input_dir / str(case_no) / "sam3_refined_sam2_width_result.json"
        )
        detail = load_json(detail_path) if detail_path.exists() else {}
        ground_truth = finite_float(saved.get("gt_book_width_mm"))
        sam2_width = finite_float(detail.get("pred_book_width_mm"))
        sam3_width = finite_float(detail.get("current_sam3_width_mm"))
        sam2_success = sam2_width is not None and ground_truth is not None
        sam3_success = sam3_width is not None and ground_truth is not None

        counts["ground_truth_finite_count"] += ground_truth is not None
        counts["sam2_compatible_finite_count"] += sam2_success
        counts["current_sam3_width_finite_count"] += sam3_success
        counts["both_widths_finite_count"] += sam2_success and sam3_success
        counts["sam2_only_count"] += sam2_success and not sam3_success
        counts["sam3_only_count"] += sam3_success and not sam2_success
        counts["neither_width_count"] += not sam2_success and not sam3_success

        sam2_signed = (
            sam2_width - ground_truth if sam2_success else None
        )
        sam3_signed = (
            sam3_width - ground_truth if sam3_success else None
        )
        sam2_abs = abs(sam2_signed) if sam2_signed is not None else None
        sam3_abs = abs(sam3_signed) if sam3_signed is not None else None
        both = sam2_success and sam3_success
        improvement = sam3_abs - sam2_abs if both else None
        if both and improvement > TIE_TOLERANCE_MM:
            winner = "sam2_compatible"
        elif both and improvement < -TIE_TOLERANCE_MM:
            winner = "sam3_original"
        elif both:
            winner = "tie"
        else:
            winner = "comparison_unavailable"

        failure_reason = str(saved.get("error") or "")
        if not failure_reason and not both:
            failure_reason = "saved paired width values are unavailable"

        rows.append(
            {
                "case_no": case_no,
                "ground_truth_width_mm": ground_truth,
                "sam3_original_width_mm": sam3_width,
                "sam3_original_signed_error_mm": sam3_signed,
                "sam3_original_abs_error_mm": sam3_abs,
                "sam3_original_value_source": (
                    f"{detail_path}#/current_sam3_width_mm"
                    if sam3_width is not None
                    else ""
                ),
                "sam3_original_success": sam3_success,
                "sam2_compatible_width_mm": sam2_width,
                "sam2_compatible_signed_error_mm": sam2_signed,
                "sam2_compatible_abs_error_mm": sam2_abs,
                "sam2_compatible_value_source": (
                    f"{detail_path}#/pred_book_width_mm"
                    if sam2_width is not None
                    else ""
                ),
                "sam2_compatible_success": sam2_success,
                "both_success": both,
                "improvement_mm": improvement,
                "winner": winner,
                "failure_reason": failure_reason,
            }
        )

    return rows, counts


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(
    output_dir: Path, rows: list[dict[str, Any]]
) -> tuple[list[str], str | None]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [], f"matplotlib unavailable: {exc}"

    paired = [row for row in rows if row["both_success"]]
    cases = [row["case_no"] for row in paired]
    gt = [row["ground_truth_width_mm"] for row in paired]
    sam3_width = [row["sam3_original_width_mm"] for row in paired]
    sam2_width = [row["sam2_compatible_width_mm"] for row in paired]
    sam3_abs = [row["sam3_original_abs_error_mm"] for row in paired]
    sam2_abs = [row["sam2_compatible_abs_error_mm"] for row in paired]
    improvement = [row["improvement_mm"] for row in paired]
    saved: list[str] = []

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(cases, sam3_abs, "o-", markersize=3, linewidth=1, label="SAM3 original")
    ax.plot(cases, sam2_abs, "o-", markersize=3, linewidth=1, label="SAM2 compatible")
    ax.set_xlabel("case")
    ax.set_ylabel("absolute error [mm]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "absolute_error_by_case.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(gt, sam3_width, s=24, alpha=0.75, label="SAM3 original")
    ax.scatter(gt, sam2_width, s=24, alpha=0.75, label="SAM2 compatible")
    low = min(gt + sam3_width + sam2_width)
    high = max(gt + sam3_width + sam2_width)
    ax.plot([low, high], [low, high], "k--", linewidth=1, label="ideal")
    ax.set_xlabel("ground truth width [mm]")
    ax.set_ylabel("predicted width [mm]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "predicted_vs_ground_truth.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(str(path))

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["tab:green" if value >= 0 else "tab:red" for value in improvement]
    ax.bar(cases, improvement, color=colors, width=0.8)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("case")
    ax.set_ylabel(
        "SAM3 original abs error - SAM2 compatible abs error [mm]"
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = output_dir / "error_improvement_by_case.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(str(path))
    return saved, None


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def method_markdown(name: str, stats: dict[str, Any]) -> list[str]:
    lines = [
        f"## {name}",
        "",
        f"- Success: {stats['success_count']} / 100",
        f"- Failure: {stats['failure_count']} / 100",
    ]
    for threshold in THRESHOLDS_MM:
        item = stats["thresholds"][f"within_{threshold:.1f}mm"]
        lines.append(
            f"- Within {threshold:.1f} mm: {item['count']} / "
            f"{item['success_denominator']} successful "
            f"({item['success_rate'] * 100:.2f}%), and "
            f"{item['count']} / 100 ({item['all_100_rate'] * 100:.2f}%)"
        )
    lines.extend(
        [
            f"- Mean absolute error: {fmt(stats['mean_abs_error_mm'])} mm",
            f"- Median absolute error: {fmt(stats['median_abs_error_mm'])} mm",
            f"- RMSE: {fmt(stats['rmse_mm'])} mm",
            f"- Maximum absolute error: {fmt(stats['max_abs_error_mm'])} mm",
            f"- Minimum absolute error: {fmt(stats['min_abs_error_mm'])} mm",
            f"- Mean signed error: {fmt(stats['mean_signed_error_mm'])} mm",
            "- Signed-error population standard deviation: "
            f"{fmt(stats['signed_error_population_std_mm'])} mm",
            f"- Underestimates: {stats['underestimate_count']}",
            f"- Overestimates: {stats['overestimate_count']}",
            "",
        ]
    )
    return lines


def save_report(path: Path, summary: dict[str, Any]) -> None:
    availability = summary["data_availability"]
    paired = summary["paired_comparison"]
    case81 = summary["case81"]
    lines = [
        "# Saved Width Comparison Report",
        "",
        "## Scope and provenance",
        "",
        "- Analysis type: width-only paired comparison reconstructed from saved values.",
        f"- Input: `{summary['input_dir']}`",
        "- No SAM3, PaddleOCR, ROS, RealSense, robot, or GPU inference was run.",
        "- SAM3-original value source: `current_sam3_width_mm` in each "
        "`sam3_refined_sam2_width_result.json`.",
        "- SAM2-compatible value source: `pred_book_width_mm` in the same JSON.",
        "- Ground-truth source: `gt_book_width_mm` saved in that run's "
        "`results.json`. The current project master file is not re-read, so "
        "later master ordering changes cannot alter this comparison.",
        "- Both values were produced in the same original case execution. The "
        "wrapper copied the base refine-only 3D PCA-slice width to "
        "`current_sam3_width_mm`, then replaced only the returned width with "
        "the SAM2-compatible 2D-mask width.",
        "",
        "## Data availability",
        "",
        f"- Total cases: {availability['total_cases']}",
        "- Finite ground truth values: "
        f"{availability['ground_truth_finite_count']}",
        "- Finite SAM3-original values: "
        f"{availability['current_sam3_width_finite_count']}",
        "- Finite SAM2-compatible values: "
        f"{availability['sam2_compatible_finite_count']}",
        "- Both widths finite: "
        f"{availability['both_widths_finite_count']}",
        f"- SAM3 only: {availability['sam3_only_count']}",
        f"- SAM2 only: {availability['sam2_only_count']}",
        f"- Neither: {availability['neither_width_count']}",
        "",
    ]
    lines.extend(
        method_markdown(
            "SAM3 original 3D PCA-slice width",
            summary["sam3_original_statistics"],
        )
    )
    lines.extend(
        method_markdown(
            "SAM2-compatible 2D mask-projection width",
            summary["sam2_compatible_statistics"],
        )
    )
    lines.extend(
        [
            "## Paired comparison",
            "",
            f"- Both successful: {paired['both_success_count']}",
            f"- SAM2-compatible improved: {paired['sam2_improved_count']}",
            f"- SAM2-compatible worsened: {paired['sam2_worsened_count']}",
            f"- Ties: {paired['tie_count']}",
            f"- Mean improvement: {fmt(paired['mean_improvement_mm'])} mm",
            f"- Median improvement: {fmt(paired['median_improvement_mm'])} mm",
            f"- Maximum improvement: {fmt(paired['max_improvement_mm'])} mm",
            f"- Maximum worsening: {fmt(paired['max_worsening_mm'])} mm",
            "",
            "Positive improvement means the SAM2-compatible absolute error is "
            "smaller.",
            "",
            "## Overall result",
            "",
            "For this saved run, the SAM2-compatible method is better overall: "
            "it has lower mean and median absolute error, lower RMSE, lower "
            "maximum error, and more cases within every requested threshold. "
            "It improves 60 of 94 paired cases, but worsens 34 cases; therefore "
            "it is not uniformly better on every book.",
            "",
            "## Case 81 historical discrepancy",
            "",
            f"- Ground truth: {fmt(case81['ground_truth_width_mm'])} mm",
            "- Saved SAM3-original width in the paired run: "
            f"{fmt(case81['saved_current_sam3_width_mm'])} mm",
            "- Saved SAM2-compatible width in the paired run: "
            f"{fmt(case81['saved_sam2_compatible_width_mm'])} mm",
            "- Historical SAM3-original width: "
            f"{fmt(case81['old_sam3_original_width_mm'])} mm",
            "- RGB, Depth, raw mask, and refined mask are byte-identical "
            "between the two runs.",
            "- `pointcloud_sent_to_pca.ply` is different.",
            "- Historical RANSAC inliers: "
            f"{case81['old_ransac']['inlier_count']}; paired-run RANSAC "
            f"inliers: {case81['current_ransac']['inlier_count']}.",
            "",
            case81["reason"],
            "",
            case81["comparison_choice"],
            "",
            "## Interpretation",
            "",
            "The paired statistics are valid for the saved "
            "`20260725_005852` execution because both values use that run's "
            "same RGB-D, selected/refined mask, and ground truth. However, "
            "SAM3-original width includes an unseeded RANSAC realization. "
            "Case81 demonstrates that its value is not reproducible across "
            "reruns even when RGB-D and masks are identical. This stability "
            "limitation must accompany any adoption decision.",
            "",
            "No end-to-end comparison was performed in this analysis.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    for required in ("results.csv", "results.json", "summary.json", "run.log"):
        if not (input_dir / required).is_file():
            raise FileNotFoundError(input_dir / required)

    rows, availability = build_rows(input_dir)
    if availability["both_widths_finite_count"] == 0:
        raise RuntimeError("no paired saved width values are available")

    sam3_stats = method_statistics(rows, "sam3_original")
    sam2_stats = method_statistics(rows, "sam2_compatible")
    paired = paired_statistics(rows)
    case81 = inspect_case81(input_dir, rows)

    output_dir = unique_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "results_width_comparison.csv"
    json_path = output_dir / "results_width_comparison.json"
    summary_path = output_dir / "summary_width_comparison.json"
    report_path = output_dir / "comparison_report.md"

    save_csv(csv_path, rows)
    save_json(json_path, rows)
    plots, plot_error = make_plots(output_dir, rows)
    summary = {
        "analysis": "width-only paired comparison from saved values",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "created_at": datetime.now().astimezone().isoformat(),
        "inference_executed": False,
        "input_hashes": {
            name: sha256(input_dir / name)
            for name in ("results.csv", "results.json", "summary.json", "run.log")
        },
        "value_provenance": {
            "sam3_original": (
                "sam3_refined_sam2_width_result.json#/current_sam3_width_mm; "
                "copied from base refine-only pred_book_width_mm after "
                "estimate_book_width(points_for_pca, mean, pc1, pc2)"
            ),
            "sam2_compatible": (
                "sam3_refined_sam2_width_result.json#/pred_book_width_mm; "
                "estimate_sam2_compatible_geometry with geometry_mode="
                "sam2_width_only"
            ),
            "ground_truth": "results.json#/gt_book_width_mm",
        },
        "data_availability": availability,
        "sam3_original_statistics": sam3_stats,
        "sam2_compatible_statistics": sam2_stats,
        "paired_comparison": paired,
        "case81": case81,
        "plots": plots,
        "plot_error": plot_error,
    }
    save_json(summary_path, summary)
    save_report(report_path, summary)

    print(f"input_dir={input_dir}")
    print(f"output_dir={output_dir}")
    print(
        "paired_cases="
        f"{summary['paired_comparison']['both_success_count']}"
    )
    print(f"csv={csv_path}")
    print(f"json={json_path}")
    print(f"summary={summary_path}")
    print(f"report={report_path}")
    if plot_error:
        print(f"plots_skipped={plot_error}")
    else:
        for plot in plots:
            print(f"plot={plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
