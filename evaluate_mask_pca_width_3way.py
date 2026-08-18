#!/usr/bin/env python3
"""Compare three mask-PCA width aggregations from one set of saved artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detection.pro_handbook.sam_py_demo.modules.sam2_compatible_geometry import (
    MASK_PCA_WIDTH_MODE_GLOBAL,
    MASK_PCA_WIDTH_MODE_SLICE_MINMAX,
    MASK_PCA_WIDTH_MODE_SLICE_P2P98,
    MASK_PCA_WIDTH_MODES,
    estimate_mask_pca_width_modes,
)


MODE_LABELS = {
    MASK_PCA_WIDTH_MODE_GLOBAL: "MASK_PCA_GLOBAL_P2P98",
    MASK_PCA_WIDTH_MODE_SLICE_P2P98: "MASK_PCA_SLICE_P2P98_MEDIAN",
    MASK_PCA_WIDTH_MODE_SLICE_MINMAX: "MASK_PCA_SLICE_MINMAX_MEDIAN",
}
PAIR_TOLERANCE_MM = 1e-6
DEPTH_FILENAMES = (
    "debug_refine_only_median_depth_filter_depth_masked.npy",
    "refine_only_median_depth_filter_depth_masked.npy",
)


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    candidate = base / stem
    if not candidate.exists():
        candidate.mkdir()
        return candidate
    for index in range(1, 1000):
        candidate = base / f"{stem}_{index:03d}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
    raise RuntimeError(f"cannot create unique comparison directory under {base}")


def load_master(path: Path) -> list[dict]:
    values = load_json(path)
    if len(values) < 20:
        raise ValueError(f"master must contain at least 20 books: {path}")
    return [
        {
            "book_name": str(item["book_name"]),
            "ground_truth_width_mm": float(item["book_width"]),
        }
        for item in values
    ]


def depth_path(case_dir: Path) -> Path:
    for name in DEPTH_FILENAMES:
        candidate = case_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"masked depth is missing under {case_dir}: {DEPTH_FILENAMES}"
    )


def load_width_inputs(case_dir: Path):
    mask_path = case_dir / "selected_mask_refined.png"
    camera_path = case_dir / "camera_params.json"
    rgb_path = case_dir / "after_init_rgb.png"
    selected_depth_path = depth_path(case_dir)
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if mask_image is None:
        raise FileNotFoundError(mask_path)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    depth = np.load(selected_depth_path, allow_pickle=False)
    camera = load_json(camera_path)
    hashes = {
        "rgb_sha256": sha256(rgb_path),
        "mask_sha256": sha256(mask_path),
        "depth_sha256": sha256(selected_depth_path),
        "camera_sha256": sha256(camera_path),
    }
    return mask_image > 0, depth, camera, rgb, hashes, selected_depth_path


def source_failure(case_dir: Path):
    path = case_dir / "pointcloud_debug_result.json"
    if not path.is_file():
        return None
    value = load_json(path)
    if value.get("success", False):
        return None
    return {
        "failure_stage": value.get("failure_stage"),
        "failure_reason": value.get("error") or "saved preprocessing failed",
    }


def mode_row_fields(prefix: str, value: dict, gt: float) -> dict:
    ok = bool(value.get("ok")) and value.get("width_mm") is not None
    width = float(value["width_mm"]) if ok else None
    error = None if width is None else float(width - gt)
    return {
        f"{prefix}_ok": ok,
        f"{prefix}_width_mm": width,
        f"{prefix}_error_mm": error,
        f"{prefix}_abs_error_mm": None if error is None else abs(error),
        f"{prefix}_pixel_width": value.get("pixel_width"),
        f"{prefix}_valid_slice_count": value.get("valid_slice_count"),
        f"{prefix}_slice_width_mean": value.get("slice_width_mean_px"),
        f"{prefix}_slice_width_std": value.get("slice_width_std_px"),
        f"{prefix}_slice_width_cv": value.get("slice_width_cv"),
        f"{prefix}_slice_width_min": value.get("slice_width_min_px"),
        f"{prefix}_slice_width_max": value.get("slice_width_max_px"),
    }


def method_summary(rows: list[dict], mode: str, total_count: int) -> dict:
    prefix = mode
    successful = [row for row in rows if row[f"{prefix}_ok"]]
    predicted = np.asarray(
        [row[f"{prefix}_width_mm"] for row in successful], np.float64
    )
    ground_truth = np.asarray(
        [row["ground_truth_width_mm"] for row in successful], np.float64
    )
    signed = predicted - ground_truth
    absolute = np.abs(signed)
    if not successful:
        return {
            "mode": mode,
            "method_name": MODE_LABELS[mode],
            "evaluated_case_count": total_count,
            "success_count": 0,
            "failure_count": total_count,
            "success_rate": 0.0,
        }
    max_index = int(np.argmax(absolute))
    thresholds = {}
    for threshold in (1.0, 1.5, 2.0):
        count = int(np.sum(absolute <= threshold))
        thresholds[f"within_{threshold:.1f}_mm"] = {
            "count": count,
            "rate_of_successes": count / len(successful),
            "rate_of_all_cases": count / total_count,
        }
    return {
        "mode": mode,
        "method_name": MODE_LABELS[mode],
        "evaluated_case_count": total_count,
        "success_count": len(successful),
        "failure_count": total_count - len(successful),
        "success_rate": len(successful) / total_count,
        "mean_predicted_width_mm": float(np.mean(predicted)),
        "mean_ground_truth_width_mm": float(np.mean(ground_truth)),
        "mean_signed_error_mm": float(np.mean(signed)),
        "mean_absolute_error_mm": float(np.mean(absolute)),
        "median_signed_error_mm": float(np.median(signed)),
        "median_absolute_error_mm": float(np.median(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(signed ** 2))),
        "signed_error_std_mm": float(np.std(signed)),
        "absolute_error_p90_mm": float(np.percentile(absolute, 90.0)),
        "absolute_error_p95_mm": float(np.percentile(absolute, 95.0)),
        "max_absolute_error_mm": float(absolute[max_index]),
        "max_absolute_error_case": int(successful[max_index]["case_no"]),
        "thresholds": thresholds,
        "overestimate_count": int(np.sum(signed > 0.0)),
        "underestimate_count": int(np.sum(signed < 0.0)),
        "overestimate_gt_2_mm_count": int(np.sum(signed > 2.0)),
        "underestimate_lt_minus_2_mm_count": int(np.sum(signed < -2.0)),
        "overestimate_ge_5_mm_count": int(np.sum(signed >= 5.0)),
        "top_10_absolute_error_cases": [
            {
                "case_no": int(successful[index]["case_no"]),
                "absolute_error_mm": float(absolute[index]),
                "signed_error_mm": float(signed[index]),
            }
            for index in np.argsort(-absolute)[:10]
        ],
    }


def pairwise(rows: list[dict], first: str, second: str) -> dict:
    paired = [row for row in rows if row[f"{first}_ok"] and row[f"{second}_ok"]]
    details = []
    for row in paired:
        first_error = float(row[f"{first}_abs_error_mm"])
        second_error = float(row[f"{second}_abs_error_mm"])
        improvement = first_error - second_error
        details.append({
            "case_no": int(row["case_no"]),
            "first_abs_error_mm": first_error,
            "second_abs_error_mm": second_error,
            "improvement_mm": improvement,
        })
    return {
        "first": first,
        "second": second,
        "paired_count": len(details),
        "second_improved_count": sum(
            item["improvement_mm"] > PAIR_TOLERANCE_MM for item in details
        ),
        "second_worsened_count": sum(
            item["improvement_mm"] < -PAIR_TOLERANCE_MM for item in details
        ),
        "same_count": sum(
            abs(item["improvement_mm"]) <= PAIR_TOLERANCE_MM for item in details
        ),
        "improved_by_at_least_0_5_mm": [
            item for item in details if item["improvement_mm"] >= 0.5
        ],
        "worsened_by_at_least_0_5_mm": [
            item for item in details if item["improvement_mm"] <= -0.5
        ],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw_slice_visualization(
    output_path: Path,
    rgb: np.ndarray,
    mask: np.ndarray,
    geometry: dict,
    mode: str,
    case_no: str,
) -> None:
    mode_result = geometry["modes"][mode]
    axis = np.asarray(geometry["mask_pca_axis_uv"], np.float64)
    normal = np.asarray(geometry["width_axis_uv"], np.float64)
    center = np.asarray(geometry["mask_center_xy"], np.float64)
    image = rgb.copy()
    green = image.copy()
    green[mask] = (0, 255, 0)
    image = cv2.addWeighted(image, 0.70, green, 0.30, 0)

    axis_scale = 220.0
    for vector, color in ((axis, (255, 0, 0)), (normal, (0, 255, 255))):
        p0 = tuple(np.rint(center - vector * axis_scale).astype(int))
        p1 = tuple(np.rint(center + vector * axis_scale).astype(int))
        cv2.line(image, p0, p1, color, 2, cv2.LINE_AA)
    slices = mode_result.get("slices") or []
    if slices:
        boundaries = [item["long_start_px"] for item in slices]
        boundaries.append(slices[-1]["long_end_px"])
        for value in boundaries:
            midpoint = center + axis * float(value)
            p0 = tuple(np.rint(midpoint - normal * 75.0).astype(int))
            p1 = tuple(np.rint(midpoint + normal * 75.0).astype(int))
            cv2.line(image, p0, p1, (128, 128, 128), 1, cv2.LINE_AA)
        for item in slices:
            midpoint = center + axis * float(item["long_center_px"])
            if not item["valid"]:
                cv2.drawMarker(
                    image,
                    tuple(np.rint(midpoint).astype(int)),
                    (0, 0, 255),
                    cv2.MARKER_TILTED_CROSS,
                    8,
                    1,
                    cv2.LINE_AA,
                )
                continue
            p0 = tuple(np.rint(
                midpoint + normal * float(item["width_low_px"])
            ).astype(int))
            p1 = tuple(np.rint(
                midpoint + normal * float(item["width_high_px"])
            ).astype(int))
            cv2.line(image, p0, p1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(
                image,
                f"{float(item['width_px']):.1f}",
                p1,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    lines = [
        f"case: {case_no}",
        f"mode: {mode}",
        f"valid slices: {mode_result.get('valid_slice_count')}",
        f"median width: {mode_result.get('pixel_width'):.3f} px",
        f"estimated width: {mode_result.get('width_mm'):.3f} mm",
    ]
    for index, line in enumerate(lines):
        y = 26 + index * 24
        cv2.putText(
            image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            (255, 255, 255), 3, cv2.LINE_AA
        )
        cv2.putText(
            image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            (0, 0, 0), 1, cv2.LINE_AA
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"failed to save visualization: {output_path}")


def evaluate_known_case(case_dir: Path, output_dir: Path, source_hashes: dict):
    mask, depth, camera, rgb, hashes, _ = load_width_inputs(case_dir)
    geometry = estimate_mask_pca_width_modes(mask, depth, camera)
    value = {
        "case_dir": str(case_dir),
        "ground_truth_width_mm": 11.5,
        "hashes": hashes,
        "matching_100_case_numbers": [
            case_no for case_no, other in source_hashes.items()
            if hashes["rgb_sha256"] == other.get("rgb_sha256")
            and hashes["mask_sha256"] == other.get("mask_sha256")
        ],
        "geometry": geometry,
        "results": {},
    }
    for mode in MASK_PCA_WIDTH_MODES:
        item = geometry["modes"][mode]
        width = item.get("width_mm")
        value["results"][mode] = {
            "ok": item.get("ok"),
            "width_mm": width,
            "signed_error_mm": None if width is None else float(width - 11.5),
            "absolute_error_mm": None if width is None else abs(float(width - 11.5)),
            "valid_slice_count": item.get("valid_slice_count"),
        }
    save_json(output_dir / "known_case_20260806_111513.json", value)
    for mode in (MASK_PCA_WIDTH_MODE_SLICE_P2P98, MASK_PCA_WIDTH_MODE_SLICE_MINMAX):
        draw_slice_visualization(
            output_dir / "visualizations" / "known_case" / f"{mode}.png",
            rgb,
            mask,
            geometry,
            mode,
            "known_20260806_111513",
        )
    return value


def run_three_way_comparison(
    source_dir: Path,
    master_json: Path,
    output_base: Path,
    *,
    start_case: int = 1,
    end_case: int = 100,
    known_case_dir: Path | None = None,
) -> Path:
    source_dir = Path(source_dir).expanduser().resolve()
    master_json = Path(master_json).expanduser().resolve()
    run_dir = unique_run_dir(Path(output_base).expanduser().resolve())
    comparison_dir = run_dir / "comparison"
    method_dirs = {mode: run_dir / mode for mode in MASK_PCA_WIDTH_MODES}
    for path in [comparison_dir, *method_dirs.values()]:
        path.mkdir(parents=True, exist_ok=True)
    books = load_master(master_json)
    run_config = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_dir": str(source_dir),
        "master_json": str(master_json),
        "master_json_sha256": sha256(master_json),
        "start_case": int(start_case),
        "end_case": int(end_case),
        "inference_executed": False,
        "saved_masks_reused": True,
        "modes": list(MASK_PCA_WIDTH_MODES),
        "method_names": MODE_LABELS,
        "shared_axis": "all nonzero refined-mask pixels, covariance, eigh, principal eigenvector",
        "slice_bin_size_px": 8.0,
        "min_pixels_in_slice": 12,
        "min_valid_slice_count": 5,
        "end_exclusion_px": 0.0,
        "slice_aggregation": "median",
        "pair_tolerance_mm": PAIR_TOLERANCE_MM,
    }
    save_json(comparison_dir / "run_config.json", run_config)

    rows = []
    method_results = {mode: [] for mode in MASK_PCA_WIDTH_MODES}
    source_hashes = {}
    run_log = run_dir / "run.log"
    for case_no in range(start_case, end_case + 1):
        case_dir = source_dir / str(case_no)
        book = books[(case_no - 1) // 5]
        gt = float(book["ground_truth_width_mm"])
        row = {
            "case_no": int(case_no),
            "book_name": book["book_name"],
            "ground_truth_width_mm": gt,
        }
        failure = source_failure(case_dir)
        geometry = None
        hashes = {}
        try:
            if failure is not None:
                raise RuntimeError(failure["failure_reason"])
            mask, depth, camera, _, hashes, selected_depth_path = load_width_inputs(
                case_dir
            )
            geometry = estimate_mask_pca_width_modes(mask, depth, camera)
            source_hashes[case_no] = hashes
            row.update({
                "mask_pca_angle_deg": geometry["mask_pca_axis_angle_deg"],
                "mask_pca_eigenvalue_ratio": geometry[
                    "mask_pca_eigenvalue_ratio"
                ],
                "mask_pixel_count": geometry["mask_pca_pixel_count"],
                "median_depth": geometry["representative_depth_m"],
                "mask_sha256": hashes["mask_sha256"],
                "depth_sha256": hashes["depth_sha256"],
                "depth_source": str(selected_depth_path),
                "failure_reason": "",
            })
            for mode in MASK_PCA_WIDTH_MODES:
                row.update(mode_row_fields(mode, geometry["modes"][mode], gt))
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            row.update({
                "mask_pca_angle_deg": None,
                "mask_pca_eigenvalue_ratio": None,
                "mask_pixel_count": None,
                "median_depth": None,
                "mask_sha256": hashes.get("mask_sha256"),
                "depth_sha256": hashes.get("depth_sha256"),
                "depth_source": None,
                "failure_reason": reason,
            })
            for mode in MASK_PCA_WIDTH_MODES:
                row.update(mode_row_fields(mode, {"ok": False}, gt))

        successful_errors = {
            mode: row[f"{mode}_abs_error_mm"]
            for mode in MASK_PCA_WIDTH_MODES
            if row[f"{mode}_ok"]
        }
        if successful_errors:
            best_error = min(successful_errors.values())
            best_modes = [
                mode for mode, error in successful_errors.items()
                if abs(error - best_error) <= PAIR_TOLERANCE_MM
            ]
            row["best_method"] = ";".join(best_modes)
            row["best_abs_error_mm"] = float(best_error)
        else:
            row["best_method"] = ""
            row["best_abs_error_mm"] = None
        rows.append(row)

        for mode in MASK_PCA_WIDTH_MODES:
            value = {
                "case_no": int(case_no),
                "book_name": book["book_name"],
                "ground_truth_width_mm": gt,
                "mode": mode,
                "method_name": MODE_LABELS[mode],
                "source_case_dir": str(case_dir),
                "hashes": hashes,
                "mask_pca": None if geometry is None else {
                    "axis_uv": geometry["mask_pca_axis_uv"],
                    "angle_deg": geometry["mask_pca_axis_angle_deg"],
                    "eigenvalues": geometry["mask_pca_eigenvalues"],
                    "eigenvalue_ratio": geometry["mask_pca_eigenvalue_ratio"],
                    "center_xy": geometry["mask_center_xy"],
                    "pixel_count": geometry["mask_pca_pixel_count"],
                },
                "result": None if geometry is None else geometry["modes"][mode],
                "failure_reason": row["failure_reason"],
            }
            save_json(method_dirs[mode] / str(case_no) / "width_result.json", value)
            method_results[mode].append(value)
        with run_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "case_no": case_no,
                "status": "success" if successful_errors else "fail",
                "failure_reason": row["failure_reason"],
            }, ensure_ascii=False) + "\n")
        print(
            f"case {case_no:03d}: "
            f"{'success' if successful_errors else row['failure_reason']}"
        )

    summaries = {
        mode: method_summary(rows, mode, end_case - start_case + 1)
        for mode in MASK_PCA_WIDTH_MODES
    }
    for mode in MASK_PCA_WIDTH_MODES:
        save_json(method_dirs[mode] / "results.json", method_results[mode])
        save_json(method_dirs[mode] / "summary.json", summaries[mode])
        method_csv_rows = []
        for row in rows:
            method_csv_rows.append({
                key: value for key, value in row.items()
                if key in {"case_no", "book_name", "ground_truth_width_mm", "failure_reason"}
                or key.startswith(f"{mode}_")
            })
        write_csv(method_dirs[mode] / "results.csv", method_csv_rows)

    pairwise_results = {
        "global_vs_slice_p2p98": pairwise(
            rows, MASK_PCA_WIDTH_MODE_GLOBAL, MASK_PCA_WIDTH_MODE_SLICE_P2P98
        ),
        "global_vs_slice_minmax": pairwise(
            rows, MASK_PCA_WIDTH_MODE_GLOBAL, MASK_PCA_WIDTH_MODE_SLICE_MINMAX
        ),
        "slice_p2p98_vs_slice_minmax": pairwise(
            rows,
            MASK_PCA_WIDTH_MODE_SLICE_P2P98,
            MASK_PCA_WIDTH_MODE_SLICE_MINMAX,
        ),
    }
    all_success = [
        row for row in rows
        if all(row[f"{mode}_ok"] for mode in MASK_PCA_WIDTH_MODES)
    ]
    all_over_2 = [
        int(row["case_no"]) for row in all_success
        if all(row[f"{mode}_abs_error_mm"] > 2.0 for mode in MASK_PCA_WIDTH_MODES)
    ]
    success_sets = {
        mode: [int(row["case_no"]) for row in rows if row[f"{mode}_ok"]]
        for mode in MASK_PCA_WIDTH_MODES
    }
    shared_success_failure = len({tuple(value) for value in success_sets.values()}) == 1
    shared_pca_consistent = all(
        row["mask_pca_angle_deg"] is not None for row in all_success
    )
    summary = {
        "run_config": run_config,
        "method_summaries": summaries,
        "pairwise_comparisons": pairwise_results,
        "success_case_sets": success_sets,
        "success_failure_sets_identical": shared_success_failure,
        "shared_pca_by_construction": True,
        "shared_pca_values_present_for_all_common_successes": shared_pca_consistent,
        "all_three_abs_error_gt_2_mm_cases": all_over_2,
        "best_method_cases": [
            {"case_no": row["case_no"], "best_method": row["best_method"]}
            for row in all_success
        ],
    }
    save_json(comparison_dir / "comparison_summary.json", summary)
    write_csv(comparison_dir / "per_case_comparison.csv", rows)
    summary_rows = []
    for mode, value in summaries.items():
        flat = {key: item for key, item in value.items() if key not in {
            "thresholds", "top_10_absolute_error_cases"
        }}
        for key, item in value.get("thresholds", {}).items():
            flat[f"{key}_count"] = item["count"]
            flat[f"{key}_rate_of_successes"] = item["rate_of_successes"]
            flat[f"{key}_rate_of_all_cases"] = item["rate_of_all_cases"]
        summary_rows.append(flat)
    write_csv(comparison_dir / "comparison_summary.csv", summary_rows)
    failed = [
        {"case_no": row["case_no"], "failure_reason": row["failure_reason"]}
        for row in rows if row["failure_reason"]
    ]
    save_json(comparison_dir / "failed_cases.json", {
        "success_failure_sets_identical": shared_success_failure,
        "failures": failed,
    })
    save_json(comparison_dir / "improved_worsened_cases.json", {
        "pairwise": pairwise_results,
        "all_three_abs_error_gt_2_mm_cases": all_over_2,
        "top_10_by_method": {
            mode: summaries[mode].get("top_10_absolute_error_cases", [])
            for mode in MASK_PCA_WIDTH_MODES
        },
        "best_method_cases": summary["best_method_cases"],
    })

    selected_visual_cases = {}
    if all_success:
        best_b = max(
            all_success,
            key=lambda row: row[f"{MASK_PCA_WIDTH_MODE_GLOBAL}_abs_error_mm"]
            - row[f"{MASK_PCA_WIDTH_MODE_SLICE_P2P98}_abs_error_mm"],
        )
        worst_b = min(
            all_success,
            key=lambda row: row[f"{MASK_PCA_WIDTH_MODE_GLOBAL}_abs_error_mm"]
            - row[f"{MASK_PCA_WIDTH_MODE_SLICE_P2P98}_abs_error_mm"],
        )
        largest_bc = max(
            all_success,
            key=lambda row: abs(
                row[f"{MASK_PCA_WIDTH_MODE_SLICE_P2P98}_width_mm"]
                - row[f"{MASK_PCA_WIDTH_MODE_SLICE_MINMAX}_width_mm"]
            ),
        )
        selected_visual_cases = {
            int(best_b["case_no"]): ["largest_B_improvement_vs_A"],
            int(worst_b["case_no"]): ["largest_B_worsening_vs_A"],
            int(largest_bc["case_no"]): ["largest_B_C_width_difference"],
        }
        for mode in MASK_PCA_WIDTH_MODES:
            case_no = int(summaries[mode]["max_absolute_error_case"])
            selected_visual_cases.setdefault(case_no, []).append(
                f"max_absolute_error_{mode}"
            )
        for case_no, reasons in selected_visual_cases.items():
            case_dir = source_dir / str(case_no)
            mask, depth, camera, rgb, _, _ = load_width_inputs(case_dir)
            geometry = estimate_mask_pca_width_modes(mask, depth, camera)
            for mode in (
                MASK_PCA_WIDTH_MODE_SLICE_P2P98,
                MASK_PCA_WIDTH_MODE_SLICE_MINMAX,
            ):
                draw_slice_visualization(
                    comparison_dir / "visualizations" / f"case_{case_no:03d}"
                    / f"{mode}.png",
                    rgb,
                    mask,
                    geometry,
                    mode,
                    str(case_no),
                )
            save_json(
                comparison_dir / "visualizations" / f"case_{case_no:03d}"
                / "selection_reason.json",
                {"case_no": case_no, "reasons": reasons},
            )
    save_json(
        comparison_dir / "visualizations" / "selected_cases.json",
        selected_visual_cases,
    )

    if known_case_dir is not None:
        known = evaluate_known_case(
            Path(known_case_dir).expanduser().resolve(),
            comparison_dir,
            source_hashes,
        )
        summary["known_case"] = known
        save_json(comparison_dir / "comparison_summary.json", summary)

    print(f"THREE_WAY_RUN_ROOT={run_dir}")
    return run_dir


if __name__ == "__main__":
    raise SystemExit(
        "Use offline_pointcloud_debug_SAM3.py --three-way-compare so the "
        "formal evaluator records its input and master configuration."
    )
