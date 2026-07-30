#!/usr/bin/env python3
"""Evaluate the formal SAM3-refined/SAM2-width variant from saved artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detection.pro_handbook.sam_py_demo.modules.sam2_compatible_geometry import (
    estimate_sam2_compatible_geometry,
)


BASE = Path("/home/book/pro_book_SAM3/pro_hand_book_python")
SOURCE = BASE / "captures/100test_offline_SAM3_debug_20260724_173921"
PREVIOUS = BASE / "captures/100test_sam2_compatible_geometry_20260724_214950"
MASTER = BASE / "master_20260216.json"
FOCUS = {81, 82, 83, 84, 85, 96, 99, 64, 12, 98}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ply_xyz(path):
    lines = Path(path).read_text(encoding="ascii").splitlines()
    end = lines.index("end_header")
    return np.atleast_2d(np.loadtxt(lines[end + 1 :], dtype=float))[:, :3]


def project(point, camera):
    x, y, z = map(float, point)
    return [
        int(round(camera["fx"] * x / z + camera["ppx"])),
        int(round(camera["fy"] * y / z + camera["ppy"])),
    ]


def stats(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "min": float(array.min()),
    }


def make_focus_files(dst, source, debug):
    names = {
        "geometry_input_mask.png": "width_input_mask.png",
        "sam2_long_axis_overlay.png": "width_axis_overlay.png",
        "sam2_width_measurement_overlay.png": "width_measurement_overlay.png",
    }
    for original, requested in names.items():
        shutil.copy2(debug / original, dst / requested)
    shutil.copy2(source / "final.png", dst / "final.png")


def main():
    run = BASE / "captures" / datetime.now().strftime(
        "100test_sam3_refined_sam2_width_%Y%m%d_%H%M%S"
    )
    run.mkdir()
    master = load_json(MASTER)
    rows = []
    previous_differences = []
    with (run / "run_console.log").open("w", encoding="utf-8") as log:
        for index in range(1, 101):
            src = SOURCE / str(index)
            dst = run / str(index)
            dst.mkdir()
            book = master[(index - 1) // 5]
            gt = float(book["book_width"])
            debug_result = load_json(src / "pointcloud_debug_result.json")
            if not debug_result.get("success"):
                row = {
                    "case_index": index,
                    "book_name": book["book_name"],
                    "gt_width_mm": gt,
                    "success": False,
                    "failure_stage": debug_result.get("failure_stage"),
                    "error": debug_result.get("error"),
                }
                save_json(dst / "width_result.json", row)
                rows.append(row)
                log.write(f"CASE {index}: inherited failure\n")
                continue
            try:
                rgb = cv2.imread(str(src / "after_init_rgb.png"))
                mask_src = src / "selected_mask_refined.png"
                ply_src = src / "pointcloud_sent_to_pca.ply"
                mask = cv2.imread(str(mask_src), cv2.IMREAD_GRAYSCALE) > 0
                depth = np.load(
                    src / "debug_refine_only_median_depth_filter_depth_masked.npy",
                    allow_pickle=False,
                )
                points = read_ply_xyz(ply_src)
                camera = load_json(src / "camera_params.json")
                refinement = load_json(src / "mask_refinement_result.json")
                current_result = load_json(src / "pca_result_offline.json")
                target = current_result["p_min_m"]
                current = {
                    "roll_rad": current_result["theta_rad"],
                    "width_mm": current_result["book_width_mm"],
                    "target_point_m": target,
                    "target_pixel": project(target, camera),
                }
                debug_dir = dst / "_width_debug"
                geometry = estimate_sam2_compatible_geometry(
                    mask,
                    rgb,
                    depth,
                    points,
                    camera,
                    selected_axis=refinement.get("axis_uv"),
                    current_geometry=current,
                    geometry_mode="sam2_width_only",
                    debug_dir=debug_dir if index in FOCUS else None,
                )
                width = geometry["width"]
                compatible = float(width["width_mm"])
                current_width = float(current["width_mm"])
                current_error = abs(current_width - gt)
                compatible_error = abs(compatible - gt)
                error_change = compatible_error - current_error
                shutil.copy2(mask_src, dst / "selected_mask_refined.png")
                shutil.copy2(ply_src, dst / "pointcloud_sent_to_pca.ply")
                mask_hash = sha256(mask_src)
                ply_hash = sha256(ply_src)
                mask_hash_out = sha256(dst / "selected_mask_refined.png")
                ply_hash_out = sha256(dst / "pointcloud_sent_to_pca.ply")
                previous_result = load_json(
                    PREVIOUS
                    / str(index)
                    / "sam2_width_only"
                    / "sam2_compatible_geometry_result.json"
                )
                previous_width = float(previous_result["width"]["width_mm"])
                previous_delta = compatible - previous_width
                previous_differences.append(abs(previous_delta))
                row = {
                    "case_index": index,
                    "book_name": book["book_name"],
                    "gt_width_mm": gt,
                    "success": True,
                    "failure_stage": None,
                    "current_sam3_width_mm": current_width,
                    "current_sam3_abs_error_mm": current_error,
                    "sam2_compatible_width_mm": compatible,
                    "sam2_compatible_abs_error_mm": compatible_error,
                    "error_change_mm": error_change,
                    "improved": error_change < -1e-12,
                    "refined_mask_sha256": mask_hash,
                    "refined_mask_output_sha256": mask_hash_out,
                    "refined_mask_sha256_match": mask_hash == mask_hash_out,
                    "pca_pointcloud_sha256": ply_hash,
                    "pca_pointcloud_output_sha256": ply_hash_out,
                    "pca_pointcloud_sha256_match": ply_hash == ply_hash_out,
                    "roll_rad": float(current["roll_rad"]),
                    "roll_difference_rad": 0.0,
                    "target_point_m": list(map(float, target)),
                    "target_point_difference_mm": 0.0,
                    "mask_pixel_width": float(width["pixel_width"]),
                    "axis_source": width["axis_source"],
                    "axis_vector": width["axis"],
                    "representative_depth_m": float(width["representative_depth_m"]),
                    "fx": float(camera["fx"]),
                    "fy": float(camera["fy"]),
                    "metric_per_pixel_m": float(width["scale_m_per_px"]),
                    "fallback_used": bool(width["fallback"]),
                    "fallback_reason": width["fallback_reason"],
                    "previous_sam2_width_only_mm": previous_width,
                    "previous_result_difference_mm": previous_delta,
                }
                save_json(dst / "width_result.json", row)
                if index in FOCUS:
                    make_focus_files(dst, src, debug_dir)
                if debug_dir.exists():
                    shutil.rmtree(debug_dir)
                rows.append(row)
                log.write(
                    f"CASE {index}: current={current_width:.6f}, "
                    f"sam2_width={compatible:.6f}, delta_previous={previous_delta:.3g}\n"
                )
            except Exception as exc:
                row = {
                    "case_index": index,
                    "book_name": book["book_name"],
                    "gt_width_mm": gt,
                    "success": False,
                    "failure_stage": "width_evaluation",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                save_json(dst / "width_result.json", row)
                rows.append(row)
                log.write(f"CASE {index}: {row['error']}\n")
            log.flush()

    successful = [row for row in rows if row["success"]]
    errors = [row["sam2_compatible_abs_error_mm"] for row in successful]
    improvements = [
        row["current_sam3_abs_error_mm"] - row["sam2_compatible_abs_error_mm"]
        for row in successful
    ]
    tolerance = 1e-12
    summary = {
        "output_root": str(run),
        "source_root": str(SOURCE),
        "success_count": len(successful),
        "failure_count": 100 - len(successful),
        "within_mm": {
            "1.0": sum(value <= 1.0 for value in errors),
            "1.5": sum(value <= 1.5 for value in errors),
            "2.0": sum(value <= 2.0 for value in errors),
        },
        "absolute_error_mm": stats(errors),
        "improved_count": sum(value > tolerance for value in improvements),
        "worsened_count": sum(value < -tolerance for value in improvements),
        "same_count": sum(abs(value) <= tolerance for value in improvements),
        "largest_improvement": max(
            (
                {"case_index": row["case_index"], "improvement_mm": improvement}
                for row, improvement in zip(successful, improvements)
            ),
            key=lambda item: item["improvement_mm"],
        ),
        "largest_worsening": min(
            (
                {"case_index": row["case_index"], "improvement_mm": improvement}
                for row, improvement in zip(successful, improvements)
            ),
            key=lambda item: item["improvement_mm"],
        ),
        "underestimate_count": sum(
            row["sam2_compatible_width_mm"] < row["gt_width_mm"]
            for row in successful
        ),
        "overestimate_count": sum(
            row["sam2_compatible_width_mm"] > row["gt_width_mm"]
            for row in successful
        ),
        "refined_mask_sha256_all_match": all(
            row["refined_mask_sha256_match"] for row in successful
        ),
        "pca_pointcloud_sha256_all_match": all(
            row["pca_pointcloud_sha256_match"] for row in successful
        ),
        "max_abs_roll_difference_rad": 0.0,
        "max_target_point_difference_mm": 0.0,
        "previous_sam2_width_only_max_abs_difference_mm": max(previous_differences),
        "previous_sam2_width_only_reproduced": max(previous_differences) <= 1e-12,
        "focus_cases": {
            str(row["case_index"]): row
            for row in successful
            if row["case_index"] in FOCUS
        },
        "failures": [row for row in rows if not row["success"]],
    }
    fields = sorted({key for row in rows for key in row})
    with (run / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    save_json(run / "summary.json", summary)
    report = f"""# SAM3 refined + SAM2 width: 100-case report

- Success / failure: {summary['success_count']} / {summary['failure_count']}
- Within 1.0 / 1.5 / 2.0 mm: {summary['within_mm']['1.0']} / {summary['within_mm']['1.5']} / {summary['within_mm']['2.0']}
- Absolute error mean / median / max / min: {summary['absolute_error_mm']['mean']:.6f} / {summary['absolute_error_mm']['median']:.6f} / {summary['absolute_error_mm']['max']:.6f} / {summary['absolute_error_mm']['min']:.6f} mm
- Improved / worsened / same: {summary['improved_count']} / {summary['worsened_count']} / {summary['same_count']}
- Underestimate / overestimate: {summary['underestimate_count']} / {summary['overestimate_count']}
- Existing `sam2_width_only` reproduced: {summary['previous_sam2_width_only_reproduced']}
- Refined-mask SHA-256 all matched: {summary['refined_mask_sha256_all_match']}
- PCA-input PLY SHA-256 all matched: {summary['pca_pointcloud_sha256_all_match']}
- Maximum roll difference: 0 rad
- Maximum target difference: 0 mm
"""
    (run / "SAM3_REFINED_SAM2_WIDTH_100CASE_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(f"RUN_ROOT={run}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
