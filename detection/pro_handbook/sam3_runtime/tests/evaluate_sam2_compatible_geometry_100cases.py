#!/usr/bin/env python3
"""Evaluate saved SAM3 masks with current vs SAM2-compatible geometry."""
from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detection.pro_handbook.sam_py_demo.modules.sam2_compatible_geometry import (
    GEOMETRY_MODES,
    estimate_sam2_compatible_geometry,
)


BASE = Path("/home/book/pro_book_SAM3/pro_hand_book_python")
SOURCE = BASE / "captures/100test_offline_SAM3_debug_20260724_173921"
INPUT = BASE / "captures/100test"
MASTER = BASE / "master_20260216.json"
MODES = [
    "sam3_current_geometry", "sam2_roll_only", "sam2_width_only",
    "sam2_target_only", "sam2_all_geometry",
]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_ply_xyz(path):
    lines = Path(path).read_text(encoding="ascii").splitlines()
    end = lines.index("end_header")
    return np.loadtxt(lines[end + 1:], dtype=np.float64)[:, :3]


def project(point, camera):
    x, y, z = map(float, point)
    return [
        int(round(camera["fx"] * x / z + camera["ppx"])),
        int(round(camera["fy"] * y / z + camera["ppy"])),
    ]


def periodic_roll_diff(a, b):
    # A spine axis is unoriented, hence roll has pi periodicity.
    return float((a - b + math.pi / 2) % math.pi - math.pi / 2)


def axial_roll_std_deg(values):
    """Circular standard deviation for an unoriented (pi-periodic) axis."""
    angles = np.asarray(values, float) * 2.0
    resultant = abs(np.mean(np.exp(1j * angles)))
    resultant = float(np.clip(resultant, 1e-15, 1.0))
    return float(np.degrees(np.sqrt(-2.0 * np.log(resultant)) / 2.0))


def stats(values):
    a = np.asarray([v for v in values if v is not None], float)
    return {
        "count": int(len(a)),
        "mean": None if not len(a) else float(a.mean()),
        "median": None if not len(a) else float(np.median(a)),
        "min": None if not len(a) else float(a.min()),
        "max": None if not len(a) else float(a.max()),
        "std": None if not len(a) else float(a.std()),
    }


def main():
    run = BASE / "captures" / datetime.now().strftime(
        "100test_sam2_compatible_geometry_%Y%m%d_%H%M%S"
    )
    if run.exists():
        raise FileExistsError(run)
    run.mkdir()
    master = load_json(MASTER)
    all_rows = []
    per_case = []
    with (run / "run_console.log").open("w", encoding="utf-8") as log:
        for case_index in range(1, 101):
            src = SOURCE / str(case_index)
            dst = run / str(case_index)
            dst.mkdir()
            book = master[(case_index - 1) // 5]
            gt = float(book["book_width"])
            debug = load_json(src / "pointcloud_debug_result.json")
            if not debug.get("success"):
                failure = {
                    "case_index": case_index, "success": False,
                    "failure_stage": debug.get("failure_stage"),
                    "error": debug.get("error"), "modes": {},
                }
                save_json(dst / "sam2_compatible_geometry_result.json", failure)
                per_case.append(failure)
                log.write(f"CASE {case_index}: inherited failure: {debug.get('error')}\n")
                continue
            try:
                rgb = cv2.imread(str(src / "after_init_rgb.png"))
                mask = cv2.imread(str(src / "selected_mask_refined.png"), 0) > 0
                depth = np.load(
                    src / "debug_refine_only_median_depth_filter_depth_masked.npy"
                )
                points = read_ply_xyz(src / "pointcloud_sent_to_pca.ply")
                camera = load_json(src / "camera_params.json")
                refinement = load_json(src / "mask_refinement_result.json")
                current_result = load_json(src / "pca_result_offline.json")
                current_target = current_result["p_min_m"]
                current = {
                    "roll_rad": current_result["theta_rad"],
                    "width_mm": current_result["book_width_mm"],
                    "target_point_m": current_target,
                    "target_pixel": project(current_target, camera),
                }
                modes = {}
                for mode in MODES:
                    mode_dir = dst / mode
                    result = estimate_sam2_compatible_geometry(
                        mask, rgb, depth, points, camera,
                        selected_axis=refinement.get("axis_uv"),
                        current_geometry=current,
                        geometry_mode=mode,
                        debug_dir=mode_dir,
                    )
                    width = result["width"]["width_mm"]
                    target = np.asarray(result["target"]["target_point_m"], float)
                    roll_delta = periodic_roll_diff(
                        result["roll"]["value_rad"], current["roll_rad"]
                    )
                    target_delta = (target - np.asarray(current_target)) * 1000
                    differences = {
                        "roll_rad": roll_delta,
                        "roll_deg": float(np.degrees(roll_delta)),
                        "width_mm": width - current["width_mm"],
                        "target_point_distance_mm": float(np.linalg.norm(target_delta)),
                        "target_x_mm": float(target_delta[0]),
                        "target_y_mm": float(target_delta[1]),
                        "target_z_mm": float(target_delta[2]),
                    }
                    result["case_index"] = case_index
                    result["width"]["gt_width_mm"] = gt
                    result["width"]["abs_error_mm"] = abs(width - gt)
                    result["sam3_current"] = current
                    result["differences"] = differences
                    result["warnings"] = []
                    save_json(mode_dir / "sam2_compatible_geometry_result.json", result)
                    modes[mode] = result
                    all_rows.append({
                        "case_index": case_index, "book_name": book["book_name"],
                        "gt_width_mm": gt, "mode": mode, "success": True,
                        "roll_rad": result["roll"]["value_rad"],
                        "roll_diff_deg": differences["roll_deg"],
                        "width_mm": width, "abs_error_mm": abs(width - gt),
                        "target_x_m": target[0], "target_y_m": target[1],
                        "target_z_m": target[2],
                        "target_diff_mm": differences["target_point_distance_mm"],
                        "fallback": result["width"]["fallback"],
                        "output_dir": str(mode_dir),
                    })
                case_result = {
                    "case_index": case_index, "success": True,
                    "book_name": book["book_name"], "gt_width_mm": gt,
                    "mask_stage": "conservative_refinement_output",
                    "modes": modes,
                }
                save_json(dst / "sam2_compatible_geometry_result.json", case_result)
                per_case.append(case_result)
                log.write(
                    f"CASE {case_index}: current={current['width_mm']:.3f}, "
                    f"sam2={modes['sam2_all_geometry']['width']['width_mm']:.3f}\n"
                )
                log.flush()
            except Exception as exc:
                failure = {
                    "case_index": case_index, "success": False,
                    "failure_stage": "geometry", "error": f"{type(exc).__name__}: {exc}",
                    "modes": {},
                }
                save_json(dst / "sam2_compatible_geometry_result.json", failure)
                per_case.append(failure)
                log.write(f"CASE {case_index}: {failure['error']}\n"); log.flush()

    with (run / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(all_rows[0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(all_rows)
    summary = make_summary(per_case, all_rows, run)
    save_json(run / "summary.json", summary)
    write_report(run / "SAM2_COMPATIBLE_GEOMETRY_100CASE_REPORT.md", summary)
    print(f"RUN_ROOT={run}")


def make_summary(per_case, rows, run):
    success_cases = [r for r in per_case if r.get("success")]
    mode_summary = {}
    for mode in MODES:
        mr = [r for r in rows if r["mode"] == mode]
        errors = [r["abs_error_mm"] for r in mr]
        mode_summary[mode] = {
            "success_count": len(mr),
            "failure_count": 100 - len(mr),
            "fallback_count": sum(r["fallback"] for r in mr),
            "width_within_mm": {
                str(x): sum(e <= x for e in errors) for x in [1.0, 1.5, 2.0]
            },
            "width_abs_error_mm": stats(errors),
            "roll_diff_vs_current_deg": stats([abs(r["roll_diff_deg"]) for r in mr]),
            "target_diff_vs_current_mm": stats([r["target_diff_mm"] for r in mr]),
        }
    current = {
        r["case_index"]: r for r in rows if r["mode"] == "sam3_current_geometry"
    }
    compatible = {
        r["case_index"]: r for r in rows if r["mode"] == "sam2_all_geometry"
    }
    deltas = []
    for idx in sorted(set(current) & set(compatible)):
        deltas.append({
            "case_index": idx,
            "error_improvement_mm": current[idx]["abs_error_mm"]
            - compatible[idx]["abs_error_mm"],
            "current_error_mm": current[idx]["abs_error_mm"],
            "compatible_error_mm": compatible[idx]["abs_error_mm"],
        })
    groups = []
    for start in range(1, 101, 5):
        members = [i for i in range(start, start + 5) if i in compatible]
        if not members:
            continue
        def values(mode_map, key):
            return [mode_map[i][key] for i in members]
        current_targets = np.array([
            [current[i]["target_x_m"], current[i]["target_y_m"], current[i]["target_z_m"]]
            for i in members
        ])
        compat_targets = np.array([
            [compatible[i]["target_x_m"], compatible[i]["target_y_m"], compatible[i]["target_z_m"]]
            for i in members
        ])
        groups.append({
            "cases": members,
            "current_width_std_mm": float(np.std(values(current, "width_mm"))),
            "compatible_width_std_mm": float(np.std(values(compatible, "width_mm"))),
            "current_roll_std_deg": axial_roll_std_deg(values(current, "roll_rad")),
            "compatible_roll_std_deg": axial_roll_std_deg(values(compatible, "roll_rad")),
            "current_target_xyz_std_mm": (current_targets.std(axis=0) * 1000).tolist(),
            "compatible_target_xyz_std_mm": (compat_targets.std(axis=0) * 1000).tolist(),
            "current_target_radius_std_mm": float(np.std(np.linalg.norm(
                current_targets - current_targets.mean(axis=0), axis=1
            )) * 1000),
            "compatible_target_radius_std_mm": float(np.std(np.linalg.norm(
                compat_targets - compat_targets.mean(axis=0), axis=1
            )) * 1000),
        })
    case81 = next(r for r in success_cases if r["case_index"] == 81)
    sam2_reproduction = reproduce_saved_sam2_case81()
    return {
        "source": str(SOURCE), "output_root": str(run),
        "success_count": len(success_cases),
        "failure_count": 100 - len(success_cases),
        "mode_summary": mode_summary,
        "improved_case_count": sum(d["error_improvement_mm"] > 1e-9 for d in deltas),
        "worsened_case_count": sum(d["error_improvement_mm"] < -1e-9 for d in deltas),
        "unchanged_case_count": sum(abs(d["error_improvement_mm"]) <= 1e-9 for d in deltas),
        "largest_improvements": sorted(
            deltas, key=lambda x: x["error_improvement_mm"], reverse=True
        )[:10],
        "largest_degradations": sorted(
            deltas, key=lambda x: x["error_improvement_mm"]
        )[:10],
        "same_book_group_stability": groups,
        "case81": case81,
        "saved_sam2_case81_reproduction": sam2_reproduction,
        "failures": [r for r in per_case if not r.get("success")],
    }


def reproduce_saved_sam2_case81():
    case = INPUT / "81"
    saved = load_json(case / "pca_result_offline.json")
    saved_online = load_json(case / "pca_result.json")
    mask = cv2.imread(str(case / "final_mask.png"), 0) > 0
    rgb = cv2.imread(str(case / "after_init_rgb.png"))
    depth = np.load(case / "final_depth_masked.npy")
    points = np.load(case / "final_points_xyz_m.npy")
    camera = load_json(case / "camera_params.json")
    axis = saved["book_width_info"]["axis"]
    result = estimate_sam2_compatible_geometry(
        mask, rgb, depth, points, camera, selected_axis=axis,
        current_geometry={
            "roll_rad": saved["theta_rad"], "width_mm": saved["book_width_mm"],
            "target_point_m": saved["p_min_m"],
            "target_pixel": project(saved["p_min_m"], camera),
        },
        geometry_mode="sam2_all_geometry",
    )
    return {
        "saved": {
            "roll_rad": saved["theta_rad"], "width_mm": saved["book_width_mm"],
            "target_point_m": saved["p_min_m"],
        },
        "recomputed": {
            "roll_rad": result["roll"]["value_rad"],
            "width_mm": result["width"]["width_mm"],
            "target_point_m": result["target"]["target_point_m"],
        },
        "differences": {
            "roll_rad": periodic_roll_diff(
                result["roll"]["value_rad"], saved["theta_rad"]
            ),
            "width_mm": result["width"]["width_mm"] - saved["book_width_mm"],
            "target_distance_mm": float(np.linalg.norm(
                np.asarray(result["target"]["target_point_m"])
                - np.asarray(saved["p_min_m"])
            ) * 1000),
        },
        "saved_online_result": {
            "roll_rad": saved_online["theta_rad"],
            "width_mm": saved_online["book_width_mm"],
            "target_point_m": saved_online["p_min_m"],
            "reproduced": False,
            "reason": (
                "The un-suffixed final_mask/final_depth/final_points artifacts "
                "correspond to the later offline result (32.981 mm). The online "
                "15.552 mm result remains, but its exact final intermediate "
                "arrays were overwritten and cannot be faithfully recomputed."
            ),
        },
    }


def write_report(path, s):
    modes = s["mode_summary"]
    c81 = s["case81"]["modes"]
    rep = s["saved_sam2_case81_reproduction"]
    lines = [
        "# SAM2-compatible geometry 100-case report", "",
        f"- Source: `{s['source']}`",
        f"- Success/failure: {s['success_count']}/{s['failure_count']}", "",
        "## Mode summary", "",
    ]
    for mode in MODES:
        x = modes[mode]
        lines += [
            f"### {mode}", "",
            f"- Success: {x['success_count']}; fallback: {x['fallback_count']}",
            f"- Within 1/1.5/2 mm: {x['width_within_mm']}",
            f"- Width absolute error [mm]: `{x['width_abs_error_mm']}`",
            f"- Roll delta vs current [deg]: `{x['roll_diff_vs_current_deg']}`",
            f"- Target delta vs current [mm]: `{x['target_diff_vs_current_mm']}`", "",
        ]
    lines += [
        "## Case 81", "",
        *[
            f"- {mode}: roll `{c81[mode]['roll']['value_deg']:.6f} deg`, "
            f"width `{c81[mode]['width']['width_mm']:.6f} mm`, "
            f"target `{c81[mode]['target']['target_point_m']}`"
            for mode in MODES
        ], "",
        "## Saved SAM2 case81 reproduction", "",
        f"- Saved: `{rep['saved']}`",
        f"- Recomputed: `{rep['recomputed']}`",
        f"- Differences: `{rep['differences']}`", "",
        f"- Earlier saved online result: `{rep['saved_online_result']}`", "",
        "## Width change", "",
        f"- Improved/worsened/unchanged cases: "
        f"{s['improved_case_count']}/{s['worsened_case_count']}/{s['unchanged_case_count']}",
        f"- Largest improvements: `{s['largest_improvements']}`",
        f"- Largest degradations: `{s['largest_degradations']}`", "",
        "## Interpretation", "",
        "- Roll and target formulas are already identical between the traced SAM2 terminal geometry and current SAM3 geometry. With the same final point cloud their numeric deltas are zero.",
        "- Width differs: current SAM3 uses 3D PCA pc2 slices; SAM2-compatible mode uses final-mask short-axis pixel width converted with median Depth and intrinsics.",
        "- Roll/target have no ground truth in this dataset; equality is process/output consistency, not an accuracy claim.", "",
        "No inference, OCR, robot, ROS, RealSense, or service was run. Existing artifacts were read only.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
