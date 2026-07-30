#!/usr/bin/env python3
"""Evaluate the new mask/depth variant using existing saved SAM3 selections."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from detection.pro_handbook.sam_py_demo import get_book_points as current
from detection.pro_handbook.sam_py_demo import (
    get_book_points_no_mask_merge_no_side_filter as stable,
)
from detection.pro_handbook.sam_py_demo.get_book_points_sam3_refined_median_depth import (
    _depth_and_pca,
    _save_depth_visualization,
    _write_json,
)
from detection.pro_handbook.sam_py_demo.sam3_mask_refinement import (
    refine_selected_sam3_mask,
)


PROJECT = Path("/home/book/pro_book_SAM3/pro_hand_book_python")
CAPTURES = PROJECT / "captures"
OUTPUT = (
    PROJECT
    / "detection/pro_handbook/sam3_runtime/tests/outputs/refined_median_depth_saved_cases"
)
COMPARISON = (
    PROJECT
    / "detection/pro_handbook/sam3_runtime/docs/sam3_refined_median_depth_comparison.json"
)


def _case_candidates():
    rows = []
    for mask_path in CAPTURES.rglob("selected_mask_before_legacy_postprocess.png"):
        case = mask_path.parent
        if not (case / "after_init_rgb.png").is_file():
            continue
        if not (case / "after_init_depth.npy").is_file():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        rows.append(
            {
                "path": case,
                "component_count": count - 1,
                "area": int(mask.sum()),
                "component_areas": stats[1:, cv2.CC_STAT_AREA].astype(int).tolist(),
            }
        )
    clean = sorted(
        (row for row in rows if row["component_count"] == 1),
        key=lambda row: str(row["path"]),
    )[:12]
    noisy = sorted(
        (row for row in rows if row["component_count"] > 1),
        key=lambda row: (-row["component_count"], str(row["path"])),
    )
    return clean, noisy


def _baseline(case):
    core_path = case / "live_core_result.json"
    pca_path = case / "pca_result_offline.json"
    core = json.loads(core_path.read_text(encoding="utf-8")) if core_path.is_file() else {}
    pca = json.loads(pca_path.read_text(encoding="utf-8")) if pca_path.is_file() else {}
    width = core.get("pred_book_width_mm", pca.get("book_width_mm"))
    return {
        "mode": "baseline",
        "success": bool(core or pca),
        "roll_rad": core.get("roll_rad", pca.get("theta_rad")),
        "pred_book_width_mm": width,
        "gt_book_width_mm": 14.8,
        "abs_width_error_mm": None if width is None else abs(width - 14.8),
        "point_3d": core.get("point_3d", pca.get("p_min_m")),
        "point_counts": core.get("point_counts"),
        "processing_seconds": None,
        "source": str(core_path if core else pca_path),
    }


def _evaluate_mode(case_row, polygon, query, mode, rgb, depth, raw, anchor_id):
    case = case_row["path"]
    out = OUTPUT / case.name / mode
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(case / "after_init_rgb.png", out / "after_init_rgb.png")
    shutil.copy2(case / "after_init_depth.npy", out / "after_init_depth.npy")
    np.save(out / "after_init_depth_raw.npy", depth)
    _save_depth_visualization(out / "depth_raw_visualization.png", depth)
    started = time.perf_counter()
    refined, refinement, metrics = refine_selected_sam3_mask(
        raw,
        ocr_polygon=polygon,
        depth_raw=depth,
        rgb_bgr=rgb,
        output_dir=out,
        mode=mode,
    )
    used = refined if mode in {"refine_only", "refine_and_median_flatten"} else raw
    if mode == "median_flatten_only":
        refinement.update(
            {
                "refinement_enabled": False,
                "kept_component_ids": list(range(1, refinement["raw_component_count"] + 1)),
                "removed_component_ids": [],
                "refined_mask_area_px": int(raw.sum()),
                "removed_area_px": 0,
                "no_op": True,
                "raw_and_refined_mask_equal": True,
                "fallback_to_raw_mask": False,
                "fallback_reason": None,
            }
        )
        cv2.imwrite(str(out / "selected_mask_refined.png"), raw.astype(np.uint8) * 255)
        cv2.imwrite(str(out / "components_kept.png"), raw.astype(np.uint8) * 255)
        cv2.imwrite(
            str(out / "components_removed.png"), np.zeros_like(raw, dtype=np.uint8)
        )
        green = rgb.copy()
        green[raw] = (0, 255, 0)
        cv2.imwrite(
            str(out / "selected_mask_refined_overlay.png"),
            cv2.addWeighted(rgb, 0.65, green, 0.35, 0),
        )
        _write_json(out / "mask_refinement_result.json", refinement)
    _, labels = cv2.connectedComponents(raw.astype(np.uint8), connectivity=8)
    anchor = labels == int(refinement["anchor_component_id"])
    intr, scale = stable._intrinsics()
    compute = _depth_and_pca(
        mode=mode,
        shot_dir=out,
        color_np=rgb,
        depth_raw=depth,
        raw_mask=raw,
        used_mask=used,
        anchor_mask=anchor,
        ocr_polygon=polygon,
        intr=intr,
        depth_scale=scale,
        depth_merge_tolerance_raw=30,
    )
    elapsed = time.perf_counter() - started
    raw_sha = stable._sha256(out / "selected_mask_raw.png")
    refined_sha = stable._sha256(out / "selected_mask_refined.png")
    result = {
        "success": True,
        "query": query,
        "source_case": str(case),
        "mode": mode,
        "raw_component_count": refinement["raw_component_count"],
        "anchor_component_id": refinement["anchor_component_id"],
        "kept_component_ids": refinement["kept_component_ids"],
        "removed_component_ids": refinement["removed_component_ids"],
        "raw_mask_area_px": int(raw.sum()),
        "refined_mask_area_px": int(used.sum()),
        "raw_and_refined_mask_equal": bool(np.array_equal(raw, used)),
        "raw_mask_sha256": raw_sha,
        "refined_mask_sha256": refined_sha,
        "mask_sha256_equal": raw_sha == refined_sha,
        "fallback": refinement["fallback_to_raw_mask"],
        "fallback_reason": refinement["fallback_reason"],
        "roll_rad": compute["roll_rad"],
        "pred_book_width_mm": compute["pred_book_width_mm"],
        "gt_book_width_mm": 14.8,
        "abs_width_error_mm": abs(compute["pred_book_width_mm"] - 14.8),
        "point_3d": compute["point_3d"],
        "point_counts": compute["point_counts"],
        "depth_flattening": compute["depth_flattening"],
        "processing_seconds": elapsed,
        "output_dir": str(out),
    }
    _write_json(out / "offline_recognition_result.json", result)
    (out / "offline_run_console.log").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    clean, noisy = _case_candidates()
    selected = [("clean", row) for row in clean] + [
        ("noisy", row) for row in noisy
    ]
    payload = {
        "evaluation_type": "saved_selected_mask_rgbd_replay",
        "sam3_inference_rerun": False,
        "sam3_inference_rerun_reason": (
            "SAM3 service model load failed twice with CUDA "
            "CUBLAS_STATUS_NOT_INITIALIZED; existing saved selections were used."
        ),
        "clean_case_count": len(clean),
        "noisy_case_count": len(noisy),
        "modes": [
            "baseline",
            "refine_only",
            "median_flatten_only",
            "refine_and_median_flatten",
        ],
        "cases": [],
        "errors": [],
    }
    for category, row in selected:
        case = row["path"]
        try:
            rgb = cv2.imread(str(case / "after_init_rgb.png"), cv2.IMREAD_COLOR)
            depth = np.load(case / "after_init_depth.npy", allow_pickle=False)
            raw = cv2.imread(
                str(case / "selected_mask_before_legacy_postprocess.png"),
                cv2.IMREAD_GRAYSCALE,
            ) > 0
            similarity = json.loads(
                (case / "similarity_scores.json").read_text(encoding="utf-8")
            )
            query = similarity["query"]
            best, debug = current._load_best_ocr_polygon_for_mask(
                case, query, raw.astype(np.uint8)
            )
            if best is None:
                raise RuntimeError(f"OCR polygon unavailable: {debug.get('reason')}")
            polygon = np.asarray(best["poly"], float).tolist()
            case_result = {
                "case": case.name,
                "source_dir": str(case),
                "category": category,
                "component_count": row["component_count"],
                "component_areas": row["component_areas"],
                "query": query,
                "ocr_text": best["text"],
                "ocr_overlap": best["overlap"],
                "modes": {"baseline": _baseline(case)},
            }
            for mode in (
                "refine_only",
                "median_flatten_only",
                "refine_and_median_flatten",
            ):
                try:
                    case_result["modes"][mode] = _evaluate_mode(
                        row, polygon, query, mode, rgb, depth, raw, None
                    )
                except Exception as exc:
                    case_result["modes"][mode] = {
                        "mode": mode,
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    payload["errors"].append(
                        {
                            "case": case.name,
                            "mode": mode,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            payload["cases"].append(case_result)
            print(
                case.name,
                category,
                row["component_count"],
                {
                    mode: data.get("success")
                    for mode, data in case_result["modes"].items()
                },
                flush=True,
            )
        except Exception as exc:
            payload["errors"].append(
                {"case": case.name, "error": f"{type(exc).__name__}: {exc}"}
            )
            print(case.name, "ERROR", exc, flush=True)

    clean_results = [
        case for case in payload["cases"] if case["category"] == "clean"
    ]
    noisy_results = [
        case for case in payload["cases"] if case["category"] == "noisy"
    ]
    payload["summary"] = {
        "clean_evaluated": len(clean_results),
        "noisy_evaluated": len(noisy_results),
        "clean_refinement_no_op": sum(
            bool(case["modes"]["refine_and_median_flatten"].get("raw_and_refined_mask_equal"))
            for case in clean_results
        ),
        "clean_new_failures": sum(
            not bool(case["modes"]["refine_and_median_flatten"].get("success"))
            for case in clean_results
        ),
        "noisy_components_removed_cases": sum(
            bool(case["modes"]["refine_and_median_flatten"].get("removed_component_ids"))
            for case in noisy_results
        ),
        "mode_success_counts": {
            mode: sum(bool(case["modes"][mode].get("success")) for case in payload["cases"])
            for mode in payload["modes"]
        },
    }
    _write_json(COMPARISON, payload)
    print("COMPARISON_JSON=" + str(COMPARISON))
    print(json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True))
    return 0 if not payload["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
