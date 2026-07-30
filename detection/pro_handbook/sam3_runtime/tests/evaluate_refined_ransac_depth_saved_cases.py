#!/usr/bin/env python3
"""Evaluate RANSAC Depth completion on the same saved 20-case cohort."""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from detection.pro_handbook.sam_py_demo import get_book_points as current
from detection.pro_handbook.sam_py_demo import (
    get_book_points_no_mask_merge_no_side_filter as stable,
)
from detection.pro_handbook.sam_py_demo.get_book_points_sam3_refined_ransac_depth import (
    _ransac_complete_and_pca,
    _write_json,
)
from detection.pro_handbook.sam_py_demo.sam3_mask_refinement import (
    refine_selected_sam3_mask,
)


PROJECT = Path("/home/book/pro_book_SAM3/pro_hand_book_python")
PRIOR_COMPARISON = (
    PROJECT
    / "detection/pro_handbook/sam3_runtime/docs/sam3_refined_median_depth_comparison.json"
)
OUTPUT = (
    PROJECT
    / "detection/pro_handbook/sam3_runtime/tests/outputs/refined_ransac_depth_saved_cases"
)
COMPARISON = (
    PROJECT
    / "detection/pro_handbook/sam3_runtime/docs/sam3_refined_ransac_depth_comparison.json"
)


def _selected_index(similarity):
    scores = similarity.get("scores") or []
    if not scores:
        return None
    match = re.search(r"(\d+)$", str(scores[0].get("name", "")))
    return int(match.group(1)) if match else None


def _prior_mode(prior_case, mode):
    value = dict(prior_case["modes"][mode])
    value["reused_from"] = str(PRIOR_COMPARISON)
    return value


def _evaluate(case_info, mode):
    case = Path(case_info["source_dir"])
    out = OUTPUT / case.name / mode
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(case / "after_init_rgb.png", out / "after_init_rgb.png")
    shutil.copy2(case / "after_init_depth.npy", out / "after_init_depth.npy")
    rgb = cv2.imread(str(case / "after_init_rgb.png"), cv2.IMREAD_COLOR)
    depth = np.load(case / "after_init_depth.npy", allow_pickle=False)
    raw = cv2.imread(
        str(case / "selected_mask_before_legacy_postprocess.png"),
        cv2.IMREAD_GRAYSCALE,
    ) > 0
    np.save(out / "after_init_depth_raw.npy", depth)
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
    started = time.perf_counter()
    refined, refinement, _ = refine_selected_sam3_mask(
        raw,
        ocr_polygon=polygon,
        depth_raw=depth,
        rgb_bgr=rgb,
        output_dir=out,
        mode=mode,
    )
    use_refinement = mode == "refine_and_ransac_complete"
    used = refined if use_refinement else raw.copy()
    if not use_refinement:
        refinement.update(
            {
                "mode": mode,
                "refinement_enabled": False,
                "kept_component_ids": list(
                    range(1, refinement["raw_component_count"] + 1)
                ),
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
    compute = _ransac_complete_and_pca(
        shot_dir=out,
        color_np=rgb,
        depth_raw=depth,
        raw_mask=raw,
        used_mask=used,
        anchor_mask=anchor,
        intr=intr,
        depth_scale=scale,
    )
    elapsed = time.perf_counter() - started
    result = {
        "success": True,
        "query": query,
        "selected_mask_index": _selected_index(similarity),
        "source_case": str(case),
        "mode": mode,
        "raw_component_count": refinement["raw_component_count"],
        "anchor_component_id": refinement["anchor_component_id"],
        "kept_component_ids": refinement["kept_component_ids"],
        "removed_component_ids": refinement["removed_component_ids"],
        "raw_mask_area_px": int(raw.sum()),
        "final_mask_area_px": int(used.sum()),
        "refinement_no_op": bool(np.array_equal(raw, used)),
        "raw_mask_sha256": stable._sha256(out / "selected_mask_raw.png"),
        "refined_mask_sha256": stable._sha256(out / "selected_mask_refined.png"),
        "fallback": refinement["fallback_to_raw_mask"],
        "fallback_reason": refinement["fallback_reason"],
        "gt_book_width_mm": 14.8,
        **compute,
        "abs_width_error_mm": abs(compute["pred_book_width_mm"] - 14.8),
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
    prior = json.loads(PRIOR_COMPARISON.read_text(encoding="utf-8"))
    payload = {
        "evaluation_type": "saved_selected_mask_rgbd_replay",
        "cohort_source": str(PRIOR_COMPARISON),
        "sam3_inference_rerun": False,
        "sam3_inference_rerun_reason": (
            "The immediately preceding fresh service attempts reproduced "
            "CUDA CUBLAS_STATUS_NOT_INITIALIZED. Per instruction, saved SAM3 "
            "selections, OCR, RGB, and Depth were reused without another start."
        ),
        "modes": [
            "baseline",
            "refine_only",
            "ransac_complete_only",
            "refine_and_ransac_complete",
        ],
        "cases": [],
        "errors": [],
    }
    for prior_case in prior["cases"]:
        case_result = {
            "case": prior_case["case"],
            "source_dir": prior_case["source_dir"],
            "category": prior_case["category"],
            "component_count": prior_case["component_count"],
            "component_areas": prior_case["component_areas"],
            "query": prior_case["query"],
            "modes": {
                "baseline": _prior_mode(prior_case, "baseline"),
                "refine_only": _prior_mode(prior_case, "refine_only"),
            },
        }
        for mode in ("ransac_complete_only", "refine_and_ransac_complete"):
            try:
                case_result["modes"][mode] = _evaluate(prior_case, mode)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                case_result["modes"][mode] = {
                    "mode": mode,
                    "success": False,
                    "error": error,
                }
                payload["errors"].append(
                    {"case": prior_case["case"], "mode": mode, "error": error}
                )
        payload["cases"].append(case_result)
        print(
            prior_case["case"],
            prior_case["category"],
            {
                mode: value.get("success")
                for mode, value in case_result["modes"].items()
            },
            flush=True,
        )
    clean = [case for case in payload["cases"] if case["category"] == "clean"]
    noisy = [case for case in payload["cases"] if case["category"] == "noisy"]
    standard = "refine_and_ransac_complete"
    payload["summary"] = {
        "case_count": len(payload["cases"]),
        "clean_case_count": len(clean),
        "noisy_case_count": len(noisy),
        "mode_success_counts": {
            mode: sum(bool(case["modes"][mode].get("success")) for case in payload["cases"])
            for mode in payload["modes"]
        },
        "clean_refinement_no_op": sum(
            bool(case["modes"][standard].get("refinement_no_op")) for case in clean
        ),
        "clean_new_failures": sum(
            not bool(case["modes"][standard].get("success")) for case in clean
        ),
        "noisy_components_removed_cases": sum(
            bool(case["modes"][standard].get("removed_component_ids")) for case in noisy
        ),
        "ransac_fallback_count": sum(
            bool(
                (case["modes"][standard].get("depth_completion") or {}).get(
                    "fallback_used"
                )
            )
            for case in payload["cases"]
        ),
    }
    _write_json(COMPARISON, payload)
    print("COMPARISON_JSON=" + str(COMPARISON))
    print(json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True))
    return 0 if not payload["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
