#!/usr/bin/env python3
"""Independently recompute three-way summary metrics from per-case CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from detection.pro_handbook.sam_py_demo.modules.sam2_compatible_geometry import (
    MASK_PCA_WIDTH_MODES,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def close(first, second, tolerance=1e-9):
    return bool(np.isclose(float(first), float(second), atol=tolerance, rtol=0.0))


def main() -> int:
    run_dir = parse_args().run_dir.expanduser().resolve()
    comparison = run_dir / "comparison"
    with (comparison / "per_case_comparison.csv").open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads(
        (comparison / "comparison_summary.json").read_text(encoding="utf-8")
    )
    if len(rows) != summary["run_config"]["end_case"] - summary["run_config"]["start_case"] + 1:
        raise AssertionError("case count mismatch")

    checks = {}
    success_sets = []
    for mode in MASK_PCA_WIDTH_MODES:
        successful = [row for row in rows if as_bool(row[f"{mode}_ok"])]
        success_sets.append(tuple(int(row["case_no"]) for row in successful))
        signed = np.asarray(
            [float(row[f"{mode}_error_mm"]) for row in successful], np.float64
        )
        absolute = np.abs(signed)
        expected = summary["method_summaries"][mode]
        calculated = {
            "success_count": len(successful),
            "failure_count": len(rows) - len(successful),
            "mean_absolute_error_mm": float(np.mean(absolute)),
            "median_absolute_error_mm": float(np.median(absolute)),
            "rmse_mm": float(np.sqrt(np.mean(signed ** 2))),
            "absolute_error_p90_mm": float(np.percentile(absolute, 90.0)),
            "absolute_error_p95_mm": float(np.percentile(absolute, 95.0)),
            "max_absolute_error_mm": float(np.max(absolute)),
            "within_1.0_mm": int(np.sum(absolute <= 1.0)),
            "within_1.5_mm": int(np.sum(absolute <= 1.5)),
            "within_2.0_mm": int(np.sum(absolute <= 2.0)),
        }
        assert calculated["success_count"] == expected["success_count"]
        assert calculated["failure_count"] == expected["failure_count"]
        for key in (
            "mean_absolute_error_mm",
            "median_absolute_error_mm",
            "rmse_mm",
            "absolute_error_p90_mm",
            "absolute_error_p95_mm",
            "max_absolute_error_mm",
        ):
            if not close(calculated[key], expected[key]):
                raise AssertionError(f"{mode} {key} mismatch")
        for threshold in (1.0, 1.5, 2.0):
            key = f"within_{threshold:.1f}_mm"
            if calculated[key] != expected["thresholds"][key]["count"]:
                raise AssertionError(f"{mode} {key} mismatch")
        checks[mode] = calculated

    if len(set(success_sets)) != 1:
        raise AssertionError("success/failure case sets differ among modes")

    for row in rows:
        if not all(as_bool(row[f"{mode}_ok"]) for mode in MASK_PCA_WIDTH_MODES):
            continue
        if (
            row[f"slice_p2p98_median_valid_slice_count"]
            != row[f"slice_minmax_median_valid_slice_count"]
        ):
            raise AssertionError(
                f"case {row['case_no']}: B/C valid slice count mismatch"
            )

    shared_pca_case_count = 0
    for case_no in success_sets[0]:
        pca_values = []
        for mode in MASK_PCA_WIDTH_MODES:
            value = json.loads(
                (run_dir / mode / str(case_no) / "width_result.json").read_text(
                    encoding="utf-8"
                )
            )
            pca_values.append(value["mask_pca"])
        canonical = [json.dumps(value, sort_keys=True) for value in pca_values]
        if len(set(canonical)) != 1:
            raise AssertionError(f"case {case_no}: shared PCA mismatch")
        shared_pca_case_count += 1

    result = {
        "ok": True,
        "run_dir": str(run_dir),
        "case_count": len(rows),
        "success_failure_sets_identical": True,
        "slice_p2p98_minmax_valid_slice_counts_identical": True,
        "shared_pca_case_count": shared_pca_case_count,
        "recomputed_metrics": checks,
    }
    output = comparison / "independent_validation.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
