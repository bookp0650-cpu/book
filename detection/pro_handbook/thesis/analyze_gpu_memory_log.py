#!/usr/bin/env python3
"""Summarize phase-aligned nvidia-smi samples from thesis_ret experiments."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    return parser.parse_args()


def _read_phases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _read_samples(path: Path) -> list[dict]:
    grouped: dict[float, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("gpu_used_memory_mib"):
                continue
            timestamp = float(row["monotonic_seconds"])
            sample = grouped.setdefault(
                timestamp,
                {
                    "monotonic_seconds": timestamp,
                    "gpu_used_memory_mib": int(row["gpu_used_memory_mib"]),
                    "gpu_total_memory_mib": int(row["gpu_total_memory_mib"]),
                    "gpu_utilization_percent": (
                        int(row["gpu_utilization_percent"])
                        if row.get("gpu_utilization_percent")
                        else None
                    ),
                    "processes": {},
                },
            )
            if row.get("process_pid"):
                sample["processes"][row["process_pid"]] = {
                    "name": row["process_name"],
                    "used_memory_mib": int(row["process_used_memory_mib"]),
                }
    return [grouped[key] for key in sorted(grouped)]


def _event_time(events: list[dict], phase: str, occurrence: int = 0) -> float | None:
    matches = [event for event in events if event.get("phase") == phase]
    if len(matches) <= occurrence:
        return None
    return float(matches[occurrence]["monotonic_seconds"])


def _peak(samples: list[dict], start: float, end: float) -> dict | None:
    selected = [s for s in samples if start <= s["monotonic_seconds"] <= end]
    if not selected:
        return None
    sample = max(selected, key=lambda item: item["gpu_used_memory_mib"])
    return {
        "gpu_used_memory_mib": sample["gpu_used_memory_mib"],
        "gpu_utilization_peak_percent": max(
            (
                item["gpu_utilization_percent"]
                for item in selected
                if item["gpu_utilization_percent"] is not None
            ),
            default=None,
        ),
        "gpu_utilization_mean_percent": (
            sum(
                item["gpu_utilization_percent"]
                for item in selected
                if item["gpu_utilization_percent"] is not None
            )
            / sum(
                1
                for item in selected
                if item["gpu_utilization_percent"] is not None
            )
            if any(item["gpu_utilization_percent"] is not None for item in selected)
            else None
        ),
        "monotonic_seconds": sample["monotonic_seconds"],
        "processes": sample["processes"],
    }


def _nearest(samples: list[dict], timestamp: float) -> dict | None:
    if not samples:
        return None
    sample = min(samples, key=lambda item: abs(item["monotonic_seconds"] - timestamp))
    return {
        "gpu_used_memory_mib": sample["gpu_used_memory_mib"],
        "gpu_utilization_percent": sample["gpu_utilization_percent"],
        "monotonic_seconds": sample["monotonic_seconds"],
        "offset_seconds": sample["monotonic_seconds"] - timestamp,
        "processes": sample["processes"],
    }


def main() -> int:
    args = _parse_args()
    events = _read_phases(args.log_dir / "phases.jsonl")
    samples = _read_samples(args.log_dir / "samples.csv")
    if not samples:
        raise RuntimeError("no valid GPU samples")

    summary: dict = {
        "gpu_total_memory_mib": samples[0]["gpu_total_memory_mib"],
        "sample_count": len(samples),
        "sample_interval_note": "phase peaks are maxima of sampled nvidia-smi values",
        "before_models": None,
        "sam3_ready": None,
        "both_models_ready": None,
        "sam3_inference_peaks": [],
        "ocr_inference_peaks": [],
        "case_end_resident_memory": [],
    }
    before = _event_time(events, "recognition_program_before_models")
    if before is not None:
        summary["before_models"] = _nearest(samples, before)
    sam_ready = _event_time(events, "sam3_service_ready")
    if sam_ready is not None:
        summary["sam3_ready"] = _nearest(samples, sam_ready)
    model_ready = _event_time(events, "ocr_model_ready")
    if model_ready is not None:
        summary["both_models_ready"] = _nearest(samples, model_ready)

    sam_starts = [e for e in events if e.get("phase") == "sam3_inference_start"]
    sam_ends = [e for e in events if e.get("phase") == "sam3_inference_end"]
    for index, (start, end) in enumerate(zip(sam_starts, sam_ends), start=1):
        summary["sam3_inference_peaks"].append(
            {"sequence": index, **(_peak(samples, float(start["monotonic_seconds"]), float(end["monotonic_seconds"])) or {})}
        )

    ocr_starts = [e for e in events if e.get("phase") == "ocr_inference_start"]
    ocr_ends = [e for e in events if e.get("phase") == "ocr_inference_end"]
    for index, (start, end) in enumerate(zip(ocr_starts, ocr_ends), start=1):
        summary["ocr_inference_peaks"].append(
            {"sequence": index, **(_peak(samples, float(start["monotonic_seconds"]), float(end["monotonic_seconds"])) or {})}
        )

    for event in (e for e in events if e.get("phase") == "case_end"):
        summary["case_end_resident_memory"].append(
            {
                "case_id": event.get("case_id"),
                **(_nearest(samples, float(event["monotonic_seconds"])) or {}),
            }
        )

    overall = max(samples, key=lambda item: item["gpu_used_memory_mib"])
    summary["overall_peak"] = {
        "gpu_used_memory_mib": overall["gpu_used_memory_mib"],
        "gpu_utilization_peak_percent": max(
            (
                sample["gpu_utilization_percent"]
                for sample in samples
                if sample["gpu_utilization_percent"] is not None
            ),
            default=None,
        ),
        "gpu_utilization_mean_percent": (
            sum(
                sample["gpu_utilization_percent"]
                for sample in samples
                if sample["gpu_utilization_percent"] is not None
            )
            / sum(
                1
                for sample in samples
                if sample["gpu_utilization_percent"] is not None
            )
            if any(sample["gpu_utilization_percent"] is not None for sample in samples)
            else None
        ),
        "monotonic_seconds": overall["monotonic_seconds"],
        "processes": overall["processes"],
    }
    summary["minimum_headroom_mib"] = (
        summary["gpu_total_memory_mib"] - overall["gpu_used_memory_mib"]
    )
    output = args.log_dir / "analysis.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
