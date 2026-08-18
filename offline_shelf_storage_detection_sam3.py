#!/usr/bin/env python3
"""Saved RGB-D only entry point for SAM3 shelf-storage recognition."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from detection.pro_handbook.sam_py_demo.modules.shelf_storage_detection_sam3 import (
    load_config,
    resolve_shot_inputs,
    run_offline_shelf_storage_detection,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parent
    / "detection"
    / "pro_handbook"
    / "sam_py_demo"
    / "config"
    / "shelf_storage_detection_sam3.json"
)
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BATCH_INPUT_ROOT = PROJECT_ROOT / "captures" / "100test"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run shelf-storage recognition from saved aligned RGB-D data. "
            "This command never opens a RealSense device or a robot connection."
        )
    )
    parser.add_argument("--shot-dir", type=Path, help="Saved capture directory")
    parser.add_argument("--rgb", type=Path, help="Saved RGB image")
    parser.add_argument("--depth", type=Path, help="Saved aligned Depth .npy")
    parser.add_argument("--intrinsics", type=Path, help="camera_params/intrinsics JSON")
    parser.add_argument("--sam3-masks", type=Path, help="Saved SAM3 masks .npz")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--batch-case-numbers",
        help=(
            "Comma-separated captures/100test case numbers. A single batch "
            "timestamp is created and cases are saved as 1, 2, ... in this order."
        ),
    )
    parser.add_argument("--book-width-mm", type=float)
    parser.add_argument("--book-height-mm", type=float)
    parser.add_argument("--book-depth-mm", type=float)
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="List/measure candidates without judging fit against a book size",
    )
    parser.add_argument(
        "--force-sam3-inference",
        action="store_true",
        help="Ignore a saved masks NPZ and call the existing SAM3 service client",
    )
    return parser


def _save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_output_root(config: dict[str, Any]) -> Path:
    root = Path(config["output"]["root_dir"]).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root.resolve()


def _allocate_batch_dir(output_root: Path) -> tuple[Path, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidates = [stamp, datetime.now().strftime("%Y%m%d_%H%M%S_%f")]
    candidates.extend(f"{stamp}_{index:02d}" for index in range(1, 100))
    for name in candidates:
        batch_dir = output_root / name
        try:
            batch_dir.mkdir(exist_ok=False)
            return batch_dir.resolve(), name
        except FileExistsError:
            continue
    raise RuntimeError("Could not allocate a unique batch timestamp directory")


def _parse_case_numbers(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("--batch-case-numbers contains an empty case number")
    numbers = [int(part) for part in parts]
    if any(number <= 0 for number in numbers):
        raise ValueError("Batch case numbers must be positive integers")
    return numbers


def _configure_dimensions(
    config: dict[str, Any], args: argparse.Namespace
) -> None:
    dims = config.setdefault("book_dimensions_mm", {})
    if args.detect_only:
        dims.update(
            {"book_width_mm": None, "book_height_mm": None, "book_depth_mm": None}
        )
    else:
        for key, value in (
            ("book_width_mm", args.book_width_mm),
            ("book_height_mm", args.book_height_mm),
            ("book_depth_mm", args.book_depth_mm),
        ):
            if value is not None:
                dims[key] = value


def _batch_summary(
    *,
    batch_timestamp: str,
    input_case_count: int,
    case_results: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    status_counts = {"accepted": 0, "uncertain": 0, "rejected": 0}
    failed = 0
    successful = 0
    for case in case_results:
        if case["pipeline_ok"]:
            successful += 1
        else:
            failed += 1
        for status, count in case["candidate_status_counts"].items():
            status_counts[status] += int(count)
    return {
        "batch_timestamp": batch_timestamp,
        "input_case_count": input_case_count,
        "completed_case_count": len(case_results),
        "successful_case_count": successful,
        "failed_case_count": failed,
        "accepted_count": status_counts["accepted"],
        "uncertain_count": status_counts["uncertain"],
        "rejected_count": status_counts["rejected"],
        "status_count_scope": "candidate",
        "case_results": case_results,
        "total_elapsed_sec": elapsed,
        "mean_elapsed_sec": elapsed / len(case_results) if case_results else None,
    }


def _run_batch(
    args: argparse.Namespace,
    config: dict[str, Any],
    case_numbers: list[int],
) -> tuple[int, dict[str, Any]]:
    batch_started = time.perf_counter()
    output_root = _resolve_output_root(config)
    batch_dir, batch_timestamp = _allocate_batch_dir(output_root)
    mapping = {
        str(output_index): source_case_number
        for output_index, source_case_number in enumerate(case_numbers, start=1)
    }
    _save_json(batch_dir / "case_mapping.json", mapping)
    batch_log_path = batch_dir / "batch_run.log"
    batch_log_lines: list[str] = []

    def batch_log(message: str) -> None:
        stamp = datetime.now().isoformat(timespec="milliseconds")
        batch_log_lines.append(f"{stamp} {message}")
        batch_log_path.write_text(
            "\n".join(batch_log_lines) + "\n", encoding="utf-8"
        )

    batch_log(
        f"batch_timestamp={batch_timestamp} output_dir={batch_dir} "
        f"case_mapping={json.dumps(mapping, ensure_ascii=False)}"
    )
    print(f"batch_output_dir={batch_dir}", flush=True)
    case_results: list[dict[str, Any]] = []
    for output_index, source_case_number in enumerate(case_numbers, start=1):
        source_dir = (DEFAULT_BATCH_INPUT_ROOT / str(source_case_number)).resolve()
        case_output_dir = batch_dir / str(output_index)
        run_metadata = {
            "output_index": output_index,
            "source_case_number": source_case_number,
            "source_shot_dir": str(source_dir),
            "output_dir": str(case_output_dir),
            "batch_timestamp": batch_timestamp,
        }
        batch_log(
            f"case_start output_index={output_index} "
            f"source_case_number={source_case_number} source_shot_dir={source_dir}"
        )
        print(
            f"[{output_index}/{len(case_numbers)}] source_case={source_case_number} start",
            flush=True,
        )
        try:
            inputs = resolve_shot_inputs(
                shot_dir=source_dir,
                masks_path=None,
                prefer_saved_masks=not args.force_sam3_inference,
            )
            case_config = copy.deepcopy(config)
            result = run_offline_shelf_storage_detection(
                config=case_config,
                output_dir=case_output_dir,
                run_metadata=run_metadata,
                **inputs,
            )
        except Exception as exc:
            case_output_dir.mkdir(parents=True, exist_ok=True)
            reason = f"batch_wrapper_error: {type(exc).__name__}: {exc}"
            result = {
                **run_metadata,
                "ok": False,
                "pipeline_ok": False,
                "failure_stage": "batch_wrapper",
                "failure_reason": reason,
                "reason": reason,
                "candidates": [],
                "processing_time_sec": None,
                "artifacts": {
                    "summary.json": str(case_output_dir / "summary.json"),
                    "run.log": str(case_output_dir / "run.log"),
                },
            }
            _save_json(case_output_dir / "summary.json", result)
            (case_output_dir / "run.log").write_text(
                (
                    f"{datetime.now().isoformat(timespec='milliseconds')} "
                    f"run_metadata={json.dumps(run_metadata, ensure_ascii=False)}\n"
                    f"{datetime.now().isoformat(timespec='milliseconds')} "
                    f"{reason}\n"
                ),
                encoding="utf-8",
            )

        candidate_status_counts = {
            status: sum(
                1
                for candidate in result.get("candidates", [])
                if candidate.get("status") == status
            )
            for status in ("accepted", "uncertain", "rejected")
        }
        case_record = {
            **run_metadata,
            "pipeline_ok": bool(result.get("pipeline_ok", result.get("ok"))),
            "failure_stage": result.get("failure_stage"),
            "failure_reason": result.get("failure_reason", result.get("reason")),
            "selected_candidate_id": result.get("selected_candidate_id"),
            "candidate_status_counts": candidate_status_counts,
            "processing_time_sec": result.get("processing_time_sec"),
        }
        case_results.append(case_record)
        batch_log(
            f"case_complete output_index={output_index} "
            f"source_case_number={source_case_number} "
            f"pipeline_ok={case_record['pipeline_ok']} "
            f"failure_stage={case_record['failure_stage']} "
            f"failure_reason={case_record['failure_reason']} "
            f"candidate_status_counts={json.dumps(candidate_status_counts)}"
        )
        print(
            f"[{output_index}/{len(case_numbers)}] source_case={source_case_number} "
            f"ok={case_record['pipeline_ok']} statuses={candidate_status_counts}",
            flush=True,
        )
        checkpoint = _batch_summary(
            batch_timestamp=batch_timestamp,
            input_case_count=len(case_numbers),
            case_results=case_results,
            started=batch_started,
        )
        _save_json(batch_dir / "batch_summary.json", checkpoint)

    summary = _batch_summary(
        batch_timestamp=batch_timestamp,
        input_case_count=len(case_numbers),
        case_results=case_results,
        started=batch_started,
    )
    _save_json(batch_dir / "batch_summary.json", summary)
    batch_log(
        f"batch_complete completed={summary['completed_case_count']} "
        f"failed={summary['failed_case_count']} "
        f"total_elapsed_sec={summary['total_elapsed_sec']:.6f}"
    )
    summary["output_dir"] = str(batch_dir)
    return (0 if summary["failed_case_count"] == 0 else 1), summary


def main() -> int:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.output_root is not None:
        config["output"]["root_dir"] = str(args.output_root)
    _configure_dimensions(config, args)

    if args.batch_case_numbers:
        try:
            case_numbers = _parse_case_numbers(args.batch_case_numbers)
            exit_code, summary = _run_batch(args, config, case_numbers)
        except Exception as exc:
            print(
                json.dumps(
                    {"ok": False, "reason": f"{type(exc).__name__}: {exc}"},
                    ensure_ascii=False,
                )
            )
            return 2
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return exit_code

    try:
        inputs = resolve_shot_inputs(
            shot_dir=args.shot_dir,
            rgb_path=args.rgb,
            depth_path=args.depth,
            intrinsics_path=args.intrinsics,
            masks_path=None if args.force_sam3_inference else args.sam3_masks,
            prefer_saved_masks=not args.force_sam3_inference,
        )
        result = run_offline_shelf_storage_detection(config=config, **inputs)
    except Exception as exc:
        print(json.dumps({"ok": False, "reason": str(exc)}, ensure_ascii=False))
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
