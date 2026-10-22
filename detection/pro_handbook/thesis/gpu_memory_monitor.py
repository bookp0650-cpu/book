#!/usr/bin/env python3
"""Sample NVIDIA GPU and per-process memory without importing model runtimes."""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


_STOP = False


def record_phase(
    phase: str,
    *,
    monotonic_seconds: float | None = None,
    **metadata,
) -> None:
    """Append a timing marker when BOOK_GPU_PHASE_LOG is configured."""
    raw_path = os.environ.get("BOOK_GPU_PHASE_LOG")
    if not raw_path:
        return
    now_monotonic = (
        time.monotonic() if monotonic_seconds is None else float(monotonic_seconds)
    )
    payload = {
        "phase": str(phase),
        "monotonic_seconds": now_monotonic,
        "epoch_seconds": time.time() + (now_monotonic - time.monotonic()),
        "emitter_pid": os.getpid(),
        **metadata,
    }
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _request_stop(_signum, _frame) -> None:
    global _STOP
    _STOP = True


def _run_query(arguments: list[str]) -> list[list[str]]:
    completed = subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[list[str]] = []
    for line in completed.stdout.splitlines():
        if line.strip():
            rows.append([part.strip() for part in line.split(",")])
    return rows


def _sample() -> tuple[list[list[str]], list[list[str]]]:
    gpu_rows = _run_query(
        [
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    process_rows = _run_query(
        [
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return gpu_rows, process_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()
    if args.interval < 0.05:
        parser.error("--interval must be at least 0.05 seconds")
    return args


def main() -> int:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    fieldnames = [
        "timestamp_utc",
        "epoch_seconds",
        "monotonic_seconds",
        "gpu_index",
        "gpu_name",
        "gpu_used_memory_mib",
        "gpu_total_memory_mib",
        "gpu_utilization_percent",
        "process_pid",
        "process_name",
        "process_used_memory_mib",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        handle.flush()
        while not _STOP:
            started = time.monotonic()
            epoch = time.time()
            try:
                gpu_rows, process_rows = _sample()
                processes = process_rows or [["", "", "", ""]]
                for gpu in gpu_rows:
                    for process in processes:
                        writer.writerow(
                            {
                                "timestamp_utc": datetime.fromtimestamp(
                                    epoch, tz=timezone.utc
                                ).isoformat(),
                                "epoch_seconds": f"{epoch:.9f}",
                                "monotonic_seconds": f"{started:.9f}",
                                "gpu_index": gpu[0],
                                "gpu_name": gpu[1],
                                "gpu_used_memory_mib": gpu[2],
                                "gpu_total_memory_mib": gpu[3],
                                "gpu_utilization_percent": gpu[4],
                                "process_pid": process[1] if len(process) > 1 else "",
                                "process_name": process[2] if len(process) > 2 else "",
                                "process_used_memory_mib": (
                                    process[3] if len(process) > 3 else ""
                                ),
                            }
                        )
                handle.flush()
            except Exception as exc:
                writer.writerow(
                    {
                        "timestamp_utc": datetime.fromtimestamp(
                            epoch, tz=timezone.utc
                        ).isoformat(),
                        "epoch_seconds": f"{epoch:.9f}",
                        "monotonic_seconds": f"{started:.9f}",
                        "process_name": f"MONITOR_ERROR: {type(exc).__name__}: {exc}",
                    }
                )
                handle.flush()
            remaining = args.interval - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
