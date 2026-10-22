#!/usr/bin/env python3
"""Re-run current storage recognition on saved failure captures only.

The original capture directories are read-only inputs.  All recognition debug
artifacts are created in a temporary directory; only the final overlay, CSV,
and log are retained under the requested output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import re
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any

from offline_100test_storage_space_sam3 import (
    BOOK_END_MODEL,
    SPINE_MODEL,
    load_npz,
)
from Storage_by_Detection0911 import (
    STORAGE_TARGET_3D_MODE,
    classify_storage_space,
)
from detection.pro_handbook.sam_py_demo.Storage_SAM3 import (
    apply_selected_triangular_space_mask_correction,
    run_saved_rgbd,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "captures" / "入庫認識失敗"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "improvement result"
SUMMARY_FIELDS = (
    "cycle",
    "view",
    "shot_dir",
    "recognition_success",
    "selected_space_id",
    "classification",
    "held_book_detected",
    "triangular_mask_correction_applied",
    "output_png",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline re-recognition of saved storage-failure captures. "
            "No RealSense, ROS, xArm, or Dynamixel entry point is used."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def discover_shot_dirs(input_root: Path, output_root: Path) -> list[Path]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root not found: {input_root}")
    output_resolved = output_root.resolve()
    shots: set[Path] = set()
    for name in ("input_rgb.npy", "input.png"):
        for input_path in input_root.rglob(name):
            shot_dir = input_path.parent
            try:
                shot_dir.resolve().relative_to(output_resolved)
            except ValueError:
                shots.add(shot_dir)
    return sorted(shots, key=lambda path: str(path.relative_to(input_root)))


def capture_identity(input_root: Path, shot_dir: Path) -> tuple[str, str, str]:
    relative = shot_dir.relative_to(input_root)
    parts = relative.parts
    cycle = parts[0] if parts else shot_dir.name
    view = next((part for part in parts if part in {"1st", "2nd", "3rd"}), "unknown")
    shot_timestamp = shot_dir.name
    return cycle, view, shot_timestamp


def preferred_output_name(cycle: str, view: str) -> str:
    match = re.match(r"^(cycle_\d+_\d{8}_\d{6})", cycle)
    cycle_label = match.group(1) if match else cycle
    return f"{cycle_label}_{view}.png"


def reserve_output_path(
    output_root: Path,
    cycle: str,
    view: str,
    shot_timestamp: str,
    reserved: set[Path],
) -> Path:
    candidate = output_root / preferred_output_name(cycle, view)
    if candidate not in reserved and not candidate.exists():
        reserved.add(candidate)
        return candidate
    stem = candidate.stem
    candidate = output_root / f"{stem}_{shot_timestamp}.png"
    suffix = 2
    while candidate in reserved or candidate.exists():
        candidate = output_root / f"{stem}_{shot_timestamp}_{suffix:02d}.png"
        suffix += 1
    reserved.add(candidate)
    return candidate


def read_partial_metadata(work_dir: Path, row: dict[str, Any]) -> None:
    metadata_path = work_dir / "storage_recognition_metadata.json"
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    row["selected_space_id"] = metadata.get("selected_space_id")
    row["held_book_detected"] = bool(
        metadata.get("held_book_spine_indices", [])
    )


def choose_available_overlay(work_dir: Path, complete: bool) -> Path | None:
    names = (
        ("final_space_target_point.png", "final_space_overlay.png")
        if complete
        else ("final_space_overlay.png", "final_space_target_point.png")
    )
    return next((work_dir / name for name in names if (work_dir / name).is_file()), None)


def process_shot(
    *,
    input_root: Path,
    output_root: Path,
    shot_dir: Path,
    reserved_outputs: set[Path],
    log_handle: Any,
) -> dict[str, Any]:
    cycle, view, shot_timestamp = capture_identity(input_root, shot_dir)
    row: dict[str, Any] = {
        "cycle": cycle,
        "view": view,
        "shot_dir": str(shot_dir.resolve()),
        "recognition_success": False,
        "selected_space_id": None,
        "classification": "",
        "held_book_detected": False,
        "triangular_mask_correction_applied": False,
        "output_png": "",
        "error": "",
    }
    output_path = reserve_output_path(
        output_root,
        cycle,
        view,
        shot_timestamp,
        reserved_outputs,
    )

    with tempfile.TemporaryDirectory(prefix="storage-reprocess-") as temp_name:
        work_dir = Path(temp_name)
        complete = False
        try:
            spine_instances = load_npz(shot_dir / SPINE_MODEL.npz_name)
            end_instances = load_npz(shot_dir / BOOK_END_MODEL.npz_name)
            depth_path = shot_dir / "depth_u16_raw.npy"
            if not depth_path.is_file():
                depth_path = shot_dir / "depth_u16.npy"
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(
                log_handle
            ):
                rgb_path = shot_dir / "input_rgb.npy"
                if not rgb_path.is_file():
                    rgb_path = shot_dir / "input.png"
                _, target_cam, result_metadata = run_saved_rgbd(
                    rgb_path,
                    depth_path,
                    shot_dir / "camera_intrinsics.json",
                    shot_dir=work_dir,
                    spine_instances=spine_instances,
                    end_instances=end_instances,
                    target_3d_mode=STORAGE_TARGET_3D_MODE,
                )
                decision = classify_storage_space(result_metadata)
                target_cam, result_metadata = (
                    apply_selected_triangular_space_mask_correction(
                        result_metadata,
                        target_cam,
                        space_type=decision.classification,
                    )
                )
            row["selected_space_id"] = result_metadata.get("selected_space_id")
            row["classification"] = decision.classification
            row["held_book_detected"] = bool(
                result_metadata.get("held_book_spine_indices", [])
            )
            correction = result_metadata.get(
                "triangular_space_mask_correction", {}
            )
            row["triangular_mask_correction_applied"] = bool(
                correction.get("mask_correction_applied", False)
            )
            complete = True
        except Exception as exc:
            read_partial_metadata(work_dir, row)
            row["error"] = f"{type(exc).__name__}: {exc}"
            log_handle.write(
                f"\n[FAILED] {shot_dir}\n{traceback.format_exc()}\n"
            )
            log_handle.flush()

        overlay_path = choose_available_overlay(work_dir, complete)
        if overlay_path is not None:
            shutil.copy2(overlay_path, output_path)
            row["output_png"] = str(output_path.resolve())
        if complete:
            if overlay_path is None:
                row["error"] = "final overlay was not generated"
            else:
                row["recognition_success"] = True
        return row


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    shots = discover_shot_dirs(input_root, output_root)
    rows: list[dict[str, Any]] = []
    reserved_outputs: set[Path] = set()
    log_path = output_root / "run.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write(f"input_root={input_root}\n")
        log_handle.write(f"output_root={output_root}\n")
        log_handle.write(f"capture_count={len(shots)}\n")
        log_handle.write("sam3_mode=precomputed_saved_instances\n")
        log_handle.write(f"target_3d_mode={STORAGE_TARGET_3D_MODE}\n")
        for index, shot_dir in enumerate(shots, start=1):
            print(f"[{index:02d}/{len(shots):02d}] {shot_dir}")
            log_handle.write(f"\n[CAPTURE {index}/{len(shots)}] {shot_dir}\n")
            row = process_shot(
                input_root=input_root,
                output_root=output_root,
                shot_dir=shot_dir,
                reserved_outputs=reserved_outputs,
                log_handle=log_handle,
            )
            rows.append(row)
            print(
                "  success="
                f"{row['recognition_success']} selected={row['selected_space_id']} "
                f"class={row['classification'] or 'n/a'} "
                f"output={Path(row['output_png']).name if row['output_png'] else 'none'}"
            )
    write_summary(output_root / "summary.csv", rows)
    success_count = sum(bool(row["recognition_success"]) for row in rows)
    print(
        f"done: captures={len(rows)} success={success_count} "
        f"failed={len(rows) - success_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
