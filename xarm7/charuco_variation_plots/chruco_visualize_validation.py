#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sampling_csv内のCSVとhandeye_pairs内のhand-eye JSONを選択し、
選択した校正値でT_base_boardを再計算して7枚のPNGに可視化する。

再計算:
    T_base_board
      = T_base_tcp
      @ JSON["T_cam_tcp"]
      @ T_cam_board

このプロジェクトでは既存互換のため、
JSON["T_cam_tcp"]の実体をcamera座標 -> TCP座標として使用する。

単位:
    CSV T_base_tcp      : mm
    JSON T_cam_tcp      : m -> mmへ変換
    CSV T_cam_board_m   : m -> mmへ変換
    再計算T_base_board : mm

出力:
    charuco_variation_plots直下へ7枚のPNGを上書き保存する。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_FILENAMES = {
    "x": "base_x.png",
    "y": "base_y.png",
    "z": "base_z.png",
    "roll": "base_roll.png",
    "pitch": "base_pitch.png",
    "yaw": "base_yaw.png",
    "distance_3d": "base_distance_3d.png",
}


def matrix_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{r}{c}" for r in range(4) for c in range(4)]


def orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=float).reshape(3, 3))
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def rotation_to_rpy_deg(rotation: np.ndarray) -> np.ndarray:
    rotation = orthonormalize_rotation(rotation)

    sy = float(np.clip(-rotation[2, 0], -1.0, 1.0))
    pitch = math.asin(sy)

    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0

    return np.degrees([roll, pitch, yaw])


def wrap_degrees(values: np.ndarray | float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values + 180.0) % 360.0 - 180.0


def circular_mean_deg(values_deg: np.ndarray) -> float:
    radians = np.radians(np.asarray(values_deg, dtype=float))
    return float(np.degrees(np.arctan2(np.mean(np.sin(radians)),
                                       np.mean(np.cos(radians)))))


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values ** 2)))


def get_reference_index(df: pd.DataFrame) -> int:
    if "sample_type" in df.columns:
        sample_type = df["sample_type"].astype(str).str.strip().str.lower()
        matches = np.flatnonzero(sample_type.eq("reference").to_numpy())
        if matches.size:
            return int(matches[0])
    return 0


def get_sample_axis(df: pd.DataFrame) -> np.ndarray:
    if "sample_id" in df.columns:
        values = pd.to_numeric(df["sample_id"], errors="coerce").to_numpy()
        if np.all(np.isfinite(values)):
            return values
    return np.arange(len(df), dtype=int)



def normalize_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4).copy()
    u, _, vt = np.linalg.svd(transform[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    transform[:3, :3] = rotation
    transform[3] = [0.0, 0.0, 0.0, 1.0]
    return transform


def load_transform_series(
    df: pd.DataFrame,
    prefix: str,
    *,
    translation_multiplier: float,
) -> list[np.ndarray]:
    columns = matrix_columns(prefix)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{prefix}の行列列が不足しています。最初の不足列: {missing[0]}"
        )

    transforms = []
    for csv_row, (_, row) in enumerate(df.iterrows(), start=2):
        values = pd.to_numeric(
            row[columns],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"CSVの{csv_row}行目にある{prefix}に欠損があります。"
            )

        transform = values.reshape(4, 4)
        transform[:3, 3] *= translation_multiplier
        transforms.append(normalize_transform(transform))

    return transforms


def load_camera_to_tcp_from_json(json_path: Path) -> np.ndarray:
    """
    既存robot_base_coordinate.pyと同じ規則で、
    JSON["T_cam_tcp"]をcamera -> TCPとして読む。
    JSONの並進はmなのでmmへ変換する。
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))

    if "T_cam_tcp" not in data:
        raise KeyError(
            f"'T_cam_tcp'がありません: {json_path}; "
            f"available keys={list(data.keys())}"
        )

    transform = np.asarray(
        data["T_cam_tcp"],
        dtype=np.float64,
    ).reshape(4, 4)

    transform[:3, 3] *= 1000.0
    return normalize_transform(transform)


def recompute_base_board_poses(
    df: pd.DataFrame,
    camera_to_tcp_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base_tcp_list = load_transform_series(
        df,
        "T_base_tcp",
        translation_multiplier=1.0,
    )
    cam_board_list = load_transform_series(
        df,
        "T_cam_board_m",
        translation_multiplier=1000.0,
    )

    positions = []
    angles = []

    for base_tcp, cam_board in zip(base_tcp_list, cam_board_list):
        base_board = base_tcp @ camera_to_tcp_mm @ cam_board
        base_board = normalize_transform(base_board)
        positions.append(base_board[:3, 3].copy())
        angles.append(rotation_to_rpy_deg(base_board[:3, :3]))

    return np.asarray(positions), np.asarray(angles)



def add_stats_box(axis: plt.Axes, text: str) -> None:
    axis.text(
        0.985,
        0.97,
        text,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "0.55",
            "alpha": 0.92,
        },
    )


def plot_position_axis(
    sample_ids: np.ndarray,
    positions_mm: np.ndarray,
    reference_index: int,
    axis_index: int,
    axis_name: str,
    output_path: Path,
) -> None:
    values = positions_mm[:, axis_index]
    mean_value = float(np.mean(values))
    reference_value = float(values[reference_index])
    errors = values - reference_value

    figure, axis = plt.subplots(figsize=(12, 6.2))

    axis.plot(
        sample_ids,
        values,
        marker="o",
        markersize=3.5,
        linewidth=1.15,
        label=f"Base {axis_name}",
    )
    axis.axhline(
        mean_value,
        linestyle="--",
        linewidth=1.4,
        label=f"Mean = {mean_value:.4f} mm",
    )
    axis.axhline(
        reference_value,
        linestyle=":",
        linewidth=1.4,
        label=f"Reference = {reference_value:.4f} mm",
    )

    axis.set_title(f"ChArUco marker: base-frame {axis_name} coordinate")
    axis.set_xlabel("Sample")
    axis.set_ylabel(f"Base {axis_name} [mm]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower left", fontsize=9)

    add_stats_box(
        axis,
        (
            f"Base {axis_name}\n"
            f"Mean: {mean_value:.4f} mm\n"
            f"RMSE from reference: {rmse(errors):.4f} mm\n"
            f"Standard deviation: {np.std(values, ddof=0):.4f} mm"
        ),
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_rotation_axis(
    sample_ids: np.ndarray,
    angles_deg: np.ndarray,
    reference_index: int,
    axis_index: int,
    axis_name: str,
    output_path: Path,
) -> None:
    raw_values = angles_deg[:, axis_index]
    reference_value = float(raw_values[reference_index])

    display_values = reference_value + wrap_degrees(raw_values - reference_value)
    mean_wrapped = circular_mean_deg(raw_values)
    mean_for_plot = reference_value + float(
        wrap_degrees(mean_wrapped - reference_value)
    )
    errors = wrap_degrees(raw_values - reference_value)

    figure, axis = plt.subplots(figsize=(12, 6.2))

    axis.plot(
        sample_ids,
        display_values,
        marker="o",
        markersize=3.5,
        linewidth=1.15,
        label=f"Base {axis_name}",
    )
    axis.axhline(
        mean_for_plot,
        linestyle="--",
        linewidth=1.4,
        label=f"Mean = {mean_wrapped:.4f} deg",
    )
    axis.axhline(
        reference_value,
        linestyle=":",
        linewidth=1.4,
        label=f"Reference = {reference_value:.4f} deg",
    )

    axis.set_title(f"ChArUco marker: base-frame {axis_name} angle")
    axis.set_xlabel("Sample")
    axis.set_ylabel(f"Base {axis_name} [deg]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower left", fontsize=9)

    add_stats_box(
        axis,
        (
            f"Base {axis_name}\n"
            f"Mean: {mean_wrapped:.4f} deg\n"
            f"RMSE from reference: {rmse(errors):.4f} deg\n"
            f"Standard deviation: {np.std(errors, ddof=0):.4f} deg"
        ),
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_distance_3d(
    sample_ids: np.ndarray,
    positions_mm: np.ndarray,
    reference_index: int,
    output_path: Path,
) -> None:
    reference_position = positions_mm[reference_index]
    distances = np.linalg.norm(positions_mm - reference_position, axis=1)

    mean_distance = float(np.mean(distances))
    distance_rmse = rmse(distances)
    distance_std = float(np.std(distances, ddof=0))

    figure, axis = plt.subplots(figsize=(12, 6.2))

    axis.plot(
        sample_ids,
        distances,
        marker="o",
        markersize=3.5,
        linewidth=1.15,
        label="3D distance from reference",
    )
    axis.axhline(
        mean_distance,
        linestyle="--",
        linewidth=1.4,
        label=f"Mean = {mean_distance:.4f} mm",
    )

    axis.set_title("ChArUco marker: 3D position error from reference")
    axis.set_xlabel("Sample")
    axis.set_ylabel("3D distance from reference [mm]")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower left", fontsize=9)

    add_stats_box(
        axis,
        (
            "3D position error\n"
            f"Mean: {mean_distance:.4f} mm\n"
            f"RMSE: {distance_rmse:.4f} mm\n"
            f"Standard deviation: {distance_std:.4f} mm"
        ),
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def select_input_csv() -> Path:
    """
    このスクリプトと同じディレクトリにある sampling_csv/ から
    読み込むCSVを選ぶ。

    入力方法:
      - 一覧番号: 1
      - ファイル名:
        charuco_handeye_samples_20260714_190215.csv
      - .csvを省略したファイル名
    """
    script_dir = Path(__file__).resolve().parent
    sampling_csv_dir = script_dir / "sampling_csv"
    sampling_csv_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(
        sampling_csv_dir.glob("*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    print("")
    print("========== SAMPLING CSV ==========")
    print(f"Directory: {sampling_csv_dir}")

    if not csv_files:
        raise ValueError(
            f"CSVがありません: {sampling_csv_dir}"
        )

    print("Available CSV files (newest first):")
    for index, csv_file in enumerate(csv_files, start=1):
        print(f"  {index:2d}: {csv_file.name}")

    print("==================================")
    user_input = input(
        "可視化するCSVの番号またはファイル名を入力してください: "
    ).strip()

    if not user_input:
        raise ValueError("CSVが選択されていません。")

    if user_input.isdigit():
        selected_index = int(user_input)
        if not 1 <= selected_index <= len(csv_files):
            raise ValueError(
                f"番号は1～{len(csv_files)}で入力してください。"
            )
        return csv_files[selected_index - 1].resolve()

    candidate = Path(user_input).expanduser()

    if candidate.suffix.lower() != ".csv":
        candidate = candidate.with_suffix(".csv")

    if candidate.is_absolute():
        return candidate.resolve()

    # ファイル名だけ入力された場合はsampling_csv内として扱う。
    if candidate.parent == Path("."):
        return (sampling_csv_dir / candidate.name).resolve()

    return (Path.cwd() / candidate).resolve()




def select_handeye_json() -> Path:
    script_dir = Path(__file__).resolve().parent
    handeye_dir = script_dir.parent / "handeye_pairs"
    handeye_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(
        handeye_dir.glob("handeye_T_tcp_cam_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    print("")
    print("========== HANDEYE JSON ==========")
    print(f"Directory: {handeye_dir}")

    if not json_files:
        raise ValueError(f"hand-eye JSONがありません: {handeye_dir}")

    print("Available JSON files (newest first):")
    for index, json_file in enumerate(json_files, start=1):
        print(f"  {index:2d}: {json_file.name}")

    print("==================================")
    user_input = input(
        "評価に使うhand-eye JSONの番号またはファイル名を入力してください: "
    ).strip()

    if not user_input:
        raise ValueError("hand-eye JSONが選択されていません。")

    if user_input.isdigit():
        selected_index = int(user_input)
        if not 1 <= selected_index <= len(json_files):
            raise ValueError(
                f"番号は1～{len(json_files)}で入力してください。"
            )
        return json_files[selected_index - 1].resolve()

    candidate = Path(user_input).expanduser()
    if candidate.suffix.lower() != ".json":
        candidate = candidate.with_suffix(".json")

    if candidate.is_absolute():
        return candidate.resolve()

    if candidate.parent == Path("."):
        return (handeye_dir / candidate.name).resolve()

    return (Path.cwd() / candidate).resolve()


def main() -> int:
    try:
        csv_path = select_input_csv()
        handeye_json_path = select_handeye_json()
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(__file__).resolve().parent

    if not csv_path.exists():
        print(f"ERROR: CSVが見つかりません: {csv_path}", file=sys.stderr)
        return 1
    if not handeye_json_path.exists():
        print(
            f"ERROR: hand-eye JSONが見つかりません: {handeye_json_path}",
            file=sys.stderr,
        )
        return 1

    print(f"[INPUT CSV] {csv_path}")
    print(f"[HANDEYE JSON] {handeye_json_path}")
    print(f"[OUTPUT DIR] {output_dir}")

    try:
        df = pd.read_csv(csv_path).dropna(how="all").reset_index(drop=True)
        if df.empty:
            raise ValueError("CSVにデータ行がありません。")

        camera_to_tcp_mm = load_camera_to_tcp_from_json(
            handeye_json_path
        )
        positions_mm, angles_deg = recompute_base_board_poses(
            df,
            camera_to_tcp_mm,
        )
        source = (
            "Recomputed from T_base_tcp, selected JSON['T_cam_tcp'], "
            "and T_cam_board_m"
        )
        reference_index = get_reference_index(df)
        sample_ids = get_sample_axis(df)

        output_dir.mkdir(parents=True, exist_ok=True)

        plot_position_axis(
            sample_ids, positions_mm, reference_index, 0, "X",
            output_dir / OUTPUT_FILENAMES["x"]
        )
        plot_position_axis(
            sample_ids, positions_mm, reference_index, 1, "Y",
            output_dir / OUTPUT_FILENAMES["y"]
        )
        plot_position_axis(
            sample_ids, positions_mm, reference_index, 2, "Z",
            output_dir / OUTPUT_FILENAMES["z"]
        )

        plot_rotation_axis(
            sample_ids, angles_deg, reference_index, 0, "Roll",
            output_dir / OUTPUT_FILENAMES["roll"]
        )
        plot_rotation_axis(
            sample_ids, angles_deg, reference_index, 1, "Pitch",
            output_dir / OUTPUT_FILENAMES["pitch"]
        )
        plot_rotation_axis(
            sample_ids, angles_deg, reference_index, 2, "Yaw",
            output_dir / OUTPUT_FILENAMES["yaw"]
        )

        plot_distance_3d(
            sample_ids, positions_mm, reference_index,
            output_dir / OUTPUT_FILENAMES["distance_3d"]
        )

        print("Generated 7 files (overwritten if they already existed):")
        for key in ("x", "y", "z", "roll", "pitch", "yaw", "distance_3d"):
            print(f"  {output_dir / OUTPUT_FILENAMES[key]}")
        print(f"Reference row index: {reference_index}")
        print(f"Pose source: {source}")
        print(f"Hand-eye JSON: {handeye_json_path}")
        return 0

    except (OSError, ValueError, KeyError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())