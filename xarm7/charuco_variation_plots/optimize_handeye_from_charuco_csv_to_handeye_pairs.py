#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sampling_csv内のCSVから、従来と同じOpenCV calibrateHandEyeで
eye-in-handの変換行列を求める。

使用する列:
    T_base_tcp
        TCP座標 -> base座標
        平行移動単位: mm

    T_cam_board_m
        ChArUco board座標 -> camera座標
        平行移動単位: m

計算:
    cv2.calibrateHandEye(
        R_gripper2base = R_base_tcp,
        t_gripper2base = t_base_tcp,
        R_target2cam   = R_cam_board,
        t_target2cam   = t_cam_board,
        method         = DANIILIDIS,
    )

OpenCVの戻り値:
    camera座標 -> TCP座標

既存のrobot_base_coordinate.pyとの互換性を維持するため、
従来JSONと同じキー配置で保存する:
    JSON["T_cam_tcp"] = camera座標 -> TCP座標
    JSON["T_tcp_cam"] = TCP座標 -> camera座標

注意:
    キー名は数学的な一般命名とは逆だが、
    既存実行コードとの互換性を優先する。

出力:
    xarm7/handeye_pairs/
      handeye_T_tcp_cam_YYYYMMDD_HHMMSS.json

VS Codeの実行ボタンから、そのまま実行できる。
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


METHOD_NAME = "DANIILIDIS"
METHOD = cv2.CALIB_HAND_EYE_DANIILIDIS


def matrix_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{row}{column}" for row in range(4) for column in range(4)]


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


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    transform = normalize_transform(transform)

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return result


def load_transform_series(
    dataframe: pd.DataFrame,
    prefix: str,
    *,
    translation_multiplier: float,
) -> list[np.ndarray]:
    columns = matrix_columns(prefix)
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            f"{prefix}の行列列が不足しています。最初の不足列: {missing[0]}"
        )

    transforms: list[np.ndarray] = []

    for csv_row, (_, row) in enumerate(dataframe.iterrows(), start=2):
        values = pd.to_numeric(
            row[columns],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"CSVの{csv_row}行目にある{prefix}にNaNまたは文字列があります。"
            )

        transform = values.reshape(4, 4)
        transform[:3, 3] *= translation_multiplier
        transform = normalize_transform(transform)

        if not np.allclose(
            transform[3],
            [0.0, 0.0, 0.0, 1.0],
            atol=1e-9,
        ):
            raise ValueError(
                f"CSVの{csv_row}行目にある{prefix}の最終行が不正です。"
            )

        transforms.append(transform)

    return transforms


def select_input_csv() -> Path:
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
        raise ValueError(f"CSVがありません: {sampling_csv_dir}")

    print("Available CSV files (newest first):")
    for index, csv_file in enumerate(csv_files, start=1):
        print(f"  {index:2d}: {csv_file.name}")

    print("==================================")
    user_input = input(
        "変換行列を計算するCSVの番号またはファイル名を入力してください: "
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

    if candidate.parent == Path("."):
        return (sampling_csv_dir / candidate.name).resolve()

    return (Path.cwd() / candidate).resolve()


def rotation_error_deg(rotation: np.ndarray) -> float:
    cosine = float(
        np.clip(
            (np.trace(rotation) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
    )
    return math.degrees(math.acos(cosine))


def evaluate_fixed_board_consistency(
    base_tcp_list_m: list[np.ndarray],
    cam_board_list_m: list[np.ndarray],
    tcp_cam_m: np.ndarray,
) -> dict[str, float]:
    """
    計算結果の診断だけを行う。
    校正値の計算には使用しない。
    """
    board_poses = [
        base_tcp @ tcp_cam_m @ cam_board
        for base_tcp, cam_board in zip(base_tcp_list_m, cam_board_list_m)
    ]

    translations = np.asarray(
        [transform[:3, 3] for transform in board_poses],
        dtype=np.float64,
    )
    reference_translation = np.median(translations, axis=0)

    rotation_sum = np.zeros((3, 3), dtype=np.float64)
    for transform in board_poses:
        rotation_sum += transform[:3, :3]

    u, _, vt = np.linalg.svd(rotation_sum)
    reference_rotation = u @ vt
    if np.linalg.det(reference_rotation) < 0.0:
        u[:, -1] *= -1.0
        reference_rotation = u @ vt

    translation_errors_m = np.linalg.norm(
        translations - reference_translation,
        axis=1,
    )

    rotation_errors_deg = []
    for transform in board_poses:
        relative_rotation = reference_rotation.T @ transform[:3, :3]
        rotation_errors_deg.append(rotation_error_deg(relative_rotation))

    rotation_errors_deg = np.asarray(rotation_errors_deg, dtype=np.float64)

    return {
        "translation_rmse_mm": float(
            np.sqrt(np.mean(translation_errors_m ** 2)) * 1000.0
        ),
        "translation_mean_mm": float(
            np.mean(translation_errors_m) * 1000.0
        ),
        "translation_max_mm": float(
            np.max(translation_errors_m) * 1000.0
        ),
        "rotation_rmse_deg": float(
            np.sqrt(np.mean(rotation_errors_deg ** 2))
        ),
        "rotation_mean_deg": float(np.mean(rotation_errors_deg)),
        "rotation_max_deg": float(np.max(rotation_errors_deg)),
    }


def to_json_matrix(transform: np.ndarray) -> list[list[float]]:
    transform = normalize_transform(transform)
    return [
        [float(value) for value in row]
        for row in transform.tolist()
    ]


def main() -> int:
    try:
        csv_path = select_input_csv()

        if not csv_path.exists():
            raise ValueError(f"CSVが見つかりません: {csv_path}")

        dataframe = (
            pd.read_csv(csv_path)
            .dropna(how="all")
            .reset_index(drop=True)
        )

        if len(dataframe) < 4:
            raise ValueError(
                f"サンプルが{len(dataframe)}件しかありません。"
                "calibrateHandEyeには最低4姿勢程度が必要です。"
            )

        # T_base_tcpはCSV内でmmなのでmへ変換。
        base_tcp_list_m = load_transform_series(
            dataframe,
            "T_base_tcp",
            translation_multiplier=1e-3,
        )

        # T_cam_board_mはCSV内ですでにm。
        cam_board_list_m = load_transform_series(
            dataframe,
            "T_cam_board_m",
            translation_multiplier=1.0,
        )

        R_gripper2base = [
            transform[:3, :3]
            for transform in base_tcp_list_m
        ]
        t_gripper2base = [
            transform[:3, 3].reshape(3, 1)
            for transform in base_tcp_list_m
        ]
        R_target2cam = [
            transform[:3, :3]
            for transform in cam_board_list_m
        ]
        t_target2cam = [
            transform[:3, 3].reshape(3, 1)
            for transform in cam_board_list_m
        ]

        # 従来コードと同じOpenCV hand-eye計算。
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base,
            t_gripper2base,
            R_target2cam,
            t_target2cam,
            method=METHOD,
        )

        # OpenCVのcam2gripperは、camera座標 -> TCP座標。
        T_camera_to_tcp = np.eye(4, dtype=np.float64)
        T_camera_to_tcp[:3, :3] = np.asarray(
            R_cam2gripper,
            dtype=np.float64,
        ).reshape(3, 3)
        T_camera_to_tcp[:3, 3] = np.asarray(
            t_cam2gripper,
            dtype=np.float64,
        ).reshape(3)
        T_camera_to_tcp = normalize_transform(T_camera_to_tcp)

        T_tcp_to_camera = inverse_transform(T_camera_to_tcp)

        metrics = evaluate_fixed_board_consistency(
            base_tcp_list_m,
            cam_board_list_m,
            T_camera_to_tcp,
        )

        script_dir = Path(__file__).resolve().parent
        handeye_pairs_dir = script_dir.parent / "handeye_pairs"
        handeye_pairs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            handeye_pairs_dir
            / f"handeye_T_tcp_cam_{timestamp}.json"
        )

        result = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_csv": str(csv_path),
            "settings": {
                "method": METHOD_NAME,
                "source_T_base_tcp_translation_unit": "mm",
                "source_T_cam_board_translation_unit": "m",
                "output_translation_unit": "m",
                "algorithm": "OpenCV calibrateHandEye",
                "uses_T_base_board_mm": False,
                "nonlinear_refinement": False,
                "json_key_convention":
                    "existing_robot_base_coordinate_compatible",
                "T_cam_tcp_meaning": "camera_to_tcp",
                "T_tcp_cam_meaning": "tcp_to_camera",
            },

            # 従来の逆命名JSONと既存実行コードに合わせる。
            "T_cam_tcp": to_json_matrix(T_camera_to_tcp),
            "T_tcp_cam": to_json_matrix(T_tcp_to_camera),
        }

        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        np.set_printoptions(precision=10, suppress=False)

        print("")
        print("========== OPENCV HAND-EYE RESULT ==========")
        print(f"CSV          : {csv_path}")
        print(f"Samples      : {len(dataframe)}")
        print(f"Method       : {METHOD_NAME}")
        print("")
        print("camera -> TCP")
        print('JSON key: "T_cam_tcp"')
        print(T_camera_to_tcp)
        print("")
        print("TCP -> camera")
        print('JSON key: "T_tcp_cam"')
        print(T_tcp_to_camera)
        print("")
        print("Fixed-board consistency diagnostics")
        print(
            "Translation mean/RMSE/max [mm]: "
            f"{metrics['translation_mean_mm']:.6f} / "
            f"{metrics['translation_rmse_mm']:.6f} / "
            f"{metrics['translation_max_mm']:.6f}"
        )
        print(
            "Rotation mean/RMSE/max [deg]: "
            f"{metrics['rotation_mean_deg']:.6f} / "
            f"{metrics['rotation_rmse_deg']:.6f} / "
            f"{metrics['rotation_max_deg']:.6f}"
        )
        print("")
        print("Generated JSON only:")
        print(f"  {output_path}")
        print("============================================")
        return 0

    except (
        OSError,
        ValueError,
        KeyError,
        cv2.error,
        pd.errors.ParserError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())