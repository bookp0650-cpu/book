#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sampling_csv内のCSVから、従来と同じOpenCV calibrateHandEyeで初期解を求め、
base_y・base_z・base_rollを強めにした重み付き非線形最適化を行う。
その後、固定ボードの自己整合性誤差からMAD法で外れ値を除外する。
さらにbase X/Y/Z/Roll/Pitch/Yawの6軸すべてに絶対上限を設け、
残ったサンプルだけでDANIILIDIS初期解と非線形最適化を再計算する。

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

    /home/book/pro_book/pro_hand_book_python/xarm7/charuco_variation_plots/sampling_csv/
      <入力CSV名>_inliers_YYYYMMDD_HHMMSS.csv

    outlier CSVは出力しない。外れ値情報はJSON内に保存する。

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
from scipy.optimize import least_squares


METHOD_NAME = "DANIILIDIS"
METHOD = cv2.CALIB_HAND_EYE_DANIILIDIS

# =========================================================
# 非線形最適化の重み
# =========================================================
# 数値は「二乗目的関数への寄与倍率」。
# 例: base_y=8.0なら、同じ大きさのbase_x誤差より8倍強く評価する。
POSITION_OBJECTIVE_WEIGHTS = np.array([1.0, 50.0, 30.0], dtype=np.float64)
ROTATION_OBJECTIVE_WEIGHTS = np.array([30.0, 1.0, 1.0], dtype=np.float64)

# 非線形最適化中の外れ値影響を抑えるロバスト損失。
ROBUST_LOSS = "soft_l1"
ROBUST_F_SCALE = 2.0
MAX_NFEV = 3000

# =========================================================
# 外れ値除外
# =========================================================
# 最適化後の各サンプルから求めたbase座標系ボード姿勢について、
# 並進ノルム誤差または回転角誤差が
#   中央値 + OUTLIER_MAD_SCALE * robust_sigma
# を超えたサンプルを除外し、残ったサンプルで再計算する。
# robust_sigma = 1.4826 * MAD
OUTLIER_REJECTION_ENABLED = True
OUTLIER_MAD_SCALE = 3.5
OUTLIER_MAX_ITERATIONS = 10
OUTLIER_MIN_TRANSLATION_SCALE_MM = 0.5
OUTLIER_MIN_ROTATION_SCALE_DEG = 0.1

# Euler角の各軸に対する強制除外上限。
# たとえば代表rollが170 deg付近なら、110 degは約60 deg差なので除外される。
OUTLIER_HARD_RPY_LIMITS_DEG = np.array(
    [15.0, 20.0, 20.0], dtype=np.float64
)

# 最終CSVを書き出す直前に、評価コードと同じreference行を使い、
# base X/Y/Z/Roll/Pitch/Yawの6軸をもう一度検査する。
# 1軸でも上限を超える行が残っていれば除外し、hand-eyeを初期解から再計算する。
#
# 単位:
#   X/Y/Z          : mm
#   Roll/Pitch/Yaw : deg
#
# 必要に応じて、この2つの配列だけ変更すれば軸ごとの許容幅を調整できる。
FINAL_EXPORT_POSITION_LIMITS_MM = np.array(
    [10.0, 10.0, 10.0], dtype=np.float64
)
FINAL_EXPORT_RPY_LIMITS_DEG = np.array(
    [20.0, 20.0, 20.0], dtype=np.float64
)
FINAL_EXPORT_MAX_ITERATIONS = 30
OUTLIER_MIN_INLIER_RATIO = 0.5
MIN_HAND_EYE_SAMPLES = 4

# 入力CSVの検索先。
SAMPLING_CSV_DIR = Path(
    "/home/book/pro_book/pro_hand_book_python/xarm7/"
    "charuco_variation_plots/sampling_csv"
)

# 外れ値除外後のCSVは、必ずこのフォルダへ保存する。
# 保存するのはinlierのみ。outlier CSVは作成しない。
INLIER_CSV_OUTPUT_DIR = Path(
    "/home/book/pro_book/pro_hand_book_python/xarm7/"
    "charuco_variation_plots/sampling_csv"
)


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

    for dataframe_index, (_, row) in enumerate(dataframe.iterrows()):
        csv_row = int(row.get("__source_csv_row__", dataframe_index + 2))
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
    sampling_csv_dir = SAMPLING_CSV_DIR
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


def wrap_degrees(values: np.ndarray | float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values + 180.0) % 360.0 - 180.0


def circular_mean_deg(values_deg: np.ndarray) -> float:
    radians = np.radians(np.asarray(values_deg, dtype=np.float64))
    return float(
        np.degrees(
            np.arctan2(
                np.mean(np.sin(radians)),
                np.mean(np.cos(radians)),
            )
        )
    )


def circular_median_deg(values_deg: np.ndarray) -> float:
    """角度差の絶対値総和を最小にする、外れ値に強い代表角を返す。"""
    values = np.asarray(values_deg, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("円周中央値を計算する角度配列が不正です。")

    costs = np.asarray([
        np.sum(np.abs(wrap_degrees(values - candidate)))
        for candidate in values
    ], dtype=np.float64)
    return float(values[int(np.argmin(costs))])


def rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll) のRPYを返す。"""
    rotation = normalize_transform(
        np.block([
            [np.asarray(rotation, dtype=np.float64).reshape(3, 3),
             np.zeros((3, 1), dtype=np.float64)],
            [np.zeros((1, 3), dtype=np.float64),
             np.ones((1, 1), dtype=np.float64)],
        ])
    )[:3, :3]

    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-8

    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0

    return np.degrees([roll, pitch, yaw])


def transform_to_parameters(transform: np.ndarray) -> np.ndarray:
    """4x4変換を [rotvec(3), translation_m(3)] に変換する。"""
    transform = normalize_transform(transform)
    rotvec, _ = cv2.Rodrigues(transform[:3, :3])
    return np.concatenate([
        np.asarray(rotvec, dtype=np.float64).reshape(3),
        transform[:3, 3],
    ])


def parameters_to_transform(parameters: np.ndarray) -> np.ndarray:
    """[rotvec(3), translation_m(3)] を4x4変換へ戻す。"""
    parameters = np.asarray(parameters, dtype=np.float64).reshape(6)
    rotation, _ = cv2.Rodrigues(parameters[:3].reshape(3, 1))

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = parameters[3:6]
    return normalize_transform(transform)


def average_board_pose(board_poses: list[np.ndarray]) -> np.ndarray:
    translations = np.asarray(
        [transform[:3, 3] for transform in board_poses],
        dtype=np.float64,
    )
    translation = np.median(translations, axis=0)

    rotation_sum = np.zeros((3, 3), dtype=np.float64)
    for transform in board_poses:
        rotation_sum += transform[:3, :3]

    u, _, vt = np.linalg.svd(rotation_sum)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def compute_board_poses(
    base_tcp_list_m: list[np.ndarray],
    cam_board_list_m: list[np.ndarray],
    camera_to_tcp_m: np.ndarray,
) -> list[np.ndarray]:
    return [
        normalize_transform(base_tcp @ camera_to_tcp_m @ cam_board)
        for base_tcp, cam_board in zip(base_tcp_list_m, cam_board_list_m)
    ]



def solve_handeye_daniilidis(
    base_tcp_list_m: list[np.ndarray],
    cam_board_list_m: list[np.ndarray],
) -> np.ndarray:
    """DANIILIDIS法でcamera座標 -> TCP座標を求める。"""
    if len(base_tcp_list_m) != len(cam_board_list_m):
        raise ValueError("T_base_tcpとT_cam_board_mのサンプル数が一致しません。")
    if len(base_tcp_list_m) < MIN_HAND_EYE_SAMPLES:
        raise ValueError(
            f"有効サンプルが{len(base_tcp_list_m)}件しかありません。"
            f"最低{MIN_HAND_EYE_SAMPLES}件必要です。"
        )

    R_gripper2base = [transform[:3, :3] for transform in base_tcp_list_m]
    t_gripper2base = [
        transform[:3, 3].reshape(3, 1)
        for transform in base_tcp_list_m
    ]
    R_target2cam = [transform[:3, :3] for transform in cam_board_list_m]
    t_target2cam = [
        transform[:3, 3].reshape(3, 1)
        for transform in cam_board_list_m
    ]

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=METHOD,
    )

    camera_to_tcp_m = np.eye(4, dtype=np.float64)
    camera_to_tcp_m[:3, :3] = np.asarray(
        R_cam2gripper,
        dtype=np.float64,
    ).reshape(3, 3)
    camera_to_tcp_m[:3, 3] = np.asarray(
        t_cam2gripper,
        dtype=np.float64,
    ).reshape(3)

    if not np.all(np.isfinite(camera_to_tcp_m)):
        raise ValueError(
            "calibrateHandEyeの結果にNaNまたはInfがあります。"
            "姿勢変化が不足している可能性があります。"
        )

    return normalize_transform(camera_to_tcp_m)


def weighted_refinement_residuals(
    parameters: np.ndarray,
    base_tcp_list_m: list[np.ndarray],
    cam_board_list_m: list[np.ndarray],
) -> np.ndarray:
    """
    12変数を最適化する。

      parameters[0:6]  : camera -> TCP
      parameters[6:12] : board -> base（全サンプル共通）

    各サンプルで
      T_base_tcp @ T_camera_to_tcp @ T_cam_board
    が共通のT_base_boardと一致するようにする。

    base_y, base_z, base_rollは上の重み設定で強く評価する。
    """
    camera_to_tcp_m = parameters_to_transform(parameters[:6])
    common_base_board_m = parameters_to_transform(parameters[6:12])

    common_position_m = common_base_board_m[:3, 3]
    common_rpy_deg = rotation_matrix_to_rpy_deg(
        common_base_board_m[:3, :3]
    )

    sqrt_position_weights = np.sqrt(POSITION_OBJECTIVE_WEIGHTS)
    sqrt_rotation_weights = np.sqrt(ROTATION_OBJECTIVE_WEIGHTS)

    residuals: list[float] = []

    for base_tcp, cam_board in zip(base_tcp_list_m, cam_board_list_m):
        predicted_base_board = normalize_transform(
            base_tcp @ camera_to_tcp_m @ cam_board
        )

        # base座標系のX/Y/Z誤差。最適化しやすいようmmで評価する。
        position_error_mm = (
            predicted_base_board[:3, 3] - common_position_m
        ) * 1000.0

        # 可視化コードと同じbase-frame RPYの差を直接評価する。
        predicted_rpy_deg = rotation_matrix_to_rpy_deg(
            predicted_base_board[:3, :3]
        )
        rpy_error_deg = wrap_degrees(
            predicted_rpy_deg - common_rpy_deg
        )

        residuals.extend(
            (position_error_mm * sqrt_position_weights).tolist()
        )
        residuals.extend(
            (rpy_error_deg * sqrt_rotation_weights).tolist()
        )

    return np.asarray(residuals, dtype=np.float64)


def refine_handeye_weighted(
    base_tcp_list_m: list[np.ndarray],
    cam_board_list_m: list[np.ndarray],
    initial_camera_to_tcp_m: np.ndarray,
):
    initial_board_poses = compute_board_poses(
        base_tcp_list_m,
        cam_board_list_m,
        initial_camera_to_tcp_m,
    )
    initial_common_base_board_m = average_board_pose(initial_board_poses)

    x0 = np.concatenate([
        transform_to_parameters(initial_camera_to_tcp_m),
        transform_to_parameters(initial_common_base_board_m),
    ])

    result = least_squares(
        weighted_refinement_residuals,
        x0,
        args=(base_tcp_list_m, cam_board_list_m),
        method="trf",
        loss=ROBUST_LOSS,
        f_scale=ROBUST_F_SCALE,
        x_scale="jac",
        max_nfev=MAX_NFEV,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        verbose=0,
    )

    optimized_camera_to_tcp_m = parameters_to_transform(result.x[:6])
    optimized_common_base_board_m = parameters_to_transform(result.x[6:12])

    return (
        optimized_camera_to_tcp_m,
        optimized_common_base_board_m,
        result,
    )



def compute_sample_consistency_errors(
    base_tcp_list_m: list[np.ndarray],
    cam_board_list_m: list[np.ndarray],
    camera_to_tcp_m: np.ndarray,
    reference_base_board_m: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """各サンプルの固定ボード自己整合性誤差を返す。"""
    board_poses = compute_board_poses(
        base_tcp_list_m,
        cam_board_list_m,
        camera_to_tcp_m,
    )

    if reference_base_board_m is None:
        reference_base_board_m = average_board_pose(board_poses)
    else:
        reference_base_board_m = normalize_transform(reference_base_board_m)

    translations_m = np.asarray(
        [transform[:3, 3] for transform in board_poses],
        dtype=np.float64,
    )
    translation_axis_errors_mm = (
        translations_m - reference_base_board_m[:3, 3]
    ) * 1000.0
    translation_norm_errors_mm = np.linalg.norm(
        translation_axis_errors_mm,
        axis=1,
    )

    rpy_values_deg = np.asarray(
        [rotation_matrix_to_rpy_deg(transform[:3, :3]) for transform in board_poses],
        dtype=np.float64,
    )
    reference_rpy_deg = rotation_matrix_to_rpy_deg(
        reference_base_board_m[:3, :3]
    )
    rpy_axis_errors_deg = wrap_degrees(
        rpy_values_deg - reference_rpy_deg
    )

    rotation_norm_errors_deg = np.asarray([
        rotation_error_deg(
            reference_base_board_m[:3, :3].T @ transform[:3, :3]
        )
        for transform in board_poses
    ], dtype=np.float64)

    return {
        "reference_base_board_m": reference_base_board_m,
        "translation_axis_errors_mm": translation_axis_errors_mm,
        "translation_norm_errors_mm": translation_norm_errors_mm,
        "rpy_axis_errors_deg": rpy_axis_errors_deg,
        "rotation_norm_errors_deg": rotation_norm_errors_deg,
    }


def robust_upper_threshold(
    values: np.ndarray,
    *,
    minimum_scale: float,
) -> dict[str, float]:
    """中央値とMADから片側上限閾値を計算する。"""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("外れ値判定用の誤差配列が不正です。")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    scale = max(robust_sigma, float(minimum_scale))
    threshold = median + OUTLIER_MAD_SCALE * scale

    return {
        "median": median,
        "mad": mad,
        "robust_sigma": robust_sigma,
        "effective_scale": scale,
        "threshold": threshold,
    }


def calculate_handeye_once(
    base_tcp_list_m: list[np.ndarray],
    cam_board_list_m: list[np.ndarray],
) -> dict[str, object]:
    """指定サンプルだけで初期解から最適化まで一度計算する。"""
    initial_camera_to_tcp_m = solve_handeye_daniilidis(
        base_tcp_list_m,
        cam_board_list_m,
    )
    initial_metrics = evaluate_fixed_board_consistency(
        base_tcp_list_m,
        cam_board_list_m,
        initial_camera_to_tcp_m,
    )

    (
        optimized_camera_to_tcp_m,
        optimized_base_board_m,
        optimization_result,
    ) = refine_handeye_weighted(
        base_tcp_list_m,
        cam_board_list_m,
        initial_camera_to_tcp_m,
    )

    optimized_metrics = evaluate_fixed_board_consistency(
        base_tcp_list_m,
        cam_board_list_m,
        optimized_camera_to_tcp_m,
        optimized_base_board_m,
    )

    return {
        "initial_camera_to_tcp_m": initial_camera_to_tcp_m,
        "optimized_camera_to_tcp_m": optimized_camera_to_tcp_m,
        "optimized_base_board_m": optimized_base_board_m,
        "optimization_result": optimization_result,
        "initial_metrics": initial_metrics,
        "optimized_metrics": optimized_metrics,
    }


def reject_outliers_and_recalculate(
    base_tcp_list_m: list[np.ndarray],
    cam_board_list_m: list[np.ndarray],
    source_csv_rows: list[int],
    reference_flags: list[bool] | None = None,
) -> dict[str, object]:
    """
    自己整合性誤差の外れ値を反復除外し、残ったサンプルで再計算する。

    外れ値条件:
      - translation normがMAD閾値を超える
      - rotation normがMAD閾値を超える
      - base roll/pitch/yawのいずれかが絶対上限を超える

    rotation normだけではEuler角の特定軸の飛びを見逃す場合があるため、
    base RPYも個別に監視する。
    """
    sample_count = len(base_tcp_list_m)
    if sample_count != len(cam_board_list_m):
        raise ValueError("変換行列のサンプル数が一致しません。")
    if sample_count != len(source_csv_rows):
        raise ValueError("CSV行番号と変換行列のサンプル数が一致しません。")
    if reference_flags is None:
        reference_flags = [False] * sample_count
    if sample_count != len(reference_flags):
        raise ValueError("referenceフラグと変換行列のサンプル数が一致しません。")

    active_indices = list(range(sample_count))
    rejected_records: list[dict[str, object]] = []
    iteration_history: list[dict[str, object]] = []
    minimum_remaining = max(
        MIN_HAND_EYE_SAMPLES,
        int(math.ceil(sample_count * OUTLIER_MIN_INLIER_RATIO)),
    )
    rpy_names = ("base_roll", "base_pitch", "base_yaw")

    if OUTLIER_REJECTION_ENABLED:
        for iteration in range(1, OUTLIER_MAX_ITERATIONS + 1):
            current_base = [base_tcp_list_m[index] for index in active_indices]
            current_cam = [cam_board_list_m[index] for index in active_indices]
            current_result = calculate_handeye_once(current_base, current_cam)

            errors = compute_sample_consistency_errors(
                current_base,
                current_cam,
                current_result["optimized_camera_to_tcp_m"],
                current_result["optimized_base_board_m"],
            )
            translation_errors = np.asarray(
                errors["translation_norm_errors_mm"], dtype=np.float64
            )
            rotation_errors = np.asarray(
                errors["rotation_norm_errors_deg"], dtype=np.float64
            )
            translation_axis_errors = np.asarray(
                errors["translation_axis_errors_mm"], dtype=np.float64
            )
            rpy_axis_errors = np.asarray(
                errors["rpy_axis_errors_deg"], dtype=np.float64
            )
            abs_rpy_axis_errors = np.abs(rpy_axis_errors)

            translation_stats = robust_upper_threshold(
                translation_errors,
                minimum_scale=OUTLIER_MIN_TRANSLATION_SCALE_MM,
            )
            rotation_stats = robust_upper_threshold(
                rotation_errors,
                minimum_scale=OUTLIER_MIN_ROTATION_SCALE_DEG,
            )

            translation_bad = (
                translation_errors > translation_stats["threshold"]
            )
            rotation_bad = (
                rotation_errors > rotation_stats["threshold"]
            )
            rpy_hard_bad_matrix = (
                abs_rpy_axis_errors > OUTLIER_HARD_RPY_LIMITS_DEG
            )
            rpy_hard_bad = np.any(rpy_hard_bad_matrix, axis=1)

            candidate_local_indices = np.flatnonzero(
                translation_bad | rotation_bad | rpy_hard_bad
            ).tolist()

            max_removable = max(0, len(active_indices) - minimum_remaining)
            if len(candidate_local_indices) > max_removable:
                def candidate_score(local_index: int) -> float:
                    return float(max(
                        translation_errors[local_index]
                        / max(translation_stats["threshold"], 1e-12),
                        rotation_errors[local_index]
                        / max(rotation_stats["threshold"], 1e-12),
                        np.max(
                            abs_rpy_axis_errors[local_index]
                            / OUTLIER_HARD_RPY_LIMITS_DEG
                        ),
                    ))

                candidate_local_indices = sorted(
                    candidate_local_indices,
                    key=candidate_score,
                    reverse=True,
                )[:max_removable]

            candidate_local_set = set(candidate_local_indices)
            rejected_this_iteration: list[dict[str, object]] = []

            for local_index in candidate_local_indices:
                original_index = active_indices[local_index]
                reasons: list[str] = []
                if translation_bad[local_index]:
                    reasons.append("translation_norm")
                if rotation_bad[local_index]:
                    reasons.append("rotation_norm")
                for axis_index, axis_name in enumerate(rpy_names):
                    if rpy_hard_bad_matrix[local_index, axis_index]:
                        reasons.append(axis_name)

                record = {
                    "iteration": iteration,
                    "sample_index_zero_based": int(original_index),
                    "source_csv_row": int(source_csv_rows[original_index]),
                    "translation_error_mm": float(
                        translation_errors[local_index]
                    ),
                    "rotation_error_deg": float(rotation_errors[local_index]),
                    "translation_axis_errors_mm": (
                        translation_axis_errors[local_index].tolist()
                    ),
                    "rpy_axis_errors_deg": (
                        rpy_axis_errors[local_index].tolist()
                    ),
                    "translation_threshold_mm": float(
                        translation_stats["threshold"]
                    ),
                    "rotation_threshold_deg": float(
                        rotation_stats["threshold"]
                    ),
                    "hard_rpy_limits_deg": (
                        OUTLIER_HARD_RPY_LIMITS_DEG.tolist()
                    ),
                    "reason": reasons,
                }
                rejected_records.append(record)
                rejected_this_iteration.append(record)

            iteration_history.append({
                "iteration": iteration,
                "sample_count_before": len(active_indices),
                "translation_statistics": translation_stats,
                "rotation_statistics": rotation_stats,
                "hard_rpy_limits_deg": OUTLIER_HARD_RPY_LIMITS_DEG.tolist(),
                "rejected_source_csv_rows": [
                    record["source_csv_row"]
                    for record in rejected_this_iteration
                ],
            })

            if not candidate_local_indices:
                break

            active_indices = [
                original_index
                for local_index, original_index in enumerate(active_indices)
                if local_index not in candidate_local_set
            ]

    # =====================================================
    # 最終出力ガード（6軸）
    # =====================================================
    # 評価スクリプトと同じreference行を基準として、base X/Y/Z/Roll/Pitch/Yaw
    # の6軸を検査する。さらに、reference行自体が外れ値である可能性に備えて、
    # 各軸の頑健な中心値（XYZは中央値、RPYは円周中央値）からの差も確認する。
    # 1軸でも上限を超えた行がなくなるまで、除外とhand-eye再計算を繰り返す。
    final_guard_history: list[dict[str, object]] = []
    position_names = ("base_x", "base_y", "base_z")

    for guard_iteration in range(1, FINAL_EXPORT_MAX_ITERATIONS + 1):
        final_base = [base_tcp_list_m[index] for index in active_indices]
        final_cam = [cam_board_list_m[index] for index in active_indices]
        final_result = calculate_handeye_once(final_base, final_cam)

        board_poses = compute_board_poses(
            final_base,
            final_cam,
            final_result["optimized_camera_to_tcp_m"],
        )

        positions_mm = np.asarray([
            transform[:3, 3] * 1000.0
            for transform in board_poses
        ], dtype=np.float64)
        rpy_values_deg = np.asarray([
            rotation_matrix_to_rpy_deg(transform[:3, :3])
            for transform in board_poses
        ], dtype=np.float64)

        robust_position_center_mm = np.median(positions_mm, axis=0)
        position_center_errors_mm = (
            positions_mm - robust_position_center_mm
        )

        robust_rpy_center_deg = np.asarray([
            circular_median_deg(rpy_values_deg[:, axis_index])
            for axis_index in range(3)
        ], dtype=np.float64)
        rpy_center_errors_deg = wrap_degrees(
            rpy_values_deg - robust_rpy_center_deg
        )

        reference_local_index = next(
            (
                local_index
                for local_index, original_index in enumerate(active_indices)
                if reference_flags[original_index]
            ),
            0,
        )
        reference_position_mm = positions_mm[reference_local_index]
        position_reference_errors_mm = (
            positions_mm - reference_position_mm
        )

        reference_rpy_deg = rpy_values_deg[reference_local_index]
        rpy_reference_errors_deg = wrap_degrees(
            rpy_values_deg - reference_rpy_deg
        )

        position_center_bad_matrix = (
            np.abs(position_center_errors_mm)
            > FINAL_EXPORT_POSITION_LIMITS_MM
        )
        rpy_center_bad_matrix = (
            np.abs(rpy_center_errors_deg)
            > FINAL_EXPORT_RPY_LIMITS_DEG
        )

        position_reference_bad_matrix = (
            np.abs(position_reference_errors_mm)
            > FINAL_EXPORT_POSITION_LIMITS_MM
        )
        rpy_reference_bad_matrix = (
            np.abs(rpy_reference_errors_deg)
            > FINAL_EXPORT_RPY_LIMITS_DEG
        )

        # reference行自身が頑健な中心から外れている場合、他の正常行をまとめて
        # 誤除外しないよう、この反復ではreference行だけを先に除外する。
        reference_is_bad = bool(
            np.any(position_center_bad_matrix[reference_local_index])
            or np.any(rpy_center_bad_matrix[reference_local_index])
        )

        if reference_is_bad:
            candidate_local_indices = [reference_local_index]
            final_position_bad_matrix = position_center_bad_matrix.copy()
            final_rpy_bad_matrix = rpy_center_bad_matrix.copy()
        else:
            final_position_bad_matrix = (
                position_center_bad_matrix | position_reference_bad_matrix
            )
            final_rpy_bad_matrix = (
                rpy_center_bad_matrix | rpy_reference_bad_matrix
            )
            candidate_local_indices = np.flatnonzero(
                np.any(final_position_bad_matrix, axis=1)
                | np.any(final_rpy_bad_matrix, axis=1)
            ).tolist()

        final_guard_history.append({
            "iteration": guard_iteration,
            "sample_count_before": len(active_indices),
            "reference_source_csv_row": int(
                source_csv_rows[active_indices[reference_local_index]]
            ),
            "reference_position_mm": reference_position_mm.tolist(),
            "reference_rpy_deg": reference_rpy_deg.tolist(),
            "robust_center_position_mm": (
                robust_position_center_mm.tolist()
            ),
            "robust_center_rpy_deg": robust_rpy_center_deg.tolist(),
            "limits_position_mm": (
                FINAL_EXPORT_POSITION_LIMITS_MM.tolist()
            ),
            "limits_rpy_deg": FINAL_EXPORT_RPY_LIMITS_DEG.tolist(),
            "reference_was_outlier": reference_is_bad,
            "rejected_source_csv_rows": [
                int(source_csv_rows[active_indices[local_index]])
                for local_index in candidate_local_indices
            ],
        })

        if not candidate_local_indices:
            # 保存直前の安全確認。6軸のどれかが基準から上限を超えていれば、
            # 外れ値を含むCSVを保存せず停止する。
            max_position_error_mm = np.max(
                np.abs(position_reference_errors_mm), axis=0
            )
            max_rpy_error_deg = np.max(
                np.abs(rpy_reference_errors_deg), axis=0
            )
            if (
                np.any(
                    max_position_error_mm
                    > FINAL_EXPORT_POSITION_LIMITS_MM + 1e-9
                )
                or np.any(
                    max_rpy_error_deg
                    > FINAL_EXPORT_RPY_LIMITS_DEG + 1e-9
                )
            ):
                raise ValueError(
                    "最終6軸検査の内部確認に失敗しました。"
                    " 外れ値を残したCSVは出力しません。"
                )
            break

        max_removable = len(active_indices) - minimum_remaining
        if len(candidate_local_indices) > max_removable:
            raise ValueError(
                "最終6軸検査で外れ値を除くと有効サンプル数が不足します。"
                f" 現在={len(active_indices)}, "
                f"除外候補={len(candidate_local_indices)}, "
                f"最低保持数={minimum_remaining}。"
                " 外れ値を残したCSVは出力しません。"
            )

        candidate_local_set = set(candidate_local_indices)
        for local_index in candidate_local_indices:
            original_index = active_indices[local_index]
            reasons: list[str] = []

            for axis_index, axis_name in enumerate(position_names):
                if final_position_bad_matrix[local_index, axis_index]:
                    reasons.append(f"final_{axis_name}")
            for axis_index, axis_name in enumerate(rpy_names):
                if final_rpy_bad_matrix[local_index, axis_index]:
                    reasons.append(f"final_{axis_name}")

            rejected_records.append({
                "iteration": f"final_guard_{guard_iteration}",
                "sample_index_zero_based": int(original_index),
                "source_csv_row": int(source_csv_rows[original_index]),
                "position_values_mm": positions_mm[local_index].tolist(),
                "rpy_values_deg": rpy_values_deg[local_index].tolist(),
                "reference_position_mm": reference_position_mm.tolist(),
                "reference_rpy_deg": reference_rpy_deg.tolist(),
                "robust_center_position_mm": (
                    robust_position_center_mm.tolist()
                ),
                "robust_center_rpy_deg": robust_rpy_center_deg.tolist(),
                "position_error_from_reference_mm": (
                    position_reference_errors_mm[local_index].tolist()
                ),
                "rpy_error_from_reference_deg": (
                    rpy_reference_errors_deg[local_index].tolist()
                ),
                "position_error_from_robust_center_mm": (
                    position_center_errors_mm[local_index].tolist()
                ),
                "rpy_error_from_robust_center_deg": (
                    rpy_center_errors_deg[local_index].tolist()
                ),
                "hard_position_limits_mm": (
                    FINAL_EXPORT_POSITION_LIMITS_MM.tolist()
                ),
                "hard_rpy_limits_deg": (
                    FINAL_EXPORT_RPY_LIMITS_DEG.tolist()
                ),
                "reason": reasons,
            })

        active_indices = [
            original_index
            for local_index, original_index in enumerate(active_indices)
            if local_index not in candidate_local_set
        ]
    else:
        raise ValueError(
            "最終6軸外れ値検査が規定回数内に収束しませんでした。"
            " 外れ値を残したCSVは出力しません。"
        )

    # ループ脱出時のfinal_resultは、外れ値が0件と確認されたデータでの結果。
    final_base = [base_tcp_list_m[index] for index in active_indices]
    final_cam = [cam_board_list_m[index] for index in active_indices]

    all_sample_metrics = evaluate_fixed_board_consistency(
        base_tcp_list_m,
        cam_board_list_m,
        final_result["optimized_camera_to_tcp_m"],
        final_result["optimized_base_board_m"],
    )

    return {
        **final_result,
        "inlier_indices": active_indices,
        "outlier_indices": sorted(
            set(range(sample_count)) - set(active_indices)
        ),
        "rejected_records": rejected_records,
        "iteration_history": iteration_history,
        "final_export_guard_history": final_guard_history,
        "all_sample_metrics_after_final_model": all_sample_metrics,
        "minimum_remaining_samples": minimum_remaining,
    }


def evaluate_fixed_board_consistency(
    base_tcp_list_m: list[np.ndarray],
    cam_board_list_m: list[np.ndarray],
    camera_to_tcp_m: np.ndarray,
    reference_base_board_m: np.ndarray | None = None,
) -> dict[str, float]:
    board_poses = compute_board_poses(
        base_tcp_list_m,
        cam_board_list_m,
        camera_to_tcp_m,
    )

    if reference_base_board_m is None:
        reference_base_board_m = average_board_pose(board_poses)
    else:
        reference_base_board_m = normalize_transform(reference_base_board_m)

    translations_m = np.asarray(
        [transform[:3, 3] for transform in board_poses],
        dtype=np.float64,
    )
    reference_translation_m = reference_base_board_m[:3, 3]
    translation_axis_errors_mm = (
        translations_m - reference_translation_m
    ) * 1000.0
    translation_norm_errors_mm = np.linalg.norm(
        translation_axis_errors_mm,
        axis=1,
    )

    rpy_values_deg = np.asarray(
        [rotation_matrix_to_rpy_deg(transform[:3, :3]) for transform in board_poses],
        dtype=np.float64,
    )
    reference_rpy_deg = rotation_matrix_to_rpy_deg(
        reference_base_board_m[:3, :3]
    )
    rpy_axis_errors_deg = wrap_degrees(
        rpy_values_deg - reference_rpy_deg
    )

    rotation_norm_errors_deg = []
    for transform in board_poses:
        relative_rotation = (
            reference_base_board_m[:3, :3].T
            @ transform[:3, :3]
        )
        rotation_norm_errors_deg.append(
            rotation_error_deg(relative_rotation)
        )
    rotation_norm_errors_deg = np.asarray(
        rotation_norm_errors_deg,
        dtype=np.float64,
    )

    def axis_rmse(values: np.ndarray, index: int) -> float:
        return float(np.sqrt(np.mean(values[:, index] ** 2)))

    return {
        "base_x_rmse_mm": axis_rmse(translation_axis_errors_mm, 0),
        "base_y_rmse_mm": axis_rmse(translation_axis_errors_mm, 1),
        "base_z_rmse_mm": axis_rmse(translation_axis_errors_mm, 2),
        "base_roll_rmse_deg": axis_rmse(rpy_axis_errors_deg, 0),
        "base_pitch_rmse_deg": axis_rmse(rpy_axis_errors_deg, 1),
        "base_yaw_rmse_deg": axis_rmse(rpy_axis_errors_deg, 2),
        "translation_rmse_mm": float(
            np.sqrt(np.mean(translation_norm_errors_mm ** 2))
        ),
        "translation_mean_mm": float(np.mean(translation_norm_errors_mm)),
        "translation_max_mm": float(np.max(translation_norm_errors_mm)),
        "rotation_rmse_deg": float(
            np.sqrt(np.mean(rotation_norm_errors_deg ** 2))
        ),
        "rotation_mean_deg": float(np.mean(rotation_norm_errors_deg)),
        "rotation_max_deg": float(np.max(rotation_norm_errors_deg)),
    }


def print_consistency_metrics(label: str, metrics: dict[str, float]) -> None:
    print(label)
    print(
        "  Base XYZ RMSE [mm]      : "
        f"{metrics['base_x_rmse_mm']:.6f} / "
        f"{metrics['base_y_rmse_mm']:.6f} / "
        f"{metrics['base_z_rmse_mm']:.6f}"
    )
    print(
        "  Base RPY RMSE [deg]     : "
        f"{metrics['base_roll_rmse_deg']:.6f} / "
        f"{metrics['base_pitch_rmse_deg']:.6f} / "
        f"{metrics['base_yaw_rmse_deg']:.6f}"
    )
    print(
        "  Translation mean/RMSE/max [mm]: "
        f"{metrics['translation_mean_mm']:.6f} / "
        f"{metrics['translation_rmse_mm']:.6f} / "
        f"{metrics['translation_max_mm']:.6f}"
    )
    print(
        "  Rotation mean/RMSE/max [deg]   : "
        f"{metrics['rotation_mean_deg']:.6f} / "
        f"{metrics['rotation_rmse_deg']:.6f} / "
        f"{metrics['rotation_max_deg']:.6f}"
    )

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

        raw_dataframe = pd.read_csv(csv_path)
        nonempty_mask = ~raw_dataframe.isna().all(axis=1)
        dataframe = raw_dataframe.loc[nonempty_mask].copy()
        dataframe.insert(
            0,
            "__source_csv_row__",
            dataframe.index.to_numpy(dtype=np.int64) + 2,
        )
        dataframe = dataframe.reset_index(drop=True)

        if len(dataframe) < MIN_HAND_EYE_SAMPLES:
            raise ValueError(
                f"サンプルが{len(dataframe)}件しかありません。"
                f"calibrateHandEyeには最低{MIN_HAND_EYE_SAMPLES}姿勢必要です。"
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
        source_csv_rows = (
            dataframe["__source_csv_row__"].astype(int).tolist()
        )
        if "sample_type" in dataframe.columns:
            reference_flags = (
                dataframe["sample_type"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("reference")
                .tolist()
            )
        else:
            reference_flags = [False] * len(dataframe)

        calculation = reject_outliers_and_recalculate(
            base_tcp_list_m,
            cam_board_list_m,
            source_csv_rows,
            reference_flags,
        )

        initial_T_camera_to_tcp = calculation["initial_camera_to_tcp_m"]
        T_camera_to_tcp = calculation["optimized_camera_to_tcp_m"]
        optimized_T_base_board = calculation["optimized_base_board_m"]
        optimization_result = calculation["optimization_result"]
        initial_metrics = calculation["initial_metrics"]
        optimized_metrics = calculation["optimized_metrics"]
        all_sample_metrics = calculation[
            "all_sample_metrics_after_final_model"
        ]
        inlier_indices = calculation["inlier_indices"]
        outlier_indices = calculation["outlier_indices"]
        rejected_records = calculation["rejected_records"]

        T_tcp_to_camera = inverse_transform(T_camera_to_tcp)

        script_dir = Path(__file__).resolve().parent
        handeye_pairs_dir = script_dir.parent / "handeye_pairs"
        handeye_pairs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            handeye_pairs_dir
            / f"handeye_T_tcp_cam_{timestamp}.json"
        )

        # 外れ値除外後のinlierだけを、元CSVと同じ列構成で保存する。
        # outlier CSVは作成しない。
        INLIER_CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        inlier_csv_path = (
            INLIER_CSV_OUTPUT_DIR
            / f"{csv_path.stem}_inliers_{timestamp}.csv"
        )
        inlier_export = (
            dataframe.iloc[inlier_indices]
            .drop(columns=["__source_csv_row__"])
            .copy()
        )
        inlier_export.to_csv(inlier_csv_path, index=False)

        result = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_csv": str(csv_path),
            "settings": {
                "method": METHOD_NAME,
                "source_T_base_tcp_translation_unit": "mm",
                "source_T_cam_board_translation_unit": "m",
                "output_translation_unit": "m",
                "algorithm": (
                    "iterative MAD outlier rejection + "
                    "OpenCV calibrateHandEye initial solution + "
                    "weighted nonlinear least_squares refinement"
                ),
                "uses_T_base_board_mm": False,
                "nonlinear_refinement": True,
                "optimized_unknown_count": 12,
                "objective_position_weights_xyz": (
                    POSITION_OBJECTIVE_WEIGHTS.tolist()
                ),
                "objective_rotation_weights_rpy": (
                    ROTATION_OBJECTIVE_WEIGHTS.tolist()
                ),
                "robust_loss": ROBUST_LOSS,
                "robust_f_scale": ROBUST_F_SCALE,
                "outlier_rejection_enabled": OUTLIER_REJECTION_ENABLED,
                "outlier_method": (
                    "translation/rotation norm MAD + hard base RPY limits + "
                    "final six-axis XYZ/RPY export guard"
                ),
                "outlier_mad_scale": OUTLIER_MAD_SCALE,
                "outlier_max_iterations": OUTLIER_MAX_ITERATIONS,
                "outlier_min_translation_scale_mm": (
                    OUTLIER_MIN_TRANSLATION_SCALE_MM
                ),
                "outlier_min_rotation_scale_deg": (
                    OUTLIER_MIN_ROTATION_SCALE_DEG
                ),
                "outlier_hard_rpy_limits_deg": (
                    OUTLIER_HARD_RPY_LIMITS_DEG.tolist()
                ),
                "final_export_position_limits_mm": (
                    FINAL_EXPORT_POSITION_LIMITS_MM.tolist()
                ),
                "final_export_rpy_limits_deg": (
                    FINAL_EXPORT_RPY_LIMITS_DEG.tolist()
                ),
                "outlier_min_inlier_ratio": OUTLIER_MIN_INLIER_RATIO,
                "json_key_convention": (
                    "existing_robot_base_coordinate_compatible"
                ),
                "T_cam_tcp_meaning": "camera_to_tcp",
                "T_tcp_cam_meaning": "tcp_to_camera",
            },

            # 従来の逆命名JSONと既存実行コードに合わせる。
            # 実行コードが読むのは外れ値除外後・最適化後の値。
            "T_cam_tcp": to_json_matrix(T_camera_to_tcp),
            "T_tcp_cam": to_json_matrix(T_tcp_to_camera),
            "outlier_rejection": {
                "original_sample_count": len(dataframe),
                "inlier_count": len(inlier_indices),
                "outlier_count": len(outlier_indices),
                "minimum_remaining_samples": calculation[
                    "minimum_remaining_samples"
                ],
                "inlier_source_csv_rows": [
                    source_csv_rows[index]
                    for index in inlier_indices
                ],
                "outlier_source_csv_rows": [
                    source_csv_rows[index]
                    for index in outlier_indices
                ],
                "rejected_samples": rejected_records,
                "iteration_history": calculation["iteration_history"],
                "final_export_guard_history": calculation[
                    "final_export_guard_history"
                ],
                "inlier_csv": str(inlier_csv_path),
            },
            "optimization": {
                "success": bool(optimization_result.success),
                "status": int(optimization_result.status),
                "message": str(optimization_result.message),
                "nfev": int(optimization_result.nfev),
                "cost": float(optimization_result.cost),
                "optimality": float(optimization_result.optimality),
                "initial_T_cam_tcp": to_json_matrix(
                    initial_T_camera_to_tcp
                ),
                "optimized_T_base_board": to_json_matrix(
                    optimized_T_base_board
                ),
                "initial_consistency_inliers": initial_metrics,
                "optimized_consistency_inliers": optimized_metrics,
                "optimized_consistency_all_samples": all_sample_metrics,

                # 旧キーも残す。内容は外れ値除外後のinlier評価。
                "initial_consistency": initial_metrics,
                "optimized_consistency": optimized_metrics,
            },
        }

        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        np.set_printoptions(precision=10, suppress=False)

        print("")
        print("===== OUTLIER-REJECTED WEIGHTED HAND-EYE RESULT =====")
        print(f"CSV          : {csv_path}")
        print(
            f"Samples      : {len(dataframe)} -> "
            f"{len(inlier_indices)} inliers "
            f"({len(outlier_indices)} rejected)"
        )
        print(f"Initial      : OpenCV {METHOD_NAME}")
        print(
            "Weights XYZ / RPY: "
            f"{POSITION_OBJECTIVE_WEIGHTS.tolist()} / "
            f"{ROTATION_OBJECTIVE_WEIGHTS.tolist()}"
        )
        print(
            f"Robust loss  : {ROBUST_LOSS}, "
            f"f_scale={ROBUST_F_SCALE}"
        )
        print(
            "Final evaluator XYZ limits: "
            f"{FINAL_EXPORT_POSITION_LIMITS_MM.tolist()} mm"
        )
        print(
            "Final evaluator RPY limits: "
            f"{FINAL_EXPORT_RPY_LIMITS_DEG.tolist()} deg"
        )
        print(
            "Outlier rule : translation/rotation MAD + hard RPY limits "
            f"{OUTLIER_HARD_RPY_LIMITS_DEG.tolist()} deg"
        )

        if rejected_records:
            print("")
            print("Rejected samples:")
            for record in rejected_records:
                reason = "+".join(record["reason"])
                if "translation_error_mm" in record:
                    detail = (
                        f"trans={record['translation_error_mm']:.6f} mm | "
                        f"rot={record['rotation_error_deg']:.6f} deg | "
                        f"rpy={np.round(record['rpy_axis_errors_deg'], 4).tolist()}"
                    )
                else:
                    xyz_error = np.round(
                        record.get("position_error_from_reference_mm", []), 4
                    ).tolist()
                    rpy_error = np.round(
                        record.get("rpy_error_from_reference_deg", []), 4
                    ).tolist()
                    detail = f"xyz_error={xyz_error} mm | rpy_error={rpy_error} deg"

                print(
                    f"  CSV row {record['source_csv_row']:4d} | "
                    f"{detail} | reason={reason} | "
                    f"iter={record['iteration']}"
                )
        else:
            print("")
            print("Rejected samples: none")

        print("")
        print_consistency_metrics(
            "Before nonlinear refinement (inliers)",
            initial_metrics,
        )
        print("")
        print_consistency_metrics(
            "After nonlinear refinement (inliers)",
            optimized_metrics,
        )
        print("")
        print_consistency_metrics(
            "Final model evaluated on all original samples",
            all_sample_metrics,
        )
        print("")
        print(
            "Optimizer     : "
            f"success={optimization_result.success}, "
            f"nfev={optimization_result.nfev}, "
            f"cost={optimization_result.cost:.10f}"
        )
        print("")
        print("Optimized camera -> TCP")
        print('JSON key: "T_cam_tcp"')
        print(T_camera_to_tcp)
        print("")
        print("Optimized TCP -> camera")
        print('JSON key: "T_tcp_cam"')
        print(T_tcp_to_camera)
        print("")
        print("Generated files:")
        print(f"  JSON     : {output_path}")
        print(f"  Inlier CSV (only): {inlier_csv_path}")
        print("  Outlier CSV      : not generated")
        print("  Outliers : not exported (details are stored in JSON)")
        print("=====================================================")
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