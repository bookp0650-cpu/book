#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hand-eye JSONの回転だけを手動補正し、修正後の行列をログ表示する。

・入力JSONは INPUT_JSON をコード内で変更する
・補正角は CORRECTION_*_DEG をコード内で変更する
・JSONファイルへの書き込みは行わない
・T_tcp_camは、修正後のT_cam_tcpから逆行列として自動計算する

このプロジェクトでの行列の意味:
    T_cam_tcp : camera座標 -> TCP（ハンド）座標
    T_tcp_cam : TCP（ハンド）座標 -> camera座標

補正方法:
    修正後の回転 = TCP座標系基準の補正回転 @ 元の回転

したがって、
    CORRECTION_ROLL_DEG  は TCPの+X軸まわり
    CORRECTION_PITCH_DEG は TCPの+Y軸まわり
    CORRECTION_YAW_DEG   は TCPの+Z軸まわり
の補正角を表す。

正の回転方向は右ねじの法則。

----------------------------------------------------------------------
現在提示されているT_cam_tcpの回転行列

    [[ 0,  0, -1],
     [ 1,  0,  0],
     [ 0, -1,  0]]

が表しているカメラ姿勢（TCP座標系に対するカメラ姿勢）は、

    Camera +X軸 = TCP +Y軸方向
    Camera +Y軸 = TCP -Z軸方向
    Camera +Z軸 = TCP -X軸方向

RPYを
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
のZYX規約で表すと、

    roll  = -90 deg
    pitch =   0 deg
    yaw   = +90 deg

となる。

※ Euler角には同じ姿勢を表す別表現が存在するため、
  行列または各座標軸の向きも合わせて確認すること。
----------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


# ============================================================
# ここを自分で変更する
# ============================================================

INPUT_JSON = Path(
"/home/book/pro_book/pro_hand_book_python/xarm7/handeye_pairs/handeye_T_tcp_cam_20260717_223007 copy.json"
)

# TCP（ハンド）座標系の各軸まわりの補正角 [deg]
#
# 例:
#   roll方向に+1度補正:
#       CORRECTION_ROLL_DEG = 1.0
#
#   逆方向へ回したい場合:
#       CORRECTION_ROLL_DEG = -1.0
#
# 一度に複数軸を変更するより、最初は1軸ずつ調整する方が分かりやすい。
CORRECTION_ROLL_DEG = 3.0
CORRECTION_PITCH_DEG = 0.0
CORRECTION_YAW_DEG = 0.0

# ログに表示する小数点以下の桁数
PRINT_DECIMALS = 10
SCRIPT_VERSION = "horizontal-v2"


# ============================================================
# 回転・変換行列
# ============================================================

def rotation_x_deg(angle_deg: float) -> np.ndarray:
    """X軸まわりの右手系回転行列。"""
    angle_rad = math.radians(angle_deg)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def rotation_y_deg(angle_deg: float) -> np.ndarray:
    """Y軸まわりの右手系回転行列。"""
    angle_rad = math.radians(angle_deg)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def rotation_z_deg(angle_deg: float) -> np.ndarray:
    """Z軸まわりの右手系回転行列。"""
    angle_rad = math.radians(angle_deg)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    """数値誤差を除き、正しい回転行列へ補正する。"""
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    u, _, vt = np.linalg.svd(rotation)
    normalized = u @ vt

    if np.linalg.det(normalized) < 0.0:
        u[:, -1] *= -1.0
        normalized = u @ vt

    return normalized


def normalize_transform(transform: np.ndarray) -> np.ndarray:
    """4×4同次変換行列を正規化する。"""
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4).copy()
    transform[:3, :3] = orthonormalize_rotation(transform[:3, :3])
    transform[3] = [0.0, 0.0, 0.0, 1.0]
    return transform


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    """剛体変換の逆行列を計算する。"""
    transform = normalize_transform(transform)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> np.ndarray:
    """
    回転行列をZYX規約のRPYへ変換する。

        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    戻り値:
        [roll, pitch, yaw] [deg]
    """
    rotation = orthonormalize_rotation(rotation)
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


def clean_small_values(matrix: np.ndarray, threshold: float = 1e-12) -> np.ndarray:
    """-0.0や極小の数値を0.0へ揃える。"""
    result = np.asarray(matrix, dtype=np.float64).copy()
    result[np.abs(result) < threshold] = 0.0
    return result


def format_number(value: float) -> str:
    """見やすいJSON用の数値文字列へ変換する。"""
    value = float(value)

    if abs(value) < 10.0 ** (-PRINT_DECIMALS):
        value = 0.0

    text = f"{value:.{PRINT_DECIMALS}f}".rstrip("0").rstrip(".")

    # JSON上で整数に見える値も、行列では0.0/1.0の形に統一する。
    if "." not in text:
        text += ".0"

    if text == "-0.0":
        text = "0.0"

    return text


def format_matrix_block(key: str, matrix: np.ndarray, *, trailing_comma: bool) -> str:
    """4×4行列を、1行につき1行列行の見やすいJSON断片へ整形する。"""
    matrix = clean_small_values(matrix)

    lines = [f'  "{key}": [']
    for row_index, row in enumerate(matrix):
        row_text = ", ".join(format_number(value) for value in row)
        comma = "," if row_index < len(matrix) - 1 else ""
        lines.append(f"    [{row_text}]{comma}")

    closing = "  ]," if trailing_comma else "  ]"
    lines.append(closing)
    return "\n".join(lines)


def format_output_fragment(
    T_cam_tcp: np.ndarray,
    T_tcp_cam: np.ndarray,
) -> str:
    """貼り付けやすい横並び形式で2つの行列を返す。"""
    return "\n".join(
        [
            format_matrix_block(
                "T_cam_tcp",
                T_cam_tcp,
                trailing_comma=True,
            ),
            format_matrix_block(
                "T_tcp_cam",
                T_tcp_cam,
                trailing_comma=False,
            ),
        ]
    )


# ============================================================
# メイン処理
# ============================================================

def main() -> int:
    try:
        if not INPUT_JSON.exists():
            raise FileNotFoundError(f"入力JSONが見つかりません: {INPUT_JSON}")

        data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))

        if "T_cam_tcp" not in data:
            raise KeyError(f"'T_cam_tcp'がありません: {INPUT_JSON}")

        original_T_cam_tcp = normalize_transform(
            np.asarray(data["T_cam_tcp"], dtype=np.float64).reshape(4, 4)
        )

        original_rpy_deg = rotation_matrix_to_rpy_deg(
            original_T_cam_tcp[:3, :3]
        )

        # TCP座標系基準の補正回転。
        # ZYX規約:
        #   R_correction = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        correction_rotation = (
            rotation_z_deg(CORRECTION_YAW_DEG)
            @ rotation_y_deg(CORRECTION_PITCH_DEG)
            @ rotation_x_deg(CORRECTION_ROLL_DEG)
        )

        corrected_T_cam_tcp = original_T_cam_tcp.copy()

        # 左から掛けるため、TCP（ハンド）座標系基準の補正になる。
        corrected_T_cam_tcp[:3, :3] = (
            correction_rotation @ original_T_cam_tcp[:3, :3]
        )
        corrected_T_cam_tcp = normalize_transform(corrected_T_cam_tcp)

        # 並進成分は変更しない。
        corrected_T_cam_tcp[:3, 3] = original_T_cam_tcp[:3, 3]

        # T_tcp_camは手入力せず、必ず逆行列から計算する。
        corrected_T_tcp_cam = inverse_transform(corrected_T_cam_tcp)

        corrected_rpy_deg = rotation_matrix_to_rpy_deg(
            corrected_T_cam_tcp[:3, :3]
        )

        print("")
        print("========== MANUAL ROTATION CORRECTION ==========")
        print(f"Script version: {SCRIPT_VERSION}")
        print(f"Input JSON : {INPUT_JSON}")
        print("")
        print(
            "Correction in TCP frame [deg]"
            f"  roll={CORRECTION_ROLL_DEG:+.6f},"
            f" pitch={CORRECTION_PITCH_DEG:+.6f},"
            f" yaw={CORRECTION_YAW_DEG:+.6f}"
        )
        print("")
        print("Original camera pose relative to TCP [roll, pitch, yaw] [deg]:")
        print(
            "  "
            f"[{original_rpy_deg[0]:.6f}, "
            f"{original_rpy_deg[1]:.6f}, "
            f"{original_rpy_deg[2]:.6f}]"
        )
        print("")
        print("Corrected camera pose relative to TCP [roll, pitch, yaw] [deg]:")
        print(
            "  "
            f"[{corrected_rpy_deg[0]:.6f}, "
            f"{corrected_rpy_deg[1]:.6f}, "
            f"{corrected_rpy_deg[2]:.6f}]"
        )
        print("")
        print("Copy the following two matrices into the JSON:")
        print("")

        # ログへ表示するだけで、ファイルへの書き込みは行わない。
        # 各4×4行列の1行を横並びで表示する。
        print(
            format_output_fragment(
                corrected_T_cam_tcp,
                corrected_T_tcp_cam,
            )
        )

        print("")
        print("No file was written.")
        print("================================================")
        return 0

    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())