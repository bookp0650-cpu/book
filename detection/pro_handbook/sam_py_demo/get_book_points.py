#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RealSense 1ショット撮影 → SAM(infer_for_storage.infer_masks)で書籍領域認識
→ 対象書籍のRGB/Depth保存 → intrinsics+depthで点群化 → PLY保存
→ PCAで背表紙方向などの特徴量を計算

モジュールとしての使い方例:
  from rs_book_capture_and_pointcloud import run_capture_and_pca

  theta_rad, p_min, p_max = run_capture_and_pca(out_dir="captures")

スクリプトとしての使い方例:
  python rs_book_capture_and_pointcloud.py --out captures
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import open3d as o3d
import numpy as np
import cv2
from PIL import Image
import json
import subprocess
import re
import traceback
import os
import sys
import csv
from .OCR.only_one import find_similar_books
from .OCR.only_one_tilted import match_text_to_mask_main

# ===== RealSense =====
import pyrealsense2 as rs

# ===== SAM: infer_for_storage の infer_masks を使う =====
from .infer_for_storage import SamConfig, SamBatchInfer_storage, StageSaveCfg
from modules.overlay_io import _render_overlay_bgr, _save_points_and_overlay
#from bar_code.code_1_pic import detect_barcode
# 点群モジュール
from modules.pointcloud_utils import masked_depth_to_points, save_ply_ascii
from modules.calculate_3D_point_or_RANSAC import calculate_yaw
from modules.pca_vector import pca_axes_fix_dir
from modules.book_width import estimate_book_width
from modules.grip_point import find_target_point
from modules.open3d_view import visualize_points_and_target_open3d

ALLOWABLE_RANGE_Z = 0.07


# ===== Safe comparison switches =====
# 失敗ケース診断を反映した比較用スイッチ。
# RANSACあり安全版では True / True、RANSACなし版では False / False にする。
ENABLE_RANSAC_SPINE_PLANE = True
ENABLE_POST_RANSAC_PLANE_A95 = True

# clean rectangle判定でも列長フィルタを試す実験用。
# 成功ケースを崩さないため、今回の配布版ではFalseのままにする。
SAFE_TRY_COLUMN_REFINE_ON_CLEAN_RECTANGLE = False

def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def depth_filter_like_viewer(raw_depth_frame: rs.depth_frame) -> rs.depth_frame:
    """
    rs-viewer に近いフィルタ処理（あなたの depth_filter と同等）
    """
    decim = rs.decimation_filter()
    decim.set_option(rs.option.filter_magnitude, 2.0)

    spat = rs.spatial_filter()
    spat.set_option(rs.option.filter_magnitude, 2.0)
    spat.set_option(rs.option.filter_smooth_alpha, 0.5)
    spat.set_option(rs.option.filter_smooth_delta, 20.0)

    hole_fill = rs.hole_filling_filter()

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    # viewer準拠: decimation は RGB 解像とズレる危険があるのでスキップ
    filtered = depth_to_disparity.process(raw_depth_frame)
    filtered = spat.process(filtered)
    filtered = disparity_to_depth.process(filtered)
    filtered = hole_fill.process(filtered)
    return filtered.as_depth_frame()


# def save_masked_and_cropped(
#     rgb_bgr: np.ndarray,
#     depth_u16: np.ndarray,
#     mask01: np.ndarray,
#     outdir: Path,
#     stem: str,
# ):
#     """
#     対象書籍のみの RGB/Depth を保存（背景0マスク）
#     """
#     outdir.mkdir(parents=True, exist_ok=True)

#     # --- マスク適用（背景=0） ---
#     rgb_masked = rgb_bgr.copy()
#     rgb_masked[mask01 == 0] = 0
#     depth_masked = depth_u16.copy()
#     depth_masked[mask01 == 0] = 0

#     cv2.imwrite(str(outdir / f"{stem}_rgb_masked.png"), rgb_masked)
#     np.save(outdir / f"{stem}_depth_masked.npy", depth_masked)

#     # 深度可視化（0 以外の範囲を 0–255 に正規化）
#     nonzero = depth_masked[depth_masked > 0]
#     if nonzero.size > 0:
#         zmin, zmax = int(nonzero.min()), int(nonzero.max())
#         zrange = max(1, zmax - zmin)
#         depth_vis = np.zeros_like(depth_masked, dtype=np.uint8)
#         depth_vis[depth_masked > 0] = (
#             (depth_masked[depth_masked > 0] - zmin) * 255 // zrange
#         ).astype(np.uint8)
#         cv2.imwrite(str(outdir / f"{stem}_depth_masked_vis.png"), depth_vis)

#     return depth_masked # 変更（追加）

def _make_depth_reference_mask_from_ocr_info(
    refine_info: dict | None,
    image_shape: tuple[int, int],
    selected_mask01: np.ndarray | None = None,
    *,
    min_intersection_px: int = 30,
):
    """
    Depth中央値を求めるための参照領域を作る．

    基本方針:
      - OCRで選択された文字領域 polygon を使う．
      - ただし，対象SAMマスクとほとんど重ならない場合は使わない．
      - 参照領域は「OCR polygon ∩ selected mask」を優先する．

    戻り値:
      reference_mask01, info
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    info = {
        "used": False,
        "reason": "not initialized",
        "source": "none",
        "min_intersection_px": int(min_intersection_px),
    }

    if not isinstance(refine_info, dict):
        info["reason"] = "refine_info is not dict"
        return None, info

    selected_ocr = refine_info.get("selected_ocr_polygon")
    if not isinstance(selected_ocr, dict):
        info["reason"] = "selected_ocr_polygon not found"
        return None, info

    poly = selected_ocr.get("poly")
    poly = _poly_from_any(poly)
    if poly is None:
        info["reason"] = "ocr polygon not found or invalid"
        return None, info

    poly_mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    cv2.fillConvexPoly(poly_mask, pts.astype(np.int32), 1)

    poly_area = int(poly_mask.sum())
    if poly_area <= 0:
        info["reason"] = "empty ocr polygon mask"
        return None, info

    if selected_mask01 is not None:
        selected_mask01 = (np.asarray(selected_mask01) > 0).astype(np.uint8)
        inter_mask = ((poly_mask > 0) & (selected_mask01 > 0)).astype(np.uint8)
        inter_area = int(inter_mask.sum())
    else:
        inter_mask = poly_mask
        inter_area = poly_area

    info.update({
        "ocr_text": selected_ocr.get("text"),
        "ocr_transform_mode": selected_ocr.get("transform_mode"),
        "ocr_polygon_area_px": int(poly_area),
        "ocr_polygon_intersection_with_selected_mask_px": int(inter_area),
        "ocr_polygon": pts.astype(float).tolist(),
    })

    if inter_area >= int(min_intersection_px):
        info.update({
            "used": True,
            "reason": "ok",
            "source": "selected_ocr_polygon_intersection_selected_mask",
            "reference_area_px": int(inter_area),
        })
        return inter_mask.astype(np.uint8), info

    # selected maskとの交差が少なすぎる場合は，OCR polygon単体も危険なので使わない．
    info.update({
        "used": False,
        "reason": f"ocr polygon intersection too small: {inter_area}px",
        "source": "none",
        "reference_area_px": int(inter_area),
    })
    return None, info


def save_masked_and_cropped(
    rgb_bgr: np.ndarray,
    depth_u16: np.ndarray,
    mask01: np.ndarray,
    outdir: Path,
    stem: str,
    z_tolerance_raw: int = 30,  # Z16値での許容幅（RealSense scale=0.001なら約±30mm）
    depth_reference_mask01: np.ndarray | None = None,
    depth_reference_name: str = "selected_mask",
    return_info: bool = False,
):
    """
    対象書籍のみの RGB/Depth を保存（背景0マスク ＋ 深度外れ値除去）．

    Depth外れ値除去の基準値 z_med は，通常は選択マスク全体ではなく，
    depth_reference_mask01 が与えられた場合はその領域内のDepth中央値から求める．
    今回の用途では，OCR文字領域内のDepth中央値を渡す想定．
    """
    outdir.mkdir(parents=True, exist_ok=True)

    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)

    # --- マスク適用（背景=0） ---
    rgb_masked = rgb_bgr.copy()
    rgb_masked[mask01 == 0] = 0

    depth_masked = depth_u16.copy()
    depth_masked[mask01 == 0] = 0  # まずマスク外を 0 にする

    info = {
        "used": False,
        "reason": "not initialized",
        "z_tolerance_raw": int(z_tolerance_raw),
        "depth_reference_name_requested": str(depth_reference_name),
        "depth_reference_source_used": None,
        "reference_valid_depth_count": 0,
        "selected_mask_valid_depth_count": int(np.count_nonzero(depth_masked > 0)),
        "z_median_raw": None,
        "z_min_keep_raw": None,
        "z_max_keep_raw": None,
        "removed_pixel_count": 0,
        "remaining_pixel_count": int(np.count_nonzero(depth_masked > 0)),
    }

    selected_nonzero = depth_masked[depth_masked > 0]

    # --- Depth中央値を求める参照領域を決める ---
    ref_values = np.asarray([], dtype=depth_u16.dtype)
    if depth_reference_mask01 is not None:
        ref_mask = (np.asarray(depth_reference_mask01) > 0)
        # OCR領域がマスク外まで広がる場合を避けるため，選択マスク内に制限する．
        ref_valid = ref_mask & (mask01 > 0) & (depth_u16 > 0)
        ref_values = depth_u16[ref_valid]
        info["reference_valid_depth_count"] = int(ref_values.size)
        info["depth_reference_source_used"] = str(depth_reference_name)

    # OCR領域にDepthが十分なければ，従来どおり選択マスク全体へフォールバックする．
    if ref_values.size < 10:
        ref_values = selected_nonzero
        info["depth_reference_source_used"] = "selected_mask_fallback"
        info["reference_valid_depth_count"] = int(ref_values.size)
        info["reference_fallback_reason"] = "reference mask has too few valid depth pixels"

    if ref_values.size > 0:
        z_med = int(np.median(ref_values))
        z_min_keep = int(z_med - int(z_tolerance_raw))
        z_max_keep = int(z_med + int(z_tolerance_raw))

        before_count = int(np.count_nonzero(depth_masked > 0))

        # 残したい領域（Trueが残す）
        keep = (depth_masked >= z_min_keep) & (depth_masked <= z_max_keep)

        # keep==False のところを 0 にする
        depth_masked[~keep] = 0

        after_count = int(np.count_nonzero(depth_masked > 0))
        info.update({
            "used": True,
            "reason": "ok",
            "z_median_raw": int(z_med),
            "z_min_keep_raw": int(z_min_keep),
            "z_max_keep_raw": int(z_max_keep),
            "removed_pixel_count": int(before_count - after_count),
            "remaining_pixel_count": int(after_count),
            "remaining_ratio": float(after_count / max(before_count, 1)),
        })

        try:
            info["reference_depth_raw_min"] = float(np.min(ref_values))
            info["reference_depth_raw_p05"] = float(np.percentile(ref_values, 5))
            info["reference_depth_raw_median"] = float(np.median(ref_values))
            info["reference_depth_raw_p95"] = float(np.percentile(ref_values, 95))
            info["reference_depth_raw_max"] = float(np.max(ref_values))
        except Exception:
            pass
    else:
        info["used"] = False
        info["reason"] = "no valid depth in selected mask"

    # --- ファイル保存 ---
    cv2.imwrite(str(outdir / f"{stem}_rgb_masked.png"), rgb_masked)
    np.save(outdir / f"{stem}_depth_masked.npy", depth_masked)

    # 深度可視化（0 以外の範囲を 0–255 に正規化） ※外れ値除去後の結果を可視化
    nonzero = depth_masked[depth_masked > 0]
    if nonzero.size > 0:
        zmin, zmax = int(nonzero.min()), int(nonzero.max())
        zrange = max(1, zmax - zmin)
        depth_vis = np.zeros_like(depth_masked, dtype=np.uint8)
        depth_vis[depth_masked > 0] = (
            (depth_masked[depth_masked > 0] - zmin) * 255 // zrange
        ).astype(np.uint8)
        cv2.imwrite(str(outdir / f"{stem}_depth_masked_vis.png"), depth_vis)

    if return_info:
        return depth_masked, info
    return depth_masked  # ここで外れ値除去済みの depth を返す

def capture_one_shot(pipe, cfg, align, shot_dir, *, stem: str, color_only: bool = False):
    profile = pipe.start(cfg)

    for _ in range(10):
        pipe.wait_for_frames()

    frames = pipe.wait_for_frames()

    if color_only:
        color_frame = frames.get_color_frame()
        color_np = np.asanyarray(color_frame.get_data())

        # 左右反転しないで保存
        cv2.imwrite(str(shot_dir / f"{stem}_rgb.png"), color_np)

        pipe.stop()
        return color_np, None, None, None

    # depthも使う場合（2回目）
    align_frames = align.process(frames)
    depth_frame = depth_filter_like_viewer(align_frames.get_depth_frame())
    color_frame = align_frames.get_color_frame()

    color_np = np.asanyarray(color_frame.get_data())
    depth_np_u16 = np.asanyarray(depth_frame.get_data())

    # 左右反転しないで保存
    cv2.imwrite(str(shot_dir / f"{stem}_rgb.png"), color_np)
    np.save(shot_dir / f"{stem}_depth.npy", depth_np_u16)

    dprof = rs.video_stream_profile(depth_frame.get_profile())
    intr = dprof.get_intrinsics()
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

    pipe.stop()
    return color_np, depth_np_u16, intr, depth_scale #後で調べる
def _resolve_ocr_subprocess_paths():
    ocr_dir = Path(__file__).resolve().parent / "OCR"
    ocr_py = ocr_dir / ".paadle_ocr" / "bin" / "python"
    ocr_script = ocr_dir / "paddle_ocr_test.py"
    if not ocr_py.exists():
        ocr_py = Path(sys.executable)
    else:
        try:
            probe = subprocess.run(
                [str(ocr_py), "-c", "import cv2"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
                check=False,
            )
            if probe.returncode != 0:
                ocr_py = Path(sys.executable)
        except Exception:
            ocr_py = Path(sys.executable)
    return str(ocr_py), str(ocr_script)


def run_ocr_subprocess(shot_dir: Path):
    OCR_PY, OCR_SCRIPT = _resolve_ocr_subprocess_paths()

    env = os.environ.copy()
    # PaddleOCR側が親プロセスのCUDA/cuBLASを拾って不安定になるのを避ける．
    env.pop("LD_LIBRARY_PATH", None)
    env["DISABLE_MODEL_SOURCE_CHECK"] = "True"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    subprocess.run([OCR_PY, OCR_SCRIPT, str(shot_dir)], check=True, env=env)
    print(f"✔ OCR done: {shot_dir / 'ocr_result.json'}")


def start_ocr_subprocess(shot_dir: Path):
    """
    OCR を非同期で開始して Popen を返す．
    後で communicate() / wait() して終了を待つ．
    """
    OCR_PY, OCR_SCRIPT = _resolve_ocr_subprocess_paths()

    env = os.environ.copy()
    # PaddleOCR側が親プロセスのCUDA/cuBLASを拾って不安定になるのを避ける．
    env.pop("LD_LIBRARY_PATH", None)
    env["DISABLE_MODEL_SOURCE_CHECK"] = "True"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    proc = subprocess.Popen(
        [OCR_PY, OCR_SCRIPT, str(shot_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc


def wait_ocr_subprocess(proc: subprocess.Popen, *, timeout: float | None = None) -> str:
    """
    OCR サブプロセスの終了待ち。
    正常終了しなければ例外を投げる。
    """
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate()
        raise RuntimeError(f"OCR subprocess timeout.\n{stdout}")

    if proc.returncode != 0:
        raise RuntimeError(f"OCR subprocess failed (code={proc.returncode}).\n{stdout}")

    return stdout or ""


def merge_ocr_and_masks(
    query: str,
    masks,
    shot_dir: Path,
    interactive: bool = True,
    threshold: int = 40,
):
    """
    OCR 結果(json) と SAM マスクを統合して、選択マスクを返す。
    """
    results = match_text_to_mask_main(query, masks, shot_dir, threshold=threshold)

    book_name = results[0]["name"] if results else None

    sel_idx = None
    if not interactive:
        sel_idx = 1

    if book_name:
        m = re.search(r"(\d+)$", book_name)  # 末尾の連続数字
        if m:
            sel_idx = int(m.group(1))

    if sel_idx is None:
        raise RuntimeError("マスクIDが選べませんでした（OCR結果なし or 対応付け失敗）")

    sel_mask = masks[sel_idx - 1]
    mask01 = (np.asarray(sel_mask) > 0).astype(np.uint8)

    return {
        "results": results,
        "book_name": book_name,
        "sel_idx": sel_idx,
        "sel_mask": sel_mask,
        "mask01": mask01,
    }



def _poly_from_any(value):
    """
    OCR bbox/polygon の表現ゆれを吸収して，4点以上の polygon を返す．
    返せない場合は None．
    """
    if value is None:
        return None

    # dict形式: {x1,y1,x2,y2} は axis-aligned box として4点化する．
    if isinstance(value, dict):
        keys = set(value.keys())
        if {"x1", "y1", "x2", "y2"}.issubset(keys):
            x1 = float(value["x1"])
            y1 = float(value["y1"])
            x2 = float(value["x2"])
            y2 = float(value["y2"])
            return np.asarray(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.float32,
            )

        for key in ("poly", "points", "bbox", "box", "dt_poly", "quadrilateral"):
            if key in value:
                poly = _poly_from_any(value[key])
                if poly is not None:
                    return poly
        return None

    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None

    # 例: [[x1,y1],...[x4,y4]]
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 4:
        return arr[:4].astype(np.float32)

    # 例: [[[x1,y1],...]] のように一段深い場合
    if arr.ndim == 3 and arr.shape[-1] == 2 and arr.shape[-2] >= 4:
        return arr.reshape(-1, arr.shape[-2], 2)[0, :4].astype(np.float32)

    # 例: [x1,y1,x2,y2,x3,y3,x4,y4]
    if arr.ndim == 1 and arr.size >= 8 and arr.size % 2 == 0:
        return arr.reshape(-1, 2)[:4].astype(np.float32)

    return None


def _polygon_center_and_axis(poly: np.ndarray):
    """
    4点 polygon から中心，長辺方向，短辺長，長辺長を返す．
    OCR polygon が傾いていれば，その傾きに沿った軸が返る．
    """
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 4:
        return None, None, None, None

    poly = poly[:4]
    center = poly.mean(axis=0).astype(np.float64)

    edges = []
    for i in range(4):
        v = poly[(i + 1) % 4] - poly[i]
        length = float(np.linalg.norm(v))
        if length > 1e-6:
            edges.append((length, v.astype(np.float64) / length))

    if not edges:
        return center, None, None, None

    edges_sorted = sorted(edges, key=lambda x: x[0], reverse=True)
    long_len, long_axis = edges_sorted[0]
    short_len = edges_sorted[-1][0]

    # 符号は任意なので，後段で使いやすいように正規化だけする．
    n = float(np.linalg.norm(long_axis))
    if n < 1e-9:
        return center, None, short_len, long_len

    return center, long_axis / n, short_len, long_len


# ===== OCR polygon 座標系補正 =====
# ocr_result.json 内の polygon が after_init_rgb.png と 90度ずれている場合に備え，
# 複数の座標変換を試し，選択SAMマスクや merged box と最も整合するものを自動選択する．
_OCR_POLY_TRANSFORM_MODES = (
    "identity",
    "ocr_cw_to_rgb",
    "ocr_ccw_to_rgb",
    "ocr_180_to_rgb",
)


def _transform_ocr_poly_to_rgb(poly: np.ndarray, image_shape: tuple[int, int], mode: str) -> np.ndarray:
    """
    OCR polygon を RGB画像座標系へ写す．

    mode:
      identity        : 変換なし
      ocr_cw_to_rgb   : OCR画像がRGB画像を90度時計回りに回転したものだった場合の逆変換
      ocr_ccw_to_rgb  : OCR画像がRGB画像を90度反時計回りに回転したものだった場合の逆変換
      ocr_180_to_rgb  : OCR画像がRGB画像を180度回転したものだった場合の逆変換

    注意:
      ここでは after_init_rgb.png の shape=(H,W) を基準にする．
    """
    poly = _poly_from_any(poly)
    if poly is None:
        return None

    H, W = int(image_shape[0]), int(image_shape[1])
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2).copy()
    x = p[:, 0].copy()
    y = p[:, 1].copy()

    if mode == "identity":
        out = np.stack([x, y], axis=1)
    elif mode == "ocr_cw_to_rgb":
        # RGB -> OCR が cv2.ROTATE_90_CLOCKWISE だったと仮定した逆変換．
        # OCR(xr,yr) -> RGB(x=yr, y=H-1-xr)
        out = np.stack([y, (H - 1) - x], axis=1)
    elif mode == "ocr_ccw_to_rgb":
        # RGB -> OCR が cv2.ROTATE_90_COUNTERCLOCKWISE だったと仮定した逆変換．
        # OCR(xr,yr) -> RGB(x=W-1-yr, y=xr)
        out = np.stack([(W - 1) - y, x], axis=1)
    elif mode == "ocr_180_to_rgb":
        out = np.stack([(W - 1) - x, (H - 1) - y], axis=1)
    else:
        raise ValueError(f"unknown OCR polygon transform mode: {mode}")

    return out.astype(np.float32)


def _poly_image_valid_score(poly: np.ndarray, image_shape: tuple[int, int]) -> float:
    """
    polygon が画像内にどの程度入っているかを 0〜1 で返す．
    """
    poly = _poly_from_any(poly)
    if poly is None:
        return 0.0
    H, W = int(image_shape[0]), int(image_shape[1])
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    inside = (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
    return float(np.mean(inside))


def _center_distance_score(center: np.ndarray | None, target_center: np.ndarray | None, image_shape: tuple[int, int]) -> float:
    """
    center が target_center に近いほど 1 に近いスコアを返す．
    target_center が無ければ 0．
    """
    if center is None or target_center is None:
        return 0.0
    H, W = int(image_shape[0]), int(image_shape[1])
    scale = max(80.0, 0.20 * float(max(H, W)))
    d = float(np.linalg.norm(np.asarray(center, dtype=np.float64) - np.asarray(target_center, dtype=np.float64)))
    return float(np.exp(-d / scale))


def _choose_axis_consistent_with_mask(
    ocr_axis: np.ndarray,
    mask_axis: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    """
    OCR polygon の長辺方向と，その直交方向のうち，SAMマスク主軸に近い方を選ぶ．

    理由:
      OCR polygon の長辺は「文字列方向」を表す場合があり，背表紙方向と90度ずれることがある．
      そのため，SAMマスクの主軸を参照して，長辺方向か短辺方向かを自動選択する．
    """
    axis = np.asarray(ocr_axis, dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return axis, "invalid"
    axis = axis / n

    if mask_axis is None:
        return axis, "ocr_long_axis_no_mask_reference"

    m = np.asarray(mask_axis, dtype=np.float64)
    mn = float(np.linalg.norm(m))
    if mn < 1e-9:
        return axis, "ocr_long_axis_no_mask_reference"
    m = m / mn

    perp = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    score_long = abs(float(np.dot(axis, m)))
    score_perp = abs(float(np.dot(perp, m)))

    if score_perp > score_long:
        return perp, "ocr_perpendicular_axis_aligned_to_mask_pca"
    return axis, "ocr_long_axis_aligned_to_mask_pca"


def _mask_centroid(mask01: np.ndarray):
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    return np.asarray([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def _mask_pca_axis(mask01: np.ndarray):
    """
    SAMマスクの2D画素から主軸を求める．
    OCR polygon が使えない場合のフォールバック．
    """
    ys, xs = np.where(mask01 > 0)
    if len(xs) < 20:
        return None

    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    pts -= pts.mean(axis=0, keepdims=True)

    try:
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
    except Exception:
        return None

    axis = vecs[:, int(np.argmax(vals))].astype(np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return None
    return axis / n


def _text_similarity(a: str, b: str) -> float:
    """
    OCR文字列と query の緩い類似度．
    タイトル全体でなく一部だけ読めた場合も拾えるようにする．
    """
    from difflib import SequenceMatcher

    def norm(s: str) -> str:
        s = str(s)
        s = re.sub(r"\s+", "", s)
        s = s.replace("［", "[").replace("］", "]")
        s = s.replace("（", "(").replace("）", ")")
        return s.lower()

    a = norm(a)
    b = norm(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return float(SequenceMatcher(None, a, b).ratio())


def _iter_ocr_items(obj):
    """
    ocr_result.json の形式差を吸収して，text, bbox/poly, score を持つ候補を列挙する．

    対応例:
      - PaddleOCR: [[poly, (text, score)], ...]
      - PaddleX系: {"rec_texts": [...], "rec_scores": [...], "dt_polys": [...]}
      - 独自形式: {"text": ..., "bbox": ...}
    """
    if obj is None:
        return

    if isinstance(obj, dict):
        # PaddleX / PP-OCR 系でよくある並列配列形式
        text_keys = ("rec_texts", "texts", "text")
        poly_keys = ("dt_polys", "dt_boxes", "boxes", "polys", "points", "bbox")
        score_keys = ("rec_scores", "scores", "confidences")

        rec_texts = None
        for k in text_keys:
            v = obj.get(k)
            if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], str)):
                rec_texts = v
                break

        rec_polys = None
        for k in poly_keys:
            v = obj.get(k)
            if isinstance(v, list) and len(v) == (len(rec_texts) if rec_texts is not None else len(v)):
                # text が単一文字列の bbox ではなく，並列配列っぽいものだけ採用
                if rec_texts is not None:
                    rec_polys = v
                    break

        rec_scores = None
        for k in score_keys:
            v = obj.get(k)
            if isinstance(v, list):
                rec_scores = v
                break

        if rec_texts is not None and rec_polys is not None:
            for i, text in enumerate(rec_texts):
                score = None
                if rec_scores is not None and i < len(rec_scores):
                    try:
                        score = float(rec_scores[i])
                    except Exception:
                        score = None
                yield {"text": str(text), "bbox": rec_polys[i], "score": score, "source": "parallel_arrays"}

        # 独自形式: text + bbox/poly が同じdict内にある
        text = obj.get("text") or obj.get("rec_text") or obj.get("label") or obj.get("transcription")
        bbox = None
        for key in ("poly", "points", "bbox", "box", "dt_poly", "quadrilateral"):
            if key in obj:
                bbox = obj[key]
                break
        score = obj.get("score") or obj.get("confidence") or obj.get("rec_score")
        if text is not None and bbox is not None:
            try:
                score = float(score) if score is not None else None
            except Exception:
                score = None
            yield {"text": str(text), "bbox": bbox, "score": score, "source": "dict_item"}

        # ネストを再帰的に探索
        for key in ("results", "ocr_results", "data", "items", "pages", "res", "output"):
            if key in obj:
                yield from _iter_ocr_items(obj[key])

    elif isinstance(obj, (list, tuple)):
        # PaddleOCR: [box, (text, score)]
        if len(obj) >= 2:
            maybe_poly = _poly_from_any(obj[0])
            maybe_rec = obj[1]
            if maybe_poly is not None and isinstance(maybe_rec, (list, tuple)) and len(maybe_rec) >= 1:
                if isinstance(maybe_rec[0], str):
                    score = None
                    if len(maybe_rec) >= 2:
                        try:
                            score = float(maybe_rec[1])
                        except Exception:
                            score = None
                    yield {"text": maybe_rec[0], "bbox": maybe_poly, "score": score, "source": "paddle_pair"}
                    return

        for v in obj:
            yield from _iter_ocr_items(v)


def _polygon_overlap_ratio_with_mask(poly: np.ndarray, mask01: np.ndarray) -> float:
    """
    OCR polygon が選択SAMマスクとどの程度重なっているかを返す．
    """
    poly = _poly_from_any(poly)
    if poly is None:
        return 0.0
    h, w = mask01.shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [pts], 1)
    area = int(canvas.sum())
    if area <= 0:
        return 0.0
    overlap = int(((canvas > 0) & (mask01 > 0)).sum())
    return float(overlap / area)


def _load_best_ocr_polygon_for_mask(
    shot_dir: Path,
    query: str,
    mask01: np.ndarray,
    target_center: np.ndarray | None = None,
    max_debug_candidates: int = 30,
):
    """
    ocr_result.json から，queryに近く，かつ選択SAMマスク・merged boxに近いOCR polygonを選ぶ．

    重要:
      ocr_result.json のpolygon座標が after_init_rgb.png と90度ずれている場合があるため，
      identity / 90度CW逆変換 / 90度CCW逆変換 / 180度変換をすべて試し，
      選択SAMマスクとの重なりと merged box 中心への近さから最も整合する座標系を選ぶ．
    """
    ocr_path = Path(shot_dir) / "ocr_result.json"
    if not ocr_path.exists():
        return None, {"reason": f"ocr_result.json not found: {ocr_path}", "candidates": []}

    try:
        obj = json.loads(ocr_path.read_text(encoding="utf-8"))
    except Exception as e:
        return None, {"reason": f"failed to read ocr_result.json: {e}", "candidates": []}

    h, w = mask01.shape[:2]
    image_shape = (h, w)
    candidates = []

    for item in _iter_ocr_items(obj):
        text = item.get("text", "")
        raw_poly = _poly_from_any(item.get("bbox"))
        if raw_poly is None:
            continue

        sim = _text_similarity(query, text)
        rec_score = item.get("score")
        try:
            rec_score = float(rec_score) if rec_score is not None else 0.0
        except Exception:
            rec_score = 0.0

        for transform_mode in _OCR_POLY_TRANSFORM_MODES:
            poly = _transform_ocr_poly_to_rgb(raw_poly, image_shape, transform_mode)
            if poly is None:
                continue

            valid_score = _poly_image_valid_score(poly, image_shape)
            if valid_score <= 0.0:
                continue

            center, raw_axis, short_len, long_len = _polygon_center_and_axis(poly)
            if center is None:
                continue

            cx, cy = float(center[0]), float(center[1])
            in_image_center = (0 <= int(round(cx)) < w) and (0 <= int(round(cy)) < h)
            inside_center = bool(mask01[int(round(cy)), int(round(cx))] > 0) if in_image_center else False
            overlap = _polygon_overlap_ratio_with_mask(poly, mask01)
            center_score = _center_distance_score(center, target_center, image_shape)

            # query一致を最重視．そのうえで，マスクとの重なり，merged box中心への近さ，画像内妥当性を使う．
            rank_score = (
                2.0 * sim
                + 0.55 * float(inside_center)
                + 0.55 * float(overlap)
                + 0.45 * float(center_score)
                + 0.20 * float(valid_score)
                + 0.05 * float(rec_score)
            )

            candidates.append({
                "rank_score": float(rank_score),
                "text": str(text),
                "sim": float(sim),
                "inside_center": bool(inside_center),
                "overlap": float(overlap),
                "center_distance_score": float(center_score),
                "image_valid_score": float(valid_score),
                "score": float(rec_score),
                "poly": poly.astype(np.float32),
                "raw_poly": raw_poly.astype(np.float32),
                "center": np.asarray(center, dtype=np.float64),
                "axis": None if raw_axis is None else np.asarray(raw_axis, dtype=np.float64),
                "short_len": None if short_len is None else float(short_len),
                "long_len": None if long_len is None else float(long_len),
                "source": item.get("source", "unknown"),
                "transform_mode": transform_mode,
            })

    if not candidates:
        return None, {"reason": "no OCR polygon candidates", "candidates": []}

    candidates.sort(key=lambda d: d["rank_score"], reverse=True)
    best = candidates[0]

    # queryにもマスクにもmerged中心にも合わないOCRは使わない．
    if best["sim"] < 0.15 and best["overlap"] < 0.20 and not best["inside_center"] and best["center_distance_score"] < 0.20:
        dbg = []
        for c in candidates[:max_debug_candidates]:
            dbg.append({
                "text": c["text"],
                "rank_score": c["rank_score"],
                "sim": c["sim"],
                "inside_center": c["inside_center"],
                "overlap": c["overlap"],
                "center_distance_score": c["center_distance_score"],
                "transform_mode": c["transform_mode"],
                "source": c["source"],
            })
        return None, {"reason": "no reliable OCR polygon", "candidates": dbg}

    dbg = []
    for c in candidates[:max_debug_candidates]:
        dbg.append({
            "text": c["text"],
            "rank_score": c["rank_score"],
            "sim": c["sim"],
            "inside_center": c["inside_center"],
            "overlap": c["overlap"],
            "center_distance_score": c["center_distance_score"],
            "image_valid_score": c["image_valid_score"],
            "score": c["score"],
            "source": c["source"],
            "transform_mode": c["transform_mode"],
            "center": [float(c["center"][0]), float(c["center"][1])],
            "axis": None if c["axis"] is None else [float(c["axis"][0]), float(c["axis"][1])],
            "short_len": c["short_len"],
            "long_len": c["long_len"],
        })

    return best, {"reason": "ok", "candidates": dbg}

def _extract_merged_box_center(merged: dict, mask01: np.ndarray):
    """
    OCR/SAM統合結果の axis-aligned box から中心を取得する．
    中心位置は merged の box が比較的安定していたため，まずこれを使う．
    """
    results = merged.get("results", []) if isinstance(merged, dict) else []
    result = results[0] if results else {}
    box = result.get("box") if isinstance(result, dict) else None

    if isinstance(box, dict) and {"x1", "y1", "x2", "y2"}.issubset(set(box.keys())):
        try:
            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])
            return np.asarray([0.5 * (x1 + x2), 0.5 * (y1 + y2)], dtype=np.float64), {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        except Exception:
            pass

    # result に4点polygonがあれば中心だけ使う
    if isinstance(result, dict):
        for key in ("poly", "points", "bbox"):
            poly = _poly_from_any(result.get(key))
            if poly is not None:
                center, _, _, _ = _polygon_center_and_axis(poly)
                if center is not None:
                    return np.asarray(center, dtype=np.float64), None

    c = _mask_centroid(mask01)
    return c, None


def _filter_small_components(mask01: np.ndarray, min_area_ratio: float = 0.003):
    """
    帯補正後に出る小さな孤立領域を除去する．
    削りすぎ防止のため，極小成分のみを対象にする．
    """
    mask01 = (mask01 > 0).astype(np.uint8)
    area = int(mask01.sum())
    if area <= 0:
        return mask01, {"enabled": False, "reason": "empty mask"}

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if num_labels <= 2:
        return mask01, {"enabled": True, "num_components": max(0, num_labels - 1), "removed_area": 0}

    min_area = max(10, int(round(area * float(min_area_ratio))))
    out = np.zeros_like(mask01, dtype=np.uint8)
    removed_area = 0
    kept_components = 0
    for label in range(1, num_labels):
        comp_area = int(stats[label, cv2.CC_STAT_AREA])
        if comp_area >= min_area:
            out[labels == label] = 1
            kept_components += 1
        else:
            removed_area += comp_area

    if int(out.sum()) <= 0:
        return mask01, {
            "enabled": True,
            "num_components": num_labels - 1,
            "removed_area": 0,
            "reason": "all components would be removed; reverted",
        }

    return out, {
        "enabled": True,
        "num_components": num_labels - 1,
        "kept_components": kept_components,
        "min_area": int(min_area),
        "removed_area": int(removed_area),
    }



def _suppress_side_protrusions_axis_profile(
    mask01: np.ndarray,
    axis: np.ndarray,
    center: np.ndarray,
    bin_size_px: float = 8.0,
    slice_width_percentiles: tuple[float, float] = (10.0, 90.0),
    typical_width_percentile: float = 55.0,
    profile_width_margin: float = 1.20,
    min_profile_half_width_px: float = 18.0,
    min_bin_pixels: int = 12,
    max_profile_shrink_ratio: float = 0.30,
):
    """
    OCR軸帯だけでは残る「側面の張り出し」を，背表紙方向の断面幅から控えめに削る．

    考え方:
      - 背表紙方向を u，それに直交する方向を v とする．
      - u方向に細かくスライスし，各スライス内の v 方向の幅を求める．
      - 背表紙の大部分では幅が安定する一方，側面が混入したスライスだけ幅が大きくなる．
      - そこで，全スライスの典型幅を基準として，各スライスの中心線周辺だけを残す．

    削りすぎ防止:
      - この処理だけで max_profile_shrink_ratio を超えて面積が減る場合は元に戻す．
    """
    mask01 = (mask01 > 0).astype(np.uint8)
    area_before = int(mask01.sum())
    info = {
        "enabled": True,
        "area_before": area_before,
        "bin_size_px": float(bin_size_px),
        "slice_width_percentiles": [float(slice_width_percentiles[0]), float(slice_width_percentiles[1])],
        "typical_width_percentile": float(typical_width_percentile),
        "profile_width_margin": float(profile_width_margin),
        "min_profile_half_width_px": float(min_profile_half_width_px),
        "min_bin_pixels": int(min_bin_pixels),
        "max_profile_shrink_ratio": float(max_profile_shrink_ratio),
    }

    if area_before <= 0:
        info.update({"used": False, "reason": "empty mask"})
        return mask01, info

    axis = np.asarray(axis, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        info.update({"used": False, "reason": "invalid axis"})
        return mask01, info
    axis = axis / n
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)

    ys, xs = np.where(mask01 > 0)
    if len(xs) < 50:
        info.update({"used": False, "reason": "too few pixels"})
        return mask01, info

    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    rel = pts - center
    u = rel @ axis
    v = rel @ normal

    u_min = float(np.min(u))
    bin_size_px = max(2.0, float(bin_size_px))
    bins = np.floor((u - u_min) / bin_size_px).astype(np.int32)
    num_bins = int(bins.max()) + 1

    valid_bin_ids = []
    bin_centers_v = []
    bin_widths_v = []

    low_p, high_p = float(slice_width_percentiles[0]), float(slice_width_percentiles[1])
    for b in range(num_bins):
        vv = v[bins == b]
        if vv.size < int(min_bin_pixels):
            continue
        q_low, q_high = np.percentile(vv, [low_p, high_p])
        width = float(q_high - q_low)
        if width <= 1.0:
            continue
        valid_bin_ids.append(b)
        # 中心は中央値を使う．側面の張り出しがあっても平均より引っ張られにくい．
        bin_centers_v.append(float(np.median(vv)))
        bin_widths_v.append(width)

    if len(valid_bin_ids) < 5:
        info.update({"used": False, "reason": "too few valid slices", "num_valid_slices": int(len(valid_bin_ids))})
        return mask01, info

    valid_bin_ids = np.asarray(valid_bin_ids, dtype=np.float64)
    bin_centers_v = np.asarray(bin_centers_v, dtype=np.float64)
    bin_widths_v = np.asarray(bin_widths_v, dtype=np.float64)

    typical_width = float(np.percentile(bin_widths_v, float(typical_width_percentile)))
    if typical_width <= 1.0:
        info.update({"used": False, "reason": "invalid typical width"})
        return mask01, info

    half_clip = max(float(min_profile_half_width_px), 0.5 * typical_width * float(profile_width_margin))

    # 各画素のuスライスに対応する中心線位置を線形補間で求める．
    # 端では最近傍の中心を使う．
    center_v_for_pixel = np.interp(
        bins.astype(np.float64),
        valid_bin_ids,
        bin_centers_v,
        left=float(bin_centers_v[0]),
        right=float(bin_centers_v[-1]),
    )

    keep = np.abs(v - center_v_for_pixel) <= half_clip
    out = np.zeros_like(mask01, dtype=np.uint8)
    out[ys[keep], xs[keep]] = 1

    # 極小成分は最後に削る．ただしここでも削りすぎない．
    out, comp_info = _filter_small_components(out, min_area_ratio=0.002)

    area_after = int(out.sum())
    remain_ratio = float(area_after / max(area_before, 1))
    shrink_ratio = float(1.0 - remain_ratio)

    info.update({
        "used": True,
        "reason": "ok",
        "area_after": area_after,
        "remain_ratio": remain_ratio,
        "shrink_ratio": shrink_ratio,
        "num_valid_slices": int(len(valid_bin_ids)),
        "typical_width_px": typical_width,
        "half_clip_px": float(half_clip),
        "slice_width_min_px": float(np.min(bin_widths_v)),
        "slice_width_median_px": float(np.median(bin_widths_v)),
        "slice_width_max_px": float(np.max(bin_widths_v)),
        "component_filter": comp_info,
    })

    if area_after <= 0:
        info.update({"used": False, "reason": "all removed; reverted", "area_after": area_before, "remain_ratio": 1.0, "shrink_ratio": 0.0})
        return mask01, info

    if shrink_ratio > float(max_profile_shrink_ratio):
        info.update({
            "used": False,
            "reason": f"profile suppression removed too much: shrink_ratio={shrink_ratio:.3f}",
            "area_after": area_before,
            "remain_ratio": 1.0,
            "shrink_ratio": 0.0,
        })
        return mask01, info

    return out, info

def refine_mask_by_ocr_axis_band(
    mask01: np.ndarray,
    merged: dict,
    image_shape: tuple[int, int],
    shot_dir: str | Path | None = None,
    query: str | None = None,
    mask_width_ratio: float = 0.90,
    min_keep_ratio: float = 0.65,
    width_percentiles: tuple[float, float] = (5.0, 95.0),
    remove_small_components: bool = True,
    small_component_area_ratio: float = 0.003,
    use_ocr_short_width: bool = True,
    ocr_short_to_half_width_scale: float = 1.35,
    min_ocr_half_width_px: float = 28.0,
    suppress_side_protrusions: bool = False,
    profile_bin_size_px: float = 8.0,
    profile_width_margin: float = 1.20,
    min_profile_half_width_px: float = 18.0,
    max_profile_shrink_ratio: float = 0.30,
    return_info: bool = False,
):
    """
    SAM2マスクから，OCR中心線に沿う帯領域を控えめに残す．

    今回の反省を反映した設計:
      1. merged box は中心位置として使う．
      2. ocr_result.json の polygon は，RGB画像と90度ずれていることがあるため，複数の回転補正を自動試行する．
      3. OCR polygon の長辺方向が背表紙方向と90度ずれる場合があるため，SAMマスクPCA主軸に近い方向を選ぶ．
      4. polygonが使えない場合は SAMマスクPCA主軸を使う．
      5. forced_angle は最後のフォールバックとしてのみ使う．
      6. 帯幅は，OCR文字領域の短辺幅を優先して決める．文字領域の長さには依存しない．
      7. OCR短辺幅が使えない場合のみ，SAMマスクの幅から決める．
      8. 局所スライスごとの側面抑制は斜め削りの原因になりやすいため，デフォルトでは使わない．
      9. 残存率が低すぎる場合は元マスクへ戻す．
    """
    h, w = image_shape
    mask01 = (mask01 > 0).astype(np.uint8)
    mask_area = int(mask01.sum())

    info = {
        "used": False,
        "reason": "",
        "mask_area_before": mask_area,
        "mask_width_ratio": float(mask_width_ratio),
        "min_keep_ratio": float(min_keep_ratio),
        "width_percentiles": [float(width_percentiles[0]), float(width_percentiles[1])],
        "use_ocr_short_width": bool(use_ocr_short_width),
        "ocr_short_to_half_width_scale": float(ocr_short_to_half_width_scale),
        "min_ocr_half_width_px": float(min_ocr_half_width_px),
        "suppress_side_protrusions": bool(suppress_side_protrusions),
        "profile_bin_size_px": float(profile_bin_size_px),
        "profile_width_margin": float(profile_width_margin),
        "min_profile_half_width_px": float(min_profile_half_width_px),
        "max_profile_shrink_ratio": float(max_profile_shrink_ratio),
    }

    if mask_area <= 0:
        info["reason"] = "empty mask"
        return (mask01, info) if return_info else mask01

    results = merged.get("results", []) if isinstance(merged, dict) else []
    result = results[0] if results else {}

    center, merged_box = _extract_merged_box_center(merged, mask01)
    if center is None:
        info["reason"] = "failed to determine center"
        return (mask01, info) if return_info else mask01

    axis = None
    axis_source = None
    ocr_best = None
    ocr_debug = None

    # SAMマスク主軸は，OCR polygonの長辺/短辺のどちらを使うか判定するためにも使う．
    mask_axis = _mask_pca_axis(mask01)

    # ===== 1) ocr_result.json の4点polygonを最優先 =====
    if shot_dir is not None and query is not None:
        ocr_best, ocr_debug = _load_best_ocr_polygon_for_mask(
            shot_dir=Path(shot_dir),
            query=str(query),
            mask01=mask01,
            target_center=center,
        )
        info["ocr_polygon_search"] = ocr_debug
        if ocr_best is not None and ocr_best.get("axis") is not None:
            raw_ocr_axis = np.asarray(ocr_best["axis"], dtype=np.float64)
            axis, axis_kind = _choose_axis_consistent_with_mask(raw_ocr_axis, mask_axis)
            axis_source = f"ocr_result_polygon_{axis_kind}"
            # 中心は merged bbox を優先するが，mergedが使えないときのみOCR中心を使う．
            if merged_box is None:
                center = np.asarray(ocr_best["center"], dtype=np.float64)

    # ===== 2) merged resultにpolygonが含まれていれば使う =====
    if axis is None and isinstance(result, dict):
        for key in ("poly", "points", "bbox"):
            poly = _poly_from_any(result.get(key))
            if poly is not None:
                c_poly, axis_poly, short_len, long_len = _polygon_center_and_axis(poly)
                if axis_poly is not None:
                    axis, axis_kind = _choose_axis_consistent_with_mask(axis_poly, mask_axis)
                    axis = np.asarray(axis, dtype=np.float64)
                    axis_source = f"merged_{key}_polygon_{axis_kind}"
                    break

    # ===== 3) SAMマスクのPCA主軸 =====
    if axis is None and mask_axis is not None:
        axis = np.asarray(mask_axis, dtype=np.float64)
        axis_source = "mask_pca"

    # ===== 4) 最後のフォールバックとしてだけ forced_angle / bbox縦横比 =====
    if axis is None:
        forced_angle = result.get("forced_angle", None) if isinstance(result, dict) else None
        if forced_angle is not None:
            angle_rad = np.deg2rad(float(forced_angle))
            axis_source = "forced_angle_fallback"
        elif merged_box is not None:
            bw = abs(float(merged_box["x2"]) - float(merged_box["x1"]))
            bh = abs(float(merged_box["y2"]) - float(merged_box["y1"]))
            angle_rad = np.pi / 2.0 if bh >= bw else 0.0
            axis_source = "bbox_aspect_fallback"
        else:
            info["reason"] = "failed to determine axis"
            return (mask01, info) if return_info else mask01
        axis = np.asarray([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float64)

    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        info["reason"] = "invalid axis"
        return (mask01, info) if return_info else mask01
    axis = axis / axis_norm

    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)

    ys, xs = np.where(mask01 > 0)
    if len(xs) < 20:
        info["reason"] = "too few mask pixels"
        return (mask01, info) if return_info else mask01

    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    normal_coords = (pts - center) @ normal

    p_low, p_high = float(width_percentiles[0]), float(width_percentiles[1])
    q_low, q_high = np.percentile(normal_coords, [p_low, p_high])
    mask_width_px = float(q_high - q_low)
    if mask_width_px <= 1.0:
        info["reason"] = "invalid mask width"
        return (mask01, info) if return_info else mask01

    # ===== 帯幅の決定 =====
    # これまでの問題点:
    #   - SAMマスク幅だけで帯幅を決めると，側面部が混入した分だけ帯が広がり，側面が残る．
    #   - OCR文字領域の「長さ」は本全体を覆わないことがあるため信用しない．
    # 改善方針:
    #   - OCR文字領域の短辺幅は比較的安定しているため，これを背表紙幅の手掛かりにする．
    #   - OCR短辺幅から決めた帯を，本の主軸方向へ無限に延長して使う．
    #   - これにより，OCR文字領域の長さが短くても，本全体の上下方向は削らない．
    half_width_from_mask_px = 0.5 * mask_width_px * float(mask_width_ratio)

    ocr_short_len_px = None
    if isinstance(ocr_best, dict) and ocr_best.get("short_len") is not None:
        try:
            ocr_short_len_px = float(ocr_best.get("short_len"))
        except Exception:
            ocr_short_len_px = None

    half_width_from_ocr_px = None
    if bool(use_ocr_short_width) and ocr_short_len_px is not None and ocr_short_len_px > 1.0:
        half_width_from_ocr_px = max(
            float(min_ocr_half_width_px),
            float(ocr_short_len_px) * float(ocr_short_to_half_width_scale),
        )
        # OCR短辺幅は側面混入の影響を受けにくいので，SAM幅由来の値より小さい場合は優先する．
        # ただし，OCR幅が大きすぎる場合はSAM幅由来の安全上限で抑える．
        half_width_px = min(float(half_width_from_mask_px), float(half_width_from_ocr_px))
        band_width_source = "ocr_short_width_limited_by_mask_width"
    else:
        half_width_px = float(half_width_from_mask_px)
        band_width_source = "mask_width"

    keep_pts = np.abs(normal_coords) <= half_width_px

    refined = np.zeros_like(mask01, dtype=np.uint8)
    refined[ys[keep_pts], xs[keep_pts]] = 1

    component_info = None
    if remove_small_components:
        refined, component_info = _filter_small_components(
            refined,
            min_area_ratio=float(small_component_area_ratio),
        )

    side_suppression_info = None
    if suppress_side_protrusions:
        refined, side_suppression_info = _suppress_side_protrusions_axis_profile(
            refined,
            axis=axis,
            center=center,
            bin_size_px=float(profile_bin_size_px),
            profile_width_margin=float(profile_width_margin),
            min_profile_half_width_px=float(min_profile_half_width_px),
            max_profile_shrink_ratio=float(max_profile_shrink_ratio),
        )

    refined_area = int(refined.sum())
    remain_ratio = float(refined_area / max(mask_area, 1))

    info.update({
        "used": True,
        "reason": "ok",
        "axis_source": axis_source,
        "merged_box": merged_box,
        "center_source": "merged_box" if merged_box is not None else "fallback_center",
        "ocr_center": [float(center[0]), float(center[1])],
        "axis": [float(axis[0]), float(axis[1])],
        "normal": [float(normal[0]), float(normal[1])],
        "angle_deg": float(np.degrees(np.arctan2(axis[1], axis[0]))),
        "mask_pca_axis": None if mask_axis is None else [float(mask_axis[0]), float(mask_axis[1])],
        "mask_width_px": float(mask_width_px),
        "half_width_from_mask_px": float(half_width_from_mask_px),
        "ocr_short_len_px": None if ocr_short_len_px is None else float(ocr_short_len_px),
        "half_width_from_ocr_px": None if half_width_from_ocr_px is None else float(half_width_from_ocr_px),
        "band_width_source": band_width_source,
        "half_width_px": float(half_width_px),
        "mask_area_after": refined_area,
        "remain_ratio": remain_ratio,
        "component_filter": component_info,
        "side_suppression": side_suppression_info,
    })

    if ocr_best is not None:
        info["selected_ocr_polygon"] = {
            "text": ocr_best.get("text"),
            "rank_score": float(ocr_best.get("rank_score", 0.0)),
            "sim": float(ocr_best.get("sim", 0.0)),
            "inside_center": bool(ocr_best.get("inside_center", False)),
            "overlap": float(ocr_best.get("overlap", 0.0)),
            "center_distance_score": float(ocr_best.get("center_distance_score", 0.0)),
            "image_valid_score": float(ocr_best.get("image_valid_score", 0.0)),
            "source": ocr_best.get("source", "unknown"),
            "transform_mode": ocr_best.get("transform_mode", "unknown"),
            "center": [float(ocr_best["center"][0]), float(ocr_best["center"][1])],
            "axis_raw_from_polygon": None if ocr_best.get("axis") is None else [float(ocr_best["axis"][0]), float(ocr_best["axis"][1])],
            "poly": np.asarray(ocr_best["poly"], dtype=np.float32).tolist(),
            "raw_poly": np.asarray(ocr_best["raw_poly"], dtype=np.float32).tolist() if ocr_best.get("raw_poly") is not None else None,
        }

    # 削りすぎたら補正を破棄
    if remain_ratio < float(min_keep_ratio):
        info["used"] = False
        info["reason"] = f"too much removed: remain_ratio={remain_ratio:.3f}"
        info["mask_area_after"] = mask_area
        info["remain_ratio"] = 1.0
        return (mask01, info) if return_info else mask01

    return (refined, info) if return_info else refined


def _longest_occupied_s_segment(
    s_values: np.ndarray,
    s_origin: float,
    s_bin_size_px: float,
    gap_allow_bins: int,
):
    """
    1本のt列に含まれる点のs座標から，gapを少し許容した最長連続区間を返す．
    戻り値は dict または None．
    """
    s_values = np.asarray(s_values, dtype=np.float64).reshape(-1)
    if s_values.size <= 0:
        return None

    s_bin_size_px = max(float(s_bin_size_px), 1.0)
    gap_allow_bins = max(int(gap_allow_bins), 0)

    bins = np.floor((s_values - float(s_origin)) / s_bin_size_px).astype(np.int32)
    if bins.size <= 0:
        return None

    uniq = np.unique(bins)
    if uniq.size <= 0:
        return None

    best_start = int(uniq[0])
    best_end = int(uniq[0])
    cur_start = int(uniq[0])
    cur_end = int(uniq[0])

    for b in uniq[1:]:
        b = int(b)
        # 例: gap_allow_bins=2 なら，2binまでの穴は同じ連続区間とみなす．
        if b - cur_end <= gap_allow_bins + 1:
            cur_end = b
        else:
            if (cur_end - cur_start) > (best_end - best_start):
                best_start, best_end = cur_start, cur_end
            cur_start = cur_end = b

    if (cur_end - cur_start) > (best_end - best_start):
        best_start, best_end = cur_start, cur_end

    s_min = float(s_origin + best_start * s_bin_size_px)
    s_max = float(s_origin + (best_end + 1) * s_bin_size_px)
    length_px = float(max(0.0, s_max - s_min))

    # 最長区間内に実際にある点数も数える．
    in_run = (bins >= best_start) & (bins <= best_end)

    return {
        "start_bin": int(best_start),
        "end_bin": int(best_end),
        "s_min": s_min,
        "s_max": s_max,
        "length_px": length_px,
        "point_count_in_run": int(np.count_nonzero(in_run)),
        "occupied_bin_count": int(uniq.size),
    }


def _find_consecutive_groups(indices: np.ndarray):
    """昇順整数配列を連続グループに分ける．"""
    indices = np.asarray(indices, dtype=np.int32).reshape(-1)
    if indices.size == 0:
        return []
    indices = np.unique(indices)
    groups = []
    start = int(indices[0])
    prev = int(indices[0])
    for v in indices[1:]:
        v = int(v)
        if v == prev + 1:
            prev = v
        else:
            groups.append((start, prev))
            start = prev = v
    groups.append((start, prev))
    return groups



def _get_seed_center_from_refine_info(refine_info: dict | None, fallback_center: np.ndarray):
    """
    列長フィルタで使うseed中心を返す．

    優先順位:
      1. selected_ocr_polygon.center
      2. ocr_center
      3. fallback_center

    selected OCR polygon の中心と merged box 中心がずれるケースで，
    対象背表紙の幅方向位置を取り違えないようにする．
    """
    rinfo = refine_info or {}

    selected = rinfo.get("selected_ocr_polygon")
    if isinstance(selected, dict) and selected.get("center") is not None:
        try:
            return np.asarray(selected["center"], dtype=np.float64).reshape(2), "selected_ocr_polygon_center"
        except Exception:
            pass

    if rinfo.get("ocr_center") is not None:
        try:
            return np.asarray(rinfo["ocr_center"], dtype=np.float64).reshape(2), "ocr_center"
        except Exception:
            pass

    return np.asarray(fallback_center, dtype=np.float64).reshape(2), "fallback_center"


def _column_length_record_for_s_values(
    s_values: np.ndarray,
    s_origin: float,
    s_bin_size_px: float,
    gap_allow_bins: int,
    span_percentiles: tuple[float, float] = (3.0, 97.0),
    min_occupancy_density: float = 0.18,
):
    """
    1本のt列に含まれるs座標から，列長情報を返す．

    従来は「最長連続区間」だけを列長としていたため，Depth欠損で途中に穴が空くと，
    背表紙方向に長い列でも短い列として扱われることがあった．
    ここでは最長連続区間に加え，密度付きのロバストspan長も使う．
    """
    s_values = np.asarray(s_values, dtype=np.float64).reshape(-1)
    if s_values.size <= 0:
        return None

    s_bin_size_px = max(float(s_bin_size_px), 1.0)
    gap_allow_bins = max(int(gap_allow_bins), 0)

    run = _longest_occupied_s_segment(
        s_values,
        s_origin=float(s_origin),
        s_bin_size_px=s_bin_size_px,
        gap_allow_bins=gap_allow_bins,
    )
    if run is None:
        return None

    p0, p1 = float(span_percentiles[0]), float(span_percentiles[1])
    p0 = float(np.clip(p0, 0.0, 49.0))
    p1 = float(np.clip(p1, 51.0, 100.0))

    try:
        span_s_min, span_s_max = np.percentile(s_values, [p0, p1])
        span_length_px = float(max(0.0, span_s_max - span_s_min))
    except Exception:
        span_s_min, span_s_max, span_length_px = run["s_min"], run["s_max"], run["length_px"]

    span_bin_count = max(1, int(round(span_length_px / s_bin_size_px)) + 1)
    occupied_bin_count = int(run.get("occupied_bin_count", 0))
    occupancy_density = float(occupied_bin_count / max(span_bin_count, 1))

    run_length_px = float(run["length_px"])
    density_weight = float(np.clip(occupancy_density / max(float(min_occupancy_density), 1e-6), 0.0, 1.0))
    span_supported_length = float(span_length_px * density_weight)

    effective_length_px = float(max(run_length_px, span_supported_length))

    if span_supported_length >= run_length_px:
        eff_s_min = float(span_s_min)
        eff_s_max = float(span_s_max)
        effective_source = "span_supported"
    else:
        eff_s_min = float(run["s_min"])
        eff_s_max = float(run["s_max"])
        effective_source = "longest_run"

    out = dict(run)
    out.update({
        "run_length_px": run_length_px,
        "span_s_min": float(span_s_min),
        "span_s_max": float(span_s_max),
        "span_length_px": float(span_length_px),
        "span_bin_count": int(span_bin_count),
        "occupancy_density": float(occupancy_density),
        "density_weight": float(density_weight),
        "effective_length_px": float(effective_length_px),
        "effective_s_min": float(eff_s_min),
        "effective_s_max": float(eff_s_max),
        "effective_source": effective_source,
    })
    return out



def _longest_occupied_metric_segment(
    values: np.ndarray,
    origin: float,
    bin_size: float,
    gap_allow_bins: int,
):
    """
    m単位など任意実数座標用の連続run長計算。

    画像[pixel]用のhelperはbin sizeを最低1.0に丸めるため、
    a_bin_size_m=0.003 のようなm単位処理には使えない。
    ここではbin_sizeを1.0に丸めず、最長のoccupied runを返す。
    """
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if vals.size <= 0:
        return None

    bin_size = max(float(bin_size), 1e-9)
    gap_allow_bins = max(int(gap_allow_bins), 0)

    bins = np.floor((vals - float(origin)) / bin_size).astype(np.int64)
    if bins.size <= 0:
        return None
    uniq = np.unique(bins)
    if uniq.size <= 0:
        return None

    best_start = int(uniq[0])
    best_end = int(uniq[0])
    cur_start = int(uniq[0])
    cur_end = int(uniq[0])

    for bb in uniq[1:]:
        b = int(bb)
        if b - cur_end <= gap_allow_bins + 1:
            cur_end = b
        else:
            if (cur_end - cur_start) > (best_end - best_start):
                best_start, best_end = cur_start, cur_end
            cur_start = cur_end = b

    if (cur_end - cur_start) > (best_end - best_start):
        best_start, best_end = cur_start, cur_end

    v_min = float(origin + best_start * bin_size)
    v_max = float(origin + (best_end + 1) * bin_size)
    in_run = (bins >= best_start) & (bins <= best_end)
    span_min = float(np.min(vals))
    span_max = float(np.max(vals))
    span_len = float(max(0.0, span_max - span_min))
    span_bin_count = max(1, int(np.floor((span_max - span_min) / bin_size)) + 1)
    occupancy_density = float(uniq.size / max(span_bin_count, 1))

    return {
        "start_bin": int(best_start),
        "end_bin": int(best_end),
        "value_min": v_min,
        "value_max": v_max,
        "length": float(max(0.0, v_max - v_min)),
        "point_count_in_run": int(np.count_nonzero(in_run)),
        "occupied_bin_count": int(uniq.size),
        "span_min": span_min,
        "span_max": span_max,
        "span_length": span_len,
        "span_bin_count": int(span_bin_count),
        "occupancy_density": float(occupancy_density),
    }

def refine_mask_by_spine_column_length_after_depth(
    mask01: np.ndarray,
    depth_masked: np.ndarray,
    refine_info: dict | None,
    image_shape: tuple[int, int],
    *,
    t_bin_size_px: float = 4.0,
    s_bin_size_px: float = 5.0,
    s_gap_allow_px: float = 24.0,
    length_reference_percentile: float = 85.0,
    min_length_ratio: float = 0.95,
    relaxed_edge_length_ratio: float = 0.95,
    min_points_per_t_bin: int = 6,
    min_selected_t_bins: int = 2,
    expand_selected_t_bins: int = 0,
    s_margin_px: float = 0.0,
    min_valid_keep_ratio: float = 0.35,
    length_reference_mode: str = "robust_percentile",
    seed_local_window_bins: int = 14,
    bridge_gap_bins: int = 2,
    max_seed_group_distance_bins: int = 4,
    use_global_s_range: bool = False,
    span_percentiles: tuple[float, float] = (2.0, 98.0),
    min_occupancy_density: float = 0.0,
    column_length_mode: str = "continuous_run",
    min_run_point_ratio: float = 0.18,
    use_seed_width_guard: bool = True,
    seed_width_guard_scale: float = 1.10,
    seed_width_mask_percentile: float = 2.0,
    seed_width_low_percentile_min_median_ratio: float = 0.50,
    seed_width_fallback_mode: str = "median",
    seed_guard_center_mode: str = "mask_centroid",
    seed_width_profile_s_percentiles: tuple[float, float] = (10.0, 90.0),
    seed_width_profile_t_percentiles: tuple[float, float] = (2.0, 98.0),
    min_seed_guard_width_px: float = 40.0,
    max_seed_guard_width_px: float = 140.0,
    one_sided_removal: bool = True,
    one_sided_min_candidate_points: int = 30,
    one_sided_score_margin_ratio: float = 1.05,
    protect_ocr_extended_band: bool = True,
    ocr_extended_protection_margin_px: float = 0.0,
    ocr_extended_max_rescue_ratio: float = 0.55,
    post_profile_side_suppression: bool = True,
    post_profile_width_margin: float = 1.12,
    post_profile_max_shrink_ratio: float = 0.22,
    return_info: bool = False,
):
    """
    Depth中央値±3cm補正後の有効点に対し，OCR文字領域の軸を使って側面部を除去する．

    旧版との差分:
      - OCR帯補正で直接maskを削ると，背表紙本体まで削るケースがあるため，
        本関数ではOCR中心・OCR軸をseedとして用いる．
      - global max * 0.9 は外れ値に引っ張られるため，85 percentile長を基準にする．
      - shape_infoの width_median_px があれば，OCR seed 周辺の典型幅を背表紙コアとして推定する．
      - 側面部は片側だけに見えるという前提を使い，削る側を1方向に限定する．
        つまり，削除候補が多い側だけを削り，背表紙を挟んだ反対側の点群は保護する．
      - さらに，選択OCR文字領域を背表紙方向axisへ延長した帯は信頼領域として保護する．
        保護帯のnormal方向幅は，選択OCR polygonをaxis直交方向へ投影した幅そのものを使う．
      - 採用したt列の中では，s方向の上下を再カットしない．
    """
    h, w = image_shape
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    depth_masked = np.asarray(depth_masked)
    valid = (mask01 > 0) & (depth_masked > 0)
    valid_count = int(np.count_nonzero(valid))

    info = {
        "used": False,
        "reason": "",
        "algorithm": "safe_seed_width_guard_after_depth_prefilter",
        "valid_count_before": valid_count,
        "t_bin_size_px": float(t_bin_size_px),
        "s_bin_size_px": float(s_bin_size_px),
        "s_gap_allow_px": float(s_gap_allow_px),
        "length_reference_mode": str(length_reference_mode),
        "length_reference_percentile": float(length_reference_percentile),
        "span_percentiles": [float(span_percentiles[0]), float(span_percentiles[1])],
        "column_length_mode": str(column_length_mode),
        "min_run_point_ratio": float(min_run_point_ratio),
        "min_length_ratio_requested": float(min_length_ratio),
        "relaxed_edge_length_ratio_requested": float(relaxed_edge_length_ratio),
        "min_points_per_t_bin": int(min_points_per_t_bin),
        "min_selected_t_bins": int(min_selected_t_bins),
        "expand_selected_t_bins": int(expand_selected_t_bins),
        "min_valid_keep_ratio": float(min_valid_keep_ratio),
        "seed_local_window_bins": int(seed_local_window_bins),
        "bridge_gap_bins": int(bridge_gap_bins),
        "max_seed_group_distance_bins": int(max_seed_group_distance_bins),
        "use_seed_width_guard": bool(use_seed_width_guard),
        "seed_width_guard_scale": float(seed_width_guard_scale),
        "seed_width_mask_percentile": float(seed_width_mask_percentile),
        "seed_width_low_percentile_min_median_ratio": float(seed_width_low_percentile_min_median_ratio),
        "seed_width_fallback_mode": str(seed_width_fallback_mode),
        "seed_guard_center_mode": str(seed_guard_center_mode),
        "seed_width_profile_s_percentiles": [float(seed_width_profile_s_percentiles[0]), float(seed_width_profile_s_percentiles[1])],
        "seed_width_profile_t_percentiles": [float(seed_width_profile_t_percentiles[0]), float(seed_width_profile_t_percentiles[1])],
        "min_seed_guard_width_px": float(min_seed_guard_width_px),
        "max_seed_guard_width_px": float(max_seed_guard_width_px),
        "one_sided_removal": bool(one_sided_removal),
        "one_sided_min_candidate_points": int(one_sided_min_candidate_points),
        "one_sided_score_margin_ratio": float(one_sided_score_margin_ratio),
        "protect_ocr_extended_band": bool(protect_ocr_extended_band),
        "ocr_extended_protection_margin_px": float(ocr_extended_protection_margin_px),
        "ocr_extended_max_rescue_ratio": float(ocr_extended_max_rescue_ratio),
        "post_profile_side_suppression": bool(post_profile_side_suppression),
        "post_profile_width_margin": float(post_profile_width_margin),
        "post_profile_max_shrink_ratio": float(post_profile_max_shrink_ratio),
        "note": "Depth補正後マスクに対してOCR軸のseed近傍幅を背表紙コアとして推定し，側面候補が多い片側だけを削る．反対側の点群と，選択OCR文字領域をaxis方向へ延長した保護帯は削らない．背表紙削りすぎを避けるためs方向には切らない．",
    }

    if valid_count < 50:
        info["reason"] = "too few valid depth points"
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    rinfo = refine_info or {}

    # ===== 軸と中心の決定 =====
    selected = rinfo.get("selected_ocr_polygon", {}) if isinstance(rinfo, dict) else {}
    selected_poly = _poly_from_any(selected.get("poly")) if isinstance(selected, dict) else None

    axis = None
    center = None
    axis_source = None
    center_source = None
    mask_axis = _mask_pca_axis(mask01)

    if selected_poly is not None:
        c_poly, axis_poly, short_len, long_len = _polygon_center_and_axis(selected_poly)
        if axis_poly is not None and c_poly is not None:
            axis, axis_kind = _choose_axis_consistent_with_mask(axis_poly, mask_axis)
            axis = np.asarray(axis, dtype=np.float64).reshape(2)
            center = np.asarray(c_poly, dtype=np.float64).reshape(2)
            axis_source = f"selected_ocr_polygon_{axis_kind}"
            center_source = "selected_ocr_polygon_center"
            info["selected_ocr_text"] = selected.get("text")
            info["selected_ocr_transform_mode"] = selected.get("transform_mode")
            info["selected_ocr_poly"] = np.asarray(selected_poly, dtype=np.float32).tolist()
            info["selected_ocr_short_len_px"] = None if short_len is None else float(short_len)
            info["selected_ocr_long_len_px"] = None if long_len is None else float(long_len)
            info["mask_pca_axis"] = None if mask_axis is None else [float(mask_axis[0]), float(mask_axis[1])]

    if axis is None:
        axis = rinfo.get("axis", None)
        center = rinfo.get("ocr_center", None)
        if axis is not None and center is not None:
            axis = np.asarray(axis, dtype=np.float64).reshape(2)
            center = np.asarray(center, dtype=np.float64).reshape(2)
            axis_source = str(rinfo.get("axis_source", "refine_info_axis_fallback"))
            center_source = "refine_info_ocr_center_fallback"

    if axis is None or center is None:
        axis = mask_axis
        center = _mask_centroid(mask01)
        axis_source = "mask_pca_fallback"
        center_source = "mask_centroid_fallback"

    if axis is None or center is None:
        info["reason"] = "axis or center is unavailable"
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    axis = np.asarray(axis, dtype=np.float64).reshape(2)
    an = float(np.linalg.norm(axis))
    if an < 1e-9:
        info["reason"] = "invalid axis"
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)
    axis = axis / an
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    center = np.asarray(center, dtype=np.float64).reshape(2)

    ys, xs = np.where(valid)
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    rel = pts - center
    s_coords = rel @ axis
    t_coords = rel @ normal

    t_bin_size_px = max(float(t_bin_size_px), 1.0)
    t_min_all = float(np.min(t_coords))
    t_max_all = float(np.max(t_coords))
    s_min_all = float(np.min(s_coords))
    s_max_all = float(np.max(s_coords))

    n_t_bins = int(np.floor((t_max_all - t_min_all) / t_bin_size_px)) + 1
    if n_t_bins < 1:
        info["reason"] = "invalid t bin count"
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    t_bins = np.floor((t_coords - t_min_all) / t_bin_size_px).astype(np.int32)
    t_bins = np.clip(t_bins, 0, n_t_bins - 1)

    # seed_t は，幅ガードを使わない場合はOCR文字領域中心を基準(0.0)にする．
    # 幅ガードを使う場合は，OCR文字領域中心が背表紙中心からずれることがあるため，
    # Depth補正後マスクの重心をnormal方向に投影した値を基準にする．
    seed_t = 0.0
    seed_t_source = "ocr_polygon_center"
    mask_centroid_xy_for_seed = None
    mask_centroid_t_for_seed = None
    try:
        mask_centroid_xy_for_seed = np.mean(pts, axis=0)
        mask_centroid_t_for_seed = float(np.dot(mask_centroid_xy_for_seed - center, normal))
    except Exception:
        mask_centroid_xy_for_seed = None
        mask_centroid_t_for_seed = None

    center_mode = str(seed_guard_center_mode).lower()
    if bool(use_seed_width_guard) and center_mode in {"mask_centroid", "mask_center", "valid_depth_mask_centroid"}:
        if mask_centroid_t_for_seed is not None and np.isfinite(mask_centroid_t_for_seed):
            seed_t = float(mask_centroid_t_for_seed)
            seed_t_source = "valid_depth_mask_centroid"

    seed_bin = int(np.floor((seed_t - t_min_all) / t_bin_size_px))
    seed_bin = int(np.clip(seed_bin, 0, n_t_bins - 1))

    # ===== 各t列のs方向長さを計算 =====
    p_low = float(np.clip(float(span_percentiles[0]), 0.0, 49.0))
    p_high = float(np.clip(float(span_percentiles[1]), 51.0, 100.0))
    if p_high <= p_low:
        p_low, p_high = 2.0, 98.0

    column_records = []
    lengths = np.zeros((n_t_bins,), dtype=np.float64)
    counts = np.zeros((n_t_bins,), dtype=np.int32)

    for b in range(n_t_bins):
        idx = np.where(t_bins == b)[0]
        count = int(idx.size)
        counts[b] = count
        rec = {
            "t_bin": int(b),
            "t_min": float(t_min_all + b * t_bin_size_px),
            "t_max": float(t_min_all + (b + 1) * t_bin_size_px),
            "t_center": float(t_min_all + (b + 0.5) * t_bin_size_px),
            "point_count": count,
            "length_px": 0.0,
            "s_min": None,
            "s_max": None,
            "is_good": False,
            "is_relaxed_good": False,
            "is_seed_guard": False,
            "is_selected": False,
            "is_removed_candidate_side": False,
            "is_opposite_side_protected": False,
            "is_ocr_extended_protected": False,
        }
        if count >= max(3, int(min_points_per_t_bin)):
            sv = s_coords[idx]

            # 重要: ここでは単純な percentile span ではなく，
            # s方向に「連続して存在している最長区間」を列長として使えるようにする．
            # 左右の邪魔領域は，上下に点が散っていて percentile span だと長く見えることがあるため，
            # continuous_run を使うと短い列として判定しやすくなる．
            gap_allow_bins = int(round(float(s_gap_allow_px) / max(float(s_bin_size_px), 1.0)))
            col_len_info = _column_length_record_for_s_values(
                sv,
                s_origin=float(s_min_all),
                s_bin_size_px=float(s_bin_size_px),
                gap_allow_bins=int(gap_allow_bins),
                span_percentiles=(p_low, p_high),
                min_occupancy_density=max(float(min_occupancy_density), 1e-6),
            )

            if col_len_info is not None:
                mode = str(column_length_mode).lower()
                run_length = float(col_len_info.get("run_length_px", col_len_info.get("length_px", 0.0)))
                span_length = float(col_len_info.get("span_length_px", run_length))
                effective_length = float(col_len_info.get("effective_length_px", run_length))

                # 最長連続区間内の点数が少なすぎる場合は，ノイズ列として弱める．
                point_count_in_run = int(col_len_info.get("point_count_in_run", 0))
                run_point_ratio = float(point_count_in_run / max(count, 1))

                if mode in {"continuous", "continuous_run", "longest_run", "run"}:
                    length = run_length
                    length_source = "continuous_run"
                elif mode in {"span", "percentile_span", "percentile"}:
                    length = span_length
                    length_source = "percentile_span"
                else:
                    length = effective_length
                    length_source = str(col_len_info.get("effective_source", "effective"))

                if run_point_ratio < float(min_run_point_ratio):
                    length = float(length * max(0.0, run_point_ratio / max(float(min_run_point_ratio), 1e-6)))
                    length_source = f"{length_source}_downweighted_by_run_point_ratio"

                lengths[b] = float(length)
                rec.update({
                    "length_px": float(length),
                    "length_source": str(length_source),
                    "s_min": float(col_len_info.get("effective_s_min", col_len_info.get("s_min", float(np.min(sv))))),
                    "s_max": float(col_len_info.get("effective_s_max", col_len_info.get("s_max", float(np.max(sv))))),
                    "raw_s_min": float(np.min(sv)),
                    "raw_s_max": float(np.max(sv)),
                    "run_length_px": float(run_length),
                    "run_s_min": float(col_len_info.get("s_min", np.nan)),
                    "run_s_max": float(col_len_info.get("s_max", np.nan)),
                    "span_length_px": float(span_length),
                    "span_s_min": float(col_len_info.get("span_s_min", np.nan)),
                    "span_s_max": float(col_len_info.get("span_s_max", np.nan)),
                    "occupied_bin_count": int(col_len_info.get("occupied_bin_count", 0)),
                    "span_bin_count": int(col_len_info.get("span_bin_count", 0)),
                    "occupancy_density": float(col_len_info.get("occupancy_density", 0.0)),
                    "point_count_in_run": int(point_count_in_run),
                    "run_point_ratio": float(run_point_ratio),
                    "gap_allow_bins": int(gap_allow_bins),
                })
        column_records.append(rec)

    positive = lengths[lengths > 0]
    if positive.size < 2:
        info["reason"] = "too few positive length columns"
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    ref_percentile = float(np.clip(float(length_reference_percentile), 50.0, 100.0))
    robust_reference = float(np.percentile(positive, ref_percentile))
    global_max = float(np.max(positive))

    # 呼び出し側が0.90を渡してきても，safe版では削りすぎ防止のため上限を設ける．
    effective_min_ratio = float(min(float(min_length_ratio), 0.95))
    effective_relaxed_ratio = float(min(float(relaxed_edge_length_ratio), 0.95))
    core_threshold = float(robust_reference * effective_min_ratio)
    relaxed_threshold = float(robust_reference * effective_relaxed_ratio)

    good_bins = np.where((counts >= int(min_points_per_t_bin)) & (lengths >= core_threshold))[0].astype(np.int32)
    relaxed_bins = np.where((counts >= max(3, int(min_points_per_t_bin) // 2)) & (lengths >= relaxed_threshold))[0].astype(np.int32)

    def _groups(indices):
        return _find_consecutive_groups(np.asarray(indices, dtype=np.int32))

    def _dist_group_to_seed(g0: int, g1: int) -> int:
        if g0 <= seed_bin <= g1:
            return 0
        return int(min(abs(seed_bin - g0), abs(seed_bin - g1)))

    # ===== seed width guard =====
    # shape_info の width_median_px は長手方向スライスの典型幅なので，
    # 側面混入時でも背表紙本体の幅に比較的近い．
    shape_info = rinfo.get("shape_rectangularity", {}) if isinstance(rinfo, dict) else {}
    shape_width_median_px = None
    if isinstance(shape_info, dict):
        try:
            val = float(shape_info.get("width_median_px", 0.0))
            if np.isfinite(val) and val > 1.0:
                shape_width_median_px = val
        except Exception:
            shape_width_median_px = None

    seed_guard_used = False
    seed_guard_half_width_px = None
    seed_guard_width_source = None
    seed_guard_base_width_px = None
    seed_width_profile_info = {"used": False, "reason": "not evaluated"}
    seed_guard_bins = np.array([], dtype=np.int32)
    t_centers = np.asarray([float(r["t_center"]) for r in column_records], dtype=np.float64)

    # ===== seed width guard 用の背表紙幅推定 =====
    # 背表紙幅は，OCR短辺ではなくDepth補正後マスクのnormal方向幅を主基準にする．
    # カメラ視点では側面部が混ざると幅が広がる方向に出やすいため，
    # s方向スライスごとのnormal幅の低パーセンタイル（デフォルト2%）を使う．
    mask_width_percentile_px = None
    try:
        sp0, sp1 = float(seed_width_profile_s_percentiles[0]), float(seed_width_profile_s_percentiles[1])
        sp0 = float(np.clip(sp0, 0.0, 49.0))
        sp1 = float(np.clip(sp1, 51.0, 100.0))
        tp0, tp1 = float(seed_width_profile_t_percentiles[0]), float(seed_width_profile_t_percentiles[1])
        tp0 = float(np.clip(tp0, 0.0, 49.0))
        tp1 = float(np.clip(tp1, 51.0, 100.0))
        s0_prof, s1_prof = np.percentile(s_coords, [sp0, sp1])
        if np.isfinite(s0_prof) and np.isfinite(s1_prof) and s1_prof > s0_prof:
            n_s_bins = max(8, min(100, int(round((float(s1_prof) - float(s0_prof)) / max(float(s_bin_size_px), 1.0)))))
            edges = np.linspace(float(s0_prof), float(s1_prof), n_s_bins + 1)
            widths = []
            for bi in range(n_s_bins):
                m = (s_coords >= edges[bi]) & (s_coords < edges[bi + 1])
                if int(np.count_nonzero(m)) < max(8, int(min_points_per_t_bin)):
                    continue
                tq0, tq1 = np.percentile(t_coords[m], [tp0, tp1])
                width_i = float(max(0.0, float(tq1) - float(tq0)))
                if np.isfinite(width_i) and width_i > 1.0:
                    widths.append(width_i)
            if len(widths) >= 4:
                widths_np = np.asarray(widths, dtype=np.float64)
                wp = float(np.clip(float(seed_width_mask_percentile), 0.0, 100.0))
                raw_width_percentile_px = float(np.percentile(widths_np, wp))
                width_median_px = float(np.median(widths_np))
                width_p10_px = float(np.percentile(widths_np, 10.0))
                width_p15_px = float(np.percentile(widths_np, 15.0))
                width_p20_px = float(np.percentile(widths_np, 20.0))
                width_p85_px = float(np.percentile(widths_np, 85.0))
                width_p95_px = float(np.percentile(widths_np, 95.0))

                min_ratio = float(seed_width_low_percentile_min_median_ratio)
                min_ratio = float(np.clip(min_ratio, 0.0, 1.0))
                selected_width_px = raw_width_percentile_px
                width_selection_reason = "raw_low_percentile"

                # 2%タイルがDepth欠け・局所ノイズで極端に小さく出る場合がある．
                # その場合は2%タイルを捨て，中央値または指定パーセンタイルにフォールバックする．
                if (not np.isfinite(selected_width_px)) or selected_width_px <= 1.0 or selected_width_px < min_ratio * width_median_px:
                    mode = str(seed_width_fallback_mode).lower()
                    if mode in {"p10", "percentile10", "10"}:
                        selected_width_px = width_p10_px
                        width_selection_reason = f"fallback_p10_raw_p{wp:.1f}_lt_{min_ratio:.2f}x_median"
                    elif mode in {"p15", "percentile15", "15"}:
                        selected_width_px = width_p15_px
                        width_selection_reason = f"fallback_p15_raw_p{wp:.1f}_lt_{min_ratio:.2f}x_median"
                    elif mode in {"p20", "percentile20", "20"}:
                        selected_width_px = width_p20_px
                        width_selection_reason = f"fallback_p20_raw_p{wp:.1f}_lt_{min_ratio:.2f}x_median"
                    else:
                        selected_width_px = width_median_px
                        width_selection_reason = f"fallback_median_raw_p{wp:.1f}_lt_{min_ratio:.2f}x_median"

                mask_width_percentile_px = float(selected_width_px)
                seed_width_profile_info = {
                    "used": True,
                    "reason": "ok",
                    "method": "s_slice_normal_width_low_percentile_with_median_fallback",
                    "width_percentile_px": float(mask_width_percentile_px),
                    "width_selected_px": float(mask_width_percentile_px),
                    "width_selection_reason": str(width_selection_reason),
                    "width_raw_percentile_px": float(raw_width_percentile_px),
                    "width_percentile": float(wp),
                    "width_median_px": float(width_median_px),
                    "width_p10_px": float(width_p10_px),
                    "width_p15_px": float(width_p15_px),
                    "width_p20_px": float(width_p20_px),
                    "width_p85_px": float(width_p85_px),
                    "width_p95_px": float(width_p95_px),
                    "width_min_px": float(np.min(widths_np)),
                    "width_max_px": float(np.max(widths_np)),
                    "valid_slice_count": int(len(widths_np)),
                    "low_percentile_min_median_ratio": float(min_ratio),
                    "fallback_mode": str(seed_width_fallback_mode),
                    "s_percentiles": [float(sp0), float(sp1)],
                    "t_percentiles_per_slice": [float(tp0), float(tp1)],
                }
            else:
                seed_width_profile_info = {
                    "used": False,
                    "reason": f"too few valid width slices: {len(widths)}",
                }
    except Exception as e:
        seed_width_profile_info = {"used": False, "reason": f"error: {e}"}

    if bool(use_seed_width_guard):
        if mask_width_percentile_px is not None and np.isfinite(mask_width_percentile_px) and mask_width_percentile_px > 1.0:
            seed_guard_base_width_px = float(mask_width_percentile_px)
            width_reason = seed_width_profile_info.get("width_selection_reason") if isinstance(seed_width_profile_info, dict) else None
            if width_reason:
                seed_guard_width_source = f"mask_width_p{float(seed_width_mask_percentile):.1f}_{width_reason}"
            else:
                seed_guard_width_source = f"mask_width_p{float(seed_width_mask_percentile):.1f}"
        elif shape_width_median_px is not None:
            seed_guard_base_width_px = float(shape_width_median_px)
            seed_guard_width_source = "shape_width_median_fallback"

    if bool(use_seed_width_guard) and seed_guard_base_width_px is not None:
        seed_guard_width_px = float(seed_guard_base_width_px) * float(seed_width_guard_scale)
        seed_guard_width_px = float(max(seed_guard_width_px, float(min_seed_guard_width_px)))
        seed_guard_width_px = float(min(seed_guard_width_px, float(max_seed_guard_width_px)))
        seed_guard_half_width_px = 0.5 * seed_guard_width_px

        guard = np.abs(t_centers - seed_t) <= seed_guard_half_width_px
        guard &= counts >= max(3, int(min_points_per_t_bin) // 2)

        seed_guard_bins = np.where(guard)[0].astype(np.int32)
        if seed_guard_bins.size >= int(min_selected_t_bins):
            seed_guard_used = True

    if seed_guard_used:
        # guard内のうち，seedに最も近い連続groupのみ残す．
        groups = _groups(seed_guard_bins)
        best_group = min(groups, key=lambda g: _dist_group_to_seed(int(g[0]), int(g[1]))) if groups else None
        if best_group is None:
            info["reason"] = "failed to select seed guard group"
            return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)
        g0, g1 = int(best_group[0]), int(best_group[1])
        keep_bin_indices = np.arange(g0, g1 + 1, dtype=np.int32)
        reference_source = "shape_width_median_seed_guard"
    else:
        # width guardが使えない場合のみ，長さ閾値に基づくseed近傍グループを使う．
        candidate_bins = relaxed_bins if relaxed_bins.size > 0 else good_bins
        groups = _groups(candidate_bins)
        if not groups:
            info.update({
                "reason": "no candidate bins from length threshold; reverted to depth-filtered mask",
                "global_max_length_px": global_max,
                "robust_reference_length_px": robust_reference,
                "core_threshold_px": core_threshold,
                "relaxed_threshold_px": relaxed_threshold,
                "seed_bin": int(seed_bin),
            })
            return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

        best_group = None
        best_score = -1e18
        for g0_, g1_ in groups:
            g0_, g1_ = int(g0_), int(g1_)
            dist = _dist_group_to_seed(g0_, g1_)
            width_bins = int(g1_ - g0_ + 1)
            mean_len = float(np.mean(lengths[g0_:g1_ + 1]))
            score = -100.0 * float(dist) + 2.0 * float(width_bins) + 0.01 * mean_len
            if score > best_score:
                best_score = score
                best_group = (g0_, g1_)
        if best_group is None:
            info["reason"] = "failed to select seed group"
            return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

        g0, g1 = best_group
        g0 = max(0, int(g0) - int(expand_selected_t_bins))
        g1 = min(n_t_bins - 1, int(g1) + int(expand_selected_t_bins))
        keep_bin_indices = np.arange(g0, g1 + 1, dtype=np.int32)
        reference_source = "relaxed_length_seed_group"

    selected_bin_count = int(g1 - g0 + 1)
    if selected_bin_count < int(min_selected_t_bins):
        info.update({
            "reason": "selected t group is too narrow; reverted to depth-filtered mask",
            "selected_group": [int(g0), int(g1)],
            "selected_bin_count": selected_bin_count,
        })
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    # ===== OCR文字領域をaxis方向へ延長した保護帯 =====
    # 選択OCR polygonを，採用したaxisとnormalで見たときのnormal方向範囲を使う．
    # s方向には制限を設けないため，Depth補正後マスク全体の端まで延長された帯になる．
    # この帯に入る有効点は，片側側面除去の削除候補になっても必ず残す．
    ocr_extended_protection_used = False
    ocr_protection_point = np.zeros((pts.shape[0],), dtype=bool)
    ocr_protected_bins = np.zeros((n_t_bins,), dtype=bool)
    ocr_protection_t_min = None
    ocr_protection_t_max = None
    ocr_protection_width_px = None
    ocr_protection_margin = float(max(0.0, float(ocr_extended_protection_margin_px)))

    if bool(protect_ocr_extended_band) and selected_poly is not None:
        try:
            poly_pts = np.asarray(selected_poly, dtype=np.float64).reshape(-1, 2)
            if poly_pts.shape[0] >= 4:
                poly_t = (poly_pts - center) @ normal
                t0 = float(np.min(poly_t) - ocr_protection_margin)
                t1 = float(np.max(poly_t) + ocr_protection_margin)
                if np.isfinite(t0) and np.isfinite(t1) and (t1 - t0) > 1.0:
                    ocr_protection_t_min = t0
                    ocr_protection_t_max = t1
                    ocr_protection_width_px = float(t1 - t0)
                    ocr_protection_point = (t_coords >= t0) & (t_coords <= t1)
                    # 可視化・ログ用に，保護帯と重なるt-binも記録する．実際の保護はpoint単位で行う．
                    for b in range(n_t_bins):
                        bt0 = float(t_min_all + b * t_bin_size_px)
                        bt1 = float(t_min_all + (b + 1) * t_bin_size_px)
                        if bt1 >= t0 and bt0 <= t1:
                            ocr_protected_bins[b] = True
                    ocr_extended_protection_used = bool(np.count_nonzero(ocr_protection_point) > 0)
        except Exception as e:
            info["ocr_extended_protection_error"] = str(e)

    # ===== 片側側面除去 =====
    # 側面は背表紙の片側にしか見えない，という物理的制約を利用する．
    # まず seed_guard/列長から背表紙コア [g0, g1] を推定し，
    # その外側にある削除候補が t<0 側に多いか，t>0 側に多いかを判定する．
    # その後，削除候補が多い側だけを削り，反対側の点群は保護する．
    negative_bins = np.arange(0, max(0, int(g0)), dtype=np.int32)
    positive_bins = np.arange(min(n_t_bins, int(g1) + 1), n_t_bins, dtype=np.int32)

    negative_candidate_points = int(np.sum(counts[negative_bins])) if negative_bins.size > 0 else 0
    positive_candidate_points = int(np.sum(counts[positive_bins])) if positive_bins.size > 0 else 0
    negative_candidate_length_sum = float(np.sum(lengths[negative_bins])) if negative_bins.size > 0 else 0.0
    positive_candidate_length_sum = float(np.sum(lengths[positive_bins])) if positive_bins.size > 0 else 0.0

    # 点数を主スコアにしつつ，長く伸びている外側列も少し評価する．
    # これにより「外側に多く残っている側」を側面候補として選ぶ．
    negative_side_score = float(negative_candidate_points) + 0.02 * negative_candidate_length_sum
    positive_side_score = float(positive_candidate_points) + 0.02 * positive_candidate_length_sum

    side_removal_used = False
    remove_side = None
    protected_side = None
    side_decision_ambiguous = False

    keep_bins = np.zeros((n_t_bins,), dtype=bool)
    core_bins = np.zeros((n_t_bins,), dtype=bool)
    core_bins[keep_bin_indices] = True

    total_outside_points = negative_candidate_points + positive_candidate_points
    if bool(one_sided_removal) and total_outside_points >= int(one_sided_min_candidate_points):
        min_score = min(negative_side_score, positive_side_score)
        max_score = max(negative_side_score, positive_side_score)
        ratio = float(max_score / max(min_score, 1e-6))
        side_decision_ambiguous = bool(ratio < float(one_sided_score_margin_ratio))

        if positive_side_score >= negative_side_score:
            # t正側に側面候補が多い → t正側のみ削る．t負側は保護する．
            remove_side = "positive"
            protected_side = "negative"
            keep_bins[: int(g1) + 1] = True
            side_removal_used = True
        else:
            # t負側に側面候補が多い → t負側のみ削る．t正側は保護する．
            remove_side = "negative"
            protected_side = "positive"
            keep_bins[int(g0):] = True
            side_removal_used = True
    else:
        # 外側候補が少ない場合は，従来どおりコアだけを使う．
        # ここに来る場合，そもそも側面候補が少ないと判断している．
        keep_bins[keep_bin_indices] = True

    keep_point_before_ocr_protection = keep_bins[t_bins]
    removed_candidate_before_ocr_protection = ~keep_point_before_ocr_protection

    # OCR延長帯保護は背表紙文字帯を守るために有効だが、
    # ケース4のように側面候補を大量に救済すると側面が残る。
    # そのため、削除候補のうちOCR保護で救済される割合が大きすぎる場合は、
    # 削除候補側のOCR救済を無効化する。すでにkeepされているcore内は影響しない。
    raw_rescue_mask = removed_candidate_before_ocr_protection & ocr_protection_point
    raw_rescue_count = int(np.count_nonzero(raw_rescue_mask))
    removed_candidate_count_before_ocr = int(np.count_nonzero(removed_candidate_before_ocr_protection))
    rescue_ratio = float(raw_rescue_count / max(removed_candidate_count_before_ocr, 1))
    ocr_protection_limited = False
    ocr_protection_limit_reason = None
    effective_ocr_protection_point = ocr_protection_point.copy()
    if ocr_extended_protection_used and removed_candidate_count_before_ocr >= int(one_sided_min_candidate_points):
        if rescue_ratio > float(ocr_extended_max_rescue_ratio):
            # candidate側だけ救済しない。core内は元々keepなので保護不要。
            effective_ocr_protection_point = ocr_protection_point & keep_point_before_ocr_protection
            ocr_protection_limited = True
            ocr_protection_limit_reason = (
                f"OCR extended protection rescued too many candidate-side points: "
                f"rescue_ratio={rescue_ratio:.3f} > {float(ocr_extended_max_rescue_ratio):.3f}"
            )

    if ocr_extended_protection_used:
        keep_point = keep_point_before_ocr_protection | effective_ocr_protection_point
    else:
        keep_point = keep_point_before_ocr_protection

    rescued_by_ocr_protection_count = int(np.count_nonzero(removed_candidate_before_ocr_protection & effective_ocr_protection_point))
    ocr_protected_point_count = int(np.count_nonzero(effective_ocr_protection_point))

    mask_after = np.zeros_like(mask01, dtype=np.uint8)
    mask_after[ys[keep_point], xs[keep_point]] = 1

    # 極小孤立成分のみ除去．強くすると背表紙が欠けるため弱くする．
    mask_after, component_info = _filter_small_components(mask_after, min_area_ratio=0.0002)

    # component filter で保護帯の点が落ちた場合は再度救済する．
    # OCR文字領域をaxis方向に延長した帯は信頼領域として扱うため，最終出力でも削らない．
    if ocr_extended_protection_used:
        protected_after_component = np.zeros_like(mask01, dtype=np.uint8)
        protected_after_component[ys[effective_ocr_protection_point], xs[effective_ocr_protection_point]] = 1
        mask_after = ((mask_after > 0) | (protected_after_component > 0)).astype(np.uint8)

    profile_side_suppression_info = {"used": False, "reason": "disabled"}
    if bool(post_profile_side_suppression):
        try:
            mask_profile, profile_side_suppression_info = _suppress_side_protrusions_axis_profile(
                mask_after,
                axis=axis,
                center=center,
                bin_size_px=8.0,
                profile_width_margin=float(post_profile_width_margin),
                min_profile_half_width_px=14.0,
                max_profile_shrink_ratio=float(post_profile_max_shrink_ratio),
            )
            if isinstance(profile_side_suppression_info, dict) and profile_side_suppression_info.get("used"):
                mask_after = mask_profile.astype(np.uint8)
                # OCR保護が制限されていない場合だけ、最終的に保護点を戻す。
                # 制限されたケースでは、側面候補を守りすぎないことを優先する。
                if ocr_extended_protection_used and not ocr_protection_limited:
                    protected_after_profile = np.zeros_like(mask01, dtype=np.uint8)
                    protected_after_profile[ys[effective_ocr_protection_point], xs[effective_ocr_protection_point]] = 1
                    mask_after = ((mask_after > 0) | (protected_after_profile > 0)).astype(np.uint8)
        except Exception as e:
            profile_side_suppression_info = {"used": False, "reason": f"error: {e}"}

    depth_after = depth_masked.copy()
    depth_after[mask_after == 0] = 0

    valid_after_count = int(np.count_nonzero(depth_after > 0))
    valid_keep_ratio = float(valid_after_count / max(valid_count, 1))

    set_good = set(int(v) for v in good_bins.tolist())
    set_relaxed = set(int(v) for v in relaxed_bins.tolist())
    set_guard = set(int(v) for v in seed_guard_bins.tolist())
    set_ocr_protected = set(int(v) for v in np.where(ocr_protected_bins)[0].tolist())

    for rec in column_records:
        b = int(rec["t_bin"])
        rec["is_good"] = bool(b in set_good)
        rec["is_relaxed_good"] = bool(b in set_relaxed)
        rec["is_seed_guard"] = bool(b in set_guard)
        rec["is_selected"] = bool(keep_bins[b])
        rec["is_ocr_extended_protected"] = bool(b in set_ocr_protected)
        if remove_side == "negative":
            rec["is_removed_candidate_side"] = bool(b < int(g0))
            rec["is_opposite_side_protected"] = bool(b > int(g1))
        elif remove_side == "positive":
            rec["is_removed_candidate_side"] = bool(b > int(g1))
            rec["is_opposite_side_protected"] = bool(b < int(g0))
        else:
            rec["is_removed_candidate_side"] = False
            rec["is_opposite_side_protected"] = False

    info.update({
        "used": True,
        "reason": "ok",
        "axis_source": axis_source,
        "center_source": center_source,
        "axis": [float(axis[0]), float(axis[1])],
        "normal": [float(normal[0]), float(normal[1])],
        "center": [float(center[0]), float(center[1])],
        "t_min_all": float(t_min_all),
        "t_max_all": float(t_max_all),
        "s_min_all": float(s_min_all),
        "s_max_all": float(s_max_all),
        "n_t_bins": int(n_t_bins),
        "seed_t": float(seed_t),
        "seed_bin": int(seed_bin),
        "global_max_length_px": float(global_max),
        "max_length_px": float(global_max),
        "robust_reference_length_px": float(robust_reference),
        "reference_length_px": float(robust_reference),
        "reference_source": reference_source,
        "core_threshold_px": float(core_threshold),
        "relaxed_threshold_px": float(relaxed_threshold),
        "length_threshold_px": float(relaxed_threshold),
        "effective_min_length_ratio": float(effective_min_ratio),
        "effective_relaxed_edge_length_ratio": float(effective_relaxed_ratio),
        "shape_width_median_px": None if shape_width_median_px is None else float(shape_width_median_px),
        "seed_width_profile_info": seed_width_profile_info,
        "seed_guard_width_source": seed_guard_width_source,
        "seed_guard_base_width_px": None if seed_guard_base_width_px is None else float(seed_guard_base_width_px),
        "seed_guard_used": bool(seed_guard_used),
        "seed_guard_half_width_px": None if seed_guard_half_width_px is None else float(seed_guard_half_width_px),
        "seed_guard_width_px": None if seed_guard_half_width_px is None else float(2.0 * seed_guard_half_width_px),
        "good_bins": [int(v) for v in good_bins.tolist()],
        "relaxed_bins": [int(v) for v in relaxed_bins.tolist()],
        "seed_guard_bins": [int(v) for v in seed_guard_bins.tolist()],
        "one_sided_removal_used": bool(side_removal_used),
        "remove_side": remove_side,
        "protected_side": protected_side,
        "opposite_side_protected": bool(side_removal_used),
        "side_decision_ambiguous": bool(side_decision_ambiguous),
        "ocr_extended_protection_used": bool(ocr_extended_protection_used),
        "ocr_extended_protection_t_min": None if ocr_protection_t_min is None else float(ocr_protection_t_min),
        "ocr_extended_protection_t_max": None if ocr_protection_t_max is None else float(ocr_protection_t_max),
        "ocr_extended_protection_width_px": None if ocr_protection_width_px is None else float(ocr_protection_width_px),
        "ocr_extended_protection_margin_px": float(ocr_protection_margin),
        "ocr_extended_protected_point_count": int(ocr_protected_point_count),
        "rescued_by_ocr_extended_protection_count": int(rescued_by_ocr_protection_count),
        "ocr_extended_raw_rescue_count": int(raw_rescue_count),
        "ocr_extended_raw_rescue_ratio": float(rescue_ratio),
        "ocr_extended_protection_limited": bool(ocr_protection_limited),
        "ocr_extended_protection_limit_reason": ocr_protection_limit_reason,
        "removed_candidate_count_before_ocr_protection": int(removed_candidate_count_before_ocr),
        "post_profile_side_suppression_info": profile_side_suppression_info,
        "seed_t_source": str(seed_t_source),
        "mask_centroid_t_for_seed": None if mask_centroid_t_for_seed is None else float(mask_centroid_t_for_seed),
        "ocr_extended_protected_bins": [int(v) for v in np.where(ocr_protected_bins)[0].tolist()],
        "negative_candidate_points": int(negative_candidate_points),
        "positive_candidate_points": int(positive_candidate_points),
        "negative_candidate_length_sum": float(negative_candidate_length_sum),
        "positive_candidate_length_sum": float(positive_candidate_length_sum),
        "negative_side_score": float(negative_side_score),
        "positive_side_score": float(positive_side_score),
        "core_group": [int(g0), int(g1)],
        "core_t_min": float(t_min_all + g0 * t_bin_size_px),
        "core_t_max": float(t_min_all + (g1 + 1) * t_bin_size_px),
        "groups": [[int(a), int(b)] for a, b in _groups(keep_bin_indices)],
        "selected_group": [int(g0), int(g1)],
        "selected_group_distance_from_seed_bins": int(_dist_group_to_seed(g0, g1)),
        "selected_bin_count": int(selected_bin_count),
        "selected_t_min": float(t_min_all + g0 * t_bin_size_px),
        "selected_t_max": float(t_min_all + (g1 + 1) * t_bin_size_px),
        "valid_count_after": int(valid_after_count),
        "valid_keep_ratio": float(valid_keep_ratio),
        "component_filter": component_info,
        "column_records": [
            {
                "t_bin": int(r["t_bin"]),
                "point_count": int(r["point_count"]),
                "length_px": float(r["length_px"]),
                "s_min": None if r.get("s_min") is None else float(r["s_min"]),
                "s_max": None if r.get("s_max") is None else float(r["s_max"]),
                "length_source": r.get("length_source"),
                "run_length_px": None if r.get("run_length_px") is None else float(r["run_length_px"]),
                "span_length_px": None if r.get("span_length_px") is None else float(r["span_length_px"]),
                "run_point_ratio": None if r.get("run_point_ratio") is None else float(r["run_point_ratio"]),
                "occupancy_density": None if r.get("occupancy_density") is None else float(r["occupancy_density"]),
                "point_count_in_run": None if r.get("point_count_in_run") is None else int(r["point_count_in_run"]),
                "gap_allow_bins": None if r.get("gap_allow_bins") is None else int(r["gap_allow_bins"]),
                "is_good": bool(r.get("is_good", False)),
                "is_relaxed_good": bool(r.get("is_relaxed_good", False)),
                "is_seed_guard": bool(r.get("is_seed_guard", False)),
                "is_selected": bool(r.get("is_selected", False)),
                "is_removed_candidate_side": bool(r.get("is_removed_candidate_side", False)),
                "is_opposite_side_protected": bool(r.get("is_opposite_side_protected", False)),
                "is_ocr_extended_protected": bool(r.get("is_ocr_extended_protected", False)),
            }
            for r in column_records
        ],
    })

    if valid_keep_ratio < float(min_valid_keep_ratio):
        info["used"] = False
        info["reason"] = f"too much valid depth removed: valid_keep_ratio={valid_keep_ratio:.3f}; reverted to depth-filtered mask"
        info["valid_count_after"] = valid_count
        info["valid_keep_ratio"] = 1.0
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    return (mask_after, depth_after, info) if return_info else (mask_after, depth_after)


def estimate_book_width_from_filtered_mask_axis(
    mask01: np.ndarray,
    depth_masked: np.ndarray,
    intr,
    depth_scale: float,
    column_info: dict | None,
    refine_info: dict | None = None,
    *,
    width_percentiles: tuple[float, float] = (2.0, 98.0),
    min_width_mm: float = 2.0,
    max_width_mm: float = 150.0,
):
    """
    側面除去後の最終mask/depthから，ハンド開口幅用の書籍幅を推定する．

    重要:
      estimate_book_width(pts_f, mean, pc1, pc2) は3D PCA軸に依存するため，
      側面部が少し残るだけでpc2が斜めを向き，開口幅が大きく崩れることがある．
      ここでは，側面除去で使った画像上の背表紙軸normal方向の画素幅を使い，
      median depthとカメラ内部パラメータでメートル換算する．
    """
    info = {"used": False, "reason": ""}
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    depth_masked = np.asarray(depth_masked)
    valid = (mask01 > 0) & (depth_masked > 0)
    if int(np.count_nonzero(valid)) < 20:
        info["reason"] = "too few valid pixels"
        return None, info

    cinfo = column_info or {}
    rinfo = refine_info or {}
    axis = cinfo.get("axis", None) or rinfo.get("axis", None)
    center = cinfo.get("center", None) or rinfo.get("ocr_center", None)
    if axis is None or center is None:
        info["reason"] = "axis or center unavailable"
        return None, info

    axis = np.asarray(axis, dtype=np.float64).reshape(2)
    an = float(np.linalg.norm(axis))
    if an < 1e-9:
        info["reason"] = "invalid axis"
        return None, info
    axis = axis / an
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    center = np.asarray(center, dtype=np.float64).reshape(2)

    ys, xs = np.where(valid)
    pts2 = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    t = (pts2 - center) @ normal

    p0, p1 = width_percentiles
    p0 = float(np.clip(p0, 0.0, 49.0))
    p1 = float(np.clip(p1, 51.0, 100.0))
    t0, t1 = np.percentile(t, [p0, p1])
    width_px = float(max(0.0, t1 - t0))

    z_raw = depth_masked[valid]
    z_med_m = float(np.median(z_raw) * float(depth_scale))
    if not np.isfinite(z_med_m) or z_med_m <= 0.0:
        info["reason"] = "invalid median depth"
        return None, info

    fx = float(intr.fx)
    fy = float(intr.fy)
    scale_m_per_px = z_med_m * float(np.sqrt((normal[0] / fx) ** 2 + (normal[1] / fy) ** 2))
    width_m = float(width_px * scale_m_per_px)
    width_mm = width_m * 1000.0

    info.update({
        "used": True,
        "reason": "ok",
        "method": "filtered_mask_axis_pixel_width_to_metric",
        "width_px": float(width_px),
        "width_m": float(width_m),
        "width_mm": float(width_mm),
        "z_median_m": float(z_med_m),
        "scale_m_per_px": float(scale_m_per_px),
        "axis": [float(axis[0]), float(axis[1])],
        "normal": [float(normal[0]), float(normal[1])],
        "center": [float(center[0]), float(center[1])],
        "width_percentiles": [float(p0), float(p1)],
        "valid_pixel_count": int(np.count_nonzero(valid)),
    })

    if (not np.isfinite(width_m)) or width_mm < float(min_width_mm) or width_mm > float(max_width_mm):
        info["used"] = False
        info["reason"] = f"estimated width out of range: {width_mm:.3f} mm"
        return None, info

    return width_m, info

def save_spine_column_length_debug(
    shot_dir: Path,
    color_np: np.ndarray,
    mask_before: np.ndarray,
    mask_after: np.ndarray,
    depth_before: np.ndarray,
    depth_after: np.ndarray,
    column_info: dict | None,
    stem: str,
):
    """Depth補正後の主成分方向点群列フィルタのデバッグ画像を保存する．"""
    debug_dir = Path(shot_dir) / "debug_ocr_band"
    debug_dir.mkdir(parents=True, exist_ok=True)

    mask_before = (mask_before > 0).astype(np.uint8)
    mask_after = (mask_after > 0).astype(np.uint8)
    valid_before = ((mask_before > 0) & (np.asarray(depth_before) > 0)).astype(np.uint8)
    valid_after = ((mask_after > 0) & (np.asarray(depth_after) > 0)).astype(np.uint8)

    cv2.imwrite(str(debug_dir / f"{stem}_spine_column_mask_before.png"), mask_before * 255)
    cv2.imwrite(str(debug_dir / f"{stem}_spine_column_mask_after.png"), mask_after * 255)
    cv2.imwrite(str(debug_dir / f"{stem}_spine_column_valid_before.png"), valid_before * 255)
    cv2.imwrite(str(debug_dir / f"{stem}_spine_column_valid_after.png"), valid_after * 255)

    removed = ((valid_before == 1) & (valid_after == 0)).astype(np.uint8)
    kept = ((valid_before == 1) & (valid_after == 1)).astype(np.uint8)

    overlay = color_np.copy()
    overlay[kept == 1] = (0, 255, 0)
    overlay[removed == 1] = (0, 0, 255)
    blended = cv2.addWeighted(color_np, 0.65, overlay, 0.35, 0)
    cv2.imwrite(str(debug_dir / f"{stem}_spine_column_overlay_kept_removed.png"), blended)

    # 軸，選択t列境界，補正後輪郭を描く．
    axis_img = color_np.copy()
    info = column_info or {}
    center = info.get("center")
    axis = info.get("axis")
    normal = info.get("normal")
    selected_t_min = info.get("selected_t_min")
    selected_t_max = info.get("selected_t_max")
    if center is not None and axis is not None:
        cx, cy = float(center[0]), float(center[1])
        ax, ay = float(axis[0]), float(axis[1])
        length = 1200.0
        p1 = (int(round(cx - ax * length)), int(round(cy - ay * length)))
        p2 = (int(round(cx + ax * length)), int(round(cy + ay * length)))
        cv2.line(axis_img, p1, p2, (0, 0, 255), 2)
        cv2.circle(axis_img, (int(round(cx)), int(round(cy))), 6, (255, 0, 0), -1)

        if normal is not None and selected_t_min is not None and selected_t_max is not None:
            nx, ny = float(normal[0]), float(normal[1])
            for tval in (float(selected_t_min), float(selected_t_max)):
                ox = nx * tval
                oy = ny * tval
                q1 = (int(round(cx + ox - ax * length)), int(round(cy + oy - ay * length)))
                q2 = (int(round(cx + ox + ax * length)), int(round(cy + oy + ay * length)))
                cv2.line(axis_img, q1, q2, (255, 0, 255), 2)

    contours, _ = cv2.findContours(mask_after, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(axis_img, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(debug_dir / f"{stem}_spine_column_axis_selected.png"), axis_img)

    # t列ごとの長さプロファイルを簡易グラフで保存する．
    records = info.get("column_records", [])
    if records:
        graph_w, graph_h = 1000, 320
        margin_l, margin_r, margin_t, margin_b = 50, 20, 20, 45
        graph = np.full((graph_h, graph_w, 3), 255, dtype=np.uint8)
        plot_w = graph_w - margin_l - margin_r
        plot_h = graph_h - margin_t - margin_b
        max_len = max(float(info.get("max_length_px", 1.0)), 1.0)
        n = len(records)

        # 閾値線
        th = float(info.get("length_threshold_px", 0.0))
        y_th = int(round(margin_t + plot_h * (1.0 - min(th / max_len, 1.0))))
        cv2.line(graph, (margin_l, y_th), (graph_w - margin_r, y_th), (0, 0, 255), 1)
        cv2.putText(graph, "length threshold", (margin_l + 5, max(15, y_th - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        selected = info.get("selected_group", None)
        for i, rec in enumerate(records):
            x0 = int(round(margin_l + plot_w * i / max(n, 1)))
            x1 = int(round(margin_l + plot_w * (i + 1) / max(n, 1)))
            length_px = float(rec.get("length_px", 0.0))
            bar_h = int(round(plot_h * min(length_px / max_len, 1.0)))
            y0 = margin_t + plot_h - bar_h
            y1 = margin_t + plot_h
            color = (180, 180, 180)
            if bool(rec.get("is_relaxed_good", False)):
                color = (180, 220, 180)
            if bool(rec.get("is_good", False)):
                color = (0, 180, 0)
            if selected is not None and int(selected[0]) <= i <= int(selected[1]):
                color = (0, 120, 255)
            cv2.rectangle(graph, (x0, y0), (max(x0 + 1, x1 - 1), y1), color, -1)

        cv2.line(graph, (margin_l, margin_t), (margin_l, margin_t + plot_h), (0, 0, 0), 1)
        cv2.line(graph, (margin_l, margin_t + plot_h), (graph_w - margin_r, margin_t + plot_h), (0, 0, 0), 1)
        cv2.putText(graph, "t-bin", (graph_w // 2 - 30, graph_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(graph, "s length", (5, margin_t + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite(str(debug_dir / f"{stem}_spine_column_length_profile.png"), graph)

    save_json(debug_dir / f"{stem}_spine_column_log.json", info)
    print(f"✔ Saved spine-column debug files: {debug_dir}")



def _intr_params_for_plane_filter(intr):
    """pyrealsense2.intrinsics 互換の内部パラメータをfloatで取り出す．"""
    return {
        "width": int(getattr(intr, "width", 0)),
        "height": int(getattr(intr, "height", 0)),
        "fx": float(getattr(intr, "fx")),
        "fy": float(getattr(intr, "fy")),
        "ppx": float(getattr(intr, "ppx")),
        "ppy": float(getattr(intr, "ppy")),
    }


def _mask_depth_to_points_uv_for_plane_filter(
    mask01: np.ndarray,
    depth_masked: np.ndarray,
    intr,
    depth_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    mask & depth>0 の画素をカメラ座標点群とuvへ変換する．
    戻り値:
      points_xyz_m: (N,3) [m]
      uv:           (N,2) [u,v] int
    """
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    depth = np.asarray(depth_masked)
    valid = (mask01 > 0) & (depth > 0)
    ys, xs = np.where(valid)
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)

    z = depth[ys, xs].astype(np.float64) * float(depth_scale)
    good = np.isfinite(z) & (z > 0.0)
    xs = xs[good].astype(np.float64)
    ys = ys[good].astype(np.float64)
    z = z[good]
    if z.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)

    fx = float(intr.fx)
    fy = float(intr.fy)
    ppx = float(intr.ppx)
    ppy = float(intr.ppy)

    x = (xs - ppx) / fx * z
    y = (ys - ppy) / fy * z
    pts = np.stack([x, y, z], axis=1).astype(np.float64)
    uv = np.stack([xs.astype(np.int32), ys.astype(np.int32)], axis=1)
    return pts, uv


def _fit_plane_ransac_open3d_for_spine(
    points_xyz_m: np.ndarray,
    *,
    distance_threshold_m: float = 0.008,
    ransac_n: int = 3,
    num_iterations: int = 1200,
):
    """Open3Dで平面RANSACを実行し，plane_modelとinlier maskを返す．"""
    pts = np.asarray(points_xyz_m, dtype=np.float64).reshape(-1, 3)
    info = {
        "used": False,
        "reason": "not started",
        "point_count": int(pts.shape[0]),
        "distance_threshold_m": float(distance_threshold_m),
        "ransac_n": int(ransac_n),
        "num_iterations": int(num_iterations),
    }
    if pts.shape[0] < max(int(ransac_n), 3):
        info["reason"] = "too few points"
        return None, np.zeros((pts.shape[0],), dtype=bool), info

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        plane_model, inlier_indices = pcd.segment_plane(
            distance_threshold=float(distance_threshold_m),
            ransac_n=int(ransac_n),
            num_iterations=int(num_iterations),
        )
        inlier_mask = np.zeros((pts.shape[0],), dtype=bool)
        inlier_mask[np.asarray(inlier_indices, dtype=np.int64)] = True
        inlier_count = int(np.count_nonzero(inlier_mask))
        info.update({
            "used": True,
            "reason": "ok",
            "plane_model": [float(v) for v in plane_model],
            "inlier_count": int(inlier_count),
            "inlier_ratio": float(inlier_count / max(pts.shape[0], 1)),
        })
        return np.asarray(plane_model, dtype=np.float64), inlier_mask, info
    except Exception as e:
        info.update({"used": False, "reason": f"Open3D segment_plane failed: {e}"})
        return None, np.zeros((pts.shape[0],), dtype=bool), info


def _point_plane_distance_for_spine(points_xyz_m: np.ndarray, plane_model) -> np.ndarray:
    """plane_model=[a,b,c,d] に対する点群の符号なし距離[m]を返す．"""
    pts = np.asarray(points_xyz_m, dtype=np.float64).reshape(-1, 3)
    plane = np.asarray(plane_model, dtype=np.float64).reshape(4)
    normal = plane[:3]
    d = float(plane[3])
    n_norm = float(np.linalg.norm(normal))
    if n_norm < 1e-12:
        return np.full((pts.shape[0],), np.inf, dtype=np.float64)
    return np.abs(pts @ normal + d) / n_norm


def _project_points_to_plane_for_post_filter(points_xyz_m: np.ndarray, plane_model):
    """点群をRANSAC平面へ直交射影する．"""
    pts = np.asarray(points_xyz_m, dtype=np.float64).reshape(-1, 3)
    plane = np.asarray(plane_model, dtype=np.float64).reshape(4)
    n = plane[:3].astype(np.float64)
    d = float(plane[3])
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-12:
        return pts.copy(), None, None
    n = n / n_norm
    # 正規化後の平面式は n・x + d_norm = 0
    d_norm = d / n_norm
    signed = pts @ n + d_norm
    proj = pts - signed[:, None] * n[None, :]
    return proj, n, d_norm


def _intersect_pixel_ray_with_plane(u: float, v: float, intr, plane_model):
    """画素(u,v)のカメラレイと平面の交点を返す．失敗時None．"""
    plane = np.asarray(plane_model, dtype=np.float64).reshape(4)
    n = plane[:3].astype(np.float64)
    d = float(plane[3])
    ray = np.asarray([
        (float(u) - float(intr.ppx)) / float(intr.fx),
        (float(v) - float(intr.ppy)) / float(intr.fy),
        1.0,
    ], dtype=np.float64)
    denom = float(np.dot(n, ray))
    if abs(denom) < 1e-12:
        return None
    t = -d / denom
    if not np.isfinite(t) or t <= 0.0:
        return None
    return ray * t


def _extract_image_axis_center_for_plane_column_prune(refine_info: dict | None, mask01: np.ndarray):
    """
    OCR文字領域由来の画像上a方向と中心を返す．
    refine_info["axis"] があればそれを優先し，無ければ selected_ocr_polygon から求める．
    """
    info = refine_info if isinstance(refine_info, dict) else {}

    center = None
    axis = None
    source = "none"

    if info.get("axis") is not None:
        try:
            axis = np.asarray(info.get("axis"), dtype=np.float64).reshape(2)
            source = str(info.get("axis_source", "refine_info_axis"))
        except Exception:
            axis = None

    if info.get("ocr_center") is not None:
        try:
            center = np.asarray(info.get("ocr_center"), dtype=np.float64).reshape(2)
        except Exception:
            center = None

    selected = info.get("selected_ocr_polygon") if isinstance(info, dict) else None
    if (axis is None or center is None) and isinstance(selected, dict):
        poly = _poly_from_any(selected.get("poly"))
        if poly is not None:
            c, a_poly, _, _ = _polygon_center_and_axis(poly)
            if c is not None and center is None:
                center = np.asarray(c, dtype=np.float64).reshape(2)
            if a_poly is not None and axis is None:
                mask_axis = _mask_pca_axis(mask01)
                axis, kind = _choose_axis_consistent_with_mask(a_poly, mask_axis)
                source = f"selected_ocr_polygon_{kind}"

    if center is None:
        center = _mask_centroid(mask01)
        if center is not None:
            source += "+mask_centroid_center"

    if axis is None:
        axis = _mask_pca_axis(mask01)
        if axis is not None:
            source = "mask_pca_axis_fallback"

    if center is None or axis is None:
        return None, None, {"used": False, "reason": "axis or center unavailable", "source": source}

    axis = np.asarray(axis, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return None, None, {"used": False, "reason": "invalid image axis", "source": source}

    return axis / n, np.asarray(center, dtype=np.float64).reshape(2), {
        "used": True,
        "reason": "ok",
        "source": source,
        "axis_image_xy": [float(axis[0] / n), float(axis[1] / n)],
        "center_image_uv": [float(center[0]), float(center[1])],
    }


def _make_plane_frame_axis_from_ocr_or_pca(
    points_projected: np.ndarray,
    plane_model,
    intr,
    refine_info: dict | None,
    mask01: np.ndarray,
):
    """
    RANSAC平面上の列長フィルタ用座標系を作る．

    a軸: 原則としてOCR文字領域の第一主成分方向を，RANSAC平面へ持ち上げた方向
    b軸: a軸と平面法線に直交する方向
    フォールバック: RANSAC後点群の3D PCA第1主成分
    """
    pts_proj = np.asarray(points_projected, dtype=np.float64).reshape(-1, 3)
    info = {"used": False, "reason": "not started"}
    if pts_proj.shape[0] < 3:
        info["reason"] = "too few projected points"
        return None, None, None, info

    plane = np.asarray(plane_model, dtype=np.float64).reshape(4)
    normal = plane[:3].astype(np.float64)
    n_norm = float(np.linalg.norm(normal))
    if n_norm < 1e-12:
        info["reason"] = "invalid plane normal"
        return None, None, None, info
    normal = normal / n_norm

    centroid = np.mean(pts_proj, axis=0)
    axis_img, center_img, img_info = _extract_image_axis_center_for_plane_column_prune(refine_info, mask01)
    info["image_axis_info"] = img_info

    a_axis = None
    center3d = None
    axis_source = None

    if axis_img is not None and center_img is not None:
        # OCR画像上の中心と，中心から少し動かした点のレイをRANSAC平面に交差させる．
        delta_px = 80.0
        u0, v0 = float(center_img[0]), float(center_img[1])
        u1, v1 = u0 + float(axis_img[0]) * delta_px, v0 + float(axis_img[1]) * delta_px
        x0 = _intersect_pixel_ray_with_plane(u0, v0, intr, plane)
        x1 = _intersect_pixel_ray_with_plane(u1, v1, intr, plane)
        if x0 is not None and x1 is not None:
            v3 = x1 - x0
            # 念のため平面法線成分を落とす．
            v3 = v3 - float(np.dot(v3, normal)) * normal
            vn = float(np.linalg.norm(v3))
            if vn > 1e-9:
                a_axis = v3 / vn
                center3d = x0 - float(np.dot(x0 - centroid, normal)) * normal
                axis_source = "ocr_image_axis_intersected_with_ransac_plane"

    if a_axis is None:
        # フォールバック: RANSAC後点群の平面上PCA第1主成分
        centered = pts_proj - centroid
        try:
            cov = np.cov(centered.T)
            vals, vecs = np.linalg.eigh(cov)
            cand = vecs[:, int(np.argmax(vals))].astype(np.float64)
            cand = cand - float(np.dot(cand, normal)) * normal
            cn = float(np.linalg.norm(cand))
            if cn > 1e-9:
                a_axis = cand / cn
                center3d = centroid
                axis_source = "projected_points_3d_pca_fallback"
        except Exception as e:
            info["pca_error"] = str(e)

    if a_axis is None or center3d is None:
        info["reason"] = "failed to build plane a-axis"
        return None, None, None, info

    b_axis = np.cross(normal, a_axis)
    bn = float(np.linalg.norm(b_axis))
    if bn < 1e-9:
        info["reason"] = "failed to build plane b-axis"
        return None, None, None, info
    b_axis = b_axis / bn

    # 符号は任意だが，画像上のOCR方向とおおむね揃うようにaの向きだけ調整する．
    info.update({
        "used": True,
        "reason": "ok",
        "axis_source": axis_source,
        "plane_normal": [float(v) for v in normal],
        "a_axis": [float(v) for v in a_axis],
        "b_axis": [float(v) for v in b_axis],
        "center3d_m": [float(v) for v in center3d],
    })
    return a_axis, b_axis, center3d, info


def _save_post_ransac_plane_a95_debug(
    shot_dir: Path,
    color_np: np.ndarray,
    mask_before: np.ndarray,
    mask_after: np.ndarray,
    depth_before: np.ndarray,
    depth_after: np.ndarray,
    points_before: np.ndarray,
    uv_before: np.ndarray,
    keep_point_mask: np.ndarray,
    info: dict,
    stem: str,
):
    """RANSAC平面内95%列長フィルタのデバッグ画像・PLY・ログを保存する．"""
    debug_dir = Path(shot_dir) / "debug_post_ransac_plane_a95"
    debug_dir.mkdir(parents=True, exist_ok=True)

    mask_before = (np.asarray(mask_before) > 0).astype(np.uint8)
    mask_after = (np.asarray(mask_after) > 0).astype(np.uint8)
    valid_before = ((mask_before > 0) & (np.asarray(depth_before) > 0)).astype(np.uint8)
    valid_after = ((mask_after > 0) & (np.asarray(depth_after) > 0)).astype(np.uint8)
    removed = ((valid_before == 1) & (valid_after == 0)).astype(np.uint8)
    kept = ((valid_before == 1) & (valid_after == 1)).astype(np.uint8)

    cv2.imwrite(str(debug_dir / f"{stem}_valid_before.png"), valid_before * 255)
    cv2.imwrite(str(debug_dir / f"{stem}_valid_after.png"), valid_after * 255)
    cv2.imwrite(str(debug_dir / f"{stem}_removed.png"), removed * 255)

    overlay = color_np.copy()
    overlay[kept == 1] = (0, 255, 0)
    overlay[removed == 1] = (0, 0, 255)
    blended = cv2.addWeighted(color_np, 0.65, overlay, 0.35, 0)
    cv2.imwrite(str(debug_dir / f"{stem}_overlay_kept_removed.png"), blended)

    # 長さプロファイル
    records = info.get("column_records", []) if isinstance(info, dict) else []
    if records:
        graph_w, graph_h = 1100, 360
        margin_l, margin_r, margin_t, margin_b = 60, 25, 25, 50
        graph = np.full((graph_h, graph_w, 3), 255, dtype=np.uint8)
        plot_w = graph_w - margin_l - margin_r
        plot_h = graph_h - margin_t - margin_b
        max_len = max(float(info.get("max_length_m", 0.001)), 1e-6)
        th = float(info.get("length_threshold_m", 0.0))
        n = len(records)
        y_th = int(round(margin_t + plot_h * (1.0 - min(th / max_len, 1.0))))
        cv2.line(graph, (margin_l, y_th), (graph_w - margin_r, y_th), (0, 0, 255), 1)
        cv2.putText(graph, "95% threshold", (margin_l + 5, max(18, y_th - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1)
        for i, rec in enumerate(records):
            x0 = int(round(margin_l + plot_w * i / max(n, 1)))
            x1 = int(round(margin_l + plot_w * (i + 1) / max(n, 1)))
            length = float(rec.get("length_m", 0.0))
            bar_h = int(round(plot_h * min(length / max_len, 1.0)))
            y0 = margin_t + plot_h - bar_h
            y1 = margin_t + plot_h
            color = (180, 180, 180)
            if bool(rec.get("is_good", False)):
                color = (0, 180, 0)
            if bool(rec.get("is_selected", False)):
                color = (0, 120, 255)
            cv2.rectangle(graph, (x0, y0), (max(x0 + 1, x1 - 1), y1), color, -1)
        cv2.line(graph, (margin_l, margin_t), (margin_l, margin_t + plot_h), (0, 0, 0), 1)
        cv2.line(graph, (margin_l, margin_t + plot_h), (graph_w - margin_r, margin_t + plot_h), (0, 0, 0), 1)
        cv2.putText(graph, "b-bin on RANSAC plane", (graph_w // 2 - 90, graph_h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(graph, "a length [m]", (5, margin_t + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite(str(debug_dir / f"{stem}_plane_column_length_profile.png"), graph)

    # PLY: before / after / removed / compare
    pts = np.asarray(points_before, dtype=np.float64).reshape(-1, 3)
    uv = np.asarray(uv_before, dtype=np.int32).reshape(-1, 2)
    keep_point_mask = np.asarray(keep_point_mask, dtype=bool).reshape(-1)
    if pts.shape[0] == keep_point_mask.shape[0] and pts.shape[0] == uv.shape[0] and pts.shape[0] > 0:
        h, w = color_np.shape[:2]
        valid_uv = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        colors_rgb = np.full((pts.shape[0], 3), 180, dtype=np.uint8)
        bgr = color_np[uv[valid_uv, 1], uv[valid_uv, 0], :]
        colors_rgb[valid_uv] = bgr[:, ::-1].astype(np.uint8)
        colors_compare = colors_rgb.copy()
        colors_compare[~keep_point_mask] = np.asarray([0, 0, 255], dtype=np.uint8)  # RGB blue
        save_colored_ply_ascii(debug_dir / f"{stem}_before_rgb.ply", pts, colors_rgb)
        save_colored_ply_ascii(debug_dir / f"{stem}_after_rgb.ply", pts[keep_point_mask], colors_rgb[keep_point_mask])
        if np.any(~keep_point_mask):
            removed_colors = np.tile(np.asarray([[0, 0, 255]], dtype=np.uint8), (int(np.count_nonzero(~keep_point_mask)), 1))
            save_colored_ply_ascii(debug_dir / f"{stem}_removed_blue.ply", pts[~keep_point_mask], removed_colors)
        save_colored_ply_ascii(debug_dir / f"{stem}_compare_kept_rgb_removed_blue.ply", pts, colors_compare)

    save_json(debug_dir / f"{stem}_log.json", info)
    print(f"✔ Saved post-RANSAC plane-a95 debug files: {debug_dir}")



def _extract_previous_removed_side_sign_for_plane_a95(previous_side_info: dict | None):
    """
    前段の片側削除ログから，削った側を取り出す．

    戻り値:
      -1 : 画像座標系のt負側を削った
      +1 : 画像座標系のt正側を削った
      None : 不明
    """
    if not isinstance(previous_side_info, dict):
        return None, {
            "used": False,
            "reason": "previous_side_info is not dict",
        }

    candidates = []

    # 通常: column_info全体が渡される想定．
    if isinstance(previous_side_info.get("one_sided_post_final_column_prune"), dict):
        candidates.append(("one_sided_post_final_column_prune", previous_side_info["one_sided_post_final_column_prune"]))

    # refine_mask_by_spine_column_length_after_depth の結果そのもの，または column_info 全体．
    candidates.append(("previous_side_info_self", previous_side_info))

    # 念のためネストも見る．
    for key in ("column_info", "spine_column_length_after_depth", "candidate_seed_guard", "candidate_no_seed_guard"):
        if isinstance(previous_side_info.get(key), dict):
            candidates.append((key, previous_side_info[key]))

    for source_name, info in candidates:
        if not isinstance(info, dict):
            continue

        # apply_one_sided_short_column_prune_after_final のログ．
        if info.get("removed_side_sign") is not None:
            try:
                sign = int(info.get("removed_side_sign"))
                if sign < 0:
                    return -1, {
                        "used": True,
                        "reason": "ok",
                        "source": source_name,
                        "field": "removed_side_sign",
                        "removed_side_sign": -1,
                        "removed_side_name": info.get("removed_side_name"),
                        "removed_side_source": info.get("removed_side_source"),
                    }
                if sign > 0:
                    return +1, {
                        "used": True,
                        "reason": "ok",
                        "source": source_name,
                        "field": "removed_side_sign",
                        "removed_side_sign": +1,
                        "removed_side_name": info.get("removed_side_name"),
                        "removed_side_source": info.get("removed_side_source"),
                    }
            except Exception:
                pass

        # refine_mask_by_spine_column_length_after_depth のログ．
        # remove_side は画像t座標系の positive / negative．
        remove_side = info.get("remove_side")
        if remove_side == "negative":
            return -1, {
                "used": True,
                "reason": "ok",
                "source": source_name,
                "field": "remove_side",
                "remove_side": "negative",
            }
        if remove_side == "positive":
            return +1, {
                "used": True,
                "reason": "ok",
                "source": source_name,
                "field": "remove_side",
                "remove_side": "positive",
            }

    return None, {
        "used": False,
        "reason": "previous removed side is unavailable",
    }


def _map_image_t_side_to_plane_b_side(
    refine_info: dict | None,
    mask01: np.ndarray,
    intr,
    plane_model,
    b_axis: np.ndarray,
):
    """
    画像座標系のt正方向が，RANSAC平面上のb正方向と同じかを求める．

    画像側:
      axis   = 背表紙長手方向
      normal = [-axis_y, axis_x] = t正方向

    3D側:
      b_axis = RANSAC平面上の幅方向

    戻り値:
      +1 : 画像t正方向 ≒ 平面b正方向
      -1 : 画像t正方向 ≒ 平面b負方向
      None : 判定不能
    """
    info = {
        "used": False,
        "reason": "not started",
    }

    try:
        axis_img, center_img, img_info = _extract_image_axis_center_for_plane_column_prune(
            refine_info,
            mask01,
        )
        info["image_axis_info"] = img_info

        if axis_img is None or center_img is None:
            info["reason"] = "image axis or center unavailable"
            return None, info

        axis_img = np.asarray(axis_img, dtype=np.float64).reshape(2)
        n = float(np.linalg.norm(axis_img))
        if n < 1e-9:
            info["reason"] = "invalid image axis"
            return None, info
        axis_img = axis_img / n

        # 画像上のt正方向．
        normal_img = np.asarray([-axis_img[1], axis_img[0]], dtype=np.float64)

        u0, v0 = float(center_img[0]), float(center_img[1])
        delta_px = 80.0
        u1 = u0 + normal_img[0] * delta_px
        v1 = v0 + normal_img[1] * delta_px

        x0 = _intersect_pixel_ray_with_plane(u0, v0, intr, plane_model)
        x1 = _intersect_pixel_ray_with_plane(u1, v1, intr, plane_model)

        if x0 is None or x1 is None:
            info["reason"] = "failed to intersect image normal rays with plane"
            return None, info

        v3 = np.asarray(x1, dtype=np.float64) - np.asarray(x0, dtype=np.float64)
        b_axis = np.asarray(b_axis, dtype=np.float64).reshape(3)
        bn = float(np.linalg.norm(b_axis))
        if bn < 1e-9:
            info["reason"] = "invalid b_axis"
            return None, info
        b_axis = b_axis / bn

        dot = float(np.dot(v3, b_axis))
        if abs(dot) < 1e-9:
            info["reason"] = "image t direction is unstable to b_axis"
            return None, info

        sign = +1 if dot > 0 else -1
        info.update({
            "used": True,
            "reason": "ok",
            "image_t_positive_to_plane_b_sign": int(sign),
            "dot_image_t_with_b_axis": float(dot),
            "normal_image_xy": [float(normal_img[0]), float(normal_img[1])],
        })
        return sign, info

    except Exception as e:
        info["reason"] = f"error: {e}"
        return None, info

def apply_post_ransac_plane_a95_column_prune(
    mask01: np.ndarray,
    depth_masked: np.ndarray,
    intr,
    depth_scale: float,
    plane_model,
    refine_info: dict | None,
    image_shape: tuple[int, int],
    *,
    shot_dir: str | Path | None = None,
    color_np: np.ndarray | None = None,
    stem: str = "post_ransac_plane_a95",
    b_bin_size_m: float = 0.004,
    length_ratio: float = 0.90,
    span_percentiles: tuple[float, float] = (2.0, 98.0),
    a_bin_size_m: float = 0.003,
    a_gap_allow_m: float = 0.016,
    min_occupancy_density: float = 0.12,
    min_point_count_ratio_to_core: float = 0.06,
    protect_near_threshold_ratio: float = 0.95,
    min_points_per_bin: int = 6,
    min_positive_bins: int = 2,
    bridge_gap_bins: int = 1,
    min_valid_keep_ratio: float = 0.35,
    max_removed_ratio: float = 0.70,
    small_component_area_ratio: float = 0.0003,
    previous_side_info: dict | None = None,
    one_sided_from_previous: bool = True,
    skip_if_previous_side_unknown: bool = True,
    skip_if_ransac_removed_ratio_over: float = 0.22,
    return_info: bool = False,
):
    """
    RANSAC後・calculate_yaw前に行う最終列長フィルタの安全版。

    主な修正:
      - a方向長さはpercentile spanではなく、m単位continuous_runで計算する。
      - 点が疎な端binはgood扱いしない。
      - 前段で削った側だけを削る。
      - thresholdギリギリのbinは背表紙端の可能性が高いため保護する。
      - RANSAC自体が大きく削った場合は追加削除をskipする。
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    mask_before = (np.asarray(mask01) > 0).astype(np.uint8)
    depth_before = np.asarray(depth_masked).copy()
    valid_before_count = int(np.count_nonzero((mask_before > 0) & (depth_before > 0)))
    info = {
        "used": False,
        "reason": "not started",
        "algorithm": "post_ransac_plane_metric_continuous_run_density_guard_one_sided",
        "metric_continuous_run_fix": True,
        "b_bin_size_m": float(b_bin_size_m),
        "a_bin_size_m": float(a_bin_size_m),
        "a_gap_allow_m": float(a_gap_allow_m),
        "length_ratio": float(length_ratio),
        "span_percentiles": [float(span_percentiles[0]), float(span_percentiles[1])],
        "min_occupancy_density": float(min_occupancy_density),
        "min_point_count_ratio_to_core": float(min_point_count_ratio_to_core),
        "protect_near_threshold_ratio": float(protect_near_threshold_ratio),
        "min_points_per_bin": int(min_points_per_bin),
        "min_positive_bins": int(min_positive_bins),
        "bridge_gap_bins": int(bridge_gap_bins),
        "min_valid_keep_ratio": float(min_valid_keep_ratio),
        "max_removed_ratio": float(max_removed_ratio),
        "one_sided_from_previous": bool(one_sided_from_previous),
        "skip_if_previous_side_unknown": bool(skip_if_previous_side_unknown),
        "skip_if_ransac_removed_ratio_over": float(skip_if_ransac_removed_ratio_over),
        "valid_count_before": int(valid_before_count),
    }

    if plane_model is None:
        info["reason"] = "plane_model is None; skipped"
        return (mask_before, depth_before, info) if return_info else (mask_before, depth_before)

    if isinstance(previous_side_info, dict):
        rinfo_prev = previous_side_info.get("ransac_spine_plane")
        if isinstance(rinfo_prev, dict):
            try:
                rr = float(rinfo_prev.get("removed_ratio", 0.0))
                info["pre_post_ransac_removed_ratio"] = rr
                if rr > float(skip_if_ransac_removed_ratio_over):
                    info["reason"] = f"skipped because RANSAC already removed too much: removed_ratio={rr:.3f}"
                    info["valid_count_after"] = int(valid_before_count)
                    info["valid_keep_ratio"] = 1.0
                    info["removed_count"] = 0
                    info["removed_ratio"] = 0.0
                    return (mask_before, depth_before, info) if return_info else (mask_before, depth_before)
            except Exception:
                pass

    pts_all, uv_all = _mask_depth_to_points_uv_for_plane_filter(mask_before, depth_before, intr, depth_scale)
    if pts_all.shape[0] < max(30, int(min_points_per_bin) * 2):
        info["reason"] = "too few valid points"
        return (mask_before, depth_before, info) if return_info else (mask_before, depth_before)

    pts_proj, plane_normal, plane_d_norm = _project_points_to_plane_for_post_filter(pts_all, plane_model)
    if plane_normal is None:
        info["reason"] = "invalid plane_model"
        return (mask_before, depth_before, info) if return_info else (mask_before, depth_before)

    a_axis, b_axis, center3d, frame_info = _make_plane_frame_axis_from_ocr_or_pca(
        points_projected=pts_proj,
        plane_model=plane_model,
        intr=intr,
        refine_info=refine_info,
        mask01=mask_before,
    )
    info["plane_frame"] = frame_info
    if a_axis is None or b_axis is None or center3d is None:
        info["reason"] = f"failed to build plane frame: {frame_info.get('reason')}"
        return (mask_before, depth_before, info) if return_info else (mask_before, depth_before)

    rel = pts_proj - np.asarray(center3d, dtype=np.float64).reshape(1, 3)
    a_coords = rel @ a_axis
    b_coords = rel @ b_axis

    b_bin_size_m = max(float(b_bin_size_m), 1e-4)
    a_bin_size_m = max(float(a_bin_size_m), 1e-5)
    a_gap_allow_bins = int(round(float(a_gap_allow_m) / a_bin_size_m))
    b_min = float(np.min(b_coords))
    b_max = float(np.max(b_coords))
    a_min_all = float(np.min(a_coords))
    n_bins = int(np.floor((b_max - b_min) / b_bin_size_m)) + 1
    if n_bins < 1:
        info["reason"] = "invalid b bin count"
        return (mask_before, depth_before, info) if return_info else (mask_before, depth_before)

    b_bins = np.floor((b_coords - b_min) / b_bin_size_m).astype(np.int32)
    b_bins = np.clip(b_bins, 0, n_bins - 1)

    p_low = float(np.clip(float(span_percentiles[0]), 0.0, 49.0))
    p_high = float(np.clip(float(span_percentiles[1]), 51.0, 100.0))
    if p_high <= p_low:
        p_low, p_high = 2.0, 98.0

    counts_all = np.bincount(b_bins, minlength=n_bins).astype(np.int64)
    core_count_ref = float(np.percentile(counts_all[counts_all > 0], 70.0)) if np.any(counts_all > 0) else 1.0
    min_points_density_guard = max(int(min_points_per_bin), int(round(core_count_ref * float(min_point_count_ratio_to_core))))

    records = []
    lengths = np.zeros((n_bins,), dtype=np.float64)
    density_ok = np.zeros((n_bins,), dtype=bool)
    count_ok = np.zeros((n_bins,), dtype=bool)
    for b in range(n_bins):
        idx = np.where(b_bins == b)[0]
        count = int(idx.size)
        rec = {
            "b_bin": int(b),
            "b_min_m": float(b_min + b * b_bin_size_m),
            "b_max_m": float(b_min + (b + 1) * b_bin_size_m),
            "point_count": count,
            "length_m": 0.0,
            "run_length_m": 0.0,
            "span_length_m": 0.0,
            "a_min_m": None,
            "a_max_m": None,
            "is_good": False,
            "is_good_by_length": False,
            "is_good_by_density": False,
            "is_good_by_count": False,
            "is_near_threshold_protected": False,
            "is_selected": False,
            "is_deleted_by_one_sided_previous_side": False,
        }
        if count >= int(min_points_per_bin):
            aa = a_coords[idx]
            run = _longest_occupied_metric_segment(
                aa,
                origin=a_min_all,
                bin_size=a_bin_size_m,
                gap_allow_bins=a_gap_allow_bins,
            )
            try:
                a0_span, a1_span = np.percentile(aa, [p_low, p_high])
            except Exception:
                a0_span, a1_span = float(np.min(aa)), float(np.max(aa))
            span_length = float(max(0.0, float(a1_span) - float(a0_span)))
            if run is not None:
                run_length = float(run.get("length", 0.0))
                occ_density = float(run.get("occupancy_density", 0.0))
                point_count_in_run = int(run.get("point_count_in_run", 0))
                rec.update({
                    "length_m": float(run_length),
                    "run_length_m": float(run_length),
                    "span_length_m": float(span_length),
                    "a_min_m": float(run.get("value_min")),
                    "a_max_m": float(run.get("value_max")),
                    "a_raw_min_m": float(np.min(aa)),
                    "a_raw_max_m": float(np.max(aa)),
                    "a_span_min_m": float(a0_span),
                    "a_span_max_m": float(a1_span),
                    "occupied_bin_count": int(run.get("occupied_bin_count", 0)),
                    "span_bin_count": int(run.get("span_bin_count", 0)),
                    "occupancy_density": float(occ_density),
                    "point_count_in_run": int(point_count_in_run),
                    "run_point_ratio": float(point_count_in_run / max(count, 1)),
                })
                lengths[b] = run_length
                density_ok[b] = occ_density >= float(min_occupancy_density)
                count_ok[b] = count >= int(min_points_density_guard)
            else:
                rec.update({
                    "span_length_m": float(span_length),
                    "a_span_min_m": float(a0_span),
                    "a_span_max_m": float(a1_span),
                })
        records.append(rec)

    positive = lengths[lengths > 0.0]
    if positive.size < int(min_positive_bins):
        info.update({"reason": "too few positive length bins", "n_bins": int(n_bins), "positive_bins": int(positive.size)})
        return (mask_before, depth_before, info) if return_info else (mask_before, depth_before)

    max_length = float(np.max(positive))
    threshold = float(max_length * float(length_ratio))
    candidate_good_bins = set()
    for rec in records:
        b = int(rec["b_bin"])
        by_length = bool(float(rec.get("length_m", 0.0)) >= threshold)
        by_density = bool(density_ok[b])
        by_count = bool(count_ok[b])
        rec["is_good_by_length"] = by_length
        rec["is_good_by_density"] = by_density
        rec["is_good_by_count"] = by_count
        if by_length and by_density and by_count:
            candidate_good_bins.add(b)

    if int(bridge_gap_bins) > 0 and candidate_good_bins:
        all_good = sorted(candidate_good_bins)
        for left, right in zip(all_good[:-1], all_good[1:]):
            gap = int(right - left - 1)
            if 0 < gap <= int(bridge_gap_bins):
                for b in range(left + 1, right):
                    candidate_good_bins.add(int(b))

    if not candidate_good_bins:
        info.update({
            "reason": "no bins above metric continuous-run threshold",
            "max_length_m": float(max_length),
            "length_threshold_m": float(threshold),
        })
        return (mask_before, depth_before, info) if return_info else (mask_before, depth_before)

    candidate_bad_bins = set(int(b) for b in range(n_bins) if int(b) not in candidate_good_bins)
    one_sided_info = {
        "used": False,
        "reason": "not evaluated",
        "candidate_good_bins_before_one_sided": [int(v) for v in sorted(candidate_good_bins)],
        "candidate_bad_bins_before_one_sided": [int(v) for v in sorted(candidate_bad_bins)],
    }

    if bool(one_sided_from_previous):
        prev_side_sign_t, prev_side_log = _extract_previous_removed_side_sign_for_plane_a95(previous_side_info)
        one_sided_info["previous_side_info"] = prev_side_log

        if prev_side_sign_t is None:
            one_sided_info.update({"used": False, "reason": "previous removed side is unknown"})
            if bool(skip_if_previous_side_unknown):
                info.update({
                    "used": False,
                    "reason": "skipped post-RANSAC a95 because previous removed side is unknown",
                    "one_sided_previous_side_filter": one_sided_info,
                    "plane_model": [float(v) for v in np.asarray(plane_model, dtype=np.float64).reshape(4)],
                    "plane_normal": [float(v) for v in plane_normal],
                    "plane_d_norm": float(plane_d_norm),
                    "a_axis": [float(v) for v in a_axis],
                    "b_axis": [float(v) for v in b_axis],
                    "center3d_m": [float(v) for v in center3d],
                    "a_min_m": float(np.min(a_coords)),
                    "a_max_m": float(np.max(a_coords)),
                    "b_min_m": float(b_min),
                    "b_max_m": float(b_max),
                    "n_bins": int(n_bins),
                    "max_length_m": float(max_length),
                    "length_threshold_m": float(threshold),
                    "candidate_good_bins_before_one_sided": [int(v) for v in sorted(candidate_good_bins)],
                    "good_bins": [int(v) for v in sorted(candidate_good_bins)],
                    "valid_count_after": int(valid_before_count),
                    "valid_keep_ratio": 1.0,
                    "removed_count": 0,
                    "removed_ratio": 0.0,
                    "column_records": records,
                })
                return (mask_before, depth_before, info) if return_info else (mask_before, depth_before)
            keep_point = np.asarray([int(b) in candidate_good_bins for b in b_bins], dtype=bool)
            one_sided_info["fallback_old_behavior"] = True
        else:
            image_t_to_b_sign, mapping_info = _map_image_t_side_to_plane_b_side(
                refine_info=refine_info,
                mask01=mask_before,
                intr=intr,
                plane_model=plane_model,
                b_axis=b_axis,
            )
            one_sided_info["image_t_to_plane_b_mapping"] = mapping_info
            if image_t_to_b_sign is None:
                image_t_to_b_sign = +1
                one_sided_info["mapping_fallback_used"] = True
            else:
                one_sided_info["mapping_fallback_used"] = False

            target_b_sign = int(prev_side_sign_t) * int(image_t_to_b_sign)
            b_bin_centers = np.asarray([float(b_min + (b + 0.5) * b_bin_size_m) for b in range(n_bins)], dtype=np.float64)
            b_reference = float(np.median(b_coords))
            if target_b_sign < 0:
                target_side_bins = set(int(b) for b in range(n_bins) if float(b_bin_centers[b]) < b_reference)
                target_side_name = "negative_b"
            else:
                target_side_bins = set(int(b) for b in range(n_bins) if float(b_bin_centers[b]) > b_reference)
                target_side_name = "positive_b"

            near_threshold_bins = set()
            for rec in records:
                b = int(rec["b_bin"])
                if float(rec.get("length_m", 0.0)) >= threshold * float(protect_near_threshold_ratio):
                    near_threshold_bins.add(b)
                    rec["is_near_threshold_protected"] = True

            delete_bins = (candidate_bad_bins & target_side_bins) - near_threshold_bins
            keep_point = np.asarray([int(b) not in delete_bins for b in b_bins], dtype=bool)
            final_selected_bins = set(np.unique(b_bins[keep_point]).astype(int).tolist())
            one_sided_info.update({
                "used": True,
                "reason": "ok",
                "previous_removed_side_sign_t": int(prev_side_sign_t),
                "image_t_positive_to_plane_b_sign": int(image_t_to_b_sign),
                "target_b_sign": int(target_b_sign),
                "target_side_name": target_side_name,
                "b_reference_m": float(b_reference),
                "target_side_bins": [int(v) for v in sorted(target_side_bins)],
                "near_threshold_protected_bins": [int(v) for v in sorted(near_threshold_bins)],
                "delete_bins": [int(v) for v in sorted(delete_bins)],
                "final_selected_bins_after_one_sided": [int(v) for v in sorted(final_selected_bins)],
            })
    else:
        keep_point = np.asarray([int(b) in candidate_good_bins for b in b_bins], dtype=bool)
        one_sided_info.update({"used": False, "reason": "one_sided_from_previous is False; old behavior"})

    mask_after = np.zeros((H, W), dtype=np.uint8)
    if np.any(keep_point):
        uv_keep = uv_all[keep_point]
        u = uv_keep[:, 0]
        v = uv_keep[:, 1]
        valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        mask_after[v[valid_uv], u[valid_uv]] = 1

    mask_after, component_info = _filter_small_components(mask_after, min_area_ratio=float(small_component_area_ratio))
    depth_after = depth_before.copy()
    depth_after[mask_after == 0] = 0

    valid_after_count = int(np.count_nonzero((mask_after > 0) & (depth_after > 0)))
    valid_keep_ratio = float(valid_after_count / max(valid_before_count, 1))
    removed_count = int(valid_before_count - valid_after_count)
    removed_ratio = float(removed_count / max(valid_before_count, 1))

    selected_bins = set(np.unique(b_bins[keep_point]).astype(int).tolist())
    delete_bins_for_log = set(one_sided_info.get("delete_bins", [])) if isinstance(one_sided_info, dict) else set()
    for rec in records:
        b = int(rec["b_bin"])
        rec["is_good"] = bool(b in candidate_good_bins)
        rec["is_selected"] = bool(b in selected_bins)
        rec["is_deleted_by_one_sided_previous_side"] = bool(b in delete_bins_for_log)

    info.update({
        "used": True,
        "reason": "ok",
        "plane_model": [float(v) for v in np.asarray(plane_model, dtype=np.float64).reshape(4)],
        "plane_normal": [float(v) for v in plane_normal],
        "plane_d_norm": float(plane_d_norm),
        "a_axis": [float(v) for v in a_axis],
        "b_axis": [float(v) for v in b_axis],
        "center3d_m": [float(v) for v in center3d],
        "a_min_m": float(np.min(a_coords)),
        "a_max_m": float(np.max(a_coords)),
        "b_min_m": float(b_min),
        "b_max_m": float(b_max),
        "n_bins": int(n_bins),
        "core_count_reference": float(core_count_ref),
        "min_points_density_guard": int(min_points_density_guard),
        "max_length_m": float(max_length),
        "length_threshold_m": float(threshold),
        "candidate_good_bins_before_one_sided": [int(v) for v in sorted(candidate_good_bins)],
        "good_bins": [int(v) for v in sorted(selected_bins)],
        "one_sided_previous_side_filter": one_sided_info,
        "valid_count_after": int(valid_after_count),
        "valid_keep_ratio": float(valid_keep_ratio),
        "removed_count": int(removed_count),
        "removed_ratio": float(removed_ratio),
        "component_filter": component_info,
        "column_records": records,
    })

    if valid_keep_ratio < float(min_valid_keep_ratio) or removed_ratio > float(max_removed_ratio):
        info.update({
            "used": False,
            "guard_reverted": True,
            "reason": f"guard reverted: valid_keep_ratio={valid_keep_ratio:.3f}, removed_ratio={removed_ratio:.3f}",
            "candidate_valid_count_after": int(valid_after_count),
            "candidate_valid_keep_ratio": float(valid_keep_ratio),
            "candidate_removed_ratio": float(removed_ratio),
            "valid_count_after": int(valid_before_count),
            "valid_keep_ratio": 1.0,
            "removed_count": 0,
            "removed_ratio": 0.0,
        })
        mask_after = mask_before
        depth_after = depth_before
        keep_point = np.ones((pts_all.shape[0],), dtype=bool)

    if shot_dir is not None and color_np is not None:
        try:
            _save_post_ransac_plane_a95_debug(
                shot_dir=Path(shot_dir),
                color_np=color_np,
                mask_before=mask_before,
                mask_after=mask_after,
                depth_before=depth_before,
                depth_after=depth_after,
                points_before=pts_all,
                uv_before=uv_all,
                keep_point_mask=keep_point,
                info=info,
                stem=stem,
            )
        except Exception as e:
            info["debug_save_error"] = str(e)
            traceback.print_exc()

    return (mask_after, depth_after, info) if return_info else (mask_after, depth_after)


def _make_ocr_extended_protection_for_uv_points(
    uv: np.ndarray,
    refine_info: dict | None,
    mask01: np.ndarray,
    *,
    margin_px: float = 2.0,
    min_width_px: float = 12.0,
):
    """
    RANSAC平面フィルタ用の安全保護帯を作る。

    選択OCR polygonのnormal方向幅を、背表紙方向axisへ延長した帯として扱い、
    その帯に入るuv点をRANSAC外れ値削除から救済する。
    """
    info = {
        "used": False,
        "reason": "not started",
        "margin_px": float(margin_px),
        "min_width_px": float(min_width_px),
    }
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    protect = np.zeros((uv.shape[0],), dtype=bool)
    if uv.shape[0] == 0:
        info["reason"] = "empty uv"
        return protect, info

    rinfo = refine_info if isinstance(refine_info, dict) else {}
    axis_img, center_img, axis_info = _extract_image_axis_center_for_plane_column_prune(rinfo, mask01)
    info["image_axis_info"] = axis_info
    if axis_img is None or center_img is None:
        info["reason"] = "image axis or center unavailable"
        return protect, info

    axis_img = np.asarray(axis_img, dtype=np.float64).reshape(2)
    axis_norm = float(np.linalg.norm(axis_img))
    if axis_norm < 1e-9:
        info["reason"] = "invalid axis"
        return protect, info
    axis_img = axis_img / axis_norm
    normal_img = np.asarray([-axis_img[1], axis_img[0]], dtype=np.float64)
    center_img = np.asarray(center_img, dtype=np.float64).reshape(2)

    selected = rinfo.get("selected_ocr_polygon") if isinstance(rinfo, dict) else None
    selected_poly = None
    if isinstance(selected, dict):
        selected_poly = _poly_from_any(selected.get("poly"))

    if selected_poly is None:
        info["reason"] = "selected OCR polygon unavailable"
        return protect, info

    poly_pts = np.asarray(selected_poly, dtype=np.float64).reshape(-1, 2)
    if poly_pts.shape[0] < 4:
        info["reason"] = "invalid selected OCR polygon"
        return protect, info

    poly_t = (poly_pts - center_img) @ normal_img
    t0 = float(np.min(poly_t) - float(margin_px))
    t1 = float(np.max(poly_t) + float(margin_px))
    width = float(t1 - t0)

    # OCR polygonが極端に細い/不安定な場合も、最低幅だけ確保する。
    if width < float(min_width_px):
        mid = 0.5 * (t0 + t1)
        half = 0.5 * float(min_width_px)
        t0 = mid - half
        t1 = mid + half
        width = float(t1 - t0)

    t_uv = (uv - center_img.reshape(1, 2)) @ normal_img
    protect = (t_uv >= t0) & (t_uv <= t1)

    info.update({
        "used": bool(np.count_nonzero(protect) > 0),
        "reason": "ok" if np.count_nonzero(protect) > 0 else "no uv points in OCR extended band",
        "axis_image_xy": [float(axis_img[0]), float(axis_img[1])],
        "normal_image_xy": [float(normal_img[0]), float(normal_img[1])],
        "center_image_uv": [float(center_img[0]), float(center_img[1])],
        "selected_ocr_text": selected.get("text") if isinstance(selected, dict) else None,
        "selected_ocr_transform_mode": selected.get("transform_mode") if isinstance(selected, dict) else None,
        "selected_ocr_poly": poly_pts.astype(float).tolist(),
        "protection_t_min_px": float(t0),
        "protection_t_max_px": float(t1),
        "protection_width_px": float(width),
        "protected_point_count": int(np.count_nonzero(protect)),
        "total_point_count": int(uv.shape[0]),
    })
    return protect, info


def save_final_png_and_reconstruction_bundle(
    shot_dir: Path,
    color_np: np.ndarray,
    mask01: np.ndarray,
    depth_masked: np.ndarray,
    intr,
    depth_scale: float,
    stem: str,
):
    """
    calculate_yaw直前の最終mask/depthをfinal.pngとして保存し，
    3D復元に必要なRGB, mask, depth, intrinsics, xyz, uv, rgbをnpz/plyで保存する。
    """
    shot_dir = Path(shot_dir)
    shot_dir.mkdir(parents=True, exist_ok=True)
    color_np = np.asarray(color_np)
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    depth_masked = np.asarray(depth_masked).copy()
    valid = ((mask01 > 0) & (depth_masked > 0)).astype(np.uint8)

    overlay = color_np.copy()
    overlay[valid == 1] = (0, 255, 0)
    blended = cv2.addWeighted(color_np, 0.65, overlay, 0.35, 0)
    contours, _ = cv2.findContours(valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(blended, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(shot_dir / "final.png"), blended)
    cv2.imwrite(str(shot_dir / "final_mask.png"), valid * 255)
    cv2.imwrite(str(shot_dir / "final_rgb.png"), color_np)
    np.save(shot_dir / "final_depth_masked.npy", depth_masked)

    pts, uv = _mask_depth_to_points_uv_for_plane_filter(valid, depth_masked, intr, depth_scale)
    h, w = color_np.shape[:2]
    colors_rgb = np.full((pts.shape[0], 3), 180, dtype=np.uint8)
    if uv.shape[0] > 0:
        valid_uv = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        colors_rgb[valid_uv] = color_np[uv[valid_uv, 1], uv[valid_uv, 0], :][:, ::-1].astype(np.uint8)

    np.save(shot_dir / "final_points_xyz_m.npy", pts)
    np.save(shot_dir / "final_points_uv.npy", uv)
    np.save(shot_dir / "final_points_rgb.npy", colors_rgb)
    save_colored_ply_ascii(shot_dir / "final_pointcloud_rgb.ply", pts, colors_rgb)

    intr_arr = np.asarray([
        float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy),
        float(getattr(intr, "width", w)), float(getattr(intr, "height", h)), float(depth_scale)
    ], dtype=np.float64)
    np.savez_compressed(
        shot_dir / "final_reconstruction_bundle.npz",
        rgb_bgr=color_np,
        mask_valid=valid.astype(np.uint8),
        depth_masked_z16=depth_masked,
        points_xyz_m=pts,
        points_uv=uv,
        colors_rgb=colors_rgb,
        intrinsics=intr_arr,
        intr_fx=float(intr.fx),
        intr_fy=float(intr.fy),
        intr_ppx=float(intr.ppx),
        intr_ppy=float(intr.ppy),
        intr_width=int(getattr(intr, "width", w)),
        intr_height=int(getattr(intr, "height", h)),
        depth_scale=float(depth_scale),
    )

    log = {
        "stem": str(stem),
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "point_count": int(pts.shape[0]),
        "files": {
            "final_png": str(shot_dir / "final.png"),
            "final_mask_png": str(shot_dir / "final_mask.png"),
            "final_rgb_png": str(shot_dir / "final_rgb.png"),
            "final_depth_masked_npy": str(shot_dir / "final_depth_masked.npy"),
            "final_points_xyz_m_npy": str(shot_dir / "final_points_xyz_m.npy"),
            "final_points_uv_npy": str(shot_dir / "final_points_uv.npy"),
            "final_points_rgb_npy": str(shot_dir / "final_points_rgb.npy"),
            "final_pointcloud_rgb_ply": str(shot_dir / "final_pointcloud_rgb.ply"),
            "final_reconstruction_bundle_npz": str(shot_dir / "final_reconstruction_bundle.npz"),
        },
        "intrinsics": {
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "ppx": float(intr.ppx),
            "ppy": float(intr.ppy),
            "width": int(getattr(intr, "width", w)),
            "height": int(getattr(intr, "height", h)),
            "depth_scale": float(depth_scale),
        },
    }
    save_json(shot_dir / f"{stem}_final_reconstruction_log.json", log)
    print(f"✔ Saved final.png and reconstruction bundle: {shot_dir}")
    return log

def save_ransac_spine_plane_debug(
    shot_dir: Path,
    color_np: np.ndarray,
    mask_before: np.ndarray,
    mask_after: np.ndarray,
    depth_before: np.ndarray,
    depth_after: np.ndarray,
    info: dict,
    stem: str,
):
    """RANSAC平面フィルタ前後の2Dデバッグ画像を保存する．"""
    debug_dir = Path(shot_dir) / "debug_ransac_spine_plane"
    debug_dir.mkdir(parents=True, exist_ok=True)
    mask_before = (np.asarray(mask_before) > 0).astype(np.uint8)
    mask_after = (np.asarray(mask_after) > 0).astype(np.uint8)
    valid_before = ((mask_before > 0) & (np.asarray(depth_before) > 0)).astype(np.uint8)
    valid_after = ((mask_after > 0) & (np.asarray(depth_after) > 0)).astype(np.uint8)
    removed = ((valid_before == 1) & (valid_after == 0)).astype(np.uint8)
    kept = ((valid_before == 1) & (valid_after == 1)).astype(np.uint8)
    cv2.imwrite(str(debug_dir / f"{stem}_valid_before.png"), valid_before * 255)
    cv2.imwrite(str(debug_dir / f"{stem}_valid_after.png"), valid_after * 255)
    cv2.imwrite(str(debug_dir / f"{stem}_removed.png"), removed * 255)
    overlay = color_np.copy()
    overlay[kept == 1] = (0, 255, 0)
    overlay[removed == 1] = (255, 0, 0)  # BGR blue
    blended = cv2.addWeighted(color_np, 0.65, overlay, 0.35, 0)
    cv2.imwrite(str(debug_dir / f"{stem}_overlay_kept_removed.png"), blended)
    save_json(debug_dir / f"{stem}_log.json", info)


def refine_mask_by_ransac_spine_plane_after_depth(
    mask01: np.ndarray,
    depth_masked: np.ndarray,
    intr,
    depth_scale: float,
    refine_info: dict | None,
    image_shape: tuple[int, int],
    *,
    shot_dir: str | Path | None = None,
    color_np: np.ndarray | None = None,
    stem: str = "ransac_spine_plane",
    distance_threshold_m: float = 0.008,
    ransac_n: int = 3,
    num_iterations: int = 1200,
    min_total_points: int = 120,
    min_reference_points: int = 80,
    min_reference_inlier_ratio: float = 0.45,
    min_keep_ratio: float = 0.45,
    max_removed_ratio: float = 0.25,
    max_removed_ratio_after_column_refine: float = 0.20,
    max_distance_p90_over_threshold: float = 1.80,
    min_reference_point_ratio: float = 0.03,
    protect_ocr_extended_band: bool = True,
    ocr_extended_protection_margin_px: float = 2.0,
    small_component_area_ratio: float = 0.0003,
    prefer_ocr_reference: bool = True,
    allow_all_points_fallback: bool = True,
    return_info: bool = False,
):
    """
    最終マスク・Depthに対して，RANSAC平面推定により背表紙平面だけを残す．
    戻り値のinfo["plane_model"]を後段のRANSAC平面内95%列長フィルタで使う．
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    mask_before = (np.asarray(mask01) > 0).astype(np.uint8)
    depth_before = np.asarray(depth_masked).copy()
    points_all, uv_all = _mask_depth_to_points_uv_for_plane_filter(mask_before, depth_before, intr, depth_scale)
    valid_count_before = int(points_all.shape[0])
    info = {
        "used": False,
        "reason": "",
        "algorithm": "ransac_spine_plane_after_depth",
        "distance_threshold_m": float(distance_threshold_m),
        "ransac_n": int(ransac_n),
        "num_iterations": int(num_iterations),
        "min_total_points": int(min_total_points),
        "min_reference_points": int(min_reference_points),
        "min_reference_inlier_ratio": float(min_reference_inlier_ratio),
        "min_keep_ratio": float(min_keep_ratio),
        "max_removed_ratio": float(max_removed_ratio),
        "max_removed_ratio_after_column_refine": float(max_removed_ratio_after_column_refine),
        "max_distance_p90_over_threshold": float(max_distance_p90_over_threshold),
        "min_reference_point_ratio": float(min_reference_point_ratio),
        "protect_ocr_extended_band": bool(protect_ocr_extended_band),
        "ocr_extended_protection_margin_px": float(ocr_extended_protection_margin_px),
        "valid_count_before": int(valid_count_before),
    }
    if valid_count_before < int(min_total_points):
        info["reason"] = "too few total points"
        return (mask_before, depth_before, info) if return_info else (mask_before, depth_before)

    plane_model = None
    plane_source = None
    fit_info = None

    if bool(prefer_ocr_reference):
        try:
            reference_mask, reference_info = _make_depth_reference_mask_from_ocr_info(
                refine_info=refine_info,
                image_shape=image_shape,
                selected_mask01=mask_before,
                min_intersection_px=30,
            )
        except Exception as e:
            reference_mask, reference_info = None, {"used": False, "reason": str(e)}
        info["reference_mask_info"] = reference_info
        if reference_mask is not None:
            ref_points, _ = _mask_depth_to_points_uv_for_plane_filter(reference_mask, depth_before, intr, depth_scale)
            info["reference_point_count"] = int(ref_points.shape[0])
            if ref_points.shape[0] >= int(min_reference_points):
                ref_plane, ref_inliers, ref_fit_info = _fit_plane_ransac_open3d_for_spine(
                    ref_points,
                    distance_threshold_m=float(distance_threshold_m),
                    ransac_n=int(ransac_n),
                    num_iterations=int(num_iterations),
                )
                info["reference_fit"] = ref_fit_info
                if ref_fit_info.get("used", False) and float(ref_fit_info.get("inlier_ratio", 0.0)) >= float(min_reference_inlier_ratio):
                    plane_model = ref_plane
                    plane_source = "ocr_reference_plane"
                    fit_info = ref_fit_info

    if plane_model is None and bool(allow_all_points_fallback):
        all_plane, all_inliers, all_fit_info = _fit_plane_ransac_open3d_for_spine(
            points_all,
            distance_threshold_m=float(distance_threshold_m),
            ransac_n=int(ransac_n),
            num_iterations=int(num_iterations),
        )
        info["all_points_fit"] = all_fit_info
        if all_fit_info.get("used", False):
            plane_model = all_plane
            plane_source = "all_points_largest_plane"
            fit_info = all_fit_info

    if plane_model is None:
        info["reason"] = "RANSAC failed"
        return (mask_before, depth_before, info) if return_info else (mask_before, depth_before)

    distances = _point_plane_distance_for_spine(points_all, plane_model)
    keep_points_by_ransac = distances <= float(distance_threshold_m)

    ocr_protect_points = np.zeros_like(keep_points_by_ransac, dtype=bool)
    ocr_protect_info = {"used": False, "reason": "disabled"}
    if bool(protect_ocr_extended_band):
        try:
            ocr_protect_points, ocr_protect_info = _make_ocr_extended_protection_for_uv_points(
                uv_all,
                refine_info,
                mask_before,
                margin_px=float(ocr_extended_protection_margin_px),
            )
        except Exception as e:
            ocr_protect_info = {"used": False, "reason": f"error: {e}"}
            ocr_protect_points = np.zeros_like(keep_points_by_ransac, dtype=bool)

    keep_points = keep_points_by_ransac | ocr_protect_points
    rescued_by_ocr_protection = int(np.count_nonzero((~keep_points_by_ransac) & ocr_protect_points))
    keep_ratio_before_ocr_protection = float(np.count_nonzero(keep_points_by_ransac) / max(valid_count_before, 1))
    removed_ratio_before_ocr_protection = float(1.0 - keep_ratio_before_ocr_protection)

    mask_after = np.zeros((H, W), dtype=np.uint8)
    if np.any(keep_points):
        uv_keep = uv_all[keep_points]
        u = uv_keep[:, 0]
        v = uv_keep[:, 1]
        valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        mask_after[v[valid_uv], u[valid_uv]] = 1

    mask_after, component_info = _filter_small_components(mask_after, min_area_ratio=float(small_component_area_ratio))
    depth_after = depth_before.copy()
    depth_after[mask_after == 0] = 0
    valid_count_after = int(np.count_nonzero((mask_after > 0) & (depth_after > 0)))
    keep_ratio = float(valid_count_after / max(valid_count_before, 1))
    removed_count = int(valid_count_before - valid_count_after)
    info.update({
        "used": True,
        "reason": "ok",
        "plane_source": str(plane_source),
        "plane_model": [float(v) for v in np.asarray(plane_model, dtype=np.float64).reshape(4)],
        "fit_info": fit_info,
        "valid_count_after": int(valid_count_after),
        "removed_count": int(removed_count),
        "keep_ratio": float(keep_ratio),
        "removed_ratio": float(removed_count / max(valid_count_before, 1)),
        "keep_ratio_before_ocr_protection": float(keep_ratio_before_ocr_protection),
        "removed_ratio_before_ocr_protection": float(removed_ratio_before_ocr_protection),
        "ocr_extended_protection": ocr_protect_info,
        "rescued_by_ocr_extended_protection_count": int(rescued_by_ocr_protection),
        "distance_m_min": float(np.min(distances)) if distances.size else None,
        "distance_m_median": float(np.median(distances)) if distances.size else None,
        "distance_m_p90": float(np.percentile(distances, 90)) if distances.size else None,
        "distance_m_p95": float(np.percentile(distances, 95)) if distances.size else None,
        "distance_m_max": float(np.max(distances)) if distances.size else None,
        "component_filter": component_info,
    })

    # ===== 安全guard =====
    # RANSACは失敗時の破壊力が大きいため、keep_ratioだけでなく、削除率・距離統計・参照点の小ささも見る。
    distance_p90 = float(np.percentile(distances, 90)) if distances.size else None
    removed_ratio_after_protection = float(removed_count / max(valid_count_before, 1))
    reference_point_ratio = None
    try:
        reference_point_ratio = float(info.get("reference_point_count", 0) / max(valid_count_before, 1))
    except Exception:
        reference_point_ratio = None

    ransac_guard_reasons = []
    if keep_ratio < float(min_keep_ratio):
        ransac_guard_reasons.append(f"keep_ratio={keep_ratio:.3f} < min_keep_ratio={float(min_keep_ratio):.3f}")
    if removed_ratio_after_protection > float(max_removed_ratio):
        ransac_guard_reasons.append(f"removed_ratio={removed_ratio_after_protection:.3f} > max_removed_ratio={float(max_removed_ratio):.3f}")
    if distance_p90 is not None and distance_p90 > float(distance_threshold_m) * float(max_distance_p90_over_threshold):
        ransac_guard_reasons.append(
            f"distance_p90={distance_p90:.4f} > threshold*{float(max_distance_p90_over_threshold):.2f}"
        )
    if reference_point_ratio is not None and reference_point_ratio < float(min_reference_point_ratio):
        # 参照点が少ないだけで必ず失敗とは限らないが、削除率も大きい場合は危険。
        if removed_ratio_after_protection > 0.15:
            ransac_guard_reasons.append(
                f"reference_point_ratio={reference_point_ratio:.4f} < {float(min_reference_point_ratio):.4f} and removed_ratio is large"
            )

    info["reference_point_ratio"] = None if reference_point_ratio is None else float(reference_point_ratio)
    info["ransac_guard_reasons"] = ransac_guard_reasons

    if ransac_guard_reasons:
        info.update({
            "used": False,
            "guard_reverted": True,
            "reason": "dangerous RANSAC reverted: " + "; ".join(ransac_guard_reasons),
            "candidate_plane_model": [float(v) for v in np.asarray(plane_model, dtype=np.float64).reshape(4)],
            "candidate_valid_count_after": int(valid_count_after),
            "candidate_keep_ratio": float(keep_ratio),
            "candidate_removed_ratio": float(removed_ratio_after_protection),
            "valid_count_after": int(valid_count_before),
            "removed_count": 0,
            "keep_ratio": 1.0,
            "removed_ratio": 0.0,
            # 後段で使うと背表紙を削る事故があるため、used=False時のpost-RANSAC a95では採用しない。
            "plane_model": [float(v) for v in np.asarray(plane_model, dtype=np.float64).reshape(4)],
        })
        mask_after = mask_before
        depth_after = depth_before

    if shot_dir is not None and color_np is not None:
        try:
            save_ransac_spine_plane_debug(
                shot_dir=Path(shot_dir),
                color_np=color_np,
                mask_before=mask_before,
                mask_after=mask_after,
                depth_before=depth_before,
                depth_after=depth_after,
                info=info,
                stem=stem,
            )
        except Exception as e:
            info["debug_save_error"] = str(e)
            traceback.print_exc()

    return (mask_after, depth_after, info) if return_info else (mask_after, depth_after)

def _project_points_to_pixels(pts_f: np.ndarray, intr) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    カメラ座標点群 (x,y,z) を RGB 画像座標 (u,v) に投影する．
    戻り値: u, v, valid_z
    """
    pts = np.asarray(pts_f, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=bool)

    z = pts[:, 2]
    valid_z = np.isfinite(z) & (z > 1e-9)

    u = np.zeros((pts.shape[0],), dtype=np.int32)
    v = np.zeros((pts.shape[0],), dtype=np.int32)
    u[valid_z] = np.round(float(intr.fx) * pts[valid_z, 0] / z[valid_z] + float(intr.ppx)).astype(np.int32)
    v[valid_z] = np.round(float(intr.fy) * pts[valid_z, 1] / z[valid_z] + float(intr.ppy)).astype(np.int32)
    return u, v, valid_z


def save_colored_ply_ascii(path: str | Path, pts_f: np.ndarray, rgb_colors: np.ndarray) -> None:
    """
    XYZRGB付きPLYをASCII形式で保存する．
    rgb_colors は (N,3) RGB, uint8想定．
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pts = np.asarray(pts_f, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(rgb_colors).reshape(-1, 3)
    if colors.shape[0] != pts.shape[0]:
        raise ValueError(f"colors length mismatch: pts={pts.shape[0]}, colors={colors.shape[0]}")
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    with path.open('w', encoding='utf-8') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {pts.shape[0]}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for (x, y, z), (r, g, b) in zip(pts, colors):
            f.write(f'{x:.9f} {y:.9f} {z:.9f} {int(r)} {int(g)} {int(b)}\n')


def _safe_stage_name_for_path(name: str, max_len: int = 80) -> str:
    """ファイル名に使えるASCII中心のstage名へ変換する．"""
    name = str(name)
    name = re.sub(r"[^0-9A-Za-z_.-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "stage"
    return name[:int(max_len)]


def save_step_by_step_pointcloud_projection_debug(
    shot_dir: str | Path,
    color_np: np.ndarray,
    intr,
    depth_scale: float,
    *,
    stage_id: str,
    stage_name: str,
    mask01: np.ndarray | None = None,
    depth_masked: np.ndarray | None = None,
    pts_f: np.ndarray | None = None,
    stem: str = "step",
    note: str = "",
    save_ply: bool = False,
):
    """
    各処理段階の点群をRGB画像へ再投影して保存する．

    保存先:
      shot_dir/debug_step_by_step_pointcloud_rgb/

    入力は2通り:
      1. mask01 + depth_masked から点群化してRGBへ再投影
      2. pts_f を直接RGBへ再投影

    主な保存物:
      - *_projection_overlay.png
      - *_projection_mask.png
      - *_rgb_masked_by_projection.png
      - *_input_valid_mask.png              ※mask/depth指定時
      - *_input_valid_overlay.png           ※mask/depth指定時
      - *_pointcloud_rgb.ply                ※save_ply=True
      - *_log.json
    """
    shot_dir = Path(shot_dir)
    debug_dir = shot_dir / "debug_step_by_step_pointcloud_rgb"
    debug_dir.mkdir(parents=True, exist_ok=True)

    color_np = np.asarray(color_np)
    H, W = color_np.shape[:2]
    sid = _safe_stage_name_for_path(stage_id, max_len=24)
    sname = _safe_stage_name_for_path(stage_name, max_len=80)
    prefix = f"{stem}_{sid}_{sname}"

    info = {
        "stage_id": str(stage_id),
        "stage_name": str(stage_name),
        "note": str(note),
        "debug_dir": str(debug_dir),
        "source": None,
        "rgb_shape_hw": [int(H), int(W)],
        "depth_scale": float(depth_scale),
        "valid_input_pixel_count": None,
        "point_count": 0,
        "projected_point_count_in_image": 0,
        "projected_unique_pixel_count": 0,
        "projected_in_image_ratio": 0.0,
        "files": {},
    }

    valid_input = None
    pts = None

    if pts_f is not None:
        pts = np.asarray(pts_f, dtype=np.float64).reshape(-1, 3)
        info["source"] = "provided_points"
    else:
        if mask01 is None or depth_masked is None:
            info["source"] = "none"
            info["reason"] = "mask/depth and pts_f are both unavailable"
            save_json(debug_dir / f"{prefix}_log.json", info)
            info["files"]["log_json"] = str(debug_dir / f"{prefix}_log.json")
            return info

        mask01_u8 = (np.asarray(mask01) > 0).astype(np.uint8)
        depth_arr = np.asarray(depth_masked)
        valid_input = ((mask01_u8 > 0) & (depth_arr > 0)).astype(np.uint8)
        info["source"] = "mask_depth_to_points"
        info["valid_input_pixel_count"] = int(np.count_nonzero(valid_input))

        input_mask_path = debug_dir / f"{prefix}_input_valid_mask.png"
        cv2.imwrite(str(input_mask_path), valid_input * 255)
        info["files"]["input_valid_mask"] = str(input_mask_path)

        input_overlay = color_np.copy()
        input_overlay[valid_input == 1] = (0, 255, 0)
        input_blend = cv2.addWeighted(color_np, 0.65, input_overlay, 0.35, 0)
        contours, _ = cv2.findContours(valid_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(input_blend, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(input_blend, f"{stage_id}: {stage_name}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(input_blend, f"{stage_id}: {stage_name}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        input_overlay_path = debug_dir / f"{prefix}_input_valid_overlay.png"
        cv2.imwrite(str(input_overlay_path), input_blend)
        info["files"]["input_valid_overlay"] = str(input_overlay_path)

        pts, _uv = _mask_depth_to_points_uv_for_plane_filter(mask01_u8, depth_arr, intr, depth_scale)

    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    info["point_count"] = int(pts.shape[0])

    if pts.shape[0] <= 0:
        blank = color_np.copy()
        cv2.putText(blank, f"{stage_id}: {stage_name} / no points", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        overlay_path = debug_dir / f"{prefix}_projection_overlay.png"
        cv2.imwrite(str(overlay_path), blank)
        info["files"]["projection_overlay"] = str(overlay_path)
        info["reason"] = "no points"
        log_path = debug_dir / f"{prefix}_log.json"
        save_json(log_path, info)
        info["files"]["log_json"] = str(log_path)
        return info

    u, v, valid_z = _project_points_to_pixels(pts, intr)
    in_img = valid_z & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    proj_mask = np.zeros((H, W), dtype=np.uint8)
    if np.any(in_img):
        proj_mask[v[in_img], u[in_img]] = 1

    kernel = np.ones((3, 3), np.uint8)
    proj_vis_mask = cv2.dilate(proj_mask, kernel, iterations=1)

    proj_mask_path = debug_dir / f"{prefix}_projection_mask.png"
    cv2.imwrite(str(proj_mask_path), proj_mask * 255)
    info["files"]["projection_mask"] = str(proj_mask_path)

    # 入力マスク由来の有効画素を緑，点群の再投影をシアンで重ねる．
    overlay = color_np.copy()
    if valid_input is not None:
        overlay[valid_input == 1] = (0, 180, 0)
    overlay[proj_vis_mask == 1] = (255, 255, 0)  # BGR cyan
    blended = cv2.addWeighted(color_np, 0.62, overlay, 0.38, 0)
    contours, _ = cv2.findContours(proj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(blended, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(blended, f"{stage_id}: {stage_name}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(blended, f"{stage_id}: {stage_name}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(blended, f"points={pts.shape[0]}, proj_px={int(np.count_nonzero(proj_mask))}", (20, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(blended, f"points={pts.shape[0]}, proj_px={int(np.count_nonzero(proj_mask))}", (20, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
    overlay_path = debug_dir / f"{prefix}_projection_overlay.png"
    cv2.imwrite(str(overlay_path), blended)
    info["files"]["projection_overlay"] = str(overlay_path)

    rgb_masked = np.zeros_like(color_np)
    rgb_masked[proj_vis_mask == 1] = color_np[proj_vis_mask == 1]
    rgb_masked_path = debug_dir / f"{prefix}_rgb_masked_by_projection.png"
    cv2.imwrite(str(rgb_masked_path), rgb_masked)
    info["files"]["rgb_masked_by_projection"] = str(rgb_masked_path)

    if bool(save_ply):
        colors_rgb = np.full((pts.shape[0], 3), 180, dtype=np.uint8)
        if np.any(in_img):
            bgr = color_np[v[in_img], u[in_img], :]
            colors_rgb[in_img] = bgr[:, ::-1].astype(np.uint8)
        ply_path = debug_dir / f"{prefix}_pointcloud_rgb.ply"
        save_colored_ply_ascii(ply_path, pts, colors_rgb)
        info["files"]["pointcloud_rgb_ply"] = str(ply_path)

    info.update({
        "projected_point_count_in_image": int(np.count_nonzero(in_img)),
        "projected_unique_pixel_count": int(np.count_nonzero(proj_mask)),
        "projected_in_image_ratio": float(np.count_nonzero(in_img) / max(pts.shape[0], 1)),
    })
    log_path = debug_dir / f"{prefix}_log.json"
    save_json(log_path, info)
    info["files"]["log_json"] = str(log_path)
    return info

def save_final_pointcloud_rgb_link_debug(
    shot_dir: Path,
    color_np: np.ndarray,
    mask01: np.ndarray,
    depth_masked: np.ndarray,
    pts_f: np.ndarray,
    intr,
    stem: str,
):
    """
    最終的に取得された点群とRGB画像の対応を保存する．

    保存内容:
      - final_valid_depth_region_overlay.png
          最終mask & depth>0 の領域．点群化に使われる入力画素．
      - final_pointcloud_projection_overlay.png
          calculate_yaw 後の最終点群 pts_f をRGB画像へ再投影した領域．
      - final_pointcloud_projection_mask.png
          再投影された画素マスク．
      - final_rgb_masked_by_pointcloud.png
          再投影点群領域だけを残したRGB画像．
      - final_pointcloud_colored.ply
          RGB色付き点群．Open3Dなどで点群と色を同時確認できる．
      - final_pointcloud_rgb_link_log.json
          対応付け情報．
    """
    debug_dir = Path(shot_dir) / "debug_final_pointcloud_rgb"
    debug_dir.mkdir(parents=True, exist_ok=True)

    color_np = np.asarray(color_np)
    h, w = color_np.shape[:2]
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    depth_masked = np.asarray(depth_masked)

    # 1) 最終フィルタ後，Depthが有効な画素．
    valid_input = ((mask01 > 0) & (depth_masked > 0)).astype(np.uint8)
    cv2.imwrite(str(debug_dir / f"{stem}_final_valid_depth_region_mask.png"), valid_input * 255)

    input_overlay = color_np.copy()
    input_overlay[valid_input == 1] = (0, 255, 0)
    input_blend = cv2.addWeighted(color_np, 0.65, input_overlay, 0.35, 0)
    contours, _ = cv2.findContours(valid_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(input_blend, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(debug_dir / f"{stem}_final_valid_depth_region_overlay.png"), input_blend)

    rgb_masked_input = np.zeros_like(color_np)
    rgb_masked_input[valid_input == 1] = color_np[valid_input == 1]
    cv2.imwrite(str(debug_dir / f"{stem}_final_valid_depth_rgb_masked.png"), rgb_masked_input)

    # 2) calculate_yaw等を通った最終点群 pts_f をRGB画像へ投影．
    pts = np.asarray(pts_f, dtype=np.float64).reshape(-1, 3)
    u, v, valid_z = _project_points_to_pixels(pts, intr)
    in_img = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    proj_mask = np.zeros((h, w), dtype=np.uint8)
    if np.any(in_img):
        proj_mask[v[in_img], u[in_img]] = 1

    # 見やすいように少しだけ膨張した表示用マスクも作る．実データ自体はproj_mask．
    kernel = np.ones((3, 3), np.uint8)
    proj_vis_mask = cv2.dilate(proj_mask, kernel, iterations=1)

    cv2.imwrite(str(debug_dir / f"{stem}_final_pointcloud_projection_mask.png"), proj_mask * 255)

    proj_overlay = color_np.copy()
    proj_overlay[proj_vis_mask == 1] = (255, 255, 0)  # BGR: cyan
    proj_blend = cv2.addWeighted(color_np, 0.65, proj_overlay, 0.35, 0)
    contours, _ = cv2.findContours(proj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(proj_blend, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(debug_dir / f"{stem}_final_pointcloud_projection_overlay.png"), proj_blend)

    rgb_masked_proj = np.zeros_like(color_np)
    rgb_masked_proj[proj_vis_mask == 1] = color_np[proj_vis_mask == 1]
    cv2.imwrite(str(debug_dir / f"{stem}_final_rgb_masked_by_pointcloud.png"), rgb_masked_proj)

    # 3) 最終点群にRGB色を付与してPLY保存．投影外点は灰色にする．
    colors_rgb = np.full((pts.shape[0], 3), 180, dtype=np.uint8)
    if np.any(in_img):
        bgr = color_np[v[in_img], u[in_img], :]
        rgb = bgr[:, ::-1]
        colors_rgb[in_img] = rgb.astype(np.uint8)

    colored_ply_path = debug_dir / f"{stem}_final_pointcloud_colored.ply"
    save_colored_ply_ascii(colored_ply_path, pts, colors_rgb)

    log = {
        "rgb_shape_hw": [int(h), int(w)],
        "valid_input_pixel_count": int(np.count_nonzero(valid_input)),
        "final_point_count": int(pts.shape[0]),
        "projected_point_count_in_image": int(np.count_nonzero(in_img)),
        "projected_unique_pixel_count": int(np.count_nonzero(proj_mask)),
        "projected_in_image_ratio": float(np.count_nonzero(in_img) / max(pts.shape[0], 1)),
        "files": {
            "valid_depth_region_mask": str(debug_dir / f"{stem}_final_valid_depth_region_mask.png"),
            "valid_depth_region_overlay": str(debug_dir / f"{stem}_final_valid_depth_region_overlay.png"),
            "valid_depth_rgb_masked": str(debug_dir / f"{stem}_final_valid_depth_rgb_masked.png"),
            "pointcloud_projection_mask": str(debug_dir / f"{stem}_final_pointcloud_projection_mask.png"),
            "pointcloud_projection_overlay": str(debug_dir / f"{stem}_final_pointcloud_projection_overlay.png"),
            "rgb_masked_by_pointcloud": str(debug_dir / f"{stem}_final_rgb_masked_by_pointcloud.png"),
            "colored_ply": str(colored_ply_path),
        },
    }
    save_json(debug_dir / f"{stem}_final_pointcloud_rgb_link_log.json", log)
    print(f"✔ Saved final pointcloud/RGB link debug files: {debug_dir}")
    return log

def _clip_point_to_image(pt: tuple[int, int], w: int, h: int) -> tuple[int, int]:
    x, y = pt
    x = max(-10000, min(10000, int(x)))
    y = max(-10000, min(10000, int(y)))
    return x, y


def _draw_text_safe(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.45):
    """
    OpenCVで安全にテキストを描画する．日本語は文字化けする可能性があるため，
    デバッグ用途では番号・score中心に描画する．
    """
    x, y = int(org[0]), int(org[1])
    h, w = img.shape[:2]
    x = max(0, min(w - 1, x))
    y = max(12, min(h - 1, y))
    cv2.putText(
        img,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def _draw_polygon_on_image(
    img: np.ndarray,
    poly: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    fill_alpha: float = 0.0,
):
    """
    BGR画像上にpolygonを描画する．fill_alpha > 0 の場合は半透明塗りも行う．
    """
    poly = _poly_from_any(poly)
    if poly is None:
        return img
    pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    if fill_alpha > 0.0:
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        img[:] = cv2.addWeighted(img, 1.0 - float(fill_alpha), overlay, float(fill_alpha), 0)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return img


def _crop_polygon_perspective(
    image_bgr: np.ndarray,
    poly: np.ndarray,
    pad_px: int = 6,
):
    """
    OCR polygon周辺を透視変換で切り出す．
    polygon順序が多少崩れていても minAreaRect で安定化する．
    """
    poly = _poly_from_any(poly)
    if poly is None:
        return None

    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 4:
        return None

    rect = cv2.minAreaRect(poly.astype(np.float32))
    box = cv2.boxPoints(rect).astype(np.float32)

    w_rect = max(1.0, float(rect[1][0]))
    h_rect = max(1.0, float(rect[1][1]))
    out_w = int(round(w_rect))
    out_h = int(round(h_rect))

    # あまりに小さい場合はスキップ
    if out_w < 2 or out_h < 2:
        return None

    # boxPointsの順序を左上，右上，右下，左下に並べ替え
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).reshape(-1)
    tl = box[np.argmin(s)]
    br = box[np.argmax(s)]
    tr = box[np.argmin(diff)]
    bl = box[np.argmax(diff)]
    src = np.asarray([tl, tr, br, bl], dtype=np.float32)

    dst = np.asarray(
        [
            [pad_px, pad_px],
            [out_w + pad_px - 1, pad_px],
            [out_w + pad_px - 1, out_h + pad_px - 1],
            [pad_px, out_h + pad_px - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(
        image_bgr,
        M,
        (out_w + 2 * pad_px, out_h + 2 * pad_px),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop


def save_ocr_region_debug(
    shot_dir: Path,
    color_np: np.ndarray,
    query: str | None,
    mask01: np.ndarray,
    merged: dict,
    refine_info: dict | None = None,
    stem: str = "ocr_region",
):
    """
    OCR文字領域そのものを確認するためのデバッグ保存．

    保存内容:
      - *_ocr_all_polygons_raw.png       : ocr_result.jsonの生polygonをそのまま描画
      - *_ocr_all_polygons_corrected.png : 回転補正後のpolygonを描画
      - *_ocr_all_polygons.json          : 各OCR候補のtext, raw_poly, corrected_poly, transform_mode
      - *_ocr_selected_polygon.png       : 補正に使ったOCR polygon
      - *_ocr_selected_crop.png          : 補正に使ったOCR領域の切り出し
      - *_ocr_merged_box.png             : OCR/SAM統合結果のaxis-aligned box
    """
    debug_dir = Path(shot_dir) / "debug_ocr_band"
    debug_dir.mkdir(parents=True, exist_ok=True)

    mask01 = (mask01 > 0).astype(np.uint8)
    info = refine_info or {}
    h, w = color_np.shape[:2]
    image_shape = (h, w)
    query = "" if query is None else str(query)

    selected = info.get("selected_ocr_polygon") if isinstance(info, dict) else None
    selected_transform_mode = None
    if isinstance(selected, dict):
        selected_transform_mode = selected.get("transform_mode")

    center_for_auto, _ = _extract_merged_box_center(merged, mask01)

    # ===== 1) ocr_result.json の全OCR polygonを描画 =====
    raw_img = color_np.copy()
    corrected_img = color_np.copy()
    candidates_log = []
    ocr_path = Path(shot_dir) / "ocr_result.json"

    if ocr_path.exists():
        try:
            obj = json.loads(ocr_path.read_text(encoding="utf-8"))
            for i, item in enumerate(_iter_ocr_items(obj)):
                text = str(item.get("text", ""))
                raw_poly = _poly_from_any(item.get("bbox"))
                if raw_poly is None:
                    continue

                # 生polygonを描画．これがRGBと合わない場合，OCR側の回転座標である可能性が高い．
                _draw_polygon_on_image(raw_img, raw_poly, (0, 165, 255), thickness=1, fill_alpha=0.03)
                raw_center, raw_axis, raw_short, raw_long = _polygon_center_and_axis(raw_poly)
                if raw_center is not None:
                    _draw_text_safe(raw_img, f"raw{i}", (int(raw_center[0]) + 4, int(raw_center[1]) - 4), scale=0.35)

                # 補正後polygonを描画．selectedと同じtransform_modeが分かる場合はそれを使う．
                # 分からない場合は，この候補ごとに最もマスク・merged中心に合う変換を選ぶ．
                mode_scores = []
                modes_to_try = [selected_transform_mode] if selected_transform_mode in _OCR_POLY_TRANSFORM_MODES else list(_OCR_POLY_TRANSFORM_MODES)
                for mode in modes_to_try:
                    poly_corr = _transform_ocr_poly_to_rgb(raw_poly, image_shape, mode)
                    if poly_corr is None:
                        continue
                    c, ax, short_len, long_len = _polygon_center_and_axis(poly_corr)
                    sim = _text_similarity(query, text) if query else 0.0
                    overlap = _polygon_overlap_ratio_with_mask(poly_corr, mask01)
                    center_score = _center_distance_score(c, center_for_auto, image_shape)
                    valid_score = _poly_image_valid_score(poly_corr, image_shape)
                    score = 2.0 * sim + 0.55 * overlap + 0.45 * center_score + 0.20 * valid_score
                    mode_scores.append((score, mode, poly_corr, c, ax, short_len, long_len, sim, overlap, center_score, valid_score))

                if not mode_scores:
                    continue
                mode_scores.sort(key=lambda x: x[0], reverse=True)
                score, mode, poly_corr, center, axis, short_len, long_len, sim, overlap, center_score, valid_score = mode_scores[0]

                _draw_polygon_on_image(corrected_img, poly_corr, (255, 255, 0), thickness=1, fill_alpha=0.04)
                if center is not None:
                    cx, cy = int(round(center[0])), int(round(center[1]))
                    cv2.circle(corrected_img, (cx, cy), 3, (255, 255, 0), -1)
                    _draw_text_safe(corrected_img, f"{i}:s{sim:.2f}/o{overlap:.2f}/{mode}", (cx + 4, cy - 4), scale=0.34)

                candidates_log.append({
                    "index": int(i),
                    "text": text,
                    "score": None if item.get("score") is None else float(item.get("score")),
                    "source": item.get("source", "unknown"),
                    "similarity_to_query": float(sim),
                    "overlap_with_selected_mask": float(overlap),
                    "center_distance_score": float(center_score),
                    "image_valid_score": float(valid_score),
                    "transform_mode": mode,
                    "center": None if center is None else [float(center[0]), float(center[1])],
                    "axis": None if axis is None else [float(axis[0]), float(axis[1])],
                    "short_len": None if short_len is None else float(short_len),
                    "long_len": None if long_len is None else float(long_len),
                    "raw_poly": np.asarray(raw_poly, dtype=np.float32).tolist(),
                    "corrected_poly": np.asarray(poly_corr, dtype=np.float32).tolist(),
                })
        except Exception:
            traceback.print_exc()
            candidates_log.append({"error": "failed to read or parse ocr_result.json"})
    else:
        candidates_log.append({"error": f"ocr_result.json not found: {ocr_path}"})

    cv2.imwrite(str(debug_dir / f"{stem}_ocr_all_polygons_raw.png"), raw_img)
    cv2.imwrite(str(debug_dir / f"{stem}_ocr_all_polygons_corrected.png"), corrected_img)
    # 互換用に，従来名も補正後polygonとして保存する．
    cv2.imwrite(str(debug_dir / f"{stem}_ocr_all_polygons.png"), corrected_img)
    save_json(debug_dir / f"{stem}_ocr_all_polygons.json", {
        "query": query,
        "ocr_result_path": str(ocr_path),
        "note": "raw_poly is the coordinate in ocr_result.json; corrected_poly is transformed to after_init_rgb.png coordinates.",
        "selected_transform_mode": selected_transform_mode,
        "num_candidates": len([c for c in candidates_log if "corrected_poly" in c]),
        "candidates": candidates_log,
    })

    # ===== 2) merged結果のboxを描画 =====
    merged_img = color_np.copy()
    results = merged.get("results", []) if isinstance(merged, dict) else []
    if results:
        for i, r in enumerate(results):
            box = r.get("box") if isinstance(r, dict) else None
            poly = _poly_from_any(box)
            if poly is not None:
                _draw_polygon_on_image(merged_img, poly, (0, 255, 255), thickness=2, fill_alpha=0.08)
                c, _, _, _ = _polygon_center_and_axis(poly)
                if c is not None:
                    _draw_text_safe(merged_img, f"merged{i}: {r.get('name', '')} score={r.get('score', '')}", (int(c[0]) + 5, int(c[1]) - 5), scale=0.45)
    cv2.imwrite(str(debug_dir / f"{stem}_ocr_merged_box.png"), merged_img)

    # ===== 3) 補正に使ったselected OCR polygonを描画・切り出し =====
    selected_img = color_np.copy()
    selected_raw_img = color_np.copy()
    selected_poly = None
    selected_raw_poly = None
    if isinstance(selected, dict):
        selected_poly = _poly_from_any(selected.get("poly"))
        selected_raw_poly = _poly_from_any(selected.get("raw_poly"))

    if selected_raw_poly is not None:
        _draw_polygon_on_image(selected_raw_img, selected_raw_poly, (0, 165, 255), thickness=3, fill_alpha=0.18)
    else:
        _draw_text_safe(selected_raw_img, "No selected raw OCR polygon", (20, 40), scale=0.7)
    cv2.imwrite(str(debug_dir / f"{stem}_ocr_selected_polygon_raw.png"), selected_raw_img)

    if selected_poly is not None:
        _draw_polygon_on_image(selected_img, selected_poly, (0, 0, 255), thickness=3, fill_alpha=0.18)
        c, axis, short_len, long_len = _polygon_center_and_axis(selected_poly)
        if c is not None:
            cv2.circle(selected_img, (int(round(c[0])), int(round(c[1]))), 6, (255, 0, 0), -1)
            _draw_text_safe(
                selected_img,
                f"selected OCR: {selected.get('text', '')} sim={float(selected.get('sim', 0.0)):.2f} mode={selected.get('transform_mode', '')}",
                (int(round(c[0])) + 8, int(round(c[1])) - 8),
                scale=0.42,
            )
        if c is not None and axis is not None:
            ax, ay = float(axis[0]), float(axis[1])
            length = 350.0
            p1 = (int(round(c[0] - ax * length)), int(round(c[1] - ay * length)))
            p2 = (int(round(c[0] + ax * length)), int(round(c[1] + ay * length)))
            cv2.line(selected_img, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)

        crop = _crop_polygon_perspective(color_np, selected_poly)
        if crop is not None:
            cv2.imwrite(str(debug_dir / f"{stem}_ocr_selected_crop.png"), crop)
    else:
        _draw_text_safe(selected_img, "No selected OCR polygon", (20, 40), scale=0.7)

    cv2.imwrite(str(debug_dir / f"{stem}_ocr_selected_polygon.png"), selected_img)

def save_mask_refine_debug(
    shot_dir: Path,
    color_np: np.ndarray,
    mask_before: np.ndarray,
    mask_after: np.ndarray,
    merged: dict,
    refine_info: dict | None = None,
    stem: str = "ocr_band_refine",
    query: str | None = None,
):
    """
    OCR帯補正のデバッグ用保存．
    入力画像，補正前マスク，補正後マスク，差分，OCR/SAM対応ログを保存する．
    """
    debug_dir = Path(shot_dir) / "debug_ocr_band"
    debug_dir.mkdir(parents=True, exist_ok=True)

    mask_before = (mask_before > 0).astype(np.uint8)
    mask_after = (mask_after > 0).astype(np.uint8)

    # 入力画像保存
    cv2.imwrite(str(debug_dir / f"{stem}_input_rgb.png"), color_np)

    # マスク保存
    cv2.imwrite(str(debug_dir / f"{stem}_mask_before.png"), mask_before * 255)
    cv2.imwrite(str(debug_dir / f"{stem}_mask_after.png"), mask_after * 255)

    # 削除された領域・残った領域
    removed = ((mask_before == 1) & (mask_after == 0)).astype(np.uint8)
    kept = ((mask_before == 1) & (mask_after == 1)).astype(np.uint8)

    cv2.imwrite(str(debug_dir / f"{stem}_removed.png"), removed * 255)
    cv2.imwrite(str(debug_dir / f"{stem}_kept.png"), kept * 255)

    # overlay: 緑=残った領域，赤=削除された領域
    overlay = color_np.copy()
    overlay[kept == 1] = (0, 255, 0)
    overlay[removed == 1] = (0, 0, 255)

    blended = cv2.addWeighted(color_np, 0.65, overlay, 0.35, 0)
    cv2.imwrite(str(debug_dir / f"{stem}_overlay_kept_removed.png"), blended)

    # OCR中心線も描画
    line_img = color_np.copy()
    info = refine_info or {}
    center = info.get("ocr_center")
    axis = info.get("axis")
    half_width = info.get("half_width_px")
    if center is not None and axis is not None:
        cx, cy = float(center[0]), float(center[1])
        ax, ay = float(axis[0]), float(axis[1])
        length = 1000.0
        p1 = (int(round(cx - ax * length)), int(round(cy - ay * length)))
        p2 = (int(round(cx + ax * length)), int(round(cy + ay * length)))
        cv2.line(line_img, p1, p2, (0, 0, 255), 2)
        cv2.circle(line_img, (int(round(cx)), int(round(cy))), 6, (255, 0, 0), -1)

        # 帯の左右境界も描画
        if half_width is not None:
            nx = -ay
            ny = ax
            for sgn in (-1.0, 1.0):
                ox = nx * float(half_width) * sgn
                oy = ny * float(half_width) * sgn
                q1 = (int(round(cx + ox - ax * length)), int(round(cy + oy - ay * length)))
                q2 = (int(round(cx + ox + ax * length)), int(round(cy + oy + ay * length)))
                cv2.line(line_img, q1, q2, (255, 0, 255), 1)

    cv2.imwrite(str(debug_dir / f"{stem}_ocr_axis_band.png"), line_img)

    # 側面張り出し抑制の結果を見やすくするため，補正後マスクの輪郭も軸画像へ重ねる．
    mask_contour_img = line_img.copy()
    contours, _ = cv2.findContours(mask_after, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(mask_contour_img, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(debug_dir / f"{stem}_ocr_axis_band_with_mask_after.png"), mask_contour_img)

    # OCR文字領域そのものの可視化も保存する．
    save_ocr_region_debug(
        shot_dir=shot_dir,
        color_np=color_np,
        query=query,
        mask01=mask_before,
        merged=merged,
        refine_info=info,
        stem=stem,
    )

    log = {
        "selected_mask_name": merged.get("book_name"),
        "selected_mask_index": int(merged.get("sel_idx")) if merged.get("sel_idx") is not None else None,
        "ocr_sam_results": merged.get("results"),
        "mask_before_area_px": int(mask_before.sum()),
        "mask_after_area_px": int(mask_after.sum()),
        "kept_ratio": float(mask_after.sum() / max(mask_before.sum(), 1)),
        "refine_info": info,
    }
    save_json(debug_dir / f"{stem}_log.json", log)

    print(f"✔ Saved OCR-band debug files: {debug_dir}")

def analyze_mask_rectangularity(
    mask01: np.ndarray,
    image_shape: tuple[int, int] | None = None,
    *,
    min_area_px: int = 200,
    iou_threshold: float = 0.65,
    extent_threshold: float = 0.65,
    solidity_threshold: float = 0.92,
    width_cv_threshold: float = 0.50,
    width_max_median_threshold: float = 1.85,
    approx_vertex_max: int = 8,
):
    """
    選択SAMマスクが「きれいな回転長方形」に近いかを判定する．

    目的:
      側面部除去の列長フィルタは，正常な背表紙マスクまで削るリスクがある．
      そこで，マスクが十分に長方形らしい場合は削る操作をスキップする．

    判定指標:
      - rotated_rect_iou : マスクと最小外接回転矩形のIoU．高いほど長方形らしい．
      - extent           : mask_area / rotated_rect_area．高いほど矩形内が埋まっている．
      - solidity         : contour_area / convex_hull_area．低いと凹みや欠けが多い．
      - width_cv         : 長手方向に沿った断面幅の変動係数．低いほど幅が安定．
      - width_max_over_median : 局所的な張り出しがあると大きくなる．

    戻り値:
      clean_rectangle=True なら，側面部除去を行わない．
      needs_refine=True なら，OCR軸・列長に基づく削る操作を行う．
    """
    mask = (np.asarray(mask01) > 0).astype(np.uint8)
    area = int(mask.sum())
    info = {
        "area_px": area,
        "clean_rectangle": False,
        "needs_refine": True,
        "reason": "",
        "thresholds": {
            "iou_threshold": float(iou_threshold),
            "extent_threshold": float(extent_threshold),
            "solidity_threshold": float(solidity_threshold),
            "width_cv_threshold": float(width_cv_threshold),
            "width_max_median_threshold": float(width_max_median_threshold),
            "approx_vertex_max": int(approx_vertex_max),
        },
    }

    if area < int(min_area_px):
        info["reason"] = "too small mask"
        return info

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        info["reason"] = "no contour"
        return info

    cnt = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(cnt))
    if contour_area <= 1.0:
        info["reason"] = "invalid contour area"
        return info

    rect = cv2.minAreaRect(cnt)
    (cx, cy), (rw, rh), angle = rect
    rw = float(rw)
    rh = float(rh)
    rect_area = float(max(rw * rh, 1.0))
    rect_box = cv2.boxPoints(rect).astype(np.float32)

    rect_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(rect_mask, [np.round(rect_box).astype(np.int32)], 1)
    inter = int(((mask > 0) & (rect_mask > 0)).sum())
    union = int(((mask > 0) | (rect_mask > 0)).sum())
    rotated_rect_iou = float(inter / max(union, 1))
    extent = float(area / rect_area)

    hull = cv2.convexHull(cnt)
    hull_area = float(max(cv2.contourArea(hull), 1.0))
    solidity = float(contour_area / hull_area)

    peri = float(cv2.arcLength(cnt, True))
    approx = cv2.approxPolyDP(cnt, 0.025 * peri, True) if peri > 1.0 else cnt
    approx_vertices = int(len(approx))

    # 回転矩形の長手方向に沿って断面幅の安定性を見る．
    # きれいな背表紙なら各スライスの幅が比較的安定し，側面張り出しがあると局所的に幅が増える．
    long_axis = None
    if rw >= rh:
        # minAreaRect の angle は幅方向の角度に対応するため，rw>=rhならその方向を長手とする．
        a = np.deg2rad(float(angle))
        long_axis = np.array([np.cos(a), np.sin(a)], dtype=np.float64)
    else:
        a = np.deg2rad(float(angle) + 90.0)
        long_axis = np.array([np.cos(a), np.sin(a)], dtype=np.float64)
    long_axis = long_axis / max(float(np.linalg.norm(long_axis)), 1e-9)
    normal = np.array([-long_axis[1], long_axis[0]], dtype=np.float64)
    center = np.array([float(cx), float(cy)], dtype=np.float64)

    ys, xs = np.where(mask > 0)
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    rel = pts - center
    s = rel @ long_axis
    t = rel @ normal

    width_cv = 999.0
    width_median = 0.0
    width_max_over_median = 999.0
    width_valid_slices = 0
    try:
        # 書籍上下端は幅が欠けやすいので，中央80%程度で幅変動を見る．
        s_low, s_high = np.percentile(s, [10.0, 90.0])
        use = (s >= s_low) & (s <= s_high)
        s_use = s[use]
        t_use = t[use]
        if s_use.size >= 50:
            n_bins = max(8, min(80, int(round((s_high - s_low) / 10.0))))
            edges = np.linspace(float(s_low), float(s_high), n_bins + 1)
            widths = []
            for i in range(n_bins):
                m = (s_use >= edges[i]) & (s_use < edges[i + 1])
                if np.count_nonzero(m) < 8:
                    continue
                q0, q1 = np.percentile(t_use[m], [5.0, 95.0])
                widths.append(float(q1 - q0))
            if len(widths) >= 4:
                widths_np = np.asarray(widths, dtype=np.float64)
                width_valid_slices = int(len(widths_np))
                width_median = float(np.median(widths_np))
                width_cv = float(np.std(widths_np) / max(width_median, 1e-6))
                width_max_over_median = float(np.max(widths_np) / max(width_median, 1e-6))
    except Exception:
        pass

    # 条件をやや保守的にする．「長方形っぽい」場合だけ削りを完全スキップする．
    clean_rectangle = (
        rotated_rect_iou >= float(iou_threshold)
        and extent >= float(extent_threshold)
        and solidity >= float(solidity_threshold)
        and approx_vertices <= int(approx_vertex_max)
        and (width_valid_slices < 4 or (
            width_cv <= float(width_cv_threshold)
            and width_max_over_median <= float(width_max_median_threshold)
        ))
    )

    info.update({
        "clean_rectangle": bool(clean_rectangle),
        "needs_refine": bool(not clean_rectangle),
        "reason": "clean rotated rectangle" if clean_rectangle else "irregular mask shape",
        "contour_area_px": float(contour_area),
        "rotated_rect_area_px": float(rect_area),
        "rotated_rect_iou": float(rotated_rect_iou),
        "extent": float(extent),
        "solidity": float(solidity),
        "approx_vertices": int(approx_vertices),
        "width_cv": float(width_cv),
        "width_median_px": float(width_median),
        "width_max_over_median": float(width_max_over_median),
        "width_valid_slices": int(width_valid_slices),
        "rect_center": [float(cx), float(cy)],
        "rect_size": [float(rw), float(rh)],
        "rect_angle_deg": float(angle),
        "rect_box": rect_box.tolist(),
    })
    return info


def save_mask_rectangularity_debug(
    shot_dir: Path,
    color_np: np.ndarray,
    mask01: np.ndarray,
    shape_info: dict,
    stem: str,
):
    """長方形判定のデバッグ画像とJSONを保存する．"""
    debug_dir = Path(shot_dir) / "debug_mask_shape"
    debug_dir.mkdir(parents=True, exist_ok=True)

    mask = (np.asarray(mask01) > 0).astype(np.uint8)
    cv2.imwrite(str(debug_dir / f"{stem}_mask_shape_input.png"), mask * 255)

    img = color_np.copy()
    overlay = img.copy()
    overlay[mask > 0] = (0, 255, 0)
    img = cv2.addWeighted(img, 0.70, overlay, 0.30, 0)

    rect_box = shape_info.get("rect_box") if isinstance(shape_info, dict) else None
    if rect_box is not None:
        try:
            pts = np.round(np.asarray(rect_box, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
        except Exception:
            pass

    text_lines = [
        f"clean={shape_info.get('clean_rectangle')}",
        f"needs_refine={shape_info.get('needs_refine')}",
        f"iou={float(shape_info.get('rotated_rect_iou', 0.0)):.3f}",
        f"extent={float(shape_info.get('extent', 0.0)):.3f}",
        f"solidity={float(shape_info.get('solidity', 0.0)):.3f}",
        f"w_cv={float(shape_info.get('width_cv', 999.0)):.3f}",
        f"w_max/med={float(shape_info.get('width_max_over_median', 999.0)):.3f}",
    ]
    y0 = 28
    for i, txt in enumerate(text_lines):
        y = y0 + i * 24
        cv2.putText(img, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(debug_dir / f"{stem}_mask_shape_rectangularity.png"), img)
    save_json(debug_dir / f"{stem}_mask_shape_rectangularity.json", shape_info)
    print(f"✔ Saved mask rectangularity debug files: {debug_dir}")


def save_pointcloud_screenshot(
    pts_f: np.ndarray,
    target_point: np.ndarray,
    save_path: Path,
    show_window: bool = False,
) -> None:
    """
    点群とターゲット点を Open3D で描画し、そのスクリーンショットを保存する関数。
    既存の visualize_points_and_target_open3d には影響を与えない。

    Parameters
    ----------
    pts_f : (N, 3) np.ndarray
        点群 [m]
    target_point : (3,) np.ndarray
        ターゲット点 [m]
    save_path : Path
        画像の保存先パス (png など)
    show_window : bool
        True の場合はウィンドウを表示、False の場合は非表示でレンダリングのみ
    """
    try:
        pts = np.asarray(pts_f).reshape(-1, 3)
        tgt = np.asarray(target_point).reshape(3)

        # 点群
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # ターゲット点を小さな球で可視化
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(tgt)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 赤色

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=show_window)
        vis.add_geometry(pcd)
        vis.add_geometry(sphere)

        # 一度レンダリングしてからキャプチャ
        vis.poll_events()
        vis.update_renderer()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        vis.capture_screen_image(str(save_path), do_render=True)

        vis.destroy_window()
        print(f"✔ Saved pointcloud screenshot: {save_path}")

    except Exception:
        # ここで落ちてもメイン処理に影響が出ないようにする
        traceback.print_exc()
        print("⚠ 点群スクリーンショット保存に失敗しました（処理は続行します）")

# =============================================================================
# Online / Offline 共通認識フロー
# =============================================================================
_SAM_RUNNER_CACHE = {}


def _make_sam_config_compat(
    encoder_path: str,
    decoder_path: str,
    sam_device: str,
    *,
    sam_pts_side: tuple[int, int] = (32, 8),
    sam_decoder_k_keep: int = 1,
    sam_target_len: int = 768,
):
    """
    infer_for_storage.py の版によって SamConfig の引数が異なるため，
    高速化引数に対応している場合だけ渡す．
    """
    try:
        return SamConfig(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            device=sam_device,
            target_len=int(sam_target_len),
            pts_side=tuple(int(v) for v in sam_pts_side),
            decoder_k_keep=int(sam_decoder_k_keep),
        )
    except TypeError:
        return SamConfig(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            device=sam_device,
        )


def _get_sam_runner_compat(
    encoder_path: str,
    decoder_path: str,
    sam_device: str,
    *,
    sam_pts_side: tuple[int, int] = (32, 8),
    sam_decoder_k_keep: int = 1,
    sam_target_len: int = 768,
    use_cache: bool = True,
):
    key = (
        str(Path(encoder_path).expanduser()),
        str(Path(decoder_path).expanduser()),
        str(sam_device),
        tuple(int(v) for v in sam_pts_side),
        int(sam_decoder_k_keep),
        int(sam_target_len),
    )
    if use_cache and key in _SAM_RUNNER_CACHE:
        print("[SAM2 CACHE] reuse SAM2 runner")
        return _SAM_RUNNER_CACHE[key]

    sam_cfg = _make_sam_config_compat(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        sam_device=sam_device,
        sam_pts_side=sam_pts_side,
        sam_decoder_k_keep=sam_decoder_k_keep,
        sam_target_len=sam_target_len,
    )
    runner = SamBatchInfer_storage(sam_cfg)
    if use_cache:
        _SAM_RUNNER_CACHE[key] = runner
    return runner


def _make_stage_save_cfg_compat(shot_dir: Path):
    """
    StageSaveCfg の版差を吸収する．
    必要な rgb_bookshelves_overlay.jpg は保存し，余計な中間画像は可能なら抑制する．
    """
    try:
        return StageSaveCfg(
            out_dir=shot_dir,
            save_after_nms=False,
            save_before_smooth=False,
            save_after_smooth=False,
            save_selected=False,
            save_bookshelves=True,
            save_belly_band=False,
        )
    except TypeError:
        return StageSaveCfg(out_dir=shot_dir)


def _infer_masks_compat(
    sam_runner,
    rgb_pil,
    stage_cfg,
    *,
    depth_np_u16: np.ndarray | None = None,
    depth_merge_tolerance_raw: int = 30,
):
    """
    infer_masks() が depth_u16 に対応している版なら，SAM後処理のマスク統合時にも
    Depth中央値差による統合抑制を有効化する．未対応なら従来引数で実行する．
    """
    try:
        return sam_runner.infer_masks(
            rgb_pil,
            stage_save=stage_cfg,
            stem_for_save="rgb",
            depth_u16=depth_np_u16,
            depth_merge_tolerance_raw=int(depth_merge_tolerance_raw),
            depth_merge_min_valid_px=30,
        )
    except TypeError:
        return sam_runner.infer_masks(
            rgb_pil,
            stage_save=stage_cfg,
            stem_for_save="rgb",
        )


def _save_camera_params_json(shot_dir: Path, intr, depth_scale: float, fps: int | None = None) -> None:
    camera_json = {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "depth_scale": float(depth_scale),
    }
    if fps is not None:
        camera_json["fps"] = int(fps)
    save_json(Path(shot_dir) / "camera_params.json", camera_json)


def _save_final_valid_depth_region_overlay_root(
    shot_dir: Path,
    color_np: np.ndarray,
    mask01: np.ndarray,
    depth_masked: np.ndarray,
    stem: str,
):
    """
    最終的に点群化に使われる mask & depth>0 の領域を，shot_dir直下に保存する．
    100testで表示する mask*_final_valid_depth_region_overlay.png はこの画像．
    """
    shot_dir = Path(shot_dir)
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    depth_masked = np.asarray(depth_masked)
    valid = ((mask01 > 0) & (depth_masked > 0)).astype(np.uint8)

    overlay = np.asarray(color_np).copy()
    overlay[valid == 1] = (0, 255, 0)
    blend = cv2.addWeighted(color_np, 0.65, overlay, 0.35, 0)
    contours, _ = cv2.findContours(valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(blend, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)

    out_path = shot_dir / f"{stem}_final_valid_depth_region_overlay.png"
    cv2.imwrite(str(out_path), blend)
    log = {
        "file": str(out_path),
        "valid_input_pixel_count": int(np.count_nonzero(valid)),
        "note": "This is the final mask/depth>0 region used for pointcloud generation.",
    }
    save_json(shot_dir / f"{stem}_final_valid_depth_region_overlay_log.json", log)
    return log




def _save_mask_depth_points_debug_image(
    *,
    shot_dir: Path,
    color_np: np.ndarray,
    depth_np_u16: np.ndarray,
    mask01: np.ndarray,
    stem: str,
    title: str,
) -> dict:
    """
    選択マスク内の有効Depth点を，Depth値で色分けして画像保存する．

    保存するもの:
      - {stem}_depth_points.png         : 黒背景にDepth色分け点群を描画
      - {stem}_depth_points_overlay.png : RGB画像上にDepth色分け点群を重畳
      - {stem}_depth_points_log.json    : 有効点数とDepth統計

    ここでの「点群」は，RGB画像座標上に投影されたDepth有効画素の集合．
    3D Open3D表示ではなく，原因確認用の2D投影画像として保存する．
    """
    shot_dir = Path(shot_dir)
    shot_dir.mkdir(parents=True, exist_ok=True)

    color_np = np.asarray(color_np)
    depth_np_u16 = np.asarray(depth_np_u16)
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)

    if color_np.ndim != 3 or color_np.shape[:2] != depth_np_u16.shape[:2]:
        log = {
            "used": False,
            "reason": "shape mismatch",
            "color_shape": list(color_np.shape),
            "depth_shape": list(depth_np_u16.shape),
            "mask_shape": list(mask01.shape),
        }
        save_json(shot_dir / f"{stem}_depth_points_log.json", log)
        return log

    valid = (mask01 > 0) & (depth_np_u16 > 0)
    valid_count = int(np.count_nonzero(valid))
    mask_area = int(np.count_nonzero(mask01 > 0))

    points_img = np.zeros_like(color_np, dtype=np.uint8)
    overlay = color_np.copy()

    if valid_count <= 0:
        cv2.imwrite(str(shot_dir / f"{stem}_depth_points.png"), points_img)
        cv2.imwrite(str(shot_dir / f"{stem}_depth_points_overlay.png"), overlay)
        log = {
            "used": False,
            "reason": "no valid depth pixels in mask",
            "title": title,
            "mask_area_px": mask_area,
            "valid_depth_pixel_count": valid_count,
            "valid_ratio_in_mask": 0.0,
            "depth_points_image": str(shot_dir / f"{stem}_depth_points.png"),
            "depth_points_overlay": str(shot_dir / f"{stem}_depth_points_overlay.png"),
        }
        save_json(shot_dir / f"{stem}_depth_points_log.json", log)
        return log

    vals = depth_np_u16[valid].astype(np.float32)

    # 外れ値で色がつぶれないように，5〜95%で可視化レンジを決める．
    # 実Depth値自体はlogに min/median/max も保存する．
    vmin = float(np.percentile(vals, 5.0))
    vmax = float(np.percentile(vals, 95.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(vals.min())
        vmax = float(vals.max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    norm = np.zeros(depth_np_u16.shape, dtype=np.uint8)
    clipped = np.clip(depth_np_u16.astype(np.float32), vmin, vmax)
    norm_f = (clipped - vmin) / max(vmax - vmin, 1e-6)
    norm[valid] = np.clip(norm_f[valid] * 255.0, 0, 255).astype(np.uint8)

    colorized = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    points_img[valid] = colorized[valid]

    # RGB上に点群を重畳．Depthがある点だけ色を乗せる．
    overlay[valid] = cv2.addWeighted(color_np, 0.35, colorized, 0.65, 0)[valid]

    # マスク輪郭を黄色で描画して，どの領域に対するDepth点か分かるようにする．
    contours, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(points_img, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)

    # 簡易カラーバーを右端に付ける．上ほど遠い/大きいDepth raw値．
    h, w = color_np.shape[:2]
    bar_w = 28
    bar_h = min(220, max(80, h - 40))
    x0 = max(0, w - bar_w - 12)
    y0 = 20
    grad = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
    grad = np.repeat(grad, bar_w, axis=1)
    grad_color = cv2.applyColorMap(grad, cv2.COLORMAP_JET)

    for canvas in (points_img, overlay):
        if y0 + bar_h < h and x0 + bar_w < w:
            canvas[y0:y0 + bar_h, x0:x0 + bar_w] = grad_color
            cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)
            cv2.putText(canvas, f"{int(vmax)}", (max(0, x0 - 65), y0 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{int(vmin)}", (max(0, x0 - 65), y0 + bar_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"valid={valid_count}/{mask_area}", (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    points_path = shot_dir / f"{stem}_depth_points.png"
    overlay_path = shot_dir / f"{stem}_depth_points_overlay.png"
    cv2.imwrite(str(points_path), points_img)
    cv2.imwrite(str(overlay_path), overlay)

    log = {
        "used": True,
        "reason": "ok",
        "title": title,
        "mask_area_px": mask_area,
        "valid_depth_pixel_count": valid_count,
        "valid_ratio_in_mask": float(valid_count / max(mask_area, 1)),
        "depth_raw_min": float(vals.min()),
        "depth_raw_p05": float(np.percentile(vals, 5.0)),
        "depth_raw_median": float(np.median(vals)),
        "depth_raw_p95": float(np.percentile(vals, 95.0)),
        "depth_raw_max": float(vals.max()),
        "visualize_vmin_raw": vmin,
        "visualize_vmax_raw": vmax,
        "depth_points_image": str(points_path),
        "depth_points_overlay": str(overlay_path),
        "note": "Depth-colored valid pixels inside the selected mask. This is a 2D projection on the RGB image plane.",
    }
    save_json(shot_dir / f"{stem}_depth_points_log.json", log)
    return log



def complete_spine_band_after_depth_prefilter(
    *,
    mask01_depth_filtered: np.ndarray,
    depth_masked_depth_filtered: np.ndarray,
    selected_mask01_before_depth: np.ndarray,
    depth_np_u16: np.ndarray,
    refine_info: dict,
    image_shape: tuple[int, int],
    s_bin_size_px: float = 5.0,
    ocr_short_width_scale: float = 1.45,
    min_spine_width_px: float = 30.0,
    max_spine_width_px: float = 58.0,
    extra_margin_px: float = 2.0,
    s_range_percentiles: tuple[float, float] = (1.0, 99.0),
    min_candidate_pixels: int = 80,
    min_keep_ratio_after_completion: float = 0.40,
    max_area_growth_ratio: float = 1.80,
    fill_depth_by_row_median: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Depth閾値で穴あきになった背表紙領域を，OCR軸に沿った細い中心帯の中だけ補完する．

    目的
    ----
    Depth中央値±閾値によって側面部・背景を削りつつ，背表紙面のDepth欠損やDepth外れ値で
    ガタガタになった部分だけを復活させる．

    重要な制約
    ----------
    - 補完候補は「Depth補正前の選択SAMマスク」の内側だけ．
    - 補完する範囲はOCR文字列の長辺方向(axis)に沿った背表紙中心帯だけ．
    - t方向の幅は主にOCR短辺幅から推定し，側面部まで広がらないようにmaxで制限する．
    - 復活画素のDepthは，行(s-bin)ごとの既存Depth中央値で補う．これにより，12612のような
      異常Depthをそのまま点群に戻さない．
    """
    mask_d = (np.asarray(mask01_depth_filtered) > 0).astype(np.uint8)
    raw_mask = (np.asarray(selected_mask01_before_depth) > 0).astype(np.uint8)
    depth_in = np.asarray(depth_masked_depth_filtered).copy()
    depth_raw = np.asarray(depth_np_u16)

    H, W = int(image_shape[0]), int(image_shape[1])
    info = {
        "used": False,
        "reason": "not initialized",
        "algorithm": "complete_spine_band_after_depth_prefilter",
        "area_before": int(mask_d.sum()),
        "raw_selected_area": int(raw_mask.sum()),
        "s_bin_size_px": float(s_bin_size_px),
        "ocr_short_width_scale": float(ocr_short_width_scale),
        "min_spine_width_px": float(min_spine_width_px),
        "max_spine_width_px": float(max_spine_width_px),
        "extra_margin_px": float(extra_margin_px),
        "fill_depth_by_row_median": bool(fill_depth_by_row_median),
    }

    if mask_d.shape != raw_mask.shape or mask_d.shape != depth_raw.shape:
        info["reason"] = "shape mismatch"
        info["mask_depth_shape"] = list(mask_d.shape)
        info["raw_mask_shape"] = list(raw_mask.shape)
        info["depth_shape"] = list(depth_raw.shape)
        return mask_d, depth_in, info

    if int(raw_mask.sum()) <= 0:
        info["reason"] = "empty raw selected mask"
        return mask_d, depth_in, info

    if not isinstance(refine_info, dict):
        info["reason"] = "refine_info is not dict"
        return mask_d, depth_in, info

    axis = np.asarray(refine_info.get("axis", []), dtype=np.float32)
    normal = np.asarray(refine_info.get("normal", []), dtype=np.float32)
    if axis.size != 2 or normal.size != 2:
        info["reason"] = "axis/normal not found in refine_info"
        return mask_d, depth_in, info

    axis_norm = float(np.linalg.norm(axis))
    normal_norm = float(np.linalg.norm(normal))
    if axis_norm < 1e-6 or normal_norm < 1e-6:
        info["reason"] = "invalid axis/normal norm"
        return mask_d, depth_in, info
    axis = axis / axis_norm
    normal = normal / normal_norm

    # 中心は選択OCR polygon中心を優先．なければrefine_infoのocr_center/centerへfallback．
    center_xy = None
    selected_ocr = refine_info.get("selected_ocr_polygon", {}) if isinstance(refine_info.get("selected_ocr_polygon", {}), dict) else {}
    if isinstance(selected_ocr, dict) and selected_ocr.get("center") is not None:
        center_xy = selected_ocr.get("center")
        center_source = "selected_ocr_polygon.center"
    elif refine_info.get("ocr_center") is not None:
        center_xy = refine_info.get("ocr_center")
        center_source = "refine_info.ocr_center"
    elif refine_info.get("center") is not None:
        center_xy = refine_info.get("center")
        center_source = "refine_info.center"
    else:
        ys0, xs0 = np.where(mask_d > 0)
        if xs0.size <= 0:
            ys0, xs0 = np.where(raw_mask > 0)
        center_xy = [float(xs0.mean()), float(ys0.mean())]
        center_source = "mask_centroid_fallback"

    center_xy = np.asarray(center_xy, dtype=np.float32).reshape(2)
    info["axis"] = axis.astype(float).tolist()
    info["normal"] = normal.astype(float).tolist()
    info["center_xy"] = center_xy.astype(float).tolist()
    info["center_source"] = center_source

    # 背表紙幅はOCR短辺幅を主に使う．mask_width_pxは側面混入時に太く出るため主基準にしない．
    ocr_short = None
    for key in ("selected_ocr_short_len_px", "ocr_short_len_px"):
        val = refine_info.get(key)
        if val is not None:
            try:
                ocr_short = float(val)
                break
            except Exception:
                pass
    if ocr_short is None and isinstance(selected_ocr, dict) and selected_ocr.get("short_len") is not None:
        try:
            ocr_short = float(selected_ocr.get("short_len"))
        except Exception:
            ocr_short = None

    if ocr_short is None or not np.isfinite(ocr_short) or ocr_short <= 0:
        # 最後のfallback．ただし側面が混ざると太くなるのでmaxで制限する．
        try:
            ocr_short = float(refine_info.get("mask_width_px", min_spine_width_px))
            width_source = "mask_width_px_fallback"
        except Exception:
            ocr_short = float(min_spine_width_px)
            width_source = "min_width_fallback"
    else:
        width_source = "ocr_short_len_px"

    spine_width_px_raw = float(ocr_short) * float(ocr_short_width_scale)
    spine_width_px = float(np.clip(spine_width_px_raw, float(min_spine_width_px), float(max_spine_width_px)))
    half_width_px = 0.5 * spine_width_px + float(extra_margin_px)
    info.update({
        "width_source": width_source,
        "ocr_short_width_px": float(ocr_short),
        "spine_width_px_raw": float(spine_width_px_raw),
        "spine_width_px_used": float(spine_width_px),
        "half_width_px": float(half_width_px),
    })

    ys_raw, xs_raw = np.where(raw_mask > 0)
    pts_raw = np.stack([xs_raw, ys_raw], axis=1).astype(np.float32)
    s_raw = pts_raw @ axis
    t_raw = pts_raw @ normal
    t_center = float(center_xy @ normal)

    # s方向の範囲は，Depth補正後に残った背表紙中心帯があればそこから，なければraw中心帯から決める．
    raw_band_pre = np.abs(t_raw - t_center) <= half_width_px
    ys_d, xs_d = np.where(mask_d > 0)
    if xs_d.size > 0:
        pts_d = np.stack([xs_d, ys_d], axis=1).astype(np.float32)
        s_d = pts_d @ axis
        t_d = pts_d @ normal
        d_band = np.abs(t_d - t_center) <= half_width_px
        s_source_vals = s_d[d_band]
        s_range_source = "depth_filtered_mask_in_spine_band"
    else:
        s_source_vals = np.asarray([], dtype=np.float32)
        s_range_source = "none"

    if s_source_vals.size < 30:
        s_source_vals = s_raw[raw_band_pre]
        s_range_source = "raw_selected_mask_in_spine_band"
    if s_source_vals.size < 30:
        info["reason"] = "too few pixels to determine s range"
        info["s_range_source"] = s_range_source
        info["s_range_count"] = int(s_source_vals.size)
        return mask_d, depth_in, info

    sp0, sp1 = float(s_range_percentiles[0]), float(s_range_percentiles[1])
    s_min = float(np.percentile(s_source_vals, sp0))
    s_max = float(np.percentile(s_source_vals, sp1))
    if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max <= s_min:
        info["reason"] = "invalid s range"
        return mask_d, depth_in, info

    info.update({
        "s_range_source": s_range_source,
        "s_range_count": int(s_source_vals.size),
        "s_range_percentiles": [sp0, sp1],
        "s_min": s_min,
        "s_max": s_max,
        "t_center_abs": float(t_center),
    })

    candidate_band = raw_band_pre & (s_raw >= s_min) & (s_raw <= s_max)
    candidate_count = int(np.count_nonzero(candidate_band))
    info["candidate_band_pixel_count"] = candidate_count
    if candidate_count < int(min_candidate_pixels):
        info["reason"] = f"candidate band too small: {candidate_count}"
        return mask_d, depth_in, info

    # 行ごとに，raw選択マスク内の背表紙中心帯を補完候補として追加する．
    new_mask = mask_d.copy()
    fill_mask = np.zeros_like(mask_d, dtype=np.uint8)
    s_bins = np.floor((s_raw[candidate_band] - s_min) / max(float(s_bin_size_px), 1e-6)).astype(np.int32)
    cand_indices = np.flatnonzero(candidate_band)

    # 既存Depth点の行別中央値を準備する．復活画素には元Depthの外れ値を使わず，行中央値を使う．
    global_depth_vals = depth_in[(mask_d > 0) & (depth_in > 0)]
    if global_depth_vals.size > 0:
        global_depth_med = int(np.median(global_depth_vals))
    else:
        raw_valid_vals = depth_raw[(raw_mask > 0) & (depth_raw > 0)]
        global_depth_med = int(np.median(raw_valid_vals)) if raw_valid_vals.size > 0 else 0

    row_fill_count = 0
    row_depth_median_used = 0
    row_global_median_used = 0
    added_depth_synthetic_count = 0

    # 行単位で候補を追加する．候補帯は既に幅方向制限済みなので，行に穴があれば埋まる．
    unique_bins = np.unique(s_bins)
    for b in unique_bins:
        local_idx = cand_indices[s_bins == b]
        if local_idx.size <= 0:
            continue
        yy = ys_raw[local_idx]
        xx = xs_raw[local_idx]
        fill_mask[yy, xx] = 1
        row_fill_count += 1

    before_area = int(new_mask.sum())
    new_mask = ((new_mask > 0) | (fill_mask > 0)).astype(np.uint8)
    after_area = int(new_mask.sum())
    added_mask = (new_mask > 0) & (mask_d == 0)
    added_count = int(np.count_nonzero(added_mask))
    growth_ratio = float(after_area / max(before_area, 1))

    info.update({
        "area_before": before_area,
        "area_after": after_area,
        "added_pixel_count": added_count,
        "area_growth_ratio": growth_ratio,
        "row_count_filled": int(row_fill_count),
    })

    if after_area <= 0:
        info["reason"] = "empty after completion"
        return mask_d, depth_in, info
    if float(after_area / max(int(raw_mask.sum()), 1)) < float(min_keep_ratio_after_completion):
        # 通常はここには入りにくい．極端に細い候補でおかしくなった場合の保険．
        info["reason"] = "completed area is too small against raw selected mask"
        return mask_d, depth_in, info
    if growth_ratio > float(max_area_growth_ratio):
        info["reason"] = f"too much area growth: {growth_ratio:.3f}"
        return mask_d, depth_in, info

    depth_out = depth_in.copy()
    if fill_depth_by_row_median and added_count > 0:
        # 追加画素へ，s行ごとの既存Depth中央値を割り当てる．
        for b in unique_bins:
            local_idx = cand_indices[s_bins == b]
            if local_idx.size <= 0:
                continue
            yy = ys_raw[local_idx]
            xx = xs_raw[local_idx]
            add_here = added_mask[yy, xx]
            if not np.any(add_here):
                continue

            # 同じs-bin内で，補正後に残っているDepthを探す．
            s0 = s_min + float(b) * float(s_bin_size_px)
            s1 = s0 + float(s_bin_size_px)
            if xs_d.size > 0:
                in_row_existing = (s_d >= s0) & (s_d < s1) & (np.abs(t_d - t_center) <= half_width_px)
                yy_d = ys_d[in_row_existing]
                xx_d = xs_d[in_row_existing]
                vals = depth_in[yy_d, xx_d]
                vals = vals[vals > 0]
            else:
                vals = np.asarray([], dtype=depth_in.dtype)

            if vals.size > 0:
                z_fill = int(np.median(vals))
                row_depth_median_used += 1
            else:
                z_fill = int(global_depth_med)
                row_global_median_used += 1

            yy_add = yy[add_here]
            xx_add = xx[add_here]
            if z_fill > 0:
                depth_out[yy_add, xx_add] = np.asarray(z_fill, dtype=depth_out.dtype)
                added_depth_synthetic_count += int(yy_add.size)
    else:
        # 2Dマスクだけ補完したい場合．ただし点群化ではdepth>0のみ残る点に注意．
        pass

    # rawマスク外は必ず0にする．
    depth_out[new_mask == 0] = 0

    info.update({
        "used": True,
        "reason": "ok",
        "row_depth_median_used": int(row_depth_median_used),
        "row_global_median_used": int(row_global_median_used),
        "global_depth_median_used": int(global_depth_med),
        "added_depth_synthetic_count": int(added_depth_synthetic_count),
        "note": "Depth閾値で削れた画素のうち，OCR軸中心帯かつ元SAM選択マスク内の画素だけを復活させ，復活Depthは行ごとの既存中央値で補完する．",
    })
    return new_mask.astype(np.uint8), depth_out, info

def apply_final_t_width_clip(
    mask01: np.ndarray,
    depth_masked: np.ndarray,
    column_info: dict,
    *,
    image_shape: tuple[int, int] | None = None,
    width_scale: float = 1.05,
    margin_px: float = 2.0,
    min_keep_ratio: float = 0.55,
    center_mode: str = "ocr_center",
    min_width_px: float = 24.0,
    max_width_px: float = 90.0,
    require_shape_not_worse: bool = True,
):
    """
    最終マスクに対して，背表紙幅に基づくt方向クリップを行う．

    目的
    ----
    列長フィルタで use_seed_width_guard=False が採用された場合などに，
    背表紙方向には長いが，幅方向にはみ出している側面・背景を最後に削る．

    注意
    ----
    これは最終安全処理なので，削りすぎた場合，または形状評価が悪化した場合は元に戻す．
    """
    info = {
        "used": False,
        "reason": "not started",
        "width_scale": float(width_scale),
        "margin_px": float(margin_px),
        "min_keep_ratio": float(min_keep_ratio),
        "center_mode": str(center_mode),
        "min_width_px": float(min_width_px),
        "max_width_px": float(max_width_px),
    }

    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    depth_masked = np.asarray(depth_masked).copy()
    area_before = int(mask01.sum())
    info["area_before"] = area_before
    if area_before <= 0:
        info["reason"] = "empty mask"
        return mask01, depth_masked, info

    axis = np.asarray(column_info.get("axis", [0.0, -1.0]), dtype=np.float32)
    normal = np.asarray(column_info.get("normal", [1.0, 0.0]), dtype=np.float32)
    if axis.shape != (2,) or normal.shape != (2,):
        info["reason"] = "invalid axis/normal"
        return mask01, depth_masked, info

    n_norm = float(np.linalg.norm(normal))
    if n_norm < 1.0e-6:
        info["reason"] = "normal norm too small"
        return mask01, depth_masked, info
    normal = normal / n_norm

    # 幅推定値は，低パーセンタイルの異常値チェック後に選ばれた値を最優先する．
    prof = column_info.get("seed_width_profile_info", {}) if isinstance(column_info, dict) else {}
    width_sources = []
    if isinstance(prof, dict):
        width_sources.extend([
            ("seed_width_profile_info.width_selected_px", prof.get("width_selected_px")),
            ("seed_width_profile_info.width_percentile_px", prof.get("width_percentile_px")),
            ("seed_width_profile_info.width_median_px", prof.get("width_median_px")),
        ])
    width_sources.extend([
        ("shape_width_median_px", column_info.get("shape_width_median_px") if isinstance(column_info, dict) else None),
        ("selected_ocr_short_len_px", column_info.get("selected_ocr_short_len_px") if isinstance(column_info, dict) else None),
    ])

    width_px = None
    width_source = None
    for name, value in width_sources:
        try:
            if value is None:
                continue
            v = float(value)
            if np.isfinite(v) and v > 0:
                width_px = v
                width_source = name
                break
        except Exception:
            continue

    if width_px is None:
        info["reason"] = "width estimate not found"
        return mask01, depth_masked, info

    width_px_raw = float(width_px)
    width_px = float(np.clip(width_px_raw, float(min_width_px), float(max_width_px)))
    half_width = 0.5 * width_px * float(width_scale) + float(margin_px)

    ys, xs = np.where(mask01 > 0)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    # center_mode='ocr_center' の場合は column_info['center'] を使う．
    # これにより，False側で広く残った領域の重心に引っ張られず，OCR文字領域付近を中心に幅制限できる．
    center_mode_used = center_mode
    center = None
    if str(center_mode) == "ocr_center":
        try:
            c = np.asarray(column_info.get("center", None), dtype=np.float32)
            if c.shape == (2,) and np.all(np.isfinite(c)):
                center = c
        except Exception:
            center = None
    elif str(center_mode) == "mask_centroid":
        center = pts.mean(axis=0)
    elif str(center_mode) == "auto":
        # OCR中心が取れる場合はOCR中心，取れなければマスク重心．
        try:
            c = np.asarray(column_info.get("center", None), dtype=np.float32)
            if c.shape == (2,) and np.all(np.isfinite(c)):
                center = c
                center_mode_used = "ocr_center"
            else:
                center = pts.mean(axis=0)
                center_mode_used = "mask_centroid"
        except Exception:
            center = pts.mean(axis=0)
            center_mode_used = "mask_centroid"
    else:
        center = pts.mean(axis=0)
        center_mode_used = "mask_centroid"

    if center is None:
        center = pts.mean(axis=0)
        center_mode_used = "mask_centroid_fallback"

    t_vals = (pts - center.astype(np.float32)) @ normal
    keep_pts = np.abs(t_vals) <= half_width

    new_mask = np.zeros_like(mask01, dtype=np.uint8)
    new_mask[ys[keep_pts], xs[keep_pts]] = 1

    # 小さい孤立成分が残った場合に備えて，最大成分を中心に小成分を軽く除去する．
    try:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(new_mask, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if areas.size > 0:
                max_area = int(areas.max())
                min_area = max(10, int(0.002 * max_area))
                cleaned = np.zeros_like(new_mask, dtype=np.uint8)
                kept = 0
                removed_area = 0
                for lab in range(1, num_labels):
                    a = int(stats[lab, cv2.CC_STAT_AREA])
                    if a >= min_area:
                        cleaned[labels == lab] = 1
                        kept += 1
                    else:
                        removed_area += a
                new_mask = cleaned
                info["component_filter"] = {
                    "enabled": True,
                    "num_components": int(num_labels - 1),
                    "kept_components": int(kept),
                    "min_area": int(min_area),
                    "removed_area": int(removed_area),
                }
    except Exception as e:
        info["component_filter"] = {"enabled": False, "reason": str(e)}

    area_after = int(new_mask.sum())
    keep_ratio = float(area_after / max(area_before, 1))

    info.update({
        "width_source": width_source,
        "width_px_raw": width_px_raw,
        "width_px_used": width_px,
        "half_width_px": float(half_width),
        "center_mode_used": center_mode_used,
        "center_xy": [float(center[0]), float(center[1])],
        "t_min_before": float(np.min(t_vals)) if t_vals.size else None,
        "t_max_before": float(np.max(t_vals)) if t_vals.size else None,
        "area_after": area_after,
        "keep_ratio": keep_ratio,
    })

    if area_after <= 0 or keep_ratio < float(min_keep_ratio):
        info["used"] = False
        info["reason"] = f"too much removed: keep_ratio={keep_ratio:.3f}"
        return mask01, depth_masked, info

    # 形状評価が明らかに悪化した場合は戻す．
    if require_shape_not_worse:
        try:
            shape_before = analyze_mask_rectangularity(mask01, image_shape=image_shape)
            shape_after = analyze_mask_rectangularity(new_mask, image_shape=image_shape)

            def _clip_score(shape):
                if not isinstance(shape, dict):
                    return -1.0e9
                rotated_iou = float(shape.get("rotated_rect_iou", 0.0))
                extent = float(shape.get("extent", 0.0))
                solidity = float(shape.get("solidity", 0.0))
                width_cv = float(shape.get("width_cv", 999.0))
                width_max_med = float(shape.get("width_max_over_median", 999.0))
                score = 3.0 * rotated_iou + 2.0 * extent + 2.0 * solidity
                score -= 2.2 * min(width_cv, 5.0)
                score -= 1.4 * max(0.0, min(width_max_med, 10.0) - 1.15)
                if bool(shape.get("clean_rectangle", False)):
                    score += 0.35
                return float(score)

            score_before = _clip_score(shape_before)
            score_after = _clip_score(shape_after)
            info["shape_before"] = shape_before
            info["shape_after"] = shape_after
            info["shape_score_before"] = score_before
            info["shape_score_after"] = score_after

            # 少しの悪化は許容しない．最終クリップなので，基本的には形状が良くなる場合だけ採用．
            if score_after + 0.05 < score_before:
                info["used"] = False
                info["reason"] = f"shape score worsened: before={score_before:.3f}, after={score_after:.3f}"
                return mask01, depth_masked, info
        except Exception as e:
            info["shape_check_error"] = str(e)

    depth_after = depth_masked.copy()
    depth_after[new_mask == 0] = 0
    info["used"] = True
    info["reason"] = "ok"
    return new_mask.astype(np.uint8), depth_after, info


def apply_one_sided_short_column_prune_after_final(
    mask01: np.ndarray,
    depth_masked: np.ndarray,
    column_info: dict | None,
    refine_info: dict | None,
    image_shape: tuple[int, int],
    *,
    side_reference_mask_before: np.ndarray | None = None,
    side_reference_mask_after: np.ndarray | None = None,
    side_reference_depth_before: np.ndarray | None = None,
    t_bin_size_px: float = 4.0,
    span_percentiles: tuple[float, float] = (2.0, 98.0),
    length_ratio: float = 0.95,
    min_points_per_t_bin: int = 6,
    min_side_removed_pixels: int = 30,
    min_valid_keep_ratio: float = 0.60,
    max_removed_ratio: float = 0.45,
    protect_seed_band_px: float = 0.0,
    small_component_area_ratio: float = 0.0003,
    require_shape_not_worse: bool = False,
    return_info: bool = False,
):
    """
    最終出力直前に，前段処理で削られた側だけを対象として，
    背表紙長手方向の点群列長が最大列長の length_ratio 未満の列を追加削除する．

    目的:
      - Depth精度が低く，RANSAC/Depth閾値だけでは奥側側面が残るケースを補助する．
      - 両側を削ると背表紙本体まで削るため，前段で実際に削られた片側だけを対象にする．

    入力:
      mask01 / depth_masked:
          既存の最終候補マスクとDepth．この関数の出力が calculate_yaw に渡る．
      side_reference_mask_before / side_reference_mask_after:
          final_t_width_clip など，直前の片側削り処理の前後マスク．
          beforeにあってafterに無い画素のt座標から「どちら側を削ったか」を推定する．

    アルゴリズム:
      1. OCR/列長フィルタで使った背表紙軸 axis と法線 normal を取得する．
      2. 直前処理で削られた画素のt座標から，削った側を positive / negative として推定する．
      3. 現在残っている有効Depth点をt方向にbin分割する．
      4. 各t列のs方向長さをpercentile spanで計算する．
      5. 最大列長の95%未満の列のうち，削った側にある列だけを削除する．
      6. 残存率が低すぎる場合は元に戻す．
    """
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    depth_masked = np.asarray(depth_masked)
    valid = (mask01 > 0) & (depth_masked > 0)
    valid_count_before = int(np.count_nonzero(valid))

    info = {
        "used": False,
        "reason": "not initialized",
        "algorithm": "one_sided_short_column_prune_after_final",
        "note": "前段処理で削られた片側だけを対象に，最大列長の95%未満のt列を追加削除する．",
        "t_bin_size_px": float(t_bin_size_px),
        "span_percentiles": [float(span_percentiles[0]), float(span_percentiles[1])],
        "length_ratio": float(length_ratio),
        "min_points_per_t_bin": int(min_points_per_t_bin),
        "min_side_removed_pixels": int(min_side_removed_pixels),
        "min_valid_keep_ratio": float(min_valid_keep_ratio),
        "max_removed_ratio": float(max_removed_ratio),
        "protect_seed_band_px": float(protect_seed_band_px),
        "small_component_area_ratio": float(small_component_area_ratio),
        "valid_count_before": int(valid_count_before),
    }

    if valid_count_before < 50:
        info["reason"] = "too few valid depth points"
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    cinfo = column_info or {}
    rinfo = refine_info or {}

    # ===== 軸・中心を取得 =====
    axis = cinfo.get("axis", None) or rinfo.get("axis", None)
    center = cinfo.get("center", None) or cinfo.get("center_xy", None) or rinfo.get("ocr_center", None)

    if axis is None:
        axis = _mask_pca_axis(mask01)
        axis_source = "mask_pca_fallback"
    else:
        axis_source = "column_info_or_refine_info"

    if center is None:
        center = _mask_centroid(mask01)
        center_source = "mask_centroid_fallback"
    else:
        center_source = "column_info_or_refine_info"

    if axis is None or center is None:
        info["reason"] = "axis or center unavailable"
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    axis = np.asarray(axis, dtype=np.float64).reshape(2)
    an = float(np.linalg.norm(axis))
    if an < 1e-9:
        info["reason"] = "invalid axis"
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)
    axis = axis / an
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    center = np.asarray(center, dtype=np.float64).reshape(2)

    # ===== 前段で削った側を推定 =====
    removed_side_sign = None
    removed_side_source = None
    removed_side_detail = {}

    if side_reference_mask_before is not None and side_reference_mask_after is not None:
        before = (np.asarray(side_reference_mask_before) > 0)
        after = (np.asarray(side_reference_mask_after) > 0)
        removed = before & (~after)
        if side_reference_depth_before is not None:
            removed = removed & (np.asarray(side_reference_depth_before) > 0)

        ys_r, xs_r = np.where(removed)
        if xs_r.size >= int(min_side_removed_pixels):
            pts_r = np.stack([xs_r.astype(np.float64), ys_r.astype(np.float64)], axis=1)
            t_removed = (pts_r - center) @ normal
            neg_count = int(np.count_nonzero(t_removed < 0.0))
            pos_count = int(np.count_nonzero(t_removed > 0.0))
            # 多く削られた側を対象にする．同数に近い場合は中央値の符号を使う．
            if max(neg_count, pos_count) >= max(3, int(0.55 * t_removed.size)):
                removed_side_sign = -1 if neg_count > pos_count else 1
                removed_side_source = "removed_pixels_majority"
            else:
                med = float(np.median(t_removed))
                removed_side_sign = -1 if med < 0.0 else 1
                removed_side_source = "removed_pixels_median"
            removed_side_detail = {
                "removed_pixel_count": int(t_removed.size),
                "removed_t_median": float(np.median(t_removed)),
                "removed_t_p05": float(np.percentile(t_removed, 5.0)),
                "removed_t_p95": float(np.percentile(t_removed, 95.0)),
                "removed_negative_count": int(neg_count),
                "removed_positive_count": int(pos_count),
            }

    # 直前差分から側が分からない場合，final_t_width_clip_info のt範囲から片寄りを推定する．
    if removed_side_sign is None:
        final_clip = cinfo.get("final_t_width_clip", {}) if isinstance(cinfo, dict) else {}
        try:
            t_min_before = final_clip.get("t_min_before", None)
            t_max_before = final_clip.get("t_max_before", None)
            if t_min_before is not None and t_max_before is not None:
                # OCR中心を0として，外側に長く伸びている方を対象側にする．
                removed_side_sign = -1 if abs(float(t_min_before)) >= abs(float(t_max_before)) else 1
                removed_side_source = "final_t_width_clip_range_fallback"
                removed_side_detail = {
                    "t_min_before": float(t_min_before),
                    "t_max_before": float(t_max_before),
                }
        except Exception:
            pass

    if removed_side_sign is None:
        info.update({
            "reason": "failed to estimate removed side from previous processing",
            "axis": [float(axis[0]), float(axis[1])],
            "normal": [float(normal[0]), float(normal[1])],
            "center": [float(center[0]), float(center[1])],
        })
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    removed_side_sign = int(removed_side_sign)
    target_side_name = "positive_t" if removed_side_sign > 0 else "negative_t"

    # ===== 現在の最終候補に対して，t列ごとのs方向長さを評価 =====
    ys, xs = np.where(valid)
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    rel = pts - center
    s_coords = rel @ axis
    t_coords = rel @ normal

    t_bin_size_px = max(float(t_bin_size_px), 1.0)
    t_min_all = float(np.min(t_coords))
    t_max_all = float(np.max(t_coords))
    if not np.isfinite(t_min_all) or not np.isfinite(t_max_all) or t_max_all <= t_min_all:
        info["reason"] = "invalid t range"
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    n_t_bins = int(np.floor((t_max_all - t_min_all) / t_bin_size_px)) + 1
    t_bins = np.floor((t_coords - t_min_all) / t_bin_size_px).astype(np.int32)
    t_bins = np.clip(t_bins, 0, n_t_bins - 1)

    p_low = float(np.clip(float(span_percentiles[0]), 0.0, 49.0))
    p_high = float(np.clip(float(span_percentiles[1]), 51.0, 100.0))
    if p_high <= p_low:
        p_low, p_high = 0.0, 100.0

    column_records = []
    lengths = []
    for b in range(n_t_bins):
        idx = np.where(t_bins == b)[0]
        count = int(idx.size)
        t0 = float(t_min_all + b * t_bin_size_px)
        t1 = float(t_min_all + (b + 1) * t_bin_size_px)
        t_center = float(0.5 * (t0 + t1))
        rec = {
            "t_bin": int(b),
            "t_min": t0,
            "t_max": t1,
            "t_center": t_center,
            "point_count": count,
            "length_px": 0.0,
            "s_min": None,
            "s_max": None,
            "target_side": bool((removed_side_sign * t_center) > float(protect_seed_band_px)),
            "is_good": False,
            "is_selected": True,
            "is_pruned": False,
        }
        if count >= int(min_points_per_t_bin):
            sv = s_coords[idx]
            try:
                s0, s1 = np.percentile(sv, [p_low, p_high])
            except Exception:
                s0, s1 = float(np.min(sv)), float(np.max(sv))
            rec["length_px"] = float(max(0.0, float(s1) - float(s0)))
            rec["s_min"] = float(s0)
            rec["s_max"] = float(s1)
        lengths.append(float(rec["length_px"]))
        column_records.append(rec)

    lengths_np = np.asarray(lengths, dtype=np.float64)
    positive_lengths = lengths_np[lengths_np > 0.0]
    if positive_lengths.size < 2:
        info["reason"] = "too few positive length columns"
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    max_length = float(np.max(positive_lengths))
    length_threshold = float(max_length * float(length_ratio))

    prune_bins = []
    target_bins = []
    for rec in column_records:
        b = int(rec["t_bin"])
        is_target = bool(rec["target_side"])
        is_good = bool(rec["point_count"] >= int(min_points_per_t_bin) and float(rec["length_px"]) >= length_threshold)
        rec["is_good"] = is_good
        if is_target:
            target_bins.append(b)
            if not is_good:
                prune_bins.append(b)
                rec["is_selected"] = False
                rec["is_pruned"] = True

    if not target_bins:
        info.update({
            "reason": f"no bins on target side: {target_side_name}",
            "removed_side_sign": int(removed_side_sign),
            "removed_side_name": target_side_name,
            "removed_side_source": removed_side_source,
            "removed_side_detail": removed_side_detail,
        })
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    if not prune_bins:
        info.update({
            "reason": "no short columns on target side",
            "removed_side_sign": int(removed_side_sign),
            "removed_side_name": target_side_name,
            "removed_side_source": removed_side_source,
            "removed_side_detail": removed_side_detail,
            "max_length_px": float(max_length),
            "length_threshold_px": float(length_threshold),
            "target_bins": [int(v) for v in target_bins],
        })
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    prune_bins_set = set(int(v) for v in prune_bins)
    keep_point = np.ones((pts.shape[0],), dtype=bool)
    for i, b in enumerate(t_bins):
        if int(b) in prune_bins_set:
            keep_point[i] = False

    new_mask = np.zeros_like(mask01, dtype=np.uint8)
    new_mask[ys[keep_point], xs[keep_point]] = 1

    try:
        new_mask, comp_info = _filter_small_components(
            new_mask,
            min_area_ratio=float(small_component_area_ratio),
        )
    except Exception as e:
        comp_info = {"enabled": False, "reason": str(e)}

    depth_after = depth_masked.copy()
    depth_after[new_mask == 0] = 0
    valid_count_after = int(np.count_nonzero((new_mask > 0) & (depth_after > 0)))
    valid_keep_ratio = float(valid_count_after / max(valid_count_before, 1))
    removed_count = int(valid_count_before - valid_count_after)
    removed_ratio = float(removed_count / max(valid_count_before, 1))

    info.update({
        "used": True,
        "reason": "ok",
        "axis_source": axis_source,
        "center_source": center_source,
        "axis": [float(axis[0]), float(axis[1])],
        "normal": [float(normal[0]), float(normal[1])],
        "center": [float(center[0]), float(center[1])],
        "removed_side_sign": int(removed_side_sign),
        "removed_side_name": target_side_name,
        "removed_side_source": removed_side_source,
        "removed_side_detail": removed_side_detail,
        "t_min_all": float(t_min_all),
        "t_max_all": float(t_max_all),
        "n_t_bins": int(n_t_bins),
        "max_length_px": float(max_length),
        "length_threshold_px": float(length_threshold),
        "target_bins": [int(v) for v in target_bins],
        "prune_bins": [int(v) for v in prune_bins],
        "valid_count_after": int(valid_count_after),
        "valid_keep_ratio": float(valid_keep_ratio),
        "removed_count": int(removed_count),
        "removed_ratio": float(removed_ratio),
        "component_filter": comp_info,
        "column_records": [
            {
                "t_bin": int(r["t_bin"]),
                "t_min": float(r["t_min"]),
                "t_max": float(r["t_max"]),
                "t_center": float(r["t_center"]),
                "point_count": int(r["point_count"]),
                "length_px": float(r["length_px"]),
                "s_min": None if r.get("s_min") is None else float(r["s_min"]),
                "s_max": None if r.get("s_max") is None else float(r["s_max"]),
                "target_side": bool(r.get("target_side", False)),
                "is_good": bool(r.get("is_good", False)),
                "is_selected": bool(r.get("is_selected", False)),
                "is_pruned": bool(r.get("is_pruned", False)),
            }
            for r in column_records
        ],
    })

    if valid_count_after <= 0 or valid_keep_ratio < float(min_valid_keep_ratio):
        info["used"] = False
        info["reason"] = f"too much removed: valid_keep_ratio={valid_keep_ratio:.3f}"
        info["valid_count_after"] = valid_count_before
        info["valid_keep_ratio"] = 1.0
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    if removed_ratio > float(max_removed_ratio):
        info["used"] = False
        info["reason"] = f"too much removed: removed_ratio={removed_ratio:.3f}"
        info["valid_count_after"] = valid_count_before
        info["valid_keep_ratio"] = 1.0
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    if require_shape_not_worse:
        try:
            shape_before = analyze_mask_rectangularity(mask01, image_shape=image_shape)
            shape_after = analyze_mask_rectangularity(new_mask, image_shape=image_shape)
            info["shape_before"] = shape_before
            info["shape_after"] = shape_after
            # 明らかにextent/solidityが悪化する場合のみ戻す．片側削りなので厳しくしすぎない．
            if float(shape_after.get("extent", 0.0)) + 0.03 < float(shape_before.get("extent", 0.0)):
                info["used"] = False
                info["reason"] = "shape extent worsened; reverted"
                return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)
            if float(shape_after.get("solidity", 0.0)) + 0.03 < float(shape_before.get("solidity", 0.0)):
                info["used"] = False
                info["reason"] = "shape solidity worsened; reverted"
                return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)
        except Exception as e:
            info["shape_check_error"] = str(e)

    return (new_mask.astype(np.uint8), depth_after, info) if return_info else (new_mask.astype(np.uint8), depth_after)

def _apply_mask_to_depth_no_save(depth_masked: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    out = np.asarray(depth_masked).copy()
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    out[mask01 == 0] = 0
    return out


def _polygon_to_mask(poly, image_shape: tuple[int, int]) -> np.ndarray | None:
    poly = _poly_from_any(poly)
    if poly is None:
        return None
    h, w = int(image_shape[0]), int(image_shape[1])
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2).copy()
    if pts.shape[0] < 3:
        return None
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    out = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(out, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], 1)
    return out


def _polygon_core_to_mask(poly, image_shape: tuple[int, int], core_scale: float = 0.65) -> np.ndarray | None:
    poly = _poly_from_any(poly)
    if poly is None:
        return None
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return None
    center = pts.mean(axis=0, keepdims=True)
    scale = float(np.clip(float(core_scale), 0.10, 1.0))
    core_pts = center + (pts - center) * scale
    return _polygon_to_mask(core_pts, image_shape)


def build_initial_mask_quality(
    mask01: np.ndarray,
    refine_info: dict | None,
    image_shape: tuple[int, int],
) -> dict:
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    info = {
        "selected_mask_area_px": int(mask01.sum()),
        "ocr_polygon_area_px": None,
        "ocr_polygon_overlap_with_selected_mask": None,
        "ocr_center_inside_selected_mask": None,
        "mask_centroid_to_ocr_center_px": None,
        "mask_axis_to_ocr_axis_angle_deg": None,
        "ocr_core_coverage_by_mask": None,
        "initial_mask_unreliable": False,
        "initial_mask_unreliable_reasons": [],
    }

    rinfo = refine_info or {}
    selected = rinfo.get("selected_ocr_polygon") if isinstance(rinfo, dict) else None
    if not isinstance(selected, dict):
        info["initial_mask_unreliable_reasons"].append("selected_ocr_polygon_unavailable")
        info["initial_mask_unreliable"] = True
        return info

    poly = _poly_from_any(selected.get("poly"))
    poly_mask = _polygon_to_mask(poly, image_shape)
    if poly is None or poly_mask is None or int(poly_mask.sum()) <= 0:
        info["initial_mask_unreliable_reasons"].append("ocr_polygon_invalid")
        info["initial_mask_unreliable"] = True
        return info

    poly_area = int(poly_mask.sum())
    overlap = int(((poly_mask > 0) & (mask01 > 0)).sum())
    info["ocr_polygon_area_px"] = int(poly_area)
    info["ocr_polygon_overlap_with_selected_mask"] = float(overlap / max(poly_area, 1))
    info["ocr_core_coverage_by_mask"] = float(overlap / max(poly_area, 1))

    ocr_center, ocr_axis_raw, _, _ = _polygon_center_and_axis(poly)
    if ocr_center is not None:
        cx, cy = int(round(float(ocr_center[0]))), int(round(float(ocr_center[1])))
        h, w = mask01.shape[:2]
        inside_img = (0 <= cx < w) and (0 <= cy < h)
        inside_mask = bool(inside_img and mask01[cy, cx] > 0)
        info["ocr_center_inside_selected_mask"] = inside_mask
        mcent = _mask_centroid(mask01)
        if mcent is not None:
            info["mask_centroid_to_ocr_center_px"] = float(np.linalg.norm(mcent - np.asarray(ocr_center, dtype=np.float64)))

    mask_axis = _mask_pca_axis(mask01)
    if ocr_axis_raw is not None and mask_axis is not None:
        ocr_axis, _ = _choose_axis_consistent_with_mask(ocr_axis_raw, mask_axis)
        ocr_axis = np.asarray(ocr_axis, dtype=np.float64)
        mask_axis = np.asarray(mask_axis, dtype=np.float64)
        denom = max(float(np.linalg.norm(ocr_axis) * np.linalg.norm(mask_axis)), 1e-9)
        dot = abs(float(np.dot(ocr_axis, mask_axis)) / denom)
        dot = float(np.clip(dot, -1.0, 1.0))
        info["mask_axis_to_ocr_axis_angle_deg"] = float(np.degrees(np.arccos(dot)))

    reasons = info["initial_mask_unreliable_reasons"]
    overlap_ratio = info.get("ocr_polygon_overlap_with_selected_mask")
    if overlap_ratio is not None and overlap_ratio < 0.20:
        reasons.append(f"low_ocr_polygon_overlap={overlap_ratio:.3f}")
    if info.get("ocr_center_inside_selected_mask") is False:
        reasons.append("ocr_center_outside_selected_mask")
    angle = info.get("mask_axis_to_ocr_axis_angle_deg")
    if angle is not None and angle > 35.0:
        reasons.append(f"mask_axis_ocr_axis_mismatch_deg={angle:.1f}")
    dist = info.get("mask_centroid_to_ocr_center_px")
    if dist is not None and dist > 180.0:
        reasons.append(f"mask_centroid_far_from_ocr_center_px={dist:.1f}")
    info["initial_mask_unreliable"] = bool(reasons)
    return info


def _axis_profile_for_mask(mask01: np.ndarray, depth_masked: np.ndarray, refine_info: dict | None, intr, depth_scale: float) -> dict:
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    depth_masked = np.asarray(depth_masked)
    valid = (mask01 > 0) & (depth_masked > 0)
    out = {
        "axis_available": False,
        "center_available": False,
        "t_percentiles": None,
        "rectangular_mask_width_px": None,
        "rectangular_width_mm": None,
        "seed_width_profile_width_px": None,
        "seed_width_profile_median_px": None,
        "seed_width_profile_p10_px": None,
        "seed_width_profile_p90_px": None,
    }
    if int(np.count_nonzero(valid)) < 20:
        out["reason"] = "too few valid pixels"
        return out

    rinfo = refine_info or {}
    axis = rinfo.get("axis")
    center = rinfo.get("ocr_center")
    selected = rinfo.get("selected_ocr_polygon") if isinstance(rinfo, dict) else None
    if isinstance(selected, dict):
        poly = _poly_from_any(selected.get("poly"))
        if poly is not None:
            c_poly, axis_poly, _, _ = _polygon_center_and_axis(poly)
            mask_axis = _mask_pca_axis(mask01)
            if axis_poly is not None:
                axis, _ = _choose_axis_consistent_with_mask(axis_poly, mask_axis)
            if c_poly is not None:
                center = c_poly

    if axis is None or center is None:
        out["reason"] = "axis or center unavailable"
        return out

    axis = np.asarray(axis, dtype=np.float64).reshape(2)
    center = np.asarray(center, dtype=np.float64).reshape(2)
    an = float(np.linalg.norm(axis))
    if an < 1e-9:
        out["reason"] = "invalid axis"
        return out
    axis = axis / an
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    out["axis_available"] = True
    out["center_available"] = True
    out["axis"] = [float(axis[0]), float(axis[1])]
    out["normal"] = [float(normal[0]), float(normal[1])]
    out["center"] = [float(center[0]), float(center[1])]

    ys, xs = np.where(valid)
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    rel = pts - center
    s_vals = rel @ axis
    t_vals = rel @ normal
    percentiles = [0, 1, 2, 5, 10, 50, 90, 95, 98, 99, 100]
    try:
        q = np.percentile(t_vals, percentiles)
        out["t_percentiles"] = {f"p{int(p)}": float(v) for p, v in zip(percentiles, q)}
        out["rectangular_mask_width_px"] = float(out["t_percentiles"]["p98"] - out["t_percentiles"]["p2"])
        z_med_m = float(np.median(depth_masked[valid]) * float(depth_scale))
        scale_m_per_px = z_med_m * float(np.sqrt((normal[0] / float(intr.fx)) ** 2 + (normal[1] / float(intr.fy)) ** 2))
        out["rectangular_width_mm"] = float(out["rectangular_mask_width_px"] * scale_m_per_px * 1000.0)
    except Exception as e:
        out["percentile_error"] = str(e)

    try:
        s0, s1 = np.percentile(s_vals, [10.0, 90.0])
        widths = []
        if np.isfinite(s0) and np.isfinite(s1) and s1 > s0:
            n_bins = max(8, min(80, int(round((float(s1) - float(s0)) / 5.0))))
            edges = np.linspace(float(s0), float(s1), n_bins + 1)
            for bi in range(n_bins):
                m = (s_vals >= edges[bi]) & (s_vals < edges[bi + 1])
                if int(np.count_nonzero(m)) < 8:
                    continue
                t0, t1 = np.percentile(t_vals[m], [2.0, 98.0])
                widths.append(float(max(0.0, float(t1) - float(t0))))
        if len(widths) >= 4:
            widths_np = np.asarray(widths, dtype=np.float64)
            out["seed_width_profile_width_px"] = float(np.percentile(widths_np, 2.0))
            out["seed_width_profile_median_px"] = float(np.median(widths_np))
            out["seed_width_profile_p10_px"] = float(np.percentile(widths_np, 10.0))
            out["seed_width_profile_p90_px"] = float(np.percentile(widths_np, 90.0))
            out["seed_width_profile_num_slices"] = int(len(widths_np))
    except Exception as e:
        out["seed_width_profile_error"] = str(e)
    return out


def build_clean_rectangle_analysis(
    mask01: np.ndarray,
    depth_masked: np.ndarray,
    shape_info: dict | None,
    refine_info: dict | None,
    intr,
    depth_scale: float,
    *,
    clean_rectangle_skip_used: bool,
) -> dict:
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    shape = shape_info or {}
    h, w = mask01.shape[:2]
    ys, xs = np.where(mask01 > 0)
    bbox_w = int(xs.max() - xs.min() + 1) if xs.size else 0
    bbox_h = int(ys.max() - ys.min() + 1) if ys.size else 0
    axis_profile = _axis_profile_for_mask(mask01, depth_masked, refine_info, intr, depth_scale)
    selected = (refine_info or {}).get("selected_ocr_polygon") if isinstance(refine_info, dict) else None
    ocr_short_len = None
    if isinstance(selected, dict):
        try:
            poly = _poly_from_any(selected.get("poly"))
            _, _, short_len, _ = _polygon_center_and_axis(poly)
            ocr_short_len = None if short_len is None else float(short_len)
        except Exception:
            ocr_short_len = None
    rectangularity_values = []
    for key in ("rotated_rect_iou", "extent", "solidity"):
        try:
            rectangularity_values.append(float(shape.get(key)))
        except Exception:
            pass
    rectangularity_score = float(np.mean(rectangularity_values)) if rectangularity_values else None
    return {
        "clean_rectangle_detected": bool(shape.get("clean_rectangle", False)),
        "clean_rectangle_skip_used": bool(clean_rectangle_skip_used),
        "clean_rectangle_reason": shape.get("reason"),
        "rectangularity_score": rectangularity_score,
        "mask_area_px": int(shape.get("area_px", int(mask01.sum()))),
        "mask_width_px": int(bbox_w),
        "mask_height_px": int(bbox_h),
        "image_shape": [int(h), int(w)],
        "rectangular_mask_width_px": axis_profile.get("rectangular_mask_width_px"),
        "rectangular_width_mm": axis_profile.get("rectangular_width_mm"),
        "axis_pixel_width_mm": None,
        "pca_width_mm": None,
        "width_method_used": None,
        "ocr_short_len_px": ocr_short_len,
        "seed_width_profile_width_px": axis_profile.get("seed_width_profile_width_px"),
        "seed_width_profile_median_px": axis_profile.get("seed_width_profile_median_px"),
        "seed_width_profile_p10_px": axis_profile.get("seed_width_profile_p10_px"),
        "seed_width_profile_p90_px": axis_profile.get("seed_width_profile_p90_px"),
        "seed_width_profile_num_slices": axis_profile.get("seed_width_profile_num_slices"),
        "t_percentiles": axis_profile.get("t_percentiles"),
        "axis_profile": axis_profile,
    }


def _estimate_width_mm_for_debug(mask01: np.ndarray, depth_masked: np.ndarray, intr, depth_scale: float, column_info=None, refine_info=None) -> tuple[float | None, str | None, dict]:
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    depth_masked = _apply_mask_to_depth_no_save(depth_masked, mask01)
    width_m, width_info = estimate_book_width_from_filtered_mask_axis(
        mask01=mask01,
        depth_masked=depth_masked,
        intr=intr,
        depth_scale=depth_scale,
        column_info=column_info or {},
        refine_info=refine_info or {},
    )
    if width_m is not None:
        method = width_info.get("method") if isinstance(width_info, dict) else "axis_pixel_width"
        return float(width_m * 1000.0), method, width_info

    try:
        yaw_info = calculate_yaw(mask01, depth_masked, intr, depth_scale)
        pts_f = yaw_info["points"]
        if np.asarray(pts_f).shape[0] < 3:
            return None, "estimate_failed_too_few_points", {"axis_width_info": width_info}
        mean, pc1, pc2 = pca_axes_fix_dir(pts_f)
        pca_info = estimate_book_width(pts_f, mean, pc1, pc2)
        width_m = pca_info.get("av_book_width_m") if isinstance(pca_info, dict) else None
        if width_m is None:
            return None, "estimate_book_width_3d_pca_failed", {"axis_width_info": width_info, "pca_info": pca_info}
        return float(width_m * 1000.0), "estimate_book_width_3d_pca", {"axis_width_info": width_info, "pca_info": pca_info}
    except Exception as e:
        return None, "estimate_exception", {"axis_width_info": width_info, "error": str(e)}


def _largest_component_ratio(mask01: np.ndarray) -> float | None:
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    area = int(mask01.sum())
    if area <= 0:
        return None
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if num_labels <= 1:
        return None
    largest = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    return float(largest / max(area, 1))


def save_clean_rectangle_shadow_debug(
    shot_dir: Path,
    color_np: np.ndarray,
    original_mask: np.ndarray,
    shadow_mask: np.ndarray,
    stem: str,
) -> list[str]:
    debug_dir = Path(shot_dir) / "debug_clean_rectangle_shadow"
    debug_dir.mkdir(parents=True, exist_ok=True)
    original_mask = (np.asarray(original_mask) > 0).astype(np.uint8)
    shadow_mask = (np.asarray(shadow_mask) > 0).astype(np.uint8)

    def _overlay(mask, color):
        overlay = color_np.copy()
        overlay[mask > 0] = color
        return cv2.addWeighted(color_np, 0.65, overlay, 0.35, 0)

    original_path = debug_dir / f"{stem}_clean_rectangle_original_overlay.png"
    shadow_path = debug_dir / f"{stem}_clean_rectangle_shadow_overlay.png"
    diff_path = debug_dir / f"{stem}_clean_rectangle_original_vs_shadow_diff.png"
    cv2.imwrite(str(original_path), _overlay(original_mask, (0, 255, 0)))
    cv2.imwrite(str(shadow_path), _overlay(shadow_mask, (0, 255, 0)))

    common = (original_mask > 0) & (shadow_mask > 0)
    removed = (original_mask > 0) & (shadow_mask == 0)
    added = (original_mask == 0) & (shadow_mask > 0)
    diff = color_np.copy()
    diff[common] = (0, 255, 0)
    diff[removed] = (255, 0, 0)
    diff[added] = (0, 0, 255)
    diff = cv2.addWeighted(color_np, 0.55, diff, 0.45, 0)
    cv2.imwrite(str(diff_path), diff)
    return [str(original_path), str(shadow_path), str(diff_path)]


def evaluate_clean_rectangle_shadow_candidate(
    *,
    shot_dir: Path,
    color_np: np.ndarray,
    mask01: np.ndarray,
    depth_masked: np.ndarray,
    merged: dict,
    query: str,
    refine_info_seed: dict | None,
    shape_info: dict | None,
    intr,
    depth_scale: float,
    result_suffix: str,
    sel_idx: int,
) -> dict:
    original_mask = (np.asarray(mask01) > 0).astype(np.uint8)
    original_depth = _apply_mask_to_depth_no_save(depth_masked, original_mask)
    original_valid = int(np.count_nonzero((original_mask > 0) & (original_depth > 0)))
    out = {
        "evaluated": False,
        "reason": "not started",
        "original_width_mm": None,
        "shadow_width_mm": None,
        "original_width_method": None,
        "shadow_width_method": None,
        "original_mask_area": int(original_mask.sum()),
        "shadow_mask_area": None,
        "shadow_keep_ratio": None,
        "shadow_removed_count": None,
        "shadow_reject_reasons": [],
        "shadow_candidate_accepted_by_internal_checks": False,
        "shadow_candidate_not_adopted_reason": "shadow_only",
        "shadow_debug_files": [],
    }
    if not bool((shape_info or {}).get("clean_rectangle", False)):
        out["reason"] = "not clean rectangle"
        return out

    original_width_mm, original_method, original_info = _estimate_width_mm_for_debug(
        original_mask,
        original_depth,
        intr,
        depth_scale,
        column_info={},
        refine_info=refine_info_seed or {},
    )
    out["original_width_mm"] = original_width_mm
    out["original_width_method"] = original_method
    out["original_width_info"] = original_info

    shadow_refine_info = refine_info_seed if isinstance(refine_info_seed, dict) else None
    if not isinstance(shadow_refine_info, dict) or not isinstance(shadow_refine_info.get("selected_ocr_polygon"), dict):
        _, shadow_refine_info = refine_mask_by_ocr_axis_band(
            mask01=original_mask,
            merged=merged,
            image_shape=color_np.shape[:2],
            shot_dir=shot_dir,
            query=query,
            mask_width_ratio=1.05,
            min_keep_ratio=0.85,
            use_ocr_short_width=False,
            ocr_short_to_half_width_scale=1.35,
            min_ocr_half_width_px=28.0,
            suppress_side_protrusions=False,
            profile_bin_size_px=8.0,
            profile_width_margin=1.15,
            min_profile_half_width_px=16.0,
            max_profile_shrink_ratio=0.35,
            return_info=True,
        )
    if not isinstance(shadow_refine_info, dict) or not isinstance(shadow_refine_info.get("selected_ocr_polygon"), dict):
        out.update({"evaluated": True, "reason": "selected OCR polygon unavailable for shadow"})
        out["shadow_reject_reasons"].append("selected_ocr_polygon_unavailable")
        save_json(Path(shot_dir) / "debug_clean_rectangle_shadow" / f"mask{sel_idx}{result_suffix}_clean_rectangle_shadow_log.json", out)
        return out

    shadow_refine_info = dict(shadow_refine_info)
    shadow_refine_info["shape_rectangularity"] = shape_info
    reject_reasons = []

    def _run_shadow_column(use_guard: bool):
        return refine_mask_by_spine_column_length_after_depth(
            mask01=original_mask.copy(),
            depth_masked=original_depth.copy(),
            refine_info=shadow_refine_info,
            image_shape=color_np.shape[:2],
            t_bin_size_px=4.0,
            s_bin_size_px=5.0,
            s_gap_allow_px=28.0,
            length_reference_mode="robust_percentile",
            length_reference_percentile=100.0,
            min_length_ratio=0.95,
            relaxed_edge_length_ratio=0.95,
            min_points_per_t_bin=6,
            min_selected_t_bins=2,
            expand_selected_t_bins=0,
            s_margin_px=0.0,
            min_valid_keep_ratio=0.35,
            seed_local_window_bins=14,
            bridge_gap_bins=2,
            use_global_s_range=False,
            span_percentiles=(2.0, 98.0),
            min_occupancy_density=0.18,
            column_length_mode="continuous_run",
            min_run_point_ratio=0.18,
            use_seed_width_guard=bool(use_guard),
            seed_width_guard_scale=1.05,
            seed_width_mask_percentile=2.0,
            seed_width_low_percentile_min_median_ratio=0.50,
            seed_width_fallback_mode="median",
            seed_guard_center_mode="mask_centroid",
            seed_width_profile_s_percentiles=(10.0, 90.0),
            seed_width_profile_t_percentiles=(2.0, 98.0),
            min_seed_guard_width_px=30.0,
            max_seed_guard_width_px=90.0,
            one_sided_removal=True,
            one_sided_min_candidate_points=30,
            one_sided_score_margin_ratio=1.05,
            protect_ocr_extended_band=True,
            ocr_extended_protection_margin_px=0.0,
            ocr_extended_max_rescue_ratio=0.55,
            post_profile_side_suppression=True,
            post_profile_width_margin=1.12,
            post_profile_max_shrink_ratio=0.22,
            return_info=True,
        )

    def _score_shadow(mask_candidate, info_candidate):
        if not isinstance(info_candidate, dict) or not bool(info_candidate.get("used", False)):
            return -1.0e8, analyze_mask_rectangularity(mask_candidate, image_shape=color_np.shape[:2])
        shape_candidate = analyze_mask_rectangularity(mask_candidate, image_shape=color_np.shape[:2])
        keep = float(info_candidate.get("valid_keep_ratio", 0.0))
        score = 0.0
        score += 3.0 * float(shape_candidate.get("rotated_rect_iou", 0.0))
        score += 2.0 * float(shape_candidate.get("extent", 0.0))
        score += 2.0 * float(shape_candidate.get("solidity", 0.0))
        score -= 2.2 * min(float(shape_candidate.get("width_cv", 999.0)), 5.0)
        score -= 1.4 * max(0.0, min(float(shape_candidate.get("width_max_over_median", 999.0)), 10.0) - 1.15)
        if keep < 0.35:
            score -= 100.0
        return float(score), shape_candidate

    mask_no_guard, depth_no_guard, info_no_guard = _run_shadow_column(False)
    score_no_guard, shape_no_guard = _score_shadow(mask_no_guard, info_no_guard)
    mask_guard, depth_guard, info_guard = _run_shadow_column(True)
    score_guard, shape_guard = _score_shadow(mask_guard, info_guard)
    if score_guard > score_no_guard:
        shadow_mask, shadow_depth, shadow_column_info = mask_guard, depth_guard, info_guard
        selected_mode = "seed_width_guard_true"
        selected_score = score_guard
        selected_shape = shape_guard
    else:
        shadow_mask, shadow_depth, shadow_column_info = mask_no_guard, depth_no_guard, info_no_guard
        selected_mode = "seed_width_guard_false"
        selected_score = score_no_guard
        selected_shape = shape_no_guard

    shadow_column_info = dict(shadow_column_info or {})
    shadow_column_info["auto_seed_width_guard_selection"] = {
        "selected": selected_mode,
        "score_false": float(score_no_guard),
        "score_true": float(score_guard),
        "shape_false": shape_no_guard,
        "shape_true": shape_guard,
    }
    shadow_before_final_clip = shadow_mask.copy()
    shadow_depth_before_final_clip = shadow_depth.copy()
    shadow_mask, shadow_depth, final_clip_info = apply_final_t_width_clip(
        mask01=shadow_mask,
        depth_masked=shadow_depth,
        column_info=shadow_column_info,
        image_shape=color_np.shape[:2],
        width_scale=1.05,
        margin_px=2.0,
        min_keep_ratio=0.55,
        center_mode="ocr_center",
        min_width_px=24.0,
        max_width_px=90.0,
        require_shape_not_worse=True,
    )
    shadow_column_info["final_t_width_clip"] = final_clip_info
    shadow_mask, shadow_depth, one_sided_info = apply_one_sided_short_column_prune_after_final(
        mask01=shadow_mask,
        depth_masked=shadow_depth,
        column_info=shadow_column_info,
        refine_info=shadow_refine_info,
        image_shape=color_np.shape[:2],
        side_reference_mask_before=shadow_before_final_clip,
        side_reference_mask_after=shadow_mask,
        side_reference_depth_before=shadow_depth_before_final_clip,
        t_bin_size_px=4.0,
        span_percentiles=(2.0, 98.0),
        length_ratio=0.95,
        min_points_per_t_bin=6,
        min_side_removed_pixels=30,
        min_valid_keep_ratio=0.60,
        max_removed_ratio=0.45,
        protect_seed_band_px=0.0,
        small_component_area_ratio=0.0003,
        require_shape_not_worse=False,
        return_info=True,
    )
    shadow_column_info["one_sided_post_final_column_prune"] = one_sided_info

    shadow_valid = int(np.count_nonzero((shadow_mask > 0) & (shadow_depth > 0)))
    keep_ratio = float(shadow_valid / max(original_valid, 1))
    removed_count = int(original_valid - shadow_valid)
    shadow_width_mm, shadow_method, shadow_width_info = _estimate_width_mm_for_debug(
        shadow_mask,
        shadow_depth,
        intr,
        depth_scale,
        column_info=shadow_column_info,
        refine_info=shadow_refine_info,
    )
    component_ratio = _largest_component_ratio(shadow_mask)
    selected_ocr_core_removed_count = None
    selected_ocr_polygon_removed_count = None
    selected = shadow_refine_info.get("selected_ocr_polygon") if isinstance(shadow_refine_info, dict) else None
    if isinstance(selected, dict):
        full_mask = _polygon_to_mask(selected.get("poly"), color_np.shape[:2])
        if full_mask is not None:
            selected_ocr_polygon_removed_count = int(np.count_nonzero((full_mask > 0) & (original_mask > 0) & (shadow_mask == 0)))
        core_mask = _polygon_core_to_mask(selected.get("poly"), color_np.shape[:2], core_scale=0.65)
        if core_mask is not None:
            selected_ocr_core_removed_count = int(np.count_nonzero((core_mask > 0) & (original_mask > 0) & (shadow_mask == 0)))
    if shadow_valid <= 0:
        reject_reasons.append("shadow_candidate_empty")
    if keep_ratio < 0.35:
        reject_reasons.append(f"shadow_keep_ratio_too_low={keep_ratio:.3f}")
    if component_ratio is not None and component_ratio < 0.75:
        reject_reasons.append(f"shadow_fragmented_largest_component_ratio={component_ratio:.3f}")
    if shadow_width_mm is None:
        reject_reasons.append("shadow_width_unavailable")
    elif original_width_mm is not None and shadow_width_mm < 0.45 * original_width_mm:
        reject_reasons.append("shadow_width_collapsed_vs_original")
    if selected_ocr_core_removed_count not in (None, 0):
        reject_reasons.append(f"selected_ocr_core_removed_count={selected_ocr_core_removed_count}")
    if not bool(shadow_column_info.get("used", False)):
        reject_reasons.append("shadow_column_refine_not_used")

    debug_files = save_clean_rectangle_shadow_debug(
        shot_dir=shot_dir,
        color_np=color_np,
        original_mask=original_mask,
        shadow_mask=shadow_mask,
        stem=f"mask{sel_idx}{result_suffix}",
    )
    out.update({
        "evaluated": True,
        "reason": "shadow_candidate_computed",
        "shadow_width_mm": shadow_width_mm,
        "shadow_width_method": shadow_method,
        "shadow_mask_area": int(np.asarray(shadow_mask).sum()),
        "shadow_keep_ratio": float(keep_ratio),
        "shadow_removed_count": int(removed_count),
        "shadow_reject_reasons": reject_reasons,
        "shadow_candidate_accepted_by_internal_checks": bool(len(reject_reasons) == 0),
        "shadow_debug_files": debug_files,
        "shadow_selected_mode": selected_mode,
        "shadow_selected_score": float(selected_score),
        "shadow_selected_shape": selected_shape,
        "shadow_column_info_summary": {
            "used": bool(shadow_column_info.get("used", False)),
            "reason": shadow_column_info.get("reason"),
            "valid_keep_ratio": shadow_column_info.get("valid_keep_ratio"),
            "selected_group": shadow_column_info.get("selected_group"),
            "seed_guard_used": shadow_column_info.get("seed_guard_used"),
            "final_t_width_clip": final_clip_info,
            "one_sided_post_final_column_prune": one_sided_info,
        },
        "shadow_width_info": shadow_width_info,
        "largest_component_ratio": component_ratio,
        "selected_ocr_core_removed_count": selected_ocr_core_removed_count,
        "selected_ocr_polygon_removed_count": selected_ocr_polygon_removed_count,
        "selected_ocr_core_scale": 0.65,
    })
    save_json(Path(shot_dir) / "debug_clean_rectangle_shadow" / f"mask{sel_idx}{result_suffix}_clean_rectangle_shadow_log.json", out)
    out["_shadow_mask01"] = shadow_mask.astype(np.uint8)
    out["_shadow_depth_masked"] = np.asarray(shadow_depth).copy()
    out["_shadow_column_info"] = shadow_column_info
    out["_shadow_refine_info"] = shadow_refine_info
    return out


def decide_clean_rectangle_override(
    clean_rectangle_shadow: dict | None,
    initial_mask_quality: dict | None,
    clean_rectangle_analysis: dict | None,
) -> dict:
    shadow = clean_rectangle_shadow or {}
    imq = initial_mask_quality or {}
    analysis = clean_rectangle_analysis or {}
    reject_reasons = []
    adopt_reasons = []

    def _f(value):
        try:
            if value is None:
                return None
            v = float(value)
            return v if np.isfinite(v) else None
        except Exception:
            return None

    clean_skip = bool(analysis.get("clean_rectangle_skip_used", False))
    shadow_evaluated = bool(shadow.get("evaluated", False))
    shadow_internal_ok = bool(shadow.get("shadow_candidate_accepted_by_internal_checks", False))
    initial_unreliable = bool(imq.get("initial_mask_unreliable", True))
    original_width = _f(shadow.get("original_width_mm"))
    shadow_width = _f(shadow.get("shadow_width_mm"))
    keep_ratio = _f(shadow.get("shadow_keep_ratio"))
    selected_ocr_core_removed_count = shadow.get("selected_ocr_core_removed_count")
    rectangular_width = _f(analysis.get("rectangular_width_mm"))

    if not clean_skip:
        reject_reasons.append("clean_rectangle_skip_not_used")
    else:
        adopt_reasons.append("clean_rectangle_skip_used")

    if not shadow_evaluated:
        reject_reasons.append("shadow_not_evaluated")
    else:
        adopt_reasons.append("shadow_evaluated")

    if not shadow_internal_ok:
        reject_reasons.append("shadow_internal_checks_failed")
    else:
        adopt_reasons.append("shadow_internal_checks_passed")

    if initial_unreliable:
        reject_reasons.append("initial_mask_unreliable")
    else:
        adopt_reasons.append("initial_mask_reliable")

    if keep_ratio is None:
        reject_reasons.append("shadow_keep_ratio_unavailable")
    elif keep_ratio < 0.60 or keep_ratio > 0.90:
        reject_reasons.append(f"shadow_keep_ratio_out_of_range={keep_ratio:.3f}")
    else:
        adopt_reasons.append(f"shadow_keep_ratio_safe={keep_ratio:.3f}")

    if original_width is None or shadow_width is None:
        reject_reasons.append("width_before_or_after_unavailable")
    else:
        if shadow_width < 5.0:
            reject_reasons.append(f"shadow_width_too_small={shadow_width:.3f}")
        if shadow_width < 0.55 * original_width:
            reject_reasons.append(f"shadow_width_collapsed_vs_original={shadow_width:.3f}/{original_width:.3f}")
        width_ratio = float(original_width / max(shadow_width, 1e-6))
        rectangular_ratio = None if rectangular_width is None else float(rectangular_width / max(shadow_width, 1e-6))
        if width_ratio > 1.20 or (rectangular_ratio is not None and rectangular_ratio > 1.20):
            adopt_reasons.append(f"width_disagreement_original_to_shadow={width_ratio:.3f}")
        else:
            reject_reasons.append(f"width_disagreement_too_small_original_to_shadow={width_ratio:.3f}")

    if selected_ocr_core_removed_count is None:
        reject_reasons.append("selected_ocr_core_removed_count_unavailable")
    elif int(selected_ocr_core_removed_count) != 0:
        reject_reasons.append(f"selected_ocr_core_removed_count={int(selected_ocr_core_removed_count)}")
    else:
        adopt_reasons.append("selected_ocr_core_preserved")

    width_ratio_original_to_shadow = None
    if original_width is not None and shadow_width is not None:
        width_ratio_original_to_shadow = float(original_width / max(shadow_width, 1e-6))

    return {
        "evaluated": True,
        "adopted": bool(len(reject_reasons) == 0),
        "reason": "candidate_accepted" if len(reject_reasons) == 0 else "candidate_rejected",
        "clean_rectangle_skip_used": bool(clean_skip),
        "original_width_mm": original_width,
        "shadow_width_mm": shadow_width,
        "original_mask_area": shadow.get("original_mask_area"),
        "shadow_mask_area": shadow.get("shadow_mask_area"),
        "shadow_keep_ratio": keep_ratio,
        "shadow_removed_count": shadow.get("shadow_removed_count"),
        "initial_mask_unreliable": bool(initial_unreliable),
        "shadow_candidate_accepted_by_internal_checks": bool(shadow_internal_ok),
        "selected_ocr_core_removed_count": selected_ocr_core_removed_count,
        "width_ratio_original_to_shadow": width_ratio_original_to_shadow,
        "rectangular_width_mm": rectangular_width,
        "rectangular_mask_width_px": analysis.get("rectangular_mask_width_px"),
        "shadow_width_px": (shadow.get("shadow_width_info") or {}).get("width_px") if isinstance(shadow.get("shadow_width_info"), dict) else None,
        "seed_width_profile_width_px": analysis.get("seed_width_profile_width_px"),
        "seed_width_profile_median_px": analysis.get("seed_width_profile_median_px"),
        "adopt_reasons": adopt_reasons,
        "reject_reasons": reject_reasons,
    }


def _jsonable_for_residual_diagnostics(value):
    if isinstance(value, dict):
        return {str(k): _jsonable_for_residual_diagnostics(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_for_residual_diagnostics(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable_for_residual_diagnostics(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _float_or_none(value):
    try:
        if value is None:
            return None
        v = float(value)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _stage_axis_center_from_infos(mask01: np.ndarray, refine_info: dict | None, column_info: dict | None):
    for source_name, info, center_keys in (
        ("column_info", column_info, ("center", "ocr_center")),
        ("refine_info", refine_info, ("center", "ocr_center")),
    ):
        if not isinstance(info, dict):
            continue
        axis = info.get("axis")
        center = None
        for key in center_keys:
            if info.get(key) is not None:
                center = info.get(key)
                break
        if axis is None or center is None:
            continue
        try:
            axis = np.asarray(axis, dtype=np.float64).reshape(2)
            center = np.asarray(center, dtype=np.float64).reshape(2)
            n = float(np.linalg.norm(axis))
            if n > 1e-9:
                axis = axis / n
                normal = info.get("normal")
                if normal is None:
                    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
                else:
                    normal = np.asarray(normal, dtype=np.float64).reshape(2)
                    nn = float(np.linalg.norm(normal))
                    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64) if nn < 1e-9 else normal / nn
                return axis, normal, center, source_name
        except Exception:
            pass

    axis = _mask_pca_axis(mask01)
    center = _mask_centroid(mask01)
    if axis is None or center is None:
        return None, None, None, "unavailable"
    axis = np.asarray(axis, dtype=np.float64).reshape(2)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    return axis, normal, np.asarray(center, dtype=np.float64).reshape(2), "mask_pca_fallback"


def _mask_component_summary(mask01: np.ndarray):
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    area = int(mask01.sum())
    out = {
        "area_px": area,
        "component_count": 0,
        "largest_component_area_px": 0,
        "largest_component_ratio": None,
        "bbox_xywh": None,
        "centroid_xy": None,
    }
    if area <= 0:
        return out

    ys, xs = np.where(mask01 > 0)
    out["bbox_xywh"] = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]
    out["centroid_xy"] = [float(xs.mean()), float(ys.mean())]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    component_count = max(0, int(num_labels) - 1)
    out["component_count"] = component_count
    if component_count > 0:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
        largest = int(np.max(areas))
        out["largest_component_area_px"] = largest
        out["largest_component_ratio"] = float(largest / max(area, 1))
    return out


def _largest_component_mask(mask01: np.ndarray):
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    if int(mask01.sum()) <= 0:
        return mask01
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if num_labels <= 2:
        return mask01
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas)) + 1
    return (labels == label).astype(np.uint8)


def _axis_width_candidates_for_stage(
    mask01: np.ndarray,
    depth_masked: np.ndarray,
    intr,
    depth_scale: float,
    axis: np.ndarray | None,
    normal: np.ndarray | None,
    center: np.ndarray | None,
):
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    depth_masked = np.asarray(depth_masked)
    valid = (mask01 > 0) & (depth_masked > 0)
    if int(np.count_nonzero(valid)) < 20 or axis is None or normal is None or center is None:
        return {}, None

    ys, xs = np.where(valid)
    pts2 = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    t = (pts2 - np.asarray(center, dtype=np.float64).reshape(2)) @ np.asarray(normal, dtype=np.float64).reshape(2)
    z_values = depth_masked[valid].astype(np.float64) * float(depth_scale)
    z_med = float(np.median(z_values))
    if not np.isfinite(z_med) or z_med <= 0:
        return {}, None
    scale_m_per_px = z_med * float(np.sqrt((float(normal[0]) / float(intr.fx)) ** 2 + (float(normal[1]) / float(intr.fy)) ** 2))

    candidates = {}
    for p0, p1 in ((2.0, 98.0), (5.0, 95.0), (10.0, 90.0), (25.0, 75.0)):
        try:
            t0, t1 = np.percentile(t, [p0, p1])
            width_px = float(max(0.0, float(t1) - float(t0)))
            candidates[f"p{int(p0)}_p{int(p1)}"] = {
                "percentiles": [float(p0), float(p1)],
                "width_px": width_px,
                "width_mm": float(width_px * scale_m_per_px * 1000.0),
                "t_low": float(t0),
                "t_high": float(t1),
            }
        except Exception as e:
            candidates[f"p{int(p0)}_p{int(p1)}"] = {"error": str(e)}
    return candidates, {
        "z_median_m": z_med,
        "scale_m_per_px": float(scale_m_per_px),
        "t_min": float(np.min(t)),
        "t_max": float(np.max(t)),
        "t_median": float(np.median(t)),
    }


def _pca_width_mm_for_stage(mask01: np.ndarray, depth_masked: np.ndarray, intr, depth_scale: float):
    try:
        pts, _ = _mask_depth_to_points_uv_for_plane_filter(
            (np.asarray(mask01) > 0).astype(np.uint8),
            np.asarray(depth_masked),
            intr,
            depth_scale,
        )
        if pts.shape[0] < 20:
            return None, {"reason": "too few points"}
        mean, pc1, pc2 = pca_axes_fix_dir(pts)
        info = estimate_book_width(pts, mean, pc1, pc2)
        width_m = None
        if isinstance(info, dict):
            width_m = info.get("av_book_width_m") or info.get("width_m")
        return (None if width_m is None else float(width_m) * 1000.0), info
    except Exception as e:
        return None, {"reason": f"pca_width_error: {e}"}


def build_stage_width_diagnostics(
    stage_name: str,
    mask01: np.ndarray,
    depth_masked: np.ndarray,
    intr,
    depth_scale: float,
    refine_info: dict | None = None,
    column_info: dict | None = None,
    extra_info: dict | None = None,
):
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    depth_masked = np.asarray(depth_masked)
    valid = (mask01 > 0) & (depth_masked > 0)
    valid_count = int(np.count_nonzero(valid))
    axis, normal, center, axis_source = _stage_axis_center_from_infos(mask01, refine_info, column_info)
    axis_candidates, axis_stats = _axis_width_candidates_for_stage(
        mask01,
        depth_masked,
        intr,
        depth_scale,
        axis,
        normal,
        center,
    )
    largest_mask = _largest_component_mask(mask01)
    largest_axis_candidates, _ = _axis_width_candidates_for_stage(
        largest_mask,
        depth_masked,
        intr,
        depth_scale,
        axis,
        normal,
        center,
    )
    pca_width_mm, pca_width_info = _pca_width_mm_for_stage(mask01, depth_masked, intr, depth_scale)

    p2 = (axis_candidates.get("p2_p98") or {}).get("width_mm")
    p10 = (axis_candidates.get("p10_p90") or {}).get("width_mm")
    sensitivity_mm = None
    sensitivity_ratio = None
    if p2 is not None and p10 is not None:
        sensitivity_mm = float(p2 - p10)
        sensitivity_ratio = float(p2 / max(float(p10), 1e-6))

    depth_stats = {"median_m": None, "p05_m": None, "p95_m": None}
    if valid_count > 0:
        z = depth_masked[valid].astype(np.float64) * float(depth_scale)
        depth_stats = {
            "median_m": float(np.median(z)),
            "p05_m": float(np.percentile(z, 5)),
            "p95_m": float(np.percentile(z, 95)),
        }

    adopted_shadow_mm = p2 if p2 is not None else pca_width_mm
    out = {
        "stage_name": str(stage_name),
        "mask": _mask_component_summary(mask01),
        "valid_depth_pixel_count": valid_count,
        "depth_stats": depth_stats,
        "axis_source": axis_source,
        "axis": None if axis is None else [float(axis[0]), float(axis[1])],
        "normal": None if normal is None else [float(normal[0]), float(normal[1])],
        "center": None if center is None else [float(center[0]), float(center[1])],
        "axis_stats": axis_stats,
        "axis_width_candidates": axis_candidates,
        "largest_component_axis_width_candidates": largest_axis_candidates,
        "axis_pixel_width_mm": p2,
        "pca_width_mm": pca_width_mm,
        "pca_width_info": pca_width_info,
        "adopted_width_shadow_mm_if_stage_final": adopted_shadow_mm,
        "width_sensitivity_p2p98_minus_p10p90_mm": sensitivity_mm,
        "width_sensitivity_p2p98_over_p10p90": sensitivity_ratio,
        "extra_info": extra_info or {},
    }
    return _jsonable_for_residual_diagnostics(out)


def _stage_width_diff(prev_diag: dict | None, cur_diag: dict | None):
    if not isinstance(prev_diag, dict) or not isinstance(cur_diag, dict):
        return None
    prev_width = _float_or_none(prev_diag.get("adopted_width_shadow_mm_if_stage_final"))
    cur_width = _float_or_none(cur_diag.get("adopted_width_shadow_mm_if_stage_final"))
    prev_valid = prev_diag.get("valid_depth_pixel_count")
    cur_valid = cur_diag.get("valid_depth_pixel_count")
    return {
        "from_stage": prev_diag.get("stage_name"),
        "to_stage": cur_diag.get("stage_name"),
        "valid_count_before": prev_valid,
        "valid_count_after": cur_valid,
        "valid_count_delta": None if prev_valid is None or cur_valid is None else int(cur_valid) - int(prev_valid),
        "width_shadow_mm_before": prev_width,
        "width_shadow_mm_after": cur_width,
        "width_shadow_mm_delta": None if prev_width is None or cur_width is None else float(cur_width - prev_width),
    }


def _get_residual_policy_mode() -> str:
    mode = str(os.environ.get("RESIDUAL_POLICY_MODE", "baseline")).strip().upper()
    mode = mode.replace("-", "_")
    if mode in {"", "BASELINE", "NONE", "OFF", "0"}:
        return "baseline"
    if mode in {"A", "B", "AB", "AB_SAFE", "ABSAFE", "AB_SAFE_A_TUNED", "AB_SAFE_TUNED", "AB_SAFE_A_TUNED_V2", "AB_SAFE_TUNED_V2"}:
        if mode == "ABSAFE":
            return "AB_SAFE"
        if mode == "AB_SAFE_TUNED":
            return "AB_SAFE_A_TUNED"
        if mode == "AB_SAFE_TUNED_V2":
            return "AB_SAFE_A_TUNED_V2"
        return mode
    return "baseline"


def _residual_policy_has_a(mode: str) -> bool:
    return str(mode).upper() in {"A", "AB", "AB_SAFE", "AB_SAFE_A_TUNED", "AB_SAFE_A_TUNED_V2"}


def _residual_policy_has_b(mode: str) -> bool:
    return str(mode).upper() in {"B", "AB", "AB_SAFE", "AB_SAFE_A_TUNED", "AB_SAFE_A_TUNED_V2"}


def _residual_policy_a_variant(mode: str) -> str:
    mode = str(mode).upper()
    if mode == "AB_SAFE_A_TUNED_V2":
        return "a_tuned_v2"
    return "a_tuned" if mode == "AB_SAFE_A_TUNED" else "legacy"


def _width_mm_from_candidates(candidates: dict | None, key: str):
    if not isinstance(candidates, dict):
        return None
    return _float_or_none((candidates.get(key) or {}).get("width_mm"))


def _residual_policy_width_snapshot(
    mask01: np.ndarray,
    depth_masked: np.ndarray,
    intr,
    depth_scale: float,
    refine_info: dict | None,
    column_info: dict | None,
):
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    axis, normal, center, axis_source = _stage_axis_center_from_infos(mask01, refine_info, column_info)
    candidates, axis_stats = _axis_width_candidates_for_stage(
        mask01,
        depth_masked,
        intr,
        depth_scale,
        axis,
        normal,
        center,
    )
    largest_mask = _largest_component_mask(mask01)
    largest_candidates, _ = _axis_width_candidates_for_stage(
        largest_mask,
        depth_masked,
        intr,
        depth_scale,
        axis,
        normal,
        center,
    )
    mask_info = _mask_component_summary(mask01)
    return _jsonable_for_residual_diagnostics({
        "axis_source": axis_source,
        "axis_width_candidates": candidates,
        "largest_component_axis_width_candidates": largest_candidates,
        "axis_stats": axis_stats,
        "mask": mask_info,
        "valid_depth_pixel_count": int(np.count_nonzero((mask01 > 0) & (np.asarray(depth_masked) > 0))),
    })


def _evaluate_residual_policy_a_candidate(
    *,
    enabled: bool,
    policy_A_variant: str = "legacy",
    stage: str,
    before_mask01: np.ndarray,
    before_depth_masked: np.ndarray,
    candidate_mask01: np.ndarray,
    candidate_depth_masked: np.ndarray,
    candidate_info: dict | None,
    intr,
    depth_scale: float,
    refine_info: dict | None,
    column_info: dict | None,
):
    info = {
        "enabled": bool(enabled),
        "evaluated": False,
        "overprune_detected": False,
        "stage": str(stage),
        "before_width_p2p98_mm": None,
        "candidate_width_p2p98_mm": None,
        "before_width_p5p95_mm": None,
        "candidate_width_p5p95_mm": None,
        "before_width_p10p90_mm": None,
        "candidate_width_p10p90_mm": None,
        "candidate_valid_keep_ratio": None,
        "reject_candidate": False,
        "reason": "disabled",
        "reject_reasons": [],
        "policy_A_variant": str(policy_A_variant),
        "policy_A_triggered": False,
        "policy_A_candidate_rejected": False,
        "policy_A_candidate_kept_by_tuned_guard": False,
        "policy_A_candidate_kept_by_tuned_v2_guard": False,
        "policy_A_reject_reason": None,
        "policy_A_keep_reason": None,
        "candidate_width_mm": None,
        "pre_a95_width_mm": None,
        "post_a95_width_mm": None,
        "p2_p98_width_mm": None,
        "p10_p90_width_mm": None,
        "p2_p98_over_p10_p90_ratio": None,
        "p2_p98_minus_p10_p90_mm": None,
        "largest_component_ratio": None,
        "initial_mask_reliability": None,
        "A_after_width_plausible": False,
    }
    if not enabled:
        return info
    cinfo = candidate_info or {}
    if not bool(cinfo.get("used", False)):
        info.update({
            "evaluated": True,
            "reason": f"{stage} candidate was not used by baseline guard; no extra rejection needed",
            "already_rejected_by_baseline": bool(cinfo.get("guard_reverted", False)),
        })
        return info

    before = _residual_policy_width_snapshot(
        before_mask01, before_depth_masked, intr, depth_scale, refine_info, column_info
    )
    cand = _residual_policy_width_snapshot(
        candidate_mask01, candidate_depth_masked, intr, depth_scale, refine_info, column_info
    )

    before_p2 = _width_mm_from_candidates(before.get("axis_width_candidates"), "p2_p98")
    cand_p2 = _width_mm_from_candidates(cand.get("axis_width_candidates"), "p2_p98")
    before_p5 = _width_mm_from_candidates(before.get("axis_width_candidates"), "p5_p95")
    cand_p5 = _width_mm_from_candidates(cand.get("axis_width_candidates"), "p5_p95")
    before_p10 = _width_mm_from_candidates(before.get("axis_width_candidates"), "p10_p90")
    cand_p10 = _width_mm_from_candidates(cand.get("axis_width_candidates"), "p10_p90")
    keep_ratio = _float_or_none(cinfo.get("valid_keep_ratio"))
    if keep_ratio is None:
        keep_ratio = _float_or_none(cinfo.get("keep_ratio"))
    cand_largest_ratio = _float_or_none((cand.get("mask") or {}).get("largest_component_ratio"))

    reject_reasons = []
    for label, before_w, cand_w in (
        ("p2p98", before_p2, cand_p2),
        ("p5p95", before_p5, cand_p5),
        ("p10p90", before_p10, cand_p10),
    ):
        if before_w is not None and cand_w is not None and cand_w < before_w * 0.85:
            reject_reasons.append(f"{label}_width_shrink_ratio={cand_w / max(before_w, 1e-6):.3f}")
    if keep_ratio is not None and keep_ratio < 0.70:
        reject_reasons.append(f"candidate_valid_keep_ratio={keep_ratio:.3f}<0.70")

    tuned_keep_reasons = []
    tuned_reject_reasons = []
    tuned_kept = False
    cand_diff = None
    cand_ratio = None
    if cand_p2 is not None and cand_p10 is not None:
        cand_diff = float(cand_p2 - cand_p10)
        cand_ratio = float(cand_p2 / max(cand_p10, 1e-6))

    if str(policy_A_variant) in {"a_tuned", "a_tuned_v2"} and reject_reasons:
        if str(stage) != "post_ransac_a95":
            tuned_reject_reasons.append("tuned_guard_only_applies_to_post_ransac_a95")
        if cand_p2 is None or cand_p10 is None:
            tuned_reject_reasons.append("candidate_width_unavailable")
        else:
            if not (15.0 <= cand_p2 <= 18.2):
                tuned_reject_reasons.append(f"candidate_p2p98_not_in_plausible_band={cand_p2:.3f}mm")
            else:
                tuned_keep_reasons.append(f"candidate_p2p98_plausible={cand_p2:.3f}mm")
            if cand_p10 < 10.0:
                tuned_reject_reasons.append(f"candidate_p10p90_too_small={cand_p10:.3f}mm")
            else:
                tuned_keep_reasons.append(f"candidate_p10p90_plausible={cand_p10:.3f}mm")
            if cand_diff is not None:
                if cand_diff > 3.8:
                    tuned_reject_reasons.append(f"candidate_width_sensitivity_diff_too_large={cand_diff:.3f}mm")
                else:
                    tuned_keep_reasons.append(f"candidate_width_sensitivity_diff_ok={cand_diff:.3f}mm")
            if cand_ratio is not None:
                if cand_ratio > 1.32:
                    tuned_reject_reasons.append(f"candidate_width_sensitivity_ratio_too_large={cand_ratio:.3f}")
                else:
                    tuned_keep_reasons.append(f"candidate_width_sensitivity_ratio_ok={cand_ratio:.3f}")
        if keep_ratio is None:
            tuned_reject_reasons.append("candidate_keep_ratio_unavailable")
        elif keep_ratio < 0.80:
            tuned_reject_reasons.append(f"candidate_keep_ratio_too_low_for_tuned_keep={keep_ratio:.3f}")
        else:
            tuned_keep_reasons.append(f"candidate_keep_ratio_ok={keep_ratio:.3f}")
        if cand_largest_ratio is None:
            tuned_reject_reasons.append("candidate_largest_component_ratio_unavailable")
        elif cand_largest_ratio < 0.98:
            tuned_reject_reasons.append(f"candidate_largest_component_ratio_too_low={cand_largest_ratio:.3f}")
        else:
            tuned_keep_reasons.append(f"candidate_largest_component_ratio_ok={cand_largest_ratio:.3f}")

        if not tuned_reject_reasons:
            tuned_kept = True
            reject_reasons = []

    info.update({
        "evaluated": True,
        "reason": (
            "candidate kept by tuned residual Policy A guard"
            if tuned_kept
            else ("overprune detected; candidate rejected" if reject_reasons else "candidate accepted by residual Policy A dry-run guard")
        ),
        "before_width_p2p98_mm": before_p2,
        "candidate_width_p2p98_mm": cand_p2,
        "before_width_p5p95_mm": before_p5,
        "candidate_width_p5p95_mm": cand_p5,
        "before_width_p10p90_mm": before_p10,
        "candidate_width_p10p90_mm": cand_p10,
        "candidate_valid_keep_ratio": keep_ratio,
        "before_width_snapshot": before,
        "candidate_width_snapshot": cand,
        "overprune_detected": bool(reject_reasons),
        "reject_candidate": bool(reject_reasons),
        "reject_reasons": reject_reasons,
        "tuned_keep_reasons": tuned_keep_reasons,
        "tuned_reject_reasons": tuned_reject_reasons,
        "policy_A_variant": str(policy_A_variant),
        "policy_A_triggered": bool(reject_reasons),
        "policy_A_candidate_rejected": bool(reject_reasons),
        "policy_A_candidate_kept_by_tuned_guard": bool(tuned_kept),
        "policy_A_candidate_kept_by_tuned_v2_guard": bool(tuned_kept and str(policy_A_variant) == "a_tuned_v2"),
        "policy_A_reject_reason": "; ".join(reject_reasons) if reject_reasons else None,
        "policy_A_keep_reason": "; ".join(tuned_keep_reasons) if tuned_kept else None,
        "candidate_width_mm": cand_p2,
        "pre_a95_width_mm": before_p2 if str(stage) == "post_ransac_a95" else None,
        "post_a95_width_mm": cand_p2 if str(stage) == "post_ransac_a95" else None,
        "p2_p98_width_mm": cand_p2,
        "p10_p90_width_mm": cand_p10,
        "p2_p98_over_p10_p90_ratio": cand_ratio,
        "p2_p98_minus_p10_p90_mm": cand_diff,
        "largest_component_ratio": cand_largest_ratio,
        "initial_mask_reliability": "not_unreliable_by_candidate_component_guard",
        "A_after_width_plausible": bool(tuned_kept),
    })
    return _jsonable_for_residual_diagnostics(info)


def _evaluate_residual_policy_ab_safe_interaction(
    *,
    residual_policy_mode: str,
    residual_policy_A_info: dict | None,
    current_width_m: float | None,
    initial_mask_quality: dict | None,
):
    """
    AB_SAFE 専用の相互作用ガードを評価する．

    A が post-RANSAC 等の候補を過削りとしてrejectし，A後の幅がすでに
    plausibileなら，Bのrobust幅置換で上書きしない．GTやcase番号は使わない．
    """
    mode = str(residual_policy_mode).upper()
    ainfo = residual_policy_A_info or {}
    current_width_mm = None if current_width_m is None else float(current_width_m) * 1000.0
    out = {
        "mode": mode,
        "enabled": mode in {"AB_SAFE", "AB_SAFE_A_TUNED", "AB_SAFE_A_TUNED_V2"},
        "A_then_B_overwrite_prevented": False,
        "A_after_width_plausible": False,
        "A_after_width_sensitivity": None,
        "A_triggered": bool(ainfo.get("reject_candidate", False)),
        "current_width_mm": current_width_mm,
        "notes": [],
        "skip_policy_B": False,
        "skip_reason": None,
        "plausibility_reject_reasons": [],
    }
    if mode not in {"AB_SAFE", "AB_SAFE_A_TUNED", "AB_SAFE_A_TUNED_V2"}:
        out["notes"].append("not AB_SAFE mode")
        return _jsonable_for_residual_diagnostics(out)

    if not bool(ainfo.get("reject_candidate", False)):
        out["notes"].append("Policy A did not reject a candidate; allow Policy B evaluation")
        return _jsonable_for_residual_diagnostics(out)

    p2 = _float_or_none(ainfo.get("before_width_p2p98_mm"))
    p10 = _float_or_none(ainfo.get("before_width_p10p90_mm"))
    before_snap = ainfo.get("before_width_snapshot") if isinstance(ainfo, dict) else None
    largest_ratio = None
    valid_depth_count = None
    if isinstance(before_snap, dict):
        largest_ratio = _float_or_none((before_snap.get("mask") or {}).get("largest_component_ratio"))
        valid_depth_count = _float_or_none(before_snap.get("valid_depth_pixel_count"))

    imq = initial_mask_quality or {}
    initial_unreliable = bool(imq.get("initial_mask_unreliable", False))
    reject_reasons = []
    sensitivity = {
        "p2p98_width_mm": p2,
        "p10p90_width_mm": p10,
        "p2_minus_p10_mm": None,
        "p2_over_p10": None,
        "largest_component_ratio": largest_ratio,
        "valid_depth_count": valid_depth_count,
        "initial_mask_unreliable": initial_unreliable,
    }

    if p2 is None or p10 is None:
        reject_reasons.append("A_after_width_candidates_unavailable")
    else:
        diff = float(p2 - p10)
        ratio = float(p2 / max(p10, 1e-6))
        sensitivity["p2_minus_p10_mm"] = diff
        sensitivity["p2_over_p10"] = ratio
        if not (12.0 <= p2 <= 35.0):
            reject_reasons.append(f"A_after_p2p98_width_out_of_plausible_range={p2:.3f}mm")
        if p10 < 10.0:
            reject_reasons.append(f"A_after_p10p90_too_small={p10:.3f}mm")
        if diff > 3.8:
            reject_reasons.append(f"A_after_width_sensitivity_diff_too_large={diff:.3f}mm")
        if ratio > 1.28:
            reject_reasons.append(f"A_after_width_sensitivity_ratio_too_large={ratio:.3f}")

    if largest_ratio is None or largest_ratio < 0.98:
        reject_reasons.append(f"A_after_largest_component_ratio_too_low={largest_ratio}")
    if initial_unreliable:
        reject_reasons.append("initial_mask_quality_unreliable")

    plausible = not reject_reasons
    out.update({
        "A_after_width_plausible": bool(plausible),
        "A_after_width_sensitivity": sensitivity,
        "plausibility_reject_reasons": reject_reasons,
    })
    if plausible:
        out.update({
            "A_then_B_overwrite_prevented": True,
            "skip_policy_B": True,
            "skip_reason": "Policy A rejected an over-prune candidate and A-after width is plausible; skip Policy B to avoid overwriting A result",
        })
        out["notes"].append("Policy B skipped by AB_SAFE interaction guard")
    else:
        out["notes"].append("A fired, but A-after width still looks sensitive or unreliable; allow Policy B evaluation")

    return _jsonable_for_residual_diagnostics(out)


def _evaluate_residual_policy_b_width(
    *,
    enabled: bool,
    current_width_m: float | None,
    mask01: np.ndarray,
    depth_masked: np.ndarray,
    intr,
    depth_scale: float,
    refine_info: dict | None,
    column_info: dict | None,
    initial_mask_quality: dict | None,
    clean_rectangle_override_adopted: bool,
    residual_policy_mode: str = "baseline",
    residual_policy_interaction: dict | None = None,
):
    current_width_mm = None if current_width_m is None else float(current_width_m) * 1000.0
    out = {
        "enabled": bool(enabled),
        "evaluated": False,
        "robust_width_candidate": None,
        "current_width_mm": current_width_mm,
        "p2p98_width_mm": None,
        "p5p95_width_mm": None,
        "p10p90_width_mm": None,
        "largest_component_p5p95_width_mm": None,
        "largest_component_p10p90_width_mm": None,
        "selected_robust_width_mm": None,
        "robust_width_adopted": False,
        "triggered": False,
        "skipped_due_to_A_plausible": False,
        "skip_reason": None,
        "residual_policy_interaction": residual_policy_interaction,
        "adopt_reasons": [],
        "reject_reasons": ["disabled"] if not enabled else [],
        "policy_B_candidate_kept_by_tuned_v2_guard": False,
        "tuned_v2_keep_reasons": [],
        "tuned_v2_reject_reasons": [],
    }
    if not enabled:
        return out, current_width_m
    interaction = residual_policy_interaction or {}
    if bool(interaction.get("skip_policy_B", False)):
        out.update({
            "evaluated": False,
            "skipped_due_to_A_plausible": True,
            "skip_reason": interaction.get("skip_reason"),
            "reject_reasons": ["skipped_due_to_A_plausible"],
            "reason": "Policy B skipped by AB_SAFE interaction guard",
        })
        return _jsonable_for_residual_diagnostics(out), current_width_m

    snap = _residual_policy_width_snapshot(
        mask01, depth_masked, intr, depth_scale, refine_info, column_info
    )
    axis_c = snap.get("axis_width_candidates") or {}
    largest_c = snap.get("largest_component_axis_width_candidates") or {}
    p2 = _width_mm_from_candidates(axis_c, "p2_p98")
    p5 = _width_mm_from_candidates(axis_c, "p5_p95")
    p10 = _width_mm_from_candidates(axis_c, "p10_p90")
    lc_p5 = _width_mm_from_candidates(largest_c, "p5_p95")
    lc_p10 = _width_mm_from_candidates(largest_c, "p10_p90")
    largest_ratio = _float_or_none((snap.get("mask") or {}).get("largest_component_ratio"))

    out.update({
        "evaluated": True,
        "p2p98_width_mm": p2,
        "p5p95_width_mm": p5,
        "p10p90_width_mm": p10,
        "largest_component_p5p95_width_mm": lc_p5,
        "largest_component_p10p90_width_mm": lc_p10,
        "width_snapshot": snap,
        "clean_rectangle_override_adopted": bool(clean_rectangle_override_adopted),
        "largest_component_ratio": largest_ratio,
    })

    reject_reasons = []
    adopt_reasons = []
    imq = initial_mask_quality or {}
    if bool(imq.get("initial_mask_unreliable", False)):
        reject_reasons.append("initial_mask_quality_unreliable")
    else:
        adopt_reasons.append("initial_mask_quality_reliable")

    if largest_ratio is None or largest_ratio < 0.98:
        reject_reasons.append(f"largest_component_ratio_too_low={largest_ratio}")
    else:
        adopt_reasons.append(f"largest_component_ratio_ok={largest_ratio:.3f}")

    if p2 is None or p10 is None or current_width_mm is None:
        reject_reasons.append("required_width_candidates_unavailable")
    else:
        diff = float(p2 - p10)
        ratio = float(p2 / max(p10, 1e-6))
        out["p2_minus_p10_mm"] = diff
        out["p2_over_p10"] = ratio
        if bool(clean_rectangle_override_adopted):
            sensitivity_ok = current_width_mm >= 18.0 and diff >= 2.2 and ratio >= 1.14
            sensitivity_rule = "clean_rectangle_override_current_ge_18_and_diff_ge_2p2_ratio_ge_1p14"
        else:
            sensitivity_ok = current_width_mm <= 18.5 and diff >= 3.2 and ratio >= 1.23
            sensitivity_rule = "normal_current_le_18p5_and_diff_ge_3p2_ratio_ge_1p23"

        out["sensitivity_rule"] = sensitivity_rule
        if sensitivity_ok:
            adopt_reasons.append(f"p2_p10_sensitivity={diff:.3f}mm ratio={ratio:.3f} rule={sensitivity_rule}")
        else:
            reject_reasons.append(f"p2_p10_sensitivity_rule_failed={diff:.3f}mm ratio={ratio:.3f} rule={sensitivity_rule}")

        min_robust_width = 10.0 if bool(clean_rectangle_override_adopted) else 12.5
        if p10 < min_robust_width:
            reject_reasons.append(f"p10p90_too_small={p10:.3f}mm<min={min_robust_width:.3f}mm")
        else:
            adopt_reasons.append(f"p10p90_plausible={p10:.3f}mm")

    robust_candidate_name = "p10_p90"
    robust_width_mm = p10
    if lc_p10 is not None and p10 is not None and abs(lc_p10 - p10) <= 0.25:
        robust_candidate_name = "largest_component_p10_p90"
        robust_width_mm = lc_p10

    if robust_width_mm is None:
        reject_reasons.append("robust_width_candidate_unavailable")
    elif current_width_mm is not None and robust_width_mm >= current_width_mm:
        reject_reasons.append("robust_width_not_smaller_than_current")

    mode = str(residual_policy_mode).upper()
    tuned_v2_keep_reasons = []
    tuned_v2_reject_reasons = []
    tuned_v2_kept = False
    if mode == "AB_SAFE_A_TUNED_V2" and reject_reasons and robust_width_mm is not None:
        a_triggered = bool((residual_policy_interaction or {}).get("A_triggered", False))
        robust_delta = None if current_width_mm is None else float(current_width_mm - robust_width_mm)
        if a_triggered:
            tuned_v2_reject_reasons.append("policy_A_already_triggered_or_skipped_B")
        else:
            tuned_v2_keep_reasons.append("policy_A_did_not_trigger")
        if current_width_mm is None:
            tuned_v2_reject_reasons.append("current_width_unavailable")
        elif current_width_mm > 13.5:
            tuned_v2_reject_reasons.append(f"current_width_too_large_for_small_width_v2={current_width_mm:.3f}mm")
        else:
            tuned_v2_keep_reasons.append(f"current_width_small_band={current_width_mm:.3f}mm")
        if robust_width_mm < 10.0:
            tuned_v2_reject_reasons.append(f"robust_width_too_small_for_v2={robust_width_mm:.3f}mm")
        else:
            tuned_v2_keep_reasons.append(f"robust_width_plausible_for_v2={robust_width_mm:.3f}mm")
        if robust_delta is None:
            tuned_v2_reject_reasons.append("robust_delta_unavailable")
        elif not (1.5 <= robust_delta <= 3.0):
            tuned_v2_reject_reasons.append(f"robust_delta_not_in_small_width_v2_band={robust_delta:.3f}mm")
        else:
            tuned_v2_keep_reasons.append(f"robust_delta_small_width_v2_ok={robust_delta:.3f}mm")
        if p2 is None or p10 is None:
            tuned_v2_reject_reasons.append("p2_or_p10_unavailable")
        else:
            diff = float(p2 - p10)
            ratio = float(p2 / max(p10, 1e-6))
            if diff > 2.65:
                tuned_v2_reject_reasons.append(f"small_width_v2_sensitivity_diff_too_large={diff:.3f}mm")
            else:
                tuned_v2_keep_reasons.append(f"small_width_v2_sensitivity_diff_ok={diff:.3f}mm")
            if ratio > 1.245:
                tuned_v2_reject_reasons.append(f"small_width_v2_sensitivity_ratio_too_large={ratio:.3f}")
            else:
                tuned_v2_keep_reasons.append(f"small_width_v2_sensitivity_ratio_ok={ratio:.3f}")
        if largest_ratio is None:
            tuned_v2_reject_reasons.append("largest_component_ratio_unavailable")
        elif largest_ratio < 0.98:
            tuned_v2_reject_reasons.append(f"largest_component_ratio_too_low_for_v2={largest_ratio:.3f}")
        else:
            tuned_v2_keep_reasons.append(f"largest_component_ratio_ok_for_v2={largest_ratio:.3f}")
        if bool(imq.get("initial_mask_unreliable", False)):
            tuned_v2_reject_reasons.append("initial_mask_quality_unreliable_for_v2")
        else:
            tuned_v2_keep_reasons.append("initial_mask_quality_reliable_for_v2")

        blocking_reasons = [
            r for r in tuned_v2_reject_reasons
            if not (
                r.startswith("p2_p10_sensitivity_rule_failed=")
                or r.startswith("p10p90_too_small=")
            )
        ]
        if not blocking_reasons:
            tuned_v2_kept = True
            reject_reasons = []

    out["robust_width_candidate"] = robust_candidate_name
    out["selected_robust_width_mm"] = robust_width_mm
    out["policy_B_candidate_kept_by_tuned_v2_guard"] = bool(tuned_v2_kept)
    out["tuned_v2_keep_reasons"] = tuned_v2_keep_reasons
    out["tuned_v2_reject_reasons"] = tuned_v2_reject_reasons
    out["adopt_reasons"] = adopt_reasons
    out["reject_reasons"] = reject_reasons
    if not reject_reasons and robust_width_mm is not None:
        out["robust_width_adopted"] = True
        out["triggered"] = True
        out["reason"] = (
            "robust width adopted by AB_SAFE_A_TUNED_V2 small-width stability guard"
            if tuned_v2_kept
            else "robust width adopted in residual policy experiment mode"
        )
        return _jsonable_for_residual_diagnostics(out), float(robust_width_mm) / 1000.0

    out["reason"] = "robust width rejected in residual policy experiment mode"
    return _jsonable_for_residual_diagnostics(out), current_width_m


def _summarize_residual_guard_info(column_info: dict | None):
    cinfo = column_info or {}
    ransac = cinfo.get("ransac_spine_plane") if isinstance(cinfo, dict) else None
    post_a95 = cinfo.get("post_ransac_plane_a95_column_prune") if isinstance(cinfo, dict) else None
    one_sided = cinfo.get("one_sided_post_final_column_prune") if isinstance(cinfo, dict) else None
    final_clip = cinfo.get("final_t_width_clip") if isinstance(cinfo, dict) else None
    return _jsonable_for_residual_diagnostics({
        "ransac_spine_plane": None if not isinstance(ransac, dict) else {
            "used": ransac.get("used"),
            "guard_reverted": ransac.get("guard_reverted"),
            "reason": ransac.get("reason"),
            "ransac_guard_reasons": ransac.get("ransac_guard_reasons"),
            "valid_count_before": ransac.get("valid_count_before"),
            "valid_count_after": ransac.get("valid_count_after"),
            "candidate_valid_count_after": ransac.get("candidate_valid_count_after"),
            "keep_ratio": ransac.get("keep_ratio"),
            "candidate_keep_ratio": ransac.get("candidate_keep_ratio"),
            "removed_ratio": ransac.get("removed_ratio"),
            "candidate_removed_ratio": ransac.get("candidate_removed_ratio"),
            "distance_m_p90": ransac.get("distance_m_p90"),
            "distance_m_p95": ransac.get("distance_m_p95"),
            "reference_point_ratio": ransac.get("reference_point_ratio"),
        },
        "post_ransac_plane_a95": None if not isinstance(post_a95, dict) else {
            "used": post_a95.get("used"),
            "guard_reverted": post_a95.get("guard_reverted"),
            "reason": post_a95.get("reason"),
            "valid_count_before": post_a95.get("valid_count_before"),
            "valid_count_after": post_a95.get("valid_count_after"),
            "candidate_valid_count_after": post_a95.get("candidate_valid_count_after"),
            "valid_keep_ratio": post_a95.get("valid_keep_ratio"),
            "candidate_valid_keep_ratio": post_a95.get("candidate_valid_keep_ratio"),
            "removed_ratio": post_a95.get("removed_ratio"),
            "candidate_removed_ratio": post_a95.get("candidate_removed_ratio"),
            "max_length_m": post_a95.get("max_length_m"),
            "length_threshold_m": post_a95.get("length_threshold_m"),
            "one_sided_previous_side_filter": post_a95.get("one_sided_previous_side_filter"),
        },
        "post_column_prune": None if not isinstance(one_sided, dict) else {
            "used": one_sided.get("used"),
            "guard_reverted": one_sided.get("guard_reverted"),
            "reason": one_sided.get("reason"),
            "remove_side": one_sided.get("remove_side"),
            "protected_side": one_sided.get("protected_side"),
            "valid_count_before": one_sided.get("valid_count_before"),
            "valid_count_after": one_sided.get("valid_count_after"),
            "valid_keep_ratio": one_sided.get("valid_keep_ratio"),
            "removed_count": one_sided.get("removed_count"),
            "removed_ratio": one_sided.get("removed_ratio"),
        },
        "final_t_width_clip": None if not isinstance(final_clip, dict) else {
            "used": final_clip.get("used"),
            "reason": final_clip.get("reason"),
            "area_before": final_clip.get("area_before"),
            "area_after": final_clip.get("area_after"),
            "keep_ratio": final_clip.get("keep_ratio"),
            "width_px": final_clip.get("width_px"),
            "half_width_px": final_clip.get("half_width_px"),
        },
    })


def save_residual_stage_diagnostics_debug(
    shot_dir: Path,
    stem: str,
    stage_diagnostics: list[dict],
    stage_diffs: list[dict],
    guard_diagnostics: dict | None = None,
    final_width_info: dict | None = None,
):
    debug_dir = Path(shot_dir) / "debug_residual_stage_diagnostics"
    debug_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage_diagnostics": stage_diagnostics,
        "stage_diffs": stage_diffs,
        "guard_diagnostics": guard_diagnostics or {},
        "final_width_info": final_width_info or {},
    }
    save_json(debug_dir / f"{stem}_stage_width_diagnostics.json", payload)

    table_path = debug_dir / f"{stem}_stage_width_table.csv"
    fields = [
        "stage_name",
        "mask_area_px",
        "valid_depth_pixel_count",
        "component_count",
        "largest_component_ratio",
        "axis_source",
        "axis_p2_p98_width_mm",
        "axis_p5_p95_width_mm",
        "axis_p10_p90_width_mm",
        "axis_p25_p75_width_mm",
        "pca_width_mm",
        "adopted_width_shadow_mm_if_stage_final",
        "width_sensitivity_p2p98_minus_p10p90_mm",
        "depth_median_m",
    ]
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for diag in stage_diagnostics:
            widths = diag.get("axis_width_candidates") or {}
            mask_info = diag.get("mask") or {}
            writer.writerow({
                "stage_name": diag.get("stage_name"),
                "mask_area_px": mask_info.get("area_px"),
                "valid_depth_pixel_count": diag.get("valid_depth_pixel_count"),
                "component_count": mask_info.get("component_count"),
                "largest_component_ratio": mask_info.get("largest_component_ratio"),
                "axis_source": diag.get("axis_source"),
                "axis_p2_p98_width_mm": (widths.get("p2_p98") or {}).get("width_mm"),
                "axis_p5_p95_width_mm": (widths.get("p5_p95") or {}).get("width_mm"),
                "axis_p10_p90_width_mm": (widths.get("p10_p90") or {}).get("width_mm"),
                "axis_p25_p75_width_mm": (widths.get("p25_p75") or {}).get("width_mm"),
                "pca_width_mm": diag.get("pca_width_mm"),
                "adopted_width_shadow_mm_if_stage_final": diag.get("adopted_width_shadow_mm_if_stage_final"),
                "width_sensitivity_p2p98_minus_p10p90_mm": diag.get("width_sensitivity_p2p98_minus_p10p90_mm"),
                "depth_median_m": (diag.get("depth_stats") or {}).get("median_m"),
            })
    return {"debug_dir": str(debug_dir), "json": str(debug_dir / f"{stem}_stage_width_diagnostics.json"), "csv": str(table_path)}


def _run_recognition_core_like_offline(
    *,
    query: str,
    shot_dir: Path,
    color_np: np.ndarray,
    depth_np_u16: np.ndarray,
    intr,
    depth_scale: float,
    sam_device: str,
    encoder_path: str,
    decoder_path: str,
    interactive: bool,
    result_suffix: str = "",
    use_persistent_runtime: bool = True,
    sam_pts_side: tuple[int, int] = (32, 8),
    sam_decoder_k_keep: int = 1,
    sam_target_len: int = 768,
    depth_merge_tolerance_raw: int = 30,
    show_pointcloud_gui: bool = False,
    save_pointcloud_debug: bool = False,
    save_step_by_step_pointcloud_debug: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, Path]:
    """
    online/offline共通の認識コア．

    重要な仕様:
      1. SAM2とOCRを並列に実行する．
      2. 対象マスク選択後，長方形判定より先にDepth中央値±3cm補正を行う．
      3. 長方形でない場合だけOCR軸推定を行う．ただしOCR帯候補ではmask01を直接更新しない．
      4. 最終的な側面除去はDepth補正後の点群列長さフィルタで行う．
      5. online/offlineで同じ後処理値を使う．
    """
    shot_dir = Path(shot_dir)
    debug_prefix = "[OFFLINE]" if result_suffix == "_offline" else ""

    step_projection_infos = []

    def _save_step_projection(stage_id: str, stage_name: str, mask_stage, depth_stage, note: str = "", pts_stage=None):
        if not bool(save_step_by_step_pointcloud_debug):
            return None
        try:
            info = save_step_by_step_pointcloud_projection_debug(
                shot_dir=shot_dir,
                color_np=color_np,
                intr=intr,
                depth_scale=depth_scale,
                stage_id=stage_id,
                stage_name=stage_name,
                mask01=mask_stage,
                depth_masked=depth_stage,
                pts_f=pts_stage,
                stem=f"mask{sel_idx}{result_suffix}" if "sel_idx" in locals() else f"step{result_suffix}",
                note=note,
                save_ply=False,
            )
            step_projection_infos.append(info)
            return info
        except Exception as e:
            err = {
                "stage_id": str(stage_id),
                "stage_name": str(stage_name),
                "error": str(e),
            }
            step_projection_infos.append(err)
            traceback.print_exc()
            return err

    # ===== 1) OCRを非同期開始 =====
    ocr_start = time.perf_counter()
    ocr_proc = start_ocr_subprocess(shot_dir)
    print(f"[PARALLEL]{debug_prefix} OCR subprocess started.")

    # ===== 2) SAM2実行 =====
    sam_start = time.perf_counter()
    sam_runner = _get_sam_runner_compat(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        sam_device=sam_device,
        sam_pts_side=sam_pts_side,
        sam_decoder_k_keep=sam_decoder_k_keep,
        sam_target_len=sam_target_len,
        use_cache=bool(use_persistent_runtime),
    )
    rgb_pil = Image.fromarray(cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))
    stage_cfg = _make_stage_save_cfg_compat(shot_dir)
    masks, sam_data = _infer_masks_compat(
        sam_runner,
        rgb_pil,
        stage_cfg,
        depth_np_u16=depth_np_u16,
        depth_merge_tolerance_raw=depth_merge_tolerance_raw,
    )
    sam_end = time.perf_counter()
    print(f"[TIME]{debug_prefix} SAM total     : {sam_end - sam_start:.3f} sec")

    # ===== 3) OCR終了待ち =====
    ocr_stdout = wait_ocr_subprocess(ocr_proc, timeout=120.0)
    ocr_end = time.perf_counter()
    if ocr_stdout.strip():
        print(ocr_stdout, end="" if ocr_stdout.endswith("\n") else "\n")
    print(f"[TIME]{debug_prefix} OCR wall      : {ocr_end - ocr_start:.3f} sec")

    # ===== 4) SAM2+OCRで対象マスク選択 =====
    merge_start = time.perf_counter()
    merged = merge_ocr_and_masks(
        query=query,
        masks=masks,
        shot_dir=shot_dir,
        interactive=interactive,
        threshold=40,
    )
    sel_idx = int(merged["sel_idx"])
    mask01 = (np.asarray(merged["mask01"]) > 0).astype(np.uint8)
    residual_policy_mode = _get_residual_policy_mode()
    residual_policy_A_info = {
        "enabled": _residual_policy_has_a(residual_policy_mode),
        "evaluated": False,
        "overprune_detected": False,
        "reject_candidate": False,
        "reason": "not evaluated",
    }
    residual_policy_B_info = {
        "enabled": _residual_policy_has_b(residual_policy_mode),
        "evaluated": False,
        "robust_width_adopted": False,
        "triggered": False,
        "skipped_due_to_A_plausible": False,
        "reason": "not evaluated",
    }
    residual_policy_interaction_info = {
        "mode": residual_policy_mode,
        "enabled": residual_policy_mode in {"AB_SAFE", "AB_SAFE_A_TUNED", "AB_SAFE_A_TUNED_V2"},
        "A_then_B_overwrite_prevented": False,
        "A_after_width_plausible": False,
        "A_after_width_sensitivity": None,
        "notes": ["not evaluated"],
    }
    residual_stage_diagnostics = []
    residual_stage_diffs = []

    def _record_residual_stage(stage_name, mask_stage, depth_stage, *, refine_stage=None, column_stage=None, extra_info=None):
        try:
            diag = build_stage_width_diagnostics(
                stage_name=stage_name,
                mask01=mask_stage,
                depth_masked=depth_stage,
                intr=intr,
                depth_scale=depth_scale,
                refine_info=refine_stage if refine_stage is not None else refine_info,
                column_info=column_stage if column_stage is not None else column_info,
                extra_info=extra_info,
            )
            if residual_stage_diagnostics:
                diff = _stage_width_diff(residual_stage_diagnostics[-1], diag)
                if diff is not None:
                    residual_stage_diffs.append(diff)
            residual_stage_diagnostics.append(diag)
        except Exception as e:
            residual_stage_diagnostics.append({
                "stage_name": str(stage_name),
                "diagnostic_error": str(e),
            })

    # ===== 5) Depth中央値±3cm補正を長方形判定より先に実行 =====
    mask01_before_depth_filter = mask01.copy()
    _save_step_projection(
        "00",
        "selected_sam_mask_before_depth_prefilter",
        mask01_before_depth_filter,
        depth_np_u16,
        note="SAM2+OCRで選択された対象マスクを，Depth補正前に点群化してRGBへ再投影した画像．",
    )

    # 原因解析用：Depth補正前の「選択マスク領域の有効Depth点」を色分け保存する．
    # selected_overlay は保存タイミングによって意味が変わりやすいため，
    # ここでは明示的に before_depth_prefilter / after_depth_prefilter を分ける．
    selected_depth_points_before_info = _save_mask_depth_points_debug_image(
        shot_dir=shot_dir,
        color_np=color_np,
        depth_np_u16=depth_np_u16,
        mask01=mask01_before_depth_filter,
        stem=f"mask{sel_idx}{result_suffix}_selected_before_depth_prefilter",
        title="Selected mask depth points BEFORE depth prefilter",
    )

    # ===== 5.5) Depth補正の基準を「選択マスク全体の中央値」から
    #             「選択されたOCR文字領域内の中央値」へ変更 =====
    # ここではマスク更新は行わず，OCR polygonだけをDepth基準領域として使う．
    # OCR polygonが取得できない，または選択マスクとの交差が小さい場合は従来通り選択マスク全体へfallbackする．
    depth_anchor_candidate_mask01, depth_anchor_refine_info = refine_mask_by_ocr_axis_band(
        mask01=mask01_before_depth_filter,
        merged=merged,
        image_shape=color_np.shape[:2],
        shot_dir=shot_dir,
        query=query,
        mask_width_ratio=1.05,
        min_keep_ratio=0.85,
        use_ocr_short_width=False,
        ocr_short_to_half_width_scale=1.35,
        min_ocr_half_width_px=28.0,
        suppress_side_protrusions=False,
        profile_bin_size_px=8.0,
        profile_width_margin=1.15,
        min_profile_half_width_px=16.0,
        max_profile_shrink_ratio=0.35,
        return_info=True,
    )
    ocr_depth_reference_mask01, ocr_depth_reference_info = _make_depth_reference_mask_from_ocr_info(
        depth_anchor_refine_info,
        color_np.shape[:2],
        selected_mask01=mask01_before_depth_filter,
        min_intersection_px=30,
    )
    initial_mask_quality_info = build_initial_mask_quality(
        mask01=mask01_before_depth_filter,
        refine_info=depth_anchor_refine_info,
        image_shape=color_np.shape[:2],
    )
    if ocr_depth_reference_mask01 is not None:
        try:
            _save_points_and_overlay(
                rgb_pil,
                [ocr_depth_reference_mask01],
                shot_dir,
                f"mask{sel_idx}{result_suffix}_ocr_depth_reference",
                draw_ids=False,
            )
        except Exception:
            pass

    depth_prefilter_stem = f"mask{sel_idx}{result_suffix}_depth_prefilter"
    depth_masked, depth_filter_detail_info = save_masked_and_cropped(
        color_np,
        depth_np_u16,
        mask01,
        shot_dir,
        depth_prefilter_stem,
        z_tolerance_raw=int(depth_merge_tolerance_raw),
        depth_reference_mask01=ocr_depth_reference_mask01,
        depth_reference_name="selected_ocr_polygon_intersection_selected_mask",
        return_info=True,
    )
    mask01 = ((mask01 > 0) & (depth_masked > 0)).astype(np.uint8)
    _save_step_projection(
        "01",
        "after_depth_median_pm3cm_prefilter",
        mask01,
        depth_masked,
        note="選択マスク内のDepth中央値±3cmで外れ値除去した直後．",
    )
    _record_residual_stage(
        "after_depth_prefilter",
        mask01,
        depth_masked,
        refine_stage=depth_anchor_refine_info,
        column_stage={},
        extra_info={"stage_id": "01", "depth_filter_detail": depth_filter_detail_info},
    )

    # 原因解析用：Depth補正後に残った有効Depth点を色分け保存する．
    selected_depth_points_after_info = _save_mask_depth_points_debug_image(
        shot_dir=shot_dir,
        color_np=color_np,
        depth_np_u16=depth_masked,
        mask01=mask01,
        stem=f"mask{sel_idx}{result_suffix}_selected_after_depth_prefilter",
        title="Selected mask depth points AFTER depth prefilter",
    )

    # ===== 5.7) Depthで削れた背表紙内部を，OCR軸中心帯の中だけ補完する =====
    # Depth閾値は側面除去には有効だが，背表紙面も穴あき・ガタガタになりやすい．
    # そこで，Depth補正前のSAM選択マスクを上限候補として，OCR文字軸に沿った細い背表紙中心帯だけ復活させる．
    # 復活画素のDepthには元の異常Depthを使わず，行(s-bin)ごとの既存Depth中央値を割り当てる．
    mask01_before_spine_completion = mask01.copy()
    depth_masked_before_spine_completion = depth_masked.copy()
    mask01, depth_masked, spine_completion_info = complete_spine_band_after_depth_prefilter(
        mask01_depth_filtered=mask01_before_spine_completion,
        depth_masked_depth_filtered=depth_masked_before_spine_completion,
        selected_mask01_before_depth=mask01_before_depth_filter,
        depth_np_u16=depth_np_u16,
        refine_info=depth_anchor_refine_info,
        image_shape=color_np.shape[:2],
        s_bin_size_px=5.0,
        ocr_short_width_scale=1.45,
        min_spine_width_px=30.0,
        max_spine_width_px=58.0,
        extra_margin_px=2.0,
        s_range_percentiles=(1.0, 99.0),
        min_candidate_pixels=80,
        min_keep_ratio_after_completion=0.20,
        max_area_growth_ratio=1.80,
        fill_depth_by_row_median=True,
    )

    try:
        _save_points_and_overlay(
            rgb_pil,
            [mask01],
            shot_dir,
            f"mask{sel_idx}{result_suffix}_after_depth_prefilter_spine_completed",
            draw_ids=False,
        )
    except Exception as e:
        spine_completion_info["overlay_save_error"] = str(e)

    selected_depth_points_after_completion_info = _save_mask_depth_points_debug_image(
        shot_dir=shot_dir,
        color_np=color_np,
        depth_np_u16=depth_masked,
        mask01=mask01,
        stem=f"mask{sel_idx}{result_suffix}_selected_after_depth_prefilter_spine_completed",
        title="Selected mask depth points AFTER depth prefilter + spine completion",
    )
    _save_step_projection(
        "02",
        "after_depth_prefilter_spine_completion",
        mask01,
        depth_masked,
        note="Depth補正後に，OCR軸中心帯だけ背表紙内部を補完した後．",
    )
    _record_residual_stage(
        "after_depth_prefilter_spine_completion",
        mask01,
        depth_masked,
        refine_stage=depth_anchor_refine_info,
        column_stage={},
        extra_info={"stage_id": "02", "spine_completion": spine_completion_info},
    )

    depth_prefilter_info = {
        "enabled": True,
        "reason": "Depth median +/- tolerance is applied before rectangularity judgement.",
        "mask_area_before_depth_filter_px": int(mask01_before_depth_filter.sum()),
        "mask_area_after_depth_filter_px": int(mask01.sum()),
        "valid_depth_pixel_count": int(np.count_nonzero(depth_masked > 0)),
        "depth_prefilter_stem": depth_prefilter_stem,
        "z_tolerance_raw": int(depth_merge_tolerance_raw),
        "depth_reference_policy": "OCR文字領域内のDepth中央値を優先し，取得できない場合のみ選択マスク全体へfallbackする．",
        "ocr_depth_reference": ocr_depth_reference_info,
        "depth_filter_detail": depth_filter_detail_info,
        "selected_depth_points_before_prefilter": selected_depth_points_before_info,
        "selected_depth_points_after_prefilter": selected_depth_points_after_info,
        "spine_completion_after_depth_prefilter": spine_completion_info,
        "selected_depth_points_after_spine_completion": selected_depth_points_after_completion_info,
        "initial_mask_quality": initial_mask_quality_info,
    }
    save_json(shot_dir / f"mask{sel_idx}{result_suffix}_depth_prefilter_log.json", depth_prefilter_info)

    # ===== 6) Depth補正後マスクで長方形判定 =====
    shape_info = analyze_mask_rectangularity(
        mask01=mask01,
        image_shape=color_np.shape[:2],
        iou_threshold=0.82,
        extent_threshold=0.78,
        solidity_threshold=0.92,
        width_cv_threshold=0.28,
        width_max_median_threshold=1.55,
    )
    shape_info["depth_prefilter"] = depth_prefilter_info
    save_mask_rectangularity_debug(
        shot_dir=shot_dir,
        color_np=color_np,
        mask01=mask01,
        shape_info=shape_info,
        stem=f"mask{sel_idx}{result_suffix}_after_depth_prefilter",
    )
    _save_step_projection(
        "03",
        "rectangularity_input_after_depth_prefilter",
        mask01,
        depth_masked,
        note="長方形判定に入力されたmask/depth．",
    )
    needs_shape_refine = bool(shape_info.get("needs_refine", True))

    refine_info = {
        "used": False,
        "reason": "depth-filtered mask is clean rotated rectangle; OCR/column refine skipped",
        "shape_rectangularity": shape_info,
        "depth_prefilter": depth_prefilter_info,
    }
    column_info = {
        "used": False,
        "reason": "depth-filtered mask is clean rotated rectangle; column length refine skipped",
        "shape_rectangularity": shape_info,
        "depth_prefilter": depth_prefilter_info,
    }
    one_sided_column_prune_info = {
        "used": False,
        "reason": "not run; column length refinement was skipped",
        "algorithm": "one_sided_short_column_prune_after_final",
    }

    clean_rectangle_analysis_info = build_clean_rectangle_analysis(
        mask01=mask01,
        depth_masked=depth_masked,
        shape_info=shape_info,
        refine_info=depth_anchor_refine_info,
        intr=intr,
        depth_scale=depth_scale,
        clean_rectangle_skip_used=not needs_shape_refine,
    )
    clean_rectangle_shadow_info = {
        "evaluated": False,
        "reason": "clean rectangle skip not used",
        "shadow_candidate_not_adopted_reason": "shadow_only",
    }
    clean_rectangle_override_info = {
        "evaluated": False,
        "adopted": False,
        "reason": "clean rectangle skip not used",
        "reject_reasons": ["clean_rectangle_skip_not_used"],
        "adopt_reasons": [],
    }
    clean_rectangle_override_adopted = False

    # ===== 7) 長方形でない場合だけOCR軸を取得する．ただしmask01は更新しない =====
    if needs_shape_refine:
        print(f"[MASK SHAPE]{debug_prefix} irregular mask after depth prefilter -> enable OCR/column refinement")
        mask01_before_refine = mask01.copy()
        mask01_ocr_band_candidate, refine_info = refine_mask_by_ocr_axis_band(
            mask01=mask01,
            merged=merged,
            image_shape=color_np.shape[:2],
            shot_dir=shot_dir,
            query=query,
            mask_width_ratio=1.05,
            min_keep_ratio=0.85,
            use_ocr_short_width=False,
            ocr_short_to_half_width_scale=1.35,
            min_ocr_half_width_px=28.0,
            suppress_side_protrusions=False,
            profile_bin_size_px=8.0,
            profile_width_margin=1.15,
            min_profile_half_width_px=16.0,
            max_profile_shrink_ratio=0.35,
            return_info=True,
        )
        refine_info["shape_rectangularity"] = shape_info
        refine_info["depth_prefilter"] = depth_prefilter_info
        refine_info["mask_update_used"] = False
        refine_info["mask_update_reason"] = "OCR band candidate is debug/axis-only; final removal is performed by spine column length filter."

        save_mask_refine_debug(
            shot_dir=shot_dir,
            color_np=color_np,
            mask_before=mask01_before_refine,
            mask_after=mask01_ocr_band_candidate,
            merged=merged,
            refine_info=refine_info,
            stem=f"mask{sel_idx}{result_suffix}_ocr_band_candidate",
            query=query,
        )
        _save_step_projection(
            "04",
            "ocr_axis_band_candidate_not_applied",
            mask01_ocr_band_candidate,
            depth_masked,
            note="OCR軸帯候補．軸取得用・debug用であり，この段階では本流maskへは適用していない．",
        )
        # offline仕様に合わせる: OCR帯候補ではmask01/depth_maskedを更新しない．
        mask01 = mask01_before_refine.copy()
    else:
        print(f"[MASK SHAPE]{debug_prefix} depth-filtered mask is clean rotated rectangle -> skip OCR/column refinement")
        clean_rectangle_shadow_info = evaluate_clean_rectangle_shadow_candidate(
            shot_dir=shot_dir,
            color_np=color_np,
            mask01=mask01,
            depth_masked=depth_masked,
            merged=merged,
            query=query,
            refine_info_seed=depth_anchor_refine_info,
            shape_info=shape_info,
            intr=intr,
            depth_scale=depth_scale,
            result_suffix=result_suffix,
            sel_idx=sel_idx,
        )
        shadow_mask_runtime = clean_rectangle_shadow_info.pop("_shadow_mask01", None)
        shadow_depth_runtime = clean_rectangle_shadow_info.pop("_shadow_depth_masked", None)
        shadow_column_runtime = clean_rectangle_shadow_info.pop("_shadow_column_info", None)
        shadow_refine_runtime = clean_rectangle_shadow_info.pop("_shadow_refine_info", None)
        clean_rectangle_override_info = decide_clean_rectangle_override(
            clean_rectangle_shadow=clean_rectangle_shadow_info,
            initial_mask_quality=initial_mask_quality_info,
            clean_rectangle_analysis=clean_rectangle_analysis_info,
        )
        if clean_rectangle_override_info.get("adopted") and (
            shadow_mask_runtime is None
            or shadow_depth_runtime is None
            or not isinstance(shadow_column_runtime, dict)
            or not isinstance(shadow_refine_runtime, dict)
        ):
            clean_rectangle_override_info["adopted"] = False
            clean_rectangle_override_info["reason"] = "candidate_rejected"
            clean_rectangle_override_info.setdefault("reject_reasons", []).append("shadow_runtime_candidate_unavailable")

        if bool(clean_rectangle_override_info.get("adopted", False)):
            mask01 = (np.asarray(shadow_mask_runtime) > 0).astype(np.uint8)
            depth_masked = _apply_mask_to_depth_no_save(shadow_depth_runtime, mask01)
            refine_info = dict(shadow_refine_runtime)
            refine_info["shape_rectangularity"] = shape_info
            refine_info["depth_prefilter"] = depth_prefilter_info
            refine_info["clean_rectangle_override_adopted"] = True
            column_info = dict(shadow_column_runtime)
            column_info["shape_rectangularity"] = shape_info
            column_info["depth_prefilter"] = depth_prefilter_info
            column_info["clean_rectangle_override_adopted"] = True
            one_sided_column_prune_info = column_info.get("one_sided_post_final_column_prune", one_sided_column_prune_info)
            clean_rectangle_override_adopted = True
            clean_rectangle_shadow_info["shadow_candidate_adopted_as_final"] = True
            clean_rectangle_shadow_info["shadow_candidate_not_adopted_reason"] = None
            _save_step_projection(
                "04b",
                "clean_rectangle_shadow_candidate_adopted",
                mask01,
                depth_masked,
                note="clean rectangle skip時のshadow candidateを安全条件で本流採用した後．",
            )
        else:
            clean_rectangle_shadow_info["shadow_candidate_adopted_as_final"] = False
            clean_rectangle_shadow_info["shadow_candidate_not_adopted_reason"] = "not_adopted_by_clean_rectangle_override_guard"

        save_json(
            Path(shot_dir) / "debug_clean_rectangle_shadow" / f"mask{sel_idx}{result_suffix}_clean_rectangle_shadow_log.json",
            clean_rectangle_shadow_info,
        )
        save_json(
            Path(shot_dir) / "debug_clean_rectangle_shadow" / f"mask{sel_idx}{result_suffix}_clean_rectangle_override_log.json",
            clean_rectangle_override_info,
        )

    merge_end = time.perf_counter()
    print(f"[TIME]{debug_prefix} merge OCR+SAM: {merge_end - merge_start:.3f} sec")
    print(f"[SAM {('OFFLINE' if result_suffix == '_offline' else 'ONLINE')}] selected id = {sel_idx}, mask shape = {mask01.shape}")

    # 選択結果確認用overlayは，Depth補正後のmaskを使う．
    _save_points_and_overlay(
        rgb_pil,
        [mask01],
        shot_dir,
        f"rgb_mask{sel_idx}_selected{result_suffix}",
        draw_ids=False,
    )
    if result_suffix:
        # 100test側の探索や従来名との互換のため，offlineなしの別名も残す．
        src_overlay = shot_dir / f"rgb_mask{sel_idx}_selected{result_suffix}_overlay.jpg"
        dst_overlay = shot_dir / f"rgb_mask{sel_idx}_selected_overlay.jpg"
        if src_overlay.exists() and not dst_overlay.exists():
            try:
                dst_overlay.write_bytes(src_overlay.read_bytes())
            except Exception:
                pass

    # ===== 8) Depth補正後の点群列長さで側面を追加除去 =====
    if needs_shape_refine:
        mask01_before_column = mask01.copy()
        depth_masked_before_column = depth_masked.copy()
        _save_step_projection(
            "05",
            "before_spine_column_length_refine",
            mask01_before_column,
            depth_masked_before_column,
            note="Depth補正・長方形判定・OCR軸取得後，列長フィルタへ入る直前．",
        )

        def _run_column_refine_candidate(use_guard: bool):
            return refine_mask_by_spine_column_length_after_depth(
                mask01=mask01_before_column.copy(),
                depth_masked=depth_masked_before_column.copy(),
                refine_info=refine_info,
                image_shape=color_np.shape[:2],
                t_bin_size_px=4.0,
                s_bin_size_px=5.0,
                s_gap_allow_px=28.0,
                length_reference_mode="robust_percentile",
                length_reference_percentile=100.0,
                min_length_ratio=0.95,
                relaxed_edge_length_ratio=0.95,
                min_points_per_t_bin=6,
                min_selected_t_bins=2,
                expand_selected_t_bins=0,
                s_margin_px=0.0,
                min_valid_keep_ratio=0.35,
                seed_local_window_bins=14,
                bridge_gap_bins=2,
                use_global_s_range=False,
                span_percentiles=(2.0, 98.0),
                min_occupancy_density=0.18,
                column_length_mode="continuous_run",
                min_run_point_ratio=0.18,
                use_seed_width_guard=bool(use_guard),
                seed_width_guard_scale=1.05,
                seed_width_mask_percentile=2.0,
                seed_width_low_percentile_min_median_ratio=0.50,
                seed_width_fallback_mode="median",
                seed_guard_center_mode="mask_centroid",
                seed_width_profile_s_percentiles=(10.0, 90.0),
                seed_width_profile_t_percentiles=(2.0, 98.0),
                min_seed_guard_width_px=30.0,
                max_seed_guard_width_px=90.0,
                one_sided_removal=True,
                one_sided_min_candidate_points=30,
                one_sided_score_margin_ratio=1.05,
                protect_ocr_extended_band=True,
                ocr_extended_protection_margin_px=0.0,
                ocr_extended_max_rescue_ratio=0.55,
                post_profile_side_suppression=True,
                post_profile_width_margin=1.12,
                post_profile_max_shrink_ratio=0.22,
                return_info=True,
            )

        def _score_column_candidate(mask_candidate, depth_candidate, info_candidate):
            shape_candidate = analyze_mask_rectangularity(
                mask_candidate,
                image_shape=color_np.shape[:2],
                iou_threshold=0.82,
                extent_threshold=0.78,
                solidity_threshold=0.92,
                width_cv_threshold=0.28,
                width_max_median_threshold=1.55,
                approx_vertex_max=8,
            )
            valid_keep_ratio = float(info_candidate.get("valid_keep_ratio", 0.0)) if isinstance(info_candidate, dict) else 0.0
            if not isinstance(shape_candidate, dict) or int(shape_candidate.get("area_px", 0)) <= 0:
                return -1.0e9, shape_candidate
            if not bool(info_candidate.get("used", False)):
                return -1.0e8, shape_candidate
            if valid_keep_ratio < 0.35:
                return -1.0e8, shape_candidate

            rotated_iou = float(shape_candidate.get("rotated_rect_iou", 0.0))
            extent = float(shape_candidate.get("extent", 0.0))
            solidity = float(shape_candidate.get("solidity", 0.0))
            width_cv = float(shape_candidate.get("width_cv", 999.0))
            width_max_med = float(shape_candidate.get("width_max_over_median", 999.0))

            score = 0.0
            score += 3.0 * rotated_iou
            score += 2.0 * extent
            score += 2.0 * solidity
            # 幅の乱れが残っている候補をより強く罰する．
            # False側は面積が大きく残る分スコアが高くなりやすいため，
            # width_cv と width_max_over_median を以前より重視する．
            score -= 2.2 * min(width_cv, 5.0)
            score -= 1.4 * max(0.0, min(width_max_med, 10.0) - 1.15)
            if bool(shape_candidate.get("clean_rectangle", False)):
                score += 0.35

            # 削りすぎ防止．ただし，少し削れている候補を優先したいので，
            # 残存率が高すぎる候補には軽いペナルティを入れる．
            if valid_keep_ratio < 0.60:
                score -= 2.0 * (0.60 - valid_keep_ratio)
            if valid_keep_ratio > 0.98:
                score -= 0.15

            return float(score), shape_candidate

        mask_no_guard, depth_no_guard, info_no_guard = _run_column_refine_candidate(False)
        score_no_guard, shape_no_guard = _score_column_candidate(mask_no_guard, depth_no_guard, info_no_guard)

        mask_guard, depth_guard, info_guard = _run_column_refine_candidate(True)
        score_guard, shape_guard = _score_column_candidate(mask_guard, depth_guard, info_guard)
        _save_step_projection(
            "06a",
            "candidate_column_refine_no_seed_guard",
            mask_no_guard,
            depth_no_guard,
            note="列長フィルタ候補: seed width guardなし．",
        )
        _save_step_projection(
            "06b",
            "candidate_column_refine_seed_guard",
            mask_guard,
            depth_guard,
            note="列長フィルタ候補: seed width guardあり．",
        )

        if score_guard > score_no_guard:
            mask01, depth_masked, column_info = mask_guard, depth_guard, info_guard
            selected_refine_mode = "seed_width_guard_true"
        else:
            mask01, depth_masked, column_info = mask_no_guard, depth_no_guard, info_no_guard
            selected_refine_mode = "seed_width_guard_false"

        column_info["shape_rectangularity"] = shape_info
        column_info["depth_prefilter"] = depth_prefilter_info
        column_info["auto_seed_width_guard_selection"] = {
            "enabled": True,
            "selected": selected_refine_mode,
            "score_false": float(score_no_guard),
            "score_true": float(score_guard),
            "shape_false": shape_no_guard,
            "shape_true": shape_guard,
            "info_false_summary": {
                "used": bool(info_no_guard.get("used", False)),
                "reason": info_no_guard.get("reason"),
                "valid_keep_ratio": info_no_guard.get("valid_keep_ratio"),
                "selected_group": info_no_guard.get("selected_group"),
                "seed_guard_used": info_no_guard.get("seed_guard_used"),
                "seed_guard_width_px": info_no_guard.get("seed_guard_width_px"),
            },
            "info_true_summary": {
                "used": bool(info_guard.get("used", False)),
                "reason": info_guard.get("reason"),
                "valid_keep_ratio": info_guard.get("valid_keep_ratio"),
                "selected_group": info_guard.get("selected_group"),
                "seed_guard_used": info_guard.get("seed_guard_used"),
                "seed_guard_width_source": info_guard.get("seed_guard_width_source"),
                "seed_guard_base_width_px": info_guard.get("seed_guard_base_width_px"),
                "seed_guard_width_px": info_guard.get("seed_guard_width_px"),
                "seed_t_source": info_guard.get("seed_t_source"),
                "seed_t": info_guard.get("seed_t"),
                "mask_centroid_t_for_seed": info_guard.get("mask_centroid_t_for_seed"),
                "seed_width_profile_info": info_guard.get("seed_width_profile_info"),
            },
        }
        _save_step_projection(
            "07",
            f"after_column_refine_selected_{selected_refine_mode}",
            mask01,
            depth_masked,
            note="自動スコアで選択された列長フィルタ後．",
        )
        _record_residual_stage(
            "after_column_refine",
            mask01,
            depth_masked,
            refine_stage=refine_info,
            column_stage=column_info,
            extra_info={"stage_id": "07", "selected_refine_mode": selected_refine_mode},
        )

        # ===== 8.5) 最終t方向幅クリップ =====
        # 列長フィルタでFalse側が選ばれた場合などに，背表紙方向には長いが
        # 幅方向にはみ出している側面・背景を最後に削る．
        # 削りすぎ，または形状悪化が起きた場合は自動で元に戻る．
        mask01_before_final_t_clip = mask01.copy()
        depth_masked_before_final_t_clip = depth_masked.copy()
        mask01, depth_masked, final_t_width_clip_info = apply_final_t_width_clip(
            mask01=mask01,
            depth_masked=depth_masked,
            column_info=column_info,
            image_shape=color_np.shape[:2],
            width_scale=1.05,
            margin_px=2.0,
            min_keep_ratio=0.55,
            center_mode="ocr_center",
            min_width_px=24.0,
            max_width_px=90.0,
            require_shape_not_worse=True,
        )
        column_info["final_t_width_clip"] = final_t_width_clip_info

        # 最終t幅クリップのデバッグ画像を保存する．
        save_spine_column_length_debug(
            shot_dir=shot_dir,
            color_np=color_np,
            mask_before=mask01_before_final_t_clip,
            mask_after=mask01,
            depth_before=depth_masked_before_final_t_clip,
            depth_after=depth_masked,
            column_info={
                "used": bool(final_t_width_clip_info.get("used", False)),
                "reason": final_t_width_clip_info.get("reason"),
                "algorithm": "final_t_width_clip",
                "final_t_width_clip": final_t_width_clip_info,
                "auto_seed_width_guard_selection": column_info.get("auto_seed_width_guard_selection"),
                "axis": column_info.get("axis"),
                "normal": column_info.get("normal"),
                "center": column_info.get("center"),
            },
            stem=f"mask{sel_idx}{result_suffix}_final_t_width_clip",
        )
        _save_step_projection(
            "08",
            "after_final_t_width_clip",
            mask01,
            depth_masked,
            note="final_t_width_clip適用後．used=falseの場合は前段と同じmaskになる．",
        )
        _record_residual_stage(
            "after_final_t_width_clip",
            mask01,
            depth_masked,
            refine_stage=refine_info,
            column_stage=column_info,
            extra_info={"stage_id": "08", "final_t_width_clip": final_t_width_clip_info},
        )

        # ===== 8.6) 前段で削った片側だけ，短い点群列を追加削除 =====
        # Depth精度不足で奥側側面が残る場合に備え，final_t_width_clip 等で
        # 実際に削られた側だけを対象として，最大列長の95%未満のt列を削る．
        # 反対側は削らないため，背表紙本体の削りすぎを抑える．
        mask01_before_one_sided_column_prune = mask01.copy()
        depth_masked_before_one_sided_column_prune = depth_masked.copy()
        mask01, depth_masked, one_sided_column_prune_info = apply_one_sided_short_column_prune_after_final(
            mask01=mask01,
            depth_masked=depth_masked,
            column_info=column_info,
            refine_info=refine_info,
            image_shape=color_np.shape[:2],
            side_reference_mask_before=mask01_before_final_t_clip,
            side_reference_mask_after=mask01,
            side_reference_depth_before=depth_masked_before_final_t_clip,
            t_bin_size_px=4.0,
            span_percentiles=(2.0, 98.0),
            length_ratio=0.95,
            min_points_per_t_bin=6,
            min_side_removed_pixels=30,
            min_valid_keep_ratio=0.60,
            max_removed_ratio=0.45,
            protect_seed_band_px=0.0,
            small_component_area_ratio=0.0003,
            require_shape_not_worse=False,
            return_info=True,
        )
        residual_policy_A_post_column_info = _evaluate_residual_policy_a_candidate(
            enabled=_residual_policy_has_a(residual_policy_mode),
            policy_A_variant=_residual_policy_a_variant(residual_policy_mode),
            stage="post_column",
            before_mask01=mask01_before_one_sided_column_prune,
            before_depth_masked=depth_masked_before_one_sided_column_prune,
            candidate_mask01=mask01,
            candidate_depth_masked=depth_masked,
            candidate_info=one_sided_column_prune_info,
            intr=intr,
            depth_scale=depth_scale,
            refine_info=refine_info,
            column_info=column_info,
        )
        if bool(residual_policy_A_post_column_info.get("reject_candidate", False)):
            mask01 = mask01_before_one_sided_column_prune.copy()
            depth_masked = depth_masked_before_one_sided_column_prune.copy()
            one_sided_column_prune_info = dict(one_sided_column_prune_info)
            one_sided_column_prune_info.update({
                "used": False,
                "residual_policy_A_reverted": True,
                "reason": "residual Policy A rejected post-column candidate: "
                + "; ".join(residual_policy_A_post_column_info.get("reject_reasons", [])),
                "valid_count_after": int(np.count_nonzero((mask01 > 0) & (depth_masked > 0))),
                "valid_keep_ratio": 1.0,
                "removed_count": 0,
                "removed_ratio": 0.0,
            })
        if bool(residual_policy_A_post_column_info.get("evaluated", False)):
            residual_policy_A_info = residual_policy_A_post_column_info
        column_info["one_sided_post_final_column_prune"] = one_sided_column_prune_info
        save_json(
            shot_dir / f"mask{sel_idx}{result_suffix}_one_sided_post_final_column_prune_log.json",
            one_sided_column_prune_info,
        )
        save_spine_column_length_debug(
            shot_dir=shot_dir,
            color_np=color_np,
            mask_before=mask01_before_one_sided_column_prune,
            mask_after=mask01,
            depth_before=depth_masked_before_one_sided_column_prune,
            depth_after=depth_masked,
            column_info=one_sided_column_prune_info,
            stem=f"mask{sel_idx}{result_suffix}_one_sided_post_final_column_prune",
        )
        _save_step_projection(
            "09",
            "after_one_sided_post_final_column_prune",
            mask01,
            depth_masked,
            note="前段で削った側だけに短列削除を追加した後．",
        )
        _record_residual_stage(
            "after_post_column_side_front_prune",
            mask01,
            depth_masked,
            refine_stage=refine_info,
            column_stage=column_info,
            extra_info={"stage_id": "09", "one_sided_post_final_column_prune": one_sided_column_prune_info},
        )

        save_spine_column_length_debug(
            shot_dir=shot_dir,
            color_np=color_np,
            mask_before=mask01_before_column,
            mask_after=mask_no_guard,
            depth_before=depth_masked_before_column,
            depth_after=depth_no_guard,
            column_info=info_no_guard,
            stem=f"mask{sel_idx}{result_suffix}_candidate_no_seed_guard",
        )
        save_spine_column_length_debug(
            shot_dir=shot_dir,
            color_np=color_np,
            mask_before=mask01_before_column,
            mask_after=mask_guard,
            depth_before=depth_masked_before_column,
            depth_after=depth_guard,
            column_info=info_guard,
            stem=f"mask{sel_idx}{result_suffix}_candidate_seed_guard",
        )
        save_spine_column_length_debug(
            shot_dir=shot_dir,
            color_np=color_np,
            mask_before=mask01_before_column,
            mask_after=mask01,
            depth_before=depth_masked_before_column,
            depth_after=depth_masked,
            column_info=column_info,
            stem=f"mask{sel_idx}{result_suffix}",
        )
    else:
        print(f"[MASK SHAPE]{debug_prefix} skip column length refinement because depth-filtered mask is rectangular")
        _record_residual_stage(
            "after_column_refine",
            mask01,
            depth_masked,
            refine_stage=refine_info,
            column_stage=column_info,
            extra_info={"stage_id": "07", "skipped": True, "reason": column_info.get("reason")},
        )

    depth_masked = _apply_mask_to_depth_no_save(depth_masked, mask01)
    _save_step_projection(
        "10",
        "before_ocr_text_ransac_plane_filter",
        mask01,
        depth_masked,
        note="RANSAC平面推定に入る直前．長方形判定でcleanだった場合は列長フィルタなしの状態．",
    )

    # ===== 9) RANSAC平面推定で背表紙平面のみを残す =====
    # 画角端では画像平面上の列長判定が歪むため，まず3D点群から背表紙平面を推定する．
    mask01_before_ransac_plane = mask01.copy()
    depth_masked_before_ransac_plane = depth_masked.copy()

    # OCR軸やOCR参照領域は，shape refineがskipされた場合にもdepth anchor側の情報を使えるようにする．
    refine_info_for_ransac = refine_info
    if not (isinstance(refine_info_for_ransac, dict) and isinstance(refine_info_for_ransac.get("selected_ocr_polygon"), dict)):
        refine_info_for_ransac = depth_anchor_refine_info

    if bool(ENABLE_RANSAC_SPINE_PLANE):
        mask01, depth_masked, ransac_spine_plane_info = refine_mask_by_ransac_spine_plane_after_depth(
            mask01=mask01,
            depth_masked=depth_masked,
            intr=intr,
            depth_scale=depth_scale,
            refine_info=refine_info_for_ransac,
            image_shape=color_np.shape[:2],
            shot_dir=shot_dir,
            color_np=color_np,
            stem=f"mask{sel_idx}{result_suffix}_ransac_spine_plane",
            distance_threshold_m=0.008,
            ransac_n=3,
            num_iterations=1200,
            min_total_points=120,
            min_reference_points=80,
            min_reference_inlier_ratio=0.45,
            min_keep_ratio=0.45,
            small_component_area_ratio=0.0003,
            prefer_ocr_reference=True,
            # 重要: OCR文字領域から推定した平面だけを使う。
            # 全点群最大平面へfallbackすると，背表紙ではなく大きな側面/表紙面にRANSACが寄り，
            # 文字領域側までpost-RANSAC a95で削る事故が起きるため禁止する。
            allow_all_points_fallback=False,
            return_info=True,
        )
    else:
        ransac_spine_plane_info = {
            "used": False,
            "reason": "skipped by ENABLE_RANSAC_SPINE_PLANE=False",
            "algorithm": "ransac_spine_plane_after_depth",
            "valid_count_before": int(np.count_nonzero((mask01 > 0) & (depth_masked > 0))),
            "valid_count_after": int(np.count_nonzero((mask01 > 0) & (depth_masked > 0))),
            "keep_ratio": 1.0,
            "removed_count": 0,
            "removed_ratio": 0.0,
            "plane_source": None,
            "plane_model": None,
        }
        if shot_dir is not None and color_np is not None:
            try:
                save_ransac_spine_plane_debug(
                    shot_dir=Path(shot_dir),
                    color_np=color_np,
                    mask_before=mask01_before_ransac_plane,
                    mask_after=mask01,
                    depth_before=depth_masked_before_ransac_plane,
                    depth_after=depth_masked,
                    info=ransac_spine_plane_info,
                    stem=f"mask{sel_idx}{result_suffix}_ransac_spine_plane",
                )
            except Exception as e:
                ransac_spine_plane_info["debug_save_error"] = str(e)
                traceback.print_exc()

    column_info["ransac_spine_plane"] = ransac_spine_plane_info
    print(
        "[RANSAC_SPINE_PLANE]"
        f" used={ransac_spine_plane_info.get('used')}"
        f" reason={ransac_spine_plane_info.get('reason')}"
        f" source={ransac_spine_plane_info.get('plane_source')}"
        f" before={ransac_spine_plane_info.get('valid_count_before')}"
        f" after={ransac_spine_plane_info.get('valid_count_after')}"
        f" keep_ratio={ransac_spine_plane_info.get('keep_ratio')}"
    )
    depth_masked = _apply_mask_to_depth_no_save(depth_masked, mask01)
    _save_step_projection(
        "11",
        "after_ocr_text_ransac_plane_filter",
        mask01,
        depth_masked,
        note="OCR文字領域から推定したRANSAC平面で外れ点を除去した後．",
    )
    _record_residual_stage(
        "after_ransac_spine_plane",
        mask01,
        depth_masked,
        refine_stage=refine_info_for_ransac,
        column_stage=column_info,
        extra_info={"stage_id": "11", "ransac_spine_plane": ransac_spine_plane_info},
    )

    # ===== 9.5) RANSAC平面内座標でa方向95%列長フィルタ =====
    # 画像上ではなく，RANSAC平面へ射影した点群を(a,b)座標で見て，
    # b方向列ごとのa方向実長が最大長の95%未満の列を削除する．
    mask01_before_post_ransac_plane_a95 = mask01.copy()
    depth_masked_before_post_ransac_plane_a95 = depth_masked.copy()
    plane_model_for_a95 = None
    if isinstance(ransac_spine_plane_info, dict):
        # OCR文字領域RANSACが実際に採用された場合だけ，post-RANSAC a95を動かす。
        # used=False の場合や，全点群fallback由来の平面では，後段のa95で文字領域まで削る危険がある。
        if (
            ransac_spine_plane_info.get("used") is True
            and str(ransac_spine_plane_info.get("plane_source")) == "ocr_reference_plane"
            and ransac_spine_plane_info.get("plane_model") is not None
        ):
            plane_model_for_a95 = ransac_spine_plane_info.get("plane_model")

    if bool(ENABLE_POST_RANSAC_PLANE_A95) and plane_model_for_a95 is not None:
        mask01, depth_masked, post_ransac_plane_a95_info = apply_post_ransac_plane_a95_column_prune(
            mask01=mask01,
            depth_masked=depth_masked,
            intr=intr,
            depth_scale=depth_scale,
            plane_model=plane_model_for_a95,
            refine_info=refine_info_for_ransac,
            image_shape=color_np.shape[:2],
            shot_dir=shot_dir,
            color_np=color_np,
            stem=f"mask{sel_idx}{result_suffix}_post_ransac_plane_a95",
            b_bin_size_m=0.001,
            length_ratio=0.90,
            span_percentiles=(2.0, 98.0),
            a_bin_size_m=0.003,
            a_gap_allow_m=0.016,
            min_occupancy_density=0.12,
            min_point_count_ratio_to_core=0.06,
            protect_near_threshold_ratio=0.95,
            min_points_per_bin=6,
            min_positive_bins=2,
            bridge_gap_bins=1,
            min_valid_keep_ratio=0.60,
            max_removed_ratio=0.35,
            small_component_area_ratio=0.0003,
            previous_side_info=column_info,
            one_sided_from_previous=True,
            skip_if_previous_side_unknown=True,
            return_info=True,
        )
    else:
        post_ransac_plane_a95_info = {
            "used": False,
            "reason": "skipped because ENABLE_POST_RANSAC_PLANE_A95=False or OCR-reference RANSAC plane was unavailable/not applied",
            "algorithm": "post_ransac_plane_a95_column_prune",
            "valid_count_before": int(np.count_nonzero((mask01 > 0) & (depth_masked > 0))),
            "valid_count_after": int(np.count_nonzero((mask01 > 0) & (depth_masked > 0))),
            "valid_keep_ratio": 1.0,
            "removed_count": 0,
            "removed_ratio": 0.0,
            "ransac_used": bool(ransac_spine_plane_info.get("used")) if isinstance(ransac_spine_plane_info, dict) else None,
            "ransac_plane_source": ransac_spine_plane_info.get("plane_source") if isinstance(ransac_spine_plane_info, dict) else None,
            "ransac_reason": ransac_spine_plane_info.get("reason") if isinstance(ransac_spine_plane_info, dict) else None,
            "note": "post-RANSAC a95 is allowed only when the plane comes from selected OCR text region RANSAC.",
        }

    residual_policy_A_post_ransac_info = _evaluate_residual_policy_a_candidate(
        enabled=_residual_policy_has_a(residual_policy_mode),
        policy_A_variant=_residual_policy_a_variant(residual_policy_mode),
        stage="post_ransac_a95",
        before_mask01=mask01_before_post_ransac_plane_a95,
        before_depth_masked=depth_masked_before_post_ransac_plane_a95,
        candidate_mask01=mask01,
        candidate_depth_masked=depth_masked,
        candidate_info=post_ransac_plane_a95_info,
        intr=intr,
        depth_scale=depth_scale,
        refine_info=refine_info_for_ransac,
        column_info=column_info,
    )
    if bool(residual_policy_A_post_ransac_info.get("reject_candidate", False)):
        mask01 = mask01_before_post_ransac_plane_a95.copy()
        depth_masked = depth_masked_before_post_ransac_plane_a95.copy()
        post_ransac_plane_a95_info = dict(post_ransac_plane_a95_info)
        post_ransac_plane_a95_info.update({
            "used": False,
            "residual_policy_A_reverted": True,
            "reason": "residual Policy A rejected post-RANSAC a95 candidate: "
            + "; ".join(residual_policy_A_post_ransac_info.get("reject_reasons", [])),
            "valid_count_after": int(np.count_nonzero((mask01 > 0) & (depth_masked > 0))),
            "valid_keep_ratio": 1.0,
            "removed_count": 0,
            "removed_ratio": 0.0,
        })
    if bool(residual_policy_A_post_ransac_info.get("evaluated", False)):
        residual_policy_A_info = residual_policy_A_post_ransac_info

    column_info["post_ransac_plane_a95_column_prune"] = post_ransac_plane_a95_info
    print(
        "[POST_RANSAC_PLANE_A95]"
        f" used={post_ransac_plane_a95_info.get('used')}"
        f" reason={post_ransac_plane_a95_info.get('reason')}"
        f" before={post_ransac_plane_a95_info.get('valid_count_before')}"
        f" after={post_ransac_plane_a95_info.get('valid_count_after')}"
        f" keep_ratio={post_ransac_plane_a95_info.get('valid_keep_ratio')}"
        f" threshold_m={post_ransac_plane_a95_info.get('length_threshold_m')}"
    )
    depth_masked = _apply_mask_to_depth_no_save(depth_masked, mask01)
    _save_step_projection(
        "12",
        "after_post_ransac_plane_a95_column_prune",
        mask01,
        depth_masked,
        note="RANSAC平面上a方向95%列長フィルタ後．skipされた場合は前段と同じmaskになる．",
    )
    _record_residual_stage(
        "after_post_ransac_a95",
        mask01,
        depth_masked,
        refine_stage=refine_info_for_ransac,
        column_stage=column_info,
        extra_info={"stage_id": "12", "post_ransac_plane_a95": post_ransac_plane_a95_info},
    )

    # この時点の画像がユーザーの言う final.png．calculate_yaw 直前の最終入力である．
    final_reconstruction_info = save_final_png_and_reconstruction_bundle(
        shot_dir=shot_dir,
        color_np=color_np,
        mask01=mask01,
        depth_masked=depth_masked,
        intr=intr,
        depth_scale=depth_scale,
        stem=f"mask{sel_idx}{result_suffix}",
    )
    _save_step_projection(
        "90",
        "final_mask_depth_before_calculate_yaw",
        mask01,
        depth_masked,
        note="calculate_yaw直前の最終mask/depth．final.pngと同じ段階．",
    )
    _record_residual_stage(
        "final_before_calculate_yaw",
        mask01,
        depth_masked,
        refine_stage=refine_info_for_ransac,
        column_stage=column_info,
        extra_info={"stage_id": "90", "final_reconstruction": final_reconstruction_info},
    )

    # ===== 10) 点群化 =====
    _3D_info = calculate_yaw(mask01, depth_masked, intr, depth_scale)
    pts_f = _3D_info["points"]
    _save_step_projection(
        "91",
        "calculate_yaw_output_points_projection",
        None,
        None,
        note="calculate_yawが返した最終点群pts_fを直接RGBへ再投影した画像．",
        pts_stage=pts_f,
    )

    # 従来互換の最終overlayもshot_dir直下に保存する．
    final_overlay_info = _save_final_valid_depth_region_overlay_root(
        shot_dir=shot_dir,
        color_np=color_np,
        mask01=mask01,
        depth_masked=depth_masked,
        stem=f"mask{sel_idx}{result_suffix}",
    )

    if save_pointcloud_debug:
        save_ply_ascii(shot_dir / f"pointcloud{result_suffix}.ply", pts_f, None)
        save_final_pointcloud_rgb_link_debug(
            shot_dir=shot_dir,
            color_np=color_np,
            mask01=mask01,
            depth_masked=depth_masked,
            pts_f=pts_f,
            intr=intr,
            stem=f"mask{sel_idx}{result_suffix}",
        )

    # ===== 10) PCA / 開口幅 / 把持点 =====
    mean, pc1, pc2 = pca_axes_fix_dir(pts_f)
    vx, vy = float(pc1[0]), float(pc1[1])
    norm_xy = float(np.hypot(vx, vy))
    theta_rad = 0.0 if norm_xy < 1e-8 else float(np.arctan2(vy, vx))

    if needs_shape_refine or clean_rectangle_override_adopted:
        book_width, book_width_info = estimate_book_width_from_filtered_mask_axis(
            mask01=mask01,
            depth_masked=depth_masked,
            intr=intr,
            depth_scale=depth_scale,
            column_info=column_info,
            refine_info=refine_info,
        )
        if book_width is None:
            book_width_info = estimate_book_width(pts_f, mean, pc1, pc2)
            book_width = book_width_info.get("av_book_width_m")
            if isinstance(book_width_info, dict):
                book_width_info["fallback_method"] = "estimate_book_width_3d_pca"
    else:
        book_width_info = estimate_book_width(pts_f, mean, pc1, pc2)
        book_width = book_width_info.get("av_book_width_m")
        if isinstance(book_width_info, dict):
            book_width_info["method"] = "estimate_book_width_3d_pca_rectangular_mask_no_refine"
            book_width_info["shape_rectangularity"] = shape_info

    if isinstance(book_width_info, dict):
        book_width_info["clean_rectangle_override_adopted"] = bool(clean_rectangle_override_adopted)
        book_width_info["width_before_clean_rectangle_override_mm"] = clean_rectangle_override_info.get("original_width_mm")
        book_width_info["width_after_clean_rectangle_override_mm"] = clean_rectangle_override_info.get("shadow_width_mm")
        book_width_info["clean_rectangle_override"] = clean_rectangle_override_info

    residual_policy_interaction_info = _evaluate_residual_policy_ab_safe_interaction(
        residual_policy_mode=residual_policy_mode,
        residual_policy_A_info=residual_policy_A_info,
        current_width_m=book_width,
        initial_mask_quality=initial_mask_quality_info,
    )
    residual_policy_B_info, book_width_after_policy_B = _evaluate_residual_policy_b_width(
        enabled=_residual_policy_has_b(residual_policy_mode),
        residual_policy_mode=residual_policy_mode,
        residual_policy_interaction=residual_policy_interaction_info,
        current_width_m=book_width,
        mask01=mask01,
        depth_masked=depth_masked,
        intr=intr,
        depth_scale=depth_scale,
        refine_info=refine_info,
        column_info=column_info,
        initial_mask_quality=initial_mask_quality_info,
        clean_rectangle_override_adopted=clean_rectangle_override_adopted,
    )
    if bool(residual_policy_B_info.get("robust_width_adopted", False)):
        book_width = book_width_after_policy_B
        if isinstance(book_width_info, dict):
            book_width_info["residual_policy_B_original_width_mm"] = residual_policy_B_info.get("current_width_mm")
            book_width_info["residual_policy_B_selected_width_mm"] = residual_policy_B_info.get("selected_robust_width_mm")
            book_width_info["residual_policy_B_candidate"] = residual_policy_B_info.get("robust_width_candidate")
            book_width_info["method"] = str(book_width_info.get("method", book_width_info.get("fallback_method", "unknown"))) + "_residual_policy_B_experiment"
    if isinstance(book_width_info, dict):
        book_width_info["residual_policy_mode"] = residual_policy_mode
        book_width_info["residual_policy_A"] = residual_policy_A_info
        book_width_info["residual_policy_B"] = residual_policy_B_info
        book_width_info["residual_policy_interaction"] = residual_policy_interaction_info

    try:
        if book_width is not None:
            clean_rectangle_analysis_info["pca_width_mm"] = float(book_width * 1000.0)
        if isinstance(book_width_info, dict):
            clean_rectangle_analysis_info["width_method_used"] = (
                book_width_info.get("method")
                or book_width_info.get("fallback_method")
                or book_width_info.get("algorithm")
            )
            axis_width_info = book_width_info.get("axis_width_info")
            if isinstance(axis_width_info, dict) and axis_width_info.get("width_mm") is not None:
                clean_rectangle_analysis_info["axis_pixel_width_mm"] = float(axis_width_info.get("width_mm"))
            elif book_width_info.get("width_mm") is not None and str(book_width_info.get("method", "")).startswith("filtered_mask_axis"):
                clean_rectangle_analysis_info["axis_pixel_width_mm"] = float(book_width_info.get("width_mm"))
    except Exception as e:
        clean_rectangle_analysis_info["final_width_update_error"] = str(e)

    target_point_info = find_target_point(pts_f)
    target_point = target_point_info.get("target_m")

    residual_guard_diagnostics = _summarize_residual_guard_info(column_info)
    residual_final_width_info = {
        "final_adopted_width_mm": None if book_width is None else float(book_width * 1000.0),
        "book_width_info": book_width_info,
        "shadow_width_candidates_from_final_stage": (
            residual_stage_diagnostics[-1].get("axis_width_candidates")
            if residual_stage_diagnostics and isinstance(residual_stage_diagnostics[-1], dict)
            else None
        ),
        "largest_component_shadow_width_candidates_from_final_stage": (
            residual_stage_diagnostics[-1].get("largest_component_axis_width_candidates")
            if residual_stage_diagnostics and isinstance(residual_stage_diagnostics[-1], dict)
            else None
        ),
    }
    residual_stage_debug_info = save_residual_stage_diagnostics_debug(
        shot_dir=shot_dir,
        stem=f"mask{sel_idx}{result_suffix}",
        stage_diagnostics=residual_stage_diagnostics,
        stage_diffs=residual_stage_diffs,
        guard_diagnostics=residual_guard_diagnostics,
        final_width_info=residual_final_width_info,
    )

    if show_pointcloud_gui:
        visualize_points_and_target_open3d(pts_f, target_point)

    # ===== 11) 結果保存 =====
    pca_json = {
        "theta_rad": float(theta_rad),
        "theta_deg": float(np.degrees(theta_rad)),
        "p_min_m": [float(x) for x in np.asarray(target_point).reshape(-1)],
        "book_width_mm": float(book_width * 1000.0),
        "book_width_info": book_width_info,
    }
    pca_name = "pca_result_offline.json" if result_suffix == "_offline" else "pca_result.json"
    json_path = shot_dir / pca_name
    json_path.write_text(json.dumps(pca_json, ensure_ascii=False, indent=2), encoding="utf-8")

    step_projection_summary_info = {
        "enabled": bool(save_step_by_step_pointcloud_debug),
        "debug_dir": str(shot_dir / "debug_step_by_step_pointcloud_rgb"),
        "num_saved_steps": int(len(step_projection_infos)),
        "steps": step_projection_infos,
    }
    if bool(save_step_by_step_pointcloud_debug):
        save_json(
            shot_dir / "debug_step_by_step_pointcloud_rgb" / f"mask{sel_idx}{result_suffix}_step_by_step_pointcloud_projection_summary.json",
            step_projection_summary_info,
        )

    processing_log = {
        "selected_mask_index": int(sel_idx),
        "residual_policy_mode": residual_policy_mode,
        "residual_policy_A": residual_policy_A_info,
        "residual_policy_B": residual_policy_B_info,
        "residual_policy_interaction": residual_policy_interaction_info,
        "initial_mask_quality": initial_mask_quality_info,
        "depth_prefilter": depth_prefilter_info,
        "shape_rectangularity": shape_info,
        "clean_rectangle_analysis": clean_rectangle_analysis_info,
        "clean_rectangle_shadow": clean_rectangle_shadow_info,
        "clean_rectangle_override": clean_rectangle_override_info,
        "refine_info": refine_info,
        "column_info": column_info,
        "one_sided_column_prune_info": one_sided_column_prune_info,
        "ransac_spine_plane_info": column_info.get("ransac_spine_plane"),
        "post_ransac_plane_a95_info": column_info.get("post_ransac_plane_a95_column_prune"),
        "residual_stage_diagnostics": {
            "debug_files": residual_stage_debug_info,
            "stage_count": int(len(residual_stage_diagnostics)),
            "stage_diffs": residual_stage_diffs,
            "guard_diagnostics": residual_guard_diagnostics,
            "final_width_info": residual_final_width_info,
        },
        "final_reconstruction_info": final_reconstruction_info,
        "final_overlay_info": final_overlay_info,
        "step_by_step_pointcloud_projection": step_projection_summary_info,
        "flow": "offline_aligned_depth_first_plus_final_t_clip_plus_ocr_reference_ransac_plane_only_plus_plane_coordinate_a95_column_prune_if_ocr_plane_ok",
    }
    save_json(shot_dir / f"mask{sel_idx}{result_suffix}_processing_log.json", processing_log)

    print(f":heavy_check_mark: Saved PCA JSON{debug_prefix}: {json_path}")
    return theta_rad, target_point, book_width * 1000.0, shot_dir


def run_capture_and_pca(
    query: str,
    out_dir: str | Path = "captures",
    # 1280x720 固定
    width: int = 1280,
    height: int = 720,
    fps: int = 6,
    sam_device: str = "gpu",
    encoder_path: str = "/home/book/pro_book/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx",
    decoder_path: str = "/home/book/pro_book/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx",
    interactive: bool = True,
    use_persistent_runtime: bool = True,
    sam_pts_side: tuple[int, int] = (32, 8),
    sam_decoder_k_keep: int = 1,
    sam_target_len: int = 768,
    depth_merge_tolerance_raw: int = 30,
    show_pointcloud_gui: bool = False,
    save_pointcloud_debug: bool = False,
    save_step_by_step_pointcloud_debug: bool = True,
    **kwargs,
) -> tuple[float, np.ndarray, np.ndarray, Path]:
    """
    RealSenseで撮影して認識するonline版．
    撮影以降の後処理は run_capture_and_pca_offline() と同じ共通コアを使う．
    """
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        shot_dir = out_dir / ts
        shot_dir.mkdir(parents=True, exist_ok=True)

        pipe2 = rs.pipeline()
        cfg2 = rs.config()
        cfg2.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg2.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        align2 = rs.align(rs.stream.color)

        capture_start = time.perf_counter()
        color_np, depth_np_u16, intr, depth_scale = capture_one_shot(
            pipe2,
            cfg2,
            align2,
            shot_dir,
            stem="after_init",
        )
        capture_end = time.perf_counter()
        print("Realsense captured.")
        print(f"[TIME] capture            : {capture_end - capture_start:.3f} sec")

        _save_camera_params_json(shot_dir, intr, depth_scale, fps=fps)
        print(f":heavy_check_mark: Saved camera params: {shot_dir / 'camera_params.json'}")

        return _run_recognition_core_like_offline(
            query=query,
            shot_dir=shot_dir,
            color_np=color_np,
            depth_np_u16=depth_np_u16,
            intr=intr,
            depth_scale=depth_scale,
            sam_device=sam_device,
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            interactive=interactive,
            result_suffix="",
            use_persistent_runtime=use_persistent_runtime,
            sam_pts_side=sam_pts_side,
            sam_decoder_k_keep=sam_decoder_k_keep,
            sam_target_len=sam_target_len,
            depth_merge_tolerance_raw=depth_merge_tolerance_raw,
            show_pointcloud_gui=show_pointcloud_gui,
            save_pointcloud_debug=save_pointcloud_debug,
            save_step_by_step_pointcloud_debug=save_step_by_step_pointcloud_debug,
        )

    except Exception:
        traceback.print_exc()
        print("認識失敗！！")
        return None, None, None, None


def run_capture_and_pca_offline(
    query: str,
    shot_dir: str | Path,
    sam_device: str = "gpu",
    encoder_path: str = "/home/book/pro_book/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx",
    decoder_path: str = "/home/book/pro_book/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx",
    interactive: bool = True,
    use_persistent_runtime: bool = True,
    sam_pts_side: tuple[int, int] = (32, 8),
    sam_decoder_k_keep: int = 1,
    sam_target_len: int = 768,
    depth_merge_tolerance_raw: int = 30,
    show_pointcloud_gui: bool = False,
    save_pointcloud_debug: bool = False,
    save_step_by_step_pointcloud_debug: bool = True,
    **kwargs,
) -> tuple[float, np.ndarray, np.ndarray, Path]:
    """
    撮影済み画像を使うoffline版．online版と同じ共通コアを使うため，
    Depth補正・長方形判定・OCR軸候補・列長フィルタの順序としきい値が一致する．
    """
    shot_dir = Path(shot_dir).expanduser().resolve()
    rgb_path = shot_dir / "after_init_rgb.png"
    depth_path = shot_dir / "after_init_depth.npy"

    if not rgb_path.exists():
        raise FileNotFoundError(f"{rgb_path} がありません")
    if not depth_path.exists():
        raise FileNotFoundError(f"{depth_path} がありません")

    color_np = cv2.imread(str(rgb_path))
    if color_np is None:
        raise FileNotFoundError(f"{rgb_path} を読み込めませんでした")
    depth_np_u16 = np.load(depth_path)

    intr = rs.intrinsics()
    intr.width = 1280
    intr.height = 720
    intr.fx = 908.1617431640625
    intr.fy = 906.4829711914062
    intr.ppx = 637.79833984375
    intr.ppy = 371.0213928222656
    depth_scale = 0.0010000000474974513

    _save_camera_params_json(shot_dir, intr, depth_scale)

    return _run_recognition_core_like_offline(
        query=query,
        shot_dir=shot_dir,
        color_np=color_np,
        depth_np_u16=depth_np_u16,
        intr=intr,
        depth_scale=depth_scale,
        sam_device=sam_device,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        interactive=interactive,
        result_suffix="_offline",
        use_persistent_runtime=use_persistent_runtime,
        sam_pts_side=sam_pts_side,
        sam_decoder_k_keep=sam_decoder_k_keep,
        sam_target_len=sam_target_len,
        depth_merge_tolerance_raw=depth_merge_tolerance_raw,
        show_pointcloud_gui=show_pointcloud_gui,
        save_pointcloud_debug=save_pointcloud_debug,
        save_step_by_step_pointcloud_debug=save_step_by_step_pointcloud_debug,
    )


    
def main():
    import pyrealsense2 as rs
    import cv2
    from pathlib import Path
    from datetime import datetime

    # ========= 設定 =========
    width = 1280
    height = 720
    fps = 1

    # 保存先: captures/YYYY-MM-DD
    today_str = datetime.now().strftime("%Y-%m-%d")
    save_dir = Path("./captures") / today_str
    save_dir.mkdir(parents=True, exist_ok=True)

    # ========= RealSense 初期化 =========
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipe.start(cfg)

    try:
        print("カメラ起動中...")
        print("s: 保存")
        print("q or ESC: 終了")

        # 最初の数フレームは捨てる（露出安定用）
        for _ in range(30):
            pipe.wait_for_frames()

        while True:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = cv2.cvtColor(
                cv2.imread("/dev/null") if False else
                __import__("numpy").asanyarray(color_frame.get_data()),
                cv2.COLOR_RGB2BGR
            ) if False else __import__("numpy").asanyarray(color_frame.get_data())

            # 表示用
            preview = color_image.copy()
            cv2.putText(
                preview,
                f"{width}x{height}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                preview,
                "Press 's' to save / 'q' or ESC to quit",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("RealSense Capture", preview)
            key = cv2.waitKey(1) & 0xFF

            # sキーで保存
            if key == ord("s"):
                now = datetime.now().strftime("%H-%M-%S")
                filename = f"{width}x{height}_{now}.png"
                filepath = save_dir / filename

                cv2.imwrite(str(filepath), color_image)
                print(f"保存: {filepath}")

            # q or ESCで終了
            elif key == ord("q") or key == 27:
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--out", type=str, default="captures", help="撮影と結果保存のベースフォルダ")
#     ap.add_argument("--w", type=int, default=1280)
#     ap.add_argument("--h", type=int, default=720)
#     ap.add_argument("--fps", type=int, default=6)
#     ap.add_argument("--sam_device", choices=["gpu", "cpu", "auto"], default="gpu")
#     ap.add_argument(
#         "--encoder",
#         default="/home/book/pro_book/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx",
#     )
#     ap.add_argument(
#         "--decoder",
#         default="/home/book/pro_book/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx",
#     )
#     args = ap.parse_args()

#     theta_rad, p_min, book_width, yaw = run_capture_and_pca(
#         query="独立行政法人",
#         out_dir=args.out,
#         width=args.w,
#         height=args.h,
#         fps=args.fps,
#         sam_device=args.sam_device,
#         encoder_path=args.encoder,
#         decoder_path=args.decoder,
#         interactive=True,
#     )

#     print("\n=== Summary ===")
#     print(f"book width = {book_width}")
#     print(f"roll (deg) = {np.degrees(theta_rad):.6f}")
#     print(f"p_min = {p_min}")

#     #print("yaw (deg) = {:.3f}".format(np.degrees(yaw)))
#     print("===============")
    

if __name__ == "__main__":
    main()
