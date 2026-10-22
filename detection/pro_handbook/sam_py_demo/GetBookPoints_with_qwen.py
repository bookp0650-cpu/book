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
# from .OCR.only_one import find_similar_books
# from .OCR.only_one_tilted import match_text_to_mask_main

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

from .OCR.qwen_mask_matcher import QwenMaskMatcher

ALLOWABLE_RANGE_Z = 0.07

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

def save_masked_and_cropped(
    rgb_bgr: np.ndarray,
    depth_u16: np.ndarray,
    mask01: np.ndarray,
    outdir: Path,
    stem: str,
    z_tolerance_raw: int = 80,  # Z16値での許容幅（例: ±80カウント ≈ ±8cm 程度）
):
    """
    対象書籍のみの RGB/Depth を保存（背景0マスク ＋ 深度外れ値除去）
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # --- マスク適用（背景=0） ---
    rgb_masked = rgb_bgr.copy()
    rgb_masked[mask01 == 0] = 0

    depth_masked = depth_u16.copy()
    depth_masked[mask01 == 0] = 0  # まずマスク外を 0 にする

    # --- ここから: マスク内の depth の外れ値除去（1D簡易版 RANSAC 的なもの） ---
    # マスク内で depth が 0 でない画素だけ取り出す
    nonzero = depth_masked[depth_masked > 0]

    if nonzero.size > 0:
        # 主体（本）の「代表的な距離」を中央値で近似
        z_med = int(np.median(nonzero))

        # 中央値から外れすぎた depth を 0 にする
        #   -> 本より手前/奥にある別の物体を消す目的
        z_min_keep = z_med - z_tolerance_raw
        z_max_keep = z_med + z_tolerance_raw

        # 残したい領域（Trueが残す）
        keep = (depth_masked >= z_min_keep) & (depth_masked <= z_max_keep)

        # keep==False のところを 0 にする
        depth_masked[~keep] = 0

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

    return depth_masked  # ここで外れ値除去済みの depth を返す

def capture_one_shot(pipe, cfg, align, shot_dir, *, stem: str, color_only: bool = False):
    profile = pipe.start(cfg)

    for _ in range(10):
        pipe.wait_for_frames()

    frames = pipe.wait_for_frames()

    if color_only:
        color_frame = frames.get_color_frame()
        color_np = np.asanyarray(color_frame.get_data())
        cv2.imwrite(str(shot_dir / f"{stem}_rgb.png"), color_np)

        pipe.stop()
        return color_np, None, None, None

    # depthも使う場合（2回目）
    align_frames = align.process(frames)
    depth_frame = depth_filter_like_viewer(align_frames.get_depth_frame())
    color_frame = align_frames.get_color_frame()

    color_np = np.asanyarray(color_frame.get_data())
    depth_np_u16 = np.asanyarray(depth_frame.get_data())

    cv2.imwrite(str(shot_dir / f"{stem}_rgb.png"), color_np)
    np.save(shot_dir / f"{stem}_depth.npy", depth_np_u16)

    dprof = rs.video_stream_profile(depth_frame.get_profile())
    intr = dprof.get_intrinsics()
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

    pipe.stop()
    return color_np, depth_np_u16, intr, depth_scale #後で調べる
def run_ocr_subprocess(shot_dir: Path):
    # OCR用仮想環境の python へのパス（あなたの環境に合わせて変更）
    # Linux venv例: /home/book/venv/ocr/bin/python
    OCR_PY = "/home/book/pro_book_SAM3/pro_hand_book_python/detection/pro_handbook/sam_py_demo/OCR/.paadle_ocr/bin/python"
    OCR_SCRIPT = "/home/book/pro_book_SAM3/pro_hand_book_python/detection/pro_handbook/sam_py_demo/OCR/paddle_ocr_test.py"

    subprocess.run([OCR_PY, OCR_SCRIPT, str(shot_dir)], check=True)
    print(f"✔ OCR done: {shot_dir / 'ocr_result.json'}")

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

def run_capture_and_pca(
    query: str,
    out_dir: str | Path = "captures",
    #1280x720 固定
    width: int = 1280,
    height: int = 720,
    fps: int = 6,
    sam_device: str = "gpu",
    encoder_path: str = "/home/book/pro_book_SAM3/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx",
    decoder_path: str = "/home/book/pro_book_SAM3/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx",
    interactive: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
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
        
        color_np, depth_np_u16, intr, depth_scale = capture_one_shot(
            pipe2, cfg2, align2, shot_dir, stem="after_init"
        )
        print("Realsense captured.")

        # ===== ★ カメラパラメータ保存（安全版） ===== # 変更（追加）
        camera_json = {
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "ppx": float(intr.ppx),
            "ppy": float(intr.ppy),
            "depth_scale": float(depth_scale),
            "fps": int(fps),
        }

        camera_json_path = shot_dir / "camera_params.json"
        camera_json_path.write_text(
            json.dumps(camera_json, indent=2),
            encoding="utf-8"
        )

        print(f"✔ Saved camera params: {camera_json_path}")

        # color_np = color_np2
        # depth_np_u16 = depth_np_u16_2
        # intr = intr2
        # depth_scale = depth_scale2
        
        # ===== 2) infer_for_storage の infer_masks で書籍マスク推論 =====
        sam_cfg = SamConfig(
            encoder_path=encoder_path,
            decoder_path=decoder_path, #opensource のものを利用
            device=sam_device, #GPU 利用指示
        )
        sam_runner = SamBatchInfer_storage(sam_cfg)

        # BGR（numpy） → RGB の PIL.Image に変換
        rgb_pil = Image.fromarray(cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))

        # ステージ保存設定（after_nms / before_smooth など）
        stage_cfg = StageSaveCfg(out_dir=shot_dir)

        masks, sam_data = sam_runner.infer_masks(
            rgb_pil,
            stage_save=stage_cfg,
            stem_for_save="rgb",
        )
        
        # ===== 3) Qwen2.5-VL による文字認識 =====
        matcher = QwenMaskMatcher(model_name="Qwen/Qwen2.5-VL-3B-Instruct")

        results = matcher.match_query_to_masks(
            query=query,
            rgb_bgr=color_np,
            masks=masks,
            shot_dir=shot_dir,
            threshold=40,
        )

        book_name = results[0]["name"] if results else None

        # マスク選択開始
        sel_idx = None
        if not interactive:
            sel_idx = 1

        if book_name:
            m = re.search(r"(\d+)$", book_name)  # mask_3 → 3
            if m:
                sel_idx = int(m.group(1))

        if sel_idx is None:
            raise RuntimeError("Qwen OCR の結果から対象マスクを選べませんでした。")

        sel_mask = masks[sel_idx - 1]
        mask01 = (np.asarray(sel_mask) > 0).astype(np.uint8)
        print(f"[SAM] selected id = {sel_idx}, mask shape = {mask01.shape}")
        # マスク選択終了

        # 選択結果のオーバーレイも保存（任意）
        _save_points_and_overlay(
            rgb_pil,
            [sel_mask],
            shot_dir,
            f"rgb_mask{sel_idx}_selected",
            draw_ids=False,
        )
        
        # ===== 4) 対象書籍のみの RGB/Depth を保存 =====
        # save_masked_and_cropped(color_np, depth_np_u16, mask01, shot_dir, f"mask{sel_idx}") # 変更
        depth_masked = save_masked_and_cropped(color_np, depth_np_u16, mask01, shot_dir, f"mask{sel_idx}")

        # ===== 5) マスク + depth → カメラ座標点群へ変換 =====
        # _3D_info = calculate_yaw(mask01, depth_np_u16, intr, depth_scale) # 変更
        _3D_info = calculate_yaw(mask01, depth_masked, intr, depth_scale)
        yaw = _3D_info["yaw"]
        pts_f = _3D_info["points"]  # (N,3)

        # pts_f: (N,3) [m]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_f)


        # ===== 6) 点群を保存（PLY / NPZ） =====
        save_ply_ascii(shot_dir / "pointcloud.ply", pts_f, None)


        # ===== 7) 把持書籍の幅を導出=====
        mean, pc1, pc2 = pca_axes_fix_dir(pts_f)
        # 2) 第一成分ベクトルの xy 平面での傾き角度 θ
        vx, vy = float(pc1[0]), float(pc1[1])
        norm_xy = float(np.hypot(vx, vy))
        if norm_xy < 1e-8:
            theta_rad = 0.0
        else:
            theta_rad = float(np.arctan2(vy, vx))  # 基準は +x 軸
        book_width_info = estimate_book_width(pts_f, mean, pc1, pc2)
        book_width = book_width_info.get("av_book_width_m")
        
        # ===== 8) 把持位置（カメラ座標）を導出=====
        target_point_info = find_target_point(pts_f)
        target_point = target_point_info.get("target_m")
        # ===== 9)可視化=====
        visualize_points_and_target_open3d(pts_f, target_point)

        # ===== 9.5) 点群ビューのスクリーンショット保存 ===== # 変更（追加）
        vis_img_path = shot_dir / "pointcloud_view.png"
        save_pointcloud_screenshot(
            pts_f=pts_f,
            target_point=target_point,
            save_path=vis_img_path,
            show_window=False,  # ここを True にすると画像保存＋別ウィンドウ表示
        )
        
        # （必要ならここで npz 保存してもいいけど、モジュール使用を前提に返り値だけ）
        print("✔ PCA result:")
        print(f"  theta_rad = {theta_rad:.6f}")
        print(f"  p_min = {target_point}")
        print("✔ Files saved under:", shot_dir)
            # ===== PCA結果を JSON に保存（shot_dir 配下） =====
        pca_json = {
            "theta_rad": float(theta_rad),
            "theta_deg": float(np.degrees(theta_rad)),
            "p_min_m": [float(x) for x in np.asarray(target_point).reshape(-1)],
            "book_width_mm": float(book_width * 1000.0),  # book_width は [m] 想定
        }

        json_path = Path(shot_dir) / "pca_result.json"
        json_path.write_text(json.dumps(pca_json, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✔ Saved PCA JSON: {json_path}")
        return theta_rad, target_point, book_width*1000.0, shot_dir  # m → mm
    
    except:
        traceback.print_exc()
        print("認識失敗！！")
        return None, None, None, None  
    
def run_capture_and_pca_offline(
    query: str,
    shot_dir: str | Path,
    sam_device: str = "gpu",
    encoder_path: str = "/home/book/pro_book_SAM3/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx",
    decoder_path: str = "/home/book/pro_book_SAM3/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx",
    interactive: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, Path]:
    """
    すでに撮影済みの画像・深度を用いて、
    run_capture_and_pca と同じ処理フロー（SAM→OCR→点群→PCA）を行うオフライン版。

    shot_dir:
        captures/20260226_170457 のようなディレクトリを指定
        中に after_init_rgb.png / after_init_depth.npy がある前提
    """
    shot_dir = Path(shot_dir)

    # 1) 画像・深度の読み込み
    rgb_path = shot_dir / "after_init_rgb.png"
    depth_path = shot_dir / "after_init_depth.npy"

    if not rgb_path.exists():
        raise FileNotFoundError(f"{rgb_path} がありません")
    if not depth_path.exists():
        raise FileNotFoundError(f"{depth_path} がありません")

    color_np = cv2.imread(str(rgb_path))  # BGRで読み込まれる
    depth_np_u16 = np.load(depth_path)    # (H,W) uint16 のはず

    # 2) intrinsics / depth_scale を仮定または手動設定
    #    ★ここは実環境のキャリブレーション値に合わせて書き換えてOK
    intr = rs.intrinsics()
    intr.width = 1280
    intr.height = 720
    intr.fx = 908.1617431640625      # 仮の値（要調整）
    intr.fy = 906.4829711914062      # 仮の値（要調整）
    intr.ppx = 637.79833984375
    intr.ppy = 371.0213928222656

    depth_scale = 0.0010000000474974513  # RealSense の定番値。環境に合わせて調整可

    # ===== 2) infer_for_storage の infer_masks で書籍マスク推論 =====
    sam_cfg = SamConfig(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        device=sam_device,
    )
    sam_runner = SamBatchInfer_storage(sam_cfg)

    # BGR（numpy） → RGB の PIL.Image に変換
    rgb_pil = Image.fromarray(cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))

    # ステージ保存設定（after_nms / before_smooth など）
    stage_cfg = StageSaveCfg(out_dir=shot_dir)

    masks, sam_data = sam_runner.infer_masks(
        rgb_pil,
        stage_save=stage_cfg,
        stem_for_save="rgb",
    )

    # ===== 3) Qwen2.5-VL による文字認識 =====
    matcher = QwenMaskMatcher(model_name="Qwen/Qwen2.5-VL-3B-Instruct")

    results = matcher.match_query_to_masks(
        query=query,
        rgb_bgr=color_np,
        masks=masks,
        shot_dir=shot_dir,
        threshold=40,
    )

    book_name = results[0]["name"] if results else None

    # マスク選択開始
    sel_idx = None
    if not interactive:
        sel_idx = 1

    if book_name:
        m = re.search(r"(\d+)$", book_name)  # mask_3 → 3
        if m:
            sel_idx = int(m.group(1))

    if sel_idx is None:
        raise RuntimeError("Qwen OCR の結果から対象マスクを選べませんでした。")

    sel_mask = masks[sel_idx - 1]
    mask01 = (np.asarray(sel_mask) > 0).astype(np.uint8)
    print(f"[SAM] selected id = {sel_idx}, mask shape = {mask01.shape}")
    # マスク選択終了

    # 選択結果のオーバーレイも保存（任意）
    _save_points_and_overlay(
        rgb_pil,
        [sel_mask],
        shot_dir,
        f"rgb_mask{sel_idx}_selected_offline",
        draw_ids=False,
    )

    # ===== 4) 対象書籍のみの RGB/Depth を保存 =====
    depth_masked = save_masked_and_cropped(
        color_np, depth_np_u16, mask01, shot_dir, f"mask{sel_idx}_offline"
    )

    # ===== 5) マスク + depth → カメラ座標点群へ変換 =====
    _3D_info = calculate_yaw(mask01, depth_masked, intr, depth_scale)
    yaw = _3D_info["yaw"]
    pts_f = _3D_info["points"]  # (N,3)

    # pts_f: (N,3) [m]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_f)

    # ===== 6) 点群を保存（PLY） =====
    save_ply_ascii(shot_dir / "pointcloud_offline.ply", pts_f, None)

    # ===== 7) 把持書籍の幅を導出 =====
    mean, pc1, pc2 = pca_axes_fix_dir(pts_f)
    vx, vy = float(pc1[0]), float(pc1[1])
    norm_xy = float(np.hypot(vx, vy))
    if norm_xy < 1e-8:
        theta_rad = 0.0
    else:
        theta_rad = float(np.arctan2(vy, vx))
    book_width_info = estimate_book_width(pts_f, mean, pc1, pc2)
    book_width = book_width_info.get("av_book_width_m")
    import matplotlib.pyplot as plt

    plt.imshow(mask01, cmap="gray")
    plt.title("mask")
    plt.show()
    plt.imshow(depth_masked, cmap="jet")
    plt.title("masked depth")
    plt.colorbar()
    plt.show()

    # ===== 8) 把持位置（カメラ座標）を導出 =====
    target_point_info = find_target_point(pts_f)
    target_point = target_point_info.get("target_m")

    # ===== 9) 可視化 =====
    visualize_points_and_target_open3d(pts_f, target_point)

    # ===== 9.5) 点群ビューのスクリーンショット保存 =====
    vis_img_path = shot_dir / "pointcloud_view_offline.png"
    save_pointcloud_screenshot(
        pts_f=pts_f,
        target_point=target_point,
        save_path=vis_img_path,
        show_window=False,
    )

    # ログ & JSON
    print("✔ [OFFLINE] PCA result:")
    print(f"  theta_rad = {theta_rad:.6f}")
    print(f"  p_min = {target_point}")
    print("✔ Files saved under:", shot_dir)

    pca_json = {
        "theta_rad": float(theta_rad),
        "theta_deg": float(np.degrees(theta_rad)),
        "p_min_m": [float(x) for x in np.asarray(target_point).reshape(-1)],
        "book_width_mm": float(book_width * 1000.0),
    }
    json_path = shot_dir / "pca_result_offline.json"
    json_path.write_text(json.dumps(pca_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✔ Saved PCA JSON (offline): {json_path}") 

    return theta_rad, target_point, book_width * 1000.0, shot_dir


    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="captures", help="撮影と結果保存のベースフォルダ")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--sam_device", choices=["gpu", "cpu", "auto"], default="gpu")
    ap.add_argument(
        "--encoder",
        default="/home/book/pro_book_SAM3/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx",
    )
    ap.add_argument(
        "--decoder",
        default="/home/book/pro_book_SAM3/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx",
    )
    args = ap.parse_args()

    theta_rad, p_min, book_width, yaw = run_capture_and_pca(
        query="独立行政法人",
        out_dir=args.out,
        width=args.w,
        height=args.h,
        fps=args.fps,
        sam_device=args.sam_device,
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        interactive=True,
    )

    print("\n=== Summary ===")
    print(f"book width = {book_width}")
    print(f"roll (deg) = {np.degrees(theta_rad):.6f}")
    print(f"p_min = {p_min}")

    #print("yaw (deg) = {:.3f}".format(np.degrees(yaw)))
    print("===============")
    

if __name__ == "__main__":
    main()
