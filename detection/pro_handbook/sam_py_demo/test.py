#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RealSense 1ショット撮影 or 撮影済みデータ再処理
→ SAM(infer_for_storage.infer_masks)で書籍領域認識
→ OCRで対象書籍マスク選択
→ 対象書籍のRGB/Depth保存
→ 点群化 → PCA → 把持点計算 → PLY/JSON保存

使い方:

[オンライン撮影]
python rs_book_capture_and_pointcloud_v2.py --query "独立行政法人"

[オフライン再処理]
python rs_book_capture_and_pointcloud_v2.py \
    --query "独立行政法人" \
    --offline captures/20260306_185322

[camera_params.json が無い場合のオフライン]
python rs_book_capture_and_pointcloud_v2.py \
    --query "独立行政法人" \
    --offline captures/20260306_185322 \
    --fx 908.1617 --fy 906.4830 --ppx 637.7983 --ppy 371.0214 \
    --depth_scale 0.001
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

# ===== RealSense =====
import pyrealsense2 as rs

# ===== SAM =====
from .infer_for_storage import SamConfig, SamBatchInfer_storage, StageSaveCfg

# ===== OCR / Utility =====
from .OCR.only_one_tilted import match_text_to_mask_main
from modules.overlay_io import _save_points_and_overlay
from modules.pointcloud_utils import save_ply_ascii
from modules.calculate_3D_point_or_RANSAC import calculate_yaw
from modules.pca_vector import pca_axes_fix_dir
from modules.book_width import estimate_book_width
from modules.grip_point import find_target_point
from modules.open3d_view import visualize_points_and_target_open3d


# =========================================
# Utility
# =========================================
def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# =========================================
# RealSense helpers
# =========================================
def depth_filter_like_viewer(raw_depth_frame: rs.depth_frame) -> rs.depth_frame:
    """
    rs-viewer に近いフィルタ処理
    """
    spat = rs.spatial_filter()
    spat.set_option(rs.option.filter_magnitude, 2.0)
    spat.set_option(rs.option.filter_smooth_alpha, 0.5)
    spat.set_option(rs.option.filter_smooth_delta, 20.0)

    hole_fill = rs.hole_filling_filter()
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    filtered = depth_to_disparity.process(raw_depth_frame)
    filtered = spat.process(filtered)
    filtered = disparity_to_depth.process(filtered)
    filtered = hole_fill.process(filtered)
    return filtered.as_depth_frame()


def capture_one_shot(pipe, cfg, align, shot_dir: Path, *, stem: str):
    """
    RealSense から1ショット取得して保存
    """
    profile = pipe.start(cfg)

    for _ in range(10):
        pipe.wait_for_frames()

    frames = pipe.wait_for_frames()
    align_frames = align.process(frames)

    depth_frame_raw = align_frames.get_depth_frame()
    color_frame = align_frames.get_color_frame()

    if not depth_frame_raw or not color_frame:
        pipe.stop()
        raise RuntimeError("color/depth frame の取得に失敗しました")

    depth_frame = depth_filter_like_viewer(depth_frame_raw)

    color_np = np.asanyarray(color_frame.get_data())
    depth_np_u16 = np.asanyarray(depth_frame.get_data())

    cv2.imwrite(str(shot_dir / f"{stem}_rgb.png"), color_np)
    np.save(shot_dir / f"{stem}_depth.npy", depth_np_u16)

    dprof = rs.video_stream_profile(depth_frame.get_profile())
    intr = dprof.get_intrinsics()
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

    pipe.stop()
    return color_np, depth_np_u16, intr, depth_scale


def save_camera_params(shot_dir: Path, intr: rs.intrinsics, depth_scale: float, fps: int):
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
    save_json(shot_dir / "camera_params.json", camera_json)


def intrinsics_from_dict(camera: dict) -> rs.intrinsics:
    intr = rs.intrinsics()
    intr.width = int(camera["width"])
    intr.height = int(camera["height"])
    intr.fx = float(camera["fx"])
    intr.fy = float(camera["fy"])
    intr.ppx = float(camera["ppx"])
    intr.ppy = float(camera["ppy"])
    return intr


def load_camera_params(
    shot_dir: Path,
    *,
    width: int | None = None,
    height: int | None = None,
    fx: float | None = None,
    fy: float | None = None,
    ppx: float | None = None,
    ppy: float | None = None,
    depth_scale: float | None = None,
):
    """
    1. camera_params.json があればそれを使用
    2. なければ引数の fx/fy/ppx/ppy/depth_scale から作る
    """
    camera_json_path = shot_dir / "camera_params.json"

    if camera_json_path.exists():
        camera = load_json(camera_json_path)
        intr = intrinsics_from_dict(camera)
        return intr, float(camera["depth_scale"])

    required = [width, height, fx, fy, ppx, ppy, depth_scale]
    if any(v is None for v in required):
        raise RuntimeError(
            "camera_params.json が見つかりません。"
            " オフライン再処理には camera_params.json を置くか、"
            "--w --h --fx --fy --ppx --ppy --depth_scale を指定してください。"
        )

    camera = {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "ppx": ppx,
        "ppy": ppy,
        "depth_scale": depth_scale,
    }
    intr = intrinsics_from_dict(camera)
    return intr, float(depth_scale)


# =========================================
# OCR
# =========================================
def run_ocr_subprocess(shot_dir: Path):
    OCR_PY = "/home/book/pro_book_SAM3/pro_hand_book_python/detection/pro_handbook/sam_py_demo/OCR/.paadle_ocr/bin/python"
    OCR_SCRIPT = "/home/book/pro_book_SAM3/pro_hand_book_python/detection/pro_handbook/sam_py_demo/OCR/paddle_ocr_test.py"

    subprocess.run([OCR_PY, OCR_SCRIPT, str(shot_dir)], check=True)
    print(f"✔ OCR done: {shot_dir / 'ocr_result.json'}")


# =========================================
# Save masked image/depth
# =========================================
def save_masked_and_cropped(
    rgb_bgr: np.ndarray,
    depth_u16: np.ndarray,
    mask01: np.ndarray,
    outdir: Path,
    stem: str,
    z_tolerance_raw: int = 80,
):
    """
    対象書籍のみの RGB/Depth を保存（背景0マスク + 深度外れ値除去）
    """
    outdir.mkdir(parents=True, exist_ok=True)

    rgb_masked = rgb_bgr.copy()
    rgb_masked[mask01 == 0] = 0

    depth_masked = depth_u16.copy()
    depth_masked[mask01 == 0] = 0

    nonzero = depth_masked[depth_masked > 0]
    if nonzero.size > 0:
        z_med = int(np.median(nonzero))
        z_min_keep = z_med - z_tolerance_raw
        z_max_keep = z_med + z_tolerance_raw
        keep = (depth_masked >= z_min_keep) & (depth_masked <= z_max_keep)
        depth_masked[~keep] = 0

    cv2.imwrite(str(outdir / f"{stem}_rgb_masked.png"), rgb_masked)
    np.save(outdir / f"{stem}_depth_masked.npy", depth_masked)

    nonzero = depth_masked[depth_masked > 0]
    if nonzero.size > 0:
        zmin, zmax = int(nonzero.min()), int(nonzero.max())
        zrange = max(1, zmax - zmin)
        depth_vis = np.zeros_like(depth_masked, dtype=np.uint8)
        depth_vis[depth_masked > 0] = (
            (depth_masked[depth_masked > 0] - zmin) * 255 // zrange
        ).astype(np.uint8)
        cv2.imwrite(str(outdir / f"{stem}_depth_masked_vis.png"), depth_vis)

    return depth_masked


# =========================================
# Visualization
# =========================================
def save_pointcloud_screenshot(
    pts_f: np.ndarray,
    target_point: np.ndarray,
    save_path: Path,
    show_window: bool = False,
) -> None:
    try:
        pts = np.asarray(pts_f).reshape(-1, 3)
        tgt = np.asarray(target_point).reshape(3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(tgt)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=show_window)
        vis.add_geometry(pcd)
        vis.add_geometry(sphere)
        vis.poll_events()
        vis.update_renderer()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        vis.capture_screen_image(str(save_path), do_render=True)
        vis.destroy_window()

        print(f"✔ Saved pointcloud screenshot: {save_path}")
    except Exception:
        traceback.print_exc()
        print("⚠ 点群スクリーンショット保存に失敗しました（処理は続行します）")


# =========================================
# Data loading
# =========================================
def prepare_online_capture(
    out_dir: str | Path,
    *,
    width: int,
    height: int,
    fps: int,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    shot_dir = out_dir / ts
    shot_dir.mkdir(parents=True, exist_ok=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    align = rs.align(rs.stream.color)

    color_np, depth_np_u16, intr, depth_scale = capture_one_shot(
        pipe, cfg, align, shot_dir, stem="after_init"
    )
    save_camera_params(shot_dir, intr, depth_scale, fps)

    print("✔ RealSense captured.")
    print(f"✔ Saved under: {shot_dir}")
    return color_np, depth_np_u16, intr, depth_scale, shot_dir


def prepare_offline_capture(
    shot_dir: str | Path,
    *,
    width: int | None = None,
    height: int | None = None,
    fx: float | None = None,
    fy: float | None = None,
    ppx: float | None = None,
    ppy: float | None = None,
    depth_scale: float | None = None,
):
    shot_dir = Path(shot_dir)

    rgb_path = shot_dir / "after_init_rgb.png"
    depth_path = shot_dir / "after_init_depth.npy"

    if not rgb_path.exists():
        raise FileNotFoundError(f"{rgb_path} がありません")
    if not depth_path.exists():
        raise FileNotFoundError(f"{depth_path} がありません")

    color_np = cv2.imread(str(rgb_path))
    depth_np_u16 = np.load(depth_path)

    if color_np is None:
        raise RuntimeError(f"{rgb_path} の読み込みに失敗しました")

    intr, ds = load_camera_params(
        shot_dir,
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        ppx=ppx,
        ppy=ppy,
        depth_scale=depth_scale,
    )

    print(f"✔ Offline data loaded from: {shot_dir}")
    return color_np, depth_np_u16, intr, ds, shot_dir


# =========================================
# Main pipeline
# =========================================
def select_mask_from_ocr_results(results, masks, interactive: bool):
    book_name = results[0]["name"] if results else None

    sel_idx = None
    if not interactive:
        sel_idx = 1

    if book_name:
        m = re.search(r"(\d+)$", book_name)
        if m:
            sel_idx = int(m.group(1))

    if sel_idx is None:
        raise RuntimeError("OCR結果からマスクIDを選択できませんでした")

    if not (1 <= sel_idx <= len(masks)):
        raise RuntimeError(f"選択されたマスクIDが不正です: {sel_idx}")

    sel_mask = masks[sel_idx - 1]
    mask01 = (np.asarray(sel_mask) > 0).astype(np.uint8)

    print(f"[SAM] selected id = {sel_idx}, mask shape = {mask01.shape}")
    return sel_idx, sel_mask, mask01


def process_pipeline(
    *,
    query: str,
    color_np: np.ndarray,
    depth_np_u16: np.ndarray,
    intr,
    depth_scale: float,
    shot_dir: Path,
    sam_device: str,
    encoder_path: str,
    decoder_path: str,
    interactive: bool = True,
    z_tolerance_raw: int = 80,
    visualize: bool = True,
):
    sam_cfg = SamConfig(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        device=sam_device,
    )
    sam_runner = SamBatchInfer_storage(sam_cfg)

    rgb_pil = Image.fromarray(cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))
    stage_cfg = StageSaveCfg(out_dir=shot_dir)

    masks, sam_data = sam_runner.infer_masks(
        rgb_pil,
        stage_save=stage_cfg,
        stem_for_save="rgb",
    )

    run_ocr_subprocess(shot_dir)
    results = match_text_to_mask_main(query, masks, shot_dir, threshold=40)

    sel_idx, sel_mask, mask01 = select_mask_from_ocr_results(results, masks, interactive)

    _save_points_and_overlay(
        rgb_pil,
        [sel_mask],
        shot_dir,
        f"rgb_mask{sel_idx}_selected",
        draw_ids=False,
    )

    depth_masked = save_masked_and_cropped(
        color_np,
        depth_np_u16,
        mask01,
        shot_dir,
        f"mask{sel_idx}",
        z_tolerance_raw=z_tolerance_raw,
    )

    _3d_info = calculate_yaw(mask01, depth_masked, intr, depth_scale)
    yaw = _3d_info["yaw"]
    pts_f = _3d_info["points"]

    if pts_f is None or len(pts_f) == 0:
        raise RuntimeError("点群が空です")

    save_ply_ascii(shot_dir / "pointcloud.ply", pts_f, None)

    mean, pc1, pc2 = pca_axes_fix_dir(pts_f)
    vx, vy = float(pc1[0]), float(pc1[1])
    norm_xy = float(np.hypot(vx, vy))
    theta_rad = 0.0 if norm_xy < 1e-8 else float(np.arctan2(vy, vx))

    book_width_info = estimate_book_width(pts_f, mean, pc1, pc2)
    book_width_m = book_width_info.get("av_book_width_m")
    if book_width_m is None:
        raise RuntimeError("book_width の計算に失敗しました")

    target_point_info = find_target_point(pts_f)
    target_point = target_point_info.get("target_m")
    if target_point is None:
        raise RuntimeError("target_point の計算に失敗しました")

    if visualize:
        visualize_points_and_target_open3d(pts_f, target_point)

    save_pointcloud_screenshot(
        pts_f=pts_f,
        target_point=target_point,
        save_path=shot_dir / "pointcloud_view.png",
        show_window=False,
    )

    pca_json = {
        "theta_rad": float(theta_rad),
        "theta_deg": float(np.degrees(theta_rad)),
        "yaw_rad": float(yaw),
        "yaw_deg": float(np.degrees(yaw)),
        "target_point_m": [float(x) for x in np.asarray(target_point).reshape(-1)],
        "book_width_mm": float(book_width_m * 1000.0),
    }
    save_json(shot_dir / "pca_result.json", pca_json)

    print("✔ PCA result:")
    print(f"  theta_rad = {theta_rad:.6f}")
    print(f"  yaw_rad   = {yaw:.6f}")
    print(f"  target    = {target_point}")
    print(f"  width_mm  = {book_width_m * 1000.0:.2f}")
    print(f"✔ Files saved under: {shot_dir}")

    return theta_rad, target_point, book_width_m * 1000.0, shot_dir


# =========================================
# Public API
# =========================================
def run_capture_and_pca(
    query: str,
    out_dir: str | Path = "captures",
    width: int = 1280,
    height: int = 720,
    fps: int = 6,
    sam_device: str = "gpu",
    encoder_path: str = "/home/book/pro_book_SAM3/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx",
    decoder_path: str = "/home/book/pro_book_SAM3/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx",
    interactive: bool = True,
    z_tolerance_raw: int = 80,
    visualize: bool = True,
):
    try:
        color_np, depth_np_u16, intr, depth_scale, shot_dir = prepare_online_capture(
            out_dir,
            width=width,
            height=height,
            fps=fps,
        )

        return process_pipeline(
            query=query,
            color_np=color_np,
            depth_np_u16=depth_np_u16,
            intr=intr,
            depth_scale=depth_scale,
            shot_dir=shot_dir,
            sam_device=sam_device,
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            interactive=interactive,
            z_tolerance_raw=z_tolerance_raw,
            visualize=visualize,
        )
    except Exception:
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
    z_tolerance_raw: int = 80,
    visualize: bool = True,
    width: int | None = None,
    height: int | None = None,
    fx: float | None = None,
    fy: float | None = None,
    ppx: float | None = None,
    ppy: float | None = None,
    depth_scale: float | None = None,
):
    try:
        color_np, depth_np_u16, intr, ds, shot_dir = prepare_offline_capture(
            shot_dir,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            ppx=ppx,
            ppy=ppy,
            depth_scale=depth_scale,
        )

        return process_pipeline(
            query=query,
            color_np=color_np,
            depth_np_u16=depth_np_u16,
            intr=intr,
            depth_scale=ds,
            shot_dir=shot_dir,
            sam_device=sam_device,
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            interactive=interactive,
            z_tolerance_raw=z_tolerance_raw,
            visualize=visualize,
        )
    except Exception:
        traceback.print_exc()
        print("オフライン認識失敗！！")
        return None, None, None, None


# =========================================
# CLI
# =========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, required=True, help="検索したい書籍文字列")

    ap.add_argument("--out", type=str, default="captures", help="オンライン撮影時の保存先ベースフォルダ")
    ap.add_argument("--offline", type=str, default=None, help="既存shot_dirを指定するとオフライン再処理")

    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=6)

    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--ppx", type=float, default=None)
    ap.add_argument("--ppy", type=float, default=None)
    ap.add_argument("--depth_scale", type=float, default=None)

    ap.add_argument("--sam_device", choices=["gpu", "cpu", "auto"], default="gpu")
    ap.add_argument(
        "--encoder",
        default="/home/book/pro_book_SAM3/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx",
    )
    ap.add_argument(
        "--decoder",
        default="/home/book/pro_book_SAM3/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx",
    )

    ap.add_argument("--non_interactive", action="store_true", help="OCR失敗時でも mask1 を使う前提で動かしたい場合")
    ap.add_argument("--no_visualize", action="store_true", help="Open3D表示をしない")
    ap.add_argument("--z_tolerance_raw", type=int, default=80, help="Depth外れ値除去の許容幅")

    args = ap.parse_args()

    interactive = not args.non_interactive
    visualize = not args.no_visualize

    if args.offline:
        print("===== OFFLINE MODE =====")
        theta_rad, p_min, book_width, shot_dir = run_capture_and_pca_offline(
            query=args.query,
            shot_dir=args.offline,
            sam_device=args.sam_device,
            encoder_path=args.encoder,
            decoder_path=args.decoder,
            interactive=interactive,
            z_tolerance_raw=args.z_tolerance_raw,
            visualize=visualize,
            width=args.w,
            height=args.h,
            fx=args.fx,
            fy=args.fy,
            ppx=args.ppx,
            ppy=args.ppy,
            depth_scale=args.depth_scale,
        )
    else:
        print("===== ONLINE MODE =====")
        theta_rad, p_min, book_width, shot_dir = run_capture_and_pca(
            query=args.query,
            out_dir=args.out,
            width=args.w,
            height=args.h,
            fps=args.fps,
            sam_device=args.sam_device,
            encoder_path=args.encoder,
            decoder_path=args.decoder,
            interactive=interactive,
            z_tolerance_raw=args.z_tolerance_raw,
            visualize=visualize,
        )

    print("\n=== Summary ===")
    print(f"book width [mm] = {book_width}")
    print(f"roll [deg]      = {np.degrees(theta_rad) if theta_rad is not None else None}")
    print(f"target point    = {p_min}")
    print(f"shot_dir        = {shot_dir}")
    print("================")


if __name__ == "__main__":
    main()