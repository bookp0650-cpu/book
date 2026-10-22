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
from modules.max_rect_choose import select_rectmax_mask, mask_rectangularity
from modules.Depth_threshold import depth_range_zscore_from_mask, filter_points_by_depth_range

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


def save_masked_and_cropped(
    rgb_bgr: np.ndarray,
    depth_u16: np.ndarray,
    mask01: np.ndarray,
    outdir: Path,
    stem: str,
):
    """
    対象書籍のみの RGB/Depth を保存（背景0マスク）
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # --- マスク適用（背景=0） ---
    rgb_masked = rgb_bgr.copy()
    rgb_masked[mask01 == 0] = 0
    depth_masked = depth_u16.copy()
    depth_masked[mask01 == 0] = 0

    cv2.imwrite(str(outdir / f"{stem}_rgb_masked.png"), rgb_masked)
    np.save(outdir / f"{stem}_depth_masked.npy", depth_masked)

    # 深度可視化（0 以外の範囲を 0–255 に正規化）
    nonzero = depth_masked[depth_masked > 0]
    if nonzero.size > 0:
        zmin, zmax = int(nonzero.min()), int(nonzero.max())
        zrange = max(1, zmax - zmin)
        depth_vis = np.zeros_like(depth_masked, dtype=np.uint8)
        depth_vis[depth_masked > 0] = (
            (depth_masked[depth_masked > 0] - zmin) * 255 // zrange
        ).astype(np.uint8)
        cv2.imwrite(str(outdir / f"{stem}_depth_masked_vis.png"), depth_vis)

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
    return color_np, depth_np_u16, intr, depth_scale
def run_ocr_subprocess(shot_dir: Path):
    # OCR用仮想環境の python へのパス（あなたの環境に合わせて変更）
    # Linux venv例: /home/book/venv/ocr/bin/python
    OCR_PY = "/home/book/pro_book_SAM3/pro_hand_book_python/detection/pro_handbook/sam_py_demo/OCR/.paadle_ocr/bin/python"
    OCR_SCRIPT = "/home/book/pro_book_SAM3/pro_hand_book_python/detection/pro_handbook/sam_py_demo/OCR/paddle_ocr_test.py"

    subprocess.run([OCR_PY, OCR_SCRIPT, str(shot_dir)], check=True)
    print(f"✔ OCR done: {shot_dir / 'ocr_result.json'}")

def run_capture_and_pca(
    query: str,
    out_dir: str | Path = "captures",
    # 2回目（SAM用）は 1280x720 固定
    width: int = 1280,
    height: int = 720,
    fps: int = 6,
    # 1回目（バーコード用）はできるだけ高解像度
    barcode_width: int = 1920,
    barcode_height: int = 1080,
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
        # shot_dir を作った直後あたり
        # latest_path = Path("/home/book/pro_book_SAM3/pro_hand_book_python/captures/LATEST_PATH.txt")
        # latest_path.write_text(str(shot_dir), encoding="utf-8")
        # print(f"✔ Saved latest shot_dir: {latest_path}")

        # ===== 1回目（バーコード用：高解像度） =====
        # pipe1 = rs.pipeline()
        # cfg1 = rs.config()
        # cfg1.enable_stream(rs.stream.color, barcode_width, barcode_height, rs.format.bgr8, fps)
        # align1 = rs.align(rs.stream.color)

        # color_np, depth_np_u16, intr, depth_scale = capture_one_shot(
        #     pipe1, cfg1, align1 , shot_dir, stem="before_init", color_only=True
        # )

        # ---- バーコード認識 ----
        # barcode_number = input("bar code number? ")
        # number_identified = detect_barcode(barcode_number, shot_dir)
        # if number_identified:
        #     print("Barcode detected:")
        # else:
        #     print("No barcode found.")
            
        
        # ===== 2回目（SAM用：1280x720） =====
        pipe2 = rs.pipeline()
        cfg2 = rs.config()
        cfg2.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg2.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        align2 = rs.align(rs.stream.color)
        
        color_np2, depth_np_u16_2, intr2, depth_scale2 = capture_one_shot(
            pipe2, cfg2, align2, shot_dir, stem="after_init"
        )
        print("Realsense captured.")
        # 以降の推論・点群処理は 2回目の RGB-D を使う
        # （既存コードの変数を差し替えるだけ）脳筋やな
        color_np = color_np2
        depth_np_u16 = depth_np_u16_2
        intr = intr2
        depth_scale = depth_scale2
        
        # ===== 2) infer_for_storage の infer_masks で書籍マスク推論 =====
        sam_cfg = SamConfig(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            device=sam_device,
        )
        sam_runner = SamBatchInfer_storage(sam_cfg)

        # BGR → RGB の PIL.Image に変換
        rgb_pil = Image.fromarray(cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))

        # ステージ保存設定（after_nms / before_smooth など）
        stage_cfg = StageSaveCfg(out_dir=shot_dir)

        masks, sam_data = sam_runner.infer_masks(
            rgb_pil,
            stage_save=stage_cfg,
            stem_for_save="rgb",
        )
        # shot_dir が captures/2025.... の Path だとして
        #save_json(Path(shot_dir) / "sam_data.json", sam_data)
        #print(f"[SAVE] {Path(shot_dir) / 'sam_data.json'}")
        # ===== rect-max 一番四角いマスクから D_min/D_max（zscore±3）を作る =====
        rect_idx, rect_mask01, rect_r = select_rectmax_mask(masks)
        D_min, D_max = depth_range_zscore_from_mask(
            depth_u16=depth_np_u16,
            mask01=rect_mask01,
            depth_scale=depth_scale,
            zscore_lim=4.0,
            min_keep=200,
        )
        D_min = D_min - ALLOWABLE_RANGE_Z  # 少し範囲広げる
        D_max = D_max + ALLOWABLE_RANGE_Z
        print(f"[RECT-MAX] id={rect_idx}, rectangularity={rect_r:.4f}")
        print(f"[RECT-MAX DEPTH] D_min={D_min:.4f} m, D_max={D_max:.4f} m (zscore±3)")

        if len(masks) == 0:
            raise RuntimeError("NO_MASK: 書籍マスクが検出されませんでした。")

        # ===== 3) マスクをオーバーレイ表示して ID を選択 =====
        # ov_bgr = _render_overlay_bgr(rgb_pil, masks, draw_ids=True)
        run_ocr_subprocess(shot_dir)

        #results = find_similar_books(query, sam_data, shot_dir, threshold=40)
        results = match_text_to_mask_main(query, masks, shot_dir, threshold=40)

        
        book_name = results[0]["name"] if results else None

        sel_idx = None
        if not interactive:
            sel_idx = 1
        if book_name:
            m = re.search(r"(\d+)$", book_name)  # 末尾の連続数字
            if m:
                sel_idx = int(m.group(1))      
            sel_mask = masks[sel_idx - 1]
        mask01 = (np.asarray(sel_mask) > 0).astype(np.uint8)
        print(f"[SAM] selected id = {sel_idx}, mask shape = {mask01.shape}")

        # 選択結果のオーバーレイも保存（任意）
        _save_points_and_overlay(
            rgb_pil,
            [sel_mask],
            shot_dir,
            f"rgb_mask{sel_idx}_selected",
            draw_ids=False,
        )
        
        # ===== 4) 対象書籍のみの RGB/Depth を保存 =====
        save_masked_and_cropped(color_np, depth_np_u16, mask01, shot_dir, f"mask{sel_idx}")

        # ===== 5) マスク + depth → カメラ座標点群へ変換 =====
        _3D_info = calculate_yaw(mask01, depth_np_u16, intr, depth_scale)
        yaw = _3D_info["yaw"]
        pts = _3D_info["points"]  # (N,3)

        # ===== rect-max 由来の D_min/D_max で ID選択点群を削る =====
        pts_f = filter_points_by_depth_range(pts, D_min, D_max)
        print(f"[POINTS] selected before={pts.shape[0]} after_depth_clip={pts_f.shape[0]}")
        

        # pts_f: (N,3) [m]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_f)

        # 距離閾値は深度ノイズ/フィルタ次第。まず 2〜5mm あたりから試す
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.003,  # 3mm
            ransac_n=3,
            num_iterations=1000
        )

        pts_in = pts_f[np.asarray(inliers, dtype=np.int64)]
        print(f"[RANSAC] inliers={pts_in.shape[0]} / {pts_f.shape[0]}")

        # ===== 6) 点群を保存（PLY / NPZ） =====
        save_ply_ascii(shot_dir / "pointcloud.ply", pts_in, None)


        # ===== 7) PCA による背表紙特徴量の計算 =====
        # （フォールバックなしでそのまま PCA する前提）
        c = pts_in.mean(axis=0)  # 重心
        X = pts_in - c           # 点群を原点中心に平行移動
        cov = X.T @ X / float(X.shape[0])
        eigvals, eigvecs = np.linalg.eigh(cov)  # 小さい順

        # 固有値の大きい順に並べ替え
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        # 第1, 第2主成分ベクトル
        v1 = eigvecs[:, 0]  # 第一成分
        v2 = eigvecs[:, 1]  # 第二成分

        # 2) 第一成分ベクトルの xy 平面での傾き角度 θ
        vx, vy = float(v1[0]), float(v1[1])
        norm_xy = float(np.hypot(vx, vy))
        if norm_xy < 1e-8:
            theta_rad = 0.0
        else:
            theta_rad = float(np.arctan2(vy, vx))  # 基準は +x 軸

        # 3) 第二成分方向の端点（x成分が最大/最小になる点）
        s2 = X @ v2  # v2座標系(一次元)に投影
        sp = 0.6  # 端を何%捨てるか（1〜5くらいで調整）
        s2_lo = float(np.percentile(s2, sp))
        s2_hi = float(np.percentile(s2, 100.0 - sp))

        # 端点復元
        p_min = c + s2_lo * v2
        p_max = c + s2_hi * v2

        # 念のため向きを揃える（必要なら）
        # 例：xが小さい方を p_min にする
        if p_min[0] > p_max[0]:
            p_min, p_max = p_max, p_min

        # z は書籍マスク全体の平均 depth で上書き
        z_mean = float(pts_in[:, 2].mean())
        p_min[2] = z_mean
        p_max[2] = z_mean
        book_width = np.linalg.norm(p_max - p_min)
        print(f"[PCA] book_width = {book_width*1000:.2f} mm")

        
        v1 = eigvecs[:, 0]  # 長手っぽい方向（仮）
        v1 = v1 / (np.linalg.norm(v1) + 1e-12)

        # ---- 符号固定（例：カメラ座標で「上」= -Y に揃える）----
        cam_up = np.array([0.0, -1.0, 0.0])
        if np.dot(v1, cam_up) > 0:
            v1 = -v1

        # ---- ここが重要：xyだけじゃなく xyz 全部ずらす ----
        GRIP_OFFSET = 0.02  # [m]
        p_min = p_min - v1 * GRIP_OFFSET
        p_max = p_max - v1 * GRIP_OFFSET
        # 球の半径（m）。距離や点群スケールに応じて調整（3〜8mmくらいが見やすい）
        r = 0.003
        # --- p_min / p_max を球で可視化（座標系は pts_in と同じ[m]前提）---
        save_ply_ascii(shot_dir / "pointcloud_rect.ply", pts_in, None)
        pcd = o3d.io.read_point_cloud(str(shot_dir / "pointcloud_rect.ply"))
        pmin = np.asarray(p_min, dtype=np.float64).reshape(3)
        pmax = np.asarray(p_max, dtype=np.float64).reshape(3)
        s_min = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        s_min.translate(pmin)
        s_min.paint_uniform_color([0.0, 0.0, 0.0])  # 黒 = p_min

        s_max = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        s_max.translate(pmax)
        s_max.paint_uniform_color([0.0, 1.0, 0.0])  # 緑 = p_max

        # p_min <-> p_max を線で結ぶ（幅方向の確認に便利）
        line = o3d.geometry.LineSet(
           points=o3d.utility.Vector3dVector(np.vstack([pmin, pmax])),
           lines=o3d.utility.Vector2iVector([[0, 1]]),
        )
        line.colors = o3d.utility.Vector3dVector([[0.0, 0.7, 4.0]])  # 水色    原点/軸も欲しければ（点群のスケールに合わせて）
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)  # 5cm

        # # 点群の見た目
        pcd.estimate_normals()  # 任意（不要なら消してOK）

        o3d.visualization.draw_geometries([pcd, s_min, s_max, line, axis])

        
        # （必要ならここで npz 保存してもいいけど、モジュール使用を前提に返り値だけ）
        print("✔ PCA result:")
        print(f"  theta_rad = {theta_rad:.6f}")
        print(f"  p_min = {p_min}")
        print(f"  p_max = {p_max}")
        print("✔ Files saved under:", shot_dir)
            # ===== PCA結果を JSON に保存（shot_dir 配下） =====
        pca_json = {
            "theta_rad": float(theta_rad),
            "theta_deg": float(np.degrees(theta_rad)),
            "p_min_m": [float(x) for x in np.asarray(p_min).reshape(-1)],
            "p_max_m": [float(x) for x in np.asarray(p_max).reshape(-1)],
            "book_width_mm": float(book_width * 1000.0),  # book_width は [m] 想定
        }

        json_path = Path(shot_dir) / "pca_result.json"
        json_path.write_text(json.dumps(pca_json, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✔ Saved PCA JSON: {json_path}")
        return theta_rad, p_min, book_width*1000.0, shot_dir  # m → mm
    
    except:
        traceback.print_exc()
        print("認識失敗！！")
        return theta_rad, p_min, book_width*1000.0, shot_dir  # m → mm


    

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
