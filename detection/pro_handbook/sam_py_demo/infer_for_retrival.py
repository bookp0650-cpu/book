#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Union

import argparse
import numpy as np
from PIL import Image
import onnxruntime as ort
import cv2
from pathlib import Path as _Path

# --- locate and add the project root that contains "modules/" to sys.path ---
for p in _Path(__file__).resolve().parents:
    if (p / "modules").is_dir():
        sys.path.insert(0, str(p))
        break


#saving helper
from modules.overlay_io import _save_points_and_overlay, _render_overlay_bgr


# ===== config =====
DEFAULT_OUT_DIR = _Path("./results")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ===== helpers =====
def build_providers(device: str) -> List[str]:
    if device == "cpu":
        print("⚙ Using CPU for inference")
        return ["CPUExecutionProvider"]
    if device == "gpu":
        print("⚙ Using GPU for inference (if available)")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]

def _bbox_iou_xywh(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    x1, y1, w1, h1 = a; x2, y2, w2, h2 = b
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter <= 0:
        return 0.0
    u = w1*h1 + w2*h2 - inter
    return 0.0 if u == 0 else inter / u

def _mask_iou_downsample(a: np.ndarray, b: np.ndarray, *, side: int = 48) -> float:
    au8 = (a.astype(np.uint8) * 255)
    bu8 = (b.astype(np.uint8) * 255)
    if au8.shape != (side, side):
        au8 = cv2.resize(au8, (side, side), interpolation=cv2.INTER_NEAREST)
    if bu8.shape != (side, side):
        bu8 = cv2.resize(bu8, (side, side), interpolation=cv2.INTER_NEAREST)
    A = au8 > 0
    B = bu8 > 0
    inter = np.logical_and(A, B).sum()
    u = A.sum() + B.sum() - inter
    return 0.0 if u == 0 else float(inter) / float(u)

def gap_fill_by_xbins(selected: List[Dict[str,Any]],
                      candidates: List[Dict[str,Any]],
                      width: int, *,
                      bins: int = 24,
                      iou_thr: float = 0.98,
                      downsample_side: int = 48) -> List[Dict[str,Any]]:
    def xc(bb): x, y, w, h = bb; return (x + x + w) // 2
    taken = np.zeros(bins, dtype=bool)
    binw = max(1, width // bins)

    for s in selected:
        bb = s.get("bbox")
        if not bb: continue
        b = int(min(bins-1, max(0, xc(bb)//binw)))
        taken[b] = True

    pool = sorted(candidates, key=lambda d: d.get("score", 0.0), reverse=True)
    for c in pool:
        bb, m = c.get("bbox"), c.get("mask")
        if not bb or m is None: continue
        b = int(min(bins-1, max(0, xc(bb)//binw)))
        if taken[b]:
            continue
        dup = False
        for s in selected:
            sbb, sm = s.get("bbox"), s.get("mask")
            if not sbb or sm is None:
                continue
            if _bbox_iou_xywh(bb, sbb) < 0.5:
                continue
            if _mask_iou_downsample(m, sm, side=downsample_side) > iou_thr:
                dup = True
                break
        if not dup:
            selected.append(c)
            taken[b] = True
    return selected
# ===== core runner =====
@dataclass
class SamConfig:
    encoder_path: str = "models/sam_vit_h_4b8939.encoder.onnx"
    decoder_path: str = "models/sam_vit_h_4b8939.decoder.onnx"
    device: str = "gpu"
    target_len: int = 768
    pts_side: Union[int, Tuple[int,int]] = (36, 12)
    min_area: int = 500
    iou_thr: float = 0.8
    # ---- AMG crop settings ----
    crop_overlap_ratio: float = 0.34       # 0..0.5 目安
    crop_points_downscale: int = 2         # 層が深いほど点密度を間引く係数(>=1)

@dataclass
class StageSaveCfg:
    out_dir: _Path = DEFAULT_OUT_DIR
    save_after_nms: bool = True
    save_before_smooth: bool = True
    save_after_smooth: bool = True
    save_selected: bool = True
    save_rectangular: bool = True
from infer_for_storage import SamBatchInfer_storage

class SamBatchInfer_retrieval:
    def __init__(self, cfg: SamConfig):
        self.cfg = cfg
        self.storage_runner = SamBatchInfer_storage(cfg)
        # single image path with interactive selection
    

    def run_on_image_path(
        self,
        img_path: _Path | str,
        *,
        stage_save_dir: Optional[_Path] = None,
        swap_lr_output: bool = False,
        min_fold_deg: float = 70.0,
        min_aspect: float = 1.3,
        axis_angle_tol_deg: float = 10.0,
        offset_width_factor: float = 0.25,
        gap_len_factor: float = 0.50,
        z_lower: Optional[float] = -4.0,
        min_len_px: float = 0.0,
        y_from_top_px: int = 230,
        scan_radius_px: int = 80,
        skip_selection: bool = False,
    ) -> Dict[str, Any]:
        out_dir = stage_save_dir or DEFAULT_OUT_DIR
        img_path = _Path(img_path)
        img = Image.open(img_path).convert("RGB")
        stem = img_path.stem

        stage_cfg = StageSaveCfg(out_dir=out_dir)
        masks = self.storage_runner.infer_masks(
            img,
            min_fold_deg=min_fold_deg,
            min_aspect=min_aspect,
            axis_angle_tol_deg=axis_angle_tol_deg,
            offset_width_factor=offset_width_factor,
            gap_len_factor=gap_len_factor,
            z_lower=z_lower,
            min_len_px=min_len_px,
            stage_save=stage_cfg,
            stem_for_save=stem,
        )
        

        #ov_bgr = _render_overlay_bgr(img, masks, draw_ids=True)
        # try:
        #     cv2.imshow("SAM before_smooth", ov_bgr)
        #     cv2.waitKey(50)
        # except cv2.error:
        #     pass
        if skip_selection or (not sys.stdin.isatty()):
            try: cv2.destroyWindow("SAM before_smooth")
            except cv2.error: pass
            return {"image_path": str(img_path), "num_masks": len(masks),
                    "selected_index_1based": None, "uv_left": None, "uv_right": None, "masks": masks}

        while True:
            try:
                sel = input(f"[select mask id 1-{len(masks)} / Enter=skip / 0=skip / q=skip] > ").strip()
            except EOFError:
                sel = "q"

            if sel == "" or sel == "0" or sel.lower() in ("q", "quit", "exit", "s", "skip"):
                try: cv2.destroyWindow("SAM before_smooth")
                except cv2.error: pass
                return {"image_path": str(img_path), "num_masks": len(masks),
                        "selected_index_1based": None, "uv_left": None, "uv_right": None, "masks": masks}
            try:
                idx = int(sel)
                if 1 <= idx <= len(masks):
                    break
            except ValueError:
                pass
            print("  !! invalid id. try again.")
        
        sel_mask = masks[idx - 1]
        #if stage_cfg.save_selected:
        #    _save_points_and_overlay(img, [sel_mask], out_dir, f"{stem}_mask{idx}_selected", draw_ids=False)

        csv_path = out_dir / f"{stem}_mask{idx}_binary.csv"
        np.savetxt(csv_path, (sel_mask > 0).astype(np.uint8), fmt="%d", delimiter=",")

        try: cv2.destroyWindow("SAM before_smooth")
        except cv2.error: pass
        # print("num_masks:", len(masks))
        return {
            "image_path": str(img_path),
            "num_masks": len(masks),
            "selected_index_1based": idx,
            "csv_path": str(csv_path),
            "masks": masks,
        }

# ===== CLI =====
def _parse_args():
    ap = argparse.ArgumentParser()
    # modes
    ap.add_argument("--image", type=str, help="単一の入力画像パス")
    ap.add_argument("--input_dir", type=str, help="画像フォルダ（--image無指定時に使用）")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR), help="出力フォルダ")
    ap.add_argument("target", nargs="?", help="画像ファイル or 画像フォルダ（省略時は ./images）")

    # model
    ap.add_argument("--encoder", default="/home/book/pro_book_SAM3/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx")
    ap.add_argument("--decoder", default="/home/book/pro_book_SAM3/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx")
    ap.add_argument("--device", choices=["gpu", "cpu", "auto"], default="gpu")
    # AMG crop params
    ap.add_argument("--crop_overlap_ratio", type=float, default=0.34, help="クロップ重なり率(0..0.5)")
    ap.add_argument("--crop_points_downscale", type=int, default=2, help="層ごとの点密度間引き係数(>=1)")
    # grid / candidates
    #ap.add_argument("--pts_side", type=int, default=36)
    ap.add_argument("--pts_side_x", type=int, default=48, help="横方向のグリッド数（指定時は --pts_side_y とセット）")
    ap.add_argument("--pts_side_y", type=int, default=12, help="縦方向のグリッド数（指定時は --pts_side_x とセット）")
    ap.add_argument("--min_area", type=int, default=800)
    ap.add_argument("--iou_thr", type=float, default=0.8)  # reserved

    # zscore / axis min len
    ap.add_argument("--z_lower", type=float, default=-1.5)
    ap.add_argument("--min_len_px", type=float, default=0.0)

    # vertical keep (fold-based)
    ap.add_argument("--min_fold_deg", type=float, default=10.0, help="fold角の下限（0=水平, 90=垂直）")
    ap.add_argument("--min_aspect", type=float, default=1.3, help="縦長判定の L/W 下限")

    # coaxial merge params
    ap.add_argument("--axis_angle_tol", type=float, default=4.0)
    ap.add_argument("--offset_width_factor", type=float, default=0.15)
    ap.add_argument("--gap_len_factor", type=float, default=0.06)

    # LR extraction
    ap.add_argument("--swap_lr_output", action="store_true", help="左右の出力を入れ替える")
    ap.add_argument("--y_from_top_px", type=int, default=230, help="左右端抽出に用いる水平ライン（画像上端からのpx）")
    ap.add_argument("--scan_radius_px", type=int, default=80, help="指定ラインで交差しない場合に±探索する半径[px]")

    ap.add_argument("--skip_selection", action="store_true", help="マスク選択をスキップして終了（UVなどは出力しない）")

    return ap.parse_args()

def _main_cli():
    args = _parse_args()
    out_dir = _Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tm = cv2.TickMeter()
    pts_side_val = (args.pts_side_x, args.pts_side_y)
    tm.start()
    cfg = SamConfig(
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        device=args.device,
        pts_side=pts_side_val,
        min_area=args.min_area,
        iou_thr=args.iou_thr,
        crop_overlap_ratio=args.crop_overlap_ratio,
        crop_points_downscale=max(1, args.crop_points_downscale),
    )
    runner = SamBatchInfer_retrieval(cfg)
    tm.stop(); print(f"[TIMER] model_load: {tm.getTimeMilli():.2f} ms")
    if args.image:
        res = runner.run_on_image_path(
            args.image,
            stage_save_dir=out_dir,
            swap_lr_output=args.swap_lr_output,
            min_fold_deg=args.min_fold_deg,
            min_aspect=args.min_aspect,
            axis_angle_tol_deg=args.axis_angle_tol,
            offset_width_factor=args.offset_width_factor,
            gap_len_factor=args.gap_len_factor,
            z_lower=args.z_lower,
            min_len_px=args.min_len_px,
            y_from_top_px=args.y_from_top_px,
            scan_radius_px=args.scan_radius_px,
            skip_selection=args.skip_selection
        )
        if res.get("selected_index_1based") is None:
            print("NO_MASK_OR_SKIPPED")
        return

    # if a positional target is provided, decide file/dir here
    if args.target:
        tgt = _Path(args.target)
        if tgt.is_file():
            res = runner.run_on_image_path(
                tgt,
                stage_save_dir=out_dir,
                swap_lr_output=args.swap_lr_output,
                min_fold_deg=args.min_fold_deg,
                min_aspect=args.min_aspect,
                axis_angle_tol_deg=args.axis_angle_tol,
                offset_width_factor=args.offset_width_factor,
                gap_len_factor=args.gap_len_factor,
                z_lower=args.z_lower,
                min_len_px=args.min_len_px,
                y_from_top_px=args.y_from_top_px,
                scan_radius_px=args.scan_radius_px,
                skip_selection=args.skip_selection
            )
            if res.get("selected_index_1based") is None:
                print("NO_MASK_OR_SKIPPED")
            return
        elif tgt.is_dir():
            in_dir = tgt
        else:
            print("Target not found:", tgt)
            return
    else:
        # batch mode: iterate over images in a folder
        in_dir = _Path(args.input_dir) if args.input_dir else _Path("images")
    img_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in IMG_EXT])
    if not img_paths:
        print("No images found in:", in_dir)
        return

    for img_path in img_paths:
        print(f"==> {img_path.name}")
        res = runner.run_on_image_path(
            img_path,
            stage_save_dir=out_dir,
            swap_lr_output=args.swap_lr_output,
            min_fold_deg=args.min_fold_deg,
            min_aspect=args.min_aspect,
            axis_angle_tol_deg=args.axis_angle_tol,
            offset_width_factor=args.offset_width_factor,
            gap_len_factor=args.gap_len_factor,
            z_lower=args.z_lower,
            min_len_px=args.min_len_px,
            y_from_top_px=args.y_from_top_px,
            scan_radius_px=args.scan_radius_px,
            skip_selection=args.skip_selection
        )
        if res.get("selected_index_1based") is None:
            print("  NO_MASK_OR_SKIPPED")
    print("✔ Done:", out_dir)
    print("length of masks =", len(res.get("masks", [])))

if __name__ == "__main__":
    _main_cli()