# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# batch_infer_v2.py — SAM batch inference (pipeline-matched to your sam_infer_module)

# Pipeline (per image):
#   1) Encoder embedding
#   2) Dense grid prompts → decoder → at each point keep top-K diverse candidates by lowres-IoU
#   3) quality_filter → greedy box-NMS → mask-IoU de-dup (modules/nms.py)
#   4) gap-fill along X bins (rescues missed center columns)
#   5) keep-vertical-by-fold (fold角, aspect ratio, min length)
#   6) remove islands (keep largest component)
#   7) merge coaxial rect masks (angle tol / offset width / gap len)
#   8) z-score prune by major-axis length
#   9) cut overlap by vertical (prefer tall masks)
#   11) stage saves: after_nms / before_smooth / after_smooth (IDs drawn only for after_smooth)
#   12) interactive selection → save selected overlay & points.npy
#   13) extract uv_left/uv_right at a scanline with fallback window search
#   14) axis_x_on_centerline → top/bottom x (keeps your original key mapping for compatibility)

# This file is designed to mirror the behavior of your long-form sam_infer_module
# shared previously, with pragmatic fallbacks if some helper modules are absent.
# """

# from __future__ import annotations
# import sys
# from dataclasses import dataclass
# from typing import List, Optional, Tuple, Dict, Any, Union

# import argparse
# import numpy as np
# from PIL import Image
# import onnxruntime as ort
# import cv2
# from pathlib import Path as _Path
# from .modules.crop_pyramid import gen_square_crops, offset_full_mask, bbox_add_offset
# from .modules.grid_points import build_grid_xy
# from .modules.lowres_decode_utils import (
#     postprocess_like_sam,              # 既存のローカル実装を置き換え
#     stability_score_from_lowres,       # 同上
#     candidate_from_lowres,             # 同上
#     lowres_iou_binary,                 # 同上
# )

# # --- locate and add the project root that contains "modules/" to sys.path ---
# for p in _Path(__file__).resolve().parents:
#     if (p / "modules").is_dir():
#         sys.path.insert(0, str(p))
#         break

# # ===== dependent modules =====
# from modules.encoder_prepare import preprocess_for_encoder

# # new: box-NMS and dedup logic (requires modules/nms.py)
# from modules.nms import quality_filter,nms_box_greedy_spatial,dedup_by_mask_iou_spatial
# # vertical filtering (fold-angle) + axis info; fallback to old window filter if needed
# from modules.axis_angle_filter import filter_keep_vertical_by_fold, get_axis_info
# # remove islands (largest component keep); fallback inline if missing
# from modules.island import remove_islands_from_masks
# # coaxial merge / zscore / smoothing / overlay / LR
# from modules.mask_merge import merge_coaxial_rect_masks
# from modules.mask_length_zscore_filter import prune_by_major_axis_zscore
# from modules.mask_smoothing import smooth_masks_with_dp
# from modules.ids_overlay import draw_mask_ids_on_overlay
# from modules.pixel_coordinates import mask_lr_on_row
# # vertical-overlap cutter (prefer tall); fallback to simple version
# from modules.overlap_filter import keep_max_rectangularity_per_pixel_rotated
# #saving helper
# from modules.overlay_io import _save_points_and_overlay, _render_overlay_bgr

# from modules.storage_perception import find_A_B_and_save

# # ===== config =====
# DEFAULT_OUT_DIR = _Path("./results")
# IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# # ===== helpers =====
# def build_providers(device: str) -> List[str]:
#     if device == "cpu":
#         print("⚙ Using CPU for inference")
#         return ["CPUExecutionProvider"]
#     if device == "gpu":
#         print("⚙ Using GPU for inference (if available)")
#         return ["CUDAExecutionProvider", "CPUExecutionProvider"]
#     return ["CUDAExecutionProvider", "CPUExecutionProvider"]

# def _bbox_iou_xywh(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
#     x1, y1, w1, h1 = a; x2, y2, w2, h2 = b
#     xa, ya = max(x1, x2), max(y1, y2)
#     xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
#     inter = max(0, xb - xa) * max(0, yb - ya)
#     if inter <= 0:
#         return 0.0
#     u = w1*h1 + w2*h2 - inter
#     return 0.0 if u == 0 else inter / u

# def _mask_iou_downsample(a: np.ndarray, b: np.ndarray, *, side: int = 48) -> float:
#     au8 = (a.astype(np.uint8) * 255)
#     bu8 = (b.astype(np.uint8) * 255)
#     if au8.shape != (side, side):
#         au8 = cv2.resize(au8, (side, side), interpolation=cv2.INTER_NEAREST)
#     if bu8.shape != (side, side):
#         bu8 = cv2.resize(bu8, (side, side), interpolation=cv2.INTER_NEAREST)
#     A = au8 > 0
#     B = bu8 > 0
#     inter = np.logical_and(A, B).sum()
#     u = A.sum() + B.sum() - inter
#     return 0.0 if u == 0 else float(inter) / float(u)

# def gap_fill_by_xbins(selected: List[Dict[str,Any]],
#                       candidates: List[Dict[str,Any]],
#                       width: int, *,
#                       bins: int = 24,
#                       iou_thr: float = 0.98,
#                       downsample_side: int = 48) -> List[Dict[str,Any]]:
#     def xc(bb): x, y, w, h = bb; return (x + x + w) // 2
#     taken = np.zeros(bins, dtype=bool)
#     binw = max(1, width // bins)

#     for s in selected:
#         bb = s.get("bbox")
#         if not bb: continue
#         b = int(min(bins-1, max(0, xc(bb)//binw)))
#         taken[b] = True

#     pool = sorted(candidates, key=lambda d: d.get("score", 0.0), reverse=True)
#     for c in pool:
#         bb, m = c.get("bbox"), c.get("mask")
#         if not bb or m is None: continue
#         b = int(min(bins-1, max(0, xc(bb)//binw)))
#         if taken[b]:
#             continue
#         dup = False
#         for s in selected:
#             sbb, sm = s.get("bbox"), s.get("mask")
#             if not sbb or sm is None:
#                 continue
#             if _bbox_iou_xywh(bb, sbb) < 0.5:
#                 continue
#             if _mask_iou_downsample(m, sm, side=downsample_side) > iou_thr:
#                 dup = True
#                 break
#         if not dup:
#             selected.append(c)
#             taken[b] = True
#     return selected

# def mask_to_bbox2pt(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
#     """mask(H,W) -> (x0,y0,x1,y1). emptyなら None"""
#     ys, xs = np.where(mask > 0)
#     if xs.size == 0:
#         return None
#     x0 = int(xs.min())
#     y0 = int(ys.min())
#     x1 = int(xs.max())
#     y1 = int(ys.max())
#     return x0, y0, x1, y1

# # ===== core runner =====
# @dataclass
# class SamConfig:
#     encoder_path: str = "models/sam_vit_h_4b8939.encoder.onnx"
#     decoder_path: str = "models/sam_vit_h_4b8939.decoder.onnx"
#     device: str = "gpu"
#     target_len: int = 768
#     pts_side: Union[int, Tuple[int,int]] = (48, 12)
#     min_area: int = 500
#     iou_thr: float = 0.8
#     # ---- AMG crop settings ----
#     crop_overlap_ratio: float = 0.34       # 0..0.5 目安
#     crop_points_downscale: int = 2         # 層が深いほど点密度を間引く係数(>=1)

# @dataclass
# class StageSaveCfg:
#     out_dir: _Path = DEFAULT_OUT_DIR
#     save_after_nms: bool = True
#     save_before_smooth: bool = True
#     save_after_smooth: bool = True
#     save_selected: bool = True

# class SamBatchInfer_storage:
#     def __init__(self, cfg: SamConfig):
#         self.cfg = cfg
#         providers = build_providers(cfg.device)
#         self.encoder_sess = ort.InferenceSession(cfg.encoder_path, providers=providers)
#         self.decoder_sess = ort.InferenceSession(cfg.decoder_path, providers=providers)
#         self.enc_input = self.encoder_sess.get_inputs()[0]

#     def _run_point_multi(self, pack, image_embeddings, x, y, oh, ow, nh, nw,
#                          k_keep: int = 3, lowres_dist_thr: float = 0.98):
#         pt_1024 = pack.coords_to_1024(np.array([[x, y]], np.float32))[None, ...]
#         lbl     = np.array([[1]], np.float32)
#         feeds = {
#             "image_embeddings": image_embeddings,
#             "point_coords":     pt_1024,
#             "point_labels":     lbl,
#             "mask_input":       np.zeros((1,1,256,256), np.float32),
#             "has_mask_input":   np.array([0], np.float32),
#             "orig_im_size":     np.array([oh, ow], np.float32),
#         }
#         outs_names = [o.name for o in self.decoder_sess.get_outputs()]
#         outs_vals  = self.decoder_sess.run(outs_names, feeds)
#         outs       = {k: v for k, v in zip(outs_names, outs_vals)}

#         use_key   = "low_res_masks" if "low_res_masks" in outs else "masks"
#         is_logits = (use_key == "low_res_masks")
#         if "iou_predictions" in outs:
#             order = list(np.argsort(outs["iou_predictions"][0])[::-1])
#             iou_pred = outs["iou_predictions"][0]
#         else:
#             order = [0]; iou_pred = None

#         picked = []
#         picked_lowres = []
#         for idx in order:
#             lowres = outs[use_key][0, idx]
#             stab = stability_score_from_lowres(lowres, delta=0.05, is_logits=is_logits)
#             iou_p = float(iou_pred[idx]) if iou_pred is not None else 1.0
#             score = iou_p * (0.2 + 0.8 * stab)

#             # different enough from already-picked ones?
#             if all(lowres_iou_binary(lowres, prev, is_logits, side=128) < lowres_dist_thr
#                    for prev in picked_lowres):
#                 cand = candidate_from_lowres(
#                     lowres_256=lowres, score=score,
#                     orig_hw=(oh, ow), new_hw=(nh, nw), is_logits=is_logits
#                 )
#                 picked.append(cand); picked_lowres.append(lowres)
#                 if len(picked) >= k_keep:
#                     break
#         return picked

#     # SAM + 後処理
#     def infer_masks(
#         self,
#         img: Image.Image,
#         *,
#         min_fold_deg: float = 10.0,
#         min_aspect: float = 1.3,
#         axis_angle_tol_deg: float = 10.0,
#         offset_width_factor: float = 0.25,
#         gap_len_factor: float = 0.50,
#         z_lower: Optional[float] = -4.0,
#         min_len_px: float = 0.0,
#         stage_save: Optional[StageSaveCfg] = None,
#         stem_for_save: str = "image",
#         return_sam_data: bool = True,   # ★追加
#     ) -> List[np.ndarray]:
#         tm=cv2.TickMeter()
#         tm.start()
#         cfg = self.cfg
#         pack = preprocess_for_encoder(img, self.enc_input, target_length=cfg.target_len)
#         emb = self.encoder_sess.run(None, {pack.enc_name: pack.feed_img})[0]
#         if emb.ndim == 3:
#             emb = emb[None, ...]
#         image_embeddings = emb.astype(np.float32)
#         oh, ow = pack.orig_hw
#         nh, nw = pack.new_hw

#         xs, ys = build_grid_xy(ow, oh, cfg.pts_side, margin=5.0)
        
#         cands: List[Dict[str,Any]] = []
#         MIN_AREA = max(3000, self.cfg.min_area)
#         for y in ys:
#             for x in xs:
#                 for c in self._run_point_multi(pack, image_embeddings, x, y, oh, ow, nh, nw,
#                                                k_keep=3, lowres_dist_thr=0.98):
#                     if c["area"] >= MIN_AREA:
#                         cands.append(c)

#         # SAM 終了 以降，後処理に入る
#         # マスクが複数物体に被っている
#         # マスクの重なり具合を計算して閾値以上だったら消す nms : BD の重複（調べたら出てくる）
#         # 重心を通る第一主成分方向の線分を各マスクで求める（傾きで閾値を区切って書架などの他の物体マスクを消す）
#         # 欠けているマスクを埋める
#         # 重心を通る第一主成分方向の線分を各マスクで求める
#         # 同一直線にある場合，マスク統合（帯を一体化）
#         # 長方形度合いが高いようにマスクを修正（マスクのはみ出しを解消）
#         # 一番でかい面積だけ残す（飛び値を消す）
#         # 第一主成分方向の長さが短いものは消す（異なる判型になってミスる理由）
#         # 本だけにマスクをかけたい
#         tm.stop(); print(f"[TIMER] candidate_gen: {tm.getTimeMilli():.2f} ms")
#         tm.reset(); tm.start()
#         # half-pitch shifted grid
#         #if len(xs) > 1:
#             #pitch = xs[1] - xs[0]
#             #xs_shift = np.clip(xs + pitch/2, 5, ow-5)
#             #for y in ys:
#                 #for x in xs_shift:
#                     #for c in self._run_point_multi(pack, image_embeddings, x, y, oh, ow, nh, nw,
#                      #                              k_keep=3, lowres_dist_thr=0.98):
#                         #if c["area"] >= MIN_AREA:
#                             #cands.append(c)
#         tm.stop(); print(f"[TIMER] shifted_candidate_gen: {tm.getTimeMilli():.2f} ms")
#         tm=cv2.TickMeter()
#         # box-NMS + mask-dup removal
#         tm.start()
#         cands = quality_filter(cands, pred_iou_thr=None, stability_thr=None)
#         tm.stop(); print(f"[TIMER] quality_filter: {tm.getTimeMilli():.2f} ms")
#         tm.reset(); tm.start()
#         kept = nms_box_greedy_spatial(cands, iou_thr=0.7, score_key="score")
#         tm.stop(); print(f"[TIMER] low_nms: {tm.getTimeMilli():.2f} ms")
#         tm.reset(); tm.start()
#         selected = dedup_by_mask_iou_spatial(
#             kept,
#             iou_thr=0.97,
#             bbox_gate=0.70,
#             downsample_side=48,
#             score_key="score",
#         )
#         tm.stop(); print(f"[TIMER] high_nms: {tm.getTimeMilli():.2f} ms")
#         tm.reset(); tm.start()
#         selected = gap_fill_by_xbins(selected, cands, width=ow, bins=24, iou_thr=0.98, downsample_side=48)
#         tm.stop(); print(f"[TIMER] hole_filling: {tm.getTimeMilli():.2f} ms")
#         tm.reset(); tm.start()
#         masks = [c["mask"] for c in selected]
#         if stage_save and stage_save.save_after_nms:
#             _save_points_and_overlay(img, masks, stage_save.out_dir, f"{stem_for_save}_after_nms", draw_ids=False)
#         tm.stop(); print(f"[TIMER] after_nms_save: {tm.getTimeMilli():.2f} ms")
#         tm.reset(); tm.start()
#         # vertical keep by fold
#         masks, _ = filter_keep_vertical_by_fold(
#             masks,
#             min_fold_deg=min_fold_deg,
#             min_aspect=min_aspect,
#             min_len_px=min_len_px,
#         )
#         tm.stop(); print(f"[TIMER] vertical_filter: {tm.getTimeMilli():.2f} ms")
#         tm.reset(); tm.start()
#         _save_points_and_overlay(img, masks, stage_save.out_dir, f"{stem_for_save}_bookshelves", draw_ids=False)
#         tm.stop(); print(f"[TIMER] bookshelves_save: {tm.getTimeMilli():.2f} ms")
#         tm.reset(); tm.start()
#         # island removal
#         masks = remove_islands_from_masks(masks)
#         tm.stop(); print(f"[TIMER] island_removal: {tm.getTimeMilli():.2f} ms")
#         tm.reset(); tm.start()
#         #coaxial merge → zscore prune
#         masks = merge_coaxial_rect_masks(
#             masks,
#             axis_angle_tol_deg=axis_angle_tol_deg,
#             offset_width_factor=offset_width_factor,
#             gap_len_factor=gap_len_factor,
#             rectify=False
#         )
#         tm.stop(); print(f"[TIMER] coaxial_merge: {tm.getTimeMilli():.2f} ms")
#         tm.reset(); tm.start()
#         _save_points_and_overlay(img, masks, stage_save.out_dir, f"{stem_for_save}_belly-band_merged", draw_ids=False)
#         tm.stop(); print(f"[TIMER] belly_band_save: {tm.getTimeMilli():.2f} ms")
#         tm.reset(); tm.start()
#         # overlap cut preferring vertical
#         masks = keep_max_rectangularity_per_pixel_rotated(masks)
#         tm.stop(); print(f"[TIMER] overlap_cut: {tm.getTimeMilli():.2f} ms")
#         tm.reset(); tm.start()    
#         #coaxial merge → zscore prune
#         masks = merge_coaxial_rect_masks(
#             masks,
#             axis_angle_tol_deg=axis_angle_tol_deg,
#             offset_width_factor=offset_width_factor,
#             gap_len_factor=gap_len_factor,
#             rectify=False
#         )
#         tm.stop(); print(f"[TIMER] coaxial_merge_2nd: {tm.getTimeMilli():.2f} ms")
#         tm.reset(); tm.start()
#         #from modules.filter_by_left_y import run_step1_filter_by_bottom_left_y
#         #masks = run_step1_filter_by_bottom_left_y(masks, z_thr=1.2) #step1
#         tm.stop(); print(f"[TIMER] filter_by_bottom_left: {tm.getTimeMilli():.2f} ms")
#         tm.reset(); tm.start()
#         if z_lower is not None:
#             masks = prune_by_major_axis_zscore(masks, z_lower=z_lower, min_len_px=min_len_px)
#         tm.stop(); print(f"[TIMER] zscore_prune: {tm.getTimeMilli():.2f} ms")
#         tm.reset(); 
#         if stage_save and stage_save.save_before_smooth:
#             _save_points_and_overlay(img, masks, stage_save.out_dir, f"{stem_for_save}_before_smooth", draw_ids=True)
#         #if z_lower is not None:
#             #masks = prune_by_major_axis_zscore(masks, z_lower=z_lower, min_len_px=min_len_px)
#         #_save_points_and_overlay(img, masks, stage_save.out_dir, f"{stem_for_save}_zscore2", draw_ids=False)    
#         sam_data: List[Dict[str, Any]] = []
#         for i, m in enumerate(masks, start=1):
#             bb = mask_to_bbox2pt(m)
#             if bb is None:
#                 sam_data.append({
#                     "name": f"book{i}",
#                     "box": {"x1": None, "y1": None, "x2": None, "y2": None},
#                 })
#             else:
#                 x0, y0, x1, y1 = bb
#                 sam_data.append({
#                     "name": f"book{i}",
#                     # intersection_area が期待してるキーに合わせる
#                     "box": {"x1": x0, "y1": y0, "x2": x1, "y2": y1},
#                 })
#         if return_sam_data:
#             return masks, sam_data

#         return masks
#     # single image path with interactive selection
#     def run_on_image_path_storage(
#         self,
#         img_path: _Path | str,
#         *,
#         stage_save_dir: Optional[_Path] = None,
#         swap_lr_output: bool = False,
#         min_fold_deg: float = 70.0,
#         min_aspect: float = 1.3,
#         axis_angle_tol_deg: float = 10.0,
#         offset_width_factor: float = 0.25,
#         gap_len_factor: float = 0.50,
#         z_lower: Optional[float] = -0.1,
#         min_len_px: float = 0.0,
#         y_from_top_px: int = 230,
#         scan_radius_px: int = 80,
#         skip_selection: bool = False,
#     ) -> Dict[str, Any]:
#         out_dir = stage_save_dir or DEFAULT_OUT_DIR
#         img_path = _Path(img_path)
#         img = Image.open(img_path).convert("RGB")
#         stem = img_path.stem
#         out_dir = stage_save_dir or DEFAULT_OUT_DIR
#         img_path = _Path(img_path)
#         img = Image.open(img_path).convert("RGB")
#         stem = img_path.stem

#         stage_cfg = StageSaveCfg(out_dir=out_dir)
#         ret = self.infer_masks(
#             img,
#             min_fold_deg=min_fold_deg,
#             min_aspect=min_aspect,
#             axis_angle_tol_deg=axis_angle_tol_deg,
#             offset_width_factor=offset_width_factor,
#             gap_len_factor=gap_len_factor,
#             z_lower=z_lower,
#             min_len_px=min_len_px,
#             stage_save=stage_cfg,
#             stem_for_save=stem,
#         )

#         print(f"[DEBUG] infer_masks return type = {type(ret)}")

#         if isinstance(ret, tuple):
#             print(f"[DEBUG] infer_masks returned tuple len = {len(ret)}")
#             masks = ret[0]
#             sam_data = ret[1] if len(ret) > 1 else None
#         else:
#             masks = ret
#             sam_data = None

#         print(f"[DEBUG] actual masks len = {len(masks)}")
#         for k, m in enumerate(masks):
#             arr = np.asarray(m)
#             print(f"[DEBUG] actual mask[{k}] shape={arr.shape}, sum={np.sum(arr > 0)}")

#         # --- ここから追加処理 ---
#         from modules.filters_for_storage import run_steps_2_to_7
#         from modules.axis_overlay import draw_axes_on_overlay
#         W, H = img.size  # PIL は (width, height)

#         # run_steps_2_to_7 は「ソート済みマスク」とコンテキストを返す想定
#         masks_sorted, ctx = run_steps_2_to_7(
#             masks,
#             image_width=W,
#             line_len=504.0,
#         )

#         # ガイドラインの始点/終点（float）を取得
#         line_p0 = ctx.get("line_p0")
#         line_p1 = ctx.get("line_p1")

#         # RGB画像 + マスクをオーバーレイした BGR 画像を生成（IDは描かない）
#         ov_bgr = _render_overlay_bgr(img, masks_sorted, draw_ids=False)
#         # 2) BGR -> RGB -> PIL に変換して、全部のマスクに中心軸を描画
#         ov_rgb = cv2.cvtColor(ov_bgr, cv2.COLOR_BGR2RGB)
#         ov_pil = Image.fromarray(ov_rgb)

#         # ★ここで「全部のマスク」に主軸をオーバーレイ★
#         #v_pil = draw_axes_on_overlay(
#            #ov_pil,
#            #masks_sorted,
#            #color=(255, 255, 0),  # BGR: 黄色っぽい色 
#            #thickness=3,
#        #)
#         ov_rgb2 = np.array(ov_pil)                       # RGB (H,W,3)
#         ov_bgr2 = cv2.cvtColor(ov_rgb2, cv2.COLOR_RGB2BGR)

#         p0_int = p1_int = None
#         if line_p0 is not None and line_p1 is not None:
#             p0_int = (int(round(line_p0[0])), int(round(line_p0[1])))
#             p1_int = (int(round(line_p1[0])), int(round(line_p1[1])))

#             # ここは OpenCV なので numpy(BGR) に対して描画する
#             cv2.line(ov_bgr2, p0_int, p1_int, (255, 0, 255), 4)#紫
#             cv2.circle(ov_bgr2, p0_int, 4, (0, 255, 255), -1)#黄
#             cv2.circle(ov_bgr2, p1_int, 4, (255, 255, 255), -1)#白

#         sub_p0 = ctx.get("sub_line_p0")
#         sub_p1 = ctx.get("sub_line_p1")
#         if sub_p0 is not None and sub_p1 is not None:
#             sp0 = (int(round(sub_p0[0])), int(round(sub_p0[1])))
#             sp1 = (int(round(sub_p1[0])), int(round(sub_p1[1])))

#             cv2.line(ov_bgr2, sp0, sp1, (0, 0, 255), 4)  #赤
#             cv2.circle(ov_bgr2, sp0, 4, (0, 255, 0), -1) #緑
#             cv2.circle(ov_bgr2, sp1, 4, (255, 0, 0), -1) #青
        
            
#         # 画像として保存（OpenCVはBGRのまま保存してOK）
#         out_dir.mkdir(parents=True, exist_ok=True)
#         guide_path = out_dir / f"{stem}_guideline_overlay.png"
#         cv2.imwrite(str(guide_path), ov_bgr2)
#         right_is_tilted = ctx.get("right_is_tilted", False)
#         is_right_half = ctx.get("right_half", False)
#         # インタラクティブな「マスクID選択」は一切やらないので、
#         # selected_index_1based は固定値などにしておく（CLI で NO_MASK を出したくなければ 1 にする）
#         res: Dict[str, Any] = {
#             "masks": masks_sorted,
#             "line_p0": p0_int,
#             "line_p1": p1_int,
#             "sub_line_p0": sub_p0,
#             "sub_line_p1": sub_p1,
#             "overlay_path": guide_path,
#             "selected_index_1based": 1 if p0_int is not None else None,
#             "right_is_tilted": right_is_tilted,
#             "is_right_half": is_right_half,
#             "pair_indices": ctx.get("pair_indices"),     # (i, j)
#             "centers": ctx.get("centers"),    
#         }
#         return res

        

# # ===== CLI =====
# def _parse_args():
#     ap = argparse.ArgumentParser()
#     # modes
#     ap.add_argument("--image", type=str, help="単一の入力画像パス")
#     ap.add_argument("--input_dir", type=str, help="画像フォルダ（--image無指定時に使用）")
#     ap.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR), help="出力フォルダ")
#     ap.add_argument("target", nargs="?", help="画像ファイル or 画像フォルダ（省略時は ./images）")

#     # model
#     ap.add_argument("--encoder", default="/home/book/pro_book_SAM3/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx")
#     ap.add_argument("--decoder", default="/home/book/pro_book_SAM3/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx")
#     ap.add_argument("--device", choices=["gpu", "cpu", "auto"], default="gpu")
#     # AMG crop params
#     ap.add_argument("--crop_overlap_ratio", type=float, default=0.34, help="クロップ重なり率(0..0.5)")
#     ap.add_argument("--crop_points_downscale", type=int, default=2, help="層ごとの点密度間引き係数(>=1)")
#     # grid / candidates
#     #ap.add_argument("--pts_side", type=int, default=36)
#     ap.add_argument("--pts_side_x", type=int, default=60, help="横方向のグリッド数（指定時は --pts_side_y とセット）")
#     ap.add_argument("--pts_side_y", type=int, default=24, help="縦方向のグリッド数（指定時は --pts_side_x とセット）")
#     ap.add_argument("--min_area", type=int, default=800)
#     ap.add_argument("--iou_thr", type=float, default=0.8)  # reserved

#     # zscore / axis min len
#     ap.add_argument("--z_lower", type=float, default=-1.0)
#     ap.add_argument("--min_len_px", type=float, default=0.0)

#     # vertical keep (fold-based)
#     ap.add_argument("--min_fold_deg", type=float, default=10.0, help="fold角の下限（0=水平, 90=垂直）")
#     ap.add_argument("--min_aspect", type=float, default=1.3, help="縦長判定の L/W 下限")

#     # coaxial merge params
#     ap.add_argument("--axis_angle_tol", type=float, default=4.0)
#     ap.add_argument("--offset_width_factor", type=float, default=0.15)
#     ap.add_argument("--gap_len_factor", type=float, default=0.06)

#     # LR extraction
#     ap.add_argument("--swap_lr_output", action="store_true", help="左右の出力を入れ替える")
#     ap.add_argument("--y_from_top_px", type=int, default=230, help="左右端抽出に用いる水平ライン（画像上端からのpx）")
#     ap.add_argument("--scan_radius_px", type=int, default=80, help="指定ラインで交差しない場合に±探索する半径[px]")

#     ap.add_argument("--skip_selection", action="store_true", help="マスク選択をスキップして終了（UVなどは出力しない）")

#     return ap.parse_args()

# def _main_cli():
#     args = _parse_args()
#     out_dir = _Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
#     tm = cv2.TickMeter()
#     pts_side_val = (args.pts_side_x, args.pts_side_y)
#     tm.start()
#     cfg = SamConfig(
#         encoder_path=args.encoder,
#         decoder_path=args.decoder,
#         device=args.device,
#         pts_side=pts_side_val,
#         min_area=args.min_area,
#         iou_thr=args.iou_thr,
#         crop_overlap_ratio=args.crop_overlap_ratio,
#         crop_points_downscale=max(1, args.crop_points_downscale),
#     )
#     runner = SamBatchInfer_storage(cfg)
#     tm.stop(); print(f"[TIMER] model_load: {tm.getTimeMilli():.2f} ms")

#     # --- 単一画像モード ----------------------------------------------------
#     if args.image:
#         res = runner.run_on_image_path_storage(
#             args.image,
#             stage_save_dir=out_dir,
#             swap_lr_output=args.swap_lr_output,
#             min_fold_deg=args.min_fold_deg,
#             min_aspect=args.min_aspect,
#             axis_angle_tol_deg=args.axis_angle_tol,
#             offset_width_factor=args.offset_width_factor,
#             gap_len_factor=args.gap_len_factor,
#             z_lower=args.z_lower,
#             min_len_px=args.min_len_px,
#             y_from_top_px=args.y_from_top_px,
#             scan_radius_px=args.scan_radius_px,
#             skip_selection=args.skip_selection,
#         )
#         overlay_path = res.get("overlay_path")
#         if overlay_path is not None:
#             print(f"[SAVE] guideline overlay: {overlay_path}")
#         else:
#             print("[WARN] guideline overlay was not generated")
#         return

#     # --- target がファイル or ディレクトリかで分岐 ------------------------
#     if args.target:
#         tgt = _Path(args.target)
#         if tgt.is_file():
#             res = runner.run_on_image_path_storage(
#                 tgt,
#                 stage_save_dir=out_dir,
#                 swap_lr_output=args.swap_lr_output,
#                 min_fold_deg=args.min_fold_deg,
#                 min_aspect=args.min_aspect,
#                 axis_angle_tol_deg=args.axis_angle_tol,
#                 offset_width_factor=args.offset_width_factor,
#                 gap_len_factor=args.gap_len_factor,
#                 z_lower=args.z_lower,
#                 min_len_px=args.min_len_px,
#                 y_from_top_px=args.y_from_top_px,
#                 scan_radius_px=args.scan_radius_px,
#                 skip_selection=args.skip_selection,
#             )
#             overlay_path = res.get("overlay_path")
#             if overlay_path is not None:
#                 print(f"[SAVE] guideline overlay: {overlay_path}")
#             else:
#                 print("[WARN] guideline overlay was not generated")
#             return
#         elif tgt.is_dir():
#             in_dir = tgt
#         else:
#             print("Target not found:", tgt)
#             return
#     else:
#         # batch mode: iterate over images in a folder
#         in_dir = _Path(args.input_dir) if args.input_dir else _Path("images")

#     img_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in IMG_EXT])
#     if not img_paths:
#         print("No images found in:", in_dir)
#         return

#     # --- 複数画像バッチ処理 -----------------------------------------------
#     for img_path in img_paths:
#         print(f"==> {img_path.name}")
#         res = runner.run_on_image_path_storage(
#             img_path,
#             stage_save_dir=out_dir,
#             swap_lr_output=args.swap_lr_output,
#             min_fold_deg=args.min_fold_deg,
#             min_aspect=args.min_aspect,
#             axis_angle_tol_deg=args.axis_angle_tol,
#             offset_width_factor=args.offset_width_factor,
#             gap_len_factor=args.gap_len_factor,
#             z_lower=args.z_lower,
#             min_len_px=args.min_len_px,
#             y_from_top_px=args.y_from_top_px,
#             scan_radius_px=args.scan_radius_px,
#             skip_selection=args.skip_selection,
#         )
#         overlay_path = res.get("overlay_path")
#         if overlay_path is not None:
#             print(f"  [SAVE] {overlay_path}")
#         else:
#             print("  [WARN] guideline overlay was not generated")

#     print("✔ Done:", out_dir)


# if __name__ == "__main__":
#     _main_cli()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_infer_v2.py — SAM batch inference (pipeline-matched to your sam_infer_module)

Pipeline (per image):
  1) Encoder embedding
  2) Dense grid prompts → decoder → at each point keep top-K diverse candidates by lowres-IoU
  3) quality_filter → greedy box-NMS → mask-IoU de-dup (modules/nms.py)
  4) gap-fill along X bins (rescues missed center columns)
  5) keep-vertical-by-fold (fold角, aspect ratio, min length)
  6) remove islands (keep largest component)
  7) merge coaxial rect masks (angle tol / offset width / gap len)
  8) z-score prune by major-axis length
  9) cut overlap by vertical (prefer tall masks)
  11) stage saves: after_nms / before_smooth / after_smooth (IDs drawn only for after_smooth)
  12) interactive selection → save selected overlay & points.npy
  13) extract uv_left/uv_right at a scanline with fallback window search
  14) axis_x_on_centerline → top/bottom x (keeps your original key mapping for compatibility)

This file is designed to mirror the behavior of your long-form sam_infer_module
shared previously, with pragmatic fallbacks if some helper modules are absent.
"""

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
from .modules.crop_pyramid import gen_square_crops, offset_full_mask, bbox_add_offset
from .modules.grid_points import build_grid_xy
from .modules.lowres_decode_utils import (
    postprocess_like_sam,              # 既存のローカル実装を置き換え
    stability_score_from_lowres,       # 同上
    candidate_from_lowres,             # 同上
    lowres_iou_binary,                 # 同上
)

# --- locate and add the project root that contains "modules/" to sys.path ---
for p in _Path(__file__).resolve().parents:
    if (p / "modules").is_dir():
        sys.path.insert(0, str(p))
        break

# ===== dependent modules =====
from modules.encoder_prepare import preprocess_for_encoder

# new: box-NMS and dedup logic (requires modules/nms.py)
from modules.nms import quality_filter,nms_box_greedy_spatial,dedup_by_mask_iou_spatial
# vertical filtering (fold-angle) + axis info; fallback to old window filter if needed
from modules.axis_angle_filter import filter_keep_vertical_by_fold, get_axis_info
# remove islands (largest component keep); fallback inline if missing
from modules.island import remove_islands_from_masks
# coaxial merge / zscore / smoothing / overlay / LR
from modules.mask_merge import merge_coaxial_rect_masks
from modules.mask_length_zscore_filter import prune_by_major_axis_zscore
from modules.mask_smoothing import smooth_masks_with_dp
from modules.ids_overlay import draw_mask_ids_on_overlay
from modules.pixel_coordinates import mask_lr_on_row
# vertical-overlap cutter (prefer tall); fallback to simple version
from modules.overlap_filter import keep_max_rectangularity_per_pixel_rotated
#saving helper
from modules.overlay_io import _save_points_and_overlay, _render_overlay_bgr

from modules.storage_perception import find_A_B_and_save

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

def mask_to_bbox2pt(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """mask(H,W) -> (x0,y0,x1,y1). emptyなら None"""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return x0, y0, x1, y1

# ===== core runner =====
@dataclass
class SamConfig:
    encoder_path: str = "models/sam_vit_h_4b8939.encoder.onnx"
    decoder_path: str = "models/sam_vit_h_4b8939.decoder.onnx"
    device: str = "gpu"
    target_len: int = 768

    # Dense grid prompt settings.
    # 既存値は (48, 12) = 576点．candidate_gen はほぼこの点数に比例する．
    # get_book_points_revised.py のoffline高速版では，まず (36, 10) = 360点を使う．
    pts_side: Union[int, Tuple[int,int]] = (48, 12)

    min_area: int = 500
    iou_thr: float = 0.8

    # Decoder candidate settings.
    # 既存値は k_keep=3．高速版では k_keep=1 にして，各点の最良候補だけ使う．
    decoder_k_keep: int = 3
    lowres_dist_thr: float = 0.98
    grid_margin_px: float = 5.0
    min_area_floor: int = 3000

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
    save_bookshelves: bool = True
    save_belly_band: bool = True

class SamBatchInfer_storage:
    def __init__(self, cfg: SamConfig):
        self.cfg = cfg
        providers = build_providers(cfg.device)
        self.encoder_sess = ort.InferenceSession(cfg.encoder_path, providers=providers)
        self.decoder_sess = ort.InferenceSession(cfg.decoder_path, providers=providers)
        self.enc_input = self.encoder_sess.get_inputs()[0]

        # decoder output names are constant; do not rebuild this list for every grid point.
        self.decoder_out_names = [o.name for o in self.decoder_sess.get_outputs()]

    def _run_point_multi(self, pack, image_embeddings, x, y, oh, ow, nh, nw,
                         k_keep: int = 3, lowres_dist_thr: float = 0.98):
        pt_1024 = pack.coords_to_1024(np.array([[x, y]], np.float32))[None, ...]
        lbl     = np.array([[1]], np.float32)
        feeds = {
            "image_embeddings": image_embeddings,
            "point_coords":     pt_1024,
            "point_labels":     lbl,
            "mask_input":       np.zeros((1,1,256,256), np.float32),
            "has_mask_input":   np.array([0], np.float32),
            "orig_im_size":     np.array([oh, ow], np.float32),
        }
        outs_vals  = self.decoder_sess.run(self.decoder_out_names, feeds)
        outs       = {k: v for k, v in zip(self.decoder_out_names, outs_vals)}

        use_key   = "low_res_masks" if "low_res_masks" in outs else "masks"
        is_logits = (use_key == "low_res_masks")
        if "iou_predictions" in outs:
            order = list(np.argsort(outs["iou_predictions"][0])[::-1])
            iou_pred = outs["iou_predictions"][0]
        else:
            order = [0]; iou_pred = None

        # Fast path: only keep the best decoder output for this prompt.
        # This reduces per-point CPU work and the number of full-resolution masks entering NMS.
        if int(k_keep) <= 1:
            idx = int(order[0])
            lowres = outs[use_key][0, idx]
            stab = stability_score_from_lowres(lowres, delta=0.05, is_logits=is_logits)
            iou_p = float(iou_pred[idx]) if iou_pred is not None else 1.0
            score = iou_p * (0.2 + 0.8 * stab)
            return [
                candidate_from_lowres(
                    lowres_256=lowres, score=score,
                    orig_hw=(oh, ow), new_hw=(nh, nw), is_logits=is_logits
                )
            ]

        picked = []
        picked_lowres = []
        for idx in order:
            lowres = outs[use_key][0, idx]
            stab = stability_score_from_lowres(lowres, delta=0.05, is_logits=is_logits)
            iou_p = float(iou_pred[idx]) if iou_pred is not None else 1.0
            score = iou_p * (0.2 + 0.8 * stab)

            # different enough from already-picked ones?
            if all(lowres_iou_binary(lowres, prev, is_logits, side=128) < lowres_dist_thr
                   for prev in picked_lowres):
                cand = candidate_from_lowres(
                    lowres_256=lowres, score=score,
                    orig_hw=(oh, ow), new_hw=(nh, nw), is_logits=is_logits
                )
                picked.append(cand); picked_lowres.append(lowres)
                if len(picked) >= int(k_keep):
                    break
        return picked

    # SAM + 後処理
    def infer_masks(
        self,
        img: Image.Image,
        *,
        min_fold_deg: float = 10.0,
        min_aspect: float = 1.3,
        axis_angle_tol_deg: float = 10.0,
        offset_width_factor: float = 0.25,
        gap_len_factor: float = 0.50,
        z_lower: Optional[float] = -4.0,
        min_len_px: float = 0.0,
        stage_save: Optional[StageSaveCfg] = None,
        stem_for_save: str = "image",
        return_sam_data: bool = True,   # ★追加
        depth_u16: Optional[np.ndarray] = None,
        depth_merge_tolerance_raw: Optional[float] = 30.0,
        depth_merge_min_valid_px: int = 30,
    ) -> List[np.ndarray]:
        tm=cv2.TickMeter()
        tm.start()
        cfg = self.cfg
        pack = preprocess_for_encoder(img, self.enc_input, target_length=cfg.target_len)
        emb = self.encoder_sess.run(None, {pack.enc_name: pack.feed_img})[0]
        if emb.ndim == 3:
            emb = emb[None, ...]
        image_embeddings = emb.astype(np.float32)
        oh, ow = pack.orig_hw
        nh, nw = pack.new_hw

        # cfg.pts_side / cfg.decoder_k_keep / cfg.lowres_dist_thr を必ず反映する．
        # ここが固定値のままだと，get_book_points_revised.py 側で
        # sam_pts_side や sam_decoder_k_keep を変更しても高速化設定が効かない．
        xs, ys = build_grid_xy(ow, oh, cfg.pts_side, margin=float(cfg.grid_margin_px))

        cands: List[Dict[str,Any]] = []
        MIN_AREA = max(int(cfg.min_area_floor), int(cfg.min_area))
        decoder_call_count = 0
        for y in ys:
            for x in xs:
                decoder_call_count += 1
                for c in self._run_point_multi(
                    pack,
                    image_embeddings,
                    x,
                    y,
                    oh,
                    ow,
                    nh,
                    nw,
                    k_keep=int(cfg.decoder_k_keep),
                    lowres_dist_thr=float(cfg.lowres_dist_thr),
                ):
                    if c["area"] >= MIN_AREA:
                        cands.append(c)

        print(
            f"[SAM2 FAST] grid={len(xs)}x{len(ys)} "
            f"decoder_calls={decoder_call_count} "
            f"k_keep={int(cfg.decoder_k_keep)} "
            f"min_area={MIN_AREA}",
            flush=True,
        )

        # SAM 終了 以降，後処理に入る
        # マスクが複数物体に被っている
        # マスクの重なり具合を計算して閾値以上だったら消す nms : BD の重複（調べたら出てくる）
        # 重心を通る第一主成分方向の線分を各マスクで求める（傾きで閾値を区切って書架などの他の物体マスクを消す）
        # 欠けているマスクを埋める
        # 重心を通る第一主成分方向の線分を各マスクで求める
        # 同一直線にある場合，マスク統合（帯を一体化）
        # 長方形度合いが高いようにマスクを修正（マスクのはみ出しを解消）
        # 一番でかい面積だけ残す（飛び値を消す）
        # 第一主成分方向の長さが短いものは消す（異なる判型になってミスる理由）
        # 本だけにマスクをかけたい
        tm.stop(); print(f"[TIMER] candidate_gen: {tm.getTimeMilli():.2f} ms")
        tm.reset(); tm.start()
        # half-pitch shifted grid
        #if len(xs) > 1:
            #pitch = xs[1] - xs[0]
            #xs_shift = np.clip(xs + pitch/2, 5, ow-5)
            #for y in ys:
                #for x in xs_shift:
                    #for c in self._run_point_multi(pack, image_embeddings, x, y, oh, ow, nh, nw,
                     #                              k_keep=3, lowres_dist_thr=0.98):
                        #if c["area"] >= MIN_AREA:
                            #cands.append(c)
        tm.stop(); print(f"[TIMER] shifted_candidate_gen: {tm.getTimeMilli():.2f} ms")
        tm=cv2.TickMeter()
        # box-NMS + mask-dup removal
        tm.start()
        cands = quality_filter(cands, pred_iou_thr=None, stability_thr=None)
        tm.stop(); print(f"[TIMER] quality_filter: {tm.getTimeMilli():.2f} ms")
        tm.reset(); tm.start()
        kept = nms_box_greedy_spatial(cands, iou_thr=0.7, score_key="score")
        tm.stop(); print(f"[TIMER] low_nms: {tm.getTimeMilli():.2f} ms")
        tm.reset(); tm.start()
        selected = dedup_by_mask_iou_spatial(
            kept,
            iou_thr=0.97,
            bbox_gate=0.70,
            downsample_side=48,
            score_key="score",
        )
        tm.stop(); print(f"[TIMER] high_nms: {tm.getTimeMilli():.2f} ms")
        tm.reset(); tm.start()
        selected = gap_fill_by_xbins(selected, cands, width=ow, bins=24, iou_thr=0.98, downsample_side=48)
        tm.stop(); print(f"[TIMER] hole_filling: {tm.getTimeMilli():.2f} ms")
        tm.reset(); tm.start()
        masks = [c["mask"] for c in selected]
        if stage_save and stage_save.save_after_nms:
            _save_points_and_overlay(img, masks, stage_save.out_dir, f"{stem_for_save}_after_nms", draw_ids=False)
        tm.stop(); print(f"[TIMER] after_nms_save: {tm.getTimeMilli():.2f} ms")
        tm.reset(); tm.start()
        # vertical keep by fold
        masks, _ = filter_keep_vertical_by_fold(
            masks,
            min_fold_deg=min_fold_deg,
            min_aspect=min_aspect,
            min_len_px=min_len_px,
        )
        tm.stop(); print(f"[TIMER] vertical_filter: {tm.getTimeMilli():.2f} ms")
        tm.reset(); tm.start()
        if stage_save and getattr(stage_save, "save_bookshelves", True):
            _save_points_and_overlay(img, masks, stage_save.out_dir, f"{stem_for_save}_bookshelves", draw_ids=False)
        tm.stop(); print(f"[TIMER] bookshelves_save: {tm.getTimeMilli():.2f} ms")
        tm.reset(); tm.start()
        # island removal
        masks = remove_islands_from_masks(masks)
        tm.stop(); print(f"[TIMER] island_removal: {tm.getTimeMilli():.2f} ms")
        tm.reset(); tm.start()
        #coaxial merge → zscore prune
        masks = merge_coaxial_rect_masks(
            masks,
            axis_angle_tol_deg=axis_angle_tol_deg,
            offset_width_factor=offset_width_factor,
            gap_len_factor=gap_len_factor,
            rectify=False,
            depth_u16=depth_u16,
            depth_merge_tolerance_raw=depth_merge_tolerance_raw,
            depth_merge_min_valid_px=depth_merge_min_valid_px,
        )
        tm.stop(); print(f"[TIMER] coaxial_merge: {tm.getTimeMilli():.2f} ms")
        tm.reset(); tm.start()
        if stage_save and getattr(stage_save, "save_belly_band", True):
            _save_points_and_overlay(img, masks, stage_save.out_dir, f"{stem_for_save}_belly-band_merged", draw_ids=False)
        tm.stop(); print(f"[TIMER] belly_band_save: {tm.getTimeMilli():.2f} ms")
        tm.reset(); tm.start()
        # overlap cut preferring vertical
        masks = keep_max_rectangularity_per_pixel_rotated(masks)
        tm.stop(); print(f"[TIMER] overlap_cut: {tm.getTimeMilli():.2f} ms")
        tm.reset(); tm.start()    
        #coaxial merge → zscore prune
        masks = merge_coaxial_rect_masks(
            masks,
            axis_angle_tol_deg=axis_angle_tol_deg,
            offset_width_factor=offset_width_factor,
            gap_len_factor=gap_len_factor,
            rectify=False,
            depth_u16=depth_u16,
            depth_merge_tolerance_raw=depth_merge_tolerance_raw,
            depth_merge_min_valid_px=depth_merge_min_valid_px,
        )
        tm.stop(); print(f"[TIMER] coaxial_merge_2nd: {tm.getTimeMilli():.2f} ms")
        tm.reset(); tm.start()
        #from modules.filter_by_left_y import run_step1_filter_by_bottom_left_y
        #masks = run_step1_filter_by_bottom_left_y(masks, z_thr=1.2) #step1
        tm.stop(); print(f"[TIMER] filter_by_bottom_left: {tm.getTimeMilli():.2f} ms")
        tm.reset(); tm.start()
        if z_lower is not None:
            masks = prune_by_major_axis_zscore(masks, z_lower=z_lower, min_len_px=min_len_px)
        tm.stop(); print(f"[TIMER] zscore_prune: {tm.getTimeMilli():.2f} ms")
        tm.reset(); 
        if stage_save and stage_save.save_before_smooth:
            _save_points_and_overlay(img, masks, stage_save.out_dir, f"{stem_for_save}_before_smooth", draw_ids=True)
        #if z_lower is not None:
            #masks = prune_by_major_axis_zscore(masks, z_lower=z_lower, min_len_px=min_len_px)
        #_save_points_and_overlay(img, masks, stage_save.out_dir, f"{stem_for_save}_zscore2", draw_ids=False)    
        sam_data: List[Dict[str, Any]] = []
        for i, m in enumerate(masks, start=1):
            bb = mask_to_bbox2pt(m)
            if bb is None:
                sam_data.append({
                    "name": f"book{i}",
                    "box": {"x1": None, "y1": None, "x2": None, "y2": None},
                })
            else:
                x0, y0, x1, y1 = bb
                sam_data.append({
                    "name": f"book{i}",
                    # intersection_area が期待してるキーに合わせる
                    "box": {"x1": x0, "y1": y0, "x2": x1, "y2": y1},
                })
        if return_sam_data:
            return masks, sam_data

        return masks
    # single image path with interactive selection
    def run_on_image_path_storage(
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
        z_lower: Optional[float] = -0.1,
        min_len_px: float = 0.0,
        y_from_top_px: int = 230,
        scan_radius_px: int = 80,
        skip_selection: bool = False,
    ) -> Dict[str, Any]:
        out_dir = stage_save_dir or DEFAULT_OUT_DIR
        img_path = _Path(img_path)
        img = Image.open(img_path).convert("RGB")
        stem = img_path.stem
        out_dir = stage_save_dir or DEFAULT_OUT_DIR
        img_path = _Path(img_path)
        img = Image.open(img_path).convert("RGB")
        stem = img_path.stem

        stage_cfg = StageSaveCfg(out_dir=out_dir)
        ret = self.infer_masks(
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

        print(f"[DEBUG] infer_masks return type = {type(ret)}")

        if isinstance(ret, tuple):
            print(f"[DEBUG] infer_masks returned tuple len = {len(ret)}")
            masks = ret[0]
            sam_data = ret[1] if len(ret) > 1 else None
        else:
            masks = ret
            sam_data = None

        print(f"[DEBUG] actual masks len = {len(masks)}")
        for k, m in enumerate(masks):
            arr = np.asarray(m)
            print(f"[DEBUG] actual mask[{k}] shape={arr.shape}, sum={np.sum(arr > 0)}")

        # --- ここから追加処理 ---
        from modules.filters_for_storage import run_steps_2_to_7
        from modules.axis_overlay import draw_axes_on_overlay
        W, H = img.size  # PIL は (width, height)

        # run_steps_2_to_7 は「ソート済みマスク」とコンテキストを返す想定
        masks_sorted, ctx = run_steps_2_to_7(
            masks,
            image_width=W,
            line_len=504.0,
        )

        # ガイドラインの始点/終点（float）を取得
        line_p0 = ctx.get("line_p0")
        line_p1 = ctx.get("line_p1")

        # RGB画像 + マスクをオーバーレイした BGR 画像を生成（IDは描かない）
        ov_bgr = _render_overlay_bgr(img, masks_sorted, draw_ids=False)
        # 2) BGR -> RGB -> PIL に変換して、全部のマスクに中心軸を描画
        ov_rgb = cv2.cvtColor(ov_bgr, cv2.COLOR_BGR2RGB)
        ov_pil = Image.fromarray(ov_rgb)

        # ★ここで「全部のマスク」に主軸をオーバーレイ★
        #v_pil = draw_axes_on_overlay(
           #ov_pil,
           #masks_sorted,
           #color=(255, 255, 0),  # BGR: 黄色っぽい色 
           #thickness=3,
       #)
        ov_rgb2 = np.array(ov_pil)                       # RGB (H,W,3)
        ov_bgr2 = cv2.cvtColor(ov_rgb2, cv2.COLOR_RGB2BGR)

        p0_int = p1_int = None
        if line_p0 is not None and line_p1 is not None:
            p0_int = (int(round(line_p0[0])), int(round(line_p0[1])))
            p1_int = (int(round(line_p1[0])), int(round(line_p1[1])))

            # ここは OpenCV なので numpy(BGR) に対して描画する
            cv2.line(ov_bgr2, p0_int, p1_int, (255, 0, 255), 4)#紫
            cv2.circle(ov_bgr2, p0_int, 4, (0, 255, 255), -1)#黄
            cv2.circle(ov_bgr2, p1_int, 4, (255, 255, 255), -1)#白

        sub_p0 = ctx.get("sub_line_p0")
        sub_p1 = ctx.get("sub_line_p1")
        if sub_p0 is not None and sub_p1 is not None:
            sp0 = (int(round(sub_p0[0])), int(round(sub_p0[1])))
            sp1 = (int(round(sub_p1[0])), int(round(sub_p1[1])))

            cv2.line(ov_bgr2, sp0, sp1, (0, 0, 255), 4)  #赤
            cv2.circle(ov_bgr2, sp0, 4, (0, 255, 0), -1) #緑
            cv2.circle(ov_bgr2, sp1, 4, (255, 0, 0), -1) #青
        
            
        # 画像として保存（OpenCVはBGRのまま保存してOK）
        out_dir.mkdir(parents=True, exist_ok=True)
        guide_path = out_dir / f"{stem}_guideline_overlay.png"
        cv2.imwrite(str(guide_path), ov_bgr2)
        right_is_tilted = ctx.get("right_is_tilted", False)
        is_right_half = ctx.get("right_half", False)
        # インタラクティブな「マスクID選択」は一切やらないので、
        # selected_index_1based は固定値などにしておく（CLI で NO_MASK を出したくなければ 1 にする）
        res: Dict[str, Any] = {
            "masks": masks_sorted,
            "line_p0": p0_int,
            "line_p1": p1_int,
            "sub_line_p0": sub_p0,
            "sub_line_p1": sub_p1,
            "overlay_path": guide_path,
            "selected_index_1based": 1 if p0_int is not None else None,
            "right_is_tilted": right_is_tilted,
            "is_right_half": is_right_half,
            "pair_indices": ctx.get("pair_indices"),     # (i, j)
            "centers": ctx.get("centers"),    
        }
        return res

        

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
    ap.add_argument("--pts_side_x", type=int, default=60, help="横方向のグリッド数（指定時は --pts_side_y とセット）")
    ap.add_argument("--pts_side_y", type=int, default=24, help="縦方向のグリッド数（指定時は --pts_side_x とセット）")
    ap.add_argument("--min_area", type=int, default=800)
    ap.add_argument("--iou_thr", type=float, default=0.8)  # reserved

    # zscore / axis min len
    ap.add_argument("--z_lower", type=float, default=-1.0)
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
        decoder_k_keep=1,
    )
    runner = SamBatchInfer_storage(cfg)
    tm.stop(); print(f"[TIMER] model_load: {tm.getTimeMilli():.2f} ms")

    # --- 単一画像モード ----------------------------------------------------
    if args.image:
        res = runner.run_on_image_path_storage(
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
            skip_selection=args.skip_selection,
        )
        overlay_path = res.get("overlay_path")
        if overlay_path is not None:
            print(f"[SAVE] guideline overlay: {overlay_path}")
        else:
            print("[WARN] guideline overlay was not generated")
        return

    # --- target がファイル or ディレクトリかで分岐 ------------------------
    if args.target:
        tgt = _Path(args.target)
        if tgt.is_file():
            res = runner.run_on_image_path_storage(
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
                skip_selection=args.skip_selection,
            )
            overlay_path = res.get("overlay_path")
            if overlay_path is not None:
                print(f"[SAVE] guideline overlay: {overlay_path}")
            else:
                print("[WARN] guideline overlay was not generated")
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

    # --- 複数画像バッチ処理 -----------------------------------------------
    for img_path in img_paths:
        print(f"==> {img_path.name}")
        res = runner.run_on_image_path_storage(
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
            skip_selection=args.skip_selection,
        )
        overlay_path = res.get("overlay_path")
        if overlay_path is not None:
            print(f"  [SAVE] {overlay_path}")
        else:
            print("  [WARN] guideline overlay was not generated")

    print("✔ Done:", out_dir)


if __name__ == "__main__":
    _main_cli()