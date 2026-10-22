from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
# ===== RealSense =====
import pyrealsense2 as rs

# ===== SAM: infer_for_storage の infer_masks を使う =====
from .infer_for_storage import SamConfig, SamBatchInfer_storage, StageSaveCfg
from .modules.overlay_io import _render_overlay_bgr, _save_points_and_overlay

# 点群モジュール
from .modules.pointcloud_utils import masked_depth_to_points, save_ply_ascii


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


def _enumerate_line_pixels(
    p0: tuple[int, int],
    p1: tuple[int, int],
    width: int,
    height: int
) -> np.ndarray:
    """
    画像座標上の2点 p0, p1 を結ぶ線分上のピクセル座標 (x, y) を列挙するヘルパ。
    戻り値: (N, 2) int32, 0 <= x < width, 0 <= y < height
    """
    x0, y0 = p0
    x1, y1 = p1

    n = int(np.hypot(x1 - x0, y1 - y0)) + 1
    if n <= 1:
        return np.array(
            [[np.clip(x0, 0, width - 1),
              np.clip(y0, 0, height - 1)]],
            dtype=np.int32
        )

    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    xs_i = np.clip(np.round(xs).astype(np.int32), 0, width - 1)
    ys_i = np.clip(np.round(ys).astype(np.int32), 0, height - 1)
    coords = np.stack([xs_i, ys_i], axis=1)   # (N, 2)
    coords = np.unique(coords, axis=0)
    return coords

def normalize_masks(masks, target_shape):
    fixed = []

    for i, m in enumerate(masks):
        m = np.array(m)

        # ❌ 1次元は即スキップ
        if m.ndim < 2:
            print(f"[SKIP] invalid mask: {m.shape}")
            continue

        # 3ch → 1ch
        if m.ndim > 2:
            m = m[:, :, 0]

        # サイズ合わせ
        if m.shape != target_shape:
            print(f"[RESIZE] {m.shape} -> {target_shape}")
            m = cv2.resize(
                m.astype(np.uint8),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # binary化
        m = (m > 0).astype(np.uint8)

        # 小さすぎるの除去（超重要）
        if np.sum(m) < 500:
            print("[SKIP] too small")
            continue

        fixed.append(m)

    print(f"[INFO] masks before={len(masks)} after={len(fixed)}")

    return fixed

def run_capture_and_pca(
    out_dir: str | Path = "captures",
    width: int = 1280,
    height: int = 720,
    fps: int = 6,
    sam_device: str = "gpu",
    encoder_path: str = "/home/book/pro_book_SAM3/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx",
    decoder_path: str = "/home/book/pro_book_SAM3/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx",
    interactive: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """
    RealSense 1ショット → SAM（入庫用 run_on_image_path_storage）→
    ガイドライン（p0_int/p1_int/sub_p0/sub_p1）のどちらかの直線を選択 →
    その直線上の点を Depth から3次元点群化 →
    ハンド挿入用の3D目標点を返すモジュール関数。

    Returns
    -------
    angle_rad : float
        画像上ガイドラインの傾き角（ラジアン）
    first_target_cam : (3,) ndarray
        カメラ座標系での第1目標点 [X, Y, Z] [m]
    final_target_cam : (3,) ndarray
        カメラ座標系での最終目標点 [X, Y, Z] [m]
    res : dict
        SAM 側の副産物（マスク等）
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    shot_dir = out_dir / ts
    shot_dir.mkdir(parents=True, exist_ok=True)

    # ===== RealSense 起動 =====
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    try:
        profile = pipe.start(cfg)
        align = rs.align(rs.stream.color)

        # ウォームアップ
        for _ in range(10):
            pipe.wait_for_frames()

        # ===== 1) RGB-D 1ショット取得 =====
        frames = pipe.wait_for_frames()
        align_frames = align.process(frames)
        depth_frame = align_frames.get_depth_frame()
        depth_frame = depth_filter_like_viewer(depth_frame)
        color_frame = align_frames.get_color_frame()

        # numpy 化（RealSense から生で取得：この時点では上下逆）
        color_np_raw = np.asanyarray(color_frame.get_data())      # BGR
        depth_np_raw_u16 = np.asanyarray(depth_frame.get_data())  # uint16 (Z16)

        # 画像サイズ（回転前と後で同じ）
        H, W = depth_np_raw_u16.shape[:2]

        # ★ SAM 用に RGB / Depth を 180度回転 ★
        color_np = cv2.rotate(color_np_raw, cv2.ROTATE_180)
        depth_np_u16 = cv2.rotate(depth_np_raw_u16, cv2.ROTATE_180)

        # 記録用に「正しい向き」の画像を保存
        rgb_png = shot_dir / "rgb.png"
        depth_npy = shot_dir / "depth.npy"
        depth_csv = shot_dir / "depth.csv"
        cv2.imwrite(str(rgb_png), color_np)
        np.save(depth_npy, depth_np_u16)
        depth_m_fake = depth_np_u16.astype(np.float32)
        np.savetxt(depth_csv, depth_m_fake, delimiter=",", fmt="%.6f")

        # ===== intrinsics / depth_scale を取得 =====
        # get_book_position.py と同じく color ストリームの intrinsics を使用
        color_prof = rs.video_stream_profile(
            profile.get_stream(rs.stream.color)
        )
        intr = color_prof.get_intrinsics()
        ds = profile.get_device().first_depth_sensor().get_depth_scale()
        depth_scale = float(ds)

        print(
            f"[intr(color)] fx={intr.fx:.3f} fy={intr.fy:.3f} "
            f"cx={intr.ppx:.3f} cy={intr.ppy:.3f} depth_scale={depth_scale}"
        )

    finally:
        try:
            pipe.stop()
        except Exception:
            pass

    # ========= ここから入庫用の処理 =========

    # --- 1) SAM (run_on_image_path_storage) を呼び出して
    #       p0_int, p1_int, sub_p0, sub_p1 を取得 ---
    sam_cfg = SamConfig(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        device=sam_device,
    )
    runner = SamBatchInfer_storage(sam_cfg)

    res = runner.run_on_image_path_storage(
        rgb_png,
        stage_save_dir=shot_dir,
        swap_lr_output=False,
        min_fold_deg=10.0,
        min_aspect=1.3,
        axis_angle_tol_deg=4.0,
        offset_width_factor=0.15,
        gap_len_factor=0.06,
        z_lower=-0.8,
        min_len_px=0.0,
        y_from_top_px=230,
        scan_radius_px=80,
        skip_selection=True,
    )
    # 🔥 ここ追加
    if "masks" in res:
        res["masks"] = normalize_masks(res["masks"], depth_np_u16.shape)

    # ===== デバッグ用：マスクoverlay保存 =====
    try:
        masks = res.get("masks", None)

        if masks is not None and len(masks) > 0:
            # normalize（さっき作ったやつ使う）
            masks = normalize_masks(masks, depth_np_u16.shape)

            # PIL Image に変換
            base_img = Image.fromarray(color_np)

            # overlay生成（あなたが修正した関数）
            ov_bgr = _render_overlay_bgr(base_img, masks, draw_ids=True)

            # 保存
            overlay_path = shot_dir / "debug_overlay.jpg"
            cv2.imwrite(str(overlay_path), ov_bgr)

            print(f"[SAVE] overlay: {overlay_path}")

        else:
            print("[WARN] masks が空")

    except Exception as e:
        print(f"[WARN] overlay保存失敗: {e}")

    pair_indices = res.get("pair_indices", None)

    if pair_indices is not None and masks is not None:
        i, j = pair_indices

        if 0 <= i < len(masks) and 0 <= j < len(masks):
            selected_masks = [masks[i], masks[j]]
            ov_bgr = _render_overlay_bgr(base_img, selected_masks, draw_ids=True)
            cv2.imwrite(str(shot_dir / "selected_overlay.jpg"), ov_bgr)
            print(f"[SAVE] selected overlay: {shot_dir / 'selected_overlay.jpg'}")
        else:
            print(f"[WARN] pair_indices out of range: {pair_indices}, len(masks)={len(masks)}")
    else:
        print("[WARN] pair_indices is None. selected_overlay.jpg will not be saved.")

    p0_int = res.get("line_p0")
    p1_int = res.get("line_p1")
    sub_p0 = res.get("sub_line_p0")
    sub_p1 = res.get("sub_line_p1")
    right_is_tilted = res.get("right_is_tilted", False)

    if p0_int is None or p1_int is None or sub_p0 is None or sub_p1 is None:
        raise RuntimeError("SAM からガイドライン情報が取得できませんでした。")

    # sub_p0 / sub_p1 は float 想定なので，画像座標に丸める
    sp0 = (int(round(float(sub_p0[0]))), int(round(float(sub_p0[1]))))
    sp1 = (int(round(float(sub_p1[0]))), int(round(float(sub_p1[1]))))

    # --- 2) x方向で p1_int[0] と sub_p1[0] を比較して，使用する直線を決定 ---
    if p1_int[0] < sp1[0]:
        use_p0 = sp0
        use_p1 = sp1
        print("[INFO] use sub-line: {use_p0} -> {use_p1}")
    else:
        use_p0 = p0_int
        use_p1 = p1_int
        print("[INFO] use main line: {use_p0} -> {use_p1}")

    # --- 3) 直線の傾き角度（ロボットのハンド回転角） ---
    dx = use_p1[0] - use_p0[0]
    dy = use_p1[1] - use_p0[1]
    angle_rad = float(np.arctan2(dy, dx))
    angle_deg = float(np.degrees(angle_rad))
    print(f"[INFO] line angle (image): {angle_deg:.2f} deg")

    # --- 4) 「右の本」の平均Depthを求める（Zのフォールバック用） ---
    masks = res.get("masks", None)
    pair = res.get("pair_indices", None)
    centers = res.get("centers", None)

    right_book_depth_m = None

    if masks is not None and pair is not None and centers is not None:
        masks_arr = np.asarray(masks, dtype=bool)
        centers_arr = np.asarray(centers, dtype=np.float32)

        i, j = pair
        if 0 <= i < masks_arr.shape[0] and 0 <= j < masks_arr.shape[0]:
            if centers_arr[i, 0] <= centers_arr[j, 0]:
                right_idx = j
            else:
                right_idx = i

            mask_right = masks_arr[right_idx]

            if mask_right.shape != depth_np_u16.shape:
                print("[WARN] right book mask shape mismatch; skip mask-based depth")
            else:
                valid = mask_right & (depth_np_u16 > 0)
                if np.any(valid):
                    depth_u16_mean = float(depth_np_u16[valid].mean())
                    right_book_depth_m = depth_u16_mean * depth_scale
                    print(
                        "[INFO] mean depth of right book: "
                        f"{right_book_depth_m:.4f} m (pixels={valid.sum()})"
                    )
                else:
                    print("[WARN] right book mask has no valid depth pixels")
    else:
        print("[WARN] masks/pair_indices/centers が無いので、平均Depthは使いません。")

    # --- 5) 選んだ直線上のピクセルを列挙 ---
    H_rot, W_rot = depth_np_u16.shape[:2]
    line_coords = _enumerate_line_pixels(use_p0, use_p1, W_rot, H_rot)
    if line_coords.size == 0:
        raise RuntimeError("選択した直線上のピクセルが取得できませんでした。")

    # --- 6) RealSense の intrinsics を使って 3D 点列に変換 ---
    pts_list = []

    for (x_rot, y_rot) in line_coords:
        # 回転後 → 回転前（180°回転の逆変換）
        x_raw = W_rot - 1 - x_rot
        y_raw = H_rot - 1 - y_rot

        d_u16 = int(depth_np_raw_u16[y_raw, x_raw])
        if d_u16 == 0:
            continue
        Z_m = float(d_u16) * depth_scale  # [m]

        X_cam, Y_cam, Z_cam = rs.rs2_deproject_pixel_to_point(
            intr, [float(x_raw), float(y_raw)], Z_m
        )
        pts_list.append([X_cam, Y_cam, Z_cam])

    if not pts_list:
        raise RuntimeError("直線上で有効なDepthが得られませんでした。")

    pts_line_cam = np.asarray(pts_list, dtype=np.float32)

    # --- 7) 直線状点群の中で一番「下」にある点（Y最大） ---
    idx_bottom = int(np.argmax(pts_line_cam[:, 1]))
    p_bottom = pts_line_cam[idx_bottom]
    print(
        f"[INFO] bottom point: X={p_bottom[0]:.4f}, "
        f"Y={p_bottom[1]:.4f}, Z={p_bottom[2]:.4f}"
    )

    # --- 8) 第一目標点：p_bottom から 8cm 離れた点 ---
    target_dist = 0.08  # [m]
    dists = np.linalg.norm(pts_line_cam - p_bottom, axis=1)
    idx_first = int(np.argmin(np.abs(dists - target_dist)))
    first_target_cam = pts_line_cam[idx_first]
    print(
        "[INFO] first target: "
        f"X={first_target_cam[0]:.4f}, "
        f"Y={first_target_cam[1]:.4f}, "
        f"Z={first_target_cam[2]:.4f}, "
        f"dist={dists[idx_first]:.4f} m"
    )

    # --- 9) 最終目標点の決定 ---
    if not right_is_tilted or right_book_depth_m is None:
        # 右の本が傾いていない or 平均Depthが取れなかった → 第一目標点 = 最終目標点
        final_target_cam = first_target_cam.copy()
        print("[INFO] right_is_tilted = False or depth None → final = first target")
    else:
        # 右の本が傾いている場合：Z だけ平均Depthに合わせる
        final_target_cam = np.array(
            [p_bottom[0], first_target_cam[1], right_book_depth_m],
            dtype=np.float32,
        )
        print(
            "[INFO] right_is_tilted = True → final target: "
            f"X={final_target_cam[0]:.4f}, "
            f"Y={final_target_cam[1]:.4f}, "
            f"Z={final_target_cam[2]:.4f}"
        )

    # --- 10) デバッグ用に直線状点群を PLY 保存 ---
    try:
        ply_path = shot_dir / "storage_line_points.ply"
        save_ply_ascii(ply_path, pts_line_cam.astype(np.float32))
        print(f"[SAVE] PLY: {ply_path}")
    except Exception as e:
        print(f"[WARN] PLY 保存に失敗しました: {e}")

    return angle_rad, first_target_cam, final_target_cam, res
import argparse

def depth_filter_like_viewer(raw_depth_frame: rs.depth_frame) -> rs.depth_frame:
    """
    RealSense Viewerに近いDepthフィルタ．
    RGBとDepthの対応を崩さないため，decimationは使わない．
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


def estimate_depth_mode(
    depth_u16: np.ndarray,
    depth_scale: float,
    z_min_m: float = 0.2,
    z_max_m: float = 1.2,
    bins: int = 100,
) -> float:
    """
    画像内で最も多いDepth帯を代表的な書籍面Depthとして推定する．
    """
    depth_m = depth_u16.astype(np.float32) * depth_scale
    valid = (depth_m > z_min_m) & (depth_m < z_max_m)

    if not np.any(valid):
        raise RuntimeError("有効なDepthがありません．")

    vals = depth_m[valid]
    hist, edges = np.histogram(vals, bins=bins, range=(z_min_m, z_max_m))
    idx = int(np.argmax(hist))
    depth_ref = 0.5 * (edges[idx] + edges[idx + 1])

    return float(depth_ref)


def make_far_space_mask(
    depth_u16: np.ndarray,
    depth_scale: float,
    depth_ref_m: float,
    delta_depth_m: float = 0.03,
) -> np.ndarray:
    """
    代表Depthより奥にある領域を収納スペース候補として二値化する．
    """
    depth_m = depth_u16.astype(np.float32) * depth_scale
    valid = depth_u16 > 0

    far = valid & (depth_m > depth_ref_m + delta_depth_m)

    mask = far.astype(np.uint8)

    # 軽いノイズ除去
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


def select_space_component(
    candidate_mask: np.ndarray,
    *,
    min_area_px: int = 500,
    horizontal_ratio_thr: float = 1.5,
) -> Optional[Dict[str, Any]]:
    """
    Depthから得た候補二値画像に対して連結成分解析を行い，
    横長領域を除外した上で，収納スペース候補を1つ選ぶ．

    horizontal_ratio_thr:
      w / h > horizontal_ratio_thr の成分を横長として除外する．
    """
    H, W = candidate_mask.shape[:2]
    mask_u8 = (candidate_mask > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )

    candidates = []

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        cx, cy = centroids[label]

        if area < min_area_px:
            continue

        if h <= 0 or w <= 0:
            continue

        # 横長領域だけ除外
        if (w / max(h, 1)) > horizontal_ratio_thr:
            continue

        component_mask = (labels == label).astype(np.uint8)

        # スコアはまず単純に面積と高さを重視
        # 三角形や縦長長方形は高さが出やすい
        score = float(area) + 2.0 * float(h)

        candidates.append({
            "label": int(label),
            "bbox": (int(x), int(y), int(w), int(h)),
            "center": (float(cx), float(cy)),
            "area": int(area),
            "width": int(w),
            "height": int(h),
            "aspect_w_h": float(w / max(h, 1)),
            "score": score,
            "mask": component_mask,
        })

    if not candidates:
        return None

    candidates.sort(key=lambda d: d["score"], reverse=True)
    return candidates[0]


def extract_left_boundary_points(component_mask: np.ndarray) -> np.ndarray:
    """
    収納スペース候補マスクの左側境界点を抽出する．

    各y行について，候補画素のうち最も左のxを取る．
    戻り値は (N, 2) の画像座標点列 [x, y]．
    """
    mask = component_mask.astype(bool)
    H, W = mask.shape[:2]

    pts = []

    ys = np.where(mask.any(axis=1))[0]

    for y in ys:
        xs = np.where(mask[y])[0]
        if xs.size == 0:
            continue

        x_left = int(xs.min())
        pts.append([x_left, int(y)])

    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    return np.asarray(pts, dtype=np.float32)


def fit_line_pca_2d(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D点列にPCAで直線フィットする．
    戻り値:
      p0, p1 : 直線端点 [x, y]
    """
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)

    if pts.shape[0] < 2:
        raise RuntimeError("直線フィットに必要な点が不足しています．")

    mean = pts.mean(axis=0)
    X = pts - mean

    _, _, vh = np.linalg.svd(X, full_matrices=False)
    axis = vh[0].astype(np.float32)

    # 画像下方向へ向くように揃える
    if axis[1] < 0:
        axis = -axis

    proj = X @ axis
    p0 = mean + axis * float(proj.min())
    p1 = mean + axis * float(proj.max())

    # p0を上側，p1を下側にそろえる
    if p0[1] > p1[1]:
        p0, p1 = p1, p0

    return p0.astype(np.float32), p1.astype(np.float32)


def enumerate_line_pixels(
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    width: int,
    height: int,
) -> np.ndarray:
    """
    画像座標上の2点を結ぶ線分上のピクセル座標を列挙する．
    """
    x0, y0 = p0
    x1, y1 = p1

    n = int(np.hypot(x1 - x0, y1 - y0)) + 1
    n = max(n, 1)

    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)

    xs_i = np.clip(np.round(xs).astype(np.int32), 0, width - 1)
    ys_i = np.clip(np.round(ys).astype(np.int32), 0, height - 1)

    coords = np.stack([xs_i, ys_i], axis=1)
    coords = np.unique(coords, axis=0)

    return coords


def deproject_line_to_3d(
    line_coords: np.ndarray,
    depth_u16: np.ndarray,
    intr: rs.intrinsics,
    depth_scale: float,
    *,
    z_min_m: float = 0.1,
    z_max_m: float = 1.5,
) -> np.ndarray:
    """
    画像上の線分座標列をDepthからカメラ座標系3D点列に変換する．
    """
    pts = []

    for x, y in line_coords:
        d = int(depth_u16[y, x])
        if d == 0:
            continue

        z = float(d) * depth_scale
        if z < z_min_m or z > z_max_m:
            continue

        X, Y, Z = rs.rs2_deproject_pixel_to_point(
            intr,
            [float(x), float(y)],
            z,
        )

        pts.append([X, Y, Z])

    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    return np.asarray(pts, dtype=np.float32)


def select_first_target_from_line_points(
    pts_line_cam: np.ndarray,
    target_dist_m: float = 0.08,
) -> np.ndarray:
    """
    3D線分点列からリーチング用第一目標点を選ぶ．
    従来コードに合わせて，一番下の点から target_dist_m 離れた点を選ぶ．
    """
    pts = np.asarray(pts_line_cam, dtype=np.float32).reshape(-1, 3)

    if pts.shape[0] == 0:
        raise RuntimeError("3D線分点群が空です．")

    # RealSense座標系ではYが画像下向きなので，Y最大を下側とする
    idx_bottom = int(np.argmax(pts[:, 1]))
    p_bottom = pts[idx_bottom]

    dists = np.linalg.norm(pts - p_bottom, axis=1)
    idx_first = int(np.argmin(np.abs(dists - target_dist_m)))

    return pts[idx_first].astype(np.float32)


def save_debug_overlay(
    rgb_bgr: np.ndarray,
    candidate_mask: np.ndarray,
    selected_mask: np.ndarray,
    line_p0: Tuple[int, int],
    line_p1: Tuple[int, int],
    first_target_px: Optional[Tuple[int, int]],
    save_path: Path,
) -> None:
    """
    検出結果をRGBに重ねて保存する．
    """
    vis = rgb_bgr.copy()

    # 候補全体を薄く表示
    cand = candidate_mask.astype(bool)
    vis[cand] = (0.5 * vis[cand] + 0.5 * np.array([0, 255, 255])).astype(np.uint8)

    # 選択領域を強調
    sel = selected_mask.astype(bool)
    vis[sel] = (0.4 * vis[sel] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)

    cv2.line(vis, line_p0, line_p1, (255, 0, 255), 4)
    cv2.circle(vis, line_p0, 5, (0, 255, 255), -1)
    cv2.circle(vis, line_p1, 5, (255, 255, 255), -1)

    if first_target_px is not None:
        cv2.circle(vis, first_target_px, 8, (0, 255, 0), -1)

    cv2.imwrite(str(save_path), vis)
    print(f"[SAVE] depth-space overlay: {save_path}")

def remove_horizontal_structures(mask: np.ndarray) -> np.ndarray:
    """
    横長の不要領域を先に除去する．
    """
    mask_u8 = (mask > 0).astype(np.uint8)

    # 横方向に長い白領域を抽出
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 7))
    horizontal = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, horizontal_kernel)

    # 元マスクから横長領域を引く
    cleaned = cv2.subtract(mask_u8, horizontal)

    # 縦方向候補を少しつなぎ直す
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, vertical_kernel)

    return cleaned



def run_capture_and_pca_depth_space(
    out_dir: str | Path = "captures_depth_space",
    width: int = 1280,
    height: int = 720,
    fps: int = 6,
    delta_depth_m: float = 0.03,
    min_area_px: int = 500,
    horizontal_ratio_thr: float = 1.5,
    target_dist_m: float = 0.08,
    rotate_180: bool = False,
):
    """
    Depth画像のみから収納スペースを推定し，
    従来の run_capture_and_pca と同じ形式で返す．

    Returns
    -------
    angle_rad : float
        収納スペース左側境界の画像上傾き [rad]
    first_target_cam : (3,) ndarray
        カメラ座標系の第一目標点 [m]
    final_target_cam : (3,) ndarray
        カメラ座標系の最終目標点 [m]
    res : dict
        デバッグ情報
    """
    out_dir = Path(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    shot_dir = out_dir / ts
    shot_dir.mkdir(parents=True, exist_ok=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    try:
        profile = pipe.start(cfg)
        align = rs.align(rs.stream.color)

        for _ in range(30):
            pipe.wait_for_frames()

        frames = pipe.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense frame取得に失敗しました．")

        depth_frame = depth_filter_like_viewer(depth_frame)

        rgb_bgr = np.asanyarray(color_frame.get_data())
        depth_u16 = np.asanyarray(depth_frame.get_data())

        color_prof = rs.video_stream_profile(
            profile.get_stream(rs.stream.color)
        )
        intr = color_prof.get_intrinsics()
        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

    finally:
        try:
            pipe.stop()
        except Exception:
            pass

    if rotate_180:
        rgb_bgr = cv2.rotate(rgb_bgr, cv2.ROTATE_180)
        depth_u16 = cv2.rotate(depth_u16, cv2.ROTATE_180)

        # 厳密に3D化するならintrinsicsも回転後に合わせる必要あり．
        # 実運用では rotate_180=False 推奨．
        print("[WARN] rotate_180=True の場合，3D復元にはintrinsics補正が必要です．")

    H, W = depth_u16.shape[:2]

    # 保存
    cv2.imwrite(str(shot_dir / "rgb.png"), rgb_bgr)
    np.save(shot_dir / "depth_u16.npy", depth_u16)

    # 1. 代表Depth推定
    depth_ref_m = estimate_depth_mode(
        depth_u16,
        depth_scale,
        z_min_m=0.2,
        z_max_m=1.2,
        bins=100,
    )
    print(f"[INFO] depth_ref_m = {depth_ref_m:.4f} m")

    # 2. 奥に抜けた領域抽出
    candidate_mask = make_far_space_mask(
        depth_u16,
        depth_scale,
        depth_ref_m,
        delta_depth_m=delta_depth_m,
    )

    candidate_mask = remove_horizontal_structures(candidate_mask)

    cv2.imwrite(str(shot_dir / "candidate_far_mask.png"), candidate_mask * 255)

    # 3. 横長領域を除外して収納スペース候補を選ぶ
    selected = select_space_component(
        candidate_mask,
        min_area_px=min_area_px,
        horizontal_ratio_thr=horizontal_ratio_thr,
    )

    if selected is None:
        raise RuntimeError("収納スペース候補が見つかりませんでした．")

    selected_mask = selected["mask"]
    bbox = selected["bbox"]
    print(f"[INFO] selected bbox={bbox}, area={selected['area']}, aspect_w_h={selected['aspect_w_h']:.3f}")

    # 4. 左側境界を抽出
    left_boundary_pts = extract_left_boundary_points(selected_mask)

    if left_boundary_pts.shape[0] < 2:
        raise RuntimeError("左側境界点が不足しています．")

    # 5. 左側境界に直線フィット
    p0, p1 = fit_line_pca_2d(left_boundary_pts)

    p0_int = (int(round(p0[0])), int(round(p0[1])))
    p1_int = (int(round(p1[0])), int(round(p1[1])))

    # 6. 画像上角度
    dx = float(p1_int[0] - p0_int[0])
    dy = float(p1_int[1] - p0_int[1])
    angle_rad = float(np.arctan2(dy, dx))
    angle_deg = float(np.degrees(angle_rad))
    print(f"[INFO] left boundary angle = {angle_deg:.2f} deg")

    # 7. ガイドライン上を3D化
    line_coords = enumerate_line_pixels(
        p0_int,
        p1_int,
        width=W,
        height=H,
    )

    pts_line_cam = deproject_line_to_3d(
        line_coords,
        depth_u16,
        intr,
        depth_scale,
        z_min_m=0.1,
        z_max_m=1.5,
    )

    if pts_line_cam.shape[0] == 0:
        raise RuntimeError("ガイドライン上で有効なDepthが得られませんでした．")

    # 8. first_target_cam / final_target_cam
    first_target_cam = select_first_target_from_line_points(
        pts_line_cam,
        target_dist_m=target_dist_m,
    )

    final_target_cam = first_target_cam.copy()

    # first_targetに近い画像座標をデバッグ用に探す
    first_target_px = None
    if pts_line_cam.shape[0] == len(line_coords):
        d3 = np.linalg.norm(pts_line_cam - first_target_cam.reshape(1, 3), axis=1)
        idx = int(np.argmin(d3))
        first_target_px = (int(line_coords[idx][0]), int(line_coords[idx][1]))

    # 9. overlay保存
    overlay_path = shot_dir / "depth_space_overlay.png"
    save_debug_overlay(
        rgb_bgr,
        candidate_mask,
        selected_mask,
        p0_int,
        p1_int,
        first_target_px,
        overlay_path,
    )

    # 10. res作成
    res = {
        "line_p0": p0_int,
        "line_p1": p1_int,
        "sub_line_p0": p0_int,
        "sub_line_p1": p1_int,
        "right_is_tilted": False,
        "is_right_half": False,
        "pair_indices": None,
        "centers": None,
        "space_bbox": bbox,
        "space_center_px": selected["center"],
        "space_area": selected["area"],
        "space_aspect_w_h": selected["aspect_w_h"],
        "depth_ref_m": depth_ref_m,
        "delta_depth_m": delta_depth_m,
        "overlay_path": str(overlay_path),
        "shot_dir": str(shot_dir),
    }

    print("[INFO] first_target_cam =", first_target_cam)
    print("[INFO] final_target_cam  =", final_target_cam)
    print("[INFO] files saved under:", shot_dir)

    return angle_rad, first_target_cam, final_target_cam, res

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

    angle_rad, first_target_cam, final_target_cam, res= run_capture_and_pca(
        out_dir=args.out,
        width=args.w,
        height=args.h,
        fps=args.fps,
        sam_device=args.sam_device,
        encoder_path=args.encoder,
        decoder_path=args.decoder,
    )

    

if __name__ == "__main__":
    main()
