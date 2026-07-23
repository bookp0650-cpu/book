#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import json
import time
import re

import cv2
import numpy as np
import open3d as o3d


# ============================================================
# Camera params
# ============================================================
def load_camera_params(camera_path: Path):
    with open(camera_path, "r", encoding="utf-8") as f:
        cam = json.load(f)

    fx = cam.get("fx", None)
    fy = cam.get("fy", None)
    cx = cam.get("cx", cam.get("ppx", None))
    cy = cam.get("cy", cam.get("ppy", None))
    depth_scale = cam.get("depth_scale", 1.0)

    if fx is None and "intrinsics" in cam:
        intr = cam["intrinsics"]
        fx = intr.get("fx", None)
        fy = intr.get("fy", None)
        cx = intr.get("cx", intr.get("ppx", None))
        cy = intr.get("cy", intr.get("ppy", None))
        depth_scale = intr.get("depth_scale", depth_scale)

    if fx is None or fy is None or cx is None or cy is None:
        raise ValueError(f"fx, fy, cx/ppx, cy/ppy を取得できませんでした: {cam}")

    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "depth_scale": float(depth_scale),
        "width": cam.get("width", None),
        "height": cam.get("height", None),
        "raw": cam,
    }


# ============================================================
# Basic loaders
# ============================================================
def load_depth(depth_path: Path):
    if not depth_path.exists():
        raise FileNotFoundError(f"depthファイルが見つかりません: {depth_path}")

    depth = np.load(str(depth_path))

    print("===== depth =====")
    print("path :", depth_path)
    print("shape:", depth.shape)
    print("dtype:", depth.dtype)
    print("min/max:", depth.min(), depth.max())
    print("=================")

    return depth


def load_rgb(rgb_path: Path):
    if not rgb_path.exists():
        print(f"⚠ RGB画像がありません: {rgb_path}")
        return None

    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        print(f"⚠ RGB画像を読み込めませんでした: {rgb_path}")
        return None

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    print("===== rgb =====")
    print("path :", rgb_path)
    print("shape:", rgb.shape)
    print("dtype:", rgb.dtype)
    print("===============")

    return rgb


def load_ply_pointcloud(ply_path: Path):
    if not ply_path.exists():
        raise FileNotFoundError(f"PLYファイルが見つかりません: {ply_path}")

    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)

    print("===== loaded ply point cloud =====")
    print("path :", ply_path)
    print("num points:", len(points))

    if len(points) > 0:
        print("x min/max:", points[:, 0].min(), points[:, 0].max())
        print("y min/max:", points[:, 1].min(), points[:, 1].max())
        print("z min/max:", points[:, 2].min(), points[:, 2].max())

    print("has colors:", pcd.has_colors())
    print("==================================")

    if len(points) == 0:
        raise ValueError(f"PLY点群が空です: {ply_path}")

    colors = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors).astype(np.float64)

    return points.astype(np.float64), colors


# ============================================================
# NPY pixel points / contours
# ============================================================
def load_pixel_points_or_contours(npy_path: Path):
    if not npy_path.exists():
        raise FileNotFoundError(f"npyファイルが見つかりません: {npy_path}")

    arr = np.load(str(npy_path), allow_pickle=True)

    print("===== loaded npy =====")
    print("path :", npy_path)
    print("shape:", getattr(arr, "shape", None))
    print("dtype:", getattr(arr, "dtype", None))
    print("======================")

    contours = []

    if isinstance(arr, np.ndarray) and arr.dtype == object:
        for i, elem in enumerate(arr):
            if elem is None:
                continue

            elem = np.asarray(elem)
            if elem.size == 0:
                continue

            if elem.ndim >= 3:
                elem = elem.reshape(-1, elem.shape[-1])

            if elem.ndim != 2 or elem.shape[1] < 2:
                print(f"skip object[{i}], invalid shape: {elem.shape}")
                continue

            uv = elem[:, :2].astype(np.int32)
            contours.append(uv)

    else:
        arr = np.asarray(arr)

        if arr.ndim >= 3:
            arr = arr.reshape(-1, arr.shape[-1])

        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"2D座標として扱えないshapeです: {arr.shape}")

        uv = arr[:, :2].astype(np.int32)
        contours.append(uv)

    if len(contours) == 0:
        raise ValueError("有効な2D座標・輪郭点が見つかりませんでした．")

    all_uv = np.vstack(contours)
    all_uv = np.unique(all_uv, axis=0)

    print("===== pixel points =====")
    print("num contours:", len(contours))
    print("uv shape:", all_uv.shape)
    print("u min/max:", all_uv[:, 0].min(), all_uv[:, 0].max())
    print("v min/max:", all_uv[:, 1].min(), all_uv[:, 1].max())
    print("========================")

    return contours


def contours_to_uv(contours, image_shape, fill_contours=True):
    H, W = image_shape[:2]
    valid_contours = []

    for c in contours:
        c = np.asarray(c, dtype=np.int32)
        if c.ndim != 2 or c.shape[1] < 2:
            continue

        c[:, 0] = np.clip(c[:, 0], 0, W - 1)
        c[:, 1] = np.clip(c[:, 1], 0, H - 1)

        if len(c) >= 2:
            valid_contours.append(c[:, :2])

    if len(valid_contours) == 0:
        raise ValueError("画像範囲内の有効なcontourがありません．")

    if fill_contours:
        mask = np.zeros((H, W), dtype=np.uint8)
        cv_contours = [c.reshape(-1, 1, 2) for c in valid_contours]

        cv2.drawContours(
            image=mask,
            contours=cv_contours,
            contourIdx=-1,
            color=255,
            thickness=-1,
        )

        v, u = np.where(mask > 0)
        uv = np.stack([u, v], axis=1).astype(np.int32)

        print("===== mask from contours =====")
        print("fill_contours: True")
        print("mask pixels:", len(uv))
        print("==============================")

    else:
        uv = np.vstack(valid_contours).astype(np.int32)
        uv = np.unique(uv, axis=0)

        print("===== contour points only =====")
        print("fill_contours: False")
        print("contour pixels:", len(uv))
        print("===============================")

    return uv


# ============================================================
# Projection / unprojection
# ============================================================
def project_points_to_uv(points: np.ndarray, cam: dict):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    z = pts[:, 2]
    valid = np.isfinite(z) & (z > 1e-9)

    u = np.full((pts.shape[0],), -1, dtype=np.int32)
    v = np.full((pts.shape[0],), -1, dtype=np.int32)

    u[valid] = np.round(
        float(cam["fx"]) * pts[valid, 0] / z[valid] + float(cam["cx"])
    ).astype(np.int32)
    v[valid] = np.round(
        float(cam["fy"]) * pts[valid, 1] / z[valid] + float(cam["cy"])
    ).astype(np.int32)

    return np.stack([u, v], axis=1).astype(np.int32), valid


def unproject_uv_to_xyz(uv: np.ndarray, z_m: float, cam: dict):
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    z = float(z_m)

    x = (uv[:, 0] - float(cam["cx"])) * z / float(cam["fx"])
    y = (uv[:, 1] - float(cam["cy"])) * z / float(cam["fy"])
    zz = np.full_like(x, z)

    return np.stack([x, y, zz], axis=1).astype(np.float64)


def uv_depth_to_pointcloud(
    uv: np.ndarray,
    depth: np.ndarray,
    rgb: np.ndarray | None,
    cam: dict,
    min_depth_m: float,
    max_depth_m: float,
):
    H, W = depth.shape[:2]

    uv = np.asarray(uv, dtype=np.int32).reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]

    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[in_img]
    v = v[in_img]

    if len(u) == 0:
        raise ValueError("uvがすべてdepth画像の範囲外です．")

    z_raw = depth[v, u].astype(np.float32)
    z = z_raw * float(cam["depth_scale"])

    valid_z = np.isfinite(z)
    valid_z &= z > float(min_depth_m)
    valid_z &= z < float(max_depth_m)

    u = u[valid_z]
    v = v[valid_z]
    z = z[valid_z]

    if len(z) == 0:
        raise ValueError("有効なdepth点がありません．")

    x = (u.astype(np.float32) - float(cam["cx"])) * z / float(cam["fx"])
    y = (v.astype(np.float32) - float(cam["cy"])) * z / float(cam["fy"])

    points = np.stack([x, y, z], axis=1).astype(np.float64)
    valid_uv = np.stack([u, v], axis=1).astype(np.int32)

    colors = None
    if rgb is not None:
        if rgb.shape[:2] != depth.shape[:2]:
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_NEAREST)
        colors = rgb[v, u].astype(np.float64) / 255.0
        colors = np.clip(colors, 0.0, 1.0)

    print("===== 3D point cloud from uv/depth =====")
    print("points shape:", points.shape)
    print("x min/max:", points[:, 0].min(), points[:, 0].max())
    print("y min/max:", points[:, 1].min(), points[:, 1].max())
    print("z min/max:", points[:, 2].min(), points[:, 2].max())
    print("========================================")

    return points, colors, valid_uv


# ============================================================
# OCR axis extraction
# ============================================================
def iter_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from iter_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_dicts(v)


def load_json_if_exists(path: Path):
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠ JSON読み込み失敗: {path}: {e}")
        return None


def polygon_pca_center_axis(poly):
    if poly is None:
        return None, None

    arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] < 2:
        return None, None

    center = arr.mean(axis=0)
    centered = arr - center

    try:
        cov = np.cov(centered.T)
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, int(np.argmax(vals))]
    except Exception:
        return None, None

    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return None, None

    axis = axis / n
    return center.astype(np.float64), axis.astype(np.float64)


def normalize_axis(axis):
    axis = np.asarray(axis, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return None
    return axis / n


def find_ocr_axis_from_logs(base: Path, mask_id: str):
    candidates = [
        base / f"{mask_id}_processing_log.json",
        base / f"{mask_id}_depth_prefilter_log.json",
        base / "debug_ocr_band" / f"{mask_id}_log.json",
        base / "debug_ocr_band" / f"{mask_id}_offline_log.json",
    ]

    for p in sorted(base.glob(f"**/{mask_id}*.json")):
        if p not in candidates:
            candidates.append(p)

    loaded = []
    for p in candidates:
        obj = load_json_if_exists(p)
        if obj is not None:
            loaded.append((p, obj))

    # 1. selected_ocr_polygon.poly からPCA
    for path, obj in loaded:
        for d in iter_dicts(obj):
            selected = d.get("selected_ocr_polygon")
            if isinstance(selected, dict):
                poly = selected.get("poly", None)
                center, axis = polygon_pca_center_axis(poly)
                if center is not None and axis is not None:
                    print("===== OCR axis =====")
                    print("source:", path)
                    print("method: selected_ocr_polygon.poly PCA")
                    print("center:", center.tolist())
                    print("axis  :", axis.tolist())
                    print("====================")
                    return {
                        "center": center,
                        "axis": axis,
                        "source": str(path),
                        "method": "selected_ocr_polygon.poly PCA",
                        "selected_ocr_polygon": selected,
                    }

    # 2. spine_completion_after_depth_prefilter
    for path, obj in loaded:
        for d in iter_dicts(obj):
            sc = d.get("spine_completion_after_depth_prefilter")
            if isinstance(sc, dict):
                if sc.get("axis") is not None and sc.get("center_xy") is not None:
                    axis = normalize_axis(sc["axis"])
                    center = np.asarray(sc["center_xy"], dtype=np.float64).reshape(2)
                    if axis is not None:
                        print("===== OCR axis =====")
                        print("source:", path)
                        print("method: spine_completion_after_depth_prefilter")
                        print("center:", center.tolist())
                        print("axis  :", axis.tolist())
                        print("====================")
                        return {
                            "center": center,
                            "axis": axis,
                            "source": str(path),
                            "method": "spine_completion_after_depth_prefilter",
                        }

    # 3. generic axis / center
    for path, obj in loaded:
        for d in iter_dicts(obj):
            if d.get("axis") is not None:
                center_value = None
                for key in ("center", "ocr_center", "center_xy"):
                    if d.get(key) is not None:
                        center_value = d.get(key)
                        break

                if center_value is None:
                    continue

                axis = normalize_axis(d["axis"])
                center = np.asarray(center_value, dtype=np.float64).reshape(2)

                if axis is not None:
                    print("===== OCR axis =====")
                    print("source:", path)
                    print("method: generic axis/center")
                    print("center:", center.tolist())
                    print("axis  :", axis.tolist())
                    print("====================")
                    return {
                        "center": center,
                        "axis": axis,
                        "source": str(path),
                        "method": "generic axis/center",
                    }

    print("⚠ OCR軸がログから取得できませんでした．点群PCA軸へfallbackします．")
    return None


def fallback_axis_from_uv(uv: np.ndarray):
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    center = uv.mean(axis=0)
    centered = uv - center

    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)

    return {
        "center": center.astype(np.float64),
        "axis": axis.astype(np.float64),
        "source": "fallback_uv_pca",
        "method": "fallback_uv_pca",
    }


# ============================================================
# Longest column
# ============================================================
def compute_longest_column_indices(
    uv: np.ndarray,
    axis_info: dict,
    *,
    t_bin_size_px: float = 4.0,
    min_points_per_bin: int = 10,
    s_percentiles: tuple[float, float] = (2.0, 98.0),
):
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)

    center = np.asarray(axis_info["center"], dtype=np.float64).reshape(2)
    axis = normalize_axis(axis_info["axis"])
    if axis is None:
        raise ValueError("axis_info の axis が不正です．")

    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)

    rel = uv - center
    s = rel @ axis
    t = rel @ normal

    t_bin_size_px = max(float(t_bin_size_px), 1.0)

    t_min = float(np.min(t))
    t_max = float(np.max(t))
    n_bins = int(np.floor((t_max - t_min) / t_bin_size_px)) + 1
    n_bins = max(n_bins, 1)

    bins = np.floor((t - t_min) / t_bin_size_px).astype(np.int32)
    bins = np.clip(bins, 0, n_bins - 1)

    p0, p1 = float(s_percentiles[0]), float(s_percentiles[1])
    p0 = float(np.clip(p0, 0.0, 49.0))
    p1 = float(np.clip(p1, 51.0, 100.0))

    records = []
    best_bin = None
    best_length = -1.0

    for b in range(n_bins):
        idx = np.where(bins == b)[0]
        count = int(idx.size)

        if count < int(min_points_per_bin):
            length = 0.0
            s0 = None
            s1 = None
        else:
            sv = s[idx]
            s0, s1 = np.percentile(sv, [p0, p1])
            length = float(max(0.0, s1 - s0))

        records.append({
            "t_bin": int(b),
            "point_count": count,
            "length_px": float(length),
            "s_min": None if s0 is None else float(s0),
            "s_max": None if s1 is None else float(s1),
            "t_min": float(t_min + b * t_bin_size_px),
            "t_max": float(t_min + (b + 1) * t_bin_size_px),
        })

        if length > best_length:
            best_length = float(length)
            best_bin = int(b)

    if best_bin is None:
        return np.zeros((uv.shape[0],), dtype=bool), {
            "used": False,
            "reason": "failed to find longest column",
            "records": records,
        }

    longest_mask = bins == int(best_bin)

    info = {
        "used": True,
        "reason": "ok",
        "best_bin": int(best_bin),
        "best_length_px": float(best_length),
        "t_bin_size_px": float(t_bin_size_px),
        "min_points_per_bin": int(min_points_per_bin),
        "s_percentiles": [float(p0), float(p1)],
        "longest_point_count": int(np.count_nonzero(longest_mask)),
        "records": records,
        "axis": axis.tolist(),
        "normal": normal.tolist(),
        "center": center.tolist(),
    }

    print("===== longest column =====")
    print("best_bin:", best_bin)
    print("best_length_px:", best_length)
    print("point_count:", int(np.count_nonzero(longest_mask)))
    print("==========================")

    return longest_mask, info


# ============================================================
# Removed points: final_uv dilation
# ============================================================
def compute_removed_points_by_uv(
    before_points: np.ndarray,
    before_uv: np.ndarray,
    final_uv: np.ndarray,
    image_shape: tuple[int, int],
    *,
    tolerance_px: int = 3,
):
    """
    before点群のうち，final点群に含まれないuvを削除点として抽出する．

    ただし，final_plyを画像平面へ再投影すると丸め誤差が出るため，
    uv完全一致ではなく，final_uvをtolerance_pxだけ膨張して近傍一致で比較する．
    """
    before_points = np.asarray(before_points, dtype=np.float64).reshape(-1, 3)
    before_uv = np.asarray(before_uv, dtype=np.int32).reshape(-1, 2)
    final_uv = np.asarray(final_uv, dtype=np.int32).reshape(-1, 2)

    H, W = image_shape[:2]

    final_mask = np.zeros((H, W), dtype=np.uint8)

    fu = final_uv[:, 0]
    fv = final_uv[:, 1]

    valid_final = (fu >= 0) & (fu < W) & (fv >= 0) & (fv < H)
    fu = fu[valid_final]
    fv = fv[valid_final]

    final_mask[fv, fu] = 255

    tolerance_px = int(max(0, tolerance_px))
    if tolerance_px > 0:
        ksize = 2 * tolerance_px + 1
        kernel = np.ones((ksize, ksize), dtype=np.uint8)
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)

    bu = before_uv[:, 0]
    bv = before_uv[:, 1]

    valid_before = (bu >= 0) & (bu < W) & (bv >= 0) & (bv < H)

    kept_by_near_final = np.zeros((before_uv.shape[0],), dtype=bool)
    kept_by_near_final[valid_before] = final_mask[
        bv[valid_before],
        bu[valid_before],
    ] > 0

    removed_mask = ~kept_by_near_final

    removed_points = before_points[removed_mask]
    removed_uv = before_uv[removed_mask]

    print("===== removed points =====")
    print("method: uv dilated final mask")
    print("tolerance_px:", tolerance_px)
    print("before points:", len(before_points))
    print("final uv pixels after dilation:", int(np.count_nonzero(final_mask)))
    print("removed points:", len(removed_points))
    print("removed ratio:", float(len(removed_points) / max(len(before_points), 1)))
    print("==========================")

    return removed_points, removed_uv, removed_mask


# ============================================================
# Coloring and geometries
# ============================================================
def make_open3d_pointcloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64).reshape(-1, 3)
    )

    colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    colors = np.clip(colors, 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def make_colors(
    n: int,
    base_rgb: np.ndarray | None,
    *,
    use_rgb: bool,
    uniform_color=(1.0, 0.0, 0.0),
    longest_mask: np.ndarray | None = None,
    longest_color=(0.0, 1.0, 0.0),
):
    if use_rgb and base_rgb is not None and len(base_rgb) == n:
        colors = np.asarray(base_rgb, dtype=np.float64).copy()
    else:
        colors = np.tile(
            np.asarray(uniform_color, dtype=np.float64).reshape(1, 3),
            (n, 1),
        )

    if longest_mask is not None:
        longest_mask = np.asarray(longest_mask, dtype=bool).reshape(-1)
        if len(longest_mask) == n:
            colors[longest_mask] = np.asarray(longest_color, dtype=np.float64)

    return np.clip(colors, 0.0, 1.0)


def make_cylinder_between_points(
    p0,
    p1,
    *,
    radius=0.004,
    color=(0.0, 0.0, 0.0),
    resolution=32,
):
    """
    2点p0,p1を結ぶ円柱を作る．
    OCR文字領域の長手方向を太く表示するために使う．
    """
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)

    vec = p1 - p0
    length = float(np.linalg.norm(vec))

    if length < 1e-9:
        return None

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=float(radius),
        height=length,
        resolution=int(resolution),
    )

    # Open3Dの円柱は初期状態でz軸方向に伸びているため，
    # z軸をvec方向へ回転させる．
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    direction = vec / length

    rot_axis = np.cross(z_axis, direction)
    rot_axis_norm = float(np.linalg.norm(rot_axis))

    if rot_axis_norm < 1e-9:
        if np.dot(z_axis, direction) < 0:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                np.array([np.pi, 0.0, 0.0], dtype=np.float64)
            )
        else:
            R = np.eye(3)
    else:
        rot_axis = rot_axis / rot_axis_norm
        angle = float(np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0)))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)

    cylinder.rotate(R, center=np.zeros(3))
    cylinder.translate((p0 + p1) / 2.0)
    cylinder.paint_uniform_color(color)

    return cylinder


def make_line_set_between_points(
    p0,
    p1,
    *,
    color=(1.0, 0.0, 1.0),
):
    """
    2点p0,p1を結ぶ細い線を作る．
    太い円柱と違う色で同じ方向を示すために使う．
    """
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)

    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(np.stack([p0, p1], axis=0))
    line.lines = o3d.utility.Vector2iVector(np.asarray([[0, 1]], dtype=np.int32))
    line.colors = o3d.utility.Vector3dVector(
        np.asarray([color], dtype=np.float64)
    )

    return line


def make_axis_direction_geometries(
    points: np.ndarray,
    uv: np.ndarray,
    axis_info: dict,
    cam: dict,
    *,
    length_percentiles=(1.0, 99.0),
    cylinder_radius_m=0.004,
    cylinder_color=(0.0, 0.0, 0.0),
    thin_line_color=(1.0, 0.0, 1.0),
    thin_line_z_offset_m=0.006,
):
    """
    OCR文字領域第一主成分方向を，
    太い円柱 + 異なる色の細線として3D表示する．

    太い円柱:
      黒色，実際の方向を太く表示する．

    細線:
      紫色，円柱に埋もれないように少しカメラ側へずらして表示する．
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)

    if len(points) == 0 or len(uv) == 0:
        return []

    center = np.asarray(axis_info["center"], dtype=np.float64).reshape(2)
    axis = normalize_axis(axis_info["axis"])
    if axis is None:
        return []

    rel = uv - center
    s = rel @ axis

    p0, p1 = length_percentiles
    s0, s1 = np.percentile(s, [float(p0), float(p1)])

    uv0 = center + axis * float(s0)
    uv1 = center + axis * float(s1)

    z_med = float(np.median(points[:, 2]))
    xyz = unproject_uv_to_xyz(np.stack([uv0, uv1], axis=0), z_med, cam)

    geoms = []

    cylinder = make_cylinder_between_points(
        xyz[0],
        xyz[1],
        radius=float(cylinder_radius_m),
        color=cylinder_color,
    )
    if cylinder is not None:
        geoms.append(cylinder)

    # 細線は円柱内部に隠れないように，少しカメラ側へずらす．
    # RealSenseのカメラ座標ではZがカメラ前方なので，Zを小さくするとカメラ側へ寄る．
    xyz_line = xyz.copy()
    xyz_line[:, 2] = np.maximum(1e-6, xyz_line[:, 2] - float(thin_line_z_offset_m))

    thin_line = make_line_set_between_points(
        xyz_line[0],
        xyz_line[1],
        color=thin_line_color,
    )
    geoms.append(thin_line)

    return geoms


def make_scene_geometries(
    points: np.ndarray,
    colors: np.ndarray,
    uv: np.ndarray,
    axis_info: dict,
    cam: dict,
    *,
    add_axis=True,
    add_frame=False,
):
    geoms = []

    pcd = make_open3d_pointcloud(points, colors)
    geoms.append(pcd)

    if add_axis:
        # OCR中心点の球は紛らわしいため表示しない．
        # 黒い太い円柱 + 紫の細線で文字領域長手方向を表示する．
        axis_geoms = make_axis_direction_geometries(
            points=points,
            uv=uv,
            axis_info=axis_info,
            cam=cam,
            cylinder_radius_m=0.0015,
            cylinder_color=(1.0, 1.0, 0.0),
            thin_line_color=(1.0, 0.0, 1.0),
            thin_line_z_offset_m=0.006,
        )
        geoms.extend(axis_geoms)

    if add_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        geoms.append(frame)

    return geoms


def save_pointcloud_ply(points: np.ndarray, colors: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcd = make_open3d_pointcloud(points, colors)
    ok = o3d.io.write_point_cloud(str(out_path), pcd)

    if ok:
        print(f"✅ saved ply: {out_path}")
    else:
        print(f"❌ failed to save ply: {out_path}")


# ============================================================
# 4-window viewer
# ============================================================
def show_four_scenes(
    scenes: list[dict],
    *,
    window_width=760,
    window_height=560,
):
    if len(scenes) != 4:
        raise ValueError("scenes は4個必要です．")

    positions = [
        (40, 40),
        (40 + window_width + 30, 40),
        (40, 40 + window_height + 70),
        (40 + window_width + 30, 40 + window_height + 70),
    ]

    visualizers = []

    try:
        for i, scene in enumerate(scenes):
            vis = o3d.visualization.Visualizer()
            left, top = positions[i]
            vis.create_window(
                window_name=scene["title"],
                width=int(window_width),
                height=int(window_height),
                left=int(left),
                top=int(top),
            )

            for g in scene["geometries"]:
                vis.add_geometry(g)

            opt = vis.get_render_option()
            opt.point_size = float(scene.get("point_size", 2.0))
            opt.background_color = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)

            visualizers.append(vis)

        print("")
        print("===== Open3D 4画面表示 =====")
        print("1: final PLY RGB")
        print("2: final PLY red + removed points blue")
        print("3: before NPY RGB")
        print("4: before NPY red")
        print("黒い太い円柱 + 紫の細線 = OCR文字領域長手方向")
        print("緑 = 最長点群列")
        print("青 = 2画面目で削られた点群")
        print("ウィンドウを閉じるか q で終了してください．")
        print("============================")
        print("")

        alive = [True] * len(visualizers)

        while any(alive):
            for i, vis in enumerate(visualizers):
                if not alive[i]:
                    continue

                alive[i] = bool(vis.poll_events())
                vis.update_renderer()

            time.sleep(0.01)

    finally:
        for vis in visualizers:
            try:
                vis.destroy_window()
            except Exception:
                pass


# ============================================================
# Utility
# ============================================================
def infer_mask_id_from_path(path: Path, default_mask_id="mask6"):
    m = re.search(r"(mask\d+)", path.name)
    if m:
        return m.group(1)
    return default_mask_id


def save_debug_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ saved json: {path}")


# ============================================================
# Main process
# ============================================================
def run_four_viewer(
    base: Path,
    final_ply_path: Path,
    before_npy_path: Path,
    depth_path: Path,
    rgb_path: Path,
    camera_path: Path,
    mask_id: str,
    out_dir: Path,
    *,
    fill_contours: bool = True,
    min_depth_m: float = 0.05,
    max_depth_m: float = 5.0,
    t_bin_size_px: float = 4.0,
    min_points_per_bin: int = 10,
    removed_tolerance_px: int = 3,
    save_outputs: bool = True,
):
    base = Path(base)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = load_camera_params(camera_path)
    depth = load_depth(depth_path)
    rgb = load_rgb(rgb_path)

    H, W = depth.shape[:2]

    # --------------------------------------------------------
    # 1. final PLY
    # --------------------------------------------------------
    final_points, final_rgb_colors = load_ply_pointcloud(final_ply_path)

    final_uv, final_uv_valid = project_points_to_uv(final_points, cam)

    # 画像内に投影される点のみ残す
    in_img = (
        final_uv_valid
        & (final_uv[:, 0] >= 0)
        & (final_uv[:, 0] < W)
        & (final_uv[:, 1] >= 0)
        & (final_uv[:, 1] < H)
    )

    final_points = final_points[in_img]
    final_uv = final_uv[in_img]

    if final_rgb_colors is not None:
        final_rgb_colors = final_rgb_colors[in_img]

    print("===== final PLY after uv filter =====")
    print("points:", len(final_points))
    print("=====================================")

    # --------------------------------------------------------
    # 3. before NPY
    # --------------------------------------------------------
    contours = load_pixel_points_or_contours(before_npy_path)
    before_uv_all = contours_to_uv(
        contours=contours,
        image_shape=depth.shape,
        fill_contours=fill_contours,
    )

    before_points, before_rgb_colors, before_uv = uv_depth_to_pointcloud(
        uv=before_uv_all,
        depth=depth,
        rgb=rgb,
        cam=cam,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )

    # --------------------------------------------------------
    # OCR axis
    # --------------------------------------------------------
    axis_info = find_ocr_axis_from_logs(base, mask_id=mask_id)
    if axis_info is None:
        axis_info = fallback_axis_from_uv(before_uv)

    # --------------------------------------------------------
    # Longest columns
    # --------------------------------------------------------
    final_longest_mask, final_col_info = compute_longest_column_indices(
        final_uv,
        axis_info,
        t_bin_size_px=t_bin_size_px,
        min_points_per_bin=min_points_per_bin,
    )

    before_longest_mask, before_col_info = compute_longest_column_indices(
        before_uv,
        axis_info,
        t_bin_size_px=t_bin_size_px,
        min_points_per_bin=min_points_per_bin,
    )

    # --------------------------------------------------------
    # Removed points for screen 2
    # --------------------------------------------------------
    removed_points, removed_uv, removed_mask = compute_removed_points_by_uv(
        before_points=before_points,
        before_uv=before_uv,
        final_uv=final_uv,
        image_shape=depth.shape,
        tolerance_px=int(removed_tolerance_px),
    )

    # --------------------------------------------------------
    # Colors
    # --------------------------------------------------------
    colors_1 = make_colors(
        n=len(final_points),
        base_rgb=final_rgb_colors,
        use_rgb=True,
        uniform_color=(0.8, 0.8, 0.8),
        longest_mask=final_longest_mask,
        longest_color=(0.0, 1.0, 0.0),
    )

    final_red_colors = make_colors(
        n=len(final_points),
        base_rgb=None,
        use_rgb=False,
        uniform_color=(1.0, 0.0, 0.0),
        longest_mask=final_longest_mask,
        longest_color=(0.0, 1.0, 0.0),
    )

    removed_blue_colors = np.tile(
        np.asarray([[0.0, 0.25, 1.0]], dtype=np.float64),
        (len(removed_points), 1),
    )

    if len(removed_points) > 0:
        points_2 = np.vstack([final_points, removed_points])
        colors_2 = np.vstack([final_red_colors, removed_blue_colors])
        uv_2 = np.vstack([final_uv, removed_uv])
    else:
        points_2 = final_points.copy()
        colors_2 = final_red_colors.copy()
        uv_2 = final_uv.copy()

    colors_3 = make_colors(
        n=len(before_points),
        base_rgb=before_rgb_colors,
        use_rgb=True,
        uniform_color=(0.8, 0.8, 0.8),
        longest_mask=before_longest_mask,
        longest_color=(0.0, 1.0, 0.0),
    )

    colors_4 = make_colors(
        n=len(before_points),
        base_rgb=None,
        use_rgb=False,
        uniform_color=(1.0, 0.0, 0.0),
        longest_mask=before_longest_mask,
        longest_color=(0.0, 1.0, 0.0),
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    if save_outputs:
        save_pointcloud_ply(
            final_points,
            colors_1,
            out_dir / f"{mask_id}_view1_final_rgb_with_longest_column.ply",
        )
        save_pointcloud_ply(
            points_2,
            colors_2,
            out_dir / f"{mask_id}_view2_final_red_removed_blue_dilate{removed_tolerance_px}.ply",
        )
        save_pointcloud_ply(
            before_points,
            colors_3,
            out_dir / f"{mask_id}_view3_before_rgb_with_longest_column.ply",
        )
        save_pointcloud_ply(
            before_points,
            colors_4,
            out_dir / f"{mask_id}_view4_before_red_with_longest_column.ply",
        )

        save_debug_json(out_dir / f"{mask_id}_four_viewer_debug.json", {
            "base": str(base),
            "mask_id": mask_id,
            "final_ply_path": str(final_ply_path),
            "before_npy_path": str(before_npy_path),
            "depth_path": str(depth_path),
            "rgb_path": str(rgb_path),
            "camera_path": str(camera_path),
            "removed_tolerance_px": int(removed_tolerance_px),
            "axis_visualization": {
                "thick_cylinder_color": "black",
                "thin_line_color": "magenta",
                "center_sphere": "disabled",
            },
            "axis_info": {
                "center": np.asarray(axis_info["center"], dtype=float).tolist(),
                "axis": np.asarray(axis_info["axis"], dtype=float).tolist(),
                "source": axis_info.get("source"),
                "method": axis_info.get("method"),
            },
            "final_point_count": int(len(final_points)),
            "before_point_count": int(len(before_points)),
            "removed_point_count": int(len(removed_points)),
            "removed_ratio_vs_before": float(len(removed_points) / max(len(before_points), 1)),
            "final_longest_column": final_col_info,
            "before_longest_column": before_col_info,
        })

    # --------------------------------------------------------
    # Make scenes
    # --------------------------------------------------------
    scene1 = {
        "title": "1 final PLY RGB + longest column",
        "point_size": 2.0,
        "geometries": make_scene_geometries(
            points=final_points,
            colors=colors_1,
            uv=final_uv,
            axis_info=axis_info,
            cam=cam,
            add_axis=True,
        ),
    }

    scene2 = {
        "title": "2 final red + removed blue",
        "point_size": 2.0,
        "geometries": make_scene_geometries(
            points=points_2,
            colors=colors_2,
            uv=uv_2,
            axis_info=axis_info,
            cam=cam,
            add_axis=True,
        ),
    }

    scene3 = {
        "title": "3 before NPY RGB + longest column",
        "point_size": 2.0,
        "geometries": make_scene_geometries(
            points=before_points,
            colors=colors_3,
            uv=before_uv,
            axis_info=axis_info,
            cam=cam,
            add_axis=True,
        ),
    }

    scene4 = {
        "title": "4 before NPY red + longest column",
        "point_size": 2.0,
        "geometries": make_scene_geometries(
            points=before_points,
            colors=colors_4,
            uv=before_uv,
            axis_info=axis_info,
            cam=cam,
            add_axis=True,
        ),
    }

    show_four_scenes([scene1, scene2, scene3, scene4])


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="4画面点群ビューア．final PLYとbefore NPYを比較表示する．"
    )

    parser.add_argument(
        "--base",
        type=str,
        default="/home/book/pro_book/pro_hand_book_python/captures/100test/62",
        help="対象ログフォルダ．",
    )

    parser.add_argument(
        "--mask-id",
        type=str,
        default=None,
        help="mask6 など．未指定ならfinal PLY名から推定し，失敗時はmask6．",
    )

    parser.add_argument(
        "--final-ply",
        type=str,
        default=None,
        help="1,2画面目に使う最終PLY．未指定なら base/mask6_offline_final_pointcloud_colored.ply．",
    )

    parser.add_argument(
        "--before-npy",
        type=str,
        default=None,
        help="3,4画面目に使うbefore NPY．未指定なら base/mask6_after_depth_prefilter_spine_completed_points.npy．",
    )

    parser.add_argument(
        "--depth",
        type=str,
        default=None,
        help="Depth npy．未指定なら base/after_init_depth.npy．",
    )

    parser.add_argument(
        "--rgb",
        type=str,
        default=None,
        help="RGB画像．未指定なら base/after_init_rgb.png．",
    )

    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="camera_params.json．未指定なら base/camera_params.json．",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="可視化用PLYとdebug jsonの保存先．未指定なら base/debug_four_viewer．",
    )

    parser.add_argument(
        "--no-fill",
        action="store_true",
        help="before NPY入力時に輪郭内部を塗りつぶさず，輪郭点のみ使う．",
    )

    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.05,
        help="Depthから3D復元するときの最小深度[m]．",
    )

    parser.add_argument(
        "--max-depth",
        type=float,
        default=5.0,
        help="Depthから3D復元するときの最大深度[m]．",
    )

    parser.add_argument(
        "--t-bin-size",
        type=float,
        default=4.0,
        help="最長点群列を探すt方向bin幅[pixel]．",
    )

    parser.add_argument(
        "--min-points-per-bin",
        type=int,
        default=10,
        help="最長点群列の候補にする最小点数．",
    )

    parser.add_argument(
        "--removed-tolerance-px",
        type=int,
        default=3,
        help="削除点判定でfinal_uvを膨張させる半径[pixel]．まず3，背表紙が青く残る場合は5を推奨．",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="可視化用PLYとdebug jsonを保存しない．",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    base = Path(args.base)

    if args.final_ply is None:
        final_ply_path = base / "mask6_offline_final_pointcloud_colored.ply"
    else:
        final_ply_path = Path(args.final_ply)

    if args.before_npy is None:
        before_npy_path = base / "mask6_after_depth_prefilter_spine_completed_points.npy"
    else:
        before_npy_path = Path(args.before_npy)

    depth_path = Path(args.depth) if args.depth is not None else base / "after_init_depth.npy"
    rgb_path = Path(args.rgb) if args.rgb is not None else base / "after_init_rgb.png"
    camera_path = Path(args.camera) if args.camera is not None else base / "camera_params.json"

    if args.mask_id is None:
        mask_id = infer_mask_id_from_path(final_ply_path, default_mask_id="mask6")
    else:
        mask_id = str(args.mask_id)

    out_dir = Path(args.out_dir) if args.out_dir is not None else base / "debug_four_viewer"

    run_four_viewer(
        base=base,
        final_ply_path=final_ply_path,
        before_npy_path=before_npy_path,
        depth_path=depth_path,
        rgb_path=rgb_path,
        camera_path=camera_path,
        mask_id=mask_id,
        out_dir=out_dir,
        fill_contours=not args.no_fill,
        min_depth_m=float(args.min_depth),
        max_depth_m=float(args.max_depth),
        t_bin_size_px=float(args.t_bin_size),
        min_points_per_bin=int(args.min_points_per_bin),
        removed_tolerance_px=int(args.removed_tolerance_px),
        save_outputs=not args.no_save,
    )


if __name__ == "__main__":
    main()