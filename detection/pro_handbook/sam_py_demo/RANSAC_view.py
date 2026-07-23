#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import time
import numpy as np
import open3d as o3d


def find_one(debug_dir: Path, pattern: str) -> Path:
    hits = sorted(debug_dir.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"not found: {pattern}")
    return hits[0]


def load_pcd(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise RuntimeError(f"empty point cloud: {path}")
    return pcd


def copy_pcd(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).copy())

    if pcd.has_colors():
        pcd2.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors).copy())

    if pcd.has_normals():
        pcd2.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals).copy())

    return pcd2


def center_pcd(pcd: o3d.geometry.PointCloud):
    """
    点群を重心中心に移動する。
    before / after を同じ位置で比較しやすくするため。
    """
    pcd2 = copy_pcd(pcd)
    pts = np.asarray(pcd2.points)
    center = pts.mean(axis=0)
    pcd2.translate(-center)
    return pcd2, center


def make_removed_points_pcd(
    before_pcd: o3d.geometry.PointCloud,
    after_pcd: o3d.geometry.PointCloud,
    match_tol: float = 1e-6,
) -> o3d.geometry.PointCloud:
    """
    beforeにあってafterにない点を、RANSACで削られた点として赤色で返す。
    """
    before_pts = np.asarray(before_pcd.points)
    after_pts = np.asarray(after_pcd.points)

    if len(before_pts) == 0:
        raise RuntimeError("before point cloud is empty")
    if len(after_pts) == 0:
        removed_pts = before_pts
    else:
        # scipyが使える場合はKDTreeで最近傍距離を見る
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(after_pts)
            dists, _ = tree.query(before_pts, k=1)
            removed_mask = dists > match_tol

        except Exception:
            # scipyがない場合のフォールバック
            # 小数を丸めて集合比較する
            decimals = max(0, int(abs(np.log10(match_tol))))
            before_key = np.round(before_pts, decimals=decimals)
            after_key = np.round(after_pts, decimals=decimals)

            after_set = {tuple(p) for p in after_key}
            removed_mask = np.array(
                [tuple(p) not in after_set for p in before_key],
                dtype=bool,
            )

        removed_pts = before_pts[removed_mask]

    removed_pcd = o3d.geometry.PointCloud()
    removed_pcd.points = o3d.utility.Vector3dVector(removed_pts)

    # 赤色
    removed_colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (len(removed_pts), 1))
    removed_pcd.colors = o3d.utility.Vector3dVector(removed_colors)

    print(f"削除点数: {len(removed_pts)}")

    return removed_pcd


def apply_same_center(pcd: o3d.geometry.PointCloud, center):
    """
    指定したcenterで点群を中心化する。
    RANSAC後と削除点を同じ座標系で重ねるために使う。
    """
    pcd2 = copy_pcd(pcd)
    pcd2.translate(-center)
    return pcd2


def create_coord_frame(size=0.03):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


def run_two_windows(before_vis_pcd, after_vis_pcd, removed_vis_pcd):
    """
    2つのOpen3Dウィンドウを同時に出す。
    左: RANSAC前
    右: RANSAC後 + 削除点赤
    """
    vis_before = o3d.visualization.Visualizer()
    vis_after = o3d.visualization.Visualizer()

    vis_before.create_window(
        window_name="Before RANSAC",
        width=700,
        height=800,
        left=50,
        top=50,
    )

    vis_after.create_window(
        window_name="After RANSAC + Removed points in red",
        width=700,
        height=800,
        left=800,
        top=50,
    )

    vis_before.add_geometry(before_vis_pcd)
    vis_before.add_geometry(create_coord_frame(size=0.03))

    vis_after.add_geometry(after_vis_pcd)
    vis_after.add_geometry(removed_vis_pcd)
    vis_after.add_geometry(create_coord_frame(size=0.03))

    # 点サイズなど
    opt_before = vis_before.get_render_option()
    opt_after = vis_after.get_render_option()

    opt_before.point_size = 2.0
    opt_after.point_size = 2.0
    opt_before.background_color = np.asarray([1.0, 1.0, 1.0])
    opt_after.background_color = np.asarray([1.0, 1.0, 1.0])

    print("\n表示:")
    print("  左ウィンドウ: RANSAC前")
    print("  右ウィンドウ: RANSAC後 + RANSACで削られた点を赤で重ね描画")
    print("  終了するには、両方のOpen3Dウィンドウを閉じてください。")

    while True:
        alive_before = vis_before.poll_events()
        alive_after = vis_after.poll_events()

        vis_before.update_renderer()
        vis_after.update_renderer()

        if not alive_before and not alive_after:
            break

        time.sleep(0.01)

    vis_before.destroy_window()
    vis_after.destroy_window()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug-dir",
        required=True,
        help="debug_step_by_step_pointcloud_rgb directory",
    )
    parser.add_argument(
        "--match-tol",
        type=float,
        default=1e-6,
        help="point matching tolerance [m]",
    )
    args = parser.parse_args()

    debug_dir = Path(args.debug_dir).expanduser().resolve()

    before_path = find_one(
        debug_dir,
        "*_04_before_ocr_text_ransac_or_final_if_clean_pointcloud_rgb.ply",
    )
    after_path = find_one(
        debug_dir,
        "*_05_after_ocr_text_ransac_plane_filter_pointcloud_rgb.ply",
    )

    print("RANSAC前:", before_path)
    print("RANSAC後:", after_path)

    pcd_before = load_pcd(before_path)
    pcd_after = load_pcd(after_path)

    print(f"RANSAC前 点数: {len(pcd_before.points)}")
    print(f"RANSAC後 点数: {len(pcd_after.points)}")

    removed_pcd = make_removed_points_pcd(
        pcd_before,
        pcd_after,
        match_tol=args.match_tol,
    )

    # beforeの重心を基準に全て同じ座標へ中心化
    before_vis_pcd, before_center = center_pcd(pcd_before)
    after_vis_pcd = apply_same_center(pcd_after, before_center)
    removed_vis_pcd = apply_same_center(removed_pcd, before_center)

    run_two_windows(
        before_vis_pcd=before_vis_pcd,
        after_vis_pcd=after_vis_pcd,
        removed_vis_pcd=removed_vis_pcd,
    )


if __name__ == "__main__":
    main()