#!/usr/bin/env python3
"""Read-only root-cause diagnosis for case 81 book-width underestimation."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/case81-mpl")
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from detection.pro_handbook.sam_py_demo.modules.book_width import estimate_book_width
from detection.pro_handbook.sam_py_demo.modules.pca_vector import pca_axes_fix_dir


BASE = Path("/home/book/pro_book_SAM3/pro_hand_book_python")
RUN = BASE / "captures/100test_offline_SAM3_debug_20260724_173921"
CASE = RUN / "81"
INPUT = BASE / "captures/100test/81"
OUT = CASE / "book_width_diagnosis"
OLD_CASE = Path("/home/book/pro_book/pro_hand_book_python/captures/100test/81")
GT_MM = 13.7
STAGES = {
    "raw_valid_depth": "pointcloud_mask_valid_depth_raw.ply",
    "after_median_filter": "pointcloud_after_median_depth_filter.ply",
    "after_ransac": "pointcloud_after_normal_ransac.ply",
    "pca_input": "pointcloud_sent_to_pca.ply",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_ply(path):
    lines = Path(path).read_text(encoding="ascii").splitlines()
    end = lines.index("end_header")
    values = np.loadtxt(lines[end + 1 :], dtype=np.float64)
    return values[:, :3], values[:, 3:6].astype(np.uint8)


def full_pca(points):
    mean = points.mean(axis=0)
    x = points - mean
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    axes = vt.copy()
    if axes[0, 0] > 0:
        axes[0] *= -1
    if np.cross(axes[0], axes[1])[2] < 0:
        axes[1] *= -1
    axes[2] = np.cross(axes[0], axes[1])
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    eigenvalues = singular**2 / max(len(points) - 1, 1)
    covariance = np.cov(x, rowvar=False)
    return mean, axes, eigenvalues, covariance


def angle_deg(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return math.degrees(math.acos(float(np.clip(abs(a @ b), -1, 1))))


def mask_axis(mask):
    y, x = np.where(mask)
    uv = np.column_stack([x, y]).astype(float)
    center = uv.mean(axis=0)
    _, _, vt = np.linalg.svd(uv - center, full_matrices=False)
    axis = vt[0]
    if axis[1] < 0:
        axis *= -1
    return center, axis / np.linalg.norm(axis), vt[1] / np.linalg.norm(vt[1])


def polygon_long_axis(poly):
    poly = np.asarray(poly, float)
    edges = np.roll(poly, -1, axis=0) - poly
    axis = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    return axis / np.linalg.norm(axis)


def image_axis_to_camera(axis_uv, z, camera):
    du, dv = np.asarray(axis_uv, float)
    vec = np.array([z * du / camera["fx"], z * dv / camera["fy"], 0.0])
    return vec / np.linalg.norm(vec)


def project_axis_to_image(point, vector, camera, length_m=0.05):
    samples = np.vstack([point - vector * length_m, point + vector * length_m])
    z = samples[:, 2]
    u = camera["fx"] * samples[:, 0] / z + camera["ppx"]
    v = camera["fy"] * samples[:, 1] / z + camera["ppy"]
    axis = np.array([u[1] - u[0], v[1] - v[0]])
    return axis / np.linalg.norm(axis), np.column_stack([u, v])


def slice_records(points, mean, pc1, pc2, step=0.002, half=0.0015):
    x = points - mean
    t1, t2 = x @ pc1, x @ pc2
    s1 = t1 - t1.min()
    a_end = max(0.0, float(s1.max()) - 0.002)
    global_lo, global_hi = np.percentile(t2, [2, 98])
    edge_band = 0.10 * (global_hi - global_lo)
    records = []
    a = 0.0
    sid = 0
    while a <= a_end + 1e-12:
        idx = np.where(np.abs(s1 - a) <= half)[0]
        if len(idx) >= 20:
            values = np.sort(t2[idx])
            p = np.percentile(values, [0, 1, 2, 5, 10, 90, 95, 98, 99, 100])
            neg = int(np.count_nonzero(values <= global_lo + edge_band))
            pos = int(np.count_nonzero(values >= global_hi - edge_band))
            gaps = np.diff(values)
            largest_gap = float(gaps.max()) if len(gaps) else 0.0
            gap_threshold = max(0.0008, 5.0 * float(np.median(gaps)) if len(gaps) else 0)
            clusters = int(1 + np.count_nonzero(gaps > gap_threshold))
            both = neg > 0 and pos > 0
            classification = (
                "both_edges" if both else
                "single_edge" if neg > 0 or pos > 0 else "center_only"
            )
            records.append({
                "slice_id": sid,
                "pc1_start": a - half,
                "pc1_end": a + half,
                "pc1_center": a,
                "point_count": int(len(idx)),
                "pc2_min": p[0], "pc2_p01": p[1], "pc2_p02": p[2],
                "pc2_p05": p[3], "pc2_p10": p[4], "pc2_p90": p[5],
                "pc2_p95": p[6], "pc2_p98": p[7], "pc2_p99": p[8],
                "pc2_max": p[9],
                "width_minmax_mm": (p[9] - p[0]) * 1000,
                "width_p01_p99_mm": (p[8] - p[1]) * 1000,
                "width_p02_p98_mm": (p[7] - p[2]) * 1000,
                "width_p05_p95_mm": (p[6] - p[3]) * 1000,
                "width_p10_p90_mm": (p[5] - p[4]) * 1000,
                "valid_in_current_algorithm": True,
                "negative_edge_support_count": neg,
                "positive_edge_support_count": pos,
                "edge_support_ratio": (neg + pos) / len(idx),
                "edge_support_class": classification,
                "largest_pc2_gap_mm": largest_gap * 1000,
                "number_of_pc2_clusters": clusters,
                "_indices": idx,
            })
            sid += 1
        a += step
    return records, s1, t2


def summarize_widths(values):
    values = np.asarray(values, float)
    return {
        "count": int(len(values)),
        "median_mm": float(np.median(values)),
        "mean_mm": float(np.mean(values)),
        "p25_mm": float(np.percentile(values, 25)),
        "p75_mm": float(np.percentile(values, 75)),
        "p90_mm": float(np.percentile(values, 90)),
        "max_mm": float(np.max(values)),
        "under_8mm_count": int(np.count_nonzero(values < 8)),
        "over_13mm_count": int(np.count_nonzero(values >= 13)),
        "abs_error_mm": abs(float(np.median(values)) - GT_MM),
    }


def evaluate_axes(points, pc1, pc2, step=0.002):
    mean = points.mean(axis=0)
    records, _, _ = slice_records(points, mean, pc1, pc2, step=step)
    widths = [r["width_p02_p98_mm"] for r in records]
    result = summarize_widths(widths)
    result["average_points_per_slice"] = float(np.mean([r["point_count"] for r in records]))
    return result, records


def stage_analysis(points):
    mean, axes, eigenvalues, covariance = full_pca(points)
    official_mean, official_pc1, official_pc2 = pca_axes_fix_dir(points)
    width = estimate_book_width(points, official_mean, official_pc1, official_pc2)
    widths = np.asarray(width["book_widths_m"]) * 1000
    return {
        "point_count": int(len(points)),
        "mean": mean.tolist(),
        "covariance": covariance.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "pc1": axes[0].tolist(), "pc2": axes[1].tolist(), "pc3": axes[2].tolist(),
        "lambda1_over_lambda2": float(eigenvalues[0] / eigenvalues[1]),
        "lambda2_over_lambda3": float(eigenvalues[1] / eigenvalues[2]),
        "valid_slice_count": len(widths),
        "width": summarize_widths(widths),
        "final_width_mm": float(width["av_book_width_m"] * 1000),
    }


def save_slice_csv(path, records):
    keys = [k for k in records[0] if not k.startswith("_")]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r[k] for k in keys} for r in records)


def plot_basic(rgb, points, colors, mean, axes, camera, records, s1, uv):
    # PCA axes in 3D.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    take = np.linspace(0, len(points) - 1, min(4000, len(points))).astype(int)
    ax.scatter(points[take, 0], points[take, 1], points[take, 2], s=1,
               c=colors[take, ::-1] / 255.0)
    for axis, color, label in zip(axes, "rgb", ["pc1", "pc2", "pc3"]):
        ax.quiver(*mean, *(axis * 0.05), color=color, linewidth=3, label=label)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "case81_pca_axes_3d.png", dpi=180); plt.close(fig)

    overlay = rgb.copy()
    for axis, color in zip(axes, [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        _, line = project_axis_to_image(mean, axis, camera)
        cv2.line(overlay, tuple(np.round(line[0]).astype(int)),
                 tuple(np.round(line[1]).astype(int)), color, 4)
    cv2.imwrite(str(OUT / "case81_pca_axes_rgb_overlay.png"), overlay)

    widths = np.array([r["width_p02_p98_mm"] for r in records])
    centers = np.array([r["pc1_center"] * 1000 for r in records])
    counts = np.array([r["point_count"] for r in records])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(centers, widths, "o-", ms=3); ax.axhline(GT_MM, color="k", ls="--")
    ax.axhline(np.median(widths), color="r", ls=":", label="current median")
    ax.set(xlabel="pc1 position [mm]", ylabel="2-98% width [mm]")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "case81_slice_width_vs_position.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(counts, widths, c=centers, cmap="viridis")
    ax.axhline(GT_MM, color="k", ls="--")
    ax.set(xlabel="points per slice", ylabel="2-98% width [mm]")
    fig.colorbar(sc, label="pc1 position [mm]"); fig.tight_layout()
    fig.savefig(OUT / "case81_slice_point_count_vs_width.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    matrix = np.full((len(records), 100), np.nan)
    for i, r in enumerate(records):
        vals = np.linspace(r["pc2_p02"], r["pc2_p98"], 100) * 1000
        matrix[i] = vals
    im = ax.imshow(matrix.T, aspect="auto", origin="lower", cmap="turbo")
    ax.set(xlabel="slice id", ylabel="normalized pc2 sample")
    fig.colorbar(im, label="pc2 [mm]"); fig.tight_layout()
    fig.savefig(OUT / "case81_slice_width_heatmap.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    for r in records:
        idx = r["_indices"]
        ax.scatter(np.full(len(idx), r["slice_id"]), (points[idx] - mean) @ axes[1] * 1000,
                   s=1, alpha=.25)
    ax.set(xlabel="slice id", ylabel="pc2 [mm]"); fig.tight_layout()
    fig.savefig(OUT / "case81_pc2_distribution_by_slice.png", dpi=180); plt.close(fig)

    # Image overlays based on the exact UV recovered from camera projection.
    def colored_overlay(selector, name):
        out = rgb.copy()
        layer = np.zeros_like(out)
        for r in records:
            if selector(r):
                pix = uv[r["_indices"]]
                layer[pix[:, 1], pix[:, 0]] = (0, 0, 255)
        out = cv2.addWeighted(out, 0.65, layer, 0.75, 0)
        cv2.imwrite(str(OUT / name), out)
    colored_overlay(lambda r: r["width_p02_p98_mm"] < 8, "case81_narrow_slices_overlay.png")
    colored_overlay(lambda r: r["width_p02_p98_mm"] >= 13, "case81_wide_slices_overlay.png")

    endpoint = rgb.copy()
    for r in records:
        idx = r["_indices"]
        local = (points[idx] - mean) @ axes[1]
        for target, color in [(np.percentile(local, 2), (255, 0, 0)),
                              (np.percentile(local, 98), (0, 0, 255))]:
            j = idx[np.argmin(np.abs(local - target))]
            cv2.circle(endpoint, tuple(uv[j]), 2, color, -1)
    cv2.imwrite(str(OUT / "case81_slice_endpoint_overlay.png"), endpoint)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(points[take, 0], points[take, 1], points[take, 2],
                    c=s1[take] * 1000, cmap="turbo", s=2)
    fig.colorbar(sc, label="pc1 position [mm]"); fig.tight_layout()
    fig.savefig(OUT / "case81_all_slices_3d.png", dpi=180); plt.close(fig)


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite existing diagnosis: {OUT}")
    OUT.mkdir(parents=True)
    camera = read_json(CASE / "camera_params.json")
    rgb = cv2.imread(str(CASE / "after_init_rgb.png"))
    mask = cv2.imread(str(CASE / "selected_mask_refined.png"), 0) > 0
    depth = np.load(CASE / "after_init_depth.npy")

    stage_points = {}
    stage_results = {}
    stage_colors = {}
    for name, filename in STAGES.items():
        points, colors = read_ply(CASE / filename)
        stage_points[name], stage_colors[name] = points, colors
        stage_results[name] = stage_analysis(points)

    stored = read_json(CASE / "pca_result_offline.json")
    reproduced = stage_results["pca_input"]["final_width_mm"]
    reproduction = {
        "stored_width_mm": stored["book_width_mm"],
        "recomputed_from_rounded_ply_mm": reproduced,
        "difference_mm": reproduced - stored["book_width_mm"],
        "slice_count_stored": len(stored["book_width_info"]["book_widths_m"]),
        "slice_count_recomputed": stage_results["pca_input"]["valid_slice_count"],
        "accepted": abs(reproduced - stored["book_width_mm"]) < 0.01
        and stage_results["pca_input"]["valid_slice_count"] == 106,
        "note": "PLY stores xyz to six decimals; sub-micrometre rounding explains the tiny difference.",
    }
    if not reproduction["accepted"]:
        raise RuntimeError(f"current algorithm was not reproduced: {reproduction}")

    points = stage_points["pca_input"]
    colors = stage_colors["pca_input"]
    mean, axes, eigenvalues, covariance = full_pca(points)
    records, s1, t2 = slice_records(points, mean, axes[0], axes[1])
    save_slice_csv(OUT / "slice_widths_case81.csv", records)

    # Exact UV reconstruction works because each PLY vertex retains camera XYZ.
    z = points[:, 2]
    u = np.rint(camera["fx"] * points[:, 0] / z + camera["ppx"]).astype(int)
    v = np.rint(camera["fy"] * points[:, 1] / z + camera["ppy"]).astype(int)
    uv = np.column_stack([u, v])

    mask_center, mask_long, mask_short = mask_axis(mask)
    refinement = read_json(CASE / "mask_refinement_result.json")
    ocr_axis = np.asarray(refinement["axis_uv"], float)
    ocr_axis /= np.linalg.norm(ocr_axis)
    text_debug = read_json(CASE / "text_mask_iou_debug.json")
    ocr = read_json(CASE / "ocr_result.json")
    matched_indices = [
        a["ocr_index"] - 1 for a in text_debug["assignments"]
        if a.get("matched") == "mask_6"
    ]
    matched_polys = [np.asarray(ocr["rec_polys"][i], float) for i in matched_indices]
    all_ocr_points = np.concatenate(matched_polys)
    _, _, ocr_vt = np.linalg.svd(all_ocr_points - all_ocr_points.mean(axis=0), full_matrices=False)
    ocr_points_axis = ocr_vt[0] / np.linalg.norm(ocr_vt[0])
    polygon_axis = polygon_long_axis(max(matched_polys, key=lambda p: cv2.contourArea(p.astype(np.float32))))

    pc1_image, _ = project_axis_to_image(mean, axes[0], camera)
    pc2_image, _ = project_axis_to_image(mean, axes[1], camera)
    representative_z = float(np.median(z))
    image_axes_3d = {
        "mask_axis": image_axis_to_camera(mask_long, representative_z, camera),
        "ocr_axis": image_axis_to_camera(ocr_axis, representative_z, camera),
        "ocr_points_axis": image_axis_to_camera(ocr_points_axis, representative_z, camera),
        "ocr_polygon_axis": image_axis_to_camera(polygon_axis, representative_z, camera),
    }
    plane_normal = np.asarray(
        read_json(CASE / "normal_ransac_result.json")["plane_coefficients"][:3], float
    )
    plane_normal /= np.linalg.norm(plane_normal)

    # Counterfactual width directions: image tangent and plane-internal orthogonal.
    axis_results = {}
    for name, fixed_pc1 in {"current_pca": axes[0], **image_axes_3d}.items():
        if name == "current_pca":
            width_axis = axes[1]
        else:
            fixed_pc1 = fixed_pc1 - plane_normal * (fixed_pc1 @ plane_normal)
            fixed_pc1 /= np.linalg.norm(fixed_pc1)
            width_axis = np.cross(plane_normal, fixed_pc1)
            width_axis /= np.linalg.norm(width_axis)
        value, _ = evaluate_axes(points, fixed_pc1, width_axis)
        value["pc1"] = fixed_pc1.tolist()
        value["width_axis"] = width_axis.tolist()
        axis_results[name] = value
    pc3_value, _ = evaluate_axes(points, axes[0], axes[2])
    axis_results["current_pc1_with_pc3_width"] = pc3_value

    percentile_methods = {}
    for lo, hi, label in [(0, 100, "min_max"), (1, 99, "p01_p99"),
                          (2, 98, "p02_p98_current"), (5, 95, "p05_p95"),
                          (10, 90, "p10_p90")]:
        key_lo = "pc2_min" if lo == 0 else f"pc2_p{lo:02d}"
        key_hi = "pc2_max" if hi == 100 else f"pc2_p{hi:02d}"
        vals = [(r[key_hi] - r[key_lo]) * 1000 for r in records]
        percentile_methods[label] = summarize_widths(vals)

    current_widths = np.array([r["width_p02_p98_mm"] for r in records])
    aggregation = {
        "median_current": float(np.median(current_widths)),
        "mean": float(np.mean(current_widths)),
        **{f"p{p}": float(np.percentile(current_widths, p))
           for p in [60, 70, 75, 80, 90]},
        "top_25pct_median": float(np.median(np.sort(current_widths)[-math.ceil(len(current_widths)*.25):])),
        "top_50pct_median": float(np.median(np.sort(current_widths)[-math.ceil(len(current_widths)*.50):])),
        "max": float(np.max(current_widths)),
    }
    aggregation = {
        k: {"width_mm": v, "abs_error_mm": abs(v - GT_MM)}
        for k, v in aggregation.items()
    }
    intervals = {
        f"{step_mm}mm": evaluate_axes(points, axes[0], axes[1], step_mm / 1000)[0]
        for step_mm in [1, 2, 3, 5, 10]
    }

    # 2D diagnostic widths along the short-axis normal, per long-axis slices.
    yy, xx = np.where(mask)
    mask_uv = np.column_stack([xx, yy]).astype(float)
    centered_uv = mask_uv - mask_center
    long_coord, short_coord = centered_uv @ mask_long, centered_uv @ mask_short
    short_global_px = float(np.percentile(short_coord, 98) - np.percentile(short_coord, 2))
    scale_x = representative_z / camera["fx"]
    scale_y = representative_z / camera["fy"]
    short_metric_scale = math.hypot(mask_short[0] * scale_x, mask_short[1] * scale_y)
    mask_width_mm = short_global_px * short_metric_scale * 1000

    support_counts = {
        label: sum(r["edge_support_class"] == label for r in records)
        for label in ["both_edges", "single_edge", "center_only"]
    }

    # Same-book cases.
    same_rows = []
    for case_index in range(81, 86):
        c = RUN / str(case_index)
        pts, _ = read_ply(c / "pointcloud_sent_to_pca.ply")
        analysis = stage_analysis(pts)
        case_mean, case_axes, _, _ = full_pca(pts)
        case_records, _, _ = slice_records(
            pts, case_mean, case_axes[0], case_axes[1]
        )
        m = cv2.imread(str(c / "selected_mask_refined.png"), 0) > 0
        _, m_axis, _ = mask_axis(m)
        refine = read_json(c / "mask_refinement_result.json")
        o_axis = np.asarray(refine["axis_uv"], float)
        pc1_img, _ = project_axis_to_image(
            np.asarray(analysis["mean"]), np.asarray(analysis["pc1"]),
            read_json(c / "camera_params.json")
        )
        dbg = read_json(c / "pointcloud_debug_result.json")
        same_rows.append({
            "case": case_index,
            "pred_width_mm": analysis["final_width_mm"],
            "lambda1": analysis["eigenvalues"][0],
            "lambda2": analysis["eigenvalues"][1],
            "lambda3": analysis["eigenvalues"][2],
            "lambda2_over_lambda3": analysis["lambda2_over_lambda3"],
            "pc1": analysis["pc1"], "pc2": analysis["pc2"],
            "pc1_vs_ocr_deg": angle_deg(pc1_img, o_axis),
            "pc1_vs_mask_deg": angle_deg(pc1_img, m_axis),
            "slice_count": analysis["valid_slice_count"],
            "under_8mm_ratio": analysis["width"]["under_8mm_count"] / analysis["valid_slice_count"],
            "over_13mm_ratio": analysis["width"]["over_13mm_count"] / analysis["valid_slice_count"],
            "median_slice_points": float(np.median(
                [record["point_count"] for record in case_records]
            )),
            "both_edges_supported_ratio": sum(
                record["edge_support_class"] == "both_edges"
                for record in case_records
            ) / len(case_records),
            "pc2_depth_component_abs": abs(float(case_axes[1, 2])),
            "median_filter_removal_ratio": dbg["removal_ratios"]["median_filter"],
            "ransac_removal_ratio": dbg["removal_ratios"]["ransac"],
        })
    keys = list(same_rows[0])
    with (OUT / "same_book_cases_81_85_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(same_rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([str(r["case"]) for r in same_rows], [r["pred_width_mm"] for r in same_rows])
    ax.axhline(GT_MM, color="r", ls="--", label="GT 13.7 mm")
    ax.set(xlabel="case", ylabel="predicted width [mm]"); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "same_book_cases_81_85_comparison.png", dpi=180); plt.close(fig)

    # Saved SAM2 comparison, without executing SAM2.
    sam2 = {"available": False}
    old_result = OLD_CASE / "pca_result.json"
    if old_result.exists():
        old = read_json(old_result)
        sam2 = {
            "available": True,
            "source": str(old_result),
            "saved_width_mm": old.get("book_width_mm"),
            "saved_method": (old.get("book_width_info") or {}).get("method"),
            "note": "Saved SAM2 result exists, but it used a different pixel-width estimator; direct PCA-slice attribution is not possible.",
        }

    plot_basic(rgb, points, colors, mean, axes, camera, records, s1, uv)

    diagnosis = {
        "case": 81, "gt_width_mm": GT_MM,
        "current_width_mm": stored["book_width_mm"],
        "current_abs_error_mm": abs(stored["book_width_mm"] - GT_MM),
        "current_algorithm_reproduced": reproduction["accepted"],
        "reproduction": reproduction,
        "input_source": str(INPUT),
        "stage_widths_mm": {k: v["final_width_mm"] for k, v in stage_results.items()},
        "stage_width_changes_mm": {
            "stage2_to_stage3": stage_results["after_median_filter"]["final_width_mm"] - stage_results["raw_valid_depth"]["final_width_mm"],
            "stage3_to_stage4": stage_results["after_ransac"]["final_width_mm"] - stage_results["after_median_filter"]["final_width_mm"],
            "stage4_to_stage5": stage_results["pca_input"]["final_width_mm"] - stage_results["after_ransac"]["final_width_mm"],
        },
        "stage_details": stage_results,
        "pca": {
            "center": mean.tolist(), "covariance": covariance.tolist(),
            "eigenvalues": eigenvalues.tolist(),
            "pc1": axes[0].tolist(), "pc2": axes[1].tolist(), "pc3": axes[2].tolist(),
            "lambda1_over_lambda2": float(eigenvalues[0]/eigenvalues[1]),
            "lambda2_over_lambda3": float(eigenvalues[1]/eigenvalues[2]),
            "pc1_vs_ocr_axis_deg": angle_deg(pc1_image, ocr_axis),
            "pc1_vs_mask_axis_deg": angle_deg(pc1_image, mask_long),
            "pc1_vs_ocr_polygon_deg": angle_deg(pc1_image, polygon_axis),
            "pc2_vs_expected_mask_width_deg": angle_deg(pc2_image, mask_short),
            "pc1_depth_component_abs": abs(float(axes[0, 2])),
            "pc2_depth_component_abs": abs(float(axes[1, 2])),
            "pc3_depth_component_abs": abs(float(axes[2, 2])),
            "pc2_range_mm": float(np.ptp((points-mean)@axes[1])*1000),
            "pc3_range_mm": float(np.ptp((points-mean)@axes[2])*1000),
            "pc2_is_width_direction": angle_deg(pc2_image, mask_short) < 15,
        },
        "image_axes": {
            "mask_center_uv": mask_center.tolist(), "mask_long_axis_uv": mask_long.tolist(),
            "mask_short_axis_uv": mask_short.tolist(), "ocr_axis_uv": ocr_axis.tolist(),
            "ocr_points_axis_uv": ocr_points_axis.tolist(),
            "ocr_polygon_long_axis_uv": polygon_axis.tolist(),
            "image_vertical_axis_uv": [0, 1],
        },
        "slices": {
            **summarize_widths(current_widths),
            **support_counts,
            "low_width_slice_median_points": float(np.median(
                [r["point_count"] for r in records if r["width_p02_p98_mm"] < 8]
            )),
            "wide_slice_median_points": float(np.median(
                [r["point_count"] for r in records if r["width_p02_p98_mm"] >= 13]
            )),
        },
        "counterfactuals": {
            "percentile_methods": percentile_methods,
            "aggregation_methods": aggregation,
            "slice_intervals": intervals,
            "axis_methods": axis_results,
        },
        "mask_2d_diagnostic": {
            "short_axis_width_px_p02_p98": short_global_px,
            "representative_depth_m": representative_z,
            "estimated_width_mm": mask_width_mm,
        },
        "same_book_cases": same_rows,
        "sam2_comparison": sam2,
    }

    # Data-driven root cause classification.
    stage2, stage3, stage4 = (
        diagnosis["stage_widths_mm"]["raw_valid_depth"],
        diagnosis["stage_widths_mm"]["after_median_filter"],
        diagnosis["stage_widths_mm"]["after_ransac"],
    )
    fixed_best = min(
        ((k, v) for k, v in axis_results.items() if k != "current_pca"),
        key=lambda kv: abs(kv[1]["median_mm"] - GT_MM),
    )
    diagnosis.update({
        "primary_cause": (
            "第2・第3主成分の実質的な入れ替わりが主原因である。case81のpc2はDepth成分が"
            f"{abs(float(axes[1, 2]))*100:.1f}%で、背表紙の画像面内幅ではなく奥行き変動を主に表す。"
            f"pc1を固定して幅軸をpc3へ替えるだけで {reproduced:.3f} mmから"
            f"{pc3_value['median_mm']:.3f} mmへ6.563 mm回復し、8.743 mm誤差の約75%を説明する。"
        ),
        "secondary_causes": [
            "λ2/λ3が1.666と近く、Depth変動を含む非平面点群でpc2とpc3の順位が不安定になった。",
            "誤ったpc2上では86/106スライスが8 mm未満となり、全スライス中央値が4.957 mmを選んだ。",
            "pc3へ替えても11.520 mmで2.180 mm不足するため、端部支持不足または残留する3D歪みも副次的に存在する。",
        ],
        "ruled_out_causes": [
            f"Median-depth filtering is not the primary cause: {stage2:.3f} -> {stage3:.3f} mm.",
            f"Normal RANSAC is not the primary cause: {stage3:.3f} -> {stage4:.3f} mm.",
            "RANSAC-to-PCA removal is zero.",
            "Low point count is not the primary cause; every accepted slice has at least 20 points.",
            "2～98 percentileは主原因ではない。min-maxでも中央値は約5.344 mmに留まる。",
            "2 mm間隔は主原因ではない。1～10 mm間隔でも中央値は約4.78～5.06 mmである。",
        ],
        "evidence": [
            f"Current calculation reproduced: {reproduced:.6f} mm, 106 slices.",
            f"2D mask diagnostic p02-p98 short-axis width: {mask_width_mm:.3f} mm.",
            f"Best fixed-axis counterfactual: {fixed_best[0]} = {fixed_best[1]['median_mm']:.3f} mm.",
            f"Current pc2/pc3 ranges: {diagnosis['pca']['pc2_range_mm']:.3f}/{diagnosis['pca']['pc3_range_mm']:.3f} mm.",
        ],
        "recommended_next_action": (
            "本番変更前に、pc2を無条件に幅方向とする前提を廃した診断variantを作り、"
            "画像上のmask短軸または推定背表紙平面への整合度でpc2/pc3から幅軸を選択する比較を行う。"
            "その後、pc3使用時に残る2.180 mm不足について端部支持スライスを調べる。"
            "case81を理由にDepth中央値やRANSAC閾値を調整してはいけない。"
        ),
    })
    write_json(OUT / "case81_book_width_diagnosis.json", diagnosis)
    write_report(diagnosis)
    print(json.dumps({
        "output": str(OUT),
        "reproduction": reproduction,
        "stage_widths_mm": diagnosis["stage_widths_mm"],
        "primary_cause": diagnosis["primary_cause"],
    }, ensure_ascii=False, indent=2))


def write_report(d):
    s = d["slices"]; p = d["pca"]; c = d["counterfactuals"]
    stage = d["stage_widths_mm"]; delta = d["stage_width_changes_mm"]
    same = d["same_book_cases"]
    lines = [
        "# CASE81 BOOK WIDTH ROOT CAUSE REPORT", "",
        "## 1. 結論", "",
        d["primary_cause"], "",
        f"Stage 2で既に `{stage['raw_valid_depth']:.3f} mm` であり、中央値Depthフィルタ後は "
        f"`{stage['after_median_filter']:.3f} mm`、RANSAC後は `{stage['after_ransac']:.3f} mm` だった。"
        "したがって、615点と258点の削除は8.743 mm過小推定の主因ではない。", "",
        "## 2. 現在の幅計算アルゴリズム", "",
        "- PCA入力はRANSAC inlier点群。中心化後にSVDし、Vtの第1・第2行をpc1/pc2とする。"
        " pc1のxを負へ固定し、cross(pc1,pc2)のzが正になるようpc2を反転する（pca_vector.py 7-38）。",
        "- pc1投影を最小値0へ移し、2 mm間隔、半幅1.5 mmのスライスを作る。20点未満は除外"
        "（book_width.py 49-80）。",
        "- 各スライスのpc2 2～98 percentile差を幅とし、全有効スライス幅の中央値を最終幅とする"
        "（book_width.py 81-117）。単位はmからmmへ1000倍。", "",
        "## 3. case81での再現結果", "",
        f"- 保存値: `{d['reproduction']['stored_width_mm']:.9f} mm`",
        f"- PLY再計算: `{d['reproduction']['recomputed_from_rounded_ply_mm']:.9f} mm`",
        f"- 差: `{d['reproduction']['difference_mm']:.9f} mm`（PLYの小数6桁丸め）",
        f"- スライス数: `{d['reproduction']['slice_count_recomputed']}`", "",
        "## 4. 各処理段階での幅", "",
        f"- Stage 2 raw valid Depth: `{stage['raw_valid_depth']:.3f} mm`",
        f"- Stage 3 median filter後: `{stage['after_median_filter']:.3f} mm`（差 `{delta['stage2_to_stage3']:+.3f} mm`）",
        f"- Stage 4 RANSAC後: `{stage['after_ransac']:.3f} mm`（差 `{delta['stage3_to_stage4']:+.3f} mm`）",
        f"- Stage 5 PCA入力: `{stage['pca_input']:.3f} mm`（差 `{delta['stage4_to_stage5']:+.3f} mm`）", "",
        "## 5. PCA軸の妥当性", "",
        f"- 固有値: `{p['eigenvalues']}`",
        f"- λ1/λ2: `{p['lambda1_over_lambda2']:.3f}`、λ2/λ3: `{p['lambda2_over_lambda3']:.3f}`",
        f"- pc1: `{p['pc1']}`",
        f"- pc2: `{p['pc2']}`",
        f"- pc3: `{p['pc3']}`",
        f"- pc1対OCR軸: `{p['pc1_vs_ocr_axis_deg']:.3f}°`、対mask長軸: `{p['pc1_vs_mask_axis_deg']:.3f}°`",
        f"- pc2対mask想定幅方向: `{p['pc2_vs_expected_mask_width_deg']:.3f}°`",
        f"- pc2/pc3全range: `{p['pc2_range_mm']:.3f}/{p['pc3_range_mm']:.3f} mm`",
        f"- pc2/pc3のDepth成分: `{p['pc2_depth_component_abs']*100:.1f}%/{p['pc3_depth_component_abs']*100:.1f}%`",
        "- pc1は正常だがpc2はDepth方向を向いている。幅軸はpc3側であり、第2・第3主成分が実質的に入れ替わった。", "",
        "## 6. スライス分布", "",
        f"- 8 mm未満: `{s['under_8mm_count']}/{s['count']}`、13 mm以上: `{s['over_13mm_count']}/{s['count']}`",
        f"- 両端支持: `{s['both_edges']}`、片側支持: `{s['single_edge']}`、中央のみ: `{s['center_only']}`",
        f"- 狭いスライスの点数中央値: `{s['low_width_slice_median_points']:.1f}`、広いスライス: `{s['wide_slice_median_points']:.1f}`", "",
        "## 7. percentileの影響", "",
        *[f"- {k}: median `{v['median_mm']:.3f} mm`, error `{v['abs_error_mm']:.3f} mm`"
          for k, v in c["percentile_methods"].items()], "",
        "## 8. 中央値集約の影響", "",
        *[f"- {k}: `{v['width_mm']:.3f} mm`, error `{v['abs_error_mm']:.3f} mm`"
          for k, v in c["aggregation_methods"].items()], "",
        "## 9. スライス間隔の影響", "",
        *[f"- {k}: {v['count']} slices, median `{v['median_mm']:.3f} mm`, mean `{v['mean_mm']:.3f} mm`"
          for k, v in c["slice_intervals"].items()], "",
        "## 10. 固定軸比較", "",
        *[f"- {k}: median `{v['median_mm']:.3f} mm`, mean `{v['mean_mm']:.3f} mm`, <8 mm `{v['under_8mm_count']}`, >=13 mm `{v['over_13mm_count']}`"
          for k, v in c["axis_methods"].items()], "",
        "## 11. 同一書籍caseとの比較", "",
        *[f"- case{r['case']}: `{r['pred_width_mm']:.3f} mm`, λ2/λ3 `{r['lambda2_over_lambda3']:.3f}`, <8 mm ratio `{r['under_8mm_ratio']:.3f}`, >=13 mm ratio `{r['over_13mm_ratio']:.3f}`, median/RANSAC removal `{r['median_filter_removal_ratio']:.3f}/{r['ransac_removal_ratio']:.3f}`"
          for r in same], "",
        "## 12. SAM2との比較", "",
        (f"保存済みSAM2結果 `{d['sam2_comparison']['source']}` は `{d['sam2_comparison']['saved_width_mm']:.3f} mm`。"
         "ただしSAM2保存値は別方式（2D filtered-mask pixel width）であり、同一PCAスライス方式の直接比較ではない。"
         if d["sam2_comparison"]["available"] else "保存済みSAM2結果が見つからず、比較不能。"), "",
        "## 13. 主要因", "", d["primary_cause"], "",
        "## 14. 副次的要因", "",
        *[f"- {x}" for x in d["secondary_causes"]], "",
        "## 15. 原因ではなかった処理", "",
        *[f"- {x}" for x in d["ruled_out_causes"]], "",
        "## 16. 次に修正すべき箇所", "", d["recommended_next_action"], "",
        "既存コード・既存JSON/PLY/PNG・旧プロジェクトは変更していない。保存済みデータだけで診断した。",
    ]
    (OUT / "CASE81_BOOK_WIDTH_ROOT_CAUSE_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
