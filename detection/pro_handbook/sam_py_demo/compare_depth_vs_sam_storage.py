from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


# Depth版
from detection.pro_handbook.sam_py_demo.Storage_rev import (
    run_capture_and_pca_depth_space,
)

# SAM版
from detection.pro_handbook.sam_py_demo.rs_book_capture_and_pointcloud_storage import (
    run_capture_and_pca,
)


# ============================================================
# Utility
# ============================================================

def normalize_result(ret: Any) -> tuple[Any, Any, Any, dict]:
    """
    認識関数の戻り値の違いを吸収する．

    Depth版:
      angle_rad, first_target_cam, res
    または
      angle_rad, first_target_cam, final_target_cam, res

    SAM版:
      angle_rad, first_target_cam, final_target_cam, res
    """
    if not isinstance(ret, tuple):
        raise RuntimeError(f"return value must be tuple, got {type(ret)}")

    if len(ret) == 3:
        angle_rad, first_target_cam, res = ret
        final_target_cam = first_target_cam
        return angle_rad, first_target_cam, final_target_cam, res

    if len(ret) == 4:
        angle_rad, first_target_cam, final_target_cam, res = ret
        return angle_rad, first_target_cam, final_target_cam, res

    raise RuntimeError(f"unexpected return tuple length: {len(ret)}")


def to_jsonable(obj: Any) -> Any:
    """
    numpy配列，Pathなどをjson保存可能な形に変換する．
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)

    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    return obj


def copy_if_exists(src: Optional[Path], dst: Path) -> Optional[Path]:
    """
    srcが存在すればdstへコピーする．
    srcとdstが同じ場合は何もしない．
    """
    if src is None:
        return None

    src = Path(src)
    if not src.exists():
        return None

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        if src.resolve() == dst.resolve():
            return dst
    except Exception:
        pass

    shutil.copy2(str(src), str(dst))
    print(f"[SAVE] copied: {dst}")
    return dst


# ============================================================
# Overlay search / drawing
# ============================================================

def find_overlay_path(
    res: dict,
    out_dir: Path,
    *,
    preferred_names: list[str],
) -> Optional[Path]:
    """
    認識結果のoverlay画像を探す．

    優先順位：
      1. res["overlay_path"]
      2. res["shot_dir"] 内の preferred_names
      3. out_dir以下の preferred_names
      4. out_dir以下の *overlay*.png / *guideline*.png
    """
    if isinstance(res, dict):
        overlay_path = res.get("overlay_path")
        if overlay_path is not None:
            p = Path(overlay_path)
            if p.exists():
                return p

        shot_dir = res.get("shot_dir")
        if shot_dir is not None:
            shot_dir = Path(shot_dir)
            if shot_dir.exists():
                for name in preferred_names:
                    p = shot_dir / name
                    if p.exists():
                        return p

    for name in preferred_names:
        found = list(out_dir.glob(f"**/{name}"))
        found = [p for p in found if p.is_file()]
        if found:
            found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return found[0]

    patterns = [
        "**/*overlay*.png",
        "**/*guideline*.png",
    ]

    found = []
    for pat in patterns:
        found.extend(out_dir.glob(pat))

    found = [p for p in found if p.is_file()]
    if not found:
        return None

    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0]


def infer_target_px_from_res(
    res: dict,
    *,
    fallback_line_ratio: float = 0.55,
) -> tuple[Optional[tuple[int, int]], Optional[tuple[int, int]], Optional[tuple[int, int]]]:
    """
    resから画像上の目標点を推定する．

    優先順位：
      1. res["first_target_px_projected"]
      2. res["first_target_px"]
      3. res["line_p0"], res["line_p1"] の間を fallback_line_ratio で補間

    SAM側で first_target_px が返っていない場合でも，
    line_p0 / line_p1 があればガイドライン上に仮の目標点を描画する．
    """
    if not isinstance(res, dict):
        return None, None, None

    line_p0 = res.get("line_p0")
    line_p1 = res.get("line_p1")

    if line_p0 is not None:
        line_p0 = (int(line_p0[0]), int(line_p0[1]))

    if line_p1 is not None:
        line_p1 = (int(line_p1[0]), int(line_p1[1]))

    target_px = None

    for key in ["first_target_px_projected", "first_target_px"]:
        v = res.get(key)
        if v is not None:
            target_px = (int(v[0]), int(v[1]))
            break

    if target_px is None and line_p0 is not None and line_p1 is not None:
        x = (1.0 - fallback_line_ratio) * line_p0[0] + fallback_line_ratio * line_p1[0]
        y = (1.0 - fallback_line_ratio) * line_p0[1] + fallback_line_ratio * line_p1[1]
        target_px = (int(round(x)), int(round(y)))

    return target_px, line_p0, line_p1


def draw_target_on_image(
    img_bgr: np.ndarray,
    *,
    target_px: Optional[tuple[int, int]] = None,
    line_p0: Optional[tuple[int, int]] = None,
    line_p1: Optional[tuple[int, int]] = None,
    label: str = "target",
) -> np.ndarray:
    """
    結果画像に目標点とガイドラインを重ねる．
    文字ラベルは描画しない．
    """
    vis = img_bgr.copy()

    # 赤いガイドライン
    if line_p0 is not None and line_p1 is not None:
        cv2.line(
            vis,
            (int(line_p0[0]), int(line_p0[1])),
            (int(line_p1[0]), int(line_p1[1])),
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    # 赤い目標点
    if target_px is not None:
        x, y = int(target_px[0]), int(target_px[1])
        cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)
        cv2.circle(vis, (x, y), 14, (0, 0, 255), 2)

    return vis

def find_rgb_path(
    res: dict,
    out_dir: Path,
    *,
    preferred_names: list[str] = ["rgb.png", "color.png", "input_rgb.png", "frame.png"],
) -> Optional[Path]:
    """
    認識結果に対応するRGB画像を探す．

    優先順位：
      1. res["rgb_path"]
      2. res["shot_dir"] 内の preferred_names
      3. out_dir 以下の preferred_names
    """
    if isinstance(res, dict):
        rgb_path = res.get("rgb_path")
        if rgb_path is not None:
            p = Path(rgb_path)
            if p.exists():
                return p

        shot_dir = res.get("shot_dir")
        if shot_dir is not None:
            shot_dir = Path(shot_dir)
            if shot_dir.exists():
                for name in preferred_names:
                    p = shot_dir / name
                    if p.exists():
                        return p

    for name in preferred_names:
        found = list(out_dir.glob(f"**/{name}"))
        found = [p for p in found if p.is_file()]
        if found:
            found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return found[0]

    return None

def make_target_overlay_from_result(
    *,
    src_overlay_path: Optional[Path],
    res: dict,
    dst_path: Path,
    method_name: str,
) -> Optional[Path]:
    """
    overlay画像に目標点を描画して保存する．
    """
    if src_overlay_path is None or not Path(src_overlay_path).exists():
        print(f"[WARN] source overlay not found for {method_name}: {src_overlay_path}")
        return None

    img = cv2.imread(str(src_overlay_path))
    if img is None:
        print(f"[WARN] failed to read overlay for {method_name}: {src_overlay_path}")
        return None

    target_px, line_p0, line_p1 = infer_target_px_from_res(res)

    if target_px is None:
        print(f"[WARN] target_px not found for {method_name}. Image is saved without target point.")

    vis = draw_target_on_image(
        img,
        target_px=target_px,
        line_p0=line_p0,
        line_p1=line_p1,
        label=f"{method_name} target",
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), vis)
    print(f"[SAVE] target overlay: {dst_path}")

    return dst_path


# ============================================================
# Side-by-side image
# ============================================================

def read_image_or_blank(
    path: Optional[Path],
    *,
    label: str,
    width: int = 640,
    height: int = 360,
) -> np.ndarray:
    """
    画像を読む．存在しない場合は黒背景画像を返す．
    """
    if path is None or not Path(path).exists():
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            img,
            f"{label}: image not found",
            (30, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return img

    img = cv2.imread(str(path))
    if img is None:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            img,
            f"{label}: failed to read",
            (30, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return img

    return img


def resize_keep_aspect(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img

    scale = target_h / float(h)
    new_w = int(round(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def add_title(img: np.ndarray, title: str) -> np.ndarray:
    h, w = img.shape[:2]
    title_h = 45

    canvas = np.zeros((h + title_h, w, 3), dtype=np.uint8)
    canvas[title_h:, :] = img

    cv2.putText(
        canvas,
        title,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


def make_side_by_side(
    depth_img: np.ndarray,
    sam_img: np.ndarray,
    *,
    target_h: int = 520,
) -> np.ndarray:
    depth_img = resize_keep_aspect(depth_img, target_h)
    sam_img = resize_keep_aspect(sam_img, target_h)

    depth_img = add_title(depth_img, "Depth-based recognition")
    sam_img = add_title(sam_img, "SAM-based recognition")

    max_h = max(depth_img.shape[0], sam_img.shape[0])

    def pad_to_h(img: np.ndarray, h: int) -> np.ndarray:
        ih, iw = img.shape[:2]
        if ih == h:
            return img

        canvas = np.zeros((h, iw, 3), dtype=np.uint8)
        canvas[:ih, :iw] = img
        return canvas

    depth_img = pad_to_h(depth_img, max_h)
    sam_img = pad_to_h(sam_img, max_h)

    gap = np.zeros((max_h, 20, 3), dtype=np.uint8)
    return np.hstack([depth_img, gap, sam_img])


# ============================================================
# Timing
# ============================================================

def load_timing_runs(json_path: Path) -> list[dict[str, Any]]:
    """
    既存の timing_results.json から過去の計算時間を読み込む．
    なければ空リストを返す．
    """
    if not json_path.exists():
        return []

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        runs = data.get("runs", [])
        if isinstance(runs, list):
            return runs

    except Exception as e:
        print(f"[WARN] failed to load timing json: {e}")

    return []


def write_timing_memo(
    memo_path: Path,
    *,
    runs: list[dict[str, Any]],
) -> None:
    """
    計算時間メモを指定形式で保存する．

    例：
      1回目：{(depth : ○○ sec. )，(SAM : ○○ sec)}
      2回目：{(depth : ○○ sec. )，(SAM : ○○ sec)}

      平均：{(depth : ○○ sec. )，(SAM : ○○ sec)}
    """
    memo_path.parent.mkdir(parents=True, exist_ok=True)

    depth_times = [float(r["depth_sec"]) for r in runs]
    sam_times = [float(r["sam_sec"]) for r in runs]

    lines = []

    for i, r in enumerate(runs, start=1):
        lines.append(
            f"{i}回目：{{(depth : {float(r['depth_sec']):.4f} sec. )，"
            f"(SAM : {float(r['sam_sec']):.4f} sec)}}"
        )

    if runs:
        depth_avg = float(np.mean(depth_times))
        sam_avg = float(np.mean(sam_times))

        lines.append("")
        lines.append(
            f"平均：{{(depth : {depth_avg:.4f} sec. )，"
            f"(SAM : {sam_avg:.4f} sec)}}"
        )

    with memo_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_timing_json(
    json_path: Path,
    *,
    runs: list[dict[str, Any]],
) -> None:
    """
    計算時間をjsonでも保存する．
    """
    depth_times = [float(r["depth_sec"]) for r in runs]
    sam_times = [float(r["sam_sec"]) for r in runs]

    data = {
        "runs": runs,
        "average": {
            "depth_sec": float(np.mean(depth_times)) if depth_times else None,
            "sam_sec": float(np.mean(sam_times)) if sam_times else None,
        },
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2)


# ============================================================
# Main
# ============================================================

def run_compare_experiment(
    *,
    out_root: str | Path = "captures_compare_storage",
    width: int = 1280,
    height: int = 720,
    fps: int = 6,
    save_side_by_side: bool = True,
):
    """
    Depth版とSAM版の書籍収納スペース認識を比較する．

    保存形式：
      captures_compare_storage/
      ├── timing_memo.txt
      ├── timing_results.json
      ├── YYYYMMDD_HHMMSS/
      │   ├── depth/
      │   │   ├── depth_space_overlay.png
      │   │   └── depth_space_overlay_with_target.png
      │   ├── sam/
      │   │   ├── sam_space_overlay.png
      │   │   └── sam_space_overlay_with_target.png
      │   ├── compare_depth_vs_sam.png
      │   └── result_summary.json
      └── ...
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")

    exp_dir = out_root / ts
    depth_dir = exp_dir / "depth"
    sam_dir = exp_dir / "sam"

    depth_dir.mkdir(parents=True, exist_ok=True)
    sam_dir.mkdir(parents=True, exist_ok=True)

    # 計算時間メモは親ディレクトリ直下に保存して蓄積する
    timing_memo_path = out_root / "timing_memo.txt"
    timing_json_path = out_root / "timing_results.json"
    timing_runs = load_timing_runs(timing_json_path)

    print("============================================================")
    print("[INFO] Storage space recognition comparison")
    print(f"[INFO] exp_dir   = {exp_dir}")
    print(f"[INFO] depth_dir = {depth_dir}")
    print(f"[INFO] sam_dir   = {sam_dir}")
    print(f"[INFO] timing_memo_path = {timing_memo_path}")
    print(f"[INFO] timing_json_path = {timing_json_path}")
    print("============================================================")

    # ========================================================
    # 1. Depth版
    # ========================================================
    print("\n============================================================")
    print("[INFO] Run Depth-based recognition")
    print("============================================================")

    depth_t0 = time.perf_counter()

    depth_ret = run_capture_and_pca_depth_space(
        out_dir=depth_dir,
        width=width,
        height=height,
        fps=fps,

        # 必要に応じて調整
        crop_y_min=120,
        crop_y_max=560,
        min_area_px=800,
        min_height_px=100,
        horizontal_ratio_thr=3.5,
        max_space_width_px=520,
        max_space_area_px=140000,
    )

    depth_sec = time.perf_counter() - depth_t0

    depth_angle, depth_first, depth_final, depth_res = normalize_result(depth_ret)

    depth_overlay_src = find_overlay_path(
        depth_res,
        depth_dir,
        preferred_names=[
            "depth_space_overlay.png",
            "selected_mask_target_overlay.png",
            "candidate_mask_target_overlay.png",
        ],
    )

    depth_overlay_path = copy_if_exists(
        depth_overlay_src,
        depth_dir / "depth_space_overlay.png",
    )

    depth_target_overlay_path = make_target_overlay_from_result(
        src_overlay_path=depth_overlay_path,
        res=depth_res,
        dst_path=depth_dir / "depth_space_overlay_with_target.png",
        method_name="depth",
    )

    print("[INFO] depth_angle =", depth_angle)
    print("[INFO] depth_first_target_cam =", depth_first)
    print(f"[INFO] depth_time = {depth_sec:.4f} sec")
    print("[INFO] depth_overlay_path =", depth_overlay_path)
    print("[INFO] depth_target_overlay_path =", depth_target_overlay_path)

    # ========================================================
    # 2. SAM版
    # ========================================================
    print("\n============================================================")
    print("[INFO] Run SAM-based recognition")
    print("============================================================")

    sam_t0 = time.perf_counter()

    sam_ret = run_capture_and_pca(
        out_dir=sam_dir,
        width=width,
        height=height,
        fps=fps,
        sam_device="gpu",
        encoder_path="/home/book/pro_book_SAM3/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx",
        decoder_path="/home/book/pro_book_SAM3/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx",
        interactive=False,
    )

    sam_sec = time.perf_counter() - sam_t0

    sam_angle, sam_first, sam_final, sam_res = normalize_result(sam_ret)

    sam_overlay_src = find_overlay_path(
        sam_res,
        sam_dir,
        preferred_names=[
            "depth_space_overlay.png",
            "guideline_overlay.png",
            "selected_mask_target_overlay.png",
            "candidate_mask_target_overlay.png",
        ],
    )

    sam_overlay_path = copy_if_exists(
        sam_overlay_src,
        sam_dir / "sam_space_overlay.png",
    )

    # ========================================================
    # SAM側は RGB画像をベースに with_target を作る
    # ========================================================
    sam_rgb_src = find_rgb_path(
        sam_res,
        sam_dir,
        preferred_names=[
            "rgb.png",
            "color.png",
            "input_rgb.png",
            "frame.png",
        ],
    )

    sam_rgb_path = copy_if_exists(
        sam_rgb_src,
        sam_dir / "sam_rgb.png",
    )

    sam_target_overlay_path = make_target_overlay_from_result(
        src_overlay_path=sam_rgb_path if sam_rgb_path is not None else sam_overlay_path,
        res=sam_res,
        dst_path=sam_dir / "sam_space_overlay_with_target.png",
        method_name="SAM",
    )

    print("[INFO] sam_angle =", sam_angle)
    print("[INFO] sam_first_target_cam =", sam_first)
    print(f"[INFO] sam_time = {sam_sec:.4f} sec")
    print("[INFO] sam_overlay_path =", sam_overlay_path)
    print("[INFO] sam_target_overlay_path =", sam_target_overlay_path)

    # ========================================================
    # 3. 計算時間保存
    # ========================================================
    run_index = len(timing_runs) + 1

    timing_runs.append(
        {
            "run_index": int(run_index),
            "timestamp": ts,
            "depth_sec": float(depth_sec),
            "sam_sec": float(sam_sec),
            "exp_dir": str(exp_dir),
        }
    )

    write_timing_memo(
        timing_memo_path,
        runs=timing_runs,
    )

    save_timing_json(
        timing_json_path,
        runs=timing_runs,
    )

    print(f"[SAVE] timing memo: {timing_memo_path}")
    print(f"[SAVE] timing json: {timing_json_path}")

    # ========================================================
    # 4. 横並び比較画像の保存
    # ========================================================
    depth_img = read_image_or_blank(
        depth_target_overlay_path or depth_overlay_path,
        label="Depth",
    )

    sam_img = read_image_or_blank(
        sam_target_overlay_path or sam_overlay_path,
        label="SAM",
    )

    side_by_side = make_side_by_side(
        depth_img,
        sam_img,
        target_h=520,
    )

    side_by_side_path = exp_dir / "compare_depth_vs_sam.png"

    if save_side_by_side:
        cv2.imwrite(str(side_by_side_path), side_by_side)
        print(f"[SAVE] side-by-side comparison: {side_by_side_path}")

    # ========================================================
    # 5. 結果サマリ保存
    # ========================================================
    result = {
        "exp_dir": str(exp_dir),
        "depth_dir": str(depth_dir),
        "sam_dir": str(sam_dir),

        "timing_memo_path": str(timing_memo_path),
        "timing_json_path": str(timing_json_path),

        "depth": {
            "angle_rad": to_jsonable(depth_angle),
            "first_target_cam": to_jsonable(depth_first),
            "final_target_cam": to_jsonable(depth_final),
            "time_sec": float(depth_sec),
            "overlay_path": str(depth_overlay_path) if depth_overlay_path else None,
            "target_overlay_path": str(depth_target_overlay_path) if depth_target_overlay_path else None,
            "res": to_jsonable(depth_res),
        },

        "sam": {
            "angle_rad": to_jsonable(sam_angle),
            "first_target_cam": to_jsonable(sam_first),
            "final_target_cam": to_jsonable(sam_final),
            "time_sec": float(sam_sec),
            "overlay_path": str(sam_overlay_path) if sam_overlay_path else None,
            "target_overlay_path": str(sam_target_overlay_path) if sam_target_overlay_path else None,
            "res": to_jsonable(sam_res),
        },

        "side_by_side_path": str(side_by_side_path),
    }

    result_json_path = exp_dir / "result_summary.json"
    with result_json_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(result), f, ensure_ascii=False, indent=2)

    print("============================================================")
    print("[INFO] Compare experiment done")
    print(f"[INFO] result parent dir: {exp_dir}")
    print(f"[INFO] result summary: {result_json_path}")
    print("============================================================")

    return result


if __name__ == "__main__":
    run_compare_experiment(
        out_root="captures_compare_storage",
        width=1280,
        height=720,
        fps=6,
        save_side_by_side=True,
    )