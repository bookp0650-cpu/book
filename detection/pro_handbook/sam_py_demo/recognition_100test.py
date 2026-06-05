#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
100冊認識精度検証用スクリプト．

使い方:
  cd /home/book/pro_book/pro_hand_book_python

  python -m detection.pro_handbook.sam_py_demo.recognition_100test \
    --master-json master_20260216.json \
    --out-dir captures/100test \
    --repeat-per-book 5 \
    --order block \
    --overwrite

仕様:
  - master_20260216.json から book_name / ISBN_number / bookshelf_ID を読み込む．
  - 20冊 × 各5回 = 100回の試験計画を作る．
  - 各試行ごとに Enter 待ち → RealSense撮影 → 認識 → 最終overlay表示 → Enterで次へ進む．
  - 保存先は captures/100test/1, captures/100test/2, ... とする．
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# python -m でも，python recognition_100test.py でも動きやすいようにする．
try:
    from .get_book_points import run_capture_and_pca
except ImportError:
    from get_book_points import run_capture_and_pca


def find_project_root() -> Path:
    """
    pro_hand_book_python 直下を想定したプロジェクトルートを推定する．
    recognition_100test.py がどこから実行されても master_20260216.json を探せるようにする．
    """
    here = Path(__file__).resolve()

    for p in [Path.cwd().resolve(), *here.parents]:
        if (p / "master_20260216.json").exists() and (p / "detection").exists():
            return p

    # 最後のフォールバック
    return Path.cwd().resolve()


def resolve_path(path_like: str | Path, *, base_dir: Path) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_master_books(master_json: str | Path, *, project_root: Path) -> list[dict[str, Any]]:
    """
    master_20260216.json を読み込み，book_name / ISBN_number / bookshelf_ID を持つ辞書リストを返す．
    """
    path = resolve_path(master_json, base_dir=project_root)

    if not path.exists():
        raise FileNotFoundError(f"master json not found: {path}")

    obj = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(obj, list):
        raise ValueError("master json must be a list of book records")

    books: list[dict[str, Any]] = []
    for i, item in enumerate(obj, start=1):
        if not isinstance(item, dict):
            continue

        book_name = str(item.get("book_name", "")).strip()
        if not book_name:
            continue

        books.append(
            {
                "master_index": i,
                "book_name": book_name,
                "ISBN_number": str(item.get("ISBN_number", "")).strip(),
                "bookshelf_ID": str(item.get("bookshelf_ID", "")).strip(),
            }
        )

    if not books:
        raise ValueError(f"no valid book_name found in: {path}")

    return books


def build_test_plan(
    books: list[dict[str, Any]],
    *,
    repeat_per_book: int = 5,
    order: str = "block",
) -> list[dict[str, Any]]:
    """
    20冊 × 各5回 = 100回 の試験計画を作る．

    order="block":
      book1を5回，book2を5回，... の順．
      本の入れ替え回数を減らしたい場合に便利．

    order="cycle":
      20冊を1周して，それを5セット繰り返す．
      セットごとのばらつきを見たい場合に便利．
    """
    repeat_per_book = int(repeat_per_book)
    if repeat_per_book <= 0:
        raise ValueError("repeat_per_book must be positive")

    order = str(order).lower()
    plan: list[dict[str, Any]] = []

    if order == "block":
        for book in books:
            for rep in range(1, repeat_per_book + 1):
                rec = dict(book)
                rec["repeat_index"] = rep
                plan.append(rec)

    elif order == "cycle":
        for rep in range(1, repeat_per_book + 1):
            for book in books:
                rec = dict(book)
                rec["repeat_index"] = rep
                plan.append(rec)

    else:
        raise ValueError("order must be 'block' or 'cycle'")

    for test_index, rec in enumerate(plan, start=1):
        rec["test_index"] = test_index

    return plan


def move_result_dir(src_dir: Path, dst_dir: Path, *, overwrite: bool = False) -> Path:
    """
    run_capture_and_pca() が作成した時刻フォルダを captures/100test/n に移動する．
    """
    src_dir = Path(src_dir).resolve()
    dst_dir = Path(dst_dir).resolve()

    if not src_dir.exists():
        raise FileNotFoundError(src_dir)

    if src_dir == dst_dir:
        return dst_dir

    if dst_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{dst_dir} already exists. Use --overwrite to replace it.")
        shutil.rmtree(dst_dir)

    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_dir), str(dst_dir))
    return dst_dir


def find_latest_new_dir(parent: Path, before: set[Path]) -> Path | None:
    """
    parent直下で，beforeに無かった最新ディレクトリを返す．
    """
    parent = Path(parent)
    if not parent.exists():
        return None

    after = {p for p in parent.iterdir() if p.is_dir()}
    new_dirs = sorted(after - before, key=lambda p: p.stat().st_mtime)
    return new_dirs[-1] if new_dirs else None


def find_final_overlay(shot_dir: Path) -> Path | None:
    """
    mask番号は毎回変わるため，mask*_final_valid_depth_region_overlay.png を探す．
    """
    shot_dir = Path(shot_dir)

    candidates = sorted(
        [p for p in shot_dir.glob("mask*_final_valid_depth_region_overlay.png") if "_offline_" not in p.name],
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    candidates = sorted(
        list(shot_dir.glob("mask*_offline_final_valid_depth_region_overlay.png")),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    return None


def show_overlay_image(image_path: Path, *, window_name: str) -> None:
    """
    最終overlay画像をOpenCVで表示する．
    Enter / Space / Esc / q のいずれかで次へ進む．
    """
    image_path = Path(image_path)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if img is None:
        print(f"⚠ overlay画像を読み込めませんでした: {image_path}")
        input("Enterで次へ進みます．")
        return

    # 画像自体が黒いかどうかも確認
    print(f"[DISPLAY] image: {image_path}")
    print(f"[DISPLAY] shape={img.shape}, min={int(img.min())}, max={int(img.max())}, mean={float(img.mean()):.2f}")

    max_w = 1400
    max_h = 900
    h, w = img.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)

    if scale < 1.0:
        img_show = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_show = img

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img_show)

    print("画像ウィンドウを選択して，Enter / Space / Esc / q のいずれかを押すと次へ進みます．")

    while True:
        key = cv2.waitKey(50) & 0xFF

        # Enter: 13, Space: 32, Esc: 27, q: 113
        if key in (13, 10, 32, 27, ord("q")):
            break

        # ウィンドウを閉じた場合も次へ進む
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_summary_csv(summary_csv: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "test_index",
        "master_index",
        "repeat_index",
        "status",
        "book_name",
        "ISBN_number",
        "bookshelf_ID",
        "shot_dir",
        "elapsed_sec",
        "roll_rad",
        "roll_deg",
        "p_xmax",
        "book_width_mm",
        "overlay_path",
        "error",
    ]

    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_csv.exists()

    with summary_csv.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def save_meta(
    shot_dir: Path,
    *,
    rec: dict[str, Any],
    status: str,
    elapsed_sec: float,
    roll: Any,
    p_xmax: Any,
    book_width: Any,
    overlay_path: Path | None,
    error: str | None = None,
) -> None:
    meta = {
        "test_index": int(rec["test_index"]),
        "master_index": int(rec["master_index"]),
        "repeat_index": int(rec["repeat_index"]),
        "book_name": rec["book_name"],
        "ISBN_number": rec["ISBN_number"],
        "bookshelf_ID": rec["bookshelf_ID"],
        "status": status,
        "elapsed_sec": float(elapsed_sec),
        "roll_rad": None if roll is None else float(roll),
        "roll_deg": None if roll is None else float(np.degrees(float(roll))),
        "p_xmax": None if p_xmax is None else np.asarray(p_xmax).reshape(-1).astype(float).tolist(),
        "book_width_mm": None if book_width is None else float(book_width),
        "overlay_path": None if overlay_path is None else str(overlay_path),
        "error": error,
    }
    write_json(Path(shot_dir) / "100test_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--master-json", type=str, default="master_20260216.json")
    parser.add_argument("--out-dir", type=str, default="captures/100test")
    parser.add_argument("--repeat-per-book", type=int, default=5)
    parser.add_argument("--order", choices=["block", "cycle"], default="block")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-show", action="store_true")

    parser.add_argument("--sam-device", type=str, default="gpu")
    parser.add_argument("--sam-pts-x", type=int, default=32)
    parser.add_argument("--sam-pts-y", type=int, default=8)
    parser.add_argument("--sam-k-keep", type=int, default=1)
    parser.add_argument("--sam-target-len", type=int, default=768)

    args = parser.parse_args()

    project_root = find_project_root()
    base_dir = resolve_path(args.out_dir, base_dir=project_root)
    tmp_dir = base_dir / "_tmp_runs"
    summary_csv = base_dir / "summary.csv"

    base_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    books = load_master_books(args.master_json, project_root=project_root)
    plan = build_test_plan(
        books,
        repeat_per_book=args.repeat_per_book,
        order=args.order,
    )

    if args.count is not None:
        end_index = min(len(plan), int(args.start) + int(args.count) - 1)
    else:
        end_index = len(plan)

    if args.start < 1 or args.start > len(plan):
        raise ValueError(f"--start must be within 1..{len(plan)}")

    selected_plan = plan[int(args.start) - 1:end_index]

    print("===== 100 books recognition test =====")
    print(f"project root       : {project_root}")
    print(f"master json        : {resolve_path(args.master_json, base_dir=project_root)}")
    print(f"books              : {len(books)}")
    print(f"repeat per book    : {args.repeat_per_book}")
    print(f"total plan         : {len(plan)}")
    print(f"run range          : {args.start} - {end_index}")
    print(f"order              : {args.order}")
    print(f"save dir           : {base_dir}")
    print(f"tmp dir            : {tmp_dir}")
    print("Ctrl+C で中断できます．")
    print("======================================")

    try:
        for rec in selected_plan:
            i = int(rec["test_index"])
            dst_dir = base_dir / str(i)

            print(f"\n========== TEST {i}/{len(plan)} ==========")
            print(f"book_name    : {rec['book_name']}")
            print(f"ISBN         : {rec['ISBN_number']}")
            print(f"bookshelf_ID : {rec['bookshelf_ID']}")
            print(f"repeat       : {rec['repeat_index']}/{args.repeat_per_book}")
            print(f"save_to      : {dst_dir}")

            if dst_dir.exists() and not args.overwrite:
                print("既に保存ディレクトリが存在するためスキップします．")
                print("--overwrite を付けると上書きできます．")
                continue

            input("対象本をセットしたら Enter を押してください．撮影・認識を開始します．")

            before_dirs = {p for p in tmp_dir.iterdir() if p.is_dir()}
            t0 = time.perf_counter()

            status = "unknown"
            error = ""
            roll = None
            p_xmax = None
            book_width = None
            returned_shot_dir = None
            final_dir = dst_dir
            overlay_path = None

            try:
                roll, p_xmax, book_width, returned_shot_dir = run_capture_and_pca(
                    query=rec["book_name"],
                    out_dir=tmp_dir,
                    sam_device=args.sam_device,
                    sam_pts_side=(args.sam_pts_x, args.sam_pts_y),
                    sam_decoder_k_keep=args.sam_k_keep,
                    sam_target_len=args.sam_target_len,
                )

                elapsed = time.perf_counter() - t0

                if returned_shot_dir is not None and Path(returned_shot_dir).exists():
                    src_dir = Path(returned_shot_dir)
                else:
                    src_dir = find_latest_new_dir(tmp_dir, before_dirs)

                if src_dir is not None and Path(src_dir).exists():
                    final_dir = move_result_dir(src_dir, dst_dir, overwrite=args.overwrite)
                else:
                    final_dir = dst_dir
                    final_dir.mkdir(parents=True, exist_ok=True)

                overlay_path = find_final_overlay(final_dir)

                if roll is None or p_xmax is None or book_width is None:
                    status = "recognition_fail"
                    error = "run_capture_and_pca returned None"
                else:
                    status = "success"

                save_meta(
                    final_dir,
                    rec=rec,
                    status=status,
                    elapsed_sec=elapsed,
                    roll=roll,
                    p_xmax=p_xmax,
                    book_width=book_width,
                    overlay_path=overlay_path,
                    error=error if error else None,
                )

                append_summary_csv(
                    summary_csv,
                    {
                        "test_index": i,
                        "master_index": rec["master_index"],
                        "repeat_index": rec["repeat_index"],
                        "status": status,
                        "book_name": rec["book_name"],
                        "ISBN_number": rec["ISBN_number"],
                        "bookshelf_ID": rec["bookshelf_ID"],
                        "shot_dir": str(final_dir),
                        "elapsed_sec": f"{elapsed:.6f}",
                        "roll_rad": "" if roll is None else f"{float(roll):.9f}",
                        "roll_deg": "" if roll is None else f"{float(np.degrees(float(roll))):.6f}",
                        "p_xmax": "" if p_xmax is None else np.asarray(p_xmax).reshape(-1).astype(float).tolist(),
                        "book_width_mm": "" if book_width is None else f"{float(book_width):.6f}",
                        "overlay_path": "" if overlay_path is None else str(overlay_path),
                        "error": error,
                    },
                )

                print("\n===== TEST RESULT =====")
                print(f"status        : {status}")
                print(f"elapsed [sec] : {elapsed:.3f}")
                print(f"shot_dir      : {final_dir}")
                print(f"roll          : {roll}")
                print(f"p_xmax        : {p_xmax}")
                print(f"book_width    : {book_width}")
                print("=======================")

                if overlay_path is not None and not args.no_show:
                    show_overlay_image(
                        overlay_path,
                        window_name=f"100test {i}: {rec['book_name']}",
                    )
                else:
                    print("⚠ 最終overlay画像が見つからない，または --no-show が指定されています．")
                    input("Enterで次へ進みます．")

            except KeyboardInterrupt:
                raise

            except Exception as e:
                elapsed = time.perf_counter() - t0
                error = traceback.format_exc()
                print("❌ TEST failed")
                print(error)

                src_dir = find_latest_new_dir(tmp_dir, before_dirs)
                if src_dir is not None and Path(src_dir).exists():
                    try:
                        final_dir = move_result_dir(src_dir, dst_dir, overwrite=args.overwrite)
                    except Exception:
                        final_dir = src_dir
                else:
                    final_dir = dst_dir
                    final_dir.mkdir(parents=True, exist_ok=True)

                save_meta(
                    final_dir,
                    rec=rec,
                    status="exception",
                    elapsed_sec=elapsed,
                    roll=None,
                    p_xmax=None,
                    book_width=None,
                    overlay_path=None,
                    error=error,
                )

                append_summary_csv(
                    summary_csv,
                    {
                        "test_index": i,
                        "master_index": rec["master_index"],
                        "repeat_index": rec["repeat_index"],
                        "status": "exception",
                        "book_name": rec["book_name"],
                        "ISBN_number": rec["ISBN_number"],
                        "bookshelf_ID": rec["bookshelf_ID"],
                        "shot_dir": str(final_dir),
                        "elapsed_sec": f"{elapsed:.6f}",
                        "error": str(e),
                    },
                )

                input("エラー内容を確認したら Enter を押してください．次の撮影に進みます．")

    except KeyboardInterrupt:
        print("\nCtrl+C detected. 中断します．")

    finally:
        cv2.destroyAllWindows()
        print("\n===== DONE =====")
        print(f"summary: {summary_csv}")


if __name__ == "__main__":
    main()
