from .get_book_points import run_capture_and_pca_offline
from pathlib import Path
import json
import csv
import time
import traceback
from datetime import datetime
import shutil
import sys
from contextlib import redirect_stdout, redirect_stderr


# =========================
# 設定
# =========================
BASE_DIR = Path("/home/book/pro_book/pro_hand_book_python").resolve()

# 元の入力データ
TEST_BASE_DIR = BASE_DIR / "captures" / "100test"

# offline実行結果の保存先
OFFLINE_BASE_DIR = BASE_DIR / "captures" / "100test_offline"

MASTER_JSON = BASE_DIR / "master_20260216.json"

SAM_DEVICE = "gpu"

START_INDEX = 1
END_INDEX = 100
REPEATS_PER_BOOK = 5

# 評価しきい値 [mm]
ERROR_THRESHOLDS_MM = [1.0, 1.5, 2.0]

# offline入力としてコピーするファイル
# run_capture_and_pca_offline はこの2つを shot_dir から読む想定
INPUT_FILES = [
    "after_init_rgb.png",
    "after_init_depth.npy",
]


class Tee:
    """stdout/stderrをターミナルとファイルの両方へ出すための簡易Tee。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def make_unique_run_dir(base_dir: Path, timestamp: str) -> Path:
    """
    captures/100test_offline/<timestamp> を作る。
    同じ秒に複数回実行して重複した場合は _001, _002 ... を付ける。
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    run_dir = base_dir / timestamp
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        return run_dir

    for i in range(1, 1000):
        candidate = base_dir / f"{timestamp}_{i:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate

    raise RuntimeError(f"unique run dir を作れませんでした: {base_dir}/{timestamp}_XXX")


def load_master(master_json: Path):
    with open(master_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    books = []
    for i, item in enumerate(data):
        try:
            book_name = item["book_name"]
            book_width_mm = float(item["book_width"])
        except Exception as e:
            raise ValueError(f"master json の {i} 番目が不正です: {item}") from e

        books.append({
            "book_name": book_name,
            "book_width_mm": book_width_mm,
            "raw": item,
        })

    return books


def get_book_info_for_test_index(master_books, test_index: int):
    """
    1〜5     -> master[0]
    6〜10    -> master[1]
    ...
    96〜100  -> master[19]
    """
    master_index = (test_index - 1) // REPEATS_PER_BOOK
    repeat_index = (test_index - 1) % REPEATS_PER_BOOK + 1

    if master_index < 0 or master_index >= len(master_books):
        raise IndexError(
            f"test_index={test_index} に対応する master_index={master_index} が存在しません。"
        )

    return master_index, repeat_index, master_books[master_index]


def safe_float_or_none(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def copy_offline_inputs(source_shot_dir: Path, run_shot_dir: Path):
    """
    元データ captures/100test/<idx> から、
    今回の実行先 captures/100test_offline/<timestamp>/<idx> に
    offline実行に必要な入力だけをコピーする。

    debug画像や過去のfinal.pngはコピーしない。
    """
    source_shot_dir = Path(source_shot_dir)
    run_shot_dir = Path(run_shot_dir)
    run_shot_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []

    for name in INPUT_FILES:
        src = source_shot_dir / name
        dst = run_shot_dir / name

        if not src.exists():
            missing.append(str(src))
            continue

        shutil.copy2(src, dst)
        copied.append({
            "name": name,
            "source": str(src),
            "destination": str(dst),
            "size_bytes": int(dst.stat().st_size),
        })

    if missing:
        raise FileNotFoundError(
            "offline入力ファイルが不足しています:\n" + "\n".join(missing)
        )

    manifest = {
        "source_shot_dir": str(source_shot_dir),
        "run_shot_dir": str(run_shot_dir),
        "input_files": copied,
        "copy_policy": "copy only after_init_rgb.png and after_init_depth.npy; do not copy previous debug/final outputs",
    }

    save_json(run_shot_dir / "offline_input_manifest.json", manifest)
    return manifest


def save_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    print("\n===== BOOK WIDTH OFFLINE EVAL START =====")
    print(f"source test dir : {TEST_BASE_DIR}")
    print(f"offline base dir: {OFFLINE_BASE_DIR}")
    print(f"master json     : {MASTER_JSON}")
    print(f"sam_device      : {SAM_DEVICE}")
    print(f"test range      : {START_INDEX} to {END_INDEX}")
    print("=========================================\n")

    master_books = load_master(MASTER_JSON)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root_dir = make_unique_run_dir(OFFLINE_BASE_DIR, timestamp)

    out_csv = run_root_dir / f"book_width_eval_results_{run_root_dir.name}.csv"
    out_json = run_root_dir / f"book_width_eval_results_{run_root_dir.name}.json"
    out_summary = run_root_dir / f"book_width_eval_summary_{run_root_dir.name}.json"
    run_config_json = run_root_dir / "eval_run_config.json"

    run_config = {
        "timestamp": timestamp,
        "run_root_dir": str(run_root_dir),
        "source_test_base_dir": str(TEST_BASE_DIR),
        "offline_base_dir": str(OFFLINE_BASE_DIR),
        "master_json": str(MASTER_JSON),
        "sam_device": SAM_DEVICE,
        "start_index": START_INDEX,
        "end_index": END_INDEX,
        "repeats_per_book": REPEATS_PER_BOOK,
        "error_thresholds_mm": ERROR_THRESHOLDS_MM,
        "input_files": INPUT_FILES,
        "output_structure": "captures/100test_offline/<timestamp>/<test_index>/",
    }
    save_json(run_config_json, run_config)

    print(f"今回のoffline実行保存先: {run_root_dir}")
    print(f"設定JSON: {run_config_json}")

    results = []
    total_trials = END_INDEX - START_INDEX + 1

    for test_index in range(START_INDEX, END_INDEX + 1):
        source_shot_dir = TEST_BASE_DIR / str(test_index)
        run_shot_dir = run_root_dir / str(test_index)
        run_shot_dir.mkdir(parents=True, exist_ok=True)

        case_log_path = run_shot_dir / "offline_run_console.log"

        # デフォルト値。例外時に未定義にならないようにする。
        master_index = None
        repeat_index = None
        book_name = ""
        gt_width_mm = None
        pred_width_mm = None
        abs_error_mm = None
        returned_shot_dir = None
        elapsed = 0.0
        row = None

        with open(case_log_path, "w", encoding="utf-8") as log_f:
            tee_out = Tee(sys.stdout, log_f)
            tee_err = Tee(sys.stderr, log_f)

            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                start = time.perf_counter()

                try:
                    master_index, repeat_index, book_info = get_book_info_for_test_index(
                        master_books,
                        test_index,
                    )

                    book_name = book_info["book_name"]
                    gt_width_mm = book_info["book_width_mm"]

                    print("\n" + "=" * 60)
                    print(f"test_index       : {test_index}")
                    print(f"master_index     : {master_index}")
                    print(f"repeat_index     : {repeat_index}")
                    print(f"book_name        : {book_name}")
                    print(f"gt_width_mm      : {gt_width_mm}")
                    print(f"source_shot_dir  : {source_shot_dir}")
                    print(f"run_shot_dir     : {run_shot_dir}")
                    print(f"case_console_log : {case_log_path}")
                    print("=" * 60)

                    if not source_shot_dir.exists():
                        raise FileNotFoundError(f"source_shot_dir が存在しません: {source_shot_dir}")

                    # 重要: 元の100testを直接使わず、timestamp配下へ入力だけコピーして実行する
                    input_manifest = copy_offline_inputs(
                        source_shot_dir=source_shot_dir,
                        run_shot_dir=run_shot_dir,
                    )
                    print("✔ copied offline inputs")
                    print(json.dumps(input_manifest, ensure_ascii=False, indent=2))

                    theta_rad, p_min, pred_width_mm, returned_shot_dir = run_capture_and_pca_offline(
                        query=book_name,
                        shot_dir=run_shot_dir.resolve(),
                        sam_device=SAM_DEVICE,
                    )

                    elapsed = time.perf_counter() - start
                    pred_width_mm = safe_float_or_none(pred_width_mm)

                    if pred_width_mm is None:
                        raise ValueError("pred_width_mm が None または float 変換不可です。")

                    abs_error_mm = abs(pred_width_mm - gt_width_mm)

                    row = {
                        "test_index": test_index,
                        "master_index": master_index,
                        "repeat_index": repeat_index,
                        "book_name": book_name,
                        "gt_book_width_mm": gt_width_mm,
                        "pred_book_width_mm": pred_width_mm,
                        "abs_error_mm": abs_error_mm,
                        "elapsed_sec": elapsed,
                        "status": "success",
                        "error": "",
                        "source_shot_dir": str(source_shot_dir),
                        "run_shot_dir": str(run_shot_dir),
                        "returned_shot_dir": str(returned_shot_dir),
                        "case_console_log": str(case_log_path),
                    }

                    print("\n--- RESULT ---")
                    print(f"pred_width_mm : {pred_width_mm:.3f}")
                    print(f"gt_width_mm   : {gt_width_mm:.3f}")
                    print(f"abs_error_mm  : {abs_error_mm:.3f}")
                    print(f"elapsed_sec   : {elapsed:.3f}")
                    print(f"outputs       : {run_shot_dir}")

                except Exception as e:
                    elapsed = time.perf_counter() - start

                    # book_info が取れる前に落ちる可能性もあるので保険
                    if not book_name:
                        try:
                            master_index, repeat_index, book_info = get_book_info_for_test_index(
                                master_books,
                                test_index,
                            )
                            book_name = book_info["book_name"]
                            gt_width_mm = book_info["book_width_mm"]
                        except Exception:
                            book_name = ""
                            gt_width_mm = None

                    row = {
                        "test_index": test_index,
                        "master_index": master_index,
                        "repeat_index": repeat_index,
                        "book_name": book_name,
                        "gt_book_width_mm": gt_width_mm,
                        "pred_book_width_mm": None,
                        "abs_error_mm": None,
                        "elapsed_sec": elapsed,
                        "status": "fail",
                        "error": str(e),
                        "source_shot_dir": str(source_shot_dir),
                        "run_shot_dir": str(run_shot_dir),
                        "returned_shot_dir": None,
                        "case_console_log": str(case_log_path),
                    }

                    print("\n❌ FAILED")
                    print(f"test_index      : {test_index}")
                    print(f"book_name       : {book_name}")
                    print(f"source_shot_dir : {source_shot_dir}")
                    print(f"run_shot_dir    : {run_shot_dir}")
                    print(f"error           : {e}")
                    traceback.print_exc()

                finally:
                    if row is not None:
                        save_json(run_shot_dir / "case_result.json", row)

        results.append(row)

        # 途中で落ちても結果が残るように毎回保存
        save_results(results, out_csv, out_json)

    summary = make_summary(results, total_trials)
    save_summary(summary, out_summary)

    print_summary(summary)

    print("\n保存先:")
    print(f"RUN ROOT: {run_root_dir}")
    print(f"CSV     : {out_csv}")
    print(f"JSON    : {out_json}")
    print(f"SUMMARY : {out_summary}")
    print("=========================================\n")


def make_summary(results, total_trials: int):
    success_results = [r for r in results if r["status"] == "success"]
    fail_results = [r for r in results if r["status"] != "success"]

    errors = [
        r["abs_error_mm"]
        for r in success_results
        if r["abs_error_mm"] is not None
    ]

    threshold_counts = {}
    for th in ERROR_THRESHOLDS_MM:
        count = sum(1 for e in errors if 0.0 <= e <= th)
        threshold_counts[f"within_{th:.1f}mm"] = {
            "count": count,
            "denominator": total_trials,
            "rate": count / total_trials if total_trials > 0 else None,
        }

    if errors:
        mean_abs_error = sum(errors) / len(errors)
        max_abs_error = max(errors)
        min_abs_error = min(errors)
    else:
        mean_abs_error = None
        max_abs_error = None
        min_abs_error = None

    summary = {
        "total_trials": total_trials,
        "success_count": len(success_results),
        "fail_count": len(fail_results),
        "threshold_counts": threshold_counts,
        "mean_abs_error_mm_success_only": mean_abs_error,
        "min_abs_error_mm_success_only": min_abs_error,
        "max_abs_error_mm_success_only": max_abs_error,
        "results": results,
    }

    return summary


def save_results(results, out_csv: Path, out_json: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "test_index",
        "master_index",
        "repeat_index",
        "book_name",
        "gt_book_width_mm",
        "pred_book_width_mm",
        "abs_error_mm",
        "elapsed_sec",
        "status",
        "error",
        "source_shot_dir",
        "run_shot_dir",
        "returned_shot_dir",
        "case_console_log",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def save_summary(summary, out_summary: Path):
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def print_summary(summary):
    print("\n\n===== BOOK WIDTH OFFLINE EVAL SUMMARY =====")

    total_trials = summary["total_trials"]

    print(f"総試行回数 : {total_trials}")
    print(f"成功回数   : {summary['success_count']} / {total_trials}")
    print(f"失敗回数   : {summary['fail_count']} / {total_trials}")
    print("")

    for th in ERROR_THRESHOLDS_MM:
        key = f"within_{th:.1f}mm"
        item = summary["threshold_counts"][key]
        count = item["count"]
        denom = item["denominator"]
        rate = item["rate"]

        if rate is None:
            rate_text = "None"
        else:
            rate_text = f"{rate * 100:.1f}%"

        print(f"0〜{th:.1f}mm以内: {count} / {denom} ({rate_text})")

    print("")

    mean_abs_error = summary["mean_abs_error_mm_success_only"]
    min_abs_error = summary["min_abs_error_mm_success_only"]
    max_abs_error = summary["max_abs_error_mm_success_only"]

    if mean_abs_error is not None:
        print(f"平均絶対誤差[成功のみ] : {mean_abs_error:.3f} mm")
        print(f"最小絶対誤差[成功のみ] : {min_abs_error:.3f} mm")
        print(f"最大絶対誤差[成功のみ] : {max_abs_error:.3f} mm")
    else:
        print("平均絶対誤差[成功のみ] : None")

    print("===========================================\n")


if __name__ == "__main__":
    main()