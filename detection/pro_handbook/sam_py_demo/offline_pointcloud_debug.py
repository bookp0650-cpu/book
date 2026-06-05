from .get_book_points import run_capture_and_pca_offline
from pathlib import Path
import numpy as np
import time
import traceback


QUERY = "WMS導入と運用のための99の極意[第2版]"

# 必ず絶対パスにする
SHOT_DIR = Path("/home/book/pro_book/pro_hand_book_python/captures/20260409_165212").resolve()

SAM_DEVICE = "gpu"
NUM_RUNS = 11


def main():
    run_times = []
    results = []

    print("\n===== OFFLINE BENCHMARK START =====")
    print(f"query     : {QUERY}")
    print(f"shot_dir  : {SHOT_DIR}")
    print(f"sam_device: {SAM_DEVICE}")
    print(f"runs      : {NUM_RUNS}")
    print("===================================\n")

    for i in range(NUM_RUNS):
        print(f"\n========== RUN {i + 1}/{NUM_RUNS} ==========")

        start = time.perf_counter()

        try:
            theta_rad, p_min, book_width, shot_dir = run_capture_and_pca_offline(
                query=QUERY,
                shot_dir=SHOT_DIR,
                sam_device=SAM_DEVICE,
            )

            elapsed = time.perf_counter() - start
            run_times.append(elapsed)

            results.append({
                "success": True,
                "elapsed": elapsed,
                "theta_rad": theta_rad,
                "p_min": p_min,
                "book_width": book_width,
                "shot_dir": shot_dir,
            })

            print("\n=== OFFLINE Summary ===")
            print(f"run time [sec]  = {elapsed:.3f}")
            print(f"book width [mm] = {book_width}")

            if theta_rad is not None:
                print(f"roll (deg)      = {np.degrees(theta_rad):.6f}")
            else:
                print("roll (deg)      = None")

            print(f"p_min           = {p_min}")
            print(f"shot_dir        = {shot_dir}")

        except Exception as e:
            elapsed = time.perf_counter() - start
            run_times.append(elapsed)

            results.append({
                "success": False,
                "elapsed": elapsed,
                "error": str(e),
            })

            print(f"❌ RUN {i + 1} failed")
            print(f"run time [sec] = {elapsed:.3f}")
            traceback.print_exc()

    print("\n\n===== OFFLINE BENCHMARK RESULT =====")

    first_time = run_times[0]
    rest_times = run_times[1:]

    print(f"1回目の実行時間 [sec]        : {first_time:.3f}")

    if rest_times:
        print(f"2〜{NUM_RUNS}回目の平均 [sec] : {float(np.mean(rest_times)):.3f}")
        print(f"2〜{NUM_RUNS}回目の標準偏差   : {float(np.std(rest_times)):.3f}")
        print(f"2〜{NUM_RUNS}回目の最小 [sec] : {float(np.min(rest_times)):.3f}")
        print(f"2〜{NUM_RUNS}回目の最大 [sec] : {float(np.max(rest_times)):.3f}")

    success_count = sum(1 for r in results if r["success"])
    fail_count = NUM_RUNS - success_count

    print(f"成功回数                    : {success_count}/{NUM_RUNS}")
    print(f"失敗回数                    : {fail_count}/{NUM_RUNS}")
    print("====================================\n")

    print("各回の実行時間:")
    for i, t in enumerate(run_times):
        status = "OK" if results[i]["success"] else "FAIL"
        print(f"  RUN {i + 1:02d}: {t:.3f} sec [{status}]")


if __name__ == "__main__":
    main()