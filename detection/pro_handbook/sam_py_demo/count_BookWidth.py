from pathlib import Path
import csv


# ===== 設定 =====
CSV_PATH = Path("/home/book/pro_book/pro_hand_book_python/captures/100test/book_width_eval_results_20260617_181313.csv")


def to_float(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() == "none":
        return None
    return float(x)


def main():
    over_2mm_rows = []

    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            status = row.get("status", "")

            gt = to_float(row.get("gt_book_width_mm"))
            pred = to_float(row.get("pred_book_width_mm"))

            if status != "success" or gt is None or pred is None:
                continue

            error = pred - gt
            abs_error = abs(error)

            # 2mmを超えるもの
            if abs_error > 2.0:
                over_2mm_rows.append({
                    "test_index": row.get("test_index"),
                    "book_name": row.get("book_name"),
                    "gt": gt,
                    "pred": pred,
                    "error": error,
                    "abs_error": abs_error,
                    "shot_dir": row.get("shot_dir"),
                })

    print("\n===== abs(error) > 2.0 mm のデータ =====")
    print(f"CSV: {CSV_PATH}")
    print(f"count: {len(over_2mm_rows)}")
    print("")

    for r in over_2mm_rows:
        print(
            f"test_index={r['test_index']}, "
            f"error={r['error']:.3f} mm, "
            f"abs_error={r['abs_error']:.3f} mm, "
            f"gt={r['gt']:.3f} mm, "
            f"pred={r['pred']:.3f} mm, "
            f"book_name={r['book_name']}, "
            f"shot_dir={r['shot_dir']}"
        )

    print("========================================\n")


if __name__ == "__main__":
    main()