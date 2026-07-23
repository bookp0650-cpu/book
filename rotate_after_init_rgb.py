#!/usr/bin/env python3
from pathlib import Path
from PIL import Image

# 元データの親ディレクトリ
SRC_ROOT = Path.home() / "pro_book/pro_hand_book_python/captures/100test"

# Box マウント先の保存ディレクトリ
DST_ROOT = Path.home() / (
    "box/Arai-lab-Box/99-Experiment-PC-Sync/"
    "pro_BOOK（出版物流）_2025/開発部/SAM3_アノテーションデータ"
)

# True にすると、既存の 1.png などを上書きする
OVERWRITE = False


def main():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"元ディレクトリが見つかりません: {SRC_ROOT}")

    DST_ROOT.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skipped_count = 0
    missing_count = 0
    failed_count = 0

    for number in range(1, 101):
        src_path = SRC_ROOT / str(number) / "after_init_rgb.png"
        dst_path = DST_ROOT / f"{number}.png"

        if not src_path.exists():
            print(f"[不足] {src_path}")
            missing_count += 1
            continue

        if dst_path.exists() and not OVERWRITE:
            print(f"[スキップ] 既に存在: {dst_path.name}")
            skipped_count += 1
            continue

        try:
            with Image.open(src_path) as image:
                rotated_image = image.rotate(180)
                rotated_image.save(dst_path, format="PNG")

            print(f"[保存] {src_path.parent.name}/after_init_rgb.png -> {dst_path}")
            success_count += 1

        except Exception as e:
            print(f"[失敗] {src_path}: {e}")
            failed_count += 1

    print("\n===== 完了 =====")
    print(f"保存成功: {success_count}")
    print(f"スキップ: {skipped_count}")
    print(f"不足:     {missing_count}")
    print(f"失敗:     {failed_count}")
    print(f"保存先:   {DST_ROOT}")


if __name__ == "__main__":
    main()