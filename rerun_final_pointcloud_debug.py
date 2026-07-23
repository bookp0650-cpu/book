from pathlib import Path
import json

from detection.pro_handbook.sam_py_demo.get_book_points import run_capture_and_pca_offline


def load_query_from_meta(shot_dir: Path) -> str:
    meta_path = shot_dir / "100test_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} がありません．queryを手動指定してください．"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # 100test_meta.json のキー名が多少違っても拾えるようにする
    candidates = [
        "book_name",
        "query",
        "title",
        "target_book_name",
        "book_title",
    ]

    for key in candidates:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise KeyError(
        f"{meta_path} から対象書籍名を取得できませんでした．"
        f"中身を確認して，queryを手動指定してください．"
    )


def main():
    shot_dir = Path(
        "/home/book/pro_book/pro_hand_book_python/captures/100test/97"
    )

    query = load_query_from_meta(shot_dir)
    print(f"[INFO] shot_dir = {shot_dir}")
    print(f"[INFO] query    = {query}")

    theta_rad, target_point, book_width_mm, out_shot_dir = run_capture_and_pca_offline(
        query=query,
        shot_dir=shot_dir,
        sam_device="gpu",
        interactive=True,
        use_persistent_runtime=True,
        show_pointcloud_gui=False,
        save_pointcloud_debug=True,
    )

    print("===== DONE =====")
    print(f"out_shot_dir   : {out_shot_dir}")
    print(f"theta_rad      : {theta_rad}")
    print(f"target_point   : {target_point}")
    print(f"book_width_mm  : {book_width_mm}")


if __name__ == "__main__":
    main()
