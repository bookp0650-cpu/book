import os, glob
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import traceback
import re

def find_video_devices_by_name(keyword="Depstech") -> list[str]:
    key = keyword.lower()
    devs = []
    for v in sorted(glob.glob("/sys/class/video4linux/video*")):
        name_path = os.path.join(v, "name")
        try:
            name = open(name_path, "r").read().strip()
        except Exception:
            continue
        if key in name.lower():
            devs.append("/dev/" + os.path.basename(v))
    if not devs:
        raise RuntimeError(f'keyword "{keyword}" を含む video デバイスが見つかりません')
    return devs

def find_first_openable_video_device(
    keyword="Depstech",
):
    """
    keyword を含む /dev/video* を列挙し、
    openできた最初のデバイスとVideoCaptureを返す。

    ここでopenしたcapを撮影までそのまま使うことで、
    カメラの二重openを防ぐ。
    """

    for dev in find_video_devices_by_name(keyword):

        match = re.fullmatch(
            r"/dev/video(\d+)",
            str(dev),
        )

        if match is None:
            continue

        device_index = int(
            match.group(1)
        )

        cap = cv2.VideoCapture(
            device_index,
            cv2.CAP_V4L2,
        )

        if cap.isOpened():
            return dev, device_index, cap

        cap.release()

    raise RuntimeError(
        f"{keyword} は見つかったが、"
        "どの /dev/video* も open できませんでした"
    )


def capture_one_depstech(
    save_path: Path,
    width: int = 3840,
    height: int = 2160,
    num_frames: int = 10,
):
    """
    Depstechカメラを自動検出して1枚撮影する。

    find_first_openable_video_device()が
    "/dev/video6"のような文字列を返す場合でも、
    OpenCVには数値インデックス6を渡す。
    """

    save_path = Path(save_path)

    # 例: "/dev/video6"
    dev, device_index, cap = (
        find_first_openable_video_device(
            "Depstech"
        )
    )

    print(
        f"[Depstech] opened {dev} "
        f"(index={device_index})"
    )

    try:
        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # 実際に設定された値を確認
        actual_width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        actual_height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        print(
            f"[Depstech] requested: "
            f"{width}x{height} @ 30 fps"
        )
        print(
            f"[Depstech] actual: "
            f"{actual_width}x{actual_height} "
            f"@ {actual_fps:.1f} fps"
        )

        ts = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        save_path_ts = save_path.with_name(
            f"{save_path.stem}_{ts}"
            f"{save_path.suffix}"
        )

        save_path_ts.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 自動露光・ホワイトバランス安定化のため捨てる
        print("[Depstech] stabilizing...")

        successful_frames = 0

        for i in range(30):
            ok, discarded_frame = cap.read()

            if ok and discarded_frame is not None:
                successful_frames += 1

        print(
            f"[Depstech] stabilization frames: "
            f"{successful_frames}/30"
        )

        # ==========================================
        # 保存用フレームを複数枚取得
        # ==========================================
        frames = []

        print(
            f"[Depstech] capturing "
            f"{num_frames} frames..."
        )

        for i in range(num_frames):

            ok, frame = cap.read()

            if ok and frame is not None:

                frames.append(frame.copy())

                print(
                    f"[Depstech] frame "
                    f"{i + 1}/{num_frames} OK "
                    f"shape={frame.shape}"
                )

            else:

                print(
                    f"[Depstech] frame "
                    f"{i + 1}/{num_frames} FAILED"
                )

        if len(frames) == 0:
            raise RuntimeError(
                f"capture failed: {dev} "
                f"(index={device_index})"
            )

        # とりあえず最後に撮れた画像を確認用として保存
        save_ok = cv2.imwrite(
            str(save_path_ts),
            frames[-1],
        )

        if not save_ok:
            raise RuntimeError(
                f"画像保存に失敗しました: "
                f"{save_path_ts}"
            )

        print(
            f"[Depstech] saved: {save_path_ts}"
        )

        # 1枚ではなく撮影した全画像を返す
        return frames, dev

    finally:
        cap.release()
        print("[Depstech] camera released")

if __name__ == "__main__":

    save_path = "/home/book/Pictures/barcode_capture.png"

    frames, dev = capture_one_depstech(
        save_path,
        width=3840,
        height=2160,
        num_frames=10,
    )

    print(
        f"Captured {len(frames)} images "
        f"from {dev}"
    )

    for i, frame in enumerate(frames):
        print(
            f"frame {i + 1}: "
        )