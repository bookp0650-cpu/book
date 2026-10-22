import os, glob
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import traceback
import re
import threading
import time
from collections import deque

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


class DepstechStream:
    """
    Depstechを一度だけopenして、
    バックグラウンドで常時readする。

    必要になった瞬間に、
    その時点以降の新しいフレームを取得する。
    """

    def __init__(
        self,
        width: int = 3840,
        height: int = 2160,
        fps: int = 30,
        buffer_size: int = 3,
    ):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self.buffer_size = max(
            int(buffer_size),
            3,
        )

        self.dev = None
        self.device_index = None
        self.cap = None

        self._running = False
        self._thread = None

        self._condition = threading.Condition()

        # (frame_id, frame)
        self._frames = deque(
            maxlen=self.buffer_size
        )

        self._frame_id = 0


    def start(self):
        """
        カメラをopenして常時read開始。
        """

        if self._running:
            return

        (
            self.dev,
            self.device_index,
            self.cap,
        ) = find_first_openable_video_device(
            "Depstech"
        )

        print(
            f"[Depstech Stream] opened "
            f"{self.dev} "
            f"(index={self.device_index})"
        )

        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(
                *"MJPG"
            ),
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width,
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.height,
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            self.fps,
        )

        actual_width = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        actual_height = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        actual_fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        print(
            "[Depstech Stream] "
            f"requested: "
            f"{self.width}x{self.height} "
            f"@ {self.fps} fps"
        )

        print(
            "[Depstech Stream] "
            f"actual: "
            f"{actual_width}x{actual_height} "
            f"@ {actual_fps:.1f} fps"
        )

        self._running = True

        self._thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
        )

        self._thread.start()

        # 最初の有効フレームだけ待つ。
        # 30枚捨てる処理はしない。
        deadline = (
            time.monotonic()
            + 2.0
        )

        with self._condition:

            while (
                self._frame_id == 0
            ):

                remaining = (
                    deadline
                    - time.monotonic()
                )

                if remaining <= 0:
                    break

                self._condition.wait(
                    timeout=remaining
                )

        if self._frame_id == 0:

            self.stop()

            raise RuntimeError(
                "Depstech stream started "
                "but no valid frame arrived"
            )

        print(
            "[Depstech Stream] "
            "stream ready"
        )


    def _reader_loop(self):
        """
        常時readして、
        最新フレームだけ保持する。
        """

        while self._running:

            try:
                ok, frame = (
                    self.cap.read()
                )

                if (
                    not ok
                    or frame is None
                ):
                    time.sleep(0.005)
                    continue

                with self._condition:

                    self._frame_id += 1

                    self._frames.append(
                        (
                            self._frame_id,
                            frame,
                        )
                    )

                    self._condition.notify_all()

            except Exception as e:

                if self._running:
                    print(
                        "[Depstech Stream] "
                        f"read error: "
                        f"{type(e).__name__}: "
                        f"{e}"
                    )

                time.sleep(0.01)


    def get_frames(
        self,
        num_frames: int = 3,
        timeout: float = 2.0,
        save_path: Path | None = None,
    ):
        """
        この関数が呼ばれた時点より後の
        新しいフレームをnum_frames枚取得する。

        30fpsなら3枚で約0.1秒。
        """

        if not self._running:
            raise RuntimeError(
                "Depstech stream is not running"
            )

        num_frames = int(
            num_frames
        )

        if num_frames <= 0:
            raise ValueError(
                "num_frames must be >= 1"
            )

        if num_frames > self.buffer_size:
            raise ValueError(
                "num_frames exceeds "
                f"buffer_size={self.buffer_size}"
            )

        deadline = (
            time.monotonic()
            + float(timeout)
        )

        with self._condition:

            # この瞬間より後のフレームだけ使う
            start_frame_id = (
                self._frame_id
            )

            while True:

                fresh = [
                    (
                        frame_id,
                        frame,
                    )
                    for (
                        frame_id,
                        frame,
                    )
                    in self._frames
                    if (
                        frame_id
                        > start_frame_id
                    )
                ]

                if (
                    len(fresh)
                    >= num_frames
                ):
                    selected = (
                        fresh[
                            -num_frames:
                        ]
                    )
                    break

                remaining = (
                    deadline
                    - time.monotonic()
                )

                if remaining <= 0:
                    raise TimeoutError(
                        "Depstech fresh frame "
                        "capture timeout"
                    )

                self._condition.wait(
                    timeout=remaining
                )

            frames = [
                frame.copy()
                for _, frame
                in selected
            ]

        print(
            "[Depstech Stream] "
            f"captured "
            f"{len(frames)} fresh frames"
        )

        for i, frame in enumerate(
            frames,
            start=1,
        ):
            print(
                "[Depstech Stream] "
                f"frame "
                f"{i}/{len(frames)} "
                f"shape={frame.shape}"
            )

        # 従来通り最後の1枚を保存
        if save_path is not None:

            save_path = Path(
                save_path
            )

            ts = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            save_path_ts = (
                save_path.with_name(
                    f"{save_path.stem}_"
                    f"{ts}"
                    f"{save_path.suffix}"
                )
            )

            save_path_ts.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            ok = cv2.imwrite(
                str(save_path_ts),
                frames[-1],
            )

            if not ok:
                raise RuntimeError(
                    "画像保存に失敗しました: "
                    f"{save_path_ts}"
                )

            print(
                "[Depstech Stream] "
                f"saved: {save_path_ts}"
            )

        return (
            frames,
            self.dev,
        )


    def stop(self):
        """
        stream停止・camera release。
        """

        if not self._running:
            return

        print(
            "[Depstech Stream] "
            "stopping..."
        )

        self._running = False

        if self._thread is not None:

            self._thread.join(
                timeout=1.0
            )

        if self.cap is not None:

            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = None
        self._thread = None

        with self._condition:
            self._frames.clear()

        print(
            "[Depstech Stream] "
            "camera released"
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