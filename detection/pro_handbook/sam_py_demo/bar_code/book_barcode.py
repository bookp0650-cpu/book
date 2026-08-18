from __future__ import annotations

import sys

sys.path.append(
    "/home/book/pro_book_SAM3/pro_hand_book_python"
)

sys.path.append(
    "/home/book/pro_book_SAM3/pro_hand_book_python/"
    "ros2_ws/src/xarm7_teaching/src"
)

from pathlib import Path
import json
from datetime import datetime
import time
import numpy as np
import traceback
from typing import Literal

from detection.pro_handbook.sam_py_demo.bar_code.web_camera_capture import (
    capture_one_depstech,
)
from detection.pro_handbook.sam_py_demo.bar_code.code_1_pic import (
    barcode_perception,
)
from xarm7.control.xarm7 import XArm7
from play_way_point import WaypointPlayer





BOOK_BARCODE_1 = [
    52.4,
    -82.0,
    178.0,
    78.0,
    204.0,
    4.6,
    -61.2,
]

BOOK_BARCODE_2 = [
    -36.9,
    -75.2,
    159.3,
    80.5,
    83.8,
    18.8,
    -34.9,
]

CONFIG_DIR = Path(
    "/home/book/pro_book_SAM3/pro_hand_book_python/"
    "ros2_ws/src/xarm7_teaching/config"
)



def get_barcode_motion_params(
    lift_height_mm: float,
):
    """
    リフトが高いほど、
    バーコード姿勢変更とMoveLを速くする。
    """

    h = float(lift_height_mm)

    if h >= 400.0:
        switch_velocity = 1.0
        switch_acceleration = 2.0

        movel_velocity = 220.0
        movel_acceleration = 220.0

    else:
        switch_velocity = 0.50
        switch_acceleration = 1.00

        movel_velocity = 180.0
        movel_acceleration = 180.0

    print(
        "[BARCODE MOTION PARAM] "
        f"lift_height={h:.1f} mm, "
        f"switch_vel={switch_velocity:.2f}, "
        f"switch_acc={switch_acceleration:.2f}, "
        f"movel_vel={movel_velocity:.1f}, "
        f"movel_acc={movel_acceleration:.1f}"
    )

    return (
        switch_velocity,
        switch_acceleration,
        movel_velocity,
        movel_acceleration,
    )



def decoded_to_dict_safe(d) -> dict:
    """barcode SDKのdecoded要素を、安全にJSON化できるdictへ変換する"""
    # data: bytes -> str
    raw_data = getattr(d, "data", None)
    if isinstance(raw_data, (bytes, bytearray)):
        data = raw_data.decode("utf-8", errors="replace")
    else:
        data = raw_data

    # rect
    rect = getattr(d, "rect", None)
    rect_dict = None
    if rect is not None:
        # rect.left/top/width/height が無い/Noneでも落ちないようにする
        rect_dict = {
            "left": int(getattr(rect, "left", 0) or 0),
            "top": int(getattr(rect, "top", 0) or 0),
            "width": int(getattr(rect, "width", 0) or 0),
            "height": int(getattr(rect, "height", 0) or 0),
        }

    # polygon
    poly = []
    for p in (getattr(d, "polygon", None) or []):
        poly.append({"x": int(getattr(p, "x", 0) or 0), "y": int(getattr(p, "y", 0) or 0)})

    # quality: int(None) で落ちるのを防ぐ
    q = getattr(d, "quality", -1)
    quality = int(q) if isinstance(q, (int, np.integer)) else -1

    return {
        "data": data,
        "type": getattr(d, "type", None),
        "quality": quality,
        "orientation": getattr(d, "orientation", None),
        "rect": rect_dict,
        "polygon": poly,
    }


BarcodeStatus = Literal[
    "success",
    "wrong_barcode",
    "no_barcode",
    "error",
]

ContainerSide = Literal[
    "left",
    "right",
    "unknown",
]


def book_barcode_sequence(
    barcode_number_input: str,
    shot_dir: Path,
    arm: XArm7,
    lift_height_mm: float,
    stage_callback=None,
) -> tuple[BarcodeStatus, ContainerSide]:
    """
    左面のバーコードを撮影する。

    左面でバーコードが検出されなかった場合は、
    右面撮影位置へ移動して再撮影する。

    Returns
    -------
    tuple[BarcodeStatus, ContainerSide]

    barcode_status:
        "success"
            指定バーコードと一致

        "wrong_barcode"
            バーコードは検出したが番号が不一致

        "no_barcode"
            バーコードを検出できなかった

        "error"
            撮影、認識、移動、保存などでエラー

    container_side:
        "left"
            従来のMove_to_Containerを使用する

        "right"
            mori側のmove_to_container_rightsideを使用する

        "unknown"
            右面への移動途中で失敗し、
            収納方向を安全に決定できない
    """

    def set_stage(stage_name: str):
        stage_name = str(stage_name).strip()

        if not stage_name:
            return

        print(
            "[book_barcode][STAGE] "
            f"{stage_name}"
        )

        if stage_callback is not None:
            stage_callback(
                stage_name
            )

    ts = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    shot_dir = Path(shot_dir)
    shot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        shot_dir
        / f"book_barcode_result_{ts}.json"
    )

    err_path = (
        shot_dir
        / f"book_barcode_error_{ts}.txt"
    )

    # 最初は左面にいる
    container_side: ContainerSide = "left"

    # 左右それぞれの認識結果を保存する
    side_results = {}

    def capture_and_perceive(
        side: str,
    ):
        """
        現在姿勢で撮影し、バーコード認識を1回実行する。
        """

        print(
            f"[book barcode] capture start: "
            f"side={side}"
        )

        capture_path = (
            shot_dir
            / f"book_barcode_capture_{side}.png"
        )

        # ==================================
        # Depstechで10枚撮影
        # ==================================
        frames, dev = capture_one_depstech(
            capture_path,
            width=3840,
            height=2160,
            num_frames=10,
        )

        if frames is None or len(frames) == 0:
            raise RuntimeError(
                "capture_one_depstech() returned no frames"
            )

        print(
            f"[book barcode] "
            f"captured {len(frames)} frames "
            f"from {dev}"
        )

        valid_statuses = {
            "success",
            "wrong_barcode",
            "no_barcode",
            "error",
        }

        # ==================================
        # 10枚を順番にバーコード認識
        # ==================================
        wrong_barcode_data = []
        error_count = 0

        for i, frame in enumerate(frames):

            print(
                f"[book barcode] "
                f"trying frame "
                f"{i + 1}/{len(frames)}"
            )

            color_frame = np.asanyarray(
                frame
            )

            if color_frame.size == 0:
                print(
                    f"[book barcode] "
                    f"frame {i + 1} is empty"
                )
                continue

            status, barcode_data = (
                barcode_perception(
                    barcode_number_input,
                    color_frame,
                )
            )

            if status not in valid_statuses:
                raise ValueError(
                    f"Unknown barcode status: "
                    f"{status!r}"
                )

            print(
                f"[book barcode] "
                f"frame={i + 1}, "
                f"status={status}"
            )

            print(
                "[book barcode] "
                "barcode_data:",
                barcode_data,
            )

            # ==================================
            # 正しいバーコードを発見
            # → 即成功
            # ==================================
            if status == "success":

                print(
                    f"[book barcode] "
                    f"SUCCESS at frame "
                    f"{i + 1}/{len(frames)}"
                )

                return (
                    "success",
                    barcode_data,
                )

            # ==================================
            # バーコードは見えたが番号違い
            # ==================================
            if status == "wrong_barcode":

                if barcode_data is not None:

                    if isinstance(
                        barcode_data,
                        (list, tuple),
                    ):
                        wrong_barcode_data.extend(
                            barcode_data
                        )

                    else:
                        wrong_barcode_data.append(
                            barcode_data
                        )

            # ==================================
            # 認識処理自体がエラー
            # ==================================
            elif status == "error":
                error_count += 1

        # ==================================
        # 10枚全部試しても正解なし
        # ==================================

        # どこかの画像で別バーコードが見えていた
        if wrong_barcode_data:

            print(
                "[book barcode] "
                "all frames checked: "
                "wrong barcode detected"
            )

            return (
                "wrong_barcode",
                wrong_barcode_data,
            )

        # 全画像で認識処理そのものがエラー
        if error_count == len(frames):

            print(
                "[book barcode] "
                "all frames returned error"
            )

            return (
                "error",
                None,
            )

        # バーコード自体が見つからなかった
        print(
            "[book barcode] "
            "all frames checked: "
            "no barcode"
        )

        return (
            "no_barcode",
            None,
        )

    def convert_barcode_data(
        status: str,
        barcode_data,
    ) -> list[dict]:
        """
        pyzbarのDecodedオブジェクトを
        JSON保存可能なdictへ変換する。
        """

        if barcode_data is None:
            return []

        if status == "success":
            # successではDecodedが1件返る
            return [
                decoded_to_dict_safe(
                    barcode_data
                )
            ]

        if status == "wrong_barcode":
            # wrong_barcodeではDecodedのリストが返る
            return [
                decoded_to_dict_safe(d)
                for d in barcode_data
            ]

        return []

    try:
        set_stage(
            "BOOK_BARCODE_START"
        )
        # ==================================
        # 左面バーコード撮影姿勢へ移動
        # ==================================
        print(
            "[book barcode] "
            "move to left barcode pose"
        )

        (
            switch_velocity,
            switch_acceleration,
            movel_velocity,
            movel_acceleration,
        ) = get_barcode_motion_params(
            lift_height_mm
        )

        print(
            "[BOOK BARCODE MOTION] "
            f"lift_height={lift_height_mm:.1f} mm, "
            f"switch_vel={switch_velocity:.2f}, "
            f"switch_acc={switch_acceleration:.2f}, "
            f"movel_vel={movel_velocity:.1f}, "
            f"movel_acc={movel_acceleration:.1f}"
        )

        arm.switch_gripper_pose(
            BOOK_BARCODE_1,
            velocity=switch_velocity,
            acceleration=switch_acceleration,
        )

        arm.moveL_relative(
            [
                -80.0,
                580.0,
                55.0,
                0.0,
                0.0,
                0.0,
            ],
            velocity=movel_velocity,
            acceleration=movel_acceleration,
        )

        time.sleep(2.0)

        # ==================================
        # 左面バーコード認識
        # ==================================
        left_status, left_data = (
            capture_and_perceive("left")
        )

        side_results["left"] = {
            "status": left_status,
            "detected_barcodes": (
                convert_barcode_data(
                    left_status,
                    left_data,
                )
            ),
        }

        # ==================================
        # 左面にバーコードがあった場合
        # ==================================
        if left_status != "no_barcode":
            final_status = left_status

        # ==================================
        # 左面にバーコードが映らなかった場合
        # → 右面へ移動して再撮影
        # ==================================
        else:
            print(
                "[book barcode] left no_barcode "
                "-> move to right side"
            )

            container_side = "unknown"

            set_stage(
                "BOOK_BARCODE_RIGHT_MOVING"
            )

            # 左側撮影位置から引き抜く
            arm.moveL_relative([
                75,
                -350.0,
                -30.0,
                0.0,
                0.0,
                0.0,
            ])

            set_stage(
                "BOOK_BARCODE_RETURNING_LEFT"
            )


            # 左側から右側へ移動
            WaypointPlayer.play_with_arm(
                arm=arm,
                yaml_path=(
                    CONFIG_DIR
                    / "leftside_to_rightside_v1.yaml"
                ),
                start_name="p2",
                end_name="p3",
                speed=1.0,
                accel=1.0,
                wait=True,
            )

            set_stage(
                "BOOK_BARCODE_RIGHT_SIDE"
            )

            container_side = "right"

            time.sleep(2.0)

            # 右側認識
            right_status, right_data = (
                capture_and_perceive("right")
            )

            side_results["right"] = {
                "status": right_status,
                "detected_barcodes": (
                    convert_barcode_data(
                        right_status,
                        right_data,
                    )
                ),
            }

            final_status = right_status


            # 右側から左側へ移動
            WaypointPlayer.play_with_arm(
                arm=arm,
                yaml_path=(
                    CONFIG_DIR
                    / "rightside_to_leftside_v1.yaml"
                ),
                start_name="p2",
                end_name="p3",
                speed=1.0,
                accel=1.0,
                wait=True,
            )




        # ==================================
        # JSON保存
        # ==================================
        payload = {
            "timestamp": ts,
            "barcode_number_input": str(
                barcode_number_input
            ).strip(),
            "final_barcode_status": final_status,
            "container_side": container_side,
            "barcode_identification": (
                final_status == "success"
            ),
            "side_results": side_results,
        }

        log_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "[book barcode] saved:",
            str(log_path),
        )

        print(
            "[book barcode] final result:",
            final_status,
            container_side,
        )

        return final_status, container_side

    except Exception:
        error_text = traceback.format_exc()

        print("[book barcode] ERROR")
        print(error_text)

        try:
            err_path.write_text(
                error_text,
                encoding="utf-8",
            )

            print(
                "[book barcode] ERROR saved:",
                str(err_path),
            )

            error_payload = {
                "timestamp": ts,
                "barcode_number_input": str(
                    barcode_number_input
                ).strip(),
                "final_barcode_status": "error",
                "container_side": container_side,
                "side_results": side_results,
                "error": error_text,
            }

            log_path.write_text(
                json.dumps(
                    error_payload,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        except Exception as save_error:
            print(
                "[book barcode] "
                "ERROR while saving error log:",
                repr(save_error),
            )

        return "error", container_side

import rclpy
from rclpy.node import Node

class DummyNode(Node):
    def __init__(self):
        super().__init__("book_barcode_node")


def main():
    rclpy.init()

    node = DummyNode()
    arm = None

    try:
        # xArm接続
        arm = XArm7(node)

        # 認識したいバーコード番号
        barcode_number_input = "1234567890"

        # 撮影画像・結果ログの保存先
        shot_dir = Path("./logs")
        shot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 左側認識
        # → 必要なら右側認識
        # → 左側へ戻る
        result = book_barcode_sequence(
            barcode_number_input=barcode_number_input,
            shot_dir=shot_dir,
            arm=arm,
        )

        barcode_status, container_side = result

        print("\n========== RESULT ==========")
        print("barcode_status :", barcode_status)
        print("container_side :", container_side)
        print("============================")

    except KeyboardInterrupt:
        print("\n[main] Interrupted")

    except Exception:
        print("\n[main] ERROR")
        print(traceback.format_exc())

    finally:
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()