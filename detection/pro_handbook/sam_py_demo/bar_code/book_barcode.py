from __future__ import annotations
import os
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
    DepstechStream,
)
from detection.pro_handbook.sam_py_demo.bar_code.code_1_pic import (
    barcode_perception,
)
from xarm7.control.xarm7 import XArm7
from play_way_point import WaypointPlayer
import cv2
from detection.pro_handbook.sam_py_demo.ocr_worker_client import (
    run_barcode_ocr_worker,
)




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

# ============================================================
# バーコードOCR fallback
# ============================================================

SAM_PY_DEMO_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


def run_barcode_ocr_once(
    barcode_number,
    img,
    out_dir,
):
    """
    バーコード認識3枚失敗後、
    書籍認識と同じ常駐PP-OCRv5 workerを使用する。
    """


    out_dir = Path(out_dir)


    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ==========================================
    # OCR入力画像保存
    # ==========================================
    input_path = (
        out_dir
        / "ocr_input.png"
    )


    ok = cv2.imwrite(
        str(input_path),
        img,
    )


    if not ok:
        print(
            "[BARCODE OCR ERROR] "
            f"cv2.imwrite failed: "
            f"{input_path}"
        )
        return False, None


    print("")
    print("========================================")
    print("[BARCODE OCR FALLBACK]")
    print("shared persistent PP-OCRv5 worker")
    print(f"input : {input_path}")
    print(f"query : {barcode_number}")
    print("========================================")


    try:
        data, stdout = (
            run_barcode_ocr_worker(
                image_path=input_path,
                target=barcode_number,
                out_dir=out_dir,
                timeout=60.0,
            )
        )


        if stdout:
            print(
                stdout,
                end=""
                if stdout.endswith("\n")
                else "\n",
            )


        matched = bool(
            data.get(
                "matched",
                False,
            )
        )


        print(
            "[BARCODE OCR RESULT] "
            f"matched={matched}"
        )


        print(
            "[BARCODE OCR RESULT] "
            f"matched_text="
            f"{data.get('matched_text')!r}"
        )


        print(
            "[BARCODE OCR RESULT] "
            f"matched_digits="
            f"{data.get('matched_digits')!r}"
        )


        return matched, data


    except Exception as e:
        print(
            "[BARCODE OCR ERROR] "
            f"{type(e).__name__}: "
            f"{e}"
        )


        traceback.print_exc()


        return False, None

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


    camera = None

    def capture_and_perceive(
        side: str,
    ):
        """
        1.
        Depstechで10枚撮影。

        2.
        10枚すべてMorphology処理後の画像で
        pyzbar認識。

        3.
        10枚すべてで目的バーコードが
        認識できなかった場合のみ、
        最後の有効フレーム1枚に対して
        PP-OCRv5を1回だけ実行。

        4.
        OCR文字列内にtarget数字列が
        含まれていればsuccess。
        """

        print(
            f"[book barcode] "
            f"capture start: "
            f"side={side}"
        )

        capture_path = (
            shot_dir
            / f"book_barcode_capture_{side}.png"
        )

        # ==================================
        # Depstechで10枚撮影
        # ==================================
        frames, dev = (
            camera.get_frames(
                num_frames=5,
                timeout=2.0,
                save_path=capture_path,
            )
        )

        if (
            frames is None
            or len(frames) == 0
        ):

            raise RuntimeError(
                "DepstechStream.get_frames() "
                "returned no frames"
            )

        print(
            f"[book barcode] "
            f"captured "
            f"{len(frames)} frames "
            f"from {dev}"
        )

        valid_statuses = {
            "success",
            "wrong_barcode",
            "no_barcode",
            "error",
        }

        wrong_barcode_data = []
        error_count = 0

        # OCRへ渡す代表画像
        # 最後に正常取得できたフレームを使用
        ocr_frame = None

        # ==================================
        # 10枚をMorphologyバーコード認識
        # ==================================
        for i, frame in enumerate(
            frames
        ):

            print(
                f"[book barcode] "
                f"trying frame "
                f"{i + 1}/"
                f"{len(frames)}"
            )

            color_frame = np.asanyarray(
                frame
            )

            if color_frame.size == 0:

                print(
                    f"[book barcode] "
                    f"frame "
                    f"{i + 1} is empty"
                )

                continue

            # 最後の正常フレームを
            # OCR代表画像として保持
            ocr_frame = (
                color_frame.copy()
            )

            # ==================================
            # 保存先
            #
            # book_barcode/
            #   left_frame_01/morph.png
            # ==================================
            preprocess_dir = (
                shot_dir
                / (
                    f"{side}_frame_"
                    f"{i + 1:02d}"
                )
            )

            status, barcode_data = (
                barcode_perception(
                    barcode_number_input,
                    color_frame,
                    debug_dir=preprocess_dir,
                )
            )

            if (
                status
                not in valid_statuses
            ):

                raise ValueError(
                    "Unknown barcode status: "
                    f"{status!r}"
                )

            print(
                f"[book barcode] "
                f"frame={i + 1}, "
                f"status={status}"
            )

            # ==================================
            # バーコードで正解
            # ==================================
            if status == "success":

                print(
                    "[book barcode] "
                    "SUCCESS by pyzbar "
                    f"at frame "
                    f"{i + 1}/"
                    f"{len(frames)}"
                )

                return (
                    "success",
                    barcode_data,
                )

            # ==================================
            # 番号違い
            # ただしまだreturnしない。
            #
            # OCRでtargetが見つかる可能性がある。
            # ==================================
            if (
                status
                == "wrong_barcode"
            ):

                if (
                    barcode_data
                    is not None
                ):

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

            elif (
                status
                == "error"
            ):

                error_count += 1

        # ==================================
        # ここまで来た
        # =
        # 10枚全部で目的バーコードなし
        #
        # OCRはここで1回だけ
        # ==================================
        if ocr_frame is not None:

            ocr_dir = (
                shot_dir
                / f"{side}_ocr"
            )

            (
                ocr_matched,
                ocr_data,
            ) = run_barcode_ocr_once(
                barcode_number_input,
                ocr_frame,
                ocr_dir,
            )

            # ==================================
            # OCR文字列中にtarget数字列あり
            # ==================================
            if ocr_matched:

                print(
                    "[book barcode] "
                    "SUCCESS by OCR fallback"
                )

                if (
                    ocr_data
                    is None
                ):
                    ocr_data = {}

                ocr_data[
                    "method"
                ] = "ocr"

                return (
                    "success",
                    ocr_data,
                )

        # ==================================
        # OCRでもtargetなし
        # → 従来判定へ
        # ==================================

        if wrong_barcode_data:

            print(
                "[book barcode] "
                "all frames checked + "
                "OCR failed: "
                "wrong barcode detected"
            )

            return (
                "wrong_barcode",
                wrong_barcode_data,
            )

        if (
            error_count
            == len(frames)
        ):

            print(
                "[book barcode] "
                "all frames returned error"
            )

            return (
                "error",
                None,
            )

        print(
            "[book barcode] "
            "all barcode frames + "
            "OCR checked: no match"
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

            # OCR fallback成功時
            if isinstance(
                barcode_data,
                dict,
            ):
                return [
                    barcode_data
                ]

            # pyzbar成功時
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

        # ==================================
        # Depstechを先に起動
        # アーム移動中も常時readする
        # ==================================
        camera = DepstechStream(
            width=3840,
            height=2160,
            fps=30,
            buffer_size=5,
        )

        camera.start()

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
                610.0,
                55.0,
                0.0,
                0.0,
                0.0,
            ],
            velocity=movel_velocity,
            acceleration=movel_acceleration,
        )


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

    finally:

        if camera is not None:

            try:
                camera.stop()

            except Exception as e:
                print(
                    "[book barcode] "
                    "Depstech stop warning: "
                    f"{type(e).__name__}: "
                    f"{e}"
                )

# ============================================================
# ISBN / BOOK MASTER LOOKUP
# ============================================================
def normalize_isbn(value):
    """
    ISBN比較用。
    ハイフン・空白などを除去する。
    """
    return "".join(
        ch
        for ch in str(value).upper()
        if ch.isdigit() or ch == "X"
    )


def find_book_by_isbn(
    books_master,
    detected_isbn,
):
    """
    読み取ったISBNからbooks_masterを逆引きする。

    Parameters
    ----------
    books_master : list[dict]
        書籍マスターJSON

    detected_isbn : str
        バーコードから読み取ったISBN

    Returns
    -------
    dict
        ISBNが一致した書籍のJSONエントリ
    """

    target_isbn = normalize_isbn(
        detected_isbn
    )

    if not target_isbn:
        raise RuntimeError(
            "detected ISBN is empty"
        )

    matches = []

    for book in books_master:

        if not isinstance(book, dict):
            continue

        json_isbn = normalize_isbn(
            book.get(
                "ISBN_number",
                "",
            )
        )

        if json_isbn == target_isbn:
            matches.append(book)

    if len(matches) == 0:
        raise RuntimeError(
            "読み取ったISBNがJSONに存在しません: "
            f"{detected_isbn}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            "同じISBNがJSONに複数存在します: "
            f"{detected_isbn}"
        )

    return matches[0]


def identify_book_isbn_at_current_pose(
    books_master,
    shot_dir,
    num_frames=5,
):
    """
    入庫用バーコード認識。

    アームは動かさない。
    現在のバーコード撮影姿勢で、

      1. Depstechから複数枚取得
      2. 出庫と同じbarcode_perception
      3. 全枚失敗したらPP-OCRv5
      4. 認識ISBNをbooks_masterから検索

    Returns
    -------
    detected_isbn : str
    matched_book : dict
    method : str
        "pyzbar" or "ocr"
    """

    shot_dir = Path(shot_dir)

    shot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    camera = None
    ocr_frame = None

    try:
        # ====================================================
        # 出庫と同じDepstech連続撮影
        # ====================================================
        camera = DepstechStream(
            width=3840,
            height=2160,
            fps=30,
            buffer_size=5,
        )

        camera.start()

        capture_path = (
            shot_dir
            / "storage_book_barcode_capture.png"
        )

        frames, dev = camera.get_frames(
            num_frames=num_frames,
            timeout=2.0,
            save_path=capture_path,
        )

        if (
            frames is None
            or len(frames) == 0
        ):
            raise RuntimeError(
                "DepstechStream.get_frames() "
                "returned no frames"
            )

        print(
            "[STORAGE BARCODE] "
            f"captured {len(frames)} frames "
            f"from {dev}"
        )

        # ====================================================
        # まずpyzbar
        # ====================================================
        for i, frame in enumerate(
            frames
        ):
            print(
                "[STORAGE BARCODE] "
                f"trying frame "
                f"{i + 1}/{len(frames)}"
            )

            color_frame = np.asanyarray(
                frame
            )

            if color_frame.size == 0:
                continue

            # OCR fallback用に最後の正常画像を保存
            ocr_frame = color_frame.copy()

            preprocess_dir = (
                shot_dir
                / f"frame_{i + 1:02d}"
            )

            status, barcode_data = (
                barcode_perception(
                    None,
                    color_frame,
                    debug_dir=preprocess_dir,
                )
            )

            if (
                status == "success"
                and barcode_data is not None
            ):
                raw_data = getattr(
                    barcode_data,
                    "data",
                    None,
                )

                if raw_data is None:
                    continue

                if isinstance(
                    raw_data,
                    (bytes, bytearray),
                ):
                    detected_isbn = (
                        raw_data.decode(
                            "utf-8",
                            errors="replace",
                        ).strip()
                    )
                else:
                    detected_isbn = str(
                        raw_data
                    ).strip()

                if not detected_isbn:
                    continue

                print(
                    "[STORAGE BARCODE] "
                    f"pyzbar detected="
                    f"{detected_isbn!r}"
                )

                # JSONに存在するISBNか確認
                try:
                    matched_book = (
                        find_book_by_isbn(
                            books_master,
                            detected_isbn,
                        )
                    )

                    print(
                        "[STORAGE BARCODE] "
                        "SUCCESS by pyzbar"
                    )

                    return (
                        detected_isbn,
                        matched_book,
                        "pyzbar",
                    )

                except RuntimeError as e:
                    # ISBN以外のバーコード等を
                    # 読んだ可能性があるので次フレームへ
                    print(
                        "[STORAGE BARCODE] "
                        f"not in books master: {e}"
                    )


        # ====================================================
        # pyzbar全滅
        # -> 出庫と同じPP-OCRv5 fallback
        # ====================================================
        if ocr_frame is None:
            raise RuntimeError(
                "OCR fallback frame is None"
            )

        print("")
        print("========================================")
        print("[STORAGE BARCODE OCR FALLBACK]")
        print("pyzbar failed -> PP-OCRv5")
        print("========================================")

        ocr_dir = (
            shot_dir
            / "ocr"
        )

        # 入庫では期待ISBNが無いので空文字。
        # OCR自体は出庫と同じworkerを使う。
        _, ocr_data = run_barcode_ocr_once(
            "",
            ocr_frame,
            ocr_dir,
        )

        if not isinstance(
            ocr_data,
            dict,
        ):
            raise RuntimeError(
                "barcode OCR returned no data"
            )

        texts = ocr_data.get(
            "texts",
            [],
        )

        if not isinstance(
            texts,
            list,
        ):
            texts = []

        print(
            "[STORAGE BARCODE OCR] "
            f"texts={texts!r}"
        )


        # ====================================================
        # OCR文字列とJSON内ISBNを照合
        #
        # 出庫：
        #   expected ISBN in OCR digits
        #
        # 入庫：
        #   JSON内ISBN in OCR digits
        # ====================================================
        found = {}

        for text in texts:

            ocr_digits = normalize_isbn(
                text
            )

            if not ocr_digits:
                continue

            for book in books_master:

                if not isinstance(
                    book,
                    dict,
                ):
                    continue

                json_isbn = normalize_isbn(
                    book.get(
                        "ISBN_number",
                        "",
                    )
                )

                if not json_isbn:
                    continue

                if json_isbn in ocr_digits:
                    found[
                        json_isbn
                    ] = book


        if len(found) == 0:
            raise RuntimeError(
                "PP-OCRv5でもJSON内ISBNを"
                "特定できませんでした"
            )

        if len(found) > 1:
            raise RuntimeError(
                "OCR結果から複数ISBNが"
                "候補になりました: "
                f"{list(found.keys())}"
            )


        detected_isbn = next(
            iter(found.keys())
        )

        matched_book = found[
            detected_isbn
        ]

        print(
            "[STORAGE BARCODE] "
            "SUCCESS by PP-OCRv5"
        )

        return (
            detected_isbn,
            matched_book,
            "ocr",
        )

    finally:
        if camera is not None:
            try:
                camera.stop()
            except Exception as e:
                print(
                    "[STORAGE BARCODE] "
                    "camera stop warning: "
                    f"{type(e).__name__}: {e}"
                )


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