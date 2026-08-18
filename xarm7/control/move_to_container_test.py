#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node

import Dynamixel_win_pro_hand_book.HandBook_Retrieval as HandBook

from xarm7.control.xarm7 import XArm7
from xarm7.control.xarm_init_to_capture_integration import WaypointPlayerNode
from xarm7.control.xarm_monitor import XArmMonitor


# =========================================================
# 設定
# =========================================================

BOOK_CAPTURE = -210.0
CONTAINER_TILT_DEG = 13.0

# コンテナ満杯とみなす累積幅 [mm]
CONTAINER_FULL_WIDTH_MM = 320.0

CONFIG_DIR = (
    "/home/book/pro_book_SAM3/pro_hand_book_python/"
    "ros2_ws/src/xarm7_teaching/config/"
)


# =========================================================
# コンテナ収納
# =========================================================

def Move_to_Container(
    offset: float,
    book_width: float,
    arm: XArm7,
    waypoint_node,
    HandMotors,
    stage_callback=None,
    resume_stage=None,
):
    """
    出庫した本をコンテナへ収納する。

    Parameters
    ----------
    offset : float
        今回の本を収納する前までに、
        コンテナへ収納済みの本の累積幅 [mm]

    book_width : float
        今回収納する本の幅 [mm]

    arm : XArm7
        xArm7制御オブジェクト

    waypoint_node
        WaypointPlayerNode

    HandMotors
        Dynamixel制御オブジェクト

    Returns
    -------
    HandMotors
        現在有効なDynamixel制御オブジェクト

    container_full : bool
        今回の本を収納した結果、
        コンテナが満杯になった場合 True
    """


    # =====================================================
    # stage / resume制御
    # =====================================================

    if resume_stage is not None:
        resume_stage = str(
            resume_stage
        ).strip()

        if not resume_stage:
            resume_stage = None

    resume_control = {
        "waiting": resume_stage is not None
    }


    def set_stage(
        stage_name: str,
    ) -> bool:

        stage_name = str(
            stage_name
        ).strip()

        if not stage_name:
            return False

        # ==============================================
        # resume_stageに到達するまでは処理をskip
        # ==============================================
        if resume_control["waiting"]:

            if stage_name != resume_stage:
                print(
                    "[Move_to_Container][SKIP] "
                    f"{stage_name}"
                )

                return False

            # 保存stageに到達
            resume_control["waiting"] = False

            print("")
            print("========================================")
            print(
                "[Move_to_Container][RESUME]"
            )
            print(
                f"resume from: {stage_name}"
            )
            print("========================================")
            print("")

        # ==============================================
        # 通常stage通知
        # ==============================================
        print(
            "[Move_to_Container][STAGE] "
            f"{stage_name}"
        )

        if stage_callback is not None:
            stage_callback(
                stage_name
            )

        return True

    # =====================================================
    # 値の確認
    # =====================================================

    offset = float(offset)
    book_width = float(book_width)

    if not math.isfinite(offset):
        raise ValueError(
            f"offset is not finite: {offset}"
        )

    if not math.isfinite(book_width):
        raise ValueError(
            f"book_width is not finite: {book_width}"
        )

    if offset < 0.0:
        raise ValueError(
            f"offset must be >= 0: {offset}"
        )

    if book_width <= 0.0:
        raise ValueError(
            f"book_width must be > 0: {book_width}"
        )

    print("")
    print("========================================")
    print("[Move_to_Container]")
    print(
        f"現在のoffset       : "
        f"{offset:.1f} mm"
    )
    print(
        f"今回のbook_width   : "
        f"{book_width:.1f} mm"
    )
    print(
        f"収納後のoffset     : "
        f"{offset + book_width:.1f} mm"
    )
    print("========================================")

    # =====================================================
    # 安全確認
    #
    # この関数に入った時点ですでに330 mm以上なら、
    # 前の本ですでにコンテナは満杯になっている。
    #
    # 正常な統合処理では、前の本の処理終了時に
    # container_full=Trueでメインループが終了するので、
    # 基本的にはここへ来ない。
    # =====================================================

    if offset >= CONTAINER_FULL_WIDTH_MM:

        print("")
        print("========================================")
        print(" コンテナはすでに満杯です")
        print(
            f"offset = {offset:.1f} mm "
            f">= {CONTAINER_FULL_WIDTH_MM:.1f} mm"
        )
        print(" 今回の本は収納しません")
        print("========================================")

        raise RuntimeError(
            "Container already full before storage: "
            f"offset={offset:.1f} mm"
        )

    # =====================================================
    # 今回の本を収納した結果、
    # コンテナが満杯になるかを事前に計算
    #
    # ただし、満杯になる場合でも今回の本は収納する。
    # =====================================================

    next_offset = (
        offset
        + book_width
    )

    container_full = (
        next_offset
        >= CONTAINER_FULL_WIDTH_MM
    )

    if container_full:

        print("")
        print("========================================")
        print(" 今回の本が最後の1冊になります")
        print(
            f"{offset:.1f} + "
            f"{book_width:.1f} = "
            f"{next_offset:.1f} mm"
        )
        print(
            " この本は通常通り"
            "コンテナへ収納します"
        )
        print(
            " 収納完了後に"
            "container_full=True を返します"
        )
        print("========================================")

    set_stage(
        "CONTAINER_START"
    )

    # =====================================================
    # 1. コンテナ手前まで移動
    # =====================================================

    print("")
    print(
        "[Move_to_Container] "
        "move_to_container_t.yaml"
    )

    if set_stage(
        "CONTAINER_APPROACHING"
    ):

        waypoint_node.reset()

        waypoint_node.play_direct(
            CONFIG_DIR
            + "move_to_container_t.yaml"
        )

        while not waypoint_node.is_finished():
            time.sleep(0.1)

        if waypoint_node.is_failed():
            raise RuntimeError(
                "move_to_container_t.yaml "
                "の実行に失敗しました: "
                f"{waypoint_node.error_message()}"
            )

        waypoint_node.reset()

    set_stage(
        "CONTAINER_APPROACH_DONE"
    )
    # =====================================================
    # 2. offsetに応じて収納位置を選択
    #
    # offsetは「今回の本を収納する前」の累積幅。
    # =====================================================

    if offset < 30.0:

        yaml_file = (
            "container_offset_30.0.yaml"
        )

    elif offset < 60.0:

        yaml_file = (
            "container_offset_60.0.yaml"
        )

    elif offset < 90.0:

        yaml_file = (
            "container_offset_90.0.yaml"
        )

    elif offset < 120.0:

        yaml_file = (
            "container_offset_120.0.yaml"
        )

    elif offset < 150.0:

        yaml_file = (
            "container_offset_150.0.yaml"
        )

    elif offset < 180.0:

        yaml_file = (
            "container_offset_180.0.yaml"
        )

    elif offset < 210.0:

        yaml_file = (
            "container_offset_210.0.yaml"
        )

    elif offset < 240.0:

        yaml_file = (
            "container_offset_240.0.yaml"
        )

    elif offset < 270.0:

        yaml_file = (
            "container_offset_270.0.yaml"
        )

    elif 270 <= offset < 300.0:
        yaml_file = (
            "container_offset_300.0.yaml"
        )

    else:
        yaml_file = (
            "container_offset_330.0.yaml"
        )


    yaml_path = (
        CONFIG_DIR
        + yaml_file
    )

    print("")
    print(
        "[container_offset] "
        f"offset={offset:.1f} mm"
    )

    print(
        "[container_offset] "
        f"使用する軌道ファイル: "
        f"{yaml_file}"
    )

    # =====================================================
    # 3. コンテナ内の収納位置まで移動
    # =====================================================
    if set_stage(
        "CONTAINER_POSITIONING"
    ):

        waypoint_node.play_direct(
            yaml_path
        )

        while not waypoint_node.is_finished():
            time.sleep(0.1)

        if waypoint_node.is_failed():
            raise RuntimeError(
                f"{yaml_file} "
                "の実行に失敗しました: "
                f"{waypoint_node.error_message()}"
            )

        waypoint_node.reset()

    set_stage(
        "CONTAINER_POSITION_DONE"
    )

    # =====================================================
    # 4. コンテナの傾きに応じてZを補正
    # =====================================================

    theta = math.radians(
        CONTAINER_TILT_DEG
    )

    z_drop = (
        BOOK_CAPTURE
        + offset
        * math.tan(theta)
    )

    print("")
    print(
        "[Move_to_Container] "
        f"z_drop={z_drop:.2f} mm"
    )

    if set_stage(
        "CONTAINER_Z_DOWN"
    ):

        arm.moveL_z_offset(
            z_drop
        )

    set_stage(
        "CONTAINER_Z_DOWN_DONE"
    )
    # =====================================================
    # 5. 本をコンテナへ置く
    #
    # ★ container_full=Trueになる最後の本でも、
    #   ここは必ず実行する。
    # =====================================================

    print("")
    print(
        "[Move_to_Container] "
        "本を離します"
    )

    if set_stage(
        "CONTAINER_RELEASE_BEGIN"
    ):

        HandMotors = (
            HandMotors.open_until_full(
                asynchronous=False,
            )
        )

        if HandMotors is None:
            raise RuntimeError(
                "open_until_full() "
                "returned None"
            )

    set_stage(
        "CONTAINER_RELEASE_DONE"
    )

    print(
        "[Move_to_Container] "
        "open_until_full completed"
    )

    # =====================================================
    # 6. 本を離した後の退避軌道
    # =====================================================

    print("")
    print(
        "[Move_to_Container] "
        "move_to_container_final.yaml"
    )

    retreat_started = False

    if set_stage(
        "CONTAINER_RETREATING"
    ):

        waypoint_node.reset()

        waypoint_node.play_direct(
            CONFIG_DIR
            + "move_to_container_final.yaml"
        )

        retreat_started = True

    # =====================================================
    # 7. ハンドを閉じる
    #
    # open_until_full()で通信復旧した場合でも、
    # 上で受け取った最新のHandMotorsを使用する。
    # =====================================================

    print(
        "[Move_to_Container] "
        "ハンドを閉じます"
    )

    if set_stage(
        "CONTAINER_HAND_CLOSE_BEGIN"
    ):

        HandMotors.grasp()

    set_stage(
        "CONTAINER_HAND_CLOSE_DONE"
    )

    # waypoint完了待ち
    if retreat_started:

        while not waypoint_node.is_finished():
            time.sleep(0.1)

        if waypoint_node.is_failed():
            raise RuntimeError(
                "move_to_container_final.yaml "
                "の実行に失敗しました: "
                f"{waypoint_node.error_message()}"
            )

    set_stage(
        "CONTAINER_RETREAT_DONE"
    )

    print("")
    print(
        "[Move_to_Container] "
        "本の収納・退避完了"
    )

    # =====================================================
    # 8. 今回の収納結果
    #
    # ★ ここではinit.yamlへ戻らない。
    #
    # 統合プログラム側には、
    # Move_to_Container終了後に
    #
    #   init.yaml
    #   ↓
    #   リニアリフトhome
    #   ↓
    #   successログ
    #   ↓
    #   /retrieval_done
    #
    # の共通処理があるため、そちらに任せる。
    # =====================================================

    if container_full:

        print("")
        print("========================================")
        print(
            " 最後の本の収納が完了しました"
        )
        print(
            f" 収納前 : "
            f"{offset:.1f} mm"
        )
        print(
            f" 本の幅 : "
            f"{book_width:.1f} mm"
        )
        print(
            f" 収納後 : "
            f"{next_offset:.1f} mm"
        )
        print(
            f" 満杯判定値 : "
            f"{CONTAINER_FULL_WIDTH_MM:.1f} mm"
        )
        print(
            " コンテナが満杯になりました"
        )
        print(
            " この本を最後に終了します"
        )
        print("========================================")

    else:

        remaining_width = (
            CONTAINER_FULL_WIDTH_MM
            - next_offset
        )

        print("")
        print("========================================")
        print(
            " コンテナ収納完了"
        )
        print(
            f" 収納後offset : "
            f"{next_offset:.1f} mm"
        )
        print(
            f" 満杯まで残り : "
            f"{remaining_width:.1f} mm"
        )
        print("========================================")

    # =====================================================
    # ★ 常に同じ形式で返す
    #
    # 1. 最新のDynamixelオブジェクト
    # 2. コンテナ満杯フラグ
    # =====================================================
    set_stage(
        "CONTAINER_COMPLETE"
    )

    if resume_control["waiting"]:
        raise RuntimeError(
            "resume_stageが"
            "Move_to_Container内にありません: "
            f"{resume_stage}"
        )

    return (
        HandMotors,
        container_full,
    )


# =========================================================
# 単体テスト用 main
# =========================================================

def main():

    rclpy.init()

    node = Node(
        "move_to_container_test"
    )

    arm = XArm7(
        node
    )

    monitor = XArmMonitor(
        arm
    )

    waypoint_node = WaypointPlayerNode(
        node_name="waypoint_player",
        arm=arm,
        yaml_path="",
        monitor=monitor,
    )

    HandMotors = None

    try:

        # =================================================
        # Dynamixel初期化
        # =================================================

        HandMotors = (
            HandBook.init_dynamixels()
        )

        print("")
        print(
            "[DXL] Dynamixel initialized"
        )

        # =================================================
        # コンテナ収納
        # =================================================

        (
            HandMotors,
            container_full,
        ) = Move_to_Container(
            offset=offset,
            book_width=book_width,
            arm=arm,
            waypoint_node=waypoint_node,
            HandMotors=HandMotors,
        )

        # =================================================
        # 結果表示
        # =================================================

        print("")
        print("========================================")
        print(
            " Move_to_Container 正常終了"
        )

        if container_full:

            print(
                " 今回の本の収納によって"
                "コンテナが満杯になりました"
            )

            print(
                " 統合プログラムでは"
                "この本の後処理完了後に終了します"
            )

        else:

            print(
                " まだコンテナに"
                "空きがあります"
            )

        print("========================================")

    # =====================================================
    # xArm / Waypoint / Dynamixel等のエラー
    # =====================================================

    except Exception as e:

        print("")
        print("========================================")
        print(
            " Move_to_Container error"
        )
        print(
            f"{type(e).__name__}: {e}"
        )
        print(
            " Stop program."
        )
        print("========================================")

    finally:

        # =================================================
        # ノード終了
        # =================================================

        try:
            waypoint_node.destroy_node()
        except Exception:
            pass

        try:
            node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    offset = float(
        input(
            "現在のコンテナoffset[mm]? "
        )
        .strip()
        .lower()
        .replace("mm", "")
    )

    book_width = float(
        input(
            "今回のbook_width[mm]? "
        )
        .strip()
        .lower()
        .replace("mm", "")
    )

    main()