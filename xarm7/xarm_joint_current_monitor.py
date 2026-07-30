#!/usr/bin/env python3

import csv
import importlib
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node


# ============================================================
# パス設定
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# /home/book/pro_book_SAM3/pro_hand_book_python
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 設定
# ============================================================

ROBOT_IP = "192.168.2.197"

# 全体の電流記録時間
RECORD_SECONDS = 20.0

# 記録開始から挿入を始めるまでの時間
MOVE_START_DELAY = 2.0

# 電流取得周期
# 0.02秒 = 目標50 Hz
SAMPLE_INTERVAL = 0.02

# ターミナルへの表示周期
PRINT_INTERVAL = 0.20

# moveL_to_insert_rightの速度・加速度
INSERT_VELOCITY = 5.0
INSERT_ACCELERATION = 50.0

# J1の電流閾値
# J1が正方向に3.0を超えたら検知
J1_CURRENT_THRESHOLD = 2.0

# 連続して閾値を超える必要があるサンプル数
# 1なら1回超えた瞬間に検知
J1_TRIGGER_COUNT_REQUIRED = 1

# 検知後に戻る関節角度 [deg]
RETURN_JOINT_ANGLES_DEG = [
    88.2,
    -55.2,
    171.3,
    65.1,
    40.1,
    11.6,
    44.4,
]

# 退避時の関節速度・加速度
RETURN_JOINT_SPEED = 10.0
RETURN_JOINT_ACCELERATION = 50.0

# 挿入停止後の待機時間
STOP_WAIT_SECONDS = 0.20

# STOP状態からREADYへ戻した後の待機時間
READY_WAIT_SECONDS = 0.20

# 20秒時点で退避中なら記録を延長する最大時間
MAX_TOTAL_SECONDS = 50.0

# CSV保存先
LOG_DIR = SCRIPT_DIR / "xarm_current_logs"


# ============================================================
# xarm7.py読み込み
# ============================================================

xarm7_module = importlib.import_module("xarm7.control.xarm7")

# xarm7.pyがXARM_HOST定数を利用している場合も上書きする
if hasattr(xarm7_module, "XARM_HOST"):
    xarm7_module.XARM_HOST = ROBOT_IP

if not hasattr(xarm7_module, "XArm7"):
    raise ImportError(
        "xarm7/control/xarm7.pyにXArm7クラスがありません"
    )

XArm7 = xarm7_module.XArm7


# ============================================================
# 補助関数
# ============================================================

def result_to_code(result):
    """
    SDKまたは独自関数の戻り値からエラーコードを取得する。
    """

    if result is None:
        return 0

    if isinstance(result, bool):
        return 0 if result else -1

    if isinstance(result, int):
        return result

    if isinstance(result, (tuple, list)):
        if len(result) >= 1:
            first = result[0]

            if isinstance(first, bool):
                return 0 if first else -1

            if isinstance(first, int):
                return first

    # 戻り値形式が不明な場合は、例外が出ていないため成功扱い
    return 0


def check_result(label, result):
    """
    戻り値がエラーなら例外を発生させる。
    """

    code = result_to_code(result)

    if code != 0:
        raise RuntimeError(
            f"{label}に失敗しました: "
            f"code={code}, result={result}"
        )

    return result


def get_current_values(arm):
    """
    arm.currentsからJ1～J7の電流を取得する。
    """

    values = arm.currents

    if values is None:
        raise RuntimeError("arm.currentsがNoneです")

    values = list(values)

    if len(values) < 7:
        raise RuntimeError(
            "関節電流の要素数が不足しています: "
            f"expected=7, actual={len(values)}, "
            f"values={values}"
        )

    currents = []

    for joint_index, value in enumerate(values[:7]):
        value = float(value)

        if not math.isfinite(value):
            raise RuntimeError(
                f"J{joint_index + 1}の電流値が不正です: "
                f"{value}"
            )

        currents.append(value)

    return currents


def format_currents(currents):
    """
    ターミナル表示用。
    """

    return " ".join(
        f"J{joint_index + 1}={value:+.4f}"
        for joint_index, value in enumerate(currents)
    )


def get_robot_state(arm):
    return getattr(arm, "state", -1)


def get_robot_mode(arm):
    return getattr(arm, "mode", -1)


def get_error_code(arm):
    return getattr(arm, "error_code", -1)


def get_warn_code(arm):
    return getattr(arm, "warn_code", -1)


def get_is_moving(arm):
    """
    xArmが動作中かを確認する。

    取得できない場合はNoneを返す。
    """

    try:
        result = arm.get_is_moving()

        if isinstance(result, bool):
            return result

        if isinstance(result, (tuple, list)):
            if len(result) >= 2:
                code = result_to_code(result)

                if code == 0:
                    return bool(result[1])

        return None

    except Exception:
        return None


def stop_insert_motion(arm):
    """
    現在の挿入動作を停止する。
    """

    print("挿入動作を停止します。")

    result = arm.set_state(4)
    check_result("set_state(4)", result)

    print(f"停止命令結果: {result}")

    time.sleep(STOP_WAIT_SECONDS)


def prepare_return_motion(arm):
    """
    STOP状態から関節退避を実行できる状態に戻す。
    """

    error_code = get_error_code(arm)

    if error_code != 0:
        raise RuntimeError(
            "停止後にxArmエラーが発生しています。"
            "安全のため自動退避しません: "
            f"error_code={error_code}"
        )

    result = arm.motion_enable(enable=True)
    check_result("motion_enable(True)", result)

    result = arm.set_mode(0)
    check_result("set_mode(0)", result)

    result = arm.set_state(0)
    check_result("set_state(0)", result)

    time.sleep(READY_WAIT_SECONDS)

    error_code = get_error_code(arm)

    if error_code != 0:
        raise RuntimeError(
            "READY状態への移行後にエラーが発生しました: "
            f"error_code={error_code}"
        )


def start_return_motion(arm):
    """
    指定した関節角度へ非同期で退避する。
    """

    print(
        "指定関節角度へ退避します: "
        f"{RETURN_JOINT_ANGLES_DEG}"
    )

    result = arm.set_servo_angle(
        angle=RETURN_JOINT_ANGLES_DEG,
        speed=RETURN_JOINT_SPEED,
        mvacc=RETURN_JOINT_ACCELERATION,
        wait=False,
        is_radian=False,
    )

    check_result("退避関節移動", result)

    print(f"退避動作開始結果: {result}")

    return result


def disconnect_robot(robot, arm):
    """
    xArmとの接続を切断する。
    """

    if robot is not None:
        disconnect_method = getattr(robot, "disconnect", None)

        if callable(disconnect_method):
            try:
                disconnect_method()
                return
            except Exception as exc:
                print(f"robot.disconnect()警告: {exc}")

    if arm is not None:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"arm.disconnect()警告: {exc}")


# ============================================================
# メイン処理
# ============================================================

def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    filename_time = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    csv_path = (
        LOG_DIR
        / f"insert_j1_trigger_current_{filename_time}.csv"
    )

    print("=" * 78)
    print("xArm7 挿入動作・J1電流検知・自動退避試験")
    print(f"接続先             : {ROBOT_IP}")
    print(f"記録時間           : {RECORD_SECONDS:.1f} 秒")
    print(f"挿入開始           : 記録開始から{MOVE_START_DELAY:.1f}秒後")
    print(f"挿入速度           : {INSERT_VELOCITY}")
    print(f"挿入加速度         : {INSERT_ACCELERATION}")
    print(f"J1検知条件         : J1 > {J1_CURRENT_THRESHOLD}")
    print(
        f"連続検知数         : "
        f"{J1_TRIGGER_COUNT_REQUIRED}サンプル"
    )
    print(f"退避関節角度       : {RETURN_JOINT_ANGLES_DEG}")
    print(f"退避速度           : {RETURN_JOINT_SPEED} deg/s")
    print(f"取得周期           : {SAMPLE_INTERVAL:.3f} 秒")
    print(f"目標取得周波数     : {1.0 / SAMPLE_INTERVAL:.1f} Hz")
    print(f"CSV保存先          : {csv_path}")
    print("=" * 78)
    print()
    print("2秒後にmoveL_to_insert_right()を開始します。")
    print("挿入中にJ1が2.5を超えると停止して退避します。")
    print()

    robot = None
    arm = None
    ros_node = None

    rclpy_initialized_here = False

    sample_count = 0
    actual_elapsed = 0.0

    motion_phase = "WAIT"

    move_started = False
    move_result = None

    trigger_count = 0
    trigger_detected = False
    trigger_time = None
    trigger_j1_current = None

    return_started = False
    return_completed = False
    return_result = None

    joint_samples = [[] for _ in range(7)]

    try:
        # ====================================================
        # ROS 2初期化
        # ====================================================

        if not rclpy.ok():
            rclpy.init(args=None)
            rclpy_initialized_here = True

        ros_node = Node(
            "xarm_joint_current_monitor"
        )

        # ====================================================
        # XArm7生成
        # XArm7はnode引数が必須
        # ====================================================

        print(
            "XArm7生成引数: "
            f"node={ros_node.get_name()}, "
            f"host={ROBOT_IP}, "
            "is_radian=False"
        )

        robot = XArm7(
            node=ros_node,
            host=ROBOT_IP,
            is_radian=False,
        )

        if not hasattr(robot, "arm"):
            raise AttributeError(
                "XArm7オブジェクトにrobot.armがありません"
            )

        arm = robot.arm

        if not arm.connected:
            raise RuntimeError(
                f"xArmへ接続できませんでした: {ROBOT_IP}"
            )

        if not hasattr(robot, "moveL_to_insert_right"):
            raise AttributeError(
                "XArm7クラスに"
                "moveL_to_insert_right()がありません"
            )

        print(f"xArm接続成功: version={arm.version}")
        print(
            f"state={get_robot_state(arm)}, "
            f"mode={get_robot_mode(arm)}, "
            f"error={get_error_code(arm)}, "
            f"warn={get_warn_code(arm)}"
        )

        # ====================================================
        # 電流レポートへ切り替え
        # ====================================================

        result = arm.set_report_tau_or_i(1)
        check_result("電流レポート切り替え", result)

        result = arm.get_report_tau_or_i()

        if not isinstance(result, (tuple, list)):
            raise RuntimeError(
                "電流レポート確認結果が不正です: "
                f"{result}"
            )

        if len(result) < 2:
            raise RuntimeError(
                "電流レポート確認結果が不足しています: "
                f"{result}"
            )

        report_code = int(result[0])
        report_mode = int(result[1])

        if report_code != 0:
            raise RuntimeError(
                "電流レポート設定の確認に失敗しました: "
                f"code={report_code}"
            )

        if report_mode != 1:
            raise RuntimeError(
                "関節電流レポートになっていません: "
                f"report_mode={report_mode}"
            )

        print("電流レポート設定: OK")
        print("レポート受信待ち: 1秒")

        time.sleep(1.0)

        initial_currents = get_current_values(arm)

        print(
            f"初期電流: {format_currents(initial_currents)}"
        )
        print()
        print("電流記録を開始します。")
        print()

        # ====================================================
        # CSV記録
        # ====================================================

        with csv_path.open(
            mode="w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:

            writer = csv.writer(csv_file)

            writer.writerow([
                "timestamp",
                "elapsed_s",
                "sample_index",
                "motion_phase",
                "event",
                "move_started",
                "trigger_detected",
                "trigger_count",
                "j1_threshold",
                "joint_1_current",
                "joint_2_current",
                "joint_3_current",
                "joint_4_current",
                "joint_5_current",
                "joint_6_current",
                "joint_7_current",
                "robot_state",
                "robot_mode",
                "error_code",
                "warn_code",
            ])

            start_time = time.monotonic()
            next_sample_time = start_time
            next_print_time = start_time
            last_flush_time = start_time
            last_return_check_time = start_time

            while True:
                now = time.monotonic()
                elapsed = now - start_time
                actual_elapsed = elapsed

                event = ""

                # ================================================
                # 記録終了条件
                # ================================================

                if elapsed >= RECORD_SECONDS:
                    # 退避中なら完了まで記録を延長
                    if return_started and not return_completed:
                        if elapsed >= MAX_TOTAL_SECONDS:
                            print()
                            print(
                                "退避動作の完了待ち時間を超えました。"
                            )

                            try:
                                arm.set_state(4)
                            except Exception:
                                pass

                            break
                    else:
                        break

                # ================================================
                # 2秒後に挿入動作開始
                # ================================================

                if (
                    not move_started
                    and elapsed >= MOVE_START_DELAY
                ):
                    print()
                    print("=" * 78)
                    print(
                        f"[{elapsed:.3f}s] "
                        "moveL_to_insert_right()を開始"
                    )

                    move_result = robot.moveL_to_insert_right(
                        velocity=INSERT_VELOCITY,
                        acceleration=INSERT_ACCELERATION,
                        asynchronous=True,
                    )

                    check_result(
                        "moveL_to_insert_right",
                        move_result,
                    )

                    move_started = True
                    motion_phase = "INSERT"
                    event = "INSERT_START"

                    print(f"挿入開始結果: {move_result}")
                    print("=" * 78)
                    print()

                # ================================================
                # 関節電流取得
                # ================================================

                currents = get_current_values(arm)
                j1_current = currents[0]

                for joint_index, value in enumerate(currents):
                    joint_samples[joint_index].append(value)

                # ================================================
                # 挿入中のJ1監視
                # ================================================

                if (
                    motion_phase == "INSERT"
                    and not trigger_detected
                ):
                    if j1_current > J1_CURRENT_THRESHOLD:
                        trigger_count += 1
                    else:
                        trigger_count = 0

                    if (
                        trigger_count
                        >= J1_TRIGGER_COUNT_REQUIRED
                    ):
                        trigger_detected = True
                        trigger_time = elapsed
                        trigger_j1_current = j1_current

                        motion_phase = "STOPPING"
                        event = "J1_THRESHOLD"

                        print()
                        print("!" * 78)
                        print("J1電流閾値を検知しました")
                        print(f"検知時刻 : {elapsed:.6f} 秒")
                        print(f"J1電流   : {j1_current:+.6f}")
                        print(
                            f"閾値     : "
                            f"{J1_CURRENT_THRESHOLD:+.6f}"
                        )
                        print("挿入動作を停止します")
                        print("!" * 78)

                        # 挿入動作停止
                        stop_insert_motion(arm)

                        # STOP状態からREADYへ戻す
                        prepare_return_motion(arm)

                        # 指定関節角度へ退避開始
                        return_result = start_return_motion(arm)

                        return_started = True
                        motion_phase = "RETURN"
                        event = "J1_THRESHOLD|RETURN_START"

                # ================================================
                # 退避完了確認
                # ================================================

                if (
                    return_started
                    and not return_completed
                    and now - last_return_check_time >= 0.20
                ):
                    last_return_check_time = now

                    # 動作開始直後の誤判定を避ける
                    if (
                        trigger_time is not None
                        and elapsed - trigger_time >= 0.50
                    ):
                        moving = get_is_moving(arm)

                        if moving is False:
                            return_completed = True
                            motion_phase = "RETURN_DONE"
                            event = "RETURN_DONE"

                            print()
                            print("=" * 78)
                            print(
                                f"[{elapsed:.3f}s] "
                                "退避動作が完了しました"
                            )
                            print("=" * 78)
                            print()

                # ================================================
                # CSV保存
                # ================================================

                writer.writerow([
                    datetime.now().isoformat(
                        timespec="milliseconds"
                    ),
                    f"{elapsed:.6f}",
                    sample_count,
                    motion_phase,
                    event,
                    int(move_started),
                    int(trigger_detected),
                    trigger_count,
                    f"{J1_CURRENT_THRESHOLD:.8f}",
                    *[
                        f"{value:.8f}"
                        for value in currents
                    ],
                    get_robot_state(arm),
                    get_robot_mode(arm),
                    get_error_code(arm),
                    get_warn_code(arm),
                ])

                sample_count += 1

                # ================================================
                # エラー・接続確認
                # ================================================

                error_code = get_error_code(arm)

                if error_code != 0:
                    raise RuntimeError(
                        "xArmエラーを検出しました: "
                        f"error_code={error_code}"
                    )

                if not arm.connected:
                    raise RuntimeError(
                        "xArmとの接続が切断されました"
                    )

                # ================================================
                # CSVを1秒ごとに確定
                # ================================================

                if now - last_flush_time >= 1.0:
                    csv_file.flush()
                    last_flush_time = now

                # ================================================
                # ターミナル表示
                # ================================================

                if now >= next_print_time:
                    print(
                        f"[{motion_phase:<11}] "
                        f"[{elapsed:6.2f}s] "
                        f"{format_currents(currents)} "
                        f"count={trigger_count}"
                    )

                    next_print_time += PRINT_INTERVAL

                # ================================================
                # 周期調整
                # ================================================

                next_sample_time += SAMPLE_INTERVAL

                sleep_seconds = (
                    next_sample_time
                    - time.monotonic()
                )

                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

                elif sleep_seconds < -SAMPLE_INTERVAL:
                    next_sample_time = time.monotonic()

            csv_file.flush()

    except KeyboardInterrupt:
        print()
        print("Ctrl+Cで試験を中断しました。")

        if arm is not None:
            try:
                arm.set_state(4)
                print("実行中の動作を停止しました。")
            except Exception as exc:
                print(f"停止処理警告: {exc}")

    except Exception as exc:
        print()
        print(f"エラー: {exc}")

        if arm is not None:
            try:
                arm.set_state(4)
                print("安全のため動作を停止しました。")
            except Exception as stop_exc:
                print(f"停止処理警告: {stop_exc}")

        raise

    finally:
        disconnect_robot(robot, arm)

        if ros_node is not None:
            try:
                ros_node.destroy_node()
            except Exception as exc:
                print(f"ROSノード破棄時の警告: {exc}")

        if rclpy_initialized_here and rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception as exc:
                print(f"rclpy終了時の警告: {exc}")

    # ========================================================
    # 結果表示
    # ========================================================

    print()
    print("=" * 78)
    print("試験結果")
    print(f"記録時間       : {actual_elapsed:.3f} 秒")
    print(f"サンプル数     : {sample_count}")
    print(f"挿入開始結果   : {move_result}")
    print(f"J1検知         : {trigger_detected}")

    if trigger_detected:
        print(f"検知時刻       : {trigger_time:.6f} 秒")
        print(
            f"検知時J1電流   : "
            f"{trigger_j1_current:+.6f}"
        )
        print(f"退避開始結果   : {return_result}")
        print(f"退避完了       : {return_completed}")

    if actual_elapsed > 0:
        print(
            f"実効取得周波数 : "
            f"{sample_count / actual_elapsed:.2f} Hz"
        )

    print()
    print("各関節の電流統計")
    print(
        "関節       最小値        最大値        平均値"
        "        標準偏差      最大-最小"
    )

    for joint_index, samples in enumerate(joint_samples):
        if not samples:
            continue

        minimum = min(samples)
        maximum = max(samples)
        mean = statistics.mean(samples)

        if len(samples) >= 2:
            std_dev = statistics.stdev(samples)
        else:
            std_dev = 0.0

        current_range = maximum - minimum

        print(
            f"J{joint_index + 1:<2} "
            f"{minimum:+12.5f} "
            f"{maximum:+12.5f} "
            f"{mean:+12.5f} "
            f"{std_dev:12.5f} "
            f"{current_range:12.5f}"
        )

    print()
    print(f"CSVログ: {csv_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()