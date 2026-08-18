#!/usr/bin/env python3
import os
import time
import threading

import serial
import minimalmodbus

# ---- ROS2 ----
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import Float32, Bool


# ============================================================
# 通信設定
# ============================================================
PORT = "/dev/ttyIAI"
SLAVE = 1

# 起動時に上から順番に試す
BAUD_CANDIDATES = (
    38400,
    115200,
)

# BAUD確認時は短いタイムアウトで試す
BAUD_CHECK_TIMEOUT_S = 0.5

# 各BAUDでPNOWを読む回数
BAUD_CHECK_RETRIES = 2

# 全BAUD候補を何周するか
BAUD_CHECK_ROUNDS = 10

# 通信確立後の通常タイムアウト
NORMAL_TIMEOUT_S = 2.0


# ============================================================
# ROS2トピック設定
# ============================================================
TARGET_TOPIC = "/target_mm"
ESTOP_TOPIC = "/emergency_stop"

# True:
#   緊急停止時にSTPだけでなくSONもOFFにする
# False:
#   STPのみONにする
SERVO_OFF_ON_ESTOP = True


# ============================================================
# 動作設定
# ============================================================
POSITION_TOLERANCE_MM = 1.0
MOVE_TIMEOUT_S = 20.0
MOVE_MONITOR_INTERVAL_S = 0.05

MOVE_VELOCITY_MM_S = 100.0
MOVE_ACCELERATION_G = 0.30
MOVE_DECELERATION_G = 0.30


# ============================================================
# レジスタ
# ============================================================
REG_PNOW_HI = 0x9000   # 現在位置（32bit, 0.01 mm）
REG_ALMC = 0x9002
REG_DSS1 = 0x9005

REG_POSR = 0x0D03

COIL_SON = 0x0403
COIL_STP = 0x040A
COIL_CSTR = 0x040C

# ポジション1
POS_NO = 1
POS_BASE = 0x1000 + 0x10 * POS_NO


# ============================================================
# スレッド同期
# ============================================================
# MinimalModbus / pyserialを複数スレッドから同時に触らないためのロック。
# RLockにして、ラッパー関数同士がネストしてもデッドロックしないようにする。
MODBUS_LOCK = threading.RLock()


# ============================================================
# 例外
# ============================================================
class EmergencyStopRequested(RuntimeError):
    pass


class MoveTimeoutError(RuntimeError):
    pass


# ============================================================
# Modbus基本処理
# ============================================================
def make_inst(baudrate, timeout_s=NORMAL_TIMEOUT_S, debug=False):
    """
    指定したBAUDでMinimalModbusインスタンスを生成する。
    """
    inst = minimalmodbus.Instrument(
        PORT,
        SLAVE,
        mode=minimalmodbus.MODE_RTU,
    )

    ser = inst.serial

    if ser is None:
        raise RuntimeError(
            "MinimalModbusのシリアルポートが初期化されていません"
        )

    ser.baudrate = int(baudrate)
    ser.bytesize = 8
    ser.parity = serial.PARITY_NONE
    ser.stopbits = 1
    ser.timeout = float(timeout_s)
    ser.write_timeout = float(timeout_s)

    inst.debug = debug
    inst.clear_buffers_before_each_transaction = True
    inst.close_port_after_each_call = False

    return inst


def close_inst(inst):
    """
    Modbusポートを安全に閉じる。
    """
    if inst is None:
        return

    try:
        with MODBUS_LOCK:
            if inst.serial is not None and inst.serial.is_open:
                inst.serial.close()
    except Exception:
        pass


def read_pnow_mm(inst):
    with MODBUS_LOCK:
        val = inst.read_long(
            REG_PNOW_HI,
            functioncode=3,
            signed=True,
            byteorder=minimalmodbus.BYTEORDER_BIG,
        )

    return val * 0.01


def read_register_u16(inst, addr):
    with MODBUS_LOCK:
        return inst.read_register(
            addr,
            number_of_decimals=0,
            functioncode=3,
        )


def write_register_u16(inst, addr, value):
    with MODBUS_LOCK:
        inst.write_register(
            addr,
            int(value),
            number_of_decimals=0,
            functioncode=6,
        )


def read_long_s32(inst, addr):
    with MODBUS_LOCK:
        return inst.read_long(
            addr,
            functioncode=3,
            signed=True,
            byteorder=minimalmodbus.BYTEORDER_BIG,
        )


def write_long_s32(inst, addr, value):
    with MODBUS_LOCK:
        inst.write_long(
            addr,
            int(value),
            signed=True,
            byteorder=minimalmodbus.BYTEORDER_BIG,
        )


def write_coil(inst, addr, on):
    with MODBUS_LOCK:
        inst.write_bit(
            addr,
            value=1 if on else 0,
            functioncode=5,
        )


def pulse(inst, addr, t=0.05):
    """
    コイルを一定時間ONにしてOFFへ戻す。
    パルス途中に別スレッドのModbus通信が割り込まないよう、
    ON -> wait -> OFFを同一ロック内で行う。
    """
    with MODBUS_LOCK:
        inst.write_bit(
            addr,
            value=1,
            functioncode=5,
        )

        time.sleep(t)

        inst.write_bit(
            addr,
            value=0,
            functioncode=5,
        )


# ============================================================
# BAUD自動判定
# ============================================================
def check_baud(baudrate):
    """
    指定BAUDでPNOWを読み取れるか確認する。

    戻り値:
        成功: (inst, pnow_mm)
        失敗: (None, None)
    """
    inst = None

    try:
        inst = make_inst(
            baudrate=baudrate,
            timeout_s=BAUD_CHECK_TIMEOUT_S,
            debug=False,
        )

        for retry_index in range(BAUD_CHECK_RETRIES):
            try:
                if inst.serial is not None:
                    inst.serial.reset_input_buffer()
                    inst.serial.reset_output_buffer()

                time.sleep(0.05)

                pnow_mm = read_pnow_mm(inst)

                print(
                    f"[AUTO BAUD] OK: "
                    f"{baudrate} bps, "
                    f"PNOW={pnow_mm:.2f} mm"
                )

                return inst, pnow_mm

            except minimalmodbus.ModbusException as e:
                print(
                    f"[AUTO BAUD] {baudrate} bps "
                    f"retry {retry_index + 1}/{BAUD_CHECK_RETRIES}: "
                    f"{type(e).__name__}"
                )

            except serial.SerialException as e:
                print(
                    f"[AUTO BAUD] serial error at "
                    f"{baudrate} bps: {e}"
                )
                break

            except Exception as e:
                print(
                    f"[AUTO BAUD] error at "
                    f"{baudrate} bps: "
                    f"{type(e).__name__}: {e}"
                )

            time.sleep(0.1)

    except serial.SerialException as e:
        print(
            f"[AUTO BAUD] cannot open {PORT}: {e}"
        )

    except Exception as e:
        print(
            f"[AUTO BAUD] setup error at "
            f"{baudrate} bps: "
            f"{type(e).__name__}: {e}"
        )

    close_inst(inst)

    return None, None


def connect_auto_baud():
    """
    BAUD候補を順番に試し、PNOWが読めた設定を採用する。
    """
    if not os.path.exists(PORT):
        raise RuntimeError(
            f"{PORT} が存在しません。"
            "USB-RS485変換器とudev設定を確認してください。"
        )

    print("========================================")
    print("[AUTO BAUD] IAI通信速度を確認します")
    print(f"[AUTO BAUD] PORT  : {PORT}")
    print(f"[AUTO BAUD] SLAVE : {SLAVE}")
    print(f"[AUTO BAUD] 候補  : {BAUD_CANDIDATES}")
    print("========================================")

    for round_index in range(BAUD_CHECK_ROUNDS):
        print(
            f"[AUTO BAUD] scan round "
            f"{round_index + 1}/{BAUD_CHECK_ROUNDS}"
        )

        for baudrate in BAUD_CANDIDATES:
            print(
                f"[AUTO BAUD] checking {baudrate} bps..."
            )

            inst, pnow_mm = check_baud(baudrate)

            if inst is not None:
                inst.serial.timeout = NORMAL_TIMEOUT_S
                inst.serial.write_timeout = NORMAL_TIMEOUT_S

                inst.serial.reset_input_buffer()
                inst.serial.reset_output_buffer()

                print("========================================")
                print(
                    f"[AUTO BAUD] CONNECTED: "
                    f"{baudrate} bps"
                )
                print(
                    f"[AUTO BAUD] PNOW: "
                    f"{pnow_mm:.2f} mm"
                )
                print("========================================")

                return inst, baudrate

        if round_index + 1 < BAUD_CHECK_ROUNDS:
            print(
                "[AUTO BAUD] 応答なし。"
                "0.5秒後に再確認します"
            )
            time.sleep(0.5)

    raise RuntimeError(
        "IAIコントローラと通信できませんでした。\n"
        f"PORT={PORT}\n"
        f"SLAVE={SLAVE}\n"
        f"BAUD候補={BAUD_CANDIDATES}\n"
        "コントローラの電源、配線、スレーブ番号、"
        "RS-485のA/B線を確認してください。"
    )


# ============================================================
# IAI状態表示
# ============================================================
def print_iai_status(inst, prefix="STATUS"):
    pnow = read_pnow_mm(inst)
    almc = int(read_register_u16(inst, REG_ALMC))
    dss1 = int(read_register_u16(inst, REG_DSS1))

    print(f"[{prefix}] PNOW = {pnow:.2f} mm")
    print(f"[{prefix}] ALMC = {hex(almc)}")
    print(f"[{prefix}] DSS1 = {hex(dss1)}")


# ============================================================
# 緊急停止処理
# ============================================================
def apply_emergency_stop(inst):
    """
    IAIに停止指令を出す。

    このプログラムでは、元コードの扱いに合わせて
        STP=True  : 停止
        STP=False : 停止解除
    としている。

    SERVO_OFF_ON_ESTOP=Trueの場合は、さらにSON=Falseで
    サーボOFFにする。
    """
    print("\n========================================")
    print("[EMERGENCY STOP] APPLY")
    print("========================================")

    errors = []

    try:
        write_coil(
            inst,
            COIL_STP,
            True,
        )
        print("[E-STOP] STP = ON")
    except Exception as e:
        errors.append(
            f"STP failed: {type(e).__name__}: {e}"
        )

    if SERVO_OFF_ON_ESTOP:
        try:
            write_coil(
                inst,
                COIL_SON,
                False,
            )
            print("[E-STOP] SON = OFF")
        except Exception as e:
            errors.append(
                f"SON OFF failed: {type(e).__name__}: {e}"
            )

    if errors:
        raise RuntimeError(" | ".join(errors))


def release_emergency_stop(inst):
    """
    緊急停止状態を解除する。

    元コードの起動順序と同様に、
        1. STP解除
        2. SON ON
    の順で処理する。
    """
    print("\n========================================")
    print("[EMERGENCY STOP] RELEASE")
    print("========================================")

    write_coil(
        inst,
        COIL_STP,
        False,
    )
    print("[E-STOP] STP = OFF")

    time.sleep(0.2)

    if SERVO_OFF_ON_ESTOP:
        write_coil(
            inst,
            COIL_SON,
            True,
        )
        print("[E-STOP] SON = ON")

    time.sleep(0.2)


def check_emergency_stop(inst, estop_event):
    """
    非常停止フラグが立っていれば、念のためIAIへ停止指令を出し、
    現在の移動処理を例外で終了する。
    """
    if estop_event is None:
        return

    if not estop_event.is_set():
        return

    try:
        apply_emergency_stop(inst)
    except Exception as e:
        raise EmergencyStopRequested(
            "Emergency stop requested, but sending the IAI stop command "
            f"also failed: {type(e).__name__}: {e}"
        ) from e

    raise EmergencyStopRequested(
        "Emergency stop requested"
    )


# ============================================================
# ポジションテーブル退避・復元
# ============================================================
def read_position_table(inst):
    return {
        "pcmd": read_long_s32(
            inst,
            POS_BASE + 0x0,
        ),
        "inp": read_long_s32(
            inst,
            POS_BASE + 0x2,
        ),
        "vcmd": read_long_s32(
            inst,
            POS_BASE + 0x4,
        ),
        "acmd": read_register_u16(
            inst,
            POS_BASE + 0xA,
        ),
        "dcmd": read_register_u16(
            inst,
            POS_BASE + 0xB,
        ),
    }


def restore_position_table(inst, table):
    """
    退避したポジションテーブルを復元する。
    復元途中で1項目失敗しても、残りの項目は可能な限り復元する。
    """
    print("[RESTORE] restoring position table...")

    errors = []

    try:
        write_long_s32(
            inst,
            POS_BASE + 0x0,
            table["pcmd"],
        )
    except Exception as e:
        errors.append(f"PCMD: {type(e).__name__}: {e}")

    try:
        write_long_s32(
            inst,
            POS_BASE + 0x2,
            table["inp"],
        )
    except Exception as e:
        errors.append(f"INP: {type(e).__name__}: {e}")

    try:
        write_long_s32(
            inst,
            POS_BASE + 0x4,
            table["vcmd"],
        )
    except Exception as e:
        errors.append(f"VCMD: {type(e).__name__}: {e}")

    try:
        write_register_u16(
            inst,
            POS_BASE + 0xA,
            table["acmd"],
        )
    except Exception as e:
        errors.append(f"ACMD: {type(e).__name__}: {e}")

    try:
        write_register_u16(
            inst,
            POS_BASE + 0xB,
            table["dcmd"],
        )
    except Exception as e:
        errors.append(f"DCMD: {type(e).__name__}: {e}")

    if errors:
        print("[RESTORE] WARNING:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("[RESTORE] done")


# ============================================================
# 移動処理
# ============================================================
def move_to_mm(inst, target_mm, estop_event=None):
    """
    指定高さへ移動する。

    estop_eventが立った場合は、移動途中でも停止指令を出して
    EmergencyStopRequestedを送出する。
    """
    check_emergency_stop(
        inst,
        estop_event,
    )

    cur = read_pnow_mm(inst)

    print(
        f"[MOVE] {cur:.2f} mm -> "
        f"{target_mm:.2f} mm"
    )

    # ---- 指令値 ----
    pcmd = int(round(target_mm * 100))
    inp = int(round(POSITION_TOLERANCE_MM * 100))
    vcmd = int(round(MOVE_VELOCITY_MM_S * 100))
    acmd = int(round(MOVE_ACCELERATION_G * 100))
    dcmd = int(round(MOVE_DECELERATION_G * 100))

    # ---- 元テーブル保存 ----
    old_table = read_position_table(inst)

    try:
        # 停止要求が来ていたら、動作開始前に中止
        check_emergency_stop(
            inst,
            estop_event,
        )

        # ---- 新テーブル書き込み ----
        write_long_s32(
            inst,
            POS_BASE + 0x0,
            pcmd,
        )

        write_long_s32(
            inst,
            POS_BASE + 0x2,
            inp,
        )

        write_long_s32(
            inst,
            POS_BASE + 0x4,
            vcmd,
        )

        write_register_u16(
            inst,
            POS_BASE + 0xA,
            acmd,
        )

        write_register_u16(
            inst,
            POS_BASE + 0xB,
            dcmd,
        )

        # ---- ポジション番号指定 ----
        write_register_u16(
            inst,
            REG_POSR,
            POS_NO,
        )

        # CSTRを出す直前に最終確認
        check_emergency_stop(
            inst,
            estop_event,
        )

        # ---- 移動開始 ----
        pulse(
            inst,
            COIL_CSTR,
            0.05,
        )

        # ---- 監視 ----
        t0 = time.time()

        while True:
            # 位置を読む前に停止確認
            check_emergency_stop(
                inst,
                estop_event,
            )

            pos = read_pnow_mm(inst)

            # Modbus読込中に停止要求が来る場合があるため、読込後にも確認
            check_emergency_stop(
                inst,
                estop_event,
            )

            err = abs(
                pos - target_mm
            )

            print(
                f"  pos={pos:.2f} mm "
                f"(err={err:.2f} mm)"
            )

            if err <= POSITION_TOLERANCE_MM:
                print("[OK] reached")
                return

            if time.time() - t0 > MOVE_TIMEOUT_S:
                # タイムアウトした状態で動作を継続させない。
                if estop_event is not None:
                    estop_event.set()

                try:
                    apply_emergency_stop(inst)
                except Exception as stop_error:
                    raise MoveTimeoutError(
                        f"Move timeout and stop command failed: {stop_error}"
                    ) from stop_error

                raise MoveTimeoutError(
                    f"Move timeout after {MOVE_TIMEOUT_S:.1f} s"
                )

            # Event.wait()を使うことで、通常のtime.sleep()より
            # E-STOP要求で早く待機解除できる。
            if estop_event is not None:
                estop_event.wait(
                    MOVE_MONITOR_INTERVAL_S
                )
            else:
                time.sleep(
                    MOVE_MONITOR_INTERVAL_S
                )

    finally:
        # 停止後であっても、プログラムが一時変更した位置テーブルは
        # 元へ戻しておく。
        restore_position_table(
            inst,
            old_table,
        )


# ============================================================
# ROS2 Node
# ============================================================
class IaiCylinderNode(Node):
    def __init__(self, inst, selected_baud):
        super().__init__(
            "iai_cylinder_node"
        )

        self.inst = inst
        self.selected_baud = selected_baud

        self.estop_event = threading.Event()

        self.state_lock = threading.Lock()
        self.busy = False

        # /target_mmが長時間動いていても、/emergency_stopを
        # 別スレッドで処理できるよう別callback groupに分離する。
        self.target_callback_group = MutuallyExclusiveCallbackGroup()
        self.estop_callback_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(
            Float32,
            TARGET_TOPIC,
            self.cb_target,
            10,
            callback_group=self.target_callback_group,
        )

        self.create_subscription(
            Bool,
            ESTOP_TOPIC,
            self.cb_emergency_stop,
            10,
            callback_group=self.estop_callback_group,
        )

        self.get_logger().info(
            f"Ready. PORT={PORT}, "
            f"BAUD={selected_baud}"
        )

        self.get_logger().info(
            f"Target topic    : {TARGET_TOPIC} "
            "(std_msgs/msg/Float32)"
        )

        self.get_logger().info(
            f"Emergency topic : {ESTOP_TOPIC} "
            "(std_msgs/msg/Bool)"
        )

        self.get_logger().info(
            "Emergency semantics: "
            "True=STOP, False=RELEASE"
        )

    # --------------------------------------------------------
    # busy状態取得
    # --------------------------------------------------------
    def is_busy(self):
        with self.state_lock:
            return self.busy

    # --------------------------------------------------------
    # 緊急停止
    # --------------------------------------------------------
    def cb_emergency_stop(self, msg):
        # ====================================================
        # True: 緊急停止
        # ====================================================
        if bool(msg.data):
            # まずソフト側の停止フラグを立てる。
            # move_to_mm()もこのフラグを監視している。
            self.estop_event.set()

            self.get_logger().fatal(
                "EMERGENCY STOP RECEIVED"
            )

            try:
                # busyかどうかに関係なく、可能な限りここからも
                # 直ちにSTP/SONへ停止指令を送る。
                apply_emergency_stop(
                    self.inst
                )

                self.get_logger().fatal(
                    "IAI emergency stop applied"
                )

            except Exception as e:
                self.get_logger().error(
                    "Failed to send emergency stop to IAI: "
                    f"{type(e).__name__}: {e}"
                )

            return

        # ====================================================
        # False: 緊急停止解除
        # ====================================================
        if not self.estop_event.is_set():
            self.get_logger().info(
                "Emergency stop is already released"
            )
            return

        if self.is_busy():
            # 移動コールバックが完全終了するまでは解除しない。
            # もう一度Falseを送れば解除できる。
            self.get_logger().warn(
                "Cannot release emergency stop while motion callback "
                "is still active. Send False again after motion aborts."
            )
            return

        try:
            release_emergency_stop(
                self.inst
            )

            self.estop_event.clear()

            self.get_logger().info(
                "Emergency stop released"
            )

        except Exception as e:
            # 解除通信に失敗した場合はestop_eventを残す。
            self.get_logger().error(
                "Failed to release emergency stop: "
                f"{type(e).__name__}: {e}"
            )

    # --------------------------------------------------------
    # 高さ指令
    # --------------------------------------------------------
    def cb_target(self, msg):
        if self.estop_event.is_set():
            self.get_logger().error(
                "Emergency stop is active. Ignore target command."
            )
            return

        with self.state_lock:
            if self.busy:
                self.get_logger().warn(
                    "Busy, ignore target command"
                )
                return

            self.busy = True

        try:
            target_mm = float(msg.data)

            self.get_logger().info(
                f"Target received: {target_mm:.2f} mm"
            )

            move_to_mm(
                self.inst,
                target_mm,
                self.estop_event,
            )

            self.get_logger().info(
                "Motion finished"
            )

        except EmergencyStopRequested as e:
            self.get_logger().fatal(
                f"Motion aborted by emergency stop: {e}"
            )

        except MoveTimeoutError as e:
            self.get_logger().error(
                f"Motion timeout -> emergency stop latched: {e}"
            )

        except Exception as e:
            self.get_logger().error(
                f"{type(e).__name__}: {e}"
            )

        finally:
            with self.state_lock:
                self.busy = False


# ============================================================
# main
# ============================================================
def main():
    inst = None
    node = None
    executor = None

    try:
        # ----------------------------------------------------
        # BAUD自動判定
        # ----------------------------------------------------
        inst, selected_baud = connect_auto_baud()

        print(
            f"[COMMUNICATION] selected BAUD = "
            f"{selected_baud}"
        )

        # ----------------------------------------------------
        # 起動時ステータス
        # ----------------------------------------------------
        print_iai_status(
            inst,
            prefix="BEFORE SON",
        )

        # ----------------------------------------------------
        # 起動時は停止解除 -> サーボON
        # ----------------------------------------------------
        write_coil(
            inst,
            COIL_STP,
            False,
        )
        time.sleep(0.2)

        write_coil(
            inst,
            COIL_SON,
            True,
        )
        time.sleep(0.2)

        print_iai_status(
            inst,
            prefix="AFTER SON",
        )

        # ----------------------------------------------------
        # ROS2
        # ----------------------------------------------------
        rclpy.init()

        node = IaiCylinderNode(
            inst,
            selected_baud,
        )

        # /target_mmの移動処理中でも、/emergency_stopを
        # 別スレッドで受信できるようMultiThreadedExecutorを使用。
        executor = MultiThreadedExecutor(
            num_threads=2
        )

        executor.add_node(node)
        executor.spin()

    except KeyboardInterrupt:
        print("\n[EXIT] KeyboardInterrupt")

    except Exception as e:
        print(
            f"[FATAL] {type(e).__name__}: {e}"
        )

    finally:
        if executor is not None:
            try:
                executor.shutdown()
            except Exception:
                pass

        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass

        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass

        close_inst(inst)


if __name__ == "__main__":
    main()