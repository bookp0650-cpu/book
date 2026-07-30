#!/usr/bin/env python3
import os
import time
import serial
import minimalmodbus

# ---- ROS2 ----
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


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
# コントローラ起動直後で応答が遅い場合にも対応
BAUD_CHECK_ROUNDS = 10

# 通信確立後の通常タイムアウト
NORMAL_TIMEOUT_S = 2.0


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
        if inst.serial is not None and inst.serial.is_open:
            inst.serial.close()
    except Exception:
        pass


def read_pnow_mm(inst):
    val = inst.read_long(
        REG_PNOW_HI,
        functioncode=3,
        signed=True,
        byteorder=minimalmodbus.BYTEORDER_BIG,
    )

    return val * 0.01


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
                # 誤ったBAUDで受信したゴミデータを残さない
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
                # 通信確立後は通常タイムアウトに戻す
                inst.serial.timeout = NORMAL_TIMEOUT_S
                inst.serial.write_timeout = NORMAL_TIMEOUT_S

                # 念のため受信バッファをクリア
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


def write_coil(inst, addr, on):
    inst.write_bit(
        addr,
        value=1 if on else 0,
        functioncode=5,
    )


def pulse(inst, addr, t=0.05):
    write_coil(inst, addr, True)
    time.sleep(t)
    write_coil(inst, addr, False)


# ============================================================
# 移動処理
# ============================================================
def move_to_mm(inst, target_mm):
    cur = read_pnow_mm(inst)

    print(
        f"[MOVE] {cur:.2f} mm → "
        f"{target_mm:.2f} mm"
    )

    # ---- 指令値 ----
    pcmd = int(round(target_mm * 100))   # 0.01 mm
    inp = int(1.0 * 100)                # 1 mm
    vcmd = int(100.0 * 100)             # 100 mm/s
    acmd = 30                            # 0.30 G
    dcmd = 30

    # ---- 元テーブル保存 ----
    old_pcmd = inst.read_long(
        POS_BASE + 0x0,
        functioncode=3,
        signed=True,
        byteorder=minimalmodbus.BYTEORDER_BIG,
    )

    old_inp = inst.read_long(
        POS_BASE + 0x2,
        functioncode=3,
        signed=True,
        byteorder=minimalmodbus.BYTEORDER_BIG,
    )

    old_vcmd = inst.read_long(
        POS_BASE + 0x4,
        functioncode=3,
        signed=True,
        byteorder=minimalmodbus.BYTEORDER_BIG,
    )

    old_acmd = inst.read_register(
        POS_BASE + 0xA,
        number_of_decimals=0,
        functioncode=3,
    )

    old_dcmd = inst.read_register(
        POS_BASE + 0xB,
        number_of_decimals=0,
        functioncode=3,
    )

    try:
        # ---- 新テーブル書き込み ----
        inst.write_long(
            POS_BASE + 0x0,
            pcmd,
            signed=True,
            byteorder=minimalmodbus.BYTEORDER_BIG,
        )

        inst.write_long(
            POS_BASE + 0x2,
            inp,
            signed=True,
            byteorder=minimalmodbus.BYTEORDER_BIG,
        )

        inst.write_long(
            POS_BASE + 0x4,
            vcmd,
            signed=True,
            byteorder=minimalmodbus.BYTEORDER_BIG,
        )

        inst.write_register(
            POS_BASE + 0xA,
            acmd,
            number_of_decimals=0,
            functioncode=6,
        )

        inst.write_register(
            POS_BASE + 0xB,
            dcmd,
            number_of_decimals=0,
            functioncode=6,
        )

        # ---- ポジション番号指定 ----
        inst.write_register(
            REG_POSR,
            POS_NO,
            number_of_decimals=0,
            functioncode=6,
        )

        # ---- 移動開始 ----
        pulse(inst, COIL_CSTR, 0.05)

        # ---- 監視 ----
        t0 = time.time()

        while True:
            pos = read_pnow_mm(inst)
            err = abs(pos - target_mm)

            print(
                f"  pos={pos:.2f} mm "
                f"(err={err:.2f} mm)"
            )

            if err <= 1.0:
                print("[OK] reached")
                break

            if time.time() - t0 > 20:
                print("[TIMEOUT]")
                break

            time.sleep(0.2)

    finally:
        # ---- 元テーブル復元 ----
        print("[RESTORE] restoring position table...")

        inst.write_long(
            POS_BASE + 0x0,
            old_pcmd,
            signed=True,
            byteorder=minimalmodbus.BYTEORDER_BIG,
        )

        inst.write_long(
            POS_BASE + 0x2,
            old_inp,
            signed=True,
            byteorder=minimalmodbus.BYTEORDER_BIG,
        )

        inst.write_long(
            POS_BASE + 0x4,
            old_vcmd,
            signed=True,
            byteorder=minimalmodbus.BYTEORDER_BIG,
        )

        inst.write_register(
            POS_BASE + 0xA,
            old_acmd,
            number_of_decimals=0,
            functioncode=6,
        )

        inst.write_register(
            POS_BASE + 0xB,
            old_dcmd,
            number_of_decimals=0,
            functioncode=6,
        )

        print("[RESTORE] done")


# ============================================================
# ROS2 Node
# ============================================================
class IaiCylinderNode(Node):
    def __init__(self, inst, selected_baud):
        super().__init__("iai_cylinder_node")

        self.inst = inst
        self.selected_baud = selected_baud
        self.busy = False

        self.create_subscription(
            Float32,
            "/target_mm",
            self.cb_target,
            10,
        )

        self.get_logger().info(
            f"Ready. PORT={PORT}, "
            f"BAUD={selected_baud}, "
            "waiting /target_mm"
        )

    def cb_target(self, msg):
        if self.busy:
            self.get_logger().warn(
                "Busy, ignore target command"
            )
            return

        self.busy = True

        try:
            move_to_mm(
                self.inst,
                float(msg.data),
            )

        except Exception as e:
            self.get_logger().error(
                f"{type(e).__name__}: {e}"
            )

        finally:
            self.busy = False


# ============================================================
# main
# ============================================================
def main():
    inst = None
    node = None

    try:
        # ここでBAUDを自動判定
        inst, selected_baud = connect_auto_baud()

        print(
            f"[COMMUNICATION] selected BAUD = "
            f"{selected_baud}"
        )

        # BAUD確定後にステータス確認
        print(
            "PNOW =",
            read_pnow_mm(inst),
            "mm",
        )

        print(
            "ALMC =",
            hex(
                int(
                    inst.read_register(
                        REG_ALMC,
                        number_of_decimals=0,
                        functioncode=3,
                    )
                )
            ),
        )

        print(
            "DSS1 =",
            hex(
                int(
                    inst.read_register(
                        REG_DSS1,
                        number_of_decimals=0,
                        functioncode=3,
                    )
                )
            ),
        )

        # 非常停止入力解除
        write_coil(
            inst,
            COIL_STP,
            False,
        )
        time.sleep(0.2)

        # サーボON
        write_coil(
            inst,
            COIL_SON,
            True,
        )
        time.sleep(0.2)

        print("[AFTER SON]")

        print(
            "ALMC =",
            hex(
                int(
                    inst.read_register(
                        REG_ALMC,
                        number_of_decimals=0,
                        functioncode=3,
                    )
                )
            ),
        )

        print(
            "DSS1 =",
            hex(
                int(
                    inst.read_register(
                        REG_DSS1,
                        number_of_decimals=0,
                        functioncode=3,
                    )
                )
            ),
        )

        rclpy.init()

        node = IaiCylinderNode(
            inst,
            selected_baud,
        )

        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\n[EXIT] KeyboardInterrupt")

    except Exception as e:
        print(
            f"[FATAL] {type(e).__name__}: {e}"
        )

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        close_inst(inst)


if __name__ == "__main__":
    main()