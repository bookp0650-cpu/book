#!/usr/bin/env python3
import time
import serial
import minimalmodbus

# ---- ROS2 ----
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

# ==== 通信設定 ====
PORT  = "/dev/ttyIAI"
SLAVE = 1
BAUD  = 115200

# ==== レジスタ ====
REG_PNOW_HI = 0x9000   # 現在位置 (32bit, 0.01mm)
REG_ALMC    = 0x9002
REG_DSS1    = 0x9005

REG_POSR    = 0x0D03
COIL_SON    = 0x0403
COIL_STP    = 0x040A
COIL_CSTR   = 0x040C

# ポジション1
POS_NO   = 1
POS_BASE = 0x1000 + 0x10 * POS_NO


# ===============================
# Modbus 基本
# ===============================
def make_inst():
    inst = minimalmodbus.Instrument(
        PORT,
        SLAVE,
        mode=minimalmodbus.MODE_RTU,
    )

    inst.serial.baudrate = 115200
    inst.serial.bytesize = 8
    inst.serial.parity = serial.PARITY_NONE
    inst.serial.stopbits = 1
    inst.serial.timeout = 2.0

    inst.clear_buffers_before_each_transaction = True
    inst.close_port_after_each_call = False

    # 原因調査中だけ有効化
    inst.debug = True

    return inst


def read_pnow_mm(inst):
    val = inst.read_long(
        REG_PNOW_HI,
        functioncode=3,
        signed=True,
        byteorder=minimalmodbus.BYTEORDER_BIG
    )
    return val * 0.01


def write_coil(inst, addr, on):
    inst.write_bit(
        addr,
        value=1 if on else 0,
        functioncode=5
    )


def pulse(inst, addr, t=0.05):
    write_coil(inst, addr, True)
    time.sleep(t)
    write_coil(inst, addr, False)


# ===============================
# 移動処理
# ===============================
def move_to_mm(inst, target_mm):
    cur = read_pnow_mm(inst)
    print(f"[MOVE] {cur:.2f} mm → {target_mm:.2f} mm")

    # ---- 指令値 ----
    pcmd = int(round(target_mm * 100))   # 0.01mm
    inp  = int(1.0 * 100)                # 1mm
    vcmd = int(100.0 * 100)              # 100mm/s
    acmd = 30                            # 0.30G
    dcmd = 30

    # ---- 元テーブル保存 ----
    old_pcmd = inst.read_long(
        POS_BASE + 0x0,
        functioncode=3,
        signed=True,
        byteorder=minimalmodbus.BYTEORDER_BIG
    )
    old_inp = inst.read_long(
        POS_BASE + 0x2,
        functioncode=3,
        signed=True,
        byteorder=minimalmodbus.BYTEORDER_BIG
    )
    old_vcmd = inst.read_long(
        POS_BASE + 0x4,
        functioncode=3,
        signed=True,
        byteorder=minimalmodbus.BYTEORDER_BIG
    )
    old_acmd = inst.read_register(
        POS_BASE + 0xA,
        number_of_decimals=0,
        functioncode=3
    )
    old_dcmd = inst.read_register(
        POS_BASE + 0xB,
        number_of_decimals=0,
        functioncode=3
    )

    # ---- 新テーブル ----
    inst.write_long(
        POS_BASE + 0x0,
        pcmd,
        signed=True,
        byteorder=minimalmodbus.BYTEORDER_BIG
    )
    inst.write_long(
        POS_BASE + 0x2,
        inp,
        signed=True,
        byteorder=minimalmodbus.BYTEORDER_BIG
    )
    inst.write_long(
        POS_BASE + 0x4,
        vcmd,
        signed=True,
        byteorder=minimalmodbus.BYTEORDER_BIG
    )
    inst.write_register(
        POS_BASE + 0xA,
        acmd,
        number_of_decimals=0,
        functioncode=6
    )
    inst.write_register(
        POS_BASE + 0xB,
        dcmd,
        number_of_decimals=0,
        functioncode=6
    )

    # ---- 実行 ----
    inst.write_register(
        REG_POSR,
        POS_NO,
        number_of_decimals=0,
        functioncode=6
    )
    pulse(inst, COIL_CSTR, 0.05)

    # ---- 監視 ----
    t0 = time.time()
    while True:
        pos = read_pnow_mm(inst)
        err = abs(pos - target_mm)
        print(f"  pos={pos:.2f} mm (err={err:.2f})")

        if err <= 1.0:
            print("[OK] reached")
            break

        if time.time() - t0 > 20:
            print("[TIMEOUT]")
            break

        time.sleep(0.2)

    # ---- 復元 ----
    inst.write_long(
        POS_BASE + 0x0,
        old_pcmd,
        signed=True,
        byteorder=minimalmodbus.BYTEORDER_BIG
    )
    inst.write_long(
        POS_BASE + 0x2,
        old_inp,
        signed=True,
        byteorder=minimalmodbus.BYTEORDER_BIG
    )
    inst.write_long(
        POS_BASE + 0x4,
        old_vcmd,
        signed=True,
        byteorder=minimalmodbus.BYTEORDER_BIG
    )
    inst.write_register(
        POS_BASE + 0xA,
        old_acmd,
        number_of_decimals=0,
        functioncode=6
    )
    inst.write_register(
        POS_BASE + 0xB,
        old_dcmd,
        number_of_decimals=0,
        functioncode=6
    )

    print("[RESTORE] done")


# ===============================
# ROS2 Node
# ===============================
class IaiCylinderNode(Node):
    def __init__(self, inst):
        super().__init__('iai_cylinder_node')
        self.inst = inst
        self.busy = False

        self.create_subscription(
            Float32,
            '/target_mm',
            self.cb_target,
            10
        )

        self.get_logger().info("Ready. Waiting /target_mm")

    def cb_target(self, msg):
        if self.busy:
            self.get_logger().warn("Busy, ignore")
            return

        self.busy = True
        try:
            move_to_mm(self.inst, msg.data)
        except Exception as e:
            self.get_logger().error(str(e))
        finally:
            self.busy = False


# ===============================
# main
# ===============================
def main():
    inst = make_inst()

    print("PNOW =", read_pnow_mm(inst), "mm")
    print("ALMC =", hex(int(inst.read_register(REG_ALMC, number_of_decimals=0, functioncode=3))))
    print("DSS1 =", hex(int(inst.read_register(REG_DSS1, number_of_decimals=0, functioncode=3))))

    write_coil(inst, COIL_STP, False)
    time.sleep(0.2)
    write_coil(inst, COIL_SON, True)
    time.sleep(0.2)

    print("[AFTER SON]")
    print("ALMC =", hex(int(inst.read_register(REG_ALMC, number_of_decimals=0, functioncode=3))))
    print("DSS1 =", hex(int(inst.read_register(REG_DSS1, number_of_decimals=0, functioncode=3))))

    rclpy.init()
    node = IaiCylinderNode(inst)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
