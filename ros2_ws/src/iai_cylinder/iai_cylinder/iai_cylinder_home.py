#!/usr/bin/env python3
import time
import serial
import minimalmodbus

import rclpy
from rclpy.node import Node

# ===============================
# 通信設定（実機に合わせて）
# ===============================
PORT  = "/dev/ttyUSB1"
SLAVE = 1
BAUD  = 38400

# ===============================
# レジスタ
# ===============================
REG_DSS1 = 0x9005

# ===============================
# デバイス制御レジスタ1（DRG1）
# bit 定義はあなたが貼った資料そのまま
# ===============================
COIL_EMG  = 0x040F   # bit15 EMG（通常は触らない）
COIL_SFTY = 0x040E   # bit14
COIL_SON  = 0x040C   # bit12 サーボON
COIL_ALRS = 0x0408   # bit8  アラームリセット
COIL_BKRL = 0x0407   # bit7
COIL_STP  = 0x0405   # bit5  一時停止
COIL_HOME = 0x0404   # bit4  原点復帰
COIL_CSTR = 0x0403   # bit3  位置決め開始（使わない）

# DSS1 ビット
HEND_BIT = 0x0010    # 原点復帰完了

# ===============================
# Modbus 初期化
# ===============================
def make_inst():
    inst = minimalmodbus.Instrument(
        PORT, SLAVE, mode=minimalmodbus.MODE_RTU
    )
    ser = inst.serial
    if ser is None:
        raise RuntimeError("MinimalModbusのシリアルポートが初期化されていません")
    ser.baudrate = BAUD
    ser.bytesize = 8
    ser.parity = serial.PARITY_NONE
    ser.stopbits = 1
    ser.timeout = 0.5
    inst.clear_buffers_before_each_transaction = True
    return inst


def write_coil(inst, addr, on):
    inst.write_bit(addr, 1 if on else 0, functioncode=5)


def pulse(inst, addr, t=0.2):
    write_coil(inst, addr, True)
    time.sleep(t)
    write_coil(inst, addr, False)


# ===============================
# 原点復帰処理
# ===============================
def home_and_wait(inst):
    print("========== HOME START ==========")

    # ① 一時停止解除
    write_coil(inst, COIL_STP, False)
    time.sleep(0.2)

    # ② サーボON
    write_coil(inst, COIL_SON, True)
    time.sleep(0.5)

    # ③ 原点復帰指令（立上りエッジ）
    print("[HOME] pulse")
    pulse(inst, COIL_HOME, 0.3)

    # ④ 完了待ち
    t0 = time.time()
    while True:
        dss1 = inst.read_register(REG_DSS1, 0, 3)
        print(f"DSS1 = 0x{dss1:04X}")

        if dss1 & HEND_BIT:
            print("[HOME DONE]")
            break

        if time.time() - t0 > 30:
            print("[HOME TIMEOUT]")
            break

        time.sleep(0.2)

    print("========== HOME END ==========")


# ===============================
# ROS2 Node（1回実行型）
# ===============================
class IaiCylinderHomeNode(Node):
    def __init__(self):
        super().__init__('iai_cylinder_home')

        self.inst = make_inst()
        self.timer = self.create_timer(0.5, self.run_once)
        self.done = False

        self.get_logger().info("IAI Cylinder HOME node started")

    def run_once(self):
        if self.done:
            return

        try:
            home_and_wait(self.inst)
        except Exception as e:
            self.get_logger().error(str(e))
        finally:
            self.done = True
            self.get_logger().info("HOME sequence finished")
            rclpy.shutdown()


# ===============================
# main
# ===============================
def main():
    rclpy.init()
    node = IaiCylinderHomeNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
