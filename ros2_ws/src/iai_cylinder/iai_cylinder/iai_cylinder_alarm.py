#!/usr/bin/env python3
import time
import serial
import minimalmodbus

import rclpy
from rclpy.node import Node

# ===============================
# 通信設定（height_controllerと同じ）
# ===============================
PORT  = "/dev/ttyUSB1"
SLAVE = 1
BAUD  = 38400

# ===============================
# レジスタ
# ===============================
REG_ALMC = 0x9002
REG_DSS1 = 0x9005

# 簡易アラーム（PM1〜PM8）
REG_PM1 = 0x9008
REG_PM2 = 0x9009
REG_PM4 = 0x900A
REG_PM8 = 0x900B


# ===============================
# 簡易アラーム表（マニュアル準拠）
# ===============================
SIMPLE_ALARM_TABLE = {
    0:  "正常",
    1:  "衝突検出 (0DF)",
    2:  "原点復帰未完了状態での移動指令 / ソフトリセット中移動",
    3:  "原点復帰未完了状態での位置指令 / サーボOFF状態での移動指令",
    4:  "ファン異常 / フィールドバス未検出",
    5:  "ロードセル異常 / フィールドバスリンク異常",
    6:  "パラメータ / ポジションデータ異常",
    7:  "原点センサー未検出 / 原点復帰タイムアウト",
    8:  "実速度過大",
    9:  "過電流 / 過電圧 / 過熱 / 電源異常",
    10: "ソフトウェア過電流",
    11: "ストロークオーバー / 偏差オーバーフロー",
    12: "サーボ異常 / 過負荷",
    13: "エンコーダ異常",
    14: "CPU / ロジック異常",
    15: "不揮発性メモリ異常",
}


# ===============================
# Modbus
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
    ser.timeout = 0.3
    inst.clear_buffers_before_each_transaction = True
    return inst


def read_simple_alarm_code(inst):
    pm1 = inst.read_register(REG_PM1, 0, 3) & 1
    pm2 = inst.read_register(REG_PM2, 0, 3) & 1
    pm4 = inst.read_register(REG_PM4, 0, 3) & 1
    pm8 = inst.read_register(REG_PM8, 0, 3) & 1
    return (pm8 << 3) | (pm4 << 2) | (pm2 << 1) | pm1


# ===============================
# ROS2 Node（表示専用）
# ===============================
class IaiCylinderAlarmNode(Node):
    def __init__(self):
        super().__init__('iai_cylinder_alarm')

        self.inst = make_inst()

        self.timer = self.create_timer(1.0, self.timer_cb)
        self.get_logger().info("IAI Cylinder Alarm Monitor started")

    def timer_cb(self):
        try:
            almc = self.inst.read_register(REG_ALMC, 0, 3)
            dss1 = self.inst.read_register(REG_DSS1, 0, 3)

            if almc == 0:
                self.get_logger().info(
                    f"ALMC=0x0000 (NORMAL) DSS1=0x{dss1:04X}"
                )
                return

            code = read_simple_alarm_code(self.inst)
            reason = SIMPLE_ALARM_TABLE.get(code, "未定義")

            self.get_logger().warn(
                f"ALARM! ALMC=0x{almc:04X} "
                f"SIMPLE_CODE={code} "
                f"REASON={reason} "
                f"DSS1=0x{dss1:04X}"
            )

        except Exception as e:
            self.get_logger().error(str(e))


def main():
    rclpy.init()
    node = IaiCylinderAlarmNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
