#!/usr/bin/env python3

import subprocess
import threading
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, Float32, String

from litime_ble import BatteryClient


# ============================================================
# LiTime Battery Settings
# ============================================================

BATTERY_ADDRESS = "C8:47:80:45:A8:03"
BATTERY_NAME = "L-24050BNNA70-B04935"

# バッテリー取得周期 [s]
POLL_INTERVAL = 3.0

# Bluetooth OFF時間 [s]
BT_OFF_WAIT = 1.0

# Bluetooth ON後の待機時間 [s]
BT_ON_WAIT = 3.0


class LiTimeBatteryNode(Node):

    def __init__(self):
        super().__init__("litime_battery_node")

        # ====================================================
        # ROS Publishers
        # ====================================================

        self.battery_pub = self.create_publisher(
            BatteryState,
            "/battery_state",
            10
        )

        self.power_pub = self.create_publisher(
            Float32,
            "/battery/power",
            10
        )

        self.connected_pub = self.create_publisher(
            Bool,
            "/battery/connected",
            10
        )

        self.error_pub = self.create_publisher(
            String,
            "/battery/error",
            10
        )

        # ====================================================
        # Shared data
        # ====================================================

        self.lock = threading.Lock()

        self.latest_status = None
        self.connected = False
        self.error_text = ""

        self.data_seq = 0
        self.last_published_seq = -1

        self.stop_event = threading.Event()

        # ====================================================
        # BLE Worker
        # ====================================================

        self.worker = threading.Thread(
            target=self.battery_worker,
            daemon=True
        )

        self.worker.start()

        # ====================================================
        # ROS Timer
        # ====================================================

        self.timer = self.create_timer(
            0.1,
            self.publish_messages
        )

        self.get_logger().info(
            f"LiTime battery monitor started: "
            f"{BATTERY_NAME} ({BATTERY_ADDRESS})"
        )

    # ========================================================
    # Bluetooth reset
    # ========================================================

    def reset_bluetooth(self):

        if self.stop_event.is_set():
            return False

        self.get_logger().warning(
            "Resetting Bluetooth adapter..."
        )

        try:
            # 念のためスキャン停止
            subprocess.run(
                ["bluetoothctl", "scan", "off"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )

        except Exception:
            pass

        try:
            # Bluetooth OFF
            subprocess.run(
                ["bluetoothctl", "power", "off"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )

            if self.stop_event.wait(BT_OFF_WAIT):
                return False

            # Bluetooth ON
            subprocess.run(
                ["bluetoothctl", "power", "on"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )

            if self.stop_event.wait(BT_ON_WAIT):
                return False

            if not self.stop_event.is_set():
                self.get_logger().info(
                    "Bluetooth reset completed"
                )

            return True

        except Exception as e:

            if not self.stop_event.is_set():
                self.get_logger().error(
                    f"Bluetooth reset failed: {e}"
                )

            return False

    # ========================================================
    # BLE Worker
    # ========================================================

    def battery_worker(self):

        # ----------------------------------------------------
        # 起動時にBluetoothを一度リセット
        # ----------------------------------------------------

        self.reset_bluetooth()

        while not self.stop_event.is_set():

            try:
                if not self.stop_event.is_set():
                    self.get_logger().info(
                        f"Connecting BLE: {BATTERY_ADDRESS}"
                    )

                client = BatteryClient(
                    address=BATTERY_ADDRESS
                )

                # litime-ble公式CLIと同じ読み出し方式
                status = client.read_once()

                if self.stop_event.is_set():
                    break

                with self.lock:
                    self.latest_status = status
                    self.connected = True
                    self.error_text = ""
                    self.data_seq += 1

                self.get_logger().info(
                    f"BLE read OK | "
                    f"SOC={status.soc_percent:.1f}% "
                    f"V={status.voltage_v:.3f}V "
                    f"I={status.current_a:+.3f}A "
                    f"P={status.power_w:+.1f}W "
                    f"state={status.charge_state.value}"
                )

                # 正常ならBluetoothリセットしない
                if self.stop_event.wait(POLL_INTERVAL):
                    break

            except Exception as e:

                if self.stop_event.is_set():
                    break

                error_text = (
                    f"{type(e).__name__}: {e}"
                )

                with self.lock:
                    self.connected = False
                    self.error_text = error_text

                self.get_logger().warning(
                    f"LiTime BLE error: {error_text}"
                )

                # ------------------------------------------------
                # 通信失敗
                # ↓
                # Bluetooth OFF → ON
                # ↓
                # LiTime再探索・再接続
                # ------------------------------------------------

                self.reset_bluetooth()

                if self.stop_event.wait(1.0):
                    break

    # ========================================================
    # ROS Publisher
    # ========================================================

    def publish_messages(self):

        with self.lock:
            status = self.latest_status
            connected = self.connected
            error_text = self.error_text
            seq = self.data_seq

        # ----------------------------------------------------
        # Connection
        # ----------------------------------------------------

        connected_msg = Bool()
        connected_msg.data = connected
        self.connected_pub.publish(connected_msg)

        # ----------------------------------------------------
        # Error
        # ----------------------------------------------------

        error_msg = String()
        error_msg.data = error_text
        self.error_pub.publish(error_msg)

        # データ未取得
        if status is None:
            return

        # 同じデータを繰り返しpublishしない
        if seq == self.last_published_seq:
            return

        self.last_published_seq = seq

        # ----------------------------------------------------
        # BatteryState
        # ----------------------------------------------------

        msg = BatteryState()

        msg.header.stamp = self.get_clock().now().to_msg()

        # 電圧
        msg.voltage = float(
            status.voltage_v
        )

        # LiTime:
        # + = charging
        # - = discharging
        msg.current = float(
            status.current_a
        )

        # Ah
        msg.charge = float(
            status.remaining_ah
        )

        msg.capacity = float(
            status.capacity_ah
        )

        msg.design_capacity = float(
            status.capacity_ah
        )

        # ROS BatteryStateは0.0～1.0
        msg.percentage = float(
            status.soc_percent / 100.0
        )

        # BMS温度
        msg.temperature = float(
            status.bms_temp_c
        )

        # ----------------------------------------------------
        # Charge state
        # ----------------------------------------------------

        state = status.charge_state.value

        if state == "charging":

            msg.power_supply_status = (
                BatteryState.POWER_SUPPLY_STATUS_CHARGING
            )

        elif state == "discharging":

            msg.power_supply_status = (
                BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
            )

        elif status.soc_percent >= 99.5:

            msg.power_supply_status = (
                BatteryState.POWER_SUPPLY_STATUS_FULL
            )

        else:

            msg.power_supply_status = (
                BatteryState.POWER_SUPPLY_STATUS_NOT_CHARGING
            )

        # ----------------------------------------------------
        # Battery type
        # ----------------------------------------------------

        msg.power_supply_health = (
            BatteryState.POWER_SUPPLY_HEALTH_UNKNOWN
        )

        msg.power_supply_technology = (
            BatteryState.POWER_SUPPLY_TECHNOLOGY_LIFE
        )

        msg.present = connected

        # ----------------------------------------------------
        # Cell voltages
        # ----------------------------------------------------

        msg.cell_voltage = [
            float(v)
            for v in status.cell_volts_v
        ]

        msg.cell_temperature = []

        msg.location = "robot_battery"
        msg.serial_number = BATTERY_NAME

        self.battery_pub.publish(msg)

        # ----------------------------------------------------
        # Power
        # ----------------------------------------------------

        power_msg = Float32()
        power_msg.data = float(
            status.power_w
        )

        self.power_pub.publish(
            power_msg
        )

    # ========================================================
    # Shutdown
    # ========================================================

    def destroy_node(self):

        self.get_logger().info(
            "Stopping LiTime battery monitor..."
        )

        self.stop_event.set()

        # BLE処理終了を待ってからROS Nodeを破棄
        if self.worker.is_alive():
            self.worker.join(
                timeout=12.0
            )

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = LiTimeBatteryNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
