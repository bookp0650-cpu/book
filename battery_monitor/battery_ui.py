#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk
from datetime import datetime

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, String


class BatteryUINode(Node):

    def __init__(self):
        super().__init__("battery_ui")

        self.latest_battery = None
        self.connected = False
        self.error_text = ""

        self.create_subscription(
            BatteryState,
            "/battery_state",
            self.battery_callback,
            10
        )

        self.create_subscription(
            Bool,
            "/battery/connected",
            self.connected_callback,
            10
        )

        self.create_subscription(
            String,
            "/battery/error",
            self.error_callback,
            10
        )

    def battery_callback(self, msg):
        self.latest_battery = msg

    def connected_callback(self, msg):
        self.connected = msg.data

    def error_callback(self, msg):
        self.error_text = msg.data


class BatteryUI:

    def __init__(self, node):

        self.node = node

        self.root = tk.Tk()
        self.root.title("LiTime Battery Monitor")
        self.root.geometry("700x650")
        self.root.minsize(620, 580)

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

        # ====================================================
        # Variables
        # ====================================================

        self.soc_var = tk.DoubleVar(value=0.0)

        self.connection_var = tk.StringVar(
            value="接続待ち..."
        )

        self.soc_text = tk.StringVar(
            value="--.- %"
        )

        self.state_text = tk.StringVar(
            value="状態: ---"
        )

        self.voltage_text = tk.StringVar(
            value="--.-- V"
        )

        self.current_text = tk.StringVar(
            value="--.-- A"
        )

        self.power_text = tk.StringVar(
            value="---- W"
        )

        self.capacity_text = tk.StringVar(
            value="--.-- / --.-- Ah"
        )

        self.temperature_text = tk.StringVar(
            value="-- ℃"
        )

        self.cells_text = tk.StringVar(
            value="セル電圧: ---"
        )

        self.updated_text = tk.StringVar(
            value="最終更新: ---"
        )

        self.error_text = tk.StringVar(
            value=""
        )

        self.build_ui()

        # ROS処理とUI更新
        self.root.after(
            100,
            self.process_ros
        )

        self.root.after(
            200,
            self.update_ui
        )

    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):

        main = ttk.Frame(
            self.root,
            padding=20
        )

        main.pack(
            fill="both",
            expand=True
        )

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        title = ttk.Label(
            main,
            text="LiTime Battery Monitor",
            font=("Sans", 22, "bold")
        )

        title.pack(pady=(0, 5))

        connection = ttk.Label(
            main,
            textvariable=self.connection_var,
            font=("Sans", 12)
        )

        connection.pack(pady=(0, 15))

        # ----------------------------------------------------
        # SOC
        # ----------------------------------------------------

        soc_label = ttk.Label(
            main,
            textvariable=self.soc_text,
            font=("Sans", 42, "bold")
        )

        soc_label.pack()

        self.soc_bar = ttk.Progressbar(
            main,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.soc_var
        )

        self.soc_bar.pack(
            fill="x",
            pady=(5, 12)
        )

        state_label = ttk.Label(
            main,
            textvariable=self.state_text,
            font=("Sans", 16, "bold")
        )

        state_label.pack(
            pady=(0, 20)
        )

        # ----------------------------------------------------
        # Electrical values
        # ----------------------------------------------------

        values = ttk.LabelFrame(
            main,
            text="バッテリー情報",
            padding=15
        )

        values.pack(
            fill="x",
            pady=5
        )

        grid = ttk.Frame(values)
        grid.pack(fill="x")

        ttk.Label(
            grid,
            text="電圧",
            font=("Sans", 12)
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=8
        )

        ttk.Label(
            grid,
            textvariable=self.voltage_text,
            font=("Sans", 17, "bold")
        ).grid(
            row=0,
            column=1,
            sticky="e",
            padx=10
        )

        ttk.Label(
            grid,
            text="電流",
            font=("Sans", 12)
        ).grid(
            row=1,
            column=0,
            sticky="w",
            padx=10,
            pady=8
        )

        ttk.Label(
            grid,
            textvariable=self.current_text,
            font=("Sans", 17, "bold")
        ).grid(
            row=1,
            column=1,
            sticky="e",
            padx=10
        )

        ttk.Label(
            grid,
            text="電力",
            font=("Sans", 12)
        ).grid(
            row=2,
            column=0,
            sticky="w",
            padx=10,
            pady=8
        )

        ttk.Label(
            grid,
            textvariable=self.power_text,
            font=("Sans", 17, "bold")
        ).grid(
            row=2,
            column=1,
            sticky="e",
            padx=10
        )

        ttk.Label(
            grid,
            text="残容量",
            font=("Sans", 12)
        ).grid(
            row=0,
            column=2,
            sticky="w",
            padx=25,
            pady=8
        )

        ttk.Label(
            grid,
            textvariable=self.capacity_text,
            font=("Sans", 17, "bold")
        ).grid(
            row=0,
            column=3,
            sticky="e",
            padx=10
        )

        ttk.Label(
            grid,
            text="BMS温度",
            font=("Sans", 12)
        ).grid(
            row=1,
            column=2,
            sticky="w",
            padx=25,
            pady=8
        )

        ttk.Label(
            grid,
            textvariable=self.temperature_text,
            font=("Sans", 17, "bold")
        ).grid(
            row=1,
            column=3,
            sticky="e",
            padx=10
        )

        grid.columnconfigure(
            1,
            weight=1
        )

        grid.columnconfigure(
            3,
            weight=1
        )

        # ----------------------------------------------------
        # Cells
        # ----------------------------------------------------

        cells_frame = ttk.LabelFrame(
            main,
            text="セル電圧",
            padding=15
        )

        cells_frame.pack(
            fill="x",
            pady=10
        )

        ttk.Label(
            cells_frame,
            textvariable=self.cells_text,
            font=("Monospace", 12),
            justify="left"
        ).pack(
            anchor="w"
        )

        # ----------------------------------------------------
        # Bottom
        # ----------------------------------------------------

        ttk.Label(
            main,
            textvariable=self.updated_text
        ).pack(
            pady=(10, 3)
        )

        ttk.Label(
            main,
            textvariable=self.error_text,
            wraplength=600
        ).pack(
            pady=3
        )

    # ========================================================
    # ROS
    # ========================================================

    def process_ros(self):

        if not rclpy.ok():
            return

        rclpy.spin_once(
            self.node,
            timeout_sec=0.0
        )

        self.root.after(
            50,
            self.process_ros
        )

    # ========================================================
    # UI update
    # ========================================================

    def update_ui(self):

        if self.node.connected:
            self.connection_var.set(
                "Bluetooth / ROS2 接続中"
            )
        else:
            self.connection_var.set(
                "Bluetooth切断 / 再接続中"
            )

        msg = self.node.latest_battery

        if msg is not None:

            soc = msg.percentage * 100.0

            self.soc_var.set(soc)

            self.soc_text.set(
                f"{soc:.1f} %"
            )

            self.voltage_text.set(
                f"{msg.voltage:.3f} V"
            )

            self.current_text.set(
                f"{msg.current:+.3f} A"
            )

            power = msg.voltage * msg.current

            self.power_text.set(
                f"{power:+.1f} W"
            )

            self.capacity_text.set(
                f"{msg.charge:.2f} / "
                f"{msg.capacity:.2f} Ah"
            )

            self.temperature_text.set(
                f"{msg.temperature:.0f} ℃"
            )

            # -----------------------------------------------
            # Battery state
            # -----------------------------------------------

            if (
                msg.power_supply_status
                == BatteryState.POWER_SUPPLY_STATUS_CHARGING
            ):
                state = "充電中"

            elif (
                msg.power_supply_status
                == BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
            ):
                state = "放電中"

            elif (
                msg.power_supply_status
                == BatteryState.POWER_SUPPLY_STATUS_FULL
            ):
                state = "満充電"

            elif (
                msg.power_supply_status
                == BatteryState.POWER_SUPPLY_STATUS_NOT_CHARGING
            ):
                state = "待機中"

            else:
                state = "不明"

            self.state_text.set(
                f"状態: {state}"
            )

            # -----------------------------------------------
            # Cell voltages
            # -----------------------------------------------

            if msg.cell_voltage:

                lines = []

                for i in range(
                    0,
                    len(msg.cell_voltage),
                    4
                ):

                    chunk = []

                    for j in range(
                        i,
                        min(
                            i + 4,
                            len(msg.cell_voltage)
                        )
                    ):

                        chunk.append(
                            f"C{j + 1}: "
                            f"{msg.cell_voltage[j]:.3f} V"
                        )

                    lines.append(
                        "    ".join(chunk)
                    )

                # 最大セル差
                spread = (
                    max(msg.cell_voltage)
                    - min(msg.cell_voltage)
                ) * 1000.0

                lines.append("")
                lines.append(
                    f"セル最大差: {spread:.0f} mV"
                )

                self.cells_text.set(
                    "\n".join(lines)
                )

            self.updated_text.set(
                "最終更新: "
                + datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

        if self.node.error_text:
            self.error_text.set(
                "BLE Error: "
                + self.node.error_text
            )
        else:
            self.error_text.set("")

        self.root.after(
            500,
            self.update_ui
        )

    # ========================================================
    # Run / Close
    # ========================================================

    def run(self):
        self.root.mainloop()

    def close(self):

        try:
            self.node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()

        self.root.destroy()


def main():

    rclpy.init()

    node = BatteryUINode()

    ui = BatteryUI(node)

    ui.run()


if __name__ == "__main__":
    main()
