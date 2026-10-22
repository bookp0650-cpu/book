#!/usr/bin/env python3
# ==========================
# xarm7 モジュールのパスを明示的に追加
# ==========================
import sys
sys.path.append("/home/book/pro_book_SAM3/pro_hand_book_python")

# ==========================
# import
# ==========================
import rclpy
from rclpy.node import Node
import yaml
import math
import time
import os

from std_msgs.msg import Bool
from xarm7.control.xarm7 import XArm7


# ==========================
# 設定
# ==========================
# XARM_HOST = "192.168.1.208" #AC
XARM_HOST = "192.168.2.197" # DC controller
BASE_DIR = os.path.expanduser(
    "~/pro_book_SAM3/pro_hand_book_python/ros2_ws/src/xarm7_teaching/config"
)

YAML_NAME = "capture_to_init_v2.yaml"

DEFAULT_SPEED = 1.0    # rad/s
DEFAULT_ACCEL = 0.5     # rad/s^2


def deg2rad_list(deg_list):
    return [math.radians(d) for d in deg_list]


class WaypointPlayer(Node):
    def __init__(self):
        super().__init__("waypoint_player")

        self.yaml_path = os.path.join(BASE_DIR, YAML_NAME)

        self.get_logger().info("Connecting to xArm...")
        self.node = rclpy.create_node("xarm_capture_to_init")
        self.arm = XArm7(
            node=self.node,
            host=XARM_HOST
        )

        time.sleep(1.5)

        self.get_logger().info("Auto start capture_to_init")
        self.play()   # ★ これがないと永遠に動かない
    # ------------------------
    # 再生本体
    # ------------------------
    def play(self):
        with open(self.yaml_path, "r") as f:
            data = yaml.safe_load(f)

        waypoints = data["waypoints"]

        self.get_logger().info(
            f"Loaded {len(waypoints)} waypoints from:\n  {self.yaml_path}"
        )

        for wp in waypoints:
            name = wp["name"]
            q_rad = deg2rad_list(wp["q"])

            self.get_logger().info(f"Move to {name}")

            ret = self.arm.arm.set_servo_angle(
                angle=q_rad,
                speed=DEFAULT_SPEED,
                mvacc=DEFAULT_ACCEL,
                is_radian=True,
                wait=False   # ★ 非同期
            )

            if ret == 0:
                time.sleep(1.0)
                continue

            elif ret == 3:
                self.get_logger().warn(
                    f"Motion warning at {name}, code=3 (continue)"
                )
                time.sleep(1.2)
                continue

            else:
                self.get_logger().error(
                    f"Fatal motion error at {name}, code={ret}"
                )
                break

        self.get_logger().info("Trajectory finished")

    # ------------------------
    # 終了処理
    # ------------------------
    def shutdown(self):
        try:
            self.arm.disconnect()
        except Exception:
            pass


# ==========================
# main
# ==========================
def main():
    rclpy.init()
    node = None

    try:
        node = WaypointPlayer()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node:
            node.shutdown()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
