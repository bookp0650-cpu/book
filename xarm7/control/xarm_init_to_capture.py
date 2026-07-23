#!/usr/bin/env python3
# ==========================
# xarm7 モジュールのパスを明示的に追加
# ==========================
import sys
sys.path.append("/home/book/pro_book/pro_hand_book_python")

# ==========================
# import
# ==========================
import rclpy
from rclpy.node import Node
import yaml
import math
import time
import os

from xarm7.control.xarm7 import XArm7


# ==========================
# 設定
# ==========================
# XARM_HOST = "192.168.1.208" # AC controller
XARM_HOST = "192.168.2.197" # DC controller

BASE_DIR = os.path.expanduser(
    "~/pro_book/pro_hand_book_python/ros2_ws/src/xarm7_teaching/config"
)

YAML_NAME = "init_to_capture_v2_integration_right.yaml"

DEFAULT_SPEED = 0.5     # rad/s
DEFAULT_ACCEL = 1.0     # rad/s^2

INIT_Q_DEG = [
    0.0,
   -4.3,
   95.7,
  164.6,
  263.1,
   96.7,
  210.0
]


def deg2rad_list(deg_list):
    return [math.radians(d) for d in deg_list]


class WaypointPlayer(Node):
    def __init__(self):
        super().__init__("waypoint_player")

        self.yaml_path = os.path.join(BASE_DIR, YAML_NAME)
        self.played = False

        # -------------------------
        # xArm 接続
        # -------------------------
        self.get_logger().info("Connecting to xArm...")

        self.arm = XArm7(
            node=self,
            host=XARM_HOST
        )

        time.sleep(1.5)

        # -------------------------
        # init 姿勢へ移動
        # -------------------------
        self.get_logger().info("Auto move to init pose")
        self.move_to_init_pose()

        # -------------------------
        # Enter待ち
        # -------------------------
        input("\nEnter を押すと撮影姿勢へ移動します...")

        self.get_logger().info("Enter pressed, start playback")
        self.played = True
        self.play()

    def move_to_init_pose(self):
        q_rad = deg2rad_list(INIT_Q_DEG)

        self.get_logger().info("Move to fixed init pose")

        ret = self.arm.arm.set_servo_angle(
            angle=q_rad,
            speed=DEFAULT_SPEED,
            mvacc=DEFAULT_ACCEL,
            is_radian=True,
            wait=True
        )

        if ret != 0:
            self.get_logger().error(
                f"Failed to move init pose, code={ret}"
            )
        else:
            self.get_logger().info(
                "Init pose reached and holding"
            )

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
                wait=False
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

    except KeyboardInterrupt:
        pass

    finally:
        if node:
            node.shutdown()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()