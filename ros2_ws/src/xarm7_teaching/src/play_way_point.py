#!/usr/bin/env python3
# ==========================
# xarm7 モジュールのパスを明示的に追加（最重要）
# ==========================
import sys
sys.path.append("/home/book/pro_book_SAM3/pro_hand_book_python")

# ==========================
# 通常import
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
XARM_HOST = "192.168.2.197"

BASE_DIR = os.path.expanduser(
    "~/pro_book_SAM3/pro_hand_book_python/ros2_ws/src/xarm7_teaching/config"
)

DEFAULT_SPEED = 0.5     # rad/s
DEFAULT_ACCEL = 1.0     # rad/s^2


def deg2rad_list(deg_list):
    return [math.radians(d) for d in deg_list]


class WaypointPlayer(Node):
    def __init__(self, yaml_path: str):
        super().__init__("waypoint_player")

        self.yaml_path = yaml_path

        self.get_logger().info("Connecting to xArm...")
        self.arm = XArm7(self, host=XARM_HOST)

        time.sleep(0.3)  # SDK ready wait

        self.play()

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
                    f"Motion warning at {name}, code=3 (continue async)"
                )
                time.sleep(1.2)  # 少し長め
                continue

            else:
                self.get_logger().error(
                    f"Fatal motion error at {name}, code={ret}"
                )
                break


    # ------------------------
    # 既存のarmを使ってWaypoint再生
    # ------------------------
    @staticmethod
    def play_with_arm(
        arm,
        yaml_path,
        start_name=None,
        end_name=None,
        skip_names=None,
        speed=DEFAULT_SPEED,
        accel=DEFAULT_ACCEL,
        wait=True,
        pause_sec=0.0,
    ):
        """
        既に接続済みのXArm7を使ってWaypointを再生する。

        WaypointPlayer自体は生成しないため、
        新しいROS NodeやxArm接続は作られない。
        """
        skip_names = set(skip_names or [])

        yaml_path = os.path.expanduser(str(yaml_path))

        if not os.path.exists(yaml_path):
            raise FileNotFoundError(
                f"Waypoint YAMLが見つかりません: {yaml_path}"
            )

        with open(
            yaml_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = yaml.safe_load(f)

        if not data:
            raise RuntimeError(
                f"YAMLが空です: {yaml_path}"
            )

        waypoints = data.get("waypoints")

        if not isinstance(waypoints, list):
            raise RuntimeError(
                f"waypointsがありません: {yaml_path}"
            )

        print(
            f"[WaypointPlayer] Loaded "
            f"{len(waypoints)} waypoints from:"
            f"\n  {yaml_path}"
        )

        started = start_name is None
        start_found = start_name is None
        end_found = end_name is None

        executed_names = []

        for wp in waypoints:
            name = wp.get("name", "")

            if not name:
                raise RuntimeError(
                    f"nameがないWaypointがあります: {wp}"
                )

            # start_nameが来るまでは実行しない
            if not started:
                if name != start_name:
                    continue

                started = True
                start_found = True

            # 指定Waypointをスキップ
            if name in skip_names:
                print(
                    f"[WaypointPlayer] Skip {name}"
                )
                continue

            q_deg = wp.get("q")

            if q_deg is None:
                raise RuntimeError(
                    f"{name}にqがありません"
                )

            if len(q_deg) != 7:
                raise RuntimeError(
                    f"{name}の関節数が不正です: "
                    f"{len(q_deg)}"
                )

            q_rad = deg2rad_list(q_deg)

            print(
                f"[WaypointPlayer] Move to {name}"
            )

            ret = arm.arm.set_servo_angle(
                angle=q_rad,
                speed=speed,
                mvacc=accel,
                is_radian=True,
                wait=wait,
            )

            # SDKによってint以外の場合にも対応
            if isinstance(ret, (list, tuple)):
                ret_code = ret[0] if ret else None
            else:
                ret_code = ret

            print(
                f"[WaypointPlayer] "
                f"{name} completed: ret={ret}"
            )

            if ret_code == 3:
                print(
                    f"[WaypointPlayer] "
                    f"Motion warning at {name}, code=3"
                )

            elif isinstance(ret_code, int) and ret_code != 0:
                raise RuntimeError(
                    f"Fatal motion error at "
                    f"{name}, code={ret_code}"
                )

            executed_names.append(name)

            if pause_sec > 0.0:
                time.sleep(pause_sec)

            if end_name is not None and name == end_name:
                end_found = True
                break

        if not start_found:
            raise RuntimeError(
                f"開始Waypointが見つかりません: "
                f"{start_name}"
            )

        if not end_found:
            raise RuntimeError(
                f"終了Waypointが見つかりません: "
                f"{end_name}"
            )

        if not executed_names:
            raise RuntimeError(
                "実行されたWaypointがありません"
            )

        print(
            "[WaypointPlayer] completed: "
            + " -> ".join(executed_names)
        )

        return executed_names



    # ------------------------
    # 終了処理
    # ------------------------
    def shutdown(self):
        try:
            self.arm.disconnect()
        except Exception:
            pass


# ==========================
# YAML選択
# ==========================
def select_yaml():
    if not os.path.isdir(BASE_DIR):
        print(f"Config directory not found:\n  {BASE_DIR}")
        sys.exit(1)

    yamls = sorted(
        [f for f in os.listdir(BASE_DIR) if f.endswith(".yaml")]
    )

    if not yamls:
        print("No YAML files found in config/")
        sys.exit(1)

    print("\nAvailable YAML files:")
    for f in yamls:
        print(f"  - {f}")

    name = input("\nEnter YAML filename to play (without .yaml): ").strip()
    if not name:
        print("No filename entered.")
        sys.exit(1)

    if not name.endswith(".yaml"):
        name += ".yaml"

    path = os.path.join(BASE_DIR, name)

    if not os.path.exists(path):
        print(f"File not found:\n  {path}")
        sys.exit(1)

    return path


# ==========================
# main
# ==========================
def main():
    # --- ROS初期化前にYAML選択 ---
    yaml_path = select_yaml()

    rclpy.init()
    node = None

    try:
        node = WaypointPlayer(yaml_path)

    except KeyboardInterrupt:
        pass

    finally:
        if node:
            node.shutdown()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
