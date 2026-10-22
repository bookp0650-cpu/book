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
import json
from std_msgs.msg import String, Float32
from std_msgs.msg import Bool
from xarm7.control.xarm7 import XArm7


# ==========================
# 設定
# ==========================
#XARM_HOST = "192.168.1.208" # AC controller
XARM_HOST = "192.168.2.197" # DC controller

# YAML は従来どおり ros2_ws 側を読む
BASE_DIR = os.path.expanduser("~/pro_book/pro_hand_book_python/ros2_ws/src/xarm7_teaching/config")
YAML_NAME = "init_to_capture_v2.yaml"

# json はこの絶対パス
LEVEL_MM_JSON_PATH = "/home/book/pro_book_SAM3/pro_hand_book_python/xarm7/config/level_to_mm.json"

DEFAULT_SPEED = 0.5     # rad/s
DEFAULT_ACCEL = 1.0     # rad/s^2
INIT_Q_DEG = [
   0.0,
   0.0,
   90.0,
   155.0,
   180.0,
   115.0,
   180.0
]


def deg2rad_list(deg_list):
    return [math.radians(d) for d in deg_list]


class WaypointPlayer(Node):
    def __init__(self):
        super().__init__("waypoint_player")

        # YAML（waypoints）
        self.yaml_path = os.path.join(BASE_DIR, YAML_NAME)

        # JSON（level→mm）
        self.level_mm_json_path = LEVEL_MM_JSON_PATH
        self.level_to_mm = self.load_level_to_mm(self.level_mm_json_path)

        # ★ try_start 用の状態
        self.shelf_ready = False
        self.goal_ready = False
        self.played = False

        # -------------------------
        # xArm 接続（先）
        # -------------------------
        self.get_logger().info("Connecting to xArm...")
        self.node = rclpy.create_node("xarm_capture_to_init")
        self.arm = XArm7(
            node=self.node,
            host=XARM_HOST
        )
        time.sleep(1.5)  # enable 安定待ち

        # -------------------------
        # init 姿勢へ移動（後）
        # -------------------------
        self.get_logger().info("Auto move to init pose")
        self.move_to_init_pose()

        # -------------------------
        # Subscriber
        # -------------------------
        self.create_subscription(Bool, "/navigation_goal", self.goal_cb, 10)
        self.create_subscription(String, "/shelf_id", self.shelf_id_cb, 10)

        # Publisher
        self.target_mm_pub = self.create_publisher(Float32, "/target_mm", 10)

        self.get_logger().info(
            "Ready. Send /shelf_id and /navigation_goal in any order; "
            "playback starts when both are received."
        )

    # ------------------------
    # json読み込み
    # ------------------------
    def load_level_to_mm(self, json_path: str) -> dict:
        self.get_logger().info(f"Loading level_to_mm json from:\n  {json_path}")

        if not os.path.exists(json_path):
            self.get_logger().error(f"level_to_mm json not found: {json_path}")
            return {}

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            conv = {}
            for k, v in data.items():
                conv[str(k)] = float(v)

            self.get_logger().info(f"Loaded level_to_mm mapping ({len(conv)} entries)")
            return conv

        except Exception as e:
            self.get_logger().error(f"Failed to load json: {json_path}, error={e}")
            return {}

    # ------------------------
    # init姿勢へ移動
    # ------------------------
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
            self.get_logger().error(f"Failed to move init pose, code={ret}")
        else:
            self.get_logger().info("Init pose reached and holding")

    # ------------------------
    # ★ 両方揃ったら開始
    # ------------------------
    def try_start(self):
        if self.played:
            return

        if not self.shelf_ready:
            return

        if not self.goal_ready:
            return

        self.played = True
        self.get_logger().info("All conditions satisfied, start playback")
        self.play()

    # ------------------------
    # navigation goal callback
    # ------------------------
    def goal_cb(self, msg: Bool):
        if not msg.data:
            return

        self.goal_ready = True
        self.get_logger().info("Navigation goal received (goal_ready=True)")
        self.try_start()

    # ------------------------
    # shelf_id callback
    # ------------------------
    def shelf_id_cb(self, msg: String):
        """
        shelf_id 例: '2-5-1-1'
        → 3番目の '1' を使って target_mm を決定（json参照）
        """
        try:
            parts = msg.data.split("-")
            level = int(parts[2])  # 3番目
        except Exception:
            self.get_logger().error(f"Invalid shelf_id format: {msg.data}")
            return

        if not self.level_to_mm:
            self.get_logger().error(
                f"level_to_mm mapping is empty. json path: {self.level_mm_json_path}"
            )
            return

        key = str(level)
        if key not in self.level_to_mm:
            self.get_logger().error(
                f"level '{level}' not found in {self.level_mm_json_path}. "
                f"Available keys: {sorted(self.level_to_mm.keys())}"
            )
            return

        self.target_mm = self.level_to_mm[key]
        self.shelf_ready = True

        self.get_logger().info(
            f"shelf_id received: {msg.data}, level={level}, "
            f"target_mm={self.target_mm:.1f} (shelf_ready=True)"
        )

        self.try_start()

    # ------------------------
    # 再生本体
    # ------------------------
    def play(self):
        # --- YAML読み込み ---
        with open(self.yaml_path, "r") as f:
            data = yaml.safe_load(f)

        waypoints = data["waypoints"]

        self.get_logger().info(
            f"Loaded {len(waypoints)} waypoints from:\n  {self.yaml_path}"
        )

        # -------------------------
        # ★ マニピュレータの動作を「完了まで待つ」方式にする
        # -------------------------
        for wp in waypoints:
            name = wp["name"]
            q_rad = deg2rad_list(wp["q"])

            self.get_logger().info(f"Move to {name} (wait=True)")

            ret = self.arm.arm.set_servo_angle(
                angle=q_rad,
                speed=DEFAULT_SPEED,
                mvacc=DEFAULT_ACCEL,
                is_radian=True,
                wait=True     # ★ ここが本質：到達するまで待つ
            )

            # if ret == 0:
            #     # 余裕を少し入れたいなら（不要なら消してOK）
            #     time.sleep(0.2)
            #     continue

            # elif ret == 3:
            #     self.get_logger().warn(
            #         f"Motion warning at {name}, code=3 (continue)"
            #     )
            #     time.sleep(0.2)
            #     continue

            # else:
            #     self.get_logger().error(
            #         f"Fatal motion error at {name}, code={ret}"
            #     )
            #     return  # 失敗したら publish しない

        # -------------------------
        # ★ ここに来た時点で「マニピュレータは全 waypoint 到達済み」
        # → その後に上下機構用の指令を publish
        # -------------------------
        msg = Float32()
        msg.data = float(self.target_mm)
        self.target_mm_pub.publish(msg)
        self.get_logger().info(f"Published /target_mm = {msg.data:.1f} (after arm finished)")

        self.get_logger().info("Trajectory finished")


    def shutdown(self):
        try:
            self.arm.disconnect()
        except Exception:
            pass


def main():
    rclpy.init()
    node = None

    try:
        node = WaypointPlayer()
        # ★ 重要：常駐させて、いつ送っても受け取れるようにする
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
