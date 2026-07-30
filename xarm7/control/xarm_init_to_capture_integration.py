#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import yaml
import threading
import time
from enum import Enum
from pathlib import Path
from typing import List
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from xarm7.control.shelf_id_manager import ShelfIDManager
from xarm7.control.xarm7 import XArm7
from xarm7.control.xarm_monitor import safe_motion
from std_msgs.msg import Float32


# ==========================
# Utility
# ==========================
def deg2rad_list(deg_list: List[float]) -> List[float]:
    return [math.radians(d) for d in deg_list]


# ==========================
# Waypoint State
# ==========================
class WaypointState(Enum):
    IDLE = 0
    PLAYING = 1
    SUCCEEDED = 2
    FAILED = 3


# ==========================
# WaypointPlayer ROS Node
# ==========================
class WaypointPlayerNode(Node):

    def __init__(
        self,
        *,
        node_name: str,
        arm: XArm7,
        yaml_path: str | Path,
        monitor,  
        speed: float = 1.0,
        accel: float = 1.0,
    ):
        super().__init__(node_name)

        self.arm = arm
        self.yaml_path = Path(yaml_path).expanduser()
        self.speed = speed
        self.accel = accel
        self.monitor = monitor
        self._lock = threading.Lock()
        self._state = WaypointState.IDLE
        self._error_msg: str | None = None
        self.shelf_manager = ShelfIDManager(self)
        # capture_to_init 用トピック
        self.create_subscription(
            Bool,
            "/navigation_goal_final",
            self.goal_cb,
            10,
        )
        # 上下機構ターゲット
        self.target_pub = self.create_publisher(
            Float32,
            "/target_mm",
            10
        )
        self.get_logger().info(
            "WaypointPlayerNode ready (state=IDLE)"
        )

    # ======================
    # Subscriber callback
    # ======================
    def goal_cb(self, msg: Bool):
        if not msg.data:
            return

        # shelf_id がまだ来てなかったら何もしない
        if not self.shelf_manager.is_received():
            self.get_logger().warn("Shelf ID not received yet")
            return

        side = self.shelf_manager.get_side()    
        height = self.shelf_manager.get_height()
        if side == "right":
            yaml_file = "~/pro_book_SAM3/pro_hand_book_python/ros2_ws/src/xarm7_teaching/config/init_to_capture_v2_integration_right.yaml"
        elif side == "left":
            yaml_file = "~/pro_book_SAM3/pro_hand_book_python/ros2_ws/src/xarm7_teaching/config/init_to_capture_v2_integration_left.yaml"
        else:
            self.get_logger().error("Invalid side")
            return

        self.get_logger().info(f"Playing YAML for side: {side}")
        self.publish_target_mm(height)
        self.play_direct(yaml_file)

    # ======================
    # Worker thread
    # ======================
    def _play_thread(self):
        try:
            # --- スレッド開始時に状態を固定 ---
            with self._lock:
                yaml_path = self.yaml_path

            if not yaml_path.exists():
                raise FileNotFoundError(f"YAML not found: {yaml_path}")

            with yaml_path.open("r") as f:
                data = yaml.safe_load(f)

            waypoints = data.get("waypoints", [])
            self.get_logger().info(
                f"Loaded {len(waypoints)} waypoints from {yaml_path}"
            )

            for wp in waypoints:
                name = wp.get("name", "?")
                q_rad = deg2rad_list(wp["q"])

                wait = wp.get("wait", True)
                sleep_t = wp.get("sleep", 0.0)

                self.get_logger().info(
                    f"Move to {name} (wait={wait}, sleep={sleep_t})"
                )
                
                safe_motion(
                    lambda: self.arm.arm.set_servo_angle(
                        angle=q_rad,
                        speed=self.speed,
                        mvacc=self.accel,
                        is_radian=True,
                        wait=wait,
                    ),
                    self.monitor,
                    f"waypoint_{name}"
                )

                if self.monitor.is_abnormal():
                    with self._lock:
                        self._state = WaypointState.FAILED
                        self._error_msg = "Monitor detected abnormal state"
                    return

                if sleep_t > 0.0:
                    time.sleep(sleep_t)

            with self._lock:
                self._state = WaypointState.SUCCEEDED

            self.get_logger().info("Waypoint trajectory succeeded")

        except Exception as e:
            with self._lock:
                self._state = WaypointState.FAILED
                self._error_msg = str(e)

            self.get_logger().error(f"Waypoint failed: {e}")

    # ======================
    # External API
    # ======================
    def play_direct(self, yaml_path: str | Path) -> bool:
        """
        init_to_capture 用の直呼びトリガ
        """
        with self._lock:
            if self._state != WaypointState.IDLE:
                self.get_logger().warn(
                    f"play_direct ignored (state={self._state.name})"
                )
                return False

            self.yaml_path = Path(yaml_path).expanduser()
            self._state = WaypointState.PLAYING
            self._error_msg = None

        self.get_logger().info(
            f"Direct play start: {self.yaml_path}"
        )

        threading.Thread(
            target=self._play_thread,
            daemon=True,
        ).start()
        return True
    
    def publish_target_mm(self, value_mm: float):
        msg = Float32()
        msg.data = float(value_mm)
        self.target_pub.publish(msg)
        self.get_logger().info(f"published /target_mm: {msg.data}")

    def reset(self):
        with self._lock:
            self._state = WaypointState.IDLE
            self._error_msg = None
        self.get_logger().info("Waypoint state reset to IDLE")

    def is_finished(self) -> bool:
        with self._lock:
            return self._state in (
                WaypointState.SUCCEEDED,
                WaypointState.FAILED,
            )

    def is_succeeded(self) -> bool:
        with self._lock:
            return self._state == WaypointState.SUCCEEDED

    def is_failed(self) -> bool:
        with self._lock:
            return self._state == WaypointState.FAILED

    def error_message(self) -> str | None:
        with self._lock:
            return self._error_msg

    # ======================
    # Shutdown
    # ======================
    def destroy_node(self):
        self.get_logger().info("Destroy WaypointPlayerNode")
        super().destroy_node()
