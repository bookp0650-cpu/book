#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


# ============================================================
# 固定パス
# ============================================================

PYTHON = (
    "/home/book/pro_book/pro_hand_book_python/"
    ".pro_hand_book_fixed/bin/python3"
)

BATTERY_NODE = (
    "/home/book/pro_book_SAM3/pro_hand_book_python/"
    "battery_monitor/litime_battery_node.py"
)

BATTERY_UI = (
    "/home/book/pro_book_SAM3/pro_hand_book_python/"
    "battery_monitor/battery_ui.py"
)


def generate_launch_description():

    use_ui = LaunchConfiguration("ui")

    return LaunchDescription([

        # ----------------------------------------------------
        # Launch arguments
        # ----------------------------------------------------

        DeclareLaunchArgument(
            "ui",
            default_value="true",
            description="Launch battery monitor GUI"
        ),

        LogInfo(
            msg="Starting LiTime battery monitor..."
        ),

        # ----------------------------------------------------
        # LiTime BLE -> ROS2
        # ----------------------------------------------------

        ExecuteProcess(
            cmd=[
                PYTHON,
                BATTERY_NODE,
            ],
            output="screen",
            respawn=True,
            respawn_delay=3.0,
        ),

        # ----------------------------------------------------
        # ROS2 -> GUI
        # ----------------------------------------------------

        ExecuteProcess(
            cmd=[
                PYTHON,
                BATTERY_UI,
            ],
            output="screen",
            condition=IfCondition(use_ui),
        ),
    ])
