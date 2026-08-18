from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_path = LaunchConfiguration("config_path")

    # ==================================================
    # 統合コード
    #
    # /dev/tty をstdinに接続
    # 異常終了したら2秒後に自動再起動
    # ==================================================
    retrieval_integration = ExecuteProcess(
        cmd=[
            "bash",
            "-lc",
            (
                "exec "
                "/home/book/pro_book/pro_hand_book_python/"
                ".pro_hand_book_fixed/bin/python3 "
                "-u "
                "/home/book/pro_book_SAM3/pro_hand_book_python/"
                "Retrieval_integration_SAM3.py "
                "< /dev/tty"
            ),
        ],

        cwd=(
            "/home/book/pro_book_SAM3/"
            "pro_hand_book_python"
        ),

        name="retrieval_integration",

        output="screen",
        emulate_tty=True,

        # 自動再起動
        respawn=True,
        respawn_delay=2.0,
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "config_path",
            default_value=(
                "/home/book/pro_book_SAM3/pro_hand_book_python/"
                "Retrieval_integration.yaml"
            ),
            description="Path to Retrieval_integration.yaml",
        ),

        # ==============================================
        # 出庫manager
        # ==============================================
        Node(
            package="retrieval_manager",
            executable="retrieval_list_trigger_node",
            name="retrieval_list_trigger_node",
            output="screen",
            parameters=[
                {
                    "config_path": config_path,
                    "initial_wait_sec": 2.0,
                    "after_shelf_id_wait_sec": 0.5,
                    "after_navigation_goal_wait_sec": 0.5,
                }
            ],
        ),

        # ==============================================
        # リニアリフト高さ制御ノード
        # ==============================================
        Node(
            package="iai_cylinder",
            executable="height_controller",
            name="height_controller",
            output="screen",
        ),

        # ==============================================
        # 統合コード
        #
        # managerを先に起動させたいので2秒遅延
        # ==============================================
        TimerAction(
            period=2.0,
            actions=[
                retrieval_integration,
            ],
        ),
    ])