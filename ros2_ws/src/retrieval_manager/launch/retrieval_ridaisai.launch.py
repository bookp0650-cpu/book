
import ast
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


BASE_DIR = "/home/book/pro_book_SAM3/pro_hand_book_python"
PYTHON = BASE_DIR + "/.pro_hand_book_fixed/bin/python3"

INTEGRATION_SCRIPT = (
    Path(BASE_DIR) / "Retrieval_integration_SAM3_ridai_festival.py"
)


def get_lift_failure_mode():
    """
    統合プログラムのLIFT_FAILURE_MODEを読み取る。
    統合プログラム自体は実行しない。
    """
    tree = ast.parse(
        INTEGRATION_SCRIPT.read_text(encoding="utf-8")
    )

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
            )

            for target in targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "LIFT_FAILURE_MODE"
                ):
                    value = ast.literal_eval(node.value)

                    if not isinstance(value, bool):
                        raise ValueError(
                            "LIFT_FAILURE_MODE は True または False にしてください"
                        )

                    return value

    raise RuntimeError(
        f"LIFT_FAILURE_MODE が見つかりません: {INTEGRATION_SCRIPT}"
    )


def generate_launch_description():
    config_path = LaunchConfiguration("config_path")

    lift_failure_mode = get_lift_failure_mode()

    print("========================================")
    print(f"[LAUNCH] LIFT_FAILURE_MODE = {lift_failure_mode}")

    if lift_failure_mode:
        print("[LAUNCH] height_controller 起動スキップ（上下機構故障モード）")
    else:
        print("[LAUNCH] height_controller 起動（通常モード）")

    print("========================================")

    # ==========================================
    # 出庫管理ノード
    # ==========================================
    retrieval_manager = ExecuteProcess(
        cmd=[
            PYTHON,
            "-u",
            BASE_DIR
            + "/ros2_ws/src/retrieval_manager/retrieval_manager/"
            + "retrieval_list_trigger_ridaisai.py",
            "--ros-args",
            "-p",
            ["config_path:=", config_path],
            "-p",
            "initial_wait_sec:=2.0",
            "-p",
            "after_shelf_id_wait_sec:=0.5",
        ],
        name="retrieval_list_trigger_ridaisai",
        output="screen",
    )

    # ==========================================
    # 出庫統合プログラム
    # ==========================================
    retrieval_integration = ExecuteProcess(
        cmd=[
            "bash",
            "-lc",
            "exec "
            + PYTHON
            + " -u "
            + str(INTEGRATION_SCRIPT)
            + " < /dev/tty",
        ],
        cwd=BASE_DIR,
        name="retrieval_integration_ridaisai",
        output="screen",
        emulate_tty=True,
        respawn=True,
        respawn_delay=2.0,
    )

    # ==========================================
    # 起動するノード一覧
    # ==========================================
    actions = [
        DeclareLaunchArgument(
            "config_path",
            default_value=BASE_DIR + "/Retrieval_integration.yaml",
            description="Path to Retrieval_integration.yaml",
        ),
        retrieval_manager,
    ]

    # 上下機構が正常な場合のみ起動
    if not lift_failure_mode:
        actions.append(
            Node(
                package="iai_cylinder",
                executable="height_controller",
                name="height_controller",
                output="screen",
            )
        )

    # 統合プログラムは故障モードでも起動
    actions.append(
        TimerAction(
            period=2.0,
            actions=[retrieval_integration],
        )
    )

    return LaunchDescription(actions)
