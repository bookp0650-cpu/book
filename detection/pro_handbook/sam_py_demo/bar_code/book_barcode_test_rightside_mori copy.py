from __future__ import annotations
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import json
from datetime import datetime
import time
import numpy as np
import traceback
import math
import yaml
import Dynamixel_win_pro_hand_book.HandBook_Retrieval as HandBook
from detection.pro_handbook.sam_py_demo.bar_code.web_camera_capture import capture_one_depstech
from detection.pro_handbook.sam_py_demo.bar_code.code_1_pic import barcode_perception
from xarm7.control.xarm7 import XArm7

BOOK_BARCODE_1 = [52.4, -82, 178, 78, 204, 4.6, -61.2]
BOOK_BARCODE_2 = [-36.9, -75.2, 159.3, 80.5, 83.8, 18.8, -34.9]

CONFIG_DIR = "/home/book/pro_book_SAM3/pro_hand_book_python/ros2_ws/src/xarm7_teaching/config"

HandMotors = HandBook.init_dynamixels()

def decoded_to_dict_safe(d) -> dict:
    raw_data = getattr(d, "data", None)
    if isinstance(raw_data, (bytes, bytearray)):
        data = raw_data.decode("utf-8", errors="replace")
    else:
        data = raw_data
    rect = getattr(d, "rect", None)
    rect_dict = None
    if rect is not None:
        rect_dict = {
            "left": int(getattr(rect, "left", 0) or 0),
            "top": int(getattr(rect, "top", 0) or 0),
            "width": int(getattr(rect, "width", 0) or 0),
            "height": int(getattr(rect, "height", 0) or 0),
        }
    poly = []
    for p in (getattr(d, "polygon", None) or []):
        poly.append({"x": int(getattr(p, "x", 0) or 0), "y": int(getattr(p, "y", 0) or 0)})
    q = getattr(d, "quality", -1)
    quality = int(q) if isinstance(q, (int, np.integer)) else -1
    return {
        "data": data,
        "type": getattr(d, "type", None),
        "quality": quality,
        "orientation": getattr(d, "orientation", None),
        "rect": rect_dict,
        "polygon": poly,
    }


def rightside_container_to_init_move(arm: XArm7):
    """
    右側収納後に初期姿勢へ戻る動作
    """
    waypoints = [
        [-62.5, -43.3, 164.2, 108.0, 49.6, -18.4, -25.5]
        #,[0, -4.3, 95.7, 164.6, 263.1, 96.7, 210],
    ]
    for i, q_deg in enumerate(waypoints):
        q_rad = [math.radians(d) for d in q_deg]
        arm.arm.set_servo_angle(
            angle=q_rad, speed=1.0, mvacc=0.5, is_radian=True, wait=True
        )
        print(f"waypoint {i} 完了")
    print("rightside_container_to_init_move 完了")

def move_to_container_rightside(offset: float, arm: XArm7, waypoint_node, HandMotors, side: str = "left"):
    """
    コンテナ収納動作
    side: "left" → 左側用yaml / "right" → 右側用yaml
    offset: 収納済み書籍の累積幅[mm]
    収納後の初期姿勢への戻りも実行する
    """
    
    
    print(f"[move_to_container_rightside] offset: {offset}, side: {side}")
    input("続行するにはEnterキーを押してください")
    arm.moveL_relative([-20, 670.0, 0.0, 0.0, 0.0, 0.0])

    waypoint_node.reset()

    waypoint_node.play_direct(f"{CONFIG_DIR}/container_right.yaml")
    while not waypoint_node.is_finished():
        time.sleep(0.1)

    waypoint_node.reset()


    # offsetに応じてyamlファイルを選択
    if offset < 30.0:
        offset_val = "30.0"
    elif offset < 60.0:
        offset_val = "60.0"
    elif offset < 90.0:
        offset_val = "90.0"
    elif offset < 120.0:
        offset_val = "120.0"
    elif offset < 150.0:
        offset_val = "150.0"
    elif offset < 180.0:
        offset_val = "180.0"
    elif offset < 210.0:
        offset_val = "210.0"
    elif offset < 240.0:
        offset_val = "240.0"
    elif offset < 270.0:
        offset_val = "270.0"
    elif offset < 300.0:
        offset_val = "300.0"
    elif offset < 330.0:
        offset_val = "330.0"
    else:
        print("本がコンテナにいっぱいです")
        # 初期姿勢へ（sideに応じて切り替え）
        if side == "right":
            rightside_container_to_init_move(arm)
        # else:
        #     waypoint_node.reset()
        #     waypoint_node.play_direct(f"{CONFIG_DIR}/init.yaml")
        #     while not waypoint_node.is_finished():
        #         time.sleep(0.1)
        return

    # side に応じてyamlファイル名を切り替え
    if side == "left":
        yaml_file = f"container_offset_{offset_val}.yaml"
    elif side == "right":
        yaml_file = f"container_offset_right_{offset_val}.yaml"
    else:
        print(f"エラー: 不正なside値 '{side}'")
        return

    yaml_path = f"{CONFIG_DIR}/{yaml_file}"
    print(f"[move_to_container_rightside] 使用するyaml: {yaml_file}")

    waypoint_node.play_direct(yaml_path)
    while not waypoint_node.is_finished():
        time.sleep(0.1)
    waypoint_node.reset()

    # Z方向の収納動作
    BOOK_CAPTURE = -80.0
    CONTAINER_TILT_DEG = 13.0
    theta = math.radians(CONTAINER_TILT_DEG)
    z_drop = BOOK_CAPTURE + offset * math.tan(theta)
    print(f"[move_to_container_rightside] z_drop: {z_drop}")
    arm.moveL_z_offset(z_drop)

    HandBook.open_until_full(HandMotors, asynchronous=False)

    waypoint_node.reset()
    # waypoint_node.play_direct(f"{CONFIG_DIR}/move_to_container_final.yaml")
    HandBook.grasp(HandMotors)
    print(1)

    arm.moveL_z_offset(-z_drop)
    print(2)
    input("収納完了 → 続行するにはEnterキーを押してください")


    # 収納後に初期姿勢へ（sideに応じて切り替え）
    # if side == "left":
    rightside_container_to_init_move(arm)
    print(3)

    # else:
    #     waypoint_node.reset()
    #     waypoint_node.play_direct(f"{CONFIG_DIR}/init.yaml")
    #     while not waypoint_node.is_finished():
    #         time.sleep(0.1)

    print("[move_to_container_rightside] 収納・初期姿勢復帰完了")


def book_barcode_sequence(barcode_number_input: str, shot_dir: Path, arm: XArm7, move_to_pose: bool = True) -> str:
    """
    - JSON保存で落ちても例外ログを shot_dir に残す
    - 途中で落ちてもロボットをできるだけ初期姿勢へ戻す
    - 戻り値は str ("success" / "no_barcode" / "wrong_barcode" / "error")
    - move_to_pose=False にすると姿勢移動をスキップ（右側認識時に使用）
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    barcode_identification = False
    barcode_data_output = None
    log_path = shot_dir / f"book_barcode_result_{ts}.json"
    err_path = shot_dir / f"book_barcode_error_{ts}.txt"

    try:
        if move_to_pose:
            arm.switch_gripper_pose(BOOK_BARCODE_1)
            arm.moveL_relative([-50.0, 630.0, 50.0, 0.0, 0.0, 0.0])

        time.sleep(2.0)

        frame = capture_one_depstech(shot_dir / "book_barcode_capture.png")
        color_frame = np.asanyarray(frame)

        barcode_identification, barcode_data_output = barcode_perception(barcode_number_input, color_frame)
        print("[book barcode] barcode_identification:", barcode_identification)
        print("[book barcode] barcode_data_output:", barcode_data_output)

        payload = {
            "timestamp": ts,
            "barcode_number_input": barcode_number_input,
            "barcode_identification": bool(barcode_identification),
            "barcode_data_output": [decoded_to_dict_safe(d) for d in (barcode_data_output or [])],
        }
        s = json.dumps(payload, ensure_ascii=False, indent=2)
        log_path.write_text(s, encoding="utf-8")
        print("[book barcode] saved:", str(log_path))

        if barcode_identification:
            return "success"
        elif not barcode_data_output:
            return "no_barcode"
        else:
            return "wrong_barcode"

    except Exception:
        try:
            err_path.write_text(traceback.format_exc(), encoding="utf-8")
            print("[book barcode] ERROR saved:", str(err_path))
        except Exception as e:
            print("[book barcode] ERROR while saving error log:", repr(e))
        return "error"


import rclpy
from rclpy.node import Node

class DummyNode(Node):
    def __init__(self):
        super().__init__("book_barcode_node")


def main():
    rclpy.init()
    node = DummyNode()
    arm = XArm7(node)

    HandMotors = HandBook.init_dynamixels()

    from xarm7.control.xarm_init_to_capture_integration import WaypointPlayerNode
    from xarm7.control.xarm_monitor import XArmMonitor
    monitor = XArmMonitor(arm)
    waypoint_node = WaypointPlayerNode(
        node_name="waypoint_player",
        arm=arm,
        yaml_path="",
        monitor=monitor
    )

    offset = float(input("offset[mm]? ").strip())

    barcode_number_input = "1234567890"
    shot_dir = Path("./logs")
    shot_dir.mkdir(parents=True, exist_ok=True)

    result = book_barcode_sequence(barcode_number_input, shot_dir, arm)
    result = "no_barcode"  # 右側テスト時は有効 / 左側テスト時はコメントアウト
    print("RESULT:", result)

    if result == "success":
        # 左側バーコード認識成功 → 左側用コンテナ収納
        print("左側バーコード認識成功 → コンテナ収納へ")
        move_to_container_rightside(
            offset=offset,
            arm=arm,
            waypoint_node=waypoint_node,
            HandMotors=HandMotors,
            side="left"
        )

    elif result == "no_barcode":
        print("no_barcode → 右面動作へ")

        # -y方向に引く
        #arm.moveL_relative([79.2, -874.6, 30.0, 0.0, 0.0, 0.0])#z=112.0
        
        arm.moveL_relative([79.2, -500, -30.0, 0.0, 0.0, 0.0])
        yaml_path = f"{CONFIG_DIR}/leftside_to_rightside_v1.yaml"
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # p1からp3までwaypoint再生
        for wp in data["waypoints"][1:]:
            if wp["name"] == "p1":
                print("p1はスキップ（直線移動に置き換え済み）")
                continue
            q_rad = [math.radians(d) for d in wp["q"]]
            arm.arm.set_servo_angle(
                angle=q_rad, speed=1.0, mvacc=1.0, is_radian=True, wait=True
            )
            print(f"{wp['name']} 完了")
            if wp["name"] == "p3":
                print("yamlファイルの再生がp3まで完了")
                break
            
        
        #arm.moveL_relative([-20, 50.0, -60.0, 0.0, 0.0, 0.0])
        
        

        # 右側バーコード認識
        print("右側バーコード認識開始")

        # テスト用：強制success
        result_right = "success"
        print("[TEST] 強制success")

        # 本番用（テスト完了後にコメントアウトを外す）
        # result_right = book_barcode_sequence(barcode_number_input, shot_dir, arm, move_to_pose=False)

        print("右側認識結果:", result_right)

        if result_right == "success":
            print("右側バーコード認識成功 → コンテナ収納へ")
            move_to_container_rightside(
                offset=offset,
                arm=arm,
                waypoint_node=waypoint_node,
                HandMotors=
                HandMotors,
                side="right"
            )

        # 本番用分岐（テスト完了後にコメントアウトを外す）
        # elif result_right in ("wrong_barcode", "no_barcode", "error"):
        #     print("右側も認識失敗 → 棚に戻す")
        #     pass

        

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
