import json
import os
import os

os.environ["BOOK_CAPTURE_ROOT"] = (
    "/home/book/pro_book_SAM3/"
    "pro_hand_book_python/"
    "captures_strage"
)
# 入庫だけOCR回転角変更
os.environ["BOOK_OCR_ANGLES"] = "0"
from detection.pro_handbook.sam3_runtime.integration_service_manager import (
    Sam3ServiceSession,
)

from detection.pro_handbook.sam_py_demo.get_book_points_sam3_refined_sam2_width import (
    run_capture_and_pca_sam3_refined_sam2_width,
)
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.executors import MultiThreadedExecutor
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool

from xarm7.control.xarm7 import (
    XArm7,
    RETRIEVAL_DX,
    TCP_VEL_1,
    TCP_ACC_1,
)
from xarm7.control.xarm_monitor import XArmMonitor, safe_motion
from xarm7.control.shelf_id_manager import ShelfIDManager
from linear_lift import TargetPublisher
from xarm7.control.robot_base_coordinate import (
    cam_mm_to_robot_mm,
    print_camera_debug_info,
)

from Dynamixel_win_pro_hand_book.dynamixel_worker_client import DynamixelWorkerClient
import threading
from detection.pro_handbook.sam_py_demo.bar_code.book_barcode import (
    identify_book_isbn_at_current_pose,
)

from detection.pro_handbook.sam_py_demo.Storage_SAM3 import run_capture_and_pca_depth_space
from detection.pro_handbook.sam_py_demo.ocr_worker_client import (
    shutdown_ocr_worker,
)


# ============================================================
# PATH / CONFIG
# ============================================================
PROJECT_DIR = Path("/home/book/pro_book_SAM3/pro_hand_book_python")
CONFIG_PATH = PROJECT_DIR / "Retrieval_integration.yaml"

INIT_TO_CONTAINER_CAPTURE_YAML = Path(
    "/home/book/pro_book_SAM3/pro_hand_book_python/"
    "ros2_ws/src/xarm7_teaching/config/init_to_container_capture.yaml"
)

STORAGE_CAPTURE_TO_BARCODE_YAML = Path(
    "/home/book/pro_book_SAM3/pro_hand_book_python/"
    "ros2_ws/src/xarm7_teaching/config/strage_capture_to_barcode.yaml"
)

BARCODE_CAPTURE_YAML = Path(
    "/home/book/pro_book_SAM3/pro_hand_book_python/"
    "ros2_ws/src/xarm7_teaching/config/barcode_to_capture.yaml"
)

# ユーザー指定のコンテナ撮影姿勢 p7
CONTAINER_CAPTURE_P7_DEG = [
   -71.1,
   -46.7,
   186.4,
   63.1,
   -152.9,
   77.1,
   151.7
]

P7_MOVEJ_VELOCITY = 1.0
P7_MOVEJ_ACCELERATION = 2.0
MAX_RECOGNITION_ATTEMPTS = 3

# 実機コンテナ高さ。実際の指令値は必ずconfigから読む。
EXPECTED_CONTAINER_HEIGHT_MM = 1100.0
CONTAINER_HEIGHT_TOLERANCE_MM = 1.0

# JSONは出庫と同じく上から順に処理する。
# 現在の前半テストは先頭(0番目)から開始。
START_BOOK_INDEX = 0
MAX_STORAGE_INSERT_RETRIES = 2

# 書架空きスペース認識
SECOND_CAPTURE_DY_MM = -140.0
STORAGE_SPACE_MOVE_VELOCITY = 40.0
STORAGE_SPACE_MOVE_ACCELERATION = 40.0


# ============================================================
# GLOBALS
# ============================================================
arm = None
lift_estop_pub = None
_hand_worker = None
_sam3_service_session = None
_lift_estop_sent = False


# ============================================================
# CONFIG
# ============================================================
def load_config(config_path: Path):
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config


def resolve_config_relative_path(config, raw_path):
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(config.get("_config_dir", PROJECT_DIR)) / path
    return path.resolve()


def load_books_master(config):
    master_file = resolve_config_relative_path(
        config,
        config["books"]["master_file"],
    )

    print(f"[MASTER] loaded: {master_file}")

    with master_file.open("r", encoding="utf-8") as f:
        books_master = json.load(f)

    if not isinstance(books_master, list):
        raise RuntimeError("books master JSON must be a list")

    return books_master


# ============================================================
# IAI E-STOP
# ============================================================
def emergency_stop_linear_lift():
    global _lift_estop_sent

    if _lift_estop_sent:
        print("[LIFT E-STOP] already sent")
        return True

    pub = globals().get("lift_estop_pub")
    if pub is None:
        print("[LIFT E-STOP] publisher is not initialized")
        return False

    try:
        msg = Bool()
        msg.data = True

        print("[LIFT E-STOP] publishing /emergency_stop=True")

        for _ in range(3):
            pub.publish(msg)
            time.sleep(0.05)

        _lift_estop_sent = True
        print("[LIFT E-STOP] emergency stop command sent")
        return True

    except Exception as e:
        print(f"[LIFT E-STOP] failed: {type(e).__name__}: {e}")
        return False


def release_linear_lift_estop():
    global _lift_estop_sent

    pub = globals().get("lift_estop_pub")
    if pub is None:
        print("[LIFT E-STOP RELEASE] publisher is not initialized")
        return False

    try:
        msg = Bool()
        msg.data = False

        print("\n========================================")
        print("[LIFT E-STOP RELEASE] releasing...")
        print("========================================")

        for _ in range(3):
            pub.publish(msg)
            time.sleep(0.1)

        _lift_estop_sent = False
        print("[LIFT E-STOP RELEASE] /emergency_stop=False sent")
        return True

    except Exception as e:
        print(f"[LIFT E-STOP RELEASE] failed: {type(e).__name__}: {e}")
        return False


# ============================================================
# SYSTEM E-STOP
# ============================================================
def system_emergency_stop(reason: str):
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("[SYSTEM E-STOP]")
    print(f"reason: {reason}")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    try:
        emergency_stop_linear_lift()
    except Exception as e:
        print(f"[SYSTEM E-STOP] lift stop failed: {e}")

    try:
        current_arm = globals().get("arm")
        if current_arm is not None:
            current_arm.emergency_stop()
    except Exception as e:
        print(f"[SYSTEM E-STOP] xArm stop failed: {e}")


def monitor_emergency_stop(msg):
    print(f"[SYSTEM E-STOP] XArmMonitor detected abnormal: {msg}")
    emergency_stop_linear_lift()


def sigint_handler(sig, frame):
    print("\nCtrl+C detected -> EMERGENCY STOP")
    system_emergency_stop("Ctrl+C")

    service_session = globals().get("_sam3_service_session")
    if service_session is not None:
        try:
            service_session.stop_if_owned()
        except Exception as e:
            print(f"[SAM3] cleanup failed: {e}")

    time.sleep(0.2)
    os._exit(1)


signal.signal(signal.SIGINT, sigint_handler)


# ============================================================
# xArm CHECK
# ============================================================
def assert_xarm_normal(node, where: str):
    state = arm.get_state()
    err, warn = arm.get_err_warn()

    node.get_logger().info(
        f"[XARM CHECK] {where}: state={state}, err={err}, warn={warn}"
    )

    if state in (4, 5) or err != 0:
        raise RuntimeError(
            f"xArm abnormal at {where}: "
            f"state={state}, err={err}, warn={warn}"
        )


# ============================================================
# YAML PLAYER
# MoveJ / MoveL / MoveL Relative
# ============================================================
def _ret_code(ret):
    if isinstance(ret, (list, tuple)):
        return ret[0] if ret else None
    return ret


def play_motion_yaml_safely(node, monitor, yaml_path):
    yaml_path = Path(yaml_path).expanduser().resolve()

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or not isinstance(data.get("waypoints"), list):
        raise RuntimeError(f"waypointsがありません: {yaml_path}")

    waypoints = data["waypoints"]

    node.get_logger().info(
        f"[YAML PLAYER] loaded {len(waypoints)} waypoints: {yaml_path}"
    )

    for index, wp in enumerate(waypoints, start=1):
        name = str(wp.get("name", "")).strip()
        if not name:
            raise RuntimeError(f"nameがないWaypointがあります: {wp}")

        move_type = str(wp.get("type", "movej")).strip().lower()
        waypoint_wait = bool(wp.get("wait", True))
        sleep_sec = float(wp.get("sleep", 0.0))

        node.get_logger().info(
            f"[YAML PLAYER] {index}/{len(waypoints)} "
            f"name={name}, type={move_type}, wait={waypoint_wait}"
        )

        if move_type in ("movej", "joint", "j"):
            q_deg = wp.get("q")
            if q_deg is None or len(q_deg) != 7:
                raise RuntimeError(
                    f"{name}: MoveJ qは7要素必要です: {q_deg}"
                )

            q_rad = np.radians(
                [float(v) for v in q_deg]
            ).tolist()

            speed = float(wp.get("speed", 0.5))
            accel = float(wp.get("accel", 1.0))

            def _movej():
                ret = arm.arm.set_servo_angle(
                    angle=q_rad,
                    speed=speed,
                    mvacc=accel,
                    is_radian=True,
                    wait=waypoint_wait,
                )

                code = _ret_code(ret)
                if isinstance(code, int) and code not in (0, 3):
                    raise RuntimeError(
                        f"{name}: MoveJ failed ret={ret}"
                    )
                return ret

            safe_motion(
                _movej,
                monitor,
                f"yaml_movej_{name}",
            )

        elif move_type in ("movel", "linear", "l"):
            pose = wp.get("pose")
            if pose is None or len(pose) != 6:
                raise RuntimeError(
                    f"{name}: MoveL poseは6要素必要です: {pose}"
                )

            x, y, z, roll, pitch, yaw = [
                float(v) for v in pose
            ]

            speed = float(wp.get("speed", 50.0))
            accel = float(wp.get("accel", 200.0))

            def _movel():
                ret = arm.arm.set_position(
                    x=x,
                    y=y,
                    z=z,
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                    speed=speed,
                    mvacc=accel,
                    is_radian=False,
                    wait=waypoint_wait,
                )

                code = _ret_code(ret)
                if isinstance(code, int) and code not in (0, 3):
                    raise RuntimeError(
                        f"{name}: MoveL failed ret={ret}"
                    )
                return ret

            safe_motion(
                _movel,
                monitor,
                f"yaml_movel_{name}",
            )

        elif move_type in (
            "movel_relative",
            "movel-relative",
            "move_l_relative",
            "linear_relative",
            "relative",
            "lr",
        ):
            pose = wp.get("pose", wp.get("delta"))
            if pose is None or len(pose) != 6:
                raise RuntimeError(
                    f"{name}: MoveL Relative poseは6要素必要です: {pose}"
                )

            relative_pose = [float(v) for v in pose]
            velocity = float(wp.get("speed", 50.0))
            acceleration = float(wp.get("accel", 200.0))

            # 相対移動は現在地基準なので、既定では必ず同期実行。
            # YAML wait:false を明示した時だけ asynchronous=True。
            safe_motion(
                lambda: arm.moveL_relative(
                    relative_pose,
                    velocity=velocity,
                    acceleration=acceleration,
                    asynchronous=(not waypoint_wait),
                ),
                monitor,
                f"yaml_movel_relative_{name}",
            )

        else:
            raise RuntimeError(
                f"{name}: 不明なmotion typeです: {move_type}"
            )

        # 非同期を指定したWaypoint直後に相対移動を置くと危険なので、
        # 今回の入庫YAMLでは基本 wait:true を推奨。
        assert_xarm_normal(node, f"after yaml waypoint {name}")

        if sleep_sec > 0.0:
            time.sleep(sleep_sec)

    node.get_logger().info("[YAML PLAYER] completed")


# ============================================================
# P7
# ============================================================
def move_to_container_capture_p7(node, monitor):
    q_rad = np.radians(CONTAINER_CAPTURE_P7_DEG).tolist()

    node.get_logger().info("[P7] move to container capture pose")

    def _motion():
        ret = arm.arm.set_servo_angle(
            angle=q_rad,
            speed=P7_MOVEJ_VELOCITY,
            mvacc=P7_MOVEJ_ACCELERATION,
            is_radian=True,
            wait=True,
        )

        if isinstance(ret, (list, tuple)):
            code = ret[0] if ret else None
        else:
            code = ret

        if isinstance(code, int) and code not in (0, 3):
            raise RuntimeError(f"p7 MoveJ failed: ret={ret}")

        return ret

    safe_motion(
        _motion,
        monitor,
        "return_container_capture_p7",
    )

    assert_xarm_normal(node, "after p7 MoveJ")


def return_to_capture_tcp_pose(node, monitor, capture_pose, label):
    node.get_logger().info(f"[RETURN CAPTURE] {label}")

    safe_motion(
        lambda: arm._moveL(
            capture_pose,
            velocity=150,
            acceleration=80,
            asynchronous=False,
        ),
        monitor,
        label,
    )

    assert_xarm_normal(node, f"after {label}")


# ============================================================
# HAND
# ============================================================
def safe_hand_open(hand_worker, book_width):
    try:
        print(f"[HAND] OPEN book_width={book_width:.2f} mm")

        updated_worker = hand_worker.open_until_width(
            float(book_width),
            gravity=False,
        )

        if updated_worker is None:
            raise RuntimeError("open_until_width returned None")

        print("[HAND] OPEN completed")
        return updated_worker

    except Exception as e:
        system_emergency_stop(
            f"Dynamixel OPEN error: {type(e).__name__}: {e}"
        )
        raise


def safe_hand_grasp(hand_worker):
    try:
        print("[HAND] GRASP / CLOSE")
        result = hand_worker.grasp()
        print(f"[HAND] GRASP result: {result}")
        return result

    except Exception as e:
        system_emergency_stop(
            f"Dynamixel GRASP error: {type(e).__name__}: {e}"
        )
        raise

def return_to_capture_with_hand_close(
    node,
    monitor,
    hand_worker,
    capture_pose,
    label,
):
    """
    撮影姿勢へ戻るのと同時にハンドをCLOSEする。
    """

    hand_errors = []

    def _close_hand():
        try:
            safe_hand_grasp(hand_worker)
        except Exception as e:
            hand_errors.append(e)

    print("\n========================================")
    print("[RETRY] return capture + hand CLOSE")
    print("========================================")

    hand_thread = threading.Thread(
        target=_close_hand,
        daemon=True,
    )

    hand_thread.start()

    try:
        return_to_capture_tcp_pose(
            node,
            monitor,
            capture_pose,
            label,
        )
    finally:
        hand_thread.join()

    if hand_errors:
        raise hand_errors[0]

    print("[RETRY] capture pose reached + hand CLOSED")
# ============================================================
# RECOGNITION
# ============================================================
def recognize_container_book(node, book_name):
    last_exception = None

    for attempt in range(1, MAX_RECOGNITION_ATTEMPTS + 1):
        node.get_logger().info(
            f"[RECOGNITION] attempt={attempt}/{MAX_RECOGNITION_ATTEMPTS}, "
            f"query={book_name!r}"
        )

        try:
            raw_roll, p_xmax, book_width, shot_dir = (
                run_capture_and_pca_sam3_refined_sam2_width(
                    query=book_name,
                    sam_device="gpu",
                )
            )

            if p_xmax is None:
                raise RuntimeError("Recognition failed: p_xmax is None")

            p_xmax = np.asarray(p_xmax, dtype=np.float64).reshape(3)
            book_width = float(book_width)
            shot_dir = Path(shot_dir)

            print("\n========== CONTAINER RECOGNITION ==========")
            print(f"raw roll [deg] : {np.degrees(raw_roll):.2f}")
            print(f"p_xmax [m]     : {p_xmax.tolist()}")
            print(f"book_width mm  : {book_width:.2f}")
            print(f"shot_dir       : {shot_dir}")
            print("===========================================\n")

            return float(raw_roll), p_xmax, book_width, shot_dir

        except Exception as e:
            last_exception = e
            node.get_logger().error(
                f"[RECOGNITION] failed {attempt}/{MAX_RECOGNITION_ATTEMPTS}: "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()

            if attempt < MAX_RECOGNITION_ATTEMPTS:
                time.sleep(1.0)

    raise RuntimeError(
        f"Container recognition failed {MAX_RECOGNITION_ATTEMPTS} times: "
        f"{last_exception}"
    )


def adjust_roll_like_retrieval(raw_roll):
    roll_deg = float(np.degrees(raw_roll))

    if roll_deg > 90.0:
        adjusted = -(raw_roll - np.radians(90.0))
    elif roll_deg < -90.0:
        adjusted = -(raw_roll + np.radians(90.0))
    else:
        adjusted = 0.0

    print(
        f"[ROLL] raw={roll_deg:.2f} deg, "
        f"adjusted={np.degrees(adjusted):.2f} deg"
    )

    return float(adjusted)


# ============================================================
# RETRIEVE AFTER GRASP
# ============================================================
def retrieve_book_from_container(node, monitor, side):
    current_pose = [
        float(v)
        for v in arm.get_tcp_pose(is_radian=True)
    ]

    target_pose = list(current_pose)

    if side == "right":
        target_pose[0] -= RETRIEVAL_DX
    elif side == "left":
        target_pose[0] += RETRIEVAL_DX
    else:
        raise RuntimeError(f"invalid side: {side}")

    node.get_logger().info(
        f"[CONTAINER RETRIEVE] X={current_pose[0]:.2f} "
        f"-> {target_pose[0]:.2f} mm"
    )

    safe_motion(
        lambda: arm._moveL(
            target_pose,
            velocity=TCP_VEL_1,
            acceleration=TCP_ACC_1,
            asynchronous=False,
        ),
        monitor,
        "container_retrieving_book",
    )

    assert_xarm_normal(node, "after container retrieving")


# ============================================================
# WAYPOINT WAIT
# ============================================================
def wait_waypoint_finished(node, executor, waypoint_node):
    while rclpy.ok() and not waypoint_node.is_finished():
        executor.spin_once(timeout_sec=0.1)

    if not rclpy.ok():
        raise RuntimeError("ROS shutdown while waiting waypoint")

    if waypoint_node.is_failed():
        raise RuntimeError(
            "Waypoint failed: "
            f"{waypoint_node.error_message()}"
        )

    node.get_logger().info(
        "[WAYPOINT] init_to_container_capture completed"
    )


# ============================================================
# INPUT
# ============================================================
def wait_user_action():
    print("\n========================================")
    print(" ハンドOPEN完了")
    print("")
    print(" Enter  : 挿入 -> 把持 -> p7撮影姿勢へ戻る")
    print(" Ctrl+D : 挿入せず p7撮影姿勢へ戻る -> ハンドCLOSE")
    print(" Ctrl+C : 全系非常停止")
    print("========================================\n")

    while True:
        try:
            value = input("操作を選択してください: ")
        except EOFError:
            print("\n[INPUT] Ctrl+D detected")
            return "ctrl_d"

        if value.strip() == "":
            print("[INPUT] Enter detected")
            return "enter"

        print("Enter または Ctrl+D を使用してください。")


class FrozenTcpArm:
    """
    cam_mm_to_robot_mm() に、
    現在TCPではなく保存済みTCPを使わせるためのラッパー。
    """

    def __init__(self, arm, tcp_pose):
        self._arm = arm
        self._tcp_pose = np.asarray(
            tcp_pose,
            dtype=np.float64,
        ).reshape(6).copy()

    def get_tcp_pose(self, is_radian=True):
        pose = self._tcp_pose.copy()

        if not is_radian:
            pose[3:6] = np.degrees(
                pose[3:6]
            )

        return pose.tolist()

    def __getattr__(self, name):
        return getattr(
            self._arm,
            name,
        )

def wait_xarm_motion_done(timeout=10.0):
    start_time = time.monotonic()
    moving_detected = False

    while True:
        result = arm.arm.get_is_moving()

        if isinstance(result, bool):
            moving = result

        elif isinstance(result, (tuple, list)) and len(result) >= 2:
            code = int(result[0])

            if code != 0:
                raise RuntimeError(
                    f"get_is_moving failed: {result}"
                )

            moving = bool(result[1])

        else:
            raise RuntimeError(
                f"unexpected get_is_moving result: {result}"
            )

        if moving:
            moving_detected = True

        if moving_detected and not moving:
            return

        elapsed = time.monotonic() - start_time

        # moving=Trueを取り逃した場合
        if elapsed >= 1.0 and not moving_detected and not moving:
            return

        if elapsed >= timeout:
            raise RuntimeError(
                f"xArm motion timeout: {elapsed:.1f} sec"
            )

        time.sleep(0.02)

# ============================================================
# SEQUENCE
# ============================================================
def run_sequence(
    config,
    books_master,
    book_index,
    node,
    executor,
    shelf_manager,
    tp,
    monitor,
    hand_worker,
):
    # --------------------------------------------------------
    # この時点でxArmは必ず初期姿勢
    # --------------------------------------------------------
    assert_xarm_normal(
        node,
        "sequence start at init pose",
    )

    # 前回値をクリア
    try:
        shelf_manager.reset()
    except Exception:
        shelf_manager.received = False

    node.get_logger().info(
        "[INIT] xArm is at initial pose"
    )
    node.get_logger().info(
        "Waiting for /shelf_id ..."
    )

    while rclpy.ok() and not shelf_manager.is_received():
        executor.spin_once(timeout_sec=0.1)

    if not rclpy.ok():
        raise RuntimeError("ROS shutdown while waiting /shelf_id")

    side = shelf_manager.get_side()
    height = shelf_manager.get_height()
    shelf_id = shelf_manager.get_shelf_id()

    print("\n========================================")
    print(f"[SHELF] id     : {shelf_id}")
    print(f"[SHELF] side   : {side}")
    print(f"[SHELF] height : {height}")
    print("========================================")

    # --------------------------------------------------------
    # 対象本は書架ID一致では選ばない。
    #
    # 出庫と同じくJSONの順番をそのまま使用する。
    #   books_master[0]
    #   books_master[1]
    #   books_master[2]
    #   ...
    #
    # /shelf_id は side / height 等の動作情報として使うだけ。
    # --------------------------------------------------------
    if (
        book_index < 0
        or book_index >= len(books_master)
    ):
        raise RuntimeError(
            "book_index out of range: "
            f"{book_index}, "
            f"book_count={len(books_master)}"
        )

    book = books_master[
        book_index
    ]

    if not isinstance(
        book,
        dict,
    ):
        raise RuntimeError(
            "book JSON entry must be dict: "
            f"index={book_index}"
        )

    book_name = str(
        book.get(
            "book_name",
            "",
        )
    ).strip()

    if not book_name:
        raise RuntimeError(
            "JSONのbook_nameが空です: "
            f"index={book_index}"
        )

    json_bookshelf_id = str(
        book.get(
            "bookshelf_ID",
            "",
        )
    ).strip()

    print("")
    print(
        "========================================"
    )
    print(
        "[BOOK ORDER] JSON top-to-bottom"
    )
    print(
        f"[BOOK ORDER] index = "
        f"{book_index}"
    )
    print(
        f"[BOOK ORDER] book = "
        f"{book_index + 1}/"
        f"{len(books_master)}"
    )
    print(
        f"[BOOK ORDER] book_name = "
        f"{book_name!r}"
    )
    print(
        f"[BOOK ORDER] JSON bookshelf_ID = "
        f"{json_bookshelf_id!r}"
    )
    print(
        f"[BOOK ORDER] received /shelf_id = "
        f"{shelf_id!r}"
    )
    print(
        "[BOOK ORDER] "
        "bookshelf_ID equality check = DISABLED"
    )
    print(
        "========================================"
    )

    node.get_logger().info(
        "[BOOK TARGET] "
        f"index={book_index}, "
        f"name={book_name!r}"
    )

    # --------------------------------------------------------
    # まず上下機構をコンテナ高さへ移動開始
    #
    # 値は固定値を指令せず、
    # Retrieval_integration.yaml の
    #   linear_lift.move_to_container
    # を唯一の指令元として使う。
    # --------------------------------------------------------
    lift_target_mm = float(
        config["linear_lift"]["move_to_container"]
    )

    print("")
    print(
        "========================================"
    )
    print(
        "[LIFT] move_to_container command"
    )
    print(
        f"[LIFT] config target = "
        f"{lift_target_mm:.1f} mm"
    )
    print(
        f"[LIFT] expected container height = "
        f"{EXPECTED_CONTAINER_HEIGHT_MM:.1f} mm"
    )
    print(
        "========================================"
    )

    # コンテナ高さ設定ミスを検出したら、
    # 間違った高さへ動かす前に停止する。
    if abs(
        lift_target_mm
        - EXPECTED_CONTAINER_HEIGHT_MM
    ) > CONTAINER_HEIGHT_TOLERANCE_MM:
        raise RuntimeError(
            "linear_lift.move_to_container が"
            "実機コンテナ高さ1100 mmと一致しません: "
            f"config={lift_target_mm:.1f} mm"
        )

    tp.publish_target_mm(
        lift_target_mm
    )

    # publishをDDSへ流す。
    # ここでは1100 mm到達を待たない。
    rclpy.spin_once(
        tp,
        timeout_sec=0.1,
    )

    node.get_logger().info(
        "[LIFT] movement started -> "
        "start arm YAML while lift is moving"
    )

    # --------------------------------------------------------
    # 上下機構を動かしている間に
    # init -> container capture YAMLを再生
    #
    # /navigation_goal は見ない。
    # YAMLはこちらから明示的に1回だけ再生。
    # type省略=MoveJ / movel / movel_relative に対応。
    # --------------------------------------------------------
    play_motion_yaml_safely(
        node,
        monitor,
        INIT_TO_CONTAINER_CAPTURE_YAML,
    )
    assert_xarm_normal(node, "after init_to_container_capture")

    # YAML最終点がp7でも、ここで指定p7へ確実に合わせる
    move_to_container_capture_p7(node, monitor)

    # p7のTCPを戻り先として保存
    capture_pose = [
        float(v)
        for v in arm.get_tcp_pose(is_radian=True)
    ]

    print(
        "[CAPTURE POSE] saved TCP: "
        f"X={capture_pose[0]:.2f}, "
        f"Y={capture_pose[1]:.2f}, "
        f"Z={capture_pose[2]:.2f}, "
        f"R={np.degrees(capture_pose[3]):.2f}, "
        f"P={np.degrees(capture_pose[4]):.2f}, "
        f"Yaw={np.degrees(capture_pose[5]):.2f}"
    )

    # --------------------------------------------------------
    # タイトル名をqueryにして認識
    #
    # 出庫と同じ認識関数を使い、
    # 現在のJSON行のbook_nameをそのままqueryへ渡す。
    # 書架ID一致ではなく、タイトル名の認識成功で対象を確認する。
    # --------------------------------------------------------
    node.get_logger().info(
        "[TITLE CHECK] "
        f"recognition query={book_name!r}"
    )

    raw_roll, p_xmax, book_width, shot_dir = recognize_container_book(
        node,
        book_name,
    )

    node.get_logger().info(
        "[TITLE CHECK] "
        f"target title recognized={book_name!r}"
    )

    adjusted_roll = adjust_roll_like_retrieval(raw_roll)

    p_max_mm = 1000.0 * p_xmax
    print_camera_debug_info(arm, p_max_mm)

    p_robot_mm = np.asarray(
        cam_mm_to_robot_mm(arm, p_max_mm),
        dtype=np.float64,
    ).reshape(3)

    print(
        "[ROBOT TARGET] "
        f"X={p_robot_mm[0]:.2f} mm, "
        f"Y={p_robot_mm[1]:.2f} mm, "
        f"Z={p_robot_mm[2]:.2f} mm"
    )

    # 本正面へリーチング
    safe_motion(
        lambda: arm.move_to_storage_target_xyz_and_roll_container(
            p_robot_mm=p_robot_mm,
            d_roll_rad=adjusted_roll,
            side=side,
        ),
        monitor,
        "storage_book_positioning",
    )

    reaching_pose = [
        float(v)
        for v in arm.get_tcp_pose(is_radian=True)
    ]

    first_reaching_robot_x_mm = float(
        reaching_pose[0]
    )

    assert_xarm_normal(node, "after storage book positioning")

    # 出庫の挿入監視関数に渡す基準Yは
    # 出庫コードと同じく「認識した本のrobot Y」を使う
    first_recognition_robot_y_mm = float(p_robot_mm[1])

    # ハンドOPEN
    hand_worker = safe_hand_open(hand_worker, book_width)

    # Enter / Ctrl+D
    action = wait_user_action()

    # --------------------------------------------------------
    # Ctrl+D
    # --------------------------------------------------------
    if action == "ctrl_d":
        node.get_logger().warn(
            "[CTRL+D] insertion SKIP -> return p7 -> hand close"
        )

        # 本正面から保存したp7 TCPへ直線退避
        return_to_capture_tcp_pose(
            node,
            monitor,
            capture_pose,
            "ctrl_d_return_capture_pose",
        )

        # p7関節姿勢へ確実に合わせる
        move_to_container_capture_p7(node, monitor)

        # p7へ戻ってから閉じる
        safe_hand_grasp(hand_worker)

        print("\n========================================")
        print(" Ctrl+D sequence completed")
        print(" p7 capture pose + hand CLOSED")
        print("========================================")

        return {
            "result": "ctrl_d",
            "book_index": book_index,
            "book_name": book_name,
            "json_bookshelf_ID": json_bookshelf_id,
            "shelf_id": shelf_id,
            "book_width": book_width,
            "shot_dir": str(shot_dir),
        }

    # --------------------------------------------------------
    # Enter -> INSERT
    # --------------------------------------------------------
    node.get_logger().info("[ENTER] start insertion")

    storage_retry_count = 0

    while True:

        # ====================================================
        # 書籍への挿入
        # ====================================================
        insert_result = (
            arm.moveL_to_storage_insert_z_with_current_monitor(
                retry_capture_pose=capture_pose,
                retry_reaching_x_mm=first_reaching_robot_x_mm,
                retry_x_offset_mm=0.0,
            )
        )

        if not isinstance(insert_result, dict):
            raise RuntimeError(
                f"insert_result is not dict: {insert_result}"
            )

        # ====================================================
        # 挿入成功
        # ====================================================
        if insert_result.get("ok", False):
            node.get_logger().info(
                f"[STORAGE INSERT] success "
                f"(retry_count={storage_retry_count})"
            )
            break

        reason = insert_result.get("reason", "")

        # ====================================================
        # 電流閾値以外は通常の異常
        # -> main()へ投げて全系E-STOP
        # ====================================================
        if reason != "current_threshold":
            raise RuntimeError(
                "Container insertion failed: "
                f"{insert_result}"
            )

        # ====================================================
        # 電流閾値超過
        # ====================================================
        trigger_joint = insert_result.get(
            "trigger_joint",
            "UNKNOWN",
        )

        node.get_logger().warn(
            f"[STORAGE RETRY] "
            f"current threshold detected: {trigger_joint}"
        )

        # リトライ回数上限
        if storage_retry_count >= MAX_STORAGE_INSERT_RETRIES:
            raise RuntimeError(
                "Container insertion current retry limit exceeded: "
                f"{insert_result}"
            )

        storage_retry_count += 1

        # ====================================================
        # xArm側で計算された
        #
        #   X = リーチング時X
        #   Y = 元の撮影姿勢Y
        #   Z = 元の撮影姿勢Z
        #
        # の退避姿勢を取得
        # ====================================================
        retry_pose = insert_result.get(
            "retry_target_pose"
        )

        if (
            retry_pose is None
            or not isinstance(retry_pose, (list, tuple))
            or len(retry_pose) != 6
        ):
            raise RuntimeError(
                "Invalid retry_target_pose: "
                f"{retry_pose}"
            )

        retry_pose = [
            float(v)
            for v in retry_pose
        ]

        print("\n========================================")
        print(
            f"[STORAGE RETRY {storage_retry_count}/"
            f"{MAX_STORAGE_INSERT_RETRIES}]"
        )
        print(
            "[RETRY CAPTURE POSE] "
            f"X={retry_pose[0]:.2f}, "
            f"Y={retry_pose[1]:.2f}, "
            f"Z={retry_pose[2]:.2f}"
        )
        print(
            f"[TRIGGER] {trigger_joint}"
        )
        print("========================================\n")

        # ====================================================
        # 電流閾値超過後
        #
        # 下位関数側ですでに一旦retry_poseまで退避済み。
        # ここから本来のp7撮影姿勢へ戻る。
        #
        # ★ p7へ戻る間にハンドをCLOSE
        # ====================================================
        return_to_capture_with_hand_close(
            node,
            monitor,
            hand_worker,
            capture_pose,
            f"storage_current_retry_capture_{storage_retry_count}",
        )

        # p7関節姿勢へ確実に合わせる
        move_to_container_capture_p7(
            node,
            monitor,
        )

        # カメラ振動待ち
        time.sleep(0.5)

        # ====================================================
        # ハンドCLOSE状態で再認識
        # ====================================================
        node.get_logger().info(
            f"[STORAGE RETRY] "
            f"recognition retry={storage_retry_count}"
        )

        raw_roll, p_xmax, book_width, shot_dir = (
            recognize_container_book(
                node,
                book_name,
            )
        )

        adjusted_roll = adjust_roll_like_retrieval(
            raw_roll
        )

        p_max_mm = 1000.0 * p_xmax

        print_camera_debug_info(
            arm,
            p_max_mm,
        )

        p_robot_mm = np.asarray(
            cam_mm_to_robot_mm(
                arm,
                p_max_mm,
            ),
            dtype=np.float64,
        ).reshape(3)

        print(
            "[RETRY ROBOT TARGET] "
            f"X={p_robot_mm[0]:.2f} mm, "
            f"Y={p_robot_mm[1]:.2f} mm, "
            f"Z={p_robot_mm[2]:.2f} mm"
        )

        # ====================================================
        # 再認識結果で再リーチング
        #
        # この時点ではハンドCLOSE
        # ====================================================
        safe_motion(
            lambda: arm.move_to_storage_target_xyz_and_roll_container(
                p_robot_mm=p_robot_mm,
                d_roll_rad=adjusted_roll,
                side=side,
            ),
            monitor,
            f"storage_retry_positioning_{storage_retry_count}",
        )

        assert_xarm_normal(
            node,
            f"after storage retry positioning "
            f"{storage_retry_count}",
        )

        # ====================================================
        # 新しいリーチング位置Xを保存
        #
        # 次にまた電流閾値超過した場合は
        # この新しいXを退避位置計算に使用する
        # ====================================================
        reaching_pose = [
            float(v)
            for v in arm.get_tcp_pose(
                is_radian=True
            )
        ]

        first_reaching_robot_x_mm = float(
            reaching_pose[0]
        )

        first_recognition_robot_y_mm = float(
            p_robot_mm[1]
        )

        print(
            "[RETRY REACHING] "
            f"X={first_reaching_robot_x_mm:.2f} mm"
        )

        # ====================================================
        # 再リーチング完了後にハンドOPEN
        # ====================================================
        hand_worker = safe_hand_open(
            hand_worker,
            book_width,
        )

        # ====================================================
        # ★ 再挿入前にもう一度input
        # ====================================================
        retry_action = wait_user_action()

        # ----------------------------------------------------
        # Ctrl+D
        # 再挿入をやめてp7へ戻る
        # ----------------------------------------------------
        if retry_action == "ctrl_d":

            node.get_logger().warn(
                "[STORAGE RETRY] "
                "Ctrl+D -> retry insertion cancelled"
            )

            return_to_capture_tcp_pose(
                node,
                monitor,
                capture_pose,
                "retry_ctrl_d_return_capture",
            )

            move_to_container_capture_p7(
                node,
                monitor,
            )

            # p7でハンドCLOSE
            safe_hand_grasp(
                hand_worker
            )

            print("\n========================================")
            print(" STORAGE RETRY cancelled")
            print(" p7 capture pose + hand CLOSED")
            print("========================================")

            return {
                "result": "ctrl_d_retry",
                "book_index": book_index,
                "book_name": book_name,
                "json_bookshelf_ID": json_bookshelf_id,
                "shelf_id": shelf_id,
                "book_width": book_width,
                "shot_dir": str(shot_dir),
                "storage_retry_count": storage_retry_count,
            }

        # ----------------------------------------------------
        # Enter
        #
        # while先頭へ戻って再挿入
        # ----------------------------------------------------
        node.get_logger().info(
            f"[STORAGE RETRY] "
            f"Enter -> retry insertion "
            f"{storage_retry_count}"
        )

        continue


    assert_xarm_normal(
        node,
        "after container insertion",
    )

    # 把持
    grasp_result = safe_hand_grasp(hand_worker)

    # 把持後はいきなりMoveJせず、既存出庫と同じ方向へ直線引き抜き
    safe_motion(
        lambda: arm.moveL_to_storage_retract_z(),
        monitor,
        "storage_retract_z",
    )

    # 引き抜いた後、保存したp7 TCPへ戻る
    return_to_capture_tcp_pose(
        node,
        monitor,
        capture_pose,
        "grasped_return_capture_pose",
    )

    # 最後にp7関節姿勢へ確実に合わせる
    move_to_container_capture_p7(node, monitor)
    
    # --------------------------------------------------------
    # p7撮影姿勢 -> バーコード読み取り姿勢
    # --------------------------------------------------------
    node.get_logger().info(
        "[BARCODE] move from storage capture pose to barcode pose"
    )

    play_motion_yaml_safely(
        node,
        monitor,
        STORAGE_CAPTURE_TO_BARCODE_YAML,
    )

    assert_xarm_normal(
        node,
        "after storage_capture_to_barcode",
    )

    node.get_logger().info(
        "[BARCODE] barcode reading pose reached"
    )

    # ========================================================
    # 出庫と同じバーコード認識
    #
    # 複数フレームpyzbar
    # -> 全滅ならPP-OCRv5
    # ========================================================
    book_barcode_shot_dir = (
        Path(shot_dir)
        / "book_barcode"
    )

    (
        detected_isbn,
        matched_book,
        barcode_method,
    ) = identify_book_isbn_at_current_pose(
        books_master=books_master,
        shot_dir=book_barcode_shot_dir,
        num_frames=5,
    )

    # ========================================================
    # これ以降PaddleOCRは不要なのでGPUメモリを解放
    # ========================================================
    print("[OCR WORKER] shutdown before Storage SAM3")
    shutdown_ocr_worker()
    print("[OCR WORKER] shutdown completed")
    # ========================================================
    # 入庫先書架ID
    # ========================================================
    destination_bookshelf_id = str(
        matched_book.get(
            "bookshelf_ID",
            "",
        )
    ).strip()

    detected_book_name = str(
        matched_book.get(
            "book_name",
            "",
        )
    ).strip()

    if not destination_bookshelf_id:
        raise RuntimeError(
            "JSONのbookshelf_IDが空です: "
            f"ISBN={detected_isbn}"
        )

    # ========================================================
    # 入庫先 bookshelf_ID
    # -> ShelfIDManager
    # -> side / lift height / TCP offset
    # ========================================================
    (
        destination_side,
        destination_lift_height_mm,
        destination_tcp_z_offset_mm,
    ) = shelf_manager.resolve_shelf_id(
        destination_bookshelf_id
    )

    node.get_logger().info(
        "[STORAGE DESTINATION] "
        f"bookshelf_ID={destination_bookshelf_id}, "
        f"side={destination_side}, "
        f"lift={destination_lift_height_mm:.1f} mm, "
        f"tcp_offset="
        f"{destination_tcp_z_offset_mm:.1f} mm"
    )


    # ========================================================
    # 書架IDが分かった瞬間に上下機構を移動開始
    # ========================================================
    node.get_logger().info(
        "[STORAGE LIFT] "
        f"move start -> "
        f"{destination_lift_height_mm:.1f} mm"
    )

    time.sleep(1.0)
    
    tp.publish_target_mm(
        destination_lift_height_mm
    )

    rclpy.spin_once(
        tp,
        timeout_sec=0.1,
    )

    print("")
    print("========================================")
    print("[STORAGE DESTINATION]")
    print(
        f"method       = "
        f"{barcode_method}"
    )
    print(
        f"ISBN         = "
        f"{detected_isbn}"
    )
    print(
        f"book_name    = "
        f"{detected_book_name}"
    )
    print(
        f"bookshelf_ID = "
        f"{destination_bookshelf_id}"
    )
    print("========================================")

    # --------------------------------------------------------
    # バーコード認識完了後
    # barcode_to_capture.yaml を再生
    # --------------------------------------------------------
    node.get_logger().info(
        "[BARCODE] barcode recognition completed "
        "-> play barcode_to_capture.yaml"
    )

    play_motion_yaml_safely(
        node,
        monitor,
        BARCODE_CAPTURE_YAML,
    )

    assert_xarm_normal(
        node,
        "after barcode_to_capture",
    )

    node.get_logger().info(
        "[BARCODE] barcode_to_capture.yaml completed"
    )


    # ========================================================
    # 書架の空きスペース認識
    #
    # 1回目撮影
    #   ↓
    # 撮影直後に -Y140mm 移動開始
    #   ↓
    # 1回目認識
    #
    # 成功:
    #   1 = 1回目結果
    #   2 = 2回目撮影
    #
    # 認識失敗:
    #   自動的に2回目撮影
    # ========================================================

    first_result = None

    def move_to_second_capture():

        print(
            "[STORAGE] start moving to "
            "2nd capture position"
        )

        arm.moveL_y_offset(
            SECOND_CAPTURE_DY_MM,
            velocity=STORAGE_SPACE_MOVE_VELOCITY,
            acceleration=STORAGE_SPACE_MOVE_ACCELERATION,
            asynchronous=True,
        )


    # ========================================================
    # 1回目撮影
    # ========================================================
    try:

        print(
            "\n========== STORAGE 1st CAPTURE =========="
        )

        first_result = run_capture_and_pca_depth_space(
            crop_y_min=120,
            crop_y_max=550,
            min_area_px=800,
            min_height_px=100,
            max_space_width_px=420,
            target_y_from_space_top_px=180,

            # 撮影直後に -Y140 mm
            after_capture_callback=move_to_second_capture,
        )

        print(
            "[STORAGE] 1st recognition SUCCESS"
        )

    except RuntimeError as e:

        error_message = str(e)

        # ====================================================
        # 2回目へ進んでよい認識エラー
        # ====================================================
        retryable_errors = (
            "収納スペース候補が見つかりませんでした",
            "target pixel depth is invalid",
            "target depth out of range",
        )

        if not any(
            msg in error_message
            for msg in retryable_errors
        ):
            raise

        print(
            "\n[STORAGE] 1st recognition FAILED"
        )

        print(
            f"[STORAGE] reason: {error_message}"
        )

        first_result = None


    # ========================================================
    # 撮影直後に開始した -Y140 mm の完了待ち
    #
    # ここでxArmは2回目撮影位置
    # ========================================================
    wait_xarm_motion_done()

    assert_xarm_normal(
        node,
        "after moving to 2nd storage capture position",
    )

    print(
        "[STORAGE] arrived at 2nd capture position"
    )


    # ========================================================
    # 1回目成功なら使用結果を選択
    # ========================================================
    if first_result is not None:

        while True:

            print("")
            print(
                "=========================================="
            )
            print(
                "収納スペース認識結果を選択してください"
            )
            print(
                "  1 : 1回目の認識結果を使う"
            )
            print(
                "  2 : 2回目を撮影して、その結果を使う"
            )
            print(
                "=========================================="
            )

            choice = input(
                "select [1/2] : "
            ).strip()

            if choice in ("1", "2"):
                break

            print(
                "[WARN] 1 または 2 を入力してください"
            )

    else:

        print(
            "[STORAGE] "
            "automatically use 2nd capture"
        )

        choice = "2"


    # ========================================================
    # 1回目結果を使用
    # ========================================================
    if choice == "1":

        print(
            "\n========== USE STORAGE 1st RESULT =========="
        )

        (
            storage_angle_rad,
            storage_target_cam,
            storage_space_result,
        ) = first_result


        # ----------------------------------------------------
        # 現在は -Y140mm の2回目撮影位置にいる。
        #
        # 1回目画像のカメラ座標を使うので、
        # 元の撮影位置へ +Y140mm 戻す。
        # ----------------------------------------------------
        print(
            "[STORAGE] return to 1st capture position"
        )

        safe_motion(
            lambda: arm.moveL_y_offset(
                -SECOND_CAPTURE_DY_MM,
                velocity=STORAGE_SPACE_MOVE_VELOCITY,
                acceleration=STORAGE_SPACE_MOVE_ACCELERATION,
                asynchronous=False,
            ),
            monitor,
            "storage_return_to_first_capture",
        )

        assert_xarm_normal(
            node,
            "after returning to 1st storage capture",
        )

        storage_recognition_pass = 1

        print(
            "[STORAGE] use 1st recognition result"
        )


    # ========================================================
    # 2回目撮影
    # ========================================================
    else:

        print(
            "\n========== STORAGE 2nd CAPTURE =========="
        )

        (
            storage_angle_rad,
            storage_target_cam,
            storage_space_result,
        ) = run_capture_and_pca_depth_space(
            crop_y_min=120,
            crop_y_max=550,
            min_area_px=800,
            min_height_px=100,
            max_space_width_px=420,
            target_y_from_space_top_px=240,

            # 2回目ではもう動かさない
            after_capture_callback=None,
        )

        storage_recognition_pass = 2

        print(
            "[STORAGE] use 2nd recognition result"
        )

    # ========================================================
    # 認識結果
    # ========================================================
    storage_angle_rad = float(
        storage_angle_rad
    )

    storage_angle_deg = float(
        np.degrees(storage_angle_rad)
    )

    storage_target_cam = np.asarray(
        storage_target_cam,
        dtype=np.float64,
    ).reshape(3)

    guide_edge_length_mm = float(
        storage_space_result.get(
            "guide_edge_length_mm",
            0.0,
        )
    )

    # 後でスペーサ挿入時に使用する水平補正量
    storage_dy_mm = (
        -guide_edge_length_mm
        * np.cos(storage_angle_rad)
        / 2.0
    )

    print("\n========================================")
    print(" STORAGE SPACE RECOGNITION SUCCESS")
    print("========================================")

    print(
        f"angle_rad = "
        f"{storage_angle_rad:.6f} rad"
    )

    print(
        f"angle_deg = "
        f"{storage_angle_deg:.2f} deg"
    )

    print(
        "[target camera m] "
        f"X={storage_target_cam[0]:.4f}, "
        f"Y={storage_target_cam[1]:.4f}, "
        f"Z={storage_target_cam[2]:.4f}"
    )

    print(
        "[target camera mm] "
        f"X={storage_target_cam[0] * 1000.0:.1f}, "
        f"Y={storage_target_cam[1] * 1000.0:.1f}, "
        f"Z={storage_target_cam[2] * 1000.0:.1f}"
    )

    print(
        f"guide_edge_length_mm = "
        f"{guide_edge_length_mm:.2f}"
    )

    print(
        f"storage_dy_mm = "
        f"{storage_dy_mm:.2f}"
    )

    print(
        "right_is_tilted = "
        f"{storage_space_result.get('right_is_tilted', False)}"
    )

    print(
        "pair_indices = "
        f"{storage_space_result.get('pair_indices', None)}"
    )

    print("========================================")



    # ========================================================
    # カメラ座標 [m] -> カメラ座標 [mm]
    # ========================================================
    storage_target_cam_mm = (
        1000.0 * storage_target_cam
    )

    print("")
    print("========================================")
    print("[STORAGE TARGET CAMERA]")
    print(
        f"X={storage_target_cam_mm[0]:.2f} mm, "
        f"Y={storage_target_cam_mm[1]:.2f} mm, "
        f"Z={storage_target_cam_mm[2]:.2f} mm"
    )
    print("========================================")


    # ========================================================
    # カメラ座標 -> ロボットベース座標
    #
    # 1回目結果を使う場合：
    #   すでに +Y140 mm 戻って1回目撮影位置にいる
    #
    # 2回目結果を使う場合：
    #   現在2回目撮影位置にいる
    #
    # したがって「現在TCP」で座標変換してOK
    # ========================================================
    storage_target_robot_mm = np.asarray(
        cam_mm_to_robot_mm(
            arm,
            storage_target_cam_mm,
        ),
        dtype=np.float64,
    ).reshape(3)

    print("")
    print("========================================")
    print("[STORAGE TARGET ROBOT]")
    print(
        f"X={storage_target_robot_mm[0]:.2f} mm, "
        f"Y={storage_target_robot_mm[1]:.2f} mm, "
        f"Z={storage_target_robot_mm[2]:.2f} mm"
    )
    print(
        f"recognition_pass = "
        f"{storage_recognition_pass}"
    )
    print("========================================")


    # ========================================================
    # リーチング前の安全確認
    # ========================================================
    try:
        input(
            "Enter: execute storage reaching / "
            "Ctrl+D: stop here : "
        )

    except EOFError:

        print("")
        print("[STORAGE] Ctrl+D detected")
        print("[STORAGE] reaching cancelled")
        print("[STORAGE] keep current capture pose")

        return {
            "result": "storage_reaching_cancelled",
            "book_index": book_index,
            "book_name": book_name,
            "detected_isbn": detected_isbn,
            "detected_book_name": detected_book_name,
            "destination_bookshelf_ID": (
                destination_bookshelf_id
            ),
            "destination_side": destination_side,
            "storage_recognition_pass": (
                storage_recognition_pass
            ),
            "storage_angle_deg": (
                storage_angle_deg
            ),
            "storage_target_cam_mm": (
                storage_target_cam_mm.tolist()
            ),
            "storage_target_robot_mm": (
                storage_target_robot_mm.tolist()
            ),
        }


    # ========================================================
    # 認識した入庫位置へリーチング
    # ========================================================
    print("")
    print("========================================")
    print("[STORAGE REACHING]")
    print("move to recognized storage target")
    print(
        f"side = {destination_side}"
    )
    print(
        f"target = "
        f"[{storage_target_robot_mm[0]:.2f}, "
        f"{storage_target_robot_mm[1]:.2f}, "
        f"{storage_target_robot_mm[2]:.2f}] mm"
    )
    print("========================================")


    safe_motion(
        lambda: arm.move_to_storage_target_xyz_and_roll(
            p_robot_mm=storage_target_robot_mm,
            d_roll_rad=0.0,
            side=destination_side,
        ),
        monitor,
        "storage_destination_reaching",
    )


    # ========================================================
    # リーチング後チェック
    # ========================================================
    assert_xarm_normal(
        node,
        "after storage destination reaching",
    )

    storage_reaching_pose = [
        float(v)
        for v in arm.get_tcp_pose(
            is_radian=True
        )
    ]

    print("")
    print("========================================")
    print(" STORAGE REACHING COMPLETED")
    print("========================================")
    print(
        "[REACHED TCP] "
        f"X={storage_reaching_pose[0]:.2f}, "
        f"Y={storage_reaching_pose[1]:.2f}, "
        f"Z={storage_reaching_pose[2]:.2f}"
    )
    print(
        f"recognition_pass = "
        f"{storage_recognition_pass}"
    )
    print("========================================")


    # ========================================================
    # 今回はここで終了
    # スペーサ展開・本挿入はまだ行わない
    # ========================================================
    return {
        "result": "storage_reaching_completed",

        "book_index": book_index,
        "book_name": book_name,

        "json_bookshelf_ID": (
            json_bookshelf_id
        ),

        "shelf_id": shelf_id,

        "book_width": book_width,
        "shot_dir": str(shot_dir),

        "grasp_result": grasp_result,
        "insert_result": insert_result,

        "detected_isbn": detected_isbn,
        "detected_book_name": (
            detected_book_name
        ),

        "destination_bookshelf_ID": (
            destination_bookshelf_id
        ),

        "destination_side": (
            destination_side
        ),

        "barcode_method": barcode_method,

        "storage_recognition_pass": (
            storage_recognition_pass
        ),

        "storage_angle_rad": (
            storage_angle_rad
        ),

        "storage_angle_deg": (
            storage_angle_deg
        ),

        "storage_target_cam_mm": (
            storage_target_cam_mm.tolist()
        ),

        "storage_target_robot_mm": (
            storage_target_robot_mm.tolist()
        ),

        "guide_edge_length_mm": (
            guide_edge_length_mm
        ),

        "storage_dy_mm": (
            storage_dy_mm
        ),

        "storage_reaching_pose": (
            storage_reaching_pose
        ),
    }

# ============================================================
# MAIN
# ============================================================
def main():
    global arm
    global lift_estop_pub
    global _hand_worker

    config = load_config(CONFIG_PATH)
    print(f"[CONFIG] loaded: {config['_config_path']}")

    books_master = load_books_master(config)

    if not INIT_TO_CONTAINER_CAPTURE_YAML.exists():
        raise FileNotFoundError(
            "init_to_container_capture.yaml not found: "
            f"{INIT_TO_CONTAINER_CAPTURE_YAML}"
        )

    if not STORAGE_CAPTURE_TO_BARCODE_YAML.exists():
        raise FileNotFoundError(
            "strage_capture_tobarcode.yaml not found: "
            f"{STORAGE_CAPTURE_TO_BARCODE_YAML}"
        )
        
    if not BARCODE_CAPTURE_YAML.exists():
        raise FileNotFoundError(
            "barcode_capture.yaml not found: "
            f"{BARCODE_CAPTURE_YAML}"
        )

    rclpy.init(
        signal_handler_options=SignalHandlerOptions.NO
    )

    node = rclpy.create_node(
        "book_storage_container_pick_test"
    )

    executor = None
    tp = None
    shelf_manager = None

    try:
        # IAI E-STOP publisher
        lift_estop_pub = node.create_publisher(
            Bool,
            "/emergency_stop",
            10,
        )
        globals()["lift_estop_pub"] = lift_estop_pub

        # DDS discovery
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.1)

        # 前回のIAI E-STOPを解除
        release_linear_lift_estop()

        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.1)

        # xArm
        xarm_host = config["robot"]["xarm"]["host"]
        arm = XArm7(
            node=node,
            host=xarm_host,
        )
        globals()["arm"] = arm

        # Executor
        executor = MultiThreadedExecutor()
        executor.add_node(node)

        # xArm monitor
        monitor = XArmMonitor(
            arm,
            check_period=0.5,
            auto_stop=True,
            on_emergency=monitor_emergency_stop,
        )

        # ----------------------------------------------------
        # 上下機構
        # ----------------------------------------------------
        tp = TargetPublisher()
        executor.add_node(tp)

        # ----------------------------------------------------
        # /shelf_id受信専用
        #
        # WaypointPlayerNodeは使わない。
        # shelf_id受信とYAML再生を分離して、
        # 必ず
        #   lift command -> YAML
        # の順番にする。
        # ----------------------------------------------------
        shelf_manager = ShelfIDManager(
            node
        )

        # Dynamixel worker
        try:
            _hand_worker = DynamixelWorkerClient()
        except Exception as e:
            system_emergency_stop(
                f"Dynamixel worker init error: {type(e).__name__}: {e}"
            )
            raise

        print("[DXL] worker initialized")

        # ----------------------------------------------------
        # ★ 起動後、最初に必ず初期姿勢へ
        #
        # /shelf_idを待つ前に初期姿勢へ移動しておく。
        # その後、初期姿勢で待機する。
        # ----------------------------------------------------
        node.get_logger().info(
            "[STARTUP] moving xArm to initial pose"
        )

        safe_motion(
            lambda: arm.moveJ_to_init_Q_DEG(),
            monitor,
            "startup_init_pose",
        )

        assert_xarm_normal(
            node,
            "after startup init pose",
        )

        node.get_logger().info(
            "[STARTUP] initial pose reached"
        )

        # ----------------------------------------------------
        # 出庫と同じくJSONは上から順番。
        #
        # このstage1では「コンテナから1冊掴んでp7へ戻る」
        # ところまでなので、まず先頭1冊を処理して終了する。
        # 棚への入庫完了まで実装した段階で、
        # book_indexを0,1,2,...と進めればそのまま全冊ループ化できる。
        # ----------------------------------------------------
        book_index = int(
            START_BOOK_INDEX
        )

        if (
            book_index < 0
            or book_index >= len(books_master)
        ):
            raise RuntimeError(
                "START_BOOK_INDEX out of range: "
                f"{book_index}, "
                f"book_count={len(books_master)}"
            )

        node.get_logger().info(
            "[BOOK ORDER] "
            f"start JSON index={book_index}"
        )

        result = run_sequence(
            config=config,
            books_master=books_master,
            book_index=book_index,
            node=node,
            executor=executor,
            shelf_manager=shelf_manager,
            tp=tp,
            monitor=monitor,
            hand_worker=_hand_worker,
        )

        print("\n[RESULT]")
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

    except KeyboardInterrupt:
        system_emergency_stop("KeyboardInterrupt")

    except Exception as e:
        print("\n========================================")
        print("[FATAL ERROR]")
        print(f"{type(e).__name__}: {e}")
        print("========================================")
        traceback.print_exc()

        system_emergency_stop(
            f"{type(e).__name__}: {e}"
        )
        raise

    finally:
        try:
            if _hand_worker is not None:
                _hand_worker.close()
        except Exception as e:
            print(
                f"[DXL] cleanup warning: "
                f"{type(e).__name__}: {e}"
            )

        try:
            if tp is not None:
                tp.destroy_node()
        except Exception:
            pass

        try:
            node.destroy_node()
        except Exception:
            pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    _sam3_service_session = Sam3ServiceSession()
    with _sam3_service_session:
        main()