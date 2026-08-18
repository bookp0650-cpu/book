from xarm7.control.xarm7 import (XArm7,RETRIEVAL_DX,TCP_VEL_1,TCP_ACC_1,)
from rs_d435i.get_book_position import GetBookSpinePosition
import Dynamixel_win_pro_hand_book.HandBook_Retrieval as HandBook_retrieval 
import Dynamixel_win_pro_hand_book.HandBook_Storage as HandBook_storage
from Dynamixel_win_pro_hand_book.dynamixel_worker_client import (DynamixelWorkerClient,)
from pathlib import Path
from detection.pro_handbook.sam3_runtime.integration_service_manager import (Sam3ServiceSession,)
from detection.pro_handbook.sam_py_demo.get_book_points_sam3_refined_sam2_width import (run_capture_and_pca_sam3_refined_sam2_width,)
from xarm7.control.move_to_container_test import Move_to_Container
from xarm7.control.shelf_id_manager import ShelfIDManager
from detection.pro_handbook.sam_py_demo.bar_code.book_barcode import book_barcode_sequence
from detection.pro_handbook.sam_py_demo.bar_code.bookshelf_barcode import bookshelf_barcode_sequence
from xarm7.control.book_return_sequence import storage_sequence
from detection.pro_handbook.sam_py_demo.Storage import run_capture_and_pca_depth_space
from linear_lift import TargetPublisher
import rclpy
from rclpy.signals import SignalHandlerOptions
import cv2
import numpy as np
import json
from xarm7.control.robot_base_coordinate import PoseChain
from xarm7.control.robot_base_coordinate import cam_mm_to_robot_mm
import traceback
import time
from rclpy.executors import MultiThreadedExecutor
from xarm7.control.xarm_init_to_capture_integration import WaypointPlayerNode
import signal, os
import math
import sys
from xarm7.control.xarm_monitor import XArmMonitor, safe_motion
from datetime import datetime
from std_msgs.msg import Bool, Int32, Float32, String
import yaml
import ezodf
from xarm7.control.robot_base_coordinate import print_camera_debug_info
from detection.pro_handbook.sam_py_demo.bar_code.code_1_pic_ros2_editing import (capture_barcode_and_x_offset,WallDistanceWatcher,BoolPulseWatcher,BoolLatchWatcher)
from rclpy.qos import (QoSProfile,ReliabilityPolicy,DurabilityPolicy)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "Retrieval_integration.yaml"

# =========================================================
# 自動復帰用JSON保存ディレクトリ
# =========================================================
RESUME_MOTION_DIR = (
    SCRIPT_DIR / "resume_motion"
)

RESUME_MOTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
# =========================================================
# 再起動時に使う book_width 保存
# =========================================================
BOOK_WIDTH_STATE_PATH = (
    RESUME_MOTION_DIR
    / "retrieval_book_width.json"
)
# =========================================================
# RETRIEVING_BOOK 復帰用状態
# =========================================================
RETRIEVAL_MOTION_STATE_PATH = (
    RESUME_MOTION_DIR
    / "retrieval_motion_resume.json"
)
# =========================================================
# HAND_OPENING / GRASPING_BOOK 復帰用状態
# =========================================================
HAND_RESUME_STATE_PATH = (
    RESUME_MOTION_DIR
    / "retrieval_hand_resume.json"
)


def save_book_width(book_width: float):
    data = {
        "book_width_mm": float(book_width)
    }

    BOOK_WIDTH_STATE_PATH.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "[BOOK WIDTH SAVE] "
        f"{float(book_width):.2f} mm"
    )


def load_book_width():
    if not BOOK_WIDTH_STATE_PATH.exists():
        return None

    try:
        data = json.loads(
            BOOK_WIDTH_STATE_PATH.read_text(
                encoding="utf-8"
            )
        )

        value = float(
            data["book_width_mm"]
        )

        print(
            "[BOOK WIDTH RESTORE] "
            f"{value:.2f} mm"
        )

        return value

    except Exception as e:
        print(
            "[BOOK WIDTH RESTORE] failed: "
            f"{type(e).__name__}: {e}"
        )

        return None

def save_hand_resume_state(
    book_width,
    raw_roll_rad,
    p_xmax,
    p_robot_mm,
    shot_dir,
    capture_pose_for_retry,
    first_recognition_robot_y_mm,
):
    data = {
        "book_width_mm": float(book_width),
        "raw_roll_rad": float(raw_roll_rad),
        "p_xmax_m": [
            float(v)
            for v in np.asarray(
                p_xmax,
                dtype=np.float64,
            ).reshape(3)
        ],
        "p_robot_mm": [
            float(v)
            for v in np.asarray(
                p_robot_mm,
                dtype=np.float64,
            ).reshape(3)
        ],
        "shot_dir": str(shot_dir),
        "capture_pose_for_retry": [
            float(v)
            for v in capture_pose_for_retry
        ],
        "first_recognition_robot_y_mm": float(
            first_recognition_robot_y_mm
        ),
    }

    HAND_RESUME_STATE_PATH.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "[HAND RESUME STATE SAVE] "
        f"book_width={book_width:.2f} mm"
    )


def load_hand_resume_state():
    if not HAND_RESUME_STATE_PATH.exists():
        return None

    try:
        data = json.loads(
            HAND_RESUME_STATE_PATH.read_text(
                encoding="utf-8"
            )
        )

        print(
            "[HAND RESUME STATE RESTORE] "
            f"book_width="
            f"{float(data['book_width_mm']):.2f} mm"
        )

        return data

    except Exception as e:
        print(
            "[HAND RESUME STATE RESTORE] failed: "
            f"{type(e).__name__}: {e}"
        )
        return None



def save_retrieval_motion_state(
    book_name,
    bookshelf_id,
    side,
    target_pose,
):
    target_pose = np.asarray(
        target_pose,
        dtype=np.float64,
    ).reshape(6)

    data = {
        "book_name": str(book_name),
        "bookshelf_id": str(bookshelf_id),
        "side": str(side),
        "target_pose": [
            float(v)
            for v in target_pose
        ],
    }

    # 書き込み途中のJSONを残さないため
    # 一度tmpへ書いてから置換する
    tmp_path = Path(
        str(RETRIEVAL_MOTION_STATE_PATH)
        + ".tmp"
    )

    tmp_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    tmp_path.replace(
        RETRIEVAL_MOTION_STATE_PATH
    )

    print(
        "[RETRIEVAL MOTION STATE SAVE] "
        f"side={side}, "
        f"target_X={target_pose[0]:.2f} mm"
    )


def load_retrieval_motion_state():
    if not RETRIEVAL_MOTION_STATE_PATH.exists():
        return None

    try:
        data = json.loads(
            RETRIEVAL_MOTION_STATE_PATH.read_text(
                encoding="utf-8"
            )
        )

        target_pose = [
            float(v)
            for v in data["target_pose"]
        ]

        if len(target_pose) != 6:
            raise RuntimeError(
                "target_pose must have 6 elements"
            )

        side = str(
            data["side"]
        )

        if side not in {
            "right",
            "left",
        }:
            raise RuntimeError(
                f"invalid side: {side}"
            )

        data["target_pose"] = (
            target_pose
        )

        print(
            "[RETRIEVAL MOTION STATE RESTORE] "
            f"side={side}, "
            f"target_X={target_pose[0]:.2f} mm"
        )

        return data

    except Exception as e:
        print(
            "[RETRIEVAL MOTION STATE RESTORE] failed: "
            f"{type(e).__name__}: {e}"
        )

        return None
    

def get_container_joint_motion_params(
    lift_height_mm: float,
) -> tuple[float, float]:

    h = float(lift_height_mm)

    if h >= 800.0:
        velocity = 1.0
        acceleration = 2.0

    else:
        velocity = 0.5
        acceleration = 1.0

    print(
        "[CONTAINER JOINT MOTION] "
        f"lift_height={h:.1f} mm, "
        f"velocity={velocity:.2f}, "
        f"acceleration={acceleration:.2f}"
    )

    return velocity, acceleration

# J1電流閾値を検知した場合の最大試行回数（初回を含む）
# 4回すべて失敗した場合は、その本を失敗扱いにして次の本へ進む。
MAX_CURRENT_INSERT_ATTEMPTS = 4

# 画像認識の最大試行回数（初回 + 再撮影2回）
# 3回すべて失敗した場合にだけ、初期姿勢へ戻して次の本へ進む。
MAX_RECOGNITION_ATTEMPTS = 3

EMPTY_GRASP_POSITION_MIN = 3700
EMPTY_GRASP_POSITION_MAX = 4000

LOG_HEADER = [
    "timestamp",
    "book_name",
    "shelf_id",
    "roll_deg",
    "estimated_book_width_mm",
    "master_book_width_mm",
    "width_error_mm",
    "camera_x_mm",
    "camera_y_mm",
    "camera_z_mm",
    "robot_x_mm",
    "robot_y_mm",
    "robot_z_mm",
    "side",
    "height_mm",
    "result",
    "shot_dir",
    "memo",
    "max_j1_current_abs",
]


def _optional_float(value):
    """Return a finite float, or None when the value is blank/invalid."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def emergency_stop_linear_lift():
    if globals().get("_lift_estop_sent", False):
        print("[LIFT E-STOP] already sent")
        return True

    pub = globals().get("lift_estop_pub", None)

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

        globals()["_lift_estop_sent"] = True

        print("[LIFT E-STOP] emergency stop command sent")
        return True

    except Exception as e:
        print(
            "[LIFT E-STOP] failed: "
            f"{type(e).__name__}: {e}"
        )
        return False

    except Exception as e:
        print(
            "[LIFT E-STOP] failed: "
            f"{type(e).__name__}: {e}"
        )
        return False


def release_linear_lift_estop():
    pub = globals().get(
        "lift_estop_pub",
        None,
    )

    if pub is None:
        print(
            "[LIFT E-STOP RELEASE] "
            "publisher is not initialized"
        )
        return False

    try:
        msg = Bool()
        msg.data = False

        print("")
        print("========================================")
        print("[LIFT E-STOP RELEASE] releasing...")
        print("========================================")

        # DDS取りこぼし対策で数回送る
        for _ in range(3):
            pub.publish(msg)
            time.sleep(0.1)

        # 次回のE-STOPを送れるように戻す
        globals()["_lift_estop_sent"] = False

        print(
            "[LIFT E-STOP RELEASE] "
            "/emergency_stop=False sent"
        )

        return True

    except Exception as e:
        print(
            "[LIFT E-STOP RELEASE] failed: "
            f"{type(e).__name__}: {e}"
        )
        return False


def monitor_emergency_stop(msg):

    print(
        "[SYSTEM E-STOP] "
        f"XArmMonitor detected abnormal: {msg}"
    )

    # 上下機構非常停止
    emergency_stop_linear_lift()
    
def _csv_value(value):
    """Write missing values as an empty CSV cell instead of the text 'None'."""
    return "" if value is None else value


def _xyz_csv_values(point_mm):
    """Return XYZ as three CSV-safe floats, or three empty cells."""
    if point_mm is None:
        return ["", "", ""]
    try:
        xyz = np.asarray(point_mm, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return ["", "", ""]
    if not np.all(np.isfinite(xyz)):
        return ["", "", ""]
    return [float(xyz[0]), float(xyz[1]), float(xyz[2])]


def _resolve_config_relative_path(config, raw_path):
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        config_dir = Path(config.get("_config_dir", SCRIPT_DIR))
        path = config_dir / path
    return path.resolve()


def _width_error_values(estimated_width_mm, master_width_mm):
    estimated = _optional_float(estimated_width_mm)
    master = _optional_float(master_width_mm)
    if estimated is None or master is None:
        return estimated, master, None
    error = estimated - master
    return estimated, master, error


def _short_shot_dir(shot_dir):
    """Return only the path beginning with /captures for CSV output."""
    if shot_dir is None or str(shot_dir).strip() in {"", "None"}:
        return ""

    path = Path(str(shot_dir)).expanduser()
    parts = path.parts

    # Keep the suffix from the first 'captures' component.
    try:
        captures_index = parts.index("captures")
    except ValueError:
        return str(path)

    suffix = Path(*parts[captures_index:]).as_posix()
    return f"/{suffix}"

def _ods_safe_value(value):
    """ODSへ書き込める値に変換する。"""
    if value is None:
        return ""

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float) and not math.isfinite(value):
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def _find_next_ods_row(sheet, column_count):
    """
    データが入っている最終行の次の行番号を返す。
    行番号は0始まり。
    """
    check_columns = min(column_count, sheet.ncols())

    for row_index in range(sheet.nrows() - 1, -1, -1):
        for col_index in range(check_columns):
            value = sheet[row_index, col_index].value

            if value is not None and str(value).strip() != "":
                return row_index + 1

    return 0

def write_log(
    config,
    book_name,
    shelf_id,
    roll_deg,
    estimated_book_width_mm,
    master_book_width_mm,
    camera_point_mm,
    robot_point_mm,
    side,
    height,
    result,
    shot_dir,
    memo,
    max_j1_current_abs=None,
):
    """retrieval_log.odsへ出庫結果を1行追記する。"""

    log_file = _resolve_config_relative_path(
        config,
        config["paths"]["log"]["retrieval"],
    )

    if not log_file.exists():
        raise FileNotFoundError(
            f"ODSログファイルが見つかりません: {log_file}"
        )

    estimated, master, error = _width_error_values(
        estimated_book_width_mm,
        master_book_width_mm,
    )

    camera_xyz = _xyz_csv_values(camera_point_mm)
    robot_xyz = _xyz_csv_values(robot_point_mm)

    # 現在のCSVと同じ18列
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        book_name,
        shelf_id,
        _csv_value(_optional_float(roll_deg)),
        _csv_value(estimated),
        _csv_value(master),
        _csv_value(error),
        *camera_xyz,
        *robot_xyz,
        side,
        height,
        result,
        _short_shot_dir(shot_dir),
        "" if memo is None else str(memo),
        _csv_value(_optional_float(max_j1_current_abs)),
    ]

    # ODSを開く
    document = ezodf.opendoc(str(log_file))

    if len(document.sheets) == 0:
        raise RuntimeError(
            f"ODS内にシートがありません: {log_file}"
        )

    # YAMLにretrieval_sheetがあればそのシートを使用
    sheet_name = (
        config.get("paths", {})
        .get("log", {})
        .get("retrieval_sheet")
    )

    if sheet_name:
        try:
            sheet = document.sheets[sheet_name]
        except KeyError:
            raise KeyError(
                f"ODS内にシート '{sheet_name}' がありません。"
                f"シート一覧: {list(document.sheets.names())}"
            )
    else:
        # 指定がない場合は一番左のシート
        sheet = document.sheets[0]

    required_columns = len(LOG_HEADER)

    # 18列未満なら列を追加
    if sheet.ncols() < required_columns:
        sheet.append_columns(
            required_columns - sheet.ncols()
        )

    # 行が存在しない場合
    if sheet.nrows() == 0:
        sheet.append_rows(1)

    # 各ヘッダセルを確認し、空欄の列だけヘッダ名を設定する
    # 既存ODSに新しい列を追加した場合にも対応できる
    for col, header in enumerate(LOG_HEADER):
        current_header = sheet[0, col].value

        if (
            current_header is None
            or str(current_header).strip() == ""
        ):
            sheet[0, col].set_value(header)

    # 最後のデータ行の次を取得
    target_row = _find_next_ods_row(
        sheet,
        required_columns,
    )

    # 足りない行を追加
    if target_row >= sheet.nrows():
        sheet.append_rows(
            target_row - sheet.nrows() + 1
        )

    # 18列を書き込む
    for col, value in enumerate(row):
        sheet[target_row, col].set_value(
            _ods_safe_value(value)
        )

    # 元ファイルへ保存
    document.save()

    print(f"[ODS LOG] saved: result={result}")
    print(f"[ODS LOG] path: {log_file}")
    print(f"[ODS LOG] sheet: {sheet.name}")
    print(f"[ODS LOG] row: {target_row + 1}")
    print(
        "[ODS LOG] width: "
        f"estimated={estimated}, master={master}, error={error}"
    )
    print(f"[ODS LOG] camera XYZ [mm]: {camera_xyz}")
    print(f"[ODS LOG] robot XYZ [mm]: {robot_xyz}")

def load_config(config_path):
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config
    
def sigint_handler(sig, frame):
    print("Ctrl+C detected → EMERGENCY STOP")

    # ==============================
    # 上下機構停止
    # ==============================
    try:
        emergency_stop_linear_lift()
    except Exception as e:
        print(
            "[LIFT E-STOP] failed in SIGINT: "
            f"{type(e).__name__}: {e}"
        )

    # ==============================
    # xArm停止
    # ==============================
    try:
        arm = globals().get("arm", None)

        if arm:
            arm.emergency_stop()

    except Exception as e:
        print(
            "[xArm E-STOP] failed: "
            f"{type(e).__name__}: {e}"
        )

    # ==============================
    # SAM3終了
    # ==============================
    service_session = globals().get(
        "_sam3_service_session"
    )

    if service_session is not None:
        try:
            service_session.stop_if_owned()
        except Exception as e:
            print(
                f"[SAM3] owned service cleanup failed: {e}"
            )

    # ROS publisherの送信時間
    time.sleep(0.2)

    os._exit(1)

signal.signal(signal.SIGINT, sigint_handler)


def release_linear_lift_estop():
    pub = globals().get(
        "lift_estop_pub",
        None,
    )

    if pub is None:
        print(
            "[LIFT E-STOP RELEASE] "
            "publisher is not initialized"
        )
        return False

    try:
        msg = Bool()
        msg.data = False

        print("")
        print("========================================")
        print("[LIFT E-STOP RELEASE] releasing...")
        print("========================================")

        # DDS取りこぼし対策で数回送る
        for _ in range(3):
            pub.publish(msg)
            time.sleep(0.1)

        # 次回のE-STOPを送れるように戻す
        globals()["_lift_estop_sent"] = False

        print(
            "[LIFT E-STOP RELEASE] "
            "/emergency_stop=False sent"
        )

        return True

    except Exception as e:
        print(
            "[LIFT E-STOP RELEASE] failed: "
            f"{type(e).__name__}: {e}"
        )
        return False


def hard_disconnect(arm):
    print("disconnect xArm NOW")
    try:
        try:
            arm.emergency_stop()
        except Exception:
            pass

        arm.disconnect()
    except Exception as e:
        print(f"disconnect failed: {e}")
        
def main_sequence(
    config,
    book_name: str,
    barcode_number: str,
    bookshelf_ID: str,
    book_width_offset: float,
    master_book_width_mm: float | None,
    tp: TargetPublisher,
    node,
    arm: XArm7,
    executor,
    waypoint_node: WaypointPlayerNode,
    shelf_manager: ShelfIDManager,
    monitor: XArmMonitor,
    done_pub,
    stage_update_pub,
    HandMotors_retrieval,
    resume_stage=None,
    recognition_restart_mode=False,
    inserting_restart_mode=False,
):

    # ==================================================
    # 統合sequence stage / resume制御
    # ==================================================
    if resume_stage is not None:
        resume_stage = str(
            resume_stage
        ).strip()

        if not resume_stage:
            resume_stage = None

    retrieval_resume_control = {
        "waiting": resume_stage is not None
    }


    def set_retrieval_stage(
        stage: str,
    ) -> bool:

        stage = str(
            stage
        ).strip()

        if not stage:
            return False

        # ==============================================
        # 復帰stageに到達するまではSKIP
        # ==============================================
        if retrieval_resume_control["waiting"]:

            if stage != resume_stage:

                node.get_logger().info(
                    "[RETRIEVAL SKIP] "
                    f"{stage}"
                )

                return False

            retrieval_resume_control[
                "waiting"
            ] = False

            node.get_logger().warn(
                "========================================"
            )
            node.get_logger().warn(
                "[RETRIEVAL RESUME]"
            )
            node.get_logger().warn(
                f"resume from: {stage}"
            )
            node.get_logger().warn(
                "========================================"
            )

        # ==============================================
        # 通常stage通知
        # ==============================================
        msg = String()
        msg.data = stage

        stage_update_pub.publish(
            msg
        )

        node.get_logger().info(
            "[RETRIEVAL STAGE UPDATE] "
            f"{stage}"
        )

        return True

    def assert_xarm_normal(where: str):
        state = arm.get_state()
        err, warn = arm.get_err_warn()

        node.get_logger().info(
            f"[XARM CHECK] {where}: "
            f"state={state}, err={err}, warn={warn}"
        )

        if state in (4, 5) or err != 0:
            raise RuntimeError(
                f"xArm abnormal at {where}: "
                f"state={state}, err={err}, warn={warn}"
            )

    runtime_log = {
        "roll_deg": None,
        "estimated_book_width_mm": None,
        "camera_point_mm": None,
        "robot_point_mm": None,
        "shot_dir": None,
        "insert_attempt": 0,
        "recognition_attempt": 0,
        "safe_stop_logged": False,
        "error_logged": False,
    }
    side = None
    height = None
    shelf_id = bookshelf_ID
    shot_dir = None
    roll = None
    p_xmax = None
    book_width = None
    successful_insert_attempt = None
    successful_recognition_attempt = None
    max_j1_current_abs = None
    container_full = False

    # ==================================================
    # 復帰モード判定
    #
    # GRASPING_BOOK:
    #   把持から直接再開
    #
    # RETRIEVING_BOOK:
    #   保存した引き抜き完了絶対姿勢へ移動
    #
    # BOOK_POSITIONING / HAND_OPENING:
    #   撮影姿勢へ戻って再認識
    # ==================================================
    retrieving_resume_mode = (
        resume_stage
        == "RETRIEVING_BOOK"
    )

    hand_resume_mode = (
        resume_stage
        in {
            "GRASPING_BOOK",
            "RETRIEVING_BOOK",
        }
    )

    hand_resume_state = None

    if (
        hand_resume_mode
        or recognition_restart_mode
    ):

        hand_resume_state = (
            load_hand_resume_state()
        )

        if hand_resume_state is None:
            raise RuntimeError(
                "復帰用状態が保存されていません"
            )

        node.get_logger().warn(
            "========================================"
        )

        if recognition_restart_mode:

            node.get_logger().warn(
                "[RECOGNITION RESTART MODE]"
            )

        else:

            node.get_logger().warn(
                "[HAND AUTO RESUME]"
            )

            node.get_logger().warn(
                f"stage={resume_stage}"
            )

        node.get_logger().warn(
            "========================================"
        )

    
    try:
        print('start sequence')
        waypoint_node.reset()
        # ==============================
        # shelf_id 受信待ち
        # ==============================
        node.get_logger().info("Waiting for /shelf_id ...")

        while rclpy.ok() and not shelf_manager.is_received():
            executor.spin_once(timeout_sec=0.1)

        side = shelf_manager.get_side()
        height = shelf_manager.get_height()
        shelf_id = shelf_manager.get_shelf_id()
        tcp_offset = shelf_manager.get_tcp_z_offset()

        def on_abnormal(msg):
            # Some motion errors can call the callback more than once.
            if runtime_log["safe_stop_logged"]:
                return
            try:
                write_log(
                    config=config,
                    book_name=book_name,
                    shelf_id=shelf_id,
                    roll_deg=runtime_log["roll_deg"],
                    estimated_book_width_mm=runtime_log[
                        "estimated_book_width_mm"
                    ],
                    master_book_width_mm=master_book_width_mm,
                    camera_point_mm=runtime_log["camera_point_mm"],
                    robot_point_mm=runtime_log["robot_point_mm"],
                    side=side,
                    height=height,
                    result="safe_stop",
                    shot_dir=runtime_log["shot_dir"],
                    memo=msg,
                )
                runtime_log["safe_stop_logged"] = True
                runtime_log["error_logged"] = True
            except Exception as log_exc:
                print(f"[CSV LOG ERROR] safe_stop logging failed: {log_exc}")

        monitor.on_abnormal = on_abnormal

        print("Shelf side:", side)
        print("Lift height:", height)
        # ==============================
        print("xarm ready")

        if not (
            hand_resume_mode
            or recognition_restart_mode
        ):

            safe_motion(
                lambda: arm.moveJ_to_init_Q_DEG(),
                monitor,
                "init_pose",
            )

        else:

            node.get_logger().warn(
                "[AUTO RESUME] "
                "init_pose SKIP"
            )

        bar_dir = Path(config["paths"]["capture"]["bookshelf_barcode"])
        bar_dir.mkdir(parents=True, exist_ok=True)

        wall_watcher = WallDistanceWatcher(node)

        # 少し待って最新値を受信
        timeout_sec = 2.0
        start = time.time()
        while rclpy.ok() and wall_watcher.get_distance() is None:
            executor.spin_once(timeout_sec=0.1)
            if time.time() - start > timeout_sec:
                break

        wall_distance = wall_watcher.get_distance()

        if wall_distance is None:
            node.get_logger().warn("wall_distance not received. Using default 0.25m")
            wall_distance = 0.25

        # ==============================
        # self-localization
        # ==============================
        # detected2, code_str, info = capture_barcode_and_x_offset(
        #     node=node,
        #     executor=executor,
        #     shot_dir=bar_dir,
        #     barcode_number=None,
        #     fx_px=2500.0,
        #     depth_m= 0.8 - wall_distance - 0.13 , # 0.8 : 通路幅， 0.13 : AMR中心からカメラまでの距離
        #     v_x=0.01,        # 好きな低速
        #     cmd_sign_x=1.0,  # 逆なら -1.0
        # )

        # ==============================
        # wall_distance watcher
        # ==============================
        wall_watcher = WallDistanceWatcher(node)

        # ==============================
        # navigation_goal / navigation_goal_final watchers
        # ==============================
        nav_goal_pulse = BoolPulseWatcher(node, "/navigation_goal")
        final_goal = BoolLatchWatcher(node, "/navigation_goal_final")

        node.get_logger().info("Start self-localization loop: wait /navigation_goal pulses until /navigation_goal_final==True")

        # デバッグ上書き回避したいならサブディレクトリを毎回作るのがおすすめ
        # 例: bar_dir / time.strftime("%Y%m%d_%H%M%S")
        bar_dir = Path("/home/book/pro_book_SAM3/pro_hand_book_python/captures/bookshelf_barcode")
        bar_dir.mkdir(parents=True, exist_ok=True)

        detected2 = False
        label_str = None
        info = None

        # while rclpy.ok() and not final_goal.is_true():
        #     executor.spin_once(timeout_sec=0.1)

        #     # /navigation_goal=True を受けるまで待つ
        #     if not nav_goal_pulse.consume():
        #         continue

        #     node.get_logger().info(
        #         "[bookshelf barcode] /navigation_goal received -> start self-localization"
        #     )

        #     wall_distance = wall_watcher.get_distance()
        #     if wall_distance is None:
        #         node.get_logger().warn("wall_distance not received. Using default 0.25m")
        #         wall_distance = 0.25

        #     detected2, label_str, info = capture_barcode_and_x_offset(
        #         node=node,
        #         executor=executor,
        #         shot_dir=bar_dir,
        #         fx_px=2500.0,
        #         depth_m=0.8 - wall_distance - 0.13,
        #         v_x=0.05,
        #         min_search_vx=0.02,
        #         max_search_vx=0.07,
        #         k_p_search=0.00008,
        #         cmd_sign_x=1.0,
        #         align_thresh_px=10.0,
        #         wait_navigation_goal=False,   # 外で待っているので False
        #         total_timeout_sec=1000.0,
        #         ocr_interval_sec=0.25,
        #         bbox_lost_grace_sec=0.5,
        #     )

        #     if detected2 and info is not None:
        #         node.get_logger().info(
        #             f"[bookshelf barcode] self-localization OK: X_offset={info['X_m']:.3f} [m]"
        #         )
        #     else:
        #         node.get_logger().warn(
        #             "[bookshelf barcode] self-localization failed"
        #         )

        # ==============================
        # ==============================
        # init → capture 姿勢へ
        # ==============================
        if not (
            hand_resume_mode
            or recognition_restart_mode
        ):

            # ==========================================
            # 通常起動
            # ==========================================
            node.get_logger().info(
                "Waiting for manual "
                "/navigation_goal_final"
            )

            while (
                rclpy.ok()
                and not waypoint_node.is_finished()
            ):
                executor.spin_once(
                    timeout_sec=0.1
                )

            if waypoint_node.is_failed():
                raise RuntimeError(
                    "Waypoint failed: "
                    f"{waypoint_node.error_message()}"
                )

            node.get_logger().info(
                "Waypoint succeeded "
                "→ start recognition"
            )

            print("TCP調整開始")

            safe_motion(
                lambda: arm.moveL_tcp_z_offset(
                    tcp_offset
                ),
                monitor,
                "tcp_z_offset",
            )

            time.sleep(1.0)

            capture_pose_for_retry = (
                arm.get_tcp_pose(
                    is_radian=True
                )
            )


        elif recognition_restart_mode:

            # ==========================================
            # BOOK_POSITIONING または
            # HAND_OPENING で停止
            #
            # 保存済み撮影姿勢へ戻して
            # 認識からやり直す
            # ==========================================
            capture_pose_for_retry = [
                float(v)
                for v in hand_resume_state[
                    "capture_pose_for_retry"
                ]
            ]

            node.get_logger().warn(
                "========================================"
            )

            node.get_logger().warn(
                "[RECOGNITION RESTART]"
            )

            node.get_logger().warn(
                "保存済み撮影姿勢へ戻します"
            )

            node.get_logger().warn(
                "========================================"
            )

            safe_motion(
                lambda: arm._moveL(
                    capture_pose_for_retry,
                    velocity=80.0,
                    acceleration=40.0,
                    asynchronous=False,
                ),
                monitor,
                "restart_return_capture_pose",
            )

            node.get_logger().info(
                "[RECOGNITION RESTART] "
                "撮影姿勢への復帰完了"
            )

            # ==========================================
            # INSERTING途中からの復帰
            #
            # 挿入途中ではハンドが開いているので、
            # まず撮影姿勢まで退避する。
            # その後ハンドを閉じてからfresh recognitionへ進む。
            # ==========================================
            if inserting_restart_mode:

                node.get_logger().warn(
                    "========================================"
                )

                node.get_logger().warn(
                    "[INSERTING RESTART]"
                )

                node.get_logger().warn(
                    "撮影姿勢へ退避済み "
                    "-> ハンドを閉じて再認識します"
                )

                node.get_logger().warn(
                    "========================================"
                )

                try:
                    HandMotors_retrieval.grasp()

                except Exception as hand_exc:
                    raise RuntimeError(
                        "INSERTING復帰時の"
                        "ハンド閉鎖に失敗しました: "
                        f"{hand_exc}"
                    ) from hand_exc

                time.sleep(1.0)

                node.get_logger().info(
                    "[INSERTING RESTART] "
                    "ハンド閉鎖完了 "
                    "-> fresh recognition"
                )


        else:

            # ==========================================
            # GRASPING_BOOKから直接復帰
            #
            # アームは現在位置を維持
            # ==========================================
            capture_pose_for_retry = [
                float(v)
                for v in hand_resume_state[
                    "capture_pose_for_retry"
                ]
            ]

            node.get_logger().warn(
                "[HAND AUTO RESUME] "
                "capture movement SKIP"
            )

        print(
            "[CAPTURE POSE SAVED] "
            f"X={capture_pose_for_retry[0]:.2f} mm, "
            f"Y={capture_pose_for_retry[1]:.2f} mm, "
            f"Z={capture_pose_for_retry[2]:.2f} mm, "
            f"Roll={np.degrees(capture_pose_for_retry[3]):.2f} deg, "
            f"Pitch={np.degrees(capture_pose_for_retry[4]):.2f} deg, "
            f"Yaw={np.degrees(capture_pose_for_retry[5]):.2f} deg"
        )

        # 最初の認識で得られた本のロボット座標Y
        if hand_resume_mode:

            first_recognition_robot_y_mm = float(
                hand_resume_state[
                    "first_recognition_robot_y_mm"
                ]
            )

        else:

            first_recognition_robot_y_mm = None


        def return_to_original_capture_y():
            """
            再認識時に変更したYだけ、通常の撮影姿勢へ戻す。
            X、Z、Roll、Pitch、Yawはすでに撮影姿勢と同じ前提。
            """
            current_pose = arm.get_tcp_pose(
                is_radian=True
            )

            dy_mm = (
                float(capture_pose_for_retry[1])
                - float(current_pose[1])
            )

            print(
                "[RETURN CAPTURE Y] "
                f"current_y={current_pose[1]:.2f} mm, "
                f"capture_y={capture_pose_for_retry[1]:.2f} mm, "
                f"dY={dy_mm:+.2f} mm"
            )

            # ほぼ同じ位置なら移動しない
            if abs(dy_mm) < 0.5:
                print(
                    "[RETURN CAPTURE Y] "
                    "すでに通常撮影Y付近です。"
                )
                return

            safe_motion(
                lambda: arm.moveL_relative(
                    [
                        0.0,
                        dy_mm,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    velocity=80.0,
                    acceleration=40.0,
                    asynchronous=False,
                ),
                monitor,
                "return_original_capture_y",
            )

            print(
                "[RETURN CAPTURE Y] "
                "通常撮影位置のYへ戻りました。"
            )


        try:
            for insert_attempt in range(
                1,
                MAX_CURRENT_INSERT_ATTEMPTS + 1,
            ):
                runtime_log["insert_attempt"] = insert_attempt

                print(
                    f"[RETRIEVAL] recognition/insert attempt "
                    f"{insert_attempt}/"
                    f"{MAX_CURRENT_INSERT_ATTEMPTS}"
                )

                # ここではTCP高さ補正を再実行しない

                # ================== 認識 ==================
                # 認識エラー時は撮影姿勢を維持したまま再撮影し、
                # 初回を含めて最大3回まで認識する。
                successful_recognition_attempt = None

                for recognition_attempt in range(
                    1,
                    MAX_RECOGNITION_ATTEMPTS + 1,
                ):
                    runtime_log["recognition_attempt"] = recognition_attempt


                    # ======================================
                    # HAND復帰時は画像認識をSKIP
                    # ======================================
                    if hand_resume_mode:

                        start = (
                            time.perf_counter()
                        )

                        roll = float(
                            hand_resume_state[
                                "raw_roll_rad"
                            ]
                        )

                        p_xmax = np.asarray(
                            hand_resume_state[
                                "p_xmax_m"
                            ],
                            dtype=np.float64,
                        )

                        book_width = float(
                            hand_resume_state[
                                "book_width_mm"
                            ]
                        )

                        shot_dir = Path(
                            hand_resume_state[
                                "shot_dir"
                            ]
                        )

                        runtime_log[
                            "estimated_book_width_mm"
                        ] = book_width

                        runtime_log[
                            "shot_dir"
                        ] = shot_dir

                        successful_recognition_attempt = 1

                        node.get_logger().warn(
                            "[HAND AUTO RESUME] "
                            "recognition SKIP"
                        )

                        break

                    # 前回の認識結果を次の試行へ持ち越さない。
                    roll = None
                    p_xmax = None
                    book_width = None
                    shot_dir = None
                    runtime_log["roll_deg"] = None
                    runtime_log["estimated_book_width_mm"] = None
                    runtime_log["camera_point_mm"] = None
                    runtime_log["robot_point_mm"] = None
                    runtime_log["shot_dir"] = None

                    print(
                        "[RECOGNITION] capture/recognition attempt "
                        f"{recognition_attempt}/"
                        f"{MAX_RECOGNITION_ATTEMPTS}"
                    )
                    start = time.perf_counter()

                    try:
                        # この関数を呼ぶたびに新しく撮影・認識する。
                        roll, p_xmax, book_width, shot_dir = (
                            run_capture_and_pca_sam3_refined_sam2_width(
                                query=book_name,
                                sam_device="gpu",
                            )
                        )

                        shot_dir = Path(shot_dir)

                        runtime_log["roll_deg"] = float(
                            np.degrees(roll)
                        )
                        runtime_log["estimated_book_width_mm"] = float(
                            book_width
                        )
                        save_book_width(
                            book_width
                        )
                        runtime_log["shot_dir"] = shot_dir

                        print(
                            f"""
                            ===== PCA RESULT =====
                            roll        : {roll}
                            p_xmax      : {p_xmax}
                            book_width  : {book_width}
                            ======================
                            """
                        )

                        if p_xmax is None:
                            raise RuntimeError(
                                "Recognition failed: p_xmax is None"
                            )

                        # p_xmaxはカメラ座標系[m]。ログでは[mm]にする。
                        runtime_log["camera_point_mm"] = (
                            np.asarray(
                                p_xmax,
                                dtype=np.float64,
                            ).reshape(3)
                            * 1000.0
                        )

                        successful_recognition_attempt = (
                            recognition_attempt
                        )
                        print(
                            "[RECOGNITION] succeeded: "
                            f"attempt={recognition_attempt}/"
                            f"{MAX_RECOGNITION_ATTEMPTS}"
                        )
                        break

                    except Exception as e:
                        is_last_recognition_attempt = (
                            recognition_attempt
                            >= MAX_RECOGNITION_ATTEMPTS
                        )
                        recognition_result = (
                            "recognition_fail"
                            if is_last_recognition_attempt
                            else "recognition_retry"
                        )
                        next_action = (
                            "return to init and skip this book"
                            if is_last_recognition_attempt
                            else "recapture and retry recognition"
                        )

                        print(
                            "[RECOGNITION] failed: "
                            f"attempt={recognition_attempt}/"
                            f"{MAX_RECOGNITION_ATTEMPTS}; "
                            f"next={next_action}; error={e}"
                        )

                        write_log(
                            config=config,
                            book_name=book_name,
                            shelf_id=shelf_id,
                            roll_deg=runtime_log["roll_deg"],
                            estimated_book_width_mm=runtime_log[
                                "estimated_book_width_mm"
                            ],
                            master_book_width_mm=master_book_width_mm,
                            camera_point_mm=runtime_log[
                                "camera_point_mm"
                            ],
                            robot_point_mm=runtime_log[
                                "robot_point_mm"
                            ],
                            side=side,
                            height=height,
                            result=recognition_result,
                            shot_dir=runtime_log["shot_dir"],
                            memo=(
                                f"insert_attempt={insert_attempt}/"
                                f"{MAX_CURRENT_INSERT_ATTEMPTS}; "
                                f"recognition_attempt="
                                f"{recognition_attempt}/"
                                f"{MAX_RECOGNITION_ATTEMPTS}; "
                                f"recognition_fail_count="
                                f"{recognition_attempt}; "
                                f"next_action={next_action}; "
                                f"{type(e).__name__}: {e}"
                            ),
                        )
                        traceback.print_exc()

                        if not is_last_recognition_attempt:
                            print(
                                "[RECOGNITION RETRY] "
                                "撮影姿勢のまま再撮影します。"
                            )
                            time.sleep(1.0)
                            continue

                        # 3回すべて認識失敗した場合だけ、
                        # 初期姿勢へ戻して次の本へ進む。

                        # 電流リトライ後は本正面のYにいる可能性があるため、
                        # 通常の撮影姿勢Yへ戻す。
                        return_to_original_capture_y()

                        tp.publish_target_mm(
                            config["linear_lift"]["home_mm"]
                        )
                        rclpy.spin_once(tp, timeout_sec=0.1)

                        waypoint_node.reset()
                        waypoint_path = config["paths"]["waypoint"][
                            "capture_to_init"
                        ][side]
                        waypoint_node.play_direct(waypoint_path)

                        while (
                            rclpy.ok()
                            and not waypoint_node.is_finished()
                        ):
                            executor.spin_once(timeout_sec=0.1)

                        if waypoint_node.is_failed():
                            raise RuntimeError(
                                "認識3回失敗後の初期姿勢復帰に失敗: "
                                f"{waypoint_node.error_message()}"
                            )

                        shelf_manager.received = False

                        return (
                            0.0,
                            HandMotors_retrieval,
                            False,
                        )

                end = time.perf_counter()
                print(f"{end - start} sec.")

                print("roll (deg) =", np.degrees(roll))
                raw_roll_for_resume = float(
                    roll
                )
                if np.degrees(roll) > 90.0:
                    roll = -(roll - np.radians(90.0))
                elif np.degrees(roll) < -90.0:
                    roll = -(roll + np.radians(90.0))
                else:
                    roll = 0.0

                out = {
                    "adjusted_roll_rad": float(roll),
                    "adjusted_roll_deg": float(
                        np.degrees(roll)
                    ),
                }
                (shot_dir / "adjusted_roll.json").write_text(
                    json.dumps(
                        out,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                print(
                    "adjusted roll (deg) =",
                    np.degrees(roll),
                )
                runtime_log["roll_deg"] = float(
                    np.degrees(roll)
                )

                p_max = 1000 * p_xmax

                print_camera_debug_info(
                    arm,
                    p_max,
                )

                # ロボット座標系での対象点[mm]
                if hand_resume_mode:

                    # GRASPING_BOOK直接復帰時は
                    # 再起動後の現在姿勢から
                    # p_robot_mmを再計算しない
                    p_robot_mm = np.asarray(
                        hand_resume_state[
                            "p_robot_mm"
                        ],
                        dtype=np.float64,
                    )

                else:

                    p_robot_mm = cam_mm_to_robot_mm(
                        arm,
                        p_max,
                    )

                # 最初に認識成功したときの本のロボット座標Yを保存
                if first_recognition_robot_y_mm is None:
                    first_recognition_robot_y_mm = float(
                        p_robot_mm[1]
                    )

                    print(
                        "[FIRST RECOGNITION Y SAVED] "
                        f"robot_y={first_recognition_robot_y_mm:.2f} mm"
                    )
                else:
                    print(
                        "[RETRY RECOGNITION] "
                        "最初に保存したY座標を維持します: "
                        f"robot_y={first_recognition_robot_y_mm:.2f} mm"
                    )

                runtime_log["camera_point_mm"] = np.asarray(
                    p_max,
                    dtype=np.float64,
                ).reshape(3).copy()
                runtime_log["robot_point_mm"] = np.asarray(
                    p_robot_mm,
                    dtype=np.float64,
                ).reshape(3).copy()

                # ==============================================
                # 復帰用状態を保存
                #
                # 必ずBOOK_POSITIONING通知より前に保存する
                # ==============================================
                if not hand_resume_mode:

                    save_hand_resume_state(
                        book_width=book_width,
                        raw_roll_rad=raw_roll_for_resume,
                        p_xmax=p_xmax,
                        p_robot_mm=p_robot_mm,
                        shot_dir=shot_dir,
                        capture_pose_for_retry=(
                            capture_pose_for_retry
                        ),
                        first_recognition_robot_y_mm=(
                            first_recognition_robot_y_mm
                        ),
                    )


                # ==============================================
                # 本正面への位置決め
                # ==============================================
                if set_retrieval_stage(
                    "BOOK_POSITIONING"
                ):

                    safe_motion(
                        lambda: arm.move_to_target_xyz_and_roll(
                            p_robot_mm=p_robot_mm,
                            d_roll_rad=roll,
                            side=side,
                        ),
                        monitor,
                        "insertz_before",
                    )


                # ==============================================
                # ハンドOPEN
                # ==============================================
                if set_retrieval_stage(
                    "HAND_OPENING"
                ):

                    HandMotors_retrieval = (
                        HandMotors_retrieval.open_until_width(
                            book_width,
                            gravity=False,
                        )
                    )


                set_retrieval_stage(
                    "HAND_OPEN_DONE"
                )
                # ================== 挿入・取り出し ==================
                #input("insert: Enter / ""return to capture: Ctrl+D / ""exit: Ctrl+C")
                time.sleep(3.0)

                do_insert = (
                    set_retrieval_stage(
                        "INSERTING"
                    )
                )

                if do_insert:

                    insert_result = (
                        arm.moveL_to_insert_right_with_current_monitor(
                            side=side,
                            retry_capture_pose=(
                                capture_pose_for_retry
                            ),
                            retry_capture_y_mm=(
                                first_recognition_robot_y_mm
                            ),
                            retry_y_offset_mm=0.0,
                        )
                    )

                else:

                    # GRASPING_BOOK復帰時は
                    # すでに挿入済みなので実動作をSKIP
                    insert_result = {
                        "ok": True,
                        "max_j1_current_abs": None,
                    }

                    node.get_logger().warn(
                        "[HAND AUTO RESUME] "
                        "INSERTING SKIP"
                    )

                max_j1_current_abs = _optional_float(
                    insert_result.get("max_j1_current_abs")
                )

                print(
                    "[J1 CURRENT] "
                    f"insert_attempt={insert_attempt}, "
                    f"max_abs={max_j1_current_abs}"
                )

                if not insert_result["ok"]:
                    reason = insert_result.get(
                        "reason",
                        "unknown",
                    )

                    if reason == "current_threshold":
                        retry_target_y_mm = insert_result.get(
                            "retry_target_y_mm"
                        )

                        print(
                            "[RETRY] J1電流閾値を検知。"
                            f"side={side}, "
                            "撮影姿勢のX/Z/RPYを維持し、"
                            f"Y={retry_target_y_mm} mmへ移動したため、"
                            "同じ本を再認識します。"
                        )

                        is_last_attempt = (
                            insert_attempt
                            >= MAX_CURRENT_INSERT_ATTEMPTS
                        )

                        retry_result = (
                            "current_threshold_fail"
                            if is_last_attempt
                            else "current_threshold_retry"
                        )

                        retry_memo = (
                            f"attempt={insert_attempt}/"
                            f"{MAX_CURRENT_INSERT_ATTEMPTS}; "
                            f"retry_count={insert_attempt}; "
                            f"side={side}; "
                            f"retry_target_y_mm={retry_target_y_mm}; "
                            "J1 current threshold detected; "
                            + (
                                "container storage skipped"
                                if is_last_attempt
                                else "retry same book"
                            )
                        )

                        write_log(
                            config=config,
                            book_name=book_name,
                            shelf_id=shelf_id,
                            roll_deg=runtime_log["roll_deg"],
                            estimated_book_width_mm=runtime_log[
                                "estimated_book_width_mm"
                            ],
                            master_book_width_mm=master_book_width_mm,
                            camera_point_mm=runtime_log[
                                "camera_point_mm"
                            ],
                            robot_point_mm=runtime_log[
                                "robot_point_mm"
                            ],
                            side=side,
                            height=height,
                            result=retry_result,
                            shot_dir=runtime_log["shot_dir"],
                            memo=retry_memo,
                            max_j1_current_abs=max_j1_current_abs,
                        )

                        print(
                            "[ODS RETRY LOG] "
                            f"attempt={insert_attempt}/"
                            f"{MAX_CURRENT_INSERT_ATTEMPTS}, "
                            f"result={retry_result}"
                        )

                        # ハンドは本幅まで開いているので閉じる
                        print(
                            "[HAND RETRY] "
                            "再認識前にハンドを閉じます。"
                        )

                        try:
                            HandMotors_retrieval.grasp()
                        except Exception as hand_exc:
                            raise RuntimeError(
                                "電流検知後のハンド閉鎖に失敗しました: "
                                f"{hand_exc}"
                            ) from hand_exc

                        time.sleep(1.0)

                        print(
                            "[HAND RETRY] "
                            "ハンド閉鎖指令完了。"
                        )

                        # ==========================================
                        # 最大回数失敗した場合
                        # ==========================================
                        if is_last_attempt:
                            print(
                                "[SKIP] J1電流閾値を"
                                f"{MAX_CURRENT_INSERT_ATTEMPTS}"
                                "回連続で検知しました。"
                                "コンテナ収納を行わず、"
                                "次の本へ進みます。"
                            )

                            # 本正面のYから通常撮影姿勢のYへ戻す
                            return_to_original_capture_y()

                            tp.publish_target_mm(
                                config["linear_lift"]["home_mm"]
                            )
                            rclpy.spin_once(
                                tp,
                                timeout_sec=0.1,
                            )

                            waypoint_node.reset()

                            waypoint_path = (
                                config["paths"]["waypoint"]
                                ["capture_to_init"][side]
                            )

                            waypoint_node.play_direct(
                                waypoint_path
                            )

                            while (
                                rclpy.ok()
                                and not waypoint_node.is_finished()
                            ):
                                executor.spin_once(
                                    timeout_sec=0.1
                                )

                            if waypoint_node.is_failed():
                                raise RuntimeError(
                                    "電流閾値失敗後の"
                                    "初期姿勢復帰に失敗: "
                                    f"{waypoint_node.error_message()}"
                                )

                            shelf_manager.received = False


                            return (
                                0.0,
                                HandMotors_retrieval,
                                False,
                            )

                        # 本正面の撮影位置から、そのまま再認識
                        continue

                    raise RuntimeError(
                        f"{side}側挿入に失敗しました: "
                        f"{insert_result}"
                    )

                # xArmが保護停止していないことを確認
                assert_xarm_normal(
                    "after INSERTING"
                )

                set_retrieval_stage(
                    "INSERT_DONE"
                )

                # ==================================================
                # 挿入成功後：把持
                # ==================================================
                grasp_position = None

                if set_retrieval_stage(
                    "GRASPING_BOOK"
                ):

                    grasp_result = (
                        HandMotors_retrieval.grasp()
                    )

                    grasp_position = int(
                        grasp_result["position"]
                    )

                    # GRASPING_BOOKからの復帰が成功したら
                    # 以降は通常処理へ戻す
                    if resume_stage == "GRASPING_BOOK":

                        hand_resume_mode = False

                        node.get_logger().info(
                            "[HAND AUTO RESUME] "
                            "GRASPING_BOOK resume completed "
                            "-> normal mode"
                        )


                elif retrieving_resume_mode:

                    # ==========================================
                    # RETRIEVING_BOOK復帰時
                    #
                    # 本はすでに把持済みなので
                    # grasp()をもう一度実行しない
                    # ==========================================
                    node.get_logger().warn(
                        "[RETRIEVING BOOK RESUME] "
                        "GRASPING_BOOK SKIP "
                        "(book is already grasped)"
                    )


                else:

                    raise RuntimeError(
                        "GRASPING_BOOKが"
                        "予期せずSKIPされました"
                    )


                if grasp_position is not None:

                    print(
                        "[GRASP CHECK] "
                        f"position={grasp_position}, "
                        f"empty_range="
                        f"{EMPTY_GRASP_POSITION_MIN}"
                        f"~{EMPTY_GRASP_POSITION_MAX}"
                    )

                # ==================================================
                # 空把持判定
                # 3700～4000で止まったら本を掴めていないと判定
                # ==================================================
                if (
                    grasp_position is not None
                    and EMPTY_GRASP_POSITION_MIN
                    <= grasp_position
                    <= EMPTY_GRASP_POSITION_MAX
                ):

                    print(
                        "[EMPTY GRASP DETECTED] "
                        "本を把持できていないと判定しました。 "
                        f"position={grasp_position}"
                    )

                    is_last_attempt = (
                        insert_attempt
                        >= MAX_CURRENT_INSERT_ATTEMPTS
                    )

                    retry_result = (
                        "empty_grasp_fail"
                        if is_last_attempt
                        else "empty_grasp_retry"
                    )

                    retry_memo = (
                        f"attempt={insert_attempt}/"
                        f"{MAX_CURRENT_INSERT_ATTEMPTS}; "
                        f"grasp_position={grasp_position}; "
                        f"empty_range="
                        f"{EMPTY_GRASP_POSITION_MIN}"
                        f"~{EMPTY_GRASP_POSITION_MAX}; "
                        "empty grasp detected; "
                        + (
                            "container storage skipped"
                            if is_last_attempt
                            else "retry same book"
                        )
                    )

                    write_log(
                        config=config,
                        book_name=book_name,
                        shelf_id=shelf_id,
                        roll_deg=runtime_log["roll_deg"],
                        estimated_book_width_mm=runtime_log[
                            "estimated_book_width_mm"
                        ],
                        master_book_width_mm=master_book_width_mm,
                        camera_point_mm=runtime_log[
                            "camera_point_mm"
                        ],
                        robot_point_mm=runtime_log[
                            "robot_point_mm"
                        ],
                        side=side,
                        height=height,
                        result=retry_result,
                        shot_dir=runtime_log["shot_dir"],
                        memo=retry_memo,
                        max_j1_current_abs=max_j1_current_abs,
                    )

                    # ==============================================
                    # 空把持時：X方向だけ撮影位置まで退避
                    # ==============================================
                    print(
                        "[EMPTY GRASP RETRY] "
                        "空把持を検知したため、X方向だけ退避します。"
                    )

                    current_pose = arm.get_tcp_pose(
                        is_radian=True
                    )

                    current_pose = [
                        float(value)
                        for value in current_pose
                    ]

                    # 現在姿勢をそのままコピー
                    retry_pose = list(current_pose)

                    # Xだけ、保存してある撮影姿勢のXへ戻す
                    retry_pose[0] = float(
                        capture_pose_for_retry[0]
                    )

                    dx_mm = (
                        retry_pose[0]
                        - current_pose[0]
                    )

                    print("")
                    print("========== EMPTY GRASP X RETREAT ==========")

                    print(
                        "[CURRENT POSE] "
                        f"X={current_pose[0]:.2f} mm, "
                        f"Y={current_pose[1]:.2f} mm, "
                        f"Z={current_pose[2]:.2f} mm, "
                        f"Roll={np.degrees(current_pose[3]):.2f} deg, "
                        f"Pitch={np.degrees(current_pose[4]):.2f} deg, "
                        f"Yaw={np.degrees(current_pose[5]):.2f} deg"
                    )

                    print(
                        "[X RETREAT] "
                        f"current_X={current_pose[0]:.2f} mm -> "
                        f"capture_X={retry_pose[0]:.2f} mm "
                        f"(dX={dx_mm:+.2f} mm)"
                    )

                    safe_motion(
                        lambda: arm._moveL(
                            retry_pose,
                            velocity=80.0,
                            acceleration=40.0,
                            asynchronous=False,
                        ),
                        monitor,
                        "empty_grasp_x_retreat",
                    )

                    after_pose = arm.get_tcp_pose(
                        is_radian=True
                    )

                    print(
                        "[AFTER RETREAT] "
                        f"X={after_pose[0]:.2f} mm, "
                        f"Y={after_pose[1]:.2f} mm, "
                        f"Z={after_pose[2]:.2f} mm"
                    )

                    print(
                        "[EMPTY GRASP RETRY] "
                        "X方向だけの退避が完了しました。"
                    )

                    print("===========================================")
                    print("")
                    # ==============================================
                    # 最大回数ならこの本をスキップ
                    # ==============================================
                    if is_last_attempt:

                        print(
                            "[SKIP] "
                            f"空把持を{MAX_CURRENT_INSERT_ATTEMPTS}"
                            "回検出したため、"
                            "この本をスキップします。"
                        )

                        tp.publish_target_mm(
                            config["linear_lift"]["home_mm"]
                        )

                        rclpy.spin_once(
                            tp,
                            timeout_sec=0.1,
                        )

                        waypoint_node.reset()

                        waypoint_path = (
                            config["paths"]["waypoint"]
                            ["capture_to_init"][side]
                        )

                        waypoint_node.play_direct(
                            waypoint_path
                        )

                        while (
                            rclpy.ok()
                            and not waypoint_node.is_finished()
                        ):
                            executor.spin_once(
                                timeout_sec=0.1
                            )

                        if waypoint_node.is_failed():
                            raise RuntimeError(
                                "空把持失敗後の"
                                "初期姿勢復帰に失敗: "
                                f"{waypoint_node.error_message()}"
                            )

                        shelf_manager.received = False

                        return (
                            0.0,
                            HandMotors_retrieval,
                            False,
                        )

                    # ==============================================
                    # まだ試行回数が残っているなら再認識
                    # ==============================================
                    print(
                        "[EMPTY GRASP RETRY] "
                        "同じ本を再認識します。"
                    )

                    continue


                # ==================================================
                # 正常把持 / RETRIEVING_BOOK復帰
                # ==================================================
                if grasp_position is not None:

                    print(
                        "[GRASP CHECK] "
                        "正常把持と判定しました。 "
                        f"position={grasp_position}"
                    )

                else:

                    node.get_logger().warn(
                        "[RETRIEVING BOOK RESUME] "
                        "book already grasped"
                    )


                set_retrieval_stage(
                    "BOOK_GRASPED"
                )


                # ==============================================
                # 引き抜き完了の絶対TCP姿勢を準備
                #
                # 通常時:
                #   現在TCP ± RETRIEVAL_DX
                #
                # RETRIEVING_BOOK復帰:
                #   前回保存した絶対姿勢
                # ==============================================
                if retrieving_resume_mode:

                    retrieval_state = (
                        load_retrieval_motion_state()
                    )

                    if retrieval_state is None:

                        raise RuntimeError(
                            "RETRIEVING_BOOK復帰用の"
                            "状態が保存されていません"
                        )


                    # ==========================================
                    # 違う本の古いJSONを
                    # 間違って使用しないための確認
                    # ==========================================
                    if (
                        str(
                            retrieval_state.get(
                                "book_name"
                            )
                        )
                        != str(book_name)
                    ):

                        raise RuntimeError(
                            "RETRIEVING_BOOK復帰状態の"
                            "book_nameが一致しません: "
                            f"saved="
                            f"{retrieval_state.get('book_name')!r}, "
                            f"current={book_name!r}"
                        )


                    if (
                        str(
                            retrieval_state.get(
                                "bookshelf_id"
                            )
                        )
                        != str(bookshelf_ID)
                    ):

                        raise RuntimeError(
                            "RETRIEVING_BOOK復帰状態の"
                            "bookshelf_idが一致しません: "
                            f"saved="
                            f"{retrieval_state.get('bookshelf_id')!r}, "
                            f"current={bookshelf_ID!r}"
                        )


                    if (
                        str(
                            retrieval_state.get(
                                "side"
                            )
                        )
                        != str(side)
                    ):

                        raise RuntimeError(
                            "RETRIEVING_BOOK復帰状態の"
                            "sideが一致しません: "
                            f"saved="
                            f"{retrieval_state.get('side')!r}, "
                            f"current={side!r}"
                        )


                    retrieval_target_pose = [
                        float(v)
                        for v in retrieval_state[
                            "target_pose"
                        ]
                    ]

                    node.get_logger().warn(
                        "[RETRIEVING BOOK RESUME] "
                        "saved absolute target restored"
                    )


                else:

                    # ==========================================
                    # 通常処理
                    #
                    # 現在TCPから引き抜き完了位置を計算
                    # ==========================================
                    retrieval_start_pose = [
                        float(v)
                        for v in arm.get_tcp_pose(
                            is_radian=True
                        )
                    ]

                    retrieval_target_pose = list(
                        retrieval_start_pose
                    )


                    if side == "right":

                        retrieval_target_pose[0] -= (
                            RETRIEVAL_DX
                        )


                    elif side == "left":

                        retrieval_target_pose[0] += (
                            RETRIEVAL_DX
                        )


                    else:

                        raise ValueError(
                            "side must be "
                            "'right' or 'left'"
                        )


                    # ==========================================
                    # ★重要
                    #
                    # 必ずRETRIEVING_BOOKを
                    # managerへ通知するより前に保存
                    # ==========================================
                    save_retrieval_motion_state(
                        book_name=book_name,
                        bookshelf_id=bookshelf_ID,
                        side=side,
                        target_pose=(
                            retrieval_target_pose
                        ),
                    )


                # ==============================================
                # RETRIEVING_BOOK
                # ==============================================
                do_retrieve = (
                    set_retrieval_stage(
                        "RETRIEVING_BOOK"
                    )
                )


                if do_retrieve:

                    node.get_logger().info(
                        "[RETRIEVING BOOK] "
                        f"target X="
                        f"{retrieval_target_pose[0]:.2f} mm"
                    )


                    safe_motion(
                        lambda: arm._moveL(
                            retrieval_target_pose,
                            velocity=TCP_VEL_1,
                            acceleration=TCP_ACC_1,
                            asynchronous=False,
                        ),
                        monitor,
                        (
                            "resume_retrieving_book"
                            if retrieving_resume_mode
                            else "retrieving_book"
                        ),
                    )

                    # 引き抜きが本当に成功したか確認
                    assert_xarm_normal(
                        "after RETRIEVING_BOOK"
                    )

                    actual_pose = arm.get_tcp_pose(
                        is_radian=True
                    )

                    retrieval_error_mm = float(
                        np.linalg.norm(
                            np.asarray(actual_pose[:3])
                            - np.asarray(
                                retrieval_target_pose[:3]
                            )
                        )
                    )

                    node.get_logger().info(
                        "[RETRIEVING CHECK] "
                        f"error={retrieval_error_mm:.2f} mm"
                    )

                    if retrieval_error_mm > 5.0:
                        raise RuntimeError(
                            "引き抜き目標位置まで到達していません: "
                            f"error={retrieval_error_mm:.2f} mm"
                        )

                    # ==========================================
                    # RETRIEVING_BOOKからの復帰完了
                    # ==========================================
                    if retrieving_resume_mode:

                        retrieving_resume_mode = False
                        hand_resume_mode = False

                        node.get_logger().info(
                            "[RETRIEVING BOOK RESUME] "
                            "retrieval completed "
                            "-> normal mode"
                        )


                # ==============================================
                # 引き抜き完了
                # ==============================================
                set_retrieval_stage(
                    "BOOK_RETRIEVED"
                )

                successful_insert_attempt = insert_attempt
                break
            # ==========================
            # リニアリフトを収納高さへ移動
            # ==========================
            tp.publish_target_mm(
                config["linear_lift"]["move_to_container"]
            )
            rclpy.spin_once(tp, timeout_sec=0.1)

            # ==========================
            # 収納へ向かう関節姿勢へ移動
            # ==========================
            if side == "right":
                target_joint_deg = [
                    85.8,
                    -56.0,
                    173.0,
                    61.5,
                    100.1,
                    7.2,
                    41.7,
                ]

                target_joint_rad = np.radians(
                    target_joint_deg
                ).tolist()

                joint_velocity, joint_acceleration = (
                    get_container_joint_motion_params(
                        height
                    )
                )

                safe_motion(
                    lambda: arm.moveJ(
                        target_joint_rad,
                        velocity=joint_velocity,
                        acceleration=joint_acceleration,
                        asynchronous=True,
                    ),
                    monitor,
                    "right_container_joint_pose",
                )

            else:
                # 左側用の姿勢は未作成なので、現在は何もしない
                print(
                    "[MOVE TO CONTAINER POSE] "
                    "左側用の関節姿勢は未設定のため、移動しません"
                )
                pass


            # ==========================
            # 書籍バーコード認識
            #
            # book_barcode_sequence内部で、
            # 左面未検出の場合は
            # 右面への移動と再撮影まで実行する
            # ==========================
            barcode_status, container_side = (
                book_barcode_sequence(
                    barcode_number_input=barcode_number,
                    shot_dir=shot_dir,
                    arm=arm,
                    lift_height_mm=height,
                    stage_callback=set_retrieval_stage,
                )
            )

            # 正しいバーコードと一致した場合だけTrue
            barcode_result = (
                barcode_status == "success"
            )


            # ==========================
            # バーコード認識結果のログ
            # ==========================
            if barcode_status == "success":
                node.get_logger().info(
                    "[book barcode] success: "
                    "target barcode confirmed"
                )

            elif barcode_status == "wrong_barcode":
                node.get_logger().warn(
                    "[book barcode] wrong_barcode: "
                    "detected barcode does not match target "
                    "-> continue to container"
                )

            elif barcode_status == "no_barcode":
                node.get_logger().warn(
                    "[book barcode] no_barcode: "
                    "barcode was not detected "
                    "-> continue to container"
                )

            elif barcode_status == "error":
                node.get_logger().error(
                    "[book barcode] error: "
                    "barcode processing failed"
                )

            else:
                node.get_logger().error(
                    f"[book barcode] unknown status: "
                    f"{barcode_status!r}"
                )


            node.get_logger().info(
                f"[book barcode] container side: "
                f"{container_side}"
            )


            # ==================================================
            # コンテナ収納
            #
            # Move_to_Containerから
            #   HandMotors_retrieval
            #   container_full
            # の2つを受け取る
            # ==================================================
            container_result = []

            def move_to_container_with_handmotor():

                result = Move_to_Container(
                    book_width_offset,
                    book_width,
                    arm,
                    waypoint_node,
                    HandMotors_retrieval,
                    stage_callback=set_retrieval_stage,
                )

                container_result.append(result)


            safe_motion(
                move_to_container_with_handmotor,
                monitor,
                "Move_to_container",
            )


            # ==================================================
            # Move_to_Containerの戻り値確認
            # ==================================================
            if not container_result:
                raise RuntimeError(
                    "Move_to_Container did not return result"
                )

            if container_result[0] is None:
                raise RuntimeError(
                    "Move_to_Container returned None"
                )


            # ==================================================
            # 最新のDynamixelと満杯フラグを取得
            # ==================================================
            (
                HandMotors_retrieval,
                container_full,
            ) = container_result[0]


            if HandMotors_retrieval is None:
                raise RuntimeError(
                    "Move_to_Container returned None "
                    "for HandMotors_retrieval"
                )


            print(
                "[DXL MAIN] "
                "HandMotors_retrieval updated after "
                "Move_to_Container"
            )

            print(
                "[CONTAINER] "
                f"container_full={container_full}"
            )

            node.get_logger().info(
                "Move_to_Container succeeded"
            )

        except EOFError:
            #ctrl + D によってその書籍出庫はスキップし初期姿勢に戻る
            print("Ctrl+D detected → return to capture")
    
            if side == "right":
                safe_motion(lambda: arm.moveL_post_grasp_right() , monitor, "retreave_right")   #書籍を引き抜く   
            else:
                safe_motion(lambda: arm.moveL_post_grasp_left() , monitor, "retreave_left")   #書籍を引き抜く

            HandMotors_retrieval.grasp()
            tp.publish_target_mm(config["linear_lift"]["home_mm"])
            #初期姿勢へ戻る
            waypoint_node.reset()

            waypoint_node.play_direct(
                config["paths"]["waypoint"]["capture_to_init"][side]
            )

            while rclpy.ok() and not waypoint_node.is_finished():
                executor.spin_once(timeout_sec=0.1)

            memo = "manual_skip_before_insertion"

            write_log(
                config=config,
                book_name=book_name,
                shelf_id=shelf_id,
                roll_deg=runtime_log["roll_deg"],
                estimated_book_width_mm=runtime_log[
                    "estimated_book_width_mm"
                ],
                master_book_width_mm=master_book_width_mm,
                camera_point_mm=runtime_log["camera_point_mm"],
                robot_point_mm=runtime_log["robot_point_mm"],
                side=side,
                height=height,
                result="ctrl+d",
                shot_dir=runtime_log["shot_dir"],
                memo=memo,
            )
            rclpy.spin_once(tp, timeout_sec=0.1)

            return (
                0.0,
                HandMotors_retrieval,
                False,
            )# ← 次の本へ


        except Exception as e:
            print("xArm7 error")
            if not runtime_log["error_logged"]:
                write_log(
                    config=config,
                    book_name=book_name,
                    shelf_id=shelf_id,
                    roll_deg=runtime_log["roll_deg"],
                    estimated_book_width_mm=runtime_log[
                        "estimated_book_width_mm"
                    ],
                    master_book_width_mm=master_book_width_mm,
                    camera_point_mm=runtime_log["camera_point_mm"],
                    robot_point_mm=runtime_log["robot_point_mm"],
                    side=side,
                    height=height,
                    result="motion_error",
                    shot_dir=runtime_log["shot_dir"],
                    memo=f"{type(e).__name__}: {e}",
                )
                runtime_log["error_logged"] = True
            raise
        
        
        waypoint_node.reset()
        waypoint_node.play_direct(
            "/home/book/pro_book_SAM3/pro_hand_book_python/ros2_ws/src/xarm7_teaching/config/init.yaml"
        )


        wait_time = 1.5
        wait_start_time = time.time()

        while (
            rclpy.ok()
            and (time.time() - wait_start_time) < wait_time
        ):
            executor.spin_once(timeout_sec=0.1)

            if waypoint_node.is_finished():
                break
            
        tp.publish_target_mm(config["linear_lift"]["home_mm"])
        rclpy.spin_once(tp, timeout_sec=0.1)
        while rclpy.ok() and not waypoint_node.is_finished():
            executor.spin_once(timeout_sec=0.1)

        shelf_manager.received = False
        if successful_insert_attempt is None:
            successful_insert_attempt = max(
                1,
                int(runtime_log.get("insert_attempt", 1)),
            )

        if successful_recognition_attempt is None:
            successful_recognition_attempt = max(
                1,
                int(runtime_log.get("recognition_attempt", 1)),
            )

        memo = (
            f"insert_attempt={successful_insert_attempt}/"
            f"{MAX_CURRENT_INSERT_ATTEMPTS}; "
            f"insert_retry_count="
            f"{max(0, successful_insert_attempt - 1)}; "
            f"recognition_attempt="
            f"{successful_recognition_attempt}/"
            f"{MAX_RECOGNITION_ATTEMPTS}; "
            f"recognition_retry_count="
            f"{max(0, successful_recognition_attempt - 1)}; "
            f"barcode_result={barcode_result}; "
            "barcode_bypass=true"
        )
        write_log(
            config=config,
            book_name=book_name,
            shelf_id=shelf_id,
            roll_deg=runtime_log["roll_deg"],
            estimated_book_width_mm=runtime_log[
                "estimated_book_width_mm"
            ],
            master_book_width_mm=master_book_width_mm,
            camera_point_mm=runtime_log["camera_point_mm"],
            robot_point_mm=runtime_log["robot_point_mm"],
            side=side,
            height=height,
            result="success",
            shot_dir=runtime_log["shot_dir"],
            memo=memo,
            max_j1_current_abs=max_j1_current_abs,
        )
        print('sequence done')

        return (
            book_width,
            HandMotors_retrieval,
            container_full,
        )

    except Exception as e:
        print("Abort sequence due to exception")
        traceback.print_exc()

        # Avoid a duplicate fatal row when XArmMonitor already wrote safe_stop.
        if not runtime_log["error_logged"]:
            try:
                write_log(
                    config=config,
                    book_name=book_name,
                    shelf_id=shelf_id,
                    roll_deg=runtime_log["roll_deg"],
                    estimated_book_width_mm=runtime_log[
                        "estimated_book_width_mm"
                    ],
                    master_book_width_mm=master_book_width_mm,
                    camera_point_mm=runtime_log["camera_point_mm"],
                    robot_point_mm=runtime_log["robot_point_mm"],
                    side=side,
                    height=height,
                    result="fatal_error",
                    shot_dir=runtime_log["shot_dir"],
                    memo=f"{type(e).__name__}: {e}",
                )
            except Exception as log_exc:
                print(f"[CSV LOG ERROR] fatal logging failed: {log_exc}")

        if HandMotors_retrieval is not None:
            try:
                HandMotors_retrieval.close()
            except Exception:
                pass
        os.kill(os.getpid(), signal.SIGINT)
        return None
        
def main():
    config = load_config(CONFIG_PATH)
    print(f"[CONFIG] loaded: {config['_config_path']}")
    print(
        "[CONFIG] retrieval log: "
        f"{_resolve_config_relative_path(config, config['paths']['log']['retrieval'])}"
    )

    # ==============================
    # マスターデータ読み込み
    # ==============================
    master_file = _resolve_config_relative_path(
        config,
        config["books"]["master_file"],
    )
    print(f"[MASTER] loaded: {master_file}")
    with master_file.open("r", encoding="utf-8") as f:
        books_master = json.load(f)

    retrieved_book_width_list = [0.0]

    # ==============================
    # ROS2 初期化
    # ==============================
    rclpy.init(
        signal_handler_options=SignalHandlerOptions.NO
    )

    # ==============================
    # メイン制御ノード
    # ==============================
    node = rclpy.create_node("book_retrieval_main")

    # /retrieval_done
    done_pub = node.create_publisher(
        Bool,
        "/retrieval_done",
        10,
    )

    # ==============================
    # retrieval stage update
    # ==============================
    stage_update_pub = node.create_publisher(
        String,
        "/retrieval_stage_update",
        10,
    )

    # ==============================
    # /retrieval_system_ready
    # /retrieval_book_index
    # ==============================
    ready_qos = QoSProfile(
        depth=1,
    )

    ready_qos.reliability = (
        ReliabilityPolicy.RELIABLE
    )

    ready_qos.durability = (
        DurabilityPolicy.TRANSIENT_LOCAL
    )

    system_ready_pub = node.create_publisher(
        Bool,
        "/retrieval_system_ready",
        ready_qos,
    )

    # ==================================================
    # autolaunch側から現在処理中の本indexを受信
    #
    # 0-based:
    #   0 = 1冊目
    #   1 = 2冊目
    # ==================================================
    book_index_state = {
        "value": None
    }

    def book_index_callback(msg: Int32):
        index = int(msg.data)

        if (
            index < 0
            or index >= len(books_master)
        ):
            node.get_logger().error(
                "[BOOK INDEX] invalid index: "
                f"{index}, "
                f"book_count={len(books_master)}"
            )
            return

        book_index_state["value"] = index

        book = books_master[index]

        node.get_logger().info(
            "========================================"
        )
        node.get_logger().info(
            "[BOOK INDEX] received "
            f"/retrieval_book_index={index}"
        )
        node.get_logger().info(
            "[BOOK INDEX] "
            f"book={index + 1}/"
            f"{len(books_master)}"
        )
        node.get_logger().info(
            "[BOOK INDEX] "
            f"book_name={book.get('book_name')!r}"
        )
        node.get_logger().info(
            "[BOOK INDEX] "
            f"bookshelf_ID={book.get('bookshelf_ID')!r}"
        )
        node.get_logger().info(
            "========================================"
        )

    book_index_sub = node.create_subscription(
        Int32,
        "/retrieval_book_index",
        book_index_callback,
        ready_qos,
    )

    # ==================================================
    # autolaunch側が保持している
    # コンテナ累積offset
    # ==================================================
    container_offset_state = {
        "value": None
    }

    def container_offset_callback(
        msg: Float32,
    ):
        value = float(msg.data)

        if (
            not math.isfinite(value)
            or value < 0.0
        ):
            node.get_logger().error(
                "[CONTAINER OFFSET] "
                f"invalid value: {value}"
            )
            return

        container_offset_state["value"] = value

        node.get_logger().info(
            "[CONTAINER OFFSET] "
            f"received: {value:.1f} mm"
        )

    container_offset_sub = (
        node.create_subscription(
            Float32,
            "/retrieval_container_offset_mm",
            container_offset_callback,
            ready_qos,
        )
    )


    # ==================================================
    # managerが保持している現在stage
    # ==================================================
    retrieval_stage_state = {
        "value": None
    }

    def retrieval_stage_callback(
        msg: String,
    ):
        stage = str(
            msg.data
        ).strip()

        if not stage:
            node.get_logger().warn(
                "[RETRIEVAL STAGE] "
                "empty stage received"
            )
            return

        retrieval_stage_state["value"] = stage

        node.get_logger().info(
            "[RETRIEVAL STAGE] "
            f"received: {stage}"
        )

    retrieval_stage_sub = (
        node.create_subscription(
            String,
            "/retrieval_stage",
            retrieval_stage_callback,
            ready_qos,
        )
    )

    # 統合側 → autolaunch側への更新
    container_offset_update_pub = (
        node.create_publisher(
            Float32,
            "/retrieval_container_offset_update_mm",
            10,
        )
    )

    # ==============================
    # 上下機構 緊急停止publisher
    # ==============================
    lift_estop_pub = node.create_publisher(
        Bool,
        "/emergency_stop",
        10,
    )

    # signal handler / monitor callbackから使えるようにする
    globals()["lift_estop_pub"] = lift_estop_pub


    # ==========================================
    # 起動時：IAI非常停止解除
    #
    # 前回Ctrl+C / 異常停止で
    # STP ON + SON OFFになっている可能性があるため、
    # 起動直後に停止解除 + SON ONへ戻す。
    #
    # 注意：
    # ここでは上下機構は移動させない。
    # ==========================================
    for _ in range(5):
        rclpy.spin_once(
            node,
            timeout_sec=0.1,
        )

    release_linear_lift_estop()

    for _ in range(5):
        rclpy.spin_once(
            node,
            timeout_sec=0.1,
        )


    XARM_HOST = config["robot"]["xarm"]["host"]

    arm = XArm7(
        node=node,
        host=XARM_HOST,
    )
    globals()["arm"] = arm
    # ★ MultiThreadedExecutor 推奨
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    monitor = XArmMonitor(
    arm,
    check_period=0.5,
    auto_stop=True,
    on_emergency=monitor_emergency_stop,
    )
    # ==============================
    # リニアリフト
    # ==============================
    tp = TargetPublisher()
    executor.add_node(tp)

    # ==============================
    # WaypointPlayerNode（初期姿勢・撮影姿勢）
    # ==============================
    waypoint_node = WaypointPlayerNode(
        node_name="xarm_init_to_capture",
        arm=arm,
        monitor=monitor,
        yaml_path=config["paths"]["waypoint"]["init_to_capture"],
        speed=1.0,
        accel=1.0,
    )

    executor.add_node(waypoint_node)
    HandMotors_retrieval = (
        DynamixelWorkerClient()
    )

    print(
        "[DXL MAIN] "
        "Dynamixel worker initialized "
        "for all books"
    )

    # ==================================================
    # READY通知 helper
    # ==================================================
    def publish_system_ready():
        ready_msg = Bool()
        ready_msg.data = True

        system_ready_pub.publish(
            ready_msg
        )

        node.get_logger().info(
            "========================================"
        )
        node.get_logger().info(
            "/retrieval_system_ready = True"
        )
        node.get_logger().info(
            "Retrieval system READY"
        )
        node.get_logger().info(
            "========================================"
        )

        # DDSにpublish処理の時間を少し与える
        for _ in range(3):
            executor.spin_once(
                timeout_sec=0.1
            )


    # ==================================================
    # 起動時にmanagerが保持しているstageを確認
    # ==================================================
    node.get_logger().info(
        "Waiting for /retrieval_stage ..."
    )

    while (
        rclpy.ok()
        and retrieval_stage_state["value"] is None
    ):
        executor.spin_once(
            timeout_sec=0.1
        )

    if not rclpy.ok():
        return

    startup_stage = str(
        retrieval_stage_state["value"]
    ).strip()

    # IDLE / BOOK_STARTなら
    # 危険な物理動作途中ではないので通常起動扱い
    resume_mode = startup_stage not in {
        "IDLE",
        "BOOK_START",
    }

    node.get_logger().info(
        "========================================"
    )
    node.get_logger().info(
        f"[STARTUP STAGE] {startup_stage}"
    )
    node.get_logger().info(
        f"[RESUME MODE] {resume_mode}"
    )
    node.get_logger().info(
        "========================================"
    )

    # 通常起動ならここでREADYを送る
    if not resume_mode:
        publish_system_ready()

    # ==================================================
    # autolaunch側から
    #   ・現在の本index
    #   ・現在のコンテナoffset
    # の両方を受け取る
    # ==================================================
    node.get_logger().info(
        "Waiting for "
        "/retrieval_book_index and "
        "/retrieval_container_offset_mm ..."
    )

    while (
        rclpy.ok()
        and (
            book_index_state["value"] is None
            or container_offset_state["value"] is None
        )
    ):
        executor.spin_once(
            timeout_sec=0.1
        )

    if not rclpy.ok():
        return

    start_book_index = int(
        book_index_state["value"]
    )

    start_container_offset_mm = float(
        container_offset_state["value"]
    )

    start_retrieval_stage = str(
        retrieval_stage_state["value"]
    ).strip()

    # 再起動前のコンテナ累積幅を復元
    retrieved_book_width_list = [
        start_container_offset_mm
    ]

    node.get_logger().info(
        "========================================"
    )
    node.get_logger().info(
        "[BOOK START] "
        f"start_index={start_book_index}"
    )

    node.get_logger().info(
        "========================================"
    )
    node.get_logger().info(
        f"[RESTORE] book_index="
        f"{start_book_index}"
    )
    node.get_logger().info(
        f"[RESTORE] container_offset="
        f"{start_container_offset_mm:.1f} mm"
    )
    node.get_logger().info(
        f"[RESTORE] stage="
        f"{start_retrieval_stage}"
    )
    node.get_logger().info(
        "========================================"
    )

    node.get_logger().info(
        "[BOOK START] "
        f"book={start_book_index + 1}/"
        f"{len(books_master)}"
    )
    node.get_logger().info(
        "[CONTAINER RESUME] "
        f"offset={start_container_offset_mm:.1f} mm"
    )
    node.get_logger().info(
        "========================================"
    )
    # ==================================================
    # 再起動復帰結果
    #
    # None:
    #   通常起動
    #
    # tuple:
    #   Move_to_Containerを途中stageから復帰済み
    # ==================================================
    resume_result = None


    # ==================================================
    # 再起動時stage通知用
    # ==================================================
    def publish_resume_stage(
        stage: str,
    ):
        stage = str(stage).strip()

        if not stage:
            return

        msg = String()
        msg.data = stage

        stage_update_pub.publish(
            msg
        )

        node.get_logger().info(
            "[RESUME STAGE UPDATE] "
            f"{stage}"
        )

        # managerへ届く時間を少し与える
        for _ in range(2):
            executor.spin_once(
                timeout_sec=0.05
            )

    # ==================================================
    # main_sequence側の復帰モード
    # ==================================================
    main_resume_stage = None

    recognition_restart_mode = False

    inserting_restart_mode = False

    # ==============================================
    # BOOK_POSITIONING / HAND_OPENING
    #
    # → 撮影姿勢へ戻して再認識
    # ==============================================
    if (
        resume_mode
        and start_retrieval_stage
        in {
            "BOOK_POSITIONING",
            "HAND_OPENING",
            "INSERTING",
        }
    ):

        recognition_restart_mode = True

        inserting_restart_mode = (
            start_retrieval_stage
            == "INSERTING"
        )

        node.get_logger().warn(
            "========================================"
        )

        node.get_logger().warn(
            "[RECOGNITION RESTART]"
        )

        node.get_logger().warn(
            f"saved stage="
            f"{start_retrieval_stage}"
        )

        node.get_logger().warn(
            "撮影姿勢へ戻して"
            "再認識からやり直します"
        )

        node.get_logger().warn(
            "========================================"
        )

        # CONTAINER専用resumeには入らない
        resume_mode = False


    # ==============================================
    # GRASPING_BOOK / RETRIEVING_BOOK
    #
    # → main_sequence内から直接再開
    # ==============================================
    elif (
        resume_mode
        and start_retrieval_stage
        in {
            "GRASPING_BOOK",
            "RETRIEVING_BOOK",
        }
    ):

        main_resume_stage = (
            start_retrieval_stage
        )

        node.get_logger().warn(
            "========================================"
        )

        node.get_logger().warn(
            "[DIRECT MAIN RESUME]"
        )

        node.get_logger().warn(
            f"stage={main_resume_stage}"
        )

        node.get_logger().warn(
            f"{main_resume_stage}"
            "から直接復帰します"
        )

        node.get_logger().warn(
            "========================================"
        )

        # CONTAINER専用resumeには入らない
        resume_mode = False

    # ==================================================
    # コンテナ処理途中からの自動復帰
    # ==================================================
    if resume_mode:

        node.get_logger().warn(
            "========================================"
        )
        node.get_logger().warn(
            "[AUTO RESUME]"
        )
        node.get_logger().warn(
            f"stage={start_retrieval_stage}"
        )
        node.get_logger().warn(
            "========================================"
        )

        # 今回はまずCONTAINER系stageだけ自動復帰
        if not start_retrieval_stage.startswith(
            "CONTAINER_"
        ):
            node.get_logger().warn(
                "[AUTO RESUME] "
                "このstageはまだ自動復帰対象外です: "
                f"{start_retrieval_stage}"
            )

            return


        # ==============================================
        # 認識時に保存したbook_widthを復元
        # ==============================================
        saved_book_width = (
            load_book_width()
        )

        if saved_book_width is None:
            raise RuntimeError(
                "AUTO RESUMEに必要な"
                "book_widthが保存されていません"
            )

        node.get_logger().info(
            "[AUTO RESUME] "
            f"book_width="
            f"{saved_book_width:.2f} mm"
        )

        node.get_logger().info(
            "[AUTO RESUME] "
            f"container_offset="
            f"{start_container_offset_mm:.2f} mm"
        )

        # ==============================================
        # 復帰開始stageを決定
        # ==============================================
        resume_stage = start_retrieval_stage

        if start_retrieval_stage == "CONTAINER_Z_DOWN":
            resume_stage = "CONTAINER_POSITIONING"

            node.get_logger().warn(
                "[AUTO RESUME] "
                "CONTAINER_Z_DOWNで停止していたため、"
                "CONTAINER_POSITIONINGから再実行します"
            )
        # ==============================================
        # 保存stageからMove_to_Containerを再開
        # ==============================================
        try:
            (
                HandMotors_retrieval,
                resume_container_full,
            ) = Move_to_Container(
                start_container_offset_mm,
                saved_book_width,
                arm,
                waypoint_node,
                HandMotors_retrieval,
                stage_callback=publish_resume_stage,
                resume_stage=resume_stage,
            )

        except Exception as e:
            node.get_logger().error(
                "[AUTO RESUME] "
                "Move_to_Container failed: "
                f"{type(e).__name__}: {e}"
            )

            traceback.print_exc()

            # 既存SIGINT handlerへ渡す
            # → lift E-STOP
            # → xArm emergency_stop
            # → プロセス終了
            os.kill(
                os.getpid(),
                signal.SIGINT,
            )

            return

        if HandMotors_retrieval is None:
            raise RuntimeError(
                "AUTO RESUME後の"
                "HandMotors_retrievalがNoneです"
            )


        node.get_logger().info(
            "[AUTO RESUME] "
            "Move_to_Container completed"
        )


        # ==============================================
        # 通常main_sequenceと同じ
        # コンテナ収納後のinit復帰
        # ==============================================
        waypoint_node.reset()

        waypoint_node.play_direct(
            "/home/book/pro_book_SAM3/"
            "pro_hand_book_python/"
            "ros2_ws/src/xarm7_teaching/"
            "config/init.yaml"
        )


        # まず1.5秒だけarmを動かす
        wait_time = 1.5
        wait_start_time = time.time()

        while (
            rclpy.ok()
            and (
                time.time()
                - wait_start_time
            ) < wait_time
        ):
            executor.spin_once(
                timeout_sec=0.1
            )

            if waypoint_node.is_finished():
                break


        # ==============================================
        # リフトhome
        # ==============================================
        tp.publish_target_mm(
            config["linear_lift"][
                "home_mm"
            ]
        )

        rclpy.spin_once(
            tp,
            timeout_sec=0.1
        )


        # xArm init完了待ち
        while (
            rclpy.ok()
            and not waypoint_node.is_finished()
        ):
            executor.spin_once(
                timeout_sec=0.1
            )


        if waypoint_node.is_failed():
            raise RuntimeError(
                "AUTO RESUME後の"
                "init.yaml復帰に失敗: "
                f"{waypoint_node.error_message()}"
            )


        waypoint_node.shelf_manager.received = False


        node.get_logger().info(
            "========================================"
        )
        node.get_logger().info(
            "[AUTO RESUME COMPLETE]"
        )
        node.get_logger().info(
            f"stage={start_retrieval_stage}"
        )
        node.get_logger().info(
            "container -> init/home completed"
        )
        node.get_logger().info(
            "========================================"
        )


        # ==============================================
        # 下の通常result処理へ渡す
        # ==============================================
        resume_result = (
            saved_book_width,
            HandMotors_retrieval,
            resume_container_full,
        )
    # ==============================
    # メインループ
    # ==============================
    try:
        for book_index in range(
            start_book_index,
            len(books_master),
        ):
            b = books_master[
                book_index
            ]

            node.get_logger().info(
                "[BOOK LOOP] "
                f"index={book_index}, "
                f"book={book_index + 1}/"
                f"{len(books_master)}, "
                f"name={b.get('book_name')!r}"
            )

            waypoint_node.reset()

            book_width_offset = sum(
                retrieved_book_width_list
            )

            # ==============================================
            # 再起動復帰した最初の本だけ、
            # main_sequenceを再実行しない
            # ==============================================
            if (
                resume_result is not None
                and book_index
                == start_book_index
            ):

                node.get_logger().warn(
                    "[AUTO RESUME] "
                    "use recovered result "
                    "instead of main_sequence"
                )

                result = resume_result

                # 次の本からは通常処理
                resume_result = None

            else:

                result = main_sequence(
                    config=config,
                    book_name=b["book_name"],
                    barcode_number=b["ISBN_number"],
                    bookshelf_ID=b["bookshelf_ID"],
                    book_width_offset=book_width_offset,
                    master_book_width_mm=_optional_float(
                        b.get("book_width")
                    ),
                    tp=tp,
                    node=node,
                    arm=arm,
                    monitor=monitor,
                    executor=executor,
                    waypoint_node=waypoint_node,
                    shelf_manager=(
                        waypoint_node.shelf_manager
                    ),
                    done_pub=done_pub,
                    stage_update_pub=stage_update_pub,
                    HandMotors_retrieval=(
                        HandMotors_retrieval
                    ),
                    resume_stage=(
                        main_resume_stage
                        if book_index
                        == start_book_index
                        else None
                    ),
                    recognition_restart_mode=(
                        recognition_restart_mode
                        if book_index
                        == start_book_index
                        else False
                    ),

                    inserting_restart_mode=(
                        inserting_restart_mode
                        if book_index
                        == start_book_index
                        else False
                    ),
                )
            if result is None:
                node.get_logger().error(
                    "Fatal error detected. "
                    "Stop processing further books."
                )
                break
            (
                retrieved_book_width,
                HandMotors_retrieval,
                container_full,
            ) = result

            # ==============================================
            # 戻り値チェック
            # ==============================================
            if retrieved_book_width is None:
                node.get_logger().error(
                    "Fatal error detected. "
                    "Stop processing further books."
                )
                break

            # ==============================================
            # 今回の本の幅を累積offsetへ反映
            #
            # skipした本は 0.0 が返るので
            # offsetは増えない
            # ==============================================
            retrieved_book_width_list.append(
                retrieved_book_width
            )

            current_container_offset_mm = sum(
                retrieved_book_width_list
            )

            node.get_logger().info(
                "[CONTAINER OFFSET] "
                f"local={current_container_offset_mm:.1f} mm"
            )

            # ==============================================
            # autolaunch側へ新しいoffsetを保存
            # ==============================================
            offset_msg = Float32()
            offset_msg.data = float(
                current_container_offset_mm
            )

            container_offset_update_pub.publish(
                offset_msg
            )

            node.get_logger().info(
                "[CONTAINER OFFSET] "
                f"update sent: "
                f"{current_container_offset_mm:.1f} mm"
            )

            # manager側に受信させる
            for _ in range(5):
                executor.spin_once(
                    timeout_sec=0.1
                )

            # ==============================================
            # 満杯でなければ、offset保存後に完了通知
            # ==============================================
            if not container_full:
                done_msg = Bool()
                done_msg.data = True

                done_pub.publish(
                    done_msg
                )

                node.get_logger().info(
                    "/retrieval_done=True published "
                    "after container offset update"
                )

                for _ in range(3):
                    executor.spin_once(
                        timeout_sec=0.1
                    )


            # ==================================================
            # コンテナ満杯
            #
            # 最後の本の収納
            #   ↓
            # init姿勢復帰
            #   ↓
            # リニアリフトhome
            #   ↓
            # successログ
            #   ↓
            # /retrieval_done
            #
            # まで完了してからここへ来る。
            # ==================================================
            if container_full:

                total_width = sum(
                    retrieved_book_width_list
                )

                print("")
                print("========================================")
                print(" CONTAINER FULL")
                print(" 最後の本の収納は完了しています")
                print(
                    f" 現在の累積幅: "
                    f"{total_width:.1f} mm"
                )
                print("")
                print(" コンテナを交換してください")
                print("========================================")
                print("")

                node.get_logger().info(
                    "========================================"
                )

                node.get_logger().info(
                    "CONTAINER FULL"
                )

                node.get_logger().info(
                    f"final container width = "
                    f"{total_width:.1f} mm"
                )

                node.get_logger().info(
                    "Waiting for container replacement."
                )

                node.get_logger().info(
                    "========================================"
                )


                # ==============================================
                # 人がコンテナを交換するまで停止
                # ==============================================
                input(
                    "コンテナを交換したら "
                    "Enter を押してください: "
                )

                # ==============================================
                # 新しい空コンテナなので
                # ローカルoffsetを0 mmへ
                # ==============================================
                retrieved_book_width_list = [
                    0.0
                ]

                # ==============================================
                # autolaunch側の保持値も0 mmへ
                # ==============================================
                offset_msg = Float32()
                offset_msg.data = 0.0

                container_offset_update_pub.publish(
                    offset_msg
                )

                node.get_logger().info(
                    "[CONTAINER OFFSET] "
                    "reset update sent: 0.0 mm"
                )

                for _ in range(5):
                    executor.spin_once(
                        timeout_sec=0.1
                    )

                # ==============================================
                # offsetを0にしてから完了通知
                # ==============================================
                done_msg = Bool()
                done_msg.data = True

                done_pub.publish(
                    done_msg
                )

                node.get_logger().info(
                    "/retrieval_done=True published "
                    "after container offset reset"
                )

                for _ in range(3):
                    executor.spin_once(
                        timeout_sec=0.1
                    )


                print("")
                print("========================================")
                print(" コンテナ交換を確認しました")
                print(" container offset = 0.0 mm")
                print("▶ 次の本へ進みます")
                print("========================================")
                print("")

                node.get_logger().info(
                    "Container replaced."
                )

                node.get_logger().info(
                    "Container offset reset to 0.0 mm."
                )

                node.get_logger().info(
                    "Continue to next book."
                )

                # breakしない
                # このままforループの次の本へ進む



            

    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted by user")

    finally:
        # ==============================
        # 終了処理（順番大事）
        # ==============================
        node.get_logger().info("Shutting down nodes...")


        # ==============================
        # Dynamixel worker終了
        # ==============================
        try:
            if HandMotors_retrieval is not None:
                HandMotors_retrieval.close()
        except Exception as exc:
            print(
                "[DXL MAIN] "
                "worker cleanup warning: "
                f"{type(exc).__name__}: {exc}"
            )


        try:
            waypoint_node.destroy_node()
        except Exception:
            pass

        try:
            tp.destroy_node()
        except Exception:
            pass

        try:
            node.destroy_node()
        except Exception:
            pass

        rclpy.shutdown()


if __name__ == '__main__':
    _sam3_service_session = Sam3ServiceSession()
    with _sam3_service_session:
        main()
