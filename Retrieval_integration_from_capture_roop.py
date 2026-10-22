import os

# ============================================================
# GPUメモリ管理
# ============================================================

# PaddleOCRを常駐させない。
# 1回の認識が終わったらOCR workerを終了し、
# OCRモデルが使用していたGPUメモリを解放する。
os.environ["BOOK_OCR_RELOAD_EACH_SEQUENCE"] = "1"

# GPUメモリ最適化:
# SAM3はGPU、PaddleOCRはCPUで実行する。
# OCRとSAM3は並列のままでよい。
os.environ["BOOK_OCR_DEVICE"] = "gpu"

# GPUモードへ戻した場合のキャッシュ解放用
os.environ["BOOK_OCR_GPU_SAFE"] = "1"


from xarm7.control.xarm7 import XArm7, CAPTURE_RIGHT

from rs_d435i.get_book_position import GetBookSpinePosition

import Dynamixel_win_pro_hand_book.HandBook_Retrieval as HandBook_retrieval
import Dynamixel_win_pro_hand_book.HandBook_Storage as HandBook_storage

from pathlib import Path
from typing import Any, cast

from detection.pro_handbook.sam3_runtime.integration_service_manager import (
    Sam3ServiceSession,
)
from detection.pro_handbook.sam_py_demo.get_book_points_sam3_refined_sam2_width import (
    run_capture_and_pca_sam3_refined_sam2_width,
)
from xarm7.control.move_to_container_test import Move_to_Container
from xarm7.control.shelf_id_manager import ShelfIDManager
from detection.pro_handbook.sam_py_demo.bar_code.book_barcode import book_barcode_sequence
from detection.pro_handbook.sam_py_demo.bar_code.bookshelf_barcode import bookshelf_barcode_sequence
from xarm7.control.book_return_sequence import storage_sequence
from detection.pro_handbook.sam_py_demo.Storage import run_capture_and_pca_depth_space
from linear_lift import TargetPublisher
import rclpy
import cv2
import numpy as np
import json
from xarm7.control.robot_base_coordinate import PoseChain
from xarm7.control.robot_base_coordinate import cam_mm_to_robot_mm
import traceback
import time
from rclpy.executors import MultiThreadedExecutor
from xarm7.control.xarm_init_to_capture_integration import WaypointPlayerNode
import signal
import math
import sys
from xarm7.control.xarm_monitor import XArmMonitor, safe_motion
from datetime import datetime
from std_msgs.msg import Bool
import yaml
import ezodf
from xarm7.control.robot_base_coordinate import print_camera_debug_info
from detection.pro_handbook.sam_py_demo.bar_code.code_1_pic_ros2_editing import (
    capture_barcode_and_x_offset,
    WallDistanceWatcher,
    BoolPulseWatcher,
    BoolLatchWatcher
)
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
)
from std_msgs.msg import Bool



SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = "/home/book/pro_book_SAM3/pro_hand_book_python/Retrieval_integration.yaml"

# J1電流閾値を検知した場合の最大試行回数（初回を含む）
# 4回すべて失敗した場合は、その本を失敗扱いにして次の本へ進む。
MAX_CURRENT_INSERT_ATTEMPTS = 4
EMPTY_GRASP_POSITION_MIN = 3200
EMPTY_GRASP_POSITION_MAX = 3800
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
]



def _require_float(value: Any, name: str) -> float:
    if value is None:
        raise RuntimeError(
            f"{name} is None"
        )
    return float(value)


def _optional_float(value):
    """Return a finite float, or None when the value is blank/invalid."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def _resolve_config_relative_path(
    config: dict[str, Any],
    raw_path: Any,
) -> Path:
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

    # シートが完全に空なら1行目へヘッダを書く
    first_row_values = [
        sheet[0, col].value
        for col in range(required_columns)
    ]

    if not any(
        value is not None and str(value).strip() != ""
        for value in first_row_values
    ):
        for col, header in enumerate(LOG_HEADER):
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

def load_config(
    config_path: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()

    with config_path.open("r", encoding="utf-8") as f:
        loaded: Any = yaml.safe_load(f)

    if not isinstance(loaded, dict):
        raise RuntimeError(
            "Retrieval_integration.yaml root must be dict: "
            f"{config_path}"
        )

    config: dict[str, Any] = cast(
        dict[str, Any],
        loaded,
    )

    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)

    return config


def sigint_handler(sig, frame):
    print("Ctrl+C detected → FORCE KILL")
    try:
        arm = globals().get("arm", None)
        if arm:
            arm.emergency_stop()
    except Exception:
        pass
    service_session = globals().get("_sam3_service_session")
    if service_session is not None:
        try:
            service_session.stop_if_owned()
        except Exception as e:
            print(f"[SAM3] owned service cleanup failed: {e}")
    os._exit(1)

signal.signal(signal.SIGINT, sigint_handler)

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
    shelf_manager: ShelfIDManager,
    monitor: XArmMonitor,
    done_pub,
    capture_home_pose,
    HandMotors_retrieval,
):
    # Values below are updated as the sequence progresses. The abnormal-motion
    # callback reads this state so a safe_stop row contains the latest result.
    runtime_log = {
        "roll_deg": None,
        "estimated_book_width_mm": None,
        "camera_point_mm": None,
        "robot_point_mm": None,
        "shot_dir": None,
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
    def return_to_capture_pose(reason: str):
        target_pose = np.asarray(
            capture_home_pose,
            dtype=np.float64,
        ).reshape(6).tolist()

        node.get_logger().info(
            "[CAPTURE RETURN] "
            f"reason={reason} -> saved capture pose"
        )

        safe_motion(
            lambda: arm._moveL(
                target_pose,
                velocity=80.0,
                acceleration=40.0,
                asynchronous=False,
            ),
            monitor,
            f"return_capture_{reason}",
        )

        actual_pose = np.asarray(
            arm.get_tcp_pose(is_radian=True),
            dtype=np.float64,
        ).reshape(6)

        position_error_mm = float(
            np.linalg.norm(
                actual_pose[:3]
                - np.asarray(target_pose[:3], dtype=np.float64)
            )
        )

        node.get_logger().info(
            "[CAPTURE RETURN] "
            f"position_error={position_error_mm:.2f} mm"
        )

        if position_error_mm > 5.0:
            raise RuntimeError(
                "撮影姿勢への復帰誤差が大きいです: "
                f"{position_error_mm:.2f} mm"
            )

    try:
        print("start sequence")

        # ==================================================
        # 書架情報はmaster JSONのbookshelf_IDから直接解決する。
        # /shelf_id や /navigation_goal_final は待たない。
        # ==================================================
        (
            side,
            height,
            tcp_offset,
        ) = shelf_manager.resolve_shelf_id(
            bookshelf_ID
        )

        shelf_id = bookshelf_ID

        node.get_logger().info(
            "[CAPTURE LOOP SHELF] "
            f"id={shelf_id}, "
            f"side={side}, "
            f"height={height:.1f} mm, "
            f"tcp_offset={tcp_offset:.1f} mm"
        )

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
                print(
                    f"[CSV LOG ERROR] "
                    f"safe_stop logging failed: {log_exc}"
                )

        monitor.on_abnormal = on_abnormal

        print("Shelf side:", side)
        print("Lift height metadata:", height)
        print("xarm ready")
        print(
            "[CAPTURE LOOP] "
            "init pose / init->capture waypoint are disabled"
        )

        # 実験版ではリフトもhomeへ戻さない。
        # 現在のリフト高さを維持したまま連続処理する。
        # height はログ・棚情報としてのみ使用する。

        # 毎冊開始時に基準撮影姿勢へ揃える
        return_to_capture_pose("book_start")

        # この本についてTCP補正は最初の1回だけ
        print("TCP調整開始")

        safe_motion(
            lambda: arm.moveL_tcp_z_offset(tcp_offset),
            monitor,
            "tcp_z_offset",
        )

        time.sleep(1.0)

        # J1電流・空把持retry時の基準撮影姿勢
        capture_pose_for_retry = [
            float(v)
            for v in arm.get_tcp_pose(
                is_radian=True
            )
        ]

        # 最初の認識で得た本のY
        first_recognition_robot_y_mm = None

        # 本正面へ移動したときのY
        reaching_y_mm = None

        # 連続実験なので手動の「本に傾きをつける」待ちは行わない。

        try:
            # 右側書架では、電流閾値を検知した場合に撮影姿勢へ戻り、
            # 同じ本の認識からやり直す。
            for insert_attempt in range(1, MAX_CURRENT_INSERT_ATTEMPTS + 1):
                print(
                    f"[RETRIEVAL] recognition/insert attempt "
                    f"{insert_attempt}/{MAX_CURRENT_INSERT_ATTEMPTS}"
                )

                # 前回の認識結果を残さない
                runtime_log["roll_deg"] = None
                runtime_log["estimated_book_width_mm"] = None
                runtime_log["camera_point_mm"] = None
                runtime_log["robot_point_mm"] = None
                runtime_log["shot_dir"] = None

                # ================== 認識 ==================
                print("認識開始")
                start = time.perf_counter()

                try:
                    roll, p_xmax, book_width, shot_dir = (
                        run_capture_and_pca_sam3_refined_sam2_width(
                            query=book_name,
                            sam_device="gpu",
                        )
                    )

                    if roll is None:
                        raise RuntimeError(
                            "Recognition failed: roll is None"
                        )
                    roll = _require_float(roll, "roll")

                    shot_dir = Path(shot_dir)

                    runtime_log["roll_deg"] = float(np.degrees(roll))
                    runtime_log["estimated_book_width_mm"] = float(book_width)
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

                except Exception as e:
                    print(
                        f" recognition failed -> "
                        f"skip this book: {e}"
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
                        result="recognition_fail",
                        shot_dir=runtime_log["shot_dir"],
                        memo=f"{type(e).__name__}: {e}",
                    )
                    traceback.print_exc()

                    # 認識失敗でも初期姿勢へは戻さない。
                    # 基準撮影姿勢へ戻して次の本へ進む。
                    return_to_capture_pose("recognition_fail")
                    shelf_manager.received = False

                    done_msg = Bool()
                    done_msg.data = True
                    done_pub.publish(done_msg)
                    rclpy.spin_once(node, timeout_sec=0.1)
                    return 0.0

                end = time.perf_counter()
                print(f"{end - start} sec.")

                print("roll (deg) =", np.degrees(roll))

                if np.degrees(roll) > 90.0:
                    roll = -(roll - np.radians(90.0))
                elif np.degrees(roll) < -90.0:
                    roll = -(roll + np.radians(90.0))
                else:
                    roll = 0.0

                out = {
                    "adjusted_roll_rad": _require_float(roll, "roll"),
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
                p_robot_mm = cam_mm_to_robot_mm(
                    arm,
                    p_max,
                )
                if first_recognition_robot_y_mm is None:
                    first_recognition_robot_y_mm = float(
                        p_robot_mm[1]
                    )

                runtime_log["camera_point_mm"] = np.asarray(
                    p_max,
                    dtype=np.float64,
                ).reshape(3).copy()
                runtime_log["robot_point_mm"] = np.asarray(
                    p_robot_mm,
                    dtype=np.float64,
                ).reshape(3).copy()
                
                safe_motion(
                    lambda: arm.move_to_target_xyz_and_roll(
                        p_robot_mm=p_robot_mm,
                        d_roll_rad=_require_float(roll, "roll"),
                        side=side,
                    ),
                    monitor,
                    "insertz_before",
                )

                reaching_pose = arm.get_tcp_pose(
                    is_radian=True
                )
                reaching_y_mm = float(
                    reaching_pose[1]
                )

                HandBook_retrieval.open_until_width(
                    HandMotors_retrieval,
                    book_width,
                    gravity=False,
                )

                # input(
                #     "位置ずれ測定後 Enter: 挿入 / "
                #     "Ctrl+D: この本をスキップして撮影姿勢へ戻る / "
                #     "Ctrl+C: 終了\n"
                # )


                # ================== 挿入・取り出し ==================
                # 連続実験なので手動Enter待ちは行わない
                time.sleep(1.0)
                if side == "right":
                    # この関数内で通常と同じmoveL挿入を開始し、
                    # 挿入中だけJ1電流を監視する。
                    insert_result = arm.moveL_to_insert_right_with_current_monitor(
                        side=side,
                        retry_capture_pose=capture_pose_for_retry,
                        retry_capture_y_mm=first_recognition_robot_y_mm,
                        retry_y_offset_mm=0.0,
                        log_dir=shot_dir,             # ログ保存先（認識時に使ったshot_dirを流用）
                        roll_deg=runtime_log["roll_deg"],  # 記録済みのroll角
                    )    

                    if not insert_result["ok"]:
                        reason = insert_result.get(
                            "reason",
                            "unknown",
                        )

                        if reason == "current_threshold":
                            print(
                                "[RETRY] J1電流閾値を検知。"
                                "後退した再撮影位置から"
                                "同じ本を再認識します。"
                            )

                            # 電流検知時、アームはCAPTURE_RIGHTへ戻るが、
                            # ハンドはopen_until_width()で開いた状態のままになる。
                            # そのままcontinueすると、再認識中もハンドが開いたままに
                            # なるため、再試行前に必ずgrasp()で閉じて初期状態へ戻す。
                            print(
                                "[HAND RETRY] 後退位置を維持したまま、"
                                "再認識前にハンドを閉じます。"
                            )
                            try:
                                HandBook_retrieval.grasp(
                                    HandMotors_retrieval
                                )
                            except Exception as hand_exc:
                                raise RuntimeError(
                                    "電流検知後のハンド閉鎖に失敗しました: "
                                    f"{hand_exc}"
                                ) from hand_exc

                            time.sleep(1.0)
                            print(
                                "[HAND RETRY] ハンド閉鎖指令完了。"
                                "同じ本の再認識へ進みます。"
                            )

                            if (
                                insert_attempt
                                >= MAX_CURRENT_INSERT_ATTEMPTS
                            ):
                                print(
                                    "[SKIP] J1電流閾値を"
                                    f"{MAX_CURRENT_INSERT_ATTEMPTS}"
                                    "回連続で検知しました。"
                                    "この本を失敗扱いにして次の本へ進みます。"
                                )

                                # 監視付き挿入関数の復帰先に依存せず、
                                # 次の本へ行く前に必ず基準撮影姿勢へ揃える。
                                return_to_capture_pose(
                                    "current_threshold_fail"
                                )

                                shelf_manager.received = False

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
                                    result="current_threshold_fail",
                                    shot_dir=runtime_log["shot_dir"],
                                    memo=(
                                        "J1 current threshold detected "
                                        f"{MAX_CURRENT_INSERT_ATTEMPTS} times; "
                                        "return to capture and skip this book"
                                    ),
                                )

                                done_msg = Bool()
                                done_msg.data = True
                                done_pub.publish(done_msg)
                                rclpy.spin_once(node, timeout_sec=0.1)

                                return 0.0

                            # forループの先頭へ戻り、
                            # TCP高さ補正→同じ本の認識を再実行する。
                            continue

                        raise RuntimeError(
                            "右挿入に失敗しました: "
                            f"{insert_result}"
                        )
                    
                    grasp_result = HandBook_retrieval.grasp(
                        HandMotors_retrieval
                    )

                    grasp_position = int(
                        grasp_result["position"]
                    )

                    print(
                        "[GRASP CHECK] "
                        f"position={grasp_position}"
                    )

                    if (
                        EMPTY_GRASP_POSITION_MIN
                        <= grasp_position
                        <= EMPTY_GRASP_POSITION_MAX
                    ):
                        print(
                            "[EMPTY GRASP] "
                            "空把持なので再試行します"
                        )

                        retry_pose = [
                            float(v)
                            for v in capture_pose_for_retry
                        ]

                        retry_pose[1] = float(
                            reaching_y_mm
                        )

                        safe_motion(
                            lambda: arm._moveL(
                                retry_pose,
                                velocity=80.0,
                                acceleration=40.0,
                                asynchronous=False,
                            ),
                            monitor,
                            "empty_grasp_retry",
                        )

                        if (
                            insert_attempt
                            >= MAX_CURRENT_INSERT_ATTEMPTS
                        ):
                            print(
                                "[EMPTY GRASP] "
                                "最大回数なのでこの本をスキップ"
                            )

                            return_to_capture_pose(
                                "empty_grasp_fail"
                            )

                            shelf_manager.received = False

                            done_msg = Bool()
                            done_msg.data = True
                            done_pub.publish(done_msg)

                            rclpy.spin_once(
                                node,
                                timeout_sec=0.1,
                            )

                            return 0.0

                        continue

                    safe_motion(
                        lambda: arm.moveL_post_grasp_right(),
                        monitor,
                        "retreave_right",
                    )
                    
                    

                else:
                    safe_motion(
                        lambda: arm.moveL_to_insert_left(),
                        monitor,
                        "insert_left",
                    )
                    HandBook_retrieval.grasp(
                        HandMotors_retrieval
                    )
                    safe_motion(
                        lambda: arm.moveL_post_grasp_left(),
                        monitor,
                        "retreave_left",
                    )

                # 挿入・取り出し成功
                break

            time.sleep(1.0)
            HandBook_retrieval.open_until_full(
                HandMotors_retrieval
            )
            time.sleep(1.0)
            HandBook_retrieval.grasp(
                HandMotors_retrieval
            )

            # 本を離したら、次の本のため撮影基準姿勢へ戻す
            return_to_capture_pose("after_retrieval")


        except EOFError:
            print(
                "\n[Ctrl+D] この本をスキップ -> 撮影姿勢へ戻ります"
            )

            # 挿入前なので、そのまま基準撮影姿勢へ戻す
            return_to_capture_pose("ctrl_d")

            # 次の認識に備えてハンドを閉じる
            HandBook_retrieval.grasp(
                HandMotors_retrieval
            )

            shelf_manager.received = False
            memo = 0

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

            done_msg = Bool()
            done_msg.data = True
            done_pub.publish(done_msg)
            rclpy.spin_once(
                node,
                timeout_sec=0.1,
            )

            return 0.0


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
        
        # after_retrieval で既に撮影姿勢へ復帰済み
        shelf_manager.received = False
        memo = 0

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
        )

        print(
            "[CAPTURE LOOP] sequence done "
            "-> capture pose restored -> next book"
        )

        done_msg = Bool()
        done_msg.data = True
        done_pub.publish(done_msg)
        rclpy.spin_once(node, timeout_sec=0.1)

        return book_width

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
                HandMotors_retrieval.disable_torque(
                    HandBook_retrieval.GRIPPER_ID
                )
                HandMotors_retrieval.close_port()
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
    rclpy.init()

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
    # /retrieval_system_ready
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

    XARM_HOST = config["robot"]["xarm"]["host"]

    arm = XArm7(
        node=node,
        host=XARM_HOST,
    )
    globals()["arm"] = arm
    # ★ MultiThreadedExecutor 推奨
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    monitor = XArmMonitor(arm)
    # ==============================
    # リニアリフト
    # ==============================
    tp = TargetPublisher()
    executor.add_node(tp)

    # ==================================================
    # 撮影姿勢固定ループ用
    # WaypointPlayerNodeは作らない。
    # → /navigation_goal_final が来てもxArmを動かさない。
    # ==================================================
    shelf_manager = ShelfIDManager(node)

    # Dynamixelは全書籍で1回だけ初期化
    HandMotors_retrieval = HandBook_retrieval.init_dynamixels()
    print("[DXL] initialized for capture-loop mode")

    # Enter後に正規の出庫撮影姿勢へ移動し、
    # 到達後の実TCP姿勢を全書籍共通の撮影基準姿勢として保存する。
    print("")
    print("========================================")
    print("[CAPTURE LOOP MODE]")
    print("Enterを押すと出庫撮影姿勢へ移動します。")
    print("撮影姿勢への移動完了後、自動で認識を開始します。")
    print("========================================")

    input("Enter: 撮影姿勢へ移動して認識を開始 ")

    print("[CAPTURE START] moveJ_to_capture_right")
    ret = arm.moveJ_to_capture_right(
        asynchronous=False
    )
    print(
        "[CAPTURE START] "
        f"moveJ_to_capture_right ret={ret}"
    )

    if ret != 0:
        raise RuntimeError(
            "撮影姿勢への移動に失敗しました: "
            f"ret={ret}"
        )

    # MoveJ完了直後の実姿勢を取得する前に少し安定待ち
    time.sleep(0.5)

    capture_home_pose = np.asarray(
        arm.get_tcp_pose(is_radian=True),
        dtype=np.float64,
    ).reshape(6).tolist()

    print(
        "[CAPTURE HOME] "
        f"X={capture_home_pose[0]:.2f}, "
        f"Y={capture_home_pose[1]:.2f}, "
        f"Z={capture_home_pose[2]:.2f} mm, "
        f"Roll={np.degrees(capture_home_pose[3]):.2f}, "
        f"Pitch={np.degrees(capture_home_pose[4]):.2f}, "
        f"Yaw={np.degrees(capture_home_pose[5]):.2f} deg"
    )

    # ==================================================
    # 固定撮影姿勢モード
    # lift height は無視する。
    # xArmも上下機構も現在位置を基準として使用する。
    # sideだけは全書籍で同一か確認する。
    # ==================================================
    resolved_shelves = [
        (
            b["bookshelf_ID"],
            *shelf_manager.resolve_shelf_id(b["bookshelf_ID"]),
        )
        for b in books_master
    ]

    unique_sides = {
        item[1]
        for item in resolved_shelves
    }

    if len(unique_sides) != 1:
        raise RuntimeError(
            "CAPTURE LOOP MODEでは全書籍を同一sideにしてください。 "
            f"sides={sorted(unique_sides)}"
        )

    print(
        "[CAPTURE LOOP CHECK] "
        f"side={next(iter(unique_sides))}, "
        "lift height = IGNORE"
    )
    # ==============================
    # 出庫システム READY
    # ==============================
    ready_msg = Bool()
    ready_msg.data = True

    system_ready_pub.publish(
        ready_msg
    )

    node.get_logger().info(
        "========================================"
    )

    node.get_logger().info(
        "/retrieval_system_ready = true"
    )

    node.get_logger().info(
        "Capture-loop system READY"
    )

    node.get_logger().info(
        "========================================"
    )

    # ==============================
    # メインループ
    # ==============================
    try:
        for book_index, b in enumerate(books_master):
            book_width_offset = sum(retrieved_book_width_list)

            node.get_logger().info(
                "========================================"
            )
            node.get_logger().info(
                f"[CAPTURE LOOP] book={book_index + 1}/{len(books_master)} "
                f"name={b.get('book_name')!r}"
            )
            node.get_logger().info(
                "========================================"
            )

            retrieved_book_width = main_sequence(
                config=config,
                book_name=b["book_name"],
                barcode_number=b["ISBN_number"],
                bookshelf_ID=b["bookshelf_ID"],
                book_width_offset=book_width_offset,
                master_book_width_mm=_optional_float(b.get("book_width")),
                tp=tp,
                node=node,
                arm=arm,
                executor=executor,
                shelf_manager=shelf_manager,
                monitor=monitor,
                done_pub=done_pub,
                capture_home_pose=capture_home_pose,
                HandMotors_retrieval=HandMotors_retrieval,
            )

            if retrieved_book_width is None:
                node.get_logger().error(
                    "Fatal error detected. Stop processing further books."
                )
                break

            retrieved_book_width_list.append(retrieved_book_width)

    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted by user")

    finally:
        # ==============================
        # 終了処理（順番大事）
        # ==============================
        node.get_logger().info("Shutting down nodes...")

        try:
            HandMotors_retrieval.disable_torque(
                HandBook_retrieval.GRIPPER_ID
            )
        except Exception:
            pass

        try:
            HandMotors_retrieval.close_port()
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
