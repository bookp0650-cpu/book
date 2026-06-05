from xarm7.control.xarm7 import XArm7
from rs_d435i.get_book_position import GetBookSpinePosition
import Dynamixel_win_pro_hand_book.HandBook_Retrieval as HandBook_retrieval 
import Dynamixel_win_pro_hand_book.HandBook_Storage as HandBook_storage
from pathlib import Path
from detection.pro_handbook.sam_py_demo.get_book_points import run_capture_and_pca
from xarm7.control.move_to_container_test import Move_to_Container
from xarm7.control.shelf_id_manager import ShelfIDManager
from detection.pro_handbook.sam_py_demo.bar_code.book_barcode_test_rightside import book_barcode_sequence
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
import signal, os
import sys
from xarm7.control.xarm_monitor import XArmMonitor, safe_motion
import csv
from datetime import datetime
from std_msgs.msg import Bool
import yaml
from xarm7.control.robot_base_coordinate import print_camera_debug_info
from detection.pro_handbook.sam_py_demo.bar_code.code_1_pic_ros2_editing import (
    capture_barcode_and_x_offset,
    WallDistanceWatcher,
    BoolPulseWatcher,
    BoolLatchWatcher
)

def write_log(config, book_name, shelf_id, roll_deg, book_width, side, height, result, shot_dir, memo):

    log_file = config["paths"]["log"]["retrieval"]

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        book_name,
        shelf_id,
        roll_deg,
        book_width,
        side,
        height,
        result,
        str(shot_dir),
        memo
    ]

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def sigint_handler(sig, frame):
    print("Ctrl+C detected → FORCE KILL")
    try:
        arm = globals().get("arm", None)
        if arm:
            arm.emergency_stop()
    except Exception:
        pass
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
    tp: TargetPublisher,
    node,
    arm: XArm7,                     
    executor,
    waypoint_node: WaypointPlayerNode,
    shelf_manager: ShelfIDManager,
    monitor: XArmMonitor,
    done_pub,
):
    try:
        print('start sequence')
        shelf_manager.received = False
        waypoint_node.reset()
        # ==============================
        # shelf_id 受信待ち
        # ==============================
        node.get_logger().info("Waiting for /shelf_id ...")

        while rclpy.ok() and not shelf_manager.is_received():
            executor.spin_once(timeout_sec=0.1)

        side = shelf_manager.get_side()
        height = shelf_manager.get_height()
        id = shelf_manager.get_shelf_id()
        tcp_offset = shelf_manager.get_tcp_z_offset()
        monitor.on_abnormal = lambda msg: write_log(
            config,
            book_name,
            id,
            None,
            None,
            side,
            height,
            "safe_stop",
            None,
            msg
        )

        print("Shelf side:", side)
        print("Lift height:", height)
        # ==============================
        HandMotors_retrieval = HandBook_retrieval.init_dynamixels() 
        print("xarm ready")
        safe_motion(lambda: arm.moveJ_to_init_Q_DEG(), monitor, "init_pose")
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
        bar_dir = Path("/home/book/pro_book/pro_hand_book_python/captures/bookshelf_barcode")
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
        # init → capture 姿勢へ（Waypoint）
        node.get_logger().info("Waiting for manual /navigation_goal_final")

        while rclpy.ok() and not waypoint_node.is_finished():
            executor.spin_once(timeout_sec=0.1)
            
        if waypoint_node.is_failed():
            raise RuntimeError(
                f"Waypoint failed: {waypoint_node.error_message()}"
            )

        node.get_logger().info("Waypoint succeeded → start recognition")

        # ===== TCP 高さ方向 微調整ここ =====

        print("TCP調整開始")
        safe_motion(lambda: arm.moveL_tcp_z_offset(tcp_offset),
                    monitor,
                    "tcp_z_offset")
        time.sleep(1.0)
        # ==============認識================
        
        print("認識開始")
        start = time.perf_counter()
        try:
            roll, target, book_width, shot_dir = run_capture_and_pca(query=book_name)
            print(f"""
            ===== PCA RESULT =====
            roll        : {roll}
            target      : {target}
            book_width  : {book_width}
            ======================
            """)

            if target is None:
                raise RuntimeError("Recognition failed: target is None")
 
        except Exception as e:
            print(f" recognition failed -> skip this book: {e}")
            write_log(
                config,
                book_name,
                id,
                None,
                None,
                side,
                height,
                "recognition_fail",
                shot_dir,
                ""
            )
            traceback.print_exc()
            

            # ===== 認識エラーになった時、アームを初期姿勢に戻す =====s
            tp.publish_target_mm(config["linear_lift"]["home_mm"])
            waypoint_node.reset()

            waypoint_path = config["paths"]["waypoint"]["capture_to_init"][side]

            waypoint_node.play_direct(waypoint_path)

            while rclpy.ok() and not waypoint_node.is_finished():
                executor.spin_once(timeout_sec=0.1)

            done_msg = Bool()
            done_msg.data = True
            done_pub.publish(done_msg)
            rclpy.spin_once(node, timeout_sec=0.1)
            return 0.0

        end = time.perf_counter()
        print(f"{end - start} sec.")
        
        print("roll (deg) =", np.degrees(roll))
        if np.degrees(roll) > 90.0: #roll方向の調整
            roll = - (roll - np.radians(90.0))
        elif np.degrees(roll) < -90.0:
            roll = - (roll + np.radians(90.0))
        else:
            roll = 0.0
        out = {
            "adjusted_roll_rad": float(roll),
            "adjusted_roll_deg": float(np.degrees(roll)),
        }
        (shot_dir / "adjusted_roll.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )    
        print("adjusted roll (deg) =", np.degrees(roll))
        target_mm = 1000 * target       #mからmmへ
        
        p_robot_mm = print_camera_debug_info(
            arm,
            target_mm
        )
        # ロボット座標系での対象点 [mm]
        p_robot_mm = cam_mm_to_robot_mm(arm, target_mm)

        
        safe_motion(
            lambda: arm.move_to_target_xyz_and_roll(
                p_robot_mm=p_robot_mm,
                d_roll_rad=roll,
                side=side   # ← ここ
            ),
            monitor,
            "insertz_before"
        )   #書籍背表紙位置まで挿入                                    #書籍背表紙位置まで移動  

        HandBook_retrieval.open_until_width(HandMotors_retrieval, book_width, gravity=False)

        try:
            #============================認識後の取り出し動作========================================================

            input('insert: Enter / return to capture: Ctrl+D / exit: Ctrl+C')

            if side == "right":
                safe_motion(lambda: arm.moveL_to_insert_right(), monitor, "insert_right")    #書籍背表紙位置まで挿入
                HandBook_retrieval.grasp(HandMotors_retrieval)              #ハンドを閉じる
                safe_motion(lambda: arm.moveL_post_grasp_right() , monitor, "retreave_right")   #書籍を引き抜く
                #arm.move_tcp_execute(dx=0.2675, executor=executor)

                               
            else:
                safe_motion(lambda: arm.moveL_to_insert_left(), monitor, "insert_left")         #書籍背表紙位置まで挿入
                HandBook_retrieval.grasp(HandMotors_retrieval)              #ハンドを閉じる
                safe_motion(lambda: arm.moveL_post_grasp_left() , monitor, "retreave_left")   #書籍を引き抜く
                #arm.move_tcp_execute(dx=0.2675, executor=executor)

            tp.publish_target_mm(config["linear_lift"]["move_to_container"])

            #==========================書籍バーコード認識============================================================

            if height < 900:
                time.sleep((900-height)*0.0075)

            result = book_barcode_sequence(barcode_number, shot_dir, arm)
            
            if result == "success":
                pass  # そのままコンテナ収納処理へ続く
            
            elif result == "no_barcode":
                #ここに右側動作の行動を呼び出すコードを書く
                pass
        
            elif result == "wrong_barcode":
                print("barcode NG")
            
                storage_sequence(arm, HandBook_storage)

                print(f" バーコード不一致 ")
                write_log(
                    config,
                    book_name,
                    id,
                    None,   
                    None,
                    side,
                    height,
                    "バーコード不一致",
                    shot_dir,
                    ""
                )

                # ===== 認識エラーになった時、アームを初期姿勢に戻す =====s
                tp.publish_target_mm(config["linear_lift"]["home_mm"])
                waypoint_node.reset()
                
                waypoint_path = config["paths"]["waypoint"]["capture_to_init"][side]

                waypoint_node.play_direct(waypoint_path)

                while rclpy.ok() and not waypoint_node.is_finished():
                    executor.spin_once(timeout_sec=0.1)
                done_msg = Bool()
                done_msg.data = True
                done_pub.publish(done_msg)
                rclpy.spin_once(node, timeout_sec=0.1)

                return 0.0

    


        except EOFError:
            #ctrl + D によってその書籍出庫はスキップし初期姿勢に戻る
            print("Ctrl+D detected → return to capture")
    
            if side == "right":
                safe_motion(lambda: arm.moveL_post_grasp_right() , monitor, "retreave_right")   #書籍を引き抜く   
            else:
                safe_motion(lambda: arm.moveL_post_grasp_left() , monitor, "retreave_left")   #書籍を引き抜く

            HandBook_retrieval.grasp(HandMotors_retrieval)
            tp.publish_target_mm(config["linear_lift"]["home_mm"])
            #初期姿勢へ戻る
            waypoint_node.reset()

            waypoint_node.play_direct(
                config["paths"]["waypoint"]["capture_to_init"][side]
            )

            while rclpy.ok() and not waypoint_node.is_finished():
                executor.spin_once(timeout_sec=0.1)
            memo = 0#input("メモあれば入力(Enterで次の本へ): ")
            write_log(
                config,
                book_name,
                id,
                float(np.degrees(roll)),
                book_width,
                side,
                height,
                "ctrl+d",
                shot_dir,
                memo
            )
            rclpy.spin_once(tp, timeout_sec=0.1)

            done_msg = Bool()
            done_msg.data = True
            done_pub.publish(done_msg)
            rclpy.spin_once(node, timeout_sec=0.1)

            return 0.0   # ← 次の本へ


        except Exception:
            print("xArm7 error")
            os.kill(os.getpid(), signal.SIGINT)
        
        tp.publish_target_mm(config["linear_lift"]["move_to_container"])
        rclpy.spin_once(tp, timeout_sec=0.1)

       #-----------------------コンテナ収納動作-----------------------------------------
        try:
            safe_motion(lambda: Move_to_Container(book_width_offset, arm, waypoint_node, HandMotors_retrieval), monitor, "Move_to_container")  
        except Exception:
            print("xArm error during Move_to_Container")
            os.kill(os.getpid(), signal.SIGINT)
            
        print('skip insertion or after insertion')
        
        waypoint_node.reset()
        waypoint_node.play_direct(
            "/home/book/pro_book/pro_hand_book_python/ros2_ws/src/xarm7_teaching/config/init.yaml"
        )

        wait_start_time = time.time()
        while rclpy.ok() and (time.time() - wait_start_time) < 1.5:  # 待つ（この間にアームの移動が終わることが多い）
            executor.spin_once(timeout_sec=0.1)  # ROSの通信を維持
            #もしアームの移動が終わってしまったら待機を終了
            if waypoint_node.is_finished():
                break

        tp.publish_target_mm(config["linear_lift"]["home_mm"])
        rclpy.spin_once(tp, timeout_sec=0.1)
        while rclpy.ok() and not waypoint_node.is_finished():
            executor.spin_once(timeout_sec=0.1)

        shelf_manager.received = False
        memo = 0#input("メモあれば入力(Enterで次の本へ): ")
        write_log(
            config,
            book_name,
            id,
            float(np.degrees(roll)),
            book_width,
            side,
            height,
            "success",
            shot_dir,
            memo
        )
        print('sequence done')

        done_msg = Bool()
        done_msg.data = True
        done_pub.publish(done_msg)
        rclpy.spin_once(node, timeout_sec=0.1)

        return book_width   

    except Exception as e:
        print("Abort sequence due to exception")
        traceback.print_exc()
        try:
            HandMotors_retrieval.disable_torque(HandBook_retrieval.GRIPPER_ID)
            HandMotors_retrieval.close_port()
        except Exception:
            pass
        os.kill(os.getpid(), signal.SIGINT)
        return None 

        
def main():
    config = load_config("Retrieval_integration.yaml")
    # ==============================
    # マスターデータ読み込み
    # ==============================
    with open(config["books"]["master_file"], "r", encoding="utf-8") as f:
        books_master = json.load(f)

    retrieved_book_width_list = [0.0]

    # ==============================
    # ROS2 初期化
    # ==============================
    rclpy.init()

    # メイン制御ノード（publish / log / trigger 用）
    node = rclpy.create_node("book_retrieval_main")
    done_pub = node.create_publisher(Bool, "/retrieval_done", 10)
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

    # ==============================
    # メインループ
    # ==============================
    try:
        for b in books_master:
            waypoint_node.reset()
            book_width_offset = sum(retrieved_book_width_list)

            retrieved_book_width = main_sequence(
                config=config,
                book_name=b["book_name"],
                barcode_number=b["ISBN_number"],
                bookshelf_ID=b["bookshelf_ID"],
                book_width_offset=book_width_offset,
                tp=tp,
                node=node,
                arm=arm, 
                monitor=monitor,
                executor=executor,
                waypoint_node=waypoint_node,   
                shelf_manager=waypoint_node.shelf_manager,
                done_pub=done_pub,
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

    main()
