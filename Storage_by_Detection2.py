from xarm7.control.xarm7 import XArm7
# from rs_d435i.get_book_position import 収納用RealSenseClass as PosGetter # TODO コーディング？
from Dynamixel_win_pro_hand_book.dynamixel_worker_client import DynamixelWorkerClient
from pathlib import Path
# from detection.pro_handbook.sam_py_demo.rs_book_capture_and_pointcloud_storage import run_capture_and_pca_depth_space
from detection.pro_handbook.sam_py_demo.Storage_SAM3 import run_capture_and_pca_depth_space
from xarm7.control.robot_base_coordinate import cam_mm_to_robot_mm
import rclpy
import time
import cv2
import numpy as np

STORAGE_CAPTURE_ROOT = Path(
    "/home/book/pro_book_SAM3/pro_hand_book_python/captures/strage"
)


class FrozenTcpArm:
    """
    cam_mm_to_robot_mm() に現在TCPではなく、
    保存済みの撮影時TCPを使わせるためのラッパー。
    実機は別の位置に移動していても、撮影時の座標変換ができる。
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
            pose[3:6] = np.degrees(pose[3:6])

        return pose.tolist()

    def __getattr__(self, name):
        return getattr(self._arm, name)

def calculate_storage_d_roll_like_retrieval(
    guide_edge_p0_cam,
    guide_edge_p1_cam,
):
    """
    Storage_SAM3で得た3D guide edgeから，
    正式出庫 Retrieval_integration_SAM3.py と同じ考え方で
    xArmへ渡すroll差分角を算出する．

    Returns
    -------
    raw_roll_like_retrieval : float
        カメラXY平面上でのguide edge絶対角 [rad]

    d_roll_rad : float
        垂直姿勢を0としたxArm用roll差分角 [rad]
    """

    p0 = np.asarray(
        guide_edge_p0_cam,
        dtype=np.float64,
    ).reshape(3)

    p1 = np.asarray(
        guide_edge_p1_cam,
        dtype=np.float64,
    ).reshape(3)

    guide_axis = p1 - p0

    xy_norm = np.hypot(
        guide_axis[0],
        guide_axis[1],
    )

    if xy_norm < 1.0e-9:
        raise RuntimeError(
            "guide edge XY direction is too small "
            "to calculate storage roll"
        )

    # 正式出庫のpca_axes_fix_dir()と同じ方向規則：
    # カメラX成分が負になる方向へ軸を統一
    if guide_axis[0] > 0.0:
        guide_axis = -guide_axis

    raw_roll = float(
        np.arctan2(
            guide_axis[1],
            guide_axis[0],
        )
    )

    raw_deg = float(np.degrees(raw_roll))

    # Retrieval_integration_SAM3.pyと同じ垂直基準補正
    if raw_deg > 90.0:
        d_roll_rad = -(
            raw_roll - np.radians(90.0)
        )

    elif raw_deg < -90.0:
        d_roll_rad = -(
            raw_roll + np.radians(90.0)
        )

    else:
        d_roll_rad = 0.0

    return raw_roll, float(d_roll_rad)

def main():
    hand_worker = None
    Xarm7 = None
    node = None

    try:
        # initialize modules
        print('start storage sequence')
        # ------------------------------------
        # PosGetter = 収納用RealSenseClass()
        # ------------------------------------
        print("[DXL] starting Dynamixel worker")
        hand_worker = DynamixelWorkerClient()
        print("[DXL] worker initialized")
        rclpy.init()
        node = rclpy.create_node("storage_node")

        # 修正
        Xarm7 = XArm7(
            node=node,
            host="192.168.2.197"   # ←自分のxArmのIP
        )
        # Xarm7.moveJ_to_capture_right(asynchronous=False)
        # ------------------------------------
        # BookDetector = SamBatchInfer(SamConfig... # TODO
        # ------------------------------------
        ret = Xarm7.moveJ_to_capture_right_strage(asynchronous=False)
        hand_worker.open_until_full(asynchronous=False)
        time.sleep(5.0)
        #input("press enter to close gripper :")
        hand_worker.grasp()
        # ============================================================
        # 収納スペース認識
        # 1回目撮影後に -Y 140mm 移動しながら1回目を認識
        # 1回目で見つからなければ、移動先で2回目撮影
        # ============================================================

        SECOND_CAPTURE_DY = -140.0
        capture_tcp_pose_1 = None

        def move_to_second_capture():
            nonlocal capture_tcp_pose_1

            # ★ 1回目撮影時のTCP姿勢を保存
            capture_tcp_pose_1 = Xarm7.get_tcp_pose(
                is_radian=True
            )

            print(
                "[STORAGE] 1st capture TCP pose =",
                capture_tcp_pose_1
            )

            # 保存してから移動開始
            Xarm7.moveL_y_offset(
                SECOND_CAPTURE_DY,
                velocity=40,
                acceleration=40,
                asynchronous=True,
            )


        def wait_xarm_motion_done(timeout=10.0):
            """xArmが動き始めてから停止するまで待つ"""

            start_time = time.monotonic()
            moving_detected = False

            while True:
                result = Xarm7.arm.get_is_moving()

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

                # 一度動いてから停止した
                if moving_detected and not moving:
                    return

                elapsed = time.monotonic() - start_time

                # moving=Trueを取り逃した場合の保険
                if elapsed >= 1.0 and not moving_detected and not moving:
                    return

                if elapsed >= timeout:
                    raise RuntimeError(
                        f"xArm motion timeout: {elapsed:.1f} sec"
                    )

                time.sleep(0.02)


        # ============================================================
        # 1回目撮影 -> 撮影直後にTCP保存 -> 2回目位置へ移動開始
        #
        # その移動中に1回目認識を継続する。
        #
        # 1回目認識 SUCCESS:
        #   2回目位置まで移動完了後、
        #   1回目撮影時TCPを使って座標変換し、
        #   2回目位置からそのままリーチング。
        #
        # 1回目認識 FAIL:
        #   2回目位置まで移動完了後、
        #   2回目撮影・認識を実行して、
        #   その結果へリーチング。
        # ============================================================

        first_result = None

        try:
            print("\n========== STORAGE 1st CAPTURE ==========")

            # run_capture_and_pca_depth_space() 内では、
            # 撮影完了後に after_capture_callback が呼ばれる。
            # callbackで
            #   1. 1回目撮影時TCP保存
            #   2. -Y140 mm移動開始
            # を行い、その後の認識処理とxArm移動を並行させる。
            first_result = run_capture_and_pca_depth_space(
                out_dir=STORAGE_CAPTURE_ROOT,
                after_capture_callback=move_to_second_capture,
            )

            print("[STORAGE] 1st recognition SUCCESS")

        except RuntimeError as e:

            # 「収納スペースなし」以外は上位へ投げる
            if "収納スペース候補が見つかりませんでした" not in str(e):
                raise

            print("\n[STORAGE] 1st recognition: NO SPACE")

        # callbackで開始した2回目撮影位置への移動完了を待つ
        wait_xarm_motion_done()

        print("[STORAGE] arrived at 2nd capture position")

        # ============================================================
        # 1回目認識成功
        # ============================================================
        if first_result is not None:

            print("\n========== USE STORAGE 1st RESULT ==========")

            if capture_tcp_pose_1 is None:
                raise RuntimeError(
                    "1st recognition succeeded but "
                    "capture_tcp_pose_1 was not saved"
                )

            angle_rad, first_target_cam, res = first_result

            # 重要:
            # 実機xArmは2回目位置にいるまま。
            #
            # ただし first_target_cam は1回目撮影画像のカメラ座標なので、
            # 座標変換だけは1回目撮影時TCPを使う。
            coordinate_transform_arm = FrozenTcpArm(
                Xarm7,
                capture_tcp_pose_1,
            )

            recognition_source = "1st"

            print(
                "[STORAGE] use 1st recognition result "
                "while staying at 2nd capture position"
            )

        # ============================================================
        # 1回目認識失敗 -> 2回目撮影・認識
        # ============================================================
        else:

            print("\n========== STORAGE 2nd CAPTURE ==========")

            angle_rad, first_target_cam, res = (
                run_capture_and_pca_depth_space(
                    out_dir=STORAGE_CAPTURE_ROOT,
                    after_capture_callback=None,
                )
            )

            # 2回目結果は現在の2回目撮影位置から得たものなので、
            # 現在TCPをそのまま座標変換に使う。
            coordinate_transform_arm = Xarm7

            recognition_source = "2nd"

            print("[STORAGE] 2nd recognition SUCCESS")

        angle_deg = float(np.degrees(angle_rad))

        oblique_line = float(
            res.get("guide_edge_length_mm", 0.0)
        )

        guide_edge_p0_cam = res.get(
            "guide_edge_p0_cam"
        )
        guide_edge_p1_cam = res.get(
            "guide_edge_p1_cam"
        )

        if guide_edge_p0_cam is None:
            raise RuntimeError(
                "guide_edge_p0_cam is missing"
            )

        if guide_edge_p1_cam is None:
            raise RuntimeError(
                "guide_edge_p1_cam is missing"
            )

        (
            storage_raw_roll,
            storage_d_roll_rad,
        ) = calculate_storage_d_roll_like_retrieval(
            guide_edge_p0_cam,
            guide_edge_p1_cam,
        )

        print(
            "[STORAGE ROLL] "
            f"image angle = {angle_deg:.2f} deg"
        )

        print(
            "[STORAGE ROLL] "
            f"3D raw-like roll = "
            f"{np.degrees(storage_raw_roll):.2f} deg"
        )

        print(
            "[STORAGE ROLL] "
            f"xArm d_roll = "
            f"{np.degrees(storage_d_roll_rad):.2f} deg"
        )

        # cosにはdegreeではなくradを入れる
        dy = -oblique_line * np.cos(angle_rad)/2
        print(f"斜辺の長さ：L={oblique_line}mm")
        print(f"マニピュレー水平移動動：dy={dy}mm")

        print("[DEBUG] after rotate_spacer")
        # ===== 返り値ログ =====
    
        print("========== run_capture_and_pca result ==========")
        print(f"[RESULT] angle_rad = {angle_rad:.6f} rad")
        print(f"[RESULT] angle_deg = {angle_deg:.2f} deg")

        print(
            "[RESULT] first_target_cam [m] = "
            f"X={first_target_cam[0]:.4f}, "
            f"Y={first_target_cam[1]:.4f}, "
            f"Z={first_target_cam[2]:.4f}"
        )

        print(
            "[RESULT] first_target_cam [mm] = "
            f"X={first_target_cam[0] * 1000:.1f}, "
            f"Y={first_target_cam[1] * 1000:.1f}, "
            f"Z={first_target_cam[2] * 1000:.1f}"
        )

        print(f"[RESULT] pair_indices = {res.get('pair_indices', None)}")
        print(f"[RESULT] line_p0 = {res.get('line_p0', None)}")
        print(f"[RESULT] line_p1 = {res.get('line_p1', None)}")
        print(f"[RESULT] sub_line_p0 = {res.get('sub_line_p0', None)}")
        print(f"[RESULT] sub_line_p1 = {res.get('sub_line_p1', None)}")
        print("================================================")

        # ===== カメラ座標[m] -> カメラ座標[mm] =====
        first_target_cam_mm = 1000.0 * first_target_cam

        print("[DEBUG] first_target_cam_mm =", first_target_cam_mm)
        # ===== カメラ座標[mm] -> ロボットベース座標[mm] =====
        #
        # 1回目結果:
        #   FrozenTcpArm -> 1回目撮影時TCPで変換
        #
        # 2回目結果:
        #   Xarm7 -> 現在の2回目撮影時TCPで変換
        first_target_robot_mm = cam_mm_to_robot_mm(
            coordinate_transform_arm,
            first_target_cam_mm,
        )

        print(
            f"[DEBUG] recognition_source = {recognition_source}"
        )
        print(
            "[DEBUG] first_target_robot_mm =",
            first_target_robot_mm,
        )

        time.sleep(5.0)
        # book_t = ．．． # TODO グリッパモータ回転位置から推定する書籍厚みからスペーサ回転量の決定するため
        try:
            input("Enter: execute reaching / Ctrl+D: retract and return : ")
        except EOFError:
            print("[INFO] Ctrl+D detected before reaching")
            print("[INFO] retract spacer and return to capture pose")

            try:
                print("[DEBUG] before contract_sp_lin_2")
                hand_worker.contract_sp_lin_2(asynchronous=False)
                print("[DEBUG] after contract_sp_lin_2")
            except Exception as e:
                print("[WARN] contract_sp_lin_2 failed:", e)

            try:
                print("[DEBUG] before moveJ_to_capture_right")
                ret = Xarm7.moveJ_to_capture_right(asynchronous=False)
                print("[DEBUG] after moveJ_to_capture_right ret =", ret)
            except Exception as e:
                print("[WARN] moveJ_to_capture_right failed:", e)

            return

        print(
            f"xArm7 reaches {recognition_source} recognition target "
            "from 2nd capture position"
        )
        print("[DEBUG] before move_to_storage_target_xyz_and_roll")

        ret = Xarm7.move_to_storage_target_xyz_and_roll(
            p_robot_mm=first_target_robot_mm,
            d_roll_rad=storage_d_roll_rad,
            side="right",
        )

        print("[DEBUG] after move_to_storage_target_xyz_and_roll ret =", ret)

        try:
            input("After reaching: Enter: expand spacer / Ctrl+D: retract and return : ")
        except EOFError:
            print("[INFO] Ctrl+D detected after reaching")
            print("[INFO] return to capture pose")

            try:
                print("[DEBUG] before moveJ_to_capture_right")
                ret = Xarm7.moveJ_to_capture_right(asynchronous=False)
                print("[DEBUG] after moveJ_to_capture_right ret =", ret)
            except Exception as e:
                print("[WARN] moveJ_to_capture_right failed:", e)

            return

        print("[DEBUG] before expand_sp_lin")
        hand_worker.expand_sp_lin(asynchronous=True)
        print("[DEBUG] after expand_sp_lin")

        print("waiting expansion")
        time.sleep(14)
        


        if angle_deg < 90:  
            hand_worker.rotate_spacer(angle_deg - 180)
            #Xarm7.moveL_y_offset(y_offset=dy)
            # hand_worker.reset_rot()
            #time.sleep(1.5)
            #Xarm7.moveL_y_offset(y_offset=-30)
            ret = Xarm7.moveL_to_insert_book_full(asynchronous=True)
            time.sleep(3.0)
            hand_worker.reset_rot()
        else:    
            hand_worker.rotate_spacer(angle_deg)
            Xarm7.moveL_y_offset(y_offset=dy)
            ret = Xarm7.moveL_to_insert_book_full(velocity= 15,acceleration= 15,asynchronous=True)       
            time.sleep(6.0)
            hand_worker.reset_rot()

        time.sleep(2.0)

        print("[DEBUG] before contract_sp_lin_1")
        hand_worker.contract_sp_lin_1(asynchronous=False)
        print("[DEBUG] after contract_sp_lin_1")

        print("[DEBUG] before move_L_to_insert_book_tip")
        ret = Xarm7.move_L_to_insert_book_tip(
            velocity=15,
            acceleration=15,
            asynchronous=False
        )
        print("[DEBUG] after move_L_to_insert_book_tip ret =", ret)

        if ret != 0:
            raise RuntimeError(
                f"move_L_to_insert_book_tip failed: ret={ret}"
            )

        time.sleep(0.5)

        print("[DEBUG] before contract_sp_lin_2")
        hand_worker.contract_sp_lin_2(
            asynchronous=False
        )
        print("[DEBUG] after contract_sp_lin_2")

        print("[DEBUG] before ungrasp")
        hand_worker.ungrasp_auto()
        print("[DEBUG] after ungrasp")

        print("[DEBUG] before post_storage")
        ret = Xarm7.moveL_to_post_storage(asynchronous=True)
        print("[DEBUG] after post_storage ret =", ret)
        time.sleep(4.0)

        print("[DEBUG] before moveJ_to_capture_right")
        ret = Xarm7.moveJ_to_capture_right(asynchronous=False)
        print("[DEBUG] after moveJ_to_capture_right ret =", ret)
        hand_worker.grasp()

        print('sequence done')

        # DynamixelのトルクOFF / port解放はworker側shutdownで行う
        if hand_worker is not None:
            hand_worker.close()

        time.sleep(0.2)

        if Xarm7 is not None:
            Xarm7.disconnect()

        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

    except KeyboardInterrupt:

        print("\nCtrl+C detected")

        try:
            if hand_worker is not None:
                hand_worker.close()
        except Exception as e:
            print("[WARN] Dynamixel worker close failed:", e)

        time.sleep(0.2)

        try:
            if Xarm7 is not None:
                Xarm7.disconnect()

            if node is not None:
                node.destroy_node()

            if rclpy.ok():
                rclpy.shutdown()

        except Exception as e:
            print("[WARN] cleanup failed:", e)

        print('dynamixel worker deactivated')

if __name__ == '__main__':
    main()


        