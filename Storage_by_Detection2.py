from xarm7.control.xarm7 import XArm7
# from rs_d435i.get_book_position import 収納用RealSenseClass as PosGetter # TODO コーディング？
import Dynamixel_win_pro_hand_book.HandBook_Storage as HandBook # TODO コーディング
from pathlib import Path
# from detection.pro_handbook.sam_py_demo.rs_book_capture_and_pointcloud_storage import run_capture_and_pca_depth_space
from detection.pro_handbook.sam_py_demo.Storage_rev import run_capture_and_pca_depth_space
from xarm7.control.robot_base_coordinate import cam_mm_to_robot_mm
import rclpy
import time
import cv2
import numpy as np


def main():
    try:
        # initialize modules
        print('start storage sequence')
        # ------------------------------------
        # PosGetter = 収納用RealSenseClass()
        # ------------------------------------
        Hand = HandBook.init_dynamixels() # TODO class化
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
        ret = Xarm7.moveJ_to_capture_right(asynchronous=False)
        time.sleep(1.0)
        angle_rad, first_target_cam, res = run_capture_and_pca_depth_space(
            crop_y_min=120,
            crop_y_max=550,
            min_area_px=800,
            min_height_px=100,
            max_space_width_px=420,
            target_y_from_space_top_px=240,
        )

        angle_deg = float(np.degrees(angle_rad))

        oblique_line = float(res.get("guide_edge_length_mm", 0.0))

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
        first_target_robot_mm = cam_mm_to_robot_mm(Xarm7, first_target_cam_mm)


        print("[DEBUG] first_target_robot_mm =", first_target_robot_mm)
        # main sequence
        # まずグリッパ全開 → 把持
        HandBook.open_until_full(dxl=Hand, asynchronous=False)

        time.sleep(5.0)
        #input("press enter to close gripper :")
        HandBook.grasp(dxl=Hand, keep_torque=True)
        # book_t = ．．． # TODO グリッパモータ回転位置から推定する書籍厚みからスペーサ回転量の決定するため
        try:
            input("Enter: execute reaching / Ctrl+D: retract and return : ")
        except EOFError:
            print("[INFO] Ctrl+D detected before reaching")
            print("[INFO] retract spacer and return to capture pose")

            try:
                print("[DEBUG] before contract_sp_lin_2")
                HandBook.contract_sp_lin_2(Hand, asynchronous=False)
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

        print("xArm7 moves to recognized storage target")
        print("[DEBUG] before move_to_storage_target_xyz_and_roll")

        ret = Xarm7.move_to_storage_target_xyz_and_roll(
            p_robot_mm=first_target_robot_mm,
            d_roll_rad= 0.0 ,
            side="right",
            y_offset_mm=0.0,
            z_offset_mm=0.0,
            roll_scale=1.0,
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
        HandBook.expand_sp_lin(dxl=Hand, asynchronous=True)
        print("[DEBUG] after expand_sp_lin")

        print("waiting expansion")
        time.sleep(14)
        


        if angle_deg < 90:  
            HandBook.rotate_spacer(Hand,angle_deg-180)
            #Xarm7.moveL_y_offset(y_offset=dy)
            #HandBook.reset_rot(Hand)
            #time.sleep(1.5)
            #Xarm7.moveL_y_offset(y_offset=-30)
            ret = Xarm7.moveL_to_insert_book_full(asynchronous=True)
            time.sleep(3.0)
            HandBook.reset_rot(Hand)
        else:    
            HandBook.rotate_spacer(Hand,angle_deg)
            Xarm7.moveL_y_offset(y_offset=dy)
            ret = Xarm7.moveL_to_insert_book_full(velocity= 15,acceleration= 15,asynchronous=True)       
            time.sleep(6.0)
            HandBook.reset_rot(Hand)

        time.sleep(2.0)

        print("[DEBUG] before contract_sp_lin_1")
        HandBook.contract_sp_lin_1(Hand, asynchronous=False)
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
        HandBook.contract_sp_lin_2(
            Hand,
            asynchronous=False
        )
        print("[DEBUG] after contract_sp_lin_2")

        print("[DEBUG] before ungrasp")
        HandBook.ungrasp_auto(Hand)
        print("[DEBUG] after ungrasp")

        print("[DEBUG] before post_storage")
        ret = Xarm7.moveL_to_post_storage(asynchronous=True)
        print("[DEBUG] after post_storage ret =", ret)
        time.sleep(4.0)

        print("[DEBUG] before moveJ_to_capture_right")
        ret = Xarm7.moveJ_to_capture_right(asynchronous=False)
        print("[DEBUG] after moveJ_to_capture_right ret =", ret)
        HandBook.grasp(dxl=Hand, keep_torque=True)

        print('sequence done')
        Hand.disable_torque(HandBook.GRIPPER_ID)
        Hand.disable_torque(HandBook.SP_LIN_ID)
        Hand.disable_torque(HandBook.SP_ROT_ID)
        time.sleep(0.2)

        Xarm7.disconnect()
        Hand.close_port()
        node.destroy_node()
        rclpy.shutdown()

    except KeyboardInterrupt:

        Hand.disable_torque(HandBook.GRIPPER_ID)
        Hand.disable_torque(HandBook.SP_LIN_ID)
        Hand.disable_torque(HandBook.SP_ROT_ID)
        time.sleep(0.2)
        Hand.close_port()

        try:
            Xarm7.disconnect()
            node.destroy_node()
            rclpy.shutdown()
        except:
            pass

        print('dynamixel deactivated')

if __name__ == '__main__':
    main()


        