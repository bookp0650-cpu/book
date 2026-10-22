from xarm7.control.xarm7 import XArm7
from rs_d435i.get_book_position import GetBookSpinePosition
import Dynamixel_win_pro_hand_book.HandBook_Retrieval as HandBook
from pathlib import Path
from detection.pro_handbook.sam_py_demo.rs_book_capture_and_pointcloud import run_capture_and_pca
from xarm7.control.Move_to_Container import Move_to_Container
from detection.pro_handbook.sam_py_demo.bar_code.book_barcode import book_barcode_sequence
from detection.pro_handbook.sam_py_demo.bar_code.bookshelf_barcode import bookshelf_barcode_sequence
from linear_lift import TargetPublisher
import rclpy
#from detection.pro_handbook.sam_py_demo.bar_code.code_1_pic import detect_barcode
#from detection.pro_handbook.sam_py_demo.bar_code.book_barcode import capture_and_print_barcode
import time
import cv2
import numpy as np
import json
from xarm7.control.robot_base_coordinate import PoseChain
from xarm7.control.robot_base_coordinate import cam_mm_to_robot_mm
from xarm7.control.xarm_init_to_capture import WaypointPlayer
# initial TCP_pose when 1st perception
# [-0.367414913194101, -0.11505935911584259, 0.3268820798200269, -0.00033660564673279066, -1.571207211242841, 0.0009980046969578957]


ROT_THRESH = np.deg2rad(2.5)
# XARM_HOST = "192.168.1.208"


#INIT_TCP_POSE = [-0.367414913194101, -0.11505935911584259, 0.3268820798200269,
#                 -0.00033660564673279066, -1.571207211242841, 0.0009980046969578957]
BOOK_CAPTURE = -260.0
BOOK_BARCODE_A5 = [
    362.5,
    -95.2,
    326.7,
    0.0,
    2.3721483951871,
    -3.141592653589793,
]  # deg, バーコード読み取り用

APPROACH_DX_MM = 30.0   # 手前停止 30mm
INSERT_DX_MM   = 35.0   # 追加で差し込み 35mm
LIFT_DZ_MM     = 20.0   # 退避で上げる 20mm



import numpy as np
import time

def main_sequence(
    book_name: str,
    barcode_number: str,
    bookshelf_ID: str,
    book_shelf_height: str,
    book_width_offset: float,
    tp: TargetPublisher
):
    
    # initialize modules
    print('start sequence')
    
    HandMotors = HandBook.init_dynamixels() # TODO class化
    print("aaa")
    Xarm7 = XArm7()
    Xarm7.moveJ_to_init_Q()
    bar_dir = Path("/home/book/pro_book_SAM3/pro_hand_book_python/captures/bookshelf_barcode")
    bar_dir.mkdir(parents=True, exist_ok=True)
    
    #Xarm7.moveL_z_offset(BOOK_CAPTURE) #2号館の書架用に撮影姿勢を下げる
    # bookshelf_barcode_identified = bookshelf_barcode_sequence(bookshelf_ID, bar_dir) #書架バーコード認識
    # if bookshelf_barcode_identified:
    #     print("[bookshelf barcode] bookshelf barcode identified")   
    # else:
    #     print("[bookshelf barcode] Oh no! bookshelf barcode not identified")
    
    tp.publish_target_mm(500.0)
    rclpy.spin_once(tp, timeout_sec=0.1)
    time.sleep(1.0)  # wait for linear lift to reach position
    input("上下機構動いた？")
        # capture book and analyze point cloud
    roll, p_xmax, book_width, shot_dir = run_capture_and_pca(query=book_name) #書籍認識

def main():
    with open("master_2026109.json", "r", encoding="utf-8") as f:
        books_master = json.load(f)

    book_names = [b["book_name"] for b in books_master]
    book_isbns = [b["ISBN_number"] for b in books_master] 
    bookshelf_id = [b["bookshelf_ID"] for b in books_master]
    bookshelf_height = [b["book_shelf_height"] for b in books_master]  
    retrieved_book_width_list = [0.0]
    rclpy.init()
    tp = TargetPublisher()
    for book_name, isbn, shelf_id, shelf_h in zip(book_names, book_isbns, bookshelf_id, bookshelf_height):

        book_width_offset =sum(retrieved_book_width_list)    
        retrieved_book_width = main_sequence(book_name, isbn, shelf_id, shelf_h, book_width_offset, tp)
        retrieved_book_width_list.append(retrieved_book_width)    #出庫した書籍の幅をリストに追加
    tp.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
