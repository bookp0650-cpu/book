from pathlib import Path
from typing import Optional, Tuple, Any, Union

import numpy as np

from pro_book.pro_hand_book_python.xarm7.control.robot_base_coordinate import cam_mm_to_robot_mm
from FixToCameraCoordinate import barcode_to_camera_xyz
from pro_book.pro_hand_book_python.xarm7.control.xarm7 import XArm7

def barcode_to_robot_base_mm(
    arm: XArm7,
    barcode_number: str,
    image_path: Union[str, Path],
    *,
    depth_m: float = 0.3,
    handeye_json_path: Union[str, Path] = "/home/book/pro_book_SAM3/pro_hand_book_python/xarm7/handeye_pairs/handeye_T_tcp_cam_20260107_124544.json",
    dy_adj_mm: float = 0.0,
    fx: float = 1362.24267578125,
    fy: float = 1359.7244873046875,
    cx: float = 956.697509765625,
    cy: float = 556.5321044921875,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[float, float]], Optional[Any]]:
    """
    バーコード -> rect -> カメラ座標(m) -> カメラ座標(mm) -> ロボットベース座標(mm)

    Returns:
      matched: bool
      p_robot_mm: np.ndarray shape (3,) or None
      p_cam_mm:   np.ndarray shape (3,) or None
      uv: (u,v) pixel or None
      rect: pyzbar rect or None
    """
    image_path = Path(image_path)

    matched, xyz_m, uv, rect = barcode_to_camera_xyz(
        barcode_number=barcode_number,
        image_path=image_path,
        depth_m=depth_m,
        fx=fx, fy=fy, cx=cx, cy=cy,
    )
    if not matched or xyz_m is None:
        return False, None, None, uv, rect

    # (X,Y,Z)[m] -> [mm]
    p_cam_mm = np.asarray(xyz_m, dtype=np.float64).reshape(3) * 1000.0

    # camera(mm) -> robot/base(mm)
    p_robot_mm = cam_mm_to_robot_mm(
        arm=arm,
        p_cam_mm=p_cam_mm,
        handeye_json_path=handeye_json_path,
        dy_adj_mm=dy_adj_mm,
    )

    return True, p_robot_mm, p_cam_mm, uv, rect


if __name__ == "__main__":
    # 例：arm の生成・接続はあなたの環境の手順に合わせてください
    arm = XArm7(ip="192.168.1.***")  # 例
    arm.connect()

    ok, p_robot_mm, p_cam_mm, uv, rect = barcode_to_robot_base_mm(
        arm=arm,
        barcode_number="15 14",
        image_path="/home/book/Desktop/before_init_rgb.png",
        depth_m=0.3,
    )

    print("matched:", ok)
    print("rect:", rect)
    print("uv:", uv)
    print("p_cam_mm:", p_cam_mm)
    print("p_robot_mm:", p_robot_mm)



