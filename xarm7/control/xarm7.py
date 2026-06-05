from xarm.wrapper import XArmAPI
import numpy as np
import math
from pathlib import Path
import time
from xarm7.control.util import convert
from rclpy.action import ActionClient
import rclpy
from xarm7_moveit_msgs.action import MoveTCP

class MoveTCPClient:
    def __init__(self, node):
        self._node = node
        self._client = ActionClient(node, MoveTCP, '/move_tcp')

    def wait(self):
        self._client.wait_for_server()

    def move_tcp(self, dx: float):
        goal = MoveTCP.Goal()
        goal.dx = float(dx)
        return self._client.send_goal_async(goal)
    

PKG_DIR = Path(__file__).resolve().parent

#XARM_HOST = "192.168.1.208" # AC controller
XARM_HOST = "192.168.2.197" # DC controller

# initial TCP_pose when 1st perception (pos[mm], rpy[deg])
INIT_TCP_POSE_A5 = [
    104.2,
    -156.2,
    558.8,
    -90.0,
    0.0,
    180.0,
] # TCP offset 修正前，使用不可

BARCODE_Q_A5 = [
    -76.5,
    -104.0,
    86.9,
    19.5,
    22.4,
    36.4,
    155.0
]  # deg, バーコード読み取り用
# INIT_Q_A5 = [
#     -22.9,
#     84.3,
#     -163.1,
#     56.3,
#     -154.6,
#     42.5,
#     -177.5
# ]#なんだっけ？？2号館上から3段目,上下機構400mm  
# INIT_Q_A5 = [
#     -217.5,
#     -20.0,
#     -240.6,
#     52.6,
#     -155.2,
#     -45.9,
#     -114.2
# ] # 2号館上から3段目,上下機構400mm  逆側（進行方向左側）書籍
# INIT_Q_A5 = [
#     -27.1,
#     96.0,
#     -189.7,
#     10.6,
#     -190.0,
#     73.7,
#     -150.4
# ]   #deg, 閉架書架 1H17


#INIT_Q_A5 = [
#    -81.1,
#    -80.7,
#    5.7,
#    9.1,
#    -81.7,
#    9.9,
#    177.3
#]   #deg, 閉架書架 1H13

#INIT_Q_A5 = [
#   -72.5,
#    -76.4,
#    2.6,
#    17.0,
#    -76.2,
#    18.7,
#    168.1
#]  # deg, 2号館

#INIT_Q_A5 = [
#    -68.6,
#    -71.7,
#    1.8,
#    26.1,
#    -69.8,
#    23.5,
#    159.8,
#]  # deg, 10度傾いているカメラ

#INIT_Q_A5 = [
#     -86.3,
#     -101.6,
#     65.6,
#     25.4,
#     16.1,
#     35.0,
#     141.1
#]  # deg, 開架B18（4,3)

CAPTURE_RIGHT = [
   86,
   -55.5,
   173.8,
   65.6,
   38.9,
   9.4,
   43.9
] 



CAPTURE_LEFT = [
   108.1,
   -53.3,
   178.9,
   58.8,
   11.5,
   2.6,
   275.8
] 


INIT_Q_DEG = [

    0.0,
  -4.3,
  95.7,
  164.6,
  263.1,
  96.7,
  210.0

]



PLACING_FOR_TEST_Q = [
    -81.1,
    56.2,
    13.9,
    72.3,
    -150.4,
    44.6,
    189.4
] # deg
# INIT_Q_A5_OFFSET = [
#     ,,,,,,]  # deg

DISTANCE_BOOKSHELF = np.radians(60.0)
INSERT_CONTAINER = -330

TCP_VEL_1 = 200
TCP_ACC_1 = 200

TCP_VEL_2 = 40
TCP_ACC_2 = 40

# J_VEL_0 = 1.0
# J_ACC_0 = 1.0

J_VEL_1 = 1.0
J_ACC_1 = 2.0

# defined position for retrieval
SECOND_POSE_DX = 177.5
#SECOND_POSE_DX = 227.5
INSERT_DX = 90.0
RETRIEVAL_DX = SECOND_POSE_DX + INSERT_DX 
PASS_POLL_DY = 500.0
PASS_POLL_DX = -60.0
SECOND_POSE_OFFSETT = 72
Z_OFFSET = 0.0

# defined position for storage
# storage 全部で 420mm 前後
STORAGE_INSERT = 290
SPACER_INSERT_DX = 100
INSERT_BOOK_TIP_DX = 90
# INSERT_BOOK_FULL_DX = STORAGE_INSERT - SPACER_INSERT_DX - INSERT_BOOK_TIP_DX
PLACE_BOOK_DZ = -5.0
INSERT_BOOK_FULL_DX = 100

def _wrap_to_pi(rad: float) -> float:
        """角度差を [-pi, pi] に正規化（回転の無駄な大回りを避ける）"""
        return float((rad + np.pi) % (2.0 * np.pi) - np.pi)


class XArm7:
    """
    UR向けのRTDEベース関数群を xArm7 + xarm_python_sdk 用に書き直し

    - UR movej  -> set_servo_angle (joint space)
    - UR movel  -> set_position   (cartesian space straight-line)
    """

    def __init__(self, node, host: str = XARM_HOST, is_radian: bool = True):
        self.node = node
        self.move_tcp = MoveTCPClient(node)
        print("wentinit")
        """
        :param host: xArm コントロールボックスの IP
        :param is_radian: xArmAPI の角度単位。True なら rad, False なら deg
        """
        self.host = host
        self.arm = XArmAPI(host, is_radian=is_radian)

        #clean error and warn
        if self.arm.warn_code != 0:
            self.arm.clean_warn()
        if self.arm.error_code != 0:
            self.arm.clean_error()

        # self.arm.set_tcp_offset([-226.96, -19.125, 54.0, 0.0, 0.0, 0.0]) # manual setting
        # self.arm.set_tcp_offset([-226.96, -35.0, 54.0, 0.0, 0.0, 0.0]) # adjusted for book hand
        # self.arm.set_tcp_load(1.924, [41.86, -9.61, 67.48]) # preset : by Ufactory Studio error here

        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0) # mode 0: position mode

        self.arm.set_state(state=0)

        self.is_radian = is_radian

    # ========= low-level helper =========

    def disconnect(self):
        self.arm.disconnect()

    def _get_tcp_pose(self, is_radian: bool = True):
        """xArm の現在TCP姿勢を [x,y,z,roll,pitch,yaw] で返す"""
        code, pose = self.arm.get_position(is_radian=is_radian)
        if code != 0:
            raise RuntimeError(f"get_position failed, code={code}")
        return pose[:6]
    
    def _get_joint_angle(self, is_radian: bool = True):
        code, angles = self.arm.get_servo_angle(is_radian=is_radian)  # angles: list（7軸なら7要素）
        if code != 0:
            raise RuntimeError(f"get_position failed, code={code}")
        return angles[:7]
        

    def _moveJ(self, joints, velocity, acceleration, asynchronous: bool = False):
        """URの moveJ 相当: joint space point-to-point"""
        wait = not asynchronous
        # joints は rad 単位で渡す想定
        return self.arm.set_servo_angle(
            angle=joints,
            speed=velocity,
            mvacc=acceleration,
            is_radian=True,
            wait=wait,
        )

    def _moveL(self, pose, velocity, acceleration, asynchronous: bool = False):
        """URの moveL 相当: TCP直線補間"""
        wait = not asynchronous
        x, y, z, roll, pitch, yaw = pose
        return self.arm.set_position(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=velocity,
            mvacc=acceleration,
            is_radian=True,   # rpy は rad
            wait=wait,
            relative=False,
        )

    # ========= public helper =========

    def get_tcp_pose(self, is_radian: bool = True):
        """現在のTCP姿勢を取得 (デバッグ用)"""
        return self._get_tcp_pose(is_radian=is_radian)
    
    def get_joint_angle(self, is_radian: bool = True):
        """現在の関節角を取得 (デバッグ用)"""
        return self._get_joint_angle(is_radian=is_radian)

    # ========= UR版 moveJ_* の移植 =========


    def moveJ_to_capture_right(self,
                        velocity: float = J_VEL_1,
                        acceleration: float = J_ACC_1,
                        asynchronous: bool = False):
        """UR版 moveJ_to_init_Q"""
        joints_deg = CAPTURE_RIGHT
        joints_rad = convert.deg_to_rad(joints_deg)
        return self._moveJ(joints_rad, velocity, acceleration, asynchronous)
    
    def moveJ_to_capture_left(self,
                        velocity: float = J_VEL_1,
                        acceleration: float = J_ACC_1,
                        asynchronous: bool = False):
        """UR版 moveJ_to_init_Q"""
        joints_deg = CAPTURE_LEFT
        joints_rad = convert.deg_to_rad(joints_deg)
        return self._moveJ(joints_rad, velocity, acceleration, asynchronous)
    


    def moveL_tcp_z_offset(self,
                        dz_mm: float,
                        velocity: float = 120,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):
        """
        棚段ごとのTCP Z微調整（相対移動）。
        dz_mm: mm単位の相対オフセット
        """

        dz_mm = float(dz_mm)

        print(f"[TCP Z OFFSET] dz = {dz_mm:.2f} mm")

        return self.moveL_relative(
            [0.0, 0.0, dz_mm, 0.0, 0.0, 0.0],
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=asynchronous,
        )



    
    def moveJ_to_init_Q_DEG(self,
                        velocity: float = J_VEL_1,
                        acceleration: float = J_ACC_1,
                        asynchronous: bool = False):
        """UR版 moveJ_to_init_Q"""
        joints_deg = INIT_Q_DEG 
        joints_rad = convert.deg_to_rad(joints_deg)
        return self._moveJ(joints_rad, velocity, acceleration, asynchronous)

    def moveJ_to_barcode(self,
                        velocity: float = J_VEL_1,
                        acceleration: float = J_ACC_1,
                        asynchronous: bool = False):
        joints_deg = BARCODE_Q_A5
        joints_rad = convert.deg_to_rad(joints_deg)
        return self._moveJ(joints_rad, velocity, acceleration, asynchronous)
    # def moveJ_to_init_Q_offset(self,
    #                            velocity: float = J_VEL_1,
    #                            acceleration: float = J_ACC_1,
    #                            asynchronous: bool = False):
    #     """元のコメントアウトされていた offset版を一応移植"""
    #     joints_deg = INIT_Q_A5_OFFSET
    #     joints_rad = convert.deg_to_rad(joints_deg)
    #     return self._moveJ(joints_rad, velocity, acceleration, asynchronous)

    # ========= UR版 moveL_* の移植 =========

    def moveL_to_init_pose(self,
                           pose=None,
                           velocity: float = TCP_VEL_1,
                           acceleration: float = TCP_ACC_1,
                           asynchronous: bool = False):
        """
        初期TCP姿勢へ直線移動。
        UR版では pose=INIT_TCP_POSE_A5 をそのまま渡していたが、
        ここでは pos[mm], rpy[deg] -> pos[mm], rpy[rad] に変換して使う。
        """
        if pose is None:
            pos = INIT_TCP_POSE_A5[:3]
            rpy_deg = INIT_TCP_POSE_A5[3:]
            rpy_rad = convert.deg_to_rad(rpy_deg)
            pose = list(pos) + list(rpy_rad)
            #print("djoint=",pose)
        return self._moveL(pose, velocity, acceleration, asynchronous)

    def moveL_relative(self,
                   next_pose_relative: list,
                   velocity: float = TCP_VEL_1,
                   acceleration: float = TCP_ACC_1,
                   asynchronous: bool = False): # TODO 公式のrelative関数使用したほうが良いかと
        """
        現在TCP姿勢 + 差分(next_pose_relative) に moveL
        next_pose_relative = [dx, dy, dz, droll, dpitch, dyaw]
        """
        curr_pose = self._get_tcp_pose(is_radian=True)
        print(curr_pose)
        next_pose = [a + b for a, b in zip(curr_pose, next_pose_relative)]
        print(next_pose)
        return self._moveL(next_pose, velocity, acceleration, asynchronous)
    
    def moveJ_relative(
                self,
                djoints: list[float],
                velocity: float = J_VEL_1,
                acceleration: float= J_ACC_1,
                asynchronous: bool = False,
                ):
        """現在関節角 + 差分(djoints) に moveJ（rad）"""
        print("djoint=",djoints)
        curr_angle = self._get_joint_angle(is_radian=True)
        target = [c + d for c, d in zip(curr_angle, djoints)]
        return self._moveJ(target, velocity, acceleration, asynchronous)

    @staticmethod
    def culc_pos_relative_from_dy(dy: float):
        """
        画像座標→カメラ座標の結果から算出した，ハンドの y 方向移動量。
        UR版 culc_pos_relative_from_dy の移植。
        """
        pose_relative = [0.0, dy, 0.0, 0.0, 0.0, 0.0]
        return pose_relative

    def moveL_to_retrieval_pose(self,
                                dy: float,
                                d_roll: float,
                                velocity: float = TCP_VEL_1,
                                acceleration: float = TCP_ACC_1,
                                asynchronous: bool = False):
        """
        UR版 moveL_to_retrieval_pose の移植。
        - y方向に dy だけ平行移動
        - roll に d_roll (rad) だけ加算
        """
        curr_pose = self._get_tcp_pose(is_radian=True)

        # 位置成分の更新
        pos_relative = [0.0, dy, 0.0, 0.0, 0.0, 0.0]
        print('pos_relative : ', pos_relative)
        next_pose = [a + b for a, b in zip(curr_pose, pos_relative)]

        # RPY の roll を加算 (curr_pose[3:] はすでに rpy(rad) 前提)
        next_pose[3] += d_roll

        return self._moveL(next_pose, velocity, acceleration, asynchronous)

    def moveL_rot_relative(self,
                       d_roll: float,
                       velocity: float = TCP_VEL_1,
                       acceleration: float = TCP_ACC_1,
                       asynchronous: bool = False):
        """
        roll だけを d_roll (rad) 変化させる moveL。
        UR版 moveL_rot_relative の移植。
        """
        curr_pose = self._get_tcp_pose(is_radian=True)
        next_pose = list(curr_pose)
        next_pose[3] += d_roll
        return self._moveL(next_pose, velocity, acceleration, asynchronous)

    def moveL_to_2nd_pos(self,
                         velocity: float = TCP_VEL_1,
                         acceleration: float = TCP_ACC_1,
                         asynchronous: bool = False):
        """
        SECOND_POSE_DX だけ x 方向に寄る。
        UR版 moveL_to_2nd_pos の移植。
        """
        curr_pose = self._get_tcp_pose(is_radian=True)
        next_pose = list(curr_pose)
        next_pose[0] += SECOND_POSE_DX
        return self._moveL(next_pose, velocity, acceleration, asynchronous)
    
    def move_to_target_xyz_and_roll(
        self,
        p_robot_mm: np.ndarray,
        d_roll_rad: float,
        side: str,
        sleep_s: float = 0.02,
        velocity: float = TCP_VEL_1,
        acceleration: float = TCP_ACC_1,
        pos_tol_mm: float = 1.0,
    ):
        p_robot_mm = np.asarray(p_robot_mm, dtype=np.float64).reshape(3)

        right = 1
        left = -1
        x_offset_mm = -0.0
        k_roll = -4.0

        if side == "right":
            x_offset_mm *= right
            y_offset_mm = k_roll*right*np.sin(d_roll_rad)
        elif side == "left":
            x_offset_mm *= left
            y_offset_mm = k_roll*left*np.sin(d_roll_rad)
        else:
            raise ValueError("side must be 'right' or 'left'")

        curr = self.get_tcp_pose(is_radian=True)

        target_pose = [
            float(p_robot_mm[0] + x_offset_mm),
            float(p_robot_mm[1] + y_offset_mm),
            float(p_robot_mm[2]),
            float(curr[3] + d_roll_rad),
            float(curr[4]),
            float(curr[5]),
        ]

        print("\n========== ABS TARGET MOVE ==========")
        print("[target pose]")
        print(
            f"X={target_pose[0]:.2f} mm, "
            f"Y={target_pose[1]:.2f} mm, "
            f"Z={target_pose[2]:.2f} mm, "
            f"roll={target_pose[3]:.4f}, "
            f"pitch={target_pose[4]:.4f}, "
            f"yaw={target_pose[5]:.4f}"
        )

        ret = self._moveL(
            target_pose,
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=False,
        )

        time.sleep(sleep_s)

        after = self.get_tcp_pose(is_radian=True)
        after_xyz = np.array(after[:3], dtype=np.float64)
        target_xyz = np.array(target_pose[:3], dtype=np.float64)

        err_xyz = target_xyz - after_xyz
        err_norm = float(np.linalg.norm(err_xyz))

        print("[after pose]")
        print(
            f"X={after[0]:.2f} mm, "
            f"Y={after[1]:.2f} mm, "
            f"Z={after[2]:.2f} mm, "
            f"roll={after[3]:.4f}, "
            f"pitch={after[4]:.4f}, "
            f"yaw={after[5]:.4f}"
        )

        print("[position error]")
        print(
            f"dX={err_xyz[0]:.2f} mm, "
            f"dY={err_xyz[1]:.2f} mm, "
            f"dZ={err_xyz[2]:.2f} mm, "
            f"|e|={err_norm:.2f} mm"
        )

        if err_norm <= pos_tol_mm:
            print(f"[OK] reached target within {pos_tol_mm:.2f} mm")
        else:
            print(f"[WARN] target error is larger than tolerance: {err_norm:.2f} mm")

        print("=====================================\n")

        return ret

    def move_to_storage_target_xyz_and_roll(
        self,
        p_robot_mm: np.ndarray,
        d_roll_rad: float,
        side: str = "right",
        x_offset_mm: float = 0.0,
        y_offset_mm: float = 0.0,
        z_offset_mm: float = 0.0,
        roll_scale: float = 1.0,
        sleep_s: float = 0.02,
    ):
        """
        入庫用：認識した収納ターゲットXYZへリーチングする関数

        p_robot_mm:
            cam_mm_to_robot_mm() で変換済みのロボットベース座標 [x,y,z] [mm]

        d_roll_rad:
            認識した本/隙間の傾き角 [rad]

        side:
            "right" or "left"

        approach_offset_mm:
            挿入方向に対して少し手前に止めたい場合のオフセット
            最初は 0.0 推奨

        y_offset_mm, z_offset_mm:
            実機合わせ込み用の微調整値

        roll_scale:
            角度方向が逆なら -1.0 にする
        """

        p_robot_mm = np.asarray(p_robot_mm, dtype=np.float64).reshape(3)

        if side not in ["right", "left"]:
            raise ValueError("side must be 'right' or 'left'")

        curr = self.get_tcp_pose(is_radian=True)
        curr_xyz = np.array(curr[:3], dtype=np.float64)

        target_xyz = p_robot_mm.copy()

        # sideごとの挿入手前オフセット
        # どの軸が「棚へ近づく方向」かは実機座標に合わせて確認。
        # ここでは仮にX方向を手前/奥方向としている。
        if side == "right":
            target_xyz[0] += 0#-100
            target_xyz[1] += 0
            target_xyz[2] += 0


        elif side == "left":
            target_xyz[0] += 0
            target_xyz[1] += 0
            target_xyz[2] += 0

        dxyz = target_xyz - curr_xyz

        print("========== STORAGE REACHING ==========")
        print("[STORAGE] curr_xyz     =", curr_xyz)
        print("[STORAGE] target_xyz   =", target_xyz)
        print("[STORAGE] dxyz         =", dxyz)
        print("[STORAGE] d_roll_rad   =", d_roll_rad)
        print("[STORAGE] d_roll_deg   =", np.degrees(d_roll_rad))
        print("======================================")

        self.moveL_relative(
            [
                0.0,
                float(dxyz[1]),
                float(dxyz[2]),
                float(roll_scale * d_roll_rad),
                0.0,
                0.0,
            ],
        )


        self.moveL_relative(
            [
                float(dxyz[0])-100,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],

            asynchronous=False
        )

        time.sleep(sleep_s)
        return 0

    def moveL_to_2nd_pos_with_offset(self,
                                     velocity: float = TCP_VEL_1,
                                     acceleration: float = TCP_ACC_1,
                                     asynchronous: bool = False):
        """
        x 方向に SECOND_POSE_DX, z 方向に Z_OFFSET を加える版。
        """
        curr_pose = self._get_tcp_pose(is_radian=True)
        next_pose = list(curr_pose)
        next_pose[0] += SECOND_POSE_DX
        next_pose[2] += Z_OFFSET
        return self._moveL(next_pose, velocity, acceleration, asynchronous)

    def moveL_to_insert_right(self,
                        velocity: float = TCP_VEL_1,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):
        """
        INSERT_DX だけ x 方向に直線移動。
        UR版 moveL_to_insert の移植。
        """
        next_pose_relative = [INSERT_DX, 0.0, 0.0, 0.0, 0.0, 0.0]
        return self.moveL_relative(
            next_pose_relative,
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=asynchronous,
        )

    def moveL_to_insert_left(self,
                        velocity: float = TCP_VEL_1,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):
        """
        INSERT_DX だけ x 方向に直線移動。
        UR版 moveL_to_insert の移植。
        """
        next_pose_relative = [-INSERT_DX, 0.0, 0.0, 0.0, 0.0, 0.0]
        return self.moveL_relative(
            next_pose_relative,
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=asynchronous,
        )


    # def moveL_post_insert(self,
    #                       velocity: float = TCP_VEL_1,
    #                       acceleration: float = TCP_ACC_1,
    #                       asynchronous: bool = False):
    #     """
    #     INSERT_DX だけ戻る。
    #     UR版 moveL_post_insert の移植。
    #     """
    #     next_pose_relative = [-INSERT_DX, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     return self.moveL_relative(
    #         next_pose_relative,
    #         velocity=velocity,
    #         acceleration=acceleration,
    #         asynchronous=asynchronous,
    #     )
    def moveL_z_offset(self,
                        z_offset: float,
                        velocity: float = TCP_VEL_2,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):
        """
        INSERT_DX だけ x 方向に直線移動。
        UR版 moveL_to_insert の移植。
        """
        next_pose_relative = [0.0, 0.0, z_offset, 0.0, 0.0, 0.0]
        return self.moveL_relative(
            next_pose_relative,
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=asynchronous,
        )

    def moveL_y_offset(self,
                        y_offset: float,
                        velocity: float = TCP_VEL_2,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = True):
        """
        INSERT_DX だけ x 方向に直線移動。
        UR版 moveL_to_insert の移植。
        """
        next_pose_relative = [0.0, y_offset,0.0, 0.0, 0.0, 0.0]
        return self.moveL_relative(
            next_pose_relative,
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=asynchronous,
        )

    def moveL_post_grasp_right(self,
                         velocity: float = TCP_VEL_1,
                         acceleration: float = TCP_ACC_1,
                         asynchronous: bool = False):
        """
        把持後に x を INIT_TCP_POSE_A5[0] に戻す。
        UR版 moveL_post_grasp の移植。
        """
        curr_pose = self._get_tcp_pose(is_radian=True)
        next_pose = list(curr_pose)
        next_pose[0] -= RETRIEVAL_DX
        return self._moveL(next_pose, velocity, acceleration, asynchronous)
    
    def moveL_post_grasp_left(self,
                         velocity: float = TCP_VEL_1,
                         acceleration: float = TCP_ACC_1,
                         asynchronous: bool = False):
        """
        把持後に x を INIT_TCP_POSE_A5[0] に戻す。
        UR版 moveL_post_grasp の移植。
        """
        curr_pose = self._get_tcp_pose(is_radian=True)
        next_pose = list(curr_pose)
        next_pose[0] += RETRIEVAL_DX
        return self._moveL(next_pose, velocity, acceleration, asynchronous)
    

    # ========= placing in container (元コードの末尾) =========

    # def moveJ_to_pre_place(self,
    #                        pre_place_q_deg,
    #                        velocity: float = J_VEL_1,
    #                        acceleration: float = J_ACC_1,
    #                        asynchronous: bool = False):
    #     """
    #     UR版 moveJ_to_pre_place の一般形。
    #     PRE_PLACE_Q_DEG を外から渡す形にしてある。
    #     """
    #     joints_rad = convert.deg_to_rad(pre_place_q_deg)
    #     return self._moveJ(joints_rad, velocity, acceleration, asynchronous)

    # def moveL_to_place_1(self,
    #                      placing_pose,
    #                      velocity: float = TCP_VEL_1,
    #                      acceleration: float = TCP_ACC_1,
    #                      asynchronous: bool = False):
    #     """
    #     UR版 moveL_to_place_1 の一般形。
    #     PLACING_1_POSE を外から渡す。
    #     placing_pose は [x,y,z,roll(rad),pitch(rad),yaw(rad)]。
    #     """
    #     return self._moveL(placing_pose, velocity, acceleration, asynchronous)

    # def moveL_to_post_place_1(self,
    #                           post_place_pose,
    #                           velocity: float = TCP_VEL_1,
    #                           acceleration: float = TCP_ACC_2,
    #                           asynchronous: bool = False):
    #     """
    #     UR版 moveL_to_post_place_1 の一般形。
    #     POST_PLACE_POSE を外から渡す。
    #     """
    #     return self._moveL(post_place_pose, velocity, acceleration, asynchronous)4

    def pass_poll(self,
                  target_pose: list,
                  velocity: float = TCP_VEL_2,
                  acceleration: float = TCP_ACC_1,
                  asynchronous: bool = False):
        """
        INSERT_DZ だけ Z 方向に直線移動。
        UR版 moveL_to_insert の移植。
        """
        curr_pose = self.get_tcp_pose(is_radian=True)
        next_pose_relative = [t - c for t, c in zip(target_pose, curr_pose)]
        return self.moveL_relative(
            next_pose_relative,
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=asynchronous,
        ) 
    def distant_hand(self,
                            velocity: float = J_VEL_1,
                            acceleration: float = J_ACC_1,
                            asynchronous: bool = False):
            """
            Joint6だけ回転
            """
            djoint = [0.0, 0.0, 0.0, 0.0, 0.0, DISTANCE_BOOKSHELF, 0.0]
            return self.moveJ_relative(
                djoint,
                velocity=velocity,
                acceleration=acceleration,
                asynchronous=asynchronous,
            )
    def rotate_hand(self,
                            velocity: float = J_VEL_1,
                            acceleration: float = J_ACC_1,
                            asynchronous: bool = False):
            """
            Joint7だけ回転
            """
            djoint = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, DISTANCE_BOOKSHELF]
            return self.moveJ_relative(
                djoint,
                velocity=velocity,
                acceleration=acceleration,
                asynchronous=asynchronous,
            )
    #進行方向左側の書架のみ，もしかしたら両方使えるかも
    def switch_gripper_pose(self,
                        joints_deg: list,
                        velocity: float = J_VEL_1,
                        acceleration: float = J_ACC_1,
                        asynchronous: bool = False
                        ):
        curr_joint = self.get_joint_angle(is_radian=True)
        target_rad = convert.deg_to_rad(joints_deg)
        djoint = [t - c for t, c in zip(target_rad, curr_joint)]
        return self.moveJ_relative(djoint, velocity, acceleration, asynchronous)
    
    
    
    def insert_container(self,
                        velocity: float = TCP_VEL_2,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):
        """
        INSERT_DZ だけ Z 方向に直線移動。
        UR版 moveL_to_insert の移植。
        """
        next_pose_relative = [0.0, 0.0, INSERT_CONTAINER, 0.0, 0.0, 0.0]
        return self.moveL_relative(
            next_pose_relative,
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=asynchronous,
        )
            
    def distant_container(self,
                        velocity: float = TCP_VEL_2,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):
        """
        INSERT_DZ だけ Z 方向に直線移動。
        UR版 moveL_to_insert の移植。
        """
        next_pose_relative = [0.0, 0.0, -INSERT_CONTAINER, 0.0, 0.0, 0.0]
        return self.moveL_relative(
            next_pose_relative,
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=asynchronous,
        )
    def container_offset(self,
                        container_offset : float,
                        velocity: float = TCP_VEL_2,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):
        """
        TCPの +X 方向へ前後にスライド
        INSERT_DZ だけ Z 方向に直線移動。
        UR版 moveL_to_insert の移植。
        """
        next_pose_relative = [container_offset, 0.0, 0.0, 0.0, 0.0, 0.0]
        return self.moveL_relative(
            next_pose_relative,
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=asynchronous,
        )
    
   


# Storage -----------------------------------------------------------------------------------------------------------------------------

    def moveL_to_storage_pose(self,
                                dy: float,
                                d_roll: float = 0,
                                velocity: float = TCP_VEL_1,
                                acceleration: float = TCP_ACC_1,
                                asynchronous: bool = False):
        """
        UR版 moveL_to_retrieval_pose の移植。
        - y方向に dy だけ平行移動
        - roll に d_roll (rad) だけ加算
        """
        curr_pose = self._get_tcp_pose(is_radian=True)

        # 位置成分の更新
        pos_relative = [0.0, dy, 0.0, 0.0, 0.0, 0.0]
        print('pos_relative : ', pos_relative)
        next_pose = [a + b for a, b in zip(curr_pose, pos_relative)]

        # RPY の roll を加算 (curr_pose[3:] はすでに rpy(rad) 前提)
        next_pose[3] += d_roll

        return self._moveL(next_pose, velocity, acceleration, asynchronous)
    

    def moveL_to_slide(self, dy: float,
                        velocity: float = TCP_VEL_2,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):
        """
        y方向に dy だけ平行移動
        """
        curr_pose = self._get_tcp_pose(is_radian=True)

        # 位置成分の更新
        pos_relative = [0.0, dy, 0.0, 0.0, 0.0, 0.0]
        print('pos_relative : ', pos_relative)
        next_pose = [a + b for a, b in zip(curr_pose, pos_relative)]

        return self._moveL(next_pose, velocity, acceleration, asynchronous)
    

    def moveL_to_insert_spacer(self,
                        velocity: float = TCP_VEL_2,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):


        curr_pose = self._get_tcp_pose(is_radian=True)
        pos_relative = [SPACER_INSERT_DX, 0.0, 0.0, 0.0, 0.0, 0.0]
        print('pos_relative : ', pos_relative)
        next_pose = [a + b for a, b in zip(curr_pose, pos_relative)]

        # RPY の roll を加算 (curr_pose[3:] はすでに rpy(rad) 前提)

        return self._moveL(next_pose, velocity, acceleration, asynchronous)
    

    def move_L_to_insert_book_tip(self,
                        velocity: float = TCP_VEL_2,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):


        curr_pose = self._get_tcp_pose(is_radian=True)
        pos_relative = [INSERT_BOOK_TIP_DX, 0.0, 0.0, 0.0, 0.0, 0.0]
        print('pos_relative : ', pos_relative)
        next_pose = [a + b for a, b in zip(curr_pose, pos_relative)]

        # RPY の roll を加算 (curr_pose[3:] はすでに rpy(rad) 前提)

        return self._moveL(next_pose, velocity, acceleration, asynchronous)
    

    def moveL_to_insert_book_full(self,
                        velocity: float = TCP_VEL_2,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):
        curr_pose = self._get_tcp_pose(is_radian=True)
        pos_relative = [INSERT_BOOK_FULL_DX, 0.0, 0.0, 0.0, 0.0, 0.0]
        next_pose = [a + b for a, b in zip(curr_pose, pos_relative)]
        return self._moveL(next_pose, velocity, acceleration, asynchronous)

    def moveL_to_place_book(self,
                        velocity: float = TCP_VEL_2,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):
        curr_pose = self._get_tcp_pose(is_radian=True)
        pos_relative = [0.0, 0.0, PLACE_BOOK_DZ, 0.0, 0.0, 0.0]
        next_pose = [a + b for a, b in zip(curr_pose, pos_relative)]

        return self._moveL(next_pose, velocity, acceleration, asynchronous)


    def moveL_to_post_storage(self,
                        velocity: float = TCP_VEL_1,
                        acceleration: float = TCP_ACC_1,
                        asynchronous: bool = False):
        curr_pose = self._get_tcp_pose(is_radian=True)
        pos_relative = [-STORAGE_INSERT, 0.0, -PLACE_BOOK_DZ, 0.0, 0.0, 0.0]
        next_pose = [a + b for a, b in zip(curr_pose, pos_relative)]

        return self._moveL(next_pose, velocity, acceleration, asynchronous)
    
    
    def move_to_target_rpy_only(
        arm: "XArm7",
        *,
        target_pose_mm_rad: np.ndarray | list,  # [x,y,z,roll,pitch,yaw] xyz[mm], rpy[rad]
        sleep_s: float = 0.02,
        wrap_angle: bool = True,
    ):
        """
        現在の xyz は維持したまま、roll/pitch/yaw だけ target に合わせる
        """
        target = np.asarray(target_pose_mm_rad, dtype=np.float64).reshape(6)

        curr = arm.get_tcp_pose(is_radian=True)  # [mm,mm,mm, roll,pitch,yaw] (想定)
        curr_rpy = np.array(curr[3:6], dtype=np.float64)
        target_rpy = target[3:6]

        drpy = target_rpy - curr_rpy
        if wrap_angle:
            drpy = np.array([_wrap_to_pi(v) for v in drpy], dtype=np.float64)

        print("drpy (deg) =", np.degrees(drpy))

        # xyzは0、rpyだけ動かす（moveL_relative は [dx,dy,dz,droll,dpitch,dyaw]）
        arm.moveL_relative(
            [0.0, 0.0, 0.0, float(drpy[0]), float(drpy[1]), float(drpy[2])],
            asynchronous=False,

        )
        time.sleep(sleep_s)


    def move_to_target_xyz_only(
        arm: "XArm7",
        *,
        target_pose_mm_rad: np.ndarray | list,  # [x,y,z,roll,pitch,yaw] xyz[mm], rpy[rad]
        sleep_s: float = 0.02,
    ):
        """
        現在の roll/pitch/yaw は維持したまま、xyz だけ target に合わせる
        """
        target = np.asarray(target_pose_mm_rad, dtype=np.float64).reshape(6)

        curr = arm.get_tcp_pose(is_radian=True)  # [mm,mm,mm, roll,pitch,yaw] (想定)
        curr_xyz = np.array(curr[:3], dtype=np.float64)
        target_xyz = target[:3]

        dxyz = target_xyz - curr_xyz  # [mm]
        print("dxyz =", dxyz)

        # rpyは0、xyzだけ動かす
        arm.moveL_relative(
            [float(dxyz[0]), float(dxyz[1]), float(dxyz[2]), 0.0, 0.0, 0.0],
            asynchronous=False,
        )
        time.sleep(sleep_s)

    
   
# for test --------------------------------------------------------------------------------------------------------
    def moveJ_to_place_for_test(self,
                           velocity: float = J_VEL_1,
                           acceleration: float = J_ACC_1,
                           asynchronous: bool = False):
        """
        テスト用の配置姿勢へ moveJ
        """
        joints_rad = convert.deg_to_rad(PLACING_FOR_TEST_Q)
        return self._moveJ(joints_rad, velocity, acceleration, asynchronous)
    
    def moveL_to_point_m(
        self,
        p_xyz_m,
        *,
        keep_rpy: bool = True,
        rpy_rad=None,                 # keep_rpy=False のとき [roll,pitch,yaw]
        approach_dz_m: float = 0.05,  # まず5cm上へ寄ってから降りる（安全用）
        velocity: float = TCP_VEL_2,
        acceleration: float = TCP_ACC_2,
        asynchronous: bool = False,
    ):
        """
        base座標系の点 p_xyz_m=[x,y,z] (m) へ moveL する（内部で mm に変換して set_position に渡す）。
        """
        # 位置: m -> mm
        x_mm, y_mm, z_mm = p_xyz_m[0], p_xyz_m[1], p_xyz_m[2]   

        # 姿勢: 現在の rpy を維持するか、指定する
        if keep_rpy:
            curr = self._get_tcp_pose(is_radian=True)  # [x,y,z,roll,pitch,yaw] :contentReference[oaicite:3]{index=3}
            roll, pitch, yaw = curr[3], curr[4], curr[5]
        else:
            if rpy_rad is None:
                raise ValueError("rpy_rad must be provided when keep_rpy=False")
            roll, pitch, yaw = map(float, rpy_rad)

        # まず上から寄る（z+dz）→ 目標zへ
        if approach_dz_m is not None and float(approach_dz_m) != 0.0:
            pre_pose = [x_mm, y_mm, z_mm + float(approach_dz_m) * 1000.0, roll, pitch, yaw]
            self._moveL(pre_pose, velocity, acceleration, asynchronous)  # :contentReference[oaicite:4]{index=4}

        pose = [x_mm, y_mm, z_mm, roll, pitch, yaw]
        return self._moveL(pose, velocity, acceleration, asynchronous)  # :contentReference[oaicite:5]{index=5}

    # ========= public generic moveJ =========
    def moveJ(self,
              joints_rad: list,
              velocity: float = J_VEL_1,
              acceleration: float = J_ACC_1,
              asynchronous: bool = False):
        """
        汎用 moveJ（関節角は rad 指定）
        """
        return self._moveJ(
            joints_rad,
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=asynchronous,
        )
    # ========= safety / state check =========

    def get_state(self) -> int:
        code, state = self.arm.get_state()
        if code != 0:
            raise RuntimeError(f"get_state failed code={code}")
        return state

    def get_err_warn(self) -> tuple[int, int]:
        """(error_code, warn_code)"""
        return self.arm.get_err_warn_code()

    def ensure_xarm_ok(arm, where=""):
        state = arm.get_state()
        err, warn = arm.get_err_warn()

        if state in (4, 5) or err != 0:
            print(
                f"[FATAL] xArm abnormal at {where} | "
                f"state={state}, err={err}, warn={warn}"
            )
            arm.emergency_stop()

        
    def is_motion_enabled(self) -> int:
        code, enabled = self.arm.get_motion_enable()
        if code != 0:
            raise RuntimeError("get_motion_enable failed")
        return enabled
        
    def emergency_stop(self):
        print("[EMERGENCY STOP]")

        try:
            # まず動作停止
            self.arm.set_state(4)  # STOP state
        except Exception as e:
            print("set_state failed:", e)

        try:
            # モーション停止
            self.arm.motion_enable(False)
        except Exception as e:
            print("motion_enable False failed:", e)

        try:
            # もしstop APIがあるなら
            self.arm.stop_lite6()  # ← ない場合は削除
        except Exception:
            pass


    def move_tcp_execute(self, dx: float, executor):

        while not self.move_tcp._client.wait_for_server(timeout_sec=0.1):
            executor.spin_once(timeout_sec=0.1)

        future = self.move_tcp.move_tcp(dx=dx)

        while rclpy.ok() and not future.done():
            executor.spin_once(timeout_sec=0.05)

        goal_handle = future.result()

        if not goal_handle.accepted:
            raise RuntimeError("MoveTCP rejected")

        result_future = goal_handle.get_result_async()

        while rclpy.ok() and not result_future.done():
            executor.spin_once(timeout_sec=0.05)

        result = result_future.result().result

        if not result.success:
            raise RuntimeError(f"MoveTCP failed: {result.message}")

        traj = result.trajectory

        # 最終関節角度だけ取得
        final_joint = traj.points[-1].positions

        # moveJモード
        self.arm.set_mode(0)
        self.arm.set_state(0)

        self.moveJ(
            final_joint,
            velocity=0.5,
            acceleration=0.5,
            asynchronous=False
        )

        return True

    def moveJ_to_return_pose_direct(
        arm,
        joint_deg=None,
        speed=20,
        mvacc=200,
        wait=True,
    ):
        """
        calibration_valid.py 内だけで使う退避姿勢移動。
        xarm7.py は変更しない。
        joint_deg 単位: deg
        """

        if joint_deg is None:
            joint_deg = RETURN_JOINT_DEG

        print("\n========== RETURN JOINT MOVE ==========")
        print("target joint deg =", joint_deg)
        print("speed =", speed)
        print("mvacc =", mvacc)
        print("=======================================\n")

        # XArm7クラスの中にSDK本体が arm.arm として入っている場合
        sdk_arm = getattr(arm, "arm", None)

        # もし arm.arm が無ければ arm._arm も見る
        if sdk_arm is None:
            sdk_arm = getattr(arm, "_arm", None)

        # それでも無ければ、XArm7自体が set_servo_angle を持っているか見る
        if sdk_arm is None:
            sdk_arm = arm

        if not hasattr(sdk_arm, "set_servo_angle"):
            raise RuntimeError(
                "set_servo_angle が見つからない。XArm7内のSDK本体の変数名を確認して。"
            )

        ret = sdk_arm.set_servo_angle(
            angle=joint_deg,
            speed=speed,
            mvacc=mvacc,
            is_radian=False,
            wait=wait,
        )

        print("[return pose ret] =", ret)
        return ret

def main():
    rclpy.init()

    node = rclpy.create_node("main_node")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    arm = XArm7(node)
    while not arm.move_tcp._client.wait_for_server(timeout_sec=0.1):
        executor.spin_once(timeout_sec=0.1)

    arm.moveJ_to_init_Q(asynchronous=False)

    future = arm.move_tcp.move_tcp(dx=-0.05)

    while rclpy.ok() and not future.done():
        executor.spin_once(timeout_sec=0.1)

    goal_handle = future.result()
    if not goal_handle.accepted:
        raise RuntimeError("MoveTCP rejected")

    result_future = goal_handle.get_result_async()
    while rclpy.ok() and not result_future.done():
        executor.spin_once(timeout_sec=0.1)

    print("MoveTCP done")

    arm.disconnect()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    # ここは元の UR コード最後のデモ相当
    # PRE_PLACE_Q_DEG, PLACING_1_POSE, POST_PLACE_POSE を定義していれば:
    # arm.moveJ_to_pre_place(PRE_PLACE_Q_DEG)
    # arm.moveL_to_place_1(PLACING_1_POSE)
    # arm.moveL_to_post_place_1(POST_PLACE_POSE)
    # arm.moveJ_to_init_Q()


