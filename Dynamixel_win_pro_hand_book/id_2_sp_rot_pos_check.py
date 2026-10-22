from .dynamixel_cross_platform import Dynamixel
from . import kbhit_lin as kbhit
import time
from .util.cfg_dict_loader import DynamixelCfg

from pathlib import Path


PKG_DIR = Path(__file__).resolve().parent
cfg_path = PKG_DIR / "config" / "Dynamixel_config.yaml"

cfg = DynamixelCfg(str(cfg_path))

PORT = cfg.id.port
BAUDRATE = cfg.id.baudrate

GRIPPER_ID = cfg.id.gripper
SP_ROT_ID = cfg.id.spacer_rot
SP_LIN_ID = cfg.id.spacer_lin

GRIPPER_BL = cfg.pos.gripper.backlash # backlash
GRIPPER_CLOSE = cfg.pos.gripper.close
GRIPPER_FULL_OPEN = cfg.pos.gripper.range + GRIPPER_CLOSE + GRIPPER_BL
GRIPPER_ROT_GAIN = cfg.cont_gain.gripper.pos_cont
GRIPPER_THETA_0 = cfg.pos.gripper.theta_zero
GRIPPER_R2S = cfg.pos.gripper.rad_to_step
GRIPPER_GR = cfg.pos.gripper.gear_ratio
RELEASE_GAIN = cfg.pos.gripper.release

SP_ROT_0 = cfg.pos.spacer_rot.zero
# TODO SP_ROT_MAX = 4095*2 + SP_ROT_0
SP_ROT_R2S = cfg.pos.spacer_rot.rad_to_step
SP_ROT_GR = cfg.pos.spacer_rot.gear_ratio
SP_ROT_BL = cfg.pos.spacer_rot.backlash

SP_LIN_BACK = cfg.pos.spacer_lin.back
SP_LIN_KEEP = -1 * cfg.pos.spacer_lin.keep
SP_LIN_FRONT = SP_LIN_BACK - cfg.pos.spacer_lin.range

VELOCITY_THRESHOLD = cfg.thresh.vel
POSITION_THRESHOLD = cfg.thresh.pos


try:

    dxl = Dynamixel("/dev/ttyUSB0", 57600)
    kb = kbhit.KBHit()
    time.sleep(0.5)

    # 制御モード変更前にトルクをOFF
    dxl.disable_torque(2)
    time.sleep(0.2)

    # 拡張位置制御モードに変更
    dxl.set_mode_ex_position(2)
    time.sleep(0.2)

    init_pos  = int(dxl.read_position(2)) #現在の位置を取得し目標値の初期値として代入
    print("current position = ", init_pos)
    curr_pos = init_pos
    curr_des_pos = init_pos
    last_sent_pos = curr_des_pos

    rot_gain = 50 # キー入力1つで何deg 回転させるか（1→0.0043 deg）50: 0.21 deg

    dxl.enable_torque(2) #トルクをオンにする（手で動かせなくなる）
    time.sleep(1)

    print("---------------------------------")
    print("   Dynamixel READY TO MOVE  ")
    print("---------------------------------")

    while True:

      if kb.kbhit(): #キーボードの打鍵があったら（無いときはここで待ち状態になる）
        ch = kb.getch()
        if ch == 'K': # L arrow
          curr_des_pos -= rot_gain  # cw
        elif ch == 'M': # R arrow
          curr_des_pos +=  rot_gain # ccw
        # curr_des_pos = max(min_1,min(curr_des_pos, max_1)) #数値的に上下限を超えないようにする
        dxl.write_position(2, curr_des_pos)  #Dynamixelに数値を書き込む
        # print(dxl.read_position(1))


except KeyboardInterrupt:
  
    # dxl.write_position(2, SP_ROT_0)
    time.sleep(5)
    print(dxl.read_position(2)) # last position check
    dxl.disable_torque(2)
    dxl.close_port()
    kb.set_normal_term()
