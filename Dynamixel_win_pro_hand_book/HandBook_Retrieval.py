from .dynamixel_cross_platform import Dynamixel
from .util.cfg_dict_loader import DynamixelCfg
from . import kbhit_lin as kbhit
import time
import math

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
GRIPPER_CALIB_A = cfg.pos.gripper.calib_width_a
GRIPPER_CALIB_B = cfg.pos.gripper.calib_width_b

VELOCITY_THRESHOLD = cfg.thresh.vel
POSITION_THRESHOLD = cfg.thresh.pos



def init_dynamixels():
    
    dxl = Dynamixel(port=PORT, baudrate=BAUDRATE)
    dxl.set_mode_ex_position(GRIPPER_ID)
    print(f'Gripper Position : {dxl.read_position(GRIPPER_ID)}')
    time.sleep(0.2)
    
    print("---------------------------------")
    print("   Dynamixel READY TO MOVE  ")
    print("---------------------------------")

    return dxl

def open_servo_key(dxl):

    kb = kbhit.KBHit()

    dxl.enable_torque(GRIPPER_ID)
    curr_des_pos = dxl.read_position(GRIPPER_ID)
    last_sent_pos = curr_des_pos

    while True:
        if kb.kbhit():
            ch = kb.getch()
            if ch == 'K': # L arrow : grippen open
                curr_des_pos += GRIPPER_ROT_GAIN
                
            elif ch == 'M': # R arrow : gripper close
                curr_des_pos -=  GRIPPER_ROT_GAIN
                
            elif ch in ('g', 'G'):  # g/G key : break and goto next sequence
                break
            
        curr_des_pos = max(GRIPPER_CLOSE, min(curr_des_pos, GRIPPER_FULL_OPEN)) # range limitation
        
        if curr_des_pos != last_sent_pos:
            dxl.write_position(GRIPPER_ID, curr_des_pos)  # set gripper position
            
def calib_width(w_hat): # 20250825時点
    # TODO この関数を使わなくて良くする（バックラッシ，theta_0，rad_to_step 等の調整）
    # width = w_hat/0.901 - 0.4627
    width = (w_hat - GRIPPER_CALIB_B)/GRIPPER_CALIB_A
    return width
    
def open_until_width(dxl, width, gravity=False): # width : mm, from close position
    
    width = calib_width(width)
    
    dxl.enable_torque(GRIPPER_ID)
    d_theta = GRIPPER_GR * math.asin((width + 20)/80 - GRIPPER_THETA_0) # radian  #ここちがうくね？
    d_step = int(d_theta * GRIPPER_R2S + GRIPPER_BL)
    curr_pos = dxl.read_position(GRIPPER_ID)
    if gravity == True:
        des_pos = max(GRIPPER_CLOSE, min((curr_pos + d_step - GRIPPER_BL), GRIPPER_FULL_OPEN)) # 重力かかったときにグリッパ開きすぎないようにバックラッシ分差し引く
    elif gravity == False:
        des_pos = max(GRIPPER_CLOSE, min((curr_pos + d_step), GRIPPER_FULL_OPEN))
        
    dxl.write_position(GRIPPER_ID, des_pos)
    

def grasp(dxl, timeout_sec=3.0):

    dxl.enable_torque(GRIPPER_ID)
    print("grasping object until detection")

    try:
        dxl.write_position(GRIPPER_ID, GRIPPER_CLOSE)
        print("grasping")
    except Exception as e:
        raise RuntimeError(f"Dynamixel write failed: {e}")

    time.sleep(0.05)

    timeout = time.time() + timeout_sec

    while True:
        # ----- タイムアウト -----
        if time.time() > timeout:
            raise RuntimeError("Grasp timeout (Dynamixel not responding)")

        # ----- 通信エラー検知 -----
        try:
            curr_vel = dxl.read_velocity(GRIPPER_ID)
            curr_pos = dxl.read_position(GRIPPER_ID)
        except Exception as e:
            raise RuntimeError(f"Dynamixel read failed: {e}")

        # ----- 把持判定 -----
        if abs(curr_vel) < VELOCITY_THRESHOLD:
            print('grasp detected')
            dxl.disable_torque(GRIPPER_ID)
            break

        elif GRIPPER_CLOSE - POSITION_THRESHOLD < curr_pos < GRIPPER_CLOSE + POSITION_THRESHOLD:
            print('gripper close')
            dxl.disable_torque(GRIPPER_ID)
            break


def open_until_full(dxl, asynchronous=False):

    print("gripper open until full")
    dxl.enable_torque(GRIPPER_ID)
    dxl.write_position(GRIPPER_ID, GRIPPER_FULL_OPEN)
    time.sleep(0.05)

    if asynchronous:
        return

    timeout = time.time() + 3.0  # 3秒タイムアウト

    while True:
        if time.time() > timeout:
            raise RuntimeError("Dynamixel timeout during open_until_full")

        try:
            curr_vel = dxl.read_velocity(GRIPPER_ID)
            curr_pos = dxl.read_position(GRIPPER_ID)
        except Exception as e:
            raise RuntimeError(f"Dynamixel communication lost: {e}")

        if curr_vel is None or curr_pos is None:
            raise RuntimeError("Dynamixel returned None")

        if abs(curr_vel) < VELOCITY_THRESHOLD:
            print("gripper full open")
            break

        if GRIPPER_FULL_OPEN - POSITION_THRESHOLD < curr_pos < GRIPPER_FULL_OPEN + POSITION_THRESHOLD:
            print("gripper full open")
            break

    dxl.disable_torque(GRIPPER_ID)

if __name__ == '__main__':
# test
    try:        
        dxl = init_dynamixels()
        # test
        open_until_width(dxl, 15.0)
        input()
        grasp(dxl)
        time.sleep(3.0)
        dxl.disable_torque(GRIPPER_ID)
        dxl.close_port()
        
    except KeyboardInterrupt:
        
        dxl.disable_torque(GRIPPER_ID)
        dxl.close_port()




