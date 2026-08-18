from .dynamixel_cross_platform import Dynamixel
from .util.cfg_dict_loader import DynamixelCfg
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


def init_dynamixels():

    dxl = Dynamixel(port=PORT, baudrate=BAUDRATE)
    time.sleep(1)

    dxl.set_mode_ex_position(GRIPPER_ID)
    print(f'Gripper Position : {dxl.read_position(GRIPPER_ID)}')
    dxl.set_mode_ex_position(SP_ROT_ID)
    print(f'Spacer-Rot Position : {dxl.read_position(SP_ROT_ID)}')
    dxl.set_mode_ex_position(SP_LIN_ID)
    print(f'Spacer-Lin Position : {dxl.read_position(SP_LIN_ID)}')
    time.sleep(1)

    print("---------------------------------")
    print("   Dynamixel READY TO MOVE  ")
    print("---------------------------------")

    return dxl


def expand_sp_lin(dxl, asynchronous = False): # TODO async 対応

    dxl.enable_torque(SP_LIN_ID)
    time.sleep(0.1)
    dxl.write_position(SP_LIN_ID, SP_LIN_FRONT)

    if asynchronous == True:
        print("spacer expanding")
        pass
    elif asynchronous == False:
        while True:
            if SP_LIN_FRONT - POSITION_THRESHOLD < dxl.read_position(SP_LIN_ID) < SP_LIN_FRONT + POSITION_THRESHOLD:
                print('spacer expanded')
                dxl.disable_torque(SP_LIN_ID)
                break



def open_until_full(dxl, asynchronous = False):
    
    print("gripper open until full")
    dxl.enable_torque(GRIPPER_ID)
    dxl.write_position(GRIPPER_ID, GRIPPER_FULL_OPEN)
    time.sleep(0.05)
    if asynchronous == True:
        print("motor torque is still enabled")
        pass
    elif asynchronous == False:
        while True:
            curr_vel = dxl.read_velocity(GRIPPER_ID)
            if abs(curr_vel) < VELOCITY_THRESHOLD: # open detection by velocity
                print('gripper full open')
                break

            elif GRIPPER_FULL_OPEN - POSITION_THRESHOLD < dxl.read_position(GRIPPER_ID) < GRIPPER_FULL_OPEN + POSITION_THRESHOLD:
                print('gripper full open')
                break

        dxl.disable_torque(GRIPPER_ID)


def grasp(dxl, timeout_sec=3.0, keep_torque=True):

    dxl.enable_torque(GRIPPER_ID)
    print("grasping object until detection")

    try:
        dxl.write_position(GRIPPER_ID, GRIPPER_CLOSE)
        print("grasping")
    except Exception as e:
        raise RuntimeError(f"Dynamixel write failed: {e}")

    # 指令直後の速度0誤判定を避ける
    time.sleep(0.4)

    timeout = time.time() + timeout_sec
    low_vel_count = 0
    LOW_VEL_REQUIRED_COUNT = 8

    while True:
        if time.time() > timeout:
            raise RuntimeError("Grasp timeout (Dynamixel not responding)")

        try:
            curr_vel = dxl.read_velocity(GRIPPER_ID)
            curr_pos = dxl.read_position(GRIPPER_ID)
        except Exception as e:
            raise RuntimeError(f"Dynamixel read failed: {e}")

        print(f"[grasp] pos={curr_pos}, vel={curr_vel}, target={GRIPPER_CLOSE}")

        # 先に「閉じ切った」を判定
        if abs(curr_pos - GRIPPER_CLOSE) < POSITION_THRESHOLD:
            print("gripper close")
            if not keep_torque:
                dxl.disable_torque(GRIPPER_ID)
            break

        # 閉じ切る前に速度が小さい状態が連続したら把持判定
        if abs(curr_vel) < VELOCITY_THRESHOLD:
            low_vel_count += 1
        else:
            low_vel_count = 0

        if low_vel_count >= LOW_VEL_REQUIRED_COUNT:
            print("grasp detected before full close")

            if keep_torque:
                hold_pos = curr_pos - 5
                dxl.write_position(GRIPPER_ID, hold_pos)
                print(f"[grasp] hold position set to {hold_pos}")
            else:
                dxl.disable_torque(GRIPPER_ID)

            break

        time.sleep(0.05)


def rotate_spacer(dxl, theta_deg = 90.0):

    dxl.enable_torque(SP_ROT_ID)
    des_pos = SP_ROT_0 - int((theta_deg * math.pi / 180.0) * SP_ROT_R2S * SP_ROT_GR + SP_ROT_BL)
    dxl.write_position(SP_ROT_ID, des_pos)
    time.sleep(0.05)
    while True:

        if des_pos - POSITION_THRESHOLD < dxl.read_position(SP_ROT_ID) < des_pos + POSITION_THRESHOLD:
            print(f'spacer rotated to {theta_deg} deg')
            dxl.disable_torque(SP_ROT_ID)
            break


def reset_rot(dxl, asynchronous = False):

    dxl.enable_torque(SP_ROT_ID)
    dxl.write_position(SP_ROT_ID, SP_ROT_0)
    time.sleep(0.05)
    if asynchronous == True:
        print("spacer rotating to zero")
        pass
    elif asynchronous == False:
        while True:

            if SP_ROT_0 - POSITION_THRESHOLD < dxl.read_position(SP_ROT_ID) < SP_ROT_0 + POSITION_THRESHOLD:
                print('spacer rotation reset to zero')
                dxl.disable_torque(SP_ROT_ID)
                break


def contract_sp_lin_1(dxl, asynchronous = False): # TODO async 対応

    dxl.enable_torque(SP_LIN_ID)
    time.sleep(0.5)
    dxl.write_position(SP_LIN_ID, SP_LIN_KEEP)

    while True:
        dxl.read_position(SP_LIN_ID)
        time.sleep(0.05)
        if SP_LIN_KEEP - POSITION_THRESHOLD < dxl.read_position(SP_LIN_ID) < SP_LIN_KEEP + POSITION_THRESHOLD:
            print('spacer contracted')
            dxl.disable_torque(SP_LIN_ID)
            break


def contract_sp_lin_2(
    dxl,
    asynchronous=False,
    timeout_sec=15.0
):
    """
    スペーサ直動軸をSP_LIN_BACKまで収縮させる。

    asynchronous=False:
        目標位置到達まで待つ。
    asynchronous=True:
        位置指令だけ送ってすぐ戻る。
    """

    print(
        f"[SP_LIN_2] start contraction: "
        f"target={SP_LIN_BACK}, "
        f"threshold={POSITION_THRESHOLD}"
    )

    dxl.enable_torque(SP_LIN_ID)
    time.sleep(0.5)

    current_pos = dxl.read_position(SP_LIN_ID)
    print(f"[SP_LIN_2] position before command={current_pos}")

    dxl.write_position(SP_LIN_ID, SP_LIN_BACK)
    print(f"[SP_LIN_2] position command sent: target={SP_LIN_BACK}")

    # 非同期の場合は指令後すぐに戻る
    if asynchronous:
        print("[SP_LIN_2] asynchronous command completed")
        return True

    start_time = time.monotonic()
    last_print_time = 0.0

    while True:
        elapsed = time.monotonic() - start_time

        try:
            current_pos = dxl.read_position(SP_LIN_ID)
        except Exception as e:
            print(f"[SP_LIN_2][WARN] read_position failed: {e}")
            current_pos = None

        # 0.2秒ごとに現在位置を表示
        if elapsed - last_print_time >= 0.2:
            print(
                f"[SP_LIN_2] "
                f"elapsed={elapsed:.1f}s, "
                f"current={current_pos}, "
                f"target={SP_LIN_BACK}"
            )
            last_print_time = elapsed

        if current_pos is not None:
            position_error = abs(current_pos - SP_LIN_BACK)

            if position_error <= POSITION_THRESHOLD:
                print(
                    f"[SP_LIN_2] spacer contracted: "
                    f"current={current_pos}, "
                    f"error={position_error}"
                )

                dxl.disable_torque(SP_LIN_ID)
                return True

        if elapsed >= timeout_sec:
            print(
                f"[SP_LIN_2][ERROR] contraction timeout: "
                f"current={current_pos}, "
                f"target={SP_LIN_BACK}, "
                f"timeout={timeout_sec}s"
            )

            # 永久に押し続けないようにトルクを切る
            try:
                dxl.disable_torque(SP_LIN_ID)
            except Exception as e:
                print(f"[SP_LIN_2][WARN] disable_torque failed: {e}")

            raise TimeoutError(
                "contract_sp_lin_2 timeout: "
                f"current={current_pos}, "
                f"target={SP_LIN_BACK}"
            )

        time.sleep(0.05)


def ungrasp_auto(
    dxl,
    timeout_sec=10.0,
    print_interval_sec=0.2
):
    """
    現在位置からRELEASE_GAINだけグリッパを開く。

    成功:
        Trueを返す。

    失敗:
        タイムアウト時にTimeoutErrorを送出する。
        安全のためグリッパのトルクを無効化する。
    """

    print("[UNGRASP] start")

    dxl.enable_torque(GRIPPER_ID)
    time.sleep(0.5)

    try:
        gripper_curr_pos = dxl.read_position(GRIPPER_ID)
    except Exception as e:
        dxl.disable_torque(GRIPPER_ID)
        raise RuntimeError(
            f"[UNGRASP] initial position read failed: {e}"
        ) from e

    if gripper_curr_pos is None:
        dxl.disable_torque(GRIPPER_ID)
        raise RuntimeError(
            "[UNGRASP] initial position is None"
        )

    gripper_ungrasp_pos = int(
        gripper_curr_pos + RELEASE_GAIN
    )

    print(
        f"[UNGRASP] current={gripper_curr_pos}, "
        f"release_gain={RELEASE_GAIN}, "
        f"target={gripper_ungrasp_pos}, "
        f"threshold={POSITION_THRESHOLD}"
    )

    try:
        dxl.write_position(
            GRIPPER_ID,
            gripper_ungrasp_pos
        )
    except Exception as e:
        dxl.disable_torque(GRIPPER_ID)
        raise RuntimeError(
            f"[UNGRASP] position command failed: {e}"
        ) from e

    start_time = time.monotonic()
    last_print_time = start_time

    while True:
        now = time.monotonic()
        elapsed = now - start_time

        try:
            current_pos = dxl.read_position(GRIPPER_ID)
        except Exception as e:
            print(
                f"[UNGRASP][WARN] "
                f"position read failed: {e}"
            )
            current_pos = None

        if current_pos is not None:
            position_error = abs(
                current_pos - gripper_ungrasp_pos
            )

            if now - last_print_time >= print_interval_sec:
                print(
                    f"[UNGRASP] "
                    f"elapsed={elapsed:.1f}s, "
                    f"current={current_pos}, "
                    f"target={gripper_ungrasp_pos}, "
                    f"error={position_error}"
                )
                last_print_time = now

            if position_error <= POSITION_THRESHOLD:
                print(
                    f"[UNGRASP] done: "
                    f"current={current_pos}, "
                    f"target={gripper_ungrasp_pos}, "
                    f"error={position_error}"
                )

                dxl.disable_torque(GRIPPER_ID)
                return True

        if elapsed >= timeout_sec:
            try:
                dxl.disable_torque(GRIPPER_ID)
            except Exception as e:
                print(
                    "[UNGRASP][WARN] "
                    f"disable_torque failed: {e}"
                )

            raise TimeoutError(
                "[UNGRASP] timeout: "
                f"current={current_pos}, "
                f"target={gripper_ungrasp_pos}, "
                f"timeout={timeout_sec}s"
            )

        time.sleep(0.05)


