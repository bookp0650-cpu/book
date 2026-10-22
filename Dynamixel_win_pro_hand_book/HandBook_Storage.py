from __future__ import annotations

import csv
import math
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar
from collections import deque
from .dynamixel_cross_platform import Dynamixel
from .util.cfg_dict_loader import DynamixelCfg


# ============================================================
# Configuration
# ============================================================
PKG_DIR = Path(__file__).resolve().parent
cfg_path = PKG_DIR / "config" / "Dynamixel_config.yaml"

cfg = DynamixelCfg(str(cfg_path))

PORT = cfg.id.port
BAUDRATE = cfg.id.baudrate

GRIPPER_ID = cfg.id.gripper
SP_ROT_ID = cfg.id.spacer_rot
SP_LIN_ID = cfg.id.spacer_lin

GRIPPER_BL = cfg.pos.gripper.backlash
GRIPPER_CLOSE = cfg.pos.gripper.close
GRIPPER_FULL_OPEN = (
    cfg.pos.gripper.range
    + GRIPPER_CLOSE
    + GRIPPER_BL
)
GRIPPER_ROT_GAIN = cfg.cont_gain.gripper.pos_cont
GRIPPER_THETA_0 = cfg.pos.gripper.theta_zero
GRIPPER_R2S = cfg.pos.gripper.rad_to_step
GRIPPER_GR = cfg.pos.gripper.gear_ratio
RELEASE_GAIN = cfg.pos.gripper.release

SP_ROT_0 = cfg.pos.spacer_rot.zero
SP_ROT_R2S = cfg.pos.spacer_rot.rad_to_step
SP_ROT_GR = cfg.pos.spacer_rot.gear_ratio
SP_ROT_BL = cfg.pos.spacer_rot.backlash

SP_LIN_BACK = cfg.pos.spacer_lin.back
SP_LIN_KEEP = -1 * cfg.pos.spacer_lin.keep
SP_LIN_FRONT = SP_LIN_BACK - cfg.pos.spacer_lin.range
# Experimental partial expansion for image-left tilt = real-world RIGHT book
# tilt.  The measured maximum stroke is approximately 240 mm for 160000 step,
# so 20000 step is approximately 30 mm; this is not a precision calibration.
RIGHT_TILTED_SP_LIN_REDUCTION_STEPS = 0
# PWMストッパー検知前の相対収縮量。短い右傾斜伸長(130000 step)にも余裕を残す。
SP_LIN_RELATIVE_BACK_STEPS =  145000
SP_LIN_RIGHT_TILTED_FRONT = (
    SP_LIN_FRONT + RIGHT_TILTED_SP_LIN_REDUCTION_STEPS
)


VELOCITY_THRESHOLD = cfg.thresh.vel
POSITION_THRESHOLD = cfg.thresh.pos
SP_ROT_POSITION_THRESHOLD = cfg.thresh.spacer_rot_pos


# ============================================================
# Communication / monitoring settings
# ============================================================
DXL_RETRIES = 2
DXL_RETRY_WAIT_SEC = 0.05
DXL_MONITOR_INTERVAL_SEC = 0.05

# 速度0を1回読んだだけで停止扱いしない。
GRIPPER_STOP_REQUIRED_COUNT = 4
GRASP_STOP_REQUIRED_COUNT = 8

# 入庫解放では、既存の位置・速度閾値を同時に満たす状態を連続確認する。
# これらは新しい実機校正値ではなく既存値の再利用であり、運用前に要実機確認。
STORAGE_RELEASE_TIMEOUT_SEC = 5.0
STORAGE_RELEASE_SUCCESS_REQUIRED_COUNT = GRIPPER_STOP_REQUIRED_COUNT

T = TypeVar("T")
# Workerと共有するDynamixel通信ロック
_DXL_IO_LOCK = threading.RLock()
SP_LIN_ABORT_EVENT = threading.Event()

def _check_sp_lin_abort() -> None:
    if SP_LIN_ABORT_EVENT.is_set():
        raise RuntimeError("SP_LIN emergency stop requested")


def set_dxl_io_lock(lock: threading.RLock) -> None:
    """Worker側から共有ロックを設定する。"""
    global _DXL_IO_LOCK
    _DXL_IO_LOCK = lock


SP_LIN_MONITOR_DIR = PKG_DIR.parent / "captures" / "strage"
_SP_LIN_MONITOR_PATH: Path | None = None
_SP_LIN_MONITOR_PID: int | None = None
_SP_LIN_MONITOR_WARNED = False


def _log_sp_lin_position(
    event: str,
    position: int,
    target: int,
    *,
    note: str = "",
) -> None:
    """CSV出力を無効化する。"""
    return


# ============================================================
# Common communication helpers
# ============================================================
def _dxl_retry(
    label: str,
    operation: Callable[[], T],
    *,
    dxl: Dynamixel,
    retries: int = DXL_RETRIES,
    wait_sec: float = DXL_RETRY_WAIT_SEC,
) -> T:
    """
    Dynamixel I/Oを短時間だけ再試行する。

    1回目失敗時は同じDynamixelオブジェクトのPortHandlerを
    close -> openしてから再試行する。

    NOTE:
        main側が保持しているdxl参照を古くしないため、
        ここではDynamixelオブジェクトそのものは作り直さない。
    """
    if retries < 1:
        raise ValueError(f"retries must be >= 1: {retries}")

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with _DXL_IO_LOCK:
                result = operation()

            if attempt > 1:
                print(
                    f"[DXL RETRY] {label} recovered "
                    f"({attempt}/{retries})"
                )

            return result

        except Exception as exc:
            last_error = exc

            print(
                f"[DXL RETRY] {label} failed "
                f"({attempt}/{retries}): "
                f"{type(exc).__name__}: {exc}"
            )

            if attempt >= retries:
                break

            # 取得済みの同一オブジェクトを維持したまま
            # シリアルポートだけ再オープンする。
            try:
                print(
                    f"[DXL RETRY] reopening serial port: {PORT}"
                )
                with _DXL_IO_LOCK:
                    dxl.reopen_port(wait_sec=wait_sec)

            except Exception as reopen_exc:
                print(
                    "[DXL RETRY][WARN] reopen_port failed: "
                    f"{type(reopen_exc).__name__}: "
                    f"{reopen_exc}"
                )

            time.sleep(wait_sec)

    raise RuntimeError(
        f"{label} failed after {retries} attempts: {last_error}"
    ) from last_error


def _safe_disable_torque(
    dxl: Dynamixel,
    dxl_id: int,
    *,
    label: str,
) -> None:
    """cleanup用。Torque OFF失敗で元の例外を隠さない。"""
    try:
        _dxl_retry(
            label,
            lambda: dxl.disable_torque(dxl_id),
            dxl=dxl,
            retries=1,
        )
    except Exception as exc:
        print(
            f"[DXL CLEANUP][WARN] {label}: "
            f"{type(exc).__name__}: {exc}"
        )


def disable_all_torque(dxl: Dynamixel) -> None:
    """ID1/ID2/ID3のTorqueを可能な範囲でOFFにする。"""
    for name, dxl_id in (
        ("GRIPPER", GRIPPER_ID),
        ("SP_ROT", SP_ROT_ID),
        ("SP_LIN", SP_LIN_ID),
    ):
        _safe_disable_torque(
            dxl,
            dxl_id,
            label=f"disable_torque({name})",
        )


def _wait_position(
    dxl: Dynamixel,
    dxl_id: int,
    target: int,
    *,
    label: str,
    timeout_sec: float,
    threshold: int = POSITION_THRESHOLD,
    print_interval_sec: float | None = None,
) -> int:
    """位置がtarget±thresholdへ入るまで待つ。永久ループしない。"""
    start = time.monotonic()
    last_print = start

    while True:
        if dxl_id == SP_LIN_ID:
            _check_sp_lin_abort()
        now = time.monotonic()
        elapsed = now - start

        position = _dxl_retry(
            f"read_position({label})",
            lambda: dxl.read_position(dxl_id),
            dxl=dxl,
        )

        error = abs(int(position) - int(target))

        if error <= threshold:
            return int(position)

        if (
            print_interval_sec is not None
            and now - last_print >= print_interval_sec
        ):
            print(
                f"[{label}] elapsed={elapsed:.1f}s, "
                f"current={position}, target={target}, "
                f"error={error}"
            )
            last_print = now

        if elapsed >= timeout_sec:
            raise TimeoutError(
                f"{label} timeout: "
                f"current={position}, target={target}, "
                f"timeout={timeout_sec}s"
            )

        time.sleep(DXL_MONITOR_INTERVAL_SEC)


def _wait_velocity_stopped(
    dxl: Dynamixel,
    dxl_id: int,
    *,
    label: str,
    timeout_sec: float,
    required_count: int,
) -> int:
    """
    abs(velocity) < VELOCITY_THRESHOLD がrequired_count回連続するまで待つ。
    """
    start = time.monotonic()
    low_vel_count = 0
    last_velocity = 0

    while True:
        elapsed = time.monotonic() - start

        if elapsed >= timeout_sec:
            raise TimeoutError(
                f"{label} timeout: "
                f"last_velocity={last_velocity}, "
                f"timeout={timeout_sec}s"
            )

        last_velocity = int(
            _dxl_retry(
                f"read_velocity({label})",
                lambda: dxl.read_velocity(dxl_id),
                dxl=dxl,
            )
        )

        if abs(last_velocity) < VELOCITY_THRESHOLD:
            low_vel_count += 1
        else:
            low_vel_count = 0

        if low_vel_count >= required_count:
            return last_velocity

        time.sleep(DXL_MONITOR_INTERVAL_SEC)


# ============================================================
# Initialization
# ============================================================
def init_dynamixels() -> Dynamixel:
    """
    Dynamixel通信を初期化して操作オブジェクトを返す。

    改良点:
    - GRIPPER / SP_ROT / SP_LIN の3台すべてを初期化
    - Operating Mode変更前に必ずTorque OFF
    - I/O失敗時は短時間reopen + retry
    - 失敗時はTorque OFFとclose_portを試す
    """
    dxl = Dynamixel(
        port=PORT,
        baudrate=BAUDRATE,
    )

    try:
        time.sleep(0.2)

        targets = (
            ("GRIPPER", GRIPPER_ID),
            ("SP_ROT", SP_ROT_ID),
            ("SP_LIN", SP_LIN_ID),
        )

        # 接続確認
        for name, dxl_id in targets:
            model = _dxl_retry(
                f"ping({name})",
                lambda dxl_id=dxl_id: dxl.ping(dxl_id),
                dxl=dxl,
            )
            print(
                f"[DXL INIT] {name}(ID={dxl_id}) "
                f"ping OK, model={model}"
            )
            time.sleep(0.03)

        # Operating Mode変更前に全台Torque OFF
        for name, dxl_id in targets:
            _dxl_retry(
                f"disable_torque({name}, init)",
                lambda dxl_id=dxl_id: dxl.disable_torque(dxl_id),
                dxl=dxl,
            )
            time.sleep(0.03)

        # 3台すべてExtended Position Modeへ
        for name, dxl_id in targets:
            _dxl_retry(
                f"set_mode_ex_position({name})",
                lambda dxl_id=dxl_id: dxl.set_mode_ex_position(dxl_id),
                dxl=dxl,
            )

            position = _dxl_retry(
                f"read_position({name}, init)",
                lambda dxl_id=dxl_id: dxl.read_position(dxl_id),
                dxl=dxl,
            )

            print(
                f"[DXL INIT] {name}(ID={dxl_id}) "
                f"Position={position}"
            )
            time.sleep(0.03)

        print("---------------------------------")
        print("   Dynamixel READY TO MOVE")
        print("   3 axes initialized safely")
        print("---------------------------------")

        return dxl

    except Exception:
        disable_all_torque(dxl)

        try:
            dxl.close_port()
        except Exception as exc:
            print(
                "[DXL CLEANUP][WARN] close_port failed: "
                f"{type(exc).__name__}: {exc}"
            )

        raise

# ============================================================
# SP_LIN stop-detection trial (ID 3 only)
# ============================================================
def home_sp_lin_stop_test(
    dxl: Dynamixel,
    *,
    pwm_cap: int = 885,
    profile_velocity: int = 500,
    max_travel_steps: int = 1000000,
    timeout_sec: float = 100.0,
) -> dict:
    """
    SP_LINを+方向へ1回の位置指令で移動させ、
    実測速度と位置停滞を監視する。

    短距離検証用。ストッパーの原点確定は行わない。
    """

    axis = SP_LIN_ID
    old_pwm = None
    old_profile = None
    torque_enabled = False

    # 停止指令に失敗した場合、その情報を保持する
    torque_off_error = None

    try:
        with _DXL_IO_LOCK:
            _check_sp_lin_abort()
            dxl.disable_torque(axis)
            dxl.set_mode_ex_position(axis)

            old_pwm = int(dxl.read_pwm_limit(axis))
            old_profile = int(dxl.read_profile_velocity(axis))

            if not (1 <= pwm_cap <= old_pwm):
                raise ValueError("Invalid PWM limit")

            if profile_velocity < 0:
                raise ValueError("Invalid profile velocity")

            if not (0 < max_travel_steps <= 1000000):
                raise ValueError("Travel limit must be 1..1000000")

            if old_pwm != pwm_cap:
                dxl.write_pwm_limit(axis, pwm_cap)

            if old_profile != profile_velocity:
                dxl.write_profile_velocity(axis, profile_velocity)

            initial = int(dxl.read_position(axis))
            target = initial + max_travel_steps

            _check_sp_lin_abort()
            dxl.write_position(axis, initial)
            dxl.enable_torque(axis)
            torque_enabled = True

            _check_sp_lin_abort()
            dxl.write_position(axis, target)

        print(
            f"[SP_LIN TEST] start={initial}, target={target}",
            flush=True
        )

        started = time.monotonic()
        pwm_history = deque(maxlen=5)

        while True:
            _check_sp_lin_abort()
            elapsed = time.monotonic() - started

            if elapsed >= timeout_sec:
                raise TimeoutError("SP_LIN test timeout")

            with _DXL_IO_LOCK:
                pos = int(dxl.read_position(axis))
                vel = int(dxl.read_velocity(axis))
                pwm = int(dxl.read_present_pwm(axis))

            pwm_history.append(abs(pwm))

            if len(pwm_history) == 5:
                pwm_avg = sum(pwm_history) / 5

                # PWM平均の確認
                if pwm_avg >= 740:
                    print(
                        f"[SP_LIN PWM STOP REQUEST] "
                        f"pos={pos}, vel={vel}, "
                        f"pwm={pwm}, pwm_avg={pwm_avg:.1f}",
                        flush=True
                    )

                    return {
                        "position": pos,
                        "pwm": pwm,
                        "pwm_avg": pwm_avg,
                        "velocity": vel,
                        "reason": "high_pwm_detected",
                        "retraction_confirmed": False,
                    }

            print(
                f"[SP_LIN MONITOR] "
                f"pos={pos}, vel={vel}, pwm={pwm}, "
                f"remaining={target - pos}",
                flush=True
            )


            if pos >= target:
                raise RuntimeError(
                    "SP_LIN reached additional travel limit "
                    "without PWM stop detection: "
                    f"pos={pos}, target={target}, pwm={pwm}"
                )

            time.sleep(DXL_MONITOR_INTERVAL_SEC)

    finally:
        # 正常・異常に関係なく停止指令を送る
        try:
            with _DXL_IO_LOCK:
                dxl.disable_torque(axis)

                torque_state = int(dxl.read_torque_enable(axis))

                if torque_state != 0:
                    raise RuntimeError(
                        "SP_LIN torque OFF verification failed: "
                        f"Torque Enable={torque_state}"
                    )

            print(
                "[SP_LIN STOP] torque OFF verified (Torque Enable=0)",
                flush=True
            )

        except Exception as exc:
            torque_off_error = exc

            print(
                f"[SP_LIN][DANGER] torque OFF failed: {exc}",
                flush=True
            )

        if old_pwm is not None and old_pwm != pwm_cap:
            try:
                with _DXL_IO_LOCK:
                    dxl.write_pwm_limit(axis, old_pwm)
            except Exception as exc:
                print(f"[SP_LIN][WARN] PWM restore failed: {exc}")

        if old_profile is not None and old_profile != profile_velocity:
            try:
                with _DXL_IO_LOCK:
                    dxl.write_profile_velocity(axis, old_profile)
            except Exception as exc:
                print(f"[SP_LIN][WARN] velocity restore failed: {exc}")
        # 停止命令が失敗していたら正常終了させない
        if torque_off_error is not None:
            raise RuntimeError(
                "SP_LIN torque OFF command failed. "
                "Motor stop is not confirmed."
            ) from torque_off_error

# ============================================================
# Spacer linear axis
# ============================================================
def expand_sp_lin(
    dxl: Dynamixel,
    asynchronous: bool = False,
    timeout_sec: float = 15.0,
):
    """スペーサ直動軸をSP_LIN_FRONTまで伸ばす。"""
    print(
        f"[SP_LIN EXPAND] target={SP_LIN_FRONT}"
    )

    _dxl_retry(
        "enable_torque(expand_sp_lin)",
        lambda: dxl.enable_torque(SP_LIN_ID),
        dxl=dxl,
    )

    time.sleep(0.1)

    # 現在位置を取得
    current_pos = int(_dxl_retry(
        "read_position(expand_sp_lin_before)",
        lambda: dxl.read_position(SP_LIN_ID),
        dxl=dxl,
    ))

    # 現在位置から相対的に150000 step伸長（マイナス方向）
    target_pos = current_pos - cfg.pos.spacer_lin.range

    print(
        f"[SP_LIN RELATIVE EXPAND] "
        f"current={current_pos}, "
        f"delta=-{cfg.pos.spacer_lin.range}, target={target_pos}"
    )

    _dxl_retry(
        "write_position(expand_sp_lin)",
        lambda: dxl.write_position(
            SP_LIN_ID,
            target_pos,
        ),
        dxl=dxl,
    )

    if asynchronous:
        print("[SP_LIN EXPAND] command sent asynchronously")
        return True

    try:
        position = _wait_position(
            dxl,
            SP_LIN_ID,
            target_pos,
            label="SP_LIN EXPAND",
            timeout_sec=timeout_sec,
            print_interval_sec=0.5,
        )

        print(
            f"[SP_LIN EXPAND] done: current={position}"
        )

        _safe_disable_torque(
            dxl,
            SP_LIN_ID,
            label="disable_torque(expand_sp_lin)",
        )
        return True

    except Exception:
        _safe_disable_torque(
            dxl,
            SP_LIN_ID,
            label="disable_torque(expand_sp_lin_error)",
        )
        raise


def expand_sp_lin_right_tilted(
    dxl: Dynamixel,
    asynchronous: bool = False,
    timeout_sec: float = 15.0,
):
    """右傾斜用に現在位置から相対的に短く伸長する。"""

    # YAMLから伸長量を取得
    expand_steps = int(cfg.pos.spacer_lin.range)
    reduction_steps = RIGHT_TILTED_SP_LIN_REDUCTION_STEPS

    relative_steps = expand_steps - reduction_steps

    if relative_steps <= 0:
        raise ValueError(
            f"Invalid right-tilted extension: "
            f"range={expand_steps}, reduction={reduction_steps}"
        )

    # 現在位置を取得
    current_pos = int(_dxl_retry(
        "read_position(expand_sp_lin_right_tilted_before)",
        lambda: dxl.read_position(SP_LIN_ID),
        dxl=dxl,
    ))

    # 現在位置から相対的に伸長
    target_pos = current_pos - relative_steps

    print(
        "[SP_LIN RIGHT-TILTED RELATIVE EXPAND] "
        f"current={current_pos}, "
        f"delta=-{relative_steps}, "
        f"target={target_pos}"
    )

    _dxl_retry(
        "enable_torque(expand_sp_lin_right_tilted)",
        lambda: dxl.enable_torque(SP_LIN_ID),
        dxl=dxl,
    )

    time.sleep(0.1)

    _dxl_retry(
        "write_position(expand_sp_lin_right_tilted)",
        lambda: dxl.write_position(
            SP_LIN_ID,
            target_pos,
        ),
        dxl=dxl,
    )

    if asynchronous:
        print(
            "[SP_LIN RIGHT-TILTED PARTIAL EXPAND] "
            "command sent asynchronously"
        )
        return True

    try:
        position = _wait_position(
            dxl,
            SP_LIN_ID,
            target_pos,
            label="SP_LIN RIGHT-TILTED PARTIAL EXPAND",
            timeout_sec=timeout_sec,
            print_interval_sec=0.5,
        )

        print(
            "[SP_LIN RIGHT-TILTED PARTIAL EXPAND] "
            f"done: current={position}"
        )

        _safe_disable_torque(
            dxl,
            SP_LIN_ID,
            label="disable_torque(expand_sp_lin_right_tilted)",
        )
        return True

    except Exception:
        _safe_disable_torque(
            dxl,
            SP_LIN_ID,
            label="disable_torque(expand_sp_lin_right_tilted_error)",
        )
        raise


def contract_sp_lin_1(
    dxl: Dynamixel,
    asynchronous: bool = False,
    timeout_sec: float = 15.0,
):
    """スペーサ直動軸をSP_LIN_KEEPまで収縮させる。"""
    print(
        f"[SP_LIN KEEP] target={SP_LIN_KEEP}"
    )

    _dxl_retry(
        "enable_torque(contract_sp_lin_1)",
        lambda: dxl.enable_torque(SP_LIN_ID),
        dxl=dxl,
    )

    time.sleep(0.1)

    _dxl_retry(
        "write_position(contract_sp_lin_1)",
        lambda: dxl.write_position(
            SP_LIN_ID,
            SP_LIN_KEEP,
        ),
        dxl=dxl,
    )

    if asynchronous:
        print("[SP_LIN KEEP] command sent asynchronously")
        return True

    try:
        position = _wait_position(
            dxl,
            SP_LIN_ID,
            SP_LIN_KEEP,
            label="SP_LIN KEEP",
            timeout_sec=timeout_sec,
            print_interval_sec=0.5,
        )

        print(
            f"[SP_LIN KEEP] done: current={position}"
        )

        _safe_disable_torque(
            dxl,
            SP_LIN_ID,
            label="disable_torque(contract_sp_lin_1)",
        )
        return True

    except Exception:
        _safe_disable_torque(
            dxl,
            SP_LIN_ID,
            label="disable_torque(contract_sp_lin_1_error)",
        )
        raise


def contract_sp_lin_2(
    dxl: Dynamixel,
    asynchronous: bool = False,
    timeout_sec: float = 15.0,
):
    """現在位置から相対収縮し、その後PWMによるストッパー検知で終了する。"""
    print(f"[SP_LIN BACK] start: relative=+{SP_LIN_RELATIVE_BACK_STEPS} steps")

    _check_sp_lin_abort()
    _dxl_retry(
        "enable_torque(contract_sp_lin_2)",
        lambda: dxl.enable_torque(SP_LIN_ID),
        dxl=dxl,
    )

    time.sleep(0.1)

    current_pos = _dxl_retry(
        "read_position(contract_sp_lin_2_before)",
        lambda: dxl.read_position(SP_LIN_ID),
        dxl=dxl,
    )

    # ここでの位置は固定の原点ではなく、毎サイクル測定した現在位置。
    target_pos = int(current_pos) + SP_LIN_RELATIVE_BACK_STEPS
    print(
        f"[SP_LIN RELATIVE BACK] current={current_pos}, "
        f"delta=+{SP_LIN_RELATIVE_BACK_STEPS}, target={target_pos}"
    )

    _log_sp_lin_position(
        "back_before_command",
        current_pos,
        target_pos,
        note="relative target based on present position",
    )

    print(
        f"[SP_LIN BACK] before command={current_pos}"
    )

    _check_sp_lin_abort()
    _dxl_retry(
        "write_position(contract_sp_lin_2)",
        lambda: dxl.write_position(
            SP_LIN_ID,
            target_pos,
        ),
        dxl=dxl,
    )

    if asynchronous:
        print("[SP_LIN BACK] command sent asynchronously")
        return True

    try:
        position = _wait_position(
            dxl,
            SP_LIN_ID,
            target_pos,
            label="SP_LIN BACK",
            timeout_sec=timeout_sec,
            print_interval_sec=0.2,
        )

        _log_sp_lin_position(
            "back_reached_torque_on",
            position,
            target_pos,
            note="relative target reached",
        )

        print(
            f"[SP_LIN BACK] done: current={position}"
        )

        _check_sp_lin_abort()
        print("[SP_LIN BACK] starting additional PWM detection")

        result = home_sp_lin_stop_test(
            dxl,
            pwm_cap=885,
            profile_velocity=500,
            max_travel_steps=1000000,
            timeout_sec=20.0,
        )

        if result.get("reason") != "high_pwm_detected":
            raise RuntimeError(
                f"SP_LIN PWM detection failed: {result}"
            )

        print("[SP_LIN BACK] PWM stop confirmed:", result)
        return True

    except Exception:
        _safe_disable_torque(
            dxl,
            SP_LIN_ID,
            label="disable_torque(contract_sp_lin_2_error)",
        )
        raise


# ============================================================
# Gripper
# ============================================================
def open_for_storage_release_full(
    dxl: Dynamixel,
    timeout_sec: float = STORAGE_RELEASE_TIMEOUT_SEC,
    position_tolerance: int = POSITION_THRESHOLD,
    velocity_threshold: int = VELOCITY_THRESHOLD,
    required_count: int = STORAGE_RELEASE_SUCCESS_REQUIRED_COUNT,
) -> bool:
    """入庫した書籍を離すため、全開位置到達と停止を同期確認する。"""
    timeout = float(timeout_sec)
    position_limit = int(position_tolerance)
    velocity_limit = int(velocity_threshold)
    consecutive_required = int(required_count)

    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError(
            "storage release timeout_sec must be finite and positive: "
            f"{timeout_sec}"
        )
    if position_limit < 0:
        raise ValueError(
            "storage release position_tolerance must be nonnegative: "
            f"{position_tolerance}"
        )
    if velocity_limit < 0:
        raise ValueError(
            "storage release velocity_threshold must be nonnegative: "
            f"{velocity_threshold}"
        )
    if consecutive_required <= 0:
        raise ValueError(
            "storage release required_count must be positive: "
            f"{required_count}"
        )

    print(
        "[STORAGE RELEASE] gripper full-open confirmation start: "
        f"target={GRIPPER_FULL_OPEN}, "
        f"position_tolerance={position_limit}, "
        f"velocity_threshold={velocity_limit}, "
        f"required_count={consecutive_required}, "
        f"timeout={timeout:.2f}s"
    )

    _dxl_retry(
        "enable_torque(open_for_storage_release)",
        lambda: dxl.enable_torque(GRIPPER_ID),
        dxl=dxl,
    )
    _dxl_retry(
        "write_position(open_for_storage_release)",
        lambda: dxl.write_position(
            GRIPPER_ID,
            GRIPPER_FULL_OPEN,
        ),
        dxl=dxl,
    )

    deadline = time.monotonic() + timeout
    success_count = 0
    last_position: int | None = None
    last_velocity: int | None = None
    last_position_error: int | None = None

    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "storage gripper release timeout: "
                f"target={GRIPPER_FULL_OPEN}, "
                f"position={last_position}, "
                f"position_error={last_position_error}, "
                f"velocity={last_velocity}, "
                f"success_count={success_count}/{consecutive_required}, "
                f"timeout={timeout:.2f}s"
            )

        last_position = int(
            _dxl_retry(
                "read_position(open_for_storage_release)",
                lambda: dxl.read_position(GRIPPER_ID),
                dxl=dxl,
            )
        )
        last_velocity = int(
            _dxl_retry(
                "read_velocity(open_for_storage_release)",
                lambda: dxl.read_velocity(GRIPPER_ID),
                dxl=dxl,
            )
        )
        last_position_error = abs(last_position - GRIPPER_FULL_OPEN)

        position_ok = last_position_error <= position_limit
        velocity_ok = abs(last_velocity) <= velocity_limit
        if position_ok and velocity_ok:
            success_count += 1
        else:
            success_count = 0

        if success_count >= consecutive_required:
            print(
                "[STORAGE RELEASE] gripper release confirmed: "
                f"position={last_position}, "
                f"position_error={last_position_error}, "
                f"velocity={last_velocity}, "
                f"success_count={success_count}"
            )
            print(
                "[STORAGE RELEASE] gripper torque remains enabled "
                "through xArm retreat"
            )
            return True

        time.sleep(DXL_MONITOR_INTERVAL_SEC)



# Default for the existing worker command's storage release.
STORAGE_RELEASE_PARTIAL_OFFSET_STEPS = 5000


def open_for_storage_release(
    dxl: Dynamixel,
    timeout_sec: float = STORAGE_RELEASE_TIMEOUT_SEC,
    position_tolerance: int = POSITION_THRESHOLD,
    velocity_threshold: int = VELOCITY_THRESHOLD,
    required_count: int = STORAGE_RELEASE_SUCCESS_REQUIRED_COUNT,
) -> bool:
    """Worker-compatible release: use measured position +2000, not full opening."""
    return open_for_storage_release_partial(
        dxl,
        offset_steps=STORAGE_RELEASE_PARTIAL_OFFSET_STEPS,
        timeout_sec=timeout_sec,
        position_tolerance=position_tolerance,
        velocity_threshold=velocity_threshold,
        required_count=required_count,
    )


def open_for_storage_release_partial(
    dxl: Dynamixel,
    offset_steps: int = 5000,
    timeout_sec: float = STORAGE_RELEASE_TIMEOUT_SEC,
    position_tolerance: int = POSITION_THRESHOLD,
    velocity_threshold: int = VELOCITY_THRESHOLD,
    required_count: int = STORAGE_RELEASE_SUCCESS_REQUIRED_COUNT,
) -> bool:
    """Open from measured gripper position by offset_steps (not full-open).

    Must confirm the book is free before commanding the robot retreat.
    Torque intentionally stays ON after successful positioning.
    """
    if not isinstance(offset_steps, int) or isinstance(offset_steps, bool) or offset_steps <= 0:
        raise ValueError(f"offset_steps must be a positive integer: {offset_steps!r}")
    timeout = float(timeout_sec)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"invalid release timeout: {timeout_sec}")
    if position_tolerance < 0 or velocity_threshold < 0 or required_count <= 0:
        raise ValueError("invalid partial release confirmation thresholds")

    _dxl_retry(
        "enable_torque(storage_release_partial)",
        lambda: dxl.enable_torque(GRIPPER_ID),
        dxl=dxl,
    )
    current_pos = int(_dxl_retry(
        "read_position(storage_release_partial_before)",
        lambda: dxl.read_position(GRIPPER_ID),
        dxl=dxl,
    ))
    if current_pos > GRIPPER_FULL_OPEN:
        raise RuntimeError(
            f"gripper measured position exceeds configured full-open: "
            f"current={current_pos}, full_open={GRIPPER_FULL_OPEN}"
        )
    target_pos = min(current_pos + offset_steps, int(GRIPPER_FULL_OPEN))
    if target_pos <= current_pos:
        raise RuntimeError(
            f"no available gripper opening stroke: current={current_pos}, "
            f"full_open={GRIPPER_FULL_OPEN}"
        )
    print(
        f"[STORAGE PARTIAL RELEASE] current={current_pos}, "
        f"offset_requested={offset_steps}, target={target_pos}, "
        f"actual_command_delta={target_pos - current_pos}"
    )
    _dxl_retry(
        "write_position(storage_release_partial)",
        lambda: dxl.write_position(GRIPPER_ID, target_pos),
        dxl=dxl,
    )

    deadline = time.monotonic() + timeout
    successes = 0
    last_pos = current_pos
    last_vel = 0
    while time.monotonic() < deadline:
        last_pos = int(_dxl_retry(
            "read_position(storage_release_partial)",
            lambda: dxl.read_position(GRIPPER_ID),
            dxl=dxl,
        ))
        last_vel = int(_dxl_retry(
            "read_velocity(storage_release_partial)",
            lambda: dxl.read_velocity(GRIPPER_ID),
            dxl=dxl,
        ))
        if (abs(last_pos - target_pos) <= int(position_tolerance)
                and abs(last_vel) <= int(velocity_threshold)):
            successes += 1
        else:
            successes = 0
        if successes >= int(required_count):
            print(
                f"[STORAGE PARTIAL RELEASE] target reached: "
                f"position={last_pos}, target={target_pos}, velocity={last_vel}; "
                "torque remains enabled"
            )
            return True
        time.sleep(DXL_MONITOR_INTERVAL_SEC)
    raise TimeoutError(
        f"storage partial release timeout: current={last_pos}, target={target_pos}, "
        f"velocity={last_vel}, confirmations={successes}/{required_count}"
    )

def open_until_full(
    dxl: Dynamixel,
    asynchronous: bool = False,
    timeout_sec: float = 5.0,
):
    """
    グリッパを全開にする。

    同期時:
      速度停止を複数回連続確認してから終了しTorque OFF。
    非同期時:
      Goal Position送信後すぐ戻り、TorqueはONのまま。
    """
    print("gripper open until full")

    _dxl_retry(
        "enable_torque(open_until_full)",
        lambda: dxl.enable_torque(GRIPPER_ID),
        dxl=dxl,
    )

    _dxl_retry(
        "write_position(open_until_full)",
        lambda: dxl.write_position(
            GRIPPER_ID,
            GRIPPER_FULL_OPEN,
        ),
        dxl=dxl,
    )

    if asynchronous:
        print(
            "[GRIPPER OPEN] asynchronous command sent; "
            "torque remains enabled"
        )
        return True

    # 指令直後の速度0誤判定を避ける
    time.sleep(0.15)

    try:
        last_velocity = _wait_velocity_stopped(
            dxl,
            GRIPPER_ID,
            label="open_until_full",
            timeout_sec=timeout_sec,
            required_count=GRIPPER_STOP_REQUIRED_COUNT,
        )

        final_position = _dxl_retry(
            "read_position(open_until_full_final)",
            lambda: dxl.read_position(GRIPPER_ID),
            dxl=dxl,
        )

        print(
            "[GRIPPER OPEN] done: "
            f"position={final_position}, "
            f"velocity={last_velocity}"
        )

        _safe_disable_torque(
            dxl,
            GRIPPER_ID,
            label="disable_torque(open_until_full)",
        )
        return True

    except Exception:
        _safe_disable_torque(
            dxl,
            GRIPPER_ID,
            label="disable_torque(open_until_full_error)",
        )
        raise


def grasp(
    dxl: Dynamixel,
    timeout_sec: float = 3.0,
    keep_torque: bool = True,
):
    """
    グリッパを閉じて把持する。

    改良点:
    - 指令直後の速度0誤判定対策
    - 監視中はread_velocityだけにして通信量を削減
    - 低速度を8回連続確認して把持判定
    - keep_torque=Trueなら現在位置付近を保持してTorque ON維持
    - keep_torque=Falseなら把持後Torque OFF
    """
    _dxl_retry(
        "enable_torque(grasp)",
        lambda: dxl.enable_torque(GRIPPER_ID),
        dxl=dxl,
    )

    print("grasping object until detection")

    try:
        _dxl_retry(
            "write_position(grasp)",
            lambda: dxl.write_position(
                GRIPPER_ID,
                0,
            ),
            dxl=dxl,
        )

        print("grasping")

        # 指令直後の速度0を把持完了と誤判定しない
        time.sleep(0.4)

        last_velocity = _wait_velocity_stopped(
            dxl,
            GRIPPER_ID,
            label="grasp",
            timeout_sec=timeout_sec,
            required_count=GRASP_STOP_REQUIRED_COUNT,
        )

        # 停止判定後だけ位置を1回読む
        curr_pos = int(
            _dxl_retry(
                "read_position(grasp_final)",
                lambda: dxl.read_position(GRIPPER_ID),
                dxl=dxl,
            )
        )

        if abs(curr_pos - GRIPPER_CLOSE) < POSITION_THRESHOLD:
            reason = "full_close"
            print("gripper close")
        else:
            reason = "object_detected"
            print("grasp detected before full close")

        if keep_torque:
            # 既存入庫コードの保持方法を維持
            hold_pos = curr_pos - 5

            _dxl_retry(
                "write_position(grasp_hold)",
                lambda: dxl.write_position(
                    GRIPPER_ID,
                    hold_pos,
                ),
                dxl=dxl,
            )

            print(
                f"[GRASP] hold position set to {hold_pos}; "
                "torque remains enabled"
            )

        else:
            _safe_disable_torque(
                dxl,
                GRIPPER_ID,
                label="disable_torque(grasp_complete)",
            )

        result = {
            "position": curr_pos,
            "velocity": int(last_velocity),
            "reason": reason,
            "keep_torque": bool(keep_torque),
        }

        print(
            "[GRASP RESULT] "
            f"position={result['position']}, "
            f"velocity={result['velocity']}, "
            f"reason={result['reason']}, "
            f"keep_torque={result['keep_torque']}"
        )

        return result

    except Exception:
        # 通信異常時に暴走保持しない
        _safe_disable_torque(
            dxl,
            GRIPPER_ID,
            label="disable_torque(grasp_error)",
        )
        raise


def ungrasp_auto(
    dxl: Dynamixel,
    timeout_sec: float = 10.0,
    print_interval_sec: float = 0.2,
):
    """現在位置からRELEASE_GAINだけグリッパを開く。"""
    print("[UNGRASP] start")

    _dxl_retry(
        "enable_torque(ungrasp)",
        lambda: dxl.enable_torque(GRIPPER_ID),
        dxl=dxl,
    )

    time.sleep(0.2)

    try:
        gripper_curr_pos = int(
            _dxl_retry(
                "read_position(ungrasp_before)",
                lambda: dxl.read_position(GRIPPER_ID),
                dxl=dxl,
            )
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

        _dxl_retry(
            "write_position(ungrasp)",
            lambda: dxl.write_position(
                GRIPPER_ID,
                gripper_ungrasp_pos,
            ),
            dxl=dxl,
        )

        final_position = _wait_position(
            dxl,
            GRIPPER_ID,
            gripper_ungrasp_pos,
            label="UNGRASP",
            timeout_sec=timeout_sec,
            threshold=25,
            print_interval_sec=print_interval_sec,
        )

        print(
            f"[UNGRASP] done: current={final_position}"
        )

        _safe_disable_torque(
            dxl,
            GRIPPER_ID,
            label="disable_torque(ungrasp_complete)",
        )

        return True

    except Exception:
        _safe_disable_torque(
            dxl,
            GRIPPER_ID,
            label="disable_torque(ungrasp_error)",
        )
        raise


# ============================================================
# Spacer rotation
# ============================================================
def rotate_spacer(
    dxl: Dynamixel,
    theta_deg: float = 90.0,
    timeout_sec: float = 10.0,
):
    """スペーサ回転軸を指定角度へ回す。"""
    des_pos = (
        SP_ROT_0
        - int(
            (theta_deg * math.pi / 180.0)
            * SP_ROT_R2S
            * SP_ROT_GR
            + SP_ROT_BL
        )
    )

    print(
        f"[SP_ROT] target angle={theta_deg:.2f}deg, "
        f"target position={des_pos}, "
        f"threshold={SP_ROT_POSITION_THRESHOLD}"
    )

    _dxl_retry(
        "enable_torque(rotate_spacer)",
        lambda: dxl.enable_torque(SP_ROT_ID),
        dxl=dxl,
    )

    try:
        _dxl_retry(
            "write_position(rotate_spacer)",
            lambda: dxl.write_position(
                SP_ROT_ID,
                des_pos,
            ),
            dxl=dxl,
        )

        position = _wait_position(
            dxl,
            SP_ROT_ID,
            des_pos,
            label="SP_ROT",
            timeout_sec=timeout_sec,
            threshold=SP_ROT_POSITION_THRESHOLD,
            print_interval_sec=0.5,
        )

        print(
            f"spacer rotated to {theta_deg:.2f} deg "
            f"(current={position}, target={des_pos}, "
            f"error={abs(position - des_pos)}, "
            f"threshold={SP_ROT_POSITION_THRESHOLD})"
        )

        _safe_disable_torque(
            dxl,
            SP_ROT_ID,
            label="disable_torque(rotate_spacer)",
        )
        return True

    except Exception:
        _safe_disable_torque(
            dxl,
            SP_ROT_ID,
            label="disable_torque(rotate_spacer_error)",
        )
        raise


def reset_rot(
    dxl: Dynamixel,
    asynchronous: bool = False,
    timeout_sec: float = 10.0,
):
    """スペーサ回転軸をゼロ位置へ戻す。"""
    print(
        f"[SP_ROT RESET] target={SP_ROT_0}, "
        f"threshold={SP_ROT_POSITION_THRESHOLD}"
    )
    _dxl_retry(
        "enable_torque(reset_rot)",
        lambda: dxl.enable_torque(SP_ROT_ID),
        dxl=dxl,
    )

    _dxl_retry(
        "write_position(reset_rot)",
        lambda: dxl.write_position(
            SP_ROT_ID,
            SP_ROT_0,
        ),
        dxl=dxl,
    )

    if asynchronous:
        print(
            "[SP_ROT RESET] asynchronous command sent; "
            "torque remains enabled"
        )
        return True

    try:
        position = _wait_position(
            dxl,
            SP_ROT_ID,
            SP_ROT_0,
            label="SP_ROT RESET",
            timeout_sec=timeout_sec,
            threshold=SP_ROT_POSITION_THRESHOLD,
            print_interval_sec=0.5,
        )

        print(
            f"spacer rotation reset to zero "
            f"(current={position}, target={SP_ROT_0}, "
            f"error={abs(position - SP_ROT_0)}, "
            f"threshold={SP_ROT_POSITION_THRESHOLD})"
        )

        _safe_disable_torque(
            dxl,
            SP_ROT_ID,
            label="disable_torque(reset_rot)",
        )
        return True

    except Exception:
        _safe_disable_torque(
            dxl,
            SP_ROT_ID,
            label="disable_torque(reset_rot_error)",
        )
        raise
