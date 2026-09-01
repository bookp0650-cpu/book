from __future__ import annotations

from typing import Callable, TypeVar
import math
import time
from pathlib import Path

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

T = TypeVar("T")


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

    _dxl_retry(
        "write_position(expand_sp_lin)",
        lambda: dxl.write_position(
            SP_LIN_ID,
            SP_LIN_FRONT,
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
            SP_LIN_FRONT,
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
    """スペーサ直動軸をSP_LIN_BACKまで収縮させる。"""
    print(
        f"[SP_LIN BACK] start: target={SP_LIN_BACK}"
    )

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

    print(
        f"[SP_LIN BACK] before command={current_pos}"
    )

    _dxl_retry(
        "write_position(contract_sp_lin_2)",
        lambda: dxl.write_position(
            SP_LIN_ID,
            SP_LIN_BACK,
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
            SP_LIN_BACK,
            label="SP_LIN BACK",
            timeout_sec=timeout_sec,
            print_interval_sec=0.2,
        )

        print(
            f"[SP_LIN BACK] done: current={position}"
        )

        _safe_disable_torque(
            dxl,
            SP_LIN_ID,
            label="disable_torque(contract_sp_lin_2)",
        )
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
                GRIPPER_CLOSE,
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
