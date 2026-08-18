"""
Dynamixel hand control for book retrieval.

主な機能
--------
- 一時的なDynamixel通信失敗の再試行
- グリッパの幅指定開閉、把持、全開
- 通常動作では追加診断readを行わず、Dynamixel通信量を最小化
- 通信失敗時にもログ処理で元の例外を隠さない

前提
----
通常制御では位置・速度・Torque・Goal Positionのみを使用します。
"""

from __future__ import annotations

import csv
import fcntl
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

from . import kbhit_lin as kbhit
from .dynamixel_cross_platform import Dynamixel
from .util.cfg_dict_loader import DynamixelCfg


# =========================================================
# パス・設定
# =========================================================
PKG_DIR = Path(__file__).resolve().parent
CFG_PATH = PKG_DIR / "config" / "Dynamixel_config.yaml"

LOG_SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
VOLTAGE_LOG_DIR = PKG_DIR.parent / "captures" / "diagnostics"
VOLTAGE_LOG_PATH = (
    VOLTAGE_LOG_DIR
    / f"dynamixel_voltage_{LOG_SESSION_ID}.csv"
)

cfg = DynamixelCfg(str(CFG_PATH))

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
GRIPPER_CALIB_A = cfg.pos.gripper.calib_width_a
GRIPPER_CALIB_B = cfg.pos.gripper.calib_width_b

VELOCITY_THRESHOLD = cfg.thresh.vel
POSITION_THRESHOLD = cfg.thresh.pos

# 通常動作中のDynamixel通信を最小化する。
# grasp/open_until_full の監視は1回のreadだけを50 ms間隔で行う。
DXL_MONITOR_INTERVAL_SEC = 0.05

# 電圧診断は通常動作では無効。
# Trueに戻した場合のみID1/ID2/ID3の電圧を読む。
ENABLE_DXL_VOLTAGE_DIAGNOSTICS = False

# =========================================================
# 通信エラー時の再試行
#
# 統合運用ではプロセス再起動による復旧を使用するため、
# 同一プロセス・同一ポート内では再試行しない。
# =========================================================
DXL_RETRIES = 1
DXL_RETRY_WAIT_SEC = 0.05

# 初回に正常取得できた電圧を、IDごとの基準電圧として保持する。
_VOLTAGE_BASELINE_BY_ID: dict[int, float] = {}

# 基準電圧からこの値以上低下した場合に警告を表示する。
VOLTAGE_DROP_WARNING_V = 0.5

T = TypeVar("T")


# =========================================================
# 通信再試行
# =========================================================
def _dxl_retry(
    label: str,
    operation: Callable[[], T],
    dxl: Dynamixel | None = None,
    retries: int = DXL_RETRIES,
    wait_sec: float = DXL_RETRY_WAIT_SEC,
) -> T:
    """
    Dynamixel操作を再試行する。

    通信失敗時、dxl が指定されていれば
    PortHandlerをclose -> openしてから再試行する。
    """

    if retries < 1:
        raise ValueError(
            f"retries must be >= 1: retries={retries}"
        )

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            result = operation()

            if attempt > 1:
                print(
                    f"[DXL RETRY] {label} recovered: "
                    f"attempt={attempt}/{retries}"
                )

            return result

        except Exception as exc:
            last_error = exc

            print(
                f"[DXL RETRY] {label} failed: "
                f"attempt={attempt}/{retries}, "
                f"error={type(exc).__name__}: {exc}"
            )

            if attempt >= retries:
                break

            # ==========================================
            # 通信失敗後にポート自体を再初期化
            # ==========================================
            if dxl is not None:
                try:
                    print(
                        f"[DXL RETRY] {label}: "
                        "reopening serial port before retry..."
                    )

                    dxl.reopen_port(
                        wait_sec=wait_sec
                    )

                except Exception as reopen_exc:
                    print(
                        f"[DXL RETRY] {label}: "
                        "port reopen failed: "
                        f"{type(reopen_exc).__name__}: "
                        f"{reopen_exc}"
                    )

            time.sleep(wait_sec)

    raise RuntimeError(
        f"{label} failed after {retries} attempts: "
        f"{last_error}"
    ) from last_error


# =========================================================
# 電圧ログ
# =========================================================
def _format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _voltage_target_ids() -> tuple[int, int, int]:
    return (
        GRIPPER_ID,
        SP_ROT_ID,
        SP_LIN_ID,
    )


def log_dynamixel_voltage(
    dxl: Dynamixel,
    phase: str,
    note: str = "",
) -> dict[int, dict[str, Any]]:
    """
    全Dynamixelの入力電圧をCSVへ保存する。

    この関数は診断ログ用なので、通信失敗やファイル保存失敗が
    ロボット本体の処理を止めないように設計している。

    Parameters
    ----------
    dxl:
        使用中のDynamixelオブジェクト。同じポートを別プロセスから
        同時に開かないこと。
    phase:
        動作フェーズ名。
    note:
        例外内容や補足情報。

    Returns
    -------
    dict
        IDごとの電圧、基準電圧、差分、通信状態。
    """

    # 通常運用ではDynamixelへの追加readを一切行わない。
    if not ENABLE_DXL_VOLTAGE_DIAGNOSTICS:
        return {}

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )[:-3]

    monotonic_sec = time.monotonic()
    results: dict[int, dict[str, Any]] = {}

    for dxl_id in _voltage_target_ids():
        try:
            voltage = float(
                dxl.read_input_voltage(dxl_id)
            )

            if dxl_id not in _VOLTAGE_BASELINE_BY_ID:
                _VOLTAGE_BASELINE_BY_ID[dxl_id] = voltage

            baseline = _VOLTAGE_BASELINE_BY_ID[dxl_id]
            delta = voltage - baseline

            results[dxl_id] = {
                "voltage": voltage,
                "baseline": baseline,
                "delta": delta,
                "ok": True,
                "error": "",
            }

        except Exception as exc:
            results[dxl_id] = {
                "voltage": "",
                "baseline": _VOLTAGE_BASELINE_BY_ID.get(
                    dxl_id,
                    "",
                ),
                "delta": "",
                "ok": False,
                "error": _format_exception(exc),
            }

    header = [
        "session_id",
        "timestamp",
        "monotonic_sec",
        "phase",
        "note",
        "all_ids_ok",
        "gripper_id",
        "gripper_voltage_v",
        "gripper_baseline_v",
        "gripper_delta_v",
        "gripper_ok",
        "gripper_error",
        "spacer_rot_id",
        "spacer_rot_voltage_v",
        "spacer_rot_baseline_v",
        "spacer_rot_delta_v",
        "spacer_rot_ok",
        "spacer_rot_error",
        "spacer_lin_id",
        "spacer_lin_voltage_v",
        "spacer_lin_baseline_v",
        "spacer_lin_delta_v",
        "spacer_lin_ok",
        "spacer_lin_error",
    ]

    all_ids_ok = all(
        bool(results[dxl_id]["ok"])
        for dxl_id in _voltage_target_ids()
    )

    row = [
        LOG_SESSION_ID,
        timestamp,
        f"{monotonic_sec:.6f}",
        phase,
        note,
        all_ids_ok,
        GRIPPER_ID,
        results[GRIPPER_ID]["voltage"],
        results[GRIPPER_ID]["baseline"],
        results[GRIPPER_ID]["delta"],
        results[GRIPPER_ID]["ok"],
        results[GRIPPER_ID]["error"],
        SP_ROT_ID,
        results[SP_ROT_ID]["voltage"],
        results[SP_ROT_ID]["baseline"],
        results[SP_ROT_ID]["delta"],
        results[SP_ROT_ID]["ok"],
        results[SP_ROT_ID]["error"],
        SP_LIN_ID,
        results[SP_LIN_ID]["voltage"],
        results[SP_LIN_ID]["baseline"],
        results[SP_LIN_ID]["delta"],
        results[SP_LIN_ID]["ok"],
        results[SP_LIN_ID]["error"],
    ]

    try:
        VOLTAGE_LOG_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        new_file = not VOLTAGE_LOG_PATH.exists()

        with VOLTAGE_LOG_PATH.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            if new_file:
                writer.writerow(header)

            writer.writerow(row)

    except Exception as exc:
        print(
            "[DXL VOLTAGE LOG ERROR] "
            f"phase={phase}, "
            f"error={_format_exception(exc)}"
        )

    display_parts = []

    for name, dxl_id in (
        ("GRIPPER", GRIPPER_ID),
        ("SP_ROT", SP_ROT_ID),
        ("SP_LIN", SP_LIN_ID),
    ):
        result = results[dxl_id]

        if result["ok"]:
            display_parts.append(
                f"{name}(ID={dxl_id})="
                f"{result['voltage']:.1f}V "
                f"(delta={result['delta']:+.1f}V)"
            )
        else:
            display_parts.append(
                f"{name}(ID={dxl_id})=READ_ERROR"
            )

    print(
        f"[DXL VOLTAGE] phase={phase} | "
        + " | ".join(display_parts)
    )

    for dxl_id, result in results.items():
        if not result["ok"]:
            print(
                "[DXL VOLTAGE WARNING] "
                f"phase={phase}, ID={dxl_id}, "
                f"error={result['error']}"
            )
            continue

        delta = float(result["delta"])

        if delta <= -VOLTAGE_DROP_WARNING_V:
            print(
                "[DXL VOLTAGE WARNING] "
                f"phase={phase}, ID={dxl_id}, "
                f"voltage={result['voltage']:.1f}V, "
                f"baseline={result['baseline']:.1f}V, "
                f"drop={-delta:.1f}V"
            )

    return results


def _log_exception_voltage(
    dxl: Dynamixel,
    phase: str,
    exc: Exception,
) -> None:
    """例外時の電圧ログを安全に保存する。"""

    log_dynamixel_voltage(
        dxl,
        phase,
        note=_format_exception(exc),
    )



# =========================================================
# 長時間通信復旧 / Ping診断
# =========================================================
# 通常の0.2秒間隔リトライで復旧しなかった場合、
# 3秒 -> 5秒 -> 10秒と待ちながらTTLバス全体の復帰を確認する。
DXL_OBJECT_RECOVERY_WAIT_SCHEDULE = (3.0, 5.0, 10.0)

# 上記の待機+pingで復旧しなかった場合だけU2D2をUSBレベルでresetする。
DXL_USB_RESET_PING_WAIT_SCHEDULE = (1.0, 3.0, 5.0)
DXL_USB_DEVICE_REAPPEAR_TIMEOUT_SEC = 8.0

# Linux usbfs: USBDEVFS_RESET = _IO('U', 20) = 0x5514
USBDEVFS_RESET_IOCTL = 0x5514


def _dynamixel_bus_targets() -> tuple[tuple[str, int], ...]:
    """復旧確認対象のDynamixel名とIDを返す。"""
    return (
        ("GRIPPER", GRIPPER_ID),
        ("SP_ROT", SP_ROT_ID),
        ("SP_LIN", SP_LIN_ID),
    )


def _probe_dynamixel_bus_once() -> bool:
    """
    新しいDynamixelオブジェクトを一時的に作り、
    ID1/ID2/ID3すべてにpingが返るか確認する。

    この関数は診断専用で、必ずポートを閉じて終了する。
    """

    probe: Dynamixel | None = None

    try:
        probe = Dynamixel(
            port=PORT,
            baudrate=BAUDRATE,
        )

    except Exception as exc:
        print(
            "[DXL BUS PROBE] failed to open probe port: "
            f"{type(exc).__name__}: {exc}"
        )
        return False

    all_ok = True

    try:
        for name, dxl_id in _dynamixel_bus_targets():
            try:
                model_number = probe.ping(dxl_id)

                print(
                    "[DXL BUS PROBE] "
                    f"{name}(ID={dxl_id}) OK, "
                    f"model={model_number}"
                )

            except Exception as exc:
                all_ok = False

                print(
                    "[DXL BUS PROBE] "
                    f"{name}(ID={dxl_id}) NO RESPONSE: "
                    f"{type(exc).__name__}: {exc}"
                )

            # 連続送信を少しだけ避ける
            time.sleep(0.03)

        return all_ok

    finally:
        try:
            probe.close_port()
        except Exception as exc:
            print(
                "[DXL BUS PROBE] close warning: "
                f"{type(exc).__name__}: {exc}"
            )


def _verify_dynamixel_bus_on_object(
    dxl: Dynamixel,
) -> None:
    """初期化済みオブジェクトでID1/ID2/ID3すべてを最終確認する。"""

    for name, dxl_id in _dynamixel_bus_targets():
        model_number = dxl.ping(dxl_id)

        print(
            "[DXL BUS VERIFY] "
            f"{name}(ID={dxl_id}) OK, "
            f"model={model_number}"
        )

        time.sleep(0.03)


def _resolve_usb_device_from_serial_port(
    port: str,
) -> dict[str, str]:
    """/dev/book_handから対応するUSBデバイス本体をsysfsで特定する。"""

    port_path = Path(port)

    if not port_path.exists():
        raise RuntimeError(
            f"serial port does not exist: {port}"
        )

    try:
        real_port = port_path.resolve(strict=True)
    except Exception as exc:
        raise RuntimeError(
            f"failed to resolve serial port: {port}: {exc}"
        ) from exc

    tty_name = real_port.name
    sys_tty_device = (
        Path("/sys/class/tty")
        / tty_name
        / "device"
    )

    if not sys_tty_device.exists():
        raise RuntimeError(
            "sysfs tty device was not found: "
            f"{sys_tty_device}"
        )

    current = sys_tty_device.resolve()

    while True:
        busnum_path = current / "busnum"
        devnum_path = current / "devnum"

        if busnum_path.exists() and devnum_path.exists():
            busnum = int(busnum_path.read_text().strip())
            devnum = int(devnum_path.read_text().strip())

            def _read_optional(name: str) -> str:
                path = current / name
                if not path.exists():
                    return ""
                try:
                    return path.read_text().strip()
                except Exception:
                    return ""

            return {
                "tty": tty_name,
                "sysfs_path": str(current),
                "bus_id": current.name,
                "busnum": f"{busnum:03d}",
                "devnum": f"{devnum:03d}",
                "usbfs_path": str(
                    Path("/dev/bus/usb")
                    / f"{busnum:03d}"
                    / f"{devnum:03d}"
                ),
                "vendor": _read_optional("idVendor"),
                "product": _read_optional("idProduct"),
                "serial": _read_optional("serial"),
            }

        parent = current.parent
        if parent == current:
            break
        current = parent

    raise RuntimeError(
        "USB parent device could not be found from "
        f"{port} ({real_port})"
    )


def _wait_for_serial_port(
    port: str,
    timeout_sec: float,
) -> bool:
    """USB reset後にudev symlinkが戻るまで待つ。"""

    deadline = time.monotonic() + float(timeout_sec)

    while time.monotonic() < deadline:
        try:
            path = Path(port)
            if path.exists():
                path.resolve(strict=True)
                return True
        except Exception:
            pass

        time.sleep(0.2)

    return False


def _usbdevfs_reset(usbfs_path: str) -> None:
    """Linux USBDEVFS_RESET ioctlでUSBデバイスをresetする。"""

    fd = os.open(
        usbfs_path,
        os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
    )

    try:
        fcntl.ioctl(
            fd,
            USBDEVFS_RESET_IOCTL,
            0,
        )
    finally:
        os.close(fd)


def _sudo_usb_unbind_bind(bus_id: str) -> None:
    """USBDEVFS_RESET権限が無い場合、sudo -nでunbind/bindを試す。"""

    unbind_path = "/sys/bus/usb/drivers/usb/unbind"
    bind_path = "/sys/bus/usb/drivers/usb/bind"

    unbind = subprocess.run(
        ["sudo", "-n", "tee", unbind_path],
        input=f"{bus_id}\n",
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )

    if unbind.returncode != 0:
        raise RuntimeError(
            "sudo USB unbind failed: "
            f"{unbind.stderr.strip()}"
        )

    time.sleep(1.0)

    bind = subprocess.run(
        ["sudo", "-n", "tee", bind_path],
        input=f"{bus_id}\n",
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )

    if bind.returncode != 0:
        raise RuntimeError(
            "sudo USB bind failed: "
            f"{bind.stderr.strip()}"
        )


def _reset_dynamixel_usb_adapter() -> bool:
    """PORTに対応するU2D2/USBシリアルだけをUSBレベルでresetする。"""

    print(
        "[DXL USB RECOVERY] resolving USB device from "
        f"{PORT} ..."
    )

    try:
        info = _resolve_usb_device_from_serial_port(PORT)
    except Exception as exc:
        print(
            "[DXL USB RECOVERY] USB device resolution failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return False

    print(
        "[DXL USB RECOVERY] target: "
        f"tty={info['tty']}, "
        f"bus_id={info['bus_id']}, "
        f"usbfs={info['usbfs_path']}, "
        f"VID:PID={info['vendor']}:{info['product']}, "
        f"serial={info['serial'] or 'N/A'}"
    )

    reset_ok = False

    try:
        print(
            "[DXL USB RECOVERY] trying USBDEVFS_RESET ioctl..."
        )
        _usbdevfs_reset(info["usbfs_path"])
        print(
            "[DXL USB RECOVERY] USBDEVFS_RESET ioctl succeeded"
        )
        reset_ok = True
    except Exception as exc:
        print(
            "[DXL USB RECOVERY] USBDEVFS_RESET ioctl failed: "
            f"{type(exc).__name__}: {exc}"
        )

    if not reset_ok:
        try:
            print(
                "[DXL USB RECOVERY] trying sudo -n USB unbind/bind..."
            )
            _sudo_usb_unbind_bind(info["bus_id"])
            print(
                "[DXL USB RECOVERY] sudo USB unbind/bind succeeded"
            )
            reset_ok = True
        except Exception as exc:
            print(
                "[DXL USB RECOVERY] sudo USB unbind/bind failed: "
                f"{type(exc).__name__}: {exc}"
            )

    if not reset_ok:
        print(
            "[DXL USB RECOVERY] USB reset unavailable. "
            "Run setup_u2d2_usb_reset_permission.sh once "
            "if USBDEVFS_RESET reports Permission denied."
        )
        return False

    print(
        "[DXL USB RECOVERY] "
        f"waiting up to {DXL_USB_DEVICE_REAPPEAR_TIMEOUT_SEC:.1f} sec "
        f"for {PORT} ..."
    )

    if not _wait_for_serial_port(
        PORT,
        DXL_USB_DEVICE_REAPPEAR_TIMEOUT_SEC,
    ):
        print(
            "[DXL USB RECOVERY] "
            f"{PORT} did not reappear after USB reset"
        )
        return False

    try:
        real_port = Path(PORT).resolve(strict=True)
    except Exception:
        real_port = Path(PORT)

    print(
        "[DXL USB RECOVERY] serial port is back: "
        f"{PORT} -> {real_port}"
    )

    return True


DXL_FRESH_PROCESS_TIMEOUT_SEC = 20.0


def _recover_via_fresh_python_process() -> bool:
    """
    完全に別のPythonインタプリタを起動して、
    手動で python を再起動した場合と同じ条件で
    Dynamixel通信を確認する。
    """

    print(
        "[DXL FRESH PROCESS] "
        "starting completely new Python interpreter..."
    )

    child_code = r'''
import sys

from Dynamixel_win_pro_hand_book.HandBook_Retrieval import (
    init_dynamixels,
    _verify_dynamixel_bus_on_object,
)

dxl = None

try:
    print(
        "[DXL CHILD] fresh Python process started",
        flush=True,
    )

    dxl = init_dynamixels()

    _verify_dynamixel_bus_on_object(dxl)

    print(
        "[DXL CHILD] DYNAMIXEL BUS READY",
        flush=True,
    )

except Exception as exc:
    print(
        "[DXL CHILD] RECOVERY FAILED: "
        f"{type(exc).__name__}: {exc}",
        flush=True,
    )
    sys.exit(1)

finally:
    if dxl is not None:
        try:
            dxl.close_port()

            print(
                "[DXL CHILD] port closed",
                flush=True,
            )

        except Exception as exc:
            print(
                "[DXL CHILD] close warning: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
'''

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                child_code,
            ],
            cwd=str(PKG_DIR.parent),
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=DXL_FRESH_PROCESS_TIMEOUT_SEC,
            check=False,
        )

    except subprocess.TimeoutExpired:
        print(
            "[DXL FRESH PROCESS] "
            "child process timed out"
        )
        return False

    except Exception as exc:
        print(
            "[DXL FRESH PROCESS] "
            "failed to start child process: "
            f"{type(exc).__name__}: {exc}"
        )
        return False

    if result.stdout:
        for line in result.stdout.splitlines():
            print(
                f"[DXL CHILD OUTPUT] {line}"
            )

    if result.returncode != 0:
        print(
            "[DXL FRESH PROCESS] "
            f"child recovery failed: "
            f"returncode={result.returncode}"
        )
        return False

    print(
        "[DXL FRESH PROCESS] "
        "fresh Python process communicated successfully"
    )

    return True


def _initialize_after_bus_probe(
    label: str,
) -> Dynamixel:
    """ping復帰確認後に本初期化と3台最終確認を行う。"""

    recovered_dxl: Dynamixel | None = None

    try:
        recovered_dxl = init_dynamixels()
        _verify_dynamixel_bus_on_object(recovered_dxl)

        print(
            "[DXL RECOVERY] full initialization succeeded: "
            f"{label}"
        )
        return recovered_dxl

    except Exception:
        if recovered_dxl is not None:
            try:
                recovered_dxl.close_port()
            except Exception:
                pass
        raise


def _recover_dynamixel_object_with_ping(
    old_dxl: Dynamixel,
    context: str,
) -> Dynamixel:
    """通常リトライで戻らないDynamixel通信を段階的に復旧する。"""

    print(
        "[DXL LONG RECOVERY] "
        f"starting: context={context}"
    )

    try:
        old_dxl.close_port()
        print(
            "[DXL LONG RECOVERY] old Dynamixel port closed"
        )
    except Exception as exc:
        print(
            "[DXL LONG RECOVERY] old close_port warning: "
            f"{type(exc).__name__}: {exc}"
        )

    last_error: Exception | None = None

    # Phase 1: 3s -> 5s -> 10s + ping
    for stage, wait_sec in enumerate(
        DXL_OBJECT_RECOVERY_WAIT_SCHEDULE,
        start=1,
    ):
        print(
            "[DXL LONG RECOVERY] "
            f"stage={stage}/"
            f"{len(DXL_OBJECT_RECOVERY_WAIT_SCHEDULE)}, "
            f"waiting {wait_sec:.1f} sec before bus probe..."
        )

        time.sleep(wait_sec)
        bus_ok = _probe_dynamixel_bus_once()

        if not bus_ok:
            print(
                "[DXL LONG RECOVERY] "
                f"stage={stage}: bus is still unavailable"
            )
            continue

        print(
            "[DXL LONG RECOVERY] "
            f"stage={stage}: all IDs responded to ping; "
            "trying full initialization..."
        )

        try:
            recovered_dxl = _initialize_after_bus_probe(
                label=(
                    f"delayed_ping_stage={stage}, "
                    f"wait={wait_sec:.1f}s"
                )
            )
            print(
                "[DXL LONG RECOVERY] "
                f"recovered successfully at stage={stage}, "
                f"wait={wait_sec:.1f} sec"
            )
            return recovered_dxl
        except Exception as exc:
            last_error = exc
            print(
                "[DXL LONG RECOVERY] "
                f"stage={stage}: full initialization failed: "
                f"{type(exc).__name__}: {exc}"
            )

    # Phase 2: 3/5/10秒で戻らない -> U2D2 USB reset
    print(
        "[DXL USB RECOVERY] "
        "3s -> 5s -> 10s ping recovery exhausted; "
        "attempting USB-level adapter reset..."
    )

    usb_reset_ok = _reset_dynamixel_usb_adapter()

    if usb_reset_ok:
        for stage, wait_sec in enumerate(
            DXL_USB_RESET_PING_WAIT_SCHEDULE,
            start=1,
        ):
            print(
                "[DXL USB RECOVERY] "
                f"post-reset stage={stage}/"
                f"{len(DXL_USB_RESET_PING_WAIT_SCHEDULE)}, "
                f"waiting {wait_sec:.1f} sec before bus probe..."
            )

            time.sleep(wait_sec)
            bus_ok = _probe_dynamixel_bus_once()

            if not bus_ok:
                print(
                    "[DXL USB RECOVERY] "
                    f"post-reset stage={stage}: "
                    "bus is still unavailable"
                )
                continue

            print(
                "[DXL USB RECOVERY] "
                f"post-reset stage={stage}: "
                "all IDs responded; trying full initialization..."
            )

            try:
                recovered_dxl = _initialize_after_bus_probe(
                    label=(
                        f"usb_reset_stage={stage}, "
                        f"wait={wait_sec:.1f}s"
                    )
                )
                print(
                    "[DXL USB RECOVERY] USB reset recovery succeeded"
                )
                return recovered_dxl
            except Exception as exc:
                last_error = exc
                print(
                    "[DXL USB RECOVERY] "
                    "full initialization after USB reset failed: "
                    f"{type(exc).__name__}: {exc}"
                )

    # =====================================================
    # Phase 3:
    # 完全に新しいPythonプロセスで通信を試す
    # =====================================================
    print(
        "[DXL PROCESS RECOVERY] "
        "in-process recovery failed; "
        "trying fresh Python process..."
    )

    # 親プロセス側の古いシリアルポートを確実に閉じる
    try:
        old_dxl.close_port()
    except Exception:
        pass

    time.sleep(0.5)

    fresh_process_ok = _recover_via_fresh_python_process()

    if fresh_process_ok:
        print(
            "[DXL PROCESS RECOVERY] "
            "fresh Python process could communicate"
        )

        # 子プロセスがポートを閉じるまで少し待つ
        time.sleep(0.5)

        print(
            "[DXL PROCESS RECOVERY] "
            "trying completely new Dynamixel object "
            "in main process..."
        )

        try:
            recovered_dxl = init_dynamixels()

            _verify_dynamixel_bus_on_object(
                recovered_dxl
            )

            print(
                "[DXL PROCESS RECOVERY] "
                "MAIN PROCESS RECOVERY SUCCESS"
            )

            return recovered_dxl

        except Exception as exc:
            last_error = exc

            print(
                "[DXL PROCESS RECOVERY] "
                "fresh child process succeeded, "
                "but main process is still unable "
                "to communicate: "
                f"{type(exc).__name__}: {exc}"
            )

    else:
        print(
            "[DXL PROCESS RECOVERY] "
            "fresh Python process also failed"
        )

    # =====================================================
    # 全部失敗
    # =====================================================
    wait_text = " -> ".join(
        f"{value:.0f}s"
        for value in DXL_OBJECT_RECOVERY_WAIT_SCHEDULE
    )

    message = (
        "Failed to recover Dynamixel bus after "
        f"delayed ping recovery ({wait_text}), "
        "USB adapter reset, "
        "and fresh Python process; "
        f"context={context}"
    )

    if last_error is not None:
        raise RuntimeError(message) from last_error

    raise RuntimeError(message)


# =========================================================
# 初期化
# =========================================================
def init_dynamixels() -> Dynamixel:
    """Dynamixel通信を初期化して操作オブジェクトを返す。"""

    dxl = Dynamixel(
        port=PORT,
        baudrate=BAUDRATE,
    )

    try:
        log_dynamixel_voltage(
            dxl,
            "init_port_open",
        )

        # ==============================================
        # Operating Mode変更前にTorque OFF
        # ==============================================
        # 通信異常から戻った直後はTorque ONのまま残っていることがある。
        # Torque ON中はOperating Modeの変更が拒否される場合があるため、
        # 先に明示的にOFFへ戻す。
        _dxl_retry(
            "disable_torque(init)",
            lambda: dxl.disable_torque(
                GRIPPER_ID
            ),
            dxl=dxl,
            retries=1,
        )

        time.sleep(0.05)

        _dxl_retry(
            "set_mode_ex_position(init)",
            lambda: dxl.set_mode_ex_position(
                GRIPPER_ID
            ),
            dxl=dxl,
        )

        log_dynamixel_voltage(
            dxl,
            "init_after_set_mode",
        )

        position = _dxl_retry(
            "read_position(init)",
            lambda: dxl.read_position(
                GRIPPER_ID
            ),
            dxl=dxl,
        )

        print(
            f"Gripper Position : {position}"
        )

        log_dynamixel_voltage(
            dxl,
            "init_complete",
            note=f"gripper_position={position}",
        )

        print("---------------------------------")
        print("   Dynamixel READY TO MOVE")
        print("   DXL voltage diagnostics: OFF")
        print("---------------------------------")

        return dxl

    except Exception as exc:
        _log_exception_voltage(
            dxl,
            "init_exception",
            exc,
        )

        # 初期化失敗時も可能ならTorqueを切る
        try:
            dxl.disable_torque(
                GRIPPER_ID
            )
        except Exception as torque_exc:
            print(
                "[DXL CLEANUP] disable_torque failed: "
                f"{_format_exception(torque_exc)}"
            )

        try:
            dxl.close_port()
        except Exception as close_exc:
            print(
                "[DXL CLEANUP] close_port failed: "
                f"{_format_exception(close_exc)}"
            )

        raise

# =========================================================
# グリッパ操作
# =========================================================
def open_servo_key(dxl: Dynamixel) -> None:
    """キーボード操作でグリッパ目標位置を調整する。"""

    kb = kbhit.KBHit()

    try:
        log_dynamixel_voltage(
            dxl,
            "open_servo_key_before",
        )

        _dxl_retry(
            "enable_torque(open_until_width)",
            lambda: dxl.enable_torque(
                GRIPPER_ID
            ),
            dxl=dxl,
        )

        log_dynamixel_voltage(
            dxl,
            "open_servo_key_after_torque_enable",
        )

        curr_pos = _dxl_retry(
            "read_position(open_until_width)",
            lambda: dxl.read_position(
                GRIPPER_ID
            ),
            dxl=dxl,
        )
        last_sent_pos = curr_des_pos

        while True:
            if kb.kbhit():
                ch = kb.getch()

                if ch == "K":
                    curr_des_pos += GRIPPER_ROT_GAIN
                elif ch == "M":
                    curr_des_pos -= GRIPPER_ROT_GAIN
                elif ch in ("g", "G"):
                    break

            curr_des_pos = max(
                GRIPPER_CLOSE,
                min(
                    curr_des_pos,
                    GRIPPER_FULL_OPEN,
                ),
            )

            if curr_des_pos != last_sent_pos:
                target_pos = curr_des_pos

                _dxl_retry(
                    "write_position(open_servo_key)",
                    lambda target_pos=target_pos: (
                        dxl.write_position(
                            GRIPPER_ID,
                            target_pos,
                        )
                    ),
                )

                last_sent_pos = target_pos

        log_dynamixel_voltage(
            dxl,
            "open_servo_key_complete",
            note=f"last_sent_position={last_sent_pos}",
        )

    except Exception as exc:
        _log_exception_voltage(
            dxl,
            "open_servo_key_exception",
            exc,
        )
        raise


def calib_width(w_hat: float) -> float:
    """認識幅をグリッパ制御用幅へ補正する。"""

    if GRIPPER_CALIB_A == 0:
        raise ZeroDivisionError(
            "GRIPPER_CALIB_A must not be zero"
        )

    return (
        float(w_hat) - GRIPPER_CALIB_B
    ) / GRIPPER_CALIB_A



def open_until_width(
    dxl: Dynamixel,
    width: float,
    gravity: bool = False,
    worker_mode: bool = False,
) -> Dynamixel:
    """
    指定された本幅に合わせてグリッパを開く。

    通信失敗時:
    1. _dxl_retry() 内で短時間のPortHandler close/openを試す
    2. それでも失敗した場合は古いポートを完全に閉じる
    3. 3秒 -> 5秒 -> 10秒の順で待つ
    4. 各段階でID1/ID2/ID3へpingしてTTLバス復帰を確認する
    5. 3台すべて返った時点でDynamixelオブジェクトを再生成する
    6. 復旧後、新しいDynamixelオブジェクトで処理をやり直す

    Returns
    -------
    Dynamixel
        現在有効な Dynamixel オブジェクト。
        通信復旧時には新しく生成したオブジェクトが返る。
    """

    requested_width = float(width)

    # open_until_width 全体としての
    # オブジェクト再生成処理は1回だけ
    object_recovery_attempted = False

    while True:
        try:
            log_dynamixel_voltage(
                dxl,
                "open_until_width_before",
                note=(
                    f"requested_width={requested_width:.3f}; "
                    f"object_recovery_attempted="
                    f"{object_recovery_attempted}"
                ),
            )

            calibrated_width = calib_width(
                requested_width
            )

            # ==============================================
            # Torque Enable
            # ==============================================
            _dxl_retry(
                "enable_torque(open_until_width)",
                lambda: dxl.enable_torque(
                    GRIPPER_ID
                ),
                dxl=dxl,
            )

            log_dynamixel_voltage(
                dxl,
                "open_until_width_after_torque_enable",
                note=(
                    f"requested_width={requested_width:.3f}; "
                    f"calibrated_width={calibrated_width:.3f}"
                ),
            )

            # ==============================================
            # 指定幅から目標ステップを計算
            # ==============================================
            asin_value = (
                (calibrated_width + 20) / 80
                - GRIPPER_THETA_0
            )

            if not -1.0 <= asin_value <= 1.0:
                raise ValueError(
                    "open_until_widthのasin入力が範囲外です: "
                    f"requested_width={requested_width}, "
                    f"calibrated_width={calibrated_width}, "
                    f"asin_value={asin_value}"
                )

            d_theta = (
                GRIPPER_GR
                * math.asin(asin_value)
            )

            d_step = int(
                d_theta * GRIPPER_R2S
                + GRIPPER_BL
            )

            # ==============================================
            # 現在位置取得
            # ==============================================
            curr_pos = _dxl_retry(
                "read_position(open_until_width)",
                lambda: dxl.read_position(
                    GRIPPER_ID
                ),
                dxl=dxl,
            )

            # ==============================================
            # 目標位置計算
            # ==============================================
            if gravity:
                des_pos = max(
                    GRIPPER_CLOSE,
                    min(
                        curr_pos
                        + d_step
                        - GRIPPER_BL,
                        GRIPPER_FULL_OPEN,
                    ),
                )
            else:
                des_pos = max(
                    GRIPPER_CLOSE,
                    min(
                        curr_pos + d_step,
                        GRIPPER_FULL_OPEN,
                    ),
                )

            # ==============================================
            # 目標位置送信
            # ==============================================
            _dxl_retry(
                "write_position(open_until_width)",
                lambda: dxl.write_position(
                    GRIPPER_ID,
                    des_pos,
                ),
                dxl=dxl,
            )

            log_dynamixel_voltage(
                dxl,
                "open_until_width_after_command",
                note=(
                    f"requested_width={requested_width:.3f}; "
                    f"calibrated_width={calibrated_width:.3f}; "
                    f"current_position={curr_pos}; "
                    f"desired_position={des_pos}; "
                    f"gravity={gravity}"
                ),
            )

            print(
                "[GRIPPER] open_until_width command sent: "
                f"requested_width={requested_width:.3f}, "
                f"calibrated_width={calibrated_width:.3f}, "
                f"current_position={curr_pos}, "
                f"desired_position={des_pos}"
            )

            # ==============================================
            # 正常終了
            # ==============================================
            return dxl

        except Exception as exc:
            _log_exception_voltage(
                dxl,
                "open_until_width_exception",
                exc,
            )

            print(
                "[DXL OBJECT RECOVERY] "
                "open_until_width failed: "
                f"{type(exc).__name__}: {exc}"
            )


            # ==============================================
            # workerモード:
            # このPythonプロセスそのものを捨てるため
            # 長時間復旧は行わず親へ例外を返す
            # ==============================================
            if worker_mode:
                print(
                    "[DXL WORKER MODE] "
                    "open_until_width communication failed. "
                    "Requesting worker process restart."
                )
                raise

            # ==============================================
            # 長時間復旧はopen_until_width全体で1回だけ
            # ==============================================
            if object_recovery_attempted:
                print(
                    "[DXL OBJECT RECOVERY] "
                    "long recovery already attempted. "
                    "Giving up."
                )
                raise

            object_recovery_attempted = True

            # ==============================================
            # 3秒 -> 5秒 -> 10秒 + pingでバス復旧を待つ
            # ==============================================
            dxl = _recover_dynamixel_object_with_ping(
                dxl,
                context="open_until_width",
            )

            print(
                "[DXL OBJECT RECOVERY] "
                "Dynamixel communication recovered"
            )

            print(
                "[DXL OBJECT RECOVERY] "
                "retrying open_until_width "
                "with recovered object..."
            )

            # while先頭へ戻り、復旧したdxlで処理をやり直す

def grasp(
    dxl: Dynamixel,
    timeout_sec: float = 3.0,
) -> dict:
    """
    グリッパを閉じ、速度停止を検出したら把持完了とする。

    通信量削減:
    - 把持監視中は read_velocity() だけを行う
    - 50 ms間隔で監視する
    - 停止検出後に read_position() を1回だけ行う
    - PWMは読まない
    """

    if timeout_sec <= 0:
        raise ValueError(
            f"timeout_sec must be > 0: {timeout_sec}"
        )

    try:
        _dxl_retry(
            "enable_torque(grasp)",
            lambda: dxl.enable_torque(
                GRIPPER_ID
            ),
            dxl=dxl,
        )

        print("grasping object until detection")

        _dxl_retry(
            "write_position(grasp)",
            lambda: dxl.write_position(
                GRIPPER_ID,
                GRIPPER_CLOSE,
            ),
            dxl=dxl,
        )

        time.sleep(0.05)

        deadline = (
            time.monotonic()
            + float(timeout_sec)
        )

        completion_reason = ""
        last_velocity: int | None = None

        while True:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Grasp timeout"
                )

            # 監視中の通信は速度1回だけ
            last_velocity = _dxl_retry(
                "read_velocity(grasp)",
                lambda: dxl.read_velocity(
                    GRIPPER_ID
                ),
                dxl=dxl,
                retries=1,
            )

            print(
                "[GRASP DEBUG] "
                f"velocity={last_velocity}"
            )

            if abs(last_velocity) < VELOCITY_THRESHOLD:
                completion_reason = "velocity_stopped"
                print("grasp detected")
                break

            time.sleep(DXL_MONITOR_INTERVAL_SEC)

        # 停止した後に最終位置を1回だけ読む
        last_position = _dxl_retry(
            "read_position(grasp_final)",
            lambda: dxl.read_position(
                GRIPPER_ID
            ),
            dxl=dxl,
            retries=1,
        )

        # 把持後はTorque OFF
        _dxl_retry(
            "disable_torque(grasp_complete)",
            lambda: dxl.disable_torque(
                GRIPPER_ID
            ),
            dxl=dxl,
            retries=1,
        )

        result = {
            "position": int(last_position),
            "velocity": int(last_velocity),
            "reason": completion_reason,
        }

        print(
            "[GRASP RESULT] "
            f"position={result['position']}, "
            f"velocity={result['velocity']}, "
            f"reason={result['reason']}"
        )

        return result

    except Exception:
        # 通信異常時に追加の電圧readは行わない
        raise


def open_until_full(
    dxl: Dynamixel,
    asynchronous: bool = False,
    timeout_sec: float = 3.0,
    worker_mode: bool = False,
) -> Dynamixel:
    """
    グリッパを全開位置まで開く。

    通信量削減:
    - 監視中は read_velocity() だけを行う
    - 50 ms間隔で監視する
    - 停止検出後に read_position() を1回だけ行う

    通信失敗時は既存の
    3秒 -> 5秒 -> 10秒 ping + USB reset復旧を使用する。
    """

    if timeout_sec <= 0:
        raise ValueError(
            f"timeout_sec must be > 0: {timeout_sec}"
        )

    print("gripper open until full")

    object_recovery_attempted = False

    while True:
        try:
            _dxl_retry(
                "enable_torque(open_until_full)",
                lambda: dxl.enable_torque(
                    GRIPPER_ID
                ),
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
                return dxl

            # 指令直後の速度0を誤検出しないよう少し待つ
            time.sleep(0.05)

            deadline = (
                time.monotonic()
                + float(timeout_sec)
            )

            last_velocity: int | None = None

            while True:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Dynamixel timeout during open_until_full"
                    )

                # 監視中の通信は速度1回だけ
                last_velocity = _dxl_retry(
                    "read_velocity(open_until_full)",
                    lambda: dxl.read_velocity(
                        GRIPPER_ID
                    ),
                    dxl=dxl,
                    retries=1,
                )

                print(
                    "[GRIPPER DEBUG] "
                    f"velocity={last_velocity}"
                )

                if abs(last_velocity) < VELOCITY_THRESHOLD:
                    print(
                        "gripper full open: velocity stopped"
                    )
                    break

                time.sleep(DXL_MONITOR_INTERVAL_SEC)

            # 停止後に位置を1回だけ確認
            last_position = _dxl_retry(
                "read_position(open_until_full_final)",
                lambda: dxl.read_position(
                    GRIPPER_ID
                ),
                dxl=dxl,
                retries=1,
            )

            print(
                "[GRIPPER RESULT] "
                f"position={last_position}, "
                f"velocity={last_velocity}"
            )

            return dxl

        except Exception as exc:
            print(
                "[DXL OBJECT RECOVERY] "
                "open_until_full failed: "
                f"{type(exc).__name__}: {exc}"
            )

            # ==============================================
            # workerモード
            # ==============================================
            if worker_mode:
                print(
                    "[DXL WORKER MODE] "
                    "open_until_full communication failed. "
                    "Requesting worker process restart."
                )
                raise


            if object_recovery_attempted:
                print(
                    "[DXL OBJECT RECOVERY] "
                    "long recovery already attempted. "
                    "Giving up."
                )
                raise

            object_recovery_attempted = True

            dxl = _recover_dynamixel_object_with_ping(
                dxl,
                context="open_until_full",
            )

            print(
                "[DXL OBJECT RECOVERY] "
                "Dynamixel communication recovered"
            )

            print(
                "[DXL OBJECT RECOVERY] "
                "retrying open_until_full "
                "with recovered object..."
            )


# =========================================================
# 単体試験
# =========================================================
def _cleanup_dynamixel(dxl: Dynamixel | None) -> None:
    if dxl is None:
        return

    log_dynamixel_voltage(
        dxl,
        "cleanup_before",
    )

    try:
        dxl.disable_torque(
            GRIPPER_ID
        )
    except Exception as exc:
        print(
            "[DXL CLEANUP] disable_torque failed: "
            f"{_format_exception(exc)}"
        )

        _log_exception_voltage(
            dxl,
            "cleanup_disable_torque_exception",
            exc,
        )

    log_dynamixel_voltage(
        dxl,
        "cleanup_before_close_port",
    )

    try:
        dxl.close_port()
    except Exception as exc:
        print(
            "[DXL CLEANUP] close_port failed: "
            f"{_format_exception(exc)}"
        )


if __name__ == "__main__":
    hand_motors: Dynamixel | None = None

    try:
        hand_motors = init_dynamixels()
        open_until_width(
            hand_motors,
            65,
        )
        input(
            "Enterで把持を開始します。"
        )
        grasp(hand_motors)
        time.sleep(3.0)

    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    except Exception as exc:
        print(
            "[DXL FATAL] "
            f"{_format_exception(exc)}"
        )
        raise

    finally:
        _cleanup_dynamixel(
            hand_motors
        )