#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import json
import os
import signal
import sys
import threading

# ============================================================
# 複数スレッドの標準出力を1行単位で排他制御する
# ============================================================
class ThreadSafeStdout:
    def __init__(self, original):
        self.original = original
        self.lock = threading.RLock()
        self.local = threading.local()

    def write(self, text):
        text = str(text)
        pending = getattr(self.local, "pending", "") + text

        while "\n" in pending:
            line, pending = pending.split("\n", 1)

            with self.lock:
                self.original.write(line + "\n")
                self.original.flush()

        self.local.pending = pending
        return len(text)

    def flush(self):
        pending = getattr(self.local, "pending", "")

        with self.lock:
            if pending:
                self.original.write(pending)
                self.local.pending = ""

            self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


sys.stdout = ThreadSafeStdout(sys.stdout)

from . import HandBook_Retrieval as HandBook
from . import HandBook_Storage as StorageHandBook


RESPONSE_PREFIX = "__DXL_WORKER_JSON__"
PR_SET_PDEATHSIG = 1

# ============================================================
# Dynamixel通信の排他制御
# ============================================================

DXL_IO_LOCK = threading.RLock()

# SP_LINバックグラウンド動作用の状態
SP_LIN_TASK_LOCK = threading.Lock()
SP_LIN_ABORT_EVENT = StorageHandBook.SP_LIN_ABORT_EVENT

SP_LIN_TASK = {
    "running": False,
    "completed": False,
    "failed": False,
    "error": None,
}
# 安全確認が終わるまでバックグラウンド駆動を禁止
ENABLE_SP_LIN_BACKGROUND = True
# SP_LINバックグラウンドスレッド本体を保持する
SP_LIN_THREAD: threading.Thread | None = None


def _sp_lin_thread_alive() -> bool:
    """SP_LINスレッドが実際に動作中か確認する。"""
    with SP_LIN_TASK_LOCK:
        return (
            SP_LIN_THREAD is not None
            and SP_LIN_THREAD.is_alive()
        )

def _prepare_storage_axes(dxl) -> None:
    """同じworkerが所有するDynamixel接続で入庫用2軸を初期化する。"""
    targets = (
        ("SP_LIN", StorageHandBook.SP_LIN_ID),
        ("SP_ROT", StorageHandBook.SP_ROT_ID),
    )

    for name, dxl_id in targets:
        print(
            f"[DXL WORKER] preparing storage axis {name}(ID={dxl_id})...",
            flush=True,
        )
        dxl.disable_torque(dxl_id)
        dxl.set_mode_ex_position(dxl_id)
        position = dxl.read_position(dxl_id)
        print(
            f"[DXL WORKER] storage axis READY: "
            f"{name}(ID={dxl_id}) position={position}",
            flush=True,
        )


def _cleanup_storage_axes(dxl) -> None:
    """終了時に入庫用2軸のトルクをOFFする。"""
    if dxl is None:
        return

    for dxl_id in (
        StorageHandBook.SP_LIN_ID,
        StorageHandBook.SP_ROT_ID,
    ):
        try:
            dxl.disable_torque(dxl_id)
        except Exception as exc:
            print(
                "[DXL WORKER] storage torque-off warning: "
                f"ID={dxl_id}, {type(exc).__name__}: {exc}",
                flush=True,
            )


def _emit(payload: dict) -> None:
    """親プロセスへ1行JSONで応答する。"""
    print(
        RESPONSE_PREFIX + json.dumps(
            payload,
            ensure_ascii=False,
        ),
        flush=True,
    )


def _set_parent_death_signal() -> None:
    """
    親プロセスが終了したらworkerもSIGTERMで終了する。

    Linux専用。
    親だけ終了してDynamixel workerが残留し、
    /dev/book_hand を保持し続けることを防ぐ。
    """
    try:
        libc = ctypes.CDLL("libc.so.6")
        result = libc.prctl(
            PR_SET_PDEATHSIG,
            signal.SIGTERM,
        )

        if result != 0:
            raise OSError(
                "prctl(PR_SET_PDEATHSIG) failed"
            )

        if os.getppid() == 1:
            os._exit(1)

    except Exception as exc:
        print(
            "[DXL WORKER] "
            "parent-death setup warning: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


def _exception_chain_text(
    exc: BaseException,
) -> str:
    """__cause__ / __context__ を含めた例外文字列を返す。"""
    parts: list[str] = []
    seen: set[int] = set()

    current: BaseException | None = exc

    while (
        current is not None
        and id(current) not in seen
    ):
        seen.add(id(current))
        parts.append(
            f"{type(current).__name__}: {current}"
        )

        if current.__cause__ is not None:
            current = current.__cause__
        else:
            current = current.__context__

    return " | ".join(parts)


def _is_restartable_communication_error(
    exc: BaseException,
) -> bool:
    """
    worker Pythonそのものを捨てるべき通信系エラーか判定する。

    True:
        Dynamixel SDK / serial / USB経路の通信異常。

    False:
        Grasp timeout、入力値エラー、機械動作上の異常など。
    """
    text = _exception_chain_text(exc).lower()

    communication_tokens = (
        "comm error",
        "[txrxresult]",
        "there is no status packet",
        "failed transmit instruction packet",
        "incorrect status packet",
        "port is in use",
        "port is not open",
        "serialexception",
        "could not open port",
        "failed to open port",
        "device or resource busy",
        "bad file descriptor",
        "input/output error",
        "i/o error",
        "no such file or directory",
    )

    return any(
        token in text
        for token in communication_tokens
    )


def _run_sp_lin_background(dxl, timeout_sec: float) -> None:
    """SP_LIN収縮とPWM監視をバックグラウンドで実行する。"""

    try:
        StorageHandBook._check_sp_lin_abort()
        result = StorageHandBook.contract_sp_lin_2(
            dxl,
            asynchronous=False,
            timeout_sec=timeout_sec,
        )

        if result is not True:
            raise RuntimeError(
                f"Unexpected SP_LIN result: {result!r}"
            )

        with SP_LIN_TASK_LOCK:
            SP_LIN_TASK["running"] = False

            # 他スレッドで異常が記録されていた場合は
            # 正常終了しても異常状態を上書きしない
            if not SP_LIN_TASK["failed"] and not SP_LIN_ABORT_EVENT.is_set():
                SP_LIN_TASK["completed"] = True
                SP_LIN_TASK["error"] = None
            else:
                SP_LIN_TASK["completed"] = False

    except Exception as exc:
        with SP_LIN_TASK_LOCK:
            SP_LIN_TASK["running"] = False
            SP_LIN_TASK["completed"] = False
            SP_LIN_TASK["failed"] = True
            SP_LIN_TASK["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        print(
            f"[SP_LIN BACKGROUND ERROR] {exc!r}",
            flush=True,
        )

def main() -> None:
    global SP_LIN_THREAD

    _set_parent_death_signal()

    dxl = None

    try:
        print(
            "[DXL WORKER] initializing Dynamixel...",
            flush=True,
        )

        dxl = HandBook.init_dynamixels()
        # HandBook_Storage.pyと通信ロックを共有する
        HandBook.set_dxl_io_lock(DXL_IO_LOCK)
        StorageHandBook.set_dxl_io_lock(DXL_IO_LOCK)

        HandBook._verify_dynamixel_bus_on_object(
            dxl
        )

        _emit(
            {
                "type": "ready",
                "ok": True,
                "pid": os.getpid(),
            }
        )

    except Exception as exc:
        print(
            "[DXL WORKER] startup failed: "
            f"{_exception_chain_text(exc)}",
            flush=True,
        )

        _emit(
            {
                "type": "ready",
                "ok": False,
                "pid": os.getpid(),
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )

        sys.stdout.flush()
        os._exit(2)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()

        if not raw_line:
            continue

        try:
            request = json.loads(
                raw_line
            )

        except Exception as exc:
            _emit(
                {
                    "type": "result",
                    "ok": False,
                    "cmd": "",
                    "error": (
                        f"invalid JSON: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "restart_worker": False,
                    "worker_will_exit": False,
                }
            )
            continue

        command = str(
            request.get(
                "cmd",
                "",
            )
        )

        # SP_LIN動作中は、同じ軸への追加指令を禁止する
        SP_LIN_COMMANDS = {
            "expand_sp_lin",
            "expand_sp_lin_right_tilted",
            "contract_sp_lin_1",
            "contract_sp_lin_2",
            "contract_sp_lin_pwm_home",
            "contract_sp_lin_2_start",
        }

        with SP_LIN_TASK_LOCK:
            sp_lin_thread_alive = (
                SP_LIN_THREAD is not None
                and SP_LIN_THREAD.is_alive()
            )
            sp_lin_busy = SP_LIN_TASK["running"] or sp_lin_thread_alive
            sp_lin_failed = SP_LIN_TASK["failed"]

        if command in SP_LIN_COMMANDS and (sp_lin_busy or sp_lin_failed):
            _emit({
                "type": "result",
                "ok": False,
                "cmd": command,
                "error": (
                    "SP_LIN command rejected: "
                    f"running={sp_lin_busy}, failed={sp_lin_failed}"
                ),
                "restart_worker": False,
                "worker_will_exit": False,
            })
            continue

        if command == "emergency_stop_sp_lin":
            # Sticky stop request: do not allow any further SP_LIN operation.
            SP_LIN_ABORT_EVENT.set()
            with SP_LIN_TASK_LOCK:
                SP_LIN_TASK["failed"] = True
                SP_LIN_TASK["completed"] = False
                SP_LIN_TASK["error"] = "Emergency stop requested"
            try:
                with DXL_IO_LOCK:
                    dxl.disable_torque(StorageHandBook.SP_LIN_ID)
                    value = int(dxl.read_torque_enable(StorageHandBook.SP_LIN_ID))
                    if value != 0:
                        raise RuntimeError(f"SP_LIN torque still enabled: {value}")
                _emit({"type": "result", "ok": True, "cmd": command,
                       "data": {"torque_off_verified": True}})
            except Exception as exc:
                _emit({"type": "result", "ok": False, "cmd": command,
                       "error": "SP_LIN motor stop NOT verified: " + str(exc),
                       "restart_worker": False, "worker_will_exit": False})
            continue

        if command == "shutdown":

            with SP_LIN_TASK_LOCK:
                sp_lin_running = SP_LIN_TASK["running"]
                sp_lin_failed = SP_LIN_TASK["failed"]
                sp_lin_thread_alive = (
                    SP_LIN_THREAD is not None
                    and SP_LIN_THREAD.is_alive()
                )

            if sp_lin_running or sp_lin_failed or sp_lin_thread_alive:
                _emit({
                    "type": "result",
                    "ok": False,
                    "cmd": command,
                    "error": (
                        "SP_LIN background task is running or failed. "
                        "Shutdown rejected."
                    ),
                    "restart_worker": False,
                    "worker_will_exit": False,
                })
                continue

            _cleanup_storage_axes(dxl)

            try:
                HandBook._cleanup_dynamixel(
                    dxl
                )
            except Exception as exc:
                print(
                    "[DXL WORKER] "
                    "shutdown cleanup warning: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

            _emit(
                {
                    "type": "result",
                    "ok": True,
                    "cmd": command,
                    "data": None,
                }
            )
            return

        try:
            if command == "sp_lin_task_status":
                with SP_LIN_TASK_LOCK:
                    data = SP_LIN_TASK.copy()
                    data["thread_alive"] = (
                        SP_LIN_THREAD is not None
                        and SP_LIN_THREAD.is_alive()
                    )

            elif command == "open_until_width":

                dxl = HandBook.open_until_width(
                    dxl,
                    float(request["width"]),
                    gravity=bool(
                        request.get(
                            "gravity",
                            False,
                        )
                    ),
                    worker_mode=True,
                )

                data = None

            elif command == "grasp":

                data = HandBook.grasp(
                    dxl,
                    timeout_sec=float(
                        request.get(
                            "timeout_sec",
                            3.0,
                        )
                    ),
                )

            elif command == "open_until_full":

                dxl = HandBook.open_until_full(
                    dxl,
                    asynchronous=bool(
                        request.get(
                            "asynchronous",
                            False,
                        )
                    ),
                    timeout_sec=float(
                        request.get(
                            "timeout_sec",
                            3.0,
                        )
                    ),
                    worker_mode=True,
                )

                data = None

            elif command == "open_for_storage_release":

                data = StorageHandBook.open_for_storage_release(
                    dxl,
                    timeout_sec=float(
                        request.get(
                            "timeout_sec",
                            StorageHandBook.STORAGE_RELEASE_TIMEOUT_SEC,
                        )
                    ),
                )

            elif command == "open_for_storage_release_partial":

                data = StorageHandBook.open_for_storage_release_partial(
                    dxl,
                    offset_steps=int(request.get("offset_steps", 2000)),
                    timeout_sec=float(request.get(
                        "timeout_sec",
                        StorageHandBook.STORAGE_RELEASE_TIMEOUT_SEC,
                    )),
                )

            elif command == "ping_all":

                HandBook._verify_dynamixel_bus_on_object(
                    dxl
                )

                data = {
                    "all_ok": True
                }

            elif command == "expand_sp_lin":

                data = StorageHandBook.expand_sp_lin(
                    dxl,
                    asynchronous=bool(
                        request.get(
                            "asynchronous",
                            False,
                        )
                    ),
                )

            elif command == "expand_sp_lin_right_tilted":

                data = StorageHandBook.expand_sp_lin_right_tilted(
                    dxl,
                    asynchronous=bool(
                        request.get(
                            "asynchronous",
                            False,
                        )
                    ),
                )

            elif command == "contract_sp_lin_1":

                data = StorageHandBook.contract_sp_lin_1(
                    dxl,
                    asynchronous=bool(
                        request.get(
                            "asynchronous",
                            False,
                        )
                    ),
                )

            elif command == "contract_sp_lin_2_start":

                if SP_LIN_ABORT_EVENT.is_set():
                    raise RuntimeError("SP_LIN emergency stop latched")
                if not ENABLE_SP_LIN_BACKGROUND:
                    raise RuntimeError(
                        "SP_LIN background operation is disabled "
                        "until safety verification is completed"
                    )

 
                with SP_LIN_TASK_LOCK:
                    if (
                        SP_LIN_TASK["running"]
                        or SP_LIN_TASK["failed"]
                        or (
                            SP_LIN_THREAD is not None
                            and SP_LIN_THREAD.is_alive()
                        )
                    ):
                        raise RuntimeError(
                            "SP_LIN background task is busy or failed"
                        )

                    SP_LIN_TASK.update({
                        "running": True,
                        "completed": False,
                        "failed": False,
                        "error": None,
                    })

                    try:
                        SP_LIN_THREAD = threading.Thread(
                            target=_run_sp_lin_background,
                            args=(
                                dxl,
                                float(request.get("timeout_sec", 20.0)),
                            ),
                            name="sp_lin_background",
                            daemon=False,
                        )
                        SP_LIN_THREAD.start()

                    except Exception as exc:
                        SP_LIN_TASK["running"] = False
                        SP_LIN_TASK["completed"] = False
                        SP_LIN_TASK["failed"] = True
                        SP_LIN_TASK["error"] = str(exc)
                        raise

                data = {"started": True}

            elif command == "contract_sp_lin_2":

                contract_sp_lin_2_kwargs = {
                    "asynchronous": bool(
                        request.get(
                            "asynchronous",
                            False,
                        )
                    ),
                }
                if "timeout_sec" in request:
                    contract_sp_lin_2_kwargs["timeout_sec"] = float(
                        request["timeout_sec"]
                    )

                data = StorageHandBook.contract_sp_lin_2(
                    dxl,
                    **contract_sp_lin_2_kwargs,
                )


            elif command == "contract_sp_lin_pwm_home":

                print(
                    "[DXL WORKER] SP_LIN PWM homing start",
                    flush=True,
                )

                data = StorageHandBook.home_sp_lin_stop_test(
                    dxl,
                    timeout_sec=float(
                        request.get("timeout_sec", 100.0)
                    ),
                )

                print(
                    "[DXL WORKER] SP_LIN PWM homing result:",
                    data,
                    flush=True,
                )



            elif command == "rotate_spacer":

                data = StorageHandBook.rotate_spacer(
                    dxl,
                    float(request["theta_deg"]),
                )

            elif command == "reset_rot":

                data = StorageHandBook.reset_rot(
                    dxl,
                    asynchronous=bool(
                        request.get(
                            "asynchronous",
                            False,
                        )
                    ),
                )

            elif command == "ungrasp_auto":

                # 現在位置 + RELEASE_GAIN の相対操作。
                # 二重実行を避けるためclient側でも自動再試行しない。
                data = StorageHandBook.ungrasp_auto(dxl)

            else:
                raise ValueError(
                    "unknown worker command: "
                    f"{command!r}"
                )

            _emit(
                {
                    "type": "result",
                    "ok": True,
                    "cmd": command,
                    "data": data,
                }
            )

        except Exception as exc:

            restart_worker = (
                _is_restartable_communication_error(
                    exc
                )
            )

            error_text = (
                f"{type(exc).__name__}: {exc}"
            )

            # SP_LIN動作中・異常状態では、
            # 通信異常によるworker自動再起動を禁止する
            if restart_worker:
                with SP_LIN_TASK_LOCK:
                    sp_lin_running = SP_LIN_TASK["running"]
                    sp_lin_failed = SP_LIN_TASK["failed"]
                    sp_lin_thread_alive = (
                        SP_LIN_THREAD is not None
                        and SP_LIN_THREAD.is_alive()
                    )

                    sp_lin_protected = (
                        sp_lin_running
                        or sp_lin_failed
                        or sp_lin_thread_alive
                    )

                    if sp_lin_protected:
                        SP_LIN_TASK["failed"] = True
                        SP_LIN_TASK["completed"] = False
                        SP_LIN_TASK["error"] = (
                            "Communication failure during SP_LIN operation: "
                            + error_text
                        )

                if sp_lin_protected:
                    print(
                        "[DXL WORKER][DANGER] "
                        "Communication failure during SP_LIN operation. "
                        "Automatic restart prohibited. "
                        f"{error_text}",
                        flush=True,
                    )

                    _emit({
                        "type": "result",
                        "ok": False,
                        "cmd": command,
                        "error": (
                            "SP_LIN fault state. "
                            "Motor stop is not confirmed. "
                            + error_text
                        ),
                        "restart_worker": False,
                        "worker_will_exit": False,
                    })

                    continue

            if restart_worker:

                print(
                    "[DXL WORKER] "
                    "communication failure detected: "
                    f"{error_text}",
                    flush=True,
                )

                # ==========================================
                # 重要:
                # 親へrestart要求を送る「前」に
                # serial portを明示的に閉じる
                # ==========================================
                if dxl is not None:
                    try:
                        print(
                            "[DXL WORKER] "
                            "closing serial port before "
                            "restart request...",
                            flush=True,
                        )

                        dxl.close_port()

                        print(
                            "[DXL WORKER] "
                            "serial port closed before "
                            "restart request",
                            flush=True,
                        )

                    except Exception as close_exc:
                        print(
                            "[DXL WORKER] "
                            "serial port close warning: "
                            f"{type(close_exc).__name__}: "
                            f"{close_exc}",
                            flush=True,
                        )

                # ★ closeが終わってから親へ通知
                _emit(
                    {
                        "type": "result",
                        "ok": False,
                        "cmd": command,
                        "error": error_text,
                        "restart_worker": True,
                        "worker_will_exit": True,
                    }
                )

                sys.stdout.flush()

                os._exit(3)

            else:

                print(
                    "[DXL WORKER] "
                    "non-communication error detected. "
                    "Worker remains alive: "
                    f"{error_text}",
                    flush=True,
                )

                _emit(
                    {
                        "type": "result",
                        "ok": False,
                        "cmd": command,
                        "error": error_text,
                        "restart_worker": False,
                        "worker_will_exit": False,
                    }
                )

if __name__ == "__main__":
    main()
