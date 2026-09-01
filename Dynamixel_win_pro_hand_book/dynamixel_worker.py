#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import json
import os
import signal
import sys

from . import HandBook_Retrieval as HandBook
from . import HandBook_Storage as StorageHandBook


RESPONSE_PREFIX = "__DXL_WORKER_JSON__"
PR_SET_PDEATHSIG = 1


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


def main() -> None:
    _set_parent_death_signal()

    dxl = None

    try:
        print(
            "[DXL WORKER] initializing Dynamixel...",
            flush=True,
        )

        dxl = HandBook.init_dynamixels()

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

        if command == "shutdown":
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
            if command == "open_until_width":

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

            elif command == "contract_sp_lin_2":

                data = StorageHandBook.contract_sp_lin_2(
                    dxl,
                    asynchronous=bool(
                        request.get(
                            "asynchronous",
                            False,
                        )
                    ),
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