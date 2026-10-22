#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time

from pathlib import Path
from typing import Any


RESPONSE_PREFIX = "__DXL_WORKER_JSON__"
DEFAULT_STORAGE_RELEASE_TIMEOUT_SEC = 5.0


class DynamixelWorkerError(
    RuntimeError
):
    """Dynamixel worker / IPC全般の例外。"""
    pass


class DynamixelWorkerCommandError(
    DynamixelWorkerError
):
    """
    workerがコマンド処理中に返した例外。

    restart_worker=True:
        通信異常。workerを完全終了→新規起動してよい。

    restart_worker=False:
        Grasp timeout / ValueError等。
        workerを再起動せず、そのまま呼び出し元へ返す。
    """

    def __init__(
        self,
        message: str,
        *,
        command: str,
        restart_worker: bool,
    ) -> None:
        super().__init__(message)

        self.command = str(
            command
        )

        self.restart_worker = bool(
            restart_worker
        )


class DynamixelWorkerClient:
    """
    親プロセス側のDynamixel操作窓口。

    - 親自身は /dev/book_hand をopenしない。
    - Dynamixel SDK / PortHandler はworker Pythonだけが所有する。
    - 通信異常時だけworkerを完全終了して新しいPythonを起動する。
    - 非通信エラーではworkerを殺さない。
    - コマンドはlockで直列化する。
    - worker stdoutは専用reader threadで常時読み、
      TextIOWrapper/selectのバッファ不整合による偽timeoutを防ぐ。
    """

    def __init__(
        self,
        startup_timeout_sec: float = 15.0,
        restart_wait_sec: float = 3.0,
    ) -> None:

        self._startup_timeout_sec = float(
            startup_timeout_sec
        )

        self._restart_wait_sec = float(
            restart_wait_sec
        )

        # Once a background start has been attempted, the motor state may be
        # unknown after IPC errors. Never discard/restart this worker automatically.
        self._sp_lin_start_attempted = False
        self._shutdown_acknowledged = False
        self._proc: subprocess.Popen[str] | None = None

        self._command_lock = (
            threading.RLock()
        )

        self._stdout_queue: (
            queue.Queue[str | None] | None
        ) = None

        self._stdout_thread: (
            threading.Thread | None
        ) = None

        self._project_dir = (
            Path(__file__)
            .resolve()
            .parent
            .parent
        )

        self.start()

    # =====================================================
    # stdout reader
    # =====================================================

    @staticmethod
    def _stdout_reader_loop(
        proc: subprocess.Popen[str],
        output_queue: queue.Queue[str | None],
    ) -> None:
        """
        worker stdoutを1本の専用threadで最後まで読む。

        select()とTextIOWrapper.readline()を組み合わせると、
        Python側の内部bufferに次行が既にあるのに
        OS fdがreadableではないためtimeoutする場合がある。
        それを避けるため、stdoutを読む主体をこのthreadだけにする。
        """
        stdout = proc.stdout

        if stdout is None:
            output_queue.put(None)
            return

        try:
            for line in stdout:
                output_queue.put(
                    line.rstrip("\n")
                )

        except Exception as exc:
            output_queue.put(
                "[DXL CLIENT READER ERROR] "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            # EOF通知
            output_queue.put(None)

    # =====================================================
    # worker起動
    # =====================================================

    def start(self) -> None:

        with self._command_lock:

            if (
                self._proc is not None
                and self._proc.poll() is None
            ):
                return

            if self._sp_lin_start_attempted:
                raise DynamixelWorkerError(
                    "SP_LIN background start was attempted; automatic worker "
                    "replacement is blocked until independent stop verification."
                )
            if self._proc is not None:
                # Only reap processes already known to have exited.
                self._hard_stop()

            print(
                "[DXL CLIENT] "
                "starting Dynamixel worker..."
            )

            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    (
                        "Dynamixel_win_pro_hand_book."
                        "dynamixel_worker"
                    ),
                ],
                cwd=str(
                    self._project_dir
                ),
                env=os.environ.copy(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            output_queue: queue.Queue[
                str | None
            ] = queue.Queue()

            stdout_thread = threading.Thread(
                target=self._stdout_reader_loop,
                args=(
                    proc,
                    output_queue,
                ),
                name=(
                    "dynamixel-worker-stdout-reader"
                ),
                daemon=True,
            )

            self._proc = proc
            self._stdout_queue = output_queue
            self._stdout_thread = stdout_thread

            stdout_thread.start()

            try:
                response = self._read_response(
                    timeout_sec=(
                        self._startup_timeout_sec
                    )
                )

            except Exception:
                # Startup timeout is not proof that the worker is idle.
                if proc.poll() is not None:
                    self._hard_stop()
                raise

            if (
                response.get("type") != "ready"
                or not response.get("ok")
            ):

                error = response.get(
                    "error",
                    "worker startup failed",
                )

                if proc.poll() is not None:
                    self._hard_stop()

                raise DynamixelWorkerError(
                    "Dynamixel worker startup "
                    f"failed: {error}"
                )

            print(
                "[DXL CLIENT] worker READY: "
                f"pid={response.get('pid')}"
            )

    # =====================================================
    # worker完全終了
    # =====================================================

    def _hard_stop(self) -> None:
        """Reap an already-exited worker only. Never signal a live worker.

        SIGTERM/SIGKILL can strand an actively moving actuator, and an IPC
        timeout cannot establish whether the motor or monitoring thread stopped.
        """
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            raise DynamixelWorkerError(
                "Refusing to terminate a live worker. Motor state may be "
                "unknown; verify stop independently before recovery."
            )

        stdout_thread = self._stdout_thread
        self._proc = None
        self._stdout_queue = None
        self._stdout_thread = None
        self._shutdown_acknowledged = False

        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
        if stdout_thread is not None and stdout_thread.is_alive():
            stdout_thread.join(timeout=1.0)

    # =====================================================
    # worker再起動
    # =====================================================
    def restart(self) -> None:
        """Only restart a worker that has independently exited, before any
        SP_LIN background start attempt in this client session.
        """
        with self._command_lock:
            if self._sp_lin_start_attempted:
                raise DynamixelWorkerError(
                    "Restart blocked: SP_LIN background start was attempted. "
                    "Physical stop is not established by task status alone."
                )
            if self._proc is not None:
                if self._proc.poll() is None:
                    raise DynamixelWorkerError(
                        "Restart blocked: worker still alive. "
                        "No automatic termination of a live worker."
                    )
                self._hard_stop()
            self.start()

    # =====================================================
    # 終了
    # =====================================================

    def close(self) -> None:
        with self._command_lock:
            proc = self._proc
            if proc is None:
                return
            if proc.poll() is not None:
                self._hard_stop()
                return

            try:
                status = self._call_once(
                    {"cmd": "sp_lin_task_status"}, timeout_sec=5.0
                )
            except Exception as exc:
                raise DynamixelWorkerError(
                    "Worker status is unknown; live process will not be killed."
                ) from exc
            if not isinstance(status, dict):
                raise DynamixelWorkerError("Invalid worker status; close blocked")
            if (status.get("running") is not False
                    or status.get("failed") is not False
                    or status.get("thread_alive") is not False):
                raise DynamixelWorkerError(
                    f"SP_LIN not verified idle; close blocked: {status!r}"
                )
            if self._sp_lin_start_attempted and status.get("completed") is not True:
                raise DynamixelWorkerError(
                    "SP_LIN completion unknown; close blocked."
                )

            # Worker shutdown must report success, then exit normally.
            self._call_once({"cmd": "shutdown"}, timeout_sec=5.0)
            self._shutdown_acknowledged = True
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired as exc:
                raise DynamixelWorkerError(
                    "Worker did not exit after acknowledged shutdown; "
                    "refusing force termination."
                ) from exc
            self._hard_stop()

    # =====================================================
    # workerから返答取得
    # =====================================================

    def _read_response(
        self,
        timeout_sec: float,
    ) -> dict[str, Any]:

        proc = self._proc
        output_queue = self._stdout_queue

        if proc is None:
            raise DynamixelWorkerError(
                "worker process is unavailable"
            )

        if output_queue is None:
            raise DynamixelWorkerError(
                "worker stdout queue is unavailable"
            )

        deadline = (
            time.monotonic()
            + float(timeout_sec)
        )

        while True:

            remaining = (
                deadline
                - time.monotonic()
            )

            if remaining <= 0:
                raise DynamixelWorkerError(
                    "worker response timeout "
                    f"after {timeout_sec:.1f} sec"
                )

            try:
                line = output_queue.get(
                    timeout=remaining
                )

            except queue.Empty:
                raise DynamixelWorkerError(
                    "worker response timeout "
                    f"after {timeout_sec:.1f} sec"
                )

            # reader threadがEOFを検出
            if line is None:

                return_code = (
                    proc.poll()
                )

                raise DynamixelWorkerError(
                    "worker exited before "
                    "response: "
                    f"returncode={return_code}"
                )

            if line.startswith(
                RESPONSE_PREFIX
            ):

                payload_text = (
                    line[
                        len(
                            RESPONSE_PREFIX
                        ):
                    ]
                )

                try:
                    payload = json.loads(
                        payload_text
                    )

                except Exception as exc:
                    raise DynamixelWorkerError(
                        "invalid worker "
                        "response JSON: "
                        f"{payload_text!r}: "
                        f"{exc}"
                    ) from exc

                if not isinstance(
                    payload,
                    dict,
                ):
                    raise DynamixelWorkerError(
                        "worker response must "
                        f"be dict: {payload!r}"
                    )

                return payload

            # HandBook_Retrieval.py側の既存printをそのまま表示
            print(
                f"[DXL WORKER] {line}"
            )

    # =====================================================
    # 1回コマンド送信
    # =====================================================

    def _call_once(
        self,
        request: dict[str, Any],
        timeout_sec: float,
    ) -> Any:

        proc = self._proc

        if (
            proc is None
            or proc.poll() is not None
        ):
            raise DynamixelWorkerError(
                "Dynamixel worker "
                "is not running"
            )

        if proc.stdin is None:
            raise DynamixelWorkerError(
                "worker stdin is unavailable"
            )

        command = str(
            request.get(
                "cmd",
                "unknown",
            )
        )

        wire = json.dumps(
            request,
            ensure_ascii=False,
        )

        try:
            proc.stdin.write(
                wire + "\n"
            )

            proc.stdin.flush()

        except Exception as exc:
            raise DynamixelWorkerError(
                "failed to send command "
                f"to worker: {exc}"
            ) from exc

        response = self._read_response(
            timeout_sec=timeout_sec
        )

        if response.get("type") != "result":
            raise DynamixelWorkerError(
                "unexpected worker response "
                f"type: {response!r}"
            )

        if not response.get("ok"):

            raise DynamixelWorkerCommandError(
                str(
                    response.get(
                        "error",
                        (
                            "Dynamixel worker "
                            "command failed"
                        ),
                    )
                ),
                command=command,
                restart_worker=bool(
                    response.get(
                        "restart_worker",
                        False,
                    )
                ),
            )

        return response.get(
            "data"
        )

    # =====================================================
    # 通信異常:
    # worker kill -> fresh worker -> 1回だけ再試行
    #
    # 非通信異常:
    # workerを殺さず、そのまま上位へ返す
    # =====================================================

    def _call_with_restart(
        self,
        request: dict[str, Any],
        timeout_sec: float,
        normalize_closed_before_retry: bool = False,
    ) -> Any:
        """Fail closed on errors: an IPC timeout does not establish a safe
        worker or motor state, and replaying movement can duplicate commands.
        """
        with self._command_lock:
            try:
                return self._call_once(request, timeout_sec=timeout_sec)
            except Exception as exc:
                print(
                    "[DXL CLIENT] command failed; automatic worker restart "
                    "and movement retry suppressed: "
                    f"cmd={request.get('cmd')}, error={type(exc).__name__}: {exc}"
                )
                raise

    # =====================================================
    # public API
    # =====================================================

    def open_until_width(
        self,
        width: float,
        gravity: bool = False,
    ) -> "DynamixelWorkerClient":

        self._call_with_restart(
            {
                "cmd": "open_until_width",
                "width": float(width),
                "gravity": bool(gravity),
            },
            timeout_sec=10.0,
            normalize_closed_before_retry=True,
        )

        return self

    def grasp(
        self,
        timeout_sec: float = 3.0,
    ) -> dict[str, Any]:

        result = self._call_with_restart(
            {
                "cmd": "grasp",
                "timeout_sec": float(
                    timeout_sec
                ),
            },
            timeout_sec=max(
                8.0,
                float(timeout_sec) + 5.0,
            ),
        )

        if not isinstance(
            result,
            dict,
        ):
            raise DynamixelWorkerError(
                "invalid grasp result "
                f"from worker: {result!r}"
            )

        return result

    def open_until_full(
        self,
        asynchronous: bool = False,
        timeout_sec: float = 3.0,
    ) -> "DynamixelWorkerClient":

        self._call_with_restart(
            {
                "cmd": "open_until_full",
                "asynchronous": bool(
                    asynchronous
                ),
                "timeout_sec": float(
                    timeout_sec
                ),
            },
            timeout_sec=max(
                8.0,
                float(timeout_sec) + 5.0,
            ),
        )

        return self

    def open_for_storage_release(
        self,
        timeout_sec: float = DEFAULT_STORAGE_RELEASE_TIMEOUT_SEC,
    ) -> bool:

        wait_timeout_sec = float(timeout_sec)
        if not math.isfinite(wait_timeout_sec) or wait_timeout_sec <= 0.0:
            raise ValueError(
                "open_for_storage_release timeout_sec must be positive: "
                f"{timeout_sec}"
            )

        result = self._call_with_restart(
            {
                "cmd": "open_for_storage_release",
                "timeout_sec": wait_timeout_sec,
            },
            timeout_sec=max(
                8.0,
                wait_timeout_sec + 5.0,
            ),
        )

        if result is not True:
            raise DynamixelWorkerError(
                "storage release worker did not confirm success: "
                f"{result!r}"
            )

        return True

    def ping_all(self) -> None:

        self._call_with_restart(
            {
                "cmd": "ping_all"
            },
            timeout_sec=5.0,
        )

    def expand_sp_lin(
        self,
        asynchronous: bool = False,
    ) -> "DynamixelWorkerClient":

        self._call_with_restart(
            {
                "cmd": "expand_sp_lin",
                "asynchronous": bool(asynchronous),
            },
            timeout_sec=(
                8.0 if asynchronous else 20.0
            ),
        )

        return self

    def expand_sp_lin_right_tilted(
        self,
        asynchronous: bool = False,
    ) -> "DynamixelWorkerClient":

        self._call_with_restart(
            {
                "cmd": "expand_sp_lin_right_tilted",
                "asynchronous": bool(asynchronous),
            },
            timeout_sec=(
                8.0 if asynchronous else 20.0
            ),
        )

        return self

    def contract_sp_lin_1(
        self,
        asynchronous: bool = False,
    ) -> "DynamixelWorkerClient":

        self._call_with_restart(
            {
                "cmd": "contract_sp_lin_1",
                "asynchronous": bool(asynchronous),
            },
            timeout_sec=(
                8.0 if asynchronous else 20.0
            ),
        )

        return self

    def contract_sp_lin_2_start(
        self,
        timeout_sec: float = 20.0,
    ) -> bool:
        """
        SP_LINの収縮・PWM監視をworker内の
        バックグラウンドスレッドで開始する。
        """
        wait_timeout_sec = float(timeout_sec)

        if not math.isfinite(wait_timeout_sec) or wait_timeout_sec <= 0:
            raise ValueError(
                f"Invalid SP_LIN timeout: {timeout_sec}"
            )

        # Record the attempt BEFORE sending. A lost response may still mean
        # the worker started the actuator.
        with self._command_lock:
            self._sp_lin_start_attempted = True
            result = self._call_once(
                {
                    "cmd": "contract_sp_lin_2_start",
                    "timeout_sec": wait_timeout_sec,
                },
                timeout_sec=8.0,
            )

        if not isinstance(result, dict) or result.get("started") is not True:
            raise DynamixelWorkerError(
                f"SP_LIN background start not confirmed: {result!r}"
            )

        return True


    def emergency_stop_sp_lin(self) -> bool:
        """Request SP_LIN torque OFF; never infer success from a timeout."""
        with self._command_lock:
            self._sp_lin_start_attempted = True
            result = self._call_once(
                {"cmd": "emergency_stop_sp_lin"}, timeout_sec=5.0
            )
        if not isinstance(result, dict) or result.get("torque_off_verified") is not True:
            raise DynamixelWorkerError(
                f"SP_LIN torque OFF not confirmed: {result!r}"
            )
        return True

    def sp_lin_task_status(self) -> dict[str, Any]:
        """
        workerからSP_LINバックグラウンド動作状態を取得する。
        """
        # 状態取得は再起動せず、元のworkerに問い合わせる
        with self._command_lock:
            result = self._call_once(
                {"cmd": "sp_lin_task_status"},
                timeout_sec=5.0,
            )

        if not isinstance(result, dict):
            raise DynamixelWorkerError(
                f"Invalid SP_LIN task status: {result!r}"
            )

        for key in ("running", "completed", "failed", "error", "thread_alive"):
            if key not in result:
                raise DynamixelWorkerError(
                    f"Missing SP_LIN task status field: {key}"
                )

        return result

    def contract_sp_lin_2(
        self,
        asynchronous: bool = False,
        timeout_sec: float | None = None,
    ) -> "DynamixelWorkerClient":

        wait_timeout_sec = 15.0 if timeout_sec is None else float(timeout_sec)
        if not math.isfinite(wait_timeout_sec) or wait_timeout_sec <= 0.0:
            raise ValueError(
                "contract_sp_lin_2 timeout_sec must be positive: "
                f"{timeout_sec}"
            )

        request = {
            "cmd": "contract_sp_lin_2",
            "asynchronous": bool(asynchronous),
        }
        if timeout_sec is not None:
            request["timeout_sec"] = wait_timeout_sec

        self._call_with_restart(
            request,
            timeout_sec=(
                8.0 if asynchronous
                else wait_timeout_sec + 110.0
            ),
        )

        return self

    def contract_sp_lin_pwm_home(
        self,
        timeout_sec: float = 100.0,
    ) -> dict[str, Any]:
        """
        SP_LINをPWM監視方式で収縮する。
        完了するまで同期的に待つ。
        """
        wait_timeout_sec = float(timeout_sec)

        if (
            not math.isfinite(wait_timeout_sec)
            or wait_timeout_sec <= 0.0
        ):
            raise ValueError(
                "Invalid SP_LIN homing timeout"
            )

        # 原点復帰は非冪等な動作なので、
        # 通信異常時に自動で再実行しない。
        with self._command_lock:
            result = self._call_once(
                {
                    "cmd": "contract_sp_lin_pwm_home",
                    "timeout_sec": wait_timeout_sec,
                },
                timeout_sec=wait_timeout_sec + 10.0,
            )

        if not isinstance(result, dict):
            raise DynamixelWorkerError(
                f"Invalid homing result: {result!r}"
            )

        if result.get("reason") != "high_pwm_detected":
            raise DynamixelWorkerError(
                f"SP_LIN PWM stop not confirmed: {result!r}"
            )

        return result

    def rotate_spacer(
        self,
        theta_deg: float,
    ) -> "DynamixelWorkerClient":

        self._call_with_restart(
            {
                "cmd": "rotate_spacer",
                "theta_deg": float(theta_deg),
            },
            timeout_sec=15.0,
        )

        return self

    def reset_rot(
        self,
        asynchronous: bool = False,
    ) -> "DynamixelWorkerClient":

        self._call_with_restart(
            {
                "cmd": "reset_rot",
                "asynchronous": bool(asynchronous),
            },
            timeout_sec=(
                8.0 if asynchronous else 15.0
            ),
        )

        return self

    def ungrasp_auto(self) -> Any:
        """
        現在位置 + RELEASE_GAIN の相対操作なので、
        応答喪失時に同じ命令を自動再実行しない。
        """
        with self._command_lock:
            return self._call_once(
                {
                    "cmd": "ungrasp_auto"
                },
                timeout_sec=15.0,
            )

    def __enter__(
        self,
    ) -> "DynamixelWorkerClient":
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:
        self.close()
