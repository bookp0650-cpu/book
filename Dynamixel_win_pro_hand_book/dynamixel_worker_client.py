#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time

from pathlib import Path
from typing import Any


RESPONSE_PREFIX = "__DXL_WORKER_JSON__"


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

        proc = self._proc
        stdout_thread = self._stdout_thread

        self._proc = None
        self._stdout_queue = None
        self._stdout_thread = None

        if proc is None:
            return

        if proc.poll() is None:

            try:
                print(
                    "[DXL CLIENT] "
                    "terminating old worker: "
                    f"pid={proc.pid}"
                )

                proc.terminate()

                proc.wait(
                    timeout=2.0
                )

            except Exception:

                try:
                    print(
                        "[DXL CLIENT] "
                        "SIGTERM timeout; "
                        "killing worker: "
                        f"pid={proc.pid}"
                    )

                    proc.kill()

                    proc.wait(
                        timeout=2.0
                    )

                except Exception:
                    pass

        for stream in (
            proc.stdin,
            proc.stdout,
        ):
            if stream is not None:

                try:
                    stream.close()

                except Exception:
                    pass

        if (
            stdout_thread is not None
            and stdout_thread.is_alive()
        ):
            stdout_thread.join(
                timeout=1.0
            )

    # =====================================================
    # worker再起動
    # =====================================================
    def restart(self) -> None:

        with self._command_lock:

            print(
                "[DXL CLIENT] "
                "===== WORKER HARD RESTART ====="
            )

            # まず旧workerを完全終了
            self._hard_stop()

            # fresh Python workerを複数回試す
            #
            # 重要:
            # 同じworkerオブジェクトを再利用するのではなく、
            # 毎回完全に新しいPythonプロセスを起動する。
            wait_schedule = (
                self._restart_wait_sec,  # 現在 3秒
                3.0,
                5.0,
            )

            last_exc = None

            for attempt, wait_sec in enumerate(
                wait_schedule,
                start=1,
            ):
                print(
                    "[DXL CLIENT] "
                    f"fresh worker restart attempt="
                    f"{attempt}/{len(wait_schedule)}, "
                    f"waiting {wait_sec:.1f} sec..."
                )

                time.sleep(wait_sec)

                try:
                    self.start()

                    print(
                        "[DXL CLIENT] "
                        "===== WORKER RESTARTED ===== "
                        f"attempt={attempt}"
                    )

                    return

                except Exception as exc:
                    last_exc = exc

                    print(
                        "[DXL CLIENT] "
                        "fresh worker startup failed: "
                        f"attempt={attempt}/"
                        f"{len(wait_schedule)}, "
                        f"error={type(exc).__name__}: "
                        f"{exc}"
                    )

                    # start()内でも失敗時hard_stopされるが、
                    # 念のため完全終了を保証
                    self._hard_stop()

            raise DynamixelWorkerError(
                "Dynamixel worker could not be restarted "
                "after multiple fresh-process attempts: "
                f"{type(last_exc).__name__}: {last_exc}"
            )
    # =====================================================
    # 終了
    # =====================================================

    def close(self) -> None:

        with self._command_lock:

            proc = self._proc

            if proc is None:
                return

            if proc.poll() is None:

                try:
                    self._call_once(
                        {
                            "cmd": "shutdown"
                        },
                        timeout_sec=3.0,
                    )

                except Exception as exc:
                    print(
                        "[DXL CLIENT] "
                        "worker shutdown warning: "
                        f"{type(exc).__name__}: {exc}"
                    )

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

        with self._command_lock:

            command = str(
                request.get(
                    "cmd",
                    "unknown",
                )
            )

            try:
                return self._call_once(
                    request,
                    timeout_sec=timeout_sec,
                )

            except Exception as first_exc:

                # ==========================================
                # 非通信エラーなら絶対にworker再起動しない
                # ==========================================
                if (
                    isinstance(
                        first_exc,
                        DynamixelWorkerCommandError,
                    )
                    and not first_exc.restart_worker
                ):

                    print(
                        "[DXL CLIENT] "
                        "non-communication error. "
                        "Worker restart is NOT performed: "
                        f"cmd={command}, "
                        f"error={first_exc}"
                    )

                    raise

                # ==========================================
                # 通信異常 or worker/IPC異常
                # ==========================================
                print(
                    "[DXL CLIENT] "
                    "restartable failure detected: "
                    f"cmd={command}, "
                    f"error="
                    f"{type(first_exc).__name__}: "
                    f"{first_exc}"
                )

                print(
                    "[DXL CLIENT] "
                    "old worker will be completely "
                    "killed and a fresh Python worker "
                    "will be started"
                )

                try:
                    self.restart()

                except Exception as restart_exc:
                    raise DynamixelWorkerError(
                        "failed to restart "
                        "Dynamixel worker after "
                        f"{command} failure: "
                        f"{restart_exc}"
                    ) from restart_exc

                # ==========================================
                # open_until_widthのみ閉状態に正規化
                # ==========================================
                if normalize_closed_before_retry:

                    print(
                        "[DXL CLIENT] "
                        "normalizing gripper "
                        "to closed state "
                        "before retry..."
                    )

                    try:
                        self._call_once(
                            {
                                "cmd": "grasp",
                                "timeout_sec": 3.0,
                            },
                            timeout_sec=8.0,
                        )

                    except Exception as normalize_exc:
                        raise DynamixelWorkerError(
                            "failed to normalize "
                            "gripper after worker "
                            "restart: "
                            f"{normalize_exc}"
                        ) from normalize_exc

                print(
                    "[DXL CLIENT] "
                    "retrying command after "
                    "worker restart: "
                    f"cmd={command}"
                )

                # 再試行は1回だけ
                try:
                    result = self._call_once(
                        request,
                        timeout_sec=timeout_sec,
                    )

                except Exception as second_exc:
                    raise DynamixelWorkerError(
                        f"{command} failed even "
                        "after one worker restart: "
                        f"{type(second_exc).__name__}: "
                        f"{second_exc}"
                    ) from second_exc

                print(
                    "[DXL CLIENT] "
                    "command recovered after "
                    "worker restart: "
                    f"cmd={command}"
                )

                return result

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

    def contract_sp_lin_2(
        self,
        asynchronous: bool = False,
    ) -> "DynamixelWorkerClient":

        self._call_with_restart(
            {
                "cmd": "contract_sp_lin_2",
                "asynchronous": bool(asynchronous),
            },
            timeout_sec=(
                8.0 if asynchronous else 20.0
            ),
        )

        return self

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