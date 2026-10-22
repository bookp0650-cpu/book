from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from detection.pro_handbook.thesis.gpu_memory_monitor import record_phase


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

OCR_DIR = (
    BASE_DIR
    / "OCR"
)

OCR_PY = (
    OCR_DIR
    / ".paadle_ocr"
    / "bin"
    / "python"
)

OCR_WORKER_SCRIPT = (
    OCR_DIR
    / "paddle_ocr_worker.py"
)

SOCKET_PATH = Path(
    f"/tmp/"
    f"pro_book_paddle_ocr_worker_"
    f"{os.getpid()}.sock"
)

_worker_proc = None
_worker_started_monotonic = None
_worker_lock = threading.Lock()


def _reload_each_sequence() -> bool:
    """Return True when the caller requires a fresh OCR worker per job."""

    return (
        os.environ.get(
            "BOOK_OCR_RELOAD_EACH_SEQUENCE",
            "0",
        )
        == "1"
    )


@dataclass
class OCRJob:
    sock: socket.socket
    mode: str
    worker_started_monotonic: float | None = None
    timing: dict | None = None
    context: str | None = None


def _build_worker_env():

    env = os.environ.copy()

    env.pop(
        "LD_LIBRARY_PATH",
        None,
    )

    env[
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"
    ] = "True"

    env[
        "PADDLE_PDX_LOCAL_FONT_FILE_PATH"
    ] = (
        "/home/book/.paddlex/"
        "fonts/simfang.ttf"
    )

    env[
        "CUDA_VISIBLE_DEVICES"
    ] = "0"

    # Production OCR is GPU-first and must not silently fall back to CPU.
    # Explicit BOOK_OCR_DEVICE=cpu remains available for controlled benchmarks.
    env.setdefault(
        "BOOK_OCR_DEVICE",
        "gpu",
    )
    env[
        "PADDLE_PDX_DISABLE_DEVICE_FALLBACK"
    ] = "True"

    return env


def _worker_alive():

    global _worker_proc

    return (
        _worker_proc is not None
        and _worker_proc.poll()
        is None
    )


def ensure_ocr_worker():

    global _worker_proc, _worker_started_monotonic

    # The reload variant must never inherit a worker from an earlier request.
    # In the normal variant this branch is disabled and the legacy persistent
    # worker behavior is kept unchanged.
    if _reload_each_sequence():
        shutdown_ocr_worker()

    with _worker_lock:

        if (
            _worker_alive()
            and SOCKET_PATH.exists()
        ):
            return

        if not OCR_PY.exists():
            raise FileNotFoundError(
                f"OCR python not found: "
                f"{OCR_PY}"
            )

        if not OCR_WORKER_SCRIPT.exists():
            raise FileNotFoundError(
                f"OCR worker not found: "
                f"{OCR_WORKER_SCRIPT}"
            )

        try:
            SOCKET_PATH.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        if _reload_each_sequence():
            print(
                "[OCR WORKER] starting new worker"
            )
        else:
            print(
                "[OCR WORKER CLIENT] "
                "starting persistent OCR worker..."
            )

        _worker_started_monotonic = time.monotonic()
        _worker_proc = subprocess.Popen(
            [
                str(OCR_PY),
                str(OCR_WORKER_SCRIPT),
                str(SOCKET_PATH),
            ],
            cwd=str(
                OCR_DIR
            ),
            env=_build_worker_env(),
        )
        record_phase(
            "ocr_worker_start",
            monotonic_seconds=_worker_started_monotonic,
            worker_pid=_worker_proc.pid,
            reload_each_sequence=_reload_each_sequence(),
        )

        deadline = (
            time.monotonic()
            + 10.0
        )

        while (
            time.monotonic()
            < deadline
        ):

            if (
                _worker_proc.poll()
                is not None
            ):
                raise RuntimeError(
                    "OCR worker terminated "
                    f"early: "
                    f"returncode="
                    f"{_worker_proc.returncode}"
                )

            if SOCKET_PATH.exists():
                if _reload_each_sequence():
                    print(
                        "[OCR WORKER] "
                        f"pid={_worker_proc.pid}"
                    )
                else:
                    print(
                        "[OCR WORKER CLIENT] "
                        "worker socket ready"
                    )
                return

            time.sleep(
                0.02
            )

        raise TimeoutError(
            "OCR worker socket "
            "startup timeout"
        )


def _send_request(
    request: dict,
) -> OCRJob:

    sock = None

    try:
        ensure_ocr_worker()

        sock = socket.socket(
            socket.AF_UNIX,
            socket.SOCK_STREAM,
        )

        sock.connect(
            str(SOCKET_PATH)
        )

        payload = (
            json.dumps(
                request,
                ensure_ascii=False,
            )
            + "\n"
        ).encode(
            "utf-8"
        )

        sock.sendall(
            payload
        )

        return OCRJob(
            sock=sock,
            mode=str(
                request.get(
                    "mode",
                    "",
                )
            ),
            worker_started_monotonic=(
                _worker_started_monotonic
            ),
            context=str(
                request.get("shot_dir")
                or request.get("image_path")
                or ""
            ),
        )

    except BaseException:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

        if _reload_each_sequence():
            shutdown_ocr_worker()

        raise


def _recv_response(
    job: OCRJob,
    timeout: float | None,
):

    sock = job.sock

    if timeout is not None:
        sock.settimeout(
            float(timeout)
        )

    buf = bytearray()

    try:

        while True:

            chunk = sock.recv(
                4096
            )

            if not chunk:
                break

            buf.extend(
                chunk
            )

            if b"\n" in buf:
                break

    finally:

        try:
            sock.close()
        except Exception:
            pass

    if not buf:
        raise RuntimeError(
            "OCR worker returned "
            "empty response"
        )

    line = bytes(
        buf
    ).split(
        b"\n",
        1,
    )[0]

    response = json.loads(
        line.decode(
            "utf-8"
        )
    )

    if not response.get(
        "ok",
        False,
    ):
        raise RuntimeError(
            "OCR worker failed:\n"
            f"{response.get('error')}\n"
            f"{response.get('traceback')}"
        )

    return response


def _apply_response_timing(job: OCRJob, response: dict) -> dict:
    timing = dict(response.get("timing") or {})
    model_ready = timing.get("model_ready_monotonic")
    if job.worker_started_monotonic is not None and model_ready is not None:
        timing["worker_start_to_model_ready_seconds"] = max(
            0.0,
            float(model_ready) - float(job.worker_started_monotonic),
        )
    job.timing = timing

    worker_pid = response.get("worker_pid")
    common = {"worker_pid": worker_pid, "context": job.context, "mode": job.mode}
    phase_fields = []
    if _reload_each_sequence() or job.mode == "ready":
        phase_fields.extend(
            [
                ("ocr_model_load_start", "model_load_start_monotonic"),
                ("ocr_model_ready", "model_ready_monotonic"),
            ]
        )
    phase_fields.extend(
        [
            ("ocr_inference_start", "inference_start_monotonic"),
            ("ocr_inference_end", "inference_end_monotonic"),
        ]
    )
    for phase, field in phase_fields:
        value = timing.get(field)
        if value is not None:
            record_phase(phase, monotonic_seconds=float(value), **common)
    return timing


def initialize_ocr_worker(*, timeout: float = 120.0) -> dict:
    """Start the persistent worker and wait until its model is ready."""
    if _reload_each_sequence():
        raise RuntimeError(
            "initialize_ocr_worker requires BOOK_OCR_RELOAD_EACH_SEQUENCE=0"
        )
    started = time.perf_counter()
    job = _send_request({"mode": "ready"})
    response = _recv_response(job, timeout)
    timing = _apply_response_timing(job, response)
    timing["initialization_seconds"] = time.perf_counter() - started
    timing["worker_pid"] = response.get("worker_pid")
    timing["runtime"] = (response.get("data") or {}).get("runtime")
    record_phase(
        "ocr_initialization_complete",
        worker_pid=response.get("worker_pid"),
        initialization_seconds=timing["initialization_seconds"],
    )
    return timing


def start_book_ocr(
    shot_dir: Path,
):
    """
    非同期開始。
    今までのstart_ocr_subprocess相当。
    """

    return _send_request(
        {
            "mode": "book",
            "shot_dir": str(
                Path(
                    shot_dir
                ).resolve()
            ),
        }
    )


def wait_book_ocr(
    job: OCRJob,
    *,
    timeout: float | None = None,
) -> str:
    """
    今までのwait_ocr_subprocess相当。
    """

    try:
        response = (
            _recv_response(
                job,
                timeout,
            )
        )

        _apply_response_timing(job, response)

        if _reload_each_sequence():
            print(
                "[OCR WORKER] recognition completed"
            )

        return str(
            response.get(
                "stdout",
                "",
            )
        )
    finally:
        if _reload_each_sequence():
            shutdown_ocr_worker()


def run_barcode_ocr_worker(
    *,
    image_path: Path,
    target,
    out_dir: Path,
    timeout: float = 60.0,
):
    """
    同じ常駐PaddleOCRで
    barcode fallbackを実行。
    """

    job = _send_request(
        {
            "mode": "barcode",
            "image_path": str(
                Path(
                    image_path
                ).resolve()
            ),
            "target": str(
                target
            ),
            "out_dir": str(
                Path(
                    out_dir
                ).resolve()
            ),
        }
    )

    try:
        response = (
            _recv_response(
                job,
                timeout,
            )
        )

        return (
            response.get(
                "data",
                {},
            ),
            str(
                response.get(
                    "stdout",
                    "",
                )
            ),
        )
    finally:
        if _reload_each_sequence():
            shutdown_ocr_worker()


def shutdown_ocr_worker():

    global _worker_proc, _worker_started_monotonic

    if _worker_proc is None:
        return

    worker_proc = _worker_proc
    worker_pid = worker_proc.pid
    log_lifecycle = _reload_each_sequence()

    if log_lifecycle:
        print(
            "[OCR WORKER] "
            f"stopping pid={worker_pid}"
        )

    try:

        if (
            worker_proc.poll()
            is None
        ):

            worker_proc.terminate()

            try:
                worker_proc.wait(
                    timeout=3.0
                )
            except subprocess.TimeoutExpired:
                worker_proc.kill()
                worker_proc.wait(
                    timeout=3.0
                )

    except Exception as exc:
        if log_lifecycle:
            print(
                "[OCR WORKER] "
                f"stop failed pid={worker_pid}: {exc}"
            )

    finally:

        _worker_proc = None
        _worker_started_monotonic = None

        try:
            SOCKET_PATH.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        if log_lifecycle:
            print(
                "[OCR WORKER] "
                f"pid={worker_pid} stopped"
            )
        record_phase(
            "ocr_worker_stopped",
            worker_pid=worker_pid,
        )


atexit.register(
    shutdown_ocr_worker
)
