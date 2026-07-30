"""Ownership-aware SAM3 service lifecycle for the retrieval integration."""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_DIR = Path(__file__).resolve().parent
START_SCRIPT = RUNTIME_DIR / "scripts" / "start_service.sh"
STOP_SCRIPT = RUNTIME_DIR / "scripts" / "stop_service.sh"
PID_FILE = RUNTIME_DIR / "logs" / "service.pid"
DEFAULT_ENDPOINT = "http://127.0.0.1:8765"
READY_TIMEOUT_SECONDS = 120.0

os.environ.setdefault("BOOK_SEGMENTATION_BACKEND", "sam3")
os.environ.setdefault("SAM3_ENDPOINT", DEFAULT_ENDPOINT)
existing_pythonpath = os.environ.get("PYTHONPATH", "")
pythonpath_parts = [part for part in existing_pythonpath.split(os.pathsep) if part]
if str(PROJECT_ROOT) not in pythonpath_parts:
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), *pythonpath_parts]
    )


def _health(endpoint: str, timeout: float = 1.0) -> tuple[bool, bool, dict | None]:
    """Return (reachable, ready, payload), including JSON from HTTP 503."""
    url = endpoint.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
        return True, bool(payload.get("ready")), payload
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            try:
                payload = json.load(exc)
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = None
            return True, False, payload
        return False, False, None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return False, False, None


def _read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def _pid_is_running(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


class Sam3ServiceSession:
    """Start SAM3 only when absent and stop only the recorded owned process."""

    def __init__(
        self,
        endpoint: str | None = None,
        ready_timeout: float = READY_TIMEOUT_SECONDS,
    ):
        self.endpoint = endpoint or os.environ["SAM3_ENDPOINT"]
        self.ready_timeout = float(ready_timeout)
        self.started_by_this_process = False
        self.owned_pid: int | None = None
        self.health_payload: dict | None = None

    def ensure_ready(self) -> dict:
        reachable, ready, payload = _health(self.endpoint)
        if ready:
            self.health_payload = payload
            print("[SAM3] using pre-existing ready service")
            return payload or {}

        if reachable:
            print("[SAM3] service is reachable but not ready; waiting")
        else:
            subprocess.run([str(START_SCRIPT)], check=True)
            self.owned_pid = _read_pid()
            if not _pid_is_running(self.owned_pid):
                raise RuntimeError("SAM3 start script did not leave a running PID")
            self.started_by_this_process = True
            print(f"[SAM3] started service PID {self.owned_pid}")

        deadline = time.monotonic() + self.ready_timeout
        last_payload = payload
        while time.monotonic() < deadline:
            reachable, ready, last_payload = _health(self.endpoint)
            if ready:
                if self.started_by_this_process and not _pid_is_running(
                    self.owned_pid
                ):
                    # Another service won a startup race. Do not claim ownership.
                    self.started_by_this_process = False
                    self.owned_pid = None
                self.health_payload = last_payload
                print("[SAM3] service ready")
                return last_payload or {}
            time.sleep(1.0)

        self.stop_if_owned()
        detail = None if last_payload is None else last_payload.get("error")
        raise TimeoutError(
            f"SAM3 service was not ready within {self.ready_timeout:.0f}s"
            + (f": {detail}" if detail else "")
        )

    def stop_if_owned(self) -> bool:
        if not self.started_by_this_process:
            return False
        pid_in_file = _read_pid()
        if pid_in_file != self.owned_pid:
            print(
                "[SAM3] PID file no longer matches owned service; "
                "refusing to stop it"
            )
            self.started_by_this_process = False
            self.owned_pid = None
            return False
        subprocess.run([str(STOP_SCRIPT)], check=True)
        print(f"[SAM3] stopped owned service PID {self.owned_pid}")
        self.started_by_this_process = False
        self.owned_pid = None
        return True

    def __enter__(self):
        self.ensure_ready()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop_if_owned()
        return False
