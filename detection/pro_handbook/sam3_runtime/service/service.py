from __future__ import annotations

import json
import logging
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from .model_loader import load_storage_models
from .predictor import infer_array, infer_storage_to_npz, infer_to_npz
from .settings import (
    BOOK_END_MODEL_PATH,
    FIXED_TEST_IMAGE,
    HOST,
    LOG_DIR,
    MODEL_PATH,
    PORT,
    PROMPT,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(LOG_DIR / "service.log"), logging.StreamHandler()])
LOG = logging.getLogger("sam3-service")
STATE = {
    "loaded": None,
    "error": None,
    "started": time.time(),
    "loads": {"book_spine": 0, "book_end": 0},
    "requests": {"single": 0, "storage": 0},
    "warmup": None,
}
INFERENCE_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        raw = json.dumps(payload).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            self._send(
                200 if STATE["loaded"] else 503,
                {
                    "ready": bool(STATE["loaded"]),
                    "error": STATE["error"],
                    "capabilities": ["single_model", "storage_dual_model"],
                    "models": {
                        "book_spine": str(MODEL_PATH),
                        "book_end": str(BOOK_END_MODEL_PATH),
                    },
                    "model_loads": STATE["loads"],
                    "request_counts": STATE["requests"],
                    "warmup": STATE["warmup"],
                    "uptime_seconds": time.time() - STATE["started"],
                },
            )
        elif self.path == "/model-info":
            loaded = STATE["loaded"]
            self._send(
                200,
                {
                    "models": {
                        "book_spine": str(MODEL_PATH),
                        "book_end": str(BOOK_END_MODEL_PATH),
                    },
                    "device": getattr(loaded, "device", None),
                    "model_loads": STATE["loads"],
                    "request_counts": STATE["requests"],
                    "total_model_load_seconds": getattr(
                        loaded, "total_model_load_seconds", None
                    ),
                    "gpu_memory_allocated_mb": getattr(
                        loaded, "gpu_memory_allocated_mb", None
                    ),
                    "gpu_memory_reserved_mb": getattr(
                        loaded, "gpu_memory_reserved_mb", None
                    ),
                    "strict_load": {
                        "book_spine": getattr(
                            getattr(loaded, "book_spine", None),
                            "strict_load",
                            False,
                        ),
                        "book_end": getattr(
                            getattr(loaded, "book_end", None),
                            "strict_load",
                            False,
                        ),
                    },
                },
            )
        else: self._send(404, {"error": "not found"})

    def do_POST(self):
        request_id = self.headers.get("X-Request-ID", str(uuid.uuid4()))
        if self.path not in {"/infer", "/infer-storage"}:
            return self._send(
                404,
                {"request_id": request_id, "error": "not found"},
            )
        if not STATE["loaded"]: return self._send(503, {"request_id": request_id, "error": STATE["error"] or "model not loaded"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 65536: raise ValueError("invalid request size")
            body = json.loads(self.rfile.read(size))
            inp = Path(body["input_npy"]).resolve()
            allowed = (Path("/tmp").resolve(), Path("/dev/shm").resolve())
            if self.path == "/infer":
                out = Path(body["output_npz"]).resolve()
                if not any(
                    inp.is_relative_to(path) and out.is_relative_to(path)
                    for path in allowed
                ):
                    raise ValueError("I/O paths must be under /tmp or /dev/shm")
                with INFERENCE_LOCK:
                    meta = infer_to_npz(
                        STATE["loaded"].book_spine,
                        inp,
                        out,
                        str(body.get("prompt", "book spine")),
                    )
                    STATE["requests"]["single"] += 1
                LOG.info(
                    "request_id=%s count=%s seconds=%.3f",
                    request_id,
                    meta["count"],
                    meta["inference_seconds"],
                )
                self._send(
                    200,
                    {"request_id": request_id, "output_npz": str(out), **meta},
                )
                return

            spine_out = Path(body["book_spine_output_npz"]).resolve()
            end_out = Path(body["book_end_output_npz"]).resolve()
            if not any(
                inp.is_relative_to(path)
                and spine_out.is_relative_to(path)
                and end_out.is_relative_to(path)
                for path in allowed
            ):
                raise ValueError("I/O paths must be under /tmp or /dev/shm")
            with INFERENCE_LOCK:
                meta = infer_storage_to_npz(
                    STATE["loaded"],
                    inp,
                    spine_out,
                    end_out,
                )
                STATE["requests"]["storage"] += 1
            LOG.info(
                "request_id=%s spine_count=%s end_count=%s seconds=%.3f",
                request_id,
                meta["book_spine"]["count"],
                meta["book_end"]["count"],
                meta["total_inference_seconds"],
            )
            self._send(
                200,
                {
                    "request_id": request_id,
                    "book_spine_output_npz": str(spine_out),
                    "book_end_output_npz": str(end_out),
                    **meta,
                },
            )
        except Exception as exc:
            LOG.exception("request_id=%s failed", request_id); self._send(400, {"request_id": request_id, "error": str(exc)})

    def log_message(self, fmt, *args): LOG.info(fmt, *args)


def main():
    try:
        loaded = load_storage_models()
        STATE["loads"]["book_spine"] += 1
        STATE["loads"]["book_end"] += 1
        fixed_rgb = np.asarray(Image.open(FIXED_TEST_IMAGE).convert("RGB"), dtype=np.uint8)
        _, _, _, spine_warmup = infer_array(
            loaded.book_spine,
            fixed_rgb,
            PROMPT,
        )
        _, _, _, end_warmup = infer_array(
            loaded.book_end,
            fixed_rgb,
            "book end",
        )
        STATE["warmup"] = {
            "inference_order": ["book_spine", "book_end"],
            "parallel_inference": False,
            "book_spine": spine_warmup,
            "book_end": end_warmup,
        }
        STATE["loaded"] = loaded
    except Exception as exc:
        STATE["error"] = str(exc); LOG.error("model load failed: %s", exc)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    LOG.info("listening on http://%s:%d ready=%s", HOST, PORT, bool(STATE["loaded"]))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("keyboard interrupt received; stopping")
    finally:
        server.server_close()


if __name__ == "__main__": main()
