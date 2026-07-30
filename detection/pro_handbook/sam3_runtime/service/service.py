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

from .model_loader import load_model
from .predictor import infer_array, infer_to_npz
from .settings import FIXED_TEST_IMAGE, HOST, LOG_DIR, MODEL_PATH, PORT, PROMPT

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(LOG_DIR / "service.log"), logging.StreamHandler()])
LOG = logging.getLogger("sam3-service")
STATE = {"loaded": None, "error": None, "started": time.time(), "loads": 0, "warmup": None}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        raw = json.dumps(payload).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            self._send(200 if STATE["loaded"] else 503, {"ready": bool(STATE["loaded"]), "error": STATE["error"], "model": str(MODEL_PATH), "model_loads": STATE["loads"], "warmup": STATE["warmup"], "uptime_seconds": time.time() - STATE["started"]})
        elif self.path == "/model-info":
            loaded = STATE["loaded"]
            self._send(200, {"model": str(MODEL_PATH), "device": getattr(loaded, "device", None), "model_loads": STATE["loads"], "model_load_seconds": getattr(loaded, "model_load_seconds", None), "strict_load": getattr(loaded, "strict_load", False), "missing_keys": getattr(loaded, "missing_keys", None), "unexpected_keys": getattr(loaded, "unexpected_keys", None)})
        else: self._send(404, {"error": "not found"})

    def do_POST(self):
        request_id = self.headers.get("X-Request-ID", str(uuid.uuid4()))
        if self.path != "/infer": return self._send(404, {"request_id": request_id, "error": "not found"})
        if not STATE["loaded"]: return self._send(503, {"request_id": request_id, "error": STATE["error"] or "model not loaded"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 65536: raise ValueError("invalid request size")
            body = json.loads(self.rfile.read(size))
            inp, out = Path(body["input_npy"]).resolve(), Path(body["output_npz"]).resolve()
            allowed = (Path("/tmp").resolve(), Path("/dev/shm").resolve())
            if not any(inp.is_relative_to(p) and out.is_relative_to(p) for p in allowed): raise ValueError("I/O paths must be under /tmp or /dev/shm")
            meta = infer_to_npz(STATE["loaded"], inp, out, str(body.get("prompt", "book spine")))
            LOG.info("request_id=%s count=%s seconds=%.3f", request_id, meta["count"], meta["inference_seconds"])
            self._send(200, {"request_id": request_id, "output_npz": str(out), **meta})
        except Exception as exc:
            LOG.exception("request_id=%s failed", request_id); self._send(400, {"request_id": request_id, "error": str(exc)})

    def log_message(self, fmt, *args): LOG.info(fmt, *args)


def main():
    try:
        loaded = load_model(); STATE["loads"] += 1
        fixed_rgb = np.asarray(Image.open(FIXED_TEST_IMAGE).convert("RGB"), dtype=np.uint8)
        _, _, _, STATE["warmup"] = infer_array(loaded, fixed_rgb, PROMPT)
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
