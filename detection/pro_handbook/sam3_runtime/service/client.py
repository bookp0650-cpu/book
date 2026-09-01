from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from core.npz_io import InstanceSet
except ModuleNotFoundError:
    # The service venv intentionally has only the runtime root on PYTHONPATH.
    # Make the bundled owner inference helpers visible without changing the
    # caller's ROS/RealSense environment.
    owner_repo = Path(__file__).resolve().parents[1] / "vendor" / "owner_repo"
    sys.path.insert(0, str(owner_repo))
    from core.npz_io import InstanceSet


class Sam3ServiceError(RuntimeError): pass


class Sam3BatchInfer:
    """Compatibility facade for the legacy `infer_masks(PIL.Image, ...)` call."""
    def __init__(self, endpoint=None, timeout=None, prompt="book spine"):
        self.endpoint = endpoint or os.getenv("SAM3_ENDPOINT", "http://127.0.0.1:8765")
        self.timeout = float(timeout or os.getenv("SAM3_TIMEOUT", "120")); self.prompt = prompt
        self.last_metadata = None

    def infer_masks(self, image, **kwargs):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        request_id = str(uuid.uuid4())
        base = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"
        with tempfile.TemporaryDirectory(prefix="sam3-", dir=base) as tmp:
            inp, out = os.path.join(tmp, "input.npy"), os.path.join(tmp, "output.npz")
            np.save(inp, rgb, allow_pickle=False)
            body = json.dumps({"input_npy": inp, "output_npz": out, "prompt": self.prompt}).encode()
            req = urllib.request.Request(self.endpoint + "/infer", data=body, headers={"Content-Type": "application/json", "X-Request-ID": request_id})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response: meta = json.load(response)
            except (urllib.error.URLError, TimeoutError) as exc:
                raise Sam3ServiceError(f"SAM3 service unavailable at {self.endpoint} (request {request_id}): {exc}") from exc
            data = np.load(out, allow_pickle=False)
            masks, boxes, scores = data["masks"].astype(bool), data["boxes"], data["scores"]
        sam_data = [{"name": f"book{i+1}", "score": float(scores[i]), "box": {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3])}} for i, b in enumerate(boxes)]
        self.last_metadata = meta
        stage_save = kwargs.get("stage_save")
        output_dir = getattr(stage_save, "out_dir", None)
        if output_dir is not None:
            output_dir = os.fspath(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(output_dir, "sam3_service_masks.npz"),
                masks=masks,
                boxes=boxes.astype(np.float32, copy=False),
                scores=scores.astype(np.float32, copy=False),
            )
            overlay = rgb.copy()
            colors = np.asarray(
                [[255, 64, 64], [64, 255, 64], [64, 128, 255], [255, 192, 64]],
                dtype=np.float32,
            )
            for index, mask in enumerate(masks):
                color = colors[index % len(colors)]
                overlay[mask] = (0.58 * overlay[mask] + 0.42 * color).astype(np.uint8)
            Image.fromarray(overlay, mode="RGB").save(
                os.path.join(output_dir, "sam3_all_masks_overlay.png")
            )
            with open(
                os.path.join(output_dir, "sam3_service_inference.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "prompt": self.prompt,
                        "raw_mask_count": meta.get("raw_count"),
                        "nms_mask_count": meta.get("nms_count", len(masks)),
                        "service_metadata": meta,
                        "instances": sam_data,
                        "mask_areas_px": [int(mask.sum()) for mask in masks],
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
        return list(masks), sam_data


def _npz_to_instances(path: str) -> InstanceSet:
    with np.load(path, allow_pickle=False) as data:
        masks = data["masks"].astype(bool, copy=False)
        boxes_xyxy = data["boxes"].astype(np.float32, copy=False)
        scores = data["scores"].astype(np.float32, copy=False)
    bboxes_xywh = boxes_xyxy.copy()
    if len(bboxes_xywh):
        bboxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0] + 1
        bboxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1] + 1
    return InstanceSet(
        masks=masks,
        bboxes=bboxes_xywh.astype(np.int32),
        scores=scores,
        instance_ids=np.arange(1, len(masks) + 1, dtype=np.int32),
        extra={},
    )


class Sam3StorageClient:
    """Client for one-image, two-resident-model sequential storage inference."""

    def __init__(self, endpoint=None, timeout=None):
        self.endpoint = endpoint or os.getenv(
            "SAM3_ENDPOINT",
            "http://127.0.0.1:8765",
        )
        self.timeout = float(timeout or os.getenv("SAM3_TIMEOUT", "120"))
        self.last_metadata = None

    def health(self) -> dict:
        try:
            with urllib.request.urlopen(
                self.endpoint.rstrip("/") + "/health",
                timeout=min(self.timeout, 5.0),
            ) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise Sam3ServiceError(
                f"SAM3 service unavailable at {self.endpoint}: {exc}"
            ) from exc

    def infer_instances(
        self,
        rgb: np.ndarray,
    ) -> tuple[InstanceSet, InstanceSet, dict]:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"expected RGB uint8 HxWx3 array, got {image.shape} {image.dtype}"
            )
        request_id = str(uuid.uuid4())
        base = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"
        with tempfile.TemporaryDirectory(prefix="sam3-storage-", dir=base) as tmp:
            inp = os.path.join(tmp, "input.npy")
            spine_out = os.path.join(tmp, "book_spine.npz")
            end_out = os.path.join(tmp, "book_end.npz")
            np.save(inp, image, allow_pickle=False)
            body = json.dumps(
                {
                    "input_npy": inp,
                    "book_spine_output_npz": spine_out,
                    "book_end_output_npz": end_out,
                }
            ).encode()
            request = urllib.request.Request(
                self.endpoint.rstrip("/") + "/infer-storage",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Request-ID": request_id,
                },
            )
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout,
                ) as response:
                    metadata = json.load(response)
            except (urllib.error.URLError, TimeoutError) as exc:
                raise Sam3ServiceError(
                    f"SAM3 storage service unavailable at {self.endpoint} "
                    f"(request {request_id}): {exc}"
                ) from exc
            spine_instances = _npz_to_instances(spine_out)
            end_instances = _npz_to_instances(end_out)
        self.last_metadata = metadata
        return spine_instances, end_instances, metadata
