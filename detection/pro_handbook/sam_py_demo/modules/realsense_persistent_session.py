#!/usr/bin/env python3
"""Persistent RealSense lifecycle for live book-recognition captures."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pyrealsense2 as rs


def _device_info(device, key):
    try:
        return device.get_info(key) if device.supports(key) else None
    except Exception:
        return None


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class RealSensePersistentSession:
    """Start one aligned RGB-D pipeline and reuse it until program shutdown."""

    def __init__(
        self,
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 6,
        warmup_frame_count: int = 10,
        max_stale_frames_to_drain: int = 32,
    ):
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("width, height, and fps must be positive")
        if warmup_frame_count < 0:
            raise ValueError("warmup_frame_count must be non-negative")
        if max_stale_frames_to_drain < 0:
            raise ValueError(
                "max_stale_frames_to_drain must be non-negative"
            )

        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.warmup_frame_count = int(warmup_frame_count)
        self.max_stale_frames_to_drain = int(
            max_stale_frames_to_drain
        )

        self.start_count = 0
        self.capture_count = 0
        self.stop_count = 0

        self._pipeline = None
        self._config = None
        self._align = None
        self._profile = None
        self._started = False
        self._ever_started = False
        self._lock = threading.RLock()

    @property
    def is_started(self) -> bool:
        return self._started

    def start(self) -> None:
        """Create and start the configured pipeline exactly once."""
        with self._lock:
            if self._started:
                raise RuntimeError(
                    "persistent RealSense session is already started"
                )
            if self._ever_started:
                raise RuntimeError(
                    "persistent RealSense session cannot be restarted"
                )

            print("[CAMERA] persistent RealSense start")
            self._pipeline = rs.pipeline()
            self._config = rs.config()
            self._config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.bgr8,
                self.fps,
            )
            self._config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps,
            )
            self._align = rs.align(rs.stream.color)

            self._profile = self._pipeline.start(self._config)
            self._started = True
            self._ever_started = True
            self.start_count += 1

            for _ in range(self.warmup_frame_count):
                self._pipeline.wait_for_frames()
            print(
                "[CAMERA] warm-up complete: "
                f"discarded={self.warmup_frame_count}"
            )

    def capture(
        self,
        shot_dir: str | Path,
        *,
        depth_filter: Callable,
    ) -> tuple[np.ndarray, np.ndarray, object, float, dict]:
        """Save one post-request aligned RGB-D pair without stopping."""
        with self._lock:
            if not self._started:
                raise RuntimeError(
                    "persistent RealSense session is not started"
                )
            if (
                self._pipeline is None
                or self._align is None
                or self._profile is None
            ):
                raise RuntimeError(
                    "persistent RealSense session state is incomplete"
                )

            shot_dir = Path(shot_dir).expanduser().resolve()
            if not shot_dir.is_dir():
                raise FileNotFoundError(
                    f"capture output directory does not exist: {shot_dir}"
                )
            if (shot_dir / "after_init_rgb.png").exists() or (
                shot_dir / "after_init_depth.npy"
            ).exists():
                raise FileExistsError(
                    f"refusing to overwrite captured RGB-D in {shot_dir}"
                )

            capture_requested_at = datetime.now().astimezone()
            request_monotonic = time.perf_counter()
            print(
                "[CAMERA] capture requested: "
                f"count={self.capture_count + 1}"
            )

            stale_frame_count = 0
            while (
                stale_frame_count < self.max_stale_frames_to_drain
            ):
                pending_frames = self._pipeline.poll_for_frames()
                if not pending_frames:
                    break
                stale_frame_count += 1

            frames = self._pipeline.wait_for_frames()
            aligned = self._align.process(frames)
            depth_frame = depth_filter(aligned.get_depth_frame())
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                raise RuntimeError(
                    "aligned RealSense color/depth frame is unavailable"
                )

            color_np = np.asanyarray(color_frame.get_data())
            depth_np = np.asanyarray(depth_frame.get_data())
            expected_rgb_shape = (self.height, self.width, 3)
            expected_depth_shape = (self.height, self.width)
            if (
                color_np.shape != expected_rgb_shape
                or color_np.dtype != np.uint8
            ):
                raise RuntimeError(
                    "unexpected RGB frame: "
                    f"shape={color_np.shape} dtype={color_np.dtype}"
                )
            if (
                depth_np.shape != expected_depth_shape
                or depth_np.dtype != np.uint16
            ):
                raise RuntimeError(
                    "unexpected Depth frame: "
                    f"shape={depth_np.shape} dtype={depth_np.dtype}"
                )

            depth_profile = rs.video_stream_profile(
                depth_frame.get_profile()
            )
            intr = depth_profile.get_intrinsics()
            device = self._profile.get_device()
            depth_scale = float(
                device.first_depth_sensor().get_depth_scale()
            )
            cv2.imwrite(
                str(shot_dir / "after_init_rgb.png"),
                color_np,
            )
            np.save(shot_dir / "after_init_depth.npy", depth_np)

            intrinsics_payload = {
                "width": int(intr.width),
                "height": int(intr.height),
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "ppx": float(intr.ppx),
                "ppy": float(intr.ppy),
                "model": str(intr.model),
                "coeffs": [float(value) for value in intr.coeffs],
                "depth_scale": depth_scale,
            }
            _write_json(
                shot_dir / "camera_intrinsics.json",
                intrinsics_payload,
            )

            self.capture_count += 1
            capture_completed_at = datetime.now().astimezone()
            metadata = {
                "capture_timestamp": capture_requested_at.isoformat(),
                "capture_completed_timestamp": (
                    capture_completed_at.isoformat()
                ),
                "capture_count": 1,
                "session_capture_count": self.capture_count,
                "warmup_frame_count": self.warmup_frame_count,
                "warmup_scope": "session_start_once",
                "stale_frames_drained_after_request": stale_frame_count,
                "capture_request_monotonic_seconds": request_monotonic,
                "color_frame_number": int(
                    color_frame.get_frame_number()
                ),
                "color_frame_timestamp_ms": float(
                    color_frame.get_timestamp()
                ),
                "depth_frame_number": int(
                    depth_frame.get_frame_number()
                ),
                "depth_frame_timestamp_ms": float(
                    depth_frame.get_timestamp()
                ),
                "rgb_shape": list(color_np.shape),
                "rgb_dtype": str(color_np.dtype),
                "depth_shape": list(depth_np.shape),
                "depth_dtype": str(depth_np.dtype),
                "rgb_format": "bgr8",
                "depth_format": "z16",
                "depth_unit": "meter per Z16 count",
                "depth_scale": depth_scale,
                "camera_intrinsics": intrinsics_payload,
                "aligned_depth_to_color": True,
                "realsense_device_name": _device_info(
                    device,
                    rs.camera_info.name,
                ),
                "serial_number": _device_info(
                    device,
                    rs.camera_info.serial_number,
                ),
                "firmware_version": _device_info(
                    device,
                    rs.camera_info.firmware_version,
                ),
                "requested_stream": {
                    "width": self.width,
                    "height": self.height,
                    "fps": self.fps,
                    "color": "bgr8",
                    "depth": "z16",
                },
                "persistent_session": {
                    "start_count": self.start_count,
                    "capture_count": self.capture_count,
                    "stop_count": self.stop_count,
                    "pipeline_kept_active": True,
                },
            }
            _write_json(
                shot_dir / "realsense_capture_metadata.json",
                metadata,
            )

            print(
                "[CAMERA] capture complete: "
                f"count={self.capture_count}, "
                f"stale_frames_drained={stale_frame_count}"
            )
            print("[CAMERA] keeping pipeline active")
            return color_np, depth_np, intr, depth_scale, metadata

    def stop(self) -> None:
        """Stop an active pipeline; repeated calls are safe no-ops."""
        with self._lock:
            if not self._started:
                return

            print("[CAMERA] persistent RealSense stop")
            try:
                self._pipeline.stop()
            finally:
                self._started = False
                self.stop_count += 1
                print(
                    "[CAMERA] lifecycle counts: "
                    f"start_count={self.start_count}, "
                    f"capture_count={self.capture_count}, "
                    f"stop_count={self.stop_count}"
                )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False
