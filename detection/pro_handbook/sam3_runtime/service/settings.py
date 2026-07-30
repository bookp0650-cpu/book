from __future__ import annotations

import os
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("SAM3_MODEL_PATH", RUNTIME_DIR / "models" / "inference_best.pt"))
HOST = os.getenv("SAM3_HOST", "127.0.0.1")
PORT = int(os.getenv("SAM3_PORT", "8765"))
PROMPT = os.getenv("SAM3_TEXT_PROMPT", "book spine")
SOURCE_DIR = Path(os.getenv("SAM3_SOURCE_DIR", RUNTIME_DIR / "sam3_source"))
OWNER_REPO_DIR = Path(os.getenv("SAM3_OWNER_REPO_DIR", RUNTIME_DIR / "vendor" / "owner_repo"))
SCORE_THRESHOLD = float(os.getenv("SAM3_SCORE_THRESHOLD", "0.3"))
PROCESSOR_CONFIDENCE_THRESHOLD = float(os.getenv("SAM3_PROCESSOR_CONFIDENCE_THRESHOLD", "0.05"))
MIN_AREA = int(os.getenv("SAM3_MIN_AREA", "200"))
NMS_IOU_THRESHOLD = float(os.getenv("SAM3_NMS_IOU_THRESHOLD", "0.5"))
BPE_PATH = Path(os.getenv("SAM3_BPE_PATH", SOURCE_DIR / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"))
FIXED_TEST_IMAGE = Path(os.getenv("SAM3_FIXED_TEST_IMAGE", RUNTIME_DIR.parents[2] / "captures" / "100test" / "1" / "after_init_rgb.png"))
LOG_DIR = RUNTIME_DIR / "logs"
