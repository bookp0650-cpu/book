from __future__ import annotations

import os
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv(
    "SAM3_MODEL_PATH",
    RUNTIME_DIR / "models" / "candidates" / "sam3_bookspine_expE_ck16"
    / "sam3_bookspine_expE_ck16_inference.pt",
))
BOOK_SPINE_MODEL_PATH = MODEL_PATH
BOOK_END_MODEL_PATH = Path(
    os.getenv(
        "SAM3_BOOK_END_MODEL_PATH",
        RUNTIME_DIR / "models" / "checkpoint_50_modelonly.pt",
    )
)
BOOK_SPINE_EXPECTED_SHA256 = os.getenv(
    "SAM3_BOOK_SPINE_EXPECTED_SHA256",
    "1b43d2906da3f11d9e49c056b169f567762fb96cf91579df0110f66fa27afcdd",
)
BOOK_END_EXPECTED_SHA256 = (
    "a1a0c32fc6f2e2f9d10612ac55d6e851566b0a26c8dc2d83b9e6ae0ac1a3e83c"
)
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
