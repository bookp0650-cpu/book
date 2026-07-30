from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import torch

from .settings import (
    BPE_PATH,
    MODEL_PATH,
    OWNER_REPO_DIR,
    PROCESSOR_CONFIDENCE_THRESHOLD,
    SOURCE_DIR,
)


@dataclass
class LoadedModel:
    model: object
    processor: object
    adapter: object
    device: str
    model_load_seconds: float
    strict_load: bool
    missing_keys: list[str]
    unexpected_keys: list[str]


def load_model() -> LoadedModel:
    builder_file = SOURCE_DIR / "sam3" / "model_builder.py"
    if not builder_file.is_file():
        raise RuntimeError(
            f"matching SAM3 source is missing: {builder_file}; copy the fine-tuning source "
            "at commit 08553cddd6a7833fecf4e99f7f4418d34490a4da, including config and BPE assets"
        )
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"SAM3 checkpoint is missing: {MODEL_PATH}")
    if not BPE_PATH.is_file():
        raise RuntimeError(f"SAM3 BPE vocabulary is missing: {BPE_PATH}")
    owner_adapter_file = OWNER_REPO_DIR / "core" / "sam3_adapter.py"
    if not owner_adapter_file.is_file():
        raise RuntimeError(f"owner inference sources are missing: {owner_adapter_file}")
    started = time.perf_counter()
    sys.path.insert(0, str(OWNER_REPO_DIR))
    sys.path.insert(0, str(SOURCE_DIR))
    from core.sam3_adapter import Sam3Adapter

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Reuse the model owner's actual fail-loud route:
    # Sam3Adapter -> load_inference_checkpoint -> strict=True -> Sam3Processor.
    adapter = Sam3Adapter(
        checkpoint=MODEL_PATH,
        sam3_root=SOURCE_DIR,
        confidence_threshold=PROCESSOR_CONFIDENCE_THRESHOLD,
        dtype_mode="bf16",
        device=device,
    )
    model = adapter.model
    missing: list[str] = []
    unexpected: list[str] = []
    return LoadedModel(
        model=model, processor=adapter.processor, adapter=adapter, device=device,
        model_load_seconds=time.perf_counter() - started, strict_load=True,
        missing_keys=missing, unexpected_keys=unexpected,
    )
