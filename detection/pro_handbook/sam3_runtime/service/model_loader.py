from __future__ import annotations

import sys
import time
import gc
import hashlib
from dataclasses import dataclass

import torch

from .settings import (
    BPE_PATH,
    BOOK_END_EXPECTED_SHA256,
    BOOK_END_MODEL_PATH,
    BOOK_SPINE_EXPECTED_SHA256,
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
    checkpoint: str | None = None
    checkpoint_format: str | None = None


@dataclass
class LoadedStorageModels:
    book_spine: LoadedModel
    book_end: LoadedModel
    device: str
    total_model_load_seconds: float
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float


def _verify_checkpoint_sha256(path, expected: str, role: str) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"{role} checkpoint SHA-256 mismatch: expected {expected}, "
            f"got {actual} ({path})"
        )
    return actual


def load_model() -> LoadedModel:
    builder_file = SOURCE_DIR / "sam3" / "model_builder.py"
    if not builder_file.is_file():
        raise RuntimeError(
            f"matching SAM3 source is missing: {builder_file}; copy the fine-tuning source "
            "at commit 08553cddd6a7833fecf4e99f7f4418d34490a4da, including config and BPE assets"
        )
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"SAM3 checkpoint is missing: {MODEL_PATH}")
    _verify_checkpoint_sha256(
        MODEL_PATH,
        BOOK_SPINE_EXPECTED_SHA256,
        "book_spine",
    )
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
        checkpoint=str(MODEL_PATH), checkpoint_format="sam3_inference",
    )


def _load_book_end_model(device: str) -> LoadedModel:
    """Load the model-only book-end checkpoint exactly like the offline runner."""
    if not BOOK_END_MODEL_PATH.is_file():
        raise RuntimeError(f"book-end SAM3 checkpoint is missing: {BOOK_END_MODEL_PATH}")
    _verify_checkpoint_sha256(
        BOOK_END_MODEL_PATH,
        BOOK_END_EXPECTED_SHA256,
        "book_end",
    )
    started = time.perf_counter()
    sys.path.insert(0, str(OWNER_REPO_DIR))
    sys.path.insert(0, str(SOURCE_DIR))
    from core.checkpoint_export import _fresh_model
    from core.sam3_adapter import Sam3Adapter

    payload = torch.load(
        str(BOOK_END_MODEL_PATH),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict) or set(payload) != {"model", "epoch"}:
        keys = list(payload) if isinstance(payload, dict) else type(payload).__name__
        raise RuntimeError(
            "book-end checkpoint is not the expected {'model', 'epoch'} "
            f"model-only wrapper: keys={keys}"
        )
    state_dict = payload["model"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError("book-end model state is empty or is not a dict")
    if not all(torch.is_tensor(value) for value in state_dict.values()):
        raise RuntimeError("book-end model state contains non-tensor values")
    model = _fresh_model(device=device, sam301_root=SOURCE_DIR)
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"strict load of book-end model-only checkpoint failed: {exc}"
        ) from exc
    if missing or unexpected:
        raise RuntimeError(
            f"book-end strict load reported missing={missing}, unexpected={unexpected}"
        )
    del state_dict, payload
    gc.collect()
    adapter = Sam3Adapter.from_model(
        model,
        sam3_root=SOURCE_DIR,
        confidence_threshold=PROCESSOR_CONFIDENCE_THRESHOLD,
        dtype_mode="bf16",
        device=device,
        checkpoint_label=str(BOOK_END_MODEL_PATH),
    )
    return LoadedModel(
        model=adapter.model,
        processor=adapter.processor,
        adapter=adapter,
        device=device,
        model_load_seconds=time.perf_counter() - started,
        strict_load=True,
        missing_keys=[],
        unexpected_keys=[],
        checkpoint=str(BOOK_END_MODEL_PATH),
        checkpoint_format="model_only_epoch_wrapper",
    )


def load_storage_models() -> LoadedStorageModels:
    """Load book-spine and book-end models once into one service process."""
    started = time.perf_counter()
    book_spine = load_model()
    book_end = _load_book_end_model(book_spine.device)
    allocated_mb = 0.0
    reserved_mb = 0.0
    if book_spine.device == "cuda":
        allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    return LoadedStorageModels(
        book_spine=book_spine,
        book_end=book_end,
        device=book_spine.device,
        total_model_load_seconds=time.perf_counter() - started,
        gpu_memory_allocated_mb=allocated_mb,
        gpu_memory_reserved_mb=reserved_mb,
    )
