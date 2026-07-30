# Environment setup

The integration environment is deliberately separate from SAM3. SAM3 uses local micromamba Python 3.12.13 in `.venv`; the earlier Python 3.10 venv is preserved as `.venv.python310_backup`.

Installed versions match the owner: PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, Triton 3.6.0, NumPy 1.26.4, OpenCV 4.11.0.86, and editable sam3 0.1.0. `LD_LIBRARY_PATH` must be unset because the login environment injects CUDA 12.2 and causes cuBLAS initialization failure. With a clean library path, RTX 5070 Ti CUDA matmul passes.

Offline operation sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. No Hugging Face download is part of startup.
