# PaddleOCR check

The copied `.paadle_ocr` spelling is intentional. The broken copied environment is preserved as `.paadle_ocr.copied_backup`; a clean Python 3.10.12 `.paadle_ocr` now resolves to `/usr/bin/python3.10`.

Core installed versions are PaddlePaddle-GPU 3.2.2 (official cu129 wheel), PaddleOCR 3.3.2, PaddleX 3.3.12, NumPy 2.2.6, opencv-contrib-python 4.10.0.84, and opencv-python 4.12.0.88. Runtime `cv2.__version__` is 4.10.0. CUDA GPU matmul passes. Fixed-image OCR used cached local PP-OCRv5 server det/rec models: creation 1.895 s, prediction 0.807 s, save 0.339 s.

OCR is called through the existing relative resolver in `get_book_points.py`. `get_book_points_revised.py` still contains legacy absolute paths and was not changed because it is not the integration import target.
