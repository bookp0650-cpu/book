# Test report (2026-07-23)

- Paths and old/new detection trees verified; excluding the copied OCR venv, `diff -qr` reported no differences before migration.
- Checkpoint SHA-256 verified before and after moving.
- Checkpoint inspected successfully with CPU `torch.load(..., weights_only=True, mmap=True)`.
- Python syntax tests: see migration report/final work report.
- GPU: RTX 5070 Ti, driver 580.159.03. PyTorch and Paddle CUDA tensor tests pass with the legacy CUDA 12.2 `LD_LIBRARY_PATH` removed.
- Strict load: success, zero missing/unexpected keys.
- Fixed image: 1280x720 uint8 BGR from OpenCV, losslessly converted to RGB; prompt `book spine`.
- The owner's branch `codex-stage-e7-online-augmentation-20260715` was captured at commit `08553cddd6a7833fecf4e99f7f4418d34490a4da`. Its actual `run_unified_inference -> run_sam3_image_directory -> Sam3Adapter -> load_inference_checkpoint -> Sam3Processor` route succeeded on the fixed PNG with the specified defaults.
- Owner route: 20 raw masks and 20 post-NMS masks (none suppressed), `[20,720,1280]` bool. Model load 18.229 s, inference 0.519 s, NMS 0.022 s.
- Owner-aligned standalone/service/`get_book_points.py` adapter: 20 bool masks shaped `[20,720,1280]`; score-sorted selected index 0, area 21,860, score 0.9609375, mask-derived XYXY `[611,95,777,505]`.
- Selected-mask IoU: standalone/service 1.0; standalone/adapter 1.0.
- The service output arrays are byte-for-byte equal to the owner's post-NMS masks. Owner-aligned standalone load 14.607 s, warm-up 0.514 s, warm inference 0.308 s, peak GPU 5088.9 MiB. Service inference was 0.389/0.379 s; client round trips were 0.486/1.557 s.
- The earlier 19-mask baseline is preserved under `tests/outputs/case_001`; the difference was the processor confidence threshold (previous default versus owner value 0.05), not RGB/BGR, resize, prompt, or checkpoint drift.
- PaddleOCR fixed-image test: success using cached local models.
- Saved RGB-D recognition-only: success; roll 1.607427 rad, width 13.159 mm, 10.089 s.
- Both Retrieval integration modules import successfully without edits.
- Robot commands were not executed.
