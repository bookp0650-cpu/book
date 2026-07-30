# Troubleshooting

- `/health` says matching source is missing: put the exact training checkout (commit `08553c...`) below `sam3_source/`, including model builder, config and BPE/tokenizer files.
- Service unavailable: run `check_service.sh`, inspect `logs/service.log` and `logs/service.stdout.log`, and confirm port 8765.
- CUDA unavailable: repair/verify the host NVIDIA driver before changing PyTorch. Do not upgrade the integration environment.
- OCR Python resolves into `/home/book/pro_book`: preserve the copied venv, rebuild the same `.paadle_ocr` path from the measured lock and local models.
- Rollback: stop SAM3 and run the untouched `/home/book/pro_book` system. Do not point the new adapter silently at SAM2.
