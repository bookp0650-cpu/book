# SAM3 service

The localhost service loads the model once at startup through the owner's `Sam3Adapter`, calls `eval()`, and performs inference under `torch.inference_mode()` and CUDA BF16 autocast. Defaults match the owner route: prompt `book spine`, processor confidence 0.05, score threshold 0.3, min area 200, and mask-IoU NMS 0.5 in suppress mode. Images cross the process boundary losslessly as RGB `uint8` NumPy arrays in `/dev/shm` (or `/tmp`), not JPEG/base64. Results are compressed NPZ: boolean `masks[N,H,W]`, float32 `boxes[N,4]` in mask-derived XYXY order, and float32 `scores[N]`.

Endpoints: `GET /health`, `GET /model-info`, `POST /infer`. Default endpoint is `127.0.0.1:8765`; `SAM3_ENDPOINT`, `SAM3_TIMEOUT`, and `SAM3_TEXT_PROMPT` configure the client. Logs are in `logs/`. Use `start_service.sh`, `check_service.sh`, and `stop_service.sh`. A stopped/unready service raises a clear error; there is no SAM2 fallback.
