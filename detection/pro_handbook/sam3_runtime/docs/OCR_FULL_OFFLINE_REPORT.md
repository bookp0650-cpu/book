# PaddleOCR full-offline migration report

Date: 2026-07-24 (Asia/Tokyo)

## Result

The SAM3 recognition path now starts PaddleOCR with explicit local PP-OCRv5
model directories, validates all required local assets before model
construction, disables PaddleX model-hoster checks with the installed
PaddleX 3.3.12 flag, and blocks Python networking APIs inside the OCR child.
No OS, ROS, SAM3-service, or localhost networking setting was changed.

The fixed-image result is unchanged: both before and after produced 92 OCR
candidates, identical recognized strings, identical confidences, and an
identical `ocr_result.json` SHA-256
`ff60a28261b169063187d75374b065922e410ac01203d830fd5bbad7d5f548ab`.
The generated `ocr_overlay.png` was also byte-identical.

## Previous communication path and measured calls

`get_book_points.py` resolves the OCR interpreter and script, then starts
`paddle_ocr_test.py` using either `subprocess.run` or `subprocess.Popen`.
The production parallel path uses `Popen`, begins SAM3 without waiting, and
later calls `communicate`.

Previously, `paddle_ocr_test.py` constructed `PaddleOCR` from
`ocr_version="PP-OCRv5"` and `lang="japan"`. PaddleOCR converted those values
to model names. PaddleX's official-model manager then built hosters, checked
their connectivity, resolved a cache directory, and retained an automatic
download path when an asset was absent.

The old code set `DISABLE_MODEL_SOURCE_CHECK=True`. PaddleX 3.3.12 does not
read that name. Its installed `paddlex/utils/flags.py` reads
`PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` and
`PADDLE_PDX_LOCAL_FONT_FILE_PATH`.

Before modification, `strace -f -e trace=network` measured:

- 18 non-local external `connect` calls to port 443.
- 4 DNS `connect` calls to `127.0.0.53:53`.
- Host lookups/logs for Hugging Face, AI Studio, ModelScope, and Paddle model
  ecology.
- External addresses included `3.164.110.3`, `3.164.110.77`,
  `3.164.110.114`, `3.164.110.128`, `45.113.194.117`,
  `47.251.62.57`, and `103.235.47.176`, plus IPv6 addresses.

After modification, the same strace test measured zero DNS calls and zero
external connects. The sole `connect` was an unsuccessful local UNIX socket
probe at `/tmp/nvidia-mps/control`; it is not network traffic. One unconnected
AF_INET6 socket creation/bind by the runtime was observed, with no IP connect.

## Offline implementation

`paddle_ocr_test.py` now:

1. Sets the verified PaddleX 3.3.12 flags before importing PaddleOCR:
   `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` and
   `PADDLE_PDX_LOCAL_FONT_FILE_PATH=/home/book/.paddlex/fonts/simfang.ttf`.
2. Adds an audit hook which rejects Python DNS/connect audit events before a
   network syscall can be issued.
3. Validates local directories, readability, non-empty inference files, the
   embedded recognition character dictionary, and the visualization font.
4. Passes `text_detection_model_dir` and `text_recognition_model_dir` as
   absolute paths. It no longer passes a model name, `lang`, or `ocr_version`.
5. Keeps document orientation, unwarping, and text-line orientation disabled.
   No unused model was enabled.
6. Records offline settings and the validated asset manifest in
   `ocr_runtime_info.json`.

`get_book_points.py` applies the two verified settings only to the OCR child
environment. It retains the existing `Popen`/`communicate` structure and does
not affect ROS, SAM3 localhost traffic, or any parent-process network setting.

Installed-code evidence:

- `paddlex/utils/flags.py:60,68-71` defines the local-font and model-source
  flags.
- `paddleocr/_pipelines/ocr.py:62-68,91-117` accepts explicit detection and
  recognition directories and only resolves `lang`/`ocr_version` to model
  names when no names/directories are supplied.
- `paddlex/utils/fonts.py:73-100` uses the local font when set and otherwise
  contains a lazy font-download path.
- `paddlex/inference/utils/official_models.py:556-633` contains hoster
  connectivity and automatic-download paths for official model names.

## Local asset manifest

| Use | Absolute path | Main file SHA-256 |
|---|---|---|
| Text detection | `/home/book/.paddlex/official_models/PP-OCRv5_server_det` | `inference.pdiparams`: `183146fe9d9910352f68482f623bcbbb9fa7b9e8fa1463b9ad288cef00524d2d` |
| Text recognition | `/home/book/.paddlex/official_models/PP-OCRv5_server_rec` | `inference.pdiparams`: `63853f062a5f4089befc16f565a68277618e0da5cb45468b49d11079de0ada77` |
| Character dictionary/config | `/home/book/.paddlex/official_models/PP-OCRv5_server_rec/config.json` | Embedded `PostProcess.character_dict`, 18,383 entries |
| Visualization font | `/home/book/.paddlex/fonts/simfang.ttf` | `521c6f7546b4eb64fa4b0cd604bbd36333a20a57e388c8e2ad2ad07b9e593864` |

Both model directories contain readable, non-empty `inference.json`,
`inference.pdiparams`, and `inference.yml`. Their complete sizes and hashes
are available from `paddle_ocr_test.py --check-offline-assets`.

The missing-asset test exited immediately with code 1 and:

```text
FileNotFoundError: Offline OCR model asset is missing. Network fallback is disabled. Missing path: /tmp/definitely_missing_offline_ocr_model
```

No download was attempted.

## Fixed-image comparison

Input:
`/home/book/pro_book_SAM3/pro_hand_book_python/captures/100test/1/after_init_rgb.png`

| Item | Before | After |
|---|---:|---:|
| Exit code | 0 | 0 |
| OCR candidates | 92 | 92 |
| JSON structure | Same | Same |
| Strings | Baseline | Exact match |
| Confidence values | Baseline | Exact match |
| `ocr_result.json` SHA-256 | `ff60a282...48ab` | `ff60a282...48ab` |
| `ocr_overlay.png` SHA-256 | `16312771...dfa0` | `16312771...dfa0` |
| External connect calls | 18 | 0 |
| DNS connect calls | 4 | 0 |

The OCR input rotation, GPU selection, thresholds, output format, matching
logic, and similarity threshold 40 were not changed.

## Ten-process GPU benchmark

Each run was a fresh OCR subprocess against the same PNG. Every exit code was
0, every run produced 92 candidates, and every result JSON had the baseline
SHA-256.

| Run | Process (s) | Model create (s) | Predict (s) | Save (s) |
|---:|---:|---:|---:|---:|
| 1 | 3.75 | 1.453 | 0.720 | 0.269 |
| 2 | 3.71 | 1.401 | 0.752 | 0.282 |
| 3 | 3.82 | 1.538 | 0.687 | 0.291 |
| 4 | 3.68 | 1.379 | 0.749 | 0.254 |
| 5 | 3.65 | 1.432 | 0.709 | 0.268 |
| 6 | 3.61 | 1.382 | 0.693 | 0.265 |
| 7 | 3.57 | 1.373 | 0.724 | 0.250 |
| 8 | 3.54 | 1.318 | 0.719 | 0.258 |
| 9 | 3.54 | 1.308 | 0.739 | 0.270 |
| 10 | 3.59 | 1.377 | 0.711 | 0.273 |

Process-time summary: min 3.54 s, median 3.63 s, mean 3.646 s, p90 3.757 s,
max 3.82 s, max/median 1.052. Runs 2-10: median 3.61 s and max/median
1.058. No 10x delay occurred.

An initial sandbox-only trial could not see the GPU and fell back to CPU; it
was discarded and is not included in the benchmark JSON.

## SAM3/OCR parallelism

The existing saved-RGB-D timing test was run once with the already-running
SAM3 service at `127.0.0.1:8765`. It copied saved RGB-D data to a temporary
directory; it did not capture from RealSense.

- OCR interval: 5155.927772 to 5159.651529 (monotonic seconds)
- SAM3 interval: 5156.054826 to 5156.753470
- Overlap: 0.698644 seconds
- Result: `parallel=true`

Thus OCR starts first, SAM3 starts without waiting for OCR, and the existing
join happens later. The parallel structure is retained.

## Scope and safeguards

Changed:

- `detection/pro_handbook/sam_py_demo/OCR/paddle_ocr_test.py`
- `detection/pro_handbook/sam_py_demo/get_book_points.py`

Created:

- `detection/pro_handbook/sam3_runtime/docs/OCR_FULL_OFFLINE_REPORT.md`
- `detection/pro_handbook/sam3_runtime/docs/ocr_full_offline_benchmark.json`

`get_book_points_no_mask_merge_no_side_filter.py` was unchanged during this
task (SHA-256 before/after
`3afa6e42d510eac177e0877620749428bd88c2389765d1bd8d229d9171b754e7`);
it imports the production module and therefore receives the corrected OCR
subprocess behavior. Integration code, SAM3 model, OCR models, virtual
environments, and package versions were not changed. The SAM3 model remains
`d8b297b0a9a8a81c7926541a0f8fb08f7a15ee7d53d210b9827190aa21b16bce`.

No robot or Height Controller was connected or initialized. No ROS topic was
published. No RealSense capture was performed. No Wi-Fi, firewall, DNS,
route, NVIDIA driver, system CUDA, `.bashrc`, or `/home/book/pro_book` file
was changed.

Remaining note: the complete-offline guarantee applies to the OCR child. Its
Python network APIs are blocked and the exercised native path produced zero
external connects under strace. SAM3 localhost communication remains
available to the parent recognition process as required.
