# SAM3 / PaddleOCR Parallelism Report

## Conclusion

`parallel=true`

The saved-RGB-D comparison pipeline starts PaddleOCR with
`subprocess.Popen`, performs the synchronous SAM3 request while that child
process is running, and calls `communicate()` only after SAM3 returns. A
single recognition-only measurement also confirmed temporal overlap.

No robot module or integration file was imported by the measurement, and no
RealSense live capture was performed.

## Code path

- Caller: `run_capture_and_pca_no_mask_merge_no_side_filter()` in
  `get_book_points_no_mask_merge_no_side_filter.py`, lines 543-588.
- Shared recognition function:
  `run_capture_and_pca_offline_no_mask_merge_no_side_filter()`, lines 190-422.
- OCR start: `current.start_ocr_subprocess(shot_dir)`, lines 212-214.
- SAM3 start/call: `_get_sam_runner_compat()` and
  `_infer_masks_compat()`, lines 216-232.
- Completion wait: `current.wait_ocr_subprocess()`, lines 234-237.
- OCR/mask association: `current.merge_ocr_and_masks()`, lines 239-248.
- Parallel mechanism: `subprocess.Popen`, defined in
  `get_book_points.py`, lines 403-423.
- Wait mechanism: `Popen.communicate(timeout=...)`, defined in
  `get_book_points.py`, lines 426-441.

The call order is:

1. Start the PaddleOCR child process.
2. Submit and synchronously wait for the SAM3 service inference.
3. After SAM3 returns, wait for PaddleOCR with `communicate()`.
4. Associate OCR results with the returned SAM3 masks.

The SAM3 service and OCR being separate processes is not the basis for the
conclusion. The conclusion is based on the overlapping invocation intervals.

## One-run measurement

Input:

`/home/book/pro_book_SAM3/pro_hand_book_python/captures/100test/1/`

Clock: `time.perf_counter()` in seconds from an implementation-defined
monotonic origin.

| Event | Timestamp |
|---|---:|
| `ocr_start` | 20456.374940711 |
| `sam3_start` | 20456.551821111 |
| `sam3_end` | 20457.342661586 |
| `ocr_end` | 20464.995229878 |
| `join_or_wait_end` | 20465.020549092 |

Calculated overlap:

`min(sam3_end, ocr_end) - max(sam3_start, ocr_start) = 0.790840475 s`

Both required conditions hold:

- `sam3_start < ocr_end`
- `ocr_start < sam3_end`

Therefore `parallel=true`.

The diagnostic wrapper is
`tests/measure_sam3_ocr_parallelism_once.py`. It copies only the saved RGB and
Depth inputs into a temporary directory, observes the OCR child-process exit
from a monitoring thread, and removes the temporary results on exit.

## SAM3 service lifecycle

`run_capture_and_pca_no_mask_merge_no_side_filter()` connects through the
existing SAM3 runtime adapter; it does not load the full model in the
integration process and does not start or stop the service itself.

For repeated integration calls, start the existing service once with
`sam3_runtime/scripts/start_service.sh`. The service loads the model once and
reuses it for subsequent requests. Stop it with the existing
`sam3_runtime/scripts/stop_service.sh` only when that service was started for
the integration run. No service-management logic was added to the integration
file because that would exceed the requested recognition-boundary-only change.

Required runtime settings include:

```bash
unset LD_LIBRARY_PATH
export BOOK_SEGMENTATION_BACKEND=sam3
export SAM3_ENDPOINT=http://127.0.0.1:8765
export PYTHONPATH=/home/book/pro_book_SAM3/pro_hand_book_python
```
