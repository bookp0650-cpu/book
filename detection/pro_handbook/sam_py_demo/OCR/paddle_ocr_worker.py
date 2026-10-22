#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import ctypes
import io
import json
import os
import re
import signal
import socket
import sys
import time
import traceback
from pathlib import Path

import cv2

import paddle_ocr_test as base_ocr
import paddle_ocr_dual_angle as dual


def set_parent_death_signal():
    """
    親のretrievalプロセスが死んだら
    OCR workerも終了させる。
    Linux専用。
    """
    try:
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL(None)
        libc.prctl(
            PR_SET_PDEATHSIG,
            signal.SIGTERM,
        )
    except Exception as e:
        print(
            f"[OCR WORKER] "
            f"parent death signal setup skipped: {e}",
            flush=True,
        )


def recv_json_line(conn: socket.socket):
    buf = bytearray()

    while True:
        chunk = conn.recv(4096)

        if not chunk:
            break

        buf.extend(chunk)

        if b"\n" in buf:
            break

    if not buf:
        raise RuntimeError(
            "empty request"
        )

    line = bytes(buf).split(
        b"\n",
        1,
    )[0]

    return json.loads(
        line.decode("utf-8")
    )


def send_json_line(
    conn: socket.socket,
    obj,
):
    payload = (
        json.dumps(
            obj,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")

    try:
        conn.sendall(payload)
        return True

    except (BrokenPipeError, ConnectionResetError):
        print(
            "[OCR WORKER] client disconnected before response",
            flush=True,
        )
        return False


def run_book_ocr(
    shot_dir: Path,
    ocr,
    ocr_runtime: dict,
):
    """
    paddle_ocr_dual_angle.py の main() と
    同じ処理を、常駐モデルで実行する。
    """

    shot_dir = (
        Path(shot_dir)
        .expanduser()
        .resolve()
    )

    img_path = (
        shot_dir
        / "after_init_rgb.png"
    )

    worker_total_start = time.perf_counter()
    stage_start = time.perf_counter()
    original = cv2.imread(
        str(img_path),
        cv2.IMREAD_COLOR,
    )
    image_read_seconds = time.perf_counter() - stage_start

    if original is None:
        raise FileNotFoundError(
            img_path
        )

    # 既存仕様:
    # まず時計回り90度
    stage_start = time.perf_counter()
    base_img = base_ocr.rotate270_cw(
        original
    )
    base_rotation_seconds = time.perf_counter() - stage_start

    all_candidates = []
    candidates_by_angle = {}
    runtimes = {}
    angle_stage_timings = {}
    per_angle_dedup_seconds = {}
    inference_start_monotonic = time.monotonic()

    for angle in dual.EXTRA_ANGLES_DEG:

        angle_timing = {}
        candidates, elapsed = (
            dual.run_one_angle(
                ocr,
                base_img,
                shot_dir,
                angle,
                timing_out=angle_timing,
            )
        )

        all_candidates.extend(
            candidates
        )
        stage_start = time.perf_counter()
        candidates_by_angle[float(angle)] = dual.merge_candidates(candidates)
        per_angle_dedup_seconds[str(angle)] = float(
            time.perf_counter() - stage_start
        )

        runtimes[angle] = elapsed
        angle_stage_timings[str(angle)] = angle_timing

    stage_start = time.perf_counter()
    merged_candidates = (
        dual.merge_candidates(
            all_candidates
        )
    )
    combined_dedup_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    final_json = (
        dual.candidates_to_legacy_json(
            merged_candidates,
            shot_dir=shot_dir,
            runtimes=runtimes,
            candidates_by_angle=candidates_by_angle,
        )
    )
    final_json_build_seconds = time.perf_counter() - stage_start
    inference_end_monotonic = time.monotonic()
    inference_seconds = (
        inference_end_monotonic
        - inference_start_monotonic
    )

    final_path = (
        shot_dir
        / "ocr_result.json"
    )

    stage_start = time.perf_counter()
    dual.save_json(
        final_path,
        final_json,
    )
    final_json_save_seconds = time.perf_counter() - stage_start

    dual.save_json(
        shot_dir
        / "ocr_runtime_info.json",
        {
            "dual_angle": True,
            "base_rotation": "CW90",
            "extra_angles_deg": [
                float(v)
                for v
                in dual.EXTRA_ANGLES_DEG
            ],
            "raw_candidate_count": int(
                len(all_candidates)
            ),
            "merged_candidate_count": int(
                len(merged_candidates)
            ),
            "predict_seconds": {
                str(angle): float(sec)
                for angle, sec
                in runtimes.items()
            },
            "detailed_timing_seconds": {
                "image_read": float(image_read_seconds),
                "base_rotation": float(base_rotation_seconds),
                "angles": angle_stage_timings,
                "per_angle_polygon_dedup": per_angle_dedup_seconds,
                "combined_polygon_dedup": float(combined_dedup_seconds),
                "final_json_build": float(final_json_build_seconds),
                "final_json_save": float(final_json_save_seconds),
                "worker_total_before_runtime_json": float(
                    time.perf_counter() - worker_total_start
                ),
            },
            "ocr_inference_seconds": float(
                inference_seconds
            ),
            "ocr_result_json": str(
                final_path
            ),
            "cached_worker": (
                os.environ.get(
                    "BOOK_OCR_RELOAD_EACH_SEQUENCE",
                    "0",
                )
                != "1"
            ),
            "worker_pid": os.getpid(),
            "device": ocr_runtime,
            "sam_compatibility": (
                "OCR polygons are inverse-transformed "
                "to the legacy CW90 frame. "
                "Existing forced_angle=90 matching "
                "remains unchanged."
            ),
        },
    )

    print(
        "[OCR WORKER] "
        f"book OCR done: {final_path}",
        flush=True,
    )

    return {
        "ocr_result_json": str(
            final_path
        ),
        "timing": {
            "inference_start_monotonic": float(
                inference_start_monotonic
            ),
            "inference_end_monotonic": float(
                inference_end_monotonic
            ),
            "inference_seconds": float(
                inference_seconds
            ),
        },
    }


def normalize_digits(text):
    return re.sub(
        r"[^0-9]",
        "",
        str(text),
    )


def collect_recognized_texts(obj):
    """
    PaddleOCR JSONから文字列を回収。
    """

    out = []

    def walk(x):

        if isinstance(x, dict):

            for key, value in x.items():

                key_lower = str(
                    key
                ).lower()

                if (
                    key_lower
                    in {
                        "rec_texts",
                        "texts",
                    }
                    and isinstance(
                        value,
                        list,
                    )
                ):
                    for item in value:

                        if (
                            isinstance(
                                item,
                                str,
                            )
                            and item.strip()
                        ):
                            out.append(
                                item.strip()
                            )

                    continue

                if (
                    key_lower
                    in {
                        "rec_text",
                        "text",
                    }
                    and isinstance(
                        value,
                        str,
                    )
                ):
                    if value.strip():
                        out.append(
                            value.strip()
                        )

                    continue

                walk(value)

        elif isinstance(
            x,
            list,
        ):
            for item in x:
                walk(item)

    walk(obj)

    unique = []
    seen = set()

    for text in out:

        if text in seen:
            continue

        seen.add(text)
        unique.append(text)

    return unique


def run_barcode_ocr(
    image_path: Path,
    target,
    out_dir: Path,
    ocr,
):
    """
    バーコードOCR fallback。
    書籍OCRと同じPaddleOCRモデルを使用。
    """

    image_path = (
        Path(image_path)
        .expanduser()
        .resolve()
    )

    out_dir = (
        Path(out_dir)
        .expanduser()
        .resolve()
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    img = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if img is None:
        raise RuntimeError(
            f"cv2.imread failed: "
            f"{image_path}"
        )

    # 現在のbarcode OCR仕様
    ocr_img = cv2.rotate(
        img,
        cv2.ROTATE_180,
    )


    # ==========================================
    # barcode OCRだけ解像度を1/2へ
    # 3840x2160 -> 1920x1080
    # ==========================================
    h, w = ocr_img.shape[:2]

    if w > 1920:
        scale = 1920.0 / float(w)

        ocr_img = cv2.resize(
            ocr_img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    print(
        "[BARCODE OCR WORKER] "
        f"OCR input size="
        f"{ocr_img.shape[1]}x{ocr_img.shape[0]}",
        flush=True,
    )

    rotated_path = (
        out_dir
        / "ocr_input_rot180.png"
    )

    cv2.imwrite(
        str(rotated_path),
        ocr_img,
    )

    target_digits = normalize_digits(
        target
    )

    print(
        "[BARCODE OCR WORKER] "
        f"target_digits={target_digits}",
        flush=True,
    )

    # ★ここではPaddleOCRを生成しない
    # 常駐モデルでpredictするだけ
    result = ocr.predict(
        ocr_img
    )

    texts = []

    raw_json_path = (
        out_dir
        / "ocr_raw.json"
    )

    overlay_path = (
        out_dir
        / "ocr_overlay.png"
    )

    if result:

        res0 = result[0]

        res0.save_to_json(
            str(raw_json_path)
        )

        res0.save_to_img(
            str(overlay_path)
        )

        raw_json = json.loads(
            raw_json_path.read_text(
                encoding="utf-8"
            )
        )

        texts = (
            collect_recognized_texts(
                raw_json
            )
        )

    matched = False
    matched_text = None
    matched_digits = None

    for text in texts:

        digits = normalize_digits(
            text
        )

        print(
            "[BARCODE OCR WORKER] "
            f"text={text!r} "
            f"digits={digits!r}",
            flush=True,
        )

        if (
            target_digits
            and target_digits
            in digits
        ):
            matched = True
            matched_text = text
            matched_digits = digits

            print(
                ">>> BARCODE OCR SUCCESS <<<",
                flush=True,
            )

            break

    payload = {
        "matched": bool(
            matched
        ),
        "target": str(
            target
        ),
        "target_digits": (
            target_digits
        ),
        "matched_text": (
            matched_text
        ),
        "matched_digits": (
            matched_digits
        ),
        "texts": texts,
        "input_image": str(
            image_path
        ),
        "method": (
            "shared_cached_paddleocr"
        ),
        "ocr_overlay_path": str(
            overlay_path
        ),
        "worker_pid": os.getpid(),
    }

    result_path = (
        out_dir
        / "ocr_match.json"
    )

    result_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "[BARCODE OCR WORKER] "
        f"matched={matched}",
        flush=True,
    )

    return payload


def main():

    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: "
            "python paddle_ocr_worker.py "
            "<socket_path>"
        )

    set_parent_death_signal()

    socket_path = Path(
        sys.argv[1]
    )

    try:
        socket_path.unlink(
            missing_ok=True
        )
    except Exception:
        pass

    server = socket.socket(
        socket.AF_UNIX,
        socket.SOCK_STREAM,
    )

    server.bind(
        str(socket_path)
    )

    server.listen(8)

    print(
        "[OCR WORKER] "
        f"listening: {socket_path}",
        flush=True,
    )

    # ==========================================
    # ここで1回だけモデル生成
    # ==========================================
    model_load_start_monotonic = time.monotonic()
    ocr = (
        base_ocr.get_ocr_model()
    )
    ocr_runtime = base_ocr.get_ocr_runtime_info(ocr)
    if ocr_runtime["resolved_device"].startswith("gpu") and not ocr_runtime[
        "gpu_verified"
    ]:
        raise RuntimeError(
            "PaddleOCR GPU verification failed: "
            f"{json.dumps(ocr_runtime, ensure_ascii=False)}"
        )
    model_ready_monotonic = time.monotonic()
    model_load_seconds = (
        model_ready_monotonic
        - model_load_start_monotonic
    )

    print(
        "[OCR WORKER] "
        f"model ready. pid={os.getpid()} "
        f"device={json.dumps(ocr_runtime, ensure_ascii=False)}",
        flush=True,
    )

    try:

        while True:

            conn, _ = (
                server.accept()
            )

            with conn:

                try:

                    request = (
                        recv_json_line(
                            conn
                        )
                    )

                    mode = str(
                        request.get(
                            "mode",
                            "",
                        )
                    )

                    log_buffer = (
                        io.StringIO()
                    )

                    with (
                        contextlib.redirect_stdout(
                            log_buffer
                        ),
                        contextlib.redirect_stderr(
                            log_buffer
                        ),
                    ):

                        # Persistent版だけは従来どおりキャッシュ済みモデルを
                        # 取得する。Reload版は起動直後に生成した上のocrをその
                        # 1リクエストだけで使い、終了時にプロセスごと解放する。
                        if (
                            os.environ.get(
                                "BOOK_OCR_RELOAD_EACH_SEQUENCE",
                                "0",
                            )
                            != "1"
                        ):
                            ocr = (
                                base_ocr.get_ocr_model()
                            )

                        if mode == "ready":

                            data = {
                                "ready": True,
                                "worker_pid": os.getpid(),
                                "runtime": ocr_runtime,
                            }

                        elif mode == "book":

                            data = run_book_ocr(
                                Path(
                                    request[
                                        "shot_dir"
                                    ]
                                ),
                                ocr,
                                ocr_runtime,
                            )

                        elif mode == "barcode":

                            data = (
                                run_barcode_ocr(
                                    Path(
                                        request[
                                            "image_path"
                                        ]
                                    ),
                                    request[
                                        "target"
                                    ],
                                    Path(
                                        request[
                                            "out_dir"
                                        ]
                                    ),
                                    ocr,
                                )
                            )

                        else:

                            raise ValueError(
                                f"unknown OCR mode: "
                                f"{mode}"
                            )

                    # BOOK_OCR_GPU_SAFE=1: 依頼ごとに未使用のGPUキャッシュを返す。
                    # モデル本体は残るので、次の依頼でも再読み込みは不要。
                    if os.environ.get("BOOK_OCR_GPU_SAFE", "0") == "1":
                        try:
                            import paddle

                            paddle.device.cuda.empty_cache()
                            print(
                                "[OCR WORKER] GPU cache released",
                                flush=True,
                            )
                        except Exception as release_exc:
                            print(
                                "[OCR WORKER] GPU cache release failed: "
                                f"{release_exc}",
                                flush=True,
                            )

                    response_timing = {
                        "model_load_start_monotonic": float(
                            model_load_start_monotonic
                        ),
                        "model_ready_monotonic": float(
                            model_ready_monotonic
                        ),
                        "model_load_seconds": float(
                            model_load_seconds
                        ),
                    }
                    if isinstance(data, dict):
                        data_timing = data.get(
                            "timing"
                        )
                        if isinstance(data_timing, dict):
                            response_timing.update(
                                data_timing
                            )

                    send_json_line(
                        conn,
                        {
                            "ok": True,
                            "data": data,
                            "worker_pid": os.getpid(),
                            "timing": response_timing,
                            "stdout": (
                                log_buffer
                                .getvalue()
                            ),
                        },
                    )

                except Exception as e:

                    send_json_line(
                        conn,
                        {
                            "ok": False,
                            "error": (
                                f"{type(e).__name__}: "
                                f"{e}"
                            ),
                            "traceback": (
                                traceback.format_exc()
                            ),
                        },
                    )

    finally:

        try:
            server.close()
        except Exception:
            pass

        try:
            socket_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
