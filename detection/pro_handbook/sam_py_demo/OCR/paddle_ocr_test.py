# from __future__ import annotations

# from pathlib import Path
# import json
# import cv2
# import numpy as np
# import sys
# from paddleocr import PaddleOCR


# def save_json(path: Path, obj) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# def draw_ocr_results(img_bgr, out, *, font_scale=0.7, thickness=2):
#     """out: [{"quad": [[x,y]x4], "text": str, "score": float}, ...]"""
#     vis = img_bgr.copy()

#     for item in out:
#         quad = item["quad"]  # [[x,y],...]
#         text = item["text"]
#         score = item.get("score", None)

#         pts = cv2.UMat(cv2.UMat.get(cv2.UMat(quad))) if False else None  # ダミー（無視してOK）

#         pts = cv2.convexHull(
#             cv2.UMat(cv2.UMat.get(cv2.UMat(quad))) if False else
#             cv2.UMat(0)
#         ) if False else None  # ダミー（無視してOK）

#         # quad -> np.int32 (OpenCV用)
#         import numpy as np
#         pts = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))

#         # 四角形を描画
#         cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

#         # テキスト表示位置（quadの左上っぽい点）
#         x0 = int(min(p[0] for p in quad))
#         y0 = int(min(p[1] for p in quad))
#         label = f"{text}"
#         if score is not None:
#             label += f" ({score:.2f})"

#         # 画像外にはみ出しにくいように少し上へ
#         y_text = max(0, y0 - 5)

#         # 背景付きで文字を描く（見やすい）
#         (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
#         x_bg1, y_bg1 = x0, max(0, y_text - th - baseline)
#         x_bg2, y_bg2 = x0 + tw, y_text + baseline
#         cv2.rectangle(vis, (x_bg1, y_bg1), (x_bg2, y_bg2), (0, 0, 0), -1)  # 黒背景
#         cv2.putText(vis, label, (x0, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

#     return vis


# def rotate270_cw(img_bgr: np.ndarray) -> np.ndarray:
#     # （あなたの意図する向きが逆だったので）反対方向へ回す
#     return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)


# def sharpen_unsharp(img_bgr: np.ndarray, sigma: float = 1.2, amount: float = 1.2) -> np.ndarray:
#     """
#     Unsharp mask によるシャープ化
#     sigma: ぼかし強さ
#     amount: シャープ量
#     """
#     blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
#     sharp = cv2.addWeighted(img_bgr, 1.0 + amount, blurred, -amount, 0)
#     return sharp

# def enhance_contrast_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
#     """BGR画像を高コントラスト化（CLAHE）。返り値もBGR。"""
#     lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     l2 = clahe.apply(l)
#     lab2 = cv2.merge([l2, a, b])
#     return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

# def OCR_main(shot_dir: str | Path):
#     shot_dir = Path(shot_dir)

#     ocr = PaddleOCR(
#         ocr_version="PP-OCRv5",
#         lang="japan",
#         use_doc_orientation_classify=False,  # ★勝手に回さない
#         use_doc_unwarping=False,
#         use_textline_orientation=False,
#     )

#     img_path = shot_dir / "after_init_rgb.png"
#     img = cv2.imread(str(img_path))
#     if img is None:
#         raise FileNotFoundError(img_path)

#     img = rotate270_cw(img)  # ★必ず270°回す（時計回り）

#     # （任意：あなたの前処理を使うならここ）
#     # img = enhance_contrast_for_ocr(img)
#     # img = sharpen_unsharp(img)

#     result = ocr.predict(img)

#     res0 = result[0]
#     cv2.imwrite(str(shot_dir / "before_init_rgb_rot270.png"), img)  # ★確認用
#     json_path = shot_dir / "ocr_result.json"
#     vis_path  = shot_dir / "ocr_overlay.png"
#     res0.save_to_json(str(json_path))
#     res0.save_to_img(str(vis_path))
#     return res0

# if __name__ == "__main__":
#     shot_dir = Path(sys.argv[1]).expanduser()
#     OCR_main(shot_dir)

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import sys
import time

# PaddleX 3.3.12 reads these flags while its modules are imported.  Keep them
# local to this OCR process and set them before importing PaddleOCR.
OCR_DETECTION_MODEL_DIR = Path(
    "/home/book/.paddlex/official_models/PP-OCRv5_server_det"
)
OCR_RECOGNITION_MODEL_DIR = Path(
    "/home/book/.paddlex/official_models/PP-OCRv5_server_rec"
)
OCR_VISUALIZATION_FONT = Path("/home/book/.paddlex/fonts/simfang.ttf")
OCR_REQUIRED_MODEL_FILES = (
    "inference.json",
    "inference.pdiparams",
    "inference.yml",
)

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_LOCAL_FONT_FILE_PATH"] = str(OCR_VISUALIZATION_FONT)


def _deny_python_network(event: str, args) -> None:
    """Fail before a Python networking API can issue a network syscall."""
    if event in {
        "socket.connect",
        "socket.connect_ex",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.gethostbyname_ex",
    }:
        raise RuntimeError(
            "Offline OCR network access is disabled. "
            f"Blocked Python audit event: {event}"
        )


sys.addaudithook(_deny_python_network)

import cv2
import numpy as np
from paddleocr import PaddleOCR

try:
    import paddle
except Exception:  # paddle環境以外での構文チェック用
    paddle = None


_OCR_MODEL = None
_OCR_CREATE_SEC = None
SAVE_ROTATED_OCR_INPUT_DEBUG = False


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _asset_file_info(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def validate_offline_ocr_assets(*, include_sha256: bool = False) -> dict:
    """Validate every local asset before PaddleOCR model construction."""
    model_dirs = {
        "text_detection": OCR_DETECTION_MODEL_DIR,
        "text_recognition": OCR_RECOGNITION_MODEL_DIR,
    }
    manifest = {"models": {}, "visualization_font": None}

    for model_type, model_dir in model_dirs.items():
        if not model_dir.is_dir():
            raise FileNotFoundError(
                "Offline OCR model asset is missing. "
                "Network fallback is disabled. "
                f"Missing path: {model_dir}"
            )
        if not os.access(model_dir, os.R_OK):
            raise PermissionError(
                "Offline OCR model asset is not readable. "
                "Network fallback is disabled. "
                f"Unreadable path: {model_dir}"
            )

        files = []
        for filename in OCR_REQUIRED_MODEL_FILES:
            path = model_dir / filename
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(
                    "Offline OCR model asset is missing or empty. "
                    "Network fallback is disabled. "
                    f"Missing path: {path}"
                )
            if not os.access(path, os.R_OK):
                raise PermissionError(
                    "Offline OCR model asset is not readable. "
                    "Network fallback is disabled. "
                    f"Unreadable path: {path}"
                )
            info = {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
            }
            if include_sha256:
                info = _asset_file_info(path)
            files.append(info)

        manifest["models"][model_type] = {
            "path": str(model_dir),
            "files": files,
        }

    recognition_config = OCR_RECOGNITION_MODEL_DIR / "config.json"
    try:
        config = json.loads(recognition_config.read_text(encoding="utf-8"))
        character_dict = config["PostProcess"]["character_dict"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "Offline OCR recognition dictionary/config is invalid. "
            "Network fallback is disabled. "
            f"Invalid path: {recognition_config}"
        ) from exc
    if not isinstance(character_dict, list) or not character_dict:
        raise RuntimeError(
            "Offline OCR recognition dictionary is empty. "
            "Network fallback is disabled. "
            f"Invalid path: {recognition_config}"
        )
    manifest["recognition_dictionary"] = {
        "path": str(recognition_config),
        "format": "embedded PostProcess.character_dict",
        "entries": len(character_dict),
    }

    if not OCR_VISUALIZATION_FONT.is_file() or OCR_VISUALIZATION_FONT.stat().st_size <= 0:
        raise FileNotFoundError(
            "Offline OCR visualization font is missing or empty. "
            "Network fallback is disabled. "
            f"Missing path: {OCR_VISUALIZATION_FONT}"
        )
    if not os.access(OCR_VISUALIZATION_FONT, os.R_OK):
        raise PermissionError(
            "Offline OCR visualization font is not readable. "
            "Network fallback is disabled. "
            f"Unreadable path: {OCR_VISUALIZATION_FONT}"
        )
    font_info = {
        "path": str(OCR_VISUALIZATION_FONT),
        "size_bytes": int(OCR_VISUALIZATION_FONT.stat().st_size),
    }
    if include_sha256:
        font_info = _asset_file_info(OCR_VISUALIZATION_FONT)
    manifest["visualization_font"] = font_info
    return manifest


def draw_ocr_results(img_bgr, out, *, font_scale=0.7, thickness=2):
    """out: [{"quad": [[x,y]x4], "text": str, "score": float}, ...]"""
    vis = img_bgr.copy()

    for item in out:
        quad = item["quad"]
        text = item["text"]
        score = item.get("score", None)

        pts = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        x0 = int(min(p[0] for p in quad))
        y0 = int(min(p[1] for p in quad))
        label = f"{text}"
        if score is not None:
            label += f" ({score:.2f})"

        y_text = max(0, y0 - 5)
        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        x_bg1, y_bg1 = x0, max(0, y_text - th - baseline)
        x_bg2, y_bg2 = x0 + tw, y_text + baseline
        cv2.rectangle(vis, (x_bg1, y_bg1), (x_bg2, y_bg2), (0, 0, 0), -1)
        cv2.putText(
            vis,
            label,
            (x0, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return vis


def rotate270_cw(img_bgr: np.ndarray) -> np.ndarray:
    # 現行コードの仕様を維持: OCR入力は必ず時計回り90度回転する．
    return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)


def sharpen_unsharp(img_bgr: np.ndarray, sigma: float = 1.2, amount: float = 1.2) -> np.ndarray:
    """Unsharp mask によるシャープ化．"""
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(img_bgr, 1.0 + amount, blurred, -amount, 0)
    return sharp


def enhance_contrast_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """BGR画像を高コントラスト化（CLAHE）．返り値もBGR．"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def _configure_paddle_device_once() -> None:
    """GPU対応Paddleなら gpu:0 を明示する．失敗してもOCR本体に任せる．"""
    if paddle is None:
        return

    try:
        print(f"[OCR CACHE] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
        print(f"[OCR CACHE] paddle.is_compiled_with_cuda={paddle.is_compiled_with_cuda()}", flush=True)
        if paddle.is_compiled_with_cuda():
            paddle.set_device("gpu:0")
        else:
            paddle.set_device("cpu")
        print(f"[OCR CACHE] paddle.device={paddle.device.get_device()}", flush=True)
    except Exception as e:
        print(f"[OCR CACHE] paddle device setup skipped: {e}", flush=True)


def create_ocr_model() -> PaddleOCR:
    """検証済みローカル資材だけで現行PP-OCRv5モデルを作る．"""
    validate_offline_ocr_assets()
    _configure_paddle_device_once()

    return PaddleOCR(
        text_detection_model_dir=str(OCR_DETECTION_MODEL_DIR),
        text_recognition_model_dir=str(OCR_RECOGNITION_MODEL_DIR),
        use_doc_orientation_classify=False,  # 勝手に回さない
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def get_ocr_model() -> PaddleOCR:
    """常駐worker内でOCRモデルをキャッシュする．"""
    global _OCR_MODEL, _OCR_CREATE_SEC

    if _OCR_MODEL is None:
        print(f"[OCR CACHE] create OCR model once. pid={os.getpid()}", flush=True)
        t0 = time.perf_counter()
        _OCR_MODEL = create_ocr_model()
        _OCR_CREATE_SEC = time.perf_counter() - t0
        print(f"[OCR CACHE] OCR model created in {_OCR_CREATE_SEC:.3f} sec", flush=True)
    else:
        print(f"[OCR CACHE] reuse OCR model. pid={os.getpid()}", flush=True)

    return _OCR_MODEL


def _run_ocr_with_model(shot_dir: str | Path, ocr: PaddleOCR):
    shot_dir = Path(shot_dir).expanduser().resolve()

    img_path = shot_dir / "after_init_rgb.png"
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)

    img = rotate270_cw(img)

    # 現行仕様では前処理は未使用．必要なら以下を戻す．
    # img = enhance_contrast_for_ocr(img)
    # img = sharpen_unsharp(img)

    t0 = time.perf_counter()
    result = ocr.predict(img)
    predict_sec = time.perf_counter() - t0
    print(f"[OCR CACHE] OCR predict finished in {predict_sec:.3f} sec", flush=True)

    res0 = result[0]
    if SAVE_ROTATED_OCR_INPUT_DEBUG:
        cv2.imwrite(str(shot_dir / "before_init_rgb_rot270.png"), img)
    json_path = shot_dir / "ocr_result.json"
    vis_path = shot_dir / "ocr_overlay.png"

    t_save = time.perf_counter()
    res0.save_to_json(str(json_path))
    res0.save_to_img(str(vis_path))
    save_sec = time.perf_counter() - t_save
    print(f"[OCR CACHE] OCR result save finished in {save_sec:.3f} sec", flush=True)

    # 計測確認用．get_book_points側のcleanupで画像制限してもjsonは残る．
    save_json(
        shot_dir / "ocr_runtime_info.json",
        {
            "cached_model": True,
            "pid": int(os.getpid()),
            "ocr_model_create_sec": None if _OCR_CREATE_SEC is None else float(_OCR_CREATE_SEC),
            "ocr_predict_sec": float(predict_sec),
            "ocr_save_sec": float(save_sec),
            "input_image": str(img_path),
            "json_path": str(json_path),
            "overlay_path": str(vis_path),
            "save_rotated_input_debug": bool(SAVE_ROTATED_OCR_INPUT_DEBUG),
            "offline": True,
            "network_fallback_disabled": True,
            "offline_environment": {
                "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": os.environ.get(
                    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"
                ),
                "PADDLE_PDX_LOCAL_FONT_FILE_PATH": os.environ.get(
                    "PADDLE_PDX_LOCAL_FONT_FILE_PATH"
                ),
            },
            "offline_assets": validate_offline_ocr_assets(),
        },
    )
    return res0


def OCR_main_cached(shot_dir: str | Path):
    """常駐worker用．OCRモデルを作り直さずに使い回す．"""
    ocr = get_ocr_model()
    return _run_ocr_with_model(shot_dir, ocr)


def OCR_main(shot_dir: str | Path):
    """
    互換用エントリポイント．

    単発実行でもこの関数を使えるようにする．同一プロセス内で複数回呼ばれた場合は
    OCR_main_cached() と同じくモデルを再利用する．
    """
    return OCR_main_cached(shot_dir)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--check-offline-assets":
        print(
            json.dumps(
                validate_offline_ocr_assets(include_sha256=True),
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(0)
    if len(sys.argv) < 2:
        raise SystemExit("usage: python paddle_ocr_test.py <shot_dir>")
    shot_dir = Path(sys.argv[1]).expanduser().resolve()
    OCR_main(shot_dir)
