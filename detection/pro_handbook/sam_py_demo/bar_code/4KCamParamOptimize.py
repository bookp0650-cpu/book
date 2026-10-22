#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from pyzbar.pyzbar import decode as zbar_decode
except Exception:
    zbar_decode = None


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def set_ctrl(dev: str, **kwargs) -> None:
    # 例: set_ctrl("/dev/video10", gain=8, exposure_time_absolute=250)
    items = [f"{k}={v}" for k, v in kwargs.items()]
    run(["v4l2-ctl", "-d", dev, "--set-ctrl=" + ",".join(items)])


def set_format(dev: str, width: int, height: int, pix: str, fps: int) -> None:
    run(["v4l2-ctl", "-d", dev, f"--set-fmt-video=width={width},height={height},pixelformat={pix}"])
    run(["v4l2-ctl", "-d", dev, f"--set-parm={fps}"])


def capture_one(dev_path: str, width: int, height: int, fourcc: str, warmup: int = 20) -> np.ndarray:
    cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {dev_path}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    # warmup
    for _ in range(warmup):
        cap.read()

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("capture failed")
    return frame


def roi_barcode_right(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    棚画像の右側ラベル帯にバーコードがある前提の雑ROI。
    必要なら比率は調整してOK。
    """
    H, W = frame.shape[:2]
    x1 = int(W * 0.68)
    x2 = int(W * 0.98)
    y1 = int(H * 0.28)
    y2 = int(H * 0.62)
    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)


def decode_try(gray: np.ndarray) -> str | None:
    if zbar_decode is None:
        return None

    # サイズ違いも試す（スケールが合うと通ることがある）
    imgs = [
        gray,
        cv2.resize(gray, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA),
        cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR),
    ]
    for im in imgs:
        res = zbar_decode(im)
        if res:
            return res[0].data.decode("utf-8", errors="ignore")
    return None


def quality_metrics(roi_gray: np.ndarray) -> tuple[float, float, float]:
    """
    - sharp: ぼけ/ピントの指標（Laplacian var）
    - edge_x: 縦線量（Sobel-x の平均絶対値）
    - noise: エッジを除外した背景ノイズ推定（grad小さい部分のstd）
    """
    sharp = float(cv2.Laplacian(roi_gray, cv2.CV_32F).var())

    sx = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.abs(sx) + np.abs(sy)
    edge_x = float(np.mean(np.abs(sx)))

    # 背景ノイズ推定：強エッジを除外してstd
    g = grad.reshape(-1)
    thr = np.percentile(g, 70)  # 上位30%のエッジを捨てる
    mask = grad < thr
    if np.count_nonzero(mask) < 100:
        noise = float(np.std(roi_gray))
    else:
        noise = float(np.std(roi_gray[mask]))

    return sharp, edge_x, noise


def score(sharp: float, edge_x: float, noise: float, gain: int, decoded: str | None) -> float:
    """
    ベスト選定用のスコア。
    - decoded成功が最優先
    - それ以外は「縦エッジ多い」「シャープ」「ノイズ少ない」を重視
    - gainが高いほどペナルティ（ノイズが増えるので）
    """
    if decoded is not None and decoded != "":
        return 1e9 - gain  # decodeできたら低gain優先で勝つ

    # 単位が違うので係数は経験則。必要なら調整。
    base = edge_x * 1.0 + sharp * 0.01 - noise * 1.2
    penalty = 0.8 * (gain / 255.0)  # gainが高いほど微ペナルティ
    return base - penalty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="/dev/video10")
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    ap.add_argument("--pix", default="MJPG")      # v4l2-ctl用
    ap.add_argument("--fourcc", default="MJPG")   # OpenCV用
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--outdir", default="/home/book/pro_book_SAM3/pro_hand_book_python/captures/bookshelf_barcode/scan4k")

    # 10枚固定：gain 2通り × exposure 5通り
    ap.add_argument("--gains", default="0,8")
    ap.add_argument("--exposures", default="80,120,160,220,300")

    # 固定設定
    ap.add_argument("--focus", type=int, default=250)  # とりあえず。合わなければ変える
    ap.add_argument("--set-50hz", action="store_true", help="power_line_frequency を 50Hz 側に寄せる（推奨）")
    ap.add_argument("--no-af", action="store_true", help="AFを止めて focus_absolute を固定（推奨）")
    ap.add_argument("--manual-exposure", action="store_true", help="auto_exposure=1(Manual) と dynamic_framerate=0（推奨）")
    ap.add_argument("--no-publish", action="store_true")  # 将来拡張用（ここでは未使用）

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gains = [int(x) for x in args.gains.split(",") if x.strip() != ""]
    expos = [int(x) for x in args.exposures.split(",") if x.strip() != ""]
    if len(gains) * len(expos) != 10:
        raise ValueError("gains×exposures が 10 になるようにしてください（例: gains=0,8 exposures=80,120,160,220,300）")

    # ---- カメラ側の固定設定（ノイズ低減寄り） ----
    # 4K MJPG 30fps 固定
    set_format(args.dev, args.width, args.height, args.pix, args.fps)

    if args.set_50hz:
        # あなたの表示では (60 Hz)=2 なので、50Hzは 1 の可能性が高い
        try:
            set_ctrl(args.dev, power_line_frequency=1)
        except Exception:
            pass

    if args.manual_exposure:
        # V4L2_CID_EXPOSURE_AUTO: 0=Auto, 1=Manual, 2=Shutter, 3=Aperture (一般的)
        try:
            set_ctrl(args.dev, auto_exposure=1, exposure_dynamic_framerate=0)
        except Exception:
            pass

    if args.no_af:
        try:
            set_ctrl(args.dev, focus_automatic_continuous=0, focus_absolute=args.focus)
        except Exception:
            pass

    # シャープ/逆光補正はノイズや偽エッジ要因になりうるので弱め（効けば）
    try:
        set_ctrl(args.dev, sharpness=1, backlight_compensation=0)
    except Exception:
        pass

    # ---- スキャン本体 ----
    csv_path = outdir / "scan.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "gain", "exposure", "decoded", "sharp", "edge_x", "noise", "score", "img", "roi"])

        best = {"score": -1e18, "gain": None, "exp": None, "decoded": None, "img": None}

        idx = 0
        for g in gains:
            for e in expos:
                idx += 1
                print(f"[{idx}/10] set gain={g}, exposure_time_absolute={e}")
                set_ctrl(args.dev, gain=g, exposure_time_absolute=e)
                time.sleep(0.15)

                frame = capture_one(args.dev, args.width, args.height, args.fourcc)

                img_path = outdir / f"shot_{idx:02d}_g{g:03d}_e{e:04d}.png"
                cv2.imwrite(str(img_path), frame)

                roi, (x1, y1, x2, y2) = roi_barcode_right(frame)
                roi_path = outdir / f"roi_{idx:02d}_g{g:03d}_e{e:04d}.png"
                cv2.imwrite(str(roi_path), roi)

                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                decoded = decode_try(roi_gray)
                sharp, edge_x, noise = quality_metrics(roi_gray)
                sc = score(sharp, edge_x, noise, g, decoded)

                print(f"  decoded={decoded} sharp={sharp:.1f} edge_x={edge_x:.1f} noise={noise:.1f} score={sc:.2f}")

                w.writerow([idx, g, e, decoded, f"{sharp:.3f}", f"{edge_x:.3f}", f"{noise:.3f}", f"{sc:.3f}", str(img_path), str(roi_path)])

                # best更新（decode成功ならgainが低い方が勝つようにscoreで調整済み）
                if sc > best["score"]:
                    best.update(score=sc, gain=g, exp=e, decoded=decoded, img=str(img_path))

    # bestを保存
    best_txt = outdir / "best.txt"
    best_txt.write_text(
        f"BEST\n"
        f"  gain={best['gain']}\n"
        f"  exposure_time_absolute={best['exp']}\n"
        f"  decoded={best['decoded']}\n"
        f"  score={best['score']}\n"
        f"  img={best['img']}\n",
        encoding="utf-8",
    )
    print("\n=== BEST ===")
    print(best_txt.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()