"""Isolated disparity-only runner for NVIDIA Fast-FoundationStereo.

Run this script with the Python interpreter of a dedicated FFS environment.
It deliberately avoids the official demo's output-directory deletion, GUI,
point-cloud generation, and metric-depth calculation.  Metric depth remains
the responsibility of ``storage_ffs_depth.py`` using live RealSense calibration.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output-disparity", type=Path, required=True)
    parser.add_argument("--output-right-disparity", type=Path)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--max-disp", type=int, default=192)
    return parser.parse_args()


def _read_three_channel(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"failed to read stereo image: {path}")
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise RuntimeError(f"unsupported stereo image shape: {path}: {image.shape}")
    return image[:, :, :3]


def main() -> None:
    args = parse_args()
    repository = args.repository.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    if not (repository / "core").is_dir() or not (repository / "Utils.py").is_file():
        raise RuntimeError(
            f"Fast-FoundationStereo repository layout is invalid: {repository}"
        )
    if not weights.is_file():
        raise RuntimeError(f"Fast-FoundationStereo weight is missing: {weights}")
    if not 0.0 < args.scale <= 1.0:
        raise RuntimeError(f"scale must be in (0, 1]: {args.scale}")
    if args.valid_iters <= 0 or args.max_disp <= 0:
        raise RuntimeError("valid-iters and max-disp must be positive")

    sys.path.insert(0, str(repository))
    import torch
    from core.utils.utils import InputPadder
    from Utils import AMP_DTYPE

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the isolated FFS environment")

    left_original = _read_three_channel(args.left)
    right_original = _read_three_channel(args.right)
    if left_original.shape != right_original.shape:
        raise RuntimeError(
            f"stereo image shape mismatch: {left_original.shape}!={right_original.shape}"
        )
    original_height, original_width = left_original.shape[:2]
    if args.scale != 1.0:
        scaled_width = max(32, int(round(original_width * args.scale)))
        scaled_height = max(32, int(round(original_height * args.scale)))
        left = cv2.resize(
            left_original, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA
        )
        right = cv2.resize(
            right_original, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA
        )
    else:
        left = left_original
        right = right_original

    torch.autograd.set_grad_enabled(False)
    model = torch.load(str(weights), map_location="cpu", weights_only=False)
    model.args.valid_iters = int(args.valid_iters)
    model.args.max_disp = int(args.max_disp)
    model.cuda().eval()

    def infer(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        first_tensor = (
            torch.as_tensor(first).cuda().float()[None].permute(0, 3, 1, 2)
        )
        second_tensor = (
            torch.as_tensor(second).cuda().float()[None].permute(0, 3, 1, 2)
        )
        padder = InputPadder(first_tensor.shape, divis_by=32, force_square=False)
        first_tensor, second_tensor = padder.pad(first_tensor, second_tensor)
        with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
            disparity = model.forward(
                first_tensor,
                second_tensor,
                iters=int(args.valid_iters),
                test_mode=True,
                optimize_build_volume="pytorch1",
            )
        disparity = padder.unpad(disparity.float())
        return disparity.detach().cpu().numpy().reshape(first.shape[:2]).clip(0, None)

    disparity_scaled = infer(left, right)
    if args.scale != 1.0:
        disparity_left = cv2.resize(
            disparity_scaled,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        ) / float(args.scale)
    else:
        disparity_left = disparity_scaled
    args.output_disparity.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        args.output_disparity,
        disparity_left.astype(np.float32),
        allow_pickle=False,
    )

    if args.output_right_disparity is not None:
        # Flip+swap preserves the model's positive left-to-right convention;
        # flipping the result back yields positive right-to-left disparity.
        right_flipped = np.ascontiguousarray(right[:, ::-1])
        left_flipped = np.ascontiguousarray(left[:, ::-1])
        disparity_right_scaled = infer(right_flipped, left_flipped)[:, ::-1]
        if args.scale != 1.0:
            disparity_right = cv2.resize(
                disparity_right_scaled,
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR,
            ) / float(args.scale)
        else:
            disparity_right = disparity_right_scaled
        args.output_right_disparity.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            args.output_right_disparity,
            disparity_right.astype(np.float32),
            allow_pickle=False,
        )


if __name__ == "__main__":
    main()
