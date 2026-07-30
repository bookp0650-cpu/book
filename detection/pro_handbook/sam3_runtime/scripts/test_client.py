from __future__ import annotations

import argparse
import json
import os
import urllib.request
from PIL import Image

from detection.pro_handbook.sam3_runtime.service.client import Sam3BatchInfer


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("image", nargs="?"); parser.add_argument("--health-only", action="store_true")
    args = parser.parse_args(); endpoint = os.getenv("SAM3_ENDPOINT", "http://127.0.0.1:8765")
    if args.health_only:
        with urllib.request.urlopen(endpoint + "/health", timeout=5) as response: print(json.dumps(json.load(response), indent=2)); return
    if not args.image: parser.error("IMAGE is required")
    masks, data = Sam3BatchInfer(endpoint=endpoint).infer_masks(Image.open(args.image).convert("RGB"))
    print(json.dumps({"mask_count": len(masks), "shape": list(masks[0].shape) if masks else None, "dtype": str(masks[0].dtype) if masks else None, "objects": data}, indent=2))


if __name__ == "__main__": main()
