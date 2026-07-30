from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from detection.pro_handbook.sam3_runtime.service.client import Sam3BatchInfer

PROMPT = "book spine"


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iou(a, b) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--image", required=True, type=Path); ap.add_argument("--output", required=True, type=Path); ap.add_argument("--standalone", required=True, type=Path); ap.add_argument("--adapter", action="store_true")
    args = ap.parse_args(); source=args.image.resolve(); out=args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    bgr=cv2.imread(str(source),cv2.IMREAD_UNCHANGED); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); pil=Image.fromarray(rgb)
    shutil.copy2(source,out/"input_copy.png")
    dump(out/"input_info.json", {"source_absolute_path":str(source),"width":rgb.shape[1],"height":rgb.shape[0],"array_shape":list(bgr.shape),"source_dtype":str(bgr.dtype),"source_channel_order":"BGR (OpenCV)","sam3_channel_order":"RGB","preprocessing":"BGR->RGB; official service Sam3Processor; no caller resize","text_prompt":PROMPT})
    started=time.perf_counter()
    if args.adapter:
        from detection.pro_handbook.sam_py_demo.get_book_points import _get_sam_runner_compat
        client=_get_sam_runner_compat("ignored","ignored","cuda",use_cache=False)
    else: client=Sam3BatchInfer(prompt=PROMPT)
    masks,data=client.infer_masks(pil); communication=time.perf_counter()-started
    masks=np.asarray(masks,dtype=bool); scores=np.asarray([x["score"] for x in data],dtype=np.float32); boxes=np.asarray([[x["box"][k] for k in ("x1","y1","x2","y2")] for x in data],dtype=np.float32)
    selected=int(np.argmax(scores)) if len(scores) else None; sm=masks[selected] if selected is not None else np.zeros(rgb.shape[:2],bool)
    np.savez_compressed(out/"raw_masks.npz",masks=masks); Image.fromarray(sm.astype(np.uint8)*255).save(out/"selected_mask.png")
    overlay=rgb.copy(); overlay[sm]=(0.45*overlay[sm]+0.55*np.array([255,0,0])).astype(np.uint8); Image.fromarray(overlay).save(out/"overlay.png")
    dump(out/"boxes.json",boxes.tolist()); dump(out/"scores.json",scores.tolist())
    standalone=np.load(args.standalone/"raw_masks.npz",allow_pickle=False)["masks"]; standalone_meta=json.loads((args.standalone/"metadata.json").read_text()); standalone_selected=standalone[int(standalone_meta["selected_index"])]
    selected_iou=iou(standalone_selected,sm)
    route="adapter" if args.adapter else "service_client"
    dump(out/"metadata.json",{"route":route,"prompt":PROMPT,"mask_count":len(masks),"mask_shape":list(masks.shape),"mask_dtype":str(masks.dtype),"mask_areas":[int(x.sum()) for x in masks],"selected_index":selected,"selected_area":int(sm.sum()),"selected_score":float(scores[selected]) if selected is not None else None,"selected_box_xyxy":boxes[selected].tolist() if selected is not None else None,"standalone_selected_mask_iou":selected_iou})
    dump(out/"runtime.json",{"communication_seconds":communication,"service":getattr(client,"last_metadata",None)})
    (out/"inference.log").write_text(f"route={route} masks={len(masks)} selected={selected} standalone_iou={selected_iou:.9f} communication_seconds={communication:.6f}\n")
    print(json.dumps({"route":route,"mask_count":len(masks),"selected_index":selected,"standalone_iou":selected_iou,"communication_seconds":communication},indent=2))


if __name__=="__main__": main()
