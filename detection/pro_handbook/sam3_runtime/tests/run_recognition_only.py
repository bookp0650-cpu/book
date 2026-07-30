from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from detection.pro_handbook.sam_py_demo.get_book_points import run_capture_and_pca_offline


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--query",required=True); ap.add_argument("--shot-dir",required=True,type=Path); args=ap.parse_args()
    started=time.perf_counter()
    result=run_capture_and_pca_offline(
        query=args.query, shot_dir=args.shot_dir, interactive=False,
        show_pointcloud_gui=False, save_pointcloud_debug=False,
        save_step_by_step_pointcloud_debug=False,
    )
    elapsed=time.perf_counter()-started
    roll,point,width,shot_dir=result
    payload={"success":roll is not None,"elapsed_seconds":elapsed,"roll_rad":None if roll is None else float(roll),"point":None if point is None else np.asarray(point).tolist(),"book_width_mm":None if width is None else float(width),"shot_dir":None if shot_dir is None else str(shot_dir)}
    (args.shot_dir/"recognition_only_result.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(payload,ensure_ascii=False,indent=2))
    if not payload["success"]: raise SystemExit(1)


if __name__=="__main__": main()
