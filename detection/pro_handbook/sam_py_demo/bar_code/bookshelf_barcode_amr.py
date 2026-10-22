from detection.pro_handbook.sam_py_demo.bar_code.web_camera_capture import capture_one_depstech
from detection.pro_handbook.sam_py_demo.bar_code.code_1_pic import barcode_perception
from detection.pro_handbook.sam_py_demo.bar_code.FixToCameraCoordinate import pixel_to_camera
from pathlib import Path
import cv2
import time
import json
from datetime import datetime

def decoded_to_jsonable(d):
    if d is None:
        return None
    if isinstance(d, list):
        return [decoded_to_jsonable(x) for x in d]  # 候補全部を保存
    return {
        "data": d.data.decode("utf-8", errors="ignore"),
        "type": d.type,
        "rect": {"left": d.rect.left, "top": d.rect.top, "width": d.rect.width, "height": d.rect.height},
        "polygon": [{"x": p.x, "y": p.y} for p in getattr(d, "polygon", [])],
        "quality": getattr(d, "quality", None),
        "orientation": getattr(d, "orientation", None),
    }

def bookshelf_barcode_sequence(barcode_number_input: str, shot_dir: Path):  
    

    time.sleep(3.0)
    color_frame = capture_one_depstech(shot_dir / "barcode_capture.png") #画像をキャプチャ
    barcode_indentification, barcode_data_output = barcode_perception(barcode_number_input, color_frame)
    print("[bookshelf barcode]barcode_indentification:", barcode_indentification)
    print("[bookshelf barcode]barcode_data_output:", barcode_data_output)
    rect = barcode_data_output.rect
    amr_X = pixel_to_camera(rect)
    print("[bookshelf barcode] amr_X:", amr_X)
    #if barcode_indentification:
        #u, v = pixel_to_camera(barcode_data_output.rect) #TODO　LiDARの点群からDepthを取得
        #print("[bookshelf barcode] u,v:", u, v)     
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = shot_dir / f"bookshelf_barcode_result_{ts}.json"

    payload = {
    "timestamp": ts,
    "barcode_number_input": barcode_number_input,
    "barcode_indentification": barcode_indentification,
    "barcode_data_output": decoded_to_jsonable(barcode_data_output),
    }


    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return barcode_indentification

if __name__ == "__main__":
    shot_dir = Path("/home/book/pro_book_SAM3/pro_hand_book_python/captures")
    barcode_number = "1H14-6"
    #barcode_number = input("barcode number?")
    bookshelf_barcode_sequence(barcode_number, shot_dir)