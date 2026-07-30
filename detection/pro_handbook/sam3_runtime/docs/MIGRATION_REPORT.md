# Migration report

Migration assets are contained under `detection/pro_handbook/sam3_runtime`, except that the source checkpoint was moved from the new project root into `models/` as explicitly requested. Neither Retrieval integration file was edited. `get_book_points.py` now chooses the SAM3 service adapter at its existing runner boundary; OCR/depth/point-cloud logic and the public return tuple are unchanged.

## Audit findings

The old and new detection directories were initially identical except for excluded venv contents. Numerous legacy absolute `/home/book/pro_book` paths remain in diagnostic, barcode, revised and older scripts. The production path resolver in `get_book_points.py` handles OCR relatively, but its default ONNX arguments are now ignored by the SAM3 adapter. The repository has substantial pre-existing user changes, including both integration/detection files; they were preserved.

## Shared source and completed validation

`vendor/sam301.zip` SHA-256 is `43511a9e...74253`; owner YAML is `30b0ccd1...81678`. The archive was integrity/path checked and extracted without Python caches. Its `.git` entry is empty, so source commit comparison remains unavailable. Python 3.12, strict checkpoint load, offline BPE, standalone/service/adapter fixed-image inference, OCR, saved RGB-D recognition-only, and both integration imports now pass. Full route results are in `tests/outputs/case_001/comparison.json`.

Compared with official `facebookresearch/sam3` HEAD `46957e47805eaa273f4aa7bbbd25a88bca9108ce`, the shared source has three code differences plus the book-spine training config. The only model-path difference is `sam3/model/vitdet.py`: it adds a gradient-aware training branch while retaining the official fused path under inference/no-grad. The other differences are training-only (`sam3_video_dataset.py`, `trainer.py`). Therefore the owner's “no inference behavior change” statement is consistent with the code path exercised here, although the source file is not byte-identical to official HEAD.
