# Integration compatibility

Both `Retrieval_integration.py` and `Retrieval_integration_editing.py` import `detection.pro_handbook.sam_py_demo.get_book_points.run_capture_and_pca`, call it with `query=book_name`, and expect `(roll, p_xmax, book_width, shot_dir)`.

Only the internal mask producer changed. `Sam3BatchInfer.infer_masks()` presents the former runner interface and returns a list of `bool` masks at original image size plus book metadata containing XYXY boxes and float scores. Existing OCR selection, depth correction, point-cloud and PCA stages remain untouched. BGR camera frames are converted to RGB PIL before the client, matching the old boundary.
