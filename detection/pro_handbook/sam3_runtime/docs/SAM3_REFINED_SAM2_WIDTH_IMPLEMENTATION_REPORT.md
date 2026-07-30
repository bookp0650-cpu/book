# SAM3 refined + SAM2 width implementation report

## Scope

This variant changes only the returned book width. It preserves the current
SAM3 refine-only recognition path, including offline PaddleOCR/SAM3 parallel
execution, score threshold 40, conservative mask refinement, original Depth,
median-range filtering, normal RANSAC, PCA, roll, `find_target_point()`, target
pixel, target 3D point, and `final.png`.

No SAM2 side-removal, side classification, Depth completion, pc2/pc3
switching, or case-specific condition was introduced.

## Implementation

- Recognition variant:
  `detection/pro_handbook/sam_py_demo/get_book_points_sam3_refined_sam2_width.py`
- Reused implementation:
  `detection/pro_handbook/sam_py_demo/modules/sam2_compatible_geometry.py`
- Evaluation:
  `detection/pro_handbook/sam3_runtime/tests/evaluate_sam3_refined_sam2_width_100cases.py`
- Integration candidate:
  `Retrieval_integration_SAM3_SAM2_WIDTH.py`

The offline entry point first calls the unchanged
`run_capture_and_pca_offline_sam3_refined_median_depth(..., mode="refine_only")`.
It then passes `selected_mask_refined.png`, the refine-only median-filtered
Depth image, the unchanged PCA-input PLY, camera intrinsics, and the saved OCR
axis to `estimate_sam2_compatible_geometry(...,
geometry_mode="sam2_width_only")`.

The adopted result is only `width.width_mm`. Roll, target pixel, and target 3D
point are copied directly from the refine-only result. SHA-256 checks before
and after the width calculation guard the refined mask and PCA-input PLY.

The width flow is:

1. Use the conservative-refinement output mask.
2. Use the selected OCR long axis and its perpendicular width axis.
3. Project valid mask pixels onto the width axis.
4. Take the 2nd-to-98th percentile span as pixel width.
5. Take median Depth within the final mask.
6. Convert using `fx`, `fy`, the width-axis direction, and Depth.
7. Convert metres to millimetres; flag values outside 2–150 mm.

`final.png` remains the refine-only output: original RGB background,
semi-transparent green refined mask, and the red projected
`find_target_point()` target. The width calculation does not redraw or move
the target.

## Saved 100-case evaluation

Output:
`captures/100test_sam3_refined_sam2_width_20260724_221309`

Saved artifacts from
`captures/100test_offline_SAM3_debug_20260724_173921` were reused. SAM3 and OCR
were not rerun.

| Metric | Result |
|---|---:|
| Success / failure | 94 / 6 |
| Within 1.0 mm | 44 |
| Within 1.5 mm | 63 |
| Within 2.0 mm | 83 |
| Mean absolute error | 1.156024 mm |
| Median absolute error | 1.024005 mm |
| Maximum absolute error | 3.786004 mm |
| Minimum absolute error | 0.011605 mm |
| Improved / worsened / same | 60 / 34 / 0 |
| Underestimate / overestimate | 60 / 34 |
| Largest improvement | case81, 8.527847 mm |
| Largest worsening | case96, 2.570793 mm worse |

The width matched the previous `sam2_width_only` result in every one of the 94
successful cases; maximum absolute numerical difference was 0.0 mm.

| Case | GT (mm) | Current SAM3 (mm) | SAM2-compatible width (mm) | Compatible abs. error (mm) |
|---:|---:|---:|---:|---:|
| 81 | 13.7 | 4.957485 | 13.914668 | 0.214668 |
| 82 | 13.7 | 13.226183 | 13.417669 | 0.282331 |
| 83 | 13.7 | 13.088801 | 14.204669 | 0.504669 |
| 84 | 13.7 | 12.411469 | 13.238506 | 0.461494 |
| 85 | 13.7 | 11.446894 | 12.674048 | 1.025952 |
| 96 | 10.9 | 10.453954 | 13.916839 | 3.016839 |
| 99 | 10.9 | 10.650121 | 13.527327 | 2.627327 |
| 64 | 12.0 | 12.064552 | 14.429367 | 2.429367 |
| 12 | 18.0 | 16.802317 | 15.036384 | 2.963616 |
| 98 | 10.9 | 10.671452 | 12.746662 | 1.846662 |

For all 94 successful cases:

- copied refined-mask SHA-256 matched the source;
- copied PCA-input PLY SHA-256 matched the source;
- roll difference was exactly 0 rad;
- target-point difference was exactly 0 mm.

The six failures were inherited from the saved baseline: cases 6–10 failed
the existing OCR/mask score threshold, and case50 lacked offline RGB-D input.

## Integration validation and limitation

`Retrieval_integration_SAM3_SAM2_WIDTH.py` differs from
`Retrieval_integration_SAM3.py` only in the recognition-function import and
call. Its `py_compile` check passed, and the recognition API imports with
signature `(query, sam_device="gpu", *, shot_dir=None)` and returns
`(roll, target_point, book_width_mm, shot_dir)`.

A direct import of the full integration file in the specified fixed virtual
environment could not complete because that environment does not expose
`rclpy`; the unchanged base integration has the same dependency. No ROS node
was started.

The requested basis `Retrieval_integration_editing2_SAM3.py` is absent from
the project, so `Retrieval_integration_editing2_SAM3_SAM2_WIDTH.py` was not
fabricated from another file. Supplying or identifying that exact basis is the
remaining prerequisite.

No existing protected source was modified. The legacy
`/home/book/pro_book` tree was read only. No robot, ROS node/topic, RealSense
live capture, Height Controller, model training, or model change was used.
