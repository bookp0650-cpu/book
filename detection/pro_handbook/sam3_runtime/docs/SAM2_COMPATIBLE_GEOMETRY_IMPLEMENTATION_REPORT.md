# SAM2-compatible geometry implementation report

## New implementation

- `sam_py_demo/modules/sam2_compatible_geometry.py`
  ports the traced SAM2 terminal geometry formulas.
- `get_book_points_sam3_refined_sam2_geometry.py`
  provides the isolated SAM3 + conservative refinement + SAM2 terminal
  geometry offline variant.
- `sam3_runtime/tests/evaluate_sam2_compatible_geometry_100cases.py`
  evaluates five geometry modes from saved artifacts only.

No stable recognition file or integration file was edited.

## Modes

| Mode | Roll | Width | Target |
|---|---|---|---|
| `sam3_current_geometry` | current | current 3D PCA slice | current |
| `sam2_roll_only` | legacy PCA-pc1 `atan2` | current | current |
| `sam2_width_only` | current | legacy 2D mask-axis/Depth conversion | current |
| `sam2_target_only` | current | current | legacy `find_target_point` |
| `sam2_all_geometry` | legacy | legacy | legacy |

Static tracing established that current and legacy roll formulas are the same,
and current and legacy target functions are the same. Therefore the only
material terminal formula change is width.

## Mask stage

The conservative SAM3 refinement output is mapped to legacy `mask01`
immediately before `calculate_yaw`. Clean SAM3 masks remain no-op. The
refine-only median-range Depth and final saved point cloud are reused without
inference.

This preserves recognition and mask selection, but it does not replay all
legacy conditional side-removal branches. Results are therefore described as
terminal-geometry compatible.

## Saved SAM2 reproduction

The un-suffixed final mask/Depth/point arrays in case81 reproduce the saved
offline SAM2 result (`pca_result_offline.json`, width about 32.981 mm).
The earlier online result (`pca_result.json`, width about 15.552 mm) remains,
but its exact final arrays were overwritten by the later offline processing.
It cannot be faithfully recomputed from the remaining saved intermediates.

## Evaluation

Output:

`/home/book/pro_book_SAM3/pro_hand_book_python/captures/100test_sam2_compatible_geometry_20260724_214950`

- 94 saved successful cases evaluated; six inherited recognition/input
  failures were not retried.
- No SAM3, SAM2, or OCR inference was run.
- No live camera, ROS, robot, or service was used.
- Each successful case records all five modes and debug overlays.

Detailed aggregate values are in `summary.json` and
`SAM2_COMPATIBLE_GEOMETRY_100CASE_REPORT.md` under the output root.

## Adoption status

The compatible width reduces aggregate error and fixes the case81 PCA-axis
failure. Roll and target merely reproduce the same formulas and point-cloud
outputs already used by the current SAM3 flow.

This is not yet recommended as a direct integration replacement because:

1. 34 of 94 cases have worse width error.
2. The exact legacy online case81 intermediates for the 15.552 mm result are
   unavailable.
3. The new module maps refined SAM3 mask to the legacy final geometry stage
   rather than replaying all legacy conditional side-removal branches.
4. Roll and target have no ground truth; equality only establishes processing
   consistency.
