# SAM2 geometry estimation flow

## Scope and entry points

Read-only source investigated:

`/home/book/pro_book/pro_hand_book_python/detection/pro_handbook/sam_py_demo/get_book_points.py`

- Online entry: `run_capture_and_pca()` at lines 9466-9544.
- Offline entry: `run_capture_and_pca_offline()` at lines 9547-9610.
- Both call `_run_recognition_core_like_offline()` at line 8093.
- Terminal result is returned at lines 9407-9463 as
  `(theta_rad, target_point, book_width_mm, shot_dir)`.

## Actual call and data flow

1. PaddleOCR and SAM2 inference select `sel_idx` and `mask01`.
2. The selected mask passes Depth prefilter, conditional shape refinement,
   side-column removal, OCR-reference plane filtering, and optional
   post-RANSAC a95 pruning in `_run_recognition_core_like_offline()`.
3. Immediately before geometry, the legacy code saves the final `mask01` and
   `depth_masked` (lines 9235-9259). This is the mask/Depth stage corresponding
   to `final.png`.
4. `calculate_yaw(mask01, depth_masked, intr, depth_scale)` converts every
   valid final mask/Depth pixel into the camera-frame point cloud `pts_f`
   (lines 9261-9271).
5. `pca_axes_fix_dir(pts_f)` returns `mean`, `pc1`, and `pc2` (line 9295).
6. Roll is `atan2(pc1_y, pc1_x)` after checking the XY norm (lines 9296-9298).
7. Width has two conditional paths:
   - Refined/non-rectangular path:
     `estimate_book_width_from_filtered_mask_axis()` (lines 9300-9313).
   - Clean rectangular path: legacy 3D PCA slice width fallback
     `estimate_book_width()` (lines 9314-9319).
8. `find_target_point(pts_f)` returns the target (lines 9377-9378).
9. Results are written to `pca_result.json` or `pca_result_offline.json`
   (lines 9407-9417).

## Roll

Implementation:

- PCA: `/home/book/pro_book/pro_hand_book_python/detection/pro_handbook/sam_py_demo/modules/pca_vector.py`, lines 7-38.
- Roll: legacy `get_book_points.py`, lines 9294-9298.

The point cloud is centered and decomposed by SVD. `pc1` is the first row of
`Vt`; its sign is fixed so its camera-X component is non-positive. Roll is the
angle of the camera-frame `pc1` projection in the camera XY plane relative to
camera +X. It is not an OCR-only angle and is not transformed into a robot
base frame. Since a PCA axis is unoriented, comparisons must use pi
periodicity.

The current SAM3 refine-only flow calls the same `pca_axes_fix_dir()` and the
same `atan2` expression. With an identical final point cloud, roll is already
SAM2-compatible.

## Width

Legacy 2D width implementation:

`estimate_book_width_from_filtered_mask_axis()` in legacy
`get_book_points.py`, lines 2537-2626.

Inputs:

- final `mask01`;
- final `depth_masked`;
- image long axis and center from side/refinement metadata;
- camera `fx`, `fy`;
- `depth_scale`.

Formula:

1. `normal = [-axis_y, axis_x]`.
2. Project valid final mask pixels onto `normal`.
3. `width_px = percentile_98(t) - percentile_2(t)`.
4. `z = median(depth_masked[valid]) * depth_scale`.
5. `metres_per_pixel = z * sqrt((normal_x/fx)^2 + (normal_y/fy)^2)`.
6. `width_m = width_px * metres_per_pixel`.
7. Accept only 2-150 mm; otherwise fall back to 3D PCA slice width.

The current SAM3 refine-only candidate instead always calls
`modules/book_width.py`, which slices a 3D PCA point cloud along `pc1`, measures
2-98 percentile ranges along `pc2`, and takes the median. This is the material
geometry difference found in case81.

## Target point

Implementation:

`/home/book/pro_book/pro_hand_book_python/detection/pro_handbook/sam_py_demo/modules/grip_point.py`,
lines 5-48.

Given final camera-frame points:

1. Find `y_min` and `y_max`.
2. Set target band center to `y_min + 0.1 m`.
3. Select points within `+/-0.003 m`.
4. Return the candidate with minimum camera-X.

The returned point is already a 3D camera-frame point in metres. No separate
target-pixel selection or robot-frame coordinate conversion occurs in this
recognition function. A target pixel used for debug display is only the
projection of this selected 3D point.

The current SAM3 refine-only flow imports and calls the same
`find_target_point()` implementation. With an identical final point cloud, the
target is already SAM2-compatible.

## Camera model and units

- Stored RGB/Depth size: 1280 x 720.
- `fx=908.1617431640625`, `fy=906.4829711914062`.
- `ppx=637.79833984375`, `ppy=371.0213928222656`.
- `depth_scale=0.0010000000474974513 m/raw-unit`.
- Point cloud and target: camera-frame metres.
- Width return value: millimetres.
- Roll: radians.

## SAM2/SAM3 stage mapping

| SAM2 variable | SAM2 stage | SAM3 corresponding value | Basis | Residual difference |
|---|---|---|---|---|
| `mask01` | final mask immediately before `calculate_yaw` | conservative `selected_mask_refined` | Both are the selected target region entering terminal geometry | SAM2 reached this stage after its extensive side/RANSAC mask processing |
| `depth_masked` | final Depth masked by `mask01` | refine-only median-range filtered Depth | Same valid-mask Depth role | Earlier SAM2 conditional filters are not re-applied |
| `pts_f` | output of `calculate_yaw` | saved final refine-only/RANSAC point cloud | Both are final camera-frame geometry points | Exact upstream point-removal histories differ |
| `column_info.axis` | image long axis | refinement `axis_uv` | Both are OCR/mask-aligned spine axes | SAM3 refinement is conservative and may retain regions SAM2 removed |

## Failure and fallback

- Too few valid mask/Depth pixels: 2D width fails.
- Missing or invalid axis/center: 2D width fails.
- Invalid median Depth: 2D width fails.
- Width outside 2-150 mm: 2D width fails.
- The legacy caller then falls back to 3D PCA slice width.
- `find_target_point()` fails if no points lie in the target Y band.

## Important compatibility boundary

The newly added module reproduces the legacy *terminal* roll, width, and target
formulas. It deliberately does not claim byte-for-byte reproduction of every
legacy mask side-removal branch. The SAM3 conservative-refinement mask is
mapped to the legacy final geometry mask stage. This boundary is recorded in
every result JSON and is why the new variant remains a comparison candidate,
not an integration replacement.
