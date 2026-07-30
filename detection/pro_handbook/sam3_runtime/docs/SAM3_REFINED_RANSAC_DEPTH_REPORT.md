# SAM3 refined-mask / RANSAC Depth-completion report

Date: 2026-07-24 (Asia/Tokyo)

## Outcome

A separate comparison variant was created in which RANSAC estimates a book
surface plane but never selects/deletes the PCA point cloud. Valid measured
Depth within 8 mm orthogonal plane distance is preserved. Invalid Depth and
valid Depth farther than 8 mm are replaced by the pixel-ray/plane intersection.
Median Depth is not used as a replacement or fallback.

The same saved 20-case cohort from the median-depth evaluation was used:
12 clean single-component masks and 8 noisy multi-component masks. All four
modes succeeded in 20/20 cases. Clean refinement remained an exact no-op in
12/12 cases and created no new recognition failure.

The standard candidate `refine_and_ransac_complete` is **not recommended for
integration**. Width error improved in 7 cases but worsened in 13. Two clean
cases degraded by about 7.43 mm and one clean safe-fallback case degraded by
8.24 mm. The method needs stricter plane/application confidence and additional
ground truth before further consideration.

## New files and flow

- `sam3_ransac_depth_completion.py`: plane fitting, quality checks, pixel-ray
  prediction, Depth classification/replacement, debug output.
- `get_book_points_sam3_refined_ransac_depth.py`: fresh-inference-compatible
  four-mode recognition variant and public APIs.
- `evaluate_refined_ransac_depth_saved_cases.py`: same-cohort saved RGB-D
  evaluator.

Public APIs:

```python
run_capture_and_pca_offline_sam3_refined_ransac_depth(
    query, shot_dir, sam_device="gpu",
    mode="refine_and_ransac_complete",
)

run_capture_and_pca_sam3_refined_ransac_depth(
    query, sam_device="gpu",
    mode="refine_and_ransac_complete",
)
```

The live-compatible API was not executed.

New Depth flow:

```text
raw selected SAM3 mask
-> unchanged conservative mask refinement
-> anchor eroded 3x3 (fallback: anchor, then final mask)
-> valid measured 3D points
-> RANSAC plane [a,b,c,d]
-> predict Z_plane for every final-mask pixel
-> keep valid measurements within 8 mm
-> replace invalid or >8 mm measurements with Z_plane
-> convert every completed pixel to XYZ
-> send the exact same unfiltered XYZ array to PCA
```

## Difference from stable RANSAC

The protected stable function uses Open3D `segment_plane` with:

- distance threshold: 0.008 m;
- RANSAC sample count: 3;
- iterations: 1200;
- returns: plane coefficients, an inlier mask, and counts/ratio;
- it does not set a random seed.

The stable comparison path then keeps only the RANSAC inliers. The new module
keeps the same fitting parameters and sets Open3D's process-local seed to zero
where supported. It records normalized plane coefficients, inliers, outliers,
and residual statistics, but never uses the inlier mask to delete final points.

## Plane input and prediction

All 20 cases selected `anchor_eroded_3x3` as the RANSAC input region. No case
needed the anchor/final-mask input fallback.

For each final-mask pixel:

```text
denominator =
    a * (u - cx) / fx
  + b * (v - cy) / fy
  + c

Z_plane = -d / denominator
```

The implementation rejects near-zero denominators, non-finite/non-positive
predictions, predictions outside 0.10-3.00 m, low plane z-normal magnitude,
low inlier ratio, excessive median residual, and insufficient input points.

Plane quality defaults:

- input points >= 100;
- inlier ratio >= 0.50;
- residual median <= 8 mm;
- normalized `abs(c) >= 0.20`;
- valid predicted Depth for every final-mask pixel.

Quality failure never falls back to median completion. The output retains only
the raw valid mask Depth, performs no point deletion, and continues if at least
three points remain.

## Modes and success

| Mode | Result |
|---|---:|
| `baseline` | 20/20 |
| `refine_only` | 20/20 |
| `ransac_complete_only` | 20/20 |
| `refine_and_ransac_complete` | 20/20 |

`baseline` and `refine_only` are the already evaluated results for this exact
cohort. The two new RANSAC-completion modes were executed from saved raw SAM3
masks, OCR, RGB, and Depth.

## Mask refinement

The existing `sam3_mask_refinement.py` was reused without modification.

- clean no-op: 12/12;
- clean new failures: 0;
- noisy deletion cases: 2/8;
- `20260724_142735...`: components 2 and 4 removed (6 and 5 px);
- `20260724_143545...`: component 2 removed (13 px).

All other uncertain components were retained. OCR overlap and the anchor were
fully preserved.

## RANSAC and completion summary

RANSAC input counts ranged from 3,983 to 19,487 points. Inlier ratios ranged
from 0.904 to 1.000 among accepted planes. Input residual medians ranged from
0.000696 to 0.003351 m; p90 ranged from 0.001748 to 0.007621 m.

One case used safe fallback:

- `20260723_170323_live_no_mask_merge_no_side_filter`
- input: 3,983 points;
- inliers: 3,799 (95.38%);
- median residual: 0.002811 m;
- p90 residual: 0.005093 m;
- reason: invalid plane-predicted Depth occurred within the final mask;
- fallback behavior: keep all 4,610 raw valid Depth points, no completion and
  no deletion.

No evaluated capture contained zero/invalid Depth inside the selected final
mask, so `invalid_depth_replaced` was zero in this cohort. Replacement therefore
consisted entirely of measured plane-outlier Depth:

- 18 cases: under 10% (including the 0%-replacement safe fallback);
- 2 cases: 10-30%;
- 0 cases: 30-50%;
- 0 cases: 50% or more;
- 1 fallback case: 0% artificial replacement.

The two highest accepted replacement ratios were:

- `20260724_130935...`: 2,254 / 20,802 = 10.84%;
- `20260724_142735...`: 1,569 / 15,390 = 10.19%.

There were no 50%-replacement warnings.

## Per-case standard-mode summary

| Case | RANSAC input | Inliers | Ratio | Kept | Invalid replaced | Outlier replaced | Replacement ratio | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `20260723_165044...` | 13038 | 12662 | 0.971 | 13789 | 0 | 326 | 2.31% | no |
| `20260723_170222...` | 13038 | 12662 | 0.971 | 13789 | 0 | 326 | 2.31% | no |
| `20260723_170323...` | 3983 | 3799 | 0.954 | 4610 | 0 | 0 | 0.00% | yes |
| `20260724_101803...` | 14444 | 14444 | 1.000 | 15523 | 0 | 0 | 0.00% | no |
| `20260724_103142...` | 15292 | 15292 | 1.000 | 16272 | 0 | 104 | 0.64% | no |
| `20260724_104121...` | 14934 | 14934 | 1.000 | 15966 | 0 | 51 | 0.32% | no |
| `20260724_104335...` | 14807 | 14807 | 1.000 | 15890 | 0 | 0 | 0.00% | no |
| `20260724_104438...` | 13119 | 13119 | 1.000 | 14221 | 0 | 0 | 0.00% | no |
| `20260724_104541...` | 15637 | 15637 | 1.000 | 16714 | 0 | 12 | 0.07% | no |
| `20260724_104645...` | 15321 | 15287 | 0.998 | 16340 | 0 | 96 | 0.58% | no |
| `20260724_104756...` | 11499 | 11459 | 0.997 | 12577 | 0 | 51 | 0.40% | no |
| `20260724_104906...` | 15197 | 15197 | 1.000 | 16321 | 0 | 8 | 0.05% | no |
| `20260724_124349...` | 14432 | 14432 | 1.000 | 16548 | 0 | 0 | 0.00% | no |
| `20260724_144704...` | 15258 | 15258 | 1.000 | 18031 | 0 | 0 | 0.00% | no |
| `20260724_155742...` | 15742 | 15742 | 1.000 | 16916 | 0 | 3 | 0.02% | no |
| `20260724_130935...` | 19487 | 17615 | 0.904 | 18548 | 0 | 2254 | 10.84% | no |
| `20260724_142735...` | 14262 | 12935 | 0.907 | 13821 | 0 | 1569 | 10.19% | no |
| `20260723_182607...` | 13171 | 13171 | 1.000 | 14242 | 0 | 0 | 0.00% | no |
| `20260724_143545...` | 14323 | 14128 | 0.986 | 15086 | 0 | 309 | 2.01% | no |
| `20260724_154852...` | 16289 | 16289 | 1.000 | 17441 | 0 | 6 | 0.03% | no |

Full plane coefficients, normals, residual mean/max, predicted Depth ranges,
per-case roll, width, target point, and timings are in the comparison JSON and
each case's `ransac_plane_result.json`.

## Point-cloud invariants

For every new-mode case:

```text
pointcloud_from_ransac_completed_depth
    == pointcloud_sent_to_pca
```

Array identity, point count, and PLY SHA-256 equality were recorded. For
accepted planes, point count equals final-mask area. The fallback case also had
valid raw Depth at every mask pixel, so it retained the same equality.

No median-range filter, RANSAC inlier selection, RANSAC outlier deletion,
side-surface filter, residual policy, statistical filter, or DBSCAN was used.

## Width, roll, and target effects

Against baseline, `refine_and_ransac_complete`:

- width-error improved: 7 cases;
- width-error worsened: 13 cases;
- median error change: +0.059 mm;
- mean error change: +1.182 mm;
- maximum improvement: 0.148 mm;
- maximum degradation: 8.243 mm.

Large degradations:

- `20260723_165044...`: width 15.402 -> 7.968 mm;
- `20260723_170222...`: width 15.402 -> 7.968 mm;
- fallback `20260723_170323...`: width 11.768 -> 3.525 mm;
- `20260724_130935...`: width 18.868 -> 21.285 mm.

The first two demonstrate that a high inlier ratio and acceptable residuals do
not by themselves guarantee that replacing a small number of measured pixels
preserves the downstream width estimator. The fallback case demonstrates that
using all raw points without the stable RANSAC deletion can itself be unsuitable
for the current width algorithm.

Equivalent-axis roll difference versus baseline had median 0.053 degrees and
range -0.386 to +9.153 degrees. Target-point displacement had median 6.71 mm
and range 4.34-28.45 mm. These captures have no roll/3D target ground truth.

## Comparison with `refine_only`

`refine_only` remained substantially safer:

- median width-error change versus baseline: approximately -0.001 mm;
- improved: 10 cases;
- worsened: 8 cases;
- median equivalent roll change: 0.052 degrees.

The RANSAC-completion standard mode should not replace it.

## Comparison with median flattening

The previous median standard mode was not rerun or modified. Using its existing
JSON:

- RANSAC completion had lower width error than median flattening in 9 cases;
- it had higher error in 11 cases;
- median RANSAC-minus-median error change: +0.063 mm;
- mean change: +1.080 mm;
- range: -0.744 to +9.185 mm;
- median target-point separation: 3.47 mm.

RANSAC completion preserves most measured Depth, unlike median flattening, but
this did not improve aggregate width accuracy on the present cohort.

## Debug outputs

Each new-mode case under
`tests/outputs/refined_ransac_depth_saved_cases/` contains the required masks,
OCR axis, anchor, RANSAC input region and PLY, inlier/outlier PLYs, plane
overlay, raw/predicted/completed Depth visualizations, replacement
classification, completed Depth NPY, both final point-cloud PLYs, `final.png`,
and all requested JSON/log files.

Classification colors are:

- green: raw Depth kept;
- blue: invalid Depth replaced;
- red: plane-outlier Depth replaced.

`final.png` displays the actual final mask as a green translucent region and
the existing target point as a red filled circle. It has no yellow contour.

## Limitation and recommendation

Fresh SAM3 service startup was not attempted again because the immediately
preceding task reproduced `CUDA CUBLAS_STATUS_NOT_INITIALIZED` twice, and the
instruction explicitly permits saved-selection replay in that condition.
The changed refinement, plane estimation, completion, point-cloud, and PCA
paths were fully executed.

Recommendation:

- do not connect `refine_and_ransac_complete` to integration code;
- retain it only as a diagnostic comparison;
- continue to prefer `refine_only` among these experiments;
- add downstream width-estimator compatibility checks to any future plane
  completion design;
- test captures with actual missing Depth, because this cohort contained none;
- acquire ground truth for width, roll, and 3D target position.

## Protected scope

The stable module, median variant, refinement module, integration code, service
manager, SAM3/OCR models, OCR offline settings, environments, ROS/robot/
RealSense settings, and `/home/book/pro_book` were not modified. No robot,
Height Controller, manipulator, hand, ROS publisher, or RealSense live capture
was used.
