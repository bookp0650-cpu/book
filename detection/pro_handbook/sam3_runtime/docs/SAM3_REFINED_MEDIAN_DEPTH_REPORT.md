# SAM3 refined-mask / median-depth comparison report

Date: 2026-07-24 (Asia/Tokyo)

## Outcome

A new isolated comparison variant was implemented. It combines:

1. conservative OCR-anchored connected-component refinement; and
2. replacement of every final-mask pixel depth by one representative median,
   followed by direct point-cloud-to-PCA processing without a depth-range
   filter or RANSAC deletion.

The stable recognition module and all integration modules remain unchanged.
The new variant is not connected to an integration entry point.

Saved RGB-D and previously selected raw SAM3 masks were evaluated in 20 cases:
12 single-component clean cases and all 8 available multi-component noisy
cases. All four modes succeeded in all 20 cases. All 12 clean masks were exact
no-ops, including identical raw/refined PNG SHA-256 values. No baseline success
became a failure.

The standard candidate remains `refine_and_median_flatten`, but the results do
not support unconditional integration yet: width error improved in 9 cases and
worsened in 11, including one 1.79 mm degradation. More scenes and ground truth
for roll and 3D target position are needed.

## New implementation

- `sam3_mask_refinement.py`
  - OCR-overlap-based anchor selection
  - OCR polygon PCA/long-edge direction and anchor-PCA fallback
  - component metrics and conservative delete policy
  - safety validation and raw-mask fallback
  - refinement debug images and JSON
- `get_book_points_sam3_refined_median_depth.py`
  - four comparison modes
  - offline and live-compatible public functions
  - representative median selection
  - flat-depth point-cloud construction
  - direct unchanged PCA input
  - green final mask and red target point
- `evaluate_refined_median_depth_saved_cases.py`
  - saved-selection/RGB-D replay evaluator

Public APIs:

```python
run_capture_and_pca_offline_sam3_refined_median_depth(
    query, shot_dir, sam_device="gpu", mode="refine_and_median_flatten"
)

run_capture_and_pca_sam3_refined_median_depth(
    query, sam_device="gpu", mode="refine_and_median_flatten"
)
```

The live-compatible function was created but not executed. No RealSense device
was opened.

## Refinement policy

Connected components use 8-connectivity. The anchor is not chosen from area
alone. Candidates are ranked by:

1. any OCR polygon overlap;
2. containing the OCR center;
3. OCR overlap ratio;
4. distance to OCR center; and
5. area only as the final tie breaker.

If neither OCR overlap nor OCR-center containment supports an anchor, deletion
is disabled and the raw mask is used.

The primary a-axis is the OCR polygon's 2D PCA axis. Fallbacks are the OCR
polygon's longest edge and then anchor-mask PCA. For every component the
implementation records area and ratios, bbox, centroid, OCR overlap, OCR center
containment, nearest anchor distance, a-gap, b-distance, valid depth count,
median, MAD, image-edge contact, decision, evidence flags, and reason.

An extra component is removed only when all mandatory conditions hold:

- it is very small relative to both the raw mask and anchor;
- it is clearly separated relative to anchor diagonal;
- it has no OCR overlap and does not contain the OCR center;
- it is outside the a-axis extension or substantially displaced on b; and
- at least four independent isolation signals are true.

Single-component masks return immediately as a semantic no-op through the same
safety logic. Uncertain components are kept.

Post-decision safety checks require complete anchor preservation, complete OCR
overlap preservation, non-empty and sufficiently large output, removal below
2.5%, long-axis change below 5 degrees, and at least 30 valid depth pixels.
Any failure restores the raw mask and continues recognition.

## Depth flattening

Representative-depth source priority:

1. anchor eroded once with a 3x3 kernel;
2. anchor/OCR intersection;
3. complete anchor;
4. complete final mask.

Each source must contain at least 30 finite, positive depth samples. The
evaluation selected `anchor_eroded_3x3` for all 20 cases. Depth zero and
non-finite samples are excluded from median estimation.

The source records valid count and ratio, raw median, MAD, p10, and p90.
`target_mask_depth_flattened.npy` is a derived array; the original
`after_init_depth.npy` is never overwritten. Every final-mask pixel, including
pixels whose original depth was zero, receives the representative median.
Pixels outside the mask remain zero.

For the two median modes:

```text
pointcloud_from_flattened_mask == pointcloud_sent_to_pca
point count == final mask area
```

This was verified by array equality, point counts, and identical PLY SHA-256
files in every evaluated case. No median-window deletion, normal RANSAC,
side-surface filtering, residual policy, reject-release, or point completion is
used in these modes.

## Modes

| Mode | Mask | Depth/PCA path |
|---|---|---|
| `baseline` | stable raw selected mask | stable median range + normal RANSAC |
| `refine_only` | conservatively refined mask | stable median range + normal RANSAC |
| `median_flatten_only` | raw selected mask | flat representative depth, no point deletion |
| `refine_and_median_flatten` | conservatively refined mask | flat representative depth, no point deletion |

All four modes succeeded in 20/20 saved cases.

## Clean cases

The 12 clean cases were:

- `20260723_165044_no_mask_merge_no_side_filter`
- `20260723_170222_no_mask_merge_no_side_filter`
- `20260723_170323_live_no_mask_merge_no_side_filter`
- `20260724_101803_live_no_mask_merge_no_side_filter`
- `20260724_103142_live_no_mask_merge_no_side_filter`
- `20260724_104121_live_no_mask_merge_no_side_filter`
- `20260724_104335_live_no_mask_merge_no_side_filter`
- `20260724_104438_live_no_mask_merge_no_side_filter`
- `20260724_104541_live_no_mask_merge_no_side_filter`
- `20260724_104645_live_no_mask_merge_no_side_filter`
- `20260724_104756_live_no_mask_merge_no_side_filter`
- `20260724_104906_live_no_mask_merge_no_side_filter`

Each had exactly one connected component. Results:

- refinement no-op: 12/12;
- raw/refined array equality: 12/12;
- raw/refined PNG SHA-256 equality: 12/12;
- anchor retained: 12/12;
- OCR overlap retained: 12/12;
- new recognition failure: 0.

## Multi-component cases

| Case | Component areas (px) | Anchor | Removed | Raw -> refined area |
|---|---|---:|---|---:|
| `20260724_124349...` | 15543, 9, 25, 3, 165, 478, 325 | 1 | none | 16548 -> 16548 |
| `20260724_144704...` | 16638, 132, 1214, 22, 13, 12 | 1 | none | 18031 -> 18031 |
| `20260724_155742...` | 16875, 10, 4, 18, 2, 10 | 1 | none | 16919 -> 16919 |
| `20260724_130935...` | 20789, 1, 3, 9 | 1 | none | 20802 -> 20802 |
| `20260724_142735...` | 15341, 6, 49, 5 | 1 | 2, 4 | 15401 -> 15390 |
| `20260723_182607...` | 14236, 6 | 1 | none | 14242 -> 14242 |
| `20260724_143545...` | 15395, 13 | 1 | 2 | 15408 -> 15395 |
| `20260724_154852...` | 17414, 33 | 1 | none | 17447 -> 17447 |

Only two cases met the high-confidence deletion policy:

- `20260724_142735...`: removed 6 px and 5 px components. Both had zero OCR
  overlap, were approximately 46-53 px from anchor, strongly b-axis displaced,
  outside the anchor bbox, and below 0.04% of raw area.
- `20260724_143545...`: removed one 13 px component. It had zero OCR overlap,
  was 51 px from anchor, strongly b-axis displaced, outside the anchor bbox,
  and 0.084% of raw area.

Small components in the other six cases were retained because the full set of
independent evidence was not met. This is intentional.

## Numerical effects

The known width for the evaluated title is 14.8 mm.

`refine_only` versus baseline:

- median absolute-width-error change: -0.001 mm;
- improved: 10 cases;
- worsened: 9 cases;
- range: -0.215 to +0.257 mm.

`median_flatten_only` versus baseline:

- median absolute-width-error change: +0.019 mm;
- mean change: +0.102 mm;
- improved: 9 cases;
- worsened: 11 cases;
- range: -0.941 to +1.790 mm.

`refine_and_median_flatten` was nearly identical to
`median_flatten_only`, because only 11 or 13 pixels were removed in the two
effective refinement cases:

- median absolute-width-error change: +0.019 mm;
- mean change: +0.102 mm;
- improved: 9 cases;
- worsened: 11 cases;
- range: -0.941 to +1.790 mm.

Largest flattening improvements:

- `20260723_170323...`: error improved by 0.941 mm.
- `20260724_143545...`: error improved by 0.318 mm.
- `20260724_104541...`: error improved by 0.192 mm.

Largest degradations:

- `20260724_130935...`: error worsened by 1.790 mm.
- `20260724_155742...`: error worsened by 0.790 mm.
- `20260724_154852...`: error worsened by 0.456 mm.

The median target-point displacement versus baseline was 6.92 mm, with a
2.07-17.09 mm range. Roll-axis sign can flip by 180 degrees because PCA axes
are undirected; after treating antiparallel axes as equivalent, most changes
were small, but roll and target position lack ground truth in these captures.

## Debug outputs

Each evaluated non-baseline mode has an output directory under:

`detection/pro_handbook/sam3_runtime/tests/outputs/refined_median_depth_saved_cases/`

The standard mode includes raw/refined masks and overlays, component IDs,
OCR-axis overlay, anchor/kept/removed masks, raw and flattened depth
visualizations, median source, derived depth NPY, both PLYs, `final.png`, all
three requested JSON diagnostics, and `offline_run_console.log`.

`final.png` uses the mask actually sent through the mode as a green translucent
region and the existing `find_target_point()` result as a red filled circle.
No yellow contour is drawn.

## Limitations and recommendation

The SAM3 service was started twice for a fresh end-to-end rerun. Both attempts
failed during model initialization with
`CUDA CUBLAS_STATUS_NOT_INITIALIZED`; each failed service was stopped. No
model, environment, driver, or CUDA setting was changed. Therefore this
evaluation replayed previously saved SAM3-selected masks, OCR outputs, RGB,
and Depth. The changed refinement/depth/PCA path was fully executed, but a
fresh SAM3/OCR inference was not.

Recommendation:

- keep `refine_only` as the safer current comparison candidate;
- retain `refine_and_median_flatten` as an experimental diagnostic mode;
- do not connect either mode to integration code yet;
- repeat fresh end-to-end tests after the unrelated SAM3 service startup issue
  is resolved;
- acquire width, roll, and 3D target ground truth across more books and
  viewpoints before deciding whether flat depth should be enabled globally.

## Safety and protected files

No stable recognition module, integration module, service manager, SAM3 model,
OCR model, OCR offline setting, virtual environment, ROS topic, robot setting,
or RealSense setting was modified. No robot, Height Controller, manipulator,
hand, ROS publisher, or RealSense live capture was used. Nothing under
`/home/book/pro_book` was written.

