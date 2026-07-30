# No-mask-merge / no-side-filter comparison flow

## Scope

The production `sam_py_demo/get_book_points.py` remains unchanged. The isolated
comparison entry point is
`sam_py_demo/get_book_points_no_mask_merge_no_side_filter.py`.

## Located processing stages

1. Owner mask NMS: `sam3_runtime/vendor/owner_repo/core/mask_nms.py:
   apply_mask_nms()`. The service calls it after `Sam3Adapter.predict()` with
   mask IoU 0.5, metric `iou`, mode `suppress`. This is retained.
2. OCR/mask association: `get_book_points.py: merge_ocr_and_masks()` calls
   `match_text_to_mask_main()` and preserves the existing candidate rule.
3. Legacy non-book removal: the SAM2 `infer_for_storage.py` path calls
   `quality_filter()`, greedy box NMS, and mask-IoU de-dup. The SAM3 service
   client bypasses this legacy runner; the comparison module never invokes it.
4. Floating fragment merge: the SAM2 `infer_for_storage.py` path calls
   `merge_coaxial_rect_masks()` twice. The SAM3 service client bypasses it and
   the comparison module does not invoke it.
5. Median-depth filter: `get_book_points.py: save_masked_and_cropped()`. The
   OCR polygon intersection remains the preferred median reference and the
   existing +/-30 Z16 threshold is retained.
6. Normal RANSAC: `get_book_points.py:
   _fit_plane_ransac_open3d_for_spine()`. The comparison applies this once to
   all points surviving the median-depth filter and retains only plane inliers.
7. Side detection/removal stages in the current flow:
   `refine_mask_by_ocr_axis_band()`,
   `refine_mask_by_spine_column_length()`,
   `apply_one_sided_short_column_prune_after_final()`, and side-related parts
   of the column profile processing.
8. Post-side additional processing:
   `apply_post_ransac_plane_a95_column_prune()`, residual-policy evaluators
   `_evaluate_residual_policy_a_candidate()`,
   `_evaluate_residual_policy_b_width()`, and
   `_evaluate_residual_policy_ab_safe_interaction()`. Modes include
   `AB_SAFE`, `AB_SAFE_A_TUNED`, and `AB_SAFE_A_TUNED_V2`. These, plus
   reject/release, candidate re-evaluation, and completion, are not called.
9. PCA input: in the comparison module, `points_for_pca = points_ransac`
   without an intervening filter. Separate PLY files are saved from the same
   array and equality is asserted in the result.
10. Roll/width/position: retained helpers are
    `pca_axes_fix_dir()`, `estimate_book_width()`, and
    `find_target_point()`. The comparison returns roll, width in millimetres,
    and the existing `target_m`/`p_min_m`-meaning point.

## Flow comparison

| Stage | Current production flow | Comparison flow |
|---|---|---|
| SAM inference | Owner `Sam3Adapter` | Same |
| Score/min-area | 0.3 / 200 | Same |
| Owner mask NMS | IoU 0.5 suppress | Same |
| Legacy SAM2 quality/merge | Bypassed by SAM3 client | Explicitly not called |
| OCR and target selection | Existing matching | Same |
| Selected mask shaping/completion | May run downstream refinements | Skipped; selected SAM3 mask is used unchanged |
| Median-depth filter | OCR reference, +/-30 Z16 | Same |
| Normal plane RANSAC | Runs within the extended production flow | One explicit retained stage |
| Side/column/post-RANSAC filtering | May run | Skipped |
| Residual policies/reject-release | Mode-dependent | Skipped |
| PCA input | Final post-processed points | Exact normal-RANSAC output |
| PCA/roll/width/position | Existing helpers | Same core helpers |

The comparison deliberately saves:

- `selected_mask_before_legacy_postprocess.png` and
  `selected_mask_used_for_depth.png` from the same array.
- `pointcloud_after_normal_ransac.ply` and
  `pointcloud_sent_to_pca.ply` from the same point array.
