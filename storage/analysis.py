from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np

from storage.config import (
    SPACER_CONTACT_HEIGHT_MM,
    RECTANGULAR_BOTTLENECK_PERCENTILE,
    CONTACT_OPENING_MIN_VALID_ROWS,
    TEST_BOOK_WIDTH_MM,
    TWO_VIEW_BOTTLENECK_TIE_TOLERANCE_MM,
    REAL_WORLD_RIGHT_TARGET_INSET_PX,
    STORAGE_SPACE_UPPER_REACH_RATIO,
)

from storage.planner import finite_float


def _contiguous_x_runs(row_mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive x-runs for one binary mask row."""
    xs = np.flatnonzero(np.asarray(row_mask, dtype=bool))
    if xs.size == 0:
        return []
    split_points = np.flatnonzero(np.diff(xs) > 1) + 1
    groups = np.split(xs, split_points)
    return [(int(group[0]), int(group[-1])) for group in groups]


def estimate_contact_height_opening_mm(
    result_metadata: dict[str, Any],
    *,
    contact_height_mm: float = SPACER_CONTACT_HEIGHT_MM,
) -> dict[str, Any]:
    """Estimate selected-space width at a physical height on the shelf plane.

    The selected mask supplies the actual (possibly sloped) boundaries. Pixel
    rays are intersected with the existing RANSAC shelf-front plane, so no
    fixed px/mm scale is assumed. The less tilted boundary supplies the local
    physical vertical direction; the first valid mask row is the shelf-bottom
    reference because Storage_SAM3 explicitly records image-top of the space
    as the physical book-bottom side.
    """
    requested_height = finite_float(contact_height_mm)
    if requested_height is None or requested_height < 0.0:
        raise RuntimeError(
            f"invalid spacer contact height: {contact_height_mm}"
        )
    if result_metadata.get("target_y_reference") != (
        "image_top_of_selected_space_equals_physical_book_bottom_side"
    ):
        raise RuntimeError(
            "physical shelf-bottom reference is missing or unsupported"
        )

    selected_mask = np.asarray(result_metadata.get("selected_space_mask"))
    if selected_mask.ndim != 2 or selected_mask.size == 0:
        raise RuntimeError(
            "selected_space_mask is required for contact-height opening"
        )
    selected_mask = selected_mask.astype(bool, copy=False)

    plane = np.asarray(result_metadata.get("plane"), dtype=np.float64)
    if plane.shape != (4,) or not np.all(np.isfinite(plane)):
        raise RuntimeError("valid RANSAC shelf-front plane is required")
    normal_scale = float(np.linalg.norm(plane[:3]))
    if normal_scale <= 1.0e-12:
        raise RuntimeError("RANSAC shelf-front plane has zero normal")
    plane_normal = plane[:3] / normal_scale
    plane_d = float(plane[3] / normal_scale)

    shot_dir_value = result_metadata.get("shot_dir")
    if not shot_dir_value:
        raise RuntimeError("shot_dir is required to load camera intrinsics")
    intrinsics_path = Path(shot_dir_value) / "camera_intrinsics.json"
    try:
        intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"failed to load camera intrinsics: {intrinsics_path}"
        ) from exc

    fx = finite_float(intrinsics.get("fx"))
    fy = finite_float(intrinsics.get("fy"))
    ppx = finite_float(intrinsics.get("ppx"))
    ppy = finite_float(intrinsics.get("ppy"))
    if (
        fx is None
        or fy is None
        or ppx is None
        or ppy is None
        or fx <= 0.0
        or fy <= 0.0
    ):
        raise RuntimeError(f"invalid camera intrinsics: {intrinsics_path}")
    coeffs = np.asarray(intrinsics.get("coeffs", []), dtype=np.float64)
    if coeffs.size and (
        not np.all(np.isfinite(coeffs))
        or float(np.max(np.abs(coeffs))) > 1.0e-9
    ):
        raise RuntimeError(
            "nonzero/invalid distortion coefficients are unsupported for "
            "contact-height opening estimation"
        )

    def pixel_ray_plane_point(x_px: int, y_px: int) -> np.ndarray:
        ray = np.asarray(
            [(float(x_px) - ppx) / fx, (float(y_px) - ppy) / fy, 1.0],
            dtype=np.float64,
        )
        denominator = float(np.dot(plane_normal, ray))
        if abs(denominator) <= 1.0e-12:
            raise RuntimeError("pixel ray is parallel to the shelf-front plane")
        distance = -plane_d / denominator
        if not math.isfinite(distance) or distance <= 0.0:
            raise RuntimeError("pixel ray/plane intersection is behind camera")
        return distance * ray

    boundary_rows: list[dict[str, Any]] = []
    for y_px in range(selected_mask.shape[0]):
        runs = _contiguous_x_runs(selected_mask[y_px])
        if not runs:
            continue
        left_x, right_x = max(
            runs,
            key=lambda run: (run[1] - run[0] + 1, -run[0]),
        )
        if right_x <= left_x:
            continue
        boundary_rows.append(
            {
                "y": y_px,
                "left_x": left_x,
                "right_x": right_x,
                "left_point": pixel_ray_plane_point(left_x, y_px),
                "right_point": pixel_ray_plane_point(right_x, y_px),
            }
        )
    if len(boundary_rows) < CONTACT_OPENING_MIN_VALID_ROWS:
        raise RuntimeError(
            "too few valid selected-mask rows for contact-height opening: "
            f"{len(boundary_rows)}<{CONTACT_OPENING_MIN_VALID_ROWS}"
        )

    left_tilt = finite_float(
        result_metadata.get("left_tilt_from_vertical_deg")
    )
    right_tilt = finite_float(
        result_metadata.get("right_tilt_from_vertical_deg")
    )
    if left_tilt is None or right_tilt is None:
        raise RuntimeError("both boundary tilt values are required")
    upright_side = "left" if left_tilt <= right_tilt else "right"
    image_rows = np.asarray(
        [row["y"] for row in boundary_rows], dtype=np.float64
    )
    upright_points = np.asarray(
        [row[f"{upright_side}_point"] for row in boundary_rows],
        dtype=np.float64,
    )
    design = np.column_stack([image_rows, np.ones_like(image_rows)])
    boundary_fit, _, fit_rank, _ = np.linalg.lstsq(
        design, upright_points, rcond=None
    )
    if fit_rank < 2:
        raise RuntimeError("upright boundary 3D fit is rank deficient")
    vertical_slope = boundary_fit[0]
    vertical_slope -= plane_normal * float(
        np.dot(vertical_slope, plane_normal)
    )
    vertical_slope_norm = float(np.linalg.norm(vertical_slope))
    if vertical_slope_norm <= 1.0e-12:
        raise RuntimeError("physical vertical direction could not be estimated")
    physical_vertical = vertical_slope / vertical_slope_norm
    if float(np.dot(upright_points[-1] - upright_points[0], physical_vertical)) < 0:
        physical_vertical *= -1.0

    vertical_mm_per_image_row = float(
        np.dot(boundary_fit[0], physical_vertical) * 1000.0
    )
    if (
        not math.isfinite(vertical_mm_per_image_row)
        or vertical_mm_per_image_row <= 0.0
    ):
        raise RuntimeError(
            "invalid physical height scale from the upright boundary: "
            f"{vertical_mm_per_image_row} mm/row"
        )

    shelf_bottom_row = float(image_rows[0])
    requested_row = shelf_bottom_row + (
        requested_height / vertical_mm_per_image_row
    )
    if requested_row < image_rows[0] or requested_row > image_rows[-1]:
        raise RuntimeError(
            "selected-space mask has no section at spacer contact height: "
            f"requested_y={requested_row:.2f}, "
            f"valid_y=[{image_rows[0]:.0f},{image_rows[-1]:.0f}]"
        )
    selected_index = int(np.argmin(np.abs(image_rows - requested_row)))
    selected_row = boundary_rows[selected_index]

    physical_horizontal = np.cross(plane_normal, physical_vertical)
    horizontal_norm = float(np.linalg.norm(physical_horizontal))
    if horizontal_norm <= 1.0e-12:
        raise RuntimeError("physical horizontal direction could not be estimated")
    physical_horizontal /= horizontal_norm
    left_point = selected_row["left_point"]
    right_point = selected_row["right_point"]
    current_opening_mm = abs(
        float(np.dot(right_point - left_point, physical_horizontal))
    ) * 1000.0
    if not math.isfinite(current_opening_mm) or current_opening_mm < 0.0:
        raise RuntimeError(
            f"invalid contact-height opening width: {current_opening_mm}"
        )

    actual_height_mm = (
        float(selected_row["y"]) - shelf_bottom_row
    ) * vertical_mm_per_image_row
    representative_depth_m = float(
        0.5 * (left_point[2] + right_point[2])
    )
    return {
        "current_opening_mm": current_opening_mm,
        "contact_height_requested_mm": requested_height,
        "contact_height_actual_mm": actual_height_mm,
        "contact_height_error_mm": actual_height_mm - requested_height,
        "mask_row_y": int(selected_row["y"]),
        "mask_row_run_xy": [
            int(selected_row["left_x"]),
            int(selected_row["right_x"]),
        ],
        "pixel_opening_width_px": int(
            selected_row["right_x"] - selected_row["left_x"] + 1
        ),
        "representative_depth_m": representative_depth_m,
        "fx_px": fx,
        "fy_px": fy,
        "upright_boundary_side": upright_side,
        "vertical_mm_per_image_row": vertical_mm_per_image_row,
        "plane_source": "non_held_book_spine_ransac_plane",
        "camera_intrinsics_path": str(intrinsics_path),
        "valid_mask_row_count": len(boundary_rows),
    }


def calculate_bottleneck_percentile_mm(
    valid_row_widths_mm: list[float] | np.ndarray,
    *,
    percentile: float = RECTANGULAR_BOTTLENECK_PERCENTILE,
    minimum_valid_rows: int = CONTACT_OPENING_MIN_VALID_ROWS,
) -> float:
    """Return the configured lower-tail percentile of valid physical widths."""
    widths = np.asarray(valid_row_widths_mm, dtype=np.float64).reshape(-1)
    widths = widths[np.isfinite(widths) & (widths > 0.0)]
    if widths.size < int(minimum_valid_rows):
        raise RuntimeError(
            "too few valid rows for rectangular bottleneck: "
            f"{widths.size}<{int(minimum_valid_rows)}"
        )
    pct = finite_float(percentile)
    if pct is None or not 0.0 <= pct <= 100.0:
        raise RuntimeError(f"invalid rectangular bottleneck percentile: {percentile}")
    return float(np.percentile(widths, pct))


def estimate_selected_space_row_widths_mm(
    result_metadata: dict[str, Any],
    *,
    minimum_valid_rows: int = CONTACT_OPENING_MIN_VALID_ROWS,
) -> dict[str, Any]:
    """Measure valid selected-mask row widths on the RANSAC shelf plane."""
    selected_mask = np.asarray(result_metadata.get("selected_space_mask"))
    if selected_mask.ndim != 2 or selected_mask.size == 0:
        raise RuntimeError(
            "selected_space_mask is required for physical row widths"
        )
    selected_mask = selected_mask.astype(bool, copy=False)

    plane = np.asarray(result_metadata.get("plane"), dtype=np.float64)
    if plane.shape != (4,) or not np.all(np.isfinite(plane)):
        raise RuntimeError("valid RANSAC shelf-front plane is required")
    normal_scale = float(np.linalg.norm(plane[:3]))
    if normal_scale <= 1.0e-12:
        raise RuntimeError("RANSAC shelf-front plane has zero normal")
    plane_normal = plane[:3] / normal_scale
    plane_d = float(plane[3] / normal_scale)

    shot_dir_value = result_metadata.get("shot_dir")
    if not shot_dir_value:
        raise RuntimeError("shot_dir is required to load camera intrinsics")
    intrinsics_path = Path(shot_dir_value) / "camera_intrinsics.json"
    try:
        intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"failed to load camera intrinsics: {intrinsics_path}"
        ) from exc

    fx = finite_float(intrinsics.get("fx"))
    fy = finite_float(intrinsics.get("fy"))
    ppx = finite_float(intrinsics.get("ppx"))
    ppy = finite_float(intrinsics.get("ppy"))
    if (
        fx is None
        or fy is None
        or ppx is None
        or ppy is None
        or fx <= 0.0
        or fy <= 0.0
    ):
        raise RuntimeError(f"invalid camera intrinsics: {intrinsics_path}")
    coeffs = np.asarray(intrinsics.get("coeffs", []), dtype=np.float64)
    if coeffs.size and (
        not np.all(np.isfinite(coeffs))
        or float(np.max(np.abs(coeffs))) > 1.0e-9
    ):
        raise RuntimeError(
            "nonzero/invalid distortion coefficients are unsupported for "
            "selected-space row-width estimation"
        )

    def pixel_ray_plane_point(x_px: int, y_px: int) -> np.ndarray:
        ray = np.asarray(
            [(float(x_px) - ppx) / fx, (float(y_px) - ppy) / fy, 1.0],
            dtype=np.float64,
        )
        denominator = float(np.dot(plane_normal, ray))
        if abs(denominator) <= 1.0e-12:
            raise RuntimeError("pixel ray is parallel to shelf-front plane")
        distance = -plane_d / denominator
        if not math.isfinite(distance) or distance <= 0.0:
            raise RuntimeError("pixel ray/plane intersection is behind camera")
        point = distance * ray
        if not np.all(np.isfinite(point)):
            raise RuntimeError("pixel ray/plane intersection is nonfinite")
        return point

    boundary_rows: list[dict[str, Any]] = []
    invalid_row_count = 0
    for y_px in range(selected_mask.shape[0]):
        runs = _contiguous_x_runs(selected_mask[y_px])
        if not runs:
            continue
        left_x, right_x = max(
            runs,
            key=lambda run: (run[1] - run[0] + 1, -run[0]),
        )
        if right_x <= left_x:
            invalid_row_count += 1
            continue
        try:
            left_point = pixel_ray_plane_point(left_x, y_px)
            right_point = pixel_ray_plane_point(right_x, y_px)
        except RuntimeError:
            invalid_row_count += 1
            continue
        boundary_rows.append(
            {
                "y": int(y_px),
                "left_x": int(left_x),
                "right_x": int(right_x),
                "left_point": left_point,
                "right_point": right_point,
            }
        )

    if len(boundary_rows) < int(minimum_valid_rows):
        raise RuntimeError(
            "too few valid selected-mask rows for physical width estimation: "
            f"{len(boundary_rows)}<{int(minimum_valid_rows)}"
        )

    left_tilt = finite_float(
        result_metadata.get("left_tilt_from_vertical_deg")
    )
    right_tilt = finite_float(
        result_metadata.get("right_tilt_from_vertical_deg")
    )
    if left_tilt is None or right_tilt is None:
        raise RuntimeError("both boundary tilt values are required")
    upright_side = "left" if left_tilt <= right_tilt else "right"
    image_rows = np.asarray(
        [row["y"] for row in boundary_rows], dtype=np.float64
    )
    upright_points = np.asarray(
        [row[f"{upright_side}_point"] for row in boundary_rows],
        dtype=np.float64,
    )
    design = np.column_stack([image_rows, np.ones_like(image_rows)])
    boundary_fit, _, fit_rank, _ = np.linalg.lstsq(
        design, upright_points, rcond=None
    )
    if fit_rank < 2:
        raise RuntimeError("upright boundary 3D fit is rank deficient")
    physical_vertical = boundary_fit[0]
    physical_vertical -= plane_normal * float(
        np.dot(physical_vertical, plane_normal)
    )
    vertical_norm = float(np.linalg.norm(physical_vertical))
    if vertical_norm <= 1.0e-12:
        raise RuntimeError("physical vertical direction could not be estimated")
    physical_vertical /= vertical_norm

    physical_horizontal = np.cross(plane_normal, physical_vertical)
    horizontal_norm = float(np.linalg.norm(physical_horizontal))
    if horizontal_norm <= 1.0e-12:
        raise RuntimeError("physical horizontal direction could not be estimated")
    physical_horizontal /= horizontal_norm

    valid_widths_mm: list[float] = []
    valid_rows: list[int] = []
    for row in boundary_rows:
        width_mm = abs(
            float(
                np.dot(
                    row["right_point"] - row["left_point"],
                    physical_horizontal,
                )
            )
        ) * 1000.0
        if not math.isfinite(width_mm) or width_mm <= 0.0:
            invalid_row_count += 1
            continue
        valid_widths_mm.append(width_mm)
        valid_rows.append(int(row["y"]))

    if len(valid_widths_mm) < int(minimum_valid_rows):
        raise RuntimeError(
            "too few valid physical selected-space row widths: "
            f"{len(valid_widths_mm)}<{int(minimum_valid_rows)}"
        )

    return {
        "valid_row_count": len(valid_widths_mm),
        "invalid_row_count": int(invalid_row_count),
        "valid_row_y_min": min(valid_rows),
        "valid_row_y_max": max(valid_rows),
        "valid_row_y_px": valid_rows,
        "row_width_min_mm": float(np.min(valid_widths_mm)),
        "row_width_median_mm": float(np.median(valid_widths_mm)),
        "row_width_max_mm": float(np.max(valid_widths_mm)),
        "row_widths_mm": valid_widths_mm,
        "upright_boundary_side": upright_side,
        "camera_intrinsics_path": str(intrinsics_path),
        "width_method": (
            "largest continuous selected-mask run per row; endpoint rays "
            "intersected with RANSAC shelf-front plane; physical-horizontal "
            "3D separation"
        ),
    }
    
def evaluate_physical_bottom_narrowing_rule(
    *,
    physical_top_width_mm: float,
    physical_bottom_width_mm: float,
    minimum_space_width_mm: float,
    held_book_width_mm: float,
    width_trend_tolerance_mm: float = TWO_VIEW_BOTTLENECK_TIE_TOLERANCE_MM,
) -> dict[str, Any]:
    """Reject only a physically bottom-narrowing space that is too narrow."""
    physical_top = finite_float(physical_top_width_mm)
    physical_bottom = finite_float(physical_bottom_width_mm)
    minimum_space = finite_float(minimum_space_width_mm)
    held_book = finite_float(held_book_width_mm)
    trend_tolerance = finite_float(width_trend_tolerance_mm)
    if physical_top is None or physical_top <= 0.0:
        raise RuntimeError(
            f"physical top width must be finite and positive: {physical_top_width_mm}"
        )
    if physical_bottom is None or physical_bottom <= 0.0:
        raise RuntimeError(
            "physical bottom width must be finite and positive: "
            f"{physical_bottom_width_mm}"
        )
    if minimum_space is None or minimum_space <= 0.0:
        raise RuntimeError(
            "minimum space width must be finite and positive: "
            f"{minimum_space_width_mm}"
        )
    if held_book is None or held_book <= 0.0:
        raise RuntimeError(
            f"held book width must be finite and positive: {held_book_width_mm}"
        )
    if trend_tolerance is None or trend_tolerance < 0.0:
        raise RuntimeError(
            "width trend tolerance must be finite and nonnegative: "
            f"{width_trend_tolerance_mm}"
        )

    bottom_narrowing = physical_bottom < physical_top - trend_tolerance
    reject = bottom_narrowing and minimum_space < held_book
    return {
        "physical_top_width_mm": physical_top,
        "physical_bottom_width_mm": physical_bottom,
        "minimum_space_width_mm": minimum_space,
        "minimum_space_width_method": (
            "configured lower-tail percentile of valid physical row widths"
        ),
        "minimum_space_width_percentile": float(
            RECTANGULAR_BOTTLENECK_PERCENTILE
        ),
        "held_book_width_mm": held_book,
        "held_book_width_source": "TEST_BOOK_WIDTH_MM",
        "physical_width_trend_tolerance_mm": trend_tolerance,
        "bottom_narrowing": bottom_narrowing,
        "bottom_narrowing_width_reject": reject,
    }


def evaluate_selected_space_vertical_width_check(
    result_metadata: dict[str, Any],
    selected_space_width_info: dict[str, Any],
    *,
    minimum_space_width_mm: float,
    held_book_width_mm: float = TEST_BOOK_WIDTH_MM,
) -> dict[str, Any]:
    """Map image bands to physical top/bottom and apply the width rule."""
    expected_reference = (
        "image_top_of_selected_space_equals_physical_book_bottom_side"
    )
    if result_metadata.get("target_y_reference") != expected_reference:
        raise RuntimeError(
            "physical top/bottom mapping is missing or unsupported"
        )

    selected_space_id = result_metadata.get("selected_space_id")
    spaces = result_metadata.get("space_candidate_metadata")
    if not isinstance(spaces, list):
        raise RuntimeError("space candidate metadata is required for width bands")
    selected_space = next(
        (
            item
            for item in spaces
            if isinstance(item, dict)
            and item.get("space_id") == selected_space_id
        ),
        None,
    )
    if selected_space is None:
        raise RuntimeError(
            "selected space metadata is missing for physical width trend"
        )

    vertical_mapping = selected_space.get("image_to_physical_vertical_mapping")
    if (
        not isinstance(vertical_mapping, dict)
        or vertical_mapping.get("image_top_small_y")
        != "physical_book_bottom_side"
        or vertical_mapping.get("image_bottom_large_y")
        != "physical_book_top_side"
    ):
        raise RuntimeError(
            "selected space has an unsupported image-to-physical mapping"
        )

    def band_range(name: str) -> tuple[int, int]:
        value = selected_space.get(name)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise RuntimeError(f"selected space is missing {name}")
        start, end = int(value[0]), int(value[1])
        if end <= start:
            raise RuntimeError(f"selected space has invalid {name}: {value}")
        return start, end

    image_top_y_start, image_top_y_end = band_range(
        "image_top_band_y_range"
    )
    image_bottom_y_start, image_bottom_y_end = band_range(
        "image_bottom_band_y_range"
    )
    row_y_px = np.asarray(
        selected_space_width_info.get("valid_row_y_px"),
        dtype=np.int64,
    ).reshape(-1)
    row_widths_mm = np.asarray(
        selected_space_width_info.get("row_widths_mm"),
        dtype=np.float64,
    ).reshape(-1)
    if row_y_px.size == 0 or row_y_px.size != row_widths_mm.size:
        raise RuntimeError(
            "physical row positions and widths are missing or inconsistent"
        )

    physical_bottom_band = row_widths_mm[
        (row_y_px >= image_top_y_start) & (row_y_px < image_top_y_end)
    ]
    physical_top_band = row_widths_mm[
        (row_y_px >= image_bottom_y_start) & (row_y_px < image_bottom_y_end)
    ]
    if physical_bottom_band.size == 0 or physical_top_band.size == 0:
        raise RuntimeError(
            "physical top/bottom width bands contain no valid rows"
        )

    check = evaluate_physical_bottom_narrowing_rule(
        physical_top_width_mm=float(np.median(physical_top_band)),
        physical_bottom_width_mm=float(np.median(physical_bottom_band)),
        minimum_space_width_mm=minimum_space_width_mm,
        held_book_width_mm=held_book_width_mm,
    )
    check.update(
        {
            "physical_bottom_width_source": (
                "median physical row width in image-top/physical-bottom band"
            ),
            "physical_top_width_source": (
                "median physical row width in image-bottom/physical-top band"
            ),
            "physical_bottom_image_y_range": [
                image_top_y_start,
                image_top_y_end,
            ],
            "physical_top_image_y_range": [
                image_bottom_y_start,
                image_bottom_y_end,
            ],
        }
    )
    return check


def estimate_rectangular_bottleneck_mm(
    result_metadata: dict[str, Any],
    *,
    percentile: float = RECTANGULAR_BOTTLENECK_PERCENTILE,
    minimum_valid_rows: int = CONTACT_OPENING_MIN_VALID_ROWS,
) -> dict[str, Any]:
    """Preserve the existing rectangular 5-percentile bottleneck result."""
    width_info = estimate_selected_space_row_widths_mm(
        result_metadata,
        minimum_valid_rows=minimum_valid_rows,
    )
    bottleneck_mm = calculate_bottleneck_percentile_mm(
        width_info["row_widths_mm"],
        percentile=percentile,
        minimum_valid_rows=minimum_valid_rows,
    )
    return {
        **width_info,
        "bottleneck_opening_mm": bottleneck_mm,
        "bottleneck_percentile": float(percentile),
    }


def select_real_world_right_side_target_pixel(
    result_metadata: dict[str, Any],
    *,
    inset_px: int = REAL_WORLD_RIGHT_TARGET_INSET_PX,
) -> tuple[tuple[int, int], dict[str, Any]]:
    """Select an image-left, mask-interior target at the existing target y.

    The camera reverses the relevant physical left/right relationship, so the
    real-world right-book side is the image-left boundary of the selected
    space.  The original target's y coordinate is retained.  The contiguous
    mask run nearest the original target is used, which avoids crossing into a
    separate component if a row contains more than one run.
    """
    selected_mask = np.asarray(result_metadata.get("selected_space_mask"))
    if selected_mask.ndim != 2:
        raise RuntimeError(
            "selected_space_mask is required for right-side target selection"
        )
    selected_mask = selected_mask.astype(bool, copy=False)
    height, width = selected_mask.shape

    original_value = result_metadata.get("first_target_px_selected")
    if original_value is None:
        original_value = result_metadata.get("first_target_px")
    if original_value is None or len(original_value) != 2:
        raise RuntimeError("original target pixel is missing")
    original_x = int(round(float(original_value[0])))
    target_y = int(round(float(original_value[1])))
    if not 0 <= target_y < height:
        raise RuntimeError(
            f"original target y is outside selected mask: y={target_y}"
        )

    safe_inset = int(inset_px)
    if safe_inset < 0:
        raise ValueError(f"target inset must be nonnegative: {safe_inset}")
    runs = _contiguous_x_runs(selected_mask[target_y])
    if not runs:
        raise RuntimeError(
            f"selected space has no mask pixels at target y={target_y}"
        )

    def distance_to_original(run: tuple[int, int]) -> tuple[int, int]:
        run_left, run_right = run
        if original_x < run_left:
            distance = run_left - original_x
        elif original_x > run_right:
            distance = original_x - run_right
        else:
            distance = 0
        return distance, -(run_right - run_left + 1)

    run_left, run_right = min(runs, key=distance_to_original)
    run_width = run_right - run_left + 1
    minimum_run_width = 2 * safe_inset + 1
    if run_width < minimum_run_width:
        raise RuntimeError(
            "selected-space row is too narrow for the existing safety inset: "
            f"width={run_width}px, required={minimum_run_width}px"
        )

    target_x = int(np.clip(run_left + safe_inset, 0, width - 1))
    if not selected_mask[target_y, target_x]:
        raise AssertionError("modified right-side target left selected-space mask")

    info = {
        "physical_mapping": "image_left_equals_real_world_right",
        "original_target_px": [original_x, target_y],
        "modified_target_px": [target_x, target_y],
        "selected_row_run_xy": [run_left, run_right],
        "target_inset_px": safe_inset,
        "left_clearance_px": target_x - run_left,
        "right_clearance_px": run_right - target_x,
        "target_inside_selected_space_mask": True,
    }
    return (target_x, target_y), info


def build_real_world_right_side_target_camera(
    result_metadata: dict[str, Any],
    original_target_cam: Any,
    *,
    inset_px: int = REAL_WORLD_RIGHT_TARGET_INSET_PX,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Move only camera X to the image-left safe target at the same Y/Z.

    Current captures use zero-distortion aligned intrinsics.  Reusing the
    recognized target's camera Y and Z preserves its height/depth while the
    selected pixel changes only horizontally within the existing space mask.
    """
    modified_target_px, info = select_real_world_right_side_target_pixel(
        result_metadata,
        inset_px=inset_px,
    )
    original_cam = np.asarray(original_target_cam, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(original_cam)) or original_cam[2] <= 0.0:
        raise RuntimeError(f"invalid original camera target: {original_cam}")

    shot_dir_value = result_metadata.get("shot_dir")
    if not shot_dir_value:
        raise RuntimeError("shot_dir is missing; camera intrinsics unavailable")
    intrinsics_path = Path(shot_dir_value) / "camera_intrinsics.json"
    try:
        intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"failed to load camera intrinsics: {intrinsics_path}: {exc}"
        ) from exc

    fx = finite_float(intrinsics.get("fx"))
    ppx = finite_float(intrinsics.get("ppx"))
    coeffs = intrinsics.get("coeffs", [])
    if fx is None or fx <= 0.0 or ppx is None:
        raise RuntimeError(f"invalid camera intrinsics: {intrinsics_path}")
    if coeffs and any(abs(float(value)) > 1.0e-9 for value in coeffs):
        raise RuntimeError(
            "right-side target horizontal shift currently requires the "
            "zero-distortion aligned intrinsics used by saved captures"
        )

    modified_cam = original_cam.copy()
    modified_cam[0] = (
        (float(modified_target_px[0]) - ppx) / fx * original_cam[2]
    )
    if not np.all(np.isfinite(modified_cam)):
        raise RuntimeError(f"invalid modified camera target: {modified_cam}")

    info.update(
        {
            "camera_target_method": "horizontal_pixel_shift_same_camera_y_z",
            "camera_intrinsics_path": str(intrinsics_path),
            "original_target_cam_m": original_cam.tolist(),
            "modified_target_cam_m": modified_cam.tolist(),
        }
    )
    return modified_cam, info


def select_real_world_left_boundary_target_pixel(
    result_metadata: dict[str, Any],
) -> tuple[tuple[int, int], dict[str, Any]]:
    """Select the image-right boundary at the existing target y coordinate."""
    selected_mask = np.asarray(result_metadata.get("selected_space_mask"))
    if selected_mask.ndim != 2:
        raise RuntimeError(
            "selected_space_mask is required for real-world-left boundary target"
        )
    selected_mask = selected_mask.astype(bool, copy=False)
    height, width = selected_mask.shape

    original_value = result_metadata.get("first_target_px_selected")
    if original_value is None:
        original_value = result_metadata.get("first_target_px")
    if original_value is None or len(original_value) != 2:
        raise RuntimeError("original target pixel is missing")
    original_x = int(round(float(original_value[0])))
    target_y = int(round(float(original_value[1])))
    if not 0 <= target_y < height:
        raise RuntimeError(
            f"original target y is outside selected mask: y={target_y}"
        )

    runs = _contiguous_x_runs(selected_mask[target_y])
    if not runs:
        raise RuntimeError(
            f"selected space has no mask pixels at target y={target_y}"
        )

    def distance_to_original(run: tuple[int, int]) -> tuple[int, int]:
        run_left, run_right = run
        if original_x < run_left:
            distance = run_left - original_x
        elif original_x > run_right:
            distance = original_x - run_right
        else:
            distance = 0
        return distance, -(run_right - run_left + 1)

    run_left, run_right = min(runs, key=distance_to_original)
    target_x = int(np.clip(run_right, 0, width - 1))
    if not selected_mask[target_y, target_x]:
        raise AssertionError(
            "real-world-left boundary target left selected-space mask"
        )

    info = {
        "physical_mapping": "image_right_equals_real_world_left",
        "original_target_px": [original_x, target_y],
        "modified_target_px": [target_x, target_y],
        "selected_row_run_xy": [run_left, run_right],
        "target_boundary": "image_right_real_world_left",
        "new_target_offset_px": 0,
        "target_inside_selected_space_mask": True,
    }
    return (target_x, target_y), info


def build_real_world_left_boundary_target_camera(
    result_metadata: dict[str, Any],
    original_target_cam: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Move only camera X to the image-right/real-world-left boundary."""
    modified_target_px, info = select_real_world_left_boundary_target_pixel(
        result_metadata
    )
    original_cam = np.asarray(original_target_cam, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(original_cam)) or original_cam[2] <= 0.0:
        raise RuntimeError(f"invalid original camera target: {original_cam}")

    shot_dir_value = result_metadata.get("shot_dir")
    if not shot_dir_value:
        raise RuntimeError("shot_dir is missing; camera intrinsics unavailable")
    intrinsics_path = Path(shot_dir_value) / "camera_intrinsics.json"
    try:
        intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"failed to load camera intrinsics: {intrinsics_path}: {exc}"
        ) from exc

    fx = finite_float(intrinsics.get("fx"))
    ppx = finite_float(intrinsics.get("ppx"))
    coeffs = intrinsics.get("coeffs", [])
    if fx is None or fx <= 0.0 or ppx is None:
        raise RuntimeError(f"invalid camera intrinsics: {intrinsics_path}")
    if coeffs and any(abs(float(value)) > 1.0e-9 for value in coeffs):
        raise RuntimeError(
            "left-boundary target horizontal shift currently requires the "
            "zero-distortion aligned intrinsics used by saved captures"
        )

    modified_cam = original_cam.copy()
    modified_cam[0] = (
        (float(modified_target_px[0]) - ppx) / fx * original_cam[2]
    )
    if not np.all(np.isfinite(modified_cam)):
        raise RuntimeError(f"invalid modified camera target: {modified_cam}")

    info.update(
        {
            "camera_target_method": (
                "horizontal_pixel_shift_same_camera_y_z_to_image_right_boundary"
            ),
            "camera_intrinsics_path": str(intrinsics_path),
            "original_target_cam_m": original_cam.tolist(),
            "modified_target_cam_m": modified_cam.tolist(),
        }
    )
    return modified_cam, info


def _load_target_geometry_intrinsics(
    result_metadata: dict[str, Any],
    intrinsics_override: dict[str, Any] | None = None,
) -> tuple[dict[str, float], str]:
    """Load the aligned pinhole intrinsics without accessing any hardware."""
    if intrinsics_override is None:
        shot_dir_value = result_metadata.get("shot_dir")
        if not shot_dir_value:
            raise RuntimeError("shot_dir is missing; camera intrinsics unavailable")
        intrinsics_path = Path(shot_dir_value) / "camera_intrinsics.json"
        try:
            raw = json.loads(intrinsics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"failed to load camera intrinsics: {intrinsics_path}: {exc}"
            ) from exc
        source = str(intrinsics_path)
    else:
        raw = dict(intrinsics_override)
        source = "explicit_intrinsics_override"

    values = {key: finite_float(raw.get(key)) for key in ("fx", "fy", "ppx", "ppy")}
    if (
        values["fx"] is None
        or values["fy"] is None
        or values["ppx"] is None
        or values["ppy"] is None
        or values["fx"] <= 0.0
        or values["fy"] <= 0.0
    ):
        raise RuntimeError(f"invalid camera intrinsics: {source}")
    coeffs = np.asarray(raw.get("coeffs", []), dtype=np.float64)
    if coeffs.size and (
        not np.all(np.isfinite(coeffs))
        or float(np.max(np.abs(coeffs))) > 1.0e-9
    ):
        raise RuntimeError(
            "B-based target requires the zero-distortion aligned intrinsics "
            "used by saved storage captures"
        )
    return {
        key: float(cast(float, value))
        for key, value in values.items()
    }, source


def _pixel_ray_plane_point(
    x_px: float,
    y_px: float,
    intrinsics: dict[str, float],
    plane_normal: np.ndarray,
    plane_d: float,
) -> np.ndarray:
    ray = np.asarray(
        [
            (float(x_px) - intrinsics["ppx"]) / intrinsics["fx"],
            (float(y_px) - intrinsics["ppy"]) / intrinsics["fy"],
            1.0,
        ],
        dtype=np.float64,
    )
    denominator = float(np.dot(plane_normal, ray))
    if abs(denominator) <= 1.0e-12:
        raise RuntimeError("pixel ray is parallel to the RANSAC shelf plane")
    scale = -float(plane_d) / denominator
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("pixel ray/RANSAC plane intersection is behind camera")
    point = scale * ray
    if not np.all(np.isfinite(point)):
        raise RuntimeError("pixel ray/RANSAC plane intersection is nonfinite")
    return point


def build_physical_shelf_frame(
    result_metadata: dict[str, Any],
    *,
    intrinsics_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build vertical/normal/real-right axes on the saved RANSAC plane.

    Cross-product order is not trusted for horizontal sign.  The sign is
    explicitly aligned with the projected image-left direction because the
    established camera mapping is image-left == real-world RIGHT.
    """
    selected_mask = np.asarray(result_metadata.get("selected_space_mask"))
    if selected_mask.ndim != 2 or selected_mask.size == 0:
        raise RuntimeError("selected_space_mask is required for shelf axes")
    selected_mask = selected_mask.astype(bool, copy=False)
    height, width = selected_mask.shape

    plane = np.asarray(result_metadata.get("plane"), dtype=np.float64)
    if plane.shape != (4,) or not np.all(np.isfinite(plane)):
        raise RuntimeError("valid non-held-book RANSAC shelf plane is required")
    normal_norm = float(np.linalg.norm(plane[:3]))
    if normal_norm <= 1.0e-12:
        raise RuntimeError("RANSAC shelf plane has zero normal")
    plane_normal = plane[:3] / normal_norm
    plane_d = float(plane[3] / normal_norm)
    intrinsics, intrinsics_source = _load_target_geometry_intrinsics(
        result_metadata,
        intrinsics_override,
    )

    left_tilt = finite_float(result_metadata.get("left_tilt_from_vertical_deg"))
    right_tilt = finite_float(result_metadata.get("right_tilt_from_vertical_deg"))
    if left_tilt is None or right_tilt is None:
        raise RuntimeError("both selected-space boundary tilts are required")
    upright_side = "left" if left_tilt <= right_tilt else "right"

    image_rows: list[int] = []
    upright_points: list[np.ndarray] = []
    for y_px in range(height):
        runs = _contiguous_x_runs(selected_mask[y_px])
        if not runs:
            continue
        run_left, run_right = max(
            runs,
            key=lambda run: (run[1] - run[0] + 1, -run[0]),
        )
        boundary_x = run_left if upright_side == "left" else run_right
        image_rows.append(y_px)
        upright_points.append(
            _pixel_ray_plane_point(
                boundary_x,
                y_px,
                intrinsics,
                plane_normal,
                plane_d,
            )
        )
    if len(image_rows) < CONTACT_OPENING_MIN_VALID_ROWS:
        raise RuntimeError(
            "too few selected-mask rows for physical shelf frame: "
            f"{len(image_rows)}<{CONTACT_OPENING_MIN_VALID_ROWS}"
        )

    row_values = np.asarray(image_rows, dtype=np.float64)
    point_values = np.asarray(upright_points, dtype=np.float64)
    design = np.column_stack([row_values, np.ones_like(row_values)])
    fit, _, fit_rank, _ = np.linalg.lstsq(design, point_values, rcond=None)
    if fit_rank < 2:
        raise RuntimeError("upright boundary 3D fit is rank deficient")
    physical_vertical = fit[0]
    physical_vertical -= plane_normal * float(
        np.dot(physical_vertical, plane_normal)
    )
    vertical_norm = float(np.linalg.norm(physical_vertical))
    if vertical_norm <= 1.0e-12:
        raise RuntimeError("physical vertical direction could not be estimated")
    physical_vertical /= vertical_norm
    if float(np.dot(point_values[-1] - point_values[0], physical_vertical)) < 0.0:
        physical_vertical *= -1.0

    physical_horizontal_right = np.cross(plane_normal, physical_vertical)
    horizontal_norm = float(np.linalg.norm(physical_horizontal_right))
    if horizontal_norm <= 1.0e-12:
        raise RuntimeError("physical horizontal direction could not be estimated")
    physical_horizontal_right /= horizontal_norm

    mapping_y = int(round(float(np.median(row_values))))
    image_left_x = max(0, int(round(0.25 * (width - 1))))
    image_right_x = min(width - 1, int(round(0.75 * (width - 1))))
    image_left_point = _pixel_ray_plane_point(
        image_left_x,
        mapping_y,
        intrinsics,
        plane_normal,
        plane_d,
    )
    image_right_point = _pixel_ray_plane_point(
        image_right_x,
        mapping_y,
        intrinsics,
        plane_normal,
        plane_d,
    )
    image_left_direction = image_left_point - image_right_point
    if float(np.dot(physical_horizontal_right, image_left_direction)) < 0.0:
        physical_horizontal_right *= -1.0
    if float(np.dot(physical_horizontal_right, image_left_direction)) <= 0.0:
        raise RuntimeError("failed to align physical horizontal with image-left")

    return {
        "physical_vertical": physical_vertical,
        "plane_normal": plane_normal,
        "physical_horizontal_right": physical_horizontal_right,
        "physical_horizontal_mapping": "image-left -> real-world right",
        "horizontal_sign_reference_row_y": mapping_y,
        "horizontal_sign_reference_pixels": [
            [image_left_x, mapping_y],
            [image_right_x, mapping_y],
        ],
        "upright_boundary_side": upright_side,
        "intrinsics": intrinsics,
        "intrinsics_source": intrinsics_source,
    }


def build_left_tilted_book_right_edge_target_camera(
    result_metadata: dict[str, Any],
    original_target_cam: Any,
    *,
    book_width_mm: float = TEST_BOOK_WIDTH_MM,
    intrinsics_override: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replace only target horizontal position with X_ref = B + w right."""
    width_mm = finite_float(book_width_mm)
    if width_mm is None or width_mm <= 0.0:
        raise RuntimeError(f"book width must be finite and positive: {book_width_mm}")
    b_px_value = result_metadata.get(
        "real_world_left_neighbor_book_inner_bottom_px"
    )
    b_cam_value = result_metadata.get(
        "real_world_left_neighbor_book_inner_bottom_cam_m"
    )
    if b_px_value is None or len(b_px_value) != 2:
        raise RuntimeError("real-world LEFT neighbor B pixel is unavailable")
    if b_cam_value is None:
        raise RuntimeError("real-world LEFT neighbor B camera point is unavailable")
    b_pixel = [int(round(float(value))) for value in b_px_value]
    b_cam = np.asarray(b_cam_value, dtype=np.float64).reshape(3)
    original_cam = np.asarray(original_target_cam, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(b_cam)) or not np.all(np.isfinite(original_cam)):
        raise RuntimeError("B/original camera target contains nonfinite values")

    shelf_frame = build_physical_shelf_frame(
        result_metadata,
        intrinsics_override=intrinsics_override,
    )
    horizontal_right = shelf_frame["physical_horizontal_right"]
    physical_vertical = shelf_frame["physical_vertical"]
    plane_normal = shelf_frame["plane_normal"]
    x_reference_cam = b_cam + (width_mm / 1000.0) * horizontal_right
    horizontal_delta_m = float(
        np.dot(x_reference_cam - original_cam, horizontal_right)
    )
    modified_cam = original_cam + horizontal_delta_m * horizontal_right

    vertical_delta_m = float(np.dot(modified_cam - original_cam, physical_vertical))
    normal_delta_m = float(np.dot(modified_cam - original_cam, plane_normal))
    horizontal_error_m = float(
        np.dot(modified_cam - x_reference_cam, horizontal_right)
    )
    tolerance_m = 1.0e-9
    if (
        abs(vertical_delta_m) > tolerance_m
        or abs(normal_delta_m) > tolerance_m
        or abs(horizontal_error_m) > tolerance_m
    ):
        raise AssertionError(
            "B-based target component preservation failed: "
            f"vertical={vertical_delta_m}, normal={normal_delta_m}, "
            f"horizontal_error={horizontal_error_m}"
        )

    intrinsics = shelf_frame["intrinsics"]
    if x_reference_cam[2] <= 0.0:
        raise RuntimeError("X reference lies behind the camera")
    x_reference_pixel = [
        int(round(intrinsics["fx"] * x_reference_cam[0] / x_reference_cam[2] + intrinsics["ppx"])),
        int(round(intrinsics["fy"] * x_reference_cam[1] / x_reference_cam[2] + intrinsics["ppy"])),
    ]
    info = {
        "neighbor_book_mask_index": result_metadata.get(
            "real_world_left_neighbor_book_mask_index"
        ),
        "B_pixel": b_pixel,
        "B_camera_m": b_cam.tolist(),
        "physical_horizontal_direction_camera": horizontal_right.tolist(),
        "physical_vertical_direction_camera": physical_vertical.tolist(),
        "shelf_plane_normal_camera": plane_normal.tolist(),
        "physical_horizontal_mapping": shelf_frame[
            "physical_horizontal_mapping"
        ],
        "book_width_mm": width_mm,
        "X_reference_definition": "B + book_width toward real-world right",
        "X_reference_camera_m": x_reference_cam.tolist(),
        "X_reference_pixel": x_reference_pixel,
        "original_target_camera_m": original_cam.tolist(),
        "modified_target_camera_m": modified_cam.tolist(),
        "horizontal_target_shift_mm": horizontal_delta_m * 1000.0,
        "vertical_component_delta_mm": vertical_delta_m * 1000.0,
        "normal_component_delta_mm": normal_delta_m * 1000.0,
        "horizontal_component_error_mm": horizontal_error_m * 1000.0,
        "vertical_component_preserved": True,
        "normal_depth_component_preserved": True,
        "upright_boundary_side": shelf_frame["upright_boundary_side"],
        "intrinsics_source": shelf_frame["intrinsics_source"],
        "target_semantics": (
            "X is the desired held-book right-edge horizontal reference; "
            "the existing recognition-target-to-storage calibration is reused"
        ),
    }
    return modified_cam, info

def save_left_tilted_book_target_geometry_overlay(
    result_metadata: dict[str, Any],
    target_info: dict[str, Any],
) -> str:
    """Save selected space, exact neighbor mask, B, and X as a new image."""
    import cv2

    shot_dir_value = result_metadata.get("shot_dir")
    if not shot_dir_value:
        raise RuntimeError("shot_dir is required for B/X debug overlay")
    shot_dir = Path(shot_dir_value)
    input_path = shot_dir / "input.png"
    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"failed to load debug input image: {input_path}")
    selected_mask = np.asarray(
        result_metadata.get("selected_space_mask"), dtype=bool
    )
    neighbor_mask = np.asarray(
        result_metadata.get("real_world_left_neighbor_book_mask"), dtype=bool
    )
    if selected_mask.shape != image_bgr.shape[:2]:
        raise RuntimeError("selected-space mask/image shape mismatch for overlay")
    if neighbor_mask.shape != image_bgr.shape[:2]:
        raise RuntimeError("neighbor-book mask/image shape mismatch for overlay")

    overlay = image_bgr.copy()
    for mask, color in (
        (selected_mask, np.asarray([255, 200, 0], dtype=np.float32)),
        (neighbor_mask, np.asarray([0, 165, 255], dtype=np.float32)),
    ):
        overlay[mask] = (
            0.65 * overlay[mask].astype(np.float32) + 0.35 * color
        ).astype(np.uint8)

    b_pixel = tuple(int(value) for value in target_info["B_pixel"])
    x_pixel = tuple(int(value) for value in target_info["X_reference_pixel"])
    cv2.line(overlay, b_pixel, x_pixel, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(overlay, b_pixel, 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(overlay, x_pixel, 6, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.putText(
        overlay,
        "B",
        (b_pixel[0] + 7, b_pixel[1] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "X",
        (x_pixel[0] + 7, x_pixel[1] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    label_position = (
        max(5, min(b_pixel[0], x_pixel[0])),
        max(22, min(b_pixel[1], x_pixel[1]) - 18),
    )
    cv2.putText(
        overlay,
        f"w={float(target_info['book_width_mm']):.0f}mm",
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    output_path = shot_dir / "left_tilted_book_target_geometry_overlay.png"
    suffix = 1
    while output_path.exists():
        output_path = shot_dir / (
            f"left_tilted_book_target_geometry_overlay_{suffix:03d}.png"
        )
        suffix += 1
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"failed to save B/X debug overlay: {output_path}")
    return str(output_path)


def _selected_space_mask_height_px(result_metadata: dict[str, Any]) -> int:
    selected_mask = np.asarray(result_metadata.get("selected_space_mask"))
    if selected_mask.ndim != 2 or selected_mask.size == 0:
        raise RuntimeError(
            "selected_space_mask is required for two-view mask height"
        )
    valid_y = np.flatnonzero(np.any(selected_mask.astype(bool), axis=1))
    if valid_y.size == 0:
        raise RuntimeError("selected_space_mask has no valid pixels")
    return int(valid_y[-1] - valid_y[0] + 1)


def evaluate_storage_space_upper_reach(
    result_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Check that the selected space reaches the non-held book-spine tops."""

    selected_mask = np.asarray(
        result_metadata.get("selected_space_mask")
    )

    if selected_mask.ndim != 2 or selected_mask.size == 0:
        raise RuntimeError(
            "selected_space_mask is required for upper-reach validation"
        )

    selected_y = np.flatnonzero(
        np.any(selected_mask.astype(bool), axis=1)
    )

    if selected_y.size == 0:
        raise RuntimeError(
            "selected_space_mask has no valid pixels"
        )

    selected_space_y_max = int(selected_y[-1])

    detections = result_metadata.get(
        "held_book_detection"
    )

    if not isinstance(detections, list):
        detections = []

    spine_y_max_list: list[int] = []
    spine_height_list: list[int] = []

    for detection in detections:
        if (
            not isinstance(detection, dict)
            or detection.get("is_held_book")
        ):
            continue

        bbox = detection.get("bbox_xyxy")

        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            continue

        y_min = int(bbox[1])

        # bbox_xyxy は half-open bbox
        y_max = int(bbox[3]) - 1

        height_px = y_max - y_min + 1

        if height_px <= 0:
            continue

        spine_y_max_list.append(y_max)
        spine_height_list.append(height_px)

    non_hold_count = len(spine_y_max_list)

    if non_hold_count == 0:
        return {
            "storage_space_upper_reach_valid": False,
            "storage_space_upper_reach_reason": (
                "no non-held book spine is available"
            ),
            "storage_space_upper_reach_ratio": (
                STORAGE_SPACE_UPPER_REACH_RATIO
            ),
            "storage_space_upper_reach_non_hold_count": 0,
            "storage_space_upper_reach_reference_top_y_px": None,
            "storage_space_upper_reach_reference_height_px": None,
            "storage_space_upper_reach_allowed_gap_px": None,
            "storage_space_upper_reach_required_y_px": None,
            "storage_space_upper_reach_selected_y_max_px": (
                selected_space_y_max
            ),
        }

    reference_top_y = float(
        np.median(spine_y_max_list)
    )

    reference_height_px = float(
        np.median(spine_height_list)
    )

    allowed_gap_px = (
        reference_height_px
        * (1.0 - STORAGE_SPACE_UPPER_REACH_RATIO)
    )

    required_y = (
        reference_top_y
        - allowed_gap_px
    )

    valid = bool(
        selected_space_y_max >= required_y
    )

    return {
        "storage_space_upper_reach_valid": valid,
        "storage_space_upper_reach_reason": (
            "selected space reaches the required y"
            if valid
            else "selected space does not reach the required y"
        ),
        "storage_space_upper_reach_ratio": (
            STORAGE_SPACE_UPPER_REACH_RATIO
        ),
        "storage_space_upper_reach_non_hold_count": (
            non_hold_count
        ),
        "storage_space_upper_reach_reference_top_y_px": (
            reference_top_y
        ),
        "storage_space_upper_reach_reference_height_px": (
            reference_height_px
        ),
        "storage_space_upper_reach_allowed_gap_px": (
            allowed_gap_px
        ),
        "storage_space_upper_reach_required_y_px": (
            required_y
        ),
        "storage_space_upper_reach_selected_y_max_px": (
            selected_space_y_max
        ),
    }
    
def log_storage_space_upper_reach_check(
    check: dict[str, Any],
    label: str,
) -> None:
    print("\n[STORAGE SPACE UPPER REACH CHECK]")
    print(f"capture = {label}")

    print(
        "non-hold book spine count = "
        f"{check['storage_space_upper_reach_non_hold_count']}"
    )

    reference_top_y = check[
        "storage_space_upper_reach_reference_top_y_px"
    ]

    reference_height = check[
        "storage_space_upper_reach_reference_height_px"
    ]

    allowed_gap = check[
        "storage_space_upper_reach_allowed_gap_px"
    ]

    required_y = check[
        "storage_space_upper_reach_required_y_px"
    ]

    if reference_top_y is None:
        print("reference top y = unavailable")
        print("reference height = unavailable")
    else:
        print(
            f"reference top y = "
            f"{reference_top_y:.1f} px"
        )
        print(
            f"reference height = "
            f"{reference_height:.1f} px"
        )

    print(
        "required upper reach ratio = "
        f"{check['storage_space_upper_reach_ratio'] * 100.0:.1f}%"
    )

    if allowed_gap is None:
        print("allowed gap = unavailable")
        print("required y = unavailable")
    else:
        print(
            f"allowed gap = "
            f"{allowed_gap:.1f} px"
        )
        print(
            f"required y = "
            f"{required_y:.1f} px"
        )

    print(
        "selected space y_max = "
        f"{check['storage_space_upper_reach_selected_y_max_px']} px"
    )

    if check["storage_space_upper_reach_valid"]:
        print("result = VALID")
    else:
        print(
            "result = INVALID -> current capture unusable"
        )