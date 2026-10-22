from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from detection.pro_handbook.sam_py_demo.book_mask_postprocess import (
    load_linked_ocr_polygons,
    postprocess_selected_mask,
)


def _vertical_ocr(x0: float, y0: float, width: float = 6.0, height: float = 50.0):
    polygon = np.asarray(
        [
            [x0, y0],
            [x0 + width, y0],
            [x0 + width, y0 + height],
            [x0, y0 + height],
        ],
        dtype=np.float64,
    )
    return {"polygon": polygon, "center": polygon.mean(axis=0), "text": "book"}


class BookMaskPostprocessTest(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((220, 120), dtype=bool)
        self.mask[10:200, 40:61] = True
        self.mask[80:89, 61:72] = True
        self.ocr = [
            _vertical_ocr(47, 35),
            _vertical_ocr(48, 95),
            _vertical_ocr(47, 145),
            # A horizontal orientation outlier must not rotate the estimate.
            {
                "polygon": np.asarray([[5, 5], [55, 5], [55, 11], [5, 11]], float),
                "center": np.asarray([30, 8], float),
                "text": "outlier",
            },
        ]

    def _config(self, method: str):
        return {
            "enabled": True,
            "method": method,
            "ocr_aspect_ratio_min": 1.5,
            "orientation_outlier_deg": 15,
            "boundary_margin_ratio": 0.05,
            "median_window": 31,
            "min_area_retention": 0.7,
            "min_width_retention": 0.8,
            "save_debug_images": False,
        }

    def test_axis_methods_remove_protrusion_without_adding_pixels(self):
        for method in ("ocr_axis_ransac", "ocr_axis_smooth"):
            with self.subTest(method=method):
                result = postprocess_selected_mask(
                    self.mask, self.ocr, self._config(method)
                )
                self.assertFalse(result.info["fallback"])
                self.assertFalse(np.any(result.mask & ~self.mask))
                self.assertLess(result.mask.sum(), self.mask.sum())
                self.assertFalse(result.mask[84, 70])
                self.assertTrue(result.mask[84, 50])
                self.assertGreaterEqual(result.info["num_valid_ocr"], 3)

    def test_no_valid_ocr_falls_back_bit_exactly(self):
        square = {
            "polygon": np.asarray([[45, 45], [55, 45], [55, 55], [45, 55]], float),
            "center": np.asarray([50, 50], float),
            "text": "square",
        }
        result = postprocess_selected_mask(
            self.mask, [square], self._config("ocr_axis_ransac")
        )
        self.assertTrue(result.info["fallback"])
        self.assertEqual(result.info["fallback_reason"], "no_valid_ocr")
        np.testing.assert_array_equal(result.mask, self.mask)

    def test_morphology_removes_isolated_component_and_never_adds(self):
        mask = self.mask.copy()
        mask[2, 2] = True
        result = postprocess_selected_mask(mask, [], self._config("morphology"))
        self.assertFalse(result.info["fallback"])
        self.assertFalse(result.mask[2, 2])
        self.assertFalse(np.any(result.mask & ~mask))

    def test_area_guard_returns_original(self):
        config = self._config("ocr_axis_smooth")
        config["min_area_retention"] = 0.9999
        result = postprocess_selected_mask(self.mask, self.ocr, config)
        self.assertTrue(result.info["fallback"])
        self.assertTrue(
            result.info["fallback_reason"].startswith(
                "area_retention_below_threshold:"
            )
        )
        np.testing.assert_array_equal(result.mask, self.mask)

    def test_remove_islands_keeps_ocr_overlap_component_bit_exactly(self):
        mask = np.zeros((80, 100), dtype=bool)
        mask[10:40, 10:40] = True  # Larger, but not linked to OCR.
        mask[20:35, 65:80] = True
        polygon = np.asarray([[67, 22], [77, 22], [77, 32], [67, 32]], float)
        linked = [{"polygon": polygon, "center": polygon.mean(axis=0), "text": "x"}]

        result = postprocess_selected_mask(
            mask, linked, self._config("remove_islands")
        )
        expected = np.zeros_like(mask)
        expected[20:35, 65:80] = True
        np.testing.assert_array_equal(result.mask, expected)
        self.assertEqual(result.info["component_count"], 2)
        self.assertEqual(result.info["removed_component_count"], 1)
        self.assertEqual(result.info["removed_pixel_count"], 900)
        self.assertEqual(
            result.info["main_component_source"], "maximum_ocr_overlap"
        )
        self.assertFalse(np.any(result.mask & ~mask))

    def test_remove_islands_without_ocr_keeps_largest_component(self):
        mask = np.zeros((60, 80), dtype=bool)
        mask[5:25, 5:25] = True
        mask[40:45, 60:65] = True
        result = postprocess_selected_mask(
            mask, [], self._config("remove_islands")
        )
        expected = np.zeros_like(mask)
        expected[5:25, 5:25] = True
        np.testing.assert_array_equal(result.mask, expected)
        self.assertEqual(
            result.info["main_component_source"], "largest_area_component"
        )
        self.assertEqual(result.info["removed_pixel_count"], 25)

    def test_remove_islands_does_not_modify_single_component(self):
        mask = np.zeros((60, 80), dtype=bool)
        mask[5:25, 5:25] = True
        result = postprocess_selected_mask(
            mask, [], self._config("remove_islands")
        )
        np.testing.assert_array_equal(result.mask, mask)
        self.assertEqual(result.info["component_count"], 1)
        self.assertEqual(result.info["removed_component_count"], 0)
        self.assertEqual(result.info["removed_pixel_count"], 0)

    def test_remove_islands_keeps_connected_protrusion_bit_exactly(self):
        mask = np.zeros((80, 100), dtype=bool)
        mask[10:70, 40:60] = True
        mask[30:45, 60:75] = True  # Connected protrusion: part of main component.
        mask[5:8, 5:8] = True  # Disconnected island: the only pixels to remove.
        polygon = np.asarray([[43, 20], [57, 20], [57, 55], [43, 55]], float)
        linked = [{"polygon": polygon, "center": polygon.mean(axis=0), "text": "x"}]

        result = postprocess_selected_mask(
            mask, linked, self._config("remove_islands")
        )

        expected = mask.copy()
        expected[5:8, 5:8] = False
        np.testing.assert_array_equal(result.mask, expected)
        self.assertTrue(np.all(result.mask[30:45, 60:75]))
        self.assertEqual(result.info["component_count"], 2)
        self.assertEqual(result.info["removed_component_count"], 1)
        self.assertEqual(result.info["removed_pixel_count"], 9)

    def test_load_linked_ocr_polygons_from_per_angle_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shot_dir = Path(temp_dir)
            (shot_dir / "ocr_sam3_matching.json").write_text(
                json.dumps(
                    {
                        "forced_angle": 0,
                        "per_angle_assignments": {
                            "0deg": [
                                {
                                    "ocr_index": 1,
                                    "text": "linked",
                                    "matched": "mask_2",
                                }
                            ],
                            "+30deg": [
                                {
                                    "ocr_index": 1,
                                    "text": "other",
                                    "matched": "mask_1",
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (shot_dir / "ocr_result.json").write_text(
                json.dumps(
                    {
                        "per_angle_ocr": {
                            "angles": {
                                "0deg": {
                                    "dt_polys": [
                                        [[10, 20], [30, 20], [30, 40], [10, 40]]
                                    ],
                                    "rec_texts": ["linked"],
                                },
                                "+30deg": {
                                    "dt_polys": [
                                        [[50, 20], [70, 20], [70, 40], [50, 40]]
                                    ],
                                    "rec_texts": ["other"],
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            linked = load_linked_ocr_polygons(shot_dir, 2, (80, 100))

        self.assertEqual(len(linked), 1)
        self.assertEqual(linked[0]["source_angle"], "0deg")
        self.assertEqual(linked[0]["text"], "linked")
        np.testing.assert_array_equal(
            linked[0]["polygon"],
            np.asarray([[10, 20], [30, 20], [30, 40], [10, 40]], dtype=float),
        )


if __name__ == "__main__":
    unittest.main()
