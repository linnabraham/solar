import unittest

from aarp_ml.ar_geometry import (
    bbox_in_frame,
    compute_pad_dim,
    load_shape_lookup,
    ar_bbox_for_entry,
    ar_mask,
)


class TestBboxInFrame(unittest.TestCase):
    def test_dim_matches_dataset(self):
        # Verified directly from data/grouped_df.csv during investigation.
        self.assertEqual(compute_pad_dim(), 2131)

    def test_known_case_aarp_3364_171A(self):
        # img_height/img_width from data/shape_limited.csv for AARP 3364, 171 Angstrom.
        # Predicted box was matched against the real bright-pixel cluster in the FITS
        # data during investigation: rows ~192-296, cols ~200-336.
        dim = compute_pad_dim()
        y0, y1, x0, x1 = bbox_in_frame(549, 745, dim, final_size=512)
        self.assertGreater(y0, 180)
        self.assertLess(y0, 200)
        self.assertGreater(y1, 300)
        self.assertLess(y1, 340)
        self.assertGreater(x0, 155)
        self.assertLess(x0, 210)
        self.assertGreater(x1, 330)
        self.assertLess(x1, 360)

    def test_final_size_224_is_scaled_512_box(self):
        dim = compute_pad_dim()
        box_512 = bbox_in_frame(549, 745, dim, final_size=512)
        box_224 = bbox_in_frame(549, 745, dim, final_size=224)
        for a, b in zip(box_512, box_224):
            self.assertAlmostEqual(a * (224 / 512), b, places=4)

    def test_rejects_native_size_larger_than_dim(self):
        with self.assertRaises(ValueError):
            bbox_in_frame(3000, 100, dim=2131, final_size=512)


class TestArBboxForEntry(unittest.TestCase):
    def setUp(self):
        self.shape_lookup = load_shape_lookup()
        self.dim = compute_pad_dim()

    def _entry_for_aarp(self, aarp_id):
        import json

        with open("solar_dataset.json") as f:
            data = json.load(f)
        for subset in ("training", "validation", "test"):
            for e in data[subset]:
                if e["aarp_id"] == aarp_id:
                    return e
        raise AssertionError(f"AARP {aarp_id} not found in solar_dataset.json")

    def test_channels_agree_for_aarp_3364(self):
        entry = self._entry_for_aarp(3364)
        # Should not warn -- all 7 channels were confirmed consistent for this AARP.
        with self.assertNoLogs(level="WARNING") if hasattr(self, "assertNoLogs") else _NullContext():
            box = ar_bbox_for_entry(entry, self.shape_lookup, self.dim, final_size=512)
        self.assertEqual(len(box), 4)
        y0, y1, x0, x1 = box
        self.assertLess(y0, y1)
        self.assertLess(x0, x1)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestArMask(unittest.TestCase):
    def test_mask_shape_and_content(self):
        mask = ar_mask((10.0, 20.0, 5.0, 15.0), shape=(32, 32))
        self.assertEqual(mask.shape, (32, 32))
        self.assertEqual(mask.dtype, bool)
        self.assertTrue(mask[15, 10])
        self.assertFalse(mask[0, 0])
        self.assertEqual(mask.sum(), 10 * 10)


if __name__ == "__main__":
    unittest.main()
