"""Leakage-policy checks for external AF data."""
import unittest

from firemon.external_af import containing_region, safe_date, validate_chip


class ExternalAFTests(unittest.TestCase):
    def test_only_winter_dates_are_eligible(self):
        self.assertTrue(safe_date("2022-12-15"))
        self.assertTrue(safe_date("2025-03-31"))
        self.assertFalse(safe_date("2022-07-15"))
        self.assertFalse(safe_date("2018-12-15"))
        self.assertFalse(safe_date("2026-01-15"))

    def test_tile_must_fit_wholly_in_remote_region(self):
        self.assertEqual(containing_region(-30, -20, 120, 150), "australia")
        self.assertIsNone(containing_region(-30, 20, 120, 150))
        self.assertIsNone(containing_region(40, 50, 35, 50))

    def test_manifest_chip_is_checked_fail_closed(self):
        item = {"chip_id": "safe", "acq_datetime": "2022-12-15T04:29:23Z",
                "region": "australia", "lat_min": -30, "lat_max": -20,
                "lon_min": 120, "lon_max": 150}
        self.assertEqual(validate_chip(item), "australia")
        item["acq_datetime"] = "2022-06-15T04:29:23Z"
        with self.assertRaises(ValueError):
            validate_chip(item)


if __name__ == "__main__":
    unittest.main()
