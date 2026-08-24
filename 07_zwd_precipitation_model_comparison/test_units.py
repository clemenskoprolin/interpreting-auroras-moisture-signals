"""
test_units.py — Unit tests for 07_zwd_precipitation_model_comparison (no Aurora import).

Safe to run locally with:
    source .venv/bin/activate
    python 07_zwd_precipitation_model_comparison/test_units.py

Tests:
  - Case ranking / z-scoring (select_precipitation_cases)
  - Spatial mask construction (interventions)
  - Intervention identities: zero dose -> zero change
  - Removal mask properties
  - Metric formula correctness
  - Conditional difference formula
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path wiring
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SEARCHLIGHT_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")

for _p in (_HERE, _SEARCHLIGHT_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Test: z-score and case ranking
# ---------------------------------------------------------------------------
class TestCaseRanking(unittest.TestCase):

    def _zscore(self, arr):
        from select_precipitation_cases import _zscore
        return _zscore(arr)

    def test_zscore_zero_mean_unit_std(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = self._zscore(x)
        self.assertAlmostEqual(float(z.mean()), 0.0, places=10)
        self.assertAlmostEqual(float(z.std()), 1.0, places=10)

    def test_zscore_constant_input(self):
        x = np.array([5.0, 5.0, 5.0])
        z = self._zscore(x)
        np.testing.assert_array_equal(z, np.zeros(3))

    def test_zscore_single_element(self):
        x = np.array([3.14])
        z = self._zscore(x)
        self.assertEqual(float(z[0]), 0.0)

    def test_greedy_diverse_respects_separation(self):
        from select_precipitation_cases import _greedy_diverse
        import pandas as pd

        times = pd.DatetimeIndex([
            "2020-01-01", "2020-01-10", "2020-01-20",
            "2020-03-01", "2020-04-01",
        ])
        scores = np.array([10.0, 8.0, 6.0, 9.0, 7.0])

        selected = _greedy_diverse(times, scores, n=2, min_sep_days=30)
        # Should pick index 0 (score=10) and then the next one >= 30 days away
        self.assertEqual(len(selected), 2)
        # Index 0 (2020-01-01) and index 3 (2020-03-01) are 60 days apart
        selected_dates = [times[i] for i in selected]
        for i in range(len(selected_dates)):
            for j in range(i + 1, len(selected_dates)):
                diff = abs((selected_dates[i] - selected_dates[j]).days)
                self.assertGreaterEqual(diff, 30, f"Separation {diff} < 30 days")

    def test_greedy_diverse_n_limit(self):
        from select_precipitation_cases import _greedy_diverse
        import pandas as pd

        times = pd.DatetimeIndex([f"2020-{m:02d}-01" for m in range(1, 13)])
        scores = np.arange(12.0, 0.0, -1.0)  # descending

        selected = _greedy_diverse(times, scores, n=3, min_sep_days=30)
        self.assertLessEqual(len(selected), 3)

    def test_combined_score_formula(self):
        """Verify score = z(box_mean) + 0.5 * z(p90) is correctly ordered."""
        box_means = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p90s = np.array([10.0, 5.0, 8.0, 2.0, 9.0])
        from select_precipitation_cases import _zscore
        z_box = _zscore(box_means)
        z_p90 = _zscore(p90s)
        scores = z_box + 0.5 * z_p90
        # Highest score should have high box_mean AND high p90
        best = int(np.argmax(scores))
        self.assertIn(best, [4])  # index 4: box_mean=5, p90=9 (both high)


# ---------------------------------------------------------------------------
# Test: spatial mask construction
# ---------------------------------------------------------------------------
class TestMasks(unittest.TestCase):

    def setUp(self):
        # Small test grid
        self.lat_vals = np.linspace(90.0, -90.0, 37)  # descending ERA5 style
        self.lon_vals = np.linspace(0.0, 357.5, 144)

    def test_mask_inside_disk_is_zero(self):
        from interventions import make_removal_mask
        mask = make_removal_mask(
            self.lat_vals, self.lon_vals,
            center_lat=45.5, center_lon=8.5,
            disk_km=500.0, taper_km=2000.0,
        )
        # Center point should have mask ≈ 0
        lat_idx = int(np.argmin(np.abs(self.lat_vals - 45.5)))
        lon_idx = int(np.argmin(np.abs(self.lon_vals - 8.5)))
        self.assertAlmostEqual(float(mask[lat_idx, lon_idx]), 0.0, places=5)

    def test_mask_far_from_disk_is_one(self):
        from interventions import make_removal_mask
        mask = make_removal_mask(
            self.lat_vals, self.lon_vals,
            center_lat=45.5, center_lon=8.5,
            disk_km=500.0, taper_km=2000.0,
        )
        # Antipodal point (lat~-45, lon~188) should have mask = 1.0
        lat_idx = int(np.argmin(np.abs(self.lat_vals - (-45.5))))
        lon_idx = int(np.argmin(np.abs(self.lon_vals - 188.5)))
        self.assertAlmostEqual(float(mask[lat_idx, lon_idx]), 1.0, places=5)

    def test_mask_values_in_range(self):
        from interventions import make_removal_mask
        mask = make_removal_mask(
            self.lat_vals, self.lon_vals,
            center_lat=45.0, center_lon=7.0,
            disk_km=1000.0, taper_km=2500.0,
        )
        self.assertTrue(np.all(mask >= 0.0))
        self.assertTrue(np.all(mask <= 1.0))

    def test_mask_shape(self):
        from interventions import make_removal_mask
        mask = make_removal_mask(
            self.lat_vals, self.lon_vals,
            center_lat=0.0, center_lon=0.0,
        )
        self.assertEqual(mask.shape, (len(self.lat_vals), len(self.lon_vals)))

    def test_mask_disk_larger_than_taper_raises(self):
        from interventions import make_removal_mask
        with self.assertRaises(ValueError):
            make_removal_mask(
                self.lat_vals, self.lon_vals,
                center_lat=0.0, center_lon=0.0,
                disk_km=3000.0, taper_km=1000.0,
            )


# ---------------------------------------------------------------------------
# Test: intervention identities
# ---------------------------------------------------------------------------
class TestInterventions(unittest.TestCase):

    def setUp(self):
        self.lat_vals = np.linspace(90.0, -90.0, 37)
        self.lon_vals = np.linspace(0.0, 357.5, 144)
        H, W = len(self.lat_vals), len(self.lon_vals)
        # Synthetic precipitation (uniform)
        self.precip = torch.ones(1, 2, H, W, dtype=torch.float32) * 2.0

    def test_zero_dose_equals_actual(self):
        """apply_precip_dose with dose_mm=0 should return unchanged precip."""
        from interventions import make_removal_mask, apply_precip_dose
        mask = make_removal_mask(self.lat_vals, self.lon_vals, 45.0, 8.0)
        result = apply_precip_dose(self.precip, mask, dose_mm=0.0, timestep=1)
        torch.testing.assert_close(result, self.precip)

    def test_removal_then_plus_dose_equals_removal(self):
        """Removing and adding 0mm dose gives same result as just removing."""
        from interventions import make_removal_mask, apply_precip_removal, apply_precip_dose
        mask = make_removal_mask(self.lat_vals, self.lon_vals, 45.0, 8.0)
        removed = apply_precip_removal(self.precip, mask, timestep=1)
        zero_dosed = apply_precip_dose(removed, mask, dose_mm=0.0, timestep=1)
        torch.testing.assert_close(removed, zero_dosed)

    def test_actual_in_build_all_unchanged(self):
        """'actual' intervention key should return unmodified precip."""
        from interventions import build_all_interventions
        interventions = build_all_interventions(
            self.precip, self.lat_vals, self.lon_vals,
            center_lat=45.0, center_lon=8.0,
            doses_mm=(1.0, 5.0),
        )
        self.assertIn("actual", interventions)
        torch.testing.assert_close(interventions["actual"], self.precip)

    def test_remove_both_zeros_in_disk(self):
        """remove_both should zero out precipitation inside the disk."""
        from interventions import make_removal_mask, build_all_interventions
        interventions = build_all_interventions(
            self.precip, self.lat_vals, self.lon_vals,
            center_lat=45.0, center_lon=8.0,
            disk_km=1000.0, taper_km=2500.0,
        )
        removed = interventions["remove_both"]
        mask = make_removal_mask(
            self.lat_vals, self.lon_vals, 45.0, 8.0,
            disk_km=1000.0, taper_km=2500.0,
        )
        # Inside disk (mask==0), both timesteps should be 0
        disk_mask = mask == 0.0
        for t in range(2):
            disk_vals = removed[0, t][torch.from_numpy(disk_mask)]
            self.assertTrue(
                torch.all(disk_vals == 0.0),
                f"remove_both: timestep {t} has non-zero values inside disk"
            )

    def test_intervention_keys_present(self):
        """build_all_interventions should return the expected set of keys."""
        from interventions import build_all_interventions
        interventions = build_all_interventions(
            self.precip, self.lat_vals, self.lon_vals,
            center_lat=45.0, center_lon=8.0,
            doses_mm=(1.0, 5.0),
        )
        expected_keys = {
            "actual", "remove_t1", "remove_t0", "remove_both",
            "dose_plus_1mm", "dose_plus_5mm",
            "dose_minus_1mm", "dose_minus_5mm",
        }
        self.assertEqual(set(interventions.keys()), expected_keys)

    def test_dose_minus_clamps_to_zero(self):
        """Subtracting a large dose should not produce negative precipitation."""
        from interventions import make_removal_mask, apply_precip_dose
        mask = make_removal_mask(self.lat_vals, self.lon_vals, 45.0, 8.0)
        result = apply_precip_dose(self.precip, mask, dose_mm=1000.0, timestep=1, subtract=True)
        self.assertTrue(torch.all(result >= 0.0), "Precipitation went negative after large dose subtraction")


# ---------------------------------------------------------------------------
# Test: metric formulas
# ---------------------------------------------------------------------------
class TestMetrics(unittest.TestCase):

    def test_box_mean_uniform(self):
        """Box mean of a uniform field should equal the field value."""
        from metrics import box_mean_metric
        lat_vals = np.linspace(90.0, -90.0, 37)
        field = np.ones((37, 72))
        mean = box_mean_metric(field, 5, 10, 10, 20, lat_vals)
        self.assertAlmostEqual(mean, 1.0, places=5)

    def test_box_mean_zero(self):
        from metrics import box_mean_metric
        lat_vals = np.linspace(90.0, -90.0, 37)
        field = np.zeros((37, 72))
        mean = box_mean_metric(field, 5, 10, 10, 20, lat_vals)
        self.assertAlmostEqual(mean, 0.0, places=10)

    def test_spatial_rms_diff_identical(self):
        """RMS diff of identical arrays should be 0."""
        from metrics import spatial_rms_diff
        lat_vals = np.linspace(90.0, -90.0, 37)
        field = np.random.rand(37, 72)
        rms = spatial_rms_diff(field, field, lat_vals)
        self.assertAlmostEqual(rms, 0.0, places=10)

    def test_spatial_rms_diff_positive(self):
        from metrics import spatial_rms_diff
        lat_vals = np.linspace(90.0, -90.0, 37)
        a = np.ones((37, 72))
        b = np.zeros((37, 72))
        rms = spatial_rms_diff(a, b, lat_vals)
        self.assertGreater(rms, 0.0)

    def test_trajectory_diff_formula(self):
        """M = baseline_w - baseline_wo."""
        from metrics import trajectory_diff_metrics
        w = [1.0, 2.0, 3.0]
        wo = [0.5, 1.0, 2.0]
        result = trajectory_diff_metrics(w, wo)
        expected_M = [0.5, 1.0, 1.0]
        for i, (got, exp) in enumerate(zip(result["M"], expected_M)):
            self.assertAlmostEqual(got, exp, places=10, msg=f"M[{i}] mismatch")

    def test_trajectory_diff_mismatched_lengths(self):
        from metrics import trajectory_diff_metrics
        with self.assertRaises(ValueError):
            trajectory_diff_metrics([1.0, 2.0], [1.0])

    def test_spearman_corr_perfect(self):
        from metrics import spearman_corr
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r, _ = spearman_corr(x, x)
        self.assertAlmostEqual(r, 1.0, places=5)

    def test_spearman_corr_anti(self):
        from metrics import spearman_corr
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r, _ = spearman_corr(x, -x)
        self.assertAlmostEqual(r, -1.0, places=5)

    def test_top_k_overlap_identical(self):
        from metrics import top_k_overlap
        x = np.random.rand(100)
        overlap = top_k_overlap(x, x, k_frac=0.1)
        self.assertAlmostEqual(overlap, 1.0, places=10)

    def test_ndcg_at_k_perfect(self):
        from metrics import ndcg_at_k
        scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        relevance = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        ndcg = ndcg_at_k(scores, relevance, k=5)
        self.assertAlmostEqual(ndcg, 1.0, places=5)

    def test_ndcg_at_k_worst(self):
        from metrics import ndcg_at_k
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        relevance = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        ndcg = ndcg_at_k(scores, relevance, k=5)
        self.assertLess(ndcg, 1.0)

    def test_top_k_recall_perfect(self):
        from metrics import top_k_recall
        scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        relevance = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        recall = top_k_recall(scores, relevance, k=3)
        self.assertAlmostEqual(recall, 1.0, places=10)

    def test_center_of_mass_displacement_same(self):
        from metrics import center_of_mass_displacement
        lat_vals = np.linspace(90.0, -90.0, 37)
        lon_vals = np.linspace(0.0, 355.0, 72)
        x = np.random.rand(37, 72)
        d = center_of_mass_displacement(x, x, lat_vals, lon_vals)
        self.assertAlmostEqual(d, 0.0, places=2)


# ---------------------------------------------------------------------------
# Test: conditional difference formula (Stage A3)
# ---------------------------------------------------------------------------
class TestConditionalDiff(unittest.TestCase):

    def test_conditional_diff_basic(self):
        """D(I, h, y) = E_with_zwd - E_without_zwd"""
        from stages import run_stage_a3_conditional_diff
        import tempfile, csv

        rows_w = [
            {"case_id": "A", "target": "t", "init_time": "2020-01-01", "role": "strong",
             "target_var": "precip", "intervention": "actual", "lead_h": 6, "score": 2.0},
            {"case_id": "A", "target": "t", "init_time": "2020-01-01", "role": "strong",
             "target_var": "precip", "intervention": "remove_t1", "lead_h": 6, "score": 1.5},
        ]
        rows_wo = [
            {"case_id": "A", "target": "t", "init_time": "2020-01-01", "role": "strong",
             "target_var": "precip", "intervention": "actual", "lead_h": 6, "score": 1.8},
            {"case_id": "A", "target": "t", "init_time": "2020-01-01", "role": "strong",
             "target_var": "precip", "intervention": "remove_t1", "lead_h": 6, "score": 1.4},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_stage_a3_conditional_diff(rows_w, rows_wo, tmpdir)

        # Check conditional differences
        actual_row = next(r for r in result if r["intervention"] == "actual")
        self.assertAlmostEqual(actual_row["D_conditional_diff"], 2.0 - 1.8, places=10)

        remove_row = next(r for r in result if r["intervention"] == "remove_t1")
        self.assertAlmostEqual(remove_row["D_conditional_diff"], 1.5 - 1.4, places=10)

    def test_conditional_diff_missing_wo(self):
        """Missing without_zwd row should produce nan D."""
        from stages import run_stage_a3_conditional_diff
        import tempfile

        rows_w = [
            {"case_id": "A", "target": "t", "init_time": "2020-01-01", "role": "strong",
             "target_var": "precip", "intervention": "actual", "lead_h": 6, "score": 2.0},
        ]
        rows_wo = []   # empty

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_stage_a3_conditional_diff(rows_w, rows_wo, tmpdir)

        self.assertEqual(len(result), 1)
        self.assertTrue(math.isnan(result[0]["D_conditional_diff"]))


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running 07_zwd_precipitation_model_comparison unit tests ...")
    unittest.main(verbosity=2)
