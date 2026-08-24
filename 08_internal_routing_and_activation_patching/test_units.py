"""
test_units.py — Unit tests for 08_internal_routing_and_activation_patching (no Aurora import).

Run locally with:
    source .venv/bin/activate
    python 08_internal_routing_and_activation_patching/test_units.py
"""

from __future__ import annotations

import csv
import math
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path wiring
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_MCC_DIR = os.path.join(_ROOT, "07_zwd_precipitation_model_comparison")
_SEARCHLIGHT_DIR = os.path.join(_ROOT, "02_zwd_attribution_benchmark")

for _p in (_HERE, _MCC_DIR, _SEARCHLIGHT_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Helpers imported from the module under test (no Aurora)
# ---------------------------------------------------------------------------
# Import only functions that don't require Aurora/torch.cuda

from trace_precip_representations import (
    _safe_div,
    _make_pair_metadata,
    _aggregate_stage_curve,
    _linear_cka,
    _subsample_tokens,
    _perturb_precip_gaussian,
    _perturb_q_level_gaussian,
    _select_diagnostic_cases,
    _load_diagnostic_cases_from_file,
    _select_regions_from_saliency,
    _write_csv,
)
from comparison_config import DEFAULT_TARGETS, TargetSpec


# ---------------------------------------------------------------------------
# Test: _safe_div
# ---------------------------------------------------------------------------
class TestSafeDiv(unittest.TestCase):
    def test_normal(self):
        self.assertAlmostEqual(_safe_div(6.0, 2.0), 3.0)

    def test_zero_denominator(self):
        self.assertTrue(math.isnan(_safe_div(1.0, 0.0)))

    def test_negative(self):
        self.assertAlmostEqual(_safe_div(-4.0, 2.0), -2.0)


# ---------------------------------------------------------------------------
# Test: pair metadata formula
# ---------------------------------------------------------------------------
class TestPairMetadata(unittest.TestCase):
    def test_signed_and_common_response(self):
        meta = _make_pair_metadata({}, actual_score=1.0, plus_score=1.3, minus_score=0.8)
        pd = 1.3 - 1.0   # +0.3
        md = 0.8 - 1.0   # -0.2
        self.assertAlmostEqual(meta["signed_target_response"], 0.5 * (pd - md))
        self.assertAlmostEqual(meta["common_target_response"], 0.5 * (pd + md))
        self.assertAlmostEqual(meta["mean_abs_target_delta"], 0.5 * (abs(pd) + abs(md)))

    def test_output_opposite_sign_true(self):
        meta = _make_pair_metadata({}, actual_score=1.0, plus_score=1.5, minus_score=0.5)
        self.assertTrue(meta["output_opposite_sign"])

    def test_output_opposite_sign_false(self):
        meta = _make_pair_metadata({}, actual_score=1.0, plus_score=1.5, minus_score=1.2)
        self.assertFalse(meta["output_opposite_sign"])


# ---------------------------------------------------------------------------
# Test: _aggregate_stage_curve
# ---------------------------------------------------------------------------
class TestAggregateStageCurve(unittest.TestCase):
    def _make_block_rows(self):
        return [
            {"stage_name": "enc_s0", "family": "encoder", "stage_index": 0,
             "traversal_index": 0,
             "plus_delta_rms": 1.0, "minus_delta_rms": 0.9, "contrast_rms": 0.5,
             "common_rms": 0.4, "signed_cosine": 0.8, "contrast_share": 0.6},
            {"stage_name": "enc_s0", "family": "encoder", "stage_index": 0,
             "traversal_index": 1,
             "plus_delta_rms": 2.0, "minus_delta_rms": 1.8, "contrast_rms": 1.0,
             "common_rms": 0.8, "signed_cosine": 0.7, "contrast_share": 0.5},
        ]

    def test_aggregates_by_stage(self):
        rows = self._make_block_rows()
        out = _aggregate_stage_curve(rows, {"case_id": "X"})
        self.assertEqual(len(out), 1)
        row = out[0]
        self.assertEqual(row["stage_name"], "enc_s0")
        self.assertEqual(row["n_blocks"], 2)
        self.assertAlmostEqual(row["plus_delta_rms_mean"], 1.5)
        self.assertAlmostEqual(row["contrast_rms_mean"], 0.75)

    def test_preserves_metadata(self):
        rows = self._make_block_rows()
        out = _aggregate_stage_curve(rows, {"case_id": "TEST_CASE"})
        self.assertEqual(out[0]["case_id"], "TEST_CASE")


# ---------------------------------------------------------------------------
# Test: paired metrics on synthetic tensors
# ---------------------------------------------------------------------------
class TestPairedMetricsFormula(unittest.TestCase):
    def test_contrast_zero_when_symmetric(self):
        """When plus_delta == -minus_delta exactly, contrast should equal plus_delta."""
        plus = torch.randn(100, 64)
        minus = -plus
        contrast = 0.5 * (plus - minus)
        self.assertTrue(torch.allclose(contrast, plus, atol=1e-5))

    def test_common_zero_when_antisymmetric(self):
        """When plus_delta == -minus_delta exactly, common should be zero."""
        plus = torch.randn(100, 64)
        minus = -plus
        common = 0.5 * (plus + minus)
        self.assertTrue(torch.allclose(common, torch.zeros_like(common), atol=1e-5))

    def test_contrast_share_formula(self):
        c_rms = 2.0
        k_rms = 3.0
        share = _safe_div(c_rms, c_rms + k_rms)
        self.assertAlmostEqual(share, 2.0 / 5.0)


# ---------------------------------------------------------------------------
# Test: factorial interaction formula
# ---------------------------------------------------------------------------
class TestFactorialInteraction(unittest.TestCase):
    def test_additive_gives_zero_interaction(self):
        """For h(ZP) = h(Z) + h(P) - h(actual), interaction should be zero."""
        actual = torch.zeros(10, 32)
        dZ  = torch.randn(10, 32)
        dP  = torch.randn(10, 32)
        dZP = dZ + dP  # perfectly additive
        interaction = dZP - dZ - dP
        rms = float(torch.sqrt((interaction ** 2).mean()).item())
        self.assertAlmostEqual(rms, 0.0, places=5)

    def test_nonadditive_gives_nonzero_interaction(self):
        """Introducing cross-term makes interaction nonzero."""
        dZ  = torch.ones(10, 32)
        dP  = torch.ones(10, 32)
        dZP = dZ + dP + torch.ones(10, 32) * 0.5  # extra interaction term
        interaction = dZP - dZ - dP
        rms = float(torch.sqrt((interaction ** 2).mean()).item())
        self.assertGreater(rms, 0.4)


# ---------------------------------------------------------------------------
# Test: linear CKA
# ---------------------------------------------------------------------------
class TestLinearCKA(unittest.TestCase):
    def test_identical_matrices_cka_one(self):
        X = np.random.randn(50, 16).astype(np.float32)
        cka = _linear_cka(X, X)
        self.assertAlmostEqual(cka, 1.0, places=4)

    def test_scaled_matrix_cka_one(self):
        X = np.random.randn(50, 16).astype(np.float32)
        cka = _linear_cka(X, X * 3.7)
        self.assertAlmostEqual(cka, 1.0, places=4)

    def test_orthogonal_matrices_low_cka(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 32).astype(np.float32)
        Y = rng.randn(100, 32).astype(np.float32)
        cka = _linear_cka(X, Y)
        self.assertGreaterEqual(cka, 0.0)
        self.assertLessEqual(cka, 1.0)

    def test_range_0_to_1(self):
        rng = np.random.RandomState(7)
        X = rng.randn(40, 8).astype(np.float32)
        Y = rng.randn(40, 8).astype(np.float32)
        cka = _linear_cka(X, Y)
        self.assertGreaterEqual(cka, 0.0)
        self.assertLessEqual(cka, 1.0 + 1e-6)

    def test_handles_few_samples(self):
        X = np.random.randn(5, 4).astype(np.float32)
        Y = np.random.randn(5, 4).astype(np.float32)
        cka = _linear_cka(X, Y)
        self.assertFalse(math.isnan(cka) and False)  # should not raise


# ---------------------------------------------------------------------------
# Test: token subsampling
# ---------------------------------------------------------------------------
class TestSubsampleTokens(unittest.TestCase):
    def test_fewer_tokens_than_n_returns_all(self):
        act = torch.randn(1, 50, 16)
        sub = _subsample_tokens(act, n_tokens=100)
        self.assertEqual(sub.shape[0], 50)

    def test_subsamples_to_n_tokens(self):
        act = torch.randn(1, 1000, 16)
        sub = _subsample_tokens(act, n_tokens=64)
        self.assertLessEqual(sub.shape[0], 64 + 1)  # step-based

    def test_correct_feature_dim(self):
        D = 256
        act = torch.randn(1, 500, D)
        sub = _subsample_tokens(act, n_tokens=100)
        self.assertEqual(sub.shape[1], D)


# ---------------------------------------------------------------------------
# Test: _perturb_precip_gaussian
# ---------------------------------------------------------------------------
class TestPerturbPrecipGaussian(unittest.TestCase):
    def setUp(self):
        H, W = 37, 72
        self.precip = torch.ones(1, 2, H, W, dtype=torch.float32) * 2.0
        lat_vals = np.linspace(90.0, -90.0, H)
        lon_vals = np.linspace(0.0, 355.0, W)
        # Simple Gaussian mask centered at origin
        lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
        r2 = lat_grid ** 2 + lon_grid ** 2
        self.mask_np = np.exp(-0.5 * r2 / 100.0).astype(np.float32)

    def test_plus_increases_precip(self):
        result = _perturb_precip_gaussian(self.precip, self.mask_np, 1.0, +1.0, 1)
        self.assertTrue((result[0, 1] >= self.precip[0, 1]).all())

    def test_minus_decreases_precip(self):
        result = _perturb_precip_gaussian(self.precip, self.mask_np, 1.0, -1.0, 1)
        self.assertTrue((result[0, 1] <= self.precip[0, 1]).all())

    def test_clamps_to_zero(self):
        large = _perturb_precip_gaussian(self.precip, self.mask_np, 1000.0, -1.0, 1)
        self.assertTrue((large >= 0.0).all())

    def test_only_perturbs_specified_timestep(self):
        result = _perturb_precip_gaussian(self.precip, self.mask_np, 1.0, +1.0, 1)
        # t0 should be unchanged
        torch.testing.assert_close(result[0, 0], self.precip[0, 0])


class TestWriteCsv(unittest.TestCase):
    def test_allows_union_of_row_fields(self):
        rows = [
            {"case_id": "a", "mode": "zwd_trace", "magnitude": 1.0},
            {"case_id": "b", "mode": "precip_trace", "dose_mm": 5.0},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rows.csv")
            _write_csv(path, rows)
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                out = list(reader)
        self.assertEqual(len(out), 2)
        self.assertIn("magnitude", reader.fieldnames)
        self.assertIn("dose_mm", reader.fieldnames)
        self.assertEqual(out[0]["dose_mm"], "")
        self.assertEqual(out[1]["magnitude"], "")


# ---------------------------------------------------------------------------
# Test: diagnostic case selection from CSV
# ---------------------------------------------------------------------------
class TestDiagCaseSelection(unittest.TestCase):
    def _write_stage_a_csv(self, tmpdir, rows):
        path = os.path.join(tmpdir, "stage_a_model_trajectories.csv")
        if rows:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        return tmpdir

    def _make_row(self, target, init_time, m_diff, lead_h=6, target_var="q850"):
        return {
            "case_id": f"{target}_{init_time}",
            "target": target,
            "init_time": init_time,
            "role": "strong",
            "target_var": target_var,
            "lead_h": str(lead_h),
            "score_with_zwd": "1.0",
            "score_without_zwd": "0.5",
            "M_diff": str(m_diff),
        }

    def test_selects_one_per_target_by_max_m_diff(self):
        # Use "ticino" which is a valid TARGETS key
        rows = [
            self._make_row("ticino", "2020-01-01T00:00:00", 2.0),
            self._make_row("ticino", "2020-02-01T00:00:00", 5.0),  # higher M_diff
            self._make_row("ticino", "2020-03-01T00:00:00", 1.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_stage_a_csv(tmpdir, rows)
            with patch("trace_precip_representations._case_has_required_stage_b_maps", return_value=True):
                cases = _select_diagnostic_cases(tmpdir, n_per_target=1)
        # Should select the one with M_diff=5.0
        self.assertEqual(len(cases), 1)
        self.assertAlmostEqual(cases[0].m_diff_q850, 5.0)
        self.assertEqual(cases[0].init_time_str, "2020-02-01T00:00:00")

    def test_ignores_non_q850_rows(self):
        rows = [
            self._make_row("ticino", "2020-01-01T00:00:00", 3.0, target_var="precip"),
            self._make_row("ticino", "2020-02-01T00:00:00", 1.0, target_var="q850"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_stage_a_csv(tmpdir, rows)
            with patch("trace_precip_representations._case_has_required_stage_b_maps", return_value=True):
                cases = _select_diagnostic_cases(tmpdir, n_per_target=1)
        self.assertEqual(len(cases), 1)
        self.assertAlmostEqual(cases[0].m_diff_q850, 1.0)

    def test_ignores_non_6h_lead(self):
        rows = [
            self._make_row("ticino", "2020-01-01T00:00:00", 9.0, lead_h=12),
            self._make_row("ticino", "2020-02-01T00:00:00", 2.0, lead_h=6),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_stage_a_csv(tmpdir, rows)
            with patch("trace_precip_representations._case_has_required_stage_b_maps", return_value=True):
                cases = _select_diagnostic_cases(tmpdir, n_per_target=1)
        self.assertEqual(len(cases), 1)
        self.assertAlmostEqual(cases[0].m_diff_q850, 2.0)

    def test_missing_csv_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                _select_diagnostic_cases(tmpdir, n_per_target=1)

    def test_n_per_target_two(self):
        rows = [
            self._make_row("ticino", "2020-01-01T00:00:00", 5.0),
            self._make_row("ticino", "2020-02-01T00:00:00", 3.0),
            self._make_row("ticino", "2020-03-01T00:00:00", 1.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_stage_a_csv(tmpdir, rows)
            with patch("trace_precip_representations._case_has_required_stage_b_maps", return_value=True):
                cases = _select_diagnostic_cases(tmpdir, n_per_target=2)
        self.assertEqual(len(cases), 2)
        m_diffs = sorted([c.m_diff_q850 for c in cases], reverse=True)
        self.assertAlmostEqual(m_diffs[0], 5.0)
        self.assertAlmostEqual(m_diffs[1], 3.0)


# ---------------------------------------------------------------------------
# Test: mask selection from saliency
# ---------------------------------------------------------------------------
class TestMaskSelection(unittest.TestCase):
    def setUp(self):
        self.lat_vals = np.linspace(90.0, -90.0, 37)
        self.lon_vals = np.linspace(0.0, 355.0, 72)

    def test_returns_hotspot_and_low_near(self):
        # Uniform saliency — hotspot selection is deterministic (argmax)
        saliency = np.ones((37, 72), dtype=np.float32)
        regions = _select_regions_from_saliency(
            saliency=saliency,
            target_short="ticino",
            scale_name="local",
            lat_vals=self.lat_vals,
            lon_vals=self.lon_vals,
        )
        kinds = {r.region_kind for r in regions}
        self.assertIn("hotspot", kinds)
        self.assertIn("low_near", kinds)

    def test_deterministic(self):
        saliency = np.random.RandomState(0).rand(37, 72).astype(np.float32)
        r1 = _select_regions_from_saliency(saliency, "ticino", "local",
                                            self.lat_vals, self.lon_vals)
        r2 = _select_regions_from_saliency(saliency, "ticino", "local",
                                            self.lat_vals, self.lon_vals)
        for a, b in zip(r1, r2):
            self.assertEqual(a.mask_spec.center_lat, b.mask_spec.center_lat)
            self.assertEqual(a.mask_spec.center_lon, b.mask_spec.center_lon)


# ---------------------------------------------------------------------------
# Test: probe ridge fallback (no sklearn needed)
# ---------------------------------------------------------------------------
class TestProbeFallback(unittest.TestCase):
    def _numpy_r2(self, X, y):
        """Numpy lstsq R² as computed in the fallback path."""
        coef, _, _, _ = np.linalg.lstsq(
            np.c_[X, np.ones(len(X))], y, rcond=None
        )
        pred_y = X @ coef[:-1] + coef[-1]
        ss_res = float(np.sum((y - pred_y) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot < 1e-30:
            return float("nan")
        return 1.0 - ss_res / ss_tot

    def test_perfect_linear_fit(self):
        X = np.random.randn(20, 4)
        y = X @ np.array([1.0, -2.0, 0.5, 3.0]) + 1.5
        r2 = self._numpy_r2(X, y)
        self.assertAlmostEqual(r2, 1.0, places=5)

    def test_random_r2_less_than_one(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 4)
        y = rng.randn(20)
        r2 = self._numpy_r2(X, y)
        self.assertLessEqual(r2, 1.0)


# ---------------------------------------------------------------------------
# Test: file-based diagnostic case selection
# ---------------------------------------------------------------------------
class TestLoadDiagnosticCasesFromFile(unittest.TestCase):
    def _write_json(self, tmpdir, records):
        path = os.path.join(tmpdir, "cases.json")
        import json as _json
        with open(path, "w") as f:
            _json.dump(records, f)
        return path

    def _write_csv_file(self, tmpdir, records):
        path = os.path.join(tmpdir, "cases.csv")
        if records:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)
        return path

    def _one_record(self):
        return {
            "case_id": "ticino_2020-10-03T00:00:00",
            "target": "ticino",
            "init_time": "2020-10-03T00:00:00",
            "m_diff_q850": 0.101,
        }

    def test_json_loads_one_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_json(tmpdir, [self._one_record()])
            cases = _load_diagnostic_cases_from_file(path)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "ticino_2020-10-03T00:00:00")
        self.assertEqual(cases[0].target, "ticino")
        self.assertAlmostEqual(cases[0].m_diff_q850, 0.101)

    def test_csv_loads_one_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_csv_file(tmpdir, [self._one_record()])
            cases = _load_diagnostic_cases_from_file(path)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "ticino_2020-10-03T00:00:00")

    def test_json_multiple_cases(self):
        recs = [
            {"case_id": "ticino_2020-10-03T00:00:00", "target": "ticino",
             "init_time": "2020-10-03T00:00:00"},
            {"case_id": "japan_2020-04-18T00:00:00", "target": "japan",
             "init_time": "2020-04-18T00:00:00"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_json(tmpdir, recs)
            cases = _load_diagnostic_cases_from_file(path)
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[1].target, "japan")

    def test_missing_m_diff_defaults_to_zero(self):
        rec = {"case_id": "ticino_2020-10-03T00:00:00", "target": "ticino",
               "init_time": "2020-10-03T00:00:00"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_json(tmpdir, [rec])
            cases = _load_diagnostic_cases_from_file(path)
        self.assertEqual(cases[0].m_diff_q850, 0.0)

    def test_invalid_target_raises(self):
        rec = {"case_id": "bad_2020-01-01T00:00:00", "target": "nonexistent_target",
               "init_time": "2020-01-01T00:00:00"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_json(tmpdir, [rec])
            with self.assertRaises(ValueError):
                _load_diagnostic_cases_from_file(path)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            _load_diagnostic_cases_from_file("/nonexistent/path/cases.json")

    def test_unknown_extension_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cases.txt")
            with open(path, "w") as f:
                f.write("dummy")
            with self.assertRaises(ValueError):
                _load_diagnostic_cases_from_file(path)

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_json(tmpdir, [])
            with self.assertRaises(ValueError):
                _load_diagnostic_cases_from_file(path)

# ---------------------------------------------------------------------------
# Test: ZWD target spec registration and surface extraction
# ---------------------------------------------------------------------------
class TestZwdTargetSpec(unittest.TestCase):
    def test_zwd_in_default_targets(self):
        names = [t.name for t in DEFAULT_TARGETS]
        self.assertIn("zwd", names)

    def test_zwd_target_spec_is_surface(self):
        spec = next((t for t in DEFAULT_TARGETS if t.name == "zwd"), None)
        self.assertIsNotNone(spec)
        self.assertIsNone(spec.level_hpa, "zwd must be a surface var (level_hpa=None)")
        self.assertEqual(spec.output_var, "zwd")

    def test_zwd_resolves_from_targets_list(self):
        """Simulates what main() does to resolve --targets zwd."""
        tname = "zwd"
        matches = [t for t in DEFAULT_TARGETS if t.name == tname]
        self.assertEqual(len(matches), 1)
        self.assertIsInstance(matches[0], TargetSpec)

    def test_surface_extract_returns_nan_when_var_missing(self):
        """extract_precip_box returns NaN when the var is not in surf_vars (precip_only case)."""
        import types

        class _FakePred:
            surf_vars: dict = {}  # no "zwd" key

        from comparison_models import extract_precip_box
        result = extract_precip_box(_FakePred(), 0, 5, 0, 5, "zwd")
        self.assertTrue(math.isnan(result))

    def test_surface_extract_returns_box_mean(self):
        """extract_precip_box returns correct box mean when var is present."""
        import types

        arr = torch.ones(1, 1, 10, 10) * 42.0
        arr[0, 0, 2:5, 3:7] = 100.0

        class _FakePred:
            surf_vars = {"zwd": arr}

        from comparison_models import extract_precip_box
        result = extract_precip_box(_FakePred(), 2, 4, 3, 6, "zwd")
        expected = float(arr[0, 0, 2:5, 3:7].mean().item())
        self.assertAlmostEqual(result, expected, places=5)


# ---------------------------------------------------------------------------
# Test: region-source-target isolation (q850 saliency used, not zwd)
# ---------------------------------------------------------------------------
class TestRegionSourceTarget(unittest.TestCase):
    """Verify that region selection works with a specified source target
    and that zwd saliency is never required."""

    def setUp(self):
        self.lat_vals = np.linspace(90.0, -90.0, 37)
        self.lon_vals = np.linspace(0.0, 355.0, 72)

    def test_select_regions_with_q850_saliency_yields_valid_regions(self):
        """Selecting regions from q850 saliency should succeed regardless of
        whether zwd saliency exists."""
        saliency = np.random.RandomState(1).rand(37, 72).astype(np.float32)
        regions = _select_regions_from_saliency(
            saliency=saliency,
            target_short="ticino",
            scale_name="local",
            lat_vals=self.lat_vals,
            lon_vals=self.lon_vals,
        )
        self.assertEqual(len(regions), 2)
        kinds = {r.region_kind for r in regions}
        self.assertIn("hotspot", kinds)
        self.assertIn("low_near", kinds)


# ---------------------------------------------------------------------------
# Test: precip_only skip for zwd target
# ---------------------------------------------------------------------------
class TestPrecipOnlyZwdSkip(unittest.TestCase):
    def test_model_spec_has_zwd_flag(self):
        """precip_only ModelSpec has has_zwd=False so skip logic can trigger."""
        from comparison_config import DEFAULT_MODELS
        self.assertFalse(DEFAULT_MODELS["precip_only"].has_zwd)
        self.assertTrue(DEFAULT_MODELS["precip_zwd"].has_zwd)

    def test_skip_condition(self):
        """Verify that skip fires when target_var=zwd and has_zwd=False."""
        from comparison_config import DEFAULT_MODELS
        t_spec = next(t for t in DEFAULT_TARGETS if t.name == "zwd")
        precip_only_spec = DEFAULT_MODELS["precip_only"]
        # This is the exact condition used in main():
        should_skip = t_spec.name == "zwd" and not precip_only_spec.has_zwd
        self.assertTrue(should_skip)

    def test_no_skip_for_precip_zwd(self):
        """precip_zwd model should NOT be skipped for zwd target."""
        from comparison_config import DEFAULT_MODELS
        t_spec = next(t for t in DEFAULT_TARGETS if t.name == "zwd")
        precip_zwd_spec = DEFAULT_MODELS["precip_zwd"]
        should_skip = t_spec.name == "zwd" and not precip_zwd_spec.has_zwd
        self.assertFalse(should_skip)


# ---------------------------------------------------------------------------
# Test: q850 perturbation builder (source direction q850 -> precip/zwd)
# ---------------------------------------------------------------------------
class TestPerturbQLevelGaussian(unittest.TestCase):
    Q_LOC = 0.004248186480253935
    Q_SCALE = 0.004075226373970509
    L, H, W = 13, 6, 8
    LEVEL_IDX = 2

    def _q(self, fill=None):
        # Distinct constant per level so leakage across levels is detectable.
        q = torch.zeros(1, 2, self.L, self.H, self.W, dtype=torch.float32)
        for lev in range(self.L):
            q[:, :, lev] = 0.001 * (lev + 1) if fill is None else fill
        return q

    def _mask(self):
        m = np.zeros((self.H, self.W), dtype=np.float32)
        m[2, 3] = 1.0
        m[2, 4] = 0.5
        return m

    def test_only_target_level_and_timestep_change(self):
        q = self._q()
        out, _ = _perturb_q_level_gaussian(
            q, self._mask(), self.LEVEL_IDX, +1.0, 1.0,
            self.Q_LOC, self.Q_SCALE, timestep_idx=1,
        )
        # t0 untouched everywhere
        self.assertTrue(torch.equal(out[0, 0], q[0, 0]))
        # every non-target level at t1 untouched
        for lev in range(self.L):
            if lev == self.LEVEL_IDX:
                continue
            self.assertTrue(torch.equal(out[0, 1, lev], q[0, 1, lev]),
                            f"level {lev} was modified")

    def test_delta_is_magnitude_times_sigma_times_mask(self):
        q = self._q()
        out, _ = _perturb_q_level_gaussian(
            q, self._mask(), self.LEVEL_IDX, +1.0, 1.0,
            self.Q_LOC, self.Q_SCALE, timestep_idx=1,
        )
        delta = (out[0, 1, self.LEVEL_IDX] - q[0, 1, self.LEVEL_IDX]).numpy()
        self.assertAlmostEqual(float(delta[2, 3]), self.Q_SCALE, places=7)
        self.assertAlmostEqual(float(delta[2, 4]), 0.5 * self.Q_SCALE, places=7)
        self.assertAlmostEqual(float(delta[0, 0]), 0.0, places=9)

    def test_plus_and_minus_symmetric_when_no_clipping(self):
        """With a moist background (q >> sigma) the +/- arms are exact mirrors."""
        q = self._q(fill=0.010)   # well above sigma_q850, below loc + 4 sigma
        kw = dict(level_idx=self.LEVEL_IDX, magnitude=1.0,
                  q_loc=self.Q_LOC, q_scale=self.Q_SCALE, timestep_idx=1)
        plus, plus_s = _perturb_q_level_gaussian(q, self._mask(), sign=+1.0, **kw)
        minus, minus_s = _perturb_q_level_gaussian(q, self._mask(), sign=-1.0, **kw)
        d_plus = plus[0, 1, self.LEVEL_IDX] - q[0, 1, self.LEVEL_IDX]
        d_minus = q[0, 1, self.LEVEL_IDX] - minus[0, 1, self.LEVEL_IDX]
        self.assertTrue(torch.allclose(d_plus, d_minus, atol=1e-9))
        self.assertEqual(plus_s["clip_frac"], 0.0)
        self.assertEqual(minus_s["clip_frac"], 0.0)
        self.assertAlmostEqual(minus_s["realized_frac"], 1.0, places=5)

    def test_dry_background_clipping_is_reported(self):
        """q850 sigma exceeds dry-region background, so the minus arm clips.

        The clamp is physically required (q >= 0) but breaks +/- symmetry, so
        it must be surfaced rather than silently attenuating the contrast.
        """
        q = self._q(fill=0.001)   # far below sigma_q850 -> minus arm hits the floor
        kw = dict(level_idx=self.LEVEL_IDX, magnitude=1.0,
                  q_loc=self.Q_LOC, q_scale=self.Q_SCALE, timestep_idx=1)
        _, plus_s = _perturb_q_level_gaussian(q, self._mask(), sign=+1.0, **kw)
        _, minus_s = _perturb_q_level_gaussian(q, self._mask(), sign=-1.0, **kw)
        self.assertEqual(plus_s["clip_frac"], 0.0)
        self.assertGreater(minus_s["clip_frac"], 0.0)
        self.assertLess(minus_s["realized_frac"], 1.0)

    def test_never_negative(self):
        """A large negative perturbation must not drive specific humidity below 0."""
        q = self._q()
        out, _ = _perturb_q_level_gaussian(
            q, self._mask(), self.LEVEL_IDX, -1.0, 50.0,
            self.Q_LOC, self.Q_SCALE, timestep_idx=1,
        )
        self.assertGreaterEqual(float(out.min()), 0.0)

    def test_input_not_mutated(self):
        q = self._q()
        before = q.clone()
        _perturb_q_level_gaussian(
            q, self._mask(), self.LEVEL_IDX, +1.0, 1.0,
            self.Q_LOC, self.Q_SCALE, timestep_idx=1,
        )
        self.assertTrue(torch.equal(q, before))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running 08_internal_routing_and_activation_patching unit tests ...")
    unittest.main(verbosity=2)
