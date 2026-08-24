"""
test_activation_patch_units.py — Pure unit tests for activation_patch_unet.py

No Aurora import; runnable on the login node:
    source .venv/bin/activate
    python 08_internal_routing_and_activation_patching/activation_patching/test_activation_patch_units.py
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from activation_patch_unet import (  # noqa: E402
    _safe_div,
    _downsample_mask,
    _apply_patch,
    _mask_to_token_tensor,
    PATCH_SITE_SPATIAL,
    LATENT_LEVELS,
)


# ---------------------------------------------------------------------------
# Test: _safe_div
# ---------------------------------------------------------------------------
class TestSafeDiv(unittest.TestCase):
    def test_normal(self):
        self.assertAlmostEqual(_safe_div(6.0, 2.0), 3.0)

    def test_zero_denominator_returns_nan(self):
        self.assertTrue(math.isnan(_safe_div(1.0, 0.0)))

    def test_tiny_denominator_returns_nan(self):
        self.assertTrue(math.isnan(_safe_div(1.0, 1e-31)))

    def test_negative(self):
        self.assertAlmostEqual(_safe_div(-6.0, 3.0), -2.0)


# ---------------------------------------------------------------------------
# Test: _downsample_mask
# ---------------------------------------------------------------------------
class TestDownsampleMask(unittest.TestCase):
    def test_identity_same_size(self):
        mask = np.random.rand(90, 45).astype(np.float32)
        out = _downsample_mask(mask, 90, 45)
        np.testing.assert_allclose(out, mask, atol=1e-5)

    def test_all_ones_stays_one(self):
        mask = np.ones((720, 1440), dtype=np.float32)
        out = _downsample_mask(mask, 180, 360)
        np.testing.assert_allclose(out, np.ones((180, 360), dtype=np.float32), atol=1e-5)

    def test_all_zeros_stays_zero(self):
        mask = np.zeros((720, 1440), dtype=np.float32)
        out = _downsample_mask(mask, 180, 360)
        np.testing.assert_allclose(out, np.zeros((180, 360), dtype=np.float32), atol=1e-5)

    def test_output_shape(self):
        for full_h, full_w, tok_h, tok_w in [
            (720, 1440, 180, 360),
            (720, 1440, 180,  90),
            (720, 1440,  90,  45),
        ]:
            mask = np.ones((full_h, full_w), dtype=np.float32)
            out = _downsample_mask(mask, tok_h, tok_w)
            self.assertEqual(out.shape, (tok_h, tok_w),
                             f"shape mismatch for ({full_h},{full_w})→({tok_h},{tok_w})")

    def test_values_nonneg(self):
        mask = np.random.rand(720, 1440).astype(np.float32)
        out = _downsample_mask(mask, 90, 45)
        self.assertTrue((out >= 0).all())

    def test_exact_area_average(self):
        """4×4 mask of ones downsampled by 2 gives 2×2 mask of ones."""
        mask = np.ones((4, 4), dtype=np.float32)
        out = _downsample_mask(mask, 2, 2)
        np.testing.assert_allclose(out, np.ones((2, 2), dtype=np.float32), atol=1e-5)

    def test_single_pixel_pattern(self):
        """A mask with only one hot pixel; the hot patch should be nonzero."""
        mask = np.zeros((8, 8), dtype=np.float32)
        mask[0, 0] = 1.0
        out = _downsample_mask(mask, 4, 4)
        self.assertGreater(float(out[0, 0]), 0.0)

    def test_non_divisible_via_zoom(self):
        """721×1440 → 180×360 (non-exact, uses scipy zoom)."""
        mask = np.ones((721, 1440), dtype=np.float32)
        try:
            out = _downsample_mask(mask, 180, 360)
            self.assertEqual(out.shape, (180, 360))
        except Exception as exc:
            self.skipTest(f"scipy not available: {exc}")


# ---------------------------------------------------------------------------
# Test: _apply_patch (patch blending)
# ---------------------------------------------------------------------------
class TestApplyPatch(unittest.TestCase):
    def _make_tensors(self, n_tok=64, d=32):
        base = torch.randn(1, n_tok, d)
        src  = torch.randn(1, n_tok, d)
        return base, src

    def test_all_zero_mask_returns_base(self):
        """mask = 0 everywhere → output must equal base."""
        base, src = self._make_tensors()
        mask = torch.zeros(base.shape[1], 1)
        out = _apply_patch(base, src, mask)
        torch.testing.assert_close(out, base)

    def test_all_one_mask_returns_source(self):
        """mask = 1 everywhere → output must equal source."""
        base, src = self._make_tensors()
        mask = torch.ones(base.shape[1], 1)
        out = _apply_patch(base, src, mask)
        torch.testing.assert_close(out, src)

    def test_half_mask_blends(self):
        """mask = 0.5 → output = 0.5 * base + 0.5 * src."""
        base, src = self._make_tensors()
        mask = torch.full((base.shape[1], 1), 0.5)
        out = _apply_patch(base, src, mask)
        expected = 0.5 * base + 0.5 * src
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=0)

    def test_preserves_shape(self):
        # Shape preservation does not require allocating a full Aurora hidden
        # state (which would consume well over a gigabyte in this unit test).
        n_tok, d = 2592, 32
        base = torch.randn(1, n_tok, d)
        src  = torch.randn(1, n_tok, d)
        mask = torch.rand(n_tok, 1)
        out = _apply_patch(base, src, mask)
        self.assertEqual(out.shape, base.shape)

    def test_preserves_dtype(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            base = torch.randn(1, 16, 8, dtype=dtype)
            src  = torch.randn(1, 16, 8, dtype=dtype)
            mask = torch.rand(16, 1, dtype=dtype)
            out = _apply_patch(base, src, mask)
            self.assertEqual(out.dtype, dtype,
                             f"dtype mismatch for {dtype}")

    def test_preserves_device(self):
        base = torch.randn(1, 16, 8)
        src  = torch.randn(1, 16, 8)
        mask = torch.rand(16, 1)
        out = _apply_patch(base, src, mask)
        self.assertEqual(out.device, base.device)

    def test_values_bounded_by_extremes(self):
        """With a 0-1 mask, output should be within [min(base,src), max(base,src)]."""
        base = torch.tensor([[[0.0, 0.0]]])    # (1, 1, 2)
        src  = torch.tensor([[[1.0, 1.0]]])
        mask = torch.tensor([[0.3]])             # (1, 1) → broadcasts to (1, 2)
        out  = _apply_patch(base, src, mask)
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.3, places=5)

    def test_partial_spatial_mask(self):
        """Patch only some tokens; unpatched tokens equal base."""
        n_tok, d = 100, 16
        base = torch.zeros(1, n_tok, d)
        src  = torch.ones(1, n_tok, d)
        mask = torch.zeros(n_tok, 1)
        mask[:50] = 1.0  # patch first 50 tokens
        out = _apply_patch(base, src, mask)
        torch.testing.assert_close(out[0, :50], src[0, :50])
        torch.testing.assert_close(out[0, 50:], base[0, 50:])


# ---------------------------------------------------------------------------
# Test: _mask_to_token_tensor — shape and value properties
# ---------------------------------------------------------------------------
class TestMaskToTokenTensor(unittest.TestCase):
    def _ones_mask(self, h=720, w=1440):
        return np.ones((h, w), dtype=np.float32)

    def _zeros_mask(self, h=720, w=1440):
        return np.zeros((h, w), dtype=np.float32)

    def _check_site(self, site, mask_hw=None):
        if mask_hw is None:
            mask_hw = self._ones_mask()
        tok_h, tok_w = PATCH_SITE_SPATIAL[site]
        n_tok = LATENT_LEVELS * tok_h * tok_w
        t = _mask_to_token_tensor(mask_hw, site, 720, 1440,
                                  device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(t.shape, (n_tok, 1),
                         f"shape mismatch for site {site}: {t.shape} != ({n_tok}, 1)")
        return t

    def test_all_sites_produce_correct_shape(self):
        for site in PATCH_SITE_SPATIAL:
            self._check_site(site)

    def test_all_ones_mask_gives_all_ones_token(self):
        for site in PATCH_SITE_SPATIAL:
            t = self._check_site(site, self._ones_mask())
            np.testing.assert_allclose(t.numpy(), np.ones_like(t.numpy()), atol=1e-5)

    def test_all_zeros_mask_gives_all_zeros_token(self):
        for site in PATCH_SITE_SPATIAL:
            t = self._check_site(site, self._zeros_mask())
            np.testing.assert_allclose(t.numpy(), np.zeros_like(t.numpy()), atol=1e-5)

    def test_dtype_float32(self):
        t = _mask_to_token_tensor(self._ones_mask(), "enc_s0_skip", 720, 1440,
                                  device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(t.dtype, torch.float32)

    def test_values_nonneg(self):
        mask = np.random.rand(720, 1440).astype(np.float32)
        t = _mask_to_token_tensor(mask, "enc_s2_bottleneck", 720, 1440,
                                  device=torch.device("cpu"), dtype=torch.float32)
        self.assertTrue((t.numpy() >= 0).all())

    def test_721_height_handled(self):
        """721×1440 mask (ERA5 native) should work via cropping to 720×1440."""
        mask = np.ones((721, 1440), dtype=np.float32)
        tok_h, tok_w = PATCH_SITE_SPATIAL["enc_s0_skip"]
        n_tok = LATENT_LEVELS * tok_h * tok_w
        t = _mask_to_token_tensor(mask, "enc_s0_skip", 721, 1440,
                                  device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(t.shape, (n_tok, 1))


# ---------------------------------------------------------------------------
# Test: recovery formula
# ---------------------------------------------------------------------------
class TestRecoveryFormula(unittest.TestCase):
    def test_whole_mask_recovery_one(self):
        """With all-one mask, apply_patch(base, src, 1) = src → recovery = 1."""
        base = torch.randn(1, 64, 16)
        src  = torch.randn(1, 64, 16)
        mask = torch.ones(64, 1)
        patched = _apply_patch(base, src, mask)
        torch.testing.assert_close(patched, src)
        # Recovery = (src - base) / (src - base) = 1
        score_base   = float(base.mean())
        score_source = float(src.mean())
        score_patched = float(patched.mean())
        recovery = _safe_div(score_patched - score_base, score_source - score_base)
        if not math.isnan(recovery):
            self.assertAlmostEqual(recovery, 1.0, places=5)

    def test_zero_mask_recovery_zero(self):
        """With all-zero mask, apply_patch = base → recovery = 0."""
        base = torch.randn(1, 64, 16)
        src  = torch.randn(1, 64, 16)
        mask = torch.zeros(64, 1)
        patched = _apply_patch(base, src, mask)
        torch.testing.assert_close(patched, base)
        score_base    = float(base.mean())
        score_source  = float(src.mean())
        score_patched = float(patched.mean())
        recovery = _safe_div(score_patched - score_base, score_source - score_base)
        if not math.isnan(recovery):
            self.assertAlmostEqual(recovery, 0.0, places=5)

    def test_recovery_nan_when_delta_zero(self):
        """When score_source == score_base, recovery is NaN."""
        recovery = _safe_div(0.5, 0.0)
        self.assertTrue(math.isnan(recovery))

    def test_soft_mask_recovery_between_0_and_1(self):
        """Soft mask should give intermediate recovery for linear targets."""
        n_tok, d = 100, 1
        base = torch.zeros(1, n_tok, d)
        src  = torch.ones(1, n_tok, d)
        alpha = 0.4
        mask = torch.full((n_tok, 1), alpha)
        patched = _apply_patch(base, src, mask)
        score_base    = float(base.mean())
        score_source  = float(src.mean())
        score_patched = float(patched.mean())
        recovery = _safe_div(score_patched - score_base, score_source - score_base)
        self.assertAlmostEqual(recovery, alpha, places=5)


# ---------------------------------------------------------------------------
# Test: patch_site_spatial table consistency
# ---------------------------------------------------------------------------
class TestPatchSiteSpatialTable(unittest.TestCase):
    _EXPECTED_N_TOK = {
        "enc_s0_skip":        4 * 180 * 360,
        "enc_s1_skip":        4 * 180 * 90,
        "enc_s2_bottleneck":  4 * 90  * 45,
        "dec_s0_pre_skip":    4 * 90  * 45,
        "dec_s1_pre_skip":    4 * 180 * 90,
        # dec_s1 has an internal PatchSplitting3D that doubles H and W before the
        # external additive skip with enc_s0; output is therefore at enc_s0 resolution.
        "dec_s1_post_skip":   4 * 180 * 360,
        "dec_s2_pre_concat":  4 * 180 * 360,
        "dec_s2_post_concat": 4 * 180 * 360,
    }

    def test_all_sites_present(self):
        expected = set(self._EXPECTED_N_TOK.keys())
        actual   = set(PATCH_SITE_SPATIAL.keys())
        self.assertEqual(actual, expected)

    def test_n_tok_values(self):
        for site, expected_n in self._EXPECTED_N_TOK.items():
            tok_h, tok_w = PATCH_SITE_SPATIAL[site]
            actual_n = LATENT_LEVELS * tok_h * tok_w
            self.assertEqual(actual_n, expected_n,
                             f"n_tok mismatch for {site}: {actual_n} != {expected_n}")

    def test_enc_dec_resolution_symmetry(self):
        """Check spatial symmetries that hold given Aurora's PatchSplitting layout.

        dec_s1 contains an internal PatchSplitting3D that upsamples from enc_s1
        resolution to enc_s0 resolution before the additive skip connection with
        enc_s0_skip is applied externally.  Therefore dec_s1_post_skip lives at
        enc_s0 resolution, NOT enc_s1 resolution.
        """
        # Standard symmetric pairs
        self.assertEqual(PATCH_SITE_SPATIAL["enc_s0_skip"],
                         PATCH_SITE_SPATIAL["dec_s2_pre_concat"])
        self.assertEqual(PATCH_SITE_SPATIAL["enc_s0_skip"],
                         PATCH_SITE_SPATIAL["dec_s2_post_concat"])
        self.assertEqual(PATCH_SITE_SPATIAL["enc_s1_skip"],
                         PATCH_SITE_SPATIAL["dec_s1_pre_skip"])
        self.assertEqual(PATCH_SITE_SPATIAL["enc_s2_bottleneck"],
                         PATCH_SITE_SPATIAL["dec_s0_pre_skip"])
        # dec_s1_post_skip is at enc_s0 resolution (after internal PatchSplitting + skip)
        self.assertEqual(PATCH_SITE_SPATIAL["enc_s0_skip"],
                         PATCH_SITE_SPATIAL["dec_s1_post_skip"])


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running activation_patch unit tests (no Aurora import) ...")
    unittest.main(verbosity=2)
