"""Nyström solver checks, including R-loss weighting and small-arm fits."""
import unittest
import numpy as np
from ours.krr import KRR, WeightedKRR
from ours.kernels import gaussian_kernel
from ours.realcause import _resolved_krr_kwargs


class NystromTests(unittest.TestCase):
    def test_full_landmarks_match_dense_ridge_for_both_losses(self):
        rng = np.random.default_rng(2026)
        x, xt = rng.normal(size=(25, 3)), rng.normal(size=(9, 3))
        y, weights = rng.normal(size=25), rng.uniform(.1, 2, size=25)
        kernel = lambda a, b: gaussian_kernel(a, b, length_scale=1.2)
        for cls, extra in [(KRR, ()), (WeightedKRR, (weights,))]:
            dense = cls(kernel, .7)
            approx = cls(kernel, .7, nystrom_m=700, nystrom_random_state=700)
            dense.fit(x, y, *extra)
            approx.fit(x, y, *extra)
            self.assertTrue(approx._use_nystrom)
            self.assertEqual(len(approx.nystrom_landmark_idx), len(x))
            np.testing.assert_allclose(approx.predict(xt), dense.predict(xt), rtol=2e-7, atol=2e-8)

    def test_landmarks_are_reproducible_and_response_independent(self):
        rng = np.random.default_rng(8)
        x, y = rng.normal(size=(45, 2)), rng.normal(size=45)
        models = [KRR(gaussian_kernel, .5, nystrom_m=12, nystrom_random_state=700) for _ in range(2)]
        models[0].fit(x, y)
        models[1].fit(x, y[::-1])
        np.testing.assert_array_equal(models[0].nystrom_landmark_idx, models[1].nystrom_landmark_idx)
        self.assertEqual(len(models[0].nystrom_landmark_idx), 12)

    def test_realcause_cannot_silently_disable_nystrom(self):
        for value in [None, 0, -1, 2.5, True]:
            with self.assertRaises(ValueError):
                _resolved_krr_kwargs({'nystrom_m': value})
