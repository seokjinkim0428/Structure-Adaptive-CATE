"""Numerical identities and shared-data uncertainty for noise-corrected risk."""
import unittest
import numpy as np
from ours.realcause_reporting import noise_adjusted_statistics


class NoiseCorrectionTests(unittest.TestCase):
    def test_constant_predictor_matches_finite_sample_mean_identity(self):
        rng = np.random.default_rng(841)
        effects = rng.normal(loc=np.arange(7), scale=2, size=(13, 7))
        predictions = np.vstack([np.arange(7), np.ones(7)])
        mse = ((predictions[None, :, :] - effects[:, None, :]) ** 2).mean(axis=2)
        report = noise_adjusted_statistics(mse, effects, np.ones(13), ["A", "B"])
        # Exact identity: subtracting unbiased sample variance from averaged
        # squared errors equals error against the sample mean minus s^2/R.
        expected = ((predictions-effects.mean(axis=0))**2).mean(axis=1) - effects.var(axis=0, ddof=1).mean()/13
        np.testing.assert_allclose(report["summary"].excess_mean, expected, atol=1e-13)

    def test_joint_jackknife_matches_full_recalculation_with_variable_scales(self):
        rng = np.random.default_rng(629)
        effects = rng.normal(size=(11, 17))
        scales = np.linspace(.6, 2.1, 11)
        mse = rng.uniform(.4, 2, size=(11, 3))
        report = noise_adjusted_statistics(mse, effects, scales, ["A", "B", "C"])
        direct = []
        for omitted in range(11):
            keep = np.arange(11) != omitted
            floor = effects[keep].var(axis=0, ddof=1).mean() * np.mean(scales[keep]**-2)
            means = mse[keep].mean(axis=0)
            direct.append(np.r_[means, means-floor, floor])
        direct = np.asarray(direct)
        np.testing.assert_allclose(report["jackknife"], direct, atol=1e-13)
        expected_cov = 10/11 * ((direct-direct.mean(axis=0)).T @ (direct-direct.mean(axis=0)))
        np.testing.assert_allclose(report["jackknife_covariance"], expected_cov, atol=1e-13)
        # A common correction cancels exactly in paired method differences,
        # including its covariance contribution to the estimated uncertainty.
        contrast = np.array([0, 0, 0, -1, 1, 0, 0])
        actual = contrast @ report["jackknife_covariance"].to_numpy() @ contrast
        expected = np.var(mse[:, 1]-mse[:, 0], ddof=1)/11
        self.assertAlmostEqual(actual, expected, places=12)

    def test_negative_estimates_are_retained_and_small_runs_rejected(self):
        effects = np.array([[-1, -1], [0, 0], [1, 1]], dtype=float)
        report = noise_adjusted_statistics(np.zeros((3, 1)), effects, np.ones(3), ["A"])
        self.assertEqual(report["summary"].excess_mean.iloc[0], -1)
        with self.assertRaises(ValueError):
            noise_adjusted_statistics(np.zeros((2, 1)), effects[:2], np.ones(2), ["A"])


if __name__ == "__main__":
    unittest.main()
