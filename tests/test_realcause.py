"""Regression checks for RealCause target leakage and solver selection."""

import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import ours.realcause as rc
from ours.realcause_experiments import experiment_config, summarize_results
from ours.realcause_experiments import run_samples


class RealCauseContracts(unittest.TestCase):
    def test_requested_sample_count_does_not_reduce_full_fit(self):
        selected = experiment_config("full", n_samples=30, start_sample=0)
        historical = experiment_config("full")
        self.assertEqual(selected["samples"], list(range(30)))
        self.assertEqual(historical["samples"], list(range(50)))
        self.assertEqual({k: v for k, v in selected.items() if k != "samples"},
                         {k: v for k, v in historical.items() if k != "samples"})
        self.assertEqual(experiment_config("full", n_samples=100, start_sample=0)["samples"],
                         list(range(100)))
        self.assertEqual(experiment_config("quick")["samples"], [0, 1])

    def test_sample_count_bounds_are_explicit(self):
        for count in (0, -1, 1.5, True, 101):
            with self.subTest(count=count), self.assertRaises(ValueError):
                experiment_config("full", n_samples=count, start_sample=0)
        with self.assertRaises(ValueError):
            experiment_config("full", n_samples=100, start_sample=1)
        for start in (-1, 100, 0.5, False):
            with self.subTest(start=start), self.assertRaises(ValueError):
                experiment_config("full", n_samples=1, start_sample=start)

    def test_effect_labels_never_enter_covariates_and_observed_y_is_used(self):
        df = pd.DataFrame({
            "age": [20, 40], "education": [10, 14], "t": [0, 1],
            "y": [2, 8], "y0": [1, 3], "y1": [4, 7],
            "ite": [3, 4], "tau": [3, 4], "cate": [2, 3],
            "yf": [2, 8], "ycf": [4, 3], "id": [100, 101],
            "Unnamed: 0": [0, 1],
        })
        X, A, Y, target = rc.map_df_to_XAYtau(df)
        np.testing.assert_array_equal(X, df[["age", "education"]].to_numpy())
        np.testing.assert_array_equal(A, [0, 1])
        np.testing.assert_array_equal(Y, [2, 8])
        np.testing.assert_array_equal(target, [3, 4])
        changed = df.copy()
        changed[["ite", "tau", "cate", "y0", "y1"]] += 1000
        np.testing.assert_array_equal(rc.map_df_to_XAYtau(changed)[0], X)

    def test_nonbinary_treatment_is_rejected(self):
        df = pd.DataFrame({"x": [0, 1], "t": [0, 0.4], "y0": [1, 2], "y1": [3, 4]})
        with self.assertRaises(ValueError):
            rc.map_df_to_XAYtau(df)

    def test_tau_scale_uses_range_without_subtracting_outcome_minimum(self):
        scaler = rc.MinMax01Scaler1D().fit([10, 30])
        np.testing.assert_allclose(scaler.transform([10, 30, 50]), [0, 1, 2])
        np.testing.assert_allclose(scaler.transform_tau([-10, 10]), [-0.5, 0.5])

    def test_real_data_has_eight_covariates_and_train_only_scaling(self):
        if not rc.DEFAULT_DATA_DIR.exists():
            self.skipTest("Bundled RealCause data unavailable")
        prepared = rc.prepare_realcause_sample("lalonde_cps", 1, max_train_size=600, max_test_size=400)
        X, A, Y, target = rc.map_df_to_XAYtau(rc.load_realcause_df("lalonde_cps", 1))
        tr, te = prepared["train_indices"], prepared["test_indices"]
        self.assertEqual(X.shape, (16177, 8))
        self.assertFalse(set(tr) & set(te))
        expected_min, expected_range = Y[tr].min(), np.ptp(Y[tr])
        self.assertEqual(prepared["metadata"]["y_train_min_raw"], expected_min)
        self.assertEqual(prepared["metadata"]["y_train_range_raw"], expected_range)
        np.testing.assert_allclose(prepared["tau_test"], target[te] / expected_range)
        np.testing.assert_allclose(prepared["X_train"].min(axis=0), np.zeros(8))

    def test_scalability_subsets_are_nested_and_test_rows_fixed(self):
        if not rc.DEFAULT_DATA_DIR.exists():
            self.skipTest("Bundled RealCause data unavailable")
        small = rc.prepare_realcause_sample("lalonde_cps", 1, max_train_size=300, max_test_size=400)
        large = rc.prepare_realcause_sample("lalonde_cps", 1, max_train_size=600, max_test_size=400)
        np.testing.assert_array_equal(small["train_indices"], large["train_indices"][:300])
        np.testing.assert_array_equal(small["test_indices"], large["test_indices"])

    def test_backend_options_are_forwarded_without_global_mutation(self):
        before = copy.deepcopy(rc.COMMON_KRR_KWARGS)
        self.assertEqual(before["nystrom_m"], 700)
        predict = lambda x: np.zeros(len(x))
        prepared = dict(X_train=np.ones((12, 2)), A_train=np.tile([0, 1], 6),
                        Y_train=np.arange(12.), X_test=np.ones((3, 2)),
                        tau_test=np.zeros(3), metadata={})
        with patch.object(rc, "prepare_realcause_sample", return_value=prepared), \
             patch.object(rc, "fit_ours", return_value=(predict, [("kh", 1.)])) as ours, \
             patch.object(rc, "fit_plugin", return_value=(predict, (1., 1.))) as plugin, \
             patch.object(rc, "fit_dr", return_value=predict) as dr, \
             patch.object(rc, "fit_r_learner", return_value=predict) as r:
            rc.run_one_dataset_sample("lalonde_cps", 0, verbose=False)
            for fitter in (ours, plugin, dr, r):
                self.assertEqual(fitter.call_args.kwargs["krr_kwargs"]["nystrom_m"], 700)
            rc.run_one_dataset_sample("lalonde_cps", 0, verbose=False,
                                      krr_kwargs={"nystrom_m": 7, "nystrom_random_state": 42})
            for fitter in (ours, plugin, dr, r):
                self.assertEqual(fitter.call_args.kwargs["krr_kwargs"]["nystrom_m"], 7)
                self.assertEqual(fitter.call_args.kwargs["krr_kwargs"]["nystrom_random_state"], 42)
        self.assertEqual(rc.COMMON_KRR_KWARGS, before)

    def test_one_sample_se_is_undefined(self):
        data = {"mse_" + k: [1.] for k in ["ours", "plugin", "dr", "r"]}
        data.update({"seconds_" + k: [1.] for k in ["ours", "plugin", "dr", "r"]})
        summary = summarize_results(pd.DataFrame(data))
        self.assertTrue(summary.mse_se.isna().all())

    def test_resume_keeps_scores_and_rejects_incompatible_settings(self):
        config = experiment_config('quick', n_samples=2)
        def fitted(sample, **kwargs):
            row = dict(sample=sample, backend='nystrom', y_train_range_raw=2.)
            row.update({'mse_' + k: .1 + sample for k in ['ours', 'plugin', 'dr', 'r']})
            row.update({'seconds_' + k: 1. for k in ['ours', 'plugin', 'dr', 'r']})
            return row
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'samples.csv'
            with patch('ours.realcause_experiments.run_one_dataset_sample', side_effect=fitted) as mock:
                first = run_samples(config, checkpoint_path=path, verbose=False)
                self.assertEqual(mock.call_count, 2)
                mock.reset_mock()
                resumed = run_samples(config, checkpoint_path=path, verbose=False)
                mock.assert_not_called()
                pd.testing.assert_frame_equal(first, resumed, check_dtype=False)
                changed = dict(config, xi=.02)
                with self.assertRaisesRegex(ValueError, 'config differs'):
                    run_samples(changed, checkpoint_path=path, verbose=False)


if __name__ == "__main__":
    unittest.main()
