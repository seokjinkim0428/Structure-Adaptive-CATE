"""Scientific regression checks for the consolidated synthetic orchestration."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from ours.synthetic_experiments import (
    build_conditions, make_kernels, run_suite, summarize_results,
)


class SyntheticRegressionTests(unittest.TestCase):
    def test_reported_protocol_is_preserved(self):
        conditions = build_conditions("full")
        reported = [c for c in conditions if c.group in
                    ("noise", "kappa", "nuisance", "representation")]
        self.assertTrue(all(c.n_replications == 30 and c.n_test == 3000 for c in reported))
        self.assertTrue(all(c.xi == 0.01 for c in reported))
        self.assertEqual({c.d for c in reported}, {1, 10, 20})
        self.assertTrue(all(c.n_replications == 100 for c in conditions if c.group == "original"))
        nuisance = [c for c in reported if c.group == "nuisance"]
        self.assertEqual({(c.nuisance_order, c.target_order) for c in nuisance}, {(1, 2), (2, 3)})

    def check_reference(self, condition_id):
        # Frozen before the module refactor; covers splits, CV and predictions.
        reference = json.loads((Path(__file__).parent / "fixtures/synthetic.json").read_text())
        expected = [r for r in reference if r["condition_id"] == condition_id]
        condition = next(c for c in build_conditions("quick") if c.condition_id == condition_id)
        with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as output:
            raw = run_suite("quick", conditions=[condition], output_dir=output, progress=False)
        self.assertTrue(raw.status.eq("ok").all())
        self.assertEqual(set(raw.seed), {100, 110})
        self.assertEqual(len(raw), len(expected))
        for row, ref in zip(raw.to_dict("records"), expected):
            self.assertEqual((row["method"], row["replication"]), (ref["method"], ref["replication"]))
            np.testing.assert_allclose(row["mise"], ref["mise"], rtol=1e-11, atol=1e-12)
            self.assertEqual(row["selected_kernels"], ref["selected_kernels"])
            self.assertEqual(row["selected_ridges"], ref["selected_ridges"])

    def test_all_five_methods_match_reference_run(self):
        self.check_reference("noise__d1_p1_sigma1")

    def test_original_multivariate_recipe_matches_reference_run(self):
        condition = next(c for c in build_conditions("quick")
                         if c.group == "original" and c.d == 10 and c.p == 4)
        _, details = make_kernels(condition)
        self.assertEqual(details["nuisance_length"], 2.6)
        self.check_reference(condition.condition_id)

    def test_resume_and_failure_accounting(self):
        condition = next(c for c in build_conditions("quick") if c.group == "noise" and c.d == 1)
        condition = replace(condition, methods=("Ours",))
        with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as output:
            first = run_suite("quick", conditions=[condition], output_dir=output, progress=False)
            resumed = run_suite("quick", conditions=[condition], output_dir=output, progress=False)
        self.assertEqual(len(first), len(resumed))
        np.testing.assert_allclose(first.mise, resumed.mise)
        first.loc[first.index[0], ["status", "mise"]] = ["failed", np.nan]
        summary = summarize_results(first).iloc[0]
        self.assertEqual(summary.n_eff, 1)
        self.assertEqual(summary.n_failed, 1)
        self.assertTrue(np.isnan(summary.mise_se))


if __name__ == "__main__":
    unittest.main()
