"""Study selection and standalone result tables."""

from dataclasses import asdict
from pathlib import Path
import unittest

import pandas as pd

from ours.demo import select_synthetic_conditions
from ours.synthetic_experiments import CORE_GROUPS, build_conditions
from ours.synthetic_reporting import comparison_table


class DemoTests(unittest.TestCase):
    def test_default_selection_preserves_full_design(self):
        expected = build_conditions("full", CORE_GROUPS)
        selected = select_synthetic_conditions("full")
        self.assertEqual([asdict(c) for c in expected], [asdict(c) for c in selected])

    def test_standalone_studies_have_complete_comparisons(self):
        summary = pd.read_csv(Path(__file__).resolve().parents[1] / "paper_results/synthetic/summary.csv")
        for study, setting in (("crossfit", "all"), ("representation", "multivariate")):
            conditions = select_synthetic_conditions("full", setting, study)
            selected = summary[summary.condition_id.isin([c.condition_id for c in conditions])]
            for dimension in ("univariate", "multivariate"):
                table = comparison_table(selected, dimension)
                if not table.empty:
                    self.assertFalse(table.isna().any().any())
            if study == "representation":
                table = comparison_table(selected, "multivariate")
                self.assertEqual(len(table), 3)
                self.assertIn("Default library", table.configuration.tolist())

    def test_replicate_override_and_invalid_selection(self):
        selected = select_synthetic_conditions("full", "univariate", "crossfit", 3)
        self.assertTrue(all(c.n_replications == 3 and c.d == 1 for c in selected))
        self.assertEqual({c.group for c in selected}, {"crossfit", "noise"})
        for n in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                select_synthetic_conditions("quick", n_replications=n)
        with self.assertRaisesRegex(ValueError, "No univariate"):
            select_synthetic_conditions("quick", "univariate", "representation")


if __name__ == "__main__":
    unittest.main()
