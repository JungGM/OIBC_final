import unittest

import numpy as np
import pandas as pd

from GM_Ver1.genetic_algorithm import fitness, repair, settlement_scenarios


class GeneticPortfolioTests(unittest.TestCase):
    def test_fitness_sums_signed_errors_and_rejects_small_group(self):
        capacity = np.full(4, 10_000.)
        population = [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]]
        values = fitness(population, np.array([[5., -5., 3., -3.]]), np.array([100.]), capacity)
        np.testing.assert_allclose(values, [0., 160., np.inf])
        repaired = repair(population[-1], capacity, 20_000., np.random.default_rng(1))
        self.assertEqual(capacity@repaired, 20_000.)

    def test_bootstrap_signed_residuals_and_first_issue_cutoff(self):
        times = pd.date_range("2025-05-30 01:00", periods=48, freq="h", tz="Asia/Seoul")
        errors = np.vstack([np.tile([5., -10.], (24, 1)), np.tile([50., -90.], (24, 1))])
        check = dict(times=times, errors=errors)
        bits = np.array([True, False])
        result = settlement_scenarios(check, bits, np.full((24, 2), 30.), bits, np.full(2, 100.),
                                      np.full(24, 100.), np.full(24, 100.), draws=5)
        self.assertEqual(result["calibration_operating_days"], 1)
        self.assertAlmostEqual(result["forecast_implied_gross_krw"], 144000.)
        base = result["scenarios"]["1.0"]
        self.assertAlmostEqual(base["mean_revenue_krw"], 156000.)
        self.assertAlmostEqual(base["mean_penalty_krw"], 3600.)
        self.assertAlmostEqual(base["mean_settlement_krw"], 152400.)


if __name__ == "__main__":
    unittest.main()
