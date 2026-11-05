import unittest
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .data import prices
from .decision import apply_blend, blend_weights, penalty, vpp
from .target67 import scenarios


class DecisionTest(unittest.TestCase):
    def test_target_scenarios_score_models_on_common_actuals(self):
        price = pd.DataFrame({"smp": np.full(72, 100.), "k": np.full(72, 3)})
        forecast = np.full((72, 2), 4.)
        first = np.array([True, False])
        actual = forecast + 1.
        history = {"ver2": forecast, "gru_v1": forecast}
        models = {"ver2": (forecast, first), "gru_v1": (actual, first)}
        result = scenarios(price, history, history, actual, models, draws=2)["ver2"]
        self.assertEqual(result["gross_mean_krw"], 72000.)
        self.assertEqual(result["candidates"]["ver2"]["mean_penalty_krw"], 4320.)
        self.assertEqual(result["candidates"]["gru_v1"]["mean_penalty_krw"], 0.)

    def test_k_uses_closing_hour(self):
        times = pd.date_range("2025-03-01 01:00+09:00", periods=24, freq="h")
        with TemporaryDirectory() as directory:
            path = Path(directory)/"smp.csv"
            pd.DataFrame({"ts": times, "smp": 1.}).to_csv(path, index=False)
            table = prices(path)
        self.assertEqual(list(table.index[table.k == 3].hour), [11, 12, 13, 14, 15])

    def test_group_aggregation_precedes_abs(self):
        err = np.array([[4., -3., 2., -2.]])
        self.assertAlmostEqual(penalty(err, np.array([300.]), np.array([1, 1, 0, 0], bool))[0], 30.)

    def test_lp_recovers_complementary_blend(self):
        actual = np.ones((6, 4))*10
        predictions = np.stack([actual+4, actual-4], axis=-1)
        q = np.ones(6)
        regime = np.arange(6) % 2
        first = np.array([1, 1, 0, 0], bool)
        weights = blend_weights(predictions, actual, q, regime, first, True)
        np.testing.assert_allclose(apply_blend(predictions, regime, weights), actual)

    def test_milp_matches_exhaustive_robust_objective(self):
        rng = np.random.default_rng(42)
        capacity = np.full(6, 9000.)
        reference = np.array([1, 1, 1, 0, 0, 0], bool)
        errors = rng.normal(size=(12, 6))
        q = rng.uniform(30, 200, 12)
        month = np.repeat([1, 2, 3], 4)
        _, report = vpp(errors, q, capacity, reference, month, seconds=3.)
        objectives = []
        for bits in product([False, True], repeat=6):
            if sum(bits) == 3:
                value = penalty(errors, q, np.array(bits))
                objectives.append(value.mean() + .2*value.reshape(3, 4).mean(axis=1).max())
        self.assertAlmostEqual(report["objective"], min(objectives), places=6)


if __name__ == "__main__":
    unittest.main()
