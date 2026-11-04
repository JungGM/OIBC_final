"""Safety-critical time and scoring checks, with synthetic inputs only."""
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import issue_deadline, penalty, robust_absolute_loss, design, forecast_day_mean, optimize_groups, TUNE_END, CAL_START, CAL_END, CHECK_START, CHECK_END, TEST_START


class BacktestContract(unittest.TestCase):
    def test_group_selection_ends_by_first_forecast_deadline(self):
        for end, start in [(TUNE_END, CAL_START), (CAL_END, CHECK_START), (CHECK_END, TEST_START)]:
            self.assertLessEqual(end, issue_deadline([start])[0])

    def test_settlement_matches_hand_calculation(self):
        # Revenue=2,500; weighted abs error=450+600=1,050; penalty=105.
        score = penalty([10, 20], [13, 16], [150, 50], [1, 3])
        # 150*3 + 50*3*4 = 1,050 (first interval is not peak).
        self.assertEqual(score["revenue_krw"], 2500)
        self.assertEqual(score["penalty_krw"], 105)
        self.assertEqual(score["settlement_krw"], 2395)

    def test_group_errors_cancel_before_absolute_value(self):
        individual = penalty([10, 10], [15, 5], [100, 100], [3, 3])
        grouped = penalty([20], [20], [100], [3])
        self.assertEqual(individual["penalty_krw"], 300)
        self.assertEqual(grouped["penalty_krw"], 0)

    def test_missing_truth_is_rejected(self):
        with self.assertRaises(ValueError):
            penalty([np.nan], [0], [100], [1])

    def test_midnight_is_previous_operating_day(self):
        ts = pd.DatetimeIndex(["2024-01-01 01:00+09:00", "2024-01-02 00:00+09:00"])
        deadlines = issue_deadline(ts)
        self.assertTrue((deadlines == pd.Timestamp("2023-12-31 17:00+09:00")).all())
        self.assertEqual(issue_deadline([pd.Timestamp("2024-01-02 01:00+09:00")])[0],
                         pd.Timestamp("2024-01-01 17:00+09:00"))

    def test_symmetric_sro_radius_does_not_shift_bid(self):
        samples = np.array([20., 40., 90.])
        bids = np.arange(0., 111.)
        ordinary = [robust_absolute_loss(b, samples, 0) for b in bids]
        robust = [robust_absolute_loss(b, samples, 15) for b in bids]
        self.assertEqual(bids[np.argmin(ordinary)], 40)
        self.assertEqual(np.argmin(ordinary), np.argmin(robust))
        np.testing.assert_allclose(np.array(robust)-ordinary, 15)

    def test_feature_projection_ignores_poisoned_targets(self):
        a = pd.DataFrame({"wb_wind_speed": [36.], "wa_temp": [20.], "energy": [1.]})
        b = a.copy()
        b["energy"] = 1e30
        features = ["wb_wind_speed", "wa_temp"]
        np.testing.assert_array_equal(design(a, features, enhanced=True), design(b, features, enhanced=True))
        self.assertEqual(design(a, features, enhanced=True)[0, 0], 10.)
        self.assertEqual(design(a, features)[0, 0], 36.)

    def test_forecast_rolling_never_crosses_issue_day(self):
        values = np.arange(48, dtype=float)
        values[:3] = np.nan
        values[10] = np.nan
        for w in [3, 7]:
            expected = np.concatenate([pd.Series(day).rolling(w, min_periods=1).mean().to_numpy()
                                       for day in values.reshape(-1, 24)])
            np.testing.assert_allclose(forecast_day_mean(values, w), expected, equal_nan=True)
            self.assertEqual(forecast_day_mean(values, w)[24], 24)

    def test_group_swaps_cancel_errors_and_preserve_capacity(self):
        ts = pd.date_range("2023-07-01 01:00", periods=168, freq="h", tz="Asia/Seoul")
        sites = pd.DataFrame({"site_id": ["a", "b", "c", "d"],
                              "current_group": ["A", "A", "B", "B"],
                              "capacity_kw": [11000]*4})
        d = pd.MultiIndex.from_product([sites.site_id, ts], names=["site_id", "ts"]).to_frame(index=False)
        d["energy"] = 10.
        pred = 10+d.site_id.map({"a": 1., "b": 1., "c": -1., "d": -1.}).to_numpy()
        prices = pd.DataFrame({"ts": ts, "smp": 100., "k": 1})
        with tempfile.TemporaryDirectory() as tmp:
            ref, groups, eligible, info = optimize_groups(d, pred, sites, prices, Path(tmp))
        self.assertEqual(info["proxy_penalty_after"], 0)
        self.assertGreater(info["proxy_penalty_before"], 0)
        self.assertEqual(set(groups.index), set(sites.site_id))
        self.assertGreaterEqual(sites.set_index("site_id").capacity_kw.groupby(groups).sum().min(), 20000)


if __name__ == "__main__":
    unittest.main()
