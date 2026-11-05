import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from GM_Ver3.settlement import issue_cutoff, score


class SettlementTest(unittest.TestCase):
    def setUp(self):
        self.sites = pd.DataFrame({"site_id": ["A", "B", "C"], "capacity_kw": [10_000, 10_000, 20_000]})
        self.composition = pd.DataFrame({"site_id": ["C", "B", "A"], "group": ["G2", "G1", "G1"]})
        times = pd.DatetimeIndex([f"2030-03-01 {hour}:00:00+09:00" for hour in (10, 11, 15, 16)])
        self.weights = pd.DataFrame({"smp": 100}, index=times.rename("ts"))
        keys = pd.MultiIndex.from_product([times, self.sites.site_id], names=["ts", "site_id"]).to_frame(index=False)
        self.actual = keys.assign(energy=[10, 10, 0] * 4)
        self.prediction = keys.assign(prediction=[15, 5, 2] * 4)

    def evaluate(self):
        return score(self.sites, self.composition, self.actual, self.prediction, self.weights)

    def test_settlement_aggregates_signed_errors_before_absolute_value(self):
        result = self.evaluate()
        self.assertEqual(result["hours_scored"], 4)
        self.assertEqual(result["revenue_krw"], 8000)
        self.assertEqual(result["penalty_by_group_krw"], {"G1": 0, "G2": 160})
        self.assertEqual(result["settlement_krw"], 7840)

    def test_missing_and_negative_actuals_drop_entire_hours(self):
        self.actual = self.actual.drop(index=0)
        self.actual.loc[3, "energy"] = -1
        result = self.evaluate()
        self.assertEqual(result["hours_excluded_invalid_actual"], 2)
        self.assertEqual(result["revenue_krw"], 4000)
        self.assertEqual(result["penalty_krw"], 80)

    def test_missing_forecast_fails(self):
        self.prediction = self.prediction.drop(index=0)
        with self.assertRaisesRegex(ValueError, "finite nonnegative forecast"):
            self.evaluate()

    def test_under_capacity_group_fails(self):
        self.sites.loc[0, "capacity_kw"] = 9_999
        with self.assertRaisesRegex(ValueError, "20,000 kW"):
            self.evaluate()

    def test_invalid_provided_k_fails(self):
        self.weights["k"] = [1, 1, 3, 1]
        with self.assertRaisesRegex(ValueError, "closing-hour weights"):
            self.evaluate()

    def test_midnight_closes_previous_operating_day(self):
        times = ["2030-03-02 00:00:00+09:00", "2030-03-02 01:00:00+09:00"]
        expected = ["2030-02-28 17:00:00+09:00", "2030-03-01 17:00:00+09:00"]
        self.assertEqual(list(issue_cutoff(times).astype(str)), expected)

    def test_cli_with_supplied_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.sites.to_csv(root / "sites.csv", index=False)
            self.composition.to_csv(root / "composition.csv", index=False)
            self.weights.assign(k=[1, 3, 3, 1]).to_csv(root / "smp.csv")
            self.actual.to_parquet(root / "actual.parquet", index=False)
            self.prediction.to_parquet(root / "predictions.parquet", index=False)
            result = subprocess.run([
                sys.executable, "-m", "GM_Ver3.settlement",
                "--sites", str(root / "sites.csv"), "--composition", str(root / "composition.csv"),
                "--smp", str(root / "smp.csv"), "--actual", str(root / "actual.parquet"),
                "--predictions", str(root / "predictions.parquet"),
            ], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout)["penalty_krw"], 160)


if __name__ == "__main__":
    unittest.main()
