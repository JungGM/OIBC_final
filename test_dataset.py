import unittest
import numpy as np
import pandas as pd
from audit_data import cutoff
from build_dataset import WA, WB, issued_table, make_month
from dataset import score_predictions

TZ = "Asia/Seoul"


def example_forecast(provider, times):
    cols = WA if provider == "wa" else WB
    parts = []
    for r in (10, 17):
        f = pd.DataFrame({provider+"_station": [provider.upper()+"001"]*len(times),
                          "round": r, "ts": times})
        for c in cols:
            f[c] = np.arange(len(times), dtype=float)+r
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


class DatasetTests(unittest.TestCase):
    def test_midnight_closes_previous_day(self):
        t = pd.DatetimeIndex(["2024-01-01 01:00", "2024-01-02 00:00", "2024-01-02 01:00"], tz=TZ)
        expected = pd.DatetimeIndex(["2023-12-31 17:00", "2023-12-31 17:00", "2024-01-01 17:00"], tz=TZ)
        self.assertTrue(cutoff(t).equals(expected))

    def test_hourly_changes_never_bridge_gaps_or_operating_days(self):
        t = pd.DatetimeIndex(["2024-01-01 01:00", "2024-01-01 03:00", "2024-01-02 00:00", "2024-01-02 01:00"], tz=TZ)
        f = issued_table(example_forecast("wa", t), "wa")
        self.assertTrue(f.wa_cloud__delta_prev.isna().all())
        self.assertTrue((f.wa_cloud__revision == 7).all())

    def test_no_silent_forecast_timestamp_rounding(self):
        t = pd.DatetimeIndex(["2024-01-01 01:55", "2024-01-01 03:00"], tz=TZ)
        f = issued_table(example_forecast("wb", t), "wb")
        self.assertEqual(len(f), 1)
        self.assertEqual(f.ts.iloc[0], t[1])

    def test_zero_negative_and_missing_targets_remain_distinct(self):
        times = pd.date_range("2024-01-01 01:00", periods=24, freq="h", tz=TZ)
        sites = pd.DataFrame(dict(site_id=["PV001"], current_group=["A"], capacity_kw=[100.],
            lat=[37.], lon=[127.], wa_station=["WA001"], wb_station=["WB001"], wa_dist=[100.], wb_dist=[100.]))
        truth = pd.DataFrame(dict(site_id=["PV001", "PV001"], ts=times[:2], energy=[0., -1.]))
        forecast = {p:example_forecast(p, times) for p in ("wa", "wb")}
        f, _ = make_month(sites, times, forecast, truth)
        self.assertEqual(f.target_eligible.sum(), 1)
        self.assertEqual(f.target_observed.sum(), 2)
        self.assertEqual(f.energy_kwh.isna().sum(), 22)
        self.assertEqual(f.energy_kwh.iloc[1], -1.)
        self.assertNotIn("current_group", f)
        self.assertNotIn("smp", f)
        self.assertEqual(f.issue_cutoff.nunique(), 1)
        second_site = sites.copy()
        second_site["site_id"] = "PV002"
        f2, _ = make_month(pd.concat([sites, second_site]), times, forecast, truth)
        self.assertTrue(f2.wa__r17_day_hours.eq(24).all())

    def test_group_error_cancels_before_absolute_value(self):
        t = pd.Timestamp("2024-01-01 12:00", tz=TZ)
        composition = pd.DataFrame(dict(site_id=["a", "b", "c"], current_group=["A", "A", "B"]))
        rows = pd.DataFrame(dict(site_id=["a", "b", "c"], ts=[t]*3, actual=[10., 10., 10.], prediction=[11., 9., 10.]))
        weights = pd.DataFrame(dict(ts=[t], smp=[100.], k=[3]))
        score, _ = score_predictions(rows, composition, weights)
        self.assertEqual(score["penalty_krw"], 0.)
        rows.loc[1, "prediction"] = 10.
        score, _ = score_predictions(rows, composition, weights)
        self.assertEqual(score["penalty_krw"], 30.)  # 0.1 * 100 * 3 * 1

    def test_incomplete_roster_is_not_zero_filled(self):
        t = pd.Timestamp("2024-01-01 12:00", tz=TZ)
        composition = pd.DataFrame(dict(site_id=["a", "b"], current_group=["A", "B"]))
        rows = pd.DataFrame(dict(site_id=["a"], ts=[t], actual=[10.], prediction=[11.]))
        weights = pd.DataFrame(dict(ts=[t], smp=[100.], k=[3]))
        score, _ = score_predictions(rows, composition, weights)
        self.assertIsNone(score["penalty_krw"])
        self.assertEqual(score["complete_hours"], 0)


if __name__ == "__main__":
    unittest.main()
