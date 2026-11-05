import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from GM_Ver3.decisions import ENSEMBLES, allocation_candidates
from GM_Ver3.fit import MODELS, dump_json
from GM_Ver3.nested import NESTED_FOLDS, NestedExperiment, hourly_diagnostics, sha256, stable_rank
from GM_Ver3.seasonal import site_metrics
from GM_Ver3.submission import validate


class ValidationTest(unittest.TestCase):
    def test_partial_fleet_metrics_use_observed_capacity_only(self):
        actual = np.array([[8., np.nan], [0., 5.]])
        prediction = np.array([[10., 7.], [0., 5.]])
        metrics = site_metrics(actual, prediction, np.array([10., 20.]), np.isfinite(actual))
        self.assertEqual(metrics["nmae_pct"], 5)

    def test_stability_ranking_penalizes_bad_months(self):
        folds = [{"site_lgbm_full": {"penalty_krw": 100}, "stable": {"penalty_krw": 90},
                  "spiky": {"penalty_krw": penalty}} for penalty in [60, 115]]
        self.assertEqual(stable_rank(folds, ["spiky", "stable"]), ["stable", "spiky"])

    def test_allocation_keeps_both_groups_feasible(self):
        rng = np.random.default_rng(7)
        error = rng.normal(size=(80, 6))
        q = rng.uniform(50, 150, size=80)
        capacity = np.array([8_000, 7_000, 6_500] * 2)
        reference = np.arange(6) < 3
        periods = np.repeat([1, 2], 40)
        solutions = allocation_candidates(error, q, capacity, np.arange(6), reference, periods)
        for first in solutions.values():
            self.assertGreaterEqual(capacity[first].sum(), 20_000)
            self.assertGreaterEqual(capacity[~first].sum(), 20_000)

    def test_hourly_penalty_and_cancellation_reconcile(self):
        truth = np.array([[10, 10, 0], [10, 10, 0]])
        site = np.array([[15, 5, 2], [10, 10, 4]])
        pred = np.array([[20, 2], [20, 4]])
        result = hourly_diagnostics(truth, pred, site, np.array([True, True, False]),
                                    np.array([100, 100]), np.array([1, 3]), np.array([10, 11]))
        self.assertEqual(result["penalty_krw"], 140)
        self.assertEqual(result["penalty_by_closing_hour_krw"]["11"], 120)
        self.assertEqual(result["bottom_up_cancellation_krw_by_group"], {"G1": 100, "G2": 0})

    def test_january_group_history_excludes_late_december_and_january_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            experiment = NestedExperiment.__new__(NestedExperiment)
            experiment.folds = NESTED_FOLDS
            experiment.archive = Path(temp)
            experiment.times = pd.date_range("2024-12-24 01:00", "2025-01-02", freq="h", tz="Asia/Seoul")
            hours = len(experiment.times)
            experiment.y = np.ones((hours, 2))
            experiment.y[0, 0] = np.nan
            experiment.x = np.zeros((hours, 2, 1))
            experiment.context = [0]
            experiment.smp = np.ones(hours)
            experiment.k = np.ones(hours)
            warmup = experiment.archive / "warmup"
            warmup.mkdir()
            original = np.broadcast_to(np.arange(192)[:, None], (192, 2))
            np.savez(warmup / "oof_recipes.npz", **dict.fromkeys((*MODELS, *ENSEMBLES), original))
            forecasts, actual, _, times, _ = experiment.history("jan")
            self.assertEqual(str(times[-1]), "2024-12-31 17:00:00+09:00")
            self.assertEqual(len(actual), 184)
            self.assertEqual(forecasts["ridge"][0, 0], 1)
            self.assertEqual(forecasts["ridge"][-1, 0], 184)

    def test_frozen_manifest_rejects_edited_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / "selection.json"
            dump_json(plan, {"selected": "ga"})
            artifact = root / "frozen.npz"
            np.savez(artifact, first=[True, False])
            dump_json(root / "frozen.json", {"selection_sha256": sha256(plan), "forecast_sha256": sha256(artifact)})
            experiment = NestedExperiment.__new__(NestedExperiment)
            experiment.archive = root
            experiment.checked_manifest()
            dump_json(plan, {"selected": "reference"})
            with self.assertRaisesRegex(ValueError, "has changed"):
                experiment.checked_manifest()

    def test_submission_rejects_duplicate_hour(self):
        sites = pd.DataFrame({"site_id": [f"S{i:03d}" for i in range(253)], "capacity_kw": 170.})
        composition = pd.DataFrame({"site_id": sites.site_id, "group": np.where(sites.index < 126, "G1", "G2")})
        times = pd.date_range("2025-06-01 01:00", periods=8_760, freq="h", tz="Asia/Seoul")
        forecast = pd.DataFrame({"group": np.repeat(["G1", "G2"], len(times)),
                                 "ts": np.tile(times.astype(str), 2), "forecast_kwh": 0.})
        self.assertEqual(validate(sites, times, composition, forecast)["forecast_rows"], 17_520)
        forecast.loc[0, "ts"] = forecast.loc[1, "ts"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate(sites, times, composition, forecast)


if __name__ == "__main__":
    unittest.main()
