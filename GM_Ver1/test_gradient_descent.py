import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from GM_Ver1.gradient_descent import (project_simplex, loss_gradient, projected_descent,
                                     aggregate_complete, fit_weights, TRAIN_END)


class GradientDescentTests(unittest.TestCase):
    def test_missing_site_excludes_whole_hour_but_zero_remains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ts = pd.date_range("2025-01-01 01:00", periods=3, freq="h", tz="Asia/Seoul")
            pd.DataFrame(dict(site_id=["a", "b"], current_group=["A", "B"],
                              capacity_kw=[100., 100.])).to_parquet(root/"reference_composition.parquet")
            pd.DataFrame(dict(ts=ts, smp=[100.]*3, k=[1]*3)).to_parquet(root/"training_weights.parquet")
            keys = pd.DataFrame(dict(site_id=["a", "b"]*3, ts=ts.repeat(2)))
            y = np.array([0., 0., 1., np.nan, -1., 2.])
            group = aggregate_complete(keys, np.zeros((6, 6)), y, np.full(6, 100.), root)
            self.assertEqual(group["eligible_hours"], 2)
            self.assertEqual(group["excluded_hours"], 1)
            np.testing.assert_allclose(group["residual"][0], 0.)
            np.testing.assert_allclose(group["residual"][1, 0], .01)

    def test_training_never_uses_hours_after_17h_cutoff(self):
        times = pd.DatetimeIndex([pd.Timestamp(TRAIN_END), pd.Timestamp(TRAIN_END)+pd.Timedelta(hours=1)])
        group = dict(times=times, residual=np.zeros((2, 2, 6)), price_weight=np.ones((2, 2)), scale=100.)
        config = dict(mode="global", delta=.001, regularization=.001)
        result = fit_weights(group, TRAIN_END, config, np.ones(6)/6)
        self.assertEqual(result["training_complete_hours"], 1)
        self.assertEqual(pd.Timestamp(result["last_training_hour"]), pd.Timestamp(TRAIN_END))

    def test_projection_obeys_constraints_and_known_solution(self):
        np.testing.assert_allclose(project_simplex([.2, -.1, 1.1]), [.05, 0., .95])
        np.testing.assert_allclose(project_simplex([0., 0., 0.]), np.ones(3)/3)
        for vector in [np.array([1e3, -1e3, .2]), np.array([.2, .3, .5])]:
            w = project_simplex(vector)
            self.assertTrue((w >= 0).all())
            self.assertAlmostEqual(w.sum(), 1.)

    def test_analytic_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(20)
        e, q = rng.normal(scale=.03, size=(40, 6)), rng.uniform(0, 300, 40)
        w, prior = np.ones(6)/6, np.array([.1, .2, .1, .3, .2, .1])
        _, gradient = loss_gradient(w, e, q, .001, .01, prior)
        numerical = []
        for j in range(6):
            shift = np.eye(6)[j]*1e-7
            numerical.append((loss_gradient(w+shift, e, q, .001, .01, prior)[0]-
                              loss_gradient(w-shift, e, q, .001, .01, prior)[0])/2e-7)
        np.testing.assert_allclose(gradient, numerical, atol=1e-9, rtol=1e-6)

    def test_projected_solver_matches_independent_slsqp(self):
        rng = np.random.default_rng(9)
        e, q, prior = rng.normal(scale=.05, size=(90, 6)), rng.uniform(1, 300, 90), np.ones(6)/6
        fun = lambda w: loss_gradient(w, e, q, .005, .001, prior)
        independent = minimize(lambda w: fun(w)[0], prior, jac=lambda w: fun(w)[1], method="SLSQP",
            bounds=[(0, 1)]*6, constraints={"type":"eq", "fun":lambda w:w.sum()-1,
            "jac":lambda w:np.ones(6)}, options={"ftol":1e-13, "maxiter":2000})
        self.assertTrue(independent.success)
        actual = projected_descent(e, q, .005, .001, prior, prior)
        self.assertLessEqual(actual["loss"]-independent.fun, 1e-9)
        self.assertLess(actual["stationarity"], 1e-7)


if __name__ == "__main__":
    unittest.main()
