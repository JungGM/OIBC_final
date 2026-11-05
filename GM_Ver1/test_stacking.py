import unittest
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from GM_Ver1.stacking import load_level1, ridge_fit


class StackingTests(unittest.TestCase):
    def test_weighted_solution_matches_independent_sklearn_solver(self):
        rng = np.random.default_rng(20)
        x = rng.normal(size=(1000, 6))
        x[:, 1] = x[:, 0]+rng.normal(scale=.01, size=1000)
        y = x@np.array([.2, .1, -.3, .4, 0., .2])+.05+rng.normal(scale=.1, size=1000)
        w = rng.uniform(.1, 5, size=1000)
        alpha = .01
        ours = ridge_fit(x, y, w, alpha)
        mean, scale = np.array(ours["training_mean"]), np.array(ours["training_scale"])
        independent = Ridge(alpha=alpha*w.sum(), solver="svd").fit((x-mean)/scale, y, sample_weight=w)
        np.testing.assert_allclose(ours["coefficients"], independent.coef_/scale, rtol=1e-9, atol=1e-9)
        self.assertAlmostEqual(ours["intercept"], independent.intercept_-mean@(independent.coef_/scale), places=10)

    def test_final_fit_predictions_cannot_train_meta_model(self):
        with self.assertRaisesRegex(ValueError, "Final base fits"):
            load_level1(Path("unused"), Path("unused"), "final", labels=True)


if __name__ == "__main__":
    unittest.main()
