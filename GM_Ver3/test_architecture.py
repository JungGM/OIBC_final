import unittest

import numpy as np
import torch

from .architecture import l1_coefficient
from .dynamic import Gate, gate_loss
from .experts import Store


class ArchitectureTests(unittest.TestCase):
    def test_sequence_history_excludes_target_and_later_forecasts(self):
        store = Store.__new__(Store)
        store.history = np.arange(10, dtype="float32").reshape(10, 1, 1)
        result = store.past_sequences(np.array([4]), np.array([0]), 3)
        np.testing.assert_array_equal(result.ravel(), [1., 2., 3.])
        self.assertTrue(np.isnan(store.past_sequences(np.array([1]), np.array([0]), 3)[0, 0, 0]))

    def test_dynamic_weights_make_convex_predictions(self):
        model = Gate(2, 3, 4, 0.)
        p = torch.tensor([[[0., 3., 9.]]])
        estimate, weights = model(torch.zeros(1, 1, 2), p)
        torch.testing.assert_close(weights.sum(-1), torch.ones(1, 1))
        torch.testing.assert_close(estimate, p.mean(-1))

    def test_economic_loss_increases_when_site_errors_align(self):
        y = torch.tensor([[.5, .5, .5, .5]])
        q = torch.ones(1)
        cap = torch.ones(4)
        groups = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        independent = y+torch.tensor([[.1, -.1, .1, -.1]])
        aligned = y+torch.tensor([[.1, .1, -.1, -.1]])
        self.assertLess(gate_loss(independent, y, q, cap, groups, True).item(),
                        gate_loss(aligned, y, q, cap, groups, True).item())

    def test_group_combination_recovers_weighted_l1_median(self):
        bottom = np.zeros((3, 2))
        direct = np.full((3, 2), 10.)
        y = np.array([[2., 2.], [3., 3.], [8., 8.]])
        self.assertAlmostEqual(l1_coefficient(y, bottom, direct, np.array([1., 1., 5.])), .8)


if __name__ == "__main__":
    unittest.main()
