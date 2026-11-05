import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

from GM_Ver1.common import Panel, write_json, sha256
from GM_Ver1.export import verify_predictions
from GM_Ver1.tcn import ForecastTCN, receptive_field


class BoundaryTests(unittest.TestCase):
    def test_selection_labels_must_be_available_at_forecast_origin(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.parquet"
            for first, valid in [("2024-01-01 01:00", False), ("2024-01-02 01:00", True)]:
                times = pd.date_range(first, periods=24, freq="h", tz="Asia/Seoul").as_unit("ns")
                pd.DataFrame(dict(site_id=["PV001"]*24, ts=times, prediction=np.zeros(24))).to_parquet(path, index=False)
                write_json(path.with_suffix(".json"), dict(sha256=sha256(path),
                    training_end="2023-12-31 17:00:00+09:00", selection_labels_end="2024-01-01 00:00:00+09:00"))
                panel = SimpleNamespace(times=lambda ids: times.asi8.reshape(1, 24),
                    site_index=np.array([0]), sites=["PV001"], nsites=1)
                if valid:
                    verify_predictions(path, panel, np.array([0]))
                else:
                    with self.assertRaises(AssertionError):
                        verify_predictions(path, panel, np.array([0]))

    def test_pandas_microseconds_cannot_bypass_17h_cutoff(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            np.save(path / "X.npy", np.zeros((2, 24, 1), dtype=np.float32))
            y = np.ones((2, 24), dtype=np.float32)
            y[0, 0], y[0, 1], y[0, 2] = np.nan, -1, 0
            np.save(path / "y.npy", y)
            write_json(path / "panel.json", dict(features=["capacity_kw"], capacity_kw=[100],
                site_ids=["PV001"], first_operating_day="2025-05-30 00:00:00+09:00"))
            panel = Panel(path)
            _, _, mask, _ = panel.batch(np.array([0, 1]), "2025-05-31 17:00:00+09:00")
            self.assertFalse(mask[0, 0])
            self.assertFalse(mask[0, 1])
            self.assertTrue(mask[0, 2])  # Real zero remains in the loss.
            self.assertEqual(mask[1].sum(), 17)
            self.assertFalse(mask[1, 17:].any())
            expected = pd.Timestamp("2025-05-31 00:00:00+09:00").value
            self.assertEqual(panel.times([0])[0, -1], expected)
            self.assertEqual(panel.days.dtype.unit, "ns")
            panel.X._mmap.close()
            panel.y._mmap.close()

    def test_tcn_is_causal_and_covers_24h(self):
        torch.set_num_threads(4)
        torch.manual_seed(3)
        config = dict(embedding_dim=4, channels=16, levels=4, kernel_size=3, dropout=.1)
        model = ForecastTCN(5, config, 2).eval()
        x = torch.randn(2, 24, 5)
        changed = x.clone()
        changed[:, 12:] += 10
        with torch.no_grad():
            a = model(x, torch.tensor([0, 1]))
            b = model(changed, torch.tensor([0, 1]))
        torch.testing.assert_close(a[:, :12], b[:, :12], rtol=0, atol=0)
        self.assertEqual(a.shape, (2, 24))
        self.assertTrue(torch.isfinite(a).all() and (a >= 0).all())
        self.assertGreaterEqual(receptive_field(config), 24)


if __name__ == "__main__":
    unittest.main()
