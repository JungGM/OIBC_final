"""Independent result reconciliation using wide matrices, not the training scorer."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out, data = args.out, args.data
    audit = json.loads((out/"audit.json").read_text(encoding="utf-8"))
    frozen = json.loads((out/"frozen_plan.json").read_text(encoding="utf-8"))
    for name, digest in audit["source_sha256"].items():
        assert file_hash(data/name) == digest, f"Source changed: {name}"
    assert file_hash(Path(__file__).with_name("backtest.py")) == frozen["script_sha256"]
    assert pd.Timestamp(frozen["fit_cutoff"]) == pd.Timestamp("2023-12-31 17:00+09:00")
    assert not set(frozen["input_features"]) & {"energy", "prediction", "smp", "k", "current_group"}
    sites = pd.read_csv(data/"sites.csv").set_index("site_id")
    selected = pd.read_csv(out/"composition_selected.csv").set_index("site_id").group
    assert len(selected) == 253 and selected.index.is_unique
    assert set(selected.index) == set(sites.index)
    assert sites.capacity_kw.groupby(selected).sum().min() >= 20000
    p = pd.read_parquet(out/"site_predictions.parquet")
    expected = pd.date_range("2024-01-01 01:00", "2025-06-01 00:00", freq="h", tz="Asia/Seoul")
    assert len(p) == len(expected)*253 and not p.duplicated(["site_id", "ts"]).any()
    assert set(p.site_id) == set(sites.index)
    assert set(p.ts) == set(expected)
    cols = ["baseline_reference", "ga_xgb_reference", "frozen_selected"]
    assert np.isfinite(p[cols]).all().all() and (p[cols] >= 0).all().all()
    truth = pd.read_parquet(data/"train/generation.parquet")
    truth = truth[truth.ts.isin(expected)].pivot(index="ts", columns="site_id", values="energy")
    truth = truth.reindex(columns=sites.index).dropna()
    prices = pd.read_csv(data/"train/smp.csv")
    prices["ts"] = pd.to_datetime(prices.ts)
    smp = prices.set_index("ts").loc[truth.index, "smp"].to_numpy()
    k = np.where(truth.index.hour.isin([11, 12, 13, 14, 15]), 3, 1)
    metrics = pd.read_csv(out/"metrics.csv")
    actual = truth.to_numpy(dtype=float)
    amounts = {}
    for name in cols:
        forecast = p.pivot(index="ts", columns="site_id", values=name).reindex(index=truth.index, columns=sites.index).to_numpy()
        groups = selected if name == "frozen_selected" else sites.current_group.map({"A": "G1", "B": "G2"})
        err = forecast-actual
        group_errors = np.column_stack([err[:, (groups.reindex(sites.index) == group).to_numpy()].sum(axis=1) for group in ["G1", "G2"]])
        charge = float(0.1*np.sum(smp*k*np.abs(group_errors).sum(axis=1)))
        revenue = float(np.sum(smp*actual.sum(axis=1)))
        reported = metrics[(metrics.model == name) & (metrics.period == "all") & (metrics.scope == "complete_253_panel")].iloc[0]
        np.testing.assert_allclose([charge, revenue-charge], [reported.penalty_krw, reported.settlement_krw], rtol=1e-10, atol=0.01)
        amounts[name] = charge
    f = pd.read_csv(out/"backtest_forecast_NOT_FOR_SUBMISSION.csv")
    f["ts"] = pd.to_datetime(f.ts)
    independent = p.assign(group=p.site_id.map(selected)).groupby(["group", "ts"]).frozen_selected.sum().sort_index()
    np.testing.assert_allclose(f.set_index(["group", "ts"]).forecast_kwh.sort_index(), independent, rtol=1e-12)
    result = {"status": "PASS", "source_files_unchanged": True, "frozen_script_matches": True,
              "all_forecast_rows_present": True, "group_capacity_valid": True,
              "complete_253_hours": len(truth), "independent_penalty_krw": amounts,
              "limitation": "Local consistency checks; not organizer certification or server acceptance."}
    (out/"verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
