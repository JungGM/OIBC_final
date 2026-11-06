"""Score validated candidate and archived CSVs on shared simulated actuals."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .fit import dump_json, input_files, start_at, tree_predict
from .settlement import price_arrays
from .submission import validate_files


def submission(sites, times, folder):
    validate_files(sites, times, folder)
    groups = pd.read_csv(folder/"composition.csv", dtype={"site_id": str}).set_index("site_id")
    first = groups.group.reindex(sites.site_id).eq("G1").to_numpy()
    frame = pd.read_csv(folder/"forecast.csv")
    frame["ts"] = pd.to_datetime(frame.ts, utc=True).dt.tz_convert("Asia/Seoul")
    forecast = frame.pivot(index="ts", columns="group", values="forecast_kwh").reindex(
        index=times, columns=["G1", "G2"]).to_numpy()
    return forecast, first


def compare(raw, base, decision, candidate, archives, threads):
    sites, train_prices, eval_prices = input_files(raw)
    index, smp, k = price_arrays(eval_prices)
    paths = {"candidate": candidate/"submission", **{folder.name: folder for folder in archives}}
    models = {label: submission(sites, index, folder) for label, folder in paths.items()}
    weights = np.stack([a for _, group in models.values() for a in (group, ~group)], axis=1).astype("float32")
    points = np.column_stack([forecast for forecast, _ in models.values()])
    y = np.load(base/"train_y.npy", mmap_mode="r")
    left = train_prices.index.searchsorted(start_at("2025-05-01"))
    truth = y[left:]
    if len(truth) % 24 or not (np.isfinite(truth) & (truth >= 0)).all():
        raise ValueError("Need complete contiguous May days")
    old = joblib.load(base/"may/lgbm_full.joblib")
    eval_x = np.load(base/"test_x.npy", mmap_mode="r")
    old_eval = tree_predict(old, eval_x, sites.capacity_kw.to_numpy(), threads)
    old_may = np.load(base/"may/lgbm_full.npy")
    recipe = json.loads((candidate/"report.json").read_text())["selected"]
    with np.load(decision/"may/oof.npz") as bundle:
        candidate_may = bundle[recipe if recipe in bundle else "economic_gate"]
    anchors = {"frozen_lightgbm": (old_eval, old_may),
               "candidate": (np.load(candidate/"selected_site.npy"), candidate_may)}
    rng = np.random.default_rng(20261106)
    blocks = rng.integers(len(truth)//24-2, size=(300, (len(smp)//24+2)//3))
    hours = (24*blocks[..., None]+np.arange(72)).reshape(len(blocks), -1)[:, :len(smp)]
    runs = {}
    for anchor, (level, before) in anchors.items():
        diff = truth-before
        gross, fine = [], []
        for ids in hours:
            actual = np.maximum(level+diff[ids], 0.) @ weights
            gross.append(smp @ actual[:, :2].sum(-1))
            penalties = ((.1*smp*k) @ np.abs(actual-points)).reshape(-1, 2).sum(-1)
            fine.append(penalties)
        fine = np.array(fine)
        net = np.asarray(gross)[:, None]-fine
        runs[anchor] = {name: dict(mean_settlement_krw=float(net[:, j].mean()),
                                  p10_settlement_krw=float(np.percentile(net[:, j], 10)),
                                  p90_settlement_krw=float(np.percentile(net[:, j], 90)),
                                  candidate_minus_archive_krw=float((net[:, 0]-net[:, j]).mean()),
                                  at_or_above_64_65=int((net[:, j] >= 6_465_000_000).sum()))
                        for j, name in enumerate(models)}
    dump_json(candidate/"archived_submission_comparison.json", dict(
        runs=runs, draws_per_anchor=len(hours), assumption="shared simulated actuals from May residual blocks"))
    print(json.dumps(runs, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "base", "decision", "candidate"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--archive", type=Path, action="append", default=[])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    compare(args.raw, args.base, args.decision, args.candidate, args.archive, args.threads)
