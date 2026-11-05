"""Compare low-penalty candidates and test the 6.7B KRW target under common actuals."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import prices, timestamps
from .decision import aggregate, apply_blend, blend_weights, vpp
from .fit import FOLDS, safe_output
from .run import Panel, calibrate, load_predictions, score


TARGET = 6_700_000_000
RECIPES = ["ver2", "gru_v1", "blend", "blend_bias"]


def load_gru(root, fold, times, sites, cutoff):
    directory = root/fold
    path = directory/"predictions.parquet"
    meta = json.loads((directory/"predictions.json").read_text())
    if any(pd.Timestamp(meta[key]) > cutoff for key in ("training_end", "selection_labels_end")):
        raise ValueError("GRU artifact exceeds forecast cutoff")
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != meta["sha256"]:
            raise ValueError("GRU artifact digest mismatch")
    frame = pd.read_parquet(path, columns=["site_id", "ts", "prediction"])
    frame["ts"] = timestamps(frame.ts)
    keys = pd.MultiIndex.from_product([times, sites], names=["ts", "site_id"])
    values = frame.set_index(["ts", "site_id"]).prediction.reindex(keys).to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Missing or invalid GRU predictions")
    return values.reshape(len(times), len(sites))


def read_panel(raw, artifacts, gru, sites):
    train = prices(raw/"train/smp.csv")
    index = train.index.searchsorted(pd.Timestamp("2024-12-24 01:00", tz=train.index.tz))
    times = train.index[index:]
    boundary = times.searchsorted(pd.Timestamp("2025-01-01 01:00", tz=times.tz))
    old = np.concatenate([
        load_gru(gru, "confirm_2024", times[:boundary], sites,
                 times[0].normalize()-pd.Timedelta(hours=7)),
        load_gru(gru, "check_2025", times[boundary:], sites,
                 times[boundary].normalize()-pd.Timedelta(hours=7)),
    ])
    forecasts = np.concatenate([load_predictions(artifacts, fold) for fold in FOLDS[:-1]])
    y = np.load(artifacts/"train_y.npy", mmap_mode="r").reshape(-1, len(sites))[index:]
    use = np.isfinite(y).all(axis=1)
    return Panel(times, np.concatenate([forecasts, old[..., None]], axis=-1), y,
                 (train.smp*train.k).to_numpy()[index:]).select(use)


def experts(panel, base_weights):
    ver2 = apply_blend(panel.x[..., :-1], panel.regime, base_weights)
    return np.stack([ver2, panel.x[..., -1]], axis=-1)


def train_pair(past, future, reference, capacity, recipe):
    base_weights = blend_weights(past.x[..., :-1], past.y, past.q, past.regime, reference, False)
    level = experts(past, base_weights)
    if recipe in ("ver2", "gru_v1"):
        weights = np.eye(2)[[int(recipe == "gru_v1")]]
    else:
        weights = blend_weights(level, past.y, past.q, past.regime, reference, False)
    meta = Panel(past.times, level, past.y, past.q)
    scale = calibrate(meta, weights) if recipe == "blend_bias" else np.ones(2)

    def predict(values, panel):
        return apply_blend(values, panel.regime, weights)*scale[panel.regime, None]

    fitted = predict(level, past)
    forecast = predict(experts(future, base_weights), future)
    first, solver = vpp(fitted-past.y, past.q, capacity, reference, past.months)
    return forecast, first, dict(base_weights=base_weights.tolist(), mixture=weights.tolist(),
                                 scale=scale.tolist(), solver=solver)


def scenarios(price, future, historical, actual, models, draws=240):
    """Two independent anchor assumptions; actual is shared across all candidates."""
    hours = len(actual)
    if hours % 24 or hours < 72:
        raise ValueError("Need contiguous full days for residual blocks")
    rng = np.random.default_rng(20261107)
    picks = rng.integers(hours//24-2, size=(draws, (len(price)//24+2)//3))
    indices = (24*picks[..., None] + np.arange(72)).reshape(draws, -1)[:, :len(price)]
    q = (price.smp*price.k).to_numpy()
    smp = price.smp.to_numpy()
    membership = np.stack([column for _, first in models.values() for column in (first, ~first)],
                          axis=1).astype("float32")
    predicted = np.concatenate([aggregate(pred, first) for pred, first in models.values()], axis=1)
    report = {}
    for anchor in ("ver2", "gru_v1"):
        residual = (actual-historical[anchor]).astype("float32")
        center = future[anchor].astype("float32")
        gross = []
        fines = []
        for ids in indices:
            # Common actuals at site level preserve equality of gross across assignments.
            realized = np.maximum(center+residual[ids], 0)
            grouped = realized @ membership
            gross.append(float(smp @ (grouped[:, 0]+grouped[:, 1])))
            by_group = .1*(q[:, None]*np.abs(grouped-predicted)).sum(axis=0)
            fines.append(by_group.reshape(-1, 2).sum(axis=1))
        gross = np.array(gross)
        fines = np.array(fines)
        report[anchor] = dict(
            gross_mean_krw=float(gross.mean()), gross_p90_krw=float(np.percentile(gross, 90)),
            no_penalty_target_hits=int((gross >= TARGET).sum()), draws=draws,
            candidates={
                name: dict(mean_penalty_krw=float(fines[:, i].mean()),
                           mean_settlement_krw=float((gross-fines[:, i]).mean()),
                           p10_krw=float(np.percentile(gross-fines[:, i], 10)),
                           p90_krw=float(np.percentile(gross-fines[:, i], 90)),
                           target_hits=int((gross-fines[:, i] >= TARGET).sum()))
                for i, name in enumerate(models)
            })
    return report


def run(raw, artifacts, gru, out):
    sites = pd.read_csv(raw/"sites.csv").sort_values("site_id")
    cap = sites.capacity_kw.to_numpy()
    reference = sites.current_group.eq("A").to_numpy()
    panel = read_panel(raw, artifacts, gru, sites.site_id)
    operating_day = (panel.times-pd.Timedelta(hours=1)).normalize()
    records = {}
    may_forecasts = {}
    may_scores = {}
    for month in (3, 4, 5):
        start = pd.Timestamp(2025, month, 1, tz=panel.times.tz)
        past = panel.select(panel.times <= start-pd.Timedelta(hours=7))
        check = panel.select((operating_day >= start) & (operating_day < start+pd.offsets.MonthBegin()))
        if month == 5:
            selected = min(RECIPES, key=lambda name: np.mean([
                values[name]/values["ver2"] for values in records.values()]))
            print(f"Selected before May comparison: {selected}", flush=True)
        row = {}
        for recipe in RECIPES:
            pred, first, _ = train_pair(past, check, reference, cap, recipe)
            result = score(check, pred, first)
            row[recipe] = result["penalty_krw"]
            if month == 5:
                may_forecasts[recipe] = pred
                may_scores[recipe] = result
        if month < 5:
            records[str(month)] = row
        print(f"Month {month} penalty: {row}", flush=True)
    deployed = selected if may_scores[selected]["penalty_krw"] < may_scores["ver2"]["penalty_krw"] else "ver2"
    price = prices(raw/"smp.csv")
    cutoff = price.index[0].normalize()-pd.Timedelta(hours=7)
    history = panel.select(panel.times <= cutoff)
    old = load_gru(gru, "final", price.index, sites.site_id, cutoff)
    x = np.concatenate([load_predictions(artifacts, "evaluation"), old[..., None]], axis=-1)
    future = Panel(price.index, x, np.empty((len(price), 0)), np.zeros(len(price)))
    models = {}
    details = {}
    for recipe in dict.fromkeys(["ver2", "gru_v1", deployed]):
        pred, first, details[recipe] = train_pair(history, future, reference, cap, recipe)
        models[recipe] = (pred, first)
    expected = scenarios(price, {k: x[0] for k, x in models.items()}, may_forecasts, check.y, models)
    pred, first = models[deployed]
    forecast = aggregate(pred, first)
    capacities = [float(cap[first].sum()), float(cap[~first].sum())]
    if forecast.shape != (8760, 2) or min(capacities) < 20000 or not np.isfinite(forecast).all() or (forecast < 0).any():
        raise ValueError("Invalid submission")
    times = price.index
    pd.DataFrame(dict(site_id=sites.site_id, group=np.where(first, "G1", "G2"))).to_csv(
        out/"composition.csv", index=False)
    pd.DataFrame(dict(group=np.repeat(["G1", "G2"], len(times)), ts=np.tile(times.astype(str), 2),
                      forecast_kwh=forecast.T.reshape(-1))).to_csv(out/"forecast.csv", index=False)
    report = dict(target_krw=TARGET, selection=records, selected=selected, deployed=deployed,
                  may=may_scores, capacity_kw=capacities, models=details,
                  scenarios=expected, assumption="May site residuals transferred by three-day blocks",
                  legacy_gru_cutoffs="2023-12-31 for December 2024, 2024-12-31 for 2025 validation")
    (out/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(deployed=deployed, scenarios=expected), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "artifacts", "gru", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    run(args.raw, args.artifacts, args.gru, safe_output(args.raw, args.out))
