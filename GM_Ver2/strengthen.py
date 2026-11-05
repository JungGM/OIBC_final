"""Expand the GRU chronologically and compare forecast/VPP pairs."""

import argparse
from pathlib import Path

import json
import numpy as np
import pandas as pd
import torch

from .data import prices
from .decision import aggregate, apply_blend, blend_weights, ga_baseline, vpp
from .fit import FOLDS, ORIGINS, safe_output
from .models import NAMES, recurrent
from .run import Panel, load_predictions, score, settlement


RECIPES = ["ver2", "expanding_gru", "expanded_l1", "half_gru"]
POLICIES = ["reference", "ga", "milp_mean", "milp_robust"]


def fit_expanding(raw, artifacts, destination, threads):
    sites = pd.read_csv(raw/"sites.csv").sort_values("site_id")
    names = json.loads((artifacts/"data_summary.json").read_text())["train"]["features"]
    x = np.load(artifacts/"train_x.npy", mmap_mode="r")
    y = np.load(artifacts/"train_y.npy", mmap_mode="r")
    eval_x = np.load(artifacts/"eval_x.npy", mmap_mode="r")
    price = prices(raw/"train/smp.csv")
    days = price.index[::24].normalize()
    starts = [days.searchsorted(pd.Timestamp(origin, tz=days.tz)) for origin in ORIGINS]
    warm = None
    for i, (fold, origin, stop) in enumerate(zip(FOLDS, ORIGINS, starts)):
        end = starts[i+1] if i+1 < len(starts) else len(days)
        cutoff = pd.Timestamp(origin, tz=days.tz)-pd.Timedelta(hours=7)
        q = (price.smp*price.k).where(price.index <= cutoff, 0).to_numpy().reshape(-1, 24)
        future = eval_x if fold == "evaluation" else x[stop:end]
        weights = destination/f"{fold}.pt"
        output = destination/f"{fold}.npy"
        if not output.exists():
            print(f"Expanding GRU {fold}, cutoff {cutoff}", flush=True)
            forecast = recurrent(x, y, q, sites.capacity_kw.to_numpy(), np.arange(stop),
                                 future, names, 2, threads, 20261106, weights, warm=warm,
                                 expanding=True, recent_start=starts[i-1]-1 if i else None)
            forecast *= future[..., names.index("solar_signed")] > -0.104528
            if not np.isfinite(forecast).all():
                raise ValueError("Nonfinite expanding GRU forecast")
            partial = output.with_suffix(".partial.npy")
            np.save(partial, forecast.astype("float32"))
            partial.replace(output)
        warm = torch.load(weights, map_location="cpu", weights_only=True)


def weights_for(panel, recipe, reference):
    if recipe == "expanding_gru":
        return np.eye(len(NAMES)+1)[[-1]]
    if recipe == "expanded_l1":
        return blend_weights(panel.x, panel.y, panel.q, panel.regime, reference, False)
    baseline = blend_weights(panel.x[..., :-1], panel.y, panel.q, panel.regime,
                             reference, False)
    if recipe == "half_gru":
        return np.c_[.5*baseline, [.5]]
    return np.c_[baseline, [0.]]


def forecast_for(panel, weights):
    return apply_blend(panel.x, panel.regime, weights)


def allocation(errors, panel, cap, reference, policy):
    if policy == "reference":
        return reference, {"source": "reference"}
    if policy == "ga":
        return ga_baseline(errors, panel.q, cap, reference), {"source": "ga"}
    return vpp(errors, panel.q, cap, reference, panel.months,
               seconds=12, risk=0. if policy == "milp_mean" else .2)


def read_experts(artifacts, destination, fold):
    original = load_predictions(artifacts, fold)
    added = np.load(destination/f"{fold}.npy").reshape(-1, 253)
    return np.concatenate([original, added[..., None]], axis=-1)


def paired_gain_interval(baseline_errors, challenger_errors, q):
    if len(q) % 24:
        raise ValueError("The paired audit requires full operating days")
    gain = .1*q*(np.abs(baseline_errors).sum(axis=1) - np.abs(challenger_errors).sum(axis=1))
    daily = gain.reshape(-1, 24).sum(axis=1)
    rng = np.random.default_rng(20261106)
    starts = rng.integers(len(daily)-2, size=(2000, (len(daily)+2)//3))
    ids = (starts[..., None] + np.arange(3)).reshape(2000, -1)[:, :len(daily)]
    draws = daily[ids].sum(axis=1)
    return dict(observed_gain_krw=float(gain.sum()), p05_krw=float(np.percentile(draws, 5)),
                p95_krw=float(np.percentile(draws, 95)), positive_share=float((draws > 0).mean()))


def compare_pairs(raw, artifacts, destination):
    sites = pd.read_csv(raw/"sites.csv").sort_values("site_id")
    cap = sites.capacity_kw.to_numpy()
    reference = sites.current_group.eq("A").to_numpy()
    price = prices(raw/"train/smp.csv")
    evaluation = prices(raw/"smp.csv")
    start = pd.Timestamp(ORIGINS[0], tz=price.index.tz) + pd.Timedelta(hours=1)
    index = price.index.searchsorted(start)
    actual = np.load(artifacts/"train_y.npy", mmap_mode="r").reshape(-1, len(sites))[index:]
    x = np.concatenate([read_experts(artifacts, destination, fold) for fold in FOLDS[:-1]])
    complete = np.isfinite(actual).all(axis=1)
    panel = Panel(price.index[index:], x, actual, (price.smp*price.k).to_numpy()[index:]).select(complete)
    operating_day = (panel.times-pd.Timedelta(hours=1)).normalize()
    records = {}
    candidates = [(recipe, policy) for recipe in RECIPES for policy in POLICIES]
    prior_pair = ("ver2", "milp_robust")
    for label in ORIGINS[1:3]:
        t = pd.Timestamp(label, tz=panel.times.tz)
        past = panel.select(panel.times <= t-pd.Timedelta(hours=7))
        present = panel.select((operating_day >= t) & (operating_day < t+pd.offsets.MonthBegin()))
        records[label] = {}
        for recipe in RECIPES:
            weights = weights_for(past, recipe, reference)
            errors = forecast_for(past, weights)-past.y
            forecast = forecast_for(present, weights)
            for policy in POLICIES:
                first, _ = allocation(errors, past, cap, reference, policy)
                records[label][f"{recipe}/{policy}"] = score(present, forecast, first)["penalty_krw"]
            print(f"Selection {label}: scored four allocations for {recipe}", flush=True)

    def selection_metric(pair):
        return np.mean([month["/".join(pair)]/month["/".join(prior_pair)]
                        for month in records.values()])

    selected = min(candidates, key=selection_metric)
    print(f"Selected before May scoring: {selected}", flush=True)
    may = pd.Timestamp(ORIGINS[3], tz=panel.times.tz)
    past = panel.select(panel.times <= may-pd.Timedelta(hours=7))
    present = panel.select(operating_day >= may)
    audit = {}
    residual = {}
    for pair in dict.fromkeys([prior_pair, selected]):
        recipe, policy = pair
        weights = weights_for(past, recipe, reference)
        first, solver = allocation(forecast_for(past, weights)-past.y, past, cap, reference, policy)
        forecast = forecast_for(present, weights)
        may_last = present.times <= pd.Timestamp(ORIGINS[4], tz=may.tz)-pd.Timedelta(hours=7)
        audit["/".join(pair)] = dict(
            full=score(present, forecast, first),
            legacy_window=score(present.select(may_last), forecast[may_last], first),
            solver=solver)
        residual[pair] = aggregate(forecast-present.y, first)
    baseline_loss = audit["/".join(prior_pair)]["full"]["penalty_krw"]
    trial_loss = audit["/".join(selected)]["full"]["penalty_krw"]
    # May is a regression guard. If it fails, keep the already established pair.
    deployed = selected if trial_loss < baseline_loss else prior_pair
    history = panel.select(panel.times <= evaluation.index[0].normalize()-pd.Timedelta(hours=7))
    weights = weights_for(history, deployed[0], reference)
    first, solver = allocation(forecast_for(history, weights)-history.y, history, cap,
                               reference, deployed[1])
    future = Panel(evaluation.index, read_experts(artifacts, destination, "evaluation"),
                   np.empty((len(evaluation), 0)), np.zeros(len(evaluation)))
    prediction = aggregate(forecast_for(future, weights), first)
    times = evaluation.index
    caps = [float(cap[first].sum()), float(cap[~first].sum())]
    if (prediction.shape != (8760, 2) or not np.isfinite(prediction).all()
            or (prediction < 0).any() or min(caps) < 20000):
        raise ValueError("Invalid strengthened submission")
    expected = settlement(prediction, evaluation, present.times, residual[deployed])
    report = dict(selected=selected, deployed=deployed, selection=records,
                  may=audit, weights=dict(zip([*NAMES, "expanding_gru"], weights[0].tolist())),
                  capacity_kw=caps, solver=solver, settlement=expected,
                  paired_may_gain=paired_gain_interval(residual[prior_pair], residual[selected], present.q))
    pd.DataFrame(dict(site_id=sites.site_id, group=np.where(first, "G1", "G2"))).to_csv(
        destination/"composition.csv", index=False)
    pd.DataFrame(dict(group=np.repeat(["G1", "G2"], len(times)), ts=np.tile(times.astype(str), 2),
                      forecast_kwh=prediction.T.reshape(-1))).to_csv(destination/"forecast.csv", index=False)
    (destination/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(deployed=deployed, may=audit, settlement=expected), indent=2), flush=True)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", default=4, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = cli()
    destination = safe_output(args.raw, args.out)
    if destination == args.artifacts.resolve():
        raise ValueError("Use a separate directory for the strengthened run")
    fit_expanding(args.raw, args.artifacts, destination, args.threads)
    compare_pairs(args.raw, args.artifacts, destination)
