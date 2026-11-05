"""Train, backtest, choose the forecast and VPP, and export local settlement scenarios."""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import prices
from .decision import aggregate, apply_blend, blend_weights, ga_baseline, penalty, vpp
from .fit import FOLDS, ORIGINS, cli, fit, safe_output
from .models import NAMES


METHODS = [*NAMES, "uniform", "global_l1", "peak_l1", "peak_l1_bias"]


@dataclass
class Panel:
    times: pd.DatetimeIndex
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray

    @property
    def regime(self):
        return self.times.hour.isin(range(11, 16)).astype(int)

    @property
    def months(self):
        operating = self.times-pd.Timedelta(hours=1)
        return operating.year*100 + operating.month

    def select(self, mask):
        return Panel(self.times[mask], self.x[mask], self.y[mask], self.q[mask])


def load_predictions(out, fold):
    return np.stack([np.load(out/fold/f"{name}.npy").reshape(-1, 253) for name in NAMES],
                    axis=-1).astype(np.float64)


def get_weights(panel, name, first):
    n = len(NAMES)
    if name in NAMES:
        return np.eye(n)[[NAMES.index(name)]]
    if name == "uniform":
        return np.full((1, n), 1/n)
    return blend_weights(panel.x, panel.y, panel.q, panel.regime, first,
                         conditional=name.startswith("peak"))


def calibrate(panel, weights):
    estimate = apply_blend(panel.x, panel.regime, weights).sum(axis=1)
    actual = panel.y.sum(axis=1)
    result = np.ones(2)
    for region in range(2):
        good = (panel.regime == region) & (estimate > 1) & (panel.q > 0)
        if good.any():
            ratio = actual[good]/estimate[good]
            weight = panel.q[good]*estimate[good]
            order = ratio.argsort()
            median = ratio[order][np.searchsorted(np.cumsum(weight[order]), weight.sum()/2)]
            result[region] = 1 + .5*(np.clip(median, .85, 1.15)-1)
    return result


def apply_method(panel, weights, scale):
    return apply_blend(panel.x, panel.regime, weights)*scale[panel.regime, None]


def learn(panel, name, reference):
    weights = get_weights(panel, name, reference)
    scale = calibrate(panel, weights) if name.endswith("_bias") else np.ones(2)
    return weights, scale


def score(panel, estimate, first):
    err = estimate-panel.y
    loss = penalty(err, panel.q, first)
    return dict(hours=len(loss), penalty_krw=float(loss.sum()),
                fleet_floor_krw=float(.1*np.dot(panel.q, np.abs(err.sum(axis=1)))),
                peak_penalty_krw=float(loss[panel.regime == 1].sum()))


def settlement(estimate, price, holdout_times, errors, draws=1000):
    # Resample joint G1/G2 signed errors in consecutive 3-day blocks, keeping
    # intraday profiles and dependence between groups. Forecast is held fixed.
    op_day = (holdout_times-pd.Timedelta(hours=1)).normalize()
    days, counts = np.unique(op_day.asi8, return_counts=True)
    by_day = [errors[op_day.asi8 == day] for day, count in zip(days, counts) if count == 24]
    complete = days[counts == 24]
    blocks = [np.stack(by_day[i:i+3]) for i in range(len(by_day)-2)
              if complete[i+2]-complete[i] == 2*86400*10**9]
    if not blocks:
        raise ValueError("Need three consecutive fully observed holdout days")
    blocks = np.stack(blocks)
    rng = np.random.default_rng(20261105)
    n_days = len(estimate)//24
    smp = price.smp.to_numpy()[:, None]
    q = (price.smp*price.k).to_numpy()[:, None]
    simulations = []
    for _ in range(draws):
        sampled = blocks[rng.integers(len(blocks), size=(n_days+2)//3)].reshape(-1, 24, 2)
        actual = np.maximum(estimate-sampled[:n_days].reshape(-1, 2), 0)
        gross = float((smp*actual).sum())
        fine = float(.1*(q*np.abs(estimate-actual)).sum())
        simulations.append([gross, fine, gross-fine])
    gross, fine, net = np.mean(simulations, axis=0)
    return dict(scenario="May held-out joint residuals transfer to evaluation year; three-day blocks",
                draws=draws, complete_holdout_days=len(by_day), available_blocks=len(blocks),
                forecast_implied_gross_krw=float((smp*estimate).sum()), mean_gross_krw=gross,
                mean_penalty_krw=fine, mean_settlement_krw=net,
                p10_settlement_krw=float(np.percentile(simulations, 10, axis=0)[2]),
                p90_settlement_krw=float(np.percentile(simulations, 90, axis=0)[2]))


def report(raw, out):
    sites = pd.read_csv(raw/"sites.csv").sort_values("site_id")
    capacity = sites.capacity_kw.to_numpy()
    reference = sites.current_group.eq("A").to_numpy()
    price = prices(raw/"train/smp.csv")
    evaluation_price = prices(raw/"smp.csv")
    start = pd.Timestamp(ORIGINS[0], tz=price.index.tz)
    index = price.index.searchsorted(start + pd.Timedelta(hours=1))
    y = np.load(out/"train_y.npy", mmap_mode="r").reshape(-1, len(sites))[index:]
    times = price.index[index:]
    x = np.concatenate([load_predictions(out, fold) for fold in FOLDS[:-1]])
    complete = np.isfinite(y).all(axis=1)
    panel = Panel(times, x, y, (price.smp*price.k).to_numpy()[index:]).select(complete)
    results = {}
    winner = None
    heldout = None
    for origin in ORIGINS[1:-1]:
        t = pd.Timestamp(origin, tz=times.tz)
        past = panel.select(panel.times <= t-pd.Timedelta(hours=7))
        present = panel.select(((panel.times-pd.Timedelta(hours=1)).normalize() >= t)
                               & ((panel.times-pd.Timedelta(hours=1)).normalize() < t+pd.offsets.MonthBegin()))
        if origin == "2025-05-01":
            # Monthly loss ratios give March and April equal influence.
            winner = min(METHODS, key=lambda name: np.mean([
                row["candidates"][name]["penalty_krw"] /
                row["candidates"]["tree_history"]["penalty_krw"] for row in results.values()]))
        candidates = {}
        for name in METHODS:
            w, scale = learn(past, name, reference)
            pred = apply_method(present, w, scale)
            candidates[name] = score(present, pred, reference)
        row = dict(candidates=candidates)
        if winner is not None:
            w, scale = learn(past, winner, reference)
            err = apply_method(past, w, scale)-past.y
            choice, solver = vpp(err, past.q, capacity, reference, past.months)
            estimate = apply_method(present, w, scale)
            baseline = ga_baseline(past.x[..., 0]-past.y, past.q, capacity, reference)
            historical_mask = present.times <= t+pd.offsets.MonthBegin()-pd.Timedelta(hours=7)
            row.update(selected=winner, selected_reference=score(present, estimate, reference),
                       selected_vpp=score(present, estimate, choice), solver=solver,
                       tree_plus_ga=score(present, present.x[..., 0], baseline),
                       selected_vpp_historical_window=score(present.select(historical_mask),
                                                           estimate[historical_mask], choice))
            heldout = (present.times, aggregate(estimate-present.y, choice))
        results[origin] = row
        print(f"{origin}: {len(present.times)} complete hours, "
              f"lowest reference penalty {min(candidates, key=lambda n: candidates[n]['penalty_krw'])}", flush=True)
    cutoff = evaluation_price.index[0].normalize()-pd.Timedelta(hours=7)
    history = panel.select(panel.times <= cutoff)
    w, scale = learn(history, winner, reference)
    choice, solver = vpp(apply_method(history, w, scale)-history.y, history.q, capacity,
                         reference, history.months)
    forecast_x = load_predictions(out, "evaluation")
    times = evaluation_price.index
    eval_panel = Panel(times, forecast_x, np.empty((len(times), 0)), np.zeros(len(times)))
    forecast = aggregate(apply_method(eval_panel, w, scale), choice)
    assignment = pd.DataFrame(dict(site_id=sites.site_id, group=np.where(choice, "G1", "G2")))
    delivery = pd.DataFrame(dict(group=np.repeat(["G1", "G2"], len(times)),
                                 ts=np.tile(times.astype(str), 2), forecast_kwh=forecast.T.reshape(-1)))
    cap = [float(capacity[choice].sum()), float(capacity[~choice].sum())]
    if (len(delivery) != 17520 or delivery.duplicated(["group", "ts"]).any()
            or not np.isfinite(forecast).all() or (forecast < 0).any() or min(cap) < 20000):
        raise ValueError("Submission checks failed")
    expected = settlement(forecast, evaluation_price, *heldout)
    result = dict(winner=winner, expert_order=NAMES, weights=w.tolist(), peak_scale=scale.tolist(),
                  final_cutoff=str(cutoff), vpp_capacity_kw=cap, vpp_solver=solver, backtest=results,
                  settlement=expected)
    assignment.to_csv(out/"composition.csv", index=False)
    delivery.to_csv(out/"forecast.csv", index=False)
    (out/"report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"winner": winner, "settlement": expected}, indent=2), flush=True)


if __name__ == "__main__":
    args = cli()
    destination = safe_output(args.raw, args.out)
    fit(args.raw, destination, args.tree_rows, args.epochs, args.threads)
    report(args.raw, destination)
