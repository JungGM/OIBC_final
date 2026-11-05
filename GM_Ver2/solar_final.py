"""Select the SolarJoint candidate and verify final settlement against shared scenarios."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .data import prices
from .decision import aggregate, apply_blend, blend_weights, vpp
from .fit import FOLDS, safe_output
from .run import Panel, load_predictions, score
from .target67 import load_gru, read_panel, scenarios


METHODS = ("incumbent", "solar_joint", "global_mix", "peak_mix", "fleet_residual")


def levels(panel, base):
    ver2 = apply_blend(panel.x[..., :-2], panel.regime, base)
    return np.stack([ver2, panel.x[..., -2], panel.x[..., -1]], axis=-1)


def correction_features(panel, features, prediction, capacity):
    summaries = features.sum(axis=1)/capacity
    pred = prediction.sum(axis=1)/capacity
    hour = panel.times.hour.to_numpy()*np.pi/12
    return np.column_stack([summaries, pred, summaries[:, 0]-summaries[:, 1],
                            summaries[:, 0]-summaries[:, 2], np.sin(hour), np.cos(hour)])


def fit_predict(past, present, recipe, reference, capacity):
    w0 = blend_weights(past.x[..., :-2], past.y, past.q, past.regime, reference, False)
    before, after = levels(past, w0), levels(present, w0)
    if recipe in ("incumbent", "refined_vpp"):
        original = blend_weights(before[..., :2], past.y, past.q, past.regime, reference, False)
        w = np.c_[original, np.zeros(len(original))]
    elif recipe == "solar_joint":
        w = np.array([[0., 0., 1.]])
    else:
        w = blend_weights(before, past.y, past.q, past.regime, reference, recipe == "peak_mix")
    fitted, prediction = apply_blend(before, past.regime, w), apply_blend(after, present.regime, w)
    correction = None
    if recipe == "fleet_residual":
        total_cap = capacity.sum()
        use = (past.q > 0) & (fitted.sum(axis=1) >= .04*total_cap)
        train_x = correction_features(past, before, fitted, total_cap)
        test_x = correction_features(present, after, prediction, total_cap)
        target = (past.y.sum(axis=1)-fitted.sum(axis=1))/total_cap
        data = xgb.DMatrix(train_x[use], label=target[use],
                           weight=np.clip(past.q[use]/past.q[use].mean(), .1, 5.))
        correction = xgb.train(dict(objective="reg:absoluteerror", tree_method="hist", max_depth=2,
                                    min_child_weight=35, eta=.05, reg_lambda=12, nthread=4,
                                    seed=20261108), data, num_boost_round=100)
        for forecasts, inputs in [(fitted, train_x), (prediction, test_x)]:
            total = forecasts.sum(axis=1)/total_cap
            delta = correction.inplace_predict(inputs)
            ratio = .5*delta/np.maximum(total, .04)
            scale = 1 + np.where(total >= .04, ratio.clip(-.12, .12), 0.)
            forecasts *= scale[:, None]
    return fitted, prediction, dict(weights=w.tolist(), ver2_weights=w0.tolist()), correction


def attach(panel, matrix):
    return Panel(panel.times, np.concatenate([panel.x, matrix[..., None]], -1), panel.y, panel.q)


def refine_membership(errors, q, capacity, month, initial, rounds=4):
    """Best feasible one-site move or two-site swap under the robust objective."""
    low = 20000.
    high = capacity.sum()-low
    bits = initial.copy()
    months = [month == value for value in np.unique(month)]
    total = errors.sum(axis=1)
    q = .1*q[:, None]
    # A sentinel column represents the unused side of a one-site move.
    extended = np.column_stack([errors, np.zeros(len(errors))])
    cap = np.r_[capacity, 0.]
    sentinel = len(capacity)

    def robust(first):
        losses = q*(np.abs(first)+np.abs(total[:, None]-first))
        worst = np.maximum.reduce([losses[mask].mean(axis=0) for mask in months])
        return losses.mean(axis=0) + .2*worst

    starting_value = float(robust((errors @ bits)[:, None])[0])
    accepted = 0
    for _ in range(rounds):
        first = (errors @ bits)[:, None]
        best = float(robust(first)[0])
        left, right = np.meshgrid(np.r_[np.flatnonzero(bits), sentinel],
                                  np.r_[np.flatnonzero(~bits), sentinel])
        left, right = left.ravel(), right.ravel()
        mass = capacity @ bits - cap[left] + cap[right]
        valid = (mass >= low) & (mass <= high)
        left, right = left[valid], right[valid]
        move = None
        for j in range(0, len(left), 192):
            a, b = left[j:j+192], right[j:j+192]
            objective = robust(first + extended[:, b] - extended[:, a])
            k = int(objective.argmin())
            if objective[k] < best-1e-8:
                move, best = (a[k], b[k]), float(objective[k])
        if move is None:
            break
        a, b = move
        if a != sentinel:
            bits[a] = False
        if b != sentinel:
            bits[b] = True
        accepted += 1
    final_value = float(robust((errors @ bits)[:, None])[0])
    return bits, dict(accepted_moves=accepted, initial=starting_value, final=final_value)


def allocate(fitted, past, capacity, reference, recipe):
    error = fitted-past.y
    first, solver = vpp(error, past.q, capacity, reference, past.months)
    if recipe == "refined_vpp":
        first, solver["refinement"] = refine_membership(error, past.q, capacity, past.months, first)
    return first, solver


def run(raw, artifacts, gru, joint, out, refine_only=False):
    roster = pd.read_csv(raw/"sites.csv").sort_values("site_id")
    capacity = roster.capacity_kw.to_numpy()
    reference = roster.current_group.eq("A").to_numpy()
    # The same target mask is used for both independent training families.
    panel = read_panel(raw, artifacts, gru, roster.site_id)
    y = np.load(artifacts/"train_y.npy", mmap_mode="r")
    matrix = np.concatenate([np.load(joint/f"{fold}.npy").reshape(-1, len(roster)) for fold in FOLDS[:-1]])
    actual_tail = y.reshape(-1, len(roster))[-len(matrix):]
    panel = attach(panel, matrix[np.isfinite(actual_tail).all(axis=1)])
    day = (panel.times-pd.Timedelta(hours=1)).normalize()
    methods = ("incumbent", "refined_vpp") if refine_only else METHODS
    records = {}
    for month in (3, 4):
        start = pd.Timestamp(2025, month, 1, tz=day.tz)
        past = panel.select(panel.times <= start-pd.Timedelta(hours=7))
        held = panel.select((day >= start) & (day < start+pd.offsets.MonthBegin()))
        records[str(month)] = {}
        for recipe in methods:
            fitted, forecast, _, _ = fit_predict(past, held, recipe, reference, capacity)
            first, _ = allocate(fitted, past, capacity, reference, recipe)
            records[str(month)][recipe] = score(held, forecast, first)
        print(f"Selection month {month}: {[(name, round(s['penalty_krw'])) for name,s in records[str(month)].items()]}",
              flush=True)

    chosen = min(methods, key=lambda name: np.mean([
        values[name]["penalty_krw"]/values["incumbent"]["penalty_krw"] for values in records.values()]))
    print(f"Selected before May: {chosen}", flush=True)
    may = pd.Timestamp("2025-05-01", tz=day.tz)
    past = panel.select(panel.times <= may-pd.Timedelta(hours=7))
    held = panel.select(day >= may)
    diagnostics = {}
    history_forecasts = {}
    diagnostic_recipes = ("incumbent", "refined_vpp") if refine_only else ("incumbent", chosen, "solar_joint")
    for recipe in dict.fromkeys(diagnostic_recipes):
        fitted, forecast, _, _ = fit_predict(past, held, recipe, reference, capacity)
        first, _ = allocate(fitted, past, capacity, reference, recipe)
        history_forecasts[recipe] = forecast
        diagnostics[recipe] = score(held, forecast, first)
    promoted = (chosen if diagnostics[chosen]["penalty_krw"] <
                diagnostics["incumbent"]["penalty_krw"] else "incumbent")
    eval_price = prices(raw/"smp.csv")
    cutoff = eval_price.index[0].normalize()-pd.Timedelta(hours=7)
    prior = panel.select(panel.times <= cutoff)
    gru_eval = load_gru(gru, "final", eval_price.index, roster.site_id, cutoff)
    joint_eval = np.load(joint/"evaluation.npy").reshape(-1, len(roster))
    forecast_x = np.concatenate([load_predictions(artifacts, "evaluation"),
                                 gru_eval[..., None], joint_eval[..., None]], -1)
    eval_panel = Panel(eval_price.index, forecast_x, np.empty((len(eval_price), 0)), np.zeros(len(eval_price)))
    comparison = {}
    results = {}
    for recipe in dict.fromkeys(("incumbent", promoted)):
        fitted, prediction, parameters, correction = fit_predict(prior, eval_panel, recipe, reference, capacity)
        first, solver = allocate(fitted, prior, capacity, reference, recipe)
        comparison[recipe] = (prediction, first)
        results[recipe] = dict(parameters=parameters, solver=solver)
        if correction is not None:
            correction.save_model(out/f"{recipe}.ubj")
    anchors_eval = {"ver2": comparison["incumbent"][0], "gru_v1": gru_eval}
    anchors_may = {"ver2": history_forecasts["incumbent"], "gru_v1": held.x[..., -2]}
    annual = scenarios(eval_price, anchors_eval, anchors_may, held.y, comparison)
    pred, first = comparison[promoted]
    group_forecast = aggregate(pred, first)
    capacity_kw = [float(capacity[first].sum()), float(capacity[~first].sum())]
    if (group_forecast.shape != (8760, 2) or min(capacity_kw) < 20000
            or not np.isfinite(group_forecast).all() or (group_forecast < 0).any()):
        raise ValueError("Invalid submission")
    pd.DataFrame(dict(site_id=roster.site_id, group=np.where(first, "G1", "G2"))).to_csv(
        out/"composition.csv", index=False)
    pd.DataFrame(dict(group=np.repeat(["G1", "G2"], len(eval_price)),
                      ts=np.tile(eval_price.index.astype(str), 2),
                      forecast_kwh=group_forecast.T.reshape(-1))).to_csv(out/"forecast.csv", index=False)
    report = dict(selected=chosen, promoted=promoted, capacity_kw=capacity_kw, selection=records,
                  may=diagnostics, final_models=results, scenario_anchors={"ver2": "prior 50:50 blend",
                  "gru_v1": "legacy GRU"}, scenarios=annual)
    (out/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(promoted=promoted, may=diagnostics, scenarios=annual), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "artifacts", "gru", "joint", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--refine-only", action="store_true")
    args = parser.parse_args()
    run(args.raw, args.artifacts, args.gru, args.joint, safe_output(args.raw, args.out), args.refine_only)
