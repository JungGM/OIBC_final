"""Rolling OOF experiment and the earlier February–April comparison."""

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .decisions import ENSEMBLES, allocation_candidates, ensemble_fit, group_sum, hourly_calibration
from .fit import FOLDS, MODELS, SEED, dump_json, input_files, start_at
from .settlement import group_capacities, group_settlement, issue_cutoff, price_arrays, score_arrays


CONTEXT = ("solar", "hour_sin", "hour_cos", "season_sin", "season_cos", "wa_cloud", "wb_cloud",
           "wb_uv_idx", "wa_cloud_revision", "wb_cloud_revision", "cloud_difference", "wa_temp", "wb_temp")
SELECTION = ("feb", "mar", "apr")


def mix_features(context, capacity, first):
    blocks = []
    for group_id, mask in enumerate((first, ~first)):
        selected = context[:, mask]
        valid = np.isfinite(selected)
        weights = capacity[mask][None, :, None]
        totals = (valid * weights).sum(axis=1)
        numerator = (np.where(valid, selected, 0) * weights).sum(axis=1)
        features = np.divide(numerator, totals, out=np.full_like(totals, np.nan), where=totals > 0)
        blocks.append(np.column_stack([features, np.full(len(features), group_id),
                                       np.full(len(features), capacity[mask].sum())]))
    return np.stack(blocks, axis=1).reshape(2 * len(context), -1)


def direct_forecast(x, y, q, future, first, capacity, threads):
    valid = (np.isfinite(y) & (y >= 0)).all(axis=1)
    totals = np.array([capacity[first].sum(), capacity[~first].sum()])
    train = mix_features(x[valid], capacity, first)
    features = mix_features(future, capacity, first)
    target = (group_sum(y[valid], first) / totals).ravel()
    params = dict(objective="regression_l1", n_estimators=145, learning_rate=.055, num_leaves=15,
                  min_child_samples=55, n_jobs=threads, random_state=SEED, verbosity=-1)
    plain = lgb.LGBMRegressor(**params).fit(train, target)
    weighted = lgb.LGBMRegressor(**params).fit(train, target, sample_weight=np.repeat(q[valid], 2))
    forecasts = [model.booster_.predict(features, num_threads=threads).reshape(-1, 2).clip(0) * totals
                 for model in (plain, weighted)]
    return forecasts, {"direct_mae": plain, "direct_weighted": weighted}


def as_composition(sites, mask):
    return pd.DataFrame({"site_id": sites.site_id, "group": np.where(mask, "G1", "G2")})


def portfolio_choices(site_prediction, past_prediction, history_y, history_q, history_hours,
                      direct_inputs, future_context, future_hours, sites, reference, threads, periods=None):
    capacity = sites.capacity_kw.to_numpy()
    history_error = past_prediction - history_y
    groups = allocation_candidates(history_error, history_q, capacity, sites.lon.to_numpy(), reference, periods)
    choices = {name: (group_sum(site_prediction, first), first) for name, first in groups.items()}
    if periods is not None:
        choices["reference"] = group_sum(site_prediction, reference), reference
    first = groups["ga_local"]
    bottom_up = choices["ga_local"][0]
    (plain, weighted), direct_models = direct_forecast(*direct_inputs, future_context, first=first,
                                                       capacity=capacity, threads=threads)
    choices["direct_weighted"] = weighted, first
    choices["direct_mix"] = .75 * bottom_up + .25 * plain, first
    choices["direct_weighted_mix"] = .75 * bottom_up + .25 * weighted, first
    corrections, offsets = hourly_calibration(group_sum(history_y, first), group_sum(past_prediction, first),
                                              history_q, history_hours, future_hours)
    choices["economic"] = np.maximum(0, bottom_up + corrections), first
    return choices, {**direct_models, "economic_offsets_kwh": offsets}


class Experiment:
    def __init__(self, raw, out, threads):
        self.out = out
        self.archive = out
        self.folds = FOLDS
        self.experts = MODELS
        self.robust_allocation = False
        self.threads = threads
        self.sites, self.weights, self.evaluation = input_files(raw)
        self.capacity = self.sites.capacity_kw.to_numpy(dtype=np.float32)
        self.reference = self.sites.current_group.eq("A").to_numpy()
        self.reference_composition = as_composition(self.sites, self.reference)
        group_capacities(self.sites, self.reference_composition)
        self.times, self.smp, self.k = price_arrays(self.weights)
        self.y = np.load(out / "train_y.npy", mmap_mode="r")
        self.x = np.load(out / "train_x.npy", mmap_mode="r")
        self.feature_names = json.loads((out / "features.json").read_text(encoding="utf-8"))
        self.context = [self.feature_names.index(name) for name in CONTEXT]

    def slice(self, fold):
        order = list(self.folds)
        first = self.times.searchsorted(start_at(self.folds[fold]))
        stop = self.times.searchsorted(start_at(self.folds[order[order.index(fold) + 1]]))
        return slice(first, stop)

    def forecasts(self, fold, times):
        directory = self.out / fold
        metadata = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        if (pd.Timestamp(metadata["last_label_time"]) > issue_cutoff([times[0]])[0]
                or pd.Timestamp(metadata["first_target"]) != times[0]
                or pd.Timestamp(metadata["last_target"]) != times[-1]):
            raise ValueError(f"Forecast provenance mismatch for {fold}")
        forecasts = {name: np.load(directory / f"{name}.npy") for name in MODELS}
        if any(matrix.shape != (len(times), len(self.sites)) or not np.isfinite(matrix).all()
               or (matrix < 0).any() for matrix in forecasts.values()):
            raise ValueError(f"Invalid site forecast for {fold}")
        return forecasts

    def history(self, fold):
        left = self.slice("warmup").start
        stop = self.times.searchsorted(issue_cutoff([start_at(self.folds[fold])])[0], side="right")
        region = slice(left, stop)
        y = self.y[region]
        complete = (np.isfinite(y) & (y >= 0)).all(axis=1)
        prior = list(self.folds)[:list(self.folds).index(fold)]
        archives = []
        for name in prior:
            with np.load(self.archive / name / "oof_recipes.npz") as archive:
                archives.append({recipe: archive[recipe] for recipe in (*MODELS, *ENSEMBLES)})
        recipes = {recipe: np.concatenate([archive[recipe] for archive in archives])[:len(y)][complete]
                   for recipe in archives[0]}
        q = (self.smp[region] * self.k[region])[complete]
        return recipes, y[complete], q, self.times[region][complete], self.x[region][complete][:, :, self.context]

    def warmup(self):
        part = self.slice("warmup")
        predictions = self.forecasts("warmup", self.times[part])
        average = (predictions["lgbm_full"] + predictions["xgb"] + predictions["catboost"]) / 3
        predictions.update({name: average for name in ENSEMBLES})
        folder = self.archive / "warmup"
        folder.mkdir(exist_ok=True)
        np.savez(folder / "oof_recipes.npz", **predictions)
        return self.site_scores(part, predictions)

    def site_scores(self, part, predictions):
        return {f"site_{name}": score_arrays(self.sites, self.reference_composition, self.y[part], prediction,
                                             self.weights.iloc[part]) for name, prediction in predictions.items()}

    def fit_decisions(self, fold, portfolio_site, future, times, target_context):
        history, y, q, prior_times, prior_context = self.history(fold)
        meta, meta_models = ensemble_fit({name: history[name] for name in self.experts}, y, q, future,
                                        prior_context, target_context, self.reference, self.capacity, self.threads)
        future.update(meta)
        cutoff = issue_cutoff([times[0]])[0]
        stop = self.times.searchsorted(cutoff, side="right")
        # The direct group fit only consumes complete training hours prior to the issue cutoff.
        start = self.slice("warmup").start
        direct_inputs = (self.x[start:stop][:, :, self.context], self.y[start:stop])
        periods = (prior_times - pd.Timedelta(hours=1)).strftime("%Y-%m").to_numpy() if self.robust_allocation else None
        choices, direct_models = portfolio_choices(future[portfolio_site], history[portfolio_site], y, q,
                                                   prior_times.hour.to_numpy(),
                                                   (*direct_inputs, (self.smp * self.k)[start:stop]), target_context,
                                                   times.hour.to_numpy(), self.sites, self.reference, self.threads, periods)
        folder = self.archive / fold
        folder.mkdir(exist_ok=True)
        joblib.dump({**meta_models, **direct_models}, folder / "decision_models.joblib", compress=3)
        choices.update({f"site_{name}": (group_sum(prediction, self.reference), self.reference)
                        for name, prediction in future.items()})
        return choices

    def run_fold(self, fold, portfolio_site):
        part = self.slice(fold)
        times = self.times[part]
        target_context = self.x[part][:, :, self.context]
        future = self.forecasts(fold, times)
        choices = self.fit_decisions(fold, portfolio_site, future, times, target_context)
        folder = self.archive / fold
        np.savez(folder / "oof_recipes.npz", **future)
        result = self.site_scores(part, future)
        truth = self.y[part]
        valid = (np.isfinite(truth) & (truth >= 0)).all(axis=1)
        for name, (prediction, first) in choices.items():
            if name in result:
                continue
            result[name] = {**group_settlement(group_sum(truth[valid], first), prediction[valid],
                                                self.smp[part][valid], self.k[part][valid]),
                            "hours_scored": int(valid.sum()),
                            "group_capacity_kw": group_capacities(self.sites, as_composition(self.sites, first)).to_dict()}
        np.savez(folder / "groups.npz", **{name: first for name, (_, first) in choices.items()})
        np.savez(folder / "group_forecasts.npz", **{name: pred for name, (pred, _) in choices.items()})
        print(f"{fold}: {int(valid.sum())} complete hours; portfolio site recipe = {portfolio_site}", flush=True)
        return result


def rank(scores, names):
    return sorted(names, key=lambda name: (sum(fold[name]["penalty_krw"] for fold in scores), name))


def compare_previous_protocol(experiment):
    out = experiment.out
    plan_path = out / "selection.json"
    if plan_path.exists():
        raise FileExistsError(plan_path)
    previous = [experiment.warmup()]
    scores = {}
    site_names = [f"site_{name}" for name in (*MODELS, *ENSEMBLES)]
    for fold in SELECTION:
        best_site = rank(previous, site_names)[0].removeprefix("site_")
        scores[fold] = experiment.run_fold(fold, best_site)
        previous.append(scores[fold])
    ordering = rank(scores.values(), scores[SELECTION[0]])
    plan = {"selected": ordering[0], "ranking": ordering,
            "portfolio_site": rank(scores.values(), site_names)[0].removeprefix("site_"),
            "folds": scores}
    dump_json(plan_path, plan)
    print(f"Previous protocol selected {plan['selected']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce the earlier February–April development comparison.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stage", required=True, choices=("select",))
    args = parser.parse_args()
    raw, out = args.raw.resolve(), args.out.resolve()
    if out.is_relative_to(raw) or out.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Place all outputs outside the repository and participant package")
    with threadpool_limits(limits=args.threads):
        compare_previous_protocol(Experiment(raw, out, args.threads))
