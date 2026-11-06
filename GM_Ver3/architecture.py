"""Run the full forecast, nested stack, portfolio and submission comparison."""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .decisions import allocation_candidates, group_sum, hourly_calibration
from .dynamic import STACKS, fit_stack
from .experts import FAMILIES, Store, TREE_TRIALS, observed_metrics
from .fit import dump_json, start_at
from .nested import NESTED_FOLDS
from .run import CONTEXT, as_composition, direct_forecast, mix_features
from .settlement import group_settlement, issue_cutoff, price_arrays
from .submission import validate_files


EXPERTS = ("baseline", *FAMILIES)
MONTHS = ("jan", "feb", "mar", "apr", "may")
LAYERS = ("mean", "weighted", "ridge", "lgb_stack", "xgb_stack", "mlp_gate", "economic_gate")
PORTFOLIOS = ("reference", "balanced", "geographic", "correlation", "ga", "ga_local")
SERVER_SCORE_USER_REPORTED = 6_435_780_471
REVIEW_TARGET = 6_465_000_000


def l1_coefficient(actual, bottom, direct, q):
    gap = direct-bottom
    good = np.abs(gap).ravel() > 1e-8
    if not good.any():
        return 0.
    ratios = ((actual-bottom).ravel()[good]/gap.ravel()[good])
    weights = np.repeat(q, 2)[good]*np.abs(gap.ravel()[good])
    order = np.argsort(ratios)
    cumulative = weights[order].cumsum()
    return float(np.clip(ratios[order][np.searchsorted(cumulative, .5*cumulative[-1])], 0., 1.))


def economic_tree(history_context, future_context, truth, hist, fut, membership, cap, q, threads):
    capacity = np.array([cap[membership].sum(), cap[~membership].sum()])
    history_x = np.column_stack([mix_features(history_context, cap, membership), (hist/capacity).ravel()])
    test_x = np.column_stack([mix_features(future_context, cap, membership), (fut/capacity).ravel()])
    target = ((truth-hist)/capacity).ravel()
    weights = np.repeat(q/max(q.mean(), 1.), 2).clip(.05, 5.)
    model = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=110, max_depth=2,
                             min_child_weight=60, learning_rate=.05, reg_lambda=12,
                             n_jobs=threads, random_state=20261106)
    model.fit(history_x, target, sample_weight=weights)
    delta = model.predict(test_x).reshape(-1, 2).clip(-.045, .045)*capacity
    return np.maximum(0., fut + .55*delta), model


class Experiment:
    def __init__(self, raw, base, trained, out, threads, wait_experts=False):
        self.store = Store(raw, base, threads)
        self.raw, self.base, self.trained, self.out = raw, base, trained, out
        self.threads = threads
        self.cap = self.store.capacity
        self.reference = self.store.sites.current_group.eq("A").to_numpy()
        self.context_ids = [self.store.index[name] for name in CONTEXT+("lat", "lon", "capacity_kw")]
        self.times, self.smp, self.k = self.store.times, self.store.smp, self.store.k
        self.left = self.times.searchsorted(start_at(NESTED_FOLDS["warmup"]))
        self.panel = {}
        self.wait_experts = wait_experts
        self.full_experts = None

    def bounds(self, fold):
        names = list(NESTED_FOLDS)
        start = start_at(NESTED_FOLDS[fold])
        first = self.times.searchsorted(start)
        end = (len(self.times) if fold == "evaluation" else self.times.searchsorted(
               start_at(NESTED_FOLDS[names[names.index(fold)+1]])))
        return start, slice(first, end)

    def baseline_eval(self):
        file = self.out/"baseline_evaluation.npy"
        if file.exists():
            return np.load(file)
        start, _ = self.bounds("evaluation")
        stop = self.times.searchsorted(issue_cutoff([start])[0], side="right")
        baseline = dict(TREE_TRIALS[0], depth=5, rounds=210, eta=.065, leaves=48, min_leaf=200,
                        colsample=.90)
        point, fit = self.store.fit("lightgbm", baseline, stop, self.store.eval_x, len(self.times),
                                    self.store.y.size, self.out/"baseline_evaluation.joblib")
        np.save(file, point)
        dump_json(self.out/"baseline_evaluation.json", dict(cutoff=str(self.times[stop-1]), **fit))
        return point

    def level(self, fold):
        start, part = self.bounds(fold)
        folder = self.trained/fold
        paths = [folder/f"{name}.npy" for name in FAMILIES]
        if self.wait_experts and not all(path.exists() for path in paths):
            print(f"Waiting for expert checkpoints: {fold}", flush=True)
            deadline = time.monotonic()+35*60
            while not all(path.exists() for path in paths) and time.monotonic() < deadline:
                time.sleep(3)
        if fold == "evaluation":
            times = self.store.eval_prices.index
            inputs = self.store.eval_x
            old = self.baseline_eval()
        else:
            times, inputs = self.times[part], self.store.x[part]
            source = "warmup" if fold == "jan" else fold
            old = np.load(self.base/source/"lgbm_full.npy", mmap_mode="r")
            if fold in ("warmup", "jan"):
                offset = part.start-self.left
                old = old[offset:offset+len(times)]
        models = [old]
        for name in FAMILIES:
            folder = self.trained/fold
            if fold == "evaluation" and self.full_experts is not None:
                replacement = self.full_experts/fold
                if (replacement/f"{name}.npy").exists():
                    folder = replacement
            meta = json.loads((folder/f"{name}.json").read_text())
            if pd.Timestamp(meta["cutoff"]) > issue_cutoff([start])[0]:
                raise ValueError(f"Invalid cutoff for {fold}/{name}")
            models.append(np.load(folder/f"{name}.npy"))
        x = np.stack(models, axis=-1).astype("float32")
        if x.shape != (len(times), len(self.cap), len(EXPERTS)) or not np.isfinite(x).all() or (x < 0).any():
            raise ValueError(f"Invalid expert predictions for {fold}")
        return x, inputs[..., self.context_ids], times

    def history(self, fold):
        start, _ = self.bounds(fold)
        stop = self.times.searchsorted(issue_cutoff([start])[0], side="right")
        region = slice(self.left, stop)
        y = self.store.y[region]
        good = (np.isfinite(y) & (y >= 0)).all(axis=1)
        earlier = list(self.panel.values())
        model = np.concatenate([rec[0] for rec in earlier])[:len(y)][good]
        oof = {name: np.concatenate([rec[1][name] for rec in earlier])[:len(y)][good]
               for name in (*EXPERTS, *STACKS)}
        return model, oof, y[good], self.store.q[region][good], self.times[region][good], (
            self.store.x[region][good][..., self.context_ids])

    def decision(self, fold, candidates, future_context, times, history, final=False):
        _, oof, y, q, prior_times, ctx = history
        period = (prior_times-pd.Timedelta(hours=1)).strftime("%Y-%m").to_numpy()
        previous = oof["economic_gate"]
        groups = {"reference": self.reference, **allocation_candidates(
            previous-y, q, self.cap.astype("float64"), self.store.sites.lon.to_numpy(),
            self.reference, period)}
        recipes = {name: (group_sum(pred, self.reference), self.reference) for name, pred in candidates.items()}
        econ = candidates["economic_gate"]
        for name in PORTFOLIOS:
            recipes[f"vpp_{name}"] = group_sum(econ, groups[name]), groups[name]
        first = groups["ga_local"]
        historic = group_sum(previous, first)
        actual = group_sum(y, first)
        bottom = group_sum(econ, first)
        cut = min(len(y)-24, max(48, (int(.68*len(y))//24)*24))
        (_, weighted_inner), _ = direct_forecast(ctx[:cut], y[:cut], q[:cut], ctx[cut:],
                                                 first, self.cap, self.threads)
        beta = l1_coefficient(actual[cut:], historic[cut:], weighted_inner, q[cut:])
        (plain, weighted), direct_models = direct_forecast(ctx, y, q, future_context,
                                                           first, self.cap, self.threads)
        recipes["direct_mae"] = plain, first
        recipes["direct_weighted"] = weighted, first
        mixture = (1-beta)*bottom+beta*weighted
        recipes["direct_mix"] = mixture, first
        offsets, med = hourly_calibration(actual, historic, q, prior_times.hour.to_numpy(),
                                          times.hour.to_numpy())
        recipes["economic_median"] = np.maximum(0., mixture+offsets), first
        adjusted, residual_model = economic_tree(ctx, future_context, actual, historic,
                                                  mixture, first, self.cap, q, self.threads)
        recipes["economic_residual"] = adjusted, first
        params = dict(direct_weight=beta, economic_offsets=med.tolist())
        if final:
            joblib.dump(dict(models=direct_models, residual=residual_model, weight=beta,
                             offsets=med), self.out/"final_portfolio.joblib", compress=1)
        return recipes, params

    def group_score(self, fold, recipes, site_predictions):
        _, part = self.bounds(fold)
        actual = self.store.y[part]
        complete = (np.isfinite(actual) & (actual >= 0)).all(axis=1)
        scores = {}
        for name, (pred, group) in recipes.items():
            scores[name] = {**group_settlement(group_sum(actual[complete], group), pred[complete],
                                                self.smp[part][complete], self.k[part][complete]),
                            "complete_hours": int(complete.sum())}
            if name in site_predictions:
                scores[name].update(observed_metrics(actual, site_predictions[name], self.store.q[part], self.cap))
        return scores

    def fold(self, fold):
        path = self.out/fold
        path.mkdir(exist_ok=True)
        level, context, times = self.level(fold)
        if (path/"report.json").exists() and fold != "evaluation":
            with np.load(path/"oof.npz") as archive:
                self.panel[fold] = level, {name: archive[name] for name in archive.files}
            return json.loads((path/"report.json").read_text())
        experts = {name: level[..., i] for i, name in enumerate(EXPERTS)}
        if fold == "warmup":
            recipes = {**experts, **{name: level.mean(-1) for name in STACKS}}
            self.panel[fold] = level, recipes
            np.savez(path/"oof.npz", **recipes)
            dump_json(path/"report.json", {})
            return {}
        history = self.history(fold)
        h, _, y, q, _, ctx = history
        final = fold == "evaluation"
        stack, meta, models = fit_stack(h, y, q, level, ctx, context, self.reference, self.cap,
                                        self.threads, final)
        site = {**experts, **stack}
        recipes, decision = self.decision(fold, site, context, times, history, final)
        if final:
            joblib.dump(models, self.out/"final_stack.joblib", compress=1)
            return recipes, {"stack": meta, "portfolio": decision}, site
        self.panel[fold] = level, site
        results = self.group_score(fold, recipes, site)
        with np.errstate(invalid="ignore", divide="ignore"):
            correlated = np.corrcoef(np.stack([(group_sum(h[..., i]-y, self.reference)).ravel()
                                               for i in range(len(EXPERTS))]))
        record = {"scores": results, "stack": meta, "portfolio": decision,
                  "expert_group_error_correlation": np.nan_to_num(correlated).tolist()}
        np.savez(path/"oof.npz", **site)
        np.savez(path/"groups.npz", **{key: value[1] for key, value in recipes.items()})
        np.savez(path/"group_forecasts.npz", **{key: value[0] for key, value in recipes.items()})
        dump_json(path/"report.json", record)
        print(f"{fold}: lowest W {min(results, key=lambda k: results[k]['penalty_krw'])}", flush=True)
        return record

    def annual_scenario(self, final, site, selected):
        _, part = self.bounds("may")
        truth = self.store.y[part]
        if len(truth) % 24 or not (np.isfinite(truth) & (truth >= 0)).all():
            raise ValueError("Scenario needs complete May operating days")
        eval_index, smp, k = price_arrays(self.store.eval_prices)
        rng = np.random.default_rng(20261106)
        choices = rng.integers(len(truth)//24-2, size=(240, (len(eval_index)//24+2)//3))
        samples = (choices[..., None]*24+np.arange(72)).reshape(240, -1)[:, :len(eval_index)]
        chosen = {name: final[name] for name in dict.fromkeys(("baseline", selected))}
        runs = {}
        for anchor in ("baseline", "economic_gate"):
            errors = truth-self.panel["may"][1][anchor]
            gross, penalties = [], []
            membership = np.stack([g for _, a in chosen.values() for g in (a, ~a)], axis=-1).astype("float32")
            predictions = np.column_stack([p for p, _ in chosen.values()])
            for ids in samples:
                actual = np.maximum(site[anchor]+errors[ids], 0.)
                groups = actual @ membership
                gross.append(float(smp @ groups[:, :2].sum(-1)))
                fines = .1*(smp*k) @ np.abs(groups-predictions)
                penalties.append(fines.reshape(-1, 2).sum(-1))
            gross = np.asarray(gross)
            w = np.asarray(penalties)
            runs[anchor] = dict(draws=len(gross), mean_gross_krw=float(gross.mean()), candidates={
                name: dict(mean_penalty_krw=float(w[:, i].mean()),
                           mean_settlement_krw=float((gross-w[:, i]).mean()),
                           p10_krw=float(np.percentile(gross-w[:, i], 10)),
                           p90_krw=float(np.percentile(gross-w[:, i], 90)),
                           at_or_above_64_65=int(((gross-w[:, i]) >= REVIEW_TARGET).sum()))
                for i, name in enumerate(chosen)})
        return runs

    def run(self, development_only=False):
        records = {}
        for fold in ("warmup", *MONTHS[:-1]):
            record = self.fold(fold)
            if record:
                records[fold] = record
        names = records["jan"]["scores"]
        ranking = sorted(names, key=lambda name: sum(r["scores"][name]["penalty_krw"] for r in records.values()))
        choice = ranking[0]
        dump_json(self.out/"selection.json", dict(recipe=choice, ranking=ranking,
                  penalty_jan_apr_krw={name: sum(r["scores"][name]["penalty_krw"] for r in records.values())
                                       for name in ranking}))
        print(f"Selection before May scoring: {choice}", flush=True)
        if development_only:
            return
        records["may"] = self.fold("may")
        final, metadata, site = self.fold("evaluation")
        p, group = final[choice]
        folder = self.out/"submission"
        folder.mkdir(exist_ok=True)
        as_composition(self.store.sites, group).to_csv(folder/"composition.csv", index=False)
        times = self.store.eval_prices.index
        pd.DataFrame(dict(group=np.repeat(["G1", "G2"], len(times)),
                          ts=np.tile(times.astype(str), 2), forecast_kwh=p.T.reshape(-1))).to_csv(
                              folder/"forecast.csv", index=False)
        checks = validate_files(self.store.sites, times, folder)
        annual = self.annual_scenario(final, site, choice)
        report = dict(selection=choice, records=records, checks=checks, final=metadata, scenarios=annual,
                      expert_order=EXPERTS, review_target_krw=REVIEW_TARGET,
                      user_reported_official_baseline_krw=SERVER_SCORE_USER_REPORTED,
                      scenario_note="Shared simulated actuals, May three-day blocks; not official")
        dump_json(self.out/"report.json", report)
        print(json.dumps(dict(selected=choice, checks=checks, scenarios=annual), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "base", "trained", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--wait-experts", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.is_relative_to(Path(__file__).resolve().parents[1]) or out.is_relative_to(args.raw.resolve()) or out == args.base.resolve():
        raise ValueError("Use a separate local result directory")
    out.mkdir(parents=True, exist_ok=True)
    Experiment(args.raw, args.base, args.trained, out, args.threads, args.wait_experts).run(args.development_only)
