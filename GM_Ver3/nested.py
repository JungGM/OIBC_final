"""Run nested validation and freeze both the May and submission forecasts before scoring May."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .decisions import ENSEMBLES, group_sum
from .fit import FOLDS, dump_json, predict_frozen, start_at
from .run import Experiment, as_composition
from .settlement import GROUPS, group_capacities, group_settlement, issue_cutoff
from .submission import validate_files


MONTHS = ("jan", "feb", "mar", "apr")
ALLOCATION = ("reference", "balanced", "geographic", "correlation", "ga", "ga_local")
NESTED_FOLDS = {"warmup": FOLDS["warmup"], "jan": "2025-01-01", **{k: FOLDS[k] for k in list(FOLDS)[1:]}}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stable_rank(folds, candidates):
    def objective(name):
        ratios = [fold[name]["penalty_krw"] / max(1., fold["site_lgbm_full"]["penalty_krw"]) for fold in folds]
        return .75 * float(np.mean(ratios)) + .25 * max(ratios), name

    return sorted(candidates, key=objective)


def hourly_diagnostics(truth, forecast, site_prediction, first, smp, k, hours):
    group_actual = group_sum(truth, first)
    site_error = site_prediction.astype(np.float64) - truth
    gross = group_sum(np.abs(site_error), first)
    net = np.abs(group_sum(site_error, first))
    weights = .1 * smp * k
    return {**group_settlement(group_actual, forecast, smp, k),
            "hours_scored": len(truth),
            "penalty_by_closing_hour_krw": {str(hour): float(weights[hours == hour] @
                    np.abs(forecast[hours == hour] - group_actual[hours == hour]).sum(axis=1)) for hour in range(24)},
            "bottom_up_cancellation_krw_by_group": dict(zip(GROUPS, (weights @ (gross - net)).tolist()))}


class NestedExperiment(Experiment):
    def __init__(self, raw, out, threads):
        super().__init__(raw, out, threads)
        self.archive = out / "validation_nested"
        self.folds = NESTED_FOLDS
        self.robust_allocation = True

    def forecasts(self, fold, times):
        if fold not in ("warmup", "jan"):
            return super().forecasts(fold, times)
        # Reuse the existing Dec 24 fit as frozen out-of-time forecasts through January.
        original = self.times[self.slice("warmup").start:self.slice("jan").stop]
        stored = super().forecasts("warmup", original)
        positions = original.get_indexer(times)
        if (positions < 0).any():
            raise ValueError("January predictions are outside the original OOF block")
        return {name: forecast[positions] for name, forecast in stored.items()}

    def month_diagnostics(self, fold, choice, portfolio_site):
        part = self.slice(fold)
        actual = self.y[part]
        complete = (np.isfinite(actual) & (actual >= 0)).all(axis=1)
        directory = self.archive / fold
        site_recipe = choice.removeprefix("site_") if choice.startswith("site_") else portfolio_site
        with (np.load(directory / "groups.npz") as groups, np.load(directory / "group_forecasts.npz") as group_fc,
              np.load(directory / "oof_recipes.npz") as oof):
            membership = groups[choice]
            details = hourly_diagnostics(actual[complete], group_fc[choice][complete], oof[site_recipe][complete], membership,
                                         self.smp[part][complete], self.k[part][complete], self.times[part].hour[complete])
        details["hours_total"] = len(actual)
        details["hours_missing_or_invalid_actual"] = int((~complete).sum())
        return details, membership

    def transfer_groups(self, site_recipes):
        """Score already-chosen memberships in later months on the same bottom-up forecast."""
        transfer = {}
        for origin_index, origin in enumerate(MONTHS[:-1]):
            with np.load(self.archive / origin / "groups.npz") as archive:
                groups = {name: archive[name] for name in ALLOCATION}
            later = {}
            for target in MONTHS[origin_index + 1:]:
                part = self.slice(target)
                y = self.y[part]
                valid = (np.isfinite(y) & (y >= 0)).all(axis=1)
                with np.load(self.archive / target / "oof_recipes.npz") as archive:
                    forecast = archive[site_recipes[target]][valid]
                actual, smp, k = y[valid], self.smp[part][valid], self.k[part][valid]
                later[target] = {name: group_settlement(group_sum(actual, first), group_sum(forecast, first), smp, k)["penalty_krw"]
                                 for name, first in groups.items()}
            transfer[origin] = later
        return transfer

    def select(self):
        plan_path = self.archive / "selection.json"
        if plan_path.exists():
            raise FileExistsError(plan_path)
        season = read_json(self.out / "seasonal/summary.json")
        self.experts = tuple(season["ranking"][:6])
        self.archive.mkdir(exist_ok=True)
        self.warmup()
        scores, recipes = {}, {}
        site_candidates = [f"site_{name}" for name in (*self.experts, *ENSEMBLES)]
        for fold in MONTHS:
            portfolio_site = (stable_rank(list(scores.values()), site_candidates)[0].removeprefix("site_")
                              if scores else self.experts[0])
            recipes[fold] = portfolio_site
            scores[fold] = self.run_fold(fold, portfolio_site)
        portfolios = [name for name in scores["jan"] if not name.startswith("site_")]
        ordering = stable_rank(list(scores.values()), site_candidates + portfolios)
        choice = ordering[0]
        diagnostics = {}
        assignments = []
        for fold in MONTHS:
            diagnostics[fold], membership = self.month_diagnostics(fold, choice, recipes[fold])
            assignments.append(membership)
        churn = {f"{left}_to_{right}": int(min(np.count_nonzero(a != b), np.count_nonzero(a == b)))
                 for left, right, a, b in zip(MONTHS, MONTHS[1:], assignments, assignments[1:])}
        common = ["feb", "mar", "apr"]
        legacy = read_json(self.out / "selection.json")
        previous_choice = legacy["selected"]
        comparison = {month: {"previous": legacy["folds"][month][previous_choice]["penalty_krw"],
                              "nested": scores[month][choice]["penalty_krw"]} for month in common}
        plan = {"selected": choice, "portfolio_site": stable_rank(list(scores.values()), site_candidates)[0].removeprefix("site_"),
                "portfolio_site_by_month": recipes, "seasonal_experts": list(self.experts), "ranking": ordering,
                "coverage": {month: {"scored": diagnostics[month]["hours_scored"], "requested": diagnostics[month]["hours_total"],
                                      "missing_or_invalid": diagnostics[month]["hours_missing_or_invalid_actual"]} for month in MONTHS},
                "monthly_selected": diagnostics, "group_assignment_churn": churn, "folds": scores,
                "fixed_membership_later_month_penalty_krw": self.transfer_groups(recipes),
                "legacy_comparison_feb_to_apr_krw": {"previous_choice": previous_choice, "monthly": comparison}}
        dump_json(plan_path, plan)
        print(f"Selected before May: {choice}; site pipeline = {plan['portfolio_site']}; seasonal shortlist = {self.experts}", flush=True)

    def freeze(self):
        path = self.archive / "frozen.json"
        if path.exists() or (self.archive / "holdout.json").exists():
            raise FileExistsError("The May and submission forecasts are already frozen")
        plan_path = self.archive / "selection.json"
        plan = read_json(plan_path)
        self.experts = tuple(plan["seasonal_experts"])
        part = self.slice("may")
        may_times = self.times[part]
        eval_times = self.evaluation.index
        future_times = may_times.append(eval_times)
        before = issue_cutoff([may_times[0]])[0]
        stop = self.times.searchsorted(before, side="right")
        test = np.load(self.out / "test_x.npy", mmap_mode="r")
        may = self.forecasts("may", may_times)
        evaluation = predict_frozen(self.out / "may", self.y[:stop], self.times[:stop], test, eval_times,
                                    self.capacity, self.feature_names, self.threads)
        future = {name: np.concatenate((may[name], evaluation[name]), axis=0) for name in may}
        context = np.concatenate((self.x[part][:, :, self.context], test[:, :, self.context]), axis=0)
        choices = self.fit_decisions("may", plan["portfolio_site"], future, future_times, context)
        forecast, first = choices[plan["selected"]]
        if not np.isfinite(forecast).all() or (forecast < 0).any():
            raise ValueError("Invalid frozen forecasts")
        capacities = group_capacities(self.sites, as_composition(self.sites, first)).to_dict()
        may_size = len(may_times)
        site_name = plan["selected"].removeprefix("site_") if plan["selected"].startswith("site_") else plan["portfolio_site"]
        artifact = self.archive / "frozen.npz"
        np.savez(artifact, forecast_may=forecast[:may_size], forecast_eval=forecast[may_size:], first=first,
                 site_may=future[site_name][:may_size], climatology=may["climatology"], full_lgbm=may["lgbm_full"])
        dump_json(path, {"selected": plan["selected"], "labels_through": str(before), "capacity_kw": capacities,
                         "selection_sha256": sha256(plan_path), "forecast_sha256": sha256(artifact)})
        print(f"Froze May and submission forecasts through label cutoff {before}; {capacities}", flush=True)

    def checked_manifest(self):
        frozen = read_json(self.archive / "frozen.json")
        if (sha256(self.archive / "selection.json") != frozen["selection_sha256"]
                or sha256(self.archive / "frozen.npz") != frozen["forecast_sha256"]):
            raise ValueError("Frozen forecast or selection has changed")
        return frozen

    def holdout(self):
        path = self.archive / "holdout.json"
        if path.exists():
            raise FileExistsError("May was already evaluated")
        frozen = self.checked_manifest()
        part = self.slice("may")
        actual = self.y[part]
        complete = (np.isfinite(actual) & (actual >= 0)).all(axis=1)
        with np.load(self.archive / "frozen.npz") as archive:
            y, smp, k = actual[complete], self.smp[part][complete], self.k[part][complete]
            chosen = hourly_diagnostics(y, archive["forecast_may"][complete], archive["site_may"][complete], archive["first"],
                                        smp, k, self.times[part].hour[complete])
            fixed = {name: group_settlement(group_sum(y, self.reference), group_sum(archive[name][complete], self.reference), smp, k)
                     for name in ("climatology", "full_lgbm")}
        report = {"frozen_forecast_sha256": frozen["forecast_sha256"], "selected_before_may": frozen["selected"],
                  "hours_requested": len(actual), "hours_scored": int(complete.sum()), "selected": chosen,
                  "predeclared_references": fixed}
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(f"May {int(complete.sum())}/{len(actual)} hours: W={chosen['penalty_krw']:,.0f} KRW", flush=True)

    def export(self):
        frozen = self.checked_manifest()
        holdout = read_json(self.archive / "holdout.json")
        if holdout["frozen_forecast_sha256"] != frozen["forecast_sha256"]:
            raise ValueError("Submission and May forecasts are from different runs")
        times = self.evaluation.index
        with np.load(self.archive / "frozen.npz") as artifact:
            composition = as_composition(self.sites, artifact["first"])
            forecast = pd.DataFrame({"group": np.repeat(GROUPS, len(times)), "ts": np.tile(times.astype(str), 2),
                                     "forecast_kwh": artifact["forecast_eval"].T.ravel()})
        destination = self.archive / "submission"
        if destination.exists():
            raise FileExistsError(destination)
        with tempfile.TemporaryDirectory(dir=self.archive) as temp:
            folder = Path(temp) / "submission"
            folder.mkdir()
            composition.to_csv(folder / "composition.csv", index=False)
            forecast.to_csv(folder / "forecast.csv", index=False)
            checks = validate_files(self.sites, times, folder)
            hashes = {name: sha256(folder / name) for name in ("composition.csv", "forecast.csv")}
            folder.rename(destination)
        dump_json(self.archive / "submission_checks.json", {**checks, "sha256": hashes, "selected": frozen["selected"]})
        print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=("select", "freeze", "holdout", "export"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    raw, out = args.raw.resolve(), args.out.resolve()
    if out.is_relative_to(raw) or out.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Place all outputs outside the repository and participant package")
    with threadpool_limits(limits=args.threads):
        experiment = NestedExperiment(raw, out, args.threads)
        getattr(experiment, args.stage)()
