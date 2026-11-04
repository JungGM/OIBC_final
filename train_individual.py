"""Run one family or all seven; artifacts stay in a local result directory."""
import argparse
import json
import os
from pathlib import Path
import gc
import numpy as np
import optuna
import pandas as pd

from backtest import START, TUNE_START, TUNE_END, CAL_START, CAL_END, CHECK_START, CHECK_END, TEST_START, END, issue_deadline, SEED
from individual_data import Dataset, site_scores, group_scores
from individual_models import params_for, fit_tree, fit_sequence, predict, save, load

FAMILIES = ("XGBoost","LightGBM","CatBoost","RandomForest","ExtraTrees","GRU","LSTM")
INNER = ((pd.Timestamp("2023-01-01 01:00", tz="Asia/Seoul"),
          pd.Timestamp("2023-04-01 00:00", tz="Asia/Seoul")),
         (pd.Timestamp("2023-04-01 01:00", tz="Asia/Seoul"),
          pd.Timestamp("2023-07-01 00:00", tz="Asia/Seoul")))
OUTER = ((CAL_START, pd.Timestamp("2023-10-01 00:00", tz="Asia/Seoul")),
         (CHECK_START, pd.Timestamp("2024-01-01 00:00", tz="Asia/Seoul")))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf8")


def fit(family, params, train, valid=None):
    if family in ("GRU","LSTM"):
        return fit_sequence(family, params, train, valid)
    return fit_tree(family, params, train, valid)


def validate(family, params, dataset, windows, *, inner=False):
    scores, rows = [], []
    for start, end in windows:
        cutoff = issue_deadline([start])[0]
        train = dataset.panel(START, cutoff)
        valid = dataset.panel(start, end)
        model = fit(family, params, train, valid if inner else None)
        row = valid.records(predict(model, family, valid))
        rows.append(row.assign(fold=str(start.date())))
        scores.append(site_scores(row))
        del train, valid, model
        gc.collect()
    return scores, pd.concat(rows, ignore_index=True)


def run_family(family, dataset, out, trials):
    family_dir = out / family
    family_dir.mkdir(exist_ok=True)
    previous = [int(p.name[3:]) for p in family_dir.glob("ver*") if p.is_dir() and p.name[3:].isdigit()]
    offset = max(previous, default=0)
    study = optuna.create_study(direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=2),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4))
    seeds = {
        "XGBoost": dict(n_estimators=180,max_depth=8,learning_rate=.03,subsample=.9,colsample_bytree=.7),
        "LightGBM": dict(n_estimators=600,num_leaves=127,learning_rate=.06,min_child_samples=200),
    }
    if family in seeds:
        study.enqueue_trial(seeds[family])

    def objective(trial):
        params = params_for(trial, family)
        metrics, _ = validate(family, params, dataset, INNER, inner=True)
        for step, item in enumerate(metrics):
            trial.report(item["MAE"], step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.average([m["MAE"] for m in metrics], weights=[m["rows"] for m in metrics])

    study.optimize(objective, n_trials=trials, gc_after_trial=True)
    candidates = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            fold, rows = validate(family, trial.params, dataset, OUTER)
            metrics = site_scores(rows)
            candidates.append((metrics["MAE"], trial.params, fold, rows, trial.number))
            print(f"{family} trial {trial.number} outer MAE {metrics['MAE']:.5f}", flush=True)
    candidates.sort(key=lambda v: (v[0], np.std([f["MAE"] for f in v[2]]), v[4]))
    for rank, (_, params, folds, oof, trial_id) in enumerate(candidates[:3], start=1):
        path = family_dir / f"ver{offset+rank}"
        path.mkdir(exist_ok=False)
        oof.to_csv(path / "oof.csv", index=False)
        cutoff = issue_deadline([TEST_START])[0]
        train = dataset.panel(START, cutoff)
        model = fit(family, params, train)
        save(model, family, path)
        reloaded = load(family, path)
        probe = dataset.panel(CHECK_START, CHECK_START + pd.Timedelta(hours=23))
        before = predict(model, family, probe)
        after = predict(reloaded, family, probe)
        if not np.allclose(before, after, rtol=1e-6, atol=1e-4):
            raise ValueError("Reload prediction mismatch")
        metrics = dict(inner_period="2023-H1", outer_folds=folds,
            outer=site_scores(oof), outer_group=group_scores(oof, dataset.sites, dataset.prices),
            reload_max_abs_kwh=float(np.max(np.abs(before-after))),
            holdout=None)
        write_json(path / "config.json", dict(family=family, params=params, seed=SEED,
            features=dataset.features, train_cutoff=str(cutoff), trial=trial_id,
            oof_periods=[(str(a),str(b)) for a,b in OUTER]))
        write_json(path / "metrics.json", metrics)
        del train, model, reloaded, probe
        gc.collect()
        print(f"{family} {path.name} saved", flush=True)
    summary = [dict(version=f"ver{offset+i}", MAE=c[0], RMSE=site_scores(c[3])["RMSE"],
                    MBE=site_scores(c[3])["MBE"],
                    fold_MAE=[f["MAE"] for f in c[2]]) for i,c in enumerate(candidates[:3], 1)]
    (family_dir / "README.md").write_text(
        f"# {family}\n\n2023년 3·4분기 외부 OOF 순위. 2023년 상반기 TPE 내부 탐색.\n\n"
        + "\n".join(f"- {s['version']} MAE {s['MAE']:.5f} kWh, RMSE {s['RMSE']:.5f}, "
                    f"MBE {s['MBE']:.5f}, fold MAE {s['fold_MAE']}" for s in summary)
        + "\n", encoding="utf8")
    return summary


def score_holdout(family, dataset, out):
    family_dir = out / family
    for path in sorted(family_dir.glob("ver*")):
        metrics_path = path / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf8"))
        if metrics["holdout"] is not None:
            continue
        model = load(family, path)
        stats = []
        for month in pd.period_range("2024-01","2025-05",freq="M"):
            start = month.start_time.tz_localize("Asia/Seoul") + pd.Timedelta(hours=1)
            end = (month+1).start_time.tz_localize("Asia/Seoul")
            panel = dataset.panel(start, end, holdout=True)
            stats.append(panel.records(predict(model, family, panel)))
            del panel
        rows = pd.concat(stats, ignore_index=True)
        metrics["holdout"] = dict(site=site_scores(rows),
            group=group_scores(rows, dataset.sites, dataset.prices))
        write_json(metrics_path, metrics)
        print(f"{family} {path.name} holdout {metrics['holdout']['site']['MAE']:.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES+("all",), default="all")
    parser.add_argument("--trials", type=int, default=4)
    args = parser.parse_args()
    if "onedrive" in str(args.out.resolve()).lower():
        raise ValueError("Keep model artifacts outside OneDrive")
    if args.trials < 4:
        raise ValueError("At least four trials needed for TPE and top three")
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(args.data, args.cache, args.feature_cache)
    families = FAMILIES if args.family == "all" else (args.family,)
    for family in families:
        run_family(family, dataset, args.out, args.trials)
    # Holdout is read only after all requested families have frozen OOF ranking.
    for family in families:
        score_holdout(family, dataset, args.out)
    records = []
    for family in families:
        for path in (args.out / family).glob("ver*"):
            metrics = json.loads((path / "metrics.json").read_text(encoding="utf8"))
            records.append(dict(model=family, version=path.name,
                **{f"OOF_{k}": v for k,v in metrics["outer"].items()},
                **{f"test_{k}": v for k,v in metrics["holdout"]["site"].items()},
                test_W=metrics["holdout"]["group"]["W"],
                test_S=metrics["holdout"]["group"]["S"],
                test_complete_hours=metrics["holdout"]["group"]["hours"]))
    pd.DataFrame(records).to_csv(args.out / "top3.csv", index=False)


if __name__ == "__main__":
    main()
