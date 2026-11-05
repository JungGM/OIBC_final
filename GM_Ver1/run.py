"""Reproducible CPU tuning, chronological checks, full-history refit and inference.

Run from repository root: python -m GM_Ver1.run --help
Competition rows, learned weights and predictions are written only to --out.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import xgboost as xgb

from train.dataset import current_dataset, score_predictions
from GM_Ver1.common import (SEED, TZ, Panel, fit_scaler, local_output, prepare_panel,
                           sample_ids, sha256, tabular, write_json)
from GM_Ver1.gru import ForecastGRU, predict as gru_predict, train as gru_train
from GM_Ver1.tcn import ForecastTCN, receptive_field

TUNE_END = "2022-12-31 17:00:00+09:00"
SPLITS = [
    ("confirm_2024", "2023-12-31 17:00:00+09:00", "2024-01-01", "2024-12-31"),
    ("check_2025", "2024-12-31 17:00:00+09:00", "2025-01-01", "2025-05-31"),
]
FINAL_END = "2025-05-31 17:00:00+09:00"


def stamp(day):
    return str(pd.Timestamp(day, tz=TZ))


def tune_xgb(panel, out, train_ids, val_ids, trials):
    xp, yp, cap = tabular(panel, train_ids, TUNE_END, 500_000)
    xv, yv, cv = tabular(panel, val_ids, None)
    # Capacity weights on validation make normalized-target MAE proportional to
    # raw-kWh MAE. No SMP or future label is an input or training weight.
    dval = xgb.DMatrix(xv, label=yv, weight=cv, feature_names=panel.features, nthread=4)
    del xv
    study = optuna.create_study(study_name="xgboost_2023", direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=4),
        storage=f"sqlite:///{(out / 'study.sqlite3').as_posix()}", load_if_exists=True)

    def objective(trial):
        config = dict(objective=trial.suggest_categorical("objective", ["reg:squarederror", "reg:absoluteerror"]),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            eta=trial.suggest_float("eta", .025, .12, log=True),
            min_child_weight=trial.suggest_float("min_child_weight", 2, 40, log=True),
            subsample=trial.suggest_float("subsample", .65, 1),
            colsample_bytree=trial.suggest_float("colsample_bytree", .65, 1),
            reg_lambda=trial.suggest_float("reg_lambda", 1, 30, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1, log=True),
            tree_method="hist", max_bin=128, nthread=4, seed=SEED, eval_metric="mae")
        weight_power = trial.suggest_categorical("weight_power", [1, 2])
        dtrain = xgb.DMatrix(xp, label=yp, weight=(cap/170)**weight_power,
                             feature_names=panel.features, nthread=4)
        started = time.time()
        model = xgb.train(config, dtrain, num_boost_round=600, evals=[(dval, "validation")],
                          early_stopping_rounds=45, verbose_eval=False)
        rounds = model.best_iteration+1
        pred = model.predict(dval, iteration_range=(0, rounds))*cv
        mae = float(np.abs(pred-yv*cv).mean())
        trial.set_user_attr("config", dict(**config, weight_power=weight_power, num_boost_round=rounds))
        trial.set_user_attr("seconds", time.time()-started)
        trial.set_user_attr("training_rows", len(yp))
        trial.set_user_attr("validation_rows", len(yv))
        print(f"XGBoost trial {trial.number}: MAE={mae:.6f}, rounds={rounds}", flush=True)
        return mae

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    study.optimize(objective, n_trials=max(0, trials-done), gc_after_trial=True)
    rows = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE], key=lambda t: t.value)
    write_json(out / "trials.json", [dict(trial=t.number, validation_mae_kwh=t.value, parameters=t.params,
                                        **t.user_attrs) for t in rows])
    return [dict(trial=t.number, validation_mae_kwh=t.value, config=t.user_attrs["config"]) for t in rows[:2]]


def tune_gru(panel, out, train_ids, val_ids, trials, family="GRU"):
    ids = sample_ids(train_ids, 20_000)
    scaler_path = out / "tune_scaler.json"
    if scaler_path.exists():
        scaler = json.loads(scaler_path.read_text())
    else:
        scaler = fit_scaler(panel, train_ids, TUNE_END)
        write_json(scaler_path, scaler)
    study = optuna.create_study(study_name=f"{family.lower()}_2023", direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=4),
        storage=f"sqlite:///{(out / 'study.sqlite3').as_posix()}", load_if_exists=True)

    def objective(trial):
        config = dict(embedding_dim=trial.suggest_categorical("embedding_dim", [4, 8]),
            dropout=trial.suggest_float("dropout", 0, .25),
            learning_rate=trial.suggest_float("learning_rate", .0005, .004, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            loss=trial.suggest_categorical("loss", ["mse", "huber"]),
            huber_beta=trial.suggest_float("huber_beta", .03, .15, log=True),
            batch_size=512, max_epochs=16, patience=4, seed=SEED)
        if family == "GRU":
            config.update(hidden_size=trial.suggest_categorical("hidden_size", [32, 48, 64]),
                num_layers=trial.suggest_int("num_layers", 1, 2),
                bidirectional=trial.suggest_categorical("bidirectional", [False, True]))
        else:
            config.update(channels=trial.suggest_categorical("channels", [32, 48, 64]),
                levels=trial.suggest_int("levels", 3, 4),
                kernel_size=trial.suggest_categorical("kernel_size", [2, 3, 5]))
            config["receptive_field"] = receptive_field(config)
        model, history, epoch = gru_train(panel, ids, TUNE_END, config, scaler, val_ids=val_ids,
            on_epoch=lambda h: write_json(out / f"trial_{trial.number:02d}_history.json", h),
            model_class=ForecastGRU if family == "GRU" else ForecastTCN)
        pred = gru_predict(model, panel, val_ids, scaler)
        actual = np.array(panel.y[val_ids])
        mask = np.isfinite(actual) & (actual >= 0)
        mae = float(np.abs(pred[mask]-actual[mask]).mean())
        config["epochs"] = epoch
        trial.set_user_attr("config", config)
        trial.set_user_attr("training_sequences", len(ids))
        trial.set_user_attr("validation_sequences", len(val_ids))
        trial.set_user_attr("validation_rows", int(mask.sum()))
        print(f"{family} trial {trial.number}: MAE={mae:.6f}, best epoch={epoch}", flush=True)
        return mae

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    study.optimize(objective, n_trials=max(0, trials-done), gc_after_trial=True)
    rows = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE], key=lambda t: t.value)
    write_json(out / "trials.json", [dict(trial=t.number, validation_mae_kwh=t.value, parameters=t.params,
                                        **t.user_attrs) for t in rows])
    return [dict(trial=t.number, validation_mae_kwh=t.value, config=t.user_attrs["config"]) for t in rows[:2]]


def fit_model(family, panel, end, cfg, directory):
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / ("model.ubj" if family == "XGBoost" else "model.pt")
    ids = panel.indices(end=end)
    if model_path.exists() and (directory / "fit.json").exists():
        if family == "XGBoost":
            model = xgb.Booster()
            model.load_model(model_path)
            return model, None
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if checkpoint["config"] != cfg or checkpoint["training_end"] != end:
            raise ValueError("GRU checkpoint configuration mismatch")
        torch.set_num_threads(4)
        cls = ForecastGRU if family == "GRU" else ForecastTCN
        model = cls(len(panel.features), cfg, panel.nsites)
        model.load_state_dict(checkpoint["state_dict"])
        return model, checkpoint["scaler"]
    print(f"Fitting {family} through {end}: {len(ids)} labeled sequences", flush=True)
    started = time.time()
    if family == "XGBoost":
        x, y, cap = tabular(panel, ids, end)
        cfg = cfg.copy()
        rounds, power = cfg.pop("num_boost_round"), cfg.pop("weight_power")
        # Retain quantized histogram input rather than another full dense copy.
        dm = xgb.QuantileDMatrix(x, label=y, weight=(cap/170)**power,
                                feature_names=panel.features, nthread=4, max_bin=cfg["max_bin"])
        nrows = len(y)
        del x, y, cap
        gc.collect()
        model = xgb.train(cfg, dm, num_boost_round=rounds, verbose_eval=False)
        model.save_model(model_path)
        scaler = None
    else:
        scaler_path = directory / "scaler.json"
        scaler = json.loads(scaler_path.read_text()) if scaler_path.exists() else fit_scaler(panel, ids, end)
        write_json(scaler_path, scaler)
        # The tuning set has at most 20k sequences; keep a comparable update
        # budget as the history expands, with at least three full passes when
        # the selected tuning epoch allows it. Every fit still uses every row.
        actual_epochs = min(cfg["epochs"], max(3, math.ceil(cfg["epochs"]*20_000/len(ids))))
        model, history, _ = gru_train(panel, ids, end, cfg, scaler, epochs=actual_epochs,
            on_epoch=lambda h: write_json(directory / "history.json", h),
            model_class=ForecastGRU if family == "GRU" else ForecastTCN)
        nrows = 0
        for block in np.array_split(ids, max(1, (len(ids)+2047)//2048)):
            nrows += int(panel.batch(block, end)[2].sum())
        torch.save(dict(state_dict=model.state_dict(), config=cfg, scaler=scaler, trained_epochs=actual_epochs,
                        features=panel.features, site_ids=panel.sites, training_end=end), model_path)
    write_json(directory / "fit.json", dict(training_start="2021-01-01 01:00:00+09:00",
        training_end=end, eligible_rows=nrows, labeled_sequences=len(ids), seconds=time.time()-started,
        model_sha256=sha256(model_path), full_history=True, excluded_missing_and_negative=True,
        trained_epochs=actual_epochs if family != "XGBoost" else None,
        refit_epoch_policy="min(selected_epoch, max(3, ceil(selected_epoch*20000/full_training_sequences)))" if family != "XGBoost" else None))
    print(f"Saved {model_path} ({time.time()-started:.1f}s)", flush=True)
    return model, scaler


def predict_to_file(family, model, scaler, panel, ids, path, *, fit_end, role):
    path = Path(path)
    if path.exists() and path.with_suffix(".json").exists():
        return
    writer = None
    total = 0
    try:
        for pos in range(0, len(ids), 2048):
            block = ids[pos:pos+2048]
            if family != "XGBoost":
                prediction = gru_predict(model, panel, block, scaler)
            else:
                x = np.array(panel.X[block], dtype=np.float32)
                dm = xgb.DMatrix(x.reshape(-1, len(panel.features)), feature_names=panel.features, nthread=4)
                prediction = model.predict(dm).reshape(-1, 24) * panel.capacity[panel.site_index[block], None]
            prediction = np.maximum(prediction, 0)
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite prediction")
            ts = pd.to_datetime(panel.times(block).reshape(-1), utc=True).tz_convert(TZ)
            keys = np.repeat(np.array(panel.sites)[panel.site_index[block]], 24)
            frame = pd.DataFrame(dict(site_id=keys, ts=ts, prediction=prediction.reshape(-1)))
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            total += len(frame)
    finally:
        if writer is not None:
            writer.close()
    expected = len(ids)*24
    if total != expected:
        raise ValueError("Missing forecast rows")
    write_json(path.with_suffix(".json"), dict(rows=total, sha256=sha256(path), training_end=fit_end,
        role=role, site_count=panel.nsites, nonnegative_finite=True,
        warning="Chronological predictions only; never use final-refit training predictions for stacking"))


def evaluate(panel, ids, prediction_file, dataset):
    p = pd.read_parquet(prediction_file)
    # Scoring reads original float64 targets rather than the float32 fitting
    # cache, preserving every observed label exactly (including negatives).
    filt = ((ds.field("ts") >= p.ts.min().as_unit("ns")) &
            (ds.field("ts") <= p.ts.max().as_unit("ns")))
    actual = ds.dataset(Path(dataset)/"training_data.parquet", format="parquet").to_table(
        columns=["site_id", "ts", "energy_kwh"], filter=filt).to_pandas().rename(columns={"energy_kwh": "actual"})
    p = p.merge(actual, on=["site_id", "ts"], how="left", validate="one_to_one", sort=False)
    y = p.actual.to_numpy()
    # Preserve negative observed targets in scoring, as the frozen scorer does.
    valid = np.isfinite(y)
    observed = p.loc[valid].copy()
    error = observed.prediction-observed.actual
    nmae = float(np.mean(np.abs(error.to_numpy()) /
                 np.repeat(panel.capacity[panel.site_index[ids]], 24)[valid]))
    report = dict(rows=int(valid.sum()), prediction_rows=len(p), site_mae_kwh=float(error.abs().mean()),
                  site_rmse_kwh=float(np.sqrt((error**2).mean())), normalized_mae=nmae,
                  negative_observed_labels=int((y[valid] < 0).sum()))
    settlement, _ = score_predictions(observed,
        pd.read_parquet(Path(dataset)/"reference_composition.parquet"),
        pd.read_parquet(Path(dataset)/"training_weights.parquet"))
    report["reference_groups"] = settlement
    monthly = []
    for month, frame in observed.groupby(observed.ts.dt.strftime("%Y-%m")):
        err = frame.prediction-frame.actual
        monthly.append(dict(month=month, rows=len(frame), mae_kwh=float(err.abs().mean()),
                            rmse_kwh=float(np.sqrt((err**2).mean()))))
    report["monthly"] = monthly
    report["limitation"] = "Reference groups fixed; only complete-roster hours in group score. 2024/2025 were seen in earlier research, not globally untouched test data."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--family", choices=["all", "XGBoost", "GRU", "TCN"], default="all")
    parser.add_argument("--xgb-trials", type=int, default=10)
    parser.add_argument("--gru-trials", type=int, default=8)
    parser.add_argument("--tcn-trials", type=int, default=8)
    parser.add_argument("--stage", choices=["all", "prepare", "tune", "fit"], default="all")
    args = parser.parse_args()
    if min(args.xgb_trials, args.gru_trials, args.tcn_trials) < 2:
        parser.error("At least two completed trials per family are required")
    dataset = args.dataset or current_dataset()
    out = local_output(args.out)
    write_json(out / "environment.json", dict(python=sys.version, platform=platform.platform(), seed=SEED,
        packages={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "pyarrow", "torch", "xgboost", "optuna"]},
        device="cpu", threads=4))
    panel = prepare_panel(dataset, out / "cache" / "training")
    evaluation = prepare_panel(dataset, out / "cache" / "evaluation", evaluation=True)
    if args.stage == "prepare":
        return
    train_ids = panel.indices(end=TUNE_END)
    val_ids = sample_ids(panel.indices(start=stamp("2023-01-01"), end=stamp("2023-12-31")), 12_000, SEED+1)
    families = ["XGBoost", "GRU", "TCN"] if args.family == "all" else [args.family]
    for family in families:
        directory = out / family
        directory.mkdir(parents=True, exist_ok=True)
        chosen_file = directory / "selected.json"
        if chosen_file.exists():
            selected = json.loads(chosen_file.read_text())
        else:
            if family == "XGBoost":
                selected = tune_xgb(panel, directory, train_ids, val_ids, args.xgb_trials)
            else:
                selected = tune_gru(panel, directory, train_ids, val_ids,
                                    args.gru_trials if family == "GRU" else args.tcn_trials, family)
            if len(selected) != 2:
                raise ValueError("Need two successful candidates")
            write_json(chosen_file, selected)
        if args.stage == "tune":
            continue
        for version, candidate in enumerate(selected, 1):
            dest = directory / f"Ver{version}"
            cfg = candidate["config"]
            write_json(dest / "config.json", dict(family=family, version=version, **candidate,
                features=panel.features, tuning_training_end=TUNE_END,
                tuning_validation_operating_days=["2023-01-01", "2023-12-31"],
                selection_metric="observed nonnegative-label site MAE in raw kWh on common 12,000-sequence validation sample",
                features_status="78 frozen default features + site code and 6 solar features, fixed across all trials; no feature-importance claim"))
            metrics = {}
            for split, train_end, left, right in SPLITS:
                model, scaler = fit_model(family, panel, train_end, cfg, dest / split)
                ids = panel.indices(start=stamp(left), end=stamp(right), labeled=False)
                pred_file = dest / split / "predictions.parquet"
                predict_to_file(family, model, scaler, panel, ids, pred_file,
                                fit_end=train_end, role="out_of_time_forecast; frozen fit for entire validation window")
                metrics[split] = evaluate(panel, ids, pred_file, dataset)
                write_json(dest / "metrics.json", metrics)
                del model
                gc.collect()
            model, scaler = fit_model(family, panel, FINAL_END, cfg, dest / "final")
            ids = np.arange(len(evaluation.X))
            predict_to_file(family, model, scaler, evaluation, ids, dest / "final" / "predictions.parquet",
                            fit_end=FINAL_END, role="competition evaluation input; labels unavailable")
            write_json(dest / "complete.json", dict(status="complete", family=family, version=version,
                config_sha256=sha256(dest / "config.json"), final_training_end=FINAL_END,
                evaluation_rows=len(ids)*24, stacking=False, vpp_optimization=False))
            del model
            gc.collect()
    print("Requested forecasting stages complete.", flush=True)


if __name__ == "__main__":
    main()
