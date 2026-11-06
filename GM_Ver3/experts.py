"""Tune heterogeneous experts and write resumable chronological predictions."""

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from catboost import CatBoostRegressor
from torch import nn

from .fit import dump_json, input_files, start_at
from .nested import NESTED_FOLDS
from .settlement import issue_cutoff, price_arrays


SEED = 20261106
FAMILIES = ("lightgbm", "xgboost", "catboost", "gru", "lstm")
TREE_TRIALS = (
    dict(depth=5, rounds=260, eta=.07, weighted=False, features="raw"),
    dict(depth=7, rounds=400, eta=.055, weighted=True, features="weather"),
    dict(depth=7, rounds=400, eta=.055, weighted=True, features="history"),
)
RNN_TRIALS = (
    dict(history=3, hidden=32, layers=1, dropout=.08, lr=.002, batch=384, epochs=5, weighted=False),
    dict(history=7, hidden=48, layers=2, dropout=.12, lr=.0012, batch=512, epochs=7, weighted=True),
)
PAST_WEATHER = ("solar", "wa_cloud", "wb_cloud", "wb_uv_idx", "wa_temp", "wb_temp",
                "cloud_difference", "wb_cloud_revision")


class TwoInputRNN(nn.Module):
    def __init__(self, weather, history, site_count, family, config):
        super().__init__()
        recurrent = nn.GRU if family == "gru" else nn.LSTM
        width = config["hidden"]
        layers = config["layers"]
        self.embedding = nn.Embedding(site_count, 10)
        self.history = recurrent(history, width, batch_first=True)
        self.future = recurrent(weather+10, width, num_layers=layers, batch_first=True,
                                bidirectional=True, dropout=config["dropout"] if layers > 1 else 0.)
        self.head = nn.Sequential(nn.Linear(width*3, width), nn.SiLU(), nn.Dropout(config["dropout"]),
                                  nn.Linear(width, 1))
        nn.init.constant_(self.head[-1].bias, -2.)

    def forward(self, history, future, sites):
        prior = self.history(history)[0][:, -1:, :]
        embed = self.embedding(sites)[:, None, :].expand(-1, future.shape[1], -1)
        next_day = self.future(torch.cat([future, embed], dim=-1))[0]
        return torch.nn.functional.softplus(self.head(torch.cat(
            [next_day, prior.expand(-1, future.shape[1], -1)], dim=-1)).squeeze(-1))


def scaler(values):
    mean = np.nan_to_num(np.nanmean(values, axis=tuple(range(values.ndim-1))))
    std = np.nan_to_num(np.nanstd(values, axis=tuple(range(values.ndim-1))), nan=1.)
    return mean.astype("float32"), np.maximum(std, 1e-4).astype("float32")


def scaled(values, stats):
    return np.nan_to_num((values-stats[0])/stats[1]).clip(-8, 8).astype("float32")


class Store:
    def __init__(self, raw, base, threads):
        self.sites, train, self.eval_prices = input_files(raw)
        self.times, price, k = price_arrays(train)
        self.smp, self.k = price, k
        self.q = price*k
        self.x = np.load(base/"train_x.npy", mmap_mode="r")
        self.y = np.load(base/"train_y.npy", mmap_mode="r")
        self.eval_x = np.load(base/"test_x.npy", mmap_mode="r")
        self.names = json.loads((base/"features.json").read_text(encoding="utf-8"))
        self.index = {name: i for i, name in enumerate(self.names)}
        self.capacity = self.sites.capacity_kw.to_numpy(dtype="float32")
        self.threads = threads
        self.n = len(self.capacity)
        self.history = np.concatenate([self.history_inputs(self.x), self.history_inputs(self.eval_x)])

    def history_inputs(self, x):
        use = [self.index[name] for name in PAST_WEATHER]
        parts = []
        for left in range(0, len(x), 24*32):
            a = x[left:left+24*32][..., use].reshape(-1, 3, 8, self.n, len(use))
            count = np.isfinite(a).sum(axis=2)
            avg = np.divide(np.nansum(a, axis=2), count, out=np.full(count.shape, np.nan, dtype=np.float32),
                            where=count > 0)
            parts.append(avg.transpose(0, 2, 1, 3).reshape(-1, self.n, 3*len(use)))
        return np.concatenate(parts)

    def climate(self, stop):
        values = self.y[:stop]
        valid = np.isfinite(values) & (values >= 0)
        totals = np.zeros((24, self.n))
        count = np.zeros_like(totals)
        np.add.at(totals, np.arange(stop)%24, np.where(valid, values, 0.))
        np.add.at(count, np.arange(stop)%24, valid)
        return (totals/np.maximum(count, 1)/self.capacity).astype("float32")

    def design(self, matrix, hour, site, mode, climatology, historical=False):
        a = matrix[hour, site]
        if mode == "raw":
            return a
        solar = np.maximum(a[:, self.index["solar"]], 0.)
        cloud = a[:, self.index["cloud_difference"]]
        rev = a[:, self.index["wa_cloud_revision"]]-a[:, self.index["wb_cloud_revision"]]
        extra = [solar**.65, solar*cloud, np.abs(cloud), solar*rev, np.abs(rev),
                 solar*a[:, self.index["wb_uv_idx"]], a[:, self.index["wa_clear_proxy"]] -
                 a[:, self.index["wb_clear_proxy"]]]
        if mode == "history":
            extra.append(climatology[hour, site] if historical else climatology[hour%24, site])
        return np.column_stack([a, *extra]).astype("float32")

    def past_sequences(self, day, site, length):
        if length == 0:
            return np.zeros((len(day), 1, self.history.shape[-1]), dtype=np.float32)
        previous = day[:, None]-np.arange(length, 0, -1)
        data = self.history[np.maximum(previous, 0), site[:, None]].copy()
        data[previous < 0] = np.nan
        return data

    def tree(self, family, config, stop, future, budget, destination):
        valid = np.flatnonzero(np.isfinite(self.y[:stop].ravel()) & (self.y[:stop].ravel() >= 0))
        rng = np.random.default_rng(SEED)
        rows = valid if budget >= len(valid) else np.sort(rng.choice(valid, budget, replace=False))
        hour, site = np.divmod(rows, self.n)
        clima = train_clima = None
        if config["features"] == "history":
            clima = self.climate(stop)
            # A training row only sees generation through two operating days
            # earlier. The inference statistic uses labels through its fit cutoff.
            used = stop//24
            targets = self.y[:used*24].reshape(used, 24, self.n)
            present = np.isfinite(targets) & (targets >= 0)
            total = np.cumsum(np.where(present, targets, 0.), axis=0, dtype=np.float32)
            count = np.cumsum(present, axis=0, dtype=np.int32)
            train_clima = np.zeros((used+2, 24, self.n), dtype=np.float32)
            np.divide(total, np.maximum(count, 1), out=train_clima[2:])
            train_clima /= self.capacity
            train_clima = train_clima.reshape(-1, self.n)
        matrix = self.design(self.x, hour, site, config["features"], train_clima, historical=True)
        y = self.y[hour, site]/self.capacity[site]
        weight = None
        if config["weighted"]:
            q = self.q[hour]
            weight = np.clip(q/max(q.mean(), 1.), .05, 5.)**.5
            weight *= np.sqrt(self.capacity[site]/np.median(self.capacity))
        if family == "lightgbm":
            model = lgb.LGBMRegressor(n_estimators=config["rounds"],
                                      num_leaves=config.get("leaves", 2**config["depth"]-1),
                                      min_child_samples=config.get("min_leaf", 150),
                                      learning_rate=config["eta"], verbosity=-1,
                                      colsample_bytree=config.get("colsample", .85),
                                      n_jobs=self.threads, random_state=SEED)
        elif family == "xgboost":
            model = xgb.XGBRegressor(n_estimators=config["rounds"], max_depth=config["depth"],
                                     max_bin=128, min_child_weight=10, learning_rate=config["eta"],
                                     subsample=.85, colsample_bytree=.85, n_jobs=self.threads,
                                     tree_method="hist", random_state=SEED)
        else:
            model = CatBoostRegressor(iterations=config["rounds"], depth=config["depth"],
                                      learning_rate=config["eta"], l2_leaf_reg=8, thread_count=self.threads,
                                      verbose=False, allow_writing_files=False, random_seed=SEED)
        model.fit(matrix, y, sample_weight=weight)
        del matrix
        result = np.empty(future.shape[:2], dtype="float32")
        for left in range(0, len(future), 24*12):
            right = min(left+24*12, len(future))
            h = np.repeat(np.arange(left, right), self.n)
            s = np.tile(np.arange(self.n), right-left)
            fx = self.design(future, h, s, config["features"], clima)
            prediction = model.predict(fx)
            result[left:right] = np.maximum(prediction.reshape(-1, self.n), 0.)*self.capacity
        if destination is not None:
            joblib.dump(dict(model=model, climatology=clima, config=config), destination, compress=1)
        return result, len(rows)

    def recurrent(self, family, config, stop, future, future_start, budget, destination):
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        days = (stop+23)//24
        flat_day, flat_site = np.meshgrid(np.arange(days), np.arange(self.n), indexing="ij")
        obs = np.isfinite(self.y[:stop]) & (self.y[:stop] >= 0)
        padded = np.pad(obs, ((0, 24*days-stop), (0, 0)))
        valid = padded.reshape(days, 24, self.n).sum(axis=1) >= 10
        day, site = flat_day[valid], flat_site[valid]
        ids = rng.choice(len(day), min(budget, len(day)), replace=False)
        day, site = day[ids], site[ids]
        length = config["history"]
        probe = np.arange(0, len(day), max(1, len(day)//1800))
        hours = np.arange(24)
        normalization = scaler(self.x[day[probe, None]*24+hours, site[probe, None]])
        history_norm = scaler(self.past_sequences(day[probe], site[probe], length))
        net = TwoInputRNN(len(self.names), self.history.shape[-1], self.n, family, config)
        optimizer = torch.optim.AdamW(net.parameters(), lr=config["lr"], weight_decay=.0003)
        q_avg = max(self.q[:stop].mean(), 1.)
        epoch_loss = []
        for epoch in range(config["epochs"]):
            net.train()
            order = rng.permutation(len(day))
            losses = []
            for left in range(0, len(day), config["batch"]):
                ix = order[left:left+config["batch"]]
                d, s = day[ix], site[ix]
                rows = 24*d[:, None]+hours
                known = (rows < stop) & padded[rows, s[:, None]]
                truth = self.y[np.minimum(rows, stop-1), s[:, None]]/self.capacity[s, None]
                truth = np.where(known, truth, 0.).astype("float32")
                weight = known.astype("float32")
                if config["weighted"]:
                    weight *= (np.clip(self.q[np.minimum(rows, stop-1)]/q_avg, .05, 4.)**.5
                               * np.sqrt(self.capacity[s, None]/np.median(self.capacity)))
                history = torch.from_numpy(scaled(self.past_sequences(d, s, length), history_norm))
                weather = torch.from_numpy(scaled(self.x[rows, s[:, None]], normalization))
                point = net(history, weather, torch.from_numpy(s))
                loss = torch.nn.functional.smooth_l1_loss(point, torch.from_numpy(truth), beta=.08,
                                                          reduction="none")
                weights = torch.from_numpy(weight)
                cost = (loss*weights).sum()/weights.sum().clamp_min(1.)
                optimizer.zero_grad(set_to_none=True)
                cost.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.)
                optimizer.step()
                losses.append(cost.item())
            epoch_loss.append(float(np.mean(losses)))
            print(f"  {family} epoch {epoch+1}/{config['epochs']}: {epoch_loss[-1]:.6f}", flush=True)
        result = np.empty(future.shape[:2], dtype="float32")
        net.eval()
        with torch.inference_mode():
            for h in range(0, len(future), 24):
                s = np.arange(self.n)
                d = np.full(self.n, (future_start+h)//24)
                hist = torch.from_numpy(scaled(self.past_sequences(d, s, length), history_norm))
                f = torch.from_numpy(scaled(future[h:h+24].transpose(1, 0, 2), normalization))
                result[h:h+24] = net(hist, f, torch.from_numpy(s)).numpy().T*self.capacity
        if destination is not None:
            torch.save(dict(state=net.state_dict(), config=config, history=[a.tolist() for a in history_norm],
                            future=[a.tolist() for a in normalization], epoch_loss=epoch_loss), destination)
        return result, len(day)

    def fit(self, family, config, stop, future, future_start, budget, destination=None):
        start = time.perf_counter()
        if family in ("gru", "lstm"):
            prediction, count = self.recurrent(family, config, stop, future, future_start, budget, destination)
        else:
            prediction, count = self.tree(family, config, stop, future, budget, destination)
        prediction = np.where(future[..., self.index["solar"]] > -.105, prediction, 0.)
        if not np.isfinite(prediction).all():
            raise ValueError(f"Nonfinite {family} prediction")
        return prediction, dict(rows_or_sequences=count, seconds=time.perf_counter()-start)


def observed_metrics(y, p, q, cap):
    valid = np.isfinite(y) & (y >= 0)
    error = (p-y)[valid].astype("float64")
    denom = valid.sum(0) @ cap
    return dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.mean(error**2))),
                mbe=float(error.mean()), nmae=float(np.abs(error).sum()/denom),
                weighted_nmae=float((np.where(valid, np.abs(p-y), 0).sum(1) @ q) /
                                    (q @ (valid @ cap))))


def tune(store, out):
    path = out/"tuning.json"
    if path.exists():
        return json.loads(path.read_text())
    records = {}
    # These held blocks predate all settlement-level selection windows.
    for family in FAMILIES:
        options = RNN_TRIALS if family in ("gru", "lstm") else TREE_TRIALS
        trials = []
        for i, conf in enumerate(options):
            trial_file = out/f"tune_{family}_{i}.json"
            if trial_file.exists():
                trials.append(json.loads(trial_file.read_text()))
                continue
            history = {}
            for date in ("2024-06-15", "2024-09-15"):
                first = start_at(date)
                left = store.times.searchsorted(first)
                stop = store.times.searchsorted(issue_cutoff([first])[0], side="right")
                use = slice(left, left+24*14)
                print(f"Tuning {family}/{i} at {date}", flush=True)
                n = 9000 if family in ("gru", "lstm") else 120000
                prediction, timing = store.fit(family, conf, stop, store.x[use], left, n)
                history[date] = {**observed_metrics(store.y[use], prediction, store.q[use],
                                                   store.capacity), **timing}
            trial = dict(config=conf, folds=history)
            dump_json(trial_file, trial)
            trials.append(trial)
        best = min(trials, key=lambda trial: np.mean([r["weighted_nmae"] for r in trial["folds"].values()]))
        records[family] = dict(selected=best["config"], trials=trials)
    dump_json(path, records)
    return records


def forecasts(store, tuning, out, tree_rows, sequence_rows, stages, families=FAMILIES):
    folds = list(NESTED_FOLDS)
    for fold in stages:
        j = folds.index(fold)
        first = start_at(NESTED_FOLDS[fold])
        stop = store.times.searchsorted(issue_cutoff([first])[0], side="right")
        left = store.times.searchsorted(first)
        if fold == "evaluation":
            future = store.eval_x
        else:
            right = store.times.searchsorted(start_at(NESTED_FOLDS[folds[j+1]]))
            future = store.x[left:right]
        dest = out/fold
        dest.mkdir(exist_ok=True)
        for family in families:
            path = dest/f"{family}.npy"
            if path.exists():
                continue
            config = tuning[family]["selected"]
            n = sequence_rows if family in ("gru", "lstm") else tree_rows
            checkpoint = dest/f"{family}.{'pt' if family in ('gru','lstm') else 'joblib'}"
            print(f"Training {fold}/{family} through {store.times[stop-1]}", flush=True)
            p, rec = store.fit(family, config, stop, future, left, n, checkpoint)
            dump_json(dest/f"{family}.json", dict(config=config, cutoff=str(store.times[stop-1]),
                                                  eligible_labels=int((np.isfinite(store.y[:stop])
                                                    & (store.y[:stop] >= 0)).sum()), **rec))
            partial = path.with_suffix(".partial.npy")
            np.save(partial, p)
            partial.replace(path)


def ablation(store, tuning, out):
    results = {}
    for family in FAMILIES:
        tuned = tuning[family]["selected"]
        if family in ("gru", "lstm"):
            designs = {"issued_history": tuned, "no_issued_history": dict(tuned, history=0)}
            budget = 9000
        else:
            designs = {name: dict(tuned, features=name) for name in ("raw", "weather", "history")}
            budget = 120000
        results[family] = {}
        for name, config in designs.items():
            file = out/f"ablation_{family}_{name}.json"
            if file.exists():
                results[family][name] = json.loads(file.read_text())
                continue
            folds = {}
            for date in ("2024-06-15", "2024-09-15"):
                start = start_at(date)
                left = store.times.searchsorted(start)
                stop = store.times.searchsorted(issue_cutoff([start])[0], side="right")
                held = slice(left, left+24*14)
                print(f"Ablation {family}/{name} {date}", flush=True)
                p, timing = store.fit(family, config, stop, store.x[held], left, budget)
                folds[date] = {**observed_metrics(store.y[held], p, store.q[held], store.capacity), **timing}
            results[family][name] = folds
            dump_json(file, folds)
    dump_json(out/"ablation.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "base", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tree-rows", type=int, default=380000)
    parser.add_argument("--sequences", type=int, default=22000)
    parser.add_argument("--stage", choices=("development", "evaluation", "all", "ablation"), default="development")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.is_relative_to(args.raw.resolve()) or out.is_relative_to(Path(__file__).resolve().parents[1]) or out == args.base.resolve():
        raise ValueError("Use a separate private output directory")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    store = Store(args.raw, args.base, args.threads)
    tuning = tune(store, out)
    if args.stage == "ablation":
        ablation(store, tuning, out)
    else:
        stages = {"development": list(NESTED_FOLDS)[:-1], "evaluation": ["evaluation"],
                  "all": list(NESTED_FOLDS)}[args.stage]
        forecasts(store, tuning, out, args.tree_rows, args.sequences, stages)
