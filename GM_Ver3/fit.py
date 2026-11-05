"""Train independent site models on expanding, cutoff-safe folds."""

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
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from torch import nn

from .features import prepare_features, read_actual, read_prices, read_roster
from .settlement import issue_cutoff, price_arrays


SEED = 20261106
FOLDS = {"warmup": "2024-12-24", "feb": "2025-02-01", "mar": "2025-03-01",
         "apr": "2025-04-01", "may": "2025-05-01", "evaluation": "2025-06-01"}
MODELS = ("climatology", "lgbm_full", "lgbm_weighted", "lgbm_round17", "xgb", "catboost",
          "extra_trees", "random_forest", "gru", "lstm")


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def start_at(day):
    return pd.Timestamp(day, tz="Asia/Seoul") + pd.Timedelta(hours=1)


def reference_climatology(y, train_times, forecast_times):
    # A site/hour fallback covers calendar months that have never been observed at a site.
    observed = pd.DataFrame(np.where(y >= 0, y, np.nan))
    history = observed.groupby(train_times.hour).mean().reindex(range(24)).to_numpy()
    history = np.nan_to_num(history)
    month_hour = observed.groupby([train_times.month, train_times.hour]).mean()
    future_keys = pd.MultiIndex.from_arrays([forecast_times.month, forecast_times.hour])
    prediction = month_hour.reindex(future_keys).to_numpy()
    return np.where(np.isfinite(prediction), prediction, history[forecast_times.hour])


class DayRNN(nn.Module):
    def __init__(self, inputs, kind):
        super().__init__()
        recurrent = nn.GRU if kind == "gru" else nn.LSTM
        self.temporal = recurrent(inputs, 32, batch_first=True, bidirectional=True)
        self.output = nn.Sequential(nn.Linear(64, 1), nn.Softplus())
        nn.init.constant_(self.output[0].bias, -2.5)

    def forward(self, inputs):
        return self.output(self.temporal(inputs)[0]).squeeze(-1)


def recurrent_fit(kind, x, y, future, caps, rng, threads, folder):
    torch.manual_seed(SEED)
    torch.set_num_threads(threads)
    days = len(x) // 24
    sites, n_features = x.shape[1:]
    observed = y[:days * 24].reshape(days, 24, sites)
    eligible = np.flatnonzero((np.isfinite(observed) & (observed >= 0)).sum(axis=1) >= 12)
    selected = rng.choice(eligible, size=min(16_000, len(eligible)), replace=False)
    day_index, site = np.divmod(selected, sites)
    hour = day_index[:, None] * 24 + np.arange(24)
    inputs = x[hour, site[:, None], :]
    labels = y[hour, site[:, None]] / caps[site, None]
    mean = np.nanmean(inputs, axis=(0, 1))
    scale = np.nanstd(inputs, axis=(0, 1))
    mean = np.nan_to_num(mean)
    scale = np.where(np.isfinite(scale) & (scale >= 1e-5), scale, 1.)

    def normalize(array):
        return np.nan_to_num((array - mean) / scale).clip(-10, 10)

    data = torch.from_numpy(normalize(inputs).astype(np.float32))
    good = torch.from_numpy(np.isfinite(labels) & (labels >= 0))
    target = torch.from_numpy(np.where(good.numpy(), labels, 0).astype(np.float32))
    model = DayRNN(n_features, kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    for _ in range(3):
        order = rng.permutation(len(data))
        for start in range(0, len(order), 256):
            batch = order[start:start + 256]
            valid = good[batch]
            prediction = model(data[batch])
            loss = torch.nn.functional.smooth_l1_loss(prediction[valid], target[batch][valid], beta=0.07)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save({"weights": model.state_dict(), "mean": mean, "scale": scale}, folder / f"{kind}.pt")
    return recurrent_predict(model, future, caps, mean, scale), len(selected)


def recurrent_predict(model, future, caps, mean, scale):
    result = np.empty(future.shape[:2], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(future), 24):
            block = future[start:start + 24].transpose(1, 0, 2)
            normalized = np.nan_to_num((block - mean) / scale).clip(-10, 10)
            prediction = model(torch.from_numpy(normalized.astype(np.float32))).numpy()
            result[start:start + 24] = prediction.T * caps
    return result


def tree_predict(model, future, caps, threads, columns=None, fill=False):
    prediction = np.empty(future.shape[:2], dtype=np.float32)
    for left in range(0, len(future), 24 * 14):
        right = min(left + 24 * 14, len(future))
        block = future[left:right].reshape(-1, future.shape[-1])
        block = block if columns is None else block[:, columns]
        if fill:
            block = np.nan_to_num(block, nan=-999.)
        if isinstance(model, lgb.LGBMRegressor):
            values = model.booster_.predict(block, num_threads=threads)
        else:
            values = model.predict(block)
        prediction[left:right] = values.reshape(-1, len(caps)).clip(0) * caps
    return prediction


def predict_frozen(folder, y, train_times, future, future_times, caps, names, threads):
    """Use the checkpoints from the May forecast fit, with no May labels."""
    predictions = {"climatology": reference_climatology(y, train_times, future_times).astype(np.float32)}
    torch.set_num_threads(threads)
    for name in MODELS[1:]:
        if name in {"gru", "lstm"}:
            saved = torch.load(folder / f"{name}.pt", map_location="cpu", weights_only=False)
            model = DayRNN(len(names), name)
            model.load_state_dict(saved["weights"])
            predictions[name] = recurrent_predict(model, future, caps, saved["mean"], saved["scale"])
        else:
            model = joblib.load(folder / f"{name}.joblib")
            columns = [i for i, value in enumerate(names) if not value.endswith("_revision")] if name == "lgbm_round17" else None
            predictions[name] = tree_predict(model, future, caps, threads, columns, name in {"extra_trees", "random_forest"})
    return predictions


def fit_predict(x, y, times, future, future_times, q, caps, names, folder, threads):
    if times[-1] > issue_cutoff([future_times[0]])[0]:
        raise ValueError("Training labels cross the first forecast cutoff")
    folder.mkdir()
    n_sites = len(caps)
    train = x.reshape(-1, len(names))
    valid = np.flatnonzero(np.isfinite(y.ravel()) & (y.ravel() >= 0))
    # For lower-budget challengers, draw across all available years using the same seed.
    rng = np.random.default_rng(SEED)
    sampled = np.sort(rng.choice(valid, min(280_000, len(valid)), replace=False))
    small = np.sort(rng.choice(sampled, min(72_000, len(sampled)), replace=False))
    keep = np.flatnonzero([not name.endswith("_revision") for name in names])
    lgb_params = dict(n_estimators=210, learning_rate=.065, num_leaves=48, min_child_samples=200,
                      colsample_bytree=.90, verbosity=-1, n_jobs=threads, random_state=SEED)
    specs = {
        "lgbm_full": (lgb.LGBMRegressor(objective="regression", **lgb_params), valid, None),
        "lgbm_weighted": (lgb.LGBMRegressor(objective="regression_l1", **lgb_params), sampled, None),
        "lgbm_round17": (lgb.LGBMRegressor(objective="regression_l1", **lgb_params), sampled, keep),
        "xgb": (xgb.XGBRegressor(n_estimators=265, max_depth=7, learning_rate=.065, tree_method="hist",
                                 n_jobs=threads, random_state=SEED), sampled, None),
        "catboost": (CatBoostRegressor(iterations=240, depth=7, learning_rate=.08, thread_count=threads,
                                       verbose=False, allow_writing_files=False, random_seed=SEED), sampled, None),
        "extra_trees": (ExtraTreesRegressor(n_estimators=40, max_depth=14, min_samples_leaf=16,
                                            n_jobs=threads, random_state=SEED), small, None),
        "random_forest": (RandomForestRegressor(n_estimators=30, max_depth=13, min_samples_leaf=18,
                                                max_features=.75, n_jobs=threads, random_state=SEED), small, None),
    }
    forecasts = {"climatology": reference_climatology(y, times, future_times).astype(np.float32)}
    records = {"climatology": {"training_rows": len(valid)}}
    for name, (model, rows, columns) in specs.items():
        start = time.perf_counter()
        inputs = train[rows] if columns is None else train[np.ix_(rows, columns)]
        fill = name in {"extra_trees", "random_forest"}
        if fill:
            inputs = np.nan_to_num(inputs, nan=-999.)
        normalized = y.ravel()[rows] / caps[rows % n_sites]
        weighted = name in {"lgbm_weighted", "lgbm_round17"}
        sample_weight = q[rows // n_sites] if weighted else None
        model.fit(inputs, normalized, sample_weight=sample_weight)
        del inputs
        prediction = tree_predict(model, future, caps, threads, columns, fill)
        joblib.dump(model, folder / f"{name}.joblib", compress=3)
        forecasts[name] = prediction
        records[name] = {"training_rows": len(rows), "seconds": round(time.perf_counter() - start, 2)}
        print(f"{folder.name} {name}: {records[name]}", flush=True)
    for name in ("gru", "lstm"):
        start = time.perf_counter()
        forecasts[name], count = recurrent_fit(name, x, y, future, caps, rng, threads, folder)
        records[name] = {"training_site_days": count, "seconds": round(time.perf_counter() - start, 2)}
        print(f"{folder.name} {name}: {records[name]}", flush=True)
    for name, pred in forecasts.items():
        if not np.isfinite(pred).all() or (pred < 0).any():
            raise ValueError(f"Invalid forecast for {name}")
        np.save(folder / f"{name}.npy", pred)
    dump_json(folder / "protocol.json", {"last_label_time": str(times[-1]), "first_issue_cutoff": str(issue_cutoff([future_times[0]])[0]),
                                        "first_target": str(future_times[0]), "last_target": str(future_times[-1]),
                                        "models": records})


def train_fold(out, fold, weights, capacity, threads):
    train_x = np.load(out / "train_x.npy", mmap_mode="r")
    y = np.load(out / "train_y.npy", mmap_mode="r")
    names = json.loads((out / "features.json").read_text(encoding="utf-8"))
    times, smp, k = price_arrays(weights)
    start = start_at(FOLDS[fold])
    cutoff = issue_cutoff([start])[0]
    stop = times.searchsorted(cutoff, side="right")
    following = list(FOLDS)[list(FOLDS).index(fold) + 1]
    part = slice(times.searchsorted(start), times.searchsorted(start_at(FOLDS[following])))
    future = train_x[part]
    future_times = times[part]
    fit_predict(train_x[:stop], y[:stop], times[:stop], future, future_times, smp[:stop] * k[:stop],
                capacity, names, out / fold, threads)


def input_files(raw):
    return (read_roster(raw), read_prices(raw / "train/smp.csv"), read_prices(raw / "smp.csv"))


def prepare(raw, out, sites, weights, evaluation):
    out.mkdir(parents=True, exist_ok=False)
    y = read_actual(raw, weights.index, sites)
    np.save(out / "train_y.npy", y)
    _, names = prepare_features(raw, "train", weights.index, sites, out / "train_x.npy")
    print(f"Prepared {len(names)} inputs; {int((np.isfinite(y) & (y >= 0)).sum()):,} valid training labels", flush=True)
    _, eval_names = prepare_features(raw, "test", evaluation.index, sites, out / "test_x.npy")
    if names != eval_names:
        raise ValueError("Training and evaluation features differ")
    dump_json(out / "features.json", names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stage", choices=("development", "may"), default="development")
    args = parser.parse_args()
    raw, out = args.raw.resolve(), args.out.resolve()
    if out.is_relative_to(raw) or out.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Place all outputs outside the repository and participant package")
    sites, weights, evaluation = input_files(raw)
    if args.stage == "development":
        prepare(raw, out, sites, weights, evaluation)
        folds = ("warmup", "feb", "mar", "apr")
    else:
        if not (out / "validation_nested/selection.json").is_file():
            raise ValueError("Finalize the nested development selection before fitting the May forecast")
        folds = (args.stage,)
    for fold in folds:
        train_fold(out, fold, weights, sites.capacity_kw.to_numpy(dtype=np.float32), args.threads)
