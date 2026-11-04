"""Local, frozen-origin OIBC experiment. Never uploads data or submits files.

Uses the supplied baseline functions and fits all model choices before 2024.
Run: py backtest.py --data PATH --out PATH_OUTSIDE_ONEDRIVE
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow
import xgboost as xgb

TZ = "Asia/Seoul"
SEED = 20261029
THREADS = 4
START = pd.Timestamp("2022-01-01 01:00", tz=TZ)
END = pd.Timestamp("2025-06-01 00:00", tz=TZ)
TUNE_START = pd.Timestamp("2023-01-01 01:00", tz=TZ)
TUNE_END = pd.Timestamp("2023-06-30 17:00", tz=TZ)
CAL_START = pd.Timestamp("2023-07-01 01:00", tz=TZ)
CAL_END = pd.Timestamp("2023-09-30 17:00", tz=TZ)
CHECK_START = pd.Timestamp("2023-10-01 01:00", tz=TZ)
CHECK_END = pd.Timestamp("2023-12-31 17:00", tz=TZ)
TEST_START = pd.Timestamp("2024-01-01 01:00", tz=TZ)
FULL_CUTOFF = pd.Timestamp("2023-12-31 17:00", tz=TZ)
T0 = time.time()


def log(message):
    print(f"[{time.time()-T0:7.1f}s] {message}", flush=True)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=str, allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def issue_deadline(ts):
    # 00:00 closes the preceding operating day.
    return (pd.DatetimeIndex(ts)-pd.Timedelta(hours=1)).normalize()-pd.Timedelta(hours=7)


def period_mask(ts, start, end):
    return (ts >= start) & (ts <= end)


def forecast_day_mean(values, window):
    """Trailing mean within each complete 24h forecast day, including NaNs."""
    a = np.asarray(values, dtype=float).reshape(-1, 24)
    finite = np.isfinite(a)
    sums = np.pad(np.cumsum(np.where(finite, a, 0), axis=1), ((0, 0), (1, 0)))
    counts = np.pad(np.cumsum(finite, axis=1), ((0, 0), (1, 0)))
    end = np.arange(1, 25)
    start = np.maximum(0, end-window)
    denom = counts[:, end]-counts[:, start]
    return np.divide(sums[:, end]-sums[:, start], denom,
                     out=np.full(a.shape, np.nan), where=denom > 0).ravel()


def penalty(actual, prediction, smp, k):
    actual, prediction, smp, k = map(lambda a: np.asarray(a, dtype=float),
                                   (actual, prediction, smp, k))
    if not all(np.isfinite(a).all() for a in (actual, prediction, smp, k)):
        raise ValueError("Score inputs must be finite; missing truth must not be zero-filled")
    if np.any(smp < 0) or not np.isin(k, [1, 3]).all():
        raise ValueError("Invalid SMP or k")
    w = float(np.sum(smp*k*np.abs(actual-prediction)))
    revenue = float(np.sum(smp*actual))
    return {"revenue_krw": revenue, "weighted_error": w,
            "penalty_krw": 0.1*w, "settlement_krw": revenue-0.1*w,
            "mae_kwh": float(np.mean(np.abs(actual-prediction))),
            "rmse_kwh": float(np.sqrt(np.mean((actual-prediction)**2)))}


def robust_absolute_loss(bid, samples, radius, weights=None):
    """Independent symmetric interval SRO, NOT the paper's ESS/MSP model.

    max_|u|<=radius |bid-(sample+u)| = |bid-sample|+radius.
    Radius therefore cannot change the optimal weighted median bid.
    """
    return float(np.average(np.abs(np.asarray(samples)-bid)+radius, weights=weights))


def audit(data, sites, gen, out):
    ts = pd.DatetimeIndex(gen.ts)
    if gen.duplicated(["site_id", "ts"]).any() or not np.isfinite(gen.energy).all():
        raise ValueError("Duplicate keys or nonfinite labels")
    if len(sites) != 253 or sites.site_id.duplicated().any():
        raise ValueError("Invalid sites")
    if set(gen.site_id)-set(sites.site_id):
        raise ValueError("Unknown generation site")
    assert ts.tz is not None
    assert (ts.minute == 0).all() and (ts.second == 0).all()
    year = (ts-pd.Timedelta(hours=1)).year
    counts = gen.groupby("ts").size()
    yearly = gen.assign(year=year).groupby("year").agg(rows=("energy", "size"), sites=("site_id", "nunique"))
    result = {"yearly": yearly.to_dict("index"), "negative_label_count": int((gen.energy < 0).sum()),
              "range": [str(ts.min()), str(ts.max())],
              "complete_253_hours_by_year": counts[counts == 253].groupby(
                  (counts[counts == 253].index-pd.Timedelta(hours=1)).year).size().to_dict(),
              "group_capacity_kw": sites.groupby("current_group").capacity_kw.sum().to_dict(),
              "labels_policy": "Missing targets excluded, never imputed. Raw observed targets retained for scoring. Baseline training target clipped [0,1.2].",
              "source_sha256": {str(p.relative_to(data)): sha256(p) for p in
                  [data/"sites.csv", data/"README.md", data/"baseline.py", data/"train/generation.parquet",
                   data/"train/forecast_wa.parquet", data/"train/forecast_wb.parquet", data/"train/smp.csv"]}}
    save_json(out/"audit.json", result)
    log(f"Audit saved: {len(gen):,} labels; complete panel {sum(counts == 253):,} hours")
    return result


def build_features(data, out, sites, gen, official):
    cache = out/"features"
    cache.mkdir(exist_ok=True)
    # Generation values are never merged into features. A full site/time grid is
    # constructed even where the generation labels do not exist.
    fa = pd.read_parquet(data/"train/forecast_wa.parquet", columns=["wa_station", "round", "ts"]+official.WA_COLS)
    fb = pd.read_parquet(data/"train/forecast_wb.parquet", columns=["wb_station", "round", "ts"]+official.WB_COLS)
    for f, station in [(fa, "wa_station"), (fb, "wb_station")]:
        f["ts"] = pd.to_datetime(f.ts)
        assert not f.duplicated([station, "round", "ts"]).any()
        assert set(f["round"].unique()) <= {10, 17}
    base_features = list(official.FEATS)
    extra = [f"{c}_round_delta" for c in ["wa_cloud", "wb_cloud", "wb_uv_idx", "wa_temp", "wb_temp"]]
    extra += [f"{c}_mean{w}" for c in ["wa_cloud", "wb_cloud", "wb_uv_idx"] for w in [3, 7]]
    extra += ["cloud_provider_delta", "temp_provider_delta", "forecast_missing_count"]
    feature_audit = []
    for month in pd.period_range("2022-01", "2025-05", freq="M"):
        path = cache/f"{month}.parquet"
        left = month.start_time.tz_localize(TZ)+pd.Timedelta(hours=1)
        right = (month+1).start_time.tz_localize(TZ)
        ix = pd.date_range(left, right, freq="h")
        a = fa.loc[period_mask(fa.ts, left, right)]
        b = fb.loc[period_mask(fb.ts, left, right)]
        frames = []
        for r in [10, 17]:
            d = official.features(None, sites, a, b, ix, r)
            frames.append(d[["site_id", "ts"]+base_features].sort_values(["site_id", "ts"]).reset_index(drop=True))
        d10, d = frames
        assert d10[["site_id", "ts"]].equals(d[["site_id", "ts"]])
        raw_weather = official.WA_COLS+official.WB_COLS
        d["forecast_missing_count"] = d[raw_weather].isna().sum(axis=1).astype("float32")
        # Correct B wind from km/h to m/s for the enhanced branch only. Keep
        # baseline input columns in their official units for exact parity.
        for c in ["wa_cloud", "wb_cloud", "wb_uv_idx", "wa_temp", "wb_temp"]:
            d[f"{c}_round_delta"] = d[c]-d10[c]
        assert len(d) % 24 == 0 and (d.ts.iloc[::24].dt.hour == 1).all()
        for c in ["wa_cloud", "wb_cloud", "wb_uv_idx"]:
            for w in [3, 7]:
                # Same forecast issue and same target day only. These are
                # forecast trajectories, never realized future weather.
                d[f"{c}_mean{w}"] = forecast_day_mean(d[c].to_numpy(), w)
        d["cloud_provider_delta"] = d.wa_cloud-d.wb_cloud
        d["temp_provider_delta"] = d.wa_temp-d.wb_temp
        for c in base_features:
            d[f"r10__{c}"] = d10[c].to_numpy()
        labels = gen.loc[period_mask(gen.ts, left, right), ["site_id", "ts", "energy"]]
        d = d.merge(labels, on=["site_id", "ts"], how="left", validate="one_to_one")
        numeric = d.columns.difference(["site_id", "ts"])
        d[numeric] = d[numeric].astype("float32")
        assert not np.isinf(d[numeric].to_numpy()).any()
        d.to_parquet(path, index=False, compression="zstd")
        feature_audit.append({"month": str(month), "grid_rows": len(d), "label_rows": int(d.energy.notna().sum()),
                              "missing_weather_cells": int(d[raw_weather].isna().sum().sum())})
        if month.month in [6, 12] or month == pd.Period("2025-05"):
            log(f"Feature cache ready through {month}")
    save_json(out/"feature_audit.json", feature_audit)
    save_json(out/"features.json", {"baseline": base_features, "enhanced": base_features+extra})
    del fa, fb
    gc.collect()
    return base_features, base_features+extra


def load_rows(out, start, end, labeled=False):
    months = pd.period_range((start-pd.Timedelta(hours=1)).tz_localize(None).to_period("M"),
                            (end-pd.Timedelta(hours=1)).tz_localize(None).to_period("M"), freq="M")
    frames = []
    for m in months:
        d = pd.read_parquet(out/"features"/f"{m}.parquet")
        d = d.loc[period_mask(d.ts, start, end)]
        if labeled:
            d = d.loc[d.energy.notna() & (d.elev > 0)]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def design(d, features, enhanced=False, round_=17):
    cols = features if round_ == 17 else [f"r10__{c}" for c in features]
    a = d[cols].to_numpy(dtype="float32", copy=True)
    if enhanced:
        a[:, features.index("wb_wind_speed")] /= 3.6
    return a


def fit_baseline(d, features, out, prefix):
    models = []
    for r in [10, 17]:
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.06, num_leaves=127,
            min_child_samples=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, verbose=-1, n_jobs=THREADS, random_state=SEED)
        m.fit(design(d, features, round_=r), (d.energy/d.cap).clip(0, 1.2).to_numpy())
        m.booster_.save_model(str(out/"models"/f"{prefix}_r{r}.txt"))
        models.append(m)
        log(f"{prefix} baseline r{r} fitted on {len(d):,} daylight labels")
    return models


def fit_xgb(d, features, params, out=None, prefix=None):
    m = xgb.XGBRegressor(**params, tree_method="hist", objective="reg:squarederror",
                        n_jobs=THREADS, random_state=SEED)
    m.fit(design(d, features, enhanced=True), (d.energy/d.cap).clip(0, 1.2).to_numpy())
    if out is not None:
        m.save_model(out/"models"/f"{prefix}.ubj")
    return m


def predict(model, d, features, kind):
    p = np.zeros(len(d), dtype="float64")
    day = (d.elev > 0).to_numpy()
    daylight = d.loc[day]
    if kind == "baseline":
        values = [np.maximum(m.booster_.predict(design(daylight, features, round_=r), num_threads=THREADS), 0)
                  for m, r in zip(model, [10, 17])]
        p[day] = np.mean(values, axis=0)*daylight.cap.to_numpy()
    else:
        p[day] = np.maximum(model.predict(design(daylight, features, enhanced=True)), 0)*daylight.cap.to_numpy()
    assert np.isfinite(p).all() and (p >= 0).all()
    return p


def observed_proxy(d, prediction, groups, prices):
    # Matched observed subset ONLY. This is not the 253-site competition score.
    z = d.loc[d.energy.notna(), ["site_id", "ts", "energy"]].copy()
    z["prediction"] = prediction[d.energy.notna().to_numpy()]
    z["group"] = z.site_id.map(groups)
    g = z.groupby(["group", "ts"])[["energy", "prediction"]].sum().reset_index()
    g = g.merge(prices, on="ts", how="left", validate="many_to_one")
    return penalty(g.energy, g.prediction, g.smp, g.k)


def ga_search(train, valid, features, groups, prices, out):
    # Pilot budget is deliberately smaller than the paper's 20 x 30 search.
    space = {"n_estimators": [100, 180, 260], "max_depth": [4, 6, 8],
             "learning_rate": [0.03, 0.06, 0.10], "subsample": [0.7, 0.9, 1.0],
             "colsample_bytree": [0.7, 0.9, 1.0]}
    rng = np.random.default_rng(SEED)
    keys = list(space)
    population = [(1, 1, 1, 1, 1), (2, 0, 1, 2, 1), (1, 2, 0, 1, 0), (0, 1, 2, 0, 2)]
    cache = {}
    history = []
    for generation in range(3):
        for chromosome in population:
            if chromosome in cache:
                continue
            params = {k: space[k][v] for k, v in zip(keys, chromosome)}
            m = fit_xgb(train, features, params)
            pred = predict(m, valid, features, "xgb")
            score = observed_proxy(valid, pred, groups, prices)
            cache[chromosome] = score["penalty_krw"]
            history.append({"generation": generation, "params": params, "proxy": score})
            save_json(out/"ga_history.json", history)
            log(f"GA generation {generation+1}, candidate {len(cache)}: observed-panel penalty {cache[chromosome]:,.0f}")
            del m
        ranked = sorted(population, key=lambda c: cache[c])
        parents = ranked[:2]
        population = parents.copy()
        while len(population) < 4:
            child = tuple(parents[int(rng.integers(2))][i] if rng.random() > 0.25 else int(rng.integers(3)) for i in range(5))
            if child not in population:
                population.append(child)
    best = min(cache, key=cache.get)
    return {k: space[k][v] for k, v in zip(keys, best)}, min(cache.values())


def optimize_groups(d, prediction, sites, prices, out):
    # Only optimize a stable, observed historical cohort; new sites stay in
    # their provided groups. Missing residuals are never treated as zero.
    z = d[["site_id", "ts", "energy"]].copy()
    z["residual"] = prediction-z.energy
    ntime = z.ts.nunique()
    coverage = z.groupby("site_id").energy.count()/ntime
    eligible = coverage[coverage >= 0.95].index.tolist()
    residual = z.pivot(index="ts", columns="site_id", values="residual")[eligible].dropna()
    ref = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
    groups = ref.copy()
    capacities = sites.set_index("site_id").capacity_kw
    r = residual.to_numpy(dtype=float)
    w = prices.set_index("ts").loc[residual.index]
    weights = (w.smp*w.k).to_numpy()
    assignments = (groups.loc[eligible] == "G1").to_numpy(copy=True)
    total = r.sum(axis=1)
    s1 = r[:, assignments].sum(axis=1)
    cost = lambda v: float(np.sum(weights*(np.abs(v)+np.abs(total-v))))
    before = cost(s1)
    best = before
    cap1 = float(capacities[groups == "G1"].sum())
    total_cap = float(capacities.sum())
    rng = np.random.default_rng(SEED)
    accepted = 0
    if len(eligible) >= 2 and len(residual) >= 168:
        for _ in range(5000):
            left, right = np.flatnonzero(assignments), np.flatnonzero(~assignments)
            if not len(left) or not len(right):
                break
            i, j = int(rng.choice(left)), int(rng.choice(right))
            new_cap = cap1-capacities[eligible[i]]+capacities[eligible[j]]
            if min(new_cap, total_cap-new_cap) < 20000:
                continue
            new_s1 = s1-r[:, i]+r[:, j]
            new_cost = cost(new_s1)
            if new_cost < best-1e-6:
                assignments[i], assignments[j] = False, True
                best, s1, cap1 = new_cost, new_s1, new_cap
                accepted += 1
    groups.loc[eligible] = np.where(assignments, "G1", "G2")
    info = {"cohort_sites": len(eligible), "complete_cohort_hours": len(residual),
            "accepted_swaps": accepted, "proxy_penalty_before": before*0.1,
            "proxy_penalty_after": best*0.1,
            "limitation": "Only 2023 observed stable cohort; unobserved sites stay at reference assignments. Not full-fleet optimality."}
    save_json(out/"group_search.json", info)
    return ref, groups, eligible, info


def cohort_score(d, prediction, groups, eligible, prices):
    z = d[["site_id", "ts", "energy"]].copy()
    z["prediction"] = prediction
    z = z.loc[z.site_id.isin(eligible)]
    count = z.groupby("ts").energy.count()
    z = z.loc[z.ts.isin(count[count == len(eligible)].index)]
    return observed_proxy(z, z.prediction.to_numpy(), groups, prices)


def evaluate(predictions, gen, sites, groups, prices, out):
    truth = gen.loc[period_mask(gen.ts, TEST_START, END), ["site_id", "ts", "energy"]]
    merged = predictions.merge(truth, on=["site_id", "ts"], how="left", validate="one_to_one")
    models = list(groups)
    rows, monthly, daily = [], [], []
    for name in models:
        for label, begin, end in [("all", TEST_START, END),
                                  ("2024", TEST_START, pd.Timestamp("2025-01-01 00:00", tz=TZ)),
                                  ("2025_Jan_May", pd.Timestamp("2025-01-01 01:00", tz=TZ), END)]:
            z = merged.loc[period_mask(merged.ts, begin, end)].copy()
            valid = z.energy.notna()
            err = z.loc[valid, name]-z.loc[valid, "energy"]
            cap = z.loc[valid, "site_id"].map(sites.set_index("site_id").capacity_kw)
            rows.append({"model": name, "period": label, "scope": "observed_site_rows", "rows": int(valid.sum()),
                         "mae_kwh": float(err.abs().mean()), "rmse_kwh": float(np.sqrt((err**2).mean())),
                         "capacity_nmae_pct": float(100*(err.abs()/cap).mean())})
            count = z.groupby("ts").energy.count()
            complete = count[count == 253].index
            z = z.loc[z.ts.isin(complete)].copy()
            z["group"] = z.site_id.map(groups[name])
            g = z.groupby(["group", "ts"])[["energy", name]].sum().reset_index()
            g = g.merge(prices, on="ts", validate="many_to_one")
            score = penalty(g.energy, g[name], g.smp, g.k)
            rows.append({"model": name, "period": label, "scope": "complete_253_panel",
                         "hours": len(complete), "group_hours": len(g), **score})
            if label == "all":
                g["day"] = (g.ts-pd.Timedelta(hours=1)).dt.strftime("%Y-%m-%d")
                for day, v in g.groupby("day"):
                    daily.append({"model": name, "day": day, "hours": v.ts.nunique(), **penalty(v.energy, v[name], v.smp, v.k)})
                g["month"] = g.day.str[:7]
                for month, v in g.groupby("month"):
                    monthly.append({"model": name, "month": month, "hours": v.ts.nunique(), **penalty(v.energy, v[name], v.smp, v.k)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out/"metrics.csv", index=False)
    pd.DataFrame(monthly).to_csv(out/"monthly_scores.csv", index=False)
    pd.DataFrame(daily).to_csv(out/"daily_scores.csv", index=False)
    save_json(out/"metrics.json", rows)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for previous_end, next_start in [(TUNE_END, CAL_START), (CAL_END, CHECK_START), (CHECK_END, TEST_START)]:
        if previous_end > issue_deadline([next_start])[0]:
            raise ValueError("Selection uses data after the next forecast deadline")
    data, out = args.data.resolve(), args.out.resolve()
    if "onedrive" in str(out).lower():
        raise ValueError("Use a local result directory outside OneDrive")
    out.mkdir(parents=True, exist_ok=True)
    if (out/"frozen_plan.json").exists():
        raise ValueError("Existing frozen run: use a new output directory to preserve results")
    (out/"models").mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location("official_baseline", data/"baseline.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    sites = pd.read_csv(data/"sites.csv")
    gen = pd.read_parquet(data/"train/generation.parquet")
    gen["ts"] = pd.to_datetime(gen.ts)
    audit(data, sites, gen, out)
    prices = pd.read_csv(data/"train/smp.csv")
    prices["ts"] = pd.to_datetime(prices.ts)
    prices["k"] = np.where(prices.ts.dt.hour.between(11, 15), 3, 1)
    assert not prices.ts.duplicated().any()
    fbase, fxgb = build_features(data, out, sites, gen, official)
    ref_groups = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
    train = load_rows(out, START, issue_deadline([TUNE_START])[0], labeled=True)
    valid = load_rows(out, TUNE_START, TUNE_END)
    baseline = fit_baseline(train, fbase, out, "tune_2022")
    base_score = observed_proxy(valid, predict(baseline, valid, fbase, "baseline"), ref_groups, prices)
    params, ga_cost = ga_search(train, valid, fxgb, ref_groups, prices, out)
    chosen = "xgb" if ga_cost < base_score["penalty_krw"] else "baseline"
    log(f"2023 H1 selection: {chosen}; baseline {base_score['penalty_krw']:,.0f}, GA-XGB {ga_cost:,.0f}")
    save_json(out/"model_selection.json", {"chosen": chosen, "ga_params": params,
        "baseline_proxy": base_score, "xgb_proxy_penalty": ga_cost,
        "selection_period": [str(TUNE_START), str(TUNE_END)]})
    del train, valid, baseline
    gc.collect()
    dev = load_rows(out, START, issue_deadline([CAL_START])[0], labeled=True)
    selected_features = fxgb if chosen == "xgb" else fbase
    model = fit_xgb(dev, fxgb, params, out, "group_development") if chosen == "xgb" else fit_baseline(dev, fbase, out, "group_development")
    del dev
    calibration = load_rows(out, CAL_START, CAL_END)
    p = predict(model, calibration, selected_features, chosen)
    ref, candidate, eligible, group_info = optimize_groups(calibration, p, sites, prices, out)
    del calibration, p
    check = load_rows(out, CHECK_START, CHECK_END)
    p = predict(model, check, selected_features, chosen)
    ref_check = cohort_score(check, p, ref, eligible, prices)
    candidate_check = cohort_score(check, p, candidate, eligible, prices)
    accept = candidate_check["penalty_krw"] < 0.99*ref_check["penalty_krw"]
    selected_groups = candidate if accept else ref
    save_json(out/"group_validation.json", {"accepted": accept, "required_improvement_pct": 1,
        "reference": ref_check, "candidate": candidate_check,
        "period": [str(CHECK_START), str(CHECK_END)]})
    candidate.rename("group").reset_index().to_csv(out/"composition_candidate.csv", index=False)
    selected_groups.rename("group").reset_index().to_csv(out/"composition_selected.csv", index=False)
    capacities = sites.set_index("site_id").capacity_kw.groupby(selected_groups).sum()
    assert capacities.min() >= 20000 and set(selected_groups) == {"G1", "G2"}
    assert FULL_CUTOFF <= issue_deadline([TEST_START])[0]
    frozen = {"chosen_model": chosen, "xgb_params": params, "group_candidate_accepted": accept,
        "group_capacity_kw": capacities.to_dict(), "fit_cutoff": str(FULL_CUTOFF),
        "test_period": [str(TEST_START), str(END)], "seed": SEED,
        "input_features": selected_features, "generation_lags_used": False,
        "observed_weather_used": False, "smp_used_as_prediction_feature": False,
        "sro": "Symmetric interval absolute-loss robustness does not change the optimal weighted median. No arbitrary down-bid or ESS model applied; full-panel 2023 calibration unavailable.",
        "selection_policy": "All model/group decisions use only pre-2024 labels. Final comparison does not change frozen choice.",
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "lightgbm": lgb.__version__, "xgboost": xgb.__version__, "pyarrow": pyarrow.__version__},
        "script_sha256": sha256(__file__)}
    save_json(out/"frozen_plan.json", frozen)
    log(f"Plan frozen before holdout scoring. Group candidate accepted={accept}")
    del check, p, model
    gc.collect()
    train = load_rows(out, START, FULL_CUTOFF, labeled=True)
    baseline = fit_baseline(train, fbase, out, "final_2022_2023")
    enhanced = fit_xgb(train, fxgb, params, out, "final_ga_xgb_2022_2023")
    log("Final GA-XGBoost fitted")
    del train
    gc.collect()
    predictions = []
    for month in pd.period_range("2024-01", "2025-05", freq="M"):
        d = pd.read_parquet(out/"features"/f"{month}.parquet")
        # No labels passed to predict().
        d = d.drop(columns="energy")
        z = d[["site_id", "ts"]].copy()
        z["baseline_reference"] = predict(baseline, d, fbase, "baseline")
        z["ga_xgb_reference"] = predict(enhanced, d, fxgb, "xgb")
        z["frozen_selected"] = z["ga_xgb_reference" if chosen == "xgb" else "baseline_reference"]
        predictions.append(z)
        log(f"Holdout forecasts completed: {month}")
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_parquet(out/"site_predictions.parquet", index=False, compression="zstd")
    group_map = {"baseline_reference": ref, "ga_xgb_reference": ref, "frozen_selected": selected_groups}
    forecast = predictions.assign(group=predictions.site_id.map(selected_groups)).groupby(["group", "ts"]).frozen_selected.sum().rename("forecast_kwh").reset_index()
    forecast.to_csv(out/"backtest_forecast_NOT_FOR_SUBMISSION.csv", index=False)
    assert not forecast.duplicated(["group", "ts"]).any()
    assert len(forecast) == 2*len(pd.date_range(TEST_START, END, freq="h"))
    metrics = evaluate(predictions, gen, sites, group_map, prices, out)
    print(metrics.loc[metrics.scope == "complete_253_panel", ["model", "period", "hours", "penalty_krw", "settlement_krw"]].to_string(index=False), flush=True)
    save_json(out/"completion.json", {"completed": True, "elapsed_seconds": time.time()-T0,
        "predicted_site_rows": len(predictions), "backtest_group_rows": len(forecast),
        "no_submission": True, "forecast_sha256": sha256(out/"site_predictions.parquet")})
    log(f"Complete. Results: {out}")


if __name__ == "__main__":
    main()
