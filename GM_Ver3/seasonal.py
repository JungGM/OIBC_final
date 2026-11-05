"""Multi-year chronological backtests using observed site-hours only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .fit import MODELS, dump_json, fit_predict, input_files, start_at
from .settlement import issue_cutoff, price_arrays


SEASONS = {12: "winter", 3: "spring", 6: "summer", 9: "autumn"}
QUARTERS = tuple(pd.date_range("2021-12-01", "2024-12-01", freq="3MS"))


def site_metrics(actual, forecast, capacity, observed):
    error = (forecast - actual)[observed]
    denominator = float(observed.sum(axis=0) @ capacity)
    return {"site_mae_kwh": float(np.abs(error).mean()), "site_rmse_kwh": float(np.sqrt(np.square(error).mean())),
            "site_mbe_kwh": float(error.mean()), "nmae_pct": float(100 * np.abs(error).sum(dtype=np.float64) / denominator)}


def seasonal_summary(folds):
    summary = {}
    for name in MODELS:
        seasons = {}
        for season in SEASONS.values():
            use = [fold for fold in folds.values() if fold["season"] == season]
            if not use:
                continue
            values = [fold["models"][name]["nmae_pct"] for fold in use]
            ratios = [fold["models"][name]["nmae_pct"] / max(1e-12, fold["models"]["climatology"]["nmae_pct"]) for fold in use]
            seasons[season] = {"mean_nmae_pct": float(np.mean(values)), "relative_to_climatology": float(np.mean(ratios)),
                               "folds": len(use)}
        ratios = [season["relative_to_climatology"] for season in seasons.values()]
        summary[name] = {"seasons": seasons, "selection_index": float(.7 * np.mean(ratios) + .3 * max(ratios))}
    return summary


def correlation_stability(folders, fold_metrics):
    pairs = []
    for a, b in zip(folders, folders[1:]):
        first = np.load(a / "lgbm_full_correlations.npy")
        second = np.load(b / "lgbm_full_correlations.npy")
        rows, columns = np.triu_indices_from(first, k=1)
        one, two = first[rows, columns], second[rows, columns]
        valid = np.isfinite(one) & np.isfinite(two)
        if valid.any():
            pairs.append({"from": a.name, "to": b.name, "site_pairs": int(valid.sum()),
                          "median_abs_change": float(np.median(np.abs(one[valid] - two[valid])))})
    same_season = []
    for a, b in zip(folders, folders[4:]):
        if fold_metrics[a.name]["season"] != fold_metrics[b.name]["season"]:
            continue
        first = np.load(a / "lgbm_full_correlations.npy")
        second = np.load(b / "lgbm_full_correlations.npy")
        valid = np.triu(np.isfinite(first) & np.isfinite(second), k=1)
        if valid.any():
            same_season.append({"from": a.name, "to": b.name, "site_pairs": int(valid.sum()),
                                "median_abs_change": float(np.median(np.abs(first[valid] - second[valid])))})
    return {"adjacent_quarters": pairs, "same_season_next_year": same_season}


def audit(raw, out, threads):
    sites, weights, _ = input_files(raw)
    times, smp, k = price_arrays(weights)
    capacity = sites.capacity_kw.to_numpy(dtype=np.float32)
    x = np.load(out / "train_x.npy", mmap_mode="r")
    y = np.load(out / "train_y.npy", mmap_mode="r")
    names = json.loads((out / "features.json").read_text(encoding="utf-8"))
    directory = out / "seasonal"
    directory.mkdir(exist_ok=True)
    folds = {}
    folders = []
    for day, following in zip(QUARTERS, QUARTERS[1:]):
        first, end = start_at(day.date()), start_at(following.date())
        cutoff = issue_cutoff([first])[0]
        fit_end = times.searchsorted(cutoff, side="right")
        held = slice(times.searchsorted(first), times.searchsorted(end))
        folder = directory / day.strftime("%Y-%m")
        folders.append(folder)
        report_path = folder / "coverage_metrics.json"
        if report_path.exists():
            folds[folder.name] = json.loads(report_path.read_text(encoding="utf-8"))
            continue
        if not (folder / "protocol.json").exists():
            fit_predict(x[:fit_end], y[:fit_end], times[:fit_end], x[held], times[held], (smp * k)[:fit_end],
                        capacity, names, folder, threads)
        actual = y[held]
        observed = np.isfinite(actual) & (actual >= 0)
        results = {}
        for name in MODELS:
            forecast = np.load(folder / f"{name}.npy")
            results[name] = site_metrics(actual, forecast, capacity, observed)
            if name == "lgbm_full":
                residual = np.where(observed, (forecast - actual) / capacity, np.nan)
                np.save(folder / "lgbm_full_correlations.npy", pd.DataFrame(residual).corr(min_periods=168).to_numpy())
        folds[folder.name] = {"season": SEASONS[day.month], "cutoff": str(cutoff), "valid_site_hours": int(observed.sum()),
                              "sites_with_observations": int(observed.any(axis=0).sum()),
                              "complete_fleet_hours": int(observed.all(axis=1).sum()),
                              "observed_capacity_fraction": float(observed.sum(axis=0) @ capacity / (len(actual) * capacity.sum())),
                              "models": results}
        dump_json(report_path, folds[folder.name])
        print(f"{folder.name}: {folds[folder.name]['valid_site_hours']:,} observed site-hours", flush=True)
    summary = seasonal_summary(folds)
    rank = sorted(summary, key=lambda name: summary[name]["selection_index"])
    dump_json(directory / "summary.json", {"folds": folds, "models": summary, "ranking": rank,
                                           "residual_correlation": correlation_stability(folders, folds)})
    print(f"Seasonal ranking: {rank}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    raw, out = args.raw.resolve(), args.out.resolve()
    if out.is_relative_to(raw) or out.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Place all outputs outside the repository and participant package")
    with threadpool_limits(limits=args.threads):
        audit(raw, out, args.threads)
