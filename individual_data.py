"""Right-closed site-day panels and observed-label evaluation."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from backtest import START, TEST_START, FULL_CUTOFF, TZ, penalty, sha256


@dataclass
class Panel:
    x: np.ndarray        # site-days, 24 hourly forecasts, features
    y: np.ndarray        # raw kWh, NaN outside period or if unobserved
    cap: np.ndarray
    site: np.ndarray
    first: pd.DatetimeIndex

    @property
    def observed(self):
        return np.isfinite(self.y)

    def records(self, predictions):
        day, hour = np.where(self.observed)
        pred = predictions[day, hour].astype(float)
        if not np.isfinite(pred).all() or (pred < 0).any():
            raise ValueError("Invalid forecast")
        return pd.DataFrame(dict(site_id=self.site[day],
            ts=self.first[day] + pd.to_timedelta(hour, unit="h"),
            actual=self.y[day, hour], prediction=pred, capacity_kw=self.cap[day]))


class Dataset:
    def __init__(self, data, cache):
        self.data, self.cache = data, cache
        manifest = json.loads((cache / "manifest.json").read_text(encoding="utf8"))
        self.feature_dir = cache
        audit = json.loads((Path(manifest["source_cache"]) / "audit.json").read_text(encoding="utf8"))
        for name, digest in audit["source_sha256"].items():
            if sha256(data / name.replace(chr(92), "/")) != digest:
                raise ValueError(f"Source mismatch: {name}")
        self.features = manifest["features"]
        self.sites = pd.read_csv(data / "sites.csv").set_index("site_id", verify_integrity=True)
        self.prices = pd.read_csv(data / "train/smp.csv")
        self.prices["ts"] = pd.to_datetime(self.prices.ts)
        self.prices = self.prices.set_index("ts", verify_integrity=True)

    def panel(self, start, end, *, holdout=False):
        if not holdout and end >= TEST_START or holdout and start < TEST_START:
            raise ValueError("Period crosses frozen test boundary")
        months = pd.period_range((start - pd.Timedelta(hours=1)).strftime("%Y-%m"),
                                 (end - pd.Timedelta(hours=1)).strftime("%Y-%m"), freq="M")
        parts = []
        for m in months:
            left = max(start, m.start_time.tz_localize(TZ) + pd.Timedelta(hours=1))
            right = min(end, (m+1).start_time.tz_localize(TZ))
            truth = pd.read_parquet(self.data / "train/generation.parquet",
                filters=[("ts", ">=", left), ("ts", "<=", right)], columns=["site_id","ts","energy"])
            grid = pd.read_parquet(self.feature_dir / f"{m}.parquet",
                                   columns=["site_id","ts"] + self.features)
            grid = grid.sort_values(["site_id","ts"]).merge(truth, how="left",
                on=["site_id","ts"], validate="one_to_one", sort=False)
            ts = pd.DatetimeIndex(grid.ts)
            first = ts[::24]
            if len(grid) != 253*m.days_in_month*24 or not (first.hour == 1).all():
                raise ValueError("Incomplete forecast grid")
            if not np.all(ts == first.repeat(24) + pd.to_timedelta(np.tile(np.arange(24), len(first)), unit="h")):
                raise ValueError("Non-contiguous day")
            sites = grid.site_id.to_numpy().reshape(-1,24)
            if not (sites == sites[:, :1]).all():
                raise ValueError("Mixed sites in one sequence")
            y = grid.energy.to_numpy(dtype=float, copy=True).reshape(-1,24)
            allowed = np.asarray((ts >= left) & (ts <= right)).reshape(-1,24)
            y[~allowed] = np.nan
            keep = np.isfinite(y).any(axis=1)
            ids = sites[keep, 0]
            x = grid[self.features].to_numpy(dtype="float32", copy=True).reshape(-1,24,len(self.features))
            parts.append((x[keep], y[keep], self.sites.capacity_kw.loc[ids].to_numpy(),
                          ids, first[keep]))
        return Panel(*(np.concatenate(v) if i != 4 else pd.DatetimeIndex(np.concatenate(v))
                       for i,v in enumerate(zip(*parts))))


def site_scores(rows):
    err = rows.prediction.to_numpy() - rows.actual.to_numpy()
    return dict(rows=len(err), hours=rows.ts.nunique(), MAE=float(np.abs(err).mean()),
                RMSE=float(np.sqrt(np.mean(err**2))), MBE=float(err.mean()),
                nMAE_pct=float(100*np.mean(np.abs(err)/rows.capacity_kw)))


def group_scores(rows, sites, prices):
    """Common complete 253-site hours only. Never invent missing truth or SMP."""
    missing = dict(W=None, S=None, R=None, hours=0)
    if (len(sites) != 253 or not sites.index.is_unique or sites.current_group.isna().any()
        or set(sites.current_group) != {"A","B"}
        or sites.groupby("current_group").capacity_kw.sum().min() < 20000):
        return dict(**missing, reason="missing or invalid fixed composition")
    if rows.duplicated(["site_id","ts"]).any() or not rows.site_id.isin(sites.index).all():
        raise ValueError("Duplicate or unknown site")
    complete = rows.groupby("ts").size().loc[lambda n: n == len(sites)].index
    if not len(complete):
        return dict(**missing, reason="no complete 253-site hours")
    g = rows.loc[rows.ts.isin(complete)].copy()
    g["group"] = g.site_id.map(sites.current_group)
    group = g.groupby(["ts","group"])[["prediction","actual"]].sum().reset_index()
    smp = group.ts.map(prices.smp)
    if smp.isna().any():
        return dict(**missing, reason="incomplete SMP")
    k = np.where(group.ts.dt.hour.between(11,15), 3, 1)
    score = penalty(group.actual, group.prediction, smp, k)
    return dict(hours=len(complete), R=score["revenue_krw"], W=score["penalty_krw"],
                S=score["settlement_krw"], group_MAE=score["mae_kwh"], group_RMSE=score["rmse_kwh"])
