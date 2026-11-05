"""Score site forecasts on complete historical hours under OIBC settlement rules."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ["G1", "G2"]


def kst_hours(values):
    times = pd.DatetimeIndex(values).as_unit("ns")
    if (str(times.tz) not in ("Asia/Seoul", "UTC+09:00") or times.hasnans
            or not times.equals(times.floor("h"))):
        raise ValueError("Expected hourly timestamps in KST")
    return times.tz_convert("Asia/Seoul")


def issue_cutoff(values):
    """Midnight is the closing hour of the preceding operating day."""
    return (kst_hours(values) - pd.Timedelta(hours=1)).normalize() - pd.Timedelta(hours=7)


def group_capacities(sites, composition):
    if (sites.site_id.isna().any() or not sites.site_id.is_unique
            or not np.isfinite(sites.capacity_kw).all() or not sites.capacity_kw.gt(0).all()):
        raise ValueError("Invalid site roster")
    if (not composition.site_id.is_unique or set(composition.site_id) != set(sites.site_id)
            or composition.group.isna().any() or set(composition.group) != set(GROUPS)):
        raise ValueError("Each site must belong to exactly one of G1 and G2")
    capacities = sites.set_index("site_id").capacity_kw.groupby(
        composition.set_index("site_id").group).sum().reindex(GROUPS)
    if capacities.lt(20_000).any():
        raise ValueError("Both groups require at least 20,000 kW")
    return capacities


def site_matrix(frame, value, times, site_ids):
    frame = frame.loc[:, ["ts", "site_id", value]].copy()
    frame["ts"] = kst_hours(frame.ts)
    if not frame.site_id.isin(site_ids).all() or frame.duplicated(["ts", "site_id"]).any():
        raise ValueError(f"Unknown site or duplicate site-hour in {value}")
    return frame.pivot(index="ts", columns="site_id", values=value).reindex(
        index=times, columns=site_ids).to_numpy(dtype=np.float64)


def price_arrays(weights):
    times = kst_hours(weights.index)
    if not len(times) or not times.is_unique:
        raise ValueError("Scoring requires a nonempty, unique hourly price index")
    smp = weights.smp.to_numpy(dtype=np.float64)
    if not np.isfinite(smp).all() or (smp < 0).any():
        raise ValueError("SMP must be finite and nonnegative")
    expected_k = np.where(times.hour.isin(range(11, 16)), 3, 1)
    k = weights.k.to_numpy() if "k" in weights else expected_k
    if not np.array_equal(k, expected_k):
        raise ValueError("Invalid KST closing-hour weights")
    return times, smp, k


def score(sites, composition, actual, forecast, weights):
    """Compute W and S over hours with valid labels for the entire roster."""
    times, _, _ = price_arrays(weights)
    site_ids = sites.site_id
    pred = site_matrix(forecast, "prediction", times, site_ids)
    truth = site_matrix(actual, "energy", times, site_ids)
    return score_arrays(sites, composition, truth, pred, weights)


def score_arrays(sites, composition, truth, pred, weights):
    capacities = group_capacities(sites, composition)
    times, smp, k = price_arrays(weights)
    if pred.shape != (len(times), len(sites)) or truth.shape != pred.shape:
        raise ValueError("Actual and forecast matrices must match the price grid and roster")
    if not np.isfinite(pred).all() or (pred < 0).any():
        raise ValueError("Every requested site-hour requires a finite nonnegative forecast")
    if np.isinf(truth).any():
        raise ValueError("Actual energy contains infinite values")
    complete = (np.isfinite(truth) & (truth >= 0)).all(axis=1)
    if not complete.any():
        raise ValueError("No hours with valid actual energy for every site")
    truth, pred = truth[complete], pred[complete]
    error = pred - truth
    group = composition.set_index("site_id").group.reindex(sites.site_id).to_numpy()
    membership = (group[:, None] == np.asarray(GROUPS)).astype(float)
    mae = float(np.abs(error).mean())
    return {
        "hours_requested": len(times),
        "hours_scored": int(complete.sum()),
        "hours_excluded_invalid_actual": int((~complete).sum()),
        "group_capacity_kw": capacities.to_dict(),
        **group_settlement(truth @ membership, pred @ membership, smp[complete], k[complete]),
        "site_mae_kwh": mae,
        "site_rmse_kwh": float(np.sqrt(np.square(error).mean())),
        "site_mbe_kwh": float(error.mean()),
        "site_nmae_pct_of_mean_capacity": 100 * mae / float(sites.capacity_kw.mean()),
    }


def group_settlement(truth, forecast, smp, k):
    """Score two aligned group series on a validated, complete grid."""
    group_absolute_error = np.abs(forecast - truth)
    by_group = (0.1 * smp * k) @ group_absolute_error
    revenue = float(smp @ truth.sum(axis=1))
    penalty = float(by_group.sum())
    return {"revenue_krw": revenue, "penalty_krw": penalty, "settlement_krw": revenue - penalty,
            "penalty_by_group_krw": dict(zip(GROUPS, by_group.tolist())),
            "group_absolute_error_kwh": float(group_absolute_error.sum()),
            "peak_penalty_krw": float(0.1 * (smp[k == 3] * k[k == 3]) @ group_absolute_error[k == 3].sum(axis=1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=Path, required=True, help="CSV with site_id and capacity_kw")
    parser.add_argument("--composition", type=Path, required=True, help="CSV with site_id and group")
    parser.add_argument("--smp", type=Path, required=True, help="CSV with ts, smp and optional k for the scoring window")
    parser.add_argument("--actual", type=Path, required=True, help="Parquet with ts, site_id and energy")
    parser.add_argument("--predictions", type=Path, required=True, help="Parquet with ts, site_id and prediction")
    args = parser.parse_args()

    sites = pd.read_csv(args.sites, dtype={"site_id": str})
    composition = pd.read_csv(args.composition, dtype={"site_id": str})
    weights = pd.read_csv(args.smp)
    weights.index = kst_hours(pd.to_datetime(weights.pop("ts")))
    forecast = pd.read_parquet(args.predictions, columns=["ts", "site_id", "prediction"])
    actual = pd.read_parquet(args.actual, columns=["ts", "site_id", "energy"])
    print(json.dumps(score(sites, composition, actual, forecast, weights), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
