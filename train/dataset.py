"""Memory-bounded training access and complete-panel group settlement."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from audit_data import TZ, cutoff, digest


def read_manifest(directory, verify=False):
    directory = Path(directory)
    m = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if verify:
        for r in m["summary"].values():
            if digest(directory/r["file"]) != r["sha256"]:
                raise ValueError("Frozen dataset SHA256 changed")
        if "ready_training" in m and digest(directory/m["ready_training"]["file"]) != m["ready_training"]["sha256"]:
            raise ValueError("Ready training dataset SHA256 changed")
    return m


def current_dataset():
    """Resolve the locally frozen dataset, checking the repository's public lock."""
    root = Path(__file__).parent
    local = json.loads((root/".dataset.local.json").read_text(encoding="utf-8"))
    lock = json.loads((root/"dataset_lock.json").read_text(encoding="utf-8"))
    if local["ready_sha256"] != lock["ready_training"]["sha256"]:
        raise ValueError("Local pointer does not match fixed dataset lock")
    return Path(local["directory"])


def iter_training_batches(directory, *, features=None, start=None, end=None, batch_size=65536,
                          verify=True):
    """Yield (X, y, keys); y is raw kWh, negative/missing labels excluded.

    No imputation/normalization is fitted here. Fit transformations on a training
    fold only. Use iter_fold for model selection, not a random row split.
    """
    directory = Path(directory)
    m = read_manifest(directory, verify=verify)
    features = m["default_features"] if features is None else list(features)
    if not features or len(features) != len(set(features)) or set(features)-set(m["features"]):
        raise ValueError("X must use unique declared model features only")
    filt = ds.field("target_eligible") == True  # noqa: E712
    for op, value in ((">=", start), ("<=", end)):
        if value is not None:
            value = pd.Timestamp(value)
            if value.tz is None:
                raise ValueError("Fold boundaries must have KST timezone")
            value = value.tz_convert(TZ).as_unit("ns")
            filt = filt & ((ds.field("ts") >= value) if op == ">=" else (ds.field("ts") <= value))
    training_file = m.get("ready_training", m["summary"]["training"])["file"]
    scanner = ds.dataset(directory/training_file, format="parquet").scanner(
        columns=["site_id", "ts", "issue_cutoff", "energy_kwh"]+features,
        filter=filt, batch_size=batch_size)
    for batch in scanner.to_batches():
        f = batch.to_pandas()
        if f.empty:
            continue
        yield f[features], f.energy_kwh, f[["site_id", "ts", "issue_cutoff"]]


def iter_fold(directory, fold, part, *, features=None, batch_size=65536, verify=True):
    splits = pd.read_csv(Path(directory)/"validation_splits.csv").set_index("fold")
    r = splits.loc[fold]
    origin = pd.Timestamp(r.first_issue_cutoff)
    first = pd.Timestamp(r.validation_start)
    if pd.Timestamp(r.train_end) != origin or origin != cutoff([first])[0]:
        raise ValueError("Training boundary violates first prediction cutoff")
    if part == "train":
        start, end = r.train_start, r.train_end
    elif part == "validation":
        start, end = r.validation_start, r.validation_end
    else:
        raise ValueError("part must be train or validation")
    yield from iter_training_batches(directory, features=features, start=start, end=end,
                                    batch_size=batch_size, verify=verify)


def score_predictions(rows, composition, weights):
    """rows: site_id,ts,actual,prediction. Only complete-roster hours are scored.

    Actual negatives are preserved for raw-target scoring. Never zero-fill
    unavailable targets. Penalty is computed AFTER signed errors cancel inside
    each group, with the 0.1 coefficient applied exactly once.
    """
    if composition.site_id.isna().any() or composition.site_id.duplicated().any():
        raise ValueError("Invalid composition roster")
    if not set(composition.current_group) <= {"A", "B", "G1", "G2"} or composition.current_group.nunique() != 2:
        raise ValueError("Expected exactly two reference groups")
    if rows.duplicated(["site_id", "ts"]).any() or not rows.site_id.isin(composition.site_id).all():
        raise ValueError("Duplicate/unknown prediction keys")
    if weights.ts.duplicated().any():
        raise ValueError("Duplicate settlement weights")
    if not np.isfinite(rows[["actual", "prediction"]].to_numpy()).all() or (rows.prediction < 0).any():
        raise ValueError("Nonfinite target/prediction or negative prediction")
    complete = rows.groupby("ts").site_id.nunique()
    complete = complete[complete == len(composition)].index
    coverage = dict(complete_hours=len(complete), available_prediction_hours=int(rows.ts.nunique()),
                    rows=len(rows), sites=len(composition))
    if not len(complete):
        return dict(**coverage, penalty_krw=None, reason="no complete-roster hours"), pd.DataFrame()
    f = rows.loc[rows.ts.isin(complete)].merge(composition[["site_id", "current_group"]],
                                               on="site_id", validate="many_to_one")
    g = f.groupby(["ts", "current_group"])[["prediction", "actual"]].sum().reset_index()
    g = g.merge(weights, on="ts", validate="many_to_one", how="left")
    if not np.isfinite(g[["smp", "k"]].to_numpy()).all() or (g.smp < 0).any() or not g.k.isin([1, 3]).all():
        raise ValueError("Missing/invalid settlement weights")
    expected_k = np.where(g.ts.dt.hour.between(11, 15), 3, 1)
    if not (g.k == expected_k).all():
        raise ValueError("k disagrees with KST ts.hour")
    g["abs_error_kwh"] = (g.prediction-g.actual).abs()
    g["weighted_absolute_error"] = g.smp*g.k*g.abs_error_kwh
    g["penalty_krw"] = 0.1*g.weighted_absolute_error
    error = rows.prediction-rows.actual
    ge = g.prediction-g.actual
    revenue = float((g.smp*g.actual).sum())
    return dict(**coverage, penalty_krw=float(g.penalty_krw.sum()),
                weighted_absolute_error=float(g.weighted_absolute_error.sum()),
                revenue_krw=revenue, settlement_krw=revenue-float(g.penalty_krw.sum()),
                site_mae=float(error.abs().mean()), site_rmse=float(np.sqrt((error**2).mean())),
                group_mae=float(ge.abs().mean()), group_rmse=float(np.sqrt((ge**2).mean()))), g
