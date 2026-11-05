from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TZ = "Asia/Seoul"
SEED = 20261105
EXTRA_FEATURES = ["site_id__code", "solar_cosz_positive", "solar_elevation",
                  "solar_azimuth_sin", "solar_azimuth_cos", "solar_daylight",
                  "solar_daylength_hours"]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_output(path):
    path = Path(path).resolve()
    if "onedrive" in str(path).lower() or any((p / ".git").exists() for p in [path, *path.parents]):
        raise ValueError("Model weights and row-level outputs must stay outside Git/OneDrive")
    path.mkdir(parents=True, exist_ok=True)
    return path


class Panel:
    """Memmapped (operating day x site, 24 forecast hours, features).

    Targets remain NaN when absent; labels are never model inputs. Every sequence
    uses a single preceding-day 17:00 forecast issue. No observed-weather input.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "panel.json").read_text(encoding="utf-8"))
        self.features = self.meta["features"]
        self.X = np.load(self.path / "X.npy", mmap_mode="r")
        self.y = np.load(self.path / "y.npy", mmap_mode="r")
        self.capacity = np.array(self.meta["capacity_kw"], dtype=np.float32)
        self.sites = self.meta["site_ids"]
        self.nsites = len(self.sites)
        self.first_day = pd.Timestamp(self.meta["first_operating_day"])
        # Pandas 3 parses strings at microsecond resolution; all arithmetic below
        # and persisted forecast keys explicitly use nanoseconds.
        self.days = (self.first_day + pd.to_timedelta(np.arange(len(self.X)) // self.nsites, unit="D")).as_unit("ns")
        self.site_index = np.arange(len(self.X)) % self.nsites

    def indices(self, start=None, end=None, labeled=True):
        mask = np.ones(len(self.X), dtype=bool)
        if start is not None:
            mask &= self.days >= pd.Timestamp(start)
        if end is not None:
            mask &= self.days <= pd.Timestamp(end).normalize()
        ids = np.flatnonzero(mask)
        if labeled:
            usable = np.isfinite(self.y[ids]) & (self.y[ids] >= 0)
            if end is not None:
                usable &= self.times(ids) <= pd.Timestamp(end).value
            ids = ids[usable.any(axis=1)]
        return ids

    def times(self, ids):
        return self.days[ids].asi8[:, None] + np.arange(1, 25, dtype=np.int64)[None, :] * 3_600_000_000_000

    def batch(self, ids, end=None):
        x = np.array(self.X[ids], dtype=np.float32)
        y = np.array(self.y[ids], dtype=np.float32)
        mask = np.isfinite(y) & (y >= 0)
        if end is not None:
            mask &= self.times(ids) <= pd.Timestamp(end).value
        cap = self.capacity[self.site_index[ids]]
        # Placeholder zero is only used under an explicit loss mask.
        target = np.where(mask, y / cap[:, None], 0).astype(np.float32)
        return x, target, mask, cap


def prepare_panel(dataset, out, evaluation=False):
    dataset, out = Path(dataset), Path(out)
    m = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    features = m["default_features"] + EXTRA_FEATURES
    src = m["summary"]["evaluation" if evaluation else "training"]
    if (out / "panel.json").exists():
        old = json.loads((out / "panel.json").read_text())
        if old["source_sha256"] != src["sha256"] or old["features"] != features:
            raise ValueError("Existing cache does not match frozen input")
        return Panel(out)
    if sha256(dataset / src["file"]) != src["sha256"]:
        raise ValueError("Frozen source hash mismatch")
    out.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(dataset / src["file"])
    nseq = src["rows"] // 24
    xout = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype="float32", shape=(nseq, 24, len(features)))
    yout = np.lib.format.open_memmap(out / "y.npy", mode="w+", dtype="float32", shape=(nseq, 24))
    pos, previous_last = 0, None
    sites, capacities, first_day = None, None, None
    for rg in range(pf.num_row_groups):
        f = pf.read_row_group(rg, columns=["site_id", "ts", "issue_cutoff", "energy_kwh"] + features).to_pandas()
        f = f.sort_values(["ts", "site_id"], ignore_index=True)
        times = f.ts.drop_duplicates()
        if previous_last is not None and times.iloc[0] != previous_last + pd.Timedelta(hours=1):
            raise ValueError("Nonconsecutive source row groups")
        previous_last = times.iloc[-1]
        if times.iloc[0].hour != 1 or times.iloc[-1].hour != 0:
            raise ValueError("Source row group must contain whole operating days")
        if not np.all(np.diff(times.array.asi8) == pd.Timedelta(hours=1).value):
            raise ValueError("Source timestamps are not hourly")
        ns = f.site_id.nunique()
        this_sites = sorted(f.site_id.unique().tolist())
        if len(f) != len(times) * ns or f.duplicated(["site_id", "ts"]).any():
            raise ValueError("Incomplete or duplicated feature grid")
        if sites is None:
            sites = this_sites
            capacities = f.iloc[:ns].capacity_kw.tolist()
            first_day = times.iloc[0].normalize()
        if sites != this_sites or not np.array_equal(f.site_id.to_numpy().reshape(-1, ns)[0], sites):
            raise ValueError("Site roster mismatch")
        expected_cutoff = (f.ts - pd.Timedelta(hours=1)).dt.normalize() - pd.Timedelta(hours=7)
        if not (f.issue_cutoff == expected_cutoff).all():
            raise ValueError("Forecast issue boundary changed")
        nd = len(times) // 24
        xx = f[features].to_numpy(dtype=np.float32).reshape(nd, 24, ns, -1).transpose(0, 2, 1, 3).reshape(-1, 24, len(features))
        yy = f.energy_kwh.to_numpy(dtype=np.float32).reshape(nd, 24, ns).transpose(0, 2, 1).reshape(-1, 24)
        xout[pos:pos+len(xx)] = xx
        yout[pos:pos+len(xx)] = yy
        pos += len(xx)
        print(f"cache {'evaluation' if evaluation else 'training'} {rg+1}/{pf.num_row_groups}", flush=True)
    assert pos == nseq
    xout.flush()
    yout.flush()
    write_json(out / "panel.json", dict(features=features, site_ids=sites, capacity_kw=capacities,
               first_operating_day=str(first_day), source_sha256=src["sha256"], rows=src["rows"]))
    return Panel(out)


def sample_ids(ids, limit, seed=SEED):
    if len(ids) <= limit:
        return ids
    return np.sort(np.random.default_rng(seed).choice(ids, limit, replace=False))


def fit_scaler(panel, ids, end):
    count = np.zeros(len(panel.features), dtype=np.float64)
    total, squares = count.copy(), count.copy()
    for block in np.array_split(ids, max(1, (len(ids)+1023)//1024)):
        x, _, mask, _ = panel.batch(block, end)
        z = x[mask].astype(np.float64)
        valid = np.isfinite(z)
        count += valid.sum(axis=0)
        z = np.where(valid, z, 0)
        total += z.sum(axis=0)
        squares += (z*z).sum(axis=0)
    mean = total / np.maximum(count, 1)
    std = np.sqrt(np.maximum(squares/np.maximum(count, 1) - mean*mean, 0))
    std[std < 1e-6] = 1
    return dict(mean=mean.tolist(), std=std.tolist(), clip=10.0, fit_end=str(end), fit_on="eligible training labels only")


def transform(x, scaler):
    z = (x - np.array(scaler["mean"], dtype=np.float32)) / np.array(scaler["std"], dtype=np.float32)
    return np.clip(np.nan_to_num(z, nan=0, posinf=10, neginf=-10), -10, 10).astype(np.float32)


def tabular(panel, ids, end, row_limit=None):
    # Build only eligible rows, in bounded chunks, rather than loading the grid.
    counts = []
    for block in np.array_split(ids, max(1, (len(ids)+2047)//2048)):
        _, _, mask, _ = panel.batch(block, end)
        counts.append(int(mask.sum()))
    total = sum(counts)
    take = None if row_limit is None or total <= row_limit else np.sort(np.random.default_rng(SEED).choice(total, row_limit, replace=False))
    n = total if take is None else len(take)
    xx = np.empty((n, len(panel.features)), dtype=np.float32)
    yy, cc = np.empty(n, dtype=np.float32), np.empty(n, dtype=np.float32)
    offset, dest = 0, 0
    for block, count in zip(np.array_split(ids, max(1, (len(ids)+2047)//2048)), counts):
        x, y, mask, cap = panel.batch(block, end)
        select = np.arange(count) if take is None else take[(take >= offset) & (take < offset+count)] - offset
        nr = len(select)
        xx[dest:dest+nr] = x[mask][select]
        yy[dest:dest+nr] = y[mask][select]
        cc[dest:dest+nr] = np.broadcast_to(cap[:, None], mask.shape)[mask][select]
        offset += count
        dest += nr
    assert dest == n
    return xx, yy, cc
