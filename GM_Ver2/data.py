"""Rebuild forecast-only day tensors from the participant package."""

import json

import numpy as np
import pandas as pd


WEATHER = {
    "wa": "cloud temp humidity ground_press wind_speed wind_dir rain snow",
    "wb": "cloud temp rel_hum wind_speed wind_dir uv_idx rain_prob total_liq snow vis ceiling",
}
KEY = ["site_id", "ts"]


def timestamps(values):
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).tz_convert("Asia/Seoul").as_unit("ns")


def prices(path):
    table = pd.read_csv(path)
    table.index = timestamps(table.pop("ts"))
    k = np.where(table.index.hour.isin(range(11, 16)), 3, 1)
    if "k" in table and not np.array_equal(table.k, k):
        raise ValueError("Provided K disagrees with the closing-hour convention")
    if (not table.index.equals(pd.date_range(table.index[0], periods=len(table), freq="h"))
            or len(table) % 24 or table.index[0].hour != 1
            or not np.isfinite(table.smp).all() or (table.smp < 0).any()):
        raise ValueError("Invalid price grid")
    return table.assign(k=k)


def read_weather(path, provider):
    station = provider + "_station"
    fields = [provider + "_" + name for name in WEATHER[provider].split()]
    frame = pd.read_parquet(path, columns=[station, "ts", "round", *fields])
    frame["ts"] = timestamps(frame.ts)
    on_grid = frame.ts.eq(frame.ts.dt.floor("h"))
    off_grid = int((~on_grid).sum())
    frame = frame.loc[on_grid].copy()
    if frame.duplicated([station, "ts", "round"]).any() or not frame["round"].isin([10, 17]).all():
        raise ValueError("Invalid forecast keys or issuance rounds")
    frame[fields] = frame[fields].astype("float32").replace([np.inf, -np.inf], np.nan)
    frames = [frame.loc[frame["round"].eq(round_id)].drop(columns="round").set_index([station, "ts"])
              for round_id in (17, 10)]
    return frames, fields, off_grid


def solar_proxy(times, sites):
    center = times - pd.Timedelta(minutes=30)
    hour = np.asarray(center.hour + center.minute / 60)
    phase = 2 * np.pi * (center.dayofyear.to_numpy() - 1 + (hour - 12) / 24) / 365.2425
    eq = 229.18 * (0.000075 + 0.001868*np.cos(phase) - 0.032077*np.sin(phase)
                  - 0.014615*np.cos(2*phase) - 0.040849*np.sin(2*phase))
    dec = (0.006918 - 0.399912*np.cos(phase) + 0.070257*np.sin(phase)
           - 0.006758*np.cos(2*phase) + 0.000907*np.sin(2*phase)
           - 0.002697*np.cos(3*phase) + 0.00148*np.sin(3*phase))
    angle = np.deg2rad((hour[:, None]*60 + eq[:, None] + 4*sites.lon.to_numpy() - 540)/4 - 180)
    lat = np.deg2rad(sites.lat.to_numpy())
    sine = np.sin(lat)*np.sin(dec[:, None]) + np.cos(lat)*np.cos(dec[:, None])*np.cos(angle)
    return sine, hour, phase


def features(times, sites, weather):
    shape = (len(times)//24, 24, len(sites))
    sine, hour, phase = solar_proxy(times, sites)
    solar = np.maximum(sine, 0).reshape(shape)
    data = {"solar": solar, "solar_signed": sine.reshape(shape)}
    for name, vals in [("hour_sin", np.sin(hour*np.pi/12)), ("hour_cos", np.cos(hour*np.pi/12)),
                       ("year_sin", np.sin(phase)), ("year_cos", np.cos(phase))]:
        data[name] = np.broadcast_to(vals.reshape(-1, 24, 1), shape)
    for name in ["capacity_kw", "lat", "lon", "wa_dist", "wb_dist", "site_code", "wa_code", "wb_code"]:
        data[name] = np.broadcast_to(sites[name].to_numpy(), shape)
    for provider, (frames, fields, _) in weather.items():
        keys = pd.MultiIndex.from_arrays([np.tile(sites[provider+"_station"], len(times)),
                                         times.repeat(len(sites))], names=frames[0].index.names)
        now, old = [f.reindex(keys).to_numpy().reshape(*shape, len(fields)) for f in frames]
        data[provider+"_missing"] = (~np.isfinite(now)).mean(axis=-1)
        for j, name in enumerate(fields):
            value = now[..., j]
            if name.endswith("wind_dir"):
                data[name+"_sin"] = np.sin(np.deg2rad(value))
                data[name+"_cos"] = np.cos(np.deg2rad(value))
                continue
            data[name] = value
            if name.split("_", 1)[1] in ("cloud", "temp", "uv_idx", "rain", "total_liq"):
                data[name+"_revision"] = value - old[..., j]
                prev = np.roll(value, 1, axis=1)
                following = np.roll(value, -1, axis=1)
                prev[:, 0] = following[:, -1] = np.nan
                data[name+"_rise"] = value - prev
                data[name+"_next"] = following - value
                count = np.isfinite(value).sum(axis=1, keepdims=True)
                mean = np.nansum(value, axis=1, keepdims=True) / np.maximum(count, 1)
                mean[count == 0] = np.nan
                data[name+"_day_mean"] = np.broadcast_to(mean, shape)
            if name.endswith("_cloud"):
                data[name+"_sun"] = solar * np.clip(1 - value/100, 0, 1)
    for variable in ("cloud", "temp"):
        data[variable+"_disagreement"] = data["wa_"+variable] - data["wb_"+variable]
    return np.stack(list(data.values()), axis=-1).astype("float32"), list(data)


def prepare(raw, out):
    sites = pd.read_csv(raw/"sites.csv").sort_values("site_id").reset_index(drop=True)
    if (len(sites) != 253 or not sites.site_id.is_unique or not sites.capacity_kw.gt(0).all()
            or set(sites.current_group) != {"A", "B"}):
        raise ValueError("Invalid competition roster")
    for name in ("site", "wa", "wb"):
        sites[name+"_code"] = pd.factorize(sites["site_id" if name == "site" else name+"_station"])[0]
    target = pd.read_parquet(raw/"train/generation.parquet", columns=[*KEY, "energy"])
    target["ts"] = timestamps(target.ts)
    if (target.duplicated(KEY).any() or not target.ts.eq(target.ts.dt.floor("h")).all()
            or not target.site_id.isin(sites.site_id).all()):
        raise ValueError("Invalid generation keys")
    target = target.set_index(["ts", "site_id"]).energy
    info = {}
    for split, directory in [("train", "train"), ("eval", "test")]:
        price = prices(raw/("train/smp.csv" if split == "train" else "smp.csv"))
        weather = {name: read_weather(raw/directory/f"forecast_{name}.parquet", name)
                   for name in WEATHER}
        n_days = len(price)//24
        for left in range(0, n_days, 31):
            right = min(left + 31, n_days)
            times = price.index[left*24:right*24]
            x, names = features(times, sites, weather)
            if left == 0:
                matrix = np.lib.format.open_memmap(out/f"{split}_x.npy", mode="w+",
                                                   dtype="float32", shape=(n_days, *x.shape[1:]))
            matrix[left:right] = x
        matrix.flush()
        del matrix
        if split == "train":
            grid = pd.MultiIndex.from_product([price.index, sites.site_id], names=["ts", "site_id"])
            y = target.reindex(grid).to_numpy(dtype=np.float64).reshape(n_days, 24, len(sites))
            np.save(out/"train_y.npy", y)
        info[split] = {"days": n_days, "features": names,
                       "off_grid_forecasts_excluded": {k: v[2] for k, v in weather.items()}}
        print(f"Prepared {split}: {n_days} days, {len(names)} features", flush=True)
    (out/"data_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return sites, info
