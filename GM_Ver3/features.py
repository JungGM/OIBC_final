"""Build forecast-only inputs from the participant package."""

from pathlib import Path

import numpy as np
import pandas as pd

from .settlement import kst_hours, price_arrays, site_matrix


WEATHER = {
    "wa": ("cloud", "temp", "humidity", "wind_speed", "rain", "snow"),
    "wb": ("cloud", "temp", "rel_hum", "wind_speed", "uv_idx", "total_liq", "snow", "rain_prob"),
}


def read_prices(path):
    weights = pd.read_csv(path)
    weights.index = kst_hours(pd.to_datetime(weights.pop("ts")))
    times, _, _ = price_arrays(weights)
    if times[0].hour != 1 or len(times) % 24 or not times.equals(pd.date_range(times[0], periods=len(times), freq="h")):
        raise ValueError("Expected a continuous grid of whole operating days")
    return weights


def read_roster(raw):
    sites = pd.read_csv(raw / "sites.csv", dtype={"site_id": str}).sort_values("site_id").reset_index(drop=True)
    if len(sites) != 253:
        raise ValueError("Expected the official 253-site roster")
    return sites


def read_weather(directory, provider):
    station = f"{provider}_station"
    keys = ["ts", station, "round"]
    names = [f"{provider}_{name}" for name in WEATHER[provider]]
    weather = pd.read_parquet(directory / f"forecast_{provider}.parquet", columns=keys + names)
    times = pd.DatetimeIndex(weather.ts)
    hourly = times == times.floor("h")
    weather = weather.loc[hourly].copy()
    weather["ts"] = kst_hours(weather.ts)
    if weather[keys].isna().any().any() or not weather["round"].isin([10, 17]).all() or weather.duplicated(keys).any():
        raise ValueError(f"Invalid forecast keys for {provider}")
    weather[names] = weather[names].astype("float32").replace([np.inf, -np.inf, 9999], np.nan)
    if provider == "wb":
        weather["wb_wind_speed"] /= 3.6
        weather["wb_snow"] *= 10
    early = [name for name in names if name.endswith(("_cloud", "_temp", "_uv_idx"))]
    late = weather.loc[weather["round"].eq(17)].set_index(keys[:2])[names].sort_index()
    morning = weather.loc[weather["round"].eq(10)].set_index(keys[:2])[early].sort_index()
    print(f"{directory.name}/{provider}: {len(late):,} r17, {len(morning):,} r10, {int((~hourly).sum())} off-grid excluded", flush=True)
    return late, morning


def solar_elevation_sine(times, sites):
    midpoint = times - pd.Timedelta(minutes=30)
    hour = midpoint.hour.to_numpy() + 0.5
    season = 2 * np.pi * (midpoint.dayofyear.to_numpy() - 1 + (hour - 12) / 24) / 365.2425
    declination = (0.006918 - 0.399912 * np.cos(season) + 0.070257 * np.sin(season)
                   - 0.006758 * np.cos(2 * season) + 0.000907 * np.sin(2 * season)
                   - 0.002697 * np.cos(3 * season) + 0.00148 * np.sin(3 * season))
    equation = 229.18 * (0.000075 + 0.001868 * np.cos(season) - 0.032077 * np.sin(season)
                         - 0.014615 * np.cos(2 * season) - 0.040849 * np.sin(2 * season))
    angle = np.deg2rad((hour[:, None] * 60 + equation[:, None] + 4 * sites.lon.to_numpy() - 540) / 4 - 180)
    latitude = np.deg2rad(sites.lat.to_numpy())
    solar = np.sin(latitude) * np.sin(declination[:, None]) + np.cos(latitude) * np.cos(declination[:, None]) * np.cos(angle)
    return solar, season


def make_features(times, sites, weather):
    shape = (len(times), len(sites))
    solar, season = solar_elevation_sine(times, sites)
    hour_angle = times.hour.to_numpy() * np.pi / 12
    out = {"solar": solar}
    for name, values in (("hour_sin", np.sin(hour_angle)), ("hour_cos", np.cos(hour_angle)),
                         ("season_sin", np.sin(season)), ("season_cos", np.cos(season))):
        out[name] = np.broadcast_to(values[:, None], shape)
    for name in ("capacity_kw", "lat", "lon", "wa_dist", "wb_dist"):
        out[name] = np.broadcast_to(sites[name].to_numpy(), shape)
    out["site_code"] = np.broadcast_to(np.arange(len(sites)), shape)
    for provider, (late, morning) in weather.items():
        stations = sites[f"{provider}_station"]
        keys = pd.MultiIndex.from_arrays([times.repeat(len(sites)), np.tile(stations, len(times))])
        for name, values in late.reindex(keys).items():
            out[name] = values.to_numpy().reshape(shape)
        for name, values in morning.reindex(keys).items():
            out[f"{name}_revision"] = out[name] - values.to_numpy().reshape(shape)
        cloud = out[f"{provider}_cloud"]
        daily = cloud.reshape(-1, 24, len(sites))
        counts = np.isfinite(daily).sum(axis=1, keepdims=True)
        cloud_mean = np.divide(np.nansum(daily, axis=1, keepdims=True), counts,
                               out=np.full(counts.shape, np.nan), where=counts > 0)
        out[f"{provider}_cloud_day"] = np.broadcast_to(cloud_mean, daily.shape).reshape(shape)
        step = np.full(daily.shape, np.nan, dtype=np.float32)
        step[:, 1:] = daily[:, 1:] - daily[:, :-1]
        out[f"{provider}_cloud_step"] = step.reshape(shape)
        out[f"{provider}_clear_proxy"] = np.maximum(solar, 0) * (1 - cloud.clip(0, 100) / 100)
    for name in ("cloud", "temp", "wind_speed"):
        out[f"{name}_difference"] = out[f"wa_{name}"] - out[f"wb_{name}"]
    return np.stack(tuple(out.values()), axis=-1, dtype=np.float32), list(out)


def prepare_features(raw, phase, times, sites, output: Path):
    weather = {provider: read_weather(raw / phase, provider) for provider in WEATHER}
    block_size = 24 * 28
    for left in range(0, len(times), block_size):
        right = min(left + block_size, len(times))
        block, names = make_features(times[left:right], sites, weather)
        if left == 0:
            matrix = np.lib.format.open_memmap(output, mode="w+", dtype="float32",
                                               shape=(len(times), len(sites), len(names)))
        matrix[left:right] = block
    matrix.flush()
    return matrix, names


def read_actual(raw, times, sites):
    actual = pd.read_parquet(raw / "train/generation.parquet", columns=["ts", "site_id", "energy"])
    if not kst_hours(actual.ts).isin(times).all() or np.isinf(actual.energy).any():
        raise ValueError("Actuals are outside the price grid or contain infinite energy")
    return site_matrix(actual, "energy", times, sites.site_id).astype(np.float32)
