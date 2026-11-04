"""Forecast-only features shared by five tree and two sequence regressors.

Input is the existing complete site/hour cache generated from the supplied
OIBC files. Each 01:00..24:00 target day is an independent 24-row block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEATHER = ("wa_cloud", "wb_cloud", "wb_uv_idx", "wa_temp", "wb_temp")
TRAJECTORY = ("wa_cloud", "wb_cloud", "wb_uv_idx")
NAMES = (
    "hour_sin", "hour_cos", "daylight", "cosz_positive",
    "cloud_mean", "cloud_absdiff", "temp_mean", "temp_absdiff",
    "humidity_absdiff", "sun_cloud_wa", "sun_cloud_wb", "sun_uv",
    "sun_cloud_disagreement", "sun_temp", "uv_cloud", "wb_wind_ms",
    "wa_cloud_missing", "wb_cloud_missing", "wb_uv_idx_missing",
    "wa_temp_missing", "wb_temp_missing",
    *(f"{c}_revision_abs" for c in WEATHER),
    *(f"{c}_delta1" for c in WEATHER),
    *(f"{c}_std3" for c in TRAJECTORY),
    *(f"{c}_day_mean" for c in TRAJECTORY),
    *(f"{c}_day_deviation" for c in TRAJECTORY),
)


def feature_names(base: list[str]) -> list[str]:
    if len(base) != len(set(base)) or set(base) & set(NAMES):
        raise ValueError("Duplicate feature name")
    return list(base) + list(NAMES)


def _day_matrix(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].to_numpy(dtype="float32").reshape(-1, 24)


def add_features(frame: pd.DataFrame, base: list[str]) -> pd.DataFrame:
    """Append issued-forecast transformations without reading labels or prices."""
    if len(frame) % 24:
        raise ValueError("Incomplete 24-hour target day")
    required = set(base) | {"site_id", "ts", "wb_rel_hum", "wa_humidity", "cosz", "hour"}
    required |= {f"{c}_round_delta" for c in WEATHER}
    absent = required - set(frame.columns)
    if absent:
        raise ValueError(f"Missing source columns: {sorted(absent)}")
    ts = pd.DatetimeIndex(frame.ts)
    ids = frame.site_id.to_numpy().reshape(-1, 24)
    days = (ts - pd.Timedelta(hours=1)).normalize().asi8.reshape(-1, 24)
    if not (ts.hour[::24] == 1).all() or not (ts.hour[23::24] == 0).all():
        raise ValueError("Wrong operating-day boundaries")
    if not (ids == ids[:, :1]).all() or not (days == days[:, :1]).all():
        raise ValueError("Mixed sites or target days")
    if frame.duplicated(["site_id", "ts"]).any():
        raise ValueError("Duplicate site/hour")

    def a(name):
        return frame[name].to_numpy(dtype="float32")

    sun = np.maximum(a("cosz"), 0)
    hour = a("hour")
    cloud_a, cloud_b = a("wa_cloud"), a("wb_cloud")
    cloud_gap = cloud_a - cloud_b
    derived = {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "daylight": (sun > 0).astype("float32"),
        "cosz_positive": sun,
        "cloud_mean": (cloud_a + cloud_b) / 2,
        "cloud_absdiff": np.abs(cloud_gap),
        "temp_mean": (a("wa_temp") + a("wb_temp")) / 2,
        "temp_absdiff": np.abs(a("wa_temp") - a("wb_temp")),
        "humidity_absdiff": np.abs(a("wa_humidity") - a("wb_rel_hum")),
        "sun_cloud_wa": sun * (1 - np.clip(cloud_a / 100, 0, 1)) ** 2,
        "sun_cloud_wb": sun * (1 - np.clip(cloud_b / 100, 0, 1)) ** 2,
        "sun_uv": sun * a("wb_uv_idx"),
        "sun_cloud_disagreement": sun * np.abs(cloud_gap),
        "sun_temp": sun * a("wb_temp"),
        "uv_cloud": a("wb_uv_idx") * (1 - np.clip(cloud_b / 100, 0, 1)),
        "wb_wind_ms": a("wb_wind_speed") / 3.6,
    }
    for c in WEATHER:
        derived[f"{c}_revision_abs"] = np.abs(a(f"{c}_round_delta"))
        values = _day_matrix(frame, c)
        difference = np.full_like(values, np.nan)
        difference[:, 1:] = values[:, 1:] - values[:, :-1]
        derived[f"{c}_delta1"] = difference.ravel()
        derived[f"{c}_missing"] = (~np.isfinite(values)).astype("float32").ravel()
    for c in TRAJECTORY:
        values = _day_matrix(frame, c)
        std = np.full_like(values, np.nan)
        for h in range(24):
            part = values[:, max(0, h - 2):h + 1]
            valid = np.isfinite(part).all(axis=1)
            std[valid, h] = part[valid].std(axis=1)
        count = np.isfinite(values).sum(axis=1)
        mean = np.divide(np.nansum(values, axis=1), count,
                         out=np.full(len(values), np.nan), where=count > 0)
        derived[f"{c}_std3"] = std.ravel()
        derived[f"{c}_day_mean"] = np.repeat(mean, 24)
        derived[f"{c}_day_deviation"] = (values - mean[:, None]).ravel()
    if set(derived) != set(NAMES):
        raise RuntimeError("Feature schema and calculation differ")
    extra = pd.DataFrame({name: np.asarray(derived[name], dtype="float32")
                          for name in NAMES}, index=frame.index)
    if np.isinf(extra.to_numpy()).any():
        raise ValueError("Infinite derived feature")
    return pd.concat([frame, extra], axis=1)
