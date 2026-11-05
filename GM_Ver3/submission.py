"""Check both OIBC submission CSVs against the participant package."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import read_prices, read_roster
from .settlement import GROUPS, group_capacities, kst_hours


def validate(sites, times, composition, forecast):
    expected = pd.date_range("2025-06-01 01:00", periods=8_760, freq="h", tz="Asia/Seoul")
    if len(sites) != 253 or not kst_hours(times).equals(expected):
        raise ValueError("Expected the official roster and evaluation grid")
    if list(composition) != ["site_id", "group"] or len(composition) != 253:
        raise ValueError("Invalid composition columns or row count")
    capacity = group_capacities(sites, composition)
    if list(forecast) != ["group", "ts", "forecast_kwh"] or len(forecast) != 17_520:
        raise ValueError("Invalid forecast columns or row count")
    if forecast.group.isna().any() or set(forecast.group) != set(GROUPS):
        raise ValueError("Invalid forecast group")
    energy = pd.to_numeric(forecast.forecast_kwh, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(energy).all() or (energy < 0).any():
        raise ValueError("Invalid forecast energy")
    for group in GROUPS:
        timestamps = forecast.loc[forecast.group.eq(group), "ts"]
        if not timestamps.astype(str).str.endswith("+09:00").all():
            raise ValueError(f"{group} requires explicit +09:00 timestamps")
        index = kst_hours(pd.to_datetime(timestamps)).sort_values()
        if not index.equals(expected):
            raise ValueError(f"{group} has duplicate, missing or unexpected timestamps")
    return {"composition_rows": len(composition), "forecast_rows": len(forecast),
            "hours_per_group": len(expected), "capacity_kw": capacity.to_dict()}


def validate_files(sites, times, folder):
    composition = pd.read_csv(folder / "composition.csv", dtype={"site_id": str})
    forecast = pd.read_csv(folder / "forecast.csv", dtype={"ts": str})
    return validate(sites, times, composition, forecast)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args()
    result = validate_files(read_roster(args.raw), read_prices(args.raw / "smp.csv").index, args.submission)
    print(json.dumps(result, indent=2))
