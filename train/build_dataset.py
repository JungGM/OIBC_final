"""Build a frozen, forecast-only OIBC training matrix from supplied raw files.

All row-level outputs stay local. No models are fitted and no network is used.
"""
from __future__ import annotations
import argparse
import gc
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from audit_data import TZ, audit, check_time, cutoff, digest, hourly_mask, save_json

WA = ["wa_cloud", "wa_temp", "wa_temp_max", "wa_temp_min", "wa_humidity",
      "wa_ground_press", "wa_wind_speed", "wa_wind_dir", "wa_rain", "wa_snow"]
WB = ["wb_temp", "wb_feel_temp", "wb_wet_bulb_temp", "wb_dew_point",
      "wb_wind_speed", "wb_wind_dir", "wb_wind_gust_spd", "wb_rel_hum", "wb_vis",
      "wb_ceiling", "wb_uv_idx", "wb_precip_prob", "wb_rain_prob", "wb_snow_prob",
      "wb_ice_prob", "wb_total_liq", "wb_rain", "wb_snow", "wb_ice", "wb_cloud"]
TRAJECTORY = ["wa_cloud", "wa_temp", "wa_rain", "wa_wind_speed", "wa_snow",
              "wb_cloud", "wb_temp", "wb_uv_idx", "wb_total_liq", "wb_wind_speed", "wb_snow", "wb_ice"]
UNITS = {"capacity_kw": "kW", "lat": "degree (anonymized)", "lon": "degree (anonymized)",
         "wa_dist": "m", "wb_dist": "m"}
for c in WA + WB:
    UNITS[c] = ("degree_C" if "temp" in c or "dew_point" in c else
                "degree" if "wind_dir" in c else
                "m/s" if c == "wa_wind_speed" else
                "km/h" if "wind" in c else
                "%" if any(s in c for s in ("cloud", "humidity", "rel_hum", "prob")) else
                "hPa" if "press" in c else
                "km" if c == "wb_vis" else "m" if c == "wb_ceiling" else
                "index" if c == "wb_uv_idx" else "cm (snow depth)" if c == "wb_snow" else "mm")
CATALOG = {}


def register(name, group, formula, source, unit="dimensionless", default=False,
             missing="preserve NaN; fold-trained imputation only", risk="issued forecast/calendar only"):
    entry = dict(feature=name, group=group, formula=formula, source=source, unit=unit,
                 default=default, availability="target operating day minus 1 day, 17:00 KST",
                 missing=missing, leakage_risk=risk)
    if name in CATALOG and CATALOG[name] != entry:
        raise ValueError(f"Inconsistent feature definition: {name}")
    CATALOG[name] = entry


def issued_table(frame, provider):
    station = provider + "_station"
    cols = WA if provider == "wa" else WB
    check_time(frame.ts, allow_off_grid=True)
    if frame.duplicated([station, "round", "ts"]).any():
        raise ValueError("Duplicate forecast keys")
    if not set(frame["round"].unique()) <= {10, 17}:
        raise ValueError("Unrecognized issue rounds")
    # Off-hour forecasts are quarantined by audit; never rounded to a target key.
    f = frame.loc[hourly_mask(frame.ts)].copy()
    f[cols] = f[cols].astype("float32")
    r17 = f.loc[f["round"] == 17, [station, "ts"] + cols].copy()
    r10 = f.loc[f["round"] == 10, [station, "ts"] + cols].rename(columns={c: c+"__r10" for c in cols})
    merged = r17.merge(r10, on=[station, "ts"], how="outer", validate="one_to_one", indicator=True)
    merged = merged.sort_values([station, "ts"]).reset_index(drop=True)
    d = {}
    group = "D" if provider == "wa" else "E"
    for c in cols:
        register(c, group, "provided round=17 forecast, unchanged", "forecast_"+provider, UNITS[c], True)
        difference = merged[c] - merged[c+"__r10"]
        if c.endswith("wind_dir"):
            difference = (difference + 180) % 360 - 180
        d[c+"__revision"] = difference
        d[c+"__revision_abs"] = difference.abs()
        for suffix, formula in (("__revision", "round17 - round10; circular shortest signed angle for wind direction"),
                                ("__revision_abs", "abs(round17 - round10); circular shortest angle for wind direction")):
            register(c+suffix, "G", formula, c+" round17 and round10", UNITS[c])
        if c.endswith("wind_dir"):
            for fn in ("sin", "cos"):
                n = c+"__"+fn
                d[n] = getattr(np, fn)(np.deg2rad(merged[c]))
                register(n, group, fn+"(degree*pi/180)", c, default=True)
    d[provider+"__r17_missing"] = (merged["_merge"] == "right_only").astype("float32")
    d[provider+"__r10_missing"] = (merged["_merge"] == "left_only").astype("float32")
    for r in (10, 17):
        register(provider+f"__r{r}_missing", "G" if r == 10 else group,
                 "forecast key absent for round="+str(r), "forecast_"+provider,
                 default=(r == 17), missing="1 if source key absent")
    day = (merged.ts - pd.Timedelta(hours=1)).dt.normalize()
    g = merged.assign(__day=day).groupby([station, "__day"], sort=False, observed=True)
    present = merged["_merge"] != "right_only"
    d[provider+"__r17_day_hours"] = present.groupby([merged[station], day], sort=False).transform("sum")
    register(provider+"__r17_day_hours", "H", "count of available round17 target hours in operating day (01..24)",
             "forecast_"+provider, "hours", missing="0 when entire source day absent")
    for c in (v for v in TRAJECTORY if v.startswith(provider+"_")):
        prev_ts, next_ts = g.ts.shift(1), g.ts.shift(-1)
        # Exact 1h adjacency is required: gaps must never become fake hourly changes.
        d[c+"__delta_prev"] = (merged[c] - g[c].shift(1)).where(merged.ts-prev_ts == pd.Timedelta(hours=1))
        d[c+"__delta_next"] = (g[c].shift(-1) - merged[c]).where(next_ts-merged.ts == pd.Timedelta(hours=1))
        for op in ("mean", "min", "max", "std", "count"):
            d[c+"__day_"+op] = g[c].transform(op, **({"ddof": 0} if op == "std" else {}))
        d[c+"__day_range"] = d[c+"__day_max"] - d[c+"__day_min"]
        for suffix in ("delta_prev", "delta_next", "day_mean", "day_min", "day_max", "day_std", "day_count", "day_range"):
            formula = ("within issued operating day, exact 1h adjacency; no cross-day differences" if "delta" in suffix else
                       "available round17 hours within issued operating day; std ddof=0; no full-day imputation")
            register(c+"__"+suffix, "H", formula, c,
                     "hours" if suffix == "day_count" else UNITS[c])
    output = pd.concat([merged[[station, "ts"] + cols], pd.DataFrame(d)], axis=1)
    return output


def solar_features(ts, lat, lon):
    """Analytic solar approximation at hourly interval midpoint; anonymous coords.

    Fractional-year declination/equation-of-time approximation, not irradiance.
    """
    t = pd.DatetimeIndex(ts) - pd.Timedelta(minutes=30)
    hour = t.hour.to_numpy()+t.minute.to_numpy()/60
    gamma = 2*np.pi/365 * (t.day_of_year.to_numpy()-1+(hour-12)/24)
    eq = 229.18*(0.000075+0.001868*np.cos(gamma)-0.032077*np.sin(gamma)
                 -0.014615*np.cos(2*gamma)-0.040849*np.sin(2*gamma))
    dec = (0.006918-0.399912*np.cos(gamma)+0.070257*np.sin(gamma)
           -0.006758*np.cos(2*gamma)+0.000907*np.sin(2*gamma)
           -0.002697*np.cos(3*gamma)+0.00148*np.sin(3*gamma))
    phi = np.deg2rad(np.asarray(lat))
    solar_minutes = hour*60+eq+4*np.asarray(lon)-60*9
    angle = np.deg2rad(solar_minutes/4-180)
    cosz = np.clip(np.sin(phi)*np.sin(dec)+np.cos(phi)*np.cos(dec)*np.cos(angle), -1, 1)
    elev = np.rad2deg(np.arcsin(cosz))
    azim = (np.rad2deg(np.arctan2(np.sin(angle), np.cos(angle)*np.sin(phi)-np.tan(dec)*np.cos(phi)))+180)%360
    h0 = np.arccos(np.clip(-np.tan(phi)*np.tan(dec), -1, 1))
    daylength = np.rad2deg(h0)*2/15
    return {"solar_elevation": elev, "solar_azimuth_sin": np.sin(np.deg2rad(azim)),
            "solar_azimuth_cos": np.cos(np.deg2rad(azim)), "solar_cosz_positive": np.maximum(cosz, 0),
            "solar_daylight": (elev > 0).astype(float), "solar_daylength_hours": daylength,
            "solar_hours_after_sunrise": hour-(12-(eq+4*np.asarray(lon)-540)/60-daylength/2),
            "solar_hours_before_sunset": 12-(eq+4*np.asarray(lon)-540)/60+daylength/2-hour}


def make_month(sites, times, forecasts, targets=None):
    check_time(times)
    f = pd.MultiIndex.from_product([times, sites.site_id], names=["ts", "site_id"]).to_frame(index=False)
    expected = len(f)
    f = f.merge(sites, on="site_id", how="left", validate="many_to_one", sort=False)
    d = {}
    for c in ("capacity_kw", "wa_dist", "wb_dist", "lat", "lon"):
        register(c, "A" if c in ("capacity_kw", "wa_dist", "wb_dist") else "location",
                 "provided site metadata, unchanged", "sites.csv", UNITS[c], c == "capacity_kw",
                 risk="anonymous coordinates only; no location recovery; static metadata")
    for c in ("site_id", "wa_station", "wb_station"):
        mapping = {v: i for i, v in enumerate(sorted(sites[c].unique()))}
        d[c+"__code"] = f[c].map(mapping)
        register(c+"__code", "identity", "sorted full supplied roster category code; treat as categorical",
                 "sites.csv", risk="site memorization/overfit; candidate only")
    ts = pd.DatetimeIndex(f.ts)
    day = ts-pd.Timedelta(hours=1)
    calendar = {"hour_closing": np.where(ts.hour == 0, 24, ts.hour),
                "month": day.month, "day_of_year": day.day_of_year,
                "day_of_week": day.day_of_week, "is_weekend": (day.day_of_week >= 5).astype(float),
                "season": (day.month%12)//3}
    for c, v in calendar.items():
        d[c] = v
        register(c, "A" if c == "hour_closing" else "B", "KST right-closed operating date (ts-1h)", "ts", default=True, missing="not missing")
    for c, values, period in (("hour", ts.hour, 24), ("annual", day.day_of_year-1, np.where(day.is_leap_year, 366, 365))):
        for fn in ("sin", "cos"):
            n = c+"_"+fn
            d[n] = getattr(np, fn)(2*np.pi*np.asarray(values)/period)
            register(n, "B", fn+"(2*pi*phase/period); leap-year aware", "ts", default=True, missing="not missing")
    d.update(solar_features(ts, f.lat, f.lon))
    for c in list(d):
        if c.startswith("solar_"):
            register(c, "C", "analytic solar approximation at ts-30min; fixed supplied anonymized coordinates",
                     "ts, sites.lat, sites.lon", "degree" if c == "solar_elevation" else "hours" if "hours" in c else "dimensionless",
                     missing="not missing", risk="anonymized location uncertainty; candidate only; not measured irradiance")
    f = pd.concat([f, pd.DataFrame(d)], axis=1)
    joins = []
    for provider in ("wa", "wb"):
        table = issued_table(forecasts[provider], provider)
        n = len(f)
        f = f.merge(table, how="left", on=[provider+"_station", "ts"], validate="many_to_one", sort=False)
        if len(f) != n:
            raise ValueError("Forecast join expanded rows")
        miss17 = f[provider+"__r17_missing"].fillna(1)
        miss10 = f[provider+"__r10_missing"].fillna(1)
        f[provider+"__r17_missing"] = miss17
        f[provider+"__r10_missing"] = miss10
        operating_day = (f.ts-pd.Timedelta(hours=1)).dt.normalize()
        day_keys = [f[provider+"_station"], operating_day]
        f[provider+"__r17_day_hours"] = ((1-miss17).groupby(day_keys, sort=False).transform("sum")
                                         /f.groupby(day_keys, sort=False).site_id.transform("nunique"))
        # A missing target-hour key still has access to the other already-issued
        # hours of its operating day. Broadcast only these day-constant summaries.
        summary_cols = [c for c in table if "__day_" in c]
        f[summary_cols] = f.groupby(day_keys, sort=False)[summary_cols].transform("max")
        joins.append(dict(source=provider, before=n, after=len(f), unmatched_round17=int(miss17.sum()),
                          unmatched_round10=int(miss10.sum())))
    derived = {}
    for c in WA+WB:
        n = c+"__missing"
        derived[n] = f[c].isna().astype(float)
        register(n, "missing", "1 iff round17 source value or source key is missing", c, default=True, missing="not missing")
    derived["wb_wind_speed_ms"] = f.wb_wind_speed/3.6
    register("wb_wind_speed_ms", "E", "wb_wind_speed / 3.6", "wb_wind_speed", "m/s", True)
    for label, ca, cb, unit in (("cloud", "wa_cloud", "wb_cloud", "%"),
                                ("temperature", "wa_temp", "wb_temp", "degree_C"),
                                ("humidity", "wa_humidity", "wb_rel_hum", "%"),
                                ("wind_speed_ms", "wa_wind_speed", "wb_wind_speed", "m/s"),
                                ("wind_direction", "wa_wind_dir", "wb_wind_dir", "degree")):
        b = f[cb]/3.6 if cb == "wb_wind_speed" else f[cb]
        delta = f[ca]-b
        if label == "wind_direction":
            delta = (delta+180)%360-180
        for suffix, values in (("diff", delta), ("absdiff", delta.abs())):
            n = "ab_"+label+"_"+suffix
            derived[n] = values
            register(n, "I", "A minus B at site-mapped stations, WB wind /3.6; shortest angle for direction; " + suffix,
                     ca+", "+cb, unit, risk="different mapped stations/definitions; no station equivalence assumed")
    # Rain/snow amount definitions are not assumed equivalent across providers.
    derived["wb_snow_depth_mm"] = f.wb_snow*10
    register("wb_snow_depth_mm", "J", "wb_snow (cm snow depth) * 10; not liquid-water equivalent", "wb_snow", "mm snow depth")
    for n, x, formula, source in (
        ("wb_snow_risk", np.maximum(f.wb_snow_prob/100, (f.wb_snow>0).astype(float).where(f.wb_snow.notna())), "max(snow_prob/100, snow>0); NaN propagates", "wb_snow_prob,wb_snow"),
        ("wb_ice_risk", np.maximum(f.wb_ice_prob/100, (f.wb_ice>0).astype(float).where(f.wb_ice.notna())), "max(ice_prob/100, ice>0); NaN propagates", "wb_ice_prob,wb_ice"),
        ("wb_freezing_wet_bulb", (f.wb_wet_bulb_temp<=0).astype(float).where(f.wb_wet_bulb_temp.notna()), "wet_bulb_temp<=0; preserve missing", "wb_wet_bulb_temp"),
        ("wa_cold_snow", ((f.wa_temp<=0)&(f.wa_snow>0)).astype(float).where(f.wa_temp.notna()&f.wa_snow.notna()), "temp<=0 and snow>0; preserve missing", "wa_temp,wa_snow")):
        derived[n] = x
        register(n, "J", formula, source)
    for provider in ("wa", "wb"):
        n = "solar_cloud_proxy_"+provider
        derived[n] = f.solar_cosz_positive*(1-f[provider+"_cloud"]/100).clip(0, 1)
        register(n, "C", "max(cos(zenith),0) * clip(1-cloud/100,0,1); proxy only", provider+"_cloud, solar_cosz_positive")
    f = pd.concat([f, pd.DataFrame(derived)], axis=1).copy()
    f["issue_cutoff"] = cutoff(f.ts)
    if targets is not None:
        if targets.duplicated(["site_id", "ts"]).any():
            raise ValueError("Duplicate targets")
        f = f.merge(targets.rename(columns={"energy": "energy_kwh"}), how="left", on=["site_id", "ts"], validate="one_to_one", sort=False)
    else:
        f["energy_kwh"] = np.nan
    f["target_observed"] = f.energy_kwh.notna()
    f["target_negative"] = f.energy_kwh.lt(0)
    f["target_eligible"] = f.target_observed & ~f.target_negative
    f["target_above_capacity"] = f.energy_kwh > f.capacity_kw
    f["energy_per_capacity"] = f.energy_kwh/f.capacity_kw
    f = f.sort_values(["ts", "site_id"]).reset_index(drop=True)
    if len(f) != expected or f.duplicated(["site_id", "ts"]).any():
        raise ValueError("Final key grid failed")
    features = sorted(CATALOG)
    converted = f[features].astype({n: "float64" if n in ("capacity_kw", "lat", "lon", "wa_dist", "wb_dist")
                                    else "float32" for n in features})
    if np.isinf(converted.to_numpy()).any():
        raise ValueError("Infinite model features")
    metadata = ["site_id", "ts", "issue_cutoff", "energy_kwh", "energy_per_capacity",
                "target_observed", "target_eligible", "target_negative", "target_above_capacity"]
    # current_group and station identifiers are not silently included in X.
    return pd.concat([f[metadata], converted], axis=1), joins


def generate_splits(start, end):
    result = []
    # The final holdout must not be used for feature or hyperparameter selection.
    definitions = [("tune_2022", "2022-01-01", "2022-12-31", "feature_selection"),
                   ("tune_2023", "2023-01-01", "2023-12-31", "feature_selection"),
                   ("confirm_2024", "2024-01-01", "2024-12-31", "confirmation"),
                   ("holdout_2025", "2025-01-01", "2025-05-31", "final_unused_holdout")]
    for name, left, right, purpose in definitions:
        val_start = pd.Timestamp(left, tz=TZ)+pd.Timedelta(hours=1)
        val_end = pd.Timestamp(right, tz=TZ)+pd.Timedelta(days=1)
        if val_start < start or val_end > end:
            continue
        train_end = cutoff([val_start])[0]
        result.append(dict(fold=name, train_start=str(start), train_end=str(train_end),
                           validation_start=str(val_start), validation_end=str(val_end),
                           first_issue_cutoff=str(train_end), purpose=purpose,
                           label_receipt_status="timestamp embargo enforced; receipt metadata unavailable"))
    return pd.DataFrame(result)


def build(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    if data == out or data in out.parents or out in data.parents:
        raise ValueError("Source and output directories must be separate")
    if any((p / ".git").exists() for p in [out, *out.parents]) or "onedrive" in str(out).lower():
        raise ValueError("Row-level outputs must stay outside Git and OneDrive")
    for n in ("training_data.parquet", "evaluation_input.parquet", "manifest.json"):
        if (out/n).exists():
            raise FileExistsError(f"Refusing to overwrite {n}")
    source_audit_path = out/"source_audit.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8")) if source_audit_path.exists() else audit(data, out)
    for name, r in source_audit["files"].items():
        if r["exists"] and digest(data/name) != r["sha256"]:
            raise ValueError("Source changed since audit: "+name)
    sites = pd.read_csv(data/"sites.csv")
    if (sites.capacity_kw <= 0).any() or sites.isna().any().any():
        raise ValueError("Missing/invalid site metadata")
    train_times = pd.to_datetime(pd.read_csv(data/"train/smp.csv").ts).dt.tz_convert(TZ).dt.as_unit("ns")
    eval_times = pd.to_datetime(pd.read_csv(data/"smp.csv").ts).dt.tz_convert(TZ).dt.as_unit("ns")
    logs, summary, source_missing = [], {}, {}
    for phase, times in (("training", train_times), ("evaluation", eval_times)):
        check_time(times)
        if not times.is_unique or not times.is_monotonic_increasing:
            raise ValueError("Invalid timestamp grid")
        if len(pd.date_range(times.min(), times.max(), freq="h")) != len(times):
            raise ValueError("SMP timestamp grid has gaps")
        folder = "train" if phase == "training" else "test"
        if any(not (data/folder/("forecast_"+p+".parquet")).exists() for p in ("wa", "wb")):
            if phase == "training":
                raise FileNotFoundError("Training forecasts missing")
            source_missing[phase] = "test forecast missing; evaluation input not generated"
            continue
        path = out/("training_data.parquet" if phase == "training" else "evaluation_input.parquet")
        partial = path.with_suffix(".partial.parquet")
        writer = None
        totals = dict(rows=0, observed=0, eligible=0, negative=0, zero_targets=0, missing_targets=0,
                      feature_missing_cells=0, months=0)
        try:
            operating = (pd.DatetimeIndex(times)-pd.Timedelta(hours=1)).strftime("%Y-%m")
            for month in sorted(set(operating)):
                selected = pd.DatetimeIndex(times[operating == month])
                filters = [("ts", ">=", selected.min()), ("ts", "<=", selected.max())]
                forecast = {p: pd.read_parquet(data/folder/("forecast_"+p+".parquet"), filters=filters) for p in ("wa", "wb")}
                target = pd.read_parquet(data/"train/generation.parquet", filters=filters) if phase == "training" else None
                frame, joins = make_month(sites, selected, forecast, target)
                for j in joins:
                    logs.append(dict(phase=phase, month=month, **j))
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(partial, table.schema, compression="zstd", compression_level=5,
                                             use_dictionary=["site_id"])
                writer.write_table(table, row_group_size=len(frame))
                totals["rows"] += len(frame)
                totals["observed"] += int(frame.target_observed.sum())
                totals["eligible"] += int(frame.target_eligible.sum())
                totals["negative"] += int(frame.target_negative.sum())
                totals["zero_targets"] += int(frame.energy_kwh.eq(0).sum())
                totals["missing_targets"] += int((~frame.target_observed).sum())
                totals["feature_missing_cells"] += int(frame[list(CATALOG)].isna().sum().sum())
                totals["months"] += 1
                print(f"BUILD {phase} {month}: {len(frame):,} rows, {int(frame.target_observed.sum()):,} labels, {len(CATALOG)} features", flush=True)
                del frame, table, forecast, target
                gc.collect()
        finally:
            if writer:
                writer.close()
        partial.replace(path)
        totals.update(file=path.name, sha256=digest(path), bytes=path.stat().st_size,
                      start=str(times.min()), end=str(times.max()))
        summary[phase] = totals
    feature_list = sorted(CATALOG)
    pd.DataFrame([CATALOG[n] for n in feature_list]).to_csv(out/"feature_dictionary.csv", index=False)
    pd.DataFrame(logs).to_csv(out/"join_audit.csv", index=False)
    generate_splits(train_times.min(), train_times.max()).to_csv(out/"validation_splits.csv", index=False)
    # Settlement columns are deliberately stored separately from the training matrix.
    for source, dest in (("train/smp.csv", "training_weights.parquet"), ("smp.csv", "evaluation_weights.parquet")):
        weights = pd.read_csv(data/source)
        weights["ts"] = pd.to_datetime(weights.ts)
        expected_k = np.where(weights.ts.dt.hour.between(11, 15), 3, 1)
        if "k" in weights and not (weights.k == expected_k).all():
            raise ValueError("Provided k differs from README")
        weights["k"] = expected_k
        weights.to_parquet(out/dest, index=False)
    sites[["site_id", "current_group", "capacity_kw"]].to_parquet(out/"reference_composition.parquet", index=False)
    plan = []
    for group in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"):
        plan.append(dict(experiment=group, status="not_run" if group != "K" else "withheld",
                         reason="dataset creation only; no performance claim" if group != "K" else "observation receipt metadata unavailable; no observed-derived X",
                         penalty_change_krw=None, penalty_reduction_pct=None))
    pd.DataFrame(plan).to_csv(out/"feature_ablation.csv", index=False)
    manifest = dict(version="fixed_training_v1", status="built_pending_independent_verification",
                    summary=summary, sources=source_audit["files"], missing=source_missing,
                    features=feature_list, default_features=[n for n in feature_list if CATALOG[n]["default"]],
                    categorical_features=[n for n in feature_list if n.endswith("__code")],
                    target="energy_kwh", training_filter="target_eligible == True",
                    key=["site_id", "ts"], timezone=TZ, interval="1h right-closed",
                    cutoff="(ts-1h).normalize()-7h; midnight closes previous operating day",
                    target_policy="preserve raw negative/above-capacity values; exclude 40 negative targets from default fit; never impute targets; preserve zero",
                    input_policy="forecast round17, issue revisions from round10, calendar and supplied metadata only; SMP/k/group/targets/observations are not X",
                    candidate_policy="default_features is a conservative unvalidated starting set; full candidate matrix is fixed, selection requires same-fold ablation",
                    quality_feature_status="K withheld; no observation receipt metadata; current-target observations never used",
                    holiday_status="weekday/weekend only; external holidays withheld",
                    penalty_coefficient=0.1, penalty="0.1 * sum(SMP * k * abs(group_prediction-group_actual))",
                    code_sha256={n:digest(Path(__file__).with_name(n)) for n in ("audit_data.py", "build_dataset.py", "dataset.py", "verify_dataset.py")},
                    versions=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, pyarrow=pa.__version__))
    save_json(out/"manifest.json", manifest)
    print(json.dumps({"summary":summary,"feature_count":len(feature_list),"default_count":len(manifest["default_features"])}, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    build(a.data, a.out)
