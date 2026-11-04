"""Read-only audit of the supplied competition files. No network access."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

TZ = "Asia/Seoul"
FILES = ["README.md", "sites.csv", "smp.csv", "train/smp.csv",
         "train/generation.parquet", "train/forecast_wa.parquet",
         "train/forecast_wb.parquet", "train/observed_wa.parquet",
         "train/observed_wb.parquet", "test/forecast_wa.parquet",
         "test/forecast_wb.parquet"]
PRECIP = ["wb_precip_prob", "wb_rain_prob", "wb_snow_prob", "wb_ice_prob",
          "wb_total_liq", "wb_rain", "wb_snow", "wb_ice"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=str, allow_nan=False), encoding="utf-8")


def check_time(ts, *, allow_off_grid=False):
    ix = pd.DatetimeIndex(ts)
    if ix.tz is None or str(ix.tz) not in (TZ, "UTC+09:00"):
        raise ValueError(f"Expected timezone-aware KST, got {ix.tz}")
    if (not allow_off_grid and not hourly_mask(ix).all()) or ix.hasnans:
        raise ValueError("Non-hourly or missing timestamps")
    return ix


def hourly_mask(ts):
    ix = pd.DatetimeIndex(ts)
    return ((ix.minute == 0) & (ix.second == 0) & (ix.microsecond == 0)
            & (ix.nanosecond == 0) & ~ix.isna())


def cutoff(ts):
    """00:00 belongs to the preceding right-closed operating day."""
    return (pd.DatetimeIndex(ts) - pd.Timedelta(hours=1)).normalize() - pd.Timedelta(hours=7)


def audit(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    if data == out or data in out.parents:
        raise ValueError("Audit output must be separate from supplied data")
    out.mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(data / "sites.csv")
    reports, coverage = {}, []
    for name in FILES:
        path = data / name
        r = {"exists": path.is_file()}
        reports[name] = r
        if not path.is_file():
            continue
        r.update(bytes=path.stat().st_size, sha256=digest(path))
        if name.endswith(".md"):
            continue
        f = pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path)
        if "ts" in f:
            if not name.endswith(".parquet"):
                f["ts"] = pd.to_datetime(f.ts)
            ts = check_time(f.ts, allow_off_grid=True)
            off_grid = ~hourly_mask(ts)
            r["off_hour_rows"] = int(off_grid.sum())
            if off_grid.any():
                f.loc[off_grid].to_parquet(out / (name.replace("/", "_") + ".off_grid.parquet"), index=False)
            r.update(start=str(ts.min()), end=str(ts.max()), timezone=str(ts.tz),
                     right_closed=True, interval="1h")
        r.update(rows=len(f), columns=len(f.columns),
                 schema={c: str(t) for c, t in f.dtypes.items()},
                 missing={c: {"count": int(f[c].isna().sum()),
                              "ratio": float(f[c].isna().mean())} for c in f})
        keys = ([c for c in ("wa_station", "wb_station", "site_id", "round", "ts") if c in f])
        r["keys"] = keys
        r["missing_key_rows"] = int(f[keys].isna().any(axis=1).sum())
        duplicates = f.duplicated(keys, keep=False)
        r["duplicate_extra_rows"] = int(f.duplicated(keys).sum())
        r["duplicate_member_rows"] = int(duplicates.sum())
        if duplicates.any():
            d = f.loc[duplicates]
            unique_records = d.drop_duplicates()
            conflict = unique_records.duplicated(keys, keep=False)
            r["conflicting_key_count"] = int(unique_records.loc[conflict, keys].drop_duplicates().shape[0])
            r["identical_extra_rows"] = int(d.duplicated().sum())
            r["duplicate_policy"] = "exact duplicate records may be collapsed; conflicting records quarantined; never merged into model X"
            d.to_parquet(out / (name.replace("/", "_") + ".duplicates.parquet"), index=False)
        else:
            r["conflicting_key_count"] = 0
        numeric = f.select_dtypes(include="number")
        r["ranges"] = {}
        for c in numeric:
            v = numeric[c]
            r["ranges"][c] = {"min": None if v.dropna().empty else float(v.min()),
                              "max": None if v.dropna().empty else float(v.max()),
                              "infinite": int(np.isinf(v).sum())}
        if "ts" in f:
            entity = next((c for c in ("site_id", "wa_station", "wb_station") if c in f), None)
            by = ([entity] if entity else []) + (["round"] if "round" in f else [])
            expected_hours = len(pd.date_range(f.ts.min(), f.ts.max(), freq="h"))
            expected_entities = (len(sites) if entity == "site_id" else sites[entity].nunique()) if entity else 1
            expected_rounds = 2 if "round" in f else 1
            r["expected_full_grid_rows"] = int(expected_hours * expected_entities * expected_rounds)
            r["absent_full_grid_keys"] = int(r["expected_full_grid_rows"] - len(f.loc[hourly_mask(f.ts)].drop_duplicates(keys)))
            groups = f.groupby(by, observed=True) if by else [((), f)]
            for k, g in groups:
                k = k if isinstance(k, tuple) else (k,)
                unique_hours = g.loc[hourly_mask(g.ts), "ts"].nunique()
                inner_grid = len(pd.date_range(g.ts.min(), g.ts.max(), freq="h"))
                coverage.append(dict(file=name, **dict(zip(by, k)), rows=len(g),
                    start=str(g.ts.min()), end=str(g.ts.max()), unique_hours=unique_hours,
                    missing_within_entity_span=inner_grid-unique_hours,
                    missing_in_file_span=expected_hours-unique_hours))
            if "round" in f:
                r["rounds"] = sorted(map(int, f["round"].unique()))
                r["issued_by_cutoff"] = bool(set(r["rounds"]) <= {10, 17})
        if name == "sites.csv":
            r["site_count"] = int(f.site_id.nunique())
            r["station_counts"] = {c: int(f[c].nunique()) for c in ("wa_station", "wb_station")}
            r["capacity_sum_kw"] = float(f.capacity_kw.sum())
            if len(f) != 253 or f.site_id.isna().any() or f.site_id.duplicated().any():
                raise ValueError("Invalid site roster")
        if "energy" in f:
            r.update(negative_energy_rows=int((f.energy < 0).sum()),
                     zero_energy_rows=int((f.energy == 0).sum()))
            cap = f.site_id.map(sites.set_index("site_id").capacity_kw)
            r["energy_above_capacity_kwh"] = int((f.energy > cap).sum())
            r["unknown_sites"] = int(cap.isna().sum())
            if r["duplicate_extra_rows"] or r["unknown_sites"]:
                raise ValueError("Invalid target keys")
            f.loc[f.energy < 0].to_parquet(out / "negative_targets.parquet", index=False)
            counts = f.groupby("ts").site_id.nunique()
            complete = counts[counts == len(sites)]
            r["complete_panel_hours"] = len(complete)
            r["complete_panel_by_operating_year"] = {
                str(y): int(n) for y, n in complete.groupby((complete.index-pd.Timedelta(hours=1)).year).size().items()}
        if name.endswith("forecast_wa.parquet"):
            r["temperature_three_columns_identical"] = bool(
                f.wa_temp.equals(f.wa_temp_max) and f.wa_temp.equals(f.wa_temp_min))
        if "forecast_wb" in name:
            m = f[PRECIP].isna()
            r["precip_missing_per_column"] = m.sum().astype(int).to_dict()
            r["precip_all_8_missing"] = int(m.all(axis=1).sum())
            r["precip_partially_missing"] = int((m.any(axis=1) & ~m.all(axis=1)).sum())
        if "smp" in f:
            r["zero_smp_hours"] = int((f.smp == 0).sum())
            r["annual_smp"] = f.assign(year=(f.ts-pd.Timedelta(hours=1)).dt.year).groupby("year").smp.agg(["count", "mean", "median", "max"]).to_dict("index")
        if "observed" in name:
            rules = ({"wa_wind_speed": 50} if "observed_wa" in name else
                     {"wb_cloud": 100, "wb_uv_idx": 15, "wb_vis": 50, "wb_wind_speed": 200})
            r["warning_outliers"] = {c+">"+str(v): int((f[c] > v).sum()) for c, v in rules.items()}
            if "wb_pressure" in f:
                r["warning_outliers"]["wb_pressure<900"] = int((f.wb_pressure < 900).sum())
        print(f"AUDIT {name}: rows={len(f):,}, duplicate_extra={r['duplicate_extra_rows']}", flush=True)
        del f
        gc.collect()
    result = {"files": reports,
              "missing_files": [n for n, r in reports.items() if not r["exists"]],
              "unverified": ["separate current problem-definition revisions and tie-breaking",
                             "actual generation/observation receipt times; no generation lags or observation-derived model inputs",
                             "external holiday data permission; holiday feature withheld"],
              "penalty": "0.1 * sum(SMP * k * abs(group_prediction - group_actual))"}
    save_json(out / "source_audit.json", result)
    pd.DataFrame(coverage).to_csv(out / "source_coverage.csv", index=False)
    lines = ["# 대회 제공 자료 감사", "", "원본은 수정하지 않음. 모든 시간은 KST, 1시간, 우측닫힘.", "",
             "벌금 = 0.1 × Σ SMP × k × |그룹 예측합 − 그룹 실적합|", "",
             "| 파일 | 행 | 열 | 중복 추가행 | 기간 |", "|---|---:|---:|---:|---|"]
    for name, r in reports.items():
        lines.append(f"| {name} | {r.get('rows', '-')} | {r.get('columns', '-')} | {r.get('duplicate_extra_rows', '-')} | {r.get('start', '')} ~ {r.get('end', '')} |")
    lines += ["", "자세한 스키마·단위별 범위·결측·중복·이상값: source_audit.json, source_coverage.csv.",
              "", "음수 타깃은 원본 보존, 기본 학습에서 제외하고 별도 플래그로 남김. 0 발전량은 유효 타깃.",
              "실측 중복은 exact duplicate와 conflicting key로 분류하며 모델 입력 조인에서 제외.",
              "", "## 미확인", *["- "+v for v in result["unverified"]]]
    (out / "data_audit.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    audit(a.data, a.out)
