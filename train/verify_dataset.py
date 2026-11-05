"""Independent full-grid, raw-label, source-hash and sampled-forecast checks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from audit_data import check_time, cutoff, digest, save_json


def verify(data, out):
    data, out = Path(data), Path(out)
    m = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    for name, r in m["sources"].items():
        if r["exists"] and digest(data/name) != r["sha256"]:
            raise AssertionError("Source changed: "+name)
    features = m["features"]
    forbidden = {"energy", "energy_kwh", "energy_per_capacity", "smp", "k", "current_group", "target_observed", "target_eligible"}
    assert not (set(features) & forbidden)
    assert not any("observed" in n or "quality" in n for n in features)
    sites = pd.read_csv(data/"sites.csv").sort_values("site_id")
    evidence = {}
    for phase, r in m["summary"].items():
        path = out/r["file"]
        assert digest(path) == r["sha256"]
        parquet = pq.ParquetFile(path)
        total = observed = negative = zeros = missing = eligible = sample_forecasts = 0
        last_time = None
        all_missing = pd.Series(0, index=features, dtype="int64")
        for group in range(parquet.num_row_groups):
            f = parquet.read_row_group(group).to_pandas()
            check_time(f.ts)
            assert not f.duplicated(["site_id", "ts"]).any()
            assert f.ts.is_monotonic_increasing
            assert last_time is None or f.ts.min() == last_time+pd.Timedelta(hours=1)
            last_time = f.ts.max()
            hours = pd.date_range(f.ts.min(), f.ts.max(), freq="h")
            assert len(f) == len(hours)*len(sites)
            assert np.array_equal(f.site_id.to_numpy().reshape(-1, len(sites)), np.tile(sites.site_id.to_numpy(), (len(hours), 1)))
            assert np.array_equal(pd.DatetimeIndex(f.ts).asi8, hours.repeat(len(sites)).asi8)
            assert np.array_equal(pd.DatetimeIndex(f.issue_cutoff).asi8, cutoff(f.ts).asi8)
            assert not np.isinf(f[features].to_numpy()).any()
            all_missing += f[features].isna().sum()
            np.testing.assert_array_equal(f.target_observed, f.energy_kwh.notna())
            np.testing.assert_array_equal(f.target_negative, f.energy_kwh.lt(0))
            np.testing.assert_array_equal(f.target_eligible, f.target_observed & ~f.target_negative)
            np.testing.assert_array_equal(f.target_above_capacity, f.energy_kwh.gt(f.capacity_kw))
            # Raw targets are float64 and must survive exactly, including negatives/zeros.
            if phase == "training":
                raw = pd.read_parquet(data/"train/generation.parquet", filters=[("ts", ">=", f.ts.min()), ("ts", "<=", f.ts.max())])
                matched = raw.merge(f[["site_id", "ts", "energy_kwh"]], on=["site_id", "ts"], validate="one_to_one", how="left")
                assert len(matched) == len(raw) == int(f.target_observed.sum())
                np.testing.assert_array_equal(matched.energy, matched.energy_kwh)
            else:
                assert f.energy_kwh.isna().all() and not f.target_eligible.any()
            # Re-read provider round17 values for a deterministic sample, independent of feature code.
            sample = f.iloc[np.linspace(0, len(f)-1, min(257, len(f)), dtype=int)].merge(
                sites[["site_id", "wa_station", "wb_station"]], on="site_id", validate="many_to_one")
            folder = "train" if phase == "training" else "test"
            for provider in ("wa", "wb"):
                raw = pd.read_parquet(data/folder/("forecast_"+provider+".parquet"),
                    filters=[("ts", ">=", f.ts.min()), ("ts", "<=", f.ts.max()), ("round", "==", 17)])
                cols = [c for c in raw.columns if c not in [provider+"_station", "round", "ts"]]
                paired = sample.merge(raw.drop(columns="round"), on=[provider+"_station", "ts"],
                                      how="left", suffixes=("", "__raw"), validate="many_to_one")
                for c in cols:
                    np.testing.assert_array_equal(paired[c].to_numpy(), paired[c+"__raw"].astype("float32").to_numpy())
                sample_forecasts += len(paired)
            total += len(f)
            observed += int(f.target_observed.sum())
            negative += int(f.target_negative.sum())
            zeros += int(f.energy_kwh.eq(0).sum())
            missing += int((~f.target_observed).sum())
            eligible += int(f.target_eligible.sum())
            print(f"VERIFY {phase} {group+1}/{parquet.num_row_groups}: keys/labels/forecasts PASS", flush=True)
        actual = dict(rows=total, observed=observed, negative=negative, zero_targets=zeros,
                      missing_targets=missing, eligible=eligible)
        for name, value in actual.items():
            assert value == r[name], (name, value, r[name])
        assert int(all_missing.sum()) == r["feature_missing_cells"]
        all_missing.rename("missing_cells").to_csv(out/(phase+"_feature_missing.csv"))
        evidence[phase] = dict(**actual, row_groups=parquet.num_row_groups,
                               forecast_sample_rows=sample_forecasts, sha256=r["sha256"])
    source_target = m["sources"]["train/generation.parquet"]
    assert evidence["training"]["observed"] == source_target["rows"]
    assert evidence["training"]["negative"] == source_target["negative_energy_rows"]
    assert evidence["training"]["zero_targets"] == source_target["zero_energy_rows"]
    splits = pd.read_csv(out/"validation_splits.csv")
    for _, s in splits.iterrows():
        assert pd.Timestamp(s.train_end) == cutoff([pd.Timestamp(s.validation_start)])[0]
        assert pd.Timestamp(s.train_end) < pd.Timestamp(s.validation_start)
    checks = dict(status="PASS", evidence=evidence, features=len(features),
                  default_features=len(m["default_features"]), folds=len(splits),
                  source_hashes_unchanged=True, raw_labels_exact=True, complete_grid_unique=True,
                  future_observations_in_X=False, realized_smp_in_X=False, penalty_coefficient=0.1,
                  limitation="structural/local verification, not feature performance or official approval")
    save_json(out/"verification.json", checks)
    (out/"quality_report.md").write_text(
        "# 학습 데이터 품질 검사\n\n상태: PASS (로컬 구조 검사)\n\n"
        +"```json\n"+json.dumps(checks, ensure_ascii=False, indent=2)+"\n```\n\n"
        +"음수 40행은 원본 보존, 기본 fit에서 제외. 0 타깃은 유지. 결측 타깃을 채우지 않음.\n"
        +"시각 미정렬 예보는 무단 반올림하지 않음. 관측 충돌은 모델 X와 결합하지 않음.\n"
        +"각 모델은 manifest의 feature 목록으로만 입력을 선택하고, 전처리를 학습 fold 안에서 fit해야 함.\n"
        +"특성별 손실 개선·악화, 중요도, 하이퍼파라미터 실험은 아직 실행하지 않음.\n", encoding="utf-8")
    m["status"] = "verified_frozen"
    m["verification_sha256"] = digest(out/"verification.json")
    save_json(out/"manifest.json", m)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    return checks


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    verify(a.data, a.out)
