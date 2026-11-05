"""Verify all six candidates and copy model artifacts into GM_Ver1/."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import xgboost as xgb

from train.dataset import current_dataset
from GM_Ver1.common import Panel, SEED, sha256, write_json
from GM_Ver1.gru import predict as neural_predict
from GM_Ver1.predict import load_candidate
from GM_Ver1.run import FINAL_END, SPLITS, evaluate, stamp
from GM_Ver1.sync_outputs import copy_if_changed


def verify_predictions(path, panel, ids):
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert sha256(path) == metadata["sha256"]
    p = pd.read_parquet(path)
    origin = (p.ts.min()-pd.Timedelta(hours=1)).normalize()-pd.Timedelta(hours=7)
    assert pd.Timestamp(metadata["training_end"]) <= origin
    assert pd.Timestamp(metadata["selection_labels_end"]) <= origin
    assert len(p) == len(ids)*24
    assert not p.duplicated(["site_id", "ts"]).any()
    assert np.isfinite(p.prediction).all() and (p.prediction >= 0).all()
    expected_t = panel.times(ids).reshape(-1)
    assert np.array_equal(p.ts.dt.as_unit("ns").array.asi8, expected_t)
    expected_s = np.repeat(np.array(panel.sites)[panel.site_index[ids]], 24)
    assert np.array_equal(p.site_id, expected_s)
    assert np.all(p.groupby("ts").site_id.nunique().to_numpy() == panel.nsites)
    return p


def export(out, repo, dataset):
    panel = Panel(out / "cache" / "training")
    evaluation = Panel(out / "cache" / "evaluation")
    lock = json.loads((repo / "train" / "dataset_lock.json").read_text(encoding="utf-8"))
    data_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    assert sha256(dataset / "training_ready.parquet") == lock["ready_training"]["sha256"]
    assert panel.meta["source_sha256"] == lock["full_grid_sha256"]
    assert evaluation.meta["source_sha256"] == lock["evaluation_input_sha256"]
    source = ds.dataset(dataset / "training_ready.parquet", format="parquet")
    expected_rows = {end: source.count_rows(filter=ds.field("ts") <= pa.scalar(
                        pd.Timestamp(end).tz_convert("Asia/Seoul"), type=pa.timestamp("ns", tz="Asia/Seoul")))
                     for end in [s[1] for s in SPLITS]+[FINAL_END]}
    results, checks, manifest = [], [], []
    for family in ["XGBoost", "GRU", "TCN"]:
        fam = out / family
        trials = json.loads((fam / "trials.json").read_text())
        expected_trials = 10 if family == "XGBoost" else 8
        assert len(trials) == expected_trials
        public = repo / "GM_Ver1" / family
        public.mkdir(parents=True, exist_ok=True)
        write_json(public / "search_trials.json", trials)
        for v in [1, 2]:
            version = f"Ver{v}"
            local = fam / version
            complete = json.loads((local / "complete.json").read_text())
            assert complete["status"] == "complete" and complete["config_sha256"] == sha256(local / "config.json")
            config = json.loads((local / "config.json").read_text())
            # Version numbering must follow tuning rank, not confirmation scores.
            assert config["trial"] == trials[v-1]["trial"]
            assert config["features"] == panel.features == evaluation.features
            scores, fits = {}, {}
            for split, end, left, right in SPLITS:
                ids = panel.indices(start=stamp(left), end=stamp(right), labeled=False)
                pred_file = local / split / "predictions.parquet"
                assert json.loads(pred_file.with_suffix(".json").read_text())["training_end"] == end
                verify_predictions(pred_file, panel, ids)
                scores[split] = evaluate(panel, ids, pred_file, dataset)
            for stage, end in [(s[0], s[1]) for s in SPLITS]+[("final", FINAL_END)]:
                fit = json.loads((local / stage / "fit.json").read_text())
                weights = local / stage / ("model.ubj" if family == "XGBoost" else "model.pt")
                assert fit["training_end"] == end and fit["eligible_rows"] == expected_rows[end]
                assert sha256(weights) == fit["model_sha256"]
                fits[stage] = fit
                manifest.append(dict(path=weights.relative_to(out).as_posix(), sha256=fit["model_sha256"], bytes=weights.stat().st_size))
            all_ids = np.arange(len(evaluation.X))
            p = verify_predictions(local / "final" / "predictions.parquet", evaluation, all_ids)
            assert json.loads((local / "final" / "predictions.json").read_text())["training_end"] == FINAL_END
            assert p.ts.min() == pd.Timestamp(data_manifest["summary"]["evaluation"]["start"])
            assert p.ts.max() == pd.Timestamp(data_manifest["summary"]["evaluation"]["end"])
            assert (p.groupby("site_id").size() == 8760).all()
            model, scaler = load_candidate(out, family, version)
            ids = np.sort(np.random.default_rng(SEED).choice(all_ids, 64, replace=False))
            if family == "XGBoost":
                x = np.array(evaluation.X[ids]).reshape(-1, len(evaluation.features))
                pred = np.maximum(model.predict(xgb.DMatrix(x, feature_names=evaluation.features, nthread=4)), 0)
                pred = pred.reshape(-1, 24) * evaluation.capacity[evaluation.site_index[ids], None]
            else:
                pred = neural_predict(model, evaluation, ids, scaler)
            expected = p.prediction.to_numpy().reshape(-1, 24)[ids]
            np.testing.assert_allclose(pred, expected, rtol=2e-5, atol=2e-4)
            del model, p
            write_json(local / "metrics.json", scores)
            target = public / version
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(local, target, dirs_exist_ok=True, copy_function=copy_if_changed)
            for stage, fit in fits.items():
                name = "model.ubj" if family == "XGBoost" else "model.pt"
                assert sha256(target / stage / name) == fit["model_sha256"]
            write_json(target / "config.json", config)
            write_json(target / "metrics.json", scores)
            write_json(target / "fit_summary.json", fits)
            checks.append(dict(family=family, version=version, full_history_rows=expected_rows[FINAL_END],
                forecast_rows=2_216_280, unique_full_grid=True, finite_nonnegative=True,
                checkpoint_reload_predictions_match=True, exact_raw_label_scoring=True,
                all_stage_hashes_match=True, all_training_boundaries_match=True))
            row = dict(model=f"{family}/{version}", tuning_mae_kwh=config["validation_mae_kwh"])
            for split, result in scores.items():
                row[split] = dict(site_mae_kwh=result["site_mae_kwh"], site_rmse_kwh=result["site_rmse_kwh"],
                    complete_hours=result["reference_groups"]["complete_hours"],
                    reference_penalty_krw=result["reference_groups"]["penalty_krw"])
            results.append(row)
    report = dict(status="PASS", expected_training_rows=expected_rows, candidates=checks,
                  dataset_ready_sha256=lock["ready_training"]["sha256"],
                  training_grid_sha256=lock["full_grid_sha256"],
                  evaluation_input_sha256=lock["evaluation_input_sha256"],
                  scope="6 forecasting candidates; no stacking or VPP optimization; local verification is not official scoring")
    write_json(out / "verification.json", report)
    write_json(repo / "GM_Ver1" / "verification.json", report)
    write_json(out / "comparison.json", results)
    write_json(repo / "GM_Ver1" / "comparison.json", results)
    write_json(out / "weights_manifest.json", manifest)
    lines = ["# 예측 모델 비교 결과", "", "후보는 2023년 튜닝 MAE 순으로 고정했다. 최종 모델은 전 기간의 유효 이력으로 다시 학습했다. 튜닝 마지막 타깃이 2024-01-01 00시이므로 2024년 확인은 1월 2일부터 사용한다.", "",
             "| 후보 | 튜닝 MAE(kWh) | 2024 MAE | 2025.1~5 MAE | 2024 참조 벌금(원) | 2025.1~5 참조 벌금(원) |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        a, b = r["confirm_2024"], r["check_2025"]
        lines.append(f"| {r['model']} | {r['tuning_mae_kwh']:.4f} | {a['site_mae_kwh']:.4f} | {b['site_mae_kwh']:.4f} | {a['reference_penalty_krw']:,.0f} | {b['reference_penalty_krw']:,.0f} |")
    lines += ["", "2024 그룹 점수는 178시간, 2025년 그룹 점수는 3,624시간의 완전 실적만 계산한다. 2024의 짧은 겨울 구간과 2025년 1~5월을 연간 성능으로 해석할 수 없다. 과거 연구에서 확인한 기간이므로 연구 전체에서 미사용인 테스트가 아니다.", "",
              f"최종 학습 타깃 마감: {FINAL_END}. 모든 후보가 동일한 {expected_rows[FINAL_END]:,}개 유효 실적을 사용했다. 전체 학습 모델은 표본 추출하지 않았다. 탐색 단계의 표본 사용과 구별한다.", "",
              "각 모델의 상세 설정·실험 이력·월별 지표·학습 행 수·해시는 해당 폴더 JSON을 참조한다. 최종 평가 예측은 후보마다 2,216,280행이며 실적이 없어 채점하지 않았다. 검증 점수는 로컬 재현 결과이며 공식 채점이 아니다."]
    (repo / "GM_Ver1" / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    shutil.copy2(repo / "GM_Ver1" / "RESULTS.md", out / "RESULTS.md")
    shutil.copy2(out / "environment.json", repo / "GM_Ver1" / "environment.json")
    shutil.copy2(out / "weights_manifest.json", repo / "GM_Ver1" / "weights_manifest.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", type=Path, default=None)
    a = parser.parse_args()
    export(a.out, a.repo, a.dataset or current_dataset())
