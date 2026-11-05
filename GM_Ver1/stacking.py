"""Chronological six-model Ridge stacking without in-sample base predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from train.dataset import current_dataset, score_predictions
from GM_Ver1.common import TZ, local_output, sha256, write_json
from GM_Ver1.run import FINAL_END, SELECTION_LABEL_END
from GM_Ver1.sync_outputs import copy_if_changed

BASE_MODELS = [f"{family}/Ver{v}" for family in ["XGBoost", "GRU", "TCN"] for v in [1, 2]]
ALPHAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, .1, 1.]
INNER_END = "2024-09-30 17:00:00+09:00"
META_CHECK_END = "2024-12-31 17:00:00+09:00"


def ridge_fit(x, y, weights, alpha):
    """Weighted mean square loss + alpha*||standardized coefficients||².

    Solve the six-dimensional normal equations directly. No iterative gradient
    optimizer or target-dependent transformation fitted outside the train fold.
    """
    if alpha <= 0 or len(x) != len(y) or len(y) != len(weights):
        raise ValueError("Invalid Ridge inputs")
    total = float(weights.sum())
    mean = np.sum(x*weights[:, None], axis=0)/total
    mean_y = float(np.dot(y, weights)/total)
    scale = np.sqrt(np.sum((x-mean)**2*weights[:, None], axis=0)/total)
    scale = np.maximum(scale, 1e-8)
    z = (x-mean)/scale
    gram = z.T@(z*weights[:, None])/total
    rhs = z.T@((y-mean_y)*weights)/total
    coef_scaled = np.linalg.solve(gram + alpha*np.eye(x.shape[1]), rhs)
    coef = coef_scaled/scale
    intercept = float(mean_y-mean@coef)
    return dict(method="weighted_standardized_ridge_closed_form", alpha=float(alpha),
        feature_order=BASE_MODELS, coefficients=coef.tolist(), intercept=intercept,
        training_mean=mean.tolist(), training_scale=scale.tolist(),
        weight_policy="capacity_kw/170", objective="weighted mean CF squared error + alpha*L2",
        coefficients_are_convex_weights=False, training_rows=len(y))


def ridge_predict(model, x, capacity):
    prediction = (x@np.array(model["coefficients"]) + model["intercept"])*capacity
    return np.maximum(prediction, 0)


def load_level1(out, dataset, stage, labels=True):
    if labels and stage == "final":
        raise ValueError("Final base fits cannot provide stacking training predictions")
    keys, x, provenance = None, None, []
    roster = pd.read_parquet(dataset / "reference_composition.parquet")
    cap_map = roster.set_index("site_id").capacity_kw
    for j, name in enumerate(BASE_MODELS):
        directory = out / name / stage
        path = directory / "predictions.parquet"
        metadata = json.loads((directory / "predictions.json").read_text())
        if sha256(path) != metadata["sha256"]:
            raise ValueError("Base prediction SHA256 mismatch")
        if labels and not metadata["role"].startswith("out_of_time_forecast"):
            raise ValueError("Meta training requires out-of-time predictions")
        p = pd.read_parquet(path)
        if p.duplicated(["site_id", "ts"]).any() or not np.isfinite(p.prediction).all():
            raise ValueError("Invalid base prediction keys/values")
        if keys is None:
            keys = p[["site_id", "ts"]].copy()
            capacity = keys.site_id.map(cap_map).to_numpy(dtype=np.float64)
            if not np.isfinite(capacity).all() or (capacity <= 0).any():
                raise ValueError("Unknown or invalid capacity")
            x = np.empty((len(p), len(BASE_MODELS)), dtype=np.float64)
        elif not p[["site_id", "ts"]].equals(keys):
            raise ValueError("Base forecast keys/order differ; cannot silently concatenate")
        x[:, j] = p.prediction.to_numpy()/capacity
        origin = (keys.ts.min()-pd.Timedelta(hours=1)).normalize()-pd.Timedelta(hours=7)
        if pd.Timestamp(metadata["training_end"]) > origin or pd.Timestamp(metadata["selection_labels_end"]) > origin:
            raise ValueError("Base training or selection labels are later than prediction origin")
        provenance.append(dict(model=name, stage=stage, prediction_sha256=metadata["sha256"],
            training_end=metadata["training_end"], rows=len(p)))
    y = None
    if labels:
        arrow_type = pa.timestamp("ns", tz=TZ)
        filt = ((ds.field("ts") >= pa.scalar(keys.ts.min(), type=arrow_type)) &
                (ds.field("ts") <= pa.scalar(keys.ts.max(), type=arrow_type)))
        raw = ds.dataset(dataset / "training_data.parquet", format="parquet").to_table(
            columns=["site_id", "ts", "energy_kwh"], filter=filt).to_pandas()
        joined = keys.merge(raw, on=["site_id", "ts"], how="left", validate="one_to_one", sort=False)
        y = joined.energy_kwh.to_numpy(dtype=np.float64)
    return keys, x, y, capacity, provenance


def fit_at(keys, x, y, capacity, alpha, end):
    eligible = np.isfinite(y) & (y >= 0) & (keys.ts <= pd.Timestamp(end)).to_numpy()
    model = ridge_fit(x[eligible], y[eligible]/capacity[eligible], capacity[eligible]/170., alpha)
    model.update(training_end=end, training_first_observed=str(keys.ts[eligible].min()),
                 training_last_observed=str(keys.ts[eligible].max()),
                 target_policy="observed nonnegative only; true zero retained; no imputed targets")
    assert pd.Timestamp(model["training_last_observed"]) <= pd.Timestamp(end)
    return model


def score(keys, prediction, y, dataset):
    valid = np.isfinite(y)
    rows = keys.loc[valid].copy()
    rows["prediction"], rows["actual"] = prediction[valid], y[valid]
    err = rows.prediction-rows.actual
    settlement, _ = score_predictions(rows,
        pd.read_parquet(dataset / "reference_composition.parquet"),
        pd.read_parquet(dataset / "training_weights.parquet"))
    return dict(site_mae_kwh=float(err.abs().mean()), site_rmse_kwh=float(np.sqrt((err**2).mean())),
        observed_rows=int(valid.sum()), reference_groups=settlement)


def save_prediction(path, keys, prediction, training_end, role):
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Invalid stacked prediction")
    path.parent.mkdir(parents=True, exist_ok=True)
    p = keys.copy()
    p["prediction"] = prediction
    p.to_parquet(path, index=False, compression="zstd")
    write_json(path.with_suffix(".json"), dict(rows=len(p), training_end=training_end,
        sha256=sha256(path), role=role, finite_nonnegative=True, unique_keys=not p.duplicated(["site_id", "ts"]).any()))


def run(out, dataset, publish):
    out = local_output(out)
    verification = json.loads((out / "verification.json").read_text())
    if verification["status"] != "PASS" or len(verification["candidates"]) != 6:
        raise ValueError("All six base candidates must pass verification first")
    dest = out / "Stacking" / "Ver1"
    dest.mkdir(parents=True, exist_ok=True)
    protocol = dict(base_models=BASE_MODELS, alpha_grid=ALPHAS,
        inner_training_end=INNER_END, inner_validation="2024-10-01 01:00 through 2024-12-31 17:00 KST",
        selection_metric="observed nonnegative-target site MAE in kWh",
        check_meta_training_end=META_CHECK_END, check_period="2025 Jan-May",
        final_meta_training_end=FINAL_END, final_meta_features="out-of-time 2024 and 2025 base predictions only",
        gradient_descent_weight_optimization=False, vpp_optimization=False,
        caution="2024/2025 were inspected in previous research; not globally untouched data")
    write_json(dest / "protocol.json", protocol)
    k24, x24, y24, c24, provenance24 = load_level1(out, dataset, "confirm_2024")
    mask = (np.isfinite(y24) & (y24 >= 0) &
            (k24.ts >= pd.Timestamp("2024-10-01 01:00:00+09:00")).to_numpy() &
            (k24.ts <= pd.Timestamp(META_CHECK_END)).to_numpy())
    trials = []
    for alpha in ALPHAS:
        model = fit_at(k24, x24, y24, c24, alpha, INNER_END)
        p = ridge_predict(model, x24[mask], c24[mask])
        trials.append(dict(alpha=alpha, validation_mae_kwh=float(np.abs(p-y24[mask]).mean()),
                           inner_training_rows=model["training_rows"], validation_rows=int(mask.sum())))
    trials.sort(key=lambda r: (r["validation_mae_kwh"], -r["alpha"]))
    alpha = trials[0]["alpha"]
    write_json(dest / "alpha_trials.json", trials)
    print(f"Stacking selected alpha={alpha}, Q4 MAE={trials[0]['validation_mae_kwh']:.6f}", flush=True)
    check_model = fit_at(k24, x24, y24, c24, alpha, META_CHECK_END)
    write_json(dest / "check_2025" / "model.json", check_model)
    k25, x25, y25, c25, provenance25 = load_level1(out, dataset, "check_2025")
    check_pred = ridge_predict(check_model, x25, c25)
    save_prediction(dest / "check_2025" / "predictions.parquet", k25, check_pred,
                    META_CHECK_END, "out_of_time_meta_forecast; settings frozen before 2025")
    metrics = score(k25, check_pred, y25, dataset)
    write_json(dest / "check_2025" / "metrics.json", metrics)
    comparison = {}
    for name in BASE_MODELS:
        original = json.loads((out / name / "metrics.json").read_text())["check_2025"]
        comparison[name] = {key: original[key] for key in ["site_mae_kwh", "site_rmse_kwh", "reference_groups"]}
    comparison["uniform_mean_6"] = score(k25, x25.mean(axis=1)*c25, y25, dataset)
    comparison["Ridge_stacking"] = metrics
    best_mae = min(comparison[n]["site_mae_kwh"] for n in BASE_MODELS)
    best_penalty = min(comparison[n]["reference_groups"]["penalty_krw"] for n in BASE_MODELS)
    result = dict(check_2025=comparison,
        stacking_mae_improvement_vs_best_single_pct=100*(1-metrics["site_mae_kwh"]/best_mae),
        stacking_penalty_improvement_vs_best_single_pct=100*(1-metrics["reference_groups"]["penalty_krw"]/best_penalty),
        validation_used_to_reselect_settings=False)
    write_json(dest / "comparison.json", result)
    # Final meta training includes all eligible OOT rows now known at the first
    # evaluation cutoff, including the 2024 closing hours embargoed above.
    all_keys = pd.concat([k24, k25], ignore_index=True)
    all_x, all_y, all_cap = np.vstack([x24, x25]), np.concatenate([y24, y25]), np.concatenate([c24, c25])
    if all_keys.duplicated(["site_id", "ts"]).any():
        raise ValueError("Overlapping meta-training periods")
    final_model = fit_at(all_keys, all_x, all_y, all_cap, alpha, FINAL_END)
    final_model["provenance"] = provenance24+provenance25
    write_json(dest / "final" / "model.json", final_model)
    del all_x, all_y, all_cap, all_keys, x24, x25
    ke, xe, _, ce, provenance_e = load_level1(out, dataset, "final", labels=False)
    final_pred = ridge_predict(final_model, xe, ce)
    save_prediction(dest / "final" / "predictions.parquet", ke, final_pred,
                    FINAL_END, "competition evaluation; no actual labels available")
    saved_model = json.loads((dest / "final" / "model.json").read_text())
    sample = np.arange(0, len(xe), 997)
    np.testing.assert_allclose(ridge_predict(saved_model, xe[sample], ce[sample]), final_pred[sample], atol=1e-9)
    saved = pd.read_parquet(dest / "final" / "predictions.parquet")
    assert len(saved) == 2_216_280 and not saved.duplicated(["site_id", "ts"]).any()
    assert (saved.groupby("site_id").size() == 8760).all()
    assert np.isfinite(saved.prediction).all() and (saved.prediction >= 0).all()
    write_json(dest / "verification.json", dict(status="PASS", final_rows=len(saved),
        final_meta_training_rows=final_model["training_rows"], final_meta_training_end=FINAL_END,
        out_of_time_training_only=True, checkpoint_reload_matches=True,
        raw_training_targets_uploaded=False, evaluation_provenance=provenance_e))
    lines = ["# Ridge 스태킹 Ver1", "", "6개 base 모델의 시간 외 예측으로 학습한 선형 메타모델이다.", "",
        "2024년 1월 2일~9월 30일 17시로 적합하고 10월 1일~12월 31일 17시 MAE로 정규화 강도 alpha를 선택했다. 해당 값으로 2024년 마감 이전 실적을 다시 학습해 2025년 1~5월을 평가했다. 2025년 점수로 설정을 바꾸지 않았다.", "",
        "| 모델 | 2025.1~5 MAE(kWh) | 참조 그룹 벌금(원) |", "|---|---:|---:|"]
    for name, row in comparison.items():
        lines.append(f"| {name} | {row['site_mae_kwh']:.4f} | {row['reference_groups']['penalty_krw']:,.0f} |")
    lines += ["", f"선택 alpha: {alpha}. 최종 메타모델 학습은 {final_model['training_rows']:,}개 유효 시간 외 예측 행을 사용했다. Base 모델은 전체 유효 학습 이력을 사용하지만, 메타모델에는 시간 외 예측이 있는 2024~2025년 행만 넣는다.", "",
        "최종 메타모델은 2024~2025년의 시간 외 예측과 첫 평가일 마감 이전 정답으로 재학습했다. 이를 6개 최종 base 모델의 평가 기간 예측에 적용했다. 최종 파일은 `Ver1/final/model.json`과 `Ver1/final/predictions.parquet`다.", "",
        "계수는 음수도 가능한 Ridge 회귀계수이며 합이 1인 구성비가 아니다. 발전소 용량으로 정규화한 6개 예측에 적용하고, kWh로 복원한 뒤 음수만 0으로 자른다. 정규화 통계는 메타 학습 fold에서만 적합한다.", "",
        "참조 그룹은 고정하며 2025년 벌금은 실적이 완전한 3,624시간만 계산한다. 2024~2025년은 과거 연구에서 확인한 기간으로 연구 전체의 미사용 테스트가 아니다. 평가 기간 정답은 없어 최종 예측의 성능은 알 수 없다. 별도의 경사하강법 가중치 최적화와 VPP 구성 최적화는 이 단계에 포함하지 않았다.", "",
        "실행: `python -m GM_Ver1.stacking --out <기존 로컬 결과 폴더>`", "",
        "[Ridge 공식 설명](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) · [Stacking의 학습 예측 주의사항](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)"]
    (dest.parent / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if publish:
        shutil.copytree(dest.parent, publish / "Stacking", dirs_exist_ok=True, copy_function=copy_if_changed)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--publish", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run(args.out, args.dataset or current_dataset(), args.publish)
