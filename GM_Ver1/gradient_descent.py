"""Projected gradient descent for competition-weighted convex forecast blending."""
from __future__ import annotations

import argparse
import json
import importlib.metadata
import platform
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from train.dataset import current_dataset
from GM_Ver1.common import local_output, sha256, write_json
from GM_Ver1.run import FINAL_END
from GM_Ver1.stacking import BASE_MODELS, load_level1, score, save_prediction
from GM_Ver1.sync_outputs import copy_if_changed

TRAIN_END = "2025-02-28 17:00:00+09:00"
SELECT_START = "2025-03-01 01:00:00+09:00"
SELECT_END = "2025-03-31 17:00:00+09:00"
CHECK_START = "2025-04-01 01:00:00+09:00"
LAMBDAS = [0., 1e-4, 1e-3, 1e-2]
DELTAS = [.001, .005]
MODES = ["global", "peak_offpeak"]


def project_simplex(vector):
    """Euclidean projection onto nonnegative weights with unit sum."""
    v = np.asarray(vector, dtype=np.float64)
    ordered = np.sort(v)[::-1]
    cumulative = np.cumsum(ordered)-1.
    active = np.flatnonzero(ordered-cumulative/(np.arange(len(v))+1) > 0)
    theta = cumulative[active[-1]]/(active[-1]+1)
    return np.maximum(v-theta, 0.)


def loss_gradient(w, residual, price_weight, delta, regularization, prior):
    """Normalized smooth group L1 plus a fixed Ridge-prior L2 penalty."""
    error = residual@w
    smooth = np.sqrt(error*error+delta*delta)
    mass = price_weight.sum()
    loss = float(np.dot(price_weight, smooth-delta)/mass + regularization*np.sum((w-prior)**2))
    gradient = residual.T@(price_weight*error/smooth)/mass+2*regularization*(w-prior)
    return loss, gradient


def projected_descent(residual, price_weight, delta, regularization, prior, start,
                      max_steps=5000, tolerance=1e-9):
    w = project_simplex(start)
    step = 1.
    history = []
    converged = False
    for iteration in range(max_steps):
        loss, gradient = loss_gradient(w, residual, price_weight, delta, regularization, prior)
        for _ in range(50):
            candidate = project_simplex(w-step*gradient)
            change = candidate-w
            next_loss, _ = loss_gradient(candidate, residual, price_weight, delta, regularization, prior)
            bound = loss+gradient@change+(change@change)/(2*step)
            if next_loss <= bound+1e-15:
                break
            step *= .5
        else:
            raise RuntimeError("Projected-gradient line search failed")
        mapping_norm = float(np.linalg.norm(change)/step)
        w = candidate
        if iteration % 100 == 0:
            history.append(dict(iteration=iteration, loss=next_loss, step=step,
                                projected_gradient_norm=mapping_norm))
        if mapping_norm <= tolerance:
            converged = True
            break
        step = min(step*1.1, 1e5)
    loss, gradient = loss_gradient(w, residual, price_weight, delta, regularization, prior)
    stationarity = float(np.linalg.norm(w-project_simplex(w-gradient)))
    history.append(dict(iteration=iteration+1, loss=loss, projected_gradient_norm=stationarity))
    if not np.isfinite(w).all() or (w < -1e-12).any() or abs(w.sum()-1) > 1e-10:
        raise AssertionError("Invalid simplex solution")
    return dict(weights=w.tolist(), loss=loss, iterations=iteration+1,
        converged=converged or stationarity < 1e-7, stationarity=stationarity, history=history)


def aggregate_complete(keys, x, y, capacity, dataset):
    """Exclude an entire hour when any roster member's actual is unavailable."""
    roster = pd.read_parquet(dataset / "reference_composition.parquet")
    actual_count = keys.assign(observed=np.isfinite(y)).groupby("ts")["observed"].sum()
    complete = actual_count.index[actual_count == len(roster)]
    use = keys.ts.isin(complete).to_numpy()
    selected = keys.loc[use].copy()
    if selected.duplicated(["site_id", "ts"]).any():
        raise ValueError("Duplicate group objective keys")
    selected["group"] = selected.site_id.map(roster.set_index("site_id").current_group)
    if selected.group.isna().any() or set(selected.site_id) != set(roster.site_id):
        raise ValueError("Objective roster mismatch")
    selected["actual"] = y[use]
    for j in range(x.shape[1]):
        selected[f"p{j}"] = x[use, j]*capacity[use]
    group_names = sorted(roster.current_group.unique())
    grouped = selected.groupby(["ts", "group"])[["actual"]+[f"p{j}" for j in range(x.shape[1])]].sum()
    index = pd.MultiIndex.from_product([complete, group_names], names=["ts", "group"])
    grouped = grouped.reindex(index)
    if not np.isfinite(grouped.to_numpy()).all():
        raise ValueError("Incomplete objective group")
    weights = pd.read_parquet(dataset / "training_weights.parquet").set_index("ts").reindex(complete)
    if not np.isfinite(weights[["smp", "k"]]).all().all() or (weights.smp < 0).any():
        raise ValueError("Invalid settlement loss weights")
    if not np.array_equal(weights.k.to_numpy(), np.where(complete.hour.isin(range(11, 16)), 3, 1)):
        raise ValueError("KST peak-hour weight mismatch")
    values = grouped.to_numpy().reshape(len(complete), len(group_names), -1)
    scale = float(roster.capacity_kw.sum()/len(group_names))
    return dict(times=complete, residual=(values[:, :, 1:]-values[:, :, :1])/scale,
        price_weight=np.repeat((weights.smp*weights.k).to_numpy()[:, None], len(group_names), axis=1),
        scale=scale, group_names=group_names, excluded_hours=int(keys.ts.nunique()-len(complete)),
        eligible_hours=len(complete))


def concatenate_groups(a, b):
    if a["group_names"] != b["group_names"] or a["scale"] != b["scale"]:
        raise ValueError("Group contract changed")
    times = a["times"].append(b["times"])
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Overlapping or unordered group objective periods")
    return dict(times=times, residual=np.concatenate([a["residual"], b["residual"]]),
        price_weight=np.concatenate([a["price_weight"], b["price_weight"]]), scale=a["scale"],
        group_names=a["group_names"], excluded_hours=a["excluded_hours"]+b["excluded_hours"],
        eligible_hours=len(times))


def fit_weights(group, end, config, prior):
    eligible = group["times"] <= pd.Timestamp(end)
    times = group["times"][eligible]
    modes = ["all"] if config["mode"] == "global" else ["offpeak", "peak"]
    fits = {}
    for segment in modes:
        subset = np.ones(len(times), dtype=bool) if segment == "all" else np.isin(times.hour, range(11, 16))
        if segment == "offpeak":
            subset = ~subset
        residual = group["residual"][eligible][subset].reshape(-1, len(BASE_MODELS))
        q = group["price_weight"][eligible][subset].reshape(-1)
        if not len(q) or q.sum() <= 0:
            raise ValueError("No positive-price training objective")
        candidates = [projected_descent(residual, q, config["delta"], config["regularization"], prior, initial)
                      for initial in [prior, np.ones(len(BASE_MODELS))/len(BASE_MODELS)]]
        chosen = min(candidates, key=lambda row: row["loss"])
        chosen["hours"] = int(subset.sum())
        fits[segment] = chosen
    return dict(method="projected_gradient_descent_with_backtracking", config=config, fits=fits,
        feature_order=BASE_MODELS, prior=prior.tolist(), prior_source="Stacking/Ver1/check_2025/model.json",
        prior_training_end="2024-12-31 17:00:00+09:00", training_end=end,
        training_complete_hours=len(times), first_training_hour=str(times.min()), last_training_hour=str(times.max()),
        constraints="weights >= 0 and sum(weights) == 1; no intercept", fixed_reference_groups=True,
        SMP_in_forecast_features=False, group_scale_kw=group["scale"])


def weight_rows(model, times):
    if model["config"]["mode"] == "global":
        return np.broadcast_to(model["fits"]["all"]["weights"], (len(times), len(BASE_MODELS)))
    peak = np.isin(times.hour if isinstance(times, pd.DatetimeIndex) else times.dt.hour, range(11, 16))
    return np.where(peak[:, None], np.array(model["fits"]["peak"]["weights"]),
                    np.array(model["fits"]["offpeak"]["weights"]))


def predict(model, keys, x, capacity):
    return np.sum(x*weight_rows(model, keys.ts), axis=1)*capacity


def group_penalty(model, group, start=None, end=None):
    mask = np.ones(len(group["times"]), dtype=bool)
    if start:
        mask &= group["times"] >= pd.Timestamp(start)
    if end:
        mask &= group["times"] <= pd.Timestamp(end)
    w = weight_rows(model, group["times"][mask])
    error = np.sum(group["residual"][mask]*w[:, None, :], axis=2)*group["scale"]
    return float(.1*np.sum(group["price_weight"][mask]*np.abs(error)))


def weights_markdown(model):
    segments = list(model["fits"])
    lines = ["", "## 최종 가중치", "", "| 모델 | " + " | ".join(segments) + " |",
             "|---|" + "---:|"*len(segments)]
    for j, name in enumerate(BASE_MODELS):
        lines.append("| " + name + " | " + " | ".join(
            f"{model['fits'][segment]['weights'][j]*100:.3f}%" for segment in segments) + " |")
    return lines


def run(out, dataset, publish):
    out = local_output(out)
    dest = out / "GradientDescent" / "Ver1"
    dest.mkdir(parents=True, exist_ok=True)
    base_verification = json.loads((out / "verification.json").read_text())
    training_hash = sha256(dataset / "training_data.parquet")
    if training_hash != base_verification["training_grid_sha256"]:
        raise ValueError("Training labels no longer match the verified frozen base-model dataset")
    write_json(dest / "dataset_provenance.json", dict(training_grid_sha256=training_hash,
        settlement_weights_sha256=sha256(dataset / "training_weights.parquet"),
        reference_composition_sha256=sha256(dataset / "reference_composition.parquet")))
    write_json(dest / "environment.json", dict(python=platform.python_version(),
        packages={name:importlib.metadata.version(name) for name in ["numpy", "pandas", "pyarrow", "scipy"]},
        optimizer="deterministic full-batch NumPy projected gradient descent"))
    if json.loads((out / "Stacking" / "Ver1" / "verification.json").read_text())["status"] != "PASS":
        raise ValueError("Verified stacking artifacts are required")
    ridge = json.loads((out / "Stacking" / "Ver1" / "check_2025" / "model.json").read_text())
    if pd.Timestamp(ridge["training_end"]) > pd.Timestamp(TRAIN_END):
        raise ValueError("Ridge prior contains future training labels")
    prior = project_simplex(ridge["coefficients"])
    protocol = dict(base_models=BASE_MODELS, train_end=TRAIN_END, selection_start=SELECT_START,
        selection_end=SELECT_END, check_start=CHECK_START, final_training_end=FINAL_END,
        modes=MODES, regularization_grid=LAMBDAS, smooth_delta_grid=DELTAS,
        selection_metric="exact 0.1*sum(SMP*k*abs(group signed error sum)) on March complete-roster hours",
        objective="smooth weighted full-roster group absolute error plus L2 distance to fixed Ridge prior",
        training_data="complete-roster hours from causal 2024 and 2025 OOT forecasts only",
        fallback_selection=False, vpp_optimization=False,
        historical_caveat="2024/2025 already inspected; April-May is a new within-run diagnostic split, not globally untouched data")
    write_json(dest / "protocol.json", protocol)
    print("Loading 2024 complete-roster gradient objective", flush=True)
    k24, x24, y24, c24, prov24 = load_level1(out, dataset, "confirm_2024")
    g24 = aggregate_complete(k24, x24, y24, c24, dataset)
    del k24, x24, y24, c24
    print("Loading 2025 OOT forecasts", flush=True)
    k25, x25, y25, c25, prov25 = load_level1(out, dataset, "check_2025")
    g25 = aggregate_complete(k25, x25, y25, c25, dataset)
    group = concatenate_groups(g24, g25)
    write_json(dest / "coverage.json", dict(complete_2024_hours=g24["eligible_hours"],
        complete_2025_hours=g25["eligible_hours"], excluded_incomplete_2024_hours=g24["excluded_hours"],
        excluded_incomplete_2025_hours=g25["excluded_hours"], training_sources=prov24+prov25))
    trials = []
    started = time.time()
    for mode in MODES:
        for regularization in LAMBDAS:
            for delta in DELTAS:
                config = dict(mode=mode, regularization=regularization, delta=delta)
                model = fit_weights(group, TRAIN_END, config, prior)
                penalty = group_penalty(model, group, SELECT_START, SELECT_END)
                trial = dict(config=config, selection_penalty_krw=penalty,
                    training_penalty_krw=group_penalty(model, group, end=TRAIN_END),
                    converged=all(f["converged"] for f in model["fits"].values()),
                    iterations={n:f["iterations"] for n,f in model["fits"].items()})
                trials.append(trial)
                write_json(dest / "trials.json", trials)
                print(f"PGD trial {len(trials)}/16: {config}, March penalty={penalty:,.2f}, converged={trial['converged']}", flush=True)
    trials.sort(key=lambda row: (row["selection_penalty_krw"], row["config"]["mode"] != "global",
                                 -row["config"]["regularization"]))
    selected = trials[0]["config"]
    write_json(dest / "trials.json", trials)
    write_json(dest / "selected.json", dict(config=selected, source="2025 March exact group penalty only",
        diagnostic_April_May_used_for_selection=False, tuning_seconds=time.time()-started))
    selection_baselines = {}
    for name, vector in [(name,np.eye(len(BASE_MODELS))[j]) for j,name in enumerate(BASE_MODELS)]+[
            ("uniform_mean_6",np.ones(len(BASE_MODELS))/len(BASE_MODELS))]:
        baseline = dict(config=dict(mode="global"), fits=dict(all=dict(weights=vector.tolist())))
        selection_baselines[name] = group_penalty(baseline, group, SELECT_START, SELECT_END)
    selection_baselines["selected_PGD"] = trials[0]["selection_penalty_krw"]
    write_json(dest / "selection_baselines.json", dict(period="2025 March through March 31 17:00",
        penalty_krw=selection_baselines, comparisons_only_no_reselection=True))
    print(f"Selected PGD config: {selected}", flush=True)
    check_model = fit_weights(group, SELECT_END, selected, prior)
    check_model["provenance"] = prov24+prov25
    check_dir = dest / "check_2025_AprMay"
    write_json(check_dir / "model.json", check_model)
    use = (k25.ts >= pd.Timestamp(CHECK_START)).to_numpy()
    keys = k25.loc[use].reset_index(drop=True)
    x, y, cap = x25[use], y25[use], c25[use]
    prediction = predict(check_model, keys, x, cap)
    save_prediction(check_dir / "predictions.parquet", keys, prediction, SELECT_END,
                    "out_of_time_April_May_diagnostic; March selection frozen")
    gd_metrics = score(keys, prediction, y, dataset)
    # Independent scorer must reproduce the residual-matrix objective exactly.
    np.testing.assert_allclose(gd_metrics["reference_groups"]["penalty_krw"],
                               group_penalty(check_model, g25, start=CHECK_START), rtol=1e-10, atol=1e-5)
    write_json(check_dir / "metrics.json", gd_metrics)
    comparison = {}
    for j, name in enumerate(BASE_MODELS):
        comparison[name] = score(keys, x[:, j]*cap, y, dataset)
    comparison["uniform_mean_6"] = score(keys, x.mean(axis=1)*cap, y, dataset)
    ridge_forecast = pd.read_parquet(out / "Stacking" / "Ver1" / "check_2025" / "predictions.parquet")
    ridge_forecast = ridge_forecast.loc[ridge_forecast.ts >= pd.Timestamp(CHECK_START)].reset_index(drop=True)
    if not ridge_forecast[["site_id", "ts"]].equals(keys):
        raise ValueError("Ridge diagnostic keys do not match")
    comparison["Ridge_stacking"] = score(keys, ridge_forecast.prediction.to_numpy(), y, dataset)
    comparison["projected_GD"] = gd_metrics
    best_single = min(comparison[n]["reference_groups"]["penalty_krw"] for n in BASE_MODELS)
    improvements = {n:100*(1-gd_metrics["reference_groups"]["penalty_krw"]/
                          comparison[n]["reference_groups"]["penalty_krw"])
                    for n in ["uniform_mean_6", "Ridge_stacking"]}
    improvements["best_single"] = 100*(1-gd_metrics["reference_groups"]["penalty_krw"]/best_single)
    write_json(dest / "comparison.json", dict(period="2025-04-01 01:00 through 2025-06-01 00:00 KST",
        metrics=comparison, penalty_improvement_pct=improvements,
        beats_all_diagnostic_baselines=all(v > 0 for v in improvements.values()),
        diagnostic_used_to_reselect_settings=False))
    del x25, y25, c25, k25, ridge_forecast, x, y, cap
    final_model = fit_weights(group, FINAL_END, selected, prior)
    final_model["provenance"] = prov24+prov25
    write_json(dest / "final" / "model.json", final_model)
    ke, xe, _, ce, prove = load_level1(out, dataset, "final", labels=False)
    final_pred = predict(final_model, ke, xe, ce)
    save_prediction(dest / "final" / "predictions.parquet", ke, final_pred, FINAL_END,
                    "competition evaluation; no target labels available")
    # Simplex blending guarantees a nonnegative prediction within the base range.
    lower, upper = xe.min(axis=1)*ce, xe.max(axis=1)*ce
    if (final_pred < lower-1e-8).any() or (final_pred > upper+1e-8).any():
        raise AssertionError("Convex blend is outside the base prediction range")
    saved = json.loads((dest / "final" / "model.json").read_text())
    sample = np.arange(0, len(ke), 997)
    np.testing.assert_allclose(predict(saved, ke.iloc[sample], xe[sample], ce[sample]), final_pred[sample], atol=1e-8)
    public_prediction = pd.read_parquet(dest / "final" / "predictions.parquet")
    assert len(public_prediction) == 2_216_280 and not public_prediction.duplicated(["site_id", "ts"]).any()
    assert (public_prediction.groupby("site_id").size() == 8760).all()
    assert np.isfinite(public_prediction.prediction).all() and (public_prediction.prediction >= 0).all()
    verification = dict(status="PASS", final_rows=len(public_prediction),
        diagnostic_complete_hours=gd_metrics["reference_groups"]["complete_hours"],
        final_training_complete_hours=final_model["training_complete_hours"],
        final_training_end=FINAL_END, simplex_constraints_verified=True,
        independent_group_penalty_matches=True, reloaded_model_matches=True,
        raw_targets_uploaded=False, VPP_composition_changed=False,
        check_optimizer_converged=all(f["converged"] for f in check_model["fits"].values()),
        final_optimizer_converged=all(f["converged"] for f in final_model["fits"].values()),
        evaluation_provenance=prove)
    write_json(dest / "verification.json", verification)
    lines = ["# 경사하강법 가중치 최적화 Ver1", "",
        "6개 예측의 가중치를 0 이상·합계 1로 제한하고 projected gradient descent와 backtracking으로 학습했다. 별도 절편 없이 예측의 볼록 결합을 생성한다. VPP 배정은 기존 참조 그룹으로 고정했다.", "",
        "2024년 완전 실적 178시간과 2025년 1~2월(2월 28일 17시까지)로 설정별 가중치를 학습했다. 3월 1일 01시~3월 31일 17시의 정확한 그룹 벌금으로 설정을 선택했다. 이후 3월 31일 17시까지 다시 적합해 4~5월을 비교했다. 마지막으로 2025년 5월 31일 17시까지 가용한 완전 실적 시간으로 최종 가중치를 적합했다.", "",
        "2024~2025년은 앞선 연구에서 확인한 데이터다. 이번 4~5월 비교는 시간 경계를 지킨 진단 구간이며 연구 전체에서 처음 보는 테스트가 아니다. 진단 결과로 설정을 다시 선택하지 않았다. 최초 평가일 cutoff 이후 마지막 7시간 실적은 최종 학습에서 제외했다.", "",
        "손실은 SMP×k 가중 그룹 합산 오차의 미분 가능한 절대값 근사와 고정 Ridge 가중치에 대한 L2 정규화다. 정규화된 그룹 오차에서 평활화 폭·정규화 강도·전 시간 공통/피크별 가중치를 16개 조합으로 비교했다. SMP는 과거 손실 가중치와 채점에만 사용하며 예측 입력에는 넣지 않는다. 불완전 실적 시간은 통째로 제외하고 0 실적은 유지한다.", "",
        "| 모델 | 2025.4~5 MAE(kWh) | 참조 그룹 벌금(원) |", "|---|---:|---:|"]
    for name, row in comparison.items():
        lines.append(f"| {name} | {row['site_mae_kwh']:.4f} | {row['reference_groups']['penalty_krw']:,.0f} |")
    lines += ["", f"진단 벌금 변화: Ridge 대비 {-improvements['Ridge_stacking']:+.2f}%, 단순 평균 대비 {-improvements['uniform_mean_6']:+.2f}%, 해당 구간 최저 단일 모델 대비 {-improvements['best_single']:+.2f}%. 양수는 벌금 증가다. 개선 여부는 최적화 수렴과 별개로 판단한다.", "",
        f"선택 설정: `{selected}`. 최종 학습 완전 실적 시간: {final_model['training_complete_hours']:,}시간.", "",
        "`Ver1/final/model.json`은 최종 가중치, `Ver1/final/predictions.parquet`는 평가 기간 전체 2,216,280행 예측이다. `check_2025_AprMay/`는 4~5월 진단 전용 모델과 예측이다. `comparison.json`, `trials.json`, `protocol.json`, `coverage.json`, `verification.json`에서 비교·선택·시점·가용성과 검사를 확인할 수 있다.", "",
        "가중치는 `model.json`의 `fits`에 있고, 공통 모드라면 `all`, 피크별 모드라면 `peak`/`offpeak`를 사용한다. 피크는 KST ts.hour 11~15다. 이후 VPP 구성이 달라지면 이 가중치의 목적함수도 달라지므로 현재 결과를 새 배정의 최적해라고 해석할 수 없다.", "",
        "실행: `python -m GM_Ver1.gradient_descent --out <기존 로컬 결과 폴더>`"]
    lines += weights_markdown(final_model)
    (dest.parent / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if publish:
        shutil.copytree(dest.parent, publish / "GradientDescent", dirs_exist_ok=True, copy_function=copy_if_changed)
    print(json.dumps(dict(selected=selected, penalty_improvement_pct=improvements,
                         gradient_metrics=gd_metrics, verification=verification), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--publish", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run(args.out, args.dataset or current_dataset(), args.publish)
