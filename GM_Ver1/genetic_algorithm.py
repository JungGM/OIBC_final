"""Capacity-constrained GA portfolios and explicitly conditional settlement scenarios."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from train.dataset import current_dataset, score_predictions
from GM_Ver1.common import local_output, sha256, write_json
from GM_Ver1.stacking import load_level1
from GM_Ver1.gradient_descent import (aggregate_complete, concatenate_groups, fit_weights,
    predict as gd_predict, project_simplex, TRAIN_END, SELECT_START, SELECT_END, CHECK_START)
from GM_Ver1.run import FINAL_END
from GM_Ver1.sync_outputs import copy_if_changed

MIN_CAPACITY = 20_000.
SEEDS = [20261105, 20261106, 20261107]
GA_CONFIG = dict(population=128, generations=120, elite=8, immigrants=8,
                 tournament=3, crossover=.8, mutation_expected_bits=1.5)


def canonical(bits):
    bits = np.asarray(bits, dtype=bool).copy()
    return bits if bits[0] else ~bits


def repair(bits, capacity, minimum, rng):
    a = np.asarray(bits, dtype=bool).copy()
    upper = capacity.sum()-minimum
    if minimum > upper:
        raise ValueError("Fleet capacity cannot support two minimum-size groups")
    for _ in range(2*len(a)+1):
        mass = float(capacity@a)
        if minimum-1e-9 <= mass <= upper+1e-9:
            return canonical(a)
        if mass < minimum:
            choices = np.flatnonzero(~a & (capacity <= upper-mass+1e-9))
        else:
            choices = np.flatnonzero(a & (capacity <= mass-minimum+1e-9))
        if not len(choices):
            raise ValueError("Cannot repair this chromosome to a feasible partition")
        j = rng.choice(choices)
        a[j] = ~a[j]
    raise RuntimeError("Capacity repair did not finish")


def fitness(population, errors, q, capacity, minimum=MIN_CAPACITY):
    population = np.asarray(population, dtype=bool)
    masses = population@capacity
    feasible = (masses >= minimum-1e-8) & (masses <= capacity.sum()-minimum+1e-8)
    result = np.full(len(population), np.inf)
    if feasible.any():
        first = errors@population[feasible].T.astype(np.float64)
        total = errors.sum(axis=1)[:, None]
        result[feasible] = .1*np.sum(q[:, None]*(np.abs(first)+np.abs(total-first)), axis=0)
    return result


def genetic_search(errors, q, capacity, reference, seed, config=GA_CONFIG, minimum=MIN_CAPACITY, warm=None):
    rng = np.random.default_rng(seed)
    n, size = len(capacity), config["population"]
    reference = repair(reference, capacity, minimum, rng)
    population, seen = [], set()

    def add(bits):
        a = repair(bits, capacity, minimum, rng)
        key = np.packbits(a).tobytes()
        if key not in seen:
            seen.add(key)
            population.append(a)

    add(reference)
    for a in warm or []:
        add(a)
    attempts = 0
    while len(population) < size:
        attempts += 1
        a = reference ^ (rng.random(n) < min(.2, 3/n)) if attempts % 3 == 0 else rng.random(n) < .5
        add(a)
        if attempts > size*100:
            raise RuntimeError("Unable to build a diverse feasible GA population")
    pop = np.array(population)
    history, best_value, stale = [], np.inf, 0
    started = time.time()
    for generation in range(config["generations"]+1):
        values = fitness(pop, errors, q, capacity, minimum)
        order = np.argsort(values, kind="stable")
        pop, values = pop[order], values[order]
        if values[0] < best_value-1e-5:
            best_value, stale = float(values[0]), 0
        else:
            stale += 1
        if generation % 10 == 0 or generation == config["generations"]:
            row = dict(generation=generation, best_penalty_krw=float(values[0]),
                       median_penalty_krw=float(np.median(values)), unique_chromosomes=len({np.packbits(a).tobytes() for a in pop}))
            history.append(row)
            print(f"GA seed={seed} generation={generation}: penalty={values[0]:,.2f}", flush=True)
        if generation == config["generations"]:
            break
        population = [a.copy() for a in pop[:config["elite"]]]
        seen = {np.packbits(a).tobytes() for a in population}
        mutation = config["mutation_expected_bits"]*(2 if stale >= 15 else 1)/n
        attempts = 0
        while len(population) < size-config["immigrants"]:
            attempts += 1
            parents = [pop[ids[np.argmin(values[ids])]] for ids in [rng.integers(0, size, config["tournament"]),
                                                               rng.integers(0, size, config["tournament"])]]
            a = np.where(rng.random(n) < .5, parents[0], parents[1]) if rng.random() < config["crossover"] else parents[0].copy()
            a ^= rng.random(n) < mutation
            add(a)
            if attempts > size*100:
                break
        while len(population) < size:
            add(rng.random(n) < .5)
        pop = np.array(population[:size])
    bound = float(.1*np.sum(q*np.abs(errors.sum(axis=1))))
    return dict(bits=pop[0], training_penalty_krw=float(values[0]), history=history,
        seconds=time.time()-started, theoretical_fleet_error_lower_bound_krw=bound,
        optimality_not_proven=True, seed=seed, config=config)


def site_panel(keys, prediction, actual, roster, dataset):
    counts = keys.assign(observed=np.isfinite(actual)).groupby("ts")["observed"].sum()
    times = counts.index[counts == len(roster)]
    mask = keys.ts.isin(times).to_numpy()
    frame = keys.loc[mask].copy()
    frame["prediction"], frame["actual"] = prediction[mask], actual[mask]
    if frame.duplicated(["site_id", "ts"]).any():
        raise ValueError("Duplicate site-hour in GA objective")
    sites = roster.site_id.tolist()
    p = frame.pivot(index="ts", columns="site_id", values="prediction").reindex(index=times, columns=sites).to_numpy()
    y = frame.pivot(index="ts", columns="site_id", values="actual").reindex(index=times, columns=sites).to_numpy()
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Incomplete GA objective panel")
    weights = pd.read_parquet(dataset / "training_weights.parquet").set_index("ts").reindex(times)
    if not np.isfinite(weights[["smp", "k"]]).all().all() or (weights.smp < 0).any():
        raise ValueError("Invalid GA settlement weights")
    return dict(times=times, prediction=p, actual=y, errors=p-y,
                q=(weights.smp*weights.k).to_numpy(), smp=weights.smp.to_numpy(), k=weights.k.to_numpy())


def join_panels(a, b):
    times = a["times"].append(b["times"])
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("GA time windows overlap")
    return {"times":times, **{key:np.concatenate([a[key], b[key]]) for key in a if key != "times"}}


def subset(panel, start=None, end=None):
    mask = np.ones(len(panel["times"]), dtype=bool)
    if start:
        mask &= panel["times"] >= pd.Timestamp(start)
    if end:
        mask &= panel["times"] <= pd.Timestamp(end)
    return {key:value[mask] for key,value in panel.items()}


def composition(bits, roster):
    return pd.DataFrame(dict(site_id=roster.site_id, group=np.where(bits, "G1", "G2")))


def panel_score(panel, bits):
    a = bits.astype(np.float64)
    ep = panel["errors"]@a
    remainder = panel["errors"].sum(axis=1)-ep
    penalty = float(.1*np.dot(panel["q"], np.abs(ep)+np.abs(remainder)))
    gross = float(np.dot(panel["smp"], panel["actual"].sum(axis=1)))
    return dict(complete_hours=len(panel["times"]), revenue_krw=gross, penalty_krw=penalty,
                settlement_krw=gross-penalty)


def serial_search(result):
    return {key:value for key,value in result.items() if key != "bits"}


def group_matrix(values, bits):
    return np.column_stack([values[:, bits].sum(axis=1), values[:, ~bits].sum(axis=1)])


def settlement_scenarios(check, check_bits, future, future_bits, capacity, smp, q, draws=1000):
    # The last calibration operating day must close by the first evaluation cutoff.
    eligible = subset(check, end=FINAL_END)
    first = eligible["times"][0]
    hours = len(eligible["times"])//24*24
    if first.hour != 1 or hours < 24 or not eligible["times"].equals(pd.date_range(first, periods=len(eligible["times"]), freq="h")):
        raise ValueError("Settlement calibration requires contiguous operating days")
    old_capacity = np.array([capacity[check_bits].sum(), capacity[~check_bits].sum()])
    new_capacity = np.array([capacity[future_bits].sum(), capacity[~future_bits].sum()])
    errors = group_matrix(eligible["errors"][:hours], check_bits).reshape(-1, 24, 2)/old_capacity
    forecasts = group_matrix(future, future_bits)
    days = len(forecasts)//24
    sampled = np.random.default_rng(SEEDS[0]).integers(len(errors), size=(draws, days))
    scenarios = {}
    for scale in [.5, 1., 1.5, 2.]:
        gross, penalty = [], []
        for batch in np.array_split(sampled, max(1, (draws+31)//32)):
            simulated_error = errors[batch].reshape(-1, len(forecasts), 2)*new_capacity*scale
            simulated_actual = np.maximum(forecasts[None, :, :]-simulated_error, 0)
            gross.extend(np.sum(simulated_actual*smp[None, :, None], axis=(1, 2)))
            penalty.extend(.1*np.sum(np.abs(forecasts-simulated_actual)*q[None, :, None], axis=(1, 2)))
        net = np.array(gross)-np.array(penalty)
        scenarios[str(scale)] = dict(mean_revenue_krw=float(np.mean(gross)), mean_penalty_krw=float(np.mean(penalty)),
            mean_settlement_krw=float(net.mean()), p10_krw=float(np.quantile(net, .1)), p90_krw=float(np.quantile(net, .9)))
    return dict(forecast_implied_gross_krw=float(np.dot(smp, forecasts.sum(axis=1))), draws=draws,
        calibration_operating_days=len(errors), calibration_start=str(first), calibration_end=str(eligible["times"][hours-1]),
        method="paired G1/G2 24h residual block bootstrap, capacity rescaling, actual floored at zero",
        scenario_assumption="April-May diagnostic residual days transfer to the full evaluation year and the refitted forecast and allocation. Provided evaluation SMP is fixed. Quantiles are conditional simulation percentiles.",
        scenarios=scenarios)


def check_submission(directory, roster, times, expected):
    assignments = pd.read_csv(directory/"composition.csv")
    forecasts = pd.read_csv(directory/"forecast.csv", float_precision="round_trip")
    assert list(assignments) == ["site_id", "group"] and list(forecasts) == ["group", "ts", "forecast_kwh"]
    assert assignments.site_id.is_unique and set(assignments.site_id) == set(roster.site_id)
    capacities = assignments.merge(roster, on="site_id", validate="one_to_one").groupby("group").capacity_kw.sum()
    assert set(capacities.index) == {"G1", "G2"} and (capacities >= MIN_CAPACITY-1e-8).all()
    assert len(forecasts) == 17520 and not forecasts.duplicated(["group", "ts"]).any()
    assert np.isfinite(forecasts.forecast_kwh).all() and (forecasts.forecast_kwh >= 0).all()
    for j, group in enumerate(["G1", "G2"]):
        rows = forecasts.loc[forecasts.group == group]
        assert pd.DatetimeIndex(pd.to_datetime(rows.ts, utc=True)).tz_convert(times.tz).equals(times)
        np.testing.assert_allclose(rows.forecast_kwh, expected[:, j], rtol=1e-12, atol=1e-8)
    return {g:float(capacities[g]) for g in capacities.index}


def run(out, dataset, publish):
    out = local_output(out)
    dest = out/"GeneticAlgorithm"/"Ver1"
    dest.mkdir(parents=True, exist_ok=True)
    gd = out/"GradientDescent"/"Ver1"
    base_verification = json.loads((out/"verification.json").read_text())
    assert base_verification["status"] == "PASS" and json.loads((gd/"verification.json").read_text())["status"] == "PASS"
    dataset_hash = sha256(dataset/"training_data.parquet")
    assert dataset_hash == base_verification["training_grid_sha256"]
    roster = pd.read_parquet(dataset/"reference_composition.parquet").sort_values("site_id").reset_index(drop=True)
    assert len(roster) == 253 and roster.site_id.is_unique
    capacity = roster.capacity_kw.to_numpy()
    reference = canonical((roster.current_group == "A").to_numpy())
    check_gd = json.loads((gd/"check_2025_AprMay"/"model.json").read_text())
    final_gd = json.loads((gd/"final"/"model.json").read_text())
    assert check_gd["training_end"] == SELECT_END and final_gd["training_end"] == FINAL_END
    assert check_gd["config"] == final_gd["config"]
    write_json(dest/"protocol.json", dict(seed_candidates=SEEDS, config=GA_CONFIG, minimum_group_kw=MIN_CAPACITY,
        training_grid_sha256=dataset_hash, weights_sha256=sha256(dataset/"training_weights.parquet"),
        evaluation_prices_sha256=sha256(dataset/"evaluation_weights.parquet"),
        composition_sha256=sha256(dataset/"reference_composition.parquet"),
        gd_check_model_sha256=sha256(gd/"check_2025_AprMay"/"model.json"), gd_final_model_sha256=sha256(gd/"final"/"model.json"),
        inner_fit_end=TRAIN_END, selection_start=SELECT_START, selection_end=SELECT_END, check_start=CHECK_START,
        final_fit_end=FINAL_END, selection="best GA seed on March penalty; GD and GA settings selected within March",
        reference_in_initial_population=True, held_out_diagnostic_used_for_reselection=False,
        historical_coverage="2024/2025 already inspected in earlier research"))
    print("Loading verified out-of-time base forecasts and historical actuals", flush=True)
    k24,x24,y24,c24,_ = load_level1(out,dataset,"confirm_2024")
    k25,x25,y25,c25,_ = load_level1(out,dataset,"check_2025")
    group = concatenate_groups(aggregate_complete(k24,x24,y24,c24,dataset), aggregate_complete(k25,x25,y25,c25,dataset))
    inner_gd = fit_weights(group, TRAIN_END, check_gd["config"], project_simplex(check_gd["prior"]))

    def panel_for(model):
        return join_panels(site_panel(k24,gd_predict(model,k24,x24,c24),y24,roster,dataset),
                           site_panel(k25,gd_predict(model,k25,x25,c25),y25,roster,dataset))

    inner = panel_for(inner_gd)
    train, selection = subset(inner,end=TRAIN_END), subset(inner,start=SELECT_START,end=SELECT_END)
    trials = []
    for seed in SEEDS:
        result = genetic_search(train["errors"], train["q"], capacity, reference, seed)
        result["selection_penalty_krw"] = panel_score(selection,result["bits"])["penalty_krw"]
        trials.append(result)
    selected = min(trials, key=lambda r:(r["selection_penalty_krw"], r["seed"]))
    write_json(dest/"selection.json", dict(selected_seed=selected["seed"], selection_hours=len(selection["times"]),
        training_hours=len(train["times"]), trials=[serial_search(r) for r in trials], reference=panel_score(selection,reference)))
    check_panel = panel_for(check_gd)
    check_train, check = subset(check_panel,end=SELECT_END), subset(check_panel,start=CHECK_START)
    check_search = genetic_search(check_train["errors"],check_train["q"],capacity,reference,selected["seed"],warm=[selected["bits"]])
    check_bits = check_search["bits"]
    baseline, diagnostic = panel_score(check,reference), panel_score(check,check_bits)
    # Verify the aggregated objective against the repository's independent site scorer.
    rows = pd.DataFrame(dict(site_id=np.tile(roster.site_id,len(check["times"])), ts=check["times"].repeat(len(roster)),
        actual=check["actual"].reshape(-1), prediction=check["prediction"].reshape(-1)))
    independent, _ = score_predictions(rows, composition(check_bits,roster).rename(columns={"group":"current_group"}),
        pd.read_parquet(dataset/"training_weights.parquet"))
    for key in ["revenue_krw","penalty_krw","settlement_krw"]:
        np.testing.assert_allclose(independent[key], diagnostic[key], rtol=1e-12, atol=1e-5)
    write_json(dest/"diagnostic.json", dict(period="2025 April-May KST", ga=diagnostic, reference=baseline,
        settlement_gain_krw=diagnostic["settlement_krw"]-baseline["settlement_krw"],
        penalty_reduction_pct=100*(1-diagnostic["penalty_krw"]/baseline["penalty_krw"]),
        fixed_composition_known_by=SELECT_END, fit_complete_hours=len(check_train["times"]), search=serial_search(check_search),
        theoretical_fleet_error_lower_bound_krw=float(.1*np.dot(check["q"],np.abs(check["errors"].sum(axis=1))))))
    composition(check_bits,roster).to_csv(dest/"diagnostic_composition.csv",index=False)
    final_train = subset(panel_for(final_gd),end=FINAL_END)
    final_search = genetic_search(final_train["errors"],final_train["q"],capacity,reference,selected["seed"],warm=[check_bits])
    final_bits = final_search["bits"]
    final_dir = dest/"final"
    final_dir.mkdir(exist_ok=True)
    composition(final_bits,roster).to_csv(final_dir/"composition.csv",index=False)
    prediction_path = gd/"final"/"predictions.parquet"
    metadata = json.loads(prediction_path.with_suffix(".json").read_text())
    assert metadata["training_end"] == FINAL_END and metadata["sha256"] == sha256(prediction_path)
    prediction = pd.read_parquet(prediction_path)
    weights = pd.read_parquet(dataset/"evaluation_weights.parquet").sort_values("ts")
    times = pd.DatetimeIndex(weights.ts)
    assert len(times) == 8760 and times.is_unique and times[0].hour == 1
    assert times.equals(pd.date_range(times[0],periods=8760,freq="h"))
    assert np.isfinite(weights[["smp","k"]]).all().all() and (weights.smp >= 0).all()
    assert np.array_equal(weights.k,np.where(times.hour.isin(range(11,16)),3,1))
    assert len(prediction) == len(times)*len(roster)
    assert set(prediction.site_id) == set(roster.site_id) and set(prediction.ts) == set(times)
    future = prediction.pivot(index="ts",columns="site_id",values="prediction").reindex(index=times,columns=roster.site_id).to_numpy()
    assert np.isfinite(future).all() and (future >= 0).all()
    grouped = group_matrix(future,final_bits)
    pd.DataFrame(dict(group=np.repeat(["G1","G2"],len(times)),ts=np.tile(times.astype(str),2),
                      forecast_kwh=grouped.T.reshape(-1))).to_csv(final_dir/"forecast.csv",index=False)
    capacities = check_submission(final_dir,roster,times,grouped)
    write_json(final_dir/"model.json",dict(training_end=FINAL_END,training_complete_hours=len(final_train["times"]),
        group_capacity_kw=capacities, group_sites={str(k):int(v) for k,v in composition(final_bits,roster).group.value_counts().items()},
        unchanged_reference_penalty_krw=panel_score(final_train,reference)["penalty_krw"],
        gd_evaluation_prediction_sha256=metadata["sha256"], search=serial_search(final_search)))
    scenario = settlement_scenarios(check,check_bits,future,final_bits,capacity,weights.smp.to_numpy(),(weights.smp*weights.k).to_numpy())
    write_json(dest/"expected_settlement.json",scenario)
    write_json(dest/"verification.json",dict(status="PASS",independent_scorer_matches=True,sites=253,
        group_capacity_kw=capacities,evaluation_hours=len(times),forecast_rows=2*len(times),
        composition_sha256=sha256(final_dir/"composition.csv"), forecast_sha256=sha256(final_dir/"forecast.csv")))
    base = scenario["scenarios"]["1.0"]
    lines = ["# GA VPP 배정과 정산", "",
        "기존 경사하강법의 발전소별 예측으로 G1·G2 배정을 학습했다. GA는 두 그룹에 각각 20 MW 이상을 배정하고 SMP×k로 가중한 그룹 합산 오차의 벌금을 최소화한다. 전력량 매출은 전체 발전량과 SMP로 결정된다.", "",
        f"최종 배정은 G1 {capacities['G1']:,.3f} kW, G2 {capacities['G2']:,.3f} kW다. `Ver1/final/composition.csv`와 `Ver1/final/forecast.csv`는 각각 253행과 17,520행이다.", "",
        "2024년 완전 실적과 2025년 1~2월을 사용해 세 시드의 GA를 학습하고 3월 31일 17시까지의 벌금으로 시드를 골랐다. 경사하강법과 GA 설정을 고르는 데 3월 데이터를 사용했다. 3월 말까지 다시 적합한 배정을 4~5월 1,464시간에 적용했다. 최종 배정은 5월 31일 17시까지의 완전 실적 전체로 적합했다. 2025년은 앞선 연구에서도 확인한 기간이다.", "",
        "| 2025년 4~5월 실적 기준 | 벌금(원) | 정산금액(원) |", "|---|---:|---:|",
        f"| 기존 배정 | {baseline['penalty_krw']:,.0f} | {baseline['settlement_krw']:,.0f} |",
        f"| GA 배정 | {diagnostic['penalty_krw']:,.0f} | {diagnostic['settlement_krw']:,.0f} |", "",
        f"평가 기간은 {times[0]}부터 {times[-1]}까지다. 제공 SMP와 예측 발전량의 곱을 합한 매출 기준값은 {scenario['forecast_implied_gross_krw']:,.0f}원이다.", "",
        f"평가 기간의 기본 시나리오 평균 정산금액은 **{base['mean_settlement_krw']:,.0f}원**이다. 평균 매출 {base['mean_revenue_krw']:,.0f}원에서 평균 벌금 {base['mean_penalty_krw']:,.0f}원을 뺐다.", "",
        f"시나리오는 첫 평가일 마감 이전 4~5월 진단의 완전한 {scenario['calibration_operating_days']}일에서 24시간 오차를 뽑아 1,000번 계산했다. 두 그룹의 오차를 같은 날에서 뽑고 최종 그룹 용량에 맞춰 조정했다. 이 계산은 봄철 오차가 다른 계절과 재학습한 모델·배정에도 적용된다고 가정한다. 평가 기간의 실제 정산은 실적이 공개된 뒤 계산할 수 있다.", "",
        "| 오차 배율 | 평균 벌금(원) | 평균 정산금액(원) | 시나리오 P10~P90(원) |", "|---:|---:|---:|---:|"]
    for scale, item in scenario["scenarios"].items():
        lines.append(f"| {scale} | {item['mean_penalty_krw']:,.0f} | {item['mean_settlement_krw']:,.0f} | {item['p10_krw']:,.0f}~{item['p90_krw']:,.0f} |")
    lines += ["", "실행 명령은 `python -m GM_Ver1.genetic_algorithm --out <기존 로컬 결과 폴더>`다."]
    (dest.parent/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if publish:
        shutil.copytree(dest.parent,publish/"GeneticAlgorithm",dirs_exist_ok=True,copy_function=copy_if_changed)
    print(json.dumps(dict(diagnostic=diagnostic, reference=baseline, capacities=capacities,
        expected_settlement=base, forecast_implied_gross_krw=scenario["forecast_implied_gross_krw"]),ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--publish", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run(args.out,args.dataset or current_dataset(),args.publish)
