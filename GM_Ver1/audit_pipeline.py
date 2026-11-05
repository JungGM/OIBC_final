"""Compare saved OIBC predictions locally on common complete-fleet hours."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from GM_Ver1.common import local_output, sha256, write_json
from GM_Ver1.genetic_algorithm import genetic_search
from train.dataset import current_dataset

START = pd.Timestamp('2025-03-01 01:00', tz='Asia/Seoul')
END = pd.Timestamp('2025-05-31 17:00', tz='Asia/Seoul')


def align(frame):
    frame['ts'] = pd.to_datetime(frame.ts, utc=True).dt.tz_convert('Asia/Seoul').dt.as_unit('ns')
    result = frame.set_index(['ts', 'site_id'])
    if not result.index.is_unique:
        raise ValueError('Duplicate forecast keys')
    return result


def loss(aggregate, smp, k):
    return .1*(smp*k)[:, None]*np.abs(aggregate)


def aggregate(residual, assignment):
    return np.column_stack((residual[:, assignment].sum(axis=1), residual[:, ~assignment].sum(axis=1)))


def block_interval(day_values):
    rng = np.random.default_rng(20261105)
    blocks = rng.integers(0, len(day_values)-2, size=(4000, (len(day_values)+2)//3))
    samples = day_values[(blocks[:, :, None]+np.arange(3)).reshape(4000, -1)[:, :len(day_values)]]
    return (np.quantile(samples.mean(axis=1), [.05, .95])*len(day_values)).tolist()


def run(dataset, old, current, destination):
    destination = local_output(destination)
    data_file = dataset/'training_data.parquet'
    data_sha = sha256(data_file)
    verified = json.loads((current/'verification.json').read_text(encoding='utf-8'))
    if data_sha != verified['training_grid_sha256']:
        raise ValueError('Audit labels must match the frozen dataset used by the verified base pipeline')
    roster = pd.read_parquet(dataset/'reference_composition.parquet').sort_values('site_id')
    sites = roster.site_id.tolist()
    reference = roster.current_group.eq('A').to_numpy()
    ga = pd.read_csv(current/'GeneticAlgorithm/Ver1/diagnostic_composition.csv').set_index('site_id').reindex(sites)
    assert ga.group.isin(['G1','G2']).all()
    ga = ga.group.eq('G1').to_numpy()
    old_submission = old/'제출'
    official = json.loads((old_submission/'공식제출결과_20261102.json').read_text(encoding='utf-8'))
    for name, meta in official['files'].items():
        assert sha256(old_submission/name) == meta['sha256']
    old_group = pd.read_csv(old_submission/'composition.csv').set_index('site_id').reindex(sites).group.eq('G1').to_numpy()
    assert np.array_equal(reference, old_group) or np.array_equal(reference, ~old_group)

    raw = pq.read_table(data_file, columns=['site_id','ts','energy_kwh'],
                        filters=[('ts','>=',START), ('ts','<=',END)]).to_pandas()
    raw = align(raw).sort_index()
    clock = raw.index.get_level_values('ts')
    date = clock-pd.Timedelta(hours=1)
    last = date.is_month_end & np.isin(clock.hour, [0,18,19,20,21,22,23])
    raw = raw.loc[~last]
    counts = raw.energy_kwh.notna().groupby(level='ts').sum()
    times = pd.DatetimeIndex(counts.index[counts == len(sites)])
    keys = pd.MultiIndex.from_product([times,sites], names=['ts','site_id'])
    y = raw.energy_kwh.reindex(keys).to_numpy().reshape(-1,len(sites))
    weights = pd.read_parquet(dataset/'training_weights.parquet').set_index('ts').reindex(times)
    smp, k = weights.smp.to_numpy(), weights.k.to_numpy()
    assert np.isfinite(y).all() and np.isfinite(smp).all()
    assert np.array_equal(k, np.where(times.hour.isin(range(11,16)),3,1))
    month = (times-pd.Timedelta(hours=1)).month.to_numpy()
    assert [(month==m).sum() for m in [3,4,5]] == [737,713,737]

    def read(path, value, selected_keys=keys):
        data = align(pd.read_parquet(path,columns=['site_id','ts',value]))
        result = data[value].reindex(selected_keys).to_numpy().reshape(-1,len(sites))
        assert np.isfinite(result).all() and (result>=0).all(), path
        return result

    legacy = old/'_local'
    baseline = read(legacy/'paper_v2_seasonal/development/site_predictions.parquet','ensemble')
    v7 = .5*baseline+.5*read(legacy/'improvement_v7_team_pair/development/predictions.parquet','new')
    v8 = .75*baseline+.25*read(legacy/'improvement_v8_components/development/predictions.parquet','new')
    names = [f'{family}/Ver{i}' for family in ('XGBoost','GRU','TCN') for i in (1,2)]
    base = {n:read(current/n/'check_2025/predictions.parquet','prediction') for n in names}
    uniform = np.mean(list(base.values()), axis=0)
    ridge = read(current/'Stacking/Ver1/check_2025/predictions.parquet','prediction')
    held = month>=4
    gd = read(current/'GradientDescent/Ver1/check_2025_AprMay/predictions.parquet','prediction',keys[np.repeat(held,len(sites))])
    legacy_variants = {}
    for label, directory in [('V11','solar_normalized_v11'),('V14','weather_augmented_v14'),('V18','spatial_v18')]:
        source = legacy/directory
        report = json.loads((source/'evaluation.json').read_text(encoding='utf-8'))
        fraction = report['selected']['strength']
        pred = (1-fraction)*v8+fraction*read(source/'development/predictions.parquet','new')
        for m, expected in zip([3,4,5],report['months']):
            mask = month==m
            actual = loss(aggregate(pred[mask]-y[mask],reference),smp[mask],k[mask]).sum()
            np.testing.assert_allclose(actual,expected['candidate_penalty'],rtol=1e-10)
        legacy_variants[label] = pred
    forecasts = {'official_Nov02':baseline,'official_V8':v8,'legacy_V7':v7,**legacy_variants,**base,'uniform':uniform,'ridge':ridge}

    # Fix one small challenger family, select in March, then record both later months.
    screen = []
    candidates = {'V8 unchanged':v8}
    for name, values in {**base,'uniform':uniform}.items():
        for fraction in (.10,.25,.50):
            candidates[f'V8 + {fraction:.0%} {name}'] = (1-fraction)*v8+fraction*values
    for name, pred in candidates.items():
        error = aggregate(pred[month==3]-y[month==3], reference)
        screen.append(dict(candidate=name,march_penalty_krw=float(loss(error,smp[month==3],k[month==3]).sum())))
    screen.sort(key=lambda row:row['march_penalty_krw'])
    selected = screen[0]['candidate']
    forecasts['March-selected challenger'] = candidates[selected]
    results, held_losses = {}, {}
    for name, forecast in {**forecasts,'current_GD':gd}.items():
        covered = held if name=='current_GD' else np.ones(len(month), dtype=bool)
        resid = forecast-y[covered]
        groups = aggregate(resid,reference)
        ga_groups = aggregate(resid,ga)
        ref_loss = loss(groups,smp[covered],k[covered]).sum(axis=1)
        ga_loss = loss(ga_groups,smp[covered],k[covered]).sum(axis=1)
        floor = loss(resid.sum(axis=1)[:,None],smp[covered],k[covered]).ravel()
        scores = {}
        for m in sorted(set(month[covered])):
            mask = month[covered]==m
            scores[str(m)] = dict(hours=int(mask.sum()), reference_penalty=float(ref_loss[mask].sum()),
                fleet_floor=float(floor[mask].sum()), site_mae=float(np.abs(resid[mask]).mean()),
                peak_penalty=float(ref_loss[mask & (k[covered]==3)].sum()))
            if m>=4:
                scores[str(m)]['GA_penalty'] = float(ga_loss[mask].sum())
        hm = month[covered]>=4
        daylight = hm & (k[covered]==3)
        night = hm & times[covered].hour.isin(range(1,5))
        zero_night_loss = loss(aggregate(-y[covered][night], reference),smp[covered][night],k[covered][night]).sum()
        results[name] = dict(months=scores, apr_may_reference=float(ref_loss[hm].sum()),
            apr_may_GA=float(ga_loss[hm].sum()), apr_may_fleet_floor=float(floor[hm].sum()),
            apr_may_peak_error_corr=float(np.corrcoef(groups[daylight].T)[0,1]),
            apr_may_peak_bias_kwh=float(groups[daylight].sum(axis=1).mean()),
            apr_may_peak_penalty_share=float(ref_loss[daylight].sum()/ref_loss[hm].sum()),
            deep_night_penalty=float(ref_loss[night].sum()),
            deep_night_zero_diagnostic_gain=float(ref_loss[night].sum()-zero_night_loss))
        held_losses[name] = (ref_loss[hm],ga_loss[hm])
    historical = pd.read_csv(legacy/'improvement_v8_components/monthly_comparison.csv')
    np.testing.assert_allclose([results['official_V8']['months'][str(m)]['reference_penalty'] for m in [3,4,5]],
                               historical.candidate_penalty_krw, rtol=1e-10)
    op_day = (times[held]-pd.Timedelta(hours=1)).normalize()
    daily = pd.DataFrame({'day':op_day,'gain':held_losses['current_GD'][0]-held_losses['current_GD'][1]}).groupby('day').gain.agg(['size','sum'])
    interval = block_interval(daily.loc[daily['size']==24,'sum'].to_numpy())

    # Post-hoc diagnostic of the strongest observed spring forecast, not a new selection protocol.
    gru = base['GRU/Ver1']-y
    march = month==3
    search = genetic_search(gru[march],(smp*k)[march],roster.capacity_kw.to_numpy(),reference,20261105)
    gru_ga = loss(aggregate(gru[held],search['bits']),smp[held],k[held]).sum()
    final_map = pd.read_csv(current/'GeneticAlgorithm/Ver1/final/composition.csv').set_index('site_id').reindex(sites)
    assert final_map.group.isin(['G1','G2']).all()
    changes = int(np.sum(final_map.group.eq('G1').to_numpy()!=ga))
    reassignment = dict(retrospective_GRU1_GA_fit='March complete hours, fixed seed; GRU1 chosen after observing spring',
        apr_may_reference=results['GRU/Ver1']['apr_may_reference'],
        apr_may_GD_GA_map=results['GRU/Ver1']['apr_may_GA'],apr_may_GRU1_GA_map=float(gru_ga),
        refit_assignment_changes_up_to_label_swap=min(changes,len(sites)-changes))

    price = pd.read_parquet(dataset/'evaluation_weights.parquet').set_index('ts').smp
    annual = {}
    for name, path in [('official_Nov02',old_submission/'forecast.csv'),
                       ('official_V8',old_submission/'improved_v8/forecast.csv'),
                       ('current_GD_GA',current/'GeneticAlgorithm/Ver1/final/forecast.csv')]:
        p = pd.read_csv(path)
        p['ts'] = pd.to_datetime(p.ts,utc=True).dt.tz_convert('Asia/Seoul')
        joined = p.join(price,on='ts',validate='many_to_one')
        assert len(joined)==17520 and not joined.smp.isna().any() and not joined.duplicated(['group','ts']).any()
        annual[name] = dict(predicted_energy_kwh=float(joined.forecast_kwh.sum()),
                            forecast_implied_gross_krw=float((joined.forecast_kwh*joined.smp).sum()))
    out = dict(training_grid_sha256=data_sha, common_hours_by_month={'March':737,'April':713,'May':737},results=results,
        GA_daily_block_bootstrap_90pct_gain_krw=interval, complete_days_used_for_interval=int((daily['size']==24).sum()),
        March_only_screen=screen, March_selected_challenger=selected, reassignment_diagnostic=reassignment,
        annual_forecast_totals_without_actuals=annual,
        comparison_caveat='Historical diagnostics. Legacy predictors fit through Feb 28 and current base predictors through Dec 31. No claim of architecture-only effect or globally untouched test.')
    write_json(destination/'audit.json',out)
    print(json.dumps({k:out[k] for k in ['March_selected_challenger','annual_forecast_totals_without_actuals',
                    'GA_daily_block_bootstrap_90pct_gain_krw','reassignment_diagnostic']},indent=2),flush=True)
    for name, row in results.items():
        print(name, *(round(row[k]) for k in ['apr_may_reference','apr_may_GA','apr_may_fleet_floor']),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--current', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    run(args.dataset or current_dataset(),args.old,args.current,args.out)
