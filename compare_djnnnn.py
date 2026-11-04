"""Compare frozen local XGBoost with a locally refitted DJN222 LightGBM.

Reads competition inputs only; writes diagnostics outside OneDrive. No uploads.
Run improved.py from the audited external checkout first, into OUT/reproduced.
The 50:50 blend is a fixed diagnostic, never a holdout-tuned deployment choice.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str,
                               allow_nan=False), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    for key in ['data', 'ours', 'repo', 'out']:
        ap.add_argument('--' + key, type=Path, required=True)
    a = ap.parse_args()
    if 'onedrive' in str(a.out.resolve()).lower():
        raise ValueError('Keep results outside OneDrive')
    a.out.mkdir(parents=True, exist_ok=True)
    audit = json.loads((a.ours / 'audit.json').read_text(encoding='utf-8'))
    for name, expected in audit['source_sha256'].items():
        assert digest(a.data / name) == expected, name
    sys.path.insert(0, str(a.repo))
    spec = importlib.util.spec_from_file_location('comparison_external', a.repo / 'improved.py')
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    b = ext.b
    # Local and linked baseline differ only in newline serialization.
    assert (a.data / 'baseline.py').read_text(encoding='utf-8') == (a.repo / 'baseline.py').read_text(encoding='utf-8')
    sites = pd.read_csv(a.data / 'sites.csv')
    groups = sites.set_index('site_id').current_group.map({'A': 'G1', 'B': 'G2'})
    selected = pd.read_csv(a.ours / 'composition_selected.csv').set_index('site_id').group
    pd.testing.assert_series_equal(groups.sort_index(), selected.sort_index(), check_names=False)
    external_groups = pd.read_csv(a.out / 'reproduced/composition.csv').set_index('site_id').group
    pd.testing.assert_series_equal(groups.sort_index(), external_groups.sort_index(), check_names=False)
    gen = pd.read_parquet(a.data / 'train/generation.parquet')
    gen.ts = pd.to_datetime(gen.ts)
    fa = pd.read_parquet(a.data / 'train/forecast_wa.parquet', columns=['wa_station', 'round', 'ts'] + b.WA_COLS)
    fb = pd.read_parquet(a.data / 'train/forecast_wb.parquet', columns=['wb_station', 'round', 'ts'] + b.WB_COLS)
    # Verify the apparent 7-hour train-boundary difference has no daytime labels.
    boundary = {}
    for year in [2022, 2023]:
        start = pd.Timestamp(f'{year}-12-31 17:00', tz='Asia/Seoul')
        end = pd.Timestamp(f'{year + 1}-01-01 00:00', tz='Asia/Seoul')
        g = gen.loc[(gen.ts > start) & (gen.ts <= end)].merge(sites, on='site_id')
        elev = b.solar(pd.DatetimeIndex(g.ts), g.lat.to_numpy(), g.lon.to_numpy())[0]
        boundary[str(year)] = {'observed_rows': len(g), 'daylight_training_rows': int((elev > 0).sum())}
        assert not (elev > 0).any()
    models = {(name, r): lgb.Booster(model_file=str(a.out / 'reproduced' / f'{name}_round{r}.txt'))
              for name in ['baseline', 'improved'] for r in [10, 17]}
    parts = []
    for month in pd.period_range('2024-01', '2025-05', freq='M'):
        start = month.start_time.tz_localize('Asia/Seoul') + pd.Timedelta(hours=1)
        end = (month + 1).start_time.tz_localize('Asia/Seoul')
        ix = pd.date_range(start, end, freq='h')
        fma = fa.loc[fa.ts.between(start, end)]
        fmb = fb.loc[fb.ts.between(start, end)]
        z = None
        for r in [10, 17]:
            d = b.features(None, sites, fma, fmb, ix, r)
            feats = ext.extra(d)
            if z is None:
                z = d[['site_id', 'ts']].copy()
                z['github_baseline'] = 0.0
                z['github_improved'] = 0.0
            else:
                assert z[['site_id', 'ts']].equals(d[['site_id', 'ts']])
            for name in ['baseline', 'improved']:
                columns = b.FEATS if name == 'baseline' else feats
                model = models[name, r]
                assert model.feature_name() == columns
                pred = np.maximum(model.predict(d[columns].astype('float32'), num_threads=4), 0)
                pred[d.elev.to_numpy() <= 0] = 0
                z['github_' + name] += pred * d.cap.to_numpy() / 2
            if r == 17:
                z['daylight'] = d.elev.to_numpy() > 0
        parts.append(z)
        print(f'Predicted linked method locally: {month}', flush=True)
    extpred = pd.concat(parts, ignore_index=True)
    extpred.to_parquet(a.out / 'linked_local_predictions.parquet', index=False)
    own = pd.read_parquet(a.ours / 'site_predictions.parquet')
    z = own.merge(extpred, on=['site_id', 'ts'], validate='one_to_one')
    assert len(z) == len(own) == 3139224
    z = z.rename(columns={'ga_xgb_reference': 'ours_xgb', 'baseline_reference': 'ours_baseline'})
    z['fixed_blend_50_50'] = (z.ours_xgb + z.github_improved) / 2
    names = ['ours_baseline', 'ours_xgb', 'github_baseline', 'github_improved', 'fixed_blend_50_50']
    assert np.isfinite(z[names]).all().all() and (z[names] >= 0).all().all()
    z = z.merge(gen[['site_id', 'ts', 'energy']], on=['site_id', 'ts'], how='left', validate='one_to_one')
    z['group'] = z.site_id.map(groups)
    z['capacity'] = z.site_id.map(sites.set_index('site_id').capacity_kw)
    z['year'] = (z.ts - pd.Timedelta(hours=1)).dt.year
    seen = set(gen.loc[gen.ts.between(pd.Timestamp('2022-01-01 01:00', tz='Asia/Seoul'),
                                    pd.Timestamp('2023-12-31 17:00', tz='Asia/Seoul')), 'site_id'])
    z['cohort'] = np.where(z.site_id.isin(seen), 'seen_2022_2023', 'unseen_2022_2023')
    observed = z.loc[z.energy.notna()].copy()
    group_data = observed.groupby(['group', 'ts'])[['energy'] + names].sum()
    group_data['n_sites'] = observed.groupby(['group', 'ts']).size()
    group_data = group_data.reset_index()
    counts = observed.groupby('ts').size()
    complete_ts = counts[counts == 253].index
    group_data['complete_253'] = group_data.ts.isin(complete_ts)
    group_data['complete_group'] = group_data.n_sites.eq(group_data.group.map(groups.value_counts()))
    prices = pd.read_csv(a.data / 'train/smp.csv')
    prices.ts = pd.to_datetime(prices.ts)
    group_data = group_data.merge(prices, on='ts', validate='many_to_one')
    assert group_data.smp.notna().all()
    group_data['k'] = np.where(group_data.ts.dt.hour.between(11, 15), 3, 1)
    group_data['month'] = (group_data.ts - pd.Timedelta(hours=1)).dt.strftime('%Y-%m')
    group_data['day'] = (group_data.ts - pd.Timedelta(hours=1)).dt.strftime('%Y-%m-%d')
    group_data['year'] = (group_data.ts - pd.Timedelta(hours=1)).dt.year
    for name in names:
        group_data[name + '_penalty'] = .1 * group_data.smp * group_data.k * abs(group_data[name] - group_data.energy)
    # Independent wide-array reconciliation avoids reusing the groupby scorer.
    truth = observed.pivot(index='ts', columns='site_id', values='energy').reindex(index=complete_ts, columns=sites.site_id)
    price = prices.set_index('ts').loc[complete_ts, 'smp'].to_numpy()
    weight = price * np.where(complete_ts.hour.isin([11, 12, 13, 14, 15]), 3, 1)
    for name in names:
        p = z.pivot(index='ts', columns='site_id', values=name).reindex(index=complete_ts, columns=sites.site_id)
        err = (p - truth).to_numpy()
        e = np.column_stack([err[:, groups.reindex(sites.site_id).eq(g).to_numpy()].sum(axis=1) for g in ['G1', 'G2']])
        charge = .1 * np.sum(weight * np.abs(e).sum(axis=1))
        np.testing.assert_allclose(charge, group_data.loc[group_data.complete_253, name + '_penalty'].sum(), atol=.01, rtol=1e-10)
    # Reconcile source repository published aggregate predictions AND local reproduction.
    parity = {}
    for name in ['baseline', 'improved']:
        current = group_data.loc[group_data.year == 2024].set_index(['group', 'ts']).sort_index()
        for kind, folder in [('published', a.repo / 'results'), ('reproduced', a.out / 'reproduced')]:
            old = pd.read_csv(folder / f'{name}_2024_groups.csv')
            old.ts = pd.to_datetime(old.ts)
            old = old.set_index(['group', 'ts']).sort_index()
            pd.testing.assert_index_equal(current.index, old.index)
            np.testing.assert_array_equal(current.n_sites, old.n_sites)
            np.testing.assert_allclose(current.energy, old.actual, atol=1e-6, rtol=1e-10)
            max_diff = float(abs(current['github_' + name] - old.prediction).max())
            parity[f'{kind}_{name}_max_group_prediction_delta_kwh'] = max_diff
            np.testing.assert_allclose(current['github_' + name], old.prediction, atol=.01, rtol=1e-8)
    scores, site_scores, monthly, slices = [], [], [], []
    for period, years in [('2024', [2024]), ('2025_Jan_May', [2025]), ('all', [2024, 2025])]:
        g0 = group_data[group_data.year.isin(years)]
        for scope in ['observed_subset', 'complete_group', 'complete_253']:
            g = g0 if scope == 'observed_subset' else g0[g0[scope]]
            for name in names:
                penalty = float(g[name + '_penalty'].sum())
                scores.append(dict(period=period, scope=scope, model=name, hours=g.ts.nunique(),
                                   group_hours=len(g), penalty_krw=penalty,
                                   settlement_krw=float((g.smp * g.energy).sum()) - penalty,
                                   group_mae_kwh=float(abs(g[name] - g.energy).mean())))
        s0 = observed[observed.year.isin(years)]
        for cohort in ['all', 'seen_2022_2023', 'unseen_2022_2023']:
            s = s0 if cohort == 'all' else s0[s0.cohort == cohort]
            for daylight in [False, True]:
                v = s[s.daylight] if daylight else s
                for name in names:
                    err = v[name] - v.energy
                    site_scores.append(dict(period=period, cohort=cohort, daylight_only=daylight, model=name,
                                            rows=len(v), sites=v.site_id.nunique(), mae_kwh=float(abs(err).mean()),
                                            rmse_kwh=float(np.sqrt((err ** 2).mean())), bias_kwh=float(err.mean()),
                                            capacity_nmae_pct=float((abs(err) / v.capacity).mean() * 100)))
        for peak in [False, True]:
            g = g0[g0.complete_253 & g0.k.eq(3 if peak else 1)]
            for name in names:
                slices.append(dict(period=period, peak=peak, model=name, group_hours=len(g),
                                   penalty_krw=float(g[name + '_penalty'].sum()),
                                   bias_kwh=float((g[name] - g.energy).mean())))
    for (month, scope), g in pd.concat([group_data.assign(scope='observed_subset'),
                                      group_data[group_data.complete_253].assign(scope='complete_253')]).groupby(['month', 'scope']):
        for name in names:
            monthly.append(dict(month=month, scope=scope, model=name, hours=g.ts.nunique(),
                                penalty_krw=float(g[name + '_penalty'].sum())))
    for filename, rows in [('scores', scores), ('site_metrics', site_scores), ('monthly', monthly), ('peak_metrics', slices)]:
        pd.DataFrame(rows).to_csv(a.out / f'{filename}.csv', index=False)
    group_data.to_parquet(a.out / 'group_comparison.parquet', index=False)
    daily = group_data.loc[group_data.year.eq(2025) & group_data.complete_253].groupby('day')[[n + '_penalty' for n in names]].sum()
    delta = (daily.github_improved_penalty - daily.ours_xgb_penalty).to_numpy()
    rng = np.random.default_rng(20261029)
    starts = rng.integers(0, len(delta), size=(3000, int(np.ceil(len(delta) / 7))))
    indices = ((starts[:, :, None] + np.arange(7)) % len(delta)).reshape(3000, -1)[:, :len(delta)]
    resampled = delta[indices].sum(axis=1)
    diagnostics = dict(source_parity=parity, source_hashes_verified=True, independent_scoring='PASS',
                       boundary_audit=boundary, seen_train_sites=len(seen), unseen_train_sites=253-len(seen),
                       jan_may_2025=dict(days=len(delta), ours_better_days=int((delta > 0).sum()),
                                        github_better_days=int((delta < 0).sum()),
                                        github_minus_ours_penalty_krw=float(delta.sum()),
                                        circular_7day_block_bootstrap_95pct_krw=np.quantile(resampled, [.025, .975]).tolist()),
                       interpretation='Positive difference favors ours. Descriptive block bootstrap on fixed predictions, not a guarantee for another season or year.',
                       blending='Fixed 50:50 exploratory diagnostic; not adopted or tuned on the holdout.',
                       script_sha256=digest(__file__))
    save(a.out / 'verification.json', diagnostics)
    print(pd.DataFrame(scores).to_string(index=False), flush=True)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
