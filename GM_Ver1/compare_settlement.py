"""Produce a GRU challenger and compare both forecasts against shared scenario actuals."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from GM_Ver1.common import local_output, sha256, write_json
from GM_Ver1.genetic_algorithm import check_submission, group_matrix, site_panel, subset
from GM_Ver1.run import FINAL_END
from train.dataset import current_dataset


def run(out, dataset):
    root = Path(__file__).resolve().parent
    ga = root / 'GeneticAlgorithm/Ver1'
    roster = pd.read_parquet(dataset/'reference_composition.parquet').sort_values('site_id').reset_index(drop=True)
    capacity = roster.capacity_kw.to_numpy()
    sites = roster.site_id.tolist()
    final_assignment = pd.read_csv(ga/'final/composition.csv').set_index('site_id').loc[sites]
    final_bits = final_assignment.group.eq('G1').to_numpy()
    check_bits = pd.read_csv(ga/'diagnostic_composition.csv').set_index('site_id').loc[sites].group.eq('G1').to_numpy()
    check = pd.read_parquet(root/'GradientDescent/Ver1/check_2025_AprMay/predictions.parquet')
    actual = pd.read_parquet(dataset/'training_data.parquet', columns=['site_id','ts','energy_kwh'],
                            filters=[('ts','>=',check.ts.min()),('ts','<=',check.ts.max())])
    check = check.merge(actual,on=['site_id','ts'],validate='one_to_one',how='left')
    panel = subset(site_panel(check[['site_id','ts']],check.prediction.to_numpy(),check.energy_kwh.to_numpy(),roster,dataset),end=FINAL_END)
    hours = len(panel['times'])//24*24
    assert hours == 1440 and panel['times'][0].hour == 1
    assert panel['times'].equals(pd.date_range(panel['times'][0],periods=len(panel['times']),freq='h'))
    old_cap = np.array([capacity[check_bits].sum(),capacity[~check_bits].sum()])
    new_cap = np.array([capacity[final_bits].sum(),capacity[~final_bits].sum()])
    residual = group_matrix(panel['errors'][:hours],check_bits).reshape(-1,24,2)/old_cap*new_cap
    prices = pd.read_parquet(dataset/'evaluation_weights.parquet').sort_values('ts')
    times = pd.DatetimeIndex(prices.ts)
    original = pd.read_csv(ga/'final/forecast.csv')
    original.ts = pd.to_datetime(original.ts,utc=True).dt.tz_convert(times.tz)
    baseline = original.pivot(index='ts',columns='group',values='forecast_kwh').reindex(times)[['G1','G2']].to_numpy()
    path = root/'GRU/Ver1/final/predictions.parquet'
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['training_end']==FINAL_END and sha256(path)==meta['sha256']
    gru = pd.read_parquet(path).pivot(index='ts',columns='site_id',values='prediction').reindex(index=times,columns=sites).to_numpy()
    challenger = group_matrix(gru.astype(np.float64),final_bits)
    night = times.hour.isin(range(1,5))
    corrected = baseline.copy()
    corrected[night] = 0
    assert np.isfinite(baseline).all() and np.isfinite(challenger).all()
    draws = np.random.default_rng(20261105).integers(len(residual),size=(1000,len(times)//24))
    smp = prices.smp.to_numpy()[None,:,None]
    weighted = smp*prices.k.to_numpy()[None,:,None]*.1
    comparison = {}
    for scale in (.5,1.,1.5,2.):
        gross, penalties = [], [[],[],[]]
        for batch in np.array_split(draws,32):
            truth = np.maximum(baseline-residual[batch].reshape(-1,len(times),2)*scale,0)
            gross.extend((truth*smp).sum(axis=(1,2)))
            for bucket, pred in zip(penalties,(baseline,challenger,corrected)):
                bucket.extend((np.abs(truth-pred)*weighted).sum(axis=(1,2)))
        revenue = float(np.mean(gross))
        gain = np.asarray(penalties[0])-np.asarray(penalties[1])
        comparison[str(scale)] = dict(shared_revenue_krw=revenue,
            current_penalty_krw=float(np.mean(penalties[0])),gru_penalty_krw=float(np.mean(penalties[1])),
            current_settlement_krw=float(np.mean(np.asarray(gross)-penalties[0])),
            gru_settlement_krw=float(np.mean(np.asarray(gross)-penalties[1])),
            gru_gain_krw=float(gain.mean()),gain_p10_p90_krw=np.quantile(gain,[.1,.9]).tolist(),
            night_corrected_settlement_krw=float(np.mean(np.asarray(gross)-penalties[2])),
            night_corrected_gain_krw=float(np.mean(np.asarray(penalties[0])-penalties[2])))
    previous = json.loads((ga/'expected_settlement.json').read_text())['scenarios']['1.0']['mean_settlement_krw']
    np.testing.assert_allclose(comparison['1.0']['current_settlement_krw'],previous,atol=1.,rtol=0)
    final_assignment.reset_index().to_csv(out/'composition.csv',index=False)
    pd.DataFrame(dict(group=np.repeat(['G1','G2'],len(times)),ts=np.tile(times.astype(str),2),
                      forecast_kwh=challenger.T.reshape(-1))).to_csv(out/'forecast.csv',index=False)
    caps = check_submission(out,roster,times,challenger)
    fixed = out/'night_corrected'
    fixed.mkdir(exist_ok=True)
    final_assignment.reset_index().to_csv(fixed/'composition.csv',index=False)
    pd.DataFrame(dict(group=np.repeat(['G1','G2'],len(times)),ts=np.tile(times.astype(str),2),
                      forecast_kwh=corrected.T.reshape(-1))).to_csv(fixed/'forecast.csv',index=False)
    check_submission(fixed,roster,times,corrected)
    write_json(out/'comparison.json',dict(method='Shared annual scenario actuals generated from current forecast and 60 April-May out-of-time error days, with capacity scaling and paired draws',
        choice='GRU1 chosen after spring diagnostic; same GA allocation and no new fitting',
        capacities_kw=caps,scenarios=comparison))
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dataset',type=Path)
    a=p.parse_args()
    run(local_output(a.out),a.dataset or current_dataset())
