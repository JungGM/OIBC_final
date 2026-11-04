"""Frozen pre-2024 model search, ensemble and group experiment. Local only."""
import argparse, json, gc
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import backtest as b

CONFIGS = [
 dict(name='old_xgb', kind='xgb', n_estimators=180,max_depth=8,learning_rate=.03,subsample=.9,colsample_bytree=.7),
 dict(name='xgb_deep',kind='xgb',n_estimators=650,max_depth=6,learning_rate=.025,subsample=.85,colsample_bytree=.85,reg_lambda=10),
 dict(name='lgb_l2_small',kind='lgb',objective='regression',n_estimators=700,num_leaves=31,min_child_samples=200),
 dict(name='lgb_l2_large',kind='lgb',objective='regression',n_estimators=900,num_leaves=63,min_child_samples=250),
 dict(name='lgb_l1',kind='lgb',objective='regression_l1',n_estimators=700,num_leaves=31,min_child_samples=200),
 dict(name='lgb_huber',kind='lgb',objective='huber',alpha=.9,n_estimators=700,num_leaves=31,min_child_samples=200),
]
def fit(d,features,c):
    kw={k:v for k,v in c.items() if k not in ['name','kind']}
    if c['kind']=='xgb':
        m=xgb.XGBRegressor(**kw,tree_method='hist',n_jobs=4,random_state=b.SEED)
    else:
        m=lgb.LGBMRegressor(**kw,learning_rate=.03,colsample_bytree=.9,reg_lambda=5,n_jobs=4,verbosity=-1,random_state=b.SEED)
    m.fit(b.design(d,features,enhanced=True),(d.energy/d.cap).clip(0,1.2))
    return m

def pred(m,d,features):
    p=np.zeros(len(d)); day=d.elev.to_numpy()>0
    p[day]=np.maximum(m.predict(b.design(d.loc[day],features,enhanced=True)),0)*d.loc[day,'cap'].to_numpy()
    assert np.isfinite(p).all()
    return p

def proxy_arrays(d,p,groups,prices):
    z=d.loc[d.energy.notna(),['site_id','ts','energy']].copy(); z['p']=p[d.energy.notna().to_numpy()]; z['group']=z.site_id.map(groups)
    g=z.groupby(['group','ts'])[['energy','p']].sum().reset_index().merge(prices,on='ts',validate='many_to_one')
    return g.energy.to_numpy(),g.p.to_numpy(),(g.smp*g.k).to_numpy()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    if a.out.exists(): raise ValueError('Use new output folder')
    a.out.mkdir(parents=True); (a.out/'models').mkdir()
    f=json.loads((a.cache/'features.json').read_text())['enhanced']; sites=pd.read_csv(a.data/'sites.csv'); ref=sites.set_index('site_id').current_group.map({'A':'G1','B':'G2'})
    prices=pd.read_csv(a.data/'train/smp.csv'); prices['ts']=pd.to_datetime(prices.ts); prices['k']=np.where(prices.ts.dt.hour.between(11,15),3,1)
    b.save_json(a.out/'protocol_before_search.json',{'configs':CONFIGS,'selection':'two pre-2024 forward folds, equal relative penalty; all pair blends 0.25/0.5/0.75; Q4 group gate 1%; no 2024 tuning','code_sha256':b.sha256(__file__),'cached_feature_audit':json.loads((a.cache/'audit.json').read_text()),'group_search_seeds':[20261029+i for i in range(5)]})
    folds=[]; last_predictions=[]
    for fold,(train_end,start,end) in enumerate([(pd.Timestamp('2022-12-31 17:00',tz=b.TZ),b.TUNE_START,b.TUNE_END),(pd.Timestamp('2023-06-30 17:00',tz=b.TZ),b.CAL_START,b.CAL_END)]):
        train=b.load_rows(a.cache,b.START,train_end,labeled=True); valid=b.load_rows(a.cache,start,end); outputs=[]; arrays=[]
        for c in CONFIGS:
            m=fit(train,f,c); p=pred(m,valid,f); y,pp,w=proxy_arrays(valid,p,ref,prices); arrays.append(pp); outputs.append(p)
            b.log(f'fold {fold+1} {c["name"]} proxy penalty {0.1*np.sum(w*abs(pp-y)):,.0f}'); del m
        folds.append((y,np.array(arrays),w)); last_predictions=outputs
        del train; gc.collect()
    weights=[]
    for i in range(len(CONFIGS)):
        v=np.zeros(len(CONFIGS)); v[i]=1; weights.append(v)
        for j in range(i):
            for alpha in [.25,.5,.75]:
                v=np.zeros(len(CONFIGS)); v[i]=alpha; v[j]=1-alpha; weights.append(v)
    history=[]
    for v in weights:
        costs=[float(.1*np.sum(w*abs(v@pp-y))) for y,pp,w in folds]
        relative=[cost/(.1*np.sum(w*abs(pp[0]-y))) for cost,(y,pp,w) in zip(costs,folds)]
        history.append({'weights':v.tolist(),'penalties':costs,'mean_relative_penalty':float(np.mean(relative))})
    best=min(history,key=lambda h:h['mean_relative_penalty']); weights=np.array(best['weights']); b.save_json(a.out/'selection.json',{'best':best,'all':history,'configs':CONFIGS})
    b.log(f'Selected weights {weights.tolist()}'); p=sum(w*q for w,q in zip(weights,last_predictions)); candidates=[]
    for seed in range(5):
        b.SEED=20261029+seed; sub=a.out/f'group_{seed}'; sub.mkdir(); _,cand,eligible,info=b.optimize_groups(valid,p,sites,prices,sub); candidates.append((info['proxy_penalty_after'],cand,eligible,info))
    _,candidate,eligible,info=min(candidates,key=lambda x:x[0]); b.SEED=20261029
    check=b.load_rows(a.cache,b.CHECK_START,b.CHECK_END); train=b.load_rows(a.cache,b.START,pd.Timestamp('2023-06-30 17:00',tz=b.TZ),labeled=True); pc=np.zeros(len(check))
    for w,c in zip(weights,CONFIGS):
        if w: m=fit(train,f,c); pc+=w*pred(m,check,f); del m
    before=b.cohort_score(check,pc,ref,eligible,prices); after=b.cohort_score(check,pc,candidate,eligible,prices); accept=after['penalty_krw']<.99*before['penalty_krw']; groups=candidate if accept else ref
    b.save_json(a.out/'group_validation.json',{'accepted':bool(accept),'reference':before,'candidate':after,'search':info}); groups.rename('group').reset_index().to_csv(a.out/'composition_selected.csv',index=False)
    # Predictive intervals describe generation uncertainty, not a claim of weather-forecast error bounds.
    qc=[]; checkq=[]
    for alpha in [.1,.9]:
        c=dict(name=f'quantile_{alpha}',kind='lgb',objective='quantile',alpha=alpha,n_estimators=500,num_leaves=31,min_child_samples=200); qc.append(c); m=fit(train,f,c); checkq.append(pred(m,check,f)); del m
    lo=np.minimum(*checkq); hi=np.maximum(*checkq); mask=check.energy.notna()&(check.elev>0)
    coverage=float(((check.energy[mask]>=lo[mask])&(check.energy[mask]<=hi[mask])).mean())
    frozen={'weights':weights.tolist(),'configs':CONFIGS,'group_candidate_accepted':bool(accept),'fit_cutoff':str(b.FULL_CUTOFF),'selection_end':str(b.CHECK_END),'features':f,'q10_q90_Q4_daylight_observed_coverage':coverage,'interval_warning':'Nominal 80% site generation interval, no calibrated fleet/weather guarantee','script_sha256':b.sha256(__file__)}
    b.save_json(a.out/'frozen_plan.json',frozen); b.log('Frozen before loading any 2024 target or reporting test metrics')
    del train,valid,check,last_predictions,folds; gc.collect()
    train=b.load_rows(a.cache,b.START,b.FULL_CUTOFF,labeled=True); models=[]
    for w,c in zip(weights,CONFIGS):
        if w:
            m=fit(train,f,c); models.append((w,m)); path=a.out/'models'/c['name']; m.save_model(str(path)+'.ubj') if c['kind']=='xgb' else m.booster_.save_model(str(path)+'.txt')
    quantiles=[fit(train,f,c) for c in qc]
    for c,m in zip(qc,quantiles): m.booster_.save_model(str(a.out/'models'/c['name'])+'.txt')
    del train; gc.collect(); rows=[]
    for month in pd.period_range('2024-01','2024-12',freq='M'):
        d=pd.read_parquet(a.cache/'features'/f'{month}.parquet').drop(columns='energy'); z=d[['site_id','ts']].copy(); z['frozen_selected']=sum(w*pred(m,d,f) for w,m in models); q=[pred(m,d,f) for m in quantiles]; z['q10']=np.minimum(*q); z['q90']=np.maximum(*q); z['daylight']=d.elev>0; rows.append(z); b.log(f'Predicted {month}')
    p=pd.concat(rows,ignore_index=True); p.to_parquet(a.out/'site_predictions.parquet',index=False); forecast=p.assign(group=p.site_id.map(groups)).groupby(['group','ts']).frozen_selected.sum().rename('forecast_kwh').reset_index(); forecast.to_csv(a.out/'backtest_forecast_NOT_FOR_SUBMISSION.csv',index=False)
    # Only now load 2024 labels for final reporting. Never update the frozen configuration.
    gen=pd.read_parquet(a.data/'train/generation.parquet'); old=pd.read_parquet(a.cache/'site_predictions.parquet'); p=p.merge(old[['site_id','ts','baseline_reference','ga_xgb_reference']],on=['site_id','ts'],validate='one_to_one'); b.END=pd.Timestamp('2025-01-01 00:00',tz=b.TZ)
    # Direct 2024 evaluation avoids the original helper's 2025 reporting branch.
    z=p.merge(gen,on=['site_id','ts'],how='left',validate='one_to_one'); counts=z.groupby('ts').energy.count(); complete=counts[counts==253].index; metrics=[]
    for name,gmap in [('baseline_reference',ref),('ga_xgb_reference',ref),('frozen_selected',groups)]:
        valid=z.energy.notna(); err=z.loc[valid,name]-z.loc[valid,'energy']; full=z[z.ts.isin(complete)].copy(); full['group']=full.site_id.map(gmap); g=full.groupby(['group','ts'])[[name,'energy']].sum().reset_index().merge(prices,on='ts',validate='many_to_one'); s=b.penalty(g.energy,g[name],g.smp,g.k); metrics.append({'model':name,'complete_hours':len(complete),'observed_rows':int(valid.sum()),'site_mae':float(err.abs().mean()),'site_rmse':float(np.sqrt((err**2).mean())),**s})
    b.save_json(a.out/'metrics.json',metrics)
    mask=z.energy.notna()&z.daylight; b.save_json(a.out/'uncertainty.json',{'nominal':.8,'2023_Q4_observed_daylight_coverage':coverage,'2024_observed_daylight_coverage':float(((z.energy[mask]>=z.q10[mask])&(z.energy[mask]<=z.q90[mask])).mean()),'2024_mean_width_kwh':float((z.q90[mask]-z.q10[mask]).mean()),'note':'No independence or distribution-free coverage guarantee; site quantiles cannot be summed into fleet quantiles'})
    b.log(str(metrics)); b.save_json(a.out/'completion.json',{'complete':True,'forecast_sha256':b.sha256(a.out/'backtest_forecast_NOT_FOR_SUBMISSION.csv')})
if __name__=='__main__': main()
