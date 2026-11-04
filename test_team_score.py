import importlib.util,tempfile,unittest
from pathlib import Path
import pandas as pd
spec=importlib.util.spec_from_file_location('team_score',Path(__file__).parent/'test기준'/'score_2024.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
class TeamScoreTests(unittest.TestCase):
 def test_hand_score_and_missing_truth(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t); (p/'train').mkdir(); ids=[f'S{i}' for i in range(253)]; pd.DataFrame({'site_id':ids,'capacity_kw':160}).to_csv(p/'sites.csv',index=False); pd.DataFrame({'site_id':ids,'group':['G1']*126+['G2']*127}).to_csv(p/'composition.csv',index=False)
   ix=pd.date_range('2024-01-01 01:00','2025-01-01 00:00',freq='h',tz='Asia/Seoul'); f=pd.MultiIndex.from_product([['G1','G2'],ix],names=['group','ts']).to_frame(index=False); f['forecast_kwh']=0.; f.loc[(f.ts==ix[10])&(f.group=='G1'),'forecast_kwh']=253; f.loc[(f.ts==ix[10])&(f.group=='G2'),'forecast_kwh']=254; f.to_csv(p/'forecast.csv',index=False)
   truth=pd.DataFrame({'site_id':ids,'ts':ix[10],'energy':2.}); partial=truth.iloc[:-1].copy(); partial['ts']=ix[11]; pd.concat([truth,partial]).to_parquet(p/'train/generation.parquet'); pd.DataFrame({'ts':ix,'smp':100.}).to_csv(p/'train/smp.csv',index=False)
   r=m.score(p,p/'composition.csv',p/'forecast.csv'); self.assertEqual(r['complete_hours'],1); self.assertEqual(r['penalty_krw'],30.)
   f.iloc[:-1].to_csv(p/'forecast.csv',index=False)
   with self.assertRaises(ValueError): m.score(p,p/'composition.csv',p/'forecast.csv')
   f.loc[0,'forecast_kwh']=-1; f.to_csv(p/'forecast.csv',index=False)
   with self.assertRaises(ValueError): m.score(p,p/'composition.csv',p/'forecast.csv')
if __name__=='__main__': unittest.main()
