import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def validate_submission(composition, forecast, data_root):
    root = Path(data_root)
    sites = pd.read_csv(root / 'sites.csv')
    times = pd.to_datetime(pd.read_csv(root / 'smp.csv').ts)
    assert list(composition) == ['site_id', 'group']
    assert list(forecast) == ['group', 'ts', 'forecast_kwh']
    assert len(composition) == 253 and (not composition.site_id.duplicated().any())
    assert set(composition.site_id) == set(sites.site_id)
    assert set(composition.group) == {'G1', 'G2'}
    assert len(forecast) == 17520 and (not forecast.duplicated(['group', 'ts']).any())
    assert not forecast.isna().any().any()
    assert np.isfinite(forecast.forecast_kwh).all() and forecast.forecast_kwh.ge(0).all()
    assert forecast.ts.astype(str).str.endswith('+09:00').all()
    for group in ['G1', 'G2']:
        actual = pd.DatetimeIndex(pd.to_datetime(forecast.loc[forecast.group.eq(group), 'ts'])).sort_values()
        assert len(actual) == 8760 and (actual == pd.DatetimeIndex(times).sort_values()).all()
    capacity = composition.merge(sites[['site_id', 'capacity_kw']], on='site_id', validate='one_to_one').groupby('group').capacity_kw.sum()
    assert capacity.ge(20000).all()
    print('VALID', len(composition), len(forecast), capacity.to_dict())
    return capacity
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--submission-dir', default='제출')
    args = parser.parse_args()
    folder = Path(args.submission_dir)
    validate_submission(pd.read_csv(folder / 'composition.csv'), pd.read_csv(folder / 'forecast.csv'), args.data_root)
