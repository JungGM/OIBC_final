"""Export and check 2024 backtest CSVs locally; these are not official submissions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--sites', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Output directory must be empty')
    args.out.mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(args.sites)
    comp = pd.read_csv(args.run / 'composition_selected.csv')
    pred = pd.read_csv(args.run / 'backtest_forecast_NOT_FOR_SUBMISSION.csv')
    pred['ts'] = pd.to_datetime(pred.ts, utc=True).dt.tz_convert('Asia/Seoul')
    start = pd.Timestamp('2024-01-01 01:00', tz='Asia/Seoul')
    end = pd.Timestamp('2025-01-01 00:00', tz='Asia/Seoul')
    pred = pred.loc[pred.ts.between(start, end), ['group', 'ts', 'forecast_kwh']].copy()
    expected = pd.MultiIndex.from_product([['G1', 'G2'], pd.date_range(start, end, freq='h')], names=['group', 'ts'])
    actual = pd.MultiIndex.from_frame(pred[['group', 'ts']])
    assert list(comp.columns) == ['site_id', 'group']
    assert len(sites) == 253 and sites.site_id.is_unique and comp.site_id.is_unique
    assert set(comp.site_id) == set(sites.site_id) and set(comp.group) == {'G1', 'G2'}
    capacity = comp.merge(sites[['site_id', 'capacity_kw']], on='site_id', validate='one_to_one').groupby('group').capacity_kw.sum()
    assert (capacity >= 20000).all()
    assert len(pred) == 17568 and actual.is_unique and len(expected.difference(actual)) == 0
    assert np.isfinite(pred.forecast_kwh).all() and (pred.forecast_kwh >= 0).all()
    comp.to_csv(args.out / 'composition.csv', index=False)
    pred.sort_values(['group', 'ts']).to_csv(args.out / 'forecast.csv', index=False)
    print(f'PASS: 253 sites, capacities {capacity.to_dict()}, 17,568 group-hour forecasts, 2024 test period')
    print('TEST ONLY: official submission period differs; no organizer acceptance implied.')


if __name__ == '__main__':
    main()
