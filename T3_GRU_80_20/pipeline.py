import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from features import build_features
from train_gru import fit_gru
from predict import predict_ensemble
from validate_submission import validate_submission
T3_PARAMETERS = dict(max_depth=6, eta=0.05, min_child_weight=10, reg_lambda=10, subsample=0.8, colsample_bytree=0.8, tree_method='hist', objective='reg:squarederror', seed=42, nthread=4)

def make_sequences(data):
    keys = pd.DataFrame(dict(site=data.site_id, day=(data.ts - pd.Timedelta(hours=1)).dt.floor('D'), row=np.arange(len(data)), ts=data.ts))
    sequences = []
    for _, day in keys.groupby(['site', 'day'], sort=False):
        if len(day) == 24 and day.ts.nunique() == 24:
            sequences.append(day.sort_values('ts').row.to_numpy())
    return np.asarray(sequences)

def run(data_root, output_dir):
    root = Path(data_root)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output directory must be empty')
    sites = pd.read_csv(root / 'sites.csv')
    generation = pd.read_parquet(root / 'train/generation.parquet')
    smp = pd.read_csv(root / 'train/smp.csv')
    smp.ts = pd.to_datetime(smp.ts)
    data = generation.merge(sites, on='site_id', validate='many_to_one').merge(smp, on='ts', validate='many_to_one')
    data, columns = build_features(data, root, 'train')
    assert len(columns) == 89
    features = data[columns].to_numpy(dtype=np.float32)
    target = data.energy.to_numpy()
    capacity = data.capacity_kw.to_numpy()
    year = data.ts.dt.year.to_numpy()
    train = np.flatnonzero((year >= 2021) & (year <= 2024))
    matrix = xgb.DMatrix(features[train], label=target[train] / capacity[train], weight=np.ones(len(train)), nthread=4)
    t3 = xgb.train(T3_PARAMETERS, matrix, num_boost_round=400)
    print('T3 training rows', len(train), flush=True)
    del matrix
    sequences = make_sequences(data)
    sequences = sequences[np.all((year[sequences] >= 2021) & (year[sequences] <= 2024), axis=1)]
    assert np.all(year[sequences] <= 2024)
    gru, mean, std = fit_gru(features, target, capacity, sequences, seed=42)
    del data, features, generation
    gc.collect()
    times = pd.read_csv(root / 'smp.csv')
    times.ts = pd.to_datetime(times.ts)
    test = sites.merge(times, how='cross')
    test['energy'] = np.nan
    test, actual_columns = build_features(test, root, 'test')
    assert actual_columns == columns and len(test) == 253 * 8760
    assert not test.duplicated(['site_id', 'ts']).any()
    sequences = make_sequences(test)
    prediction, t3_prediction, gru_prediction = predict_ensemble(test, columns, t3, gru, mean, std, sequences)
    test['group'] = test.current_group.map({'A': 'G1', 'B': 'G2'})
    test['forecast_kwh'] = prediction
    composition = sites[['site_id', 'current_group']].rename(columns={'current_group': 'group'})
    composition['group'] = composition.group.map({'A': 'G1', 'B': 'G2'})
    forecast = test.groupby(['group', 'ts'], as_index=False).forecast_kwh.sum().sort_values(['group', 'ts'])
    validate_submission(composition, forecast, root)
    output.mkdir(parents=True, exist_ok=True)
    composition.to_csv(output / 'composition.csv', index=False, mode='x')
    forecast.to_csv(output / 'forecast.csv', index=False, mode='x')
    validate_submission(pd.read_csv(output / 'composition.csv'), pd.read_csv(output / 'forecast.csv'), root)
    print('SAVED', output.resolve(), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', default='제출')
    args = parser.parse_args()
    run(args.data_root, args.output_dir)
