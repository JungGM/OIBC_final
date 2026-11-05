import numpy as np
import pandas as pd

def evaluate(data, prediction):
    frame = data[['ts', 'current_group', 'smp', 'hour', 'energy']].copy()
    frame['prediction'] = prediction
    grouped = frame.groupby(['current_group', 'ts']).agg(prediction=('prediction', 'sum'), actual=('energy', 'sum'), smp=('smp', 'first'), hour=('hour', 'first'))
    grouped['penalty'] = 0.1 * grouped.smp * np.where(grouped.hour.between(11, 15), 3, 1) * (grouped.prediction - grouped.actual).abs()
    error = prediction - data.energy.to_numpy()
    return dict(group_penalty=grouped.groupby(level=0).penalty.sum().to_dict(), total_penalty=grouped.penalty.sum(), peak_penalty=grouped.loc[grouped.hour.between(11, 15), 'penalty'].sum(), mae=np.abs(error).mean(), rmse=np.sqrt(np.mean(error ** 2)), wape=100 * np.abs(error).sum() / data.energy.abs().sum())
