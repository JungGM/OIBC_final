import numpy as np
import torch
import xgboost as xgb
from features import night_mask
from train_gru import scale_features

def predict_ensemble(data, columns, t3, gru, mean, std, sequences):
    features = data[columns].to_numpy(dtype=np.float32)
    capacity = data.capacity_kw.to_numpy()
    t3_prediction = np.maximum(t3.predict(xgb.DMatrix(features, nthread=4)) * capacity, 0)
    scaled = scale_features(features, mean, std)
    output = np.full(len(data), np.nan)
    gru.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), 512):
            rows = sequences[start:start + 512]
            output[rows.ravel()] = gru(torch.from_numpy(scaled[rows])).numpy().ravel()
    gru_prediction = np.clip(output, 0, 1) * capacity
    dark = night_mask(data)
    t3_prediction[dark] = 0
    gru_prediction[dark] = 0
    prediction = 0.8 * t3_prediction + 0.2 * gru_prediction
    if not np.isfinite(prediction).all():
        raise ValueError('Incomplete GRU prediction grid')
    return (prediction, t3_prediction, gru_prediction)
