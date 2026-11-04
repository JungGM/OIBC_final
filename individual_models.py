"""Independent regressors sharing a kWh MAE contract."""
import joblib
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

SEED = 20261029
THREADS = 4
torch.set_num_threads(THREADS)


def params_for(trial, family):
    if family == "XGBoost":
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 300),
            max_depth=trial.suggest_int("max_depth", 4, 9),
            learning_rate=trial.suggest_float("learning_rate", .02, .09, log=True),
            subsample=trial.suggest_float("subsample", .65, 1),
            colsample_bytree=trial.suggest_float("colsample_bytree", .55, 1))
    if family == "LightGBM":
        return dict(n_estimators=trial.suggest_int("n_estimators", 200, 700),
            num_leaves=trial.suggest_int("num_leaves", 20, 127),
            learning_rate=trial.suggest_float("learning_rate", .02, .09, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 80, 300))
    if family == "CatBoost":
        return dict(iterations=trial.suggest_int("iterations", 150, 500),
            depth=trial.suggest_int("depth", 4, 8),
            learning_rate=trial.suggest_float("learning_rate", .02, .09, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 2, 10))
    if family in ("RandomForest", "ExtraTrees"):
        return dict(n_estimators=trial.suggest_int("n_estimators", 12, 35),
            max_depth=trial.suggest_int("max_depth", 12, 22),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 30),
            max_features=trial.suggest_float("max_features", .35, .8))
    return dict(hidden_size=trial.suggest_categorical("hidden_size", [24, 48]),
        learning_rate=trial.suggest_float("learning_rate", .001, .01, log=True),
        dropout=trial.suggest_float("dropout", 0, .25), epochs=trial.suggest_int("epochs", 4, 8))


def flat(panel):
    observed = panel.observed
    x = panel.x[observed]
    y = np.clip(panel.y[observed] /
                np.broadcast_to(panel.cap[:, None], observed.shape)[observed], 0, 1.2)
    cap = np.broadcast_to(panel.cap[:, None], observed.shape)[observed]
    return np.nan_to_num(x, nan=-999, posinf=-999, neginf=-999), y, cap


def fit_tree(family, params, train, valid=None):
    x, y, cap = flat(train)
    if family == "XGBoost":
        model = xgb.XGBRegressor(**params, objective="reg:absoluteerror",
            tree_method="hist", n_jobs=THREADS, random_state=SEED,
            early_stopping_rounds=25 if valid is not None else None)
        options = dict(sample_weight=cap)
        if valid is not None:
            vx, vy, vc = flat(valid)
            options.update(eval_set=[(vx, vy)], sample_weight_eval_set=[vc], verbose=False)
    elif family == "LightGBM":
        model = lgb.LGBMRegressor(**params, objective="regression_l1",
            n_jobs=THREADS, random_state=SEED, verbose=-1)
        options = dict(sample_weight=cap)
        if valid is not None:
            vx, vy, vc = flat(valid)
            options.update(eval_set=[(vx,vy)], eval_sample_weight=[vc],
                           callbacks=[lgb.early_stopping(25, verbose=False)])
    elif family == "CatBoost":
        model = CatBoostRegressor(**params, loss_function="MAE", thread_count=THREADS,
            random_seed=SEED, verbose=False, allow_writing_files=False)
        options = dict(sample_weight=cap)
        if valid is not None:
            vx, vy, vc = flat(valid)
            options.update(eval_set=(vx,vy), early_stopping_rounds=25)
    else:
        klass = RandomForestRegressor if family == "RandomForest" else ExtraTreesRegressor
        model = klass(**params, max_leaf_nodes=512, n_jobs=THREADS, random_state=SEED)
        options = dict(sample_weight=cap)
    model.fit(x, y, **options)
    return model


class SequenceRegressor(nn.Module):
    def __init__(self, family, n_features, hidden_size, dropout):
        super().__init__()
        kind = nn.GRU if family == "GRU" else nn.LSTM
        self.rnn = kind(n_features, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.head(self.rnn(x)[0]).squeeze(-1)


def sequence_arrays(panel, mean, scale):
    x = np.nan_to_num((panel.x - mean) / scale, nan=0, posinf=0, neginf=0)
    y = np.clip(panel.y / panel.cap[:, None], 0, 1.2)
    return torch.from_numpy(x.astype("float32")), torch.from_numpy(
        np.nan_to_num(y).astype("float32")), torch.from_numpy(panel.observed)


def fit_sequence(family, params, train, valid=None):
    torch.manual_seed(SEED)
    values = train.x.reshape(-1, train.x.shape[-1])
    mean = np.nanmean(values, axis=0)
    scale = np.nanstd(values, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0).astype("float32")
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1).astype("float32")
    net = SequenceRegressor(family, len(mean), params["hidden_size"], params["dropout"])
    optimizer = torch.optim.Adam(net.parameters(), lr=params["learning_rate"])
    x, y, mask = sequence_arrays(train, mean, scale)
    dataset = torch.utils.data.TensorDataset(x,y,mask)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True,
        generator=torch.Generator().manual_seed(SEED))
    best, patience = float("inf"), 0
    for epoch in range(params["epochs"]):
        net.train()
        for bx, by, bm in loader:
            optimizer.zero_grad()
            loss = (net(bx)[bm] - by[bm]).abs().mean()
            loss.backward()
            optimizer.step()
        if valid is not None:
            pred = predict_sequence((net, mean, scale), valid)
            score = np.mean(np.abs(pred[valid.observed] - valid.y[valid.observed]))
            if score < best:
                best, patience = score, 0
                state = {k: v.detach().clone() for k,v in net.state_dict().items()}
            else:
                patience += 1
                if patience >= 3:
                    break
    if valid is not None:
        net.load_state_dict(state)
    return net, mean, scale


def predict_sequence(model, panel):
    net, mean, scale = model
    x, _, _ = sequence_arrays(panel, mean, scale)
    net.eval()
    with torch.no_grad():
        p = np.concatenate([net(batch).numpy() for batch in x.split(512)])
    return np.maximum(p, 0) * panel.cap[:, None]


def predict(model, family, panel):
    if family in ("GRU", "LSTM"):
        return predict_sequence(model, panel)
    x = np.nan_to_num(panel.x.reshape(-1, panel.x.shape[-1]),
        nan=-999, posinf=-999, neginf=-999)
    return np.maximum(model.predict(x).reshape(-1,24), 0) * panel.cap[:, None]


def save(model, family, path):
    if family in ("GRU", "LSTM"):
        net, mean, scale = model
        torch.save(dict(state=net.state_dict(), mean=mean, scale=scale,
            hidden_size=net.rnn.hidden_size, dropout=net.head[0].p),
            path / "model.pt")
    else:
        joblib.dump(model, path / "model.joblib")


def load(family, path):
    if family in ("GRU", "LSTM"):
        saved = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        net = SequenceRegressor(family, len(saved["mean"]), saved["hidden_size"], saved["dropout"])
        net.load_state_dict(saved["state"])
        return net, saved["mean"], saved["scale"]
    return joblib.load(path / "model.joblib")
