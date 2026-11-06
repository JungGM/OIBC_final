"""Chronological nonlinear stackers and a condition-dependent economic gate."""

import lightgbm as lgb
import numpy as np
import torch
import xgboost as xgb
from sklearn.linear_model import Ridge
from torch import nn
from torch.nn import functional as F

from .decisions import blend_l1
from .experts import scaled, scaler


SEED = 20261106
STACKS = ("mean", "weighted", "ridge", "lgb_stack", "xgb_stack", "mlp_gate", "economic_gate")


class Gate(nn.Module):
    def __init__(self, features, experts, hidden, dropout):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(features, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, experts))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, context, predictions):
        weights = self.network(context).softmax(dim=-1)
        return (weights*predictions).sum(-1), weights


def gate_loss(point, actual, q, cap, group, economic):
    err = point-actual
    site = F.smooth_l1_loss(point, actual, beta=.035, reduction="none")
    if not economic:
        return site.mean()
    weighted_site = (q[:, None]*site).sum()/(q.sum()*site.shape[1]).clamp_min(1.)
    aggregate = ((err*cap) @ group)/(cap @ group)
    group_loss = F.smooth_l1_loss(aggregate, torch.zeros_like(aggregate), beta=.018, reduction="none")
    return weighted_site + .85*(q[:, None]*group_loss).sum()/(q.sum()*2).clamp_min(1.)


def gate_fit(h, f, y, q, context, future_context, capacity, first, economic, threads, final=False):
    torch.set_num_threads(threads)
    cap = torch.from_numpy(capacity.astype("float32"))
    group = torch.from_numpy(np.stack([first, ~first], axis=1).astype("float32"))
    y = torch.from_numpy((y/capacity).astype("float32"))
    q = torch.from_numpy(np.clip(q/max(q.mean(), 1.), 0., 6.).astype("float32"))
    h_t = torch.from_numpy(h.astype("float32"))
    hist = np.concatenate([h, context], axis=-1).astype("float32")
    future = np.concatenate([f, future_context], axis=-1).astype("float32")
    # Validation is a later block of OOF hours, never sampled from fitting rows.
    split = min(len(h)-24, max(48, (int(len(h)*.78)//24)*24))
    config_grid = ((24, .0015, .10), (48, .0006, .16))
    trials = []
    best = None
    for hidden, lr, dropout in config_grid:
        stats = scaler(hist[:split])
        x = torch.from_numpy(scaled(hist, stats))
        torch.manual_seed(SEED)
        net = Gate(x.shape[-1], h.shape[-1], hidden, dropout)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=.006)
        rng = np.random.default_rng(SEED)
        record = []
        best_loss, best_epoch = np.inf, 0
        for epoch in range(16):
            net.train()
            order = rng.permutation(split)
            for j in range(0, len(order), 32):
                ix = order[j:j+32]
                pred, weights = net(x[ix], h_t[ix])
                loss = gate_loss(pred, y[ix], q[ix], cap, group, economic)
                loss += .0006*(weights-1/h.shape[-1]).square().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.)
                opt.step()
            net.eval()
            with torch.no_grad():
                pred, _ = net(x[split:], h_t[split:])
                val = gate_loss(pred, y[split:], q[split:], cap, group, economic).item()
            record.append(val)
            if val < best_loss-1e-6:
                best_loss, best_epoch = val, epoch+1
            if epoch+1-best_epoch >= 4:
                break
        trials.append(dict(hidden=hidden, learning_rate=lr, dropout=dropout, validation_loss=record,
                           epochs=best_epoch, best_validation_loss=best_loss))
        if best is None or best_loss < best["best_validation_loss"]:
            best = trials[-1]
    stats = scaler(hist)
    x = torch.from_numpy(scaled(hist, stats))
    torch.manual_seed(SEED)
    net = Gate(x.shape[-1], h.shape[-1], best["hidden"], best["dropout"])
    opt = torch.optim.AdamW(net.parameters(), lr=best["learning_rate"], weight_decay=.006)
    rng = np.random.default_rng(SEED)
    for _ in range(best["epochs"]):
        order = rng.permutation(len(h))
        for j in range(0, len(order), 32):
            ix = order[j:j+32]
            pred, weights = net(x[ix], h_t[ix])
            loss = gate_loss(pred, y[ix], q[ix], cap, group, economic)
            loss += .0006*(weights-1/h.shape[-1]).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
    predictions, mean_weights = [], []
    net.eval()
    with torch.inference_mode():
        for j in range(0, len(f), 128):
            x = torch.from_numpy(scaled(future[j:j+128], stats))
            pred, weights = net(x, torch.from_numpy(f[j:j+128].astype("float32")))
            predictions.append(pred.numpy()*capacity)
            mean_weights.append(weights.mean((0, 1)).numpy())
    metadata = dict(trials=trials, chosen=best, mean_inference_weights=np.mean(mean_weights, axis=0).tolist())
    checkpoint = dict(state=net.state_dict(), scaler=[a.tolist() for a in stats], chosen=best) if final else None
    return np.concatenate(predictions), metadata, checkpoint


def fit_stack(history, actual, q, future, context, f_context, first, cap, threads, final=False):
    weights = blend_l1(history, actual, q, first)
    point = {"mean": future.mean(axis=-1), "weighted": future @ weights}
    h = (history/cap[None, :, None]).astype("float32")
    f = (future/cap[None, :, None]).astype("float32")
    shape = f.shape[:2]
    rows = np.sort(np.random.default_rng(SEED).choice(actual.size, min(180000, actual.size), replace=False))
    target = (actual/cap).reshape(-1)[rows]
    inputs = h.reshape(-1, h.shape[-1])[rows]
    outputs = f.reshape(-1, f.shape[-1])
    ridge = Ridge(alpha=6.).fit(inputs, target)
    point["ridge"] = ridge.predict(outputs).reshape(shape).clip(0)*cap
    meta = np.column_stack([inputs, context.reshape(actual.size, -1)[rows]])
    test = np.column_stack([outputs, f_context.reshape(len(outputs), -1)])
    residual = target-inputs.mean(-1)
    site_price = np.clip(q[rows//len(cap)]/max(q.mean(), 1.), .10, 5.)**.5
    models = dict(ridge=ridge, lp_weights=weights)
    fits = {
        "lgb_stack": lgb.LGBMRegressor(objective="regression_l1", n_estimators=180, num_leaves=15,
                         min_child_samples=500, learning_rate=.04, reg_lambda=8,
                         verbosity=-1, random_state=SEED, n_jobs=threads),
        "xgb_stack": xgb.XGBRegressor(objective="reg:pseudohubererror", huber_slope=.05,
                         n_estimators=180, max_depth=3, min_child_weight=50, reg_lambda=10,
                         learning_rate=.04, n_jobs=threads, random_state=SEED),
    }
    for name, model in fits.items():
        model.fit(meta, residual, sample_weight=site_price)
        delta = np.clip(model.predict(test), -.18, .18)
        point[name] = (outputs.mean(-1)+delta).reshape(shape).clip(0)*cap
        models[name] = model
    logs = {"weighted_coefficients": weights.tolist()}
    for name, economic in (("mlp_gate", False), ("economic_gate", True)):
        point[name], logs[name], checkpoint = gate_fit(h, f, actual, q, context, f_context,
                                                       cap, first, economic, threads, final)
        if final:
            models[name] = checkpoint
    return {k: np.asarray(v, dtype="float32") for k, v in point.items()}, logs, models
