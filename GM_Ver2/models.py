"""Capacity-normalized boosted trees and a small joint day forecaster."""

import numpy as np
import torch
import xgboost as xgb
from torch import nn
from torch.nn import functional as F


NAMES = ["tree_history", "tree_recent", "tree_weather_shape", "joint_transformer",
         "tree_dense_squared", "bidirectional_gru"]


def tree(x, y, q, capacity, allowed, predict_x, names, variant, budget, threads, seed, path):
    rng = np.random.default_rng(seed)
    if variant == 1:
        allowed = allowed[-365:]
    if variant == 4:
        budget = max(budget, 520000)
    daylight = x[..., names.index("solar_signed")] > -0.104528
    day_mask = np.zeros(len(x), dtype=bool)
    day_mask[allowed] = True
    good = daylight & np.isfinite(y) & (y >= 0) & (q > 0)[..., None] & day_mask[:, None, None]
    rows = np.flatnonzero(good.ravel())
    if len(rows) > budget:
        rows = np.sort(rng.choice(rows, budget, replace=False))
    day, hour, site = np.unravel_index(rows, y.shape)
    use = [i for i, n in enumerate(names) if variant != 0 or not
           n.endswith(("_revision", "_rise", "_next", "_day_mean", "_disagreement"))]
    X = x[day, hour, site][:, use]
    weight = np.clip(q[day, hour]/q[day, hour].mean(), .15, 5.)**(1 if variant == 2 else .5)
    weight *= np.sqrt(capacity[site]/np.median(capacity))
    settings = dict(objective="reg:squarederror" if variant == 4 else "reg:pseudohubererror",
                    base_score=.25, tree_method="hist", max_bin=128,
                    max_depth={0: 6, 1: 6, 2: 8, 4: 7}[variant],
                    eta=.07, subsample=.85, colsample_bytree=.85, reg_lambda=4,
                    min_child_weight=8, nthread=threads, seed=seed)
    train = xgb.QuantileDMatrix(X, y[day, hour, site]/capacity[site], weight=weight,
                                max_bin=128, nthread=threads)
    if variant != 4:
        settings["huber_slope"] = .15
    rounds = 300 if variant == 4 else (190 if budget < 700000 else 360)
    booster = xgb.train(settings, train, num_boost_round=rounds,
                        verbose_eval=False)
    booster.save_model(path)
    result = []
    for part in np.array_split(predict_x, max(1, (len(predict_x)+13)//14)):
        pred = booster.inplace_predict(part.reshape(-1, x.shape[-1])[:, use])
        result.append(pred.reshape(part.shape[:-1]))
    return np.maximum(np.concatenate(result), 0)*capacity


class DayModel(nn.Module):
    def __init__(self, features, capacity, groups, mapping):
        super().__init__()
        self.site = nn.Embedding(len(capacity), 8)
        self.projection = nn.Linear(features * 2 + 8, 32)
        block = nn.TransformerEncoderLayer(32, 4, 64, .1, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.temporal = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 1))
        self.register_buffer("cap", torch.tensor(capacity, dtype=torch.float32))
        self.register_buffer("groups", torch.tensor(groups, dtype=torch.float32))
        self.register_buffer("mapping", torch.tensor(mapping, dtype=torch.float32))

    def forward(self, x, solar):
        # Each token is already known at the same day-ahead issuance.
        n, h, s, f = x.shape
        valid = torch.isfinite(x)
        id_features = self.site.weight.expand(n, h, -1, -1)
        x = torch.cat([torch.nan_to_num(x).clamp(-8, 8), valid.float(), id_features], dim=-1)
        z = self.projection(x).transpose(1, 2).reshape(n*s, h, 32)
        z = self.temporal(z).reshape(n, s, h, 32).transpose(1, 2)
        local = torch.einsum("bhif,ij->bhjf", z, self.mapping)
        global_context = (z*self.cap[None, None, :, None]).sum(2, keepdim=True)/self.cap.sum()
        cf = F.softplus(self.output(torch.cat([z, local, global_context.expand_as(z)], dim=-1))
                        .squeeze(-1) - 2.)
        return cf * (solar > -0.104528)


def day_loss(cf, truth, q, cap, groups):
    valid = torch.isfinite(truth) & (q > 0).unsqueeze(-1)
    err = cf - torch.nan_to_num(truth/cap)
    w = q.unsqueeze(-1)
    good = valid & (truth >= 0)
    site = (F.smooth_l1_loss(err, torch.zeros_like(err), beta=.05, reduction="none")
            * good * w).sum() / (good*w).sum().clamp_min(1)
    complete = valid.all(dim=-1)
    kwh = err * cap
    group = (kwh @ groups) / (cap @ groups)
    fleet = kwh.sum(-1) / cap.sum()
    total = group.abs().mean(-1) + .5*fleet.abs()
    aggregate = (total * q * complete).sum() / (q * complete).sum().clamp_min(1)
    return site + .7*aggregate


def neural(x, y, q, capacity, groups, allowed, predict_x, sites, names, epochs, threads, seed, path):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    coord = sites[["lat", "lon"]].to_numpy()
    dist = ((coord[:, None, :] - coord[None, :, :])**2).sum(-1)
    np.fill_diagonal(dist, np.inf)
    nearest = np.argsort(dist, axis=0)[:8]
    mapping = np.zeros_like(dist)
    np.put_along_axis(mapping, nearest, 1/len(nearest), axis=0)
    use = [i for i, name in enumerate(names) if not name.endswith("_code")]
    probe = x[allowed[::max(1, len(allowed)//36)]][..., use]
    mean = np.nan_to_num(np.nanmean(probe, axis=(0, 1, 2)))
    scale = np.maximum(np.nan_to_num(np.nanstd(probe, axis=(0, 1, 2))), .001)
    model = DayModel(len(use), capacity, groups, mapping)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    sun = names.index("solar_signed")
    positive = q > 0
    q = np.where(positive, (q / q[positive].mean()).clip(.15, 5), 0).astype("float32")

    def forecast(part):
        scaled = ((part[..., use]-mean)/scale).astype("float32")
        return model(torch.from_numpy(scaled), torch.from_numpy(part[..., sun].copy()))

    for epoch in range(epochs):
        model.train()
        # Prioritize covered days. Each sampled day still sees all sites and hours.
        complete = (np.isfinite(y[allowed]).all(axis=-1) & positive[allowed]).sum(axis=1)
        prob = 1 + 6 * complete / 24
        sample = rng.choice(allowed, size=min(len(allowed), 64), replace=False, p=prob/prob.sum())
        mean_loss = 0.
        for ids in np.array_split(sample, (len(sample)+1)//2):
            cf = forecast(x[ids])
            loss = day_loss(cf, torch.from_numpy(y[ids].astype("float32")),
                            torch.from_numpy(q[ids]), model.cap, model.groups)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            mean_loss += loss.item()/((len(sample)+1)//2)
        print(f"  Joint epoch {epoch+1}/{epochs}: {mean_loss:.5f}", flush=True)
    torch.save(dict(state=model.state_dict(), mean=mean.tolist(), scale=scale.tolist(),
                    features=[names[i] for i in use]), path)
    model.eval()
    with torch.inference_mode():
        return np.concatenate([forecast(part).numpy() for part in
                               np.array_split(predict_x, (len(predict_x)+1)//2)])*capacity


class SequenceGRU(nn.Module):
    def __init__(self, features, n_sites):
        super().__init__()
        self.embedding = nn.Embedding(n_sites, 8)
        self.gru = nn.GRU(features+8, 32, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(64, 24), nn.SiLU(), nn.Dropout(.08), nn.Linear(24, 1))

    def forward(self, x, sites):
        emb = self.embedding(sites)[:, None, :].expand(-1, x.shape[1], -1)
        h, _ = self.gru(torch.cat([x, emb], dim=-1))
        return F.softplus(self.head(h).squeeze(-1)-2)


def recurrent(x, y, q, capacity, allowed, predict_x, names, epochs, threads, seed, path):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed)
    use = [i for i, name in enumerate(names) if not name.endswith(("_code", "_next", "_rise"))]
    probe = x[allowed[::max(1, len(allowed)//36)]][..., use]
    mean = np.nan_to_num(np.nanmean(probe, axis=(0, 1, 2)))
    scale = np.maximum(np.nan_to_num(np.nanstd(probe, axis=(0, 1, 2))), .001)
    allowed_targets = np.isfinite(y[allowed]) & (y[allowed] >= 0) & (q[allowed] > 0)[..., None]
    day, site = np.where(allowed_targets.sum(axis=1) >= 8)
    day = allowed[day]
    model = SequenceGRU(len(use), len(capacity))
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0003)
    typical_price = q[q > 0].mean()

    def encode(values):
        return torch.from_numpy(np.nan_to_num((values[..., use]-mean)/scale).clip(-8, 8))

    for epoch in range(epochs+2):
        model.train()
        ids = rng.choice(len(day), min(16000, len(day)), replace=False)
        for chunk in np.array_split(ids, (len(ids)+255)//256):
            d, s = day[chunk], site[chunk]
            xt = encode(x[d, :, s])
            yt = y[d, :, s]/capacity[s, None]
            weight = np.sqrt(np.clip(q[d]/typical_price, 0., 5.))*np.sqrt(capacity[s, None])
            weight *= np.isfinite(yt) & (yt >= 0)
            weight = torch.from_numpy(weight.astype("float32"))
            err = model(xt, torch.from_numpy(s)) - torch.from_numpy(np.nan_to_num(yt).astype("float32"))
            loss = (weight*err.square()).sum()/weight.sum().clamp_min(1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        print(f"  GRU epoch {epoch+1}/{epochs+2}", flush=True)
    torch.save(dict(state=model.state_dict(), mean=mean.tolist(), scale=scale.tolist(),
                    features=[names[i] for i in use]), path)
    result = []
    model.eval()
    with torch.inference_mode():
        for batch in np.array_split(predict_x, (len(predict_x)+1)//2):
            data = batch.transpose(0, 2, 1, 3).reshape(-1, 24, batch.shape[-1])
            ids = np.tile(np.arange(len(capacity)), len(batch))
            pred = model(encode(data), torch.from_numpy(ids)).numpy()
            result.append(pred.reshape(len(batch), len(capacity), 24).transpose(0, 2, 1)*capacity)
    return np.concatenate(result)
