from __future__ import annotations

import copy
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from GM_Ver1.common import SEED, transform


def seed_everything(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


class ForecastGRU(nn.Module):
    """24-hour forecast sequence -> 24 capacity-normalized outputs.

    Bidirectionality, when enabled, sees only the same issue's future forecasts,
    never realized generation or future observations.
    """
    def __init__(self, input_size, config, nsites=253):
        super().__init__()
        hidden, layers = config["hidden_size"], config["num_layers"]
        self.embedding = nn.Embedding(nsites, config["embedding_dim"])
        self.gru = nn.GRU(input_size + config["embedding_dim"], hidden, num_layers=layers,
                          dropout=config["dropout"] if layers > 1 else 0,
                          bidirectional=config["bidirectional"], batch_first=True)
        width = hidden * (2 if config["bidirectional"] else 1)
        self.head = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Dropout(config["dropout"]), nn.Linear(hidden, 1))
        nn.init.constant_(self.head[-1].bias, -2.0)

    def forward(self, x, site):
        emb = self.embedding(site).unsqueeze(1).expand(-1, x.shape[1], -1)
        out, _ = self.gru(torch.cat([x, emb], dim=-1))
        return F.softplus(self.head(out).squeeze(-1))


@torch.no_grad()
def predict(model, panel, ids, scaler, batch_size=512):
    model.eval()
    out = np.empty((len(ids), 24), dtype=np.float32)
    for pos in range(0, len(ids), batch_size):
        block = ids[pos:pos+batch_size]
        x = torch.from_numpy(transform(np.array(panel.X[block]), scaler))
        site = torch.from_numpy(panel.site_index[block]).long()
        out[pos:pos+len(block)] = model(x, site).numpy() * panel.capacity[panel.site_index[block], None]
    return out


def train(panel, ids, end, config, scaler, *, val_ids=None, epochs=None, on_epoch=None, model_class=ForecastGRU):
    seed_everything(config.get("seed", SEED))
    model = model_class(len(panel.features), config, panel.nsites)
    optim = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    rng = np.random.default_rng(config.get("seed", SEED))
    best, best_state, best_epoch, wait, history = float("inf"), None, 0, 0, []
    n_epochs = epochs or config["max_epochs"]
    for epoch in range(1, n_epochs+1):
        start = time.time()
        model.train()
        shuffled = rng.permutation(ids)
        loss_sum, count = 0., 0
        for pos in range(0, len(ids), config["batch_size"]):
            block = shuffled[pos:pos+config["batch_size"]]
            x, y, mask, cap = panel.batch(block, end)
            xt = torch.from_numpy(transform(x, scaler))
            yt, mt = torch.from_numpy(y), torch.from_numpy(mask)
            site = torch.from_numpy(panel.site_index[block]).long()
            pred = model(xt, site)
            if config["loss"] == "mse":
                err = (pred-yt).square()
            else:
                err = F.smooth_l1_loss(pred, yt, reduction="none", beta=config["huber_beta"])
            weights = torch.from_numpy((cap / 170.0).astype(np.float32))[:, None]
            denom = (weights * mt).sum()
            if denom.item() <= 0:
                continue
            loss = (err * weights * mt).sum() / denom
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            loss_sum += float(loss.detach()) * int(mt.sum())
            count += int(mt.sum())
        rec = dict(epoch=epoch, train_loss=loss_sum/max(count, 1), seconds=time.time()-start)
        if val_ids is not None:
            prediction = predict(model, panel, val_ids, scaler)
            y = np.array(panel.y[val_ids])
            valid = np.isfinite(y) & (y >= 0)
            mae = float(np.abs(prediction[valid]-y[valid]).mean())
            rec["validation_mae_kwh"] = mae
            if mae < best - 1e-5:
                best, best_state, best_epoch, wait = mae, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                wait += 1
        history.append(rec)
        print(f"{model_class.__name__} epoch {epoch}/{n_epochs}: {rec}", flush=True)
        if on_epoch:
            on_epoch(history)
        if val_ids is not None and wait >= config["patience"]:
            break
    if val_ids is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = n_epochs
    return model, history, best_epoch
