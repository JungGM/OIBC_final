"""Day-ahead solar model with partial-panel and full-panel settlement losses."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .data import prices
from .fit import FOLDS, ORIGINS, safe_output


class SolarJoint(nn.Module):
    def __init__(self, features, capacities, groups):
        super().__init__()
        n = len(capacities)
        self.site_embedding = nn.Embedding(n, 12)
        self.local = nn.Sequential(nn.Linear(features*2+12, 48), nn.LayerNorm(48), nn.SiLU())
        self.context = nn.Linear(48*3, 40)
        self.temporal = nn.Sequential(nn.Conv1d(40, 40, 3, padding=1, groups=5), nn.SiLU(),
                                      nn.Conv1d(40, 40, 5, padding=2, groups=5))
        self.out = nn.Sequential(nn.Linear(40, 24), nn.SiLU(), nn.Linear(24, 1))
        self.register_buffer("capacity", torch.tensor(capacities, dtype=torch.float32))
        self.register_buffer("groups", torch.tensor(groups, dtype=torch.float32))

    def forward(self, inputs, sun):
        valid = torch.isfinite(inputs)
        n, h, s, _ = inputs.shape
        ids = self.site_embedding.weight.expand(n, h, s, -1)
        local = self.local(torch.cat([torch.nan_to_num(inputs).clamp(-8, 8), valid.float(), ids], -1))
        weight = self.capacity/self.capacity.sum()
        fleet = (local*weight[None, None, :, None]).sum(2, keepdim=True).expand_as(local)
        day = local.mean(1, keepdim=True).expand_as(local)
        z = self.context(torch.cat([local, fleet, day], -1)).permute(0, 2, 3, 1).reshape(n*s, 40, h)
        z = (z+self.temporal(z)).reshape(n, s, 40, h).permute(0, 3, 1, 2)
        envelope = (sun+.105).clamp(0, 1.1).pow(.55)
        return envelope * F.softplus(self.out(z).squeeze(-1)-1.7)


def objective(model, point, actual, prices, group_weight):
    cap, group = model.capacity, model.groups
    valid = torch.isfinite(actual) & (actual >= 0) & (prices > 0).unsqueeze(-1)
    residual = point - torch.where(valid, actual, 0.)/cap
    base_weight = prices.sqrt().unsqueeze(-1)*valid*torch.sqrt(cap/cap.mean())
    site = (base_weight*residual.square()).sum()/base_weight.sum().clamp_min(1.)
    # The same observed members appear on both sides of the sum.
    # Use the actual complete settlement loss whenever the whole group is present.
    coverage = (valid.float()*cap) @ group
    group_error = ((residual*valid*cap) @ group)/coverage.clamp_min(1.)
    share = coverage/(cap @ group)
    weight = prices.unsqueeze(-1)*share.square()*(share >= .3)
    smooth = F.smooth_l1_loss(group_error, torch.zeros_like(group_error), beta=.025, reduction="none")
    groups = (weight*smooth).sum()/weight.sum().clamp_min(1.)
    return site + group_weight*groups


def train(raw, artifacts, out, threads):
    torch.set_num_threads(threads)
    torch.manual_seed(20261108)
    rng = np.random.default_rng(20261108)
    roster = pd.read_csv(raw/"sites.csv").sort_values("site_id")
    capacity = roster.capacity_kw.to_numpy()
    a = roster.current_group.eq("A").to_numpy()
    groups = np.stack([a, ~a, np.ones_like(a)], axis=1)
    feature_names = json.loads((artifacts/"data_summary.json").read_text())["train"]["features"]
    use = [i for i, name in enumerate(feature_names) if not name.endswith("_code")]
    sun_index = feature_names.index("solar_signed")
    x = np.load(artifacts/"train_x.npy", mmap_mode="r")
    y = np.load(artifacts/"train_y.npy", mmap_mode="r")
    eval_x = np.load(artifacts/"eval_x.npy", mmap_mode="r")
    price = prices(raw/"train/smp.csv")
    days = price.index[::24].normalize()
    stops = [days.searchsorted(pd.Timestamp(origin, tz=days.tz)) for origin in ORIGINS]
    probe = x[np.arange(0, stops[0], 24)][..., use]
    mean = np.nan_to_num(np.nanmean(probe, axis=(0, 1, 2)))
    scale = np.maximum(np.nan_to_num(np.nanstd(probe, axis=(0, 1, 2))), .001)
    net = SolarJoint(len(use), capacity, groups)

    def predict(batch):
        values = torch.from_numpy(((batch[..., use]-mean)/scale).astype("float32"))
        sun = torch.from_numpy(batch[..., sun_index].copy())
        return net(values, sun)

    for i, (label, origin, stop) in enumerate(zip(FOLDS, ORIGINS, stops)):
        output = out/f"{label}.npy"
        checkpoint = out/f"{label}.pt"
        if output.exists():
            net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state"])
            continue
        end = stops[i+1] if i+1 < len(stops) else len(days)
        cutoff = pd.Timestamp(origin, tz=days.tz)-pd.Timedelta(hours=7)
        q = (price.smp*price.k).where(price.index <= cutoff, 0).to_numpy().reshape(-1, 24)
        positive = q > 0
        q = np.where(positive, (q/q[positive].mean()).clip(.05, 4), 0).astype("float32")
        good_days = np.flatnonzero((np.isfinite(y[:stop]) & positive[:stop, :, None]).sum((1, 2)) >= 24)
        optimizer = torch.optim.AdamW(net.parameters(), lr=.0015 if i == 0 else .00045,
                                      weight_decay=.0005)
        fresh = good_days[good_days >= stops[i-1]-1] if i else good_days
        historical = good_days[good_days < stops[i-1]-1] if i else good_days[:0]
        for epoch in range(3 if i == 0 else 2):
            net.train()
            replay = rng.choice(historical, min(len(historical), 128), replace=False)
            order = rng.permutation(np.r_[fresh, replay])
            losses = []
            for batch in np.array_split(order, (len(order)+3)//4):
                prediction = predict(x[batch])
                loss = objective(net, prediction, torch.from_numpy(y[batch].astype("float32")),
                                 torch.from_numpy(q[batch]), .35 if i or epoch else .15)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.)
                optimizer.step()
                losses.append(loss.item())
            print(f"SolarJoint {label} epoch {epoch+1}, {len(order)} days, loss={np.mean(losses):.5f}",
                  flush=True)
        net.eval()
        future = eval_x if label == "evaluation" else x[stop:end]
        with torch.inference_mode():
            result = np.concatenate([predict(batch).numpy() for batch in
                                     np.array_split(future, (len(future)+3)//4)])*capacity
        if result.shape != future.shape[:-1] or not np.isfinite(result).all():
            raise ValueError("Invalid SolarJoint forecast")
        torch.save(dict(state=net.state_dict(), cutoff=str(cutoff), mean=mean.tolist(),
                        scale=scale.tolist(), features=[feature_names[j] for j in use]), checkpoint)
        partial = output.with_suffix(".partial.npy")
        np.save(partial, result.astype("float32"))
        partial.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "artifacts", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--threads", default=4, type=int)
    args = parser.parse_args()
    destination = safe_output(args.raw, args.out)
    if destination == args.artifacts.resolve():
        raise ValueError("Use a separate SolarJoint output directory")
    train(args.raw, args.artifacts, destination, args.threads)
