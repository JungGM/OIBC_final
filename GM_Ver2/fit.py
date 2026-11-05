"""Fit each expert at strictly chronological origins and export local predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import prepare, prices
from .models import NAMES, neural, recurrent, tree


ORIGINS = ["2024-12-24", "2025-03-01", "2025-04-01", "2025-05-01", "2025-06-01"]
FOLDS = ["winter", "march", "april", "may", "evaluation"]


def fit(raw, out, budget, epochs, threads):
    if not (out/"data_summary.json").exists():
        prepare(raw, out)
    names = json.loads((out/"data_summary.json").read_text())["train"]["features"]
    sites = pd.read_csv(raw/"sites.csv").sort_values("site_id")
    capacity = sites.capacity_kw.to_numpy()
    g = sites.current_group.eq("A").to_numpy()
    groups = np.stack([g, ~g], axis=1)
    price = prices(raw/"train/smp.csv")
    days = price.index[::24].normalize()
    x = np.load(out/"train_x.npy", mmap_mode="r")
    y = np.load(out/"train_y.npy", mmap_mode="r")
    eval_x = np.load(out/"eval_x.npy", mmap_mode="r")
    for i, (label, origin) in enumerate(zip(FOLDS, ORIGINS)):
        start = pd.Timestamp(origin, tz=days.tz)
        cutoff = start - pd.Timedelta(hours=7)
        stop = days.searchsorted(start)
        end = days.searchsorted(pd.Timestamp(ORIGINS[i+1], tz=days.tz)) if i+1 < len(ORIGINS) else len(days)
        allowed = np.arange(stop)
        # The day immediately preceding the origin is only partially admissible.
        q = (price.smp * price.k).where(price.index <= cutoff, 0).to_numpy().reshape(-1, 24)
        predict_x = x[stop:end] if label != "evaluation" else eval_x
        print(f"{label}: cutoff {cutoff}; predict {len(predict_x)} days", flush=True)
        dest = out/label
        dest.mkdir(exist_ok=True)
        protocol = dict(budget=budget, epochs=epochs, cutoff=str(cutoff), days=len(predict_x))
        protocol_file = dest/"fit_protocol.json"
        if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
            raise ValueError("Use a fresh output directory for different fit settings")
        protocol_file.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        for variant, name in enumerate(NAMES):
            target = dest/f"{name}.npy"
            if target.exists():
                continue
            print(f"  Fitting {name}", flush=True)
            if variant in (0, 1, 2, 4):
                prediction = tree(x, y, q, capacity, allowed, predict_x, names, variant,
                                  budget, threads, 20261105, dest/f"{name}.ubj")
            elif variant == 3:
                prediction = neural(x, y, q, capacity, groups, allowed, predict_x, sites,
                                    names, epochs, threads, 20261105, dest/f"{name}.pt")
            else:
                prediction = recurrent(x, y, q, capacity, allowed, predict_x, names, epochs,
                                       threads, 20261105, dest/f"{name}.pt")
            prediction *= predict_x[..., names.index("solar_signed")] > -0.104528
            if prediction.shape != predict_x.shape[:-1] or not np.isfinite(prediction).all():
                raise ValueError("Invalid model outputs")
            temporary = target.with_suffix(".partial.npy")
            np.save(temporary, prediction.astype("float32"))
            temporary.replace(target)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tree-rows", type=int, default=240000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def safe_output(raw, out):
    repo = Path(__file__).resolve().parents[1]
    out = out.resolve()
    if out.is_relative_to(repo) or out.is_relative_to(raw.resolve()):
        raise ValueError("Use a local artifact directory outside the repository and raw package")
    out.mkdir(parents=True, exist_ok=True)
    return out


if __name__ == "__main__":
    args = cli()
    fit(args.raw, safe_output(args.raw, args.out), args.tree_rows, args.epochs, args.threads)
