"""Compare VPP policies on the winning site recipe without changing its forecasts."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .architecture import MONTHS
from .decisions import allocation_candidates, group_sum
from .fit import dump_json, input_files, start_at
from .nested import NESTED_FOLDS
from .settlement import group_settlement, issue_cutoff, price_arrays


POLICIES = ("reference", "balanced", "correlation", "ga", "ga_local", "robust_ga", "robust_local")


def read_when_ready(path, wait=False):
    if wait:
        deadline = time.monotonic()+25*60
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(3)
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def run(raw, base, decision, out, wait=False):
    winner = json.loads((decision/"selection.json").read_text())["recipe"]
    roster, prices, _ = input_files(raw)
    cap = roster.capacity_kw.to_numpy()
    loc = roster.lon.to_numpy()
    ref = roster.current_group.eq("A").to_numpy()
    ts, smp, k = price_arrays(prices)
    y = np.load(base/"train_y.npy", mmap_mode="r")
    folds = ["warmup", *MONTHS, "evaluation"]
    first = ts.searchsorted(start_at(NESTED_FOLDS["warmup"]))
    predictions = []
    scores = {}
    order = None
    for j, label in enumerate(folds):
        start = start_at(NESTED_FOLDS[label])
        begin = ts.searchsorted(start)
        if j:
            historical = np.concatenate(predictions, axis=0)
            stop = ts.searchsorted(issue_cutoff([start])[0], side="right")
            truth = y[first:stop]
            valid = (np.isfinite(truth) & (truth >= 0)).all(axis=1)
            err = historical[:len(truth)][valid]-truth[valid]
            q = (smp[first:stop]*k[first:stop])[valid]
            month = (ts[first:stop][valid]-pd.Timedelta(hours=1)).month
            simple = allocation_candidates(err, q, cap, loc, ref)
            robust = allocation_candidates(err, q, cap, loc, ref, month)
            maps = {"reference": ref, **simple, "robust_ga": robust["ga"], "robust_local": robust["ga_local"]}
        if label == "evaluation":
            np.savez(out/"evaluation_membership.npz", **{name: maps[name] for name in POLICIES})
            break
        record_path = out/f"{label}.json"
        saved = read_when_ready(decision/label/"oof.npz", wait)
        point = saved.get(winner, saved["economic_gate"])
        predictions.append(point)
        if not j:
            continue
        end = ts.searchsorted(start_at(NESTED_FOLDS[folds[j+1]]))
        if label == "may":
            order = sorted(POLICIES, key=lambda name: sum(block[name] for block in scores.values()))
        if record_path.exists():
            scores[label] = json.loads(record_path.read_text())
            continue
        truth = y[begin:end]
        complete = (np.isfinite(truth) & (truth >= 0)).all(axis=1)
        scores[label] = {name: group_settlement(group_sum(truth[complete], maps[name]),
                       group_sum(point[complete], maps[name]), smp[begin:end][complete],
                       k[begin:end][complete])["penalty_krw"] for name in POLICIES}
        dump_json(record_path, scores[label])
        print(f"{label} allocation comparison: {scores[label]}", flush=True)
    selection = order[0]
    # May is a regression check, not a sweep for the minimum result.
    promoted = selection if scores["may"][selection] < scores["may"]["reference"] else "reference"
    dump_json(out/"report.json", dict(site_recipe=winner, selection=selection, promoted=promoted,
                                    months=scores, ranking=order))
    print(f"Portfolio candidate after May check: {promoted}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "base", "decision", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if (out.is_relative_to(Path(__file__).resolve().parents[1]) or out.is_relative_to(args.raw.resolve())
            or out in (args.base.resolve(), args.decision.resolve())):
        raise ValueError("Use a separate local portfolio directory")
    out.mkdir(parents=True, exist_ok=True)
    run(args.raw, args.base, args.decision, out, args.wait)
