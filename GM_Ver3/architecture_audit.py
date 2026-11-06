"""Summarize stage gains and transfer fixed VPP assignments to later months."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .architecture import FAMILIES, MONTHS, PORTFOLIOS
from .decisions import group_sum
from .fit import dump_json, input_files, start_at
from .nested import NESTED_FOLDS
from .settlement import group_settlement, price_arrays


def audit(raw, base, decision):
    report = json.loads((decision/"report.json").read_text())
    ranks = json.loads((decision/"selection.json").read_text())["penalty_jan_apr_krw"]
    stages = dict(M1="baseline", M2=min(FAMILIES, key=ranks.get), M3="weighted",
                  M4=min(("lgb_stack", "xgb_stack"), key=ranks.get), M5="mlp_gate",
                  M6="economic_gate", M7="vpp_ga_local", M8="direct_mix",
                  M9=min(("economic_median", "economic_residual"), key=ranks.get))
    stage_table = {}
    for stage, recipe in stages.items():
        row = {fold: report["records"][fold]["scores"][recipe]["penalty_krw"] for fold in MONTHS}
        stage_table[stage] = dict(recipe=recipe, jan_apr_krw=ranks[recipe],
                                  relative_reduction=1-ranks[recipe]/ranks["baseline"], monthly=row)
    sites, prices, _ = input_files(raw)
    times, smp, k = price_arrays(prices)
    actual = np.load(base/"train_y.npy", mmap_mode="r")
    transfer = {}
    for left, origin in enumerate(MONTHS[:-1]):
        with np.load(decision/origin/"groups.npz") as data:
            choices = {key: data[f"vpp_{key}"] for key in PORTFOLIOS}
        for later in MONTHS[left+1:]:
            t = start_at(NESTED_FOLDS[later])
            start = times.searchsorted(t)
            end = times.searchsorted(t.normalize()+pd.offsets.MonthBegin()+pd.Timedelta(hours=1))
            y = actual[start:end]
            valid = (np.isfinite(y) & (y >= 0)).all(axis=1)
            with np.load(decision/later/"oof.npz") as data:
                p = data["economic_gate"][valid]
            transfer[f"{origin}_to_{later}"] = {key: group_settlement(
                group_sum(y[valid], first), group_sum(p, first), smp[start:end][valid],
                k[start:end][valid])["penalty_krw"] for key, first in choices.items()}
    dump_json(decision/"architecture_audit.json", dict(stages=stage_table, transfer=transfer,
              sites=len(sites), selected=report["selection"]))
    print(json.dumps(stage_table, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "base", "decision"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    audit(args.raw, args.base, args.decision)
