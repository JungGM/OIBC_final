"""Refit the selected recipe on all available eligible source labels and export."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .architecture import EXPERTS, FAMILIES, MONTHS, Experiment
from .decisions import group_sum
from .experts import forecasts
from .fit import dump_json, tree_predict
from .run import as_composition
from .settlement import group_settlement, price_arrays
from .submission import validate_files


def score_scenarios(exp, selected_point, selected_membership, baseline_point, baseline_membership, new_site):
    import joblib
    _, part = exp.bounds("may")
    y = exp.store.y[part]
    old_site_may = np.load(exp.base/"may/lgbm_full.npy")
    new_site_may = exp.panel["may"][1].get(selected_point[0], exp.panel["may"][1]["economic_gate"])
    old = joblib.load(exp.base/"may/lgbm_full.joblib")
    old_site = tree_predict(old, exp.store.eval_x, exp.cap, exp.threads)
    _, smp, k = price_arrays(exp.store.eval_prices)
    rng = np.random.default_rng(20261106)
    if len(y) % 24 or not (np.isfinite(y) & (y >= 0)).all():
        raise ValueError("Expected complete May days for the settlement scenarios")
    blocks = rng.integers(len(y)//24-2, size=(300, (len(smp)//24+2)//3))
    ids = (24*blocks[..., None]+np.arange(72)).reshape(len(blocks), -1)[:, :len(smp)]
    member = np.stack([baseline_membership, ~baseline_membership,
                       selected_membership, ~selected_membership], axis=-1).astype("float32")
    pred = np.column_stack([baseline_point, selected_point[1]])
    result = {}
    for name, center, before in (("frozen_lightgbm", old_site, old_site_may),
                                  ("new_architecture", new_site, new_site_may)):
        res = y-before
        sims = []
        for draw in ids:
            actual = np.maximum(center+res[draw], 0.) @ member
            revenue = smp @ actual[:, :2].sum(-1)
            loss = ((.1*smp*k) @ np.abs(pred-actual)).reshape(2, 2).sum(-1)
            sims.append([revenue, loss[0], loss[1]])
        sims = np.asarray(sims)
        net = sims[:, 0]-sims[:, 2]
        gain = sims[:, 1]-sims[:, 2]
        result[name] = dict(draws=len(sims), mean_gross_krw=float(sims[:, 0].mean()),
                            mean_settlement_krw=float(net.mean()), p10_krw=float(np.percentile(net, 10)),
                            p90_krw=float(np.percentile(net, 90)),
                            expected_gain_vs_frozen_krw=float(gain.mean()),
                            p10_gain_krw=float(np.percentile(gain, 10)),
                            p90_gain_krw=float(np.percentile(gain, 90)),
                            at_or_above_64_65=int((net >= 6_465_000_000).sum()))
    return result


def refit(raw, base, trained, decisions, out, threads):
    selection = json.loads((decisions/"selection.json").read_text())
    chosen = selection["recipe"]
    exp = Experiment(raw, base, trained, out, threads)
    for fold in ("warmup", *MONTHS):
        x, _, _ = exp.level(fold)
        with np.load(decisions/fold/"oof.npz") as archive:
            exp.panel[fold] = x, {name: archive[name] for name in archive.files}
    experts = out/"refitted_experts"
    experts.mkdir(exist_ok=True)
    torch.set_num_threads(threads)
    tuning = json.loads((trained/"tuning.json").read_text())
    families = (chosen,) if chosen in FAMILIES else (() if chosen == "baseline" else FAMILIES)
    if families:
        forecasts(exp.store, tuning, experts, exp.store.y.size, exp.store.y.size,
                  ["evaluation"], families=families)
    exp.full_experts = experts
    # Keep the full-history LGB baseline fitted by the original architecture run.
    for name in ("baseline_evaluation.npy", "baseline_evaluation.json"):
        if (decisions/name).exists() and not (out/name).exists():
            shutil.copyfile(decisions/name, out/name)
    if chosen in EXPERTS:
        index = EXPERTS.index(chosen)
        all_sites, _, _ = exp.level("evaluation")
        site = all_sites[..., index]
        first = exp.reference
        point = group_sum(site, first)
        metadata = dict(mode="single_model")
    else:
        recipes, metadata, site_recipes = exp.fold("evaluation")
        point, first = recipes[chosen]
        site = site_recipes.get(chosen, site_recipes["economic_gate"])
    times = exp.store.eval_prices.index
    folder = out/"submission"
    folder.mkdir(exist_ok=True)
    as_composition(exp.store.sites, first).to_csv(folder/"composition.csv", index=False)
    pd.DataFrame(dict(group=np.repeat(["G1", "G2"], len(times)), ts=np.tile(times.astype(str), 2),
                      forecast_kwh=point.T.reshape(-1))).to_csv(folder/"forecast.csv", index=False)
    checks = validate_files(exp.store.sites, times, folder)
    np.save(out/"selected_site.npy", site.astype("float32"))
    with np.load(base/"validation_nested/frozen.npz") as baseline:
        scenarios = score_scenarios(exp, (chosen, point), first, baseline["forecast_eval"],
                                    baseline["first"], site)
    refit_metadata = {name: json.loads((experts/"evaluation"/f"{name}.json").read_text())
                      for name in families}
    if chosen == "baseline":
        refit_metadata[chosen] = json.loads((out/"baseline_evaluation.json").read_text())
    dump_json(out/"report.json", dict(selected=chosen, validation=checks, full_refit=refit_metadata,
                                     scenario=scenarios, metadata=metadata,
                                     official_score=None, threshold_krw=6_465_000_000))
    print(json.dumps(dict(selected=chosen, validation=checks, scenario=scenarios), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "base", "trained", "decisions", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    out = args.out.resolve()
    if (out.is_relative_to(Path(__file__).resolve().parents[1]) or out.is_relative_to(args.raw.resolve())
            or out in (args.base.resolve(), args.decisions.resolve(), args.trained.resolve())):
        raise ValueError("Use a distinct local full-refit output directory")
    out.mkdir(parents=True, exist_ok=True)
    refit(args.raw, args.base, args.trained, args.decisions, out, args.threads)
