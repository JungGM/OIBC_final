"""Load an exported candidate and reproduce its evaluation forecasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

from train.dataset import current_dataset
from GM_Ver1.common import local_output, prepare_panel
from GM_Ver1.gru import ForecastGRU
from GM_Ver1.tcn import ForecastTCN


def load_candidate(artifact_root, family, version, stage="final"):
    root = Path(artifact_root) / family / version
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))["config"]
    if family == "XGBoost":
        model = xgb.Booster()
        model.load_model(root / stage / "model.ubj")
        model.set_param({"nthread": 4})
        return model, None
    saved = torch.load(root / stage / "model.pt", map_location="cpu", weights_only=True)
    if cfg != saved["config"]:
        raise ValueError("Exported config does not match checkpoint")
    torch.set_num_threads(4)
    cls = ForecastGRU if family == "GRU" else ForecastTCN
    model = cls(len(saved["features"]), cfg, len(saved["site_ids"]))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved["scaler"]


def candidate_main(family, version):
    parser = argparse.ArgumentParser(description=f"Reproduce {family}/{version} forecasts")
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    local_output(args.output.parent)
    panel = prepare_panel(args.dataset or current_dataset(), args.output.parent / "cache_evaluation", evaluation=True)
    expected = json.loads((args.artifact_root / family / version / "config.json").read_text())["features"]
    if panel.features != expected:
        raise ValueError("Feature order changed")
    model, scaler = load_candidate(args.artifact_root, family, version)
    from GM_Ver1.run import FINAL_END, predict_to_file
    predict_to_file(family, model, scaler, panel, np.arange(len(panel.X)), args.output,
                    fit_end=FINAL_END, role="reproduced competition evaluation forecast")
    print(args.output)
