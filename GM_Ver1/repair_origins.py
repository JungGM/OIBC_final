"""Apply the explicit HPO availability embargo to saved chronological forecasts."""
import argparse
import json
from pathlib import Path
import pandas as pd
from GM_Ver1.common import sha256, write_json
from GM_Ver1.run import SELECTION_LABEL_END


def repair(out):
    for family in ["XGBoost", "GRU", "TCN"]:
        for version in ["Ver1", "Ver2"]:
            root = out / family / version
            if not (root / "complete.json").exists():
                raise ValueError("Wait until each base candidate is complete")
            for stage in ["confirm_2024", "check_2025", "final"]:
                path = root / stage / "predictions.parquet"
                sidecar = path.with_suffix(".json")
                meta = json.loads(sidecar.read_text())
                if meta.get("selection_labels_end") == SELECTION_LABEL_END:
                    continue
                if sha256(path) != meta["sha256"]:
                    raise ValueError("Saved forecast hash mismatch")
                if stage == "confirm_2024":
                    p = pd.read_parquet(path)
                    left = pd.Timestamp("2024-01-02 01:00:00+09:00")
                    p = p.loc[p.ts >= left].reset_index(drop=True)
                    assert len(p) == 2_216_280
                    p.to_parquet(path, compression="zstd", index=False)
                    meta.update(rows=len(p), sha256=sha256(path), first_prediction=str(p.ts.min()),
                        embargo_reason="HPO validation labels available through 2024-01-01 00:00; exclude Jan 1 operating day")
                meta["selection_labels_end"] = SELECTION_LABEL_END
                write_json(sidecar, meta)
                fit_path = root / stage / "fit.json"
                fit = json.loads(fit_path.read_text())
                fit["selection_labels_end"] = SELECTION_LABEL_END
                write_json(fit_path, fit)
            config_path = root / "config.json"
            config = json.loads(config_path.read_text())
            config["selection_labels_end"] = SELECTION_LABEL_END
            config["confirmation_operating_days"] = ["2024-01-02", "2024-12-31"]
            write_json(config_path, config)
            complete_path = root / "complete.json"
            complete = json.loads(complete_path.read_text())
            complete["config_sha256"] = sha256(config_path)
            write_json(complete_path, complete)
            print(f"{family}/{version}: selection-origin guard applied", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    repair(p.parse_args().out)
