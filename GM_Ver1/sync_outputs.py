"""Copy completed model artifacts into GM_Ver1/ without input caches or labels."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3

from GM_Ver1.common import write_json


def sync(out, destination):
    status = {}
    for family in ["XGBoost", "GRU", "TCN"]:
        src, dst = out / family, destination / family
        dst.mkdir(parents=True, exist_ok=True)
        for path in src.glob("*.json"):
            shutil.copy2(path, dst / path.name)
        if (src / "selected.json").exists() and (src / "study.sqlite3").exists():
            with sqlite3.connect(src / "study.sqlite3") as a, sqlite3.connect(dst / "study.sqlite3") as b:
                a.backup(b)
        status[family] = {}
        for version in ["Ver1", "Ver2"]:
            local, public = src / version, dst / version
            public.mkdir(parents=True, exist_ok=True)
            for name in ["config.json", "metrics.json", "complete.json"]:
                if (local / name).exists():
                    shutil.copy2(local / name, public / name)
            for stage in ["confirm_2024", "check_2025", "final"]:
                folder = local / stage
                if (folder / "fit.json").exists() and (folder / "predictions.json").exists():
                    shutil.copytree(folder, public / stage, dirs_exist_ok=True)
            status[family][version] = "complete" if (local / "complete.json").exists() else ("in_progress" if local.exists() else "pending")
    write_json(destination / "run_status.json", status)
    if (out / "environment.json").exists():
        shutil.copy2(out / "environment.json", destination / "environment.json")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--destination", type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    sync(args.out, args.destination)
