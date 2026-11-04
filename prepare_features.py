"""Materialize the shared forecast-only feature table outside GitHub."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import add_features, feature_names


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True,
                        help="Existing OIBC backtest cache with features.json and features/YYYY-MM.parquet")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2022-01")
    parser.add_argument("--end", default="2025-05")
    args = parser.parse_args()
    source, out = args.cache.resolve(), args.out.resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("Use a separate output directory")
    if "onedrive" in str(out).lower() or "github" in str(out).lower():
        raise ValueError("Keep row-level feature tables outside OneDrive and GitHub")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty")
    base = json.loads((source / "features.json").read_text(encoding="utf8"))["enhanced"]
    names = feature_names(base)
    periods = pd.period_range(args.start, args.end, freq="M")
    if not len(periods):
        raise ValueError("Empty period")
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for month in periods:
        path = source / "features" / f"{month}.parquet"
        grid = pd.read_parquet(path, columns=["site_id", "ts"] + base)
        grid = grid.sort_values(["site_id", "ts"]).reset_index(drop=True)
        improved = add_features(grid, base)
        if improved.columns.tolist() != ["site_id", "ts"] + names:
            raise ValueError("Unexpected feature columns")
        improved.to_parquet(out / f"{month}.parquet", index=False, compression="zstd")
        summary.append({"month": str(month), "rows": len(improved),
                        "sites": int(improved.site_id.nunique()),
                        "feature_count": len(names),
                        "derived_missing_cells": int(improved[list(names[len(base):])].isna().sum().sum()),
                        "source_bytes": path.stat().st_size,
                        "source_mtime_ns": path.stat().st_mtime_ns})
        print(f"{month}: {len(improved):,} rows, {len(names)} features", flush=True)
    manifest = {"version": 1, "source_cache": str(source),
                "feature_code_sha256": digest(Path(__file__).with_name("feature_engineering.py")),
                "features": names, "base_count": len(base), "derived_count": len(names) - len(base),
                "months": summary, "total_rows": int(sum(v["rows"] for v in summary)),
                "note": "Forecast and supplied metadata only; no generation labels, actual weather, or SMP in X."}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({k: manifest[k] for k in ("base_count", "derived_count", "total_rows")}), flush=True)


if __name__ == "__main__":
    main()
