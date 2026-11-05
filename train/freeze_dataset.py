"""Export verified eligible rows as the canonical ready-to-fit dataset."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from audit_data import digest, save_json


def fingerprint(path, *, eligible_only=False):
    dataset = ds.dataset(path, format="parquet")
    scanner = dataset.scanner(columns=["site_id", "ts", "energy_kwh", "target_eligible"],
        filter=(ds.field("target_eligible") == True) if eligible_only else None,
        batch_size=65536)
    h = hashlib.sha256()
    rows = zeros = 0
    for b in scanner.to_batches():
        f = b.to_pandas()
        assert f.target_eligible.all()
        assert np.isfinite(f.energy_kwh).all() and f.energy_kwh.ge(0).all()
        h.update(pd.util.hash_pandas_object(f[["site_id", "ts", "energy_kwh"]], index=False,
                                           categorize=False).to_numpy(dtype="uint64").tobytes())
        rows += len(f)
        zeros += int(f.energy_kwh.eq(0).sum())
    return dict(rows=rows, zero_targets=zeros, keys_targets_fingerprint=h.hexdigest())


def freeze(out, publish=None):
    out = Path(out)
    m = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    v = json.loads((out/"verification.json").read_text(encoding="utf-8"))
    if m["status"] != "verified_frozen" or v["status"] != "PASS":
        raise ValueError("Full dataset must pass independent verification first")
    src = out/m["summary"]["training"]["file"]
    if digest(src) != m["summary"]["training"]["sha256"]:
        raise ValueError("Full grid changed after verification")
    dest = out/"training_ready.parquet"
    if dest.exists():
        raise FileExistsError(dest)
    scanner = ds.dataset(src, format="parquet").scanner(filter=ds.field("target_eligible") == True,
                                                        batch_size=65536)
    partial = dest.with_suffix(".partial.parquet")
    with pq.ParquetWriter(partial, scanner.projected_schema, compression="zstd", compression_level=5,
                          use_dictionary=["site_id"]) as writer:
        for batch in scanner.to_batches():
            if batch.num_rows:
                writer.write_batch(batch)
    partial.replace(dest)
    expected = fingerprint(src, eligible_only=True)
    actual = fingerprint(dest)
    if actual != expected or actual["rows"] != m["summary"]["training"]["eligible"]:
        raise AssertionError("Ready rows differ from eligible full-grid rows")
    if pq.read_schema(src) != pq.read_schema(dest):
        raise AssertionError("Ready schema differs from full grid")
    m["ready_training"] = dict(file=dest.name, sha256=digest(dest), bytes=dest.stat().st_size,
                               **actual, features=len(m["features"]),
                               default_features=len(m["default_features"]), status="PASS")
    m["freeze_code_sha256"] = digest(Path(__file__))
    m["access_code_sha256"] = digest(Path(__file__).with_name("dataset.py"))
    save_json(out/"manifest.json", m)
    with (out/"quality_report.md").open("a", encoding="utf-8") as f:
        f.write("\n## 고정 학습 파일\n\ntraining_ready.parquet: 결측·음수 타깃 제외, 실제 0 유지.\n\n```json\n"
                +json.dumps(m["ready_training"], ensure_ascii=False, indent=2)+"\n```\n")
    lock = dict(version=m["version"], ready_training=m["ready_training"],
                full_grid_sha256=m["summary"]["training"]["sha256"],
                evaluation_input_sha256=m["summary"].get("evaluation", {}).get("sha256"),
                target=m["target"], key=m["key"], default_features=m["default_features"],
                all_candidate_features=m["features"], timezone=m["timezone"],
                penalty_coefficient=0.1, code_sha256=m["code_sha256"],
                freeze_code_sha256=m["freeze_code_sha256"],
                access_code_sha256=m["access_code_sha256"],
                status="local dataset verified; no performance or official approval claim")
    save_json(out/"dataset_lock.json", lock)
    if publish:
        publish = Path(publish)
        save_json(publish/"dataset_lock.json", lock)
        save_json(publish/".dataset.local.json", {"directory":str(out.resolve()),
                                                 "ready_sha256":m["ready_training"]["sha256"]})
        for name in ("feature_dictionary.csv", "validation_splits.csv", "feature_ablation.csv"):
            (publish/name).write_bytes((out/name).read_bytes())
    print(json.dumps(m["ready_training"], ensure_ascii=False, indent=2), flush=True)
    return m["ready_training"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--publish-code", type=Path)
    a = p.parse_args()
    freeze(a.out, a.publish_code)
