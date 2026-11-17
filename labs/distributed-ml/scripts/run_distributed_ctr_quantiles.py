#!/usr/bin/env python3
"""Run and summarize the frozen 2026-11-07 paired CTR latency matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = Path(os.environ.get("CTR_LAB_INPUT_ROOT", str(ROOT))).expanduser().resolve()
RUN = ROOT / "results/stage3/distributed_ctr_quantiles_20261107"
BIN = Path(os.environ.get("CTR_BENCH_BIN_DIR", str(ROOT / "bin"))).expanduser().resolve()
AMAZON = INPUT_ROOT / "results/stage2/amazon_ctr_spark_mapped_20261107"
CRITEO = INPUT_ROOT / "results/stage2/criteo_x1_ctr_spark2_20261107"
CRITEO_POOL = Path(os.environ.get("CTR_CRITEO_POOL", str(ROOT / "tmp/criteo_x1_holdout_4cat_20261107.dsv")))
CRITEO_POOL_SHA256 = "4c58bae2c4fc1dfec20455d414fda613b5a5a921e81895196576024f233071f4"
MODELS = {
    **{
        f"amazon_ctr_{workers}w": (
            AMAZON / f"{workers}w/model.cbm",
            AMAZON / f"{workers}w/holdout_indexed.tsv",
            AMAZON / f"{workers}w/holdout_indexed.cd",
            8192, "ctr", workers,
        )
        for workers in (1, 2, 4)
    },
    "criteo_ctr_2w": (
        CRITEO / "model.cbm",
        CRITEO_POOL,
        CRITEO / "selected.cd",
        2_000_000, "ctr", 2,
    ),
    "amazon_noctr_2w": (
        INPUT_ROOT / "results/stage2/amazon_onehot_dist2/model.cbm",
        INPUT_ROOT / "data/amazon/holdout.tsv",
        INPUT_ROOT / "data/amazon/train.cd",
        8192, "no_ctr", 2,
    ),
}
ARMS = ("stock", "production")
BATCHES = (128, 4096)
SESSIONS = range(1, 10)
QUANTILES = (50, 90, 95, 99)
THREADS = 2
WARMUP = 50
REPETITIONS = 1000
DRAWS = 20_000
SEED = 20261107


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_criteo_pool() -> None:
    """Recreate the selected four-category holdout from saved source data."""
    if not CRITEO_POOL.is_file():
        source = INPUT_ROOT / "data/criteo_x1/holdout_ctr.dsv"
        if not source.is_file():
            raise FileNotFoundError(source)
        temporary = CRITEO_POOL.with_suffix(".tmp")
        with temporary.open("wb") as target:
            subprocess.run(["cut", "-f1-18", str(source)], stdout=target, check=True)
        temporary.replace(CRITEO_POOL)
    if sha256(CRITEO_POOL) != CRITEO_POOL_SHA256:
        raise RuntimeError("Criteo selected holdout SHA-256 differs from frozen input")


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def command(model: dict, arm: str, batch: int, warmup: int,
            repetitions: int, raw: Path | None = None,
            full_dataset: bool = False) -> dict:
    args = [str(BIN / arm), "--model", model["model"]["path"],
            "--pool", model["pool"]["path"], "--cd", model["cd"]["path"],
            "--block-size", str(batch), "--threads", str(THREADS),
            "--warmup", str(warmup), "--repetitions", str(repetitions)]
    if raw is not None:
        args += ["--raw-out", str(raw)]
    if full_dataset:
        args.append("--full-dataset")
    proc = subprocess.run(args, cwd=INPUT_ROOT, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"{arm} exit={proc.returncode}: {proc.stderr[-2000:]} {proc.stdout[-1000:]}")
    start = proc.stdout.find("{\n")
    if start < 0:
        raise RuntimeError(f"{arm}: JSON output not found: {proc.stdout[-1000:]}")
    result = json.loads(proc.stdout[start:])
    if result.get("pool_rows") != model["rows"]:
        raise RuntimeError(f"{arm}: expected {model['rows']} pool rows, got {result.get('pool_rows')}")
    if result.get("profile_enabled") is not False:
        raise RuntimeError(f"{arm}: profiling enabled")
    return result


def identity() -> dict:
    models = {}
    for name, (model, pool, cd, rows, role, workers) in MODELS.items():
        files = {}
        for key, path in (("model", model), ("pool", pool), ("cd", cd)):
            if not path.is_file():
                raise FileNotFoundError(path)
            files[key] = {"path": str(path.resolve()), "sha256": sha256(path),
                          "bytes": path.stat().st_size}
        models[name] = {**files, "rows": rows, "role": role, "workers": workers}
    binaries = {}
    for arm in ARMS:
        path = BIN / arm
        manifest = Path(str(path) + ".manifest.json")
        if not path.is_file() or not manifest.is_file():
            raise FileNotFoundError(path)
        binaries[arm] = {"path": str(path.resolve()), "sha256": sha256(path),
                         "manifest": str(manifest.resolve()),
                         "manifest_sha256": sha256(manifest)}
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol": "protocol.md (frozen before measurements)",
        "platform": platform.platform(), "machine": platform.machine(),
        "binaries": binaries, "models": models,
        "matrix": {"batches": list(BATCHES), "threads": THREADS,
                   "sessions": len(SESSIONS), "repetitions": REPETITIONS,
                   "warmup": WARMUP, "quantiles": list(QUANTILES),
                   "bootstrap_draws": DRAWS, "seed": SEED},
    }


def check_equivalence(inputs: dict) -> dict:
    results = {}
    for name, model in inputs["models"].items():
        output = {arm: command(model, arm, 4096, 1, 1, full_dataset=True)
                  for arm in ARMS}
        fields = ("rows", "chunks", "chunk_rows", "pred_hash_full", "chunk_hashes")
        stock = output["stock"]["full_dataset"]
        production = output["production"]["full_dataset"]
        equal = (all(stock.get(key) == production.get(key) for key in fields)
                 and stock.get("rows") == model["rows"]
                 and output["stock"]["pred_hash"] == output["production"]["pred_hash"])
        results[name] = {"passed": equal, "stock": stock, "production": production,
                         "first_block_hash_equal": output["stock"]["pred_hash"] == output["production"]["pred_hash"]}
        save(RUN / "equivalence.json", results)
        print(f"equivalence {name}: {'PASS' if equal else 'FAIL'}", flush=True)
        if not equal:
            raise RuntimeError(f"full-holdout prediction mismatch: {name}")
    return results


def raw_paths(name: str, batch: int, session: int, arm: str) -> tuple[Path, Path]:
    prefix = RUN / "raw" / name / f"batch_{batch}" / f"session_{session:02d}_{arm}"
    return prefix.with_suffix(".json"), prefix.with_suffix(".latencies.json")


def run_sessions(inputs: dict) -> None:
    for name, model in inputs["models"].items():
        for batch in BATCHES:
            for session in SESSIONS:
                order = ARMS if session % 2 else ARMS[::-1]
                for arm in order:
                    record_path, raw_path = raw_paths(name, batch, session, arm)
                    if record_path.exists() or raw_path.exists():
                        # A stopped run may be resumed, but only complete pairs are accepted.
                        if not record_path.is_file() or not raw_path.is_file():
                            raise RuntimeError(f"partial measurement: {record_path}")
                        print(f"existing {name} b{batch} s{session} {arm}", flush=True)
                        continue
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    result = command(model, arm, batch, WARMUP, REPETITIONS, raw=raw_path)
                    if result.get("repetitions") != REPETITIONS or result.get("predicted_rows") != batch:
                        raise RuntimeError(f"wrong workload: {name} b{batch} s{session} {arm}")
                    raw = json.loads(raw_path.read_text())
                    values = np.asarray(raw.get("per_rep_ms"), dtype=float)
                    if raw.get("repetitions") != REPETITIONS or values.shape != (REPETITIONS,) or not np.isfinite(values).all() or (values <= 0).any():
                        raise RuntimeError(f"invalid raw latencies: {raw_path}")
                    save(record_path, {"model": name, "batch": batch, "session": session,
                                       "arm": arm, "order": list(order),
                                       "model_sha256": model["model"]["sha256"],
                                       "pool_sha256": model["pool"]["sha256"],
                                       "cd_sha256": model["cd"]["sha256"],
                                       "binary_sha256": inputs["binaries"][arm]["sha256"],
                                       "raw_sha256": sha256(raw_path), "result": result})
                    print(f"measured {name} b{batch} s{session} {arm}", flush=True)


def summarize(inputs: dict, equivalence: dict) -> dict:
    if set(equivalence) != set(MODELS) or not all(x["passed"] for x in equivalence.values()):
        raise RuntimeError("equivalence matrix incomplete")
    rng = np.random.default_rng(SEED)
    cells = []
    for name, model in inputs["models"].items():
        for batch in BATCHES:
            session_quantiles = {arm: {q: [] for q in QUANTILES} for arm in ARMS}
            hashes = []
            for session in SESSIONS:
                pair = {}
                for arm in ARMS:
                    record_path, raw_path = raw_paths(name, batch, session, arm)
                    if not record_path.is_file() or not raw_path.is_file():
                        raise RuntimeError(f"missing measurement: {record_path}")
                    record, raw = json.loads(record_path.read_text()), json.loads(raw_path.read_text())
                    required = {"model": name, "batch": batch, "session": session,
                                "arm": arm, "model_sha256": model["model"]["sha256"],
                                "pool_sha256": model["pool"]["sha256"],
                                "cd_sha256": model["cd"]["sha256"],
                                "binary_sha256": inputs["binaries"][arm]["sha256"],
                                "raw_sha256": sha256(raw_path)}
                    if any(record.get(key) != value for key, value in required.items()):
                        raise RuntimeError(f"measurement identity mismatch: {record_path}")
                    values = np.asarray(raw["per_rep_ms"], dtype=float)
                    if raw.get("repetitions") != REPETITIONS or values.shape != (REPETITIONS,) or not np.isfinite(values).all() or (values <= 0).any():
                        raise RuntimeError(f"invalid raw latencies: {raw_path}")
                    result = record["result"]
                    if result.get("profile_enabled") is not False or result.get("pool_rows") != model["rows"] or result.get("predicted_rows") != batch or result.get("threads") != THREADS or result.get("repetitions") != REPETITIONS:
                        raise RuntimeError(f"wrong workload: {record_path}")
                    expected_order = list(ARMS if session % 2 else ARMS[::-1])
                    if record.get("order") != expected_order:
                        raise RuntimeError(f"wrong arm order: {record_path}")
                    for q in QUANTILES:
                        session_quantiles[arm][q].append(float(np.percentile(values, q, method="linear")))
                    pair[arm] = result["pred_hash"]
                if pair["stock"] != pair["production"]:
                    raise RuntimeError(f"prediction mismatch: {name} b{batch} s{session}")
                hashes.append(pair["stock"])
            if len(set(hashes)) != 1:
                raise RuntimeError(f"prediction hash changes between sessions: {name} b{batch}")
            cell = {"model": name, "role": model["role"], "workers": model["workers"],
                    "batch": batch, "threads": THREADS, "sessions": len(SESSIONS),
                    "pred_hash": hashes[0], "quantiles": {}}
            for q in QUANTILES:
                stock = np.asarray(session_quantiles["stock"][q])
                production = np.asarray(session_quantiles["production"][q])
                changes = 100 * (production / stock - 1)
                idx = rng.integers(0, len(changes), size=(DRAWS, len(changes)))
                boot = np.median(changes[idx], axis=1)
                ci = np.percentile(boot, [2.5, 97.5], method="linear")
                cell["quantiles"][f"p{q}"] = {
                    "stock_median_ms": float(np.median(stock)),
                    "production_median_ms": float(np.median(production)),
                    "paired_median_change_pct": float(np.median(changes)),
                    "paired_ci95_pct": [float(ci[0]), float(ci[1])],
                    "production_faster_sessions": int((changes < 0).sum()),
                    "session_changes_pct": [float(x) for x in changes],
                    "session_stock_ms": [float(x) for x in stock],
                    "session_production_ms": [float(x) for x in production],
                    "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
                }
            cells.append(cell)
            p95 = cell["quantiles"]["p95"]
            print(f"summary {name} b{batch}: p95 {p95['paired_median_change_pct']:+.2f}% {p95['paired_ci95_pct']}", flush=True)
    return {"primary": "median of paired session p95 changes", "quantiles": list(QUANTILES),
            "bootstrap_draws": DRAWS, "bootstrap_seed": SEED,
            "models": list(MODELS), "batches": list(BATCHES), "cells": cells}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if not (RUN / "protocol.txt").is_file():
        raise RuntimeError("frozen protocol missing")
    if args.summarize_only:
        inputs = json.loads((RUN / "inputs.json").read_text())
        equivalence = json.loads((RUN / "equivalence.json").read_text())
    else:
        ensure_criteo_pool()
        current = identity()
        inputs_path = RUN / "inputs.json"
        if inputs_path.exists():
            inputs = json.loads(inputs_path.read_text())
            if {key: inputs[key] for key in ("models", "binaries", "matrix")} != {key: current[key] for key in ("models", "binaries", "matrix")}:
                raise RuntimeError("inputs changed during measurement")
        else:
            save(inputs_path, current)
            inputs = current
        equivalence = check_equivalence(inputs)
        run_sessions(inputs)
    save(RUN / "summary.json", summarize(inputs, equivalence))


if __name__ == "__main__":
    main()
