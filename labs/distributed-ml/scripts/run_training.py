#!/usr/bin/env python3
"""Train the Stage 1 baselines and the Stage 2 worker-count grid.

Every run records wall time, worker startup time, CPU layout, model hash,
holdout Logloss/AUC from the same CLI, and the model's CTR structure.

Distributed runs use the stock CatBoost protocol: one `run-worker` process per
worker and one master process reading a hosts file.  All processes are on this
host and talk over 127.0.0.1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Built from the repository root (cmake --build build); override with CATBOOST_BIN.
CATBOOST = Path(os.environ.get('CATBOOST_BIN', ROOT.parents[1] / 'build/Release/catboost/app/catboost'))
STRUCTURE = ROOT / 'scripts/structure.py'
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
SEED = 20260928
BASE_ITERS = 200
DEPTH = 6
LR = 0.1

DATASETS = {
    'amazon': {
        'full': {'train': DATA / 'amazon/train.tsv', 'holdout': DATA / 'amazon/holdout.tsv',
                 'cd': DATA / 'amazon/train.cd'},
        'onehot': {'train': DATA / 'amazon/train.tsv', 'holdout': DATA / 'amazon/holdout.tsv',
                   'cd': DATA / 'amazon/train.cd',
                   'extra': ['--one-hot-max-size', '65535', '--max-ctr-complexity', '1']},
        'numeric': None,
    },
    'kdd': {
        'full': {'train': DATA / 'kdd/train.tsv', 'holdout': DATA / 'kdd/holdout.tsv',
                 'cd': DATA / 'kdd/train.cd'},
        'onehot': None,
        'numeric': {'train': DATA / 'kdd/train_num.tsv', 'holdout': DATA / 'kdd/holdout_num.tsv',
                    'cd': DATA / 'kdd/num.cd'},
    },
    'criteo': {
        'full': {'train': DATA / 'criteo/train.tsv', 'holdout': DATA / 'criteo/holdout.tsv',
                 'cd': DATA / 'criteo/train.cd'},
        'onehot': None,
        'numeric': {'train': DATA / 'criteo/train_num.tsv', 'holdout': DATA / 'criteo/holdout_num.tsv',
                    'cd': DATA / 'criteo/num.cd'},
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) != 0


def wait_port(port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex(('127.0.0.1', port)) == 0:
                return True
        time.sleep(0.1)
    return False


def parse_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = list(csv.reader(path.open(), delimiter='\t'))
    if len(rows) < 2:
        return {}
    header = rows[0]
    last = rows[-1]
    out = {}
    for key, value in zip(header, last):
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def run_eval(model: Path, spec: dict, out_dir: Path) -> dict:
    metrics_path = out_dir / 'metrics.tsv'
    cmd = [str(CATBOOST), 'eval-metrics',
           '--model-file', str(model),
           '--input-path', str(spec['holdout']),
           '--column-description', str(spec['cd']),
           '--metrics', 'Logloss,AUC',
           '--thread-count', '8',
           '--output-path', str(metrics_path)]
    with (out_dir / 'eval.log').open('w') as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=out_dir)
    out = parse_metrics(metrics_path)
    out['eval_exit_code'] = proc.returncode
    return out


def run_structure(model: Path, out_dir: Path, keep_json: bool) -> dict:
    cmd = [sys.executable, str(STRUCTURE), str(model), '--out', str(out_dir / 'structure.json')]
    if keep_json:
        cmd.append('--keep-json')
    with (out_dir / 'structure.log').open('w') as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    path = out_dir / 'structure.json'
    if proc.returncode == 0 and path.exists():
        return json.loads(path.read_text())[0]
    return {'error': f'structure exit {proc.returncode}', 'log': str(out_dir / 'structure.log')}


def train_one(name: str, spec: dict, out_dir: Path, *, mode: str, threads: int,
              iterations: int, extra: list[str], workers: int = 0,
              base_port: int = 22000) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dir = out_dir / 'train'
    model = out_dir / 'model.cbm'
    worker_logs = []
    worker_procs = []
    total_start = time.monotonic()

    cmd = [str(CATBOOST), 'fit',
           '--learn-set', str(spec['train']),
           '--column-description', str(spec['cd']),
           '--loss-function', 'Logloss',
           '--iterations', str(iterations),
           '--depth', str(DEPTH),
           '--learning-rate', str(LR),
           '--random-seed', str(SEED),
           '--model-file', str(model),
           '--train-dir', str(train_dir),
           '--allow-writing-files', 'false',
           '--verbose', '20'] + extra

    startup_start = time.monotonic()
    if mode == 'dist':
        ports = []
        port = base_port
        for _ in range(workers):
            while not port_free(port):
                port += 1
            ports.append(port)
            port += 1
        hosts = out_dir / 'hosts.txt'
        hosts.write_text('\n'.join(f'127.0.0.1:{p}' for p in ports) + '\n')
        for i, p in enumerate(ports):
            log_path = out_dir / f'worker-{i}.log'
            worker_logs.append(log_path)
            worker_procs.append(subprocess.Popen(
                [str(CATBOOST), 'run-worker', '--node-port', str(p), '-T', str(threads)],
                stdout=log_path.open('w'), stderr=subprocess.STDOUT,
                start_new_session=True))
        ready = all(wait_port(p, 30.0) for p in ports)
        if not ready:
            for worker in worker_procs:
                worker.kill()
            proc = subprocess.CompletedProcess(cmd, 125)
            fit_s = 0.0
            startup_s = time.monotonic() - startup_start
            total_s = time.monotonic() - total_start
            record = {
                'name': name, 'mode': mode, 'workers': workers,
                'worker_threads': threads, 'threads': threads,
                'iterations': iterations, 'extra_args': extra,
                'fit_wall_s': round(fit_s, 3), 'startup_s': round(startup_s, 3),
                'total_wall_s': round(total_s, 3), 'exit_code': proc.returncode,
                'model': None, 'model_sha256': None, 'model_bytes': None,
                'fit_log': str(out_dir / 'fit.log'),
                'worker_logs': [str(p) for p in worker_logs],
                'failure_tail': {'workers': f'did not open ports: {ports}'},
            }
            (out_dir / 'record.json').write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + '\n')
            return record
        cmd += ['--node-type', 'Master', '--file-with-hosts', str(hosts),
                '--thread-count', '2']
    else:
        cmd += ['--thread-count', str(threads)]

    startup_s = time.monotonic() - startup_start
    fit_start = time.monotonic()
    with (out_dir / 'fit.log').open('w') as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    fit_s = time.monotonic() - fit_start

    if mode == 'dist':
        for p in worker_procs:
            if p.poll() is None:
                p.terminate()
        for p in worker_procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)
        for log_path in worker_logs:
            try:
                log_path.open('a').close()
            except OSError:
                pass

    total_s = time.monotonic() - total_start
    record = {
        'name': name,
        'mode': mode,
        'workers': workers,
        'worker_threads': threads if mode == 'dist' else None,
        'threads': threads,
        'iterations': iterations,
        'extra_args': extra,
        'train_rows': sum(1 for _ in spec['train'].open()),
        'holdout_rows': sum(1 for _ in spec['holdout'].open()),
        'fit_wall_s': round(fit_s, 3),
        'startup_s': round(startup_s, 3),
        'total_wall_s': round(total_s, 3),
        'exit_code': proc.returncode,
        'model': str(model) if model.exists() else None,
        'model_sha256': sha256(model) if model.exists() else None,
        'model_bytes': model.stat().st_size if model.exists() else None,
        'fit_log': str(out_dir / 'fit.log'),
        'worker_logs': [str(p) for p in worker_logs],
    }
    if proc.returncode == 0 and model.exists():
        record['quality'] = run_eval(model, spec, out_dir)
        record['structure'] = run_structure(model, out_dir, keep_json=not name.endswith(('_num', '_onehot')))
    else:
        tails = {}
        for log_path in [out_dir / 'fit.log'] + worker_logs:
            if log_path.exists():
                tails[log_path.name] = log_path.read_text(errors='replace').splitlines()[-12:]
        record['failure_tail'] = tails
    (out_dir / 'record.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    return record


def stage1() -> list[dict]:
    out = []
    out_dir = RESULTS / 'stage1'
    for ds in ('amazon', 'kdd', 'criteo'):
        out.append(train_one(f'{ds}_ctr', DATASETS[ds]['full'], out_dir / f'{ds}_ctr',
                             mode='single', threads=8, iterations=BASE_ITERS, extra=[]))
        if ds == 'amazon':
            spec = DATASETS[ds]['onehot']
            extra = spec['extra']
            name = 'amazon_onehot'
        else:
            spec = DATASETS[ds]['numeric']
            extra = []
            name = f'{ds}_num'
        out.append(train_one(name, spec, out_dir / name,
                             mode='single', threads=8, iterations=BASE_ITERS, extra=extra))
    return out


def stage2() -> list[dict]:
    out = []
    out_dir = RESULTS / 'stage2'
    choices = {'amazon': ('onehot', 'amazon_onehot'), 'kdd': ('numeric', 'kdd_num'),
               'criteo': ('numeric', 'criteo_num')}
    for ds, (kind, label) in choices.items():
        spec = DATASETS[ds][kind]
        extra = spec.get('extra', []) if kind == 'onehot' else []
        for tag, mode, threads, workers in (
            ('single3', 'single', 3, 0),
            ('single12', 'single', 12, 0),
            ('dist1', 'dist', 3, 1),
            ('dist2', 'dist', 3, 2),
            ('dist4', 'dist', 3, 4),
        ):
            out.append(train_one(f'{label}_{tag}', spec, out_dir / f'{label}_{tag}',
                                 mode=mode, threads=threads, iterations=BASE_ITERS,
                                 extra=extra, workers=workers, base_port=23000))
    # Reproduce the blocker for CTR models in the stock distributed CPU mode.
    for ds in ('amazon', 'kdd', 'criteo'):
        out.append(train_one(f'{ds}_ctr_dist2_failure', DATASETS[ds]['full'],
                             out_dir / f'{ds}_ctr_dist2_failure',
                             mode='dist', threads=3, iterations=10, extra=[],
                             workers=2, base_port=24000))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('stage1', 'stage2'))
    args = ap.parse_args()
    fn = stage1 if args.stage == 'stage1' else stage2
    records = fn()
    path = RESULTS / args.stage / 'records.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(records, indent=1, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
