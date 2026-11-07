#!/usr/bin/env python3
"""Build one tab-separated train/holdout pair per dataset for the distributed study.

The splits are fixed before any model is trained:

* Amazon Employee Access: 75/25 deterministic shuffle with seed 20260928,
  the same seed and fraction used for KDD Cup 2009 in the previous stage.
* KDD Cup 2009 small (churn): reuse the already fixed 37 500/12 500 split.
* Criteo Click Logs sample: reuse the fixed 2015-02-16 / 2015-02-23 split.

The first column is always the target, the rest are features.  A matching CD
file marks column 0 as Target, the numeric prefix as Num and the categorical
suffix as Categ.  Labels are written as 0/1 because Logloss in the CLI expects
a binary target.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

SRW2 = Path('/Users/a.baryaev/Documents/itmo/SRW2iter')
CROSS = SRW2 / 'results/local/cross-dataset-20260928'
AMAZON_CSV = SRW2 / 'data/amazon_employee_access.csv'
SEED = 20260928


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_cd(path: Path, numeric: int, total: int) -> None:
    lines = ['0\tTarget']
    for i in range(1, total + 1):
        lines.append(f'{i}\t{"Num" if i - 1 < numeric else "Categ"}')
    path.write_text('\n'.join(lines) + '\n')


def amazon(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    with AMAZON_CSV.open(newline='') as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    target_idx = header.index('ACTION')
    order = list(range(len(body)))
    random.Random(SEED).shuffle(order)
    n_holdout = int(len(order) * 0.25)
    holdout_idx = sorted(order[:n_holdout])
    train_idx = sorted(order[n_holdout:])

    def emit(name: str, indices: list[int]) -> int:
        positives = 0
        with (out / f'{name}.tsv').open('w') as fh:
            for i in indices:
                row = body[i]
                label = 1 if row[target_idx].strip() == '1' else 0
                positives += label
                features = [v for j, v in enumerate(row) if j != target_idx]
                fh.write(str(label) + '\t' + '\t'.join(features) + '\n')
        return positives

    tr_pos = emit('train', train_idx)
    ho_pos = emit('holdout', holdout_idx)
    write_cd(out / 'train.cd', 0, len(header) - 1)
    return {
        'source': str(AMAZON_CSV),
        'split_rule': f'random.Random({SEED}).shuffle 75/25',
        'rows': {'train': len(train_idx), 'holdout': len(holdout_idx)},
        'positives': {'train': tr_pos, 'holdout': ho_pos},
        'columns': {'total': len(header) - 1, 'numeric': 0,
                    'categorical': len(header) - 1,
                    'names': [c for c in header if c != 'ACTION']},
    }


def kdd(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    src = CROSS / 'data/kdd2009'
    labels = (src / 'kdd_train.labels').read_text().split()
    val_labels = (src / 'kdd_holdout.labels').read_text().split()

    def convert(features: Path, labs: list[str], dst: Path) -> int:
        positives = 0
        with features.open() as fin, dst.open('w') as fout:
            for line, lab in zip(fin, labs):
                label = 1 if int(lab) == 1 else 0
                positives += label
                fout.write(f'{label}\t{line.rstrip(chr(10))}\n')
        return positives

    tr_pos = convert(src / 'kdd_train.dsv', labels, out / 'train.tsv')
    ho_pos = convert(src / 'kdd_holdout.dsv', val_labels, out / 'holdout.tsv')
    write_cd(out / 'train.cd', 190, 230)
    return {
        'source': 'KDD Cup 2009 small, orange_small_train',
        'split_rule': 'fixed previous split: random.Random(20260928).shuffle 75/25',
        'rows': {'train': len(labels), 'holdout': len(val_labels)},
        'positives': {'train': tr_pos, 'holdout': ho_pos},
        'columns': {'total': 230, 'numeric': 190, 'categorical': 40},
    }


def criteo(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    src = CROSS / 'data/criteo'
    labels = (src / 'criteo_train.labels').read_text().split()
    val_labels = (src / 'criteo_holdout.labels').read_text().split()

    def convert(features: Path, labs: list[str], dst: Path) -> int:
        positives = 0
        with features.open() as fin, dst.open('w') as fout:
            for line, lab in zip(fin, labs):
                label = 1 if int(lab) == 1 else 0
                positives += label
                fout.write(f'{label}\t{line.rstrip(chr(10))}\n')
        return positives

    tr_pos = convert(src / 'criteo_train.dsv', labels, out / 'train.tsv')
    ho_pos = convert(src / 'criteo_holdout.dsv', val_labels, out / 'holdout.tsv')
    write_cd(out / 'train.cd', 13, 39)
    return {
        'source': 'Criteo Click Logs sample: train 2015-02-16 (2 shards), holdout 2015-02-23 (1 shard)',
        'split_rule': 'fixed previous split, later day held out',
        'rows': {'train': len(labels), 'holdout': len(val_labels)},
        'positives': {'train': tr_pos, 'holdout': ho_pos},
        'columns': {'total': 39, 'numeric': 13, 'categorical': 26},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(Path(__file__).resolve().parent.parent / 'data'))
    args = ap.parse_args()
    root = Path(args.root)
    manifest = {
        'seed': SEED,
        'label_encoding': '0/1',
        'layout': 'target first, then features; one CD per dataset',
        'datasets': {},
    }
    for name, fn in (('amazon', amazon), ('kdd', kdd), ('criteo', criteo)):
        info = fn(root / name)
        info['sha256'] = {p.name: sha256(p) for p in sorted((root / name).iterdir())}
        manifest['datasets'][name] = info
        print(f'{name}: train={info["rows"]["train"]} holdout={info["rows"]["holdout"]} '
              f'features={info["columns"]["total"]} '
              f'(num={info["columns"]["numeric"]}, cat={info["columns"]["categorical"]})')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print('manifest:', root / 'manifest.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
