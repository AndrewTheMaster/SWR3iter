#!/usr/bin/env python3
"""Convert downloaded Criteo_x1 parquet shards into the two experimental views.

Shards 0000-0011 are train, 0012-0014 are holdout (fixed before conversion).
Numeric view: label + I1..I13, for CatBoost CPU distributed training.
Full view: label + I1..I13 + C1..C26, for the CTR model used to validate the
cache patch.  Categorical columns are integer-coded in this dataset, but they
are still marked Categ so CatBoost builds CTRs from them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.csv as pc
import pyarrow.parquet as pq

NUM = ['label'] + [f'I{i}' for i in range(1, 14)]
CAT = [f'C{i}' for i in range(1, 27)]
ALL = NUM + CAT
WRITE = pc.WriteOptions(delimiter='\t', include_header=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def append(table, columns, path: Path, limit: int | None = None) -> int:
    if limit is not None:
        table = table.slice(0, min(limit, table.num_rows))
    with path.open('ab') as fh:
        pc.write_csv(table.select(columns), fh, write_options=WRITE)
    return table.num_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(Path(__file__).resolve().parent.parent / 'data/criteo_x1'))
    ap.add_argument('--parquet-dir', default=str(Path(__file__).resolve().parent.parent /
                                                 'data/criteo_x1_parquet/default/train'))
    ap.add_argument('--train-shards', type=int, default=12)
    ap.add_argument('--cache-train-rows', type=int, default=10_000_000)
    ap.add_argument('--cache-holdout-rows', type=int, default=2_000_000)
    args = ap.parse_args()

    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    shards = sorted(Path(args.parquet_dir).glob('*.parquet'))
    if len(shards) < args.train_shards + 1:
        raise SystemExit(f'need at least {args.train_shards+1} shards, found {len(shards)}')
    train_shards = shards[:args.train_shards]
    holdout_shards = shards[args.train_shards:]

    num_train = root / 'train_num.dsv'
    num_holdout = root / 'holdout_num.dsv'
    ctr_train = root / 'train_ctr.dsv'
    ctr_holdout = root / 'holdout_ctr.dsv'
    for p in (num_train, num_holdout, ctr_train, ctr_holdout):
        p.write_bytes(b'')

    counts = {'num_train': 0, 'num_holdout': 0, 'ctr_train': 0, 'ctr_holdout': 0}
    for shard in train_shards:
        t = pq.read_table(shard)
        counts['num_train'] += append(t, NUM, num_train)
        if counts['ctr_train'] < args.cache_train_rows:
            counts['ctr_train'] += append(t, ALL, ctr_train,
                                          args.cache_train_rows - counts['ctr_train'])
    for shard in holdout_shards:
        t = pq.read_table(shard)
        counts['num_holdout'] += append(t, NUM, num_holdout)
        if counts['ctr_holdout'] < args.cache_holdout_rows:
            counts['ctr_holdout'] += append(t, ALL, ctr_holdout,
                                            args.cache_holdout_rows - counts['ctr_holdout'])

    full_cd = ['0\tTarget'] + [f'{i}\tNum' for i in range(1, 14)] + \
              [f'{i}\tCateg' for i in range(14, 40)]
    (root / 'full.cd').write_text('\n'.join(full_cd) + '\n')
    num_cd = ['0\tTarget'] + [f'{i}\tNum' for i in range(1, 14)]
    (root / 'num.cd').write_text('\n'.join(num_cd) + '\n')

    manifest = {
        'source': 'reczoo/Criteo_x1, refs/convert/parquet',
        'split_rule': f'first {args.train_shards} shards train, remaining {len(holdout_shards)} holdout',
        'train_shards': [p.name for p in train_shards],
        'holdout_shards': [p.name for p in holdout_shards],
        'rows': counts,
        'outputs': {},
    }
    for p in (num_train, num_holdout, ctr_train, ctr_holdout,
              root / 'full.cd', root / 'num.cd'):
        manifest['outputs'][p.name] = {'bytes': p.stat().st_size, 'sha256': sha256(p)}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(manifest, indent=1, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
