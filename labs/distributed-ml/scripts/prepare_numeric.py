#!/usr/bin/env python3
"""Create the float-only variants used by CatBoost's distributed CPU mode.

CatBoost's CPU distributed training is documented as supporting only float
features; CTRs on workers are not computed.  These files let Stage 2 measure
the distributed machinery without hitting that limitation, and Stage 3 use the
resulting models as the no-CTR negative control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def cut(src: Path, dst: Path, fields: int) -> int:
    rows = 0
    with src.open() as fin, dst.open('w') as fout:
        for line in fin:
            fout.write('\t'.join(line.rstrip('\n').split('\t')[:fields]) + '\n')
            rows += 1
    return rows


def cd(path: Path, numeric: int) -> None:
    lines = ['0\tTarget'] + [f'{i}\tNum' for i in range(1, numeric + 1)]
    path.write_text('\n'.join(lines) + '\n')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(Path(__file__).resolve().parent.parent / 'data'))
    args = ap.parse_args()
    root = Path(args.root)
    info = {}
    for name, numeric in (('kdd', 190), ('criteo', 13)):
        d = root / name
        rows_train = cut(d / 'train.tsv', d / 'train_num.tsv', numeric + 1)
        rows_holdout = cut(d / 'holdout.tsv', d / 'holdout_num.tsv', numeric + 1)
        cd(d / 'num.cd', numeric)
        info[name] = {'numeric_features': numeric, 'train_rows': rows_train,
                      'holdout_rows': rows_holdout}
        print(f'{name}: numeric={numeric} train={rows_train} holdout={rows_holdout}')
    (root / 'numeric-manifest.json').write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
