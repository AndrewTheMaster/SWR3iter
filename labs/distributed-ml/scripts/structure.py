#!/usr/bin/env python3
"""Describe a saved CatBoost model's CTR structure and write a JSON record."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from catboost import CatBoost


def describe(cbm: Path, keep_json: bool) -> dict:
    model = CatBoost()
    model.load_model(str(cbm))
    js = cbm.with_suffix('.model.json')
    model.save_model(str(js), format='json')
    data = json.loads(js.read_text())
    ctrs = (data.get('features_info') or {}).get('ctrs') or []
    tables = Counter()
    projections = set()
    lengths = Counter()
    types = Counter()
    skipped = 0
    for entry in ctrs:
        identifier = entry.get('identifier') or ''
        tables[identifier] += 1
        elements = entry.get('elements') or []
        cat_elements = [e for e in elements
                        if e.get('combination_element') == 'cat_feature_value'
                        and 'cat_feature_index' in e]
        skipped += len(elements) - len(cat_elements)
        projections.add(tuple(sorted(e['cat_feature_index'] for e in cat_elements)))
        lengths[len(cat_elements)] += 1
        types[entry.get('ctr_type', '?')] += 1

    adjacent = 0
    runs = []
    run = 1
    for i in range(1, len(ctrs)):
        if ctrs[i].get('identifier') == ctrs[i - 1].get('identifier'):
            adjacent += 1
            run += 1
        else:
            if run > 1:
                runs.append(run)
            run = 1
    if run > 1:
        runs.append(run)

    trees = data.get('oblivious_trees')
    result = {
        'model': cbm.name,
        'model_path': str(cbm),
        'ctr_features': len(ctrs),
        'ctr_tables': len(tables),
        'projections': len(projections),
        'ctr_per_projection': round(len(ctrs) / len(projections), 4) if projections else 0.0,
        'ctr_per_table': round(len(ctrs) / len(tables), 4) if tables else 0.0,
        'adjacent_same_table_pairs': adjacent,
        'adjacent_same_table_fraction': round(adjacent / max(len(ctrs) - 1, 1), 4) if len(ctrs) > 1 else 0.0,
        'runs_of_same_table': len(runs),
        'longest_run': max(runs) if runs else 0,
        'mean_run_length': round(sum(runs) / len(runs), 4) if runs else 0.0,
        'elements_without_cat_index': skipped,
        'complexity_histogram': {str(k): v for k, v in sorted(lengths.items())},
        'ctr_types': dict(sorted(types.items())),
        'trees': len(trees) if isinstance(trees, (list, dict)) else None,
        'categorical_features': len((data.get('features_info') or {}).get('categorical_features') or []),
        'numeric_features': len((data.get('features_info') or {}).get('float_features') or []),
        'model_bytes': cbm.stat().st_size,
    }
    if not keep_json:
        js.unlink(missing_ok=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='+')
    ap.add_argument('--keep-json', action='store_true')
    ap.add_argument('--out')
    args = ap.parse_args()
    records = [describe(Path(m), args.keep_json) for m in args.models]
    text = json.dumps(records, indent=2, ensure_ascii=False) + '\n'
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
