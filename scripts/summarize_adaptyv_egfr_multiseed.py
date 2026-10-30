"""Merge frozen EGFR evaluations and describe seed variability without selection."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from evaluate_adaptyv_egfr import ROOT, file_digest, read, scopes, summarize
from train_affinity import json_save

ARMS = ('structure_transfer', 'fusion_absolute', 'fusion_transfer', 'fusion_shuffled')


def close_tree(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            close_tree(a[key], b[key])
    elif isinstance(a, (float, int)):
        assert np.isclose(a, b, rtol=1e-9, atol=1e-11), (a, b)
    else:
        assert a == b, (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = parser.parse_args()
    assert len(args.seeds) == len(set(args.seeds))
    assert all(args.out.resolve() != p.resolve() for p in args.inputs)
    records_path = ROOT / 'data/processed/adaptyv_egfr_v1/audited_records.csv'
    records = read(records_path)
    combined = None
    predictions = {}
    sources = []
    for folder in args.inputs:
        result = json.loads((folder / 'metrics.json').read_text())
        assert not result['smoke_test']
        assert result['records_sha256'] == file_digest(records_path)
        if combined is None:
            combined = {k: v for k, v in result.items() if k != 'models'}
            combined['models'] = {}
        else:
            for key in ('policy', 'features', 'records_sha256', 'baselines'):
                close_tree(combined[key], result[key])
        rows = read(folder / 'predictions.csv')
        assert {r['model'] for r in rows} == set(result['models'])
        for name, model in result['models'].items():
            selected = [r for r in rows if r['model'] == name]
            scores = {r['sample_id']: float(r['predicted_pkd']) for r in selected}
            assert len(scores) == len(selected) == 587
            assert sum(r['round'] == '2' for r in selected) == 400
            for scope, cohort in scopes(records):
                close_tree(summarize(cohort, scores), model['scopes'][scope])
            if name in combined['models']:
                # The historical sequence reference is repeated by the frozen evaluator.
                assert name == 'historical_seq_B_seed42'
                assert combined['models'][name]['checkpoint_sha256'] == model['checkpoint_sha256']
                previous = {r['sample_id']: float(r['predicted_pkd']) for r in predictions[name]}
                assert previous.keys() == scores.keys()
                assert all(np.isclose(previous[k], scores[k], rtol=1e-5, atol=1e-5) for k in scores)
            else:
                combined['models'][name] = model
                predictions[name] = selected
        sources.append(dict(directory=str(folder.resolve()),
                            metrics_sha256=file_digest(folder / 'metrics.json'),
                            predictions_sha256=file_digest(folder / 'predictions.csv')))
    expected = {f'{arm}_seed{seed}/{choice}' for arm in ARMS for seed in args.seeds
                for choice in ('best', 'best_absolute')} | {'historical_seq_B_seed42'}
    assert set(combined['models']) == expected
    combined['merge_provenance'] = sources
    summary = dict(seeds=args.seeds, sources=sources,
                   note='Mean and sample SD across seeds, not independent test cohorts. No seed or checkpoint selected on EGFR. Historical sequence reference remains seed42 only.',
                   checkpoints={})
    columns = [('binding_detection', 'auroc'), ('binding_detection', 'average_precision'),
               ('affinity_primary', 'mae'), ('affinity_primary', 'pearson'),
               ('affinity_primary', 'spearman')]
    lines = ['# Frozen EGFR evaluation across seeds', '', summary['note'], '',
             'Round2/all is primary. Round1, strata and 10uM sensitivity remain separate in metrics.json.', '']
    for choice in ('best', 'best_absolute'):
        table = {}
        lines += [f'## {choice}', '', '|Arm|AUROC|AP|pKd MAE|Pearson|Spearman|',
                  '|---|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            per_seed = {str(seed): combined['models'][f'{arm}_seed{seed}/{choice}']['scopes']
                        for seed in args.seeds}
            aggregates = {}
            for scope in next(iter(per_seed.values())):
                aggregates[scope] = {}
                for group, metric in columns:
                    values = [v[scope].get(group, {}).get(metric) for v in per_seed.values()]
                    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
                    aggregates[scope][metric] = dict(n=len(finite),
                        mean=float(np.mean(finite)) if finite else None,
                        sample_sd=float(np.std(finite, ddof=1)) if len(finite) > 1 else None)
            table[arm] = dict(per_seed=per_seed, aggregates=aggregates)
            cells = []
            for _, metric in columns:
                stats = aggregates['round2/all'][metric]
                mean, sd = stats['mean'], stats['sample_sd']
                cells.append('NA' if mean is None else f'{mean:.3f}' + (f' ± {sd:.3f}' if sd is not None else ''))
            lines.append('|' + '|'.join([arm] + cells) + '|')
        summary['checkpoints'][choice] = table
        lines.append('')
    args.out.mkdir(parents=True, exist_ok=True)
    allrows = [r for selected in predictions.values() for r in selected]
    with (args.out / 'predictions.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(allrows[0]))
        writer.writeheader()
        writer.writerows(allrows)
    json_save(combined, args.out / 'metrics.json')
    json_save(summary, args.out / 'multiseed_summary.json')
    (args.out / 'multiseed_report.md').write_text('\n'.join(lines) + '\n')
    print(f'Verified and merged {len(expected)} models; 400/400 Round2 predictions each.', flush=True)


if __name__ == '__main__':
    main()
