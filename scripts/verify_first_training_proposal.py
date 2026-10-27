"""Read-only verification of the 2026-10-26 proposal against existing artifacts.

Writes diagnostics only; does not change training data, features, or checkpoints.
Affine fits use validation predictions only. Test-set oracle quantities are
descriptive diagnostics, never deployable baselines or model selection criteria.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from audit_ppb_residues import check_row, read_residues
from sequence_mapping import check_sequences

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/first_training_proposal_verification'


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def stats(pred, target):
    p, y = np.asarray(pred, float), np.asarray(target, float)
    mse = float(np.mean((p-y)**2))
    sp, sy = float(p.std()), float(y.std())
    r = float(np.corrcoef(p, y)[0, 1]) if sp > 0 and sy > 0 else None
    return dict(n=len(y), rmse=float(np.sqrt(mse)), bias=float((p-y).mean()),
                bias_share=float((p-y).mean()**2/mse) if mse else None,
                std_pred=sp, std_target=sy, shrinkage=sp/sy if sy else None,
                pearson=r, r2=1-mse/sy**2 if sy else None,
                oracle_affine_rmse=sy*np.sqrt(max(0, 1-r*r)) if r is not None else None)


def complex_key(row):
    if '#Pdb' in row:
        pdb, a, b = row['#Pdb'].split('_')
        return (pdb.upper(), *sorted((''.join(sorted(a)), ''.join(sorted(b)))))
    return (row['PDB'].upper(), *sorted(''.join(sorted(c.strip() for c in row[k].split(',')))
                                       for k in ('Ligand Chains', 'Receptor Chains')))


def composition(rows, kind='absolute'):
    groups = Counter(complex_key(r) for r in rows)
    sources = Counter(r.get('Source Data Set', 'SKEMPI v2.0') for r in rows)
    y = np.array([float(r['label_pkd'] if kind == 'absolute' else r['delta_pkd']) for r in rows])
    counts = np.array(list(groups.values()))
    by_source = {}
    for source in sorted(sources):
        selected = [r for r in rows if r.get('Source Data Set', 'SKEMPI v2.0') == source]
        ys = [float(r['label_pkd'] if kind == 'absolute' else r['delta_pkd']) for r in selected]
        by_source[source] = dict(n=len(selected), fraction=len(selected)/len(rows),
                                 mean=float(np.mean(ys)), std=float(np.std(ys)),
                                 n_complexes=len({complex_key(r) for r in selected}))
    return dict(n=len(rows), mean=float(y.mean()), std=float(y.std()), by_source=by_source,
                n_complexes=len(groups), n_pdb=len({complex_key(r)[0] for r in rows}),
                cluster_concentration_effective_count=float(counts.sum()**2/(counts**2).sum()),
                largest_cluster=int(counts.max()),
                complex_id_unique=len({r.get('Complex ID', r.get('#Pdb')) for r in rows}))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = ROOT / 'data/processed/joint_training_v2'
    tables = {(k, s): read(data/f'{k}_{s}.csv') for k in ('absolute', 'pairs')
              for s in ('train', 'val', 'test')}
    report = {'input_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in data.glob('*.csv')}, 'composition': {}, 'runs': {}}
    for kind in ('absolute', 'pairs'):
        block = {s: composition(tables[kind, s], kind) for s in ('train', 'val', 'test')}
        block['complex_overlap'] = {a+'_'+b: len({complex_key(r) for r in tables[kind, a]} &
                                               {complex_key(r) for r in tables[kind, b]})
                                    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
        report['composition'][kind] = block
    original = read(ROOT/'data/manifests/ppb_affinity/structure_manifest.csv')
    report['published_composition'] = {s: composition([r for r in original if r['split'] == s])
                                       for s in ('train', 'val', 'test')}
    train = tables['absolute', 'train']
    by_feature = defaultdict(list)
    for row in train:
        by_feature[row['feature_path']].append(float(row['label_pkd']))
    report['duplicate_features'] = dict(
        n_records=len(train), n_unique_paths=len(by_feature),
        paths_with_label_spread=sum(max(v)-min(v) > 1e-6 for v in by_feature.values()),
        within_path_oracle_rmse=float(np.sqrt(sum(sum((np.array(v)-np.mean(v))**2)
                                                  for v in by_feature.values())/len(train))))
    report['pair_sample_coefficient_ratio_at_lambda_1'] = len(train)/len(tables['pairs', 'train'])
    for mode in ('absolute', 'joint'):
        for seed in (42, 43, 44):
            run = ROOT/f'runs/affinity_v1/{mode}_seed{seed}'
            config = json.loads((run/'config.json').read_text())
            for name, digest in config['input_hashes'].items():
                assert hashlib.sha256((data/name).read_bytes()).hexdigest() == digest
            block = {}
            for kind in ('absolute', 'pairs'):
                predictions = {}
                for split in ('val', 'test'):
                    rows = read(run/f'predictions_{kind}_{split}.csv')
                    key = 'csv_line' if kind == 'absolute' else 'skempi_csv_line'
                    meta = {r[key]: r for r in tables[kind, split]}
                    assert len(rows) == len(meta) and {r['source_line'] for r in rows} == set(meta)
                    label = 'label_pkd' if kind == 'absolute' else 'delta_pkd'
                    assert all(np.isclose(float(r['target']), float(meta[r['source_line']][label]),
                                          rtol=0, atol=2e-6) for r in rows)
                    p = np.array([float(r['prediction']) for r in rows])
                    y = np.array([float(r['target']) for r in rows])
                    predictions[split] = p, y
                    if split == 'val':
                        a, b = np.polyfit(p, y, 1)
                        shift = float((y-p).mean())
                    sources = [meta[r['source_line']].get('Source Data Set', 'SKEMPI v2.0') for r in rows]
                    block[kind+'_'+split] = dict(raw=stats(p, y), affine_from_val=dict(a=float(a), b=float(b)),
                        recalibrated=stats(a*p+b, y), bias_only_from_val=stats(p+shift, y),
                        by_source={s: stats(p[np.array(sources) == s], y[np.array(sources) == s])
                                   for s in sorted(set(sources))})
            # A shared affine score S'=a*S+b implies delta'=a*delta, without an intercept.
            a = block['absolute_val']['affine_from_val']['a']
            for split in ('val', 'test'):
                p, y = predictions[split]
                block['pairs_'+split]['shared_absolute_affine'] = stats(a*p, y)
            report['runs'][run.name] = block
    samples = read(ROOT/'data/processed/ppb_preflight_v3/samples.csv')
    report['preflight_status'] = dict(Counter(r['build_status'] for r in samples))
    report['warning_types'] = dict(Counter(r['quality_warnings'] for r in samples
                                          if r['build_status'] == 'audit_ready_review'))
    failed = [r for r in samples if r['error'].startswith('observed_sequence_not_in_csv')]
    seq_counts, source_counts, seq_rows = Counter(), defaultdict(Counter), []
    last_path, residues = None, None
    for row in sorted(failed, key=lambda r: r['structure_path']):
        path = Path(row['structure_path'])
        if path != last_path:
            residues, last_path = read_residues(path), path
        audit = check_row(row, residues)
        assert not audit['issues']
        outcome = {}
        for name, muts in [('wt', []), ('mutant', audit['mutations'])]:
            try:
                check_sequences(row, residues, muts)
                outcome[name] = True
            except ValueError:
                outcome[name] = False
        assert not outcome['mutant']
        status = 'wt_only_match' if outcome['wt'] else 'neither_match'
        seq_counts[status] += 1
        seq_counts['with_mutation' if audit['mutations'] else 'without_mutation'] += 1
        source_counts[row['Source Data Set']][status] += 1
        seq_rows.append(dict(csv_line=row['csv_line'], pdb=row['PDB'], source=row['Source Data Set'],
                             has_mutation=bool(audit['mutations']), **outcome))
    report['sequence_direction_probe'] = dict(n=len(failed), counts=dict(seq_counts),
                                             by_source={k: dict(v) for k, v in source_counts.items()})
    (OUT/'sequence_direction_rows.json').write_text(json.dumps(seq_rows, indent=2)+'\n')
    (OUT/'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, indent=2))
    for name, block in report['runs'].items():
        print(name, json.dumps({k: {field: v[field] for field in ['raw', 'affine_from_val', 'recalibrated']}
                                for k, v in block.items() if k.endswith('test')}))


if __name__ == '__main__':
    main()
