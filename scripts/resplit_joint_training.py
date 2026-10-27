"""Retain all eligible records and make a fixed PDB/feature-grouped split.

Assignment uses record counts, sources and group counts, never affinity labels.
No additional sequence similarity filtering or clustering is performed.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'val', 'test')
KINDS = ('absolute', 'pairs')
FEATURE_KEYS = ('feature_path', 'wt_feature_path', 'mutant_feature_path')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_id(kind, row):
    return row['csv_line' if kind == 'absolute' else 'skempi_csv_line']


def pdb_id(kind, row):
    return (row['PDB'] if kind == 'absolute' else row['#Pdb'].split('_')[0]).strip().upper()


def feature_paths(row):
    return [str(Path(row[k]).resolve()) for k in FEATURE_KEYS if row.get(k)]


def load_tables(directory):
    tables, hashes = {k: [] for k in KINDS}, {}
    for kind in KINDS:
        for split in SPLITS:
            path = directory / f'{kind}_{split}.csv'
            hashes[path.name] = sha(path)
            with path.open(newline='') as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                if row['split'] != split or row['training_eligible'] != 'True':
                    raise ValueError(f'Ineligible or misplaced record in {path}')
                if 'original_split' in row or 'group_id' in row:
                    raise ValueError('Expected original baseline tables, not an earlier resplit')
            tables[kind].extend(rows)
        tables[kind].sort(key=lambda r: int(record_id(kind, r)))
        ids = [record_id(kind, r) for r in tables[kind]]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f'Empty table or duplicate record IDs: {kind}')
    return tables, hashes


def make_groups(tables):
    records = [(kind, row) for kind in KINDS for row in tables[kind]]
    parent = list(range(len(records)))

    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        parent[max(a, b)] = min(a, b)

    owners, absolute = {}, {}
    for i, (kind, row) in enumerate(records):
        if kind == 'absolute':
            absolute[row['csv_line']] = (i, row)
        pdb = pdb_id(kind, row)
        if not pdb:
            raise ValueError('Missing PDB ID')
        for key in [('pdb', pdb), *(('feature', p) for p in feature_paths(row))]:
            union(i, owners.setdefault(key, i))
    for i, (kind, row) in enumerate(records):
        if kind != 'pairs':
            continue
        for rid in row['ppb_csv_lines'].split(','):
            if rid not in absolute:
                raise ValueError('Pair references missing absolute record: ' + rid)
            j, a = absolute[rid]
            if pdb_id(kind, row) != pdb_id('absolute', a):
                raise ValueError('Pair and absolute PDB disagree')
            if (Path(row['mutant_feature_path']).resolve() != Path(a['feature_path']).resolve()
                    or Path(row['wt_feature_path']).resolve() != Path(a['wt_feature_path']).resolve()):
                raise ValueError('Pair and absolute features disagree')
            if not np.isclose(float(row['pkd_mut']), float(a['label_pkd']), rtol=0, atol=1e-8):
                raise ValueError('Pair and absolute labels disagree')
            union(i, j)
        if not np.isclose(float(row['delta_pkd']), float(row['pkd_mut'])-float(row['pkd_wt']),
                          rtol=0, atol=1e-8):
            raise ValueError('Pair difference label disagrees with endpoints')
    components = defaultdict(list)
    for i, record in enumerate(records):
        components[find(i)].append(record)
    groups = []
    for members in components.values():
        pdbs = sorted({pdb_id(k, r) for k, r in members})
        groups.append(dict(group_id='pdb_' + '_'.join(pdbs), pdbs=pdbs, records=members))
    return sorted(groups, key=lambda g: g['group_id'])


def group_vectors(groups):
    sources = sorted({r['Source Data Set'] for g in groups for k, r in g['records'] if k == 'absolute'})
    vectors = []
    for group in groups:
        counts = Counter(k for k, _ in group['records'])
        sc = Counter(r['Source Data Set'] for k, r in group['records'] if k == 'absolute')
        vectors.append([counts['absolute'], counts['pairs'], *(sc[s] for s in sources),
                        int(counts['pairs'] > 0), 1])
    return (np.asarray(vectors, dtype=float),
            ['absolute', 'pairs', *sources, 'groups_with_pairs', 'groups'],
            np.asarray([2., 2., *([1.] * len(sources)), .25, .1]))


def assign_groups(vectors, fractions, weights, seed):
    """Deterministic count balancing: largest-first, then whole-group moves/swaps."""
    if len(vectors) < len(fractions):
        raise ValueError('Too few groups for nonempty splits')
    rng = np.random.default_rng(seed)
    totals = vectors.sum(0)
    targets = np.asarray(fractions)[:, None] * totals[None, :]
    inv = weights[None, :] / np.maximum(targets, 1.)**2

    def objective(counts):
        return float(((counts-targets)**2 * inv).sum())

    size = (vectors / np.maximum(totals, 1.)).max(1)
    order = sorted(range(len(vectors)), key=lambda i: (-size[i], i))
    counts = np.zeros_like(targets)
    assignments = np.full(len(vectors), -1, dtype=int)
    for i in order:
        v = vectors[i]
        costs = (inv * (2*(counts-targets)*v + v*v)).sum(1)
        dst = int(np.argmin(costs))
        assignments[i] = dst
        counts[dst] += v
    # A fixed number of deterministic optimization passes, using counts only.
    for _ in range(12):
        improved = False
        for i in rng.permutation(len(vectors)):
            src, v = assignments[i], vectors[i]
            remove_cost = (inv[src] * (-2*(counts[src]-targets[src])*v + v*v)).sum()
            costs = remove_cost + (inv * (2*(counts-targets)*v + v*v)).sum(1)
            costs[src] = 0.
            dst = int(np.argmin(costs))
            if costs[dst] < -1e-12 and counts[src, -1] > 1:
                counts[src] -= v
                counts[dst] += v
                assignments[i] = dst
                improved = True
        for i in rng.permutation(len(vectors)):
            src = assignments[i]
            candidates = np.flatnonzero(assignments != src)
            dsts = assignments[candidates]
            delta = vectors[candidates] - vectors[i]
            costs = (2*delta*(inv[src]*(counts[src]-targets[src])
                             - inv[dsts]*(counts[dsts]-targets[dsts]))
                     + delta*delta*(inv[src]+inv[dsts])).sum(1)
            if not len(costs):
                continue
            best = int(np.argmin(costs))
            if costs[best] < -1e-12:
                j, dst, d = candidates[best], dsts[best], delta[best]
                counts[src] += d
                counts[dst] -= d
                assignments[i], assignments[j] = dst, src
                improved = True
        if not improved:
            break
    if (counts[:, :2] <= 0).any():
        raise ValueError('Could not produce nonempty absolute/pair tables in every split')
    return assignments, counts, targets, objective(counts)


def export_rows(groups, assignments):
    tables = {k: [] for k in KINDS}
    for g, assigned in zip(groups, assignments):
        for kind, r in g['records']:
            tables[kind].append(dict(r, original_split=r['split'], group_id=g['group_id'],
                                     split=SPLITS[assigned]))
    for k in KINDS:
        tables[k].sort(key=lambda r: int(record_id(k, r)))
    return tables


def validate_export(original, exported, check_files=True):
    pdb_splits, feature_splits, group_splits = {}, {}, {}
    absolute = {r['csv_line']: r for r in exported['absolute']}
    for kind in KINDS:
        old = {record_id(kind, r): r for r in original[kind]}
        new = {record_id(kind, r): r for r in exported[kind]}
        if len(new) != len(exported[kind]) or old.keys() != new.keys():
            raise ValueError('Records were lost or duplicated')
        for rid, r in new.items():
            expected = dict(old[rid], split=r['split'], original_split=old[rid]['split'], group_id=r['group_id'])
            if r != expected or r['split'] not in SPLITS:
                raise ValueError('A field other than split/provenance was changed')
            for key, owner in [(pdb_id(kind, r), pdb_splits), (r['group_id'], group_splits),
                               *((p, feature_splits) for p in feature_paths(r))]:
                if owner.setdefault(key, r['split']) != r['split']:
                    raise ValueError('PDB, group or feature crosses splits: ' + key)
            if kind == 'pairs':
                if any(absolute[i]['split'] != r['split'] for i in r['ppb_csv_lines'].split(',')):
                    raise ValueError('Pair and linked absolute record cross splits')
    if check_files:
        for path in feature_splits:
            if not Path(path).is_file():
                raise ValueError('Missing feature file: ' + path)
    return dict(all_records_retained=True, original_fields_preserved=True,
                pdb_cross_split=0, feature_cross_split=0, group_cross_split=0,
                pair_absolute_split_conflicts=0, unique_feature_files=len(feature_splits),
                unique_pdbs=len(pdb_splits), n_groups=len(group_splits),
                feature_files_checked=check_files)


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refusing to write an empty table: ' + str(path))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'data/processed/joint_training_v2')
    p.add_argument('--out', type=Path, default=ROOT/'data/processed/joint_training_resplit_v1')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fractions', nargs=3, type=float, default=[.85, .075, .075])
    args = p.parse_args()
    if any(not np.isfinite(x) or x <= 0 for x in args.fractions) or not np.isclose(sum(args.fractions), 1.):
        p.error('Fractions must be positive and sum to one')
    if args.out.exists():
        p.error('Output already exists; use a new directory to preserve the fixed split')
    tables, input_hashes = load_tables(args.data)
    groups = make_groups(tables)
    vectors, columns, weights = group_vectors(groups)
    assignment, counts, targets, score = assign_groups(vectors, args.fractions, weights, args.seed)
    exported = export_rows(groups, assignment)
    audit = validate_export(tables, exported)
    group_rows = [dict(group_id=g['group_id'], pdbs=';'.join(g['pdbs']), split=SPLITS[assignment[i]],
                       **{k: int(v) for k, v in zip(columns, vectors[i])}) for i, g in enumerate(groups)]
    mapping = [dict(kind=k, record_id=record_id(k, r), pdb=pdb_id(k, r), group_id=r['group_id'],
                    original_split=r['original_split'], split=r['split']) for k in KINDS for r in exported[k]]
    report = dict(seed=args.seed, fractions=dict(zip(SPLITS, args.fractions)),
                  source_directory=str(args.data.resolve()), input_hashes=input_hashes,
                  script_sha256=sha(Path(__file__)), numpy_version=np.__version__,
                  policy='Retain every eligible input row. Union same PDB, shared resolved feature paths, '
                         'and pair-linked absolute records. No sequence-homology grouping or new exclusions. '
                         'Balance counts only, never labels; retain historical split and audit fields.',
                  scope='New mixed-source development split; includes previously used train/val/test records. '
                        'Not a pristine external test, published split reproduction, or homology-disjoint benchmark.',
                  algorithm='largest-first greedy then at most 12 seeded whole-group move/swap passes',
                  balancing_weights=dict(zip(columns, weights.tolist())), objective=score,
                  totals={k: len(v) for k, v in exported.items()}, audit=audit,
                  maximum_group_absolute=int(vectors[:, 0].max()),
                  maximum_group_pairs=int(vectors[:, 1].max()), splits={}, transitions={})
    source_rows = []
    for i, split in enumerate(SPLITS):
        block = dict(counts={k: int(v) for k, v in zip(columns, counts[i])},
                     targets=dict(zip(columns, targets[i].tolist())), labels={})
        for kind, label in [('absolute', 'label_pkd'), ('pairs', 'delta_pkd')]:
            y = np.array([float(r[label]) for r in exported[kind] if r['split'] == split])
            block['labels'][kind] = dict(mean=float(y.mean()), std=float(y.std()))
        for source in columns[2:-2]:
            n = block['counts'][source]
            global_fraction = float(vectors[:, columns.index(source)].sum()/vectors[:, 0].sum())
            source_rows.append(dict(split=split, source=source, n=n,
                                    fraction=n/block['counts']['absolute'], global_fraction=global_fraction,
                                    deviation_percentage_points=100*(n/block['counts']['absolute']-global_fraction)))
        report['splits'][split] = block
    for kind in KINDS:
        report['transitions'][kind] = {old: dict(Counter(r['split'] for r in exported[kind]
                                                       if r['original_split'] == old)) for old in SPLITS}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=args.out.name+'.tmp-', dir=args.out.parent))
    try:
        for kind in KINDS:
            for split in SPLITS:
                write_csv(staging/f'{kind}_{split}.csv', [r for r in exported[kind] if r['split'] == split])
        write_csv(staging/'group_assignments.csv', group_rows)
        write_csv(staging/'record_assignments.csv', mapping)
        write_csv(staging/'source_composition.csv', source_rows)
        report['output_hashes'] = {f.name: sha(f) for f in sorted(staging.glob('*.csv'))}
        (staging/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        staging.rename(args.out)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(json.dumps({s: report['splits'][s]['counts'] for s in SPLITS}, indent=2))
    print(json.dumps(audit, indent=2))
    print('Wrote ' + str(args.out))


if __name__ == '__main__':
    main()
