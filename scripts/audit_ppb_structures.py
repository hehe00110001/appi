"""Match PPB rows to existing source-specific PDB filenames without modifying structures.

This establishes file coverage only; it does not validate chain/sequence/mutation mapping.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=Path('data/raw/ppb_affinity/filtered.csv'))
    parser.add_argument('--pdb-root', type=Path,
                        default=Path('/home/weizg/wei/dataset/PPI/PDB'))
    parser.add_argument('--out', type=Path, default=Path('data/manifests/ppb_affinity'))
    args = parser.parse_args()
    root = args.pdb_root.resolve(strict=True)
    with args.csv.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)
    required = {'Source Data Set', 'Complex ID', 'PDB', 'Mutations', 'Ligand Chains',
                'Receptor Chains', 'KD(M)', 'split'}
    if not required.issubset(columns or []):
        raise ValueError(f'Missing fields: {required - set(columns or [])}')

    index = defaultdict(list)
    inventory = {}
    # Only the official source directories and their direct PDB files are used;
    # derived structures in nested directories must not silently become candidates.
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        files = sorted(p for p in directory.iterdir()
                       if p.is_file() and p.suffix.lower() == '.pdb')
        inventory[directory.name] = len(files)
        for path in files:
            # Handles both 1ABC.pdb and 1abc.ent.pdb.
            pdb = path.name.split('.')[0].upper()
            index[(directory.name, pdb)].append(path)

    matches = []
    counts = Counter()
    by_source = defaultdict(Counter)
    matched_files = set()
    for number, row in enumerate(rows, start=2):
        candidates = index[(row['Source Data Set'], row['PDB'].strip().upper())]
        status = 'missing' if not candidates else 'ambiguous' if len(candidates) > 1 else 'matched'
        path = candidates[0] if len(candidates) == 1 else None
        if path is not None and path.stat().st_size == 0:
            status = 'empty_file'
        pkd = ''
        try:
            kd = float(row['KD(M)'])
            valid_kd = math.isfinite(kd) and kd > 0
            if valid_kd:
                pkd = -math.log10(kd)
        except (ValueError, TypeError):
            valid_kd = False
        counts[status] += 1
        counts['invalid_kd'] += not valid_kd
        by_source[row['Source Data Set']][status] += 1
        if status == 'matched':
            matched_files.add(str(path))
        matches.append({
            **row, 'csv_line': number, 'structure_status': status,
            'structure_path': str(path) if path is not None else '',
            'candidate_paths': json.dumps([str(p) for p in candidates]),
            'label_pkd': pkd, 'valid_kd': valid_kd,
        })

    report = {
        'csv': str(args.csv.resolve()),
        'csv_sha256': hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        'pdb_root': str(root), 'rows': len(rows),
        'splits': dict(Counter(r['split'] for r in rows)),
        'sources': dict(Counter(r['Source Data Set'] for r in rows)),
        'rows_with_mutation_annotations': sum(bool(r['Mutations'].strip()) for r in rows),
        'direct_pdb_files_by_source': inventory, 'coverage': dict(counts),
        'coverage_by_source': {k: dict(v) for k, v in by_source.items()},
        'unique_matched_structure_files': len(matched_files),
        'scope': 'Filename/source and nonempty-file coverage only. Chain, sequence, mutation, '
                 'structure provenance and leakage checks remain pending; not a training-ready manifest.',
    }
    args.out.mkdir(parents=True, exist_ok=True)
    if matches:
        with (args.out / 'structure_manifest.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(matches[0]))
            writer.writeheader()
            writer.writerows(matches)
    (args.out / 'structure_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
