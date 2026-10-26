"""Read-only chain/mutation audit. No sequence repair, mutation generation or encoding."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re


AA3 = dict(zip(
    'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),
    'ARNDCQEGHILKMFPSTWYV',
))
AA = ''.join(AA3.values())
MUTATION = re.compile(rf'^([^\s_,])_([{AA}])(-?\d+)([A-Za-z]?)([{AA}])$')


def read_residues(path):
    """Use ATOM records from the first model, preserving insertion codes and alt names."""
    residues = {}
    with path.open(errors='strict') as f:
        for line in f:
            if line.startswith('ENDMDL'):
                break
            if not line.startswith('ATOM  '):
                continue
            if len(line) < 54:
                raise ValueError('Truncated ATOM record')
            key = (line[21], int(line[22:26]), line[26].strip())
            residue = residues.setdefault(key, {'names': set(), 'atoms': set()})
            residue['names'].add(line[17:20].strip())
            residue['atoms'].add(line[12:16].strip())
    if not residues:
        raise ValueError('No ATOM residues in first model')
    return residues


def check_row(row, residues):
    issues, warnings = [], []
    ligand = [c.strip() for c in row['Ligand Chains'].split(',') if c.strip()]
    receptor = [c.strip() for c in row['Receptor Chains'].split(',') if c.strip()]
    chains = set(k[0] for k in residues)
    partners = set(ligand + receptor)
    if not ligand or not receptor:
        issues.append('empty_partner')
    if len(set(ligand)) != len(ligand) or len(set(receptor)) != len(receptor):
        issues.append('duplicate_partner_chain')
    if set(ligand) & set(receptor):
        issues.append('overlapping_partners')
    missing = sorted(partners - chains)
    if missing:
        issues.append('missing_partner_chain')
    for chain in sorted(partners & chains):
        selected = [r for k, r in residues.items() if k[0] == chain]
        if not any('CA' in r['atoms'] for r in selected):
            issues.append('partner_chain_without_CA')
        if any('CA' not in r['atoms'] for r in selected):
            warnings.append('partner_residue_without_CA')
        if any(len(r['names']) != 1 for r in selected):
            warnings.append('alternative_residue_identities')
        if any(any(n not in AA3 for n in r['names']) for r in selected):
            warnings.append('nonstandard_ATOM_residue')

    mutations, sites = [], set()
    text = row['Mutations'].strip()
    for token in text.split(',') if text else []:
        token = token.strip()
        match = MUTATION.fullmatch(token)
        item = {'annotation': token}
        mutations.append(item)
        if match is None:
            item['status'] = 'unsupported_mutation_notation'
            issues.append(item['status'])
            continue
        chain, wt, number, insertion, mut = match.groups()
        key = (chain, int(number), insertion)
        item.update(chain=chain, residue_number=int(number), insertion_code=insertion,
                    expected_wt=wt, mutant=mut)
        if key in sites:
            issues.append('duplicate_mutation_site')
        sites.add(key)
        if wt == mut:
            warnings.append('identity_mutation')
        if chain not in partners:
            issues.append('mutation_outside_partners')
        residue = residues.get(key)
        if residue is None:
            item['status'] = 'mutation_site_missing'
            item['same_number_insertion_codes'] = [k[2] for k in residues if k[:2] == key[:2]]
        else:
            item['observed_resnames'] = sorted(residue['names'])
            if len(residue['names']) != 1:
                item['status'] = 'ambiguous_mutation_residue'
            else:
                name = next(iter(residue['names']))
                if name not in AA3:
                    item['status'] = 'nonstandard_mutation_residue'
                elif AA3[name] != wt:
                    item['status'] = 'wt_residue_mismatch'
                elif 'CA' not in residue['atoms']:
                    item['status'] = 'mutation_site_without_CA'
                else:
                    item['status'] = 'matched'
        if item['status'] != 'matched':
            issues.append(item['status'])
    return {
        'issues': sorted(set(issues)), 'warnings': sorted(set(warnings)),
        'missing_partner_chains': missing, 'mutations': mutations,
        'partner_observed_residue_counts': {
            c: sum(k[0] == c for k in residues) for c in sorted(partners)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=Path('data/manifests/ppb_affinity/structure_manifest.csv'))
    parser.add_argument('--out', type=Path,
                        default=Path('data/manifests/ppb_affinity/residue_audit'))
    args = parser.parse_args()
    with args.manifest.open(newline='') as f:
        rows = list(csv.DictReader(f))
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[row['structure_path']].append(i)
    details = [None] * len(rows)
    for count, (filename, indices) in enumerate(grouped.items(), 1):
        try:
            if not filename:
                raise ValueError('No uniquely matched structure path')
            residues = read_residues(Path(filename))
        except (OSError, ValueError, UnicodeError) as exc:
            for i in indices:
                details[i] = {'issues': ['structure_read_error'], 'warnings': [],
                              'error': str(exc), 'mutations': []}
        else:
            for i in indices:
                details[i] = check_row(rows[i], residues)
        if count % 100 == 0 or count == len(grouped):
            print(f'Checked {count}/{len(grouped)} structure files', flush=True)

    totals, issues, warnings = Counter(), Counter(), Counter()
    by_split = defaultdict(Counter)
    by_source = defaultdict(Counter)
    output_rows = []
    for row, detail in zip(rows, details):
        status = 'needs_review' if detail['issues'] else 'pass_with_warnings' if detail['warnings'] else 'pass'
        totals[status] += 1
        issues.update(detail['issues'])
        warnings.update(detail['warnings'])
        by_split[row['split']][status] += 1
        by_source[row['Source Data Set']][status] += 1
        output_rows.append({**row, 'residue_audit_status': status,
                            'residue_audit_details': json.dumps(detail)})
    report = {
        'rows': len(rows), 'unique_structure_paths': len(grouped),
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'status_counts': dict(totals), 'issue_row_counts': dict(issues),
        'warning_row_counts': dict(warnings),
        'by_split': {k: dict(v) for k, v in by_split.items()},
        'by_source': {k: dict(v) for k, v in by_source.items()},
        'policy': 'First model ATOM records only; exact case-sensitive chain and insertion-code '
                  'matching. No automatic WT/already-mutant substitution or numbering correction.',
        'scope': 'Chain/mutation audit only. Full sequence correspondence, coordinates, interface '
                 'coverage and leakage still require checks. No records deleted or splits changed.',
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, records in [('all_rows.csv', output_rows),
                              ('review_rows.csv', [r for r in output_rows if r['residue_audit_status'] != 'pass'])]:
        with (args.out / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(records)
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    print(f'Audit completed: {args.out.resolve()}')


if __name__ == '__main__':
    main()
