"""Join original SKEMPI experiments to verified PPB features; no training or encoding.

Match either explicitly published numbering scheme, retaining the mapping used.
All training_eligible fields remain false pending sequence-level leakage screening.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from appi.units import pkd_from_molar, delta_pkd_to_ddg

AA = 'ACDEFGHIKLMNPQRSTVWY'


def exact_positive(text):
    # Raw fields retain bounds; parsed fields can erase that information.
    value = float(text.strip())
    if not math.isfinite(value) or value <= 0:
        raise ValueError('nonpositive_or_nonfinite')
    return value


def mutations(text, skempi=False):
    result = []
    for token in text.split(','):
        token = token.strip()
        pattern = (rf'([{AA}])([^\s_,])(-?\d+)([A-Za-z]?)([{AA}])' if skempi
                   else rf'([^\s_,])_([{AA}])(-?\d+)([A-Za-z]?)([{AA}])')
        m = re.fullmatch(pattern, token)
        if m is None:
            raise ValueError('unsupported_mutation')
        a, b, number, insertion, mutant = m.groups()
        wt, chain = (a, b) if skempi else (b, a)
        result.append((chain, wt, int(number), insertion, mutant))
    if len({(m[0], m[2], m[3]) for m in result}) != len(result):
        raise ValueError('duplicate_mutation_site')
    return tuple(sorted(result))


def partner_key(a, b):
    return tuple(sorted((tuple(sorted(a)), tuple(sorted(b)))))


def ppb_key(row):
    return (row['PDB'].upper(), partner_key(row['Ligand Chains'].replace(' ', '').split(','),
            row['Receptor Chains'].replace(' ', '').split(',')), mutations(row['Mutations']))


def skempi_key(row):
    pdb, a, b = row['#Pdb'].split('_')
    return pdb.upper(), partner_key(a, b), mutations(row['Mutation(s)_PDB'], True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path, delimiter=','):
    with path.open(newline='') as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def build(raw, encoded, manifest):
    index = defaultdict(list)
    for row in encoded:
        if row['Source Data Set'] == 'SKEMPI v2.0' and row['Mutations'].strip() and row['build_status'] == 'encoded':
            index[ppb_key(row)].append(row)
    # Check all published rows, including records held by feature preprocessing.
    pdb_splits = defaultdict(set)
    for row in manifest:
        pdb_splits[row['PDB'].upper()].add(row['split'])
    results = []
    duplicates = {}
    for line, raw_row in enumerate(raw, 2):
        r = dict(raw_row, skempi_csv_line=line, pair_status='', reason='', ppb_csv_lines='',
                 split='', wt_feature_path='', mutant_feature_path='', pkd_wt='', pkd_mut='',
                 delta_pkd='', ddg_kcal_mol='', temperature_k='', temperature_assumed='',
                 same_pdb_splits='', duplicate_of='', numbering_match='', matched_ppb_mutations='', training_eligible=False)
        try:
            key = skempi_key(raw_row)
            wt = exact_positive(raw_row['Affinity_wt (M)'])
            mut = exact_positive(raw_row['Affinity_mut (M)'])
        except (ValueError, KeyError) as exc:
            r.update(pair_status='label_or_notation_review', reason=str(exc))
            results.append(r)
            continue
        r.update(pkd_wt=pkd_from_molar(wt), pkd_mut=pkd_from_molar(mut))
        r['delta_pkd'] = r['pkd_mut'] - r['pkd_wt']
        temp = re.fullmatch(r'(\d+(?:\.\d+)?)(\(assumed\))?', raw_row['Temperature'].strip())
        if temp and float(temp[1]) > 0:
            r.update(temperature_k=float(temp[1]), temperature_assumed=bool(temp[2]))
            r['ddg_kcal_mol'] = delta_pkd_to_ddg(r['delta_pkd'], r['temperature_k'])
        schemes = {'PDB': key}
        try:
            schemes['cleaned'] = (*key[:2], mutations(raw_row['Mutation(s)_cleaned'], True))
        except ValueError:
            pass
        candidates_by_line = {}
        scheme_by_line = defaultdict(set)
        for scheme, candidate_key in schemes.items():
            for candidate in index.get(candidate_key, []):
                candidates_by_line[candidate['csv_line']] = candidate
                scheme_by_line[candidate['csv_line']].add(scheme)
        candidates = list(candidates_by_line.values())
        matches = [p for p in candidates
                   if math.isclose(float(p['KD(M)']), mut, rel_tol=1e-8, abs_tol=0)
                   and p['Affinity PubMed ID'].strip() == raw_row['Reference'].strip()
                   and p['Temperature(K)'].strip() == raw_row['Temperature'].strip()]
        r['same_pdb_splits'] = ','.join(sorted(pdb_splits[key[0]]))
        if not matches:
            r.update(pair_status='no_feature_match', reason='experiment_metadata_mismatch' if candidates else 'no_encoded_pdb_partners_mutation')
        else:
            signatures = {(p['split'], p['feature_path'], p['wt_feature_path'], ppb_key(p)) for p in matches}
            if len(signatures) != 1:
                r.update(pair_status='ambiguous_feature_match', reason='multiple_splits_or_feature_pairs')
            else:
                split, mutant_path, wt_path, _ = signatures.pop()
                r.update(split=split, mutant_feature_path=mutant_path, wt_feature_path=wt_path,
                         numbering_match=','.join(sorted(set.union(*(scheme_by_line[p['csv_line']] for p in matches)))),
                         matched_ppb_mutations=matches[0]['Mutations'],
                         ppb_csv_lines=','.join(p['csv_line'] for p in matches), pair_status='paired_pending_leakage_audit')
                if not all(p and Path(p).is_file() for p in (mutant_path, wt_path)):
                    r.update(pair_status='missing_feature_file', reason='feature_path_not_found')
                elif len(pdb_splits[key[0]]) != 1:
                    r.update(pair_status='split_conflict', reason='same_pdb_in_multiple_published_splits')
                elif not temp:
                    r.update(pair_status='temperature_review', reason='no_numeric_temperature_for_ddg')
                elif temp[2]:
                    r.update(pair_status='paired_assumed_temperature', reason='ddg_uses_source_assumed_temperature')
        # Preserve repeated measurements, but identify indistinguishable supervision.
        identity = (key, wt, mut, raw_row['Reference'], raw_row['Temperature'], raw_row['Method'])
        if identity in duplicates:
            r['duplicate_of'] = duplicates[identity]
        else:
            duplicates[identity] = line
        results.append(r)
    return results


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT/'data/raw/skempi2/skempi_v2.csv')
    parser.add_argument('--features', type=Path, default=ROOT/'data/processed/ppb_ppiformer_ready_v3/samples.csv')
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/manifests/ppb_affinity/structure_manifest.csv')
    parser.add_argument('--out', type=Path, default=ROOT/'data/processed/skempi_pairs_v2')
    args = parser.parse_args()
    raw = read_csv(args.raw, ';')
    rows = build(raw, read_csv(args.features), read_csv(args.manifest))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'pairs_audit.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paired = [r for r in rows if r['pair_status'].startswith('paired_') and not r['duplicate_of']]
    with (args.out/'paired_candidates.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(paired)
    summary = dict(raw_rows=len(raw), counts=dict(Counter(r['pair_status'] for r in rows)),
                   unique_paired_candidates=len(paired), candidate_splits=dict(Counter(r['split'] for r in paired)),
                   duplicate_records=sum(bool(r['duplicate_of']) for r in rows),
                   candidate_temperatures=dict(Counter(str(r['temperature_assumed']) for r in paired)),
                   candidate_numbering=dict(Counter(r['numbering_match'] for r in paired)),
                   inputs={str(p):sha(p) for p in [args.raw,args.features,args.manifest]},
                   script_sha256=sha(Path(__file__)),
                   policy='Either published PDB or cleaned mutation numbering + unordered partner groups + mutant Kd + reference + raw temperature. Reject conflicting mappings; record numbering used. Existing encoded samples already passed residue validation. No arbitrary chain renaming, fitted labels or split reassignment. Assumed temperatures flagged. Same-PDB split conflicts held. All training_eligible=False pending sequence-level leakage and external holdout audit.')
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
