"""Export a published-split baseline, with explicit overlaps excluded.

Eligibility is scoped to this screening policy and frozen-feature baseline.
It is not a claim of compliance with CATH-ddG or a full official test benchmark.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(newline='') as f: return list(csv.DictReader(f))


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = ROOT/'data/processed/joint_split_audit_v1'
    out = ROOT/'data/processed/joint_training_v2'
    summary = json.loads((audit/'summary.json').read_text())
    for path, digest in summary['inputs'].items():
        assert sha(Path(path)) == digest, 'Audit input changed: '+path
    for path, digest in summary['external_files'].items():
        assert sha(Path(path)) == digest, 'External structure changed: '+path
    all_abs = read(audit/'absolute_screened.csv')
    all_pairs = read(audit/'pairs_screened.csv')
    priority = {'train':0,'val':1,'test':2}
    original = read(ROOT/'data/manifests/ppb_affinity/structure_manifest.csv')
    groups = defaultdict(list)
    for row in original:
        key = tuple(sorted(tuple(sorted(s.strip() for s in row[field].split(',')))
                           for field in ['Ligand Sequences','Receptor Sequences']))
        groups[key].append(row)
    whole_complex_conflicts = set()
    for group in groups.values():
        maximum = max(priority[r['split']] for r in group)
        whole_complex_conflicts.update(r['csv_line'] for r in group if priority[r['split']] < maximum)
    hard_reasons = {'external_chain_similarity','external_chain_identity','pdb_cross_split','feature_cross_split'}
    absolute = []
    excluded = []
    for row in all_abs:
        reasons = set(row['exclusion_reasons'].split(';')) & hard_reasons
        if row['csv_line'] in whole_complex_conflicts: reasons.add('identical_complex_cross_split')
        row['similarity_flags'] = row['exclusion_reasons']
        row['exclusion_reasons'] = ';'.join(sorted(reasons))
        row['screening_pass'] = not reasons
        (excluded if reasons else absolute).append(row)
    allowed_ids = {r['csv_line'] for r in absolute}
    pairs = []
    for row in all_pairs:
        if all(i in allowed_ids for i in row['ppb_csv_lines'].split(',')):
            row['similarity_flags'] = row['exclusion_reasons']
            row.update(screening_pass=True, exclusion_reasons='')
            pairs.append(row)
    byid = {r['csv_line']:r for r in absolute}
    assert len(byid)==len(absolute)
    for r in pairs:
        for i in r['ppb_csv_lines'].split(','):
            a = byid[i]
            assert r['split']==a['split']
            assert r['mutant_feature_path']==a['feature_path']
            assert r['wt_feature_path']==a['wt_feature_path']
    feature_splits = {}
    for row in absolute + pairs:
        for key in ['feature_path','wt_feature_path','mutant_feature_path']:
            if row.get(key):
                path = Path(row[key])
                assert path.is_file(), str(path)
                assert feature_splits.setdefault(str(path),row['split'])==row['split'], 'Cross-split feature leakage'
    out.mkdir(parents=True,exist_ok=True)
    if excluded:
        with (out/'excluded_absolute.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(excluded[0]));w.writeheader();w.writerows(excluded)
    counts = {}
    for name, rows in [('absolute',absolute),('pairs',pairs)]:
        assert rows, 'No remaining '+name+' rows'
        for row in rows: row['training_eligible']='True'
        counts[name]=dict(Counter(r['split'] for r in rows))
        for split in ['train','val','test']:
            with (out/f'{name}_{split}.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
                w.writerows(r for r in rows if r['split']==split)
    report={'counts':counts,'audit_summary_sha256':sha(audit/'summary.json'),
            'script_sha256':sha(Path(__file__)),
            'input_hashes':{str(audit/p):sha(audit/p) for p in ['absolute_screened.csv','pairs_screened.csv']},
            'policy':'Preserve published split; remove external-chain hits, same-PDB/shared-feature cross-split conflicts, and identical full partner sequence pairs across splits. Holdout precedence test > val > train. Single-chain similarity is reported, not excluded by default; strict single-chain screen is retained separately in joint_training_v1.',
            'excluded_absolute':len(excluded),
            'eligibility_scope':'Frozen-feature first baseline. Does not establish chain-disjoint or remote-homology-free evaluation; not CATH-ddG benchmark. Difference loss should use delta_pkd directly; source-assumed temperatures remain flagged for kcal/mol reporting. No new physical scalar features or ranking groups prepared here.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(counts,indent=2))


if __name__=='__main__': main()
