"""Prepare immutable external EGFR evaluation records; never creates training data."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_joint_splits import observed
from audit_ppb_residues import read_residues, AA3
from appi.structure_context_data import file_digest

BASE = ROOT / 'data/external/adaptyv_egfr'
OUT = ROOT / 'data/processed/adaptyv_egfr_v1'
CONTROLS = {'Cetuximab_scFv', 'Human_EGF'}


def read(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def write(p, rows):
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def binding_status(reps):
    labels = {r['binding'].lower() for r in reps}
    if labels & {'true', 'weak'}:
        return 'true'
    if 'false' in labels:
        return 'false'
    return 'unknown'


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    publication = read(BASE / 'post_competition/adaptyv/data/all_submissions.csv')
    annotations = defaultdict(list)
    for item in publication:
        annotations[int(item['round']), item['sequence']].append(item)
    rows, structure_records, sequence_set = [], {}, {}
    r1structures = defaultdict(list)
    for p in sorted((BASE / 'round1_docking_predictions/docking_predictions/predicted_complexes/candidate_complexes').glob('*.pdb')):
        chains = observed(p)
        if len(chains) != 2:
            continue
        for c, seq in chains.items():
            r1structures[seq].append((p, c, chains))
    for rnd in (1, 2):
        summary = read(BASE / f'round{rnd}/results/result_summary.csv')
        reps = defaultdict(list)
        for r in read(BASE / f'round{rnd}/results/replicate_summary.csv'):
            reps[r['name']].append(r)
        for i, row in enumerate(summary):
            name, sequence = row['name'], row['sequence'].strip()
            sample = f'r{rnd}_{i:04d}'
            assert sequence and set(sequence) <= set('ACDEFGHIKLMNPQRSTVWY')
            sid = hashlib.sha256(sequence.encode()).hexdigest()
            annotation = annotations[rnd, sequence]
            categories = sorted({a['design_category'] for a in annotation})
            sequence_set[sid] = sequence
            rr = reps[name]
            assert rr, name
            binding = row['binding'] if rnd == 2 else binding_status(rr)
            expression = row.get('expression', '') or '|'.join(sorted({r['nc_adjusted_expression'] for r in rr}))
            kd = float(row['kd']) if row['kd'] else None
            if kd is not None:
                assert math.isfinite(kd) and kd > 0 and binding == 'true'
            boundary = kd is not None and math.isclose(kd, 1e-5, rel_tol=1e-10)
            # Numeric 10uM values are conservatively excluded from the primary
            # quantitative analysis pending interpretation of raw fit/reporting limits.
            control = name in CONTROLS
            quantitative = kd is not None and not boundary and not control
            path, ligand, chains = None, None, None
            if rnd == 2 and not control:
                path = BASE / 'round2_structures/structure_predictions' / (name + '.pdb')
                chains = observed(path)
                matches = [c for c, s in chains.items() if s == sequence]
                assert len(chains) == 2 and len(matches) == 1, name
                ligand = matches[0]
            elif rnd == 1 and not control:
                matches = r1structures[sequence]
                if len(matches) == 1:
                    path, ligand, chains = matches[0]
            if path is not None:
                chain_info = {}
                for c, seq in chains.items():
                    h = hashlib.sha256(seq.encode()).hexdigest(); sequence_set[h] = seq
                    chain_info[c] = dict(sequence_id=h, partner=c == ligand)
                residues = read_residues(path)
                positions, counts = {}, Counter()
                for (c, num, insertion), residue in residues.items():
                    positions[f'{c}:{num}:{insertion}'] = counts[c]; counts[c] += 1
                structure_records[sample] = dict(chains=chain_info, positions=positions,
                    structure_path=str(path), structure_sha256=file_digest(path), ligand_chain=ligand,
                    structure_method='ColabFold AF2 multimer' if rnd == 2 else 'ColabFold monomer + HADDOCK3')
            rows.append(dict(sample_id=sample, round=rnd, name=name, submitter=row['username'], sequence_id=sid,
                sequence=sequence, control=control, binding=binding, expression=expression,
                replicate_binding_states='|'.join(sorted({r['binding'].lower() for r in rr})),
                replicate_count=len(rr), kd_molar='' if kd is None else kd, pkd='' if kd is None else -math.log10(kd),
                boundary_10um_review=boundary, quantitative_primary=quantitative,
                binding_evaluation=not control and binding in ('true', 'false'),
                structure_available=path is not None, structure_path='' if path is None else str(path),
                design_category='|'.join(categories), annotation_matches=len(annotation),
                publication_de_novo='|'.join(sorted({a['de_novo'] for a in annotation})),
                selection_stratum='|'.join(sorted({a['selected'] for a in annotation})),
                design_methods=row.get('methods', row.get('design_models', '')), pae_interaction=row.get('pae_interaction', ''),
                iptm=row.get('iptm', ''), esm_pll=row.get('esm_pll', ''), plddt=row.get('plddt', ''),
                similarity_check=row.get('similarity_check', '')))
    train = json.loads((ROOT / 'data/processed/esm2_complex40_v1/manifest.json').read_text())
    paths = set()
    for kind in ('absolute', 'pairs'):
        for r in read(ROOT / f'data/processed/joint_training_complex40_v1/{kind}_train.csv'):
            for key in ('feature_path', 'wt_feature_path', 'mutant_feature_path'):
                if r.get(key): paths.add(r[key])
    training_sequences = {c['sequence_id']: train['sequences'][c['sequence_id']]
                          for p in paths for c in train['graphs'][p]['chains'].values()}
    with (OUT / 'training_chains.fasta').open('w') as f:
        for sid, seq in sorted(training_sequences.items()): f.write(f'>{sid}\n{seq}\n')
    with (OUT / 'external_chains.fasta').open('w') as f:
        for sid, seq in sorted(sequence_set.items()): f.write(f'>{sid}\n{seq}\n')
    hashes = {str(p.relative_to(ROOT)): file_digest(p) for rnd in (1, 2) for p in (BASE / f'round{rnd}/results').glob('*.csv')}
    hashes[str((BASE/'post_competition/adaptyv/data/all_submissions.csv').relative_to(ROOT))] = file_digest(BASE/'post_competition/adaptyv/data/all_submissions.csv')
    provenance = dict(source_repositories={name: subprocess.check_output(['git','-C',str(BASE/name),'rev-parse','HEAD'],text=True).strip() for name in ('round1', 'round2', 'post_competition')},
                      input_hashes=hashes, script_sha256=file_digest(Path(__file__)),
                      training_manifest_sha256=file_digest(ROOT/'data/processed/esm2_complex40_v1/manifest.json'),
                      archive_hashes={p.name:file_digest(p) for p in (BASE/'downloads').glob('*.zip')},
                      paper_sha256=file_digest(BASE/'downloads/paper_v2.pdf'),
                      label_policy='Published KD in molar; pKd=-log10(KD). Blank KD never imputed. Controls separate. 10uM numeric boundary excluded from quantitative primary, retained in sensitivity.',
                      round1_policy='Binding aggregate: any True/weak positive; else any False negative; else unknown. Replicate disagreement retained. Round1 HADDOCK separate from Round2 AF2.',
                      external_only=True)
    summary = {}
    for rnd in (1, 2):
        rr = [r for r in rows if r['round'] == rnd and not r['control']]
        summary[f'round{rnd}'] = dict(designs=len(rr), binding=dict(Counter(r['binding'] for r in rr)),
            numeric_kd=sum(r['kd_molar'] != '' for r in rr), quantitative_primary=sum(r['quantitative_primary'] for r in rr),
            boundary_review=sum(r['boundary_10um_review'] for r in rr), structures=sum(r['structure_available'] for r in rr),
            quantitative_with_structure=sum(r['quantitative_primary'] and r['structure_available'] for r in rr),
            unique_sequences=len({r['sequence'] for r in rr}), submitters=len({r['submitter'] for r in rr}))
    ids1={r['sequence_id'] for r in rows if r['round']==1 and not r['control']}
    ids2={r['sequence_id'] for r in rows if r['round']==2 and not r['control']}
    summary['cross_round_identical_sequences'] = len(ids1 & ids2)
    summary['exact_training_chain_matches'] = sorted(set(sequence_set) & set(training_sequences))
    write(OUT/'records.csv', rows)
    (OUT/'structure_manifest.json').write_text(json.dumps(dict(sequences=sequence_set,graphs=structure_records),indent=2)+'\n')
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':main()
