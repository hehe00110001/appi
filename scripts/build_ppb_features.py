"""Build frozen PPIformer features for PPB absolute labels, with resumable caches.

Outputs retain published splits; external holdout exclusion is NOT yet applied.
Uses exact checkpoint graph preprocessing, then modifies residue one-hot features.
Interface = union of residues with heavy-atom distance < radius across partner groups.
"""
import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
from scipy.spatial import cKDTree
from omegaconf import OmegaConf
import hydra
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from appi.model import symmetric_pool
from audit_ppb_residues import AA3, read_residues, check_row
from sequence_mapping import check_sequences
from ppiformer.model.ppiformer import PPIformer
from ppiformer.utils.bio import amino_acid_to_class

ROOT = Path(__file__).resolve().parents[1]


class DataReview(ValueError):
    """Expected data eligibility issue, distinct from an execution failure."""


def validate_record(row, residues):
    audit = check_row(row, residues)
    if audit['issues']:
        raise DataReview('residue_audit:' + ','.join(audit['issues']))
    muts = sorted(audit['mutations'], key=lambda m: (m['chain'], m['residue_number'], m['insertion_code']))
    # Do not silently turn noncanonical or ambiguous residue identities into sequence letters.
    if any(w in audit['warnings'] for w in ('nonstandard_ATOM_residue', 'alternative_residue_identities')):
        raise DataReview('residue_identity_review:' + ','.join(audit['warnings']))
    try:
        check_sequences(row, residues, muts)
    except ValueError as exc:
        raise DataReview(str(exc)) from exc
    return muts, audit['warnings']


def validate_reuse(old, current):
    # Only the explicitly reviewed v1 writer is compatible; do not accept arbitrary scripts.
    if old['script_sha256'] != 'c4db519d93e17b41bfe592e18ec43858031e1e412339598c8fcbf05a8b4d9206':
        raise ValueError('Unsupported old feature writer')
    for field in ('checkpoint_sha256','radius_angstrom','device','packages','features',
                  'mutation_policy','pool_code_sha256','audit_code_sha256'):
        if old[field] != current[field]:
            raise ValueError(f'Incompatible old cache: {field}')


def row_base_key(row):
    return (row['structure_path'],
            tuple(sorted(c.strip() for c in row['Ligand Chains'].split(','))),
            tuple(sorted(c.strip() for c in row['Receptor Chains'].split(','))))


def label_pkd(row):
    kd = float(row['KD(M)'])
    if not math.isfinite(kd) or kd <= 0:
        raise DataReview('invalid_kd')
    value = -math.log10(kd)
    if row.get('label_pkd', '').strip():
        previous = float(row['label_pkd'])
        if not math.isfinite(previous) or not math.isclose(value, previous, rel_tol=0, abs_tol=1e-8):
            raise DataReview('manifest_pkd_disagrees_with_kd')
    return value


def output_columns(row):
    return list(dict.fromkeys([*row, 'label_pkd', 'build_status','feature_path',
                              'wt_feature_path','error','quality_warnings','training_eligible','sequence_mapping']))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def interface(path, ligand, receptor, radius, output):
    """Select first-model canonical ATOM residues; choose altloc per atom by occupancy.

    Hydrogens and HETATM are excluded. Group only cross-partner contacts, never
    antibody heavy/light contacts within the same partner. Preserve insertion codes.
    """
    chosen = {}
    with path.open() as f:
        for line in f:
            if line.startswith('ENDMDL'):
                break
            if not line.startswith('ATOM  ') or line[21] not in ligand + receptor:
                continue
            if line[17:20] not in AA3:
                continue
            atom = line[12:16].strip()
            element = line[76:78].strip() or atom.lstrip('0123456789')[:1]
            if element.upper() in ('H', 'D'):
                continue
            residue = (line[21], int(line[22:26]), line[26].strip())
            atom_key = (*residue, atom)
            occupancy = float(line[54:60].strip() or 0)
            priority = (occupancy, line[16] == ' ', line[16] == 'A')
            if atom_key not in chosen or priority > chosen[atom_key][0]:
                chosen[atom_key] = (priority, line[:16] + ' ' + line[17:])
    records = [(k[:3], v[1]) for k, v in sorted(chosen.items())]
    xyz = np.array([[float(line[a:b]) for a, b in ((30,38),(38,46),(46,54))]
                    for _, line in records])
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('empty_or_invalid_coordinates')
    mask = np.array([r[0] in ligand for r, _ in records])
    if not mask.any() or mask.all():
        raise ValueError('empty_partner_heavy_atoms')
    a, b = np.flatnonzero(mask), np.flatnonzero(~mask)
    da = cKDTree(xyz[b]).query(xyz[a], k=1)[0]
    db = cKDTree(xyz[a]).query(xyz[b], k=1)[0]
    selected = {records[i][0] for i in np.concatenate((a[da < radius], b[db < radius]))}
    if not selected:
        raise ValueError('no_cross_partner_interface')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(line for residue, line in records if residue in selected) + 'END\n')
    return selected


def node_key(node):
    parts = node.split(':')
    return (parts[0], int(parts[2]), parts[3] if len(parts) > 3 else '')


def apply_mutations(graph, mutations):
    result = graph.clone()
    ids = list(result.node_id)
    positions = {node_key(n): i for i, n in enumerate(ids)}
    if len(positions) != len(ids):
        raise ValueError('duplicate_graph_residue')
    reverse = {v: k for k, v in AA3.items()}
    for mut in mutations:
        residue = (mut['chain'], mut['residue_number'], mut['insertion_code'])
        if residue not in positions:
            raise ValueError(f'mutation_outside_encoded_interface:{mut["annotation"]}')
        i = positions[residue]
        if result.f[i].argmax().item() != amino_acid_to_class(mut['expected_wt']):
            raise ValueError('graph_wt_type_mismatch')
        result.f[i].zero_()
        cls = amino_acid_to_class(mut['mutant'])
        result.f[i, cls] = 1
        result.y[i] = cls
        parts = ids[i].split(':')
        parts[1] = reverse[mut['mutant']]
        ids[i] = ':'.join(parts)
    result.node_id = ids
    return result



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT/'data/manifests/ppb_affinity/structure_manifest.csv')
    p.add_argument('--out', type=Path, default=ROOT/'data/processed/ppb_ppiformer_v2')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'external/PPIformer/weights/masked_modeling.ckpt')
    p.add_argument('--radius', type=float, default=10.)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--limit', type=int, default=0, help='First N rows; 0 means all. Use a separate output for smoke tests.')
    p.add_argument('--audit-only', action='store_true', help='Check data and interface coverage without loading/encoding the model')
    p.add_argument('--reuse-features-from', type=Path, help='Read-only reuse of verified v1 features')
    args = p.parse_args()
    if not math.isfinite(args.radius) or args.radius <= 0 or args.threads < 1 or args.limit < 0:
        p.error('Invalid radius, threads or limit')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/'.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    config = {
        'format_version': 2, 'script_sha256': sha(__file__),
        'audit_code_sha256': sha(Path(__file__).with_name('audit_ppb_residues.py')),
        'sequence_mapping_sha256': sha(Path(__file__).with_name('sequence_mapping.py')),
        'pool_code_sha256': sha(ROOT/'appi/model.py'),
        'manifest_sha256': sha(args.manifest), 'checkpoint_sha256': sha(args.checkpoint),
        'radius_angstrom': args.radius, 'device': args.device, 'limit': args.limit,
        'packages': {n: importlib.metadata.version(n) for n in
                     ['torch','torch-geometric','graphein','equiformer-pytorch','ppiformer']},
        'split_policy': 'published_split_only_external_holdout_exclusion_pending',
        'features': 'residue_embeddings_128_and_symmetric_mean_sum_768; no BSA/contact chemistry scalars yet',
        'mutation_policy': 'fixed_WT_geometry_graph_type_substitution; reject_outside_interface',
        'audit_only': args.audit_only,
        'reuse_features_from': str(args.reuse_features_from.resolve()) if args.reuse_features_from else None,
    }
    config_path = args.out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('Output config changed. Select a new --out directory; existing cache is retained.')
    atomic_json(config_path, config)
    old_config = None
    if args.reuse_features_from:
        args.reuse_features_from = args.reuse_features_from.resolve()
        old_config = json.loads((args.reuse_features_from/'config.json').read_text())
        validate_reuse(old_config, config)
    if not args.audit_only:
        model = PPIformer.load_from_checkpoint(args.checkpoint, map_location='cpu').to(args.device).eval()
        model.requires_grad_(False)
        cfg = OmegaConf.create(model.hparams['run_cfg'])
        transforms = [t for t in cfg.val_dataloader.pretransform
                      if t['_target_'] != 'ppiformer.data.transforms.DDGLabelPretransform']
        if len(transforms) != 1 or transforms[0]['_target_'] != 'ppiformer.data.transforms.PDBToPyGPretransform':
            raise RuntimeError('Unexpected checkpoint preprocessing; requires explicit review')
        transform = hydra.utils.instantiate(transforms[0])
    with args.manifest.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError('Empty manifest')
    # Apply --limit before grouping: it still means the first N input rows.
    # csv_line remains the original identity; split and labels are unchanged.
    rows.sort(key=row_base_key)
    # One base graph cached in RAM at a time; complete embeddings persist on disk.
    hashes, last_base, base_graph = {}, None, None
    last_path, residues = None, None
    last_interface, selected = None, None
    counts, per_split, reasons = Counter(), {}, Counter()
    reused = set()
    output_fields = output_columns(rows[0])
    with (args.out/'samples.csv').open('w', newline='') as output, (args.out/'errors.jsonl').open('w') as errors:
        writer = csv.DictWriter(output, fieldnames=output_fields)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            result = {**row, 'build_status':'failed','feature_path':'','wt_feature_path':'','error':'',
                      'quality_warnings':'','training_eligible':False}
            try:
                result['label_pkd'] = label_pkd(row)
                if row['split'] not in ('train','val','test'):
                    raise DataReview('invalid_split')
                if row['structure_status'] != 'matched':
                    raise ValueError('unmatched_structure')
                path = Path(row['structure_path'])
                if str(path) not in hashes:
                    hashes[str(path)] = sha(path)
                if path != last_path:
                    residues = read_residues(path)
                    last_path = path
                muts, warnings = validate_record(row, residues)
                result['quality_warnings'] = ','.join(warnings)
                result['sequence_mapping'] = json.dumps(check_sequences(row, residues, muts), sort_keys=True)
                ligand = sorted(c.strip() for c in row['Ligand Chains'].split(','))
                receptor = sorted(c.strip() for c in row['Receptor Chains'].split(','))
                base = key([hashes[str(path)], ligand, receptor, args.radius])
                variant = key([base, muts])
                feature_path = args.out/'features'/f'{variant}.pt'
                wt_path = args.out/'features'/f'{key([base, []])}.pt'
                if args.audit_only:
                    interface_path = args.out/'interfaces'/f'{base}.pdb'
                    if last_interface != base:
                        selected = interface(path, ligand, receptor, args.radius, interface_path)
                        last_interface = base
                    for m in muts:
                        site = (m['chain'],m['residue_number'],m['insertion_code'])
                        if site not in selected or 'CA' not in residues[site]['atoms']:
                            raise DataReview(f'mutation_outside_encoded_interface:{m["annotation"]}')
                    result['build_status'] = 'audit_ready_review' if warnings else 'audit_ready'
                elif old_config is not None:
                    for label, destination, applied in [('mut',feature_path,muts),('wt',wt_path,[])]:
                        candidate = args.reuse_features_from/'features'/destination.name
                        if candidate.is_file():
                            saved = torch.load(candidate, map_location='cpu', weights_only=False)
                            if (saved['config_sha256'] != key(old_config) or
                                saved['structure_sha256'] != hashes[str(path)] or saved['mutations'] != applied):
                                raise RuntimeError(f'Old feature provenance mismatch: {candidate}')
                            if label == 'mut':
                                feature_path = candidate
                            else:
                                wt_path = candidate
                            reused.add(str(candidate))
                # Also preserve WT provenance if the mutant later fails to encode.
                if not args.audit_only and wt_path.is_file():
                    result['wt_feature_path'] = str(wt_path)
                if not args.audit_only and not (feature_path.exists() and wt_path.exists()):
                    if last_base != base:
                        interface_path = args.out/'interfaces'/f'{base}.pdb'
                        interface(path, ligand, receptor, args.radius, interface_path)
                        base_graph = transform(interface_path)
                        last_base = base
                    try:
                        graph = apply_mutations(base_graph, muts)
                    except ValueError as exc:
                        raise DataReview(str(exc)) from exc
                    for current, destination, applied in [(base_graph, wt_path, []), (graph, feature_path, muts)]:
                        if destination.exists():
                            continue
                        ids = list(current.node_id)
                        chain_order = [n.split(':')[0] for n in ids]
                        if len([c for j,c in enumerate(chain_order) if j==0 or c!=chain_order[j-1]]) != len(set(chain_order)):
                            raise ValueError('noncontiguous_chain_nodes')
                        partner = torch.tensor([c in ligand for c in chain_order], dtype=torch.bool)
                        with torch.inference_mode():
                            h = model(Batch.from_data_list([current]).to(args.device)).detach().cpu()
                            if h.shape != (len(ids),128) or not torch.isfinite(h).all():
                                raise ValueError('invalid_embedding')
                            pooled = symmetric_pool(h, partner)
                        payload = {'h':h, 'partner':partner, 'pooled':pooled,
                                   'node_id':ids, 'coordinates':current.x.cpu(),
                                   'mutations':applied, 'structure_sha256':hashes[str(path)],
                                   'ligand_chains':ligand,'receptor_chains':receptor,
                                   'config_sha256':key(config), 'geometry':'WT_fixed'}
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        temporary = destination.with_suffix('.pt.tmp')
                        torch.save(payload, temporary)
                        temporary.replace(destination)
                        if destination == wt_path:
                            result['wt_feature_path'] = str(wt_path)
                if not args.audit_only:
                    result.update(build_status='encoded_review' if warnings else 'encoded',
                                  feature_path=str(feature_path),wt_feature_path=str(wt_path))
            except DataReview as exc:
                result.update(build_status='needs_review',error=str(exc))
            except Exception as exc:
                result['error'] = str(exc)
                errors.write(json.dumps({'csv_line':row['csv_line'],'complex_id':row['Complex ID'],
                                          'error':str(exc),'traceback':traceback.format_exc()})+'\n')
                errors.flush()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            writer.writerow(result)
            output.flush()
            counts[result['build_status']] += 1
            if result['error']:
                reasons[result['error'].split(':')[0]] += 1
            per_split.setdefault(row['split'],Counter())[result['build_status']] += 1
            if index % 100 == 0 or index == len(rows):
                summary = {'rows_processed':index,'rows_total':len(rows),'counts':dict(counts),
                           'reason_counts':dict(reasons),'by_split':{k:dict(v) for k,v in per_split.items()},
                           'unique_reused_features':len(reused),'config':config,
                           'status':'preflight' if args.audit_only else 'feature_preparation_only_not_leakage_cleared'}
                atomic_json(args.out/'progress.json',summary)
                print(f'{index}/{len(rows)} {dict(counts)} reasons={dict(reasons)}',flush=True)
    atomic_json(args.out/'summary.json', {'rows':len(rows),'counts':dict(counts),
                'by_split':{k:dict(v) for k,v in per_split.items()},'config':config,
                'status':'feature_preparation_only_not_leakage_cleared',
                'source_structure_hashes':hashes, 'reason_counts':dict(reasons),
                'unique_reused_features':len(reused)})
    print(f'Completed. Review {args.out / "summary.json"} and errors.jsonl',flush=True)
    if not args.audit_only and not (counts['encoded'] + counts['encoded_review']):
        raise SystemExit('No samples encoded; inspect errors.jsonl')


if __name__ == '__main__':
    main()
