"""Map full experimental chain sequences onto frozen interface nodes, without labels."""
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import torch
from Bio.Align import PairwiseAligner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_ppb_residues import AA3, read_residues
from prepare_residue_cache import node_key
from train_affinity import read_tables, digest, json_save


@lru_cache(maxsize=20000)
def align_indices(observed, full):
    """Exact subsequence alignment, preferring contiguous gaps; report optimum ties."""
    if observed == full:
        return list(range(len(full))), False
    a = PairwiseAligner()
    a.match_score = 2
    a.mismatch_score = -100000
    a.open_gap_score = -2
    a.extend_gap_score = -.1
    a.target_gap_score = -100000  # Never insert residues absent from full sequence.
    alignments = a.align(full, observed)
    first = alignments[0]
    indices = np.full(len(observed), -1, dtype=int)
    for (s, e), (u, v) in zip(*first.aligned):
        indices[u:v] = np.arange(s, e)
    if (indices < 0).any() or ''.join(full[i] for i in indices) != observed:
        raise ValueError('Structure sequence is not an exact subsequence of CSV sequence')
    try:
        ambiguous = not np.array_equal(first.indices, alignments[1].indices)
    except IndexError:
        ambiguous = False
    return indices.tolist(), ambiguous


@lru_cache(maxsize=4096)
def structure_residues(path):
    return read_residues(Path(path))


def make_record(row, payload, wildtype=False):
    residues = structure_residues(row['structure_path'])
    mapping = json.loads(row['sequence_mapping'])
    changes = {(m['chain'], m['residue_number'], m['insertion_code']): m
               for m in payload['mutations']}
    chains, locations, ties = {}, {}, []
    for side in ('Ligand', 'Receptor'):
        seqs = [s.strip() for s in row[side+' Sequences'].split(',')]
        for chain, seq_index in mapping[side].items():
            keys = sorted(k for k in residues if k[0] == chain)
            observed = ''.join(changes[k]['mutant'] if k in changes else
                               AA3[next(iter(residues[k]['names']))] for k in keys)
            sequence = seqs[seq_index]
            indices, ambiguous = align_indices(observed, sequence)
            if ambiguous:
                ties.append(chain)
            if wildtype:
                sequence = list(sequence)
                for key, index in zip(keys, indices):
                    if key in changes:
                        if sequence[index] != changes[key]['mutant']:
                            raise ValueError('Incorrect mutant sequence mapping')
                        sequence[index] = changes[key]['expected_wt']
                sequence = ''.join(sequence)
            chains[chain] = dict(sequence=sequence, partner=side == 'Ligand')
            locations.update({key: index for key, index in zip(keys, indices)})
    nodes = []
    for i, node in enumerate(payload['node_id']):
        key = node_key(node)
        if bool(payload['partner'][i]) != chains[key[0]]['partner']:
            raise ValueError('Partner mismatch')
        expected = AA3[next(iter(residues[key]['names']))]
        if key in changes and not wildtype:
            expected = changes[key]['mutant']
        if chains[key[0]]['sequence'][locations[key]] != expected:
            raise ValueError('Node sequence identity mismatch')
        nodes.append([key[0], locations[key]])
    return dict(chains=chains, nodes=nodes, alignment_tied_chains=ties,
                structure_path=row['structure_path'], structure_sha256=digest(Path(row['structure_path'])))


def main():
    torch.set_num_threads(2)
    data = ROOT/'data/processed/joint_training_complex40_v1'
    out = ROOT/'data/processed/esm2_complex40_v1'
    out.mkdir(exist_ok=True)
    tables = read_tables(data, 0)
    records = {}
    for split in ('train', 'val', 'test'):
        for number, row in enumerate(tables['absolute', split]):
            path = row['feature_path']
            if path not in records:
                payload = torch.load(path, map_location='cpu', weights_only=True)
                records[path] = make_record(row, payload)
                wt = row['wt_feature_path']
                if wt not in records:
                    records[wt] = make_record(row, payload, wildtype=True)
            if number % 500 == 0:
                print(split, number, 'mapped graphs', len(records), flush=True)
    expected = {r[k] for rows in tables.values() for r in rows
                for k in ('feature_path', 'wt_feature_path', 'mutant_feature_path') if r.get(k)}
    assert expected <= records.keys()
    external_graphs = torch.load(ROOT/'outputs/binder16_residue_v1/residue_features.pt',
                                 map_location='cpu', weights_only=True)
    structures = Path('/home/weizg/wei/fangyh/af3/IL23R_IL17A/all_pdb_files')
    with (structures.parent/'data.csv').open(encoding='utf-8-sig') as f:
        source = list(csv.reader(f))[1:]
    seqs = {r[0].lower()+'_'+r[3].lower(): r for r in source if any(r)}
    for name, payload in external_graphs.items():
        row = dict(structure_path=str(structures/(name+'.pdb')),
                   sequence_mapping=json.dumps({'Ligand': {'A': 0}, 'Receptor': {'B': 0}}))
        row.update({'Ligand Sequences': seqs[name][1].strip(), 'Receptor Sequences': seqs[name][4].strip()})
        records[name] = make_record(row, dict(payload, mutations=[]))
    sequences = {}
    for record in records.values():
        for chain in record['chains'].values():
            seq = chain.pop('sequence')
            key = hashlib.sha256(seq.encode()).hexdigest()
            sequences[key] = seq
            chain['sequence_id'] = key
    summary = dict(graphs=len(records), sequences=len(sequences),
                   ambiguous_graphs=sum(bool(r['alignment_tied_chains']) for r in records.values()),
                   max_sequence_length=max(map(len, sequences.values())),
                   input_hashes={p.name: digest(p) for p in data.glob('*.csv')},
                   source_script_sha256=digest(Path(__file__)),
                   sequence_policy='Full CSV sequences; reverse validated mutations for WT; exact gap alignment; optimum ties audited',
                   external_sequence_table_sha256=digest(structures.parent/'data.csv'))
    json_save(dict(metadata=summary, sequences=sequences, graphs=records), out/'manifest.json')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
