"""Offline evidence for WT CSV/ATOM discrepancies; never changes labels or coordinates."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from audit_ppb_residues import AA3, read_residues


def subsequence_positions(observed, full):
    result, start = [], 0
    for aa in observed:
        pos = full.find(aa, start)
        if pos < 0:
            return None
        result.append(pos)
        start = pos + 1
    return result


def seqres_chains(path):
    seqs = defaultdict(str)
    for line in path.read_text().splitlines():
        if line.startswith('SEQRES'):
            seqs[line[11]] += ''.join(AA3.get(x, 'X') for x in line[19:70].split())
    return dict(seqs)


def diagnose_chain(observed, expected, seqres):
    detail = {'atom_length':len(observed),'csv_length':len(expected),'seqres_length':len(seqres)}
    if observed and subsequence_positions(observed, expected) is not None:
        return {**detail,'status':'original_check_pass'}
    if not seqres or 'X' in seqres or 'X' in observed:
        return {**detail,'status':'missing_or_noncanonical_sequence'}
    starts = [i for i in range(len(seqres)) if seqres.startswith(expected,i)] if expected else []
    # Strong evidence only: a unique exact CSV segment, equal CSV/ATOM lengths,
    # same first 10 residues, and zero substitutions in the ATOM-to-SEQRES mapping.
    if len(starts) == 1 and len(observed) == len(expected) and len(observed) >= 20:
        start = starts[0]
        positions = subsequence_positions(observed, seqres[start:])
        if (positions is not None and observed[:10] == expected[:10]
                and positions[0] == 0 and positions[-1]+1 > len(expected)):
            # Require a unique order-preserving mapping by comparing greedy left/right maps.
            reverse = subsequence_positions(observed[::-1], seqres[start:][::-1])
            rightmost = [len(seqres[start:])-1-i for i in reverse[::-1]]
            if positions == rightmost:
                return {**detail,'status':'seqres_prefix_truncation_evidence',
                        'seqres_start_0based':start,'reconstructed_span_length':positions[-1]+1,
                        'internal_missing':positions[-1]+1-len(observed),
                        'observed_residues_after_csv_end':sum(i>=len(expected) for i in positions)}
            return {**detail,'status':'prefix_pattern_ambiguous_alignment'}
    return {**detail,'status':'unresolved_sequence_difference'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,default=Path('data/processed/ppb_ppiformer/samples.csv'))
    p.add_argument('--out',type=Path,default=Path('outputs/sequence_diagnosis'))
    args=p.parse_args()
    with args.samples.open() as f:
        rows=[r for r in csv.DictReader(f) if r['error'].startswith('observed_sequence_not_in_csv') and not r['Mutations']]
    args.out.mkdir(parents=True,exist_ok=True)
    cache, counts, chains, results = {},Counter(),Counter(),[]
    for i,row in enumerate(rows,1):
        path=Path(row['structure_path'])
        if str(path) not in cache:
            cache[str(path)] = (read_residues(path),seqres_chains(path))
        residues,seqres=cache[str(path)]
        details=[]
        for side in ('Ligand','Receptor'):
            for chain, expected in zip(row[side+' Chains'].split(','),row[side+' Sequences'].split(',')):
                chain=chain.strip()
                observed=''.join(AA3.get(next(iter(r['names'])),'X') if len(r['names'])==1 else 'X'
                                 for k,r in sorted(residues.items()) if k[0]==chain)
                d=diagnose_chain(observed,expected.strip(),seqres.get(chain,''))
                d.update(chain=chain,side=side)
                chains[d['status']]+=1;details.append(d)
        statuses={d['status'] for d in details}
        supported=statuses <= {'original_check_pass','seqres_prefix_truncation_evidence'}
        status='all_chain_differences_explained' if supported else 'needs_further_review'
        counts[status]+=1
        results.append({'csv_line':row['csv_line'],'PDB':row['PDB'],'source':row['Source Data Set'],
                        'split':row['split'],'status':status,'chains':details})
        if i%100==0: print(i,len(rows),dict(counts),flush=True)
    with (args.out/'rows.jsonl').open('w') as f:
        for result in results:f.write(json.dumps(result)+'\n')
    report={'input_sha256':hashlib.sha256(args.samples.read_bytes()).hexdigest(),
            'rows':len(rows),'row_counts':dict(counts),'chain_counts':dict(chains),
            'scope':'WT only; evidence classification, not permission to change constructs or labels'}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
