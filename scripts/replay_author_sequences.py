"""Reproduce the pinned author's sequence extraction offline for cancelled-job WT failures.

Requires external/ppiseq-reference/pdb_processor.py, fetched from commit
f15f4a919e80d6945e84fd5570c5dda7561da322. No PDB download is allowed.
This is diagnostic only, never used to rewrite CSVs or allow training records.
"""
import csv
import json
from pathlib import Path
import sys
import warnings
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'external/ppiseq-reference'))
from pdb_processor import PDBProcessor


def main():
    out=ROOT/'outputs/sequence_diagnosis'
    out.mkdir(parents=True,exist_ok=True)
    processor=PDBProcessor(raw_pdbs_dir=str(out/'offline_raw'))
    def no_download(*args):
        raise RuntimeError('raw_header_unavailable_offline')
    processor.download_pdb=no_download
    warnings.simplefilter('ignore')
    with (ROOT/'data/processed/ppb_ppiformer/samples.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['error'].startswith('observed_sequence') and not r['Mutations']]
    cache,counts={},Counter()
    with (out/'author_replay.jsonl').open('w') as output:
        for i,row in enumerate(rows,1):
            chains=[c.strip() for side in ('Ligand','Receptor') for c in row[side+' Chains'].split(',')]
            k=(row['structure_path'],tuple(chains))
            if k not in cache:
                try:
                    cache[k]=processor.pdb_to_processed_seqs(row['PDB'],row['structure_path'],chains)
                except Exception as exc:
                    cache[k]=str(exc)
            got=cache[k]
            if isinstance(got,str):
                status='author_replay_unavailable';details=[got]
            else:
                details=[]
                for side in ('Ligand','Receptor'):
                    for chain,expected in zip(row[side+' Chains'].split(','),row[side+' Sequences'].split(',')):
                        sequence=got[chain.strip()]
                        kind='exact' if sequence==expected else 'prefix' if sequence.startswith(expected) else 'different'
                        details.append({'chain':chain.strip(),'kind':kind,'author_length':len(sequence),'csv_length':len(expected)})
                status='exact_reproduction' if all(d['kind']=='exact' for d in details) else 'exact_or_prefix_reproduction' if all(d['kind'] in ('exact','prefix') for d in details) else 'different'
            counts[status]+=1
            output.write(json.dumps({'csv_line':row['csv_line'],'PDB':row['PDB'],'status':status,'chains':details})+'\n')
            output.flush()
            if i%100==0: print(i,dict(counts),flush=True)
    (out/'author_replay_summary.json').write_text(json.dumps(dict(counts),indent=2)+'\n')
    print(dict(counts))


if __name__=='__main__':main()
