"""Resolve tied alignments explicitly by averaging candidate node positions, never dropping rows."""
import csv
import json
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
from Bio.Align import PairwiseAligner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from audit_ppb_residues import read_residues, check_row, AA3
from train_affinity import json_save, digest


@lru_cache(maxsize=4096)
def read_structure(path):
    return read_residues(Path(path))


def main():
    out=ROOT/'data/processed/esm2_complex40_v1'
    m=json.loads((out/'manifest.json').read_text())
    rows={r['feature_path']:r for p in (ROOT/'data/processed/joint_training_complex40_v1').glob('absolute_*.csv') for r in csv.DictReader(p.open())}
    a=PairwiseAligner()
    a.match_score=2; a.mismatch_score=-100000
    a.open_gap_score=-2; a.extend_gap_score=-.1; a.target_gap_score=-100000
    report=[]
    for path,r in m['graphs'].items():
        if not r['alignment_tied_chains']:
            continue
        residues=read_structure(r['structure_path'])
        mutations=check_row(rows[path],residues)['mutations'] if path in rows else []
        changes={(v['chain'],v['residue_number'],v['insertion_code']):v for v in mutations}
        ambiguous_nodes=[]
        for c in r['alignment_tied_chains']:
            seq=m['sequences'][r['chains'][c]['sequence_id']]
            keys=sorted(k for k in residues if k[0]==c)
            obs=''.join(changes[k]['mutant'] if k in changes else AA3[next(iter(residues[k]['names']))] for k in keys)
            alignments=a.align(seq,obs)
            if len(alignments)>4096:
                raise ValueError('Too many optimal alignments; manual resolution required')
            maps=[]
            for alignment in alignments:
                positions=np.full(len(obs),-1,dtype=int)
                for (s,e),(u,v) in zip(*alignment.aligned):positions[u:v]=np.arange(s,e)
                assert (positions>=0).all() and ''.join(seq[i] for i in positions)==obs
                maps.append(positions)
            inverse={int(index):i for i,index in enumerate(maps[0])}
            # Reverse mutation must be unambiguous in full CSV sequence.
            for i,k in enumerate(keys):
                if k in changes:
                    assert len({int(v[i]) for v in maps})==1, ('Ambiguous mutation site',path,k)
            for node,(chain,index) in enumerate(r['nodes']):
                if chain!=c:continue
                candidates=sorted({int(v[inverse[index]]) for v in maps})
                if len(candidates)>1:
                    ambiguous_nodes.append(dict(node=node,chain=c,indices=candidates))
        r['ambiguous_nodes']=ambiguous_nodes
        report.append(dict(path=path,structure=r['structure_path'],ambiguous_nodes=len(ambiguous_nodes)))
    for p in (ROOT/'data/processed/joint_training_complex40_v1').glob('pairs_*.csv'):
        for row in csv.DictReader(p.open()):
            wt,mut=(m['graphs'][row[k]] for k in ('wt_feature_path','mutant_feature_path'))
            assert wt['nodes']==mut['nodes'], 'WT/mutant node indices differ'
            assert wt.get('ambiguous_nodes',[])==mut.get('ambiguous_nodes',[]), 'WT/mutant alignment uncertainty differs'
    m['metadata'].update(alignment_audit_sha256=digest(Path(__file__)),
        node_ambiguity_policy='Mean ESM representation over distinct positions from all optimal exact subsequence alignments; mutation sites must be unique',
        graphs_with_ambiguous_nodes=sum(r['ambiguous_nodes']>0 for r in report),
        ambiguous_node_count=sum(r['ambiguous_nodes'] for r in report))
    json_save(m,out/'manifest.json')
    json_save(dict(metadata=m['metadata'],records=report),out/'alignment_audit.json')
    print(json.dumps(m['metadata'],indent=2))


if __name__=='__main__':main()
