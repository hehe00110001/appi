"""External chain similarity and label/structure inventory; no external calibration."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from collections import Counter, defaultdict

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/processed/adaptyv_egfr_v1'

def rows(p):
    with p.open() as f:return list(csv.DictReader(f))

def main():
    records=rows(BASE/'records.csv')
    manifest=json.loads((BASE/'structure_manifest.json').read_text())
    hits=defaultdict(list)
    with (BASE/'training_similarity.tsv').open() as f:
        for line in f:
            q,s,identity,length,qlen,slen,qstart,qend,sstart,send,evalue,bits=line.rstrip().split('\t')
            coverage=(int(qend)-int(qstart)+1)/int(qlen)
            if float(identity)>=30 and coverage>=.8:
                hits[q].append(dict(training_sequence=s,identity=float(identity),query_coverage=coverage,evalue=float(evalue)))
    seq={r['sequence_id']:r['sequence'] for r in records if r['control']=='False'}
    fasta=BASE/'binder_sequences.fasta'
    fasta.write_text(''.join(f'>{k}\n{v}\n' for k,v in sorted(seq.items())))
    cluster_path=BASE/'binder_pair_similarity.tsv'
    subprocess.run(['blastp','-query',str(fasta),'-subject',str(fasta),'-out',str(cluster_path),'-seg','no','-evalue','1e-3',
                    '-max_target_seqs','10000','-outfmt','6 qseqid sseqid pident length qlen slen qstart qend sstart send'],check=True)
    parent={k:k for k in seq}
    def find(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    with cluster_path.open() as f:
        for line in f:
            q,s,pid,length,ql,sl,qs,qe,ss,se=line.rstrip().split('\t')
            if float(pid)>=80 and (int(qe)-int(qs)+1)/int(ql)>=.8 and (int(se)-int(ss)+1)/int(sl)>=.8:
                a,b=find(q),find(s)
                if a!=b:parent[max(a,b)]=min(a,b)
    for r in records:
        r['binder_train_similar_30pct_80cov']=bool(hits[r['sequence_id']])
        r['binder_sequence_cluster80']=find(r['sequence_id']) if r['sequence_id'] in parent else 'control'
        g=manifest['graphs'].get(r['sample_id'])
        r['target_train_similar_30pct_80cov']=bool(g and any(hits[c['sequence_id']] for c in g['chains'].values() if not c['partner']))
    with (BASE/'audited_records.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    result=dict(policy='Operational BLAST screen: >=30% identity, >=80% query-chain coverage, evalue<=1e-3; not proof of no homology or pretraining separation.',
                family_policy='Connected components at >=80% identity and >=80% coverage of both binder sequences; deterministic across rounds.',rounds={})
    for rnd in ('1','2'):
        rs=[r for r in records if r['round']==rnd and r['control']=='False']
        result['rounds'][rnd]=dict(designs=len(rs),sequence_clusters=len({r['binder_sequence_cluster80'] for r in rs}),
          binder_similar_to_train=sum(r['binder_train_similar_30pct_80cov'] for r in rs),
          target_similar_to_train=sum(r['target_train_similar_30pct_80cov'] for r in rs),
          categories=dict(Counter(r['design_category'] for r in rs)),
          missing_category=sum(not r['design_category'] for r in rs),
          classification_with_structure=sum(r['binding_evaluation']=='True' and r['structure_available']=='True' for r in rs),
          regression_with_structure=sum(r['quantitative_primary']=='True' and r['structure_available']=='True' for r in rs))
    result['similar_sequence_hits']=dict(hits)
    (BASE/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='similar_sequence_hits'},indent=2))

if __name__=='__main__':main()
