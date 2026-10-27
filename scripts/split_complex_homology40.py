"""PPiSeq-style concatenated-complex MMseqs2 clustering, with no record deletion.

Reference: Proteinea/ppiseq data_postprocessing.ipynb, commit
f15f4a919e80d6945e84fd5570c5dda7561da322. Change identity to 0.4;
retain author concatenation and bidirectional 0.8 coverage, cluster mode 1.
Differences: retain only existing eligible records; preserve initial assignments
where practical; count-balanced assignment; all-vs-all audit joins missed edges.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from resplit_joint_training import (load_tables, make_groups, group_vectors, export_rows,
                                   validate_export, write_csv, sha, SPLITS, KINDS, record_id)


class Union:
    def __init__(self,n):self.parent=list(range(n))
    def find(self,i):
        while self.parent[i]!=i:
            self.parent[i]=self.parent[self.parent[i]];i=self.parent[i]
        return i
    def join(self,i,j):
        a,b=self.find(i),self.find(j);self.parent[max(a,b)]=min(a,b)


def combined_sequence(row):
    # Exactly the author's order: ligand chains in supplied order, then receptor.
    seq=''.join(
        ''.join(row[k].replace(',','').split()) for k in ['Ligand Sequences','Receptor Sequences'])
    if not seq or not set(seq)<=set('ACDEFGHIKLMNPQRSTVWYX'):
        raise ValueError('Invalid concatenated sequence')
    return seq


def assign_preserving(groups, fractions, seed=42):
    vectors,columns,weights=group_vectors(groups)
    # Affinity values never enter the objective. A modest movement term favors
    # preserving the first-round split when alternatives have similar balance.
    targets=np.asarray(fractions)[:,None]*vectors.sum(0)[None,:]
    inv=weights[None,:]/np.maximum(targets,1.)**2
    movement=np.array([[sum(k=='absolute' and r['split']!=s for k,r in g['records']) for s in SPLITS] for g in groups],float)
    movement*=.15/sum(vectors[:,0])
    assignments=np.argmin(movement,axis=1)
    counts=np.stack([vectors[assignments==s].sum(0) for s in range(3)])
    rng=np.random.default_rng(seed)
    for _ in range(30):
        improved=False
        for i in rng.permutation(len(groups)):
            src=assignments[i];v=vectors[i]
            remove=(inv[src]*(-2*(counts[src]-targets[src])*v+v*v)).sum()
            costs=remove+(inv*(2*(counts-targets)*v+v*v)).sum(1)+movement[i]-movement[i,src]
            costs[src]=0;dst=int(np.argmin(costs))
            if costs[dst]<-1e-12:
                counts[src]-=v;counts[dst]+=v;assignments[i]=dst;improved=True
        for i in rng.permutation(len(groups)):
            src=assignments[i];js=np.flatnonzero(assignments!=src)
            if not len(js):continue
            dst=assignments[js];d=vectors[js]-vectors[i]
            costs=(2*d*(inv[src]*(counts[src]-targets[src])-inv[dst]*(counts[dst]-targets[dst]))+d*d*(inv[src]+inv[dst])).sum(1)
            costs+=movement[i,dst]-movement[i,src]+movement[js,src]-movement[js,dst]
            best=int(np.argmin(costs))
            if costs[best]<-1e-12:
                j=js[best];s=assignments[j];counts[src]+=d[best];counts[s]-=d[best]
                assignments[i],assignments[j]=s,src;improved=True
        if not improved:break
    if (counts[:,:2]<=0).any():raise ValueError('Empty absolute or pair split')
    return assignments,counts,columns,targets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/processed/joint_training_v2')
    p.add_argument('--out',type=Path,default=ROOT/'data/processed/joint_training_complex40_v1')
    p.add_argument('--work',type=Path,default=ROOT/'outputs/homology40_protocol/mmseqs')
    p.add_argument('--mmseqs',default='/home/weizg/wei/soft/MMseqs2/build_gpu/bin/mmseqs')
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    if args.out.exists():raise ValueError('Output exists; refusing to overwrite split')
    args.work.mkdir(parents=True,exist_ok=True)
    tables,hashes=load_tables(args.data);base=make_groups(tables)
    absolute=tables['absolute'];ids={r['csv_line']:i for i,r in enumerate(absolute)};u=Union(len(absolute))
    for g in base:
        members=[ids[r['csv_line']] for k,r in g['records'] if k=='absolute']
        for i in members[1:]:u.join(members[0],i)
    fasta=args.work/'complexes.fasta'
    fasta.write_text(''.join(f">r{r['csv_line']}\n{combined_sequence(r)}\n" for r in absolute))
    version=subprocess.check_output([args.mmseqs,'version'],text=True).strip()
    commands=[
        [args.mmseqs,'easy-cluster',str(fasta),str(args.work/'complex40'),str(args.work/'cluster_tmp'),
         '--min-seq-id','0.4','-c','0.8','--cov-mode','0','--cluster-mode','1',
         '--single-step-clustering','1','--remove-tmp-files','1','--threads',str(args.threads),
         '--split-memory-limit','12G'],
        [args.mmseqs,'easy-search',str(fasta),str(fasta),str(args.work/'all_hits.tsv'),str(args.work/'search_tmp'),
         '--min-seq-id','0.4','-c','0.8','--cov-mode','0','--alignment-mode','3',
         '--max-seqs',str(len(absolute)),'--threads',str(args.threads),'--split-memory-limit','12G',
         '--format-output','query,target,fident,qcov,tcov,evalue','--remove-tmp-files','1']]
    for i,command in enumerate(commands):
        print('Running '+command[1],flush=True)
        with (args.work/f'command_{i}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    cluster_reps=set()
    with (args.work/'complex40_cluster.tsv').open() as f:
        for line in f:
            a,b=line.strip().split('\t');cluster_reps.add(a);u.join(ids[a[1:]],ids[b[1:]])
    edges=[]
    with (args.work/'all_hits.tsv').open() as f:
        for line in f:
            a,b,identity,qcov,tcov,evalue=line.strip().split('\t')
            if a==b:continue
            # MMseqs2 applies thresholds at alignment precision; TSV is rounded.
            if min(float(qcov),float(tcov))<.799 or float(identity)<.399:raise ValueError('Unexpected MMseqs2 hit')
            i,j=ids[a[1:]],ids[b[1:]];u.join(i,j);edges.append((i,j))
    groups_by_root=defaultdict(list)
    for group in base:
        member=next(r for k,r in group['records'] if k=='absolute')
        groups_by_root[u.find(ids[member['csv_line']])].append(group)
    groups=[]
    for components in groups_by_root.values():
        pdbs=sorted({pdb for g in components for pdb in g['pdbs']})
        records=[record for g in components for record in g['records']]
        groups.append(dict(group_id='complex40_'+pdbs[0],pdbs=pdbs,records=records))
    groups.sort(key=lambda g:g['group_id'])
    assignments,counts,columns,targets=assign_preserving(groups,[.85,.075,.075])
    exported=export_rows(groups,assignments);audit=validate_export(tables,exported)
    assigned={r['csv_line']:r['split'] for r in exported['absolute']}
    conflicts=sum(assigned[absolute[i]['csv_line']]!=assigned[absolute[j]['csv_line']] for i,j in edges)
    assert conflicts==0
    fractions=counts[:,:2]/counts[:,:2].sum(0)
    if not (.78<=fractions[0,0]<=.90 and all(.04<=v<=.14 for v in fractions[1:,0])):
        raise ValueError('Absolute split proportions outside agreed approximate range: '+str(fractions))
    args.out.mkdir(parents=True)
    for kind in KINDS:
        for split in SPLITS:write_csv(args.out/f'{kind}_{split}.csv',[r for r in exported[kind] if r['split']==split])
    write_csv(args.out/'group_assignments.csv',[dict(group_id=g['group_id'],pdbs=';'.join(g['pdbs']),split=SPLITS[assignments[i]],
              absolute=sum(k=='absolute' for k,r in g['records']),pairs=sum(k=='pairs' for k,r in g['records'])) for i,g in enumerate(groups)])
    report=dict(method='PPiSeq concatenated-complex MMseqs2 connected components at40%, both coverages80%; preserve PDB/features/pairs; retain all records',
                reference='https://github.com/Proteinea/ppiseq/blob/f15f4a919e80d6945e84fd5570c5dda7561da322/data_processing/data_postprocessing.ipynb',
                differences_from_author='40% instead of30%; eligible subset; preserve original assignments where balance permits; all-vs-all search audit; count-only optimization toward85/7.5/7.5',
                input_hashes=hashes,script_sha256=sha(Path(__file__)),mmseqs_version=version,commands=commands,
                mmseqs_clusters=len(cluster_reps),final_groups=len(groups),directed_alignment_edges=len(edges),cross_split_hits=conflicts,
                largest_groups=sorted([dict(id=g['group_id'],pdbs=len(g['pdbs']),absolute=sum(k=='absolute' for k,r in g['records']),pairs=sum(k=='pairs' for k,r in g['records'])) for g in groups],key=lambda g:-g['absolute'])[:10],
                audit=audit,splits={s:{col:int(v) for col,v in zip(columns,counts[i])} for i,s in enumerate(SPLITS)},
                transitions={kind:dict(Counter(r['original_split']+'->'+r['split'] for r in exported[kind])) for kind in KINDS})
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['mmseqs_clusters','final_groups','largest_groups','splits','transitions','cross_split_hits']},indent=2),flush=True)


if __name__=='__main__':main()
