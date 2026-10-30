"""Frozen checkpoint evaluation: binding detection and measured affinity stay separate."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from appi.structure_context_data import StructureContextData,file_digest
from appi.structure_context_model import StructureContextRegressor
from appi.esm_residue_model import ESMResidueAffinityRegressor
from appi.esm_residue_data import collate_graphs as esm_collate
from train_affinity import metrics,json_save


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def summarize(rows,scores):
    binding=[i for i,r in enumerate(rows) if r['binding_evaluation']=='True' and r['sample_id'] in scores]
    measured=[i for i,r in enumerate(rows) if r['quantitative_primary']=='True' and r['sample_id'] in scores]
    inclusive=[i for i,r in enumerate(rows) if r['kd_molar'] and r['control']=='False' and r['sample_id'] in scores]
    result=dict(total_designs=len(rows),predicted=sum(r['sample_id'] in scores for r in rows),
                unknown_binding=sum(r['binding']=='unknown' for r in rows))
    if binding:
        y=np.array([int(rows[i]['binding']=='true') for i in binding]);p=np.array([scores[rows[i]['sample_id']] for i in binding])
        c=dict(n=len(y),positives=int(y.sum()),prevalence=float(y.mean()),auroc=None,average_precision=None)
        if len(np.unique(y))==2:
            c.update(auroc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)))
        # Fractional credit at the cutoff avoids arbitrary file-order tie breaks.
        for k in (10,20,50):
            if len(y)<k:continue
            cutoff=np.sort(p)[-k];above=p>cutoff;ties=p==cutoff
            c[f'precision_at_{k}']=float((y[above].sum()+(k-above.sum())*y[ties].mean())/k)
        result['binding_detection']=c
    for name,ids in [('affinity_primary',measured),('affinity_including_10um_values',inclusive)]:
        result[name]=metrics([scores[rows[i]['sample_id']] for i in ids],[float(rows[i]['pkd']) for i in ids])
        result[name]['sequence_clusters']=len({rows[i]['binder_sequence_cluster80'] for i in ids})
    return result


def scopes(rows):
    for rnd in ('1','2'):
        rr=[r for r in rows if r['round']==rnd and r['control']=='False']
        yield f'round{rnd}/all',rr
        yield f'round{rnd}/binder_no_train_similarity',[r for r in rr if r['binder_train_similar_30pct_80cov']=='False']
        for category in ('De novo','Hallucination','Optimized binder','Diversified binder','Not mentioned'):
            yield f'round{rnd}/category/{category}',[r for r in rr if r['design_category']==category]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,default=ROOT/'data/processed/adaptyv_egfr_features_v1')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/adaptyv_egfr_v1')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--seeds',type=int,nargs='+',default=[42])
    args=p.parse_args();torch.set_num_threads(4)
    complete=json.loads((args.features/'complete.json').read_text())
    assert file_digest(args.features/'residue_features.pt')==complete['structure_cache_sha256']
    config=json.loads((args.features/'config.json').read_text())
    path=ROOT/'data/processed/adaptyv_egfr_v1/audited_records.csv'
    assert file_digest(path)==config['records_sha256']
    rows=read(path)
    if complete['smoke_test']:rows=[r for r in rows if r['sample_id'] in torch.load(args.features/'residue_features.pt',map_location='cpu',weights_only=True)]
    graphs=torch.load(args.features/'residue_features.pt',map_location='cpu',weights_only=True)
    data=StructureContextData.load(graphs,args.features)
    assert data.provenance['sequence_set_sha256']==complete['sequence_set_sha256']
    names=sorted(graphs);args.out.mkdir(parents=True,exist_ok=True)
    report=dict(smoke_test=complete['smoke_test'],policy='Frozen, preselected checkpoints; no fitting/calibration/threshold tuning. Round2 AF2 primary, Round1 HADDOCK secondary. Unknown/unexpressed never assigned negative affinity.',
                features=complete,records_sha256=file_digest(path),models={},baselines={})
    long=[]
    def record(name,pred,checkpoint):
        assert all(np.isfinite(v) for v in pred.values())
        report['models'][name]=dict(checkpoint_sha256=file_digest(checkpoint),scopes={scope:summarize(rs,pred) for scope,rs in scopes(rows)})
        for r in rows:
            if r['sample_id'] in pred:long.append(dict(model=name,**r,predicted_pkd=pred[r['sample_id']]))
        print('Evaluated '+name,flush=True)
    for seed in args.seeds:
        for arm in ('structure_transfer','fusion_absolute','fusion_transfer','fusion_shuffled'):
            run=ROOT/'runs/structure_context_pilot_v1'/f'{arm}_seed{seed}'
            c=json.loads((run/'config.json').read_text())
            for source,h in c['code_hashes'].items():assert file_digest(ROOT/source)==h
            for key in ('checkpoint_sha256','regression_sha256','model','layer','window','stride','overlap','inference_dtype','storage_dtype','fair_esm_version'):
                assert c['esm']['extraction'][key]==data.provenance['extraction'][key],key
            assert not c['arguments']['limit']
            for choice in ('best','best_absolute'):
                checkpoint=run/'joint'/(choice+'.pt');saved=torch.load(checkpoint,map_location=args.device,weights_only=True)
                assert saved['config']==c
                model=StructureContextRegressor(**c['model']).to(args.device);model.load_state_dict(saved['model']);model.set_stage('joint');model.eval().requires_grad_(False)
                pred={}
                with torch.inference_mode():
                    for ids in data.microbatches([[n] for n in names],4,1000000,arm!='structure_transfer'):
                        selected=[names[i] for i in ids]
                        values=model(**data.collate(selected,args.device,include_memory=arm!='structure_transfer')).cpu().tolist()
                        pred.update(zip(selected,values))
                record(f'{arm}_seed{seed}/{choice}',pred,checkpoint)
                del model
    # Historical full-chain ESM-B sequence reference, with its own saved normalization.
    run=ROOT/'runs/esm2_pilot_v1/seq_B_seed42';checkpoint=run/'best.pt'
    saved=torch.load(checkpoint,map_location=args.device,weights_only=True);c=saved['config'];w=saved['model']
    assert file_digest(ROOT/'appi/esm_residue_model.py')==c['model_sha256']
    old=ESMResidueAffinityRegressor(*(w[k] for k in ('feature_mean','feature_scale','target_mean','target_scale')),
            mode='seq',width=c['width'],layers=c['layers'],heads=c['heads'],modality_dropout=c['modality_dropout']).to(args.device)
    old.load_state_dict(w);old.eval().requires_grad_(False)
    seqgraphs={}
    for name,g in data.manifest['graphs'].items():
        h=torch.zeros(2,1408)
        for i,side in enumerate((True,False)):
            blocks=[data.bank[v['sequence_id']].float() for v in g['chains'].values() if v['partner']==side]
            length=sum(len(b) for b in blocks);h[i,0]=length
            h[i,128:]=(sum(b.sum(0) for b in blocks)/length).half().float()
        seqgraphs[name]=dict(h=h,coordinates=torch.zeros(2,3),partner=torch.tensor([True,False]))
    with torch.inference_mode():
        pred={name:float(old(**esm_collate(seqgraphs,[name],args.device)).item()) for name in names}
    record('historical_seq_B_seed42',pred,checkpoint)
    for metric,sign in [('iptm',1),('plddt',1),('pae_interaction',-1),('esm_pll',1)]:
        pred={r['sample_id']:sign*float(r[metric]) for r in rows if r[metric]}
        # Surrogate units are not pKd: only classification and rank correlations are meaningful.
        values={}
        for scope,rs in scopes(rows):
            s=summarize(rs,pred)
            for k in ('affinity_primary','affinity_including_10um_values'):
                s[k]={key:v for key,v in s[k].items() if key in ('n','pearson','spearman','sequence_clusters')}
            values[scope]=s
        report['baselines'][metric]=values
    with (args.out/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(long[0]));writer.writeheader();writer.writerows(long)
    json_save(report,args.out/'metrics.json')
    print('Completed EGFR external evaluation',flush=True)

if __name__=='__main__':main()
