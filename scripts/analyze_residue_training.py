"""Audit completed residue-model runs and export diagnostics without training or selecting models."""
import csv,json,hashlib,statistics,math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]; data=root/'data/processed/joint_training_resplit_v1'; runs=root/'runs/affinity_residue_v1'; out=root/'outputs/residue_training_analysis';out.mkdir(exist_ok=True)
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def stat(p,y):
 p=np.asarray(p);y=np.asarray(y)
 return dict(n=len(y),rmse=float(np.mean((p-y)**2)**.5),mae=float(np.mean(abs(p-y))),bias=float(np.mean(p-y)),std_pred=float(np.std(p)),std_target=float(np.std(y)))
tables={(k,s):read(data/f'{k}_{s}.csv') for k in ['absolute','pairs'] for s in ['train','val','test']}
report={'runs':{},'aggregate':{},'baselines':{}}
fig,axes=plt.subplots(2,2,figsize=(11,7))
configs={}
for mode,color in [('absolute','#2878B5'),('joint','#E07A24')]:
 for seed in [42,43,44]:
  name=f'{mode}_seed{seed}';path=runs/name
  c=json.loads((path/'config.json').read_text());m=json.loads((path/'metrics.json').read_text());configs[name]=c
  assert not m['smoke_test'] and c['limit']==0 and c['training_counts']=={'absolute':4461,'pairs':3063}
  for name_,digest in c['input_hashes'].items():assert hashlib.sha256((data/name_).read_bytes()).hexdigest()==digest
  for k,f in [('model_sha256','appi/residue_model.py'),('data_code_sha256','appi/residue_data.py'),('trainer_sha256','scripts/train_residue_affinity.py'),('shared_utilities_sha256','scripts/train_affinity.py')]:assert hashlib.sha256((root/f).read_bytes()).hexdigest()==c[k]
  epochs=[json.loads(p.read_text()) for p in sorted(path.glob('epoch_*.json'))]
  assert [e['epoch'] for e in epochs]==list(range(1,len(epochs)+1))
  assert all(e['optimizer_steps']==70 for e in epochs)
  assert m['best_epoch']==min(epochs,key=lambda e:e['val_absolute']['rmse'])['epoch']
  last=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
  steps={int(s['step']) for s in last['optimizer']['state'].values()}
  assert steps=={len(epochs)*70} and last['stale']==25 and last['epoch']==len(epochs)
  record=dict(best_epoch=m['best_epoch'],epochs=len(epochs),optimizer_steps=list(steps),timing=json.loads((path/'timing.json').read_text()),metrics=m['metrics'],by_group={},by_source={},first=epochs[0],last=epochs[-1])
  for kind in ['absolute','pairs']:
   for split in ['val','test']:
    block=kind+'_'+split;rr=read(path/f'predictions_{block}.csv');key='csv_line' if kind=='absolute' else 'skempi_csv_line';label='label_pkd' if kind=='absolute' else 'delta_pkd'
    meta={r[key]:r for r in tables[kind,split]}
    assert len(rr)==len(meta) and {r['source_line'] for r in rr}==set(meta)
    p=np.array([float(r['prediction']) for r in rr]);y=np.array([float(r['target']) for r in rr]);assert np.all(np.isfinite(p))
    assert all(abs(float(r['target'])-float(meta[r['source_line']][label]))<2e-6 for r in rr)
    assert abs(stat(p,y)['rmse']-m['metrics'][block]['rmse'])<1e-8
    for field,dest in [('group_id','by_group'),('Source Data Set','by_source')]:
     if kind=='pairs' and dest=='by_source':continue
     gs=np.array([meta[r['source_line']][field] for r in rr]);record[dest][block]={g:stat(p[gs==g],y[gs==g]) for g in sorted(set(gs))}
  report['runs'][name]=record
  for ax,vals in zip(axes.flat,[[e['train_abs_mse_scaled'] for e in epochs],[e['val_absolute']['rmse'] for e in epochs],[e['train_diff_mse_scaled'] for e in epochs],[e['val_delta_pkd']['rmse'] for e in epochs]]):ax.plot(range(1,len(epochs)+1),vals,color=color,alpha=.65,label=mode if seed==42 else None)
  print(name,'best/end',m['best_epoch'],len(epochs),'minutes',round(record['timing']['total_seconds']/60,1),flush=True)
 for block in ['absolute_val','absolute_test','pairs_val','pairs_test']:
  rr=[report['runs'][f'{mode}_seed{s}'] for s in [42,43,44]]
  agg={k:{'mean':statistics.mean(r['metrics'][block][k] for r in rr),'seed_sd':statistics.stdev(r['metrics'][block][k] for r in rr)} for k in ['rmse','mae','pearson','spearman','bias','std_pred','std_target']}
  agg['macro_group_mae']=statistics.mean(statistics.mean(g['mae'] for g in r['by_group'][block].values()) for r in rr)
  report['aggregate'][mode+'_'+block]=agg
  print(mode,block,json.dumps(agg),flush=True)
for seed in [42,43,44]:
 a,b=[configs[f'{mode}_seed{seed}'] for mode in ['absolute','joint']]
 for k in ['initial_state_sha256','cache_sha256','input_hashes']:assert a[k]==b[k]
for kind,label in [('absolute','label_pkd'),('pairs','delta_pkd')]:
 mean=np.mean([float(r[label]) for r in tables[kind,'train']])
 for split in ['val','test']:
  y=np.array([float(r[label]) for r in tables[kind,split]])
  report['baselines'][kind+'_'+split]=dict(train_mean=stat(np.full_like(y,mean),y),zero=stat(np.zeros_like(y),y))
for block in ['absolute_test','pairs_test']:
 groups=report['runs']['absolute_seed42']['by_group'][block]
 diffs={g:statistics.mean(report['runs'][f'joint_seed{s}']['by_group'][block][g]['mae']-report['runs'][f'absolute_seed{s}']['by_group'][block][g]['mae'] for s in [42,43,44]) for g in groups}
 report[block+'_group_mae_change_joint_minus_absolute']=diffs
 print(block,'group MAE improved',sum(v<0 for v in diffs.values()),'of',len(diffs),flush=True)
 print('largest groups',sorted([(g,groups[g]['n'],round(diffs[g],4)) for g in groups],key=lambda x:-x[1])[:8],flush=True)
for ax,title in zip(axes.flat,['Absolute training MSE (scaled)','Absolute validation RMSE (pKd)','Difference training MSE (scaled)','Difference validation RMSE (delta pKd)']):
 ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
axes[0,0].legend();fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)
(out/'diagnostics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('All six runs, hashes, prediction exports, selection and optimizer steps verified.',flush=True)
