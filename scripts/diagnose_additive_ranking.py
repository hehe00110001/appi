"""Evaluate saved best checkpoints on external16 and complete TRAIN ranking, without selection."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from appi.additive_residue_model import AdditiveResidueAffinityRegressor as ResidueAffinityRegressor
from appi.residue_data import collate_graphs
from appi.group_ranking import build_ranking_groups,ranking_metrics
from train_group_ranking import evaluate
from train_affinity import read_tables,digest,json_save,metrics
from analyze_group_generalization import read,write_csv

def main():
 p=argparse.ArgumentParser();p.add_argument('--external-only',action='store_true');p.add_argument('--device',default='cpu');args=p.parse_args()
 torch.set_num_threads(4)
 out=ROOT/'outputs/additive_ranking_v1';base=ROOT/'runs/additive_ranking_v1'
 labels_path=ROOT/'data/manifests/binder16_labels_user_20261026.csv';labels=read(labels_path)
 ref=ROOT/'outputs/binder16_residue_v1';refconfig=json.loads((ref/'config.json').read_text())
 assert digest(labels_path)==refconfig['labels_sha256']
 graphs=torch.load(ref/'residue_features.pt',map_location='cpu',weights_only=True)
 rows=[];results={};diagnostics={}
 if not args.external_only:
  tables=read_tables(ROOT/'data/processed/joint_training_complex40_v1',0)
  cache=ROOT/'data/processed/residue_cache_complex40_v1.pt';cache_hash=digest(cache)
  train_graphs=torch.load(cache,map_location='cpu',weights_only=True)['graphs']
  groups=build_ranking_groups(tables['pairs','train'])
 for seed in [42,43,44]:
  for arm in ['B','C','R']:
   name=f'{arm}_seed{seed}';run=base/name
   state=torch.load(run/'best.pt',map_location='cpu',weights_only=True);c=json.loads((run/'config.json').read_text());m=json.loads((run/'metrics.json').read_text())
   assert state['epoch']==m['best_epoch'] and state['config']==c
   assert c['model_sha256']==digest(ROOT/'appi/additive_residue_model.py')
   w=state['model'];model=ResidueAffinityRegressor(*(w[k] for k in ['feature_mean','feature_scale','target_mean','target_scale']),width=c['width'],layers=c['layers'],heads=c['heads']).to(args.device)
   model.load_state_dict(w);model.eval()
   with torch.inference_mode():pred=np.array([model(**collate_graphs(graphs,[r['sample_id']],args.device)).item() for r in labels])
   assert np.isfinite(pred).all();y=np.array([float(r['pkd']) for r in labels]);scopes={}
   for scope in ['all16','without_boundary15']:
    for target in ['all','IL17A','IL23R']:
     ix=[i for i,r in enumerate(labels) if (scope=='all16' or r['sample_id']!='23r-91_il23r') and (target=='all' or r['target']==target)]
     scopes[scope+'/'+target]=metrics(pred[ix],y[ix])
   results[name]=dict(best_epoch=state['epoch'],checkpoint_sha256=digest(run/'best.pt'),metrics=scopes)
   rows.extend(dict(arm=arm,seed=seed,**r,predicted_pkd=float(v)) for r,v in zip(labels,pred))
   json_save(dict(label_policy=refconfig['label_policy'],labels_sha256=digest(labels_path),runs=results),out/'external16_metrics.json');write_csv(out/'external16_predictions.csv',rows)
   if not args.external_only:
    assert cache_hash==c['cache_sha256']
    values,delta=evaluate(model,train_graphs,tables['pairs','train'],'pairs',args.device,8,1000000)
    rank=ranking_metrics(groups,delta)
    diagnostics[name]=dict(best_epoch=state['epoch'],train_pairs=values,train_ranking=rank,val_pairs=m['metrics']['pairs_val'],val_ranking=m['ranking']['val'],test_pairs=m['metrics']['pairs_test'],test_ranking=m['ranking']['test'])
    json_save(diagnostics,out/'train_generalization_diagnostic.json')
   print('Evaluated',name,scopes['all16/all'],flush=True)
 lines=['# 本轮9个模型的自有16条评估','','均使用绝对验证RMSE选定的best checkpoint。没有使用外部标签拟合、校准或选模。所有数字为三个seed指标均值，不是集成预测。','','|组别|16条Pearson|16条Spearman|16条RMSE|16条MAE|IL17A Pearson (6条)|IL23R Pearson (10条)|排除界限标签15条Pearson|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for arm in ['B','C','R']:
  def avg(scope,key):return np.mean([results[f'{arm}_seed{s}']['metrics'][scope][key] for s in [42,43,44]])
  vals=[avg('all16/all',k) for k in ['pearson','spearman','rmse','mae']]+[avg('all16/IL17A','pearson'),avg('all16/IL23R','pearson'),avg('without_boundary15/all','pearson')]
  lines.append('|'+arm+'|'+'|'.join(f'{v:.3f}' for v in vals)+'|')
 lines+=['','23r-91有界限标签；all16暂按提供的pKd=12计算，without_boundary15排除它。分靶点、逐seed、逐样本数据见JSON/CSV。这16条此前已被查看，仅作外部诊断。']
 (out/'external16_report.md').write_text('\n'.join(lines)+'\n')
 print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
