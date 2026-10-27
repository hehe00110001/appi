"""Verify and evaluate the preregistered A/B/C SKEMPI ablation on common tests."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from appi.residue_model import ResidueAffinityRegressor
from appi.residue_data import collate_graphs
from train_affinity import metrics,digest,json_save
from analyze_group_generalization import analyze,read,stats,write_csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    torch.set_num_threads(4)
    out=ROOT/'outputs/skempi_ablation_v1';out.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/processed/joint_training_complex40_v1'
    base=ROOT/'runs/skempi_ablation_v1'
    labels_path=ROOT/'data/manifests/binder16_labels_user_20261026.csv'
    labels=read(labels_path)
    baseline=ROOT/'outputs/binder16_residue_v1'
    baseline_config=json.loads((baseline/'config.json').read_text())
    assert digest(labels_path)==baseline_config['labels_sha256']
    graphs=torch.load(baseline/'residue_features.pt',map_location='cpu',weights_only=True)
    report={'seed':42,'selection':'Common full absolute validation RMSE only',
            'notes':['Fixed 40% split; all arms same test records and initial state.',
                     'A excludes SKEMPI from supervised training and normalization, but common validation selection includes SKEMPI.',
                     'All arms balanced by source then PDB; no pair endpoint loss. C alone adds balanced delta-pKd loss lambda=0.3.',
                     'Same 4471 absolute draws / 70 updates per epoch; A repeats 1015 rows. Every active record visited each epoch.',
                     'Centered correlations use true test group means for diagnosis only. Macro correlations require n>=5 and variable values.',
                     'External16 labels/results were previously examined, never used here for fitting, calibration, or selection.',
                     'Single seed; descriptive differences do not establish seed robustness.'],
            'labels_sha256':digest(labels_path),'label_policy':baseline_config['label_policy'],'runs':{}}
    groups_out=[];external_rows=[];joined=[];init_hashes=set();normalization_ids=[];manifest_B=None
    metric_errors=[]
    for arm in ['A','B','C']:
        p=base/f'{arm}_seed42'
        c=json.loads((p/'config.json').read_text());m=json.loads((p/'metrics.json').read_text())
        manifest=json.loads((p/'training_manifest.json').read_text())
        assert c['arm']==arm and c['seed']==42 and c['limit']==0 and not m['smoke_test']
        assert (c['width'],c['layers'],c['heads'],c['pair_absolute_weight'],c['epoch_records'])==(128,2,4,0,4471)
        assert c['effective_lambda_diff']==(.3 if arm=='C' else 0)
        for key,path in [('model_sha256','appi/residue_model.py'),('trainer_sha256','scripts/train_skempi_ablation.py'),('ablation_utilities_sha256','appi/skempi_ablation.py')]:
            assert c[key]==digest(ROOT/path)
        assert c['input_hashes']=={q.name:digest(q) for q in sorted(data.glob('*.csv'))}
        assert c['training_manifest_sha256']==hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        assert c['training_counts']=={'absolute':1015 if arm=='A' else 4471,'pairs':3061 if arm=='C' else 0}
        assert all(r['weight']>0 for rows in manifest.values() for r in rows)
        if arm=='A':assert all('SKEMPI' not in r['source'].upper() for r in manifest['absolute'])
        if arm=='B':manifest_B=manifest['absolute']
        if arm=='C':assert manifest_B==manifest['absolute']
        init_hashes.add(c['initial_state_sha256']);normalization_ids.append(c['normalization_record_ids'])
        epochs=[json.loads(q.read_text()) for q in sorted(p.glob('epoch_*.json'))]
        assert min(epochs,key=lambda e:e['val_absolute']['rmse'])['epoch']==m['best_epoch']
        assert all(e['optimizer_steps']==70 and e['absolute_draws']==4471 and e['unique_absolute_records']==c['training_counts']['absolute'] and e['unique_pair_records']==c['training_counts']['pairs'] for e in epochs)
        record={'best_epoch':m['best_epoch'],'end_epoch':epochs[-1]['epoch'],'training_counts':c['training_counts'],
                'internal':m['metrics'],'groups':{},'source_metrics':{},'external':{},
                'timing':json.loads((p/'timing.json').read_text()),'checkpoint_sha256':digest(p/'best.pt')}
        for kind in ['absolute','pairs']:
            key='csv_line' if kind=='absolute' else 'skempi_csv_line'
            meta={r[key]:r for r in read(data/f'{kind}_test.csv')};preds=read(p/f'predictions_{kind}_test.csv')
            assert len(meta)==len(preds)==len({r['source_line'] for r in preds})
            rows=[]
            for r in preds:
                z=meta[r['source_line']];y=float(z['label_pkd' if kind=='absolute' else 'delta_pkd'])
                assert np.isclose(y,float(r['target']),atol=1e-10,rtol=0)
                rows.append(dict(arm=arm,task=kind,source_line=r['source_line'],target=y,prediction=float(r['prediction']),
                                 pdb=z['PDB'].upper() if kind=='absolute' else z['#Pdb'].split('_')[0].upper(),
                                 homology=z['group_id'],source=z['Source Data Set'] if kind=='absolute' else 'SKEMPI pairs'))
            joined.extend(rows)
            independent=stats([r['target'] for r in rows],[r['prediction'] for r in rows])
            for k in ['pearson','spearman','rmse','mae']:
                metric_errors.append(abs(independent[k]-m['metrics'][kind+'_test'][k]))
            for level in ['pdb','homology']:
                summary,gs=analyze(rows,level)
                largest=max(gs,key=lambda g:g['n'])
                remaining=[r for r in rows if r[level]!=largest['group']]
                sensitivity,_=analyze(remaining,level)
                summary['without_largest_group']={'omitted_group':largest['group'],'omitted_n':largest['n'],
                                                  'overall':sensitivity['overall'],'within_group_centered':sensitivity['within_group_centered'],
                                                  'macro_pearson_n5':sensitivity['macro_pearson_n5']}
                record['groups'][kind+'/'+level]=summary
                groups_out.extend(dict(arm=arm,task=kind,level=level,**g) for g in gs)
            for source in sorted({r['source'] for r in rows}):
                sr=[r for r in rows if r['source']==source]
                record['source_metrics'][kind+'/'+source]=stats([r['target'] for r in sr],[r['prediction'] for r in sr])
        state=torch.load(p/'best.pt',map_location='cpu',weights_only=True)
        assert state['epoch']==m['best_epoch']
        weights=state['model']
        model=ResidueAffinityRegressor(*(weights[k] for k in ['feature_mean','feature_scale','target_mean','target_scale']),width=128,layers=2,heads=4)
        model.load_state_dict(weights);model.eval()
        with torch.inference_mode():pred=np.array([model(**collate_graphs(graphs,[r['sample_id']],'cpu')).item() for r in labels])
        assert np.isfinite(pred).all()
        y=np.array([float(r['pkd']) for r in labels])
        for scope in ['all16','without_boundary15']:
            for target in ['all','IL17A','IL23R']:
                ix=[i for i,r in enumerate(labels) if (scope=='all16' or r['sample_id']!='23r-91_il23r') and (target=='all' or r['target']==target)]
                record['external'][scope+'/'+target]=metrics(pred[ix],y[ix])
        external_rows.extend(dict(arm=arm,**r,predicted_pkd=float(v)) for r,v in zip(labels,pred))
        report['runs'][arm]=record
    assert len(init_hashes)==1 and normalization_ids[0]==normalization_ids[1]==normalization_ids[2]
    assert max(metric_errors)<1e-10
    report['verification']={'metric_max_error':max(metric_errors),'shared_initial_state_sha256':next(iter(init_hashes)),
                             'same_B_C_absolute_manifest':True,'all_active_training_rows_seen_each_epoch':True,
                             'analysis_sha256':digest(Path(__file__)),'group_analysis_sha256':digest(ROOT/'scripts/analyze_group_generalization.py')}
    json_save(report,out/'comparison.json');write_csv(out/'group_metrics.csv',groups_out)
    write_csv(out/'external_predictions.csv',external_rows);write_csv(out/'joined_test_predictions.csv',joined)
    lines=['# SKEMPI A/B/C 消融结果（40%固定划分，seed42）','',
           'A：非SKEMPI绝对监督；B：全部来源均衡绝对监督；C：B加均衡差分监督（λ=0.3）。三组均不使用额外配对端点损失。',
           '每轮4471条绝对记录呈现、70次更新；A重复较小数据集。所有模型仅按相同绝对验证RMSE选模。','',
           '|组别|最佳/停止epoch|绝对test Pearson|绝对test RMSE|差分test Pearson|差分test RMSE|外部16 Pearson|IL17A Pearson|IL23R Pearson|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,r in report['runs'].items():
        a=r['internal']['absolute_test'];b=r['internal']['pairs_test'];e=r['external']
        lines.append(f"|{arm}|{r['best_epoch']}/{r['end_epoch']}|{a['pearson']:.3f}|{a['rmse']:.3f}|{b['pearson']:.3f}|{b['rmse']:.3f}|{e['all16/all']['pearson']:.3f}|{e['all16/IL17A']['pearson']:.3f}|{e['all16/IL23R']['pearson']:.3f}|")
    lines+=['','## 预先指定的组内指标（每组等权，至少5条）','','|组别|任务/分组|有效组数|组均Pearson|组均Spearman|组均排序正确率|','|---|---|---:|---:|---:|---:|']
    for arm,r in report['runs'].items():
        for name,g in r['groups'].items():
            lines.append(f"|{arm}|{name}|{g['eligible_rank_groups_n5']}|{g['macro_pearson_n5']:.3f}|{g['macro_spearman_n5']:.3f}|{g['macro_order_accuracy_n5']:.1%}|")
    lines+=['','组内排序只比较实验值相差至少0.1的样本对，预测平局计半分。详细组指标、移除最大组敏感性检查、分来源结果和外部界限样本敏感性结果见comparison.json。',
            '每条训练权重见runs/skempi_ablation_v1/{A,B,C}_seed42/training_manifest.json。三组同初始化、同测试ID与标签、B/C同绝对权重、每轮全覆盖均已验证。',
            'A仅在训练中排除SKEMPI，验证选模仍使用共同验证集。外部16条此前已经查看，只作诊断；单seed不能证明稳定收益。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for arm in ['A','B','C']:
        es=[json.loads(p.read_text()) for p in sorted((base/f'{arm}_seed42').glob('epoch_*.json'))]
        xs=[e['epoch'] for e in es]
        axes[0].plot(xs,[e['train_abs_mse_scaled'] for e in es],label=arm)
        axes[1].plot(xs,[e['val_absolute']['rmse'] for e in es],label=arm)
    axes[0].set_ylabel('Weighted train absolute MSE / shared variance');axes[0].set_yscale('log')
    axes[1].set_ylabel('Common validation absolute RMSE (pKd)')
    for ax in axes:ax.set_xlabel('Epoch (70 updates)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=180);fig.savefig(out/'learning_curves.pdf');plt.close(fig)
    print('Completed: '+str(out/'report.md'),flush=True)


if __name__=='__main__':main()
