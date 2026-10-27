"""Compare fixed seed42 across three splits, including the common external16 panel."""
import csv
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


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    torch.set_num_threads(4)
    out=ROOT/'outputs/split_comparison_v1';out.mkdir(parents=True,exist_ok=True)
    roots={'original':ROOT/'runs/affinity_split_comparison_v1/original',
           'complex40':ROOT/'runs/affinity_split_comparison_v1/complex40',
           'pdb_resplit':ROOT/'runs/affinity_residue_v1'}
    data={'original':ROOT/'data/processed/joint_training_v2',
          'complex40':ROOT/'data/processed/joint_training_complex40_v1',
          'pdb_resplit':ROOT/'data/processed/joint_training_resplit_v1'}
    common={kind:set.intersection(*[{r['csv_line' if kind=='absolute' else 'skempi_csv_line'] for r in read(p/f'{kind}_test.csv')} for p in data.values()]) for kind in ['absolute','pairs']}
    labels_path=ROOT/'data/manifests/binder16_labels_user_20261026.csv'
    labels=read(labels_path)
    baseline=ROOT/'outputs/binder16_residue_v1'
    baseline_config=json.loads((baseline/'config.json').read_text())
    assert digest(labels_path)==baseline_config['labels_sha256']
    graphs=torch.load(baseline/'residue_features.pt',map_location='cpu',weights_only=True)
    y=np.array([float(r['pkd']) for r in labels]);ids=[r['sample_id'] for r in labels]
    report={'seed':42,'common_test_counts':{k:len(v) for k,v in common.items()},'runs':{},
            'comparison_note':'Internal tests differ; external16 is the same panel, already examined earlier. No fitting or selection on external labels.',
            'label_policy':baseline_config['label_policy']}
    all_predictions=[]
    for variant,path in roots.items():
        for mode in ['absolute','joint']:
            name=variant+'/'+mode;p=path/f'{mode}_seed42'
            c=json.loads((p/'config.json').read_text());m=json.loads((p/'metrics.json').read_text())
            assert c['seed']==42 and c['limit']==0 and not m['smoke_test']
            assert (c['width'],c['layers'],c['heads'],c['pair_absolute_weight'])==(128,2,4,1)
            assert c['lambda_diff']==(0 if mode=='absolute' else .3)
            assert c['model_sha256']==digest(ROOT/'appi/residue_model.py')
            assert c['trainer_sha256']==digest(ROOT/'scripts/train_residue_affinity.py')
            assert c['input_hashes']=={q.name:digest(q) for q in sorted(data[variant].glob('*.csv'))}
            state=torch.load(p/'best.pt',map_location='cpu',weights_only=True)
            assert state['epoch']==m['best_epoch']
            weights=state['model']
            model=ResidueAffinityRegressor(*(weights[k] for k in ['feature_mean','feature_scale','target_mean','target_scale']),width=128,layers=2,heads=4)
            model.load_state_dict(weights);model.eval()
            with torch.inference_mode():pred=np.array([model(**collate_graphs(graphs,[n],'cpu')).item() for n in ids])
            assert np.isfinite(pred).all()
            external={}
            for scope in ['all16','without_boundary15']:
                for target in ['all','IL17A','IL23R']:
                    ix=[i for i,r in enumerate(labels) if (scope=='all16' or r['sample_id']!='23r-91_il23r') and (target=='all' or r['target']==target)]
                    external[scope+'/'+target]=metrics(pred[ix],y[ix])
            common_results={}
            for kind in ['absolute','pairs']:
                rows=[r for r in read(p/f'predictions_{kind}_test.csv') if r['source_line'] in common[kind]]
                common_results[kind]=metrics([float(r['prediction']) for r in rows],[float(r['target']) for r in rows])
            report['runs'][name]=dict(best_epoch=m['best_epoch'],training_counts=c['training_counts'],
                                      internal=m['metrics'],external=external,common_internal_test=common_results,
                                      checkpoint_sha256=digest(p/'best.pt'))
            for r,v in zip(labels,pred):all_predictions.append(dict(variant=variant,mode=mode,**r,predicted_pkd=float(v)))
            print(name,json.dumps(dict(internal=m['metrics'],external=external)),flush=True)
    with (out/'external_predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_predictions[0]));writer.writeheader();writer.writerows(all_predictions)
    json_save(report,out/'comparison.json')
    lines=['# 三种划分、当前残基交互模型、固定 seed42','',
           '原划分和40%拼接复合物同源划分各重新训练两种监督；当前PDB划分复用已完成结果。内部test样本不同，不能仅凭不同test分数判断优劣。共同外部16条标签已经被查看，未用于本次训练、校准或选模。','',
           '| 划分/监督 | 最佳epoch | 绝对test n | 绝对test Pearson | 差分test n | 差分test Pearson | 外部16 Pearson | IL17A Pearson | IL23R Pearson |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in report['runs'].items():
        a=r['internal']['absolute_test'];d=r['internal']['pairs_test'];e=r['external']
        vals=[a['pearson'],d['pearson'],e['all16/all']['pearson'],e['all16/IL17A']['pearson'],e['all16/IL23R']['pearson']]
        lines.append(f"| {name} | {r['best_epoch']} | {a['n']} | {vals[0]:.3f} | {d['n']} | {vals[1]:.3f} | {vals[2]:.3f} | {vals[3]:.3f} | {vals[4]:.3f} |")
    lines+=['',f"三套划分共同属于内部test的记录数：{report['common_test_counts']}。这些交集的指标也保存在comparison.json，但交集可能很小且不具代表性。",'',
            '所有外部指标以用户给定pKd为准：23r-22=7.37，23r-91主分析代入12；排除23r-91界限样本的结果见comparison.json。', '',
            '绝对和差分test的RMSE、MAE、Spearman以及逐条外部预测均已保存。本次为单seed对照，不能据此估计随机种子方差。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('Completed: '+str(out/'report.md'),flush=True)


if __name__=='__main__':main()
