"""Verify nine preregistered runs; report equal-group ranking and clustered uncertainty."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from appi.group_ranking import build_ranking_groups, ranking_metrics
from train_affinity import digest, json_save
from analyze_group_generalization import analyze, read, stats, write_csv


def paired_cluster_bootstrap(records_a, records_b, repetitions=2000):
    """Bootstrap homology clusters, after averaging each group's seed differences.

    Preserve group-equal estimand by pooling all groups of each sampled cluster.
    Three seeds share tests: they are not independent test observations.
    """
    a = {(r['seed'],r['group']):r for r in records_a}
    b = {(r['seed'],r['group']):r for r in records_b}
    assert a.keys() == b.keys()
    grouped = {}
    for key in a:
        r = a[key]
        grouped.setdefault((r['homology'],r['group']), []).append(b[key]['order_accuracy']-r['order_accuracy'])
    clusters = {}
    for (cluster,group), values in grouped.items():
        clusters.setdefault(cluster, []).append(float(np.mean(values)))
    values = list(clusters.values())
    if len(values)<2:
        return dict(clusters=len(values), interval=None)
    rng=np.random.default_rng(20261027)
    samples=[np.mean([v for i in rng.integers(len(values),size=len(values)) for v in values[i]]) for _ in range(repetitions)]
    return dict(clusters=len(values), experimental_groups=len(grouped),
                mean_difference=float(np.mean([v for group in values for v in group])),
                percentile_interval_95=np.quantile(samples,[.025,.975]).tolist(), repetitions=repetitions,
                note='Descriptive interval on previously examined fixed test; resampling homology clusters, not mutation rows or seeds')


def main():
    data=ROOT/'data/processed/joint_training_complex40_v1'
    base=ROOT/'runs/group_ranking_v1'
    out=ROOT/'outputs/group_ranking_v1'; out.mkdir(parents=True,exist_ok=True)
    report=dict(selection='minimum absolute validation RMSE, identical rule in all arms', runs={})
    all_groups=[]; rank_records={a:[] for a in ['B','C','R']}; joined=[]
    tables={kind:read(data/f'{kind}_test.csv') for kind in ['absolute','pairs']}
    input_hashes={p.name:digest(p) for p in sorted(data.glob('*.csv'))}
    group_definitions=build_ranking_groups(tables['pairs'], .3)
    for seed in [42,43,44]:
        inits=set(); manifests={}; normalizations=[]
        for arm in ['B','C','R']:
            name=f'{arm}_seed{seed}'; run=base/name
            config=json.loads((run/'config.json').read_text())
            result=json.loads((run/'metrics.json').read_text())
            manifest=json.loads((run/'training_manifest.json').read_text())
            assert config['arm']==arm and config['seed']==seed and config['limit']==0 and not result['smoke_test']
            assert config['input_hashes']==input_hashes
            assert config['training_counts']==dict(absolute=4471,pairs=0 if arm=='B' else 3061)
            assert config['effective_lambda_diff']==(0 if arm=='B' else .3)
            assert config['effective_lambda_rank']==(.1 if arm=='R' else 0)
            assert (config['rank_deadband'],config['rank_margin'],config['rank_groups_per_step'],config['rank_pairs_per_group'])==(.3,.25,4,4)
            assert config['training_manifest_sha256']==hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
            for field,file in [('model_sha256','appi/residue_model.py'),('trainer_sha256','scripts/train_group_ranking.py'),
                               ('ranking_code_sha256','appi/group_ranking.py'),('ablation_utilities_sha256','appi/skempi_ablation.py'),
                               ('data_code_sha256','appi/residue_data.py'),('shared_utilities_sha256','scripts/train_affinity.py')]:
                assert config[field]==digest(ROOT/file), (name,field)
            epochs=[json.loads(p.read_text()) for p in sorted(run.glob('epoch_*.json'))]
            assert [e['epoch'] for e in epochs]==list(range(1,len(epochs)+1))
            assert min(epochs,key=lambda e:e['val_absolute']['rmse'])['epoch']==result['best_epoch']
            for e in epochs:
                assert e['optimizer_steps']==70 and e['absolute_draws']==4471 and e['unique_absolute_records']==4471
                assert e['unique_pair_records']==(0 if arm=='B' else 3061)
                assert e['rank_comparison_draws']==(1120 if arm=='R' else 0)
                assert e['unique_rank_groups']==(config['ranking_audit']['train']['eligible_groups'] if arm=='R' else 0)
            inits.add(config['initial_state_sha256']); manifests[arm]=manifest
            normalizations.append(config['normalization_record_ids'])
            record=dict(best_epoch=result['best_epoch'], end_epoch=len(epochs), metrics=result['metrics'],
                        timing=json.loads((run/'timing.json').read_text()), groups={},
                        checkpoint_sha256=digest(run/'best.pt'))
            for kind in ['absolute','pairs']:
                field='csv_line' if kind=='absolute' else 'skempi_csv_line'
                predictions=read(run/f'predictions_{kind}_test.csv')
                pred={r['source_line']:r for r in predictions}
                assert len(pred)==len(predictions)==len(tables[kind])
                rows=[]
                for r in tables[kind]:
                    p=pred[r[field]]; y=float(r['label_pkd' if kind=='absolute' else 'delta_pkd'])
                    assert np.isclose(float(p['target']),y,atol=1e-10,rtol=0)
                    rows.append(dict(arm=arm,seed=seed,task=kind,source_line=r[field],target=y,prediction=float(p['prediction']),
                                     pdb=r['PDB'].upper() if kind=='absolute' else r['#Pdb'].split('_')[0].upper(),
                                     homology=r['group_id'],source=r['Source Data Set'] if kind=='absolute' else 'SKEMPI pairs'))
                recalculated=stats([r['target'] for r in rows],[r['prediction'] for r in rows])
                for k in ['pearson','spearman','rmse','mae']:
                    assert np.isclose(recalculated[k],result['metrics'][kind+'_test'][k],atol=1e-10,rtol=0)
                joined.extend(rows)
                for level in ['pdb','homology']:
                    summary,groups=analyze(rows,level)
                    largest=max(groups,key=lambda g:g['n'])
                    sensitivity,_=analyze([r for r in rows if r[level]!=largest['group']],level)
                    summary['without_largest_group']=dict(omitted_group=largest['group'],omitted_n=largest['n'],
                        overall=sensitivity['overall'],within_group_centered=sensitivity['within_group_centered'])
                    record['groups'][kind+'/'+level]=summary
                    all_groups.extend(dict(arm=arm,seed=seed,task=kind,level=level,**g) for g in groups)
                if kind=='pairs':
                    ranking=ranking_metrics(group_definitions,[r['prediction'] for r in rows])
                    assert ranking==result['ranking']['test']
                    record['experimental_group_ranking']=ranking
                    rank_records[arm].extend(dict(arm=arm,seed=seed,**g) for g in ranking['groups'])
            report['runs'][name]=record
        assert len(inits)==1 and normalizations[0]==normalizations[1]==normalizations[2]
        assert manifests['C']==manifests['R']
        assert manifests['B']['absolute']==manifests['C']['absolute']
    report['ranking_differences']={f'{b}-{a}':paired_cluster_bootstrap(rank_records[a],rank_records[b])
                                   for a,b in [('B','C'),('C','R'),('B','R')]}
    report['verification']='Input/code hashes, common initialization/normalization, common absolute data/weights, C/R pair data/weights, full epoch coverage, selection and prediction metrics checked'
    report['analysis_sha256']=digest(Path(__file__))
    json_save(report,out/'comparison.json')
    write_csv(out/'group_metrics.csv',all_groups)
    write_csv(out/'experimental_group_metrics.csv',[r for rows in rank_records.values() for r in rows])
    write_csv(out/'joined_test_predictions.csv',joined)
    lines=['# 组内排序监督实验结果','',
           'B：均衡绝对监督；C：B + 0.3数值差分；R：C + 0.1组等权排序。当前交互模型不变，三个seed，统一按绝对验证RMSE选模。',
           '排序组按WT图、文献、方法、温度记录和WT测量值定义；同组重复突变图取平均标签。实验差异严格大于0.3 pKd的比较有效，预测平局半分。',
           '','|运行|最佳epoch|绝对test RMSE|差分test MAE|实验组等权排序正确率|实验组等权Spearman|',
           '|---|---:|---:|---:|---:|---:|']
    for name,r in report['runs'].items():
        q=r['experimental_group_ranking']
        rho='NA' if q['macro_spearman'] is None else f"{q['macro_spearman']:.3f}"
        lines.append(f"|{name}|{r['best_epoch']}|{r['metrics']['absolute_test']['rmse']:.3f}|{r['metrics']['pairs_test']['mae']:.3f}|{q['macro_order_accuracy']:.1%}|{rho}|")
    lines += ['', '三个seed使用相同测试记录；随机种子波动不是独立测试不确定性。以下按40%同源组进行配对bootstrap，先对每个实验组的三个seed差值求平均。']
    for name,d in report['ranking_differences'].items():
        interval=d.get('percentile_interval_95')
        if interval:
            lines.append(f"- {name}：排序正确率变化 {d['mean_difference']:+.1%}，95%区间 [{interval[0]:+.1%}, {interval[1]:+.1%}]；{d['clusters']}个同源组。")
    lines += ['', '以上是已查看过的固定测试集上的描述性结果。详细PDB/同源组指标、移除最大组敏感性检查和各运行耗时见comparison.json。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('Verified and saved',out/'report.md')


if __name__=='__main__':
    main()
