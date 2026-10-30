"""Summarize the frozen EGFR test with sequence-cluster uncertainty."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from train_affinity import json_save

ROOT=Path(__file__).resolve().parents[1]

def bootstrap(rows,repetitions=1000):
    clusters=sorted({r['binder_sequence_cluster80'] for r in rows})
    if len(clusters)<2:return dict(clusters=len(clusters),intervals={})
    groups=[[r for r in rows if r['binder_sequence_cluster80']==c] for c in clusters]
    rng=np.random.default_rng(20261029)
    values={k:[] for k in ('binding_auroc','binding_average_precision','affinity_mae','affinity_pearson')}
    for _ in range(repetitions):
        sampled=[r for j in rng.integers(len(groups),size=len(groups)) for r in groups[j]]
        br=[r for r in sampled if r['binding_evaluation']=='True']
        y=np.array([r['binding']=='true' for r in br]);p=np.array([float(r['predicted_pkd']) for r in br])
        if len(set(y))==2:
            values['binding_auroc'].append(float(roc_auc_score(y,p)))
            values['binding_average_precision'].append(float(average_precision_score(y,p)))
        qr=[r for r in sampled if r['quantitative_primary']=='True']
        if qr:
            y=np.array([float(r['pkd']) for r in qr]);p=np.array([float(r['predicted_pkd']) for r in qr])
            values['affinity_mae'].append(float(np.mean(abs(y-p))))
            if len(y)>=3 and np.std(y)>1e-10 and np.std(p)>1e-10:values['affinity_pearson'].append(float(np.corrcoef(y,p)[0,1]))
    return dict(clusters=len(clusters),repetitions=repetitions,
        note='Conditional on one EGFR target; resample binder sequence clusters. Does not quantify unseen-target generalization.',
        intervals={k:dict(valid_repetitions=len(v),percentile_95=np.quantile(v,[.025,.975]).tolist() if v else None) for k,v in values.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'outputs/adaptyv_egfr_v1')
    args=p.parse_args()
    result=json.loads((args.out/'metrics.json').read_text());assert not result['smoke_test']
    with (args.out/'predictions.csv').open() as f:rows=list(csv.DictReader(f))
    intervals={}
    for model in result['models']:
        intervals[model]=bootstrap([r for r in rows if r['model']==model and r['round']=='2' and r['control']=='False'])
    json_save(intervals,args.out/'cluster_intervals.json')
    lines=['# EGFR external evaluation','','Frozen checkpoints, no EGFR fitting/calibration. Round2 is the primary structural cohort. Unknown/unexpressed omitted from binding classification; missing KD never imputed. Quantitative primary excludes unresolved 10uM boundary values. This is a known/homologous-target external design test.','',
        '|Model/checkpoint|Predicted/total|Binding n (+)|AUROC|Average precision|KD n|pKd MAE|pKd Pearson|pKd Spearman|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:'NA' if v is None else f'{v:.3f}'
    for name,r in result['models'].items():
        s=r['scopes']['round2/all'];b=s['binding_detection'];a=s['affinity_primary']
        lines.append(f"|{name}|{s['predicted']}/{s['total_designs']}|{b['n']} ({b['positives']})|{fmt(b['auroc'])}|{fmt(b['average_precision'])}|{a['n']}|{fmt(a.get('mae'))}|{fmt(a.get('pearson'))}|{fmt(a.get('spearman'))}|")
    lines+=['','Sequence-cluster intervals are in cluster_intervals.json; strata and boundary sensitivity in metrics.json. Round1 uses a different docking pipeline and is reported separately in metrics.json. Primary and secondary checkpoints stay separate; neither is chosen by external performance.']
    (args.out/'report.md').write_text('\n'.join(lines)+'\n')
    print('EGFR report complete',flush=True)

if __name__=='__main__':main()
