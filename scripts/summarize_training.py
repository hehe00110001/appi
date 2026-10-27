"""Summarize all fixed seeds; never select a seed by test performance."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--runs',type=Path,default=Path(__file__).resolve().parents[1]/'runs/affinity_v1')
    p.add_argument('--arms',nargs='+',default=['absolute','joint'],
                   help='Run directory prefixes before _seed42/_seed43/_seed44')
    args=p.parse_args()
    report={}
    missing=[]
    for mode in args.arms:
        records=[]
        for seed in [42,43,44]:
            path=args.runs/f'{mode}_seed{seed}'/'metrics.json'
            if not path.exists(): missing.append(str(path));continue
            r=json.loads(path.read_text())
            if r['smoke_test']:raise ValueError('Smoke-test output cannot enter formal summary')
            records.append(r)
        report[mode]={'completed_seeds':[r['seed'] for r in records],'metrics':{}}
        if records:
            for split in records[0]['metrics']:
                values={}
                for metric in ['rmse','mae','pearson','spearman']:
                    v=[r['metrics'][split][metric] for r in records if r['metrics'][split][metric] is not None]
                    values[metric]={'mean':statistics.mean(v) if v else None,
                                    'std':statistics.stdev(v) if len(v)>1 else None,'seeds':len(v)}
                report[mode]['metrics'][split]=values
    report['missing']=missing
    print(json.dumps(report,indent=2))
    if not missing:
        (args.runs/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
