"""Read-only diagnostics for the fixed six runs; no tuning or checkpoint selection."""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def read(path):
    with path.open() as f:return list(csv.DictReader(f))

def stats(pred,y):
    pred,y=np.asarray(pred),np.asarray(y)
    return dict(n=len(y),rmse=float(np.sqrt(np.mean((pred-y)**2))),mae=float(np.mean(abs(pred-y))),
                bias=float(np.mean(pred-y)),prediction_std=float(np.std(pred)),target_std=float(np.std(y)))

def main():
    out=ROOT/'outputs/first_training_analysis';out.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/processed/joint_training_v2';runs=ROOT/'runs/affinity_v1'
    train=read(data/'absolute_train.csv')
    train_mean=np.mean([float(r['label_pkd']) for r in train])
    report={'baseline':{},'runs':{},'source_counts':{}}
    for split in ['train','val','test']:
        rows=read(data/f'absolute_{split}.csv')
        report['source_counts'][split]=dict(Counter(r['Source Data Set'] for r in rows))
    for split in ['val','test']:
        a=read(data/f'absolute_{split}.csv');d=read(data/f'pairs_{split}.csv')
        report['baseline'][split]={'absolute_train_mean':stats([train_mean]*len(a),[float(r['label_pkd']) for r in a]),
                                  'delta_zero':stats([0.]*len(d),[float(r['delta_pkd']) for r in d])}
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for mode,color in [('absolute','#2166ac'),('joint','#b2182b')]:
        for seed in [42,43,44]:
            name=f'{mode}_seed{seed}';p=runs/name
            m=json.loads((p/'metrics.json').read_text())
            epochs=[json.loads(f.read_text()) for f in sorted(p.glob('epoch_*.json'))]
            record={'best_epoch':m['best_epoch'],'epochs':len(epochs),'metrics':m['metrics'],'by_source':{},'diagnostics':{}}
            for split in ['val','test']:
                metadata={r['csv_line']:r for r in read(data/f'absolute_{split}.csv')}
                rows=read(p/f'predictions_absolute_{split}.csv')
                record['by_source'][split]={}
                for source in sorted({r['Source Data Set'] for r in metadata.values()}):
                    selected=[r for r in rows if metadata[r['source_line']]['Source Data Set']==source]
                    record['by_source'][split][source]=stats([float(r['prediction']) for r in selected],[float(r['target']) for r in selected])
                for kind in ['absolute','pairs']:
                    pr=read(p/f'predictions_{kind}_{split}.csv')
                    record['diagnostics'][kind+'_'+split]=stats([float(r['prediction']) for r in pr],[float(r['target']) for r in pr])
            report['runs'][name]=record
            xs=[r['epoch'] for r in epochs]
            axes[0,0].plot(xs,[r['val_absolute']['rmse'] for r in epochs],color=color,alpha=.65,label=mode if seed==42 else None)
            axes[0,1].plot(xs,[r['train_abs_mse_scaled'] for r in epochs],color=color,alpha=.65)
            axes[1,0].plot(xs,[r['val_delta_pkd']['rmse'] for r in epochs],color=color,alpha=.65)
            axes[1,1].plot(xs,[r['train_diff_mse_scaled'] for r in epochs],color=color,alpha=.65)
            axes[0,0].scatter(m['best_epoch'],epochs[m['best_epoch']-1]['val_absolute']['rmse'],color=color,s=25)
    for ax,title in zip(axes.flat,['Absolute validation RMSE (pKd)','Absolute training MSE (scaled)',
                                   'Difference validation RMSE (delta pKd)','Difference training MSE (scaled)']):
        ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axes[0,0].legend();fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)
    (out/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Baselines:',json.dumps(report['baseline'],indent=2))
    for mode in ['absolute','joint']:
        rs=[report['runs'][f'{mode}_seed{s}'] for s in [42,43,44]]
        print(mode,'test by source RMSE', {k:round(float(np.mean([r['by_source']['test'][k]['rmse'] for r in rs])),4) for k in rs[0]['by_source']['test']})
        print(mode,'test diagnostics', {k:{stat:round(float(np.mean([r['diagnostics'][k][stat] for r in rs])),4) for stat in ['bias','prediction_std','target_std']} for k in ['absolute_test','pairs_test']})

if __name__=='__main__':main()
