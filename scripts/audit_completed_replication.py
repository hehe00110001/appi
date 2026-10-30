"""Verify saved three-seed/EGFR results without fitting or checkpoint selection."""
import csv
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from appi.structure_context_data import file_digest
from train_affinity import metrics,json_save
from evaluate_adaptyv_egfr import summarize


def read(p):
    with p.open() as f:return list(csv.DictReader(f))

def close_tree(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:close_tree(a[k],b[k])
    elif isinstance(a,(float,int)):
        assert np.isclose(a,b,rtol=1e-10,atol=1e-12),(a,b)
    else:assert a==b,(a,b)


def main():
    out=ROOT/'outputs/structure_context_pilot_v1'
    report=json.loads((out/'comparison.json').read_text())
    multi=json.loads((out/'multiseed_reproducibility.json').read_text())
    checked={}
    for name,record in report['runs'].items():
        run=ROOT/'runs/structure_context_pilot_v1'/name
        config=json.loads((run/'config.json').read_text())
        assert config['arguments']['limit']==0
        for path,h in config['code_hashes'].items():assert file_digest(ROOT/path)==h
        with ThreadPoolExecutor(max_workers=8) as pool:
            epochs=list(pool.map(lambda p:json.loads(p.read_text()),sorted((run/'joint').glob('epoch_*.json'))))
        selected=min(epochs,key=lambda e:(-e['val_ranking']['macro_order_accuracy'],e['val_difference']['mae']))
        absolute=min(epochs,key=lambda e:e['val_absolute']['mae'])
        assert selected['epoch']==record['result']['checkpoints']['best']['epoch']
        assert absolute['epoch']==record['result']['checkpoints']['best_absolute']['epoch']
        for choice in ('best','best_absolute'):
            for kind in ('absolute','pairs'):
                rows=read(run/f'evaluation_{choice}/{kind}_test.csv')
                m=metrics([float(r['prediction']) for r in rows],[float(r['target']) for r in rows])
                expected=record['result']['checkpoints'][choice]['metrics'][kind+'_test']
                for k,v in m.items():assert np.isclose(v,expected[k])
        seed=config['arguments']['seed']
        baseline=report['runs'][f'fusion_absolute_seed{seed}']['result']['checkpoints']['best_absolute']['metrics']['absolute_val']['mae']
        eligible=[e for e in epochs if e['val_absolute']['mae']<=1.05*baseline]
        checked[name]=dict(epochs=len(epochs),selected_epoch=selected['epoch'],minimum_val_mae=absolute['val_absolute']['mae'],
            proposed_5pct_threshold=1.05*baseline,epochs_meeting_proposed_constraint=[e['epoch'] for e in eligible],
            note='Validation-only feasibility diagnostic. No new model selected or test evaluation performed.')
    egfr=ROOT/'outputs/adaptyv_egfr_v1'
    external=json.loads((egfr/'metrics.json').read_text());assert not external['smoke_test']
    allrows=read(egfr/'predictions.csv')
    for name,record in external['models'].items():
        rows=[r for r in allrows if r['model']==name and r['round']=='2']
        assert len(rows)==len({r['sample_id'] for r in rows})==400
        scores={r['sample_id']:float(r['predicted_pkd']) for r in rows}
        measured=summarize(rows,scores);saved=record['scopes']['round2/all']
        close_tree(measured,saved)
    result=dict(verified_runs=len(checked),egfr_models=len(external['models']),round2_prediction_coverage='400/400 each model',runs=checked)
    json_save(result,out/'completion_audit_20261030.json')
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    colors=['#777777','#bf812d','#35978f','#2166ac']
    arms=['fusion_absolute','fusion_shuffled','structure_transfer','fusion_transfer']
    names=['Fusion absolute','Fusion shuffled','Structure true','Fusion true']
    for arm,label,color in zip(arms,names,colors):
        rows=multi['arms'][arm]['per_seed']
        axs[0].plot([r['seed'] for r in rows],[r['difference_mae'] for r in rows],'-o',color=color,label=label)
        axs[1].plot([r['seed'] for r in rows],[100*r['order_n5_no1EMV'] for r in rows],'-o',color=color)
    axs[0].axhline(.8211124,ls='--',color='black',lw=.8,label='Zero delta')
    axs[0].set(title='Mutation error repeats across seeds',ylabel='Test delta pKd MAE',xlabel='Seed',xticks=[42,43,44]);axs[0].legend(fontsize=8)
    axs[1].axhline(50,ls='--',color='black',lw=.8)
    axs[1].set(title='>=5 variants / group, excluding 1EMV',ylabel='Macro order accuracy (%)',xlabel='Seed',xticks=[42,43,44])
    modelnames=[a+'_seed42/best' for a in arms]+['historical_seq_B_seed42']
    auc=[external['models'][m]['scopes']['round2/all']['binding_detection']['auroc'] for m in modelnames]
    axs[2].barh(names+['Historical ESM-B'],auc,color=colors+['#88419d'])
    axs[2].axvline(.5,ls='--',color='black',lw=.8)
    axs[2].set(title='EGFR Round 2: seed42 only',xlabel='Binding AUROC (378 designs)',xlim=(0,1));axs[2].invert_yaxis()
    fig.savefig(out/'replication_egfr_20261030.png',dpi=180);fig.savefig(out/'replication_egfr_20261030.pdf');plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
