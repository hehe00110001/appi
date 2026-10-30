"""Read-only follow-up analysis of the completed fixed-seed ESM pilot."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/esm2_pilot_v1'
sys.path.insert(0,str(ROOT))
from analyze_group_ranking import paired_cluster_bootstrap


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cluster_interval(groups, repetitions=10000):
    clusters = sorted({g['homology'] for g in groups})
    values = [[g['order_accuracy'] for g in groups if g['homology'] == c] for c in clusters]
    rng = np.random.default_rng(20261028)
    samples = [np.mean([v for i in rng.integers(len(values), size=len(values)) for v in values[i]])
               for _ in range(repetitions)]
    return dict(groups=len(groups), clusters=len(clusters),
                accuracy=float(np.mean([g['order_accuracy'] for g in groups])),
                descriptive_cluster_interval_95=np.quantile(samples, [.025, .975]).tolist())


def main():
    report = json.loads((OUT/'comparison.json').read_text())
    external = list(csv.DictReader((OUT/'external16_predictions.csv').open()))
    data = ROOT/'data/processed/joint_training_complex40_v1'
    pair_rows = list(csv.DictReader((data/'pairs_test.csv').open()))
    abs_rows = list(csv.DictReader((data/'absolute_test.csv').open()))
    true_delta = np.array([float(r['delta_pkd']) for r in pair_rows])
    output = dict(source_report_sha256=digest(OUT/'comparison.json'),
                  script_sha256=digest(Path(__file__)),
                  zero_delta_baseline=dict(mae=float(abs(true_delta).mean()),
                      rmse=float(np.sqrt((true_delta**2).mean())), sd=float(true_delta.std())), runs={})
    for name, record in report['runs'].items():
        run = ROOT/'runs/esm2_pilot_v1'/name
        config = json.loads((run/'config.json').read_text())
        for key, file in [('model_sha256','appi/esm_residue_model.py'),
                          ('trainer_sha256','scripts/train_esm_ranking.py'),
                          ('data_code_sha256','appi/esm_residue_data.py'),
                          ('interaction_code_sha256','appi/residue_model.py')]:
            assert config[key] == digest(ROOT/file), (name, key)
        assert config['input_hashes'] == {p.name:digest(p) for p in data.glob('*.csv')}
        assert record['checkpoint_sha256'] == digest(run/'best.pt')
        for kind, original in [('absolute',abs_rows),('pairs',pair_rows)]:
            predictions = list(csv.DictReader((run/f'predictions_{kind}_test.csv').open()))
            assert len(predictions) == len(original)
            source = 'csv_line' if kind == 'absolute' else 'skempi_csv_line'
            label = 'label_pkd' if kind == 'absolute' else 'delta_pkd'
            for p, r in zip(predictions, original):
                assert p['source_line'] == r[source] and np.isclose(float(p['target']),float(r[label]))
            y = np.array([float(p['target']) for p in predictions])
            pred = np.array([float(p['prediction']) for p in predictions])
            m = record['result']['metrics'][kind+'_test']
            assert np.isclose(np.corrcoef(y,pred)[0,1],m['pearson'])
            assert np.isclose(abs(pred-y).mean(),m['mae'])
        groups = record['result']['ranking']['test']['groups']
        subsets = dict(all=groups, n5=[g for g in groups if g['n']>=5],
                       n5_without_1EMV=[g for g in groups if g['n']>=5 and g['homology']!='complex40_1EMV'])
        epochs = [json.loads(p.read_text()) for p in sorted(run.glob('epoch_*.json'))]
        assert min(epochs,key=lambda e:e['val_absolute']['rmse'])['epoch'] == record['result']['best_epoch']
        summary = dict(ranking={k:cluster_interval(v) for k,v in subsets.items()},
                       best_rmse_epoch=record['result']['best_epoch'],
                       best_mae_epoch=min(epochs,key=lambda e:e['val_absolute']['mae'])['epoch'],
                       last_epoch=epochs[-1]['epoch'], external_distributions={})
        robust = subsets['n5_without_1EMV']
        summary['n5_without_1EMV_leave_one_homology_out'] = {
            h:float(np.mean([g['order_accuracy'] for g in robust if g['homology']!=h]))
            for h in sorted({g['homology'] for g in robust})}
        for target in ['IL17A','IL23R']:
            rows = [r for r in external if r['mode']==config['mode'] and r['arm']==config['arm'] and r['target']==target]
            y = np.array([float(r['pkd']) for r in rows])
            pred = np.array([float(r['predicted_pkd']) for r in rows])
            summary['external_distributions'][target] = dict(n=len(rows),
                target_mean=float(y.mean()), prediction_mean=float(pred.mean()),
                target_sd=float(y.std()), prediction_sd=float(pred.std()))
        output['runs'][name] = summary
    output['sequence_B_vs_ppi_B'] = {}
    reference = report['reference_ppi_only']['B']['result']['ranking']['test']['groups']
    sequence = report['runs']['seq_B_seed42']['result']['ranking']['test']['groups']
    for scope in ['all','n5','n5_without_1EMV']:
        def keep(g):
            return scope=='all' or g['n']>=5 and (scope=='n5' or g['homology']!='complex40_1EMV')
        output['sequence_B_vs_ppi_B'][scope] = paired_cluster_bootstrap(
            [dict(seed=42,**g) for g in reference if keep(g)],
            [dict(seed=42,**g) for g in sequence if keep(g)])
    (OUT/'followup_analysis.json').write_text(json.dumps(output,indent=2)+'\n')

    names = ['seq_B_seed42','esm_nogeom_B_seed42','esm_graph_B_seed42','fusion_B_seed42',
             'seq_R_seed42','esm_nogeom_R_seed42','esm_graph_R_seed42','fusion_R_seed42']
    labels = ['Sequence B','No distance B','ESM graph B','Fusion B',
              'Sequence R','No distance R','ESM graph R','Fusion R']
    colors = ['#2578a9']*4+['#d57d31']*4
    fig, axes = plt.subplots(1,3,figsize=(16,5),layout='constrained')
    raw = [report['runs'][n]['result']['metrics']['absolute_test']['pearson'] for n in names]
    centered = [report['runs'][n]['groups']['absolute/homology']['without_largest_group']['metrics']['within_group_centered']['pearson'] for n in names]
    pos = np.arange(len(names))
    axes[0].barh(pos-.17,raw,.32,label='Overall absolute test r',color='#2578a9')
    axes[0].barh(pos+.17,centered,.32,label='Within homology, without 1EMV',color='#d57d31')
    axes[0].set(yticks=pos,yticklabels=labels,xlim=(-.3,.85),title='Overall correlation vs within-group signal')
    axes[0].invert_yaxis();axes[0].axvline(0,color='gray',lw=.8);axes[0].legend(fontsize=8,loc='lower right')
    for i,n in enumerate(names):
        v=output['runs'][n]['ranking']['n5_without_1EMV'];mean=v['accuracy']
        lo,hi=v['descriptive_cluster_interval_95']
        axes[1].errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=colors[i],capsize=3)
    axes[1].set(yticks=pos,yticklabels=labels,xlim=(.1,.8),title='Ranking: groups with >=5 variants, no 1EMV',xlabel='Macro order accuracy; descriptive cluster 95% interval')
    axes[1].invert_yaxis();axes[1].axvline(.5,color='gray',ls='--',lw=1)
    rows=[r for r in external if r['mode']=='seq' and r['arm']=='B']
    for target,color in [('IL17A','#d57d31'),('IL23R','#2578a9')]:
        selected=[r for r in rows if r['target']==target]
        axes[2].scatter([float(r['pkd']) for r in selected],[float(r['predicted_pkd']) for r in selected],label=target,color=color,s=40)
    boundary=next(r for r in rows if r['sample_id']=='23r-91_il23r')
    axes[2].annotate('23r-91: pKd >= 12',(12,float(boundary['predicted_pkd'])),xytext=(9.2,6.8),arrowprops=dict(arrowstyle='->'),fontsize=9)
    axes[2].plot([5,12],[5,12],ls='--',color='gray',lw=1)
    axes[2].set(xlabel='Experimental pKd (boundary plotted at 12)',ylabel='Predicted pKd',title='Full-chain sequence B: external16',xlim=(5,12.5),ylim=(5,12.5))
    axes[2].legend()
    fig.suptitle('Frozen ESM2 pilot: seed 42; validation-RMSE selection; no external fitting',fontsize=13)
    fig.savefig(OUT/'followup_diagnostics.png',dpi=180)
    fig.savefig(OUT/'followup_diagnostics.pdf')
    plt.close(fig)
    print('Verified eight runs; wrote followup_analysis.json and diagnostic figures')


if __name__=='__main__':main()
