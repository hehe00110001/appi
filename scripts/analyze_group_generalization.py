"""Audit existing seed42 test predictions by PDB background and fixed 40% group.

No fitting, checkpoint selection, or label-based changes to the split. Centering
uses test group means ONLY as an error decomposition, never as a usable model.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/group_generalization_v1'
DATA = ROOT / 'data/processed'
RUNS = {
    'original': (DATA / 'joint_training_v2', ROOT / 'runs/affinity_split_comparison_v1/original'),
    'complex40': (DATA / 'joint_training_complex40_v1', ROOT / 'runs/affinity_split_comparison_v1/complex40'),
    'pdb_resplit': (DATA / 'joint_training_resplit_v1', ROOT / 'runs/affinity_residue_v1'),
}


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.ptp(x) < 1e-10 or np.ptp(y) < 1e-10:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def stats(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pearson = correlation(y, p)
    return dict(n=len(y), pearson=pearson,
                spearman=float(spearmanr(y, p).statistic) if pearson is not None else None,
                rmse=float(np.sqrt(np.mean((p-y)**2))), mae=float(np.mean(abs(p-y))),
                bias=float(np.mean(p-y)), target_mean=float(np.mean(y)), prediction_mean=float(np.mean(p)),
                target_std=float(np.std(y)), prediction_std=float(np.std(p)))


def order_accuracy(y, p, threshold=0.1):
    """Within-group pair order; skip truth gaps <0.1 pKd, prediction ties=half."""
    i, j = np.triu_indices(len(y), 1)
    dy, dp = y[i]-y[j], p[i]-p[j]
    mask = np.abs(dy) >= threshold
    product = dy[mask]*dp[mask]
    ties = np.abs(dp[mask]) < 1e-10
    correct = int(np.sum((product > 0) & ~ties))
    n = len(product)
    return n, correct + float(np.sum(ties))/2


def analyze(rows, level):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row[level]].append(row)
    group_stats, yc, pc, raw_y, raw_p = [], [], [], [], []
    total_pairs, total_correct = 0, 0.
    all_y = np.array([r['target'] for r in rows]); all_p = np.array([r['prediction'] for r in rows])
    within_cov, between_cov, within_var = 0., 0., 0.
    for group, rs in sorted(buckets.items()):
        y = np.array([r['target'] for r in rs]); p = np.array([r['prediction'] for r in rs])
        s = stats(y, p); count, correct = order_accuracy(y, p)
        s.update(group=group, pdbs=';'.join(sorted({r['pdb'] for r in rs})),
                 sources=';'.join(sorted({r['source'] for r in rs})),
                 order_pairs=count, order_correct_credit=correct,
                 order_accuracy=correct/count if count else None,
                 zero_prediction_rmse=float(np.sqrt(np.mean(y*y))))
        group_stats.append(s)
        cy, cp = y-y.mean(), p-p.mean()
        within_cov += float(np.sum(cy*cp))/len(rows)
        between_cov += len(rs)*(float(y.mean())-all_y.mean())*(float(p.mean())-all_p.mean())/len(rows)
        within_var += float(np.sum(cy*cy))/len(rows)
        # Exclude singletons AND groups with constant labels from ranking summaries.
        if len(y) >= 3 and np.ptp(y) >= 1e-10:
            yc.extend(cy); pc.extend(cp); raw_y.extend(y); raw_p.extend(p)
        total_pairs += count; total_correct += correct
    eligible = [g for g in group_stats if g['n'] >= 5 and g['pearson'] is not None]
    order_groups = [g for g in group_stats if g['n'] >= 5 and g['order_pairs'] > 0]
    total_cov = float(np.mean((all_y-all_y.mean())*(all_p-all_p.mean())))
    assert np.isclose(total_cov, within_cov+between_cov, atol=1e-12)
    # Exact MSE decomposition into group-mean error and centered residual error.
    mean_error_mse = sum(g['n']*g['bias']**2 for g in group_stats)/len(rows)
    residual_error_mse = sum(sum(((r['prediction']-g['prediction_mean'])-(r['target']-g['target_mean']))**2 for r in buckets[g['group']]) for g in group_stats)/len(rows)
    assert np.isclose(np.mean((all_y-all_p)**2), mean_error_mse+residual_error_mse, atol=1e-12)
    summary = dict(overall=stats(all_y, all_p), groups=len(buckets),
        eligible_rank_groups_n5=len(eligible), eligible_rank_records_n5=sum(g['n'] for g in eligible),
        macro_pearson_n5=float(np.mean([g['pearson'] for g in eligible])) if eligible else None,
        median_pearson_n5=float(np.median([g['pearson'] for g in eligible])) if eligible else None,
        positive_pearson_groups_n5=sum(g['pearson'] > 0 for g in eligible),
        macro_spearman_n5=float(np.mean([g['spearman'] for g in eligible])) if eligible else None,
        centered_subset_n=len(yc), centered_subset_groups=sum(g['n'] >= 3 and g['target_std'] > 1e-10 for g in group_stats),
        raw_same_subset=stats(raw_y, raw_p) if raw_y else None,
        within_group_centered=stats(yc, pc) if yc else None,
        between_group_means=stats([g['target_mean'] for g in group_stats],[g['prediction_mean'] for g in group_stats]),
        within_target_variance_fraction=within_var/float(np.var(all_y)),
        within_covariance=within_cov, between_covariance=between_cov,
        between_covariance_fraction=between_cov/total_cov if abs(total_cov)>1e-12 else None,
        group_mean_error_mse=mean_error_mse, within_residual_error_mse=residual_error_mse,
        group_mean_error_fraction=mean_error_mse/float(np.mean((all_y-all_p)**2)),
        order_pairs=total_pairs, micro_order_accuracy=total_correct/total_pairs if total_pairs else None,
        macro_order_accuracy_n5=float(np.mean([g['order_accuracy'] for g in order_groups])) if order_groups else None,
        zero_prediction_rmse=float(np.sqrt(np.mean(all_y*all_y))))
    return summary, group_stats


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    homology = {}
    for split in ['train','val','test']:
        for r in read(DATA / f'joint_training_complex40_v1/absolute_{split}.csv'):
            pdb = r['PDB'].upper()
            assert pdb not in homology or homology[pdb] == r['group_id']
            homology[pdb] = r['group_id']
    # Sanity check: perfect group offsets can give high overall correlation with
    # completely reversed ordering within each group; decomposition must detect it.
    synthetic = [dict(pdb=str(g), target=float(10*g+i), prediction=float(10*g-i),source='synthetic') for g in range(3) for i in range(5)]
    check, _ = analyze(synthetic, 'pdb')
    assert check['overall']['pearson'] > .9
    assert np.isclose(check['within_group_centered']['pearson'], -1)
    assert check['micro_order_accuracy'] == 0
    result = {'seed':42, 'notes':[
        'PDB is used as a complex-background proxy; multiple chain interfaces in a PDB are pooled.',
        'Homology groups are the fixed complex40 groups, mapped consistently onto all three tests.',
        'Within-group centered correlation uses groups n>=3 with variable labels and subtracts true and predicted group means; diagnostic only, not deployable calibration.',
        'Macro correlations use groups n>=5 with nonconstant labels and predictions; each group receives equal weight.',
        'Order accuracy compares all within-group pairs with absolute true pKd (or delta-pKd) difference >=0.1; predicted ties receive half credit. Pair comparisons are dependent, not independent trials.',
        'Between covariance fraction is signed covariance decomposition, not explained variance or causal attribution.',
        'All metrics are descriptive on already examined test data; no retraining or selection.'], 'runs':{}}
    all_groups, joined = [], []
    hashes = {}
    for variant, (data, runs) in RUNS.items():
        for mode in ['absolute','joint']:
            for kind in ['absolute','pairs']:
                dp = data/f'{kind}_test.csv'; pp = runs/f'{mode}_seed42/predictions_{kind}_test.csv'
                for path in [dp,pp]:hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
                key = 'csv_line' if kind == 'absolute' else 'skempi_csv_line'
                meta = {r[key]:r for r in read(dp)}; predictions = read(pp)
                assert len(meta) == len(predictions) == len({r['source_line'] for r in predictions})
                rows = []
                for r in predictions:
                    m = meta[r['source_line']]
                    pdb = m['PDB'].upper() if kind == 'absolute' else m['#Pdb'].split('_')[0].upper()
                    target = float(m['label_pkd'] if kind == 'absolute' else m['delta_pkd'])
                    assert np.isclose(target, float(r['target']), atol=1e-10, rtol=0)
                    rows.append(dict(variant=variant, mode=mode, task=kind, source_line=r['source_line'], pdb=pdb,
                                     homology=homology[pdb], source=m['Source Data Set'] if kind=='absolute' else 'SKEMPI pairs',
                                     target=target, prediction=float(r['prediction'])))
                joined.extend(rows)
                for level in ['pdb','homology']:
                    summary, groups = analyze(rows, level)
                    name = '/'.join([variant,mode,kind,level]); result['runs'][name] = summary
                    for g in groups:all_groups.append(dict(variant=variant,mode=mode,task=kind,level=level,**g))
                    # Sensitivity: omit the largest test group, without refitting.
                    largest=max(groups,key=lambda g:g['n'])
                    remaining=[r for r in rows if r[level]!=largest['group']]
                    if remaining:
                        without,_=analyze(remaining,level)
                        summary['without_largest_group']={'omitted_group':largest['group'],'omitted_n':largest['n'],
                            'overall':without['overall'],'within_group_centered':without['within_group_centered'],
                            'macro_pearson_n5':without['macro_pearson_n5']}
    result['input_sha256']=hashes
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write_csv(OUT/'group_metrics.csv',all_groups);write_csv(OUT/'joined_test_predictions.csv',joined)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,kind in zip(axes,['absolute','pairs']):
        gs=[g for g in all_groups if g['variant']=='complex40' and g['task']==kind and g['level']=='pdb' and g['mode']=='absolute' and g['n']>=5 and g['pearson'] is not None]
        gs.sort(key=lambda g:-g['n']);labels=[f"{g['group']} (n={g['n']})" for g in gs];x=np.arange(len(gs))
        for offset,mode,color in [(-.18,'absolute','#377eb8'),(.18,'joint','#e66101')]:
            vals={g['group']:g['pearson'] for g in all_groups if g['variant']=='complex40' and g['task']==kind and g['level']=='pdb' and g['mode']==mode}
            ax.bar(x+offset,[vals[g['group']] for g in gs],width=.35,label=mode,color=color)
        ax.set_xticks(x,labels,rotation=65,ha='right');ax.set_ylim(-1,1);ax.axhline(0,color='gray',lw=.8);ax.set_title('Absolute pKd' if kind=='absolute' else 'Mutation delta-pKd');ax.set_ylabel('Within-PDB Pearson');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fixed 40% split, seed42: groups with n >= 5 and variable labels');fig.tight_layout();fig.savefig(OUT/'within_pdb_pearson.png',dpi=180);fig.savefig(OUT/'within_pdb_pearson.pdf');plt.close(fig)
    print('Saved',OUT)
    for key,v in result['runs'].items():
        if key.startswith('complex40/'):
            print(key,json.dumps({k:v[k] for k in ['groups','eligible_rank_groups_n5','macro_pearson_n5','macro_spearman_n5','within_group_centered','between_covariance_fraction','group_mean_error_fraction','micro_order_accuracy','macro_order_accuracy_n5','zero_prediction_rmse']},ensure_ascii=False))


if __name__ == '__main__':
    main()
