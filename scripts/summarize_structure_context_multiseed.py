"""Three-seed reproducibility report; same held-out groups are not independent across seeds."""
import json
from pathlib import Path
import statistics
from analyze_group_ranking import paired_cluster_bootstrap
from train_affinity import json_save

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/structure_context_pilot_v1'

def main():
    report=json.loads((OUT/'comparison.json').read_text())
    arms=['fusion_absolute','fusion_shuffled','structure_transfer','fusion_transfer']
    result=dict(seeds=[42,43,44],policy='Report each seed and sample SD; bootstrap homology clusters after averaging matched seed gains. No test checkpoint selection.',arms={},comparisons={})
    for arm in arms:
        per_seed=[]
        for seed in result['seeds']:
            r=report['runs'][f'{arm}_seed{seed}'];m=r['result']['checkpoints']['best']
            per_seed.append(dict(seed=seed,absolute_mae=m['metrics']['absolute_test']['mae'],difference_mae=m['metrics']['pairs_test']['mae'],
                                 order=m['ranking']['test']['macro_order_accuracy'],order_n5_no1EMV=r['group_sensitivity']['best']['at_least5_without_1EMV'],
                                 external16_pearson=r['external']['best']['all16/all']['pearson']))
        result['arms'][arm]=dict(per_seed=per_seed,summary={k:dict(mean=statistics.mean(r[k] for r in per_seed),sd=statistics.stdev(r[k] for r in per_seed)) for k in per_seed[0] if k!='seed'})
    for before in arms[:-1]:
        for scope in ('all','at_least5','at_least5_without_1EMV'):
            groups=[]
            for arm in (before,'fusion_transfer'):
                g=[dict(seed=s,**g) for s in result['seeds'] for g in report['runs'][f'{arm}_seed{s}']['result']['checkpoints']['best']['ranking']['test']['groups']]
                groups.append([r for r in g if scope=='all' or (r['n']>=5 and (scope!='at_least5_without_1EMV' or r['homology']!='complex40_1EMV'))])
            c=paired_cluster_bootstrap(*groups,repetitions=10000)
            per_seed={}
            for s in result['seeds']:
                a={r['group']:r for r in groups[0] if r['seed']==s}
                per_seed[str(s)]=statistics.mean(r['order_accuracy']-a[r['group']]['order_accuracy'] for r in groups[1] if r['seed']==s)
            c['per_seed_gain']=per_seed;c['positive_seeds']=sum(x>0 for x in per_seed.values())
            result['comparisons'][f'fusion_transfer-{before}/{scope}']=c
    json_save(result,OUT/'multiseed_reproducibility.json')
    lines=['# Three-seed reproducibility','','Primary checkpoints only. Mean ± sample SD across seeds 42/43/44.','','|Arm|Absolute MAE|Difference MAE|All-group order|>=5, no1EMV order|','|---|---:|---:|---:|---:|']
    for arm,r in result['arms'].items():
        x=r['summary'];fmt=lambda k,pct=False:f"{x[k]['mean']*(100 if pct else 1):.3f} ± {x[k]['sd']*(100 if pct else 1):.3f}"+('%' if pct else '')
        lines.append(f"|{arm}|{fmt('absolute_mae')}|{fmt('difference_mae')}|{fmt('order',True)}|{fmt('order_n5_no1EMV',True)}|")
    (OUT/'multiseed_reproducibility.md').write_text('\n'.join(lines)+'\n')
    print('Three-seed report complete')

if __name__=='__main__':main()
