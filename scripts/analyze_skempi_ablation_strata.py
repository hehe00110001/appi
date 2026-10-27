"""Post-hoc mutation-type/direction diagnostics on unchanged A/B/C test predictions."""
import json
from pathlib import Path
import numpy as np
from analyze_group_generalization import read, stats, analyze, write_csv
from train_affinity import digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/skempi_ablation_v1'
DATA=ROOT/'data/processed/joint_training_complex40_v1'


def main():
    meta={r['skempi_csv_line']:r for r in read(DATA/'pairs_test.csv')}
    inputs=read(OUT/'joined_test_predictions.csv')
    result={'analysis_type':'Post-hoc descriptive diagnostics; no training or selection',
            'direction_threshold':0.1,'notes':[
                'Delta-pKd > 0 means stronger binding; < 0 means weaker binding.',
                'Direction recall uses experimental delta >=0.1 or <=-0.1; correct sign of prediction is required; prediction zero is incorrect.',
                'Mutation-type strata are exclusive: single to A, other single, multiple. All alanine substitutions within multiple mutants remain in multiple.',
                'Within-stratum correlations have restricted target ranges and small group counts; do not rank strata by correlation alone.'],
            'runs':{},'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [DATA/'pairs_test.csv',OUT/'joined_test_predictions.csv']},
            'script_sha256':digest(Path(__file__))}
    flat=[]
    for arm in 'ABC':
        rs=[dict(r,target=float(r['target']),prediction=float(r['prediction'])) for r in inputs if r['arm']==arm and r['task']=='pairs']
        assert len(rs)==len(meta)==271
        for r in rs:
            m=meta[r['source_line']];assert np.isclose(r['target'],float(m['delta_pkd']),atol=1e-10,rtol=0)
            tokens=m['Mutation(s)_PDB'].split(',')
            r['mutation_type']='multiple' if len(tokens)>1 else ('single_to_A' if tokens[0].strip().endswith('A') else 'single_other')
        summary={}
        subsets={'all':rs}
        subsets.update({name:[r for r in rs if r['mutation_type']==name] for name in ['single_to_A','single_other','multiple']})
        subsets.update({name:[r for r in rs if rule(r['target'])] for name,rule in {
            'stronger_ge_0.1':lambda y:y>=.1,'weaker_le_minus0.1':lambda y:y<=-.1,'near_zero_abs_lt0.1':lambda y:abs(y)<.1}.items()})
        for name,rows in subsets.items():
            y=np.array([r['target'] for r in rows]);p=np.array([r['prediction'] for r in rows])
            g,_=analyze(rows,'pdb')
            s=stats(y,p)
            s.update(zero_prediction_rmse=float(np.sqrt(np.mean(y*y))),
                     sign_accuracy_excluding_near_zero=float(np.mean(y[abs(y)>=.1]*p[abs(y)>=.1]>0)) if np.any(abs(y)>=.1) else None,
                     eligible_rank_groups_n5=g['eligible_rank_groups_n5'],macro_spearman_n5=g['macro_spearman_n5'],macro_order_accuracy_n5=g['macro_order_accuracy_n5'])
            summary[name]=s;flat.append(dict(arm=arm,stratum=name,**s))
        up=summary['stronger_ge_0.1']['sign_accuracy_excluding_near_zero'];down=summary['weaker_le_minus0.1']['sign_accuracy_excluding_near_zero']
        summary['direction_balanced_accuracy']=(up+down)/2
        result['runs'][arm]=summary
    (OUT/'mutation_strata.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_csv(OUT/'mutation_strata.csv',flat)
    for arm,rs in result['runs'].items():
        print(arm,'balanced direction',rs['direction_balanced_accuracy'])
        for name in ['single_to_A','single_other','multiple','stronger_ge_0.1','weaker_le_minus0.1']:
            s=rs[name];print(name,{k:s[k] for k in ['n','pearson','spearman','rmse','sign_accuracy_excluding_near_zero','macro_spearman_n5','eligible_rank_groups_n5']})


if __name__=='__main__':main()
