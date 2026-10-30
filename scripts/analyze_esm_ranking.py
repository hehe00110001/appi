"""Fixed-seed pilot: validate selection, group sensitivity, external16 and branch reliance."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from appi.esm_residue_model import ESMResidueAffinityRegressor
from appi.esm_residue_data import join_graphs, collate_graphs
from train_affinity import digest, metrics, json_save, read_tables
from train_esm_ranking import evaluate
from appi.group_ranking import build_ranking_groups, ranking_metrics
from analyze_group_ranking import paired_cluster_bootstrap
from analyze_group_generalization import analyze, write_csv
from extract_esm2_features import file_digest


def main():
    torch.set_num_threads(4)
    device = 'cuda'
    out = ROOT/'outputs/esm2_pilot_v1'
    out.mkdir(parents=True, exist_ok=True)
    data = ROOT/'data/processed/joint_training_complex40_v1'
    tables = read_tables(data, 0)
    hashes = {p.name:digest(p) for p in data.glob('*.csv')}
    base = torch.load(ROOT/'data/processed/residue_cache_complex40_v1.pt', map_location='cpu', weights_only=True)['graphs']
    external = torch.load(ROOT/'outputs/binder16_residue_v1/residue_features.pt', map_location='cpu', weights_only=True)
    esm_path = ROOT/'data/processed/esm2_complex40_v1/features.pt'
    esm_hash = file_digest(esm_path)
    esm = torch.load(esm_path, map_location='cpu', weights_only=True)['graphs']
    esm_manifest = json.loads((esm_path.parent/'manifest.json').read_text())
    ambiguous_paths = {path for path,g in esm_manifest['graphs'].items() if g.get('ambiguous_nodes')}
    labels_path = ROOT/'data/manifests/binder16_labels_user_20261026.csv'
    labels = list(csv.DictReader(labels_path.open()))
    assert digest(labels_path) == json.loads((ROOT/'outputs/binder16_residue_v1/config.json').read_text())['labels_sha256']
    test_groups = build_ranking_groups(tables['pairs','test'])
    train_groups = build_ranking_groups(tables['pairs','train'])
    report = dict(selection='Minimum absolute validation RMSE; one seed pilot; fixed test previously inspected',
                  external_policy='Diagnostic only, no selection/calibration; 23r-91 lower-bound label excluded in sensitivity',
                  labels_sha256=digest(labels_path), runs={})
    ext_rows, group_rows, comparison_groups = [], [], {}
    for mode in ('seq', 'esm_graph', 'esm_nogeom', 'fusion'):
        graphs = join_graphs(base, esm, mode)
        external_graphs = join_graphs(external, esm, mode)
        initial = []
        for arm in ('B', 'R'):
            name = f'{mode}_{arm}_seed42'
            run = ROOT/'runs/esm2_pilot_v1'/name
            c = json.loads((run/'config.json').read_text())
            result = json.loads((run/'metrics.json').read_text())
            epochs = [json.loads(p.read_text()) for p in sorted(run.glob('epoch_*.json'))]
            assert c['input_hashes'] == hashes and c['esm_cache_sha256'] == esm_hash
            assert c['limit'] == 0 and not result['smoke_test']
            for key, path in [('model_sha256','appi/esm_residue_model.py'),('data_code_sha256','appi/esm_residue_data.py'),
                              ('trainer_sha256','scripts/train_esm_ranking.py'),('interaction_code_sha256','appi/residue_model.py')]:
                assert c[key] == digest(ROOT/path)
            assert min(epochs, key=lambda e:e['val_absolute']['rmse'])['epoch'] == result['best_epoch']
            assert all(e['unique_absolute_records'] == 4471 and e['optimizer_steps'] == 70 for e in epochs)
            assert all(e['unique_pair_records'] == (3061 if arm == 'R' else 0) for e in epochs)
            state = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
            assert state['config'] == c and state['epoch'] == result['best_epoch']
            initial.append(c['initial_state_sha256'])
            w = state['model']
            model = ESMResidueAffinityRegressor(*(w[k] for k in ('feature_mean','feature_scale','target_mean','target_scale')),
                        mode=mode, width=c['width'], layers=c['layers'], heads=c['heads'], modality_dropout=c['modality_dropout']).to(device)
            model.load_state_dict(w)
            record = dict(result=result, checkpoint_sha256=file_digest(run/'best.pt'), external={}, groups={}, inference_ablations={}, alignment_sensitivity={})
            for kind in ('absolute', 'pairs'):
                rows = tables[kind, 'test']
                measured, predictions = evaluate(model, graphs, rows, kind, device, 8, 1000000)
                for metric in ('rmse','mae','pearson','spearman'):
                    assert np.isclose(measured[metric], result['metrics'][kind+'_test'][metric], atol=1e-6)
                keep = [i for i,r in enumerate(rows) if all(r.get(k) not in ambiguous_paths
                        for k in ('feature_path','mutant_feature_path','wt_feature_path'))]
                record['alignment_sensitivity'][kind] = dict(excluded=len(rows)-len(keep),
                    metrics=metrics([predictions[i] for i in keep],
                                    [float(rows[i]['label_pkd' if kind=='absolute' else 'delta_pkd']) for i in keep]))
                joined = [dict(target=float(r['label_pkd' if kind == 'absolute' else 'delta_pkd']), prediction=p,
                               source=r['Source Data Set'] if kind == 'absolute' else 'SKEMPI pairs',
                               homology=r['group_id'], pdb=r['PDB'] if kind == 'absolute' else r['#Pdb'].split('_')[0])
                          for r,p in zip(rows,predictions)]
                for level in ('pdb','homology'):
                    summary, groups = analyze(joined,level)
                    largest = max(groups,key=lambda r:r['n'])
                    reduced,_ = analyze([r for r in joined if r[level] != largest['group']],level)
                    summary['without_largest_group'] = dict(omitted=largest['group'], metrics=reduced)
                    record['groups'][kind+'/'+level] = summary
                if kind == 'pairs':
                    ranking = ranking_metrics(test_groups,predictions)
                    comparison_groups[mode,arm] = [dict(seed=42,**g) for g in ranking['groups']]
                    group_rows.extend(dict(mode=mode,arm=arm,seed=42,**g) for g in ranking['groups'])
            train_m, train_p = evaluate(model,graphs,tables['pairs','train'],'pairs',device,8,1000000)
            record['train_pairs'] = train_m
            record['train_ranking'] = ranking_metrics(train_groups,train_p)
            with torch.inference_mode():
                pred = np.array([model(**collate_graphs(external_graphs,[r['sample_id']],device)).item() for r in labels])
            target = np.array([float(r['pkd']) for r in labels])
            for scope in ('all16','without_boundary15'):
                for protein in ('all','IL17A','IL23R'):
                    ix = [i for i,r in enumerate(labels) if (scope=='all16' or r['sample_id']!='23r-91_il23r') and (protein=='all' or r['target']==protein)]
                    record['external'][scope+'/'+protein] = metrics(pred[ix],target[ix])
            ext_rows.extend(dict(mode=mode,arm=arm,seed=42,**r,predicted_pkd=float(v)) for r,v in zip(labels,pred))
            if mode == 'fusion':
                for label, mask in [('no_ppi',(0.,1.)),('no_esm',(1.,0.))]:
                    model.evaluation_modality_mask = mask
                    m,p = evaluate(model,graphs,tables['pairs','test'],'pairs',device,8,1000000)
                    record['inference_ablations'][label] = dict(pairs=m,ranking=ranking_metrics(test_groups,p),
                        note='Reliance diagnostic with train-time modality dropout; not causal proof or replacement for separately trained controls')
                model.evaluation_modality_mask = (1.,1.)
            report['runs'][name] = record
            json_save(report,out/'comparison.json')
            write_csv(out/'external16_predictions.csv',ext_rows)
            print('Evaluated',name,flush=True)
        assert initial[0] == initial[1]
    report['paired_ranking_comparisons'] = {}
    report['reference_ppi_only'] = {}
    for arm in ('B','R'):
        reference = ROOT/f'runs/group_ranking_v1/{arm}_seed42'
        config = json.loads((reference/'config.json').read_text())
        assert config['input_hashes'] == hashes
        result = json.loads((reference/'metrics.json').read_text())
        report['reference_ppi_only'][arm] = dict(result=result, checkpoint_sha256=file_digest(reference/'best.pt'),
            note='Previously trained frozen PPIformer baseline; same split/supervision/seed, different input projection')
        comparison_groups['ppi_reference',arm] = [dict(seed=42,**g) for g in result['ranking']['test']['groups']]
    for arm in ('B','R'):
        for before,after in [('esm_nogeom','esm_graph'),('esm_graph','fusion'),('seq','fusion'),('ppi_reference','fusion')]:
            a,b = comparison_groups[before,arm],comparison_groups[after,arm]
            for scope in ('all','at_least5','at_least5_without_1EMV'):
                keep=lambda r: scope=='all' or (r['n']>=5 and (scope!='at_least5_without_1EMV' or r['homology']!='complex40_1EMV'))
                report['paired_ranking_comparisons'][f'{arm}/{after}-{before}/{scope}'] = paired_cluster_bootstrap(list(filter(keep,a)),list(filter(keep,b)))
    json_save(report,out/'comparison.json')
    write_csv(out/'experimental_group_metrics.csv',group_rows)
    lines=['# Frozen ESM2 controlled pilot','','One seed (42); B=balanced absolute, R=B+0.3 difference+0.1 ranking. Selection: absolute validation RMSE. External16 is a previously inspected diagnostic, never a selection set.','',
           '|Run|Best epoch|Test RMSE|Test difference MAE|Macro group order accuracy|External16 r|IL17A r|IL23R r|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,record in report['runs'].items():
        r=record['result'];e=record['external']
        lines.append(f"|{name}|{r['best_epoch']}|{r['metrics']['absolute_test']['rmse']:.3f}|{r['metrics']['pairs_test']['mae']:.3f}|{r['ranking']['test']['macro_order_accuracy']:.1%}|{e['all16/all']['pearson']:.3f}|{e['all16/IL17A']['pearson']:.3f}|{e['all16/IL23R']['pearson']:.3f}|")
    lines+=['','Matched esm_graph vs esm_nogeom isolates distance input with the same interface crop. seq uses full CSV chains. Fusion uses 30% total modality dropout. Single-seed differences are exploratory; inspect >=5-variant groups, exclusion of 1EMV homology, clustered intervals, and target-specific external results before expansion.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
