"""Read-only scientific audit of the structure-context pilot's saved predictions.

No checkpoint reselection, test calibration, or training changes. Bootstrap
intervals describe a previously inspected fixed test, with homology as unit.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from appi.structure_context_data import file_digest
from appi.group_ranking import build_ranking_groups, ranking_metrics
from train_affinity import metrics, json_save
from analyze_group_ranking import paired_cluster_bootstrap
from analyze_group_generalization import analyze, write_csv

OUT = ROOT / 'outputs/structure_context_pilot_v1'
BASE = ROOT / 'runs/structure_context_pilot_v1'
DATA = ROOT / 'data/processed/joint_training_complex40_v1'


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    comparison = json.loads((OUT / 'comparison.json').read_text())
    tables = {k: read(DATA / (k + '_test.csv')) for k in ('absolute', 'pairs')}
    hashes = {p.name: file_digest(p) for p in DATA.glob('*.csv')}
    groups = build_ranking_groups(tables['pairs'])
    report = dict(policy='Saved preselected checkpoints only; no calibration. Single seed42. Descriptive homology-cluster bootstrap on previously inspected test.', runs={}, comparisons={})
    ranking_records, prediction_records, external_rows, timeline_rows = {}, {}, [], []
    for name, entry in comparison['runs'].items():
        run = BASE / name
        config = json.loads((run / 'config.json').read_text())
        assert config['input_hashes'] == hashes and config['arguments']['limit'] == 0
        for source, expected in config['code_hashes'].items():
            assert file_digest(ROOT / source) == expected
        for stage, expected_epochs in [('warmup', 0 if config['arguments']['supervision'] == 'absolute' else 30),
                                       ('joint', 130 if config['arguments']['supervision'] == 'absolute' else 100)]:
            epochs = [json.loads(p.read_text()) for p in sorted((run / stage).glob('epoch_*.json'))]
            assert [e['epoch'] for e in epochs] == list(range(1, expected_epochs + 1))
            for e in epochs:
                assert e['optimizer_steps'] == 70
                for task, count in e['unique_records'].items():
                    assert count == (4471 if task == 'absolute' else 3061)
                steps = e['steps']
                row = dict(run=name, stage=stage, epoch=e['epoch'],
                           val_order=e['val_ranking']['macro_order_accuracy'], val_abs_mae=e['val_absolute']['mae'],
                           val_diff_mae=e['val_difference']['mae'], context_fraction=np.mean([s['context_active'] for s in steps]),
                           train_abs_loss=e['losses'].get('absolute'), train_diff_loss=e['losses'].get('pairs'),
                           weight_abs=np.mean([s['weights'][0] for s in steps]) if stage == 'joint' and len(steps[0]['weights']) == 2 else None,
                           weight_diff=np.mean([s['weights'][1] for s in steps]) if len(steps[0]['weights']) == 2 else None)
                timeline_rows.append(row)
            if epochs:
                best = min(epochs, key=lambda e: (-e['val_ranking']['macro_order_accuracy'], e['val_difference']['mae']))
                complete = json.loads((run / stage / 'complete.json').read_text())
                assert file_digest(run / stage / 'best.pt') == complete['best_sha256']
                if stage == 'joint':
                    assert best['epoch'] == entry['result']['checkpoints']['best']['epoch']
                    assert min(epochs, key=lambda e: e['val_absolute']['mae'])['epoch'] == entry['result']['checkpoints']['best_absolute']['epoch']
        output = dict(checkpoints={}, shuffle=config['shuffle'], gradient=entry['gradient_diagnostics'])
        for choice, result in entry['result']['checkpoints'].items():
            detail = {}
            for kind, rows in tables.items():
                csvrows = read(run / ('evaluation_' + choice) / (kind + '_test.csv'))
                assert len(csvrows) == len(rows)
                predicted = {r['source_line']: r for r in csvrows}
                assert len(predicted) == len(rows)
                joined = []
                for r in rows:
                    p = predicted[r['csv_line' if kind == 'absolute' else 'skempi_csv_line']]
                    y = float(r['label_pkd' if kind == 'absolute' else 'delta_pkd'])
                    assert abs(float(p['target']) - y) < 1e-10
                    joined.append(dict(target=y, prediction=float(p['prediction']), homology=r['group_id'],
                                       pdb=r['PDB'].upper() if kind == 'absolute' else r['#Pdb'].split('_')[0].upper(),
                                       source=r['Source Data Set'] if kind == 'absolute' else 'SKEMPI pairs'))
                m = metrics([r['prediction'] for r in joined], [r['target'] for r in joined])
                for k in ('mae', 'rmse', 'pearson', 'spearman'):
                    assert np.isclose(m[k], result['metrics'][kind + '_test'][k])
                summary, bygroup = analyze(joined, 'homology')
                remaining = [r for r in joined if r['homology'] != 'complex40_1EMV']
                without, _ = analyze(remaining, 'homology')
                detail[kind] = dict(homology=summary, groups=bygroup, without_1EMV=without)
                if kind == 'pairs':
                    rm = ranking_metrics(groups, [r['prediction'] for r in joined])
                    assert np.isclose(rm['macro_order_accuracy'], result['ranking']['test']['macro_order_accuracy'])
                    ranking_records[name, choice] = [dict(seed=42, **g) for g in rm['groups']]
                    prediction_records[name, choice] = joined
                    detail['difference_sign_accuracy_nontrivial'] = float(np.mean([
                        (r['prediction'] * r['target'] > 0) + .5 * (r['prediction'] == 0)
                        for r in joined if abs(r['target']) > .3]))
            ext = read(run / ('evaluation_' + choice) / 'external16.csv')
            external = json.loads((run / ('evaluation_' + choice) / 'external16.json').read_text())
            assert file_digest(run / 'joint' / (choice + '.pt')) == external['checkpoint_sha256']
            detail['external'] = external['metrics']
            external_rows.extend(dict(run=name, checkpoint=choice, **r) for r in ext)
            output['checkpoints'][choice] = detail
        report['runs'][name] = output
    old = json.loads((ROOT / 'outputs/esm2_pilot_v1/comparison.json').read_text())
    ranking_records['old_seq_B', 'best'] = [dict(seed=42, **g) for g in old['runs']['seq_B_seed42']['result']['ranking']['test']['groups']]
    comparisons = [('fusion_absolute_seed42', 'fusion_transfer_seed42'), ('fusion_shuffled_seed42', 'fusion_transfer_seed42'),
                   ('structure_transfer_seed42', 'fusion_transfer_seed42'), ('old_seq_B', 'fusion_transfer_seed42')]
    group_differences = []
    for before, after in comparisons:
        a, b = ranking_records[before, 'best'], ranking_records[after, 'best']
        for scope in ('all', 'at_least5', 'at_least5_without_1EMV'):
            keep = lambda r: scope == 'all' or (r['n'] >= 5 and (scope != 'at_least5_without_1EMV' or r['homology'] != 'complex40_1EMV'))
            aa, bb = list(filter(keep, a)), list(filter(keep, b))
            comparison_record = paired_cluster_bootstrap(aa, bb, repetitions=10000)
            pairs = {r['group']: r for r in aa}
            per_cluster = defaultdict(list)
            for r in bb:
                delta = r['order_accuracy'] - pairs[r['group']]['order_accuracy']
                per_cluster[r['homology']].append(delta)
            comparison_record['cluster_mean_gains'] = {k: float(np.mean(v)) for k, v in per_cluster.items()}
            comparison_record['leave_one_cluster_out_gain'] = {k: float(np.mean([v for c, vs in per_cluster.items() if c != k for v in vs])) for k in per_cluster}
            report['comparisons'][after + '-' + before + '/' + scope] = comparison_record
        aa = {r['group']: r for r in a}
        for r in b:
            group_differences.append(dict(before=before, after=after, group=r['group'], homology=r['homology'], pdb=r['pdb'],
                                          n=r['n'], baseline=aa[r['group']]['order_accuracy'], new=r['order_accuracy'],
                                          gain=r['order_accuracy'] - aa[r['group']]['order_accuracy']))
    write_csv(OUT / 'experimental_group_comparisons.csv', group_differences)
    write_csv(OUT / 'external16_all_models.csv', external_rows)
    write_csv(OUT / 'training_timeline.csv', timeline_rows)
    json_save(report, OUT / 'detailed_analysis.json')
    # Standalone research figure; no fitted/test-calibrated predictions.
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {'fusion_absolute_seed42': '#777777', 'fusion_shuffled_seed42': '#b27823',
              'fusion_transfer_seed42': '#2166ac', 'structure_transfer_seed42': '#339966'}
    labels = {'fusion_absolute_seed42': 'Fusion / absolute', 'fusion_shuffled_seed42': 'Fusion / shuffled delta',
              'fusion_transfer_seed42': 'Fusion / true delta', 'structure_transfer_seed42': 'Structure / true delta'}
    for name in colors:
        y = prediction_records[name, 'best']
        axs[0, 0].scatter([r['target'] for r in y], [r['prediction'] for r in y], s=12, alpha=.35, color=colors[name], label=labels[name])
        t = [r for r in timeline_rows if r['run'] == name and r['stage'] == 'joint']
        if name == 'fusion_transfer_seed42':
            axs[1, 0].plot([r['epoch'] for r in t], [r['weight_abs'] for r in t], label='Absolute task weight')
            axs[1, 0].plot([r['epoch'] for r in t], [r['weight_diff'] for r in t], label='Difference task weight')
    axs[0, 0].plot([-6, 4], [-6, 4], 'k--', lw=1)
    axs[0, 0].set(xlabel='Measured delta pKd', ylabel='Predicted delta pKd', title='A. Held-out mutations: primary checkpoints')
    axs[0, 0].legend(fontsize=8)
    subset = [r for r in group_differences if r['before'] == 'fusion_absolute_seed42' and r['n'] >= 5 and r['homology'] != 'complex40_1EMV']
    axs[0, 1].bar(range(len(subset)), [100*r['gain'] for r in subset], color=['#2166ac' if r['gain'] >= 0 else '#b2182b' for r in subset])
    axs[0, 1].set_xticks(range(len(subset)), [r['pdb'] for r in subset], rotation=60)
    axs[0, 1].axhline(0, color='black', lw=.7)
    axs[0, 1].set(ylabel='Order accuracy gain (percentage points)', title='B. True delta vs absolute; >=5 variants, no 1EMV')
    axs[1, 0].set(xlabel='Joint-training epoch', ylabel='GradNorm weight (sum = 2)', title='C. Adaptive weights: fusion / true delta')
    axs[1, 0].legend()
    for protein, color in [('IL17A', '#b2182b'), ('IL23R', '#2166ac')]:
        rs = [r for r in external_rows if r['run'] == 'fusion_transfer_seed42' and r['checkpoint'] == 'best' and r['target'] == protein]
        axs[1, 1].scatter([float(r['pkd']) for r in rs], [float(r['predicted_pkd']) for r in rs], color=color, label=protein)
        for r in rs:
            if r['sample_id'] in ('17-53_il17a', '17-51_il17a', '23r-91_il23r'):
                axs[1, 1].annotate(r['sample_id'].split('_')[0], (float(r['pkd']), float(r['predicted_pkd'])), xytext=(4, 4), textcoords='offset points', fontsize=8)
    axs[1, 1].plot([6, 13], [6, 13], 'k--', lw=1)
    axs[1, 1].set(xlabel='Measured pKd (23r-91 is a lower bound)', ylabel='Predicted pKd', title='D. External binders: fusion / true delta')
    axs[1, 1].legend()
    fig.savefig(OUT / 'result_diagnostics.png', dpi=180)
    fig.savefig(OUT / 'result_diagnostics.pdf')
    plt.close(fig)
    print('Verified epochs, complete row coverage, code/data hashes, selection, prediction metrics and external checkpoint hashes.')
    for key, record in report['comparisons'].items():
        if key.endswith('at_least5_without_1EMV'):
            print(key, json.dumps(record))
    for name, x in report['runs'].items():
        m = x['checkpoints']['best']
        print(name, 'absolute centered r/no1EMV', m['absolute']['homology']['within_group_centered']['pearson'], m['absolute']['without_1EMV']['within_group_centered']['pearson'],
              'delta no1EMV', m['pairs']['without_1EMV']['overall'])


if __name__ == '__main__':
    main()
