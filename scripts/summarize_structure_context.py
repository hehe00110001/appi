"""Report preselected checkpoints, difference baselines and group sensitivity."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from appi.structure_context_data import file_digest
from train_affinity import json_save


def group_summary(groups):
    def average(rows):
        return float(np.mean([r['order_accuracy'] for r in rows])) if rows else None
    eligible = [g for g in groups if g['n'] >= 5]
    homologies = sorted({g['homology'] for g in eligible})
    leave_one_out = {h: average([g for g in eligible if g['homology'] != h]) for h in homologies}
    return dict(all_groups=average(groups), at_least5=average(eligible),
                at_least5_without_1EMV=average([g for g in eligible if g['homology'] != 'complex40_1EMV']),
                eligible_groups=len(eligible), homology_groups=len(homologies), leave_one_homology_out=leave_one_out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'runs/structure_context_pilot_v1')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/structure_context_pilot_v1')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(policy='Primary: validation macro order, tie difference MAE; secondary: validation absolute MAE. Test and external16 never select.',
                  limitation='One-seed results are exploratory. Same optimizer-update budget does not imply same task exposures or compute.', runs={})
    for run in sorted(args.root.iterdir()):
        if not (run / 'complete.json').exists():
            continue
        config = json.loads((run / 'config.json').read_text())
        for path, expected in config['code_hashes'].items():
            if file_digest(ROOT / path) != expected:
                raise ValueError('Implementation changed: ' + path)
        result = json.loads((run / 'metrics.json').read_text())
        if result['smoke_test']:
            raise ValueError('Do not mix smoke outputs into a scientific report')
        stages = {stage: [json.loads(p.read_text()) for p in sorted((run / stage).glob('epoch_*.json'))]
                  for stage in ('warmup', 'joint')}
        joint = stages['joint']
        selected = min(joint, key=lambda e: (-e['val_ranking']['macro_order_accuracy'], e['val_difference']['mae']))
        assert selected['epoch'] == result['checkpoints']['best']['epoch']
        assert min(joint, key=lambda e: e['val_absolute']['mae'])['epoch'] == result['checkpoints']['best_absolute']['epoch']
        records = [step for e in joint for step in e['steps'] if len(step['weights']) == 2]
        diagnostics = {}
        if records:
            w = np.array([r['weights'] for r in records])
            norms = np.array([r['effective_shared_gradient_norms'] for r in records])
            cosine = np.array([r['shared_gradient_cosine'] for r in records])
            diagnostics = dict(task_order=['absolute', 'difference'], weight_min=w.min(0).tolist(), weight_max=w.max(0).tolist(),
                               weight_mean=w.mean(0).tolist(), effective_gradient_norm_mean=norms.mean(0).tolist(),
                               zero_effective_gradient_steps=(norms <= 1e-12).sum(0).tolist(),
                               conflicting_step_fraction=float((cosine < 0).mean()), mean_gradient_cosine=float(cosine.mean()))
        entry = dict(result=result, gradient_diagnostics=diagnostics, stages={stage: dict(epochs=len(es),
                     optimizer_steps=sum(e['optimizer_steps'] for e in es),
                     task_draws={kind: sum(e['draws'].get(kind, 0) for e in es) for kind in ('absolute', 'pairs')}) for stage, es in stages.items()},
                     group_sensitivity={}, external={})
        for choice, selected_result in result['checkpoints'].items():
            entry['group_sensitivity'][choice] = group_summary(selected_result['ranking']['test']['groups'])
            external = run / ('evaluation_' + choice) / 'external16.json'
            if external.exists():
                entry['external'][choice] = json.loads(external.read_text())['metrics']
        if stages['warmup']:
            best = min(stages['warmup'], key=lambda e: (-e['val_ranking']['macro_order_accuracy'], e['val_difference']['mae']))
            entry['warmup_validation'] = dict(epoch=best['epoch'], difference=best['val_difference'], ranking=best['val_ranking'])
        report['runs'][run.name] = entry
    if not report['runs']:
        raise ValueError('No completed full-data runs')
    json_save(report, args.out / 'comparison.json')
    lines = ['# Structure-context pilot', '', report['policy'], '', report['limitation'], '',
             '|Run|Absolute test MAE|Difference MAE|Zero-change MAE|Macro order|≥5, excluding 1EMV|',
             '|---|---:|---:|---:|---:|---:|']
    for name, record in report['runs'].items():
        m = record['result']['checkpoints']['best']['metrics']
        g = record['group_sensitivity']['best']
        reduced = 'NA' if g['at_least5_without_1EMV'] is None else format(g['at_least5_without_1EMV'], '.1%')
        lines.append(f"|{name}|{m['absolute_test']['mae']:.3f}|{m['pairs_test']['mae']:.3f}|{m['pairs_test']['zero_change_baseline']['mae']:.3f}|{g['all_groups']:.1%}|{reduced}|")
    (args.out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('Written: ' + str(args.out / 'comparison.json'))


if __name__ == '__main__':
    main()
