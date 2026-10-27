"""Audit completed resplit runs and describe results/timing; no model selection or training."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from appi.model import CachedAffinityRegressor

ARMS = ['absolute', 'joint_lambda0p05', 'joint_lambda0p1', 'joint_lambda0p3']
SEEDS = [42, 43, 44]


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def stats(pred, y):
    p, y = np.asarray(pred, float), np.asarray(y, float)
    return dict(n=len(y), rmse=float(np.sqrt(np.mean((p-y)**2))),
                mae=float(np.mean(abs(p-y))), bias=float((p-y).mean()),
                std_pred=float(p.std()), std_target=float(y.std()))


def main():
    runs = ROOT/'runs/affinity_resplit_v1'
    data = ROOT/'data/processed/joint_training_resplit_v1'
    out = ROOT/'outputs/resplit_training_analysis'
    out.mkdir(parents=True, exist_ok=True)
    tables = {(k, s): read(data/f'{k}_{s}.csv') for k in ['absolute', 'pairs']
              for s in ['train', 'val', 'test']}
    model = CachedAffinityRegressor(torch.zeros(768), torch.ones(768), 0., 1.)
    report = dict(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  training_counts={k: len(tables[k, 'train']) for k in ['absolute', 'pairs']},
                  baselines={}, runs={}, aggregate={},
                  timing_note='File modification times approximate the interval after config creation, '
                              'which follows loading and initialization; not a profiled training-only timer.')
    amean = np.mean([float(r['label_pkd']) for r in tables['absolute', 'train']])
    dmean = np.mean([float(r['delta_pkd']) for r in tables['pairs', 'train']])
    for split in ['val', 'test']:
        ay = np.array([float(r['label_pkd']) for r in tables['absolute', split]])
        dy = np.array([float(r['delta_pkd']) for r in tables['pairs', split]])
        report['baselines'][split] = dict(
            absolute_train_mean=stats(np.full_like(ay, amean), ay),
            delta_zero=stats(np.zeros_like(dy), dy),
            delta_train_mean=stats(np.full_like(dy, dmean), dy))
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for arm, color in zip(ARMS, ['#222222', '#1b9e77', '#377eb8', '#d95f02']):
        for seed in SEEDS:
            path = runs/f'{arm}_seed{seed}'
            config = json.loads((path/'config.json').read_text())
            metrics = json.loads((path/'metrics.json').read_text())
            assert not metrics['smoke_test'] and config['limit'] == 0
            assert config['pair_absolute_weight'] == 1 and config['device'] == 'cpu'
            assert (ROOT/config['data']).resolve() == data.resolve()
            assert config['trainer_hash'] == hashlib.sha256((ROOT/'scripts/train_affinity.py').read_bytes()).hexdigest()
            assert config['model_hash'] == hashlib.sha256((ROOT/'appi/model.py').read_bytes()).hexdigest()
            for name, digest in config['input_hashes'].items():
                assert hashlib.sha256((data/name).read_bytes()).hexdigest() == digest
            files = sorted(path.glob('epoch_*.json'))
            epochs = [json.loads(p.read_text()) for p in files]
            assert [e['epoch'] for e in epochs] == list(range(1, len(epochs)+1))
            last = torch.load(path/'last.pt', map_location='cpu', weights_only=True)
            best = torch.load(path/'best.pt', map_location='cpu', weights_only=True)
            steps = {int(s['step'].item()) for s in last['optimizer']['state'].values()}
            assert steps == {len(epochs)*math.ceil(len(tables['absolute', 'train'])/config['batch_size'])}
            assert last['stale'] == config['patience'] == 25
            assert last['epoch'] == len(epochs)
            assert best['epoch'] == metrics['best_epoch']
            record = dict(best_epoch=metrics['best_epoch'], epochs=len(epochs), optimizer_steps=sorted(steps),
                          post_config_seconds=(path/'metrics.json').stat().st_mtime-(path/'config.json').stat().st_mtime,
                          first_to_last_epoch_seconds=files[-1].stat().st_mtime-files[0].stat().st_mtime,
                          first_losses={k:v for k,v in epochs[0].items() if k.startswith('train_')},
                          last_losses={k:v for k,v in epochs[-1].items() if k.startswith('train_')},
                          metrics=metrics['metrics'], diagnostics={}, by_source={}, by_group={})
            for kind in ['absolute', 'pairs']:
                for split in ['val', 'test']:
                    rows = read(path/f'predictions_{kind}_{split}.csv')
                    key = 'csv_line' if kind == 'absolute' else 'skempi_csv_line'
                    label = 'label_pkd' if kind == 'absolute' else 'delta_pkd'
                    metadata = {r[key]:r for r in tables[kind, split]}
                    assert len(rows) == len(metadata) and {r['source_line'] for r in rows} == set(metadata)
                    assert all(np.isclose(float(r['target']), float(metadata[r['source_line']][label]),
                                          rtol=0, atol=2e-6) for r in rows)
                    p = np.array([float(r['prediction']) for r in rows])
                    y = np.array([float(r['target']) for r in rows])
                    record['diagnostics'][kind+'_'+split] = stats(p, y)
                    for field, name in [('group_id', 'by_group'), ('Source Data Set', 'by_source')]:
                        if kind == 'pairs' and name == 'by_source':
                            continue
                        groups = np.array([metadata[r['source_line']][field] for r in rows])
                        record[name][kind+'_'+split] = {g:stats(p[groups==g], y[groups==g])
                                                        for g in sorted(set(groups))}
            report['runs'][path.name] = record
            x = [e['epoch'] for e in epochs]
            for ax, values in zip(axes.flat, [
                    [e['train_abs_mse_scaled'] for e in epochs],
                    [e['val_absolute']['rmse'] for e in epochs],
                    [e['train_pair_abs_mse_scaled'] for e in epochs],
                    [e['val_delta_pkd']['rmse'] for e in epochs]]):
                ax.plot(x, values, color=color, alpha=.7, label=arm if seed==42 else None)
        report['aggregate'][arm] = {}
        for block in ['absolute_val', 'absolute_test', 'pairs_val', 'pairs_test']:
            rs = [report['runs'][f'{arm}_seed{s}'] for s in SEEDS]
            report['aggregate'][arm][block] = {
                name:dict(mean=statistics.mean(r['diagnostics'][block][name] for r in rs),
                          seed_std=statistics.stdev(r['diagnostics'][block][name] for r in rs))
                for name in ['rmse', 'mae', 'bias', 'std_pred', 'std_target']}
            report['aggregate'][arm][block]['macro_group_mae'] = statistics.mean(
                statistics.mean(g['mae'] for g in r['by_group'][block].values()) for r in rs)
    for ax, title in zip(axes.flat, ['Absolute training MSE (scaled)', 'Absolute validation RMSE (pKd)',
                                     'Pair endpoint training MSE (scaled)', 'Difference validation RMSE (delta pKd)']):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out/'learning_curves.png', dpi=160)
    plt.close(figure)
    accounting = subprocess.run(['sacct', '-j', '15763', '--format=JobID,State,ExitCode,Elapsed,TotalCPU,AllocCPUS,MaxRSS', '-P'],
                                capture_output=True, text=True, check=True)
    (out/'slurm_accounting.txt').write_text(accounting.stdout)
    (out/'diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Trainable parameters:', report['trainable_parameters'])
    print('Baselines:', json.dumps(report['baselines']))
    for arm in ARMS:
        print(arm, json.dumps(report['aggregate'][arm]))
    print('All 12 configurations, input hashes, prediction labels and optimizer step counts verified.')


if __name__ == '__main__':
    main()
