"""Require complete GPU training for every mode and supervision before the pilot."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for mode in ('seq', 'esm_graph', 'esm_nogeom', 'fusion'):
        configs = []
        for arm in ('B', 'R'):
            path = ROOT/f'runs/esm2_smoke_v1/{mode}_{arm}_seed42'
            config = json.loads((path/'config.json').read_text())
            result = json.loads((path/'metrics.json').read_text())
            epochs = [json.loads(p.read_text()) for p in sorted(path.glob('epoch_*.json'))]
            assert result['smoke_test'] and len(epochs) == 2
            assert all(e['unique_absolute_records'] == 8 for e in epochs)
            if arm == 'R':
                assert all(e['train_rank_hinge_scaled'] > 0 and e['rank_comparison_draws'] == 16 for e in epochs)
            assert all(e['branch_gradient_norm_mean']['esm'] > 0 or mode == 'fusion' for e in epochs)
            configs.append(config)
        assert configs[0]['initial_state_sha256'] == configs[1]['initial_state_sha256']
        assert configs[0]['normalization_record_ids'] == configs[1]['normalization_record_ids']
    print('PASS: all eight GPU smoke runs, matched initialization and active ranking loss')


if __name__ == '__main__':
    main()
