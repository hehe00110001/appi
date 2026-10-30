"""Fail the GPU smoke job unless all arms ran and ranking contributed gradients."""
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    runs = ROOT/'runs/group_ranking_smoke_v1'
    configs, manifests, states = {}, {}, {}
    for arm in ['B','C','R']:
        path = runs/f'{arm}_seed42'
        config = json.loads((path/'config.json').read_text())
        result = json.loads((path/'metrics.json').read_text())
        epochs = [json.loads(p.read_text()) for p in sorted(path.glob('epoch_*.json'))]
        assert result['smoke_test'] and len(epochs) == 2
        assert all(e['unique_absolute_records'] == 8 for e in epochs)
        if arm == 'R':
            assert all(e['unique_rank_groups'] == config['ranking_audit']['train']['eligible_groups'] for e in epochs)
            assert all(e['rank_comparison_draws'] == 16 and e['train_rank_hinge_scaled'] > 0 for e in epochs)
        else:
            assert all(e['rank_comparison_draws'] == 0 for e in epochs)
        configs[arm] = config
        manifests[arm] = json.loads((path/'training_manifest.json').read_text())
        states[arm] = torch.load(path/'last.pt', map_location='cpu', weights_only=True)['model']
    assert len({c['initial_state_sha256'] for c in configs.values()}) == 1
    assert manifests['C'] == manifests['R']
    assert manifests['B']['absolute'] == manifests['C']['absolute']
    assert any(not torch.equal(states['C'][k], states['R'][k]) for k in states['C'])
    print('PASS: GPU training, common initialization, matched manifests, ranking coverage and parameter updates')


if __name__ == '__main__':
    main()
