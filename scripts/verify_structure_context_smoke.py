"""Audit four GPU controls, task routing, model selection, and exact staged resume."""
import argparse
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal_tree(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT / 'runs/structure_context_smoke_v1')
    args = p.parse_args()
    for name in ('structure_transfer', 'fusion_absolute', 'fusion_transfer', 'fusion_shuffled'):
        directory = args.root / name
        config = json.loads((directory / 'config.json').read_text())
        result = json.loads((directory / 'metrics.json').read_text())
        assert result['smoke_test'] and config['arguments']['device'] == 'cuda'
        joint = [json.loads(p.read_text()) for p in sorted((directory / 'joint').glob('epoch_*.json'))]
        assert len(joint) == 2
        winner = min(joint, key=lambda x: (-x['val_ranking']['macro_order_accuracy'], x['val_difference']['mae']))
        assert result['checkpoints']['best']['epoch'] == winner['epoch']
        winner_abs = min(joint, key=lambda x: x['val_absolute']['mae'])
        assert result['checkpoints']['best_absolute']['epoch'] == winner_abs['epoch']
        steps = [s for epoch in joint for s in epoch['steps']]
        assert all(s['branch_gradient_norms']['structure'] > 0 for s in steps)
        if name.startswith('fusion'):
            assert any(s['context_active'] and s['branch_gradient_norms']['esm'] > 0 for s in steps)
        else:
            assert all(not s['context_active'] and s['branch_gradient_norms']['esm'] == 0 for s in steps)
        if name != 'fusion_absolute':
            warm = [json.loads(p.read_text()) for p in sorted((directory / 'warmup').glob('epoch_*.json'))]
            assert len(warm) == 2
            assert all(not s['context_active'] and s['tasks'] == ['pairs'] and s['branch_gradient_norms']['esm'] == 0 for e in warm for s in e['steps'])
            assert all(min(s['raw_shared_gradient_norms']) > 0 and abs(sum(s['weights']) - 2) < 1e-5 for s in steps)
            assert any(abs(s['weights'][0] - 1) > 1e-5 for s in steps)
        else:
            assert all(s['tasks'] == ['absolute'] for s in steps)
        if name == 'fusion_shuffled':
            assert config['shuffle']['changed_rows'] > 0
    for stage in ('warmup', 'joint'):
        a = torch.load(args.root / 'fusion_transfer' / stage / 'last.pt', map_location='cpu', weights_only=True)
        b = torch.load(args.root / 'resumed' / stage / 'last.pt', map_location='cpu', weights_only=True)
        for key in ('model', 'optimizer', 'controller', 'controller_optimizer', 'rng', 'cuda_rng', 'epoch', 'best', 'stale'):
            equal_tree(a[key], b[key])
    external = json.loads((args.root / 'fusion_transfer/evaluation_best/external16.json').read_text())
    assert external['metrics']['all16/all']['n'] == 16 and external['metrics']['without_boundary15/all']['n'] == 15
    (args.root / 'verified.json').write_text(json.dumps(dict(passed=True, controls=4, exact_resume=True, external16=True), indent=2) + '\n')
    print('PASS: four GPU controls, both tasks active, selection correct, staged resume bitwise identical, external16 complete')


if __name__ == '__main__':
    main()
