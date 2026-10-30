"""Training-contract tests: microbatch gradients and label-shuffle boundaries."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_structure_context import backward_task, shuffled_targets, selection_key
from appi.structure_context_model import StructureContextRegressor
from appi.structure_context_data import StructureContextData


def test_microbatch_gradients_match_full_batch_for_shared_difference():
    torch.manual_seed(7)
    graphs = {str(i): dict(h=torch.randn(4 + i, 128), coordinates=torch.randn(4 + i, 3),
                          partner=torch.arange(4 + i) % 2 == 0) for i in range(4)}
    manifest = dict(graphs={p: dict(chains={'A': dict(sequence_id='a', partner=True), 'B': dict(sequence_id='b', partner=False)},
                                  nodes=[['A' if bool(g['partner'][j]) else 'B', j] for j in range(len(g['h']))]) for p, g in graphs.items()})
    data = StructureContextData(graphs, manifest, {'a': torch.randn(10, 1280), 'b': torch.randn(10, 1280)})
    model = StructureContextRegressor(width=16, context_width=16, heads=4, layers=1, modality_dropout=0).train()
    other = copy.deepcopy(model)
    rows = [dict(mutant_feature_path='0', wt_feature_path='1'), dict(mutant_feature_path='2', wt_feature_path='3')]
    targets, weights = torch.tensor([1., -.4]), torch.tensor([.5, 1.5])
    losses = []
    for m, size in [(model, 1), (other, 2)]:
        args = SimpleNamespace(microbatch_size=size, attention_limit=1000000, device='cpu')
        m.start_step()
        losses.append(backward_task(m, data, rows, 'pairs', [0, 1], weights, targets, args))
    assert abs(losses[0] - losses[1]) < 1e-6
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4)


def test_shuffling_keeps_groups_and_absolute_labels_intact():
    rows = []
    for g in range(2):
        for i in range(8):
            rows.append(dict(wt_feature_path=f'w{g}', mutant_feature_path=f'm{g}_{i}',
                             Reference=f'study{g}', Method='SPR', Temperature='298', pkd_wt='7',
                             pkd_mut=str(7 + g * 10 + i), delta_pkd=str(g * 10 + i),
                             **{'#Pdb': f'pdb{g}', 'group_id': f'homology{g}'}))
    before = copy.deepcopy(rows)
    shuffled, audit = shuffled_targets(rows, 42)
    assert rows == before and audit['changed_rows'] > 0
    for start in (0, 8):
        assert sorted(shuffled[start:start + 8].tolist()) == sorted(float(r['delta_pkd']) for r in rows[start:start + 8])
    assert selection_key(dict(macro_order_accuracy=.7), dict(mae=2.)) < selection_key(dict(macro_order_accuracy=.6), dict(mae=.1))
    assert selection_key(dict(macro_order_accuracy=.7), dict(mae=.1)) < selection_key(dict(macro_order_accuracy=.7), dict(mae=.2))
