import copy
import torch
import pytest
from appi.structure_context_model import StructureContextRegressor
from appi.structure_context_data import StructureContextData
from appi.gradnorm import GradNorm, combine_gradients, gradient_diagnostics


def fixture():
    torch.manual_seed(19)
    model = StructureContextRegressor(width=32, context_width=16, heads=4, layers=2, modality_dropout=0).eval()
    batch = dict(h=torch.randn(2, 6, 128), coordinates=torch.randn(2, 6, 3) * 3,
                 partner=torch.tensor([[0, 0, 0, 1, 1, 1]] * 2).bool(), mask=torch.ones(2, 6).bool(),
                 memory=torch.randn(2, 11, 1280), memory_mask=torch.ones(2, 11).bool(),
                 memory_relation=torch.randint(0, 4, (2, 6, 11)))
    return model, batch


def test_permutation_rigid_motion_and_partner_exchange():
    model, x = fixture()
    expected = model(**x)
    p, q = torch.randperm(6), torch.randperm(11)
    z = {k: v.clone() for k, v in x.items()}
    for k in ('h', 'coordinates', 'partner', 'mask'):
        z[k] = z[k][:, p]
    for k in ('memory', 'memory_mask'):
        z[k] = z[k][:, q]
    z['memory_relation'] = z['memory_relation'][:, p][:, :, q]
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    z['coordinates'] = z['coordinates'] @ rotation + 12
    z['partner'] = ~z['partner']
    torch.testing.assert_close(model(**z), expected, atol=1e-6, rtol=1e-5)


def test_padding_nonfinite_and_batch_independence():
    model, x = fixture()
    expected = model(**x)
    z = {}
    for k, v in x.items():
        if k == 'memory_relation':
            z[k] = torch.nn.functional.pad(v, (0, 3, 0, 2))
        elif v.ndim == 3:
            z[k] = torch.nn.functional.pad(v, (0, 0, 0, 3 if k == 'memory' else 2), value=float('nan'))
        else:
            z[k] = torch.nn.functional.pad(v, (0, 3 if k == 'memory_mask' else 2), value=False)
    torch.testing.assert_close(model(**z), expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(model(**{k: v[:1] for k, v in x.items()}), expected[:1], atol=1e-6, rtol=1e-5)


def test_shared_score_difference_algebra():
    model, x = fixture()
    a, b = model(**x)
    c = model(**{**x, 'h': x['h'] + .2})[0]
    assert float((a - a).detach()) == 0
    torch.testing.assert_close(a - b, -(b - a))
    torch.testing.assert_close((a - b) + (b - c), a - c)


def test_sequence_routes_structure_and_gradients_reach_both_branches():
    model, x = fixture()
    x['h'].requires_grad_()
    x['memory'].requires_grad_()
    pred = model(**x)
    pred.sum().backward()
    assert x['h'].grad.norm() > 0 and x['memory'].grad.norm() > 0
    assert model.structure_adapter[0].weight.grad.norm() > 0
    assert model.esm_projection[0].weight.grad.norm() > 0
    changed = model(**{**x, 'memory': torch.randn_like(x['memory'])})
    assert not torch.allclose(pred, changed, atol=1e-7, rtol=1e-7)
    # Equal structural values cannot acquire new sequence content through routing.
    z = {**x, 'h': x['h'][:, :1].expand(-1, 6, -1)}
    torch.testing.assert_close(model(**z), model(**{**z, 'memory': torch.randn_like(x['memory'])}), atol=1e-6, rtol=1e-5)


def test_warmup_and_shared_step_dropout_remove_context_only():
    model, x = fixture()
    model.set_stage('warmup')
    pred = model(**x)
    pred.sum().backward()
    assert model.esm_projection[0].weight.grad is None
    assert model.structure_adapter[0].weight.grad.norm() > 0
    torch.testing.assert_close(pred, model(**{**x, 'memory': None}))
    model.set_stage('joint')
    model.train()
    model.modality_dropout = 1
    model.start_step()
    assert not model.context_active
    torch.testing.assert_close(model(**x), pred)
    torch.testing.assert_close(model(**{**x, 'memory': None}), pred)
    model.eval()
    with pytest.raises(ValueError, match='memory'):
        model(**{**x, 'memory': None})


def test_mapping_ambiguity_and_full_chain_retention():
    g = dict(h=torch.randn(3, 128), coordinates=torch.randn(3, 3), partner=torch.tensor([0, 0, 1]).bool())
    record = dict(chains={'A': dict(sequence_id='a', partner=False), 'B': dict(sequence_id='b', partner=True)},
                  nodes=[['A', 1], ['A', 3], ['B', 2]], ambiguous_nodes=[dict(node=0, chain='A', indices=[1, 2])])
    data = StructureContextData({'x': g}, dict(graphs={'x': record}), {'a': torch.randn(8, 1280), 'b': torch.randn(5, 1280)})
    b = data.collate(['x'], 'cpu')
    assert b['memory'].shape == (1, 13, 1280)
    assert b['memory_relation'][0, 0, 1:3].tolist() == [3, 3]
    assert b['memory_relation'][0, 2, 10] == 3
    assert b['memory_relation'][0, 0, 8:].sum() == 0
    assert list(data.microbatches([['x', 'x']] * 3, 8, 1)) == [[0], [1], [2]]
    record['nodes'][0][1] = 20
    with pytest.raises(ValueError, match='mapping'):
        StructureContextData({'x': g}, dict(graphs={'x': record}), data.bank)


def test_gradnorm_and_exact_gradient_combination():
    controller = GradNorm()
    opt = torch.optim.SGD(controller.parameters(), lr=.01)
    loss = controller.objective(torch.tensor([1., 1.], requires_grad=True), torch.tensor([1., 10.], requires_grad=True))
    loss.backward()
    opt.step()
    w = controller.weights().detach()
    assert w[0] > 1 and w[1] < 1 and (w > 0).all()
    torch.testing.assert_close(w.sum(), torch.tensor(2.))
    p, q = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(1))
    gradients = [[torch.tensor([1., 2.]), None], [torch.tensor([-2., 1.]), torch.tensor([3.])]]
    combine_gradients([p, q], gradients, w)
    torch.testing.assert_close(p.grad, w[0] * gradients[0][0] + w[1] * gradients[1][0])
    torch.testing.assert_close(q.grad, w[1] * gradients[1][1])
    norms, cosine = gradient_diagnostics([p, q], [p], gradients)
    assert cosine == 0
    torch.testing.assert_close(norms, torch.tensor([5., 5.]).sqrt())
    restored = GradNorm()
    restored.load_state_dict(copy.deepcopy(controller.state_dict()))
    torch.testing.assert_close(restored.weights(), controller.weights())
