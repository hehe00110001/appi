import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from appi.esm_residue_model import ESMResidueAffinityRegressor
from appi.esm_residue_data import join_graphs, collate_graphs, backward_ranking
from prepare_esm_manifest import align_indices
from extract_esm2_features import window_starts


def fixture(mode):
    torch.manual_seed(42)
    model = ESMResidueAffinityRegressor(torch.zeros(1408), torch.ones(1408), 7., 2.,
                                       width=16, layers=1, heads=2, mode=mode)
    base = {'a':dict(h=torch.randn(4,128), coordinates=torch.randn(4,3), partner=torch.tensor([1,1,0,0],dtype=torch.bool))}
    esm = {'a':dict(h=torch.randn(4,1280), sequence_pooled=torch.randn(2,1280), sequence_lengths=torch.tensor([55,90]))}
    graphs = join_graphs(base, esm, mode)
    return model, graphs, collate_graphs(graphs,['a'],'cpu')


@pytest.mark.parametrize('mode',['seq','esm_graph','esm_nogeom','fusion'])
def test_invariance_and_shared_pair_score(mode):
    model, _, x = fixture(mode)
    model.eval()
    pred = model(**x)
    perm = torch.arange(x['h'].shape[1]-1,-1,-1)
    permuted = {k:v[:,perm] for k,v in x.items()}
    assert torch.allclose(pred,model(**permuted),atol=2e-6)
    assert torch.allclose(pred, model(**dict(x, partner=~x['partner'])),atol=2e-6)
    assert torch.equal(model.delta(x,x),torch.zeros(1))
    if mode != 'seq':
        padded={k:torch.cat([v,torch.zeros((1,2,*v.shape[2:]),dtype=v.dtype)],1) for k,v in x.items()}
        padded['h'][:, -2:] = float('nan'); padded['coordinates'][:, -2:] = float('nan')
        assert torch.allclose(pred,model(**padded),atol=2e-6)
    moved=dict(x,coordinates=x['coordinates']+500)
    assert torch.allclose(pred,model(**moved),atol=1e-5)


@pytest.mark.parametrize('mode',['seq','esm_nogeom'])
def test_controls_ignore_geometry(mode):
    model, _, x=fixture(mode)
    assert torch.equal(model(**x),model(**dict(x,coordinates=torch.randn_like(x['coordinates'])*100)))


def test_fusion_dropout_shared_across_comparisons_and_gradients():
    model, graphs, x=fixture('fusion')
    model.train()
    for mask in [(1.,1.),(0.,1.),(1.,0.)]:
        model.modality_mask=mask
        assert torch.equal(model.delta(x,x),torch.zeros(1))
        model.zero_grad()
        model(**x).sum().backward()
        for active,branch in zip(mask,[model.ppi_projection,model.esm_projection]):
            norm=sum(p.grad.abs().sum() for p in branch.parameters())
            assert bool(norm>0)==bool(active)
    graphs['b']={k:v.clone() for k,v in graphs['a'].items()}
    graphs['b']['h']+=.3
    model.modality_mask=(1.,1.)
    model.zero_grad()
    backward_ranking(model,graphs,[('a','b',1,'group')],'cpu',2,1000,100.,.1)
    assert model.esm_projection[0].weight.grad.norm()>0


def test_alignment_and_long_window_coverage():
    indices,tie=align_indices('ACDFGH','MACDEFGHQQ')
    assert ''.join('MACDEFGHQQ'[i] for i in indices)=='ACDFGH'
    with pytest.raises(ValueError): align_indices('AZ','AAAA')
    for length in [1,1022,1023,1800,5000]:
        count=torch.zeros(length)
        for start in window_starts(length):count[start:start+1022]+=1
        assert (count>0).all()


def test_sequence_control_independent_of_interface_crop():
    _,graphs,x=fixture('seq')
    assert x['h'].shape==(1,2,1408)
    assert x['h'][0,:,0].tolist()==[55,90]
