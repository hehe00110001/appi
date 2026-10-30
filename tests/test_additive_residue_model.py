import torch
from appi.residue_model import ResidueAffinityRegressor
from appi.additive_residue_model import AdditiveResidueAffinityRegressor

def setup():
 stats=(torch.zeros(128),torch.ones(128),torch.tensor(7.),torch.tensor(2.))
 torch.manual_seed(42); base=ResidueAffinityRegressor(*stats,width=16,layers=1,heads=2)
 torch.manual_seed(42); add=AdditiveResidueAffinityRegressor(*stats,width=16,layers=1,heads=2)
 x=dict(h=torch.randn(2,5,128),coordinates=torch.randn(2,5,3),partner=torch.tensor([[0,0,1,1,0],[1,1,0,0,0]],dtype=torch.bool),mask=torch.tensor([[1,1,1,1,0],[1,1,1,1,1]],dtype=torch.bool))
 return base,add,x

def test_initial_predictions_and_shared_parameters_equal_baseline():
 base,add,x=setup()
 assert all(torch.equal(v,add.state_dict()[k]) for k,v in base.state_dict().items())
 assert torch.equal(base(**x),add(**x))
 assert torch.equal(add.delta(x,x),torch.zeros(2))

def test_nonzero_local_branch_preserves_padding_symmetry_and_antisymmetry():
 _,add,x=setup();torch.nn.init.normal_(add.local_score[-1].weight,std=.01)
 pred=add(**x)
 swapped=dict(x,partner=~x['partner']);assert torch.allclose(pred,add(**swapped),atol=1e-6)
 padded={k:torch.cat([v,torch.zeros((2,2,*v.shape[2:]),dtype=v.dtype)],dim=1) for k,v in x.items()}
 padded['h'][:,5:]=float('nan');padded['coordinates'][:,5:]=float('nan')
 assert torch.allclose(pred,add(**padded),atol=1e-6)
 mutant=dict(x,h=x['h']+.1)
 assert torch.allclose(add.delta(mutant,x),-add.delta(x,mutant))

def test_local_branch_receives_gradients_and_can_learn():
 _,add,x=setup();p0=add(**x).detach();loss=(add(**x)-torch.tensor([4.,9.])).square().mean();loss.backward()
 assert add.local_score[-1].weight.grad.norm()>0
 with torch.no_grad():add.local_score[-1].weight.add_(add.local_score[-1].weight.grad,alpha=-1e-5)
 assert not torch.equal(add(**x),p0)
 assert add.last_components[1].abs().sum()>0
