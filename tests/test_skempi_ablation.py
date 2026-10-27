import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from appi.skempi_ablation import balanced_weights, epoch_indices, weighted_mse_sum


def test_hierarchical_weights_remove_source_and_complex_count_dominance():
    rows = [dict(**{'Source Data Set':s,'PDB':p}) for s,p in [('a','1'),('a','1'),('a','2'),('b','3')]]
    w = balanced_weights(rows,'absolute')
    assert torch.all(w>0) and torch.allclose(w.mean(),torch.tensor(1.,dtype=torch.float64))
    assert w[:3].sum() == w[3]
    assert w[:2].sum() == w[2]
    pairs=[{'#Pdb':x} for x in ['1_A_B','1_A_B','2_A_B']]
    p=balanced_weights(pairs,'pairs')
    assert p[:2].sum()==p[2]


def test_epoch_coverage_repetition_and_independent_streams():
    ids=epoch_indices(7,19,42)
    assert len(ids)==19 and set(ids)==set(range(7))
    assert sorted(ids[:7])==list(range(7)) and sorted(ids[7:14])==list(range(7))
    assert ids==epoch_indices(7,19,42)
    epoch_indices(9,19,99)  # Pair sampling must not change absolute order.
    assert ids==epoch_indices(7,19,42)


def test_weighted_microbatches_preserve_loss_and_gradient_for_both_scores():
    truth=torch.tensor([1.,-2.,.5,4.,-1.],dtype=torch.float64)
    weights=torch.tensor([.1,.5,2.,1.4,1.],dtype=torch.float64)
    full=torch.tensor([[.2,.7],[.3,-.2],[.6,.1],[1.,-1.],[.1,.4]],dtype=torch.float64,requires_grad=True)
    partial=full.detach().clone().requires_grad_()
    loss=weighted_mse_sum(full[:,0]-full[:,1],truth,weights,2.,5)
    loss.backward()
    total=0.
    for sl in [slice(0,2),slice(2,3),slice(3,5)]:
        part=weighted_mse_sum(partial[sl,0]-partial[sl,1],truth[sl],weights[sl],2.,5)
        total+=float(part.detach());part.backward()
    assert abs(total-float(loss.detach()))<1e-12
    assert torch.allclose(full.grad,partial.grad,atol=1e-12)


def test_training_step_matches_exact_objective_without_endpoint_loss():
    # Exercise the actual training loop, including pair subtraction and microbatches.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from train_skempi_ablation import train_epoch
    class TinyScore(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.w=torch.nn.Parameter(torch.tensor(.2))
            self.register_buffer('target_scale',torch.tensor(2.))
        def forward(self,h,coordinates,partner,mask):
            return self.w*(h[:,:,0]*mask).sum(1)/mask.sum(1)
    graphs={str(i):dict(h=torch.full((2,128),float(i)),coordinates=torch.zeros(2,3),partner=torch.tensor([False,True])) for i in range(1,5)}
    absolute=[{'feature_path':str(i),'label_pkd':str(.1*i)} for i in range(1,5)]
    pairs=[{'mutant_feature_path':str(i),'wt_feature_path':'1','delta_pkd':str(-.1*i)} for i in range(2,5)]
    aw=torch.tensor([.5,.5,1.,2.]);pw=torch.tensor([.5,1.,1.5])
    for lam in [0.,.3]:
        actual=TinyScore();expected=TinyScore()
        opt=torch.optim.SGD(actual.parameters(),lr=.1)
        refopt=torch.optim.SGD(expected.parameters(),lr=.1)
        loss=((expected.w*torch.arange(1.,5.)-torch.arange(1.,5.)*.1)/2).square().mul(aw).mean()
        if lam:
            ix=epoch_indices(3,4,42+20000)
            values=torch.tensor([2.,3.,4.])[ix]
            loss=loss+lam*((expected.w*(values-1)+.1*values)/2).square().mul(pw[ix]).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(expected.parameters(),5.);refopt.step()
        result,steps=train_epoch(actual,opt,graphs,absolute,pairs,aw,pw,0,42,4,'cpu',4,1,100,lam)
        assert torch.allclose(actual.w,expected.w,atol=1e-7)
        assert steps==1 and result['unique_absolute_records']==4
        assert result['pair_draws']==(4 if lam else 0)
