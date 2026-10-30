import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from appi.group_ranking import build_ranking_groups, ranking_epoch_plan, backward_ranking, ranking_metrics


def row(i, value, **kw):
    return dict(wt_feature_path='wt', mutant_feature_path=str(i), Reference='paper',
                Method='SPR', Temperature='298', pkd_wt='8', pkd_mut=str(value),
                **{'#Pdb':'1ABC_A_B', 'group_id':'homology'}, **kw)


def test_groups_separate_experiments_and_average_replicates():
    rows = [row(1, 8), row(1, 8.2), row(2, 9)]
    for field in ['Reference', 'Method', 'Temperature', 'wt_feature_path', 'pkd_wt']:
        other = row(3, 10); other[field] = '9' if field == 'pkd_wt' else 'different'
        rows.append(other)
    for i in range(2):
        other = row(4+i, 11+i); other['Method']=''; rows.append(other)
    groups = build_ranking_groups(rows)
    active = [g for g in groups if g['comparisons']]
    assert len(active) == 1 and active[0]['targets'] == [8.1, 9.]
    assert len(groups) == 8 and active[0]['comparisons'] == [(0, 1, -1.)]


def test_sampling_equal_groups_coverage_and_reproducibility():
    rows = [row(i, i) for i in range(8)]
    other = [row(20+i, i) for i in range(2)]
    for r in other: r['Reference']='another'
    groups = build_ranking_groups(rows+other)
    plan = ranking_epoch_plan(groups, 2, 2, 3, 42)
    assert plan == ranking_epoch_plan(groups, 2, 2, 3, 42)
    assert all(len(b) == 6 for b in plan)
    for batch in plan:
        assert sorted([sum(x[3] == gid for x in batch) for gid in {x[3] for x in batch}]) == [3,3]


def test_microbatch_ranking_gradient_equals_full_objective():
    class Score(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.w=torch.nn.Parameter(torch.tensor(.01))
            self.register_buffer('target_scale', torch.tensor(2.))
        def forward(self, h, coordinates, partner, mask):
            return self.w*(h[:,:,0]*mask).sum(1)/mask.sum(1)
    graphs={str(i):dict(h=torch.full((2,128),float(i)), coordinates=torch.zeros(2,3),
                       partner=torch.tensor([False,True])) for i in range(4)}
    comparisons=[('0','1',-1.,'a'),('2','3',1.,'b'),('0','3',-1.,'a'),('1','2',1.,'b')]
    actual=Score(); expected=Score()
    gap=expected.w*torch.tensor([-1.,-1.,-3.,-1.])
    loss=torch.relu(.25-torch.tensor([-1.,1.,-1.,1.])*gap).mean()/2
    (.1*loss).backward()
    got=backward_ranking(actual,graphs,comparisons,'cpu',1,8,.25,.1)
    assert np.isclose(got, float(loss.detach()))
    assert torch.allclose(actual.w.grad,expected.w.grad,atol=1e-8)


def test_constant_predictions_stay_in_ranking_denominator():
    groups=build_ranking_groups([row(0,8),row(1,9),row(2,10)])
    result=ranking_metrics(groups,[0.,0.,0.])
    assert result['macro_order_accuracy']==.5 and result['constant_prediction_groups']==1
    assert ranking_metrics(groups,[-1.,0.,1.])['macro_order_accuracy']==1.
    assert ranking_metrics(groups,[1.,0.,-1.])['macro_order_accuracy']==0.


def test_training_ranking_changes_only_ranking_arm():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from train_group_ranking import train_epoch
    from train_skempi_ablation import train_epoch as original
    class Score(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.w=torch.nn.Parameter(torch.tensor(.01))
            self.register_buffer('target_scale',torch.tensor(2.))
        def forward(self,h,coordinates,partner,mask):
            return self.w*(h[:,:,0]*mask).sum(1)/mask.sum(1)
    graphs={str(i):dict(h=torch.full((2,128),float(i)),coordinates=torch.zeros(2,3),partner=torch.tensor([False,True])) for i in range(4)}
    absolute=[dict(feature_path=str(i),label_pkd=str(i*.1)) for i in range(4)]
    pairs=[dict(mutant_feature_path=str(i),wt_feature_path='0',delta_pkd=str(i*.2)) for i in range(1,4)]
    for lam in [0.,.3]:
        a,b=Score(),Score()
        args=(graphs,absolute,pairs,torch.ones(4),torch.ones(3),0,42,4,'cpu',4,1,8,lam)
        train_epoch(a,torch.optim.SGD(a.parameters(),lr=.1),*args)
        original(b,torch.optim.SGD(b.parameters(),lr=.1),*args)
        assert torch.equal(a.w,b.w)
    groups=build_ranking_groups([row(i,i) for i in range(4)])
    a=Score()
    logs,_=train_epoch(a,torch.optim.SGD(a.parameters(),lr=.1),*args,
                       ranking_groups=groups,lambda_rank=.1,rank_groups_per_step=1,rank_pairs_per_group=2)
    assert logs['unique_rank_groups']==1 and logs['rank_comparison_draws']==2
    assert not torch.equal(a.w,b.w)
