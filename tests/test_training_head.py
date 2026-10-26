import unittest
import torch
from appi.model import CachedAffinityRegressor, AffinityHead, symmetric_pool


class CachedHeadTests(unittest.TestCase):
    def test_cached_and_residue_paths_agree(self):
        torch.manual_seed(7)
        head=AffinityHead(dropout=0.)
        h=torch.randn(9,128);partner=torch.arange(9)<4
        self.assertTrue(torch.allclose(head(h,partner),head.forward_pooled(symmetric_pool(h,partner))))

    def test_difference_cancels_bias_and_reverses_sign(self):
        model=CachedAffinityRegressor(torch.zeros(768),torch.ones(768),7.,2.)
        model.train()
        wt=torch.randn(5,768);mut=torch.randn(5,768)
        self.assertTrue(torch.equal(model.delta(wt,wt),torch.zeros(5)))
        self.assertTrue(torch.equal(model.delta(mut,wt),-model.delta(wt,mut)))
        model.delta(mut,wt).sum().backward()
        self.assertTrue(torch.equal(model.predictor.head[-1].bias.grad,torch.zeros(1)))
        self.assertGreater(model.predictor.head[1].weight.grad.abs().sum().item(),0.)


if __name__=='__main__': unittest.main()
