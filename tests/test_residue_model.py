import copy
import unittest

import torch

from appi.residue_model import ResidueAffinityRegressor
from appi.residue_data import collate_graphs, microbatches, training_statistics


class ResidueModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.model = ResidueAffinityRegressor(torch.zeros(128), torch.ones(128), 7., 2.,
                                              width=32, layers=2, heads=4)
        self.graphs = {str(i): dict(h=torch.randn(n, 128), coordinates=torch.randn(n, 3)*4,
                                    partner=torch.arange(n) < n//2) for i, n in enumerate([7, 11, 5])}

    def batch(self, paths=('0',)):
        return collate_graphs(self.graphs, list(paths), 'cpu')

    def test_partner_swap_node_permutation_and_rigid_motion(self):
        batch = self.batch()
        score = self.model(**batch)
        swapped = {**batch, 'partner': ~batch['partner']}
        torch.testing.assert_close(score, self.model(**swapped), atol=1e-5, rtol=1e-5)
        order = torch.randperm(7)
        permuted = {k: v[:, order] for k, v in batch.items()}
        torch.testing.assert_close(score, self.model(**permuted), atol=1e-5, rtol=1e-5)
        rotation, _ = torch.linalg.qr(torch.randn(3, 3))
        moved = {**batch, 'coordinates': batch['coordinates'] @ rotation + 13.}
        torch.testing.assert_close(score, self.model(**moved), atol=1e-5, rtol=1e-5)

    def test_padding_and_other_graphs_do_not_change_score(self):
        alone = self.model(**self.batch())
        mixed = self.batch(('0', '1'))
        torch.testing.assert_close(alone, self.model(**mixed)[:1], atol=1e-5, rtol=1e-5)
        mixed['h'][~mixed['mask']] = float('nan')
        mixed['coordinates'][~mixed['mask']] = float('nan')
        torch.testing.assert_close(alone, self.model(**mixed)[:1], atol=1e-5, rtol=1e-5)

    def test_difference_zero_antisymmetry_and_gradients_reach_interactions(self):
        wt = self.batch()
        mut = {k: v.clone() for k, v in wt.items()}
        mut['h'][0, 2] += torch.randn(128)
        self.assertTrue(torch.equal(self.model.delta(wt, wt), torch.zeros(1)))
        torch.testing.assert_close(self.model.delta(mut, wt), -self.model.delta(wt, mut))
        (self.model.delta(mut, wt)-1.).square().sum().backward()
        for parameter in [self.model.blocks[0].qkv.weight,
                          self.model.blocks[0].distance_bias[0].weight,
                          self.model.pool_gate[0].weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def test_single_partner_rejected(self):
        batch = self.batch()
        batch['partner'].fill_(True)
        with self.assertRaisesRegex(ValueError, 'both partners'):
            self.model(**batch)

    def test_microbatch_accumulation_preserves_mean_loss_gradient(self):
        other = copy.deepcopy(self.model)
        targets = torch.tensor([8., 6., 9.])
        pred = self.model(**self.batch(('0', '1', '2')))
        (pred-targets).square().mean().backward()
        for i in range(3):
            ((other(**self.batch((str(i),)))-targets[i]).square()/3).sum().backward()
        for a, b in zip(self.model.parameters(), other.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-4)

    def test_bounded_microbatches_keep_duplicates_and_large_graphs(self):
        indices = [0, 1, 1, 2, 3]
        lengths = [5, 7, 20, 3]
        chunks = list(microbatches(indices, lengths, 3, 200, multiplicity=2))
        self.assertEqual([i for chunk in chunks for i in chunk], indices)
        self.assertIn([2], chunks)
        for chunk in chunks:
            self.assertTrue(len(chunk)==1 or 2*len(chunk)*max(lengths[i] for i in chunk)**2 <= 200)

    def test_normalization_uses_only_unique_training_graphs(self):
        rows = [dict(feature_path='0', label_pkd='7'), dict(feature_path='0', label_pkd='9')]
        before = training_statistics(self.graphs, rows)
        self.graphs['1']['h'].fill_(1e6)
        after = training_statistics(self.graphs, rows)
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(before[0], self.graphs['0']['h'].mean(0))
        self.assertEqual(before[2].item(), 8.)


if __name__ == '__main__':
    unittest.main()
