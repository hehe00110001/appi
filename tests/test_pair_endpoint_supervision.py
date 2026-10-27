from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from train_affinity import pair_endpoint_targets, pair_losses


class PairEndpointTests(unittest.TestCase):
    def test_mean_endpoint_and_difference_losses_have_expected_units(self):
        endpoints = torch.tensor([[11., 9.], [7., 10.]])
        absolute, difference = pair_losses(torch.tensor([10., 7.]), torch.tensor([9., 8.]),
                                           endpoints, torch.tensor([2., -3.]), 2.)
        self.assertEqual(absolute.item(), .3125)
        self.assertEqual(difference.item(), .625)

    def test_absolute_endpoints_anchor_common_shift_while_difference_cancels_it(self):
        bias = torch.tensor(1., requires_grad=True)
        endpoints = torch.tensor([[8., 9.]])
        absolute, difference = pair_losses(endpoints[:, 0]+bias, endpoints[:, 1]+bias,
                                           endpoints, torch.tensor([-1.]), 1.)
        self.assertEqual(difference.item(), 0.)
        self.assertEqual(absolute.item(), 1.)
        absolute.backward()
        self.assertEqual(bias.grad.item(), 2.)

    def test_inconsistent_or_nonfinite_endpoint_labels_are_rejected(self):
        valid = dict(pkd_mut='8.1', pkd_wt='9.3', delta_pkd='-1.2')
        self.assertEqual(tuple(pair_endpoint_targets([valid]).shape), (1, 2))
        for update in [dict(delta_pkd='1.2'), dict(pkd_wt='nan')]:
            with self.assertRaisesRegex(ValueError, 'inconsistent'):
                pair_endpoint_targets([{**valid, **update}])


if __name__ == '__main__':
    unittest.main()
