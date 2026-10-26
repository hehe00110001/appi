import unittest

import torch

from appi.model import AffinityHead, symmetric_pool


class ModelTests(unittest.TestCase):
    def test_partner_exchange_and_residue_order(self):
        torch.manual_seed(0)
        h = torch.randn(7, 128)
        partner = torch.tensor([True] * 3 + [False] * 4)
        model = AffinityHead().eval()
        order = torch.randperm(7)
        with torch.no_grad():
            score = model(h, partner)
            torch.testing.assert_close(score, model(h, ~partner))
            torch.testing.assert_close(score, model(h[order], partner[order]))

    def test_sum_retains_size(self):
        h = torch.ones(4, 2)
        partner = torch.tensor([True, True, False, False])
        z = symmetric_pool(h, partner)
        duplicated = symmetric_pool(h.repeat(2, 1), partner.repeat(2))
        torch.testing.assert_close(z[:6], duplicated[:6])
        self.assertFalse(torch.equal(z[6:], duplicated[6:]))

    def test_missing_partner_rejected(self):
        with self.assertRaises(ValueError):
            symmetric_pool(torch.ones(4, 2), torch.ones(4, dtype=torch.bool))

