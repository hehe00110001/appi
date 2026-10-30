"""Controlled additive-readout ablation; interaction encoder matches the baseline.

Local output has no final bias, is initialized to zero, and sums all valid nodes.
No mutation masks or approximate difference computations are used.
"""
import torch
from torch import nn
from appi.residue_model import ResidueAffinityRegressor


class AdditiveResidueAffinityRegressor(ResidueAffinityRegressor):
    def __init__(self, *stats, width=128, layers=2, heads=4):
        super().__init__(*stats, width=width, layers=layers, heads=heads)
        self.local_score = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1, bias=False))
        nn.init.zeros_(self.local_score[-1].weight)
        self.last_components = None

    def forward(self, h, coordinates, partner, mask):
        if h.ndim != 3 or h.shape[-1] != 128 or coordinates.shape != (*h.shape[:2], 3):
            raise ValueError('Expected padded h [B,N,128] and coordinates [B,N,3]')
        if partner.shape != h.shape[:2] or mask.shape != h.shape[:2]:
            raise ValueError('Partner and validity masks must match nodes')
        if partner.dtype != torch.bool or mask.dtype != torch.bool:
            raise ValueError('Masks must be boolean')
        sides = (mask & partner, mask & ~partner)
        if not all(side.any(1).all() for side in sides):
            raise ValueError('Every interface must contain both partners')
        # Zero padding before arithmetic, including arbitrary/nonfinite padded values.
        h = h.masked_fill(~mask[..., None], 0.)
        coordinates = coordinates.masked_fill(~mask[..., None], 0.)
        h = self.input_projection((h - self.feature_mean) / self.feature_scale)
        h = h.masked_fill(~mask[..., None], 0.)
        distance = torch.cdist(coordinates, coordinates, compute_mode='donot_use_mm_for_euclid_dist')
        radial = torch.exp(-((distance[..., None] - self.distance_centers) / 2.)**2)
        relation = (partner[:, :, None] != partner[:, None, :]).long()
        for block in self.blocks:
            h = block(h, radial, relation, mask)
        h = self.pool_norm(h)
        logits = self.pool_gate(h).squeeze(-1)
        pooled = []
        for side in sides:
            weights = logits.masked_fill(~side, torch.finfo(logits.dtype).min).softmax(-1)
            pooled.append((h * weights[..., None]).sum(1))
        u, v = pooled
        # Retain interface-size information that a normalized attention mean lacks.
        na, nb = [side.sum(1).to(h.dtype) for side in sides]
        sizes = torch.stack((torch.log1p(na + nb), torch.log1p((na - nb).abs())), -1)
        z = torch.cat((u + v, (u - v).abs(), u * v, sizes), -1)
        global_score = self.score(z).squeeze(-1)
        contributions = self.local_score(h).squeeze(-1).masked_fill(~mask, 0.)
        local_score = contributions.sum(1)
        self.last_components = (global_score.detach(), local_score.detach())
        return (global_score + local_score) * self.target_scale + self.target_mean

