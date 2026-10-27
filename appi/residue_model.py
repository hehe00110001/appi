"""Shared affinity score with distance-biased residue attention and learned pooling."""
import math

import torch
from torch import nn


class ResidueInteractionBlock(nn.Module):
    def __init__(self, width=128, heads=4, radial_features=16):
        super().__init__()
        if width % heads:
            raise ValueError('Width must be divisible by attention heads')
        self.heads, self.head_dim = heads, width // heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.distance_bias = nn.Sequential(nn.Linear(radial_features, 32), nn.GELU(), nn.Linear(32, heads))
        # Same/opposite partner is unchanged when the two partners exchange names.
        self.partner_bias = nn.Embedding(2, heads)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, h, radial, relation, mask):
        batch, nodes, width = h.shape
        q, k, v = self.qkv(self.norm1(h)).reshape(batch, nodes, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
        bias = (self.distance_bias(radial) + self.partner_bias(relation)).permute(0, 3, 1, 2)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim) + bias
        logits = logits.masked_fill(~mask[:, None, None, :], torch.finfo(logits.dtype).min)
        attended = (logits.softmax(-1) @ v).transpose(1, 2).reshape(batch, nodes, width)
        h = h + self.output(attended)
        h = h + self.ffn(self.norm2(h))
        return h.masked_fill(~mask[..., None], 0.)


class ResidueAffinityRegressor(nn.Module):
    """One standalone score for either WT or mutant; no mutation site is required.

    Shared partner pooling and invariant combinations enforce partner exchange
    symmetry. Coordinates enter only through distances. No dropout is used.
    """
    def __init__(self, feature_mean, feature_scale, target_mean, target_scale,
                 width=128, layers=2, heads=4):
        super().__init__()
        if layers < 1:
            raise ValueError('At least one interaction layer is required')
        for name, value in [('feature_mean', feature_mean), ('feature_scale', feature_scale),
                            ('target_mean', target_mean), ('target_scale', target_scale)]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32).clone())
        if self.feature_mean.shape != (128,) or self.feature_scale.shape != (128,):
            raise ValueError('Expected 128-dimensional residue normalization')
        self.input_projection = nn.Linear(128, width)
        self.register_buffer('distance_centers', torch.linspace(0., 30., 16))
        self.blocks = nn.ModuleList([ResidueInteractionBlock(width, heads, 16) for _ in range(layers)])
        self.pool_norm = nn.LayerNorm(width)
        self.pool_gate = nn.Sequential(nn.Linear(width, width // 2), nn.Tanh(), nn.Linear(width // 2, 1))
        self.score = nn.Sequential(nn.LayerNorm(3 * width + 2), nn.Linear(3 * width + 2, 256),
                                   nn.GELU(), nn.Linear(256, 1))

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
        return self.score(z).squeeze(-1) * self.target_scale + self.target_mean

    def delta(self, mutant, wt):
        return self(**mutant) - self(**wt)
