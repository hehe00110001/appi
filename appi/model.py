"""Head for cached residue embeddings; partner masks must come from graph nodes."""

import torch
from torch import nn


def symmetric_pool(h, partner):
    """Pool a single interface, including both mean and sum (output: 6 * D).

    partner is a boolean [N] tensor, identifying two nonempty partner groups.
    A partner may contain multiple chains. Node ordering must match h exactly.
    """
    if h.ndim != 2 or partner.shape != h.shape[:1] or partner.dtype != torch.bool:
        raise ValueError("Expected h [N,D] and boolean partner [N]")
    if not partner.any() or partner.all():
        raise ValueError("Both partners must contain interface residues")
    a, b = h[partner], h[~partner]
    features = []
    for reduce in (lambda x: x.mean(0), lambda x: x.sum(0)):
        u, v = reduce(a), reduce(b)
        features.extend((u + v, (u - v).abs(), u * v))
    return torch.cat(features)


class AffinityHead(nn.Module):
    """Scalar inputs must themselves be invariant to partner exchange.

    Encode two lengths as sum/absolute difference, never as ordered A/B values.
    Swap invariance is evaluated in eval mode (training dropout is stochastic).
    """

    def __init__(self, embedding_dim=128, n_scalars=0, dropout=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_scalars = n_scalars
        self.head = nn.Sequential(
            nn.LayerNorm(6 * embedding_dim + n_scalars),
            nn.Linear(6 * embedding_dim + n_scalars, 256),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1),
        )

    def forward(self, h, partner, scalars=None):
        if h.shape[-1] != self.embedding_dim:
            raise ValueError("Unexpected encoder feature dimension")
        z = symmetric_pool(h, partner)
        if scalars is None:
            scalars = h.new_empty(0)
        if scalars.shape != (self.n_scalars,):
            raise ValueError("Unexpected scalar feature dimension")
        return self.forward_pooled(torch.cat((z, scalars)))

    def forward_pooled(self, z):
        if z.shape[-1] != 6 * self.embedding_dim + self.n_scalars:
            raise ValueError('Unexpected pooled feature dimension')
        return self.head(z).squeeze(-1)


class CachedAffinityRegressor(nn.Module):
    """Predict pKd from frozen pooled features; statistics are training-only.

    No dropout so identical WT/mutant inputs give exactly zero difference,
    including during training. Weight decay supplies baseline regularization.
    """

    def __init__(self, feature_mean, feature_scale, target_mean, target_scale):
        super().__init__()
        for name, value in [('feature_mean', feature_mean), ('feature_scale', feature_scale),
                            ('target_mean', target_mean), ('target_scale', target_scale)]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32).clone())
        self.predictor = AffinityHead(dropout=0.)

    def forward(self, pooled):
        normalized = (pooled - self.feature_mean) / self.feature_scale
        return self.predictor.forward_pooled(normalized) * self.target_scale + self.target_mean

    def delta(self, mutant, wt):
        return self(mutant) - self(wt)

