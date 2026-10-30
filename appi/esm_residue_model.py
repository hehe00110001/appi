"""Sequence-only, ESM residue graph, no-geometry control and multimodal graph."""
import torch
from torch import nn
from appi.residue_model import ResidueInteractionBlock


class ESMResidueAffinityRegressor(nn.Module):
    def __init__(self, feature_mean, feature_scale, target_mean, target_scale,
                 width=128, layers=2, heads=4, mode='fusion', modality_dropout=.3):
        super().__init__()
        if mode not in ('seq', 'esm_graph', 'esm_nogeom', 'fusion'):
            raise ValueError(mode)
        self.mode, self.modality_dropout = mode, modality_dropout
        self.modality_mask = (1., 1.)
        self.evaluation_modality_mask = (1., 1.)
        for name, value in [('feature_mean', feature_mean), ('feature_scale', feature_scale),
                            ('target_mean', target_mean), ('target_scale', target_scale)]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32).clone())
        self.esm_projection = nn.Sequential(nn.Linear(1280, width), nn.LayerNorm(width))
        if mode == 'fusion':
            self.ppi_projection = nn.Sequential(nn.Linear(128, width), nn.LayerNorm(width))
        if mode != 'seq':
            self.register_buffer('distance_centers', torch.linspace(0., 30., 16))
            self.blocks = nn.ModuleList([ResidueInteractionBlock(width, heads) for _ in range(layers)])
            self.pool_norm = nn.LayerNorm(width)
            self.pool_gate = nn.Sequential(nn.Linear(width, width//2), nn.Tanh(), nn.Linear(width//2, 1))
        self.score = nn.Sequential(nn.LayerNorm(3*width+2), nn.Linear(3*width+2, 256), nn.GELU(), nn.Linear(256, 1))

    def start_step(self):
        """One mask for ALL records and WT/mutant/ranking comparisons in an update."""
        self.modality_mask = (1., 1.)
        if self.training and self.mode == 'fusion':
            u = torch.rand(()).item()
            if u < self.modality_dropout/2:
                self.modality_mask = (0., 1.)
            elif u < self.modality_dropout:
                self.modality_mask = (1., 0.)

    def forward(self, h, coordinates, partner, mask):
        sides = (mask & partner, mask & ~partner)
        if h.shape[-1] != 1408 or not all(s.any(1).all() for s in sides):
            raise ValueError('Expected packed PPI128+ESM1280 and two nonempty partners')
        raw = h.masked_fill(~mask[..., None], 0.)
        x = (raw-self.feature_mean)/self.feature_scale
        h = self.esm_projection(x[..., 128:])
        if self.mode == 'fusion':
            ppi_on, esm_on = self.modality_mask if self.training else self.evaluation_modality_mask
            h = esm_on*h + ppi_on*self.ppi_projection(x[..., :128])
        h = h.masked_fill(~mask[..., None], 0.)
        if self.mode == 'seq':
            # Exactly two full-partner sequence means; no structure-derived crop/count.
            u, v = h[:, 0], h[:, 1]
            na, nb = raw[:, 0, 0], raw[:, 1, 0]
        else:
            if self.mode == 'esm_nogeom':
                radial = h.new_zeros((*mask.shape, mask.shape[1], 16))
            else:
                xyz = coordinates.masked_fill(~mask[..., None], 0.)
                distance = torch.cdist(xyz, xyz, compute_mode='donot_use_mm_for_euclid_dist')
                radial = torch.exp(-((distance[..., None]-self.distance_centers)/2.)**2)
            relation = (partner[:, :, None] != partner[:, None, :]).long()
            for block in self.blocks:
                h = block(h, radial, relation, mask)
            h = self.pool_norm(h)
            logits = self.pool_gate(h).squeeze(-1)
            u, v = [(h*logits.masked_fill(~s, torch.finfo(logits.dtype).min).softmax(-1)[..., None]).sum(1)
                    for s in sides]
            na, nb = [s.sum(1).to(h.dtype) for s in sides]
        sizes = torch.stack((torch.log1p(na+nb), torch.log1p((na-nb).abs())), -1)
        z = torch.cat((u+v, (u-v).abs(), u*v, sizes), -1)
        return self.score(z).squeeze(-1)*self.target_scale+self.target_mean

    def delta(self, mutant, wt):
        return self(**mutant)-self(**wt)
