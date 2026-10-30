"""Structure-valued attention, full-chain sequence routing, one shared pKd score.

Frozen encoder caches are inputs. ESM never supplies a readout/value shortcut;
its context affects only the logits of the structural message-passing blocks.
"""
import math
import torch
from torch import nn


class StructureRoutingBlock(nn.Module):
    def __init__(self, width, context_width, heads):
        super().__init__()
        self.heads, self.dim = heads, width // heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.context_qk = nn.Linear(context_width, 2 * width, bias=False)
        self.context_norm = nn.LayerNorm(context_width, elementwise_affine=False)
        self.geometry = nn.Linear(16, heads, bias=False)
        self.relation = nn.Embedding(2, heads)
        self.output = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.last_terms = {}

    def forward(self, s, context, radial, same_partner, mask):
        b, n, w = s.shape
        q, k, v = self.qkv(self.norm(s)).view(b, n, 3, self.heads, self.dim).permute(2, 0, 3, 1, 4).unbind(0)
        cq, ck = self.context_qk(self.context_norm(context)).view(b, n, 2, self.heads, self.dim).permute(2, 0, 3, 1, 4).unbind(0)
        terms = [q @ k.transpose(-2, -1), q @ ck.transpose(-2, -1), cq @ k.transpose(-2, -1)]
        valid = (mask[:, None, :, None] & mask[:, None, None, :]).expand(-1, self.heads, -1, -1)
        with torch.no_grad():
            self.last_terms = {name: float((t / math.sqrt(self.dim))[valid].square().mean().sqrt())
                               for name, t in zip(('structure_structure', 'structure_context', 'context_structure'), terms)}
        logits = sum(terms) / math.sqrt(self.dim)
        logits = logits + (self.geometry(radial) + self.relation(same_partner.long())).permute(0, 3, 1, 2)
        attention = logits.masked_fill(~mask[:, None, None, :], -torch.inf).softmax(-1)
        # Deliberate constraint: v derives exclusively from the structural state.
        update = (attention @ v).transpose(1, 2).reshape(b, n, w)
        s = s + self.output(update)
        return (s + self.ff(s)).masked_fill(~mask[..., None], 0)


class StructureContextRegressor(nn.Module):
    def __init__(self, ppi_mean=None, ppi_scale=None, esm_mean=None, esm_scale=None,
                 target_mean=0., target_scale=1., width=128, context_width=64,
                 heads=4, layers=2, mode='fusion', modality_dropout=.3, contact_cutoff=16.):
        super().__init__()
        if mode not in ('fusion', 'structure') or width % heads or context_width % heads:
            raise ValueError('Invalid mode or attention dimensions')
        if not 0 <= modality_dropout <= 1 or contact_cutoff <= 0 or layers < 1:
            raise ValueError('Invalid dropout, cutoff or layer count')
        for name, value, default in [('ppi_mean', ppi_mean, torch.zeros(128)), ('ppi_scale', ppi_scale, torch.ones(128)),
                                     ('esm_mean', esm_mean, torch.zeros(1280)), ('esm_scale', esm_scale, torch.ones(1280))]:
            self.register_buffer(name, default if value is None else torch.as_tensor(value).float().clone())
        self.register_buffer('target_mean', torch.as_tensor(target_mean).float().clone())
        self.register_buffer('target_scale', torch.as_tensor(target_scale).float().clone())
        self.register_buffer('centers', torch.linspace(0, 30, 16))
        self.mode, self.modality_dropout, self.contact_cutoff = mode, modality_dropout, contact_cutoff
        self.stage, self.context_active = 'joint', mode == 'fusion'
        self.heads, self.context_width = heads, context_width
        self.structure_adapter = nn.Sequential(nn.Linear(128, width), nn.LayerNorm(width), nn.GELU())
        self.esm_projection = nn.Sequential(nn.Linear(1280, context_width), nn.LayerNorm(context_width))
        self.memory_query = nn.Linear(width, context_width)
        self.memory_kv = nn.Linear(context_width, 2 * context_width)
        # Other partner / same partner / same chain / corresponding residue.
        self.memory_relation = nn.Embedding(4, heads)
        self.blocks = nn.ModuleList([StructureRoutingBlock(width, context_width, heads) for _ in range(layers)])
        self.contact_node = nn.Sequential(nn.Linear(128 + width, 64), nn.LayerNorm(64), nn.GELU())
        self.contact_pair = nn.Sequential(nn.Linear(3 * 64 + 16, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU())
        self.score = nn.Sequential(nn.LayerNorm(64 + 3 * (128 + width) + 3),
                                   nn.Linear(64 + 3 * (128 + width) + 3, 128), nn.GELU(), nn.Linear(128, 1))

    def set_stage(self, stage):
        if stage not in ('warmup', 'joint'):
            raise ValueError(stage)
        self.stage = stage
        self.context_active = self.mode == 'fusion' and stage == 'joint'

    def start_step(self):
        """Call once per optimizer update, shared by all tasks and both endpoints."""
        self.context_active = self.mode == 'fusion' and self.stage == 'joint'
        if self.training and self.context_active:
            self.context_active = bool(torch.rand(()) >= self.modality_dropout)

    def shared_balance_parameters(self):
        # Fixed shared structural layers; neither task-specific head nor ESM-only branch.
        return [self.blocks[-1].qkv.weight, self.contact_node[0].weight]

    def forward(self, h, coordinates, partner, mask, memory=None, memory_mask=None, memory_relation=None):
        if not ((mask & partner).any(1) & (mask & ~partner).any(1)).all():
            raise ValueError('Each complex requires residues from both partners')
        raw = ((h.masked_fill(~mask[..., None], 0) - self.ppi_mean) / self.ppi_scale).masked_fill(~mask[..., None], 0)
        xyz = coordinates.masked_fill(~mask[..., None], 0)
        s = self.structure_adapter(raw).masked_fill(~mask[..., None], 0)
        b, n, _ = s.shape
        use_context = self.mode == 'fusion' and self.stage == 'joint' and (not self.training or self.context_active)
        context = s.new_zeros(b, n, self.context_width)
        if use_context:
            if memory is None or memory_mask is None or memory_relation is None or not memory_mask.any(1).all():
                raise ValueError('Full sequence memory and residue mapping required')
            clean = memory.masked_fill(~memory_mask[..., None], 0)
            e = self.esm_projection((clean - self.esm_mean) / self.esm_scale)
            d = self.context_width // self.heads
            q = self.memory_query(s).view(b, n, self.heads, d).transpose(1, 2)
            k, v = self.memory_kv(e).view(b, e.shape[1], 2, self.heads, d).permute(2, 0, 3, 1, 4).unbind(0)
            logits = q @ k.transpose(-2, -1) / math.sqrt(d)
            logits = logits + self.memory_relation(memory_relation.long()).permute(0, 3, 1, 2)
            attention = logits.masked_fill(~memory_mask[:, None, None, :], -torch.inf).softmax(-1)
            context = (attention @ v).transpose(1, 2).reshape(b, n, self.context_width)
        distances = torch.cdist(xyz, xyz, compute_mode='donot_use_mm_for_euclid_dist')
        radial = torch.exp(-.5 * ((distances[..., None] - self.centers) / 2.) ** 2)
        same_partner = partner[:, :, None] == partner[:, None, :]
        for block in self.blocks:
            s = block(s, context, radial, same_partner, mask)
        # Independent raw PPIformer channel survives every sequence-conditioned update.
        node = torch.cat([raw, s], -1)
        contact_node = self.contact_node(node)
        readouts = []
        for i in range(b):
            left, right = mask[i] & partner[i], mask[i] & ~partner[i]
            edge = left[:, None] & right[None, :] & (distances[i] <= self.contact_cutoff)
            a, z = edge.nonzero(as_tuple=True)
            x, y = contact_node[i, a], contact_node[i, z]
            pair = self.contact_pair(torch.cat([x + y, (x - y).abs(), x * y, radial[i, a, z]], -1))
            summed = pair.sum(0)
            mean = summed / max(len(a), 1)
            u, v = node[i, left].mean(0), node[i, right].mean(0)
            counts = torch.stack([torch.log1p(edge.sum()), torch.log1p(left.sum() + right.sum()),
                                  torch.log1p(left.sum() * right.sum())]).to(node.dtype)
            readouts.append(torch.cat([mean, summed / 100., u + v, (u - v).abs(), u * v, counts]))
        return self.target_mean + self.target_scale * self.score(torch.stack(readouts)).squeeze(-1)
