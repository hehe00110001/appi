"""Padding and bounded microbatches for cached per-residue features."""
import torch


def collate_graphs(graphs, paths, device):
    if not paths:
        raise ValueError('Cannot collate an empty graph batch')
    n = max(len(graphs[p]['h']) for p in paths)
    h = torch.zeros(len(paths), n, 128)
    coordinates = torch.zeros(len(paths), n, 3)
    partner = torch.zeros(len(paths), n, dtype=torch.bool)
    mask = torch.zeros_like(partner)
    for i, path in enumerate(paths):
        g = graphs[path]
        size = len(g['h'])
        h[i, :size], coordinates[i, :size] = g['h'], g['coordinates']
        partner[i, :size], mask[i, :size] = g['partner'], True
    return {k: v.to(device) for k, v in dict(h=h, coordinates=coordinates, partner=partner, mask=mask).items()}


def microbatches(indices, lengths, max_batch, max_attention_elements, multiplicity=1):
    """Bound B*N^2, retaining order and every record. A single large graph is never dropped."""
    chunk, largest = [], 0
    for i in indices:
        size = max(largest, lengths[i])
        if chunk and (len(chunk) >= max_batch or multiplicity * (len(chunk)+1) * size**2 > max_attention_elements):
            yield chunk
            chunk, largest = [], 0
        chunk.append(i)
        largest = max(largest, lengths[i])
    if chunk:
        yield chunk


def training_statistics(graphs, absolute_rows):
    """Equal weight per unique absolute training graph, then per residue within graph."""
    paths = sorted({r['feature_path'] for r in absolute_rows})
    means, seconds = [], []
    for path in paths:
        h = graphs[path]['h'].double()
        means.append(h.mean(0))
        seconds.append(h.square().mean(0))
    mean = torch.stack(means).mean(0)
    scale = (torch.stack(seconds).mean(0) - mean.square()).clamp_min(1e-12).sqrt()
    y = torch.tensor([float(r['label_pkd']) for r in absolute_rows])
    return mean.float(), scale.float(), y.mean(), y.std(unbiased=False).clamp_min(1e-6)
