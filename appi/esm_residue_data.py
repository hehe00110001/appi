"""Join frozen modalities; sequence-only inputs use complete chains, never the crop."""
import torch
from appi.residue_data import microbatches


def join_graphs(base, esm, mode):
    graphs = {}
    for path, g in base.items():
        e = esm[path]
        if mode == 'seq':
            h = torch.zeros(2, 1408)
            h[:, 128:] = e['sequence_pooled'].float()
            h[:, 0] = e['sequence_lengths'].float()
            graphs[path] = dict(h=h, coordinates=torch.zeros(2, 3), partner=torch.tensor([True, False]))
        else:
            if len(g['h']) != len(e['h']):
                raise ValueError('Node mapping size mismatch')
            graphs[path] = dict(g, h=torch.cat((g['h'].float(), e['h'].float()), -1))
    return graphs


def collate_graphs(graphs, paths, device):
    n = max(len(graphs[p]['h']) for p in paths)
    h, coordinates = torch.zeros(len(paths), n, 1408), torch.zeros(len(paths), n, 3)
    partner = torch.zeros(len(paths), n, dtype=torch.bool)
    mask = torch.zeros_like(partner)
    for i, path in enumerate(paths):
        g = graphs[path]
        size = len(g['h'])
        h[i, :size], coordinates[i, :size] = g['h'], g['coordinates']
        partner[i, :size], mask[i, :size] = g['partner'], True
    return {k: v.to(device) for k, v in dict(h=h, coordinates=coordinates, partner=partner, mask=mask).items()}


def backward_ranking(model, graphs, comparisons, device, microbatch_size, attention_limit, margin, coefficient):
    lengths = [max(len(graphs[a]['h']), len(graphs[b]['h'])) for a, b, _, _ in comparisons]
    total = 0.
    for ids in microbatches(range(len(comparisons)), lengths, microbatch_size, attention_limit, multiplicity=2):
        paths = [comparisons[i][0] for i in ids]+[comparisons[i][1] for i in ids]
        prediction = model(**collate_graphs(graphs, paths, device))
        signs = prediction.new_tensor([comparisons[i][2] for i in ids])
        loss = torch.relu(margin-signs*(prediction[:len(ids)]-prediction[len(ids):])).sum()/model.target_scale/len(comparisons)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite ranking loss')
        (coefficient*loss).backward()
        total += float(loss.detach())
    return total
