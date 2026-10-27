"""Predeclared, label-independent sampling and weights for the SKEMPI ablation."""
from collections import Counter, defaultdict

import torch


def is_skempi(row):
    return 'SKEMPI' in row['Source Data Set'].upper()


def balanced_weights(rows, kind):
    """Mean-one weights: equal sources, equal PDBs per source, equal rows per PDB.

    Pairs have a single source, hence equal PDB weight. All weights are positive.
    """
    if not rows:
        raise ValueError('Cannot weight empty training data')
    keys = [(r['Source Data Set'], r['PDB'].upper()) if kind == 'absolute'
            else ('SKEMPI pairs', r['#Pdb'].split('_')[0].upper()) for r in rows]
    counts = Counter(keys)
    groups = defaultdict(set)
    for source, pdb in keys:
        groups[source].add(pdb)
    return torch.tensor([len(rows)/(len(groups)*len(groups[source])*counts[source, pdb])
                         for source, pdb in keys], dtype=torch.float64)


def epoch_indices(n, budget, seed):
    """Shuffled complete passes plus a shuffled prefix, with a fixed update budget.

    If budget >= n, every row is visited at least once per epoch. Absolute and
    pair streams use independent seeds so B and C have identical absolute order.
    """
    if n < 1 or budget < n:
        raise ValueError('Epoch budget must cover every training row')
    generator = torch.Generator().manual_seed(seed)
    indices = []
    while len(indices) < budget:
        indices.extend(torch.randperm(n, generator=generator).tolist())
    return indices[:budget]


def weighted_mse_sum(prediction, target, weights, scale, denominator):
    """Microbatch contribution to a full optimizer batch's weighted mean loss."""
    return ((((prediction-target)/scale)**2)*weights).sum()/denominator
