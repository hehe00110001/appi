"""Conservative experimental groups and equal-group mutation ranking supervision."""
from collections import defaultdict
import hashlib
import json
import math

import numpy as np
import torch
from scipy.stats import spearmanr

from appi.residue_data import collate_graphs, microbatches
from appi.skempi_ablation import epoch_indices


def build_ranking_groups(rows, deadband=.3):
    """Group by WT graph, study, method, reported temperature and WT measurement.

    Missing metadata isolates a record. Replicate mutant graphs within the same
    experimental group are averaged, so duplicate rows do not multiply rank weight.
    WT measurement is a reference-identity key, not an outcome threshold.
    """
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        fields = [str(row.get(k, '')).strip() for k in
                  ('wt_feature_path', 'Reference', 'Method', 'Temperature')]
        missing = any(v.lower() in ('', 'nan', 'none', 'unknown') for v in fields)
        wt = float(row['pkd_wt'])
        if not math.isfinite(wt) or not math.isfinite(float(row['pkd_mut'])):
            raise ValueError('Nonfinite experimental target')
        key = (*fields, format(wt, '.12g'), str(i) if missing else '')
        buckets[key].append(i)
    groups = []
    for key, ids in sorted(buckets.items()):
        variants = defaultdict(list)
        for i in ids:
            variants[rows[i]['mutant_feature_path']].append(i)
        paths = sorted(variants)
        members = [variants[p] for p in paths]
        targets = [float(np.mean([float(rows[i]['pkd_mut']) for i in ix])) for ix in members]
        comparisons = [(i, j, 1. if targets[i] > targets[j] else -1.)
                       for i in range(len(paths)) for j in range(i+1, len(paths))
                       if abs(targets[i]-targets[j]) > deadband]
        groups.append(dict(id=hashlib.sha256(json.dumps(key).encode()).hexdigest(),
                           key=list(key), paths=paths, members=members, targets=targets,
                           comparisons=comparisons, records=len(ids),
                           pdb=rows[ids[0]]['#Pdb'].split('_')[0].upper(),
                           homology=rows[ids[0]]['group_id']))
    return groups


def group_audit(groups):
    eligible = [g for g in groups if g['comparisons']]
    return dict(groups=len(groups), eligible_groups=len(eligible),
                eligible_records=sum(g['records'] for g in eligible),
                unique_variants=sum(len(g['paths']) for g in groups),
                comparisons=sum(len(g['comparisons']) for g in eligible),
                eligible_pdbs=len({g['pdb'] for g in eligible}),
                eligible_homology_groups=len({g['homology'] for g in eligible}))


def ranking_epoch_plan(groups, steps, groups_per_step, pairs_per_group, seed):
    """Full shuffled group passes plus prefix; uniform comparisons within groups.

    Each selected group has exactly the same number of comparison draws. Random
    streams are independent of absolute/difference sampling and model randomness.
    """
    eligible = [g for g in groups if g['comparisons']]
    if not eligible:
        raise ValueError('No eligible ranking groups')
    order = epoch_indices(len(eligible), steps*groups_per_step, seed)
    rng = torch.Generator().manual_seed(seed+1)
    plan = []
    for start in range(0, len(order), groups_per_step):
        batch = []
        for gid in order[start:start+groups_per_step]:
            g = eligible[gid]
            draws = torch.randint(len(g['comparisons']), (pairs_per_group,), generator=rng)
            for index in draws.tolist():
                i, j, sign = g['comparisons'][index]
                batch.append((g['paths'][i], g['paths'][j], sign, g['id']))
        plan.append(batch)
    return plan


def backward_ranking(model, graphs, comparisons, device, microbatch_size,
                     attention_limit, margin, coefficient):
    """Exact sampled objective despite microbatching: never split a comparison.

    Equal comparisons per selected group makes the comparison mean equal to the
    mean of group means. Scores/margin are pKd; divide hinge by training target SD.
    """
    lengths = [max(len(graphs[a]['h']), len(graphs[b]['h'])) for a,b,_,_ in comparisons]
    total = 0.
    for ids in microbatches(range(len(comparisons)), lengths, microbatch_size,
                           attention_limit, multiplicity=2):
        paths = [comparisons[i][0] for i in ids]+[comparisons[i][1] for i in ids]
        prediction = model(**collate_graphs(graphs, paths, device))
        signs = prediction.new_tensor([comparisons[i][2] for i in ids])
        gap = prediction[:len(ids)]-prediction[len(ids):]
        loss = torch.relu(margin-signs*gap).sum()/model.target_scale/len(comparisons)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite ranking loss')
        (coefficient*loss).backward()
        total += float(loss.detach())
    return total


def ranking_metrics(groups, delta_predictions):
    """Same-reference deltas have the same ordering as absolute mutant scores.

    Constant predictions score 0.5 in order accuracy and remain in the denominator.
    Spearman is reported separately with its defined/undefined group counts.
    """
    records = []
    for g in groups:
        if not g['comparisons']:
            continue
        pred = np.array([np.mean([delta_predictions[i] for i in ix]) for ix in g['members']])
        credits = []
        for i,j,sign in g['comparisons']:
            gap = pred[i]-pred[j]
            credits.append(.5 if abs(gap) < 1e-10 else float(sign*gap > 0))
        rho = float(spearmanr(pred, g['targets']).statistic) if np.ptp(pred)>1e-10 else None
        records.append(dict(group=g['id'], pdb=g['pdb'], homology=g['homology'],
                            n=len(pred), comparisons=len(credits), order_accuracy=float(np.mean(credits)),
                            spearman=rho))
    defined = [r['spearman'] for r in records if r['spearman'] is not None]
    return dict(eligible_groups=len(records),
                macro_order_accuracy=float(np.mean([r['order_accuracy'] for r in records])) if records else None,
                macro_spearman=float(np.mean(defined)) if defined else None,
                defined_spearman_groups=len(defined), constant_prediction_groups=len(records)-len(defined),
                groups=records)
