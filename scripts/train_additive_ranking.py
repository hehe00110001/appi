"""Fixed-split absolute/difference/group-ranking comparison; no endpoint loss."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from appi.group_ranking import (build_ranking_groups, group_audit, ranking_epoch_plan, backward_ranking, ranking_metrics)
from appi.skempi_ablation import is_skempi, balanced_weights, epoch_indices, weighted_mse_sum
from appi.additive_residue_model import AdditiveResidueAffinityRegressor as ResidueAffinityRegressor
from appi.residue_data import collate_graphs, microbatches, training_statistics
from train_affinity import read_tables, pair_endpoint_targets, pair_losses, metrics, digest, atomic_save, json_save


def predictions(model, graphs, rows, kind, device, microbatch_size, attention_limit):
    results = []
    paths = ([r['feature_path'] for r in rows] if kind == 'absolute'
             else [r['mutant_feature_path'] for r in rows])
    lengths = [len(graphs[p]['h']) for p in paths]
    with torch.no_grad():
        for ids in microbatches(range(len(rows)), lengths, microbatch_size, attention_limit,
                               multiplicity=1 if kind == 'absolute' else 2):
            selected = [paths[i] for i in ids]
            if kind == 'pairs':
                selected += [rows[i]['wt_feature_path'] for i in ids]
            p = model(**collate_graphs(graphs, selected, device))
            if kind == 'pairs':
                p = p[:len(ids)] - p[len(ids):]
            results.extend(p.cpu().tolist())
    return results


def evaluate(model, graphs, rows, kind, device, microbatch_size, attention_limit):
    model.eval()
    pred = predictions(model, graphs, rows, kind, device, microbatch_size, attention_limit)
    target = [float(r['label_pkd' if kind == 'absolute' else 'delta_pkd']) for r in rows]
    values = metrics(pred, target)
    error = np.array(pred)-np.array(target)
    values.update(bias=float(error.mean()), std_pred=float(np.std(pred)), std_target=float(np.std(target)))
    return values, pred


def train_epoch(model, optimizer, graphs, absolute, pairs, abs_weights, pair_weights,
                epoch, seed, budget, device, batch_size, microbatch_size, attention_limit, lambda_diff, ranking_groups=None, lambda_rank=0.,
                rank_groups_per_step=4, rank_pairs_per_group=4, rank_margin=.25):
    model.train()
    paths = [r['feature_path'] for r in absolute]
    lengths = [len(graphs[p]['h']) for p in paths]
    y = torch.tensor([float(r['label_pkd']) for r in absolute])
    permutation = epoch_indices(len(absolute), budget, seed+10000+epoch)
    pair_order = epoch_indices(len(pairs), budget, seed+20000+epoch) if lambda_diff else []
    if lambda_diff:
        mutant_paths = [r['mutant_feature_path'] for r in pairs]
        wt_paths = [r['wt_feature_path'] for r in pairs]
        pair_lengths = [max(len(graphs[m]['h']), len(graphs[w]['h'])) for m,w in zip(mutant_paths,wt_paths)]
        delta = torch.tensor([float(r['delta_pkd']) for r in pairs])
    rank_plan = ranking_epoch_plan(ranking_groups, math.ceil(budget/batch_size),
                                   rank_groups_per_step, rank_pairs_per_group, seed+30000+epoch) if lambda_rank else []
    rank_total = 0.
    component_abs_sum = np.zeros(2)
    component_count = 0
    totals = np.zeros(2)
    steps = 0
    for start in range(0,budget,batch_size):
        ids = permutation[start:start+batch_size]
        optimizer.zero_grad(set_to_none=True)
        sums = np.zeros(2)
        for chunk in microbatches(ids,lengths,microbatch_size,attention_limit):
            pred = model(**collate_graphs(graphs,[paths[i] for i in chunk],device))
            if model.last_components is not None:
                component_abs_sum += np.array([float(v.abs().sum()) for v in model.last_components])
                component_count += len(chunk)
            loss = weighted_mse_sum(pred,y[chunk].to(device),abs_weights[chunk].to(device),model.target_scale,len(ids))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite absolute loss')
            loss.backward(); sums[0] += float(loss.detach())
        if lambda_diff:
            dids = pair_order[start:start+batch_size]
            for chunk in microbatches(dids,pair_lengths,microbatch_size,attention_limit,multiplicity=2):
                selected = [mutant_paths[i] for i in chunk]+[wt_paths[i] for i in chunk]
                pred = model(**collate_graphs(graphs,selected,device))
                loss = weighted_mse_sum(pred[:len(chunk)]-pred[len(chunk):],delta[chunk].to(device),
                                        pair_weights[chunk].to(device),model.target_scale,len(ids))
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite difference loss')
                (lambda_diff*loss).backward(); sums[1] += float(loss.detach())
        if lambda_rank:
            rank_total += backward_ranking(model, graphs, rank_plan[steps], device, microbatch_size,
                                           attention_limit, rank_margin, lambda_rank)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        optimizer.step(); totals += sums*len(ids); steps += 1
    return dict(train_global_abs_scaled=float(component_abs_sum[0]/max(component_count,1)),
                train_local_abs_scaled=float(component_abs_sum[1]/max(component_count,1)),
                train_abs_mse_scaled=float(totals[0]/budget), train_diff_mse_scaled=float(totals[1]/budget),
                absolute_draws=budget, unique_absolute_records=len(set(permutation)),
                pair_draws=len(pair_order), unique_pair_records=len(set(pair_order)),
                train_rank_hinge_scaled=rank_total/steps,
                rank_comparison_draws=sum(map(len,rank_plan)),
                rank_group_draws=len(rank_plan)*rank_groups_per_step,
                unique_rank_groups=len({x[3] for batch in rank_plan for x in batch})),steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'data/processed/joint_training_complex40_v1')
    p.add_argument('--cache', type=Path, default=ROOT/'data/processed/residue_cache_complex40_v1.pt')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arm', choices=['B','C','R'], required=True)
    p.add_argument('--epoch-records', type=int, default=4471)
    p.add_argument('--lambda-diff', type=float, default=.3)
    p.add_argument('--lambda-rank', type=float, default=.1)
    p.add_argument('--rank-deadband', type=float, default=.3)
    p.add_argument('--rank-margin', type=float, default=.25)
    p.add_argument('--rank-groups-per-step', type=int, default=4)
    p.add_argument('--rank-pairs-per-group', type=int, default=4)
    
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=64, help='Effective optimizer batch size, using gradient accumulation')
    p.add_argument('--microbatch-size', type=int, default=8)
    p.add_argument('--attention-limit', type=int, default=1000000, help='Maximum padded B*N*N per microbatch; one large graph is retained')
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--limit', type=int, default=0, help='Smoke test only; 0 uses all records')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after-epoch', type=int, default=0, help='Operational checkpoint stop for resume verification; not part of config')
    args = p.parse_args()
    if (min(args.epochs, args.patience, args.batch_size, args.microbatch_size, args.attention_limit,
            args.layers, args.width, args.heads, args.threads, args.epoch_records, args.rank_groups_per_step, args.rank_pairs_per_group) < 1 or args.width % args.heads
            or args.limit < 0 or args.stop_after_epoch < 0
            or any(not math.isfinite(v) or v < 0 for v in [args.lambda_diff, args.weight_decay, args.lambda_rank, args.rank_deadband, args.rank_margin])
            or not math.isfinite(args.lr) or args.lr <= 0):
        p.error('Invalid parameters')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable; use a GPU allocation')
    started = time.perf_counter()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sampler = torch.Generator().manual_seed(args.seed + 10000)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tables = read_tables(args.data, args.limit)
    hashes = {p.name: digest(p) for p in sorted(args.data.glob('*.csv'))}
    cache_hash = digest(args.cache)
    saved = torch.load(args.cache, map_location='cpu', weights_only=True)
    if saved['metadata']['input_hashes'] != hashes or saved['metadata']['format_version'] != 1:
        raise ValueError('Consolidated residue cache does not match the split tables')
    graphs = saved['graphs']
    paths = {r[k] for rows in tables.values() for r in rows
             for k in ['feature_path', 'wt_feature_path', 'mutant_feature_path'] if r.get(k)}
    if not paths <= graphs.keys():
        raise ValueError('Missing graphs in residue cache')
    # Release graphs not needed by a smoke test; no effect on full training.
    graphs = {k: v for k, v in graphs.items() if k in paths}
    del saved
    absolute, pairs = tables['absolute', 'train'], tables['pairs', 'train']
    normalization_rows = [r for r in absolute if not is_skempi(r)]
    if not normalization_rows:
        raise ValueError('Non-SKEMPI training data required for shared normalization')
    rank_groups = {split: build_ranking_groups(tables['pairs', split], args.rank_deadband)
                   for split in ['train', 'val', 'test']}
    rank_audit = {split: group_audit(groups) for split, groups in rank_groups.items()}
    if args.arm == 'R' and (not rank_audit['train']['eligible_groups'] or args.lambda_rank <= 0):
        raise ValueError('Ranking arm needs positive weight and eligible groups')
    if args.arm == 'B': pairs = []
    if args.epoch_records < max(len(absolute), len(pairs)):
        raise ValueError('Epoch budget must retain all active training records')
    abs_weights = balanced_weights(absolute, 'absolute').float()
    pair_weights = balanced_weights(pairs, 'pairs').float() if pairs else torch.empty(0)
    manifests = {}
    for kind, rows, weights in [('absolute', absolute, abs_weights), ('pairs', pairs, pair_weights)]:
        manifest = [dict(source_line=r['csv_line' if kind=='absolute' else 'skempi_csv_line'],
                         source=r['Source Data Set'] if kind=='absolute' else 'SKEMPI pairs',
                         pdb=r['PDB'].upper() if kind=='absolute' else r['#Pdb'].split('_')[0].upper(),
                         group_id=r['group_id'], weight=float(w)) for r,w in zip(rows,weights)]
        manifests[kind] = manifest
    manifest_hash = hashlib.sha256(json.dumps(manifests, sort_keys=True).encode()).hexdigest()
    for (kind, _), rows in tables.items():
        if not all(math.isfinite(float(r['label_pkd' if kind == 'absolute' else 'delta_pkd'])) for r in rows):
            raise ValueError('Nonfinite label')
    stats = training_statistics(graphs, normalization_rows)
    model = ResidueAffinityRegressor(*stats, width=args.width, layers=args.layers, heads=args.heads).to(device)
    baseline_hash = hashlib.sha256()
    for key, value in model.state_dict().items():
        if not key.startswith('local_score.'):
            baseline_hash.update(key.encode()); baseline_hash.update(value.detach().cpu().numpy().tobytes())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    initial_hash = hashlib.sha256()
    for key, value in model.state_dict().items():
        initial_hash.update(key.encode())
        initial_hash.update(value.detach().cpu().numpy().tobytes())
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ['resume', 'stop_after_epoch']}
    config.update(input_hashes=hashes, cache_sha256=cache_hash, initial_state_sha256=initial_hash.hexdigest(), baseline_initial_state_sha256=baseline_hash.hexdigest(),
                  model_sha256=digest(ROOT/'appi/additive_residue_model.py'), base_model_sha256=digest(ROOT/'appi/residue_model.py'), readout='global plus zero-initialized bias-free local sum', data_code_sha256=digest(ROOT/'appi/residue_data.py'),
                  trainer_sha256=digest(Path(__file__)), shared_utilities_sha256=digest(ROOT/'scripts/train_affinity.py'),
                  torch_version=str(torch.__version__), trainable_parameters=sum(p.numel() for p in model.parameters()),
                  training_counts=dict(absolute=len(absolute), pairs=len(pairs)),
                  ranking_code_sha256=digest(ROOT/'appi/group_ranking.py'), ranking_audit=rank_audit,
                  effective_lambda_rank=args.lambda_rank if args.arm=='R' else 0,
                  ranking_sampling='Equal experimental groups; 4 uniform comparison draws/group by default; full group coverage each epoch',
                  ranking_group_key='WT feature path + reference + method + raw temperature + WT pKd (12 significant digits); missing metadata isolated',
                  ranking_scale='Hinge(pKd margin - signed pKd prediction gap) / shared training target SD',
                  training_manifest_sha256=manifest_hash,
                  ablation_utilities_sha256=digest(ROOT/'appi/skempi_ablation.py'),
                  normalization_record_ids=[r['csv_line'] for r in normalization_rows],
                  pair_absolute_weight=0, effective_lambda_diff=args.lambda_diff if args.arm in ('C','R') else 0,
                  weighting='Equal sources, then equal PDBs within each source; pairs equal PDBs; mean-one weights',
                  epoch_sampling='Complete shuffled passes plus prefix to fixed epoch-records; independent absolute/pair seeds',
                  selection='minimum absolute validation RMSE; test only after epoch selection',
                  normalization='All arms use only non-SKEMPI absolute TRAIN graphs/labels for identical normalization and initialization',
                  objective='weighted L_abs + lambda_diff * weighted L_diff (C/R) + lambda_rank * equal-group ranking (R); no endpoint loss',
                  architecture='Baseline interaction and global readout plus sum of learned local residue scores; zero local output at initialization',
                  effective_batch='Gradient accumulation preserves per-record mean losses and 64-record default optimizer batches')
    if (args.out/'config.json').exists():
        if json.loads((args.out/'config.json').read_text()) != config:
            raise ValueError('Config changed; select a new output directory')
        if not args.resume:
            raise ValueError('Output exists; use --resume or a new directory')
    else:
        if args.resume:
            raise ValueError('Cannot resume without an existing config')
        json_save(config, args.out/'config.json')
    json_save(manifests, args.out/'training_manifest.json')
    json_save(rank_groups, args.out/'ranking_groups.json')
    start, best, stale = 0, float('inf'), 0
    if args.resume:
        state = torch.load(args.out/'last.pt', map_location=device, weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, best, stale = state['epoch'], state['best'], state['stale']
        sampler.set_state(state['sampler_rng'].cpu())
        torch.set_rng_state(state['rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    setup_seconds = time.perf_counter()-started
    print(json.dumps(dict(event='ready', setup_seconds=setup_seconds, parameters=config['trainable_parameters'],
                          absolute=len(absolute), pairs=len(pairs), device=str(device),
                          device_name=torch.cuda.get_device_name() if device.type=='cuda' else 'CPU')), flush=True)
    train_started = time.perf_counter()
    for epoch in range(start, args.epochs):
        if stale >= args.patience:
            break
        epoch_started = time.perf_counter()
        losses, steps = train_epoch(model, optimizer, graphs, absolute, pairs, abs_weights, pair_weights,
                                   epoch, args.seed, args.epoch_records, device,
                                   args.batch_size, args.microbatch_size, args.attention_limit,
                                   args.lambda_diff if args.arm in ('C','R') else 0.,
                                   ranking_groups=rank_groups['train'], lambda_rank=args.lambda_rank if args.arm=='R' else 0.,
                                   rank_groups_per_step=args.rank_groups_per_step, rank_pairs_per_group=args.rank_pairs_per_group,
                                   rank_margin=args.rank_margin)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        train_seconds = time.perf_counter()-epoch_started
        av, _ = evaluate(model, graphs, tables['absolute', 'val'], 'absolute', device, args.microbatch_size, args.attention_limit)
        dv, delta_predictions = evaluate(model, graphs, tables['pairs', 'val'], 'pairs', device, args.microbatch_size, args.attention_limit)
        vr = ranking_metrics(rank_groups['val'], delta_predictions)
        vr.pop('groups')
        validation_seconds = time.perf_counter()-epoch_started-train_seconds
        improved = av['rmse'] < best
        if improved:
            best, stale = av['rmse'], 0
        else:
            stale += 1
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch+1,
                     best=best, stale=stale, sampler_rng=sampler.get_state(), rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [], config=config)
        if improved:
            atomic_save(state, args.out/'best.pt')
        atomic_save(state, args.out/'last.pt')
        record = dict(epoch=epoch+1, **losses, val_absolute=av, val_delta_pkd=dv, val_ranking=vr,
                      best_epoch_updated=improved, optimizer_steps=steps, train_seconds=train_seconds,
                      validation_seconds=validation_seconds, epoch_seconds=time.perf_counter()-epoch_started)
        json_save(record, args.out/f'epoch_{epoch+1:04d}.json')
        print(json.dumps(record), flush=True)
        if args.stop_after_epoch and epoch+1 >= args.stop_after_epoch:
            print('Stopped after committed checkpoint as requested', flush=True)
            return
    state = torch.load(args.out/'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(state['model'])
    result = dict(arm=args.arm, seed=args.seed, best_epoch=state['epoch'], smoke_test=bool(args.limit), metrics={})
    for split in ['val', 'test']:
        for kind in ['absolute', 'pairs']:
            rows = tables[kind, split]
            m, pred = evaluate(model, graphs, rows, kind, device, args.microbatch_size, args.attention_limit)
            result['metrics'][kind+'_'+split] = m
            if kind == 'pairs':
                result.setdefault('ranking', {})[split] = ranking_metrics(rank_groups[split], pred)
            with (args.out/f'predictions_{kind}_{split}.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['source_line', 'split', 'target', 'prediction', 'unit'])
                writer.writeheader()
                for r, v in zip(rows, pred):
                    writer.writerow(dict(source_line=r['csv_line' if kind=='absolute' else 'skempi_csv_line'],
                                         split=split, target=float(r['label_pkd' if kind=='absolute' else 'delta_pkd']),
                                         prediction=v, unit='pKd' if kind=='absolute' else 'delta_pKd'))
    json_save(result, args.out/'metrics.json')
    json_save(dict(setup_seconds=setup_seconds, fit_and_final_evaluation_seconds=time.perf_counter()-train_started,
                   total_seconds=time.perf_counter()-started,
                   peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0),
              args.out/'timing.json')
    print('Completed: ' + str(args.out/'metrics.json'), flush=True)


if __name__ == '__main__':
    main()
