"""Train a shared residue-attention affinity score on frozen cached embeddings."""
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
from appi.residue_model import ResidueAffinityRegressor
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


def train_epoch(model, optimizer, graphs, absolute, pairs, endpoints, sampler, device,
                batch_size, microbatch_size, attention_limit, pair_absolute_weight, lambda_diff):
    model.train()
    paths = [r['feature_path'] for r in absolute]
    mutant_paths = [r['mutant_feature_path'] for r in pairs]
    wt_paths = [r['wt_feature_path'] for r in pairs]
    lengths = [len(graphs[p]['h']) for p in paths]
    pair_lengths = [max(len(graphs[m]['h']), len(graphs[w]['h'])) for m, w in zip(mutant_paths, wt_paths)]
    y = torch.tensor([float(r['label_pkd']) for r in absolute])
    delta = torch.tensor([float(r['delta_pkd']) for r in pairs])
    permutation = torch.randperm(len(absolute), generator=sampler).tolist()
    totals = np.zeros(3)
    steps = 0
    for start in range(0, len(permutation), batch_size):
        ids = permutation[start:start+batch_size]
        dids = torch.randint(len(pairs), (len(ids),), generator=sampler).tolist()
        optimizer.zero_grad(set_to_none=True)
        sums = np.zeros(3)
        for chunk in microbatches(ids, lengths, microbatch_size, attention_limit):
            pred = model(**collate_graphs(graphs, [paths[i] for i in chunk], device))
            loss = ((pred-y[chunk].to(device))/model.target_scale).square().sum()/len(ids)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite absolute loss')
            loss.backward()
            sums[0] += float(loss.detach())
        # Preserve duplicate sampled records; each pair contributes once per draw.
        for chunk in microbatches(dids, pair_lengths, microbatch_size, attention_limit, multiplicity=2):
            selected = [mutant_paths[i] for i in chunk] + [wt_paths[i] for i in chunk]
            pred = model(**collate_graphs(graphs, selected, device))
            endpoint_loss, diff_loss = pair_losses(pred[:len(chunk)], pred[len(chunk):],
                                                   endpoints[chunk].to(device), delta[chunk].to(device),
                                                   model.target_scale)
            fraction = len(chunk)/len(ids)
            loss = fraction*(pair_absolute_weight*endpoint_loss + lambda_diff*diff_loss)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite pair loss')
            loss.backward()
            sums[1:] += fraction*np.array([float(endpoint_loss.detach()), float(diff_loss.detach())])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        totals += sums*len(ids)
        steps += 1
    return dict(zip(['train_abs_mse_scaled', 'train_pair_abs_mse_scaled', 'train_diff_mse_scaled'],
                    (totals/len(absolute)).tolist())), steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'data/processed/joint_training_resplit_v1')
    p.add_argument('--cache', type=Path, default=ROOT/'data/processed/residue_cache_v1.pt')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['absolute', 'joint'], required=True)
    p.add_argument('--lambda-diff', type=float, default=.3)
    p.add_argument('--pair-absolute-weight', type=float, default=1.)
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
            args.layers, args.width, args.heads, args.threads) < 1 or args.width % args.heads
            or args.limit < 0 or args.stop_after_epoch < 0
            or any(not math.isfinite(v) or v < 0 for v in [args.lambda_diff, args.pair_absolute_weight, args.weight_decay])
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
    endpoints = pair_endpoint_targets(pairs)
    for (kind, _), rows in tables.items():
        if not all(math.isfinite(float(r['label_pkd' if kind == 'absolute' else 'delta_pkd'])) for r in rows):
            raise ValueError('Nonfinite label')
    stats = training_statistics(graphs, absolute)
    model = ResidueAffinityRegressor(*stats, width=args.width, layers=args.layers, heads=args.heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    initial_hash = hashlib.sha256()
    for key, value in model.state_dict().items():
        initial_hash.update(key.encode())
        initial_hash.update(value.detach().cpu().numpy().tobytes())
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ['resume', 'stop_after_epoch']}
    config.update(input_hashes=hashes, cache_sha256=cache_hash, initial_state_sha256=initial_hash.hexdigest(),
                  model_sha256=digest(ROOT/'appi/residue_model.py'), data_code_sha256=digest(ROOT/'appi/residue_data.py'),
                  trainer_sha256=digest(Path(__file__)), shared_utilities_sha256=digest(ROOT/'scripts/train_affinity.py'),
                  torch_version=str(torch.__version__), trainable_parameters=sum(p.numel() for p in model.parameters()),
                  training_counts=dict(absolute=len(absolute), pairs=len(pairs)),
                  selection='minimum absolute validation RMSE; test only after epoch selection',
                  normalization='Equal weight per unique absolute training graph, then per residue; absolute training label mean/std',
                  objective='L_abs + pair_absolute_weight * L_pair_endpoints + (lambda_diff if joint else 0) * L_diff',
                  architecture='Two-partner symmetric distance-biased residue self-attention; shared attention pooling; invariant log size counts',
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
        losses, steps = train_epoch(model, optimizer, graphs, absolute, pairs, endpoints, sampler, device,
                                   args.batch_size, args.microbatch_size, args.attention_limit,
                                   args.pair_absolute_weight, args.lambda_diff if args.mode == 'joint' else 0.)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        train_seconds = time.perf_counter()-epoch_started
        av, _ = evaluate(model, graphs, tables['absolute', 'val'], 'absolute', device, args.microbatch_size, args.attention_limit)
        dv, _ = evaluate(model, graphs, tables['pairs', 'val'], 'pairs', device, args.microbatch_size, args.attention_limit)
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
        record = dict(epoch=epoch+1, **losses, val_absolute=av, val_delta_pkd=dv,
                      best_epoch_updated=improved, optimizer_steps=steps, train_seconds=train_seconds,
                      validation_seconds=validation_seconds, epoch_seconds=time.perf_counter()-epoch_started)
        json_save(record, args.out/f'epoch_{epoch+1:04d}.json')
        print(json.dumps(record), flush=True)
        if args.stop_after_epoch and epoch+1 >= args.stop_after_epoch:
            print('Stopped after committed checkpoint as requested', flush=True)
            return
    state = torch.load(args.out/'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(state['model'])
    result = dict(mode=args.mode, seed=args.seed, best_epoch=state['epoch'], smoke_test=bool(args.limit), metrics={})
    for split in ['val', 'test']:
        for kind in ['absolute', 'pairs']:
            rows = tables[kind, split]
            m, pred = evaluate(model, graphs, rows, kind, device, args.microbatch_size, args.attention_limit)
            result['metrics'][kind+'_'+split] = m
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
