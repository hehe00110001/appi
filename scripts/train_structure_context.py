"""Structure-first difference pretraining followed by shared-score GradNorm training."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from appi.structure_context_model import StructureContextRegressor
from appi.structure_context_data import StructureContextData, file_digest
from appi.gradnorm import GradNorm, gradient_snapshot, combine_gradients, gradient_diagnostics
from appi.residue_data import training_statistics
from appi.skempi_ablation import balanced_weights, epoch_indices, is_skempi, weighted_mse_sum
from appi.group_ranking import build_ranking_groups, ranking_metrics, group_audit
from train_affinity import read_tables, metrics, atomic_save, json_save, pair_endpoint_targets


def examples(rows, kind):
    return [[r['feature_path']] if kind == 'absolute' else [r['mutant_feature_path'], r['wt_feature_path']] for r in rows]


def uses_memory(model):
    return model.mode == 'fusion' and model.stage == 'joint' and (not model.training or model.context_active)


def predict_batch(model, data, selected, kind, device):
    paths = [p for example in selected for p in example]
    scores = model(**data.collate(paths, device, include_memory=uses_memory(model)))
    return scores if kind == 'absolute' else scores[0::2] - scores[1::2]


@torch.no_grad()
def evaluate(model, data, rows, kind, args):
    model.eval()
    pred = []
    batch = examples(rows, kind)
    for ids in data.microbatches(batch, args.microbatch_size, args.attention_limit, uses_memory(model)):
        pred.extend(predict_batch(model, data, [batch[i] for i in ids], kind, args.device).cpu().tolist())
    target = [float(r['label_pkd' if kind == 'absolute' else 'delta_pkd']) for r in rows]
    result = metrics(pred, target)
    result.update(std_pred=float(np.std(pred)), std_target=float(np.std(target)), bias=float(np.mean(np.array(pred) - target)))
    if kind == 'pairs':
        result['zero_change_baseline'] = metrics(np.zeros(len(target)), target)
    return result, pred


def shuffled_targets(rows, seed):
    """Shuffle only auxiliary TRAIN labels within conservative experimental groups."""
    target = torch.tensor([float(r['delta_pkd']) for r in rows])
    source = list(range(len(rows)))
    rng = np.random.default_rng(seed)
    groups = build_ranking_groups(rows)
    for group in groups:
        ids = sorted(i for members in group['members'] for i in members)
        shuffled = rng.permutation(ids).tolist()
        for i, j in zip(ids, shuffled):
            source[i] = j
    shuffled = target[source]
    return shuffled, dict(seed=seed, source_indices=source, groups=len(groups),
                          changed_rows=int((shuffled != target).sum()), total_rows=len(rows),
                          policy='Only training delta targets; within WT/study/method/temperature/WT-measurement groups')


def backward_task(model, data, rows, kind, ids, weights, targets, args):
    selected = [rows[i] for i in ids]
    batch = examples(selected, kind)
    total = 0.
    for local in data.microbatches(batch, args.microbatch_size, args.attention_limit, uses_memory(model)):
        ix = [ids[i] for i in local]
        pred = predict_batch(model, data, [batch[i] for i in local], kind, args.device)
        loss = weighted_mse_sum(pred, targets[ix].to(args.device), weights[ix].to(args.device), model.target_scale, len(ids))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite ' + kind + ' loss')
        loss.backward()
        total += float(loss.detach())
    return total


def train_epoch(model, optimizer, controller, controller_opt, data, tables, weights, targets, args, stage, epoch):
    model.train()
    parameters = list(model.parameters())
    tasks = ['pairs'] if stage == 'warmup' else (['absolute'] if args.supervision == 'absolute' else ['absolute', 'pairs'])
    orders = {kind: epoch_indices(len(tables[kind, 'train']), args.epoch_records,
                                  args.seed + (10000 if kind == 'absolute' else 20000) + epoch + (100000 if stage == 'joint' else 0))
              for kind in tasks}
    records = []
    for start in range(0, args.epoch_records, args.batch_size):
        model.start_step()
        losses, gradients = [], []
        for kind in tasks:
            optimizer.zero_grad(set_to_none=True)
            ids = orders[kind][start:start + args.batch_size]
            losses.append(backward_task(model, data, tables[kind, 'train'], kind, ids, weights[kind], targets[kind], args))
            gradients.append(gradient_snapshot(parameters))
        if len(tasks) == 2:
            norms, cosine = gradient_diagnostics(parameters, model.shared_balance_parameters(), gradients)
            effective_weights = controller.weights().detach().clone()
            controller_opt.zero_grad(set_to_none=True)
            balance_loss = controller.objective(norms.new_tensor(losses), norms)
            balance_loss.backward()
            controller_opt.step()
            record = dict(raw_shared_gradient_norms=norms.tolist(), shared_gradient_cosine=cosine,
                          effective_shared_gradient_norms=(effective_weights * norms).tolist(),
                          gradnorm_objective=float(balance_loss.detach()))
        else:
            effective_weights = torch.ones(1, device=args.device)
            record = {}
        combine_gradients(parameters, gradients, effective_weights)
        branch_norms = {name: sum(float(p.grad.square().sum()) for p in branch.parameters() if p.grad is not None) ** .5
                        for name, branch in [('structure', model.structure_adapter), ('esm', model.esm_projection)]}
        preclip = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        records.append(dict(step=len(records) + 1, context_active=bool(model.context_active), tasks=tasks,
                            losses=losses, weights=effective_weights.tolist(), branch_gradient_norms=branch_norms,
                            last_microbatch_attention_terms=[block.last_terms for block in model.blocks],
                            preclip_gradient_norm=float(preclip), **record))
    return dict(steps=records, optimizer_steps=len(records),
                losses={kind: float(np.average([r['losses'][j] for r in records],
                                               weights=[min(args.batch_size, args.epoch_records - i * args.batch_size) for i in range(len(records))]))
                        for j, kind in enumerate(tasks)},
                draws={kind: len(order) for kind, order in orders.items()},
                unique_records={kind: len(set(order)) for kind, order in orders.items()})


def selection_key(ranking, difference):
    accuracy = ranking['macro_order_accuracy']
    if accuracy is None:
        raise ValueError('No eligible validation ranking groups; cannot use the declared selection rule')
    return (-accuracy, difference['mae'])


def rng_restore(state, device):
    torch.set_rng_state(state['rng'].cpu())
    if device == 'cuda':
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])


def fit_stage(model, data, tables, weights, targets, groups, config, args, stage):
    directory = args.out / stage
    directory.mkdir(exist_ok=True)
    model.set_stage(stage)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    controller = GradNorm(args.gradnorm_alpha).to(args.device)
    controller_opt = torch.optim.Adam(controller.parameters(), lr=args.gradnorm_lr)
    start, best, stale, best_absolute = 0, (float('inf'), float('inf')), 0, float('inf')
    last = directory / 'last.pt'
    if args.resume and last.exists():
        state = torch.load(last, map_location=args.device, weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        controller.load_state_dict(state['controller'])
        controller_opt.load_state_dict(state['controller_optimizer'])
        rng_restore(state, args.device)
        start, best, stale, best_absolute = state['epoch'], tuple(state['best']), state['stale'], state['best_absolute']
    epochs = args.warmup_epochs if stage == 'warmup' else args.epochs
    patience = args.warmup_patience if stage == 'warmup' else args.patience
    if not (directory / 'complete.json').exists():
        for epoch in range(start, epochs):
            if stale >= patience:
                break
            began = time.perf_counter()
            training = train_epoch(model, optimizer, controller, controller_opt, data, tables, weights, targets, args, stage, epoch)
            av, _ = evaluate(model, data, tables['absolute', 'val'], 'absolute', args)
            dv, pred = evaluate(model, data, tables['pairs', 'val'], 'pairs', args)
            ranking = ranking_metrics(groups['val'], pred)
            key = selection_key(ranking, dv)
            improved, absolute_improved = key < best, av['mae'] < best_absolute
            if improved:
                best, stale = key, 0
            else:
                stale += 1
            if absolute_improved:
                best_absolute = av['mae']
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), controller=controller.state_dict(),
                         controller_optimizer=controller_opt.state_dict(), epoch=epoch + 1, stage=stage,
                         best=list(best), stale=stale, best_absolute=best_absolute, config=config,
                         rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else [])
            if improved:
                atomic_save(state, directory / 'best.pt')
            if absolute_improved:
                atomic_save(state, directory / 'best_absolute.pt')
            atomic_save(state, last)
            record = dict(stage=stage, epoch=epoch + 1, **training, val_absolute=av, val_difference=dv,
                          val_ranking=ranking, improved=improved, seconds=time.perf_counter() - began)
            json_save(record, directory / f'epoch_{epoch + 1:04d}.json')
            print(json.dumps({k: v for k, v in record.items() if k not in ('steps', 'val_ranking')}), flush=True)
            if args.stop_stage == stage and args.stop_after_epoch and epoch + 1 >= args.stop_after_epoch:
                print('Stopped at committed stage checkpoint', flush=True)
                return False
        if not last.exists():
            raise RuntimeError('Stage has no checkpoint')
        state = torch.load(last, map_location='cpu', weights_only=True)
        json_save(dict(epochs=state['epoch'], best_key=state['best'], best_sha256=file_digest(directory / 'best.pt')),
                  directory / 'complete.json')
    # Transfer the validation-selected weights, but preserve end-of-stage RNG.
    final = torch.load(last, map_location='cpu', weights_only=True)
    selected = torch.load(directory / 'best.pt', map_location=args.device, weights_only=True)
    model.load_state_dict(selected['model'])
    rng_restore(final, args.device)
    return True


def final_evaluation(model, data, tables, groups, args):
    result = dict(mode=args.mode, supervision=args.supervision, smoke_test=bool(args.limit), checkpoints={})
    model.set_stage('joint')
    for choice in ('best', 'best_absolute'):
        state = torch.load(args.out / 'joint' / (choice + '.pt'), map_location=args.device, weights_only=True)
        model.load_state_dict(state['model'])
        destination = args.out / ('evaluation_' + choice)
        destination.mkdir(exist_ok=True)
        summary = dict(epoch=state['epoch'], selection='validation macro order, tie difference MAE' if choice == 'best' else 'validation absolute MAE', metrics={}, ranking={})
        for split in ('val', 'test'):
            for kind in ('absolute', 'pairs'):
                rows = tables[kind, split]
                m, pred = evaluate(model, data, rows, kind, args)
                summary['metrics'][kind + '_' + split] = m
                if kind == 'pairs':
                    summary['ranking'][split] = ranking_metrics(groups[split], pred)
                with (destination / f'{kind}_{split}.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['source_line', 'target', 'prediction'])
                    writer.writeheader()
                    writer.writerows(dict(source_line=r['csv_line' if kind == 'absolute' else 'skempi_csv_line'],
                                          target=r['label_pkd' if kind == 'absolute' else 'delta_pkd'], prediction=v) for r, v in zip(rows, pred))
        result['checkpoints'][choice] = summary
    json_save(result, args.out / 'metrics.json')


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT / 'data/processed/joint_training_complex40_v1')
    p.add_argument('--cache', type=Path, default=ROOT / 'data/processed/residue_cache_complex40_v1.pt')
    p.add_argument('--esm-directory', type=Path, default=ROOT / 'data/processed/esm2_complex40_v1')
    p.add_argument('--mode', choices=['structure', 'fusion'], default='fusion')
    p.add_argument('--supervision', choices=['absolute', 'transfer', 'shuffled'], default='transfer')
    for name, default in [('seed', 42), ('epochs', 100), ('patience', 20), ('warmup-epochs', 30), ('warmup-patience', 10),
                          ('epoch-records', 4471), ('batch-size', 64), ('microbatch-size', 4), ('attention-limit', 1000000),
                          ('width', 128), ('context-width', 64), ('heads', 4), ('layers', 2), ('threads', 4), ('limit', 0), ('stop-after-epoch', 0)]:
        p.add_argument('--' + name, type=int, default=default)
    for name, default in [('lr', .001), ('weight-decay', .0001), ('gradnorm-lr', .01), ('gradnorm-alpha', 1.5),
                          ('modality-dropout', .3), ('contact-cutoff', 16.), ('rank-deadband', .3)]:
        p.add_argument('--' + name, type=float, default=default)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-stage', choices=['warmup', 'joint'], default='joint')
    args = p.parse_args()
    positive = ['epochs', 'patience', 'warmup_epochs', 'warmup_patience', 'epoch_records', 'batch_size', 'microbatch_size',
                'attention_limit', 'width', 'context_width', 'heads', 'layers', 'threads', 'lr', 'gradnorm_lr', 'contact_cutoff']
    if any(not math.isfinite(getattr(args, k)) or getattr(args, k) <= 0 for k in positive):
        p.error('Positive finite training parameters required')
    if (not 0 <= args.modality_dropout <= 1 or args.width % args.heads or args.context_width % args.heads
        or any(not math.isfinite(v) or v < 0 for v in [args.gradnorm_alpha, args.weight_decay, args.rank_deadband, args.limit, args.stop_after_epoch])):
        p.error('Invalid dimensions, dropout or training parameters')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; request a GPU allocation')
    return args


def main():
    args = arguments()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tables = read_tables(args.data, args.limit)
    hashes = {p.name: file_digest(p) for p in sorted(args.data.glob('*.csv'))}
    saved = torch.load(args.cache, map_location='cpu', weights_only=True)
    if saved['metadata']['input_hashes'] != hashes or saved['metadata']['format_version'] != 1:
        raise ValueError('Structure cache/split mismatch')
    paths = {p for (kind, _), rows in tables.items() for example in examples(rows, kind) for p in example}
    data = StructureContextData.load({p: saved['graphs'][p] for p in sorted(paths)}, args.esm_directory, hashes)
    del saved
    normalization_rows = [r for r in tables['absolute', 'train'] if not is_skempi(r)]
    if not normalization_rows:
        raise ValueError('Non-SKEMPI TRAIN normalization set is empty')
    pm, ps, ym, ys = training_statistics(data.graphs, normalization_rows)
    em, es = data.esm_statistics([r['feature_path'] for r in normalization_rows])
    model_args = dict(width=args.width, context_width=args.context_width, heads=args.heads, layers=args.layers,
                      mode=args.mode, modality_dropout=args.modality_dropout, contact_cutoff=args.contact_cutoff)
    model = StructureContextRegressor(pm, ps, em, es, ym, ys, **model_args).to(args.device)
    groups = {split: build_ranking_groups(tables['pairs', split], args.rank_deadband) for split in ('train', 'val', 'test')}
    if not group_audit(groups['val'])['eligible_groups']:
        raise ValueError('No eligible validation groups')
    weights = {kind: balanced_weights(tables[kind, 'train'], kind) for kind in ('absolute', 'pairs')}
    pair_endpoint_targets(tables['pairs', 'train'])
    targets = {kind: torch.tensor([float(r['label_pkd' if kind == 'absolute' else 'delta_pkd']) for r in tables[kind, 'train']])
               for kind in ('absolute', 'pairs')}
    shuffle = None
    if args.supervision == 'shuffled':
        targets['pairs'], shuffle = shuffled_targets(tables['pairs', 'train'], args.seed + 70000)
    for target in targets.values():
        if not torch.isfinite(target).all():
            raise ValueError('Nonfinite labels')
    source_files = ['appi/structure_context_model.py', 'appi/structure_context_data.py', 'appi/gradnorm.py',
                    'appi/residue_data.py', 'appi/skempi_ablation.py', 'appi/group_ranking.py',
                    'scripts/train_structure_context.py', 'scripts/train_affinity.py']
    config = dict(arguments={k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
                             if k not in ('resume', 'stop_stage', 'stop_after_epoch')},
                  model=model_args, input_hashes=hashes, structure_cache_sha256=file_digest(args.cache),
                  esm=data.provenance, code_hashes={p: file_digest(ROOT / p) for p in source_files},
                  normalization_record_ids=[r['csv_line'] for r in normalization_rows],
                  parameters=sum(p.numel() for p in model.parameters()),
                  selection='max validation experimental-group macro order (>0.3 pKd default); tie min difference MAE',
                  secondary_selection='minimum absolute validation MAE',
                  objective='warmup: difference only, ESM off; joint: GradNorm absolute+difference; absolute control skips warmup',
                  weighting='Mean-one equal source/PDB absolute weights; equal PDB difference weights',
                  sampling='Complete shuffled passes + prefix; same fixed task draw budget; shared optimizer-step ESM dropout',
                  difference='S(mutant)-S(WT) in pKd, experimental delta_pkd; no independent difference head',
                  scalar_cache_only=True, shuffle=shuffle)
    config_path = args.out / 'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError('Configuration/provenance changed; use a new output directory')
        if not args.resume:
            raise ValueError('Output exists; use --resume or a new directory')
    else:
        if args.resume:
            raise ValueError('Cannot resume without config')
        json_save(config, config_path)
    json_save(groups, args.out / 'ranking_groups.json')
    print(json.dumps(dict(event='ready', parameters=config['parameters'], graphs=len(data.graphs), sequences=len(data.bank),
                          mode=args.mode, supervision=args.supervision, device=args.device,
                          device_name=torch.cuda.get_device_name() if args.device == 'cuda' else 'CPU')), flush=True)
    if args.supervision != 'absolute':
        if not fit_stage(model, data, tables, weights, targets, groups, config, args, 'warmup'):
            return
    if not fit_stage(model, data, tables, weights, targets, groups, config, args, 'joint'):
        return
    final_evaluation(model, data, tables, groups, args)
    json_save(dict(complete=True, smoke=bool(args.limit), peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0),
              args.out / 'complete.json')
    print('Completed: ' + str(args.out), flush=True)


if __name__ == '__main__':
    main()
