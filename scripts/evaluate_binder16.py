"""Evaluate frozen residue-affinity checkpoints on 16 external structures; no fitting."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch_geometric.data import Batch
from ppiformer.model.ppiformer import PPIformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from appi.residue_model import ResidueAffinityRegressor
from appi.residue_data import collate_graphs
from audit_joint_splits import observed, read, write
from audit_ppb_residues import check_row, read_residues
from build_ppb_features import interface, node_key, sha, atomic_json
from train_affinity import metrics, atomic_save


def subsets(rows):
    for scope in ['user_labels_all16', 'exclude_23r91_boundary']:
        for target in ['all', 'IL17A', 'IL23R']:
            ids = [i for i, r in enumerate(rows)
                   if (scope == 'user_labels_all16' or r['sample_id'] != '23r-91_il23r')
                   and (target == 'all' or r['target'] == target)]
            yield scope + '/' + target, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structures', type=Path, default=Path('/home/weizg/wei/fangyh/af3/IL23R_IL17A/all_pdb_files'))
    parser.add_argument('--labels', type=Path, default=ROOT/'data/manifests/binder16_labels_user_20261026.csv')
    parser.add_argument('--runs', type=Path, default=ROOT/'runs/affinity_residue_v1')
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/binder16_residue_v1')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = read(args.labels)
    assert len(rows) == len({r['sample_id'] for r in rows}) == 16
    assert {r['sample_id'] for r in rows} == {p.stem for p in args.structures.glob('*.pdb')}
    y = np.array([float(r['pkd']) for r in rows])
    assert np.isfinite(y).all()
    # Use the old table for sequence identity only; labels come exclusively from the user's table.
    sequence_table = args.structures.parent/'data.csv'
    with sequence_table.open(encoding='utf-8-sig', newline='') as f:
        source = list(csv.reader(f))
    assert source[0] == ['binder', 'sequence', 'BLI', 'receptor', 'sequence']
    sequences = {r[0].lower()+'_'+r[3].lower(): r for r in source[1:] if any(r)}
    audit_path = ROOT/'data/processed/joint_split_audit_v1/summary.json'
    audit = json.loads(audit_path.read_text())
    preprocessing = json.loads((ROOT/'data/processed/ppb_ppiformer/config.json').read_text())
    checkpoint = ROOT/'external/PPIformer/weights/masked_modeling.ckpt'
    assert sha(checkpoint) == preprocessing['checkpoint_sha256']
    screened = {r['csv_line']: r for r in read(ROOT/'data/processed/joint_split_audit_v1/absolute_screened.csv')}
    training = read(ROOT/'data/processed/joint_training_resplit_v1/absolute_train.csv')
    assert all('external_chain_similarity' not in screened[r['csv_line']]['exclusion_reasons'] for r in training)
    structure_hashes = {}
    for r in rows:
        name = r['sample_id']; p = args.structures/(name+'.pdb')
        structure_hashes[name] = sha(p)
        assert structure_hashes[name] == audit['external_files'][str(p)]
        chains = observed(p); raw = sequences[name]
        assert set(chains) == {'A', 'B'} and chains['A'] == raw[1].strip() and chains['B'] == raw[4].strip()
        assert r['target'] == raw[3]
        review = check_row({'Ligand Chains':'A', 'Receptor Chains':'B', 'Mutations':''}, read_residues(p))
        assert not review['issues'] and not review['warnings'], (name, review)
    model_paths = {f'{mode}_seed{seed}': args.runs/f'{mode}_seed{seed}'/'best.pt'
                   for mode in ['absolute', 'joint'] for seed in [42, 43, 44]}
    config = dict(labels_path=str(args.labels.resolve()), labels_sha256=sha(args.labels),
                  sequence_table_sha256=sha(sequence_table), structure_hashes=structure_hashes,
                  encoder_sha256=sha(checkpoint), radius_angstrom=preprocessing['radius_angstrom'], device='cpu',
                  script_sha256=sha(__file__), preprocessing_script_sha256=sha(ROOT/'scripts/build_ppb_features.py'),
                  checkpoints={name:sha(p) for name,p in model_paths.items()},
                  selection='Existing absolute-validation-selected best.pt for all six runs; no external-label selection or fitting',
                  label_policy='Use user-supplied pKd for all16, including 23r-22=7.37 and 23r-91=12; separately exclude 23r-91 because legacy BLI is <1pM',
                  isolation='Structure hashes match previous external-chain screen; no external_chain_similarity hits in retained absolute training records. Pretraining overlap not established.')
    if (args.out/'config.json').exists():
        assert json.loads((args.out/'config.json').read_text()) == config, 'Output provenance changed; choose a new output directory'
    atomic_json(args.out/'config.json', config)
    cache_path = args.out/'residue_features.pt'
    if cache_path.exists():
        graphs = torch.load(cache_path, map_location='cpu', weights_only=True)
    else:
        # The verified official Lightning checkpoint contains configuration objects.
        previous = os.environ.get('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD')
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
        try:
            encoder = PPIformer.load_from_checkpoint(checkpoint, map_location='cpu').eval()
        finally:
            if previous is None:
                os.environ.pop('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', None)
            else:
                os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = previous
        encoder.requires_grad_(False)
        cfg = OmegaConf.create(encoder.hparams['run_cfg'])
        transforms = [t for t in cfg.val_dataloader.pretransform if t['_target_'] != 'ppiformer.data.transforms.DDGLabelPretransform']
        assert len(transforms) == 1 and transforms[0]['_target_'] == 'ppiformer.data.transforms.PDBToPyGPretransform'
        transform = hydra.utils.instantiate(transforms[0])
        graphs = {}
        for r in rows:
            name = r['sample_id']; p = args.structures/(name+'.pdb')
            cropped = args.out/'interfaces'/(name+'.pdb')
            selected = interface(p, ['A'], ['B'], config['radius_angstrom'], cropped)
            graph = transform(cropped)
            ids = list(graph.node_id)
            assert {node_key(n) for n in ids} == selected and len(ids) == len(selected)
            partner = torch.tensor([n.split(':')[0] == 'A' for n in ids], dtype=torch.bool)
            assert partner.any() and (~partner).any()
            with torch.inference_mode():
                h = encoder(Batch.from_data_list([graph])).detach().cpu().clone()
            assert h.shape == (len(ids),128) and torch.isfinite(h).all() and torch.isfinite(graph.x).all()
            graphs[name] = dict(h=h, coordinates=graph.x.cpu().clone(), partner=partner, node_id=ids)
            print(json.dumps(dict(event='encoded',sample=name,nodes=len(ids))), flush=True)
        atomic_save(graphs, cache_path)
        del encoder
    assert set(graphs) == {r['sample_id'] for r in rows}
    report = dict(label_policy=config['label_policy'], per_model={}, seed_summary={}, ensemble={}, boundary_predictions={})
    predictions, long_rows = {}, []
    names = [r['sample_id'] for r in rows]
    for name,path in model_paths.items():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        c, state = saved['config'], saved['model']
        assert c['model_sha256'] == sha(ROOT/'appi/residue_model.py')
        assert c['data_code_sha256'] == sha(ROOT/'appi/residue_data.py')
        recorded = json.loads((path.parent/'metrics.json').read_text())
        assert saved['epoch'] == recorded['best_epoch'] and not recorded['smoke_test']
        model = ResidueAffinityRegressor(*(state[k] for k in ['feature_mean','feature_scale','target_mean','target_scale']),
                                         width=c['width'], layers=c['layers'], heads=c['heads'])
        model.load_state_dict(state);model.eval();model.requires_grad_(False)
        with torch.inference_mode():
            p = np.array([float(model(**collate_graphs(graphs,[n],'cpu')).item()) for n in names])
        assert np.isfinite(p).all()
        predictions[name] = p
        report['per_model'][name] = {group:metrics(p[ids],y[ids]) for group,ids in subsets(rows)}
        for r,value in zip(rows,p):
            long_rows.append(dict(**r,model=name,best_epoch=saved['epoch'],predicted_pkd=float(value),interface_nodes=len(graphs[r['sample_id']]['h'])))
        report['boundary_predictions'][name] = float(p[names.index('23r-91_il23r')])
        print(name,json.dumps(report['per_model'][name]),flush=True)
    for mode in ['absolute','joint']:
        matrix = np.stack([predictions[f'{mode}_seed{s}'] for s in [42,43,44]])
        report['ensemble'][mode] = {group:metrics(matrix.mean(0)[ids],y[ids]) for group,ids in subsets(rows)}
        report['seed_summary'][mode] = {}
        for group,_ in subsets(rows):
            values = [report['per_model'][f'{mode}_seed{s}'][group] for s in [42,43,44]]
            report['seed_summary'][mode][group] = {metric:dict(mean=statistics.mean(r[metric] for r in values),sd=statistics.stdev(r[metric] for r in values)) for metric in ['pearson','spearman','rmse','mae']}
    wide_rows = []
    for i,r in enumerate(rows):
        item = dict(r)
        for name,values in predictions.items():item[name+'_pkd'] = float(values[i])
        for mode in ['absolute','joint']:
            values = [predictions[f'{mode}_seed{s}'][i] for s in [42,43,44]]
            item[mode+'_mean_pkd'] = statistics.mean(values)
            item[mode+'_seed_sd'] = statistics.stdev(values)
        wide_rows.append(item)
    write(args.out/'predictions_long.csv',long_rows)
    write(args.out/'predictions.csv',wide_rows)
    atomic_json(args.out/'metrics.json',report)
    atomic_json(args.out/'timing.json',dict(total_seconds=time.perf_counter()-started))
    print('Completed: '+str(args.out/'metrics.json'),flush=True)


if __name__ == '__main__':
    main()
