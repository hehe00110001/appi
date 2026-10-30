"""External16 diagnostics for a preselected checkpoint; never selects or calibrates."""
import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from appi.structure_context_model import StructureContextRegressor
from appi.structure_context_data import StructureContextData, file_digest
from train_structure_context import evaluate
from train_affinity import metrics, json_save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--checkpoint', choices=['best', 'best_absolute'], default='best')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    torch.set_num_threads(4)
    config = json.loads((args.run / 'config.json').read_text())
    for name, expected in config['code_hashes'].items():
        if file_digest(ROOT / name) != expected:
            raise ValueError('Training code changed: ' + name)
    path = args.run / 'joint' / (args.checkpoint + '.pt')
    saved = torch.load(path, map_location=args.device, weights_only=True)
    if saved['config'] != config:
        raise ValueError('Checkpoint/config mismatch')
    model = StructureContextRegressor(**config['model']).to(args.device)
    model.load_state_dict(saved['model'])
    model.set_stage('joint')
    cache = ROOT / 'outputs/binder16_residue_v1/residue_features.pt'
    graphs = torch.load(cache, map_location='cpu', weights_only=True)
    data = StructureContextData.load(graphs, config['arguments']['esm_directory'], config['input_hashes'])
    if data.provenance['extraction'] != config['esm']['extraction']:
        raise ValueError('External ESM encoder differs from training')
    label_path = ROOT / 'data/manifests/binder16_labels_user_20261026.csv'
    reference = json.loads((ROOT / 'outputs/binder16_residue_v1/config.json').read_text())
    if file_digest(label_path) != reference['labels_sha256']:
        raise ValueError('External labels changed')
    rows = list(csv.DictReader(label_path.open()))
    eval_args = SimpleNamespace(device=args.device, microbatch_size=4, attention_limit=1000000)
    _, predictions = evaluate(model, data, [dict(feature_path=r['sample_id'], label_pkd=r['pkd']) for r in rows], 'absolute', eval_args)
    target, pred = np.array([float(r['pkd']) for r in rows]), np.array(predictions)
    result = dict(checkpoint_sha256=file_digest(path), epoch=saved['epoch'], smoke_test=bool(config['arguments']['limit']),
                  label_sha256=file_digest(label_path), structure_cache_sha256=file_digest(cache), esm=data.provenance,
                  policy='Previously inspected diagnostic only; no calibration, no checkpoint choice; report censored-label exclusion', metrics={})
    for scope in ('all16', 'without_boundary15'):
        for protein in ('all', 'IL17A', 'IL23R'):
            ids = [i for i, r in enumerate(rows) if (scope == 'all16' or r['sample_id'] != '23r-91_il23r') and (protein == 'all' or r['target'] == protein)]
            result['metrics'][scope + '/' + protein] = metrics(pred[ids], target[ids])
    directory = args.run / ('evaluation_' + args.checkpoint)
    directory.mkdir(exist_ok=True)
    with (directory / 'external16.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[*rows[0], 'predicted_pkd'])
        writer.writeheader()
        writer.writerows(dict(**r, predicted_pkd=v) for r, v in zip(rows, predictions))
    json_save(result, directory / 'external16.json')
    print('External diagnostics written: ' + str(directory), flush=True)


if __name__ == '__main__':
    main()
