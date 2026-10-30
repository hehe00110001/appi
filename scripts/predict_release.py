"""Score compatible frozen PPIformer/ESM caches with an exported APPI model."""
import argparse
import csv
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from appi.structure_context_data import StructureContextData, file_digest
from appi.structure_context_model import StructureContextRegressor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=ROOT)
    parser.add_argument('--features', type=Path, required=True,
                        help='Directory with residue_features.pt, manifest.json, extraction_config.json, config.json and sequences/')
    parser.add_argument('--ids', nargs='+', help='Optional sample IDs; default all cached complexes')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    manifest = json.loads((args.release/'manifest.json').read_text())
    for file, expected in manifest['files_sha256'].items():
        if file_digest(args.release/file) != expected:
            raise ValueError('Release checksum mismatch: ' + file)
    config = json.loads((args.release/'config.json').read_text())
    # Also check the executing implementation, not only files inside the bundle.
    for name in ('structure_context_model.py', 'structure_context_data.py'):
        if file_digest(ROOT/'appi'/name) != manifest['files_sha256']['appi/'+name]:
            raise ValueError('Executing model/data implementation differs from release')
    graphs = torch.load(args.features/'residue_features.pt', map_location='cpu', weights_only=True)
    names = sorted(graphs) if args.ids is None else args.ids
    if not names or len(set(names)) != len(names) or not set(names) <= set(graphs):
        raise ValueError('Sample IDs must be nonempty, unique and present in the graph cache')
    graphs = {name: graphs[name] for name in names}
    data = StructureContextData.load(graphs, args.features)
    for key in ('checkpoint_sha256', 'regression_sha256', 'model', 'layer', 'window',
                'stride', 'overlap', 'inference_dtype', 'storage_dtype', 'fair_esm_version'):
        if data.provenance['extraction'][key] != config['esm_extraction'][key]:
            raise ValueError('Incompatible ESM extraction: ' + key)
    extraction = json.loads((args.features/'config.json').read_text())
    for key, expected in config['ppi_feature_requirements'].items():
        if extraction.get(key) != expected:
            raise ValueError('Incompatible PPIformer extraction: ' + key)
    model = StructureContextRegressor(**config['model']).to(args.device)
    model.load_state_dict(torch.load(args.release/'model.pt', map_location='cpu', weights_only=True))
    model.set_stage('joint')
    model.eval().requires_grad_(False)
    rows = []
    with torch.inference_mode():
        for ids in data.microbatches([[n] for n in names], 4, 1000000, True):
            selected = [names[i] for i in ids]
            values = model(**data.collate(selected, args.device)).cpu().tolist()
            if not all(torch.isfinite(torch.tensor(values))):
                raise ValueError('Nonfinite prediction')
            rows.extend(dict(sample_id=n, predicted_pkd=v) for n, v in zip(selected, values))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'predicted_pkd'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Scored {len(rows)} complexes; higher predicted pKd means stronger predicted binding.')


if __name__ == '__main__':
    main()
