"""Consolidate existing residue embeddings without running PPIformer or changing splits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_affinity import read_tables, digest, atomic_save, json_save


def node_key(node):
    parts = node.split(':')
    return (parts[0], int(parts[2]), parts[3] if len(parts) > 3 else '')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'data/processed/joint_training_resplit_v1')
    p.add_argument('--out', type=Path, default=ROOT/'data/processed/residue_cache_v1.pt')
    args = p.parse_args()
    if args.out.exists():
        p.error('Output exists; retain the original cache and choose a new path')
    torch.set_num_threads(2)
    started = time.perf_counter()
    tables = read_tables(args.data, 0)
    hashes = {p.name: digest(p) for p in sorted(args.data.glob('*.csv'))}
    paths = sorted({r[k] for rows in tables.values() for r in rows
                    for k in ['feature_path', 'wt_feature_path', 'mutant_feature_path'] if r.get(k)})
    graphs, keys, provenance = {}, {}, {}
    content_hash = hashlib.sha256()
    for i, path in enumerate(paths, 1):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        g = {k: saved[k].detach().cpu().clone() for k in ['h', 'coordinates', 'partner']}
        g['h'], g['coordinates'] = g['h'].float(), g['coordinates'].float()
        n = len(g['h'])
        if (g['h'].shape != (n, 128) or g['coordinates'].shape != (n, 3)
                or g['partner'].shape != (n,) or g['partner'].dtype != torch.bool
                or not g['partner'].any() or g['partner'].all()
                or not torch.isfinite(g['h']).all() or not torch.isfinite(g['coordinates']).all()):
            raise ValueError('Invalid residue cache: ' + path)
        keys[path] = [node_key(node) for node in saved['node_id']]
        if len(keys[path]) != n or len(set(keys[path])) != n:
            raise ValueError('Invalid node identity mapping: ' + path)
        graphs[path] = g
        provenance[path] = {k: saved[k] for k in ['structure_sha256', 'config_sha256', 'geometry']}
        content_hash.update(path.encode())
        for k in ['h', 'coordinates', 'partner']:
            content_hash.update(g[k].numpy().tobytes())
        if i % 500 == 0:
            print(f'Consolidated {i}/{len(paths)} graphs', flush=True)
    checked = set()
    for split in ['train', 'val', 'test']:
        for row in tables['pairs', split]:
            mut, wt = row['mutant_feature_path'], row['wt_feature_path']
            if keys[mut] != keys[wt]:
                raise ValueError('WT/mutant node identities differ')
            if any(not torch.equal(graphs[mut][k], graphs[wt][k]) for k in ['coordinates', 'partner']):
                raise ValueError('WT/mutant fixed geometry or partner masks differ')
            checked.add((mut, wt))
    lengths = torch.tensor([len(g['h']) for g in graphs.values()], dtype=torch.float32)
    metadata = dict(format_version=1, input_hashes=hashes, source_data=str(args.data.resolve()),
                    preparation_script_sha256=digest(Path(__file__)), content_sha256=content_hash.hexdigest(),
                    n_graphs=len(graphs), n_nodes=int(lengths.sum()), min_nodes=int(lengths.min()),
                    max_nodes=int(lengths.max()), mean_nodes=float(lengths.mean()),
                    node_quantiles={str(q): float(torch.quantile(lengths, q)) for q in [.5, .9, .99]},
                    unique_pairs_checked=len(checked), torch_version=str(torch.__version__),
                    preprocessing_seconds=time.perf_counter()-started,
                    scope='Existing frozen residue embeddings, coordinates and partner masks; no encoding or filtering')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(dict(graphs=graphs, metadata=metadata, provenance=provenance), args.out)
    metadata['file_sha256'] = digest(args.out)
    json_save(metadata, args.out.with_suffix('.json'))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
