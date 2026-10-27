"""Measure an actual largest WT/mutant pair before submitting full residue training."""
import csv
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from appi.residue_model import ResidueAffinityRegressor
from appi.residue_data import collate_graphs
from train_affinity import pair_losses, pair_endpoint_targets


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Run this probe in a GPU allocation')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    saved = torch.load(ROOT/'data/processed/residue_cache_v1.pt', map_location='cpu', weights_only=True)
    graphs = saved['graphs']
    rows = []
    for split in ['train', 'val', 'test']:
        with (ROOT/f'data/processed/joint_training_resplit_v1/pairs_{split}.csv').open(newline='') as f:
            rows.extend(csv.DictReader(f))
    row = max(rows, key=lambda r: len(graphs[r['mutant_feature_path']]['h']))
    paths = [row['mutant_feature_path'], row['wt_feature_path']]
    model = ResidueAffinityRegressor(torch.zeros(128), torch.ones(128), 7., 2.).cuda()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    prediction = model(**collate_graphs(graphs, paths, 'cuda'))
    a, d = pair_losses(prediction[:1], prediction[1:], pair_endpoint_targets([row]).cuda(),
                       torch.tensor([float(row['delta_pkd'])], device='cuda'), model.target_scale)
    (a+.3*d).backward()
    torch.cuda.synchronize()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError('Missing or nonfinite gradients in largest-pair GPU probe')
    report = dict(device=torch.cuda.get_device_name(), nodes=len(graphs[paths[0]]['h']),
                  pdb=row['#Pdb'], seconds=time.perf_counter()-started,
                  peak_memory_bytes=torch.cuda.max_memory_allocated(),
                  trainable_parameters=sum(p.numel() for p in model.parameters()),
                  all_gradients_finite=True)
    out = ROOT/'outputs/residue_gpu_probe.json'
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
