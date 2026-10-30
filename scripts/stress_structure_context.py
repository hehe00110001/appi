"""Real-cache largest-complex forward/backward smoke; no affinity claim."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from appi.structure_context_data import StructureContextData
from appi.structure_context_model import StructureContextRegressor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    directory = ROOT / 'data/processed/esm2_complex40_v1'
    manifest = json.loads((directory / 'manifest.json').read_text())
    saved = torch.load(ROOT / 'data/processed/residue_cache_complex40_v1.pt', map_location='cpu', weights_only=True)
    def size(p):
        g = manifest['graphs'][p]
        n = len(g['nodes'])
        m = sum(len(manifest['sequences'][c['sequence_id']]) for c in g['chains'].values())
        return n * n + n * m
    largest = max(saved['graphs'], key=size)
    data = StructureContextData.load({largest: saved['graphs'][largest]}, directory)
    del saved
    model = StructureContextRegressor(modality_dropout=0).cuda().train()
    model.start_step()
    began = time.perf_counter()
    # Batch two full complexes, matching the minimum indivisible WT/mutant batch.
    pred = model(**data.collate([largest, largest], 'cuda'))
    pred.square().mean().backward()
    assert torch.isfinite(pred).all() and model.esm_projection[0].weight.grad.norm() > 0
    torch.cuda.synchronize()
    result = dict(passed=True, nodes=data.sizes[largest][0], full_sequence_residues=data.sizes[largest][1], batch=2,
                  seconds=time.perf_counter() - began, peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                  device=torch.cuda.get_device_name(), graph=largest)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
