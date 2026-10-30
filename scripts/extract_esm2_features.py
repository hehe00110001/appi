"""Frozen ESM2-650M layer 33, full-chain inference with audited overlapping long windows."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import torch
import esm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_affinity import atomic_save, json_save, read_tables


def file_digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def window_starts(length, size=1022, stride=766):
    if length <= size:
        return [0]
    return sorted(set(list(range(0, length-size+1, stride))+[length-size]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke-only', action='store_true', help='Encode the exact graphs required by --limit 8; separate output file')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    out = ROOT/'data/processed/esm2_complex40_v1'
    lock = (out/'extract.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((out/'manifest.json').read_text())
    assert 'alignment_audit_sha256' in manifest['metadata'], 'Run audit_esm_alignment.py first'
    selected = manifest['graphs']
    if args.smoke_only:
        tables = read_tables(ROOT/'data/processed/joint_training_complex40_v1', 8)
        paths = {r[k] for rows in tables.values() for r in rows
                 for k in ('feature_path','wt_feature_path','mutant_feature_path') if r.get(k)}
        selected = {k:v for k,v in selected.items() if k in paths}
    sequence_ids = {c['sequence_id'] for g in selected.values() for c in g['chains'].values()}
    sequences = {k:v for k,v in manifest['sequences'].items() if k in sequence_ids}
    checkpoint = Path('/home/weizg/wei/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt')
    regression = checkpoint.with_name('esm2_t33_650M_UR50D-contact-regression.pt')
    config = dict(manifest_sha256=file_digest(out/'manifest.json'), checkpoint_sha256=file_digest(checkpoint),
                  regression_sha256=file_digest(regression),
                  script_sha256=file_digest(Path(__file__)), model='esm2_t33_650M_UR50D', layer=33,
                  window=1022, stride=766, overlap='Arithmetic mean; no truncation',
                  inference_dtype='float32', storage_dtype='float16', fair_esm_version=esm.__version__)
    if (out/'extraction_config.json').exists():
        assert config == json.loads((out/'extraction_config.json').read_text()), 'Cache provenance changed'
    else:
        json_save(config, out/'extraction_config.json')
    # The locally cached official ESM checkpoint includes an argparse Namespace.
    # Scope pickle loading to this known pretrained asset, not dataset caches.
    model, alphabet = esm.pretrained.load_model_and_alphabet_core(
        'esm2_t33_650M_UR50D', torch.load(checkpoint, map_location='cpu', weights_only=False),
        torch.load(regression, map_location='cpu', weights_only=True))
    model.eval().to(args.device)
    converter = alphabet.get_batch_converter()
    cache = out/'sequences'
    cache.mkdir(exist_ok=True)
    for number, (key, seq) in enumerate(sorted(sequences.items()), 1):
        path = cache/(key+'.pt')
        if not path.exists():
            result = torch.zeros(len(seq), 1280)
            count = torch.zeros(len(seq), 1)
            for start in window_starts(len(seq)):
                subseq = seq[start:start+1022]
                _, _, tokens = converter([(key, subseq)])
                with torch.inference_mode():
                    h = model(tokens.to(args.device), repr_layers=[33], return_contacts=False)['representations'][33]
                result[start:start+len(subseq)] += h[0, 1:len(subseq)+1].cpu().float()
                count[start:start+len(subseq)] += 1
            assert (count > 0).all()
            result /= count
            assert torch.isfinite(result).all()
            atomic_save(dict(sequence=seq, h=result.half()), path)
        if number % 100 == 0 or number == 1:
            print(f'Encoded {number}/{len(sequences)} unique sequences', flush=True)
    del model
    torch.cuda.empty_cache()
    # CPU consolidation reuses per-chain embeddings, including WT/mutant variants.
    encoded = {key: torch.load(cache/(key+'.pt'), map_location='cpu', weights_only=True)
               for key in sequences}
    for key, seq in sequences.items():
        assert encoded[key]['sequence'] == seq
    graphs = {}
    for name, record in selected.items():
        chains = record['chains']
        h = torch.stack([encoded[chains[c]['sequence_id']]['h'][i] for c, i in record['nodes']])
        for node in record.get('ambiguous_nodes', []):
            chain_h = encoded[chains[node['chain']]['sequence_id']]['h']
            h[node['node']] = chain_h[node['indices']].float().mean(0).half()
        pooled, lengths = [], []
        for side in (True, False):
            blocks = [encoded[c['sequence_id']]['h'].float() for c in chains.values() if c['partner'] == side]
            length = sum(map(len, blocks))
            pooled.append(sum(b.sum(0) for b in blocks)/length)
            lengths.append(length)
        graphs[name] = dict(h=h, sequence_pooled=torch.stack(pooled).half(),
                            sequence_lengths=torch.tensor(lengths))
    suffix = '_smoke' if args.smoke_only else ''
    feature_path = out/f'features{suffix}.pt'
    atomic_save(dict(metadata=manifest['metadata'], extraction_config=config, graphs=graphs), feature_path)
    json_save(dict(graphs=len(graphs), sequences=len(encoded), complete=True,
                   smoke_only=args.smoke_only, features_sha256=file_digest(feature_path)), out/f'complete{suffix}.json')
    print('Complete', flush=True)


if __name__ == '__main__':
    main()
