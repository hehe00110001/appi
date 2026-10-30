"""Full-chain sequence memory with exact cached PPIformer node correspondence."""
import hashlib
import json
from pathlib import Path
import torch


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


class StructureContextData:
    def __init__(self, graphs, manifest, bank):
        self.graphs, self.manifest, self.bank = graphs, manifest, bank
        self.sizes = {}
        for path, graph in graphs.items():
            record = manifest['graphs'][path]
            nodes = record['nodes']
            if len(nodes) != len(graph['h']) or graph['h'].shape[1] != 128:
                raise ValueError('PPIformer/sequence node mismatch: ' + path)
            for i, (chain, position) in enumerate(nodes):
                info = record['chains'][chain]
                if bool(graph['partner'][i]) != info['partner'] or not 0 <= position < len(bank[info['sequence_id']]):
                    raise ValueError('Invalid node mapping: ' + path)
            for entry in record.get('ambiguous_nodes', []):
                chain, _ = nodes[entry['node']]
                if chain != entry['chain'] or any(not 0 <= x < len(bank[record['chains'][chain]['sequence_id']]) for x in entry['indices']):
                    raise ValueError('Invalid ambiguous mapping: ' + path)
            if not torch.isfinite(graph['h']).all() or not torch.isfinite(graph['coordinates']).all():
                raise ValueError('Nonfinite structure: ' + path)
            if not graph['partner'].any() or graph['partner'].all():
                raise ValueError('Missing binding partner: ' + path)
            self.sizes[path] = (len(nodes), sum(len(bank[c['sequence_id']]) for c in record['chains'].values()))

    @classmethod
    def load(cls, graphs, directory, input_hashes=None):
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text())
        config = json.loads((directory / 'extraction_config.json').read_text())
        if file_digest(directory / 'manifest.json') != config['manifest_sha256']:
            raise ValueError('ESM extraction manifest changed')
        if input_hashes is not None and manifest['metadata']['input_hashes'] != input_hashes:
            raise ValueError('ESM split provenance mismatch')
        sequence_ids = sorted({c['sequence_id'] for p in graphs for c in manifest['graphs'][p]['chains'].values()})
        bank, hashes = {}, {}
        for sid in sequence_ids:
            path = directory / 'sequences' / (sid + '.pt')
            saved = torch.load(path, map_location='cpu', weights_only=True)
            sequence = saved['sequence']
            if sequence != manifest['sequences'][sid] or hashlib.sha256(sequence.encode()).hexdigest() != sid:
                raise ValueError('Sequence identity mismatch: ' + sid)
            h = saved['h']
            if h.shape != (len(sequence), 1280) or not torch.isfinite(h).all():
                raise ValueError('Invalid full-chain embedding: ' + sid)
            bank[sid], hashes[sid] = h, file_digest(path)
        instance = cls(graphs, manifest, bank)
        instance.provenance = dict(extraction=config, sequence_files=hashes,
                                   sequence_set_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())
        return instance

    def esm_statistics(self, paths):
        """Equal unique TRAIN complex weight, then equal full-chain residue weight."""
        moments = {}
        means, seconds = [], []
        for path in sorted(set(paths)):
            ids = [c['sequence_id'] for c in self.manifest['graphs'][path]['chains'].values()]
            for sid in ids:
                if sid not in moments:
                    h = self.bank[sid].double()
                    moments[sid] = (h.sum(0), h.square().sum(0), len(h))
            length = sum(moments[sid][2] for sid in ids)
            means.append(sum(moments[sid][0] for sid in ids) / length)
            seconds.append(sum(moments[sid][1] for sid in ids) / length)
        mean = torch.stack(means).mean(0)
        scale = (torch.stack(seconds).mean(0) - mean.square()).clamp_min(1e-12).sqrt()
        return mean.float(), scale.float()

    def collate(self, paths, device, include_memory=True):
        if not paths:
            raise ValueError('Empty batch')
        b, n = len(paths), max(self.sizes[p][0] for p in paths)
        result = dict(h=torch.zeros(b, n, 128), coordinates=torch.zeros(b, n, 3),
                      partner=torch.zeros(b, n, dtype=torch.bool), mask=torch.zeros(b, n, dtype=torch.bool))
        if include_memory:
            m = max(self.sizes[p][1] for p in paths)
            result.update(memory=torch.zeros(b, m, 1280), memory_mask=torch.zeros(b, m, dtype=torch.bool),
                          memory_relation=torch.zeros(b, n, m, dtype=torch.uint8))
        for i, path in enumerate(paths):
            g, record = self.graphs[path], self.manifest['graphs'][path]
            count = len(g['h'])
            for key in ('h', 'coordinates', 'partner'):
                result[key][i, :count] = g[key]
            result['mask'][i, :count] = True
            if not include_memory:
                continue
            offsets, offset = {}, 0
            for chain, info in sorted(record['chains'].items()):
                h = self.bank[info['sequence_id']]
                offsets[chain] = offset
                result['memory'][i, offset:offset + len(h)] = h
                result['memory_mask'][i, offset:offset + len(h)] = True
                for j, (node_chain, position) in enumerate(record['nodes']):
                    relation = 2 if node_chain == chain else int(bool(g['partner'][j]) == info['partner'])
                    result['memory_relation'][i, j, offset:offset + len(h)] = relation
                    if node_chain == chain:
                        result['memory_relation'][i, j, offset + position] = 3
                offset += len(h)
            for entry in record.get('ambiguous_nodes', []):
                # All tied alignments receive the same correspondence bias; no arbitrary anchor.
                positions = torch.tensor(entry['indices']) + offsets[entry['chain']]
                result['memory_relation'][i, entry['node'], positions] = 3
        return {key: value.to(device) for key, value in result.items()}

    def microbatches(self, examples, max_batch, attention_limit, include_memory=True):
        """Examples are lists of graph paths; keep WT/mutant inseparable.

        Bound padded B*(N^2+N*M), excluding heads/layers. A single oversized
        example is retained, so this is a batching heuristic, not an OOM guarantee.
        """
        chunk, n, m, batch = [], 0, 0, 0
        for i, paths in enumerate(examples):
            new_n = max(n, *(self.sizes[p][0] for p in paths))
            new_m = max(m, *(self.sizes[p][1] for p in paths)) if include_memory else 0
            cost = (batch + len(paths)) * (new_n ** 2 + new_n * new_m)
            if chunk and (len(chunk) >= max_batch or cost > attention_limit):
                yield chunk
                chunk, n, m, batch = [], 0, 0, 0
            chunk.append(i)
            n = max(n, *(self.sizes[p][0] for p in paths))
            m = max(m, *(self.sizes[p][1] for p in paths)) if include_memory else 0
            batch += len(paths)
        if chunk:
            yield chunk
