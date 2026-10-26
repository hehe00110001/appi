"""Diagnostic residue-name substitution on the official example; not a physical mutant."""

import csv
import json
from pathlib import Path

import torch

from ppiformer.model.ppiformer import PPIformer
from ppiformer.utils.api import embed
from ppiformer.definitions import PPIFORMER_WEIGHTS_DIR, PPIFORMER_TEST_DATA_DIR


def canonical_node(node):
    parts = node.split(':')
    return (parts[0], *parts[2:])  # retain chain, residue number and insertion code


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    out = Path(__file__).resolve().parents[1] / 'outputs/mutation_sensitivity'
    out.mkdir(parents=True, exist_ok=True)
    source = PPIFORMER_TEST_DATA_DIR / '1bui_A_C.pdb'
    lines = source.read_text().splitlines(keepends=True)
    changed_lines = []
    changed_atoms = 0
    for line in lines:
        if (line.startswith('ATOM  ') and line[21] == 'C'
                and line[22:26].strip() == '16' and line[26] == ' '):
            if line[17:20] != 'SER':
                raise RuntimeError('Expected SER at C:16')
            line = line[:17] + 'ALA' + line[20:]
            changed_atoms += 1
        changed_lines.append(line)
    if not changed_atoms:
        raise RuntimeError('Target residue C:16 was not found')
    # Keep identical basenames to avoid filename/partner metadata differences.
    paths = {}
    for name, text in [('wt', ''.join(lines)), ('mut', ''.join(changed_lines))]:
        path = out / name / source.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        paths[name] = path
    print(f'Substitution: C:16 SER -> ALA; changed {changed_atoms} residue-name fields')
    print('All original atom records and coordinates retained (diagnostic only).')

    model = PPIformer.load_from_checkpoint(
        PPIFORMER_WEIGHTS_DIR / 'masked_modeling.ckpt', map_location='cpu'
    ).eval()
    model.requires_grad_(False)

    # Capture the actual graph entering embed's forward pass, not an independent parse.
    captured = []

    def capture(module, args):
        graph = args[0]
        nodes = graph.node_id
        if nodes and isinstance(nodes[0], (list, tuple)):
            nodes = [node for group in nodes for node in group]
        record = {'node_id': list(nodes)}
        for name in ('x', 'f', 'edge_index', 'virtual_c_beta_vector',
                     'sequence_neighbour_vector_n_to_c',
                     'sequence_neighbour_vector_c_to_n'):
            value = getattr(graph, name, None)
            if torch.is_tensor(value):
                record[name] = value.detach().cpu().clone()
        captured.append(record)

    hook = model.register_forward_pre_hook(capture)
    embeddings, graphs = {}, {}
    try:
        for name, path in [('wt', paths['wt']), ('wt_repeat', paths['wt']),
                           ('mut', paths['mut'])]:
            print(f'Encoding {name} ...', flush=True)
            captured.clear()
            with torch.inference_mode():
                h = embed(model, path).detach().cpu().clone()
            if len(captured) != 1:
                raise RuntimeError('Expected exactly one graph per encoding')
            graph = captured[0]
            if h.shape != (len(graph['node_id']), 128) or not torch.isfinite(h).all():
                raise RuntimeError(f'Invalid embedding: {name}, {tuple(h.shape)}')
            embeddings[name], graphs[name] = h, graph
    finally:
        hook.remove()

    wt, mut = graphs['wt'], graphs['mut']
    keys = [canonical_node(n) for n in wt['node_id']]
    if len(set(keys)) != len(keys):
        raise RuntimeError('Duplicate residue keys')
    for graph in graphs.values():
        if keys != [canonical_node(n) for n in graph['node_id']]:
            raise RuntimeError('Residue identities/order changed')
        for name in wt.keys() - {'node_id', 'f'}:
            if name not in graph or not torch.equal(wt[name], graph[name]):
                raise RuntimeError(f'Geometry/topology changed: {name}')
    changed = (wt['f'] != mut['f']).any(dim=1).nonzero().flatten().tolist()
    target = wt['node_id'].index('C:SER:16')
    if changed != [target] or mut['node_id'][target] != 'C:ALA:16':
        raise RuntimeError(f'Unexpected residue-feature changes: {changed}')
    if not torch.equal(wt['f'], graphs['wt_repeat']['f']):
        raise RuntimeError('Input features are unstable on repeated WT encoding')

    h, repeat, mutant = (embeddings[n] for n in ('wt', 'wt_repeat', 'mut'))
    noise = (h - repeat).abs().max().item()
    difference = (mutant - h).abs()
    maximum = difference.max().item()
    stable = torch.allclose(h, repeat, atol=1e-6, rtol=1e-5)
    sensitive = (not torch.allclose(h, mutant, atol=1e-6, rtol=1e-5)
                 and maximum > max(1e-6, 10 * noise))
    report = {
        'mutation': 'SC16A', 'shape': list(h.shape),
        'node_mapping_preserved': True, 'graph_geometry_preserved': True,
        'changed_input_feature_nodes': changed,
        'wt_repeat_max_abs_difference': noise,
        'mutant_max_abs_difference': maximum,
        'mutant_mean_abs_difference': difference.mean().item(),
        'target_residue_embedding_l2_difference': (mutant[target] - h[target]).norm().item(),
        'wt_repeat_stable': stable, 'mutation_sensitive': sensitive,
        'passed': stable and sensitive,
        'scope': 'Residue-name diagnostic only; no physical mutant geometry or affinity validation.',
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    with (out / 'node_mapping.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row', 'wt_node', 'mut_node', 'embedding_l2_difference'])
        for i, (a, b) in enumerate(zip(wt['node_id'], mut['node_id'])):
            writer.writerow([i, a, b, (mutant[i] - h[i]).norm().item()])
    torch.save(embeddings, out / 'embeddings.pt')
    print(json.dumps(report, indent=2))
    print(f'Report: {out / "report.json"}')
    if not report['passed']:
        raise SystemExit('FAIL: inspect repeatability and mutation sensitivity before proceeding')
    print('PASS: fixed-geometry residue substitution changes the embedding.')


if __name__ == '__main__':
    main()
