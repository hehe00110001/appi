"""Reuse identical frozen graphs with a different audited split; no re-encoding."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_affinity import read_tables, digest, atomic_save, json_save


def comparable(tables):
    ignored = {'split', 'original_split', 'group_id'}
    return {(kind, r['csv_line' if kind == 'absolute' else 'skempi_csv_line']):
            {k:v for k,v in r.items() if k not in ignored}
            for (kind, _), rows in tables.items() for r in rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'data/processed/residue_cache_v1.pt')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Output exists; retain it and choose a new path')
    saved = torch.load(args.source, map_location='cpu', weights_only=True)
    previous = dict(saved['metadata'])
    source_data = Path(previous['source_data'])
    assert previous['input_hashes'] == {x.name:digest(x) for x in sorted(source_data.glob('*.csv'))}
    assert comparable(read_tables(source_data,0)) == comparable(read_tables(args.data,0)), 'Records or labels changed'
    expected = {r[k] for rows in read_tables(args.data,0).values() for r in rows
                for k in ['feature_path','wt_feature_path','mutant_feature_path'] if r.get(k)}
    assert expected == set(saved['graphs'])
    saved['metadata'].update(input_hashes={x.name:digest(x) for x in sorted(args.data.glob('*.csv'))},
                             source_data=str(args.data.resolve()),
                             rebind_source_sha256=digest(args.source),
                             rebind_script_sha256=digest(Path(__file__)),
                             rebind_verified='All record IDs, labels and feature paths unchanged; only split/group provenance differs')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(saved,args.out)
    meta=dict(saved['metadata'],file_sha256=digest(args.out))
    json_save(meta,args.out.with_suffix('.json'))
    print(json.dumps(dict(output=str(args.out),graphs=len(saved['graphs']),content_sha256=meta['content_sha256'])),flush=True)


if __name__ == '__main__':main()
