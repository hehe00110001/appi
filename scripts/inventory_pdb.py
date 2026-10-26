"""Read first-model ATOM residue identities without modifying source structures."""

import argparse
import csv
from pathlib import Path


def inventory(directory):
    for path in sorted(directory.glob('*.pdb')):
        chains = {}
        for line in path.read_text().splitlines():
            if line.startswith('ENDMDL'):
                break
            if not line.startswith('ATOM  '):
                continue
            chain = line[21:22].strip() or '<blank>'
            residue = (line[22:26].strip(), line[26:27])
            chains.setdefault(chain, {})[residue] = line[17:20].strip()
        for chain, residues in chains.items():
            yield {'file': str(path.resolve()), 'chain': chain,
                   'observed_residues': len(residues),
                   'below_40': len(residues) < 40}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    rows = list(inventory(args.directory))
    if not rows:
        parser.error('No ATOM residues found in input PDB files')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} chain records to {args.output}')
