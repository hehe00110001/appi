"""Match sequences within each partner without assuming CSV list order is chain order."""
from audit_ppb_residues import AA3


def is_subsequence(observed, expected):
    if not observed:
        return False
    it = iter(expected)
    return all(any(x == aa for x in it) for aa in observed)


def match_partner(chains, observed, expected):
    if len(chains) != len(expected):
        raise ValueError('chain_sequence_count_mismatch')
    edges = [[j for j, seq in enumerate(expected) if is_subsequence(obs, seq)] for obs in observed]
    # Preserve supplied order where it already works, including identical chains.
    if all(i in candidates for i, candidates in enumerate(edges)):
        return dict(zip(chains, range(len(chains))))
    owners = {}

    def assign(i, visited):
        for j in edges[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in owners or assign(owners[j], visited):
                owners[j] = i
                return True
        return False

    for i in range(len(chains)):
        if not assign(i, set()):
            raise ValueError(f'observed_sequence_not_in_csv:{chains[i]}')
    return {chains[i]: j for j, i in owners.items()}


def check_sequences(row, residues, mutations):
    changes = {(m['chain'],m['residue_number'],m['insertion_code']): m['mutant'] for m in mutations}
    mappings = {}
    for side in ('Ligand','Receptor'):
        chains = [c.strip() for c in row[f'{side} Chains'].split(',')]
        seqs = [s.strip() for s in row[f'{side} Sequences'].split(',')]
        observed = [''.join(changes.get(k, AA3[next(iter(r['names']))])
                            for k,r in sorted(residues.items()) if k[0] == chain)
                    for chain in chains]
        mappings[side] = match_partner(chains, observed, seqs)
    return mappings
