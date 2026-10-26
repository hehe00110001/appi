"""Audit published splits and external chains; never change split assignments.

BLAST screening is an operational similarity screen, not a proof of no homology.
One similar chain is sufficient to flag a complex (conservative policy).
"""
import csv
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path
import subprocess
import argparse
from audit_ppb_residues import read_residues, AA3

ROOT = Path(__file__).resolve().parents[1]


def alignment_passes(identity, qcov, scov, qlen, slen, external=False):
    coverage = qcov if external or qlen <= slen else scov
    return identity >= 30 and coverage >= .8


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed(path):
    chains = defaultdict(list)
    for (chain, number, insertion), residue in read_residues(path).items():
        chains[chain].append(AA3.get(next(iter(residue['names'])), 'X') if len(residue['names']) == 1 else 'X')
    return {k: ''.join(v) for k, v in chains.items()}


def write(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', type=Path, default=ROOT/'data/processed/joint_split_audit_v1')
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT/'data/manifests/ppb_affinity/structure_manifest.csv'
    feature_path = ROOT/'data/processed/ppb_ppiformer_ready_v3/samples.csv'
    pair_path = ROOT/'data/processed/skempi_pairs_v2/paired_candidates.csv'
    manifest, encoded, pairs = map(read, [manifest_path, feature_path, pair_path])
    print(f'Collecting chains from {len(manifest)} published rows and {len(encoded)} encoded rows', flush=True)
    rows = {r['csv_line']:r for r in manifest}
    seq_ids, memberships, sources = {}, defaultdict(set), defaultdict(set)

    def add(seq, rowid=None, source=''):
        seq = seq.strip().upper()
        if not seq or not set(seq) <= set('ACDEFGHIKLMNPQRSTVWYX'):
            raise ValueError('Unsupported sequence: '+source)
        sid = seq_ids.setdefault(seq, f's{len(seq_ids)}')
        if rowid is not None:
            memberships[sid].add(rowid)
        sources[sid].add(source)
        return sid

    # Published sequences include unresolved residues; observed WT sequences
    # additionally cover the WT inputs used for difference supervision.
    for r in manifest:
        for field in ['Ligand Sequences', 'Receptor Sequences']:
            for seq in r[field].split(','):
                add(seq, r['csv_line'], 'published')
    cache = {}
    for r in encoded:
        path = r['structure_path']
        if path not in cache:
            cache[path] = observed(Path(path))
        for chain in (r['Ligand Chains']+','+r['Receptor Chains']).split(','):
            add(cache[path][chain.strip()], r['csv_line'], 'observed_WT')
    external = defaultdict(list)
    external_hashes = {}
    for path in sorted((Path('/home/weizg/wei/fangyh/af3/IL23R_IL17A/all_pdb_files')).glob('*.pdb')):
        external_hashes[str(path)] = sha(path)
        for chain, seq in observed(path).items():
            external[add(seq, source='external')].append(f'{path.name}:{chain}')
    assert len(external_hashes) == 16
    with (out/'database.fasta').open('w') as f, (out/'queries.fasta').open('w') as q:
        for seq, sid in seq_ids.items():
            if memberships[sid]: f.write(f'>{sid}\n{seq}\n')
            if sid in external or any(rows[i]['split'] in ['val','test'] for i in memberships[sid]):
                q.write(f'>{sid}\n{seq}\n')
    (out/'sequence_membership.json').write_text(json.dumps({sid:{'rows':sorted(memberships[sid]),'external':external.get(sid,[]),'sources':sorted(sources[sid])} for sid in seq_ids.values()}))
    print(f'Screening {len(seq_ids)} unique sequences, including {len(external)} external chains', flush=True)
    commands = [
        ['makeblastdb','-in',str(out/'database.fasta'),'-dbtype','prot','-out',str(out/'db')],
        ['blastp','-query',str(out/'queries.fasta'),'-db',str(out/'db'),'-out',str(out/'hits.tsv'),
         '-num_threads',str(args.threads),'-evalue','1e-3','-seg','no','-max_target_seqs',str(len(seq_ids)),
         '-outfmt','6 qseqid sseqid pident length qstart qend sstart send qlen slen evalue bitscore']]
    with (out/'blast.log').open('w') as log:
        for cmd in commands:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    reasons = defaultdict(set)
    evidence = []
    priority = {'train':0,'val':1,'test':2}

    def cross(a, b, reason):
        for i in a:
            for j in b:
                if priority[rows[i]['split']] < priority[rows[j]['split']]:
                    reasons[i].add(reason)

    # Same PDB or identical cached features cannot be shared across splits.
    groups = defaultdict(set)
    for r in manifest: groups[('pdb',r['PDB'].upper())].add(r['csv_line'])
    for r in encoded:
        for field in ['feature_path','wt_feature_path']:
            if r[field]: groups[('feature',r[field])].add(r['csv_line'])
    for (kind, _), ids in groups.items(): cross(ids,ids,kind+'_cross_split')
    with (out/'hits.tsv').open() as f:
        for line in f:
            q,s,identity,length,qs,qe,ss,se,ql,sl,evalue,bits = line.rstrip().split('\t')
            identity = float(identity)
            qcov=(int(qe)-int(qs)+1)/int(ql)
            scov=(int(se)-int(ss)+1)/int(sl)
            # A single HSP must cover 80% of the shorter sequence. Keep the
            # evidence explicit; this is not equivalent to MMseqs clustering.
            if not alignment_passes(identity,qcov,scov,int(ql),int(sl)): continue
            kind = ''
            if q in external and alignment_passes(identity,qcov,scov,int(ql),int(sl),external=True):
                for i in memberships[s]: reasons[i].add('external_chain_similarity')
                kind = 'external'
            if any(rows[i]['split'] != rows[j]['split'] for i in memberships[q] for j in memberships[s]):
                cross(memberships[q],memberships[s],'chain_similarity_cross_split')
                cross(memberships[s],memberships[q],'chain_similarity_cross_split')
                kind = kind+';cross_split'
            if kind:
                evidence.append(dict(query=q,subject=s,identity=identity,qcov=qcov,scov=scov,kind=kind))
    # Exact identity screen does not depend on BLAST sensitivity or HSP length.
    for sid, ids in memberships.items():
        cross(ids,ids,'identical_chain_cross_split')
        if sid in external:
            for i in ids: reasons[i].add('external_chain_identity')
    absolute = []
    for r in encoded:
        absolute.append(dict(r, screening_pass=not reasons[r['csv_line']],
                             exclusion_reasons=';'.join(sorted(reasons[r['csv_line']])), training_eligible=False))
    allowed = {r['csv_line'] for r in absolute if r['screening_pass']}
    paired = []
    for r in pairs:
        ids = r['ppb_csv_lines'].split(',')
        rs = set.union(*(reasons[i] for i in ids))
        passed = all(i in allowed for i in ids)
        paired.append(dict(r, screening_pass=passed,exclusion_reasons=';'.join(sorted(rs)),training_eligible=False))
    write(out/'absolute_screened.csv',absolute)
    write(out/'pairs_screened.csv',paired)
    write(out/'similarity_evidence.csv',evidence)
    summary = {'absolute':dict(Counter((r['split']+':'+str(r['screening_pass'])) for r in absolute)),
               'pairs':dict(Counter((r['split']+':'+str(r['screening_pass'])) for r in paired)),
               'absolute_reasons':dict(Counter(x for r in absolute for x in r['exclusion_reasons'].split(';') if x)),
               'inputs':{str(p):sha(p) for p in [manifest_path,feature_path,pair_path]},
               'external_files':external_hashes, 'script_sha256':sha(Path(__file__)), 'commands':commands,
               'policy':'Published splits preserved. Lower-priority split excluded on conflict (test > val > train). Any chain match >=30% identity, >=80% shorter-chain HSP coverage, E<=1e-3. External requires >=80% external-chain coverage. Exact chain, PDB and feature checks included. Operational conservative screen; not proof of absent remote homology. training_eligible remains false until review.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['absolute','pairs','absolute_reasons']},indent=2))


if __name__=='__main__': main()
