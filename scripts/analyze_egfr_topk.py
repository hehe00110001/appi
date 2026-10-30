"""Exploratory frozen-score top-k diagnostics; no fitting or model selection."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/adaptyv_egfr_multiseed_a6000_v1'


def cutoff_weights(scores, k):
    scores = np.asarray(scores)
    assert len(scores) >= k and np.isfinite(scores).all()
    cutoff = np.sort(scores)[-k]
    above, ties = scores > cutoff, scores == cutoff
    weights = above.astype(float) + ties * ((k - above.sum()) / ties.sum())
    assert np.isclose(weights.sum(), k)
    return weights


def evaluate(rows, scores):
    known = [r for r in rows if r['binding_evaluation'] == 'True']
    quant = [r for r in rows if r['quantitative_primary'] == 'True']
    result = {'binding_n': len(known), 'quantitative_n': len(quant), 'binding': {}, 'high_affinity': {}}
    binding = np.array([r['binding'] == 'true' for r in known])
    for k in (10, 20, 50):
        weights = cutoff_weights([scores[r['sample_id']] for r in known], k)
        hits = float(weights @ binding)
        result['binding'][k] = dict(hits=hits, precision=hits/k,
                                   enrichment=(hits/k)/binding.mean(), recall=hits/binding.sum())
    for threshold in (10, 100, 1000):
        strong = np.array([r['quantitative_primary'] == 'True' and float(r['kd_molar']) <= threshold * 1e-9 for r in known])
        unresolved = np.array([r['binding'] == 'true' and r['quantitative_primary'] != 'True' for r in known])
        entry = dict(confirmed_strong=int(strong.sum()), unresolved_affinity=int(unresolved.sum()), screened_pool={}, measured_only={})
        for k in (10, 20, 50):
            w = cutoff_weights([scores[r['sample_id']] for r in known], k)
            hits, unknown = float(w @ strong), float(w @ unresolved)
            entry['screened_pool'][k] = dict(confirmed_hits=hits, unresolved_selected=unknown,
                confirmed_yield=hits/k, possible_yield_upper=(hits+unknown)/k,
                confirmed_hit_recall=hits/strong.sum())
        y = np.array([float(r['kd_molar']) <= threshold * 1e-9 for r in quant])
        for k in (5, 10, 20):
            w = cutoff_weights([scores[r['sample_id']] for r in quant], k)
            hits = float(w @ y)
            entry['measured_only'][k] = dict(hits=hits, precision=hits/k, enrichment=(hits/k)/y.mean(), recall=hits/y.sum())
        result['high_affinity'][threshold] = entry
    return result


def main():
    with (OUT/'predictions.csv').open() as f:
        predictions = list(csv.DictReader(f))
    with (ROOT/'data/processed/adaptyv_egfr_v1/audited_records.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['round'] == '2' and r['control'] == 'False']
    results = {}
    for name in sorted({r['model'] for r in predictions}):
        scores = {r['sample_id']: float(r['predicted_pkd']) for r in predictions if r['model'] == name and r['round'] == '2'}
        assert len(scores) == 400
        results[name] = evaluate(rows, scores)
    for name, sign in [('iptm', 1), ('plddt', 1), ('pae_interaction', -1), ('esm_pll', 1)]:
        assert all(r[name] for r in rows)
        results[name] = evaluate(rows, {r['sample_id']: sign * float(r[name]) for r in rows})
    output = dict(note='Exploratory after prior EGFR inspection. Thresholds 10/100/1000 nM are illustrations, not prespecified success criteria. No retraining, calibration or seed selection. Ties receive fractional credit. Screened pool is 378 known-status candidates; 22 unknown/unexpressed excluded. Nine unresolved 10uM affinity entries are not called high-affinity negatives: report selected unresolved count and a possible upper yield separately. Measured-only pool is 44 selected binders, not prospective screening. Nonbinding status is operational assay failure, not a numeric KD.', results=results)
    (OUT/'topk_exploratory.json').write_text(json.dumps(output, indent=2)+'\n')
    for name, r in results.items():
        if name.endswith('/best') or '/' not in name:
            print(name, 'binding hits@20/50', r['binding'][20]['hits'], r['binding'][50]['hits'],
                  '100nM screened@20', r['high_affinity'][100]['screened_pool'][20],
                  '100nM measured@10', r['high_affinity'][100]['measured_only'][10])


if __name__ == '__main__':
    main()
