"""Plot external-test predictions; ensemble averages are fixed over all three seeds."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT/'outputs/binder16_residue_v1'
    with (out/'predictions.csv').open() as f:
        rows = list(csv.DictReader(f))
    metrics = json.loads((out/'metrics.json').read_text())
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True)
    colors = {'IL17A': '#2878B5', 'IL23R': '#E07A24'}
    limits = [min(float(r[k]) for r in rows for k in ['pkd', 'absolute_mean_pkd', 'joint_mean_pkd'])-.5,
              max(float(r[k]) for r in rows for k in ['pkd', 'absolute_mean_pkd', 'joint_mean_pkd'])+.5]
    for i, mode in enumerate(['absolute', 'joint']):
        for j, target in enumerate(['all', 'IL17A', 'IL23R']):
            ax = axes[i, j]
            selected = [r for r in rows if target == 'all' or r['target'] == target]
            for r in selected:
                bound = r['sample_id'] == '23r-91_il23r'
                ax.errorbar(float(r['pkd']), float(r[mode+'_mean_pkd']),
                            yerr=float(r[mode+'_seed_sd']), fmt='D' if bound else 'o',
                            color=colors[r['target']], mfc='white' if bound else colors[r['target']],
                            capsize=2, markersize=5, alpha=.85)
                if target != 'all':
                    ax.annotate(r['sample_id'].split('_')[0]+('*' if bound else ''),
                                (float(r['pkd']),float(r[mode+'_mean_pkd'])),
                                xytext=(4,4),textcoords='offset points',fontsize=7)
            m = metrics['ensemble'][mode]['user_labels_all16/'+target]
            ax.plot(limits, limits, '--', color='#888888', linewidth=1)
            ax.set_title(f"{mode} | {target} | n={m['n']}\nPearson={m['pearson']:.3f}, Spearman={m['spearman']:.3f}")
            ax.set_xlim(limits);ax.set_ylim(limits);ax.grid(alpha=.2)
            ax.set_xlabel('Experimental pKd (user-supplied)')
            if j == 0:ax.set_ylabel('Predicted pKd (three-seed mean)')
    fig.suptitle('External binder evaluation: frozen models, no fitting\nBars: seed SD; open diamond: 23r-91 entered as pKd=12 (legacy Kd<1 pM)',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.94])
    fig.savefig(out/'predicted_vs_experimental.png',dpi=180)
    fig.savefig(out/'predicted_vs_experimental.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
