"""Archive current research evidence as reviewable Git-sized files."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'results/research_review_20261030'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise FileExistsError('Snapshot already exists; do not overwrite frozen evidence')
    DEST.mkdir(parents=True)
    records = []

    def copy(source, target):
        target = DEST/target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/source, target)
        records.append(dict(source=str(source), path=str(target.relative_to(DEST)),
                            bytes=target.stat().st_size, sha256=digest(target)))

    for name in ('group_ranking_v1', 'additive_ranking_v1', 'esm2_pilot_v1',
                 'structure_context_pilot_v1', 'adaptyv_egfr_multiseed_a6000_v1'):
        for path in sorted((ROOT/'outputs'/name).iterdir()):
            if path.is_file() and path.suffix in ('.json', '.csv', '.md', '.png', '.pdf'):
                copy(path.relative_to(ROOT), Path('evaluations')/name/path.name)
    for run in sorted((ROOT/'runs/structure_context_pilot_v1').glob('*_seed4[234]')):
        for name in ('config.json', 'metrics.json', 'complete.json'):
            if (run/name).exists():
                copy((run/name).relative_to(ROOT), Path('runs')/run.name/name)
        for choice in ('best', 'best_absolute'):
            for path in sorted((run/f'evaluation_{choice}').iterdir()):
                if path.suffix in ('.csv', '.json'):
                    copy(path.relative_to(ROOT), Path('runs')/run.name/f'evaluation_{choice}'/path.name)
    for path in ('data/processed/adaptyv_egfr_v1/audited_records.csv',
                 'data/processed/adaptyv_egfr_features_v1/config.json',
                 'data/processed/adaptyv_egfr_features_v1/extraction_config.json',
                 'data/processed/adaptyv_egfr_features_v1/complete.json',
                 'outputs/adaptyv_egfr_seed43_44_a6000_v1/submission_19980.json'):
        copy(Path(path), Path('provenance')/Path(path).parent.name/Path(path).name)
    (DEST/'manifest.json').write_text(json.dumps(dict(
        note='File-identical snapshots; paths inside saved configurations refer to the original cluster. Checkpoints and feature tensors excluded. Earlier one-seed analyses are historical, not replacements for three-seed evidence.',
        files=records), indent=2)+'\n')
    (DEST/'README.md').write_text('''# Research snapshot — 2026-10-30

本目录将本轮结果复制入Git；每个文件与原输出逐字节一致，来源及SHA256见manifest.json。

## 当前主要证据

- [EGFR 三 seed 主/次checkpoint汇总](evaluations/adaptyv_egfr_multiseed_a6000_v1/multiseed_report.md)
- [EGFR 全部25模型逐项结果](evaluations/adaptyv_egfr_multiseed_a6000_v1/report.md)
- [EGFR 预测](evaluations/adaptyv_egfr_multiseed_a6000_v1/predictions.csv)、[置信区间](evaluations/adaptyv_egfr_multiseed_a6000_v1/cluster_intervals.json)、[探索性Top-k](evaluations/adaptyv_egfr_multiseed_a6000_v1/topk_exploratory.json)
- [内部三seed复现](evaluations/structure_context_pilot_v1/multiseed_reproducibility.json)
- [内部完整比较](evaluations/structure_context_pilot_v1/comparison.json)
- [seed44训练配置](runs/fusion_transfer_seed44/config.json)及同目录下两套checkpoint逐条预测

## 历史阶段

group_ranking_v1、additive_ranking_v1、esm2_pilot_v1保留此前排序、可加读出和ESM消融。structure_context_pilot_v1内detailed_analysis/result_diagnostics等早期文件为seed42阶段历史分析；当前三seed主证据以comparison.json与multiseed_reproducibility.json为准。旧图replication_egfr_20261030的EGFR面板明确为seed42，不是三seed图。

快照不含编码器、特征张量或训练checkpoint。配置中的绝对路径仅供溯源；第三方实验数据的权利仍归其原始来源。公开EGFR原始仓库：https://github.com/adaptyvbio/egfr_competition_1 和 https://github.com/adaptyvbio/egfr_competition_2 。

发布候选与完整边界见[模型卡](../../releases/v0.1.0-rc1-seed44/MODEL_CARD.md)。seed44在查看外部结果后指定，不能以单seed替代完整对照。
''')
    print(json.dumps(dict(files=len(records), bytes=sum(r['bytes'] for r in records), directory=str(DEST))))


if __name__ == '__main__':
    main()
