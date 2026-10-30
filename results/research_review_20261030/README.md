# Research snapshot — 2026-10-30

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
