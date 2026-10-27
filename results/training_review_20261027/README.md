# 本阶段训练与评估结果快照（2026-10-27）

这是本地 `outputs/`、`runs/` 和划分清单中可审核结果的 Git 快照。原目录继续被 `.gitignore` 忽略；此目录保存小型结果与复现元数据。模型权重、PPIformer依赖、原始结构和残基特征缓存仍保留在原工作区，没有包含在本快照中。仅克隆仓库可查看结果；重新训练还需要原始数据、模型及特征环境。

## 推荐阅读顺序

1. [SKEMPI A/B/C 完整结果解读](../../docs/SKEMPI三组消融结果-20261027.md)
2. [A/B/C 自动汇总](evaluations/skempi_ablation_v1/report.md)和[完整指标 JSON](evaluations/skempi_ablation_v1/comparison.json)
3. [突变类型/方向分层 CSV](evaluations/skempi_ablation_v1/mutation_strata.csv)
4. [此前三种划分对照](evaluations/split_comparison_v1/report.md)
5. [此前组内排序诊断](evaluations/group_generalization_v1/report.md)

本轮A/B/C固定40%划分、seed42；所有指标来自已完成的模型，未根据测试结果重新选择checkpoint。A排除SKEMPI训练记录，B加入均衡绝对标签，C在B上加λ=0.3差分损失；三组均无额外端点损失。更详细的设计和限制见[实验方案](../../docs/SKEMPI三组消融实验-20261027.md)。

## 目录内容

| 目录 | 内容 |
|---|---|
| `evaluations/skempi_ablation_v1/` | A/B/C指标、组指标、内部/外部逐条预测、方向分层、曲线、实验计划 |
| `evaluations/group_generalization_v1/` | 三种划分的组内排序、1EMV固定几何来源核查、突变方向偏差 |
| `evaluations/split_comparison_v1/` | 首轮30%、新40%、PDB划分的seed42对照 |
| `evaluations/binder16_residue_v1/` | 原残基模型三个seed的外部16条评估与验证 |
| `evaluations/residue_training_analysis/` | 原残基模型的内部诊断和学习曲线 |
| `runs/skempi_ablation_v1/` | A/B/C配置、逐条训练权重、合并逐轮历史、指标、验证/测试预测 |
| `runs/affinity_split_comparison_v1/` | 首轮/40%划分两个监督模型的对应运行记录 |
| `runs/affinity_residue_v1/` | PDB划分残基模型两个监督×三个seed的运行记录 |
| `split_membership/` | 三种固定划分的样本ID/PDB/来源/组归属；可用时附分组说明与审计 |
| `external_labels/` | 用户提供的外部16条pKd标签及界限备注 |

[manifest.json](manifest.json)逐项记录快照的原路径、SHA-256和大小。`history.json`将原逐epoch JSON合并为数组，manifest同时记录每个原文件的哈希；划分成员表是原清单字段子集，manifest记录源表哈希。复制的JSON/CSV/图保持原内容，配置中的绝对路径反映实际运行环境。原报告中提及的 `outputs/...` 与 `runs/...` 路径可按本目录结构找到对应快照。

主要结论：移除SKEMPI未解决外部泛化；加入其绝对标签改善部分突变指标；额外差分损失的总体相关收益没有转化为稳定组内排序，而且真实增强突变的方向识别由B的52.2%降到C的39.1%。这是单seed、固定表示下的结果，不能断言SKEMPI或固定骨架表示普遍无效。
