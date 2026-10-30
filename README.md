# APPI：共享 PPI 亲和力预测

使用冻结 PPIformer 的残基特征训练共享预测头，比较绝对 pKd 监督与绝对＋突变差分监督。
当前代码和结果更新于 **2026-10-30**。最新模型加入全链 ESM-2 上下文路由和结构保留通道，使用共享 pKd 评分器接受绝对及实验差分监督。

## 最新结论与结果归档

结构主导模型的4组×3seed训练和EGFR外部推理已完成。真实差分融合相对无差分的内部差分MAE在三个seed改善约14–17%，但相对置换标签与纯结构的优势不稳定。EGFR Round2三seed AUROC为0.486/0.575/0.672，定量Pearson为0.110/-0.011/0.231，尚未证明可靠的binder绝对亲和力预测或高亲和力Top-k富集。

- [2026-10-30完整结果快照](results/research_review_20261030/README.md)：包含所有seed与对照、逐条预测、区间、Top-k及运行配置。
- [EGFR三seed分析](docs/EGFR三seed外部推理结果-20261030.md)与[内部三seed分析](docs/三seed复现与EGFR评估结果-20261030.md)。
- [seed44研究预发布模型卡](releases/v0.1.0-rc1-seed44/MODEL_CARD.md)、[推理说明](releases/v0.1.0-rc1-seed44/INFERENCE.md)、[手动推送与发布](releases/v0.1.0-rc1-seed44/PUBLISH.md)。seed44是在查看结果后指定的候选，不能将其表现当作无偏的三seed结论。
- [外部数据候选与Top-k评估](docs/外部数据候选与TopK评估-20261030.md)。

此前SKEMPI A/B/C单seed消融保留为历史阶段：差分test Pearson为-0.044/0.260/0.386，组内排序52.7%/56.1%/56.2%。

- [A/B/C 实验方案与运行记录](docs/SKEMPI三组消融实验-20261027.md)
- [A/B/C 完整结果解读](docs/SKEMPI三组消融结果-20261027.md)
- [已纳入 Git 的结果快照与文件索引](results/training_review_20261027/README.md)：评估、逐条预测、组内排序、方向偏差、曲线、训练配置/权重/历史与固定划分成员。

## 前序实验记录

- PPB-Affinity 原始 filtered 清单 8207 条，5263 条通过预检并成功编码。
- SKEMPI 原始 7085 条，经原始/cleaned 编号与实验元数据匹配得到 3603 条配对候选。
- 首轮数据（训练脚本默认）：绝对 train/val/test=4470/459/330；差分=3556/32/13。
- 新固定划分 `joint_training_resplit_v1`：绝对 4461/397/401；差分 3063/268/270。保留全部合格记录，按 PDB/共享特征分组并平衡来源；后续对照需显式指定此 `--data`。
- 首轮六个训练任务已完成。绝对测试 RMSE：仅绝对监督 1.644±0.013，联合监督 1.931±0.029 pKd。
- 差分测试 RMSE：仅绝对监督 0.984±0.020，联合监督 0.930±0.019 ΔpKd。
- 等权差分监督尚未改善整体模型；差分测试仅13条，不能作强结论。

详细交接见 [当前结果摘要](docs/当前结果摘要-20261026.md)。
新清单的规则、审计和复现见 [固定划分说明](docs/固定划分-20261026.md)。
新划分上的公平标签对照及启动命令见 [第二轮训练启动](docs/第二轮训练启动-20261026.md)；池化保持 mean+sum。
第二轮 12 个任务已正常完成，实际训练 30–32 轮；联合监督的差分测试 RMSE 改善不超过约 0.41%。详见 [第二轮结果与耗时核查](docs/第二轮训练结果-20261026.md)。
第三轮两层残基交互注意力与可学习池化的 6 个任务（15821）已全部完成。新架构 absolute 的绝对/差分测试 RMSE 为 2.126/1.841，joint（λ=0.3）为 2.047/1.909；架构升级降低了误差，附加差分监督收益仍不一致。详见 [第三轮结果核查](docs/残基交互模型结果-20261026.md)，配置见 [残基交互模型训练](docs/残基交互模型训练-20261026.md)。
自有 IL17A/IL23R 共 16 条外部 binder 的 6 模型评估已完成（15891），使用用户提供的 pKd 标签。总体 Pearson 的种子均值为 absolute=-0.208、joint=-0.364，外部表现尚不可靠。分靶点、界限样本敏感性和逐条预测见 [自有16条外部评估](docs/自有16条外部评估-20261026.md)。
已按数据来源作者的复合物拼接序列 MMseqs2 方法生成40%划分，全部合格记录保留，绝对4471/394/394，配对3061/269/271。首轮划分与新划分的当前模型单seed对照及汇总已完成（15912/15915），详见 [三种划分结果](docs/三种划分训练结果-20261026.md)。方法出处、对此前单链筛查说明的纠正和作业信息见 [40%划分对照](docs/同源划分方法核实与40百分比对照-20261026.md)。
原始施工文档是历史设计参考，具体实现与结果以日期较新的审计和运行文档为准。

## 核心文件

- appi/structure_context_model.py、structure_context_data.py、gradnorm.py：结构保留、全链ESM路由、共享评分与动态任务平衡。
- scripts/train_structure_context.py、evaluate_structure_context.py：当前模型训练与评估。
- scripts/prepare_adaptyv_egfr.py、extract_adaptyv_egfr.py、evaluate_adaptyv_egfr.py：EGFR准备、特征提取与冻结外部评估。
- scripts/export_seed44_release.py、predict_release.py：seed44研究模型包导出及兼容缓存推理。
- appi/model.py：对称池化、768→256→1 共享预测头和缓存特征归一化。
- appi/residue_model.py、residue_data.py：残基交互注意力、可学习池化及显存受限的小批次处理。
- appi/units.py：Kd/pKd/ΔpKd/ΔΔG 单位转换。
- scripts/build_ppb_features.py：界面截取、类型替换突变、冻结编码。
- scripts/build_skempi_pairs.py：SKEMPI 实验配对。
- scripts/audit_joint_splits.py、prepare_joint_training.py：划分审计与训练清单。
- scripts/resplit_joint_training.py：保留全部合格样本，生成固定的来源平衡分组划分。
- scripts/train_affinity.py、train_affinity.sbatch：训练与 Slurm 提交。
- scripts/prepare_residue_cache.py、train_residue_affinity.py、train_residue_affinity.sbatch：残基缓存整合、新模型训练与 GPU 提交。
- scripts/summarize_training.py、analyze_first_training.py：结果汇总与分析。
- tests/：数据处理、单位、模型与配对测试。

## 集群运行

Python 3.10，独立环境 appi-ppiformer。当前脚本包含集群绝对路径，换机器需要修改路径。
编码另需准备 external/PPIformer 源码及官方 masked_modeling.ckpt，原始数据另行下载。
Git包含整理后的实验结果、部分标签/预测和训练配置快照；不包含原始结构全集、编码器权重、缓存特征或训练checkpoint。seed44评分网络另以本地Release资产打包，须手动上传；当前不是任意PDB一键预测服务。

```bash
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python -m unittest discover -s tests -v
sbatch scripts/train_affinity.sbatch
python scripts/summarize_training.py
```

首轮配置与恢复方式见 [训练说明](docs/首轮训练-20261025.md)。
当前已完成绝对/差分监督、独立排序损失实验、结构与ESM消融、自有binder及EGFR冻结评估。最新结构主导模型采用GradNorm平衡绝对/差分回归，组内排序用于主checkpoint选择。绝对性能硬约束训练、物理标量、删失监督与CATH-ddG独立基准尚未接入。
