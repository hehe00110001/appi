# APPI：共享 PPI 亲和力预测

使用冻结 PPIformer 的残基特征训练共享预测头，比较绝对 pKd 监督与绝对＋突变差分监督。
当前代码和运行状态更新于 2026-10-26；首轮结果来自 2026-10-25。

## 当前状态

- PPB-Affinity 原始 filtered 清单 8207 条，5263 条通过预检并成功编码。
- SKEMPI 原始 7085 条，经原始/cleaned 编号与实验元数据匹配得到 3603 条配对候选。
- 默认训练数据：绝对 train/val/test=4470/459/330；差分=3556/32/13。
- 首轮六个训练任务已完成。绝对测试 RMSE：仅绝对监督 1.644±0.013，联合监督 1.931±0.029 pKd。
- 差分测试 RMSE：仅绝对监督 0.984±0.020，联合监督 0.930±0.019 ΔpKd。
- 等权差分监督尚未改善整体模型；差分测试仅13条，不能作强结论。

详细交接见 [当前结果摘要](docs/当前结果摘要-20261026.md)。
原始施工文档是历史设计参考，具体实现与结果以日期较新的审计和运行文档为准。

## 核心文件

- appi/model.py：对称池化、768→256→1 共享预测头和缓存特征归一化。
- appi/units.py：Kd/pKd/ΔpKd/ΔΔG 单位转换。
- scripts/build_ppb_features.py：界面截取、类型替换突变、冻结编码。
- scripts/build_skempi_pairs.py：SKEMPI 实验配对。
- scripts/audit_joint_splits.py、prepare_joint_training.py：划分审计与训练清单。
- scripts/train_affinity.py、train_affinity.sbatch：训练与 Slurm 提交。
- scripts/summarize_training.py、analyze_first_training.py：结果汇总与分析。
- tests/：数据处理、单位、模型与配对测试。

## 集群运行

Python 3.10，独立环境 appi-ppiformer。当前脚本包含集群绝对路径，换机器需要修改路径。
编码另需准备 external/PPIformer 源码及官方 masked_modeling.ckpt，原始数据另行下载。
本仓库不包含数据、权重、缓存特征、训练模型或日志，不是开箱即跑的数据包。

```bash
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python -m unittest discover -s tests -v
sbatch scripts/train_affinity.sbatch
python scripts/summarize_training.py
```

首轮配置与恢复方式见 [训练说明](docs/首轮训练-20261025.md)。
当前仅实现绝对和差分监督；排序损失、物理标量、删失监督、CATH-ddG 独立基准及自有 binder 评估尚未接入。
