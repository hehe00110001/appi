# PPIformer 维度与 ddG 接入核查

本记录响应用户对 128 维输出及“注意力接入 ddG、取消固定 0.3 强度”的质疑。它记录已核实的实现和用户重申的方向，不把旧方案附件中的建议自动当成执行指令。

## 实际 checkpoint 与缓存

- 权重：`external/PPIformer/weights/masked_modeling.ckpt`，使用结构掩码预训练 checkpoint，并非直接使用 `ddg_regression` 微调模型作为教师。
- 读取实际 checkpoint：`embedding_dim=128`，`input_dim=[20,1]`，encoder `dim=(128,64)`、`num_degrees=2`；encoder 参数量 12,330,786。
- 128 是每残基的 type-0 标量通道数。界面有 N 个残基，输出为 N×128，不是整个界面压成 128 个数。
- Equiformer 内部同时使用 type-1 向量通道；当前官方 `PPIformer.forward` 取 `out.type0[inputs['mask']]` 返回。我们保存该完整返回值，没有额外切片压缩成 128；也未将 20 类分类 logits 误当 embedding。
- 因此，当前缓存没有显式保存 type-1 输出或中间层。type-0 已经过几何编码，不能把没有显式向量缓存解释为没有结构信息。未来若使用向量通道，需要保持旋转等变/不变关系，不能直接展平到普通 MLP。
- PPIformer 是粗粒度界面残基结构掩码模型，不等于 MSMS/APBS/dMaSIF 意义上的显式分子表面编码器。当前输入不提供完整表面网格、电势场和溶剂信息。

依据：本地 `ppiformer/model/ppiformer.py`、实际 checkpoint、`scripts/build_ppb_features.py`，以及[官方仓库](https://github.com/anton-bushuiev/PPIformer)和[ICLR 2024 论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/5b4263be85820683d78675cc18d2efc7-Paper-Conference.pdf)。

## 当前监督没有实现用户重申的机制

当前 ESM pilot 的 R 目标仍为：

`L_abs + 0.3 * L_diff + 0.1 * L_rank`

已有 attention 是残基自注意力和伙伴内注意力池化。差分损失通过共享打分器回传到这些模块，但没有专门的 WT/mutant 差分 attention、ddG 任务表征通道或取代固定损失系数的注意力机制。

历史文件 `docs/残基交互模型训练-20261026.md` 明确记录了“升级 attention 读出，同时保留 0.3”。它能解释代码配置的来源，不能用来证明用户所说的机制已经落实。用户本轮重申的要求优先于该历史实现说明。

注意力权重负责表示聚合和信息流；任务损失权重负责优化目标的相对贡献，二者不自动等价。不能只新增 attention，然后继续保留固定 0.3，却宣称已经完成取消固定强度的要求；也不能把无约束可学习标量乘以非负损失、任其趋零，当作有效替代。

术语上，SKEMPI 实验 ΔΔG/ΔpKd 属于监督信号；结构掩码预测属于自监督。若有教师预测或自生成突变一致性目标，应另外明确其数据来源与定义，不能与实验差分回归混称。

## 对既有结论的限制与后续约束

- 本轮结果仅检验：冻结 type-0 特征、界面节点提前相加融合、现有 B/R 损失、绝对验证 RMSE 选模。
- 不据此声称 PPIformer 完整几何表征无效，也不据此否定尚未实现的 ddG 注意力迁移方案。
- 不能用 128 对比 1280 就判定信息量相差十倍；单纯将 128 升维不会创造额外预训练信息。
- 后续设计必须明确 ddG 表征从哪里来、attention 的 Q/K/V 对应什么、梯度更新哪些模块、预测时是否需要 WT 和突变标签，以及如何取消固定 0.3 的优化依赖。推理不能读取实验 ddG 标签。
- 在当前可读取记录中未找到足以还原上述注意力方案全部模块定义的证据；不把临时提出的新结构冒称为此前已经约定的实现。

本次只完成核查与记录，未修改已完成实验的训练代码/缓存，也未提交新训练。
