# 实施交接文档：基于 PPIformer 的 PPI 亲和力模型

**日期**：2026-10-22
**用途**：交接给服务器端编码助手。本文是**可直接实施的技术规格**，不是研究讨论。
**前置背景**（如需理解动机再看，实施不依赖）：
`讨论记录-蛋白质亲和力预测的瓶颈与突破方向.md`、`论文主线-靶点内排序方案.md`

> **数字标注**：**【已核实】** 来自论文/代码/本项目实验记录；**【推断】** 由已核实数字推导；**【待核实】** 需实施前自查。

---

## 0. 一页总览

### 要做什么

训练一个**共享打分函数** `S_θ(A,B)`，用三种监督同时约束：

```
L = L_abs  +  λ₁·L_diff  +  λ₂·L_rank
     ↑            ↑              ↑
  绝对 pKd     SKEMPI ΔpKd    同靶点内排序
```

编码器用 **PPIformer 预训练权重**（冻结起步），**不自己做预训练**，**不自己设计网络结构**。唯一要写的是对称池化、一个两层 MLP 头、三项损失。

### 为什么这样做（三句话）

1. **跨体系绝对 ΔG 已接近饱和**：推断噪声天花板 ~0.71 **【推断】**，本项目纯序列基线已达 0.703。不在这个指标上较劲。
2. **ΔΔG 侧有空间但很挤**：已发表天花板 sup(ρ)=0.89 **【已核实, Tsishyn 2024】**，严格划分下 SOTA 仅 0.499 **【已核实, CATH-ddG】**。但 GearBind / PPIformer / CATH-ddG / StaB-ddG / USP-ddG 全部只做 ΔΔG。
3. **未被占的位置只有一个**：把相对监督（ΔΔG）的收益**传递到可用于 de novo binder 筛选的打分函数**。所有 ΔΔG 方法不做绝对值，所有绝对值方法不用差分监督，两边没有交集。

### 已被否决的方向（不要重新提议）

| 方向 | 理由 |
|---|---|
| MSMS 三角化表面 + APBS PB 静电 + patch 图 | 本项目 30 个受控版本（含置换对照）一致 null；surface-only 回归仅 0.3491 **【已核实】**；且 APBS 分钟级太慢、对预测结构的侧链误差敏感 |
| 自己做界面掩码预训练 | PPIformer 已做（ICLR 2024，权重公开），自训终点是追平它 |
| 自建严格同源划分 | CATH-ddG 划分已公开且被 USP-ddG 沿用 |
| 调层数 / 激活函数 / 网络类型 | 本项目前 30 个版本做过，收益为零 |
| AF2 生成突变体结构 | 见 §5.2 |
| OOF 叠加融合 | 见 §6.3 |

---

## 1. 最终数据清单

| # | 数据 | 角色 | 获取 |
|---|---|---|---|
| 1 | **Alsamkary 清洗版 PPB-Affinity** | 绝对 pKd 训练池 + 30% 划分 + PLM 基线 | HF `proteinea/ppb_affinity`（**processed** config）｜代码 `github.com/Proteinea/ppiseq`｜论文 arXiv:2505.20036 |
| 2 | 同上 **raw** config | σ_noise 分析（可选扩展） | 同上 |
| 3 | **SKEMPI 2.0 原始版** | ΔΔG 差分监督 + 元数据 + 删失标签 | `life.bsc.es/pid/skempi2` |
| 4 | **CATH-ddG 划分文件** | ΔΔG 侧的严格划分（直接用，不自建） | `github.com/ak422/CATH-ddG` |
| 5 | **Adaptyv R1 + R2** | de novo 外部验证 | `github.com/adaptyvbio/egfr_competition_1` / `_2` |
| 6 | **PPIformer 权重** | 结构编码器 + 零样本 ΔΔG 基线 | `download_from_zenodo('weights.zip')` |
| 7 | 本项目自有 16 条 binder | 独立湿实验验证（n=16） | 本地 |

### 1.1 为什么 1 不能替代 3 **【已核实】**

Alsamkary 版（和 PPB-Affinity 本身）把 SKEMPI 摊平成绝对亲和力记录，丢失了：

- **WT ↔ 突变体的显式配对**（`L_diff` 必需）
- **ΔΔG 本身**
- 温度 / pH / 实验方法 / 文献来源
- ΔH / ΔS / kon / koff（部分条目）
- **约 440 条"突变后无法检测到结合"** **【待核实确切条数】** ← 删失标签，`L_diff` 的不等式约束要用
- 正反向突变关系

### 1.2 Alsamkary 清洗的已知细节 **【已核实】**

从 12,062 条 → **8,207 条唯一 PPI**：
- 从 SEQRES 补回缺失的非末端残基（4,946 条链）
- **删除任一链 < 40 残基的条目（2,753 条）** ← 见 §7.2，这条有重要后果
- **按已知测量方法取平均合并重复条目（移除 1,088 条）** ← 这破坏了 σ_noise 分析所需的分歧数据，故需 raw config
- train/val/test 按 **≤30% 序列同一性**划分

### 1.3 去重顺序（不可颠倒）

```
第 1 步  冻结测试集：
         Adaptyv(R1+R2) + 自有 16 条 + CATH-ddG holdout
         + 历史对比用 test1(S79) / test2(S82)

第 2 步  从训练池剔除：先按 PDB ID，再按双边序列同一性（30%，两条链都卡）

第 3 步  绝对池 = Alsamkary 8,207 − 第 1 步全部
第 4 步  差分池 = SKEMPI 2.0 − CATH-ddG holdout − 第 1 步全部
```

### 1.4 ⚠️ 三个必须实施前查清的重叠

1. **Adaptyv 打的是 EGFR。** 训练池里很可能有 EGFR 复合物（EGFR-Cetuximab、EGFR-EGF 等）。**不剔除则 Adaptyv 验证完全失效。** 自有 16 条的靶点同理。
2. **test1(S79) 来自 Affinity Benchmark v5.5，而 PPB-Affinity 包含 Affinity Benchmark v5.5** → 历史测试集就在训练池内。
3. **SKEMPI 同时出现在 1 和 3 里。** 建议 precedence：**配对信息以 SKEMPI 原始为准，绝对值以 Alsamkary 版为准，冲突时记录分歧**（这批分歧就是 σ_noise 分析的材料）。

### 1.5 不要收集（第一版）

ddPCA、Starr RBD DMS、Megascale、DIPS-Plus、PROXiMATE、AB-Bind、ATLAS 单独版。

- **ATLAS 不单独当测试集**：它是 PPB-Affinity 五个来源之一，已在训练池内，当测试集是泄漏。TCR-pMHC 只做测试集内的**分层报告**。
- ATLAS 本身也不适合当训练源：仅 694 条、112 个实验晶体结构，**>60% 的亲和力挤在 2 kcal/mol 区间内** **【已核实】**（量程压缩 → Pearson 天生难看）。
- DIPS-Plus 若将来需要预训练语料，用 PPIRef 替代（`download_from_zenodo('ppi_6A.zip')` / `ppi_10A.zip`，与 PPIformer 同源，可比性更好）。

---

## 2. 单位与符号约定（全局，不得混用）

```
pKd   = −log10(Kd / M)
ΔpKd  = pKd_mut − pKd_WT              正值 = 亲和力增强
ΔΔG   = ΔG_mut − ΔG_WT = −RT·ln(10)·ΔpKd

T = 298.15 K:   ΔΔG [kcal/mol] = −1.364 × ΔpKd
```

**注意：标签换算带负号，MAE 幅度换算用正系数 1.364。**

**绝不把不同单位的 MAE 放进同一列。** 所有结果双单位报告，并写明换算常数与温度假设。

> **【待核实】** 本项目历史 MAE 的单位。仓库 `SGNet/aligned_io.py` 的 `pk_from_molar()` 返回 `-math.log10(value)`，README 定义目标为 pK，故推定为 log10。但 V28 历史标签路径需单独追溯。
> 若为 log10：V28 test1 MAE 1.1805 → **1.610 kcal/mol**（竞品 1.36–1.52，本项目垫底）。
> 若为 ln（系数 0.592）：→ 0.699，全场最好。**整个竞争态势判断取决于这一个常数。**

---

## 3. 架构数据流

```
①  原始 PDB（晶体 或 AF2/Boltz 预测）
         ↓   PPIExtractor(kind='heavy', radius=10., bsa=True)
②  界面 PDB 文件  xxxx_A_B.pdb   +   BSA 标量
         ↓   PPIformer  masked_modeling.ckpt        ← 冻结起步
③  H ∈ R^{N×128}  （N = 界面残基数，type-0 不变特征）
         +  chain_mask（残基归属 A / B）
         ↓   对称池化
④  z = [h_A+h_B , |h_A−h_B| , h_A⊙h_B , BSA, 接触计数, 链长, ...]
         ↓   两层 MLP 头                            ← 唯一需要设计的部分
⑤  S_θ(A,B)  →  标量（pKd 尺度）
         ↓
⑥  L = L_abs + λ₁·L_diff + λ₂·L_rank
```

### 3.1 代码量分工

| 级 | 组件 | 实施工作量 |
|---|---|---|
| ① | `ppiref.extraction.PPIExtractor` | 库调用 |
| ② | 文件管理 | 几十行（注意 §7.1 命名坑） |
| ③ | `ppiformer.utils.api.embed` | **库调用，0 行** |
| ④ | 对称池化 | 复用本项目 V23 的"和 / 绝对差 / 乘积"构造 |
| ⑤ | MLP 头 | ~50 行 |
| ⑥ | 三项损失 | ~80 行 |

**总计新增代码约 200–300 行，其余为库调用与胶水。**

### 3.2 现成 API（已核实）

```python
# 界面提取
from ppiref.extraction import PPIExtractor
extractor = PPIExtractor(out_dir=ppi_dir, kind='heavy', radius=10., bsa=True)
extractor.extract(pdb_file, partners=['A', 'C'])   # partners=None 则提取全部二聚界面

# 编码
import torch
from ppiformer.tasks.node import PPIformer, DDGPPIformer
from ppiformer.utils.api import download_from_zenodo, predict_ddg, embed
from ppiformer.definitions import PPIFORMER_WEIGHTS_DIR

download_from_zenodo('weights.zip')
model = PPIformer.load_from_checkpoint(
    PPIFORMER_WEIGHTS_DIR / 'masked_modeling.ckpt',
    map_location=torch.device('cpu')).to(device).eval()

H = embed(model, ppi_path)      # torch.Size([N, 128])

# 零样本 ΔΔG 基线（SKEMPI 微调过的 3 模型 ensemble）
models = [DDGPPIformer.load_from_checkpoint(
    PPIFORMER_WEIGHTS_DIR / f'ddg_regression/{i}.ckpt',
    map_location=torch.device('cpu')).eval().to(device) for i in range(3)]
ddg = predict_ddg(models, ppi_path, ['SC16A', 'FC47A', 'SC16A,FC47A'])
```

`ppi_path` 接受**任意 PDB 文件**，无"必须是晶体结构"的要求 → **AF2/Boltz 预测结构可直接输入**。
PPIformer 为 **coarse-grained** 表征，对侧链精度不敏感 → 在预测结构上比 MSMS 类特征鲁棒得多。

### 3.3 ④ 对称池化的两个要点

1. **必须交换不变**：`S(A,B) = S(B,A)`。用 `[h_A+h_B, |h_A−h_B|, h_A⊙h_B]`（本项目 V23 已实现）。
2. **必须同时保留 mean 和 sum**。mean 是内禀量；**sum 保留"界面有多大"这个外延信息**——这正是 PRODIGY 靠接触计数达到 ~0.73 的那部分信息 **【待核实该数值】**。只用 mean pooling 会把它洗掉。

显式标量通道（廉价，毫秒级，不要省）：
- BSA / ΔSASA（`PPIExtractor(bsa=True)` 或 FreeSASA 的 Shrake-Rupley）
- 分型接触计数（极性/非极性/带电）
- 氢键、盐桥（简单几何判据，**不需要三角化网格**）
- 两条链长度、链数、NIS 组成

---

## 4. 损失函数

```python
# 绝对项（Alsamkary 8,207）
s = S(wt_interface)
L_abs = huber(s, pkd_true)

# 差分项（SKEMPI）—— 同一个 S，调用两次
s_wt  = S(wt_interface)
s_mut = S(mut_interface)
L_diff = huber(s_mut - s_wt, delta_pkd_true)

# 同靶点内排序（de novo 筛选 / 同体系突变）
L_rank = margin_ranking(S(design_i), S(design_j), sign)

L = L_abs + lambda1 * L_diff + lambda2 * L_rank
```

### 4.1 结构性优势（不需额外约束项）

- **权重共享自动成立**：`L_diff` 里调用的是同一个 `S_θ`。
- **反对称自动成立**：`s_mut − s_wt = −(s_wt − s_mut)` 是恒等式，不需要加惩罚项。
- **绝对尺度由 `L_abs` 锚定**：只学差值无法确定每个家族的绝对基准（整族加常数差值不变），所以两项必须联合。

### 4.2 删失标签处理

SKEMPI 中"突变后无法检测到结合"的条目**不要赋随意数值**。按检测阈值用不等式约束：

```python
# 若 mut 为 non-detectable，真实 pKd_mut ≤ threshold
L_censored = relu(s_mut - pkd_threshold) ** 2
```

### 4.3 采样与加权

- **per-complex 重加权**：SKEMPI 中少数体系有大量突变，不加权会被它们支配。
- **反向突变增广**：`(mut→WT, −ΔpKd)` 作为训练样本加入。
- **WT + 全部突变体 + 重复测量 + 正反向突变必须同折。**

### 4.4 λ 的处理

λ₁、λ₂ 和学习率是**本轮唯一允许调的超参**。在内层验证集上选，预算要有限且一致（见 §8.3）。

---

## 5. 关键工程难点

### 5.1 突变体表征怎么拿（唯一的未知项）

没有突变体结构，要算 `S(mut_interface)`，需让 PPIformer 看到"几何不变、该位置残基类型改变"。

**先做这个 10 分钟测试**（它决定整条 ΔΔG 支路的实现路径）：

```python
# 直接改 PDB 里的残基名，保留骨架
H_wt  = embed(model, "complex.pdb")
H_mut = embed(model, "complex_A123V.pdb")   # 仅把 ALA123 的残基名改为 VAL
print(torch.allclose(H_wt, H_mut))          # 期望 False
```

- **输出变了** → 一直用现成 `embed()` API，不碰 PPIformer 内部。**最省事路径。**（PPIformer 是粗粒度表征，侧链原子大概率不参与编码，故此路大概率可行。）
- **输出没变** → 残基类型从别处读取，需往下挖一层。PPIformer 的图表示是 `(G, X, E, F₀, F₁)`，残基类型在 `F₀`，需在构图步骤直接替换。**这一层的具体 API 未核实。**

### 5.2 不要用 AF2 生成突变体结构

1. **AF2 对点突变几乎不敏感**：MSA 驱动，单点突变对 MSA profile 扰动极小，返回的基本是野生型结构。付全额算力拿近乎相同的输出。
2. **引入处理管线差异**：WT 用晶体、mut 用 AF2 → 模型可识别"这来自 AF2"而非识别物理差异（结构生成偏差）。

**若后续确需显式突变体几何**：用 FoldX `BuildModel` / Rosetta **只做侧链重排**，且**野生型也走同一套重建流程**（即使有晶体结构也重建一遍侧链），保证两侧管线一致。

---

## 6. 超参与训练策略

### 6.1 头部结构（刻意保守，不要调）

```python
head = nn.Sequential(
    nn.LayerNorm(d_in),              # d_in = 128*3 + n_scalars
    nn.Linear(d_in, 256), nn.GELU(), nn.Dropout(0.2),
    nn.Linear(256, 1),
)
```

两层，约 20 万参数。理由：8,207 样本 + 冻结 128 维编码器，容量给大即过拟合。
（参照：本项目 V30 可训练参数 1,087,708，在此设定下已偏多。）

**不要调层数、激活函数、网络类型。** 本项目前 30 个版本已验证此类改动收益为零。

### 6.2 编码器冻结策略（三档，按顺序）

| 档 | 做法 | 可训练参数 | 何时用 |
|---|---|---|---|
| 1 | 全冻结，只训头 | ~20 万 | **先做这个**，几小时出结果 |
| 2 | **AdaptFormer 式轻量 adapter** | ~百万 | CATH-ddG 用的方案，"不更新原预训练参数即获得迁移性"，已在本任务验证 **【已核实】** |
| 3 | 解冻最后 k 层 | 数百万 | 仅当 1、2 都显示欠拟合 |

### 6.3 若加入 ESM2

拼接在 ④ 级（concat 进 `z`），**且必须用模态 dropout（p=0.3–0.5）随机把 ESM2 段整体置零**。

**不要用 OOF 叠加。** 本项目 V25/V27/V29 冻结序列分支 + 零初始化残差的设计导致：残差目标里主要是标签噪声，结构分支拿不到有效梯度，表面权重≈0。模态 dropout 强迫结构通道独立可预测。

### 6.4 ⚠️ de novo 场景的 ESM2 confound

Adaptyv 表达率从 73% → 95%，失败的 87% 同时包含**不表达**和**不结合**两种。ESM2 的伪似然与"序列能否表达/是否稳定"高度相关。

→ **必须分层报告**：只在"已成功表达"的子集上再算一次 binder/non-binder 指标。否则你的 AUROC 可能来自表达预测而非亲和力预测。

---

## 7. 已知的坑（逐条检查）

### 7.1 界面文件命名

`abcd.pdb` 的 A-B 界面输出为 `abcd_A_B.pdb`，且**输入文件名中的下划线会被替换为短横线**。PPIformer 的 HF demo 有专门一行做转换：

```python
new_pdb_path = temp_dir / f"pdb/{pdb_path.name.replace('_', '-')}"
```

Adaptyv 的文件名大概率带下划线，提前处理。

### 7.2 Alsamkary 的 "<40 残基删除" 可能砍掉整个类别

该规则移除了 **2,753 条**。短链恰好是：

- **pMHC 的多肽仅 8–11 残基** → TCR-pMHC 复合物可能被整类删除
- PDZ / SH3 的配体也是短肽 → 蛋白-肽相互作用可能消失

**后果 **【待核实】****：
- 若确认如此，TCR-pMHC 分层报告在该数据集上作废（省事，但要知道）
- **更要紧**：若本项目自有 binder 是**微蛋白**且 < 40 残基，则该清洗规则**正好排除了应用场景**——训练分布无短链结合体，而预测目标恰为短链。**实施前必须确认自有 16 条的长度。**
- Adaptyv 无此问题（要求单链 scFv 或 nanobody，远超 40 残基）

### 7.3 30% 同一性是单边还是双边 **【待核实】**

PPI 必须**两条链都卡**才算隔离。只卡一条是最常见的隐蔽泄漏。
（本项目 V24 做对了：要求"两方都能匹配同一训练复合物才连边"。）
若 Alsamkary 是单边，需在其基础上再收紧一次。

### 7.4 划分性质必须准确命名

本项目 V24 的划分是**成对同源隔离**，它**不保证测试集中每一方都没在训练里出现过**。论文中不得笼统称为"去同源划分"。

需要两层：
- 主泛化评估：成对同源隔离（规模可用）
- 严格子集：测试双方**都**与训练蛋白低同源（30%）。为此移除的桥接样本须**公开移除规则与保留数量**。

### 7.5 Ig 超家族主导

CATH-ddG 指出超家族图中连接度最高的是 **2.60.40.10（immunoglobulin-like）** **【已核实】**。抗体与 TCR 均属此类。

→ **按体系类别分层评估是强制的**（Ab-Ag / TCR-pMHC / 酶-抑制剂 / 其他），否则 Ig 主导会掩盖失败。

界面建模对 Ab/TCR 是**优势而非劣势**：Ig 框架高度保守、仅 CDR 变化，全蛋白输入会让模型读到"这是抗体"这个家族身份信号（检索捷径），界面建模强迫其关注 CDR。

### 7.6 界面半径当超参

`PPIExtractor(radius=...)`；PPIRef 现成提供 6Å 与 10Å 两套，两个都试成本很低。
另需保留一个小的全蛋白标量通道（总长度、NIS 组成、链数），不要完全丢弃 NIS 效应。

---

## 8. 评估协议

### 8.1 三层评估

| 层次 | 数据与划分 | 作用 |
|---|---|---|
| **历史可比性** | ProAffinity 的 test1(S79) / test2(S82) | 与既有论文比较。**已被反复查看，属回顾性证据，不得称为完全盲测** |
| **主泛化评估** | Alsamkary 的 30% 划分（绝对侧）+ CATH-ddG holdout（ΔΔG 侧） | **核心证据** |
| **de novo 外部验证** | Adaptyv R1+R2 + 自有 16 条 | 应用验证 |

### 8.2 指标

**绝对亲和力任务**
- 主：**pKd MAE**
- 次：RMSE、Spearman、PCC、**校准误差**
- **强制报告每个测试集的 σ(ΔG)** ← 见下

> **量程压缩陷阱 **【已核实数据】****：V28 的 test1 MAE 1.1805 / PCC 0.7032，test2 MAE 1.1852 / PCC 0.4450。**MAE 几乎相同（差 0.0047），PCC 差 0.26。** 这不是泛化差异，是 test2 的 σ(ΔG) 更小。→ 以 MAE/RMSE 为主指标，PCC 为辅。两个测试集**分别报告**，合并结果只作补充。

**ΔΔG 任务**
- ΔpKd / ΔΔG 的 MAE
- **每个体系内部的 Spearman**，再对体系汇总（提前规定哪些体系样本量足够）
- 方向的 balanced accuracy / MCC；接近零、处于测量不确定范围内的变化单独处理
- **分层报告**：增强 / 削弱 / 丙氨酸 / 非丙氨酸突变
- **不得把所有体系的突变混在一起算一个 PCC**

> **突变偏差 **【已核实, Tsishyn 2024】****：SKEMPI 非均匀覆盖，丙氨酸扫描与削弱型突变较多。除 SSIPe、BindProfX 外所有预测器在**反向突变**上显著退化；**纯 ML 方法比物理方法退化更严重**。不分层报告会过于乐观。

**de novo 筛选任务（最贴近应用）**
- **binder / non-binder 的 AUROC + AUPRC** ← 一阶问题，87% 的设计根本不结合
- **top-10 / top-20 / top-50 命中率**
- **enrichment**
- **给定实验预算的累计发现曲线**
- 在有可靠 Kd 的子集上报 pKd MAE 与 Spearman

> 实验室真正关心的是："**如果只能合成 20 条序列，模型能不能让这 20 条里出现更多真正的 binder？**" Spearman 对整个候选列表等权，top-k 才对应真实决策。

### 8.3 选择纪律

- 外层分组五折，外层每折内部再留出**分组验证集**
- 内层验证集选 λ₁、λ₂、学习率、训练轮数、checkpoint
- **外层测试折只生成最终预测，不参与任何选择**
- 标准化、标签权重、校准**只从相应训练部分获得**
- 核心模型用相同划分 + **至少 3 个随机种子**，保存逐样本预测
- 不必做完整内层五折；每个外层训练集内一个固定分组验证集 + 有限且一致的调参预算即可

> **外部结果出来后继续改模型 = 新一轮开发，不得仍称同一次盲测。**

### 8.4 统计

- 模型间的 MAE 差与 PCC 差做**配对 bootstrap**，**重采样单位为体系或同源簇**
- **不得把同一复合物的 100 个突变当成 100 个独立体系**；不得把训练种子数当成新增测试样本数
- 报告效应量 + 95% CI + 各种子波动，不只报最好结果
- PCC +0.02 / MAE −0.03 可用于内部筛选，**不是论文合格线**

---

## 9. 实验矩阵

### 9.1 第 0 步：零样本与廉价基线（优先，几乎不占 GPU）

| 实验 | 内容 |
|---|---|
| 0.1 | §5.1 的残基名测试（10 分钟）← **先做这个** |
| 0.2 | `embed` + 线性/小头，在自有 16 条上跑排序，对比既有 surface 模型的 r=0.2 **【已核实】** |
| 0.3 | `predict_ddg` 零样本基线（PPIformer 现成的 3 模型 ensemble） |
| 0.4 | 界面"天然度"零样本打分：masked modeling 的 `Σ log p(实际残基 \| 上下文)` 作为 de novo 排序分 **【API 是否暴露 per-residue logits 未核实】** |
| 0.5 | kNN 检索基线（双边同一性取最近训练邻居的 pKd）、PRODIGY、训练均值、随机 |
| 0.6 | σ(ΔG) 实测、`bias = mean(pred − true)` 偏移诊断、单位核算 |

> **0.4 的边界**：天然度 ≠ 亲和力（天然蛋白并不以最大亲和力为优化目标）。作为零样本基线成本为零，但不能当作结论。

### 9.2 第 1 步：主对照

| 实验 | 结构预训练 | PDBbind/Alsamkary 绝对 | SKEMPI 绝对 | SKEMPI 差值 |
|---|---|---|---|---|
| **A** | − | ✓ | − | − |
| **A′** | − | ✓ | ✓ | − |
| **C** | − | ✓ | ✓ | ✓ |
| **B** | ✓ | ✓ | − | − |
| **B′** | ✓ | ✓ | ✓ | − |
| **D** | ✓ | ✓ | ✓ | ✓ |

- `A′−A`、`B′−B`：**增加亲和力记录**的作用
- `C−A′`、`D−B′`：**相同可用标签下，差值约束的额外作用** ← 关键
- `B−A`、`B′−A′`、`D−C`：预训练在不同监督条件下的作用

**首轮只跑 A、A′、C**，确认差值监督的信号后再扩展。

**关键对照**：独立 ΔΔG 头（验证收益是否真的传到了共享打分函数本身，而非只在 ΔΔG 上好看）。

**公平性**：核心各组架构一致、训练预算相近、数据采样规则相同。**不得让"无预训练"用小网络而"有预训练"同时换大网络。** 额外预训练成本单独记录。

> **注意 **【已核实】****：差值从绝对标签算出时**并未凭空增加独立实验信息**。可能的收益来自目标函数、误差抵消、对局部变化的强调。因此必须通过上述对照验证，不能假定成立。

### 9.3 外部基线

| 类别 | 基线 |
|---|---|
| 廉价 | 训练均值、kNN、ESM2 简单回归 |
| 纯序列 SOTA | **Alsamkary 的 HP / PAD 架构**（已发表，同一数据同一划分，直接引用） |
| 绝对 ΔG | ProAffinity-GNN（有代码）、PRODIGY |
| ΔΔG | FoldX、**PPIformer**、**GearBind**（可重训）、**StaB-ddG**、CATH-ddG |
| de novo | **AF2 ipTM / ipAE**（已知在 Adaptyv 上表现差 **【已核实】**，是必比的对照） |

> 只预测 ΔΔG 的方法不适合硬放进绝对 pKd 排行榜，分表报告。

---

## 10. 结论作用域规则（预先写死）

- 仅在 EGFR 上排序改善 → 只能声称"在该靶点上有效"
- 跨 N 个留出靶点均改善 → 方可声称"跨靶点可迁移"，**N 须写进摘要**
- 仅同体系排序改善、绝对误差未改善 → 收窄为**局部亲和力优化能力**
- 仅随机划分改善、严格隔离后消失 → **不得声称突破外推能力**
- 未能复现他人基线 → 先审计数据、代码版本、划分、实现与训练差异，**不得直接断定对方口径有问题**

**Adaptyv 的定位**：**只作外部验证集，不作训练集。** 跨靶点泛化的有效样本量是**靶点数**而非设计数——601 个 EGFR binder 对"能否推广到新靶点"而言 n=1，加 TREM2 为 n=2。

**必须在论文中明说的风险**：de novo binder（理想化骨架、超稳定、界面组成异于天然复合物）与天然复合物是两个分布。在天然复合物上训练、在 de novo 上测试是一个**强测试**，失败未必源于模型缺陷，也可能是分布外推。

---

## 11. 待核实清单（实施前）

**最高优先级**
- [ ] §5.1 残基名测试结果（决定 ΔΔG 支路实现路径）
- [ ] 本项目历史 MAE 的单位是 log10 还是 ln
- [ ] 自有 16 条 binder 的序列长度（是否 < 40 残基，见 §7.2）
- [ ] 自有 16 条的 **Kd 动态范围**（若全挤在 1 个 log 内，r=0.2 这个结果不可解读）
- [ ] 训练池中是否含 EGFR 复合物（见 §1.4）

**数据层**
- [ ] Alsamkary 的 30% 同一性是单边还是双边（§7.3）
- [ ] Alsamkary 的 structure config 具体提供什么（PDB 文件？链配对？仅 ID？）
- [ ] "<40 残基删除" 对 TCR-pMHC / 蛋白-肽的实际影响（§7.2）
- [ ] Alsamkary 版是否保留 WT↔突变体配对链接
- [ ] SKEMPI 中"无法检测到结合"的确切条数（约 440？）
- [ ] test1 / test2 的 σ(ΔG) 实测值

**API 层**
- [ ] PPIformer 是否暴露 per-residue logits（§9.1 的 0.4）
- [ ] 若残基名测试失败，`(G, X, E, F₀, F₁)` 构图步骤的替换接口

**文献层**
- [ ] PRODIGY 在 81 复合物基准上的 R（≈0.73？拟合值还是留出值）
- [ ] 现有亲和力预测器在 Adaptyv 上的已发表零样本结果（若已有，9.1 的 0.2/0.4 新颖性需重估）

---

## 12. 参考链接

**数据**
- Alsamkary 清洗版 — https://arxiv.org/pdf/2505.20036 ｜HF `proteinea/ppb_affinity` ｜https://github.com/Proteinea/ppiseq
- PPB-Affinity 原始 — https://www.nature.com/articles/s41597-024-03997-4 ｜https://github.com/ChenPy00/PPB-Affinity
- SKEMPI 2.0 — `life.bsc.es/pid/skempi2`
- Adaptyv R1 — https://github.com/adaptyvbio/egfr_competition_1 ｜R2 — https://github.com/adaptyvbio/egfr_competition_2 ｜论文 — https://www.biorxiv.org/content/10.1101/2025.04.17.648362v2
- PPIRef — https://github.com/anton-bushuiev/PPIRef ｜文档 https://ppiref.readthedocs.io

**模型与基线**
- PPIformer (ICLR 2024) — https://arxiv.org/pdf/2310.18515 ｜https://github.com/anton-bushuiev/PPIformer
- CATH-ddG (*Bioinformatics* 2025) — https://academic.oup.com/bioinformatics/article/41/Supplement_1/i362/8199350 ｜https://github.com/ak422/CATH-ddG
- GearBind (*Nat Commun* 2024) — https://www.nature.com/articles/s41467-024-51563-8
- StaB-ddG (ICML 2025) — https://arxiv.org/abs/2507.05502 ｜https://github.com/LDeng0205/StaB-ddG
- USP-ddG (*Bioinformatics* 2026) — https://academic.oup.com/bioinformatics/article/42/Supplement_1/btag249/8726290
- Tsishyn 偏差与误差界限 (*Brief Bioinform* 2024) — https://academic.oup.com/bib/article/25/1/bbad491/7513597 ｜https://github.com/3BioCompBio/DDGb_bias

---

## 附：本项目既有基线（供对照，MAE 单位为 log10(Kd)） **【已核实】**

| 配置 | 划分 | MAE | Pearson |
|---|---|---|---|
| 表面单模态 (V1) | 随机五折 | 1.3692 | **0.3491** |
| 纯序列 (V28) | seed20 随机五折 | 1.0470 | **0.6709** |
| 纯序列 (V26) | identity-40 | 1.1821 | **0.5616** |
| + 表面残差 (V29) | seed20 | 1.0514 | 0.6688 (−0.0021) |
| + 表面残差 (V27) | identity-40 | 1.1830 | 0.5711 (+0.0094) |

外部集（V28）：test1(79) MAE 1.1805 / PCC 0.7032；test2(82) MAE 1.1852 / PCC 0.4450
自有 16 条 binder 的 per-target 排序：**r ≈ 0.2**（n=16，95% CI 约 [−0.33, 0.63]，**统计上不可解读**）

**30 个受控版本的一致结论**：骨架二面角、原子图、完整 patch 图、MNN 残基对、物理能量项、残基级辅助监督、表面→序列蒸馏，**全部未稳定超过纯序列基线**；辅助监督的 real 与 shuffled **都**改善（说明是正则化效应而非信息）。

---

## 13. 训练监测（TensorBoard）

见仓库根目录 `README.md` 第 3 节，内容为单一来源，此处不重复。要点：

- 基于 PyTorch Lightning 的 `TensorBoardLogger`，run 命名 `runs/<实验编号>_<划分名>/seed<N>/`
- **三项损失必须分开记**（只记 total 看不出哪一项在起作用）
- **分通道梯度范数**（`grad_norm/struct_branch` vs `grad_norm/esm2_branch` 及其比值）—— 这是针对本项目 V25/V27/V29 模态坍塌失败的直接诊断
- **`val/auroc_expressed_only`** —— 防 §6.4 的 ESM2 表达 confound
- `dist/pred` vs `dist/true` 直方图并排 —— 判断"排序好但标定差"最快的方式
- λ₁ / λ₂ / lr / 冻结档位走 `add_hparams`，不要手工记表格
- **外层测试折的指标绝不进 TensorBoard 曲线**，不得用于 early-stopping 或 checkpoint 选择
