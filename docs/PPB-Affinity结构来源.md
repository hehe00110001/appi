# PPB-Affinity 结构来源（2026-10-23 核对）

已找到原始数据作者发布的完整结构包，并找到集群已有结构目录。
2026-10-23 完成按来源与文件名的覆盖检查，当前无需重新下载结构包。

## 集群现有结构与覆盖检查

- 表格：`data/raw/ppb_affinity/filtered.csv`，8207 行。
- SHA-256：`852176677ee6a7bde914349f5677239ef92929d6e2e7ca8a467af177285cfa79`。
- 结构根目录：`/home/weizg/wei/dataset/PPI/PDB`，保持原地只读复用。
- 来源目录与官方包组织一致；尚未验证这些本地文件与官方压缩包逐字节一致。
- 按来源 + PDB 编号匹配：8207/8207 条均唯一匹配非空文件，涉及 2580 个不同路径。
- PDBbind v2020 的 `xxxx.ent.pdb` 命名已处理；未使用嵌套目录中的衍生产物。
- CSV 实际 split 为 train=6485、val=965、test=757（注意 CSV 使用 val）。
- 四个来源分别为 PDBbind v2020=2242、Affinity Benchmark v5.5=130、
  SKEMPI v2.0=5109、SAbDab=726；表格没有单独标为 ATLAS 来源的行。
- 4823 行含非空突变标注；所有 KD(M) 均为有限正数。
- 审计脚本：`scripts/audit_ppb_structures.py`。
- 完整报告：`data/manifests/ppb_affinity/structure_audit.json`。
- 逐样本匹配：`data/manifests/ppb_affinity/structure_manifest.csv`。

此检查仅确认文件覆盖，不代表链、序列、突变或划分泄漏已通过审计。
正式批量编码前仍需验证以上对应关系，不能将所有突变记录直接使用 WT 编码。

## 下载来源

- 作者代码仓库：https://github.com/ChenPy00/PPB-Affinity
- 仓库链接到的 Zenodo 发布页（v5）：https://zenodo.org/records/14271435
- 主结构包：https://zenodo.org/records/14271435/files/PDB.zip?download=1
- 页面标注大小：3.1 GB。
- 页面标注 MD5：`88ba34c314b2820435afa7ccb8005b1a`。
- 建议上传位置：`/home/weizg/wei/fangyh/appi/download/PDB.zip`。

官方目录说明将结构按 Affinity Benchmark v5.5、ATLAS、PDBbind v2020、SAbDab、
SKEMPI v2.0 五个来源组织。保留来源目录，不能只按四位 PDB ID 扁平化覆盖文件。

发布页同时列有 PPB-Affinity.xlsx（1.2 MB）、PPB-Affinity-AF.zip（3.2 MB）和
samples_deleted.zip（2.1 MB）。当前优先获取主结构包 PDB.zip；其他文件视映射核查需要获取。

## 与清洗版表格的关系

Hugging Face 清洗版仓库：https://huggingface.co/datasets/proteinea/ppb_affinity/tree/main
其文件列表包含 CSV 和加载脚本，未见单独的清洗版 PDB 包。

清洗论文：https://arxiv.org/html/2505.20036v1 （2.1.1–2.1.2）

论文从原始 PPB-Affinity 的 PDB 提取序列，并修正链标注和突变位置，
例如 ATLAS 的 3QIB 链 C 突变编号修正。之后从 SEQRES 补回缺失的序列残基、
过滤短链、合并重复序列与亲和力标签。补回序列不等于补建三维坐标。

实施时以 filtered.csv 的清洗后记录为标签/序列依据，使用原始结构包作结构来源，
按来源、PDB、partner 链、突变及序列建立可追溯映射。
必须继续统计链、序列与突变映射不一致的样本；文件覆盖完整不等于
全部 8207 条均有可直接使用的独立结构。
