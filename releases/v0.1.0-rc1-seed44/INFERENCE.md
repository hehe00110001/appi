# seed44 推理说明

这是冻结特征上的评分网络。输出pKd越高表示模型预测结合越强，不是已校准的概率。原始PDB/序列必须先经与训练一致的特征提取和节点映射，不能用任意ESM池化向量代替全链记忆。

## 依赖

发布包内的纯推理代码只依赖 Python 和 PyTorch；`requirements.txt` 记录导出环境的PyTorch版本。GPU版本安装按机器CUDA环境配置。PPIformer和ESM特征提取还需要各自依赖与独立获取的编码器权重，不包含在包内。

## 输入缓存

`--features` 目录包含：

- `residue_features.pt`：sample_id→图；`h`形状[N,128]、`coordinates`[N,3]、`partner`[N]。
- `manifest.json`：图节点到全链序列位置的映射、链伙伴和序列ID。
- `sequences/<sequence_sha256>.pt`：完整序列及[L,1280]的ESM特征。
- `extraction_config.json`：ESM版本、层、滑窗和校验值。
- `config.json`：PPIformer checkpoint、界面提取代码校验、裁剪半径和编码设备。

本项目的 `scripts/extract_adaptyv_egfr.py` 已生成这种格式。通用输入适配器尚未发布；不能直接传入任意PDB文件。发布推理入口会检查权重、实现代码和关键提取配置。

## 集群现有缓存示例

在仓库目录下执行，`--ids`省略时评估缓存全部样本：

```bash
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  /home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python \
  scripts/predict_release.py \
  --release outputs/releases/v0.1.0-rc1-seed44 \
  --features data/processed/adaptyv_egfr_features_v1 \
  --device cpu --out outputs/releases/seed44_predictions.csv
```

解压后的独立发布包也包含相同入口：

```bash
python scripts/predict_release.py \
  --features /absolute/path/to/compatible_features \
  --device cuda --out predictions.csv
```

独立包默认以自身目录为`--release`。CSV提供`sample_id,predicted_pkd`。建议在同一靶点/测量条件内比较分数；切勿据外部结果翻转分数、重新拟合归一化或挑选checkpoint后仍称冻结验证。
