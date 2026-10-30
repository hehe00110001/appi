# 手动推送与预发布

Git远端：`https://github.com/hehe00110001/appi.git`。
准备分支：`codex/seed44-release-20261030`。
版本建议：`v0.1.0-rc1-seed44`（研究预发布候选）。本步骤不重训模型。

## 1. 推送代码、文档和结果

```bash
cd /home/weizg/wei/fangyh/appi
git status --short
git log -1 --oneline
git push -u origin codex/seed44-release-20261030
```

推送的是准备分支，不强推或覆盖main。可在GitHub中审阅后合并。Git中包含结果快照、模型卡、导出与推理代码；权重包保存在本地outputs，单独作为Release资产上传。

## 2. 校验模型包

```bash
cd /home/weizg/wei/fangyh/appi/outputs/releases
sha256sum -c v0.1.0-rc1-seed44.tar.gz.sha256
```

可在同目录找到归档及其校验文件。`releases/v0.1.0-rc1-seed44/artifact.json`记录本次实际包的摘要；原始模型checkpoint摘要在包内manifest中。

## 3. 可选：发布GitHub预发布版本

在第1步推送成功后，若集群已安装并登录GitHub CLI：

```bash
cd /home/weizg/wei/fangyh/appi
gh release create v0.1.0-rc1-seed44 \
  outputs/releases/v0.1.0-rc1-seed44.tar.gz \
  outputs/releases/v0.1.0-rc1-seed44.tar.gz.sha256 \
  --repo hehe00110001/appi \
  --target codex/seed44-release-20261030 \
  --prerelease \
  --title 'APPI seed44 research candidate v0.1.0-rc1' \
  --notes-file releases/v0.1.0-rc1-seed44/RELEASE_NOTES.md
```

此命令会创建标签并公开预发布资产；不是本次代理已执行的动作。不安装CLI时，可以在GitHub Releases页面手动选择该分支创建pre-release，上传同名两个文件，并复制RELEASE_NOTES.md正文。若该版本已存在，先核对既有资产，不覆盖它来冒充相同版本。

## 复现导出

原训练运行与缓存仍在本集群时，可用`python scripts/export_seed44_release.py --out /new/path/v0.1.0-rc1-seed44`重新导出；导出器拒绝覆盖已有目录。包内张量应一致，但压缩文件含文件时间等元数据，重新打包的归档SHA256未必相同，应使用该次生成的校验文件。
