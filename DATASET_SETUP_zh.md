# 远程数据放置说明

请把这个压缩包解压到 exo-base-model 仓库根目录。无需修改三个现有配置的数据路径。

## 解压后目录

exo-base-model/
  configs/DIST.json
  configs/BE.json
  configs/NP.json
  data/processed/
    DIST.npz
    BE.npz
    NP.npz
    BE_irregular.csv
    NP_irregular.csv
    preparation_manifest.json
    SHA256SUMS
  DATASET_SETUP_zh.md

注意不要解压成 exo-base-model/data/data/processed/ 或 exo-base-model/exo-base-model/data/processed/。
压缩包已经包含 data/processed/ 这一层路径，因此请直接在仓库根目录解压。

## 远程操作

先把 exo-base-datasets-20261017.zip 上传到服务器任意位置，然后进入已有仓库：

    cd /你的仓库路径/exo-base-model
    git pull --ff-only
    unzip /你上传的位置/exo-base-datasets-20261017.zip -d .
    sha256sum -c data/processed/SHA256SUMS

不要用 unzip -o 强制覆盖不确定版本的现有数据；请先确认目录或用新的仓库副本。

如尚未安装当前 Python 环境的依赖：

    python -m pip install -e .

分别训练三份数据（按需运行）：

    python -u -m exobase.train --config configs/DIST.json
    python -u -m exobase.train --config configs/BE.json
    python -u -m exobase.train --config configs/NP.json

有 CUDA 的 PyTorch 环境会默认选择 CUDA；需要明确选择时，在命令后加 --device cuda 或 --device cpu。
默认模型 dag-lift，默认按时间划分、只用训练记录拟合归一化。正式训练不加 --max-records。

先做小规模流程检查可运行：

    python -u -m exobase.train --config configs/DIST.json --max-records 8 --epochs 2 --batch-size 1 --d-model 8 --n-heads 2 --state-num 2 --save-path results/remote_smoke_DIST

## 数据是什么

- DIST.npz：来自 DIST_irregular_9x9.csv，保留原本不规则时间和缺失 mask，没有再随机丢弃；9 个 PV 为 Y，9 个 OP 为 X。重复的同变量同时间观测取均值（483 条合并）。
- BE.npz、NP.npz：从原始数据按变量、逐观测独立随机丢弃约 30%，seed=2021。丢弃覆盖整个数据流，包括未来 Y 标签；评估仅统计保留标签。
- 两份 *_irregular.csv 是对应丢弃后的长表，供检查/重新处理；训练配置默认读取 npz，不需要额外转换 CSV。
- npz 是未归一化、不规则观测缓存，并非预先训练好的模型、固定划分或已经划好的 Patch。时间、值、mask、变量名保存在一起；训练时才分块和归一化，因此调整 Patch 宽度不需要重新生成该缓存。
- 原始 500MB 以上的 DIST CSV 及未丢弃的 BE/NP 原始文件未放入本包。使用现有配置训练不需要它们。
- BE/NP 的外生变量名保留原始前导空格，勿在配置里自行删除空格。

本包的数据字节与已验证本地数据一致。preparation_manifest.json 只将机器相关路径改成相对路径，并保留原始来源 SHA256、丢弃设置与计数。
当前代码版本至少应包含 2fa42e0（DAG + X→Y LIFT 迁移）。实验范围与限制见仓库 docs/VALIDATION.md。
