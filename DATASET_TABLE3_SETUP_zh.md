# 图中 12 个数据集的不规则版本

包含 NP、PJM、BE、FR、DE、Energy、Sdwpfm1、Sdwpfm2、Sdwpfh1、Sdwpfh2、Colbun、Rapel，共 12 份。相较之前新增 10 份；NP、BE 的 CSV/NPZ 与之前提供的版本逐字节相同。

## 远程放置

将 ZIP 上传到服务器后，在当前 exo-base-model 项目根目录执行：

```bash
unzip /上传路径/exo-table3-irregular-30pct-seed2021.zip -d .
sha256sum -c data/processed/table3_irregular/SHA256SUMS
```

解压目录是：

```text
项目根目录/
  exobase/
  configs/
    table3_irregular/
      NP.json
      NP_long.json
      PJM.json
      PJM_long.json
      ...其余数据集短期与长期配置...
  data/processed/
    table3_irregular/
      NP.npz
      NP_irregular.csv
      PJM.npz
      PJM_irregular.csv
      ...其余数据集...
      dataset_summary.csv
      table3_irregular_manifest.json
      validation_summary.json
      SHA256SUMS
  DATASET_TABLE3_SETUP_zh.md
```

本包采用独立子目录，不替换之前的 DIST、NP、BE 文件与原有配置。适用于当前 DAG/LIFT 代码（提交 2fa42e0），新增数据集通过本包附带的 JSON 配置读取，无须修改模型代码。代码 ZIP 版本也可直接解压使用本包。

## 训练（GPU 1）

例如，短期 PJM：

```bash
python -u -m exobase.train --config configs/table3_irregular/PJM.json --device cuda:1
```

长期 PJM：

```bash
python -u -m exobase.train --config configs/table3_irregular/PJM_long.json --device cuda:1
```

Energy、Sdwpfm1、Colbun 等同样替换配置文件名即可。cuda:1 表示当前进程可见的编号 1 卡；若已设置 CUDA_VISIBLE_DEVICES=1，则命令内应使用 cuda:0。

首次运行先检查流程：

```bash
python -u -m exobase.train --config configs/table3_irregular/PJM.json --device cuda:1 --max-records 8 --epochs 2 --batch-size 1 --d-model 8 --n-heads 2 --state-num 2 --save-path results/remote_smoke_PJM
```

正式实验移除 max-records，并使用新的 save-path 保存不同实验。默认模型 dag-lift、按时间划分、仅使用训练记录拟合归一化。短/长配置结果输出目录不同。

## 变量与时间间隔

目标选择对应原 DAG 脚本 target_channel=[-1]，保留原始变量顺序、名称、空格与拼写。其他变量作为 X；没有统一重命名为 OT。

| 数据集 | Y 目标 | X 数量 | 原始采样间隔 |
|---|---|---:|---|
| NP | OT | 2 | 1 小时 |
| PJM | OT | 2 | 1 小时 |
| BE | OT | 2 | 1 小时 |
| FR | OT | 2 | 1 小时 |
| DE | OT | 2 | 1 小时 |
| Energy | Thermoelectric | 5 | 1 小时 |
| Sdwpfm1 | Patv | 6 | 30 分钟 |
| Sdwpfm2 | Patv | 6 | 30 分钟 |
| Sdwpfh1 | Patv | 6 | 1 小时 |
| Sdwpfh2 | Patv | 6 | 1 小时 |
| Colbun | Water_Level_daily_avg | 2 | 1 天 |
| Rapel | Water_Level_daily_avg | 2 | 1 天 |

全部 X 变量名、实际丢弃比例与文件映射见 dataset_summary.csv 和 manifest。PJM 原始变量名中的 foecast 等拼写、部分名称的前导空格均保留。

## 两种预测长度

- 前 10 份数据：短期历史 168 步、预测 24 步；长期历史 720 步、预测 360 步。
- Colbun、Rapel：短期历史 60 步、预测 10 步；长期历史 180 步、预测 30 步。
- 每步按上表原始时间间隔转换成秒。Sdwpfm 的 24 步表示 12 小时；水文数据的 10 步表示 10 天。
- 当前 Patch 宽度为一个原始时间步。这里对应图中的两种预测长度，但记录划分、归一化、模型与评价仍采用当前框架；不是原论文实验结果的复现。
- 长期配置使用 batch_size=2，短期为 16；这些是可调整的起点，未做完整精度调参。

## 不规则化方式

每个数据集采用 seed=2021，对长表每条变量观测独立作 Bernoulli 丢弃，请求丢弃率 30%，实际约 29.61%–30.12%。不同变量的保留时间集合不同；不是同步删除整个多变量时刻。

只随机移除观测，保留剩余观测原有时间戳和数值，不添加时间扰动、不插值、不压缩缺口。丢弃覆盖整个流，包括未来 Y 标签；评估只计算保留标签。若要固定完整测试标签，需要另设数据协议。

CSV 保存丢弃后的长表。NPZ 保存未归一化的时间并集、值、独立 mask 和变量名；它还没有划分记录或完成 Patch 聚合。训练时才按配置完成分块与缩放。真实零值仍为有效观测。

原始完整数据和 DIST 不包含在本包中；DIST 可继续使用之前的数据包。原始输入文件没有改动，原文件 SHA256、逐变量观测数、转换结果哈希保存在 manifest。

## 已验证范围

- ZIP 文件与全部打包文件有校验。
- 12 份数据保留行均与原始观测数值/时间戳核对。
- 所有 24 个配置均通过当前框架的完整数据加载、按时间划分、归一化和小模型前向检查。
- 新增 10 份数据各通过 8 条记录、1 epoch 的训练—验证—测试、checkpoint 与滞后文件保存检查。
- 以上为 CPU 小规模流程验证，不是正式精度比较；未执行完整多种子训练或 GPU 验证。
