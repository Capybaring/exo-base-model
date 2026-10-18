# exo-base-model

面向 **不规则观测、已知未来控制量 X、过程量 Y 的滞后预测 baseline**。

当前训练默认使用架构 v3：**原始观测值/时间/mask → t-PatchGNN TTCN 向量 Patch → DAG 双注意力 → X→Y 向量 LIFT → 实际查询时间解码**。不使用 patch 标量均值，不实例化 t-PatchGNN 的 GNN。具体改编边界见 [TTCN v3 说明](docs/TTCN_V3.md)、[来源版本](docs/PROVENANCE.md)。旧架构 v2 的均值模型仍可按 checkpoint 原配置重载。

这是可运行的研究 baseline，尚未完成正式精度比较。估计的是预测领先关系，不把相关性直接解释为已识别的物理因果时延。

## 安装

Python 3.10+，已验证 macOS ARM64、Python 3.12、CPU。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

正式训练只需 `pip install -e .`。k-Shape/tslearn 不再是默认模型的依赖；仅旧 delay 变体和完整回归测试使用，可安装 `.[legacy]` 或 `.[test]`。实际验证版本见 requirements-tested.txt。

## 数据准备

支持 `date,data,cols` 或 `time,value,variable` 长表，以及 `.npz` 缓存。保留时间戳、变量独立 mask 和真实零值；同变量同时间重复观测取均值。不修改原始输入文件。

```bash
python -m exobase.prepare \
  --dist /path/to/DIST_irregular_9x9.csv \
  --dataset-dir /path/to/dataset --drop-rate 0.3 --seed 2021
```

输出在 data/processed/：DIST 保留原有不规则性、不额外丢弃；BE、NP 对各变量逐观测独立随机丢弃 30%，保存丢弃长表、缓存和 preparation_manifest.json。默认丢弃也作用于未来 Y 标签，评估只统计保留标签。这不同于“只丢输入、固定完整测试标签”的协议。

代码仓库不包含原始数据、生成的数据缓存、权重或预测文件，这些文件均被 Git 忽略。

## 规则化接口与配置

Patch 延续 t-PatchGNN 的**固定真实时间区间归属**：同一块内每变量可有不同数量的观测。每个观测先拼接可学习时间编码，再经官方 TTCN 的逐观测动态滤波，生成 d_model 维 token（含非空标记）。非空 patch 的公式与固定官方源码对齐；补充全空和隐藏 NaN 防护。DAG 直接处理多维 token，未先投影成一个标量。

| 配置 | Y / X | 历史 / 预测 | Patch 宽度 | 最大候选时延 |
|---|---|---|---|---|
| configs/DIST.json | 9 个 PV / 9 个 OP | 4 小时 / 2 小时 | 60 秒 | 60 块 = 60 分钟 |
| configs/BE.json | OT / 两个预测协变量 | 168 小时 / 24 小时 | 1 小时 | 48 块 = 48 小时 |
| configs/NP.json | OT / 两个预测协变量 | 168 小时 / 24 小时 | 1 小时 | 48 块 = 48 小时 |

BE/NP 变量名的前导空格保留原样。时间参数单位为秒，max_lag 单位为块。配置是实验起点，时延上限尚未经过任务验证。DAG 变体要求非重叠、历史和预测边界对齐的 Patch。

空块不删除、不压缩时间，TTCN 输出全零及非空标记 0。滞后估计使用非空 mask，仅统计有效 patch 对。每个未来块输出完整特征 token，再结合真实查询时间解码，因此同一块内不再强制输出常数。滞后网格的分辨率仍为 patch 宽度。

## 训练与消融

```bash
python -u -m exobase.train --config configs/DIST.json
python -u -m exobase.train --config configs/BE.json
python -u -m exobase.train --config configs/NP.json
```

命令行覆盖 JSON 配置。当前变体共用同一数据接口：

| variant | 基础骨架 | 未来 X 如何进入 Y 预测 |
|---|---|---|
| dag-history | DAG 时间/变量双分支 | 不直接读取，用历史预测的 X 驱动变量分支 |
| dag | DAG 时间/变量双分支 | 直接进入原 DAG 变量分支，无显式时延约束 |
| dag-lift（默认） | 与 dag-history 相同 | 仅经 X→Y 领先估计、移位对齐与频域修正进入 |

DAG 的外生预测辅助任务只用 X 历史预测 X，变量分支重构/预测 Y。dag-history/dag-lift 在训练与推理中都采用预测 X 的基础分支路径，避免未来 X 绕过滞后模块。未来 X 仍可监督外生预测辅助任务；未来 Y 只用于损失和评估。

```bash
bash scripts/run_ablations.sh BE --epochs 50
```

建议先做小规模管线检查：

```bash
python -u -m exobase.train --config configs/DIST.json \
  --max-records 8 --epochs 2 --batch-size 1 \
  --d-model 8 --n-heads 2 --state-num 2 \
  --save-path results/smoke_DIST_dag_lift
```

max-records 改变划分和归一化，只用于检查，正式实验应移除。默认 `chronological` 按记录时间约 60/20/20 划分，`normalization=train` 只用训练记录拟合缩放。仍提供 `official-random` 与 `seen` 对照旧 t-PatchGNN 协议，它们不是当前默认值。

## LIFT 迁移内容

- 仅估计 X 历史领先 Y 历史的正时延，不学习 Y→X 候选图；不同 X/Y 对可以有不同领先步数。
- 逐记录计算共享 TTCN 特征空间中的带 mask、非循环相关：先对每个特征按有效配对时间中心化，再汇总特征协方差及方差。以绝对相关选择时延与 top-K 外生来源。它不等于原量纲 Pearson；历史原始常量变量额外排除，防止仅时间编码产生伪 leader。配置中的最少配对数按有效 patch 对计数。
- 对未来第 h 块，按 `X[T+h-delta_ji]` 从历史/已知未来 X 取值，不用预测的未来 Y 估计时延。
- 保留 LIFT 状态相关滤波器工厂，以及“领先信号 / 与基础预测之差 / 基础预测”三路频域混合。FFT 仅沿 patch 时间轴，完整特征维保留；滤波器在特征间共享。状态分类器读取完整历史特征，未做 patch 内均值。
- 默认 prefix 适配逐输出截断可用的对齐控制轨迹，防止全预测区间混合让晚发生的控制提前影响输出。
- `--refiner-mode global` 提供全预测区间频域混合对照；不保证逐输出的控制时间边界。没有有效领先关系时回退到基础预测。

这不是未经修改的原始 LIFT：源/目标集合限制、带 mask 非循环估计、已知未来 X、prefix 约束及 DAG 骨架都是明确适配。时延估计器为非参数，滤波器与骨架通过损失训练。

## 输出、重载与诊断

训练保存 manifest.json、best_model.pt、metrics.json；dag-lift 额外保存 test_lags.csv，包括记录 ID、X 源、Y 目标、候选时延块/秒、相关系数、有效配对数、是否有效、是否选入 top-K。valid=0 的候选不用于修正；没有足够配对时，时延留空，不表示真实零延迟。

```bash
python -m exobase.predict \
  --checkpoint results/DIST/dag-lift/best_model.pt \
  --output results/DIST/dag-lift/test_predictions.csv \
  --lag-output results/DIST/dag-lift/reloaded_lags.csv
```

导出原始量纲预测、真实相对秒、标签观测标志与有效标签。重载校验数据 SHA256、划分和模型配置；覆盖已有结果需 `--overwrite`。

主损失是原始有效 Y 查询上的 MinMax masked MSE；辅助任务同样在原始观测时间解码并监督未来 X 和历史 Y，不将可学习 token 自身作为标签。评估主 MSE/MAE 使用仅训练集原始观测拟合的 Z-score，累计各变量误差/计数后宏平均；另记录原量纲、模型 MinMax、micro 和持久性基线。RMSE=sqrt(MSE)。

旧 tpatch-y/tpatch/exo/delay 变体及 checkpoint 重载仍保留，必须显式选择；它们不具备默认模型的方向/时延约束。历史说明见 [v0.1 存档](docs/LEGACY_V1.md)，当前验证范围见 [验证记录](docs/VALIDATION.md)。
