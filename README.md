# exo-base-model

面向不规则观测、已知未来外生变量 X、目标变量 Y 的预测模型。

当前默认架构 v4：**历史 Y → TTCN patch → 时间注意力基础预测；历史 X 与已知未来 X → 独立 TTCN → lag 对齐 → X→Y 特征映射 → LIFT 残差修正 → 实际查询时间解码**。

主干只使用历史 Y。没有 DAG 注意力迁移、X 未来预测、历史 Y 重构或对应辅助损失。X/Y 按角色分别共享 TTCN，X 的历史/未来使用同一个编码器。TTCN 的点值、时间、mask 聚合公式保持不变，不实例化 GNN。详细方程和适配边界见 [v4 结构说明](docs/Y_BASE_V4.md)。

## 安装与数据

Python 3.10+：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

正式训练可安装 `pip install -e .`；旧 delay 模式库和全部回归测试需要 `.[legacy]` 或 `.[test]`。验证版本见 requirements-tested.txt。

支持 `date,data,cols` / `time,value,variable` 长表及 NPZ。数据准备保留实际时间、独立 mask 和真实零值，同变量同时间重复观测取均值，不修改原始输入：

```bash
python -m exobase.prepare \
  --dist /path/to/DIST_irregular_9x9.csv \
  --dataset-dir /path/to/dataset --drop-rate 0.3 --seed 2021
```

DIST 保留原有不规则性、不额外丢弃；BE/NP 逐变量独立随机丢弃 30%，默认也作用于未来 Y 标签，评估仅统计保留标签。数据、权重、预测文件被 Git 忽略，需要单独准备。12 份 Table-3 数据与全部运行入口见 [批量运行](docs/BATCH_RUNS.md)。

## Base、patch 与 LIFT 实验

```bash
git pull --ff-only origin master

# 全部 13 个数据集，依次独立训练三种结构，默认 24→24 原生步
python -u scripts/run_all.py --device cuda:0 \
  --variants y-base patch-base patch-lift

# 后续与已有 patch+DAG 比较
python -u scripts/run_all.py --device cuda:0 --variants dag
```

| variant | Y 基础预测 | 已知未来 X 的预测路径 | 辅助损失 |
|---|---|---|---|
| y-base | 历史 Y 固定网格标量均值 + 时间注意力 | 无 | 无 |
| patch-base | 历史 Y 的 TTCN + 相同时间注意力 | 无 | 无 |
| patch-lift（默认） | 与 patch-base 相同 | 仅经过 lag 对齐和 LIFT | 无 |
| dag | 保留原 TTCN + DAG 双分支 | 直接进入 DAG 变量分支 | 预测未来 X、重构历史 Y |
| dag-history | 原 DAG 历史驱动双分支 | 无，使用历史预测出的 X | 同上 |
| dag-lift | 与 dag-history 相同 | 仅经过原 v3 的 LIFT | 同上 |

`y-base` 仍按同一时间网格形成标量序列，只是不使用 TTCN 的可学习 patch 聚合；不是完全不分块的原始不规则点 Transformer。`patch-base` 与 `patch-lift` 在相同 seed 下有完全相同的 Y 主干初始参数，随后分别独立训练。

`--no-lift` 默认选择 patch-base；显式 `--variant dag-lift --no-lift` 仍选择旧 dag-history。原来的 DAG 及旧 checkpoint 保留，见 [v3 文档](docs/TTCN_V3.md)、[LIFT 官方对齐记录](docs/LIFT_OFFICIAL.md)。旧 tpatch-y/tpatch/exo/delay 变体需显式指定。

批量默认 DIST + 12 份 Table-3 数据。`--datasets BE NP PJM` 选子集，`--suite table3` 排除 DIST，`--data-dir /path/to/npz` 指定含 DATASET.npz 的目录。24 步对小时数据为 24 小时、半小时数据为 12 小时、日数据为 24 天；DIST 批量协议使用 1 小时 patch。

```bash
# 小规模检查，不作正式精度结果
python -u scripts/run_all.py --datasets BE --device cpu \
  --variants y-base patch-base patch-lift --smoke

# BE 单数据集 24 小时→24 小时 patch+LIFT
python -u -m exobase.train --config configs/BE.json --variant patch-lift \
  --history 86400 --forecast 86400 --patch-seconds 3600 --patch-stride 3600 \
  --save-path results/BE/patch-lift-24to24
```

单数据集 JSON 保留各自时长（例如 configs/BE.json 为 168→24 小时），命令行覆盖 JSON。正式 24→24 对照优先使用批量入口。

## Lag 与特征修正

- v4 在历史观测的标量 patch 均值上估计 X→Y lag；摘要只用于 lag，主干和修正保留完整 TTCN 特征。估计不读取编码器参数或时间特征。
- 默认循环 FFT、局部峰值、绝对相关 top-K，搜索正 lag `1…L−2`。`--lag-estimator pearson` 才使用 max_lag/min_pairs/min_correlation。空/常量通道排除，缺失块不压缩、不插值。lag 是非参数估计，不反向传播。
- 按 `X[L+h-lag]` 从历史和实际已知未来 X 取特征，按历史 X 统计量归一化并按相关符号对齐，再经可训练映射进入 Y 修正坐标。
- 复用状态估计、2K+1 滤波器、三路频域修正、复数混合和残差。v4 允许梯度流入 X 编码器与映射层；旧 DAG/LIFT 的 leader detach 行为保留。
- 默认 global 在整个预测区间混合，不保证逐输出的外生时间边界；prefix 是显式对照。无可用 leader 时严格回退基础预测。

这些是针对本任务的适配，不声称未经改编复现原 LIFT。正 lag 不包含同时刻关系，相关领先也不等于已识别的因果时延。

## 损失、输出与重载

v4 三个变体只优化**未来 Y 有效查询点的 MinMax masked variable-macro MSE**，辅助损失为零；beta 只用于旧 DAG。验证选模和测试主指标使用仅训练集拟合的 Z-score，逐变量累计误差/计数后宏平均；另保存原量纲、MinMax、micro 和持续性/训练均值基线，RMSE=sqrt(MSE)。详见 [指标说明](README_zh.md)。

查询解码器将所属未来 token 与实际查询时间编码结合，同 patch 内不同时间点可得到不同值。未来 Y 标签和标签 mask 不参与模型前向。

训练保存 manifest.json、best_model.pt、metrics.json；有 LIFT 时另存 test_lags.csv。批量输出 summary.csv、ablation_comparison.csv、component_comparison.csv、日志和计划。

```bash
python -m exobase.predict \
  --checkpoint results/BE/patch-lift-24to24/best_model.pt \
  --output results/BE/patch-lift-24to24/test_predictions.csv \
  --lag-output results/BE/patch-lift-24to24/reloaded_lags.csv

python -m exobase.evaluate \
  --checkpoint results/BE/patch-lift-24to24/best_model.pt
```

按 checkpoint 保存配置构建模型，校验数据 SHA256 与划分；旧权重保持旧架构。新结构需要重新训练，功能验证不代表精度提高。见 [验证记录](docs/VALIDATION.md)、[来源](docs/PROVENANCE.md)。
