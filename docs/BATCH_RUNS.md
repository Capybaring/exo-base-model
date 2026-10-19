# 全部数据运行与无 LIFT 消融

入口：`python -u scripts/run_all.py`。旧 `scripts/train.sh` 只运行单数据集，根目录 `run_remaining_dag.py` 只覆盖九个数据集；新的入口统一覆盖全部 13 份数据。

## 常用命令

在已安装依赖并准备好数据的仓库中：

```bash
git pull --ff-only origin master

# 全部数据，有 LIFT：TTCN + DAG + FFT/global LIFT
python -u scripts/run_all.py --device cuda:0

# 全部数据，无 LIFT：相同基础预测路径，仅移除 LIFT
python -u scripts/run_all.py --device cuda:0 --no-lift

# 推荐的成对实验：每个数据集依次运行有/无 LIFT，自动汇总差值
python -u scripts/run_all.py --device cuda:0 --variants dag-lift dag-history
```

这三条是三种运行选择；已经执行成对命令后，无需再重复执行前两条。`cuda:0` 可改为实际可见卡号（如 `cuda:1`），CPU 使用 `--device cpu`。省略 device 时按当前 PyTorch 的 CUDA 可用性自动选择。

每个任务默认沿用训练入口的 100 epochs、patience=10、d_model=64，以及各数据集预设的 batch size；可统一覆盖，例如 `--epochs 100 --patience 10 --batch-size 8 --d-model 32`。不会添加 max-records 或执行小样本限制，除非显式使用 `--smoke`。

## 数据范围与时间窗口

默认 all：DIST、NP、PJM、BE、FR、DE、Energy、Sdwpfm1、Sdwpfm2、Sdwpfh1、Sdwpfh2、Colbun、Rapel。

**批量入口默认 24 步历史 → 24 步预测**，明确覆盖原 JSON 的 168→24 等时长，避免把两种任务混在同一结果中：

| 数据 | 一个 patch/步 | 默认历史与预测各多长 |
|---|---|---|
| DIST | 1 小时（本批量协议的显式选择） | 24 小时 |
| NP、PJM、BE、FR、DE、Energy、Sdwpfh1、Sdwpfh2 | 1 小时 | 24 小时 |
| Sdwpfm1、Sdwpfm2 | 30 分钟 | 12 小时 |
| Colbun、Rapel | 1 天 | 24 天 |

`--history-steps` 和 `--forecast-steps` 可统一调整步数。时间换算只改变本次运行的 patch 网格和窗口参数，不修改 TTCN 编码方程、向量维度设计、查询时间解码或数据文件。

默认读取现有配置中的 NPZ。BE/NP 若主数据目录不存在而 table3 包中存在对应 NPZ，会使用 table3 包的文件并将实际路径、SHA256 保存到计划中。根目录数据文件不随 Git 下载，仍需先解压之前准备的数据包。

```bash
# 只跑表格中的 12 份，排除 DIST
python -u scripts/run_all.py --suite table3 --device cuda:0 --variants dag-lift dag-history

# 只跑 BE/NP，或指定装有 DATASET.npz 的数据目录
python -u scripts/run_all.py --datasets BE NP --device cuda:0 --no-lift
python -u scripts/run_all.py --suite table3 --data-dir /path/to/table3_irregular --device cuda:0
```

## “无 LIFT”的两个对照必须区分

| variant | Y 基础预测的输入路径 | LIFT |
|---|---|---|
| dag-lift | 历史预测未来 X，再驱动变量分支；实际未来 X 通过 LIFT 对齐支路进入 | 有 |
| dag-history / --no-lift | 同一个历史驱动的基础预测路径 | 无 |
| dag | 实际已知未来 X 直接进入 DAG 变量分支 | 无 |

`--no-lift` 是 `dag-history` 的明确别名：去掉领先估计、对齐和频域修正模块。保留同样的 TTCN、DAG 双分支、查询解码、未来 X 预测辅助任务和历史 Y 重构辅助任务。实际未来 X 在此变体中仍可用于训练辅助监督，但不作为最终预测的直接输入。

因此 dag-lift 对比 dag-history 测的是“移除整个 LIFT 路径”的效果，同时也移除了通过该路径使用实际未来 X 的能力。要进一步区分“利用已知未来 X”与“LIFT 修正机制”的收益，加入直接未来 X 的 dag 对照：

```bash
python -u scripts/run_all.py --device cuda:0 --variants dag-lift dag-history dag
```

三个变体沿用既有实现，没有改动 patch 或基础预测方程。相同数据集的变体使用相同数据文件、记录划分、归一化协议、窗口、模型宽度和随机种子；各自独立训练，不是把某个有 LIFT 权重临时关闭来充当重训后的消融结果。不同结构的随机数消耗可能不同，单个 seed 不代表统计显著性，可另设 `--seed 2`、`--seed 3` 重复。

单数据集也支持明确的移除开关：

```bash
python -u -m exobase.train --config configs/BE.json --no-lift \
  --history 86400 --forecast 86400 --patch-seconds 3600 --patch-stride 3600 \
  --save-path results/BE/no-lift-24to24
```

## 检查、输出与失败处理

```bash
# 加载并验证所有数据、打印计划，不训练也不生成结果目录
python -u scripts/run_all.py --device cuda:0 --variants dag-lift dag-history --dry-run

# 八条记录、小模型、一轮的管线检查；不能用于精度表
python -u scripts/run_all.py --datasets BE --device cpu --variants dag-lift dag-history --smoke
```

开始任何训练前检查全部指定数据文件、时间单位、有效记录、划分和训练集统计量。缺少数据时一起报告并退出，不先运行其它数据集。

默认生成独立且唯一的 `results/all_24to24/run_<UTC时间>/`，包含：

- `plan.json`：实际窗口、数据 SHA256、记录 ID、随机种子、全部命令。
- `DATASET/variant/`：各自 manifest、权重、metrics，有 LIFT 时额外包含 test_lags.csv。
- `DATASET_variant.log`：每个训练任务的完整日志。
- `summary.csv`：测试标准化 MSE/MAE/RMSE、持续值基线、best_epoch 和成功/失败状态。
- `ablation_comparison.csv`：有/无 LIFT 配对结果；差值为“有 LIFT − 无 LIFT”，负值表示有 LIFT 的误差更低。只有两个任务都成功时才写入对应数据集。

默认顺序执行，不同时启动多个 GPU 作业；一个任务失败会记录并继续，最终返回非零退出码。`--fail-fast` 可在首次失败时停止。显式 `--output-dir` 必须是新目录，不会覆盖旧结果。大规模训练未在本机代跑；自动测试和小样本检查仅验证功能及实验隔离。
