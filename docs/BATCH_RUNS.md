# 全部数据、base/patch/LIFT 与 DAG 对照

从仓库根目录执行，三条训练命令按需要选择：

```bash
git pull --ff-only origin master

# 当前主对照：相同已知未来外生信息，按各数据集短期窗长
python -u scripts/run_all.py --device cuda:0 --protocol dataset-short --variants patch-lift dag

# 后续运行原 patch+DAG，保持相同 seed/数据/窗口
python -u scripts/run_all.py --device cuda:0 --variants dag

# 如需历史 Y-only 参考模型，显式运行
python -u scripts/run_all.py --device cuda:0 --variants patch-base
```

默认只运行 patch-lift；`--no-lift` 只运行 patch-base。显式 `--variants dag-lift --no-lift` 仍选旧 dag-history，原 `--variants dag-lift dag-history dag` 组合保留。

## 变体与比较含义

- y-base：历史 Y 固定网格标量均值 + 输入投影 + 时间注意力，无 TTCN。
- patch-base：原 TTCN + 同一 Y-only 时间主干，无辅助任务、不读取 X。
- patch-lift：相同 Y 主干，X/Y 独立 TTCN，lag 对齐后映射并频域修正，无辅助任务。
- dag：保留原共享 TTCN + DAG 双分支/迁移，已知未来 X 直接进入变量分支；仍训练预测 X 与重构 Y 的辅助任务。
- dag-history/dag-lift：保留原 v3 的历史驱动 DAG 基线及其 LIFT。

`y-base` 移除的是可学习 TTCN 聚合，仍在固定时间网格上处理不规则观测。patch-base→patch-lift 同时增加 LIFT 与已知未来 X 路径。dag 与 patch-lift 的容量、编码器共享、训练目标也不同，不能把差值解释为只替换一个滤波算子。详见 [v4](Y_BASE_V4.md)。

## 数据与窗口

默认 all：DIST、NP、PJM、BE、FR、DE、Energy、Sdwpfm1、Sdwpfm2、Sdwpfh1、Sdwpfh2、Colbun、Rapel。`--suite table3` 排除 DIST；`--datasets BE NP PJM` 选子集。

默认 **dataset-short**，读取 JSON 中的历史、预测、patch 时长：

| 数据 | 每个 patch | 历史→预测 |
|---|---|---|
| DIST | 1 分钟 | 2 小时→30 分钟（120→30 patches） |
| NP、PJM、BE、FR、DE、Energy、Sdwpfh1、Sdwpfh2 | 1 小时 | 168 小时→24 小时 |
| Sdwpfm1、Sdwpfm2 | 30 分钟 | 84 小时→12 小时（168→24 原生步） |
| Colbun、Rapel | 1 天 | 60 天→10 天 |

`dataset-long` 对 BE/NP 等使用 720→360 原生步，日数据为 180→30 天；DIST 不在 DAG 论文中，没有论文长期预设，使用此协议时需排除 DIST。`uniform` 显式保留之前的 24→24 协议（DIST 为一小时 patch），此时才允许 `--history-steps`、`--forecast-steps` 覆盖。详情和论文原始来源见 [预测任务标准](FORECAST_TASKS.md)。

同一任务的变体保持相同 seed、数据、chronological 划分和 train-only normalization，各自独立训练。patch-base/patch-lift 的 Y 模块在同 seed 下逐值相同初始化。

默认读取配置路径；BE/NP 主目录文件不存在时回退 table3 目录。可用 `--data-dir /path/to/npz` 指向含 DATASET.npz 的平铺目录。数据不随 Git 下载。

```bash
python -u scripts/run_all.py --datasets BE NP PJM --device cuda:0 \
  --protocol dataset-short --variants patch-lift dag --epochs 100 --patience 10

python -u scripts/run_all.py --suite table3 --data-dir /path/to/npz \
  --device cuda:0 --protocol dataset-long --variants patch-lift dag

# 显式重跑旧 24 小时→24 小时任务
python -u scripts/run_all.py --datasets DIST BE NP --protocol uniform \
  --history-steps 24 --forecast-steps 24 --device cuda:0 --variants patch-lift dag
```

默认 100 epochs 上限、patience=10、d_model=64，batch size 来自预设；可统一覆盖。beta 对 v4 自动设为 0，仅在旧 DAG 生效。FFT/global 默认保留，标量摘要只用于新 lag 或 y-base；patch-base/patch-lift 主路径必须使用 TTCN。

## TTCN 批量计算试验

`--ttcn-execution full` 把同一次编码的全部“样本×变量×patch”单元合并计算，不使用 TTCN 内部的 16 单元循环或梯度检查点。这遵循官方 TTCN 的批量执行方式；继续使用本仓库的安全缺失处理、广播聚合和全空 patch 处理，并非替换为整套官方 t-PatchGNN。实际时间点解码的分块/检查点不变。公式、参数、观测、窗口、划分、损失和 batch size 均不因此改变。

默认 `chunked` 完整保留此前 16 单元分块与重计算方式，旧 checkpoint 缺少该设置时也使用 `chunked`。运行设置记录在 manifest 的 args/model_config 中，重载时自动恢复，不改变 state_dict 参数键。支持 patch-base、patch-lift 和采用 TTCN 的 DAG 变体；不适用于 y-base/mean。

```bash
# 在原 batch size=64 下试跑 3 轮，比较每轮 seconds；每次创建新结果目录
python -u scripts/run_all.py --datasets DIST --protocol dataset-short \
  --variants patch-lift --ttcn-execution full --batch-size 64 \
  --device cuda:0 --epochs 3 --patience 3

# 同一设置的原执行方式对照
python -u scripts/run_all.py --datasets DIST --protocol dataset-short \
  --variants patch-lift --ttcn-execution chunked --batch-size 64 \
  --device cuda:0 --epochs 3 --patience 3
```

full 保留更多反向传播中间结果，显存需求较高。若发生 CUDA OOM，显式降低 batch size，并在两种执行方式下使用相同的新 batch size 对照；不自动回退或更改任务。GPU 加速幅度需实测，参数量不会反映执行开销。确认资源和耗时后，可改为 `--epochs 100 --patience 10` 正式训练；3 轮仅用于运行效率检查。

## 检查和输出

### BE/NP：12h patch、每小时样本滑窗

在仓库根目录执行，两个数据集依次训练；输出使用新目录：

```bash
git pull --ff-only origin master
RUN_ID=$(date +%Y%m%d_%H%M%S)
for DATASET in NP BE; do
  python -u -m exobase.train \
    --config "configs/${DATASET}.json" \
    --variant patch-lift --patch-encoder ttcn \
    --task-protocol dataset-short \
    --history 604800 --forecast 86400 \
    --patch-seconds 43200 --patch-stride 43200 \
    --sample-stride 3600 \
    --lag-estimator fft --refiner-mode global \
    --split chronological --normalization train \
    --batch-size 64 --epochs 100 --patience 10 \
    --device cuda:0 --threads 2 --seed 1 \
    --save-path "results/rolling_1h_patch12h/${RUN_ID}/${DATASET}/patch-lift" || break
done
```

`--sample-stride 3600` 只控制样本起点；`--patch-stride 43200` 保持内部 patch 不重叠。模型和训练损失不改，保留此前 TTCN 默认 chunked 执行。滚动划分、去重缩放与评估单位见 [预测任务标准](FORECAST_TASKS.md)。样本数量显著增加，每轮耗时会增加，不能按旧版 3 个训练 batch 的耗时估算。没有在本地代跑完整精度训练。

批量入口也支持 `--sample-stride 3600` 并将其传给各训练任务；该入口仍读取配置中的 patch 宽度，**不会自动设为 12h**。上面的直接训练命令完整指定了本轮所需设置。summary.csv 新增 sampling_mode/sample_stride_seconds，避免不同采样实验混淆。

```bash
# 全部数据与计划预检查，不写结果、不训练
python -u scripts/run_all.py --device cuda:0 --variants y-base patch-base patch-lift --dry-run

# 小样本功能检查，不能作精度结果
python -u scripts/run_all.py --datasets BE --device cpu \
  --variants y-base patch-base patch-lift --smoke
```

先统一校验数据、时间单位、记录划分和训练统计，再逐个执行作业。失败写日志并继续，最终返回非零，`--fail-fast` 可立即停止。

默认新建 `results/dataset-short/run_<UTC时间>/`；长任务和旧任务分别在 dataset-long、uniform_24to24 下，避免混表：

- plan.json：窗口、数据 SHA256、记录 ID、seed、全部命令。
- DATASET/variant/：manifest、最佳权重、metrics；LIFT 另有 test_lags.csv。
- DATASET_variant.log：完整日志。
- summary.csv：protocol、历史/预测秒数、patch 秒数、patch 数、标准化指标、持续性基线、best_epoch、任务状态。
- ablation_comparison.csv：patch-lift/patch-base、dag-lift/dag-history 配对差，列明变体。
- component_comparison.csv：y-base→patch-base、patch-base→patch-lift 及可用 DAG 比较。

差值为“variant − reference”，负值表示误差减少；配对双方成功后才写入。`--output-dir` 必须为新目录，不覆盖旧结果。`--smoke` 只验证功能，正式运行不加此参数。
