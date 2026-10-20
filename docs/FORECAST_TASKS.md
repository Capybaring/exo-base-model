# 预测任务标准与 DAG 窗长对齐

## 2026-10-20：可选的每小时滑窗

新增 `exobase.train --sample-stride 3600`：样本起点每 1 小时移动，保持原始时间、观测值、缺失掩码与已知未来 X；不改变模型结构、TTCN、LIFT、训练损失或评估误差公式。`sample-stride` 是样本起点间距，独立于 `patch-seconds` 与 `patch-stride`。BE/NP 当前运行固定为历史 168h→未来 24h，patch 宽度和内部步长均为 12h（43200 秒），即 14→2 个 patch。

滚动模式先在完整物理时间范围按既有 60/20/20 比例确定边界，再构造/分配样本。每个样本的整个未来区间必须落入一个分区；跨界的未来区间排除。验证/测试可以使用边界之前的历史上下文，未来 Y 标签时间不跨集合重叠。禁止滚动样本使用随机划分。这里仅改为滑窗，不同时改为论文的 70/10/20。

小时网格覆盖范围包含最后观测所在的 1h 区间，区间均左闭右开，不插值、不压缩缺失时刻。当前 BE/NP 各 52,416h：52,225 个候选窗口中排除 46 个跨界窗口，train/val/test 为 31,258/10,460/10,461。相邻样本高度重叠，这些数量不代表同等数量的独立序列。

训练 MinMax 和评估 Z-score 只根据训练样本覆盖的原始观测拟合；重叠窗口的同一观测对统计只计一次。预测指标仍对每个“预测起点、提前量、目标变量”的有效预测计分；同一时间戳由不同起点预测属于不同预测任务。保存的 manifest 包含样本步长、物理边界和跨界排除数量，重载评估/预测自动恢复，预测 CSV 使用真实绝对时间。

不提供 `--sample-stride` 时保留之前的不重叠构造，旧 checkpoint 继续按原构造重载；以下旧版记录数量和运行示例描述的是该兼容模式。DIST 不自动改为滑窗。滚动模式的 `--max-records` 小样本检查分别从三个物理分区取样，不能用来判断完整训练精度。

依据：用户提供的 arXiv:2509.14933v3，正文 5.1.3（第 8 页）、附录 A.2（第 14–15 页）、Table 1（第 7 页）、Table 5（第 17 页），以及[官方已知未来 X 脚本](https://github.com/decisionintelligence/DAG/blob/main/scripts/w_future/DAG.sh)。

## 论文中 BE/NP 并非 24→24

| 任务 | 历史长度 | 预测长度 | BE/NP 的物理时间 |
|---|---:|---:|---|
| 短期 | 168 原生步 | 24 原生步 | 7 天→1 天 |
| 长期 | 720 原生步 | 360 原生步 | 30 天→15 天 |
| Colbun/Rapel 短期 | 60 步 | 10 步 | 60 天→10 天 |
| Colbun/Rapel 长期 | 180 步 | 30 步 | 180 天→30 天 |

BE/NP 采样间隔为 1 小时。168/24 是观测时间步数，不是 patch 数，也不是 batch size。官方原 DAG 的 patch_len 超参数与本仓库 TTCN 的真实时间分块是不同接口，不机械地令二者相等。

以前 configs/BE.json、configs/NP.json 本来已是 168→24 小时；旧 run_all.py 强制覆盖为统一 24→24，导致单数据集预设与批量入口任务不同。本次默认修正为读取数据集预设。

Table 2 是短/长两个预测任务的平均值；已知未来 X 的单独短期结果在 Table 5，不应拿 Table 2 Avg 与单个 24 小时预测结果比较。论文 Table 3 是不提供未来 X 的另一个实验，不作为本研究的必要消融；本仓库 table3_irregular 仅是既有数据包目录名称。

## DIST 初始任务

DIST 不在 DAG 的 12 个 TSF-X 数据集中。当前初始任务设为：

**历史 2 小时（7200 秒）→未来 30 分钟（1800 秒），TTCN patch=1 分钟（60 秒），120→30 个 patch。**

理由：DIST 的原始观测间隔中位数通常为 5–10 秒，首个训练窗口每个非空分钟 patch 的观测数中位数为 11。分钟网格可聚合局部观测，同时把 lag 分辨率从 1 小时细化到 1 分钟。保留全部原始时间点查询、mask 和已知未来 X，不重采样成一分钟标量预测，也不修改 TTCN 聚合公式。

仅依据时间尺度、窗口数量和计算规模选择，未使用测试集模型精度挑选：旧 24h→24h 任务只有 41 个可用记录（24/8/9）；新任务有 716 个（429/143/144）。这表示不同预测任务，不应把旧表与新表当成同条件精度比较。此初始任务不是已验证的最优工业预测跨度，真实业务若要求更长提前量，应先固定业务任务再训练。

## 明确的对齐范围

本次对齐的是**历史与预测窗长、时间单位和任务记录**。现有不规则观测实验仍保留：

- 数据准备的原始不规则性/既有缺失模式；无新增删点或插值。
- 独立的不重叠记录，chronological 约 60/20/20 划分。
- MinMax 模型输入与 masked MSE 训练，训练集标准差归一化评估。
- 当前 TTCN/向量 DAG/LIFT 适配及每个模型的结构。

论文 Table 1 对这些数据使用 70/10/20；[官方 TFB 配置](https://github.com/decisionintelligence/DAG/blob/main/config/rolling_forecast_config.json)采用滚动预测、stride=1；附录 A.2 和官方脚本使用 L1/MAE 训练。完整观测原论文与本仓库不规则/masked 任务也有区别。因此本次不能称为完整复现 DAG 论文指标，更不能仅凭改了 168→24 就直接比较绝对分数。

BE/NP 新任务在本地数据上为 272 个独立记录（163/54/55），少于旧任务的 1091 个；这是独立记录构造造成的，不是论文滚动窗口数量。改变窗长也会改变标签时间点和缩放统计，所有主要方法必须在同一新任务下重训。

## 可复现入口

```bash
# 三个数据集的短期主对照，保持实际已知未来 X
python -u scripts/run_all.py --datasets DIST BE NP \
  --protocol dataset-short --variants patch-lift dag --device cuda:0

# 两个电力数据的论文长期窗长
python -u scripts/run_all.py --datasets BE NP \
  --protocol dataset-long --variants patch-lift dag --device cuda:0

# 原 24→24 任务复查，必须显式声明 uniform
python -u scripts/run_all.py --datasets DIST BE NP \
  --protocol uniform --history-steps 24 --forecast-steps 24 \
  --variants patch-lift dag --device cuda:0
```

单独运行 DIST 可使用 `python -m exobase.train --config configs/DIST.json`；BE/NP 的同类命令使用各自 168h→24h 预设。命令行仍可覆盖秒数，但应使用新输出路径并记录新任务。

manifest 的 forecast_task、plan.json 和 summary.csv 记录 protocol、实际秒数及 patch 数；不同协议默认使用不同结果目录。这里的“数据集预设”是在为不同时间尺度定义任务，不是移除未来外生信息的消融。
