**评估口径修复包（2026-10-17）**

本文保留统一指标与旧权重重评估说明。当前默认架构为 [v4：Y-only 主干、patch 与 LIFT](docs/Y_BASE_V4.md)；旧 v2/v3 权重仍按其原保存配置重载。
包内只有 3 个运行代码文件：

- exobase/metrics.py：替换，增加训练集标准化指标、逐变量/全观测点平均和基线。
- exobase/train.py：替换，训练时自动使用统一评估，保存统计量与指标协议。
- exobase/evaluate.py：新增，直接对已有 checkpoint 重新评估。

在仓库根目录解压，覆盖前两个文件并添加第三个文件即可。无需改配置文件或重新生成数据；无需额外安装依赖。未推送 GitHub。

**先重评你已经训练的 BE 模型**

在项目根目录执行：

```bash
python -u -m exobase.evaluate \
  --checkpoint results/BE/dag-lift/best_model.pt \
  --device cuda:1
```

默认生成 results/BE/dag-lift/metrics_unified_test.json，不修改权重，也不覆盖原 metrics.json。结果文件已经存在时，换 --output 路径，或者显式使用 --overwrite。
如果数据文件移动过，增加 --data-path data/processed/BE.npz；程序会核对训练时的数据 SHA256 和样本划分。
如 GPU 显存不足，可加 --batch-size 1，或改 --device cpu。

**以后训练**

原命令仍然可用，建议保存到新目录：

```bash
python -u -m exobase.train --config configs/BE.json --device cuda:1 \
  --save-path results/BE/dag-lift-unified
```

新训练的主 mse/mae 是 train_standard + variable_macro，验证集早停也使用该 MSE。
训练输入、模型结构、MinMax 训练 MSE 和辅助损失保持原实现；训练损失不等于日志中的标准化评估指标。这次统一的是评估，不是重新实现 DAG 全部训练协议。
旧 checkpoint 的最佳轮次仍然是原来选出的轮次；重评不重新选择 checkpoint。结果里的 checkpoint_selection 会标明这一点。

**为什么与 DAG 的评估缩放一致**

DAG/TFB 先对模型预测反归一化，再用仅训练数据拟合的 StandardScaler 计算 mse_norm/mae_norm。
这里也对每个 Y 使用训练集观测值的均值与总体标准差（ddof=0），不使用验证/测试集，不把缺失占位零当成观测，不对 patch 均值拟合统计量。常量特征的 scale 取 1，与 StandardScaler 的约定一致。

当前模型输出是 MinMax 值。设每个目标的模型缩放范围为 R_j，训练标准差为 sigma_j，则标准化误差是：

    e_standard,j = (prediction_minmax,j - target_minmax,j) * R_j / sigma_j

这与先恢复原量纲、再分别做 Z-score 后相减完全等价；均值在误差相减时抵消。必须先逐变量转换，不能对多变量最终总分乘一个统一缩放系数。
模型内部使用 MinMax 不影响最终评估使用 DAG 的 Z-score 尺度。

**求和顺序**

在全部评估 batch 上，先为每个目标累积有效误差和 S_j 与观测数 N_j，最后再除，不对 batch 指标简单平均。

    macro MSE = mean_j(S_j / N_j)
    micro MSE = sum_j(S_j) / sum_j(N_j)

MAE 同理。N_j=0 的目标不进入 macro，逐目标结果写 null。
默认 macro 保持 t-PatchGNN 的逐变量等权原则；micro 则按每个真实观测点等权。两种结果都明确输出。

观测完整、固定长度且各变量观测数相同：macro = micro = DAG 的全元素 MSE/MAE，也等于 DAG 各固定长度窗口 MSE/MAE 的平均。
不规则观测数不等：两者可能不同；固定窗口上先平均、再对窗口平均也可能不同，因为每个窗口观测数不等。
只有一个 Y 时：macro 与 micro 必然相同。因此你这次 BE 的大幅差别主要来自 MinMax/Z-score，不是逐变量求和顺序。
不能使用 mean(mask * error)：这会把缺失和 padding 计入分母，导致误差虚低。

RMSE 另需区分：这里是 sqrt(最终 MSE)，符合 t-PatchGNN。DAG 的 rolling_forecast 对各窗口 RMSE 再平均；开平方与平均不交换，即使完整规则数据也可能不同。因此本补丁与 DAG 对齐的是 MSE/MAE 的缩放和完整数据时的平均，不声称所有 RMSE 聚合完全一致。
原带符号的 MinMax MAPE 不作为新评估指标。

**结果读取**

- test.mse / mae / rmse：训练集 Z-score，逐变量等权。
- test.by_scale.standard.macro / micro：标准化尺度，两种平均方式。
- test.by_scale.raw.macro / micro：原量纲。
- test.by_scale.model_minmax.macro / micro：模型原 MinMax 尺度，可对照旧日志。
- 每个尺度中的 per_target：各目标分别计算的指标。
- test.observed_count_by_target：真实参与评分的观测数。
- evaluation_protocol.statistics：训练均值、标准差、实际使用的 scale 与观测数。
- test.baselines.persistence / train_mean：同一测试 mask 上的历史最后观测保持预测与训练均值预测。
- persistence 某目标没有历史观测时回退到该目标训练均值，并报告回退记录数。
- loss 是为旧接口保留的 mse 别名，不表示包含辅助损失的训练总损失。

**本次核对的源码**

- t-PatchGNN，lib/utils.py:314（MinMax）、lib/parse_datasets.py:38–50（seen_data 拟合）、lib/evaluation.py:256 和 :335（逐变量累计后平均）：
  https://github.com/usail-hkust/t-PatchGNN/blob/00c94e7bbaf21c71b03ed84ff690ae59e37129e5/lib/evaluation.py#L256
- DAG，ts_benchmark/evaluation/strategy/forecasting.py:82（仅训练集拟合 StandardScaler）：
  https://github.com/Capybaring/dag/blob/f4468901331f53bb84bbc405219150df997e26c5/ts_benchmark/evaluation/strategy/forecasting.py#L82
- DAG，ts_benchmark/evaluation/metrics/regression_metrics.py:128–155（mse_norm、mae_norm）：
  https://github.com/Capybaring/dag/blob/f4468901331f53bb84bbc405219150df997e26c5/ts_benchmark/evaluation/metrics/regression_metrics.py#L128
- DAG，ts_benchmark/evaluation/strategy/rolling_forecast.py:426–433（逐窗口算指标再平均）：
  https://github.com/Capybaring/dag/blob/f4468901331f53bb84bbc405219150df997e26c5/ts_benchmark/evaluation/strategy/rolling_forecast.py#L426
- DAG，ts_benchmark/baselines/deep_forecasting_model_base.py:848（预测恢复原量纲）：
  https://github.com/Capybaring/dag/blob/f4468901331f53bb84bbc405219150df997e26c5/ts_benchmark/baselines/deep_forecasting_model_base.py#L848

实际执行了上述 DAG 的 mse_norm/mae_norm 源码，与新评估在完整多变量样例上的结果一致至 float64 舍入误差；另验证了缺失数量不同、batch 拆分、padding、常量变量、训练统计不受验证/测试变化影响，以及旧 checkpoint 的加载和预测不变。

**你提供的 BE 结果**

本地数据 SHA256 与你的 manifest 一致；当前划分有 163/54/55 个训练/验证/测试 record。
OT 的训练观测均值 46.3912678279，标准差 26.2533528973，观测数 21903；测试观测数 932。
用你提供的 MinMax 汇总指标和 R=3199 换算：

    standard MSE ≈ 0.430763
    standard MAE ≈ 0.498536
    standard RMSE ≈ 0.656325

这是对你日志的换算，不是重新运行你远程训练好的模型；请使用上面的重评命令得到实际结果。

最后，尺度一致不等于与论文表格的完整实验条件一致：当前仍使用不规则 mask、独立 record、当前训练划分；DAG 原基准是规则序列上的滚动窗口。训练样本范围不同也会导致拟合出的标准差不同。不能仅凭尺度对齐就把新分数与论文表格当作严格同设置比较。
