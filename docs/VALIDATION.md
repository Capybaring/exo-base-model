# 验证记录（2026-10-17）

环境：macOS ARM64、Python 3.12.14、CPU。实际依赖版本保存在 `requirements-tested.txt`。使用用户提供的本地文件，原始文件未修改，数据和权重没有提交到 Git。

## 数据准备

```bash
python -m exobase.prepare --dist /path/to/DIST_irregular_9x9.csv \
  --dataset-dir /path/to/dataset --drop-rate 0.3 --seed 2021
```

| 数据 | 时间并集 × 变量 | 有效独立记录 | 默认 train / val / test |
|---|---|---|---|
| DIST | 1,275,114 × 18 | 302 | 180 / 61 / 61 |
| BE 丢弃后 | 见本地 preparation manifest | 272 | 162 / 55 / 55 |
| NP 丢弃后 | 见本地 preparation manifest | 272 | 162 / 55 / 55 |

DIST 同变量同时间合并 483 条重复记录，保留原始时间原点和间隔；50 条候选记录因没有历史或未来 Y 标签跳过。尾部不足完整记录的部分不参与。

BE、NP 原来各变量均为 52,416 条观测。seed=2021、drop_rate=0.3 后，两份文件按其源行排列分别得到：两个 X 保留 36,759、36,538 条，Y/OT 保留 36,701 条。实际删除率约 29.87%、30.29%、29.98%。不同变量不是同步删掉整时刻。

## 自动检查

`python -m pytest -q`：**16 passed**。

- 实际时间区间分块，真实零值与空 Patch、时间缺口不被压缩。
- 长表日期转秒、异步 mask、重复均值。
- 与固定官方提交的 `split_and_patch_batch` 对照 Patch 数据、时间和 mask。
- 相同权重下，历史 t-PatchGNN 模式与上游类的有效非空输入输出对照。
- 与官方 `compute_error` 对照 MSE、MAE、MAPE 兼容分支。
- 四种变体的梯度、隐藏 NaN 不变性、future Y 与标签 mask 不进入预测。
- 历史模式不使用未来 X；exo 模式确实读取已知未来 X，未来 X 全缺时保持有限输出。
- 修改验证/测试数值不改变训练模式库；数据划分无记录交集。
- 指标按全局误差/计数累计，不平均 batch 均值。
- 固定种子删除可复现，源文件内容保持不变。
- 缺少时间块时 DFT 窗口不跨缺口拼接。

另外通过 Python 编译检查、Shell 语法检查、可编辑安装、三个 CLI 入口和 `pip check`。

## 真实数据小规模训练

下表仅证明训练—验证—测试管线可运行。限制记录数量会改变划分与缩放；不是完整数据精度，不用于比较模型优劣。

| 数据 / 模式 | 记录上限 | d_model | epoch | 最优验证 MSE | 测试 MSE |
|---|---:|---:|---:|---:|---:|
| DIST / delay | 8 | 8 | 2 | 0.077609 | 0.073292 |
| DIST / delay | 8 | 64 | 1 | 0.096308 | 0.068181 |
| BE / delay | 12 | 8 | 2 | 0.008123 | 0.023132 |
| NP / delay | 12 | 8 | 2 | 0.023490 | 0.021297 |

小模型使用 `te_dim=3, n_heads=2, n_patterns=3, max_pattern_windows=200`；DIST 的 64 维检查使用默认 te_dim/heads、4 个模式、300 个候选窗口。

BE 上的 tpatch-y、tpatch、exo 三个变体另各完成一轮小规模训练/验证/测试。保存后的 DIST checkpoint 已重载，并导出 14,292 条原量纲预测，包含真实相对秒和标签观测标志。

## 尚未验证

- 没有运行完整数据、多随机种子的正式训练。
- CUDA/MPS 尚未实测。
- 当前是 PDFormer 的 Patch 均值描述符迁移，未宣称与原始规则交通输入 DFT 数值等价。
- 已知未来 X 的实际可获得性与因果时延解释需要由后续实验设定和验证。
