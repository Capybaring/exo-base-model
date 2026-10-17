# v0.1 历史设计与验证（仅供旧版本对照）

以下描述不适用于当前默认 dag-lift。旧 checkpoint 仍可重载，旧变体需显式选择。

# 方法和适配边界

## 结构

```text
长表观测 (time, variable, value)
  -> 时间并集 + [Y,X] 值/mask
  -> 非重叠独立记录 + 真实预测查询
  -> 真实时间区间 PatchBuilder
  -> t-PatchGNN LearnableTE + TTCN
  -> t-PatchGNN 时间 Transformer + 时间自适应图 + 图卷积
  -> 历史状态 + 查询时间的原版 MLP 解码器
       ^
       | exo/delay: 外生上下文加到目标历史状态
       |
  已知未来 X -> 共享 TTCN -> X-to-Y 通道注意力
                               ^
                               | delay: Key += 检索到的模式表示
  X 的固定时间 Patch 均值窗口 -> 训练集 k-Shape 模式库
```

`TPatchBackbone` 封装历史编码；`PatternMemory` 封装 DFT；`ExogenousTPatchGNN` 连接主干与融合。未来可替换历史编码或融合方式，保持 `exobase.data` 的观测/时间/mask/查询契约。当前 X→Y 注意力沿用此前外生预测的思路作为临时 baseline，不宣称已经完成最优主干设计。

## t-PatchGNN 的保留部分

Patch 归属由真实时间范围确定，同一个时间区间每变量可以有不同数量的观测。输出 `[B,K,L,C]`，L 是 batch 内每块每变量最大有效观测数，padding 不算观测。TE 为线性分量与可学习正弦分量；TTCN 沿块内观测维生成、归一化滤波权重，附加块有效标志。

历史主干直接基于官方类：每层时间 Transformer -> 动态节点嵌入与邻接学习 -> GNN，保留后续层的残差、Linear 时间聚合和时间查询解码器。`upstream_model.py` 只删除未使用的可选导入，取消节点参数初始化时的强制 `.cuda()`。子类 TTCN 增加隐藏 NaN 清除、空块清零和安全归一化；其它历史计算沿用官方实现。

原版的 Transformer/GNN 可以保留和建模空时间块，空块不会从序列中删除。这里没有新增强制的自回归因果 Transformer mask；历史编码器只接收预测起点前的 Y/X。

默认 patch stride=width。非 delay 模式支持重叠时间 Patch。为了让 DFT 的每个窗口偏移对应同一物理时间步，第一版 delay 要求非重叠且 history 边界对齐 Patch 网格。

## 外生融合（当前的独立设计选择）

令历史主干输出每变量状态 s，已知未来 X 的对应 Patch token 为 H。对预测查询 h，选最新起点且覆盖该查询的未来 Patch（最后一块做边界截断）。

```text
u_i,h = H_i,patch(h) + W_history s_Xi + e_Xi
q_j,h = Wq (s_Yj + W_time TE(t_h))
k_i,h = Wk u_i,h
v_i,h = Wv u_i,h
alpha_j,h,i = masked_softmax_i(q_j,h dot k_i,h / sqrt(d))
context_j,h = sum_i alpha_j,h,i v_i,h
prediction_j,h = upstream_decoder([s_Yj + context_j,h, TE(t_h)])
```

当前 future X encoder 与历史主干共享 TTCN。历史 Y/X 共同进入动态图，不强制单向图。仅 tpatch-y 禁用历史 X，tpatch 禁用未来 X，exo 与 delay 假设未来 X 在预测时已知；仍保留其原始观测缺失 mask。若某查询对应未来 X 全空，外生上下文为零。

这不是原 DAG 的双相关网络，也不声称是完整 t-PatchGNN/PDFormer 原模型；它是可替换的外生融合 baseline。

## PDFormer 式 DFT 的迁移

保留原方法的三个核心步骤：训练窗口聚类得到模式库、当前窗口对模式库的注意力检索、检索结果只加到 Key。没有使用上一版的“窗口 token 展平 -> MLP -> 同时加 Key/Value”。

原 PDFormer 窗口是规则采样原始值。本实现的**明确改编**是：每个固定真实时间 Patch 对有效 X 值求均值，得到标量序列；连续 S 个标量构成窗口，包含当前块。主路径的 TTCN 仍使用全部原始观测与时间，不插值。全空块保持在原位置，任何缺少一块的窗口不做模式检索，不跨空缺拼接。忽略恒定候选窗口，训练与推理均对窗口做 z-normalization。多个外生变量共享模式库，候选窗口来自所有训练记录的 X。

训练集候选窗口使用确定性的 reservoir sampling 限制数量，k-Shape 聚类得到固定标量模式；这些模式随模型保存为 buffer，投影参数与模型联合训练。窗口是数据统计量，不随隐藏编码器坐标改变，避免“用训练前 token 聚类、训练后坐标漂移”的问题。若没有完整非恒定候选窗口会明确报错，不能静默伪造模式库。

注意力检索的 Query 来自当前 X 的窗口，Key/Value 来自模式库。检索结果加到 X→Y 注意力的 Key，Value 保持基础定义。模式库检索中的 Value 与 X→Y 注意力的 Value 是两套变量。

DFT 窗口跨越历史/已知未来 X 的边界。它提供滞后相关的窗口特征，不输出真实成对时延，也不提供因果识别保证。同一未来 Patch 内可使用晚于当前 Y 查询的 X，因为该分支明确假设整个未来 X 已知。若研究严格的时间点滞后，需要另加查询级时间约束，当前版本没有宣称实现这一目标。

## 数据协议

连续工业流不是天然独立个体数据；这里显式切成非重叠完整记录，记录切分方式属于本项目的任务适配。随机独立记录协议可用于与上游 seen/test 流程对应，chronological 则用于时间外推。不要混用两种划分的结果。

长表的重复同变量同时间取均值，真实零值保留 mask=True。日期时间保留到整数纳秒后减统一原点，避免 pandas 不同时间精度导致单位错误；数值时间按相对秒解释，不压缩间隔。训练/验证/测试都在有 Y 标签的位置监督评估，未来 Y 及其 mask 不被模型 forward 读取。

BE/NP 默认按每个变量观测独立删除 30%，不是删除相同的整行时刻。相同 seed 在相同排列的源文件上产生相同随机序列，方便复现；不同缺失实验必须保存其 preparation manifest。DIST 原有不规则性不再叠加随机删除。


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
