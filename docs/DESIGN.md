# 架构 v2：DAG 标量均值 Patch + 外生领先关系

本文保留 v2 的公式以解释旧 checkpoint。当前新训练默认使用 TTCN 向量 patch，见 [架构 v3](TTCN_V3.md)；不应将下文的均值汇总理解为当前默认实现。

## 1. 任务与 Patch 接口

X 是已知未来的控制/协变量，Y 是待预测过程量。起点 T 后不读取未来 Y；候选方向限定为 X_i(s)→Y_j(s+δ_ji)，δ>0。

固定真实时间窗口 I_p=[pΔ,(p+1)Δ)，定义：

\[
v_{c,p}=\frac{\sum_{t\in I_p}m_c(t)v_c(t)}{\max(1,\sum_{t\in I_p}m_c(t))},\quad
M_{c,p}=\mathbf1[\sum_{t\in I_p}m_c(t)>0].
\]

exobase/data.py 的 pack_patches 负责和 t-PatchGNN 一样的物理时间归属；model.py 的 scalar_patch_means 形成规则标量接口。当前保留分块思想，不使用 TTCN/GNN。块内均值丢弃部分细节，是明确的 baseline 简化；全空块保留位置和 mask。

DAG 输入空单元用该变量该记录历史有效块均值填充，全空历史变量填 0。未来 X 空单元用历史 X 均值。填充仅为适配稠密接口，不成为滞后估计样本。

## 2. DAG 双分支

代码：exobase/dag_model.py、exobase/dag_layers/。保留原 TemporalCausalityEncoder、CovCausalityEncoder、投影器、注意力迁移/门控、归一化和组合：

\[
\widehat Y^{base}=\alpha\widehat Y^{temporal}+(1-\alpha)\widehat Y^{covariate}.
\]

时间分支分别处理 X/Y 历史；外生编码器在 X 上学习预测任务，其注意力参数用于目标侧时间依赖建模。其 X 预测值仅依赖 X 历史。变量分支保留外生历史重构 Y、外生未来预测 Y 的机制。Y 历史统计量可条件化目标预测，不构成一个 Y→X 预测分支。

历史长度 L、预测长度 H 均以块计。原 DAG 时间 PatchEmbedding 设 patch_len=stride=1，仅投影已有标量块，不再合并多块；其原有最终边界复制 token 保留。滞后估计只使用 L 个真实时间块，不包含复制 token。

| 变体 | 变量分支未来输入 | 修正 |
|---|---|---|
| dag-history | 时间分支预测的 X | 无 |
| dag | 已知未来 X | 无 |
| dag-lift | 时间分支预测的 X | 已知 X 经正时延对齐后进入 LIFT |

对 dag-history/dag-lift，预测 X 路径在训练和推理一致，基础预测不依赖实际未来 X，避免未对齐输入绕过滞后分支。dag 保留直接未来 X 的原方法对照。

未来 X 可监督外生预测任务，历史 Y 监督变量分支重构，各自使用真实 mask：

\[
\mathcal L=\operatorname{MaskedMSE}_{query}(\widehat Y,Y)
+\beta[\operatorname{MaskedMSE}(\widehat X^{temporal},X_{future})
+\operatorname{MaskedMSE}(\widehat Y^{covariate}_{history},Y_{history})].
\]

不将 Y 作为外生预测输入，不在 [Y,X] 上学习双向动态图。未来 Y 及标签 mask 不进入 forward。

## 3. X→Y 领先估计

默认使用 `exobase/lead_estimator.py`，与固定官方 efficient/local_max 版本一致的循环 FFT 互相关、局部峰值与绝对相关 Top-K。目标/来源限定为 Y/X，搜索 lag 为 1…L−2，不使用额外 max_lag、min_pairs 或 min_correlation 筛选。完整标量输入与官方数值一致。

现有 TTCN 向量接口保留：逐特征按观测历史中心化，使用单变量联合 RMS，逐特征目标频谱×来源共轭频谱取均值，再 inverse FFT/L。缺失 patch 以归一化 0 表示；空/常量和完全无有效配对的未定义关系排除。配对数按循环位置计数，不对相关重新加权。它是共享 TTCN 表征上的扩展，不能解释成物理因果边。

原非循环、逐 lag 重中心化的 mask-aware Pearson 通过 `--lag-estimator pearson` 显式选择，此时 max_lag/min_pairs/min_correlation 生效。估计器始终无参数，未来值不进入 lag 估计。详见 [LIFT 官方对齐](LIFT_OFFICIAL.md)。

## 4. 已知未来 X 对齐

未来目标块 h=0,...,H-1，取：

\[
Z_{ji,h}=\operatorname{sign}(\rho_{ji})\widetilde X_{i,L+h-\widehat\delta_{ji}}.
\]

将 X 历史与已知未来拼接，使用每个 X 的历史均值/标准差归一化；缺失位置在归一化尺度填 0，保留可用性 mask。索引<L 使用历史，否则使用已知未来 X，无需先预测这些已知控制量。

例如 δ=3 时，未来 X 第 5 块改变，最早影响未来 Y 第 8 块（从 0 计数）。该时间边界通过自动测试。

## 5. 官方 global 频域修正及可选 prefix

保留官方 FilterFactory/ComplexLinear。状态概率来自 Y 历史和目标状态参数；相关强度驱动 2K+1 组状态相关滤波器。三路信号沿用官方实现：

\[
U_k=\mathcal F(Z_{jk})\odot r_{U,k},\quad
D_k=(U_k-\mathcal F(\widehat Y_j^{base}))\odot r_{D,k},\quad
V=\mathcal F(\widehat Y_j^{base})\odot r_V,
\]

\[
\widehat Y_j=\widehat Y_j^{base}
+\mathcal F^{-1}(W_{complex}[\sum_kU_k,\sum_kD_k,V]+b).
\]

在 Y 历史归一化尺度计算，再将残差变回 Y 尺度。基础预测和修正共同训练，温度按官方直接使用可学习参数；对齐 leader 支路停止梯度。无效/完全不可用槽位从相关特征和差分聚合中排除。

默认 global 对完整预测区间频域混合，可能让晚时刻已知控制影响较早输出。可选 prefix 模式对每个 h 独立构造：

\[
Z^{(h)}_{ji,u}=Z_{ji,u}\mathbf1[u\le h].
\]

用相同滤波器及完整基础预测运行修正，仅取第 h 个输出。基础预测和滤波器仅依赖历史，因此该输出不依赖 X_i 在 L+h-δ_ji 之后的实际未来数值。过去控制对多个后续时刻的持续影响仍可学习；前缀无可用领先输入时残差为 0。

prefix 是本场景的额外时间约束，不是原始 LIFT。默认 global 模式省略 prefix，不作逐输出时间边界保证。prefix 的控制中间量约为 O(B·Y·K·H²)，长预测区间需注意计算规模。

## 6. 查询与解释边界

块预测映射回原始未来查询，每块内预测相同。主损失在真实观测的 Y 查询计算，保留不规则评估；不同块观测数不同，因此不是每块等权损失。

默认 chronological + train-only scaling，随机独立记录/seen 缩放仍可选。完整记录不重叠，跳过无历史、无未来标签及不完整尾段并报告。

已知未来 X 是任务条件。闭环控制量未必统计外生，这里施加 X→Y 的预测结构先验，并未完成干预因果识别。时延尺度为 Δ；块内细节、变化工况、非线性与多控制量混杂仍需专门验证。
