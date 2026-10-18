# 架构 v3：完整 TTCN patch → 向量 DAG → 向量 LIFT

新训练默认 `--patch-encoder ttcn`，manifest 的 architecture_version=3。

## Patch 内编码

输入仍为 data.py 打包的 [B,K,观测槽数,C] 值、实际时间和独立 mask。每个观测先组成 [value; LearnableTE(time)]，官方 TTCN 的三层动态滤波器为每个输出通道和输入特征生成逐观测 softmax 权重。聚合输出 d_model-1 个特征，再追加 patch 非空标志。非空方程与 upstream/tpatchgnn 的 LearnableTE/TTCN 对齐；全空 patch 显式清零。

不先求 patch 标量均值、不把观测数当时间、不实例化 GNN。TTCN 仍是有损的可学习聚合，不意味着无损保留所有原始信息。历史与未来 X 共享同一个编码器，未来 X 的每个 patch 独立编码。

## DAG 接口

时间分支在每个变量的 patch token 之间做注意力。变量分支把完整 [时间×特征] 编码成外生变量 token，再做变量注意力，保留 DAG 的迁移注意力、双 softmax 与上下文门控思想。向量历史归一化使用有效 patch 上的逐特征均值和单变量联合 RMS；这不是对 patch 原始观测求均值。

dag-history/dag-lift 的基础预测器只读取历史。时间分支预测未来 X 的完整特征序列，供变量分支使用；dag 变体可直接使用已知未来 X token。时间分支和变量分支各输出未来 Y 特征，按 alpha 融合。

## 滞后估计与 LIFT

默认 `--lag-estimator fft`：对观测历史逐特征中心化、按变量联合 RMS 标准化，沿 patch 时间轴执行长度 L 的循环 FFT，汇总目标与源共轭频谱的逐特征乘积，按官方局部峰值规则在 1…L−2 选择 lag，再按绝对相关 top-K。单特征完整输入与官方标量估计器数值一致；多特征是共享 TTCN 表征上的扩展，不改 patch 表示。

默认不使用相关阈值、最少配对数或 max_lag 截断。缺失 patch 在归一化尺度填 0，空/常量等未定义通道排除。`--lag-estimator pearson` 保留原逐候选重新中心化的非循环实现，max_lag/min_pairs/min_correlation 仅对此模式生效。历史原始值完全不变的变量仍额外排除，避免仅时间编码产生 leader。lag 不反向传播，不能直接当作物理因果效应。

按照 X[L+h-lag] 从历史或已知未来 X token 取完整特征向量。历史统计量用于标准化，负相关翻转标准化偏差方向。LIFT 的状态分类器读取完整历史特征；状态概率和相关特征生成 2K+1 组频域滤波器。对每个特征通道沿 patch 时间轴执行三路 FFT 滤波/差异/自身频谱混合，保留向量维；频率滤波器和复数混合层在特征通道间共享。单特征输入退化为 v2 的公式。

默认 global 沿用官方全区间频率混合，leader 支路停止梯度，温度直接使用可学习参数；无效槽位显式屏蔽。prefix 仅作为可选时间约束：输出 h 使用 n<=h 的对齐前缀，逆 FFT 后取对角位置。无有效输入时回退到基础 token。global 不提供 prefix 的 patch 级时间边界保证。详细边界见 [LIFT 官方对齐说明](LIFT_OFFICIAL.md)。

## 解码、监督与评估

最终 [B,Y,H,d_model-1] token 按真实查询所属 patch 取出，追加预测有效标记 1，与该查询的 LearnableTE 拼接，经 MLP 输出 MinMax 标量。与官方时间条件解码思想一致，同 patch 的不同查询不再强制相同预测。

主损失仍是原始未来 PV 有效查询的 MinMax variable-macro MSE。两个辅助任务也经查询解码器分别监督原始未来 X 和历史 Y；不使用会随编码器变化的 token 作为监督标签，避免自监督目标随编码器共同塌缩。未来 Y 不作为预测输入。

评估保持统一版本：训练集原始观测的 StandardScaler(ddof=0)，有效查询 mask、全 batch 累计后逐变量 macro；同时导出 micro/raw/model_minmax 和持久性基线。训练参数量和优化问题已变化，需要重新训练，不能从旧精度表推断新模型效果。

## 兼容与内存

旧 checkpoint 的 model_config 没有 patch_encoder 字段，ModelConfig 保留 mean 的反序列化默认值；训练入口显式选择 ttcn，因此旧权重仍装入 v2，新权重装入 v3。手工构造 v3 ModelConfig 时也要明确 patch_encoder='ttcn'。`--patch-encoder mean` 可显式运行旧对照，正式新模型保持默认 ttcn。

TTCN 每组 16 个单变量 patch 编码，并在训练中分块重算；原始观测辅助解码、预测查询也分块重算，避免为 DIST 长 patch 同时保留全部动态滤波器和点解码激活。数据文件与 data.py 不变，不需要重新生成 NPZ。
