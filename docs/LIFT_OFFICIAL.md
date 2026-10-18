# LIFT 默认实现与官方版本对齐（2026-10-18）

本次从 master 的 TTCN 更新 `c2ea365` 开始，仅调整 LIFT。`exobase/ttcn.py`、`exobase/data.py`、TTCN 特征维度、patch 编码与查询时间解码保持不变。

## 默认配置

```bash
python -m exobase.train --config configs/BE.json \
  --lag-estimator fft --refiner-mode global --save-path results/BE/lift_official
```

全部数据集预设同步采用 `fft + global`，不再携带旧的 max_lag/min_pairs/min_correlation 设置。三个参数仅在显式选择 `--lag-estimator pearson` 时生效；FFT 的搜索范围固定为官方 efficient/local_max 默认的 `1…L−2`，L 为历史 patch 数。至少需要 3 个历史 patch。

## 官方参照及一致部分

固定参照：[SJTU-DMTai/LIFT@79cf8f1](https://github.com/SJTU-DMTai/LIFT/tree/79cf8f157bb9a616c9733daf1e9c2cfefdc98c94)。vendored 原文件未修改。

- 历史窗口归一化后沿 patch 时间轴做长度 L 的循环 FFT 互相关；不按 lag 重新中心化，不按有效重叠长度重加权。
- 对绝对相关执行相邻点局部峰值筛选，然后每对变量选最大绝对值及 lag，再在 X 候选中选择 top-K；保留负相关符号，索引仍为 X[L+h−lag]。
- 默认没有可配置的相关阈值和最少配对数拒绝规则，也没有额外 max_lag 截断。
- FilterFactory 和 ComplexLinear 类保持官方定义。状态分类器读取完整历史特征，滤波器数量为 2K+1。
- 相关特征沿用 `softmax([1, abs(r)] / temperature)` 后去掉首项；温度直接使用可学习参数，移除旧的 abs/0.05 下界，以保持官方数值行为。
- 三路采用当前官方代码的顺序：A=r_U·FFT(leader)，D=r_Δ·(A−FFT(base))，Q=r_V·FFT(base)，复数混合后逆 FFT，残差加回 base。
- 对齐的 leader 频域输入与官方 no_grad 的 shifted_leader_seq 一样停止梯度；基础预测、历史状态分支和滤波器仍接受训练梯度。共享 TTCN 编码器可从历史基础分支、状态分支及已有辅助任务训练。
- global 在整个预测区间执行一次频域修正，不再默认逐输出截断前缀。

参照是当前官方代码，不是混合论文展示公式：官方差异流已经先过滤 leader，且显式使用残差连接。官方 README 也说明投稿后改过实现。本次没有把这些恢复成与当前官方不同的论文展示式。

## 保留的任务适配

X→Y 源/目标限制、已知未来 X、DAG 骨架以及用户现有 TTCN 向量 patch 都保留。FFT 始终沿时间轴，不把特征维展成时间，也不把向量 patch 改回标量均值。

向量相关先对各特征按观测历史时间中心化，使用单变量联合 RMS 归一化；各特征的目标频谱与源共轭频谱逐特征相乘后取特征均值。因此标量/P=1 情况与官方相同，P>1 为共享 TTCN 基础上的明确扩展，不声称与原始标量 LIFT 完全等价。状态分类器仍读取 L×P 个完整特征，滤波器与混合器在特征间共享。

缺失 patch 采用历史特征均值填充，在归一化尺度等价于 0；保留原时间位置和独立 mask。常量、空通道、观测不足以定义窗口变化或选中 lag 没有任何有效配对时排除，避免将未定义关系送入修正。这是必要的缺失安全边界，不是旧的 min_pairs/min_correlation 筛选。TTCN 历史原始常量通道排除沿用既有实现。

诊断 observed_pairs 在 FFT 模式表示选中 lag 的循环有效配对数，不再是有限窗口非循环配对数；manifest 的 lag_protocol 标记算法、计数口径、完整搜索上界、关闭的阈值和 leader 梯度设置。没有有效 leader 时仍返回基础预测。

## 无效 leader 修复

当有效 leader 少于 K，top-K 的无效占位槽在相关 softmax 中被显式屏蔽；差分分支也显式排除无效槽。某个 leader 在全区间（或可选 prefix 的当前前缀）完全无观测时，同样不产生 `−FFT(base)` 的差分贡献。上层传错 available=True 也不能重新激活 leader_valid=False 的槽。

该修复在完整有效输入下不改变官方计算；只改变官方原任务没有缺失槽位定义的情况。

## 可选消融与旧 checkpoint

```bash
# 非循环 Pearson；这三个筛选参数仅在此模式生效
python -m exobase.train --config configs/BE.json \
  --lag-estimator pearson --max-lag 48 --min-pairs 24 --min-correlation .3 \
  --refiner-mode global --save-path results/BE/lift_pearson

# 在官方估计上单独测试 prefix 约束
python -m exobase.train --config configs/BE.json \
  --lag-estimator fft --refiner-mode prefix --save-path results/BE/lift_prefix
```

global 不提供逐输出的控制时间边界保证：已知的晚时刻未来 X 可能经全区间频率混合影响较早输出。整个未来 X 可知时这不构成未来 Y 泄漏，但不等于物理因果响应边界。prefix 继续作为额外约束供对照。

旧 checkpoint 未保存 lag_estimator 时，预测和重评估入口通过 ModelConfig.from_saved 恢复 pearson，并尊重原 refiner_mode；不把旧权重静默切到 FFT/global。参数形状和 patch 架构兼容。不过无效槽位修复和官方温度/leader 梯度行为属于本次修正，旧实现的边界输出不保证逐位一致。新默认协议应重新训练，不能把旧训练权重直接切换协议来报告收益。

## 数值验证范围

新增测试直接执行固定 upstream 定义，而非只比较迁移代码自身：

- 完整标量输入、奇偶历史长度：所有变量对的相关/lag 及 top-K 来源、顺序、符号与官方一致。
- 完整向量输入：按特征与官方 scalar refiner 对照，覆盖多状态、单状态、正温度、负温度及小正温度；共享历史的受控案例验证多特征分支。
- 一般向量相关另用直接循环平移点积作为独立参考，验证 FFT 方向及缺失配对计数。
- 无效槽、整条 leader 缺失、全部缺失、隐藏 NaN、常量通道、P=1 退化、训练梯度和 checkpoint 重载。

这些验证支持公式与管线正确性，不证明实际数据上的精度收益；TTCN 向量、缺失适配及 X→Y 任务限制仍意味着整个模型不是原论文未经修改的复现。
