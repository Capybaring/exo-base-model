# v4：历史 Y 时间基础预测 + 分组 TTCN + LIFT

## 任务分工

```text
历史 Y → E_Y → Y-only 时间注意力 → 未来基础 token ───────┐
                                                        ├→ LIFT → 查询时间解码 → Y
历史 X + 已知未来 X → E_X → lag 对齐 → X→Y 线性映射 ───┘
历史 X、Y 的观测标量 patch 均值 → lag/leader ────────────┘
```

无 X→X 未来预测、X→历史 Y 重构、DAG 注意力迁移/上下文门控/变量分支或对应辅助损失。

## 基础主干

- y-base：同网格历史 Y 的观测均值及时间均值，线性输入投影，无 TTCN 动态点聚合。
- patch-base：E_Y 使用原样保留的 TTCNPatchEncoder，完整 d_model−1 特征加非空标记进入时间主干。
- patch-lift：相同 Y 主干，另有独立 E_X、X→Y 映射和 LIFT。

时间主干合并 batch 与 Y 变量轴，对每个 Y 独立做历史时间注意力，参数在 Y 间共享。缺失位置保留但不作为 key；全空历史保留一个安全占位 key。读出用共享 `Linear(L,H)` 再映射未来特征，不使用原 DAG 的 `Linear(L×D,H×P)` 大头。

Y=1、X=2、L=H=24、D=64、单层/4头/8状态下：y-base 64,608 参数；patch-base 113,055；patch-lift 180,960；原 patch+DAG 5,332,276。这是更简单的基础网络，不是仅关闭 DAG 的单一开关。

先初始化全部 Y 模块，再初始化可选 X 模块，保证同 seed 的 patch-base/patch-lift 初始 Y 参数相同。所有 X 变量、历史与未来 X 共用 E_X；Y 变量共用 E_Y；E_X/E_Y 参数独立。TTCN 方程与 data.py 的真实时间分块保持不变。

## Lag 与特征修正

独立编码器之间没有天然逐维对应关系，所以 lag 只读取历史 patch 标量均值与非空 mask；此摘要不替代 TTCN 主路径。不读取未来 X/Y、时间编码或模型参数，缺失块不插值或压缩。

默认保留循环 FFT/local peaks 的正 lag 搜索 1…L−2；Pearson 为显式选项。空/常量通道无效。正 lag 不包含同时刻关系，尚未增加零 lag。

用历史 X token 的逐特征均值与联合 RMS 标准化，按 `L+h−lag` 从历史/已知未来取特征，保持相关符号对齐先验。`Linear(P,P,bias=False)` 将对齐 X 映射到 Y 修正坐标，由最终 Y 损失训练。这是可训练特征融合，不宣称 latent 各维具有物理量含义。

复用 LIFT 的历史状态估计、2K+1 滤波器、leader/差分/基础三路、复数混合与残差。FFT 沿 H 轴，滤波器和混合器在特征间共享。global 默认全区间混合，不提供 prefix 的逐输出边界保证。无有效/可用 leader 时严格回退基础 token。

新路径令 `detach_leaders=False`，使没有 DAG 辅助任务的 E_X 获得主任务梯度。只放开已观测 X token/映射的梯度，lag 的 argmax 仍不可导。原 TokenLeadRefiner 的 detach 默认 True，旧 DAG/LIFT 行为保留。

## 解码与训练

未来 token 与 E_Y 的实际查询时间编码拼接，经 MLP 输出各查询时刻的 Y。未来 Y/标签 mask 仅由训练/评估器使用，不进入模型前向。

三个新模型只有未来 Y 查询的 MinMax masked variable-macro MSE；辅助项恒为零，训练入口 beta=0。保留训练集标准化验证选模、相同测试 mask 和原量纲/MinMax/micro/persistence 指标，不更换评估口径。

manifest 标记 architecture_version=4、lag 特征空间、leader 梯度和无辅助任务。旧 DAG/v2/tpatch/exo/delay 的工厂路由与保存配置重载保留；新结构必须重新训练，dag 对照仍使用原结构和原目标，不是严格的单因素替换实验。
