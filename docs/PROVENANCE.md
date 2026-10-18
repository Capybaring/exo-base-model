# 来源与固定版本

| 来源 | 固定提交 | 当前用途 |
|---|---|---|
| [t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN) | 00c94e7bbaf21c71b03ed84ff690ae59e37129e5 | 真实时间分块、LearnableTE/TTCN、查询时间解码结构、旧模型及评估对照 |
| [用户 DAG](https://github.com/Capybaring/dag) | f4468901331f53bb84bbc405219150df997e26c5 | 长表适配与时间/变量双分支骨架 |
| [LIFT](https://github.com/SJTU-DMTai/LIFT) | 79cf8f157bb9a616c9733daf1e9c2cfefdc98c94 | 领先估计、目标对齐、状态相关频域修正 |
| [PDFormer](https://github.com/BUAABIGSCity/PDFormer) | f8c8f6ad007a04fad3baee958b89504711852ce9 | 仅保留旧 delay 对照，不用于默认模型 |

## DAG

upstream/dag/ 保存固定提交的 DAGModel 及五个依赖原文件。exobase/dag_layers/ 是独立导入的运行副本：

- DAGModel、TC_EncDec、CC_EncDec、Transformer_EncDec 仅加来源头并修改导入路径。
- Embed 仅保留 CompressAndProject、PositionalEmbedding、PatchEmbedding，类体不变。
- SelfAttention_Family 仅保留 FullAttention、AttentionLayer，删除无关 reformer/einops 依赖；补充等价三角 mask，当前调用均不启用该 mask。
- exobase/dag_model.py 保留 v2 标量均值模型，用于旧 checkpoint 重载。
- exobase/token_dag.py 是 v3 的向量适配：复用双 softmax 注意力融合、时间/变量双分支、共享注意力编码器与上下文门控；embedding、预测头和归一化扩展为完整 TTCN 特征。主预测与两个辅助任务均由 token+查询时间解码为原始标量监督。这不是未经修改的官方 DAG。

运行副本另清理行尾空白；上游快照保留原始字节内容。MIT 声明位于 upstream/notices/DAG-LICENSE.txt。

## LIFT

upstream/lift/ 保留官方 models/LIFT.py、util/lead_estimate.py、README 原内容，不含数据或 Git 历史。exobase/lift_layers.py 只提取 FilterFactory、ComplexLinear，类体不变。

exobase/lags.py 是迁移实现：X→Y 限制、非循环有限窗口带 mask Pearson、已知未来 X、三路频域混合及 prefix 时间约束。不能声称估计器和官方 FFT 代码数值等价，也不声称原论文未经修改的复现。

exobase/token_lift.py 是 v3 向量适配：历史 TTCN 特征逐通道中心化后计算联合相关；原始常量来源在 token_dag.py 排除。完整特征保留，FFT 沿时间轴；同一组状态相关频率滤波器应用于各特征。单特征情形与 v2 LIFT 两种模式数值一致，作为回归检查。prefix 约束仍限定 patch 级控制影响，不能解释为已识别的物理因果延迟。

官方 README 说明论文提交后调整过代码且尚未重跑全部实验，因此明确固定提交。此提交无根 LICENSE 文件，保留作者/来源，不另行声称代码授权。

论文：[Zhao and Shen，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/b52b07a239a7afa155ca25cf17a55074-Paper-Conference.pdf)。

## t-PatchGNN 与旧版本

upstream/tpatchgnn/ 保留 Python 源码、运行脚本、README、requirements，不含数据和历史。exobase/upstream_model.py 删除无关导入及强制 .cuda() 初始化；旧模型 TTCN 增加空块与隐藏 NaN 防护。

exobase/ttcn.py 单独提取官方 LearnableTE/TTCN 方程及同结构参数，不实例化 Transformer/GNN。非空 patch 对官方原函数做数值对照；空 patch 清零、隐藏 NaN 清理和分块重算是额外防护/内存适配。v3 默认 DAG 模式使用该编码器；v2 标量均值模型仅作为旧实现保留。旧图模型仅在显式 tpatch-y/tpatch/exo/delay 下实例化。

t-PatchGNN 固定提交无根 LICENSE 文件，不另行声称代码授权。PDFormer MIT 声明位于 upstream/notices/PDFormer-LICENSE.txt。

参考论文：[t-PatchGNN，ICML 2024](https://proceedings.mlr.press/v235/zhang24bw.html)、[PDFormer，AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25556)。v0.1 的方法和原始验证记录位于 docs/LEGACY_V1.md，不作为当前模型的精度证据。
