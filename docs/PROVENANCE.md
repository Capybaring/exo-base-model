# 来源与固定版本

| 来源 | 固定提交 | 当前用途 |
|---|---|---|
| [t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN) | 00c94e7bbaf21c71b03ed84ff690ae59e37129e5 | 真实时间分块/打包接口、旧模型及评估对照 |
| [用户 DAG](https://github.com/Capybaring/dag) | f4468901331f53bb84bbc405219150df997e26c5 | 长表适配与时间/变量双分支骨架 |
| [LIFT](https://github.com/SJTU-DMTai/LIFT) | 79cf8f157bb9a616c9733daf1e9c2cfefdc98c94 | 领先估计、目标对齐、状态相关频域修正 |
| [PDFormer](https://github.com/BUAABIGSCity/PDFormer) | f8c8f6ad007a04fad3baee958b89504711852ce9 | 仅保留旧 delay 对照，不用于默认模型 |

## DAG

upstream/dag/ 保存固定提交的 DAGModel 及五个依赖原文件。exobase/dag_layers/ 是独立导入的运行副本：

- DAGModel、TC_EncDec、CC_EncDec、Transformer_EncDec 仅加来源头并修改导入路径。
- Embed 仅保留 CompressAndProject、PositionalEmbedding、PatchEmbedding，类体不变。
- SelfAttention_Family 仅保留 FullAttention、AttentionLayer，删除无关 reformer/einops 依赖；补充等价三角 mask，当前调用均不启用该 mask。
- exobase/dag_model.py 封装规则 Patch 接口、两个 masked 辅助损失和真实/预测未来 X 路径选择；这些适配与原方法应分别说明。

运行副本另清理行尾空白；上游快照保留原始字节内容。MIT 声明位于 upstream/notices/DAG-LICENSE.txt。

## LIFT

upstream/lift/ 保留官方 models/LIFT.py、util/lead_estimate.py、README 原内容，不含数据或 Git 历史。exobase/lift_layers.py 只提取 FilterFactory、ComplexLinear，类体不变。

exobase/lags.py 是迁移实现：X→Y 限制、非循环有限窗口带 mask Pearson、已知未来 X、三路频域混合及 prefix 时间约束。不能声称估计器和官方 FFT 代码数值等价，也不声称原论文未经修改的复现。

官方 README 说明论文提交后调整过代码且尚未重跑全部实验，因此明确固定提交。此提交无根 LICENSE 文件，保留作者/来源，不另行声称代码授权。

论文：[Zhao and Shen，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/b52b07a239a7afa155ca25cf17a55074-Paper-Conference.pdf)。

## t-PatchGNN 与旧版本

upstream/tpatchgnn/ 保留 Python 源码、运行脚本、README、requirements，不含数据和历史。exobase/upstream_model.py 删除无关导入及强制 .cuda() 初始化；旧模型 TTCN 增加空块与隐藏 NaN 防护。

当前 DAG 模式只复用时间分块接口；均值汇总为项目适配，非官方 TTCN。旧图模型仅在显式 tpatch-y/tpatch/exo/delay 下实例化，旧 checkpoint 仍可重载。

t-PatchGNN 固定提交无根 LICENSE 文件，不另行声称代码授权。PDFormer MIT 声明位于 upstream/notices/PDFormer-LICENSE.txt。

参考论文：[t-PatchGNN，ICML 2024](https://proceedings.mlr.press/v235/zhang24bw.html)、[PDFormer，AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25556)。v0.1 的方法和原始验证记录位于 docs/LEGACY_V1.md，不作为当前模型的精度证据。
