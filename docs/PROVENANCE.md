# 来源与版本

本仓库是独立研究工程。`upstream/tpatchgnn/` 保存官方仓库的 Python 源码、运行脚本、README 和 requirements，保持原文件内容；不搬运数据、图片、字节码、旧 Git 历史。

| 来源 | 固定提交 | 用途 |
|---|---|---|
| [t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN) | `00c94e7bbaf21c71b03ed84ff690ae59e37129e5` | 官方 Patch/TE/TTCN、Transformer、动态图 GNN、解码器和评估协议 |
| [用户 DAG](https://github.com/Capybaring/dag) | `f4468901331f53bb84bbc405219150df997e26c5` | 最新不规则长表/记录/collator 与 masked 评估适配；没有复用其模型 |
| [PDFormer](https://github.com/BUAABIGSCity/PDFormer) | `f8c8f6ad007a04fad3baee958b89504711852ce9` | 模式聚类、模式检索和 Key 增强机制参考 |

`exobase/upstream_model.py` 从官方 `tPatchGNN/model/tPatchGNN.py` 复制，删除未使用的 `model.Transformer_EncDec`、`model.SelfAttention_Family`、`lib.utils`、`lib.evaluation` 导入，以及两个参数初始化的强制 `.cuda()`。这避免导入无关依赖，并支持统一的 `model.to(device)`。

`exobase/data.py`、`metrics.py` 基于用户 DAG 最新不规则适配重构，补充独立缓存、可选时间划分、重叠 Patch、原时间导出和独立包接口。DAG 的 MIT 声明保存在 `upstream/notices/DAG-LICENSE.txt`；PDFormer 的 MIT 声明保存在 `upstream/notices/PDFormer-LICENSE.txt`。本次取得的 t-PatchGNN 提交没有根 LICENSE 文件；原作者说明和源码保留在 upstream，不对其代码另行声称授权。

请引用基础方法：

- Weijia Zhang et al. *Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach*. ICML 2024. [论文](https://proceedings.mlr.press/v235/zhang24bw.html)。
- Jiawei Jiang et al. *PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction*. AAAI 2023. [论文](https://ojs.aaai.org/index.php/AAAI/article/view/25556)。

Patch 级均值模式库、外生变量注意力和连续流记录协议是本项目适配，不能标注为原论文未经修改的完整复现。
