**TTCN + DAG + LIFT 替换包（2026-10-18）**

适用：此前 exo-base-model-2fa42e0，以及已经安装统一评估补丁的版本。把 ZIP 解压到远程项目根目录，与 exobase、configs 同级，覆盖同名文件即可。不需要上传 GitHub，不需要重新生成数据或安装新的依赖。

**文件**

本次模型核心改动 5 个文件：

- exobase/model.py：模型配置及工厂；旧 checkpoint 自动保留 v2。
- exobase/train.py：新训练默认选择 TTCN，记录 architecture_version=3。
- exobase/ttcn.py：新增，官方点值+时间 TTCN 方程与实际查询时间解码。
- exobase/token_dag.py：新增，完整向量 token 的 DAG 时间/变量双注意力。
- exobase/token_lift.py：新增，完整向量序列的滞后相关、对齐、三路频域修正。

另附 exobase/metrics.py、exobase/evaluate.py 两个此前统一评估文件，保证从原始代码包直接覆盖也能运行。附带源码说明及测试；不包含数据或模型权重。

**远程替换和训练**

在项目根目录执行，例如：

```bash
unzip -o /path/to/exo-ttcn-dag-lift-update-20261018.zip -d .

python -u -m exobase.train \
  --config configs/DIST.json \
  --variant dag-lift \
  --patch-encoder ttcn \
  --history 86400 \
  --forecast 86400 \
  --patch-seconds 3600 \
  --patch-stride 3600 \
  --max-lag 12 \
  --min-pairs 8 \
  --device cuda:1 \
  --save-path results/DIST/ttcn-dag-lift-24to24-hourly
```

这是过去24小时预测未来24小时，每步1小时，有LIFT。不要添加 --max-records 或 --smoke，它们仅用于小规模管线检查。新训练需要使用新目录；已有 best_model.pt 会受到保护。内存不足时先加 --batch-size 1。TTCN 和点解码已经分块，减少长 patch 的激活内存。

BE/NP/PJM/FR/DE/Energy/小时风电若同样采用每步1小时24→24，可沿用上述时长与patch参数，仅替换配置和保存目录。半小时风电24步对应12小时：history=forecast=43200、patch_seconds=patch_stride=1800。Colbun/Rapel为日级，24步对应24天：history=forecast=2073600、patch_seconds=patch_stride=86400。全部时间参数单位为秒。

此前 run_remaining_dag.py 固定 --variant dag，运行它仍是无LIFT对照；本次完整模型请使用上面的 --variant dag-lift 命令。

**实际变化**

1. 每个patch内保留各观测值、原始时间和mask，经可学习时间编码与官方TTCN动态聚合，产生d_model维token。非空公式已与官方原函数数值对齐；全空patch明确输出0及有效标记0。
2. DAG直接处理多维token。时间注意力跨patch；变量注意力跨外生变量。没有再把TTCN输出压成单个标量后送入旧DAG，也没有实例化GNN。
3. dag-lift基础预测仅用历史；真实未来X只经过对齐后的LIFT进入最终预测。LIFT对每个特征沿时间轴做FFT，保留向量维，prefix保持patch级滞后边界。
4. 最后将未来token与真实查询时间编码拼接，解码成PV数值。同一patch内不同查询时间不再被强制预测为同一个数值。
5. 辅助任务监督的是原始观测未来X与历史Y，不用会与编码器一起变化的token作为标签。未来Y仍只用于主预测损失/评估。
6. 滞后仍是历史统计搜索，不直接反向传播。它现在基于共享TTCN多维表示的逐特征时间中心化相关，不是原始OP/PV的Pearson。完全不变的历史原始变量额外排除，防止仅时间特征产生leader。相关性不是物理因果效应。

**兼容和评估**

- 模型结构已改变，需要重新训练，不能将旧权重当成新TTCN模型继续训练。
- 旧 checkpoint 缺少patch_encoder字段时仍加载原均值架构。已检查一份真实v2 checkpoint，重载预测CSV与此前逐字节一致。
- 新 checkpoint 的 model_config.patch_encoder='ttcn'，architecture_version=3；predict/evaluate原命令可继续使用。
- 评估仍为仅训练集原始有效观测拟合的Z-score、variable_macro，同时记录micro/raw/model_minmax及持久性基线。主训练损失仍为MinMax masked MSE加辅助损失。
- 旧版本的显式 --patch-encoder mean 仅用于对照；新模型请保持ttcn。

**验证**

60项自动测试通过；包括官方TTCN非空数值对照、相同均值但不同分布/时间的区分、padding/全空/隐藏NaN安全、向量正负滞后恢复、单特征LIFT与旧公式对照、未来控制prefix边界、无未来Y泄漏、梯度与checkpoint重载。

真实DIST数据用1小时patch、24→24、d_model=64、batch=1完成小规模一轮训练/验证/测试。BE及其余11个表格数据集也完成小规模训练、checkpoint、评估和滞后CSV导出检查。这些不是正式精度实验，没有声称比旧模型更准确。

详细接口和来源见 docs/TTCN_V3.md 与 docs/PROVENANCE.md。
