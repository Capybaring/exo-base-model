# exo-base-model

面向 **不规则观测 + 外生变量 + 滞后模式** 的研究 baseline。以官方 t-PatchGNN 为基础，继承它的真实时间 Patch、TTCN、时间 Transformer、动态图学习、图卷积和查询解码器；新增历史/已知未来外生变量接口，以及 PDFormer 式短期模式检索。**没有导入或继承 DAGModel、PatchExoModel、PatchDelayModel。**

目前是可运行的研究起点，不是已经完成精度验证的新方法。历史主干、外生注意力和模式检索可分别替换。来源、公式和适配边界见 [设计说明](docs/DESIGN.md)、[来源与版本](docs/PROVENANCE.md)。

## 安装

Python 3.10+，推荐独立环境。已在 macOS ARM64、Python 3.12、CPU 上验证；CUDA/MPS 未实测。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

`requirements.txt` 给出兼容依赖范围；`requirements-tested.txt` 记录本次实际验证版本，不保证其中所有固定版本都有其他平台的 wheel。`tslearn` 的 k-Shape 用于 `delay` 模式；HDF5 功能不使用，无须安装可选 h5py。

## 本地数据准备

支持 `date,data,cols` 或 `time,value,variable` 长表。时间可以是相对秒或日期字符串。**不需要把原始连续流改成患者数据文件**；读取器将其转换为独立、非重叠的时间记录，再由 collator 按真实时间分块。可直接训练 CSV，也可使用准备命令生成的压缩 `.npz` 缓存，避免重复解析大 CSV。

```bash
python -m exobase.prepare \
  --dist /path/to/DIST_irregular_9x9.csv \
  --dataset-dir /path/to/dataset \
  --drop-rate 0.3 --seed 2021
```

输出到 `data/processed/`：

- `DIST.npz`：保留原有不规则观测，不额外丢弃；同变量同时间的重复记录取均值。
- `BE_irregular.csv`、`NP_irregular.csv`：在源长表上逐观测独立进行 Bernoulli 丢弃。时间戳不改变，不插值，不压缩时间缺口。
- `BE.npz`、`NP.npz`：对应的不规则观测缓存。
- `preparation_manifest.json`：原文件 SHA256、变量顺序、实际保留数量、重复记录和时间范围。

输入文件从不修改。生成物、模型权重、预测值均被 Git 忽略，不上传到代码仓库。准备好的输出默认不覆盖，重新生成请使用新目录或显式 `--overwrite`。

默认 30% 丢弃作用于 Y/X 的整个流，所以训练、验证、测试标签也变为不规则观测；评估仅统计保留标签。这与“只丢历史、用完整标签测试”的协议不同。不同丢弃率实验应明确使用哪一种评价协议，不直接把变化后的测试点集当作固定测试集。

## 三个数据配置

| 配置 | Y / X | 历史 / 预测 | Patch 时间宽度 |
|---|---|---|---|
| `configs/DIST.json` | 9 个 PV / 对应 9 个 OP | 4 小时 / 2 小时 | 960 秒 |
| `configs/BE.json` | OT / 两个预测协变量 | 168 小时 / 24 小时 | 24 小时 |
| `configs/NP.json` | OT / 两个预测协变量 | 168 小时 / 24 小时 | 24 小时 |

BE/NP 的源变量名称有前导空格，配置保留其原名；不要随意去掉配置中的空格。DIST 的回路顺序为 FIC05、FIC09、FIC11、LIC02、LIC03、LIC04、LIC05、LIC06、LIC12。

各配置仅是可修改的起点。`history`、`forecast`、`patch_seconds`、`patch_stride` 单位都是秒，不是行数。已知未来 X 是实验假设：DIST 未来 OP 是否实际可获得，需要由具体任务确认；不成立时使用 `tpatch` 历史外生模式。

## 训练与消融

```bash
python -u -m exobase.train --config configs/DIST.json
python -u -m exobase.train --config configs/BE.json
python -u -m exobase.train --config configs/NP.json
```

命令行参数覆盖配置。`python -m exobase.train --help` 查看全部参数。模型变体为：

| `--variant` | 历史主干输入 | 已知未来 X | 模式检索 |
|---|---|---|---|
| `tpatch-y` | Y | 否 | 否 |
| `tpatch` | Y + X | 否 | 否 |
| `exo` | Y + X | 是，X→Y 注意力 | 否 |
| `delay` | Y + X | 是，X→Y 注意力 | 是，只增强 Key |

全部变体只预测并监督 Y。`tpatch-y`、`tpatch` 保留上游预测结构，除空 Patch 数值防护外，非空输入上的同权重输出有对照测试。`exo`/`delay` 在其历史表示上加入外生注意力，是扩展模型。

```bash
bash scripts/run_ablations.sh BE --epochs 50
```

每次使用不同 `--save-path`；覆盖既有 checkpoint 需要显式 `--overwrite`。首轮建议先做流程检查：

```bash
python -u -m exobase.train --config configs/DIST.json \
  --max-records 12 --epochs 2 --batch-size 1 \
  --d-model 8 --te-dim 3 --n-heads 2 \
  --n-patterns 3 --max-pattern-windows 200 \
  --save-path results/smoke_DIST
```

`--max-records` 会影响划分、缩放和模式库，仅用于流程验证，不能报告为完整实验成绩。正式运行移除此参数。

## 数据划分和评估协议

默认使用不重叠的 `history+forecast` 时间记录，保留空 Patch 的物理位置。最后不足一条记录的尾段、没有历史观测或没有未来 Y 标签的记录不参与，数量写入 manifest。

- `--split official-random`：复用 t-PatchGNN 的 80% seen / 20% test，再将 seen 的 75% 用于训练；固定 random_state=42，约 60/20/20。
- `--split chronological`：按记录时间依次分为 60/20/20，更适合连续工业数据的时间外推实验。
- `--normalization seen`：默认与上游协议一致，在 train+validation 的有效观测上拟合 min-max；不是严格 train-only 缩放。
- `--normalization train`：只在训练记录拟合缩放。推荐正式时间外推时配合 chronological 使用。

模式库无论选择哪种缩放协议，都只用训练记录中的 X 窗口拟合，测试不参与；已知未来 X 可参与所属训练记录的模式构建。所有划分、缩放和模式库信息随 checkpoint 保存。

损失为归一化尺度的 masked MSE：先每个目标变量对有效位置取均值，再对有标签变量宏平均。评估先跨全部 batch 累计每变量误差和计数，最后宏平均；RMSE=sqrt(MSE)。主要输出 MSE/MAE/RMSE；`mape_official_signed` 仅为上游兼容诊断，沿用归一化目标的有符号分母，可能为负，不应当成通常定义的原量纲 MAPE。

## 重载与预测导出

```bash
python -m exobase.predict \
  --checkpoint results/DIST/delay/best_model.pt \
  --output results/DIST/delay/test_predictions.csv
```

默认预测 checkpoint 保存的 test 记录，也可选 train/val。可用 `--data-path` 指向移动后的同一文件；SHA256 必须匹配，以免误用另一版丢弃数据或缩放。导出原始量纲的预测、记录 ID、真实相对秒、目标名、标签是否观测以及有效标签。日期型原文件内部转为相对秒。

每次训练保存 `manifest.json`、`best_model.pt`、`metrics.json`；delay 模式另外保存 `patterns.npz`。本次验证范围见 [验证记录](docs/VALIDATION.md)。
