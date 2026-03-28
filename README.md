# TCN-TimeOmni-Hybrid

TCN-TimeOmni-Hybrid 是一个面向涡扇发动机剩余寿命预测（RUL）与维护解释生成的融合式仓库。

该项目将以下三部分整合到同一套流程中：

- 基于时序模型的 RUL 预测
- 基于预测区间的不确定性感知告警
- 基于 TimeOmni 的结构化维护解释生成

这个仓库既可以单独用于寿命预测实验，也可以用于生成面向运维人员的风险总结、维护建议和证据解释。

## 项目能力

给定训练好的模型和 C-MAPSS 测试数据，当前流程可以：

1. 生成发动机级别的 RUL 预测结果
2. 对支持的模型输出预测不确定性
3. 将预测结果映射为告警等级
4. 总结近期关键传感器变化趋势
5. 导出 prompt，或调用本地 TimeOmni 生成结构化维护解释

仓库同时支持：

- 点预测模型
- 不确定性模型
- 数值预测评估
- 解释质量评估

## 主要组成

- `tcn_core/`：RUL 建模核心代码，包括数据集、模型、损失函数、指标和训练脚本
- `src/hybrid_rul/`：融合推理层，负责连接预测结果、告警逻辑、传感器摘要和解释生成
- `timeomni_core/`：仓库内置的 TimeOmni 相关代码
- `configs/tcn/`：不同子集、不同 backbone、不同预测模式的训练配置
- `configs/hybrid/`：融合解释阶段的配置
- `scripts/`：预处理、训练、评估、可视化和 hybrid demo 的统一入口
- `artifacts/`：默认输出目录，包含 checkpoint、日志、图像和 hybrid 结果
- `results/`：导出的报告、prompt 和附加分析结果

## 当前支持范围

当前仓库已经提供以下配置：

- C-MAPSS 子集：`FD001`、`FD002`、`FD003`、`FD004`
- 预测模式：`point`、`uncertainty`
- 模型骨干：`tcn`、`lstm`、`gru`、`transformer`

对于不确定性模型，当前流程支持：

- 预测均值和方差输出
- 后验 `sigma` 校准
- 基于区间质量的模型选取
- 基于高不确定性的告警升级

## 安装

安装基础依赖：

```bash
pip install -r requirements.txt
```

或者：

```bash
make install
```

说明：

- 如果你只跑 RUL 预测主流程，基础依赖就够了。
- 如果你只想导出 prompt，而不做本地大模型推理，也不需要准备 TimeOmni 本地模型。
- 如果你计划同时运行预测和本地大模型推理，通常建议将 forecasting 和 LLM 推理拆成两个独立环境。

## 数据准备

请将 C-MAPSS 数据放到 `data/` 目录下。

例如运行 `FD001` 时，至少需要：

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

`FD002`、`FD003`、`FD004` 也采用同样的命名方式。

也可以通过环境变量指定数据目录：

```bash
export CMAPSS_DATA_DIR=data/
```

## 可选的 TimeOmni 配置

如果你需要在本地直接生成解释结果，需要在 hybrid 配置中指定本地模型目录，或者使用环境变量。

相关配置项为：

- `paths.timeomni_model_dir`

也可以直接设置：

```bash
export TIMEOMNI_MODEL_DIR=/path/to/local/timeomni/model
```

如果没有本地模型，当前仓库仍然可以正常导出 prompts，供外部推理流程使用。

## 快速开始

### 1. 训练模型

训练主线不确定性 TCN：

```bash
python scripts/train_tcn.py --config configs/tcn/fd001_tcn_uncertainty_tuned.yaml
```

训练点预测版本：

```bash
python scripts/train_tcn.py --config configs/tcn/fd001_tcn_point_tuned.yaml
```

如果你想切换到其他子集或其他 backbone，只需要替换配置文件。

### 2. 评估模型

```bash
python scripts/evaluate_tcn.py --config configs/tcn/fd001_tcn_uncertainty_tuned.yaml
```

### 3. 运行 Hybrid Demo

运行不确定性版本：

```bash
python scripts/run_hybrid_demo.py --config configs/hybrid/fd001_hybrid_local.yaml
```

运行点预测版本：

```bash
python scripts/run_hybrid_demo.py --config configs/hybrid/fd001_hybrid_point_local.yaml
```

### 4. 一键串联训练与解释

```bash
python scripts/run_full_pipeline.py \
  --tcn-config configs/tcn/fd001_tcn_uncertainty_tuned.yaml \
  --hybrid-config configs/hybrid/fd001_hybrid_local.yaml
```

如果你已经有 checkpoint，只想运行 hybrid 阶段：

```bash
python scripts/run_full_pipeline.py \
  --tcn-config configs/tcn/fd001_tcn_uncertainty_tuned.yaml \
  --hybrid-config configs/hybrid/fd001_hybrid_local.yaml \
  --skip-train
```

## 配置文件命名规则

### 预测配置

`configs/tcn/` 下的文件遵循以下命名方式：

```text
fd00X_<backbone>_<mode>_tuned.yaml
```

例如：

- `fd001_tcn_uncertainty_tuned.yaml`
- `fd003_lstm_point_tuned.yaml`
- `fd004_gru_uncertainty_tuned.yaml`

这些配置主要定义：

- 数据子集和窗口参数
- 模型骨干与网络超参数
- 优化器和训练策略
- 告警阈值
- 输出目录

### Hybrid 配置

`configs/hybrid/` 下的配置用于控制解释生成流程。

- `fd001_hybrid_local.yaml`：不确定性版本的 hybrid 推理
- `fd001_hybrid_point_local.yaml`：点预测版本的 hybrid 推理

这些配置主要定义：

- 加载哪个预测配置
- checkpoint 路径
- 是否调用本地 TimeOmni
- 生成参数
- 输出目录

## 输出结果

默认情况下，主要输出目录包括：

- `artifacts/tcn/checkpoints/`：训练得到的 checkpoint
- `artifacts/tcn/logs/`：训练日志
- `artifacts/tcn/results/`：数值评估输出
- `artifacts/tcn/figures/`：可视化结果
- `artifacts/hybrid/`：hybrid demo 输出
- `results/`：导出的 prompt、报告以及附加分析文件

典型 hybrid 输出包括：

- 发动机级预测结果
- 告警等级
- prompt 导出文件
- LLM 原始输出
- 清洗后的结构化解释
- 解释质量评估结果

## 评估指标

### 数值预测指标

当前预测流程支持：

- `RMSE`
- `PHM Score`

对于不确定性模型，还支持：

- `PICP`
- `MPIW`
- `Interval Score`

其中，不确定性模型的选模逻辑不是单纯看 RMSE，而是优先基于校准后的区间质量。

### 解释质量指标

Hybrid 流程还会对解释结果进行自动评估，包括：

- 原始格式合规性
- 清洗后格式合规性
- 是否落到预测值和传感器证据上
- 是否与 warning level 保持一致
- 动作建议是否合理

这样不仅可以评估解释“能不能生成”，还可以评估解释“是否忠实于底层数值决策”。

## 常用命令

`Makefile` 中已经提供了一些常见快捷入口：

```bash
make install
make train
make eval
make hybrid
make full
```

如果你需要切换配置，仍然建议直接使用 Python 脚本入口。

## 仓库结构

```text
TCN-TimeOmni-Hybrid/
├── artifacts/
├── configs/
│   ├── hybrid/
│   └── tcn/
├── data/
├── results/
├── scripts/
├── src/
│   └── hybrid_rul/
├── tcn_core/
└── timeomni_core/
```

## 给外部使用者的说明

- 预测模块和 LLM 模块是松耦合的。你可以只使用 RUL 预测，也可以只用它来导出 prompts。
- Hybrid 流程默认保留预测结果、告警决策、prompt、原始回答、清洗后回答和质量评估，因此整个解释链路是可审计的。
- 如果你要做模型对比，建议固定评估逻辑，只替换 `configs/tcn/` 中的预测配置。

## 使用说明

如果你在使用本仓库中的 vendored 组件或本地 TimeOmni 权重，请自行确认对应模型、代码和数据的许可证与使用约束。
