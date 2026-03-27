# TCN-TimeOmni-Hybrid

一个单仓库的杂交项目，面向以下完整流程：

1. 从头训练 TCN RUL 模型
2. 生成带不确定性的 RUL 预测结果
3. 基于预测结果和近期传感器变化构造维护分析 prompt
4. 调用 TimeOmni 做推理解释和决策建议

## 目录

```text
TCN-TimeOmni-Hybrid/
├── artifacts/                      # 训练结果、checkpoint、日志、混合报告
├── configs/
│   ├── hybrid/
│   │   ├── fd001_hybrid_local.yaml
│   │   └── fd001_hybrid_point_local.yaml
│   └── tcn/
│       ├── fd001_tcn_uncertainty_tuned.yaml
│       └── fd001_tcn_point_tuned.yaml
├── data/                           # 放 CMAPSS 数据，也可放 TimeOmni 测试集
├── scripts/
│   ├── preprocess_tcn.py
│   ├── train_tcn.py
│   ├── evaluate_tcn.py
│   ├── run_hybrid_demo.py
│   └── run_full_pipeline.py
├── src/hybrid_rul/                 # 融合层
├── tcn_core/                       # vendored TCN 项目代码
└── timeomni_core/                  # vendored TimeOmni 项目代码
```

## 环境

基础安装：

```bash
pip install -r requirements.txt
```

也可以直接：

```bash
make install
```

如果你要完整跑 `timeomni_core/eval/` 里的 vLLM 评测，再额外安装对应环境。

## 数据准备

### CMAPSS

把以下文件放到仓库根目录下的 `data/`：

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

也兼容把 `CMAPSSData.zip` 放在仓库根目录。

### TimeOmni 模型

如果只导出 prompt，不需要本地模型。

如果要实际做推理，在 `configs/hybrid/fd001_hybrid_local.yaml` 中填入：

- `paths.timeomni_model_dir`

它应该指向本地可被 `transformers` 读取的 TimeOmni 模型目录。

仓库里已经提供了环境变量模板：

`.env.example`

## 从头训练 TCN

使用当前主线的 tuned 配置：

```bash
python scripts/train_tcn.py --config configs/tcn/fd001_tcn_uncertainty_tuned.yaml
```

或者：

```bash
make train
```

做点预测消融时可以直接换成：

```bash
python scripts/train_tcn.py --config configs/tcn/fd001_tcn_point_tuned.yaml
```

可选预处理查看：

```bash
python scripts/preprocess_tcn.py --config configs/tcn/fd001_tcn_uncertainty_tuned.yaml
```

训练完成后，checkpoint 默认输出到：

```text
artifacts/tcn/checkpoints/
```

如果你的 checkpoint 已经在云平台上，直接放到 `artifacts/tcn/checkpoints/` 就可以。

如果你想显式指定某个 checkpoint，也可以直接设置：

```bash
export TCN_CHECKPOINT=artifacts/tcn/checkpoints/best_model_fd001_tcn_uncertainty_tuned.pth
```


## 运行融合推理

```bash
python scripts/run_hybrid_demo.py --config configs/hybrid/fd001_hybrid_local.yaml
```

或者：

```bash
make hybrid
```

TCN 原来的可视化入口也已经补成了统一包装脚本：

```bash
python scripts/visualize_tcn.py --config configs/tcn/fd001_tcn_uncertainty_tuned.yaml
python scripts/visualize_tcn.py --config configs/tcn/fd001_tcn_point_tuned.yaml
```

或者：

```bash
make visualize
make visualize-point
```

点模型消融时，对应使用：

```bash
python scripts/run_hybrid_demo.py --config configs/hybrid/fd001_hybrid_point_local.yaml
```

默认会：

1. 读取本仓库里的 TCN tuned 配置和训练好的 checkpoint
2. 对测试集发动机做预测
3. 生成传感器趋势摘要
4. 导出 TimeOmni prompts
5. 如果配置里启用了本地模型，再调用 TimeOmni 生成维护分析结果

输出默认在：

```text
artifacts/hybrid/fd001_demo/
```

## 一键串起训练和融合

```bash
python scripts/run_full_pipeline.py \
  --tcn-config configs/tcn/fd001_tcn_uncertainty_tuned.yaml \
  --hybrid-config configs/hybrid/fd001_hybrid_local.yaml
```

或者：

```bash
make full
```

如果只想跳过训练、直接读已有 checkpoint：

```bash
python scripts/run_full_pipeline.py \
  --tcn-config configs/tcn/fd001_tcn_uncertainty_tuned.yaml \
  --hybrid-config configs/hybrid/fd001_hybrid_local.yaml \
  --skip-train
```

点模型版本只需要把两个配置一起替换：

```bash
python scripts/run_full_pipeline.py \
  --tcn-config configs/tcn/fd001_tcn_point_tuned.yaml \
  --hybrid-config configs/hybrid/fd001_hybrid_point_local.yaml
```

## 环境变量

当前支持的关键环境变量：

- `CMAPSS_DATA_DIR`
- `TCN_RESULTS_DIR`
- `TCN_FIGURES_DIR`
- `TCN_CHECKPOINT_DIR`
- `TCN_LOGS_DIR`
- `TCN_CHECKPOINT`
- `TIMEOMNI_MODEL_DIR`
- `HYBRID_OUTPUT_DIR`

这些变量会在加载配置时自动展开。


