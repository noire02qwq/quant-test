# Trend Regression Smooth Experiment

独立实验围绕 QQQ 2014-2024 的 trend-scanning 打标，但将价格序列换成 `EMA5` 的对数并将未来观察窗口限制在 5-15 日。标签字段 `trend_ret_pct` 表示 t1 相比 t0 的收益率百分比，模型回归目标即为该列。

## 运行步骤

1. **数据准备**
   ```bash
   cd module-experiments/trend-regression-smooth
   python prepare_data.py --symbols QQQ
   ```
   - 会在 `data/raw` 下载 CSV，并在 `data/qlib` 生成 qlib 二进制数据。默认会在 2013-07 至 2025-07 的采样范围内留出缓冲。

2. **标签计算 + 可视化**
   ```bash
   python compute_labels.py
   ```
   - 产物：`data/processed/qqq_trend_regression_smooth_labels.csv`、`outputs/QQQ_trend_2024.png`（K 线背景+四条折线）、`outputs/QQQ_trend_hist.png`（t-value/ret 直方图）。
   - 可在 `trend_tag_analysis.TrendScanParams` 中调节窗口或 EMA 平滑参数。

3. **训练**
   ```bash
   python train.py --config configs/transformer_alpha.json
   ```
   - `feature_mode` 固定为 `"alpha"`，通过修改 JSON 中的 `feature_mode_params.alpha_kind` 在 `alpha158` 与 `alpha360` 之间切换（无需更换配置文件）。
   - 训练程序统一：
     * 自动划分 8:1:1（2014-2021 / 2022 / 2023），在 `outputs/<run>/checkpoints` 记录阶段性模型，并 **每 10 个 epoch 固定保存一次 checkpoint**。
     * 只使用 `trend_r2 ≥ 0.75` 的样本，EMA5 标签保持不变。
     * 使用 MAE loss，且只有当 `train_mae` 降到初始值的 1/4 以内才允许刷新最佳模型，防止保存未收敛的 checkpoint。
     * `metadata_*.json` 会保存运行参数、特征列、best checkpoint 路径等，训练结束还会生成 `training_summary.txt`（含 best epoch、best val/test MAE/MSE）。

4. **测试 / 推理可视化**
   ```bash
   python test.py --config configs/transformer_alpha.json --split test
   ```
   - 默认读取 `metadata_*.json` 中记录的最佳 checkpoint，可通过 `--checkpoint` 重载。
   - 结果包含每个切分的 MSE/MAE、`test_predictions.csv`，以及 `*_test_true_ret.png` / `*_test_pred_ret.png`（2023 年 K 线 + 真值/预测收益率百分比折线）。

## 数据管线与特征切换
- 仅保留 alpha 输入：`feature_mode` 固定为 `"alpha"`，并通过 `feature_mode_params.alpha_kind` 设置 `alpha158` 或 `alpha360`。
- 数据集附带 `trend_tvalue`/`trend_window`/`trend_r2` 供分析，但训练时只使用 `label`（ret%），同时会自动过滤 `trend_r2 < 0.75` 的样本。

## 其他说明
- 所有脚本默认只处理 QQQ，可通过命令行 `--symbols` 扩展，feature pipeline 会对每个 symbol 单独滑窗再拼接。
- `python -m compileall module-experiments/trend-regression-smooth` 已通过，用于快速语法校验。
- 运行前请确认虚拟环境中安装了 `numpy`, `pandas`, `torch`, `matplotlib`, `qlib`, `yfinance`, `openpyxl`（写 Excel）等依赖。

## Transformer 调参脚本
- `tune_transformer.py` 会遍历 draft 中的网格，每组超参生成一个独立文件夹（`transformer_158_001`, `transformer_360_001`, ...），目录内包含该组 config、checkpoints、best 模型、训练曲线以及 `training_summary.txt`。
- 运行示例：
  ```bash
  # 仅跑 4 组用于冒烟
  python tune_transformer.py --max-configs 4

  # 按需继续，可跳过已完成项
  python tune_transformer.py --skip-existing --max-configs 20
  ```
  - `--dry-run` 只生成配置不执行训练。
- 所有运行结果会汇总到 `<output-root>/tuning_summary.xlsx`，每一行记录超参取值及对应的 best/test 指标，便于后续筛选。
