# experiment-1003: Transformer 时序分类独立实验

本目录包含一个独立的时间序列分类实验，用于熟悉 Transformer 在金融 K 线数据上的建模流程。该实验不依赖仓库其他目录（除 CSV 数据外），可直接运行训练脚本开始训练。

## 目录结构
- `model_demo_draft.md`：实验方案与规范说明。
- `tsm_20140101_20241231.csv`：从 Qlib 数据导出的 TSM 日线 CSV（含 OHLCV）。
- `config.json`：实验配置（数据路径、模型结构、训练超参数、时间切分）。
- `data.py`：数据加载、特征构建（EMA/RSI/MACD/ATR/gap_days）、长线标签生成、滑窗数据集封装。
- `model.py`：Transformer Encoder 分类模型（使用 `torch.nn` 标准模块）。
- `train.py`：训练与验证、保存最优模型。
- `validate.py`：加载最优模型，评估验证集。
- `test.py`：加载最优模型，评估测试集。

## 环境依赖
- Python 3.9+
- PyTorch（CPU 或 CUDA 环境均可）
- pandas、numpy

## 数据与特征
- 输入 CSV：`config.json` 的 `data_path` 指向 `experiment-1003/tsm_20140101_20241231.csv`。
- 特征：
  - 原始：`open, high, low, close, volume`
  - 技术：`ret`、`EMA(5/10/20/60)` 及斜率、`MACD(dif/dea/hist)`、`RSI14`、`ATR14`
  - 休假：`gap_days = (当前交易日 − 上一交易日).days − 1`，连续交易日为 0，周末为 2
- 标签（长线）：窗口 15 自然日；同时评估“x 日收盘买入”与“x+1 日开盘买入”，止损 `2×ATR`、止盈 `3×ATR`，标签取两者**最差**（胜=1、平=0、负=−1 → 训练时映射到 {2,1,0}）。

## 训练与评估
1. 训练与验证（保存最优模型到 `experiment-1003/outputs/models/best.pt`）：
   ```bash
   python experiment-1003/train.py
   ```
2. 验证集评估：
   ```bash
   python experiment-1003/validate.py
   ```
3. 测试集评估：
   ```bash
   python experiment-1003/test.py
   ```

## 配置说明（`config.json`）
- `window_size`：滑窗长度（默认 120）。
- `label_window_days`：标签观察窗口（默认 15）。
- `splits`：按日期划分训练/验证/测试区间。
- `model`：Transformer 结构参数（`d_model/nhead/num_layers/dim_feedforward/dropout/pooling`）。
- `learning_rate/batch_size/num_epochs`：训练超参数。

## 备注
- 代码中包含中文注释，标明各步骤的具体作用。
- 可根据需要扩展更多特征（如 KDJ、BOLL、二级信号等），或替换为其他标的与时间范围。

