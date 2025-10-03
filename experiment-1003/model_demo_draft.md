# Model Demo Plan (Transformer Timeseries Classification)

## 1. Objective
- 搭建一个独立的时序模型实验，用于熟悉 Transformer 在金融时间序列分类中的流程。
- 当实验流程成熟后，再与量化研究主线对接。

## 2. Dataset
- 标的：`TSM`
- 时间范围：`2014-01-01` 至 `2024-12-31`
- 数据文件：`experiment-1003/tsm_20140101_20241231.csv`
  - 字段：`open, high, low, close, volume`
  - 由 `qlib` 数据集导出，未来可替换为其他标的或频率。
- 数据划分：按时间顺序划分训练 / 验证 / 测试 = 7 / 2 / 1
  - 训练集：2014-01-01 ~ 2020-12-31
  - 验证集：2021-01-01 ~ 2022-12-31
  - 测试集：2023-01-01 ~ 2024-12-31

## 3. Feature Engineering
- 原始特征：`open, high, low, close, volume`
- 传统指标：与 `scripts/technical_indicators.py` 一致，包括 EMA/BOLL/SAR/KC、MACD/DMI/DMA/ADTM、DDI/DPO/OSC、CCI/KDJ/RSI、ARBR/PSY/VR、ATR、Mass Index、SRMI 等。
- 休假指标：`gap_days = (当前交易日日期 − 上一交易日日期) − 1`
  - 连续交易日取 0；周末或假期跨越返回休市天数。
- 二级信号：`scripts/secondary_signals.py` 提供的当日标签，可选作为额外输入。
- 标签：采用 `scripts/trade_labels.py` 中的胜/平/负定义。
  - 长线：窗口 15 日，止损 `Close − 2×ATR`，止盈 `Close + 3×ATR`。
  - 短线：窗口 5 日，止损 `Next Open − 1×ATR`，止盈 `Next Open + 1.5×ATR`。
  - 标签取“当日收盘买入”与“次日开盘买入”的最差结果，胜=1、平=0、负=−1。

## 4. Experiment Pipeline
1. **配置管理**
   - 新建 `config/model_demo.json`，定义数据路径、窗口长度、特征列表、超参数等。
2. **数据处理脚本**
   - 读取 CSV → 计算技术指标、gap_days、二级信号 → 生成长/短线标签。
   - 视需求仅保留需要的特征，处理缺失值（例如填充或剔除窗口前若干行）。
   - 将样本滑动窗口化：窗口长度 120 日，预测窗口末日的标签。
3. **数据集封装**
   - 使用 PyTorch `Dataset`/`DataLoader`，提供 `train/val/test` 三个拆分。
   - 支持 `cuda` 加速、`pin_memory` 等参数。
4. **模型构成**
   - Encoder-only Transformer：数层 `nn.TransformerEncoderLayer`，含多头自注意力、前馈网络、LayerNorm、Dropout。
   - 输入嵌入：对连续特征采用 `nn.Linear` + `Positional Encoding`。
   - 分类头：平均池化或取最后 token → 全连接层输出三分类 logits。
5. **训练配置**
   - 损失函数：`nn.CrossEntropyLoss`（可考虑类别权重）。
   - 优化器：`AdamW`，初始学习率如 `1e-4`，配合 `StepLR` 或 `ReduceLROnPlateau`。
   - 训练循环：
     - 每固定 `N` 步保存一次模型到 `outputs/models/`。
     - 监控训练/验证损失与指标（Accuracy、F1）。
     - 早停策略可选。
6. **评估与报告**
   - 测试集评估：Accuracy、F1（macro/micro）、混淆矩阵。
   - 标签分布、阈值敏感性分析。
   - 生成可视化（训练曲线、注意力热力图等）。

## 5. Task Checklist
- [x] 导出 TSM 2014-2024 CSV。
- [ ] 编写特征构建脚本，生成训练样本。
- [ ] 实现 Transformer 模型与训练脚本。
- [ ] 跑通训练 → 验证 → 测试，并归档模型/日志。
- [ ] 输出实验报告并同步到主研究计划。
