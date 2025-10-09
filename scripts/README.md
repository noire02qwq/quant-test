# Scripts Overview

本目录包含与数据准备、技术指标与标签分析相关的脚本。所有脚本均假设本仓库根目录下存在 `data/qlib_us_selected/qlib_data` 数据集（可通过 `build_us_dataset.py` 构建）。如直接运行脚本，内部会调用 `qlib.init` 指向该目录。

## build_us_dataset.py
- 功能：下载指定标的的 Yahoo Finance 日线数据，转换为 Qlib 二进制格式。
- 核心流程：补丁 Yahoo collector 标的列表 → 下载 CSV → `DumpDataAll` 转换 → 输出行业分组清单。
- 使用示例：
  ```bash
  python scripts/build_us_dataset.py --start 2010-01-01 --end 2025-09-30 --force-refresh
  ```

## technical_indicators.py
- 功能：计算传统技术指标并绘制示例 K 线图。
- 指标分类：
  - 主图：EMA(5/10/20/60)、BOLL、SAR、Keltner Channel
  - 趋势：MACD、DMI、DMA、ADTM、EMA 斜率（5/10/20/60）、DDI、DPO
  - 超买超卖：CCI、KDJ、RSI、OSC
  - 能量：ARBR、PSY、VR
  - 摆动：ATR、Mass Index、SRMI
- 接口：
  - `TechnicalIndicatorCalculator(data, params).compute()` → 返回带所有指标列的 DataFrame。
  - 可选参数：`IndicatorParams` 控制周期、标准差倍数、ATR 倍数等。
- 使用示例：
  ```bash
  python scripts/technical_indicators.py
  ```

## secondary_signals.py
- 功能：基于一级指标计算二级买入信号，并生成带有半透明背景的信号图（展示布林带）。
- 信号定义（`compute_secondary_signals`）：
  - `signal_macd`：MACD DIF 上穿 DEA（金叉），且柱状图为正。
  - `signal_kdj`：K 上穿 D（金叉）且 K ≤ 70。
  - `signal_ema`：EMA20 > EMA60，且当日阳线包住 EMA20。
  - `signal_sar`：SAR 从 K 线上方翻至下方。
  - `signal_dmi`：PDI 上穿 MDI 或 ADX 上穿 ADXR。
  - `signal_adtm`：ADTM 上穿 ADTMMA 且 ADTM < 0.5。
  - `signal_ddi`：DDI 从负值转为正值。
  - `signal_dpo`：DPO 从负值转为正值。
  - `signal_osc`：OSC 上穿 OSCEMA。
  - `signal_srmi`：SRMI 从负值转为正值。
  - `signal_boll_close_break`：当日收盘价高于布林线上轨。
  - `signal_boll_open_break`：当日开盘价低于布林线下轨。
- 可视化（`plot_with_signals` + `save_chart_and_legend`）：
  - 默认绘制布林带（Mid/Up/Low），信号触发区域以彩色半透明背景高亮。
  - 触发信号的日期以彩色半透明背景高亮。
- 输出：
  - `outputs/secondary_signals_chart.png` — 主图（含布林带与信号高亮）。
  - `outputs/secondary_signals_legend.png` — 图例说明。
- 使用示例：
  ```bash
  python scripts/secondary_signals.py
  ```

> 注：二级信号仅在触发当日标记为 True，未进行有效期扩展；如需调整，可修改 `compute_secondary_signals`。

## trade_labels.py
- 功能：依据交易纪律计算中长期（20 日窗口、止损 -2×ATR、止盈 +3×ATR）与短期（5 日窗口、止损 -1×ATR、止盈 +1.5×ATR）胜/平/负标签（1/0/-1），并同时可视化长短两套策略的蜡烛图及 ATR。
- 特性：
  - 自动检查是否有足够的未来数据（长线 20，自然日；短线 5）。
  - 蜡烛图背景中胜为红、负为绿，平不高亮。
  - 底部子图展示 ATR。
- 使用示例：
  ```bash
  python scripts/trade_labels.py
  ```

> 提示：若需自定义窗口或止盈止损倍数，可调整脚本中的参数后重新运行。
