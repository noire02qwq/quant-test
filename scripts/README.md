# Scripts Overview

本目录包含与数据准备和技术指标可视化相关的脚本。各脚本均依赖本仓库根目录下的 `data/qlib_us_selected/qlib_data` 数据集（可通过 `build_us_dataset.py` 构建）。使用前请先执行 `qlib.init` 或运行脚本内置的入口。

## build_us_dataset.py
- 功能：下载指定标的的 Yahoo Finance 日线数据，转换为 Qlib 二进制格式。
- 核心步骤：
  1. 补丁 Qlib Yahoo collector 的标的列表。
  2. 调用 `Run.download_data` 拉取 CSV。
  3. 使用 `DumpDataAll` 转换为二进制数据并生成行业分组清单。
- 使用示例：
  ```bash
  python scripts/build_us_dataset.py --start 2010-01-01 --end 2025-09-30 --force-refresh
  ```

## technical_indicators.py
- 功能：在 Qlib 数据基础上计算传统技术指标（EMA/BOLL/SAR/KC、MACD/DMI/DMA/ADTM、DDI/DPO/OSC、CCI/KDJ/RSI、ARBR/PSY/VR、ATR、Mass Index、SRMI 等）并绘制示例图。
- 数据接口：封装在 `TechnicalIndicatorCalculator` 中，调用 `compute()` 返回指标 DataFrame。
- 可视化：`plot_kline_with_indicators` 用于叠加 EMA、MACD、KDJ；脚本入口展示 TSM 日线示例。
- 使用示例：
  ```bash
  python scripts/technical_indicators.py
  ```

## secondary_signals.py
- 功能：基于 `technical_indicators.py` 输出的一级指标计算二级买入信号（MACD、KDJ、EMA、SAR、DMI、ADTM、DDI、DPO、OSC、SRMI），并绘制包含高亮信号的蜡烛图。
- 输出：
  - `outputs/secondary_signals_chart.png` — 主图，包含 EMA 曲线及信号背景。
  - `outputs/secondary_signals_legend.png` — 图例说明。
- 使用示例：
  ```bash
  python scripts/secondary_signals.py
  ```

> 注：二级信号仅在触发当日为 True，未进行有效期扩展；如需扩展，可在 `compute_secondary_signals` 中调整逻辑。
