# Selected US Daily Dataset

## 本次工作内容
- 编写 `scripts/build_selected_us_dataset.py`，从 Yahoo Finance 拉取所选 37 支美国股票 2010-01-01 至 2025-10-01 的日频数据，并转换成 Qlib 所需的二进制格式。
- 生成的 CSV 临时文件存放在 `~/.qlib/stock_data/selected_us_daily/csv`，Qlib 数据目录为 `~/.qlib/qlib_data/selected_us_daily`。
- 自动根据行业分组输出 `tech.txt`、`financial.txt`、`industrial.txt` 等股票列表，便于回测或研究时直接引用。

## 使用方法
- 初次获取该数据集：
  1. 在本仓库根目录执行 `python scripts/build_selected_us_dataset.py build --start_date 2010-01-01 --end_date 2025-10-01`
     （如需使用不同时间范围，可自行调整参数）。
  2. 该命令会自动在 `~/.qlib/stock_data/selected_us_daily/csv` 生成原始 CSV，并在 `~/.qlib/qlib_data/selected_us_daily` 生成 Qlib 数据目录。
  3. 初次构建完成后即可按下述方式加载数据。
- 初始化 Qlib 时指向新的数据目录：
  ```python
  import qlib
  qlib.init(provider_uri="~/.qlib/qlib_data/selected_us_daily", region="us")
  ```
- 之后即可通过 `from qlib.data import D` 获取行情或因子，例如：
  ```python
  from qlib.data import D
  data = D.features(["AAPL"], ["$close"], start_time="2024-01-01", end_time="2025-10-01", freq="day")
  ```
- 行业分组文件位于 `~/.qlib/qlib_data/selected_us_daily/instruments/`，可在回测配置中引用对应名单。

## 后续更新建议
- 每个交易日收盘后执行一次：
  ```bash
  python scripts/build_selected_us_dataset.py build --end_date <最新交易日>
  ```
  该命令会重新下载最新数据并覆盖现有 Qlib 目录。
- 如需仅补充数据，可在命令行指定不同的 `start_date`/`end_date` 或调整行业分组列表。
- 建议将上述命令加入定时任务（如 cron），确保数据持续更新。
