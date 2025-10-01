# Selected US Daily Dataset (Qlib Pipeline)

该脚本使用 **qlib 自带的 Yahoo 数据采集器** 与 **DumpDataAll 工具**，在本仓库内完成所选美股（2010 年至今）的日频数据构建。无需进入 qlib 仓库执行脚本。

## 先决条件
- 已安装 qlib（当前环境已满足）。
- 已按照 `scripts/data_collector/yahoo/requirements.txt` 安装依赖（本环境亦已安装）。
- 可选：为了避免旧文件干扰，初次运行可使用 `--force-refresh` 清理输出目录。

## 目录约定
- 原始 CSV：`data/qlib_us_selected/source/`
- Qlib 二进制数据：`data/qlib_us_selected/qlib_data/`
- 行业分组：生成于 `data/qlib_us_selected/qlib_data/instruments/`，包括 `tech.txt`、`financial.txt`、`industrial.txt`。

## 构建数据集
```bash
python experiment-1001/build_us_dataset.py \
  --start 2010-01-01 \
  --end 2025-09-30 \
  --force-refresh
```
参数说明：
- `--start` / `--end`：设置采集时间范围；若不指定 `--end`，脚本默认使用当日。
- `--data-root`：自定义输出目录，默认 `data/qlib_us_selected`。
- `--max-workers`：并发线程数（下载与 dump 用）。
- `--delay`：Yahoo 请求间隔（秒）。
- `--force-refresh`：清空输出目录后重新构建。

## 使用数据
```python
import qlib
from qlib.constant import REG_US
from qlib.data import D

qlib.init(provider_uri="data/qlib_us_selected/qlib_data", region=REG_US)
frame = D.features(["TSM"], ["$open", "$high", "$low", "$close", "$volume"],
                   start_time="2025-01-01", end_time="2025-09-30", freq="day")
print(frame.tail())
```
行业分组文件可在策略/回测配置中直接引用，如 `instruments/tech.txt`。

## 更新建议
- 每次需要刷新数据时重新运行脚本，建议指定最新交易日作为 `--end`，并视情况开启 `--force-refresh`。
- 若仅增量补数据，也可保持默认目录并省略 `--force-refresh`，脚本会覆盖已有 CSV 并重新生成 Qlib 数据。
