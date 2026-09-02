# 数据准备

本项目使用 GEFCom2014 Solar Track。原始数据可公开获取，但来源页面没有给出明确的再分发许可证，因此原始压缩包、CSV、中间数据和处理后的数组均不进入公开仓库。仓库只保留下载说明、校验值、处理代码、汇总指标和自行生成的图表。

GEFCom2014 数据不受本仓库 MIT 许可证覆盖。数据使用者应从原始来源获取数据，并自行确认和遵守其适用条款。

## 1. 获取官方数据

优先使用下列来源：

- [IEEE PES 数据共享目录](https://ieee-pes-data-sharing.org/datasets/detail/b1680aa5-a4b8-4423-8760-e509094cacec)
- [GEFCom2014 论文及补充材料](https://doi.org/10.1016/j.ijforecast.2016.02.001)

本项目核验过的官方论文附件文件信息：

```text
文件名：GEFCom2014_official.zip
大小：126,360,077 bytes
SHA-256：D68D957270EDD93B26A37D0F9B5E901F942ABDF34C75EACBE14E417BEB16E154
```

下载并解压后，确保规范输入文件位于：

```text
data/raw/gefcom2014/solar/Solar/Task 15/predictors15.csv
```

如原始压缩包目录结构不同，只需整理到上述相对路径；不要修改配置中的时间切分或字段定义来适配错误文件。

## 2. 生成处理数据

在仓库根目录运行：

```powershell
python scripts/preprocess_data.py --config configs/day_ahead.yaml
```

脚本将生成：

```text
data/interim/gefcom2014_solar_hourly.csv.gz
data/processed/day_ahead/train.npz
data/processed/day_ahead/validation.npz
data/processed/day_ahead/test.npz
data/processed/day_ahead/nwp_scaler.json
data/processed/day_ahead/metadata.json
```

预处理固定执行累计 NWP 转逐小时非负增量、24→24 窗口构造、UTC 时间顺序切分，以及仅用训练集拟合 NWP 标准化统计量。完整证据见 [数据来源记录](../docs/data_source_record.md) 和 [数据审计报告](../docs/data_audit.md)。

## 3. 本地数据边界

- `data/raw/`、`data/interim/` 和 `data/processed/` 已由 `.gitignore` 排除。
- 不要把原始附件或处理后数组提交到公开仓库。
- 三个 `.gitkeep` 文件只用于保留空目录结构。
