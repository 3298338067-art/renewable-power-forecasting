# GEFCom2014 Solar 数据来源记录

## 1. 数据集身份

- 数据集名称：GEFCom2014 Solar Track
- 竞赛名称：Global Energy Forecasting Competition 2014
- 发布者：GEFCom2014 组织方
- 数据类型：真实光伏功率与数值天气预报数据
- 时间分辨率：1 小时
- 空间范围：澳大利亚同一区域内的 3 个相邻光伏场，精确位置未公开

## 2. 可靠来源

### IEEE PES 数据共享目录

- 页面：https://ieee-pes-data-sharing.org/datasets/detail/b1680aa5-a4b8-4423-8760-e509094cacec
- 页面状态：标记为 Public access
- 页面所列许可证：N/A

### Elsevier 官方论文附件

- 下载地址：https://ars.els-cdn.com/content/image/1-s2.0-S0169207016000133-mmc1.zip
- 对应论文：https://doi.org/10.1016/j.ijforecast.2016.02.001

本项目使用的是 Elsevier 官方论文附件，不使用来源不明的网盘副本。

## 3. 正式引用

Hong, T., Pinson, P., Fan, S., Zareipour, H., Troccoli, A., & Hyndman, R. J. (2016). Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond. *International Journal of Forecasting*, 32(3), 896-913. https://doi.org/10.1016/j.ijforecast.2016.02.001

## 4. 时间标准核验

本地官方包中的 `Solar/Instructions.txt` 只说明 12 个 NWP 字段及单位，没有单独声明时区。为避免从功率曲线形状反推当地钟表时间，本项目进一步核对了使用同一 Task 15 数据的公开论文：

- Marques et al. (2019) 明确将完整功率时段写为 `2012-04-01 01:00` 至 `2014-07-01 00:00 UTC`，并说明 ECMWF 预报在每日 `00:00 UTC` 发布未来 24 小时结果。
- 数据对应澳大利亚三个场站，但精确位置未公开，无法可靠确定当地时区及夏令时规则。

核验来源：[Improving Prediction Intervals Using Measured Solar Power with a Multi-Objective Approach](https://www.mdpi.com/1996-1073/12/24/4713)。

因此，本项目采用以下固定规则：

1. 原始时间、预测起点、目标时刻、数据连接、切分和日历特征全部保持 UTC。
2. 数组字段显式命名为 `origin_timestamp_utc` 和 `target_timestamp_utc`。
3. 图表横轴明确标注 UTC。
4. 不将 UTC 小时直接解释为场站当地白天或夜间，也不做无依据的本地时区换算。

## 5. 本地文件记录

- 原始附件：`data/raw/GEFCom2014_official.zip`
- 文件大小：126,360,077 bytes
- SHA-256：`D68D957270EDD93B26A37D0F9B5E901F942ABDF34C75EACBE14E417BEB16E154`
- 太阳能赛道归档：`data/raw/gefcom2014/GEFCom2014-S_V2.zip`
- 解压目录：`data/raw/gefcom2014/solar/Solar/`

## 6. 使用与分发限制

IEEE PES 页面显示该数据可以公开访问，但许可证字段为 `N/A`，因此不能把“可公开下载”直接等同于“允许任意重新分发”。本项目采用以下保守规则：

1. 原始压缩包和解压后的 CSV 只保存在本地，不上传 GitHub。
2. GitHub 仓库只提供下载说明、数据字段说明、处理代码和校验值。
3. 报告和 README 中正式引用原论文。
4. 公开展示仅使用自行生成的汇总统计和图表，不重新打包发布原始数据。

## 7. 来源核验结论

该数据来自 GEFCom2014 官方论文附件，并由 IEEE PES 数据共享目录收录，来源可靠，适合用于本项目。需要保留的唯一限制是：由于未给出明确开源许可证，原始文件不在公开仓库中重新分发。
