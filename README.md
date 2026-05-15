# A 股低估值股票分析

这个小工具会做两件事：

1. 抓取全市场 A 股当日指标。
2. 按估值规则筛出低估值股票。
3. 支持按行业、银行、地产、央国企筛选。
4. 支持一个简单网页界面，一键刷新结果。

程序会自动按下面的顺序尝试免费数据源：

1. 东方财富直连
2. 腾讯免费接口
3. AkShare 的新浪免费接口

默认使用的字段包括：

- `代码`
- `名称`
- `风险分数 / 风险等级`
- `建议买入下限 / 建议买入上限 / 止损参考价 / 目标参考价 / 买入建议状态`
- `股息率 / ROE / 营收增长率 / 净利润增长率 / 资产负债率 / 毛利率 / 净利率`
- `K值 / D值 / J值`
- `KDJ金叉`
- `DIF / DEA / MACD`
- `MA / RSI / BOLL / ATR / 回撤 / 波动率 / 相对沪深300强度`
- `最新价`
- `涨跌幅`
- `换手率`
- `市盈率-动态`
- `市净率`
- `总市值`
- `流通市值`
- `60日涨跌幅`
- `年初至今涨跌幅`

## 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

## 运行

```bash
python3 stock_analysis.py
```

按分类筛选：

```bash
python3 stock_analysis.py --category 银行
python3 stock_analysis.py --category 地产
python3 stock_analysis.py --category 央国企
python3 stock_analysis.py --industry-keyword 白酒
```

筛选当日 KDJ 出现金叉的股票：

```bash
python3 stock_analysis.py --require-kdj-gold-cross
python3 stock_analysis.py --category 银行 --require-kdj-gold-cross
```

运行后会在 `output/` 目录生成两个文件：

- `a_share_daily_metrics_YYYYMMDD.csv`：全市场每日指标
- `filtered_market_YYYYMMDD.csv`：本次筛选后的市场结果
- `low_valuation_stocks_YYYYMMDD.csv`：低估值股票结果
- `a_share_daily_metrics_history.csv`：自动累计的历史总表

其中 `a_share_daily_metrics_YYYYMMDD.csv` 始终导出全部 A 股，不会因为你选择了银行、地产或 KDJ 金叉而变成局部结果。

## 默认筛选规则

- 动态市盈率 `> 0` 且 `<= 15`
- 市净率 `> 0` 且 `<= 1.5`
- 总市值 `>= 30 亿元`
- 默认排除 `ST`

低估值结果会根据 `市盈率-动态` 和 `市净率` 的综合排名生成一个 `估值分数`，分数越低说明估值越便宜。

## 常用参数

```bash
python3 stock_analysis.py --max-pe 12 --max-pb 1.2 --min-market-cap-yi 50 --top-n 100
```

如果你想保留 `ST` 股票：

```bash
python3 stock_analysis.py --include-st
```

如果你的本机网络必须走系统代理：

```bash
python3 stock_analysis.py --use-env-proxy
```

## 网页界面

启动服务：

```bash
python3 web_app.py
```

然后打开：

```text
http://127.0.0.1:8501
```

页面里可以直接：

- 选择 `全部 / 银行 / 地产 / 央国企`
- 输入行业关键字
- 调整 `PE / PB / 市值 / Top N`
- 设置 `风险分数`、`RSI6`、`距52周新高`、`60日最大回撤`
- 勾选 `站上 MA20 / 均线多头 / 强于沪深300`
- 勾选 `仅显示可关注/低吸区`
- 勾选 `仅保留当日 KDJ 金叉`
- 点击按钮刷新全市场和低估值结果

## 注意

- 数据源支持东方财富直连、腾讯免费接口、AkShare 的新浪免费接口自动切换。
- 这里的“每日指标”是你运行脚本当时抓到的当日市场快照；如果你每天运行一次，就会自动累积到历史总表里。
- 同一天内重复运行时，会覆盖当天的 `filtered_market_YYYYMMDD.csv` 和 `low_valuation_stocks_YYYYMMDD.csv`，保留最新一次筛选结果。
- 当你启用 `KDJ` 相关统计时，程序会同时从同一份历史日线里计算并导出 `DIF / DEA / MACD`。
- `风险分数` 是一个 `0-100` 的综合分，越高代表短线波动和交易风险越大；当前主要综合了 `振幅`、`换手率`、`60日波动`、`总市值`、`估值水平` 和 `ST` 标记。
- `建议买入价` 采用“估值锚 + 技术锚 + 风险折扣”的方式生成，输出建议买入区间、止损参考价、目标参考价和当前状态。
- `建议买入价` 支持 `激进 / 标准 / 保守` 三档风格：激进更愿意靠近当前趋势买入，保守会给出更低、更窄的买入区间。
- 基本面增强字段会尽量用免费数据源补齐；如果当天免费源不可用，对应列会留空，但不会影响主流程运行。
- 如果走到腾讯或新浪兜底，少数字段可能为空，具体取决于免费源实际返回内容。
- `央国企` 标签是基于免费“实际控制人”数据和公司信息做的识别，适合日常筛选，但不应替代正式股权穿透判断。
- `KDJ 金叉` 需要额外抓取个股近几个月日线数据，第一次跑会比普通筛选慢一些，结果会缓存到本地，后续同日再次筛选会快很多。
- 低估值只是量化初筛，不代表可以直接买入，最好再结合行业、盈利质量、负债率、现金流和分红情况继续研究。
