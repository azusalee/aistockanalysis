#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import sys

import akshare as ak
from akshare.utils import demjson
import pandas as pd
import requests


NUMERIC_COLUMNS = [
    "最新价",
    "涨跌幅",
    "涨跌额",
    "成交量",
    "成交额",
    "振幅",
    "最高",
    "最低",
    "今开",
    "昨收",
    "量比",
    "换手率",
    "市盈率-动态",
    "市净率",
    "总市值",
    "流通市值",
    "涨速",
    "5分钟涨跌",
    "60日涨跌幅",
    "年初至今涨跌幅",
]

OUTPUT_COLUMNS = [
    "日期",
    "代码",
    "名称",
    "行业",
    "央国企标签",
    "问题股标签",
    "行业PE分位(%)",
    "行业PB分位(%)",
    "行业估值分位(%)",
    "行业便宜度排名",
    "行业强度排名",
    "行业稳健度排名",
    "风险分数",
    "风险等级",
    "波动预警",
    "20日波动预警",
    "60日波动预警",
    "估值因子分",
    "趋势因子分",
    "质量因子分",
    "风险调整分",
    "组合总分",
    "建议仓位",
    "策略标签",
    "建议买入下限",
    "建议买入上限",
    "建议买入价距离(%)",
    "追高提示",
    "止损参考价",
    "止损比例(%)",
    "目标参考价",
    "盈亏比",
    "买入建议状态",
    "买入风格",
    "股息率(%)",
    "ROE(%)",
    "营收增长率(%)",
    "净利润增长率(%)",
    "资产负债率(%)",
    "毛利率(%)",
    "净利率(%)",
    "商誉占净资产比(%)",
    "MA5",
    "MA10",
    "MA20",
    "MA60",
    "MA120",
    "MA250",
    "均线多头排列",
    "RSI6",
    "RSI12",
    "RSI24",
    "BOLL中轨",
    "BOLL上轨",
    "BOLL下轨",
    "ATR14",
    "成交量MA5",
    "成交量MA10",
    "5日涨跌幅",
    "10日涨跌幅",
    "20日涨跌幅",
    "120日涨跌幅",
    "250日涨跌幅",
    "BIAS20",
    "BIAS60",
    "距52周新高(%)",
    "距52周新低(%)",
    "60日最大回撤(%)",
    "120日最大回撤(%)",
    "250日最大回撤(%)",
    "相对沪深300强度20日",
    "相对沪深300强度60日",
    "相对沪深300强度120日",
    "20日波动率(%)",
    "60日波动率(%)",
    "K值",
    "D值",
    "J值",
    "KDJ金叉",
    "RSI低位回升",
    "DIF",
    "DEA",
    "MACD",
    "MACD翻红",
    "放量突破",
    "最新价",
    "涨跌幅",
    "成交额",
    "换手率",
    "市盈率-动态",
    "市净率",
    "总市值",
    "流通市值",
    "60日涨跌幅",
    "年初至今涨跌幅",
    "估值分数",
]

EASTMONEY_URL = "https://82.push2.eastmoney.com/api/qt/clist/get"
EASTMONEY_FIELDS = [
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "f12",
    "f13",
    "f14",
    "f15",
    "f16",
    "f17",
    "f18",
    "f20",
    "f21",
    "f23",
    "f24",
    "f25",
    "f22",
    "f11",
    "f62",
    "f128",
    "f136",
    "f115",
    "f152",
]
EASTMONEY_COLUMN_NAMES = [
    "index",
    "_",
    "最新价",
    "涨跌幅",
    "涨跌额",
    "成交量",
    "成交额",
    "振幅",
    "换手率",
    "市盈率-动态",
    "量比",
    "5分钟涨跌",
    "代码",
    "_unused_1",
    "名称",
    "最高",
    "最低",
    "今开",
    "昨收",
    "总市值",
    "流通市值",
    "涨速",
    "市净率",
    "60日涨跌幅",
    "年初至今涨跌幅",
    "_unused_2",
    "_unused_3",
    "_unused_4",
    "_unused_5",
]
QQ_STOCK_URL = "https://proxy.finance.qq.com/cgi/cgi-bin/rank/hs/getBoardRankList"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
INDUSTRY_CACHE_PATH = CACHE_DIR / "industry_lookup.json"
FUNDAMENTAL_CACHE_PATH = CACHE_DIR / "fundamental_lookup.json"
TECHNICAL_WINDOWS = [5, 10, 20, 60, 120, 250]
HISTORY_LOOKBACK_DAYS = 520
BENCHMARK_SYMBOL = "sh000300"


@dataclass
class FilterConfig:
    max_pe: float
    max_pb: float
    min_market_cap_yi: float
    exclude_st: bool
    top_n: int
    category: str = "全部"
    industry_keyword: str = ""
    require_kdj_gold_cross: bool = False
    kdj_window: int = 9
    max_risk_score: float | None = None
    min_rsi6: float | None = None
    max_rsi6: float | None = None
    require_above_ma20: bool = False
    require_bullish_alignment: bool = False
    max_distance_to_52w_high: float | None = None
    max_drawdown_60: float | None = None
    require_relative_strength_positive: bool = False
    buy_style: str = "标准"
    strategy_preset: str = "全部"
    watch_only: bool = False
    low_absorption_only: bool = False
    exclude_problem_stocks: bool = False
    min_turnover_rate: float | None = None
    min_amount_yi: float | None = None


def apply_st_name_filter(df: pd.DataFrame, exclude_st: bool) -> pd.DataFrame:
    filtered = df.copy()
    if not exclude_st or "名称" not in filtered.columns:
        return filtered
    name_series = filtered["名称"].astype(str)
    return filtered[~name_series.str.contains(r"\*?ST", case=False, regex=True, na=False)].copy()


def normalize_stock_code(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.extract(r"(\d{6})", expand=False)
        .fillna(series.astype(str))
    )


def build_eastmoney_params(page_number: int, page_size: int) -> dict[str, str]:
    return {
        "pn": str(page_number),
        "pz": str(page_size),
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f12",
        "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
        "fields": ",".join(EASTMONEY_FIELDS),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="抓取 A 股全市场每日指标，并筛选低估值股票。"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="导出目录，默认 output",
    )
    parser.add_argument(
        "--max-pe",
        type=float,
        default=15.0,
        help="低估值筛选的最大动态市盈率，默认 15",
    )
    parser.add_argument(
        "--max-pb",
        type=float,
        default=1.5,
        help="低估值筛选的最大市净率，默认 1.5",
    )
    parser.add_argument(
        "--min-market-cap-yi",
        type=float,
        default=30.0,
        help="低估值筛选的最小总市值（亿元），默认 30",
    )
    parser.add_argument(
        "--include-st",
        action="store_true",
        help="默认会排除 ST 股票；传入此参数则保留",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="低估值结果最多导出多少只，默认 50",
    )
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="默认禁用系统代理；传入此参数后允许请求继承当前环境代理",
    )
    parser.add_argument(
        "--category",
        default="全部",
        choices=["全部", "银行", "地产", "央国企"],
        help="分类筛选，默认 全部",
    )
    parser.add_argument(
        "--industry-keyword",
        default="",
        help="按行业关键字筛选，比如 银行、房地产、白酒",
    )
    parser.add_argument(
        "--require-kdj-gold-cross",
        action="store_true",
        help="仅保留当日 KDJ 出现金叉的股票",
    )
    parser.add_argument(
        "--kdj-window",
        type=int,
        default=9,
        help="KDJ 计算窗口，默认 9",
    )
    parser.add_argument(
        "--buy-style",
        default="标准",
        choices=["激进", "标准", "保守"],
        help="建议买入价风格，默认 标准",
    )
    parser.add_argument(
        "--strategy-preset",
        default="全部",
        choices=["全部", "红利策略", "成长策略", "反转策略", "趋势策略"],
        help="策略预设筛选，默认 全部",
    )
    parser.add_argument(
        "--watch-only",
        action="store_true",
        help="仅保留买入建议状态为 可关注 的股票",
    )
    parser.add_argument(
        "--low-absorption-only",
        action="store_true",
        help="仅保留买入建议状态为 进入低吸区 的股票",
    )
    parser.add_argument(
        "--exclude-problem-stocks",
        action="store_true",
        help="排除问题股：连续亏损、高负债、低毛利、商誉高、ST、退市风险",
    )
    parser.add_argument(
        "--min-turnover-rate",
        type=float,
        default=None,
        help="最小换手率(%%)，低于该值的股票会被过滤",
    )
    parser.add_argument(
        "--min-amount-yi",
        type=float,
        default=None,
        help="最小成交额(亿元)，低于该值的股票会被过滤",
    )
    return parser.parse_args()


def request_eastmoney_page(
    session: requests.Session,
    page_number: int,
    page_size: int = 100,
    max_retries: int = 3,
) -> dict:
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = session.get(
                EASTMONEY_URL,
                params=build_eastmoney_params(page_number=page_number, page_size=page_size),
                timeout=20,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Referer": "https://quote.eastmoney.com/",
                    "Accept": "application/json,text/plain,*/*",
                },
            )
            response.raise_for_status()
            data = response.json()
            if not data.get("data") or not data["data"].get("diff"):
                raise RuntimeError(f"第 {page_number} 页返回为空")
            return data
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"东方财富接口第 {page_number} 页请求失败") from last_error


def normalize_eastmoney_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df.columns = EASTMONEY_COLUMN_NAMES
    raw_df.rename(columns={"index": "序号"}, inplace=True)
    raw_df["代码"] = normalize_stock_code(raw_df["代码"])
    normalized = raw_df[
        [
            "序号",
            "代码",
            "名称",
            "最新价",
            "涨跌幅",
            "涨跌额",
            "成交量",
            "成交额",
            "振幅",
            "最高",
            "最低",
            "今开",
            "昨收",
            "量比",
            "换手率",
            "市盈率-动态",
            "市净率",
            "总市值",
            "流通市值",
            "涨速",
            "5分钟涨跌",
            "60日涨跌幅",
            "年初至今涨跌幅",
        ]
    ]
    for column in NUMERIC_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def fetch_market_snapshot_direct(use_env_proxy: bool) -> pd.DataFrame:
    session = requests.Session()
    session.trust_env = use_env_proxy

    first_page = request_eastmoney_page(session=session, page_number=1)
    total_count = int(first_page["data"]["total"])
    first_page_rows = first_page["data"]["diff"]
    page_size = len(first_page_rows)
    total_pages = (total_count + page_size - 1) // page_size

    frames = [pd.DataFrame(first_page_rows)]
    for page_number in range(2, total_pages + 1):
        page_data = request_eastmoney_page(
            session=session,
            page_number=page_number,
            page_size=page_size,
        )
        frames.append(pd.DataFrame(page_data["data"]["diff"]))
        time.sleep(0.2)

    merged = pd.concat(frames, ignore_index=True)
    merged.sort_values(by=["f3"], ascending=False, inplace=True, ignore_index=True)
    merged.reset_index(inplace=True)
    merged["index"] = merged["index"].astype(int) + 1
    return normalize_eastmoney_dataframe(merged)


def fetch_market_snapshot_with_akshare() -> pd.DataFrame:
    try:
        df = ak.stock_zh_a_spot_em()
    except Exception as exc:  # pragma: no cover - depends on external network
        raise RuntimeError("AkShare 抓取失败") from exc

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "代码" in df.columns:
        df["代码"] = normalize_stock_code(df["代码"])

    return df


def request_qq_page(
    session: requests.Session,
    offset: int,
    count: int = 200,
    max_retries: int = 3,
) -> dict:
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = session.get(
                QQ_STOCK_URL,
                params={
                    "_appver": "11.17.0",
                    "board_code": "aStock",
                    "sort_type": "price",
                    "direct": "down",
                    "offset": str(offset),
                    "count": str(count),
                },
                timeout=20,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Referer": "https://stockapp.finance.qq.com/",
                    "Accept": "application/json,text/plain,*/*",
                },
            )
            response.raise_for_status()
            data = response.json()
            if "data" not in data or "rank_list" not in data["data"]:
                raise RuntimeError(f"腾讯接口 offset={offset} 返回结构异常: {data.get('msg')}")
            return data
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"腾讯接口 offset={offset} 请求失败") from last_error


def normalize_qq_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df = df.rename(
        columns={
            "code": "代码",
            "name": "名称",
            "zxj": "最新价",
            "zdf": "涨跌幅",
            "zd": "涨跌额",
            "turnover": "成交量",
            "volume": "成交额",
            "zf": "振幅",
            "lb": "量比",
            "hsl": "换手率",
            "pe_ttm": "市盈率-动态",
            "pn": "市净率",
            "zsz": "总市值",
            "ltsz": "流通市值",
            "speed": "涨速",
            "zdf_d60": "60日涨跌幅",
            "zdf_y": "年初至今涨跌幅",
        }
    )

    df["代码"] = normalize_stock_code(df["代码"])
    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # 腾讯市值字段单位是亿元，统一转成元，便于和其余数据源保持一致。
    if "总市值" in df.columns:
        df["总市值"] = df["总市值"] * 1e8
    if "流通市值" in df.columns:
        df["流通市值"] = df["流通市值"] * 1e8

    df["序号"] = range(1, len(df) + 1)
    df["日期"] = datetime.now().strftime("%Y-%m-%d")
    return df[
        [
            "序号",
            "代码",
            "名称",
            "最新价",
            "涨跌幅",
            "涨跌额",
            "成交量",
            "成交额",
            "振幅",
            "最高",
            "最低",
            "今开",
            "昨收",
            "量比",
            "换手率",
            "市盈率-动态",
            "市净率",
            "总市值",
            "流通市值",
            "涨速",
            "5分钟涨跌",
            "60日涨跌幅",
            "年初至今涨跌幅",
            "日期",
        ]
    ]


def fetch_market_snapshot_with_qq(use_env_proxy: bool) -> pd.DataFrame:
    session = requests.Session()
    session.trust_env = use_env_proxy

    first_page = request_qq_page(session=session, offset=0)
    total_count = int(first_page["data"]["total"])
    first_rows = first_page["data"]["rank_list"]
    page_size = len(first_rows)
    if page_size == 0:
        raise RuntimeError("腾讯接口第一页返回为空")

    frames = [pd.DataFrame(first_rows)]
    for offset in range(page_size, total_count, page_size):
        page_data = request_qq_page(session=session, offset=offset, count=page_size)
        frames.append(pd.DataFrame(page_data["data"]["rank_list"]))
        time.sleep(0.2)

    merged = pd.concat(frames, ignore_index=True)
    return normalize_qq_dataframe(merged)


def fetch_market_snapshot(use_env_proxy: bool) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    try:
        df = fetch_market_snapshot_direct(use_env_proxy=use_env_proxy)
        source_name = "东方财富直连"
    except Exception as exc:  # pragma: no cover - depends on external network
        errors.append(f"东方财富直连失败: {exc}")
        try:
            df = fetch_market_snapshot_with_qq(use_env_proxy=use_env_proxy)
            source_name = "腾讯免费接口"
        except Exception as qq_exc:  # pragma: no cover - depends on external network
            errors.append(f"腾讯免费接口失败: {qq_exc}")
            try:
                df = fetch_market_snapshot_with_akshare()
                source_name = "AkShare-新浪"
            except Exception as ak_exc:  # pragma: no cover - depends on external network
                errors.append(f"AkShare-新浪回退失败: {ak_exc}")
                raise RuntimeError(
                    "抓取 A 股数据失败。请检查网络连接，或稍后重试。\n"
                    + "\n".join(errors)
                ) from ak_exc

    df = df.copy()
    if "日期" not in df.columns:
        df["日期"] = datetime.now().strftime("%Y-%m-%d")
    return df, source_name


def get_technical_cache_path(window: int) -> Path:
    return CACHE_DIR / f"technical_signals_tx_v3_{datetime.now().strftime('%Y%m%d')}_w{window}.json"


def load_technical_cache(window: int) -> dict[str, dict]:
    cache_path = get_technical_cache_path(window)
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_technical_cache(window: int, cache: dict[str, dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    get_technical_cache_path(window).write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def infer_tx_symbol(code: str) -> str:
    normalized_code = str(code)
    if normalized_code.startswith(("5", "6", "9")):
        return f"sh{normalized_code}"
    if normalized_code.startswith(("4", "8")):
        return f"bj{normalized_code}"
    return f"sz{normalized_code}"


def fetch_tx_hist_df(
    symbol: str,
    use_env_proxy: bool,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    session = requests.Session()
    session.trust_env = use_env_proxy
    range_start = int(start_date[:4])
    range_end = int(end_date[:4]) + 1
    frames: list[pd.DataFrame] = []

    for year in range(range_start, range_end):
        response = session.get(
            "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get",
            params={
                "_var": f"kline_day{year}",
                "param": f"{symbol},day,{year}-01-01,{year + 1}-12-31,640,",
                "r": "0.8205512681390605",
            },
            timeout=20,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Referer": "https://gu.qq.com/",
                "Accept": "*/*",
            },
        )
        response.raise_for_status()
        data_text = response.text
        data_json = demjson.decode(data_text[data_text.find("={") + 1 :])["data"][symbol]
        if "day" in data_json:
            temp_df = pd.DataFrame(data_json["day"])
        elif "hfqday" in data_json:
            temp_df = pd.DataFrame(data_json["hfqday"])
        else:
            temp_df = pd.DataFrame(data_json.get("qfqday", []))
        if not temp_df.empty:
            frames.append(temp_df.copy())

    if not frames:
        return pd.DataFrame()

    hist_df = pd.concat(frames, ignore_index=True)
    raw_column_count = hist_df.shape[1]
    if raw_column_count >= 8:
        hist_df = hist_df.iloc[:, :8].copy()
        hist_df.columns = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率"]
    elif raw_column_count == 7:
        hist_df = hist_df.iloc[:, :7].copy()
        hist_df.columns = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额"]
        hist_df["换手率"] = pd.NA
    elif raw_column_count == 6:
        hist_df = hist_df.iloc[:, :6].copy()
        hist_df.columns = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
        hist_df["成交额"] = pd.NA
        hist_df["换手率"] = pd.NA
    else:
        return pd.DataFrame()

    hist_df["日期"] = pd.to_datetime(hist_df["日期"], errors="coerce")
    for column in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率"]:
        hist_df[column] = pd.to_numeric(hist_df[column], errors="coerce")
    hist_df = hist_df.drop_duplicates(subset=["日期"]).sort_values(by="日期")
    hist_df = hist_df[
        (hist_df["日期"] >= pd.to_datetime(start_date))
        & (hist_df["日期"] <= pd.to_datetime(end_date))
    ].reset_index(drop=True)
    return hist_df


def fetch_stock_hist_df(
    code: str,
    use_env_proxy: bool,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    return fetch_tx_hist_df(
        symbol=infer_tx_symbol(code),
        use_env_proxy=use_env_proxy,
        start_date=start_date,
        end_date=end_date,
    )


def get_return_value(close_series: pd.Series, periods: int) -> float | None:
    if len(close_series) <= periods:
        return None
    value = close_series.pct_change(periods=periods).iloc[-1] * 100
    return round(float(value), 2) if pd.notna(value) else None


def get_latest_rolling_mean(series: pd.Series, window: int) -> float | None:
    if series.dropna().shape[0] < window:
        return None
    value = series.rolling(window=window, min_periods=window).mean().iloc[-1]
    return round(float(value), 2) if pd.notna(value) else None


def calculate_rsi(close_series: pd.Series, window: int) -> float | None:
    if close_series.dropna().shape[0] < window + 1:
        return None
    delta = close_series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    if pd.isna(rsi.iloc[-1]) and avg_loss.iloc[-1] == 0 and avg_gain.iloc[-1] > 0:
        return 100.0
    value = rsi.iloc[-1]
    return round(float(value), 2) if pd.notna(value) else None


def calculate_atr(high_series: pd.Series, low_series: pd.Series, close_series: pd.Series) -> float | None:
    if close_series.dropna().shape[0] < 15:
        return None
    prev_close = close_series.shift(1)
    tr_components = pd.concat(
        [
            high_series - low_series,
            (high_series - prev_close).abs(),
            (low_series - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr = true_range.rolling(window=14, min_periods=14).mean().iloc[-1]
    return round(float(atr), 2) if pd.notna(atr) else None


def calculate_max_drawdown(close_series: pd.Series, window: int) -> float | None:
    if close_series.dropna().shape[0] < 2:
        return None
    series = close_series.dropna().tail(window)
    if len(series) < 2:
        return None
    drawdown = (series / series.cummax() - 1) * 100
    return round(float(drawdown.min()), 2)


def calculate_volatility(close_series: pd.Series, window: int) -> float | None:
    if close_series.dropna().shape[0] < window + 1:
        return None
    daily_returns = close_series.pct_change().dropna()
    if len(daily_returns) < window:
        return None
    value = daily_returns.tail(window).std(ddof=0) * (252 ** 0.5) * 100
    return round(float(value), 2) if pd.notna(value) else None


def calculate_benchmark_return_map(
    use_env_proxy: bool,
    end_date: str,
) -> dict[int, float]:
    start_date = (datetime.now() - timedelta(days=HISTORY_LOOKBACK_DAYS)).strftime(
        "%Y%m%d"
    )
    try:
        hist_df = fetch_tx_hist_df(
            symbol=BENCHMARK_SYMBOL,
            use_env_proxy=use_env_proxy,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception:
        return {}

    close_series = hist_df["收盘"].dropna()
    return_map: dict[int, float] = {}
    for window in [20, 60, 120]:
        value = get_return_value(close_series, window)
        if value is not None:
            return_map[window] = value
    return return_map


def calculate_kdj_signal(hist_df: pd.DataFrame, window: int) -> dict | None:
    if hist_df.empty or len(hist_df) < window + 1:
        return None

    df = hist_df[["日期", "收盘", "最高", "最低"]].dropna().copy()
    if len(df) < window + 1:
        return None

    low_n = df["最低"].rolling(window=window, min_periods=window).min()
    high_n = df["最高"].rolling(window=window, min_periods=window).max()
    denominator = (high_n - low_n).replace(0, pd.NA)
    rsv = ((df["收盘"] - low_n) / denominator * 100).fillna(50)

    k_values: list[float] = []
    d_values: list[float] = []
    k_prev = 50.0
    d_prev = 50.0
    for value in rsv.tolist():
        k_prev = 2 / 3 * k_prev + 1 / 3 * float(value)
        d_prev = 2 / 3 * d_prev + 1 / 3 * k_prev
        k_values.append(k_prev)
        d_values.append(d_prev)

    df["K值"] = k_values
    df["D值"] = d_values
    df["J值"] = 3 * df["K值"] - 2 * df["D值"]
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    gold_cross = previous["K值"] <= previous["D值"] and latest["K值"] > latest["D值"]
    return {
        "K值": round(float(latest["K值"]), 2),
        "D值": round(float(latest["D值"]), 2),
        "J值": round(float(latest["J值"]), 2),
        "KDJ金叉": "是" if gold_cross else "否",
    }


def calculate_macd_signal(hist_df: pd.DataFrame) -> dict | None:
    if hist_df.empty or len(hist_df) < 35:
        return None

    df = hist_df[["日期", "收盘"]].dropna().copy()
    if len(df) < 35:
        return None

    ema12 = df["收盘"].ewm(span=12, adjust=False).mean()
    ema26 = df["收盘"].ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = (dif - dea) * 2
    latest = df.iloc[-1]
    latest_index = latest.name
    return {
        "DIF": round(float(dif.loc[latest_index]), 4),
        "DEA": round(float(dea.loc[latest_index]), 4),
        "MACD": round(float(macd.loc[latest_index]), 4),
    }


def build_empty_technical_signal() -> dict[str, float | str | None]:
    return {
        "MA5": None,
        "MA10": None,
        "MA20": None,
        "MA60": None,
        "MA120": None,
        "MA250": None,
        "均线多头排列": "",
        "RSI6": None,
        "RSI12": None,
        "RSI24": None,
        "BOLL中轨": None,
        "BOLL上轨": None,
        "BOLL下轨": None,
        "ATR14": None,
        "成交量MA5": None,
        "成交量MA10": None,
        "5日涨跌幅": None,
        "10日涨跌幅": None,
        "20日涨跌幅": None,
        "120日涨跌幅": None,
        "250日涨跌幅": None,
        "BIAS20": None,
        "BIAS60": None,
        "距52周新高(%)": None,
        "距52周新低(%)": None,
        "60日最大回撤(%)": None,
        "120日最大回撤(%)": None,
        "250日最大回撤(%)": None,
        "相对沪深300强度20日": None,
        "相对沪深300强度60日": None,
        "相对沪深300强度120日": None,
        "20日波动率(%)": None,
        "60日波动率(%)": None,
        "K值": None,
        "D值": None,
        "J值": None,
        "KDJ金叉": "否",
        "DIF": None,
        "DEA": None,
        "MACD": None,
    }


def calculate_price_technical_signal(
    hist_df: pd.DataFrame,
    benchmark_returns: dict[int, float],
    window: int,
) -> dict[str, float | str | None]:
    signal = build_empty_technical_signal()
    if hist_df.empty:
        return signal

    df = hist_df.copy()
    df = df.dropna(subset=["收盘", "最高", "最低"]).reset_index(drop=True)
    if df.empty:
        return signal

    close_series = df["收盘"]
    high_series = df["最高"]
    low_series = df["最低"]
    volume_series = pd.to_numeric(df.get("成交量"), errors="coerce")

    for ma_window in TECHNICAL_WINDOWS:
        signal[f"MA{ma_window}"] = get_latest_rolling_mean(close_series, ma_window)

    ma20 = signal["MA20"]
    ma60 = signal["MA60"]
    ma120 = signal["MA120"]
    latest_close = float(close_series.iloc[-1])
    if all(value is not None for value in [ma20, ma60, ma120]):
        signal["均线多头排列"] = (
            "是" if latest_close > ma20 > ma60 > ma120 else "否"
        )

    signal["RSI6"] = calculate_rsi(close_series, 6)
    signal["RSI12"] = calculate_rsi(close_series, 12)
    signal["RSI24"] = calculate_rsi(close_series, 24)

    if len(close_series.dropna()) >= 20:
        boll_mid = close_series.rolling(window=20, min_periods=20).mean().iloc[-1]
        boll_std = close_series.rolling(window=20, min_periods=20).std(ddof=0).iloc[-1]
        if pd.notna(boll_mid):
            signal["BOLL中轨"] = round(float(boll_mid), 2)
            signal["BOLL上轨"] = round(float(boll_mid + 2 * boll_std), 2)
            signal["BOLL下轨"] = round(float(boll_mid - 2 * boll_std), 2)

    signal["ATR14"] = calculate_atr(high_series, low_series, close_series)
    signal["成交量MA5"] = get_latest_rolling_mean(volume_series, 5)
    signal["成交量MA10"] = get_latest_rolling_mean(volume_series, 10)

    for return_window in [5, 10, 20, 120, 250]:
        signal[f"{return_window}日涨跌幅"] = get_return_value(close_series, return_window)

    if ma20 not in (None, 0):
        signal["BIAS20"] = round((latest_close / ma20 - 1) * 100, 2)
    if ma60 not in (None, 0):
        signal["BIAS60"] = round((latest_close / ma60 - 1) * 100, 2)

    high_52w = high_series.dropna().tail(250).max()
    low_52w = low_series.dropna().tail(250).min()
    if pd.notna(high_52w) and high_52w != 0:
        signal["距52周新高(%)"] = round((latest_close / float(high_52w) - 1) * 100, 2)
    if pd.notna(low_52w) and low_52w != 0:
        signal["距52周新低(%)"] = round((latest_close / float(low_52w) - 1) * 100, 2)

    signal["60日最大回撤(%)"] = calculate_max_drawdown(close_series, 60)
    signal["120日最大回撤(%)"] = calculate_max_drawdown(close_series, 120)
    signal["250日最大回撤(%)"] = calculate_max_drawdown(close_series, 250)
    signal["20日波动率(%)"] = calculate_volatility(close_series, 20)
    signal["60日波动率(%)"] = calculate_volatility(close_series, 60)

    for rs_window in [20, 60, 120]:
        stock_return = signal.get(f"{rs_window}日涨跌幅")
        benchmark_return = benchmark_returns.get(rs_window)
        if stock_return is not None and benchmark_return is not None:
            signal[f"相对沪深300强度{rs_window}日"] = round(
                float(stock_return) - float(benchmark_return),
                2,
            )

    kdj_signal = calculate_kdj_signal(hist_df=df, window=window)
    macd_signal = calculate_macd_signal(hist_df=df)
    if kdj_signal is not None:
        signal.update(kdj_signal)
    if macd_signal is not None:
        signal.update(macd_signal)
    return signal


def compute_kdj_for_code(
    code: str,
    use_env_proxy: bool,
    window: int,
    benchmark_returns: dict[int, float],
) -> tuple[str, dict]:
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=HISTORY_LOOKBACK_DAYS)).strftime("%Y%m%d")
    try:
        hist_df = fetch_stock_hist_df(
            code=code,
            use_env_proxy=use_env_proxy,
            start_date=start_date,
            end_date=end_date,
        )
        signal = calculate_price_technical_signal(
            hist_df=hist_df,
            benchmark_returns=benchmark_returns,
            window=window,
        )
    except Exception:
        signal = build_empty_technical_signal()
    return code, signal


def enrich_kdj_signals(
    df: pd.DataFrame, use_env_proxy: bool, window: int
) -> pd.DataFrame:
    enriched = df.copy()
    cache = load_technical_cache(window)
    codes = enriched["代码"].astype(str).tolist()
    missing_codes = [code for code in codes if code not in cache]
    benchmark_returns = calculate_benchmark_return_map(
        use_env_proxy=use_env_proxy,
        end_date=datetime.now().strftime("%Y%m%d"),
    )

    if missing_codes:
        max_workers = min(12, max(1, len(missing_codes)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    compute_kdj_for_code,
                    code,
                    use_env_proxy,
                    window,
                    benchmark_returns,
                ): code
                for code in missing_codes
            }
            for future in as_completed(futures):
                code, signal = future.result()
                cache[code] = signal
        save_technical_cache(window, cache)

    signal_df = pd.DataFrame(
        [{"代码": code, **cache.get(code, {})} for code in codes]
    )
    return enriched.merge(signal_df, on="代码", how="left")


def fetch_control_info() -> pd.DataFrame:
    try:
        control_df = ak.stock_hold_control_cninfo(symbol="全部")
    except Exception:
        return pd.DataFrame()
    if control_df.empty:
        return control_df
    control_df = control_df.copy()
    control_df["证券代码"] = normalize_stock_code(control_df["证券代码"])
    control_df["变动日期"] = pd.to_datetime(control_df["变动日期"], errors="coerce")
    control_df = control_df.sort_values(by=["证券代码", "变动日期"])
    control_df = control_df.drop_duplicates(subset=["证券代码"], keep="last")
    return control_df


def load_cached_industry_lookup() -> dict[str, str]:
    if not INDUSTRY_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(INDUSTRY_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cached_industry_lookup(lookup: dict[str, str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INDUSTRY_CACHE_PATH.write_text(
        json.dumps(lookup, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_cached_fundamental_lookup() -> dict[str, dict]:
    if not FUNDAMENTAL_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(FUNDAMENTAL_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cached_fundamental_lookup(lookup: dict[str, dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FUNDAMENTAL_CACHE_PATH.write_text(
        json.dumps(lookup, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_empty_fundamental_signal() -> dict[str, float | None]:
    return {
        "股息率(%)": None,
        "ROE(%)": None,
        "营收增长率(%)": None,
        "净利润增长率(%)": None,
        "资产负债率(%)": None,
        "毛利率(%)": None,
        "净利率(%)": None,
        "商誉占净资产比(%)": None,
    }


def normalize_metric_number(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "None", "nan", "--", "-"}:
        return None
    text = text.replace(",", "").replace("%", "")
    try:
        return float(text)
    except Exception:
        return None


def extract_metric_from_frame(df: pd.DataFrame, keywords: list[str]) -> float | None:
    if df is None or df.empty:
        return None

    for column in df.columns:
        if any(keyword in str(column) for keyword in keywords):
            series = df[column].dropna()
            if not series.empty:
                for value in series.iloc[::-1]:
                    numeric_value = normalize_metric_number(value)
                    if numeric_value is not None:
                        return numeric_value

    first_column = str(df.columns[0]) if len(df.columns) > 0 else ""
    if any(token in first_column for token in ["指标", "项目", "报告期", "日期"]):
        label_series = df.iloc[:, 0].astype(str)
        value_columns = list(df.columns[1:])
        for idx, label in label_series.items():
            if any(keyword in label for keyword in keywords):
                row = df.loc[idx, value_columns]
                for value in row.iloc[::-1]:
                    numeric_value = normalize_metric_number(value)
                    if numeric_value is not None:
                        return numeric_value

    return None


def fetch_fundamental_signal_for_code(code: str) -> dict[str, float | None]:
    signal = build_empty_fundamental_signal()
    frames: list[pd.DataFrame] = []

    for fetcher, kwargs in [
        ("stock_financial_analysis_indicator_em", {"symbol": code}),
        ("stock_financial_abstract_ths", {"symbol": code}),
    ]:
        try:
            frame = getattr(ak, fetcher)(**kwargs)
        except Exception:
            continue
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frames.append(frame)

    for frame in frames:
        metric_map = {
            "股息率(%)": ["股息率", "股利支付率"],
            "ROE(%)": ["净资产收益率", "ROE"],
            "营收增长率(%)": ["营业总收入同比增长", "营业收入同比增长", "营收同比增长"],
            "净利润增长率(%)": ["净利润同比增长", "归母净利润同比增长", "扣非净利润同比增长"],
            "资产负债率(%)": ["资产负债率"],
            "毛利率(%)": ["销售毛利率", "毛利率"],
            "净利率(%)": ["销售净利率", "净利率"],
            "商誉占净资产比(%)": ["商誉占净资产比", "商誉/净资产", "商誉净资产比"],
        }
        for column_name, keywords in metric_map.items():
            if signal[column_name] is None:
                signal[column_name] = extract_metric_from_frame(frame, keywords)

    return signal


def enrich_fundamental_indicators(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    cache = load_cached_fundamental_lookup()

    for column_name, default_value in build_empty_fundamental_signal().items():
        if column_name not in enriched.columns:
            enriched[column_name] = default_value

    codes = enriched["代码"].astype(str).tolist()

    def merge_cached_signals(current_df: pd.DataFrame) -> pd.DataFrame:
        if not cache:
            return current_df
        fundamental_df = pd.DataFrame(
            [{"代码": code, **cache.get(code, build_empty_fundamental_signal())} for code in codes]
        )
        merged = current_df.merge(
            fundamental_df,
            on="代码",
            how="left",
            suffixes=("", "_fund"),
        )
        for column_name in build_empty_fundamental_signal():
            fallback_column = f"{column_name}_fund"
            if fallback_column in merged.columns:
                merged[column_name] = merged[column_name].fillna(merged[fallback_column])
                merged = merged.drop(columns=[fallback_column])
        return merged

    enriched = merge_cached_signals(enriched)

    probe_ok = False
    try:
        probe_df = ak.stock_financial_analysis_indicator_em(symbol="000001")
        probe_ok = isinstance(probe_df, pd.DataFrame) and not probe_df.empty
    except Exception:
        probe_ok = False

    if not probe_ok:
        return enriched

    missing_codes = []
    for code in codes:
        cached_signal = cache.get(code)
        if cached_signal is None:
            missing_codes.append(code)
            continue
        if all(value is None for value in cached_signal.values()):
            missing_codes.append(code)

    for code in missing_codes:
        cache[code] = fetch_fundamental_signal_for_code(code)
        time.sleep(0.02)
    if missing_codes:
        save_cached_fundamental_lookup(cache)
        enriched = merge_cached_signals(enriched)
    return enriched


def fetch_bulk_industry_lookup() -> dict[str, str]:
    lookup: dict[str, str] = load_cached_industry_lookup()

    try:
        sz_df = ak.stock_info_sz_name_code(symbol="A股列表")
        for _, row in sz_df.iterrows():
            code = normalize_stock_code(pd.Series([row["A股代码"]])).iloc[0]
            lookup[code] = str(row.get("所属行业", ""))
    except Exception:
        pass

    try:
        bj_df = ak.stock_info_bj_name_code()
        for _, row in bj_df.iterrows():
            code = normalize_stock_code(pd.Series([row["证券代码"]])).iloc[0]
            lookup[code] = str(row.get("所属行业", ""))
    except Exception:
        pass

    save_cached_industry_lookup(lookup)
    return lookup


def fill_missing_industry_lookup(codes: list[str], lookup: dict[str, str]) -> dict[str, str]:
    missing_codes = [code for code in codes if not lookup.get(code)]
    for code in missing_codes:
        try:
            info_df = ak.stock_individual_info_em(symbol=code)
        except Exception:
            continue
        if info_df.empty:
            continue
        info_map = dict(zip(info_df["item"], info_df["value"]))
        lookup[code] = str(info_map.get("行业", ""))
        time.sleep(0.01)

    save_cached_industry_lookup(lookup)
    return lookup


def select_codes_for_industry_lookup(
    df: pd.DataFrame, category: str, industry_keyword: str
) -> list[str]:
    if category == "银行":
        mask = df["名称"].astype(str).str.contains("银行", na=False)
        return df.loc[mask, "代码"].astype(str).tolist()

    if category == "地产":
        name_mask = df["名称"].astype(str).str.contains(
            "地产|置业|城建|新城|金地|保利|万科|招商蛇口",
            regex=True,
            na=False,
        )
        return df.loc[name_mask, "代码"].astype(str).tolist()

    if industry_keyword:
        name_mask = df["名称"].astype(str).str.contains(industry_keyword, na=False)
        return df.loc[name_mask, "代码"].astype(str).tolist()

    return []


def infer_state_owned_label(row: pd.Series) -> str:
    text = " ".join(
        [
            str(row.get("实际控制人名称", "")),
            str(row.get("直接控制人名称", "")),
            str(row.get("控制类型", "")),
            str(row.get("公司名称", "")),
            str(row.get("经营范围", "")),
        ]
    )
    keywords = [
        "国有",
        "国务院国资委",
        "国资委",
        "人民政府",
        "财政局",
        "财政厅",
        "国有资产监督管理",
        "中央汇金",
        "国资",
        "省人民政府",
        "市人民政府",
        "自治区人民政府",
        "国务院",
    ]
    return "央国企" if any(keyword in text for keyword in keywords) else ""


def enrich_market_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    industry_lookup = fetch_bulk_industry_lookup()
    enriched["行业"] = enriched["代码"].map(industry_lookup).fillna("")

    control_df = fetch_control_info()
    if not control_df.empty:
        control_subset = control_df[
            ["证券代码", "实际控制人名称", "直接控制人名称", "控制类型"]
        ].rename(columns={"证券代码": "代码"})
        enriched = enriched.merge(control_subset, on="代码", how="left")
    else:
        enriched["实际控制人名称"] = ""
        enriched["直接控制人名称"] = ""
        enriched["控制类型"] = ""

    enriched["央国企标签"] = enriched.apply(infer_state_owned_label, axis=1)
    return enriched


def ensure_industry_for_filter(
    df: pd.DataFrame, category: str, industry_keyword: str
) -> pd.DataFrame:
    enriched = df.copy()
    lookup = load_cached_industry_lookup()
    target_codes = select_codes_for_industry_lookup(
        df=enriched,
        category=category,
        industry_keyword=industry_keyword,
    )
    missing_codes = [code for code in target_codes if not lookup.get(code)]
    if missing_codes:
        lookup = fill_missing_industry_lookup(missing_codes, lookup)
    enriched["行业"] = enriched["代码"].map(lookup).fillna(enriched["行业"])
    return enriched


def apply_category_filters(df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
    filtered = apply_st_name_filter(df, exclude_st=config.exclude_st)

    if "行业" not in filtered.columns:
        filtered["行业"] = ""
    if "央国企标签" not in filtered.columns:
        filtered["央国企标签"] = ""

    if config.industry_keyword:
        filtered = filtered[
            filtered["行业"].astype(str).str.contains(config.industry_keyword, na=False)
            | filtered["名称"].astype(str).str.contains(config.industry_keyword, na=False)
        ]

    if config.category == "银行":
        filtered = filtered[
            filtered["行业"].astype(str).str.contains("银行", na=False)
            | filtered["名称"].astype(str).str.contains("银行", na=False)
        ]
    elif config.category == "地产":
        filtered = filtered[
            filtered["行业"].astype(str).str.contains("房地产|地产", regex=True, na=False)
            | filtered["名称"].astype(str).str.contains(
                "地产|置业|城建|新城|金地|保利|万科|招商蛇口",
                regex=True,
                na=False,
            )
        ]
    elif config.category == "央国企":
        filtered = filtered[filtered["央国企标签"] == "央国企"]

    return filtered


def apply_indicator_filters(df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
    filtered = df.copy()

    if config.max_risk_score is not None:
        risk_series = pd.to_numeric(filtered["风险分数"], errors="coerce")
        filtered = filtered[risk_series <= config.max_risk_score]

    if config.min_rsi6 is not None:
        rsi_series = pd.to_numeric(filtered["RSI6"], errors="coerce")
        filtered = filtered[rsi_series >= config.min_rsi6]

    if config.max_rsi6 is not None:
        rsi_series = pd.to_numeric(filtered["RSI6"], errors="coerce")
        filtered = filtered[rsi_series <= config.max_rsi6]

    if config.require_above_ma20:
        close_series = pd.to_numeric(filtered["最新价"], errors="coerce")
        ma20_series = pd.to_numeric(filtered["MA20"], errors="coerce")
        filtered = filtered[close_series > ma20_series]

    if config.require_bullish_alignment:
        filtered = filtered[filtered["均线多头排列"] == "是"]

    if config.max_distance_to_52w_high is not None:
        high_distance_series = pd.to_numeric(filtered["距52周新高(%)"], errors="coerce")
        filtered = filtered[high_distance_series >= -config.max_distance_to_52w_high]

    if config.max_drawdown_60 is not None:
        drawdown_series = pd.to_numeric(filtered["60日最大回撤(%)"], errors="coerce")
        filtered = filtered[drawdown_series >= -config.max_drawdown_60]

    if config.require_relative_strength_positive:
        rs_series = pd.to_numeric(filtered["相对沪深300强度20日"], errors="coerce")
        filtered = filtered[rs_series > 0]

    if config.min_turnover_rate is not None:
        turnover_series = pd.to_numeric(filtered["换手率"], errors="coerce")
        filtered = filtered[turnover_series >= config.min_turnover_rate]

    if config.min_amount_yi is not None:
        amount_series = pd.to_numeric(filtered["成交额"], errors="coerce")
        filtered = filtered[amount_series >= config.min_amount_yi * 1e4]

    if config.exclude_problem_stocks:
        filtered = filtered[filtered["问题股标签"].fillna("").astype(str) == ""]

    if config.watch_only and config.low_absorption_only:
        filtered = filtered[
            filtered["买入建议状态"].isin(["进入低吸区", "可关注"])
        ]
    elif config.watch_only:
        filtered = filtered[filtered["买入建议状态"] == "可关注"]
    elif config.low_absorption_only:
        filtered = filtered[filtered["买入建议状态"] == "进入低吸区"]

    filtered = apply_strategy_preset(filtered, config)

    return filtered


def apply_kdj_filter(df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
    filtered = df.copy()
    if not config.require_kdj_gold_cross:
        return filtered

    if "KDJ金叉" not in filtered.columns:
        filtered["KDJ金叉"] = "否"
    return filtered[filtered["KDJ金叉"] == "是"].copy()


def risk_percentile(series: pd.Series, higher_is_risk: bool = True) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")
    if numeric_series.notna().sum() == 0:
        return pd.Series(0.5, index=series.index, dtype="float64")
    ranked = numeric_series.rank(
        pct=True,
        ascending=higher_is_risk,
        method="average",
    )
    return ranked.fillna(0.5)


def build_risk_level(score: float) -> str:
    if score >= 80:
        return "极高"
    if score >= 60:
        return "高"
    if score >= 30:
        return "中"
    return "低"


def enrich_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    st_flag = (
        enriched["名称"].astype(str).str.contains("ST", na=False).astype("float64")
    )
    amplitude_pct = risk_percentile(enriched["振幅"], higher_is_risk=True)
    turnover_pct = risk_percentile(enriched["换手率"], higher_is_risk=True)
    swing_pct = risk_percentile(
        enriched["60日涨跌幅"].abs(),
        higher_is_risk=True,
    )
    size_pct = risk_percentile(enriched["总市值"], higher_is_risk=False)

    pe_series = pd.to_numeric(enriched["市盈率-动态"], errors="coerce")
    pb_series = pd.to_numeric(enriched["市净率"], errors="coerce")
    pe_pct = risk_percentile(pe_series.where(pe_series > 0), higher_is_risk=True)
    pb_pct = risk_percentile(pb_series.where(pb_series > 0), higher_is_risk=True)
    valuation_pct = pd.concat([pe_pct, pb_pct], axis=1).max(axis=1).fillna(0.5)
    valuation_pct = valuation_pct.where(pe_series > 0, 1.0)

    risk_score = (
        amplitude_pct * 25
        + swing_pct * 20
        + turnover_pct * 20
        + size_pct * 15
        + valuation_pct * 10
        + st_flag * 10
    ).round(2)
    enriched["风险分数"] = risk_score
    enriched["风险等级"] = risk_score.apply(build_risk_level)
    return enriched


def clamp_score(series: pd.Series, low: float, high: float, inverse: bool = False) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")
    if high == low:
        score = pd.Series(50.0, index=series.index, dtype="float64")
    else:
        score = ((numeric_series - low) / (high - low) * 100).clip(0, 100)
    if inverse:
        score = 100 - score
    return score.fillna(50.0)


def build_position_advice(risk_score: float | None, total_score: float | None) -> str:
    if risk_score is None or pd.isna(risk_score):
        return "观察"
    if risk_score >= 75:
        return "回避"
    if total_score is None or pd.isna(total_score):
        return "观察"
    if risk_score <= 30 and total_score >= 75:
        return "中仓"
    if risk_score <= 50 and total_score >= 60:
        return "轻仓"
    if risk_score <= 70 and total_score >= 45:
        return "轻仓试探"
    return "观察"


def build_problem_stock_tags(row: pd.Series) -> str:
    tags: list[str] = []
    name_text = str(row.get("名称", "") or "")
    risk_level = str(row.get("风险等级", "") or "")
    debt_ratio = normalize_metric_number(row.get("资产负债率(%)"))
    gross_margin = normalize_metric_number(row.get("毛利率(%)"))
    profit_growth = normalize_metric_number(row.get("净利润增长率(%)"))
    revenue_growth = normalize_metric_number(row.get("营收增长率(%)"))
    goodwill_ratio = normalize_metric_number(row.get("商誉占净资产比(%)"))

    if "ST" in name_text.upper():
        tags.append("ST")
    if any(keyword in name_text for keyword in ["退", "退市", "整理"]):
        tags.append("退市风险")
    if debt_ratio is not None and debt_ratio >= 75:
        tags.append("高负债")
    if gross_margin is not None and gross_margin <= 10:
        tags.append("低毛利")
    if goodwill_ratio is not None and goodwill_ratio >= 30:
        tags.append("商誉高")
    if (
        profit_growth is not None
        and revenue_growth is not None
        and profit_growth < 0
        and revenue_growth < 0
    ):
        tags.append("连续亏损风险")
    elif profit_growth is not None and profit_growth < -20:
        tags.append("连续亏损风险")
    if risk_level == "极高":
        tags.append("高风险")
    return "、".join(tags)


def enrich_advanced_factors(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if "行业" not in enriched.columns:
        enriched["行业"] = ""
    industry_group = enriched["行业"].fillna("").replace("", "未分类")

    latest_price = pd.to_numeric(enriched.get("最新价"), errors="coerce")
    ma20 = pd.to_numeric(enriched.get("MA20"), errors="coerce")
    rsi6 = pd.to_numeric(enriched.get("RSI6"), errors="coerce")
    rsi12 = pd.to_numeric(enriched.get("RSI12"), errors="coerce")
    pe_series = pd.to_numeric(enriched.get("市盈率-动态"), errors="coerce").where(
        lambda series: series > 0
    )
    pb_series = pd.to_numeric(enriched.get("市净率"), errors="coerce").where(
        lambda series: series > 0
    )
    macd_series = pd.to_numeric(enriched.get("MACD"), errors="coerce")
    dif_series = pd.to_numeric(enriched.get("DIF"), errors="coerce")
    dea_series = pd.to_numeric(enriched.get("DEA"), errors="coerce")
    volume_series = pd.to_numeric(enriched.get("成交量"), errors="coerce")
    volume_ma5 = pd.to_numeric(enriched.get("成交量MA5"), errors="coerce")
    volume_ma10 = pd.to_numeric(enriched.get("成交量MA10"), errors="coerce")
    daily_change = pd.to_numeric(enriched.get("涨跌幅"), errors="coerce")
    drawdown_60 = pd.to_numeric(enriched.get("60日最大回撤(%)"), errors="coerce")
    distance_low_52w = pd.to_numeric(enriched.get("距52周新低(%)"), errors="coerce")
    risk_score = pd.to_numeric(enriched.get("风险分数"), errors="coerce")
    dividend_yield = pd.to_numeric(enriched.get("股息率(%)"), errors="coerce")
    roe_series = pd.to_numeric(enriched.get("ROE(%)"), errors="coerce")
    revenue_growth = pd.to_numeric(enriched.get("营收增长率(%)"), errors="coerce")
    profit_growth = pd.to_numeric(enriched.get("净利润增长率(%)"), errors="coerce")
    debt_ratio = pd.to_numeric(enriched.get("资产负债率(%)"), errors="coerce")
    gross_margin = pd.to_numeric(enriched.get("毛利率(%)"), errors="coerce")
    net_margin = pd.to_numeric(enriched.get("净利率(%)"), errors="coerce")
    goodwill_ratio = pd.to_numeric(enriched.get("商誉占净资产比(%)"), errors="coerce")
    rs20 = pd.to_numeric(enriched.get("相对沪深300强度20日"), errors="coerce")
    rs60 = pd.to_numeric(enriched.get("相对沪深300强度60日"), errors="coerce")
    ret20 = pd.to_numeric(enriched.get("20日涨跌幅"), errors="coerce")
    vol20 = pd.to_numeric(enriched.get("20日波动率(%)"), errors="coerce")
    vol60 = pd.to_numeric(enriched.get("60日波动率(%)"), errors="coerce")

    pe_pct = (pe_series.groupby(industry_group).rank(pct=True, ascending=True, method="average") * 100)
    pb_pct = (pb_series.groupby(industry_group).rank(pct=True, ascending=True, method="average") * 100)
    valuation_pct = pd.concat([pe_pct, pb_pct], axis=1).mean(axis=1, skipna=True).fillna(50.0)
    enriched["行业PE分位(%)"] = pe_pct.round(2)
    enriched["行业PB分位(%)"] = pb_pct.round(2)
    enriched["行业估值分位(%)"] = valuation_pct.round(2)

    valuation_score = pd.concat(
        [
            (100 - pe_pct).fillna(50.0),
            (100 - pb_pct).fillna(50.0),
            clamp_score(dividend_yield, 0, 6),
        ],
        axis=1,
    ).mean(axis=1, skipna=True).round(2)

    rsi_rebound_flag = (rsi6 >= 25) & (rsi6 <= 55) & (rsi6 >= rsi12)
    macd_turn_positive_flag = (macd_series > 0) & (dif_series >= dea_series)
    volume_breakout_flag = (
        (volume_series > volume_ma5)
        & (volume_ma5 >= volume_ma10.fillna(volume_ma5))
        & (latest_price > ma20)
        & (daily_change > 0)
    )

    enriched["RSI低位回升"] = rsi_rebound_flag.map({True: "是", False: "否"})
    enriched["MACD翻红"] = macd_turn_positive_flag.map({True: "是", False: "否"})
    enriched["放量突破"] = volume_breakout_flag.map({True: "是", False: "否"})

    trend_score = pd.concat(
        [
            clamp_score(rs20, -10, 20),
            clamp_score(rs60, -15, 30),
            clamp_score(ret20, -15, 25),
            macd_turn_positive_flag.astype("float64") * 100,
            (enriched["均线多头排列"] == "是").astype("float64") * 100,
            (latest_price > ma20).astype("float64") * 100,
            (enriched["KDJ金叉"] == "是").astype("float64") * 100,
            volume_breakout_flag.astype("float64") * 100,
        ],
        axis=1,
    ).mean(axis=1, skipna=True).round(2)

    quality_score = pd.concat(
        [
            clamp_score(roe_series, 0, 20),
            clamp_score(revenue_growth, -10, 30),
            clamp_score(profit_growth, -10, 30),
            clamp_score(gross_margin, 10, 50),
            clamp_score(net_margin, 0, 25),
            clamp_score(debt_ratio, 20, 80, inverse=True),
            clamp_score(dividend_yield, 0, 6),
        ],
        axis=1,
    ).mean(axis=1, skipna=True).round(2)

    risk_adjusted_score = (100 - risk_score).clip(0, 100).fillna(50.0).round(2)
    total_score = (
        valuation_score * 0.30
        + trend_score * 0.25
        + quality_score * 0.25
        + risk_adjusted_score * 0.20
    ).round(2)

    enriched["估值因子分"] = valuation_score
    enriched["趋势因子分"] = trend_score
    enriched["质量因子分"] = quality_score
    enriched["风险调整分"] = risk_adjusted_score
    enriched["组合总分"] = total_score
    enriched["建议仓位"] = [
        build_position_advice(risk, total)
        for risk, total in zip(risk_score.tolist(), total_score.tolist())
    ]

    cheap_rank = valuation_pct.groupby(industry_group).rank(method="dense", ascending=True)
    strength_rank = trend_score.groupby(industry_group).rank(method="dense", ascending=False)
    stability_rank = risk_score.groupby(industry_group).rank(method="dense", ascending=True)
    enriched["行业便宜度排名"] = cheap_rank.round(0).astype("Int64")
    enriched["行业强度排名"] = strength_rank.round(0).astype("Int64")
    enriched["行业稳健度排名"] = stability_rank.round(0).astype("Int64")

    vol20_warn_threshold = vol20.quantile(0.9) if vol20.notna().any() else None
    vol60_warn_threshold = vol60.quantile(0.9) if vol60.notna().any() else None
    vol20_warning = (
        (vol20_warn_threshold is not None)
        & vol20.notna()
        & (vol20 >= float(vol20_warn_threshold))
    )
    vol60_warning = (
        (vol60_warn_threshold is not None)
        & vol60.notna()
        & (vol60 >= float(vol60_warn_threshold))
    )
    enriched["20日波动预警"] = vol20_warning.map({True: "是", False: "否"})
    enriched["60日波动预警"] = vol60_warning.map({True: "是", False: "否"})
    enriched["波动预警"] = [
        "20日+60日异常"
        if warn20 and warn60
        else "20日异常"
        if warn20
        else "60日异常"
        if warn60
        else ""
        for warn20, warn60 in zip(vol20_warning.tolist(), vol60_warning.tolist())
    ]
    enriched["问题股标签"] = enriched.apply(build_problem_stock_tags, axis=1)

    strategy_tags: list[str] = []
    for idx, row in enriched.iterrows():
        tags: list[str] = []
        if (
            normalize_metric_number(row.get("股息率(%)")) is not None
            and normalize_metric_number(row.get("股息率(%)")) >= 2.5
            and normalize_metric_number(row.get("20日波动率(%)")) is not None
            and normalize_metric_number(row.get("20日波动率(%)")) <= 35
            and normalize_metric_number(row.get("60日最大回撤(%)")) is not None
            and normalize_metric_number(row.get("60日最大回撤(%)")) >= -20
        ):
            tags.append("红利")
        if (
            normalize_metric_number(row.get("营收增长率(%)")) is not None
            and normalize_metric_number(row.get("营收增长率(%)")) >= 10
            and normalize_metric_number(row.get("净利润增长率(%)")) is not None
            and normalize_metric_number(row.get("净利润增长率(%)")) >= 10
            and normalize_metric_number(row.get("相对沪深300强度20日")) is not None
            and normalize_metric_number(row.get("相对沪深300强度20日")) > 0
            and normalize_metric_number(row.get("最新价")) is not None
            and normalize_metric_number(row.get("MA20")) is not None
            and normalize_metric_number(row.get("最新价")) > normalize_metric_number(row.get("MA20"))
        ):
            tags.append("成长")
        if row.get("KDJ金叉") == "是" and row.get("RSI低位回升") == "是":
            low_distance = normalize_metric_number(row.get("距52周新低(%)"))
            if low_distance is not None and low_distance <= 35:
                tags.append("反转")
        if row.get("MACD翻红") == "是" and row.get("均线多头排列") == "是" and row.get("放量突破") == "是":
            tags.append("趋势")
        strategy_tags.append("、".join(tags))
    enriched["策略标签"] = strategy_tags
    return enriched


def apply_strategy_preset(df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
    preset = config.strategy_preset
    if preset == "全部":
        return df

    filtered = df.copy()
    dividend_yield = pd.to_numeric(filtered["股息率(%)"], errors="coerce")
    volatility_20 = pd.to_numeric(filtered["20日波动率(%)"], errors="coerce")
    drawdown_60 = pd.to_numeric(filtered["60日最大回撤(%)"], errors="coerce")
    revenue_growth = pd.to_numeric(filtered["营收增长率(%)"], errors="coerce")
    profit_growth = pd.to_numeric(filtered["净利润增长率(%)"], errors="coerce")
    rs20 = pd.to_numeric(filtered["相对沪深300强度20日"], errors="coerce")
    latest_price = pd.to_numeric(filtered["最新价"], errors="coerce")
    ma20 = pd.to_numeric(filtered["MA20"], errors="coerce")
    low_distance = pd.to_numeric(filtered["距52周新低(%)"], errors="coerce")
    pb_series = pd.to_numeric(filtered["市净率"], errors="coerce")
    risk_score = pd.to_numeric(filtered["风险分数"], errors="coerce")
    trend_score = pd.to_numeric(filtered["趋势因子分"], errors="coerce")
    ret20 = pd.to_numeric(filtered["20日涨跌幅"], errors="coerce")
    state_owned = filtered["央国企标签"] == "央国企"

    if preset == "红利策略":
        has_dividend_data = dividend_yield.notna().sum() > 0
        if has_dividend_data:
            mask = (
                ((dividend_yield >= 2.5) | (state_owned & (dividend_yield >= 2.0)))
                & (volatility_20 <= 35)
                & (drawdown_60 >= -20)
            )
        else:
            mask = (
                (volatility_20 <= 30)
                & (drawdown_60 >= -18)
                & ((state_owned) | (pb_series <= 1.2) | (risk_score <= 35))
            )
        filtered = filtered[mask].copy()
        if filtered.empty:
            return filtered
        filtered["_strategy_priority"] = state_owned.astype("int64")
        filtered = filtered.sort_values(
            by=["_strategy_priority", "股息率(%)", "风险分数", "20日波动率(%)"],
            ascending=[False, False, True, True],
        ).drop(columns=["_strategy_priority"])
        return filtered

    if preset == "成长策略":
        has_growth_data = revenue_growth.notna().sum() > 0 and profit_growth.notna().sum() > 0
        if has_growth_data:
            mask = (
                (revenue_growth >= 10)
                & (profit_growth >= 10)
                & (rs20 > 0)
                & (latest_price > ma20)
            )
        else:
            mask = (
                (rs20 > 0)
                & (latest_price > ma20)
                & ((filtered["均线多头排列"] == "是") | (trend_score >= 60))
                & (ret20 > 0)
            )
        return filtered[mask].copy()

    if preset == "反转策略":
        mask = (
            (filtered["KDJ金叉"] == "是")
            & (filtered["RSI低位回升"] == "是")
            & (low_distance <= 35)
        )
        return filtered[mask].copy()

    if preset == "趋势策略":
        mask = (
            (filtered["MACD翻红"] == "是")
            & (filtered["均线多头排列"] == "是")
            & (filtered["放量突破"] == "是")
        )
        return filtered[mask].copy()

    return filtered


def risk_discount_factor(risk_score: float | None) -> float:
    if risk_score is None or pd.isna(risk_score):
        return 0.95
    if risk_score < 30:
        return 1.0
    if risk_score < 60:
        return 0.95
    if risk_score < 80:
        return 0.90
    return 0.85


def get_buy_style_profile(style: str) -> dict[str, float]:
    style_profiles = {
        "激进": {
            "upper_multiplier": 1.03,
            "buy_band_ratio": 0.08,
            "stop_band_ratio": 0.04,
            "target_reward_ratio": 2.5,
            "min_lower_ratio": 0.82,
            "min_stop_ratio": 0.76,
        },
        "保守": {
            "upper_multiplier": 0.95,
            "buy_band_ratio": 0.15,
            "stop_band_ratio": 0.08,
            "target_reward_ratio": 1.8,
            "min_lower_ratio": 0.70,
            "min_stop_ratio": 0.65,
        },
        "标准": {
            "upper_multiplier": 1.0,
            "buy_band_ratio": 0.12,
            "stop_band_ratio": 0.06,
            "target_reward_ratio": 2.0,
            "min_lower_ratio": 0.76,
            "min_stop_ratio": 0.70,
        },
    }
    return style_profiles.get(style, style_profiles["标准"])


def build_buy_signal_for_row(
    row: pd.Series,
    style: str,
) -> dict[str, float | str | None]:
    current_price = normalize_metric_number(row.get("最新价"))
    pe = normalize_metric_number(row.get("市盈率-动态"))
    pb = normalize_metric_number(row.get("市净率"))
    ma20 = normalize_metric_number(row.get("MA20"))
    boll_mid = normalize_metric_number(row.get("BOLL中轨"))
    atr14 = normalize_metric_number(row.get("ATR14"))
    risk_score = normalize_metric_number(row.get("风险分数"))

    result = {
        "建议买入下限": None,
        "建议买入上限": None,
        "建议买入价距离(%)": None,
        "追高提示": "",
        "止损参考价": None,
        "止损比例(%)": None,
        "目标参考价": None,
        "盈亏比": None,
        "买入建议状态": "",
        "买入风格": style,
    }
    if current_price is None or current_price <= 0:
        return result

    valuation_candidates: list[float] = []
    if pe is not None and pe > 0:
        valuation_candidates.append(current_price * min(15.0, pe) / pe)
    if pb is not None and pb > 0:
        valuation_candidates.append(current_price * min(1.5, pb) / pb)

    technical_candidates = [value for value in [ma20, boll_mid] if value is not None and value > 0]

    base_upper = None
    if valuation_candidates and technical_candidates:
        base_upper = min(min(valuation_candidates), min(technical_candidates))
    elif valuation_candidates:
        base_upper = min(valuation_candidates)
    elif technical_candidates:
        base_upper = min(technical_candidates)

    if base_upper is None or base_upper <= 0:
        return result

    profile = get_buy_style_profile(style)
    adjusted_upper = round(
        base_upper * risk_discount_factor(risk_score) * profile["upper_multiplier"],
        2,
    )
    if adjusted_upper <= 0:
        return result
    atr_value = atr14 if atr14 is not None and atr14 > 0 else max(current_price * 0.03, 0.1)
    max_buy_band = max(
        adjusted_upper * profile["buy_band_ratio"],
        current_price * (profile["buy_band_ratio"] * 0.6),
        0.2,
    )
    effective_band = min(atr_value, max_buy_band)
    buy_lower = round(
        max(adjusted_upper - effective_band, adjusted_upper * profile["min_lower_ratio"], 0.01),
        2,
    )
    stop_band = min(
        0.5 * atr_value,
        adjusted_upper * profile["stop_band_ratio"],
        current_price * (profile["stop_band_ratio"] * 0.8),
    )
    stop_loss = round(
        max(buy_lower - stop_band, adjusted_upper * profile["min_stop_ratio"], 0.01),
        2,
    )
    target_price = round(
        adjusted_upper + profile["target_reward_ratio"] * (adjusted_upper - stop_loss),
        2,
    )

    if current_price <= buy_lower:
        status = "进入低吸区"
    elif current_price <= adjusted_upper:
        status = "可关注"
    else:
        status = "等待回落"

    distance_to_buy_upper = round((current_price / adjusted_upper - 1) * 100, 2)
    if distance_to_buy_upper >= 15:
        chase_hint = "严重追高"
    elif distance_to_buy_upper >= 8:
        chase_hint = "不宜追高"
    elif distance_to_buy_upper >= 3:
        chase_hint = "偏离买点"
    else:
        chase_hint = ""

    stop_ratio = round((1 - stop_loss / current_price) * 100, 2)
    profit_ratio = (target_price - current_price) / current_price if current_price > 0 else None
    loss_ratio = (current_price - stop_loss) / current_price if current_price > 0 else None
    if profit_ratio is not None and loss_ratio not in (None, 0):
        reward_risk_ratio = round(profit_ratio / loss_ratio, 2)
    else:
        reward_risk_ratio = None

    result["建议买入下限"] = buy_lower
    result["建议买入上限"] = adjusted_upper
    result["建议买入价距离(%)"] = distance_to_buy_upper
    result["追高提示"] = chase_hint
    result["止损参考价"] = stop_loss
    result["止损比例(%)"] = stop_ratio
    result["目标参考价"] = target_price
    result["盈亏比"] = reward_risk_ratio
    result["买入建议状态"] = status
    return result


def enrich_buy_price_signals(df: pd.DataFrame, style: str) -> pd.DataFrame:
    enriched = df.copy()
    buy_signal_df = enriched.apply(
        build_buy_signal_for_row,
        axis=1,
        result_type="expand",
        style=style,
    )
    for column in buy_signal_df.columns:
        enriched[column] = buy_signal_df[column]
    return enriched


def build_low_valuation_candidates(
    df: pd.DataFrame, config: FilterConfig
) -> pd.DataFrame:
    filtered = df.copy()
    filtered = filtered.dropna(subset=["市盈率-动态", "市净率", "总市值"])
    filtered = filtered[filtered["市盈率-动态"] > 0]
    filtered = filtered[filtered["市净率"] > 0]
    filtered = filtered[filtered["市盈率-动态"] <= config.max_pe]
    filtered = filtered[filtered["市净率"] <= config.max_pb]
    filtered = filtered[filtered["总市值"] >= config.min_market_cap_yi * 1e8]

    if filtered.empty:
        filtered = filtered.copy()
        filtered["估值分数"] = pd.Series(dtype="float64")
        return filtered

    pe_rank = filtered["市盈率-动态"].rank(pct=True, ascending=True)
    pb_rank = filtered["市净率"].rank(pct=True, ascending=True)
    filtered["估值分数"] = ((pe_rank + pb_rank) / 2 * 100).round(2)

    filtered = filtered.sort_values(
        by=["估值分数", "市净率", "市盈率-动态", "总市值"],
        ascending=[True, True, True, False],
    )
    return filtered.head(config.top_n).copy()


def format_output(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    if "估值分数" not in output.columns:
        output["估值分数"] = pd.NA

    missing_cols = [column for column in OUTPUT_COLUMNS if column not in output.columns]
    for column in missing_cols:
        output[column] = pd.NA

    output = output[OUTPUT_COLUMNS]
    output["代码"] = normalize_stock_code(output["代码"])
    output["行业"] = output["行业"].fillna("")
    output["央国企标签"] = output["央国企标签"].fillna("")
    output["KDJ金叉"] = output["KDJ金叉"].fillna("")
    output["总市值"] = (output["总市值"] / 1e8).round(2)
    output["流通市值"] = (output["流通市值"] / 1e8).round(2)
    return output.rename(columns={"总市值": "总市值(亿元)", "流通市值": "流通市值(亿元)"})


def update_history_file(all_df: pd.DataFrame, output_dir: Path) -> Path:
    history_path = output_dir / "a_share_daily_metrics_history.csv"
    history_frame = format_output(all_df)

    if history_path.exists():
        old_history = pd.read_csv(history_path, dtype={"代码": str})
        history_frame = pd.concat([old_history, history_frame], ignore_index=True)
        history_frame = history_frame.drop_duplicates(
            subset=["日期", "代码"], keep="last"
        )

    history_frame = history_frame.sort_values(by=["日期", "代码"], ascending=[True, True])
    history_frame.to_csv(history_path, index=False, encoding="utf-8-sig")
    return history_path


def save_reports(
    market_df: pd.DataFrame,
    filtered_market_df: pd.DataFrame,
    low_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%Y%m%d")

    market_path = output_dir / f"a_share_daily_metrics_{date_tag}.csv"
    filtered_market_path = output_dir / f"filtered_market_{date_tag}.csv"
    value_path = output_dir / f"low_valuation_stocks_{date_tag}.csv"

    format_output(market_df).to_csv(market_path, index=False, encoding="utf-8-sig")
    format_output(filtered_market_df).to_csv(
        filtered_market_path, index=False, encoding="utf-8-sig"
    )
    format_output(low_df).to_csv(value_path, index=False, encoding="utf-8-sig")
    history_path = update_history_file(all_df=market_df, output_dir=output_dir)
    return market_path, filtered_market_path, value_path, history_path


def print_summary(
    market_df: pd.DataFrame,
    filtered_market_df: pd.DataFrame,
    low_df: pd.DataFrame,
    market_path: Path,
    filtered_market_path: Path,
    value_path: Path,
    history_path: Path,
    source_name: str,
) -> None:
    print(f"数据源: {source_name}")
    print(f"抓取完成，共 {len(market_df)} 只 A 股。")
    print(f"筛选后保留: {len(filtered_market_df)} 只。")
    print(f"全市场指标已导出到: {market_path}")
    print(f"筛选后市场结果已导出到: {filtered_market_path}")
    print(f"低估值结果已导出到: {value_path}")
    print(f"历史总表已更新到: {history_path}")

    if low_df.empty:
        print("当前筛选条件下没有找到低估值股票，请放宽阈值后重试。")
        return

    preview_columns = [
        "代码",
        "名称",
        "最新价",
        "市盈率-动态",
        "市净率",
        "总市值",
        "估值分数",
    ]
    preview = low_df[preview_columns].copy()
    preview["总市值"] = (preview["总市值"] / 1e8).round(2)
    preview = preview.rename(columns={"总市值": "总市值(亿元)"})
    print("")
    print("低估值候选股预览:")
    print(preview.head(10).to_string(index=False))


def run_analysis(config: FilterConfig, output_dir: Path, use_env_proxy: bool) -> dict:
    market_df, source_name = fetch_market_snapshot(use_env_proxy=use_env_proxy)
    market_df = enrich_market_dataframe(market_df)
    market_df = ensure_industry_for_filter(
        df=market_df,
        category=config.category,
        industry_keyword=config.industry_keyword,
    )
    market_df = enrich_kdj_signals(
        df=market_df,
        use_env_proxy=use_env_proxy,
        window=config.kdj_window,
    )
    market_df = enrich_fundamental_indicators(market_df)
    market_df = enrich_risk_score(market_df)
    market_df = enrich_advanced_factors(market_df)
    market_df = enrich_buy_price_signals(market_df, style=config.buy_style)
    filtered_market_df = apply_category_filters(market_df, config)
    filtered_market_df = apply_indicator_filters(filtered_market_df, config)
    if config.require_kdj_gold_cross:
        filtered_market_df = apply_kdj_filter(filtered_market_df, config)
    low_df = build_low_valuation_candidates(filtered_market_df, config)
    market_path, filtered_market_path, value_path, history_path = save_reports(
        market_df=market_df,
        filtered_market_df=filtered_market_df,
        low_df=low_df,
        output_dir=output_dir,
    )
    return {
        "market_df": market_df,
        "filtered_market_df": filtered_market_df,
        "low_df": low_df,
        "source_name": source_name,
        "market_path": market_path,
        "filtered_market_path": filtered_market_path,
        "value_path": value_path,
        "history_path": history_path,
    }


def main() -> int:
    args = parse_args()
    config = FilterConfig(
        max_pe=args.max_pe,
        max_pb=args.max_pb,
        min_market_cap_yi=args.min_market_cap_yi,
        exclude_st=not args.include_st,
        top_n=args.top_n,
        category=args.category,
        industry_keyword=args.industry_keyword,
        require_kdj_gold_cross=args.require_kdj_gold_cross,
        kdj_window=args.kdj_window,
        buy_style=args.buy_style,
        strategy_preset=args.strategy_preset,
        watch_only=args.watch_only,
        low_absorption_only=args.low_absorption_only,
        exclude_problem_stocks=args.exclude_problem_stocks,
        min_turnover_rate=args.min_turnover_rate,
        min_amount_yi=args.min_amount_yi,
    )

    try:
        result = run_analysis(
            config=config,
            output_dir=Path(args.output_dir),
            use_env_proxy=args.use_env_proxy,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print_summary(
        result["market_df"],
        result["filtered_market_df"],
        result["low_df"],
        result["market_path"],
        result["filtered_market_path"],
        result["value_path"],
        result["history_path"],
        result["source_name"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
