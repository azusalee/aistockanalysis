#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from collections import Counter
from datetime import datetime
import json

import pandas as pd

from flask import Flask, render_template, request

from stock_analysis import (
    FilterConfig,
    apply_category_filters,
    apply_indicator_filters,
    apply_kdj_filter,
    apply_st_name_filter,
    build_low_valuation_candidates,
    enrich_buy_price_signals,
    format_output,
    run_analysis,
    save_reports,
)


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
WEB_SNAPSHOT_PATH = CACHE_DIR / "web_market_snapshot.pkl"
WEB_META_PATH = CACHE_DIR / "web_market_snapshot_meta.json"
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

SORT_OPTIONS = {
    "默认": None,
    "PE": "市盈率-动态",
    "PB": "市净率",
    "股息率": "股息率(%)",
    "风险分数": "风险分数",
    "K值": "K值",
    "D值": "D值",
    "J值": "J值",
    "MACD": "MACD",
    "涨跌幅": "涨跌幅",
    "60日涨跌幅": "60日涨跌幅",
    "建议买入价距离": "建议买入价距离(%)",
}
SORT_COLUMN_TO_OPTION = {
    column: option for option, column in SORT_OPTIONS.items() if column is not None
}

DEFAULT_VISIBLE_COLUMNS = [
    "代码",
    "名称",
    "最新价",
    "行业",
    "风险分数",
    "风险等级",
    "买入建议状态",
    "建议买入下限",
    "建议买入上限",
    "目标参考价",
    "K值",
    "D值",
    "J值",
    "MACD",
    "涨跌幅",
    "市盈率-动态",
    "市净率",
    "总市值(亿元)",
]

PINNED_DISPLAY_COLUMNS = ["代码", "名称", "最新价"]


def parse_bool(value: str | None) -> bool:
    return value in {"1", "true", "on", "yes"}


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def parse_positive_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        parsed = int(text)
    except ValueError:
        return default
    return max(parsed, 1)


def parse_multi_values(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, list):
        items = values
    else:
        items = [values]
    return [str(item).strip() for item in items if str(item).strip()]


def build_config_from_form(form: dict) -> FilterConfig:
    return FilterConfig(
        max_pe=float(form["max_pe"]),
        max_pb=float(form["max_pb"]),
        min_market_cap_yi=float(form["min_market_cap_yi"]),
        exclude_st=not form["include_st"],
        top_n=int(form["top_n"]),
        category=form["category"],
        industry_keyword=form["industry_keyword"].strip(),
        require_kdj_gold_cross=form["require_kdj_gold_cross"],
        max_risk_score=parse_optional_float(form["max_risk_score"]),
        min_rsi6=parse_optional_float(form["min_rsi6"]),
        max_rsi6=parse_optional_float(form["max_rsi6"]),
        require_above_ma20=form["require_above_ma20"],
        require_bullish_alignment=form["require_bullish_alignment"],
        max_distance_to_52w_high=parse_optional_float(form["max_distance_to_52w_high"]),
        max_drawdown_60=parse_optional_float(form["max_drawdown_60"]),
        require_relative_strength_positive=form["require_relative_strength_positive"],
        buy_style=form["buy_style"],
        watch_only=form["watch_only"],
        low_absorption_only=form["low_absorption_only"],
    )


def save_web_snapshot(market_df: pd.DataFrame, source_name: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    market_df.to_pickle(WEB_SNAPSHOT_PATH)
    WEB_META_PATH.write_text(
        json.dumps(
            {
                "source_name": source_name,
                "refreshed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_web_snapshot() -> tuple[pd.DataFrame | None, dict]:
    if not WEB_SNAPSHOT_PATH.exists():
        return None, {}
    try:
        market_df = pd.read_pickle(WEB_SNAPSHOT_PATH)
    except Exception:
        return None, {}

    meta: dict = {}
    if WEB_META_PATH.exists():
        try:
            meta = json.loads(WEB_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return market_df, meta


def build_result_from_market_df(
    market_df: pd.DataFrame,
    config: FilterConfig,
    output_dir: Path,
    source_name: str,
    snapshot_meta: dict | None = None,
    persist_reports: bool = True,
) -> dict:
    working_df = enrich_buy_price_signals(market_df.copy(), style=config.buy_style)
    filtered_market_df = apply_category_filters(working_df, config)
    filtered_market_df = apply_indicator_filters(filtered_market_df, config)
    if config.require_kdj_gold_cross:
        filtered_market_df = apply_kdj_filter(filtered_market_df, config)
    low_df = build_low_valuation_candidates(filtered_market_df, config)
    if persist_reports:
        market_path, filtered_market_path, value_path, history_path = save_reports(
            market_df=working_df,
            filtered_market_df=filtered_market_df,
            low_df=low_df,
            output_dir=output_dir,
        )
    else:
        date_tag = datetime.now().strftime("%Y%m%d")
        market_path = output_dir / f"a_share_daily_metrics_{date_tag}.csv"
        filtered_market_path = output_dir / f"filtered_market_{date_tag}.csv"
        value_path = output_dir / f"low_valuation_stocks_{date_tag}.csv"
        history_path = output_dir / "a_share_daily_metrics_history.csv"
    return {
        "market_df": working_df,
        "filtered_market_df": filtered_market_df,
        "low_df": low_df,
        "source_name": source_name,
        "market_path": market_path,
        "filtered_market_path": filtered_market_path,
        "value_path": value_path,
        "history_path": history_path,
        "snapshot_meta": snapshot_meta or {},
    }


def build_summary(result: dict) -> dict:
    buy_status_counts = Counter(
        status
        for status in result["filtered_market_df"]["买入建议状态"].fillna("").astype(str)
        if status
    )
    return {
        "source_name": result["source_name"],
        "raw_market_count": len(result["market_df"]),
        "market_count": len(result["filtered_market_df"]),
        "value_count": len(result["low_df"]),
        "market_path": str(result["market_path"]),
        "filtered_market_path": str(result["filtered_market_path"]),
        "value_path": str(result["value_path"]),
        "market_filename": Path(result["market_path"]).name,
        "filtered_market_filename": Path(result["filtered_market_path"]).name,
        "value_filename": Path(result["value_path"]).name,
        "buy_status_counts": {
            "进入低吸区": buy_status_counts.get("进入低吸区", 0),
            "可关注": buy_status_counts.get("可关注", 0),
            "等待回落": buy_status_counts.get("等待回落", 0),
        },
        "refreshed_at": result.get("snapshot_meta", {}).get("refreshed_at", ""),
        "data_mode": result.get("snapshot_meta", {}).get("data_mode", ""),
    }


def enrich_display_metrics(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if "建议买入价距离(%)" not in enriched.columns:
        latest_price = pd.to_numeric(enriched.get("最新价"), errors="coerce")
        buy_upper = pd.to_numeric(enriched.get("建议买入上限"), errors="coerce")
        enriched["建议买入价距离(%)"] = ((latest_price / buy_upper) - 1) * 100
    return enriched


def sort_dataframe(df: pd.DataFrame, sort_key: str, sort_order: str) -> pd.DataFrame:
    target_column = SORT_OPTIONS.get(sort_key)
    if not target_column or target_column not in df.columns:
        return df

    sorted_df = df.copy()
    numeric_series = pd.to_numeric(sorted_df[target_column], errors="coerce")
    ascending = sort_order == "asc"
    sorted_df = sorted_df.assign(_sort_value=numeric_series)
    sorted_df = sorted_df.sort_values(
        by="_sort_value",
        ascending=ascending,
        na_position="last",
        kind="mergesort",
    ).drop(columns=["_sort_value"])
    return sorted_df


def select_visible_columns(df: pd.DataFrame, visible_columns: list[str]) -> pd.DataFrame:
    if not visible_columns:
        return df
    selected_columns = [column for column in visible_columns if column in df.columns]
    if not selected_columns:
        return df
    pinned_columns = [column for column in PINNED_DISPLAY_COLUMNS if column in selected_columns]
    remaining_columns = [column for column in selected_columns if column not in pinned_columns]
    selected_columns = pinned_columns + remaining_columns
    return df[selected_columns]


def paginate_dataframe(
    df: pd.DataFrame,
    page: int,
    page_size: int,
) -> tuple[list[dict], dict]:
    total_rows = len(df)
    total_pages = max((total_rows + page_size - 1) // page_size, 1)
    current_page = min(max(page, 1), total_pages)
    start_index = (current_page - 1) * page_size
    end_index = start_index + page_size
    paged_rows = df.iloc[start_index:end_index].to_dict(orient="records")
    return paged_rows, {
        "page": current_page,
        "page_size": page_size,
        "total_rows": total_rows,
        "total_pages": total_pages,
        "has_prev": current_page > 1,
        "has_next": current_page < total_pages,
        "start_row": start_index + 1 if total_rows else 0,
        "end_row": min(end_index, total_rows),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    requested_visible_columns = parse_multi_values(request.values.getlist("visible_columns"))
    visible_columns = requested_visible_columns or DEFAULT_VISIBLE_COLUMNS
    form = {
        "action": request.values.get("action", "filter"),
        "max_pe": request.values.get("max_pe", "15"),
        "max_pb": request.values.get("max_pb", "1.5"),
        "min_market_cap_yi": request.values.get("min_market_cap_yi", "30"),
        "top_n": request.values.get("top_n", "50"),
        "category": request.values.get("category", "全部"),
        "industry_keyword": request.values.get("industry_keyword", ""),
        "include_st": parse_bool(request.values.get("include_st")),
        "use_env_proxy": parse_bool(request.values.get("use_env_proxy")),
        "require_kdj_gold_cross": parse_bool(request.values.get("require_kdj_gold_cross")),
        "max_risk_score": request.values.get("max_risk_score", ""),
        "min_rsi6": request.values.get("min_rsi6", ""),
        "max_rsi6": request.values.get("max_rsi6", ""),
        "require_above_ma20": parse_bool(request.values.get("require_above_ma20")),
        "require_bullish_alignment": parse_bool(request.values.get("require_bullish_alignment")),
        "max_distance_to_52w_high": request.values.get("max_distance_to_52w_high", ""),
        "max_drawdown_60": request.values.get("max_drawdown_60", ""),
        "require_relative_strength_positive": parse_bool(
            request.values.get("require_relative_strength_positive")
        ),
        "buy_style": request.values.get("buy_style", "标准"),
        "watch_only": parse_bool(request.values.get("watch_only")),
        "low_absorption_only": parse_bool(request.values.get("low_absorption_only")),
        "active_tab": request.values.get("active_tab", "value"),
        "sort_by": request.values.get("sort_by", "默认"),
        "sort_order": request.values.get("sort_order", "asc"),
        "visible_columns": visible_columns,
        "market_page": str(parse_positive_int(request.values.get("market_page"), 1)),
        "filtered_page": str(parse_positive_int(request.values.get("filtered_page"), 1)),
        "value_page": str(parse_positive_int(request.values.get("value_page"), 1)),
    }

    context = {
        "form": form,
        "summary": None,
        "market_rows": [],
        "filtered_rows": [],
        "value_rows": [],
        "market_pagination": None,
        "filtered_pagination": None,
        "value_pagination": None,
        "sort_options": list(SORT_OPTIONS.keys()),
        "sort_column_to_option": SORT_COLUMN_TO_OPTION,
        "current_sort_column": SORT_OPTIONS.get(form["sort_by"]),
        "default_visible_columns": DEFAULT_VISIBLE_COLUMNS,
        "available_columns": DEFAULT_VISIBLE_COLUMNS + [
            "股息率(%)",
            "ROE(%)",
            "MA20",
            "MA60",
            "RSI6",
            "KDJ金叉",
            "DIF",
            "DEA",
            "60日涨跌幅",
            "年初至今涨跌幅",
            "流通市值(亿元)",
        ],
        "error": None,
    }

    if request.method == "GET":
        config = build_config_from_form(form)
        cached_market_df, snapshot_meta = load_web_snapshot()
        if cached_market_df is not None:
            try:
                snapshot_meta["data_mode"] = "打开页面自动加载本地数据"
                result = build_result_from_market_df(
                    market_df=cached_market_df,
                    config=config,
                    output_dir=BASE_DIR / "output",
                    source_name=snapshot_meta.get("source_name", "本地缓存快照"),
                    snapshot_meta=snapshot_meta,
                    persist_reports=False,
                )
                market_df = enrich_display_metrics(format_output(
                    apply_st_name_filter(result["market_df"], exclude_st=config.exclude_st)
                ))
                filtered_df = enrich_display_metrics(format_output(result["filtered_market_df"]))
                value_df = enrich_display_metrics(format_output(result["low_df"]))
                market_df = select_visible_columns(
                    sort_dataframe(market_df, form["sort_by"], form["sort_order"]),
                    visible_columns,
                )
                filtered_df = select_visible_columns(
                    sort_dataframe(filtered_df, form["sort_by"], form["sort_order"]),
                    visible_columns,
                )
                value_df = select_visible_columns(
                    sort_dataframe(value_df, form["sort_by"], form["sort_order"]),
                    visible_columns,
                )
                context["summary"] = build_summary(result)
                context["market_rows"], context["market_pagination"] = paginate_dataframe(
                    market_df,
                    page=parse_positive_int(form["market_page"], 1),
                    page_size=50,
                )
                context["filtered_rows"], context["filtered_pagination"] = paginate_dataframe(
                    filtered_df,
                    page=parse_positive_int(form["filtered_page"], 1),
                    page_size=50,
                )
                context["value_rows"], context["value_pagination"] = paginate_dataframe(
                    value_df,
                    page=parse_positive_int(form["value_page"], 1),
                    page_size=30,
                )
            except Exception as exc:
                context["error"] = str(exc)

    if request.method == "POST":
        config = build_config_from_form(form)
        try:
            if form["action"] == "refresh":
                result = run_analysis(
                    config=config,
                    output_dir=BASE_DIR / "output",
                    use_env_proxy=form["use_env_proxy"],
                )
                save_web_snapshot(
                    market_df=result["market_df"],
                    source_name=result["source_name"],
                )
                result["snapshot_meta"] = {
                    "refreshed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data_mode": "已重新请求市场数据",
                }
            else:
                cached_market_df, snapshot_meta = load_web_snapshot()
                if cached_market_df is None:
                    result = run_analysis(
                        config=config,
                        output_dir=BASE_DIR / "output",
                        use_env_proxy=form["use_env_proxy"],
                    )
                    save_web_snapshot(
                        market_df=result["market_df"],
                        source_name=result["source_name"],
                    )
                    result["snapshot_meta"] = {
                        "refreshed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "data_mode": "首次无缓存，已自动刷新市场数据",
                    }
                else:
                    snapshot_meta["data_mode"] = "本地快速筛选"
                    result = build_result_from_market_df(
                        market_df=cached_market_df,
                        config=config,
                        output_dir=BASE_DIR / "output",
                        source_name=snapshot_meta.get("source_name", "本地缓存快照"),
                        snapshot_meta=snapshot_meta,
                        persist_reports=False,
                    )
            market_df = enrich_display_metrics(format_output(
                apply_st_name_filter(result["market_df"], exclude_st=config.exclude_st)
            ))
            filtered_df = enrich_display_metrics(format_output(result["filtered_market_df"]))
            value_df = enrich_display_metrics(format_output(result["low_df"]))
            market_df = select_visible_columns(
                sort_dataframe(market_df, form["sort_by"], form["sort_order"]),
                visible_columns,
            )
            filtered_df = select_visible_columns(
                sort_dataframe(filtered_df, form["sort_by"], form["sort_order"]),
                visible_columns,
            )
            value_df = select_visible_columns(
                sort_dataframe(value_df, form["sort_by"], form["sort_order"]),
                visible_columns,
            )
            context["summary"] = build_summary(result)
            context["market_rows"], context["market_pagination"] = paginate_dataframe(
                market_df,
                page=parse_positive_int(form["market_page"], 1),
                page_size=50,
            )
            context["filtered_rows"], context["filtered_pagination"] = paginate_dataframe(
                filtered_df,
                page=parse_positive_int(form["filtered_page"], 1),
                page_size=50,
            )
            context["value_rows"], context["value_pagination"] = paginate_dataframe(
                value_df,
                page=parse_positive_int(form["value_page"], 1),
                page_size=30,
            )
        except Exception as exc:
            context["error"] = str(exc)

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=False, use_reloader=False, threaded=True)
