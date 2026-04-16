import pandas as pd
import numpy as np


def add_trade_features(df):
    df = df.copy()

    if "event" not in df.columns:
        df["event"] = "TRADE"
    if "leverage" not in df.columns:
        df["leverage"] = 1.0
    if "side" not in df.columns:
        df["side"] = "UNKNOWN"

    df["notional_value"] = df["execution_price"] * df["size"]
    df["is_long"]  = (df["side"] == "BUY").astype(int)
    df["is_short"] = (df["side"] == "SELL").astype(int)
    df["is_liquidation"] = (df["event"] == "LIQUIDATION").astype(int)
    df["pnl_positive"] = (df["closed_pnl"] > 0).astype(int)

    df["leverage_bucket"] = pd.cut(
        df["leverage"],
        bins=[0, 2, 5, 10, 25, 200],
        labels=["1-2x", "3-5x", "6-10x", "11-25x", "25x+"]
    )

    df["sentiment_bucket"] = pd.cut(
        df["sentiment_value"],
        bins=[0, 24, 44, 55, 74, 100],
        labels=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    )

    return df


def trader_stats(df):
    grp = df.groupby("account")

    stats = pd.DataFrame({
        "total_trades":    grp.size(),
        "total_pnl":       grp["closed_pnl"].sum(),
        "avg_pnl":         grp["closed_pnl"].mean(),
        "win_rate":        grp["pnl_positive"].mean(),
        "avg_leverage":    grp["leverage"].mean(),
        "max_leverage":    grp["leverage"].max(),
        "liquidations":    grp["is_liquidation"].sum(),
        "total_volume":    grp["notional_value"].sum(),
        "symbols_traded":  grp["symbol"].nunique(),
        "long_ratio":      grp["is_long"].mean(),
    }).reset_index()

    stats["profitable"] = (stats["total_pnl"] > 0).astype(int)
    stats["liq_rate"]   = stats["liquidations"] / stats["total_trades"]

    return stats


def sentiment_performance(df):
    grp = df.groupby("sentiment_label")
    result = pd.DataFrame({
        "trade_count":      grp.size(),
        "avg_pnl":          grp["closed_pnl"].mean(),
        "total_pnl":        grp["closed_pnl"].sum(),
        "win_rate":         grp["pnl_positive"].mean(),
        "avg_leverage":     grp["leverage"].mean(),
        "liquidation_rate": grp["is_liquidation"].mean(),
        "long_ratio":       grp["is_long"].mean(),
    }).reset_index()

    order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    result["sentiment_label"] = pd.Categorical(result["sentiment_label"], categories=order, ordered=True)
    result = result.sort_values("sentiment_label").reset_index(drop=True)
    return result