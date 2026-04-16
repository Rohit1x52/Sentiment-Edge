import pandas as pd
import numpy as np
from scipy import stats


def sentiment_pnl_correlation(merged_df):
    r, p = stats.pearsonr(
        merged_df["sentiment_value"].dropna(),
        merged_df.loc[merged_df["sentiment_value"].notna(), "closed_pnl"]
    )
    return {"pearson_r": round(r, 4), "p_value": round(p, 6)}


def leverage_risk_profile(merged_df):
    grp = merged_df.groupby("leverage_bucket", observed=True)
    result = pd.DataFrame({
        "trade_count":    grp.size(),
        "avg_pnl":        grp["closed_pnl"].mean().round(2),
        "win_rate":       (grp["pnl_positive"].mean() * 100).round(1),
        "liq_rate":       (grp["is_liquidation"].mean() * 100).round(2),
        "avg_loss":       grp["closed_pnl"].apply(lambda x: x[x < 0].mean()).round(2),
    }).reset_index()
    return result


def fear_vs_greed_behavior(merged_df):
    fear_mask  = merged_df["sentiment_label"].isin(["Extreme Fear", "Fear"])
    greed_mask = merged_df["sentiment_label"].isin(["Greed", "Extreme Greed"])

    def metrics(mask):
        sub = merged_df[mask]
        return {
            "trades":        len(sub),
            "avg_pnl":       round(sub["closed_pnl"].mean(), 2),
            "win_rate":      round(sub["pnl_positive"].mean() * 100, 1),
            "avg_leverage":  round(sub["leverage"].mean(), 2),
            "liq_rate":      round(sub["is_liquidation"].mean() * 100, 2),
            "long_ratio":    round(sub["is_long"].mean() * 100, 1),
        }

    return {
        "fear_periods":  metrics(fear_mask),
        "greed_periods": metrics(greed_mask),
    }


def contrarian_signal(merged_df):
    # check if trading against sentiment (long during fear, short during greed) is profitable
    long_fear  = merged_df[(merged_df["is_long"] == 1) &
                           (merged_df["sentiment_label"].isin(["Extreme Fear", "Fear"]))]
    short_greed = merged_df[(merged_df["is_short"] == 1) &
                            (merged_df["sentiment_label"].isin(["Greed", "Extreme Greed"]))]

    long_trend  = merged_df[(merged_df["is_long"] == 1) &
                            (merged_df["sentiment_label"].isin(["Greed", "Extreme Greed"]))]
    short_trend = merged_df[(merged_df["is_short"] == 1) &
                            (merged_df["sentiment_label"].isin(["Extreme Fear", "Fear"]))]

    def safe_mean(df):
        return round(df["closed_pnl"].mean(), 2) if len(df) > 0 else 0.0

    return {
        "contrarian_long_fear_avg_pnl":   safe_mean(long_fear),
        "contrarian_short_greed_avg_pnl": safe_mean(short_greed),
        "trend_long_greed_avg_pnl":       safe_mean(long_trend),
        "trend_short_fear_avg_pnl":       safe_mean(short_trend),
    }


def top_performer_sentiment_profile(merged_df, trader_stats_df, top_n=10):
    top_accounts = trader_stats_df.nlargest(top_n, "total_pnl")["account"].tolist()
    top_trades   = merged_df[merged_df["account"].isin(top_accounts)]

    profile = top_trades.groupby("sentiment_label").agg(
        trade_count=("closed_pnl", "count"),
        avg_pnl=("closed_pnl", "mean"),
        win_rate=("pnl_positive", "mean"),
    ).reset_index()

    return profile


def print_insights(merged_df, trader_stats_df):
    print("\n" + "=" * 55)
    print("KEY INSIGHTS")
    print("=" * 55)

    corr = sentiment_pnl_correlation(merged_df)
    print(f"\nSentiment vs PnL Correlation: r={corr['pearson_r']}  p={corr['p_value']}")
    if abs(corr["pearson_r"]) > 0.1:
        direction = "positive" if corr["pearson_r"] > 0 else "negative"
        print(f"  -> {direction} relationship between sentiment and trade PnL")

    fvg = fear_vs_greed_behavior(merged_df)
    print("\nFear vs Greed Periods:")
    for period, m in fvg.items():
        print(f"  {period}:")
        print(f"    avg pnl={m['avg_pnl']}  win_rate={m['win_rate']}%  "
              f"liq_rate={m['liq_rate']}%  long_ratio={m['long_ratio']}%")

    cs = contrarian_signal(merged_df)
    print("\nContrarian vs Trend-following signals:")
    print(f"  Long during Fear  avg_pnl: {cs['contrarian_long_fear_avg_pnl']}")
    print(f"  Short during Greed avg_pnl: {cs['contrarian_short_greed_avg_pnl']}")
    print(f"  Long during Greed avg_pnl: {cs['trend_long_greed_avg_pnl']}")
    print(f"  Short during Fear avg_pnl: {cs['trend_short_fear_avg_pnl']}")

    lev = leverage_risk_profile(merged_df)
    print("\nLeverage Risk Profile:")
    print(lev.to_string(index=False))

    print("\nTop 10 Traders by PnL:")
    top = trader_stats_df.nlargest(10, "total_pnl")[
        ["account", "total_pnl", "win_rate", "total_trades", "avg_leverage", "liquidations"]
    ].copy()
    top["account"]   = top["account"].str[:12]
    top["total_pnl"] = top["total_pnl"].round(0)
    top["win_rate"]  = (top["win_rate"] * 100).round(1)
    print(top.to_string(index=False))
    print("=" * 55)