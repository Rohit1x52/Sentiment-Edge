import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="darkgrid", palette="muted")
os.makedirs("data/processed", exist_ok=True)


def plot_sentiment_distribution(fg_df):
    counts = fg_df["classification"].value_counts()
    order  = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    counts = counts.reindex([o for o in order if o in counts.index])

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1a7a1a"])
    ax.set_title("Fear & Greed Index — Day Count by Classification")
    ax.set_xlabel("Classification")
    ax.set_ylabel("Days")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig("data/processed/sentiment_distribution.png", dpi=120)
    plt.close()
    print("saved: sentiment_distribution.png")


def plot_sentiment_over_time(fg_df):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(fg_df["date"], fg_df["value"], linewidth=0.8, color="#5878b0")
    ax.axhline(25, color="#d62728", linestyle="--", linewidth=0.7, alpha=0.6, label="Fear boundary")
    ax.axhline(75, color="#2ca02c", linestyle="--", linewidth=0.7, alpha=0.6, label="Greed boundary")
    ax.fill_between(fg_df["date"], fg_df["value"], 50,
                    where=(fg_df["value"] < 50), alpha=0.15, color="#d62728")
    ax.fill_between(fg_df["date"], fg_df["value"], 50,
                    where=(fg_df["value"] >= 50), alpha=0.15, color="#2ca02c")
    ax.set_title("Bitcoin Fear & Greed Index Over Time")
    ax.set_ylabel("Index Value (0-100)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("data/processed/sentiment_over_time.png", dpi=120)
    plt.close()
    print("saved: sentiment_over_time.png")


def plot_pnl_by_sentiment(merged_df):
    order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    present = [o for o in order if o in merged_df["sentiment_label"].values]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    avg_pnl = merged_df.groupby("sentiment_label")["closed_pnl"].mean().reindex(present)
    colors  = ["#d62728" if v < 0 else "#2ca02c" for v in avg_pnl.values]
    axes[0].bar(avg_pnl.index, avg_pnl.values, color=colors)
    axes[0].set_title("Avg PnL per Trade by Sentiment")
    axes[0].set_ylabel("Avg Closed PnL (USD)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].axhline(0, color="white", linewidth=0.8)

    win_rate = merged_df.groupby("sentiment_label")["pnl_positive"].mean().reindex(present) * 100
    axes[1].bar(win_rate.index, win_rate.values, color="#5878b0")
    axes[1].set_title("Win Rate (%) by Sentiment")
    axes[1].set_ylabel("Win Rate %")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].axhline(50, color="orange", linestyle="--", linewidth=0.8, label="50% baseline")
    axes[1].legend(fontsize=8)

    plt.suptitle("Trader Performance vs Market Sentiment", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("data/processed/pnl_by_sentiment.png", dpi=120)
    plt.close()
    print("saved: pnl_by_sentiment.png")


def plot_leverage_analysis(merged_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    present = [o for o in order if o in merged_df["sentiment_label"].values]

    avg_lev = merged_df.groupby("sentiment_label")["leverage"].mean().reindex(present)
    axes[0].bar(avg_lev.index, avg_lev.values, color="#9467bd")
    axes[0].set_title("Avg Leverage Used by Sentiment")
    axes[0].set_ylabel("Avg Leverage")
    axes[0].tick_params(axis="x", rotation=20)

    liq_rate = merged_df.groupby("sentiment_label")["is_liquidation"].mean().reindex(present) * 100
    axes[1].bar(liq_rate.index, liq_rate.values, color="#d62728")
    axes[1].set_title("Liquidation Rate (%) by Sentiment")
    axes[1].set_ylabel("Liquidation Rate %")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig("data/processed/leverage_analysis.png", dpi=120)
    plt.close()
    print("saved: leverage_analysis.png")


def plot_top_traders(trader_stats_df):
    top = trader_stats_df.nlargest(15, "total_pnl")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(top["account"].str[:10], top["total_pnl"], color="#2ca02c")
    axes[0].set_title("Top 15 Traders by Total PnL")
    axes[0].set_xlabel("Total PnL (USD)")

    axes[1].scatter(trader_stats_df["win_rate"] * 100,
                    trader_stats_df["total_pnl"],
                    alpha=0.4, color="#5878b0", s=20)
    axes[1].set_title("Win Rate vs Total PnL")
    axes[1].set_xlabel("Win Rate %")
    axes[1].set_ylabel("Total PnL (USD)")
    axes[1].axhline(0, color="orange", linewidth=0.7, linestyle="--")

    plt.tight_layout()
    plt.savefig("data/processed/top_traders.png", dpi=120)
    plt.close()
    print("saved: top_traders.png")


def plot_symbol_heatmap(merged_df):
    order  = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    pivot  = merged_df.groupby(["symbol", "sentiment_label"])["closed_pnl"].mean().unstack()
    pivot  = pivot[[c for c in order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                linewidths=0.4, ax=ax)
    ax.set_title("Avg PnL by Symbol and Market Sentiment")
    ax.set_xlabel("Market Sentiment")
    ax.set_ylabel("Symbol")
    plt.tight_layout()
    plt.savefig("data/processed/symbol_sentiment_heatmap.png", dpi=120)
    plt.close()
    print("saved: symbol_sentiment_heatmap.png")


def run_all(merged_df, fg_df, trader_stats_df):
    print("\nRunning EDA...")
    plot_sentiment_distribution(fg_df)
    plot_sentiment_over_time(fg_df)
    plot_pnl_by_sentiment(merged_df)
    plot_leverage_analysis(merged_df)
    plot_top_traders(trader_stats_df)
    plot_symbol_heatmap(merged_df)
    print("All EDA charts saved to data/processed/")