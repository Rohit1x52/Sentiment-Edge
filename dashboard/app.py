import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.loader import load_fear_greed, load_trades, merge_with_sentiment
from utils.features import add_trade_features, trader_stats, sentiment_performance
from analysis.insights import (
    fear_vs_greed_behavior, contrarian_signal,
    leverage_risk_profile, top_performer_sentiment_profile
)

st.set_page_config(page_title="Trader Sentiment Analysis", layout="wide")

SENTIMENT_ORDER  = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
SENTIMENT_COLORS = {
    "Extreme Fear": "#d62728", "Fear": "#ff7f0e",
    "Neutral": "#bcbd22", "Greed": "#2ca02c", "Extreme Greed": "#1a7a1a"
}


@st.cache_data
def load_all():
    fg     = load_fear_greed()
    trades = load_trades()
    merged = merge_with_sentiment(trades, fg)
    merged = add_trade_features(merged)
    t_stats = trader_stats(merged)
    return fg, trades, merged, t_stats


fg, trades, merged, t_stats = load_all()

# sidebar filters
st.sidebar.title("Filters")
symbols  = ["All"] + sorted(merged["symbol"].unique().tolist())
selected_symbol = st.sidebar.selectbox("Symbol", symbols)

sentiments = ["All"] + SENTIMENT_ORDER
selected_sent = st.sidebar.selectbox("Sentiment Period", sentiments)

min_trades = st.sidebar.slider("Min trades per account", 1, 50, 5)

df = merged.copy()
if selected_symbol != "All":
    df = df[df["symbol"] == selected_symbol]
if selected_sent != "All":
    df = df[df["sentiment_label"] == selected_sent]

sent_perf = sentiment_performance(df)

filtered_stats = trader_stats(df)
filtered_stats = filtered_stats[filtered_stats["total_trades"] >= min_trades]

# header
st.title("Bitcoin Trader Performance vs Market Sentiment")
st.caption("Hyperliquid historical trades analyzed against the Fear & Greed Index")

# KPI row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Trades",     f"{len(df):,}")
c2.metric("Unique Traders",   f"{df['account'].nunique():,}")
c3.metric("Total PnL",        f"${df['closed_pnl'].sum():,.0f}")
c4.metric("Avg Win Rate",     f"{df['pnl_positive'].mean()*100:.1f}%")
c5.metric("Liquidations",     f"{df['is_liquidation'].sum():,}")

st.markdown("---")

# row 1: sentiment over time + pnl by sentiment
col1, col2 = st.columns([2, 1])

with col1:
    fig = px.line(fg, x="date", y="value", title="Fear & Greed Index Over Time",
                  color_discrete_sequence=["#5878b0"])
    fig.add_hline(y=25, line_dash="dot", line_color="#d62728", opacity=0.5)
    fig.add_hline(y=75, line_dash="dot", line_color="#2ca02c", opacity=0.5)
    fig.update_layout(height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    present = [s for s in SENTIMENT_ORDER if s in sent_perf["sentiment_label"].values]
    sp_plot = sent_perf[sent_perf["sentiment_label"].isin(present)]
    fig2 = px.bar(sp_plot, x="sentiment_label", y="avg_pnl",
                  title="Avg PnL by Sentiment",
                  color="sentiment_label",
                  color_discrete_map=SENTIMENT_COLORS,
                  category_orders={"sentiment_label": present})
    fig2.update_layout(showlegend=False, height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig2, use_container_width=True)

# row 2: win rate + leverage heatmap
col3, col4 = st.columns(2)

with col3:
    fig3 = px.bar(sp_plot, x="sentiment_label", y="win_rate",
                  title="Win Rate by Sentiment Period",
                  color="sentiment_label",
                  color_discrete_map=SENTIMENT_COLORS,
                  category_orders={"sentiment_label": present})
    fig3.add_hline(y=0.5, line_dash="dash", line_color="orange", opacity=0.6)
    fig3.update_layout(showlegend=False, height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    lev_fig_data = sp_plot[["sentiment_label", "avg_leverage", "liquidation_rate"]].copy()
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(name="Avg Leverage", x=lev_fig_data["sentiment_label"],
                          y=lev_fig_data["avg_leverage"], marker_color="#9467bd"))
    fig4.add_trace(go.Scatter(name="Liq Rate %", x=lev_fig_data["sentiment_label"],
                              y=lev_fig_data["liquidation_rate"] * 100,
                              yaxis="y2", mode="lines+markers",
                              line=dict(color="#d62728", width=2)))
    fig4.update_layout(
        title="Avg Leverage & Liquidation Rate",
        yaxis2=dict(overlaying="y", side="right", title="Liq Rate %"),
        height=280, margin=dict(t=40, b=20), legend=dict(orientation="h")
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# symbol heatmap
st.subheader("Avg PnL by Symbol and Sentiment")
pivot = df.groupby(["symbol", "sentiment_label"])["closed_pnl"].mean().unstack()
pivot = pivot[[c for c in SENTIMENT_ORDER if c in pivot.columns]]
fig5  = px.imshow(pivot, color_continuous_scale="RdYlGn", text_auto=".1f",
                  aspect="auto", title="Avg PnL Heatmap")
fig5.update_layout(height=320, margin=dict(t=40, b=20))
st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# contrarian vs trend
st.subheader("Contrarian vs Trend-following Strategy")
cs = contrarian_signal(df)
cc1, cc2, cc3, cc4 = st.columns(4)
cc1.metric("Long in Fear",    f"${cs['contrarian_long_fear_avg_pnl']}", help="Avg PnL going long during fear")
cc2.metric("Short in Greed",  f"${cs['contrarian_short_greed_avg_pnl']}", help="Avg PnL going short during greed")
cc3.metric("Long in Greed",   f"${cs['trend_long_greed_avg_pnl']}", help="Avg PnL going long during greed")
cc4.metric("Short in Fear",   f"${cs['trend_short_fear_avg_pnl']}", help="Avg PnL going short during fear")

st.markdown("---")

# trader leaderboard
st.subheader("Trader Leaderboard")
display_cols = ["account", "total_pnl", "win_rate", "total_trades",
                "avg_leverage", "liquidations", "liq_rate", "profitable"]
leaderboard = filtered_stats[display_cols].copy()
leaderboard["win_rate"]  = (leaderboard["win_rate"]  * 100).round(1)
leaderboard["liq_rate"]  = (leaderboard["liq_rate"]  * 100).round(2)
leaderboard["total_pnl"] = leaderboard["total_pnl"].round(0)
leaderboard = leaderboard.sort_values("total_pnl", ascending=False).reset_index(drop=True)
st.dataframe(leaderboard, use_container_width=True, height=350)

# scatter: win rate vs pnl
fig6 = px.scatter(filtered_stats, x="win_rate", y="total_pnl",
                  size="total_trades", color="avg_leverage",
                  hover_data=["account", "total_trades", "liquidations"],
                  title="Win Rate vs Total PnL (size = trade count, color = avg leverage)",
                  color_continuous_scale="RdYlGn_r",
                  labels={"win_rate": "Win Rate", "total_pnl": "Total PnL (USD)"})
fig6.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
fig6.add_vline(x=0.5, line_dash="dash", line_color="orange", opacity=0.3)
fig6.update_layout(height=400)
st.plotly_chart(fig6, use_container_width=True)