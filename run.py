import os
import sys
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Generate and use sample data")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip launching dashboard")
    args = parser.parse_args()

    if args.sample or not (
        os.path.exists("Datasets/historical_data.csv") and
        os.path.exists("Datasets/fear_greed_index.csv")
    ):
        pass

    print("Loading data...")
    from utils.loader import load_fear_greed, load_trades, merge_with_sentiment
    from utils.features import add_trade_features, trader_stats

    fg     = load_fear_greed()
    trades = load_trades()
    merged = merge_with_sentiment(trades, fg)
    merged = add_trade_features(merged)
    t_stats = trader_stats(merged)

    print("Running EDA...")
    from analysis.eda import run_all
    run_all(merged, fg, t_stats)

    print("Computing insights...")
    from analysis.insights import print_insights
    print_insights(merged, t_stats)

    merged.to_csv("data/processed/merged_trades.csv", index=False)
    t_stats.to_csv("data/processed/trader_stats.csv", index=False)
    print("\nProcessed data saved to data/processed/")

    if not args.no_dashboard:
        print("\nLaunching dashboard...")
        os.system("streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
    