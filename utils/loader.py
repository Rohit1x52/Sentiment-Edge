import pandas as pd
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


def _resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


def load_fear_greed(path="Datasets/fear_greed_index.csv"):
    df = pd.read_csv(_resolve_path(path))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)

    if "classification" not in df.columns:
        raise ValueError("fear_greed.csv missing 'classification' column")

    df["classification"] = df["classification"].str.strip()
    return df


def load_trades(path="Datasets/historical_data.csv"):
    df = pd.read_csv(_resolve_path(path))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "timestamp_ist" in df.columns:
        df["time"] = df["timestamp_ist"]
    elif "timestamp" in df.columns:
        df["time"] = df["timestamp"]

    rename_map = {
        "px": "execution_price",
        "price": "execution_price",
        "sz": "size",
        "size_tokens": "size",
        "coin": "symbol",
        "closedpnl": "closed_pnl",
    }
    df.rename(columns=rename_map, inplace=True)

    df = df.loc[:, ~df.columns.duplicated()]

    if "timestamp" in df.columns:
        df.drop(columns=["timestamp"], inplace=True)
    if "timestamp_ist" in df.columns:
        df.drop(columns=["timestamp_ist"], inplace=True)

    if "time" not in df.columns:
        raise ValueError("Trades CSV missing time/timestamp column")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["date"] = df["time"].dt.normalize()

    numeric_cols = ["execution_price", "size", "closed_pnl", "leverage", "start_position", "size_usd"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "leverage" not in df.columns:
        df["leverage"] = 1.0
    if "start_position" not in df.columns:
        df["start_position"] = 0.0
    if "event" not in df.columns:
        df["event"] = "TRADE"
    if "side" not in df.columns:
        df["side"] = "UNKNOWN"
    if "symbol" not in df.columns:
        df["symbol"] = "UNKNOWN"
    if "account" not in df.columns:
        df["account"] = "UNKNOWN"

    df["side"] = df["side"].astype(str).str.upper().str.strip()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    df = df.dropna(subset=["execution_price", "size", "closed_pnl"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def merge_with_sentiment(trades_df, fg_df):
    merged = trades_df.merge(
        fg_df[["date", "value", "classification"]],
        on="date",
        how="left"
    )
    merged.rename(columns={"value": "sentiment_value",
                            "classification": "sentiment_label"}, inplace=True)
    merged["sentiment_value"] = merged["sentiment_value"].ffill()
    merged["sentiment_label"] = merged["sentiment_label"].ffill()
    return merged