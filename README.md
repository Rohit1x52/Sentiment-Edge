# Bitcoin Trader Sentiment Analysis

Explores the relationship between Hyperliquid trader performance and the Bitcoin Fear & Greed Index. The goal is to find patterns in how market sentiment affects trade outcomes, leverage usage, and liquidation risk.

## Setup

```bash
pip install -r requirements.txt
```

Place your datasets in `Datasets/`:
- `historical_data.csv` — Hyperliquid trade data
- `fear_greed_index.csv` — Bitcoin Fear & Greed Index

Download links:
- https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs/view?usp=sharing
- https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf/view?usp=sharing

## Usage

```bash
# full run with real data
python run.py

# skip dashboard, just run analysis
python run.py --no-dashboard
```

## Structure

```
Sentiment Edge/
├── Datasets/              # input CSVs
├── utils/
│   ├── loader.py          # data loading and merging
│   └── features.py        # feature engineering
├── analysis/
│   ├── eda.py             # exploratory charts
│   └── insights.py        # statistical analysis
├── dashboard/
│   └── app.py             # Streamlit dashboard
├── run.py                 # pipeline runner
└── requirements.txt
```

## Key Questions Answered

- Do traders perform better during fear or greed periods?
- Does higher leverage lead to more liquidations across sentiment regimes?
- Is contrarian trading (long during fear, short during greed) profitable?
- Which symbols perform best under different sentiment conditions?
- What separates top-performing traders from the rest?

## Notes

The Fear & Greed Index is matched to trades by date. Days with missing sentiment data are forward-filled. Liquidation events are flagged separately from normal closes.

Made by Rohit Ranjan Kumar. 
Contact ranjanrohit908@gmail.com