"""
Phase 3: Statistical Analysis - The Heart of the Project
=========================================================
For each (event, pair), compute: average range, median, std, max, min.

This is what will power the Pine Script predictions.

Run:
    python analyze_events.py
"""

import pandas as pd
import numpy as np

from config import MASTER_FILE, CALENDAR_FILE, STATS_FILE, FOREX_PAIRS


def extract_event_keywords(event_name: str) -> str:
    """
    Normalize event names to group similar events together.
    Example: 'Non-Farm Employment Change' -> 'NFP'
             'Core CPI m/m' -> 'CPI'
    """
    if not isinstance(event_name, str):
        return "Unknown"

    name = event_name.lower()

    # Common important events
    patterns = {
        "NFP": ["non-farm", "nonfarm", "non farm"],
        "CPI": ["cpi", "consumer price"],
        "PPI": ["ppi", "producer price"],
        "FOMC": ["fomc", "federal funds", "fed rate"],
        "ECB": ["ecb", "main refinancing"],
        "BOE": ["boe", "bank rate", "official bank"],
        "BOJ": ["boj", "bank of japan"],
        "BOC": ["boc", "overnight rate"],
        "RBA": ["rba", "cash rate"],
        "GDP": ["gdp", "gross domestic"],
        "Unemployment": ["unemployment"],
        "Retail_Sales": ["retail sales"],
        "PMI": ["pmi", "purchasing managers"],
        "Powell": ["powell", "fed chair"],
        "Lagarde": ["lagarde", "ecb press"],
    }

    for category, keywords in patterns.items():
        if any(kw in name for kw in keywords):
            return category

    return "Other"


def compute_event_statistics(master: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    For every event type × pair combination:
    - Mean range (pips)
    - Median range (pips)
    - Std (volatility of the range itself)
    - Max/Min
    - Sample count (how many times we observed this)
    - Average Range/ATR ratio (is this event abnormally volatile?)
    """
    # Add event category to calendar
    calendar = calendar.copy()
    calendar["Event_Category"] = calendar["Event"].apply(extract_event_keywords)
    calendar["Date"] = pd.to_datetime(calendar["Date"]).dt.normalize()

    # For each (Date, Currency, Event_Category), we have at least one event
    daily_categories = (
        calendar.groupby(["Date", "Currency", "Event_Category"]).size().reset_index(name="n")
    )

    # Pair currencies mapping
    PAIR_CURRENCIES = {
        "EURUSD": ["EUR", "USD"],
        "GBPUSD": ["GBP", "USD"],
        "USDJPY": ["USD", "JPY"],
        "USDCHF": ["USD", "CHF"],
        "AUDUSD": ["AUD", "USD"],
        "USDCAD": ["USD", "CAD"],
        "NZDUSD": ["NZD", "USD"],
        "EURJPY": ["EUR", "JPY"],
    }

    stats_rows = []

    for pair in FOREX_PAIRS.keys():
        df_pair = master[master["Pair"] == pair].copy()
        df_pair["Date"] = pd.to_datetime(df_pair["Date"]).dt.normalize()

        relevant_events = daily_categories[
            daily_categories["Currency"].isin(PAIR_CURRENCIES[pair])
        ]

        # For each event category that affects this pair
        for category in relevant_events["Event_Category"].unique():
            # Dates when this category event occurred (for either currency)
            event_dates = relevant_events[
                relevant_events["Event_Category"] == category
            ]["Date"].unique()

            pair_on_event_days = df_pair[df_pair["Date"].isin(event_dates)]

            if len(pair_on_event_days) < 3:
                continue  # Too few samples to be meaningful

            stats_rows.append(
                {
                    "Pair": pair,
                    "Event_Category": category,
                    "Sample_Size": len(pair_on_event_days),
                    "Mean_Range": round(pair_on_event_days["Range_Pips"].mean(), 1),
                    "Median_Range": round(pair_on_event_days["Range_Pips"].median(), 1),
                    "Std_Range": round(pair_on_event_days["Range_Pips"].std(), 1),
                    "Max_Range": round(pair_on_event_days["Range_Pips"].max(), 1),
                    "Min_Range": round(pair_on_event_days["Range_Pips"].min(), 1),
                    "P25": round(pair_on_event_days["Range_Pips"].quantile(0.25), 1),
                    "P75": round(pair_on_event_days["Range_Pips"].quantile(0.75), 1),
                    "Avg_Range_vs_ATR": round(
                        pair_on_event_days["Range_vs_ATR"].mean(), 2
                    ),
                }
            )

    stats_df = pd.DataFrame(stats_rows)
    return stats_df.sort_values(["Pair", "Mean_Range"], ascending=[True, False])


def compute_baseline(master: pd.DataFrame) -> pd.DataFrame:
    """Days WITHOUT any event - our control group."""
    quiet = master[~master["Has_Event"]]
    baseline = (
        quiet.groupby("Pair")["Range_Pips"]
        .agg(["mean", "median", "std", "count"])
        .round(1)
        .rename(columns={
            "mean": "Baseline_Mean",
            "median": "Baseline_Median",
            "std": "Baseline_Std",
            "count": "Baseline_N",
        })
    )
    return baseline


def main():
    print("Running statistical analysis...")
    master = pd.read_csv(MASTER_FILE, parse_dates=["Date"])
    calendar = pd.read_csv(CALENDAR_FILE, parse_dates=["Date"])

    # Event statistics
    stats = compute_event_statistics(master, calendar)
    stats.to_csv(STATS_FILE, index=False)
    print(f"\nSaved {len(stats)} event statistics to {STATS_FILE}")

    # Baseline (quiet days)
    baseline = compute_baseline(master)
    print(f"\n=== BASELINE (quiet days, no events) ===")
    print(baseline)

    # Top movers
    print(f"\n=== TOP 20 EVENT IMPACTS (by mean range) ===")
    print(stats.head(20).to_string(index=False))

    # Per pair: which event moves it most?
    print(f"\n=== MOST IMPACTFUL EVENT PER PAIR ===")
    top_per_pair = stats.loc[stats.groupby("Pair")["Mean_Range"].idxmax()]
    print(top_per_pair[["Pair", "Event_Category", "Mean_Range", "Sample_Size"]].to_string(index=False))


if __name__ == "__main__":
    main()
