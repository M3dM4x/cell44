"""
Phase 4a: Weekly Prediction Generator
======================================
For the upcoming week, for each high-impact event:
  - Which pairs get affected
  - Expected range (mean, median, p25-p75)
  - Confidence level (based on sample size)

Outputs: predictions.json (consumed by dashboard + telegram bot)

Run:
    python generate_predictions.py
"""

import json
import pandas as pd
from datetime import datetime, timedelta

from config import STATS_FILE, CALENDAR_FILE, DATA_DIR
from analyze_events import extract_event_keywords


# Same mapping as build_master.py
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


def confidence_level(sample_size: int) -> str:
    """Simple heuristic for confidence."""
    if sample_size >= 30:
        return "High"
    if sample_size >= 15:
        return "Medium"
    if sample_size >= 8:
        return "Low"
    return "Very Low"


def parse_number(value) -> float:
    """Parse numeric string like '2.5%', '140K', '-€1.2B' to float."""
    if value is None or value == "" or (isinstance(value, float) and value != value):
        return None
    try:
        s = str(value).strip()
        if not s or s.lower() in ("n/a", "tentative"):
            return None
        s = s.replace(",", "").replace("%", "").replace("€", "").replace("$", "").replace("£", "").replace("¥", "")
        
        multiplier = 1
        if s.endswith("K") or s.endswith("k"):
            multiplier = 1_000
            s = s[:-1]
        elif s.endswith("M") or s.endswith("m"):
            multiplier = 1_000_000
            s = s[:-1]
        elif s.endswith("B") or s.endswith("b"):
            multiplier = 1_000_000_000
            s = s[:-1]
        elif s.endswith("T") or s.endswith("t"):
            multiplier = 1_000_000_000_000
            s = s[:-1]
        
        return float(s) * multiplier
    except (ValueError, TypeError):
        return None


def predict_direction(forecast, previous, currency: str, event_name: str) -> dict:
    """Predict direction impact on the currency based on forecast vs previous.
    
    Logic: if forecast > previous, currency likely strengthens (bullish for that currency).
    If forecast < previous, currency likely weakens (bearish).
    
    Returns dict with prediction for each pair direction.
    """
    f = parse_number(forecast)
    p = parse_number(previous)
    
    result = {
        "forecast_num": f,
        "previous_num": p,
        "bias": "neutral",
        "bias_ar": "محايد",
        "confidence": "low",
        "reasoning_ar": "البيانات غير كافية للتنبؤ بالاتجاه",
    }
    
    if f is None or p is None:
        result["reasoning_ar"] = "لم تتوفر بيانات Forecast أو Previous للتحليل المسبق"
        return result
    
    event_lower = event_name.lower()
    is_inverse = any(keyword in event_lower for keyword in [
        "unemployment rate", "claimant count", "jobless", "inflation rate"
    ])
    
    diff_pct = ((f - p) / abs(p)) * 100 if p != 0 else 0
    
    if abs(diff_pct) < 2:
        result["bias"] = "neutral"
        result["bias_ar"] = "محايد"
        result["reasoning_ar"] = f"الفارق بين المتوقع ({f}) والسابق ({p}) صغير جداً، لا اتجاه واضح"
        return result
    
    if is_inverse:
        if f > p:
            result["bias"] = "bearish"
            result["bias_ar"] = f"سلبي لـ {currency}"
            result["reasoning_ar"] = f"المتوقع {f} أعلى من السابق {p} — ارتفاع البطالة/التضخم يضعف العملة"
        else:
            result["bias"] = "bullish"
            result["bias_ar"] = f"إيجابي لـ {currency}"
            result["reasoning_ar"] = f"المتوقع {f} أقل من السابق {p} — انخفاض البطالة/التضخم يقوي العملة"
    else:
        if f > p:
            result["bias"] = "bullish"
            result["bias_ar"] = f"إيجابي لـ {currency}"
            result["reasoning_ar"] = f"المتوقع {f} أعلى من السابق {p} — نمو اقتصادي يقوي العملة"
        else:
            result["bias"] = "bearish"
            result["bias_ar"] = f"سلبي لـ {currency}"
            result["reasoning_ar"] = f"المتوقع {f} أقل من السابق {p} — تباطؤ اقتصادي يضعف العملة"
    
    result["confidence"] = "high" if abs(diff_pct) > 10 else "medium"
    return result


def pair_direction(event_currency: str, pair: str, bias: str) -> str:
    """Determine if the pair will go up or down based on event currency impact.
    
    If event is bullish for USD and pair is EURUSD: USD strength = pair goes DOWN
    If event is bullish for USD and pair is USDJPY: USD strength = pair goes UP
    """
    if bias == "neutral":
        return "neutral"
    
    base = pair[:3]
    quote = pair[3:]
    
    if event_currency == base:
        return "up" if bias == "bullish" else "down"
    elif event_currency == quote:
        return "down" if bias == "bullish" else "up"
    else:
        return "neutral"


def fetch_upcoming_events() -> pd.DataFrame:
    """
    Fetch next 7 days of economic calendar events.
    Uses our custom ff_scraper with cloudscraper (no external calendar tool).
    """
    from ff_scraper import scrape_week, get_mondays, create_session

    today = datetime.now()
    next_week = today + timedelta(days=7)

    print(f"Fetching upcoming events: {today.date()} -> {next_week.date()}")

    session = create_session()
    all_events = []
    for monday in get_mondays(today.strftime("%Y-%m-%d"), next_week.strftime("%Y-%m-%d")):
        events = scrape_week(monday, session)
        all_events.extend(events)

    if not all_events:
        return pd.DataFrame()

    df = pd.DataFrame(all_events)
    df["DateTime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce",
    )
    df = df[df["Impact"] == "High"]
    df = df[df["Currency"].isin(["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"])]

    # Keep only events within our window
    df = df[(df["DateTime"] >= today) & (df["DateTime"] <= next_week)]

    return df.reset_index(drop=True)


def predict_ranges(upcoming: pd.DataFrame, stats: pd.DataFrame) -> list:
    """
    For each upcoming event, predict the range for every affected pair.
    Returns a list of dicts ready for JSON serialization.
    """
    predictions = []

    for _, event in upcoming.iterrows():
        category = extract_event_keywords(event["Event"])
        currency = event["Currency"]
        event_dt = event["DateTime"]
        event_name = event["Event"]
        forecast_val = event.get("Forecast", "")
        previous_val = event.get("Previous", "")

        direction_info = predict_direction(forecast_val, previous_val, currency, event_name)

        affected_pairs = [p for p, curs in PAIR_CURRENCIES.items() if currency in curs]

        event_predictions = {
            "datetime": event_dt.isoformat() if pd.notna(event_dt) else None,
            "date": event_dt.strftime("%Y-%m-%d") if pd.notna(event_dt) else None,
            "day": event_dt.strftime("%A") if pd.notna(event_dt) else None,
            "time": event_dt.strftime("%H:%M") if pd.notna(event_dt) else None,
            "currency": currency,
            "event": event_name,
            "category": category,
            "forecast": str(forecast_val),
            "previous": str(previous_val),
            "direction": direction_info,
            "pairs": [],
        }

        for pair in affected_pairs:
            match = stats[(stats["Pair"] == pair) & (stats["Event_Category"] == category)]

            if match.empty:
                continue

            row = match.iloc[0]
            
            pair_dir = pair_direction(currency, pair, direction_info["bias"])
            
            if pair_dir == "up":
                pair_dir_ar = f"صعود محتمل لـ {pair}"
            elif pair_dir == "down":
                pair_dir_ar = f"هبوط محتمل لـ {pair}"
            else:
                pair_dir_ar = "اتجاه غير محدد"
            
            event_predictions["pairs"].append(
                {
                    "pair": pair,
                    "mean_range": float(row["Mean_Range"]),
                    "median_range": float(row["Median_Range"]),
                    "p25": float(row["P25"]),
                    "p75": float(row["P75"]),
                    "max_observed": float(row["Max_Range"]),
                    "sample_size": int(row["Sample_Size"]),
                    "confidence": confidence_level(int(row["Sample_Size"])),
                    "vs_normal_day": float(row["Avg_Range_vs_ATR"]),
                    "direction": pair_dir,
                    "direction_ar": pair_dir_ar,
                }
            )

        event_predictions["pairs"].sort(key=lambda x: x["mean_range"], reverse=True)

        if event_predictions["pairs"]:
            predictions.append(event_predictions)

    predictions.sort(key=lambda x: x["datetime"] or "")
    return predictions


def main():
    print("Generating weekly predictions...")

    # Load historical stats
    stats = pd.read_csv(STATS_FILE)
    print(f"Loaded {len(stats)} statistical records")

    # Fetch upcoming events
    try:
        upcoming = fetch_upcoming_events()
        print(f"Found {len(upcoming)} high-impact events in the next 7 days")
    except Exception as e:
        print(f"ERROR fetching upcoming events: {e}")
        print("Falling back to empty prediction (dashboard will show 'no data')")
        upcoming = pd.DataFrame()

    # Build predictions
    predictions = predict_ranges(upcoming, stats) if not upcoming.empty else []

    # Wrap with metadata
    output = {
        "generated_at": datetime.now().isoformat(),
        "week_start": datetime.now().strftime("%Y-%m-%d"),
        "week_end": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
        "total_events": len(predictions),
        "events": predictions,
    }

    # Save JSON
    output_file = f"{DATA_DIR}/predictions.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nSaved {len(predictions)} event predictions to {output_file}")

    # Print preview
    if predictions:
        print("\n=== PREVIEW ===")
        for p in predictions[:3]:
            print(f"\n{p['day']} {p['date']} {p['time']} - {p['currency']} {p['event']}")
            for pair_pred in p["pairs"][:3]:
                print(
                    f"  {pair_pred['pair']:>7}: "
                    f"~{pair_pred['mean_range']:.0f} pips "
                    f"(range {pair_pred['p25']:.0f}-{pair_pred['p75']:.0f}) "
                    f"[{pair_pred['confidence']}]"
                )


if __name__ == "__main__":
    main()
