"""CELL 44 Weekly COT Runner

Triggered by GitHub Actions cron Monday 09:00 Casablanca (08:00 UTC).
Fetches CFTC Commitments of Traders for all 10 BIAS_PAIRS, then generates
AI insight for each pair (10 Gemini calls total).

Output: pipeline/data/cot.json (top-level cot.json copied by workflow).
"""

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests

CASA_TZ = timezone(timedelta(hours=1), name="Africa/Casablanca")

DATA_DIR = Path(__file__).parent / "data"
COT_FILE = DATA_DIR / "cot.json"
RETENTION_WEEKS = 8

CFTC_BASE = "https://publicreporting.cftc.gov/resource/jun7-fc8e.json"

# CFTC contract codes (matches CELL 44 CONFIG.BIAS_CFTC)
BIAS_CFTC = {
    "EUR/USD": "099741",
    "GBP/USD": "096742",
    "USD/JPY": "097741",
    "USD/CHF": "092741",
    "AUD/USD": "232741",
    "NZD/USD": "112741",
    "USD/CAD": "090741",
    "EUR/GBP": "099741",  # closest proxy
    "EUR/JPY": "099741",
    "GBP/JPY": "096742",
}
BIAS_PAIRS = list(BIAS_CFTC.keys())

# USD-base pairs need bias inversion (for these, USD strength = bullish for the pair)
USD_BASE = {"USD/JPY", "USD/CHF", "USD/CAD"}

MONTHS_AR = [
    "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
    "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"
]

# Hardcoded seasonality (matches CELL 44 CONFIG.BIAS_SEASON)
BIAS_SEASON = {
    "EUR/USD": [{"avg":0.35,"bull":55},{"avg":-0.55,"bull":40},{"avg":-0.40,"bull":42},{"avg":0.45,"bull":57},{"avg":-0.65,"bull":37},{"avg":-0.20,"bull":46},{"avg":0.72,"bull":60},{"avg":0.55,"bull":58},{"avg":-0.80,"bull":35},{"avg":0.28,"bull":53},{"avg":-0.48,"bull":40},{"avg":0.82,"bull":72}],
    "GBP/USD": [{"avg":0.15,"bull":50},{"avg":-0.38,"bull":43},{"avg":-0.42,"bull":40},{"avg":0.55,"bull":58},{"avg":-0.52,"bull":40},{"avg":0.10,"bull":49},{"avg":0.65,"bull":57},{"avg":0.38,"bull":55},{"avg":-0.88,"bull":33},{"avg":0.05,"bull":49},{"avg":-0.65,"bull":38},{"avg":0.48,"bull":63}],
    "USD/JPY": [{"avg":0.42,"bull":58},{"avg":0.25,"bull":54},{"avg":0.62,"bull":61},{"avg":-0.35,"bull":42},{"avg":-0.22,"bull":44},{"avg":0.28,"bull":54},{"avg":-0.15,"bull":47},{"avg":-0.45,"bull":40},{"avg":0.85,"bull":66},{"avg":0.52,"bull":61},{"avg":0.75,"bull":66},{"avg":-0.18,"bull":44}],
    "USD/CHF": [{"avg":-0.22,"bull":43},{"avg":0.35,"bull":55},{"avg":0.42,"bull":57},{"avg":-0.45,"bull":39},{"avg":0.48,"bull":59},{"avg":0.18,"bull":51},{"avg":-0.62,"bull":37},{"avg":-0.28,"bull":42},{"avg":0.72,"bull":64},{"avg":-0.12,"bull":47},{"avg":0.38,"bull":55},{"avg":-0.55,"bull":38}],
    "AUD/USD": [{"avg":0.52,"bull":61},{"avg":0.28,"bull":54},{"avg":-0.38,"bull":41},{"avg":0.18,"bull":52},{"avg":-0.75,"bull":34},{"avg":-0.52,"bull":39},{"avg":0.62,"bull":62},{"avg":0.82,"bull":66},{"avg":-0.32,"bull":43},{"avg":-0.22,"bull":44},{"avg":-0.62,"bull":37},{"avg":0.38,"bull":57}],
    "NZD/USD": [{"avg":0.58,"bull":61},{"avg":0.18,"bull":52},{"avg":-0.32,"bull":42},{"avg":0.08,"bull":50},{"avg":-0.82,"bull":31},{"avg":-0.42,"bull":39},{"avg":0.72,"bull":64},{"avg":0.92,"bull":68},{"avg":-0.22,"bull":44},{"avg":-0.12,"bull":47},{"avg":-0.52,"bull":39},{"avg":0.28,"bull":57}],
    "USD/CAD": [{"avg":-0.28,"bull":42},{"avg":0.08,"bull":50},{"avg":0.48,"bull":59},{"avg":-0.18,"bull":44},{"avg":0.38,"bull":57},{"avg":0.18,"bull":52},{"avg":-0.42,"bull":39},{"avg":-0.62,"bull":36},{"avg":0.28,"bull":54},{"avg":0.08,"bull":50},{"avg":0.58,"bull":61},{"avg":-0.12,"bull":47}],
    "EUR/GBP": [{"avg":0.08,"bull":49},{"avg":-0.08,"bull":48},{"avg":0.18,"bull":52},{"avg":-0.22,"bull":43},{"avg":-0.08,"bull":48},{"avg":-0.28,"bull":42},{"avg":0.08,"bull":51},{"avg":0.18,"bull":54},{"avg":0.08,"bull":50},{"avg":0.18,"bull":54},{"avg":0.08,"bull":51},{"avg":0.28,"bull":57}],
    "EUR/JPY": [{"avg":0.78,"bull":64},{"avg":-0.22,"bull":43},{"avg":0.28,"bull":54},{"avg":0.18,"bull":52},{"avg":-0.72,"bull":36},{"avg":0.08,"bull":49},{"avg":0.58,"bull":59},{"avg":0.18,"bull":52},{"avg":-1.15,"bull":29},{"avg":0.88,"bull":67},{"avg":0.18,"bull":51},{"avg":0.48,"bull":59}],
    "GBP/JPY": [{"avg":0.88,"bull":64},{"avg":-0.12,"bull":47},{"avg":0.38,"bull":56},{"avg":0.28,"bull":54},{"avg":-0.82,"bull":34},{"avg":0.18,"bull":51},{"avg":0.48,"bull":57},{"avg":0.08,"bull":50},{"avg":-1.35,"bull":27},{"avg":0.68,"bull":61},{"avg":0.28,"bull":54},{"avg":0.58,"bull":61}]
}


def fetch_cot(pair: str) -> dict:
    """Fetch CFTC COT for one pair. Returns dict with longs, shorts, percent, history."""
    code = BIAS_CFTC[pair]
    params = {
        "cftc_contract_market_code": code,
        "$limit": 8,
        "$order": "report_date_as_yyyy_mm_dd DESC"
    }
    r = requests.get(CFTC_BASE, params=params, timeout=20,
                     headers={"Accept": "application/json"})
    r.raise_for_status()
    rows = r.json()
    if not rows:
        raise RuntimeError("No CFTC rows returned")

    history = []
    for row in reversed(rows):  # oldest first for chart
        history.append({
            "date": (row.get("report_date_as_yyyy_mm_dd") or "").split("T")[0],
            "longs": int(row.get("noncomm_positions_long_all") or 0),
            "shorts": int(row.get("noncomm_positions_short_all") or 0),
        })

    latest = history[-1]
    total = latest["longs"] + latest["shorts"]
    long_pct = (latest["longs"] / total * 100) if total > 0 else 50.0

    return {
        "longs": latest["longs"],
        "shorts": latest["shorts"],
        "long_percent": round(long_pct, 1),
        "short_percent": round(100 - long_pct, 1),
        "report_date": latest["date"],
        "history": history,
    }


def cot_summary_text(pair: str, cot: dict) -> str:
    """Compose a one-line Arabic COT summary for the AI prompt."""
    direction = "long" if cot["long_percent"] > 50 else "short"
    return (
        f"نسبة الشراء {cot['long_percent']:.1f}% مقابل البيع {cot['short_percent']:.1f}% "
        f"بتاريخ التقرير {cot['report_date']} (المضاربون الكبار يميلون إلى {direction})."
    )


def season_summary_text(pair: str, month_idx: int) -> str:
    """Compose Arabic seasonality summary."""
    s = BIAS_SEASON[pair][month_idx]
    direction = "صاعد" if s["avg"] > 0 else "هابط"
    return (
        f"متوسط الأداء التاريخي {s['avg']:+.2f}% ({direction}) "
        f"مع نسبة شموع {direction}ة {s['bull']}%."
    )


def generate_insight(pair: str, cot: dict, month_idx: int) -> str:
    """Call Gemini for AI insight per pair."""
    from prompts import bias_insight_prompt
    from gemini_client import call_with_search

    cot_info = cot_summary_text(pair, cot)
    season_info = season_summary_text(pair, month_idx)
    month_ar = MONTHS_AR[month_idx]

    prompt = bias_insight_prompt(pair, cot_info, month_ar, season_info)
    return call_with_search(prompt, max_tokens=2048, temperature=0.6)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(CASA_TZ)
    month_idx = now.month - 1

    print(f"Casablanca time: {now.isoformat()}")
    print(f"Current month: {MONTHS_AR[month_idx]} (idx {month_idx})")
    print()

    out = {
        "updated_at": now.isoformat(),
        "tz": "Africa/Casablanca",
        "month_idx": month_idx,
        "month_ar": MONTHS_AR[month_idx],
        "pairs": {}
    }

    failed = []
    for pair in BIAS_PAIRS:
        print(f"→ {pair}")
        try:
            print("   • Fetching CFTC...")
            cot = fetch_cot(pair)
            print(f"     long {cot['long_percent']}% / short {cot['short_percent']}% "
                  f"(report {cot['report_date']})")

            print("   • AI insight...")
            insight = generate_insight(pair, cot, month_idx)
            print(f"     ✓ {len(insight)} chars")

            out["pairs"][pair] = {
                "cot": cot,
                "season": BIAS_SEASON[pair][month_idx],
                "insight": insight,
            }
            time.sleep(1.5)  # gentle pacing — avoid hitting RPM limits

        except Exception as e:
            print(f"   ✗ {pair} failed: {e}")
            failed.append((pair, str(e)))

    if failed and len(failed) == len(BIAS_PAIRS):
        # All failed — don't overwrite good data
        print(f"\n✗ ALL pairs failed. Refusing to overwrite cot.json.")
        sys.exit(1)

    COT_FILE.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print()
    print(f"✓ Saved {COT_FILE}")
    print(f"  Pairs successful: {len(out['pairs'])}/{len(BIAS_PAIRS)}")
    if failed:
        print(f"  Failures: {[p for p,_ in failed]}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"✗ FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
