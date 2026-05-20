"""CELL 44 Talaria Runner

Triggered by GitHub Actions cron 4×/day (Casablanca times):
  00:00 (Tokyo open)        → 23:00 UTC previous day
  07:00 (Pre-London)        → 06:00 UTC
  13:30 (Pre-NY)            → 12:30 UTC
  20:00 (NY close)          → 19:00 UTC

Workflow:
  1. yfinance fetch: 7 forex pairs + 7 macro instruments
  2. Compute currency strength (8 currencies) from forex pairs
  3. Compute trading opportunities for each pair (bias, score, entry quality, plan)
  4. Compose sentiment summary
  5. Write pipeline/data/talaria.json (then workflow copies to talaria.json root)

Safety: if more than 2 forex pairs fail, refuse to overwrite (preserve last good).

Output JSON structure:
  {
    "meta": {generated_at, generated_at_local, timezone, data_quality, stats},
    "forex": {EURUSD: {...}, GBPUSD: {...}, ...},
    "macro": {DXY: {...}, US10Y: {...}, ...},
    "strength": {USD: -0.167, EUR: 0.36, ...},
    "opportunities": [{pair, bias, score, ...}, ...],
    "sentiment": {top_pick_msg, strength_msg, dxy_msg, risk_msg, ...}
  }
"""

import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
import yfinance as yf
import pandas as pd
import numpy as np

CASA_TZ = timezone(timedelta(hours=1), name="Africa/Casablanca")
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = DATA_DIR / "talaria.json"

# ────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────
FOREX_PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X",
}

MACRO_INSTRUMENTS = {
    "DXY":   "DX-Y.NYB",
    "US10Y": "^TNX",
    "US02Y": "^IRX",
    "SPX":   "^GSPC",
    "GOLD":  "GC=F",
    "VIX":   "^VIX",
    "OIL":   "CL=F",
}

# Currencies for strength calculation
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

# ────────────────────────────────────────────
# Indicators (pure numpy, no extra dependencies)
# ────────────────────────────────────────────
def sma(series: pd.Series, period: int) -> float:
    if len(series) < period:
        return float('nan')
    return float(series.tail(period).mean())


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return float('nan')
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return float(tr.tail(period).mean())


def rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return float('nan')
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return float(100 - (100 / (1 + rs)).iloc[-1])


def trend_label(close: pd.Series, sma20: float, sma50: float) -> str:
    if np.isnan(sma20) or np.isnan(sma50):
        return "sideways"
    last = float(close.iloc[-1])
    if last > sma20 > sma50:
        return "bullish"
    if last < sma20 < sma50:
        return "bearish"
    return "sideways"


def range_position(price: float, today_high: float, today_low: float) -> float:
    """0 = at low, 100 = at high, scaled to 4 (so values are 0-4)"""
    rng = today_high - today_low
    if rng <= 0:
        return 2.0
    pos = (price - today_low) / rng
    return round(pos * 4, 1)


def extension_atr(price: float, sma20: float, atr_val: float) -> float:
    """How many ATRs price is above/below the 20 SMA"""
    if np.isnan(sma20) or np.isnan(atr_val) or atr_val == 0:
        return 0.0
    return round((price - sma20) / atr_val, 2)


# ────────────────────────────────────────────
# yfinance fetch with retry
# ────────────────────────────────────────────
def fetch_with_retry(ticker: str, period: str = "60d", interval: str = "1d", retries: int = 3) -> Optional[pd.DataFrame]:
    """Fetch OHLC with retry on transient failures."""
    last_err = None
    for attempt in range(retries):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
            if df is not None and len(df) >= 5:
                return df
            last_err = f"empty data (got {len(df) if df is not None else 'None'} rows)"
        except Exception as e:
            last_err = str(e)
        wait = 2 ** attempt
        print(f"  ⚠ {ticker} attempt {attempt+1} failed: {last_err}. Retry in {wait}s...")
        time.sleep(wait)
    print(f"  ✗ {ticker} failed after {retries} attempts: {last_err}")
    return None


# ────────────────────────────────────────────
# NaN cleaner (NaN is invalid JSON — browsers reject it)
# ────────────────────────────────────────────
def _clean_nan(obj):
    """Recursively replace NaN/Infinity with None for valid JSON output."""
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ────────────────────────────────────────────
# Per-instrument analysis
# ────────────────────────────────────────────
def analyze_instrument(symbol: str, df: pd.DataFrame) -> Dict:
    """Compute all indicators for one instrument."""
    close = df['Close']
    high = df['High']
    low = df['Low']

    if len(close) < 2:
        return {}

    current = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    change_pct = ((current - prev_close) / prev_close) * 100 if prev_close else 0.0

    # Weekly change (last 5 trading days)
    if len(close) >= 6:
        weekly_change_pct = ((current - float(close.iloc[-6])) / float(close.iloc[-6])) * 100
    else:
        weekly_change_pct = 0.0

    sma_20 = sma(close, 20)
    sma_50 = sma(close, 50)
    atr_14 = atr(high, low, close, 14)
    rsi_14 = rsi(close, 14)

    # 52-week range
    if len(close) >= 252:
        high_52w = float(close.tail(252).max())
        low_52w = float(close.tail(252).min())
    else:
        high_52w = float(close.max())
        low_52w = float(close.min())

    # Previous day high/low
    pdh = float(high.iloc[-2]) if len(high) >= 2 else current
    pdl = float(low.iloc[-2]) if len(low) >= 2 else current

    # Today's range
    today_high = float(high.iloc[-1])
    today_low = float(low.iloc[-1])
    daily_range = today_high - today_low

    rp = range_position(current, today_high, today_low)
    ext = extension_atr(current, sma_20, atr_14)
    trend = trend_label(close, sma_20, sma_50)

    return {
        "current": round(current, 5),
        "change_pct": round(change_pct, 3),
        "weekly_change_pct": round(weekly_change_pct, 3),
        "sma_20": round(sma_20, 5) if not np.isnan(sma_20) else None,
        "sma_50": round(sma_50, 5) if not np.isnan(sma_50) else None,
        "atr_14": round(atr_14, 5) if not np.isnan(atr_14) else None,
        "daily_range": round(daily_range, 5),
        "trend": trend,
        "high_52w": round(high_52w, 5),
        "low_52w": round(low_52w, 5),
        "pdh": round(pdh, 5),
        "pdl": round(pdl, 5),
        "today_high": round(today_high, 5),
        "today_low": round(today_low, 5),
        "range_position": rp,
        "extension_atr": ext,
        "rsi_14": round(rsi_14, 1) if not np.isnan(rsi_14) else None,
    }


# ────────────────────────────────────────────
# Currency strength calculation
# Uses change_pct of forex pairs to derive 8 currency strengths
# ────────────────────────────────────────────
def compute_strength(forex: Dict[str, Dict]) -> Dict[str, float]:
    """Each currency's strength = avg of its appearances in pair changes,
    inverted when it's the quote currency."""
    contributions: Dict[str, List[float]] = {c: [] for c in CURRENCIES}

    for pair_key, data in forex.items():
        if not data or 'change_pct' not in data:
            continue
        change = data['change_pct']
        # parse base/quote from "EURUSD" → ("EUR", "USD")
        base = pair_key[:3]
        quote = pair_key[3:]
        if base in contributions:
            contributions[base].append(change)
        if quote in contributions:
            contributions[quote].append(-change)

    strength = {}
    for ccy, vals in contributions.items():
        strength[ccy] = round(sum(vals) / len(vals), 3) if vals else 0.0

    return strength


# ────────────────────────────────────────────
# Opportunity generation
# Each pair gets a bias, score, and entry plan
# ────────────────────────────────────────────
def pip_size(pair: str) -> float:
    """1 pip = 0.0001 for most, 0.01 for JPY pairs"""
    return 0.01 if "JPY" in pair else 0.0001


def compute_opportunity(pair: str, data: Dict, strength: Dict[str, float]) -> Dict:
    """Build an opportunity card for a pair."""
    base = pair[:3]
    quote = pair[3:]

    base_str = strength.get(base, 0.0)
    quote_str = strength.get(quote, 0.0)
    divergence = base_str - quote_str  # >0 → long bias, <0 → short bias

    bias = "long" if divergence >= 0 else "short"
    bias_ar = "شراء" if bias == "long" else "بيع"

    # Score = magnitude of divergence × volatility weighting
    atr_val = data.get('atr_14') or 0.0
    current = data.get('current') or 0.0
    volatility_pct = (atr_val / current * 100) if current > 0 else 0.0

    trend = data.get('trend', 'sideways')
    rsi_14 = data.get('rsi_14') or 50.0
    range_pos = data.get('range_position') or 2.0
    ext = data.get('extension_atr') or 0.0

    # Score base = |divergence| × 30 (so divergence of 1.0 → 30 points)
    score = abs(divergence) * 30

    # Trend alignment bonus
    if (bias == "long" and trend == "bullish") or (bias == "short" and trend == "bearish"):
        score += 10
    elif (bias == "long" and trend == "bearish") or (bias == "short" and trend == "bullish"):
        score -= 5

    # RSI alignment
    if bias == "long" and rsi_14 > 55:
        score += 3
    elif bias == "short" and rsi_14 < 45:
        score += 3

    score = max(0, round(score, 2))

    # Entry quality assessment
    entry_quality, entry_quality_ar, entry_reason_ar = assess_entry_quality(
        bias, range_pos, ext, rsi_14
    )

    # Trade plan: SL = 15 pips (Lamine's FTMO style), TP1 = 1.5R, TP2 = 2.5R
    pip = pip_size(pair)
    risk_pips = 15
    risk_amount = risk_pips * pip

    if bias == "long":
        suggested_entry = current
        suggested_sl = current - risk_amount
        suggested_tp1 = current + risk_amount * 1.5
        suggested_tp2 = current + risk_amount * 2.5
    else:
        suggested_entry = current
        suggested_sl = current + risk_amount
        suggested_tp1 = current - risk_amount * 1.5
        suggested_tp2 = current - risk_amount * 2.5

    return {
        "pair": pair,
        "bias": bias,
        "bias_ar": bias_ar,
        "score": score,
        "divergence": round(divergence, 3),
        "trend": trend,
        "base_strength": round(base_str, 3),
        "quote_strength": round(quote_str, 3),
        "volatility_pct": round(volatility_pct, 3),
        "current_price": current,
        "daily_change": data.get('change_pct'),
        "atr": atr_val,
        "sma_20": data.get('sma_20'),
        "sma_50": data.get('sma_50'),
        "rsi_14": rsi_14,
        "pdh": data.get('pdh'),
        "pdl": data.get('pdl'),
        "range_position": range_pos,
        "extension_atr": ext,
        "entry_quality": entry_quality,
        "entry_quality_ar": entry_quality_ar,
        "entry_reason_ar": entry_reason_ar,
        "suggested_entry": round(suggested_entry, 5),
        "suggested_sl": round(suggested_sl, 5),
        "suggested_tp1": round(suggested_tp1, 5),
        "suggested_tp2": round(suggested_tp2, 5),
        "risk_pips": risk_pips,
        "tp1_pips": int(risk_pips * 1.5),
        "tp2_pips": int(risk_pips * 2.5),
    }


def assess_entry_quality(bias: str, range_pos: float, ext: float, rsi_14: float):
    """
    Returns (entry_quality, entry_quality_ar, entry_reason_ar).

    entry_quality:
      - "fresh"     → conditions favor immediate entry
      - "pullback"  → wait for price to return to mean
      - "extended"  → too far from mean, risky
      - "unknown"   → couldn't assess
    """
    reasons = []

    # Extension check (most important)
    abs_ext = abs(ext)
    if abs_ext > 2.0:
        reasons.append("السعر ممتد بعيداً عن المتوسط — الدخول مخاطرة عالية")
        return "extended", "ممتد — مخاطرة", " · ".join(reasons)

    if abs_ext > 1.3:
        if (bias == "long" and ext > 0) or (bias == "short" and ext < 0):
            reasons.append("السعر منخفض، انتظر ارتداد للمتوسط" if bias == "long"
                           else "السعر مرتفع، انتظر ارتداد للمتوسط")
            return "pullback", "انتظر تصحيح", " · ".join(reasons)

    # Range position check
    if bias == "long" and range_pos < 1.0:
        reasons.append("السعر قرب قاع اليوم")
    elif bias == "short" and range_pos > 3.0:
        reasons.append("السعر قرب قمة اليوم")
    elif bias == "long" and range_pos > 3.5:
        reasons.append("السعر قرب قمة اليوم — انتظر تصحيح")
        return "pullback", "انتظر تصحيح", " · ".join(reasons)
    elif bias == "short" and range_pos < 0.5:
        reasons.append("السعر قرب قاع اليوم — انتظر ارتداد")
        return "pullback", "انتظر تصحيح", " · ".join(reasons)

    # RSI check
    if bias == "long" and rsi_14 > 70:
        reasons.append("RSI مرتفع جداً")
        return "extended", "ممتد — مخاطرة", " · ".join(reasons)
    if bias == "short" and rsi_14 < 30:
        reasons.append("RSI منخفض جداً")
        return "extended", "ممتد — مخاطرة", " · ".join(reasons)

    if not reasons:
        reasons.append("الظروف مناسبة للدخول المباشر")

    return "fresh", "دخول جيد", " · ".join(reasons)


# ────────────────────────────────────────────
# Sentiment composer
# ────────────────────────────────────────────
def compose_sentiment(forex: Dict, macro: Dict, strength: Dict, opportunities: List, calendar_events_count: int = 0) -> Dict:
    """Build the Arabic sentiment messages."""
    sent = {}

    # DXY message
    dxy = macro.get('DXY', {})
    if dxy and 'change_pct' in dxy:
        dxy_chg = dxy['change_pct']
        if dxy_chg < -0.3:
            sent['dxy_msg'] = f"الدولار ضعيف اليوم ({dxy_chg:+.3f}%) — فرصة للعملات المقابلة"
            sent['dxy_color'] = "bearish"
        elif dxy_chg > 0.3:
            sent['dxy_msg'] = f"الدولار قوي اليوم ({dxy_chg:+.3f}%) — ضغط على العملات المقابلة"
            sent['dxy_color'] = "bullish"
        else:
            sent['dxy_msg'] = f"الدولار مستقر ({dxy_chg:+.3f}%)"
            sent['dxy_color'] = "neutral"

    # Yields message
    us10y = macro.get('US10Y', {})
    if us10y and 'current' in us10y:
        sent['yields_msg'] = f"عائد السندات 10 سنوات عند {us10y['current']:.2f}%"

    # Risk mode (based on VIX)
    vix = macro.get('VIX', {})
    if vix and 'current' in vix:
        vix_val = vix['current']
        if vix_val < 20:
            sent['risk_msg'] = f"السوق هادئ (VIX={vix_val:.1f}) — بيئة مناسبة للتداول"
            sent['risk_mode'] = "risk_on"
        elif vix_val < 25:
            sent['risk_msg'] = f"حذر معتدل (VIX={vix_val:.1f}) — تذبذب متوسط"
            sent['risk_mode'] = "neutral"
        else:
            sent['risk_msg'] = f"خوف في السوق (VIX={vix_val:.1f}) — قلّل الحجم"
            sent['risk_mode'] = "risk_off"

    # Top pick
    if opportunities:
        top = opportunities[0]
        sent['top_pick_msg'] = f"أفضل فرصة اليوم: {top['bias_ar']} {top['pair']} (قوة: {top['score']:.2f})"

    # Strongest / weakest
    if strength:
        sorted_str = sorted(strength.items(), key=lambda x: x[1], reverse=True)
        strongest = sorted_str[0]
        weakest = sorted_str[-1]
        sent['strongest_currency'] = strongest[0]
        sent['weakest_currency'] = weakest[0]
        sent['strength_msg'] = f"أقوى عملة: {strongest[0]} ({strongest[1]:+.3f}%) | أضعف عملة: {weakest[0]} ({weakest[1]:+.3f}%)"

    sent['high_impact_events_count'] = calendar_events_count

    return sent


# ────────────────────────────────────────────
# Main
# ────────────────────────────────────────────
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc)
    now_casa = datetime.now(CASA_TZ)
    print(f"Casablanca time: {now_casa.isoformat()}")
    print(f"UTC: {now_utc.isoformat()}")
    print()

    # Fetch forex pairs
    print("─── Fetching forex pairs ───")
    forex_data = {}
    forex_failed = []
    for pair_key, ticker in FOREX_PAIRS.items():
        print(f"→ {pair_key} ({ticker})")
        df = fetch_with_retry(ticker)
        if df is not None:
            forex_data[pair_key] = analyze_instrument(pair_key, df)
            print(f"  ✓ current={forex_data[pair_key]['current']}, change={forex_data[pair_key]['change_pct']:+.3f}%")
        else:
            forex_failed.append(pair_key)

    print()

    # Fetch macro instruments
    print("─── Fetching macro instruments ───")
    macro_data = {}
    macro_failed = []
    for key, ticker in MACRO_INSTRUMENTS.items():
        print(f"→ {key} ({ticker})")
        df = fetch_with_retry(ticker)
        if df is not None:
            macro_data[key] = analyze_instrument(key, df)
            print(f"  ✓ current={macro_data[key]['current']}")
        else:
            macro_failed.append(key)

    print()

    # Safety guard: refuse to overwrite if too much data is missing
    forex_success = len(forex_data)
    macro_success = len(macro_data)

    if forex_success < 5:
        print(f"✗ FAILED: Only {forex_success}/7 forex pairs succeeded — refusing to overwrite talaria.json")
        print(f"  Failed pairs: {forex_failed}")
        sys.exit(1)

    # Compute strength
    print("─── Computing currency strength ───")
    strength = compute_strength(forex_data)
    for ccy, val in strength.items():
        print(f"  {ccy}: {val:+.3f}")

    print()

    # Compute opportunities
    print("─── Computing opportunities ───")
    opportunities = []
    for pair_key, data in forex_data.items():
        opp = compute_opportunity(pair_key, data, strength)
        opportunities.append(opp)

    # Sort by score (desc)
    opportunities.sort(key=lambda o: o['score'], reverse=True)
    for opp in opportunities:
        print(f"  {opp['pair']}: {opp['bias_ar']} score={opp['score']} quality={opp['entry_quality_ar']}")

    print()

    # Compose sentiment
    print("─── Composing sentiment ───")
    sentiment = compose_sentiment(forex_data, macro_data, strength, opportunities)
    for k, v in sentiment.items():
        print(f"  {k}: {v}")

    print()

    # Determine data quality
    if forex_success == 7 and macro_success == 7:
        data_quality = "complete"
    elif forex_success >= 6 and macro_success >= 5:
        data_quality = "complete"
    else:
        data_quality = "partial"

    # Compose final JSON
    output = {
        "meta": {
            "generated_at": now_utc.isoformat(),
            "generated_at_local": now_casa.isoformat(),
            "timezone": "Africa/Casablanca",
            "data_quality": data_quality,
            "stats": {
                "forex_success": forex_success,
                "forex_total": len(FOREX_PAIRS),
                "forex_failed": forex_failed,
                "macro_success": macro_success,
                "macro_total": len(MACRO_INSTRUMENTS),
                "macro_failed": macro_failed,
            }
        },
        "forex": forex_data,
        "macro": macro_data,
        "strength": strength,
        "opportunities": opportunities,
        "sentiment": sentiment,
    }

    # Clean NaN/Infinity values before JSON serialization
    # (NaN is valid Python but INVALID JSON — browsers reject "NaN" tokens)
    cleaned_output = _clean_nan(output)

    # Atomic write: write to .tmp first, then atomic rename via os.replace.
    # This guarantees talaria.json is either fully written or unchanged —
    # never partially written, even if the script is killed mid-write.
    tmp_file = OUTPUT_FILE.with_suffix('.json.tmp')
    tmp_file.write_text(
        json.dumps(cleaned_output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    os.replace(tmp_file, OUTPUT_FILE)

    print(f"✓ Saved {OUTPUT_FILE} (atomic)")
    print(f"  Data quality: {data_quality}")
    print(f"  Forex: {forex_success}/{len(FOREX_PAIRS)} | Macro: {macro_success}/{len(MACRO_INSTRUMENTS)}")
    print(f"  Top pick: {opportunities[0]['bias_ar']} {opportunities[0]['pair']} (score {opportunities[0]['score']})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"✗ FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
