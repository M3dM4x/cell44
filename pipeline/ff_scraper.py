"""
Custom Forex Factory Scraper - v2 with Cloudflare bypass
=========================================================
يستخدم curl_cffi بدلاً من requests لتجاوز حماية Cloudflare.
curl_cffi تحاكي متصفح Chrome حقيقي على مستوى TLS fingerprint.

Run:
    python ff_scraper.py
"""

import os
import re
import time
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

from curl_cffi import requests as curl_requests

from config import (
    START_DATE,
    END_DATE,
    MAJOR_CURRENCIES,
    IMPACT_LEVELS,
    DATA_DIR,
    CALENDAR_FILE,
)


MONTH_ABBR = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
    7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec",
}

try:
    from zoneinfo import ZoneInfo
    FF_TZ = ZoneInfo("Etc/GMT+5")
    LOCAL_TZ = ZoneInfo("Africa/Casablanca")
    USE_ZONEINFO = True
except ImportError:
    FF_TZ = None
    LOCAL_TZ = None
    USE_ZONEINFO = False

FALLBACK_OFFSET_HOURS = 6


def parse_ff_time(time_str: str):
    """Parse Forex Factory time string into (hour, minute). Returns None for unparseable."""
    if not time_str:
        return None
    time_str = time_str.strip().lower()
    if time_str in ("all day", "tentative", "", "n/a"):
        return None
    try:
        is_pm = time_str.endswith("pm")
        is_am = time_str.endswith("am")
        time_part = time_str[:-2].strip() if (is_pm or is_am) else time_str

        if ":" in time_part:
            h, m = time_part.split(":")
            hour, minute = int(h), int(m)
        else:
            hour, minute = int(time_part), 0

        if is_pm and hour != 12:
            hour += 12
        elif is_am and hour == 12:
            hour = 0
        return hour, minute
    except (ValueError, IndexError):
        return None


def convert_ff_time_to_morocco(time_str: str, event_date: datetime = None) -> str:
    """Convert Forex Factory time (US Eastern, 12h am/pm) to Morocco 24h format.

    Uses zoneinfo when available for accurate DST-aware conversion on the
    specific event date. This handles:
      - US DST transitions (EDT UTC-4 / EST UTC-5)
      - Morocco DST transitions (WEST UTC+1 / Ramadan UTC+0)
    Falls back to a fixed offset on older Python versions.
    """
    if not time_str or time_str.lower().strip() in ("all day", "tentative", "", "n/a"):
        return time_str or ""

    parsed = parse_ff_time(time_str)
    if parsed is None:
        return time_str
    hour, minute = parsed

    if USE_ZONEINFO and event_date is not None:
        try:
            dt_eastern = datetime(
                event_date.year, event_date.month, event_date.day,
                hour, minute, tzinfo=FF_TZ
            )
            dt_morocco = dt_eastern.astimezone(LOCAL_TZ)
            return dt_morocco.strftime("%H:%M")
        except Exception:
            pass

    hour = (hour + FALLBACK_OFFSET_HOURS) % 24
    return f"{hour:02d}:{minute:02d}"


def build_week_url(monday: datetime) -> str:
    """Build Forex Factory URL for a given Monday's week."""
    month = MONTH_ABBR[monday.month]
    return f"https://www.forexfactory.com/calendar?week={month}{monday.day}.{monday.year}"


def get_mondays(start: str, end: str):
    """Generate all Mondays between start and end dates."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    days_since_monday = start_dt.weekday()
    current = start_dt - timedelta(days=days_since_monday)

    while current <= end_dt:
        yield current
        current += timedelta(days=7)


def parse_impact(td_impact) -> str:
    """Extract impact level from the impact cell.
    
    Forex Factory uses color classes:
      icon--ff-impact-red = High
      icon--ff-impact-ora = Medium (orange)
      icon--ff-impact-yel = Low (yellow)
      icon--ff-impact-gra = Holiday/Bank (gray)
    """
    if not td_impact:
        return ""
    span = td_impact.find("span")
    if not span:
        return ""
    
    # Check span class names for color indicator
    span_classes = " ".join(span.get("class", []))
    
    if "impact-red" in span_classes:
        return "High"
    if "impact-ora" in span_classes:
        return "Medium"
    if "impact-yel" in span_classes:
        return "Low"
    if "impact-gra" in span_classes:
        return "Holiday"
    
    # Fallback: check title attribute (old format)
    title = span.get("title", "")
    if "High" in title:
        return "High"
    if "Medium" in title:
        return "Medium"
    if "Low" in title:
        return "Low"
    if "Non-Economic" in title:
        return "Holiday"
    
    return ""


def clean_value(text: str) -> str:
    if not text or text.strip() == "":
        return ""
    return text.strip()


def create_session():
    """Create a curl_cffi session that impersonates Chrome."""
    return curl_requests.Session(impersonate="chrome131")


def scrape_week(monday: datetime, session) -> list:
    """Scrape one week of events from Forex Factory using curl_cffi.
    
    Forex Factory shows the date only on the first event of each day.
    We need to carry it over to subsequent events.
    """
    url = build_week_url(monday)
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"    ERROR fetching {url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    calendar_table = soup.find("table", class_="calendar__table")
    if not calendar_table:
        return []

    events = []
    current_date = None
    sunday = monday + timedelta(days=6)

    rows = calendar_table.find_all("tr", class_=re.compile(r"calendar__row|calendar_row"))

    for row in rows:
        # Check if this row starts a new day (has a non-empty date cell)
        date_cell = row.find("td", class_=re.compile(r"calendar__date"))
        if date_cell:
            date_text = date_cell.get_text(strip=True)
            if date_text:
                # Format is like "MonMar4" or "Mar4" - extract the day number
                day_match = re.search(r"(\d+)$", date_text)
                if day_match:
                    day_num = int(day_match.group(1))
                    
                    # Try to find which month this day belongs to (this week's range)
                    # The week spans monday -> sunday, possibly crossing a month boundary
                    found_date = None
                    for offset in range(7):
                        candidate = monday + timedelta(days=offset)
                        if candidate.day == day_num:
                            found_date = candidate
                            break
                    
                    if found_date:
                        current_date = found_date

        # Check if this row is an event row
        time_cell = row.find("td", class_=re.compile(r"calendar__time"))
        currency_cell = row.find("td", class_=re.compile(r"calendar__currency"))
        impact_cell = row.find("td", class_=re.compile(r"calendar__impact"))
        event_cell = row.find("td", class_=re.compile(r"calendar__event"))

        if not (currency_cell and event_cell):
            continue

        currency = currency_cell.get_text(strip=True)
        if not currency or currency not in MAJOR_CURRENCIES:
            continue

        event_title = event_cell.get_text(strip=True)
        if not event_title:
            continue

        # Skip if we don't have a date yet (shouldn't happen, but safety)
        if current_date is None:
            continue

        impact = parse_impact(impact_cell)
        raw_time = time_cell.get_text(strip=True) if time_cell else ""
        time_text = convert_ff_time_to_morocco(raw_time, current_date)

        actual_cell = row.find("td", class_=re.compile(r"calendar__actual"))
        forecast_cell = row.find("td", class_=re.compile(r"calendar__forecast"))
        previous_cell = row.find("td", class_=re.compile(r"calendar__previous"))

        events.append({
            "Date": current_date.strftime("%Y-%m-%d"),
            "Time": time_text,
            "Currency": currency,
            "Event": event_title,
            "Impact": impact,
            "Actual": clean_value(actual_cell.get_text(strip=True)) if actual_cell else "",
            "Forecast": clean_value(forecast_cell.get_text(strip=True)) if forecast_cell else "",
            "Previous": clean_value(previous_cell.get_text(strip=True)) if previous_cell else "",
        })

    return events


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Scraping Forex Factory: {START_DATE} -> {END_DATE}")
    print(f"Currencies: {MAJOR_CURRENCIES}")
    print(f"Impact filter: {IMPACT_LEVELS}")
    print(f"Using curl_cffi with Chrome120 impersonation\n")

    mondays = list(get_mondays(START_DATE, END_DATE))
    print(f"Total weeks to scrape: {len(mondays)}")
    print(f"Estimated time: {len(mondays) * 3 / 60:.1f} minutes\n")

    session = create_session()

    # Connection test first
    print("Testing connection to Forex Factory...")
    try:
        test_resp = session.get("https://www.forexfactory.com/calendar", timeout=30)
        if test_resp.status_code == 200:
            print(f"✓ Connection successful (status {test_resp.status_code})\n")
        else:
            print(f"✗ Got status {test_resp.status_code}. Cloudflare may be blocking us.")
            print("  Try again in a few minutes, or use a VPN.")
            return
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return

    all_events = []
    consecutive_errors = 0

    for i, monday in enumerate(mondays, 1):
        print(f"[{i}/{len(mondays)}] Week of {monday.strftime('%Y-%m-%d')}...", end=" ")
        events = scrape_week(monday, session)
        print(f"{len(events)} events")

        if len(events) == 0:
            consecutive_errors += 1
            if consecutive_errors >= 5:
                print("\n⚠️  5 consecutive empty weeks. Taking a 30s break...")
                time.sleep(30)
                consecutive_errors = 0
                session = create_session()
        else:
            consecutive_errors = 0

        all_events.extend(events)

        if i < len(mondays):
            time.sleep(2)

    if not all_events:
        raise RuntimeError("No events scraped! Check your internet or try a VPN.")

    df = pd.DataFrame(all_events)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    if IMPACT_LEVELS:
        df = df[df["Impact"].isin(IMPACT_LEVELS)]

    df = df.drop_duplicates().reset_index(drop=True)

    for col in ["Actual", "Forecast", "Previous"]:
        df[f"{col}_num"] = pd.to_numeric(
            df[col].astype(str).str.replace(r"[^\d.\-]", "", regex=True),
            errors="coerce",
        )

    df["Surprise"] = (df["Actual_num"] - df["Forecast_num"]) / df["Forecast_num"].abs().replace(0, pd.NA)

    df.to_csv(CALENDAR_FILE, index=False)

    print(f"\n✓ Saved {len(df)} events to {CALENDAR_FILE}")
    print(f"\nEvents per currency:")
    print(df["Currency"].value_counts())
    print(f"\nSample:")
    print(df[["Date", "Currency", "Event", "Impact"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
