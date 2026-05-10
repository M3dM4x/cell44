"""CELL 44 Daily Analyses Runner

Triggered by GitHub Actions cron 4x/day:
  06:00 Casablanca (05:00 UTC) → full
  10:00 Casablanca (09:00 UTC) → quick
  13:00 Casablanca (12:00 UTC) → full
  16:00 Casablanca (15:00 UTC) → quick

For workflow_dispatch (manual): pass --type quick or --type full.

Output: pipeline/data/analyses.json (top-level analyses.json copied by workflow).
Format:
  {
    "updated_at": ISO timestamp (Casablanca tz),
    "tz": "Africa/Casablanca",
    "next_slot": {"type": "...", "scheduled_for": "..."},
    "slots": [
      {"id", "ts", "type": "full"|"quick", "macro?", "news?", "final?", "quick?"}
    ]  // newest first, pruned to 7 days
  }
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Casablanca = UTC+1 year-round (Morocco abolished DST in 2018)
CASA_TZ = timezone(timedelta(hours=1), name="Africa/Casablanca")

DATA_DIR = Path(__file__).parent / "data"
ANALYSES_FILE = DATA_DIR / "analyses.json"
RETENTION_DAYS = 7

# Map UTC cron → Casablanca slot
SLOT_MAP = {
    "0 5 * * *": ("06:00", "full"),
    "0 9 * * *": ("10:00", "quick"),
    "0 12 * * *": ("13:00", "full"),
    "0 15 * * *": ("16:00", "quick"),
}

# Schedule order for next-slot calculation
DAILY_SCHEDULE = [
    ("06:00", "full"),
    ("10:00", "quick"),
    ("13:00", "full"),
    ("16:00", "quick"),
]


def now_casa() -> datetime:
    return datetime.now(CASA_TZ)


def determine_type() -> str:
    """Determine analysis type from CRON_SCHEDULE env var or --type arg."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["full", "quick"], default=None)
    args = parser.parse_args()

    if args.type:
        return args.type

    cron = os.environ.get("CRON_SCHEDULE", "").strip()
    if cron in SLOT_MAP:
        _, slot_type = SLOT_MAP[cron]
        return slot_type

    # Fallback: pick by current time (workflow_dispatch with no arg)
    h = now_casa().hour
    if h < 8:
        return "full"  # treat early morning as full
    if h < 12:
        return "quick"
    if h < 15:
        return "full"
    return "quick"


def compute_next_slot(now: datetime) -> dict:
    """Find the next scheduled slot after `now` (Casablanca time)."""
    today = now.date()
    for hhmm, slot_type in DAILY_SCHEDULE:
        h, m = map(int, hhmm.split(":"))
        candidate = datetime.combine(today, datetime.min.time(), CASA_TZ).replace(hour=h, minute=m)
        if candidate > now:
            return {"type": slot_type, "scheduled_for": candidate.isoformat()}

    # No more slots today — first slot tomorrow
    tomorrow = today + timedelta(days=1)
    h, m = map(int, DAILY_SCHEDULE[0][0].split(":"))
    next_dt = datetime.combine(tomorrow, datetime.min.time(), CASA_TZ).replace(hour=h, minute=m)
    return {"type": DAILY_SCHEDULE[0][1], "scheduled_for": next_dt.isoformat()}


def load_existing() -> dict:
    if ANALYSES_FILE.exists():
        try:
            return json.loads(ANALYSES_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠ Failed to read existing analyses.json: {e} — starting fresh")
    return {
        "updated_at": None,
        "tz": "Africa/Casablanca",
        "next_slot": None,
        "slots": []
    }


def prune_old(slots: list, now: datetime) -> list:
    """Keep only slots within RETENTION_DAYS."""
    cutoff = now - timedelta(days=RETENTION_DAYS)
    kept = []
    for s in slots:
        try:
            ts = datetime.fromisoformat(s["ts"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=CASA_TZ)
            if ts >= cutoff:
                kept.append(s)
        except Exception:
            pass  # drop malformed entries
    return kept


def run_full(now: datetime) -> dict:
    """Full analysis: MACRO + NEWS + FINAL (3 Gemini calls)."""
    from prompts import SYSTEM_INSTRUCTION, P_MACRO, P_NEWS, P_FINAL
    from gemini_client import call_with_search, call_text

    print("→ MACRO (with Google Search)...")
    macro = call_with_search(P_MACRO, system_instruction=SYSTEM_INSTRUCTION,
                             max_tokens=16000, temperature=0.5)
    print(f"   ✓ {len(macro)} chars")

    print("→ NEWS (with Google Search)...")
    news = call_with_search(P_NEWS, system_instruction=SYSTEM_INSTRUCTION,
                            max_tokens=16000, temperature=0.5)
    print(f"   ✓ {len(news)} chars")

    print("→ FINAL (synthesis, no search)...")
    final_prompt = (
        f"{P_FINAL}\n\n"
        f"---\n"
        f"## تحليل الماكرو\n{macro}\n\n"
        f"---\n"
        f"## تحليل الأخبار\n{news}"
    )
    final = call_text(final_prompt, max_tokens=4096, temperature=0.4)
    print(f"   ✓ {len(final)} chars")

    return {
        "id": now.isoformat(),
        "ts": now.isoformat(),
        "type": "full",
        "macro": macro,
        "news": news,
        "final": final,
    }


def run_quick(now: datetime) -> dict:
    """Quick analysis: only QUICK prompt (1 Gemini call)."""
    from prompts import SYSTEM_INSTRUCTION, P_QUICK
    from gemini_client import call_with_search

    print("→ QUICK (with Google Search)...")
    quick = call_with_search(P_QUICK, system_instruction=SYSTEM_INSTRUCTION,
                             max_tokens=4096, temperature=0.5)
    print(f"   ✓ {len(quick)} chars")

    return {
        "id": now.isoformat(),
        "ts": now.isoformat(),
        "type": "quick",
        "quick": quick,
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    slot_type = determine_type()
    now = now_casa()
    print(f"Casablanca time: {now.isoformat()}")
    print(f"Slot type: {slot_type}")
    print()

    # Run analysis
    if slot_type == "full":
        slot_data = run_full(now)
    else:
        slot_data = run_quick(now)

    # Load, prepend, prune, save
    state = load_existing()
    slots = state.get("slots", [])
    slots.insert(0, slot_data)  # newest first
    slots = prune_old(slots, now)

    state["slots"] = slots
    state["updated_at"] = now.isoformat()
    state["tz"] = "Africa/Casablanca"
    state["next_slot"] = compute_next_slot(now)

    ANALYSES_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print()
    print(f"✓ Saved {ANALYSES_FILE}")
    print(f"  Total slots: {len(slots)}")
    print(f"  Newest: {slots[0]['ts']} ({slots[0]['type']})")
    print(f"  Next scheduled: {state['next_slot']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"✗ FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
