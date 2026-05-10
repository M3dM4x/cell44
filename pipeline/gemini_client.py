"""Gemini API client — mirrors CELL 44 retry logic.
Uses generateContent (non-streaming) since we don't need real-time UI updates server-side.
"""

import os
import time
import json
import requests
from typing import Optional

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
MODEL = "gemini-2.5-flash-lite"
RETRIES = 3
TIMEOUT = 120  # seconds — full analyses can take a while with grounding


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY env var missing")
    return key


def call_with_search(prompt: str, system_instruction: Optional[str] = None,
                     max_tokens: int = 16000, temperature: float = 0.5) -> str:
    """Call Gemini with Google Search grounding enabled. Returns full text response.
    Used for MACRO, NEWS, QUICK (anything needing live data).
    """
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "tools": [{"googleSearch": {}}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    return _call(body)


def call_text(prompt: str, max_tokens: int = 2048, temperature: float = 0.4) -> str:
    """Call Gemini without search grounding. Used for FINAL (synthesis from given context)."""
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }
    return _call(body)


def _call(body: dict) -> str:
    """Core call with retry on transient errors (503/429/overload)."""
    url = f"{API_BASE}/{MODEL}:generateContent?key={_api_key()}"
    headers = {"Content-Type": "application/json"}

    last_err = None
    for attempt in range(RETRIES):
        try:
            r = requests.post(url, json=body, headers=headers, timeout=TIMEOUT)
            if r.ok:
                data = r.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise RuntimeError(f"No candidates in response: {json.dumps(data)[:300]}")
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts).strip()
                if not text:
                    raise RuntimeError(f"Empty text in response: {json.dumps(data)[:300]}")
                return text

            # Non-OK response
            try:
                err_body = r.json()
                msg = err_body.get("error", {}).get("message") or f"HTTP {r.status_code}"
            except Exception:
                msg = f"HTTP {r.status_code}: {r.text[:200]}"

            transient = r.status_code in (503, 429) or any(
                kw in msg.lower() for kw in ("overload", "unavailable", "high demand", "temporary")
            )
            if not transient or attempt == RETRIES - 1:
                raise RuntimeError(msg)

            wait = (2 ** attempt) * 2.5
            print(f"  ⚠ Transient error (attempt {attempt+1}/{RETRIES}): {msg}. Retry in {wait}s...")
            time.sleep(wait)
            last_err = msg

        except requests.RequestException as e:
            if attempt == RETRIES - 1:
                raise RuntimeError(f"Network error after {RETRIES} attempts: {e}")
            wait = (2 ** attempt) * 2.5
            print(f"  ⚠ Network error (attempt {attempt+1}/{RETRIES}): {e}. Retry in {wait}s...")
            time.sleep(wait)
            last_err = str(e)

    raise RuntimeError(f"Failed after {RETRIES} attempts. Last error: {last_err}")
