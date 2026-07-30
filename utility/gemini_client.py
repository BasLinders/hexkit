"""
gemini_client.py
Thin wrapper for the optional "AI conclusion" step in the Automation wizard —
sends the already-computed result payload to Gemini and asks for a written,
plain-language interpretation. Mirrors the return-a-dict-with-error style
used in airtable_client.py/bq_client.py rather than raising, since callers
are Streamlit pages that just want to show st.error().
"""
from __future__ import annotations

import json
from typing import Any, Optional

import streamlit as st

DEFAULT_MODEL = "gemini-2.5-flash"

_PROMPT_INSTRUCTIONS = """\
You are a conversion-rate-optimization analyst reviewing the results of an
A/B test. Below is a JSON object with the computed statistics (frequentist,
Bayesian, and/or continuous-metric analysis, and an optional pre-test MDE
projection) for a control ("Control") vs. a variation ("Variation").

Write a short, plain-language conclusion (3-6 sentences) a stakeholder
without a statistics background could act on. Cover:
- Whether the result is statistically significant / conclusive, and how confident to be.
- The practical size of the effect (uplift, revenue impact) if available.
- A clear recommendation: ship the variation, keep testing, or stop/iterate.
Do not restate raw numbers already visible in the data verbatim; interpret them.

Data:
"""


def _secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        # Covers both a missing key and no secrets.toml existing at all
        # (StreamlitSecretNotFoundError, a FileNotFoundError subclass) —
        # either way, the credential just isn't configured yet.
        return ""


def get_api_key() -> str:
    return _secret("GEMINI_API_KEY")


def is_configured() -> bool:
    return bool(get_api_key())


def generate_conclusion(
    data: dict[str, Any],
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> dict:
    """
    Sends `data` (typically the Airtable payload plus each method's
    conclusion string) to Gemini and asks for a written interpretation.
    Returns {"ok": bool, "text": Optional[str], "error": Optional[str]}.
    """
    key = api_key or get_api_key()
    if not key:
        return {"ok": False, "text": None, "error": "No Gemini API key configured."}

    try:
        from google import genai
    except ImportError as e:
        return {"ok": False, "text": None, "error": f"google-genai isn't installed: {e}"}

    prompt = _PROMPT_INSTRUCTIONS + json.dumps(data, indent=2, default=str)

    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(model=model, contents=prompt)
        text = (response.text or "").strip()
        if not text:
            return {"ok": False, "text": None, "error": "Gemini returned an empty response."}
        return {"ok": True, "text": text, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "text": None, "error": str(e)}
