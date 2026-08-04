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

DEFAULT_MODEL = "gemini-3.6-flash"

_PROMPT_INSTRUCTIONS = """\
You are a conversion-rate-optimization analyst reviewing the results of an
A/B test. Below is a JSON object with the computed statistics (frequentist,
Bayesian, and/or continuous-metric analysis, and an optional pre-test MDE
projection) for a control ("Control") vs. a variation ("Variation"). It may
also include a "custom_code" field — the actual implementation code for the
variation — use it only to understand what was really being tested, not as
something to comment on directly.

Make sure that you structure your answer thorougly by KPI. Each KPI should have a headline like this before summarizing the results:
KPI: [KPI] ([test type], [Percentage] Confidence, [Percentage] Power)

Write a short, plain-language conclusion (3-6 sentences), IN DUTCH, that a
stakeholder without a statistics background could act on. Cover:
- Whether the result is statistically significant / conclusive, and how confident to be (per KPI).
- The practical size of the effect (uplift, revenue impact) if available (per KPI).
- A clear recommendation: ship the variation, keep testing, or stop/iterate (based on all KPIs).
- If there is sufficient data to infer it, also write a possible explanation from a psychological angle using known principles, fallacies, biases (based on all KPIs).
- Be cautious in your final conclusions and rather use 'the data suggests' instead of 'it is clear that'. when drawing a conclusion. Use a scientific mind, but plain and concise language to communicate the findings.

Do not restate raw numbers already visible in the data verbatim; interpret them.
Respond entirely in Dutch, including the recommendation.

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
