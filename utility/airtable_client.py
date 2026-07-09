"""
airtable_client.py
Thin REST wrapper for pushing automation results to Airtable.

Mirrors the return-a-dict-with-error style used in bq_client.py rather than
raising, since callers are Streamlit pages that just want to show st.error().
"""
from __future__ import annotations

from typing import Any, Optional

import requests
import streamlit as st

API_URL = "https://api.airtable.com/v0"


def _secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        # Covers both a missing key and no secrets.toml existing at all
        # (StreamlitSecretNotFoundError, a FileNotFoundError subclass) —
        # either way, the credential just isn't configured yet.
        return ""


def get_credentials() -> dict[str, str]:
    """
    Reads Airtable credentials from session state first (set via the UI, since
    no Airtable token exists yet), falling back to st.secrets:
    AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME.
    """
    return {
        "api_key": st.session_state.get("airtable_api_key") or _secret("AIRTABLE_API_KEY"),
        "base_id": st.session_state.get("airtable_base_id") or _secret("AIRTABLE_BASE_ID"),
        "table_name": st.session_state.get("airtable_table_name") or _secret("AIRTABLE_TABLE_NAME"),
    }


def is_configured() -> bool:
    return all(get_credentials().values())


def push_record(base_id: str, table_name: str, api_key: str, fields: dict[str, Any]) -> dict:
    """
    Creates a single record in an Airtable table.
    Returns {"ok": bool, "record_id": Optional[str], "error": Optional[str]}.
    """
    url = f"{API_URL}/{base_id}/{table_name}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"records": [{"fields": fields}], "typecast": True}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        record = resp.json()["records"][0]
        return {"ok": True, "record_id": record["id"], "error": None}
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = resp.json().get("error", {}).get("message", "")
        except Exception:
            pass
        return {"ok": False, "record_id": None, "error": detail or str(e)}
    except Exception as e:
        return {"ok": False, "record_id": None, "error": str(e)}
