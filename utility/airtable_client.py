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
    api_key is an operational secret (one PAT covers every base/workspace it's
    been granted access to) — it only ever comes from st.secrets, never from
    user input. base_id/table_name are just the currently selected base/table
    (set live by the discovery dropdowns in the Send step), falling back to
    st.secrets defaults to seed their initial selection.
    """
    return {
        "api_key": _secret("AIRTABLE_API_KEY"),
        "base_id": st.session_state.get("airtable_base_id") or _secret("AIRTABLE_BASE_ID"),
        "table_name": st.session_state.get("airtable_table_name") or _secret("AIRTABLE_TABLE_NAME"),
    }


def is_configured() -> bool:
    return all(get_credentials().values())


def _extract_error(resp: "requests.Response") -> str:
    try:
        return resp.json().get("error", {}).get("message", "")
    except Exception:
        return ""


def push_record(base_id: str, table_name: str, api_key: str, fields: dict[str, Any]) -> dict:
    """
    Creates a single record in an Airtable table. `table_name` may be either a
    table name or a table ID — Airtable's API accepts both in this URL slot,
    and the Send step now passes the ID (stable across renames).
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
        return {"ok": False, "record_id": None, "error": _extract_error(resp) or str(e)}
    except Exception as e:
        return {"ok": False, "record_id": None, "error": str(e)}


def update_record(base_id: str, table_name: str, api_key: str, record_id: str, fields: dict[str, Any]) -> dict:
    """
    Patches an existing record (partial update — untouched fields keep their
    current value). Used when the Send step's record lookup finds an existing
    row (e.g. by an experiment-ID field) that this result should be appended
    to, rather than always creating a new record.
    Returns {"ok": bool, "record_id": Optional[str], "error": Optional[str]}.
    """
    url = f"{API_URL}/{base_id}/{table_name}/{record_id}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"fields": fields, "typecast": True}

    try:
        resp = requests.patch(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        record = resp.json()
        return {"ok": True, "record_id": record["id"], "error": None}
    except requests.exceptions.HTTPError as e:
        return {"ok": False, "record_id": None, "error": _extract_error(resp) or str(e)}
    except Exception as e:
        return {"ok": False, "record_id": None, "error": str(e)}


def search_records(
    base_id: str, table_name: str, api_key: str, field_name: str, search_term: str, max_records: int = 25,
) -> dict:
    """
    Finds existing records whose `field_name` value contains `search_term`
    (case-insensitive substring match), via Airtable's `filterByFormula`.
    Used by the Send step to look up an existing record (e.g. by an
    experiment-ID field, often an Airtable autonumber) so a result can be
    appended to it instead of always creating a new record.
    Returns {"ok": bool, "records": [{"id", "fields"}], "error": Optional[str]}.
    """
    escaped_field = field_name.replace('"', '\\"')
    escaped_term = search_term.replace("\\", "\\\\").replace('"', '\\"')
    formula = f'SEARCH(LOWER("{escaped_term}"), LOWER({{{escaped_field}}}))'

    url = f"{API_URL}/{base_id}/{table_name}"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"filterByFormula": formula, "maxRecords": max_records}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        records = [
            {"id": r["id"], "fields": r.get("fields", {})}
            for r in resp.json().get("records", [])
        ]
        return {"ok": True, "records": records, "error": None}
    except requests.exceptions.HTTPError as e:
        return {"ok": False, "records": [], "error": _extract_error(resp) or str(e)}
    except Exception as e:
        return {"ok": False, "records": [], "error": str(e)}


def list_bases(api_key: str) -> dict:
    """
    Lists every Airtable base this PAT can see, via the metadata API
    (GET /v0/meta/bases — requires the `schema.bases:read` scope on the token).
    Paginates via `offset` since a token spanning many workspaces can exceed
    the 1000-base single-page cap.
    Returns {"ok": bool, "bases": {base_id: name}, "error": Optional[str]}.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    bases: dict[str, str] = {}
    params: dict[str, str] = {}
    try:
        while True:
            resp = requests.get(f"{API_URL}/meta/bases", headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            bases.update({b["id"]: b["name"] for b in data.get("bases", [])})
            offset = data.get("offset")
            if not offset:
                break
            params = {"offset": offset}
        return {"ok": True, "bases": bases, "error": None}
    except requests.exceptions.HTTPError as e:
        return {"ok": False, "bases": {}, "error": _extract_error(resp) or str(e)}
    except Exception as e:
        return {"ok": False, "bases": {}, "error": str(e)}


def list_tables(api_key: str, base_id: str) -> dict:
    """
    Lists the tables in a base, each with its field names, via the metadata
    API (GET /v0/meta/bases/{base_id}/tables — same `schema.bases:read` scope).
    Returns {"ok": bool, "tables": [{"id", "name", "fields": [str, ...]}], "error": Optional[str]}.
    """
    url = f"{API_URL}/meta/bases/{base_id}/tables"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = [
            {
                "id": t["id"],
                "name": t["name"],
                "fields": [f["name"] for f in t.get("fields", [])],
            }
            for t in resp.json().get("tables", [])
        ]
        return {"ok": True, "tables": tables, "error": None}
    except requests.exceptions.HTTPError as e:
        return {"ok": False, "tables": [], "error": _extract_error(resp) or str(e)}
    except Exception as e:
        return {"ok": False, "tables": [], "error": str(e)}
