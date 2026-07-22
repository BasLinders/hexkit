"""
bq_client.py
Thin wrapper around google-cloud-bigquery and google-auth-oauthlib.

Streamlit Cloud notes
---------------------
- Secrets are read from st.secrets, not os.getenv.
- When Google redirects back after OAuth, the browser navigates away and
  returns, breaking the WebSocket and starting a fresh Streamlit session.
  Session state written before the redirect is therefore NOT available after.
  Fix: always reconstruct the Flow from st.secrets on the callback side;
  skip OAuth state verification (acceptable for server-side apps where the
  redirect URI is already locked down).
- Credentials ARE stored in session state — they survive for the duration of
  the browser session (until the tab is closed or the page is hard-refreshed).
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.parse
from typing import Optional

import streamlit as st

try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import Flow
    from google.cloud import bigquery          # type: ignore[import-untyped]
    from google.cloud import resourcemanager_v3  # type: ignore[import-untyped]
    import pandas as pd
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


# ---------------------------------------------------------------------------
# OAuth config
# ---------------------------------------------------------------------------

SCOPES = [
    "https://www.googleapis.com/auth/bigquery",
    "https://www.googleapis.com/auth/cloud-platform.read-only",
    "https://www.googleapis.com/auth/drive.file",   # for Sheets export
]


def _get_redirect_base_url() -> str:
    """
    Read the app's base URL from secrets so it can be changed per deployment
    without touching code. Must be a single string with no page path —
    get_redirect_uri() appends the page path. Register each page's full URL
    (e.g. https://hexkit.streamlit.app/automation) as its own authorised
    redirect URI in the Google Cloud OAuth client, not here.
    Expected: BQ_REDIRECT_URI = "https://hexkit.streamlit.app"
    """
    try:
        base = st.secrets["BQ_REDIRECT_URI"]
    except Exception:
        # Covers both a missing key and no secrets.toml existing at all
        # (StreamlitSecretNotFoundError, a FileNotFoundError subclass).
        return "https://hexkit.streamlit.app/"
    if not isinstance(base, str):
        raise TypeError(
            "BQ_REDIRECT_URI in secrets must be a single string with no page "
            'path, e.g. BQ_REDIRECT_URI = "https://hexkit.streamlit.app" — '
            f"got {base!r}. Page paths are appended automatically; register "
            "each page's full URL as its own authorised redirect URI in the "
            "Google Cloud OAuth client instead."
        )
    return base


def get_redirect_uri(page_path: str = "") -> str:
    """
    Build the OAuth redirect URI for a given page.
    Each Streamlit page needs its own redirect URI (e.g. .../data_export,
    .../automation) registered as an authorised redirect URI in the Google
    Cloud OAuth client, so the callback lands back on the page that started
    the flow rather than a fixed shared page.
    Omitting page_path returns the bare base URL (legacy single-redirect setups).
    """
    base = _get_redirect_base_url()
    if not page_path:
        return base
    return f"{base.rstrip('/')}/{page_path.strip('/')}"


def _build_client_config(client_id: str, client_secret: str, redirect_uri: str) -> dict:
    """Build a Flow-compatible client config dict from individual credentials."""
    return {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri],
        }
    }


def get_auth_url(
    client_id: str,
    client_secret: str,
    page_path: str = "",
    extra_state: Optional[dict] = None,
) -> str:
    redirect_uri = get_redirect_uri(page_path)
    config = _build_client_config(client_id, client_secret, redirect_uri)
    flow = Flow.from_client_config(config, scopes=SCOPES, redirect_uri=redirect_uri)

    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    # Encode verifier, credentials, the originating page, AND any caller-supplied
    # extra_state into state — all need to survive the redirect, since it starts
    # a fresh Streamlit session with none of this in session state. extra_state
    # lets a caller (e.g. an admin-gated page) carry its own session flags
    # (e.g. admin_authenticated) across that same discontinuity — see
    # exchange_code_for_credentials, which restores them into session state.
    verifier: str = getattr(flow, "code_verifier", None) or ""
    state_data = json.dumps({
        "verifier":      verifier,
        "client_id":     client_id,
        "client_secret": client_secret,
        "page_path":     page_path,
        "extra":         extra_state or {},
    })
    encoded = base64.urlsafe_b64encode(state_data.encode()).decode().rstrip("=")

    parsed = urllib.parse.urlparse(auth_url)
    qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    qs["state"] = [encoded]
    new_query = urllib.parse.urlencode({k: v[0] for k, v in qs.items()})
    auth_url = urllib.parse.urlunparse(parsed._replace(query=new_query))

    return auth_url


def exchange_code_for_credentials(
    code: str,
    state: str = "",
    client_id: str = "",
    client_secret: str = "",
) -> Credentials:
    # Decode state — verifier, credentials, the originating page, and any
    # caller-supplied extra_state (e.g. admin_authenticated) are all embedded here
    verifier: Optional[str] = None
    page_path = ""
    extra_state: dict = {}
    if state:
        try:
            padding = 4 - len(state) % 4
            padded = state + ("=" * (padding % 4))
            state_data = json.loads(base64.urlsafe_b64decode(padded).decode())
            verifier      = state_data.get("verifier", "") or None
            client_id     = state_data.get("client_id", client_id)
            client_secret = state_data.get("client_secret", client_secret)
            page_path     = state_data.get("page_path", "")
            extra_state   = state_data.get("extra", {}) or {}
        except Exception:
            verifier = None

    redirect_uri = get_redirect_uri(page_path)
    config = _build_client_config(client_id, client_secret, redirect_uri)

    flow = Flow.from_client_config(config, scopes=SCOPES, redirect_uri=redirect_uri)
    fetch_kwargs: dict = {"code": code}
    if verifier:
        fetch_kwargs["code_verifier"] = verifier
    flow.fetch_token(**fetch_kwargs)

    # Restore caller-supplied session flags now that this fresh session has a
    # session_state to restore them into — see get_auth_url's extra_state param.
    for key, value in extra_state.items():
        st.session_state[key] = value

    creds = flow.credentials
    st.session_state["credentials"] = _creds_to_dict(creds)
    return creds


def get_credentials() -> Optional[Credentials]:
    """Return valid credentials from session state, refreshing if expired."""
    creds_dict = st.session_state.get("credentials")
    if not creds_dict:
        return None
    creds = Credentials(
        token=creds_dict["token"],
        refresh_token=creds_dict.get("refresh_token"),
        token_uri=creds_dict.get("token_uri"),
        client_id=creds_dict.get("client_id"),
        client_secret=creds_dict.get("client_secret"),
        scopes=creds_dict.get("scopes"),
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        st.session_state["credentials"] = _creds_to_dict(creds)
    return creds if creds.valid else None


def _creds_to_dict(creds: Credentials) -> dict:
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else [],
    }


def is_authenticated() -> bool:
    return get_credentials() is not None


def sign_out():
    for key in ("credentials",):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# BQ client factory
# ---------------------------------------------------------------------------

def get_bq_client(project: Optional[str] = None) -> "bigquery.Client":
    creds = get_credentials()
    if not creds:
        raise RuntimeError("Not authenticated. Please sign in first.")
    return bigquery.Client(credentials=creds, project=project)


# ---------------------------------------------------------------------------
# Project / dataset discovery
# ---------------------------------------------------------------------------

def list_projects() -> dict[str, str]:
    """
    List GCP projects accessible to the authenticated user.
    Returns {project_id: display_name}. display_name may be empty string
    if the project has no friendly name set.
    """
    creds = get_credentials()
    if not creds:
        return {}
    try:
        client = resourcemanager_v3.ProjectsClient(credentials=creds)
        result = {
            p.project_id: p.display_name or ""
            for p in client.search_projects()
        }
        return dict(sorted(result.items()))
    except Exception as e:
        st.warning(f"Could not list projects: {e}")
        return {}


def list_datasets(project: str) -> dict[str, str]:
    """
    List BigQuery datasets in a project.
    Returns {dataset_id: friendly_name}. friendly_name may be empty string
    if the dataset has no friendly name set — this is common for GA4 exports.
    """
    try:
        client = get_bq_client(project)
        result = {
            d.dataset_id: d.friendly_name or ""
            for d in client.list_datasets()
        }
        return dict(sorted(result.items()))
    except Exception as e:
        st.warning(f"Could not list datasets in {project}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Dry run — bytes scanned estimation
# ---------------------------------------------------------------------------

def dry_run(project: str, sql: str) -> dict:
    """
    Returns estimated bytes scanned via BQ dry run.
    Note: dry run does NOT work for scripts containing DDL/DML statements
    (CREATE TABLE, INSERT, IF blocks). The sequential export page handles
    this by skipping the dry-run step and showing a warning instead.
    """
    try:
        client = get_bq_client(project)
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = client.query(sql, job_config=job_config)
        bytes_processed = job.total_bytes_processed
        gb = bytes_processed / 1e9
        tb_free_gb = 1_000
        return {
            "bytes": bytes_processed,
            "gb": round(gb, 2),
            "display": _format_bytes(bytes_processed),
            "free_tier_pct": round((gb / tb_free_gb) * 100, 3),
            "error": None,
            "is_dml": False,
        }
    except Exception as e:
        err_str = str(e)
        # BQ raises an error if the script contains DDL/DML and dry_run=True
        is_dml = any(kw in err_str.upper() for kw in ("DDL", "DML", "SCRIPT", "CREATE", "INSERT"))
        return {
            "bytes": 0, "gb": 0, "display": "N/A",
            "free_tier_pct": 0,
            "error": err_str,
            "is_dml": is_dml,
        }


def _format_bytes(n: int) -> str:
    if n < 1_000:
        return f"{n} B"
    if n < 1_000_000:
        return f"{n / 1_000:.1f} KB"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.1f} MB"
    return f"{n / 1_000_000_000:.2f} GB"


# ---------------------------------------------------------------------------
# Monthly free-tier budget tracking
# ---------------------------------------------------------------------------

_FREE_TIER_BYTES: int = 1_000_000_000_000   # 1 TB
_USAGE_CACHE_TTL: int = 300                  # seconds — re-query at most once per 5 min


def _dataset_region(project: str, dataset_id: str) -> str:
    """
    Returns the INFORMATION_SCHEMA region prefix for a dataset's location.
    e.g. 'EU' → 'region-eu', 'us-central1' → 'region-us-central1'.
    Falls back to 'region-eu' on error.
    """
    try:
        client = get_bq_client(project)
        ds = client.get_dataset(f"{project}.{dataset_id}")
        return f"region-{ds.location.lower()}"
    except Exception:
        return "region-eu"


def get_monthly_usage(project: str, dataset_id: str) -> dict:
    """
    Queries INFORMATION_SCHEMA.JOBS_BY_PROJECT for bytes processed this calendar
    month, then derives remaining free-tier budget.

    Result is cached in st.session_state for _USAGE_CACHE_TTL seconds to avoid
    spending extra scan budget on repeated calls within the same session.

    INFORMATION_SCHEMA queries are free (don't count against the 1 TB quota).

    Returns:
        {
          "used_bytes":        int,
          "used_gb":           float,
          "used_display":      str,
          "remaining_bytes":   int,
          "remaining_gb":      float,
          "remaining_display": str,
          "used_pct":          float,   # 0–100, capped at 100
          "query_pct_of_remaining": float | None,   # filled in by caller
          "free_tier_bytes":   int,
          "error":             str | None,
          "permission_denied": bool,
        }
    """
    cache_key = f"monthly_usage_{project}"
    cached = st.session_state.get(cache_key)
    if cached and (time.time() - cached.get("_ts", 0)) < _USAGE_CACHE_TTL:
        return cached

    result: dict = {
        "used_bytes": 0,
        "used_gb": 0.0,
        "used_display": "0 B",
        "remaining_bytes": _FREE_TIER_BYTES,
        "remaining_gb": round(_FREE_TIER_BYTES / 1e9, 2),
        "remaining_display": _format_bytes(_FREE_TIER_BYTES),
        "used_pct": 0.0,
        "query_pct_of_remaining": None,
        "free_tier_bytes": _FREE_TIER_BYTES,
        "error": None,
        "permission_denied": False,
        "_ts": time.time(),
    }

    try:
        region = _dataset_region(project, dataset_id)
        sql = f"""
SELECT IFNULL(SUM(total_bytes_processed), 0) AS bytes_used
FROM `{region}`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE DATE(creation_time) >= DATE_TRUNC(CURRENT_DATE(), MONTH)
  AND job_type = 'QUERY'
  AND state = 'DONE'
  AND cache_hit = FALSE
"""
        client = get_bq_client(project)
        rows = list(client.query(sql).result())
        used = int(rows[0]["bytes_used"]) if rows else 0
        remaining = max(_FREE_TIER_BYTES - used, 0)
        used_pct = min(round((used / _FREE_TIER_BYTES) * 100, 2), 100.0)

        result.update({
            "used_bytes": used,
            "used_gb": round(used / 1e9, 2),
            "used_display": _format_bytes(used),
            "remaining_bytes": remaining,
            "remaining_gb": round(remaining / 1e9, 2),
            "remaining_display": _format_bytes(remaining),
            "used_pct": used_pct,
            "_ts": time.time(),
        })

    except Exception as e:
        err = str(e)
        result["error"] = err
        result["permission_denied"] = (
            "ACCESS_DENIED" in err.upper()
            or "PERMISSION_DENIED" in err.upper()
            or "does not have bigquery.jobs.list" in err
        )

    st.session_state[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def run_query(project: str, sql: str) -> "pd.DataFrame":
    """Execute SQL and return a DataFrame."""
    client = get_bq_client(project)
    job = client.query(sql)
    return job.result().to_dataframe()


def create_scan_session(project: str, create_temp_table_sql: str) -> str:
    """
    Runs a `CREATE TEMP TABLE ... AS SELECT ...` statement with a BigQuery
    session enabled, waits for it to finish, and returns the session_id so
    later queries can read that temp table via run_query_in_session.
    """
    client = get_bq_client(project)
    job_config = bigquery.QueryJobConfig(create_session=True)
    job = client.query(create_temp_table_sql, job_config=job_config)
    job.result()
    return job.session_info.session_id


def run_query_in_session(project: str, sql: str, session_id: str) -> "pd.DataFrame":
    """Runs a query against an existing BigQuery session (e.g. to read a
    TEMP TABLE created by create_scan_session) and returns a DataFrame."""
    client = get_bq_client(project)
    job_config = bigquery.QueryJobConfig(
        connection_properties=[bigquery.ConnectionProperty(key="session_id", value=session_id)]
    )
    job = client.query(sql, job_config=job_config)
    return job.result().to_dataframe()


def run_combined_export(
    project: str,
    create_temp_sql: str,
    select_sqls: dict[str, str],
) -> dict[str, "pd.DataFrame"]:
    """
    Materializes the shared scan once (via a BigQuery session), then runs
    each labeled SELECT against it in that same session — e.g.
    {"binomial": df, "continuous": df} — billing the big scan only once.
    """
    session_id = create_scan_session(project, create_temp_sql)
    return {
        label: run_query_in_session(project, sql, session_id)
        for label, sql in select_sqls.items()
    }


def run_preview(project: str, sql: str, limit: int = 25) -> "pd.DataFrame":
    """
    Strips trailing semicolons, appends LIMIT, runs the query.
    Only valid for pure SELECT queries (not DDL/DML scripts).
    """
    clean = sql.rstrip().rstrip(";")
    last_line = clean.upper().rsplit("\n", 1)[-1]
    if "LIMIT" not in last_line:
        clean = clean + f"\nLIMIT {limit}"
    return run_query(project, clean)


# ---------------------------------------------------------------------------
# Auto-detect helpers
# ---------------------------------------------------------------------------

def autodetect_variants(
    project: str,
    dataset: str,
    start_date: str,
    end_date: str,
    param_key: str,
    prefix: str,
) -> list[str]:
    from utility.sql_builder import build_autodetect_variants_query
    from datetime import datetime, timedelta

    # Use only the last day of the selected date range to minimise scan cost.
    # Variant strings are stable — they don't change over the course of an experiment —
    # so a 1-day window is sufficient to discover all variants.
    # The full date range is preserved for the actual export query.
    end_dt   = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = max(
        end_dt - timedelta(days=1),
        datetime.strptime(start_date, "%Y-%m-%d"),
    )
    sample_start = start_dt.strftime("%Y-%m-%d")
    sample_end   = end_date  # end stays the same

    sql = build_autodetect_variants_query(
        project, dataset, sample_start, sample_end, param_key, prefix
    )
    cost = dry_run(project, sql)
    st.caption(
        f"Auto-detect scanned the last day of your date range "
        f"({sample_start} → {sample_end}): "
        f"**{cost['display']}** — separate from your export query cost."
    )
    df = run_query(project, sql)
    if df.empty:
        if end_dt.date() >= datetime.now().date():
            st.warning(
                f"No variant strings found on {sample_start} → {sample_end}. "
                f"Your end date is today (or later) — BigQuery's GA4 export table "
                f"for today is usually not finalised until several hours after the "
                f"day ends, so it may not contain data yet. Try setting the end date "
                f"to yesterday or earlier and auto-detect again."
            )
        else:
            st.warning(
                f"No variant strings found on {sample_start} → {sample_end} "
                f"for param key '{param_key}' and experiment ID '{prefix}'. "
                f"Double-check the ID and param key, or widen the date range."
            )
        return []
    return df["variant_string"].tolist()


def autodetect_kpis(
    project: str,
    dataset: str,
    start_date: str,
    end_date: str,
) -> list[str]:
    from utility.sql_builder import build_autodetect_kpi_query
    sql = build_autodetect_kpi_query(project, dataset, start_date, end_date)
    df = run_query(project, sql)
    return df["event_name"].tolist() if not df.empty else []


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def df_to_csv_bytes(df: "pd.DataFrame") -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def export_to_sheets(df: "pd.DataFrame", title: str = "BQ Export") -> str:
    """
    Creates a new Google Sheet and writes the DataFrame to it.
    Returns the spreadsheet URL.
    Requires drive.file scope — already requested in OAuth flow.
    Requires: pip install gspread gspread-dataframe
    """
    try:
        import gspread
        from gspread_dataframe import set_with_dataframe
        creds = get_credentials()
        gc = gspread.authorize(creds)
        sh = gc.create(title)
        ws = sh.get_worksheet(0)
        set_with_dataframe(ws, df)
        return sh.url
    except ImportError:
        st.error(
            "Sheets export requires `gspread` and `gspread-dataframe`. "
            "Add them to `requirements.txt`."
        )
        return ""
    except Exception as e:
        st.error(f"Sheets export failed: {e}")
        return ""
