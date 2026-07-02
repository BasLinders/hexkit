"""
ui_components.py
Reusable Streamlit UI blocks shared across all export modes.
"""
from __future__ import annotations
from typing import Literal, Optional, cast
import streamlit as st
from bq_client import (
    is_authenticated, get_auth_url, exchange_code_for_credentials,
    sign_out, list_projects, list_datasets, dry_run, run_preview,
    run_query, df_to_csv_bytes, export_to_sheets, autodetect_variants,
    autodetect_kpis
)
from sql_builder import VariantPair, ExperimentConfig


# ---------------------------------------------------------------------------
# Auth panel
# ---------------------------------------------------------------------------

def render_auth_panel() -> bool:
    """Renders sign-in/sign-out. Returns True if authenticated."""
    client_id     = st.session_state.get("gcp_client_id", "")
    client_secret = st.session_state.get("gcp_client_secret", "")

    params = st.query_params
    if "code" in params and not is_authenticated():
        try:
            state = params.get("state", "")
            exchange_code_for_credentials(params["code"], state, client_id, client_secret)
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {e}")

    if is_authenticated():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.success("✓ Connected to Google")
        with col2:
            if st.button("Sign out", use_container_width=True):
                sign_out()
                st.rerun()
        return True
    else:
        st.info(
            "Sign in with your Google account to access BigQuery. "
            "You'll be taken to Google in a new tab — after signing in, "
            "continue in that tab."
        )
        auth_url = get_auth_url(client_id, client_secret)
        st.link_button(
            "🔐 Sign in with Google",
            auth_url,
            use_container_width=True,
            type="primary",
        )
        return False


# ---------------------------------------------------------------------------
# Connection selectors (project / dataset)
# ---------------------------------------------------------------------------

def render_connection_selectors() -> tuple[Optional[str], Optional[str]]:
    """Renders project + dataset dropdowns. Returns (project_id, dataset_id)."""
    st.subheader("BigQuery connection")

    # --- Project -------------------------------------------------------------
    if "projects_cache" not in st.session_state:
        with st.spinner("Loading projects…"):
            st.session_state["projects_cache"] = list_projects()

    # projects is {project_id: display_name}
    projects: dict[str, str] = st.session_state["projects_cache"]
    if not projects:
        st.warning("No projects found for this account.")
        return None, None

    project_ids = list(projects.keys())

    def _project_label(pid: str) -> str:
        name = projects.get(pid, "")
        return f"{name}  ({pid})" if name else pid

    prior_project = st.session_state.get("selected_project")
    project = st.selectbox(
        "Project",
        options=project_ids,
        index=project_ids.index(prior_project) if prior_project in project_ids else 0,
        format_func=_project_label,
        key="selected_project",
        help="GCP project that owns the BigQuery dataset.",
    )

    # --- Dataset -------------------------------------------------------------
    dataset_cache_key = f"datasets_{project}"
    if dataset_cache_key not in st.session_state:
        with st.spinner(f"Loading datasets for {project}…"):
            st.session_state[dataset_cache_key] = list_datasets(project)

    # datasets is {dataset_id: friendly_name}
    datasets: dict[str, str] = st.session_state[dataset_cache_key]
    if not datasets:
        st.warning(f"No datasets found in {project}.")
        return project, None

    dataset_ids = list(datasets.keys())

    def _dataset_label(did: str) -> str:
        name = datasets.get(did, "")
        return f"{name}  ({did})" if name else did

    dataset = st.selectbox(
        "Dataset",
        options=dataset_ids,
        format_func=_dataset_label,
        key="selected_dataset",
        help="BigQuery dataset containing the GA4 events_* tables.",
    )

    return project, dataset


# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------

def render_date_range() -> tuple[str, str]:
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start date", key="start_date")
    with col2:
        end = st.date_input("End date", key="end_date")
    return str(start), str(end)


# ---------------------------------------------------------------------------
# Variant inputs with auto-detect
# ---------------------------------------------------------------------------

def render_variant_inputs(
    project: str,
    dataset: str,
    start_date: str,
    end_date: str,
    key_prefix: str = "main",
    show_multi_experiment: bool = True,
) -> tuple[str, Literal["exact", "like"], str, list[ExperimentConfig]]:
    """
    Returns: (param_key, match_strategy, exp_prefix, experiments)
    exp_prefix is the ID of the first experiment (or empty for multi-experiment mode).
    """
    st.subheader("Experiment configuration")

    param_key = st.text_input(
        "Variant parameter key",
        value="exp_variant_string",
        help="The event_params key used to identify experiment variants.",
        key=f"{key_prefix}_param_key",
    )

    match_strategy = cast(
        Literal["exact", "like"],
        st.radio(
            "Variant matching strategy",
            options=["exact", "like"],
            format_func=lambda x: "Exact match" if x == "exact" else "Prefix / LIKE match",
            horizontal=True,
            key=f"{key_prefix}_match_strategy",
            help=(
                "Exact match: variant strings must equal the entered value exactly. "
                "Prefix / LIKE match: matches any string starting with the experiment ID — "
                "recommended for Convert and most tag-based platforms."
            ),
        ),
    )

    multi_exp = False
    if show_multi_experiment:
        multi_exp = st.toggle(
            "Multi-experiment mode (queue multiple experiments in one run)",
            key=f"{key_prefix}_multi_exp",
        )

    n_experiments = 1
    if multi_exp:
        n_experiments = st.number_input(
            "Number of experiments",
            min_value=1,
            max_value=20,
            value=st.session_state.get(f"{key_prefix}_n_experiments", 1),
            step=1,
            key=f"{key_prefix}_n_experiments",
        )

    experiments: list[ExperimentConfig] = []
    first_exp_id = ""

    for exp_idx in range(int(n_experiments)):
        exp_label = f"Experiment {exp_idx + 1}" if n_experiments > 1 else "Experiment"
        with st.expander(exp_label, expanded=True):

            # --- Experiment ID (always first, always required for auto-detect) ---
            exp_id = st.text_input(
                "Experiment ID",
                placeholder="e.g. 1004194641",
                help=(
                    "The numerical experiment ID as assigned by your testing tool. "
                    "Used to filter auto-detect results to only this experiment's variant strings — "
                    "any variant string containing this ID will be returned, "
                    "regardless of tool-specific prefixes like 'CONV-' or 'EXP-'."
                ),
                key=f"{key_prefix}_exp_{exp_idx}_id",
            )
            if exp_idx == 0:
                first_exp_id = exp_id

            # Auto-detect button — gated on experiment ID being filled
            can_autodetect = bool(param_key) and bool(exp_id)
            if not can_autodetect:
                st.caption("ℹ️ Enter an experiment ID above to enable auto-detect.")

            autodetect_key = f"{key_prefix}_exp_{exp_idx}_detected"
            if st.button(
                "🔍 Auto-detect variant strings",
                disabled=not can_autodetect or not project or not dataset,
                key=f"{key_prefix}_exp_{exp_idx}_autodetect_btn",
            ):
                with st.spinner("Querying distinct variant strings…"):
                    found = autodetect_variants(
                        project, dataset, start_date, end_date,
                        param_key, exp_id,
                    )
                st.session_state[autodetect_key] = found

            detected = st.session_state.get(autodetect_key, [])
            if detected:
                st.caption(
                    f"{len(detected)} variant string(s) detected. "
                    "Assign a label to each one. Set unused variants to **Skip**."
                )
                _label_opts = ["A — Control", "B", "C", "D", "E", "F", "G", "H", "Skip"]

                variants = []
                seen_labels: set[str] = set()
                has_control = False

                for vi, variant_str in enumerate(detected):
                    default_idx = min(vi, len(_label_opts) - 2)
                    col_str, col_sel = st.columns([3, 1])
                    with col_str:
                        st.text(variant_str)
                    with col_sel:
                        assigned = st.selectbox(
                            "Label",
                            options=_label_opts,
                            index=default_idx,
                            key=f"{key_prefix}_exp_{exp_idx}_assign_{vi}",
                            label_visibility="collapsed",
                        )

                    if assigned == "Skip":
                        continue

                    label = assigned[0]  # "A — Control" → "A", "B" → "B"
                    if label == "A":
                        has_control = True
                    if label in seen_labels:
                        st.warning(f"Label **{label}** is assigned more than once — each variant needs a unique label.")
                    seen_labels.add(label)
                    variants.append(VariantPair(label=label, string=variant_str))

                if not has_control and variants:
                    st.warning("No variant is assigned as **A (Control)**. Assign at least one variant to A.")

            else:
                # Manual entry — shown when auto-detect hasn't been run or returned nothing
                n_variants = st.number_input(
                    "Number of variants",
                    min_value=2,
                    max_value=26,
                    value=st.session_state.get(f"{key_prefix}_exp_{exp_idx}_n_variants", 2),
                    step=1,
                    key=f"{key_prefix}_exp_{exp_idx}_n_variants",
                )
                label_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                variants = []
                for vi in range(int(n_variants)):
                    v_str = st.text_input(
                        f"Variant {label_chars[vi]} string",
                        key=f"{key_prefix}_exp_{exp_idx}_variant_{vi}",
                    )
                    variants.append(VariantPair(label=label_chars[vi], string=v_str))

            experiments.append(ExperimentConfig(
                experiment_id=exp_id or f"exp_{exp_idx + 1}",
                prefix=exp_id,
                variants=variants,
            ))

    return param_key, match_strategy, first_exp_id, experiments


# ---------------------------------------------------------------------------
# KPI checkboxes
# ---------------------------------------------------------------------------

EVENT_KPI_MAP = {
    "purchase": ["kpi_transactions", "kpi_aov"],
    "add_to_cart": ["kpi_add_to_cart"],
    "add_payment_info": ["kpi_ideal"],
    "page_view": ["kpi_login", "kpi_create_account"],
}

KPI_LABELS = {
    "kpi_transactions": "Transactions",
    "kpi_aov": "Average order value",
    "kpi_add_to_cart": "Add to cart",
    "kpi_ideal": "iDEAL payment",
    "kpi_device_split": "Device split (mobile / desktop)",
    "kpi_login": "⚠️ Login page visits",
    "kpi_create_account": "⚠️ Account creation page visits",
}

COST_WARNING_KPIS = {"kpi_login", "kpi_create_account"}

ZERO_COST_KPIS = {
    "kpi_transactions", "kpi_aov", "kpi_add_to_cart",
    "kpi_ideal", "kpi_device_split"
}


def render_kpi_checkboxes(
    project: Optional[str],
    dataset: Optional[str],
    start_date: str,
    end_date: str,
    key_prefix: str = "kpi",
    show_device_split: bool = True,
) -> dict[str, bool]:
    st.subheader("KPIs")
    st.caption(
        "Checked KPIs are included in the query. "
        "⚠️ items add a page_view scan and meaningfully increase bytes processed."
    )

    # Auto-detect button
    autodetect_key = f"{key_prefix}_detected_events"
    if st.button(
        "🔍 Auto-detect available KPIs",
        disabled=not project or not dataset,
        key=f"{key_prefix}_autodetect_btn",
    ):
        if project and dataset:  # narrows Optional[str] → str for type checker
            with st.spinner("Scanning for available event types…"):
                found_events = autodetect_kpis(project, dataset, start_date, end_date)
                st.session_state[autodetect_key] = found_events

    detected_events = st.session_state.get(autodetect_key, [])
    if detected_events:
        st.success(f"Detected events: {', '.join(detected_events)}")

    def _is_available(kpi_key: str) -> bool:
        if not detected_events:
            return True  # no detection run yet → all enabled
        for event, kpis in EVENT_KPI_MAP.items():
            if kpi_key in kpis and event not in detected_events:
                return False
        return True

    kpis: dict[str, bool] = {}

    st.markdown("**Standard KPIs** (no extra scan cost)")
    for key in ZERO_COST_KPIS:
        if key == "kpi_device_split" and not show_device_split:
            continue
        available = _is_available(key)
        kpis[key] = st.checkbox(
            KPI_LABELS[key],
            value=True,
            disabled=not available,
            help="Not detected in this dataset." if not available else None,
            key=f"{key_prefix}_{key}",
        )

    st.markdown("**Extended KPIs** (adds page_view to scan — increases bytes processed)")
    for key in COST_WARNING_KPIS:
        available = _is_available(key)
        kpis[key] = st.checkbox(
            KPI_LABELS[key],
            value=False,
            disabled=not available,
            help="Not detected in this dataset." if not available else None,
            key=f"{key_prefix}_{key}",
        )

    return kpis


# ---------------------------------------------------------------------------
# Pre-execution gate + query runner
# ---------------------------------------------------------------------------

def render_execution_gate(
    project: str,
    sql: str,
    result_key: str = "query_result",
    allow_preview: bool = True,
):
    """
    Always runs dry-run and shows bytes.
    Offers optional preview (25 rows).
    Shows confirm button to run full query.
    """
    st.divider()
    st.subheader("Pre-execution check")

    with st.spinner("Estimating query cost…"):
        cost = dry_run(project, sql)

    if cost["error"]:
        st.error(f"Dry-run failed: {cost['error']}")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Estimated scan", cost["display"])
    with col2:
        st.metric("Free tier usage", f"{cost['free_tier_pct']}%", help="Based on 1 TB/month free tier")

    if allow_preview:
        if st.button("👁️ Preview (25 rows)", key=f"{result_key}_preview_btn"):
            with st.spinner("Running preview…"):
                try:
                    df = run_preview(project, sql, limit=25)
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"Preview failed: {e}")

    st.markdown("---")
    if st.button("✅ Run full query", type="primary", key=f"{result_key}_run_btn"):
        with st.spinner("Running query… this may take a moment."):
            try:
                df = run_query(project, sql)
                st.session_state[result_key] = df
                st.success(f"Done — {len(df):,} rows returned.")
            except Exception as e:
                st.error(f"Query failed: {e}")

    df = st.session_state.get(result_key)
    if df is not None:
        st.dataframe(df, use_container_width=True)
        render_export_options(df, result_key, project)


# ---------------------------------------------------------------------------
# Export options
# ---------------------------------------------------------------------------

def render_export_options(df, key_prefix: str, project: str):
    st.subheader("Export")
    col1, col2 = st.columns(2)
    with col1:
        csv = df_to_csv_bytes(df)
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name=f"{key_prefix}_export.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        if st.button("📊 Export to Google Sheets", use_container_width=True, key=f"{key_prefix}_sheets_btn"):
            with st.spinner("Creating Google Sheet…"):
                url = export_to_sheets(df, title=f"{key_prefix} export")
            if url:
                st.markdown(f"[Open in Google Sheets]({url})")


# ---------------------------------------------------------------------------
# SQL viewer (collapsible)
# ---------------------------------------------------------------------------

def render_sql_viewer(sql: str, key: str = "sql_viewer"):
    with st.expander("View generated SQL", expanded=False):
        st.code(sql, language="sql")
        st.download_button(
            "Download SQL",
            data=sql.encode(),
            file_name="query.sql",
            mime="text/plain",
            key=f"{key}_download",
        )
