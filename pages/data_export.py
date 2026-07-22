# ============================================================================
# 1. LIBRARIES
# ============================================================================

from __future__ import annotations
from typing import Literal, Optional, Union, cast

import streamlit as st

from utility.bq_client import (
    dry_run,
    get_monthly_usage,
    run_query,
    run_preview,
    df_to_csv_bytes,
    export_to_sheets,
    autodetect_variants,
    autodetect_kpis,
)
from utility.bq_ui_components import (
    render_gcp_credentials_gate,
    render_connection_selectors,
    render_date_range,
    render_variant_inputs,
    render_kpi_checkboxes,
    render_sql_viewer,
    render_export_options,
)
from utility.sql_builder import (
    BaselineParams,
    BinomialParams,
    ContinuousParams,
    SequentialParams,
    InteractionParams,
    ExperimentConfig,
    VariantPair,
    build_baseline,
    build_binomial,
    build_continuous,
    build_sequential,
    build_interaction,
    build_autodetect_variants_query,
)


# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

# --- Mode registry -----------------------------------------------------------
# Add new export modes here. Each entry: (key, sidebar_label).
# The key must match the dispatch dict in run().

EXPORT_MODES: list[tuple[str, str]] = [
    ("baseline",     "📊 Baseline export"),
    ("binomial",     "🔢 Experiment — Binomial"),
    ("continuous",   "📈 Experiment — Continuous"),
    ("sequential",   "🔁 Sequential test"),
    ("interaction",  "🔀 Interaction export"),
]


# --- Input renderers ---------------------------------------------------------
# Each function renders the mode-specific UI and returns a filled params
# dataclass, or None if the inputs are not yet valid enough to build SQL.

def _render_baseline_inputs(
    project: str,
    dataset: str,
    start_date: str,
    end_date: str,
) -> Optional[BaselineParams]:

    st.subheader("Output data")
    output_type = cast(
        Literal["binomial", "revenue"],
        st.radio(
            "Output type",
            options=["binomial", "revenue"],
            format_func=lambda x: (
                "Binomial (visitors / conversions)" if x == "binomial"
                else "Revenue (RPV / RPT)"
            ),
            horizontal=True,
            key="bl_output_type",
            help=(
                "Binomial: visitor and conversion counts, for conversion-rate sample "
                "size planning. Revenue: purchase revenue alongside visitors/transactions, "
                "for RPV (revenue per visitor) or RPT/AOV (revenue per transaction) "
                "sample size planning."
            ),
        ),
    )

    shape_options = ["aggregate", "daily"]
    if output_type == "revenue":
        shape_options.append("per_user")
    shape_labels = {
        "aggregate": "Aggregate (single row for the date range)",
        "daily": "Daily rows (one row per day)",
        "per_user": "Per-order raw values (for model fitting)",
    }
    if st.session_state.get("bl_output_shape") not in shape_options:
        st.session_state["bl_output_shape"] = "aggregate"
    output_shape = cast(
        Literal["aggregate", "daily", "per_user"],
        st.radio(
            "Output shape",
            options=shape_options,
            format_func=lambda x: shape_labels[x],
            horizontal=True,
            key="bl_output_shape",
            help=(
                "Aggregate: one summary row, for the pre-test tool's manual baseline "
                "inputs. Daily rows: one row per day, for the pre-test tool's seasonal "
                "forecasting CSV upload. Per-order raw values: one row per completed "
                "order, for fitting a Negative Binomial or Gamma model from raw data "
                "in the pre-test tool's continuous KPI mode."
            ),
        ),
    )

    st.divider()
    st.subheader("Page filter")
    st.caption(
        "Optional. Filter to users who visited specific pages before inclusion. "
        "Leave disabled to include all users in the date range."
    )

    use_filter = st.toggle(
        "Enable page filter",
        value=False,
        key="bl_use_filter",
        help=(
            "When enabled, only users who visited a matching page during the date range "
            "are included. Useful for scoping analysis to a specific section of the site — "
            "for example, a product category or checkout flow."
        ),
    )
    page_filter_type = None
    page_filter_value = ""

    if use_filter:
        page_filter_type = cast(
            Literal["regex", "contains"],
            st.radio(
                "Filter type",
                options=["contains", "regex"],
                format_func=lambda x: "URL contains" if x == "contains" else "Regex pattern",
                horizontal=True,
                key="bl_filter_type",
                help=(
                    "URL contains: simple substring match against the full page URL. "
                    "Regex: BigQuery REGEXP_CONTAINS syntax for more specific patterns, "
                    "e.g. r'\\.html$' to match only HTML product pages."
                ),
            ),
        )
        page_filter_value = st.text_input(
            "Filter value",
            placeholder=".html  |  /products/  |  \\.html$",
            key="bl_filter_value",
            help=(
                "The string or pattern to match against page_location. "
                "Contains example: '/products/shoes/'. "
                "Regex example: r'/(product|category)/'. "
                "Matching is case-sensitive."
            ),
        )
        if not page_filter_value:
            st.warning("Enter a filter value or disable the page filter.")
            return None

    return BaselineParams(
        project=project,
        dataset=dataset,
        start_date=start_date,
        end_date=end_date,
        output_type=output_type,
        output_shape=output_shape,
        page_filter_type=page_filter_type,
        page_filter_value=page_filter_value,
    )


def _render_binomial_inputs(
    project: str,
    dataset: str,
    start_date: str,
    end_date: str,
) -> Optional[BinomialParams]:

    param_key, match_strategy, exp_prefix, experiments = render_variant_inputs(
        project, dataset, start_date, end_date,
        key_prefix="bin",
        show_multi_experiment=True,
    )
    match_strategy = cast(Literal["exact", "like"], match_strategy)

    st.divider()
    post_exposure = st.toggle(
        "Post-exposure filtering",
        value=True,
        help="Only count events that occurred after the user's first experiment exposure. Recommended.",
        key="bin_post_exposure",
    )

    st.divider()
    kpis = render_kpi_checkboxes(
        project, dataset, start_date, end_date,
        key_prefix="bin_kpi",
        show_device_split=True,
    )

    if not _experiments_valid(experiments):
        st.warning("All experiments need at least two filled variant strings to continue.")
        return None

    return BinomialParams(
        project=project,
        dataset=dataset,
        start_date=start_date,
        end_date=end_date,
        param_key=param_key,
        match_strategy=match_strategy,
        experiments=experiments,
        post_exposure_filter=post_exposure,
        kpi_transactions=kpis.get("kpi_transactions", True),
        kpi_add_to_cart=kpis.get("kpi_add_to_cart", True),
        kpi_aov=kpis.get("kpi_aov", True),
        kpi_ideal=kpis.get("kpi_ideal", False),
        kpi_device_split=kpis.get("kpi_device_split", True),
        kpi_login=kpis.get("kpi_login", False),
        kpi_create_account=kpis.get("kpi_create_account", False),
    )


def _render_continuous_inputs(
    project: str,
    dataset: str,
    start_date: str,
    end_date: str,
) -> Optional[ContinuousParams]:

    st.info(
        "Continuous queries return one row per user (or user-transaction). "
        "Review the preview and scan estimate before running.",
        icon="💡",
    )

    param_key, match_strategy, exp_prefix, experiments = render_variant_inputs(
        project, dataset, start_date, end_date,
        key_prefix="cont",
        show_multi_experiment=False,
    )
    match_strategy = cast(Literal["exact", "like"], match_strategy)

    st.divider()
    post_exposure = st.toggle(
        "Post-exposure filtering",
        value=True,
        help="Only count purchases that occurred after the user's first exposure.",
        key="cont_post_exposure",
    )

    col1, col2 = st.columns(2)
    with col1:
        device_filter = cast(
            Literal["all", "desktop", "mobile"],
            st.radio(
                "Device filter",
                options=["all", "desktop", "mobile"],
                format_func=str.capitalize,
                horizontal=True,
                key="cont_device_filter",
                help="Applied at query level — only the selected device type is returned.",
            ),
        )
    with col2:
        query_mode = cast(
            Literal["all_users", "revenue_only"],
            st.radio(
                "Query mode",
                options=["all_users", "revenue_only"],
                format_func=lambda x: (
                    "All exposed users (revenue per visitor)"
                    if x == "all_users"
                    else "Users with revenue only (revenue per transaction)"
                ),
                key="cont_query_mode",
                help=(
                    "All exposed users: LEFT JOIN on purchases — non-buyers get revenue = 0. "
                    "Use this for revenue-per-visitor analysis where zero values matter. "
                    "Revenue only: INNER JOIN — excludes non-buyers entirely. "
                    "Use this for revenue-per-transaction analysis."
                ),
            ),
        )

    if not _experiments_valid(experiments):
        st.warning("Configure at least one experiment with filled variant strings to continue.")
        return None

    return ContinuousParams(
        project=project,
        dataset=dataset,
        start_date=start_date,
        end_date=end_date,
        param_key=param_key,
        match_strategy=match_strategy,
        experiments=experiments,
        device_filter=device_filter,
        query_mode=query_mode,
        post_exposure_filter=post_exposure,
    )


def _render_sequential_inputs(
    project: str,
    dataset: str,
    start_date: str,
    end_date: str,
) -> Optional[SequentialParams]:

    param_key, match_strategy, exp_prefix, experiments = render_variant_inputs(
        project, dataset, start_date, end_date,
        key_prefix="seq",
        show_multi_experiment=False,
    )
    match_strategy = cast(Literal["exact", "like"], match_strategy)

    st.divider()
    st.subheader("Persistence")

    use_persistence = st.toggle(
        "Use cumulative persistence",
        value=True,
        help=(
            "New users are appended to a permanent BQ table. "
            "The final result aggregates over all historical runs. "
            "Disable to query only the current date range."
        ),
        key="seq_persistence",
    )

    cumulative_table = ""
    reset_cumulative = False

    if use_persistence:
        cumulative_table = st.text_input(
            "Cumulative table",
            value=f"{project}.{dataset}.cumulative_test_data",
            help="Full path: project.dataset.table_name. Created automatically if it does not exist.",
            key="seq_cumulative_table",
        )

        st.markdown("---")
        st.markdown("**⚠️ Danger zone**")
        reset_cumulative = st.toggle(
            "Reset — truncate all historical data",
            value=False,
            help="TRUNCATES the cumulative table. This cannot be undone.",
            key="seq_reset",
        )
        if reset_cumulative:
            st.error(
                f"This will delete **all** data in `{cumulative_table}`. "
                "Cannot be undone. Check the box below to confirm."
            )
            confirmed = st.checkbox(
                "I understand — permanently delete all cumulative data",
                key="seq_reset_confirm",
            )
            if not confirmed:
                reset_cumulative = False

    st.divider()
    kpis = render_kpi_checkboxes(
        project, dataset, start_date, end_date,
        key_prefix="seq_kpi",
        show_device_split=True,
    )

    if not _experiments_valid(experiments):
        st.warning("Configure at least one experiment with filled variant strings to continue.")
        return None

    return SequentialParams(
        project=project,
        dataset=dataset,
        start_date=start_date,
        end_date=end_date,
        param_key=param_key,
        experiments=experiments,
        use_persistence=use_persistence,
        reset_cumulative_data=reset_cumulative,
        cumulative_table=cumulative_table,
        kpi_transactions=kpis.get("kpi_transactions", True),
        kpi_add_to_cart=kpis.get("kpi_add_to_cart", True),
        kpi_aov=kpis.get("kpi_aov", True),
        kpi_ideal=kpis.get("kpi_ideal", False),
        kpi_device_split=kpis.get("kpi_device_split", True),
        kpi_login=kpis.get("kpi_login", False),
        kpi_create_account=kpis.get("kpi_create_account", False),
    )


def _render_interaction_inputs(
    project: str,
    dataset: str,
    start_date: str,
    end_date: str,
) -> Optional[InteractionParams]:

    st.caption(
        "Classifies users by their variant assignment across multiple simultaneous experiments. "
        "Produces one row per combination (AA, AB, BA, BB, …) for interaction effect analysis."
    )

    param_key = st.text_input(
        "Variant parameter key",
        value="exp_variant_string",
        key="int_param_key",
        help=(
            "The GA4 event_params key that stores the experiment variant string. "
            "Common values: 'exp_variant_string' (Convert), "
            "'varify_abtestshort' (Varify). "
            "Check your tagging setup if unsure."
        ),
    )

    n_experiments = st.number_input(
        "Number of experiments",
        min_value=2,
        max_value=20,
        value=2,
        step=1,
        key="int_n_experiments",
        help="Each experiment contributes one character (A or B) to the variant combination label.",
    )

    experiments: list[ExperimentConfig] = []

    for i in range(int(n_experiments)):
        with st.expander(f"Experiment {i + 1}", expanded=(i == 0)):
            exp_id = st.text_input(
                "Label (optional)",
                placeholder=f"experiment_{i + 1}",
                key=f"int_exp_{i}_id",
                help="A short name for this experiment used in debug output. Does not affect the query.",
            )
            prefix = st.text_input(
                "Variant prefix (required for auto-detect)",
                placeholder="e.g. EXP-2-",
                key=f"int_exp_{i}_prefix",
                help=(
                    "The string that all variant IDs for this experiment start with. "
                    "Used to filter auto-detect results to only this experiment's variants. "
                    "Example: 'CONV-1004144465-' or 'EXP-42-'."
                ),
            )

            detect_key = f"int_exp_{i}_detected"
            if st.button(
                "🔍 Auto-detect",
                disabled=not param_key or not prefix,
                key=f"int_exp_{i}_detect_btn",
            ):
                with st.spinner("Querying distinct variant strings…"):
                    found = autodetect_variants(
                        project, dataset, start_date, end_date, param_key, prefix
                    )
                    st.session_state[detect_key] = found

            detected = st.session_state.get(detect_key, [])

            if detected:
                sel_a = cast(str, st.selectbox(
                    "Variant A", options=detected, key=f"int_exp_{i}_a_sel",
                    help="The variant ID string assigned to the control / original experience.",
                ))
                remaining = [s for s in detected if s != sel_a]
                sel_b = cast(str, st.selectbox(
                    "Variant B", options=remaining, key=f"int_exp_{i}_b_sel",
                    help="The variant ID string assigned to the challenger / treatment experience.",
                ))
                variants = [VariantPair("A", sel_a), VariantPair("B", sel_b)]
            else:
                a_str = st.text_input(
                    "Variant A string",
                    key=f"int_exp_{i}_a",
                    help=(
                        "The full variant ID as it appears in BigQuery. "
                        "Example: 'CONV-1004144465-1004341242'. "
                        "Use auto-detect above to look these up from the dataset."
                    ),
                )
                b_str = st.text_input(
                    "Variant B string",
                    key=f"int_exp_{i}_b",
                    help=(
                        "The full variant ID as it appears in BigQuery. "
                        "Example: 'CONV-1004144465-1004341243'. "
                        "Use auto-detect above to look these up from the dataset."
                    ),
                )
                variants = [VariantPair("A", a_str), VariantPair("B", b_str)]

            experiments.append(ExperimentConfig(
                experiment_id=exp_id or f"experiment_{i + 1}",
                prefix=prefix,
                variants=variants,
            ))

    st.divider()
    st.subheader("KPIs")
    kpi_transactions = st.checkbox(
        "Transactions",
        value=True,
        key="int_kpi_tx",
        help="Count users with at least one purchase and total transactions per variant combination.",
    )
    kpi_add_to_cart = st.checkbox(
        "Add to cart",
        value=True,
        key="int_kpi_atc",
        help="Count users who added at least one item to cart per variant combination.",
    )

    if not _experiments_valid(experiments):
        st.warning("All experiments need a filled Variant A and Variant B string.")
        return None

    return InteractionParams(
        project=project,
        dataset=dataset,
        start_date=start_date,
        end_date=end_date,
        param_key=param_key,
        experiments=experiments,
        kpi_transactions=kpi_transactions,
        kpi_add_to_cart=kpi_add_to_cart,
    )


# --- Validation --------------------------------------------------------------

def _experiments_valid(experiments: list[ExperimentConfig]) -> bool:
    """True if every experiment has at least two non-empty variant strings."""
    if not experiments:
        return False
    for exp in experiments:
        filled = [v for v in exp.variants if v.string and v.string.strip()]
        if len(filled) < 2:
            return False
    return True


# --- SQL dispatcher ----------------------------------------------------------

def _build_sql(
    mode: str,
    params: Union[
        BaselineParams, BinomialParams, ContinuousParams,
        SequentialParams, InteractionParams,
    ],
) -> str:
    """Route params to the correct SQL builder."""
    builders = {
        "baseline":    lambda p: build_baseline(p),
        "binomial":    lambda p: build_binomial(p),
        "continuous":  lambda p: build_continuous(p),
        "sequential":  lambda p: build_sequential(p),
        "interaction": lambda p: build_interaction(p),
    }
    return builders[mode](params)


# --- Execution gate ----------------------------------------------------------

def _render_execution_gate(
    project: str,
    dataset: str,
    sql: str,
    mode: str,
) -> None:
    """
    Pre-execution check panel.
    - Always runs BQ dry-run and shows estimated scan size.
    - Fetches this month's project-level usage from INFORMATION_SCHEMA
      and shows remaining free-tier budget with a progress bar.
    - Sequential mode skips dry-run (DDL/DML not supported by BQ dry-run).
    - Preview (LIMIT 25) available for SELECT-only modes.
    - Full run always gated behind confirm button.
    """
    st.divider()
    st.subheader("Pre-execution check")

    result_key  = f"{mode}_result"
    is_dml_mode = mode == "sequential"

    # --- Dry-run cost estimate ------------------------------------------------
    if is_dml_mode:
        st.info(
            "Sequential queries contain DDL/DML statements (CREATE TABLE, INSERT). "
            "BigQuery does not support dry-run cost estimation for scripts — "
            "scan cost cannot be shown in advance.",
            icon="ℹ️",
        )
        cost: dict = {"bytes": 0, "gb": 0.0, "display": "N/A", "error": "dml", "is_dml": True}
    else:
        with st.spinner("Estimating scan cost…"):
            cost = dry_run(project, sql)

        if cost["error"] and not cost["is_dml"]:
            st.error(f"Dry-run failed: {cost['error']}")
            return

    # --- Monthly budget panel -------------------------------------------------
    with st.spinner("Fetching monthly usage…"):
        usage = get_monthly_usage(project, dataset)

    # Annotate cost dict with share of remaining budget (if both are known)
    query_pct_of_remaining: Optional[float] = None
    if not is_dml_mode and not cost["error"] and usage["remaining_bytes"] > 0:
        query_pct_of_remaining = round(
            (cost["bytes"] / usage["remaining_bytes"]) * 100, 3
        )

    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "This query",
            cost["display"] if not is_dml_mode else "N/A (script)",
            help="Estimated bytes scanned by this query (BigQuery dry-run).",
        )
    with col2:
        st.metric(
            "Used this month",
            usage["used_display"] if not usage["error"] else "Unavailable",
            help=(
                "Bytes processed by all queries in this project since the 1st of the month. "
                "Sourced from INFORMATION_SCHEMA.JOBS_BY_PROJECT. "
                "Cached for 5 minutes."
            ),
        )
    with col3:
        st.metric(
            "Remaining free tier",
            usage["remaining_display"] if not usage["error"] else "Unavailable",
            help="Remaining bytes in the 1 TB/month BigQuery free tier for this project.",
        )

    # Progress bar + contextual labels
    if not usage["error"]:
        used_pct   = usage["used_pct"] / 100          # st.progress expects 0.0–1.0
        bar_colour = (
            "normal" if usage["used_pct"] < 75
            else "inverse" if usage["used_pct"] < 90
            else "off"
        )
        st.progress(
            min(used_pct, 1.0),
            text=(
                f"{usage['used_display']} of 1 TB used this month "
                f"({usage['used_pct']}%)"
            ),
        )
        if query_pct_of_remaining is not None:
            st.caption(
                f"This query consumes **{query_pct_of_remaining}%** "
                f"of your remaining {usage['remaining_display']} budget."
            )
    elif usage["permission_denied"]:
        st.caption(
            "ℹ️ Monthly usage unavailable — "
            "the authenticated account needs `bigquery.jobs.list` at project level."
        )
    else:
        st.caption(f"ℹ️ Monthly usage unavailable: {usage['error']}")

    # --- Preview + run -------------------------------------------------------
    if not is_dml_mode:
        if st.button("👁️ Preview (25 rows)", key=f"{result_key}_preview"):
            with st.spinner("Running preview…"):
                try:
                    df_preview = run_preview(project, sql, limit=25)
                    st.dataframe(df_preview, use_container_width=True)
                except Exception as e:
                    st.error(f"Preview failed: {e}")

    st.markdown("---")
    if st.button("✅ Run full query", type="primary", key=f"{result_key}_run"):
        with st.spinner("Running query…"):
            try:
                df = run_query(project, sql)
                st.session_state[result_key] = df
                # Invalidate usage cache so the next gate shows updated consumption
                st.session_state.pop(f"monthly_usage_{project}", None)
                st.success(f"Done — {len(df):,} rows returned.")
            except Exception as e:
                st.error(f"Query failed: {e}")

    df_result = st.session_state.get(result_key)
    if df_result is not None:
        st.dataframe(df_result, use_container_width=True)
        _render_export_row(df_result, result_key, project)


def _render_export_row(df, key_prefix: str, project: str) -> None:
    st.subheader("Export")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download CSV",
            data=df_to_csv_bytes(df),
            file_name=f"{key_prefix}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        if st.button(
            "📊 Export to Google Sheets",
            use_container_width=True,
            key=f"{key_prefix}_sheets",
        ):
            with st.spinner("Creating Google Sheet…"):
                url = export_to_sheets(df, title=f"{key_prefix}")
            if url:
                st.markdown(f"[Open spreadsheet]({url})")


# ============================================================================
# 3. RUN
# ============================================================================

def run() -> None:
    st.set_page_config(
        page_title="Data Export",
        page_icon="🗄️",
        layout="wide",
    )

    st.title("Data export")
    st.caption("Export BigQuery experiment data for statistical analysis.")

    # --- GCP credentials + Google sign-in -------------------------------------
    st.divider()
    if not render_gcp_credentials_gate("data_export"):
        st.stop()

    # --- Connection ----------------------------------------------------------
    st.divider()
    project, dataset = render_connection_selectors()
    if not project or not dataset:
        st.stop()

    # --- Date range ----------------------------------------------------------
    st.divider()
    st.subheader("Date range")
    start_date, end_date = render_date_range()

    # --- Mode selector -------------------------------------------------------
    st.divider()
    mode_labels = [label for _, label in EXPORT_MODES]
    mode_keys   = [key   for key, _ in EXPORT_MODES]

    selected_label = st.selectbox(
        "Export mode",
        options=mode_labels,
        key="export_mode",
        help=(
            "Baseline: aggregate site metrics for sample size planning — no experiment variables needed. "
            "Binomial: aggregated conversion counts per variant for transaction rate and binary KPIs. "
            "Continuous: row-level data per user for revenue per visitor or per transaction analysis. "
            "Sequential: binomial export with optional persistence for sequential testing across runs. "
            "Interaction: classify users by variant combination across multiple simultaneous experiments."
        ),
    )
    mode = mode_keys[mode_labels.index(selected_label)]

    # --- Mode-specific inputs ------------------------------------------------
    st.divider()

    dispatch = {
        "baseline":    _render_baseline_inputs,
        "binomial":    _render_binomial_inputs,
        "continuous":  _render_continuous_inputs,
        "sequential":  _render_sequential_inputs,
        "interaction": _render_interaction_inputs,
    }

    params = dispatch[mode](project, dataset, start_date, end_date)

    if params is None:
        st.stop()

    # --- SQL preview ---------------------------------------------------------
    sql = _build_sql(mode, params)
    render_sql_viewer(sql, key=f"{mode}_sql")

    # --- Execution gate ------------------------------------------------------
    _render_execution_gate(project, dataset, sql, mode)

# ============================================================================
# 4. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run()
