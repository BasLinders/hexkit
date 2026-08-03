import streamlit as st

# Google's OAuth redirect ends the WebSocket session and wipes session_state,
# including admin_authenticated, so a bare admin check would block the page
# before the returning `code` can be exchanged. Let the callback through;
# everything past the credentials gate still requires admin_authenticated.
if not st.session_state.get("admin_authenticated") and "code" not in st.query_params:
    st.error("Access denied.")
    st.stop()

import json
import math
from datetime import date, datetime, timedelta
from typing import Literal, Optional, cast

import pandas as pd

from utility.bq_ui_components import (
    render_gcp_credentials_gate,
    render_connection_selectors,
    render_date_range,
    render_variant_inputs,
    render_execution_gate,
    render_combined_execution_gate,
    render_sql_viewer,
)
from utility.sql_builder import (
    BaselineParams,
    BinomialParams,
    ContinuousParams,
    binomial_shared_scan_flags,
    build_baseline,
    build_shared_scan_select,
    build_binomial_from_shared_scan,
    build_continuous_from_shared_scan,
    build_experiment_single_output_sql,
    build_experiment_shared_scan_temp_table_sql,
    build_experiment_session_output_sql,
)
from utility.automation_engine import (
    VariantData,
    TAILS,
    FIELD_LABELS,
    run_frequentist_analysis,
    run_bayesian_analysis,
    run_continuous_analysis,
    run_pretest_analysis,
    build_airtable_payload,
    apply_field_map,
    best_match_field,
)
from utility.airtable_client import (
    get_credentials, push_record, update_record, search_records, list_bases, list_tables,
)
from utility import gemini_client


STEPS = [
    "1. Fetch data",
    "2. Choose analysis",
    "3. Review results",
    "4. AI conclusion",
    "5. Send results",
]

PRETEST_KPI_OPTIONS = {
    "Transactions (purchases)": "purchase",
    "Add to cart": "add_to_cart",
}


def _monday_on_or_before(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _pretest_baseline_range(experiment_start: date, weeks_before: int) -> tuple[date, date]:
    """
    Full Monday-Sunday weeks immediately preceding the week the experiment
    starts in. If `experiment_start` isn't itself a Monday, the partial
    week before it is excluded entirely, since only full weeks are wanted.
    """
    baseline_end = _monday_on_or_before(experiment_start) - timedelta(days=1)  # preceding Sunday
    baseline_start = baseline_end - timedelta(days=7 * weeks_before - 1)  # N full weeks, Monday-aligned
    return baseline_start, baseline_end


_ID_FIELD_HINTS = ("experiment id", "experiment_id", "test id", "test_id", "id")


def _guess_id_field(fields: list[str]) -> Optional[str]:
    """Best-effort default for 'which field identifies existing records' —
    e.g. an Airtable autonumber field named 'Experiment ID'."""
    lowered = {f.lower(): f for f in fields}
    for hint in _ID_FIELD_HINTS:
        if hint in lowered:
            return lowered[hint]
    return None


_HYPOTHESIS_FIELD_HINTS = ("hypothesis",)


def _guess_hypothesis_field(fields: list[str]) -> Optional[str]:
    lowered = {f.lower(): f for f in fields}
    for hint in _HYPOTHESIS_FIELD_HINTS:
        if hint in lowered:
            return lowered[hint]
    return None


_CUSTOM_CODE_FIELD_HINTS = ("custom code", "code")


def _guess_custom_code_field(fields: list[str]) -> Optional[str]:
    lowered = {f.lower(): f for f in fields}
    for hint in _CUSTOM_CODE_FIELD_HINTS:
        if hint in lowered:
            return lowered[hint]
    return None


def _reset_automation_state():
    for key in list(st.session_state.keys()):
        if key.startswith("auto_") or key.startswith("autofetch_"):
            del st.session_state[key]


def _render_stepper(stage: int):
    cols = st.columns(len(STEPS))
    for i, col in enumerate(cols, start=1):
        with col:
            if i < stage:
                st.markdown(f"✅ {STEPS[i - 1]}")
            elif i == stage:
                st.markdown(f"**➡️ {STEPS[i - 1]}**")
            else:
                st.markdown(f"⬜ {STEPS[i - 1]}")


# ---------------------------------------------------------------------------
# Step 1 — Fetch data
# ---------------------------------------------------------------------------

def _render_stage_fetch():
    st.subheader("Step 1 — Fetch data from BigQuery")
    st.caption(
        "Same experiment data as Data Export's binomial/continuous modes, "
        "restricted to exactly one control (A) and one variation (B)."
    )

    # The OAuth redirect starts a fresh Streamlit session with a blank
    # session_state, which would otherwise look logged out of the admin gate
    # even though the user never left it. extra_state carries admin_authenticated
    # across that redirect — see render_gcp_credentials_gate.
    if not render_gcp_credentials_gate("automation", extra_state={"admin_authenticated": True}):
        return

    st.divider()
    project, dataset = render_connection_selectors()
    if not project or not dataset:
        return

    st.divider()
    st.subheader("Date range")
    start_date, end_date = render_date_range()

    st.divider()
    st.subheader("Pre-test baseline (optional)")
    st.caption(
        "Feeds the Pre-Test Analysis method in the next step — projects the "
        "MDE this experiment's traffic could detect, using site traffic from "
        "before the experiment started as the baseline."
    )
    want_pretest = st.checkbox(
        "Fetch pre-test baseline data",
        value=False,
        key="autofetch_want_pretest",
    )
    if want_pretest:
        pc1, pc2 = st.columns(2)
        with pc1:
            weeks_before = st.number_input(
                "Full weeks (Mon-Sun) of baseline data before the experiment start date",
                min_value=1, max_value=52, value=4, step=1,
                key="autofetch_pretest_weeks",
            )
        with pc2:
            kpi_choice = st.selectbox(
                "Conversion KPI to use as baseline",
                options=list(PRETEST_KPI_OPTIONS.keys()),
                key="autofetch_pretest_kpi",
            )

        use_page_filter = st.toggle(
            "Enable page filter",
            value=False,
            key="autofetch_pretest_use_filter",
            help=(
                "When enabled, only baseline users who visited a matching page "
                "are included. Useful for scoping the baseline to a specific "
                "section of the site — e.g. a product category or checkout flow."
            ),
        )
        page_filter_type = None
        page_filter_value = ""
        if use_page_filter:
            fc1, fc2 = st.columns([1, 2])
            with fc1:
                page_filter_type = cast(
                    Literal["regex", "contains"],
                    st.radio(
                        "Filter type",
                        options=["contains", "regex"],
                        format_func=lambda x: "URL contains" if x == "contains" else "Regex pattern",
                        horizontal=True,
                        key="autofetch_pretest_filter_type",
                    ),
                )
            with fc2:
                page_filter_value = st.text_input(
                    "Filter value",
                    placeholder=".html  |  /products/  |  \\.html$",
                    key="autofetch_pretest_filter_value",
                    help=(
                        "The string or pattern to match against page_location. "
                        "Regex uses BigQuery REGEXP_CONTAINS syntax. Matching is case-sensitive."
                    ),
                )
            if not page_filter_value:
                st.warning("Enter a filter value or disable the page filter.")

        try:
            experiment_start = datetime.strptime(start_date, "%Y-%m-%d").date()
        except ValueError:
            experiment_start = None

        if experiment_start is None:
            st.warning("Set a valid experiment start date above first.")
        else:
            baseline_start, baseline_end = _pretest_baseline_range(experiment_start, int(weeks_before))
            st.caption(
                f"Baseline period: **{baseline_start}** to **{baseline_end}** "
                f"({int(weeks_before)} full week(s), Monday-Sunday)."
            )

            baseline_params = BaselineParams(
                project=project,
                dataset=dataset,
                start_date=str(baseline_start),
                end_date=str(baseline_end),
                output_type="binomial",
                output_shape="aggregate",
                kpi_add_to_cart=(PRETEST_KPI_OPTIONS[kpi_choice] == "add_to_cart"),
                page_filter_type=page_filter_type if page_filter_value else None,
                page_filter_value=page_filter_value,
            )
            baseline_sql = build_baseline(baseline_params)
            render_sql_viewer(baseline_sql, key="auto_pretest_sql")
            render_execution_gate(
                project, baseline_sql, result_key="auto_pretest_result", allow_preview=True,
            )

    st.divider()
    st.subheader("Data to fetch")
    col1, col2 = st.columns(2)
    with col1:
        want_binomial = st.checkbox(
            "Binomial (conversion rate / AOV)",
            value=True,
            key="autofetch_want_binomial",
            help="Enables Frequentist and Bayesian analysis in the next step.",
        )
    with col2:
        want_continuous = st.checkbox(
            "Continuous (revenue per visitor)",
            value=False,
            key="autofetch_want_continuous",
            help="Enables Continuous Analysis in the next step.",
        )
    if not want_binomial and not want_continuous:
        st.warning("Select at least one data set — Binomial, Continuous, or both.")
        return
    if want_binomial and want_continuous:
        st.info(
            "Both selected — events_* is scanned once and shared between them, "
            "instead of scanning it twice.",
            icon="💡",
        )

    st.divider()
    param_key, match_strategy, exp_prefix, experiments = render_variant_inputs(
        project, dataset, start_date, end_date,
        key_prefix="autofetch",
        show_multi_experiment=False,
    )

    labels_present = {
        v.label for v in experiments[0].variants if v.string and v.string.strip()
    }
    if labels_present != {"A", "B"}:
        st.warning(
            "Automation currently supports exactly one control (A) and one "
            "variation (B). Assign exactly those two labels above to continue."
        )
        return

    match_strategy = cast(Literal["exact", "like"], match_strategy)

    binomial_params: Optional[BinomialParams] = None
    if want_binomial:
        binomial_params = BinomialParams(
            project=project,
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
            param_key=param_key,
            match_strategy=match_strategy,
            experiments=experiments,
            post_exposure_filter=True,
            kpi_transactions=True,
            kpi_aov=True,
            kpi_add_to_cart=False,
            kpi_ideal=False,
            kpi_device_split=False,
            kpi_login=False,
            kpi_create_account=False,
        )

    continuous_params: Optional[ContinuousParams] = None
    if want_continuous:
        continuous_params = ContinuousParams(
            project=project,
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
            param_key=param_key,
            match_strategy=match_strategy,
            experiments=experiments,
            device_filter="all",
            query_mode="all_users",  # RPV — non-buyers included, matches run_continuous_analysis's per-visitor assumption
            post_exposure_filter=True,
        )

    need_page_location, need_payment_type = binomial_shared_scan_flags(binomial_params)
    shared_scan_select = build_shared_scan_select(
        project, dataset, start_date, end_date, param_key,
        need_page_location, need_payment_type,
    )

    if binomial_params and continuous_params:
        create_temp_sql = build_experiment_shared_scan_temp_table_sql(shared_scan_select)
        binomial_sql   = build_experiment_session_output_sql(build_binomial_from_shared_scan(binomial_params))
        continuous_sql = build_experiment_session_output_sql(build_continuous_from_shared_scan(continuous_params))

        render_sql_viewer(
            f"{create_temp_sql}\n{binomial_sql}\n{continuous_sql}",
            key="auto_sql",
        )
        render_combined_execution_gate(
            project, dataset, shared_scan_select, create_temp_sql,
            {"binomial": binomial_sql, "continuous": continuous_sql},
            result_key_prefix="auto",
        )
    else:
        label = "binomial" if binomial_params else "continuous"
        chain = (
            build_binomial_from_shared_scan(binomial_params) if binomial_params
            else build_continuous_from_shared_scan(continuous_params)
        )
        sql = build_experiment_single_output_sql(shared_scan_select, chain)
        render_sql_viewer(sql, key="auto_sql")
        render_execution_gate(project, sql, result_key=f"auto_{label}_result", allow_preview=True)

    df_binomial = st.session_state.get("auto_binomial_result")
    df_continuous = st.session_state.get("auto_continuous_result")
    if df_binomial is None and df_continuous is None:
        return

    if df_binomial is not None and not {"A", "B"}.issubset(set(df_binomial["experience_variant_label"])):
        st.error("Binomial result is missing rows for control (A) and/or variation (B).")
        return
    if df_continuous is not None and not {"A", "B"}.issubset(set(df_continuous["experience_variant_label"])):
        st.error("Continuous result is missing rows for control (A) and/or variation (B).")
        return

    st.divider()
    if st.button("Continue to choose analysis method(s) →", type="primary"):
        st.session_state["auto_df_binomial"] = df_binomial
        st.session_state["auto_df_continuous"] = df_continuous
        st.session_state["auto_exp_prefix"] = exp_prefix

        df_pretest = st.session_state.get("auto_pretest_result")
        st.session_state["auto_df_pretest"] = df_pretest
        if df_pretest is not None and not df_pretest.empty:
            weeks = int(st.session_state.get("autofetch_pretest_weeks", 1))
            kpi_choice = st.session_state.get("autofetch_pretest_kpi", "Transactions (purchases)")
            row = df_pretest.iloc[0]
            conversions_col = (
                "total_add_to_cart_conversions"
                if PRETEST_KPI_OPTIONS[kpi_choice] == "add_to_cart"
                else "total_conversions"
            )
            st.session_state["auto_pretest_meta"] = {
                "weeks": weeks,
                "kpi_label": kpi_choice,
                "total_visitors": float(row["total_visitors"]),
                "total_conversions": float(row[conversions_col]),
            }
        else:
            st.session_state["auto_pretest_meta"] = None

        st.session_state["auto_stage"] = 2
        st.rerun()


# ---------------------------------------------------------------------------
# Step 2 — Choose analysis method(s) and settings
# ---------------------------------------------------------------------------

def _variant_from_row(row, label: str) -> VariantData:
    aov = row["average_order_value"]
    return VariantData(
        label=label,
        visitors=int(row["visitors"]),
        conversions=int(row["users_with_transaction"]),
        aov=float(aov) if pd.notna(aov) else 0.0,
    )


def _render_stage_configure():
    df_binomial = st.session_state.get("auto_df_binomial")
    df_continuous = st.session_state.get("auto_df_continuous")
    pretest_meta = st.session_state.get("auto_pretest_meta")
    if df_binomial is None and df_continuous is None:
        st.session_state["auto_stage"] = 1
        st.rerun()
        return

    st.subheader("Step 2 — Choose analysis method(s)")

    control = variation = None
    if df_binomial is not None:
        control = _variant_from_row(df_binomial[df_binomial["experience_variant_label"] == "A"].iloc[0], "Control")
        variation = _variant_from_row(df_binomial[df_binomial["experience_variant_label"] == "B"].iloc[0], "Variation")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Control — visitors", f"{control.visitors:,}")
            rate = control.conversions / control.visitors if control.visitors else 0.0
            st.metric("Control — conversions", f"{control.conversions:,}", help=f"Rate: {rate:.2%}")
            st.metric("Control — AOV", f"€{control.aov:,.2f}")
        with col2:
            st.metric("Variation — visitors", f"{variation.visitors:,}")
            rate = variation.conversions / variation.visitors if variation.visitors else 0.0
            st.metric("Variation — conversions", f"{variation.conversions:,}", help=f"Rate: {rate:.2%}")
            st.metric("Variation — AOV", f"€{variation.aov:,.2f}")

    if df_continuous is not None:
        st.markdown("**Continuous data — revenue per visitor**")
        cont_summary = (
            df_continuous.assign(purchase_revenue=pd.to_numeric(df_continuous["purchase_revenue"], errors="coerce").fillna(0.0))
            .groupby("experience_variant_label")["purchase_revenue"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "revenue per visitor", "count": "visitors"})
        )
        st.dataframe(cont_summary, use_container_width=True)

    st.divider()
    st.markdown("**Analysis method(s)**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        use_frequentist = st.checkbox(
            "Frequentist Analysis", value=df_binomial is not None,
            disabled=df_binomial is None, key="auto_use_frequentist",
            help=None if df_binomial is not None else "Requires binomial data — fetch it in step 1.",
        )
    with c2:
        use_bayesian = st.checkbox(
            "Bayesian Analysis", value=df_binomial is not None,
            disabled=df_binomial is None, key="auto_use_bayesian",
            help=None if df_binomial is not None else "Requires binomial data — fetch it in step 1.",
        )
    with c3:
        use_continuous = st.checkbox(
            "Continuous Analysis", value=df_continuous is not None,
            disabled=df_continuous is None, key="auto_use_continuous",
            help=None if df_continuous is not None else "Requires continuous data — fetch it in step 1.",
        )
    with c4:
        use_pretest = st.checkbox(
            "Pre-Test Analysis", value=pretest_meta is not None,
            disabled=pretest_meta is None, key="auto_use_pretest",
            help=None if pretest_meta is not None else "Requires pre-test baseline data — fetch it in step 1.",
        )

    # Streamlit persists a checkbox's checked state across reruns even while
    # disabled=True, so a method checked before a re-fetch dropped its data
    # would otherwise stay "on" here despite being greyed out in the UI.
    use_frequentist = use_frequentist and df_binomial is not None
    use_bayesian = use_bayesian and df_binomial is not None
    use_continuous = use_continuous and df_continuous is not None
    use_pretest = use_pretest and pretest_meta is not None

    if not use_frequentist and not use_bayesian and not use_continuous and not use_pretest:
        st.warning("Select at least one analysis method to continue.")
        return

    start = st.session_state.get("start_date")
    end = st.session_state.get("end_date")
    runtime_days = max((end - start).days + 1, 1) if start and end else 1
    if control is not None and variation is not None:
        total_visitors = control.visitors + variation.visitors
    else:
        # Per-visitor rows (RPV query mode) — one row per exposed user.
        total_visitors = len(df_continuous)
    daily_visitors = total_visitors / runtime_days

    st.divider()
    if use_frequentist or use_continuous or use_pretest:
        with st.expander("Significance settings", expanded=True):
            fc1, fc2 = st.columns(2)
            with fc1:
                confidence_level = st.slider(
                    "Confidence level", min_value=0.80, max_value=0.99,
                    value=0.95, step=0.01, key="auto_confidence_level",
                )
            with fc2:
                tail = st.radio(
                    "Tail", options=TAILS, index=0, horizontal=True, key="auto_tail",
                )
    else:
        confidence_level, tail = 0.95, "Two-sided"

    if use_bayesian:
        with st.expander("Bayesian settings", expanded=True):
            n_samples = st.select_slider(
                "Monte Carlo samples",
                options=[10_000, 50_000, 100_000, 250_000],
                value=100_000,
                key="auto_n_samples",
            )
    else:
        n_samples = 100_000

    if use_continuous:
        with st.expander("Continuous settings", expanded=True):
            continuous_mode = st.radio(
                "Metric definition",
                options=["rpv", "rpt"],
                format_func=lambda m: (
                    "Revenue per visitor (RPV)" if m == "rpv" else "Revenue per transaction (RPT)"
                ),
                horizontal=True,
                key="auto_continuous_mode",
                help=(
                    "RPV: every exposed visitor counts, non-buyers as €0 — captures both "
                    "conversion-rate and spend effects together. RPT: only buyers count "
                    "(zero-revenue rows stripped before analysis) — isolates the order-value "
                    "effect alone. Both use the same fetched data; only the calculation differs."
                ),
            )
    else:
        continuous_mode = "rpv"

    if use_pretest:
        with st.expander("Pre-test settings", expanded=True):
            trust_pct = st.slider(
                "Power / trustworthiness target (%)", min_value=50, max_value=99,
                value=80, step=1, key="auto_pretest_trust",
            )
            weekly_visitors = pretest_meta["total_visitors"] / pretest_meta["weeks"]
            weekly_conversions = pretest_meta["total_conversions"] / pretest_meta["weeks"]
            st.caption(
                f"Baseline: {pretest_meta['kpi_label']} over {pretest_meta['weeks']} week(s) → "
                f"{weekly_visitors:,.0f} visitors/week, {weekly_conversions:,.1f} conversions/week."
            )
    else:
        trust_pct = 80

    with st.expander("Revenue projection", expanded=True):
        rc1, rc2 = st.columns(2)
        with rc1:
            projection_days = st.number_input(
                "Projection period (days)", min_value=1, value=183, key="auto_projection_days",
            )
        with rc2:
            aov_cv = st.slider(
                "AOV variability (CV)", min_value=0.0, max_value=1.5,
                value=0.0, step=0.05, key="auto_aov_cv",
                help="0 treats AOV as a known constant. Above 0 propagates AOV sampling uncertainty (Frequentist only).",
            )
        st.caption(f"Test runtime: {runtime_days} day(s) · Daily visitors: {daily_visitors:,.0f}")

    active_methods = [
        m for m, using in (
            ("Frequentist", use_frequentist),
            ("Bayesian", use_bayesian),
            ("Continuous", use_continuous),
        ) if using
    ]
    revenue_source = active_methods[0].lower() if active_methods else "frequentist"
    if len(active_methods) > 1:
        # Frequentist/Bayesian default to checked whenever Binomial data was
        # fetched (Streamlit's checkbox `value=` only applies once, on first
        # render — see the checkboxes above), so this radio's options can
        # silently change shape without the user touching either checkbox.
        # If the active set changed since the last run, drop the stale
        # session_state pick instead of carrying over a choice made for a
        # different combination of methods (e.g. "Bayesian" picked back when
        # Continuous wasn't active yet, silently kept once it is).
        if st.session_state.get("auto_revenue_source_methods") != active_methods:
            st.session_state.pop("auto_revenue_source_radio", None)
            st.session_state["auto_revenue_source_methods"] = active_methods

        choice = st.radio(
            "Use for the shared 'effect on revenue' field:",
            options=active_methods,
            horizontal=True,
            key="auto_revenue_source_radio",
        )
        revenue_source = choice.lower()

    st.info(f"💰 'Effect on revenue' will be sent from: **{revenue_source.capitalize()}**.")

    st.divider()
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back to data fetch"):
            st.session_state["auto_stage"] = 1
            st.rerun()
    with col_next:
        if st.button("Run analysis →", type="primary"):
            results = {}
            if use_frequentist:
                results["frequentist"] = run_frequentist_analysis(
                    control, variation,
                    confidence_level=confidence_level,
                    tail=tail,
                    daily_visitors=daily_visitors,
                    projection_days=int(projection_days),
                    aov_cv=aov_cv,
                )
            if use_bayesian:
                results["bayesian"] = run_bayesian_analysis(
                    control, variation,
                    runtime_days=runtime_days,
                    projection_days=int(projection_days),
                    n_samples=int(n_samples),
                )
            if use_continuous:
                results["continuous"] = run_continuous_analysis(
                    df_continuous,
                    control_label="A",
                    variation_label="B",
                    daily_visitors=daily_visitors,
                    projection_days=int(projection_days),
                    confidence_level=confidence_level,
                    tail=tail,
                    mode=continuous_mode,
                )
            if use_pretest:
                results["pretest"] = run_pretest_analysis(
                    weekly_visitors=pretest_meta["total_visitors"] / pretest_meta["weeks"],
                    weekly_conversions=pretest_meta["total_conversions"] / pretest_meta["weeks"],
                    weeks_used=pretest_meta["weeks"],
                    kpi_label=pretest_meta["kpi_label"],
                    confidence_level=confidence_level,
                    tail=tail,
                    trust_pct=trust_pct,
                )
            st.session_state["auto_control"] = control
            st.session_state["auto_variation"] = variation
            st.session_state["auto_results"] = results
            st.session_state["auto_revenue_source"] = revenue_source
            st.session_state["auto_stage"] = 3
            st.rerun()


# ---------------------------------------------------------------------------
# Step 3 — Review results
# ---------------------------------------------------------------------------

def _fmt_money(x: float) -> str:
    return "Unbounded" if math.isinf(x) else f"€{x:,.0f}"


def _render_stage_results():
    results = st.session_state.get("auto_results")
    control = st.session_state.get("auto_control")
    variation = st.session_state.get("auto_variation")
    if not results:
        st.session_state["auto_stage"] = 2
        st.rerun()
        return

    st.subheader("Step 3 — Review results")

    freq = results.get("frequentist")
    bayes = results.get("bayesian")
    cont = results.get("continuous")
    pretest = results.get("pretest")

    # Which method's effect_on_revenue actually feeds the shared Airtable
    # field — decided back in Step 2 (see the radio there), surfaced again
    # here per-method so it's never ambiguous which number is really used,
    # even if the three methods' monetary estimates disagree substantially.
    revenue_source = st.session_state.get("auto_revenue_source", "frequentist")

    def _revenue_tag(method: str) -> str:
        return " 💰 *(used for shared 'effect on revenue')*" if revenue_source == method else ""

    if freq:
        st.markdown(f"### Frequentist{_revenue_tag('frequentist')}")
        c1, c2, c3 = st.columns(3)
        c1.metric("P-value", f"{freq['p_value']:.4f}")
        c2.metric("Significant?", "Yes" if freq["is_significant"] else "No")
        c3.metric("Uplift", f"{freq['uplift']:+.2%}")
        ci_low, ci_high = freq["effect_on_revenue_ci"]
        st.metric(
            f"Effect on revenue ({freq['projection_days']}d)",
            _fmt_money(freq["effect_on_revenue"]),
            help=f"CI: {_fmt_money(ci_low)} to {_fmt_money(ci_high)}",
        )
        st.caption(freq["conclusion"])
        st.divider()

    if bayes:
        st.markdown(f"### Bayesian{_revenue_tag('bayesian')}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Probability to beat control", f"{bayes['probability_pct']:.1f}%")
        c2.metric("Probability to be best", f"{bayes['prob_being_best']:.1%}")
        c3.metric(
            f"Effect on revenue ({bayes['projection_days']}d)",
            _fmt_money(bayes["effect_on_revenue"]),
            help=(
                f"Expected uplift {_fmt_money(bayes['expected_revenue_uplift'])} / "
                f"expected risk {_fmt_money(bayes['expected_revenue_risk'])}"
            ),
        )
        st.caption(bayes["conclusion"])
        st.divider()

    if cont:
        mode_label = "RPV" if cont.get("mode", "rpv") == "rpv" else "RPT"
        st.markdown(f"### Continuous ({mode_label}){_revenue_tag('continuous')}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Test used", cont["test_name"])
        c2.metric("P-value", f"{cont['p_value']:.4f}")
        c3.metric("Significant?", "Yes" if cont["is_significant"] else "No")
        ci_low, ci_high = cont["effect_on_revenue_ci"]
        st.metric(
            f"Effect on revenue ({cont['projection_days']}d)",
            _fmt_money(cont["effect_on_revenue"]),
            help=f"CI: {_fmt_money(ci_low)} to {_fmt_money(ci_high)}",
        )
        st.caption(cont["conclusion"])
        st.divider()

    if pretest:
        st.markdown("### Pre-Test Analysis")
        st.caption(
            f"Baseline: {pretest['kpi']} · {pretest['weeks_used']} week(s) · "
            f"{pretest['weekly_visitors']:,.0f} visitors/week, "
            f"{pretest['weekly_conversions']:,.1f} conversions/week."
        )
        mde_df = pd.DataFrame(pretest["table"])
        st.dataframe(mde_df, use_container_width=True)
        st.caption(pretest["conclusion"])
        st.divider()

    start_dt = st.session_state.get("start_date")
    end_dt = st.session_state.get("end_date")
    payload = build_airtable_payload(
        control, variation, freq, bayes, cont, revenue_source=revenue_source,
        start_date=str(start_dt) if start_dt else None,
        end_date=str(end_dt) if end_dt else None,
        pretest_result=pretest,
    )
    st.session_state["auto_payload"] = payload

    st.markdown("**Result summary**")
    st.caption(
        f"💰 'effect_on_revenue' below is sourced from **{revenue_source.capitalize()}** "
        "— change the source in Step 2 if that's not what you expected. "
        "Airtable field names are chosen in the next step, based on the base/table you send to."
    )
    st.json(payload)

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back to configuration"):
            st.session_state["auto_stage"] = 2
            st.rerun()
    with col_next:
        if st.button("Continue to AI conclusion →", type="primary"):
            st.session_state["auto_stage"] = 4
            st.rerun()


# ---------------------------------------------------------------------------
# Step 4 — AI conclusion (optional)
# ---------------------------------------------------------------------------

def _lookup_experiment_record() -> tuple[list[str], Optional[dict], str]:
    """
    Searches the configured default Airtable base/table (AIRTABLE_BASE_ID/
    AIRTABLE_TABLE_NAME) for the record matching this experiment (Step 1's
    exp_prefix, via the same id-field guess the Send step uses). Shared by
    the Hypothesis gate (required, blocks generation) and the Custom Code
    lookup (optional context for Gemini) so both reuse one round-trip.
    Returns (table_field_names, matched_record_fields_or_None, message).
    """
    exp_prefix = st.session_state.get("auto_exp_prefix")
    creds = get_credentials()
    api_key, base_id, table_name = creds["api_key"], creds["base_id"], creds["table_name"]

    if not (api_key and base_id and table_name):
        return [], None, (
            "Airtable isn't fully configured (AIRTABLE_API_KEY/BASE_ID/TABLE_NAME) — "
            "can't look up the experiment record."
        )
    if not exp_prefix:
        return [], None, "No experiment ID from Step 1 to look up."

    tables_result = list_tables(api_key, base_id)
    if not tables_result["ok"]:
        return [], None, f"Couldn't load Airtable tables: {tables_result['error']}"

    table = next(
        (t for t in tables_result["tables"] if table_name in (t["id"], t["name"])),
        None,
    )
    if table is None:
        return [], None, f"Configured table '{table_name}' wasn't found in the base."

    id_field = _guess_id_field(table["fields"])
    if not id_field:
        return table["fields"], None, "Couldn't find an ID field on the configured table."

    search_result = search_records(base_id, table["id"], api_key, id_field, exp_prefix)
    if not search_result["ok"]:
        return table["fields"], None, f"Airtable lookup failed: {search_result['error']}"

    matches = search_result["records"]
    if not matches:
        return table["fields"], None, f"No Airtable record found yet for experiment '{exp_prefix}'."

    # Prefer a match that already has a Hypothesis filled in, if more than one
    # record matched the search term — otherwise just take the first.
    hypothesis_field = _guess_hypothesis_field(table["fields"])
    if hypothesis_field:
        with_hypothesis = next(
            (r for r in matches if str(r["fields"].get(hypothesis_field, "")).strip()), None,
        )
        if with_hypothesis:
            return table["fields"], with_hypothesis["fields"], f"Matched Airtable record for '{exp_prefix}'."

    return table["fields"], matches[0]["fields"], f"Matched Airtable record for '{exp_prefix}'."


def _render_stage_ai():
    payload = st.session_state.get("auto_payload")
    results = st.session_state.get("auto_results") or {}
    if payload is None:
        st.session_state["auto_stage"] = 3
        st.rerun()
        return

    st.subheader("Step 4 — AI conclusion (optional)")
    st.caption(
        "Sends the result payload to Gemini for a short written interpretation. "
        "Entirely optional — skip straight to sending if you don't want one."
    )

    table_fields, record_fields, lookup_message = _lookup_experiment_record()

    hypothesis_field = _guess_hypothesis_field(table_fields) if table_fields else None
    hypothesis_ok = bool(
        record_fields is not None and hypothesis_field
        and str(record_fields.get(hypothesis_field, "")).strip()
    )

    custom_code_field = _guess_custom_code_field(table_fields) if table_fields else None
    custom_code = (
        str(record_fields.get(custom_code_field, "")).strip()
        if record_fields is not None and custom_code_field else ""
    )

    ai_input = {
        "payload": payload,
        "conclusions": {
            method: result["conclusion"]
            for method, result in results.items()
            if result.get("conclusion")
        },
    }
    if custom_code:
        # Optional context, not gated on — helps Gemini interpret what was
        # actually being tested (e.g. which element/variant the code touches).
        ai_input["custom_code"] = custom_code

    with st.expander("Data that would be sent to Gemini", expanded=False):
        st.json(ai_input)

    if not gemini_client.is_configured():
        st.info(
            "No Gemini API key configured — set `GEMINI_API_KEY` in Streamlit secrets "
            "to enable this step. You can still continue without an AI conclusion."
        )
    else:
        if not hypothesis_ok:
            st.warning(
                f"{lookup_message} Document the Hypothesis in Airtable first — "
                "an AI conclusion can't be generated without it."
            )
        generate_clicked = st.button(
            "🤖 Generate AI conclusion", type="primary",
            disabled=not hypothesis_ok, key="auto_generate_ai_btn",
        )
        if generate_clicked:
            with st.spinner("Asking Gemini…"):
                result = gemini_client.generate_conclusion(ai_input)
            if result["ok"]:
                st.session_state["auto_ai_conclusion"] = result["text"]
            else:
                st.error(f"Gemini request failed: {result['error']}")

    ai_conclusion = st.session_state.get("auto_ai_conclusion")
    if ai_conclusion:
        st.markdown("**Conclusion**")
        st.markdown(ai_conclusion)
        include = st.checkbox(
            "Include this conclusion in the Airtable payload", value=True, key="auto_ai_include",
        )
        payload = dict(payload)
        if include:
            payload["ai_conclusion"] = ai_conclusion
        else:
            payload.pop("ai_conclusion", None)
        st.session_state["auto_payload"] = payload

    st.divider()
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back to results"):
            st.session_state["auto_stage"] = 3
            st.rerun()
    with col_next:
        label = "Continue to send →" if ai_conclusion else "Skip — continue to send →"
        if st.button(label, type="primary"):
            st.session_state["auto_stage"] = 5
            st.rerun()


# ---------------------------------------------------------------------------
# Step 5 — Send results
# ---------------------------------------------------------------------------

def _render_json_download(payload: dict):
    st.download_button(
        "⬇️ Download raw result as JSON",
        data=json.dumps(payload, indent=2).encode(),
        file_name="automation_payload.json",
        mime="application/json",
    )


def _render_back_to_results_button():
    if st.button("← Back to AI conclusion"):
        st.session_state["auto_stage"] = 4
        st.rerun()


def _render_stage_send():
    payload = st.session_state.get("auto_payload")
    if payload is None:
        st.session_state["auto_stage"] = 4
        st.rerun()
        return

    st.subheader("Step 5 — Send results")

    st.markdown("#### Airtable")
    creds = get_credentials()
    api_key = creds["api_key"]

    if not api_key:
        st.warning(
            "No Airtable token is configured — set `AIRTABLE_API_KEY` in Streamlit secrets "
            "(a single personal access token with the `schema.bases:read` scope, granted "
            "access to every base/workspace results should be sent to). It's an operational "
            "secret, not something entered here."
        )
        st.divider()
        _render_json_download(payload)
        _render_back_to_results_button()
        return

    # --- Base discovery --------------------------------------------------
    if st.session_state.get("airtable_bases_for_key") != api_key:
        with st.spinner("Loading Airtable bases…"):
            result = list_bases(api_key)
        if not result["ok"]:
            st.error(f"Couldn't list Airtable bases: {result['error']}")
            st.divider()
            _render_json_download(payload)
            _render_back_to_results_button()
            return
        st.session_state["airtable_bases_cache"] = result["bases"]
        st.session_state["airtable_bases_for_key"] = api_key

    bases: dict = st.session_state["airtable_bases_cache"]
    if not bases:
        st.warning("This token can't see any bases — check its access under Airtable's token settings.")
        st.divider()
        _render_json_download(payload)
        _render_back_to_results_button()
        return

    base_ids = list(bases.keys())
    prior_bases = st.session_state.get("airtable_base_ids") or (
        [creds["base_id"]] if creds["base_id"] in base_ids else []
    )
    default_bases = [b for b in prior_bases if b in base_ids] or base_ids[:1]
    selected_base_ids = st.multiselect(
        "Send to (base(s))",
        options=base_ids,
        default=default_bases,
        format_func=lambda bid: bases.get(bid, bid),
        key="airtable_base_ids",
        help="Pick every base that should receive this result. Each base gets its "
             "own table selection and field mapping below.",
    )
    if not selected_base_ids:
        st.info("Select at least one base above to continue.")
        st.divider()
        _render_json_download(payload)
        _render_back_to_results_button()
        return

    # --- Table discovery + selection per base -------------------------------
    tables_by_base: dict[str, list] = {}
    for base_id in selected_base_ids:
        tables_cache_key = f"airtable_tables_{base_id}"
        if tables_cache_key not in st.session_state:
            with st.spinner(f"Loading tables for {bases.get(base_id, base_id)}…"):
                result = list_tables(api_key, base_id)
            if not result["ok"]:
                st.error(f"Couldn't list tables in {bases.get(base_id, base_id)}: {result['error']}")
                st.divider()
                _render_json_download(payload)
                _render_back_to_results_button()
                return
            st.session_state[tables_cache_key] = result["tables"]
        tables_by_base[base_id] = st.session_state[tables_cache_key]

    st.divider()
    st.markdown("#### Tables, record lookup, and field mapping")
    st.caption(
        "Pick tables per base. For each table, either find an existing record "
        "to append this result to, or create a new one. Field mapping is "
        "remembered per base/table, so you only need to set it up once."
    )

    # One entry per (base, table) the user wants to send to, across every
    # selected base — each carries its own create/update mode and fields.
    send_plan: list[dict] = []
    for base_id, base_tab in zip(selected_base_ids, st.tabs([bases.get(b, b) for b in selected_base_ids])):
        with base_tab:
            tables = tables_by_base[base_id]
            if not tables:
                st.warning("This base has no tables.")
                continue

            table_names = [t["name"] for t in tables]
            names_key = f"airtable_table_names_{base_id}"
            stored_selection = st.session_state.get(names_key, [])
            experiment_tables = [n for n in table_names if n.lower() == "experiments"] or [
                n for n in table_names if "experiments" in n.lower()
            ]
            default_selection = (
                [n for n in stored_selection if n in table_names]
                or experiment_tables
                or table_names[:1]
            )
            selected_names = st.multiselect(
                "Tables — send this result to each one selected",
                options=table_names,
                default=default_selection,
                key=names_key,
            )
            selected_tables = [t for t in tables if t["name"] in selected_names]

            for tab, table in zip(st.tabs(selected_names), selected_tables):
                with tab:
                    lookup_key = f"airtable_lookup_{base_id}_{table['id']}"

                    st.markdown("**Existing record**")
                    id_field_options = ["— Always create new —"] + table["fields"]
                    default_id_field = _guess_id_field(table["fields"])
                    default_idx = (
                        id_field_options.index(default_id_field)
                        if default_id_field in id_field_options else 0
                    )
                    id_field = st.selectbox(
                        "Field to search on (e.g. an experiment ID Airtable generated)",
                        options=id_field_options, index=default_idx,
                        key=f"{lookup_key}_idfield",
                    )

                    mode = "create"
                    record_id: Optional[str] = None
                    if id_field != "— Always create new —":
                        search_col, btn_col = st.columns([3, 1])
                        with search_col:
                            search_term = st.text_input(
                                "Search for existing record (matches part of the ID)",
                                key=f"{lookup_key}_term",
                            )
                        with btn_col:
                            st.write("")
                            search_clicked = st.button("🔍 Search", key=f"{lookup_key}_btn")

                        if search_clicked and search_term.strip():
                            with st.spinner("Searching…"):
                                result = search_records(
                                    base_id, table["id"], api_key, id_field, search_term.strip(),
                                )
                            if result["ok"]:
                                st.session_state[f"{lookup_key}_results"] = result["records"]
                            else:
                                st.error(f"Search failed: {result['error']}")

                        matches = st.session_state.get(f"{lookup_key}_results", [])
                        match_options = ["+ Create new record"] + [
                            f"{r['fields'].get(id_field, '(blank)')}  ·  …{r['id'][-6:]}"
                            for r in matches
                        ]
                        choice = st.selectbox(
                            "Record to update", options=match_options,
                            key=f"{lookup_key}_choice",
                            help="Pick a match to append this result to it, or keep "
                                 "'Create new record' to add a new row instead.",
                        )
                        if choice != "+ Create new record":
                            record_id = matches[match_options.index(choice) - 1]["id"]
                            mode = "update"

                    description = ""
                    if mode == "create":
                        description = st.text_input(
                            "Short description for this new record",
                            key=f"{lookup_key}_description",
                            help="Required when creating a new record — shows up as a "
                                 "mappable field below, same as the computed results.",
                        )
                        if not description.strip():
                            st.warning(
                                "Enter a short description (or search for an existing "
                                "record above) to enable sending for this table."
                            )

                    table_payload = dict(payload)
                    if mode == "create" and description.strip():
                        table_payload["description"] = description.strip()

                    st.markdown("**Field mapping**")
                    field_options = ["— Skip —"] + table["fields"]
                    mapping_key = f"airtable_fieldmap_{base_id}_{table['id']}"
                    stored_mapping: dict = st.session_state.get(mapping_key, {})

                    field_map: dict = {}
                    for key in table_payload:
                        label = FIELD_LABELS.get(key, key)
                        default = stored_mapping.get(key) or best_match_field(key, table["fields"])
                        field_default_idx = field_options.index(default) if default in field_options else 0
                        chosen = st.selectbox(
                            label, options=field_options, index=field_default_idx,
                            key=f"{mapping_key}_{key}",
                        )
                        if chosen != "— Skip —":
                            field_map[key] = chosen

                    st.session_state[mapping_key] = field_map

                    if mode == "update" or description.strip():
                        send_plan.append({
                            "base_id": base_id,
                            "table": table,
                            "mode": mode,
                            "record_id": record_id,
                            "fields": apply_field_map(table_payload, field_map),
                        })

    if not send_plan:
        st.info(
            "Select at least one table above, and either provide a description "
            "(new record) or pick an existing record to update, to continue."
        )
        st.divider()
        _render_json_download(payload)
        _render_back_to_results_button()
        return

    ready_plan = [p for p in send_plan if p["fields"]]
    skipped_plan = [p for p in send_plan if not p["fields"]]

    def _label(p: dict) -> str:
        base_name = bases.get(p["base_id"], p["base_id"])
        verb = "update existing" if p["mode"] == "update" else "create new"
        return f"{base_name} → {p['table']['name']} ({verb})"

    with st.expander("Preview what will be sent", expanded=False):
        st.json({_label(p): p["fields"] for p in send_plan})

    if st.button("🚀 Send to Airtable", type="primary", disabled=not ready_plan):
        with st.spinner(f"Sending to {len(ready_plan)} table(s) across {len(selected_base_ids)} base(s)…"):
            send_results = [
                (
                    _label(p),
                    update_record(p["base_id"], p["table"]["id"], api_key, p["record_id"], p["fields"])
                    if p["mode"] == "update"
                    else push_record(p["base_id"], p["table"]["id"], api_key, p["fields"]),
                )
                for p in ready_plan
            ]
        for name, result in send_results:
            if result["ok"]:
                verb = "updated" if "update existing" in name else "created"
                st.success(f"{name}: record {verb} ({result['record_id']})")
            else:
                st.error(f"{name}: {result['error']}")
        if skipped_plan:
            skipped_names = ", ".join(_label(p) for p in skipped_plan)
            st.caption(f"Skipped (no fields mapped): {skipped_names}")

    st.divider()
    st.markdown("#### Other destinations")
    st.caption("More destinations (Slack, HubSpot, …) can be wired in here later.")

    st.divider()
    _render_json_download(payload)
    _render_back_to_results_button()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    st.set_page_config(page_title="Automation", page_icon="⚙️", layout="wide")
    st.title("Automation")
    st.caption(
        "Fetch experiment data from BigQuery, run it through the analysis engine, "
        "and push results to Airtable."
    )

    stage = st.session_state.get("auto_stage", 1)
    _render_stepper(stage)
    if st.button("↺ Start over"):
        _reset_automation_state()
        st.rerun()
    st.divider()

    if stage == 1:
        _render_stage_fetch()
    elif stage == 2:
        _render_stage_configure()
    elif stage == 3:
        _render_stage_results()
    elif stage == 4:
        _render_stage_ai()
    elif stage == 5:
        _render_stage_send()


if __name__ == "__main__":
    run()
