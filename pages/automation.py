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
    BinomialParams,
    ContinuousParams,
    binomial_shared_scan_flags,
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
    run_frequentist_analysis,
    run_bayesian_analysis,
    run_continuous_analysis,
    build_airtable_payload,
)
from utility.airtable_client import get_credentials, push_record


STEPS = ["1. Fetch data", "2. Choose analysis", "3. Review results", "4. Send results"]


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
        st.checkbox(
            "Pre-Test Analysis", value=False, disabled=True,
            key="auto_use_pretest", help="Coming soon.",
        )

    # Streamlit persists a checkbox's checked state across reruns even while
    # disabled=True, so a method checked before a re-fetch dropped its data
    # would otherwise stay "on" here despite being greyed out in the UI.
    use_frequentist = use_frequentist and df_binomial is not None
    use_bayesian = use_bayesian and df_binomial is not None
    use_continuous = use_continuous and df_continuous is not None

    if not use_frequentist and not use_bayesian and not use_continuous:
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
    if use_frequentist or use_continuous:
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
    revenue_source = active_methods[0].lower()
    if len(active_methods) > 1:
        choice = st.radio(
            "Use for the shared 'effect on revenue' field:",
            options=active_methods,
            horizontal=True,
            key="auto_revenue_source_radio",
        )
        revenue_source = choice.lower()

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

    if freq:
        st.markdown("### Frequentist")
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
        st.markdown("### Bayesian")
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
        st.markdown("### Continuous")
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

    revenue_source = st.session_state.get("auto_revenue_source", "frequentist")
    payload = build_airtable_payload(control, variation, freq, bayes, cont, revenue_source=revenue_source)
    st.session_state["auto_payload"] = payload

    st.markdown("**Airtable payload preview**")
    st.caption("Field names are provisional — see `AIRTABLE_FIELD_MAP` in utility/automation_engine.py.")
    st.json(payload)

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back to configuration"):
            st.session_state["auto_stage"] = 2
            st.rerun()
    with col_next:
        if st.button("Continue to send →", type="primary"):
            st.session_state["auto_stage"] = 4
            st.rerun()


# ---------------------------------------------------------------------------
# Step 4 — Send results
# ---------------------------------------------------------------------------

def _render_stage_send():
    payload = st.session_state.get("auto_payload")
    if payload is None:
        st.session_state["auto_stage"] = 3
        st.rerun()
        return

    st.subheader("Step 4 — Send results")

    st.markdown("#### Airtable")
    creds = get_credentials()
    col1, col2 = st.columns(2)
    with col1:
        base_id = st.text_input("Base ID", value=creds["base_id"], key="airtable_base_id")
    with col2:
        table_name = st.text_input("Table name", value=creds["table_name"], key="airtable_table_name")
    api_key = st.text_input(
        "API key (personal access token)", value=creds["api_key"],
        type="password", key="airtable_api_key",
    )
    st.caption(
        "No Airtable token is configured yet — set `AIRTABLE_API_KEY`, `AIRTABLE_BASE_ID` "
        "and `AIRTABLE_TABLE_NAME` in Streamlit secrets to prefill these, or paste them "
        "here for a one-off send (kept in this session only, never written to disk)."
    )

    if st.button(
        "🚀 Send to Airtable", type="primary",
        disabled=not (base_id and table_name and api_key),
    ):
        with st.spinner("Sending to Airtable…"):
            result = push_record(base_id, table_name, api_key, payload)
        if result["ok"]:
            st.success(f"Record created: {result['record_id']}")
        else:
            st.error(f"Airtable request failed: {result['error']}")

    st.divider()
    st.markdown("#### Other destinations")
    st.caption("More destinations (Slack, HubSpot, …) can be wired in here later.")

    st.divider()
    st.download_button(
        "⬇️ Download payload as JSON",
        data=json.dumps(payload, indent=2).encode(),
        file_name="automation_payload.json",
        mime="application/json",
    )

    if st.button("← Back to results"):
        st.session_state["auto_stage"] = 3
        st.rerun()


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
        _render_stage_send()


if __name__ == "__main__":
    run()
