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
from typing import Literal, cast

import pandas as pd

from utility.bq_ui_components import (
    render_gcp_credentials_gate,
    render_connection_selectors,
    render_date_range,
    render_variant_inputs,
    render_execution_gate,
    render_sql_viewer,
)
from utility.sql_builder import (
    BinomialParams,
    binomial_shared_scan_flags,
    build_shared_scan_select,
    build_binomial_from_shared_scan,
    build_experiment_single_output_sql,
)
from utility.automation_engine import (
    VariantData,
    TAILS,
    run_frequentist_analysis,
    run_bayesian_analysis,
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
        "Runs the same binomial export used in Data Export, restricted to "
        "exactly one control (A) and one variation (B)."
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

    params = BinomialParams(
        project=project,
        dataset=dataset,
        start_date=start_date,
        end_date=end_date,
        param_key=param_key,
        match_strategy=cast(Literal["exact", "like"], match_strategy),
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
    need_page_location, need_payment_type = binomial_shared_scan_flags(params)
    shared_scan_select = build_shared_scan_select(
        project, dataset, start_date, end_date, param_key,
        need_page_location, need_payment_type,
    )
    sql = build_experiment_single_output_sql(shared_scan_select, build_binomial_from_shared_scan(params))
    render_sql_viewer(sql, key="auto_sql")

    render_execution_gate(project, sql, result_key="auto_query_result", allow_preview=True)

    df = st.session_state.get("auto_query_result")
    if df is None:
        return

    row_labels = set(df["experience_variant_label"])
    if not {"A", "B"}.issubset(row_labels):
        st.error("Query result is missing rows for control (A) and/or variation (B).")
        return

    st.divider()
    if st.button("Continue to choose analysis method(s) →", type="primary"):
        st.session_state["auto_df"] = df
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
    df = st.session_state.get("auto_df")
    if df is None:
        st.session_state["auto_stage"] = 1
        st.rerun()
        return

    st.subheader("Step 2 — Choose analysis method(s)")

    control = _variant_from_row(df[df["experience_variant_label"] == "A"].iloc[0], "Control")
    variation = _variant_from_row(df[df["experience_variant_label"] == "B"].iloc[0], "Variation")

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

    st.divider()
    st.markdown("**Analysis method(s)**")
    c1, c2, c3 = st.columns(3)
    with c1:
        use_frequentist = st.checkbox("Frequentist Analysis", value=True, key="auto_use_frequentist")
    with c2:
        use_bayesian = st.checkbox("Bayesian Analysis", value=True, key="auto_use_bayesian")
    with c3:
        st.checkbox(
            "Pre-Test Analysis", value=False, disabled=True,
            key="auto_use_pretest", help="Coming soon.",
        )

    if not use_frequentist and not use_bayesian:
        st.warning("Select at least one analysis method to continue.")
        return

    start = st.session_state.get("start_date")
    end = st.session_state.get("end_date")
    runtime_days = max((end - start).days + 1, 1) if start and end else 1
    daily_visitors = (control.visitors + variation.visitors) / runtime_days

    st.divider()
    if use_frequentist:
        with st.expander("Frequentist settings", expanded=True):
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

    revenue_source = "frequentist"
    if use_frequentist and use_bayesian:
        choice = st.radio(
            "Use for the shared 'effect on revenue' field:",
            options=["Frequentist", "Bayesian"],
            horizontal=True,
            key="auto_revenue_source_radio",
        )
        revenue_source = choice.lower()
    elif use_bayesian:
        revenue_source = "bayesian"

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
    if not results or control is None or variation is None:
        st.session_state["auto_stage"] = 2
        st.rerun()
        return

    st.subheader("Step 3 — Review results")

    freq = results.get("frequentist")
    bayes = results.get("bayesian")

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

    revenue_source = st.session_state.get("auto_revenue_source", "frequentist")
    payload = build_airtable_payload(control, variation, freq, bayes, revenue_source=revenue_source)
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
