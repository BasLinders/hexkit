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
    render_user_filter,
    render_execution_gate,
    render_combined_execution_gate,
    render_sql_viewer,
)
from utility.sql_builder import (
    BaselineParams,
    BinomialParams,
    ContinuousParams,
    experiment_shared_scan_flags,
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
    MONETARY_METHOD_NOTES,
    MONETARY_METHOD_GUIDANCE,
    run_frequentist_analysis,
    run_bayesian_analysis,
    run_continuous_analysis,
    run_pretest_analysis,
    run_pretest_analysis_seasonal,
    collapse_to_per_visitor,
    build_airtable_payload,
    apply_field_map,
    best_match_field,
    guess_field_by_hints,
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


# Each is tried via guess_field_by_hints — exact match first, then a
# normalized (spaces/punctuation stripped) exact match, then a normalized
# substring match — so naming variants (incl. Dutch, since these bases
# aren't necessarily English) still resolve without an exact hint hit.
_ID_FIELD_HINTS = ("experiment id", "experiment_id", "test id", "test_id", "id")


def _guess_id_field(fields: list[str]) -> Optional[str]:
    """Best-effort default for 'which field identifies existing records' —
    e.g. an Airtable autonumber field named 'Experiment ID'."""
    return guess_field_by_hints(_ID_FIELD_HINTS, fields)


_HYPOTHESIS_FIELD_HINTS = ("hypothesis", "hypothese")


def _guess_hypothesis_field(fields: list[str]) -> Optional[str]:
    return guess_field_by_hints(_HYPOTHESIS_FIELD_HINTS, fields)


_CUSTOM_CODE_FIELD_HINTS = (
    "custom code", "code", "custom js", "js code", "code snippet", "aangepaste code",
)


def _guess_custom_code_field(fields: list[str]) -> Optional[str]:
    return guess_field_by_hints(_CUSTOM_CODE_FIELD_HINTS, fields)


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
        use_seasonal_pretest = st.checkbox(
            "Use seasonal forecast (Prophet) instead of a flat weekly average",
            value=False,
            key="autofetch_pretest_seasonal",
            help=(
                "Fits a Prophet model to daily baseline traffic/conversions and "
                "projects the next 6 weeks with weekly/yearly seasonality, instead "
                "of assuming flat traffic — useful around holidays or promo periods "
                "that would otherwise be averaged away. Needs at least 14 days of "
                "daily baseline history (hard minimum), but yearly seasonality is "
                "only meaningfully estimated with several months of history — with "
                "just a few weeks, later forecast weeks can flatten out unreliably. "
                "Prefer 8+ weeks above when this is on."
            ),
        )

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
            baseline_days = (baseline_end - baseline_start).days + 1
            if use_seasonal_pretest and baseline_days < 14:
                st.warning(
                    "Seasonal forecasting needs at least 14 days of daily baseline "
                    "history — pick 2 or more weeks above, or turn it off."
                )
            elif use_seasonal_pretest and baseline_days < 56:
                st.caption(
                    "ℹ️ With under ~8 weeks of history, Prophet's yearly-seasonality "
                    "component is poorly estimated and later forecast weeks can "
                    "flatten out unrealistically — treat this as directional, or "
                    "pick more weeks above for a more reliable projection."
                )

            filter_type, filter_value = render_user_filter(
                project, dataset, str(baseline_start), str(baseline_end),
                key_prefix="autofetch_pretest",
            )

            baseline_params = BaselineParams(
                project=project,
                dataset=dataset,
                start_date=str(baseline_start),
                end_date=str(baseline_end),
                output_type="binomial",
                output_shape="daily" if use_seasonal_pretest else "aggregate",
                kpi_add_to_cart=(PRETEST_KPI_OPTIONS[kpi_choice] == "add_to_cart"),
                filter_type=filter_type,
                filter_value=filter_value,
            )
            baseline_sql = build_baseline(baseline_params)
            render_sql_viewer(baseline_sql, key="auto_pretest_sql")
            pretest_result_key = "auto_pretest_daily_result" if use_seasonal_pretest else "auto_pretest_result"
            render_execution_gate(
                project, baseline_sql, result_key=pretest_result_key, allow_preview=True,
                dataset=dataset,
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

    st.divider()
    filter_type, filter_value = render_user_filter(
        project, dataset, start_date, end_date, key_prefix="autofetch_exp",
    )

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
            filter_type=filter_type,
            filter_value=filter_value,
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
            filter_type=filter_type,
            filter_value=filter_value,
        )

    need_page_location, need_payment_type = experiment_shared_scan_flags(binomial_params, continuous_params)
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
        render_execution_gate(
            project, sql, result_key=f"auto_{label}_result", allow_preview=True, dataset=dataset,
        )

    # Gated on the current want_binomial/want_continuous checkbox state, not
    # just whether a result happens to be cached under these keys -- without
    # that gate, unchecking one after a previous combined fetch (e.g. after
    # seeing the continuous scan's cost estimate) would silently carry the
    # stale result forward into Step 2/3 with no warning, even mismatched
    # against a since-changed experiment ID, date range, or filter if the
    # user also changed those before re-fetching just the other one.
    df_binomial = st.session_state.get("auto_binomial_result") if want_binomial else None
    df_continuous = st.session_state.get("auto_continuous_result") if want_continuous else None
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

        # Computed from the local start_date/end_date (reliable right here,
        # right after render_date_range) and stashed under an auto_-prefixed
        # key rather than re-derived later from the bare start_date/end_date
        # session-state keys — those are plain st.date_input widget state,
        # not carried across an OAuth token-refresh redirect the way
        # admin_authenticated is (see render_gcp_credentials_gate's
        # extra_state above), so re-reading them live in Step 2 could
        # silently reset runtime_days to 1 (today - today) mid-flow and
        # inflate every "effect on revenue" projection by the true runtime.
        try:
            _start = datetime.strptime(start_date, "%Y-%m-%d").date()
            _end = datetime.strptime(end_date, "%Y-%m-%d").date()
            st.session_state["auto_runtime_days"] = max((_end - _start).days + 1, 1)
        except ValueError:
            st.session_state["auto_runtime_days"] = 1

        # Gated on the current want_pretest checkbox state, same reason as the
        # want_binomial/want_continuous gate above: without it, unchecking
        # "Fetch pre-test baseline data" after a previous fetch (e.g. going
        # back and deciding not to use it this run) wouldn't actually drop
        # the stale result -- the elif below would still pick it up and build
        # auto_pretest_meta from it, silently keeping Pre-Test Analysis
        # available with baseline data from a possibly different
        # weeks/KPI/experiment configuration.
        df_pretest = st.session_state.get("auto_pretest_result") if want_pretest else None
        df_pretest_daily = st.session_state.get("auto_pretest_daily_result") if want_pretest else None
        st.session_state["auto_df_pretest"] = df_pretest
        use_seasonal = want_pretest and bool(st.session_state.get("autofetch_pretest_seasonal", False))
        kpi_choice = st.session_state.get("autofetch_pretest_kpi", "Transactions (purchases)")

        # Seasonal mode fetches a separate daily query (auto_pretest_daily_result)
        # from the flat aggregate one -- toggling seasonal on doesn't retroactively
        # fetch it. Without this check, falling through to the elif below would
        # silently build a "seasonal": False result from a stale aggregate fetch
        # (or an earlier experiment's), with no indication the toggle was ignored.
        if use_seasonal and (df_pretest_daily is None or df_pretest_daily.empty):
            st.warning(
                "Seasonal forecast is turned on, but the daily baseline query "
                "above hasn't been run yet — click '✅ Run full query' in the "
                "pre-test section above, or turn off seasonal forecasting, "
                "before continuing."
            )
            return

        if use_seasonal and df_pretest_daily is not None and not df_pretest_daily.empty:
            weeks = int(st.session_state.get("autofetch_pretest_weeks", 1))
            value_col = (
                "add_to_cart_conversions"
                if PRETEST_KPI_OPTIONS[kpi_choice] == "add_to_cart"
                else "conversions"
            )
            st.session_state["auto_pretest_meta"] = {
                "seasonal": True,
                "weeks": weeks,
                "kpi_label": kpi_choice,
                "daily_df": df_pretest_daily,
                "value_col": value_col,
            }
        elif df_pretest is not None and not df_pretest.empty:
            weeks = int(st.session_state.get("autofetch_pretest_weeks", 1))
            row = df_pretest.iloc[0]
            conversions_col = (
                "total_add_to_cart_conversions"
                if PRETEST_KPI_OPTIONS[kpi_choice] == "add_to_cart"
                else "total_conversions"
            )
            st.session_state["auto_pretest_meta"] = {
                "seasonal": False,
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

    # The raw continuous fetch is one row per (visitor, order) pair, not one
    # row per visitor -- see collapse_to_per_visitor's docstring. Computed
    # once here and reused everywhere in this function that needs a visitor
    # count or a per-visitor revenue figure (the preview table below,
    # total_visitors, daily_visitors_continuous), so none of them can drift
    # out of sync on what "visitor" means by re-deriving it locally.
    per_visitor_continuous = (
        collapse_to_per_visitor(df_continuous) if df_continuous is not None else None
    )

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

    if per_visitor_continuous is not None:
        st.markdown("**Continuous data — revenue per visitor**")
        cont_summary = (
            per_visitor_continuous
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

    # Persisted at the Step 1 -> Step 2 transition, not re-derived from the
    # bare start_date/end_date session-state keys here — those can silently
    # reset (e.g. an OAuth token-refresh redirect mid-flow), which would
    # otherwise collapse runtime_days to 1 and inflate every "effect on
    # revenue" projection by a factor equal to the true test runtime.
    runtime_days = st.session_state.get("auto_runtime_days", 1)
    if control is not None and variation is not None:
        total_visitors = control.visitors + variation.visitors
    else:
        # per_visitor_continuous is already collapsed to one row per exposed
        # visitor (see collapse_to_per_visitor) -- unlike df_continuous
        # itself, which is one row per (visitor, order) and would over-count
        # any repeat purchaser.
        total_visitors = len(per_visitor_continuous)
    daily_visitors = total_visitors / runtime_days

    # Continuous's own population can differ from Binomial's even for the
    # same experiment/date range: its shared-scan query INNER JOINs against
    # device_data (device_category IN desktop/mobile only), while Binomial's
    # fetch here doesn't split by device at all (kpi_device_split=False) and
    # so isn't restricted by device category. Reusing the Binomial-derived
    # daily_visitors above to scale Continuous's own rate/mean (measured on
    # its own, potentially smaller population) would systematically over- or
    # under-state its monetary projection — so Continuous gets its own,
    # computed from its own fetched rows regardless of whether Binomial was
    # also fetched. Uses per_visitor_continuous's row count, not
    # len(df_continuous), for the same reason as total_visitors above.
    daily_visitors_continuous = (
        len(per_visitor_continuous) / runtime_days if per_visitor_continuous is not None else daily_visitors
    )

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
            if pretest_meta.get("seasonal"):
                st.caption(
                    f"Baseline: {pretest_meta['kpi_label']} · {pretest_meta['weeks']} week(s) of "
                    "daily history · seasonal (Prophet) forecast, not a flat weekly average."
                )
            else:
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
                help=(
                    "0 treats AOV as a known constant for both methods. Above 0: Frequentist "
                    "propagates it as estimation-uncertainty on the AOV mean (widens the CI); "
                    "Bayesian samples AOV from a log-normal with this CV on every simulated draw "
                    "(widens the posterior spread). Same input, different mechanics per method."
                ),
            )
        if use_continuous and daily_visitors_continuous != daily_visitors:
            st.caption(
                f"Test runtime: {runtime_days} day(s) · Daily visitors: {daily_visitors:,.0f} "
                f"(Frequentist/Bayesian) · {daily_visitors_continuous:,.0f} (Continuous — own population)"
            )
        else:
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
                    aov_cv=aov_cv,
                )
            if use_continuous:
                results["continuous"] = run_continuous_analysis(
                    df_continuous,
                    control_label="A",
                    variation_label="B",
                    daily_visitors=daily_visitors_continuous,
                    projection_days=int(projection_days),
                    confidence_level=confidence_level,
                    tail=tail,
                    mode=continuous_mode,
                )
            if use_pretest:
                if pretest_meta.get("seasonal"):
                    results["pretest"] = run_pretest_analysis_seasonal(
                        daily_df=pretest_meta["daily_df"],
                        value_col=pretest_meta["value_col"],
                        kpi_label=pretest_meta["kpi_label"],
                        weeks_used=pretest_meta["weeks"],
                        confidence_level=confidence_level,
                        tail=tail,
                        trust_pct=trust_pct,
                    )
                else:
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

    def _render_monetary_notes(method: str):
        notes = MONETARY_METHOD_NOTES.get(method)
        if not notes:
            return
        with st.expander(f"Pros & cons of {method.capitalize()}'s monetary estimate"):
            st.markdown("**Pros**")
            for point in notes["pros"]:
                st.markdown(f"- {point}")
            st.markdown("**Cons**")
            for point in notes["cons"]:
                st.markdown(f"- {point}")

    active_monetary_methods = [m for m in ("frequentist", "bayesian", "continuous") if results.get(m)]
    if len(active_monetary_methods) > 1:
        st.info(f"💡 {MONETARY_METHOD_GUIDANCE}", icon="💡")

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
        if freq.get("monetary_conclusion"):
            st.caption(freq["monetary_conclusion"])
        _render_monetary_notes("frequentist")
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
        if bayes.get("monetary_conclusion"):
            st.caption(bayes["monetary_conclusion"])
        _render_monetary_notes("bayesian")
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
        if cont.get("monetary_conclusion"):
            st.caption(cont["monetary_conclusion"])
        _render_monetary_notes("continuous")
        st.divider()

    if pretest:
        st.markdown("### Pre-Test Analysis")
        if pretest.get("seasonal"):
            st.caption(
                f"Baseline: {pretest['kpi']} · {pretest['weeks_used']} week(s) of daily "
                f"history · seasonal (Prophet) forecast. {pretest.get('forecast_summary', '')}"
            )
        else:
            st.caption(
                f"Baseline: {pretest['kpi']} · {pretest['weeks_used']} week(s) · "
                f"{pretest['weekly_visitors']:,.0f} visitors/week, "
                f"{pretest['weekly_conversions']:,.1f} conversions/week."
            )
        if pretest["table"]:
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

def _render_lookup_base_table_picker(api_key: str) -> tuple[Optional[str], Optional[dict]]:
    """
    Lets the user pick which Airtable base/table holds this experiment's
    tracking record, via the same live discovery the Send step uses.

    get_credentials()'s base_id/table_name are NOT usable here: they only
    ever reflect Step 5's own selections (airtable_base_ids /
    airtable_table_names_{base_id}, both plural/multi-base since the
    multi-base send refactor) or a legacy AIRTABLE_BASE_ID/AIRTABLE_TABLE_NAME
    secret pair the app no longer needs anywhere else -- and since this step
    (4) runs before Send (5) in the wizard, neither is necessarily set yet on
    a fresh run. Shares Step 5's bases/tables cache keys (same underlying
    data) but keeps its own selection keys so the two steps' choices don't
    collide.
    Returns (base_id, table_dict) — table_dict is None until a base with at
    least one table is actually resolved.
    """
    if st.session_state.get("airtable_bases_for_key") != api_key:
        with st.spinner("Loading Airtable bases…"):
            result = list_bases(api_key)
        if not result["ok"]:
            st.error(f"Couldn't list Airtable bases: {result['error']}")
            return None, None
        st.session_state["airtable_bases_cache"] = result["bases"]
        st.session_state["airtable_bases_for_key"] = api_key

    bases: dict = st.session_state.get("airtable_bases_cache", {})
    if not bases:
        st.warning("This token can't see any bases — check its access under Airtable's token settings.")
        return None, None

    base_ids = list(bases.keys())
    prior_base = st.session_state.get("auto_lookup_base_id")
    base_id = st.selectbox(
        "Base to check for this experiment's record",
        options=base_ids,
        index=base_ids.index(prior_base) if prior_base in base_ids else 0,
        format_func=lambda bid: bases.get(bid, bid),
        key="auto_lookup_base_id",
    )

    tables_cache_key = f"airtable_tables_{base_id}"
    if tables_cache_key not in st.session_state:
        with st.spinner(f"Loading tables for {bases.get(base_id, base_id)}…"):
            result = list_tables(api_key, base_id)
        if not result["ok"]:
            st.error(f"Couldn't list tables in {bases.get(base_id, base_id)}: {result['error']}")
            return base_id, None
        st.session_state[tables_cache_key] = result["tables"]

    tables: list = st.session_state.get(tables_cache_key, [])
    if not tables:
        st.warning("This base has no tables.")
        return base_id, None

    table_names = [t["name"] for t in tables]
    prior_table = st.session_state.get("auto_lookup_table_name")
    table_choice = st.selectbox(
        "Table",
        options=table_names,
        index=table_names.index(prior_table) if prior_table in table_names else 0,
        key="auto_lookup_table_name",
    )
    return base_id, next(t for t in tables if t["name"] == table_choice)


def _lookup_experiment_record() -> tuple[list[str], Optional[dict], str]:
    """
    Searches the user-picked Airtable base/table (see
    _render_lookup_base_table_picker) for the record matching this
    experiment (Step 1's exp_prefix, via the same id-field guess the Send
    step uses). Shared by the Hypothesis gate (required, blocks generation)
    and the Custom Code lookup (optional context for Gemini) so both reuse
    one round-trip.
    Returns (table_field_names, matched_record_fields_or_None, message).
    """
    exp_prefix = st.session_state.get("auto_exp_prefix")
    creds = get_credentials()
    api_key = creds["api_key"]

    if not api_key:
        return [], None, (
            "No Airtable token configured (AIRTABLE_API_KEY) — can't look up the experiment record."
        )
    if not exp_prefix:
        return [], None, "No experiment ID from Step 1 to look up."

    base_id, table = _render_lookup_base_table_picker(api_key)
    if table is None:
        return [], None, "Pick a base and table above to look up the experiment record."

    # Persisted so Step 5 can pre-select this same base/table (and, once a
    # record is found below, that same record) instead of making the user
    # pick and search for something already resolved here.
    resolved = {"base_id": base_id, "table_id": table["id"], "table_name": table["name"]}
    st.session_state["auto_lookup_resolved"] = resolved

    id_field = _guess_id_field(table["fields"])
    if not id_field:
        return table["fields"], None, "Couldn't find an ID field on the selected table."
    resolved["id_field"] = id_field

    search_result = search_records(base_id, table["id"], api_key, id_field, exp_prefix)
    if not search_result["ok"]:
        return table["fields"], None, f"Airtable lookup failed: {search_result['error']}"

    matches = search_result["records"]
    if not matches:
        return table["fields"], None, f"No Airtable record found yet for experiment '{exp_prefix}'."

    hypothesis_field = _guess_hypothesis_field(table["fields"])

    if len(matches) > 1:
        # search_records matches by substring (see its own docstring), so more
        # than one record legitimately containing exp_prefix is a real
        # possibility (e.g. "104" also matching "1041..."), not just a rare
        # edge case. Silently picking one on the user's behalf risked
        # attaching this result — and the AI conclusion — to the wrong
        # experiment's record with no indication anything was ambiguous.
        # Surface every candidate and let the user confirm, defaulting to the
        # same "prefer a match with a filled Hypothesis" pick as before.
        def _match_label(r: dict) -> str:
            id_value = r["fields"].get(id_field, "(blank)")
            has_hyp = bool(hypothesis_field and str(r["fields"].get(hypothesis_field, "")).strip())
            marker = "  ·  ✓ has hypothesis" if has_hyp else ""
            return f"{id_value}  ·  …{r['id'][-6:]}{marker}"

        default_match = next(
            (r for r in matches if hypothesis_field and str(r["fields"].get(hypothesis_field, "")).strip()),
            matches[0],
        )
        options = [_match_label(r) for r in matches]
        st.warning(
            f"{len(matches)} Airtable records matched '{exp_prefix}' on **{id_field}** — "
            "confirm which one this experiment's result actually belongs to."
        )
        # Scoped per base/table/search-term, not a bare key — an unscoped key
        # would keep the previous pick's value in session_state when the
        # matched candidates change (a different base/table, or a second
        # experiment's exp_prefix in the same session). Streamlit doesn't
        # error when that stored value no longer matches the new options; it
        # silently resets to the new list's first entry with no indication
        # anything changed, discarding the "prefer a filled Hypothesis"
        # default -- and that silently-reset pick is what pre-seeds Step 5's
        # "existing record to update", risking a result landing on the wrong
        # experiment's record.
        choice_key = f"auto_lookup_record_choice_{base_id}_{table['id']}_{exp_prefix}"
        choice = st.selectbox(
            "Matched record", options=options,
            index=matches.index(default_match),
            key=choice_key,
        )
        matched = matches[options.index(choice)]
        message = f"Matched Airtable record for '{exp_prefix}' ({len(matches)} candidates — confirm above)."
    else:
        matched = matches[0]
        message = f"Matched Airtable record for '{exp_prefix}'."

    resolved["record_id"] = matched["id"]
    resolved["record_fields"] = matched["fields"]
    resolved["search_term"] = exp_prefix
    return table["fields"], matched["fields"], message


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

    st.markdown("**Where's this experiment tracked in Airtable?**")
    st.caption(
        "Used to check the Hypothesis field before generating a conclusion, and to "
        "pull in Custom Code as extra context — independent of where you actually "
        "send results in Step 5."
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
        # Each method's own written-output methods: "conclusion" is the
        # statistical read (significance only); "monetary_conclusion" (where
        # available) is the money-framed narrative — handles one-sided CI
        # wording, non-significant hedging, and Bayesian's risk/reward
        # framing correctly, which payload's bare numeric fields don't.
        "conclusions": {
            method: {
                key: result[key]
                for key in ("conclusion", "monetary_conclusion")
                if result.get(key)
            }
            for method, result in results.items()
            if result.get("conclusion") or result.get("monetary_conclusion")
        },
    }
    active_monetary_methods = [
        m for m in ("frequentist", "bayesian", "continuous") if results.get(m)
    ]
    if active_monetary_methods:
        # Lets Gemini reason about which method's "effect on revenue" is most
        # defensible for this specific experiment (sample size, whether the
        # KPI is a rate or revenue itself, significance status) rather than
        # treating the three as interchangeable — same notes shown to the
        # human in Step 3.
        ai_input["monetary_method_notes"] = {
            m: MONETARY_METHOD_NOTES[m] for m in active_monetary_methods
        }
        if len(active_monetary_methods) > 1:
            ai_input["monetary_method_guidance"] = MONETARY_METHOD_GUIDANCE
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
    lookup_resolved = st.session_state.get("auto_lookup_resolved") or {}
    prior_bases = st.session_state.get("airtable_base_ids") or (
        [creds["base_id"]] if creds["base_id"] in base_ids else []
    ) or (
        # Nothing picked here yet — default to whatever base Step 4 already
        # resolved this experiment's record to, so it doesn't need picking twice.
        [lookup_resolved["base_id"]] if lookup_resolved.get("base_id") in base_ids else []
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
            # If Step 4 already resolved this base to a specific table, prefer
            # that over the generic "experiment(s)"-named-table guess.
            lookup_table_name = (
                [lookup_resolved["table_name"]]
                if lookup_resolved.get("base_id") == base_id
                and lookup_resolved.get("table_name") in table_names
                else []
            )
            default_selection = (
                [n for n in stored_selection if n in table_names]
                or lookup_table_name
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

                    # Pre-seed this exact (base, table)'s lookup widgets from
                    # Step 4's already-resolved record — before they're
                    # created below, so the user doesn't have to pick the ID
                    # field or search again for something already found.
                    # Only seeds keys that don't exist yet, so it never
                    # clobbers a choice the user has since changed.
                    is_lookup_resolved_table = (
                        lookup_resolved.get("base_id") == base_id
                        and lookup_resolved.get("table_id") == table["id"]
                    )
                    if is_lookup_resolved_table and lookup_resolved.get("id_field"):
                        idfield_key = f"{lookup_key}_idfield"
                        if idfield_key not in st.session_state:
                            st.session_state[idfield_key] = lookup_resolved["id_field"]
                        if lookup_resolved.get("record_id"):
                            term_key = f"{lookup_key}_term"
                            if term_key not in st.session_state and lookup_resolved.get("search_term"):
                                st.session_state[term_key] = lookup_resolved["search_term"]
                            results_key = f"{lookup_key}_results"
                            if results_key not in st.session_state:
                                st.session_state[results_key] = [{
                                    "id": lookup_resolved["record_id"],
                                    "fields": lookup_resolved.get("record_fields") or {},
                                }]
                            choice_key = f"{lookup_key}_choice"
                            if choice_key not in st.session_state:
                                record_fields = lookup_resolved.get("record_fields") or {}
                                id_value = record_fields.get(lookup_resolved["id_field"], "(blank)")
                                st.session_state[choice_key] = (
                                    f"{id_value}  ·  …{lookup_resolved['record_id'][-6:]}"
                                )

                    st.markdown("**Existing record**")
                    id_field_options = ["— Always create new —"] + table["fields"]
                    idfield_key = f"{lookup_key}_idfield"
                    # index= only applies the very first time this key is
                    # rendered — passing it alongside a key that's already in
                    # session_state (e.g. pre-seeded from Step 4 above) is
                    # redundant and Streamlit warns about it, so omit it once
                    # the key already has a value.
                    default_id_field = _guess_id_field(table["fields"])
                    default_idx = (
                        id_field_options.index(default_id_field)
                        if default_id_field in id_field_options else 0
                    )
                    id_field = st.selectbox(
                        "Field to search on (e.g. an experiment ID Airtable generated)",
                        options=id_field_options,
                        **({} if idfield_key in st.session_state else {"index": default_idx}),
                        key=idfield_key,
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

                    # apply_field_map has no way to detect this: two internal
                    # keys mapped to the same Airtable field silently collapse
                    # to whichever one dict iteration processes last, dropping
                    # the other's value with no error. Only a UI-level warning
                    # can catch it, since the collision only exists at the
                    # point the user picks the mapping.
                    chosen_fields = list(field_map.values())
                    dupes = sorted({f for f in chosen_fields if chosen_fields.count(f) > 1})
                    if dupes:
                        dupe_list = ", ".join(dupes)
                        st.warning(
                            f"Multiple result fields are mapped to the same Airtable column "
                            f"({dupe_list}) — only one value will actually be sent for each, "
                            "silently dropping the rest. Fix the mapping above before sending."
                        )

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
