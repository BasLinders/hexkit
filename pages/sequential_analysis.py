import logging
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import uuid
from pydantic import ValidationError
from scipy.stats import norm as scipy_norm
from st_supabase_connection import SupabaseConnection
from foe.bayesian.operations import BayesianEngine, get_lift_prior
from foe.core.models import BusinessCaseInput, ExperimentInput

# --- CONSTANTS & CONFIGURATION ---
st.set_page_config(page_title="Sequential Analysis", layout="wide")

logger = logging.getLogger(__name__)

TEST_TYPE_ONE_SAMPLE = "One-sample (fixed baseline)"
TEST_TYPE_MULTI_SAMPLE = "Multi-sample (Control vs. Variants)"

# Initialize Supabase Connection
conn = st.connection("supabase", type=SupabaseConnection)

# Session-state keys for the ad-hoc Revenue Impact overlay, reset whenever a
# new or different experiment is loaded so settings never leak between them.
REVENUE_SETTINGS_KEYS = (
    "seq_aov", "seq_aov_cv", "seq_use_lift_prior",
    "seq_expected_lift_pct", "seq_skepticism",
)


def is_valid_uuid(val):
    """Validates if a string is a properly formatted UUID."""
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False


def get_deduped_variant_df(df, variant_name):
    """
    Returns variant_name's rows collapsed to one row per date (last write
    wins for duplicate dates). Shared by the LLR/decision-card path and the
    Revenue Impact batch precompute so "what counts as latest" has exactly
    one implementation.
    """
    variant_df = df[df["variant_name"] == variant_name].copy()
    return variant_df.groupby("measurement_date").last().reset_index()


# --- STATISTICAL FUNCTIONS ---

def calculate_msprt_boundaries(alpha, beta, num_variants=1):
    """
    Calculates boundaries for mSPRT.
    Upper bound applies a Bonferroni correction when testing multiple variants,
    raising the bar to control the family-wise error rate.
    """
    upper = np.log(num_variants / alpha)
    lower = np.log(beta)
    return upper, lower


def calculate_msprt_llr(visitors_base, conversions_base, visitors_var, conversions_var,
                         tau=0.0004, fixed_baseline_cr=None):
    """
    Calculates the Log-Likelihood Ratio.
    Handles both two-sample pooled variance and one-sample fixed variance.
    """
    if visitors_var == 0:
        return 0.0

    p_var = conversions_var / visitors_var

    if fixed_baseline_cr is not None:
        # ONE-SAMPLE: Variance of a single proportion
        p_base = fixed_baseline_cr
        var = p_var * (1 - p_var) / visitors_var
        if var == 0:
            return 0.0
        diff = p_var - p_base
    else:
        # TWO-SAMPLE: Pooled variance of the difference
        if visitors_base == 0:
            return 0.0
        p_base = conversions_base / visitors_base
        p_pool = (conversions_base + conversions_var) / (visitors_base + visitors_var)
        if p_pool <= 0 or p_pool >= 1:
            return 0.0
        var = p_pool * (1 - p_pool) * (1 / visitors_base + 1 / visitors_var)
        if var == 0:
            return 0.0
        diff = p_var - p_base

    # mSPRT LLR Formula
    llr = 0.5 * (np.log(var / (var + tau)) + (diff ** 2 / var) * (tau / (var + tau)))
    return llr


def calculate_instantaneous_power(n_var, p0, mde, alpha, n_ctrl=None):
    """
    Estimates statistical power at the current sample sizes using a normal approximation.

    For one-sample: n_ctrl is None; variance is computed from the fixed baseline p0.
    For two-sample: both n_ctrl and n_var are used; variance accounts for both group sizes.

    Returns (power, beta_est) as floats.

    Important: this is a fixed-horizon approximation. mSPRT power is structurally lower
    due to the always-valid guarantee — treat the result as an optimistic upper bound.
    """
    if n_var <= 0 or mde <= 0 or p0 <= 0:
        return 0.0, 1.0

    p0 = float(np.clip(p0, 0.001, 0.999))
    p1 = float(np.clip(p0 + mde, 0.001, 0.999))
    z_alpha = scipy_norm.ppf(1 - alpha)

    if n_ctrl is None:
        # One-sample: variance under H1 using the alternative proportion
        se = np.sqrt(p1 * (1 - p1) / n_var)
    else:
        # Two-sample: variance of the difference under H1
        n_ctrl = max(int(n_ctrl), 1)
        se = np.sqrt(p0 * (1 - p0) / n_ctrl + p1 * (1 - p1) / n_var)

    if se == 0:
        return 0.0, 1.0

    z_power = mde / se - z_alpha
    power = float(scipy_norm.cdf(z_power))
    return power, 1.0 - power


# --- BUSINESS IMPACT (BAYESIAN REVENUE OVERLAY) ---
# Ad-hoc monetary lens on top of the locked mSPRT test: delegates the Beta
# posterior CR sampling, log-normal AOV sampling, lift-plausibility weighting,
# and monetary projection to the shared FOE BayesianEngine (foe.bayesian.operations)
# rather than re-implementing it here. Never feeds back into alpha/beta/tau or
# the LLR decision itself. Requires a real Control group, so it only applies
# to Multi-sample tests -- the engine has no concept of a fixed baseline CR.

BUSINESS_PROJECTION_DAYS = 183  # ~6 months, matches the Business Case page


@st.cache_data(ttl=60, show_spinner="Simulating revenue impact...")
def _compute_revenue_impact(
    batch_visitors, batch_conversions, batch_labels,
    aov_value, aov_cv, runtime_days,
    expected_lift_pct, skepticism,
):
    """
    Runs the FOE BayesianEngine Monte Carlo simulation once per unique
    combination of inputs (cached across Streamlit reruns triggered by
    unrelated widgets). Takes hashable primitives only -- the lift prior is
    rebuilt from its (expected_lift_pct, skepticism) inputs inside the cache
    boundary rather than passed in as an object, so caching doesn't depend on
    how the foe package's dataclass happens to hash.
    """
    lift_prior = get_lift_prior(expected_lift_pct=expected_lift_pct, skepticism=skepticism)
    engine = BayesianEngine(seed=42)

    experiment_input = ExperimentInput(
        visitors=list(batch_visitors),
        conversions=list(batch_conversions),
        labels=list(batch_labels),
    )
    probability_results = engine.run_probability_analysis(experiment_input, lift_prior=lift_prior)
    # index 0 (Control) is unused by run_monetary_projection.
    prob_best_overall = [0.0] + [r.prob_being_best for r in probability_results]

    biz_case = BusinessCaseInput(
        aovs={label: aov_value for label in batch_labels},
        runtime_days=runtime_days,
        projection_period=BUSINESS_PROJECTION_DAYS,
    )
    monetary_results = engine.run_monetary_projection(
        visitors=list(batch_visitors),
        conversions=list(batch_conversions),
        biz_case=biz_case,
        prob_best_overall=prob_best_overall,
        variant_labels=list(batch_labels),
        lift_prior=lift_prior,
        aov_cv=aov_cv,
    )
    return {r["variant_label"]: r for r in monetary_results}


# --- DATABASE FUNCTIONS ---

def get_experiment_params(experiment_id):
    """Fetch setup parameters for an ID."""
    try:
        response = (
            conn.table("experiment_params")
            .select("*")
            .eq("experiment_id", experiment_id)
            .execute()
        )
        if len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        st.error(f"Error fetching params: {e}")
        return None


def save_experiment_params(experiment_id, p0, tau, alpha, beta, max_visitors, test_type, num_variants):
    """Save the immutable rules of the experiment."""
    try:
        data = {
            "experiment_id": experiment_id,
            "p0": float(p0),
            "tau": float(tau),
            "alpha": float(alpha),
            "beta": float(beta),
            "max_visitors": int(max_visitors),
            "test_type": str(test_type),
            "num_variants": int(num_variants),
        }
        conn.table("experiment_params").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Error creating experiment: {e}")
        return False


@st.cache_data(ttl=60)
def get_experiment_data(experiment_id):
    """Fetches data in the LONG format (variant_name, visitors, conversions)."""
    try:
        response = (
            conn.table("msprt_data")
            .select("*")
            .eq("experiment_id", experiment_id)
            .order("measurement_date")
            .execute()
        )
        if len(response.data) > 0:
            df = pd.DataFrame(response.data)
            df["measurement_date"] = pd.to_datetime(df["measurement_date"]).dt.date
            df["visitors"] = df["visitors"].fillna(0).astype(int)
            df["conversions"] = df["conversions"].fillna(0).astype(int)
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


def save_data_points(experiment_id, date, variant_data_list):
    """Insert a batch of data points for a single date."""
    try:
        insert_payload = []
        for data in variant_data_list:
            insert_payload.append(
                {
                    "experiment_id": experiment_id,
                    "measurement_date": str(date),
                    "variant_name": data["variant_name"],
                    "visitors": int(data["visitors"]),
                    "conversions": int(data["conversions"]),
                }
            )
        conn.table("msprt_data").insert(insert_payload).execute()
        st.toast("Data points saved successfully!", icon="✅")
        get_experiment_data.clear()
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False


def delete_data_points_by_date(experiment_id, date):
    """Deletes all rows for a specific date."""
    try:
        (
            conn.table("msprt_data")
            .delete()
            .eq("experiment_id", experiment_id)
            .eq("measurement_date", str(date))
            .execute()
        )
        st.toast(f"Entries for {date} deleted", icon="🗑️")
        get_experiment_data.clear()
        return True
    except Exception as e:
        st.error(f"Error deleting data: {e}")
        return False


# --- VISUALIZATIONS ---

def show_visualization(chart_df, upper_bound, lower_bound):
    if chart_df.empty:
        st.warning("No data to visualize yet.")
        return

    y_values = [chart_df["llr"].max(), chart_df["llr"].min(), upper_bound, lower_bound]
    padding = (max(y_values) - min(y_values)) * 0.1
    if padding == 0:
        padding = 1.0
    max_y = max(y_values) + padding
    min_y = min(y_values) - padding

    line = alt.Chart(chart_df).mark_line(point=True).encode(
        x=alt.X("measurement_date:T", title="Date"),
        y=alt.Y(
            "llr:Q",
            title="Log Likelihood Ratio",
            scale=alt.Scale(domain=[min_y, max_y]),
        ),
        color=alt.Color("variant_name:N", title="Variant"),
    )

    success_zone = (
        alt.Chart(pd.DataFrame({"y": [upper_bound], "y2": [max_y]}))
        .mark_rect(color="green", opacity=0.1)
        .encode(y="y", y2="y2")
    )
    futility_zone = (
        alt.Chart(pd.DataFrame({"y": [min_y], "y2": [lower_bound]}))
        .mark_rect(color="red", opacity=0.1)
        .encode(y="y", y2="y2")
    )
    upper_line = (
        alt.Chart(pd.DataFrame({"y": [upper_bound]}))
        .mark_rule(color="green", strokeDash=[5, 5])
        .encode(y="y")
    )
    lower_line = (
        alt.Chart(pd.DataFrame({"y": [lower_bound]}))
        .mark_rule(color="red", strokeDash=[5, 5])
        .encode(y="y")
    )

    chart = (
        (success_zone + futility_zone + upper_line + lower_line + line)
        .properties(height=400)
        .interactive()
    )

    st.markdown("### Test Trajectory")
    st.altair_chart(chart, width="stretch")


def show_power_chart(power_df, target_power):
    """
    Shows estimated power over time for all variants, with a reference line at the target.
    The power values are fixed-horizon approximations — labelled accordingly.
    """
    if power_df.empty:
        return

    y_min = max(power_df["power"].min() - 0.05, 0.0)
    y_max = 1.05

    power_line = alt.Chart(power_df).mark_line(point=True).encode(
        x=alt.X("measurement_date:T", title="Date"),
        y=alt.Y(
            "power:Q",
            title="Estimated Power",
            scale=alt.Scale(domain=[y_min, y_max]),
            axis=alt.Axis(format="%"),
        ),
        color=alt.Color("variant_name:N", title="Variant"),
    )

    target_rule = (
        alt.Chart(pd.DataFrame({"y": [target_power]}))
        .mark_rule(color="orange", strokeDash=[5, 5])
        .encode(y="y")
    )

    target_label = (
        alt.Chart(
            pd.DataFrame(
                {"y": [target_power], "label": [f"Target: {target_power:.0%}"]}
            )
        )
        .mark_text(align="left", dx=6, dy=-8, color="orange", fontSize=11)
        .encode(
            y=alt.Y("y:Q"),
            x=alt.value(0),
            text="label",
        )
    )

    chart = (
        (power_line + target_rule + target_label)
        .properties(height=220)
        .interactive()
    )

    st.markdown("### Power Trajectory")
    st.caption(
        "Fixed-horizon normal approximation at current sample sizes. "
        "mSPRT power is structurally lower — this is an optimistic bound. "
        "Once the estimated power crosses the target line, you have accumulated enough "
        "observations that a fixed-horizon test would be sufficiently powered."
    )
    st.altair_chart(chart, width="stretch")


# --- ANALYSIS ---

def analysis_section(df, params):
    st.divider()
    st.subheader("Sequential Analysis")

    alpha = params.get("alpha", 0.05)
    beta = params.get("beta", 0.20)
    tau_param = params.get("tau", 0.0004)
    test_type = params.get("test_type", TEST_TYPE_ONE_SAMPLE)
    max_visitors = params.get("max_visitors", 10000)
    num_variants = params.get("num_variants", 1)

    upper_bound, lower_bound = calculate_msprt_boundaries(alpha, beta, num_variants=num_variants)

    variants_to_test = [v for v in df["variant_name"].unique() if v != "Control"]
    chart_data = []
    power_data = []
    mde = np.sqrt(tau_param)

    # --- BUSINESS IMPACT SETTINGS (ad-hoc, does not affect the locked test) ---
    with st.expander("💰 Revenue Impact Settings (optional)", expanded=False):
        st.caption(
            "Ad-hoc monetary overlay on top of the locked mSPRT test. Entering an AOV "
            "translates the current evidence into an estimated revenue impact — it never "
            "changes alpha/beta/tau or the LLR decision itself."
        )
        aov_value = st.number_input(
            "Average Order Value (€)",
            min_value=0.0, step=0.01,
            value=st.session_state.get("seq_aov", 0.0),
            key="seq_aov",
            help="Leave at 0 to hide the revenue impact section.",
        )
        aov_cv = st.slider(
            "AOV Variability",
            min_value=0.1, max_value=2.0, step=0.1,
            value=st.session_state.get("seq_aov_cv", 0.5),
            key="seq_aov_cv",
            help="Coefficient of variation for order value (std / mean). Does not affect "
                 "the average prediction, only the spread.",
        )

        use_lift_prior = st.checkbox(
            "Apply a lift prior?",
            value=st.session_state.get("seq_use_lift_prior", False),
            key="seq_use_lift_prior",
            help="Express skepticism about large lifts before seeing the data.",
        )
        if use_lift_prior:
            lp_col1, lp_col2 = st.columns(2)
            expected_lift_pct = lp_col1.number_input(
                "Expected Lift (%)", min_value=-99.0, max_value=1000.0, step=0.1,
                value=st.session_state.get("seq_expected_lift_pct", 0.0),
                key="seq_expected_lift_pct",
            )
            skepticism = lp_col2.selectbox(
                "Skepticism", ["skeptical", "moderate", "uninformative"],
                index=["skeptical", "moderate", "uninformative"].index(
                    st.session_state.get("seq_skepticism", "skeptical")
                ),
                key="seq_skepticism",
            )
        else:
            expected_lift_pct = 0.0
            skepticism = "uninformative"

    revenue_enabled = aov_value > 0 and test_type == TEST_TYPE_MULTI_SAMPLE
    if aov_value > 0 and test_type == TEST_TYPE_ONE_SAMPLE:
        st.caption(
            "💡 Revenue Impact needs a real Control group with its own posterior, so it "
            "isn't available for One-sample tests (the baseline here is a fixed historical "
            "rate, not measured data)."
        )

    # --- REVENUE IMPACT (batch, via the shared FOE BayesianEngine) ---
    # Computed once for every variant together (not per-variant) because the
    # engine's monetary projection and "probability of being best" are inherently
    # multi-arm comparisons. revenue_unavailable_reason distinguishes *why* the
    # per-variant fallback caption applies, so it never contradicts a real error
    # shown above it.
    revenue_by_label = {}
    revenue_unavailable_reason = None  # None | "no_traffic" | "misaligned_dates" | "error"

    if revenue_enabled and variants_to_test:
        ctrl_dedup = get_deduped_variant_df(df, "Control")

        if ctrl_dedup.empty:
            revenue_unavailable_reason = "no_traffic"
        else:
            latest_shared_date = ctrl_dedup["measurement_date"].max()
            ctrl_latest = ctrl_dedup[ctrl_dedup["measurement_date"] == latest_shared_date].iloc[-1]

            batch_labels = ["Control"] + variants_to_test
            batch_visitors = [int(ctrl_latest["visitors"])]
            batch_conversions = [int(ctrl_latest["conversions"])]

            if batch_visitors[0] <= 0:
                revenue_unavailable_reason = "no_traffic"
            else:
                for v in variants_to_test:
                    v_dedup = get_deduped_variant_df(df, v)
                    if v_dedup.empty:
                        revenue_unavailable_reason = "no_traffic"
                        break
                    if v_dedup["measurement_date"].max() != latest_shared_date:
                        # Same guarantee the main per-variant loop gets from its
                        # pd.merge(..., on="measurement_date") below -- Control and
                        # every variant must share their latest date before their
                        # cumulative counts are combined into one revenue figure.
                        revenue_unavailable_reason = "misaligned_dates"
                        break
                    v_latest = v_dedup.iloc[-1]
                    vis = int(v_latest["visitors"])
                    if vis <= 0:
                        revenue_unavailable_reason = "no_traffic"
                        break
                    batch_visitors.append(vis)
                    batch_conversions.append(int(v_latest["conversions"]))

        if revenue_unavailable_reason is None:
            try:
                overall_first = df["measurement_date"].min()
                overall_last = df["measurement_date"].max()
                runtime_days = max((overall_last - overall_first).days, 1)

                revenue_by_label = _compute_revenue_impact(
                    tuple(batch_visitors), tuple(batch_conversions), tuple(batch_labels),
                    aov_value, aov_cv, runtime_days,
                    expected_lift_pct, skepticism,
                )
            except (ValidationError, ValueError) as e:
                st.warning(f"Could not compute revenue impact — the current data doesn't fit the model: {e}")
                revenue_unavailable_reason = "error"
            except Exception:
                logger.exception("Unexpected error computing revenue impact")
                st.warning(
                    "Could not compute revenue impact due to an unexpected error. "
                    "This has been logged for investigation."
                )
                revenue_unavailable_reason = "error"

    for variant in variants_to_test:
        var_df = get_deduped_variant_df(df, variant)

        if test_type == TEST_TYPE_MULTI_SAMPLE:
            ctrl_df = get_deduped_variant_df(df, "Control")

            merged = pd.merge(
                var_df, ctrl_df, on="measurement_date", suffixes=("_var", "_ctrl")
            )

            if merged.empty:
                st.warning(
                    f"Waiting for aligned Control & Variant dates for {variant}."
                )
                continue

            merged["llr"] = merged.apply(
                lambda row: calculate_msprt_llr(
                    visitors_base=row["visitors_ctrl"],
                    conversions_base=row["conversions_ctrl"],
                    visitors_var=row["visitors_var"],
                    conversions_var=row["conversions_var"],
                    tau=tau_param,
                ),
                axis=1,
            )

            # Per-row power: use observed control CR at each date as the baseline
            def _power_two_sample(row):
                p_base = (
                    row["conversions_ctrl"] / row["visitors_ctrl"]
                    if row["visitors_ctrl"] > 0
                    else 0.1
                )
                power, _ = calculate_instantaneous_power(
                    n_var=row["visitors_var"],
                    p0=p_base,
                    mde=mde,
                    alpha=alpha,
                    n_ctrl=row["visitors_ctrl"],
                )
                return power

            merged["power"] = merged.apply(_power_two_sample, axis=1)

            latest_vis = merged.iloc[-1]["visitors_var"]
            latest_conv = merged.iloc[-1]["conversions_var"]
            base_vis = merged.iloc[-1]["visitors_ctrl"]
            base_cr = (
                merged.iloc[-1]["conversions_ctrl"] / base_vis if base_vis > 0 else 0
            )

        else:
            # One-sample logic
            merged = var_df.copy()
            if merged.empty:
                continue

            p0_param = params.get("p0", 0.10)
            merged["llr"] = merged.apply(
                lambda row: calculate_msprt_llr(
                    visitors_base=0,
                    conversions_base=0,
                    visitors_var=row["visitors"],
                    conversions_var=row["conversions"],
                    tau=tau_param,
                    fixed_baseline_cr=p0_param,
                ),
                axis=1,
            )

            merged["power"] = merged.apply(
                lambda row: calculate_instantaneous_power(
                    n_var=row["visitors"],
                    p0=p0_param,
                    mde=mde,
                    alpha=alpha,
                    n_ctrl=None,
                )[0],
                axis=1,
            )

            latest_vis = merged.iloc[-1]["visitors"]
            latest_conv = merged.iloc[-1]["conversions"]
            base_cr = p0_param

        merged["variant_name"] = variant
        vis_col = "visitors_var" if "visitors_var" in merged.columns else "visitors"
        chart_data.append(
            merged[["measurement_date", "variant_name", "llr", vis_col]]
        )
        power_data.append(
            merged[["measurement_date", "variant_name", "power"]]
        )

        # --- VARIANT DECISION CARDS ---
        latest_llr = merged.iloc[-1]["llr"]
        latest_cr = latest_conv / latest_vis if latest_vis > 0 else 0

        with st.expander(f"Metrics: {variant}", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Lower Bound (Futility)", f"{lower_bound:.2f}")
            col2.metric("Current LLR", f"{latest_llr:.2f}")
            col3.metric("Upper Bound (Success)", f"{upper_bound:.2f}")

            # Note Bonferroni adjustment when multiple variants are present
            if num_variants > 1:
                st.caption(
                    f"ℹ️ Upper bound raised from {np.log(1 / alpha):.2f} to {upper_bound:.2f} "
                    f"(Bonferroni correction for {num_variants} variants — controls family-wise false positive rate)."
                )

            st.write(
                f"**Observed CR:** {latest_cr:.2%} vs **Baseline CR:** {base_cr:.2%}"
            )

            # --- POWER METRICS ---
            latest_power = merged.iloc[-1]["power"]
            latest_beta_est = 1.0 - latest_power
            target_power = 1.0 - beta

            pm_col1, pm_col2, pm_col3 = st.columns(3)
            pm_col1.metric(
                "Target Power",
                f"{target_power:.0%}",
                help="1 − β as set when the experiment was locked.",
            )
            pm_col2.metric(
                "Estimated Power",
                f"{latest_power:.0%}",
                delta=f"{latest_power - target_power:+.0%} vs target",
                delta_color="normal",
            )
            pm_col3.metric(
                "Est. False Negative Risk",
                f"{latest_beta_est:.0%}",
                help="1 − estimated power at the current sample size.",
            )
            st.caption(
                "⚠️ Fixed-horizon approximation — mSPRT power is structurally lower. "
                "Use this as a directional signal, not a guarantee."
            )

            # --- PROGRESS BAR ---
            # Show progress toward whichever boundary the LLR is heading for.
            # A test heading toward futility is making a decision too — it deserves its own signal.
            if latest_llr >= 0:
                progress = min((latest_llr / upper_bound) * 100, 100)
                progress_label = "Progress to Success Boundary"
            else:
                # Both latest_llr and lower_bound are negative; their ratio is positive.
                progress = min((latest_llr / lower_bound) * 100, 100)
                progress_label = "Progress to Futility Boundary"

            st.write(f"**{progress_label}:** {progress:.0f}%")
            st.progress(progress / 100)

            # --- TIME ESTIMATION ---
            first_date = merged["measurement_date"].min()
            last_date = merged["measurement_date"].max()
            days_elapsed = max((last_date - first_date).days, 1)
            est_days = None

            if latest_llr > 0 and not (
                latest_llr > upper_bound or latest_llr < lower_bound
            ):
                avg_daily_visitors = latest_vis / days_elapsed
                llr_per_vis = latest_llr / latest_vis
                remaining_llr = upper_bound - latest_llr

                if llr_per_vis > 0 and avg_daily_visitors > 0:
                    est_vis_needed = remaining_llr / llr_per_vis
                    est_days = est_vis_needed / avg_daily_visitors
                    st.info(
                        f"**Rough Estimate:** Assuming linear growth, you need approx. "
                        f"**{est_vis_needed:.0f}** more visitors (**{est_days:.1f} days**) to reach success."
                    )

            # --- DECISION LOGIC ---
            if latest_llr > upper_bound:
                st.success(
                    f"**Result: SIGNIFICANT POSITIVE** - {variant} is superior. You can stop."
                )
            elif latest_llr < lower_bound:
                relative_diff = (
                    (latest_cr - base_cr) / base_cr if base_cr > 0 else 0
                )
                if relative_diff < -0.05:
                    st.error(
                        f"**Result: NEGATIVE IMPACT** - {variant} is noticeably worse than the baseline "
                        f"({relative_diff:.1%} relative drop). Stop testing."
                    )
                elif relative_diff < 0:
                    st.warning(
                        f"**Result: FLAT / FUTILITY** - {variant} is practically tied with the baseline. "
                        f"It will not reach your target lift."
                    )
                else:
                    st.warning(
                        f"**Result: FLAT / FUTILITY** - {variant} is slightly ahead, "
                        f"but will not reach your target lift."
                    )
            else:
                if latest_vis >= max_visitors:
                    st.warning("Maximum sample size reached without a decision.")
                else:
                    st.info("INCONCLUSIVE - Continue collecting data.")

            # --- REVENUE IMPACT (via the shared FOE BayesianEngine) ---
            revenue_result = revenue_by_label.get(variant)
            if revenue_enabled and revenue_result is not None:
                st.divider()
                st.markdown("##### 💰 Revenue Impact")

                rev_col1, rev_col2, rev_col3 = st.columns(3)
                rev_col1.metric(
                    f"Value if you stop now ({BUSINESS_PROJECTION_DAYS}d)",
                    f"€{revenue_result['expected_total_contribution']:,.0f}",
                )
                rev_col2.metric("Expected Uplift", f"€{revenue_result['expected_uplift']:,.0f}")
                # expected_risk is always a non-negative magnitude (see
                # BayesianEngine.run_monetary_projection) -- displayed as a plain
                # positive figure, matching the "cost of waiting" caption below and
                # the engine's own conclusion text, both of which use the word
                # "risk"/"downside" to carry the negative connotation instead of a sign.
                rev_col3.metric("Expected Risk", f"€{revenue_result['expected_risk']:,.0f}")
                st.caption(revenue_result["conclusion"])

                if est_days is not None:
                    # expected_uplift/risk scale linearly with the projection period, so
                    # rescaling by est_days/BUSINESS_PROJECTION_DAYS recovers the same
                    # per-day rate the engine used, without re-simulating.
                    scale = est_days / BUSINESS_PROJECTION_DAYS
                    waiting_gain = revenue_result["expected_uplift"] * scale
                    waiting_risk = revenue_result["expected_risk"] * scale
                    st.info(
                        f"⏳ **Cost of waiting:** continuing for the estimated **{est_days:.1f} more "
                        f"days** to reach a decision puts roughly **€{waiting_gain:,.0f} of upside** "
                        f"and **€{waiting_risk:,.0f} of downside** on the table before you know the "
                        f"outcome."
                    )
            elif revenue_enabled:
                if revenue_unavailable_reason == "misaligned_dates":
                    st.caption(
                        "💡 Revenue Impact is paused: Control and this variant's latest "
                        "entries are on different dates. Add matching data points for both "
                        "to re-enable it."
                    )
                elif revenue_unavailable_reason == "error":
                    st.caption(
                        "💡 Revenue Impact couldn't be computed this time — see the message "
                        "above."
                    )
                else:
                    st.caption(
                        "💡 Revenue Impact needs at least one visitor recorded for every "
                        "variant (including Control) to compute a posterior."
                    )

    if chart_data:
        final_chart_df = pd.concat(chart_data)
        if "visitors_var" in final_chart_df.columns:
            final_chart_df = final_chart_df.rename(
                columns={"visitors_var": "visitors"}
            )
        show_visualization(final_chart_df, upper_bound, lower_bound)

    if power_data:
        final_power_df = pd.concat(power_data)
        target_power = 1.0 - beta
        show_power_chart(final_power_df, target_power)


# --- DOCUMENTATION ---

def show_documentation():
    with st.expander("Benefits of mSPRT", expanded=False):
        st.markdown("""
        #### Benefits of mSPRT
        Compared to fixed-horizon testing, mSPRT has certain advantages.
        * **Stop winners early:** Deploy successful features days or weeks faster.
        * **Cut losers fast:** Identify "futility" (no chance of winning) early to save traffic and/or money.
        * **Rigorous:** Mathematically valid stopping rules, unlike standard z-tests (The mSPRT boundaries are "always-valid.").
        """)

    with st.expander("Tradeoffs & Limitations of mSPRT", expanded=False):
        st.markdown("""
        #### What mSPRT costs you: statistical power

        The always-valid peeking guarantee is not free. To be valid at *every* point in time,
        mSPRT requires wider decision boundaries than a fixed-horizon test at the same α and β.
        In practice, this means:

        * **More observations on average** to reach a decision when the true effect is small-to-moderate.
          At typical CRO lifts (1–5% relative), expect to need roughly 20–50% more total traffic
          compared to an equivalently-powered fixed-horizon test.
        * **MDE calibration matters.** The sensitivity parameter τ is derived from your Minimum
          Detectable Effect and should match the lift you actually expect to find. If your MDE is
          set too high relative to the real effect, power drops further.
          A rough rule: τ = MDE² (absolute lift, squared). This tool derives τ for you automatically.
        * **Large effects are where mSPRT wins.** When an experiment turns out to have a big effect,
          mSPRT stops early and more than compensates. The power deficit hurts most when effects are
          small and you have to run all the way to the maximum sample size anyway.

        #### When is fixed-horizon the more efficient choice?
        If your primary goal is detecting a small lift with the fewest possible observations, a
        fixed-horizon test is structurally more efficient. The tradeoff above is not a calibration
        artifact — it is a mathematical consequence of always-valid inference. You are paying for
        the right to peek continuously.
        """)

    with st.expander("When to use mSPRT", expanded=False):
        st.markdown("""
        ### When to use mSPRT
        Sequential probability ratio testing is an agile tool. You should use it when:
        * **Safety is a concern:** You want to kill a 'losing' experiment immediately if it's tanking metrics.
        * **The observed effect is huge:** If the new feature is a massive success, mSPRT will let you ship it in (for example) 3 days instead of 14.
        * **The cost of testing is high:** If every user in the experiment costs money, stopping early saves budget.
        * **If you don't believe in p-values:** This tool looks for the Log-Likelihood ratio to cross upper- and lower boundaries, calculated from your alpha and beta values (still technically frequentist, but practical).

        ### When to use fixed-horizon testing
        * If you have a **strict deadline**.
        * mSPRT requires **more average observations** to reach the same power as a fixed-horizon test —
          a structural tradeoff for the always-valid peeking guarantee.
          For small targeted lifts (< 5% relative), this gap is meaningful.
        * If stakeholders are better aligned with clear deadlines and mid- to long-term planning of experiments.
        """)

    with st.expander("How to use mSPRT", expanded=False):
        st.markdown("""
        #### How to use mSPRT
        1.  **Start New:** Generate a unique ID and define your success metrics (Alpha / significance, Beta / power).
            * *Note: These are locked once the test starts to ensure integrity.*
        2.  **Update Regularly:** Come back daily/weekly to input your **cumulative** data.
        3.  **Check the Graph:**
            * **Upper Limit:** Success! (Reject Null)
            * **Lower Limit:** Futility/Failure. (Accept Null)
            * **In Between lines:** Inconclusive - keep testing.

        > * **Important:** Data is stored for **42 days (6 weeks)** and then automatically deleted.
        > * **Save your Experiment ID!** It is the only key to retrieve your data.
        """)


# --- USER INPUT ---

def setup_sidebar(defaults, is_locked):
    with st.sidebar:
        st.header("1. Experiment Setup")

        # 1. Test Type Selection
        options = [TEST_TYPE_ONE_SAMPLE, TEST_TYPE_MULTI_SAMPLE]
        saved_type = defaults.get("test_type", options[0])
        default_index = options.index(saved_type) if saved_type in options else 0

        test_type = st.radio(
            "Test format", options, index=default_index, disabled=is_locked
        )

        if test_type == TEST_TYPE_MULTI_SAMPLE:
            num_variants_val = int(defaults.get("num_variants", 1))
            num_variants = st.number_input(
                "Number of Variants (excluding Control)",
                min_value=1,
                max_value=10,
                value=num_variants_val,
                step=1,
                disabled=is_locked,
            )
            p0_label = "Estimated baseline CR (p0)"
            p0_help = "Used only for sample size estimates. Control CR will be measured live."
        else:
            num_variants = 1
            p0_label = "Baseline CR (p0)"
            p0_help = "The fixed historical conversion rate (CR) you want to beat."

        # 2. Mode Selection
        mode = st.radio(
            "Mode",
            ["Load Existing", "Start New"],
            label_visibility="collapsed",
        )

        # 3. ID Management
        if mode == "Start New":
            if st.button("Generate New ID"):
                st.session_state["exp_id"] = str(uuid.uuid4())
                st.session_state["params_locked"] = False
                st.session_state["fetched_params"] = {}
                for key in REVENUE_SETTINGS_KEYS:
                    st.session_state.pop(key, None)
                st.rerun()

            if st.session_state.get("exp_id"):
                st.success("Save this ID to load your test later:")
                st.code(st.session_state["exp_id"])
        else:
            input_id = st.text_input("Paste Experiment UUID")
            if st.button("Load"):
                if is_valid_uuid(input_id):
                    st.session_state["exp_id"] = input_id
                    params = get_experiment_params(input_id)
                    if params:
                        st.session_state["fetched_params"] = params
                        st.session_state["params_locked"] = True
                        for key in REVENUE_SETTINGS_KEYS:
                            st.session_state.pop(key, None)
                        st.toast("Parameters loaded and locked!", icon="🔒")
                    else:
                        st.error("Experiment ID not found or no parameters set.")
                    st.rerun()
                else:
                    st.error("Invalid UUID format.")

        st.divider()

        # --- PARAMETER INPUTS ---
        st.subheader("2. Test Parameters")

        if is_locked:
            st.info("Parameters are locked for this ID.")

        p0_param = float(defaults.get("p0", 0.10))

        with st.form("setup_form"):
            alpha_val = float(defaults.get("alpha", 0.05))
            beta_val = float(defaults.get("beta", 0.20))
            max_visitors_val = int(defaults.get("max_visitors", 10000))

            if test_type == TEST_TYPE_ONE_SAMPLE:
                p0_param = st.number_input(
                    p0_label,
                    value=p0_param,
                    format="%.4f",
                    disabled=is_locked,
                    help=p0_help,
                )
            else:
                st.caption(
                    f"**{p0_label}:** Not required for strict calculations in Multi-sample mode."
                )
                p0_param = 0.01

            # Derive display MDE from stored tau (sqrt), or use a sensible default.
            # Default tau 0.0004 = MDE 0.02 (a 2% absolute lift), a reasonable CRO starting point.
            saved_tau = float(defaults.get("tau") or 0.0004)
            default_mde = float(np.sqrt(saved_tau))

            mde_input = st.number_input(
                "Minimum Detectable Effect — MDE (absolute)",
                value=default_mde,
                min_value=0.0001,
                format="%.4f",
                disabled=is_locked,
                help=(
                    "The smallest absolute lift in conversion rate you want to reliably detect. "
                    "Example: to detect a move from 10% → 12% CR, enter 0.02. "
                    "τ is derived as MDE² and is the single biggest lever on statistical power — "
                    "a miscalibrated MDE quietly increases the sample size you need."
                ),
            )

            # Tau is always derived from MDE; never set manually.
            tau_param = mde_input ** 2
            st.caption(
                f"Derived τ = {tau_param:.6f} — stored and used in all calculations."
            )

            max_visitors = st.number_input(
                "Max Visitors (Safety Cap)",
                value=max_visitors_val,
                step=100,
                disabled=is_locked,
            )

            c1, c2 = st.columns(2)
            alpha = c1.number_input(
                "Alpha",
                value=alpha_val,
                step=0.01,
                disabled=is_locked,
                help="The risk of a False Positive.",
            )
            beta = c2.number_input(
                "Beta",
                value=beta_val,
                step=0.01,
                disabled=is_locked,
                help="The risk of a False Negative.",
            )

            if not is_locked:
                submitted = st.form_submit_button("Start & Lock Experiment")
                if submitted:
                    if not st.session_state.get("exp_id"):
                        st.error("Generate an ID first!")
                    elif tau_param <= 0:
                        st.error("MDE must be greater than 0.")
                    else:
                        saved = save_experiment_params(
                            st.session_state["exp_id"],
                            p0_param,
                            tau_param,
                            alpha,
                            beta,
                            max_visitors,
                            test_type,
                            num_variants,
                        )
                        if saved:
                            st.session_state["params_locked"] = True
                            st.session_state["fetched_params"] = {
                                "p0": p0_param,
                                "tau": tau_param,
                                "alpha": alpha,
                                "beta": beta,
                                "max_visitors": max_visitors,
                                "test_type": test_type,
                                "num_variants": num_variants,
                            }
                            st.rerun()
            else:
                st.form_submit_button("Parameters Locked", disabled=True)

    return st.session_state.get("fetched_params", {})


def render_data_entry_form(exp_id, df, params):
    st.subheader("Update Data")
    st.info(
        "💡 **Reminder:** Enter the **cumulative** totals up to this date, not just the daily increment."
    )

    current_test_type = params.get("test_type", TEST_TYPE_ONE_SAMPLE)
    num_variants = params.get("num_variants", 1)

    with st.form("entry_form"):
        d_date = st.date_input("Date")
        variant_data_list = []

        def get_prev(v_name):
            if not df.empty:
                v_df = df[df["variant_name"] == v_name]
                if not v_df.empty:
                    return int(v_df.iloc[-1]["visitors"]), int(
                        v_df.iloc[-1]["conversions"]
                    )
            return 0, 0

        if current_test_type == TEST_TYPE_MULTI_SAMPLE:
            st.divider()
            st.markdown("### Control Group")
            p_vis_c, p_conv_c = get_prev("Control")
            c1, c2 = st.columns(2)
            d_vis_c = c1.number_input(
                f"Cumulative Visitors (Prev: {p_vis_c})",
                min_value=p_vis_c,
                value=p_vis_c,
                key="ctrl_v",
            )
            d_conv_c = c2.number_input(
                f"Cumulative Conversions (Prev: {p_conv_c})",
                min_value=p_conv_c,
                value=p_conv_c,
                key="ctrl_c",
            )
            variant_data_list.append(
                {"variant_name": "Control", "visitors": d_vis_c, "conversions": d_conv_c}
            )

            st.markdown("### Variant Groups")
            for i in range(1, num_variants + 1):
                v_name = f"Variant {i}"
                p_vis, p_conv = get_prev(v_name)
                c3, c4 = st.columns(2)
                d_vis = c3.number_input(
                    f"{v_name} Visitors (Prev: {p_vis})",
                    min_value=p_vis,
                    value=p_vis,
                    key=f"v{i}_v",
                )
                d_conv = c4.number_input(
                    f"{v_name} Conversions (Prev: {p_conv})",
                    min_value=p_conv,
                    value=p_conv,
                    key=f"v{i}_c",
                )
                variant_data_list.append(
                    {"variant_name": v_name, "visitors": d_vis, "conversions": d_conv}
                )

        else:  # One-sample
            st.divider()
            st.markdown("### Variant Data")
            p_vis, p_conv = get_prev("Variant 1")
            c1, c2 = st.columns(2)
            d_vis = c1.number_input(
                f"Cumulative Visitors (Prev: {p_vis})",
                min_value=p_vis,
                value=p_vis,
                key="1s_v",
            )
            d_conv = c2.number_input(
                f"Cumulative Conversions (Prev: {p_conv})",
                min_value=p_conv,
                value=p_conv,
                key="1s_c",
            )
            variant_data_list.append(
                {"variant_name": "Variant 1", "visitors": d_vis, "conversions": d_conv}
            )

        st.divider()
        if st.form_submit_button("Add Data Point"):
            if any(
                item["visitors"] < item["conversions"] for item in variant_data_list
            ):
                st.error("Visitors cannot be less than conversions for any group.")
            else:
                save_data_points(exp_id, d_date, variant_data_list)
                st.rerun()


# --- UI LOGIC / ORCHESTRATION ---

def run():
    st.title("Sequential Experiment Analysis (SPRT)")
    st.markdown("""
    ### Faster A/B Testing with Sequential Analysis
    Standard A/B tests require you to wait for a fixed sample size to avoid "peeking" errors.
    **This tool is different.** It uses **mixture Sequential Probability Ratio Testing (mSPRT)**,
    allowing you to update data and check results **any time** without invalidating your statistics.
    """)

    show_documentation()

    if "params_locked" not in st.session_state:
        st.session_state["params_locked"] = False
    if "fetched_params" not in st.session_state:
        st.session_state["fetched_params"] = {}

    defaults = st.session_state.get("fetched_params", {})
    is_locked = st.session_state.get("params_locked", False)

    # --- SIDEBAR ---
    params = setup_sidebar(defaults, is_locked)

    # --- MAIN CONTENT ---
    exp_id = st.session_state.get("exp_id")

    if not exp_id or not st.session_state.get("params_locked"):
        st.info("**To Begin:** Select 'Start New' to generate an ID and lock your parameters.")
        st.stop()

    st.markdown(f"### Experiment: `{exp_id}`")

    df = get_experiment_data(exp_id)

    # --- DATA ENTRY ---
    render_data_entry_form(exp_id, df, params)

    # --- UNDO FUNCTIONALITY ---
    if not df.empty:
        last_date = df["measurement_date"].max()
        with st.expander("Danger Zone: Undo Entries"):
            st.warning("Deleting data cannot be undone.")
            if st.button(
                f"Delete ALL variant entries for {last_date}", type="primary"
            ):
                delete_data_points_by_date(exp_id, last_date)
                st.rerun()

    # --- ANALYSIS SECTION ---
    if not df.empty:
        analysis_section(df, params)

        with st.expander("View Raw Data"):
            st.dataframe(df.sort_values("measurement_date", ascending=False))
    else:
        st.write("Waiting for data entries...")


if __name__ == "__main__":
    run()
