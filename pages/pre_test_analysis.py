from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import zt_ind_solve_power
from prophet import Prophet  # type: ignore[import-untyped]  # Prophet stubs are incomplete

st.set_page_config(
    page_title="Pre-test analysis",
    page_icon="🔢",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BaselineInputs:
    num_variants: int
    baseline_visitors: int
    baseline_conversions: int
    risk: float          # e.g. 95  (percentage)
    trust: float         # e.g. 80  (percentage)
    tails: str           # 'One-sided' | 'Two-sided'


@dataclass
class MDERow:
    week: int
    visitors_per_variant: int
    relative_mde_pct: float


@dataclass
class PowerResult:
    power: float
    target_power: float
    rate_b: float


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def holm_bonferroni_z(num_comparisons: int, alpha: float, tails: str) -> float:
    """
    Return the most conservative critical z-value under the Holm-Bonferroni
    step-down procedure.

    The Holm procedure orders hypotheses by p-value, adjusting alpha at each
    step as alpha / (m - rank + 1).  For pre-test planning we don't yet have
    p-values, so we conservatively use the *first* (most stringent) step:
        adjusted_alpha = alpha / num_comparisons

    This is identical to simple Bonferroni for the first comparison and is
    the standard conservative choice when planning sample size.
    """
    adjusted_alpha = alpha / num_comparisons
    if tails == "Two-sided":
        return float(norm.ppf(1 - adjusted_alpha / 2))
    return float(norm.ppf(1 - adjusted_alpha))


def get_critical_z(num_variants: int, alpha: float, tails: str) -> float:
    """Return the adjusted critical z-value, applying Holm-Bonferroni when
    there are more than two variants."""
    if num_variants > 2:
        return holm_bonferroni_z(num_variants - 1, alpha, tails)
    if tails == "Two-sided":
        return float(norm.ppf(1 - alpha / 2))
    return float(norm.ppf(1 - alpha))


def mde_from_visitors(
    visitors_per_variant: float,
    baseline_rate: float,
    z_alpha: float,
    z_power: float,
) -> float:
    """
    Two-proportion z-test MDE formula.

    Under H₀ both variants share `baseline_rate`, so the pooled SE is:
        SE = sqrt(2 * p * (1 - p) / n)
    The factor of 2 comes from adding the variance of two equal-sized groups.
    """
    se = float(np.sqrt(2 * baseline_rate * (1 - baseline_rate) / visitors_per_variant))
    mde_absolute = (z_alpha + z_power) * se
    return float((mde_absolute / baseline_rate) * 100)  # relative MDE as a percentage


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

def perform_mde_calculation(
    inputs: BaselineInputs,
    traffic_multiplier: float = 1.0,
) -> list[MDERow]:
    """
    Calculate relative MDE for weeks 1–6.

    `traffic_multiplier` scales the weekly visitor volume to model traffic
    scenarios (e.g. 0.7 = −30%, 1.3 = +30%).
    """
    alpha = 1 - (inputs.risk / 100)
    power = inputs.trust / 100
    baseline_rate = inputs.baseline_conversions / inputs.baseline_visitors

    z_alpha = get_critical_z(inputs.num_variants, alpha, inputs.tails)
    z_power = float(norm.ppf(power))

    weekly_visitors = int(
        np.ceil(inputs.baseline_visitors * traffic_multiplier / inputs.num_variants)
    )

    results: list[MDERow] = []
    for week in range(1, 7):
        visitors_per_variant = weekly_visitors * week
        relative_mde = mde_from_visitors(
            visitors_per_variant, baseline_rate, z_alpha, z_power
        )
        results.append(MDERow(week, visitors_per_variant, relative_mde))

    return results


def perform_mde_calculation_forecast(
    forecast_df: pd.DataFrame,
    inputs: BaselineInputs,
) -> list[MDERow]:
    """
    Calculate MDE using accumulated forecasted traffic rather than a static
    weekly average, accounting for seasonality.
    """
    alpha = 1 - (inputs.risk / 100)
    power = inputs.trust / 100

    z_alpha = get_critical_z(inputs.num_variants, alpha, inputs.tails)
    z_power = float(norm.ppf(power))

    results: list[MDERow] = []
    for week in range(1, 7):
        current_slice = forecast_df.head(week * 7)
        total_visitors = current_slice["pred_visitors"].sum()
        total_conversions = current_slice["pred_conversions"].sum()

        if total_visitors <= 0:
            results.append(MDERow(week, 0, float("nan")))
            continue

        baseline_rate = float(np.clip(total_conversions / total_visitors, 1e-4, 1 - 1e-4))
        visitors_per_variant = total_visitors / inputs.num_variants

        relative_mde = mde_from_visitors(
            visitors_per_variant, baseline_rate, z_alpha, z_power
        )
        results.append(MDERow(week, int(visitors_per_variant), relative_mde))

    return results


def calculate_sample_size_required(
    inputs: BaselineInputs,
    mde_pct: float,
) -> tuple[int, int | str]:
    """
    Return (sample_size_per_group, estimated_days).

    Uses the two-proportion z-test formula consistent with `perform_mde_calculation`.
    estimated_days is an int on success, or a descriptive string on failure.
    """
    alpha = 1 - (inputs.risk / 100)
    power = inputs.trust / 100
    p1 = inputs.baseline_conversions / inputs.baseline_visitors
    mde_absolute = p1 * (mde_pct / 100)
    p2 = p1 + mde_absolute

    z_alpha = get_critical_z(inputs.num_variants, alpha, inputs.tails)
    z_power = float(norm.ppf(power))

    var1 = p1 * (1 - p1)
    var2 = float(np.clip(p2 * (1 - p2), 0, None))

    term1 = z_alpha * np.sqrt(2 * p1 * (1 - p1))
    term2 = z_power * np.sqrt(var1 + var2)

    sample_size = int(np.ceil(((term1 + term2) ** 2) / (mde_absolute ** 2)))

    # Estimate runtime
    daily_per_group = (inputs.baseline_visitors / 7) / inputs.num_variants
    if daily_per_group <= 0:
        estimated_days: int | str = "infinite (zero daily visitors per group)"
    else:
        estimated_days = int(np.ceil(sample_size / daily_per_group))

    return sample_size, estimated_days


def calculate_power(
    inputs: BaselineInputs,
    expected_lift_pct: float,
    weeks_to_run: int,
) -> PowerResult:
    """
    Calculate statistical power using the two-proportion z-test (via
    `zt_ind_solve_power`) with Holm-Bonferroni correction when num_variants > 2.

    This is consistent with the MDE formula used throughout the tool:
    both are grounded in the two-proportion z-test framework.
    """
    alpha = 1 - (inputs.risk / 100)
    target_power = inputs.trust / 100

    # Apply Holm-Bonferroni: for power planning we use the first (most
    # conservative) step, i.e. alpha / num_comparisons.
    num_comparisons = inputs.num_variants - 1
    if num_comparisons > 1:
        alpha = alpha / num_comparisons

    rate_a = inputs.baseline_conversions / inputs.baseline_visitors
    rate_b = rate_a * (1 + expected_lift_pct / 100)
    rate_b = np.clip(rate_b, 0, 1)

    # Effect size as Cohen's h (arcsine transformation), required by
    # zt_ind_solve_power which models proportion differences via the
    # normal approximation to the binomial.
    effect_size_h = abs(proportion_effectsize(rate_a, rate_b))

    alternative = "two-sided" if inputs.tails == "Two-sided" else "larger"
    n_per_variant = inputs.baseline_visitors * weeks_to_run

    power = zt_ind_solve_power(
        effect_size=effect_size_h,
        nobs1=n_per_variant,
        ratio=1.0,
        alpha=alpha,
        alternative=alternative,
    )

    return PowerResult(power=float(power), target_power=float(target_power), rate_b=float(rate_b))


# ---------------------------------------------------------------------------
# Prophet forecasting
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_prophet_forecast(
    df: pd.DataFrame,
    periods: int = 42,
    interval_width: float = 0.95,
) -> pd.DataFrame:
    """Fit Prophet models on visitors and conversions and return a 6-week
    daily forecast.  Results are cached by Streamlit based on the DataFrame
    hash; note that hashing large DataFrames may add overhead."""

    def _fit_predict(series_df: pd.DataFrame, col: str) -> pd.DataFrame:
        m = Prophet(  # type: ignore[call-arg]  # Prophet stubs are incomplete
            yearly_seasonality=True,  # type: ignore[arg-type]  # stub types seasonality as str; bool is valid at runtime
            weekly_seasonality=True,  # type: ignore[arg-type]
            interval_width=interval_width,
        )
        m.fit(series_df[["ds", col]].rename(columns={col: "y"}))
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        last_date = series_df["ds"].max()
        return (
            forecast[forecast["ds"] > last_date][["ds", "yhat", "yhat_lower", "yhat_upper"]]
            .rename(columns={
                "yhat": f"pred_{col}",
                "yhat_lower": f"{col[:3]}_lower",
                "yhat_upper": f"{col[:3]}_upper",
            })
        )

    future_vis = _fit_predict(df, "visitors")
    future_conv = _fit_predict(df, "conversions")

    forecast_final = pd.merge(future_vis, future_conv, on="ds")

    clip_cols = [
        "pred_visitors", "vis_lower", "vis_upper",
        "pred_conversions", "con_lower", "con_upper",
    ]
    for col in clip_cols:
        if col in forecast_final.columns:
            forecast_final[col] = forecast_final[col].clip(lower=0)

    return forecast_final


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _mde_color_scale() -> list[list[float | str]]:
    """Green → yellow → red colorscale mapped to MDE 0–20%+."""
    return [
        [0.00, "rgb(56, 161, 105)"],
        [0.25, "rgb(154, 205, 90)"],
        [0.50, "rgb(236, 201, 75)"],
        [0.75, "rgb(237, 137, 54)"],
        [1.00, "rgb(229, 62, 62)"],
    ]


def validate_baseline(
    baseline_visitors: int,
    baseline_conversions: int,
    risk: float,
    trust: float,
    tails: str,
) -> str | None:
    """Return an error message string if inputs are invalid, else None."""
    if baseline_visitors <= 0:
        return "Baseline visitors must be greater than 0."
    if baseline_conversions < 0:
        return "Baseline conversions cannot be negative."
    if baseline_conversions > baseline_visitors:
        return "Baseline conversions cannot exceed baseline visitors."
    if not (0 < risk <= 100):
        return "Confidence level must be between 0 and 100."
    if not (0 < trust <= 100):
        return "Power must be between 0 and 100."
    if tails not in ("One-sided", "Two-sided"):
        return "Hypothesis type must be 'One-sided' or 'Two-sided'."
    return None


def get_baseline_inputs(include_mde: bool = False, include_weeks: bool = False) -> BaselineInputs:
    """Render common input widgets and return a BaselineInputs dataclass.
    Widget state is read from / persisted in st.session_state."""
    st.write("### Baseline Data")
    st.write("Enter weekly visitors, weekly conversions and test parameters.")

    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "Number of variants (including control):",
            min_value=2, step=1,
            value=st.session_state.get("num_variants", 2),
            key="num_variants",
        )
        st.number_input(
            "Visitors in baseline variant:",
            min_value=0, step=1,
            value=st.session_state.get("baseline_visitors", 0),
            key="baseline_visitors",
        )
        st.number_input(
            "Conversions in baseline variant:",
            min_value=0, step=1,
            value=st.session_state.get("baseline_conversions", 0),
            key="baseline_conversions",
        )
    with col2:
        st.number_input(
            "Desired confidence level (e.g., 95%):",
            min_value=0, max_value=100, step=1,
            value=st.session_state.get("risk", 95),
            key="risk",
        )
        st.number_input(
            "Minimum trustworthiness / Power (e.g., 80%):",
            min_value=0, max_value=100, step=1,
            value=st.session_state.get("trust", 80),
            key="trust",
        )
        if include_mde:
            st.number_input(
                "What MDE / expected lift are you aiming for (%):",
                min_value=1, max_value=100, step=1,
                value=st.session_state.get("mde", 5),
                key="mde",
            )
        if include_weeks:
            st.slider(
                "Test Duration (Weeks)",
                min_value=1, max_value=6, value=4,
                key="weeks_to_run",
            )

    st.radio(
        "Hypothesis type:",
        options=["One-sided", "Two-sided"],
        index=["One-sided", "Two-sided"].index(
            st.session_state.get("tails", "One-sided")
        ),
        horizontal=True,
        key="tails",
        help=(
            "Choose **One-sided** when testing only for improvement (B > A) or decline (B < A). "
            "Choose **Two-sided** when testing for any difference regardless of direction."
        ),
    )

    return BaselineInputs(
        num_variants=st.session_state["num_variants"],
        baseline_visitors=st.session_state["baseline_visitors"],
        baseline_conversions=st.session_state["baseline_conversions"],
        risk=st.session_state["risk"],
        trust=st.session_state["trust"],
        tails=st.session_state["tails"],
    )


# ---------------------------------------------------------------------------
# Mode renderers
# ---------------------------------------------------------------------------

def render_mde_mode() -> None:
    inputs = get_baseline_inputs()

    if st.button("Calculate MDE", type="primary"):
        error = validate_baseline(
            inputs.baseline_visitors, inputs.baseline_conversions,
            inputs.risk, inputs.trust, inputs.tails,
        )
        if error:
            st.error(error)
        else:
            st.session_state["mde_results"] = perform_mde_calculation(inputs)
            st.session_state["mde_inputs"] = inputs

    if "mde_results" in st.session_state:
        _display_mde_table(
            st.session_state["mde_inputs"],
            st.session_state["mde_results"],
        )


def _display_mde_table(inputs: BaselineInputs, results: list[MDERow]) -> None:
    st.write("## MDE Calculation Results")
    st.write(
        "This table shows the smallest relative effect detectable each week. "
        "An MDE below 5% is generally testworthy; 5–10% is debatable."
    )
    if inputs.num_variants > 2:
        st.info(
            f"Holm-Bonferroni correction applied "
            f"({inputs.num_variants - 1} comparisons), tightening the required significance level."
        )

    tab_standard, tab_sensitivity = st.tabs(["Standard MDE Table", "Sensitivity Matrix"])

    with tab_standard:
        df = pd.DataFrame(
            {
                "Week": [r.week for r in results],
                "Visitors / Variant": [r.visitors_per_variant for r in results],
                "Relative MDE (%)": [f"{r.relative_mde_pct:.2f}%" for r in results],
            }
        )
        st.write(df.to_html(index=False), unsafe_allow_html=True)

    with tab_sensitivity:
        st.write(
            "**How would traffic spikes or drops affect your MDE?**  "
            "Each cell shows the Relative MDE (%) for a given week and traffic scenario. "
            "Colors: 🟢 <5% · 🟡 5–10% · 🔴 >10%."
        )

        multipliers = [0.50, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.50]
        scenario_labels = [
            f"{int((m - 1) * 100):+d}%" if m != 1.00 else "Baseline"
            for m in multipliers
        ]
        weeks = list(range(1, 7))

        mde_matrix = [
            [
                round(
                    perform_mde_calculation(inputs, traffic_multiplier=m)[week_idx].relative_mde_pct,
                    2,
                )
                for m in multipliers
            ]
            for week_idx in range(6)
        ]

        z = np.array(mde_matrix)
        text_labels = [[f"{v:.1f}%" for v in row] for row in mde_matrix]
        y_labels = [f"Week {w}" for w in weeks]
        z_capped = np.clip(z, 0, 20)

        fig = go.Figure(
            data=go.Heatmap(
                z=z_capped,
                x=scenario_labels,
                y=y_labels,
                text=text_labels,
                texttemplate="%{text}",
                textfont={"size": 13, "color": "white"},
                colorscale=_mde_color_scale(),
                zmin=0,
                zmax=20,
                showscale=True,
                colorbar=dict(
                    title="MDE (%)",
                    tickvals=[0, 5, 10, 15, 20],
                    ticktext=["0%", "5%", "10%", "15%", "≥20%"],
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Traffic scenario: <b>%{x}</b><br>"
                    "Relative MDE: <b>%{text}</b><extra></extra>"
                ),
            )
        )

        baseline_col_idx = scenario_labels.index("Baseline")
        fig.add_shape(
            type="rect",
            x0=baseline_col_idx - 0.5,
            x1=baseline_col_idx + 0.5,
            y0=-0.5,
            y1=len(weeks) - 0.5,
            line=dict(color="white", width=2.5),
            fillcolor="rgba(0,0,0,0)",
        )

        fig.update_layout(
            xaxis=dict(
                title="Traffic vs. Baseline",
                side="bottom",
                type="category",
                tickfont=dict(size=12),
                automargin=True,
            ),
            yaxis=dict(
                title="",
                tickfont=dict(size=12),
                autorange="reversed",
            ),
            margin=dict(l=80, r=60, t=40, b=80),
            height=340,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Traffic multipliers apply uniformly across all days in a given week. "
            "For week-by-week seasonality modelling, use the **Seasonal (Prophet Forecast)** mode."
        )


def render_sample_size_mode() -> None:
    inputs = get_baseline_inputs(include_mde=True)

    if st.button("Calculate Sample Size", type="primary"):
        error = validate_baseline(
            inputs.baseline_visitors, inputs.baseline_conversions,
            inputs.risk, inputs.trust, inputs.tails,
        )
        mde = st.session_state.get("mde", 5)

        if error:
            st.error(error)
        elif mde <= 0:
            st.error("MDE must be greater than 0%.")
        else:
            p1 = inputs.baseline_conversions / inputs.baseline_visitors
            p2 = p1 * (1 + mde / 100)
            if p2 > 1.0:
                st.warning(
                    f"The expected treatment conversion rate ({p2:.2%}) exceeds 100% "
                    f"given the baseline rate ({p1:.2%}) and MDE ({mde}%). "
                    "Please review your inputs."
                )

            sample_size, estimated_days = calculate_sample_size_required(inputs, mde)
            st.session_state["ss_result"] = (sample_size, estimated_days, inputs, mde)

    if "ss_result" in st.session_state:
        sample_size, estimated_days, inputs, mde = st.session_state["ss_result"]

        st.write("## Sample Size Calculation Results")
        st.write(f"Required sample size for a desired relative MDE of **{mde}%**.")

        if inputs.num_variants > 2:
            st.info(
                f"Holm-Bonferroni correction applied "
                f"({inputs.num_variants - 1} comparisons), tightening the required significance level."
            )

        st.write(
            f"The required sample size per group (including control) is **{sample_size:,}**."
        )
        if isinstance(estimated_days, int):
            st.write(
                f"With **{inputs.baseline_visitors:,}** total visitors per week, "
                f"your test is estimated to run for approximately **{estimated_days} days** "
                "to reach the required sample size per group."
            )
        else:
            st.write(f"Estimated runtime: {estimated_days}.")


def render_power_mode() -> None:
    inputs = get_baseline_inputs(include_mde=True, include_weeks=True)

    if st.button("Calculate Power", type="primary"):
        error = validate_baseline(
            inputs.baseline_visitors, inputs.baseline_conversions,
            inputs.risk, inputs.trust, inputs.tails,
        )
        mde = st.session_state.get("mde", 5)
        weeks_to_run = st.session_state.get("weeks_to_run", 4)

        if error:
            st.error(error)
        elif mde <= 0:
            st.error("Expected lift must be greater than 0%.")
        else:
            result = calculate_power(inputs, mde, weeks_to_run)
            if result.rate_b > 1.0:
                st.warning(
                    f"An expected lift of {mde}% pushes the variant conversion rate over 100%. "
                    "The rate has been capped at 100%."
                )
            st.session_state["power_result"] = (result, inputs, mde, weeks_to_run)

    if "power_result" in st.session_state:
        result, inputs, mde, weeks_to_run = st.session_state["power_result"]

        st.divider()
        st.write(f"### Results for {weeks_to_run}-Week Test with {inputs.num_variants} Variants")

        st.metric(
            label=f"Statistical Power — probability of detecting a {mde}% lift if it truly exists",
            value=f"{result.power:.1%}",
        )
        st.write(
            f"In other words: if a real {mde}% improvement exists, this test setup has a "
            f"**{result.power:.1%} chance** of successfully detecting it."
        )

        if result.power < result.target_power:
            st.warning(
                f"⚠️ **Underpowered.** Your power ({result.power:.1%}) is below your "
                f"minimum threshold of {result.target_power:.1%}. Consider running the test "
                "longer or targeting a higher expected lift."
            )
        else:
            st.success(
                f"✅ **Adequately Powered.** Your test meets the "
                f"{result.target_power:.1%} trustworthiness requirement."
            )


def render_seasonal_mode() -> None:
    st.write("### Upload Historical Data")
    st.info(
        "Upload a CSV with columns: `date` (YYYY-MM-DD), `visitors` (count), "
        "`conversions` (count). Ideally 1–2 years of daily data."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "Number of variants:",
            min_value=2, value=st.session_state.get("seas_variants", 2),
            key="seas_variants",
        )
    with col2:
        st.number_input(
            "Confidence level (%):",
            value=st.session_state.get("seas_risk", 95),
            key="seas_risk",
        )
        st.number_input(
            "Power (%):",
            value=st.session_state.get("seas_trust", 80),
            key="seas_trust",
        )
        st.radio(
            "Hypothesis:",
            ["One-sided", "Two-sided"],
            key="seas_tails",
            horizontal=True,
        )

    if uploaded_file is None:
        return

    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns:
            df = df.rename(columns={"date": "ds"})

        required = {"ds", "visitors", "conversions"}
        if not required.issubset(df.columns):
            st.error("CSV must contain columns: 'date' (or 'ds'), 'visitors', 'conversions'.")
            return

        df["ds"] = pd.to_datetime(df["ds"], dayfirst=True)

        if st.button("Generate Forecast & Analysis", type="primary"):
            seasonal_inputs = BaselineInputs(
                num_variants=st.session_state["seas_variants"],
                baseline_visitors=0,   # not used in forecast path
                baseline_conversions=0,
                risk=st.session_state["seas_risk"],
                trust=st.session_state["seas_trust"],
                tails=st.session_state["seas_tails"],
            )
            forecast_confidence = seasonal_inputs.risk / 100

            with st.spinner("Running Prophet Forecast…"):
                forecast_data = run_prophet_forecast(
                    df, periods=42, interval_width=forecast_confidence
                )

            st.write("### Traffic Forecast (Next 6 Weeks)")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_data["ds"],
                y=forecast_data["pred_visitors"],
                mode="lines",
                name="Predicted Visitors",
                line=dict(color="#0072B2"),
            ))
            fig.add_trace(go.Scatter(
                x=forecast_data["ds"],
                y=forecast_data["vis_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=forecast_data["ds"],
                y=forecast_data["vis_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(0, 114, 178, 0.2)",
                name=f"Confidence Interval ({int(forecast_confidence * 100)}%)",
            ))
            fig.update_layout(
                title="Daily Visitor Forecast",
                yaxis_title="Visitors",
                xaxis_title="Date",
                hovermode="x",
            )
            st.plotly_chart(fig, width="stretch")

            results = perform_mde_calculation_forecast(forecast_data, seasonal_inputs)

            res_df = pd.DataFrame({
                "Week": [r.week for r in results],
                "Avg Visitors / Variant": [r.visitors_per_variant for r in results],
                "Relative MDE (%)": [
                    f"{r.relative_mde_pct:.2f}%" if not np.isnan(r.relative_mde_pct) else "N/A"
                    for r in results
                ],
            })

            st.write("### Seasonal MDE Results")
            st.write(
                "MDE calculated using **predicted** traffic and conversion rate for each "
                "specific week, accounting for seasonality."
            )
            st.table(res_df)

    except Exception as e:
        st.error(f"Error parsing file: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    st.title("Pre-test analysis")
    st.write("This tool helps you plan the runtime and power of your A/B experiments.")

    with st.expander("How to choose the right method", expanded=False):
        st.markdown("""
        **1. MDE Projection (Fixed Duration)**
        * *Best for:* Strict deadlines.
        * *Scenario:* "We only have 4 weeks. What's the smallest effect we can reliably detect?"
        * *Output:* MDE for Weeks 1–6, plus a sensitivity matrix for traffic scenarios.

        **2. Sample Size Calculation (Target Effect)**
        * *Best for:* Specific improvement goals.
        * *Scenario:* "We need to detect a 5% lift. How long will that take?"
        * *Output:* Required sample size per group and estimated runtime in days.

        **3. Power Calculation for Desired Lift**
        * *Best for:* Reality checks and resource allocation.
        * *Scenario:* "Product expects a 5% lift, and we have 4 weeks. What are the odds we detect it?"
        * *Output:* Statistical power (probability of detecting the expected lift) vs. your threshold.

        **4. Seasonal Forecasting (Prophet)**
        * *Best for:* Volatile or event-driven sites (e.g., approaching Black Friday).
        * *Why:* Standard calculators assume flat traffic. This method forecasts daily traffic from
          historical data, preventing under-powering during traffic dips.
        """)

    calculation_mode = st.selectbox(
        "Select Calculation Mode:",
        (
            "Calculate MDE based on Runtime",
            "Calculate Sample Size based on MDE",
            "Calculate Power for Desired Lift",
            "Seasonal (Prophet Forecast)",
        ),
        help=(
            "Choose MDE or Sample Size for stable traffic. "
            "Choose Seasonal for volatile/seasonal traffic. "
            "Choose Power to validate whether an expected uplift is detectable."
        ),
        key="calculation_mode",
    )

    if calculation_mode == "Calculate MDE based on Runtime":
        render_mde_mode()
    elif calculation_mode == "Calculate Sample Size based on MDE":
        render_sample_size_mode()
    elif calculation_mode == "Calculate Power for Desired Lift":
        render_power_mode()
    else:
        render_seasonal_mode()


if __name__ == "__main__":
    run()