from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass
from scipy.stats import norm
from scipy.stats import gamma as scipy_gamma
from scipy.stats import nbinom
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import zt_ind_solve_power
from prophet import Prophet  # type: ignore[import-untyped]  # Prophet stubs are incomplete
from foe.pretest.operations import PretestEngine
from foe.continuous.operations import ContinuousMetricEngine
from foe.core.models import AlternativeHypothesis as FoeAlternative

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
    """Shared experiment-planning inputs.

    `baseline_visitors` is the TOTAL weekly number of units across *all*
    variants combined (visitors for a binomial KPI, transactions/sessions for
    a continuous KPI).  Every calculation divides by `num_variants` to obtain
    a per-variant figure, so the convention is consistent across all modes.

    For a binomial KPI the per-unit metric is the conversion rate, summarised
    by `baseline_conversions`.  For a continuous KPI the per-unit metric is a
    real-valued quantity (e.g. revenue), summarised by `kpi_mean` and
    `kpi_variance` (the variance of a single observation).
    """
    kpi_type: str        # 'Binomial' | 'Continuous'
    num_variants: int
    baseline_visitors: int   # TOTAL weekly units across all variants
    risk: float          # e.g. 95  (percentage)
    trust: float         # e.g. 80  (percentage)
    tails: str           # 'One-sided' | 'Two-sided'
    baseline_conversions: int = 0   # binomial only
    kpi_mean: float = 0.0           # continuous only
    kpi_variance: float = 0.0       # continuous only (variance of one observation)
    variance_scaling: str = "equal" # continuous only: 'equal' | 'cv_constant'
    kpi_cv: float = 0.0             # continuous + seasonal only: coefficient of variation
    analysis_unit: str = "per_transaction"  # continuous only: 'per_visitor' | 'per_transaction'


@dataclass
class MDERow:
    week: int
    visitors_per_variant: int
    relative_mde_pct: float
    period_mean: float | None = None   # forecasted per-unit mean (seasonal mode)


@dataclass
class PowerResult:
    power: float
    target_power: float
    rate_b: float        # expected treatment mean/rate (uncapped, for reporting)
    capped: bool         # True if a binomial rate exceeded 1 and was capped


@dataclass
class GammaFit:
    k: float             # shape
    theta: float         # scale
    mean: float          # k * theta
    variance: float      # k * theta**2
    cv: float            # 1 / sqrt(k)
    log_likelihood: float
    n: int


@dataclass
class NegBinFit:
    mean: float
    variance: float      # mean + alpha * mean**2 (NB2)
    alpha: float         # dispersion parameter
    log_likelihood: float
    n: int


# ---------------------------------------------------------------------------
# Metric abstraction
# ---------------------------------------------------------------------------

def metric_moments(inputs: BaselineInputs) -> tuple[float, float]:
    """Return (mean, per-unit variance) of the chosen KPI.

    This is the single point where the binomial vs continuous distinction is
    resolved.  Every downstream formula is written in terms of a mean and a
    variance, so both KPI families share the same z-test machinery:

        * Binomial    -> mean = p,    variance = p * (1 - p)
        * Continuous  -> mean = mu,   variance = sigma**2
    """
    if inputs.kpi_type == "Continuous":
        return inputs.kpi_mean, inputs.kpi_variance
    p = inputs.baseline_conversions / inputs.baseline_visitors
    return p, p * (1 - p)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _foe_alternative(tails: str) -> FoeAlternative:
    """Map this file's 'One-sided'/'Two-sided' tails choice onto foe's
    AlternativeHypothesis enum (GREATER is treated identically to LESS by
    PretestEngine.get_z_alpha, since both use the one-tailed critical value)."""
    return FoeAlternative.TWO_SIDED if tails == "Two-sided" else FoeAlternative.GREATER


def get_critical_z(num_variants: int, alpha: float, tails: str) -> float:
    """Return the adjusted critical z-value via foe's PretestEngine, which
    applies Holm-Bonferroni step-down correction when there are more than two
    variants (shared with the post-test/business engines elsewhere in the app)."""
    return PretestEngine.get_z_alpha(num_variants, alpha, _foe_alternative(tails))


def mde_from_visitors(
    visitors_per_variant: float,
    mean: float,
    variance: float,
    z_alpha: float,
    z_power: float,
) -> float:
    """
    Minimum detectable effect for a two-sample test of means, expressed as a
    percentage of the baseline `mean`. Delegates to foe's PretestEngine, which
    implements the same two-group SE formula this file used to compute locally:
        SE = sqrt(2 * variance / n); MDE = (z_alpha + z_power) * SE / mean * 100

    For a binomial KPI pass variance = p*(1-p) and mean = p; for a continuous
    KPI pass variance = sigma**2 and mean = mu.
    """
    return PretestEngine._relative_mde(mean, variance, visitors_per_variant, z_alpha, z_power)


def fit_gamma(data: np.ndarray) -> GammaFit:
    """
    Fit a Gamma distribution (shape k, scale theta) by maximum likelihood, via
    foe's ContinuousMetricEngine (the same Gamma MLE the post-test tool uses).

    The Gamma is a natural model for positive, right-skewed continuous KPIs
    such as revenue per visitor or items per transaction.  Its first two
    moments give exactly what the experiment-planning formulas need:
        mean     = k * theta
        variance = k * theta**2
        CV       = 1 / sqrt(k)

    Only strictly positive observations are used, since the Gamma is defined
    on x > 0.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = data[data > 0]
    if data.size < 2:
        raise ValueError("Need at least two positive observations to fit a Gamma model.")
    if float(np.var(data)) <= 0:
        raise ValueError("Data has zero variance; cannot fit a Gamma model.")

    k_mle, theta_mle, log_lik = ContinuousMetricEngine.fit_gamma(data)

    return GammaFit(
        k=k_mle,
        theta=theta_mle,
        mean=k_mle * theta_mle,
        variance=k_mle * theta_mle ** 2,
        cv=1.0 / np.sqrt(k_mle),
        log_likelihood=float(log_lik),
        n=int(data.size),
    )


def fit_negbin_count(data: np.ndarray) -> NegBinFit:
    """
    Fit a Negative Binomial (NB2) model to a discrete count KPI (e.g. items
    per order) via foe's ContinuousMetricEngine -- the same engine, and the
    same NB2 MLE, the post-test tool uses to actually test this kind of
    metric. Reusing it (rather than a separate method-of-moments estimator)
    keeps the pre-test variance assumption consistent with how the collected
    data will later be analyzed.

    fit_negbin's default (no exog) is an intercept-only NB2 regression, which
    is exactly a single-group NB2 MLE: mean = exp(intercept), and the NB2
    mean-variance relationship gives variance = mean + alpha * mean**2.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = data[data >= 0]
    if data.size < 2:
        raise ValueError("Need at least two non-negative observations to fit a Negative Binomial model.")

    log_lik, alpha, res = ContinuousMetricEngine.fit_negbin(data)
    mean = float(np.exp(res.params[0]))
    variance = mean + alpha * mean ** 2

    return NegBinFit(
        mean=mean,
        variance=variance,
        alpha=float(alpha),
        log_likelihood=float(log_lik),
        n=int(data.size),
    )


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

def perform_mde_calculation(
    inputs: BaselineInputs,
    traffic_multiplier: float = 1.0,
) -> list[MDERow]:
    """
    Calculate relative MDE for weeks 1-6.

    `traffic_multiplier` scales the weekly visitor volume to model traffic
    scenarios (e.g. 0.7 = -30%, 1.3 = +30%).
    """
    alpha = 1 - (inputs.risk / 100)
    power = inputs.trust / 100
    mean, variance = metric_moments(inputs)

    z_alpha = get_critical_z(inputs.num_variants, alpha, inputs.tails)
    z_power = float(norm.ppf(power))

    weekly_visitors = int(
        np.ceil(inputs.baseline_visitors * traffic_multiplier / inputs.num_variants)
    )

    results: list[MDERow] = []
    for week in range(1, 7):
        visitors_per_variant = weekly_visitors * week
        relative_mde = mde_from_visitors(
            visitors_per_variant, mean, variance, z_alpha, z_power
        )
        results.append(MDERow(week, visitors_per_variant, relative_mde))

    return results


def perform_mde_calculation_forecast(
    forecast_df: pd.DataFrame,
    inputs: BaselineInputs,
) -> list[MDERow]:
    """
    Calculate MDE using accumulated forecasted volume rather than a static
    weekly average, accounting for seasonality.

    Works for both KPI families by reconstructing the per-unit mean from the
    forecasted count and the forecasted sum-of-metric for each cumulative week:

        * Binomial   -> mean = sum(pred_conversions) / sum(pred_visitors)
                        variance = mean * (1 - mean)
        * Continuous -> mean = sum(pred_kpi_total) / sum(pred_visitors)
                        variance = (kpi_cv * mean)**2

    Here `pred_visitors` is the forecasted count of the analysis unit; visitors
    for a per-visitor metric, transactions for a per-transaction metric (the
    caller standardises the chosen count column to this name before forecasting).
    The continuous variance uses the user-supplied coefficient of variation,
    assumed stationary even as seasonal volume (and hence the weekly mean)
    shifts. Aggregated daily data cannot reveal the per-observation spread on
    its own, so the CV must be provided.
    """
    alpha = 1 - (inputs.risk / 100)
    power = inputs.trust / 100

    z_alpha = get_critical_z(inputs.num_variants, alpha, inputs.tails)
    z_power = float(norm.ppf(power))

    is_binomial = inputs.kpi_type == "Binomial"
    value_col = "pred_conversions" if is_binomial else "pred_kpi_total"

    results: list[MDERow] = []
    for week in range(1, 7):
        current_slice = forecast_df.head(week * 7)
        total_visitors = current_slice["pred_visitors"].sum()
        total_value = current_slice[value_col].sum()

        if total_visitors <= 0:
            results.append(MDERow(week, 0, float("nan")))
            continue

        if is_binomial:
            mean = float(np.clip(total_value / total_visitors, 1e-4, 1 - 1e-4))
            variance = mean * (1 - mean)
        else:
            mean = float(total_value / total_visitors)
            if mean <= 0:
                results.append(MDERow(week, int(total_visitors / inputs.num_variants),
                                      float("nan"), period_mean=mean))
                continue
            variance = (inputs.kpi_cv * mean) ** 2

        visitors_per_variant = total_visitors / inputs.num_variants
        relative_mde = mde_from_visitors(
            visitors_per_variant, mean, variance, z_alpha, z_power
        )
        results.append(MDERow(
            week, int(visitors_per_variant), relative_mde, period_mean=mean
        ))

    return results


def calculate_sample_size_required(
    inputs: BaselineInputs,
    mde_pct: float,
) -> tuple[int, int | str]:
    """
    Return (sample_size_per_group, estimated_days).

    Uses the two-sample formula consistent with `perform_mde_calculation`:
        n = (z_alpha * sqrt(V0) + z_power * sqrt(V1))**2 / delta**2
    where V0 is the summed two-group variance under H0 and V1 under H1.

    For a binomial KPI the alternative variance changes with the lifted rate
    (p2). For a continuous KPI the treatment variance depends on the chosen
    `variance_scaling`:
        * 'equal'       -> sigma_treat**2 = sigma**2 (homoscedastic; V1 = 2*sigma**2)
        * 'cv_constant' -> the lift scales the mean and the coefficient of
                           variation is held constant, so the std scales with
                           the mean: sigma_treat**2 = sigma**2 * (1 + r)**2.

    estimated_days is an int on success, or a descriptive string on failure.
    """
    alpha = 1 - (inputs.risk / 100)
    power = inputs.trust / 100
    mean, var0 = metric_moments(inputs)
    mde_absolute = mean * (mde_pct / 100)

    z_alpha = get_critical_z(inputs.num_variants, alpha, inputs.tails)
    z_power = float(norm.ppf(power))

    if inputs.kpi_type == "Binomial":
        p2 = mean + mde_absolute
        var_treat = float(np.clip(p2 * (1 - p2), 0, None))
    elif inputs.variance_scaling == "cv_constant":
        # Constant CV: std scales with the mean.
        var_treat = var0 * (1 + mde_pct / 100) ** 2
    else:
        # Equal-variance assumption for a continuous metric.
        var_treat = var0

    sample_size = PretestEngine._sample_size_per_variant(
        mde_absolute=mde_absolute,
        var_null_sum=2 * var0,
        var_alt_sum=var0 + var_treat,
        z_alpha=z_alpha,
        z_power=z_power,
    )

    # Estimate runtime (per-variant daily volume from the weekly total)
    daily_per_group = (inputs.baseline_visitors / 7) / inputs.num_variants
    if daily_per_group <= 0:
        estimated_days: int | str = "infinite (zero daily volume per group)"
    else:
        estimated_days = int(np.ceil(sample_size / daily_per_group))

    return sample_size, estimated_days


def calculate_power(
    inputs: BaselineInputs,
    expected_lift_pct: float,
    weeks_to_run: int,
) -> PowerResult:
    """
    Calculate statistical power with Holm-Bonferroni correction when
    num_variants > 2.

    The two KPI families use different but exact engines:
        * Binomial   -> Cohen's h (arcsine transform) via `zt_ind_solve_power`.
        * Continuous -> the closed-form two-sample z-test power, which lets the
          control and treatment variances differ. Under H0 both groups sit at
          the baseline variance (SE_null); under H1 the treatment variance
          follows `variance_scaling`. This reduces to the standard equal-
          variance result when scaling is 'equal', and is the exact inverse of
          the sample-size formula above.

    The per-variant sample size is derived from the TOTAL weekly volume,
    divided by the number of variants, matching the MDE and sample-size modes.
    """
    alpha = 1 - (inputs.risk / 100)
    target_power = inputs.trust / 100

    # Apply Holm-Bonferroni: for power planning we use the first (most
    # conservative) step, i.e. alpha / num_comparisons.
    num_comparisons = inputs.num_variants - 1
    if num_comparisons > 1:
        alpha = alpha / num_comparisons

    mean, var0 = metric_moments(inputs)
    alternative = "two-sided" if inputs.tails == "Two-sided" else "larger"

    # FIX (#1): per-variant n derived from the total weekly volume.
    n_per_variant = (inputs.baseline_visitors / inputs.num_variants) * weeks_to_run

    capped = False
    if inputs.kpi_type == "Binomial":
        rate_b_raw = mean * (1 + expected_lift_pct / 100)
        capped = rate_b_raw > 1.0
        rate_b_used = float(np.clip(rate_b_raw, 0, 1))
        effect_size = abs(proportion_effectsize(mean, rate_b_used))
        reported_rate = float(rate_b_raw)
        power = float(zt_ind_solve_power(
            effect_size=effect_size,
            nobs1=n_per_variant,
            ratio=1.0,
            alpha=alpha,
            alternative=alternative,
        ))
    else:
        delta = mean * (expected_lift_pct / 100)
        reported_rate = float(mean + delta)

        if inputs.variance_scaling == "cv_constant":
            var_treat = var0 * (1 + expected_lift_pct / 100) ** 2
        else:
            var_treat = var0

        if alternative == "two-sided":
            z_crit = float(norm.ppf(1 - alpha / 2))
        else:
            z_crit = float(norm.ppf(1 - alpha))

        power = PretestEngine._power_two_sample(
            delta=delta,
            var_null_sum=2 * var0,
            var_alt_sum=var0 + var_treat,
            n_per_variant=n_per_variant,
            z_alpha=z_crit,
            alternative=_foe_alternative("Two-sided" if alternative == "two-sided" else "One-sided"),
        )

    return PowerResult(
        power=float(power),
        target_power=float(target_power),
        rate_b=reported_rate,
        capped=bool(capped),
    )


# ---------------------------------------------------------------------------
# Prophet forecasting
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_prophet_forecast(
    df: pd.DataFrame,
    value_col: str = "conversions",
    periods: int = 42,
    interval_width: float = 0.95,
) -> pd.DataFrame:
    """Fit Prophet models on `visitors` and a second `value_col` (the daily
    sum-of-metric: `conversions` for a binomial KPI, `kpi_total` for a
    continuous one) and return a 6-week daily forecast.  Results are cached by
    Streamlit based on the arguments; hashing large DataFrames may add
    overhead."""

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
    future_val = _fit_predict(df, value_col)

    forecast_final = pd.merge(future_vis, future_val, on="ds")

    # All forecasted quantities (counts and metric sums) are non-negative.
    for col in forecast_final.columns:
        if col != "ds":
            forecast_final[col] = forecast_final[col].clip(lower=0)

    return forecast_final


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _unit_label(kpi_type: str, analysis_unit: str = "per_transaction") -> str:
    """Plural noun for the experiment unit, used in labels and table headers."""
    if kpi_type != "Continuous":
        return "Visitors"
    return "Visitors" if analysis_unit == "per_visitor" else "Transactions"


def compound_per_visitor_moments(
    txn_mean: float, txn_variance: float, conversion_rate: float
) -> tuple[float, float]:
    """Per-visitor mean and variance of a revenue-style metric, built from the
    per-transaction (buyer) distribution and the conversion rate, via foe's
    PretestEngine (shared with the post-test/business engines elsewhere in
    the app):
        E[R]   = p * m
        Var[R] = p * v + m**2 * p * (1 - p)
    """
    return PretestEngine.compound_per_visitor_moments(txn_mean, txn_variance, conversion_rate)


def _mde_color_scale() -> list[list[float | str]]:
    """Green -> yellow -> red colorscale mapped to MDE 0-20%+."""
    return [
        [0.00, "rgb(56, 161, 105)"],
        [0.25, "rgb(154, 205, 90)"],
        [0.50, "rgb(236, 201, 75)"],
        [0.75, "rgb(237, 137, 54)"],
        [1.00, "rgb(229, 62, 62)"],
    ]


def validate_baseline(inputs: BaselineInputs) -> str | None:
    """Return an error message string if inputs are invalid, else None."""
    if inputs.baseline_visitors <= 0:
        return "Total weekly volume must be greater than 0."
    if not (0 < inputs.risk < 100):
        return "Confidence level must be strictly between 0 and 100."
    if not (0 < inputs.trust < 100):
        return "Power must be strictly between 0 and 100."
    if inputs.tails not in ("One-sided", "Two-sided"):
        return "Hypothesis type must be 'One-sided' or 'Two-sided'."

    if inputs.kpi_type == "Binomial":
        if inputs.baseline_conversions < 0:
            return "Baseline conversions cannot be negative."
        if inputs.baseline_conversions > inputs.baseline_visitors:
            return "Baseline conversions cannot exceed baseline visitors."
        if inputs.baseline_conversions == 0:
            return "Baseline conversions must be greater than 0 to define a rate."
    else:
        if inputs.kpi_mean <= 0:
            return "KPI mean must be greater than 0 (provide data or summary statistics)."
        if inputs.kpi_variance <= 0:
            return "KPI variance must be greater than 0 (provide data or summary statistics)."
    return None


def _parse_numeric(raw: str) -> np.ndarray:
    tokens = re.split(r"[\s,;]+", raw.strip())
    vals: list[float] = []
    for tok in tokens:
        if not tok:
            continue
        try:
            vals.append(float(tok))
        except ValueError:
            continue
    return np.array(vals, dtype=float)


def _render_gamma_fit(fit: GammaFit, values: np.ndarray) -> None:
    """Show fitted Gamma parameters and an overlay of the empirical histogram
    against the fitted density."""
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean", f"{fit.mean:,.2f}")
    m2.metric("Std dev", f"{np.sqrt(fit.variance):,.2f}")
    m3.metric("CV", f"{fit.cv:.2f}")
    m4.metric("Shape k", f"{fit.k:.3f}")

    pos = values[values > 0]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pos,
        histnorm="probability density",
        name="Observed",
        opacity=0.55,
        marker_color="#0072B2",
    ))
    xs = np.linspace(max(float(pos.min()), 1e-6), float(pos.max()), 250)
    pdf = scipy_gamma.pdf(xs, a=fit.k, scale=fit.theta)
    fig.add_trace(go.Scatter(
        x=xs, y=pdf, mode="lines", name="Fitted Gamma",
        line=dict(color="#E69F00", width=2.5),
    ))
    fig.update_layout(
        title=f"KPI distribution & fitted Gamma (n = {fit.n:,})",
        xaxis_title="KPI value",
        yaxis_title="Density",
        bargap=0.02,
        height=320,
        margin=dict(l=60, r=30, t=50, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")


def _render_negbin_fit(fit: NegBinFit, values: np.ndarray) -> None:
    """Show fitted Negative Binomial parameters and an overlay of the
    empirical histogram against the fitted probability mass function."""
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean", f"{fit.mean:,.2f}")
    m2.metric("Std dev", f"{np.sqrt(fit.variance):,.2f}")
    m3.metric("Dispersion (α)", f"{fit.alpha:.3f}")

    pos = values[values >= 0]
    # NB2 -> scipy's (n, p) parameterization: n = 1/alpha, p = n / (n + mean).
    n_param = 1.0 / fit.alpha
    p_param = n_param / (n_param + fit.mean)
    xs = np.arange(0, int(pos.max()) + 1)
    pmf = nbinom.pmf(xs, n_param, p_param)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pos,
        histnorm="probability density",
        name="Observed",
        opacity=0.55,
        marker_color="#0072B2",
        xbins=dict(start=-0.5, end=float(pos.max()) + 0.5, size=1),
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=pmf, mode="markers+lines", name="Fitted Negative Binomial",
        line=dict(color="#E69F00", width=2.5),
    ))
    fig.update_layout(
        title=f"KPI distribution & fitted Negative Binomial (n = {fit.n:,})",
        xaxis_title="KPI value (count)",
        yaxis_title="Probability",
        bargap=0.02,
        height=320,
        margin=dict(l=60, r=30, t=50, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")


def _read_continuous_values(key_prefix: str) -> np.ndarray | None:
    """Shared paste/upload widget returning a numeric array (or None)."""
    src = st.radio(
        "Data source:",
        ["Paste values", "Upload CSV"],
        horizontal=True,
        key=f"{key_prefix}_src",
    )
    values: np.ndarray | None = None
    if src == "Paste values":
        raw = st.text_area(
            "Paste values (comma-, space-, or newline-separated):",
            key=f"{key_prefix}_raw",
            height=120,
            placeholder="12.50, 41.20, 8.75, 102.00, ...",
        )
        if raw and raw.strip():
            values = _parse_numeric(raw)
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key=f"{key_prefix}_csv")
        if uploaded is not None:
            try:
                dfc = pd.read_csv(uploaded)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read CSV: {exc}")
                return None
            numeric_cols = dfc.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                st.error("No numeric columns found in the uploaded CSV.")
                return None
            col = st.selectbox(
                "Which column holds the values?",
                numeric_cols,
                key=f"{key_prefix}_col",
            )
            values = dfc[col].dropna().to_numpy()
    return values


def _continuous_metric_inputs(
    analysis_unit: str,
    show_variance_toggle: bool = False,
) -> tuple[float, float, str]:
    """Render continuous-KPI distribution inputs for the chosen analysis unit.

    Returns (mean, variance, variance_scaling) of the *per-unit* metric, where
    the unit is a visitor (`per_visitor`, zeros included) or a transaction
    (`per_transaction`, positive values only). Returns (0.0, 0.0, scaling) when
    the distribution is not yet specified.
    """
    per_visitor = analysis_unit == "per_visitor"

    scaling = "equal"
    if show_variance_toggle:
        choice = st.radio(
            "Treatment variance assumption:",
            ["Equal variance (homoscedastic)", "Scale with mean (constant CV)"],
            horizontal=True,
            key="cont_var_scaling",
            help=(
                "**Equal variance** keeps the treatment group's spread the same "
                "as the control's (the standard two-sample default). "
                "**Scale with mean** assumes the lift multiplies the mean while "
                "holding the coefficient of variation constant. For a positive "
                "lift this is the more conservative choice (larger treatment "
                "variance → more sample needed / lower power)."
            ),
        )
        scaling = "cv_constant" if choice.startswith("Scale") else "equal"

    method = st.radio(
        "How do you want to specify the KPI distribution?",
        ["Fit a model from data", "Enter summary statistics"],
        horizontal=True,
        key="cont_method",
        help=(
            "Fit a model to a sample of individual transaction values, or enter "
            "summary numbers directly. The fit auto-detects discrete count "
            "metrics (e.g. items per order) and uses a Negative Binomial model "
            "instead of Gamma for those, matching how the post-test tool treats "
            "the same kind of metric. For a per-visitor metric you also supply "
            "the conversion rate, which folds the non-converting (zero) "
            "visitors into the mean and variance."
        ),
    )

    # ---- Summary-statistics path ------------------------------------------
    if method == "Enter summary statistics":
        if per_visitor:
            label = "Mean revenue per visitor (averaged over all visitors, incl. non-buyers):"
            cv_help = (
                "Per-visitor revenue is very dispersed because most visitors "
                "contribute 0; a CV of 3-6+ is common. CV = std ÷ mean."
            )
        else:
            label = "Mean per transaction (e.g. average order value):"
            cv_help = (
                "Order values are right-skewed; a CV of roughly 1-3 is common. "
                "CV = std ÷ mean."
            )
        c1, c2 = st.columns(2)
        with c1:
            mean = st.number_input(
                label, min_value=0.0, step=1.0,
                value=st.session_state.get("cont_mean", 50.0), key="cont_mean",
            )
        with c2:
            cv = st.number_input(
                "Coefficient of variation (std ÷ mean):",
                min_value=0.0, step=0.1,
                value=st.session_state.get("cont_cv", 1.0),
                key="cont_cv", help=cv_help,
            )
        if mean <= 0 or cv <= 0:
            st.info("Enter a positive mean and coefficient of variation.")
            return 0.0, 0.0, scaling
        variance = (cv * mean) ** 2
        return float(mean), float(variance), scaling

    # ---- Fit-from-data path -----------------------------------------------
    st.caption(
        "Paste or upload a sample of **individual transaction values** "
        "(positive amounts, or counts, one per order)."
    )
    values = _read_continuous_values("cont")
    if values is None or values.size == 0:
        st.info("Provide transaction values to fit a model.")
        return 0.0, 0.0, scaling

    positive = values[values > 0]
    n_zeros = int(values.size - positive.size)
    if positive.size < 2:
        st.warning("Need at least two positive values to fit a model.")
        return 0.0, 0.0, scaling

    # Count-data gate: a discrete, non-negative count metric (e.g. items or
    # tickets per order) is structurally unsuited to the Gamma's continuous,
    # right-skewed density -- and typically overdispersed (variance > mean),
    # which Gamma doesn't capture either. Route it to Negative Binomial
    # instead, mirroring the post-test tool's count-data gate so pre-test
    # planning and post-test hypothesis testing agree on how the metric is
    # modelled.
    if ContinuousMetricEngine.is_count_kpi(values):
        st.info(
            "**Detected: discrete count metric** (e.g. items per order). Fitting "
            "a Negative Binomial model instead of Gamma -- count data is "
            "discrete and typically overdispersed (variance > mean), which the "
            "Gamma's continuous density doesn't capture. This matches how the "
            "post-test tool models the same kind of metric."
        )
        try:
            nb_fit = fit_negbin_count(values)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Negative Binomial fit failed: {exc}")
            return 0.0, 0.0, scaling

        if n_zeros > 0:
            st.caption(
                f"{n_zeros:,} negative value(s) were excluded from the fit "
                "(a count metric cannot be negative)."
            )
        _render_negbin_fit(nb_fit, values)
        m, v = nb_fit.mean, nb_fit.variance
    else:
        try:
            fit = fit_gamma(values)
        except ValueError as exc:
            st.warning(str(exc))
            return 0.0, 0.0, scaling

        if n_zeros > 0:
            st.caption(
                f"{n_zeros:,} non-positive value(s) were excluded from the Gamma fit "
                "(the Gamma is defined for x > 0; zeros are handled via the "
                "conversion rate below for per-visitor metrics)."
            )
        _render_gamma_fit(fit, values)

        # Transaction-level moments from the fit.
        m, v = fit.mean, fit.variance

    if not per_visitor:
        return float(m), float(v), scaling

    # Per-visitor: combine the buyer distribution with the conversion rate.
    cr_pct = st.number_input(
        "Conversion rate (% of visitors who transact):",
        min_value=0.0, max_value=100.0, step=0.1,
        value=st.session_state.get("cont_conv_rate", 5.0),
        key="cont_conv_rate",
        help=(
            "Used to fold non-converting visitors into the per-visitor mean and "
            "variance: mean = p·m, variance = p·v + m²·p(1−p)."
        ),
    )
    p = cr_pct / 100.0
    if p <= 0:
        st.info("Enter a conversion rate above 0% for a per-visitor metric.")
        return 0.0, 0.0, scaling

    mean_r, var_r = compound_per_visitor_moments(m, v, p)
    d1, d2, d3 = st.columns(3)
    d1.metric("Per-visitor mean", f"{mean_r:,.2f}")
    d2.metric("Per-visitor std dev", f"{np.sqrt(var_r):,.2f}")
    d3.metric("Per-visitor CV", f"{np.sqrt(var_r) / mean_r:.2f}" if mean_r > 0 else "-")
    st.caption(
        f"Derived from average order value {m:,.2f}, order-value variance "
        f"{v:,.1f}, and a {cr_pct:.1f}% conversion rate."
    )
    return float(mean_r), float(var_r), scaling


def get_baseline_inputs(
    kpi_type: str,
    include_mde: bool = False,
    include_weeks: bool = False,
) -> BaselineInputs:
    """Render common input widgets and return a BaselineInputs dataclass.
    Widget state is read from / persisted in st.session_state."""
    st.write("### Baseline Data")

    # Analysis unit is chosen first for a continuous KPI, because it determines
    # what the weekly-volume field counts (visitors vs transactions).
    analysis_unit = "per_transaction"
    if kpi_type == "Continuous":
        unit_choice = st.radio(
            "Analysis unit:",
            [
                "Revenue per visitor (includes non-converting visitors)",
                "Revenue per transaction (converting units only)",
            ],
            horizontal=True,
            key="cont_analysis_unit",
            help=(
                "**Per visitor** averages the metric over every visitor, counting "
                "non-buyers as 0; the usual business outcome and the unit you "
                "randomise on. Volume is counted in **visitors**.\n\n"
                "**Per transaction** measures average order value among buyers only. "
                "Volume is counted in **transactions** (≈ visitors × conversion rate)."
            ),
        )
        analysis_unit = (
            "per_visitor" if unit_choice.startswith("Revenue per visitor")
            else "per_transaction"
        )

    unit = _unit_label(kpi_type, analysis_unit).lower()
    st.write(f"Enter weekly {unit}, the KPI baseline, and the test parameters.")

    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "Number of variants (including control):",
            min_value=2, step=1,
            value=st.session_state.get("num_variants", 2),
            key="num_variants",
        )
        st.number_input(
            f"Total weekly {unit} (all variants combined):",
            min_value=0, step=1,
            value=st.session_state.get("baseline_visitors", 0),
            key="baseline_visitors",
            help=(
                f"The combined weekly number of {unit} across every variant. "
                "Each calculation divides this by the number of variants to get a "
                "per-variant figure. This must match the analysis unit above."
            ),
        )
        if kpi_type == "Binomial":
            st.number_input(
                "Conversions among those weekly visitors:",
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

    kpi_mean, kpi_variance, variance_scaling = 0.0, 0.0, "equal"
    if kpi_type == "Continuous":
        st.write("#### Continuous KPI")
        # The treatment-variance assumption only matters when a target effect
        # is present (sample-size and power modes set include_mde=True). MDE
        # projection uses the baseline variance only, so the toggle is hidden.
        kpi_mean, kpi_variance, variance_scaling = _continuous_metric_inputs(
            analysis_unit, show_variance_toggle=include_mde
        )

    return BaselineInputs(
        kpi_type=kpi_type,
        num_variants=st.session_state["num_variants"],
        baseline_visitors=st.session_state["baseline_visitors"],
        risk=st.session_state["risk"],
        trust=st.session_state["trust"],
        tails=st.session_state["tails"],
        baseline_conversions=st.session_state.get("baseline_conversions", 0),
        kpi_mean=kpi_mean,
        kpi_variance=kpi_variance,
        variance_scaling=variance_scaling,
        analysis_unit=analysis_unit,
    )


# ---------------------------------------------------------------------------
# Mode renderers
# ---------------------------------------------------------------------------

def render_mde_mode(kpi_type: str) -> None:
    inputs = get_baseline_inputs(kpi_type)

    if st.button("Calculate MDE", type="primary"):
        error = validate_baseline(inputs)
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
    unit = _unit_label(inputs.kpi_type, inputs.analysis_unit)
    st.write("## MDE Calculation Results")
    st.write(
        "This table shows the smallest relative effect detectable each week. "
        "An MDE below 5% is generally testworthy; 5-10% is debatable."
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
                f"{unit} / Variant": [r.visitors_per_variant for r in results],
                "Relative MDE (%)": [f"{r.relative_mde_pct:.2f}%" for r in results],
            }
        )
        st.write(df.to_html(index=False), unsafe_allow_html=True)

    with tab_sensitivity:
        st.write(
            "**How would traffic spikes or drops affect your MDE?**  "
            "Each cell shows the Relative MDE (%) for a given week and traffic scenario. "
            "Colors: 🟢 <5% · 🟡 5-10% · 🔴 >10%."
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


def render_sample_size_mode(kpi_type: str) -> None:
    inputs = get_baseline_inputs(kpi_type, include_mde=True)

    if st.button("Calculate Sample Size", type="primary"):
        error = validate_baseline(inputs)
        mde = st.session_state.get("mde", 5)

        if error:
            st.error(error)
        elif mde <= 0:
            st.error("MDE must be greater than 0%.")
        else:
            if inputs.kpi_type == "Binomial":
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
        unit = _unit_label(inputs.kpi_type, inputs.analysis_unit).lower()

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
        if inputs.kpi_type == "Continuous":
            assumption = (
                "treatment variance scales with the mean (constant CV)"
                if inputs.variance_scaling == "cv_constant"
                else "equal variance across groups (homoscedastic)"
            )
            st.caption(f"Continuous KPI assumption: {assumption}.")
        if isinstance(estimated_days, int):
            st.write(
                f"With **{inputs.baseline_visitors:,}** total weekly {unit}, "
                f"your test is estimated to run for approximately **{estimated_days} days** "
                "to reach the required sample size per group."
            )
        else:
            st.write(f"Estimated runtime: {estimated_days}.")


def render_power_mode(kpi_type: str) -> None:
    inputs = get_baseline_inputs(kpi_type, include_mde=True, include_weeks=True)

    if st.button("Calculate Power", type="primary"):
        error = validate_baseline(inputs)
        mde = st.session_state.get("mde", 5)
        weeks_to_run = st.session_state.get("weeks_to_run", 4)

        if error:
            st.error(error)
        elif mde <= 0:
            st.error("Expected lift must be greater than 0%.")
        else:
            result = calculate_power(inputs, mde, weeks_to_run)
            if result.capped:
                st.warning(
                    f"An expected lift of {mde}% pushes the variant conversion rate over 100% "
                    f"({result.rate_b:.1%}). The rate was capped at 100% for the calculation."
                )
            st.session_state["power_result"] = (result, inputs, mde, weeks_to_run)

    if "power_result" in st.session_state:
        result, inputs, mde, weeks_to_run = st.session_state["power_result"]

        st.divider()
        st.write(f"### Results for {weeks_to_run}-Week Test with {inputs.num_variants} Variants")

        st.metric(
            label=f"Statistical Power: probability of detecting a {mde}% lift if it truly exists",
            value=f"{result.power:.1%}",
        )
        st.write(
            f"In other words: if a real {mde}% improvement exists, this test setup has a "
            f"**{result.power:.1%} chance** of successfully detecting it."
        )
        if inputs.kpi_type == "Continuous":
            assumption = (
                "treatment variance scales with the mean (constant CV)"
                if inputs.variance_scaling == "cv_constant"
                else "equal variance across groups (homoscedastic)"
            )
            st.caption(f"Continuous KPI assumption: {assumption}.")

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


def _seasonal_data_expander(kpi_type: str, analysis_unit: str = "per_transaction") -> None:
    """Explain the CSV schema (and the CV input for continuous) the user must
    supply for seasonal forecasting."""
    with st.expander("What data do I need to upload?", expanded=False):
        st.markdown(
            "Provide **daily historical data**: ideally **1-2 years** so Prophet "
            "can learn weekly and yearly seasonality. One row per day, no gaps."
        )
        if kpi_type == "Binomial":
            st.markdown("""
            **Required columns (binomial KPI):**
            * `date`: the day, as `YYYY-MM-DD` (a `ds` column is also accepted).
            * `visitors`: number of visitors that day (the denominator).
            * `conversions`: number of converting visitors that day (the count of 1's).

            The per-day conversion rate is reconstructed as `conversions ÷ visitors`,
            and its variance follows directly from the rate, so **no extra input is needed**.
            """)
            example = pd.DataFrame({
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "visitors": [5120, 4880, 6010],
                "conversions": [262, 244, 331],
            })
        elif analysis_unit == "per_visitor":
            st.markdown("""
            **Required columns (revenue per visitor):**
            * `date`: the day, as `YYYY-MM-DD` (a `ds` column is also accepted).
            * `visitors`: number of visitors that day (the denominator and the
              sample-size unit; non-buyers are included).
            * `kpi_total`: the **sum** of your KPI across all visitors that day
              (e.g. total daily revenue). *Not* the average. If your column has
              another name, you'll be able to pick it after upload.

            The per-visitor mean for each forecast week is `kpi_total ÷ visitors`.
            Daily aggregates can't reveal how spread out individual values are, so
            you also supply the **coefficient of variation** of revenue *per visitor*
            (CV = std ÷ mean), assumed roughly stable over time.
            """)
            example = pd.DataFrame({
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "visitors": [5120, 4880, 6010],
                "kpi_total": [41230.50, 39870.00, 52110.75],
            })
        else:  # per_transaction
            st.markdown("""
            **Required columns (revenue per transaction):**
            * `date` : the day, as `YYYY-MM-DD` (a `ds` column is also accepted).
            * `transactions` : number of **orders** that day (the denominator and the
              sample-size unit). Use the actual transaction count from your data;
              don't derive it from visitors × conversion rate, since that ratio
              drifts day to day and the two counts come from different GA4 tables.
            * `kpi_total` : the **sum** of your KPI across those orders that day
              (e.g. total daily revenue). *Not* the average. If your column has
              another name, you'll be able to pick it after upload.

            The per-order mean (average order value) for each forecast week is
            `kpi_total ÷ transactions`. You also supply the **coefficient of
            variation** of *order value* (CV = std ÷ mean), assumed roughly stable.
            """)
            example = pd.DataFrame({
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "transactions": [262, 244, 331],
                "kpi_total": [41230.50, 39870.00, 52110.75],
            })
        st.caption("Example of the expected shape:")
        st.table(example)


def render_seasonal_mode(kpi_type: str) -> None:
    is_binomial = kpi_type == "Binomial"

    st.write("### Upload Historical Data")

    analysis_unit = "per_transaction"
    if not is_binomial:
        unit_choice = st.radio(
            "Analysis unit:",
            [
                "Revenue per visitor (includes non-converting visitors)",
                "Revenue per transaction (converting units only)",
            ],
            horizontal=True,
            key="seas_analysis_unit",
            help=(
                "Determines the denominator and the sample-size unit. Per visitor "
                "uses the `visitors` column; per transaction uses the `transactions` "
                "column. Provide the matching count column in your CSV."
            ),
        )
        analysis_unit = (
            "per_visitor" if unit_choice.startswith("Revenue per visitor")
            else "per_transaction"
        )

    unit = _unit_label(kpi_type, analysis_unit)
    # Internal name for the denominator/sample-size count column.
    count_col = "visitors" if (is_binomial or analysis_unit == "per_visitor") else "transactions"

    _seasonal_data_expander(kpi_type, analysis_unit)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "Number of variants:",
            min_value=2, value=st.session_state.get("seas_variants", 2),
            key="seas_variants",
        )
        if not is_binomial:
            cv_label = (
                "CV of revenue per visitor (std ÷ mean):"
                if analysis_unit == "per_visitor"
                else "CV of order value (std ÷ mean):"
            )
            cv_help = (
                "Per-visitor revenue is very dispersed because most visitors "
                "contribute 0; a CV of 3-6+ is common."
                if analysis_unit == "per_visitor"
                else "Order values are right-skewed; a CV of roughly 1-3 is common."
            ) + " Estimate it in the non-seasonal Continuous mode via the fit-from-data option if unsure (auto-detects Gamma vs. Negative Binomial for count metrics)."
            st.number_input(
                cv_label,
                min_value=0.0, step=0.1,
                value=st.session_state.get("seas_cv", 1.0),
                key="seas_cv",
                help=cv_help,
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

        if "ds" not in df.columns:
            st.error("CSV must contain a 'date' (or 'ds') column.")
            return
        if count_col not in df.columns:
            st.error(
                f"CSV must contain a '{count_col}' column for "
                f"{'a binomial KPI' if is_binomial else _unit_label(kpi_type, analysis_unit).lower() + ' analysis'}."
            )
            return

        if is_binomial:
            if "conversions" not in df.columns:
                st.error("CSV must contain a 'conversions' column for a binomial KPI.")
                return
            value_col = "conversions"
        else:
            value_col = "kpi_total"
            if "kpi_total" not in df.columns:
                # Let the user pick which numeric column is the daily KPI total.
                candidates = [
                    c for c in df.select_dtypes(include="number").columns
                    if c not in ("visitors", "transactions")
                ]
                if not candidates:
                    st.error(
                        "CSV must contain a 'kpi_total' column (daily sum of the KPI), "
                        "or another numeric column to use as the total."
                    )
                    return
                value_col = st.selectbox(
                    "Which column holds the daily KPI total (sum, not average)?",
                    candidates,
                    key="seas_value_col",
                )

        df["ds"] = pd.to_datetime(df["ds"], dayfirst=True)

        cv = float(st.session_state.get("seas_cv", 1.0))
        if not is_binomial and cv <= 0:
            st.error("Coefficient of variation must be greater than 0 for a continuous KPI.")
            return

        if st.button("Generate Forecast & Analysis", type="primary"):
            seasonal_inputs = BaselineInputs(
                kpi_type=kpi_type,
                num_variants=st.session_state["seas_variants"],
                baseline_visitors=0,   # not used in forecast path
                baseline_conversions=0,
                risk=st.session_state["seas_risk"],
                trust=st.session_state["seas_trust"],
                tails=st.session_state["seas_tails"],
                kpi_cv=cv,
                analysis_unit=analysis_unit,
            )
            forecast_confidence = seasonal_inputs.risk / 100

            # Standardise the chosen count column to 'visitors' and the value
            # column to the name the forecast/MDE helpers expect, then forecast
            # only those two series. For a per-transaction KPI the real visitors
            # column (if any) is dropped to avoid a name collision.
            forecast_value = "conversions" if is_binomial else "kpi_total"
            keep = ["ds", count_col, value_col]
            fit_df = df[keep].rename(columns={
                count_col: "visitors",
                value_col: forecast_value,
            })

            with st.spinner("Running Prophet Forecast…"):
                forecast_data = run_prophet_forecast(
                    fit_df, value_col=forecast_value,
                    periods=42, interval_width=forecast_confidence,
                )

            st.write(f"### {unit} Forecast (Next 6 Weeks)")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_data["ds"],
                y=forecast_data["pred_visitors"],
                mode="lines",
                name=f"Predicted {unit}",
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
                title=f"Daily {unit} Forecast",
                yaxis_title=unit,
                xaxis_title="Date",
                hovermode="x",
            )
            st.plotly_chart(fig, width="stretch")

            results = perform_mde_calculation_forecast(forecast_data, seasonal_inputs)

            table: dict[str, list] = {
                "Week": [r.week for r in results],
                f"Avg {unit} / Variant": [r.visitors_per_variant for r in results],
            }
            if not is_binomial:
                table["Forecasted mean (KPI / unit)"] = [
                    f"{r.period_mean:,.2f}" if r.period_mean is not None else "N/A"
                    for r in results
                ]
            table["Relative MDE (%)"] = [
                f"{r.relative_mde_pct:.2f}%" if not np.isnan(r.relative_mde_pct) else "N/A"
                for r in results
            ]
            res_df = pd.DataFrame(table)

            st.write("### Seasonal MDE Results")
            if is_binomial:
                st.write(
                    "MDE calculated using **predicted** traffic and conversion rate for each "
                    "specific week, accounting for seasonality."
                )
            else:
                denom = "visitors" if analysis_unit == "per_visitor" else "transactions"
                mean_name = (
                    "revenue per visitor" if analysis_unit == "per_visitor"
                    else "average order value"
                )
                st.write(
                    f"MDE calculated using **predicted** {denom} and {mean_name} for each "
                    f"specific week, with variance from your coefficient of variation "
                    f"(CV = {cv:.2f}), accounting for seasonality."
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
        **KPI type**
        * *Binomial:* a yes/no outcome per visitor; conversion rate, signup rate, bounce.
        * *Continuous:* a real-valued outcome per unit; revenue per visitor, time on
          site. A Gamma model captures the positive, right-skewed shape of these
          metrics and supplies the mean and variance the calculations need. Discrete
          count metrics (e.g. items per order) are auto-detected when fitting from
          data and use a Negative Binomial model instead, since count data is
          overdispersed in a way Gamma doesn't capture.

        **1. MDE Projection (Fixed Duration)**
        * *Best for:* Strict deadlines.
        * *Scenario:* "We only have 4 weeks. What's the smallest effect we can reliably detect?"
        * *Output:* MDE for Weeks 1-6, plus a sensitivity matrix for traffic scenarios.

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
          historical data, preventing under-powering during traffic dips. *(Works for both
          binomial and continuous KPIs; a continuous KPI also needs a coefficient of variation.)*
        """)

    kpi_choice = st.radio(
        "KPI type:",
        ["Binomial (conversion rate)", "Continuous (revenue, items per order, …)"],
        horizontal=True,
        key="kpi_type_radio",
        help=(
            "Choose Binomial for yes/no outcomes (conversion). Choose Continuous for "
            "real-valued outcomes (revenue, items per order, …) -- modelled with a "
            "Gamma fit, or a Negative Binomial fit if the data looks like a count."
        ),
    )
    kpi_type = "Binomial" if kpi_choice.startswith("Binomial") else "Continuous"

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
        render_mde_mode(kpi_type)
    elif calculation_mode == "Calculate Sample Size based on MDE":
        render_sample_size_mode(kpi_type)
    elif calculation_mode == "Calculate Power for Desired Lift":
        render_power_mode(kpi_type)
    else:
        render_seasonal_mode(kpi_type)


if __name__ == "__main__":
    run()
