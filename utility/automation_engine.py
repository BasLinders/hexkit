"""
automation_engine.py
Non-UI pipeline for the Automation page: turns a control/variation pair of
(visitors, conversions, AOV) into FOE engine results and an Airtable payload.
Kept separate from pages/automation.py so it stays testable without Streamlit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from foe.core.models import (
    AlternativeHypothesis,
    AnalysisUnit,
    BusinessCaseInput,
    ContinuousApproach,
    ContinuousMetricConfig,
    ExperimentInput,
)
from foe.frequentist.operations import FrequentistEngine
from foe.bayesian.operations import BayesianEngine
from foe.continuous.operations import ContinuousMetricEngine
from foe.pretest.operations import PretestEngine

# The Send step discovers each destination base/table's real field names live
# via Airtable's metadata API (see utility/airtable_client.list_tables) and
# lets the user assign them — no hardcoded-per-base mapping to maintain here.
# These two dicts are just cosmetic/convenience: friendly labels for the
# assignment UI, and a same-name-match hint to auto-preselect a sensible
# default when a base happens to already use these names.
FIELD_LABELS = {
    "start_date": "Start date",
    "end_date": "End date",
    "visitors_control": "Visitors — Control",
    "visitors_variation": "Visitors — Variation",
    "conversions_control": "Conversions — Control",
    "conversions_variation": "Conversions — Variation",
    "p_value": "P-value (Frequentist)",
    "probability_pct": "Probability to beat control % (Bayesian)",
    "continuous_p_value": "P-value (Continuous)",
    "continuous_test_name": "Test used (Continuous)",
    "effect_on_revenue": "Effect on revenue",
    "pretest_mde_table": "Pre-test MDE table",
    "ai_conclusion": "AI conclusion",
    "description": "Short description",
}

# Each key maps to a list of candidate names, tried in order — covers common
# English/Dutch naming variants since Airtable bases here aren't necessarily
# English (e.g. "Startdatum" instead of "Start date").
DEFAULT_FIELD_NAME_HINTS: dict[str, list[str]] = {
    "start_date": ["start date", "startdate", "startdatum"],
    "end_date": ["end date", "enddate", "einddatum"],
    "visitors_control": ["visitors - control", "visitors control"],
    "visitors_variation": ["visitors - variation", "visitors variation"],
    "conversions_control": ["conversions - control", "conversions control"],
    "conversions_variation": ["conversions - variation", "conversions variation"],
    "probability_pct": ["probability (%)", "probability"],
    "p_value": ["p-value"],
    "continuous_p_value": ["p-value (continuous)"],
    "continuous_test_name": ["test used (continuous)"],
    "effect_on_revenue": ["effect on revenue", "effect op omzet"],
    "pretest_mde_table": ["pre-test mde table", "pre-test analysis"],
    "ai_conclusion": ["ai conclusion", "conclusion", "conclusie"],
    "description": ["description"],
}

_TAIL_MAP = {
    "Two-sided": AlternativeHypothesis.TWO_SIDED,
    "Greater": AlternativeHypothesis.GREATER,
    "Less": AlternativeHypothesis.LESS,
}

TAILS = list(_TAIL_MAP.keys())

# Each method's monetary estimate rests on different assumptions and is
# appropriate in different situations -- surfaced both in the Step 3 UI and
# in the AI-conclusion payload, so a human (or Gemini) picking which
# "effect on revenue" number to trust for a given experiment can weigh the
# trade-offs rather than treating all three as interchangeable.
MONETARY_METHOD_NOTES: dict[str, dict[str, list[str]]] = {
    "frequentist": {
        "pros": [
            "Deterministic, closed-form estimate tied directly to the same confidence interval as the significance test -- no simulation noise, identical result every run.",
            "Correctly handles one-sided tests: an unbounded CI renders as an explicit 'at least X' / 'at most Y' claim rather than a fabricated number.",
            "Cheapest to compute and the easiest of the three to explain to a non-technical audience.",
        ],
        "cons": [
            "Decomposes revenue into rate x AOV separately, rather than modelling revenue's actual (right-skewed, non-negative) distribution.",
            "Relies on a normal approximation (CLT); the interval can be less reliable on small samples or highly variable AOV.",
            "A non-significant result renders only as a blunt 'treat as illustrative' hedge -- no expected-value read on whether shipping anyway would be a reasonable bet.",
        ],
    },
    "bayesian": {
        "pros": [
            "Full posterior distribution, not just a point estimate + CI -- captures asymmetric uncertainty naturally instead of assuming a symmetric normal spread.",
            "Separates expected upside from expected downside risk explicitly, giving a decision-theoretic 'is this bet worth it' read even before conventional significance is reached.",
            "A lift prior guards against small-sample overestimation of implausibly large effects.",
        ],
        "cons": [
            "Monte Carlo based -- more expensive to compute, and the exact figure carries sampling noise across runs (controlled by the sample-size setting, but never fully eliminated).",
            "Always produces an expected-value number, even for a genuinely inconclusive result -- read alongside prob_beat_control/prob_being_best, never the money figure alone.",
            "AOV is modelled as log-normal with an assumed variability (CV); if that's a poor match for the real order-value distribution, the estimate's spread (not its mean) is affected.",
            "Projects future volume by scaling each variant's OWN observed visitors by runtime_days -- i.e. assumes the traffic split observed during the test continues. Frequentist/Continuous instead scale a single combined daily-visitor figure -- i.e. assume the winner is rolled out to 100% of future traffic. The two are not projecting the same future and their revenue figures are not directly comparable.",
        ],
    },
    "continuous": {
        "pros": [
            "Uses the actual row-level revenue data directly (Gamma-fitted), rather than reconstructing revenue from rate x AOV -- the most direct measure when revenue itself is the KPI.",
            "RPV vs RPT isolates whether an effect comes from more people converting, from each buyer spending more, or both.",
            "Correctly models revenue's right-skewed, non-negative shape instead of assuming normality on the raw values.",
        ],
        "cons": [
            "Still uses a normal-approximation (delta-method) CI on top of the Gamma-fitted group means -- inherits the same CLT caveats as Frequentist, just applied to fitted statistics instead of a raw proportion.",
            "RPT counts buyers only, so its effective sample size can be much smaller than RPV's -- wider, less stable intervals on modest traffic.",
            "Needs the separate continuous/row-level data fetch (extra BigQuery scan cost) that the other two methods don't require.",
        ],
    },
}

MONETARY_METHOD_GUIDANCE = (
    "Rough guide to which source tends to fit best: a rate-based test with fairly stable AOV -- "
    "Frequentist or Bayesian on binomial data is simplest and cheapest. Revenue or AOV itself is the "
    "KPI (pricing, upsell, merchandising) -- Continuous is the most direct measure. Small sample or a "
    "pre-significance ship/hold decision -- Bayesian's expected-value framing is more informative than "
    "a flat non-significant verdict. Need a simple, easily-defensible number for a non-technical or "
    "finance audience -- Frequentist's closed-form CI is the most transparent. IMPORTANT: the three "
    "figures are not a like-for-like comparison even before uncertainty is considered -- Frequentist "
    "and Continuous project revenue assuming the winning variant is rolled out to 100% of future "
    "traffic (today's combined daily visitors), while Bayesian projects forward assuming the SAME "
    "traffic split observed during the test continues (each arm scaled by its own historical daily "
    "visitors). In an even split, Bayesian's figure will run roughly half of Frequentist/Continuous's "
    "for a comparable effect size -- that gap reflects differing rollout assumptions, not disagreement "
    "about the underlying effect."
)


@dataclass
class VariantData:
    label: str
    visitors: int
    conversions: int
    aov: float = 0.0


def run_frequentist_analysis(
    control: VariantData,
    variation: VariantData,
    confidence_level: float,
    tail: Literal["Two-sided", "Greater", "Less"],
    daily_visitors: float,
    projection_days: int,
    aov_cv: float = 0.0,
) -> dict:
    """Runs the FOE FrequentistEngine (z-test) on a single control/variation pair."""
    alternative = _TAIL_MAP[tail]
    alpha = 1.0 - confidence_level

    data = ExperimentInput(
        visitors=[control.visitors, variation.visitors],
        conversions=[control.conversions, variation.conversions],
        alternative=alternative,
        confidence_level=confidence_level,
        labels=[control.label, variation.label],
    )
    result = FrequentistEngine().run_synthesis(data)[0]

    p_ctrl = control.conversions / control.visitors
    p_var = variation.conversions / variation.visitors
    se_ctrl = math.sqrt(p_ctrl * (1 - p_ctrl) / control.visitors)
    se_var = math.sqrt(p_var * (1 - p_var) / variation.visitors)
    se_aov_ctrl = (control.aov * aov_cv) / math.sqrt(control.conversions) if aov_cv > 0 and control.conversions > 0 else 0.0
    se_aov_var = (variation.aov * aov_cv) / math.sqrt(variation.conversions) if aov_cv > 0 and variation.conversions > 0 else 0.0

    monetary = FrequentistEngine.estimate_monetary_impact_per_variant(
        p_ctrl=p_ctrl, se_ctrl=se_ctrl, aov_ctrl=control.aov,
        p_chal=p_var, se_chal=se_var, aov_chal=variation.aov,
        daily_visitors=daily_visitors,
        alpha=alpha,
        alternative=alternative,
        projection_period=projection_days,
        se_aov_ctrl=se_aov_ctrl,
        se_aov_chal=se_aov_var,
    )

    monetary_conclusion = FrequentistEngine.generate_monetary_conclusion(
        variant_name=variation.label,
        monetary_result=monetary,
        is_significant=result.is_significant,
        alternative=alternative,
    )

    return {
        "method": "frequentist",
        "p_value": result.p_value,
        "is_significant": result.is_significant,
        "uplift": result.uplift,
        "ci_diff": result.ci_diff,
        "conclusion": result.conclusion,
        "monetary_conclusion": monetary_conclusion,
        "effect_on_revenue": monetary["point_estimate"],
        "effect_on_revenue_ci": (monetary["ci_low"], monetary["ci_high"]),
        "projection_days": projection_days,
    }


def run_bayesian_analysis(
    control: VariantData,
    variation: VariantData,
    runtime_days: int,
    projection_days: int,
    n_samples: int = 100_000,
    aov_cv: float = 0.5,
) -> dict:
    """
    Runs the FOE BayesianEngine (Beta-Binomial Monte Carlo) on a control/variation pair.

    aov_cv is the same "AOV variability (CV)" the UI collects for Frequentist's
    se_aov propagation, passed through to BayesianEngine.run_monetary_projection's
    log-normal AOV sampling instead of silently falling back to its 0.5 default
    regardless of what the user set — the two methods use the CV differently
    (Frequentist propagates it as an estimation-uncertainty SE; Bayesian samples
    AOV directly from it each draw), but both should reflect the same input.
    """
    data = ExperimentInput(
        visitors=[control.visitors, variation.visitors],
        conversions=[control.conversions, variation.conversions],
        labels=[control.label, variation.label],
    )
    engine = BayesianEngine()
    prob_result = engine.run_probability_analysis(data, n_samples=n_samples)[0]

    # BayesianEngine._sample_aov draws AOV samples via
    # np.log(mean_aov) -- an AOV of exactly 0.0 (VariantData.aov defaults to
    # 0.0 when a variant has zero conversions, e.g. very early in a test)
    # sends that straight to log(0) = -inf, raising a RuntimeWarning on every
    # such run. It doesn't crash -- exp(-inf) underflows cleanly to 0.0,
    # which is the right degenerate answer (no order data to model an AOV
    # distribution from) -- but relying on float underflow instead of an
    # explicit guard is fragile. Clamped to a negligible epsilon instead:
    # indistinguishable from 0 in any reported figure (every downstream
    # value rounds to cents) but keeps log() finite.
    _AOV_EPSILON = 1e-6
    biz_case = BusinessCaseInput(
        aovs={
            control.label: control.aov or _AOV_EPSILON,
            variation.label: variation.aov or _AOV_EPSILON,
        },
        runtime_days=runtime_days,
        projection_period=projection_days,
    )
    # Exact for 2 variants: exactly one of {control, variation} wins each Monte
    # Carlo draw, so prob_best_overall[control] = 1 - prob_being_best(variation).
    prob_best_overall = [1.0 - prob_result.prob_being_best, prob_result.prob_being_best]

    monetary = engine.run_monetary_projection(
        visitors=[control.visitors, variation.visitors],
        conversions=[control.conversions, variation.conversions],
        biz_case=biz_case,
        prob_best_overall=prob_best_overall,
        variant_labels=[control.label, variation.label],
        aov_cv=aov_cv,
        n_simulations=n_samples,
    )[0]

    return {
        "method": "bayesian",
        "probability_pct": prob_result.prob_beat_control * 100,
        "prob_being_best": prob_result.prob_being_best,
        "expected_uplift": prob_result.expected_uplift,
        "expected_loss": prob_result.expected_loss,
        "conclusion": prob_result.conclusion,
        # generate_bayesian_conclusion's output, already computed inside
        # run_monetary_projection (Strong Winner / Clear Loser / Asymmetric
        # Bet / Inconclusive framing) — prob_result.conclusion above is
        # probability-only and never mentions money at all.
        "monetary_conclusion": monetary["conclusion"],
        "effect_on_revenue": monetary["expected_total_contribution"],
        "expected_revenue_uplift": monetary["expected_uplift"],
        "expected_revenue_risk": monetary["expected_risk"],
        "projection_days": projection_days,
    }


def collapse_to_per_visitor(df: "pd.DataFrame", kpi: str = "purchase_revenue") -> "pd.DataFrame":
    """
    Collapses a raw continuous fetch to exactly one row per (variant, visitor),
    summing `kpi` across any separate orders a visitor placed in the date range.

    automation.py's continuous fetch (build_continuous_from_shared_scan) is
    one row per (visitor, order) pair, not one row per visitor -- its
    ecommerce_data CTE groups by (user, transaction) to preserve per-order
    values for RPT mode, and the final SELECT joins that back to one-row-per-
    user variant_data on user_pseudo_id alone. A visitor with 2+ separate
    purchases in range therefore gets 2+ rows; non-buyers (NULL revenue from
    the LEFT JOIN, zero-filled here) get exactly one.

    Every "per visitor" computation -- RPV's mean and its significance test,
    a "how many visitors" count, revenue-per-visitor summaries, the future
    daily-visitor scale used for revenue projections -- needs one row per
    visitor, or it silently over-counts the population and under-counts
    revenue-per-visitor for any repeat purchaser. This is the one place that
    collapse happens; every one of those call sites should build on this
    (or on `df["variant_user_pseudo_id"].nunique()` for a plain count) rather
    than re-deriving it locally, so they can't drift out of sync on what
    "visitor" means.

    RPT mode wants the raw per-order rows instead (each order is its own
    observation) -- do not use this for that path.
    """
    work = df.copy()
    work[kpi] = pd.to_numeric(work[kpi], errors="coerce").fillna(0.0)
    return (
        work.groupby(["experience_variant_label", "variant_user_pseudo_id"], as_index=False)[kpi]
        .sum()
    )


def run_continuous_analysis(
    df: "pd.DataFrame",
    control_label: str,
    variation_label: str,
    daily_visitors: float,
    projection_days: int,
    confidence_level: float = 0.95,
    tail: Literal["Two-sided", "Greater", "Less"] = "Two-sided",
    kpi: str = "purchase_revenue",
    mode: Literal["rpv", "rpt"] = "rpv",
) -> dict:
    """
    Runs the FOE ContinuousMetricEngine on row-level revenue data via the
    Gamma / two-part likelihood-ratio path (rather than the generic heuristic
    decision tree), since revenue is non-negative and right-skewed — the
    Gamma family models that shape directly instead of picking a generic test
    via a normality/variance decision tree. Also computes a revenue-impact
    projection.

    automation.py's continuous fetch always uses the "all_users" (LEFT JOIN)
    query mode, so every exposed visitor gets at least one row, non-buyers as
    NULL revenue (zero-filled below) rather than dropped. A visitor who placed
    more than one separate order within the date range gets *multiple* rows,
    though — one per transaction, from the underlying SQL's per-order grain —
    so the raw fetch is really "one row per (visitor, order)", not "one row
    per visitor". That distinction matters differently per mode:

    mode="rpv" (revenue per visitor): needs exactly one row per visitor
    (AnalysisUnit.PER_VISITOR — FOE's own hurdle model assumes the group mean
    already averages over every visitor). Rows are first collapsed to one per
    (variant, visitor) by summing revenue, so a repeat purchaser contributes
    their total spend as a single observation instead of being counted (and
    weighted in the significance test) once per order.
    mode="rpt" (revenue per transaction): the raw per-order grain *is* what's
    wanted — only buyers should count, so zero-revenue rows are stripped
    before analysis (AnalysisUnit.PER_TRANSACTION) — isolates the order-value
    effect alone, one observation per order.

    Each variant's total exposed visitor count (visitor_counts) is always the
    count of *distinct* visitors, regardless of mode — computed before any
    mode-specific collapsing/stripping, since FOE's PER_TRANSACTION business
    case needs a true visitor count (not an order count) to derive an order
    rate (n orders / total visitors); a repeat purchaser's extra rows must not
    inflate it.
    """
    alternative = _TAIL_MAP[tail]
    alpha = 1.0 - confidence_level

    # per_visitor is already collapsed to one row per (variant, visitor), so
    # a plain per-variant row count on it IS the true visitor_counts -- no
    # separate nunique() needed, and it can't drift from what the RPV branch
    # below actually analyzes since both come from the same collapse.
    per_visitor = collapse_to_per_visitor(df, kpi)
    visitor_counts = per_visitor.groupby("experience_variant_label").size().to_dict()

    work = df.copy()
    work[kpi] = pd.to_numeric(work[kpi], errors="coerce").fillna(0.0)

    if mode == "rpt":
        work = work[work[kpi] != 0.0]
        unit = AnalysisUnit.PER_TRANSACTION
    else:
        work = per_visitor
        unit = AnalysisUnit.PER_VISITOR

    config = ContinuousMetricConfig(
        kpi=kpi,
        group_col="experience_variant_label",
        approach=ContinuousApproach.GAMMA_GLM,
        unit=unit,
        control_label=control_label,
        alpha=alpha,
    )
    engine = ContinuousMetricEngine()
    result = engine.run_comparison_suite(work, config)

    group_stats = {row["experience_variant_label"]: row for row in result.summary_stats}
    monetary_list = engine.run_business_case(
        group_stats=group_stats,
        control_label=control_label,
        unit=unit,
        daily_visitors=daily_visitors,
        visitor_counts=visitor_counts,
        alpha=alpha,
        alternative=alternative,
        projection_period=projection_days,
        significance_by_variant={variation_label: result.is_significant},
    )
    monetary = next(
        (m for m in monetary_list if m["variant"] == variation_label),
        monetary_list[0] if monetary_list else None,
    )

    monetary_conclusion = (
        ContinuousMetricEngine.generate_monetary_conclusion(
            variant_name=variation_label,
            monetary_result=monetary,
            is_significant=result.is_significant,
        )
        if monetary else None
    )

    return {
        "method": "continuous",
        "mode": mode,
        "kpi": kpi,
        "test_name": result.test_name,
        "p_value": result.p_value,
        "is_significant": result.is_significant,
        "conclusion": result.conclusion,
        "monetary_conclusion": monetary_conclusion,
        "summary_stats": result.summary_stats,
        "effect_on_revenue": monetary["point_estimate"] if monetary else 0.0,
        "effect_on_revenue_ci": (monetary["ci_low"], monetary["ci_high"]) if monetary else (0.0, 0.0),
        "projection_days": projection_days,
    }


def run_pretest_analysis(
    weekly_visitors: float,
    weekly_conversions: float,
    weeks_used: int,
    kpi_label: str,
    confidence_level: float,
    tail: Literal["Two-sided", "Greater", "Less"],
    trust_pct: float = 80.0,
) -> dict:
    """
    Runs the FOE PretestEngine's 6-week MDE table on a pre-experiment baseline,
    fixed to num_variants=2 (this automation flow only supports one control +
    one variation). `weekly_visitors`/`weekly_conversions` are the pre-period
    total for the chosen KPI, already averaged down to a single full
    Monday-Sunday week (see automation.py's baseline-fetch step, which sums
    the whole pre-period and divides by the number of weeks queried).
    """
    alternative = _TAIL_MAP[tail]
    risk_pct = confidence_level * 100

    result = PretestEngine.calculate_mde_table(
        num_variants=2,
        baseline_visitors=int(round(weekly_visitors)),
        baseline_conversions=int(round(weekly_conversions)),
        risk_pct=risk_pct,
        trust_pct=trust_pct,
        alternative=alternative,
    )

    return {
        "method": "pretest",
        "kpi": kpi_label,
        "weekly_visitors": weekly_visitors,
        "weekly_conversions": weekly_conversions,
        "weeks_used": weeks_used,
        "table": result["table"],
        "conclusion": result["conclusion"],
    }


def run_pretest_analysis_seasonal(
    daily_df: "pd.DataFrame",
    value_col: str,
    kpi_label: str,
    weeks_used: int,
    confidence_level: float,
    tail: Literal["Two-sided", "Greater", "Less"],
    trust_pct: float = 80.0,
) -> dict:
    """
    Seasonal counterpart to run_pretest_analysis: fits a Prophet model to daily
    baseline visitors/conversions (via FOE's TrafficForecastingEngine) and
    projects a 6-week MDE table from that forecast instead of a flat weekly
    average, so weekday/holiday patterns in the baseline period inform the
    projection rather than being averaged away.

    Note: pages/pre_test_analysis.py has its own, independent Prophet-based
    seasonal forecast (run_prophet_forecast + perform_mde_calculation_forecast)
    for its dedicated planning tool. This uses FOE's TrafficForecastingEngine /
    PretestEngine.calculate_mde_from_forecast instead, since that page's helpers
    aren't safely importable here (it calls st.set_page_config at module level,
    which collides with automation.py's own call) — the two seasonal paths are
    functionally equivalent (same Prophet weekly/yearly-seasonality fit, same
    cumulative-week MDE formula) but independently implemented.

    daily_df needs a 'report_date' column (as produced by
    BaselineParams(output_shape="daily")) plus 'visitors' and `value_col`
    (either 'conversions' or 'add_to_cart_conversions', matching the chosen
    pretest KPI). Falls back to an explanatory empty-table result if there
    isn't enough daily history (Prophet needs >= 14 days) or the forecast
    comes back empty.
    """
    from foe.core.models import AnalysisUnit
    from foe.pretest.forecasting import TrafficForecastingEngine

    alternative = _TAIL_MAP[tail]
    risk_pct = confidence_level * 100

    df = daily_df.rename(columns={"report_date": "ds"})
    df["ds"] = pd.to_datetime(df["ds"])
    forecast_result = TrafficForecastingEngine.run_seasonal_forecast(
        df, periods=42, interval=confidence_level,
        unit=AnalysisUnit.PER_VISITOR, count_col="visitors", value_col=value_col,
    )
    records = forecast_result.get("forecast") or []
    if not records:
        return {
            "method": "pretest",
            "seasonal": True,
            "kpi": kpi_label,
            "weeks_used": weeks_used,
            "table": [],
            "conclusion": forecast_result.get(
                "conclusion",
                "Not enough daily baseline history for a seasonal forecast (need at least 14 days).",
            ),
        }

    result = PretestEngine.calculate_mde_from_forecast(
        forecast_records=records,
        num_variants=2,
        risk_pct=risk_pct,
        trust_pct=trust_pct,
        alternative=alternative,
        unit=AnalysisUnit.PER_VISITOR,
    )

    return {
        "method": "pretest",
        "seasonal": True,
        "kpi": kpi_label,
        "weeks_used": weeks_used,
        "table": result["table"],
        "conclusion": result["conclusion"],
        "forecast_summary": forecast_result["conclusion"],
    }


def format_pretest_table(table: list[dict]) -> str:
    """
    Renders the pre-test MDE table as plain text, one line per week, instead
    of json.dumps(...) — Airtable long-text fields don't render JSON or
    markdown tables, so a JSON blob just shows up as raw markup in the field.

    row['MDE'] is already a percentage (PretestEngine._relative_mde multiplies
    by 100 internally, matching pre_test_analysis.py's own relative_mde_pct
    convention) -- NOT a 0-1 fraction. Using Python's ':.2%' format spec here
    would multiply it by 100 a second time (12.42 -> "1242.29%" instead of
    "12.42%"), inflating every MDE in the table by exactly 100x before it
    ever reaches Airtable.

    MDE can be None on the seasonal path (PretestEngine.calculate_mde_from_forecast
    leaves a week's MDE unset when that week's forecasted volume is zero or
    negative) -- run_pretest_analysis's flat-average path never produces None,
    but this function is shared by both, so it's guarded here either way.
    """
    lines = []
    for row in table:
        size_key = next((k for k in row if k not in ("Week", "MDE")), None)
        size_label = size_key.replace("_", " ").lower() if size_key else ""
        size_val = row.get(size_key) if size_key else None
        size_part = f"{size_val:,} {size_label}" if size_val is not None else ""
        mde_val = row.get("MDE")
        mde_part = f"{mde_val:.2f}%" if mde_val is not None else "N/A"
        lines.append(f"Week {row['Week']}: {size_part} — MDE {mde_part}")
    return "\n".join(lines)


def build_airtable_payload(
    control: Optional[VariantData],
    variation: Optional[VariantData],
    frequentist_result: Optional[dict] = None,
    bayesian_result: Optional[dict] = None,
    continuous_result: Optional[dict] = None,
    revenue_source: Literal["frequentist", "bayesian", "continuous"] = "frequentist",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    pretest_result: Optional[dict] = None,
    ai_conclusion: Optional[str] = None,
) -> dict:
    """
    Builds the automation result keyed by internal semantic names (e.g.
    "visitors_control", "p_value") — NOT Airtable field names. Which base and
    table this is headed to (and therefore which field names exist) isn't
    known yet at this point in the flow; the Send step resolves that live and
    translates via apply_field_map.
    """
    fields: dict = {}
    if start_date:
        fields["start_date"] = start_date
    if end_date:
        fields["end_date"] = end_date

    if control is not None and variation is not None:
        fields.update({
            "visitors_control": control.visitors,
            "visitors_variation": variation.visitors,
            "conversions_control": control.conversions,
            "conversions_variation": variation.conversions,
        })

    if pretest_result:
        fields["pretest_mde_table"] = format_pretest_table(pretest_result["table"])

    if ai_conclusion:
        fields["ai_conclusion"] = ai_conclusion

    if frequentist_result:
        fields["p_value"] = round(frequentist_result["p_value"], 6)
    if bayesian_result:
        # Airtable's own Percent field format multiplies the stored value by
        # 100 for display, so it expects a raw proportion (0-1) — sending the
        # already-computed percentage (e.g. 83.42) makes Airtable show 8342%.
        fields["probability_pct"] = round(bayesian_result["probability_pct"] / 100, 6)
    if continuous_result:
        fields["continuous_p_value"] = round(continuous_result["p_value"], 6)
        fields["continuous_test_name"] = continuous_result["test_name"]

    revenue_result = {
        "frequentist": frequentist_result,
        "bayesian": bayesian_result,
        "continuous": continuous_result,
    }.get(revenue_source) or frequentist_result or bayesian_result or continuous_result
    if revenue_result:
        fields["effect_on_revenue"] = round(revenue_result["effect_on_revenue"], 2)

    return fields


def apply_field_map(payload: dict, field_map: dict) -> dict:
    """
    Translates an internal-key payload (from build_airtable_payload) into
    Airtable's actual field names for the chosen base/table, per field_map
    (internal key -> Airtable field name). Keys with no chosen mapping are
    omitted rather than sent under their internal name.
    """
    return {
        field_map[key]: value
        for key, value in payload.items()
        if field_map.get(key)
    }


def _normalize_field_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def guess_field_by_hints(hints: list[str], available_fields: list) -> Optional[str]:
    """
    Matches a list of candidate names against a table's real field names —
    shared by best_match_field (Airtable-push field assignment) and
    automation.py's read-side lookups (experiment ID, Hypothesis, Custom
    Code fields on an existing record). Returns None if there's no match.

    Tries, in order: case-insensitive exact match against any hint, then a
    normalized (spaces/punctuation stripped) exact match, then a normalized
    substring match — e.g. "start_date" matches a field named "Test start
    date" or "Startdatum" even though neither is an exact hint.
    """
    if not hints:
        return None

    lowered = {f.lower(): f for f in available_fields}
    for hint in hints:
        if hint.lower() in lowered:
            return lowered[hint.lower()]

    normalized_fields = {_normalize_field_name(f): f for f in available_fields}
    for hint in hints:
        norm_hint = _normalize_field_name(hint)
        if norm_hint in normalized_fields:
            return normalized_fields[norm_hint]

    for hint in hints:
        norm_hint = _normalize_field_name(hint)
        for norm_field, original in normalized_fields.items():
            if norm_hint in norm_field or norm_field in norm_hint:
                return original
    return None


def best_match_field(internal_key: str, available_fields: list) -> Optional[str]:
    """
    Matches internal_key's default hints (see DEFAULT_FIELD_NAME_HINTS)
    against a table's real field names — used only to auto-preselect a
    sensible default in the assignment UI. Returns None if there's no hint
    or no match, leaving the choice to the user.
    """
    return guess_field_by_hints(DEFAULT_FIELD_NAME_HINTS.get(internal_key, []), available_fields)
