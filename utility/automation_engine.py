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
    "description": "Description",
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
    "pretest_mde_table": ["pre-test mde table"],
    "ai_conclusion": ["ai conclusion"],
    "description": ["description"],
}

_TAIL_MAP = {
    "Two-sided": AlternativeHypothesis.TWO_SIDED,
    "Greater": AlternativeHypothesis.GREATER,
    "Less": AlternativeHypothesis.LESS,
}

TAILS = list(_TAIL_MAP.keys())


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

    return {
        "method": "frequentist",
        "p_value": result.p_value,
        "is_significant": result.is_significant,
        "uplift": result.uplift,
        "ci_diff": result.ci_diff,
        "conclusion": result.conclusion,
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
) -> dict:
    """Runs the FOE BayesianEngine (Beta-Binomial Monte Carlo) on a control/variation pair."""
    data = ExperimentInput(
        visitors=[control.visitors, variation.visitors],
        conversions=[control.conversions, variation.conversions],
        labels=[control.label, variation.label],
    )
    engine = BayesianEngine()
    prob_result = engine.run_probability_analysis(data, n_samples=n_samples)[0]

    biz_case = BusinessCaseInput(
        aovs={control.label: control.aov, variation.label: variation.aov},
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
        n_simulations=n_samples,
    )[0]

    return {
        "method": "bayesian",
        "probability_pct": prob_result.prob_beat_control * 100,
        "prob_being_best": prob_result.prob_being_best,
        "expected_uplift": prob_result.expected_uplift,
        "expected_loss": prob_result.expected_loss,
        "conclusion": prob_result.conclusion,
        "effect_on_revenue": monetary["expected_total_contribution"],
        "expected_revenue_uplift": monetary["expected_uplift"],
        "expected_revenue_risk": monetary["expected_risk"],
        "projection_days": projection_days,
    }


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
    query mode, so every exposed visitor is a row, non-buyers as NULL revenue
    (zero-filled below) rather than dropped — this is what makes both modes
    below possible from the *same* fetched data:

    mode="rpv" (revenue per visitor): every row counts as-is — measures
    conversion-rate AND spend effects together (AnalysisUnit.PER_VISITOR).
    mode="rpt" (revenue per transaction): only buyers should count, so zero-
    revenue rows are stripped before analysis (AnalysisUnit.PER_TRANSACTION)
    — isolates the order-value effect alone. Each variant's total exposed
    visitor count is taken from the data *before* that stripping and passed
    through to the monetary projection regardless of mode, since FOE's
    PER_TRANSACTION business case needs it to derive an order rate (n
    orders / total visitors) — without it, a per-order lift can't be scaled
    to daily traffic.
    """
    alternative = _TAIL_MAP[tail]
    alpha = 1.0 - confidence_level

    work = df.copy()
    work[kpi] = pd.to_numeric(work[kpi], errors="coerce").fillna(0.0)
    visitor_counts = work["experience_variant_label"].value_counts().to_dict()

    if mode == "rpt":
        work = work[work[kpi] != 0.0]
        unit = AnalysisUnit.PER_TRANSACTION
    else:
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

    return {
        "method": "continuous",
        "mode": mode,
        "kpi": kpi,
        "test_name": result.test_name,
        "p_value": result.p_value,
        "is_significant": result.is_significant,
        "conclusion": result.conclusion,
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


def format_pretest_table(table: list[dict]) -> str:
    """
    Renders the pre-test MDE table as plain text, one line per week, instead
    of json.dumps(...) — Airtable long-text fields don't render JSON or
    markdown tables, so a JSON blob just shows up as raw markup in the field.
    """
    lines = []
    for row in table:
        size_key = next((k for k in row if k not in ("Week", "MDE")), None)
        size_label = size_key.replace("_", " ").lower() if size_key else ""
        size_val = row.get(size_key) if size_key else None
        size_part = f"{size_val:,} {size_label}" if size_val is not None else ""
        lines.append(f"Week {row['Week']}: {size_part} — MDE {row['MDE']:.2%}")
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


def best_match_field(internal_key: str, available_fields: list) -> Optional[str]:
    """
    Matches internal_key's default hints (see DEFAULT_FIELD_NAME_HINTS)
    against a table's real field names — used only to auto-preselect a
    sensible default in the assignment UI. Returns None if there's no hint
    or no match, leaving the choice to the user.

    Tries, in order: case-insensitive exact match against any hint, then a
    normalized (spaces/punctuation stripped) exact match, then a normalized
    substring match — e.g. "start_date" matches a field named "Test start
    date" or "Startdatum" even though neither is an exact hint.
    """
    hints = DEFAULT_FIELD_NAME_HINTS.get(internal_key, [])
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
