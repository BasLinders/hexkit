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

# Provisional — the real Airtable base's field names aren't known yet.
# Update this mapping once they are; nothing else in this module needs to change.
AIRTABLE_FIELD_MAP = {
    "visitors_control": "visitors - control",
    "visitors_variation": "visitors - variation",
    "conversions_control": "conversions - control",
    "conversions_variation": "conversions - variation",
    "probability_pct": "probability (%)",
    "p_value": "p-value",
    "continuous_p_value": "p-value (continuous)",
    "continuous_test_name": "test used (continuous)",
    "effect_on_revenue": "effect on revenue",
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
) -> dict:
    """
    Runs the FOE ContinuousMetricEngine on row-level revenue-per-visitor data
    (the heuristic path auto-selects Mann-Whitney / ANOVA / Welch / Negative-
    Binomial as appropriate), plus a revenue-impact projection.

    Assumes per-visitor data — automation.py's continuous fetch always uses
    the "all_users" (RPV) query mode, whose LEFT JOIN leaves non-buyers as
    NULL rather than 0; zero-filled here so they're correctly treated as
    non-converting visitors rather than dropped.
    """
    alternative = _TAIL_MAP[tail]
    alpha = 1.0 - confidence_level

    work = df.copy()
    work[kpi] = pd.to_numeric(work[kpi], errors="coerce").fillna(0.0)

    config = ContinuousMetricConfig(
        kpi=kpi,
        group_col="experience_variant_label",
        approach=ContinuousApproach.HEURISTIC,
        unit=AnalysisUnit.PER_VISITOR,
        control_label=control_label,
        alpha=alpha,
    )
    engine = ContinuousMetricEngine()
    result = engine.run_comparison_suite(work, config)

    group_stats = {row["experience_variant_label"]: row for row in result.summary_stats}
    monetary_list = engine.run_business_case(
        group_stats=group_stats,
        control_label=control_label,
        unit=AnalysisUnit.PER_VISITOR,
        daily_visitors=daily_visitors,
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


def build_airtable_payload(
    control: Optional[VariantData],
    variation: Optional[VariantData],
    frequentist_result: Optional[dict] = None,
    bayesian_result: Optional[dict] = None,
    continuous_result: Optional[dict] = None,
    revenue_source: Literal["frequentist", "bayesian", "continuous"] = "frequentist",
) -> dict:
    """Maps engine outputs onto Airtable field names via AIRTABLE_FIELD_MAP."""
    fields: dict = {}
    if control is not None and variation is not None:
        fields.update({
            AIRTABLE_FIELD_MAP["visitors_control"]: control.visitors,
            AIRTABLE_FIELD_MAP["visitors_variation"]: variation.visitors,
            AIRTABLE_FIELD_MAP["conversions_control"]: control.conversions,
            AIRTABLE_FIELD_MAP["conversions_variation"]: variation.conversions,
        })

    if frequentist_result:
        fields[AIRTABLE_FIELD_MAP["p_value"]] = round(frequentist_result["p_value"], 6)
    if bayesian_result:
        fields[AIRTABLE_FIELD_MAP["probability_pct"]] = round(bayesian_result["probability_pct"], 2)
    if continuous_result:
        fields[AIRTABLE_FIELD_MAP["continuous_p_value"]] = round(continuous_result["p_value"], 6)
        fields[AIRTABLE_FIELD_MAP["continuous_test_name"]] = continuous_result["test_name"]

    revenue_result = {
        "frequentist": frequentist_result,
        "bayesian": bayesian_result,
        "continuous": continuous_result,
    }.get(revenue_source) or frequentist_result or bayesian_result or continuous_result
    if revenue_result:
        fields[AIRTABLE_FIELD_MAP["effect_on_revenue"]] = round(revenue_result["effect_on_revenue"], 2)

    return fields
