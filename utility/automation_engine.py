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

from foe.core.models import AlternativeHypothesis, BusinessCaseInput, ExperimentInput
from foe.frequentist.operations import FrequentistEngine
from foe.bayesian.operations import BayesianEngine

# Provisional — the real Airtable base's field names aren't known yet.
# Update this mapping once they are; nothing else in this module needs to change.
AIRTABLE_FIELD_MAP = {
    "visitors_control": "visitors - control",
    "visitors_variation": "visitors - variation",
    "conversions_control": "conversions - control",
    "conversions_variation": "conversions - variation",
    "probability_pct": "probability (%)",
    "p_value": "p-value",
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


def build_airtable_payload(
    control: VariantData,
    variation: VariantData,
    frequentist_result: Optional[dict] = None,
    bayesian_result: Optional[dict] = None,
    revenue_source: Literal["frequentist", "bayesian"] = "frequentist",
) -> dict:
    """Maps engine outputs onto Airtable field names via AIRTABLE_FIELD_MAP."""
    fields = {
        AIRTABLE_FIELD_MAP["visitors_control"]: control.visitors,
        AIRTABLE_FIELD_MAP["visitors_variation"]: variation.visitors,
        AIRTABLE_FIELD_MAP["conversions_control"]: control.conversions,
        AIRTABLE_FIELD_MAP["conversions_variation"]: variation.conversions,
    }

    if frequentist_result:
        fields[AIRTABLE_FIELD_MAP["p_value"]] = round(frequentist_result["p_value"], 6)
    if bayesian_result:
        fields[AIRTABLE_FIELD_MAP["probability_pct"]] = round(bayesian_result["probability_pct"], 2)

    revenue_result = (
        bayesian_result if revenue_source == "bayesian" and bayesian_result else frequentist_result
    ) or bayesian_result
    if revenue_result:
        fields[AIRTABLE_FIELD_MAP["effect_on_revenue"]] = round(revenue_result["effect_on_revenue"], 2)

    return fields
