"""
Shared UI helpers for FOE's lift-prior widths, used by the Experiment Analysis
and Sequential Analysis pages.

The plausible-lift ranges in the labels are computed from FOE's own
get_lift_prior, so they stay in sync with the engine if its widths change.
"""

from typing import Optional, Tuple

import numpy as np

from foe.bayesian.operations import LiftPrior, get_lift_prior

PRIOR_WIDTHS = ["narrow", "medium", "wide"]

# Names used before FOE renamed the widths; FOE still accepts them, but a
# value stored in session state must map onto the current dropdown options.
_LEGACY_WIDTHS = {"skeptical": "narrow", "moderate": "medium", "uninformative": "wide"}

_Z_95 = 1.96


def prior_range_pct(lift_prior: LiftPrior) -> Tuple[float, float]:
    """Relative-lift range (in %) holding 95% of the prior's mass."""
    low = np.expm1(lift_prior.mean_log_lift - _Z_95 * lift_prior.std_log_lift) * 100
    high = np.expm1(lift_prior.mean_log_lift + _Z_95 * lift_prior.std_log_lift) * 100
    return float(low), float(high)


def width_label(width: str) -> str:
    """Dropdown label, e.g. 'Narrow (about -18% to +22%)' (at 0% expected lift)."""
    low, high = prior_range_pct(get_lift_prior(expected_lift_pct=0.0, skepticism=width))
    return f"{width.capitalize()} (about {low:+.0f}% to {high:+.0f}%)"


def normalize_width(value: Optional[str], default: str = "narrow") -> str:
    """Maps a stored width (current or legacy name) onto PRIOR_WIDTHS."""
    value = _LEGACY_WIDTHS.get(value, value)
    return value if value in PRIOR_WIDTHS else default
