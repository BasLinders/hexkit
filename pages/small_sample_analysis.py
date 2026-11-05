"""
Small Sample Analysis — exact testing for conversion data with few units.

Runs Boschloo's exact unconditional test (FOE BoschlooEngine) when every arm
is small enough for it to finish in reasonable time, and falls back to the
z-test (FOE FrequentistEngine.run_synthesis) above that size, telling the user
which test ran and why.

An optional email mode relabels the inputs for campaign data (delivered,
unique opens/clicks/conversions), lets the user pick the rate to test, and
projects the impact of sending the winner to the rest of the list.
"""

import concurrent.futures
import math
import string
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from foe.core.models import AlternativeHypothesis, ExperimentInput
from foe.frequentist.boschloo import BoschlooEngine
from foe.frequentist.operations import FrequentistEngine, apply_sidak

from utility.small_sample import BOSCHLOO_MAX_PER_ARM

st.set_page_config(page_title="Small Sample Analysis", page_icon="🔎", layout="wide")

# Runtime model for one Boschloo comparison, fitted on scipy 1.15 with the
# engine's default 32 sampling points: ~0.14s at 100x100, 0.7s at 300x300,
# 3s at 500x500, ~20s at 1000x1000. Only drives the progress bar.
_EST_BASE_S = 0.1
_EST_COEF = 8e-8
_EST_EXP = 1.4

# The z-test's normal approximation is shaky when any arm has fewer than this
# many successes or failures.
MIN_EXPECTED_CELL = 5

_TAIL_TO_ALTERNATIVE = {
    "Two-sided": AlternativeHypothesis.TWO_SIDED,
    "Greater": AlternativeHypothesis.GREATER,
    "Less": AlternativeHypothesis.LESS,
}

# Email metrics: (numerator field, denominator field). Fields are unique
# per-recipient counts so each recipient contributes one binary outcome.
EMAIL_FIELDS = {
    "delivered": "Delivered",
    "opens": "Unique opens",
    "clicks": "Unique clicks",
    "conversions": "Conversions",
}
EMAIL_METRICS: Dict[str, Tuple[str, str]] = {
    "Open rate": ("opens", "delivered"),
    "Click rate (CTR)": ("clicks", "delivered"),
    "Click-to-open rate (CTOR)": ("clicks", "opens"),
    "Conversion rate": ("conversions", "delivered"),
}


@dataclass
class ComparisonResult:
    """Common shape for Boschloo and z-test results so one renderer serves both."""

    variant_label: str
    control_label: str
    control_rate: float
    rate: float
    diff: float
    uplift: Optional[float]
    p_value: float
    alpha: float
    is_significant: bool
    ci: Tuple[float, float]
    conclusion: str


# --------------------------------------------------------------------------- #
#  State
# --------------------------------------------------------------------------- #

def initialize_session_state():
    st.session_state.setdefault("ssa_email_mode", False)
    st.session_state.setdefault("ssa_num_variants", 2)
    st.session_state.setdefault("ssa_conf", 95)
    st.session_state.setdefault("ssa_tail", "Two-sided")
    st.session_state.setdefault("ssa_metric", "Click rate (CTR)")
    st.session_state.setdefault("ssa_remaining", 0)

    # Inputs handed over from Experiment Analysis. Applied before any widget
    # renders so the widgets pick the values up from their keys.
    prefill = st.session_state.pop("ssa_prefill", None)
    if prefill:
        st.session_state.ssa_email_mode = False
        st.session_state.ssa_num_variants = len(prefill["visitors"])
        for i, (n, c) in enumerate(zip(prefill["visitors"], prefill["conversions"])):
            st.session_state[f"ssa_n_{i}"] = int(n)
            st.session_state[f"ssa_c_{i}"] = int(c)
        st.session_state.ssa_conf = int(prefill["confidence_level"])
        st.session_state.ssa_tail = prefill["tail"]
        st.toast("Loaded your inputs from Experiment Analysis.")


def default_label(i: int) -> str:
    letter = string.ascii_uppercase[i]
    return f"{letter} (control)" if i == 0 else letter


# --------------------------------------------------------------------------- #
#  Inputs
# --------------------------------------------------------------------------- #

def get_general_inputs(num_variants: int):
    """Units and conversions per variant. Returns (labels, n, successes)."""
    st.caption(
        "Enter one count per independent unit (user, recipient, account). "
        "Each unit can convert at most once."
    )
    labels, visitors, conversions = [], [], []
    for i in range(num_variants):
        c1, c2, c3 = st.columns([2, 2, 2])
        st.session_state.setdefault(f"ssa_label_{i}", default_label(i))
        st.session_state.setdefault(f"ssa_n_{i}", 0)
        st.session_state.setdefault(f"ssa_c_{i}", 0)
        with c1:
            labels.append(st.text_input(f"Variant {i + 1} name", key=f"ssa_label_{i}"))
        with c2:
            visitors.append(st.number_input(
                f"Sample size ({labels[i]})", min_value=0, step=1, key=f"ssa_n_{i}"
            ))
        with c3:
            conversions.append(st.number_input(
                f"Conversions ({labels[i]})", min_value=0, step=1, key=f"ssa_c_{i}"
            ))
    return labels, visitors, conversions, "conversions", "units"


def get_email_inputs(num_variants: int):
    """
    Campaign counts per variant, reduced to the chosen metric's numerator and
    denominator. Returns (labels, denominators, numerators, metric noun, unit noun).
    """
    metric = st.selectbox(
        "Metric to test",
        list(EMAIL_METRICS),
        key="ssa_metric",
        help=(
            "Open rate and CTR use delivered emails as the base. CTOR uses unique "
            "opens as the base, so it measures the content of the email for people "
            "who opened it."
        ),
    )
    num_field, den_field = EMAIL_METRICS[metric]

    if "opens" in (num_field, den_field):
        st.warning(
            "**Open tracking is unreliable.** Apple Mail Privacy Protection and some "
            "corporate security scanners load the tracking pixel without a person "
            "opening the email. Open counts are inflated, and how much differs by "
            "audience. Prefer CTR or conversions to decide a winner."
        )
    if metric == "Click-to-open rate (CTOR)":
        st.info(
            "CTOR divides by opens. If your variants differ in subject line or "
            "sender, they also change *who* opens, so a CTOR difference mixes "
            "audience and content effects. CTOR fits best when only the email body "
            "differs."
        )

    st.caption(
        "Use **unique** counts: each recipient counts once, however many times "
        "they opened or clicked. Use delivered emails, not sent ones (exclude bounces)."
    )

    labels, denominators, numerators = [], [], []
    fields = [den_field, num_field]
    for i in range(num_variants):
        cols = st.columns([2] + [2] * len(fields))
        st.session_state.setdefault(f"ssa_label_{i}", default_label(i))
        with cols[0]:
            labels.append(st.text_input(f"Variant {i + 1} name", key=f"ssa_label_{i}"))
        values = {}
        for col, field in zip(cols[1:], fields):
            key = f"ssa_email_{field}_{i}"
            st.session_state.setdefault(key, 0)
            with col:
                values[field] = st.number_input(
                    f"{EMAIL_FIELDS[field]} ({labels[i]})", min_value=0, step=1, key=key
                )
        denominators.append(values[den_field])
        numerators.append(values[num_field])

    return labels, denominators, numerators, EMAIL_FIELDS[num_field].lower(), EMAIL_FIELDS[den_field].lower()


def validate(labels, denominators, numerators, num_noun, den_noun) -> bool:
    ok = True
    for label, n, c in zip(labels, denominators, numerators):
        if n <= 0:
            st.error(f"**{label}**: {den_noun} must be greater than zero.")
            ok = False
        elif c > n:
            st.error(
                f"**{label}**: {num_noun} ({c:,}) exceed {den_noun} ({n:,}). "
                "Check that you used unique counts."
            )
            ok = False
    if len(set(labels)) != len(labels):
        st.error("Variant names must be unique.")
        ok = False
    return ok


# --------------------------------------------------------------------------- #
#  Method selection and execution
# --------------------------------------------------------------------------- #

def choose_method(visitors: List[int]) -> Tuple[str, str]:
    """Returns ("boschloo" | "ztest", reason shown to the user)."""
    largest = max(visitors)
    if largest <= BOSCHLOO_MAX_PER_ARM:
        return "boschloo", (
            f"Every variant has at most {BOSCHLOO_MAX_PER_ARM:,} units, so the exact "
            "test (Boschloo) is used. It controls the false-positive rate at any "
            "sample size and handles zero conversions and uneven splits."
        )
    return "ztest", (
        f"The largest variant has {largest:,} units, above the exact-test limit of "
        f"{BOSCHLOO_MAX_PER_ARM:,}. At this size the z-test is accurate and the exact "
        "test would take too long, so the z-test is used."
    )


def has_sparse_cells(visitors: List[int], conversions: List[int]) -> bool:
    return any(min(c, n - c) < MIN_EXPECTED_CELL for n, c in zip(visitors, conversions))


def estimate_seconds(visitors: List[int]) -> float:
    ctrl = visitors[0]
    return sum(_EST_BASE_S + _EST_COEF * (ctrl * n) ** _EST_EXP for n in visitors[1:])


def _progress_fraction(elapsed: float, estimate: float) -> float:
    """Linear to 90% at the estimate, then creeps toward 99% if it runs long."""
    if elapsed <= estimate:
        return 0.9 * elapsed / estimate
    return 0.9 + 0.09 * (1 - math.exp(-(elapsed - estimate) / estimate))


def run_boschloo_with_progress(data: ExperimentInput) -> list:
    """
    Runs BoschlooEngine.run in a worker thread and drives a progress bar from
    the runtime estimate. scipy's boschloo_exact is a single blocking call with
    no progress hook, so the bar reflects expected rather than measured progress.
    """
    estimate = max(estimate_seconds(data.visitors), 0.2)
    bar = st.progress(0.0, text="Running Boschloo's exact test…")
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(BoschlooEngine(max_visitors_per_arm=BOSCHLOO_MAX_PER_ARM).run, data)
        while not future.done():
            elapsed = time.monotonic() - start
            bar.progress(
                _progress_fraction(elapsed, estimate),
                text=(
                    f"Running Boschloo's exact test… {elapsed:.0f}s elapsed "
                    f"(estimated ~{math.ceil(estimate)}s)"
                ),
            )
            time.sleep(0.1)
        results = future.result()
    bar.progress(1.0, text=f"Done in {time.monotonic() - start:.1f}s")
    return results


def run_analysis(method: str, data: ExperimentInput) -> List[ComparisonResult]:
    p_ctrl = data.conversions[0] / data.visitors[0]

    if method == "boschloo":
        return [
            ComparisonResult(
                variant_label=r.variant_label,
                control_label=r.control_label,
                control_rate=r.control_rate,
                rate=r.conversion_rate,
                diff=r.absolute_diff,
                uplift=r.uplift,
                p_value=r.p_value,
                alpha=r.alpha,
                is_significant=r.is_significant,
                ci=r.ci_diff,
                conclusion=r.conclusion,
            )
            for r in run_boschloo_with_progress(data)
        ]

    alpha = apply_sidak(1.0 - data.confidence_level, len(data.visitors))
    with st.spinner("Running z-test…"):
        raw = FrequentistEngine().run_synthesis(data)
    out = []
    for r in raw:
        diff = r.conversion_rate - p_ctrl
        out.append(ComparisonResult(
            variant_label=r.variant_label,
            control_label=r.control_label,
            control_rate=p_ctrl,
            rate=r.conversion_rate,
            diff=diff,
            # run_synthesis reports 0.0 for a zero control rate; relative lift
            # is undefined there, so show it as such.
            uplift=diff / p_ctrl if p_ctrl > 0 else None,
            p_value=r.p_value,
            alpha=alpha,
            is_significant=r.is_significant,
            ci=r.ci_diff,
            conclusion=r.conclusion,
        ))
    return out


# --------------------------------------------------------------------------- #
#  Output
# --------------------------------------------------------------------------- #

def fmt_pp(x: float) -> str:
    """Formats a rate difference in percentage points; infinities as unbounded."""
    if math.isinf(x):
        return "+∞" if x > 0 else "−∞"
    return f"{x * 100:+.2f} pp"


def fmt_ci(ci: Tuple[float, float]) -> str:
    return f"[{fmt_pp(ci[0])}, {fmt_pp(ci[1])}]"


def display_comparison(r: ComparisonResult, confidence_level: int, method: str, sidak: bool):
    st.subheader(f"{r.variant_label} vs {r.control_label}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{r.control_label} rate", f"{r.control_rate:.2%}")
    c2.metric(
        f"{r.variant_label} rate",
        f"{r.rate:.2%}",
        delta=fmt_pp(r.diff),
    )
    c3.metric(
        "Relative uplift",
        f"{r.uplift:+.2%}" if r.uplift is not None else "n/a",
        help=None if r.uplift is not None else "Undefined: the control had zero conversions.",
    )
    c4.metric(
        "p-value",
        f"{r.p_value:.4f}",
        help=f"Significant when below α = {r.alpha:.4f}.",
    )

    ci_name = "Newcombe interval" if method == "boschloo" else "Confidence interval"
    st.write(
        f"**{ci_name} for the difference ({confidence_level}%"
        f"{', Šidák-adjusted' if sidak else ''}):** "
        f"{fmt_ci(r.ci)}"
    )

    if not r.is_significant:
        st.info(r.conclusion)
    elif r.diff > 0:
        st.success(r.conclusion)
    else:
        st.error(r.conclusion)


def plot_differences(results: List[ComparisonResult]) -> alt.Chart:
    """Point estimate and interval for each variant's difference vs control."""
    finite = [b for r in results for b in (r.diff, *r.ci) if not math.isinf(b)]
    span = max((abs(b) for b in finite), default=0.01) or 0.01
    pad = span * 0.5

    rows = []
    for r in results:
        lo = r.ci[0] if not math.isinf(r.ci[0]) else -span - pad
        hi = r.ci[1] if not math.isinf(r.ci[1]) else span + pad
        rows.append({
            "Variant": r.variant_label,
            "Difference": r.diff * 100,
            "Lower": lo * 100,
            "Upper": hi * 100,
            "Significant": "Yes" if r.is_significant else "No",
        })
    df = pd.DataFrame(rows)

    base = alt.Chart(df).encode(y=alt.Y("Variant:N", title=None))
    rules = base.mark_rule(strokeWidth=3).encode(
        x=alt.X("Lower:Q", title="Difference vs control (percentage points)"),
        x2="Upper:Q",
        color=alt.Color("Significant:N", scale=alt.Scale(domain=["Yes", "No"], range=["#2a9d8f", "#9aa5b1"])),
    )
    points = base.mark_point(filled=True, size=90, color="black").encode(
        x="Difference:Q",
        tooltip=[
            "Variant",
            alt.Tooltip("Difference:Q", format="+.2f"),
            alt.Tooltip("Lower:Q", format="+.2f"),
            alt.Tooltip("Upper:Q", format="+.2f"),
        ],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="x:Q")
    return (rules + points + zero).properties(height=60 + 40 * len(results))


def display_projection(results: List[ComparisonResult], remaining: int, num_noun: str):
    """Extra outcomes if each variant were sent to the rest of the list instead of control."""
    st.subheader("Projection for the rest of the list")
    st.caption(
        f"If you send a variant instead of {results[0].control_label} to the remaining "
        f"{remaining:,} recipients, these are the expected extra {num_noun}. The range "
        "uses the same interval as above."
    )

    def bound(x: float) -> str:
        if math.isinf(x):
            return "unbounded"
        return f"{x * remaining:+,.0f}"

    rows = [{
        "Variant": r.variant_label,
        f"Expected extra {num_noun}": f"{r.diff * remaining:+,.0f}",
        "Low end": bound(r.ci[0]),
        "High end": bound(r.ci[1]),
        "Significant": "Yes" if r.is_significant else "No",
    } for r in results]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    winners = [r for r in results if r.is_significant and r.diff > 0]
    if winners:
        best = max(winners, key=lambda r: r.diff)
        st.success(
            f"**{best.variant_label}** is the significant winner. Sending it to the "
            f"remaining list adds about {best.diff * remaining:,.0f} {num_noun}."
        )
    else:
        st.info(
            "No variant beats the control significantly. The projection shows the "
            "plausible range, but the data does not favour any variant yet."
        )


# --------------------------------------------------------------------------- #
#  Page
# --------------------------------------------------------------------------- #

def display_documentation():
    with st.expander("When to use this page", expanded=False):
        st.markdown(f"""
The z-test in **Experiment Analysis** relies on a normal approximation. That
approximation breaks down when samples are small: few conversions, a variant
at 0% or 100%, or a very uneven split (for example a 90/10 holdout). It can
then report significance that isn't there, or miss a real effect.

This page uses **Boschloo's exact test**, which computes the p-value from
every possible outcome instead of an approximation. It keeps the
false-positive rate at your chosen level at any sample size, and it is more
powerful than Fisher's exact test.

**How the test is chosen**
- Every variant has at most {BOSCHLOO_MAX_PER_ARM:,} units: Boschloo's exact test.
- Larger: the z-test. It is accurate at that size, and the exact test would
  take much longer.

**Typical uses**
- Email A/B tests on small lists or segments (turn on *Email campaign mode*).
- Low-traffic pages, B2B funnels, or rare micro-conversions.
- Pilots and holdout groups with an uneven split.

**Limits**
- One outcome per independent unit. Counts must be unique users or recipients,
  not sessions or total clicks.
- Tests "is there a difference" only. Non-inferiority and overdispersion
  correction are in Experiment Analysis.
- With three or more variants, α is Šidák-adjusted for multiple comparisons.
""")


def run():
    initialize_session_state()

    st.title("Small Sample Analysis")
    st.markdown(
        "Exact significance testing for conversion data with small samples, "
        "zero conversions or uneven splits."
    )
    display_documentation()
    st.write("---")

    email_mode = st.toggle(
        "Email campaign mode",
        key="ssa_email_mode",
        help="Enter delivered, unique opens, clicks and conversions per variant, "
             "and choose which rate to test.",
    )
    num_variants = st.number_input(
        "Number of variants (including control)",
        min_value=2, max_value=5, step=1, key="ssa_num_variants",
    )
    st.write("")

    if email_mode:
        labels, visitors, conversions, num_noun, den_noun = get_email_inputs(num_variants)
    else:
        labels, visitors, conversions, num_noun, den_noun = get_general_inputs(num_variants)

    st.write("---")
    c1, c2 = st.columns(2)
    with c1:
        confidence_level = st.slider(
            "Confidence level (%)", min_value=50, max_value=99, step=1, key="ssa_conf"
        )
    with c2:
        tail = st.radio(
            "Hypothesis",
            list(_TAIL_TO_ALTERNATIVE),
            horizontal=True,
            key="ssa_tail",
            help="'Two-sided' (B ≠ A), 'Greater' (B > A), 'Less' (B < A). "
                 "Pick one-sided only if you decided the direction before the test.",
        )

    remaining = 0
    if email_mode and EMAIL_METRICS[st.session_state.ssa_metric][1] == "delivered":
        remaining = st.number_input(
            "Remaining list size (optional)",
            min_value=0, step=100, key="ssa_remaining",
            help="If this was a test on part of your list, enter how many recipients "
                 "will get the winning version. The results then project the extra "
                 "outcomes from sending each variant.",
        )

    st.write("")
    if not st.button("Calculate", type="primary"):
        return
    if not validate(labels, visitors, conversions, num_noun, den_noun):
        return

    method, reason = choose_method(visitors)
    st.write("---")
    st.info(f"**Test used: {'Boschloo exact test' if method == 'boschloo' else 'z-test'}.** {reason}")
    if method == "ztest" and has_sparse_cells(visitors, conversions):
        st.warning(
            f"At least one variant has fewer than {MIN_EXPECTED_CELL} {num_noun} (or "
            f"non-{num_noun}). The z-test is approximate here, so treat a borderline "
            "p-value with caution."
        )

    data = ExperimentInput(
        visitors=[int(v) for v in visitors],
        conversions=[int(c) for c in conversions],
        alternative=_TAIL_TO_ALTERNATIVE[tail],
        confidence_level=confidence_level / 100,
        labels=labels,
    )
    try:
        results = run_analysis(method, data)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return

    for r in results:
        display_comparison(r, confidence_level, method, sidak=len(results) > 1)
        st.write("")

    st.subheader("Overview")
    st.altair_chart(plot_differences(results), width="stretch")
    if any(math.isinf(b) for r in results for b in r.ci):
        st.caption("One-sided test: the unbounded side of each interval is cut off at the chart edge.")

    if remaining > 0:
        st.write("---")
        display_projection(results, remaining, num_noun)


if __name__ == "__main__":
    run()
