from email import message
import streamlit as st
import pandas as pd
import numpy as np
import string
import io
import concurrent.futures
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib import colormaps
from matplotlib.figure import Figure
import plotly.graph_objects as go
from scipy.stats import beta, norm, chisquare
import math
from dataclasses import dataclass
from typing import Literal, List, Tuple, Dict, Any, cast


st.set_page_config(
    page_title="Experiment Analysis",
    page_icon="🔢",
    layout="wide",
)

def initialize_session_state():
    st.session_state.setdefault("num_variants", 2)
    num_variants = st.session_state.num_variants
    st.session_state.setdefault("visitor_counts", [0] * num_variants)
    st.session_state.setdefault("conversion_counts", [0] * num_variants)
    st.session_state.setdefault("aovs", [None] * num_variants)
    st.session_state.setdefault("aov_cv", 0.5)
    st.session_state.setdefault("confidence_level", 95)
    st.session_state.setdefault("test_duration", 7)
    st.session_state.setdefault("tail", 'Greater')
    st.session_state.setdefault("probability_winner", 80.0)
    st.session_state.setdefault("runtime_days", 0)

# --- Information / documentation ---
def display_dynamic_documentation(analysis_method):
    st.subheader(f"Documentation: {analysis_method}")

    if analysis_method == "Frequentist Analysis":
        with st.expander("Frequentist Analysis: What it provides"):
            st.markdown("""
            This engine focuses on **Null Hypothesis Statistical Testing (NHST)** to determine if the observed difference between variants is statistically significant.
            
            * **Fixed-Horizon Testing:** Ideal for tests where you have a pre-determined sample size.
            * **Error Control:** Strictly controls for **Type I Errors** (False Positives) via $p$-values.
            * **Dual-Gate Evaluation:** First, it checks for a traditional superior win ($p < 0.05$). If that isn't met, it automatically pivots to a **Non-Inferiority Test**.
            * **Risk Mitigation:** Specifically designed for feature parity tests or migrations where the primary goal is "Do No Harm."
            * **Variance-Adjusted Sensitivity:** Uses historical daily aggregate data to detect whether your conversion rate is more stable or noisier than pure binomial sampling predicts, allowing for a tighter - or more conservative - assessment of the difference between variants.
            """)

        with st.expander("Frequentist: How it works"):
            st.markdown(r"""
            1.  **Metric Calculation:** Calculates the standard conversion rates and uses a $z$-test for proportions.
            2.  **Standard Error Calculation:** The engine uses an **unpooled standard error** $(SE_{unpooled})$ that incorporates the variance adjustment factor ($\varphi$):
                $$SE = \sqrt{\frac{p_c(1-p_c)\varphi}{n_c} + \frac{p_v(1-p_v)\varphi}{n_v}}$$
            3.  **Non-Inferiority Z-Score:** We calculate the $Z$-stat by adding the **Non-Inferiority Margin ($\Delta$)** back into the observed difference:
                $$Z_{NI} = \frac{(CR_v - CR_c) + \Delta}{SE_{unpooled}}$$
            4.  **Lower Bound Estimation:** The engine calculates the lower bound of the difference. If this bound stays above $-\Delta$, the variant is considered "safe."
            5.  **Variance Adjustment:** If historical daily data is provided, the observed day-to-day variance of the conversion rate is compared to what pure binomial sampling would predict. The ratio $\varphi = \sigma^2_{observed} / \sigma^2_{binomial}$ is used to scale the standard error, shrinking it when the rate is historically stable ($\varphi < 1$), and inflating it conservatively when overdispersion is present ($\varphi > 1$).
            """)

        with st.expander("Frequentist: Interpretation"):
            st.markdown(r"""
            * **p-value (Z-test):** If $p < 0.05$, we reject the null hypothesis. There is less than a 5% chance the observed lift is due to random noise.
            * **Confidence Intervals (CI):** If the CI for the *relative lift* does not cross $0\%$, the result is statistically significant.
            * **P-value (Non-Inferiority):** Tests the null hypothesis that the Variant is worse than the Control by more than the margin. If $p \le \alpha$, we reject the idea that the variant is a "loser" and label it non-inferior.
            * **Lower Bound of Difference:** Represents the "worst-case scenario" for the variant's performance. If the bound is $-0.5\%$ and your limit is $-1.0\%$, the test passes.
            * **Success Status:** A green success message indicates that while the variant might not be a "winner," it is statistically unlikely to cause a regression beyond your defined tolerance.
            """)

    else:  # Bayesian
        with st.expander("Bayesian Analysis: What it provides"):
            st.markdown("""
            This engine provides a **probabilistic** view of the experiment, moving away from "significant/not significant" to "how much better is it?"
            
            * **Risk-Aware Decisioning:** Quantifies the **Expected Monetary Risk**: the literal monetary amount you stand to lose if the variant is actually worse.
            * **Business Impact:** Translates statistical lift into a **6-month revenue projection** based on your AOV as a constant.
            * **Direct Probabilities:** Gives you a clear "Chance to Beat Control" percentage.
            """)

        with st.expander("Bayesian: How it works"):
            st.markdown(r"""
            1.  **Prior & Posterior:** Unless specified by entering expectations, we start with a **Beta-Binomial conjugate prior** ($\alpha=1, \beta=1$). As data comes in, we update this to a posterior distribution.
            2. **Uplift Distribution:** Calculates the difference in daily conversions between variants across all simulations to build a full distribution of potential outcomes.
            3.  **Monte Carlo Simulation:** We run 20,000 simulations per variant to model the probability density.
            4.  **Revenue Modeling:** Projects the cumulative effect of the daily difference multiplied by the Average Order Value (AOV) over a **183-day horizon**.
            """)

        with st.expander("Bayesian: Interpretation"):
            st.markdown(r"""
            * **Chance to Beat Control:** The probability that the Variant is superior to the Control. A value $>95\%$ is a strong winner.
            * **Expected Total Contribution:** The net expected monetary gain over 6 months. This accounts for both the upside and the downside risk.
            * **Probability Density Graph:** Visualizes the uncertainty. Thinner, taller peaks indicate higher certainty in the conversion rate.
            """)

# -- Data input functions
def get_bayesian_inputs():
    st.session_state.num_variants = st.number_input(
        "How many variants did your experiment have (including control)?",
        min_value=2, max_value=10, step=1,
        value=st.session_state.num_variants,
        key="bayesian_num_variants"
    )

    num_variants = st.session_state.num_variants

    for key, default_value in [("visitor_counts", 0), ("conversion_counts", 0), ("aovs", 0.0)]:
        if len(st.session_state[key]) != num_variants:
            current_len = len(st.session_state[key])
            if num_variants > current_len:
                st.session_state[key].extend([default_value] * (num_variants - current_len))
            else:
                st.session_state[key] = st.session_state[key][:num_variants]

    alphabet = string.ascii_uppercase
    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Visitors")
    with col2:
        st.write("#### Conversions")

    for i in range(num_variants):
        with col1:
            st.session_state.visitor_counts[i] = st.number_input(
                f"Visitors for variant {alphabet[i]}",
                min_value=0, step=1,
                value=st.session_state.visitor_counts[i],
                key=f"b_visitors_{i}",
                label_visibility="visible"
            )
        with col2:
            st.session_state.conversion_counts[i] = st.number_input(
                f"Conversions for variant {alphabet[i]}",
                min_value=0, step=1,
                value=st.session_state.conversion_counts[i],
                key=f"b_conversions_{i}",
                label_visibility="visible"
            )

    st.write("---")
    st.write("#### Average Order Value (€)")

    for i in range(num_variants):
        st.session_state.aovs[i] = st.number_input(
            f"Average Order Value for Variant {alphabet[i]}",
            min_value=0.0, step=0.01,
            value=st.session_state.aovs[i],
            key=f"b_aov_{i}"
        )

    st.session_state.aov_cv = st.slider(
        "AOV Variability",
        min_value=0.1, max_value=2.0, step=0.1,
        value=st.session_state.get("aov_cv", 0.5),
        help="Coefficient of variation for order value (std / mean). "
             "Lower = stable pricing, higher = wide price range. "
             "Does not affect the average prediction, only the spread."
    )

    st.write("---")
    st.write("### General Test Settings")

    st.session_state.probability_winner = st.number_input(
        "Minimum probability for a winner?",
        min_value=0.0, max_value=100.0, step=0.01,
        value=st.session_state.probability_winner,
        help="Enter the success rate that determines if your test has a winner."
    )

    st.session_state.runtime_days = st.number_input(
        "How many days did your test run?",
        min_value=0, step=1,
        value=st.session_state.runtime_days
    )

    use_priors = st.checkbox(
        "Apply a lift prior?",
        help="Express skepticism about large lifts before seeing the data. Recommended for most experiments."
    )

    if use_priors:
        st.write("##### Prior Beliefs")
        col1, col2 = st.columns(2)
        with col1:
            expected_lift_pct = st.number_input(
                "Expected Lift (%)",
                min_value=-99.0, max_value=1000.0, step=0.1, value=0.0,
                help="Your best guess at the relative lift before seeing results. Use 0 if you have no directional expectation."
            )
        with col2:
            raw_skepticism = st.selectbox(
                "Skepticism",
                ["skeptical", "moderate", "uninformative"],
                index=0,
                help="How strongly you believe large lifts are implausible. 'Skeptical' resists extreme results; 'uninformative' applies almost no pressure."
            )

        beta_prior = get_beta_priors()
        valid_scepticism = cast(Literal["skeptical", "moderate", "uninformative"], raw_skepticism)
        lift_prior = get_lift_prior(expected_lift_pct=expected_lift_pct, skepticism=valid_scepticism)

        st.caption(
            f"A **{valid_scepticism}** prior is applied with an expected lift of **{expected_lift_pct:+.1f}%**. "
            f"At your typical sample sizes, this will only meaningfully affect results when the data is ambiguous."
        )
    else:
        beta_prior = get_beta_priors()
        lift_prior = get_lift_prior(expected_lift_pct=0.0, skepticism="uninformative")

    return (
        st.session_state.visitor_counts,
        st.session_state.conversion_counts,
        st.session_state.aovs,
        st.session_state.aov_cv,
        beta_prior,
        lift_prior,
        st.session_state.probability_winner,
        st.session_state.runtime_days
    )


def get_frequentist_inputs():
    st.session_state.num_variants = st.number_input(
        "How many variants did your experiment have (including control)?",
        min_value=2, max_value=10, step=1,
        value=st.session_state.num_variants,
        key="frequentist_num_variants"
    )

    st.write("---")
    num_variants = st.session_state.num_variants

    for key, default_value in [("visitor_counts", 0), ("conversion_counts", 0), ("aovs", 0.0)]:
        if len(st.session_state[key]) != num_variants:
            current_len = len(st.session_state[key])
            if num_variants > current_len:
                st.session_state[key].extend([default_value] * (num_variants - current_len))
            else:
                st.session_state[key] = st.session_state[key][:num_variants]

    alphabet = string.ascii_uppercase
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Visitors")
    with col2:
        st.write("### Conversions")

    for i in range(st.session_state.num_variants):
        with col1:
            st.session_state.visitor_counts[i] = st.number_input(
                f"Visitors for variant {alphabet[i]}",
                min_value=0, step=1,
                value=st.session_state.visitor_counts[i],
                key=f"f_visitors_{i}"
            )
        with col2:
            st.session_state.conversion_counts[i] = st.number_input(
                f"Conversions for variant {alphabet[i]}",
                min_value=0, step=1,
                value=st.session_state.conversion_counts[i],
                key=f"f_conversions_{i}"
            )

    st.write("---")
    st.session_state.confidence_level = st.number_input(
        "In %, how confident do you want to be in the results?",
        min_value=0, step=1,
        value=st.session_state.get("confidence_level", 95),
        help="Set the confidence level for which you want to test (enter 90, 95, etc)."
    )
    st.session_state.test_duration = st.number_input(
        "How many days has this test been running?",
        min_value=1,
        value=st.session_state.get("test_duration", 7),
        help="Enter the number of days the experiment has been running. Used to estimate potential time savings from variance adjustment."
    )

    st.write("---")
    st.write("### Variance Adjustment (Optional)")
    with st.expander("How variance adjustment works", expanded=False):
        st.markdown(r"""
            ### Aggregate Time-Series Variance Adjustment

            This tool uses a historical daily aggregate of your conversion data to estimate how
            stable or noisy your conversion rate truly is, and adjusts the experiment's standard
            errors accordingly.

            Rather than requiring individual user-level data (which demands returning visitors and
            a complex export), you only need a simple day-by-day summary: total visitors and
            total conversions per day from a comparable period before the experiment.

            **How it works**

            We compare the *observed* day-to-day variance of your historical conversion rate to
            what pure binomial sampling would predict for the same traffic volumes.

            The dispersion ratio is:
            $$\varphi = \frac{\sigma^2_{\text{observed}}}{\sigma^2_{\text{binomial}}}$$

            Both variances are weighted by daily visitor volume to avoid small-day distortion.

            - $\varphi < 1$ - your rate is **more stable** than binomial theory predicts.
              Standard errors are reduced: confidence intervals tighten and you reach
              significance faster.
            - $\varphi \approx 1$ - your rate behaves as pure binomial sampling expects.
              No meaningful adjustment is applied.
            - $\varphi > 1$ - **overdispersion detected** (e.g. campaign bursts, day-of-week
              spikes, seasonality). Standard errors are *inflated* conservatively to protect
              your false positive rate.

            The standard error is scaled as:
            $$SE_{\text{adjusted}} = \sqrt{\frac{p(1-p)\,\varphi}{n}}$$

            **What to upload**

            A daily aggregate CSV with at least 14 rows (one per day) and three columns:
            date, visitors, conversions. Any analytics tool (e.g. GA4, Piano, Matomo, your own BI)
            can produce this without a user-level join.
        """)

    reduction_factor = render_variance_reduction_ui()

    st.write("---")
    st.write("### False Positive / Negative Risk")
    with st.expander("What is this?"):
        st.markdown(r"""
        Standard p-values tell you P(data | H0), not P(H0 | data). These metrics flip that using your prior belief that the variant has a real effect.

        | | Significant result | Non-significant result |
        |---|---|---|
        | **Greater** | **FPR**: probability the apparent improvement is false | **FNDR**: probability a real improvement was missed |
        | **Less** | **FPR**: probability the apparent harm is false | **FNDR**: probability real harm was missed |
        | **Two-sided** | **FPR**: probability the detected difference (either direction) is false | **FNDR**: probability a real difference (either direction) was missed |

        The prior belief setting controls how much weight is given to the hypothesis that a real effect exists before seeing the data:
        - **Skeptical** (10%): most experiments don't move the needle
        - **Neutral** (50%): no strong expectation either way
        - **Optimistic** (90%): strong belief the variant has a real effect
        """)

    sensitivity_mode = st.selectbox(
        "Prior belief that the variant has a real effect",
        ["skeptical", "neutral", "optimistic", "custom"],
        index=1,
        help="Skeptical = 10% prior, Neutral = 50%, Optimistic = 90%."
    )
    custom_prior = None
    if sensitivity_mode == "custom":
        custom_prior = st.number_input(
            "Custom prior probability P(H1)",
            min_value=0.0, max_value=1.0, step=0.01, value=0.5
        )

    return (
        st.session_state.visitor_counts,
        st.session_state.conversion_counts,
        st.session_state.confidence_level,
        reduction_factor,
        st.session_state.test_duration,
        sensitivity_mode,
        custom_prior
    )


def validate_inputs(visitors, conversions, aovs=None) -> bool:
    visitors_list = visitors if isinstance(visitors, list) else [visitors]
    conversions_list = conversions if isinstance(conversions, list) else [conversions]

    aovs_list = []
    if aovs is not None:
        aovs_list = aovs if isinstance(aovs, list) else [aovs]

    for i in range(len(visitors_list)):
        v = visitors_list[i]
        c = conversions_list[i]
        variant_name = chr(65 + i)

        if not isinstance(v, int) or not isinstance(c, int):
            st.error(f"Error for Variant {variant_name}: Visitors and conversions must be whole numbers.")
            return False
        if v < 0 or c < 0:
            st.error(f"Error for Variant {variant_name}: Visitors and conversions cannot be negative.")
            return False
        if c > v:
            st.error(f"Error for Variant {variant_name}: The amount of conversions ({c}) cannot exceed the amount of visitors ({v}).")
            return False

        if aovs_list and i < len(aovs_list):
            a = aovs_list[i]
            if not isinstance(a, (int, float)) or a < 0:
                st.error(f"Error for Variant {variant_name}: AOV must be a non-negative number.")
                return False

    return True


# -- Bayesian helper functions --

@dataclass(frozen=True)
class BetaPrior:
    alpha: float
    beta: float

@dataclass(frozen=True)
class LiftPrior:
    mean_log_lift: float
    std_log_lift: float

_LIFT_PRIOR_STD: dict[str, float] = {
    "skeptical": 0.10,
    "moderate": 0.25,
    "uninformative": 1.00,
}

def calculate_probabilities(
    visitor_counts,
    conversion_counts,
    beta_prior: BetaPrior,
    lift_prior: LiftPrior,
    num_samples: int = 10000,
    seed: int = 42,
) -> Tuple[List[float], np.ndarray]:
    rng = np.random.default_rng(seed=seed)
    num_variants = len(visitor_counts)

    all_samples = []
    for i in range(num_variants):
        alpha_post = beta_prior.alpha + conversion_counts[i]
        beta_post = beta_prior.beta + (visitor_counts[i] - conversion_counts[i])
        samples = rng.beta(alpha_post, beta_post, size=num_samples)
        all_samples.append(samples)

    samples_matrix = np.array(all_samples)
    control_samples = samples_matrix[0]

    if num_variants > 1:
        per_challenger_weights = np.array([
            compute_lift_weights(control_samples, samples_matrix[i], lift_prior)
            for i in range(1, num_variants)
        ])
        weights = per_challenger_weights.mean(axis=0)
        weights /= weights.sum()
    else:
        weights = np.ones(num_samples) / num_samples

    best_variant_indices = np.argmax(samples_matrix, axis=0)
    probabilities_to_be_best = [
        np.average(best_variant_indices == i, weights=weights)
        for i in range(num_variants)
    ]

    return probabilities_to_be_best, samples_matrix


def get_beta_priors() -> BetaPrior:
    """
    Returns a fixed uninformative prior. At typical sample sizes the data
    dominates regardless; beliefs about lift are expressed via get_lift_prior.
    """
    return BetaPrior(alpha=1.0, beta=1.0)


def get_lift_prior(
    expected_lift_pct: float,
    skepticism: Literal["skeptical", "moderate", "uninformative"],
) -> LiftPrior:
    if skepticism not in _LIFT_PRIOR_STD:
        raise ValueError(
            f"skepticism must be one of {list(_LIFT_PRIOR_STD)}, got {skepticism!r}."
        )
    if expected_lift_pct <= -100:
        raise ValueError("expected_lift_pct must be greater than -100.")

    return LiftPrior(
        mean_log_lift=np.log1p(expected_lift_pct / 100.0),
        std_log_lift=_LIFT_PRIOR_STD[skepticism],
    )


def compute_lift_weights(
    cr_samples_control: np.ndarray,
    cr_samples_challenger: np.ndarray,
    prior: LiftPrior,
) -> np.ndarray:
    log_lift = np.log(cr_samples_challenger) - np.log(cr_samples_control)
    log_w = norm.logpdf(log_lift, prior.mean_log_lift, prior.std_log_lift)
    w = np.exp(log_w - log_w.max())
    return w / w.sum()


def simulate_uplift_distributions(
    visitor_counts,
    conversion_counts,
    beta_prior: BetaPrior,
    lift_prior: LiftPrior,
    num_samples: int = 20000,
    seed: int = 42,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed=seed)
    num_variants = len(visitor_counts)

    all_samples = []
    for i in range(num_variants):
        alpha_post = beta_prior.alpha + conversion_counts[i]
        beta_post = beta_prior.beta + (visitor_counts[i] - conversion_counts[i])
        all_samples.append(rng.beta(alpha_post, beta_post, size=num_samples))

    samples_matrix = np.array(all_samples)
    control_samples = samples_matrix[0]

    uplift_distributions = []
    for i in range(1, num_variants):
        challenger_samples = samples_matrix[i]
        weights = compute_lift_weights(control_samples, challenger_samples, lift_prior)
        uplift = (challenger_samples - control_samples) / (control_samples + 1e-9)
        resampled_indices = rng.choice(num_samples, size=num_samples, p=weights)
        uplift_distributions.append(uplift[resampled_indices])

    return uplift_distributions


def plot_uplift_histograms(uplift_distributions, observed_uplifts):
    num_challengers = len(uplift_distributions)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    fig, axes = plt.subplots(
        nrows=num_challengers,
        ncols=1,
        figsize=(14, 8 * num_challengers),
        squeeze=False
    )
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        diffs_percentage = uplift_distributions[i] * 100
        observed_uplift = float(observed_uplifts[i] * 100)

        challenger_label = alphabet[i + 1]
        control_label = alphabet[0]

        def calculate_optimal_bins(data):
            n = len(data)
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            if iqr == 0: return int(1 + np.log2(n))
            bin_width_fd = 2 * iqr * (n ** (-1/3))
            if bin_width_fd == 0: return int(1 + np.log2(n))
            return min(int(np.ceil((np.max(data) - np.min(data)) / bin_width_fd)), 200)

        num_bins = calculate_optimal_bins(diffs_percentage)
        n, bins, patches = ax.hist(diffs_percentage, bins=num_bins, edgecolor='black', alpha=0.6)

        for patch in patches:
            if patch.get_x() < 0:
                patch.set_facecolor('lightcoral')
            else:
                patch.set_facecolor('lightgreen')

        mean_diff = np.mean(diffs_percentage)
        std_diff = np.std(diffs_percentage)
        range_min, range_max = mean_diff - 3.5 * std_diff, mean_diff + 3.5 * std_diff
        ax.set_xlim(range_min, range_max)

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2f}%'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        line_label = f'Observed Uplift ({challenger_label} vs {control_label}): {observed_uplift:.2f}%'
        line_observed_uplift = ax.axvline(x=observed_uplift, color='red', linestyle='--', linewidth=2, label=line_label)

        patch_a = mpatches.Patch(color='lightcoral', label=f'{control_label} is better')
        patch_b = mpatches.Patch(color='lightgreen', label=f'{challenger_label} is better')

        ax.set_title(f'Distribution of Simulated Uplift: Variant {challenger_label} vs. Variant {control_label}')
        ax.set_xlabel('Percentage Uplift in Conversion Rate (%)')
        ax.set_ylabel('Frequency')
        ax.legend(handles=[line_observed_uplift, patch_a, patch_b])
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout(pad=3.0, rect=(0, 0, 1, 1))
    st.pyplot(fig)
    plt.close(fig)


def plot_winner_probabilities_chart(probabilities_to_be_best):
    num_variants = len(probabilities_to_be_best)
    alphabet = string.ascii_uppercase
    variant_labels = [f"Variant {alphabet[i]}" for i in range(num_variants)]
    cmap = colormaps['viridis']
    colors = cmap(np.linspace(0, 1, num_variants))

    fig_height = 2 + num_variants * 0.8
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.barh(variant_labels, probabilities_to_be_best, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Chance for Variants to be the Best')
    ax.set_title('Chance per Variant to generate the most Conversions')
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()

    def get_contrast_color(rgb):
        r, g, b = rgb[:3]
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return 'black' if luminance > 0.5 else 'white'

    for index, (value, color) in enumerate(zip(probabilities_to_be_best, colors)):
        text_color = get_contrast_color(color)
        if value > 0.9:
            ax.text(value - 0.02, index, f"{value:.2%}", ha='right', va='center',
                    color=text_color, fontweight='bold', fontsize=12)
        else:
            ax.text(value + 0.01, index, f"{value:.2%}",
                    ha='left', va='center', color='black', fontsize=11)

    st.pyplot(fig)
    plt.close(fig)


def sample_aov(mean_aov: float, cv: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw AOV samples from a log-normal distribution with the given mean
    and coefficient of variation (CV = std / mean).

    The parameterisation preserves E[AOV] = mean_aov exactly, so predictions
    are not inflated or deflated; only variance is added.

    Typical CV for e-commerce AOV: 0.5 (stable catalogue) to 1.5 (wide price range).
    """
    sigma2 = np.log1p(cv ** 2)
    mu = np.log(mean_aov) - sigma2 / 2
    return rng.lognormal(mean=mu, sigma=np.sqrt(sigma2), size=n_samples)


def perform_multi_variant_risk_assessment(
    visitor_counts,
    conversion_counts,
    aovs,
    probabilities_to_be_best,
    runtime_days,
    beta_prior: BetaPrior,
    lift_prior: LiftPrior,
    aov_cv: float = 0.5,
    projection_period: int = 183,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    num_variants = len(visitor_counts)
    if runtime_days == 0:
        return pd.DataFrame()

    n_simulations = 20000
    cr_samples = []
    all_daily_conversion_samples = []

    for i in range(num_variants):
        alpha_post = beta_prior.alpha + conversion_counts[i]
        beta_post = beta_prior.beta + (visitor_counts[i] - conversion_counts[i])
        samples_cr = rng.beta(alpha_post, beta_post, size=n_simulations)
        cr_samples.append(samples_cr)
        all_daily_conversion_samples.append((samples_cr * visitor_counts[i]) / runtime_days)

    control_samples = all_daily_conversion_samples[0]
    control_cr = cr_samples[0]

    results = []
    for i in range(1, num_variants):
        challenger_samples = all_daily_conversion_samples[i]
        challenger_cr = cr_samples[i]

        weights = compute_lift_weights(control_cr, challenger_cr, lift_prior)

        sampled_control_aov = sample_aov(aovs[0], aov_cv, n_simulations, rng)
        sampled_challenger_aov = sample_aov(aovs[i], aov_cv, n_simulations, rng)

        difference_samples = challenger_samples - control_samples
        positive_mask = difference_samples > 0
        negative_mask = difference_samples < 0

        prob_challenger_is_better = np.average(positive_mask, weights=weights)
        prob_control_is_better = np.average(negative_mask, weights=weights)

        expected_daily_gain = (
            np.average(difference_samples * sampled_challenger_aov, weights=weights * positive_mask)
            if positive_mask.any() else 0
        )
        expected_daily_loss = (
            np.average(difference_samples * sampled_control_aov, weights=weights * negative_mask)
            if negative_mask.any() else 0
        )

        expected_monetary_uplift = expected_daily_gain * projection_period * prob_challenger_is_better
        expected_monetary_risk = expected_daily_loss * projection_period * prob_control_is_better
        total_contribution = expected_monetary_uplift + expected_monetary_risk

        results.append({
            "Variant": string.ascii_uppercase[i],
            "Chance to Beat Control": round(prob_challenger_is_better * 100, 2),
            "Chance to be Best Overall": round(probabilities_to_be_best[i] * 100, 2),
            "Expected Monetary Uplift": round(expected_monetary_uplift, 2),
            "Expected Monetary Risk": round(expected_monetary_risk, 2),
            "Expected Total Contribution": round(total_contribution, 2),
        })

    if not results:
        return pd.DataFrame(columns=[
            "Variant", "Chance to Beat Control", "Chance to be Best Overall",
            "Expected Monetary Uplift", "Expected Monetary Risk", "Expected Total Contribution"
        ])
    return pd.DataFrame(results)


def display_results_per_variant(
    probabilities_to_be_best,
    observed_uplifts,
    probability_winner,
    aovs,
    runtime_days,
    df=None
):
    num_variants = len(probabilities_to_be_best)
    alphabet = string.ascii_uppercase

    st.write("### Results Summary")
    st.write("")

    for i in range(1, num_variants):
        challenger_index = i
        control_index = 0

        challenger_label = alphabet[challenger_index]
        probability_challenger_better = probabilities_to_be_best[challenger_index]
        probability_control_better = probabilities_to_be_best[control_index]
        observed_uplift_challenger = observed_uplifts[i - 1] * 100

        if round(probability_challenger_better * 100, 2) >= probability_winner:
            bayesian_result = "a <span style='color: green; font-weight: bold;'>winner</span>"
        elif round(probability_control_better * 100, 2) >= probability_winner:
            bayesian_result = "a <span style='color: red; font-weight: bold;'>loss averted</span>"
        else:
            bayesian_result = "<span style='color: black; font-weight: bold;'>inconclusive</span>. There is no real effect to be found, or you need to collect more data"

        st.write(f"#### Variant {challenger_label} vs Control (A)")
        st.markdown(
            f"Variant {challenger_label} has a {round(probability_challenger_better * 100, 2)}% chance to win with a relative change of {round(observed_uplift_challenger, 2)}%. "
            f"Because your winning threshold was set to {int(probability_winner)}%, this experiment is {bayesian_result}.",
            unsafe_allow_html=True
        )

        if num_variants > 2:
            st.write("---")

    if all(aov > 0 for aov in aovs) and runtime_days > 0:
        if df is not None:
            st.write("#### Business Risk Assessment")
            st.write("""
                The table below shows the potential contribution to revenue over 6 months.
                AOV is modelled as a log-normal variable around your input values, adding
                realistic spread without inflating or deflating the point estimates.
                On smaller datasets of < 1000 conversions, interpret with care.
            """)
            with st.expander("How is the business case calculated?"):
                st.markdown(r"""
                    The business risk assessment translates the Bayesian simulation into a 6-month monetary projection.
                    It runs 20,000 simulations per variant and applies the following logic:

                    **Conversion rate sampling**
                    For each simulation, a conversion rate is drawn from the posterior Beta distribution, updated with
                    the observed visitors and conversions. The daily conversion volume is then estimated as:
                    $$\text{Daily Conversions} = CR_{sampled} \times \frac{\text{Visitors}}{\text{Runtime (days)}}$$

                    **AOV sampling**
                    Rather than treating AOV as a fixed constant, it is drawn from a log-normal distribution on each
                    simulation. The distribution is parameterised so that the expected value equals your input AOV exactly -
                    only variance is added, not bias. The degree of spread is controlled by the AOV Variability slider
                    (coefficient of variation: std / mean).

                    **Lift prior**
                    If a lift prior is applied, importance weights are computed for each simulation based on how
                    plausible the observed lift is under your prior beliefs. These weights adjust all monetary
                    figures without discarding any simulations.

                    **Uplift and risk**
                    The daily difference in conversions between the challenger and control is calculated per simulation.
                    Positive and negative differences are separated:

                    | | Formula |
                    |---|---|
                    | Expected Daily Gain | Mean of positive differences × sampled challenger AOV |
                    | Expected Daily Loss | Mean of negative differences × sampled control AOV |
                    | Monetary Uplift | Daily Gain × 183 days × P(challenger better) |
                    | Monetary Risk | Daily Loss × 183 days × P(control better) |
                    | Total Contribution | Monetary Uplift + Monetary Risk |

                    **Interpretation**
                    - A positive Total Contribution means the expected gain outweighs the expected loss over 6 months.
                    - Monetary Risk is always negative or zero. It represents the downside if the variant underperforms.
                    - On datasets with fewer than 1,000 conversions, the simulation may surface more extreme values.
                    Treat projections on small samples as directional, not precise.

                    _This table is a measurement of potential impact only; not a guarantee of future revenue._
                """)
            st.dataframe(df)
    else:
        st.write("")
        st.warning("Business case data is missing or incomplete. Skipping monetary calculations.")


# -- Frequentist helper functions --

# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE TIME-SERIES VARIANCE ADJUSTMENT
#
# Core idea:
#   The standard frequentist test assumes binomial variance p*(1-p)/n.
#   We compare the *observed* day-to-day variance of the historical conversion
#   rate against what pure binomial sampling would predict.
#
#   Dispersion φ = observed_variance / expected_binomial_variance
#     φ < 1  ->  the rate is MORE stable than binomial theory assumes
#               -> SE is defensibly shrunk -> reduction_factor = φ
#     φ ≈ 1  ->  binomial assumption is accurate -> no adjustment
#     φ > 1  ->  the rate is NOISIER than binomial -> SE is inflated (conservative)
#               -> reduction_factor = φ, shown as a WARNING in the UI
# ─────────────────────────────────────────────────────────────────────────────

def get_timeseries_template() -> str:
    """
    Returns a minimal CSV template: one row per day with total visitors
    and total conversions. Easily exported from any analytics tool without
    requiring a user-level join across periods.
    """
    template_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=14).strftime("%Y-%m-%d"),
        "visitors": [1200, 980, 1100, 1350, 900, 800, 1050,
                    1180, 1020, 1090, 1300, 950, 820, 1070],
        "conversions": [48, 39, 44, 54, 36, 32, 42,
                        47, 41, 43, 52, 38, 33, 43],
    })
    buffer = io.StringIO()
    template_df.to_csv(buffer, index=False)
    return buffer.getvalue()


def validate_timeseries_data(
    df: pd.DataFrame,
    visitors_col: str,
    conversions_col: str,
) -> Tuple[bool, str, int]:
    """
    Validates uploaded aggregate time-series data before computing the
    dispersion ratio.

    Returns:
        is_valid bool - whether the data passes all checks
        message str - human-readable status or error description
        n_periods int - number of rows (days) in the upload
    """
    n_periods = len(df)

    if n_periods < 14:
        return (
            False,
            f"Too few periods ({n_periods} rows). At least 14 days are needed "
            "for a reliable dispersion estimate.",
            n_periods,
        )

    if (df[visitors_col] <= 0).any():
        return False, "Some rows have zero or negative visitor counts.", n_periods

    if (df[conversions_col] < 0).any():
        return False, "Conversion column contains negative values.", n_periods

    if (df[conversions_col] > df[visitors_col]).any():
        return (
            False,
            "Some rows show more conversions than visitors - check your column mapping.",
            n_periods,
        )

    rates = df[conversions_col] / df[visitors_col]
    if rates.std() == 0:
        return False, "Conversion rate is identical across all periods (zero variance).", n_periods

    return True, "Data quality looks good.", n_periods


def calculate_timeseries_reduction_factor(
    df: pd.DataFrame,
    visitors_col: str,
    conversions_col: str,
) -> Tuple[float, float, str]:
    """
    Estimates a variance scaling factor (φ) from aggregate historical
    time-series data.

    Method:
      1. Compute the visitor-weighted observed variance of daily conversion rates.
      2. Compute the expected binomial variance for the same days (p*(1-p)/n),
         again visitor-weighted to prevent small-traffic days from dominating.
      3. φ = observed / expected.
         φ < 1 -> stable process, SE is shrunk.
         φ > 1 -> overdispersed process, SE is inflated (conservative).
         φ = 1 -> no adjustment.

    Returns:
        reduction_factor float - multiply variance by this; <1 shrinks, >1 inflates
        phi float - raw dispersion ratio (shown to the user)
        regime str - 'stable' | 'neutral' | 'noisy' | 'high_noise'
    """
    visitors = df[visitors_col].astype(float)
    conversions = df[conversions_col].astype(float)
    rates = conversions / visitors

    weights = visitors / visitors.sum()
    weighted_mean = (rates * weights).sum()

    observed_var = (weights * (rates - weighted_mean) ** 2).sum()
    expected_binomial_var = (weights * rates * (1 - rates) / visitors).sum()

    if expected_binomial_var == 0:
        return 1.0, 1.0, "neutral"

    phi = observed_var / expected_binomial_var
    reduction_factor = float(np.clip(phi, 0.10, None))

    if phi < 0.80:
        regime = "stable"
    elif phi < 1.05:
        regime = "neutral"
    elif phi < 1.50:
        regime = "noisy"
    else:
        regime = "high_noise"

    return reduction_factor, phi, regime


def render_variance_reduction_ui() -> float:
    """
    Renders the variance adjustment section and returns a reduction_factor
    to be passed into calculate_frequentist_statistics().

    reduction_factor < 1.0 -> SE shrunk (process more stable than binomial)
    reduction_factor = 1.0 -> no adjustment
    reduction_factor > 1.0 -> SE inflated (overdispersion detected)
    """
    template_csv = get_timeseries_template()
    st.download_button(
        label="Download CSV Template",
        data=template_csv,
        file_name="timeseries_template.csv",
        mime="text/csv",
        help=(
            "One row per day: date, visitors, conversions. "
            "Export this from GA4, Piano, or any BI tool; "
            "no user-level join required."
        ),
    )

    use_vr = st.checkbox("Apply Variance Adjustment via Historical Benchmark")
    reduction_factor = 1.0

    if not use_vr:
        return reduction_factor

    st.info(
        "**How this works:** Upload a daily aggregate export from a comparable "
        "historical period (e.g. the 4-8 weeks before your experiment). "
        "We compare the observed day-to-day volatility of your conversion rate "
        "to what pure binomial sampling would predict.\n\n"
        "- If your rate is **more stable** than theory expects -> SE is reduced, "
        "reaching significance faster.\n"
        "- If your rate is **noisier** (e.g. campaign bursts, seasonality) -> "
        "SE is conservatively inflated, protecting against false positives."
    )

    uploaded_file = st.file_uploader(
        "Upload Historical Daily Aggregate CSV",
        type="csv",
        help="Columns required: a date column, a visitors column, a conversions column.",
    )

    if not uploaded_file:
        return reduction_factor

    df_hist = pd.read_csv(uploaded_file)
    cols = df_hist.columns.tolist()

    st.write("### Map your columns")
    c1, c2 = st.columns(2)
    with c1:
        default_visitors = next(
            (i for i, c in enumerate(cols) if "visit" in c.lower() or "session" in c.lower()), 0
        )
        visitors_col = st.selectbox("Visitors column", cols, index=default_visitors, key="ts_visitors")
    with c2:
        default_conv = next(
            (i for i, c in enumerate(cols) if "conv" in c.lower() or "purchas" in c.lower()), 0
        )
        conversions_col = st.selectbox("Conversions column", cols, index=default_conv, key="ts_conversions")

    is_valid, message, n_periods = validate_timeseries_data(df_hist, visitors_col, conversions_col)

    if not is_valid:
        st.warning(message)
        return reduction_factor

    reduction_factor, phi, regime = calculate_timeseries_reduction_factor(
        df_hist, visitors_col, conversions_col
    )

    with st.expander("Data Quality & Variance Adjustment Details", expanded=False):
        regime_labels = {
            "stable": ("**Stable**", "green", "Your conversion rate is more consistent than binomial theory predicts. SE will be reduced."),
            "neutral": ("**Neutral**", "blue", "Your conversion rate behaves close to the binomial assumption. Little adjustment applied."),
            "noisy": ("**Noisy**", "orange", "Extra day-to-day volatility detected (campaigns? seasonality?). SE is conservatively inflated."),
            "high_noise": ("**High Noise**","red", "Strong overdispersion. SE inflated substantially. Consider a longer or calmer baseline period."),
        }
        label, color, advice = regime_labels[regime]

        st.markdown(f"""
        | Metric | Value |
        | :--- | :--- |
        | **Historical periods** | {n_periods} days |
        | **Dispersion φ** | `{phi:.3f}` |
        | **Process regime** | {label} |
        | **Advice** | {advice} |
        """)

        st.caption(
            "φ = observed day-to-day variance ÷ expected binomial variance. "
            "φ < 1 enables SE reduction; φ > 1 triggers conservative inflation."
        )

    if regime in ("stable", "neutral"):
        pct_change = (1 - reduction_factor) * 100
        st.success(f"**φ = {phi:.2f}**: process is stable.")
        st.metric(
            "Variance Adjustment",
            f"{pct_change:.1f}% reduction",
            delta=f"-{pct_change:.1f}% noise",
            delta_color="normal",
        )
    else:
        pct_inflation = (reduction_factor - 1) * 100
        st.warning(
            f"**φ = {phi:.2f}**: overdispersion detected. "
            f"SE inflated by {pct_inflation:.1f}% to protect against false positives."
        )
        st.metric(
            "Variance Adjustment",
            f"+{pct_inflation:.1f}% (conservative inflation)",
            delta=f"+{pct_inflation:.1f}% caution added",
            delta_color="inverse",
        )

    return reduction_factor


def plot_variance_adjusted_comparison(results, visitor_counts) -> go.Figure:
    """
    Plots the probability density curves for each variant after variance
    adjustment. The title reflects the direction of adjustment (reduction
    or inflation) based on the dispersion ratio φ.
    """
    fig = go.Figure()

    all_means = np.array(results['conversion_rates'])
    all_ses = np.array(results['standard_errors'])
    reach = 4 * np.maximum(all_ses, 1e-9)
    plot_min = np.min(all_means - reach)
    plot_max = np.max(all_means + reach)
    x_min = max(0.0, plot_min)
    x_max = min(1.0, plot_max)
    x = np.linspace(x_min, x_max, 500)

    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for i in range(results['num_variants']):
        cr = results['conversion_rates'][i]
        se = results['standard_errors'][i]
        y = norm.pdf(x, cr, se)
        label = f"Variant {alphabet[i]} (Control)" if i == 0 else f"Variant {alphabet[i]}"

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name=label,
            line=dict(color=colors[i % len(colors)], width=3),
            fill='tozeroy'
        ))

    rf = results['reduction_factor']
    if rf < 1.0:
        adjustment_label = f"SE Reduced by {(1 - rf) * 100:.1f}% (φ = {rf:.2f})"
    elif rf > 1.0:
        adjustment_label = f"SE Inflated by {(rf - 1) * 100:.1f}% (φ = {rf:.2f})"
    else:
        adjustment_label = "No Variance Adjustment Applied"

    fig.update_layout(
        title=f"Probability Density - Variance Adjustment: {adjustment_label}",
        xaxis_title="Conversion Rate",
        yaxis_title="Probability Density",
        template="plotly_white",
        hovermode="x unified"
    )

    return fig


def calculate_time_savings(reduction_factor, days_running) -> float:
    """
    Estimates how many additional days would have been needed to achieve
    the same precision without variance adjustment.

    Only meaningful when reduction_factor < 1.0 (i.e. SE was shrunk).
    Returns 0 for neutral or inflated cases.
    """
    if reduction_factor >= 1.0 or days_running <= 0:
        return 0
    total_days_required_without_adjustment = days_running / reduction_factor
    days_saved = total_days_required_without_adjustment - days_running
    return round(days_saved, 1)


def display_ci_chart(results, current_variant_idx, alphabet) -> alt.LayerChart:
    indices = [0, current_variant_idx]
    names = [f"({alphabet[0]}) Control", f"({alphabet[current_variant_idx]}) Challenger"]

    data = pd.DataFrame({
        'Variant': names,
        'CR': [results['conversion_rates'][i] * 100 for i in indices],
        'Lower': [results['confidence_intervals'][i][0] * 100 for i in indices],
        'Upper': [results['confidence_intervals'][i][1] * 100 for i in indices]
    })

    points = alt.Chart(data).mark_point(
        filled=True, size=100, color='#009900'
    ).encode(
        x=alt.X('CR:Q', scale=alt.Scale(zero=False), title='Conversion Rate (%)'),
        y=alt.Y('Variant:N', title=None),
        tooltip=['Variant', alt.Tooltip('CR:Q', format='.2f')]
    )

    error_bars = alt.Chart(data).mark_errorbar(thickness=3, color='#b7e1cd').encode(
        x='Lower:Q',
        x2='Upper:Q',
        y='Variant:N'
    )

    chart = (error_bars + points).properties(width='container', height=150)
    return chart


def calculate_frequentist_statistics(
    visitor_counts,
    conversion_counts,
    confidence_level,
    tail,
    reduction_factor=1.0
) -> Dict[str, Any]:
    if sum(visitor_counts) == 0 or any(v < 0 for v in visitor_counts):
        raise ValueError("Visitor counts must be positive and sum to a non-zero value.")

    num_variants = len(visitor_counts)
    alpha = 1 - (confidence_level / 100)

    sidak_alpha = 1 - (1 - alpha)**(1 / (num_variants - 1)) if num_variants > 2 else alpha

    conversion_rates = [c / v if v > 0 else 0 for c, v in zip(conversion_counts, visitor_counts)]
    standard_errors = [
        np.sqrt(cr * (1 - cr) * reduction_factor / v) if v > 0 else 0
        for cr, v in zip(conversion_rates, visitor_counts)
    ]

    z_critical = norm.ppf(1 - (alpha / 2))
    margins_of_error = [z_critical * se for se in standard_errors]
    confidence_intervals = [
        (cr - moe, cr + moe)
        for cr, moe in zip(conversion_rates, margins_of_error)
    ]

    lower_boundaries = [interval[0] for interval in confidence_intervals]
    upper_boundaries = [interval[1] for interval in confidence_intervals]
    lowest_interval = min(lower_boundaries)
    highest_interval = max(upper_boundaries)

    confidence_intervals_diff = []
    for i in range(1, num_variants):
        diff_cr = conversion_rates[i] - conversion_rates[0]
        se_diff = np.sqrt(standard_errors[i]**2 + standard_errors[0]**2)
        moe_diff = z_critical * se_diff
        ci_diff = (diff_cr - moe_diff, diff_cr + moe_diff)
        confidence_intervals_diff.append(ci_diff)

    observed = np.array(visitor_counts)
    expected = np.array([sum(observed) / num_variants] * num_variants)
    _, srm_p_value = chisquare(f_obs=observed, f_exp=expected)

    pooled_proportion = sum(conversion_counts) / sum(visitor_counts)

    z_stats = [
        (conversion_rates[i] - conversion_rates[0]) / np.sqrt(standard_errors[i]**2 + standard_errors[0]**2)
        if (standard_errors[i]**2 + standard_errors[0]**2) > 0 else 0
        for i in range(1, num_variants)
    ]

    if tail == 'Greater':
        p_values = [1 - norm.cdf(z) for z in z_stats]
    elif tail == 'Less':
        p_values = [norm.cdf(z) for z in z_stats]
    else:
        p_values = [2 * (1 - norm.cdf(abs(z))) for z in z_stats]

    significant_results = [p <= sidak_alpha for p in p_values]

    power_method_used = ""
    observed_powers = []

    if all(v > 1000 for v in visitor_counts):
        power_method_used = "Analytical"
        for i in range(1, num_variants):
            se_diff = np.sqrt(standard_errors[i]**2 + standard_errors[0]**2)
            if se_diff == 0:
                observed_powers.append(1.0)
                continue
            z_delta = abs(conversion_rates[i] - conversion_rates[0]) / se_diff
            if tail in ['Greater', 'Less']:
                z_alpha = norm.ppf(1 - sidak_alpha)
                power = norm.cdf(z_delta - z_alpha)
            else:
                z_alpha = norm.ppf(1 - sidak_alpha / 2)
                power = norm.cdf(z_delta - z_alpha) + norm.cdf(-z_delta - z_alpha)
            observed_powers.append(power)

    else:
        if reduction_factor != 1.0:
            power_method_used = "Analytical"
            st.warning(
                "Variance adjustment is active. Bootstrap power estimation is not compatible "
                "with a scaled standard error - falling back to the analytical method."
            )
            for i in range(1, num_variants):
                se_diff = np.sqrt(standard_errors[i]**2 + standard_errors[0]**2)
                if se_diff == 0:
                    observed_powers.append(1.0)
                    continue
                z_delta = abs(conversion_rates[i] - conversion_rates[0]) / se_diff
                if tail in ['Greater', 'Less']:
                    z_alpha = norm.ppf(1 - sidak_alpha)
                    power = norm.cdf(z_delta - z_alpha)
                else:
                    z_alpha = norm.ppf(1 - sidak_alpha / 2)
                    power = norm.cdf(z_delta - z_alpha) + norm.cdf(-z_delta - z_alpha)
                observed_powers.append(power)

        else:
            power_method_used = "Bootstrap"

            def bootstrap_sample(data_control, data_variant, alpha, tail):
                sample_control = np.random.choice(data_control, size=len(data_control), replace=True)
                sample_variant = np.random.choice(data_variant, size=len(data_variant), replace=True)
                pooled_p = (np.sum(sample_control) + np.sum(sample_variant)) / (len(sample_control) + len(sample_variant))
                se = np.sqrt(pooled_p * (1 - pooled_p) * (1 / len(sample_control) + 1 / len(sample_variant)))
                z_stat = 0 if se == 0 else (np.mean(sample_variant) - np.mean(sample_control)) / se
                if tail == 'Greater':
                    return (1 - norm.cdf(z_stat)) < alpha
                elif tail == 'Less':
                    return norm.cdf(z_stat) < alpha
                else:
                    return (2 * (1 - norm.cdf(abs(z_stat)))) < alpha

            def bootstrap_power(data_control, data_variant, alpha, tail, n_bootstraps=10000):
                significant_count = 0
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(bootstrap_sample, data_control, data_variant, alpha, tail)
                        for _ in range(n_bootstraps)
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        if future.result():
                            significant_count += 1
                return significant_count / n_bootstraps

            data_controls = [
                np.concatenate([np.ones(c), np.zeros(v - c)])
                for c, v in zip(conversion_counts, visitor_counts)
            ]
            observed_powers = [
                bootstrap_power(data_controls[0], data_controls[i], alpha, tail)
                for i in range(1, num_variants)
            ]

    results = {
        "num_variants": num_variants,
        "tail": tail,
        "confidence_intervals": confidence_intervals,
        "confidence_intervals_diff": confidence_intervals_diff,
        "conversion_rates": conversion_rates,
        "lowest boundary": lowest_interval,
        "highest boundary": highest_interval,
        "standard_errors": standard_errors,
        "z_stats": z_stats,
        "p_values": p_values,
        "is_significant": significant_results,
        "observed_powers": observed_powers,
        "power_method": power_method_used,
        "srm_p_value": srm_p_value,
        "sidak_alpha": sidak_alpha,
        "alpha": alpha,
        "confidence_level": confidence_level,
        "reduction_factor": reduction_factor
    }

    return results


def plot_conversion_distributions(results, reduction_factor=1.0):
    if not results:
        st.warning("Cannot generate visualization because calculation results are missing.")
        return

    conversion_rates = results['conversion_rates']
    se_list = results['standard_errors']
    num_variants = results['num_variants']
    significant_results = results['is_significant']
    sidak_alpha = results['sidak_alpha']

    st.write("")
    st.write("### Probability Density of Estimated Conversion Rates")

    fig, ax = plt.subplots(figsize=(12, 7))

    all_means = np.array(conversion_rates)
    all_ses = np.array(se_list)
    plot_min = np.min(all_means - 4 * np.maximum(all_ses, 1e-9))
    plot_max = np.max(all_means + 4 * np.maximum(all_ses, 1e-9))
    x_min = max(0, plot_min)
    x_max = min(1, plot_max)
    if x_max <= x_min: x_max = x_min + 1e-6
    x_range = np.linspace(x_min, x_max, 1000)

    colors = ['#808080', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    shade_colors = {'better': '#90EE90', 'worse': '#F08080'}
    base_alpha = 0.9
    shade_alpha = 0.3

    pdfs = []
    for i in range(num_variants):
        se = max(se_list[i], 1e-9)
        pdf = norm.pdf(x_range, conversion_rates[i], se)
        pdfs.append(pdf)

        variant_label = f'Variant {string.ascii_uppercase[i]}' if i > 0 else 'Control (A)'
        line_color = colors[i % len(colors)]

        ax.plot(x_range * 100, pdf, label=variant_label, color=line_color, alpha=base_alpha, linewidth=1.5)
        ax.axvline(conversion_rates[i] * 100, color=line_color, linestyle='--', alpha=base_alpha * 0.8)

        text_left_margin = 0.005
        ax.text(conversion_rates[i] * 100 + text_left_margin,
                ax.get_ylim()[1] * 0.03,
                f' {string.ascii_uppercase[i]}: {conversion_rates[i]*100:.2f}%',
                color=line_color, ha='left', rotation=90, va='bottom', fontsize=9)

    control_cr = conversion_rates[0]
    control_se = max(se_list[0], 1e-9)

    for i in range(1, num_variants):
        if significant_results[i - 1]:
            variant_cr = conversion_rates[i]
            variant_se = max(se_list[i], 1e-9)
            pdf_variant = pdfs[i]
            is_better = variant_cr > control_cr
            shade_color = shade_colors['better'] if is_better else shade_colors['worse']
            variant_label_char = string.ascii_uppercase[i]
            control_label_char = string.ascii_uppercase[0]

            mean_diff = variant_cr - control_cr
            se_diff = math.sqrt(variant_se**2 + control_se**2)
            prob_variant_better = 0.5
            if se_diff > 1e-9:
                z_score = mean_diff / se_diff
                prob_variant_better = norm.cdf(z_score)
            prob_control_better = 1 - prob_variant_better

            if is_better:
                lower_bound = norm.ppf(sidak_alpha, loc=variant_cr, scale=variant_se)
                fill_condition = (x_range >= lower_bound)
                bound_line_value = lower_bound * 100
            else:
                upper_bound = norm.ppf(1 - sidak_alpha, loc=variant_cr, scale=variant_se)
                fill_condition = (x_range <= upper_bound)
                bound_line_value = upper_bound * 100

            if prob_variant_better > prob_control_better:
                label_text = f'{variant_label_char} vs {control_label_char} (Significant)'
            elif prob_control_better > prob_variant_better:
                label_text = f'{control_label_char} vs {variant_label_char} (Significant)'
            else:
                label_text = ''

            ax.fill_between(x_range * 100, pdf_variant, 0, where=fill_condition.tolist(),
                            color=shade_color, alpha=shade_alpha, label=label_text)

            prob_text_display = f"P({variant_label_char}>{control_label_char}): {prob_variant_better*100:.1f}%"
            ax.axvline(float(bound_line_value), color='grey', linestyle=':', linewidth=1, alpha=0.7)

            mid_point_cr = (control_cr + variant_cr) / 2.0
            current_ylim = ax.get_ylim()
            y_pos_text = current_ylim[1] * 0.85
            ax.text(mid_point_cr * 100, y_pos_text, prob_text_display,
                    color='black', ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='none'))
            ax.set_ylim(current_ylim)

    ax.set_xlabel('Conversion rate (%)')
    ax.set_ylabel('Probability density')

    # Build a dynamic title that incorporates the variance adjustment so a
    # separate Plotly chart is no longer needed to communicate phi.
    if reduction_factor < 1.0:
        pct = (1 - reduction_factor) * 100
        adjustment_note = f" | SE reduced {pct:.1f}% (φ = {reduction_factor:.2f})"
    elif reduction_factor > 1.0:
        pct = (reduction_factor - 1) * 100
        adjustment_note = f" | SE inflated {pct:.1f}% (φ = {reduction_factor:.2f})"
    else:
        adjustment_note = ""

    ax.set_title(f'Comparison of Estimated Conversion Rate Distributions{adjustment_note}')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.set_ylim(bottom=0)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 0.85, 1))
    st.pyplot(fig)
    plt.close(fig)


def resolve_prior(sensitivity_mode='neutral', custom_prior=None):
    priors = {'skeptical': 0.1, 'neutral': 0.5, 'optimistic': 0.9}
    if sensitivity_mode == 'custom':
        if custom_prior is None:
            raise ValueError("custom_prior must be provided when sensitivity_mode is 'custom'.")
        if not (0.0 <= custom_prior <= 1.0):
            raise ValueError(f"custom_prior must be between 0 and 1, got {custom_prior}.")
        return custom_prior
    if sensitivity_mode not in priors:
        raise ValueError(f"Unknown sensitivity_mode '{sensitivity_mode}'.")
    return priors[sensitivity_mode]


def calculate_fpr(alpha, power, prior):
    p1, p0 = prior, 1 - prior
    numerator = alpha * p0
    denominator = (alpha * p0) + (power * p1)
    return numerator / denominator if denominator else 0.0


def calculate_fndr(alpha, power, prior):
    p1, p0 = prior, 1 - prior
    beta = 1 - power
    specificity = 1 - alpha
    numerator = beta * p1
    denominator = (beta * p1) + (specificity * p0)
    return numerator / denominator if denominator else 0.0


def display_frequentist_summary(
    results,
    visitor_counts,
    conversion_counts,
    non_inferiority_margin=0.01,
    confidence_noninf=95,
    reduction_factor=1.0,
    prior=0.5
):
    if not results:
        st.error("Calculation results are missing, cannot display summary.")
        return

    num_variants = results['num_variants']
    srm_p_value = results['srm_p_value']
    is_significant = results['is_significant']
    p_values = results['p_values']
    observed_powers = results['observed_powers']
    conversion_rates = results['conversion_rates']
    sidak_alpha = results['sidak_alpha']
    alpha_unadjusted = results['alpha']
    tail = results['tail']
    alphabet = string.ascii_uppercase

    st.write("### SRM Check")
    if srm_p_value > 0.01:
        st.write("This test is <span style='color: #009900; font-weight: 600;'>valid</span>. The distribution is as expected.", unsafe_allow_html=True)
    else:
        st.write("This test is <span style='color: #FF6600; font-weight: 600;'>invalid</span>: The distribution of traffic shows a statistically significant deviation...", unsafe_allow_html=True)

    if num_variants >= 3:
        st.write("### Šidák Correction applied")
        st.info(f"The Šidák correction was applied due to 3 or more variants in the test. The alpha threshold has been set to **{results['sidak_alpha']:.4f}** instead of {alpha_unadjusted:.4f}.")

    st.write("## Results summary")
    st.write("---")

    if reduction_factor < 1.0:
        pct_reduced = (1 - reduction_factor) * 100
        st.info(
            f"**Variance Adjustment Active** - SE reduced by {pct_reduced:.1f}% "
            f"(φ = {reduction_factor:.3f}). Your historical conversion rate is more "
            f"stable than the binomial assumption; confidence intervals are tightened accordingly."
        )
    elif reduction_factor > 1.0:
        pct_inflated = (reduction_factor - 1) * 100
        st.info(
            f"**Variance Adjustment Active** - SE inflated by {pct_inflated:.1f}% "
            f"(φ = {reduction_factor:.3f}). Overdispersion was detected in your historical data; "
            f"confidence intervals are widened to maintain a honest false positive rate."
        )

    for i in range(1, num_variants):
        challenger_index_in_lists = i - 1

        st.write(f"### Test results for {alphabet[i]} vs {alphabet[0]}")

        control_ci = results['confidence_intervals'][0]
        challenger_ci = results['confidence_intervals'][i]
        ci_difference = results['confidence_intervals_diff'][challenger_index_in_lists]
        observed_diff = conversion_rates[i] - conversion_rates[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label=f"Conversion Rate Control ({alphabet[0]})",
                value=f"{results['conversion_rates'][0]*100:.2f}%",
                help=f"The {results['confidence_level']}% confidence interval is [{control_ci[0]*100:.2f}% - {control_ci[1]*100:.2f}%]"
            )
        with col2:
            st.metric(
                label=f"Conversion Rate Challenger ({alphabet[i]})",
                value=f"{results['conversion_rates'][i]*100:.2f}%",
                help=f"The {results['confidence_level']}% confidence interval is [{challenger_ci[0]*100:.2f}% - {challenger_ci[1]*100:.2f}%]"
            )
        with col3:
            st.metric(
                label=f"Uplift CI ({alphabet[i]} vs {alphabet[0]})",
                value=f"{observed_diff*100:+.2f}%",
                help=f"The {results['confidence_level']}% confidence interval for the uplift is from {ci_difference[0]*100:+.2f}% to {ci_difference[1]*100:+.2f}%."
            )

        st.write("#### Confidence Interval Comparison")
        fig = display_ci_chart(results, i, alphabet)
        st.altair_chart(fig, width='stretch')
        st.write("")

        if is_significant[challenger_index_in_lists]:
            st.markdown(f" * **Statistically significant result** for {alphabet[i]} with p-value: {p_values[challenger_index_in_lists]:.4f}!")
            st.markdown(f" * **Conversion rate change** for {alphabet[i]}: {((conversion_rates[i] - conversion_rates[0]) / conversion_rates[0]) * 100:.2f}%")
            if conversion_rates[i] > conversion_rates[0]:
                st.success(f"Variant **{alphabet[i]}** is a **winner**, congratulations!")
            else:
                st.warning(f"**Loss averted** with variant **{alphabet[i]}**! Congratulations with this valuable insight.")

        else:
            st.markdown(f" * The Z-test is not statistically significant (p = {p_values[challenger_index_in_lists]:.4f}).")
            st.markdown(f" * **Conversion rate change for {alphabet[i]}:** {((conversion_rates[i] - conversion_rates[0]) / conversion_rates[0]) * 100:.2f}%")

            if tail == 'Greater' or tail == 'Two-sided':
                se_unpooled = np.sqrt(
                    (conversion_rates[0] * (1 - conversion_rates[0]) * reduction_factor / visitor_counts[0]) +
                    (conversion_rates[i] * (1 - conversion_rates[i]) * reduction_factor / visitor_counts[i])
                )

                z_stat_noninf = (conversion_rates[i] - conversion_rates[0] + non_inferiority_margin) / se_unpooled
                p_value_noninf = 1 - norm.cdf(z_stat_noninf)
                alpha_noninf = 1 - (confidence_noninf / 100)
                z_crit_ni = norm.ppf(1 - alpha_noninf)
                lower_bound_diff = (conversion_rates[i] - conversion_rates[0]) - (z_crit_ni * se_unpooled)

                st.markdown(f" * **P-value (non-inferiority test):** {p_value_noninf:.4f} (margin: {non_inferiority_margin*100:.1f}%)")
                st.markdown(f" * **Lower Bound of Difference:** {lower_bound_diff*100:.2f}% (Limit: {-non_inferiority_margin*100:.2f}%)")

                if p_value_noninf <= alpha_noninf:
                    st.success(f"Although not a winner, the non-inferiority test suggests that {alphabet[i]} is **not significantly worse** than {alphabet[0]} within the predefined margin.")
                else:
                    st.warning(f"The non-inferiority test does not provide sufficient evidence to conclude that {alphabet[i]} performs at least as well as {alphabet[0]}.")
            else:
                st.info(f"There is no strong evidence of a difference, and the effect size remains uncertain.")

        st.write("#### False Positive / Negative Risk")
        power = observed_powers[challenger_index_in_lists]
        fpr_alpha = results['sidak_alpha']

        if tail == 'Greater':
            fpr_label = "False Positive Risk"
            fpr_help = "Probability this apparent improvement is a false positive, given your prior."
            fpr_warning_suffix = "Consider replication before acting."
            fndr_label = "False Negative Discovery Rate"
            fndr_help = "Probability a real improvement was missed, given your prior and observed power."
            fndr_warning_suffix = "Power may be insufficient to detect a real improvement."
            fndr_success = "False Negative Risk is low: If a real improvement exists, this test was likely sensitive enough to find it."
        elif tail == 'Less':
            fpr_label = "False Positive Risk (Harm Detection)"
            fpr_help = "Probability this apparent harm is a false positive, given your prior."
            fpr_warning_suffix = "Review before concluding the variant is worse."
            fndr_label = "False Negative Discovery Rate (Missed Harm)"
            fndr_help = "Probability real harm was missed, given your prior and observed power."
            fndr_warning_suffix = "Power may be insufficient to detect real harm."
            fndr_success = "False Negative Risk is low: If the variant truly underperforms, this test was likely sensitive enough to detect it."
        else:
            fpr_label = "False Positive Risk"
            fpr_help = "Probability this detected difference (in either direction) is a false positive, given your prior."
            fpr_warning_suffix = "Direction is uncertain: Treat with caution before acting."
            fndr_label = "False Negative Discovery Rate"
            fndr_help = "Probability a real difference in either direction was missed, given your prior and observed power."
            fndr_warning_suffix = "Power may be insufficient to detect a real difference in either direction."
            fndr_success = "False Negative Risk is low: If a real difference exists in either direction, this test was likely sensitive enough to find it."

        if is_significant[challenger_index_in_lists]:
            fpr = calculate_fpr(alpha=fpr_alpha, power=power, prior=prior)
            st.metric(label=fpr_label, value=f"{fpr:.1%}", help=fpr_help)
            if fpr > 0.20:
                st.warning(f"With a {prior:.0%} prior, there's a {fpr:.1%} chance this result is a false positive. {fpr_warning_suffix}")
            else:
                st.success(f"False Positive Risk is low at {fpr:.1%}.")
        else:
            fndr = calculate_fndr(alpha=fpr_alpha, power=power, prior=prior)
            st.metric(label=fndr_label, value=f"{fndr:.1%}", help=fndr_help)
            if fndr > 0.20:
                st.warning(f"With a {prior:.0%} prior, there's a {fndr:.1%} chance a real effect was missed. {fndr_warning_suffix}")
            else:
                st.success(fndr_success)


# Main logic

def run():
    st.title("Experiment Analysis")
    st.markdown("""
    This app provides methods for Bayesian (beta RVS) and Frequentist analysis (z-test). Choose the appropriate method for your case.
    """)
    st.write("---")
    initialize_session_state()

    analysis_method = st.radio(
        "Choose your analysis method:",
        ("Frequentist Analysis", "Bayesian Analysis"),
        horizontal=True,
        help="Frequentist analysis uses z-tests and confidence intervals to assess statistical significance. Bayesian analysis uses simulations to estimate probabilities and potential business impact."
    )
    st.write("---")
    display_dynamic_documentation(analysis_method)
    st.write("---")

    # ==============================================================================
    #                             BAYESIAN ANALYSIS FLOW
    # ==============================================================================
    if analysis_method == "Bayesian Analysis":
        st.header("Bayesian Analysis Inputs")

        (
            visitor_counts,
            conversion_counts,
            aovs,
            aov_cv,
            beta_prior,
            lift_prior,
            probability_winner,
            runtime_days
        ) = get_bayesian_inputs()

        st.write("")
        if st.button("Calculate Bayesian Results", type="primary"):
            if validate_inputs(visitor_counts, conversion_counts, aovs):
                st.write("---")
                try:
                    cr_control = conversion_counts[0] / visitor_counts[0] if visitor_counts[0] > 0 else 0
                    observed_uplifts = [
                        ((conversion_counts[i] / visitor_counts[i]) - cr_control) / cr_control if cr_control > 0 and visitor_counts[i] > 0 else 0.0
                        for i in range(1, len(visitor_counts))
                    ]

                    probabilities_to_be_best, _ = calculate_probabilities(
                        visitor_counts, conversion_counts,
                        beta_prior=beta_prior, lift_prior=lift_prior,
                    )
                    uplift_distributions = simulate_uplift_distributions(
                        visitor_counts, conversion_counts,
                        beta_prior=beta_prior, lift_prior=lift_prior
                    )
                    df_business = perform_multi_variant_risk_assessment(
                        visitor_counts, conversion_counts, aovs,
                        probabilities_to_be_best, runtime_days,
                        beta_prior=beta_prior, lift_prior=lift_prior, aov_cv=aov_cv
                    )

                    plot_winner_probabilities_chart(probabilities_to_be_best)
                    plot_uplift_histograms(uplift_distributions, observed_uplifts)
                    display_results_per_variant(
                        probabilities_to_be_best, observed_uplifts,
                        probability_winner, aovs, runtime_days, df=df_business
                    )

                except Exception as e:
                    st.error(f"An error occurred during calculation: {e}")

    # ==============================================================================
    #                           FREQUENTIST ANALYSIS FLOW
    # ==============================================================================
    elif analysis_method == "Frequentist Analysis":
        st.header("Frequentist Analysis Inputs")
        visitor_counts, conversion_counts, confidence_level, reduction_factor, test_duration, sensitivity_mode, custom_prior = get_frequentist_inputs()
        st.write("---")
        st.session_state.tail = st.radio(
            "Select the test hypothesis (tail):",
            ('Greater', 'Less', 'Two-sided'),
            horizontal=True,
            help="'Two-sided' (A != B), 'Greater' (B > A), 'Less' (B < A)."
        )
        non_inferiority_margin = st.number_input(
            "Non-inferiority margin (absolute %)",
            min_value=0.0, max_value=10.0, value=1.0, step=0.1,
            help="Set the acceptable negative performance margin for non-significant results."
        ) / 100
        st.write("")

        if st.button("Calculate Frequentist Results", type="primary"):
            if validate_inputs(visitor_counts, conversion_counts):
                prior = resolve_prior(sensitivity_mode, custom_prior)
                st.write("---")
                try:
                    with st.spinner("Analysis in progress..."):
                        test_results = calculate_frequentist_statistics(
                            visitor_counts,
                            conversion_counts,
                            confidence_level,
                            st.session_state.tail,
                            reduction_factor=reduction_factor,
                        )

                    rf = test_results['reduction_factor']

                    # -- Variance adjustment feedback (metrics + prose only) ----------
                    # The φ value is now embedded in the density chart title, so no
                    # separate Plotly chart is needed here.
                    if rf < 1.0:
                        pct_reduced = (1 - rf) * 100
                        days_saved = calculate_time_savings(rf, test_duration)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Variance Reduction",
                                f"{pct_reduced:.1f}%",
                                help="How much noise was removed relative to the binomial baseline."
                            )
                        with col2:
                            st.metric(
                                "Estimated Time Saved",
                                f"{days_saved} Days",
                                delta="Faster Significance",
                                delta_color="normal",
                            )
                        st.success(
                            f"**Variance Adjustment:** Your historical conversion rate is more stable "
                            f"than pure binomial sampling predicts (φ = {rf:.2f}). "
                            f"Confidence intervals tightened - equivalent to running the experiment "
                            f"**{days_saved} days longer** without the adjustment."
                        )

                    elif rf > 1.0:
                        pct_inflated = (rf - 1) * 100
                        st.warning(
                            f"⚠️ **Overdispersion detected (φ = {rf:.2f}):** Your historical conversion "
                            f"rate varies more than binomial theory predicts - likely due to "
                            f"campaign bursts, seasonality, or other structural noise. "
                            f"Standard errors have been inflated by {pct_inflated:.1f}% to keep "
                            f"your false positive rate honest. Consider a quieter baseline period."
                        )

                    if test_results:
                        plot_conversion_distributions(test_results, reduction_factor=rf)
                        display_frequentist_summary(
                            test_results,
                            visitor_counts,
                            conversion_counts,
                            non_inferiority_margin=non_inferiority_margin,
                            reduction_factor=reduction_factor,
                            prior=prior,
                        )

                except Exception as e:
                    st.error(f"Analysis failed: {e}")

if __name__ == "__main__":
    run()
