import pandas as pd
import numpy as np
import csv
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from foe.continuous.operations import ContinuousMetricEngine
from foe.core.models import (
    AlternativeHypothesis,
    AnalysisUnit,
    ContinuousApproach,
    ContinuousMetricConfig,
)
from utility.automation_engine import winsorize_continuous_data

st.set_page_config(
    page_title="User level analysis",
    page_icon="🔢",
    layout="wide",
)

# Documentation
def render_documentation():
    # --- Expander 1: What the app does ---
    with st.expander("What this app does"):
        st.markdown(r"""
        This tool automates the selection and execution of statistical tests to compare different experimental variants. It uses two primary logic paths:

        ### 1. The Heuristic Framework
        This is a decision-engine that evaluates the underlying distribution of your data to pick the most mathematically sound test. It is particularly useful when evaluating metrics with negative values (i.e. profit).
        * **Count-Data Gate:** Before anything else, checks whether the KPI is a discrete, non-negative count (e.g. items or tickets per buyer). If so, routes straight to **Negative Binomial regression** instead of the continuous-data tree below, since count metrics violate the continuous assumptions those tests are built on regardless of what a normality test reports.
        * **Normality Check:** Uses D'Agostino's K² omnibus test (skewness + kurtosis) to see if residuals follow a normal distribution.
        * **Homogeneity Check:** Uses Levene’s test to verify if groups have equal variances.
        * **The Branches:**
            * *Negative Binomial Regression:* Used when the KPI is a discrete count metric (handles overdispersion natively via its dispersion parameter α).
            * *Standard ANOVA:* Used when data is normal and variances are equal.
            * *Welch's ANOVA:* Used when data is normal but variances are unequal.
            * *Kruskal-Wallis / Mann-Whitney:* Non-parametric alternatives used when data is non-normal.

        ### 2. The Gamma Model (MLE)
        Specifically designed for continuous, strictly positive, and right-skewed data (like **Revenue**). Discrete count metrics (e.g. items or tickets per buyer) are handled by the Heuristic path's count-data gate instead — see below.
        * It fits a Gamma distribution to each variant using **Maximum Likelihood Estimation (MLE)**.
        * The probability density function used is: $$f(x; k, \theta) = \frac{x^{k-1} e^{-x/\theta}}{\theta^k \Gamma(k)}$$
        * It compares models using a **Likelihood Ratio Test (LRT)** to determine if the differences in means are statistically significant.
        """)

    # --- Expander 2: Analysis unit (per visitor vs per transaction) ---
    with st.expander("Analysis unit: per visitor vs per transaction"):
        st.markdown(r"""
        Before testing, choose **what one row represents** for the comparison. This must
        match how you plan the experiment (and how you size it in the pre-test tool).
        The same choice applies whatever KPI you're analyzing — revenue, item quantity,
        profit, etc.

        ### Per transaction (positive rows only)
        * The unit is a **transaction**; the metric is e.g. average order value or items per order.
        * Rows with a value of **0 are excluded** — they aren't orders.
        * Gamma path: a single Gamma per variant (strictly positive data).
        * Answers *"did the value/quantity of an order change?"* It does **not** capture a change
          in how many people buy, and carries a mild selection effect (the set of orders
          is itself influenced by the treatment).

        ### Per visitor (includes zero rows)
        * The unit is a **visitor**; non-buyers count as **0**.
        * **All rows are kept**, including the zeros.
        * This is usually the real business outcome, and the unit you randomise on, so it
          captures both *more people buying* and *buyers buying/spending more*.
        * Gamma path: a **two-part (hurdle) model** — a Bernoulli component for the
          probability of converting, multiplied by a Gamma component for the value of
          those who do. The log-likelihoods of both parts are added, and the Likelihood
          Ratio Test uses the combined model. This is how the Gamma approach is made to
          *accept zeros* instead of dropping them.

        **Note:** the per-visitor option needs zero-value rows in your file (the
        non-converting visitors). If your CSV only contains orders, the two units are
        equivalent and you should upload visitor-level data to use per-visitor analysis.
        """)

    # --- Expander 3: How to use the app ---
    with st.expander("How to use the app"):
        st.markdown(r"""
        ### Data Prerequisites
        Your CSV must be in "long format" and contain at least two columns:
        1.  **Variant Label:** A categorical column (e.g., `experience_variant_label`) identifying the group (Control, Treatment A, etc.).
        2.  **KPI:** A numeric column (e.g., `purchase_revenue`, `profit`) containing the metric to test.

        For **per-visitor** analysis, include one row per visitor, with `0` in the KPI
        column for visitors who didn't buy. For **per-transaction** analysis, one row per
        order is enough.

        ### Outlier Handling
        Before running tests, you can choose how to handle extreme values. Zero-value rows
        (non-converting visitors) are never treated as outliers and are never capped.
        * **None:** Keep all data points as-is. Outliers (at 5 standard deviations) are still detected and reported.
        * **Capping (Winsorization):** Replaces values outside mean ± $n$ standard deviations, or outside the central $n$% of values (e.g. 95 → the 2.5th and 97.5th percentiles), with that bound. The bounds come from the non-zero rows. This reduces noise without losing sample size.
        * **Log Transform:** Tests $\log(1 + x)$ instead of the raw values.
        * **Removal:** Entirely deletes rows flagged as outliers. Up to 10,000 non-zero rows, a row is flagged by an OLS influence check (studentized residual, leverage or DFFITS beyond the threshold); above that, by each variant's mean ± $n$ standard deviations. With the Percentile method, removal uses a 5 standard deviation threshold. Use this if you suspect outliers are tracking errors.

        ### Control Variant
        The control you select is the baseline for the business case and for the post-hoc
        comparisons of the Gamma and Negative Binomial tests (each variant against the control).
        The Tukey, Games-Howell and Dunn post-hocs compare all pairs and don't use it.

        ### Choosing an Approach
        * Use **Heuristic (Auto-detect)** for metrics that can contain negative values (i.e. profit), or discrete count metrics (i.e. items/tickets per buyer) — the count-data gate inside this path routes those automatically to Negative Binomial regression.
        * Use **Gamma GLM** for monetary values or any metric where the data cannot be negative and has a long right-hand tail.
        """)

    # --- Expander 4: How to read the results ---
    with st.expander("How to read the results"):
        st.markdown(r"""
        ### 1. The P-Value
        The end statistic. We typically use a threshold of **0.05**.
        * **p < 0.05:** Significant result. There is a less than 5% chance the observed difference happened by random noise.
        * **p ≥ 0.05:** Not significant. We fail to reject the null hypothesis; the variants behave similarly.

        ### 2. Effect Size
        While the p-value tells you *if* there is a difference, the effect size tells you *how much it matters*. 
        * For Standard ANOVA, we report **Eta-Squared** (between-group share of the total variance); for Welch's ANOVA, **Partial Eta-Squared** (the same quantity in a one-way design).
        * For Kruskal-Wallis, we report **Eta-squared_H**, clamped at 0 (no effect).
        * For Mann-Whitney U, we report the **Rank-Biserial Correlation**; positive means the second variant (in alphabetical order) tends to be higher.
        * For Negative Binomial regression, we report the dispersion parameter **α** and defer to the relative lift in means for practical magnitude. The Gamma test has no effect size; use the business case or the relative lift.

        ### 3. Post-Hoc Analysis
        If you have more than two groups and the main test is significant, look at the **Post-Hoc** table.
        * After ANOVA (**Tukey HSD**), Welch's ANOVA (**Games-Howell**) or Kruskal-Wallis (**Dunn**, Bonferroni), every pair is compared. **"X vs Y"** reads as X minus Y; Tukey and Games-Howell also show the mean difference with its confidence interval.
        * After the Gamma or Negative Binomial test, each variant is compared against the control you selected, with pairwise Likelihood Ratio Tests and a Bonferroni correction.
        * The corrections prevent "p-hacking" or false positives that occur when running multiple simultaneous tests.
        * If a model can't be fitted (e.g. too few positive values in a variant), the test is treated as not significant (p = 1) and a warning explains why.
        
        ### 4. Gamma GLM Specifics
        When using the Gamma approach, the output focuses on **Model Fit** rather than variance:
        
        * **Log-Likelihood:** This represents how well the Gamma distribution fits your data. Higher (less negative) values indicate a better fit.
        * **Deviance:** A measure of the error in the model. When comparing variants, we look for a significant reduction in deviance.
        * **Likelihood Ratio Test (LRT):** This is the p-value source for Gamma. It tests the "Null Model" (one mean for all variants) against the "Alternative Model" (separate means for each variant). 
            * If **p < 0.05**, we conclude that at least one variant's mean is statistically different from the others under the Gamma assumption.
            * For **per-visitor** analysis the model is the two-part hurdle (Bernoulli × Gamma), so each variant carries one extra parameter (the conversion rate) and the test has correspondingly more degrees of freedom.
        * **Scale Parameter ($\theta$):** Indicates the "spread" or "dispersion" of the revenue. High scale parameters often point to high-variance "Whale" customers in revenue data.

        ### 5. Negative Binomial Specifics
        Used automatically for discrete count KPIs (e.g. items or tickets per buyer) inside the Heuristic path:

        * **Why not a regular z-test/ANOVA:** Count metrics are typically overdispersed (variance > mean) and, when conditioned on buyers, can have a mean above 1 — the binomial `p(1-p)` variance formula and continuous-data assumptions don't apply.
        * **Dispersion parameter (α):** Estimated directly from the data. Higher α indicates more overdispersion relative to a Poisson model.
        * **Likelihood Ratio Test (LRT):** Compares a Null model (one mean for all variants) against an Alternative model (separate means per variant), same logic as the Gamma LRT.
        * **Post-Hoc:** Pairwise Negative Binomial LRTs against a chosen control, Bonferroni-corrected.
        """)
    st.write("---")

# Preprocess data
def preprocess_data(df):
    errors = []
    # Normalize column names: strip spaces & convert to lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Rename columns based on keywords
    for col in df.columns:
        if 'variant' in col:
            df.rename(columns={col: 'experience_variant_label'}, inplace=True)

    # Validate that experience_variant_label exists
    if 'experience_variant_label' not in df.columns:
        errors.append("Column 'experience_variant_label' is missing after preprocessing. Please check your CSV file.")
        return df, errors  # Prevent further processing

    # Check for missing values in label
    if df['experience_variant_label'].isnull().any():
        errors.append("'experience_variant_label' contains null values.")

    # Ensure categorical variable
    df['experience_variant_label'] = pd.Categorical(df['experience_variant_label'])

    # Convert to numeric FIRST, then fill missing with 0
    numeric_cols = ['total_item_quantity', 'purchase_revenue', 'profit']
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df, errors

def _fmt_pair(a, b, p=None):
    """Format a post-hoc pair for the conclusion, e.g. 'B vs A (p=0.0012)'."""
    label = f"{a} vs {b}"
    if p is not None and pd.notna(p):
        label += f" (p={p:.4g})"
    return label

# Log transform data
def log_transform_data(df, kpi):
    df_copy = df.copy() # Create a copy to avoid SettingWithCopyWarning
    df_copy[kpi] = np.log1p(df_copy[kpi])  # log1p prevents log(0) issues
    return df_copy

# Perform statistical tests and provide conclusions

def _fmt_money(x):
    """Formats a monetary value for display, handling unbounded CI edges."""
    if x is None or (isinstance(x, float) and (np.isinf(x) or np.isnan(x))):
        return "Unbounded"
    return f"€{x:,.0f}"


GROUP_COL = 'experience_variant_label'

APPROACHES = {
    "Heuristic (Auto-detect)": ContinuousApproach.HEURISTIC,
    "Gamma GLM (Best for Revenue)": ContinuousApproach.GAMMA_GLM,
}

NB_TEST = "Negative Binomial Regression (LRT)"

# Why FOE picked each test, keyed by ContinuousMetricResult.test_name.
TEST_REASONS = {
    "Standard ANOVA": "Residuals are normal and variances are homogeneous.",
    "Welch's ANOVA": "Residuals are normal, but variances are heterogeneous. Welch's is robust to unequal variances.",
    "Kruskal-Wallis": "Data is not normally distributed; non-parametric test suitable for 3+ groups.",
    "Mann-Whitney U": "Data is not normally distributed (Non-parametric alternative to ANOVA).",
    NB_TEST: (
        "KPI is a discrete, non-negative count metric — Negative Binomial models the "
        "mean-variance relationship natively (including overdispersion) rather than "
        "assuming a continuous or bounded-proportion distribution."
    ),
    "Gamma GLM (Likelihood Ratio Test)": (
        "Gamma GLM selected. A Gamma model per variant is compared against one shared "
        "Gamma model with a Likelihood Ratio Test."
    ),
    "Two-Part Hurdle (Likelihood Ratio Test)": (
        "Gamma GLM selected with per-visitor analysis. A two-part model per variant "
        "(Bernoulli for converting × Gamma for the value of converters) is compared "
        "against one shared model with a Likelihood Ratio Test."
    ),
}

STAT_LABELS = {
    "F": "F-statistic",
    "H": "H-statistic",
    "U": "U-statistic",
    "LR": "Likelihood Ratio Statistic",
}

EFFECT_LABELS = {
    "eta_squared": "Eta-Squared",
    "partial_eta_squared": "Partial Eta-Squared",
    "eta_squared_h": "Eta-squared_H",
    "rank_biserial": "Rank-Biserial Correlation",
}

ADJUSTMENT_LABELS = {
    "bonferroni": "Bonferroni",
    "tukey": "Tukey",
    "games-howell": "Games-Howell",
}


def _posthoc_table(posthoc):
    """One row per PairwiseComparisonResult; 'X vs Y' means X minus Y."""
    adjustment = ADJUSTMENT_LABELS.get(posthoc[0].adjustment, posthoc[0].adjustment)
    stat_name = posthoc[0].statistic_name
    has_ci = any(r.mean_diff is not None for r in posthoc)
    rows = []
    for r in posthoc:
        row = {"Comparison": r.comparison, f"Statistic ({stat_name})": round(r.statistic, 4)}
        if posthoc[0].adjustment == "bonferroni":
            # Tukey / Games-Howell p-values are adjusted by construction, so
            # only Bonferroni has a separate raw p-value worth showing.
            row["p-value"] = round(r.p_value, 4)
        row[f"p-adj ({adjustment})"] = round(r.p_adj, 4)
        if has_ci:
            row["Mean diff (X − Y)"] = round(r.mean_diff, 4)
            row["CI low"] = round(r.ci_low, 4)
            row["CI high"] = round(r.ci_high, 4)
        row["Significant"] = "✅" if r.is_significant else "❌"
        rows.append(row)
    return pd.DataFrame(rows)


def _significance_vs_control(result, control, labels):
    """Per-challenger significance against the control, for the business case."""
    if len(labels) == 2:
        return {label: result.is_significant for label in labels if label != control}
    significance = {}
    for r in result.posthoc_results or []:
        if control in (r.group_a, r.group_b):
            other = r.group_b if r.group_a == control else r.group_a
            significance[other] = r.is_significant
    return significance


def perform_stat_tests_and_conclusions(
    df, kpi, approach, unit="per_transaction", control_label=None,
    count_max_unique=50, count_max_unique_ratio=0.05,
    include_business_case=False, test_duration_days=None, visitor_counts=None,
):
    st.write("---") # Separator
    st.write("## Statistical Test Results")
    kpi_label = kpi.replace('_', ' ')
    unit_label = f"{kpi_label} per visitor" if unit == "per_visitor" else f"{kpi_label} per transaction"
    st.write(f"_Analysis unit: **{unit_label}**._")

    # FOE compares variant labels as strings. Casting once keeps the control,
    # the post-hoc labels and the business-case keys aligned, also for
    # numeric labels such as 0 / 1.
    analysis_df = df[[GROUP_COL, kpi]].copy()
    analysis_df[GROUP_COL] = analysis_df[GROUP_COL].astype(str)
    control = str(control_label) if control_label is not None else None

    try:
        config = ContinuousMetricConfig(
            kpi=kpi,
            group_col=GROUP_COL,
            approach=APPROACHES[approach],
            unit=AnalysisUnit(unit),
            control_label=control,
            count_max_unique=int(count_max_unique),
            count_max_unique_ratio=float(count_max_unique_ratio),
        )
        result = ContinuousMetricEngine().run_comparison_suite(analysis_df, config)
    except ValueError as e:
        st.error(f"The analysis could not be run: {e}")
        return
    except Exception as e:
        st.error(f"An error occurred during the statistical analysis: {e}")
        return

    is_heuristic = config.approach == ContinuousApproach.HEURISTIC
    is_count = result.test_name == NB_TEST
    labels = [row[GROUP_COL] for row in result.summary_stats]  # sorted, as FOE tests them
    num_groups = len(labels)

    # --- 1. Assumption Checks ---
    if is_heuristic:
        st.write("### 1. Assumption Checks")
        if is_count:
            st.info("**Detected: discrete count metric.** Routing to Negative Binomial "
                    "regression instead of the continuous-data decision tree.")
            kpi_values = pd.to_numeric(analysis_df[kpi], errors="coerce").dropna()
            ratio = result.variance_mean_ratio
            ratio_text = f" (Variance/Mean ratio = {ratio:.2f})" if ratio is not None else ""
            st.write(f"* Mean = {kpi_values.mean():.4f}, Variance = {kpi_values.var():.4f}{ratio_text}")
            if ratio is not None and ratio <= 1.05:
                st.caption("_Variance ≈ Mean: a Poisson model would likely also fit, "
                           "but Negative Binomial is used as the general-purpose default._")
            else:
                st.caption("_Variance > Mean: overdispersion present, consistent with "
                           "why Negative Binomial (rather than Poisson or a binomial "
                           "z-test) is the appropriate model here._")
        else:
            st.write("_(Based on Normality of Residuals and Homogeneity of Variance)_")
            st.write("**Normality of Residuals (D'Agostino's K² Test):**")
            if result.normality_stat is not None:
                st.write(f"* Statistic (K²) = {result.normality_stat:.4f}")
                st.write(f"* p-value = {result.normality_p:.4f}")
                st.write(f"* _Conclusion: Residuals are likely {'normally distributed' if result.is_normal else 'NOT normally distributed'}._")
            else:
                st.caption(
                    "Fewer than 20 residuals — too few for a reliable omnibus "
                    "normality test; defaulting to non-parametric handling."
                )

            st.write("**Homogeneity of Variance (Levene's Test):**")
            if result.levene_stat is not None:
                st.write(f"* Statistic = {result.levene_stat:.4f}")
                st.write(f"* p-value = {result.levene_p:.4f}")
            else:
                st.write("* Levene's test could not be computed (e.g. a variant without spread).")
            st.write(f"* _Conclusion: Variances are likely {'homogeneous (equal)' if result.is_homogeneous else 'NOT homogeneous (unequal)'}._")
    else:
        if unit == "per_visitor":
            st.info("Diagnostic tests (normality/Levene) skipped. Using a two-part hurdle model (Bernoulli × Gamma) that includes non-converting visitors (zeros).")
        else:
            st.info("Diagnostic tests (normality/Levene) skipped. Gamma GLM is robust to non-normal, skewed data.")

    # --- 2. Main Statistical Test ---
    st.write("### 2. Main Comparison Test")
    st.info(f"**Test Chosen:** {result.test_name}")
    reason = TEST_REASONS.get(result.test_name)
    if reason:
        st.markdown(f"_Reason: {reason}_")

    if result.test_statistic is not None:
        stat_line = f"* {STAT_LABELS.get(result.test_statistic_name, result.test_statistic_name)} = {result.test_statistic:.4f}"
        if result.test_statistic_name == "U":
            stat_line += f" (for '{labels[0]}')"
        st.write(stat_line)
    st.write(f"* p-value = {result.p_value:.4g}")
    if result.dispersion_alpha is not None:
        st.write(f"* Estimated dispersion (α): {result.dispersion_alpha:.4f}")
        if result.dispersion_alpha == 0.0:
            st.caption("α = 0: the counts show no overdispersion, so the Negative Binomial "
                       "model reduces to Poisson. The test is still valid.")
    if result.effect_size is not None:
        effect_label = EFFECT_LABELS.get(result.effect_size_name, result.effect_size_name)
        st.write(f"* _Effect Size ({effect_label}): {result.effect_size:.4f}_")
        if result.effect_size_name == "rank_biserial":
            st.caption(f"Positive values mean '{labels[1]}' tends to be higher than '{labels[0]}'.")
        elif result.effect_size_name == "eta_squared_h":
            st.caption("Clamped at 0: values below 0 mean no effect.")

    for warning in result.warnings:
        st.warning(warning)

    # --- Post-hoc ---
    significant_pairs = []
    posthoc = result.posthoc_results
    if posthoc:
        st.info(f"**Post-Hoc Test ({result.posthoc_test_name}):**")
        if posthoc[0].statistic_name == "LR":
            st.markdown(f"_Reason: the global test was significant. Each treatment variant is "
                        f"compared against the control **'{control}'** with a pairwise Likelihood "
                        f"Ratio Test, Bonferroni-corrected._")
        else:
            st.markdown("_Reason: the global test was significant. Every pair of variants is "
                        "compared to identify which specific groups differ. **'X vs Y'** "
                        "reads as X minus Y._")
        st.dataframe(_posthoc_table(posthoc), hide_index=True)
        significant_pairs = [_fmt_pair(r.group_a, r.group_b, r.p_adj) for r in posthoc if r.is_significant]
    elif num_groups > 2 and not result.is_significant:
        st.info("Global test is not significant; pairwise comparisons are not required.")

    # --- 3. Summary Statistics ---
    st.write("### 3. Descriptive Statistics")
    st.caption(f"Means and standard deviations are computed per {unit_label.split(' per ')[-1]} (analysis unit: {unit_label}).")
    summary_stats = pd.DataFrame(result.summary_stats).set_index(GROUP_COL)
    st.dataframe(summary_stats)
    if not summary_stats.empty:
        highest_mean_variant = summary_stats['mean'].idxmax()
        highest_mean_value = summary_stats['mean'].max()
        st.write(f"* The variant with the highest mean is **'{highest_mean_variant}'** ({highest_mean_value:.4f}).")

        # Check if std calculation is possible (requires >1 sample per group)
        if summary_stats['count'].min() > 1:
            highest_std_variant = summary_stats['std'].idxmax()
            highest_std_value = summary_stats['std'].max()
            st.write(f"* The variant with the highest standard deviation is **'{highest_std_variant}'** ({highest_std_value:.4f}).")
        else:
            st.write("* Standard deviation calculation requires more than one sample per group.")

    # --- Business Case (optional) ---
    if include_business_case and test_duration_days:
        st.write("### Business Case")
        with st.expander("How is this calculated?"):
            st.markdown(
                "For **per visitor** analysis, each variant's mean KPI value already "
                "represents € (or units) per visitor, since non-converting visitors "
                "are counted as 0. For **per transaction** analysis, the mean is € "
                "per order, so it's combined with each variant's order rate "
                "(orders ÷ total visitors) to get the same per-visitor economics. "
                "The difference between a variant and the control is then projected "
                "over 183 days using the site's average daily visitor volume, with a "
                "95% confidence range from the standard errors of the underlying means "
                "(and order rates, for per-transaction)."
            )

        st.caption(f"_Control (baseline) variant: **{control}**._")
        group_stats = {row[GROUP_COL]: row for row in result.summary_stats}

        if control not in group_stats:
            st.error(
                f"Control variant '{control}' has no data after processing "
                "(e.g. it may have been entirely removed by outlier handling)."
            )
        elif unit == "per_transaction" and not visitor_counts:
            st.info(
                "Provide total visitor counts per variant above to enable the "
                "business case for per-transaction analysis."
            )
        else:
            try:
                if unit == "per_visitor":
                    daily_visitors = len(analysis_df) / test_duration_days
                    bc_counts = None
                else:
                    daily_visitors = sum(visitor_counts.values()) / test_duration_days
                    bc_counts = {str(k): v for k, v in visitor_counts.items()}

                bc_results = ContinuousMetricEngine().run_business_case(
                    group_stats=group_stats,
                    control_label=control,
                    unit=config.unit,
                    daily_visitors=daily_visitors,
                    visitor_counts=bc_counts,
                    alternative=AlternativeHypothesis.TWO_SIDED,
                    projection_period=183,
                    significance_by_variant=_significance_vs_control(result, control, labels),
                )

                for res in bc_results:
                    st.write(f"**{res['variant']}** vs **{control}**")
                    bc_col1, bc_col2 = st.columns(2)
                    bc_col1.metric(
                        f"Projected Impact ({res['projection_period']}d)",
                        _fmt_money(res["point_estimate"]),
                    )
                    bc_col2.metric(
                        "Confidence Range",
                        f"{_fmt_money(res['ci_low'])} to {_fmt_money(res['ci_high'])}",
                    )
                    st.caption(res["conclusion"])
            except Exception as e:
                st.error(f"Business case calculation failed: {e}")

        st.write("---")

    # --- 4. Conclusion ---
    st.write("### 4. Conclusion")
    effect_size = result.effect_size

    if result.is_significant:
        st.success(f"**A statistically significant difference was detected between the groups (p = {result.p_value:.4g}, using {result.test_name}).**")

        if num_groups > 2:
            if significant_pairs:
                st.write("**Significant pairwise differences** (after multiple-comparison correction):")
                for pair in significant_pairs:
                    st.markdown(f"* {pair}")
                st.markdown("_Refer to the summary statistics above to see which side of each pair is higher._")
            elif posthoc:
                st.write("The global test is significant, but no individual pair remained significant after multiple-comparison correction. See the post-hoc table above.")
            else:
                st.write("No post-hoc comparisons were available; see the warnings above.")

        elif num_groups == 2:
            st.write(f"The difference between the two groups ('{labels[0]}' and '{labels[1]}') is statistically significant.")

        # Effect Size Guidance
        if effect_size is not None:
            st.write(f"_Consider the effect size ({effect_size:.4f}) to evaluate the practical importance of this difference._")
        else:
            st.write("_Consider evaluating the relative lift to understand the practical importance of this difference._")

    else:
        st.info(f"**No statistically significant difference was detected between the groups (p = {result.p_value:.4g}, using {result.test_name}).**")
        # Add advice on checking effect size even if not significant
        if effect_size is not None:
            st.write(f"_The observed differences are likely due to chance. The effect size ({effect_size:.4f}) may indicate the magnitude of any observed, non-significant difference._")
        else:
            st.write("_The observed differences are likely due to chance. Consider calculating the effect size if understanding the magnitude of non-significant differences is important._")
        st.markdown("_If a meaningful difference was expected, potential reasons include insufficient data (low power) or a truly small effect. Consider effect size and sample size._")

    st.write("---")

@st.cache_data(show_spinner=True)
def _read_csv_cached(uploaded_file):
    sample = uploaded_file.read(2048).decode('utf-8', errors='ignore')
    uploaded_file.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=';,')
        sep = dialect.delimiter
    except csv.Error:
        sep = ','  # fallback default
    return pd.read_csv(uploaded_file, sep=sep)


def _distribution_plot_frame(frame, kpi, unit):
    """Pick the rows and y-axis window for the distribution charts.

    Per-visitor revenue is heavily zero-inflated (most visitors don't convert),
    which crushes the box plot to a flat line at 0 and the histogram to a single
    spike — no axis scaling fixes that, because the quartiles really are 0. In
    per-visitor mode we therefore chart only the converting (non-zero) rows so
    the spend distribution is legible, and return a note saying how many zeros
    were set aside (they remain in the statistical test). The window is +/- 3.5
    std around the mean of the plotted rows, floored at 0 only for non-negative
    data so genuine negatives (e.g. profit) stay visible.
    """
    if unit == "per_visitor":
        plot_frame = frame[frame[kpi] != 0]
        n_zero = len(frame) - len(plot_frame)
        note = (
            f"Showing the {len(plot_frame):,} converting (non-zero) rows; "
            f"{n_zero:,} zero-value visitors are omitted from this chart for "
            f"readability (they remain included in the statistical test)."
        ) if n_zero else None
    else:
        plot_frame = frame
        note = None

    series = plot_frame[kpi]
    if len(series) < 2:
        return plot_frame, None, None, note
    mean, std = series.mean(), series.std()
    lower = mean - 3.5 * std
    low = lower if series.min() < 0 else max(0, lower)
    return plot_frame, low, mean + 3.5 * std, note

# Main Streamlit app
def run():
    st.title("User Level Analysis")
    """
    This calculator lets you analyze revenue data or the amount of items of ecommerce transactions (or leads) for your online experiments. See the example CSV file for what you need to upload. 
    You're not limited to just A and B, but can add more labels when applicable (C, D, etc.).

    The app will identify outliers, fit models, and perform statistical tests. Based on the test results and the output of the highest average and highest standard deviation, you can determine which variant won.
    """
    # Load documentation
    render_documentation()

    st.download_button(
        label="Download CSV Template",
        data=pd.DataFrame({
            "experience_variant_label": ["A", "B", "B", "A"],
            "total_item_quantity": [5, 2, 4, 1],
            "purchase_revenue": [114.35, 45.74, 91.48, 22.87],
            "profit": [34.10, 12.50, 27.30, 5.60]
        }).to_csv(index=False),
        file_name="template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # --- Preprocessing progress bar ---
        prep_bar = st.progress(0, text="Reading the uploaded file…")
        raw_df = _read_csv_cached(uploaded_file)
        prep_bar.progress(55, text="Cleaning and validating columns…")
        df, errors = preprocess_data(raw_df.copy())
        prep_bar.progress(100, text="Preprocessing complete.")
        prep_bar.empty()

        if errors:
            for error in errors:
                st.error(error)
            return

        st.write("### A random sample of your data:")
        st.write(df.sample(min(10, len(df))))

        possible_kpis = ['purchase_revenue', 'total_item_quantity', 'profit']
        available_kpis = [col for col in possible_kpis if col in df.columns]

        if 'profit' in df.columns:
            # Check for negative values
            if df['profit'].min() < 0:
                st.warning("Warning: Your 'profit' column contains negative values. This is common (losses), but be aware of how this impacts log transformations or specific tests.")

        # If no valid KPIs are found, stop
        if not available_kpis:
            st.error("Error: None of the expected columns (purchase_revenue, total_item_quantity, profit) were found in your CSV.")
            return

        kpi = st.selectbox("Select the KPI to analyze:", available_kpis)

        # --- Analysis unit selector (per visitor vs per transaction) ---
        # Generic wording: this selector applies to whichever KPI is chosen
        # (revenue, item quantity, profit, etc.), not just revenue, so the
        # labels use the KPI's own name rather than hardcoding "Revenue".
        kpi_label = kpi.replace('_', ' ')
        unit_choice = st.selectbox(
            "Select the analysis unit:",
            [
                f"{kpi_label} per visitor (includes zero-value rows)",
                f"{kpi_label} per transaction (positive rows only)",
            ],
            help=(
                "Per visitor averages over every visitor, counting non-buyers as 0 "
                "(captures conversion-rate and volume/spend effects together; needs "
                "zero rows in your file). Per transaction looks at the value among "
                "buyers/orders only."
            ),
        )
        unit = "per_visitor" if "per visitor" in unit_choice else "per_transaction"

        # Inspect the zero structure of the chosen KPI to guide the user.
        kpi_series = df[kpi].dropna()
        n_zeros = int((kpi_series == 0).sum())
        n_neg = int((kpi_series < 0).sum())

        if unit == "per_visitor":
            if n_zeros == 0:
                st.info(
                    "No zero-value rows were found for this KPI, so per-visitor and "
                    "per-transaction analysis are equivalent here. To capture "
                    "non-converting visitors, upload visitor-level data with 0 for non-buyers."
                )
            else:
                st.success(f"Per-visitor analysis will include all rows, "
                           f"of which {n_zeros:,} are non-converting (zero-value) visitors.")
        else:  # per_transaction
            if n_zeros > 0:
                st.info(f"Per-transaction analysis will exclude {n_zeros:,} zero-value row(s) (non-orders).")

        # --- Control variant (one selector for post-hocs and business case) ---
        control_label_input = st.selectbox(
            "Select the control (baseline) variant:",
            df['experience_variant_label'].unique(),
            key="control_label",
            help=(
                "Used as the baseline for the post-hoc comparisons of the Gamma and "
                "Negative Binomial tests (3+ variants) and for the business case. "
                "The Tukey, Games-Howell and Dunn post-hocs compare all pairs and "
                "don't need a control."
            ),
        )

        # --- Business case (optional) ---
        st.write("#### Business Case (optional)")
        include_business_case = st.checkbox(
            "Calculate a monetary business case",
            value=False,
            help=(
                "Projects the KPI difference between each challenger and the control "
                "into a future € impact, using your test's traffic and duration."
            ),
        )
        test_duration_days = None
        visitor_counts_input = None
        if include_business_case:
            test_duration_days = st.number_input(
                "Test duration so far (days):", min_value=1, value=14, step=1,
            )
            if unit == "per_transaction":
                st.caption(
                    "Per-transaction analysis only counts orders, so the business case "
                    "needs each variant's total visitor count to derive its order rate."
                )
                visitor_counts_input = {}
                for group in df['experience_variant_label'].unique():
                    visitor_counts_input[group] = st.number_input(
                        f"Total visitors for '{group}':",
                        min_value=1, value=1000, step=1, key=f"bc_visitors_{group}",
                    )

        # Select analysis approach (Gamma model for revenue, heuristic framework for e.g. profit or item counts
        analysis_approach = st.selectbox(
            "Select Statistical Approach:",
            ["Heuristic (Auto-detect)", "Gamma GLM (Best for Revenue)"],
            help="Heuristic follows a count-data gate (Negative Binomial for discrete counts) then a Normality/Variance decision tree. Gamma GLM is optimized for strictly positive, right-skewed continuous data. For per-visitor analysis, Gamma GLM uses a two-part hurdle model so zeros are included."
        )

        # Guard: the Gamma family needs non-negative data.
        if analysis_approach == "Gamma GLM (Best for Revenue)" and n_neg > 0:
            st.warning(
                f"The Gamma model requires non-negative values, but '{kpi}' contains "
                f"{n_neg:,} negative value(s). Use the Heuristic approach for this metric, "
                "or pick a non-negative KPI."
            )

        # --- Count-data detection thresholds (Heuristic path only) ---
        # What "looks like a count metric" varies a lot by business context —
        # a low-volume B2B funnel might have counts spanning dozens of
        # distinct values, while a high-volume consumer funnel might have a
        # coincidentally-round continuous KPI with few distinct values.
        # Exposed here rather than hardcoded so it can be tuned per dataset.
        count_max_unique = 50
        count_max_unique_ratio = 5.0
        if analysis_approach == "Heuristic (Auto-detect)":
            with st.expander("Count-metric detection settings (advanced)"):
                st.markdown(
                    "The Heuristic path checks whether the KPI looks like a discrete "
                    "count (e.g. items/tickets per buyer) before running normality "
                    "checks, and routes count metrics to Negative Binomial regression "
                    "instead. Adjust these thresholds if your KPI is being "
                    "mis-classified as continuous or as a count."
                )
                count_max_unique = st.number_input(
                    "Max distinct values to still call it a count metric",
                    min_value=2, max_value=1000, value=50, step=1,
                    help=(
                        "If the KPI has this many or fewer distinct values (and is "
                        "non-negative and integer-valued), it's treated as a count "
                        "metric regardless of dataset size."
                    ),
                )
                count_max_unique_ratio = st.number_input(
                    "Max distinct-values / row-count ratio (%) to still call it a count metric",
                    min_value=0.1, max_value=99.9, value=5.0, step=0.5,
                    help=(
                        "Fallback for large datasets: even if distinct values exceed "
                        "the cap above, the KPI is still treated as a count metric if "
                        "distinct values are below this percentage of total rows."
                    ),
                )


        # Select how to handle outliers
        outlier_handling = st.selectbox("Select how to handle outliers:", ['None', 'Winsorizing (STD/Percentile)', 'Log Transform', 'Removal'], help='Choose the method for handling outliers. "None" uses a default > 5 standard deviation definition for detection purposes. Only use Log Transform when your data does not contain negative numbers.')

        method = None
        outlier_stdev = None
        percentile = None

        if outlier_handling not in ['None', 'Log Transform']:
            method = st.selectbox("Select outlier detection method:", ['Standard Deviation', 'Percentile'])
            if method == 'Standard Deviation':
                outlier_stdev = st.selectbox("How many standard deviations define an outlier?", [2, 3, 4, 5])
            elif method == 'Percentile':
                percentile = st.selectbox("Select percentile for Winsorization:", [90, 95, 99])

        # Detection threshold: the chosen SD count, else a default of 5 SD
        # (for reporting under None / Log Transform / Percentile, and for
        # Removal + Percentile).
        detect_k = outlier_stdev if method == 'Standard Deviation' else 5
        with st.spinner("Scanning for outliers…"):
            try:
                outlier_res = ContinuousMetricEngine.detect_outliers(
                    df, kpi, group_col='experience_variant_label',
                    threshold=detect_k, exclude_zeros=True,
                )
                outliers_mask = pd.Series(outlier_res.mask, index=df.index)
            except Exception as e:
                st.error(f"Error during outlier detection: {e}")
                outlier_res = None
                outliers_mask = pd.Series(False, index=df.index)

        if outlier_res is not None:
            st.info(f"Dataset has {len(df):,} rows ({outlier_res.n_considered:,} non-zero).")
            if outlier_res.n_outliers > 0:
                if outlier_res.method == "per_variant_sd":
                    how = (f"With more than 10,000 non-zero rows, values outside each variant's "
                           f"mean ± {detect_k} standard deviations were flagged.")
                else:
                    how = (f"An OLS influence check was used: a row is flagged when its studentized "
                           f"residual, leverage or DFFITS exceeds its {detect_k}-SD threshold.")
                st.warning(f"{outlier_res.n_outliers:,} outlier(s) detected. {how} Zero-value rows are "
                           "never flagged. You can adjust the outlier handling method in the options above.")

        # Distribution charts. For zero-inflated per-visitor data, chart the
        # converting (non-zero) rows so the zero spike doesn't crush the scale.
        num_variants = len(df['experience_variant_label'].unique())
        raw_plot_df, raw_min, raw_max, raw_note = _distribution_plot_frame(df, kpi, unit)

        st.write("### Raw Data Box Plot")
        fig_box, ax_box = plt.subplots()
        sns.boxplot(x='experience_variant_label', y=kpi, data=raw_plot_df, palette=sns.color_palette("hls", num_variants), hue='experience_variant_label', legend=False)
        if raw_min is not None:
            ax_box.set_ylim(raw_min, raw_max)
        st.pyplot(fig_box)
        if raw_note:
            st.caption(raw_note)
        plt.clf()

        st.write("### Raw Data Histogram with KDE")
        fig_hist, ax_hist = plt.subplots()
        hist_kwargs = {"kde": True, "bins": 30, "ax": ax_hist}
        if raw_min is not None:
            # Bin within the visible window so a long tail/outlier doesn't make
            # the bins so wide the data collapses into one bar.
            hist_kwargs["binrange"] = (raw_min, raw_max)
        sns.histplot(raw_plot_df[kpi], **hist_kwargs)
        if raw_min is not None:
            ax_hist.set_xlim(raw_min, raw_max)
        plt.title("Raw Data Histogram with KDE")
        st.pyplot(fig_hist)
        if raw_note:
            st.caption(raw_note)
        plt.clf()


        if st.button("Calculate my test results", type="primary"):
            action_bar = st.progress(0, text="Preparing data…")
            processed_df = df.copy()

            # --- Analysis-unit zero handling ---
            # Per transaction: a row must be an actual order, so drop zero-value rows.
            # Per visitor: keep all rows (the zeros are non-converting visitors).
            if unit == "per_transaction":
                initial_rows = len(processed_df)
                processed_df = processed_df[processed_df[kpi] != 0]
                rows_removed = initial_rows - len(processed_df)
                if rows_removed > 0:
                    st.info(f"Per-transaction analysis: filtered out {rows_removed:,} zero-value (non-order) rows.")
            else:
                if (processed_df[kpi] == 0).any():
                    st.info("Per-visitor analysis: zero-value (non-converting) rows are kept.")

            action_bar.progress(20, text="Applying outlier handling…")

            # --- Outlier Handling ---
            if outlier_handling == 'Winsorizing (STD/Percentile)':
                # FOE's winsorize_series with exclude_zeros=True: caps come from
                # the non-zero rows and zero rows stay at 0.
                processed_df, lower_cap, upper_cap, cap_desc = winsorize_continuous_data(
                    processed_df, kpi, method, outlier_stdev, percentile,
                )

                # Display a message based on the description returned by the function
                if lower_cap is not None and upper_cap is not None:
                    st.success(f"Winsorizing applied: Values capped between {lower_cap:.4f} and {upper_cap:.4f} (based on {cap_desc}).")
                else:
                    st.warning(f"Winsorizing attempted but no capping applied ({cap_desc}).")
            elif outlier_handling == 'Log Transform':
                processed_df = log_transform_data(processed_df, kpi)
                st.warning(r"""
                    **Important: Log Transformation Detected**
                    
                    By selecting Log Transformation, the statistical tests will be performed on the 
                    **log-scaled values** (using log1p). 
                    
                    * **Interpretation:** You are now evaluating **proportional (rate-based) changes** rather than absolute differences. 
                    * **Requirement:** Ensure your KPI does not contain negative values. Zeros are safely handled.
                    * **Back-transformation:** The reported means will represent the **Geometric Mean** of your data, which is less sensitive to extreme right-skewed outliers.
                """)
            elif outlier_handling == 'Removal':
                # Align the precomputed mask to the (possibly zero-filtered) rows.
                mask_aligned = outliers_mask.reindex(processed_df.index, fill_value=False)
                variants_before = set(processed_df['experience_variant_label'].unique())
                processed_df = processed_df[~mask_aligned]
                st.write(f"Outliers removed: {int(mask_aligned.sum())} rows affected.")
                emptied = variants_before - set(processed_df['experience_variant_label'].unique())
                if emptied:
                    st.warning(
                        f"Outlier removal removed every row of variant(s) "
                        f"{', '.join(repr(str(v)) for v in sorted(emptied, key=str))}. This happens "
                        "when a variant is much smaller than the others (its rows all get high "
                        "leverage). Consider Winsorizing or a higher threshold instead."
                    )
            else:
                st.warning("No outlier handling applied.")

            action_bar.progress(45, text="Fitting model…")

            # --- Refit the OLS model after outlier handling ---
            # Only used for the residual charts below; the tests run in FOE.
            model_after = None
            if analysis_approach != "Gamma GLM (Best for Revenue)":
                try:
                    model_after = smf.ols(f'{kpi} ~ C(experience_variant_label)', data=processed_df).fit()
                except Exception as e:
                    st.warning(f"Could not fit the OLS model for the residual charts: {e}")

            action_bar.progress(60, text="Rendering diagnostic charts…")

            # --- Processed Data Plots (using the processed data) ---
            num_variants = len(processed_df['experience_variant_label'].unique())
            proc_plot_df, processed_min, processed_max, proc_note = _distribution_plot_frame(processed_df, kpi, unit)

            st.write("### Refitted Data Box Plot")
            fig_box, ax_box = plt.subplots()
            sns.boxplot(x='experience_variant_label', y=kpi, data=proc_plot_df, palette=sns.color_palette("hls", num_variants), hue='experience_variant_label', legend=False)
            if processed_min is not None:
                ax_box.set_ylim(processed_min, processed_max)
            st.pyplot(fig_box)
            if proc_note:
                st.caption(proc_note)
            plt.clf()
            
            if model_after is not None:
                st.write("### Refitted QQ Plot")
                influence = model_after.get_influence()
                std_residuals = pd.Series(influence.resid_studentized_internal)
                plot_data = std_residuals
                caption_text = ""

                if len(std_residuals) > 5000:
                    plot_data = std_residuals.sample(5000, random_state=42)  # Sample to avoid performance issues
                    caption_text = f"Note: Due to a large dataset, only a 5000 points (out of {len(std_residuals):,}) are plotted for efficiency."

                fig_qq = plt.figure()
                ax_qq = fig_qq.add_subplot(111)

                sm.qqplot(plot_data, line='45', ax=ax_qq, alpha=0.2, markersize=4, marker='o')
                plt.title("QQ Plot of Standardized Residuals")
                plt.ylabel("Standardized Residuals (Z-Score)")
                plt.xlabel("Theoretical Quantiles")

                st.pyplot(fig_qq)
                if caption_text:
                    st.caption(caption_text)

                plt.clf()

                st.write("### Refitted Data Histogram with KDE")
                resid_series = model_after.resid
                resid_note = None
                if unit == "per_visitor":
                    # Residuals are zero-inflated too; show converting rows so the
                    # zero spike doesn't collapse the histogram into a single bar.
                    resid_series = resid_series[processed_df[kpi] != 0]
                    resid_note = "Residuals shown for converting (non-zero) rows only."
                resid_mean, resid_std = resid_series.mean(), resid_series.std()
                resid_min = resid_mean - 3.5 * resid_std
                resid_max = resid_mean + 3.5 * resid_std
                fig_hist, ax_hist = plt.subplots()
                hist_kwargs = {"kde": True, "bins": 30, "ax": ax_hist}
                if pd.notna(resid_std) and resid_std > 0:
                    hist_kwargs["binrange"] = (resid_min, resid_max)
                sns.histplot(resid_series, **hist_kwargs)  # residuals from the refit
                if pd.notna(resid_std) and resid_std > 0:
                    ax_hist.set_xlim(resid_min, resid_max)
                plt.title("Histogram of Residuals with KDE")
                st.pyplot(fig_hist)
                if resid_note:
                    st.caption(resid_note)
                plt.clf()

            action_bar.progress(80, text="Running statistical tests…")
            perform_stat_tests_and_conclusions(
                processed_df, kpi, analysis_approach, unit,
                control_label=control_label_input,
                count_max_unique=count_max_unique,
                count_max_unique_ratio=count_max_unique_ratio / 100.0,
                include_business_case=include_business_case,
                test_duration_days=test_duration_days,
                visitor_counts=visitor_counts_input,
            )
            action_bar.progress(100, text="Analysis complete.")
            action_bar.empty()
            
if __name__ == "__main__":
    run()
