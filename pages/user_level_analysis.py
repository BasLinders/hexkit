import pandas as pd
import numpy as np
import csv
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as dm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats
from scipy.stats import normaltest, levene, kruskal, mannwhitneyu
from scipy.optimize import minimize
from scipy.special import gammaln
import seaborn as sns
import matplotlib.pyplot as plt
from pingouin import welch_anova, qqplot, pairwise_gameshowell
import scikit_posthocs as sp
import streamlit as st
import itertools

from foe.continuous.operations import ContinuousMetricEngine
from foe.core.models import AnalysisUnit, AlternativeHypothesis

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
        Before running tests, you can choose how to handle extreme values:
        * **None:** Keep all data points as-is.
        * **Capping (Winsorization):** Replaces values above the $n^{th}$ percentile with the $n^{th}$ percentile value (or $n$ standard deviations). This reduces noise without losing sample size.
        * **Removal:** Entirely deletes rows where the KPI exceeds the $n^{th}$ percentile (or $n$ standard deviations). Use this if you suspect outliers are tracking errors.

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
        * For ANOVA, we report **Partial Eta-Squared**.
        * For non-parametric tests, we report **Rank-Biserial Correlation** or **Eta-squared_H**.
        * For Negative Binomial regression, we report the dispersion parameter **α** and defer to the relative lift in means for practical magnitude.

        ### 3. Post-Hoc Analysis
        If you have more than two groups and the main test is significant, look at the **Post-Hoc** table.
        * It performs pairwise comparisons (Group A vs Group B, B vs C, etc.).
        * It uses corrections (like **Tukey**, **Games-Howell**, **Bonferroni**, or pairwise NB LRTs with Bonferroni) to prevent "p-hacking" or false positives that occur when running multiple simultaneous tests.
        
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

# Gamma model
def fit_gamma(data):
    """
    Fits a Gamma distribution (shape k, scale theta) to data using MLE.
    Returns: (k, theta, log_likelihood)
    """
    # Remove zeros/negatives as Gamma is defined for x > 0
    data = data[data > 0]
    
    # Initial guesses using Method of Moments
    mean_x = np.mean(data)
    var_x = np.var(data)
    k_start = mean_x**2 / var_x
    theta_start = var_x / mean_x

    # Negative Log-Likelihood function
    def neg_log_likelihood(params):
        k, theta = params
        if k <= 0 or theta <= 0:
            return 1e10
        # Gamma PDF log-likelihood formula
        n = len(data)
        ll = (n * (k - 1) * np.mean(np.log(data)) - 
              n * k * np.log(theta) - 
              n * np.mean(data) / theta - 
              n * gammaln(k))
        return -ll

    res = minimize(neg_log_likelihood, [k_start, theta_start], method='L-BFGS-B', bounds=[(1e-5, None), (1e-5, None)])
    k_mle, theta_mle = res.x
    return k_mle, theta_mle, -res.fun


def fit_unit_model(data, unit):
    """
    Fit the likelihood model for the chosen analysis unit and return
    (log_likelihood, num_parameters).

    * per_transaction -> a single Gamma on the positive values (2 params: k, theta).
    * per_visitor     -> a two-part hurdle model that *accepts zeros*:
                         a Bernoulli on P(value > 0) plus a Gamma on the positive
                         values. The log-likelihoods add; parameters are
                         (p, k, theta) = 3.

    The per-visitor model is what lets the Gamma approach handle non-converting
    visitors (zeros) instead of silently discarding them.
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]

    if unit == "per_visitor":
        n = len(data)
        positives = data[data > 0]
        n_pos = len(positives)
        if n == 0:
            return np.nan, 3

        # Bernoulli (conversion) part, evaluated at its MLE p = n_pos / n.
        p = n_pos / n
        if 0.0 < p < 1.0:
            ll_bern = n_pos * np.log(p) + (n - n_pos) * np.log(1.0 - p)
        else:
            # p == 0 (no buyers) or p == 1 (no zeros): Bernoulli LL is 0 at the MLE.
            ll_bern = 0.0

        # Gamma (spend among buyers) part.
        if n_pos >= 2 and np.var(positives) > 0:
            _, _, ll_gamma = fit_gamma(positives)
        else:
            ll_gamma = 0.0

        return ll_bern + ll_gamma, 3

    # per_transaction
    positives = data[data > 0]
    if len(positives) < 2 or np.var(positives) == 0:
        return np.nan, 2
    _, _, ll_gamma = fit_gamma(positives)
    return ll_gamma, 2


def perform_gamma_test(df, kpi, unit="per_transaction"):
    # 1. Null Model: one model fit to all data
    all_data = df[kpi].values
    ll_null, n_params_per_model = fit_unit_model(all_data, unit)

    # 2. Alternative Model: a separate model fit to each variant
    variants = df['experience_variant_label'].unique()
    num_variants = len(variants)

    ll_alt = 0.0
    for v in variants:
        variant_data = df[df['experience_variant_label'] == v][kpi].values
        ll_v, _ = fit_unit_model(variant_data, unit)
        ll_alt += ll_v

    if np.isnan(ll_null) or np.isnan(ll_alt):
        return np.nan, np.nan

    # 3. Likelihood Ratio Test
    #    df = (params in Alt) - (params in Null)
    #       = n_params_per_model * num_variants - n_params_per_model
    df_diff = n_params_per_model * (num_variants - 1)
    lr_stat = max(2.0 * (ll_alt - ll_null), 0.0)
    p_value = stats.chi2.sf(lr_stat, df=df_diff)

    return p_value, lr_stat


# ---------------------------------------------------------------------------
# Negative Binomial model (discrete count KPIs, e.g. items/tickets per buyer)
# ---------------------------------------------------------------------------

def is_count_kpi(series, max_unique_for_check=50, max_unique_ratio=0.05):
    """
    Heuristic gate: treat the KPI as a discrete count metric if all
    non-null values are non-negative integers (or integer-valued floats,
    e.g. 3.0), AND the cardinality looks like a true count rather than a
    continuous KPI that happens to have round values (e.g. revenue
    rounded to whole euros).

    This check runs BEFORE the normality/homogeneity assumption checks,
    since a count metric (discrete, often mean > 1, frequently
    overdispersed) violates the continuous-data assumptions those tests
    are built on regardless of what a normality test reports on a large
    sample.

    Both cardinality thresholds are exposed as parameters (rather than
    hardcoded) because what "looks like a count" varies a lot by business
    context — a KPI with 40 distinct values might be a clear count metric
    for a low-volume B2B funnel, or a coincidentally-round continuous KPI
    for a high-volume consumer funnel. Callers are expected to surface
    these as UI inputs rather than relying on one global default.

    max_unique_for_check: absolute cap on distinct values to still call it
        a count metric (e.g. 50 -> KPIs with more than 50 distinct values
        are treated as continuous unless the ratio check below applies).
    max_unique_ratio: distinct-values-to-row-count ratio below which the
        KPI is still treated as a count metric even if it exceeds the
        absolute cap above (relevant for very large datasets where a
        genuine count metric can have many distinct values in absolute
        terms while still being a tiny fraction of total rows).
    """
    s = series.dropna()
    if s.empty:
        return False
    if (s < 0).any():
        return False
    if not np.allclose(s, np.round(s)):
        return False
    n_unique = s.nunique()
    return n_unique <= max_unique_for_check or (n_unique / len(s)) < max_unique_ratio


def fit_negbin(data, exog=None):
    """
    Fits a Negative Binomial (NB2) model via MLE.
    If exog is None, fits an intercept-only model.
    Returns (log_likelihood, alpha, fitted_results).
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if exog is None:
        exog = np.ones((n, 1))
    model = dm.NegativeBinomial(data, exog, loglike_method='nb2')
    try:
        res = model.fit(disp=0, maxiter=200)
    except Exception:
        start = np.zeros(exog.shape[1] + 1)
        start[0] = np.log(np.mean(data) + 1e-6)
        start[-1] = 1.0
        res = model.fit(disp=0, maxiter=200, start_params=start)
    return res.llf, res.params[-1], res


def perform_negbin_test(df, kpi, group_col='experience_variant_label'):
    """
    Likelihood Ratio Test comparing a Null NB model (single mean for all
    variants) against an Alternative model (separate mean per variant).
    Mirrors perform_gamma_test's LRT structure.

    Returns (p_value, lr_stat, alpha_estimate, fitted_alt_model).
    """
    data = df[kpi].values.astype(float)
    groups = pd.Categorical(df[group_col])
    dummies = pd.get_dummies(groups, drop_first=True).values.astype(float)
    n = len(data)

    # Null model: intercept only
    exog_null = np.ones((n, 1))
    ll_null, _, _ = fit_negbin(data, exog_null)

    # Alternative model: intercept + variant dummies
    exog_alt = np.column_stack([np.ones(n), dummies]) if dummies.shape[1] > 0 else exog_null
    ll_alt, alpha_alt, res_alt = fit_negbin(data, exog_alt)

    df_diff = exog_alt.shape[1] - exog_null.shape[1]
    if df_diff <= 0:
        return np.nan, np.nan, alpha_alt, res_alt

    lr_stat = max(2.0 * (ll_alt - ll_null), 0.0)
    p_value = stats.chi2.sf(lr_stat, df=df_diff)

    return p_value, lr_stat, alpha_alt, res_alt


def _fmt_pair(a, b, p=None):
    """Format a post-hoc pair for the conclusion, e.g. 'B vs A (p=0.0012)'."""
    label = f"{a} vs {b}"
    if p is not None and pd.notna(p):
        label += f" (p={p:.4g})"
    return label


def run_gamma_posthoc(df, kpi, group_col, control_label, unit="per_transaction"):
    st.write("### Pairwise Post-Hoc Comparisons (Gamma LRT)")
    model_desc = "two-part hurdle" if unit == "per_visitor" else "Gamma"
    st.markdown(f"_Each treatment variant is compared against the control using "
                f"{model_desc} Likelihood Ratio Tests._")

    variants = [v for v in df[group_col].unique() if v != control_label]
    posthoc_data = []
    significant_pairs = []

    # Bonferroni correction factor
    num_comparisons = len(variants)

    for variant in variants:
        # Filter data for pairwise comparison
        pair_df = df[df[group_col].isin([control_label, variant])]

        # 1. Null Model (single model for both groups combined)
        null_log_lik, n_params_per_model = fit_unit_model(pair_df[kpi].values, unit)

        # 2. Alternative Model (separate models for control and variant)
        ctrl_data = pair_df[pair_df[group_col] == control_label][kpi].values
        var_data = pair_df[pair_df[group_col] == variant][kpi].values
        ll_ctrl, _ = fit_unit_model(ctrl_data, unit)
        ll_var, _ = fit_unit_model(var_data, unit)
        alt_log_lik = ll_ctrl + ll_var

        if np.isnan(null_log_lik) or np.isnan(alt_log_lik):
            p_val = np.nan
            lrt_stat = np.nan
            adj_p = np.nan
            sig = "—"
        else:
            # df = params(Alt) - params(Null) = 2*k - k = k (2 or 3)
            lrt_stat = max(2.0 * (alt_log_lik - null_log_lik), 0.0)
            p_val = stats.chi2.sf(lrt_stat, df=n_params_per_model)
            adj_p = min(p_val * num_comparisons, 1.0)
            sig = "✅" if adj_p < 0.05 else "❌"
            if adj_p < 0.05:
                significant_pairs.append(_fmt_pair(variant, control_label, adj_p))

        posthoc_data.append({
            "Comparison": f"{variant} vs {control_label}",
            "LRT Stat": np.round(lrt_stat, 4) if not np.isnan(lrt_stat) else "N/A",
            "p-value": np.round(p_val, 4) if not np.isnan(p_val) else "N/A",
            "p-adj (Bonferroni)": np.round(adj_p, 4) if not np.isnan(adj_p) else "N/A",
            "Significant": sig
        })

    st.dataframe(pd.DataFrame(posthoc_data))
    return significant_pairs


def run_negbin_posthoc(df, kpi, group_col, control_label):
    """
    Pairwise Negative Binomial LRTs against a chosen control, Bonferroni
    corrected. Mirrors run_gamma_posthoc's structure and output format so
    significant pairs fold into the same conclusion formatting.
    """
    st.write("### Pairwise Post-Hoc Comparisons (Negative Binomial LRT)")
    st.markdown("_Each treatment variant is compared against the control using "
                "pairwise Negative Binomial Likelihood Ratio Tests._")

    variants = [v for v in df[group_col].unique() if v != control_label]
    posthoc_data = []
    significant_pairs = []
    num_comparisons = len(variants)

    for variant in variants:
        pair_df = df[df[group_col].isin([control_label, variant])]

        try:
            p_val, lrt_stat, alpha_est, _ = perform_negbin_test(pair_df, kpi, group_col)
            if pd.isna(p_val):
                raise ValueError("LRT degrees of freedom <= 0")
            adj_p = min(p_val * num_comparisons, 1.0)
            sig = "✅" if adj_p < 0.05 else "❌"
            if adj_p < 0.05:
                significant_pairs.append(_fmt_pair(variant, control_label, adj_p))
        except Exception as e:
            st.error(f"Error during pairwise NB test ({variant} vs {control_label}): {e}")
            p_val, lrt_stat, adj_p, sig = np.nan, np.nan, np.nan, "—"

        posthoc_data.append({
            "Comparison": f"{variant} vs {control_label}",
            "LRT Stat": np.round(lrt_stat, 4) if pd.notna(lrt_stat) else "N/A",
            "p-value": np.round(p_val, 4) if pd.notna(p_val) else "N/A",
            "p-adj (Bonferroni)": np.round(adj_p, 4) if pd.notna(adj_p) else "N/A",
            "Significant": sig,
        })

    st.dataframe(pd.DataFrame(posthoc_data))
    return significant_pairs


# Detect outliers

def detect_outliers(df, kpi, outlier_stdev, large_file_threshold=10000):
    try:
        # An outlier is an unusually extreme actual value (e.g. an abnormal
        # order). Zero rows (non-converting visitors, or non-order rows that
        # haven't been filtered out yet) are never outliers, and including
        # them would skew the mean/std or OLS fit used to find the real ones.
        # Bounds are computed on non-zero rows only; zero rows always stay
        # non-outliers in the returned mask.
        nonzero_df = df[df[kpi] != 0]
        outliers_mask = pd.Series([False] * len(df), index=df.index)

        if nonzero_df.empty:
            return outliers_mask, None, large_file_threshold

        if len(nonzero_df) > large_file_threshold:
            st.info(f"Dataset has {len(df):,} rows ({len(nonzero_df):,} non-zero).")
            for variant in nonzero_df['experience_variant_label'].unique():
                variant_data = nonzero_df[nonzero_df['experience_variant_label'] == variant][kpi].dropna()
                if not variant_data.empty:
                    # Calculate actual standard deviations instead of IQR
                    mean_val = variant_data.mean()
                    std_val = variant_data.std()
                    lower_bound = mean_val - (outlier_stdev * std_val)
                    upper_bound = mean_val + (outlier_stdev * std_val)

                    variant_outliers = (variant_data < lower_bound) | (variant_data > upper_bound)
                    outliers_mask[variant_data.index] = variant_outliers
            return outliers_mask, None, large_file_threshold
        else:
            st.info(f"Dataset has {len(df):,} rows ({len(nonzero_df):,} non-zero).")
            model = smf.ols(f'{kpi} ~ C(experience_variant_label)', data=nonzero_df).fit()
            influence = model.get_influence()
            standardized_residuals = influence.resid_studentized_internal
            leverage = influence.hat_matrix_diag
            dffits = influence.dffits[0]

            residual_threshold = outlier_stdev
            leverage_threshold = outlier_stdev * (model.df_model + 1) / len(nonzero_df)
            dffits_threshold = outlier_stdev * np.sqrt((model.df_model + 1) / len(nonzero_df))

            residuals_outliers = np.abs(standardized_residuals) > residual_threshold
            leverage_outliers = leverage > leverage_threshold
            dffits_outliers = np.abs(dffits) > dffits_threshold
            nonzero_outliers = residuals_outliers | leverage_outliers | dffits_outliers
            outliers_mask[nonzero_df.index] = nonzero_outliers
            return outliers_mask, model, large_file_threshold

    except Exception as e:
        st.error(f"Error during outlier detection: {e}")
        return pd.Series([False] * len(df), index=df.index), None, large_file_threshold

# Winsorize and IQR filter combined

def winsorize_data(df, kpi, method, outlier_stdev=None, percentile=None):
    df_copy = df.copy() # Create a copy to avoid modifying the original DataFrame
    lower_cap = None
    upper_cap = None
    cap_description = "No capping applied" # Default description

    # Bounds should reflect the actual value distribution among real
    # orders/spenders, not the zero mass (non-converting visitors, or
    # not-yet-filtered non-order rows) — otherwise a positive lower_cap would
    # also get applied to zero rows below, incorrectly raising them above zero.
    # Capping itself is likewise restricted to non-zero rows further down.
    nonzero_mask = df_copy[kpi] != 0
    nonzero_values = df_copy.loc[nonzero_mask, kpi]

    if nonzero_values.empty:
        st.warning(f"Warning: No non-zero {kpi} values to compute Winsorization bounds from. No capping applied.")
        return df_copy, None, None, "No non-zero values"

    if method == 'Standard Deviation':
        if outlier_stdev is None:
            outlier_stdev = 3 # Sensible default
            st.warning(f"Winsorization Standard Deviation not specified, defaulting to {outlier_stdev}.") # Inform user
        if nonzero_values.std() == 0:
             st.warning(f"Warning: Standard deviation of non-zero {kpi} values is zero. Cannot apply standard deviation Winsorization.")
             return df_copy, None, None, "Standard deviation is zero"

        mean = nonzero_values.mean()
        std_dev = nonzero_values.std()
        lower_cap = mean - (outlier_stdev * std_dev)
        upper_cap = mean + (outlier_stdev * std_dev)
        cap_description = f"{outlier_stdev} standard deviations from the mean (non-zero rows)"
        st.write(f"_Calculating Winsorization bounds based on: mean ± {outlier_stdev} * std_dev, non-zero rows_") # Add clarity

    elif method == 'Percentile':
        if percentile is None:
            percentile = 95 # Sensible default
            st.warning(f"Winsorization Percentile not specified, defaulting to {percentile}th.") # Inform user

        # Calculate the actual lower/upper percentile values (e.g., 95th -> 2.5th and 97.5th)
        lower_p = (100.0 - percentile) / 2.0
        upper_p = 100.0 - lower_p
        lower_cap = np.percentile(nonzero_values, lower_p) # Use np.percentile for robustness
        upper_cap = np.percentile(nonzero_values, upper_p) # Use np.percentile for robustness
        # Provide more specific description
        cap_description = f"the {lower_p:.1f}th and {upper_p:.1f}th percentiles (non-zero rows)"
        st.write(f"_Calculating Winsorization bounds based on percentiles: {lower_p:.1f} and {upper_p:.1f}, non-zero rows_") # Add clarity

    else:
        # This case should ideally not be reached if UI logic is correct
        st.error(f"Error: Unknown Winsorization method '{method}' provided.")
        return df_copy, None, None, "Unknown method" # Return original df

    # Apply capping using the determined lower and upper bounds
    # Ensure caps are not None before proceeding
    if lower_cap is not None and upper_cap is not None:
        capped = df_copy.loc[nonzero_mask, kpi]
        capped = np.where(capped < lower_cap, lower_cap, capped)
        capped = np.where(capped > upper_cap, upper_cap, capped)
        df_copy.loc[nonzero_mask, kpi] = capped
        st.write(f"_Applied capping between {lower_cap:.4f} and {upper_cap:.4f} (non-zero rows only)_")
    else:
        st.warning("Warning: Could not determine capping bounds. No Winsorization applied.")
        cap_description = "No capping applied (bounds indeterminate)"


    return df_copy, lower_cap, upper_cap, cap_description

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


def perform_stat_tests_and_conclusions(
    df, kpi, model_after, approach, unit="per_transaction",
    count_max_unique=50, count_max_unique_ratio=0.05,
    include_business_case=False, test_duration_days=None, visitor_counts=None,
    control_label=None,
):
    st.write("---") # Separator
    st.write("## Statistical Test Results")
    kpi_label = kpi.replace('_', ' ')
    unit_label = f"{kpi_label} per visitor" if unit == "per_visitor" else f"{kpi_label} per transaction"
    st.write(f"_Analysis unit: **{unit_label}**._")
    st.write("_(Based on Normality of Residuals and Homogeneity of Variance)_")

    # --- Input Validation ---
    if kpi not in df.columns:
        st.error(f"Error: KPI column '{kpi}' not found in DataFrame.")
        return
    if 'experience_variant_label' not in df.columns:
        st.error("Error: Column 'experience_variant_label' not found in DataFrame.")
        return
    if not hasattr(model_after, 'resid'):
         st.error("Error: The provided 'model' object does not have residuals ('model.resid'). Please provide a fitted statsmodels model.")
         return

    # Drop rows where KPI or group label is missing for group-based tests
    df_clean = df[[kpi, 'experience_variant_label']].dropna()
    if df_clean.empty:
        st.warning("Warning: No valid data remaining after dropping missing KPI or group labels.")
        return

    # Get unique groups and count
    unique_groups = df_clean['experience_variant_label'].unique()
    num_groups = len(unique_groups)

    if num_groups < 2:
        st.warning(f"Warning: Only {num_groups} group found. Cannot perform comparison tests.")
        return

    # --- Assumption Checks ---
    is_count = False

    if approach == "Heuristic (Auto-detect)":
        st.write("### 1. Assumption Checks")

        # --- Count-data gate: check BEFORE normality/Levene ---
        # Discrete, non-negative count metrics (e.g. items/tickets per buyer)
        # are structurally unsuited to ANOVA/Welch/KW's continuous-data
        # assumptions even when they "pass" normality by accident of large N,
        # and the binomial-style p(1-p) variance base breaks once the mean
        # exceeds 1. These route to Negative Binomial regression instead.
        is_count = is_count_kpi(
            df_clean[kpi],
            max_unique_for_check=count_max_unique,
            max_unique_ratio=count_max_unique_ratio,
        )

        if is_count:
            st.info("**Detected: discrete count metric.** Routing to Negative Binomial "
                    "regression instead of the continuous-data decision tree.")
            mean_val = df_clean[kpi].mean()
            var_val = df_clean[kpi].var()
            ratio = var_val / mean_val if mean_val > 0 else np.nan
            st.write(f"* Mean = {mean_val:.4f}, Variance = {var_val:.4f} "
                     f"(Variance/Mean ratio = {ratio:.2f})")
            if pd.notna(ratio) and ratio <= 1.05:
                st.caption("_Variance ≈ Mean: a Poisson model would likely also fit, "
                           "but Negative Binomial is used as the general-purpose default._")
            else:
                st.caption("_Variance > Mean: overdispersion present, consistent with "
                           "why Negative Binomial (rather than Poisson or a binomial "
                           "z-test) is the appropriate model here._")
            is_normal = False
            is_homogeneous = False
        else:
            # Normality of residuals via D'Agostino's K^2 omnibus test (skewness +
            # kurtosis). Unlike Shapiro-Wilk it is reliable for large samples and
            # emits no N > 5000 warning, so we use the full residual vector. Its
            # kurtosis component is only valid for N >= 20, so very small samples
            # fall back to non-parametric handling.
            try:
                resid_for_test = model_after.resid.dropna()
                st.write("**Normality of Residuals (D'Agostino's K² Test):**")
                if len(resid_for_test) >= 20:
                    norm_stat, norm_p_val = normaltest(resid_for_test)
                    st.write(f"* Statistic (K²) = {norm_stat:.4f}")
                    st.write(f"* p-value = {norm_p_val:.4f}")
                    is_normal = norm_p_val >= 0.05
                    st.write(f"* _Conclusion: Residuals are likely {'normally distributed' if is_normal else 'NOT normally distributed'}._")
                else:
                    is_normal = False
                    st.caption(
                        "Fewer than 20 residuals — too few for a reliable omnibus "
                        "normality test; defaulting to non-parametric handling."
                    )
            except Exception as e:
                st.error(f"Error during normality test: {e}")
                st.write("_Skipping further analysis due to normality test error._")
                return

            # Prepare groups for Levene's test (and potential non-parametric tests)
            # Use the cleaned data to avoid errors with tests
            groups = [group_data[kpi] for _, group_data in df_clean.groupby('experience_variant_label', observed=True)]

            # Homogeneity of Variance (Levene's Test)
            try:
                # Ensure there's data in each group being passed to Levene
                if any(len(g) < 1 for g in groups):
                    st.error("Error: At least one group has no data after cleaning. Cannot perform Levene's test.")
                    return
                # Levene's test requires at least 2 samples per group if center='median' (default)
                # or center='mean'. Check group sizes. Let's use default ('median').
                min_group_size = min(len(g) for g in groups)
                if min_group_size < 2 and num_groups > 1 :
                    st.warning(f"Warning: Levene's test might be unreliable as at least one group has only {min_group_size} sample(s).")

                levene_stat, levene_p_val = levene(*groups)
                st.write("**Homogeneity of Variance (Levene's Test):**")
                st.write(f"* Statistic = {levene_stat:.4f}")
                st.write(f"* p-value = {levene_p_val:.4f}")
                is_homogeneous = levene_p_val >= 0.05
                st.write(f"* _Conclusion: Variances are likely {'homogeneous (equal)' if is_homogeneous else 'NOT homogeneous (unequal)'}._")
            except Exception as e:
                st.error(f"Error during Levene's test: {e}")
                st.write("_Skipping further analysis due to variance test error._")
                return

    elif approach == "Gamma GLM (Best for Revenue)":
        # Skip diagnostics and go straight to GLM
        if unit == "per_visitor":
            st.info("Diagnostic tests (normality/Levene) skipped. Using a two-part hurdle model (Bernoulli × Gamma) that includes non-converting visitors (zeros).")
        else:
            st.info("Diagnostic tests (normality/Levene) skipped. Gamma GLM is robust to non-normal, skewed data.")
        is_normal = False
        is_homogeneous = False

    # --- Main Statistical Test ---
    st.write("### 2. Main Comparison Test")

    test_name = "Generic Comparison"
    p_value = np.nan
    test_statistic = np.nan
    effect_size = None
    posthoc_results = None
    significant_pairs = []
    is_significant = False # Initialize

    try:
        if approach == "Gamma GLM (Best for Revenue)":
            if unit == "per_visitor":
                st.info("Native two-part (hurdle) Likelihood Ratio Test")
            else:
                st.info("Native Gamma Likelihood Ratio Test")
            try:
                p_value, lr_stat = perform_gamma_test(df_clean, kpi, unit)

                if pd.isna(p_value):
                    st.warning("The likelihood model could not be fit (check that each variant has enough positive values).")
                else:
                    st.write(f"* Likelihood Ratio Statistic: {lr_stat:.4f}")
                    st.write(f"* p-value: {p_value:.4g}")

                    is_significant = p_value < 0.05
                    test_name = "Two-Part Hurdle MLE" if unit == "per_visitor" else "Native Gamma MLE"

                    if is_significant and num_groups > 2:
                        control_label = st.selectbox("Select Control Variant for Post-Hoc", df_clean['experience_variant_label'].unique())

                        significant_pairs = run_gamma_posthoc(
                            df=df_clean,
                            kpi=kpi,
                            group_col='experience_variant_label',
                            control_label=control_label,
                            unit=unit,
                        )

                    elif not is_significant and num_groups > 2:
                        st.info("Global test is not significant; pairwise comparisons are not required.")

            except Exception as e:
                st.error(f"Native Gamma Fit failed: {e}")
                p_value = np.nan

        elif approach == "Heuristic (Auto-detect)" and is_count:
            st.info("**Test Chosen:** Negative Binomial Regression (LRT)")
            st.markdown("_Reason: KPI is a discrete, non-negative count metric — Negative "
                        "Binomial models the mean-variance relationship natively (including "
                        "overdispersion) rather than assuming a continuous or bounded-proportion "
                        "distribution._")
            try:
                p_value, lr_stat, alpha_est, res_alt = perform_negbin_test(df_clean, kpi)

                if pd.isna(p_value):
                    st.warning("The Negative Binomial model could not be fit or compared "
                               "(check that there is more than one variant with data).")
                else:
                    test_statistic = lr_stat
                    test_name = "Negative Binomial Regression (LRT)"
                    st.write(f"* Likelihood Ratio Statistic: {lr_stat:.4f}")
                    st.write(f"* p-value: {p_value:.4g}")
                    st.write(f"* Estimated dispersion (α): {alpha_est:.4f}")

                    is_significant = p_value < 0.05

                    if is_significant and num_groups > 2:
                        control_label = st.selectbox(
                            "Select Control Variant for Post-Hoc",
                            df_clean['experience_variant_label'].unique(),
                        )
                        significant_pairs = run_negbin_posthoc(
                            df=df_clean,
                            kpi=kpi,
                            group_col='experience_variant_label',
                            control_label=control_label,
                        )
                    elif not is_significant and num_groups > 2:
                        st.info("Global test is not significant; pairwise comparisons are not required.")

            except Exception as e:
                st.error(f"Negative Binomial fit failed: {e}")
                p_value = np.nan

        elif approach == "Heuristic (Auto-detect)":
            if is_normal and is_homogeneous:
                # Standard ANOVA
                test_name = "Standard One-Way ANOVA"
                st.info(f"**Test Chosen:** {test_name}")
                st.markdown("_Reason: Residuals are normal and variances are homogeneous._")
                anova_results = sm.stats.anova_lm(model_after, typ=2)
                p_value = anova_results['PR(>F)'].iloc[0]
                test_statistic = anova_results['F'].iloc[0]

                st.dataframe(anova_results)
                is_significant = p_value < 0.05
                
                if is_significant and num_groups > 2:
                    st.info("**Post-Hoc Test (Tukey's HSD):**")
                    st.markdown("_Reason: ANOVA was significant, identifying which specific groups differ._")
                    tukey_results = pairwise_tukeyhsd(df_clean[kpi], df_clean['experience_variant_label'], alpha=0.05)
                    st.write(tukey_results.summary())
                    posthoc_results = tukey_results
                    tukey_groups = [str(g) for g in tukey_results.groupsunique]
                    tukey_combos = list(itertools.combinations(tukey_groups, 2))
                    tukey_pvals = getattr(tukey_results, "pvalues", [None] * len(tukey_combos))
                    for (a, b), reject, pv in zip(tukey_combos, tukey_results.reject, tukey_pvals):
                        if reject:
                            significant_pairs.append(_fmt_pair(a, b, pv))

            elif is_normal and not is_homogeneous: 
                # Normal, but Heterogeneous Variances (This is where Welch's belongs)
                test_name = "Welch's ANOVA"
                st.info(f"**Test Chosen:** {test_name}")
                st.markdown("_Reason: Residuals are normal, but variances are heterogeneous. Welch's is robust to unequal variances._")
                aov = welch_anova(data=df_clean, dv=kpi, between='experience_variant_label')
                p_value = aov['p-unc'].iloc[0]
                test_statistic = aov['F'].iloc[0]
                effect_size = aov['np2'].iloc[0] # Partial eta-squared from pingouin
                st.dataframe(aov)
                if effect_size is not None:
                    st.write(f"* _Effect Size (Partial Eta-Squared): {effect_size:.4f}_")
                is_significant = p_value < 0.05
                if is_significant and num_groups > 2:
                    st.info("**Post-Hoc Test (Games-Howell):**")
                    st.markdown("_Reason: Welch's ANOVA was significant, identifying which specific groups differ (suitable for unequal variances)._")
                    posthoc_results = pairwise_gameshowell(data=df_clean, dv=kpi, between='experience_variant_label')
                    st.dataframe(posthoc_results)
                    for _, gh_row in posthoc_results.iterrows():
                        if pd.notna(gh_row['pval']) and gh_row['pval'] < 0.05:
                            significant_pairs.append(_fmt_pair(gh_row['A'], gh_row['B'], gh_row['pval']))
    
            else: 
                # Non-Normal Data -> Drop to Non-parametric tests regardless of variance
                if num_groups > 2:
                    # Kruskal-Wallis
                    test_name = "Kruskal-Wallis H Test"
                    st.info(f"**Test Chosen:** {test_name}")
                    st.markdown("_Reason: Data is not normally distributed; non-parametric test suitable for 3+ groups._")
                    statistic, p_value = kruskal(*groups)
                    test_statistic = statistic
                    st.write(f"* H-statistic = {statistic:.4f}")
                    st.write(f"* p-value = {p_value:.4g}") # Use general format for potentially small p-values
                    is_significant = p_value < 0.05
                     # Calculate effect size: Eta-squared_H = (H - k + 1) / (n - k) where k=num groups, n=total samples
                    n_total = len(df_clean)
                    if n_total > num_groups: # Avoid division by zero or negative
                        effect_size_eta_h = (test_statistic - num_groups + 1) / (n_total - num_groups)
                        st.write(f"* _Approx. Effect Size (Eta-squared_H): {effect_size_eta_h:.4f}_")
    
                    if is_significant:
                        st.info("**Post-Hoc Test (Dunn's Test with Bonferroni correction):**")
                        st.markdown("_Reason: Kruskal-Wallis was significant, identifying which specific groups differ._")
    
                        # Dunn's test for post-hoc analysis                
                        try:
                            posthoc_results_df = sp.posthoc_dunn(groups, p_adjust='bonferroni')
                            group_names = df_clean['experience_variant_label'].unique()
                            name_map = {i+1: name for i, name in enumerate(group_names)}
                            posthoc_results_df.rename(columns=name_map, index=name_map, inplace=True)
                            
                            st.write("_(p-values adjusted using Bonferroni method)_")
                            st.dataframe(posthoc_results_df)
                            dunn_cols = list(posthoc_results_df.columns)
                            for i in range(len(dunn_cols)):
                                for j in range(i + 1, len(dunn_cols)):
                                    pv = posthoc_results_df.iloc[i, j]
                                    if pd.notna(pv) and pv < 0.05:
                                        significant_pairs.append(_fmt_pair(dunn_cols[i], dunn_cols[j], pv))
                        except Exception as posthoc_e:
                             st.error(f"Error during Dunn's post-hoc test: {posthoc_e}")
    
                else: # Exactly 2 groups with Heterogeneous Variances
                     # Mann-Whitney U
                    test_name = "Mann-Whitney U Test"
                    st.info(f"**Test Chosen:** {test_name}")
                    st.markdown("_Reason: Data is not normally distributed (Non-parametric alternative to ANOVA)._")
                    # Ensure groups have data
                    if len(groups[0]) > 0 and len(groups[1]) > 0:
                        statistic, p_value = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                        test_statistic = statistic
                        st.write(f"* U-statistic = {statistic:.4f}")
                        st.write(f"* p-value = {p_value:.4g}")
                        is_significant = p_value < 0.05
                        # Calculate effect size: Rank-Biserial Correlation r = 1 - (2*U) / (n1*n2)
                        n1 = len(groups[0])
                        n2 = len(groups[1])
                        effect_size_rank_biserial = 1 - (2 * test_statistic) / (n1 * n2)
                        st.write(f"* _Effect Size (Rank-Biserial Correlation): {effect_size_rank_biserial:.4f}_")
                    else:
                        st.warning("Warning: Cannot perform Mann-Whitney U test as at least one group has no data.")
                        p_value = np.nan # Ensure no significance is declared

    except Exception as e:
        st.error(f"An error occurred during the main statistical test ({test_name}): {e}")
        p_value = np.nan # Prevent incorrect conclusion

    # --- Summary Statistics ---
    st.write("### 3. Descriptive Statistics")
    st.caption(f"Means and standard deviations are computed per {unit_label.split(' per ')[-1]} (analysis unit: {unit_label}).")
    try:
        summary_stats = df_clean.groupby('experience_variant_label', observed=True)[kpi].agg(['mean', 'std', 'count'])
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
    except Exception as e:
        st.error(f"An error occurred during summary statistics calculation: {e}")
        summary_stats = None

    # --- Business Case (optional) ---
    if include_business_case and test_duration_days and summary_stats is not None:
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

        st.caption(f"_Control (baseline) variant: **{control_label}**._")

        if control_label not in summary_stats.index:
            st.error(
                f"Control variant '{control_label}' has no data after processing "
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
                    daily_visitors = len(df_clean) / test_duration_days
                    bc_unit = AnalysisUnit.PER_VISITOR
                else:
                    daily_visitors = sum(visitor_counts.values()) / test_duration_days
                    bc_unit = AnalysisUnit.PER_TRANSACTION

                group_stats = summary_stats.to_dict(orient="index")
                bc_results = ContinuousMetricEngine().run_business_case(
                    group_stats=group_stats,
                    control_label=control_label,
                    unit=bc_unit,
                    daily_visitors=daily_visitors,
                    visitor_counts=visitor_counts,
                    alternative=AlternativeHypothesis.TWO_SIDED,
                    projection_period=183,
                )

                for res in bc_results:
                    st.write(f"**{res['variant']}** vs **{control_label}**")
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

    # --- Conclusion ---
    st.write("### 4. Conclusion")

    if pd.isna(p_value):
        st.warning("Conclusion cannot be drawn due to errors in statistical testing.")
    elif is_significant:
        st.success(f"**A statistically significant difference was detected between the groups (p = {p_value:.4g}, using {test_name}).**")
        
        if num_groups > 2:
            if significant_pairs:
                st.write("**Significant pairwise differences** (after multiple-comparison correction):")
                for pair in significant_pairs:
                    st.markdown(f"* {pair}")
                st.markdown("_Refer to the summary statistics above to see which side of each pair is higher._")
            elif approach == "Gamma GLM (Best for Revenue)":
                st.write("The Likelihood Ratio Test indicates that variant means differ significantly. See the **Pairwise Post-Hoc Comparisons** above to identify which treatments outperformed the control.")
            elif is_count:
                st.write("The Likelihood Ratio Test indicates that variant means differ significantly. See the **Pairwise Post-Hoc Comparisons** above to identify which treatments outperformed the control.")
            elif posthoc_results is not None:
                st.write("The global test is significant, but no individual pair remained significant after multiple-comparison correction. See the post-hoc table above.")
            else:
                st.write("See the post-hoc test results above to determine which specific groups differ significantly.")
        
        elif num_groups == 2:
            st.write(f"The difference between the two groups ('{unique_groups[0]}' and '{unique_groups[1]}') is statistically significant.")
        
        # Effect Size Guidance
        if effect_size is not None:
            st.write(f"_Consider the effect size ({effect_size:.4f}) to evaluate the practical importance of this difference._")
        else:
            st.write("_Consider evaluating the relative lift to understand the practical importance of this difference._")

    else:
        st.info(f"**No statistically significant difference was detected between the groups (p = {p_value:.4g}, using {test_name}).**")
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
        control_label_input = None
        if include_business_case:
            test_duration_days = st.number_input(
                "Test duration so far (days):", min_value=1, value=14, step=1,
            )
            control_label_input = st.selectbox(
                "Select the control (baseline) variant for the business case:",
                df['experience_variant_label'].unique(),
                key="bc_control_label",
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
                    min_value=0.1, max_value=100.0, value=5.0, step=0.5,
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

        with st.spinner("Scanning for outliers…"):
            outliers_mask, initial_model, large_file_threshold = detect_outliers(df, kpi, outlier_stdev if method == 'Standard Deviation' else 5)  # Default 5 STD for detection purposes
        #st.write(f"Number of detected outliers: {outliers_mask.sum()}")
        if (len(df) >= large_file_threshold) and (outliers_mask.sum() > 0):
            st.warning(f"{outliers_mask.sum()} Outliers detected in a large dataset. The IQR method was used for outlier detection for efficient computation. You can adjust the outlier handling method in the options above.")
        elif (len(df) < large_file_threshold) and (outliers_mask.sum() > 0):
            st.warning(f"{outliers_mask.sum()} Outliers detected in a relatively small dataset. The OLS method was used for outlier detection. You can adjust the outlier handling method in the options above.")

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
                processed_df, lower_cap, upper_cap, cap_desc = winsorize_data(processed_df, kpi, method, outlier_stdev, percentile)

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
                processed_df = processed_df[~mask_aligned]
                st.write(f"Outliers removed: {int(mask_aligned.sum())} rows affected.")
            else:
                st.warning("No outlier handling applied.")

            action_bar.progress(45, text="Fitting model…")

            # --- Refit the model after outlier handling ---
            model_after = smf.ols(f'{kpi} ~ C(experience_variant_label)', data=processed_df).fit()

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
            
            if analysis_approach != "Gamma GLM (Best for Revenue)":
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
                processed_df, kpi, model_after, analysis_approach, unit,
                count_max_unique=count_max_unique,
                count_max_unique_ratio=count_max_unique_ratio / 100.0,
                include_business_case=include_business_case,
                test_duration_days=test_duration_days,
                visitor_counts=visitor_counts_input,
                control_label=control_label_input,
            )
            action_bar.progress(100, text="Analysis complete.")
            action_bar.empty()
            
if __name__ == "__main__":
    run()
