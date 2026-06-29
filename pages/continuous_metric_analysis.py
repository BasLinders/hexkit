import pandas as pd
import numpy as np
import statsmodels.api as sm
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

st.set_page_config(
    page_title="Continuous metric analysis",
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
        * **Normality Check:** Uses D'Agostino's K² omnibus test (skewness + kurtosis) to see if residuals follow a normal distribution.
        * **Homogeneity Check:** Uses Levene’s test to verify if groups have equal variances.
        * **The Branches:**
            * *Standard ANOVA:* Used when data is normal and variances are equal.
            * *Welch's ANOVA:* Used when data is normal but variances are unequal.
            * *Kruskal-Wallis / Mann-Whitney:* Non-parametric alternatives used when data is non-normal.

        ### 2. The Gamma Model (MLE)
        Specifically designed for continuous, strictly positive, and right-skewed data (like **Revenue** or **Items per transaction**). 
        * It fits a Gamma distribution to each variant using **Maximum Likelihood Estimation (MLE)**.
        * The probability density function used is: $$f(x; k, \theta) = \frac{x^{k-1} e^{-x/\theta}}{\theta^k \Gamma(k)}$$
        * It compares models using a **Likelihood Ratio Test (LRT)** to determine if the differences in means are statistically significant.
        """)

    # --- Expander 2: Analysis unit (per visitor vs per transaction) ---
    with st.expander("Analysis unit: per visitor vs per transaction"):
        st.markdown(r"""
        Before testing, choose **what one row represents** for the comparison. This must
        match how you plan the experiment (and how you size it in the pre-test tool).

        ### Revenue per transaction (positive rows only)
        * The unit is a **transaction**; the metric is e.g. average order value.
        * Rows with a value of **0 are excluded** — they aren't orders.
        * Gamma path: a single Gamma per variant (strictly positive data).
        * Answers *"did the value of an order change?"* It does **not** capture a change
          in how many people buy, and carries a mild selection effect (the set of orders
          is itself influenced by the treatment).

        ### Revenue per visitor (includes zero rows)
        * The unit is a **visitor**; non-buyers count as **0**.
        * **All rows are kept**, including the zeros.
        * This is usually the real business outcome, and the unit you randomise on, so it
          captures both *more people buying* and *buyers spending more*.
        * Gamma path: a **two-part (hurdle) model** — a Bernoulli component for the
          probability of converting, multiplied by a Gamma component for the spend of
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
        * Use **Heuristic (Auto-detect)** for metrics that can contain negative values (i.e. profit).
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

        ### 3. Post-Hoc Analysis
        If you have more than two groups and the main test is significant, look at the **Post-Hoc** table.
        * It performs pairwise comparisons (Group A vs Group B, B vs C, etc.).
        * It uses corrections (like **Tukey**, **Games-Howell**, or **Bonferroni**) to prevent "p-hacking" or false positives that occur when running multiple simultaneous tests.
        
        ### 4. Gamma GLM Specifics
        When using the Gamma approach, the output focuses on **Model Fit** rather than variance:
        
        * **Log-Likelihood:** This represents how well the Gamma distribution fits your data. Higher (less negative) values indicate a better fit.
        * **Deviance:** A measure of the error in the model. When comparing variants, we look for a significant reduction in deviance.
        * **Likelihood Ratio Test (LRT):** This is the p-value source for Gamma. It tests the "Null Model" (one mean for all variants) against the "Alternative Model" (separate means for each variant). 
            * If **p < 0.05**, we conclude that at least one variant's mean is statistically different from the others under the Gamma assumption.
            * For **per-visitor** analysis the model is the two-part hurdle (Bernoulli × Gamma), so each variant carries one extra parameter (the conversion rate) and the test has correspondingly more degrees of freedom.
        * **Scale Parameter ($\theta$):** Indicates the "spread" or "dispersion" of the revenue. High scale parameters often point to high-variance "Whale" customers in revenue data.
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

    # Check for missing values
    if df['experience_variant_label'].isnull().any():
        errors.append("'experience_variant_label' contains null values.")
    if 'total_item_quantity' in df and df['total_item_quantity'].isnull().any():
        errors.append("'total_item_quantity' contains null values.")
    if 'purchase_revenue' in df and df['purchase_revenue'].isnull().any():
        errors.append("'purchase_revenue' contains null values.")
    if 'profit' in df and df['profit'].isnull().any():
        errors.append("'profit' contains null values.")

    # Ensure categorical variable
    df['experience_variant_label'] = pd.Categorical(df['experience_variant_label'])

    # Convert to numeric
    if 'total_item_quantity' in df:
        df['total_item_quantity'] = pd.to_numeric(df['total_item_quantity'], errors='coerce')
    if 'purchase_revenue' in df:
        df['purchase_revenue'] = pd.to_numeric(df['purchase_revenue'], errors='coerce')
    if 'profit' in df:
        df['profit'] = pd.to_numeric(df['profit'], errors='coerce')

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

# Detect outliers

def detect_outliers(df, kpi, outlier_stdev, large_file_threshold=10000):
    try:
        if len(df) > large_file_threshold:
            st.info(f"Dataset has {len(df):,} rows.")
            outliers_mask = pd.Series([False] * len(df))
            for variant in df['experience_variant_label'].unique():
                variant_data = df[df['experience_variant_label'] == variant][kpi].dropna()
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
            st.info(f"Dataset has {len(df):,} rows.")
            model = smf.ols(f'{kpi} ~ C(experience_variant_label)', data=df).fit()
            influence = model.get_influence()
            standardized_residuals = influence.resid_studentized_internal
            leverage = influence.hat_matrix_diag
            dffits = influence.dffits[0]

            residual_threshold = outlier_stdev
            leverage_threshold = outlier_stdev * (model.df_model + 1) / len(df)
            dffits_threshold = outlier_stdev * np.sqrt((model.df_model + 1) / len(df))

            residuals_outliers = np.abs(standardized_residuals) > residual_threshold
            leverage_outliers = leverage > leverage_threshold
            dffits_outliers = np.abs(dffits) > dffits_threshold
            outliers_mask = residuals_outliers | leverage_outliers | dffits_outliers
            return outliers_mask, model, large_file_threshold

    except Exception as e:
        st.error(f"Error during outlier detection: {e}")
        return pd.Series([False] * len(df)), None, large_file_threshold

# Winsorize and IQR filter combined

def winsorize_data(df, kpi, method, outlier_stdev=None, percentile=None):
    df_copy = df.copy() # Create a copy to avoid modifying the original DataFrame
    lower_cap = None
    upper_cap = None
    cap_description = "No capping applied" # Default description

    if method == 'Standard Deviation':
        if outlier_stdev is None:
            outlier_stdev = 3 # Sensible default
            st.warning(f"Winsorization Standard Deviation not specified, defaulting to {outlier_stdev}.") # Inform user
        if df_copy[kpi].std() == 0:
             st.warning(f"Warning: Standard deviation of {kpi} is zero. Cannot apply standard deviation Winsorization.")
             return df_copy, None, None, "Standard deviation is zero"

        mean = df_copy[kpi].mean()
        std_dev = df_copy[kpi].std()
        lower_cap = mean - (outlier_stdev * std_dev)
        upper_cap = mean + (outlier_stdev * std_dev)
        cap_description = f"{outlier_stdev} standard deviations from the mean"
        st.write(f"_Calculating Winsorization bounds based on: mean ± {outlier_stdev} * std_dev_") # Add clarity

    elif method == 'Percentile':
        if percentile is None:
            percentile = 95 # Sensible default
            st.warning(f"Winsorization Percentile not specified, defaulting to {percentile}th.") # Inform user

        # Calculate the actual lower/upper percentile values (e.g., 95th -> 2.5th and 97.5th)
        lower_p = (100.0 - percentile) / 2.0
        upper_p = 100.0 - lower_p
        lower_cap = np.percentile(df_copy[kpi].dropna(), lower_p) # Use np.percentile for robustness
        upper_cap = np.percentile(df_copy[kpi].dropna(), upper_p) # Use np.percentile for robustness
        # Provide more specific description
        cap_description = f"the {lower_p:.1f}th and {upper_p:.1f}th percentiles"
        st.write(f"_Calculating Winsorization bounds based on percentiles: {lower_p:.1f} and {upper_p:.1f}_") # Add clarity

    else:
        # This case should ideally not be reached if UI logic is correct
        st.error(f"Error: Unknown Winsorization method '{method}' provided.")
        return df_copy, None, None, "Unknown method" # Return original df

    # Apply capping using the determined lower and upper bounds
    # Ensure caps are not None before proceeding
    if lower_cap is not None and upper_cap is not None:
        df_copy[kpi] = np.where(df_copy[kpi] < lower_cap, lower_cap, df_copy[kpi])
        df_copy[kpi] = np.where(df_copy[kpi] > upper_cap, upper_cap, df_copy[kpi])
        st.write(f"_Applied capping between {lower_cap:.4f} and {upper_cap:.4f}_")
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

def perform_stat_tests_and_conclusions(df, kpi, model_after, approach, unit="per_transaction"):
    st.write("---") # Separator
    st.write("## Statistical Test Results")
    unit_label = "revenue per visitor" if unit == "per_visitor" else "revenue per transaction"
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

    if approach == "Heuristic (Auto-detect)":
        st.write("### 1. Assumption Checks")
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

    elif approach == "Gamma GLM (Best for Revenue/Items)":
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
        if approach == "Gamma GLM (Best for Revenue/Items)":
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
            elif approach == "Gamma GLM (Best for Revenue/Items)":
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

@st.cache_data(show_spinner=False)
def _read_csv_cached(file_contents):
    """Cached raw CSV read (the expensive step). Preprocessing is run separately
    in run() so the progress bar can report reading and cleaning as two phases."""
    return pd.read_csv(file_contents)


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
    st.title("Continuous Metric Analysis")
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
        unit_choice = st.selectbox(
            "Select the analysis unit:",
            [
                "Revenue per visitor (includes zero-value rows)",
                "Revenue per transaction (positive rows only)",
            ],
            help=(
                "Per visitor averages over every visitor, counting non-buyers as 0 "
                "(captures conversion-rate and spend effects together; needs zero rows "
                "in your file). Per transaction looks at order value among buyers only."
            ),
        )
        unit = "per_visitor" if unit_choice.startswith("Revenue per visitor") else "per_transaction"

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

        # Select analysis approach (Gamma model for revenue/items per transaction, heuristic framework for e.g. profit
        analysis_approach = st.selectbox(
            "Select Statistical Approach:",
            ["Heuristic (Auto-detect)", "Gamma GLM (Best for Revenue/Items)"],
            help="Heuristic follows a Normality/Variance decision tree. Gamma GLM is optimized for strictly positive, right-skewed continuous data. For per-visitor analysis, Gamma GLM uses a two-part hurdle model so zeros are included."
        )

        # Guard: the Gamma family needs non-negative data.
        if analysis_approach == "Gamma GLM (Best for Revenue/Items)" and n_neg > 0:
            st.warning(
                f"The Gamma model requires non-negative values, but '{kpi}' contains "
                f"{n_neg:,} negative value(s). Use the Heuristic approach for this metric, "
                "or pick a non-negative KPI."
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
            
            if analysis_approach != "Gamma GLM (Best for Revenue/Items)":
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
            perform_stat_tests_and_conclusions(processed_df, kpi, model_after, analysis_approach, unit)
            action_bar.progress(100, text="Analysis complete.")
            action_bar.empty()
            
if __name__ == "__main__":
    run()
