import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats
from scipy.stats import shapiro, levene, kruskal, mannwhitneyu
from scipy.optimize import minimize
from scipy.special import gammaln
import seaborn as sns
import matplotlib.pyplot as plt
from pingouin import welch_anova, qqplot, pairwise_gameshowell
import scikit_posthocs as sp
import streamlit as st

st.set_page_config(
    page_title="Continuous metric analysis",
    page_icon="🔢",
)

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

def perform_gamma_test(df, kpi):
    # 1. Null Model: Fit one Gamma to all data
    all_data = df[kpi].values
    _, _, ll_null = fit_gamma(all_data)
    
    # 2. Alternative Model: Fit Gamma to each variant separately
    ll_alt = 0
    variants = df['experience_variant_label'].unique()
    num_variants = len(variants)
    
    for v in variants:
        variant_data = df[df['experience_variant_label'] == v][kpi].values
        _, _, ll_v = fit_gamma(variant_data)
        ll_alt += ll_v
        
    # 3. Likelihood Ratio Test
    # Degrees of freedom = (params in Alt) - (params in Null)
    # Alt has 2 parameters (k, theta) per variant. Null has 2.
    df_diff = (2 * num_variants) - 2
    lr_stat = 2 * (ll_alt - ll_null)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=df_diff)
    
    return p_value, lr_stat

# Detect outliers

def detect_outliers(df, kpi, outlier_stdev, large_file_threshold=10000):
    try:
        if len(df) > large_file_threshold:
            st.info(f"Dataset has {len(df):,} rows.")
            outliers_mask = pd.Series([False] * len(df))
            for variant in df['experience_variant_label'].unique():
                variant_data = df[df['experience_variant_label'] == variant][kpi].dropna()
                if not variant_data.empty:
                    Q1 = variant_data.quantile(0.25)
                    Q3 = variant_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - outlier_stdev * IQR # or - 1.5 as standard convention
                    upper_bound = Q3 + outlier_stdev * IQR # or + 1.5 as standard convention
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
        return pd.Series([False] * len(df)), None

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

def perform_stat_tests_and_conclusions(df, kpi, model_after, approach):
    st.write("---") # Separator
    st.write("## Statistical Test Results")
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
    st.write("### 1. Assumption Checks")

    # Normality Test (Shapiro-Wilk on Residuals)
    try:
        shapiro_stat, shapiro_p_val = shapiro(model_after.resid)
        st.write("**Normality of Residuals (Shapiro-Wilk Test):**")
        st.write(f"* Statistic = {shapiro_stat:.4f}")
        st.write(f"* p-value = {shapiro_p_val:.4f}")
        is_normal = shapiro_p_val >= 0.05
        st.write(f"* _Conclusion: Residuals are likely {'normally distributed' if is_normal else 'NOT normally distributed'}._")
    except Exception as e:
        st.error(f"Error during Shapiro-Wilk test: {e}")
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

    # --- Main Statistical Test ---
    st.write("### 2. Main Comparison Test")

    test_name = "Generic Comparison"
    p_value = np.nan
    test_statistic = np.nan
    effect_size = None
    posthoc_results = None
    is_significant = False # Initialize

    try:
        if approach == "Gamma GLM (Best for Revenue/Items)":
            st.write("### Native Gamma Likelihood Ratio Test")
            try:
                # Fixed the function call name here to match your definition
                p_value, lr_stat = perform_gamma_test(df_clean, kpi)
                
                st.write(f"* Likelihood Ratio Statistic: {lr_stat:.4f}")
                st.write(f"* p-value: {p_value:.4g}")
                
                is_significant = p_value < 0.05
                test_name = "Native Gamma MLE"
                
            except Exception as e:
                st.error(f"Native Gamma Fit failed: {e}")
                p_value = np.nan

        elif approach == "Heuristic (Auto-detect)":
            if is_normal and is_homogeneous:
                # Standard ANOVA
                test_name = "Standard One-Way ANOVA"
                st.write(f"**Test Chosen:** {test_name}")
                st.markdown("_Reason: Residuals are normal and variances are homogeneous._")
                anova_results = sm.stats.anova_lm(model_after, typ=2)
                p_value = anova_results['PR(>F)'].iloc[0]
                test_statistic = anova_results['F'].iloc[0]

                st.dataframe(anova_results)
                is_significant = p_value < 0.05
                
                if is_significant and num_groups > 2:
                    st.write("**Post-Hoc Test (Tukey's HSD):**")
                    st.markdown("_Reason: ANOVA was significant, identifying which specific groups differ._")
                    tukey_results = pairwise_tukeyhsd(df_clean[kpi], df_clean['experience_variant_label'], alpha=0.05)
                    st.write(tukey_results.summary())
                    posthoc_results = tukey_results

            elif is_normal and not is_homogeneous: 
                # Normal, but Heterogeneous Variances (This is where Welch's belongs)
                test_name = "Welch's ANOVA"
                st.write(f"**Test Chosen:** {test_name}")
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
                    st.write("**Post-Hoc Test (Games-Howell):**")
                    st.markdown("_Reason: Welch's ANOVA was significant, identifying which specific groups differ (suitable for unequal variances)._")
                    posthoc_results = pairwise_gameshowell(data=df_clean, dv=kpi, between='experience_variant_label')
                    st.dataframe(posthoc_results)
    
            else: 
                # Non-Normal Data -> Drop to Non-parametric tests regardless of variance
                if num_groups > 2:
                    # Kruskal-Wallis
                    test_name = "Kruskal-Wallis H Test"
                    st.write(f"**Test Chosen:** {test_name}")
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
                        st.write("**Post-Hoc Test (Dunn's Test with Bonferroni correction):**")
                        st.markdown("_Reason: Kruskal-Wallis was significant, identifying which specific groups differ._")
    
                        # Dunn's test for post-hoc analysis                
                        try:
                            posthoc_results_df = sp.posthoc_dunn(groups, p_adjust='bonferroni')
                            group_names = df_clean['experience_variant_label'].unique()
                            name_map = {i+1: name for i, name in enumerate(group_names)}
                            posthoc_results_df.rename(columns=name_map, index=name_map, inplace=True)
                            
                            st.write("_(p-values adjusted using Bonferroni method)_")
                            st.dataframe(posthoc_results_df)
                        except Exception as posthoc_e:
                             st.error(f"Error during Dunn's post-hoc test: {posthoc_e}")
    
                else: # Exactly 2 groups with Heterogeneous Variances
                     # Mann-Whitney U
                    test_name = "Mann-Whitney U Test"
                    st.write(f"**Test Chosen:** {test_name}")
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
        if posthoc_results is not None and num_groups > 2:
             st.write("See post-hoc test results above to determine which specific groups differ significantly.")
        elif num_groups == 2:
             st.write(f"The difference between the two groups ('{unique_groups[0]}' and '{unique_groups[1]}') is statistically significant.")
        # Add advice on checking effect size
        if effect_size is not None:
             st.write(f"_Consider the effect size ({effect_size:.4f}) to evaluate the practical importance of this difference._")
        else:
             st.write("_Consider calculating and evaluating the effect size to understand the practical importance of this difference._")

    else:
        st.info(f"**No statistically significant difference was detected between the groups (p = {p_value:.4g}, using {test_name}).**")
        # Add advice on checking effect size even if not significant
        if effect_size is not None:
             st.write(f"_The observed differences are likely due to chance. The effect size ({effect_size:.4f}) may indicate the magnitude of any observed, non-significant difference._")
        else:
             st.write("_The observed differences are likely due to chance. Consider calculating the effect size if understanding the magnitude of non-significant differences is important._")
        st.markdown("_If a meaningful difference was expected, potential reasons include insufficient data (low power) or a truly small effect. Consider effect size and sample size._")

    st.write("---")

# Main Streamlit app
def run():
    st.title("Continuous Metric Analysis")
    """
    This calculator lets you analyze revenue data or the amount of items of ecommerce transactions (or leads) for your online experiments. See the example CSV file for what you need to upload. 
    You're not limited to just A and B, but can add more labels when applicable (C, D, etc.).

    The app will identify outliers, fit models, and perform statistical tests. Based on the test results and the output of the highest average and highest standard deviation, you can determine which variant won.

    How to use:
    1. Upload the CSV (download the example to see the column names)
    2. Select the KPI to analyze
    3. Select how to handle outliers (Winsorization, log transform or removal)
    4. Choose outlier handling method (percentile, standard deviation)
    5. Push the button!

    When choosing an outlier handling method:
    - Choose Winsorization to cap outlier values at a chosen threshold and not lose data points
    - Choose log transform when the data is heavily right-skewed to compress high values
    - Choose removal when there are very few, very extreme values that affect conclusions

    """
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
        df = pd.read_csv(uploaded_file)
        df, errors = preprocess_data(df)
        if errors:
            for error in errors:
                st.error(error)
            return

        st.write("### A random sample of your data:")
        st.write(df.sample(10))

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

        filter_zero_profit = False
        
        # Only show this checkbox if the user SELECTED profit AND the column actually has zeros
        if kpi == 'profit' and (df['profit'] == 0).any():
             filter_zero_profit = st.checkbox("Exclude rows where profit is zero (recommended for ANOVA)", value=True)
            
        # Select how to handle outliers
        outlier_handling = st.selectbox("Select how to handle outliers:", ['None', 'Winsorizing (STD/Percentile)', 'Log Transform', 'Removal'], help='Choose the method for handling outliers. "None" uses a default > 5 standard deviation definition for detection purposes. Only use Log Transform when your data does not contain negative numbers.')

        # Select analysis approach (Gamma model for revenue/items per transaction, heuristic framework for e.g. profit
        analysis_approach = st.selectbox(
            "Select Statistical Approach:",
            ["Heuristic (Auto-detect)", "Gamma GLM (Best for Revenue/Items)"],
            help="Heuristic follows a Normality/Variance decision tree. Gamma GLM is optimized for strictly positive, right-skewed continuous data."
        )

        method = None
        outlier_stdev = None
        percentile = None

        if outlier_handling not in ['None', 'Log Transform']:
            method = st.selectbox("Select outlier detection method:", ['Standard Deviation', 'Percentile'])
            if method == 'Standard Deviation':
                outlier_stdev = st.selectbox("How many standard deviations define an outlier?", [2, 3, 4, 5])
            elif method == 'Percentile':
                percentile = st.selectbox("Select percentile for Winsorization:", [90, 95, 99])

        outliers_mask, initial_model, large_file_threshold = detect_outliers(df, kpi, outlier_stdev if method == 'Standard Deviation' else 5)  # Default 5 STD for detection purposes
        #st.write(f"Number of detected outliers: {outliers_mask.sum()}")
        if (len(df) >= large_file_threshold) and (outliers_mask.sum() > 0):
            st.warning(f"{outliers_mask.sum()} Outliers detected in a large dataset. The IQR method was used for outlier detection for efficient computation. You can adjust the outlier handling method in the options above.")
        elif (len(df) < large_file_threshold) and (outliers_mask.sum() > 0):
            st.warning(f"{outliers_mask.sum()} Outliers detected in a relatively small dataset. The OLS method was used for outlier detection. You can adjust the outlier handling method in the options above.")

        # Show raw data plots before any processing
        data_mean = df[kpi].mean()
        data_std = df[kpi].std()
        raw_min = max(0, data_mean - 3.5 * data_std) 
        raw_max = data_mean + 3.5 * data_std
        num_variants = len(df['experience_variant_label'].unique())

        st.write("### Raw Data Box Plot")
        fig_box, ax_box = plt.subplots()
        sns.boxplot(x='experience_variant_label', y=kpi, data=df, palette=sns.color_palette("hls", num_variants), hue='experience_variant_label', legend=False)
        ax_box.set_ylim(raw_min, raw_max)
        st.pyplot(fig_box)
        plt.clf()

        st.write("### Raw Data Histogram with KDE")
        fig_hist, ax_hist = plt.subplots()
        sns.histplot(df[kpi], kde=True, bins=30, ax=ax_hist)
        ax_hist.set_xlim(raw_min, raw_max)
        plt.title("Raw Data Histogram with KDE")
        st.pyplot(fig_hist)
        plt.clf()


        if st.button("Calculate my test results", type="primary"):
            processed_df = df.copy()
            
            if kpi == 'profit' and filter_zero_profit:
                initial_rows = len(processed_df)
                processed_df = processed_df[processed_df['profit'] > 0]
                rows_removed = initial_rows - len(processed_df)
                st.info(f"Filtered out {rows_removed} rows where profit was zero.")

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
                st.write("Log transformation applied.")
            elif outlier_handling == 'Removal':
                processed_df = processed_df[~outliers_mask]
                st.write(f"Outliers removed: {outliers_mask.sum()} rows affected.")
            else:
                st.write("No outlier handling applied.")

            # --- Refit the model after outlier handling ---
            model_after = smf.ols(f'{kpi} ~ C(experience_variant_label)', data=processed_df).fit()

            # --- Processed Data Plots (using the processed data) ---
            data_mean = processed_df[kpi].mean()
            data_std = processed_df[kpi].std()
            processed_min = max(0, data_mean - 3.5 * data_std)
            processed_max = data_mean + 3.5 * data_std
            num_variants = len(processed_df['experience_variant_label'].unique())
            
            st.write("### Refitted Data Box Plot")
            fig_box, ax_box = plt.subplots()
            sns.boxplot(x='experience_variant_label', y=kpi, data=processed_df, palette=sns.color_palette("hls", num_variants), hue='experience_variant_label', legend=False)
            ax_box.set_ylim(processed_min, processed_max)
            st.pyplot(fig_box)
            plt.clf()

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
            resid_std = model_after.resid.std()
            resid_min = -3.5 * resid_std
            resid_max = 3.5 * resid_std
            fig_hist, ax_hist = plt.subplots()
            sns.histplot(model_after.resid, kde=True, bins=30, ax=ax_hist) # Use residuals from the new model
            ax_hist.set_xlim(resid_min, resid_max)
            plt.title("Histogram of Residuals with KDE")
            st.pyplot(fig_hist)
            plt.clf()

            perform_stat_tests_and_conclusions(processed_df, kpi, model_after, analysis_approach)
            
if __name__ == "__main__":
    run()
