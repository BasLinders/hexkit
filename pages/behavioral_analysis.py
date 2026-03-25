import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Behavioral Metric Analysis",
    page_icon="🔢"
)

# -------------------------------------------------------------------------
# 1. PREPROCESSING
# -------------------------------------------------------------------------
def preprocess_data(df):
    errors = []

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Rename columns based on keywords if exact match missing
    # We expect: 'experience_variant_label' and 'visitor_metric'
    for col in df.columns:
        if 'variant' in col:
            df.rename(columns={col: 'experience_variant_label'}, inplace=True)
        elif 'metric' in col or 'value' in col:
             df.rename(columns={col: 'visitor_metric'}, inplace=True)

    # Hard validation
    required_cols = ['experience_variant_label', 'visitor_metric']
    for req in required_cols:
        if req not in df.columns:
            errors.append(f"Missing required column: '{req}'. Please check your CSV.")
    
    if errors:
        return df, errors

    # Check for nulls
    if df[required_cols].isnull().any().any():
        errors.append("Dataset contains null values in required columns.")

    # Ensure types
    df['experience_variant_label'] = df['experience_variant_label'].astype(str)
    df['visitor_metric'] = pd.to_numeric(df['visitor_metric'], errors='coerce')
    
    # Drop NaNs created by coercion
    if df['visitor_metric'].isnull().any():
        df = df.dropna(subset=['visitor_metric'])
        st.warning("Dropped rows with non-numeric values in 'visitor_metric'.")

    return df, errors

# -------------------------------------------------------------------------
# 2. OUTLIER DETECTION
# -------------------------------------------------------------------------
def detect_outliers(df, kpi, outlier_stdev, large_file_threshold=10000):
    try:
        # Method A: IQR for Large Files (Faster)
        if len(df) > large_file_threshold:
            st.info(f"Dataset is large ({len(df):,} rows). Using vectorized IQR method for outlier detection.")
            
            q1 = df.groupby('experience_variant_label')[kpi].transform(lambda x: x.quantile(0.25))
            q3 = df.groupby('experience_variant_label')[kpi].transform(lambda x: x.quantile(0.75))
            iqr = q3 - q1
            
            lower_bound = q1 - (outlier_stdev * iqr)
            upper_bound = q3 + (outlier_stdev * iqr)
            
            mask = (df[kpi] < lower_bound) | (df[kpi] > upper_bound)
            return mask.fillna(False)

        # Method B: OLS Influence for Smaller Files (More Precise)
        else:
            st.info(f"Dataset is small ({len(df):,} rows). Using OLS Influence method for detection.")
            # Escape column name for statsmodels formula if it has spaces (though we cleaned it)
            formula = f"{kpi} ~ C(experience_variant_label)"
            model = smf.ols(formula, data=df).fit()
            
            influence = model.get_influence()
            standardized_residuals = influence.resid_studentized_internal
            
            # Simple residual threshold based on user input
            residuals_outliers = np.abs(standardized_residuals) > outlier_stdev
            return pd.Series(residuals_outliers, index=df.index)

    except Exception as e:
        st.error(f"Error during outlier detection: {e}")
        return pd.Series([False] * len(df), index=df.index)

# -------------------------------------------------------------------------
# 3. OUTLIER HANDLING (Winsorization / Log)
# -------------------------------------------------------------------------
def winsorize_data(df, kpi, method, outlier_stdev=None, percentile=None):
    df_copy = df.copy()
    lower_cap, upper_cap = None, None
    cap_desc = ""

    if method == 'Standard Deviation':
        if outlier_stdev is None: outlier_stdev = 3
        mean = df_copy[kpi].mean()
        std_dev = df_copy[kpi].std()
        lower_cap = mean - (outlier_stdev * std_dev)
        upper_cap = mean + (outlier_stdev * std_dev)
        cap_desc = f"Mean ± {outlier_stdev} SD"

    elif method == 'Percentile':
        if percentile is None: percentile = 95
        lower_p = (100.0 - percentile) / 2.0
        upper_p = 100.0 - lower_p
        lower_cap = np.percentile(df_copy[kpi], lower_p)
        upper_cap = np.percentile(df_copy[kpi], upper_p)
        cap_desc = f"{lower_p:.1f}th and {upper_p:.1f}th Percentiles"

    # Apply Caps
    if lower_cap is not None and upper_cap is not None:
        # Ensure we don't cap below 0 for behavioral data (metrics usually >= 0)
        lower_cap = max(0, lower_cap) 
        df_copy[kpi] = np.clip(df_copy[kpi], lower_cap, upper_cap)
    
    return df_copy, lower_cap, upper_cap, cap_desc

def log_transform_data(df, kpi):
    df_copy = df.copy()
    # log1p is crucial for behavioral data which often contains 0s
    df_copy[kpi] = np.log1p(df_copy[kpi])
    return df_copy

# -------------------------------------------------------------------------
# 4. STATISTICAL ENGINE (Welch's T-Test)
# -------------------------------------------------------------------------
def perform_welch_test(df, kpi):
    st.write("---")
    st.header("Test Results (Welch's t-test)")
    st.markdown("""
    Perform statistical inference with Welch's t-test because behavioral data often has unequal variances 
    (e.g., one group might behave erratically while the other is stable).
    """)

    # 1. Validation
    variants = df['experience_variant_label'].unique()
    if len(variants) != 2:
        st.error(f"Error: Welch's t-test requires exactly 2 variants. Your file has {len(variants)}: {list(variants)}.")
        return

    group_a_label = variants[0]
    group_b_label = variants[1]

    # 2. Preparation
    data_a = df[df['experience_variant_label'] == group_a_label][kpi]
    data_b = df[df['experience_variant_label'] == group_b_label][kpi]
    
    # 3. Calculation
    # equal_var=False triggers Welch's t-test
    t_stat, p_val = stats.ttest_ind(data_b, data_a, equal_var=False)
    
    # Calculate Means & Lift
    mean_a = data_a.mean()
    mean_b = data_b.mean()
    lift = (mean_b - mean_a) / mean_a if mean_a != 0 else 0
    
    # 4. Display Summary Stats
    st.subheader("1. Descriptive Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"Mean: {group_a_label}", value=f"{mean_a:.2f}")
        st.caption(f"Sample Size: {len(data_a):,}")
    with col2:
        st.metric(label=f"Mean: {group_b_label}", value=f"{mean_b:.2f}", delta=f"{lift:.2%}")
        st.caption(f"Sample Size: {len(data_b):,}")

    # 5. Display Significance
    st.subheader("2. Statistical Conclusion")
    
    # Interpret P-Value
    is_sig = p_val < 0.05
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.markdown(f"**P-Value:** `{p_val:.4f}`")
        if is_sig:
            st.success("Significant Difference")
        else:
            st.warning("No Significant Difference")
            
    with res_col2:
        if is_sig:
            if mean_b > mean_a:
                st.write(f"**{group_b_label}** performs significantly better than **{group_a_label}**.")
            else:
                st.write(f"**{group_b_label}** performs significantly worse than **{group_a_label}**.")
        else:
            st.write(f"We cannot distinguish between **{group_a_label}** and **{group_b_label}** with statistical confidence. The observed difference may be due to noise.")

    st.divider()

# -------------------------------------------------------------------------
# 5. MAIN APP
# -------------------------------------------------------------------------
def run():
    st.title("Behavioral Metric Analysis")
    st.markdown("""
    This tool analyzes visitor-level behavioral metrics (e.g., **pages per visitor**, **time on site**, **clicks per visitor**).
    
    **Assumptions:**
    * Input data is aggregated (1 row = 1 visitor).
    * We use **Welch's t-test** (robust against unequal variance and skewed data).
    """)

    # --- DOWNLOAD TEMPLATE ---
    with st.expander("Need a CSV Template?"):
        st.download_button(
            label="Download Template CSV",
            data=pd.DataFrame({
                "experience_variant_label": ["A", "B", "B", "A", "B"],
                "visitor_metric": [8, 10, 0, 19, 42]
            }).to_csv(index=False),
            file_name="behavioral_template.csv",
            mime="text/csv"
        )

    # --- FILE UPLOAD ---
    uploaded_file = st.file_uploader("Upload your visitor-level data (CSV)", type="csv")
    
    if uploaded_file is not None:
        # Load
        df_raw = pd.read_csv(uploaded_file)
        
        # Preprocess
        df, errors = preprocess_data(df_raw)
        if errors:
            for e in errors: st.error(e)
            return

        # Preview
        with st.expander("Preview Data"):
            st.dataframe(df.head())
            st.caption(f"Total Rows: {len(df):,}")

        # Config Panel
        st.markdown("### Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
             outlier_handling = st.selectbox(
                "Outlier Handling Method",
                ['None', 'Winsorizing (Cap Values)', 'Log Transform', 'Removal'],
                help="Behavioral data often has 'whales' (bots/power users). Winsorizing or Log Transform is recommended."
            )
        
        # Dynamic Options based on selection
        method, outlier_stdev, percentile = None, None, None
        
        with col2:
            if outlier_handling == 'Winsorizing (Cap Values)':
                method = st.selectbox("Capping Logic", ['Percentile', 'Standard Deviation'])
                if method == 'Percentile':
                    percentile = st.selectbox("Percentile Cutoff", [95, 99, 99.5], index=1)
                else:
                    outlier_stdev = st.number_input("Standard Deviations", value=3, min_value=1)
            elif outlier_handling in ['Removal', 'None']:
                 # Only needed for the detection algorithm display
                 outlier_stdev = 5 

        # --- OUTLIER DETECTION VISUALIZATION ---
        kpi = 'visitor_metric'
        
        outliers_mask = detect_outliers(df, kpi, outlier_stdev if outlier_stdev else 5)
        n_outliers = outliers_mask.sum()
        
        if n_outliers > 0:
            st.info(f"**Data Quality Check:** Detected {n_outliers:,} potential outliers in the raw data.")

        # --- VISUALIZATION (PRE-CALC) ---
        st.subheader("Data Distribution")
        viz_col1, viz_col2 = st.columns(2)
        data_mean = df[kpi].mean()
        data_std = df[kpi].std()
        range_min = max(0, data_mean - 3.5 * data_std)
        range_max = data_mean + 3.5 * data_std
        
        with viz_col1:
            st.caption("Boxplot (Visualizing Skew)")
            fig_box, ax_box = plt.subplots()
            sns.boxplot(x='experience_variant_label', y=kpi, data=df, ax=ax_box, palette=sns.color_palette("hls", 8))
            ax_box.set_ylim(range_min, range_max)
            st.pyplot(fig_box)
            
        with viz_col2:
            st.caption("Distribution (Histogram + KDE)")
            fig_hist, ax_hist = plt.subplots()
            #sns.kdeplot(data=df, x=kpi, hue='experience_variant_label', fill=True, ax=ax_hist)
            sns.histplot(data=df, x=kpi, hue='experience_variant_label', kde=True, ax=ax_hist, element="step", alpha=0.3)
            ax_hist.set_xlim(range_min, range_max)
            st.pyplot(fig_hist)

        # --- EXECUTION BUTTON ---
        if st.button("Run Analysis", type="primary"):
            
            processed_df = df.copy()

            # Apply Outlier Logic
            if outlier_handling == 'Winsorizing (Cap Values)':
                processed_df, low, high, desc = winsorize_data(processed_df, kpi, method, outlier_stdev, percentile)
                st.success(f"Data Winsorized: Values capped at {high:.2f} ({desc})")
            
            elif outlier_handling == 'Log Transform':
                processed_df = log_transform_data(processed_df, kpi)
                st.success("Data Log-Transformed (ln(x+1)) to normalize skew.")
            
            elif outlier_handling == 'Removal':
                processed_df = processed_df[~outliers_mask]
                st.warning(f"Removed {n_outliers} outlier rows.")

            # --- RUN STATS ---
            perform_welch_test(processed_df, kpi)

if __name__ == "__main__":
    run()
