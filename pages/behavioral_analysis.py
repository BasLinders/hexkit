import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from foe.behavioral.operations import BehavioralEngine

# Outlier detection, winsorizing / log transforms and Welch's t-test all run in
# the shared FOE BehavioralEngine (foe.behavioral.operations). This page only
# handles CSV preprocessing, visualisation and presentation.
_engine = BehavioralEngine()
LARGE_FILE_THRESHOLD = 10000
WELCH_ALPHA = 0.05

# -------------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Behavioral Metric Analysis",
    page_icon="🔢",
    layout="wide",
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
def detect_outliers(df, kpi, outlier_stdev, large_file_threshold=LARGE_FILE_THRESHOLD):
    if len(df) > large_file_threshold:
        st.info(f"Dataset is large ({len(df):,} rows). Using vectorized IQR method for outlier detection.")
    else:
        st.info(f"Dataset is small ({len(df):,} rows). Using OLS Influence method for detection.")

    try:
        mask = BehavioralEngine.detect_outliers_mask(
            df, kpi, threshold=outlier_stdev, large_file_limit=large_file_threshold
        )
        return pd.Series(mask, index=df.index, dtype=bool)

    except Exception as e:
        st.error(f"Error during outlier detection: {e}")
        return pd.Series([False] * len(df), index=df.index)

# -------------------------------------------------------------------------
# 3. OUTLIER HANDLING (Winsorization / Log)
# -------------------------------------------------------------------------
def winsorize_data(df, kpi, method, outlier_stdev=None, percentile=None):
    if method == 'Standard Deviation':
        if outlier_stdev is None: outlier_stdev = 3
        params = {"logic": "Standard Deviation", "stdev": outlier_stdev}
        cap_desc = f"Mean ± {outlier_stdev} SD"
    else:  # 'Percentile'
        if percentile is None: percentile = 95
        params = {"logic": "Percentile", "percentile": percentile}
        lower_p = (100.0 - percentile) / 2.0
        upper_p = 100.0 - lower_p
        cap_desc = f"{lower_p:.1f}th and {upper_p:.1f}th Percentiles"

    # FOE floors the lower cap at 0, since behavioral metrics are >= 0
    df_copy, meta = BehavioralEngine.apply_transformations(df, kpi, "Winsorizing", params)
    return df_copy, meta["lower_cap"], meta["upper_cap"], cap_desc

def log_transform_data(df, kpi):
    # FOE uses log1p, which safely handles the 0s common in behavioral data
    df_copy, _ = BehavioralEngine.apply_transformations(df, kpi, "Log Transform", {})
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

    # 1. Validation (kept here so the error can list the variants found)
    variants = df['experience_variant_label'].unique()
    if len(variants) != 2:
        st.error(f"Error: Welch's t-test requires exactly 2 variants. Your file has {len(variants)}: {list(variants)}.")
        return

    # 2. Calculation via FOE. The first variant in the file is the control,
    # matching this page's previous behaviour.
    result = _engine.run_welch_inference(df, kpi, control_label=variants[0], alpha=WELCH_ALPHA)
    if "error" in result:
        st.error(result["error"])
        return

    group_a_label = result["group_control"]["label"]
    group_b_label = result["group_challenger"]["label"]
    mean_a = result["group_control"]["mean"]
    mean_b = result["group_challenger"]["mean"]
    n_a = result["group_control"]["n"]
    n_b = result["group_challenger"]["n"]
    p_val = result["p_value"]
    lift = result["lift"]
    is_sig = result["is_significant"]

    # 4. Display Summary Stats
    st.subheader("1. Descriptive Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"Mean: {group_a_label}", value=f"{mean_a:.2f}")
        st.caption(f"Sample Size: {n_a:,}")
    with col2:
        st.metric(label=f"Mean: {group_b_label}", value=f"{mean_b:.2f}", delta=f"{lift:.2%}")
        st.caption(f"Sample Size: {n_b:,}")

    # 5. Display Significance
    st.subheader("2. Statistical Conclusion")
    
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
