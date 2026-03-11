import streamlit as st
import scipy.stats as stats
import altair as alt
import pandas as pd
import statistics
import string
import math

st.set_page_config(
    page_title="SRM calculator",
    page_icon="🔢",
)

def calculate_srm(visitor_counts, expected_proportions):
    """
    Performs the Chi-squared test to check for Sample Ratio Mismatch.
    """
    total_visitors = sum(visitor_counts)
    expected_distribution = [p / 100 for p in expected_proportions]
    
    # Calculate expected frequencies: E = Total * p
    expected_counts = [total_visitors * p for p in expected_distribution]
    
    # Perform the chi-squared test
    chi2, p_value = stats.chisquare(f_obs=visitor_counts, f_exp=expected_counts)
    
    return {
        "p_value": p_value,
        "expected_counts": expected_counts,
        "mean_expected": statistics.mean(expected_counts),
        "is_mismatch": p_value < 0.01
    }
    
def initialize_state(num_variants):
    for i in range(num_variants):
        if f"obs_{i}" not in st.session_state:
            st.session_state[f"obs_{i}"] = 0
        if f"exp_{i}" not in st.session_state:
            st.session_state[f"exp_{i}"] = 50.0 # Default to 50/50 for 2 variants

def render_results(results, visitor_counts, num_variants):
    alphabet = string.ascii_uppercase
    p_value = results["p_value"]
    
    # Prepare Data for Altair
    raw_data = {
        "Variant": [alphabet[i] for i in range(num_variants)],
        "Observed": visitor_counts,
        "Expected": [round(x) for x in results["expected_counts"]]
    }
    df = pd.DataFrame(raw_data)
    
    # Transform data from columns to rows (Melt)
    df_melted = df.melt('Variant', var_name='Metric', value_name='Count')

    st.write("### Visual Comparison")

    # Create the Altair Chart
    chart = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X('Variant:N', title='Experiment Variant', 
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Count:Q', title='Number of Visitors'),
        color=alt.Color('Metric:N', scale=alt.Scale(range=['#ff7f0e', '#1f77b4'])),
        xOffset='Metric:N' # Creates the "grouped" bar effect
    ).properties(width='container', height=400)

    st.altair_chart(chart, width="stretch")

    if results["is_mismatch"]:
        st.error(f"SRM Detected! P-value: {p_value:.4f}")
        st.markdown("The distribution of data significantly deviates from what was expected. This usually suggests an issue with the randomization engine or tracking implementation.")
    else:
        st.success(f"No SRM Detected. P-value: {p_value:.4f}")
        st.markdown("The visitor distribution is within the expected range.")

    # Show the table below the chart
    with st.expander("View Raw Data Table"):
        st.dataframe(raw_data)

def run():
    st.title("Sample Ratio Mismatch (SRM) Checker")
    """
    This calculator lets you see if your online experiment correctly divided visitors among the variants, or if something went wrong and there was a mismatch with 
    the expected amount of visitors per variant. Enter the values below to get started. 

    Happy Learning!
    """
    
    num_variants = st.number_input("Number of variants?", min_value=2, max_value=26, step=1)
    
    col1, col2 = st.columns(2)
    alphabet = string.ascii_uppercase

    visitor_counts = []
    expected_proportions = []

    for i in range(num_variants):
        with col1:
            # Use 'key' to let streamlit save to session state
            val_obs = st.number_input(f"Visitors ({alphabet[i]})", min_value=0, step=1, key=f"obs_{i}")
            visitor_counts.append(val_obs)
        with col2:
            val_exp = st.number_input(f"Expected % ({alphabet[i]})", min_value=0.0, max_value=100.0, key=f"exp_{i}")
            expected_proportions.append(val_exp)
        
    if st.button("Check for SRM", type="primary"):
        # Validation logic
        if not math.isclose(sum(expected_proportions), 100.0, rel_tol=1e-5):
            st.error(f"The total expected percentage must equal 100% (currently {sum(expected_proportions):.2f}%).")
        elif sum(visitor_counts) == 0:
            st.error("Please enter visitor counts.")
        else:
            results = calculate_srm(visitor_counts, expected_proportions)
            render_results(results, visitor_counts, num_variants)

if __name__ == "__main__":
    run()