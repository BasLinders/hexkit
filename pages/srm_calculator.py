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
    sum_props = sum(expected_proportions)

    if sum_props == 0:
        raise ValueError("Total proportions must be greater than zero.")

    expected_distribution = [p / sum_props for p in expected_proportions]
    
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
    df_melted['opacity'] = df_melted['Metric'].apply(lambda x: 0.4 if x == 'Expected' else 1.0)

    st.write("### Visual Comparison")

    # Create the Altair Chart
    chart = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X('Variant:N', title='Experiment Variant', 
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Count:Q', title='Number of Visitors'),
        color=alt.Color('Metric:N', scale=alt.Scale(range=['#ff7f0e', '#1f77b4'])),
        opacity=alt.Opacity('opacity:Q', legend=None),
        xOffset='Metric:N' # Creates the "grouped" bar effect
    ).properties(height=400)

    st.altair_chart(chart, use_container_width=True)

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

    initialize_state(num_variants)
    
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
            val_exp = st.number_input(f"Expected % ({alphabet[i]})", min_value=0, max_value=100, key=f"exp_{i}")
            expected_proportions.append(val_exp)
        
    if st.button("Check for SRM", type="primary"):
        total_visitors = sum(visitor_counts)
        current_sum = sum(expected_proportions)

        # Critical Blockers: No data entered
        if total_visitors == 0:
            st.error("Please enter visitor counts.")
        
        # Critical Blockers: Total proportions are zero or negative
        elif current_sum <= 0:
            st.error("Total expected proportions must be greater than 0%.")
        elif any(p == 0 for p in expected_proportions):
            st.error("Expected % must be greater than 0 for all active variants.")

        # Successful path (with optional non-blocking warnings)
        else:
            # Show helpful tips if the math isn't exactly 100
            if not math.isclose(current_sum, 100.0, abs_tol=0.1):
                avg_p = current_sum / num_variants
                st.warning(f"Note: Total proportions add up to **{current_sum}%**. We've normalized these to 100% for the calculation.")
                
                # Scalable tip for the '50% default' mistake
                if num_variants > 2 and math.isclose(avg_p, 50.0, abs_tol=1.0):
                    suggested_p = round(100 / num_variants)
                    st.info(f"For {num_variants} variants, an even split would be **{suggested_p}%** each.")

            # Run and render
            try:
                results = calculate_srm(visitor_counts, expected_proportions)
                render_results(results, visitor_counts, num_variants)
            except ValueError as e:
                st.error("Cannot calculate SRM with the current inputs. Ensure all variants have visitors and the expected percentage is greater than 0.")
            except Exception as e:
                st.error("An unexpected error occurred. Please check your inputs.")

if __name__ == "__main__":
    run()
