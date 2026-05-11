import streamlit as st
from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from typing import List

st.set_page_config(
    page_title="Pre-test analysis",
    page_icon="🔢",
    layout="wide",
)

# --- USER INPUT ---

# User input for both MDE and sample size calculation
def get_user_input() -> None:
    st.write("### Baseline Data")
    st.write("Enter weekly visitors, weekly conversions and test parameters.")

    col1, col2 = st.columns(2)
    # Baseline data
    with col1:
        st.number_input("Number of variants (including control):", min_value=2, step=1, value=st.session_state.get("num_variants", 2), key="num_variants")
        st.number_input("Visitors in baseline variant:", min_value=0, step=1, value=st.session_state.get("baseline_visitors", 0), key="baseline_visitors")
        st.number_input("Conversions in baseline variant:", min_value=0, step=1, value=st.session_state.get("baseline_conversions", 0), key="baseline_conversions")

    # Test parameter input
    with col2:
        st.number_input("Desired confidence level (e.g., 90%):", min_value=0, max_value=100, step=1, value=st.session_state.get("risk", 95), key="risk")
        st.number_input("Minimum trustworthiness (Power) (e.g., 80%):", min_value=0, max_value=100, step=1, value=st.session_state.get("trust", 80), key="trust")
        if st.session_state.get("calculation_mode") == "Calculate Sample Size based on MDE" or st.session_state.get("calculation_mode") == "Calculate Power for Desired Lift":
            st.number_input("What MDE are you aiming for?", min_value=1, max_value=100, step=1, value=st.session_state.get("mde", 5), key="mde")
    st.radio(
        "Hypothesis type ('One-sided' or 'Two-sided'): ",
        options=['One-sided', 'Two-sided'], 
        index=['One-sided', 'Two-sided'].index(st.session_state.get("tails", 'One-sided')),
        horizontal=True,
        key="tails",
        help="Choose 'One-sided' when testing only for improvement (B > A) or decline (B < A); this requires fewer samples and results in a possible lower MDE. Choose 'Two-sided' when testing for any difference (better or worse); this is more comprehensive because it can detect significant effects in either direction, but generally requires more samples and possibly raises the MDE."
    )
    if st.session_state.get("calculation_mode") == "Calculate Power for Desired Lift":
        weeks_to_run = st.slider("Test Duration (Weeks)", min_value=1, max_value=6, value=4, key="weeks_to_run")

# --- HELPER FUNCTIONS ---

# Holm-Bonferroni correction for MDE calculation
def holm_bonferroni(
        num_variants: int, 
        alpha: float, 
        tails: str
        ) -> float:
    adjusted_alpha = alpha / np.arange(num_variants, 0, -1)
    if tails == 'Two-sided':
        z_alpha = norm.ppf(1 - adjusted_alpha / 2)
    else:
        z_alpha = norm.ppf(1 - adjusted_alpha)
    return np.max(z_alpha)

def perform_mde_calculation(
        num_variants: int, 
        baseline_visitors: int, 
        baseline_conversions: int, 
        risk: float, 
        trust: float, 
        tails: str, 
        traffic_multiplier: float=1.0
        ) -> List[List[float | int]]:
    """
    Core MDE calculation. Accepts an optional traffic_multiplier to model
    spikes or drops in visitor volume (e.g. 0.7 = -30%, 1.3 = +30%).
    """

    alpha = 1 - (risk / 100)
    power = trust / 100

    # Calculate baseline conversion rate
    baseline_rate = baseline_conversions / baseline_visitors

    # Adjust alpha for multiple comparisons
    if num_variants > 2:
        adjusted_z_alpha = holm_bonferroni(num_variants - 1, alpha, tails)
    else:
        adjusted_z_alpha = norm.ppf(1 - alpha) if tails == 'One-sided' else norm.ppf(1 - alpha / 2)

    # Z-score for power
    z_power = norm.ppf(power)

    # Weekly increments
    weeks = range(1, 7)  # For 6 weeks
    weekly_visitors = int(np.ceil(baseline_visitors * traffic_multiplier / num_variants))

    # Prepare list to store results
    results = []
    for week in weeks:
        visitors_per_variant_weekly = weekly_visitors * week

        # Standard error and MDE calculations
        se = np.sqrt(2 * baseline_rate * (1 - baseline_rate) / visitors_per_variant_weekly)
        mde_absolute = (adjusted_z_alpha + z_power) * se
        mde_relative = (mde_absolute / baseline_rate) * 100

        # Store results
        results.append([week, visitors_per_variant_weekly, mde_relative])

    return results

def _mde_color_scale() -> List[List[float | str]]:
    """Returns a Plotly colorscale: green → yellow → red mapped to MDE 0–20%+."""
    return [
        [0.0,  "rgb(56, 161, 105)"], # green (<5%)
        [0.25, "rgb(154, 205, 90)"], # yellow-green
        [0.5,  "rgb(236, 201, 75)"], # yellow (~10%)
        [0.75, "rgb(237, 137, 54)"], # orange
        [1.0,  "rgb(229, 62, 62)"], # red (>20%)
    ]

def display_mde_table(
        num_variants: int, 
        baseline_visitors: int, 
        baseline_conversions: int, 
        risk: float, 
        trust: float, 
        tails: str
        ) -> None:
    st.write("## MDE Calculation Results")
    st.write("""
        This table displays the minimum effect size detectable each week. An MDE of <5% is usually testworthy; 5–10% is debatable.
        For larger MDEs, consider whether the effect size can be achieved.
    """)
    if num_variants > 2:
        st.write(f"*Note: The Holm-Bonferroni correction was applied ({num_variants - 1} comparisons) affecting the required significance level.*")
 
    tab_standard, tab_sensitivity = st.tabs(["Standard MDE Table", "Sensitivity Matrix"])
 
    # --- Tab 1: Standard table ---
    with tab_standard:
        results = perform_mde_calculation(
            num_variants, baseline_visitors, baseline_conversions, risk, trust, tails
        )
        df = pd.DataFrame(results, columns=['Week', 'Visitors / Variant', 'Relative MDE (%)'])
        df['Relative MDE (%)'] = df['Relative MDE (%)'].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
        st.write(df.to_html(index=False), unsafe_allow_html=True)
 
    # --- Tab 2: Sensitivity matrix ---
    with tab_sensitivity:
        st.write("""
            **How would traffic spikes or drops affect your MDE?**  
            Each cell shows the Relative MDE (%) for a given week and traffic scenario.
            Hover over a cell for exact values. Colors: 🟢 < 5% · 🟡 5–10% · 🔴 > 10%.
        """)
 
        # Define traffic multiplier scenarios
        multipliers = [0.50, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.50]
        scenario_labels = [f"{int((m - 1) * 100):+d}%" if m != 1.00 else "Baseline" for m in multipliers]
        weeks = list(range(1, 7))
 
        # Build MDE matrix (rows = weeks, cols = scenarios)
        mde_matrix = []
        for week_idx, week in enumerate(weeks):
            row = []
            for m in multipliers:
                res = perform_mde_calculation(
                    num_variants, baseline_visitors, baseline_conversions, risk, trust, tails,
                    traffic_multiplier=m
                )
                mde_val = res[week_idx][2] # relative MDE for this week
                row.append(round(mde_val, 2))
            mde_matrix.append(row)
 
        z = np.array(mde_matrix)
        text_labels = [[f"{v:.1f}%" for v in row] for row in mde_matrix]
        y_labels = [f"Week {w}" for w in weeks]
 
        # Cap colorscale at 20% so cells above 20% are all deep red
        z_capped = np.clip(z, 0, 20)
 
        fig = go.Figure(data=go.Heatmap(
            z=z_capped,
            x=scenario_labels,
            y=y_labels,
            text=text_labels,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "white"},
            colorscale=_mde_color_scale(),
            zmin=0,
            zmax=20,
            showscale=True,
            colorbar=dict(
                title="MDE (%)",
                tickvals=[0, 5, 10, 15, 20],
                ticktext=["0%", "5%", "10%", "15%", "≥20%"],
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Traffic scenario: <b>%{x}</b><br>"
                "Relative MDE: <b>%{text}</b><extra></extra>"
            ),
        ))
 
        # Highlight baseline column with a border effect via shape
        baseline_col_idx = scenario_labels.index("Baseline")
        fig.add_shape(
            type="rect",
            x0=baseline_col_idx - 0.5,
            x1=baseline_col_idx + 0.5,
            y0=-0.5,
            y1=len(weeks) - 0.5,
            line=dict(color="white", width=2.5),
            fillcolor="rgba(0,0,0,0)",
        )
 
        fig.update_layout(
            xaxis=dict(
                title="Traffic vs. Baseline",
                side="bottom",
                type="category",
                tickfont=dict(size=12),
                automargin=True,
            ),
            yaxis=dict(
                title="",
                tickfont=dict(size=12),
                autorange="reversed", # Week 1 at top
            ),
            margin=dict(l=80, r=60, t=40, b=80),
            height=340,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
 
        st.plotly_chart(fig, width='stretch')
 
        st.caption(
            "Traffic multipliers apply uniformly across all days in a given week. "
            "For week-by-week seasonality modelling, use the **Seasonal (Prophet Forecast)** mode."
        )

def calculate_sample_size(
        num_variants: int, 
        baseline_visitors: int, 
        baseline_conversions: int, 
        mde: float, 
        risk: float, 
        trust: float, 
        tails: str
        ) -> None:

    # --- Input Validation and Parameter Conversion ---

    if baseline_visitors <= 0:
        st.error("Baseline visitors must be greater than 0.")
        return # Stop calculation
    if baseline_conversions < 0: # Allow 0 conversions, but not negative
        st.error("Baseline conversions cannot be negative.")
        return
    if baseline_conversions > baseline_visitors:
        st.error("Baseline conversions cannot be greater than baseline visitors.")
        return
    if mde <= 0:
        st.error("Minimum Detectable Effect (MDE) must be greater than 0%.")
        return # Stop calculation

    try:
        mde_relative = mde / 100
        alpha = 1 - (risk / 100) # Significance level
        power = trust / 100 # Statistical power
        beta = 1 - power # Type II error rate

        # Baseline conversion rate
        p = baseline_conversions / baseline_visitors

        # Minimum Detectable Effect (absolute)
        mde_absolute = p * mde_relative
        effect_size = mde_absolute

        # Expected conversion rates
        p1 = p
        p2 = p + mde_absolute # Treatment group rate

        if p2 > 1.0:
            st.warning(f"The calculated treatment conversion rate ({p2:.2%}) exceeds 100% based on the baseline rate ({p:.2%}) and MDE ({mde}%). Please check your inputs.")
        if p2 < 0.0:
             st.warning(f"The calculated treatment conversion rate ({p2:.2%}) is negative based on the baseline rate ({p:.2%}) and MDE ({mde}%). Please check your inputs.")

    except ZeroDivisionError:
        st.error("Error during parameter calculation (potentially division by zero). Please ensure baseline visitors > 0.")
        return
    except Exception as e:
        st.error(f"An unexpected error occurred during parameter setup: {e}")
        return

    st.write("## Sample Size Calculation Results")
    st.write(f"Calculating required sample size for a desired relative MDE of **{mde}%**.")

    # --- Z-Score Calculation ---

    num_comparisons = 0
    correction_applied = False

    try:
        # Adjust alpha for multiple comparisons if necessary
        if num_variants > 2:
            num_comparisons = num_variants - 1
            z_alpha_adjusted = holm_bonferroni(num_comparisons, alpha, tails)
            correction_applied = True
        else: # num_variants == 2
            if tails == 'One-sided':
                z_alpha_adjusted = norm.ppf(1 - alpha)
            else: # Two-sided
                z_alpha_adjusted = norm.ppf(1 - alpha / 2)
            correction_applied = False

        # Z-score for power (positive value corresponding to 1-beta or power)
        z_power = norm.ppf(1 - beta) # Equivalent to norm.ppf(power)

        if z_alpha_adjusted is None or z_power is None:
             raise ValueError("Z-score calculation failed.")

    except AttributeError:
         st.error("Error: norm.ppf function not found. Ensure scipy is correctly installed.")
         return
    except Exception as e:
        st.error(f"An error occurred during Z-score calculation: {e}")
        return

    # --- Sample Size Formula ---
    try:
        # Variance terms
        var1 = p1 * (1 - p1)
        var2 = p2 * (1 - p2)

        # Ensure variances are non-negative (can happen with p=0 or p=1)
        var1 = max(var1, 0)
        var2 = max(var2, 0)

        # Use approximation p*(1-p) for the first term's variance
        # Or use pooled variance: p_pooled = (p1+p2)/2; var_pooled = p_pooled*(1-p_pooled)
        term1 = z_alpha_adjusted * np.sqrt(2 * p * (1 - p))
        term2 = z_power * np.sqrt(var1 + var2)

        # Required sample size per group
        if effect_size == 0: # Should be caught by MDE > 0 validation earlier
             st.error("Effect size is zero. Cannot calculate sample size.")
             return

        sample_size_per_group = ((term1 + term2) ** 2) / (effect_size ** 2)
        sample_size_per_group = np.ceil(sample_size_per_group)

        if not np.isfinite(sample_size_per_group) or sample_size_per_group <= 0:
            st.error("Calculated sample size is invalid or non-positive. Please check inputs (especially MDE and baseline rates).")
            return

    except ZeroDivisionError:
        st.error("Error calculating sample size (division by zero). This might happen if the MDE is zero.")
        return
    except Exception as e:
        st.error(f"An unexpected error occurred during sample size calculation: {e}")
        return

    # --- Runtime Estimation ---
    def estimate_runtime(
            ss_per_group: float, 
            visitors_per_week: int, 
            n_variants: int
            ) -> str | int:
        try:
            if visitors_per_week <= 0 or n_variants <= 0:
                return "infinite (zero baseline visitors or variants)"

            daily_visitors_total = visitors_per_week / 7
            visitors_per_group_per_day = daily_visitors_total / n_variants

            if visitors_per_group_per_day <= 0:
                return "infinite (zero daily visitors per group)"

            days_required = ss_per_group / visitors_per_group_per_day
            estimated_days = int(np.ceil(days_required))
            return estimated_days
        except Exception as e:
            st.warning(f"Could not estimate runtime: {e}")
            return "unavailable"

    estimated_days = estimate_runtime(sample_size_per_group, baseline_visitors, num_variants)

    # --- Display Results ---
    st.write(f"The required sample size per group (including control) is **{int(sample_size_per_group):,}**.")
    st.write(f"With an average of **{int(baseline_visitors):,}** total visitors per week, your test is estimated to run for approximately **{estimated_days}** days to reach the required sample size per group.")

    if correction_applied:
        st.write(f"*Note: The {holm_bonferroni.__name__ if 'holm_bonferroni' in globals() else 'configured multiple comparison correction'} correction was applied ({num_comparisons} comparisons) affecting the required significance level.*")

# Forecasting with Prophet
@st.cache_data(show_spinner=False)
def run_prophet_forecast(
    df: pd.DataFrame, 
    periods: int = 42, 
    interval_width: float = 0.95
    ) -> pd.DataFrame:

    df_vis = df[['ds', 'visitors']].rename(columns={'visitors': 'y'})
    m_vis = Prophet(yearly_seasonality=True, weekly_seasonality=True, interval_width=interval_width) # type: ignore[arg-type]
    m_vis.fit(df_vis)
    future_vis = m_vis.make_future_dataframe(periods=periods)
    forecast_vis = m_vis.predict(future_vis)
 
    df_conv = df[['ds', 'conversions']].rename(columns={'conversions': 'y'})
    m_conv = Prophet(yearly_seasonality=True, weekly_seasonality=True, interval_width=interval_width) # type: ignore[arg-type]
    m_conv.fit(df_conv)
    future_conv = m_conv.make_future_dataframe(periods=periods)
    forecast_conv = m_conv.predict(future_conv)
 
    last_date = df['ds'].max()
    cols_to_keep = [
        'ds', 
        'yhat', 
        'yhat_lower', 
        'yhat_upper'
    ]
 
    future_vis = forecast_vis[forecast_vis['ds'] > last_date][cols_to_keep].rename(
        columns={
            'yhat': 'pred_visitors', 
            'yhat_lower': 'vis_lower', 
            'yhat_upper': 'vis_upper'
        }
    )
    future_conv = forecast_conv[forecast_conv['ds'] > last_date][cols_to_keep].rename(
        columns={
            'yhat': 'pred_conversions', 
            'yhat_lower': 'conv_lower', 
            'yhat_upper': 'conv_upper'
        }
    )
 
    forecast_final = pd.merge(future_vis, future_conv, on='ds')
 
    cols_to_clip = ['pred_visitors', 'vis_lower', 'vis_upper', 'pred_conversions', 'conv_lower', 'conv_upper']
    for col in cols_to_clip:
        forecast_final[col] = forecast_final[col].clip(lower=0)
 
    return forecast_final

def perform_mde_calculation_forecast(
        forecast_df: pd.DataFrame, 
        num_variants: int, 
        risk: float, 
        trust: float, 
        tails: str
         ) -> List[List[float | int | str]]:
    """
    Calculates MDE based on accumulating forecasted data rather than static averages.
    """
    alpha = 1 - (risk / 100)
    power = trust / 100

    # Adjust alpha for multiple comparisons
    if num_variants > 2:
        adjusted_z_alpha = holm_bonferroni(num_variants - 1, alpha, tails)
    else:
        adjusted_z_alpha = norm.ppf(1 - alpha) if tails == 'One-sided' else norm.ppf(1 - alpha / 2)

    z_power = norm.ppf(power)

    results = []
    
    # Iterate through weeks 1 to 6
    for week in range(1, 7):
        days_needed = week * 7
        
        # Slice the forecast for this duration
        current_slice = forecast_df.head(days_needed)
        
        # Sum the traffic and conversions to get the "seasonal baseline" for this specific window
        total_visitors = current_slice['pred_visitors'].sum()
        total_conversions = current_slice['pred_conversions'].sum()
        
        if total_visitors <= 0:
            results.append([week, 0, np.nan])
            continue
            
        # Weighted Baseline Conversion Rate for this specific period
        baseline_rate = total_conversions / total_visitors
        baseline_rate = max(0.0001, min(0.9999, baseline_rate)) # ensure rate is valid
        
        # Visitors per variant
        visitors_per_variant = total_visitors / num_variants
        
        # Standard Error & MDE
        se = np.sqrt(2 * baseline_rate * (1 - baseline_rate) / visitors_per_variant)
        mde_absolute = (adjusted_z_alpha + z_power) * se
        mde_relative = (mde_absolute / baseline_rate) * 100
        
        results.append([week, int(visitors_per_variant), mde_relative])
        
    return results

def calculate_cohens_h(rate_a, rate_b, absolute=True):
    """
    Calculates Cohen's h for the effect size between two proportions.
    """
    # Safeguard: Rates cannot exceed 100% or be negative for the arcsine transformation
    rate_a = np.clip(rate_a, 0.0, 1.0)
    rate_b = np.clip(rate_b, 0.0, 1.0)

    # Apply the arcsine transformation to both proportions
    phi_a = 2 * np.arcsin(np.sqrt(rate_a))
    phi_b = 2 * np.arcsin(np.sqrt(rate_b))
    
    # Find the difference
    h = phi_b - phi_a
    
    # Return the absolute value (magnitude) or the directional value
    cohens_h = abs(h) if absolute else h
    
    return cohens_h

def calculate_power(
    risk_level: float, 
    expected_lift_pct: float,
    tails: str, 
    visitors_per_week: int, 
    conversions_per_week: int,
    weeks_to_run: int,
    num_variants: int
) -> tuple[float, float, float]:

    rate_a = conversions_per_week / visitors_per_week
    rate_b = rate_a * (1 + (expected_lift_pct / 100.0))
    
    alpha = 1.0 - (risk_level / 100.0)

    # --- MULTIPLE COMPARISONS CORRECTION ---
    num_comparisons = num_variants - 1
    if num_comparisons > 1:
        # Standard Bonferroni correction for power planning
        alpha = alpha / num_comparisons
    # ---------------------------------------
    
    alternative = 'two-sided' if tails == 'Two-sided' else 'larger'
    
    # Calculate total traffic per variant over the duration
    n_a = visitors_per_week * weeks_to_run
    
    # Calculate Effect Size & Power
    effect_size_h = calculate_cohens_h(rate_a, rate_b)
    power_analysis = NormalIndPower()
    
    power = power_analysis.solve_power(
        effect_size=effect_size_h, 
        nobs1=n_a, 
        ratio=1.0,
        alpha=alpha, 
        alternative=alternative
    )
    
    target_power = st.session_state.get("trust", 80) / 100.0
    
    return power, target_power, rate_b

def run() -> None:
    st.title("Pre-test analysis")
    """
    This calculator helps you plan for the runtime of your experiment.

    Enter the values below to start.
    """
    with st.expander("How to plan your tests", expanded=False):
        st.markdown("""
        ### How to choose the right method:

        **1. MDE Projection (Fixed Duration)**
        * **Best for:** Strict deadlines.
        * *Scenario:* "We only have 4 weeks to run this test. What is the smallest impact we can reliably detect?"
        * *Output:* A table showing the Minimum Detectable Effect (MDE) achievable for Weeks 1 through 6.

        **2. Sample Size Calculation (Target Effect)**
        * **Best for:** Specific improvement goals.
        * *Scenario:* "We need to detect a 5% lift to justify this feature. How long will that take?"
        * *Output:* The total sample size required and the estimated runtime in days (based on average traffic).
        
        **3. Power Calculation for Desired Lift**
        * **Best for:** Reality checks and resource allocation.
        * *Scenario:* "Product expects a specific 5% lift, and we only have 4 weeks to run the test. What are the actual odds we detect it?"
        * *Output:* The statistical power (probability of successfully detecting the expected lift) and a clear pass/fail against your minimum trustworthiness threshold.

        **4. Seasonal Forecasting (Prophet)**
        * **Best for:** Volatile, high-traffic, or event-driven sites.
        * *Scenario:* "Our traffic spikes on weekends or is approaching a holiday (e.g., Black Friday)."
        * *Why:* Standard calculators assume flat traffic. This method uses your historical data to **forecast** future daily traffic, preventing you from under-powering your test during traffic dips.
        """)

    # Selectbox for choosing the calculation mode
    calculation_mode = st.selectbox(
        "Select Calculation Mode:",
        ("Calculate MDE based on Runtime", "Calculate Sample Size based on MDE", "Calculate Power for Desired Lift", "Seasonal (Prophet Forecast)"),
        help="For stable traffic / conversions, choose either MDE or sample size calculation. If traffic and conversion is seasonal (or highly volatile), choose Seasonal. To validate whether an expected uplift is enough to detect a trustworthy effect, choose 'calculate power'.",
        key="calculation_mode"
    )

    if calculation_mode == "Calculate MDE based on Runtime":
        get_user_input()
        if st.button("Calculate MDE", type="primary"):
            # Validate input using st.session_state
            if (st.session_state.get("baseline_visitors", 0) <= 0 or
                st.session_state.get("baseline_conversions", 0) < 0 or
                st.session_state.get("baseline_conversions", 0) > st.session_state.get("baseline_visitors", 0) or
                not (0 < st.session_state.get("risk", 0) <= 100) or
                not (0 < st.session_state.get("trust", 0) <= 100) or
                st.session_state.get("tails") not in ['One-sided', 'Two-sided']):
                st.write("<span style='color: #ff6600;'>*Please enter valid inputs for all fields (Visitors > 0, Conversions >= 0 and <= Visitors, Risk/Trust between 0-100).</span>", unsafe_allow_html=True)
            else:
                display_mde_table(st.session_state.get("num_variants", 2),
                                  st.session_state.get("baseline_visitors", 0),
                                  st.session_state.get("baseline_conversions", 0),
                                  st.session_state.get("risk", 95),
                                  st.session_state.get("trust", 80),
                                  st.session_state.get("tails", 'One-sided'))

    elif calculation_mode == "Calculate Sample Size based on MDE":
        get_user_input()
        if st.button("Calculate Sample Size", type="primary"):
            # Add Validation Block using st.session_state
            if (st.session_state.get("baseline_visitors", 0) <= 0 or
                    st.session_state.get("baseline_conversions", 0) < 0 or
                    st.session_state.get("baseline_conversions", 0) > st.session_state.get("baseline_visitors", 0) or
                    not (0 < st.session_state.get("risk", 90) <= 100) or
                    not (0 < st.session_state.get("trust", 80) <= 100) or
                    st.session_state.get("mde", 5) <= 0 or
                    st.session_state.get("tails", 'One-sided') not in ['One-sided', 'Two-sided']):
                # If input is INVALID, show an error message
                st.write("<span style='color: #ff6600;'>*Please enter valid inputs for all fields (Visitors > 0, Conversions >= 0 and <= Visitors, Risk/Trust between 0-100, MDE > 0).</span>", unsafe_allow_html=True)
            else:
                # If input IS VALID, call the calculation function
                calculate_sample_size(st.session_state.get("num_variants", 2),
                                      st.session_state.get("baseline_visitors", 0),
                                      st.session_state.get("baseline_conversions", 0),
                                      st.session_state.get("mde", 5),
                                      st.session_state.get("risk", 95),
                                      st.session_state.get("trust", 80),
                                      st.session_state.get("tails", 'One-sided'))
    elif calculation_mode == "Calculate Power for Desired Lift":
        get_user_input()
        if st.button("Calculate Power", type="primary"):
            num_variants = st.session_state.get("num_variants", 2)
            visitors_per_week = st.session_state.get("baseline_visitors", 0)
            conversions_per_week = st.session_state.get("baseline_conversions", 0)
            risk_level = st.session_state.get("risk", 95)
            expected_lift_pct = st.session_state.get("mde", 5)
            tails = st.session_state.get("tails", "One-sided")
            weeks_to_run = st.session_state.get("weeks_to_run", 4)
        
            # Solve for Power
            if visitors_per_week > 0 and conversions_per_week > 0 and conversions_per_week <= visitors_per_week:
                power, target_power, rate_b = calculate_power(
                    risk_level, 
                    expected_lift_pct,
                    tails, 
                    visitors_per_week, 
                    conversions_per_week,
                    weeks_to_run,
                    num_variants
                )
    
                if rate_b > 1.0:
                    st.warning(f"Note: An expected lift of {expected_lift_pct}% pushes your variant's conversion rate over 100%. The calculator has capped the expected rate at 100%.")
    
                # Display Results
                st.divider()
                st.write(f"### Results for {weeks_to_run}-Week Test")
                st.metric(label=f"Statistical Power (Probability of detecting a {expected_lift_pct}% lift)", value=f"{power:.1%}")
                
                if power < target_power:
                    st.warning(f"⚠️ **Underpowered.** Your power is below your minimum threshold of {target_power:.1%}. You need to run the test longer or accept a higher expected lift.")
                else:
                    st.success(f"✅ **Adequately Powered.** Your test meets your {target_power:.1%} trustworthiness requirement.")
            else:
                st.info("Please enter valid baseline visitors and conversions to calculate power. Visitors must be > 0, and conversions must be >= 0 and <= visitors.")
    else:
        st.write("### Upload Historical Data")
        st.info("Upload a CSV with columns: `date` (YYYY-MM-DD), `visitors` (count), `conversions` (count). Ideally 1-2 years of data (not more!).")
        
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        # Common inputs for the seasonal mode
        col1, col2 = st.columns(2)
        with col1:
             num_variants_s = st.number_input("Number of variants:", min_value=2, value=2, key="seas_variants")
        with col2:
             risk_s = st.number_input("Confidence level (%):", value=95, key="seas_risk")
             trust_s = st.number_input("Power (%):", value=80, key="seas_trust")
             tails_s = st.radio("Hypothesis:", ['One-sided', 'Two-sided'], key="seas_tails", horizontal=True)

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                # Normalize columns for Prophet
                df.columns = [c.lower() for c in df.columns]
                
                # Simple column mapping attempt
                if 'date' in df.columns:
                    df = df.rename(columns={'date': 'ds'})
                
                # Check for required columns
                if not {'ds', 'visitors', 'conversions'}.issubset(df.columns):
                    st.error("CSV must contain columns: 'date' (or 'ds'), 'visitors', 'conversions'")
                else:
                    df['ds'] = pd.to_datetime(df['ds'], dayfirst=True)
                    forecast_confidence = risk_s / 100
                    
                    if st.button("Generate Forecast & Analysis", type="primary"):
                        with st.spinner("Running Prophet Forecast..."):
                            # Run Forecast
                            forecast_data = run_prophet_forecast(df, periods=42, interval_width=forecast_confidence) # type: ignore[call-arg]
                            
                            # Display Forecast Plot
                            st.write("### Traffic Forecast (Next 6 Weeks)")
                            
                            # Create Plotly Figure
                            fig = go.Figure()

                            # Main Line (Predicted Visitors)
                            fig.add_trace(go.Scatter(
                                x=forecast_data['ds'], 
                                y=forecast_data['pred_visitors'],
                                mode='lines',
                                name='Predicted Visitors',
                                line=dict(color='#0072B2')
                            ))

                            # Confidence Interval (Upper Bound) - Invisible line for filling
                            fig.add_trace(go.Scatter(
                                x=forecast_data['ds'], 
                                y=forecast_data['vis_upper'],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            ))

                            # Confidence Interval (Lower Bound) - Fills up to the Upper Bound
                            fig.add_trace(go.Scatter(
                                x=forecast_data['ds'], 
                                y=forecast_data['vis_lower'],
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty', # Fills to the previous trace (vis_upper)
                                fillcolor='rgba(0, 114, 178, 0.2)', # Same blue, 0.2 opacity
                                name=f'Confidence Interval ({int(forecast_confidence * 100)}%)'
                            ))

                            fig.update_layout(
                                title="Daily Visitor Forecast",
                                yaxis_title="Visitors",
                                xaxis_title="Date",
                                hovermode="x"
                            )

                            # Render
                            st.plotly_chart(fig, width="stretch")
                            
                            # Run Calculation
                            results = perform_mde_calculation_forecast(
                                forecast_data, num_variants_s, risk_s, trust_s, tails_s
                            )
                            
                            # Display Results
                            res_df = pd.DataFrame(results, columns=['Week', 'Avg Visitors / Variant', 'Relative MDE (%)'])
                            res_df['Relative MDE (%)'] = res_df['Relative MDE (%)'].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
                            
                            st.write("### Seasonal MDE Results")
                            st.write("This table calculates MDE using the **predicted** traffic and conversion rate for each specific week, accounting for seasonality.")
                            st.table(res_df)
                            
            except Exception as e:
                st.error(f"Error parsing file: {e}")

if __name__ == "__main__":
    run()
