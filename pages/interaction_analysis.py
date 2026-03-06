import pandas as pd
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
import textwrap
from typing import Dict, Any, List
import streamlit as st

st.set_page_config(
    page_title="Interaction Analysis",
    page_icon="🔢"
)

def user_input():
    st.sidebar.header("Configuration")
    
    num_tests = st.sidebar.number_input("Number of concurrent tests", min_value=1, max_value=5, value=2)
    
    test_names: List[str] = []
    for i in range(num_tests):
        name = st.sidebar.text_input(f"Test {i+1} Name", value=f"test_{i+1}")
        test_names.append(name)

    st.write("### Experiment Data Entry")
    
    # Generate all combinations of Control/Variant
    variants = [['A', 'B'] for _ in range(num_tests)]
    combinations = list(itertools.product(*variants))
    
    # --- Fix: Build rows safely to avoid Pylance Type Mismatch ---
    init_data: List[Dict[str, Any]] = []
    for combo in combinations:
        # 1. Create the test variant part (Strings)
        test_part = {test_names[i]: str(combo[i]) for i in range(num_tests)}
        # 2. Create the data part (Integers)
        data_part = {"visitors": 0, "conversions": 0}
        # 3. Merge them using the union operator (Python 3.9+)
        full_row = test_part | data_part
        init_data.append(full_row)
    
    # Initialize DataFrame with explicit types for numeric columns
    default_df = pd.DataFrame(init_data)
    default_df["visitors"] = default_df["visitors"].astype(int)
    default_df["conversions"] = default_df["conversions"].astype(int)
    
    st.markdown(f"Fill in the data for all **{len(combinations)}** combinations below.")
    
    # use_container_width replaces width='stretch'
    edited_df = st.data_editor(default_df, num_rows="fixed", use_container_width=True)
    
    return edited_df, test_names

def convert_data(input_data, test_cols):
    """
    Handles multi-variant strings by converting them to dummy variables 
    before building the model.
    """
    rows = []
    for _, row in input_data.iterrows():
        # Add Successes
        rows.append({**{col: str(row[col]) for col in test_cols}, 'Conversion': 1, 'Count': row['conversions']})
        # Add Failures
        rows.append({**{col: str(row[col]) for col in test_cols}, 'Conversion': 0, 'Count': row['visitors'] - row['conversions']})
    
    return pd.DataFrame(rows)

def perform_logistic_regression(df, test_cols):
    # Use C(test) so statsmodels will recognize these are categorical factors
    # This handles multi-variant tests automatically
    formula = "Conversion ~ " + " * ".join([f"C({col})" for col in test_cols])
    
    try:
        model = sm.GLM.from_formula(
            formula, 
            data=df, 
            family=sm.families.Binomial(), 
            freq_weights=df['Count']
        ).fit()

        # --- Clean up the summary table for better readability ---
        summary_table = model.summary2().tables[1]
        
        new_names = {}
        for old_name in summary_table.index:
            if old_name == 'Intercept':
                new_names[old_name] = 'Baseline (Control)'
                continue
            
            # Clean up the name
            # 1. Handle Interactions (containing ':')
            if ':' in old_name:
                parts = old_name.split(':')
                clean_parts = [p.replace('C(', '').split(')')[0] for p in parts]
                new_names[old_name] = f"{' & '.join(clean_parts)} Interaction"
            # 2. Handle Main Effects
            else:
                clean_name = old_name.replace('C(', '').split(')')[0]
                variant_name = old_name.split('[T.')[1].replace(']', '')
                new_names[old_name] = f"{clean_name} ({variant_name})"
        
        # Apply the new names to the summary table
        summary_table.index = summary_table.index.map(new_names)
        
        st.write("### Model Summary")
        #st.write(model.summary())
        st.dataframe(summary_table.astype(float).round(4), use_container_width=True)
        
        # Interaction Logic
        p_values = model.pvalues
        significant_interactions = p_values[(p_values < 0.05) & (p_values.index.str.contains(':'))]
        
        st.write("### Interaction Analysis")
        if not significant_interactions.empty:
            st.warning(f"Detected {len(significant_interactions)} significant interaction(s):")
            for idx, pval in significant_interactions.items():
                coef = model.params[idx]
                display_name = new_names.get(idx, idx)
                st.write(f"- **{display_name}**: p={pval:.2e}, Coef: {coef:.4f} ({'Positive' if coef > 0 else 'Negative'} Interaction)")
        else:
            st.success("No significant test interactions detected.")
            
        return model
    except Exception as e:
        st.error(f"Model Error: {e}. Check if you have enough data combinations for the number of tests.")
        return None

def visualize_forest_plot(model):
    """
    Forest Plot: Visualizes the impact of every variant and every interaction 
    relative to the Control. This is the best way to see multi-variant clashes.
    """
    st.write("### Coefficient Forest Plot (Effect vs. Global Control)")
    st.info("Dots represent the effect size. Horizontal lines are 95% Confidence Intervals. If a line crosses the 0.0 vertical, it's not statistically significant.")

    # Extract params and confidence intervals (ignoring Intercept)
    params = model.params[1:]
    conf = model.conf_int()[1:]
    
    results = pd.DataFrame({
        'Feature': params.index,
        'Coefficient': params.values,
        'Lower': conf[0].values,
        'Upper': conf[1].values
    }).sort_values('Coefficient')

    fig, ax = plt.subplots(figsize=(10, len(results) * 0.4 + 2))
    
    # Color code by significance (doesn't cross 0)
    for i, (idx, row) in enumerate(results.iterrows()):
        is_sig = not (row['Lower'] <= 0 <= row['Upper'])
        color = '#ff4b4b' if is_sig else '#7d7d7d'
        
        ax.errorbar(row['Coefficient'], i, 
                    xerr=[[row['Coefficient'] - row['Lower']], [row['Upper'] - row['Coefficient']]], 
                    fmt='o', color=color, capsize=3, markersize=8)

    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(results['Feature'])
    ax.set_xlabel("Log-Odds Effect Size")
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    
    st.pyplot(fig)

def visualize_interaction_heatmap(model):
    st.write("### Multi-Test Interaction Matrix")
    params = model.params
    interaction_params = params[params.index.str.contains(':')]
    
    if interaction_params.empty:
        st.info("No 2-way interactions found.")
        return

    # Simplify names for the heatmap
    items = interaction_params.index.str.replace(r'C\(', '', regex=True).str.replace(r'\)', '', regex=True)
    matrix_data = pd.DataFrame({'Effect': interaction_params.values}, index=items)
    
    st.dataframe(matrix_data) # Heatmap for multi-variant is complex; a table/forest plot is clearer.

def run():
    st.title("Interaction Analysis")
    st.markdown("""
    This tool helps you analyze the interactions between multiple concurrent A/B tests.
    """)

    df, test_cols = user_input()

    # --- SQL Query Helper ---
    with st.expander("SQL Query Helper (Get your data)"):
        st.markdown("""
        To get the data for this tool, you need to group your users by **every** experiment they were exposed to. 
        Use the template below in your data warehouse (BigQuery, Snowflake, etc.):
        """)
        
        # Dynamically generate the column names based on user input
        test_cols_sql = ",\n    ".join([f"{name}_variant" for name in test_cols])
        group_by_sql = ", ".join([str(i+1) for i in range(len(test_cols))])
        
        sql_lines = [
            "SELECT",
            f"    {test_cols_sql},",
            "    COUNT(user_id) AS visitors,",
            "    SUM(conversion_flag) AS conversions",
            "FROM user_experiment_log",
            f"GROUP BY {group_by_sql}"
        ]

        sql_code = "\n".join(sql_lines)
        st.code(sql_code, language="sql")
        st.info("**Note:** Ensure your variant names (e.g., 'A', 'B') match the names you type into the table below.")
    
    # --- When to use this tool ---
    with st.expander("Why Use This Tool?"):
        st.markdown("""
        ### The Problem: Interaction Bias
        When you run multiple experiments at the same time, you risk **Interaction Bias**. This occurs when the effect of one change (e.g., a new button color) is influenced by another change (e.g., a new pricing model). 
        Even though real-life interaction effects are rare, the risk for them increases exponentially with more concurrently running experiments.
        
        **Standard A/B test dashboards often fail here because:**
        * They assume experiments are independent.
        * They can't tell you if Variant A only works when Variant B is also present.
        * They might report a "winner" that actually performs poorly when combined with other live features.

        ### Solution: Factorial interaction Analysis
        The tool moves beyond simple averages. This model calculates the **Combined Effect**. 
        
        * **Detect "Clashes":** Identify if two great features actually hurt conversion when shown together (Negative Interaction).
        * **Discover "Synergies":** Find combinations where 1 + 1 = 3 (Positive Interaction).
        * **Clean Results:** Get the "Pure" lift of your experiment by mathematically removing the noise caused by other concurrent tests.
        """)
        
        st.info("Use this tool whenever you have overlapping traffic between two or more experiments to ensure your 'winning' variants are truly compatible.")

    # --- How to Use ---
    with st.expander("How to Use This Tool"):
        st.markdown("""
        1. **Sidebar Setup**: Use the sidebar to set the number of tests you are running and name them.
        2. **Enter Data**: Fill in the table for every combination that occurred (e.g., Control/Control, B/Control, B/X).
        3. **Rows**: Use the **(+)** button at the bottom of the table to add new combination rows.
        4. **Baseline**: Ensure at least one row represents your 'Control' group for each test.
        """)
        st.info("**Important:** Be sure to group experiment variants together when you enter data (e.g., B and B/X) to capture the interaction effects properly. Read rows from left to right to know which combinations to enter data for.")

    # --- Methodology ---
    with st.expander("Methodology & Statistical Approach"):
        st.markdown(r"""
        This tool uses a **Generalized Linear Model (GLM)** with a binomial family to perform logistic regression.
        
        **The Interaction Formula:**
        For two tests, the model calculates:
        $$ \text{logit}(p) = \beta_0 + \beta_1 \text{Test}_1 + \beta_2 \text{Test}_2 + \beta_3 (\text{Test}_1 \times \text{Test}_2) $$
        
        The **interaction term** ($\beta_3$) tells us if the combined effect of two variants is significantly different from the sum of their individual effects.
        """)

    if st.button("Calculate Interaction Effects", type="primary"):
        if not df.empty and (df['visitors'] > 0).all():
            data_long = convert_data(df, test_cols)
            model = perform_logistic_regression(data_long, test_cols)
            if model:
                # Forest Plot is the primary visualization for multi-variant scale
                visualize_forest_plot(model)
        else:
            st.error("Please ensure all rows have a visitor count greater than 0.")

if __name__ == "__main__":
    run()