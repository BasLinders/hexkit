import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st

st.set_page_config(
    page_title="Interaction Analysis",
    page_icon="🔢",
)

def user_input():
    # Initialize session state defaults if they don't exist
    for key in ["AA_u", "AB_u", "BA_u", "BB_u", "AA_c", "AB_c", "BA_c", "BB_c"]:
        st.session_state.setdefault(key, 0)

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Visitors")
        AA_u = st.number_input("AA Visitors", value=st.session_state.AA_u, help="Visitors that saw the control of both experiment 1 and 2.")
        AB_u = st.number_input("AB Visitors", value=st.session_state.AB_u, help="Visitors that saw the control of experiment 1 and the treatment variant of experiment 2.")
        BA_u = st.number_input("BA Visitors", value=st.session_state.BA_u, help="Visitors that saw the treatment variant of experiment 1 and the control of experiment 2.")
        BB_u = st.number_input("BB Visitors", value=st.session_state.BB_u, help="Visitors that saw the treatment variant of both experiment 1 and 2.")

    with col2:
        st.write("### Conversions")
        AA_c = st.number_input("AA Conversions", value=st.session_state.AA_c, help="Conversions from visitors that saw the control of both experiment 1 and 2.")
        AB_c = st.number_input("AB Conversions", value=st.session_state.AB_c, help="Conversions from visitors that saw the control of experiment 1 and the treatment variant of experiment 2")
        BA_c = st.number_input("BA Conversions", value=st.session_state.BA_c, help="Conversions from visitors that saw the treatment variant of experiment 1 and the control of experiment 2.")
        BB_c = st.number_input("BB Conversions", value=st.session_state.BB_c, help="Conversions from visitors that saw the treatment variant of both experiment 1 and 2.")

    return AA_u, AB_u, BA_u, BB_u, AA_c, AB_c, BA_c, BB_c

def convert_data(AA_u, AB_u, BA_u, BB_u, AA_c, AB_c, BA_c, BB_c):
    """
    Optimized data conversion: Creates 8 rows representing the total 
    outcomes instead of millions of rows.
    """
    if AA_u > 0 and AB_u > 0 and BA_u > 0 and BB_u > 0:
        data = [
            {'Combination': 'AA', 'Conversion': 1, 'Count': AA_c},
            {'Combination': 'AA', 'Conversion': 0, 'Count': AA_u - AA_c},
            {'Combination': 'AB', 'Conversion': 1, 'Count': AB_c},
            {'Combination': 'AB', 'Conversion': 0, 'Count': AB_u - AB_c},
            {'Combination': 'BA', 'Conversion': 1, 'Count': BA_c},
            {'Combination': 'BA', 'Conversion': 0, 'Count': BA_u - BA_c},
            {'Combination': 'BB', 'Conversion': 1, 'Count': BB_c},
            {'Combination': 'BB', 'Conversion': 0, 'Count': BB_u - BB_c},
        ]
        df = pd.DataFrame(data)
        df['Combination'] = df['Combination'].astype('category')
        df = pd.get_dummies(df, columns=['Combination'], drop_first=True, dtype=int)

        for col in ['Combination_AB', 'Combination_BA', 'Combination_BB']:
            if col not in df.columns:
                df[col] = 0

        return df
    return None

def perform_logistic_regression(data_long):
    # Setup variables for GLM (using weights for performance)
    X = data_long[['Combination_AB', 'Combination_BA', 'Combination_BB']]
    y = data_long['Conversion']
    weights = data_long['Count']
    X = sm.add_constant(X)

    try:
        # GLM with Binomial family is equivalent to Logit but supports frequency weights
        model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights).fit()
        st.write(model.summary())

        # Extracting coefficients and p-values using summary2 format for your logic
        coefficients_table = model.summary2().tables[1]

        # Your Original Results Logic
        st.write("### Results summary")
        st.write("Below is an interpretation of the coefficients and p-values in the model. The focus lies on the interaction between both 'B' variants. " \
                 "If there is no measurable negative interaction in that group or other groups, you're safe to rely on individual test results for inference.")
        st.write("")
        
        bb_impact = False
        negative_impact_ab_ba = False
        no_significant_outcomes = True

        for combination in ['Combination_AB', 'Combination_BA', 'Combination_BB']:
            coef = coefficients_table.loc[combination, 'Coef.']
            p_value = coefficients_table.loc[combination, 'P>|z|']

            if combination == 'Combination_BB' and p_value < 0.05:
                bb_impact = True
                no_significant_outcomes = False
                st.write(f"Visitors that saw both test variants (BB) have a significant impact on conversion with a p-value of {p_value:.2e} and a coefficient of {coef:.4f}.")
                if coef < 0:
                    st.write("The coefficient is negative, indicating a potential negative interaction effect. Interpret your individual test results with caution.")
                else:
                    st.write("The coefficient is positive, indicating a potential positive interaction effect. Keeping both variants would likely be beneficial to user behavior.")
                st.write("")

            elif combination in ['Combination_AB', 'Combination_BA'] and p_value < 0.05 and coef < 0:
                negative_impact_ab_ba = True
                no_significant_outcomes = False
                st.write(f"Combination {combination} shows a significant negative impact with a p-value of {p_value:.2e} and a coefficient of {coef:.4f}.")
                st.write("While there is no significant impact on users who saw both variants (BB), you should interpret your test results with care.")
                st.write("")

        if no_significant_outcomes:
            st.write("No significant interaction effects were observed across the combinations. You can interpret the results of your experiments as usual.")
            st.write("")

        return model

    except Exception as e:
        st.write(f"Error fitting the model: {e}")
        return None

def visualize_results(model):
    predict_data = pd.DataFrame({
        'const': 1,
        'Combination_AB': [0, 1, 0, 0],
        'Combination_BA': [0, 0, 1, 0],
        'Combination_BB': [0, 0, 0, 1]
    })

    predict_data['prob'] = model.predict(predict_data)

    def y_fmt(x, _):
        return f'{x:.3f}'

    fig, ax = plt.subplots()
    ax.plot(['A', 'B'], [predict_data['prob'][0], predict_data['prob'][1]], label='Test1 A', marker='o', color='blue')
    ax.plot(['A', 'B'], [predict_data['prob'][2], predict_data['prob'][3]], label='Test1 B', marker='o', color='orange')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['A', 'B'])

    # 0.0 corresponds to 'A', 1.0 corresponds to 'B' - to avoid type errors
    ax.text(0.0, predict_data['prob'][0], 'AA', horizontalalignment='right', color='blue')
    ax.text(1.0, predict_data['prob'][1], 'AB', horizontalalignment='left', color='blue')
    ax.text(0.0, predict_data['prob'][2], 'BA', horizontalalignment='right', color='orange')
    ax.text(1.0, predict_data['prob'][3], 'BB', horizontalalignment='left', color='orange')

    ax.set_xlabel('Test 2 Level')
    ax.set_ylabel('Predicted Conversion Rate')
    ax.set_title('Interaction Effect of Test1 and Test2 on Conversion')
    ax.legend()
    ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))
    ax.grid(True)
    st.pyplot(fig)

def run():
    st.title("Interaction Analysis")
    st.markdown("""
    This calculator lets you see if your variants from two experiments that ran concurrently influenced each other. 
    Enter your data (AA, AB, BA, BB) below and the algorithm will determine if you should proceed with caution.
    """)

    AA_u, AB_u, BA_u, BB_u, AA_c, AB_c, BA_c, BB_c = user_input()

    st.write("")
    if st.button("Calculate interaction effect", type="primary"):
        # Check for valid inputs
        if AA_u > 0 and AB_u > 0 and BA_u > 0 and BB_u > 0:
            data_long = convert_data(AA_u, AB_u, BA_u, BB_u, AA_c, AB_c, BA_c, BB_c)
            model = perform_logistic_regression(data_long)
            if model:
                visualize_results(model)
        else:
            st.write("")
            st.error("Please enter valid inputs for all groups to begin")

if __name__ == "__main__":
    run()
