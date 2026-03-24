import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import uuid
from st_supabase_connection import SupabaseConnection

# --- CONSTANTS & CONFIGURATION ---
st.set_page_config(page_title="Sequential Analysis", layout="wide")

TEST_TYPE_ONE_SAMPLE = "One-sample (fixed baseline)"
TEST_TYPE_MULTI_SAMPLE = "Multi-sample (Control vs. Variants)"

# Initialize Supabase Connection
conn = st.connection("supabase", type=SupabaseConnection)

def is_valid_uuid(val):
    """Validates if a string is a properly formatted UUID."""
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False

# --- STATISTICAL FUNCTIONS ---

def calculate_msprt_boundaries(alpha, beta, num_variants=1):
    """Calculates boundaries for mSPRT."""
    upper = np.log(num_variants / alpha) 
    lower = np.log(beta) 
    return upper, lower

def calculate_msprt_llr(visitors_base, conversions_base, visitors_var, conversions_var, tau=0.01, fixed_baseline_cr=None):
    """
    Calculates the Log-Likelihood Ratio. 
    Handles both two-sample pooled variance and one-sample fixed variance.
    """
    if visitors_var == 0:
        return 0.0

    p_var = conversions_var / visitors_var

    if fixed_baseline_cr is not None:
        # ONE-SAMPLE: Variance of a single proportion
        p_base = fixed_baseline_cr
        var = p_var * (1 - p_var) / visitors_var
        
        if var == 0:
            return 0.0
        diff = p_var - p_base
    else:
        # TWO-SAMPLE: Pooled variance of the difference
        if visitors_base == 0:
            return 0.0
            
        p_base = conversions_base / visitors_base
        p_pool = (conversions_base + conversions_var) / (visitors_base + visitors_var)

        if p_pool <= 0 or p_pool >= 1:
            return 0.0

        var = p_pool * (1 - p_pool) * (1/visitors_base + 1/visitors_var)
        
        if var == 0:
            return 0.0
        diff = p_var - p_base

    # mSPRT LLR Formula
    llr = 0.5 * (np.log(var / (var + tau)) + (diff**2 / var) * (tau / (var + tau)))

    return llr

# --- DATABASE FUNCTIONS ---

def get_experiment_params(experiment_id):
    """Fetch setup parameters for an ID."""
    try:
        response = conn.table("experiment_params").select("*").eq("experiment_id", experiment_id).execute()
        if len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        st.error(f"Error fetching params: {e}")
        return None

def save_experiment_params(experiment_id, p0, tau, alpha, beta, max_visitors, test_type, num_variants):
    """Save the immutable rules of the experiment."""
    try:
        data = {
            "experiment_id": experiment_id,
            "p0": float(p0),
            "tau": float(tau),
            "alpha": float(alpha),
            "beta": float(beta),
            "max_visitors": int(max_visitors),
            "test_type": str(test_type),
            "num_variants": int(num_variants)
        }
        conn.table("experiment_params").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Error creating experiment: {e}")
        return False

@st.cache_data(ttl=60)
def get_experiment_data(experiment_id):
    """Fetches data in the LONG format (variant_name, visitors, conversions)."""
    try:
        response = conn.table("msprt_data").select("*").eq("experiment_id", experiment_id).order("measurement_date").execute()
        if len(response.data) > 0:
            df = pd.DataFrame(response.data)
            df['measurement_date'] = pd.to_datetime(df['measurement_date']).dt.date
            df['visitors'] = df['visitors'].fillna(0).astype(int)
            df['conversions'] = df['conversions'].fillna(0).astype(int)
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def save_data_points(experiment_id, date, variant_data_list):
    """Insert a batch of data points for a single date."""
    try:
        insert_payload = []
        for data in variant_data_list:
            insert_payload.append({
                "experiment_id": experiment_id,
                "measurement_date": str(date),
                "variant_name": data["variant_name"],
                "visitors": int(data["visitors"]),
                "conversions": int(data["conversions"])
            })
        
        conn.table("msprt_data").insert(insert_payload).execute()
        st.toast("Data points saved successfully!", icon="✅")
        get_experiment_data.clear() # Invalidate cache
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False

def delete_data_points_by_date(experiment_id, date):
    """Deletes all rows for a specific date."""
    try:
        conn.table("msprt_data").delete().eq("experiment_id", experiment_id).eq("measurement_date", str(date)).execute()
        st.toast(f"Entries for {date} deleted", icon="🗑️")
        get_experiment_data.clear() # Invalidate cache
        return True
    except Exception as e:
        st.error(f"Error deleting data: {e}")
        return False

# --- VISUALIZATIONS ---

def show_visualization(chart_df, upper_bound, lower_bound):
    if chart_df.empty:
        st.warning("No data to visualize yet.")
        return
        
    y_values = [chart_df['llr'].max(), chart_df['llr'].min(), upper_bound, lower_bound]
    
    # Additive padding handles negative numbers safely
    padding = (max(y_values) - min(y_values)) * 0.1
    if padding == 0: padding = 1.0 # Fallback
    
    max_y = max(y_values) + padding
    min_y = min(y_values) - padding

    line = alt.Chart(chart_df).mark_line(point=True).encode(
        x = alt.X('measurement_date:T', title='Date'),
        y = alt.Y('llr:Q', title='Log Likelihood Ratio', scale=alt.Scale(domain=[min_y, max_y])),
        color = alt.Color('variant_name:N', title='Variant')
    )

    success_zone = alt.Chart(pd.DataFrame({'y': [upper_bound], 'y2': [max_y]})).mark_rect(color='green', opacity=0.1).encode(y='y', y2='y2')
    futility_zone = alt.Chart(pd.DataFrame({'y': [min_y], 'y2': [lower_bound]})).mark_rect(color='red', opacity=0.1).encode(y='y', y2='y2')

    upper_line = alt.Chart(pd.DataFrame({'y': [upper_bound]})).mark_rule(color='green', strokeDash=[5,5]).encode(y='y')
    lower_line = alt.Chart(pd.DataFrame({'y': [lower_bound]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y')
    
    chart = (success_zone + futility_zone + upper_line + lower_line + line).properties(height=400).interactive()

    st.markdown("### Test Trajectory")
    st.altair_chart(chart, use_container_width=True)

# --- ANALYSIS ---

def analysis_section(df, params):
    st.divider()
    st.subheader("Sequential Analysis")

    alpha = params.get('alpha', 0.05)
    beta = params.get('beta', 0.20)
    tau_param = params.get('tau', 0.01)
    test_type = params.get('test_type', TEST_TYPE_ONE_SAMPLE)
    max_visitors = params.get('max_visitors', 10000)
    num_variants = params.get('num_variants', 1)

    upper_bound, lower_bound = calculate_msprt_boundaries(alpha, beta, num_variants=num_variants)
    
    variants_to_test = [v for v in df['variant_name'].unique() if v != "Control"]
    chart_data = []

    for variant in variants_to_test:
        var_df = df[df['variant_name'] == variant].copy()
        
        # Guard against duplicate dates
        var_df = var_df.groupby('measurement_date').last().reset_index()
        
        if test_type == TEST_TYPE_MULTI_SAMPLE:
            ctrl_df = df[df['variant_name'] == 'Control'].copy()
            ctrl_df = ctrl_df.groupby('measurement_date').last().reset_index()
            
            merged = pd.merge(var_df, ctrl_df, on='measurement_date', suffixes=('_var', '_ctrl'))
            
            if merged.empty:
                st.warning(f"Waiting for aligned Control & Variant dates for {variant}.")
                continue
                
            merged['llr'] = merged.apply(lambda row: calculate_msprt_llr(
                visitors_base=row['visitors_ctrl'], conversions_base=row['conversions_ctrl'],
                visitors_var=row['visitors_var'], conversions_var=row['conversions_var'], 
                tau=tau_param
            ), axis=1)
            
            latest_vis = merged.iloc[-1]['visitors_var']
            latest_conv = merged.iloc[-1]['conversions_var']
            base_vis = merged.iloc[-1]['visitors_ctrl']
            base_cr = merged.iloc[-1]['conversions_ctrl'] / base_vis if base_vis > 0 else 0

        else:
            # One-sample logic
            merged = var_df.copy()
            if merged.empty:
                continue
                
            p0_param = params.get('p0', 0.10)
            merged['llr'] = merged.apply(lambda row: calculate_msprt_llr(
                visitors_base=0, conversions_base=0, 
                visitors_var=row['visitors'], conversions_var=row['conversions'], 
                tau=tau_param, fixed_baseline_cr=p0_param 
            ), axis=1)
            
            latest_vis = merged.iloc[-1]['visitors']
            latest_conv = merged.iloc[-1]['conversions']
            base_cr = p0_param

        merged['variant_name'] = variant
        chart_data.append(merged[['measurement_date', 'variant_name', 'llr', 'visitors_var' if 'visitors_var' in merged.columns else 'visitors']])

        # --- VARIANT DECISION CARDS ---
        latest_llr = merged.iloc[-1]['llr']
        latest_cr = latest_conv / latest_vis if latest_vis > 0 else 0

        with st.expander(f"Metrics: {variant}", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Lower Bound (Futility)", f"{lower_bound:.2f}")
            col2.metric("Current LLR", f"{latest_llr:.2f}")
            col3.metric("Upper Bound (Success)", f"{upper_bound:.2f}")
            
            st.write(f"**Observed CR:** {latest_cr:.2%} vs **Baseline CR:** {base_cr:.2%}")
            
            # --- PROGRESS BAR ---
            progress = min(max((latest_llr / upper_bound) * 100, 0), 100) if latest_llr > 0 else 0
            st.write(f"**Progress to Decision Boundary:** {progress:.0f}%")
            st.progress(progress / 100)

            # --- TIME ESTIMATION ---
            if latest_llr > 0 and not (latest_llr > upper_bound or latest_llr < lower_bound):
                first_date = merged['measurement_date'].min()
                last_date = merged['measurement_date'].max()
                days_elapsed = max((last_date - first_date).days, 1)
                
                avg_daily_visitors = latest_vis / days_elapsed
                llr_per_vis = latest_llr / latest_vis
                remaining_llr = upper_bound - latest_llr
                
                if llr_per_vis > 0 and avg_daily_visitors > 0:
                    est_vis_needed = remaining_llr / llr_per_vis
                    est_days = est_vis_needed / avg_daily_visitors
                    st.info(f"**Rough Estimate:** Assuming linear growth, you need approx. **{est_vis_needed:.0f}** more visitors (**{est_days:.1f} days**) to reach success.")

            # --- DECISION LOGIC ---
            if latest_llr > upper_bound:
                st.success(f"**Result: SIGNIFICANT POSITIVE** - {variant} is superior. You can stop.")
            elif latest_llr < lower_bound:
                if latest_cr < base_cr:
                    st.error(f"**Result: SIGNIFICANT NEGATIVE** - {variant} is worse than baseline. Stop immediately.")
                else:
                    st.error(f"**Result: FUTILITY** - {variant} is unlikely to reach the target.")
            else:
                if latest_vis >= max_visitors:
                    st.warning("Maximum sample size reached without a decision.")
                else:
                    st.info("INCONCLUSIVE - Continue collecting data.")

    if chart_data:
        final_chart_df = pd.concat(chart_data)
        if 'visitors_var' in final_chart_df.columns:
            final_chart_df = final_chart_df.rename(columns={'visitors_var': 'visitors'})
        show_visualization(final_chart_df, upper_bound, lower_bound)

# --- DOCUMENTATION ---
def show_documentation():
    with st.expander("Benefits of mSPRT", expanded=False):
        st.markdown("""
        #### Benefits of mSPRT
        Compared to fixed-horizon testing, SPRT has certain advantages.
        * **Stop winners early:** Deploy successful features days or weeks faster.
        * **Cut losers fast:** Identify "futility" (no chance of winning) early to save traffic and/or money.
        * **Rigorous:** Mathematically valid stopping rules, unlike standard z-tests (The mSPRT boundaries are "always-valid.").
        """)
    with st.expander("When to use mSPRT", expanded=False):
        st.markdown("""
        ### When to use mSPRT
        Sequential probability ratio testing is an agile tool. You should use it when:
        * **Safety is a concern:** You want to kill a 'losing' experiment immediately if it's tanking metrics.
        * **The observed effect is huge:** If the new feature is a massive success, SPRT will let you ship it in (for example) 3 days instead of 14.
        * **The cost of testing is high:** If every user in the experiment costs money, stopping early saves budget.
        * **If you don't believe in p-values:** This tool looks for the Log-Likelihood ratio to cross upper- and lower boundaries, calculated from your alpha and beta values (still technically frequentist, but practical).
    
        ### When to use fixed-horizon testing
        * If you have a **strict deadline**.
        * SPRT has a **slightly wider confidence interval** and will thus overestimate the treatment effect.
        * If you're looking for a **tiny lift** (e.g. 0.5% increase) it might take longer to reach a conclusion than a fixed-horizon test as in this case, SPRT has generally less statistical power.
        * If stakeholders are better aligned with clear deadlines and mid- to long-term planning of experiments.
        """)
    with st.expander("How to use mSPRT", expanded=False):
        st.markdown("""
        #### How to use mSPRT
        1.  **Start New:** Generate a unique ID and define your success metrics (Alpha / significance, Beta / power). 
            * *Note: These are locked once the test starts to ensure integrity.*
        2.  **Update Regularly:** Come back daily/weekly to input your **cumulative** data.
        3.  **Check the Graph:** * **Upper Limit:** Success! (Reject Null)
            * **Lower Limit:** Futility/Failure. (Accept Null)
            * **In Between lines:** Inconclusive - keep testing.
        
        > * **Important:** Data is stored for **42 days (6 weeks)** and then automatically deleted.
        > * **Save your Experiment ID!** It is the only key to retrieve your data.
        """)

# --- USER INPUT ---

def setup_sidebar(defaults, is_locked):
    with st.sidebar:
        st.header("1. Experiment Setup")

        # 1. Test Type Selection
        options = [TEST_TYPE_ONE_SAMPLE, TEST_TYPE_MULTI_SAMPLE]
        saved_type = defaults.get('test_type', options[0])
        default_index = options.index(saved_type) if saved_type in options else 0

        test_type = st.radio("Test format", options, index=default_index, disabled=is_locked)
        
        if test_type == TEST_TYPE_MULTI_SAMPLE:
            num_variants_val = int(defaults.get('num_variants', 1))
            num_variants = st.number_input("Number of Variants (excluding Control)", min_value=1, max_value=10, value=num_variants_val, step=1, disabled=is_locked)
            p0_label = "Estimated baseline CR (p0)"
            p0_help = "Used only for sample size estimates. Control CR will be measured live."
        else:
            num_variants = 1 
            p0_label = "Baseline CR (p0)"
            p0_help = "The fixed historical conversion rate (CR) you want to beat."

        # 2. Mode Selection
        mode = st.radio("Mode", ["Load Existing", "Start New"], label_visibility="collapsed")
        
        # 3. ID Management
        if mode == "Start New":
            if st.button("Generate New ID"):
                st.session_state['exp_id'] = str(uuid.uuid4())
                st.session_state['params_locked'] = False
                st.session_state['fetched_params'] = {}
                st.rerun()
            
            if st.session_state.get('exp_id'):
                st.success("Save this ID to load your test later:")
                st.code(st.session_state['exp_id']) 
        else: 
            input_id = st.text_input("Paste Experiment UUID")
            if st.button("Load"):
                if is_valid_uuid(input_id):
                    st.session_state['exp_id'] = input_id
                    params = get_experiment_params(input_id)
                    if params:
                        st.session_state['fetched_params'] = params
                        st.session_state['params_locked'] = True
                        st.toast("Parameters loaded and locked!", icon="🔒")
                    else:
                        st.error("Experiment ID not found or no parameters set.")
                    st.rerun()
                else:
                    st.error("Invalid UUID format.")

        st.divider()

        # --- PARAMETER INPUTS ---
        st.subheader("2. Test Parameters")

        if is_locked:
            st.info("Parameters are locked for this ID.")

        p0_param = float(defaults.get('p0', 0.10))

        with st.form("setup_form"):
            tau_val = float(defaults.get('tau') or 0.01)
            alpha_val = float(defaults.get('alpha', 0.05))
            beta_val = float(defaults.get('beta', 0.20))
            max_visitors_val = int(defaults.get('max_visitors', 10000))

            if test_type == TEST_TYPE_ONE_SAMPLE:
                p0_param = st.number_input(p0_label, value=p0_param, format="%.4f", disabled=is_locked, help=p0_help)
            else:
                st.caption(f"**{p0_label}:** Not required for strict calculations in Multi-sample mode.")
                p0_param = 0.01

            tau_param = st.select_slider(
                "Test sensitivity (Tau)",
                options=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
                value=tau_val,
                help="Lower values (0.001) are more conservative. Higher values (0.05) detect large effects faster.",
                disabled=is_locked
            )
            max_visitors = st.number_input("Max Visitors (Safety Cap)", value=max_visitors_val, step=100, disabled=is_locked)
            
            c1, c2 = st.columns(2)
            alpha = c1.number_input("Alpha", value=alpha_val, step=0.01, disabled=is_locked, help="The risk of a False Positive.")
            beta = c2.number_input("Beta", value=beta_val, step=0.01, disabled=is_locked, help="The risk of a False Negative.")

            if not is_locked:
                submitted = st.form_submit_button("Start & Lock Experiment")
                if submitted:
                    if not st.session_state.get('exp_id'):
                        st.error("Generate an ID first!")
                    elif tau_param <= 0:
                        st.error("Sensitivity (Tau) must be greater than 0.")
                    else:
                        saved = save_experiment_params(st.session_state['exp_id'], p0_param, tau_param, alpha, beta, max_visitors, test_type, num_variants)
                        if saved:
                            st.session_state['params_locked'] = True
                            st.session_state['fetched_params'] = {
                                'p0': p0_param, 'tau': tau_param, 'alpha': alpha, 'beta': beta, 
                                'max_visitors': max_visitors, 'test_type': test_type, 'num_variants': num_variants
                            }
                            st.rerun()
            else:
                st.form_submit_button("Parameters Locked", disabled=True)
                
    return st.session_state.get('fetched_params', {})

def render_data_entry_form(exp_id, df, params):
    st.subheader("Update Data")
    st.info("💡 **Reminder:** Enter the **cumulative** totals up to this date, not just the daily increment.")
    
    current_test_type = params.get('test_type', TEST_TYPE_ONE_SAMPLE)
    num_variants = params.get('num_variants', 1)

    with st.form("entry_form"):
        d_date = st.date_input("Date")
        variant_data_list = []

        def get_prev(v_name):
            if not df.empty:
                v_df = df[df['variant_name'] == v_name]
                if not v_df.empty:
                    return int(v_df.iloc[-1]['visitors']), int(v_df.iloc[-1]['conversions'])
            return 0, 0

        if current_test_type == TEST_TYPE_MULTI_SAMPLE:
            st.divider()
            st.markdown("### Control Group")
            p_vis_c, p_conv_c = get_prev("Control")
            c1, c2 = st.columns(2)
            d_vis_c = c1.number_input(f"Cumulative Visitors (Prev: {p_vis_c})", min_value=p_vis_c, value=p_vis_c, key="ctrl_v")
            d_conv_c = c2.number_input(f"Cumulative Conversions (Prev: {p_conv_c})", min_value=p_conv_c, value=p_conv_c, key="ctrl_c")
            variant_data_list.append({"variant_name": "Control", "visitors": d_vis_c, "conversions": d_conv_c})
            
            st.markdown("### Variant Groups")
            for i in range(1, num_variants + 1):
                v_name = f"Variant {i}"
                p_vis, p_conv = get_prev(v_name)
                c3, c4 = st.columns(2)
                d_vis = c3.number_input(f"{v_name} Visitors (Prev: {p_vis})", min_value=p_vis, value=p_vis, key=f"v{i}_v")
                d_conv = c4.number_input(f"{v_name} Conversions (Prev: {p_conv})", min_value=p_conv, value=p_conv, key=f"v{i}_c")
                variant_data_list.append({"variant_name": v_name, "visitors": d_vis, "conversions": d_conv})

        else: # One-sample
            st.divider()
            st.markdown("### Variant Data")
            p_vis, p_conv = get_prev("Variant 1")
            c1, c2 = st.columns(2)
            d_vis = c1.number_input(f"Cumulative Visitors (Prev: {p_vis})", min_value=p_vis, value=p_vis, key="1s_v")
            d_conv = c2.number_input(f"Cumulative Conversions (Prev: {p_conv})", min_value=p_conv, value=p_conv, key="1s_c")
            variant_data_list.append({"variant_name": "Variant 1", "visitors": d_vis, "conversions": d_conv})

        st.divider()
        if st.form_submit_button("Add Data Point"):
            if any(item['visitors'] < item['conversions'] for item in variant_data_list):
                st.error("Visitors cannot be less than conversions for any group.")
            else:
                save_data_points(exp_id, d_date, variant_data_list)
                st.rerun()

# --- UI LOGIC / ORCHESTRATION ---

def run():
    st.title("Sequential Experiment Analysis (SPRT)")
    st.markdown("""
    ### Faster A/B Testing with Sequential Analysis
    Standard A/B tests require you to wait for a fixed sample size to avoid "peeking" errors. 
    **This tool is different.** It uses **mixture Sequential Probability Ratio Testing (mSPRT)**, allowing you to update data and check results **any time** without invalidating your statistics.
    """)

    show_documentation()
    
    if 'params_locked' not in st.session_state: st.session_state['params_locked'] = False
    if 'fetched_params' not in st.session_state: st.session_state['fetched_params'] = {}

    defaults = st.session_state.get('fetched_params', {})
    is_locked = st.session_state.get('params_locked', False)
    
    # --- SIDEBAR ---
    params = setup_sidebar(defaults, is_locked)

    # --- MAIN CONTENT ---
    exp_id = st.session_state.get('exp_id')
    
    if not exp_id or not st.session_state.get('params_locked'):
        st.info("**To Begin:** Select 'Start New' to generate an ID and lock your parameters.")
        st.stop()

    st.markdown(f"### Experiment: `{exp_id}`")
    
    df = get_experiment_data(exp_id)
    
    # --- DATA ENTRY ---
    render_data_entry_form(exp_id, df, params)

    # --- UNDO FUNCTIONALITY ---
    if not df.empty:
        last_date = df['measurement_date'].max()
        with st.expander("Danger Zone: Undo Entries"):
            st.warning("Deleting data cannot be undone.")
            if st.button(f"Delete ALL variant entries for {last_date}", type="primary"):
                delete_data_points_by_date(exp_id, last_date)
                st.rerun()
    
    # --- ANALYSIS SECTION ---
    if not df.empty:
        analysis_section(df, params)
            
        with st.expander("View Raw Data"):
            st.dataframe(df.sort_values("measurement_date", ascending=False))
    else:
        st.write("Waiting for data entries...")

if __name__ == "__main__":
    run()