import streamlit as st

# --- Admin login in sidebar ---
with st.sidebar:
    if not st.session_state.get("admin_authenticated"):
        with st.expander("🔐 Admin"):
            pwd = st.text_input("Password", type="password", key="admin_pwd")
            if st.button("Unlock"):
                if pwd == st.secrets["ADMIN_PASSWORD"]:
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.error("Wrong password")
    else:
        st.caption("Logged in as admin")
        if st.button("Logout"):
            st.session_state.admin_authenticated = False
            st.rerun()

# --- Build page list conditionally ---
client_pages = [
    st.Page("pages/data_export.py",                 title="Data Export",                 icon="📤"),
    st.Page("pages/pre_test_analysis.py",           title="Pre-Test Analysis",           icon="🔬"),
    st.Page("pages/srm_calculator.py",              title="SRM Check",                   icon="⚖️"),
    st.Page("pages/experiment_analysis.py",         title="Experiment Analysis",         icon="🧪"),
    st.Page("pages/continuous_metric_analysis.py",  title="Continuous Metric Analysis",  icon="📐"),
    st.Page("pages/behavioral_analysis.py",         title="Behavioral Analysis",         icon="🧠"),
    st.Page("pages/interaction_analysis.py",        title="Interaction Analysis",        icon="🔀"),
    st.Page("pages/sequential_analysis.py",         title="Sequential Analysis",         icon="⏱️"),
    st.Page("pages/experimentation_growth.py",      title="Experimentation Growth",      icon="📈"),
]

admin_pages = [
    st.Page("pages/automation.py", title="Automation", icon="⚙️"),
]

pages = (admin_pages + client_pages) if st.session_state.get("admin_authenticated") else client_pages

pg = st.navigation(pages)
pg.run()

# --- Standard Hexkit flow ---

st.set_page_config(
    page_title="HEXKIT",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Main Page UI
logo_url = "https://raw.githubusercontent.com/BasLinders/hexkit/main/hexkit-logo-final_small.png"
st.sidebar.image(logo_url, width="stretch")

# st.title("HEXKIT")
# st.image(logo_url)
st.write("### <span style='color: orange;'>v1.5</span>", unsafe_allow_html=True)
st.write("""
This is the main page for the Happy Horizon Experimentation Toolkit. You can navigate to individual apps using the sidebar.

### What you're looking at
This toolkit has been created for the purposes of analyzing data from online controlled experiments ('A/B tests') to learn from and better understand user behavior.  

### Features
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Behavioral metric analysis**: Analyze higher-level metrics in your experiments<br>
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Continuous metric analysis**: Analyze metrics such as revenue / items per transaction<br>
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Data export**: Connect to BigQuery with OpenAuth and pull experiment data for analysis<br>
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Experiment analysis**: Use Frequentist and Bayesian methods to analyze test results<br>
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Experimentation growth estimation**: Calculate annual compound growth potential<br>
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Interaction analysis**: Verify if your experiments negatively impacted each other or not<br>
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Pre-test analysis**: Calculate the runtime to reach an effect<br>
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Sequential analysis**: Get instant feedback about test significance without fixed samples<br>
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**SRM calculator**: Identify if your visitors were distributed as expected in your experiment<br>

### How to Use
- Select a page from the sidebar to view different tools.
- Each page contains a single tool for the purposes described above.

### About
Happy Horizon is a creative digital agency of experts in strategic thinking, analysis, creativity, digital services and technology.
""", unsafe_allow_html=True)

linkedin_url = "https://www.linkedin.com/in/blinders/"
footnote_text = f"""Engineered and developed by <a href="{linkedin_url}" target="_blank">Bas Linders</a>"""
st.markdown(footnote_text, unsafe_allow_html=True)
