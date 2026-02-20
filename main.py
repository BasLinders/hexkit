import streamlit as st

st.set_page_config(
    page_title="HEXKIT",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Main Page UI

st.title("HEXKIT")
st.write("### <span style='color: orange;'>v0.7.2-beta</span>", unsafe_allow_html=True)
st.write("""
This is the main page for the Happy Horizon Experimentation Toolkit. You can navigate to individual apps using the sidebar.

### What you're looking at
This toolkit has been created for the purposes of analyzing data from online controlled experiments ('A/B tests') to learn from and better understand user behavior.  

### Features
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Behavioral metric analysis**: Analyze higher-level metrics in your experiments<br>
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Experiment analysis**: Use Frequentist and Bayesian methods to analyze test results<br>
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Continuous metric analysis**: Analyze metrics such as revenue / items per transaction<br>
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Experimentation growth estimation**: Calculate annual compound growth potential for experimentation<br>
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Interaction analysis**: Verify if your experiments negatively impacted each other or not<br>
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Sequential analysis**: Get instant feedback about test significance without the need for fixed sample sizes<br>
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**SRM calculator**: Identify if your visitors were distributed as expected in your experiment<br>
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Pre-test analysis**: Calculate the runtime to reach an effect

### How to Use
- Select a page from the sidebar to view different tools.
- Each page contains a single tool for the purposes described above.

### About
Happy Horizon is a creative digital agency of experts in strategic thinking, analysis, creativity, digital services and technology.
""", unsafe_allow_html=True)

linkedin_url = "https://www.linkedin.com/in/blinders/"
happyhorizon_url = "https://happyhorizon.com/"
footnote_text = f"""Engineered and developed by <a href="{linkedin_url}" target="_blank">Bas Linders</a>"""
st.markdown(footnote_text, unsafe_allow_html=True)
