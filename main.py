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
    st.Page("pages/home.py",                        title="Home",                        icon="🏠"),
    st.Page("pages/data_export.py",                 title="Data Export",                 icon="📤"),
    st.Page("pages/pre_test_analysis.py",           title="Pre-Test Analysis",           icon="🔬"),
    st.Page("pages/srm_calculator.py",              title="SRM Check",                   icon="⚖️"),
    st.Page("pages/experiment_analysis.py",         title="Experiment Analysis",         icon="🧪"),
    st.Page("pages/user_level_analysis.py",         title="User Level Analysis",         icon="📐"),
    st.Page("pages/behavioral_analysis.py",         title="Behavioral Analysis",         icon="🧠"),
    st.Page("pages/interaction_analysis.py",        title="Interaction Analysis",        icon="🔀"),
    st.Page("pages/sequential_analysis.py",         title="Sequential Analysis",         icon="⏱️"),
]

is_admin = st.session_state.get("admin_authenticated")

# Admin pages stay routable even when logged out — Google's OAuth redirect
# ends the WebSocket session and wipes session_state, including
# admin_authenticated, so the callback must still be able to land on
# /automation. The page itself re-checks admin_authenticated and blocks.
admin_visibility = "visible" if is_admin else "hidden"
admin_pages = [
    st.Page("pages/automation.py", title="Automation", icon="⚙️", visibility=admin_visibility),
    st.Page("pages/experimentation_growth.py", title="Experimentation Growth", icon="📈", visibility=admin_visibility),
]

pages = {"": client_pages, "Admin": admin_pages}

pg = st.navigation(pages)
pg.run()
