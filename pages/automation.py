import streamlit as st

from bq_ui_components import render_gcp_credentials_gate

# Conditional presence
if not st.session_state.get("admin_authenticated"):
    st.error("Access denied.")
    st.stop()


def run():
    st.set_page_config(
        page_title="Automation",
        page_icon="⚙️",
        layout="wide",
    )

    st.title("Automation")
    st.caption("Automated BigQuery workflows.")

    # --- GCP credentials + Google sign-in -------------------------------------
    st.divider()
    if not render_gcp_credentials_gate("automation"):
        st.stop()

    # --- Automation content ---------------------------------------------------
    st.divider()
    st.info("Page for the automation flow. Coming soon.")


if __name__ == "__main__":
    run()
