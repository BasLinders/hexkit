import streamlit as st

# Conditional presence
if not st.session_state.get("admin_authenticated"):
    st.error("Access denied.")
    st.stop()

# HELPERS

def run()
  pass

if __name__ == "__main__":
  run()
