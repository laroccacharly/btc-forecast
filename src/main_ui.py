import streamlit as st
from .log_forecast import log_forecast
from .general_view import general_view
from .baseline import baseline

def main_ui():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["General View", "Baseline Model", "Log Forecast Model"], index=2)

    if page == "General View":
        general_view()
    elif page == "Baseline Model":
        baseline()
    elif page == "Log Forecast Model":
        log_forecast()