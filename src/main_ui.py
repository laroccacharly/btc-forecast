import streamlit as st
from .log_forecast import log_forecast
from .general_view import general_view
from .baseline import baseline
from .hypertune import hypertune_app

def main_ui():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["General View", "Baseline Model", "Log Forecast Model", "Hyperparameter Tuning"], index=2)

    if page == "General View":
        general_view()
    elif page == "Baseline Model":
        baseline()
    elif page == "Log Forecast Model":
        log_forecast()
    elif page == "Hyperparameter Tuning":
        hypertune_app()