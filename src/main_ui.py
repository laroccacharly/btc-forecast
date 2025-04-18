import streamlit as st
from src.general_view import general_view
from src.baseline import baseline

def main_ui():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["General View", "Baseline Model"])

    if page == "General View":
        general_view()
    elif page == "Baseline Model":
        baseline()