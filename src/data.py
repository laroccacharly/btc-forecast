import pandas as pd
import streamlit as st

@st.cache_data
def load_data(): 
    url = "https://github.com/laroccacharly/btc-price-history/raw/refs/heads/main/btc_price_history.parquet"
    df = pd.read_parquet(url)
    df = df.sort_values('timestamp') # Ensure data is sorted chronologically
    return df
