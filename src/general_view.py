import pandas as pd
import streamlit as st
import plotly.express as px

@st.cache_data
def load_data(): 
    url = "https://github.com/laroccacharly/btc-price-history/raw/refs/heads/main/btc_price_history.parquet"
    df = pd.read_parquet(url)
    df = df.sort_values('timestamp') # Ensure data is sorted chronologically
    return df


def general_view(): 
    df = load_data()
    print(df.head())
    st.title('General View of the Data')
    
    st.subheader('Dataset Information')
    st.write(df.describe())

    st.subheader('First 5 rows of the dataset')
    st.write(df.head())

    st.subheader('Price Over Time')
    # Calculate 20-day moving average
    df['MA200'] = df['price'].rolling(window=200).mean()

    fig = px.line(df, x='timestamp', y='price', title='BTC Price Over Time')
    # Add the moving average trace
    fig.add_scatter(x=df['timestamp'], y=df['MA200'], mode='lines', name='200-Day MA')
    st.plotly_chart(fig)

    st.subheader('Log Price Over Time')
    fig_log = px.line(df, x='timestamp', y='price', title='BTC Price Over Time (Log Scale)', log_y=True)
    # Add the moving average trace to the log chart
    fig_log.add_scatter(x=df['timestamp'], y=df['MA200'], mode='lines', name='200-Day MA')
    st.plotly_chart(fig_log)

    
