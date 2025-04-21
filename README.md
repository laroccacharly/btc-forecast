# Can Prophet Predict Bitcoin Prices? 🔮

This repository explores the use of Facebook's Prophet time series forecasting model to predict Bitcoin (BTC) prices. It provides a Streamlit web application that allows users to:

1.  📊 **View General Data:** Explore historical BTC price data, including descriptive statistics and plots (linear and log scale) with moving averages.
2.  📈 **Run a Baseline Forecast:** Fit a Prophet model directly to the raw price data with default parameters and visualize the forecast.
3.  🧮 **Run a Log-Transformed Forecast:** Fit a Prophet model to the logarithm of the price data (often better for financial time series).
4.  ⚙️ **Tune Hyperparameters:** Perform cross-validation to find optimal Prophet hyperparameters (`changepoint_prior_scale`, `seasonality_prior_scale`) for improved forecast accuracy.


References: 
- Prophet: https://facebook.github.io/prophet/
- BTC price history: https://github.com/laroccacharly/btc-price-history
- Yahoo finance: https://yfinance-python.org/