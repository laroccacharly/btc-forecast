import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import os
import matplotlib.pyplot as plt

def load_or_download_data(url, local_path):
    """Loads data from a local CSV file or downloads it if it doesn't exist."""
    if not os.path.exists(local_path):
        print(f"Downloading data from {url}...")
        df = pd.read_csv(url)
        df.to_csv(local_path, index=False)
        print(f"Data saved locally to {local_path}")
    else:
        print(f"Loading data from local file: {local_path}")
        df = pd.read_csv(local_path)
    return df

def run_prophet_forecast():
    csv_url = 'https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv'
    local_csv_path = 'peyton_manning.csv'

    # Load or download the data
    df = load_or_download_data(csv_url, local_csv_path)

    # Initialize the model
    m = Prophet()

    # Fit the model
    print("\nFitting the model...")
    m.fit(df)
    print("Model fitting complete.")

    # --- Cross Validation --- #
    print("\nRunning Cross Validation...")
    df_cv = cross_validation(m, initial='1825 days', period='180 days', horizon = '365 days')
    print("Cross Validation complete. CV Results head:")
    print(df_cv.head())

    print("\nCalculating performance metrics...")
    df_p = performance_metrics(df_cv)
    print("Performance Metrics head:")
    print(df_p.head())

    print("\nGenerating and saving CV plot (RMSE)...")
    fig_cv = plot_cross_validation_metric(df_cv, metric='rmse')
    plt.savefig('cv_rmse_plot.png')
    print("CV RMSE plot saved to cv_rmse_plot.png")
    plt.close(fig_cv)
    # --- End Cross Validation --- #

    # Create a future dataframe for 365 days for the final forecast
    future = m.make_future_dataframe(periods=365)
    print("\nFuture dataframe tail:")
    print(future.tail())

    # Make predictions
    print("\nMaking final forecast...")
    forecast = m.predict(future)
    print("Final forecast complete.")

    # Display the forecast
    print("\nForecast tail:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Generate and save plots using Matplotlib
    print("\nGenerating and saving forecast plots using Matplotlib...")
    fig1 = m.plot(forecast)
    plt.savefig('forecast_plot.png') # Use plt.savefig
    print("Forecast plot saved to forecast_plot.png")
    plt.close(fig1) # Close the figure

    fig2 = m.plot_components(forecast)
    plt.savefig('components_plot.png') # Use plt.savefig
    print("Components plot saved to components_plot.png")
    plt.close(fig2) # Close the figure

def main():
    run_prophet_forecast()


if __name__ == "__main__":
    main()
