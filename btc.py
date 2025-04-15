import pandas as pd
from prophet import Prophet
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt

def analyze_btc_history(file_path='btc_history.csv'):
    """
    Analyze Bitcoin price history data from a CSV file and perform Prophet forecasting.
    
    Args:
        file_path (str): Path to the CSV file containing Bitcoin price history data
        
    Returns:
        tuple: (original_df, forecast_df) - The original and forecasted DataFrames
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Display basic information about the dataset
        print("\nDataset Info:")
        print(df.info())
        
        # Display statistical description of the dataset
        print("\nStatistical Description:")
        print(df.describe())
        
        # Display the first few rows
        print("\nFirst few rows of the dataset:")
        print(df.head())
        
        # Prepare data for Prophet
        prophet_df = df[['Date', 'Price']].copy()
        # Remove any commas from Price column and convert to float
        prophet_df['Price'] = prophet_df['Price'].str.replace(',', '').astype(float)
        # Rename columns to Prophet requirements
        prophet_df = prophet_df.rename(columns={'Date': 'ds', 'Price': 'y'})
        
        # Initialize and fit the Prophet model
        print("\nInitializing and fitting Prophet model...")
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        model.fit(prophet_df)
        
        # Create future dates for forecasting (365 days)
        future = model.make_future_dataframe(periods=365)
        
        # Make predictions
        print("\nGenerating forecast...")
        forecast = model.predict(future)
        
        # Display forecast results
        print("\nForecast tail:")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        
        # Generate and save plots
        print("\nGenerating and saving forecast plots...")
        fig1 = model.plot(forecast)
        plt.savefig('btc_forecast_plot.png')
        plt.close(fig1)
        
        fig2 = model.plot_components(forecast)
        plt.savefig('btc_components_plot.png')
        plt.close(fig2)
        
        return df, forecast
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Execute the analysis when the script is run directly
    analyze_btc_history()
