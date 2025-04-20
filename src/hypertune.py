import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
import logging
from .data import load_data # Changed to relative import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the parameter grid - Based on Prophet documentation recommendations
param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

# Metrics to calculate (use mean across horizon)
METRICS = ['mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage']

@st.cache_data # Cache the tuning results
def run_hyperparameter_tuning(df_log):
    """
    Performs hyperparameter tuning for the Prophet model using cross-validation.
    Calculates multiple performance metrics.

    Args:
        df_log (pd.DataFrame): DataFrame with 'ds' and log-transformed 'y' columns.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with parameters and their corresponding mean metrics.
            - dict: The best parameter combination found (based on lowest mean RMSE).
            - float: The lowest mean RMSE achieved.
    """
    results = []
    total_combinations = len(all_params)
    # Status updates need to be handled carefully with caching and button clicks
    # We'll display progress within the function but the final status outside
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Starting hyperparameter tuning for {total_combinations} combinations...") 

    for i, params in enumerate(all_params):
        # Note: status_text updates here will only be visible during the cached run
        status_text.text(f"Testing combination {i+1}/{total_combinations}: {params}")
        logging.info(f"Testing parameters: {params}")
        
        current_metrics = {'params': params}
        for metric in METRICS:
            current_metrics[metric] = float('inf')

        try:
            m = Prophet(**params).fit(df_log)
            df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days', parallel="processes", disable_tqdm=True)
            df_p = performance_metrics(df_cv, metrics=METRICS, rolling_window=0.1)

            if not df_p.empty:
                for metric in METRICS:
                    if metric in df_p.columns and not df_p[metric].isnull().all():
                        current_metrics[metric] = df_p[metric].mean()
                    else: 
                        current_metrics[metric] = float('inf')
            
            logging.info(f"Parameters: {params} - Metrics: { {k: f'{v:.4f}' if isinstance(v, float) else v for k, v in current_metrics.items() if k != 'params'} }")

        except Exception as e:
            logging.error(f"Failed for parameters {params}: {e}")

        results.append(current_metrics)
        progress_bar.progress((i + 1) / total_combinations)

    # Clear the progress/status text after completion
    status_text.text("Hyperparameter tuning function complete.")
    progress_bar.empty()
    
    results_df = pd.DataFrame(results)
    for metric in METRICS:
        results_df[metric] = pd.to_numeric(results_df[metric], errors='coerce').fillna(float('inf'))
         
    results_df = results_df.sort_values(by='rmse')
    
    if not results_df.empty and results_df.iloc[0]['rmse'] != float('inf'):
        best_params = results_df.iloc[0]['params']
        best_rmse = results_df.iloc[0]['rmse']
        logging.info(f"Best Parameters (based on RMSE): {best_params} with RMSE: {best_rmse:.4f}")
    else:
        best_params = "N/A"
        best_rmse = float('inf')
        logging.warning("Could not find best parameters. All runs may have failed or produced invalid metrics.")
    
    return results_df, best_params, best_rmse

def hypertune_app(): 
    st.title("Prophet Hyperparameter Tuning for BTC Log Forecast")

    st.write("Loading and preparing data...")
    try:
        df = load_data()
        df = df.copy()
        df = df.rename(columns={'price': 'y', 'timestamp': 'ds'})
        df['y_orig'] = df['y'] 
        df['y'] = np.log(df['y'])
        st.write("Data loaded successfully. Using log-transformed prices.")
        data_loaded = True
        df_subset = df[['ds', 'y']] # Prepare data subset for tuning function
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        data_loaded = False
        df_subset = None

    if data_loaded:
        st.subheader("Parameter Grid")
        st.json(param_grid)
        
        st.subheader("Start Cross-Validation")
        st.info("This process computes multiple metrics (MSE, RMSE, MAE, MAPE, MDAPE, SMAPE, Coverage) across several parameter combinations and can take several minutes.")
        
        # Add the button
        start_tuning = st.button("Start Hyperparameter Tuning")

        # Only run tuning and display results if the button is clicked
        if start_tuning and df_subset is not None:
            with st.spinner("Running cross-validation... Please wait."):
                results_df, best_params, best_rmse = run_hyperparameter_tuning(df_subset)

            st.subheader("Tuning Results (Sorted by RMSE)")
            
            results_display = results_df.copy()
            if not results_display.empty and 'params' in results_display.columns and isinstance(results_display['params'].iloc[0], dict):
                params_df = results_display['params'].apply(pd.Series)
                results_display = pd.concat([params_df, results_display.drop(['params'], axis=1)], axis=1)
                 
            cols_order = list(param_grid.keys()) + METRICS
            cols_order = [col for col in cols_order if col in results_display.columns] 
            if cols_order: # Only proceed if columns exist
                results_display = results_display[cols_order]
                
                float_cols = results_display.select_dtypes(include=['float']).columns
                format_dict = {col: '{:.4f}' for col in float_cols}
                
                st.dataframe(results_display.style.format(format_dict))
            else:
                 st.warning("Result columns could not be prepared for display.")   

            st.subheader("Best Parameters Found (based on lowest RMSE)")
            if best_rmse != float('inf') and best_params != "N/A":
                st.success(f"Lowest Mean RMSE: {best_rmse:.4f}")
                st.json(best_params)
            else:
                st.warning("Hyperparameter tuning failed to find optimal parameters based on RMSE.")
        elif start_tuning and df_subset is None:
             st.error("Data could not be prepared for tuning.")
# Removed the if __name__ == "__main__": block 