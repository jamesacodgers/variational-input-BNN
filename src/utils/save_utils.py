import pandas as pd
from datetime import datetime


def save_results_to_csv(model, input_weighted_model, variational_input_weighted_model, 
                       mse, lpd, iw_mse, iw_lpd, viw_mse, viw_lpd, 
                       filename='experiment_results.csv'):
    """
    Save experiment results to a CSV file.
    If the file exists, append the new results.
    """
    # Create a dictionary with the results
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'standard_mse': mse,
        'standard_lpd': lpd,
        'input_weighted_mse': iw_mse,
        'input_weighted_lpd': iw_lpd,
        'variational_input_weighted_mse': viw_mse,
        'variational_input_weighted_lpd': viw_lpd,
        'variational_mean': variational_input_weighted_model.mean.item(),
        'variational_std': variational_input_weighted_model.log_std.exp().item(),
        'standard_log_sigma2': model.log_sigma2.item(),
        'input_weighted_log_sigma2': input_weighted_model.log_sigma2.item(),
        'variational_log_sigma2': variational_input_weighted_model.log_sigma2.item(),
        'device': device.type
    }
    
    # Convert to DataFrame
    results_df = pd.DataFrame([results])
    
    try:
        # Try to read existing CSV
        existing_df = pd.read_csv(filename)
        # Append new results
        updated_df = pd.concat([existing_df, results_df], ignore_index=True)
    except FileNotFoundError:
        # If file doesn't exist, create new DataFrame
        updated_df = results_df
    
    # Save to CSV
    updated_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")