import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def plot_metrics(csv_file='experiment_results_1000.csv'):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, figsize=(15, 10))
    fig.suptitle('Experimental Results Comparison', fontsize=16)
    
    # Plot MSE comparison
    mse_data = pd.melt(df, 
                       value_vars=['standard_mse', 'input_weighted_mse', 'variational_input_weighted_mse'],
                       var_name='Model', value_name='MSE')
    sns.boxplot(data=mse_data, x='Model', y='MSE', ax=axes[0])
    axes[0].set_title('MSE Comparison')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot LPD comparison
    lpd_data = pd.melt(df, 
                       value_vars=['standard_lpd', 'input_weighted_lpd', 'variational_input_weighted_lpd'],
                       var_name='Model', value_name='LPD')
    sns.boxplot(data=lpd_data, x='Model', y='LPD', ax=axes[1])
    axes[1].set_title('Log Predictive Density Comparison')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot log_sigma2 comparison
    
    # Adjust layout and save
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_metrics()