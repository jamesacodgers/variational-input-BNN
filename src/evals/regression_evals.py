import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Make predictions with uncertainty
def predict(model, x_test, n_samples=500):
    model.eval()
    preds = []
    
    for _ in range(n_samples):
        with torch.no_grad():
            y_pred = model(x_test)
            preds.append(y_pred)
    
    # Stack predictions
    preds = torch.stack(preds, dim=0)
    
    # Compute mean and std
    mean = preds.mean(dim=0)
    std = preds.std(dim=0)
    
    return mean, std

# Visualize the results
def plot_results(models, x_train, y_train, x_test, y_test, losses_list, model_names=None, var_dist=None, hidden_size="not_given"):
    # Create a figure with 2 subplots, sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[4, 1], sharex=True)
    plt.subplots_adjust(hspace=0.05)  # Reduce space between subplots
    
    # Create a grid of x values for plotting
    x_grid = torch.linspace(-3, 3, 100).reshape(-1, 1).to(device)
    
    # Convert to numpy arrays for training and test data
    x_train_np = x_train.cpu() if device.type == 'cuda' else x_train
    y_train_np = y_train.cpu() if device.type == 'cuda' else y_train
    x_test_np = x_test.cpu() if device.type == 'cuda' else x_test
    y_test_np = y_test.cpu() if device.type == 'cuda' else y_test
    x_grid_np = x_grid.cpu() if device.type == 'cuda' else x_grid
    
    # Compute the true function values
    y_true = true_function(x_grid_np)
    with torch.no_grad():
            
        # Plot the training and test data
        ax1.scatter(x_train_np, y_train_np, alpha=0.4, color='gray', label='Training data')
        ax1.scatter(x_test_np, y_test_np, alpha=0.4, marker='x', color='black', label='Test data')
        
        # Plot true function
        ax1.plot(x_grid_np, y_true, 'k--', linewidth=2, label='True function')
        
        # Colors for different models
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        # If model_names not provided, create generic names
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(models))]
        
        # Plot each model's predictions
        for i, model in enumerate(models):
            color = colors[i % len(colors)]
            
            # Predict with the model
            mean, std = predict(model, x_grid)
            mean_np = mean.cpu().numpy() if device.type == 'cuda' else mean.numpy()
            std_np = std.cpu().numpy() if device.type == 'cuda' else std.numpy()
            
            # Plot model prediction with uncertainty
            ax1.plot(x_grid_np, mean_np, '-', color=color, linewidth=2, label=f'{model_names[i]} mean')
            ax1.fill_between(x_grid_np.flatten(), 
                            (mean_np - 2*std_np).flatten(), 
                            (mean_np + 2*std_np).flatten(), 
                            color=color, alpha=0.2, label=f'{model_names[i]} 95% CI')
        
        ax1.set_ylabel('y')
        ax1.set_title(f'Comparison of BNN Models \n hidden size = {hidden_size}')
        ax1.legend(loc='upper right', fontsize='small')
        ax1.grid(True)
    
    # Plot the input density on the lower subplot
    # Use histogram of training data
        ax2.hist(x_train_np, bins=20, density=True, alpha=0.5, color='blue', label='Training data')
    
    # Plot the actual N(0,1) density function
        x_range = torch.linspace(-3, 3, 500, device=device)
        pdf = 1/np.sqrt(2*np.pi) * torch.exp(-0.5 * x_range**2)  # Standard normal PDF
        pdf_np = pdf.cpu().numpy() if device.type == 'cuda' else pdf.numpy()
        x_range_np = x_range.cpu().numpy() if device.type == 'cuda' else x_range.numpy()

        ax2.plot(x_range_np, pdf_np, 'k-', label='N(0,1) density')
        if var_dist is not None:
            var_pdf = var_dist.log_prob(x_range).exp()
            var_pdf_np = var_pdf.cpu().numpy() if device.type == 'cuda' else var_pdf.numpy()
            ax2.plot(x_range_np, var_pdf_np, 'r-', label='Variational density')
        
        # Mark test data points on x-axis
        ax2.scatter(x_test_np, np.zeros_like(x_test_np) + 0.02, marker='|', 
                    color='black', s=20, label='Test points')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('Density')
    ax2.legend(loc='upper right', fontsize='small')
    ax2.grid(True)
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig.savefig(f'figs/{hidden_size}_comparison_plot_{timestamp}.png', dpi=300, bbox_inches='tight')

# Calculate evaluation metrics
def calculate_metrics(model, x_test, y_test, n_samples=16):
    # Predict mean and std for test points
    mean, std = predict(model, x_test, n_samples)
    
    # Calculate MSE
    mse = torch.mean((mean - y_test)**2).item()
    
    # Calculate Log Predictive Density
    log_2pi = np.log(2 * np.pi)
    var = std**2
    lpd = -0.5 * (log_2pi + torch.log(var) + (y_test - mean)**2 / var)
    mean_lpd = torch.mean(lpd).item()
    
    return mse, mean_lpd