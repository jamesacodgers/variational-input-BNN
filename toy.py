import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from bayesian_torch.layers import LinearReparameterization

import pandas as pd
from datetime import datetime

# Set random seed for reproducibility
# torch.manual_seed(42)

# Define the cubic polynomial function (ground truth)
# def true_function(x):
#     return 2 * x**3 - 0.5 * x**2 - 5.0 * x + 2.0
def true_function(x):
    return np.sin(2*np.pi*x)

# Generate synthetic data
def generate_data(n_samples=100):
    # Sample x from N(0, 1) as specified in the requirements
    x = torch.randn(n_samples, 1)
    
    # Compute y = f(x) + noise
    y = true_function(x) + 0.2 * torch.randn(n_samples, 1)
    
    return x, y

# Define a Bayesian MLP using mean field variational inference
class BayesianMLP(nn.Module):
    def __init__(self, hidden_size=20, prior_variance=5.0, posterior_rho_init=-3.0):
        super(BayesianMLP, self).__init__()
        # First Bayesian layer
        self.layer1 = LinearReparameterization(
            1, hidden_size,
            prior_mean=0.0,
            prior_variance=prior_variance,
            posterior_mu_init=0.0,
            posterior_rho_init=posterior_rho_init
        )
        # Activation function
        self.relu = nn.ReLU()
        # Second Bayesian layer
        self.layer2 = LinearReparameterization(
            hidden_size, hidden_size,
            prior_mean=0.0,
            prior_variance=prior_variance,
            posterior_mu_init=0.0,
            posterior_rho_init=posterior_rho_init
        )
        # Output Bayesian layer
        self.output_layer = LinearReparameterization(
            hidden_size, 1,
            prior_mean=0.0,
            prior_variance=prior_variance,
            posterior_mu_init=0.0,
            posterior_rho_init=posterior_rho_init
        )
        self.log_sigma2 = nn.Parameter(torch.zeros(1))
        # self.likelihood = torch.distributions.Normal(0, self.log_sigma2.exp())
    
    def forward(self, x):
        x = self.layer1(x, return_kl=False)
        x = self.relu(x)
        x = self.layer2(x, return_kl=False)
        x = self.relu(x)
        return self.output_layer(x, return_kl=False)
    
    def kl_loss(self, num_samples):
        # Compute KL divergence between posterior and prior for all layers
        return self.layer1.kl_loss() + self.layer2.kl_loss() + self.output_layer.kl_loss()
    
    def get_normal_log_likelihood(self, x_train, y_train):
        y_pred = self(x_train)
        diff = y_pred - y_train
        return -0.5*(torch.log(torch.tensor(2*torch.pi))+torch.log(self.log_sigma2.exp())+diff**2/self.log_sigma2.exp())
    
    def get_weighted_log_likelihood(self, x_train, y_train):
        return self.get_normal_log_likelihood(x_train, y_train).sum()
    
    def elbo(self, x_train, y_train): 
        weighted_log_likelihood = self.get_weighted_log_likelihood(x_train, y_train)
        kl = self.kl_loss(x_train.shape[0])
        
        # Total loss: negative ELBO = MSE + KL/N
        # The KL term is normalized by the dataset size
        return weighted_log_likelihood - kl 
    
class InputWeigtedBNN(BayesianMLP):
    def __init__(self, hidden_size=20):
        super(InputWeigtedBNN, self).__init__(hidden_size)

    def get_weighted_log_likelihood(self, x_train, y_train):
        log_likelihoods = self.get_normal_log_likelihood(x_train, y_train)
        pdf = torch.distributions.Normal(0,1).log_prob(x_train).exp()
        return (log_likelihoods*pdf).sum()

class VariationalInputWeigtedBNN(InputWeigtedBNN):
    def __init__(self, hidden_size=20):
        super(VariationalInputWeigtedBNN, self).__init__(hidden_size)

        self.mean = nn.Parameter(torch.randn(1)/100)
        self.log_std = nn.Parameter(torch.randn(1)/100)


    def get_variational_distribution(self):
        return torch.distributions.Normal(self.mean, self.log_std.exp())

    def get_weighted_log_likelihood(self, x_train, y_train):
        log_likelihoods = self.get_normal_log_likelihood(x_train, y_train)
        pdf = self.get_variational_distribution().log_prob(x_train).exp()
        return (log_likelihoods*pdf).sum()
    
    def kl_loss(self, num_samples):
        return super(VariationalInputWeigtedBNN,self).kl_loss(num_samples) + num_samples*torch.distributions.kl_divergence(self.get_variational_distribution(), torch.distributions.Normal(0,1))
    


# Train the BNN model
def train_model(model, x_train, y_train, n_epochs=10000, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # mse_loss = nn.MSELoss(reduction='none')
    
    losses = []
    for epoch in range(n_epochs):
        # Forward pass

        loss = - model.elbo(x_train, y_train)
        losses.append(loss.item())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 200 == 0:
            # print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, MSE: {mse.item():.4f}, KL: {kl.item():.4f}')
            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}')
    
    return model, losses

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
# Visualize the results with density plot and metrics
# Visualize the results with multiple models
def plot_results(models, x_train, y_train, x_test, y_test, losses_list, model_names=None, var_dist=None, hidden_size="not_given"):
    # Create a figure with 2 subplots, sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[4, 1], sharex=True)
    plt.subplots_adjust(hspace=0.05)  # Reduce space between subplots
    
    # Create a grid of x values for plotting
    x_grid = torch.linspace(-3, 3, 100).reshape(-1, 1)
    
    # Convert to numpy arrays for training and test data
    x_train_np = x_train.numpy()
    y_train_np = y_train.numpy()
    x_test_np = x_test.numpy()
    y_test_np = y_test.numpy()
    x_grid_np = x_grid.numpy()
    
    # Compute the true function values
    y_true = true_function(x_grid).numpy()
    
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
        mean_np = mean.numpy()
        std_np = std.numpy()
        
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
    with torch.no_grad():
        
        x_range = torch.linspace(-3, 3, 500)
        pdf = 1/np.sqrt(2*np.pi) * torch.exp(-0.5 * x_range**2)  # Standard normal PDF

        ax2.plot(x_range, pdf, 'k-', label='N(0,1) density')
        if var_dist is not None:
            var_pdf = var_dist.log_prob(x_range).exp().numpy()
            ax2.plot(x_range, var_pdf, 'r-', label='Variational density')
        # Mark test data points on x-axis
        ax2.scatter(x_test_np, np.zeros_like(x_test_np) + 0.02, marker='|', 
                    color='black', s=20, label='Test points')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('Density')
    ax2.legend(loc='upper right', fontsize='small')
    ax2.grid(True)
    
    # plt.show()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig.savefig(f'figs/{hidden_size}_comparison_plot_{timestamp}.png', dpi=300, bbox_inches='tight')
    
    # # Plot loss curves in a separate figure (if provided)
    # if losses_list:
    #     plt.figure(figsize=(10, 5))
    #     for i, losses in enumerate(losses_list):
    #         if losses is not None:
    #             color = colors[i % len(colors)]
    #             plt.plot(losses, color=color, label=model_names[i])
        
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss (Negative ELBO)')
    #     plt.title('Training Losses')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

# Calculate evaluation metrics
def calculate_metrics(model, x_test, y_test, n_samples=1000):
    # Predict mean and std for test points
    mean, std = predict(model, x_test, n_samples)
    
    # Calculate MSE
    mse = torch.mean((mean - y_test)**2).item()
    
    # Calculate Log Predictive Density
    # LPD = log p(y|x) for Normal distribution
    # log p(y|x) = -0.5 * log(2πσ²) - 0.5 * (y-μ)²/σ²
    log_2pi = np.log(2 * np.pi)
    var = std**2
    lpd = -0.5 * (log_2pi + torch.log(var) + (y_test - mean)**2 / var)
    mean_lpd = torch.mean(lpd).item()
    
    return mse, mean_lpd

def save_results_to_csv( model, input_weighted_model, variational_input_weighted_model, 
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
        'variational_log_sigma2': variational_input_weighted_model.log_sigma2.item()
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




# Main execution
# Main execution
if __name__ == "__main__":
    # Generate training data
    x_train, y_train = generate_data(n_samples=100)
    
    # Generate test data (from the same distribution)
    x_test, y_test = generate_data(n_samples=200)
    
    hidden_size=1000
    # Create and train the model
    model = BayesianMLP(hidden_size=hidden_size)
    input_weighted_model = InputWeigtedBNN(hidden_size=hidden_size)
    variational_input_weighted_model = VariationalInputWeigtedBNN(hidden_size=hidden_size)
    model, losses = train_model(model, x_train, y_train)
    input_weighted_model, losses = train_model(input_weighted_model, x_train, y_train)
    variational_input_weighted_model, losses = train_model(variational_input_weighted_model, x_train, y_train)
    
    # Calculate metrics on test data
    mse, lpd = calculate_metrics(model, x_test, y_test)
    iw_mse, iw_pd = calculate_metrics(input_weighted_model, x_test, y_test)
    viw_mse, viw_pd = calculate_metrics(input_weighted_model, x_test, y_test)
    print(f"Test MSE: {mse:.4f}, {iw_mse:.4f}, {viw_mse:.4f}")
    print(f"Test Log Predictive Density: {lpd:.4f}, {iw_pd:.4f}, {viw_pd:.4f}")
    print(variational_input_weighted_model.mean, variational_input_weighted_model.log_std.exp())
    print(model.log_sigma2, input_weighted_model.log_sigma2, variational_input_weighted_model.log_sigma2)
    # Plot the results
    plot_results([
        model, 
        input_weighted_model,
        variational_input_weighted_model
        ], 
        x_train, y_train, x_test, y_test, losses, var_dist=variational_input_weighted_model.get_variational_distribution(), hidden_size=hidden_size)
    

        # After calculating metrics and before plotting, add:
    save_results_to_csv(
        model, 
        input_weighted_model, 
        variational_input_weighted_model,
        mse, lpd, 
        iw_mse, iw_pd, 
        viw_mse, viw_pd, filename=f"experiment_results_{hidden_size}.csv"
    )