import torch
from sklearn.model_selection import train_test_split

def get_train_test_dataloaders(n_samples=256, train_split = 0.8, batch_size=64, device='cpu',random_seed=42):
    """
    Generate training and test datasets.
    
    Args:
        n_samples (int): Number of samples to generate for each dataset.
        device (str): Device to use for tensors ('cpu' or 'cuda').
    
    Returns:
        tuple: (x_train, y_train, x_test, y_test) where each is a tensor.
    """
    x,y = generate_data(n_samples=n_samples, device=device)
    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_split, random_state=random_seed)
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=n_samples, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=n_samples, shuffle=False)
    return train_loader, test_loader
    

def true_function(x):
    return torch.sin(2*torch.pi*x)

# Generate synthetic data
def generate_data(n_samples=100, device='cpu'):
    # Sample x from N(0, 1) as specified in the requirements
    x = torch.randn(n_samples, 1, device=device)
    
    # Compute y = f(x) + noise
    # Convert to numpy for true_function, then back to tensor
    y_np = true_function(x)
    y = torch.tensor(y_np, device=device, dtype=torch.float32) + 0.2 * torch.randn(n_samples, 1, device=device)
    
    return x, y