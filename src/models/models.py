import torch 
from bayesian_torch.layers import LinearReparameterization
import torch.nn as nn
import torch.distributions

def create_model(model_type, hidden_size=32):
    if model_type == 'BayesianMLP':
        return BayesianMLP(hidden_size=hidden_size)
    elif model_type == 'InputWeigtedBNN':
        return InputWeigtedBNN(hidden_size=hidden_size)
    elif model_type == 'IncorrectVariationalInputWeigtedBNN':
        return IncorrectVariationalInputWeigtedBNN(hidden_size=hidden_size)
    elif model_type == 'VariationalInputWeigtedBNN':
        return VariationalInputWeigtedBNN(hidden_size=hidden_size)
    elif model_type == 'TemperedVaritaionalBNN':
        return TemperedVaritaionalBNN(hidden_size=hidden_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

class BayesianMLP(nn.Module):
    def __init__(self, hidden_size=20, prior_variance=5.0, posterior_rho_init=-3.0, device='cpu'):
        self.device = device
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
        return -0.5*(torch.log(torch.tensor(2*torch.pi, device=self.device))+torch.log(self.log_sigma2.exp())+diff**2/self.log_sigma2.exp())
    
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
        pdf_x = torch.distributions.Normal(0,1).log_prob(x_train).exp()
        return (log_likelihoods*pdf_x).sum()

class IncorrectVariationalInputWeigtedBNN(InputWeigtedBNN):
    def __init__(self, hidden_size=20):
        super(IncorrectVariationalInputWeigtedBNN, self).__init__(hidden_size)

        self.mean = nn.Parameter(torch.randn(1)/100)
        self.log_std = nn.Parameter(torch.randn(1)/100)


    def get_variational_distribution(self):
        return torch.distributions.Normal(self.mean, self.log_std.exp())

    def get_weighted_log_likelihood(self, x_train, y_train):
        log_likelihoods_y = self.get_normal_log_likelihood(x_train, y_train)
        pdf = self.get_variational_distribution().log_prob(x_train).exp()
        return (log_likelihoods_y*pdf).sum()
    
    def kl_loss(self, num_samples):
        return super(IncorrectVariationalInputWeigtedBNN,self).kl_loss(num_samples) + num_samples*torch.distributions.kl_divergence(self.get_variational_distribution(), torch.distributions.Normal(0,1))
    
class VariationalInputWeigtedBNN(InputWeigtedBNN):
    def __init__(self, hidden_size=20):
        super(VariationalInputWeigtedBNN, self).__init__(hidden_size)

        self.mean = nn.Parameter(torch.randn(1, device=self.device)/100)
        self.log_std = nn.Parameter(torch.randn(1, device=self.device)/100)

    def get_variational_distribution(self):
        return torch.distributions.Normal(self.mean, self.log_std.exp())

    def get_weighted_log_likelihood(self, x_train, y_train):
        log_likelihoods_y = self.get_normal_log_likelihood(x_train, y_train)
        log_likelihoods_x = torch.distributions.Normal(0,1).log_prob(x_train)
        pdf = self.get_variational_distribution().log_prob(x_train).exp()
        return (log_likelihoods_y*pdf+ log_likelihoods_x).sum()
    
    def kl_loss(self, num_samples):
        return super(VariationalInputWeigtedBNN,self).kl_loss(num_samples) + num_samples*torch.distributions.kl_divergence(self.get_variational_distribution(), torch.distributions.Normal(0,1))
    

class TemperedVaritaionalBNN(BayesianMLP):
    def __init__(self, hidden_size=20, temperature=0.8):
        super(TemperedVaritaionalBNN, self).__init__(hidden_size)

        self.mean = nn.Parameter(torch.randn(1)/100)
        self.log_std = nn.Parameter(torch.randn(1)/100)
        self.temperature=temperature

    def get_weighted_log_likelihood(self, x_train, y_train):
        log_likelihoods = self.get_normal_log_likelihood(x_train, y_train)
        return (log_likelihoods/self.temperature).sum()