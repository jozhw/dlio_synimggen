import numpy as np
from scipy.stats import beta, skew


def estimate_beta_params(mu, var, skew):
    """
    Estimate the alpha and beta parameters of the beta distribution
    from the sample mean, variance, and skewness.
    """
    a = (mu * (mu * (1 - mu) / var - 1) - skew) / (skew - 2 * mu * (1 - mu) / var)
    b = (1 - mu) * (mu * (1 - mu) / var - 1 - skew) / (skew - 2 * mu * (1 - mu) / var)
    return a, b


# Calculate sample statistics from your dataset
data = np.array([...])  # Replace with your dataset
mu = np.mean(data)
var = np.var(data)
skew = skew(data)

# Estimate alpha and beta parameters
alpha, b = estimate_beta_params(mu, var, skew)

# Generate random integer samples from the fitted beta distribution
n_samples = 10000  # Number of samples to generate
low, high = np.floor(np.min(data)), np.ceil(np.max(data))  # Range of integer values
random_samples = np.random.randint(
    low,
    high=high - 1,  # high is exclusive in np.random.random_integers
    size=n_samples,
    p=beta.pdf(np.linspace(low, high, high - low + 1), alpha, b),
)
