import torch
import math
import numpy as np

# 1. Parametric Gaussian estimator (differentiable)
def entropy_gaussian_torch(samples: torch.Tensor) -> torch.Tensor:
    """
    Estimate entropy assuming E ~ Gaussian.
    Differentiable w.r.t. samples.
    H(E) = 0.5 * log(2 * pi * e * sigma^2)
    """
    sigma2 = torch.var(samples, unbiased=True)
    return 0.5 * torch.log(2 * math.pi * math.e * sigma2)


# 2. KDE-based entropy estimator (differentiable approximation)
def entropy_kde_torch(samples: torch.Tensor, bandwidth: float = 0.2) -> torch.Tensor:
    """
    Estimate entropy via differentiable Gaussian KDE in PyTorch.
    Uses pairwise distances and Gaussian kernels.
    """
    samples = samples.view(-1, 1)
    n = samples.size(0)

    # Pairwise squared distances
    dists2 = (samples - samples.T).pow(2)

    # Gaussian kernel density estimate
    kernel = torch.exp(-0.5 * dists2 / (bandwidth ** 2)) / (math.sqrt(2 * math.pi) * bandwidth)
    p_hat = kernel.mean(dim=1)

    # Entropy estimate
    return -torch.mean(torch.log(p_hat + 1e-12))

def generate_lambda_choices(n_layers, max_choices=7):
    """
    Generate a list of lambda vectors for each layer.
    
    Criteria:
    - Values in (0, 1)
    - Up to max_choices different patterns
    - Patterns: increasing, decreasing, constant
    """
    choices = []

    # 1. Constant values
    const_values = np.linspace(0.1, 0.9, max_choices)
    for v in const_values:
        choices.append([v] * n_layers)

    # 2. Increasing across layers
    for i in range(max_choices):
        start = 0.05 + i * 0.1
        end = min(start + 0.5, 0.95)
        choices.append(list(np.linspace(start, end, n_layers)))

    # 3. Decreasing across layers
    for i in range(max_choices):
        start = 0.95 - i * 0.1
        end = max(start - 0.5, 0.05)
        choices.append(list(np.linspace(start, end, n_layers)))

    # truncate if too many
    if len(choices) > max_choices:
        choices = choices[:max_choices]

    return choices

