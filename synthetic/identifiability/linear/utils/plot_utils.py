import numpy as np
import matplotlib.pyplot as plt

def scatter_itself(data):
    if isinstance(data, list):
        try:
            data = np.concatenate(data, axis=-1)
        except Exception as e:
            data = np.concatenate([d.detach().cpu().numpy() if hasattr(d, "detach") else np.asarray(d) for d in data], axis=-1)
        
    dim = data.shape[1]
    data = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
    fig, axes = plt.subplots(dim, dim, figsize=(12, 12), sharex=False, sharey=False)
    for i in range(dim):
        for j in range(dim):
            ax = axes[i, j]
            if i == j:
                ax.hist(data[:, i], bins=30, alpha=0.7, edgecolor='black')
                if i == dim - 1: ax.set_xlabel(f'data{i}', fontsize=9)
                if j == 0: ax.set_ylabel('Frequency', fontsize=9)
            else:
                ax.scatter(data[:, j], data[:, i], alpha=0.4, s=10)
                if i == dim - 1: ax.set_xlabel(f'data{j}', fontsize=9)
                if j == 0: ax.set_ylabel(f'data{i}', fontsize=9)