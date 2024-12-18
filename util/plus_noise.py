import numpy as np
import torch
from util.label_noise import noisify_p
def add_noise_to_subgraphs(subgraphs, noisy_rate, noisy_type, num_classes):
    client_noisy_rates = []
    torch.manual_seed(2024)
    for subgraph in subgraphs:
        client_y = subgraph.y.numpy()
        train_mask = subgraph.train_idx.numpy() 

        if noisy_rate == 0.0:
            upper_bound = 0.0
        elif noisy_type == 'uniform':
            upper_bound = 0.6
        elif noisy_type == 'pair':
            upper_bound = 0.45
        noisy_rate_now = torch.rand(1).item() * (upper_bound - noisy_rate) + noisy_rate
        print(f"Generated noisy rate: {noisy_rate_now}")

        client_noisy_rates.append(noisy_rate_now)

        train_indices = np.where(train_mask)[0]
        client_y_noisy = client_y.copy()

        noisy_labels, _ = noisify_p(client_y[train_indices], noisy_type, float(noisy_rate_now), n_class=num_classes, random_state=42)
        client_y_noisy[train_indices] = noisy_labels

        subgraph.y = torch.from_numpy(client_y_noisy)

    return client_noisy_rates


