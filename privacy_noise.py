import torch
import numpy as np
import scipy.linalg


# Generate the static private-key noise.
def generate_strict_eta_c(n, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    signs = torch.where(torch.rand(n) > 0.5,
                        torch.ones(n),
                        -torch.ones(n))
    magnitudes = 0.9 + 0.1 * torch.rand(n)
    eta_c = signs * magnitudes

    return eta_c.float()


# Build candidate null-space noise directions.
def prepare_R_from_Z(Z, c_ratio=None):
    if isinstance(Z, torch.Tensor):
        Z = Z.numpy()

    n, d = Z.shape

    if c_ratio is None:
        c = max(int(np.sqrt(d)), 1)
    else:
        c = min(c_ratio, d)
    c = max(c, 1)


    perm = np.random.permutation(d)
    Z_shuffled = Z[:, perm]


    col_min = Z_shuffled.min(axis=0, keepdims=True)
    col_max = Z_shuffled.max(axis=0, keepdims=True)
    col_range = col_max - col_min
    col_range = np.where(col_range < 1e-8, 1.0, col_range)
    Z_scaled = 2 * (Z_shuffled - col_min) / col_range - 1


    R = np.zeros((n, c), dtype=np.float32)
    group_size = d // c

    for i in range(c):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size if i < c - 1 else d
        R[:, i] = Z_scaled[:, start_idx:end_idx].sum(axis=1)


        r_min, r_max = R[:, i].min(), R[:, i].max()
        if r_max - r_min > 1e-8:
            R[:, i] = 2 * (R[:, i] - r_min) / (r_max - r_min) - 1

    return torch.tensor(R, dtype=torch.float32)


def generate_strict_eta_r(R, num_samples, p=0.5, device='cpu'):
    if isinstance(R, np.ndarray):
        R = torch.tensor(R, dtype=torch.float32)

    R = R.to(device)
    n, c = R.shape
    k = max(1, int(c * p))

    indices = torch.stack([
        torch.randperm(c, device=device)[:k]
        for _ in range(num_samples)
    ])

    R_T = R.T
    eta_r = torch.zeros(num_samples, n, device=device)

    for i in range(k):
        col_indices = indices[:, i]
        eta_r += R_T[col_indices]

    eta_min = eta_r.min(dim=1, keepdim=True)[0]
    eta_max = eta_r.max(dim=1, keepdim=True)[0]
    eta_range = eta_max - eta_min
    eta_range = torch.where(eta_range < 1e-8,
                            torch.ones_like(eta_range),
                            eta_range)
    eta_r = 2 * (eta_r - eta_min) / eta_range - 1

    return eta_r


# Generate dynamic null-space noise in batch.
def generate_eta_r_fast(R, num_samples, p=0.5, device='cpu'):
    if isinstance(R, np.ndarray):
        R = torch.tensor(R, dtype=torch.float32)

    R = R.to(device)
    n, c = R.shape

    mask = (torch.rand(num_samples, c, device=device) < p).float()

    zero_rows = mask.sum(dim=1) == 0
    if zero_rows.any():
        random_cols = torch.randint(0, c, (zero_rows.sum(),), device=device)
        mask[zero_rows, random_cols] = 1.0

    eta_r = mask @ R.T

    eta_min = eta_r.min(dim=1, keepdim=True)[0]
    eta_max = eta_r.max(dim=1, keepdim=True)[0]
    eta_range = eta_max - eta_min
    eta_range = torch.where(eta_range < 1e-8,
                            torch.ones_like(eta_range),
                            eta_range)
    eta_r = 2 * (eta_r - eta_min) / eta_range - 1

    return eta_r


# Apply image-domain privacy perturbation.
def add_mlaas_noise_to_signal(x, eta_c, eta_r, lambda_noise):
    if eta_c.dim() == 1:
        eta_c = eta_c.unsqueeze(0).expand(x.shape[0], -1)
    x_noise = (x + lambda_noise * (eta_c + eta_r)) / (2 * lambda_noise + 1)
    return x_noise