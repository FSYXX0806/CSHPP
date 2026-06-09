import torch
import numpy as np
import scipy.linalg
import os
import argparse
from models.Sample import HPP_sampling
from utils import MainConfig
from privacy_noise import generate_strict_eta_c, prepare_R_from_Z

# Extract the learned measurement matrix from the sampler.
def get_phi_from_model(model, device, config):
    print("直接从模型参数中搜索...")
    for name, param in model.named_parameters():
        if 'phi' in name.lower() or 'sampling' in name.lower():
            return param.detach().cpu().view(param.shape[0], -1)

    print("运行前向传播...")
    block_size = 32
    dummy_block = torch.randn(1, 1, block_size, block_size).to(device)
    try:
        results = model(dummy_block)
        if isinstance(results, tuple) and len(results) >= 2:
            return results[1].detach().cpu().view(results[1].shape[0], -1)
    except:
        pass
    raise ValueError("无法提取 Phi，请检查 models/Sample.py")

# Prepare Phi, null-space assets, and static key noise.
def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate', default=0.25, type=float, help='sampling rate')
    parser.add_argument('--device', default='0')
    parser.add_argument('--lr', default='1e-4')
    opt = parser.parse_args()

    config = MainConfig(opt.rate, opt.lr, opt.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Step 1: Prepare Assets (Rate={opt.rate})")


    rate_int = int(opt.rate * 100)
    auto_save_path = f"results/HPP/{rate_int}/models/cs_sampling.pth"
    model_path = auto_save_path if os.path.exists(auto_save_path) else "cs_sampling.pth"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    cs_sampling = HPP_sampling(config.ratio, config.init_size).to(device)
    cs_sampling.load_state_dict(torch.load(model_path, map_location=device), strict=False)


    phi = get_phi_from_model(cs_sampling, device, config)
    torch.save(phi, 'phi_key.pt')
    print(f"[Saved] phi_key.pt ({phi.shape})")


    print("Calculating SVD (this takes time)...")
    # Compute the null-space basis of Phi.
    U, s, Vh = scipy.linalg.svd(phi.numpy())
    m, n = phi.shape
    Z = torch.tensor(Vh[m:, :].T, dtype=torch.float32)

    print("Generating R matrix...")
    R = prepare_R_from_Z(Z, c_ratio=Z.shape[1])
    torch.save(R, 'R_matrix_key.pt')
    print("[Saved] R_matrix_key.pt")


    print("Generating Eta_c...")
    eta_c = generate_strict_eta_c(n)
    torch.save(eta_c, 'eta_c_key.pt')
    print("[Saved] eta_c_key.pt")

if __name__ == '__main__':
    prepare()