import argparse
import os
from PIL import Image
import PIL

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import MainConfig, AverageMeter
import models
from models.Sample import HPP_sampling
from vit import ViT


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def generate_eta_c(n, seed=42):
    torch.manual_seed(seed)
    signs = torch.where(torch.rand(n) > 0.5, torch.ones(n), -torch.ones(n))
    magnitudes = 0.9 + 0.1 * torch.rand(n)
    return (signs * magnitudes).float()


# Select the first existing checkpoint from candidate paths.
def find_existing_file(candidates):
    for path in candidates:
        if path is not None and os.path.exists(path):
            return path
    return None


# Verify User A reconstruction after measurement-domain decryption.
def verify_user_a_fixed(config, lambda_noise, model_base_path=None, key_dir='./'):
    print("\n" + "=" * 70)
    print(" Task A: Reconstruction (Baseline and User A)")
    print(" Data Range: [0, 1]")
    print(" Formula: y_noise = (y + lambda * Phi @ eta_c + 2 * lambda * Phi @ 1) / (4 * lambda + 1)")
    print("=" * 70)

    device = config.device
    batch_size = 1

    if model_base_path is None:
        model_base_path = f"./results/HPP/{int(config.ratio * 100)}/models"

    cs_sampling = HPP_sampling(config.ratio, config.init_size).to(device)
    cs_net = models.HybridNet(config).to(device)

    sampling_path = os.path.join(model_base_path, "cs_sampling.pth")
    model_path = os.path.join(model_base_path, "cs_model.pth")

    if not os.path.exists(sampling_path) or not os.path.exists(model_path):
        print(f"[Error] Reconstruction models not found in {model_base_path}")
        return {}

    cs_sampling.load_state_dict(torch.load(sampling_path, map_location=device), strict=False)
    cs_net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    print(f">> Loaded reconstruction models from {model_base_path}")

    cs_sampling.eval()
    cs_net.eval()

    Phi = cs_sampling.phi
    M, N = Phi.shape

    eta_c_path = os.path.join(key_dir, 'eta_c_key.pt')
    if os.path.exists(eta_c_path):
        eta_c = torch.load(eta_c_path, map_location=device).to(device)
        print(f">> Loaded eta_c from {eta_c_path}")
    else:
        eta_c = generate_eta_c(N, seed=42).to(device)
        print(">> eta_c_key.pt not found; generated eta_c with seed=42")

    ones_vec = torch.ones(N, device=device)
    Phi_eta_c = Phi @ eta_c
    Phi_ones = Phi @ ones_vec

    scale = 4 * lambda_noise + 1
    offset_secret = (lambda_noise * Phi_eta_c).view(1, M, 1, 1)
    offset_bias = (2 * lambda_noise * Phi_ones).view(1, M, 1, 1)

    folder_name = "SET11"
    val_root = config.CS_val_path + '/'
    path = os.path.join(val_root, folder_name)

    if not os.path.exists(path):
        for try_path in ["./dataset/val/SET11", "./dataset/val/Set11"]:
            if os.path.exists(try_path):
                path = try_path
                break

    img_list = []
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            if root == path:
                for f in files:
                    if f.lower().endswith('.png'):
                        img_list.append(f)

    if len(img_list) == 0:
        print(f"[Error] No images found in {path}")
        return {}

    img_list = sorted(img_list)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    results = {'base': [], 'user_a': []}

    print(f"\n>> Testing {len(img_list)} images...")

    with torch.no_grad():
        for img_name in img_list:
            name = os.path.join(path, img_name)
            img = Image.open(name).convert('L')
            img_tensor = transform(img).to(device)

            x = img_tensor.squeeze(0).float()
            ori_x = x.clone()

            h, w = x.size()
            h_lack, w_lack = 0, 0

            if h % config.block_size != 0:
                h_lack = config.block_size - h % config.block_size
                temp_h = torch.zeros(h_lack, w, device=device)
                x = torch.cat((x, temp_h), 0)
                h = h + h_lack

            if w % config.block_size != 0:
                w_lack = config.block_size - w % config.block_size
                temp_w = torch.zeros(h, w_lack, device=device)
                x = torch.cat((x, temp_w), 1)
                w = w + w_lack

            x = x.unsqueeze(0).unsqueeze(0)

            idx_h = range(0, h, config.block_size)
            idx_w = range(0, w, config.block_size)
            num_patches = h * w // (config.block_size ** 2)

            temp_base = torch.zeros(
                num_patches, batch_size, 1, config.block_size, config.block_size, device=device
            )
            temp_user_a = torch.zeros_like(temp_base)

            count = 0
            for a in idx_h:
                for b in idx_w:
                    block = x[:, :, a:a + config.block_size, b:b + config.block_size]

                    y, Phi_ret, Q_ret = cs_sampling(block)

                    y_noise = (y + offset_secret + offset_bias) / scale
                    y_decrypt = y_noise * scale - offset_secret - offset_bias

                    out_base = cs_net(y, Phi_ret, Q_ret, batch_size)
                    out_user_a = cs_net(y_decrypt, Phi_ret, Q_ret, batch_size)

                    if isinstance(out_base, (list, tuple)):
                        out_base = out_base[0]
                    if isinstance(out_user_a, (list, tuple)):
                        out_user_a = out_user_a[0]

                    temp_base[count] = out_base
                    temp_user_a[count] = out_user_a
                    count += 1

            def assemble_image(patches, idx_h, idx_w, h, w, h_lack, w_lack, block_size):
                img_out = torch.zeros(1, 1, h, w, device=patches.device)
                count_local = 0
                for a in idx_h:
                    for b in idx_w:
                        img_out[:, :, a:a + block_size, b:b + block_size] = patches[count_local]
                        count_local += 1
                return img_out[:, :, 0:h - h_lack, 0:w - w_lack].squeeze().detach().cpu()

            recon_base = assemble_image(temp_base, idx_h, idx_w, h, w, h_lack, w_lack, config.block_size)
            recon_user_a = assemble_image(temp_user_a, idx_h, idx_w, h, w, h_lack, w_lack, config.block_size)
            ori_x = ori_x.detach().cpu()

            def calc_psnr(recon, orig):
                recon_np = recon.clamp(0, 1).numpy()
                orig_np = orig.clamp(0, 1).numpy()
                mse = np.mean(np.square(recon_np - orig_np))
                if mse < 1e-10:
                    return 100.0
                return 10 * np.log10(1.0 / mse)

            p_base = calc_psnr(recon_base, ori_x)
            p_user_a = calc_psnr(recon_user_a, ori_x)

            results['base'].append(p_base)
            results['user_a'].append(p_user_a)

            print(f"   [{img_name:15s}] Baseline:{p_base:.2f} | UserA:{p_user_a:.2f}")

    avg_base = float(np.mean(results['base']))
    avg_user_a = float(np.mean(results['user_a']))

    print(f"\n>> Summary:")
    print(f"   Baseline: {avg_base:.2f} dB")
    print(f"   User A:   {avg_user_a:.2f} dB")
    print(f"   Gap:      {avg_base - avg_user_a:.2f} dB")

    return {
        'baseline_psnr': avg_base,
        'user_a_psnr': avg_user_a,
        'user_a_gap': avg_base - avg_user_a,
    }


# Verify User B inference on protected measurements.
def verify_user_b_fixed(config, lambda_noise, num_batches=100, model_base_path=None, key_dir='./', user_b_path=None):
    print("\n" + "=" * 70)
    print(" Task B: Inference (User B only)")
    print(" Data Range: [-1, 1]")
    print(" Formula: y_noise = (y + lambda * Phi @ eta_c) / (2 * lambda + 1)")
    print("=" * 70)

    device = config.device
    if model_base_path is None:
        model_base_path = f"./results/HPP/{int(config.ratio * 100)}/models"

    cs_sampling = HPP_sampling(config.ratio, config.init_size, config.img_size).to(device)
    sampling_path = os.path.join(model_base_path, "cs_sampling.pth")

    if not os.path.exists(sampling_path):
        print(f"[Error] Sampling model not found: {sampling_path}")
        return {}

    cs_sampling.load_state_dict(torch.load(sampling_path, map_location=device), strict=False)
    cs_sampling.eval()

    Phi = cs_sampling.phi
    Q = cs_sampling.Q
    M, N = Phi.shape

    print(f">> Phi: {Phi.shape}, Q: {Q.shape}")

    eta_c_path = os.path.join(key_dir, 'eta_c_key.pt')
    if os.path.exists(eta_c_path):
        eta_c = torch.load(eta_c_path, map_location=device).to(device)
        print(f">> Loaded eta_c from {eta_c_path}")
    else:
        eta_c = generate_eta_c(N, seed=42).to(device)
        print(">> eta_c_key.pt not found; generated eta_c with seed=42")

    Phi_eta_c = Phi @ eta_c

    scale = 2 * lambda_noise + 1
    offset_secret = (lambda_noise * Phi_eta_c).view(1, M, 1, 1)

    PhiWeight = Phi.contiguous().view(M, 1, config.init_size, config.init_size)
    QWeight = Q.t().contiguous().view(N, M, 1, 1)

    user_b_vit = ViT(
        config.arch,
        pretrained=False,
        image_size=(config.img_size, config.img_size),
        num_classes=1000,
    ).to(device)


    user_b_candidates = []
    if user_b_path is not None:
        user_b_candidates.append(user_b_path)
    user_b_candidates.extend([
        f'user_b_vit_rate_{config.ratio}_lambda_{lambda_noise}.pth',
        f'user_b_vit_lambda_{lambda_noise}.pth',
        os.path.join(model_base_path, f'user_b_vit_rate_{config.ratio}_lambda_{lambda_noise}.pth'),
        os.path.join(model_base_path, f'user_b_vit_lambda_{lambda_noise}.pth'),
    ])

    selected_user_b_path = find_existing_file(user_b_candidates)

    if selected_user_b_path is None:
        print("[Error] User B model not found. Tried:")
        for p in user_b_candidates:
            print(f"   {p}")
        return {}

    user_b_vit.load_state_dict(torch.load(selected_user_b_path, map_location=device), strict=False)
    user_b_vit.eval()
    print(f">> Loaded User B model: {selected_user_b_path}")

    val_dir = os.path.join(config.IF_data_dir, 'val')
    if not os.path.exists(val_dir):
        print(f"[Error] Val dir not found: {val_dir}")
        return {}

    normalize = transforms.Normalize(0.5, 0.5)
    val_transforms = transforms.Compose([
        transforms.Resize(config.img_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(config.img_size),
        transforms.ToTensor(),
        normalize,
    ])
    val_loader = DataLoader(
        datasets.ImageFolder(val_dir, val_transforms),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    meter_user_b = AverageMeter('UserB', ':6.2f')

    print(f"\n>> Testing validation set (same loop as the original script; num_infer={num_batches} is only kept for compatibility)...")

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            batch_size = images.size(0)

            recon_channels = []
            for c in range(3):
                img_c = images[:, c:c + 1, :, :]
                y_c = F.conv2d(img_c, PhiWeight, stride=config.init_size, padding=0)

                y_c_noise = (y_c + offset_secret) / scale

                x_hat_c = F.conv2d(y_c_noise, QWeight, padding=0)
                x_hat_c = F.pixel_shuffle(x_hat_c, upscale_factor=config.init_size)
                recon_channels.append(x_hat_c)

            features_noise = torch.cat(recon_channels, dim=1)
            features_noise = F.interpolate(
                features_noise,
                size=(config.img_size, config.img_size),
                mode='bilinear',
                align_corners=False,
            )

            out_user_b = user_b_vit(features_noise)
            acc_user_b = accuracy(out_user_b, target, topk=(1,))[0]
            meter_user_b.update(acc_user_b.item(), batch_size)

            if i % 20 == 0:
                print(f"   [{i:3d}/{num_batches}] UserB:{meter_user_b.avg:.2f}%")

    print(f"\n>> Final:")
    print(f"   User B: {meter_user_b.avg:.2f}%")

    return {
        'user_b_top1': float(meter_user_b.avg),
    }


# Run the main evaluation pipeline.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate', default=0.25, type=float)
    parser.add_argument('--device', default='0')
    parser.add_argument('--lambda_noise', default=1.0, type=float)
    parser.add_argument('--lr', default='1e-4')
    parser.add_argument('--key_dir', default='./')
    parser.add_argument('--user_b_path', default=None, help='Optional explicit path to the User B checkpoint.')
    parser.add_argument('--num_infer', default=100, type=int)
    parser.add_argument('--skip_recon', action='store_true')
    parser.add_argument('--skip_infer', action='store_true')
    opt = parser.parse_args()

    device = torch.device(f'cuda:{opt.device}' if torch.cuda.is_available() else 'cpu')
    config = MainConfig(opt.rate, opt.lr, opt.device)
    config.device = device

    print("=" * 70)
    print(" HPP Main Evaluation")
    print("=" * 70)
    print(f" Rate={opt.rate}, Lambda={opt.lambda_noise}")
    print(" Task A: Baseline reconstruction + User A reconstruction")
    print(" Task B: User B inference on protected measurements")
    print("=" * 70)

    results = {}

    if not opt.skip_recon:
        results['recon'] = verify_user_a_fixed(config, opt.lambda_noise, key_dir=opt.key_dir)

    if not opt.skip_infer:
        results['infer'] = verify_user_b_fixed(
            config,
            opt.lambda_noise,
            opt.num_infer,
            key_dir=opt.key_dir,
            user_b_path=opt.user_b_path,
        )

    print("\n" + "#" * 70)
    print(" FINAL REPORT")
    print("#" * 70)

    if 'recon' in results and results['recon']:
        r = results['recon']
        print(f"\n [Task A] Reconstruction:")
        print(f"   Baseline: {r['baseline_psnr']:.2f} dB")
        print(f"   User A:   {r['user_a_psnr']:.2f} dB")

    if 'infer' in results and results['infer']:
        r = results['infer']
        print(f"\n [Task B] Inference:")
        print(f"   User B:   {r['user_b_top1']:.2f}%")


if __name__ == '__main__':
    main()
