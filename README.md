# CSHPP: Hierarchical Privacy Preservation Scheme Based on Compressive Sensing

This repository provides the core PyTorch implementation of **CSHPP** . It includes the main training and evaluation pipeline for shared compressive sampling, User A reconstruction, privacy asset generation, and User B inference.

## File Structure

```text
CSHPP/
├── README.md
├── requirements.txt
├── main.py
├── privacy_noise.py
├── step1_prepare_assets.py
├── step2_train_user_b.py
├── step3_full_test.py
├── models/
├── utils/
├── vit/
└── pretrain/
```

Main scripts:

| File | Purpose |
|---|---|
| `main.py` | Jointly trains the shared sampler, reconstruction network, and clean inference model. |
| `step1_prepare_assets.py` | Extracts the learned measurement matrix and generates privacy assets. |
| `step2_train_user_b.py` | Trains the noise-adapted User B inference model. |
| `step3_full_test.py` | Evaluates User A reconstruction and User B inference on protected measurements. |
| `privacy_noise.py` | Provides privacy noise and null-space noise utilities. |

## Environment

The experiments were conducted with:

```text
Python 3.13.2
PyTorch 2.9.1+cu128
TorchVision 0.24.1+cu128
CUDA 12.8
GPU: NVIDIA RTX A6000
CPU: Intel Core i9-14900K
```

Install PyTorch according to your CUDA version. For CUDA 12.8:

```bash
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Data and Pretrained Weights

Datasets and model weights are not included in this repository. Please download them from their official sources or public dataset pages and follow their license terms.

Recommended dataset sources:

| Dataset | Usage | Source |
|---|---|---|
| Set11 | Reconstruction evaluation | Common CS benchmark; place the images under `dataset/val/SET11/`. |
| BSDS500 | Reconstruction training / visualization | [Berkeley Segmentation Dataset and Benchmark](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) |
| ImageNet / ImageNet subset | Classification inference | [ImageNet download page](https://www.image-net.org/download.php) |
| CelebA | Attribute recognition experiments | [CelebA official page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |

Default reconstruction paths in `utils/config.py`:

```python
self.CS_train_path = "./dataset/train"
self.CS_val_path = "./dataset/val"
```

Default ImageNet-style inference path in `utils/config.py`:

```python
self.IF_data_dir = "/home/compressedsensing/dataset/imagenet-100"
```

Please modify `self.IF_data_dir` before running the code. The inference dataset should follow the `ImageFolder` format:

```text
imagenet-100/
├── train/
│   ├── class_1/
│   └── class_2/
└── val/
    ├── class_1/
    └── class_2/
```

The ViT pretrained weight should be downloaded from the PyTorch-Pretrained-ViT project releases:

```text
https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases
```

Download the ViT-B/32 ImageNet-1k weight and place it as:

```text
pretrain/B_32_imagenet1k.pth
```

The default configuration uses:

```python
self.pretrain_weight = "./pretrain/B_32_imagenet1k.pth"
```

The joint training script also uses torchvision VGG16 pretrained weights for the reconstruction loss. If internet access is available, torchvision can download them automatically; otherwise, please prepare them in the local PyTorch cache in advance.

## Running the Main Pipeline

The following example uses sampling rate `r=0.04` and noise intensity `lambda_noise=5.0`.

### 1. Joint Training

```bash
python main.py \
  --rate 0.04 \
  --epochs 50 \
  --lr 1e-4 \
  --device 0
```

Expected outputs:

```text
results/HPP/4/models/cs_sampling.pth
results/HPP/4/models/cs_model.pth
results/HPP/4/models/if_model.pth
```

### 2. Prepare Privacy Assets

```bash
python step1_prepare_assets.py \
  --rate 0.04 \
  --device 0
```

This generates:

```text
phi_key.pt
R_matrix_key.pt
eta_c_key.pt
```

Move them to a rate-specific folder:

```bash
mkdir -p keys/4
mv phi_key.pt R_matrix_key.pt eta_c_key.pt keys/4/
```

### 3. Train User B

```bash
python step2_train_user_b.py \
  --rate 0.04 \
  --lambda_noise 5.0 \
  --epochs 50 \
  --batch_size 64 \
  --device 0 \
  --key_dir ./keys/4
```

Expected output:

```text
user_b_vit_rate_0.04_lambda_5.0.pth
```

### 4. Evaluate the Main Pipeline

```bash
python step3_full_test.py \
  --rate 0.04 \
  --lambda_noise 5.0 \
  --device 0 \
  --key_dir ./keys/4
```

If the User B checkpoint is not found automatically, specify it explicitly:

```bash
python step3_full_test.py \
  --rate 0.04 \
  --lambda_noise 5.0 \
  --device 0 \
  --key_dir ./keys/4 \
  --user_b_path ./user_b_vit_rate_0.04_lambda_5.0.pth
```

To evaluate only reconstruction or only inference:

```bash
python step3_full_test.py --rate 0.04 --lambda_noise 5.0 --device 0 --key_dir ./keys/4 --skip_infer
python step3_full_test.py --rate 0.04 --lambda_noise 5.0 --device 0 --key_dir ./keys/4 --skip_recon
```

## Running Multiple Sampling Rates

The tested sampling rates are:

```text
0.01, 0.04, 0.10, 0.25
```

After running `main.py` for each rate, run:

```bash
for r in 0.01 0.04 0.10 0.25
do
  rf=$(python -c "print(int(round(float('$r') * 100)))")

  python step1_prepare_assets.py --rate $r --device 0

  mkdir -p keys/$rf
  mv phi_key.pt R_matrix_key.pt eta_c_key.pt keys/$rf/

  python step2_train_user_b.py \
    --rate $r \
    --lambda_noise 5.0 \
    --epochs 50 \
    --batch_size 64 \
    --device 0 \
    --key_dir ./keys/$rf

  python step3_full_test.py \
    --rate $r \
    --lambda_noise 5.0 \
    --device 0 \
    --key_dir ./keys/$rf
done
```

## Files Not Included

The following files are intentionally excluded:

```text
*.pth
*.pt
*.log
pretrain/*.pth
results/
dataset/
data/
reconstructed_images/
__pycache__/
.ipynb_checkpoints/
```

This repository does not include large model checkpoints, generated privacy keys, training logs, datasets, or cache files. Please generate or download them following the instructions above.

## Citation

The paper is currently under review. Citation information will be updated after publication.
