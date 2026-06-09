from .config import GetConfig, GetIFConfig, MainConfig
from .loader import train_loader
from .util_meter import AverageMeter, ProgressMeter, adjust_learning_rate, accuracy
import os
import torch
import PIL
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# --- 生产环境：真实数据加载器 ---

def get_if_data(config):
    print(f">> [Data] Loading real data from: {config.IF_data_dir}")
    train_dir = os.path.join(config.IF_data_dir, 'train')
    val_dir = os.path.join(config.IF_data_dir, 'val')
    normalize = transforms.Normalize(0.5, 0.5)
    
    # 训练集预处理
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(config.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.if_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    
    # 验证集预处理
    val_transforms = transforms.Compose([
        transforms.Resize(config.img_size, interpolation=PIL.Image.BICUBIC),  # 三次插值
        transforms.CenterCrop(config.img_size),  # 居中裁剪
        transforms.ToTensor(),
        normalize
    ])
    
    val_dataloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(val_dir, val_transforms),
        batch_size=config.if_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,)
        
    return train_loader, val_dataloader