import PIL
import torchvision.datasets as datasets
import os
import torch
import torch.nn as nn
import models
import utils
import time
import argparse
import numpy as np
import torch.utils.data
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as SSIM
from scipy import stats
from PIL import Image
from models.Sample import *
from utils import *
from vit import ViT
from torchvision.models import vgg16
from pytorch_msssim import SSIM as TorchSSIM
from torchvision.utils import save_image
import torch.nn.functional as F

# Parse the main training configuration.
parser = argparse.ArgumentParser()
parser.add_argument('--rate', default=0.25, type=float, help='sampling rate')
parser.add_argument('--device', default='0')
parser.add_argument('--lr', default='1e-4')
parser.add_argument('--epochs', default=60, type=int, help='number of training epochs')
opt = parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# Build ImageNet-style dataloaders for the inference branch.
def get_if_data(config):
    train_dir = os.path.join(config.IF_data_dir, 'train')
    val_dir = os.path.join(config.IF_data_dir, 'val')
    normalize = transforms.Normalize(0.5, 0.5)
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
    val_transforms = transforms.Compose([
        transforms.Resize(config.img_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(config.img_size),
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

def adapt_MAE_loss(predicted, observed, l1, l2, l3):
    batch_size = predicted.shape[0]
    mae = torch.mean(torch.abs(predicted - observed))
    mbe = torch.abs(predicted.mean() - observed.mean())
    adjust_predicted = predicted - mbe
    obs_np = observed.view(batch_size, -1).detach().cpu().numpy()
    adj_np = adjust_predicted.view(batch_size, -1).detach().cpu().numpy()
    slope_list = []
    intercept_list = []
    for i in range(batch_size):
        if np.std(obs_np[i]) > 1e-3:
            slope, intercept, _, _, _ = stats.linregress(obs_np[i], adj_np[i])
            slope_list.append(slope)
            intercept_list.append(intercept)
        else:
            slope_list.append(0)
            intercept_list.append(obs_np[i][0] - adj_np[i][0])
    proportionality_error = torch.mean(torch.abs(slope * observed + intercept - adjust_predicted))
    unsystematic_error = torch.abs(mae - mbe - proportionality_error)
    weights = torch.softmax(torch.stack([l1, l2, l3]), dim=0)
    return weights[0] * mbe + weights[1] * proportionality_error + weights[2] * unsystematic_error

def ssim_loss(pre, obs):
    ssim_val = SSIM(pre, obs, data_range=1.0)
    return 1 - torch.tensor(ssim_val, dtype=torch.float32)

def get_mae_iter_loss(predicted, observed):
    mae = torch.nn.functional.l1_loss(predicted, observed)
    mbe = torch.abs(predicted.mean() - observed.mean())
    proportionality_error = ssim_loss(predicted.detach().cpu().numpy(), observed.detach().cpu().numpy())
    return 0.33 * mae + 0.33 * mbe + 0.33 * proportionality_error

def get_iter_loss(predicted, observed):
    mae = torch.mean(torch.abs(predicted - observed))
    mbe = torch.abs(predicted.mean() - observed.mean())
    adjust_predicted = predicted - mbe
    obs_np = observed.view(-1).detach().cpu().numpy()
    adj_np = adjust_predicted.view(-1).detach().cpu().numpy()
    slope = 0
    intercept = 0
    if np.std(obs_np) > 1e-3:
        slope, intercept, _, _, _ = stats.linregress(obs_np, adj_np)
    else:
        slope = 0
        intercept = obs_np.mean() - adj_np.mean()
    proportionality_error = torch.mean(torch.abs(slope * torch.tensor(obs_np) + intercept - adjust_predicted.reshape(-1).detach().cpu()))
    return 0.5 * mbe + 0.5 * proportionality_error

def MAE_loss(predicted, observed):
    t_loss = 0.0
    for i in range(predicted.shape[0]):
        t_loss += get_mae_iter_loss(predicted[i].squeeze(), observed[i].squeeze())
    return t_loss / predicted.shape[0]
# Reconstruction loss used for the CS branch.
class HybridLoss(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.mse = nn.MSELoss()


        self.fft_weight = 0.1


        self.vgg = vgg16(pretrained=True).features[:16].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.perceptual_weight = 0.05


        self.ssim_loss = TorchSSIM(data_range=1.0, channel=1, size_average=True).to(device)
        self.ssim_weight = 0.1

    @staticmethod
    def normalize(x):
        x = (x + 1) / 2.0
        x = x.repeat(1, 3, 1, 1)
        return x

    def forward(self, pred, target):

        spatial_loss = self.mse(pred, target)


        pred_fft = torch.fft.fftn(pred, dim=(-2, -1))
        target_fft = torch.fft.fftn(target, dim=(-2, -1))
        freq_loss = F.l1_loss(pred_fft.abs(), target_fft.abs())


        pred_vgg = self.vgg(self.normalize(pred))
        with torch.no_grad():
            target_vgg = self.vgg(self.normalize(target))
        percep_loss = F.mse_loss(pred_vgg, target_vgg)


        ssim_val = self.ssim_loss(pred, target)
        ssim_loss = 1.0 - ssim_val


        total_loss = (spatial_loss +
                            self.fft_weight * freq_loss +
                            self.perceptual_weight * percep_loss +
                            self.ssim_weight * ssim_loss)

        return total_loss


# Jointly train the shared sampler, reconstruction network, and inference model.
def main():
    set_seed(22)
    config = MainConfig(opt.rate, opt.lr, opt.device)
    config.check()
    torch.cuda.empty_cache()
    cs_dataset_train = utils.train_loader(config.cs_batch_size)
    if_dataset_train, if_dataset_val = get_if_data(config)
    cs_sampling = HPP_sampling(config.ratio, config.init_size).to(config.device)
    cs_net = models.HybridNet(config).to(config.device)
    if_net = ViT(config.arch, pretrained=True,
                image_size=(config.img_size, config.img_size),
                num_classes=1000,
                weights_path=config.pretrain_weight).to(config.device)

    cs_criterion = HybridLoss(config.device).to(config.device)


    # Use separate learning rates for the three trainable modules.
    optimizer = optim.Adam([
        {'params': cs_sampling.parameters(), 'lr': 1e-3},
        {'params': cs_net.parameters(), 'lr': 0.0001},
        {'params': if_net.parameters(), 'lr': 1e-5}
    ])

    scheduler = LS.StepLR(optimizer, step_size=3, gamma=0.9)

    config.start_epoch = 1
    config.epochs = opt.epochs

    best = 0
    acc_1 = 0
    print("=> no checkpoint found, start new epoch")

    cs_sampling.train()
    cs_net.train()
    if_net.train()
    save_dir = "reconstructed_images"
    # Main joint-training loop.
    for epoch in range(config.start_epoch, config.epochs + 1):
        losses = AverageMeter('Loss', ':.4e')
        best_res = AverageMeter('CS_loss', ':.4f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        train_size = min(len(cs_dataset_train), len(if_dataset_train))
        progress = ProgressMeter(
            train_size, losses, best_res, top1, top5, prefix=f"Epoch: [{epoch}]")

        if epoch == 1:
            if not os.path.isfile(config.log):
                output_file = open(config.log, 'w')
                output_file.write("HPP Game Begin")
                output_file.close()
        print(f"{epoch} Learning rate is {optimizer.param_groups[0]['lr']}")

        for i, ((if_img, target), cs_img) in enumerate(zip(if_dataset_train, cs_dataset_train)):
            if_img = if_img.to(config.device)
            if_img = cs_sampling(if_img)
            target = target.to(config.device)
            output = if_net(if_img)
            if_loss = torch.nn.CrossEntropyLoss()(output, target).to(config.device)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            cs_img = cs_img.to(config.device)
            xs, phi, Q = cs_sampling(cs_img)
            [xo, layers_sym, layers_st] = cs_net(xs, phi, Q, config.cs_batch_size)

            loss_constraint = 0
            for k, _ in enumerate(layers_sym, 0):
                loss_constraint += torch.mean(torch.pow(layers_sym[k], 2))
            sparsity_constraint = 0
            for k, _ in enumerate(layers_st, 0):
                sparsity_constraint += torch.mean(torch.abs(layers_st[k]))

            cs_main_loss = cs_criterion(xo, cs_img)

            cs_loss = cs_main_loss + 0.01 * loss_constraint + 0.001 * sparsity_constraint

            loss = 0.2 * if_loss + 0.8 * cs_loss
            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(cs_net.parameters(), max_norm=1.0)

            optimizer.step()
            losses.update(loss.item(), if_img.size(0) + cs_img.size(0))
            top1.update(acc1[0], if_img.size(0))
            top5.update(acc5[0], if_img.size(0))

            if i % config.print_freq == 0:
                progress.print(i)
        if_loss, if_acc1 = if_validate(
            if_dataset_val, if_net, torch.nn.CrossEntropyLoss(), config, cs_sampling)
        cs_res = cs_validate(config, cs_net, cs_sampling, epoch)

        print(f"CS Validation PSNR: {cs_res}")


        psnr_improved = cs_res >= best
        acc_improved = if_acc1 >= acc_1

        if psnr_improved:
            best = cs_res
            torch.save(cs_sampling.state_dict(), config.sample)
            torch.save(cs_net.state_dict(), config.cs_model)

        if acc_improved:
            acc_1 = if_acc1
            torch.save(if_net.state_dict(), config.if_model)


        if psnr_improved or acc_improved:
            torch.save(optimizer.state_dict(), config.optimizer)

            info = {
                'epoch': epoch,
                'best': best,
                'acc_1': acc_1,
            }
            torch.save(info, config.info)
            output_file = open(config.log, 'r+')
            old = output_file.read()
            output_file.seek(0)
            output_file.write(
                f"Epoch {epoch}, Loss of train{losses.avg}, Acc of if {if_acc1}, PSNR of cs {best}\n")
            output_file.write(old)
            output_file.close()

        if epoch == 15:
            cs_validate(config, cs_net, cs_sampling, epoch=epoch, save_dir=save_dir)
            print(f"已在目录 '{save_dir}' 中保存了8张重构的图片。")
        scheduler.step()


# Validate inference accuracy on clean sampled features.
def if_validate(val_loader, model, criterion, config, cs_sampling):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progess = ProgressMeter(len(val_loader), batch_time,
                            losses, top1, top5, prefix='Test: ')
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(config.device)
            target = target.to(config.device)
            images = cs_sampling(images)
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            if i % config.print_freq == 0:
                progess.print(i)
        print(f"Acc@1 {top1.avg} Acc@5 {top5.avg}")
    return losses.avg, top1.avg


# Validate reconstruction quality on Set11.
def cs_validate(config, net, sampling, epoch=None, save_dir=None):
    batch_size = 1
    net = net.eval()
    sampling = sampling.eval()
    folder_name = "SET11"
    val_root = config.CS_val_path + '/'
    path = os.path.join(val_root, folder_name)
    img_list = []
    for root, dirs, file in os.walk(path):
        if root == path:
            for f in file:
                if f[-4:] != '.png':
                    continue
                img_list.append(f)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    p_total = 0
    saved_count = 0
    if save_dir is not None and epoch == 15:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, img_name in enumerate(img_list):
            name = os.path.join(path, img_name)
            img = Image.open(name).convert('L')
            img_tensor = transform(img).squeeze(0).to(config.device)
            x = img_tensor
            x = x.float()
            ori_x = x

            h = x.size()[0]
            h_lack = 0
            w = x.size()[1]
            w_lack = 0

            if h % config.block_size != 0:
                h_lack = config.block_size - h % config.block_size
                temp_h = torch.zeros(h_lack, w).to(config.device)
                h = h + h_lack
                x = torch.cat((x, temp_h), 0)
            if w % config.block_size != 0:
                w_lack = config.block_size - w % config.block_size
                temp_w = torch.zeros(h, w_lack).to(config.device)
                w = w + w_lack
                x = torch.cat((x, temp_w), 1)
            x = torch.unsqueeze(x, 0)
            x = torch.unsqueeze(x, 0)

            idx_h = range(0, h, config.block_size)
            idx_w = range(0, w, config.block_size)
            num_patches = h * w // (config.block_size ** 2)
            temp = torch.zeros(
                num_patches, batch_size, 1, config.block_size, config.block_size
            )
            count = 0
            for a in idx_h:
                for b in idx_w:
                    y, Phi, Q = sampling(
                        x[:, :, a:a + config.block_size, b:b + config.block_size])
                    [output, symloss, layers_st] = net(y, Phi, Q, batch_size)
                    temp[count, :, :, :, :] = output
                    count = count + 1
            y = torch.zeros(batch_size, 1, h, w)
            count = 0
            for a in idx_h:
                for b in idx_w:
                    y[:, :, a:a + config.block_size, b:b + config.block_size] = temp[count, :, :, :, :]
                    count = count + 1
            recon_x = y[:, :, 0:h - h_lack, 0:w - w_lack]
            recon_x = torch.squeeze(recon_x).to("cpu")
            ori_x = ori_x.to("cpu")

            mse = np.mean(np.square(recon_x.numpy() - ori_x.numpy()))
            p = 10 * np.log10(1 / mse)
            p_total = p_total + p


            if save_dir is not None and epoch == 15 and saved_count < 8:
                save_image(recon_x.unsqueeze(0), os.path.join(save_dir, f"epoch_{epoch}_recon_{saved_count + 1}.png"))
                saved_count += 1
                if saved_count >= 8:
                    break

        return p_total / len(img_list)


if __name__ == '__main__':
    main()