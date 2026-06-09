import os
import torch


class MainConfig:
    def __init__(self, ratio=0.5, lr=1e-4, device='0') -> None:
        self.arch = "B_32_imagenet1k"
        self.ratio = ratio
        
        # ==== 修改 1: 修复 NameError，将设备判断逻辑放入 __init__ ====
        # 如果传入的是 'cpu'，或者没有 GPU，则使用 cpu
        if device == 'cpu' or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f"cuda:{device}")
        # ========================================================

        self.epochs = 200
        self.init_size = 32
        self.block_size = 96
        self.IF_data_dir = '/home/compressedsensing/dataset/imagenet-100'
        self.start_epoch = 0
        self.if_batch_size = 50
        self.cs_batch_size = 64
        self.lr = lr
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.print_freq = 40
        self.resume_dir = ''
        self.seed = 22
        self.img_size = 384
        
        # ==== 修改 2: 使用新的实验结果目录 ====
        # Log dir
        # self.results = "./results/HPP" 
        self.results = "./results/HPP"
        # ====================================
        
        self.log = os.path.join(self.results, str(
            int(self.ratio * 100)), "log.txt")
        self.folder = os.path.join(self.results, str(
            int(self.ratio*100)), "models")
        self.sample = os.path.join(self.folder, "cs_sampling.pth")
        self.cs_model = os.path.join(self.folder, "cs_model.pth")
        self.if_model = os.path.join(self.folder, "if_model.pth")
        self.optimizer = os.path.join(self.folder, "optimizer.pth")
        self.hyp_params_dir=os.path.join(self.folder,"hyper_params.pth")
        self.info = os.path.join(self.folder, "info.pth")
        self.pretrain_weight = "./pretrain/B_32_imagenet1k.pth"

        self.CS_train_path = "./dataset/train"
        if not os.path.isdir(self.CS_train_path):
            os.mkdir(self.CS_train_path)
        self.CS_val_path = "./dataset/val"
        if not os.path.isdir(self.CS_val_path):
            os.mkdir(self.CS_val_path)
        self.CS_test_path = "./dataset/test"

    def check(self):
        if not os.path.isdir(self.results):
            os.mkdir(self.results)
        sub_dir = os.path.join(self.results, str(int(self.ratio*100)))
        if not os.path.isdir(sub_dir):
            os.mkdir(sub_dir)
            print(f"Make dir: {sub_dir}")
        models_path = os.path.join(sub_dir, "models")
        if not os.path.isdir(models_path):
            os.mkdir(models_path)
            print(f"Make dir: {models_path}")

    def show(self):
        print("\n=> Your configs are:")
        print("="*70)
        for item in self.__dict__:
            print("{:<20s}".format(item+":") +
                  "{:<30s}".format(str(self.__dict__[item])))
            print("-"*70)
        print("\n")


class GetIFConfig:
    def __init__(self, ratio=0.1, batch_size=64, device="cuda:0"):
        self.arch = "B_32_imagenet1k"
        self.ratio = ratio
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu")
        self.epochs = 1000
        self.init_size = 32
        self.block_size = 96
        self.data_dir = '/home/compressedsensing/dataset/imagenet-100'
        self.start_epoch = 0
        self.batch_size = batch_size
        self.lr = 1e-4
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.print_freq = 40
        self.resume_dir = ''
        self.seed = 22
        self.img_size = 384
        # Log dir
        self.results = "./results/IF"
        self.log = os.path.join(self.results, str(
            int(self.ratio * 100)), "log.txt")
        self.folder = os.path.join(self.results, str(
            int(self.ratio*100)), "models")
        self.sample = os.path.join(self.folder, "cs_sampling.pth")
        self.model = os.path.join(self.folder, "model.pth")
        self.optimizer = os.path.join(self.folder, "optimizer.pth")
        self.info = os.path.join(self.folder, "info.pth")
        self.pretrain_weight = "./pretrain/B_32_imagenet1k.pth"

    def check(self):
        if not os.path.isdir(self.results):
            os.mkdir(self.results)
        sub_dir = os.path.join(self.results, str(int(self.ratio*100)))
        if not os.path.isdir(sub_dir):
            os.mkdir(sub_dir)
            print(f"Make dir: {sub_dir}")
        models_path = os.path.join(sub_dir, "models")
        if not os.path.isdir(models_path):
            os.mkdir(models_path)
            print(f"Make dir: {models_path}")

    def show(self):
        print("\n=> Your configs are:")
        print("="*70)
        for item in self.__dict__:
            print("{:<20s}".format(item+":") +
                  "{:<30s}".format(str(self.__dict__[item])))
            print("-"*70)
        print("\n")


class GetConfig:
    def __init__(self, ratio=0.1, device="cuda:0", batch_size=64):
        self.ratio = ratio
        self.epochs = 200
        self.channel = 1
        self.init_size = 32
        self.block_size = 96
        self.batch_size = batch_size
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu")
        # self.init_sense = './dataset/init_sense_matrix1021024.xlsx'
        self.init_sense = './dataset/output.xlsx'
        # Paths
        self.results = "./results/CS"
        # self.results = "./results/res_init"
        self.log = os.path.join(self.results, str(
            int(self.ratio * 100)), "log.txt")

        self.folder = os.path.join(
            self.results, str(int(self.ratio * 100)), "models")
        self.model = os.path.join(self.folder, "model.pth")
        self.optimizer = os.path.join(self.folder, "optimizer.pth")
        self.info = os.path.join(self.folder, "info.pth")
        self.cs_sample = os.path.join(self.folder, "cs_sampling.pth")

        self.train_path = "./dataset/train"
        if not os.path.isdir(self.train_path):
            os.mkdir(self.train_path)
        self.val_path = "./dataset/val"
        if not os.path.isdir(self.val_path):
            os.mkdir(self.val_path)
        self.test_path = "./dataset/test"

    def check(self):
        if not os.path.isdir(self.results):
            os.mkdir(self.results)

        sub_path = os.path.join(self.results, str(int(self.ratio * 100)))
        if not os.path.isdir(sub_path):
            os.mkdir(sub_path)
            print("Mkdir: " + sub_path)

        models_path = os.path.join(sub_path, "models")
        if not os.path.isdir(models_path):
            os.mkdir(models_path)
            print("Mkdir: " + models_path)

    def show(self):
        print("\n=> Your configs are:")
        print("=" * 70)
        for item in self.__dict__:
            print("{:<20s}".format(item + ":") +
                  "{:<30s}".format(str(self.__dict__[item])))
            print("-" * 70)
        print("\n")


if __name__ == "__main__":
    config = GetConfig()
    config.check()
    config.show()