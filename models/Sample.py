import torch
import torch.nn as nn
from torch.nn import init
import cv2
import numpy as np
import random
import torch.nn.functional as F


class HPP_sampling(torch.nn.Module):
    def __init__(self, cs_ratio=0.25, block_size=32, im_size=384):
        super(HPP_sampling, self).__init__()
        print('HPP')
        self.blocksize = block_size
        self.phi_size = block_size
        self.im_size = im_size
        points = block_size**2  # 1024
        self.n_input = int(cs_ratio*points)
        self.n_output = points
        phi_init = np.random.normal(
            0.0, (1/points)**0.5, size=(int(self.n_input), int(self.n_output))
        )
        self.phi = nn.Parameter(torch.from_numpy(
            phi_init).float(), requires_grad=True)
        self.Q = nn.Parameter(torch.from_numpy(
            np.transpose(phi_init)).float(), requires_grad=True)

    def forward(self, x):
        # [100,1,96,96]
        if x.size()[1] == 1:
            Phi = self.phi
            Q = self.Q
            PhiWeight = Phi.contiguous().reshape(
                int(self.n_input), 1, self.phi_size, self.phi_size)
            # [102,1024] ->[102,1,32,32]
            # 利用卷积进行采样 转化为 [n_input,phi_size]  phi
            # print(PhiWeight.size())
            # print(x[:, 0:1, :, :].size())
            Phix = F.conv2d(x[:, 0:1, :, :], PhiWeight, padding=0,
                            stride=self.phi_size, bias=None)   # Get measurements
            y = Phix  # [100,102,3,3]

            return y, Phi, Q
        else:
            Phi_R = self.phi
            Phi_G = self.phi
            Phi_B = self.phi
            # Sampling base matrix
            PhiWeight_R = Phi_R.contiguous().view(
                int(self.n_input), 1, self.blocksize, self.blocksize)
            PhiWeight_G = Phi_G.contiguous().view(
                int(self.n_input), 1, self.blocksize, self.blocksize)
            PhiWeight_B = Phi_B.contiguous().view(
                int(self.n_input), 1, self.blocksize, self.blocksize)
            # 利用卷积进行采样 转化为 [n_input,blocksize,blocksize]  phi
            # print(x.size())
            Phix_R = F.conv2d(x[:, 0:1, :, :], PhiWeight_R, padding=0,
                              stride=self.blocksize, bias=None)  # Get measurements
            Phix_G = F.conv2d(x[:, 1:2, :, :], PhiWeight_G, padding=0,
                              stride=self.blocksize, bias=None)  # Get measurements
            Phix_B = F.conv2d(x[:, 2:3, :, :], PhiWeight_B, padding=0,
                              stride=self.blocksize, bias=None)  # Get measurements
            # print(f"phix :{Phix_R.size()}")  # [2,64,30,31]
            # Initialization-subnet y=phi*x
            # 得到phiw的转置矩阵

            PhiTWeight_R = self.Q.t().contiguous().view(self.n_output, self.n_input, 1, 1)
            PhiTb_R = F.conv2d(Phix_R, PhiTWeight_R, padding=0, bias=None)
            # phiT size torch.Size([2, 256, 20, 31])
            # print(f'phiT size {PhiTb_R.size()}')
            PhiTb_R = torch.nn.PixelShuffle(self.blocksize)(PhiTb_R)
            x_R = PhiTb_R  # Conduct initialization

            PhiTWeight_G = self.Q.t().contiguous().view(self.n_output, self.n_input, 1, 1)
            PhiTb_G = F.conv2d(Phix_G, PhiTWeight_G, padding=0, bias=None)
            PhiTb_G = torch.nn.PixelShuffle(self.blocksize)(PhiTb_G)
            x_G = PhiTb_G

            PhiTWeight_B = self.Q.t().contiguous().view(self.n_output, self.n_input, 1, 1)
            PhiTb_B = F.conv2d(Phix_B, PhiTWeight_B, padding=0, bias=None)
            PhiTb_B = torch.nn.PixelShuffle(self.blocksize)(PhiTb_B)
            x_B = PhiTb_B

            x = torch.cat([x_R, x_G, x_B], dim=1)
            # print(PhiTb_B.size())
            # print(x.size())
            # 使用双线性插值法对图像进行上采样
            x = F.interpolate(
                x, size=(self.im_size, self.im_size), mode='bilinear')

            return x


class Sampling_conv_from_init(torch.nn.Module):
    '''
    return y Phi Q
    '''

    def __init__(self, cs_ratio=0.25, block_size=32, phi_init=None):
        super(Sampling_conv_from_init, self).__init__()
        print('cs')
        self.phi_size = block_size
        points = block_size**2  # 1024
        self.n_input = cs_ratio*points
        self.n_output = points
        self.phi = nn.Parameter(torch.from_numpy(
            phi_init).float(), requires_grad=True)
        self.Q = nn.Parameter(torch.from_numpy(
            np.transpose(phi_init)).float(), requires_grad=True)

    def forward(self, x):
        # [100,1,96,96]
        Phi = self.phi
        Q = self.Q
        PhiWeight = Phi.contiguous().reshape(
            int(self.n_input), 1, self.phi_size, self.phi_size)
        # [102,1024] ->[102,1,32,32]
        # 利用卷积进行采样 转化为 [n_input,phi_size]  phi
        # print(PhiWeight.size())
        # print(x[:, 0:1, :, :].size())
        Phix = F.conv2d(x[:, 0:1, :, :], PhiWeight, padding=0,
                        stride=self.phi_size, bias=None)   # Get measurements
        y = Phix  # [100,102,3,3]
        return y, Phi, Q


class Sampling_conv(torch.nn.Module):
    '''
    return y Phi Q
    '''

    def __init__(self, cs_ratio=0.25, block_size=32):
        super(Sampling_conv, self).__init__()
        print('cs')
        self.phi_size = block_size
        points = block_size**2  # 1024
        self.n_input = cs_ratio*points
        self.n_output = points
        phi_init = np.random.normal(
            0.0, (1/points)**0.5, size=(int(self.n_input), int(self.n_output))
        )
        self.phi = nn.Parameter(torch.from_numpy(
            phi_init).float(), requires_grad=True)
        self.Q = nn.Parameter(torch.from_numpy(
            np.transpose(phi_init)).float(), requires_grad=True)
        # self.phi = nn.Parameter(init.xavier_normal_(torch.Tensor(
        #     int(cs_ratio * points), points)).float(), requires_grad=True)
        # self.Q = nn.Parameter(torch.from_numpy(
        #     self.phi.t().detach().numpy()), requires_grad=True)

    def forward(self, x):
        # [100,1,96,96]

        Phi = self.phi
        Q = self.Q
        PhiWeight = Phi.contiguous().reshape(
            int(self.n_input), 1, self.phi_size, self.phi_size)
        # [102,1024] ->[102,1,32,32]
        # 利用卷积进行采样 转化为 [n_input,phi_size]  phi
        # print(PhiWeight.size())
        # print(x[:, 0:1, :, :].size())
        Phix = F.conv2d(x[:, 0:1, :, :], PhiWeight, padding=0,
                        stride=self.phi_size, bias=None)   # Get measurements
        y = Phix  # [100,102,3,3]

        return y, Phi, Q


class CS_Sampling4CL(torch.nn.Module):
    def __init__(self, n_channels=1, cs_ratio=0.25, block_size=32, im_size=384):
        super(CS_Sampling4CL, self).__init__()
        print('CS')

        points = block_size**2
        self.n_output = int(points)
        self.n_input = int(cs_ratio*points)

        phi_init = np.random.normal(
            0.0, (1/points)**0.5, size=(int(self.n_input), int(self.n_output))
        )
        self.phi = nn.Parameter(torch.from_numpy(
            phi_init).float(), requires_grad=True)
        self.Q = nn.Parameter(torch.from_numpy(
            np.transpose(phi_init)).float(), requires_grad=True)
        self.init_block = block_size

        # self.phi = nn.Parameter(init.xavier_normal_(torch.Tensor(
        #     int(config.ratio * points), points)).float(), requires_grad=True)
        # self.Q = nn.Parameter(torch.from_numpy(
        #     self.phi.t().detach().numpy()), requires_grad=True)

    def forward(self, x):
        # x [1,1,96,96]
        x = torch.cat(torch.split(
            x, split_size_or_sections=self.init_block, dim=3), dim=0)
        x = torch.cat(torch.split(
            x, split_size_or_sections=self.init_block, dim=2), dim=0)
        # print(x.size())  # torch.Size([900, 1, 32, 32])
        x = torch.reshape(x, [-1, self.init_block**2])
        x = torch.transpose(x, 0, 1)
        y = torch.matmul(self.phi, x)

        # print(y.size())  # 512 900
        # exit(0)
        return y, self.phi, self.Q


class CS_Sampling_from_init(torch.nn.Module):
    def __init__(self, n_channels=3, cs_ratio=0.25, blocksize=32, im_size=384, phi=None, Q=None):
        super(CS_Sampling_from_init, self).__init__()
        print('BCS from CS checkpoint')
        self.phi = phi
        self.Q = Q
        n_output = int(blocksize ** 2)
        n_input = int(cs_ratio * n_output)

        self.PhiR = nn.Parameter(phi, requires_grad=False)
        self.PhiG = nn.Parameter(phi, requires_grad=False)
        self.PhiB = nn.Parameter(phi, requires_grad=False)
        # self.PhiG = nn.Parameter(init.xavier_normal_(
        #     torch.Tensor(n_input, n_output)))
        # self.PhiB = nn.Parameter(init.xavier_normal_(
        #     torch.Tensor(n_input, n_output)))

        self.n_channels = n_channels
        self.n_input = n_input
        self.n_output = n_output
        self.blocksize = blocksize  # [32,32]
        self.im_size = im_size

    def forward(self, x):
        # [3,384,384]
        Phi_R = self.PhiR
        Phi_G = self.PhiG
        Phi_B = self.PhiB
        # Sampling base matrix
        PhiWeight_R = Phi_R.contiguous().view(
            int(self.n_input), 1, self.blocksize, self.blocksize)
        PhiWeight_G = Phi_G.contiguous().view(
            int(self.n_input), 1, self.blocksize, self.blocksize)
        PhiWeight_B = Phi_B.contiguous().view(
            int(self.n_input), 1, self.blocksize, self.blocksize)
        # 利用卷积进行采样 转化为 [n_input,blocksize,blocksize]  phi
        # print(x.size())
        Phix_R = F.conv2d(x[:, 0:1, :, :], PhiWeight_R, padding=0,
                          stride=self.blocksize, bias=None)  # Get measurements
        Phix_G = F.conv2d(x[:, 1:2, :, :], PhiWeight_G, padding=0,
                          stride=self.blocksize, bias=None)  # Get measurements
        Phix_B = F.conv2d(x[:, 2:3, :, :], PhiWeight_B, padding=0,
                          stride=self.blocksize, bias=None)  # Get measurements
        # print(f"phix :{Phix_R.size()}")  # [2,64,30,31]
        # Initialization-subnet y=phi*x
        # 得到phiw的转置矩阵

        PhiTWeight_R = self.Q.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_R = F.conv2d(Phix_R, PhiTWeight_R, padding=0, bias=None)
        # phiT size torch.Size([2, 256, 20, 31])
        # print(f'phiT size {PhiTb_R.size()}')
        PhiTb_R = torch.nn.PixelShuffle(self.blocksize)(PhiTb_R)
        x_R = PhiTb_R  # Conduct initialization

        PhiTWeight_G = self.Q.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_G = F.conv2d(Phix_G, PhiTWeight_G, padding=0, bias=None)
        PhiTb_G = torch.nn.PixelShuffle(self.blocksize)(PhiTb_G)
        x_G = PhiTb_G

        PhiTWeight_B = self.Q.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_B = F.conv2d(Phix_B, PhiTWeight_B, padding=0, bias=None)
        PhiTb_B = torch.nn.PixelShuffle(self.blocksize)(PhiTb_B)
        x_B = PhiTb_B

        x = torch.cat([x_R, x_G, x_B], dim=1)
        # print(PhiTb_B.size())
        # print(x.size())
        # 使用双线性插值法对图像进行上采样
        x = F.interpolate(
            x, size=(self.im_size, self.im_size), mode='bilinear')

        return x


class CS_Sampling(torch.nn.Module):
    def __init__(self, n_channels=3, cs_ratio=0.25, blocksize=32, im_size=384):
        super(CS_Sampling, self).__init__()
        print('BCS')

        n_output = int(blocksize ** 2)
        n_input = int(cs_ratio * n_output)

        self.PhiR = nn.Parameter(init.xavier_normal_(
            torch.Tensor(n_input, n_output)))
        self.PhiG = nn.Parameter(init.xavier_normal_(
            torch.Tensor(n_input, n_output)))
        self.PhiB = nn.Parameter(init.xavier_normal_(
            torch.Tensor(n_input, n_output)))

        self.n_channels = n_channels
        self.n_input = n_input
        self.n_output = n_output
        self.blocksize = blocksize  # [32,32]

        self.im_size = im_size

    def forward(self, x):
        # [3,384,384]
        Phi_R = self.PhiR
        Phi_G = self.PhiG
        Phi_B = self.PhiB
        # Sampling base matrix
        PhiWeight_R = Phi_R.contiguous().view(
            int(self.n_input), 1, self.blocksize, self.blocksize)
        PhiWeight_G = Phi_G.contiguous().view(
            int(self.n_input), 1, self.blocksize, self.blocksize)
        PhiWeight_B = Phi_B.contiguous().view(
            int(self.n_input), 1, self.blocksize, self.blocksize)
        # 利用卷积进行采样 转化为 [n_input,blocksize,blocksize]  phi
        Phix_R = F.conv2d(x[:, 0:1, :, :], PhiWeight_R, padding=0,
                          stride=self.blocksize, bias=None)  # Get measurements
        Phix_G = F.conv2d(x[:, 1:2, :, :], PhiWeight_G, padding=0,
                          stride=self.blocksize, bias=None)  # Get measurements
        Phix_B = F.conv2d(x[:, 2:3, :, :], PhiWeight_B, padding=0,
                          stride=self.blocksize, bias=None)  # Get measurements
        # print(f"phix :{Phix_R.size()}")  # [2,64,30,31]
        # Initialization-subnet y=phi*x
        # 得到phiw的转置矩阵
        PhiTWeight_R = Phi_R.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_R = F.conv2d(Phix_R, PhiTWeight_R, padding=0, bias=None)
        # phiT size torch.Size([2, 256, 20, 31])
        # print(f'phiT size {PhiTb_R.size()}')
        PhiTb_R = torch.nn.PixelShuffle(self.blocksize)(PhiTb_R)
        x_R = PhiTb_R  # Conduct initialization

        PhiTWeight_G = Phi_G.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_G = F.conv2d(Phix_G, PhiTWeight_G, padding=0, bias=None)
        PhiTb_G = torch.nn.PixelShuffle(self.blocksize)(PhiTb_G)
        x_G = PhiTb_G

        PhiTWeight_B = Phi_B.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_B = F.conv2d(Phix_B, PhiTWeight_B, padding=0, bias=None)
        PhiTb_B = torch.nn.PixelShuffle(self.blocksize)(PhiTb_B)
        x_B = PhiTb_B

        x = torch.cat([x_R, x_G, x_B], dim=1)
        # print(PhiTb_B.size())
        # print(x.size())
        # 使用双线性插值法对图像进行上采样
        x = F.interpolate(
            x, size=(self.im_size, self.im_size), mode='bilinear')

        return x


# if __name__ == '__main__':
    # cs_sampling = CS_Sampling4CL(cs_ratio=0.5)
    # cs_sampling = Sampling_conv(0.5)
    # img = torch.randn(1, 1, 96, 96)
    # # output = cs_sampling(img)
    # output, phi, q = cs_sampling(img)
    # print(output.size())
    # print(phi.size())

    # cs_sampling = CS_Sampling(
    #     n_channels=3, cs_ratio=0.1, blocksize=32, im_size=384)
    # cs_sampling4cl = CS_Sampling4CL(cs_ratio=0.1)
    # cs_conv = Sampling_conv()
    # # cs_sampling = CS_Sampling(n_channels=3, cs_ratio=0.25, blocksize=16)
    # input_img = torch.randn(2, 3, 512, 512)
    # output_img = cs_sampling(input_img)
    # _, _, output_imgcv = cs_conv(input_img)
    # # print(input_img[:, 0:1, :, :].size())
    # output_img4cl = cs_sampling4cl(input_img[:, 0:1, :, :])
    # print(f'cs :{output_img.shape}')
    # print(f'cs4cl :{output_img4cl.shape}')
    # print(f'cs conv {output_imgcv.shape}')
    # # print(cs_sampling.PhiB.size())
