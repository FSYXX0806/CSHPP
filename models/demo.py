import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from itertools import repeat
import torch.nn.functional as F
# from torch._six import container_abcs
from torch.nn.modules.module import Module


import models


def _ntuple(n):
    def parse(x):
        # if isinstance(x, container_abcs.Iterable):
        #     return x
        return tuple(repeat(x, n))
    return parse


_pair = _ntuple(2)


class Conv(Module):
    def __init__(self, config, ic, oc):
        super(Conv, self).__init__()
        self.config = config
        self.ic = ic
        self.oc = oc
        self.w = nn.Parameter(torch.Tensor(oc, ic, 9))
        self.padding = _pair(1)
        self.init = nn.Parameter(torch.zeros([ic, 9, 9], dtype=torch.float32))
        init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, inputs):
        init = self.init + torch.eye(9, dtype=torch.float32).unsqueeze(0).repeat((self.ic, 1, 1)).to(self.config.device)
        weight = torch.reshape(torch.einsum('abc, dac->dab', init, self.w), (self.oc, self.ic, 3, 3))
        outputs = F.conv2d(inputs, weight, None, 1, self.padding)
        return outputs


class pre_layer(nn.Module):
    def __init__(self, config):
        super(pre_layer, self).__init__()

        self.num = 4

        self.conv_in = nn.Sequential(
            Conv(config, 1, 32),
            nn.BatchNorm2d(32),
            nn.ELU())

        self.conv = nn.ModuleList()
        for i in range(self.num):
            self.conv.append(nn.Sequential(
                Conv(config, 32, 32),
                nn.BatchNorm2d(32),
                nn.ELU())
            )

        self.conv_out = nn.Sequential(
            Conv(config, 32, 1))

    def forward(self, x_recon):
        x_recon = torch.transpose(x_recon, 0, 1).reshape([-1, 1, 32, 32])
        x_input = self.conv_in(x_recon)
        x_mid = x_input
        for i in range(self.num):
            x_mid = self.conv[i](x_mid)
        x_output = self.conv_out(x_mid)
        x_output = torch.transpose(x_output.reshape(-1, 1024), 0, 1)
        return x_output


class post_layer(nn.Module):
    def __init__(self, config):
        super(post_layer, self).__init__()

        self.num = 4

        self.conv_in = nn.Sequential(
            Conv(config, 1, 32),
            nn.BatchNorm2d(32),
            nn.ELU())

        self.conv = nn.ModuleList()
        for i in range(self.num):
            self.conv.append(nn.Sequential(
                Conv(config, 32, 32),
                nn.BatchNorm2d(32),
                nn.ELU())
            )

        self.conv_out = nn.Sequential(
            Conv(config, 32, 1))

    def forward(self, x_recon):
        x_input = self.conv_in(x_recon)
        x_mid = x_input
        for i in range(self.num):
            x_mid = self.conv[i](x_mid)
        x_output = self.conv_out(x_mid)
        return x_output


class Trans(nn.Module):
    def __init__(self, config, dim):
        super(Trans, self).__init__()
        self.config = config
        self.threshold = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)
        self.encoder = models.Encoder(dim=dim)
        self.decoder = models.Decoder(dim=dim)

    def forward(self, inputs):
        outputs = self.encoder(inputs)# 软阈值迭代
        outputs = torch.mul(torch.sign(outputs), F.relu(torch.abs(outputs) - self.threshold))
        outputs = self.decoder(inputs, outputs)
        return outputs


class HybridNet(nn.Module):
    def __init__(self, config):
        super(HybridNet, self).__init__()
        self.config = config
        self.phi_size = 32

         # two-step update weight
        self.Sp = nn.Softplus()
        self.w_rho = nn.Parameter(torch.Tensor([0.5]))#FAST的参数
        self.b_rho = nn.Parameter(torch.Tensor([0]))#FAST的参数   

        self.num_layers = 6
        self.pre_block = nn.ModuleList()
        self.n_output = self.phi_size**2
        self.n_input = config.ratio*self.n_output
        # self.batch_size = config.batch_size
        self.batch_size = config.cs_batch_size
        for i in range(self.num_layers):
            self.pre_block.append(pre_layer(config))

        self.post_block = nn.ModuleList()
        for i in range(self.num_layers):
            self.post_block.append(post_layer(config))

        self.trans = nn.ModuleList()
        for i in range(self.num_layers):
            self.trans.append(Trans(config, dim=8 ** 2))

        self.weights = []
        self.etas = []
        for i in range(self.num_layers):
            self.weights.append(nn.Parameter(
                torch.tensor(1.), requires_grad=True))
            self.register_parameter(
                "eta_" + str(i + 1), nn.Parameter(torch.tensor(0.1), requires_grad=True))  # todo
            self.etas.append(eval("self.eta_" + str(i + 1)))

    def forward(self, inputs, Phi, Q, batch_size):
        # input shape : [batch_size,1,96,96]
        # batch_size = inputs.size(0)
        y = inputs
        # print(y.size())  # [100,512,3,3]
        # print(Phi.size())
        # print(f"batch size:{batch_size}")  # 100
        # exit(0)
        recon = self.recon(y, self.phi_size, batch_size, Phi, Q)
        return recon

    def recon(self, y, init_block, batch_size, Phi, Q):

        idx = int(self.config.block_size / init_block)

        Qweight = Q.contiguous().view(int(self.n_output), int(self.n_input), 1, 1)
        # 初始重构
        recon = F.conv2d(y, Qweight, padding=0, bias=None)  # [100,1024,3,3]
        recon = torch.cat(torch.split(
            recon, split_size_or_sections=1, dim=3), dim=0)
        recon = torch.cat(torch.split(
            recon, split_size_or_sections=1, dim=2), dim=0)
        recon = torch.reshape(recon, [-1, 1024])
        recon = torch.transpose(recon, 0, 1)
        '''
        修改bug 由y [100,512,3,3] -> yo [512,900]
        必须先进行维度对齐再进行转置
        通过torch.split将数据对齐至batch_size的维度
        最后进行转置
        '''
        yo = torch.cat(torch.split(
            y, split_size_or_sections=1, dim=3
        ), dim=0)
        yo = torch.cat(torch.split(yo, split_size_or_sections=1, dim=2), dim=0)
        yo = torch.reshape(yo, [-1, yo.size()[1]])
        yo = torch.transpose(yo, 0, 1)

        yfast = recon #
        layers_sym = []#
        layers_st = []#

        # recon = torch.mm(Q, y)
        for i in range(self.num_layers):
            xold=recon#
            [recon,symloss,x_st]=self.fast2(yo,yfast, init_block, batch_size, idx, i, Phi)#恢复过程函数
            xnew=recon#
            rho_ = (self.Sp(self.w_rho * i + self.b_rho) -  self.Sp(self.b_rho)) / self.Sp(self.w_rho * i + self.b_rho)#
            yfast = xnew + rho_ * (xnew - xold) # two-step update
            layers_sym.append(symloss)
            layers_st.append(x_st)

        recon = torch.reshape(torch.transpose(recon, 0, 1),
                              [-1, 1, init_block, init_block])
        recon = torch.cat(torch.split(
            recon, split_size_or_sections=idx * batch_size, dim=0), dim=2)
        recon = torch.cat(torch.split(
            recon, split_size_or_sections=batch_size, dim=0), dim=3)
        # print(recon.size())
        # exit(0)
        return [recon,symloss,layers_st]

    def fast(self, y, yfast,init_block, batch_size, idx, i):
        
        recon = recon - self.weights[i] * torch.mm(torch.transpose(Phi, 0, 1), (torch.mm(Phi, recon) - y))#将算式里的recon换成了yfast
        recon = recon - self.pre_block[i](recon)
        recon = torch.reshape(torch.transpose(recon, 0, 1), [-1, 1, init_block, init_block])
        recon = torch.cat(torch.split(recon, split_size_or_sections=idx * batch_size, dim=0), dim=2)
        recon = torch.cat(torch.split(recon, split_size_or_sections=batch_size, dim=0), dim=3)
        recon = self.size256to8(recon)
        recon = recon - self.etas[i] * self.trans[i](recon)

        recon = self.size8to256(recon)
        recon = recon - self.post_block[i](recon)

        recon = torch.cat(torch.split(recon, split_size_or_sections=init_block, dim=3), dim=0)
        recon = torch.cat(torch.split(recon, split_size_or_sections=init_block, dim=2), dim=0)
        recon = torch.reshape(recon, [-1, init_block ** 2])
        recon = torch.transpose(recon, 0, 1)
        return recon

    def fast2(self, yo, yfast,init_block, batch_size, idx, i, Phi):
         
        recon = yfast - self.weights[i] * torch.mm(torch.transpose(Phi, 0, 1), (torch.mm(Phi, yfast) - yo))#压缩感知的梯度下降模块,用的是FISTA算法
        
        x_f=recon
        recon = recon - self.pre_block[i](recon)#ISTA-Net的CNN模块的一部分
        # 重构图像块
        recon = torch.reshape(torch.transpose(recon, 0, 1), [-1, 1, init_block, init_block])
        recon = torch.cat(torch.split(recon, split_size_or_sections=idx * batch_size, dim=0), dim=2)
        recon = torch.cat(torch.split(recon, split_size_or_sections=batch_size, dim=0), dim=3)#Fvec的逆
        recon = self.size256to8(recon)#FB一撇
        x_st=self.trans[i](recon)#进入Transformer模块
        recon = recon - self.etas[i] * x_st
        recon = self.size8to256(recon)#FB一撇逆
        recon = recon - self.post_block[i](recon)#ISTA-Net的CNN模块的另一部分
        # 返回重构后的数据
        recon = torch.cat(torch.split(recon, split_size_or_sections=init_block, dim=3), dim=0)
        recon = torch.cat(torch.split(recon, split_size_or_sections=init_block, dim=2), dim=0)#Fvec
        recon = torch.reshape(recon, [-1, init_block ** 2])
        recon = torch.transpose(recon, 0, 1)
        symloss=recon-x_f
        return [recon,symloss,x_st]


    def size8to256(self, inputs):
        idx = int(self.config.block_size / 8)
        # print(f"idx is {idx}")  # 12
        # print(f"inputs size {inputs.size()}")  # [64,144,8,8]
        outputs = torch.cat(torch.split(
            inputs, split_size_or_sections=idx, dim=1), dim=2)  # [64,12,96,8]

        outputs = torch.cat(torch.split(
            outputs, split_size_or_sections=1, dim=1), dim=3)  # [64,1,96,96]
        return outputs

    def size256to8(self, inputs):
        inputs = torch.cat(torch.split(
            inputs, split_size_or_sections=8, dim=3), dim=1)
        inputs = torch.cat(torch.split(
            inputs, split_size_or_sections=8, dim=2), dim=1)
        return inputs

# if __name__ == '__main__':
#     # cs_sampling = CS_Sampling(n_channels=3, cs_ratio=0.25, blocksize=16, im_size=384)
#     # cs_sampling = CS_Sampling_shuffle(n_channels=3, cs_ratio=0.25, blocksize=16)
#     input_img = torch.randn(2, 3,333 , 500)
#     sampling=Sampling()
#     output_img = sampling(input_img)
#     print(output_img.shape)
#     # print(cs_sampling.PhiB.size())
