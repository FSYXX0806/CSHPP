
# import torch
# import torch.nn as nn
# from vit import ViT 

# class InferenceNet(nn.Module):
#     def __init__(self, phi_matrix, vit_config, arch='resnet50'):
#         super(InferenceNet, self).__init__()
        
#         # 获取维度: [M, N]
#         M, N = phi_matrix.shape
        
#         # [修复] 为了匹配 checkpoint 中的 "projection.weight"
#         # 我们使用 nn.Linear 替代 register_buffer
#         self.projection = nn.Linear(M, N, bias=False)
        
#         # 初始化权重 (虽然 load_state_dict 会覆盖它，但初始化一下更安全)
#         # nn.Linear 的 weight 是 [Out, In] 即 [N, M]，所以需要转置 phi
#         with torch.no_grad():
#             self.projection.weight.copy_(phi_matrix.t())
            
#         # 固定投影层 (通常 MLaaS 中这一层是不更新的，或者与 phi 绑定)
#         # 如果你 step2 训练时没锁这一层，那它可能微调过，这里加载回来正好
#         for param in self.projection.parameters():
#             param.requires_grad = False
        
#         # 后端分类器 (ViT)
#         # 保持之前的修复：直接传 arch
#         self.classifier = ViT(
#             arch,
#             image_size=vit_config['image_size'],
#             num_classes=vit_config['num_classes'],
#             pretrained=True, 
#             weights_path=vit_config.get('weights_path', None)
#         )

#     def forward(self, y_blocks, shape_info):
#         """
#         y_blocks: [Total_Blocks, M] (测量值)
#         shape_info: (B, C, H, W) 用于重组图片
#         """
#         B, C, H, W = shape_info
        
#         # 1. 线性投影 (Linear Projection)
#         # 使用 projection 层替代 @ self.phi
#         # checkpoint 中是 projection.weight，对应 y @ weight.T
#         x_proxy_flat = self.projection(y_blocks)
        
#         # 2. 重组回图像 (Reshape)
#         # N 是输出维度
#         N = self.projection.out_features
#         block_size = int(N**0.5)
        
#         h_blocks = H // block_size
#         w_blocks = W // block_size
        
#         # View: [B, C, h_b, w_b, b, b]
#         x_view = x_proxy_flat.view(B, C, h_blocks, w_blocks, block_size, block_size)
        
#         # Permute: [B, C, h_b, b, w_b, b] -> [B, C, H, W]
#         x_img = x_view.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
        
#         # 3. ViT 分类
#         logits = self.classifier(x_img)
        
#         return logits

import torch
import torch.nn as nn
from vit import ViT 

class InferenceNet(nn.Module):
    def __init__(self, phi_matrix, vit_config, arch='resnet50'):
        super(InferenceNet, self).__init__()
        
        # 1. 获取维度: [M, N]
        # M: 压缩后的维度 (测量值)
        # N: 原始块向量维度 (e.g., 32*32=1024)
        M, N = phi_matrix.shape
        
        # 2. 线性投影层 (Proxy Reconstruction)
        # 这一层充当 "解码器"，将测量值 y 映射回近似的图像块 x'
        # 我们使用 nn.Linear 替代 register_buffer 以匹配 checkpoint 中的 "projection.weight"
        self.projection = nn.Linear(M, N, bias=False)
        
        # 3. 初始化权重
        # 理论上最佳初始化是 Phi 的伪逆，这里使用转置 Phi.T 作为近似
        # nn.Linear(M, N) 的权重形状是 [N, M]，所以需要传入 phi_matrix.t()
        with torch.no_grad():
            self.projection.weight.copy_(phi_matrix.t())
            
        # 4. 固定投影层
        # MLaaS 场景下，用户 B 无法更新测量矩阵及其逆变换，因此这一层通常是固定的
        for param in self.projection.parameters():
            param.requires_grad = False
        
        # 5. 后端分类器 (ViT)
        self.classifier = ViT(
            arch,
            image_size=vit_config['image_size'],
            num_classes=vit_config['num_classes'],
            pretrained=True, 
            weights_path=vit_config.get('weights_path', None)
        )

    def forward(self, y_blocks, shape_info):
        """
        前向传播: 测量域 -> 图像域 -> 分类结果
        
        Args:
            y_blocks: [Total_Blocks, M] 展平的测量值
            shape_info: (B, C, H, W) 原始图像的形状信息，用于重组
        """
        B, C, H, W = shape_info
        
        # 1. 线性投影 (Linear Projection)
        # y [Total, M] -> x' [Total, N]
        x_proxy_flat = self.projection(y_blocks)
        
        # 2. 重组回图像 (Reshape / Un-patchify)
        # 获取块大小 (假设 N = block_size^2)
        N = self.projection.out_features
        block_size = int(N**0.5)
        
        h_blocks = H // block_size
        w_blocks = W // block_size
        
        # View: 将平铺的向量还原为块张量
        # [B*C*h*w, b*b] -> [B, C, h_blocks, w_blocks, b, b]
        x_view = x_proxy_flat.view(B, C, h_blocks, w_blocks, block_size, block_size)
        
        # Permute: 交换维度以还原图像空间结构
        # [B, C, h, w, b, b] -> [B, C, h, b, w, b] -> [B, C, H, W]
        x_img = x_view.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
        
        # 3. ViT 分类推理
        logits = self.classifier(x_img)
        
        return logits