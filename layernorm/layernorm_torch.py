import torch
import torch.nn as nn

class TorchLayerNorm(nn.Module):
    # 归一化层输出的以为项链过的大小, 定义eps, 可学习参数gamma(权重缩放参数)以及β(平移参数)是否启用, 
    def __init__(self, hidden_size, eps=1e-5, elementwise_affine=True, device="cuda"):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=elementwise_affine).to(device)

    def forward(self, x):
        return self.ln(x)
