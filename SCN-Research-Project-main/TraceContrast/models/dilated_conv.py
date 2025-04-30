import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class SamePadConv(nn.Module):
    """
    具有相同填充的一维卷积层。
    
    该层确保输出张量的长度与输入张量一致，通过自动计算适当的填充值。
    
    Parameters
    ----------
    in_channels : int
        输入通道数
    out_channels : int
        输出通道数
    kernel_size : int
        卷积核大小
    dilation : int, 可选
        膨胀率, 默认为1
    groups : int, 可选
        分组卷积的组数, 默认为1
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        """
        前向传播函数。
        
        Parameters
        ----------
        x : torch.Tensor
            输入张量，形状为 [batch_size, in_channels, seq_len]
            
        Returns
        -------
        torch.Tensor
            输出张量，形状为 [batch_size, out_channels, seq_len]
        """
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    """
    卷积块，包含两个相同填充的卷积层和残差连接。
    
    Parameters
    ----------
    in_channels : int
        输入通道数
    out_channels : int
        输出通道数
    kernel_size : int
        卷积核大小
    dilation : int
        膨胀率
    final : bool, 可选
        是否为最终块，默认为False
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        """
        前向传播函数。
        
        Parameters
        ----------
        x : torch.Tensor
            输入张量，形状为 [batch_size, in_channels, seq_len]
            
        Returns
        -------
        torch.Tensor
            输出张量，形状为 [batch_size, out_channels, seq_len]
        """
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    """
    膨胀卷积编码器，由多个卷积块组成，每个块的膨胀率逐渐增加。
    
    该编码器能够捕获长距离依赖关系，同时保持计算效率。
    
    Parameters
    ----------
    in_channels : int
        输入通道数
    channels : list
        每个卷积块的输出通道数列表
    kernel_size : int
        所有卷积块使用的卷积核大小
    """
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        """
        前向传播函数。
        
        Parameters
        ----------
        x : torch.Tensor
            输入张量，形状为 [batch_size, in_channels, seq_len]
            
        Returns
        -------
        torch.Tensor
            输出张量，形状为 [batch_size, channels[-1], seq_len]
        """
        return self.net(x)
