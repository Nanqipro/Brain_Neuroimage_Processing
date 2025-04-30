import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder

def generate_continuous_mask(B, T, n=5, l=0.1):
    """
    生成连续的掩码矩阵，用于屏蔽时间序列中的连续段落。
    
    Parameters
    ----------
    B : int
        批次大小
    T : int
        时间序列长度
    n : int 或 float, 可选
        每个样本中掩码段落的数量，默认为5。如果为浮点数，则视为T的比例
    l : int 或 float, 可选
        每个掩码段落的长度，默认为0.1。如果为浮点数，则视为T的比例
        
    Returns
    -------
    torch.Tensor
        布尔型掩码张量，形状为 [B, T]，其中False表示被掩蔽的位置
    """
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    """
    生成二项分布掩码矩阵，用于随机屏蔽时间序列中的点。
    
    Parameters
    ----------
    B : int
        批次大小
    T : int
        时间序列长度
    p : float, 可选
        每个点被保留（值为True）的概率，默认为0.5
        
    Returns
    -------
    torch.Tensor
        布尔型掩码张量，形状为 [B, T]，其中False表示被掩蔽的位置
    """
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class TSEncoder(nn.Module):
    """
    时间序列编码器，使用膨胀卷积网络提取特征。
    
    该编码器能够处理缺失值，并支持多种掩码策略进行训练。
    
    Parameters
    ----------
    input_dims : int
        输入特征维度
    output_dims : int
        输出特征维度
    hidden_dims : int, 可选
        隐藏层维度，默认为64
    depth : int, 可选
        膨胀卷积的深度，默认为10
    mask_mode : str, 可选
        掩码生成模式，默认为'binomial'
    """
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3 # TODO 3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):  # x: B x T x input_dims
        """
        前向传播函数。
        
        Parameters
        ----------
        x : torch.Tensor
            输入时间序列张量，形状为 [batch_size, seq_len, input_dims]
        mask : torch.Tensor 或 str, 可选
            掩码张量或掩码生成模式，默认为None，在训练时使用self.mask_mode指定的模式
            
        Returns
        -------
        torch.Tensor
            编码后的特征张量，形状为 [batch_size, seq_len, output_dims]
        """
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        
        return x
        