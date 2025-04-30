import torch
from torch import nn
import torch.nn.functional as F

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    """
    计算层次化对比损失，结合实例级对比损失和时间级对比损失。
    
    该函数通过递归下采样的方式，在多个尺度上计算对比损失。
    
    Parameters
    ----------
    z1 : torch.Tensor
        第一个特征表示张量，形状为 [batch_size, seq_len, feature_dim]
    z2 : torch.Tensor
        第二个特征表示张量，形状为 [batch_size, seq_len, feature_dim]
    alpha : float, 可选
        实例级对比损失的权重，默认为0.5
    temporal_unit : int, 可选
        开始计算时间级对比损失的层级，默认为0
        
    Returns
    -------
    torch.Tensor
        计算得到的层次化对比损失值，标量
    """
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2):
    """
    计算实例级对比损失，用于区分不同批次样本的表示。
    
    该损失鼓励来自同一样本的两种表示相似，而来自不同样本的表示不同。
    
    Parameters
    ----------
    z1 : torch.Tensor
        第一个特征表示张量，形状为 [batch_size, seq_len, feature_dim]
    z2 : torch.Tensor
        第二个特征表示张量，形状为 [batch_size, seq_len, feature_dim]
        
    Returns
    -------
    torch.Tensor
        计算得到的实例级对比损失值，标量
    """
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    """
    计算时间级对比损失，用于区分序列中不同时间点的表示。
    
    该损失鼓励同一时间点的两种表示相似，而不同时间点的表示不同。
    
    Parameters
    ----------
    z1 : torch.Tensor
        第一个特征表示张量，形状为 [batch_size, seq_len, feature_dim]
    z2 : torch.Tensor
        第二个特征表示张量，形状为 [batch_size, seq_len, feature_dim]
        
    Returns
    -------
    torch.Tensor
        计算得到的时间级对比损失值，标量
    """
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss
