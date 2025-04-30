import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tslearn.metrics import dtw, dtw_path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import warnings
import math
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
from pytorch_wavelets import DWT1DForward, DWT1DInverse


"""
TraceContrast模型实现

本模块实现了TraceContrast模型，该模型使用对比学习方法对神经元钙信号时间序列数据进行表征学习。
模型的主要组件包括：
1. 自定义数据集类(MyDataset) - 用于加载和预处理SCN神经元钙信号数据
2. TraceContrast模型类 - 包含模型架构定义、训练和推理逻辑
3. 辅助函数 - 用于数据处理和模型评估

作者: SCN研究团队
日期: 2023
"""


class MyDataset(Dataset):
    """
    自定义数据集类，用于加载SCN神经元钙信号数据
    
    属性:
        data_path (str): 数据文件路径
        trace_size (tuple): 轨迹数据的形状 (n_traces, trace_length)
        temporal_unit (int): 时间单位，用于数据增强
    """
    def __init__(self, data_path, trace_size, temporal_unit=0):
        """
        初始化数据集
        
        参数:
            data_path (str): 数据文件路径，支持npy或npz格式
            trace_size (tuple): 轨迹数据的形状 (n_traces, trace_length)
            temporal_unit (int): 时间单位，用于数据增强和窗口切分
        """
        super(MyDataset, self).__init__()
        self.data_path = data_path
        self.trace_size = trace_size
        self.temporal_unit = temporal_unit
        
        # 加载数据文件
        if '.npy' in data_path:
            self.traces = np.load(data_path)
        elif '.npz' in data_path:
            self.traces = np.load(data_path)['traces']
        else:
            raise ValueError(f"不支持的文件格式: {data_path}")
        
        # 打印加载的数据形状
        print(f"轨迹数据形状: {self.traces.shape}")
    
    def __len__(self):
        """
        返回数据集中样本的数量
        
        返回:
            int: 数据集大小
        """
        return len(self.traces)
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        
        参数:
            idx (int): 样本索引
        
        返回:
            dict: 包含轨迹数据和索引的字典
        """
        # 获取指定索引的轨迹数据
        trace = self.traces[idx]
        if trace.shape[-1] != self.trace_size[-1]:
            # 如果形状不符合要求，进行转置
            trace = trace.T.reshape(-1, self.trace_size[-1])
        
        # 确保数据类型为float32
        if not torch.is_tensor(trace):
            trace = torch.FloatTensor(trace)
            
        return {'trace': trace, 'idx': idx}


class TSEncoder(nn.Module):
    """
    时间序列编码器 (Time Series Encoder)
    
    使用多层级的卷积网络将时间序列数据编码为潜在表示。
    该编码器能够捕获时间序列数据中的时间依赖关系和模式特征。
    
    属性:
        input_dims (int): 输入数据的维度
        output_dims (int): 输出特征的维度
        hidden_dims (int): 隐藏层的维度
        depth (int): 网络深度，决定卷积层的数量
        mask_mode (str): 掩码模式，可选'binomial'或'continuous'
    """
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        """
        初始化时间序列编码器
        
        参数:
            input_dims (int): 输入数据的维度
            output_dims (int): 输出特征的维度
            hidden_dims (int): 隐藏层的维度，默认为64
            depth (int): 网络深度，默认为10
            mask_mode (str): 掩码模式，默认为'binomial'
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入时间序列数据，形状为[B, L, C]
            mask (torch.Tensor, optional): 掩码张量，用于屏蔽部分数据
            
        返回:
            torch.Tensor: 编码后的特征表示
        """
        nan_mask = ~x.isnan().any(dim=-1)
        if mask is None:
            if self.training and self.mask_mode == 'binomial':
                # 在训练过程中使用二项分布生成随机掩码
                mask = torch.rand(x.shape[0], x.shape[1], device=x.device)
                mask = mask.ge(0.2).bool()
                mask = mask & nan_mask
            elif self.training and self.mask_mode == 'continuous':
                # 在训练过程中使用连续分布生成随机掩码
                mask = torch.rand(x.shape[0], device=x.device)
                mask = torch.stack([mask] * x.shape[1], dim=1)
                mask = mask.ge(0.2).bool()
                mask = mask & nan_mask
            elif self.training and self.mask_mode == 'all_true':
                # 使用全部为True的掩码
                mask = nan_mask
            elif not self.training:
                # 测试阶段使用非NaN值的掩码
                mask = nan_mask
            else:
                # 默认情况
                raise Exception(f"掩码模式 {self.mask_mode} 不受支持")
                
        # 替换NaN值为0
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        # 应用线性层和特征提取器
        x = self.input_fc(x)  # 线性变换: [B, L, C] -> [B, L, H]
        
        # 在长度维度进行掩码处理
        mask_x = x * mask.unsqueeze(-1)  # 应用掩码: [B, L, H]
        
        # 提取特征
        x = self.feature_extractor(mask_x)  # 特征提取: [B, L, H]
        
        if self.training:
            x = self.repr_dropout(x)  # 应用Dropout正则化
            
        return x


class DilatedConvEncoder(nn.Module):
    """
    膨胀卷积编码器 (Dilated Convolutional Encoder)
    
    使用一系列具有不同膨胀率的卷积层来捕获不同时间尺度上的模式。
    膨胀卷积可以在不增加参数数量的情况下扩大感受野。
    
    属性:
        input_dims (int): 输入数据的维度
        output_dims_list (list): 每层输出通道数的列表
        kernel_size (int): 卷积核大小
    """
    def __init__(self, input_dims, output_dims_list, kernel_size=3):
        """
        初始化膨胀卷积编码器
        
        参数:
            input_dims (int): 输入数据的维度
            output_dims_list (list): 每层输出通道数的列表
            kernel_size (int): 卷积核大小，默认为3
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dims_list = output_dims_list
        self.kernel_size = kernel_size
        
        layers = []
        dims = input_dims
        for i, output_dims in enumerate(output_dims_list):
            # 计算膨胀率，随着层数增加而增加
            dilation = 2 ** i
            layers.append(
                ConvBlock(
                    dims, output_dims, kernel_size,
                    dilation=dilation
                )
            )
            dims = output_dims
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量，形状为[B, L, H]
            
        返回:
            torch.Tensor: 卷积编码后的特征表示
        """
        # 调整输入形状以适应卷积操作（B批次大小，H通道数，L序列长度）
        x = x.transpose(1, 2)
        
        # 依次通过所有卷积层
        for layer in self.layers:
            x = layer(x)
        
        # 恢复原始形状
        return x.transpose(1, 2)


class ConvBlock(nn.Module):
    """
    卷积块 (Convolutional Block)
    
    包含一个带膨胀的一维卷积层，后跟归一化和激活函数。
    
    属性:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小
        dilation (int): 膨胀率
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        """
        初始化卷积块
        
        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            kernel_size (int): 卷积核大小
            dilation (int): 膨胀率，默认为1
        """
        super().__init__()
        # 计算填充大小，以保持输出长度与输入相同
        padding = (dilation * (kernel_size - 1)) // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        # 实例归一化，对每个样本的每个通道进行单独归一化
        self.norm = nn.InstanceNorm1d(out_channels)
        # ELU激活函数
        self.activation = nn.ELU()
        
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量，形状为[B, H, L]
            
        返回:
            torch.Tensor: 卷积处理后的特征表示
        """
        # 应用卷积、归一化和激活
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class TraceContrast:
    """
    TraceContrast模型类
    
    这是一个对比学习框架，用于从时间序列数据中学习表示。
    它使用编码器将时间序列转换为特征向量，并通过对比学习方法进行训练。
    
    属性:
        device (torch.device): 运行模型的设备（CPU或GPU）
        config (dict): 模型配置参数
        encoder (nn.Module): 时间序列编码器
        optimizer (torch.optim.Optimizer): 模型优化器
    """
    
    def __init__(
        self,
        input_dims,
        output_dims=128,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None
    ):
        '''
        初始化TraceContrast模型
        
        参数:
            input_dims (int): 输入维度。对于单变量时间序列，应设置为1
            output_dims (int): 表征维度，即编码器输出的特征维度
            hidden_dims (int): 编码器的隐藏层维度
            depth (int): 编码器中隐藏残差块的数量
            device (str): 用于训练和推理的设备('cuda'或'cpu')
            lr (float): 学习率
            batch_size (int): 批次大小
            max_train_length (int或None): 训练允许的最大序列长度。对于长度大于max_train_length的序列，
                                        会被裁剪成多个长度小于max_train_length的子序列
            temporal_unit (int): 执行时间对比的最小单位，训练很长序列时，此参数有助于减少时间和内存消耗
            after_iter_callback (函数或None): 每次迭代后调用的回调函数
            after_epoch_callback (函数或None): 每个训练轮次后调用的回调函数
        '''
        
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        
        # 初始化时间序列编码器(TSEncoder)作为主要网络
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        # 使用随机权重平均(SWA)技术来提高模型泛化能力
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0
    
    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=True):
        '''
        训练模型
        
        参数:
            train_data (numpy.ndarray): 训练数据，形状为(样本数, 时间点数, 特征数)
                                       所有缺失数据应设置为NaN
            n_epochs (int或None): 训练轮数，达到此值时训练停止
            n_iters (int或None): 迭代次数，达到此值时训练停止
                                如果n_epochs和n_iters都未指定，则使用默认设置：
                                数据集大小<=100000时设置n_iters为200，否则为600
            verbose (bool): 是否在每个轮次后打印训练损失
            
        返回:
            loss_log: 包含每个轮次训练损失的列表
        '''
        assert train_data.ndim == 3  # 确保输入数据是三维的
        
        # 如果序列太长，将其分割成多个子序列
        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        # 检查时间维度是否存在缺失值，若首尾有缺失则居中对齐序列
        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)
                
        # 移除全部为NaN的样本
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        
        # 创建数据集和数据加载器
        train_dataset = MyDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        
        # 使用AdamW优化器
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
        loss_log = []
        
        # 主训练循环
        while True:
            print("Epoch:", self.n_epochs)
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            batch_num = 0
            
            interrupted = False
            # 遍历每个批次
            for batch_idx, new_batch in enumerate(train_loader):
                batch = new_batch[0]
                sample_idx = new_batch[1]
                batch_num += 1
                if n_epoch_iters % 500 == 0:
                    print("  Iter:", n_epoch_iters)
                
                x = batch
                # 如果序列太长，随机裁剪一段作为训练样本
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)
                
                # 随机裁剪不同的数据片段，用于对比学习
                ts_l = x.size(1)
                # 随机选择裁剪长度
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                # 随机选择裁剪起始位置
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                # 扩展裁剪范围，用于生成第二个片段
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
                optimizer.zero_grad()
                
                # 提取第一个片段并通过编码器
                out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]
                
                # 提取第二个片段并通过编码器
                out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]

                # 给第二个输出添加噪声，增强模型鲁棒性
                random_noise = torch.normal(0, 0.01, out2.shape)
                out2 += random_noise

                # 使用小波变换对第二个输出进行数据增强
                dwt = DWT1DForward(wave='db6', J=3)  # 使用Daubechies 6小波，3级分解
                idwt = DWT1DInverse(wave='db6')  # 对应的小波逆变换
                yl, yh = dwt(out2)  # 分解为低频和高频成分
                pad_yl = torch.zeros_like(yl)  # 将低频成分置零
                yi = idwt((pad_yl, yh))  # 仅保留高频成分进行重构
                out2 = yi  # 更新第二个输出

                # 计算层次对比损失
                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=self.temporal_unit
                )
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                    
                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
                
                # 如果设置了迭代回调函数，则调用
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

            
            # 计算平均损失
            cum_loss /= n_epoch_iters

            # 如果损失变化很小，提前停止训练
            if len(loss_log) > 1 and np.abs(cum_loss - loss_log[-1]) < 0.01:
                break

            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
            
            # 如果设置了轮次回调函数，则调用
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)
            
        return loss_log
    
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        """
        使用池化操作对编码结果进行评估
        
        参数:
            x: 输入数据
            mask: 掩码
            slicing: 切片范围
            encoding_window: 编码窗口类型
            
        返回:
            池化后的编码结果
        """
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            # 使用最大池化对整个序列进行汇总
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
            
        elif isinstance(encoding_window, int):
            # 使用指定大小的窗口进行滑动池化
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            
            if slicing is not None:
                out = out[:, slicing]
            
        elif encoding_window == 'multiscale':
            # 多尺度池化
            p = 0
            reprs = []
            while True:
                t_pool = 2**p
                if t_pool > out.size(1):
                    break
                # 多尺度最大池化
                reprs.append(F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = t_pool,
                    stride = t_pool,
                    padding = 0
                ).transpose(1, 2))
                p += 1
            out = torch.cat(reprs, dim=1)
            
            if slicing is not None:
                out = out[:, slicing]
            
        return out
    
    def encode(self, data, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
        """
        对输入数据进行编码，生成表征向量
        
        参数:
            data (numpy.ndarray): 输入数据，形状为(样本数, 时间点数, 特征数)
            mask (numpy.ndarray, 可选): 掩码矩阵，用于处理有效数据点
            encoding_window (str, 可选): 编码窗口类型，可选值有:
                - 'full_series': 对整个序列进行编码并池化
                - 'sliding': 使用滑动窗口进行编码
            casual (bool): 是否使用因果模式（只考虑当前及过去的数据点）
            sliding_length (int, 可选): 滑动窗口的长度，仅在encoding_window='sliding'时有效
            sliding_padding (int): 滑动窗口的填充长度
            batch_size (int, 可选): 批处理大小，用于大数据集处理
            
        返回:
            numpy.ndarray: 编码后的表征向量
        """
        # 确保输入数据是三维的
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        # 获取缺失数据的掩码
        org_training = self._net.training
        self._net.eval()
        self.net.eval()
        
        # 调整数据格式以适配模型输入
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            if encoding_window == 'full_series':
                # 对整个序列进行编码
                out = []
                for batch in loader:
                    x = batch[0]
                    if self.max_train_length is not None and x.size(1) > self.max_train_length:
                        window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                        x = x[:, window_offset : window_offset + self.max_train_length]
                    # 通过自定义函数进行评估，使用全序列池化
                    out.append(self._eval_with_pooling(x, mask, encoding_window=encoding_window))
                # 拼接所有批次的输出结果
                out = torch.cat(out, dim=0)
                
            elif encoding_window == 'sliding':
                # 使用滑动窗口进行编码
                
                # 确保定义了滑动窗口长度
                if sliding_length is None:
                    sliding_length = self.max_train_length
                
                # 将数据划分为滑动的子窗口
                sliding_padding = min(sliding_padding, sliding_length - 1)
                sliding_window_l = sliding_length + 2 * sliding_padding
                # 使用NaN填充，以允许不同长度的序列
                x = torch_pad_nan(
                    torch.from_numpy(data).to(torch.float),
                    left=sliding_padding,
                    right=sliding_padding + (sliding_window_l - data.shape[1] % sliding_window_l) % sliding_window_l,
                    dim=1
                )
                
                # 计算滑动窗口个数
                n_sliding_windows = math.floor((x.shape[1] - sliding_padding * 2 - sliding_length) / sliding_length + 2)
                
                # 将数据重塑为滑动窗口形式
                x = x.unfold(dimension=1, size=sliding_window_l, step=sliding_length).transpose(1, 2)
                
                # 初始化输出数组
                out = []
                for i in range(0, n_sliding_windows, batch_size // x.shape[0] + 1):
                    batch_sliding = x[:, i : i + batch_size // x.shape[0] + 1]
                    batch_sliding = batch_sliding.reshape(-1, sliding_window_l, batch_sliding.shape[-1])
                    
                    # 去除全为NaN的窗口
                    valid_idx = ~torch.isnan(batch_sliding).all(axis=(1, 2))
                    if valid_idx.any():
                        batch_sliding = batch_sliding[valid_idx]
                        # 对滑动窗口进行评估
                        batch_out = self._eval_with_pooling(
                            batch_sliding,
                            mask=None,
                            slicing=slice(sliding_padding, sliding_padding + sliding_length) if not casual 
                            else slice(sliding_padding, sliding_padding + sliding_length),
                            encoding_window='full_series'
                        )
                        out.append(batch_out)
                
                out = torch.cat(out, dim=0)
                
                out_length = (data.shape[1] - sliding_length) // sliding_length + 1
                if out_length < out.shape[0]:
                    out = out[:out_length]
            else:
                # 如果未指定编码窗口类型，报错
                raise ValueError(f"Unsupported encoding window: {encoding_window}")
            
        self._net.train(org_training)
        self.net.train(org_training)
        # 将结果转换为numpy数组返回
        return out.cpu().numpy()
    
    def save(self, fn):
        """
        保存模型参数到文件
        
        参数:
            fn (str): 保存模型参数的文件路径
        """
        ckpt = {
            'net_state_dict': self._net.state_dict(),
            'net_avg_state_dict': self.net.module.state_dict() 
            if isinstance(self.net, torch.nn.DataParallel) else self.net.state_dict(),
            'n_epochs': self.n_epochs,
            'n_iters': self.n_iters,
        }
        torch.save(ckpt, fn)
    
    def load(self, fn):
        """
        从文件加载模型参数
        
        参数:
            fn (str): 模型参数文件的路径
            
        返回:
            self: 加载参数后的模型实例
        """
        ckpt = torch.load(fn, map_location=self.device)
        self._net.load_state_dict(ckpt['net_state_dict'])
        
        if isinstance(self.net, torch.nn.DataParallel):
            self.net.module.load_state_dict(ckpt['net_avg_state_dict'])
        else:
            self.net.load_state_dict(ckpt['net_avg_state_dict'])
        
        self.n_epochs = ckpt['n_epochs']
        self.n_iters = ckpt['n_iters']
        return self
    
