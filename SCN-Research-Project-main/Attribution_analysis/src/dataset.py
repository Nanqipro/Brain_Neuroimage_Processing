"""
神经元活动数据集模块

该模块定义了加载和处理神经元活动数据的数据集类。
将MATLAB格式的神经元活动数据转换为PyTorch可用的Dataset格式，
便于在神经网络训练过程中进行批处理。

作者: SCN研究小组
日期: 2023
"""

from torch.utils.data import Dataset
import torch
import os
import numpy as np
import scipy.io as scio 

class NeuronData(Dataset):
    """
    神经元活动数据集类
    
    加载MAT格式的神经元活动数据，并将其转换为适合神经网络训练的格式。
    数据组织形式为时间点分类任务，每个样本代表一个时间点的神经元活动。
    
    参数
    ----------
    path : str
        MAT格式数据文件的路径
    average : int, 可选
        是否对数据进行平均处理，默认为1
    class_num : int, 可选
        时间类别数量，默认为24
    """
    def __init__(self, path, average=1, class_num=24):
        super(NeuronData, self).__init__()
        
        # 加载MAT文件中的dff_set数据
        data = scio.loadmat(path, variable_names=['dff_set'])['dff_set']
        _data = list()
        label = list()
        time_class = data.shape[1]  # 时间类别数量
        num_neuron = data.shape[0]  # 神经元数量
        
        # 重组数据结构
        for i in range(data.shape[0]):  # 遍历每个神经元
            for j in range(data.shape[1]):  # 遍历每个时间点
                _data.append(data[i, j])
                label.append(j)
                
        # 数据重塑为 [神经元数量, 时间类别数, 序列长度]
        self.data = np.concatenate(_data, axis=0).reshape(num_neuron, time_class, -1).astype(np.float32)
        # 对序列长度维度求平均，得到每个神经元在每个时间点的平均活动
        self.data = np.mean(self.data, axis=-1)
        # 标签只需要第一个神经元的所有时间点，因为所有神经元共享相同的时间标签
        self.label = np.array(label).reshape(num_neuron, time_class).astype(np.int64)[0, :]  # 24个时间点
        
    
    def __len__(self):
        """
        返回数据集的样本数量
        
        返回
        -------
        int
            数据集中的样本数量，等于时间类别数量
        """
        return 24  # 返回时间类别数量

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        
        参数
        ----------
        idx : int
            样本索引，代表时间点的索引
            
        返回
        -------
        tuple
            (trace_data, label)，其中：
            - trace_data: 所有神经元在指定时间点的活动数据，torch.Tensor类型
            - label: 对应的时间点标签，形状为[1]的torch.Tensor
        """
        # 获取所有神经元在idx时间点的活动数据
        data = self.data[:, idx]  # 形状为 [神经元数量]
        # 获取对应的时间点标签
        label = self.label[idx].reshape(-1)  # 转换为[1]形状
        
        # 转换为PyTorch张量
        trace_data = torch.from_numpy(data).float()
        return trace_data, label
