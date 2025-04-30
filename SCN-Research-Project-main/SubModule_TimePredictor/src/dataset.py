from torch.utils.data import Dataset
import numpy as np


class TrainDataset(Dataset):
    """
    训练数据集类
    
    用于加载和处理训练神经网络的数据，支持随机采样神经元
    专为子模块时间预测器设计，从同类标签中采样神经元
    
    参数
    ----------
    x : numpy.ndarray
        输入特征数据
    y : numpy.ndarray
        标签数据
    num_neuron : int
        要采样的神经元数量
    """
    def __init__(self, x, y, num_neuron):
        super(TrainDataset, self).__init__()
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        self.x = x[idx]
        self.y = y[idx]
        self.num_neuron = num_neuron
        # 按标签将数据索引分组，用于同类别内采样
        self.location = [
                np.argwhere(self.y == label)[..., 0] for label in range(self.y.max()+1)
        ]
        
    def __getitem__(self, index):
        """
        获取数据项
        
        从同一时间标签的数据中随机选择指定数量的神经元
        
        参数
        ----------
        index : int
            数据索引
            
        返回值
        ----------
        tuple
            (选择的神经元数据, 标签)
        """
        label = self.y[index]
        if self.num_neuron >= self.x.shape[0]:
            self.num_neuron = self.x.shape[0] -1 
        # 从同一标签的样本中随机选择神经元
        random_idx = np.random.choice(self.location[label], self.num_neuron, replace=False)
        return self.x[random_idx], label
    
    def __len__(self):
        """
        返回数据集长度
        
        返回值
        ----------
        int
            数据集中样本数量
        """
        return self.x.shape[0]

class TestDataset(Dataset):
    """
    测试数据集类
    
    用于加载和处理测试神经网络的数据，支持随机采样神经元
    用于子模块时间预测器的测试，可以在不同空间类别间进行交叉测试
    
    参数
    ----------
    x : numpy.ndarray
        输入特征数据
    y : numpy.ndarray
        标签数据
    num_neuron : int
        要采样的神经元数量
    """
    def __init__(self, x, y, num_neuron):
        super(TestDataset, self).__init__()
        self.x = x
        self.y = y
        self.num_neuron = num_neuron

    def __getitem__(self, index):
        """
        获取数据项
        
        从所有测试数据中随机选择指定数量的神经元
        
        参数
        ----------
        index : int
            数据索引
            
        返回值
        ----------
        tuple
            (选择的神经元数据, 标签)
        """
        label = self.y[index]
        if self.num_neuron >= self.x.shape[0]:
            self.num_neuron = self.x.shape[0] -1 
        # 随机选择神经元
        random_idx = np.random.choice(self.x.shape[0], self.num_neuron, replace=False)
        return self.x[random_idx], label

    def __len__(self):
        """
        返回数据集长度
        
        返回值
        ----------
        int
            数据集中样本数量
        """
        return self.x.shape[0]
