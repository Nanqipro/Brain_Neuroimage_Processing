import torch
import torch.nn as nn


class resBlock(nn.Module):
    """
    残差块模型
    
    用于构建卷积神经网络中的残差连接，提高模型性能和训练稳定性
    
    参数
    ----------
    inplanes : int
        输入通道数
    planes : int
        中间层通道数
    kernel_size : int, 可选
        卷积核大小，默认为3
    """
    def __init__(self, inplanes, planes, kernel_size=3):
        super(resBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, inplanes, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
        self.bn2 = nn.BatchNorm1d(planes)

    def forward(self, x):
        """
        前向传播函数
        
        参数
        ----------
        x : torch.Tensor
            输入张量
            
        返回值
        ----------
        torch.Tensor
            经过残差块处理后的输出张量
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out
    
class CNN(nn.Module):
    """
    用于时间预测的卷积神经网络模型
    
    该模型用于处理神经元活动数据，预测时间类别
    
    参数
    ----------
    time_len : int
        时间序列长度
    num_seq : int
        神经元序列数量
    num_class : int
        输出类别数量
    """
    def __init__(self, time_len, num_seq, num_class):
        super(CNN, self).__init__()
        self.time_len = time_len
        self.num_seq = num_seq
        self.num_class = num_class
        
        # 卷积层序列
        self.conv = nn.Sequential(*[
            nn.Conv1d(self.num_seq, 32, kernel_size=7, bias=True),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(5),
            resBlock(32, 32, kernel_size=3),
            nn.MaxPool1d(5),
            nn.Flatten(start_dim=1, end_dim=-1),
        ])
        
        # 动态计算全连接层输入维度
        with torch.no_grad():
            x = torch.zeros(1, self.num_seq, self.time_len)
            x = self.conv(x)
            self.fc_dim = x.reshape(1, -1).shape[-1]
        
        # 全连接层
        self.fc = nn.Sequential(*[
            nn.Linear(self.fc_dim, self.num_class)
        ])
    
    def forward(self, x):
        """
        前向传播函数
        
        参数
        ----------
        x : torch.Tensor
            输入张量，形状为 [batch_size, num_seq, time_len]
            
        返回值
        ----------
        torch.Tensor
            模型输出的预测结果，形状为 [batch_size, num_class]
        """
        return self.fc(self.conv(x))