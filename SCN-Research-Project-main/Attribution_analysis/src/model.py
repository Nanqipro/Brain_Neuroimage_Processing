"""
神经元活动分析模型定义模块

该模块定义了用于神经元活动分析的神经网络模型，包括基于LSTM和CNN的两种架构。
这些模型用于从神经元活动数据中提取时间相关的特征模式，实现时间点分类预测任务。

作者: SCN研究小组
日期: 2023
"""

import torch
import torch.nn as nn

class LSTM(torch.nn.Module):
    """
    基于LSTM的神经元活动序列分析模型
    
    该模型利用LSTM网络处理神经元活动的序列数据，
    捕获时间序列中的依赖关系，用于时间点分类。
    
    参数
    ----------
    c_in : int, 可选
        输入特征维度，对应神经元数量，默认6049
    num_class : int, 可选
        输出类别数量，对应时间点数，默认24
    seq_len : int, 可选
        序列长度，默认200
    hidden_size : int, 可选
        LSTM隐藏层大小，默认32
    rnn_layers : int, 可选
        LSTM层数，默认1
    bias : bool, 可选
        是否使用偏置，默认True
    cell_dropout : float, 可选
        LSTM单元内部dropout比率，默认0
    rnn_dropout : float, 可选
        LSTM层之间的dropout比率，默认0.2
    bidirectional : bool, 可选
        是否使用双向LSTM，默认False
    shuffle : bool, 可选
        输入是否需要打乱，默认False
    fc_dropout : float, 可选
        全连接层dropout比率，默认0.1
    """
    def __init__(self, c_in=6049, num_class=24, seq_len=200, hidden_size=32, 
                 rnn_layers=1, bias=True, cell_dropout=0, rnn_dropout=0.2, 
                 bidirectional=False, shuffle=False, fc_dropout=0.1):
        super(LSTM, self).__init__()
                    
        # LSTM层定义
        self.rnn = nn.LSTM(c_in, hidden_size, num_layers=rnn_layers, bias=bias, batch_first=True, 
                          dropout=cell_dropout, bidirectional=bidirectional)
        # LSTM后的dropout层
        self.rnn_dropout = nn.Dropout(rnn_dropout) if rnn_dropout else nn.Identity()
                
        # 全连接分类层
        self.fc_dropout = nn.Dropout(fc_dropout) if fc_dropout else nn.Identity()
        # 输出层，考虑单向/双向LSTM的输出维度差异
        self.fc = nn.Linear(hidden_size * (1 + bidirectional), num_class)
        

    def forward(self, x):
        """
        前向传播函数
        
        参数
        ----------
        x : torch.Tensor
            输入神经元活动数据，形状为[batch_size, n_vars, seq_len]
            
        返回
        -------
        torch.Tensor
            时间点分类的预测结果，形状为[batch_size, num_class]
        """
        # 调整输入维度顺序以适应LSTM的batch_first=True
        rnn_input = x.permute(0, 2, 1)  # 变为[batch_size, seq_len, n_vars]
        # LSTM前向计算
        output, _ = self.rnn(rnn_input)
        # 取序列最后一步的输出（多对一模式）
        last_out = output[:, -1]
        # 应用dropout
        last_out = self.rnn_dropout(last_out)
        # 全连接层处理
        x = self.fc_dropout(last_out)
        x = self.fc(x)
        return x

class CNN(nn.Module):
    """
    基于CNN的神经元活动分析模型
    
    该模型使用一维卷积网络处理神经元活动数据，
    学习神经元之间的空间关系用于时间点分类。
    
    参数
    ----------
    time_len : int, 可选
        时间序列长度，默认1
    num_seq : int, 可选
        神经元数量，默认6049
    base_channel : int, 可选
        基础通道数，默认32
    num_class : int, 可选
        输出类别数量，对应时间点数，默认24
    """
    def __init__(self, time_len=1, num_seq=6049, base_channel=32, num_class=24):
        super(CNN, self).__init__()
        self.time_len = time_len
        self.num_seq = num_seq
        self.num_class = num_class
        
        # 卷积网络部分：使用线性层、层归一化和LeakyReLU激活函数
        self.conv = nn.Sequential(*[
            nn.Linear(self.num_seq, base_channel),          # 第一个线性层，降维到base_channel
            nn.LayerNorm(base_channel),                     # 层归一化
            nn.LeakyReLU(inplace=True),                     # LeakyReLU激活
            nn.Linear(base_channel, base_channel),          # 第二个线性层
            nn.LayerNorm(base_channel),                     # 层归一化
            nn.LeakyReLU(inplace=True),                     # LeakyReLU激活
            nn.Flatten(start_dim=1, end_dim=-1),            # 扁平化处理
        ])
        
        # 使用空输入预计算全连接层的输入维度
        with torch.no_grad():
            x = torch.zeros(1, self.num_seq)
            x = self.conv(x)
            self.fc_dim = x.reshape(1, -1).shape[-1]
            # print(self.fc_dim)
            
        # 全连接分类层
        self.fc = nn.Sequential(*[
            nn.Linear(self.fc_dim, self.num_class)  # 最终分类层
        ])
    
    def forward(self, x):
        """
        前向传播函数
        
        参数
        ----------
        x : torch.Tensor
            输入神经元活动数据，形状为[batch_size, num_seq]
            
        返回
        -------
        torch.Tensor
            时间点分类的预测结果，形状为[batch_size, num_class]
        """
        # 依次通过卷积网络和全连接层
        return self.fc(self.conv(x))

# 测试代码（已注释）
# if __name__ == '__main__':
#     input_data = torch.rand(8, 6049)
#     model = CNN()
#     output = model(input_data)
#     print(output.shape)
