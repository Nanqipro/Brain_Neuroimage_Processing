"""
神经元活动归因分析的基础训练模块

该模块实现了用于分析神经元活动与时间状态关系的CNN模型训练流程。
通过训练一个CNN模型来预测基于神经元活动的时间状态，为后续的神经元贡献度分析做准备。

作者: SCN研究小组
日期: 2023
"""

import numpy as np
import pickle, sys, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.lr_updater import CosineLrUpdater  # 余弦学习率调整器
from src.model import CNN                    # CNN模型
from src.dataset import NeuronData          # 神经元数据集
from scipy.io import loadmat                # 加载MATLAB格式数据

def accuracy(output, target, topk=(1)):
    """
    计算预测结果在topk指标下的准确率
    
    参数
    ----------
    output : torch.Tensor
        模型的输出预测
    target : torch.Tensor
        目标真实标签
    topk : tuple
        要计算的top-k准确率，默认为(1)
        
    返回
    -------
    list
        包含各个k值对应准确率的列表
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # 获取topk的预测结果索引
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # 计算前k个预测中命中目标的数量
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # 转换为百分比
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# 从命令行参数获取数据文件名
filename = sys.argv[1]
# 加载MAT格式的dff_set数据
data = loadmat(filename, variable_names=['dff_set'])['dff_set']

# 从命令行参数获取随机种子并设置
seed = int(sys.argv[2])
np.random.seed(seed)
torch.manual_seed(seed)

batch_size = 1  # 批次大小

# 初始化训练和测试性能记录列表
train_acc_list = list()   # 训练准确率列表
train_loss_list = list()  # 训练损失列表
test_loss_list = list()   # 测试损失列表
test_acc_list = list()    # 测试准确率列表

criterion = nn.CrossEntropyLoss()  # 损失函数：交叉熵
cuda = torch.cuda.is_available()   # 检查CUDA是否可用

# 从命令行参数获取神经元数量
num_neuron = int(sys.argv[3])

# 创建训练和测试数据加载器
train_dataloader = DataLoader(NeuronData(filename, 24), batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(NeuronData(filename, 24), batch_size=batch_size, shuffle=True)

# 初始化CNN模型
cnn = CNN(num_class=24, num_seq=num_neuron, base_channel=16)
        
# 如果CUDA可用，将模型移至GPU
if cuda:
    cnn.cuda()

# 设置优化器
optimizer = optim.Adam(cnn.parameters(), lr=0.0001)

# 设置余弦学习率调度器
scheduler = CosineLrUpdater(optimizer, 
                          periods=[500],         # 学习率变化周期
                          by_epoch=True,         # 按轮次更新学习率
                          warmup='linear',       # 线性预热
                          warmup_iters=1,        # 预热迭代次数
                          warmup_ratio=1.0 / 3,  # 预热比例
                          warmup_by_epoch=False, # 不按轮次进行预热
                          restart_weights=[1],   # 重启权重
                          min_lr=[1e-7],         # 最小学习率
                          min_lr_ratio=None)     # 最小学习率比例

# 创建保存目录
os.makedirs('./training_log', exist_ok=True)
# 生成保存文件名
save_filename = os.path.join('training_log', f"{os.path.basename(filename).split('_SCNProject')[0]}_log_num_neuron{num_neuron}_seed{seed}.pkl")

# 尝试加载之前保存的训练状态（支持继续训练）
start_iter = 0
try:
    with open(save_filename, 'rb') as f:
        res = pickle.load(f)
    # 加载之前的训练记录
    train_acc_list = res['train_acc_list']
    train_loss_list = res['train_loss_list']
    test_loss_list = res['test_loss_list']
    test_acc_list = res['test_acc_list']
    st = len(train_loss_list[-1])
    # 加载模型和优化器状态
    cnn.load_state_dict(res['model'])
    optimizer.load_state_dict(res['optimizer'])
    start_iter = res['step']
except FileNotFoundError:
    # 如果没有找到之前的训练记录，初始化新的记录列表
    train_acc_list.append(list())
    train_loss_list.append(list())
    test_loss_list.append(list())
    test_acc_list.append(list())
    st = 0

# 训练参数设置
epoch = 500            # 总训练轮数
current_iters = start_iter  # 当前迭代次数

# 学习率调度器初始化
scheduler.before_run()

# 开始训练循环
for current_epoch in range(st, epoch):  
    scheduler.before_train_epoch(24, current_epoch, current_iters)        
    cnn.train()  # 设置模型为训练模式
    train_loss = 0.0
    train_acc = 0.0
    
    # 遍历训练数据集
    for i, (x, y) in enumerate(train_dataloader):
        scheduler.before_train_iter(current_epoch, current_iters)

        # 将数据移至GPU（如果可用）
        if cuda:
            x = x.cuda()
            y = y.cuda()
            
        optimizer.zero_grad()  # 清除梯度
        outputs = cnn(x)       # 前向传播
        # 计算损失
        loss = criterion(outputs, y.squeeze(-1).to(torch.int64))
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()

        current_iters += 1
        train_loss += loss.item()
        # 计算训练准确率
        train_acc += accuracy(outputs, y.squeeze(-1), topk=(1, 2))[0].cpu().numpy()
        
    # 计算平均训练损失和准确率
    train_loss /= 24
    train_acc /= 24

    # 在测试集上评估模型
    test_loss = 0.
    test_acc = 0.
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for x, y in test_dataloader:
            if cuda:
                x = x.cuda()
                y = y.cuda()
            outputs = cnn(x)
            loss = criterion(outputs, y.squeeze(-1).to(torch.int64))
            
            current_iters += 1
            test_loss += loss.item()
            test_acc += accuracy(outputs, y.squeeze(-1), topk=(1, 2))[0].cpu().numpy()
            
    # 计算平均测试损失和准确率
    test_loss /= 24
    test_acc /= 24
    
    # 每20轮输出一次训练状态
    if current_epoch % 20 == 0:
        print(f'epoch {current_epoch} train_loss: {np.round(train_loss, 5)}; train_acc: {np.round(train_acc, 5)}; test_loss: {np.round(test_loss, 5)}; test_acc: {np.round(test_acc, 3)}')
    
    # 记录训练和测试结果
    train_acc_list[-1].append(train_acc)
    train_loss_list[-1].append(train_loss)
    test_loss_list[-1].append(test_loss)
    test_acc_list[-1].append(test_acc)

    # 保存训练状态和模型
    pickle.dump({
        'train_loss_list': train_loss_list,
        'train_acc_list': train_acc_list,
        'test_loss_list': test_loss_list,
        'test_acc_list': test_acc_list,
        'model': cnn.state_dict(),
        'optimizer': optimizer.state_dict(),
        "step": current_iters,
    }, open(save_filename, 'wb'))
