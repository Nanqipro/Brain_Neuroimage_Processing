"""
子模块时间预测器训练脚本

该脚本用于在特定空间区域的神经元数据上训练时间预测模型，
并在其他空间区域上进行交叉测试，研究空间区域之间的时间编码迁移性。

使用方法: python training_base_1for3.py <空间类别> <随机种子> <神经元数量>
"""

import numpy as np
import pickle, sys, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.lr_updater import CosineLrUpdater
from src.model import CNN

from src.dataset import TrainDataset, TestDataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


# 加载数据集
filename = '../SCNData/Dataset1_SCNProject.mat'
data = loadmat(filename, variable_names=['dff_set'])['dff_set']
_data = list()
label = list()
time_class = data.shape[1]
num_neuron = data.shape[0]
# 重组数据结构
for i in range(data.shape[0]): # 神经元
    for j in range(data.shape[1]): # 时间
        _data.append(data[i, j])
        label.append(j)
data = np.concatenate(_data, axis=0).reshape(num_neuron, time_class, -1).astype(np.float32)
label = np.array(label).reshape(num_neuron, time_class).astype(np.int64)
time_len = data.shape[-1]

# 设置随机种子
seed = int(sys.argv[2])
np.random.seed(seed)
torch.manual_seed(seed)

# 训练参数设置
batch_size = 32
criterion = nn.CrossEntropyLoss()
cuda = torch.cuda.is_available()
num_neuron = int(sys.argv[3])  # 要使用的神经元数量
spatial_class = int(sys.argv[1])  # 当前训练使用的空间类别

# 加载3个空间类别的标签
index_for_spatial_label = loadmat(f'../SCNData/3-class.mat',variable_names=['index'])['index']
index_for_spatial_label = index_for_spatial_label.squeeze(-1)

# 初始化训练和测试性能记录列表
train_loss_list = list()
train_acc_list = list()
test_loss1_list = list()
test_acc1_list = list()
test_loss2_list = list()
test_acc2_list = list()
test_loss3_list = list()
test_acc3_list = list()

# 获取当前空间类别的神经元索引
spatial_class_index = list(np.where(index_for_spatial_label==spatial_class)[0])

# 获取当前空间类别的数据
X = data[spatial_class_index, ...]
y = label[spatial_class_index, ...]
# 分割训练集和测试集3（当前空间类别的测试集）
X_train_val, X_test3, y_train_val, y_test3 = train_test_split(X, y, test_size=0.3, random_state=74)
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, random_state=74)
train_size = y_train.shape[0]
test_size3 = y_test3.shape[0]

# 根据当前训练的空间类别，获取其他两个空间类别的测试数据
if spatial_class==1:
    # 如果当前训练空间类别为1，则测试集1为空间类别2的数据
    spatial_class_index = list(np.where(index_for_spatial_label==2)[0])
    X_test1 = data[spatial_class_index, ...]
    y_test1 = label[spatial_class_index, ...]
    test_size1 = len(spatial_class_index)
    # 测试集2为空间类别3的数据
    spatial_class_index = list(np.where(index_for_spatial_label==3)[0])
    X_test2 = data[spatial_class_index, ...]
    y_test2 = label[spatial_class_index, ...]
    test_size2 = len(spatial_class_index)
elif spatial_class==2:
    # 如果当前训练空间类别为2，则测试集1为空间类别1的数据
    spatial_class_index = list(np.where(index_for_spatial_label==1)[0])
    X_test1 = data[spatial_class_index, ...]
    y_test1 = label[spatial_class_index, ...]
    test_size1 = len(spatial_class_index)
    # 测试集2为空间类别3的数据
    spatial_class_index = list(np.where(index_for_spatial_label==3)[0])
    X_test2 = data[spatial_class_index, ...]
    y_test2 = label[spatial_class_index, ...]
    test_size2 = len(spatial_class_index)
elif spatial_class==3:
    # 如果当前训练空间类别为3，则测试集1为空间类别1的数据
    spatial_class_index = list(np.where(index_for_spatial_label==1)[0])
    X_test1 = data[spatial_class_index, ...]
    y_test1 = label[spatial_class_index, ...]
    test_size1 = len(spatial_class_index)
    # 测试集2为空间类别2的数据
    spatial_class_index = list(np.where(index_for_spatial_label==2)[0])
    X_test2 = data[spatial_class_index, ...]
    y_test2 = label[spatial_class_index, ...]
    test_size2 = len(spatial_class_index)

# 检查神经元数量是否超过训练集大小，如果超过则退出
if num_neuron >= train_size:
    exit()
else:
    training_num=num_neuron

# 创建训练数据加载器
train_dataloader = DataLoader(TrainDataset(X_train.reshape(-1, time_len), y_train.reshape(-1), training_num), batch_size=batch_size, num_workers=16, shuffle=True)

# 确定各测试集使用的神经元数量
test_num1 = test_size1-1 if num_neuron >= test_size1 else num_neuron
test_num2 = test_size2-1 if num_neuron >= test_size2 else num_neuron
test_num3 = test_size3-1 if num_neuron >= test_size3 else num_neuron
# 创建当前空间类别测试集的数据加载器
if num_neuron < test_size3:
    test_dataloader3 = DataLoader(TestDataset(X_test3, y_test3, num_neuron), batch_size=1, shuffle=True)

# 初始化CNN模型
cnn = CNN(time_len, num_neuron, label.max()+1)
if cuda:
    cnn.cuda()

# 设置优化器和学习率调度器
optimizer = optim.Adam(cnn.parameters(), lr=0.0001)
scheduler = CosineLrUpdater(optimizer, 
                                periods=[100],
                                by_epoch=True,
                                warmup='linear',
                                warmup_iters=200,
                                warmup_ratio=1.0 / 3,
                                warmup_by_epoch=False,
                                restart_weights=[1],
                                min_lr=[1e-7],
                                min_lr_ratio=None)

# 创建日志目录
os.makedirs('./training_log_1for3', exist_ok=True)
save_filename = os.path.join('training_log_1for3', f"train_spatial{spatial_class}_log_train_num{training_num}_test1_num{test_num1}_test2_num{test_num2}_test3_num{test_num3}_seed{seed}.pkl")
start_iter = 0

# 尝试加载之前的训练状态（断点续训）
try:
    with open(save_filename, 'rb') as f:
        res = pickle.load(f)
    train_loss_list = res['train_loss_list']
    train_acc_list = res['train_acc_list']
    st = len(train_loss_list[-1])
    cnn.load_state_dict(res['model'])
    optimizer.load_state_dict(res['optimizer'])
    start_iter=res['step']
except FileNotFoundError:
    # 初始化训练日志列表
    train_loss_list.append(list())
    train_acc_list.append(list())
    st = 0

# 训练参数
no_improve = 10  # 早停计数器
last_val_loss = 1e8
epochs = 100
current_iters = start_iter
scheduler.before_run()

# 开始训练循环
for current_epoch in range(st, epochs):  # 多次遍历数据集
    scheduler.before_train_epoch(train_size, current_epoch, current_iters)   
    cnn.train()
    train_loss = 0.0
    train_acc = 0.0
    
    # 训练数据批处理
    for i, (x, y) in enumerate(train_dataloader):
        scheduler.before_train_iter(current_epoch, current_iters)
        if cuda:
            x = x.cuda()
            y = y.cuda()
        optimizer.zero_grad()
        outputs = cnn(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.shape[0]
        pred = torch.argmax(outputs, dim=1).cpu().numpy()
        train_acc += (y.cpu().numpy() == pred).sum()
    
    # 计算平均训练损失和准确率
    train_loss /= train_size * 24
    train_acc /= train_size * 24
    
    # 设置为评估模式
    cnn.eval()
    
    # 在当前空间类别的测试集上评估
    if num_neuron < test_size3:
        test_loss3 = 0.
        test_acc3 = 0.
        with torch.no_grad():
            for x, y in test_dataloader3:
                if cuda:
                    x = x.cuda()
                    y = y.cuda()
                # 重新排列数据维度
                x = x.permute(0,2,1,3).reshape(-1, num_neuron, time_len)
                y = y.reshape(-1)
                outputs = cnn(x)
                pred = torch.argmax(outputs, dim=1).cpu().numpy()
                test_acc3 += (y.cpu().numpy() == pred).sum()
                test_loss3 += criterion(outputs, y).item() * x.shape[0]
        test_loss3 /= test_size3 * 24
        test_acc3 /= test_size3 * 24

    # 每10个epoch打印一次训练进度
    if current_epoch % 10 == 0:
        if num_neuron < test_size3:
            print(f'epoch {current_epoch} train_loss: {np.round(train_loss, 5)}; test_loss_self: {np.round(test_loss3, 5)}; test_acc_self: {np.round(test_acc3, 3)}')
        else:
            print(f'epoch {current_epoch} train_loss: {np.round(train_loss, 5)}; train_acc:{np.round(train_acc, 5)}')
    
    # 记录训练指标
    train_loss_list[-1].append(train_loss)
    train_acc_list[-1].append(train_acc)

    # 保存模型和训练状态
    pickle.dump({
        'train_loss_list': train_loss_list,
        'train_acc_list': train_acc_list,       
        'optimizer': optimizer.state_dict(),
        'model': cnn.state_dict()
    }, open(save_filename, 'wb'))
