"""
脑网络状态分类器主程序

该模块实现了基于图卷积网络(GCN)的脑网络状态分类系统，用于分析和分类神经元活动状态。
主要功能包括模型训练、验证和测试过程的实现。

作者: SCN研究小组
日期: 2023
"""

import torch
from utils import *
from model import *
import random
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# 配置参数
num_epochs = 160  # 训练轮数
learning_rate = 0.001  # 学习率
# num_edges = 0

def train(model, optimizer, train_dataloader, valid_dataloader, test_dataloader, device, epoch):
    """
    训练模型的函数
    
    参数
    ----------
    model : torch.nn.Module
        待训练的模型
    optimizer : torch.optim.Optimizer
        优化器
    train_dataloader : DataLoader
        训练数据加载器
    valid_dataloader : DataLoader
        验证数据加载器
    test_dataloader : DataLoader
        测试数据加载器
    device : torch.device
        计算设备(CPU/GPU)
    epoch : int
        当前训练轮次
        
    返回
    -------
    None
    """
    model.train()
    loss = 0
    optimizer.zero_grad()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')  # 使用交叉熵损失函数
    loss_epoch = []
    val_min = 1000
    best_model = model

    for i, batch in enumerate(train_dataloader):
        # 提取批次数据
        x = batch[0].squeeze().to(device)  # 节点特征
        y = batch[1].to(device)  # 标签
        edge = batch[2].squeeze().to(device)  # 边连接信息
        
        # 数据增强：随机使用部分图结构
        random_num = np.random.uniform()
        if random_num > 0.5 and random_num < 0.75:
            # 只使用前87条边和对应节点
            edge = edge[:,:87]
            x = x[:88,:]
        elif random_num > 0.75:
            # 只使用后面的边和节点
            edge = edge[:,87:]-87
            x = x[87:,:]

        optimizer.zero_grad()
        pred = model(edge, x)  # 前向传播

        loss = criterion(pred, y)  # 计算损失
        loss_epoch.append(loss.item())
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
    
    # 计算并打印平均损失
    mean_loss = np.mean(loss_epoch)
    print(f'epoch {epoch + 1} meanloss: {mean_loss}')


def test(model, device, test_loader):
    """
    测试模型性能的函数
    
    参数
    ----------
    model : torch.nn.Module
        待测试的模型
    device : torch.device
        计算设备(CPU/GPU)
    test_loader : DataLoader
        测试数据加载器
        
    返回
    -------
    tuple
        (整体准确率, 平均类别准确率)
    """
    model.eval()  # 设置为评估模式
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    loss_values = []
    pred_scores = []
    true_scores = []
    
    for i, batch in enumerate(test_loader):
        # 提取批次数据
        x = batch[0].squeeze().to(device)
        label = batch[1].to(device)
        edge = batch[2].squeeze().to(device)
        
        # 无梯度计算模型输出
        with torch.no_grad():
            y = model(edge, x)
            
        loss = criterion(y, label)
        loss_values.append(loss.item())
        
        # 获取预测结果并保存
        pred = y.max(dim=1)[1]
        pred_scores.append(pred.data.cpu().numpy())
        true_scores.append(label.data.cpu().numpy())

    # 拼接所有批次的结果
    pred_scores = np.concatenate(pred_scores)
    true_scores = np.concatenate(true_scores)
    
    # 计算评估指标
    mean_loss = np.mean(loss_values)
    overall_acc = accuracy_score(true_scores, pred_scores)  # 整体准确率
    avg_class_acc = balanced_accuracy_score(true_scores, pred_scores)  # 平均类别准确率
    
    return overall_acc, avg_class_acc


def main():
    """
    主函数，包含模型训练和评估的完整流程
    """
    # 设置计算设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    
    # 加载数据集
    data_path = './data'
    train_dataloader, valid_dataloader, test_dataloader = get_dataset(data_path)
    
    # 创建模型实例
    model = MultiLayerGCN(num_classes=6).to(DEVICE)
    print('-----model:-----')
    print(model)
    print(len(train_dataloader), len(valid_dataloader), len(test_dataloader))
    
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=learning_rate)
    
    # 记录最佳性能
    best_acc = 0.0
    best_epoch = 0
    best_avg_class_acc = 0.0
    
    # 开始训练循环
    for epoch in range(num_epochs):
        # 学习率衰减策略
        if epoch > 0 and epoch % 20 == 0:
            optimizer.param_groups[0]['lr'] *= 0.75
            
        # 训练模型
        train(model, optimizer, train_dataloader, valid_dataloader, test_dataloader, DEVICE, epoch)
        
        # 在验证集上评估
        acc, avg_class_acc = test(model, DEVICE, valid_dataloader)
        
        # 保存最佳模型
        if best_acc < acc: 
            best_acc = acc 
            best_epoch = epoch
            torch.save(model.state_dict(), f'./result/best_scn.pt')
            
        if best_avg_class_acc < avg_class_acc:
            best_avg_class_acc = avg_class_acc
            
        print("acc is: {:.4f}, best acc is {:.4f} in epoch {}, bset avg_class_acc is {:.4f}\n".format(
            acc, best_acc, best_epoch + 1, best_avg_class_acc))

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(f'./result/best_scn.pt'))
    model.eval()
    acc, avg_class_acc = test(model, DEVICE, test_dataloader)
    print("Test:acc is: {:.4f}, avg_class_acc is {:.4f}\n".format(acc, avg_class_acc))


if __name__ == '__main__':
    # 设置随机种子，确保结果可复现
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    main()
