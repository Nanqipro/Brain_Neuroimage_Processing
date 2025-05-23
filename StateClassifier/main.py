"""
脑网络状态分类器主程序

该模块实现了基于图卷积网络(GCN)的脑网络状态分类系统，用于分析和分类神经元活动状态。
主要功能包括模型训练、验证和测试过程的实现。

作者: Clade 4
日期: 2025年5月23日
"""

import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import logging
import os

# 导入项目模块
from utils import get_dataset
from model import MultiLayerGCN
from config import config

# 配置日志
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def set_random_seeds():
    """
    设置随机种子以确保结果可复现
    """
    torch.manual_seed(config.TORCH_SEED)
    torch.cuda.manual_seed_all(config.TORCH_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)


def train_epoch(model, optimizer, train_dataloader, device, epoch):
    """
    训练单个epoch的函数
    
    Parameters
    ----------
    model : torch.nn.Module
        待训练的模型
    optimizer : torch.optim.Optimizer
        优化器
    train_dataloader : DataLoader
        训练数据加载器
    device : torch.device
        计算设备(CPU/GPU)
    epoch : int
        当前训练轮次
        
    Returns
    -------
    float
        该epoch的平均损失
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    loss_epoch = []
    
    for i, batch in enumerate(train_dataloader):
        # 提取批次数据
        x = batch[0].squeeze().to(device)  # 节点特征
        y = batch[1].to(device)  # 标签
        edge = batch[2].squeeze().to(device)  # 边连接信息
        
        # 数据增强：随机使用部分图结构
        random_num = np.random.uniform()
        if 0.5 < random_num < 0.75:
            # 只使用前87条边和对应节点
            edge = edge[:, :87]
            x = x[:88, :]
        elif random_num > 0.75:
            # 只使用后面的边和节点
            edge = edge[:, 87:] - 87
            x = x[87:, :]

        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        pred = model(edge, x)
        
        # 计算损失
        loss = criterion(pred, y)
        loss_epoch.append(loss.item())
        
        # 反向传播和参数更新
        loss.backward()
        optimizer.step()
    
    # 计算并返回平均损失
    mean_loss = np.mean(loss_epoch)
    logger.info(f'Epoch {epoch + 1} 平均损失: {mean_loss:.4f}')
    
    return mean_loss


def evaluate_model(model, device, data_loader):
    """
    评估模型性能的函数
    
    Parameters
    ----------
    model : torch.nn.Module
        待评估的模型
    device : torch.device
        计算设备(CPU/GPU)
    data_loader : DataLoader
        数据加载器
        
    Returns
    -------
    tuple
        (平均损失, 整体准确率, 平均类别准确率)
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    loss_values = []
    pred_scores = []
    true_scores = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # 提取批次数据
            x = batch[0].squeeze().to(device)
            label = batch[1].to(device)
            edge = batch[2].squeeze().to(device)
            
            # 模型预测
            y = model(edge, x)
            
            # 计算损失
            loss = criterion(y, label)
            loss_values.append(loss.item())
            
            # 获取预测结果
            pred = y.max(dim=1)[1]
            pred_scores.append(pred.data.cpu().numpy())
            true_scores.append(label.data.cpu().numpy())

    # 拼接所有批次的结果
    pred_scores = np.concatenate(pred_scores)
    true_scores = np.concatenate(true_scores)
    
    # 计算评估指标
    mean_loss = np.mean(loss_values)
    overall_acc = accuracy_score(true_scores, pred_scores)
    avg_class_acc = balanced_accuracy_score(true_scores, pred_scores)
    
    return mean_loss, overall_acc, avg_class_acc


def train_model(model, train_dataloader, valid_dataloader, device):
    """
    完整的模型训练流程
    
    Parameters
    ----------
    model : torch.nn.Module
        待训练的模型
    train_dataloader : DataLoader
        训练数据加载器
    valid_dataloader : DataLoader
        验证数据加载器
    device : torch.device
        计算设备
        
    Returns
    -------
    torch.nn.Module
        训练好的最佳模型
    """
    # 设置优化器
    optimizer = torch.optim.Adam(
        model.parameters(), 
        weight_decay=config.WEIGHT_DECAY, 
        lr=config.LEARNING_RATE
    )
    
    # 记录最佳性能
    best_acc = 0.0
    best_epoch = 0
    best_avg_class_acc = 0.0
    best_model_state = None
    
    logger.info(f"开始训练，总共 {config.NUM_EPOCHS} 个epoch")
    
    # 训练循环
    for epoch in range(config.NUM_EPOCHS):
        # 学习率衰减策略
        if epoch > 0 and epoch % config.LR_DECAY_STEP == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= config.LR_DECAY_FACTOR
            logger.info(f"学习率衰减至: {optimizer.param_groups[0]['lr']:.6f}")
            
        # 训练一个epoch
        train_loss = train_epoch(model, optimizer, train_dataloader, device, epoch)
        
        # 在验证集上评估
        val_loss, val_acc, val_avg_class_acc = evaluate_model(model, device, valid_dataloader)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            
        if val_avg_class_acc > best_avg_class_acc:
            best_avg_class_acc = val_avg_class_acc
            
        # 打印训练进度
        logger.info(
            f"Epoch {epoch + 1}/{config.NUM_EPOCHS} - "
            f"训练损失: {train_loss:.4f}, "
            f"验证准确率: {val_acc:.4f}, "
            f"最佳准确率: {best_acc:.4f} (Epoch {best_epoch + 1}), "
            f"最佳平均类别准确率: {best_avg_class_acc:.4f}"
        )

    # 保存最佳模型
    config.create_directories()
    torch.save(best_model_state, config.BEST_MODEL_PATH)
    logger.info(f"最佳模型已保存至: {config.BEST_MODEL_PATH}")
    
    # 加载最佳模型状态
    model.load_state_dict(best_model_state)
    
    return model


def main():
    """
    主函数，包含模型训练和评估的完整流程
    """
    logger.info("=" * 60)
    logger.info("脑网络状态分类器训练程序启动")
    logger.info("=" * 60)
    
    # 设置随机种子
    set_random_seeds()
    logger.info("✓ 随机种子设置完成")
    
    # 设置计算设备
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"✓ 使用GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("✓ 使用CPU")
    
    # 加载数据集
    logger.info("加载数据集...")
    try:
        train_dataloader, valid_dataloader, test_dataloader = get_dataset(
            data_path=str(config.DATA_DIR),
            train_propotion=config.TRAIN_RATIO,
            valid_propotion=config.VALID_RATIO,
            BATCH_SIZE=config.BATCH_SIZE
        )
        logger.info(f"✓ 数据集加载成功")
        logger.info(f"  - 训练集批次数: {len(train_dataloader)}")
        logger.info(f"  - 验证集批次数: {len(valid_dataloader)}")
        logger.info(f"  - 测试集批次数: {len(test_dataloader)}")
    except Exception as e:
        logger.error(f"✗ 数据集加载失败: {e}")
        return
    
    # 创建模型实例
    model = MultiLayerGCN(
        dropout=config.DROPOUT_RATE,
        num_classes=config.NUM_CLASSES
    ).to(device)
    
    logger.info("✓ 模型创建成功")
    logger.info(f"模型架构:\n{model}")
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    
    # 训练模型
    logger.info("\n开始模型训练...")
    trained_model = train_model(model, train_dataloader, valid_dataloader, device)
    
    # 在测试集上评估
    logger.info("\n在测试集上评估最终性能...")
    test_loss, test_acc, test_avg_class_acc = evaluate_model(trained_model, device, test_dataloader)
    
    logger.info("=" * 60)
    logger.info("最终测试结果:")
    logger.info(f"  测试损失: {test_loss:.4f}")
    logger.info(f"  测试准确率: {test_acc:.4f}")
    logger.info(f"  平均类别准确率: {test_avg_class_acc:.4f}")
    logger.info("=" * 60)
    
    return trained_model, test_acc, test_avg_class_acc


if __name__ == '__main__':
    try:
        # 验证配置
        config.validate_config()
        
        # 运行主程序
        main()
        
    except KeyboardInterrupt:
        logger.info("\n用户中断程序")
    except Exception as e:
        logger.error(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
