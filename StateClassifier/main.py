"""
脑网络状态分类器主程序

该模块实现了基于图卷积网络(GCN)的脑网络状态分类系统，用于分析和分类神经元活动状态。
主要功能包括模型训练、验证和测试过程的实现。

作者: Clade 4
日期: 2025年5月23日
改进版本: 增加Focal Loss、早停机制、学习率调度等先进训练技术
"""

import torch
import torch.nn.functional as F
import random
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
import logging
import os

# 导入项目模块
from utils import get_dataset
from model import MultiLayerGCN
from config import config

# 配置日志（使用统一的日志配置方法）
logger = config.setup_logging(config.TRAINING_LOG_FILE, __name__)


class FocalLoss(torch.nn.Module):
    """
    Focal Loss - 处理类别不平衡问题
    专门设计用来解决类别不平衡导致的训练问题
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        初始化Focal Loss
        
        Parameters
        ----------
        alpha : float
            平衡因子，用于调节正负样本的权重
        gamma : float
            聚焦参数，用于降低易分类样本的权重
        reduction : str
            损失聚合方式：'mean', 'sum', 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        前向传播计算Focal Loss
        
        Parameters
        ----------
        inputs : torch.Tensor
            模型预测结果 (logits)
        targets : torch.Tensor
            真实标签
            
        Returns
        -------
        torch.Tensor
            Focal Loss值
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EarlyStopping:
    """
    早停机制 - 防止过拟合并节省训练时间
    """
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        """
        初始化早停机制
        
        Parameters
        ----------
        patience : int
            耐心值，连续多少个epoch没有改进就停止
        min_delta : float
            最小改进量，小于此值不认为是改进
        restore_best_weights : bool
            是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        """
        检查是否应该早停
        
        Parameters
        ----------
        val_loss : float
            当前验证损失
        model : nn.Module
            当前模型
            
        Returns
        -------
        bool
            是否应该早停
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """保存最佳模型权重"""
        self.best_weights = model.state_dict().copy()


def set_random_seeds():
    """
    设置随机种子以确保结果可复现
    """
    torch.manual_seed(config.TORCH_SEED)
    torch.cuda.manual_seed_all(config.TORCH_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)


def train_epoch(model, optimizer, train_dataloader, device, epoch, criterion):
    """
    训练单个epoch的函数（改进版本）
    
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
    criterion : torch.nn.Module
        损失函数
        
    Returns
    -------
    float
        该epoch的平均损失
    """
    model.train()
    loss_epoch = []
    
    for i, batch in enumerate(train_dataloader):
        # 提取批次数据
        x = batch[0].squeeze().to(device)  # 节点特征
        y = batch[1].to(device)  # 标签
        edge = batch[2].squeeze().to(device)  # 边连接信息
        
        # 改进的数据增强策略
        random_num = np.random.uniform()
        if 0.3 < random_num < 0.6:
            # 随机丢弃一部分边（Edge Dropout）
            num_edges = edge.shape[1]
            keep_ratio = 0.8
            keep_indices = np.random.choice(num_edges, int(num_edges * keep_ratio), replace=False)
            edge = edge[:, keep_indices]
            
        elif 0.6 < random_num < 0.8:
            # 节点特征噪声
            noise = torch.randn_like(x) * 0.01
            x = x + noise

        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        pred = model(edge, x)
        
        # 计算损失
        loss = criterion(pred, y)
        loss_epoch.append(loss.item())
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 参数更新
        optimizer.step()
    
    # 计算并返回平均损失
    mean_loss = np.mean(loss_epoch)
    logger.info(f'Epoch {epoch + 1} 训练损失: {mean_loss:.4f}')
    
    return mean_loss


def evaluate_model(model, device, data_loader, criterion):
    """
    评估模型性能的函数（改进版本）
    
    Parameters
    ----------
    model : torch.nn.Module
        待评估的模型
    device : torch.device
        计算设备(CPU/GPU)
    data_loader : DataLoader
        数据加载器
    criterion : torch.nn.Module
        损失函数
        
    Returns
    -------
    tuple
        (平均损失, 整体准确率, 平均类别准确率, 预测结果, 真实标签)
    """
    model.eval()
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
    
    return mean_loss, overall_acc, avg_class_acc, pred_scores, true_scores


def train_model(model, train_dataloader, valid_dataloader, device):
    """
    完整的模型训练流程（大幅改进版本）
    
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
    # 使用Focal Loss处理类别不平衡
    criterion = FocalLoss(alpha=1, gamma=2)
    logger.info("✓ 使用Focal Loss处理类别不平衡问题")
    
    # 设置AdamW优化器（相比Adam有更好的泛化性能）
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        weight_decay=config.WEIGHT_DECAY, 
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 学习率调度器：当验证损失不再下降时降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    
    # 早停机制
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    
    # 记录最佳性能
    best_acc = 0.0
    best_epoch = 0
    best_avg_class_acc = 0.0
    best_model_state = None
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    val_accs = []
    
    logger.info(f"开始改进版训练，总共 {config.NUM_EPOCHS} 个epoch")
    logger.info("训练技术: Focal Loss + AdamW + 学习率调度 + 早停 + 梯度裁剪")
    
    # 训练循环
    for epoch in range(config.NUM_EPOCHS):
        # 训练一个epoch
        train_loss = train_epoch(model, optimizer, train_dataloader, device, epoch, criterion)
        
        # 在验证集上评估
        val_loss, val_acc, val_avg_class_acc, _, _ = evaluate_model(model, device, valid_dataloader, criterion)
        
        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
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
            f"验证损失: {val_loss:.4f}, "
            f"验证准确率: {val_acc:.4f}, "
            f"最佳准确率: {best_acc:.4f} (Epoch {best_epoch + 1}), "
            f"学习率: {current_lr:.2e}"
        )
        
        # 早停检查
        if early_stopping(val_loss, model):
            logger.info(f"早停触发，在第 {epoch + 1} 个epoch停止训练")
            logger.info(f"最佳验证准确率: {best_acc:.4f} (Epoch {best_epoch + 1})")
            break

    # 保存最佳模型
    config.create_directories()
    torch.save(best_model_state, config.BEST_MODEL_PATH)
    logger.info(f"最佳模型已保存至: {config.BEST_MODEL_PATH}")
    
    # 加载最佳模型状态
    model.load_state_dict(best_model_state)
    
    # 保存训练历史
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_epoch': best_epoch,
        'best_acc': best_acc
    }
    
    return model, training_history


def generate_detailed_report(true_labels, predictions, num_classes=6):
    """
    生成详细的分类报告
    
    Parameters
    ----------
    true_labels : np.ndarray
        真实标签
    predictions : np.ndarray
        预测标签
    num_classes : int
        类别数量
        
    Returns
    -------
    str
        详细的分类报告
    """
    target_names = [f'状态_{i}' for i in range(num_classes)]
    report = classification_report(
        true_labels, 
        predictions, 
        target_names=target_names,
        digits=4
    )
    
    return report


def main():
    """
    主函数，包含模型训练和评估的完整流程（改进版本）
    """
    logger.info("=" * 80)
    logger.info("脑网络状态分类器训练程序启动 - 改进版本")
    logger.info("改进内容: Focal Loss + 早停 + 学习率调度 + 梯度裁剪")
    logger.info("=" * 80)
    
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
    
    # 创建改进的模型实例
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
    logger.info("\n开始改进版模型训练...")
    trained_model, training_history = train_model(model, train_dataloader, valid_dataloader, device)
    
    # 在测试集上评估
    logger.info("\n在测试集上评估最终性能...")
    criterion = FocalLoss(alpha=1, gamma=2)  # 使用相同的损失函数评估
    test_loss, test_acc, test_avg_class_acc, test_predictions, test_labels = evaluate_model(
        trained_model, device, test_dataloader, criterion
    )
    
    # 生成详细报告
    detailed_report = generate_detailed_report(test_labels, test_predictions, config.NUM_CLASSES)
    
    logger.info("=" * 80)
    logger.info("最终测试结果:")
    logger.info(f"  测试损失: {test_loss:.4f}")
    logger.info(f"  测试准确率: {test_acc:.4f}")
    logger.info(f"  测试平衡准确率: {test_avg_class_acc:.4f}")
    logger.info("=" * 80)
    logger.info("详细分类报告:")
    logger.info(f"\n{detailed_report}")
    logger.info("=" * 80)
    
    # 计算性能提升
    logger.info("性能分析:")
    logger.info(f"  最佳训练epoch: {training_history['best_epoch'] + 1}")
    logger.info(f"  最佳验证准确率: {training_history['best_acc']:.4f}")
    logger.info(f"  最终测试准确率: {test_acc:.4f}")
    
    if test_acc > 0.167:  # 随机准确率为1/6 ≈ 0.167
        improvement = (test_acc - 0.167) / 0.167 * 100
        logger.info(f"  相对于随机分类的提升: {improvement:.1f}%")
    
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
