#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的训练系统
==============

实现先进的训练技术，包括学习率调度、早停、交叉验证等

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CyclicLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss - 处理类别不平衡问题
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        初始化Focal Loss
        
        Parameters
        ----------
        alpha : float
            平衡因子
        gamma : float
            聚焦参数
        reduction : str
            损失聚合方式
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        前向传播
        
        Parameters
        ----------
        inputs : torch.Tensor
            模型预测结果
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


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失函数
    """
    
    def __init__(self, num_classes, smoothing=0.1):
        """
        初始化标签平滑损失
        
        Parameters
        ----------
        num_classes : int
            类别数量
        smoothing : float
            平滑参数
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        """
        前向传播
        
        Parameters
        ----------
        inputs : torch.Tensor
            模型预测结果
        targets : torch.Tensor
            真实标签
            
        Returns
        -------
        torch.Tensor
            标签平滑损失值
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # 创建平滑标签
        smooth_labels = torch.zeros_like(log_probs)
        smooth_labels.fill_(self.smoothing / (self.num_classes - 1))
        smooth_labels.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        loss = -torch.sum(smooth_labels * log_probs, dim=1)
        return loss.mean()


class EarlyStopping:
    """
    早停机制
    """
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        """
        初始化早停机制
        
        Parameters
        ----------
        patience : int
            耐心值
        min_delta : float
            最小改进量
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
            验证损失
        model : nn.Module
            模型
            
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


class AdvancedTrainer:
    """
    高级训练器
    """
    
    def __init__(self, 
                 model, 
                 device,
                 num_classes=6,
                 loss_type='focal',
                 optimizer_type='adamw',
                 scheduler_type='reduce_lr',
                 use_early_stopping=True):
        """
        初始化高级训练器
        
        Parameters
        ----------
        model : nn.Module
            待训练的模型
        device : torch.device
            计算设备
        num_classes : int
            类别数量
        loss_type : str
            损失函数类型
        optimizer_type : str
            优化器类型
        scheduler_type : str
            学习率调度器类型
        use_early_stopping : bool
            是否使用早停
        """
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.use_early_stopping = use_early_stopping
        
        # 初始化损失函数
        self._init_loss_function()
        
        # 初始化优化器
        self._init_optimizer()
        
        # 初始化学习率调度器
        self._init_scheduler()
        
        # 初始化早停
        if self.use_early_stopping:
            self.early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        
        # 训练历史
        self.history = defaultdict(list)
        
    def _init_loss_function(self):
        """初始化损失函数"""
        if self.loss_type == 'focal':
            self.criterion = FocalLoss(alpha=1, gamma=2)
        elif self.loss_type == 'label_smoothing':
            self.criterion = LabelSmoothingLoss(self.num_classes, smoothing=0.1)
        elif self.loss_type == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"未知的损失函数类型: {self.loss_type}")
        
        logger.info(f"使用损失函数: {self.loss_type}")
    
    def _init_optimizer(self, lr=0.001, weight_decay=1e-4):
        """初始化优化器"""
        if self.optimizer_type == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.optimizer_type == 'adamw':
            self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.optimizer_type == 'sgd':
            self.optimizer = SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"未知的优化器类型: {self.optimizer_type}")
        
        logger.info(f"使用优化器: {self.optimizer_type}, 学习率: {lr}")
    
    def _init_scheduler(self):
        """初始化学习率调度器"""
        if self.scheduler_type == 'reduce_lr':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        elif self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50)
        elif self.scheduler_type == 'cyclic':
            self.scheduler = CyclicLR(
                self.optimizer, base_lr=1e-5, max_lr=1e-2, step_size_up=10
            )
        else:
            self.scheduler = None
        
        logger.info(f"使用学习率调度器: {self.scheduler_type}")
    
    def train_epoch(self, train_loader):
        """
        训练一个epoch
        
        Parameters
        ----------
        train_loader : DataLoader
            训练数据加载器
            
        Returns
        -------
        float
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc='训练中')
        
        for batch in progress_bar:
            # 提取批次数据
            x = batch[0].squeeze().to(self.device)
            labels = batch[1].to(self.device)
            edge_index = batch[2].squeeze().to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(edge_index, x)
            
            # 计算损失
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 更新学习率（对于cyclic调度器）
            if self.scheduler_type == 'cyclic':
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({'损失': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def evaluate(self, data_loader):
        """
        评估模型
        
        Parameters
        ----------
        data_loader : DataLoader
            数据加载器
            
        Returns
        -------
        tuple
            (损失, 准确率, 平衡准确率, 所有预测, 所有标签)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # 提取批次数据
                x = batch[0].squeeze().to(self.device)
                labels = batch[1].to(self.device)
                edge_index = batch[2].squeeze().to(self.device)
                
                # 前向传播
                outputs = self.model(edge_index, x)
                
                # 计算损失
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1
                
                # 获取预测
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / num_batches
        accuracy = accuracy_score(all_labels, all_predictions)
        balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy, balanced_acc, all_predictions, all_labels
    
    def train(self, train_loader, val_loader, epochs=100, verbose=True):
        """
        完整训练流程
        
        Parameters
        ----------
        train_loader : DataLoader
            训练数据加载器
        val_loader : DataLoader
            验证数据加载器
        epochs : int
            训练轮数
        verbose : bool
            是否显示详细信息
            
        Returns
        -------
        dict
            训练历史
        """
        logger.info(f"开始训练，总共 {epochs} 个epoch")
        
        best_val_acc = 0.0
        best_epoch = 0
        
        for epoch in range(epochs):
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc, val_balanced_acc, _, _ = self.evaluate(val_loader)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_balanced_acc'].append(val_balanced_acc)
            
            # 更新学习率（对于reduce_lr和cosine调度器）
            if self.scheduler_type in ['reduce_lr']:
                self.scheduler.step(val_loss)
            elif self.scheduler_type in ['cosine']:
                self.scheduler.step()
            
            # 检查是否是最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
            
            # 早停检查
            if self.use_early_stopping:
                if self.early_stopping(val_loss, self.model):
                    logger.info(f"早停在第 {epoch + 1} 个epoch")
                    break
            
            # 打印进度
            if verbose:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"训练损失: {train_loss:.4f}, "
                    f"验证损失: {val_loss:.4f}, "
                    f"验证准确率: {val_acc:.4f}, "
                    f"验证平衡准确率: {val_balanced_acc:.4f}, "
                    f"最佳准确率: {best_val_acc:.4f} (Epoch {best_epoch + 1})"
                )
        
        logger.info(f"训练完成！最佳验证准确率: {best_val_acc:.4f}")
        return dict(self.history)
    
    def cross_validate(self, dataset, labels, n_splits=5, epochs=50):
        """
        K折交叉验证
        
        Parameters
        ----------
        dataset : list
            数据集
        labels : np.ndarray
            标签
        n_splits : int
            折数
        epochs : int
            每折训练轮数
            
        Returns
        -------
        dict
            交叉验证结果
        """
        logger.info(f"开始 {n_splits} 折交叉验证")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = {
            'fold_accuracies': [],
            'fold_balanced_accuracies': [],
            'all_predictions': [],
            'all_labels': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
            logger.info(f"训练第 {fold + 1} 折...")
            
            # 创建数据加载器
            from torch.utils.data import DataLoader, Subset
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
            
            # 重新初始化模型（为每折重新开始）
            self.model.apply(self._reset_weights)
            self._init_optimizer()
            self._init_scheduler()
            
            # 训练
            self.train(train_loader, val_loader, epochs=epochs, verbose=False)
            
            # 评估
            val_loss, val_acc, val_balanced_acc, fold_predictions, fold_labels = self.evaluate(val_loader)
            
            cv_results['fold_accuracies'].append(val_acc)
            cv_results['fold_balanced_accuracies'].append(val_balanced_acc)
            cv_results['all_predictions'].extend(fold_predictions)
            cv_results['all_labels'].extend(fold_labels)
            
            logger.info(f"第 {fold + 1} 折完成 - 准确率: {val_acc:.4f}")
        
        # 计算平均结果
        mean_acc = np.mean(cv_results['fold_accuracies'])
        std_acc = np.std(cv_results['fold_accuracies'])
        mean_balanced_acc = np.mean(cv_results['fold_balanced_accuracies'])
        std_balanced_acc = np.std(cv_results['fold_balanced_accuracies'])
        
        logger.info(f"交叉验证完成!")
        logger.info(f"平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")
        logger.info(f"平均平衡准确率: {mean_balanced_acc:.4f} ± {std_balanced_acc:.4f}")
        
        cv_results['mean_accuracy'] = mean_acc
        cv_results['std_accuracy'] = std_acc
        cv_results['mean_balanced_accuracy'] = mean_balanced_acc
        cv_results['std_balanced_accuracy'] = std_balanced_acc
        
        return cv_results
    
    def _reset_weights(self, m):
        """重置模型权重"""
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    
    def plot_training_history(self, save_path=None):
        """
        绘制训练历史
        
        Parameters
        ----------
        save_path : str, optional
            保存路径
        """
        if not self.history:
            logger.warning("没有训练历史数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.history['train_loss'], label='训练损失', color='blue')
        axes[0, 0].plot(self.history['val_loss'], label='验证损失', color='red')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(self.history['val_acc'], label='验证准确率', color='green')
        axes[0, 1].plot(self.history['val_balanced_acc'], label='平衡准确率', color='orange')
        axes[0, 1].set_title('准确率曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('准确率')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率曲线（如果可用）
        if hasattr(self.scheduler, 'get_last_lr'):
            lrs = [self.scheduler.get_last_lr()[0] for _ in range(len(self.history['train_loss']))]
            axes[1, 0].plot(lrs, label='学习率', color='purple')
            axes[1, 0].set_title('学习率曲线')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('学习率')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, '学习率信息不可用', ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 训练统计
        best_val_acc = max(self.history['val_acc'])
        best_epoch = self.history['val_acc'].index(best_val_acc)
        final_train_loss = self.history['train_loss'][-1]
        final_val_loss = self.history['val_loss'][-1]
        
        stats_text = f"""
        最佳验证准确率: {best_val_acc:.4f}
        最佳epoch: {best_epoch + 1}
        最终训练损失: {final_train_loss:.4f}
        最终验证损失: {final_val_loss:.4f}
        总epoch数: {len(self.history['train_loss'])}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, fontsize=12,
                        verticalalignment='center', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        axes[1, 1].set_title('训练统计')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练历史图已保存: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, predictions, labels, save_path=None):
        """
        绘制混淆矩阵
        
        Parameters
        ----------
        predictions : list
            预测结果
        labels : list
            真实标签
        save_path : str, optional
            保存路径
        """
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[f'类别{i}' for i in range(self.num_classes)],
                   yticklabels=[f'类别{i}' for i in range(self.num_classes)])
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵已保存: {save_path}")
        
        plt.show()
    
    def generate_classification_report(self, predictions, labels):
        """
        生成分类报告
        
        Parameters
        ----------
        predictions : list
            预测结果
        labels : list
            真实标签
            
        Returns
        -------
        str
            分类报告
        """
        target_names = [f'类别{i}' for i in range(self.num_classes)]
        report = classification_report(labels, predictions, target_names=target_names)
        
        logger.info("分类报告:")
        logger.info("\n" + report)
        
        return report


if __name__ == "__main__":
    # 测试代码
    import torch
    from torch_geometric.data import Data, DataLoader
    from improved_models import create_improved_model
    
    # 创建测试数据
    num_samples = 100
    num_nodes = 50
    num_features = 3
    num_classes = 6
    
    data_list = []
    labels = []
    
    for i in range(num_samples):
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, (2, 100))
        label = torch.randint(0, num_classes, (1,)).item()
        
        data_list.append((x, label, edge_index))
        labels.append(label)
    
    # 创建数据加载器
    train_loader = DataLoader(data_list[:80], batch_size=16, shuffle=True)
    val_loader = DataLoader(data_list[80:], batch_size=16, shuffle=False)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_improved_model(
        model_type='advanced',
        input_dim=num_features,
        hidden_dims=[32, 64, 32],
        num_classes=num_classes,
        dropout=0.2,
        architecture='hybrid'
    )
    
    # 创建训练器
    trainer = AdvancedTrainer(
        model=model,
        device=device,
        num_classes=num_classes,
        loss_type='focal',
        optimizer_type='adamw',
        scheduler_type='reduce_lr',
        use_early_stopping=True
    )
    
    # 训练模型
    history = trainer.train(train_loader, val_loader, epochs=20)
    
    # 评估模型
    val_loss, val_acc, val_balanced_acc, predictions, true_labels = trainer.evaluate(val_loader)
    
    print(f"最终验证结果:")
    print(f"  损失: {val_loss:.4f}")
    print(f"  准确率: {val_acc:.4f}")
    print(f"  平衡准确率: {val_balanced_acc:.4f}")
    
    # 绘制结果
    trainer.plot_training_history()
    trainer.plot_confusion_matrix(predictions, true_labels)
    trainer.generate_classification_report(predictions, true_labels) 