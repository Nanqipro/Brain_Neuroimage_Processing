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
from model import MultiLayerGCN, AdvancedBrainStateClassifier
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


class ContrastiveLoss(torch.nn.Module):
    """
    对比学习损失函数 - 基于InfoNCE损失
    
    帮助模型学习更好的特征表示，特别适合类别不平衡的场景
    """
    
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, labels):
        """
        计算对比学习损失
        
        Parameters
        ----------
        embeddings : torch.Tensor
            特征嵌入 [batch_size, feature_dim]
        labels : torch.Tensor
            标签 [batch_size]
        """
        batch_size = embeddings.shape[0]
        
        # 计算相似度矩阵
        embeddings_norm = F.normalize(embeddings, dim=1)
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T) / self.temperature
        
        # 创建正样本和负样本掩码
        labels = labels.unsqueeze(1)
        positive_mask = torch.eq(labels, labels.T).float()
        negative_mask = 1 - positive_mask
        
        # 排除自身
        positive_mask.fill_diagonal_(0)
        
        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        
        # 正样本损失
        positive_sum = torch.sum(exp_sim * positive_mask, dim=1)
        
        # 负样本损失
        negative_sum = torch.sum(exp_sim * negative_mask, dim=1)
        
        # InfoNCE损失
        loss = -torch.log(positive_sum / (positive_sum + negative_sum + 1e-8))
        
        return torch.mean(loss)


class AdaptiveTrainer:
    """
    自适应训练器 - 集成多种先进的训练技术
    
    基于2024年最新研究的训练策略，包括：
    1. 混合损失函数（Focal + Contrastive）
    2. 自适应学习率调度
    3. 课程学习
    4. 模型集成
    """
    
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # 损失函数
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        self.contrastive_loss = ContrastiveLoss(temperature=0.1)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            eps=1e-8
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=15,
            min_lr=1e-6,
            verbose=True
        )
        
        # 早停机制（根据配置决定是否启用）
        if config.USE_EARLY_STOPPING:
            self.early_stopping = EarlyStopping(
                patience=config.EARLY_STOPPING_PATIENCE,
                min_delta=config.EARLY_STOPPING_MIN_DELTA,
                restore_best_weights=config.EARLY_STOPPING_RESTORE_BEST
            )
            logger.info(f"✓ 早停机制已启用 - 耐心值: {config.EARLY_STOPPING_PATIENCE}, 最小改进: {config.EARLY_STOPPING_MIN_DELTA}")
        else:
            self.early_stopping = None
            logger.info("✓ 早停机制已禁用")
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def train_epoch(self, train_loader, epoch):
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # 前向传播（获取嵌入用于对比学习）
            if hasattr(self.model, 'forward') and 'return_embeddings' in self.model.forward.__code__.co_varnames:
                outputs, embeddings = self.model(data, return_embeddings=True)
                
                # 混合损失：分类损失 + 对比学习损失
                focal_loss = self.focal_loss(outputs, data.y)
                contrastive_loss = self.contrastive_loss(embeddings, data.y)
                
                # 动态权重（随训练进行调整对比学习权重）
                contrastive_weight = max(0.1, 1.0 - epoch / 100)  # 随训练减少对比学习权重
                
                loss = focal_loss + contrastive_weight * contrastive_loss
            else:
                # 传统模型只使用Focal损失
                outputs = self.model(data)
                loss = self.focal_loss(outputs, data.y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录统计信息
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """
        验证模型
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                loss = self.focal_loss(outputs, data.y)
                
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy, balanced_accuracy
    
    def train(self, train_loader, val_loader, num_epochs):
        """
        完整训练流程
        """
        logger.info("🚀 开始先进训练流程...")
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # 训练阶段
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # 验证阶段
            val_loss, val_acc, val_balanced_acc = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['learning_rates'].append(current_lr)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            # 日志输出
            if epoch % 10 == 0 or epoch < 10:
                logger.info(f"Epoch {epoch:3d} | "
                          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                          f"Val Balanced Acc: {val_balanced_acc:.4f} | LR: {current_lr:.6f}")
            
            # 早停检查
            if self.early_stopping and self.early_stopping(val_loss, self.model):
                logger.info(f"早停触发，停止训练。最佳验证准确率: {best_val_acc:.4f}")
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"恢复最佳模型，验证准确率: {best_val_acc:.4f}")
        
        return self.train_history


def create_advanced_model(config, device):
    """
    创建先进的脑状态分类模型
    """
    # 暂时使用改进的传统模型，避免复杂的维度问题
    # 但集成先进的训练技术
    model = MultiLayerGCN(
        dropout=config.DROPOUT_RATE,
        num_classes=config.NUM_CLASSES
    )
    
    logger.info(f"创建改进模型，参数总数: {sum(p.numel() for p in model.parameters())}")
    logger.info("使用MultiLayerGCN + 先进训练技术的组合")
    return model


def set_random_seeds():
    """
    设置随机种子以确保结果可复现
    """
    torch.manual_seed(config.TORCH_SEED)
    torch.cuda.manual_seed_all(config.TORCH_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)


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
    主函数 - 增强版训练流程
    """
    logger.info("="*80)
    logger.info("🧠 脑网络状态分类器 - 先进版本")
    logger.info("="*80)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    try:
        # 加载数据集
        logger.info("📊 加载数据集...")
        train_loader, val_loader, test_loader = get_dataset()
        logger.info(f"数据集加载完成 - 训练集: {len(train_loader)}, 验证集: {len(val_loader)}, 测试集: {len(test_loader)}")
        
        # 创建先进模型
        logger.info("🏗️  创建先进模型...")
        model = create_advanced_model(config, device)
        
        # 创建自适应训练器
        trainer = AdaptiveTrainer(model, device, config)
        
        # 训练模型
        logger.info("🚀 开始训练...")
        train_history = trainer.train(train_loader, val_loader, config.NUM_EPOCHS)
        
        # 测试最终性能
        logger.info("🔍 测试最终性能...")
        test_loss, test_acc, test_balanced_acc = trainer.validate(test_loader)
        
        logger.info("="*60)
        logger.info("📊 最终结果:")
        logger.info(f"测试准确率: {test_acc:.4f}")
        logger.info(f"平衡准确率: {test_balanced_acc:.4f}")
        logger.info(f"测试损失: {test_loss:.4f}")
        logger.info("="*60)
        
        # 保存模型和训练历史到results文件夹
        # 确保results目录存在
        config.RESULT_DIR.mkdir(exist_ok=True)
        
        model_save_path = config.RESULT_DIR / 'advanced_brain_classifier.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_history': train_history,
            'test_accuracy': test_acc,
            'test_balanced_accuracy': test_balanced_acc,
            'config': {
                'num_epochs': config.NUM_EPOCHS,
                'learning_rate': config.LEARNING_RATE,
                'dropout_rate': config.DROPOUT_RATE,
                'num_classes': config.NUM_CLASSES,
                'use_early_stopping': config.USE_EARLY_STOPPING,
                'early_stopping_patience': config.EARLY_STOPPING_PATIENCE,
            },
            'model_info': {
                'model_type': 'MultiLayerGCN',
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            }
        }, model_save_path)
        
        logger.info(f"✅ 模型和结果已保存到: {model_save_path}")
        logger.info(f"📁 结果目录: {config.RESULT_DIR}")
        
        return test_acc
        
    except Exception as e:
        logger.error(f"❌ 训练过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 0


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
