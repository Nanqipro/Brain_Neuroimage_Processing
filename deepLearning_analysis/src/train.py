import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
from typing import Tuple, List, Optional

# 使用绝对导入
from config.config import Config
from utils.logger import setup_logger
from models.cnn3d import CNN3D
from utils.visualization import plot_training_history, visualize_sample, plot_confusion_matrix

# 设置日志
logger = setup_logger('training', Config.OUTPUT_DIR / 'training.log')

class NeuroDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, config: Config):
        """
        初始化数据集
        Args:
            X: 输入数据，形状为 (N, T, H, W, C)
            y: 标签数据
            config: 配置对象
        """
        # 调整输入尺寸
        self.X = torch.tensor(self._resize_data(X, config), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
        logger.info(f"X shape after resize: {self.X.shape}")
        logger.info(f"y shape: {self.y.shape}")

    def _resize_data(self, X: np.ndarray, config: Config) -> np.ndarray:
        """调整数据尺寸"""
        N, T, H, W, C = X.shape
        if H > config.INPUT_HEIGHT or W > config.INPUT_WIDTH:
            from scipy.ndimage import zoom
            scale_h = config.INPUT_HEIGHT / H
            scale_w = config.INPUT_WIDTH / W
            resized_X = np.zeros((N, T, config.INPUT_HEIGHT, config.INPUT_WIDTH, C))
            
            for i in range(N):
                for t in range(T):
                    for c in range(C):
                        resized_X[i, t, :, :, c] = zoom(X[i, t, :, :, c], (scale_h, scale_w))
            
            return resized_X
        return X

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        x = x.permute(3, 0, 1, 2)
        return x, self.y[idx]

class Trainer:
    def __init__(self, model: nn.Module, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        # 移除 verbose 参数
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        # 设置 CUDA 内存分配器
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.backends.cudnn.benchmark = True
            
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 清空优化器梯度
        self.optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # 缩放损失以适应梯度累积
            loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS
            
            # 反向传播
            loss.backward()
            
            # 梯度累积
            if (i + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            running_loss += loss.item() * inputs.size(0) * self.config.GRADIENT_ACCUMULATION_STEPS
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        return val_loss, val_acc

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[List[float], List[float], List[float], List[float]]:
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            logger.info(
                f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"  # 添加学习率日志
            )
            
            self.scheduler.step(val_loss)
            
            # 检查学习率是否改变
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                logger.info(f"Learning rate adjusted from {current_lr:.6f} to {new_lr:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_PATH)
            else:
                patience_counter += 1
                if patience_counter >= self.config.EARLY_STOP_PATIENCE:
                    logger.info("Early stopping triggered")
                    break
        
        return train_losses, val_losses, train_accuracies, val_accuracies

def main():
    # 设置配置
    Config.setup()
    
    # 加载数据
    X = np.load(Config.X_PATH)
    y = np.load(Config.Y_PATH)
    
    # 打印数据形状
    logger.info(f"原始数据形状 - X: {X.shape}, y: {y.shape}")
    
    # 确保 y 是一维的
    if y.ndim > 1:
        y = y.ravel()
    
    # 将字符串标签转换为数值
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    logger.info(f"标签类别: {label_encoder.classes_}")
    logger.info(f"转换后的标签形状: {y.shape}")
    
    # 数据集划分
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # 创建数据加载器
    train_dataset = NeuroDataset(X_train, y_train, Config)
    val_dataset = NeuroDataset(X_val, y_val, Config)
    test_dataset = NeuroDataset(X_test, y_test, Config)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS
    )
    test_loader = DataLoader(  # 添加测试数据加载器
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    
    # 获取类别数量
    num_classes = len(label_encoder.classes_)
    logger.info(f"类别数量: {num_classes}")
    
    # 创建模型和训练器
    model = CNN3D(num_classes=num_classes)  # 使用实际的类别数量
    trainer = Trainer(model, Config)
    
    # 训练模型
    history = trainer.train(train_loader, val_loader)
    
    # 绘制训练历史
    plot_training_history(*history)

    # 在测试集上评估模型
    model.load_state_dict(torch.load(Config.MODEL_PATH, weights_only=True))
    trainer.model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(trainer.device)
            outputs = trainer.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 绘制混淆矩阵
    classes = label_encoder.classes_  # 使用原始的类别名称
    plot_confusion_matrix(all_labels, all_preds, classes, normalize=True)

if __name__ == "__main__":
    main()
