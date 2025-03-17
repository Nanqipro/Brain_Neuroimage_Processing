import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import copy
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
def set_random_seed(seed):
    """
    设置随机种子以确保结果可重现
    
    参数:
        seed (int): 随机种子值
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

# LSTM数据集类
class NeuronDataset(Dataset):
    """
    神经元数据集类，用于处理序列数据
    
    属性:
        X: 神经元活动数据
        y: 行为标签
        sequence_length: 序列长度
    """
    def __init__(self, X, y, sequence_length):
        """
        参数:
            X (numpy.ndarray): 神经元活动数据
            y (numpy.ndarray): 行为标签
            sequence_length (int): 序列长度
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.sequence_length = sequence_length
        
    def __len__(self):
        """返回数据集长度"""
        return len(self.X) - self.sequence_length
        
    def __getitem__(self, idx):
        """获取特定索引的样本"""
        return (self.X[idx:idx+self.sequence_length], 
                self.y[idx+self.sequence_length-1])

class NeuronAutoencoder(nn.Module):
    """
    神经元自编码器
    用于提取神经元活动的潜在特征
    
    属性:
        input_size: 输入特征维度
        latent_dim: 潜在特征维度
        encoder: 编码器网络
        decoder: 解码器网络
    """
    def __init__(self, input_size, hidden_size, latent_dim):
        """
        参数:
            input_size (int): 输入特征维度
            hidden_size (int): 隐藏层维度
            latent_dim (int): 潜在特征维度
        """
        super(NeuronAutoencoder, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, latent_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, input_size),
            nn.Tanh()  # 使用Tanh确保输出在[-1,1]范围内
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量
        
        返回:
            tuple: (encoded, decoded) 编码后和解码后的张量
        """
        # 检查输入维度是否与模型匹配
        if x.size(-1) != self.input_size:
            # 处理输入维度不匹配的情况
            if x.size(-1) < self.input_size:
                # 输入维度小于模型期望，填充零
                padding_size = self.input_size - x.size(-1)
                padding = torch.zeros(*x.shape[:-1], padding_size, device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                # 输入维度大于模型期望，截断
                x = x[..., :self.input_size]
        
        try:
            # 处理3D输入 (batch_size, seq_len, features)
            original_shape = x.shape
            if len(original_shape) == 3:
                # 将3D张量重塑为2D
                batch_size, seq_len, features = original_shape
                x_reshaped = x.reshape(-1, features)
                
                # 通过编码器和解码器
                encoded = self.encoder(x_reshaped)
                decoded = self.decoder(encoded)
                
                # 重塑回原始形状
                encoded = encoded.reshape(batch_size, seq_len, self.latent_dim)
                decoded = decoded.reshape(original_shape)
                
                return encoded, decoded
            else:
                # 处理2D输入 (对单个样本处理)
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded
                
        except RuntimeError as e:
            print(f"自编码器前向传播错误: {str(e)}")
            print(f"输入形状: {x.shape}, 期望输入大小: {self.input_size}")
            # 返回零张量
            return (
                torch.zeros(*x.shape[:-1], self.latent_dim, device=x.device),
                torch.zeros_like(x)
            )

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    用于捕捉神经元之间的关联关系
    
    属性:
        num_heads: 注意力头数量
        head_dim: 每个头的维度
        query/key/value: 线性变换层
        dropout: Dropout层
        output_linear: 输出线性层
    """
    def __init__(self, input_dim, num_heads, dropout=0.1):
        """
        参数:
            input_dim (int): 输入维度
            num_heads (int): 注意力头数量
            dropout (float): Dropout比率
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert self.head_dim * num_heads == input_dim, "input_dim必须能被num_heads整除"
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(input_dim, input_dim)
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量
            mask (torch.Tensor, optional): 掩码张量
        
        返回:
            tuple: (output, attention_weights) 输出张量和注意力权重
        """
        batch_size = x.size(0)
        
        # 线性变换并分头
        query = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, value)
        
        # 重组并线性变换
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.output_linear(context)
        
        return output, attention_weights

class TemporalAttention(nn.Module):
    """
    时间注意力机制
    用于捕捉时间序列中的重要模式
    
    属性:
        attention: 注意力计算层序列
    """
    def __init__(self, hidden_size):
        """
        参数:
            hidden_size (int): 隐藏层大小
        """
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, hidden_states):
        """
        前向传播
        
        参数:
            hidden_states (torch.Tensor): 隐藏状态张量
        
        返回:
            tuple: (attended, attention_weights) 加权后的隐藏状态和注意力权重
        """
        attention_weights = self.attention(hidden_states)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended = torch.sum(hidden_states * attention_weights, dim=1)
        return attended, attention_weights

class EnhancedNeuronLSTM(nn.Module):
    """
    增强版神经元LSTM模型
    整合了自编码器、双向LSTM和多层注意力机制
    
    属性:
        input_size: 输入特征维度
        hidden_size: 隐藏层大小
        num_layers: LSTM层数
        num_classes: 输出类别数
        autoencoder: 自编码器实例
        lstm: 双向LSTM实例
        multihead_attention: 多头注意力实例
        temporal_attention: 时间注意力实例
        batch_norm: 批标准化层
        classifier: 分类器网络
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                 latent_dim=32, num_heads=4, dropout=0.2):
        """
        参数:
            input_size (int): 输入特征维度
            hidden_size (int): 隐藏层大小
            num_layers (int): LSTM层数
            num_classes (int): 输出类别数
            latent_dim (int): 潜在特征维度
            num_heads (int): 注意力头数量
            dropout (float): Dropout比率
        """
        super(EnhancedNeuronLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # 确保隐藏层大小是多头注意力头数的倍数
        if hidden_size % num_heads != 0:
            adjusted_hidden_size = (hidden_size // num_heads) * num_heads
            print(f"警告: 隐藏层大小({hidden_size})不是头数({num_heads})的倍数，调整为{adjusted_hidden_size}")
            hidden_size = adjusted_hidden_size
            self.hidden_size = hidden_size
        
        # 自编码器用于特征提取
        self.autoencoder = NeuronAutoencoder(input_size, hidden_size, latent_dim)
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 多头注意力机制
        self.multihead_attention = MultiHeadAttention(hidden_size * 2, num_heads)
        
        # 时间注意力机制
        self.temporal_attention = TemporalAttention(hidden_size * 2)
        
        # 批标准化
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 记录模型配置
        self.config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_classes': num_classes,
            'latent_dim': latent_dim,
            'num_heads': num_heads,
            'dropout': dropout
        }
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量
        
        返回:
            tuple: (output, attention_weights, temporal_weights) 分类输出和注意力权重
        """
        batch_size = x.size(0)
        
        # 检查输入维度是否与模型匹配
        if x.size(-1) != self.input_size:
            # 处理输入维度不匹配的情况
            if x.size(-1) < self.input_size:
                # 输入维度小于模型期望，填充零
                padding = torch.zeros(batch_size, x.size(1), self.input_size - x.size(-1), device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                # 输入维度大于模型期望，截断
                x = x[..., :self.input_size]
            
        try:
            # 通过自编码器提取特征
            encoded, decoded = self.autoencoder(x)
            
            # 通过LSTM处理序列
            lstm_out, _ = self.lstm(encoded)
            
            # 应用多头注意力
            attended, attention_weights = self.multihead_attention(lstm_out, None)
            
            # 应用时间注意力
            temporal_context, temporal_weights = self.temporal_attention(attended)
            
            # 批标准化 - 注意temporal_context是2D的
            normalized = self.batch_norm(temporal_context)
            
            # 分类
            output = self.classifier(normalized)
            
            return output, attention_weights, temporal_weights
            
        except RuntimeError as e:
            # 处理可能的运行时错误
            if "size mismatch" in str(e) or "dimension" in str(e):
                print(f"前向传播中的维度错误: {str(e)}")
                print(f"输入形状: {x.shape}, 模型输入大小: {self.input_size}")
                # 返回一个全零的输出张量和空的注意力权重
                return (
                    torch.zeros(batch_size, self.num_classes, device=x.device),
                    torch.zeros(batch_size, 1, 1, device=x.device),  # 空的attention_weights
                    torch.zeros(batch_size, 1, device=x.device)      # 空的temporal_weights
                )
            else:
                # 重新抛出其他类型的错误
                raise e
    
    def get_config(self):
        """
        获取模型配置
        
        返回:
            dict: 配置字典
        """
        return self.config
    
    def summary(self):
        """
        打印模型摘要信息
        
        返回:
            dict: 包含参数统计的字典
        """
        print("\n模型摘要:")
        print(f"输入大小: {self.input_size}")
        print(f"隐藏层大小: {self.hidden_size}")
        print(f"LSTM层数: {self.num_layers}")
        print(f"类别数: {self.num_classes}")
        print(f"潜在维度: {self.latent_dim}")
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params
        }

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, config, early_stopping_enabled=False):
    """
    训练增强型神经元LSTM模型
    
    参数:
        model (EnhancedNeuronLSTM): 待训练的模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        criterion (nn.Module): 损失函数
        optimizer (torch.optim.Optimizer): 优化器
        device (torch.device): 计算设备
        num_epochs (int): 训练轮数
        config (object): 配置对象
        early_stopping_enabled (bool): 是否启用早停机制，默认为False
    
    返回:
        tuple: (model, metrics_dict) 训练后的模型和指标字典
    """
    model.train()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    reconstruction_losses = []
    learning_rates = []
    best_val_acc = 0.0
    
    # 保存最后一个批次的注意力权重用于可视化
    last_attention_weights = None
    last_temporal_weights = None
    
    # 添加重构损失
    reconstruction_criterion = nn.MSELoss()
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 确保训练指标日志目录存在
    os.makedirs(os.path.dirname(config.metrics_log), exist_ok=True)
    
    # 创建或截断训练指标日志文件
    with open(config.metrics_log, 'w') as f:
        f.write('epoch,train_loss,train_acc,val_loss,val_acc,reconstruction_loss\n')
    
    # 早停相关参数
    patience = config.analysis_params.get('early_stopping_patience', 20)
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        correct = 0
        total = 0
        
        # 训练循环
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs, attention_weights, temporal_weights = model(batch_X)
            
            # 保存最后一个批次的注意力权重
            if attention_weights is not None:
                last_attention_weights = attention_weights
            if temporal_weights is not None:
                last_temporal_weights = temporal_weights
            
            # 计算分类损失
            loss = criterion(outputs, batch_y)
            
            # 计算重构损失
            _, decoded = model.autoencoder(batch_X.view(-1, batch_X.size(-1)))
            reconstruction_loss = reconstruction_criterion(decoded, batch_X.view(-1, batch_X.size(-1)))
            
            # 总损失
            recon_weight = config.analysis_params.get('reconstruction_loss_weight', 0.1)
            total_batch_loss = loss + recon_weight * reconstruction_loss
            
            # 反向传播
            total_batch_loss.backward()
            
            # 梯度裁剪
            max_norm = config.analysis_params.get('gradient_clip_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            # 更新参数
            optimizer.step()
            
            # 统计
            total_loss += loss.item() * batch_X.size(0)
            total_recon_loss += reconstruction_loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        # 计算平均损失和准确率
        avg_train_loss = total_loss / total
        avg_recon_loss = total_recon_loss / total
        train_accuracy = 100 * correct / total
        
        # 记录训练指标
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        reconstruction_losses.append(avg_recon_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                val_outputs, _, _ = model(val_X)
                batch_loss = criterion(val_outputs, val_y)
                val_loss += batch_loss.item() * val_X.size(0)
                
                _, val_predicted = torch.max(val_outputs, 1)
                val_correct += (val_predicted == val_y).sum().item()
                val_total += val_y.size(0)
        
        # 计算验证损失和准确率
        avg_val_loss = val_loss / val_total
        val_accuracy = 100 * val_correct / val_total
        
        # 记录验证指标
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # 打印训练信息
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, LR: {current_lr:.6f}")
        
        # 记录指标到日志文件
        with open(config.metrics_log, 'a') as f:
            f.write(f'{epoch+1},{avg_train_loss:.4f},{train_accuracy:.2f},{avg_val_loss:.4f},{val_accuracy:.2f},{avg_recon_loss:.4f}\n')
        
        # 早停检查
        if early_stopping_enabled:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}, best val loss: {best_val_loss:.4f}")
                    break
        else:
            # 如果禁用早停，仍然保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
        
        # 调整学习率
        scheduler.step(avg_val_loss)
    
    # 如果禁用早停且完成所有epoch，打印最终结果
    if not early_stopping_enabled:
        print(f"Training completed for all {num_epochs} epochs, best val loss: {best_val_loss:.4f}")
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"已恢复最佳模型 (验证损失: {best_val_loss:.4f})")
    
    # 保存最佳模型到文件
    try:
        os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'attention_weights': last_attention_weights.detach().cpu().numpy() if last_attention_weights is not None else None,
            'temporal_weights': last_temporal_weights.detach().cpu().numpy() if last_temporal_weights is not None else None
        }, config.model_path)
        print(f"模型已保存到: {config.model_path}")
    except Exception as e:
        print(f"保存模型时出错: {str(e)}")
        with open(config.error_log, 'a') as f:
            f.write(f"保存模型失败 - {str(e)}\n")
    
    # 返回训练结果
    metrics_dict = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': max(val_accuracies) if val_accuracies else 0,
        'reconstruction_losses': reconstruction_losses,
        'learning_rates': learning_rates,
        'attention_weights': last_attention_weights.detach().cpu().numpy() if last_attention_weights is not None else None,
        'temporal_weights': last_temporal_weights.detach().cpu().numpy() if last_temporal_weights is not None else None
    }
    
    return model, metrics_dict

def plot_training_metrics(metrics, config):
    """
    绘制训练和验证指标的变化曲线
    包括损失值和准确率
    
    参数:
        metrics (dict): 包含训练指标的字典
        config (object): 配置对象，包含保存路径
    """
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(metrics['train_losses'], label='Training Loss', linewidth=2)
    plt.plot(metrics['val_losses'], label='Validation Loss', linewidth=2)
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(metrics['train_accuracies'], label='Training Accuracy', linewidth=2)
    plt.plot(metrics['val_accuracies'], label='Validation Accuracy', linewidth=2)
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # 绘制重构损失曲线
    plt.subplot(1, 3, 3)
    plt.plot(metrics['reconstruction_losses'], label='Reconstruction Loss', linewidth=2)
    plt.title('Reconstruction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(config.accuracy_plot, dpi=300, bbox_inches='tight')
    plt.close()
