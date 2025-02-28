import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import warnings
import os
import datetime
from analysis_config import AnalysisConfig
warnings.filterwarnings('ignore')

# Set random seed
def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

# 数据加载和预处理类
class NeuronDataProcessor:
    def __init__(self, config):
        """
        初始化神经元数据处理器
        参数:
            config: 配置对象,包含数据文件路径等参数
        """
        self.config = config
        self.data = pd.read_excel(config.data_file)
        self.scaler = StandardScaler()  # 用于数据标准化
        self.label_encoder = LabelEncoder()  # 用于行为标签编码
        
    def preprocess_data(self):
        """
        数据预处理函数：
        1. 提取神经元数据
        2. 处理缺失值
        3. 标准化数据
        4. 编码行为标签
        """
        # Extract neuron columns
        neuron_cols = [f'n{i}' for i in range(1, 63)]
        
        # Check available columns
        available_cols = [col for col in neuron_cols if col in self.data.columns]
        print(f"Total neurons: {len(neuron_cols)}")
        print(f"Available neurons: {len(available_cols)}")
        print(f"Missing neurons: {set(neuron_cols) - set(available_cols)}")
        
        # 保存可用的神经元列名列表，以便其他方法使用
        self.available_neuron_cols = available_cols
        
        # Get only available neuron data
        X = self.data[available_cols].values
        
        # Handle missing values
        if np.isnan(X).any():
            print("Found missing values, filling with mean values")
            # Fill missing values with mean of each column
            X = np.nan_to_num(X, nan=np.nanmean(X))
        
        # Standardize neuron data
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode behavior labels
        if 'behavior' not in self.data.columns:
            raise ValueError("Behavior column not found in the dataset")
            
        # Handle missing behavior labels
        behavior_data = self.data['behavior'].fillna('unknown')
        y = self.label_encoder.fit_transform(behavior_data)
        
        # Print label encoding information
        label_mapping = dict(zip(self.label_encoder.classes_, 
                               self.label_encoder.transform(self.label_encoder.classes_)))
        print("\nBehavior label mapping:")
        for label, code in label_mapping.items():
            count = sum(y == code)
            print(f"{label}: {code} (Count: {count})")
        
        return X_scaled, y

    def apply_kmeans(self, X):
        # 应用K-means聚类
        print(f"\n使用 {self.config.n_clusters} 个聚类进行K-means聚类")
        kmeans = KMeans(n_clusters=self.config.n_clusters, 
                       random_state=self.config.random_seed,
                       n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # 打印聚类分布情况
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print("\n聚类分布:")
        for label, count in zip(unique_labels, counts):
            print(f"聚类 {label}: {count} 个样本")
        
        return kmeans, cluster_labels

# LSTM dataset class
class NeuronDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        """
        参数：
        X: 神经元活动数据
        y: 行为标签
        sequence_length: 序列长度
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length
        
    def __getitem__(self, idx):
        return (self.X[idx:idx+self.sequence_length], 
                self.y[idx+self.sequence_length-1])

class NeuronAutoencoder(nn.Module):
    """神经元活动数据的自编码器
    用于特征提取和降维
    """
    def __init__(self, input_size, hidden_size, latent_dim):
        super(NeuronAutoencoder, self).__init__()
        
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class MultiHeadAttention(nn.Module):
    """多头注意力机制
    用于捕捉神经元之间的关联关系
    """
    def __init__(self, input_dim, num_heads, dropout=0.1):
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
    """时间注意力机制
    用于捕捉时间序列中的重要模式
    """
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended = torch.sum(hidden_states * attention_weights, dim=1)
        return attended, attention_weights

class EnhancedNeuronLSTM(nn.Module):
    """增强版神经元LSTM模型
    整合了自编码器、双向LSTM和多层注意力机制
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                 latent_dim=32, num_heads=4, dropout=0.2):
        super(EnhancedNeuronLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
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
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. 通过自编码器进行特征提取
        encoded_features = []
        for t in range(x.size(1)):
            encoded, _ = self.autoencoder(x[:, t, :])
            encoded_features.append(encoded)
        encoded_sequence = torch.stack(encoded_features, dim=1)
        
        # 2. 双向LSTM处理
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(encoded_sequence, (h0, c0))
        
        # 3. 多头注意力处理
        attended_features, attention_weights = self.multihead_attention(lstm_out)
        
        # 4. 时间注意力处理
        temporal_context, temporal_weights = self.temporal_attention(attended_features)
        
        # 5. 批标准化
        normalized_features = self.batch_norm(temporal_context)
        
        # 6. 分类
        output = self.classifier(normalized_features)
        
        return output, attention_weights, temporal_weights

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, config):
    """
    增强版模型训练函数
    添加了自编码器损失和注意力权重的可视化
    """
    model.train()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    reconstruction_losses = []
    best_val_acc = 0.0
    
    # 添加重构损失
    reconstruction_criterion = nn.MSELoss()
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # 创建指标日志文件
    with open(config.metrics_log, 'w') as f:
        f.write('epoch,train_loss,train_acc,val_loss,val_acc,reconstruction_loss\n')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs, attention_weights, temporal_weights = model(batch_X)
            
            # 计算分类损失
            classification_loss = criterion(outputs, batch_y)
            
            # 计算重构损失
            _, decoded = model.autoencoder(batch_X.view(-1, batch_X.size(-1)))
            reconstruction_loss = reconstruction_criterion(decoded, batch_X.view(-1, batch_X.size(-1)))
            
            # 总损失
            loss = classification_loss + 0.1 * reconstruction_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.analysis_params['gradient_clip_norm'])
            optimizer.step()
            
            total_loss += classification_loss.item()
            total_recon_loss += reconstruction_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        
        # 记录训练指标
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        reconstruction_losses.append(avg_recon_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs, _, _ = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        # 记录验证指标
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # 更新学习率
        scheduler.step(val_accuracy)
        
        # 保存最佳模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            try:
                os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_acc': best_val_acc,
                    'attention_weights': attention_weights.detach().cpu().numpy(),
                    'temporal_weights': temporal_weights.detach().cpu().numpy()
                }, config.model_path)
                print(f"模型已保存到: {config.model_path}")
            except Exception as e:
                print(f"保存模型时出错: {str(e)}")
                with open(config.error_log, 'a') as f:
                    f.write(f"Epoch {epoch+1}: 保存模型失败 - {str(e)}\n")
        
        # 记录指标
        with open(config.metrics_log, 'a') as f:
            f.write(f'{epoch+1},{avg_train_loss:.4f},{train_accuracy:.2f},{avg_val_loss:.4f},{val_accuracy:.2f},{avg_recon_loss:.4f}\n')
        
        # 打印训练信息
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]:')
            print(f'  Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
            print(f'  Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
            print(f'  Reconstruction Loss: {avg_recon_loss:.4f}')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'reconstruction_losses': reconstruction_losses,
        'attention_weights': attention_weights.detach().cpu().numpy(),
        'temporal_weights': temporal_weights.detach().cpu().numpy()
    }

def plot_training_metrics(metrics, config):
    """
    绘制训练和验证指标的变化曲线
    包括损失值和准确率
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

def main():
    """
    主函数：
    1. 初始化配置
    2. 数据预处理
    3. 应用K-means聚类
    4. 训练LSTM模型
    5. 保存结果和可视化
    """
    config = None
    try:
        # 初始化配置
        config = AnalysisConfig()
        
        # 验证和创建目录（validate_paths 现在会自动调用 setup_directories）
        print("正在验证并创建目录结构...")
        config.validate_paths()
        
        print("正在设置随机种子...")
        set_random_seed(config.random_seed)
        
        print("正在进行数据预处理...")
        processor = NeuronDataProcessor(config)
        X_scaled, y = processor.preprocess_data()
        
        print("正在应用K-means聚类...")
        kmeans, cluster_labels = processor.apply_kmeans(X_scaled)
        
        print("正在准备训练数据...")
        X_with_clusters = np.column_stack((X_scaled, cluster_labels))
        X_train, X_test, y_train, y_test = train_test_split(
            X_with_clusters, y, test_size=config.test_size, random_state=config.random_seed
        )
        
        # 创建数据加载器
        train_dataset = NeuronDataset(X_train, y_train, config.sequence_length)
        val_dataset = NeuronDataset(X_test, y_test, config.sequence_length)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )
        
        print("正在初始化模型...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        input_size = X_with_clusters.shape[1]
        num_classes = len(np.unique(y))
        model = EnhancedNeuronLSTM(
            input_size, 
            config.hidden_size, 
            config.num_layers, 
            num_classes
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.analysis_params['weight_decay']
        )
        
        print("\n开始训练模型...")
        metrics = train_model(
            model, 
            train_loader,
            val_loader, 
            criterion, 
            optimizer, 
            device, 
            config.num_epochs,
            config
        )
        
        print("\n正在保存训练指标...")
        plot_training_metrics(metrics, config)
        
        print("\n训练完成！结果已保存：")
        print(f"训练指标曲线: {config.accuracy_plot}")
        print(f"训练日志: {config.metrics_log}")
        print(f"最佳验证集准确率: {metrics['best_val_acc']:.2f}%")
        print(f"所有训练相关文件已保存到: {config.train_dir}")
        
    except FileNotFoundError as e:
        print(f"文件未找到: {str(e)}")
    except PermissionError as e:
        print(f"权限错误: {str(e)}")
    except RuntimeError as e:
        print(f"运行时错误: {str(e)}")
    except Exception as e:
        print(f"发生未预期的错误: {str(e)}")
        if config:
            # 保存错误日志
            try:
                with open(config.error_log, 'a') as f:
                    f.write(f"\n{'-'*50}\n")
                    f.write(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"错误: {str(e)}\n")
                print(f"错误日志已保存到: {config.error_log}")
            except:
                print("无法保存错误日志")

if __name__ == "__main__":
    main() 