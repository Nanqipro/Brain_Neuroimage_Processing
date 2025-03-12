# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import seaborn as sns
# from torch.utils.data import Dataset, DataLoader
# import warnings
# import os
# import datetime
# from analysis_config import AnalysisConfig
# import re
# import copy
# warnings.filterwarnings('ignore')

# # Set random seed
# def set_random_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)

# # 数据加载和预处理类
# class NeuronDataProcessor:
#     def __init__(self, config):
#         """
#         初始化神经元数据处理器
#         参数:
#             config: 配置对象,包含数据文件路径等参数
#         """
#         self.config = config
#         try:
#             self.data = pd.read_excel(config.data_file)
#             print(f"成功加载数据文件: {config.data_file}")
#             print(f"数据形状: {self.data.shape}")
#         except Exception as e:
#             print(f"加载数据文件时出错: {str(e)}")
#             # 创建一个空的DataFrame以避免后续错误
#             self.data = pd.DataFrame()
            
#         self.scaler = StandardScaler()  # 用于数据标准化
#         self.label_encoder = LabelEncoder()  # 用于行为标签编码
        
#     def preprocess_data(self):
#         """
#         数据预处理函数：
#         1. 提取神经元数据
#         2. 处理缺失值
#         3. 标准化数据
#         4. 编码行为标签
#         """
#         if self.data.empty:
#             raise ValueError("数据为空，无法进行预处理")
            
#         # 检查数据列，自动识别神经元列
#         neuron_pattern = re.compile(r'^n\d+$')
#         neuron_cols = [col for col in self.data.columns if neuron_pattern.match(col)]
        
#         if not neuron_cols:
#             # 尝试查找其他可能的神经元列名格式
#             neuron_pattern = re.compile(r'^neuron\d+$', re.IGNORECASE)
#             neuron_cols = [col for col in self.data.columns if neuron_pattern.match(col)]
            
#         if not neuron_cols:
#             # 如果仍然找不到，尝试使用数字列作为神经元数据
#             numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
#             # 排除可能的非神经元数值列
#             exclude_patterns = ['id', 'index', 'time', 'label', 'cluster']
#             neuron_cols = [col for col in numeric_cols if not any(pattern in str(col).lower() for pattern in exclude_patterns)]
            
#         if not neuron_cols:
#             raise ValueError("无法识别神经元数据列，请检查数据格式")
            
#         # 按列名排序，确保顺序一致
#         neuron_cols.sort()
        
#         # 检查可用列
#         available_cols = [col for col in neuron_cols if col in self.data.columns]
#         print(f"Total neurons: {len(neuron_cols)}")
#         print(f"Available neurons: {len(available_cols)}")
        
#         if len(neuron_cols) != len(available_cols):
#             missing_cols = set(neuron_cols) - set(available_cols)
#             print(f"Missing neurons: {missing_cols}")
        
#         # 保存可用的神经元列名列表，以便其他方法使用
#         self.available_neuron_cols = available_cols
        
#         # 获取可用神经元数据
#         try:
#             X = self.data[available_cols].values
#         except Exception as e:
#             print(f"提取神经元数据时出错: {str(e)}")
#             # 尝试一列一列地提取，跳过有问题的列
#             valid_cols = []
#             for col in available_cols:
#                 try:
#                     _ = self.data[col].values
#                     valid_cols.append(col)
#                 except:
#                     print(f"列 {col} 提取失败，将被跳过")
            
#             if not valid_cols:
#                 raise ValueError("没有可用的神经元数据列")
                
#             X = self.data[valid_cols].values
#             self.available_neuron_cols = valid_cols
        
#         # 处理缺失值
#         if np.isnan(X).any():
#             print("发现缺失值，使用列均值填充")
#             # 使用每列的均值填充缺失值
#             X = np.nan_to_num(X, nan=np.nanmean(X))
        
#         # 检查是否有无效值(inf)
#         if not np.isfinite(X).all():
#             print("发现无限值，将替换为有限值")
#             X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
#         # 标准化神经元数据
#         try:
#             X_scaled = self.scaler.fit_transform(X)
#         except Exception as e:
#             print(f"标准化数据时出错: {str(e)}")
#             # 尝试手动标准化
#             means = np.nanmean(X, axis=0)
#             stds = np.nanstd(X, axis=0)
#             stds[stds == 0] = 1.0  # 避免除以零
#             X_scaled = (X - means) / stds
        
#         # 编码行为标签
#         behavior_col = None
#         for col_name in ['behavior', 'Behavior', 'label', 'Label', 'class', 'Class']:
#             if col_name in self.data.columns:
#                 behavior_col = col_name
#                 break
                
#         if behavior_col is None:
#             raise ValueError("找不到行为标签列，请确保数据中包含'behavior'或'label'列")
            
#         # 处理缺失的行为标签
#         behavior_data = self.data[behavior_col].fillna('unknown')
        
#         # 确保行为标签是字符串类型
#         behavior_data = behavior_data.astype(str)
        
#         try:
#             y = self.label_encoder.fit_transform(behavior_data)
#         except Exception as e:
#             print(f"编码行为标签时出错: {str(e)}")
#             # 手动编码
#             unique_labels = np.unique(behavior_data)
#             label_map = {label: i for i, label in enumerate(unique_labels)}
#             y = np.array([label_map[label] for label in behavior_data])
#             self.label_encoder.classes_ = unique_labels
        
#         # 打印标签编码信息
#         label_mapping = dict(zip(self.label_encoder.classes_, 
#                                self.label_encoder.transform(self.label_encoder.classes_)))
#         print("\nBehavior label mapping:")
#         for label, code in label_mapping.items():
#             count = sum(y == code)
#             print(f"{label}: {code} (Count: {count})")
        
#         return X_scaled, y

#     def apply_kmeans(self, X):
#         # 应用K-means聚类
#         print(f"\n使用 {self.config.n_clusters} 个聚类进行K-means聚类")
#         kmeans = KMeans(n_clusters=self.config.n_clusters, 
#                        random_state=self.config.random_seed,
#                        n_init=10)
#         cluster_labels = kmeans.fit_predict(X)
        
#         # 打印聚类分布情况
#         unique_labels, counts = np.unique(cluster_labels, return_counts=True)
#         print("\n聚类分布:")
#         for label, count in zip(unique_labels, counts):
#             print(f"聚类 {label}: {count} 个样本")
        
#         return kmeans, cluster_labels

# # LSTM dataset class
# class NeuronDataset(Dataset):
#     def __init__(self, X, y, sequence_length):
#         """
#         参数：
#         X: 神经元活动数据
#         y: 行为标签
#         sequence_length: 序列长度
#         """
#         self.X = torch.FloatTensor(X)
#         self.y = torch.LongTensor(y)
#         self.sequence_length = sequence_length
        
#     def __len__(self):
#         return len(self.X) - self.sequence_length
        
#     def __getitem__(self, idx):
#         return (self.X[idx:idx+self.sequence_length], 
#                 self.y[idx+self.sequence_length-1])

# class NeuronAutoencoder(nn.Module):
#     """神经元自编码器
#     用于提取神经元活动的潜在特征
#     """
#     def __init__(self, input_size, hidden_size, latent_dim):
#         super(NeuronAutoencoder, self).__init__()
#         self.input_size = input_size
#         self.latent_dim = latent_dim
        
#         # 编码器
#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_size),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_size // 2),
#             nn.Linear(hidden_size // 2, latent_dim)
#         )
        
#         # 解码器
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, hidden_size // 2),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_size // 2),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_size // 2, hidden_size),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_size),
#             nn.Linear(hidden_size, input_size),
#             nn.Tanh()  # 使用Tanh确保输出在[-1,1]范围内
#         )
        
#     def forward(self, x):
#         # 检查输入维度是否与模型匹配
#         if x.size(-1) != self.input_size:
#             # 处理输入维度不匹配的情况
#             if x.size(-1) < self.input_size:
#                 # 输入维度小于模型期望，填充零
#                 padding_size = self.input_size - x.size(-1)
#                 padding = torch.zeros(*x.shape[:-1], padding_size, device=x.device)
#                 x = torch.cat([x, padding], dim=-1)
#             else:
#                 # 输入维度大于模型期望，截断
#                 x = x[..., :self.input_size]
        
#         try:
#             # 处理3D输入 (batch_size, seq_len, features)
#             original_shape = x.shape
#             if len(original_shape) == 3:
#                 # 将3D张量重塑为2D
#                 batch_size, seq_len, features = original_shape
#                 x_reshaped = x.reshape(-1, features)
                
#                 # 通过编码器和解码器
#                 encoded = self.encoder(x_reshaped)
#                 decoded = self.decoder(encoded)
                
#                 # 重塑回原始形状
#                 encoded = encoded.reshape(batch_size, seq_len, self.latent_dim)
#                 decoded = decoded.reshape(original_shape)
                
#                 return encoded, decoded
#             else:
#                 # 处理2D输入 (对单个样本处理)
#                 encoded = self.encoder(x)
#                 decoded = self.decoder(encoded)
#                 return encoded, decoded
                
#         except RuntimeError as e:
#             print(f"自编码器前向传播错误: {str(e)}")
#             print(f"输入形状: {x.shape}, 期望输入大小: {self.input_size}")
#             # 返回零张量
#             return (
#                 torch.zeros(*x.shape[:-1], self.latent_dim, device=x.device),
#                 torch.zeros_like(x)
#             )

# class MultiHeadAttention(nn.Module):
#     """多头注意力机制
#     用于捕捉神经元之间的关联关系
#     """
#     def __init__(self, input_dim, num_heads, dropout=0.1):
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = input_dim // num_heads
#         assert self.head_dim * num_heads == input_dim, "input_dim必须能被num_heads整除"
        
#         self.query = nn.Linear(input_dim, input_dim)
#         self.key = nn.Linear(input_dim, input_dim)
#         self.value = nn.Linear(input_dim, input_dim)
        
#         self.dropout = nn.Dropout(dropout)
#         self.output_linear = nn.Linear(input_dim, input_dim)
        
#     def forward(self, x, mask=None):
#         batch_size = x.size(0)
        
#         # 线性变换并分头
#         query = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         key = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         value = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
#         # 计算注意力分数
#         scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
        
#         attention_weights = torch.softmax(scores, dim=-1)
#         attention_weights = self.dropout(attention_weights)
        
#         # 应用注意力权重
#         context = torch.matmul(attention_weights, value)
        
#         # 重组并线性变换
#         context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
#         output = self.output_linear(context)
        
#         return output, attention_weights

# class TemporalAttention(nn.Module):
#     """时间注意力机制
#     用于捕捉时间序列中的重要模式
#     """
#     def __init__(self, hidden_size):
#         super(TemporalAttention, self).__init__()
#         self.attention = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.Tanh(),
#             nn.Linear(hidden_size // 2, 1)
#         )
        
#     def forward(self, hidden_states):
#         attention_weights = self.attention(hidden_states)
#         attention_weights = torch.softmax(attention_weights, dim=1)
#         attended = torch.sum(hidden_states * attention_weights, dim=1)
#         return attended, attention_weights

# class EnhancedNeuronLSTM(nn.Module):
#     """增强版神经元LSTM模型
#     整合了自编码器、双向LSTM和多层注意力机制
#     """
#     def __init__(self, input_size, hidden_size, num_layers, num_classes, 
#                  latent_dim=32, num_heads=4, dropout=0.2):
#         super(EnhancedNeuronLSTM, self).__init__()
        
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.num_classes = num_classes
#         self.latent_dim = latent_dim
        
#         # 确保隐藏层大小是多头注意力头数的倍数
#         if hidden_size % num_heads != 0:
#             adjusted_hidden_size = (hidden_size // num_heads) * num_heads
#             print(f"警告: 隐藏层大小({hidden_size})不是头数({num_heads})的倍数，调整为{adjusted_hidden_size}")
#             hidden_size = adjusted_hidden_size
#             self.hidden_size = hidden_size
        
#         # 自编码器用于特征提取
#         self.autoencoder = NeuronAutoencoder(input_size, hidden_size, latent_dim)
        
#         # 双向LSTM
#         self.lstm = nn.LSTM(
#             input_size=latent_dim,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=True,
#             dropout=dropout if num_layers > 1 else 0
#         )
        
#         # 多头注意力机制
#         self.multihead_attention = MultiHeadAttention(hidden_size * 2, num_heads)
        
#         # 时间注意力机制
#         self.temporal_attention = TemporalAttention(hidden_size * 2)
        
#         # 批标准化
#         self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        
#         # 分类器
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_size * 2, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.BatchNorm1d(hidden_size),
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.BatchNorm1d(hidden_size // 2),
#             nn.Linear(hidden_size // 2, num_classes)
#         )
        
#         # 记录模型配置
#         self.config = {
#             'input_size': input_size,
#             'hidden_size': hidden_size,
#             'num_layers': num_layers,
#             'num_classes': num_classes,
#             'latent_dim': latent_dim,
#             'num_heads': num_heads,
#             'dropout': dropout
#         }
        
#     def forward(self, x):
#         batch_size = x.size(0)
        
#         # 检查输入维度是否与模型匹配
#         if x.size(-1) != self.input_size:
#             # 处理输入维度不匹配的情况
#             if x.size(-1) < self.input_size:
#                 # 输入维度小于模型期望，填充零
#                 padding = torch.zeros(batch_size, x.size(1), self.input_size - x.size(-1), device=x.device)
#                 x = torch.cat([x, padding], dim=-1)
#             else:
#                 # 输入维度大于模型期望，截断
#                 x = x[..., :self.input_size]
            
#         try:
#             # 通过自编码器提取特征
#             encoded, decoded = self.autoencoder(x)
            
#             # 确保encoded是3D张量 (batch_size, sequence_length, latent_dim)
#             # 不再需要unsqueeze，因为autoencoder现在已返回正确形状
            
#             # 通过LSTM处理序列
#             lstm_out, _ = self.lstm(encoded)
            
#             # 应用多头注意力
#             attended, attention_weights = self.multihead_attention(lstm_out, None)
            
#             # 应用时间注意力
#             temporal_context, temporal_weights = self.temporal_attention(attended)
            
#             # 批标准化 - 注意temporal_context是2D的
#             normalized = self.batch_norm(temporal_context)
            
#             # 分类
#             output = self.classifier(normalized)
            
#             return output, attention_weights, temporal_weights
            
#         except RuntimeError as e:
#             # 处理可能的运行时错误
#             if "size mismatch" in str(e) or "dimension" in str(e):
#                 print(f"前向传播中的维度错误: {str(e)}")
#                 print(f"输入形状: {x.shape}, 模型输入大小: {self.input_size}")
#                 # 返回一个全零的输出张量和空的注意力权重
#                 return (
#                     torch.zeros(batch_size, self.num_classes, device=x.device),
#                     torch.zeros(batch_size, 1, 1, device=x.device),  # 空的attention_weights
#                     torch.zeros(batch_size, 1, device=x.device)      # 空的temporal_weights
#                 )
#             else:
#                 # 重新抛出其他类型的错误
#                 raise e
    
#     def get_config(self):
#         """
#         获取模型配置
#         """
#         return self.config
    
#     def summary(self):
#         """
#         打印模型摘要信息
#         """
#         print("\n模型摘要:")
#         print(f"输入大小: {self.input_size}")
#         print(f"隐藏层大小: {self.hidden_size}")
#         print(f"LSTM层数: {self.num_layers}")
#         print(f"类别数: {self.num_classes}")
#         print(f"潜在维度: {self.latent_dim}")
        
#         # 计算参数数量
#         total_params = sum(p.numel() for p in self.parameters())
#         trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
#         print(f"总参数数量: {total_params:,}")
#         print(f"可训练参数数量: {trainable_params:,}")
        
#         return {
#             'total_params': total_params,
#             'trainable_params': trainable_params
#         }

# # Training function
# def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, config, early_stopping_enabled=True):
#     """
#     训练增强型神经元LSTM模型
    
#     参数:
#         model: 待训练的模型
#         train_loader: 训练数据加载器
#         val_loader: 验证数据加载器
#         criterion: 损失函数
#         optimizer: 优化器
#         device: 计算设备
#         num_epochs: 训练轮数
#         config: 配置对象
#         early_stopping_enabled: 是否启用早停机制，默认为True
    
#     返回:
#         model: 训练后的模型
#         metrics_dict: 包含训练指标的字典
#     """
#     model.train()
#     train_losses = []
#     train_accuracies = []
#     val_losses = []
#     val_accuracies = []
#     reconstruction_losses = []
#     learning_rates = []
#     best_val_acc = 0.0
    
#     # 保存最后一个批次的注意力权重用于可视化
#     last_attention_weights = None
#     last_temporal_weights = None
    
#     # 添加重构损失
#     reconstruction_criterion = nn.MSELoss()
    
#     # 创建学习率调度器
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=5, verbose=True
#     )
    
#     # 确保训练指标日志目录存在
#     os.makedirs(os.path.dirname(config.metrics_log), exist_ok=True)
    
#     # 创建或截断训练指标日志文件
#     with open(config.metrics_log, 'w') as f:
#         f.write('epoch,train_loss,train_acc,val_loss,val_acc,reconstruction_loss\n')
    
#     # 早停相关参数
#     patience = config.analysis_params.get('early_stopping_patience', 20)
#     best_val_loss = float('inf')
#     counter = 0
#     best_model_state = None
    
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         total_recon_loss = 0
#         correct = 0
#         total = 0
        
#         # 训练循环
#         for batch_X, batch_y in train_loader:
#             batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
#             # 清零梯度
#             optimizer.zero_grad()
            
#             # 前向传播
#             outputs, attention_weights, temporal_weights = model(batch_X)
            
#             # 保存最后一个批次的注意力权重
#             if attention_weights is not None:
#                 last_attention_weights = attention_weights
#             if temporal_weights is not None:
#                 last_temporal_weights = temporal_weights
            
#             # 计算分类损失
#             loss = criterion(outputs, batch_y)
            
#             # 计算重构损失
#             _, decoded = model.autoencoder(batch_X.view(-1, batch_X.size(-1)))
#             reconstruction_loss = reconstruction_criterion(decoded, batch_X.view(-1, batch_X.size(-1)))
            
#             # 总损失
#             recon_weight = config.analysis_params.get('reconstruction_loss_weight', 0.1)
#             total_batch_loss = loss + recon_weight * reconstruction_loss
            
#             # 反向传播
#             total_batch_loss.backward()
            
#             # 梯度裁剪
#             max_norm = config.analysis_params.get('gradient_clip_norm', 1.0)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
#             # 更新参数
#             optimizer.step()
            
#             # 统计
#             total_loss += loss.item() * batch_X.size(0)
#             total_recon_loss += reconstruction_loss.item() * batch_X.size(0)
#             _, predicted = torch.max(outputs, 1)
#             correct += (predicted == batch_y).sum().item()
#             total += batch_y.size(0)
        
#         # 计算平均损失和准确率
#         avg_train_loss = total_loss / total
#         avg_recon_loss = total_recon_loss / total
#         train_accuracy = 100 * correct / total
        
#         # 记录训练指标
#         train_losses.append(avg_train_loss)
#         train_accuracies.append(train_accuracy)
#         reconstruction_losses.append(avg_recon_loss)
        
#         # 验证
#         model.eval()
#         val_loss = 0
#         val_correct = 0
#         val_total = 0
        
#         with torch.no_grad():
#             for val_X, val_y in val_loader:
#                 val_X, val_y = val_X.to(device), val_y.to(device)
#                 val_outputs, _, _ = model(val_X)
#                 batch_loss = criterion(val_outputs, val_y)
#                 val_loss += batch_loss.item() * val_X.size(0)
                
#                 _, val_predicted = torch.max(val_outputs, 1)
#                 val_correct += (val_predicted == val_y).sum().item()
#                 val_total += val_y.size(0)
        
#         # 计算验证损失和准确率
#         avg_val_loss = val_loss / val_total
#         val_accuracy = 100 * val_correct / val_total
        
#         # 记录验证指标
#         val_losses.append(avg_val_loss)
#         val_accuracies.append(val_accuracy)
        
#         # 记录学习率
#         current_lr = optimizer.param_groups[0]['lr']
#         learning_rates.append(current_lr)
        
#         # 打印训练信息
#         if (epoch+1) % 10 == 0:
#             print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, LR: {current_lr:.6f}")
        
#         # 记录指标到日志文件
#         with open(config.metrics_log, 'a') as f:
#             f.write(f'{epoch+1},{avg_train_loss:.4f},{train_accuracy:.2f},{avg_val_loss:.4f},{val_accuracy:.2f},{avg_recon_loss:.4f}\n')
        
#         # 早停检查
#         if early_stopping_enabled:
#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 best_model_state = copy.deepcopy(model.state_dict())
#                 counter = 0
#             else:
#                 counter += 1
#                 if counter >= patience:
#                     print(f"Early stopping at epoch {epoch+1}, best val loss: {best_val_loss:.4f}")
#                     break
#         else:
#             # 如果禁用早停，仍然保存最佳模型
#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 best_model_state = copy.deepcopy(model.state_dict())
        
#         # 调整学习率
#         scheduler.step(avg_val_loss)
    
#     # 如果禁用早停且完成所有epoch，打印最终结果
#     if not early_stopping_enabled:
#         print(f"Training completed for all {num_epochs} epochs, best val loss: {best_val_loss:.4f}")
    
#     # 恢复最佳模型
#     if best_model_state is not None:
#         model.load_state_dict(best_model_state)
#         print(f"已恢复最佳模型 (验证损失: {best_val_loss:.4f})")
    
#     # 保存最佳模型到文件
#     try:
#         os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
#         torch.save({
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'best_val_loss': best_val_loss,
#             'attention_weights': last_attention_weights.detach().cpu().numpy() if last_attention_weights is not None else None,
#             'temporal_weights': last_temporal_weights.detach().cpu().numpy() if last_temporal_weights is not None else None
#         }, config.model_path)
#         print(f"模型已保存到: {config.model_path}")
#     except Exception as e:
#         print(f"保存模型时出错: {str(e)}")
#         with open(config.error_log, 'a') as f:
#             f.write(f"保存模型失败 - {str(e)}\n")
    
#     # 返回训练结果
#     metrics_dict = {
#         'train_losses': train_losses,
#         'train_accuracies': train_accuracies,
#         'val_losses': val_losses,
#         'val_accuracies': val_accuracies,
#         'best_val_acc': max(val_accuracies) if val_accuracies else 0,
#         'reconstruction_losses': reconstruction_losses,
#         'learning_rates': learning_rates,
#         'attention_weights': last_attention_weights.detach().cpu().numpy() if last_attention_weights is not None else None,
#         'temporal_weights': last_temporal_weights.detach().cpu().numpy() if last_temporal_weights is not None else None
#     }
    
#     return model, metrics_dict

# def plot_training_metrics(metrics, config):
#     """
#     绘制训练和验证指标的变化曲线
#     包括损失值和准确率
#     """
#     plt.figure(figsize=(15, 5))
    
#     # 绘制损失曲线
#     plt.subplot(1, 3, 1)
#     plt.plot(metrics['train_losses'], label='Training Loss', linewidth=2)
#     plt.plot(metrics['val_losses'], label='Validation Loss', linewidth=2)
#     plt.title('Loss Curves')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
    
#     # 绘制准确率曲线
#     plt.subplot(1, 3, 2)
#     plt.plot(metrics['train_accuracies'], label='Training Accuracy', linewidth=2)
#     plt.plot(metrics['val_accuracies'], label='Validation Accuracy', linewidth=2)
#     plt.title('Accuracy Curves')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy (%)')
#     plt.legend()
#     plt.grid(True)
    
#     # 绘制重构损失曲线
#     plt.subplot(1, 3, 3)
#     plt.plot(metrics['reconstruction_losses'], label='Reconstruction Loss', linewidth=2)
#     plt.title('Reconstruction Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.savefig(config.accuracy_plot, dpi=300, bbox_inches='tight')
#     plt.close()

# def main():
#     """
#     主函数：
#     1. 初始化配置
#     2. 数据预处理
#     3. 应用K-means聚类
#     4. 训练LSTM模型
#     5. 保存结果和可视化
#     """
#     config = None
#     try:
#         # 初始化配置
#         config = AnalysisConfig()
        
#         # 验证和创建目录（validate_paths 现在会自动调用 setup_directories）
#         print("正在验证并创建目录结构...")
#         config.validate_paths()
        
#         print("正在设置随机种子...")
#         set_random_seed(config.random_seed)
        
#         print("正在进行数据预处理...")
#         processor = NeuronDataProcessor(config)
#         X_scaled, y = processor.preprocess_data()
        
#         print("正在应用K-means聚类...")
#         kmeans, cluster_labels = processor.apply_kmeans(X_scaled)
        
#         print("正在准备训练数据...")
#         X_with_clusters = np.column_stack((X_scaled, cluster_labels))
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_with_clusters, y, test_size=config.test_size, random_state=config.random_seed
#         )
        
#         # 创建数据加载器
#         train_dataset = NeuronDataset(X_train, y_train, config.sequence_length)
#         val_dataset = NeuronDataset(X_test, y_test, config.sequence_length)
        
#         train_loader = DataLoader(
#             train_dataset, 
#             batch_size=config.batch_size, 
#             shuffle=True
#         )
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=config.batch_size,
#             shuffle=False
#         )
        
#         print("正在初始化模型...")
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"使用设备: {device}")
        
#         input_size = X_with_clusters.shape[1]
#         num_classes = len(np.unique(y))
        
#         # 打印模型输入输出尺寸，帮助调试
#         print(f"模型输入尺寸: {input_size}")
#         print(f"行为类别数: {num_classes}")
#         print(f"序列长度: {config.sequence_length}")
#         print(f"隐藏层大小: {config.hidden_size}")
        
#         model = EnhancedNeuronLSTM(
#             input_size, 
#             config.hidden_size, 
#             config.num_layers, 
#             num_classes
#         ).to(device)
        
#         model.summary()  # 打印模型摘要
        
#         criterion = nn.CrossEntropyLoss()
#         optimizer = torch.optim.AdamW(
#             model.parameters(),
#             lr=config.learning_rate,
#             weight_decay=config.analysis_params['weight_decay']
#         )
        
#         print("\n开始训练模型...")
#         # 从配置中获取早停设置
#         early_stopping_enabled = config.analysis_params.get('early_stopping_enabled', True)
        
#         # 训练模型
#         trained_model, metrics = train_model(
#             model=model,
#             train_loader=train_loader,
#             val_loader=val_loader,
#             criterion=criterion,
#             optimizer=optimizer,
#             device=device,
#             num_epochs=config.num_epochs,
#             config=config,
#             early_stopping_enabled=early_stopping_enabled
#         )
        
#         # 绘制训练指标
#         print("\n正在绘制训练指标...")
#         plot_training_metrics(metrics, config)
        
#         print("\n训练完成！结果已保存：")
#         print(f"训练指标曲线: {config.accuracy_plot}")
#         print(f"训练日志: {config.metrics_log}")
#         print(f"最佳验证集准确率: {metrics['best_val_acc']:.2f}%")
#         print(f"所有训练相关文件已保存到: {config.train_dir}")
        
#     except FileNotFoundError as e:
#         print(f"文件未找到: {str(e)}")
#     except PermissionError as e:
#         print(f"权限错误: {str(e)}")
#     except RuntimeError as e:
#         print(f"运行时错误: {str(e)}")
#     except Exception as e:
#         print(f"发生未预期的错误: {str(e)}")
#         if config:
#             # 保存错误日志
#             try:
#                 with open(config.error_log, 'a') as f:
#                     f.write(f"\n{'-'*50}\n")
#                     f.write(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#                     f.write(f"错误: {str(e)}\n")
#                 print(f"错误日志已保存到: {config.error_log}")
#             except:
#                 print("无法保存错误日志")

# if __name__ == "__main__":
#     main() 

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
from torch.utils.data import DataLoader
import warnings
import os
import datetime
from analysis_config import AnalysisConfig
import re
import copy
warnings.filterwarnings('ignore')

# 导入从neuron_lstm.py迁移的类和函数
from neuron_lstm import (
    set_random_seed, 
    NeuronDataset, 
    NeuronAutoencoder, 
    MultiHeadAttention, 
    TemporalAttention, 
    EnhancedNeuronLSTM, 
    train_model, 
    plot_training_metrics
)

# 数据加载和预处理类
class NeuronDataProcessor:
    def __init__(self, config):
        """
        初始化神经元数据处理器
        参数:
            config: 配置对象,包含数据文件路径等参数
        """
        self.config = config
        try:
            self.data = pd.read_excel(config.data_file)
            print(f"成功加载数据文件: {config.data_file}")
            print(f"数据形状: {self.data.shape}")
        except Exception as e:
            print(f"加载数据文件时出错: {str(e)}")
            # 创建一个空的DataFrame以避免后续错误
            self.data = pd.DataFrame()
            
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
        if self.data.empty:
            raise ValueError("数据为空，无法进行预处理")
            
        # 检查数据列，自动识别神经元列
        neuron_pattern = re.compile(r'^n\d+$')
        neuron_cols = [col for col in self.data.columns if neuron_pattern.match(col)]
        
        if not neuron_cols:
            # 尝试查找其他可能的神经元列名格式
            neuron_pattern = re.compile(r'^neuron\d+$', re.IGNORECASE)
            neuron_cols = [col for col in self.data.columns if neuron_pattern.match(col)]
            
        if not neuron_cols:
            # 如果仍然找不到，尝试使用数字列作为神经元数据
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            # 排除可能的非神经元数值列
            exclude_patterns = ['id', 'index', 'time', 'label', 'cluster']
            neuron_cols = [col for col in numeric_cols if not any(pattern in str(col).lower() for pattern in exclude_patterns)]
            
        if not neuron_cols:
            raise ValueError("无法识别神经元数据列，请检查数据格式")
            
        # 按列名排序，确保顺序一致
        neuron_cols.sort()
        
        # 检查可用列
        available_cols = [col for col in neuron_cols if col in self.data.columns]
        print(f"Total neurons: {len(neuron_cols)}")
        print(f"Available neurons: {len(available_cols)}")
        
        if len(neuron_cols) != len(available_cols):
            missing_cols = set(neuron_cols) - set(available_cols)
            print(f"Missing neurons: {missing_cols}")
        
        # 保存可用的神经元列名列表，以便其他方法使用
        self.available_neuron_cols = available_cols
        
        # 获取可用神经元数据
        try:
            X = self.data[available_cols].values
        except Exception as e:
            print(f"提取神经元数据时出错: {str(e)}")
            # 尝试一列一列地提取，跳过有问题的列
            valid_cols = []
            for col in available_cols:
                try:
                    _ = self.data[col].values
                    valid_cols.append(col)
                except:
                    print(f"列 {col} 提取失败，将被跳过")
            
            if not valid_cols:
                raise ValueError("没有可用的神经元数据列")
                
            X = self.data[valid_cols].values
            self.available_neuron_cols = valid_cols
        
        # 处理缺失值
        if np.isnan(X).any():
            print("发现缺失值，使用列均值填充")
            # 使用每列的均值填充缺失值
            X = np.nan_to_num(X, nan=np.nanmean(X))
        
        # 检查是否有无效值(inf)
        if not np.isfinite(X).all():
            print("发现无限值，将替换为有限值")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 标准化神经元数据
        try:
            X_scaled = self.scaler.fit_transform(X)
        except Exception as e:
            print(f"标准化数据时出错: {str(e)}")
            # 尝试手动标准化
            means = np.nanmean(X, axis=0)
            stds = np.nanstd(X, axis=0)
            stds[stds == 0] = 1.0  # 避免除以零
            X_scaled = (X - means) / stds
        
        # 编码行为标签
        behavior_col = None
        for col_name in ['behavior', 'Behavior', 'label', 'Label', 'class', 'Class']:
            if col_name in self.data.columns:
                behavior_col = col_name
                break
                
        if behavior_col is None:
            raise ValueError("找不到行为标签列，请确保数据中包含'behavior'或'label'列")
            
        # 处理缺失的行为标签
        behavior_data = self.data[behavior_col].fillna('unknown')
        
        # 确保行为标签是字符串类型
        behavior_data = behavior_data.astype(str)
        
        try:
            y = self.label_encoder.fit_transform(behavior_data)
        except Exception as e:
            print(f"编码行为标签时出错: {str(e)}")
            # 手动编码
            unique_labels = np.unique(behavior_data)
            label_map = {label: i for i, label in enumerate(unique_labels)}
            y = np.array([label_map[label] for label in behavior_data])
            self.label_encoder.classes_ = unique_labels
        
        # 打印标签编码信息
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

def main():
    """
    主函数：
    1. 初始化配置
    2. 数据预处理
    3. 聚类分析
    4. 序列数据集创建
    5. 模型训练
    6. 结果可视化
    """
    # 初始化配置
    config = AnalysisConfig()
    set_random_seed(config.random_seed)
    
    # 创建或清空错误日志
    os.makedirs(os.path.dirname(config.error_log), exist_ok=True)
    with open(config.error_log, 'w') as f:
        f.write(f"错误日志创建于 {datetime.datetime.now()}\n")
        
    try:
        # 数据预处理
        processor = NeuronDataProcessor(config)
        X_scaled, y = processor.preprocess_data()
        
        # 聚类分析
        kmeans, cluster_labels = processor.apply_kmeans(X_scaled)
        
        # 创建序列数据集
        print(f"\n创建序列数据集 (序列长度: {config.sequence_length})")
        dataset = NeuronDataset(X_scaled, y, config.sequence_length)
        
        # 分割训练和验证集
        dataset_size = len(dataset)
        train_size = int(dataset_size * 0.8)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        
        print(f"训练集大小: {train_size}")
        print(f"验证集大小: {val_size}")
        
        # 确定设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 初始化模型
        num_classes = len(set(y))
        model = EnhancedNeuronLSTM(
            input_size=X_scaled.shape[1],
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_classes=num_classes,
            latent_dim=config.latent_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        ).to(device)
        
        model.summary()
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        # 训练模型
        print(f"\n开始训练模型 (epochs: {config.num_epochs})")
        model, metrics = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, 
            config.num_epochs, config, early_stopping_enabled=config.early_stopping
        )
        
        # 可视化训练结果
        plot_training_metrics(metrics, config)
        
        print(f"\n训练完成! 最佳验证准确率: {metrics['best_val_acc']:.2f}%")
        
    except Exception as e:
        print(f"主函数执行错误: {str(e)}")
        with open(config.error_log, 'a') as f:
            f.write(f"主函数执行错误 - {str(e)}\n")
        raise e

if __name__ == "__main__":
    main() 