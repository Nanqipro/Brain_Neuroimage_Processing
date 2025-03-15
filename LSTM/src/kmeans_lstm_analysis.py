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
        
        # 根据include_cd1_behavior配置决定是否纳入CD1行为标签
        if not self.config.include_cd1_behavior:
            print("\n配置设置为不纳入CD1行为标签")
            # 创建新的DataFrame以保存非CD1数据
            mask = (behavior_data != 'CD1')
            original_shape = behavior_data.shape[0]
            behavior_data = behavior_data[mask]
            self.data = self.data[mask].reset_index(drop=True)
            X = X[mask]
            X_scaled = X_scaled[mask]
            print(f"\n已排除CD1行为数据: 从 {original_shape} 条数据减少到 {behavior_data.shape[0]} 条")
        
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