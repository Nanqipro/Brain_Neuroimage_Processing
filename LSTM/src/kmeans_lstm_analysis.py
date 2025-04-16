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
from typing import Tuple, List, Dict, Any, Optional, Union
warnings.filterwarnings('ignore')

# 导入从neuron_lstm.py迁移的类和函数
from neuron_lstm import (
    set_random_seed, 
    NeuronDataset, 
    # NeuronAutoencoder, 
    # MultiHeadAttention, 
    # TemporalAttention, 
    EnhancedNeuronLSTM, 
    train_model, 
    plot_training_metrics
)

# 数据加载和预处理类
class NeuronDataProcessor:
    """
    神经元数据处理器类：负责加载、预处理神经元活动数据和行为标签
    
    主要功能:
    1. 加载神经元数据文件
    2. 识别和预处理神经元活动数据
    3. 处理行为标签
    4. 应用标准化和聚类分析
    
    参数
    ----------
    config : AnalysisConfig
        配置对象，包含数据文件路径等参数
    """
    def __init__(self, config: AnalysisConfig) -> None:
        """
        初始化神经元数据处理器
        
        参数
        ----------
        config : AnalysisConfig
            配置对象，包含数据文件路径等参数
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
        self.available_neuron_cols: List[str] = []  # 存储可用的神经元列名
        
    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据预处理函数
        
        执行步骤:
        1. 提取神经元数据
        2. 处理缺失值
        3. 标准化数据
        4. 编码行为标签
        
        返回
        ----------
        X_scaled : np.ndarray
            标准化后的神经元数据
        y : np.ndarray
            编码后的行为标签
            
        异常
        ----------
        ValueError
            当数据为空或无法识别神经元数据列时抛出
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

    def apply_kmeans(self, X: np.ndarray) -> Tuple[KMeans, np.ndarray]:
        """
        应用K-means聚类分析
        
        参数
        ----------
        X : np.ndarray
            用于聚类的数据
            
        返回
        ----------
        kmeans : KMeans
            训练好的KMeans模型
        cluster_labels : np.ndarray
            聚类标签数组
        """
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

def time_aware_validation_split(X_scaled, y, config):
    """
    时间感知的验证集和测试集划分，保持时间序列的连续性
    
    参数:
        X_scaled (numpy.ndarray): 标准化后的神经元数据
        y (numpy.ndarray): 编码后的行为标签
        config (AnalysisConfig): 配置对象
    
    返回:
        tuple: 训练、验证和测试数据加载器
    """
    sequence_length = config.sequence_length
    
    # 计算有效样本数 (考虑序列长度)
    n_samples = len(X_scaled) - sequence_length
    
    # 计算测试集和验证集大小
    test_size = int(n_samples * 0.2)  # 20%用于测试
    val_size = int(n_samples * 0.2)   # 20%用于验证
    train_size = n_samples - test_size - val_size  # 剩余60%用于训练
    
    print(f"\n使用时间感知的数据集划分策略:")
    print(f"训练集: {train_size} 样本 (前60%)")
    print(f"验证集: {val_size} 样本 (中间20%)")
    print(f"测试集: {test_size} 样本 (后20%)")
    
    # 使用连续的时间段划分
    # 训练集: 前60%的数据
    # 验证集: 中间20%的数据
    # 测试集: 最后20%的数据
    X_train = X_scaled[:train_size + sequence_length]
    y_train = y[:train_size + sequence_length]
    
    X_val = X_scaled[train_size:train_size + val_size + sequence_length]
    y_val = y[train_size:train_size + val_size + sequence_length]
    
    X_test = X_scaled[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # 创建序列数据集
    train_dataset = NeuronDataset(X_train, y_train, sequence_length)
    val_dataset = NeuronDataset(X_val, y_val, sequence_length)
    test_dataset = NeuronDataset(X_test, y_test, sequence_length)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    return train_loader, val_loader, test_loader

def stratified_data_split(X_scaled, y, config):
    """
    分层抽样的数据集划分，确保各类别样本分布均衡
    
    参数:
        X_scaled (numpy.ndarray): 标准化后的神经元数据
        y (numpy.ndarray): 编码后的行为标签
        config (AnalysisConfig): 配置对象
    
    返回:
        tuple: 训练、验证和测试数据加载器
    """
    from sklearn.model_selection import train_test_split
    
    # 先分出测试集 (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=config.random_seed, stratify=y
    )
    
    # 再将剩余数据分为训练集 (75%) 和验证集 (25%)
    # 这样总体比例为: 训练集 60%, 验证集 20%, 测试集 20%
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=config.random_seed, stratify=y_temp
    )
    
    print(f"\n使用分层抽样的数据集划分策略:")
    print(f"训练集: {len(X_train)} 样本 (60%)")
    print(f"验证集: {len(X_val)} 样本 (20%)")
    print(f"测试集: {len(X_test)} 样本 (20%)")
    
    # 创建序列数据集
    train_dataset = NeuronDataset(X_train, y_train, config.sequence_length)
    val_dataset = NeuronDataset(X_val, y_val, config.sequence_length)
    test_dataset = NeuronDataset(X_test, y_test, config.sequence_length)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    return train_loader, val_loader, test_loader

def filter_minority_classes(X, y, min_samples=2):
    """
    过滤掉样本数量少于指定阈值的类别，并重新编码标签
    
    参数:
        X (numpy.ndarray): 特征数据
        y (numpy.ndarray): 标签数据
        min_samples (int): 每个类别最少样本数量阈值
        
    返回:
        tuple: 过滤后的特征和标签数据
    """
    # 计算每个类别的样本数量
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_counts_dict = dict(zip(unique_classes, class_counts))
    
    # 输出每个类别的样本数量
    print("\n各类别样本数量:")
    for cls, count in class_counts_dict.items():
        print(f"类别 {cls}: {count} 个样本")
    
    # 找出满足最小样本数要求的类别
    valid_classes = [cls for cls, count in class_counts_dict.items() if count >= min_samples]
    
    # 如果有类别被过滤掉，输出提示信息
    filtered_classes = set(unique_classes) - set(valid_classes)
    if filtered_classes:
        print(f"\n过滤掉样本数量少于 {min_samples} 的类别: {filtered_classes}")
    
    # 创建掩码，保留有效类别的样本
    mask = np.isin(y, valid_classes)
    
    # 应用掩码
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    print(f"过滤前: {len(y)} 个样本，过滤后: {len(y_filtered)} 个样本")
    
    # 重要：重新编码标签，确保标签值是从0开始连续的整数
    if len(valid_classes) < len(unique_classes):
        print("重新编码标签以确保连续性...")
        old_to_new = {old_cls: new_idx for new_idx, old_cls in enumerate(sorted(valid_classes))}
        y_remapped = np.array([old_to_new[cls] for cls in y_filtered])
        
        # 输出标签映射信息
        print("标签重映射:")
        for old_cls, new_cls in old_to_new.items():
            count = np.sum(y_filtered == old_cls)
            print(f"原标签 {old_cls} -> 新标签 {new_cls} (样本数: {count})")
        
        return X_filtered, y_remapped
    
    return X_filtered, y_filtered

def handle_class_imbalance(y):
    """
    处理类别不平衡问题，计算样本权重
    
    参数:
        y (numpy.ndarray): 编码后的行为标签
    
    返回:
        torch.Tensor: 类别权重张量
    """
    try:
        # 使用neuron_lstm.py中的函数
        from neuron_lstm import compute_class_weights
        return compute_class_weights(y)
    except Exception as e:
        print(f"计算类别权重时出错: {str(e)}")
        return None

def main() -> None:
    """
    主函数 (改进版)
    
    执行步骤:
    1. 初始化配置
    2. 数据预处理
    3. 聚类分析
    4. 改进的数据集划分 (包括测试集)
    5. 处理类别不平衡
    6. 模型训练与评估
    7. 结果可视化
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
        
        # 过滤掉样本数量过少的类别
        X_filtered, y_filtered = filter_minority_classes(X_scaled, y, min_samples=2)
        
        # 创建改进的序列数据集划分 (包括测试集)
        print(f"\n创建序列数据集 (序列长度: {config.sequence_length})")
        
        # 使用配置选择划分策略
        if config.analysis_params.get('use_time_aware_split', True):
            # 时间感知的划分 - 适合时间序列数据
            train_loader, val_loader, test_loader = time_aware_validation_split(X_filtered, y_filtered, config)
        else:
            # 分层抽样的划分 - 确保类别分布一致
            train_loader, val_loader, test_loader = stratified_data_split(X_filtered, y_filtered, config)
            
        # 确定计算设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 处理类别不平衡问题
        print("\n计算类别权重以处理不平衡数据...")
        class_weights = handle_class_imbalance(y_filtered)
        if class_weights is not None:
            print(f"类别权重: {class_weights}")
            # 将类别权重保存到配置中，传递给后续流程
            config.class_weights = class_weights.to(device)
        else:
            config.class_weights = None
            print("使用默认的均匀类别权重")
        
        # 设备已在前面定义
        
        # 初始化模型
        num_classes = len(set(y_filtered))
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
        
        # 训练并评估模型
        print(f"\n开始训练模型 (epochs: {config.num_epochs})")
        
        # 如果存在类别权重，使用加权损失函数
        if config.class_weights is not None:
            print("使用加权损失函数处理类别不平衡")
            weighted_criterion = nn.CrossEntropyLoss(weight=config.class_weights)
            criterion = weighted_criterion
        
        # 使用修改后的train_model函数，添加测试集参数
        model, metrics = train_model(
            model, train_loader, val_loader, test_loader, criterion, optimizer, device, 
            config.num_epochs, config, class_weights=config.class_weights,
            early_stopping_enabled=config.early_stopping
        )
        
        # 可视化训练结果
        print("\n生成训练过程可视化图表...")
        plot_training_metrics(metrics, config)
        
    except Exception as e:
        print(f"主函数执行错误: {str(e)}")
        with open(config.error_log, 'a') as f:
            f.write(f"主函数执行错误 - {str(e)}\n")
        raise e

if __name__ == "__main__":
    main() 