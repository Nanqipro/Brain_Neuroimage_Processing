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
from analysis_utils import split_data # 导入新的分割函数
from joblib import dump, load # 用于保存和加载 scaler
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
    """
    神经元数据处理器类：负责加载、预处理神经元活动数据和行为标签
    
    主要功能:
    1. 加载神经元数据文件
    2. 识别和预处理神经元活动数据 (不含标准化)
    3. 处理行为标签
    4. 提供标准化方法 (fit 和 transform 分离)
    5. 应用K-means聚类 (现在可选地在标准化数据上进行)
    
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
            
        self.scaler = StandardScaler()  # 初始化 StandardScaler
        self.scaler_fitted = False # 跟踪 scaler 是否已拟合
        self.label_encoder = LabelEncoder()  # 用于行为标签编码
        self.available_neuron_cols: List[str] = []  # 存储可用的神经元列名
        
    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据预处理函数 (不包含标准化)
        返回未标准化的 X 和编码后的 y
        
        执行步骤:
        1. 提取神经元数据
        2. 处理缺失值
        3. 编码行为标签
        
        返回
        ----------
        X : np.ndarray
            预处理后但未标准化的神经元数据
        y : np.ndarray
            编码后的行为标签
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
        print(f"可用神经元数量: {len(available_cols)} / {len(neuron_cols)}")
        
        if len(neuron_cols) != len(available_cols):
            missing_cols = set(neuron_cols) - set(available_cols)
            print(f"Missing neurons: {missing_cols}")
        
        # 保存可用的神经元列名列表，以便其他方法使用
        self.available_neuron_cols = available_cols
        
        # 获取可用神经元数据
        try:
            X = self.data[available_cols].values.astype(float) # 确保是浮点数
        except Exception as e:
            print(f"提取神经元数据时出错: {e}")
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
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])
        
        # 检查是否有无效值(inf)
        if not np.isfinite(X).all():
            print("发现无限值，将替换为有限值")
            X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(X.dtype).max, neginf=np.finfo(X.dtype).min)
        
        # 编码行为标签
        behavior_col = None
        for col_name in ['behavior', 'Behavior', 'label', 'Label', 'class', 'Class']:
            if col_name in self.data.columns:
                behavior_col = col_name
                break
                
        if behavior_col is None:
            raise ValueError("找不到行为标签列，请确保数据中包含'behavior'或'label'列")
            
        # 处理缺失的行为标签
        behavior_data = self.data[behavior_col].fillna('unknown').astype(str)
        
        # 根据include_cd1_behavior配置决定是否纳入CD1行为标签
        if not self.config.include_cd1_behavior:
            print("\n配置设置为不纳入CD1行为标签")
            # 创建新的DataFrame以保存非CD1数据
            mask = (behavior_data != 'CD1')
            original_shape = behavior_data.shape[0]
            behavior_data = behavior_data[mask]
            self.data = self.data[mask].reset_index(drop=True)
            X = X[mask]
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
        
        print("数据基础预处理完成 (未标准化)。")
        return X, y

    def fit_scaler(self, X_train: np.ndarray) -> None:
        """
        在训练数据上拟合 StandardScaler。
        
        参数:
            X_train (np.ndarray): 训练集的特征数据。
        """
        print("--- 在训练数据上拟合 StandardScaler ---")
        if X_train.ndim != 2:
            raise ValueError(f"拟合 scaler 的输入数据维度应为 2，实际为 {X_train.ndim}")
        if X_train.shape[0] == 0:
            raise ValueError("拟合 scaler 的训练数据不能为空。")
            
        try:
            self.scaler.fit(X_train)
            self.scaler_fitted = True
            print(f"StandardScaler 拟合完成。均值形状: {self.scaler.mean_.shape}, 标准差形状: {self.scaler.scale_.shape}")
        except Exception as e:
            print(f"拟合 StandardScaler 时出错: {e}")
            self.scaler_fitted = False
            raise
            
    def scale_data(self, X: np.ndarray) -> np.ndarray:
        """
        使用已拟合的 StandardScaler 转换数据。
        
        参数:
            X (np.ndarray): 需要标准化的数据 (训练、验证或测试集)。
        
        返回:
            np.ndarray: 标准化后的数据。
            
        异常:
            RuntimeError: 如果 scaler 尚未拟合。
        """
        if not self.scaler_fitted:
            raise RuntimeError("StandardScaler 尚未拟合，请先调用 fit_scaler。")
        if X.ndim != 2:
            raise ValueError(f"进行 scale 的输入数据维度应为 2，实际为 {X.ndim}")
            
        print(f"--- 使用已拟合的 StandardScaler 转换数据 (形状: {X.shape}) ---")
        try:
            X_scaled = self.scaler.transform(X)
            return X_scaled
        except Exception as e:
             print(f"转换数据时出错: {e}")
             raise

    def apply_kmeans(self, X: np.ndarray, use_scaled: bool = True) -> Tuple[KMeans, np.ndarray]:
        """
        应用K-means聚类分析
        
        参数
        ----------
        X : np.ndarray
            用于聚类的数据 (可以是原始的或标准化的)
        use_scaled : bool, optional
            是否在应用KMeans之前对数据进行标准化, by default True
            如果为 True，会先对数据进行标准化
            如果为 False，将直接在传入的 X 上进行聚类
        """
        if use_scaled:
            print("KMeans将在标准化数据上执行。")
            if not self.scaler_fitted:
                print("警告：Scaler尚未拟合，将在当前数据上临时拟合并进行标准化用于KMeans。这可能导致数据泄露，建议先在训练集上拟合。")
                # 临时拟合（不推荐在主流程中使用，但保留KMeans的独立可用性）
                temp_scaler = StandardScaler().fit(X)
                X_to_cluster = temp_scaler.transform(X)
            else:
                X_to_cluster = self.scale_data(X) # 使用已经拟合好的 scaler
        else:
            print("KMeans将在未标准化数据上执行。")
            X_to_cluster = X
            
        print(f"\n使用 {self.config.n_clusters} 个聚类进行K-means聚类")
        kmeans = KMeans(n_clusters=self.config.n_clusters, 
                       random_state=self.config.random_seed,
                       n_init=10)
        cluster_labels = kmeans.fit_predict(X_to_cluster)
        
        # 打印聚类分布情况
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print("\n聚类分布:")
        for label, count in zip(unique_labels, counts):
            print(f"聚类 {label}: {count} 个样本")
        
        return kmeans, cluster_labels

def main() -> None:
    """
    主函数
    执行步骤:
    1. 初始化配置
    2. 数据预处理 (获取未缩放数据)
    3. 聚类分析 (可选，在标准化数据上)
    4. 数据按时间顺序分割
    5. 拟合 Scaler 并标准化数据
    6. 创建序列数据集和加载器
    7. 模型训练
    8. 保存 Scaler
    9. 结果可视化
    """
    config = AnalysisConfig()
    set_random_seed(config.random_seed)
    os.makedirs(os.path.dirname(config.error_log), exist_ok=True)
    with open(config.error_log, 'w') as f: f.write(f"错误日志创建于 {datetime.datetime.now()}\n")
        
    try:
        # 1. 数据预处理 (获取未缩放数据)
        processor = NeuronDataProcessor(config)
        X, y = processor.preprocess_data() # 获取未标准化的 X 和 y
        
        # 2. 聚类分析 (现在可选地在标准化数据上进行)
        # 注意：如果在标准化之前调用，可能需要先fit scaler
        # kmeans, cluster_labels = processor.apply_kmeans(X, use_scaled=True) 

        # 3. 数据按时间顺序分割
        # 创建临时的完整数据集对象以获取长度，但数据本身未处理
        temp_dataset = NeuronDataset(X, y, config.sequence_length) 
        y_for_split = y[config.sequence_length-1:]
        if len(y_for_split) != len(temp_dataset):
            print(f"警告: y_for_split 长度与数据集不匹配。")
            min_len = min(len(y_for_split), len(temp_dataset))
            y_for_split = y_for_split[:min_len]
        
        print("\n执行 Train/Validation/Test 数据分割 (基于原始数据索引)...")
        # 使用临时的 dataset 进行分割以获取索引
        train_subset_indices, val_subset_indices, test_subset_indices = split_data(
            temp_dataset, y_for_split, config
        )
        # 根据索引获取分割后的原始（未缩放）数据
        X_train, y_train = X[train_subset_indices], y[train_subset_indices]
        X_val, y_val = X[val_subset_indices], y[val_subset_indices]
        X_test, y_test = X[test_subset_indices], y[test_subset_indices]
        # (X_test, y_test 在此脚本中不直接使用，但分割完成)
        del temp_dataset # 释放临时对象
        print(f"原始数据分割完成: 训练集 {X_train.shape}, 验证集 {X_val.shape}, 测试集 {X_test.shape}")

        # 4. 拟合 Scaler 并标准化数据
        processor.fit_scaler(X_train) # 在训练集上拟合 scaler
        X_train_scaled = processor.scale_data(X_train) # 标准化训练集
        X_val_scaled = processor.scale_data(X_val)     # 标准化验证集
        # (测试集将在评估脚本中标准化)

        # 5. 创建序列数据集和加载器 (使用标准化后的数据)
        print("\n创建序列数据集 (使用标准化数据)...")
        train_dataset = NeuronDataset(X_train_scaled, y_train, config.sequence_length)
        val_dataset = NeuronDataset(X_val_scaled, y_val, config.sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        print(f"数据加载器创建完成: 训练集 {len(train_loader.dataset)}, 验证集 {len(val_loader.dataset)}")

        # 6. 模型训练 (与之前类似，但使用新的加载器)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        num_classes = len(processor.label_encoder.classes_)
        model = EnhancedNeuronLSTM(
            input_size=X_train_scaled.shape[1], # 使用标准化数据的维度
            hidden_size=config.hidden_size, 
            num_layers=config.num_layers,
            num_classes=num_classes,
            latent_dim=config.latent_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        ).to(device)
        model.summary()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        print(f"\n开始训练模型 (epochs: {config.num_epochs})")
        model, metrics = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, 
            config.num_epochs, config, early_stopping_enabled=config.analysis_params.get('early_stopping_enabled', False)
        )
        
        # 7. 保存 Scaler (重要！评估时需要加载)
        scaler_path = os.path.join(config.model_dir, f'scaler_{config.data_identifier}.joblib')
        try:
            dump(processor.scaler, scaler_path)
            print(f"Scaler 已保存到: {scaler_path}")
        except Exception as e:
            print(f"保存 Scaler 时出错: {e}")
        
        # 8. 结果可视化 (与之前相同)
        plot_training_metrics(metrics, config)
        print(f"\n训练完成! 最佳验证准确率: {metrics['best_val_acc']:.2f}%")
        print(f"注意: 测试集未在此脚本中使用。请运行 evaluate_lstm.py 在测试集上进行最终评估。")
        
    except Exception as e:
        print(f"主函数执行错误: {str(e)}")
        with open(config.error_log, 'a') as f:
            f.write(f"主函数执行错误 - {str(e)}\n")
        raise e

if __name__ == "__main__":
    main() 