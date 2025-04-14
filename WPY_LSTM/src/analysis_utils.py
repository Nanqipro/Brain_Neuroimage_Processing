import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
import json
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from typing import Tuple, List, Dict, Any
from analysis_config import AnalysisConfig

class DataProcessor:
    """数据处理器类,用于数据预处理和平衡"""
    @staticmethod
    def balance_data(X, y, min_samples):
        """
        通过下采样多数类来平衡数据集
        参数:
            X: 特征矩阵
            y: 标签数组
            min_samples: 每个类别的最小样本数
        返回:
            平衡后的特征矩阵和标签数组
        """
        unique_labels, counts = np.unique(y, return_counts=True)
        min_count = max(min_samples, min(counts[counts > 0]))
        
        balanced_indices = []
        for label in unique_labels:
            label_indices = np.where(y == label)[0]
            if len(label_indices) >= min_samples:
                balanced_indices.extend(
                    np.random.choice(label_indices, min_count, replace=False)
                )
        
        return X[balanced_indices], y[balanced_indices]
    
    @staticmethod
    def merge_rare_behaviors(X, y, labels, min_samples):
        """
        将罕见行为合并到'其他'类别中
        参数:
            X: 特征矩阵
            y: 标签数组
            labels: 标签名称列表
            min_samples: 最小样本数阈值
        返回:
            处理后的特征矩阵、标签数组和更新的标签名称列表
        """
        unique_labels, counts = np.unique(y, return_counts=True)
        rare_labels = unique_labels[counts < min_samples]
        
        if len(rare_labels) > 0:
            new_y = y.copy()
            new_labels = labels.copy()
            
            # 创建'其他'类别
            other_idx = len(labels)
            for label in rare_labels:
                new_y[y == label] = other_idx
            
            new_labels = np.append(new_labels, 'Other')
            return X, new_y, new_labels
        
        return X, y, labels

class StatisticalAnalyzer:
    """统计分析器类,用于执行各种统计分析"""
    def __init__(self, config):
        self.config = config
    
    def perform_anova(self, X, y):
        """
        对每个神经元执行单因素方差分析
        参数:
            X: 特征矩阵
            y: 标签数组
        返回:
            F值和校正后的p值列表
        """
        f_values = []
        p_values = []
        
        for neuron_idx in range(X.shape[1]):
            groups = [X[y == label, neuron_idx] for label in np.unique(y)]
            f_val, p_val = stats.f_oneway(*groups)
            f_values.append(f_val)
            p_values.append(p_val)
        
        # 多重比较校正
        _, p_corrected, _, _ = multipletests(p_values, method='bonferroni')
        
        return f_values, p_corrected
    
    def calculate_effect_sizes(self, X, y, behavior_labels):
        """
        计算每个神经元-行为对的Cohen's d效应量
        参数:
            X: 特征矩阵
            y: 标签数组
            behavior_labels: 行为标签列表
        返回:
            包含效应量和显著神经元的字典
        """
        effect_sizes = {}
        
        for behavior_idx, behavior in enumerate(behavior_labels):
            behavior_mask = (y == behavior_idx)
            behavior_data = X[behavior_mask]
            other_data = X[~behavior_mask]
            
            # 计算每个神经元的效应量
            behavior_mean = np.mean(behavior_data, axis=0)
            other_mean = np.mean(other_data, axis=0)
            behavior_std = np.std(behavior_data, axis=0)
            other_std = np.std(other_data, axis=0)
            
            pooled_std = np.sqrt((behavior_std**2 + other_std**2) / 2)
            effect_size = np.abs(behavior_mean - other_mean) / pooled_std
            
            effect_sizes[behavior] = {
                'effect_sizes': effect_size,
                'significant_neurons': np.where(effect_size > self.config.analysis_params['effect_size_threshold'])[0]
            }
        
        return effect_sizes
    
    def analyze_temporal_correlations(self, X, y):
        """
        分析神经元之间的时间相关性
        参数:
            X: 特征矩阵
            y: 标签数组
        返回:
            不同时间窗口的相关性字典
        """
        correlations = {}
        
        # 确保数据有效
        if len(X) == 0:
            return correlations
            
        # 对每个时间窗口计算相关性
        for window in self.config.analysis_params['correlation_windows']:
            # 确保窗口大小不超过数据长度
            if window >= len(X):
                print(f"Warning: Window size {window} is larger than data length {len(X)}. Skipping.")
                continue
                
            window_correlations = []
            for i in range(len(X) - window):
                window_data = X[i:i+window]
                # 计算相关系数矩阵
                try:
                    # 移除包含NaN的行
                    valid_data = window_data[~np.isnan(window_data).any(axis=1)]
                    if len(valid_data) > 1:  # 确保有足够的数据点
                        corr_matrix = np.corrcoef(valid_data.T)
                        # 只取上三角矩阵的值（排除对角线）
                        upper_triangle = np.triu(corr_matrix, k=1)
                        # 计算平均相关性（排除0值）
                        non_zero = upper_triangle[upper_triangle != 0]
                        if len(non_zero) > 0:
                            mean_corr = np.mean(np.abs(non_zero))
                            window_correlations.append(mean_corr)
                        else:
                            window_correlations.append(0)
                    else:
                        window_correlations.append(0)
                except Exception as e:
                    print(f"Warning: Error calculating correlation for window {i}: {str(e)}")
                    window_correlations.append(0)
            
            # 只有在有有效的相关性值时才保存结果
            if window_correlations:
                correlations[window] = window_correlations
            
        return correlations

class ResultSaver:
    """结果保存器类,用于保存分析结果"""
    def __init__(self, config):
        self.config = config
    
    def save_statistical_results(self, f_values, p_values, effect_sizes):
        """
        保存统计分析结果
        参数:
            f_values: F检验值列表
            p_values: p值列表
            effect_sizes: 效应量字典
        """
        # 保存F检验结果
        stats_df = pd.DataFrame({
            'Neuron': [f'Neuron_{i+1}' for i in range(len(f_values))],
            'F_value': f_values,
            'P_value': p_values,
            'Significant': p_values < self.config.analysis_params['p_value_threshold']
        })
        stats_df.to_csv(self.config.statistical_results_csv, index=False)
        
        # 保存效应量
        with open(self.config.neuron_specificity_json, 'w') as f:
            json.dump(effect_sizes, f, indent=4, cls=NumpyEncoder)
    
    def save_temporal_correlations(self, correlations):
        """
        保存时间相关性结果
        参数:
            correlations: 时间相关性字典
        """
        if not correlations:  # 如果字典为空
            print("Warning: No temporal correlations to save.")
            return
            
        # 找到最长的时间序列长度
        max_length = max(len(corr) for corr in correlations.values())
        
        # 创建一个填充了NaN的DataFrame
        data = {}
        for window, corr_values in correlations.items():
            # 将较短的序列用NaN填充到相同长度
            padded_values = corr_values + [np.nan] * (max_length - len(corr_values))
            data[f'Window_{window}'] = padded_values
        
        # 创建DataFrame
        corr_df = pd.DataFrame(data)
        corr_df.index.name = 'Time Point'
        
        # 保存结果
        output_path = os.path.join(self.config.analysis_dir, 'temporal_correlations.csv')
        corr_df.to_csv(output_path)
        
        # 计算并保存每个窗口的平均相关性
        mean_correlations = {window: np.nanmean(values) for window, values in correlations.items()}
        mean_df = pd.DataFrame(list(mean_correlations.items()), 
                             columns=['Window Size', 'Mean Correlation'])
        mean_output_path = os.path.join(self.config.analysis_dir, 'temporal_correlations_summary.csv')
        mean_df.to_csv(mean_output_path, index=False)

class NumpyEncoder(json.JSONEncoder):
    """用于numpy类型的特殊JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def split_data(dataset: Dataset, 
               y_targets: np.ndarray, 
               config: AnalysisConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将数据集按时间顺序划分为训练集、验证集和测试集索引。

    返回索引数组，而不是 Subset 对象。

    参数:
        dataset (Dataset): 完整的 PyTorch 数据集 (主要用于获取长度)。
        y_targets (np.ndarray): 数据集中每个样本对应的类别标签 (可选，用于打印分布)。
        config (AnalysisConfig): 配置对象，包含分割比例和随机种子。

    返回:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 训练集、验证集和测试集的索引数组。
    """
    num_samples = len(dataset)
    indices = np.arange(num_samples)

    # 检查配置中的比例参数
    val_test_ratio = config.val_test_split_ratio
    test_ratio_in_val_test = config.test_split_ratio

    if not (0 < val_test_ratio < 1):
        raise ValueError("val_test_split_ratio 必须在 0 和 1 之间。")
    if not (0 < test_ratio_in_val_test < 1):
        raise ValueError("test_split_ratio 必须在 0 和 1 之间。")

    # 计算分割点索引
    num_val_test = int(num_samples * val_test_ratio)
    num_test = int(num_val_test * test_ratio_in_val_test)
    num_val = num_val_test - num_test
    num_train = num_samples - num_val_test

    if num_train <= 0 or num_val <= 0 or num_test <= 0:
        raise ValueError(f"数据集太小，无法按指定比例分割: 训练集 {num_train}, 验证集 {num_val}, 测试集 {num_test}")

    # 按顺序分配索引
    train_indices = indices[:num_train]
    val_indices = indices[num_train : num_train + num_val]
    test_indices = indices[num_train + num_val :]

    print(f"按时间顺序分割完成 (返回索引): 训练集 {len(train_indices)}, 验证集 {len(val_indices)}, 测试集 {len(test_indices)}")

    # 可选：检查每个子集的标签分布（不用于分层，仅供参考）
    y_targets_full = y_targets # 假设 y_targets 长度与 dataset 匹配
    if len(y_targets_full) != num_samples:
         print(f"警告: 提供的 y_targets 长度 ({len(y_targets_full)}) 与数据集长度 ({num_samples}) 不符，无法准确显示子集标签分布。")
    else:
        for name, subset_indices in zip(["训练集", "验证集", "测试集"], [train_indices, val_indices, test_indices]):
             unique, counts = np.unique(y_targets_full[subset_indices], return_counts=True)
             print(f"{name} 类别分布 (仅供参考): {dict(zip(unique, counts))}")

    # 直接返回索引数组
    return train_indices, val_indices, test_indices 