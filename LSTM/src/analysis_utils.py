import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
import json
import os

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
        min_length = float('inf')
        
        # 首先计算每个窗口的相关性，并找到最短长度
        for window in self.config.analysis_params['correlation_windows']:
            window_correlations = []
            for i in range(len(X) - window):
                window_data = X[i:i+window]
                corr_matrix = np.corrcoef(window_data.T)
                window_correlations.append(np.mean(np.abs(corr_matrix)))
            correlations[window] = window_correlations
            min_length = min(min_length, len(window_correlations))
        
        # 截断所有数组到相同长度
        for window in correlations:
            correlations[window] = correlations[window][:min_length]
        
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
        # 创建时间索引
        time_points = range(len(next(iter(correlations.values()))))
        
        # 创建DataFrame
        corr_df = pd.DataFrame(correlations, index=time_points)
        corr_df.index.name = 'Time Point'
        corr_df.columns.name = 'Window Size'
        
        # 保存结果
        output_path = os.path.join(self.config.analysis_dir, 'temporal_correlations.csv')
        corr_df.to_csv(output_path)

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