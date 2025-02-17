import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
import json
import os

class DataProcessor:
    @staticmethod
    def balance_data(X, y, min_samples):
        """Balance dataset by downsampling majority classes"""
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
        """Merge rare behaviors into an 'Other' category"""
        unique_labels, counts = np.unique(y, return_counts=True)
        rare_labels = unique_labels[counts < min_samples]
        
        if len(rare_labels) > 0:
            new_y = y.copy()
            new_labels = labels.copy()
            
            # Create 'Other' category
            other_idx = len(labels)
            for label in rare_labels:
                new_y[y == label] = other_idx
            
            new_labels = np.append(new_labels, 'Other')
            return X, new_y, new_labels
        
        return X, y, labels

class StatisticalAnalyzer:
    def __init__(self, config):
        self.config = config
    
    def perform_anova(self, X, y):
        """Perform one-way ANOVA for each neuron"""
        f_values = []
        p_values = []
        
        for neuron_idx in range(X.shape[1]):
            groups = [X[y == label, neuron_idx] for label in np.unique(y)]
            f_val, p_val = stats.f_oneway(*groups)
            f_values.append(f_val)
            p_values.append(p_val)
        
        # Multiple comparison correction
        _, p_corrected, _, _ = multipletests(p_values, method='bonferroni')
        
        return f_values, p_corrected
    
    def calculate_effect_sizes(self, X, y, behavior_labels):
        """Calculate Cohen's d effect size for each neuron-behavior pair"""
        effect_sizes = {}
        
        for behavior_idx, behavior in enumerate(behavior_labels):
            behavior_mask = (y == behavior_idx)
            behavior_data = X[behavior_mask]
            other_data = X[~behavior_mask]
            
            # Calculate effect size for each neuron
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
        """Analyze temporal correlations between neurons"""
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
    def __init__(self, config):
        self.config = config
    
    def save_statistical_results(self, f_values, p_values, effect_sizes):
        """Save statistical analysis results"""
        # Save F-test results
        stats_df = pd.DataFrame({
            'Neuron': [f'Neuron_{i+1}' for i in range(len(f_values))],
            'F_value': f_values,
            'P_value': p_values,
            'Significant': p_values < self.config.analysis_params['p_value_threshold']
        })
        stats_df.to_csv(self.config.statistical_results_csv, index=False)
        
        # Save effect sizes
        with open(self.config.neuron_specificity_json, 'w') as f:
            json.dump(effect_sizes, f, indent=4, cls=NumpyEncoder)
    
    def save_temporal_correlations(self, correlations):
        """Save temporal correlation results"""
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
    """Special JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj) 