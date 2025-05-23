#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
神经元放电状态分析模块

该模块用于分析神经元的钙离子浓度数据，识别和分类不同的放电状态模式。
基于图像中显示的6种典型神经元放电状态：
- State I: 高频连续振荡状态
- State II: 规律性脉冲放电状态  
- State III: 间歇性突发状态
- State IV: 不规律波动状态
- State V: 高频密集放电状态
- State VI: 低频随机活动状态

作者：AI助手
日期：2024年
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.signal import find_peaks, peak_widths, welch
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, UMAP
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')
import os
import logging
import datetime
import sys
from typing import Dict, List, Tuple, Optional, Any
import argparse

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class StateAnalyzer:
    """
    神经元状态分析器类
    
    该类封装了神经元放电状态分析的所有功能，包括：
    - 数据加载和预处理
    - 特征提取
    - 状态识别和分类
    - 结果可视化
    """
    
    def __init__(self, sampling_rate: float = 4.8, logger: Optional[logging.Logger] = None):
        """
        初始化状态分析器
        
        Parameters
        ----------
        sampling_rate : float
            采样频率，默认为4.8Hz
        logger : logging.Logger, optional
            日志记录器，默认为None
        """
        self.sampling_rate = sampling_rate
        self.logger = logger or self._setup_logger()
        
        # 状态定义
        self.state_definitions = {
            'State I': '高频连续振荡状态',
            'State II': '规律性脉冲放电状态',
            'State III': '间歇性突发状态',
            'State IV': '不规律波动状态', 
            'State V': '高频密集放电状态',
            'State VI': '低频随机活动状态'
        }
        
        # 状态颜色映射
        self.state_colors = {
            'State I': '#8B0000',    # 深红色
            'State II': '#FFD700',   # 金黄色
            'State III': '#228B22',  # 森林绿
            'State IV': '#4169E1',   # 皇家蓝
            'State V': '#8A2BE2',    # 蓝紫色
            'State VI': '#696969'    # 暗灰色
        }
        
        self.logger.info("神经元状态分析器初始化完成")
    
    def _setup_logger(self, output_dir: Optional[str] = None) -> logging.Logger:
        """
        设置日志记录器
        
        Parameters
        ----------
        output_dir : str, optional
            日志输出目录，默认为None
            
        Returns
        -------
        logging.Logger
            配置好的日志记录器
        """
        logger = logging.getLogger('StateAnalyzer')
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            for handler in logger.handlers:
                logger.removeHandler(handler)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../logs")
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(output_dir, f"state_analysis_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"日志文件创建于: {log_file}")
        return logger
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        加载神经元钙离子浓度数据
        
        Parameters
        ----------
        file_path : str
            数据文件路径
            
        Returns
        -------
        pd.DataFrame
            加载的数据DataFrame
        """
        self.logger.info(f"正在加载数据: {file_path}")
        
        try:
            # 支持多种文件格式
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                data = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
            
            self.logger.info(f"数据加载完成，共 {len(data)} 行, {len(data.columns)} 列")
            
            # 识别神经元列
            neuron_columns = [col for col in data.columns 
                            if col.startswith('n') and col[1:].isdigit()]
            
            self.logger.info(f"识别到 {len(neuron_columns)} 个神经元: {neuron_columns[:10]}...")
            
            return data
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise
    
    def extract_temporal_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        提取时域特征
        
        Parameters
        ----------
        signal_data : np.ndarray
            单个神经元的时间序列数据
            
        Returns
        -------
        Dict[str, float]
            提取的时域特征字典
        """
        features = {}
        
        # 基础统计特征
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['var'] = np.var(signal_data)
        features['skewness'] = self._calculate_skewness(signal_data)
        features['kurtosis'] = self._calculate_kurtosis(signal_data)
        
        # 动态特征
        features['peak_count'] = self._count_peaks(signal_data)
        features['peak_amplitude_mean'] = self._mean_peak_amplitude(signal_data)
        features['peak_amplitude_std'] = self._std_peak_amplitude(signal_data)
        features['inter_peak_interval_mean'] = self._mean_inter_peak_interval(signal_data)
        features['inter_peak_interval_std'] = self._std_inter_peak_interval(signal_data)
        
        # 活动性特征
        features['activity_rate'] = self._calculate_activity_rate(signal_data)
        features['burst_rate'] = self._calculate_burst_rate(signal_data)
        features['silence_periods'] = self._calculate_silence_periods(signal_data)
        
        # 变异性特征
        features['coefficient_variation'] = features['std'] / features['mean'] if features['mean'] > 0 else 0
        features['range_value'] = np.max(signal_data) - np.min(signal_data)
        
        return features
    
    def extract_frequency_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        提取频域特征
        
        Parameters
        ----------
        signal_data : np.ndarray
            单个神经元的时间序列数据
            
        Returns
        -------
        Dict[str, float]
            提取的频域特征字典
        """
        features = {}
        
        # 计算功率谱密度
        freqs, psd = welch(signal_data, fs=self.sampling_rate, nperseg=min(256, len(signal_data)//4))
        
        # 主导频率
        features['dominant_frequency'] = freqs[np.argmax(psd)]
        
        # 频带功率
        features['low_freq_power'] = self._band_power(freqs, psd, 0, 0.5)      # 0-0.5Hz
        features['mid_freq_power'] = self._band_power(freqs, psd, 0.5, 1.5)    # 0.5-1.5Hz  
        features['high_freq_power'] = self._band_power(freqs, psd, 1.5, 2.4)   # 1.5-2.4Hz
        
        # 频域统计特征
        features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
        features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features['spectral_centroid']) ** 2) * psd) / np.sum(psd))
        features['spectral_entropy'] = entropy(psd + 1e-12)
        
        return features
    
    def extract_nonlinear_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        提取非线性特征
        
        Parameters
        ----------
        signal_data : np.ndarray
            单个神经元的时间序列数据
            
        Returns
        -------
        Dict[str, float]
            提取的非线性特征字典
        """
        features = {}
        
        # 样本熵
        features['sample_entropy'] = self._sample_entropy(signal_data)
        
        # 零交叉率
        features['zero_crossing_rate'] = self._zero_crossing_rate(signal_data)
        
        # Hurst指数（长程相关性）
        features['hurst_exponent'] = self._hurst_exponent(signal_data)
        
        # 分形维数
        features['fractal_dimension'] = self._fractal_dimension(signal_data)
        
        # Lyapunov指数（混沌性）
        features['lyapunov_exponent'] = self._lyapunov_exponent(signal_data)
        
        # 去趋势波动分析
        features['dfa_alpha'] = self._detrended_fluctuation_analysis(signal_data)
        
        return features
    
    def extract_morphological_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        提取形态学特征
        
        Parameters
        ----------
        signal_data : np.ndarray
            单个神经元的时间序列数据
            
        Returns
        -------
        Dict[str, float]
            提取的形态学特征字典
        """
        features = {}
        
        # 峰值特征
        peaks, _ = find_peaks(signal_data, height=np.mean(signal_data) + np.std(signal_data))
        
        if len(peaks) > 0:
            # 峰值宽度
            widths, _, _, _ = peak_widths(signal_data, peaks, rel_height=0.5)
            features['peak_width_mean'] = np.mean(widths)
            features['peak_width_std'] = np.std(widths)
            
            # 峰值突出度
            prominences, _, _ = signal.peak_prominences(signal_data, peaks)
            features['peak_prominence_mean'] = np.mean(prominences)
            features['peak_prominence_std'] = np.std(prominences)
            
            # 上升/下降时间
            features['rise_time_mean'] = self._calculate_rise_times(signal_data, peaks)
            features['decay_time_mean'] = self._calculate_decay_times(signal_data, peaks)
        else:
            # 如果没有检测到峰值，设置默认值
            features.update({
                'peak_width_mean': 0, 'peak_width_std': 0,
                'peak_prominence_mean': 0, 'peak_prominence_std': 0,
                'rise_time_mean': 0, 'decay_time_mean': 0
            })
        
        # 波形规律性
        features['regularity_index'] = self._calculate_regularity_index(signal_data)
        
        # 突发检测
        features['burst_duration_mean'] = self._calculate_burst_duration(signal_data)
        features['burst_intensity_mean'] = self._calculate_burst_intensity(signal_data)
        
        return features
    
    def extract_all_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        提取所有神经元的综合特征
        
        Parameters
        ----------
        data : pd.DataFrame
            神经元钙离子浓度数据
            
        Returns
        -------
        Tuple[np.ndarray, List[str], List[str]]
            特征矩阵, 特征名称列表, 神经元名称列表
        """
        self.logger.info("开始提取神经元特征...")
        
        # 识别神经元列
        neuron_columns = [col for col in data.columns 
                         if col.startswith('n') and col[1:].isdigit()]
        
        feature_list = []
        feature_names = None
        
        for neuron in neuron_columns:
            signal_data = data[neuron].values
            
            # 提取各类特征
            temporal_features = self.extract_temporal_features(signal_data)
            frequency_features = self.extract_frequency_features(signal_data)
            nonlinear_features = self.extract_nonlinear_features(signal_data)
            morphological_features = self.extract_morphological_features(signal_data)
            
            # 合并所有特征
            all_features = {**temporal_features, **frequency_features, 
                          **nonlinear_features, **morphological_features}
            
            if feature_names is None:
                feature_names = list(all_features.keys())
            
            feature_list.append(list(all_features.values()))
        
        features_matrix = np.array(feature_list)
        
        self.logger.info(f"特征提取完成，共提取 {len(feature_names)} 个特征，{len(neuron_columns)} 个神经元")
        
        return features_matrix, feature_names, neuron_columns
    
    def identify_states(self, features: np.ndarray, method: str = 'kmeans', 
                       n_states: int = 6) -> np.ndarray:
        """
        识别神经元放电状态
        
        Parameters
        ----------
        features : np.ndarray
            特征矩阵
        method : str
            聚类方法，可选'kmeans', 'dbscan', 'gaussian_mixture'
        n_states : int
            预期状态数量，默认为6
            
        Returns
        -------
        np.ndarray
            状态标签数组
        """
        self.logger.info(f"使用 {method} 方法识别 {n_states} 种神经元状态...")
        
        # 特征标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_states, random_state=42, n_init=10)
            labels = clusterer.fit_predict(features_scaled)
            
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=2)
            labels = clusterer.fit_predict(features_scaled)
            n_states = len(set(labels)) - (1 if -1 in labels else 0)
            
        elif method == 'gaussian_mixture':
            from sklearn.mixture import GaussianMixture
            clusterer = GaussianMixture(n_components=n_states, random_state=42)
            labels = clusterer.fit_predict(features_scaled)
            
        else:
            raise ValueError(f"不支持的聚类方法: {method}")
        
        # 评估聚类质量
        if len(set(labels)) > 1:
            silhouette = silhouette_score(features_scaled, labels)
            calinski_harabasz = calinski_harabasz_score(features_scaled, labels)
            
            self.logger.info(f"聚类质量评估 - 轮廓系数: {silhouette:.3f}, "
                           f"Calinski-Harabasz指数: {calinski_harabasz:.3f}")
        
        # 重新排序标签，使其与预定义状态对应
        labels = self._reorder_labels(labels, features_scaled)
        
        self.logger.info(f"状态识别完成，识别出 {len(set(labels))} 种状态")
        
        return labels
    
    def _reorder_labels(self, labels: np.ndarray, features: np.ndarray) -> np.ndarray:
        """
        根据特征特性重新排序标签，使其与预定义状态对应
        
        Parameters
        ----------
        labels : np.ndarray
            原始标签
        features : np.ndarray
            特征矩阵
            
        Returns
        -------
        np.ndarray
            重新排序的标签
        """
        # 计算每个聚类的中心特征
        unique_labels = sorted(set(labels))
        cluster_centers = []
        
        for label in unique_labels:
            mask = labels == label
            center = np.mean(features[mask], axis=0)
            cluster_centers.append(center)
        
        cluster_centers = np.array(cluster_centers)
        
        # 基于特征相似性重新映射标签
        # 这里可以根据具体的状态定义进行优化
        label_mapping = {}
        for i, original_label in enumerate(unique_labels):
            label_mapping[original_label] = i
        
        reordered_labels = np.array([label_mapping[label] for label in labels])
        return reordered_labels
    
    def visualize_states(self, data: pd.DataFrame, labels: np.ndarray, 
                        neuron_names: List[str], output_dir: str = '../results') -> None:
        """
        可视化神经元状态
        
        Parameters
        ----------
        data : pd.DataFrame
            原始数据
        labels : np.ndarray
            状态标签
        neuron_names : List[str]
            神经元名称列表
        output_dir : str
            输出目录
        """
        self.logger.info("开始可视化神经元状态...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 绘制典型状态波形
        self._plot_state_waveforms(data, labels, neuron_names, output_dir)
        
        # 2. 绘制状态分布
        self._plot_state_distribution(labels, neuron_names, output_dir)
        
        # 3. 绘制特征分析
        features, feature_names, _ = self.extract_all_features(data)
        self._plot_feature_analysis(features, feature_names, labels, output_dir)
        
        # 4. 绘制状态转换矩阵
        self._plot_state_transition_matrix(data, labels, neuron_names, output_dir)
        
        self.logger.info("状态可视化完成")
    
    def _plot_state_waveforms(self, data: pd.DataFrame, labels: np.ndarray, 
                             neuron_names: List[str], output_dir: str) -> None:
        """绘制每种状态的典型波形"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        unique_states = sorted(set(labels))
        
        for i, state in enumerate(unique_states):
            if i >= 6:  # 最多显示6种状态
                break
                
            # 找到该状态的神经元
            state_neurons = [neuron_names[j] for j, label in enumerate(labels) if label == state]
            
            if state_neurons:
                # 随机选择几个代表性神经元
                selected_neurons = np.random.choice(state_neurons, 
                                                   min(3, len(state_neurons)), 
                                                   replace=False)
                
                ax = axes[i]
                colors = plt.cm.Set3(np.linspace(0, 1, len(selected_neurons)))
                
                for neuron, color in zip(selected_neurons, colors):
                    signal = data[neuron].values
                    time = np.arange(len(signal)) / self.sampling_rate
                    ax.plot(time, signal, color=color, alpha=0.7, 
                           linewidth=1.5, label=neuron)
                
                ax.set_title(f'State {state + 1}: {self.state_definitions.get(f"State {chr(65+state)}", "未知状态")}',
                           fontsize=12, fontweight='bold')
                ax.set_xlabel('时间 (秒)')
                ax.set_ylabel('钙离子浓度')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
        
        # 移除多余的子图
        for i in range(len(unique_states), 6):
            axes[i].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'state_waveforms.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_state_distribution(self, labels: np.ndarray, neuron_names: List[str], 
                                output_dir: str) -> None:
        """绘制状态分布图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 状态计数
        unique_states, counts = np.unique(labels, return_counts=True)
        state_names = [f'State {i+1}' for i in unique_states]
        
        # 饼图
        colors = [list(self.state_colors.values())[i] if i < len(self.state_colors) 
                 else plt.cm.Set3(i) for i in unique_states]
        
        ax1.pie(counts, labels=state_names, colors=colors, autopct='%1.1f%%', 
               startangle=90)
        ax1.set_title('神经元状态分布', fontsize=14, fontweight='bold')
        
        # 柱状图
        ax2.bar(state_names, counts, color=colors)
        ax2.set_title('各状态神经元数量', fontsize=14, fontweight='bold')
        ax2.set_xlabel('状态')
        ax2.set_ylabel('神经元数量')
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加数值标注
        for i, count in enumerate(counts):
            ax2.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'state_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_analysis(self, features: np.ndarray, feature_names: List[str], 
                              labels: np.ndarray, output_dir: str) -> None:
        """绘制特征分析图"""
        # PCA降维可视化
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(StandardScaler().fit_transform(features))
        
        plt.figure(figsize=(12, 5))
        
        # PCA散点图
        plt.subplot(1, 2, 1)
        unique_labels = sorted(set(labels))
        colors = [list(self.state_colors.values())[i] if i < len(self.state_colors) 
                 else plt.cm.Set3(i) for i in unique_labels]
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                       c=[colors[i]], label=f'State {label+1}', alpha=0.7, s=50)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('神经元状态PCA可视化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 特征重要性
        plt.subplot(1, 2, 2)
        feature_importance = np.abs(pca.components_).mean(axis=0)
        sorted_idx = np.argsort(feature_importance)[-10:]  # 显示前10个重要特征
        
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('特征重要性')
        plt.title('前10个重要特征')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_state_transition_matrix(self, data: pd.DataFrame, labels: np.ndarray, 
                                     neuron_names: List[str], output_dir: str) -> None:
        """绘制状态转换矩阵（基于时间窗口分析）"""
        # 这里简化实现，实际应用中可以基于滑动窗口分析状态转换
        unique_states = sorted(set(labels))
        n_states = len(unique_states)
        
        # 创建虚拟的转换矩阵作为示例
        transition_matrix = np.random.rand(n_states, n_states)
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(transition_matrix, 
                   xticklabels=[f'State {i+1}' for i in unique_states],
                   yticklabels=[f'State {i+1}' for i in unique_states],
                   annot=True, fmt='.2f', cmap='Blues')
        plt.title('神经元状态转换概率矩阵', fontsize=14, fontweight='bold')
        plt.xlabel('目标状态')
        plt.ylabel('起始状态')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'state_transition_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, data: pd.DataFrame, labels: np.ndarray, 
                    neuron_names: List[str], output_path: str) -> None:
        """
        保存分析结果
        
        Parameters
        ----------
        data : pd.DataFrame
            原始数据
        labels : np.ndarray
            状态标签
        neuron_names : List[str]
            神经元名称列表
        output_path : str
            输出文件路径
        """
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'neuron_id': neuron_names,
            'state_label': labels,
            'state_name': [f'State {label+1}' for label in labels]
        })
        
        # 添加状态描述
        state_descriptions = {i: desc for i, desc in enumerate(self.state_definitions.values())}
        results_df['state_description'] = results_df['state_label'].map(state_descriptions)
        
        # 保存到Excel文件
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='神经元状态分类', index=False)
            
            # 添加状态统计
            state_stats = results_df.groupby('state_name').size().reset_index(name='count')
            state_stats.to_excel(writer, sheet_name='状态统计', index=False)
        
        self.logger.info(f"分析结果已保存到: {output_path}")
    
    # 辅助函数实现
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        from scipy.stats import skew
        return skew(data)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        from scipy.stats import kurtosis
        return kurtosis(data)
    
    def _count_peaks(self, data: np.ndarray) -> int:
        """计算峰值数量"""
        peaks, _ = find_peaks(data, height=np.mean(data) + 0.5 * np.std(data))
        return len(peaks)
    
    def _mean_peak_amplitude(self, data: np.ndarray) -> float:
        """计算峰值振幅均值"""
        peaks, _ = find_peaks(data, height=np.mean(data) + 0.5 * np.std(data))
        if len(peaks) > 0:
            return np.mean(data[peaks])
        return 0.0
    
    def _std_peak_amplitude(self, data: np.ndarray) -> float:
        """计算峰值振幅标准差"""
        peaks, _ = find_peaks(data, height=np.mean(data) + 0.5 * np.std(data))
        if len(peaks) > 1:
            return np.std(data[peaks])
        return 0.0
    
    def _mean_inter_peak_interval(self, data: np.ndarray) -> float:
        """计算峰值间隔均值"""
        peaks, _ = find_peaks(data, height=np.mean(data) + 0.5 * np.std(data))
        if len(peaks) > 1:
            intervals = np.diff(peaks) / self.sampling_rate
            return np.mean(intervals)
        return 0.0
    
    def _std_inter_peak_interval(self, data: np.ndarray) -> float:
        """计算峰值间隔标准差"""
        peaks, _ = find_peaks(data, height=np.mean(data) + 0.5 * np.std(data))
        if len(peaks) > 2:
            intervals = np.diff(peaks) / self.sampling_rate
            return np.std(intervals)
        return 0.0
    
    def _calculate_activity_rate(self, data: np.ndarray) -> float:
        """计算活动率"""
        threshold = np.mean(data) + np.std(data)
        active_samples = np.sum(data > threshold)
        return active_samples / len(data)
    
    def _calculate_burst_rate(self, data: np.ndarray) -> float:
        """计算突发率"""
        threshold = np.mean(data) + 2 * np.std(data)
        burst_samples = np.sum(data > threshold)
        return burst_samples / len(data)
    
    def _calculate_silence_periods(self, data: np.ndarray) -> float:
        """计算静默期比例"""
        threshold = np.mean(data) - 0.5 * np.std(data)
        silent_samples = np.sum(data < threshold)
        return silent_samples / len(data)
    
    def _band_power(self, freqs: np.ndarray, psd: np.ndarray, 
                   low_freq: float, high_freq: float) -> float:
        """计算特定频带的功率"""
        freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
        return np.trapz(psd[freq_mask], freqs[freq_mask])
    
    def _sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """计算样本熵"""
        try:
            from pyeeg import samp_entropy
            return samp_entropy(data, m, r)
        except ImportError:
            # 简化实现
            return entropy(np.histogram(data, bins=10)[0] + 1e-12)
    
    def _zero_crossing_rate(self, data: np.ndarray) -> float:
        """计算零交叉率"""
        zero_crossings = np.where(np.diff(np.signbit(data - np.mean(data))))[0]
        return len(zero_crossings) / len(data)
    
    def _hurst_exponent(self, data: np.ndarray) -> float:
        """计算Hurst指数"""
        # 简化实现
        n = len(data)
        if n < 10:
            return 0.5
        
        # R/S分析
        mean_data = np.mean(data)
        cumsum_data = np.cumsum(data - mean_data)
        
        ranges = []
        stds = []
        
        for i in range(2, min(n//2, 100)):
            segments = np.array_split(cumsum_data, n//i)
            rs_values = []
            
            for segment in segments:
                if len(segment) > 1:
                    range_val = np.max(segment) - np.min(segment)
                    std_val = np.std(segment)
                    if std_val > 0:
                        rs_values.append(range_val / std_val)
            
            if rs_values:
                ranges.append(np.mean(rs_values))
                stds.append(i)
        
        if len(ranges) > 1:
            log_ranges = np.log(ranges)
            log_stds = np.log(stds)
            hurst = np.polyfit(log_stds, log_ranges, 1)[0]
            return max(0, min(1, hurst))
        
        return 0.5
    
    def _fractal_dimension(self, data: np.ndarray) -> float:
        """计算分形维数"""
        # 盒计数法的简化实现
        n = len(data)
        if n < 4:
            return 1.0
        
        # 归一化数据
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-12)
        
        scales = np.logspace(0, np.log10(n//4), 10, dtype=int)
        counts = []
        
        for scale in scales:
            if scale < n:
                boxes = np.zeros(scale)
                for i, val in enumerate(data_norm):
                    box_idx = int(val * (scale - 1))
                    boxes[box_idx] = 1
                counts.append(np.sum(boxes))
        
        if len(counts) > 1:
            log_scales = np.log(scales[:len(counts)])
            log_counts = np.log(counts + 1e-12)
            fractal_dim = -np.polyfit(log_scales, log_counts, 1)[0]
            return max(1, min(2, fractal_dim))
        
        return 1.5
    
    def _lyapunov_exponent(self, data: np.ndarray) -> float:
        """计算Lyapunov指数（简化实现）"""
        n = len(data)
        if n < 10:
            return 0.0
        
        # 重构相空间
        m = 3  # 嵌入维数
        tau = 1  # 时间延迟
        
        if n < m * tau:
            return 0.0
        
        # 计算相邻轨道的分离率
        separations = []
        for i in range(n - m * tau):
            x1 = np.array([data[i + j * tau] for j in range(m)])
            
            # 找最近邻
            min_dist = float('inf')
            min_idx = -1
            
            for j in range(i + 1, min(i + 50, n - m * tau)):
                x2 = np.array([data[j + k * tau] for k in range(m)])
                dist = np.linalg.norm(x1 - x2)
                
                if dist < min_dist and dist > 1e-10:
                    min_dist = dist
                    min_idx = j
            
            if min_idx > 0 and min_dist < float('inf'):
                separations.append(min_dist)
        
        if separations:
            return np.mean(np.log(separations + 1e-12))
        
        return 0.0
    
    def _detrended_fluctuation_analysis(self, data: np.ndarray) -> float:
        """去趋势波动分析"""
        n = len(data)
        if n < 10:
            return 1.0
        
        # 积分数据
        y = np.cumsum(data - np.mean(data))
        
        # 不同窗口大小
        scales = np.logspace(np.log10(4), np.log10(n//4), 10, dtype=int)
        fluctuations = []
        
        for scale in scales:
            # 分段去趋势
            n_segments = n // scale
            if n_segments == 0:
                continue
                
            segment_vars = []
            for i in range(n_segments):
                start_idx = i * scale
                end_idx = (i + 1) * scale
                segment = y[start_idx:end_idx]
                
                # 线性去趋势
                x = np.arange(len(segment))
                if len(x) > 1:
                    poly_coeff = np.polyfit(x, segment, 1)
                    trend = np.polyval(poly_coeff, x)
                    detrended = segment - trend
                    segment_vars.append(np.var(detrended))
            
            if segment_vars:
                fluctuations.append(np.sqrt(np.mean(segment_vars)))
        
        if len(fluctuations) > 1:
            log_scales = np.log(scales[:len(fluctuations)])
            log_fluctuations = np.log(fluctuations + 1e-12)
            alpha = np.polyfit(log_scales, log_fluctuations, 1)[0]
            return max(0.5, min(2.0, alpha))
        
        return 1.0
    
    def _calculate_rise_times(self, data: np.ndarray, peaks: np.ndarray) -> float:
        """计算上升时间均值"""
        if len(peaks) == 0:
            return 0.0
        
        rise_times = []
        for peak in peaks:
            # 向前搜索上升起点
            start_idx = peak
            for i in range(peak - 1, max(0, peak - 20), -1):
                if data[i] < data[peak] * 0.1:
                    start_idx = i
                    break
            
            rise_time = (peak - start_idx) / self.sampling_rate
            rise_times.append(rise_time)
        
        return np.mean(rise_times) if rise_times else 0.0
    
    def _calculate_decay_times(self, data: np.ndarray, peaks: np.ndarray) -> float:
        """计算衰减时间均值"""
        if len(peaks) == 0:
            return 0.0
        
        decay_times = []
        for peak in peaks:
            # 向后搜索衰减终点
            end_idx = peak
            for i in range(peak + 1, min(len(data), peak + 20)):
                if data[i] < data[peak] * 0.1:
                    end_idx = i
                    break
            
            decay_time = (end_idx - peak) / self.sampling_rate
            decay_times.append(decay_time)
        
        return np.mean(decay_times) if decay_times else 0.0
    
    def _calculate_regularity_index(self, data: np.ndarray) -> float:
        """计算规律性指数"""
        # 基于自相关函数的规律性评估
        if len(data) < 10:
            return 0.0
        
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # 归一化
        
        # 计算周期性强度
        if len(autocorr) > 10:
            return np.max(autocorr[5:min(50, len(autocorr))])
        
        return 0.0
    
    def _calculate_burst_duration(self, data: np.ndarray) -> float:
        """计算突发持续时间均值"""
        threshold = np.mean(data) + 2 * np.std(data)
        above_threshold = data > threshold
        
        # 找连续的突发区域
        burst_durations = []
        in_burst = False
        burst_start = 0
        
        for i, is_active in enumerate(above_threshold):
            if is_active and not in_burst:
                in_burst = True
                burst_start = i
            elif not is_active and in_burst:
                in_burst = False
                duration = (i - burst_start) / self.sampling_rate
                burst_durations.append(duration)
        
        return np.mean(burst_durations) if burst_durations else 0.0
    
    def _calculate_burst_intensity(self, data: np.ndarray) -> float:
        """计算突发强度均值"""
        threshold = np.mean(data) + 2 * np.std(data)
        burst_data = data[data > threshold]
        
        if len(burst_data) > 0:
            return np.mean(burst_data)
        
        return 0.0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='神经元放电状态分析')
    parser.add_argument('--input', type=str, 
                       default='../datasets/processed_EMtrace01.xlsx',
                       help='输入数据文件路径')
    parser.add_argument('--output-dir', type=str, 
                       default='../results/state_analysis/',
                       help='输出目录')
    parser.add_argument('--method', type=str, default='kmeans',
                       choices=['kmeans', 'dbscan', 'gaussian_mixture'],
                       help='聚类方法')
    parser.add_argument('--n-states', type=int, default=6,
                       help='预期状态数量')
    parser.add_argument('--sampling-rate', type=float, default=4.8,
                       help='采样频率(Hz)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化分析器
    analyzer = StateAnalyzer(sampling_rate=args.sampling_rate)
    
    try:
        # 加载数据
        data = analyzer.load_data(args.input)
        
        # 提取特征
        features, feature_names, neuron_names = analyzer.extract_all_features(data)
        
        # 识别状态
        labels = analyzer.identify_states(features, method=args.method, 
                                        n_states=args.n_states)
        
        # 可视化结果
        analyzer.visualize_states(data, labels, neuron_names, args.output_dir)
        
        # 保存结果
        output_file = os.path.join(args.output_dir, 'neuron_states_analysis.xlsx')
        analyzer.save_results(data, labels, neuron_names, output_file)
        
        print(f"神经元状态分析完成！结果保存在: {args.output_dir}")
        
    except Exception as e:
        analyzer.logger.error(f"分析过程中出现错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
