#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版神经元放电状态分析模块（支持时间窗口动态状态分析）

该模块融合了StateClassifier中的先进方法，包括：
- 相空间重构技术，将时间序列转换为3D相空间流形
- 图卷积网络(GCN)进行状态分类
- 时序图构建方法
- 多特征融合分析
- **时间窗口动态状态分析**：一个神经元可以在不同时间段表现出不同的放电状态

基于4种典型神经元放电状态：
- State I: 高频连续振荡状态
- State II: 规律性脉冲放电状态  
- State III: 间歇性突发状态
- State IV: 不规律波动状态

作者：Clade 4
日期：2025年5月23日
版本：2.0 - 支持时间窗口动态分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.signal import find_peaks, peak_widths, welch
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# 尝试导入UMAP
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP未安装，将使用PCA替代降维可视化")

import warnings
warnings.filterwarnings('ignore')
import os
import logging
import datetime
import sys
from typing import Dict, List, Tuple, Optional, Any
import argparse

# 尝试导入图神经网络相关库
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch Geometric未安装，将使用传统机器学习方法")

# 导入额外的可视化库
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly未安装，将只使用matplotlib进行可视化")

# 强力抑制所有matplotlib字体警告
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 设置matplotlib字体和警告配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False  # 完全禁用Unicode负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['mathtext.fontset'] = 'dejavusans'  # 使用DejaVu字体集
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['text.usetex'] = False  # 禁用LaTeX渲染

# 禁用所有相关警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', module='matplotlib')

# 设置matplotlib日志级别
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.ERROR)


class TimeWindowAnalyzer:
    """
    时间窗口分析器类
    
    支持将神经元信号分割成多个时间窗口，分析每个窗口的状态
    """
    
    def __init__(self, window_duration: float = 30.0, overlap_ratio: float = 0.5, 
                 sampling_rate: float = 4.8):
        """
        初始化时间窗口分析器
        
        Parameters
        ----------
        window_duration : float
            时间窗口长度（秒），默认30秒
        overlap_ratio : float
            窗口重叠比例，默认0.5（50%重叠）
        sampling_rate : float
            采样频率，默认4.8Hz
        """
        self.window_duration = window_duration
        self.overlap_ratio = overlap_ratio
        self.sampling_rate = sampling_rate
        self.window_size = int(window_duration * sampling_rate)
        self.step_size = int(self.window_size * (1 - overlap_ratio))
    
    def split_signal_to_windows(self, signal: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        将信号分割成多个时间窗口
        
        Parameters
        ----------
        signal : np.ndarray
            输入信号
            
        Returns
        -------
        Tuple[List[np.ndarray], List[float]]
            窗口信号列表, 窗口起始时间列表
        """
        windows = []
        start_times = []
        
        # 确保窗口大小不超过信号长度
        if len(signal) < self.window_size:
            # 如果信号太短，直接返回整个信号作为一个窗口
            windows.append(signal)
            start_times.append(0.0)
            return windows, start_times
        
        for start_idx in range(0, len(signal) - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size
            window = signal[start_idx:end_idx]
            start_time = start_idx / self.sampling_rate
            
            windows.append(window)
            start_times.append(start_time)
        
        return windows, start_times
    
    def extract_window_features(self, window: np.ndarray, feature_extractor) -> Dict[str, float]:
        """
        提取单个窗口的特征
        
        Parameters
        ----------
        window : np.ndarray
            时间窗口信号
        feature_extractor : object
            特征提取器对象
            
        Returns
        -------
        Dict[str, float]
            窗口特征字典
        """
        # 提取传统特征
        temporal_features = feature_extractor.extract_temporal_features(window)
        frequency_features = feature_extractor.extract_frequency_features(window)
        nonlinear_features = feature_extractor.extract_nonlinear_features(window)
        morphological_features = feature_extractor.extract_morphological_features(window)
        
        # 提取相空间特征
        phase_space_features = feature_extractor.extract_phase_space_features(window)
        
        # 提取图特征
        graph_features = feature_extractor.extract_graph_features(window)
        
        # 合并所有特征
        all_features = {
            **temporal_features, 
            **frequency_features,
            **nonlinear_features, 
            **morphological_features,
            **phase_space_features,
            **graph_features
        }
        
        return all_features


class PhaseSpaceAnalyzer:
    """
    相空间分析器类
    
    实现时间序列的相空间重构，用于提取非线性动力学特征
    """
    
    def __init__(self, embedding_dim: int = 3, max_tau: int = 50):
        """
        初始化相空间分析器
        
        Parameters
        ----------
        embedding_dim : int
            嵌入维度，默认为3
        max_tau : int
            最大时间延迟，默认为50
        """
        self.embedding_dim = embedding_dim
        self.max_tau = max_tau
    
    def mutual_information(self, x: np.ndarray, y: np.ndarray, bins: int = 50) -> float:
        """
        计算两个信号之间的互信息
        
        Parameters
        ----------
        x, y : np.ndarray
            输入信号
        bins : int
            直方图分箱数
            
        Returns
        -------
        float
            互信息值
        """
        # 离散化信号
        x_discrete = np.digitize(x, np.histogram(x, bins)[1])
        y_discrete = np.digitize(y, np.histogram(y, bins)[1])
        
        return mutual_info_score(x_discrete, y_discrete)
    
    def find_optimal_tau(self, time_series: np.ndarray) -> int:
        """
        通过互信息法寻找最优时间延迟
        
        Parameters
        ----------
        time_series : np.ndarray
            输入时间序列
            
        Returns
        -------
        int
            最优时间延迟
        """
        mi_values = []
        
        for tau in range(1, min(self.max_tau, len(time_series)//4)):
            if tau < len(time_series):
                x = time_series[:-tau]
                y = time_series[tau:]
                mi = self.mutual_information(x, y)
                mi_values.append(mi)
        
        if not mi_values:
            return 1
        
        # 寻找第一个局部最小值
        mi_values = np.array(mi_values)
        peaks, _ = find_peaks(-mi_values)
        
        if len(peaks) > 0:
            return peaks[0] + 1
        else:
            # 如果没有找到局部最小值，返回互信息最小的tau
            return np.argmin(mi_values) + 1
    
    def phase_space_reconstruction(self, time_series: np.ndarray, 
                                 tau: Optional[int] = None) -> np.ndarray:
        """
        进行相空间重构
        
        Parameters
        ----------
        time_series : np.ndarray
            输入时间序列
        tau : int, optional
            时间延迟，如果未指定则自动计算
            
        Returns
        -------
        np.ndarray
            相空间坐标，形状为 (n_points, embedding_dim)
        """
        if tau is None:
            tau = self.find_optimal_tau(time_series)
        
        # 标准化时间序列
        ts_normalized = (time_series - np.mean(time_series)) / np.std(time_series)
        
        n = len(ts_normalized)
        m = self.embedding_dim
        
        # 检查是否有足够的数据点进行重构
        if n < (m - 1) * tau + 1:
            return np.array([])
        
        # 相空间重构
        phase_space = np.zeros((n - (m - 1) * tau, m))
        
        for i in range(m):
            start_idx = i * tau
            end_idx = start_idx + (n - (m - 1) * tau)
            phase_space[:, i] = ts_normalized[start_idx:end_idx]
        
        return phase_space
    
    def extract_phase_space_features(self, phase_space: np.ndarray) -> Dict[str, float]:
        """
        从相空间中提取特征
        
        Parameters
        ----------
        phase_space : np.ndarray
            相空间坐标
            
        Returns
        -------
        Dict[str, float]
            相空间特征字典
        """
        if phase_space.size == 0:
            return {
                'ps_volume': 0.0,
                'ps_surface_area': 0.0,
                'ps_max_distance': 0.0,
                'ps_mean_distance': 0.0,
                'ps_trajectory_length': 0.0,
                'ps_lyapunov_estimate': 0.0
            }
        
        features = {}
        
        # 计算相空间体积（使用凸包）
        try:
            from scipy.spatial import ConvexHull
            if len(phase_space) > 3 and phase_space.shape[1] >= 3:
                hull = ConvexHull(phase_space)
                features['ps_volume'] = hull.volume
                features['ps_surface_area'] = hull.area
            else:
                features['ps_volume'] = 0.0
                features['ps_surface_area'] = 0.0
        except:
            features['ps_volume'] = 0.0
            features['ps_surface_area'] = 0.0
        
        # 计算最大距离
        if len(phase_space) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(phase_space)
            features['ps_max_distance'] = np.max(distances)
            features['ps_mean_distance'] = np.mean(distances)
        else:
            features['ps_max_distance'] = 0.0
            features['ps_mean_distance'] = 0.0
        
        # 计算轨迹长度
        if len(phase_space) > 1:
            trajectory_segments = np.diff(phase_space, axis=0)
            segment_lengths = np.sqrt(np.sum(trajectory_segments**2, axis=1))
            features['ps_trajectory_length'] = np.sum(segment_lengths)
        else:
            features['ps_trajectory_length'] = 0.0
        
        # 估计最大Lyapunov指数
        features['ps_lyapunov_estimate'] = self._estimate_lyapunov(phase_space)
        
        return features
    
    def _estimate_lyapunov(self, phase_space: np.ndarray) -> float:
        """估计最大Lyapunov指数"""
        if len(phase_space) < 10:
            return 0.0
        
        try:
            n_points = len(phase_space)
            lyap_estimates = []
            
            for i in range(min(50, n_points-5)):
                # 寻找最近邻
                distances = np.sqrt(np.sum((phase_space - phase_space[i])**2, axis=1))
                # 排除自身和过近的点
                valid_indices = np.where((distances > 1e-10) & (distances < np.percentile(distances, 10)))[0]
                
                if len(valid_indices) > 0:
                    nearest_idx = valid_indices[np.argmin(distances[valid_indices])]
                    
                    # 计算后续几个时间步的分离
                    max_steps = min(10, n_points - max(i, nearest_idx) - 1)
                    if max_steps > 0:
                        separation_growth = []
                        for step in range(1, max_steps):
                            if i + step < n_points and nearest_idx + step < n_points:
                                sep = np.sqrt(np.sum((phase_space[i + step] - phase_space[nearest_idx + step])**2))
                                if sep > 1e-10:
                                    separation_growth.append(np.log(sep))
                        
                        if len(separation_growth) > 1:
                            # 使用线性回归估计增长率
                            time_steps = np.arange(len(separation_growth))
                            lyap_est = np.polyfit(time_steps, separation_growth, 1)[0]
                            lyap_estimates.append(lyap_est)
            
            return np.mean(lyap_estimates) if lyap_estimates else 0.0
            
        except:
            return 0.0


class TemporalGraphBuilder:
    """
    时序图构建器
    
    将时间序列数据转换为图数据结构
    """
    
    def __init__(self, window_size: int = 100, stride: int = 50):
        """
        初始化时序图构建器
        
        Parameters
        ----------
        window_size : int
            滑动窗口大小
        stride : int
            滑动步长
        """
        self.window_size = window_size
        self.stride = stride
    
    def build_temporal_graph(self, time_series: np.ndarray) -> List[np.ndarray]:
        """
        构建时序图
        
        Parameters
        ----------
        time_series : np.ndarray
            输入时间序列
            
        Returns
        -------
        List[np.ndarray]
            图节点特征列表
        """
        graphs = []
        
        for start in range(0, len(time_series) - self.window_size + 1, self.stride):
            window = time_series[start:start + self.window_size]
            
            # 每个时间点作为一个节点，值作为节点特征
            node_features = window.reshape(-1, 1)
            
            # 可以添加更多节点特征，如一阶差分、二阶差分等
            if len(window) > 1:
                first_diff = np.diff(window)
                first_diff = np.append(first_diff, first_diff[-1])  # 保持长度一致
                node_features = np.column_stack([node_features, first_diff.reshape(-1, 1)])
            
            if len(window) > 2:
                second_diff = np.diff(window, n=2)
                second_diff = np.append(second_diff, [second_diff[-1], second_diff[-1]])
                node_features = np.column_stack([node_features, second_diff.reshape(-1, 1)])
            
            graphs.append(node_features)
        
        return graphs
    
    def build_edges(self, graph_size: int) -> np.ndarray:
        """
        构建图的边连接
        
        Parameters
        ----------
        graph_size : int
            图中节点数量
            
        Returns
        -------
        np.ndarray
            边连接矩阵，形状为 (2, num_edges)
        """
        # 构建时序连接：每个节点连接到下一个节点
        edges = []
        for i in range(graph_size - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])  # 双向连接
        
        # 可以添加更多连接模式，如间隔连接
        # for i in range(graph_size - 2):
        #     edges.append([i, i + 2])
        #     edges.append([i + 2, i])
        
        return np.array(edges).T if edges else np.array([[], []]).astype(int)


class EnhancedStateAnalyzer:
    """
    增强版神经元状态分析器（支持时间窗口动态分析）
    
    融合了相空间重构、图卷积网络和传统机器学习方法
    现在支持时间窗口分析，一个神经元可以在不同时间段表现出不同的放电状态
    """
    
    def __init__(self, sampling_rate: float = 4.8, window_duration: float = 30.0, 
                 overlap_ratio: float = 0.5, logger: Optional[logging.Logger] = None):
        """
        初始化增强版状态分析器
        
        Parameters
        ----------
        sampling_rate : float
            采样频率，默认为4.8Hz
        window_duration : float
            时间窗口长度（秒），默认30秒
        overlap_ratio : float
            窗口重叠比例，默认0.5（50%重叠）
        logger : logging.Logger, optional
            日志记录器，默认为None
        """
        self.sampling_rate = sampling_rate
        self.window_duration = window_duration
        self.overlap_ratio = overlap_ratio
        self.logger = logger or self._setup_logger()
        
        # 初始化子模块
        self.phase_analyzer = PhaseSpaceAnalyzer()
        self.graph_builder = TemporalGraphBuilder()
        self.time_window_analyzer = TimeWindowAnalyzer(
            window_duration=window_duration,
            overlap_ratio=overlap_ratio,
            sampling_rate=sampling_rate
        )
        
        # 状态定义
        self.state_definitions = {
            'State I': '高频连续振荡状态',
            'State II': '规律性脉冲放电状态',
            'State III': '间歇性突发状态',
            'State IV': '不规律波动状态'
        }
        
        # 状态颜色映射
        self.state_colors = {
            'State I': '#8B0000',    # 深红色
            'State II': '#FFD700',   # 金黄色
            'State III': '#228B22',  # 森林绿
            'State IV': '#4169E1'    # 皇家蓝
        }
        
        self.logger.info(f"增强版神经元状态分析器初始化完成 - 时间窗口: {window_duration}秒, 重叠率: {overlap_ratio}")
    
    def _setup_logger(self, output_dir: Optional[str] = None) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('EnhancedStateAnalyzer')
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
        log_file = os.path.join(output_dir, f"enhanced_state_analysis_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"日志文件创建于: {log_file}")
        return logger
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载神经元钙离子浓度数据"""
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
    
    def extract_windowed_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        """
        基于时间窗口提取特征
        
        Parameters
        ----------
        data : pd.DataFrame
            神经元钙离子浓度数据
            
        Returns
        -------
        Tuple[np.ndarray, List[str], pd.DataFrame]
            特征矩阵, 特征名称列表, 窗口信息DataFrame
        """
        self.logger.info("开始基于时间窗口提取神经元特征...")
        
        # 识别神经元列
        neuron_columns = [col for col in data.columns 
                         if col.startswith('n') and col[1:].isdigit()]
        
        feature_list = []
        feature_names = None
        window_info_list = []
        
        for neuron in neuron_columns:
            signal_data = data[neuron].values
            
            # 将信号分割成时间窗口
            windows, start_times = self.time_window_analyzer.split_signal_to_windows(signal_data)
            
            for window_idx, (window, start_time) in enumerate(zip(windows, start_times)):
                # 提取窗口特征
                window_features = self.time_window_analyzer.extract_window_features(window, self)
                
                if feature_names is None:
                    feature_names = list(window_features.keys())
                
                feature_list.append(list(window_features.values()))
                
                # 记录窗口信息
                window_info_list.append({
                    'neuron_id': neuron,
                    'window_idx': window_idx,
                    'start_time': start_time,
                    'end_time': start_time + self.window_duration,
                    'duration': self.window_duration
                })
        
        features_matrix = np.array(feature_list)
        window_info_df = pd.DataFrame(window_info_list)
        
        self.logger.info(f"时间窗口特征提取完成，共提取 {len(feature_names)} 个特征，"
                        f"{len(neuron_columns)} 个神经元，{len(window_info_df)} 个时间窗口")
        
        return features_matrix, feature_names, window_info_df

    def analyze_temporal_states(self, data: pd.DataFrame, method: str = 'ensemble', 
                              n_states: int = 4) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        进行时间窗口状态分析
        
        Parameters
        ----------
        data : pd.DataFrame
            神经元钙离子浓度数据
        method : str
            分类方法
        n_states : int
            预期状态数量
            
        Returns
        -------
        Tuple[np.ndarray, pd.DataFrame]
            状态标签数组, 包含详细信息的结果DataFrame
        """
        self.logger.info("开始时间窗口状态分析...")
        
        # 提取时间窗口特征
        features, feature_names, window_info_df = self.extract_windowed_features(data)
        
        # 状态识别
        labels = self.identify_states_enhanced(features, method=method, n_states=n_states)
        
        # 创建结果DataFrame
        results_df = window_info_df.copy()
        results_df['state_label'] = labels
        results_df['state_name'] = [f'State {label+1}' for label in labels]
        
        # 添加状态描述
        state_descriptions = {i: desc for i, desc in enumerate(self.state_definitions.values())}
        results_df['state_description'] = results_df['state_label'].map(state_descriptions)
        
        # 计算状态统计信息
        self._calculate_temporal_statistics(results_df)
        
        return labels, results_df

    def _calculate_temporal_statistics(self, results_df: pd.DataFrame) -> None:
        """计算时间相关的统计信息"""
        self.logger.info("计算时间统计信息...")
        
        # 每个神经元的状态分布
        neuron_stats = results_df.groupby('neuron_id')['state_label'].agg(['count', 'nunique']).reset_index()
        neuron_stats.columns = ['neuron_id', 'total_windows', 'unique_states']
        
        # 每种状态的时间占比
        state_stats = results_df.groupby('state_name').size().reset_index(name='window_count')
        state_stats['time_percentage'] = state_stats['window_count'] / len(results_df) * 100
        
        self.logger.info(f"状态分布统计:")
        for _, row in state_stats.iterrows():
            self.logger.info(f"  {row['state_name']}: {row['window_count']} 窗口 ({row['time_percentage']:.1f}%)")
        
        # 状态转换统计
        self._analyze_state_transitions(results_df)
    
    def _analyze_state_transitions(self, results_df: pd.DataFrame) -> None:
        """分析状态转换模式"""
        self.logger.info("分析状态转换模式...")
        
        transition_counts = {}
        
        for neuron in results_df['neuron_id'].unique():
            neuron_data = results_df[results_df['neuron_id'] == neuron].sort_values('window_idx')
            states = neuron_data['state_label'].values
            
            for i in range(len(states) - 1):
                from_state = states[i]
                to_state = states[i + 1]
                transition = (from_state, to_state)
                
                if transition not in transition_counts:
                    transition_counts[transition] = 0
                transition_counts[transition] += 1
        
        # 输出主要转换模式
        sorted_transitions = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)
        self.logger.info("主要状态转换模式:")
        for (from_state, to_state), count in sorted_transitions[:10]:
            self.logger.info(f"  State {from_state+1} → State {to_state+1}: {count} 次")

    def identify_states_enhanced(self, features: np.ndarray, method: str = 'ensemble', 
                               n_states: int = 4) -> np.ndarray:
        """
        使用增强方法识别神经元放电状态
        
        Parameters
        ----------
        features : np.ndarray
            特征矩阵
        method : str
            分类方法，可选'kmeans', 'dbscan', 'ensemble', 'gcn'
        n_states : int
            预期状态数量，默认为4
            
        Returns
        -------
        np.ndarray
            状态标签数组
        """
        self.logger.info(f"使用 {method} 方法识别 {n_states} 种神经元状态...")
        
        # 特征标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        if method == 'ensemble':
            # 集成多种方法
            labels = self._ensemble_clustering(features_scaled, n_states)
            
        elif method == 'gcn' and TORCH_AVAILABLE:
            # 使用图卷积网络
            labels = self._gcn_clustering(features_scaled, n_states)
            
        elif method == 'kmeans':
            clusterer = KMeans(n_clusters=n_states, random_state=42, n_init=10)
            labels = clusterer.fit_predict(features_scaled)
            
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=2)
            labels = clusterer.fit_predict(features_scaled)
            n_states = len(set(labels)) - (1 if -1 in labels else 0)
            
        else:
            raise ValueError(f"不支持的聚类方法: {method}")
        
        # 评估聚类质量
        if len(set(labels)) > 1:
            silhouette = silhouette_score(features_scaled, labels)
            calinski_harabasz = calinski_harabasz_score(features_scaled, labels)
            
            self.logger.info(f"聚类质量评估 - 轮廓系数: {silhouette:.3f}, "
                           f"Calinski-Harabasz指数: {calinski_harabasz:.3f}")
        
        self.logger.info(f"状态识别完成，识别出 {len(set(labels))} 种状态")
        
        return labels
    
    def _ensemble_clustering(self, features: np.ndarray, n_states: int) -> np.ndarray:
        """集成聚类方法"""
        # 使用多种聚类方法
        kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(features)
        
        # 使用不同参数的KMeans
        kmeans2 = KMeans(n_clusters=n_states, random_state=123, n_init=10)
        kmeans2_labels = kmeans2.fit_predict(features)
        
        # 计算轮廓系数选择最佳结果
        score1 = silhouette_score(features, kmeans_labels)
        score2 = silhouette_score(features, kmeans2_labels)
        
        if score1 >= score2:
            return kmeans_labels
        else:
            return kmeans2_labels
    
    def _gcn_clustering(self, features: np.ndarray, n_states: int) -> np.ndarray:
        """使用图卷积网络进行聚类（简化实现）"""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch Geometric未安装，回退到KMeans方法")
            clusterer = KMeans(n_clusters=n_states, random_state=42, n_init=10)
            return clusterer.fit_predict(features)
        
        # 这里可以实现更复杂的GCN聚类
        # 由于需要大量的图数据预处理，这里简化为传统聚类
        clusterer = KMeans(n_clusters=n_states, random_state=42, n_init=10)
        return clusterer.fit_predict(features)
    
    def visualize_phase_space(self, data: pd.DataFrame, labels: np.ndarray, 
                            neuron_names: List[str], output_dir: str) -> None:
        """可视化相空间重构结果"""
        self.logger.info("生成相空间可视化...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 选择几个代表性神经元进行相空间可视化
        unique_states = sorted(set(labels))
        
        fig = plt.figure(figsize=(16, 12))
        
        plot_idx = 1
        for state in unique_states:
            if plot_idx > 4:  # 最多显示4个状态
                break
                
            # 找到该状态的神经元
            state_neurons = [neuron_names[i] for i, label in enumerate(labels) if label == state]
            
            if state_neurons:
                # 随机选择一个代表性神经元
                selected_neuron = np.random.choice(state_neurons)
                signal_data = data[selected_neuron].values
                
                # 进行相空间重构
                phase_space = self.phase_analyzer.phase_space_reconstruction(signal_data)
                
                if phase_space.size > 0 and phase_space.shape[1] >= 3:
                    # 3D相空间图
                    ax = fig.add_subplot(2, 2, plot_idx, projection='3d')
                    
                    # 使用状态颜色
                    color = list(self.state_colors.values())[state] if state < len(self.state_colors) else 'blue'
                    
                    ax.plot(phase_space[:, 0], phase_space[:, 1], phase_space[:, 2], 
                           color=color, alpha=0.7, linewidth=1)
                    ax.scatter(phase_space[0, 0], phase_space[0, 1], phase_space[0, 2], 
                             color='red', s=50, label='Start')
                    ax.scatter(phase_space[-1, 0], phase_space[-1, 1], phase_space[-1, 2], 
                             color='black', s=50, label='End')
                    
                    ax.set_title(f'State {state+1}: {selected_neuron}\nPhase Space Reconstruction', 
                               fontsize=12, fontweight='bold')
                    ax.set_xlabel('X(t)')
                    ax.set_ylabel('X(t-τ)')
                    ax.set_zlabel('X(t-2τ)')
                    ax.legend()
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'phase_space_reconstruction.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_enhanced_states(self, data: pd.DataFrame, labels: np.ndarray, 
                                neuron_names: List[str], output_dir: str = '../results') -> None:
        """增强版状态可视化"""
        self.logger.info("开始增强版神经元状态可视化...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 绘制典型状态波形（保持原有功能）
        self._plot_state_waveforms(data, labels, neuron_names, output_dir)
        
        # 2. 绘制状态分布
        self._plot_state_distribution(labels, neuron_names, output_dir)
        
        # 3. 绘制相空间可视化
        self.visualize_phase_space(data, labels, neuron_names, output_dir)
        
        # 4. 绘制增强特征分析
        features, feature_names, _ = self.extract_comprehensive_features(data)
        self._plot_enhanced_feature_analysis(features, feature_names, labels, output_dir)
        
        # 5. 新增可视化功能
        self._plot_state_heatmap(data, labels, neuron_names, output_dir)
        self._plot_temporal_dynamics(data, labels, neuron_names, output_dir)
        self._plot_frequency_analysis(data, labels, neuron_names, output_dir)
        self._plot_correlation_matrix(data, labels, neuron_names, output_dir)
        self._plot_state_transitions(data, labels, neuron_names, output_dir)
        
        # 6. 如果可用，生成交互式可视化
        if PLOTLY_AVAILABLE:
            self._create_interactive_visualizations(data, labels, neuron_names, features, feature_names, output_dir)
        
        self.logger.info("增强版状态可视化完成")
    
    def _plot_enhanced_feature_analysis(self, features: np.ndarray, feature_names: List[str], 
                                      labels: np.ndarray, output_dir: str) -> None:
        """绘制增强特征分析图"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # PCA和UMAP降维可视化
            pca = PCA(n_components=2)
            features_scaled = StandardScaler().fit_transform(features)
            features_pca = pca.fit_transform(features_scaled)
            
            # 尝试UMAP降维
            if UMAP_AVAILABLE:
                umap_reducer = UMAP(n_components=2, random_state=42)
                features_umap = umap_reducer.fit_transform(features_scaled)
            else:
                features_umap = features_pca  # 如果UMAP不可用，使用PCA结果
            
            plt.figure(figsize=(20, 8))
            
            unique_labels = sorted(set(labels))
            colors = [list(self.state_colors.values())[i] if i < len(self.state_colors) 
                     else plt.cm.Set3(i) for i in unique_labels]
            
            # PCA可视化
            plt.subplot(1, 3, 1)
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                           c=[colors[i]], label=f'State {label+1}', alpha=0.7, s=50)
            
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.title('PCA Dimensionality Reduction')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # UMAP可视化
            plt.subplot(1, 3, 2)
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(features_umap[mask, 0], features_umap[mask, 1], 
                           c=[colors[i]], label=f'State {label+1}', alpha=0.7, s=50)
            
            plt.xlabel('UMAP 1' if UMAP_AVAILABLE else 'PC1')
            plt.ylabel('UMAP 2' if UMAP_AVAILABLE else 'PC2')
            plt.title('UMAP Dimensionality Reduction' if UMAP_AVAILABLE else 'PCA Dimensionality Reduction (UMAP fallback)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 特征重要性分析
            plt.subplot(1, 3, 3)
            
            # 使用随机森林评估特征重要性
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(features_scaled, labels)
            
            feature_importance = rf.feature_importances_
            sorted_idx = np.argsort(feature_importance)[-15:]  # 显示前15个重要特征
            
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
            plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Important Features (Random Forest)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'enhanced_feature_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    # 保持原有的可视化方法
    def _plot_state_waveforms(self, data: pd.DataFrame, labels: np.ndarray, 
                             neuron_names: List[str], output_dir: str) -> None:
        """绘制每种状态的典型波形"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        unique_states = sorted(set(labels))
        
        for i, state in enumerate(unique_states):
            if i >= 4:  # 最多显示4种状态
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
                
                # 修改状态描述访问方式
                state_desc = list(self.state_definitions.values())[state] if state < len(self.state_definitions) else "Unknown State"
                ax.set_title(f'State {state + 1}: {state_desc}',
                           fontsize=12, fontweight='bold')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Calcium Concentration')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
        
        # 移除多余的子图（如果状态数少于4）
        for i in range(len(unique_states), 4):
            axes[i].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'enhanced_state_waveforms.png'), 
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
        ax1.set_title('Neuron State Distribution', fontsize=14, fontweight='bold')
        
        # 柱状图
        ax2.bar(state_names, counts, color=colors)
        ax2.set_title('Number of Neurons per State', fontsize=14, fontweight='bold')
        ax2.set_xlabel('State')
        ax2.set_ylabel('Number of Neurons')
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加数值标注
        for i, count in enumerate(counts):
            ax2.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'enhanced_state_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_enhanced_results(self, data: pd.DataFrame, labels: np.ndarray, 
                            neuron_names: List[str], features: np.ndarray,
                            feature_names: List[str], output_path: str) -> None:
        """保存增强分析结果"""
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'neuron_id': neuron_names,
            'state_label': labels,
            'state_name': [f'State {label+1}' for label in labels]
        })
        
        # 添加状态描述
        state_descriptions = {i: desc for i, desc in enumerate(self.state_definitions.values())}
        results_df['state_description'] = results_df['state_label'].map(state_descriptions)
        
        # 创建特征DataFrame
        features_df = pd.DataFrame(features, columns=feature_names)
        features_df['neuron_id'] = neuron_names
        features_df['state_label'] = labels
        
        # 保存到Excel文件
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Neuron State Classification', index=False)
            
            # 添加状态统计
            state_stats = results_df.groupby('state_name').size().reset_index(name='count')
            state_stats.to_excel(writer, sheet_name='State Statistics', index=False)
            
            # 添加特征数据
            features_df.to_excel(writer, sheet_name='Feature Data', index=False)
            
            # 添加特征重要性分析
            if len(set(labels)) > 1:
                from sklearn.ensemble import RandomForestClassifier
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(features, labels)
                
                importance_df = pd.DataFrame({
                    'feature_name': feature_names,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_df.to_excel(writer, sheet_name='Feature Importance', index=False)
        
        self.logger.info(f"增强分析结果已保存到: {output_path}")
    
    # 保持所有原有的辅助函数
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
    
    def _sample_entropy(self, data: np.ndarray) -> float:
        """计算样本熵"""
        try:
            from pyeeg import samp_entropy
            return samp_entropy(data, m=2, r=0.2)
        except ImportError:
            # 简化实现
            return entropy(np.histogram(data, bins=10)[0] + 1e-12)
    
    def _zero_crossing_rate(self, data: np.ndarray) -> float:
        """计算零交叉率"""
        zero_crossings = np.where(np.diff(np.signbit(data - np.mean(data))))[0]
        return len(zero_crossings) / len(data)
    
    def _hurst_exponent(self, data: np.ndarray) -> float:
        """计算Hurst指数"""
        # 简化实现（保持原有代码）
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
        # 保持原有实现
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
            counts_array = np.array(counts)  # 转换为numpy数组
            log_counts = np.log(counts_array + 1e-12)
            fractal_dim = -np.polyfit(log_scales, log_counts, 1)[0]
            return max(1, min(2, fractal_dim))
        
        return 1.5
    
    def _lyapunov_exponent(self, data: np.ndarray) -> float:
        """计算Lyapunov指数（简化实现）"""
        # 保持原有实现
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
            separations_array = np.array(separations)  # 转换为numpy数组
            return np.mean(np.log(separations_array + 1e-12))
        
        return 0.0
    
    def _detrended_fluctuation_analysis(self, data: np.ndarray) -> float:
        """去趋势波动分析"""
        # 保持原有实现
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
            fluctuations_array = np.array(fluctuations)  # 转换为numpy数组
            log_fluctuations = np.log(fluctuations_array + 1e-12)
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

    def _plot_state_heatmap(self, data: pd.DataFrame, labels: np.ndarray, 
                           neuron_names: List[str], output_dir: str) -> None:
        """绘制状态热图，显示不同神经元在时间上的状态分布"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.logger.info("生成状态热图...")
            
            # 创建状态矩阵（简化版，基于神经元的状态标签）
            unique_states = sorted(set(labels))
            state_matrix = np.zeros((len(unique_states), len(neuron_names)))
            
            for i, state in enumerate(unique_states):
                state_neurons = [j for j, label in enumerate(labels) if label == state]
                for j in state_neurons:
                    state_matrix[i, j] = 1
            
            plt.figure(figsize=(15, 8))
            sns.heatmap(state_matrix, 
                       xticklabels=[f'{name}' for name in neuron_names[::max(1, len(neuron_names)//20)]], 
                       yticklabels=[f'State {s+1}' for s in unique_states],
                       cmap='YlOrRd', 
                       cbar_kws={'label': 'State Assignment'})
            
            plt.title('Neuron State Assignment Heatmap', fontsize=16, fontweight='bold')
            plt.xlabel('Neurons')
            plt.ylabel('States')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'state_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_temporal_dynamics(self, data: pd.DataFrame, labels: np.ndarray, 
                               neuron_names: List[str], output_dir: str) -> None:
        """绘制时间动态特性分析"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.logger.info("生成时间动态特性分析...")
            
            unique_states = sorted(set(labels))
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 每种状态的平均活动强度随时间变化
        ax1 = axes[0, 0]
        for state in unique_states:
            state_neurons = [neuron_names[i] for i, label in enumerate(labels) if label == state]
            if state_neurons:
                # 计算该状态神经元的平均信号
                state_signals = [data[neuron].values for neuron in state_neurons]
                mean_signal = np.mean(state_signals, axis=0)
                
                time = np.arange(len(mean_signal)) / self.sampling_rate
                color = list(self.state_colors.values())[state]
                ax1.plot(time, mean_signal, color=color, label=f'State {state+1}', alpha=0.8)
        
        ax1.set_title('Average Activity Over Time by State', fontweight='bold')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Average Calcium Concentration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 状态活动强度的箱线图
        ax2 = axes[0, 1]
        state_intensities = []
        state_labels_plot = []
        
        for state in unique_states:
            state_neurons = [neuron_names[i] for i, label in enumerate(labels) if label == state]
            for neuron in state_neurons:
                signal = data[neuron].values
                state_intensities.append(np.mean(signal))
                state_labels_plot.append(f'State {state+1}')
        
        df_intensity = pd.DataFrame({'State': state_labels_plot, 'Intensity': state_intensities})
        sns.boxplot(data=df_intensity, x='State', y='Intensity', ax=ax2)
        ax2.set_title('Activity Intensity Distribution by State', fontweight='bold')
        ax2.set_ylabel('Mean Calcium Concentration')
        
        # 3. 峰值频率分析
        ax3 = axes[1, 0]
        peak_frequencies = []
        state_labels_peaks = []
        
        for state in unique_states:
            state_neurons = [neuron_names[i] for i, label in enumerate(labels) if label == state]
            for neuron in state_neurons:
                signal = data[neuron].values
                peaks, _ = find_peaks(signal, height=np.mean(signal) + np.std(signal))
                peak_freq = len(peaks) / (len(signal) / self.sampling_rate)  # peaks per second
                peak_frequencies.append(peak_freq)
                state_labels_peaks.append(f'State {state+1}')
        
        df_peaks = pd.DataFrame({'State': state_labels_peaks, 'Peak_Frequency': peak_frequencies})
        sns.boxplot(data=df_peaks, x='State', y='Peak_Frequency', ax=ax3)
        ax3.set_title('Peak Frequency by State', fontweight='bold')
        ax3.set_ylabel('Peaks per Second')
        
        # 4. 变异系数分析
        ax4 = axes[1, 1]
        cvs = []
        state_labels_cv = []
        
        for state in unique_states:
            state_neurons = [neuron_names[i] for i, label in enumerate(labels) if label == state]
            for neuron in state_neurons:
                signal = data[neuron].values
                cv = np.std(signal) / np.mean(signal) if np.mean(signal) > 0 else 0
                cvs.append(cv)
                state_labels_cv.append(f'State {state+1}')
        
        df_cv = pd.DataFrame({'State': state_labels_cv, 'CV': cvs})
        sns.boxplot(data=df_cv, x='State', y='CV', ax=ax4)
        ax4.set_title('Coefficient of Variation by State', fontweight='bold')
        ax4.set_ylabel('Coefficient of Variation')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temporal_dynamics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_frequency_analysis(self, data: pd.DataFrame, labels: np.ndarray, 
                                neuron_names: List[str], output_dir: str) -> None:
        """绘制频率域分析"""
        # 局部抑制字体警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.logger.info("生成频率域分析...")
            
            unique_states = sorted(set(labels))
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 功率谱密度对比
        ax1 = axes[0, 0]
        for state in unique_states:
            state_neurons = [neuron_names[i] for i, label in enumerate(labels) if label == state]
            if state_neurons:
                all_psds = []
                for neuron in state_neurons[:5]:  # 最多选择5个代表性神经元
                    signal = data[neuron].values
                    freqs, psd = welch(signal, fs=self.sampling_rate, nperseg=min(256, len(signal)//4))
                    all_psds.append(psd)
                
                mean_psd = np.mean(all_psds, axis=0)
                color = list(self.state_colors.values())[state]
                ax1.semilogy(freqs, mean_psd, color=color, label=f'State {state+1}', alpha=0.8)
        
        ax1.set_title('Power Spectral Density by State', fontweight='bold')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power Spectral Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 主导频率分布
        ax2 = axes[0, 1]
        dominant_freqs = []
        state_labels_freq = []
        
        for state in unique_states:
            state_neurons = [neuron_names[i] for i, label in enumerate(labels) if label == state]
            for neuron in state_neurons:
                signal = data[neuron].values
                freqs, psd = welch(signal, fs=self.sampling_rate, nperseg=min(256, len(signal)//4))
                dominant_freq = freqs[np.argmax(psd)]
                dominant_freqs.append(dominant_freq)
                state_labels_freq.append(f'State {state+1}')
        
        df_freq = pd.DataFrame({'State': state_labels_freq, 'Dominant_Frequency': dominant_freqs})
        sns.boxplot(data=df_freq, x='State', y='Dominant_Frequency', ax=ax2)
        ax2.set_title('Dominant Frequency Distribution', fontweight='bold')
        ax2.set_ylabel('Dominant Frequency (Hz)')
        
        # 3. 频带功率分析
        ax3 = axes[1, 0]
        freq_bands = ['Low (0-0.5Hz)', 'Mid (0.5-1.5Hz)', 'High (1.5-2.4Hz)']
        band_powers = {band: [] for band in freq_bands}
        state_labels_bands = {band: [] for band in freq_bands}
        
        for state in unique_states:
            state_neurons = [neuron_names[i] for i, label in enumerate(labels) if label == state]
            for neuron in state_neurons:
                signal = data[neuron].values
                freqs, psd = welch(signal, fs=self.sampling_rate, nperseg=min(256, len(signal)//4))
                
                low_power = self._band_power(freqs, psd, 0, 0.5)
                mid_power = self._band_power(freqs, psd, 0.5, 1.5)
                high_power = self._band_power(freqs, psd, 1.5, 2.4)
                
                band_powers['Low (0-0.5Hz)'].append(low_power)
                band_powers['Mid (0.5-1.5Hz)'].append(mid_power)
                band_powers['High (1.5-2.4Hz)'].append(high_power)
                
                for band in freq_bands:
                    state_labels_bands[band].append(f'State {state+1}')
        
        # 合并数据
        all_powers = []
        all_states = []
        all_bands = []
        
        for band in freq_bands:
            all_powers.extend(band_powers[band])
            all_states.extend(state_labels_bands[band])
            all_bands.extend([band] * len(band_powers[band]))
        
        df_bands = pd.DataFrame({'State': all_states, 'Power': all_powers, 'Band': all_bands})
        sns.boxplot(data=df_bands, x='State', y='Power', hue='Band', ax=ax3)
        ax3.set_title('Frequency Band Power Distribution', fontweight='bold')
        ax3.set_ylabel('Power')
        
        # 4. 频谱重心分析
        ax4 = axes[1, 1]
        centroids = []
        state_labels_centroid = []
        
        for state in unique_states:
            state_neurons = [neuron_names[i] for i, label in enumerate(labels) if label == state]
            for neuron in state_neurons:
                signal = data[neuron].values
                freqs, psd = welch(signal, fs=self.sampling_rate, nperseg=min(256, len(signal)//4))
                centroid = np.sum(freqs * psd) / np.sum(psd)
                centroids.append(centroid)
                state_labels_centroid.append(f'State {state+1}')
        
        df_centroid = pd.DataFrame({'State': state_labels_centroid, 'Spectral_Centroid': centroids})
        sns.boxplot(data=df_centroid, x='State', y='Spectral_Centroid', ax=ax4)
        ax4.set_title('Spectral Centroid by State', fontweight='bold')
        ax4.set_ylabel('Spectral Centroid (Hz)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'frequency_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_correlation_matrix(self, data: pd.DataFrame, labels: np.ndarray, 
                                neuron_names: List[str], output_dir: str) -> None:
        """绘制神经元相关性矩阵"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.logger.info("生成相关性矩阵...")
            
            unique_states = sorted(set(labels))
            
            # 为每种状态生成相关性矩阵
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            axes = axes.flatten()
            
            for i, state in enumerate(unique_states):
                if i >= 4:  # 最多显示4个状态
                    break
                    
                state_neurons = [neuron_names[j] for j, label in enumerate(labels) if label == state]
                
                if len(state_neurons) > 1:
                    # 计算该状态神经元之间的相关性
                    state_data = data[state_neurons]
                    correlation_matrix = state_data.corr()
                    
                    ax = axes[i]
                    sns.heatmap(correlation_matrix, 
                               annot=False, 
                               cmap='coolwarm', 
                               center=0,
                               square=True,
                               ax=ax,
                               cbar_kws={'label': 'Correlation Coefficient'})
                    
                    ax.set_title(f'State {state+1} Neuron Correlation Matrix\n({len(state_neurons)} neurons)', 
                               fontweight='bold')
                    ax.set_xlabel('Neurons')
                    ax.set_ylabel('Neurons')
                else:
                    axes[i].text(0.5, 0.5, f'State {state+1}\nInsufficient neurons\nfor correlation analysis', 
                               ha='center', va='center', transform=axes[i].transAxes, fontsize=12)
                    axes[i].set_xlim(0, 1)
                    axes[i].set_ylim(0, 1)
            
            # 移除多余的子图
            for i in range(len(unique_states), 4):
                axes[i].remove()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_matrices.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_state_transitions(self, data: pd.DataFrame, labels: np.ndarray, 
                               neuron_names: List[str], output_dir: str) -> None:
        """绘制状态转换分析（基于信号的时间窗口分析）"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.logger.info("生成状态转换分析...")
            
            # 选择几个代表性神经元进行时间窗口分析
            unique_states = sorted(set(labels))
            selected_neurons = []
            
            for state in unique_states:
                state_neurons = [neuron_names[i] for i, label in enumerate(labels) if label == state]
                if state_neurons:
                    selected_neurons.append(np.random.choice(state_neurons))
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, neuron in enumerate(selected_neurons[:4]):
                signal = data[neuron].values
                state = labels[neuron_names.index(neuron)]
                
                # 滑动窗口分析，检测局部状态变化
                window_size = int(self.sampling_rate * 10)  # 10秒窗口
                step_size = int(window_size / 2)
                
                windows = []
                window_states = []
                
                for start in range(0, len(signal) - window_size, step_size):
                    window = signal[start:start + window_size]
                    
                    # 简单的状态分类：基于活动强度
                    mean_activity = np.mean(window)
                    std_activity = np.std(window)
                    
                    if mean_activity > np.mean(signal) + np.std(signal):
                        window_state = 'High Activity'
                    elif mean_activity < np.mean(signal) - 0.5 * np.std(signal):
                        window_state = 'Low Activity'
                    else:
                        window_state = 'Medium Activity'
                    
                    windows.append(start / self.sampling_rate)
                    window_states.append(window_state)
                
                ax = axes[i]
                
                # 绘制原始信号
                time = np.arange(len(signal)) / self.sampling_rate
                ax.plot(time, signal, color='lightgray', alpha=0.7, label='Signal')
                
                # 绘制状态变化
                colors = {'High Activity': 'red', 'Medium Activity': 'blue', 'Low Activity': 'green'}
                
                for j, (window_time, window_state) in enumerate(zip(windows, window_states)):
                    if j < len(windows) - 1:
                        ax.axvspan(window_time, windows[j+1], 
                                 color=colors[window_state], alpha=0.3, label=window_state if j == 0 else "")
                
                ax.set_title(f'{neuron} (State {state+1}) - Temporal Activity Patterns', fontweight='bold')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Calcium Concentration')
                
                # 只在第一个子图中显示图例
                if i == 0:
                    handles, labels_legend = ax.get_legend_handles_labels()
                    unique_labels = []
                    unique_handles = []
                    for handle, label in zip(handles, labels_legend):
                        if label not in unique_labels:
                            unique_labels.append(label)
                            unique_handles.append(handle)
                    ax.legend(unique_handles, unique_labels)
                
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'state_transitions.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def _create_interactive_visualizations(self, data: pd.DataFrame, labels: np.ndarray, 
                                         neuron_names: List[str], features: np.ndarray,
                                         feature_names: List[str], output_dir: str) -> None:
        """创建交互式可视化（使用Plotly）"""
        if not PLOTLY_AVAILABLE:
            return
            
        self.logger.info("生成交互式可视化...")
        
        # 1. 交互式3D散点图（PCA降维）
        pca = PCA(n_components=3)
        features_scaled = StandardScaler().fit_transform(features)
        features_pca = pca.fit_transform(features_scaled)
        
        unique_states = sorted(set(labels))
        colors = [f'rgb({int(c[1:3], 16)}, {int(c[3:5], 16)}, {int(c[5:7], 16)})' 
                 for c in list(self.state_colors.values())[:len(unique_states)]]
        
        fig = go.Figure()
        
        for i, state in enumerate(unique_states):
            mask = labels == state
            fig.add_trace(go.Scatter3d(
                x=features_pca[mask, 0],
                y=features_pca[mask, 1],
                z=features_pca[mask, 2],
                mode='markers',
                name=f'State {state+1}',
                marker=dict(size=5, color=colors[i]),
                text=[neuron_names[j] for j in range(len(neuron_names)) if mask[j]],
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Interactive 3D PCA Visualization of Neuron States',
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)'
            ),
            width=800,
            height=600
        )
        
        fig.write_html(os.path.join(output_dir, 'interactive_3d_pca.html'))
        
        # 2. 交互式时间序列图
        fig = make_subplots(
            rows=len(unique_states), cols=1,
            subplot_titles=[f'State {s+1} Representative Signals' for s in unique_states],
            shared_xaxes=True
        )
        
        for i, state in enumerate(unique_states):
            state_neurons = [neuron_names[j] for j, label in enumerate(labels) if label == state]
            if state_neurons:
                selected_neuron = np.random.choice(state_neurons)
                signal = data[selected_neuron].values
                time = np.arange(len(signal)) / self.sampling_rate
                
                fig.add_trace(
                    go.Scatter(
                        x=time,
                        y=signal,
                        mode='lines',
                        name=f'{selected_neuron} (State {state+1})',
                        line=dict(color=colors[i]),
                        hovertemplate='Time: %{x:.2f}s<br>Concentration: %{y:.3f}<extra></extra>'
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title='Interactive Time Series by State',
            height=200 * len(unique_states),
            showlegend=True
        )
        
        fig.update_xaxes(title_text='Time (s)', row=len(unique_states), col=1)
        fig.update_yaxes(title_text='Calcium Concentration')
        
        fig.write_html(os.path.join(output_dir, 'interactive_timeseries.html'))
        
        # 3. 交互式特征重要性图
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(features_scaled, labels)
        
        feature_importance = rf.feature_importances_
        sorted_idx = np.argsort(feature_importance)[-20:]  # 显示前20个重要特征
        
        fig = go.Figure(go.Bar(
            x=feature_importance[sorted_idx],
            y=[feature_names[i] for i in sorted_idx],
            orientation='h',
            marker_color='rgba(55, 128, 191, 0.7)',
            hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Interactive Feature Importance Analysis',
            xaxis_title='Feature Importance',
            yaxis_title='Features',
            height=600,
            margin=dict(l=200)
        )
        
        fig.write_html(os.path.join(output_dir, 'interactive_feature_importance.html'))

    def extract_phase_space_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """提取相空间特征"""
        # 进行相空间重构
        phase_space = self.phase_analyzer.phase_space_reconstruction(signal_data)
        
        # 从相空间中提取特征
        return self.phase_analyzer.extract_phase_space_features(phase_space)
    
    def extract_graph_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """提取图特征"""
        features = {}
        
        # 构建时序图
        graphs = self.graph_builder.build_temporal_graph(signal_data)
        
        if graphs:
            # 计算图的基本统计特征
            all_node_features = np.concatenate(graphs, axis=0)
            
            features['graph_mean_node_value'] = np.mean(all_node_features[:, 0])
            features['graph_std_node_value'] = np.std(all_node_features[:, 0])
            features['graph_max_node_value'] = np.max(all_node_features[:, 0])
            features['graph_min_node_value'] = np.min(all_node_features[:, 0])
            
            # 计算图的连通性特征
            features['graph_num_windows'] = len(graphs)
            features['graph_avg_window_size'] = np.mean([len(g) for g in graphs])
            
            # 计算节点特征的变异性
            if all_node_features.shape[1] > 1:
                features['graph_first_diff_mean'] = np.mean(all_node_features[:, 1])
                features['graph_first_diff_std'] = np.std(all_node_features[:, 1])
            else:
                features['graph_first_diff_mean'] = 0.0
                features['graph_first_diff_std'] = 0.0
                
            if all_node_features.shape[1] > 2:
                features['graph_second_diff_mean'] = np.mean(all_node_features[:, 2])
                features['graph_second_diff_std'] = np.std(all_node_features[:, 2])
            else:
                features['graph_second_diff_mean'] = 0.0
                features['graph_second_diff_std'] = 0.0
        else:
            # 如果无法构建图，返回默认值
            features.update({
                'graph_mean_node_value': 0.0,
                'graph_std_node_value': 0.0,
                'graph_max_node_value': 0.0,
                'graph_min_node_value': 0.0,
                'graph_num_windows': 0,
                'graph_avg_window_size': 0.0,
                'graph_first_diff_mean': 0.0,
                'graph_first_diff_std': 0.0,
                'graph_second_diff_mean': 0.0,
                'graph_second_diff_std': 0.0
            })
        
        return features
    
    # 保持原有的特征提取方法
    def extract_temporal_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """提取时域特征"""
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
        """提取频域特征"""
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
        """提取非线性特征"""
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
        """提取形态学特征"""
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

    def visualize_temporal_states(self, data: pd.DataFrame, results_df: pd.DataFrame, 
                                 output_dir: str = '../results') -> None:
        """时间窗口状态可视化"""
        self.logger.info("开始时间窗口状态可视化...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 绘制神经元状态时间线
        self._plot_neuron_state_timeline(data, results_df, output_dir)
        
        # 2. 绘制状态转换矩阵
        self._plot_state_transition_matrix(results_df, output_dir)
        
        # 3. 绘制状态持续时间分布
        self._plot_state_duration_distribution(results_df, output_dir)
        
        # 4. 绘制神经元状态多样性分析
        self._plot_neuron_state_diversity(results_df, output_dir)
        
        # 5. 绘制时间窗口状态热图
        self._plot_temporal_state_heatmap(results_df, output_dir)
        
        # 6. 绘制代表性状态波形
        self._plot_representative_state_waveforms(data, results_df, output_dir)
        
        # 7. 如果可用，生成交互式时间线
        if PLOTLY_AVAILABLE:
            self._create_interactive_timeline(data, results_df, output_dir)
        
        self.logger.info("时间窗口状态可视化完成")
    
    def _plot_neuron_state_timeline(self, data: pd.DataFrame, results_df: pd.DataFrame, 
                                   output_dir: str) -> None:
        """绘制神经元状态时间线"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.logger.info("生成神经元状态时间线...")
            
            # 选择前6个神经元进行可视化
            neurons = results_df['neuron_id'].unique()[:6]
            
            fig, axes = plt.subplots(len(neurons), 1, figsize=(16, 2.5 * len(neurons)), sharex=True)
            if len(neurons) == 1:
                axes = [axes]
            
            colors = [list(self.state_colors.values())[i] for i in range(4)]
            
            for idx, neuron in enumerate(neurons):
                ax = axes[idx]
                
                # 获取该神经元的状态数据
                neuron_states = results_df[results_df['neuron_id'] == neuron].sort_values('window_idx')
                
                # 绘制原始信号
                signal = data[neuron].values
                time_signal = np.arange(len(signal)) / self.sampling_rate
                ax.plot(time_signal, signal, color='lightgray', alpha=0.7, linewidth=0.8, label='Signal')
                
                # 绘制状态区域
                for _, row in neuron_states.iterrows():
                    color = colors[row['state_label']] if row['state_label'] < len(colors) else 'gray'
                    ax.axvspan(row['start_time'], row['end_time'], 
                             color=color, alpha=0.3, label=f"State {row['state_label']+1}" if idx == 0 else "")
                
                ax.set_ylabel(f'{neuron}\nConcentration', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # 只在第一个子图显示图例
                if idx == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    unique_labels = []
                    unique_handles = []
                    for handle, label in zip(handles, labels):
                        if label not in unique_labels:
                            unique_labels.append(label)
                            unique_handles.append(handle)
                    ax.legend(unique_handles, unique_labels, loc='upper right', fontsize=8)
            
            axes[-1].set_xlabel('Time (s)', fontsize=12)
            plt.suptitle('Neuron State Timeline Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'neuron_state_timeline.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_state_transition_matrix(self, results_df: pd.DataFrame, output_dir: str) -> None:
        """绘制状态转换矩阵"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.logger.info("生成状态转换矩阵...")
            
            # 计算转换矩阵
            unique_states = sorted(results_df['state_label'].unique())
            n_states = len(unique_states)
            transition_matrix = np.zeros((n_states, n_states))
            
            for neuron in results_df['neuron_id'].unique():
                neuron_data = results_df[results_df['neuron_id'] == neuron].sort_values('window_idx')
                states = neuron_data['state_label'].values
                
                for i in range(len(states) - 1):
                    from_state = states[i]
                    to_state = states[i + 1]
                    transition_matrix[from_state, to_state] += 1
            
            # 归一化
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # 避免除零
            transition_matrix_norm = transition_matrix / row_sums
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(transition_matrix_norm, 
                       annot=True, 
                       fmt='.3f',
                       cmap='Blues',
                       xticklabels=[f'State {s+1}' for s in unique_states],
                       yticklabels=[f'State {s+1}' for s in unique_states],
                       cbar_kws={'label': 'Transition Probability'})
            
            plt.title('State Transition Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('To State')
            plt.ylabel('From State')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'state_transition_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_state_duration_distribution(self, results_df: pd.DataFrame, output_dir: str) -> None:
        """绘制状态持续时间分布"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.logger.info("生成状态持续时间分布...")
            
            # 计算连续状态的持续时间
            state_durations = []
            state_labels = []
            
            for neuron in results_df['neuron_id'].unique():
                neuron_data = results_df[results_df['neuron_id'] == neuron].sort_values('window_idx')
                
                current_state = None
                current_duration = 0
                
                for _, row in neuron_data.iterrows():
                    if current_state == row['state_label']:
                        current_duration += row['duration']
                    else:
                        if current_state is not None:
                            state_durations.append(current_duration)
                            state_labels.append(f'State {current_state+1}')
                        current_state = row['state_label']
                        current_duration = row['duration']
                
                # 添加最后一个状态
                if current_state is not None:
                    state_durations.append(current_duration)
                    state_labels.append(f'State {current_state+1}')
            
            # 创建DataFrame
            duration_df = pd.DataFrame({'State': state_labels, 'Duration': state_durations})
            
            plt.figure(figsize=(12, 8))
            
            # 箱线图
            plt.subplot(2, 1, 1)
            sns.boxplot(data=duration_df, x='State', y='Duration')
            plt.title('State Duration Distribution (Box Plot)', fontweight='bold')
            plt.ylabel('Duration (s)')
            
            # 直方图
            plt.subplot(2, 1, 2)
            unique_states = duration_df['State'].unique()
            colors = [list(self.state_colors.values())[i] for i in range(len(unique_states))]
            
            for i, state in enumerate(unique_states):
                state_data = duration_df[duration_df['State'] == state]['Duration']
                plt.hist(state_data, alpha=0.7, label=state, color=colors[i], bins=20)
            
            plt.title('State Duration Distribution (Histogram)', fontweight='bold')
            plt.xlabel('Duration (s)')
            plt.ylabel('Frequency')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'state_duration_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_neuron_state_diversity(self, results_df: pd.DataFrame, output_dir: str) -> None:
        """绘制神经元状态多样性分析"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.logger.info("生成神经元状态多样性分析...")
            
            # 计算每个神经元的状态多样性
            neuron_diversity = []
            
            for neuron in results_df['neuron_id'].unique():
                neuron_data = results_df[results_df['neuron_id'] == neuron]
                
                # 状态计数
                state_counts = neuron_data['state_label'].value_counts()
                total_windows = len(neuron_data)
                
                # 计算Shannon熵作为多样性指标
                probabilities = state_counts / total_windows
                shannon_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                
                # 计算状态切换次数
                states = neuron_data.sort_values('window_idx')['state_label'].values
                switches = np.sum(np.diff(states) != 0)
                
                neuron_diversity.append({
                    'neuron_id': neuron,
                    'unique_states': len(state_counts),
                    'shannon_entropy': shannon_entropy,
                    'state_switches': switches,
                    'switch_rate': switches / (total_windows - 1) if total_windows > 1 else 0
                })
            
            diversity_df = pd.DataFrame(neuron_diversity)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 唯一状态数分布
            axes[0, 0].hist(diversity_df['unique_states'], bins=range(1, 6), alpha=0.7, color='skyblue')
            axes[0, 0].set_title('Distribution of Unique States per Neuron', fontweight='bold')
            axes[0, 0].set_xlabel('Number of Unique States')
            axes[0, 0].set_ylabel('Number of Neurons')
            
            # Shannon熵分布
            axes[0, 1].hist(diversity_df['shannon_entropy'], bins=20, alpha=0.7, color='lightgreen')
            axes[0, 1].set_title('State Diversity (Shannon Entropy)', fontweight='bold')
            axes[0, 1].set_xlabel('Shannon Entropy')
            axes[0, 1].set_ylabel('Number of Neurons')
            
            # 状态切换次数
            axes[1, 0].hist(diversity_df['state_switches'], bins=20, alpha=0.7, color='salmon')
            axes[1, 0].set_title('State Switches per Neuron', fontweight='bold')
            axes[1, 0].set_xlabel('Number of State Switches')
            axes[1, 0].set_ylabel('Number of Neurons')
            
            # 切换率 vs 多样性
            scatter = axes[1, 1].scatter(diversity_df['switch_rate'], diversity_df['shannon_entropy'], 
                                       alpha=0.7, c=diversity_df['unique_states'], cmap='viridis')
            axes[1, 1].set_title('Switch Rate vs State Diversity', fontweight='bold')
            axes[1, 1].set_xlabel('State Switch Rate')
            axes[1, 1].set_ylabel('Shannon Entropy')
            plt.colorbar(scatter, ax=axes[1, 1], label='Unique States')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'neuron_state_diversity.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_temporal_state_heatmap(self, results_df: pd.DataFrame, output_dir: str) -> None:
        """绘制时间窗口状态热图"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.logger.info("生成时间窗口状态热图...")
            
            # 选择前20个神经元
            neurons = results_df['neuron_id'].unique()[:20]
            max_window = results_df['window_idx'].max()
            
            # 创建状态矩阵
            state_matrix = np.full((len(neurons), max_window + 1), -1)
            
            for i, neuron in enumerate(neurons):
                neuron_data = results_df[results_df['neuron_id'] == neuron]
                for _, row in neuron_data.iterrows():
                    state_matrix[i, row['window_idx']] = row['state_label']
            
            # 创建自定义颜色映射
            cmap = plt.cm.Set3
            
            plt.figure(figsize=(16, 10))
            im = plt.imshow(state_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=3)
            
            plt.title('Temporal State Heatmap (Neurons vs Time Windows)', fontsize=16, fontweight='bold')
            plt.xlabel('Time Window Index')
            plt.ylabel('Neurons')
            plt.yticks(range(len(neurons)), [f'{n}' for n in neurons])
            
            # 创建颜色条
            cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
            cbar.set_label('State')
            cbar.set_ticklabels(['State 1', 'State 2', 'State 3', 'State 4'])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'temporal_state_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_representative_state_waveforms(self, data: pd.DataFrame, results_df: pd.DataFrame, 
                                           output_dir: str) -> None:
        """绘制代表性状态波形"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.logger.info("生成代表性状态波形...")
            
            unique_states = sorted(results_df['state_label'].unique())
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, state in enumerate(unique_states):
                if i >= 4:
                    break
                
                ax = axes[i]
                
                # 获取该状态的代表性窗口
                state_windows = results_df[results_df['state_label'] == state]
                
                # 随机选择几个窗口
                selected_windows = state_windows.sample(min(5, len(state_windows)))
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(selected_windows)))
                
                for (_, row), color in zip(selected_windows.iterrows(), colors):
                    neuron = row['neuron_id']
                    start_idx = int(row['start_time'] * self.sampling_rate)
                    end_idx = int(row['end_time'] * self.sampling_rate)
                    
                    if start_idx < len(data[neuron]) and end_idx <= len(data[neuron]):
                        window_signal = data[neuron].iloc[start_idx:end_idx].values
                        time_window = np.arange(len(window_signal)) / self.sampling_rate
                        
                        ax.plot(time_window, window_signal, color=color, alpha=0.7, 
                               linewidth=1.5, label=f'{neuron} (t={row["start_time"]:.1f}s)')
                
                state_desc = list(self.state_definitions.values())[state] if state < len(self.state_definitions) else "Unknown State"
                ax.set_title(f'State {state + 1}: {state_desc}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Calcium Concentration')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # 移除多余的子图
            for i in range(len(unique_states), 4):
                axes[i].remove()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'representative_state_waveforms.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_interactive_timeline(self, data: pd.DataFrame, results_df: pd.DataFrame, 
                                   output_dir: str) -> None:
        """创建交互式时间线（使用Plotly）"""
        if not PLOTLY_AVAILABLE:
            return
        
        self.logger.info("生成交互式时间线...")
        
        # 选择前3个神经元进行交互式可视化
        neurons = results_df['neuron_id'].unique()[:3]
        
        fig = make_subplots(
            rows=len(neurons), cols=1,
            subplot_titles=[f'Neuron {neuron} State Timeline' for neuron in neurons],
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        colors = ['red', 'gold', 'green', 'blue']
        
        for idx, neuron in enumerate(neurons):
            # 添加原始信号
            signal = data[neuron].values
            time_signal = np.arange(len(signal)) / self.sampling_rate
            
            fig.add_trace(
                go.Scatter(
                    x=time_signal,
                    y=signal,
                    mode='lines',
                    name=f'{neuron} Signal',
                    line=dict(color='lightgray', width=1),
                    hovertemplate='Time: %{x:.2f}s<br>Concentration: %{y:.3f}<extra></extra>',
                    showlegend=(idx == 0)
                ),
                row=idx+1, col=1
            )
            
            # 添加状态区域
            neuron_states = results_df[results_df['neuron_id'] == neuron].sort_values('window_idx')
            
            for _, row in neuron_states.iterrows():
                color = colors[row['state_label']] if row['state_label'] < len(colors) else 'gray'
                
                fig.add_vrect(
                    x0=row['start_time'], x1=row['end_time'],
                    fillcolor=color, opacity=0.3,
                    line_width=0,
                    row=idx+1, col=1
                )
        
        fig.update_layout(
            title='Interactive Neuron State Timeline',
            height=200 * len(neurons),
            showlegend=True
        )
        
        fig.update_xaxes(title_text='Time (s)', row=len(neurons), col=1)
        fig.update_yaxes(title_text='Calcium Concentration')
        
        fig.write_html(os.path.join(output_dir, 'interactive_timeline.html'))
    
    def save_temporal_results(self, results_df: pd.DataFrame, features: np.ndarray,
                            feature_names: List[str], output_path: str) -> None:
        """保存时间窗口分析结果"""
        self.logger.info(f"保存时间窗口分析结果到: {output_path}")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 保存主要结果
            results_df.to_excel(writer, sheet_name='Temporal State Analysis', index=False)
            
            # 保存神经元状态统计
            neuron_stats = results_df.groupby('neuron_id').agg({
                'state_label': ['count', 'nunique'],
                'duration': 'sum'
            }).round(3)
            neuron_stats.columns = ['Total_Windows', 'Unique_States', 'Total_Duration']
            neuron_stats.reset_index().to_excel(writer, sheet_name='Neuron Statistics', index=False)
            
            # 保存状态统计
            state_stats = results_df.groupby('state_name').agg({
                'neuron_id': 'nunique',
                'window_idx': 'count',
                'duration': 'sum'
            }).round(3)
            state_stats.columns = ['Unique_Neurons', 'Window_Count', 'Total_Duration']
            state_stats.reset_index().to_excel(writer, sheet_name='State Statistics', index=False)
            
            # 保存特征数据（窗口级别）
            features_df = pd.DataFrame(features, columns=feature_names)
            window_features = pd.concat([
                results_df[['neuron_id', 'window_idx', 'start_time', 'state_label']],
                features_df
            ], axis=1)
            window_features.to_excel(writer, sheet_name='Window Features', index=False)

    def extract_comprehensive_features_traditional(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        提取传统综合特征（基于整个信号，保持兼容性）
        
        Parameters
        ----------
        data : pd.DataFrame
            神经元钙离子浓度数据
            
        Returns
        -------
        Tuple[np.ndarray, List[str], List[str]]
            特征矩阵, 特征名称列表, 神经元名称列表
        """
        self.logger.info("开始提取传统综合神经元特征...")
        
        # 识别神经元列
        neuron_columns = [col for col in data.columns 
                         if col.startswith('n') and col[1:].isdigit()]
        
        feature_list = []
        feature_names = None
        
        for neuron in neuron_columns:
            signal_data = data[neuron].values
            
            # 提取传统特征
            temporal_features = self.extract_temporal_features(signal_data)
            frequency_features = self.extract_frequency_features(signal_data)
            nonlinear_features = self.extract_nonlinear_features(signal_data)
            morphological_features = self.extract_morphological_features(signal_data)
            
            # 提取相空间特征
            phase_space_features = self.extract_phase_space_features(signal_data)
            
            # 提取图特征
            graph_features = self.extract_graph_features(signal_data)
            
            # 合并所有特征
            all_features = {
                **temporal_features, 
                **frequency_features,
                **nonlinear_features, 
                **morphological_features,
                **phase_space_features,
                **graph_features
            }
            
            if feature_names is None:
                feature_names = list(all_features.keys())
            
            feature_list.append(list(all_features.values()))
        
        features_matrix = np.array(feature_list)
        
        self.logger.info(f"传统特征提取完成，共提取 {len(feature_names)} 个特征，{len(neuron_columns)} 个神经元")
        
        return features_matrix, feature_names, neuron_columns


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强版神经元放电状态分析（支持时间窗口动态分析）')
    parser.add_argument('--input', type=str, 
                       default='../datasets/processed_EMtrace01.xlsx',
                       help='输入数据文件路径')
    parser.add_argument('--output-dir', type=str, 
                       default='../results/temporal_state_analysis/',
                       help='输出目录')
    parser.add_argument('--method', type=str, default='ensemble',
                       choices=['kmeans', 'dbscan', 'ensemble', 'gcn'],
                       help='聚类方法')
    parser.add_argument('--n-states', type=int, default=4,
                       help='预期状态数量')
    parser.add_argument('--sampling-rate', type=float, default=4.8,
                       help='采样频率(Hz)')
    parser.add_argument('--window-duration', type=float, default=30.0,
                       help='时间窗口长度（秒）')
    parser.add_argument('--overlap-ratio', type=float, default=0.5,
                       help='窗口重叠比例（0-1）')
    parser.add_argument('--analysis-mode', type=str, default='temporal',
                       choices=['temporal', 'traditional'],
                       help='分析模式：temporal(时间窗口分析) 或 traditional(传统全信号分析)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化增强版分析器
    analyzer = EnhancedStateAnalyzer(
        sampling_rate=args.sampling_rate,
        window_duration=args.window_duration,
        overlap_ratio=args.overlap_ratio
    )
    
    try:
        # 加载数据
        data = analyzer.load_data(args.input)
        
        if args.analysis_mode == 'temporal':
            # 时间窗口分析模式（推荐）
            analyzer.logger.info("=== 进行时间窗口动态状态分析 ===")
            
            # 进行时间窗口状态分析
            labels, results_df = analyzer.analyze_temporal_states(
                data, method=args.method, n_states=args.n_states
            )
            
            # 提取特征用于保存
            features, feature_names, _ = analyzer.extract_windowed_features(data)
            
            # 时间窗口可视化
            analyzer.visualize_temporal_states(data, results_df, args.output_dir)
            
            # 保存时间窗口分析结果
            output_file = os.path.join(args.output_dir, 'temporal_neuron_states_analysis.xlsx')
            analyzer.save_temporal_results(results_df, features, feature_names, output_file)
            
            # 输出关键统计信息
            analyzer.logger.info("\n=== 时间窗口分析结果摘要 ===")
            analyzer.logger.info(f"总时间窗口数: {len(results_df)}")
            analyzer.logger.info(f"分析神经元数: {results_df['neuron_id'].nunique()}")
            analyzer.logger.info(f"识别状态数: {results_df['state_label'].nunique()}")
            
            # 每个神经元的状态多样性
            neuron_diversity = results_df.groupby('neuron_id')['state_label'].nunique()
            analyzer.logger.info(f"神经元状态多样性: 平均 {neuron_diversity.mean():.2f} 种状态/神经元")
            analyzer.logger.info(f"状态切换最多的神经元: {neuron_diversity.idxmax()} ({neuron_diversity.max()} 种状态)")
            
            print(f"\n🎉 时间窗口神经元状态分析完成！")
            print(f"📊 结果保存在: {args.output_dir}")
            print(f"📈 共分析了 {results_df['neuron_id'].nunique()} 个神经元的 {len(results_df)} 个时间窗口")
            print(f"🔍 识别出 {results_df['state_label'].nunique()} 种不同的放电状态")
            
        else:
            # 传统分析模式（兼容原有功能）
            analyzer.logger.info("=== 进行传统全信号状态分析 ===")
            
            # 提取综合特征（原有方法，基于整个信号）
            features, feature_names, neuron_names = analyzer.extract_comprehensive_features_traditional(data)
            
            # 识别状态
            labels = analyzer.identify_states_enhanced(features, method=args.method, 
                                                     n_states=args.n_states)
            
            # 传统可视化
            analyzer.visualize_enhanced_states(data, labels, neuron_names, args.output_dir)
            
            # 保存传统分析结果
            output_file = os.path.join(args.output_dir, 'traditional_neuron_states_analysis.xlsx')
            analyzer.save_enhanced_results(data, labels, neuron_names, features, 
                                         feature_names, output_file)
            
            print(f"\n🎉 传统神经元状态分析完成！")
            print(f"📊 结果保存在: {args.output_dir}")
            print(f"📈 共分析了 {len(neuron_names)} 个神经元")
            print(f"🔍 识别出 {len(set(labels))} 种不同的放电状态")
        
    except Exception as e:
        analyzer.logger.error(f"分析过程中出现错误: {str(e)}")
        print(f"❌ 分析失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()
