#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的标签生成系统
===================

基于时间序列特征和无监督聚类方法生成更有意义的神经状态标签

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import entropy
from scipy.signal import find_peaks, periodogram
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AdvancedLabelGenerator:
    """
    高级标签生成器
    
    基于多种时间序列特征和无监督聚类方法，
    为钙成像数据生成更有意义的神经状态标签。
    """
    
    def __init__(self, num_classes: int = 6, random_state: int = 42):
        """
        初始化标签生成器
        
        Parameters
        ----------
        num_classes : int
            目标类别数量
        random_state : int
            随机种子
        """
        self.num_classes = num_classes
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def extract_time_series_features(self, signal: np.ndarray) -> np.ndarray:
        """
        从时间序列中提取多维特征
        
        Parameters
        ----------
        signal : np.ndarray
            输入的时间序列信号
            
        Returns
        -------
        np.ndarray
            提取的特征向量
        """
        features = []
        
        # 基础统计特征
        features.extend([
            np.mean(signal),           # 均值
            np.std(signal),            # 标准差
            np.var(signal),            # 方差
            np.median(signal),         # 中位数
            np.max(signal),            # 最大值
            np.min(signal),            # 最小值
            np.ptp(signal),            # 峰峰值
        ])
        
        # 高阶统计特征
        from scipy.stats import skew, kurtosis
        features.extend([
            skew(signal),              # 偏度
            kurtosis(signal),          # 峰度
        ])
        
        # 能量特征
        features.extend([
            np.sum(signal**2),         # 总能量
            np.sqrt(np.mean(signal**2)), # RMS
        ])
        
        # 频域特征
        try:
            freqs, psd = periodogram(signal, fs=1.0)
            features.extend([
                np.sum(psd),                    # 总功率
                np.argmax(psd),                 # 主频率索引
                np.sum(psd * freqs) / np.sum(psd), # 功率谱质心
            ])
        except:
            features.extend([0, 0, 0])
        
        # 时域复杂度特征
        features.extend([
            len(find_peaks(signal)[0]),         # 峰值数量
            np.sum(np.diff(signal) > 0),        # 上升点数量
            np.sum(np.abs(np.diff(signal))),    # 总变化量
        ])
        
        # 熵特征
        try:
            # 离散化信号计算熵
            hist, _ = np.histogram(signal, bins=20, density=True)
            hist = hist[hist > 0]  # 去除零值
            features.append(entropy(hist))     # 香农熵
        except:
            features.append(0)
        
        # 分形维数近似
        try:
            # Higuchi分形维数的简化版本
            def higuchi_fd(X, Kmax=10):
                N = len(X)
                Lk = []
                for k in range(1, Kmax+1):
                    Lmk = []
                    for m in range(k):
                        Lmki = 0
                        for i in range(1, int((N-m)/k)):
                            Lmki += abs(X[m+i*k] - X[m+(i-1)*k])
                        Lmki = Lmki * (N-1) / (int((N-m)/k) * k) / k
                        Lmk.append(Lmki)
                    Lk.append(np.log(np.mean(Lmk)))
                
                # 线性拟合
                x = np.log(range(1, Kmax+1))
                coeffs = np.polyfit(x, Lk, 1)
                return -coeffs[0]
            
            features.append(higuchi_fd(signal))
        except:
            features.append(1.0)
        
        return np.array(features)
    
    def extract_phasespace_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        从相空间轨迹中提取特征
        
        Parameters
        ----------
        trajectory : np.ndarray
            相空间轨迹，形状为 (time_points, dimensions)
            
        Returns
        -------
        np.ndarray
            提取的相空间特征
        """
        features = []
        
        # 轨迹基础特征
        features.extend([
            trajectory.shape[0],                    # 轨迹长度
            np.mean(np.linalg.norm(trajectory, axis=1)), # 平均距离原点的距离
            np.std(np.linalg.norm(trajectory, axis=1)),  # 距离标准差
        ])
        
        # 轨迹几何特征
        if len(trajectory) > 1:
            # 计算相邻点间距离
            distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
            features.extend([
                np.mean(distances),                 # 平均步长
                np.std(distances),                  # 步长变异性
                np.sum(distances),                  # 总路径长度
            ])
            
            # 计算轨迹的曲率
            if len(trajectory) > 2:
                try:
                    # 简化的曲率计算
                    v1 = np.diff(trajectory[:-1], axis=0)
                    v2 = np.diff(trajectory[1:], axis=0)
                    curvatures = []
                    for i in range(len(v1)):
                        if np.linalg.norm(v1[i]) > 0 and np.linalg.norm(v2[i]) > 0:
                            cos_angle = np.dot(v1[i], v2[i]) / (np.linalg.norm(v1[i]) * np.linalg.norm(v2[i]))
                            cos_angle = np.clip(cos_angle, -1, 1)  # 确保在有效范围内
                            curvatures.append(np.arccos(cos_angle))
                    
                    if curvatures:
                        features.extend([
                            np.mean(curvatures),        # 平均曲率
                            np.std(curvatures),         # 曲率变异性
                        ])
                    else:
                        features.extend([0, 0])
                except:
                    features.extend([0, 0])
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # 轨迹边界框特征
        if len(trajectory) > 0:
            mins = np.min(trajectory, axis=0)
            maxs = np.max(trajectory, axis=0)
            ranges = maxs - mins
            features.extend([
                np.prod(ranges),                    # 边界框体积
                np.mean(ranges),                    # 平均范围
                np.std(ranges),                     # 范围标准差
            ])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def extract_network_features(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        从网络邻接矩阵中提取图特征
        
        Parameters
        ----------
        adjacency_matrix : np.ndarray
            邻接矩阵
            
        Returns
        -------
        np.ndarray
            网络特征向量
        """
        features = []
        
        # 确保是方阵
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            # 如果不是方阵，创建一个简单的对称矩阵
            n = min(adjacency_matrix.shape)
            adj = adjacency_matrix[:n, :n]
        else:
            adj = adjacency_matrix.copy()
        
        # 基础网络特征
        n_nodes = adj.shape[0]
        n_edges = np.sum(adj > 0) // 2 if np.allclose(adj, adj.T) else np.sum(adj > 0)
        
        features.extend([
            n_nodes,                            # 节点数
            n_edges,                            # 边数
            n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0,  # 密度
        ])
        
        # 度分布特征
        degrees = np.sum(adj > 0, axis=1)
        features.extend([
            np.mean(degrees),                   # 平均度
            np.std(degrees),                    # 度标准差
            np.max(degrees) if len(degrees) > 0 else 0,  # 最大度
        ])
        
        # 权重特征
        if np.any(adj):
            weights = adj[adj > 0]
            features.extend([
                np.mean(weights),               # 平均权重
                np.std(weights),                # 权重标准差
                np.max(weights),                # 最大权重
            ])
        else:
            features.extend([0, 0, 0])
        
        # 谱特征
        try:
            eigenvals = np.linalg.eigvals(adj)
            eigenvals = eigenvals[np.isreal(eigenvals)].real
            eigenvals = np.sort(eigenvals)[::-1]
            
            features.extend([
                eigenvals[0] if len(eigenvals) > 0 else 0,     # 最大特征值
                eigenvals[-1] if len(eigenvals) > 0 else 0,    # 最小特征值
                np.sum(eigenvals),                              # 特征值和
            ])
        except:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def generate_labels_from_features(self, features: np.ndarray, method: str = 'ensemble') -> np.ndarray:
        """
        基于特征生成标签
        
        Parameters
        ----------
        features : np.ndarray
            特征矩阵，形状为 (n_samples, n_features)
        method : str
            聚类方法: 'kmeans', 'spectral', 'gmm', 'ensemble'
            
        Returns
        -------
        np.ndarray
            生成的标签
        """
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        # 降维处理
        if features_scaled.shape[1] > 10:
            pca = PCA(n_components=min(10, features_scaled.shape[0]-1))
            features_scaled = pca.fit_transform(features_scaled)
            logger.info(f"PCA降维: {features.shape[1]} -> {features_scaled.shape[1]}")
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=self.num_classes, random_state=self.random_state, n_init=10)
        elif method == 'spectral':
            clusterer = SpectralClustering(n_clusters=self.num_classes, random_state=self.random_state)
        elif method == 'gmm':
            clusterer = GaussianMixture(n_components=self.num_classes, random_state=self.random_state)
        elif method == 'ensemble':
            # 集成方法：结合多种聚类算法
            return self._ensemble_clustering(features_scaled)
        else:
            raise ValueError(f"未知的聚类方法: {method}")
        
        labels = clusterer.fit_predict(features_scaled)
        
        # 评估聚类质量
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(features_scaled, labels)
            ch_score = calinski_harabasz_score(features_scaled, labels)
            logger.info(f"聚类质量评估 - 轮廓系数: {sil_score:.3f}, CH指数: {ch_score:.3f}")
        
        return labels
    
    def _ensemble_clustering(self, features: np.ndarray) -> np.ndarray:
        """
        集成聚类方法
        
        Parameters
        ----------
        features : np.ndarray
            标准化后的特征矩阵
            
        Returns
        -------
        np.ndarray
            集成聚类结果
        """
        # 多种聚类方法
        methods = [
            KMeans(n_clusters=self.num_classes, random_state=self.random_state, n_init=10),
            SpectralClustering(n_clusters=self.num_classes, random_state=self.random_state),
            GaussianMixture(n_components=self.num_classes, random_state=self.random_state)
        ]
        
        # 获取每种方法的聚类结果
        all_labels = []
        scores = []
        
        for method in methods:
            try:
                if hasattr(method, 'fit_predict'):
                    labels = method.fit_predict(features)
                else:
                    labels = method.fit(features).predict(features)
                
                # 评估聚类质量
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(features, labels)
                    all_labels.append(labels)
                    scores.append(score)
                    logger.info(f"{method.__class__.__name__} 轮廓系数: {score:.3f}")
            except Exception as e:
                logger.warning(f"{method.__class__.__name__} 聚类失败: {e}")
        
        # 选择最佳聚类结果
        if all_labels:
            best_idx = np.argmax(scores)
            best_labels = all_labels[best_idx]
            logger.info(f"选择最佳聚类方法: {methods[best_idx].__class__.__name__}")
            return best_labels
        else:
            # 如果所有方法都失败，使用随机标签
            logger.warning("所有聚类方法都失败，使用随机标签")
            return np.random.randint(0, self.num_classes, size=features.shape[0])


def generate_improved_labels(xyz_trim: list, pred_num: int, num_classes: int = 6) -> np.ndarray:
    """
    生成改进的标签
    
    Parameters
    ----------
    xyz_trim : list
        裁剪后的相空间数据
    pred_num : int
        预测样本数量
    num_classes : int
        类别数量
        
    Returns
    -------
    np.ndarray
        改进的标签
    """
    logger.info("开始生成改进的标签...")
    
    # 创建标签生成器
    label_generator = AdvancedLabelGenerator(num_classes=num_classes)
    
    # 提取所有有效样本
    valid_samples = []
    for cell_data in xyz_trim:
        for trajectory in cell_data:
            if trajectory is not None and len(trajectory) > 0:
                valid_samples.append(trajectory)
    
    if len(valid_samples) == 0:
        logger.warning("没有有效样本，使用随机标签")
        return np.random.randint(0, num_classes, size=pred_num)
    
    # 限制样本数量
    if len(valid_samples) > pred_num:
        valid_samples = valid_samples[:pred_num]
    
    logger.info(f"处理 {len(valid_samples)} 个有效样本")
    
    # 提取多维特征
    all_features = []
    
    for i, trajectory in enumerate(valid_samples):
        features = []
        
        # 从相空间轨迹提取特征
        try:
            if isinstance(trajectory, np.ndarray) and trajectory.ndim == 2:
                # 相空间特征
                ps_features = label_generator.extract_phasespace_features(trajectory)
                features.extend(ps_features)
                
                # 时间序列特征（使用第一个维度）
                if trajectory.shape[1] > 0:
                    ts_features = label_generator.extract_time_series_features(trajectory[:, 0])
                    features.extend(ts_features)
                
            elif isinstance(trajectory, np.ndarray) and trajectory.ndim == 1:
                # 一维时间序列
                ts_features = label_generator.extract_time_series_features(trajectory)
                features.extend(ts_features)
                
        except Exception as e:
            logger.warning(f"样本 {i} 特征提取失败: {e}")
            # 使用零特征作为备选
            features = [0] * 30  # 假设特征维度
        
        all_features.append(features)
    
    # 转换为numpy数组
    feature_matrix = np.array(all_features)
    
    # 处理无效值
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
    
    logger.info(f"特征矩阵形状: {feature_matrix.shape}")
    
    # 生成标签
    labels = label_generator.generate_labels_from_features(feature_matrix, method='ensemble')
    
    # 确保标签数量正确
    if len(labels) < pred_num:
        # 补齐标签
        additional_labels = np.random.choice(labels, pred_num - len(labels), replace=True)
        labels = np.concatenate([labels, additional_labels])
    elif len(labels) > pred_num:
        # 截断标签
        labels = labels[:pred_num]
    
    # 记录标签分布
    unique_labels, counts = np.unique(labels, return_counts=True)
    logger.info(f"改进标签分布: {dict(zip(unique_labels, counts))}")
    
    return labels


if __name__ == "__main__":
    # 测试代码
    import matplotlib.pyplot as plt
    
    # 生成测试数据
    np.random.seed(42)
    test_signals = []
    for i in range(100):
        t = np.linspace(0, 10, 1000)
        if i < 33:
            # 类型1: 正弦波
            signal = np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(len(t))
        elif i < 66:
            # 类型2: 锯齿波
            signal = 2 * (t % 2) - 1 + 0.1 * np.random.randn(len(t))
        else:
            # 类型3: 噪声
            signal = np.random.randn(len(t))
        test_signals.append(signal)
    
    # 测试标签生成
    generator = AdvancedLabelGenerator(num_classes=3)
    features = []
    for signal in test_signals:
        feat = generator.extract_time_series_features(signal)
        features.append(feat)
    
    feature_matrix = np.array(features)
    labels = generator.generate_labels_from_features(feature_matrix)
    
    print(f"生成的标签分布: {np.bincount(labels)}") 