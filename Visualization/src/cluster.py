#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import os
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import argparse  # 导入命令行参数处理模块
import glob
import logging
import datetime
import sys

# 初始化日志记录器
def setup_logger(output_dir=None, prefix="cluster", capture_all_output=True):
    """
    设置日志记录器，将日志消息输出到控制台和文件
    
    参数:
        output_dir: 日志文件输出目录，默认为输出到当前脚本所在目录的logs文件夹
        prefix: 日志文件名称前缀
        capture_all_output: 是否捕获所有标准输出到日志文件
    
    返回:
        logger: 已配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(prefix)
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器，避免重复添加
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了输出目录，则添加文件处理器
    if output_dir is None:
        # 默认在当前脚本目录下创建 logs 文件夹
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建日志文件名，包含时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(output_dir, f"{prefix}_{timestamp}.log")
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 创建输出重定向类，捕获所有的标准输出和错误输出
    class OutputRedirector:
        def __init__(self, original_stream, logger, level=logging.INFO):
            self.original_stream = original_stream
            self.logger = logger
            self.level = level
            self.buffer = ''
        
        def write(self, buf):
            self.original_stream.write(buf)
            self.buffer += buf
            if '\n' in buf:
                self.flush()
        
        def flush(self):
            if self.buffer.strip():
                for line in self.buffer.rstrip().split('\n'):
                    if line.strip():  # 只记录非空行
                        self.logger.log(self.level, f"OUTPUT: {line.rstrip()}")
            self.buffer = ''
    
    # 重定向标准输出和错误输出到日志文件
    if capture_all_output:
        sys.stdout = OutputRedirector(sys.stdout, logger, logging.INFO)
        sys.stderr = OutputRedirector(sys.stderr, logger, logging.ERROR)
    
    logger.info(f"日志文件创建于: {log_file}")
    return logger

def load_data(file_path, logger=None):
    """
    加载钙爆发数据
    
    参数
    ----------
    file_path : str
        数据文件路径
    logger : logging.Logger, 可选
        日志记录器实例，默认为None
        
    返回
    -------
    df : pandas.DataFrame
        加载的数据
    """
    # 如果未提供日志记录器，创建一个默认的
    if logger is None:
        logger = setup_logger()
    
    logger.info(f"正在从{file_path}加载数据...")
    df = pd.read_excel(file_path)
    logger.info(f"成功加载数据，共{len(df)}行")
    return df

def enhance_preprocess_data(df, feature_weights=None):
    """
    增强版预处理功能，支持子峰分析和更多特征，并支持特征权重调整
    
    参数
    ----------
    df : pandas.DataFrame
        原始数据
    feature_weights : dict, 可选
        特征权重字典，键为特征名称，值为权重值，默认为None（所有特征权重相等）
        
    返回
    -------
    features_scaled : numpy.ndarray
        标准化并应用权重后的特征数据
    feature_names : list
        特征名称列表
    """
    # 基础特征集
    feature_names = ['amplitude', 'duration', 'fwhm', 'rise_time', 'decay_time', 'auc', 'snr']
    
    # 检查是否包含波形类型信息，增加波形分类特征
    if 'wave_type' in df.columns:
        df['is_complex'] = df['wave_type'].apply(lambda x: 1 if x == 'complex' else 0)
        feature_names.append('is_complex')
    
    # 检查是否包含子峰信息
    if 'subpeaks_count' in df.columns:
        feature_names.append('subpeaks_count')
    
    # 将特征值转为数值类型
    for col in feature_names:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 删除缺失值
    df = df.dropna(subset=feature_names)
    
    # 提取特征
    features = df[feature_names].values
    
    # 特征标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 应用特征权重
    if feature_weights is not None:
        weights_array = np.ones(len(feature_names))
        weight_info = []
        
        # 构建权重数组
        for i, feature in enumerate(feature_names):
            if feature in feature_weights:
                weights_array[i] = feature_weights[feature]
                weight_info.append(f"{feature}:{feature_weights[feature]:.2f}")
            else:
                weight_info.append(f"{feature}:1.00")
        
        # 应用权重
        features_scaled = features_scaled * weights_array.reshape(1, -1)
        print(f"应用特征权重: {', '.join(weight_info)}")
    else:
        print("未设置特征权重，所有特征权重相等")
    
    print(f"预处理完成，保留{len(df)}个有效样本，使用特征: {', '.join(feature_names)}")
    return features_scaled, feature_names, df

def determine_optimal_k(features_scaled, max_k=10, output_dir='../results'):
    """
    确定最佳聚类数
    
    参数
    ----------
    features_scaled : numpy.ndarray
        标准化后的特征数据
    max_k : int, 可选
        最大测试聚类数，默认为10
    output_dir : str, 可选
        输出目录路径，默认为'../results'
        
    返回
    -------
    optimal_k : int
        最佳聚类数
    """
    print("正在确定最佳聚类数...")
    inertia = []
    silhouette_scores = []
    
    # 计算不同k值的肘部指标和轮廓系数
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features_scaled, kmeans.labels_))
    
    # 绘制肘部法则图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), inertia, 'o-', color='blue')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True, alpha=0.3)
    # 绘制轮廓系数图
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores, 'o-', color='green')
    plt.title('Silhouette Score')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/optimal_k_determination.png', dpi=300)
    
    # 找到轮廓系数最高的k值
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"基于轮廓系数，最佳聚类数为{optimal_k}")
    
    return optimal_k

def cluster_kmeans(features_scaled, n_clusters):
    """
    使用K均值聚类
    
    参数
    ----------
    features_scaled : numpy.ndarray
        标准化后的特征数据
    n_clusters : int
        聚类数
        
    返回
    -------
    labels : numpy.ndarray
        聚类标签
    """
    print(f"使用K均值聚类算法，聚类数={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    return labels

def cluster_dbscan(features_scaled):
    """
    使用DBSCAN聚类
    
    参数
    ----------
    features_scaled : numpy.ndarray
        标准化后的特征数据
        
    返回
    -------
    labels : numpy.ndarray
        聚类标签
    """
    print("使用DBSCAN聚类算法...")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(features_scaled)
    return labels

def visualize_clusters_2d(features_scaled, labels, feature_names, method='pca', output_dir='../results'):
    """
    使用降维方法可视化聚类结果
    
    参数
    ----------
    features_scaled : numpy.ndarray
        标准化后的特征数据
    labels : numpy.ndarray
        聚类标签
    feature_names : list
        特征名称列表
    method : str, 可选
        降维方法，'pca'或't-sne'，默认为'pca'
    output_dir : str, 可选
        输出目录路径，默认为'../results'
    """
    n_clusters = len(np.unique(labels))
    if -1 in np.unique(labels):  # DBSCAN可能有噪声点标记为-1
        n_clusters -= 1
    
    # 创建随机颜色映射
    cmap = ListedColormap(plt.cm.tab10(np.linspace(0, 1, n_clusters+1)))
    
    # 降维到2D
    if method == 'pca':
        print("使用PCA降维可视化...")
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(features_scaled)
        title = 'PCA Dimensionality Reduction Cluster Visualization'
        filename = f'{output_dir}/cluster_visualization_pca.png'
    else:  # t-SNE
        print("使用t-SNE降维可视化...")
        reducer = TSNE(n_components=2, random_state=42)
        embedding = reducer.fit_transform(features_scaled)
        title = 't-SNE Dimensionality Reduction Cluster Visualization'
        filename = f'{output_dir}/cluster_visualization_tsne.png'
    
    # 创建散点图
    plt.figure(figsize=(10, 8))
    for i in np.unique(labels):
        if i == -1:  # DBSCAN noise points
            plt.scatter(embedding[labels==i, 0], embedding[labels==i, 1], 
                       c='black', marker='x', label='Noise')
        else:
            plt.scatter(embedding[labels==i, 0], embedding[labels==i, 1], 
                       c=[cmap(i)], marker='o', label=f'Cluster {i+1}')
    
    plt.title(title)
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(filename, dpi=300)

def visualize_feature_distribution(df, labels, output_dir='../results'):
    """
    可视化各个簇的特征分布
    
    参数
    ----------
    df : pandas.DataFrame
        原始数据
    labels : numpy.ndarray
        聚类标签
    output_dir : str, 可选
        输出目录路径，默认为'../results'
    """
    print("可视化各个簇的特征分布...")
    # 将标签添加到数据框
    df_cluster = df.copy()
    df_cluster['cluster'] = labels
    
    # 特征分布图
    features = ['amplitude', 'duration', 'fwhm', 'rise_time', 'decay_time', 'auc', 'snr']
    n_clusters = len(np.unique(labels))
    if -1 in np.unique(labels):  # DBSCAN可能有噪声点标记为-1
        n_clusters -= 1
    
    # 设置图形尺寸
    fig, axes = plt.subplots(len(features), 1, figsize=(12, 4*len(features)))
    
    # 遍历每个特征并创建箱形图
    for i, feature in enumerate(features):
        sns.boxplot(x='cluster', y=feature, hue='cluster', data=df_cluster, ax=axes[i], palette='Set2', legend=False)
        axes[i].set_title(f'{feature} Distribution in Each Cluster')
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel(feature)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/cluster_feature_distribution.png', dpi=300)

def analyze_clusters(df, labels, output_dir='../results'):
    """
    分析各个簇的特征统计信息
    
    参数
    ----------
    df : pandas.DataFrame
        原始数据
    labels : numpy.ndarray
        聚类标签
    output_dir : str, 可选
        输出目录路径，默认为'../results'
        
    返回
    -------
    cluster_stats : pandas.DataFrame
        每个簇的特征统计信息
    """
    print("分析各个簇的特征统计信息...")
    # 将标签添加到数据框
    df_cluster = df.copy()
    df_cluster['cluster'] = labels
    
    # 特征列表
    features = ['amplitude', 'duration', 'fwhm', 'rise_time', 'decay_time', 'auc', 'snr']
    
    # 计算每个簇的特征均值
    cluster_means = df_cluster.groupby('cluster')[features].mean()
    
    # 计算每个簇的标准差
    cluster_stds = df_cluster.groupby('cluster')[features].std()
    
    # 计算每个簇的样本数
    cluster_counts = df_cluster.groupby('cluster').size().rename('count')
    
    # 合并统计信息
    cluster_stats = pd.concat([cluster_means, cluster_stds.add_suffix('_std'), cluster_counts], axis=1)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 保存到CSV
    cluster_stats.to_csv(f'{output_dir}/cluster_statistics.csv')
    
    print(f"聚类统计信息已保存到 '{output_dir}/cluster_statistics.csv'")
    return cluster_stats

def visualize_cluster_radar(cluster_stats, output_dir='../results'):
    """
    使用雷达图可视化各簇的特征
    
    参数
    ----------
    cluster_stats : pandas.DataFrame
        每个簇的特征统计信息
    output_dir : str, 可选
        输出目录路径，默认为'../results'
    """
    print("使用雷达图可视化各簇的特征...")
    features = ['amplitude', 'duration', 'fwhm', 'rise_time', 'decay_time', 'auc', 'snr']
    
    # 获取均值数据
    means = cluster_stats[features]
    
    # 标准化均值，使其适合雷达图
    scaler = StandardScaler()
    means_scaled = pd.DataFrame(scaler.fit_transform(means), 
                               index=means.index, columns=means.columns)
    
    # 准备绘图
    n_clusters = len(means_scaled)
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    for i, idx in enumerate(means_scaled.index):
        values = means_scaled.loc[idx].values.tolist()
        values += values[:1]  # 闭合雷达图
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {idx+1}')
        ax.fill(angles, values, alpha=0.1)
    
    # 设置图的属性
    ax.set_thetagrids(np.degrees(angles[:-1]), features)
    ax.set_title('Comparison of Cluster Features using Radar Chart')
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/cluster_radar.png', dpi=300)

def add_cluster_to_excel(input_file, output_file, labels):
    """
    将聚类标签添加到原始Excel文件
    
    参数
    ----------
    input_file : str
        输入文件路径
    output_file : str
        输出文件路径
    labels : numpy.ndarray
        聚类标签
    """
    print("将聚类标签添加到原始数据...")
    # 读取原始数据
    df = pd.read_excel(input_file)
    
    # 添加聚类列
    df['cluster'] = labels
    
    # 保存到新的Excel文件
    df.to_excel(output_file, index=False)
    print(f"聚类结果已保存到 {output_file}")

def visualize_neuron_cluster_distribution(df, labels, k_value=None, output_dir='../results'):
    """
    可视化不同神经元的聚类分布
    
    参数
    ----------
    df : pandas.DataFrame
        原始数据
    labels : numpy.ndarray
        聚类标签
    k_value : int, 可选
        当前使用的K值，用于文件命名
    output_dir : str, 可选
        输出目录路径，默认为'../results'
    """
    print("可视化不同神经元的聚类分布...")
    # 将标签添加到数据框
    df_cluster = df.copy()
    df_cluster['cluster'] = labels
    
    # 计算每个神经元不同簇的数量
    cluster_counts = df_cluster.groupby(['neuron', 'cluster']).size().unstack().fillna(0)
    
    # 绘制堆叠条形图
    ax = cluster_counts.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab10')
    ax.set_title(f'Cluster Distribution for Different Neurons (k={len(np.unique(labels))})')
    ax.set_xlabel('Neuron')
    ax.set_ylabel('Number of Calcium Transients')
    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 根据k_value调整文件名
    if k_value:
        filename = f'{output_dir}/neuron_cluster_distribution_k{k_value}.png'
    else:
        filename = f'{output_dir}/neuron_cluster_distribution.png'
    
    plt.savefig(filename, dpi=300)
    print(f"神经元聚类分布图已保存到: {filename}")

def visualize_wave_type_distribution(df, labels, output_dir='../results'):
    """
    可视化不同波形类型在各聚类中的分布
    
    参数
    ----------
    df : pandas.DataFrame
        包含wave_type信息的数据
    labels : numpy.ndarray
        聚类标签
    output_dir : str, 可选
        输出目录路径
    """
    if 'wave_type' not in df.columns:
        print("数据中没有wave_type信息，跳过波形类型分布可视化")
        return
        
    print("可视化不同波形类型在各聚类中的分布...")
    
    # 将标签添加到数据框
    df_cluster = df.copy()
    df_cluster['cluster'] = labels
    
    # 计算每个聚类中不同波形类型的分布
    wave_type_counts = df_cluster.groupby(['cluster', 'wave_type']).size().unstack().fillna(0)
    
    # 计算百分比
    wave_type_pcts = wave_type_counts.div(wave_type_counts.sum(axis=1), axis=0) * 100
    
    # 绘制堆叠条形图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绝对数量图
    wave_type_counts.plot(kind='bar', stacked=True, ax=ax1, colormap='Set3')
    ax1.set_title('Wave Type Count in Each Cluster')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Count')
    ax1.legend(title='Wave Type')
    ax1.grid(True, alpha=0.3)
    
    # 百分比图
    wave_type_pcts.plot(kind='bar', stacked=True, ax=ax2, colormap='Set3')
    ax2.set_title('Wave Type Percentage in Each Cluster')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Percentage (%)')
    ax2.legend(title='Wave Type')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/wave_type_distribution.png', dpi=300)
    print(f"波形类型分布图已保存到: {output_dir}/wave_type_distribution.png")

def analyze_subpeaks(df, labels, output_dir='../results'):
    """
    分析各聚类中子峰特征
    
    参数
    ----------
    df : pandas.DataFrame
        包含subpeaks_count信息的数据
    labels : numpy.ndarray
        聚类标签
    output_dir : str, 可选
        输出目录路径
    """
    if 'subpeaks_count' not in df.columns:
        print("数据中没有subpeaks_count信息，跳过子峰分析")
        return
        
    print("分析各聚类中的子峰特征...")
    
    # 将标签添加到数据框
    df_cluster = df.copy()
    df_cluster['cluster'] = labels
    
    # 绘制子峰数量箱线图 - 修复FutureWarning
    plt.figure(figsize=(10, 6))
    # 修改前: sns.boxplot(x='cluster', y='subpeaks_count', data=df_cluster, palette='Set2')
    # 修改后: 将x变量分配给hue，并设置legend=False
    sns.boxplot(x='cluster', y='subpeaks_count', hue='cluster', data=df_cluster, palette='Set2', legend=False)
    plt.title('Distribution of Subpeaks Count in Each Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Subpeaks')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/subpeaks_distribution.png', dpi=300)
    
    # 计算各聚类中子峰的统计信息
    subpeak_stats = df_cluster.groupby('cluster')['subpeaks_count'].agg(['mean', 'median', 'std', 'min', 'max'])
    subpeak_stats.to_csv(f'{output_dir}/subpeaks_statistics.csv')
    print(f"子峰统计信息已保存到 {output_dir}/subpeaks_statistics.csv")

def compare_multiple_k(features_scaled, feature_names, df_clean, k_values, input_file, output_dir='../results'):
    """
    比较不同K值的聚类效果
    
    参数
    ----------
    features_scaled : numpy.ndarray
        标准化后的特征数据
    feature_names : list
        特征名称列表
    df_clean : pandas.DataFrame
        清洗后的数据
    k_values : list
        要比较的K值列表
    input_file : str
        输入文件路径，用于生成输出文件名
    output_dir : str, 可选
        输出目录路径，默认为'../results'
    """
    print(f"正在比较不同K值的聚类效果: {k_values}...")
    
    # 计算每个K值的轮廓系数
    silhouette_scores_dict = {}
    
    # 创建比较图
    fig, axes = plt.subplots(1, len(k_values), figsize=(5*len(k_values), 5))
    if len(k_values) == 1:
        axes = [axes]
    
    for i, k in enumerate(k_values):
        # 执行K-means聚类
        labels = cluster_kmeans(features_scaled, k)
        
        # 计算轮廓系数
        sil_score = silhouette_score(features_scaled, labels)
        silhouette_scores_dict[k] = sil_score
        
        # 使用PCA降维可视化
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(features_scaled)
        
        # 绘制聚类结果
        cmap = ListedColormap(plt.cm.tab10(np.linspace(0, 1, k+1)))
        
        # 在子图中绘制
        for j in range(k):
            axes[i].scatter(embedding[labels==j, 0], embedding[labels==j, 1], 
                         c=[cmap(j)], marker='o', label=f'Cluster {j+1}')
        
        axes[i].set_title(f'K={k}, Silhouette={sil_score:.3f}')
        axes[i].set_xlabel('PCA Dimension 1')
        axes[i].set_ylabel('PCA Dimension 2')
        axes[i].grid(True, alpha=0.3)
        
        # 保存该K值的结果
        output_file = f'{output_dir}/all_neurons_transients_clustered_k{k}.xlsx'
        add_cluster_to_excel(input_file, output_file, labels)
        
        # 生成该K值的特征分布图
        visualize_feature_distribution(df_clean, labels, output_dir=output_dir)
        plt.savefig(f'{output_dir}/cluster_feature_distribution_k{k}.png', dpi=300)
        
        # 神经元簇分布
        visualize_neuron_cluster_distribution(df_clean, labels, k_value=k, output_dir=output_dir)
        
        # 增加波形类型分析
        visualize_wave_type_distribution(df_clean, labels, output_dir=f'{output_dir}/k{k}')
        
        # 增加子峰分析
        analyze_subpeaks(df_clean, labels, output_dir=f'{output_dir}/k{k}')
    
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/k_comparison.png', dpi=300)
    
    # 绘制轮廓系数比较图
    plt.figure(figsize=(8, 5))
    plt.bar(silhouette_scores_dict.keys(), silhouette_scores_dict.values(), color='skyblue')
    plt.title('Silhouette Score Comparison for Different K Values')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(True, alpha=0.3)
    plt.xticks(list(silhouette_scores_dict.keys()))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/silhouette_comparison.png', dpi=300)
    
    print("不同K值对比完成，结果已保存")
    
    # 返回轮廓系数最高的K值
    best_k = max(silhouette_scores_dict, key=silhouette_scores_dict.get)
    return best_k

def visualize_cluster_waveforms(df, labels, output_dir='../results', raw_data_path=None, raw_data_dir=None, logger=None):
    """
    可视化不同聚类类别的平均钙爆发波形
    
    参数
    ----------
    df : pandas.DataFrame
        包含钙爆发特征的数据框，必须包含start_idx, peak_idx, end_idx和neuron字段
    labels : numpy.ndarray
        聚类标签
    output_dir : str, 可选
        输出目录路径，默认为'../results'
    raw_data_path : str, 可选
        单个原始数据文件路径
    raw_data_dir : str, 可选
        原始数据文件目录，用于查找多个数据文件
    """
    print("正在可视化不同聚类类别的平均钙爆发波形...")
    
    # 设置时间窗口（采样点数）- 减小窗口大小以提高匹配成功率
    time_window = 200
    
    # 将标签添加到数据框
    df_cluster = df.copy()
    df_cluster['cluster'] = labels
    
    # 检查必要的字段
    required_fields = ['start_idx', 'peak_idx', 'end_idx', 'neuron']
    if not all(field in df_cluster.columns for field in required_fields):
        print("错误: 数据中缺少必要字段(start_idx, peak_idx, end_idx, neuron)，无法绘制波形")
        return
    
    # 获取聚类的数量
    n_clusters = len(np.unique(labels))
    if -1 in np.unique(labels):  # DBSCAN可能有噪声点标记为-1
        n_clusters -= 1
    
    # 尝试加载原始数据
    raw_data_dict = {}
    
    # 检查是否存在dataset列，表示合并了不同数据集
    has_dataset_column = 'dataset' in df_cluster.columns
    
    # 检查是否包含源文件信息
    has_source_info = all(col in df_cluster.columns for col in ['source_file', 'source_path', 'source_abs_path'])
    if has_source_info:
        print("检测到源文件路径信息，将优先使用这些信息加载原始数据")
    
    if raw_data_path:
        # 如果指定了单个原始数据文件路径
        try:
            print(f"加载原始数据从: {raw_data_path}")
            raw_data = pd.read_excel(raw_data_path)
            # 使用文件名作为数据集名称
            dataset_name = os.path.splitext(os.path.basename(raw_data_path))[0]
            raw_data_dict[dataset_name] = raw_data
            print(f"  已加载数据集: {dataset_name}, 形状: {raw_data.shape}")
        except Exception as e:
            print(f"无法加载原始数据: {str(e)}")
            return
    elif raw_data_dir:
        # 如果指定了原始数据目录，查找所有Excel文件
        try:
            excel_files = glob.glob(os.path.join(raw_data_dir, "**/*.xlsx"), recursive=True)
            print(f"在目录{raw_data_dir}下找到{len(excel_files)}个Excel文件")
            
            for file in excel_files:
                # 使用文件名作为数据集名称，而不是目录名
                dataset_name = os.path.splitext(os.path.basename(file))[0]
                try:
                    raw_data = pd.read_excel(file)
                    raw_data_dict[dataset_name] = raw_data
                    print(f"  已加载数据集: {dataset_name}, 形状: {raw_data.shape}")
                except Exception as e:
                    print(f"  加载数据集{dataset_name}失败: {str(e)}")
        except Exception as e:
            print(f"搜索原始数据文件时出错: {str(e)}")
            return
    else:
        # 使用源文件信息加载原始数据（如果可用）
        if has_source_info:
            # 获取不同的源文件
            unique_source_files = df_cluster['source_path'].unique()
            print(f"从事件数据中检测到 {len(unique_source_files)} 个不同的源文件")
            
            # 使用项目根目录（通常是工作目录的上一级）作为基础
            root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
            print(f"使用项目根目录: {root_dir}")
            
            # 加载每个源文件
            for source_path in unique_source_files:
                try:
                    # 构建绝对路径
                    if os.path.isabs(source_path):
                        abs_path = source_path
                    else:
                        abs_path = os.path.join(root_dir, source_path)
                    
                    if os.path.exists(abs_path):
                        print(f"加载源文件: {abs_path}")
                        raw_data = pd.read_excel(abs_path)
                        dataset_name = os.path.splitext(os.path.basename(abs_path))[0]
                        raw_data_dict[dataset_name] = raw_data
                        print(f"  成功加载数据集: {dataset_name}, 形状: {raw_data.shape}")
                    else:
                        print(f"  源文件不存在: {abs_path}")
                except Exception as e:
                    print(f"  加载源文件 {source_path} 失败: {str(e)}")
        
        # 如果无法使用源文件信息或未找到任何文件，尝试使用默认位置
        if not raw_data_dict:
            try:
                # 直接指定原始数据路径
                raw_data_path = "../datasets/processed_EMtrace.xlsx"
                print(f"尝试加载默认原始数据从: {raw_data_path}")
                
                # 加载原始数据
                raw_data = pd.read_excel(raw_data_path)
                dataset_name = os.path.splitext(os.path.basename(raw_data_path))[0]
                raw_data_dict[dataset_name] = raw_data
                print(f"成功加载原始数据，形状: {raw_data.shape}")
            except Exception as e:
                print(f"无法加载默认原始数据: {str(e)}")
                print("尝试在../datasets目录下搜索原始数据...")
                
                try:
                    # 尝试搜索datasets目录下的所有Excel文件
                    datasets_dir = "../datasets"
                    excel_files = glob.glob(os.path.join(datasets_dir, "*.xlsx"))
                    
                    if excel_files:
                        for file in excel_files:
                            dataset_name = os.path.splitext(os.path.basename(file))[0]
                            try:
                                raw_data = pd.read_excel(file)
                                raw_data_dict[dataset_name] = raw_data
                                print(f"  已加载数据集: {dataset_name}, 形状: {raw_data.shape}")
                            except Exception as e:
                                print(f"  加载数据集{dataset_name}失败: {str(e)}")
                    else:
                        print("在../datasets目录下未找到任何Excel文件")
                        return
                except Exception as e:
                    print(f"搜索原始数据时出错: {str(e)}")
                    return
    
    if not raw_data_dict:
        print("未能加载任何原始数据，无法可视化波形")
        return
    
    # 打印所有可用神经元列以供调试
    print("原始数据中的神经元列：")
    for dataset_name, data in raw_data_dict.items():
        neuron_cols = [col for col in data.columns if col.startswith('n') and col[1:].isdigit()]
        print(f"  数据集 {dataset_name}: {len(neuron_cols)} 个神经元列 - {neuron_cols[:5]}...")
    
    # 打印钙爆发数据中的神经元名称以供调试
    unique_neurons = df_cluster['neuron'].unique()
    print(f"钙爆发数据中的神经元: {len(unique_neurons)} 个 - {unique_neurons[:5]}...")
    
    # 创建神经元名称映射，处理可能的命名不一致问题
    neuron_mapping = {}
    for neuron_name in unique_neurons:
        # 检查神经元名称是否以'n'开头，并且第二个字符是数字
        if isinstance(neuron_name, str) and neuron_name.startswith('n') and neuron_name[1:].isdigit():
            # 保持原名
            neuron_mapping[neuron_name] = neuron_name
        elif isinstance(neuron_name, (int, float)) or (isinstance(neuron_name, str) and neuron_name.isdigit()):
            # 如果是纯数字，则转为"n数字"格式
            formatted_name = f"n{int(float(neuron_name))}"
            neuron_mapping[neuron_name] = formatted_name
    
    print(f"创建了 {len(neuron_mapping)} 个神经元名称映射")
    
    # 创建颜色映射 - 修复弃用的get_cmap方法
    try:
        # 尝试使用新的推荐方法
        cmap = plt.colormaps['tab10']
    except (AttributeError, KeyError):
        # 如果失败，回退到旧方法
        cmap = plt.cm.get_cmap('tab10', n_clusters)
    
    # 为每个聚类提取和平均波形
    plt.figure(figsize=(12, 8))
    
    # 记录不同聚类的平均波形数据，用于保存
    avg_waveforms = {}
    
    for cluster_id in range(n_clusters):
        # 获取当前聚类的所有钙爆发事件
        cluster_events = df_cluster[df_cluster['cluster'] == cluster_id]
        
        if len(cluster_events) == 0:
            continue
        
        # 收集所有波形，指定固定长度
        all_waveforms = []
        fixed_length = 2 * time_window + 1  # 固定长度：前后各time_window个点，加上峰值点
        time_points = np.linspace(-time_window, time_window, fixed_length)
        print(f"聚类 {cluster_id+1}: 使用固定长度{fixed_length}处理波形...")
        
        # 对每个事件，提取波形
        for idx, event in cluster_events.iterrows():
            neuron_col = event['neuron']
            
            # 应用神经元名称映射
            if neuron_col in neuron_mapping:
                neuron_col = neuron_mapping[neuron_col]
            
            # 确定使用哪个原始数据集
            raw_data = None
            
            # 1. 优先使用源文件信息精确匹配
            if has_source_info and 'source_file' in event:
                source_file = event['source_file']
                # 提取不带扩展名的文件名作为数据集名称
                source_dataset = os.path.splitext(source_file)[0]
                if source_dataset in raw_data_dict:
                    raw_data = raw_data_dict[source_dataset]
                    # 只对每个聚类的第一个事件显示此消息，避免输出过多
                    if idx == cluster_events.index[0]:
                        print(f"聚类 {cluster_id+1}: 使用源文件信息匹配原始数据")
            
            # 2. 如果无法通过源文件匹配，尝试使用dataset列
            if raw_data is None and has_dataset_column and 'dataset' in event and event['dataset'] in raw_data_dict:
                # 如果事件有数据集标识且该数据集已加载
                raw_data = raw_data_dict[event['dataset']]
            
            # 3. 如果前两种方法都失败，尝试所有数据集进行列名匹配
            if raw_data is None or neuron_col not in raw_data.columns:
                # 尝试所有数据集，查找包含此神经元的数据集
                for dataset_name, dataset_raw_data in raw_data_dict.items():
                    if neuron_col in dataset_raw_data.columns:
                        raw_data = dataset_raw_data
                        break
            
            if raw_data is None or neuron_col not in raw_data.columns:
                # 如果还找不到，尝试其他命名方式
                for dataset_name, dataset_raw_data in raw_data_dict.items():
                    # 尝试格式如 "n3" 或 "3" 等
                    if neuron_col.lstrip('n') in dataset_raw_data.columns:
                        neuron_col = neuron_col.lstrip('n')
                        raw_data = dataset_raw_data
                        break
                    elif f"n{neuron_col}" in dataset_raw_data.columns:
                        neuron_col = f"n{neuron_col}"
                        raw_data = dataset_raw_data
                        break
                
                # 如果仍找不到，则跳过此事件
                if raw_data is None or neuron_col not in raw_data.columns:
                    continue
            
            # 提取中心在peak_idx的时间窗口数据
            try:
                peak_idx = int(event['peak_idx'])
                start = max(0, peak_idx - time_window)
                end = min(len(raw_data), peak_idx + time_window + 1)
                
                # 如果提取的窗口不完整，尝试缩小窗口而不是完全跳过
                if end - start < 2 * time_window + 1:
                    actual_window = min(peak_idx, len(raw_data) - peak_idx - 1)
                    if actual_window < 10:  # 如果窗口太小则跳过
                        continue
                    
                    # 使用较小的窗口
                    start = max(0, peak_idx - actual_window)
                    end = min(len(raw_data), peak_idx + actual_window + 1)
                    # 调整时间点以匹配新窗口
                    fixed_length = 2 * actual_window + 1
                    time_points = np.linspace(-actual_window, actual_window, fixed_length)
                
                # 提取波形
                waveform = raw_data[neuron_col].values[start:end]
                
                # 确保波形长度与时间点匹配
                if len(waveform) != len(time_points):
                    # 重新生成一个固定长度的时间点数组，确保所有波形使用相同的时间轴
                    fixed_length = 2 * time_window + 1  # 固定长度：前后各time_window个点，加上峰值点
                    # 修剪或填充波形以匹配固定长度
                    if len(waveform) > fixed_length:
                        waveform = waveform[:fixed_length]
                    else:
                        # 填充不足部分
                        padding = np.full(fixed_length - len(waveform), np.nan)
                        waveform = np.concatenate([waveform, padding])
                    # 重置时间点数组为固定长度
                    time_points = np.linspace(-time_window, time_window, fixed_length)
                
                # 归一化处理：减去基线并除以峰值振幅
                # 忽略NaN值
                valid_indices = ~np.isnan(waveform)
                if np.sum(valid_indices) > 10:  # 确保有足够的有效点
                    baseline = np.nanmin(waveform)
                    amplitude = np.nanmax(waveform) - baseline
                    if amplitude > 0:  # 避免除以零
                        norm_waveform = (waveform - baseline) / amplitude
                        all_waveforms.append(norm_waveform)
            except Exception as e:
                print(f"处理事件 {idx} 时出错: {str(e)}")
                continue
        
        # 如果没有有效波形，跳过此聚类
        if len(all_waveforms) == 0:
            print(f"警告: 聚类 {cluster_id+1} 没有有效波形")
            continue
        
        # 预处理所有波形，确保长度一致
        # 转换为统一长度的波形数组之前，先确认所有波形长度是否已一致
        wave_lengths = [len(w) for w in all_waveforms]
        if len(set(wave_lengths)) > 1:
            # 存在长度不一致的情况，调整为统一长度
            max_len = max(wave_lengths)
            standardized_waveforms = []
            for w in all_waveforms:
                if len(w) < max_len:
                    padding = np.full(max_len - len(w), np.nan)
                    std_w = np.concatenate([w, padding])
                else:
                    std_w = w
                standardized_waveforms.append(std_w)
            all_waveforms = standardized_waveforms
            # 调整时间点以匹配
            time_points = np.linspace(-time_window, time_window, max_len)
            
        # 计算平均波形（忽略NaN值）
        all_waveforms_array = np.array(all_waveforms)
        avg_waveform = np.nanmean(all_waveforms_array, axis=0)
        std_waveform = np.nanstd(all_waveforms_array, axis=0)
        
        # 存储平均波形
        avg_waveforms[f"Cluster_{cluster_id+1}"] = {
            "time": time_points,
            "mean": avg_waveform,
            "std": std_waveform,
            "n_samples": len(all_waveforms)
        }
        
        # 绘制平均波形 - 移除标准差范围，仅绘制平均曲线
        plt.plot(time_points, avg_waveform, 
                 color=cmap(cluster_id), 
                 linewidth=2.5,  # 增加线宽使曲线更突出
                 label=f'Cluster {cluster_id+1} (n={len(all_waveforms)})')
        
        # 移除标准差范围的绘制
        # plt.fill_between(time_points, 
        #                  avg_waveform - std_waveform, 
        #                  avg_waveform + std_waveform, 
        #                  color=cmap(cluster_id), 
        #                  alpha=0.2)
    
    # 检查是否有任何有效的聚类波形
    if not avg_waveforms:
        # 如果函数被传入了logger参数，则使用它，否则创建一个新的logger
        if 'logger' in locals() and logger is not None:
            logger.warning("没有找到任何有效的波形数据，无法生成波形图")
        else:
            print("没有找到任何有效的波形数据，无法生成波形图")
        return
    
    # 设置图表属性
    plt.axvline(x=0, color='grey', linestyle='--', alpha=0.7)  # Mark peak position
    plt.title('Typical Calcium Wave Shape Comparison (Average Waveforms of Clusters)', fontsize=14)  # Change title to English
    plt.xlabel('Relative Time to Peak', fontsize=12)  # Change X-axis label to English
    plt.ylabel('Normalized Fluorescence Intensity (F/F0)', fontsize=12)  # Change Y-axis label to English
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # 保存图表
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_average_waveforms.png', dpi=300)
    
    # 保存平均波形数据 - 修复append方法已弃用的问题
    waveform_data = []
    for cluster_name, waveform_data_dict in avg_waveforms.items():
        for i, t in enumerate(waveform_data_dict["time"]):
            waveform_data.append({
                "cluster": cluster_name,
                "time_point": t,
                "mean_intensity": waveform_data_dict["mean"][i],
                "std_intensity": waveform_data_dict["std"][i],
                "n_samples": waveform_data_dict["n_samples"]
            })
    
    # 创建DataFrame
    waveform_df = pd.DataFrame(waveform_data)
    waveform_df.to_csv(f'{output_dir}/cluster_average_waveforms.csv', index=False)
    
    # 如果函数被传入了logger参数，则使用它，否则创建一个新的logger
    if 'logger' in locals() and logger is not None:
        logger.info(f"平均钙爆发波形可视化已保存到 {output_dir}/cluster_average_waveforms.png")
        logger.info(f"波形数据已保存到 {output_dir}/cluster_average_waveforms.csv")
    else:
        print(f"平均钙爆发波形可视化已保存到 {output_dir}/cluster_average_waveforms.png")
        print(f"波形数据已保存到 {output_dir}/cluster_average_waveforms.csv")

def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='钙爆发事件聚类分析工具')
    parser.add_argument('--k', type=int, help='指定聚类数K，不指定则自动确定最佳值')
    parser.add_argument('--compare', type=str, help='比较多个K值的效果，格式如"2,3,4,5"')
    parser.add_argument('--input', type=str, default='../results/processed_EMtrace/all_neurons_transients.xlsx', 
                        help='输入数据文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录，不指定则根据数据集名称自动生成')
    parser.add_argument('--weights', type=str, help='特征权重，格式如"amplitude:1.2,duration:0.8"')
    parser.add_argument('--raw_data_path', type=str, help='单个原始数据文件路径，用于波形可视化')
    parser.add_argument('--raw_data_dir', type=str, help='原始数据文件目录，用于波形可视化')
    parser.add_argument('--skip_waveform', action='store_true', help='跳过波形可视化步骤')
    parser.add_argument('--log_dir', type=str, default=None, help='日志文件保存目录，默认为输出目录下的logs文件夹')
    args = parser.parse_args()
    
    # 根据输入文件名生成输出目录
    if args.output is None:
        # 提取输入文件目录
        input_dir = os.path.dirname(args.input)
        # 如果输入在datasets目录，则用文件名，否则用所在目录名
        if 'datasets' in input_dir:
            # 提取数据文件名（不含扩展名）
            data_basename = os.path.basename(args.input)
            dataset_name = os.path.splitext(data_basename)[0]
            output_dir = f"../results/{dataset_name}"
        else:
            # 使用输入文件所在的目录名
            dir_name = os.path.basename(input_dir)
            output_dir = f"../results/{dir_name}"
    else:
        output_dir = args.output
    
    # 设置日志目录
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = os.path.join(output_dir, 'logs')
    
    # 初始化日志记录器
    logger = setup_logger(log_dir, "cluster")
    logger.info(f"开始钙爆发聚类分析任务")
    logger.info(f"输出目录设置为: {output_dir}")
    
    # 确保结果目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    input_file = args.input
    logger.info(f"开始加载数据文件: {input_file}")
    df = load_data(input_file, logger=logger)
    
    # 使用增强版预处理函数替换原预处理函数 - 修改为无条件应用默认权重
    feature_weights = {
        'amplitude': 1.5,  # 振幅权重更高
        'duration': 1.5,   # 持续时间权重更高
        'rise_time': 0.6,  # 上升时间权重较低
        'decay_time': 0.6, # 衰减时间权重较低
        'snr': 1.0,        # 信噪比正常权重
        'fwhm': 1.0,       # 半高宽正常权重
        'auc': 1.0         # 曲线下面积正常权重
    }
    
    # 如果用户指定了权重，则覆盖默认值
    if args.weights:
        # 解析用户指定的权重
        for pair in args.weights.split(','):
            feature, weight = pair.split(':')
            feature_weights[feature] = float(weight)
    
    logger.info("使用权重设置进行聚类分析")
    logger.info(f"使用的特征权重: {feature_weights}")
    features_scaled, feature_names, df_clean = enhance_preprocess_data(df, feature_weights=feature_weights)
    logger.info(f"提取的特征名称: {feature_names}")
    
    # 处理聚类数K
    if args.compare:
        # 如果需要比较多个K值
        k_values = [int(k) for k in args.compare.split(',')]
        logger.info(f"比较不同K值的聚类效果: {k_values}")
        best_k = compare_multiple_k(features_scaled, feature_names, df_clean, k_values, input_file, output_dir=output_dir)
        logger.info(f"在比较的K值中，K={best_k}的轮廓系数最高")
        # 使用最佳K值进行后续分析
        optimal_k = best_k
    else:
        # 如果指定了K值，使用指定值
        if args.k:
            optimal_k = args.k
            logger.info(f"使用指定的聚类数: K={optimal_k}")
        else:
            # 自动确定最佳聚类数
            logger.info("自动确定最佳聚类数...")
            optimal_k = determine_optimal_k(features_scaled, output_dir=output_dir)
            logger.info(f"确定的最佳聚类数: K={optimal_k}")
    
    # K均值聚类
    kmeans_labels = cluster_kmeans(features_scaled, optimal_k)
    
    # 可视化聚类结果
    visualize_clusters_2d(features_scaled, kmeans_labels, feature_names, method='pca', output_dir=output_dir)
    visualize_clusters_2d(features_scaled, kmeans_labels, feature_names, method='t-sne', output_dir=output_dir)
    
    # 特征分布可视化
    visualize_feature_distribution(df_clean, kmeans_labels, output_dir=output_dir)
    
    # 分析聚类结果
    cluster_stats = analyze_clusters(df_clean, kmeans_labels, output_dir=output_dir)
    
    # 雷达图可视化
    visualize_cluster_radar(cluster_stats, output_dir=output_dir)
    
    # 神经元簇分布
    visualize_neuron_cluster_distribution(df_clean, kmeans_labels, output_dir=output_dir)
    
    # 添加波形类型分析
    visualize_wave_type_distribution(df_clean, kmeans_labels, output_dir=output_dir)
    
    # 添加子峰分析
    analyze_subpeaks(df_clean, kmeans_labels, output_dir=output_dir)
    
    # 可视化不同聚类的平均钙爆发波形
    if not args.skip_waveform:
        logger.info("开始可视化各聚类的平均钙爆发波形...")
        visualize_cluster_waveforms(df_clean, kmeans_labels, output_dir=output_dir, 
                                    raw_data_path=args.raw_data_path, raw_data_dir=args.raw_data_dir,
                                    logger=logger)
    else:
        logger.info("跳过波形可视化步骤")
    
    # 将聚类标签添加到Excel
    output_file = f'{output_dir}/all_neurons_transients_clustered_k{optimal_k}.xlsx'
    add_cluster_to_excel(input_file, output_file, kmeans_labels)
    
    logger.info("聚类分析完成!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 设置应急日志
        error_logger = setup_logger(None, "cluster-error")
        error_logger.error(f"程序运行时出错: {str(e)}", exc_info=True)
        print(f"程序运行时出错: {str(e)}")
        raise