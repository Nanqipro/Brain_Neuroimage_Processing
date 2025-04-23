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
import glob  # 导入用于文件路径模式匹配的模块
import json
import pickle
from datetime import datetime
from scipy.spatial.distance import cdist

def load_data(file_path):
    """
    加载钙爆发数据
    
    参数
    ----------
    file_path : str
        数据文件路径
        
    返回
    -------
    df : pandas.DataFrame
        加载的数据
    """
    print(f"正在从{file_path}加载数据...")
    df = pd.read_excel(file_path)
    print(f"成功加载数据，共{len(df)}行")
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

def assign_wave_colors(cluster_centers, feature_names=None, color_map='tab10', persistent_file=None):
    """
    基于波形特征为聚类分配一致的颜色
    
    参数
    ----------
    cluster_centers : numpy.ndarray
        聚类中心点，每一行代表一个聚类的特征向量
    feature_names : list, 可选
        特征名称，用于确定哪些特征权重更高
    color_map : str, 可选
        使用的颜色映射名称，默认为'tab10'
    persistent_file : str, 可选
        保存颜色映射的文件路径，用于在不同运行之间保持一致
        
    返回
    -------
    color_indices : numpy.ndarray
        每个聚类对应的颜色索引，可用于cmap(color_indices[i])
    color_map_obj : matplotlib.colors.Colormap
        颜色映射对象
    """
    # 1. 归一化特征，使不同量纲的特征可比较
    scaler = StandardScaler()
    scaled_centers = scaler.fit_transform(cluster_centers)
    
    # 2. 特征加权（可选）
    # 如果提供了特征名称，根据特征重要性调整权重
    if feature_names:
        # 默认权重
        weights = np.ones(len(feature_names))
        
        # 增加重要特征的权重 - 提高波形特征权重
        important_features = ['amplitude', 'duration', 'fwhm', 'rise_time', 'decay_time']
        for i, feature in enumerate(feature_names):
            if feature in important_features:
                weights[i] = 2.0  # 增加权重，从1.5提高到2.0
        
        # 应用权重
        scaled_centers = scaled_centers * weights
        print(f"应用了特征权重，增强了波形形态特征的影响")
    
    # 3. 应用降维，将特征压缩到一维或二维
    if scaled_centers.shape[1] > 2:
        pca = PCA(n_components=2) # 改为使用2个主成分提供更好的聚类区分度
        reduced_features = pca.fit_transform(scaled_centers)
        # 合并两个主成分，但给第一个主成分更高的权重
        reduced_features = reduced_features[:, 0] * 0.8 + reduced_features[:, 1] * 0.2
    else:
        reduced_features = scaled_centers.mean(axis=1)
    
    # 4. 排序并分配颜色索引
    # 特征值越相似的聚类，颜色索引也越接近
    sorted_indices = np.argsort(reduced_features)
    n_clusters = len(cluster_centers)
    
    # 5. 创建颜色映射 - 使用更鲜明的颜色图
    colormap_choices = ['tab10', 'Set1', 'Dark2', 'hsv']
    try:
        # 尝试使用新的推荐方法
        cmap = plt.colormaps[color_map]
    except (AttributeError, KeyError):
        # 如果失败，回退到旧方法
        cmap = plt.cm.get_cmap(color_map, n_clusters)
    
    # 6. 将排序后的索引映射到颜色范围 [0, 1]，并增加颜色区分度
    # 先创建一个同样长度的原始索引数组
    original_indices = np.arange(n_clusters)
    # 然后计算每个原始索引在排序后的位置（反向映射）
    inverse_mapping = np.zeros_like(original_indices)
    for i, idx in enumerate(sorted_indices):
        inverse_mapping[idx] = i
    
    # 归一化到 [0, 1] 区间，但避免颜色太接近，增加区分度
    if n_clusters > 1:
        # 使用非线性映射增加颜色间隔
        color_indices = (inverse_mapping / (n_clusters - 1)) * 0.8 + 0.1
    else:
        color_indices = np.array([0.5])  # 单个聚类使用中间颜色
    
    # 7. 尝试保存/加载持久化的颜色映射
    if persistent_file:
        # 如果提供了持久化文件，尝试加载或保存
        try:
            if os.path.exists(persistent_file):
                # 加载已有的持久化颜色映射
                with open(persistent_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    saved_features = saved_data.get('features', [])
                    saved_colors = saved_data.get('colors', [])
                    
                    if len(saved_features) > 0 and len(saved_colors) > 0:
                        # 如果有历史数据，尝试匹配当前聚类中心与历史中心
                        # 计算距离矩阵
                        current_features = scaled_centers
                        distance_matrix = cdist(current_features, saved_features)
                        
                        # 为每个当前聚类中心找到最接近的历史中心
                        new_color_indices = np.zeros_like(color_indices)
                        for i in range(len(current_features)):
                            closest_idx = np.argmin(distance_matrix[i])
                            new_color_indices[i] = saved_colors[closest_idx]
                        
                        # 如果成功匹配，使用历史颜色
                        if len(new_color_indices) == len(color_indices):
                            print(f"从持久化文件加载颜色映射并匹配当前聚类: {persistent_file}")
                            color_indices = new_color_indices
            
            # 更新并保存当前颜色映射到持久化文件
            with open(persistent_file, 'wb') as f:
                # 保存缩放后的特征和颜色，供后续匹配使用
                data_to_save = {
                    'features': scaled_centers,
                    'colors': color_indices,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'n_clusters': n_clusters,
                    'feature_names': feature_names
                }
                pickle.dump(data_to_save, f)
                print(f"颜色映射已保存到: {persistent_file}")
        except Exception as e:
            print(f"处理持久化颜色映射时出错: {e}")
    
    return color_indices, cmap

def visualize_clusters_2d(features_scaled, labels, feature_names, method='pca', output_dir='../results', color_indices=None):
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
    color_indices : numpy.ndarray, 可选
        每个聚类的颜色索引，如果提供则使用这些颜色
    """
    n_clusters = len(np.unique(labels))
    if -1 in np.unique(labels):  # DBSCAN可能有噪声点标记为-1
        n_clusters -= 1
    
    # 创建颜色映射
    try:
        # 尝试使用新的推荐方法
        cmap = plt.colormaps['tab10']
    except (AttributeError, KeyError):
        # 如果失败，回退到旧方法
        cmap = plt.cm.get_cmap('tab10', n_clusters)
    
    # 降维到2D
    if method == 'pca':
        print("使用PCA降维可视化...")
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(features_scaled)
        title = 'PCA降维聚类可视化'
        filename = f'{output_dir}/cluster_visualization_pca.png'
    else:  # t-SNE
        print("使用t-SNE降维可视化...")
        reducer = TSNE(n_components=2, random_state=42)
        embedding = reducer.fit_transform(features_scaled)
        title = 't-SNE降维聚类可视化'
        filename = f'{output_dir}/cluster_visualization_tsne.png'
    
    # 创建散点图
    plt.figure(figsize=(10, 8))
    
    # 使用预先计算的颜色索引（如果提供）
    for i in np.unique(labels):
        if i == -1:  # DBSCAN噪声点
            plt.scatter(embedding[labels==i, 0], embedding[labels==i, 1], 
                       c='black', marker='x', label='噪声')
        else:
            if color_indices is not None:
                cluster_color = cmap(color_indices[i])
            else:
                # 使用默认颜色分配
                cluster_color = cmap(i / n_clusters if n_clusters > 1 else 0)
            
            plt.scatter(embedding[labels==i, 0], embedding[labels==i, 1], 
                       color=cluster_color, marker='o', label=f'聚类 {i+1}')
    
    plt.title(title, fontsize=14)
    plt.xlabel(f"{method.upper()} 维度 1", fontsize=12)
    plt.ylabel(f"{method.upper()} 维度 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(filename, dpi=300)
    
    return embedding

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

def add_cluster_to_excel(input_file, output_file, labels, df=None):
    """
    将聚类标签添加到原始Excel文件
    
    参数
    ----------
    input_file : str
        输入文件路径，如果为"combined_data"则使用传入的df参数
    output_file : str
        输出文件路径
    labels : numpy.ndarray
        聚类标签
    df : pandas.DataFrame, 可选
        当input_file为"combined_data"时使用的数据框
    """
    print("将聚类标签添加到原始数据...")
    
    if input_file == "combined_data" and df is not None:
        # 使用已有的数据框
        df_output = df.copy()
    else:
        # 读取原始数据
        df_output = pd.read_excel(input_file)
    
    # 添加聚类列
    df_output['cluster'] = labels
    
    # 保存到新的Excel文件
    df_output.to_excel(output_file, index=False)
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
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
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
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # 确保主输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算每个K值的轮廓系数
    silhouette_scores_dict = {}
    
    # 创建比较图
    fig, axes = plt.subplots(1, len(k_values), figsize=(5*len(k_values), 5))
    if len(k_values) == 1:
        axes = [axes]
    
    for i, k in enumerate(k_values):
        # 为每个K值创建单独的输出目录
        k_output_dir = os.path.join(output_dir, f'k{k}')
        os.makedirs(k_output_dir, exist_ok=True)
        
        print(f"\n分析K={k}的聚类效果...")
        
        # 执行K-means聚类
        labels = cluster_kmeans(features_scaled, k)
        
        # 计算轮廓系数
        sil_score = silhouette_score(features_scaled, labels)
        silhouette_scores_dict[k] = sil_score
        print(f"K={k}的轮廓系数: {sil_score:.4f}")
        
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
        output_file = f'{output_dir}/transients_clustered_k{k}.xlsx'
        add_cluster_to_excel(input_file, output_file, labels, df=df_clean)
        
        # 生成该K值的特征分布图
        visualize_feature_distribution(df_clean, labels, output_dir=k_output_dir)
        
        # 神经元簇分布
        visualize_neuron_cluster_distribution(df_clean, labels, k_value=k, output_dir=k_output_dir)
        
        # 波形类型分析
        visualize_wave_type_distribution(df_clean, labels, output_dir=k_output_dir)
        
        # 子峰分析
        analyze_subpeaks(df_clean, labels, output_dir=k_output_dir)
    
    plt.tight_layout()
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

def visualize_cluster_waveforms(df, labels, output_dir='../results', raw_data_path=None, raw_data_dir=None, color_indices=None):
    """
    可视化不同聚类的平均钙瞬变波形。
    
    Parameters
    ----------
    df : pandas.DataFrame
        包含计算的波形特征的数据帧
    labels : numpy.ndarray
        聚类标签
    output_dir : str, optional
        输出目录路径
    raw_data_path : str, optional
        原始数据文件路径，用于读取完整钙瞬变数据
    raw_data_dir : str, optional
        原始数据目录路径，用于读取多个钙瞬变数据文件
    color_indices : numpy.ndarray, optional
        颜色索引，用于一致的颜色映射
    """
    # 检查df中是否存在必要的字段
    required_fields = ['trace_id', 'raw_data_file', 'event_start_idx', 'event_end_idx', 'rise_start_idx', 'peak_idx', 'decay_end_idx']
    missing_fields = [field for field in required_fields if field not in df.columns]
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始钙瞬变波形可视化...")
    
    # 将裁剪的数据帧与标签一起保存
    if len(missing_fields) == 0:
        # 检查是否有原始数据路径
        if raw_data_path is not None:
            print(f"从单个文件加载原始钙数据: {raw_data_path}")
            raw_data = pd.read_excel(raw_data_path)
            plot_waveforms_from_raw(df, labels, raw_data, output_dir, color_indices)
            
        elif raw_data_dir is not None:
            print(f"从目录加载原始钙数据: {raw_data_dir}")
            # 收集原始数据文件列表
            raw_data_files = {}
            for file in os.listdir(raw_data_dir):
                if file.endswith('.xlsx') or file.endswith('.xls'):
                    full_path = os.path.join(raw_data_dir, file)
                    raw_data_files[file] = full_path
            
            if not raw_data_files:
                print(f"警告: 在目录 {raw_data_dir} 中未找到Excel文件")
                return
            
            # 按文件分组数据
            grouped_data = df.groupby('raw_data_file')
            
            # 对每个文件组执行波形处理
            all_mean_waves = []
            all_cluster_info = []
            
            for file_name, group in grouped_data:
                file_path = raw_data_files.get(file_name)
                if file_path and os.path.exists(file_path):
                    # 加载原始数据
                    print(f"处理文件: {file_name}")
                    raw_data = pd.read_excel(file_path)
                    
                    # 从此文件中获取标签
                    group_indices = group.index
                    group_labels = labels[group_indices]
                    
                    # 画波形并获取平均波形数据
                    mean_waves, cluster_info = plot_waveforms_from_raw(
                        group, group_labels, raw_data, 
                        os.path.join(output_dir, f"waveforms_{os.path.splitext(file_name)[0]}"),
                        color_indices
                    )
                    
                    # 收集所有平均波形
                    all_mean_waves.append(mean_waves)
                    all_cluster_info.append(cluster_info)
                else:
                    print(f"警告: 找不到文件 {file_name} 或文件路径 {file_path}")
            
            # 合并所有平均波形并创建总体图
            if all_mean_waves:
                # 找出所有数据集中的所有聚类
                all_clusters = set()
                for info in all_cluster_info:
                    all_clusters.update(info.keys())
                
                # 为每个聚类合并数据
                merged_waves = {}
                for cluster in all_clusters:
                    cluster_waves = []
                    for mean_wave, info in zip(all_mean_waves, all_cluster_info):
                        if cluster in info:
                            wave_data = mean_wave.get(cluster)
                            if wave_data is not None and len(wave_data) > 0:
                                # 确保有数据
                                cluster_waves.append(wave_data)
                    
                    if cluster_waves:
                        # 标准化波形长度
                        max_len = max(len(w) for w in cluster_waves)
                        norm_waves = []
                        for wave in cluster_waves:
                            # 如果波形太短，使用零填充
                            if len(wave) < max_len:
                                padded = np.zeros(max_len)
                                padded[:len(wave)] = wave
                                norm_waves.append(padded)
                            else:
                                norm_waves.append(wave[:max_len])
                        
                        # 计算平均波形
                        merged_waves[cluster] = np.mean(norm_waves, axis=0)
                
                # 创建整体波形图，带有一致的颜色映射
                plot_merged_waveforms(merged_waves, os.path.join(output_dir, "merged_waveforms"), color_indices)
        else:
            print("警告: 未提供原始数据路径，无法可视化波形")
    else:
        print(f"警告: 缺少波形可视化所需的字段: {missing_fields}")

def plot_merged_waveforms(merged_waves, output_path, color_indices=None):
    """
    绘制合并后的所有聚类平均波形。
    
    Parameters
    ----------
    merged_waves : dict
        每个聚类的平均波形数据
    output_path : str
        输出文件路径（不带扩展名）
    color_indices : numpy.ndarray, optional
        颜色索引，用于一致的颜色映射
    """
    plt.figure(figsize=(12, 8))
    
    clusters = sorted(merged_waves.keys())
    n_clusters = len(clusters)
    
    # 确定颜色映射
    if color_indices is not None:
        # 使用提供的颜色索引
        cmap = plt.cm.get_cmap('tab10', n_clusters)
        colors = [cmap(color_indices[cluster]) if cluster < len(color_indices) else cmap(cluster % n_clusters) 
                for cluster in clusters]
    else:
        # 使用默认颜色循环
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # 绘制每个聚类的平均波形
    for i, cluster in enumerate(clusters):
        wave = merged_waves[cluster]
        x = np.arange(len(wave))
        plt.plot(x, wave, label=f'聚类 {cluster}', color=colors[i], linewidth=2)
    
    plt.title('所有聚类的平均钙瞬变波形')
    plt.xlabel('时间点')
    plt.ylabel('荧光强度 (归一化)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300)
    plt.savefig(f"{output_path}.svg", format='svg')
    plt.close()

def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='钙爆发事件聚类分析工具')
    parser.add_argument('--k', type=int, help='指定聚类数K，不指定则自动确定最佳值')
    parser.add_argument('--compare', type=str, help='比较多个K值的效果，格式如"2,3,4,5"')
    parser.add_argument('--input', type=str, default='../results/all_datasets_transients/all_datasets_transients.xlsx', 
                       help='输入数据文件路径，可以是单个文件或合并后的文件')
    parser.add_argument('--input_dir', type=str, help='输入数据目录，会处理该目录下所有的transients.xlsx文件')
    parser.add_argument('--combine', action='store_true', 
                       help='是否合并指定目录下的所有钙爆发数据再进行聚类（与--input_dir一起使用）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录，不指定则根据数据集名称自动生成')
    parser.add_argument('--raw_data_dir', type=str, help='原始数据文件所在目录，用于波形可视化')
    parser.add_argument('--raw_data_path', type=str, help='单个原始数据文件路径，用于波形可视化')
    parser.add_argument('--skip_waveform', action='store_true', help='跳过波形可视化步骤')
    parser.add_argument('--weights', type=str, help='特征权重，格式如"amplitude:1.2,duration:0.8"')
    parser.add_argument('--color_mapping', type=str, default=None, 
                       help='颜色映射文件路径，用于保持不同运行之间聚类颜色的一致性')
    args = parser.parse_args()
    
    # 处理输入文件路径
    input_files = []
    
    if args.input_dir:
        # 如果提供了输入目录，搜索该目录下所有的transients.xlsx文件
        pattern = os.path.join(args.input_dir, "**", "*transients.xlsx")
        input_files = glob.glob(pattern, recursive=True)
        if not input_files:
            print(f"错误: 在目录{args.input_dir}下未找到任何匹配的transients.xlsx文件")
            return
        print(f"在目录{args.input_dir}下找到{len(input_files)}个钙爆发数据文件")
    else:
        # 否则使用单个输入文件
        if not os.path.exists(args.input):
            print(f"错误: 输入文件 {args.input} 不存在")
            return
        input_files = [args.input]
    
    # 如果需要合并多个输入文件
    if args.input_dir and args.combine and len(input_files) > 1:
        print("正在合并多个钙爆发数据文件...")
        
        all_data = []
        for file in input_files:
            try:
                df = load_data(file)
                # 添加数据源标识，使用文件名而非目录名
                dataset_name = os.path.splitext(os.path.basename(file))[0]
                df['dataset'] = dataset_name
                all_data.append(df)
            except Exception as e:
                print(f"处理文件{file}时出错: {str(e)}")
        
        if all_data:
            # 合并所有数据
            df = pd.concat(all_data, ignore_index=True)
            print(f"成功合并{len(all_data)}个数据文件，总共{len(df)}行数据")
            
            # 设置输出目录
            if args.output is None:
                output_dir = "../results/combined_transients_clustering"
            else:
                output_dir = args.output
        else:
            print("未能加载任何有效数据，请检查输入路径")
            return
    else:
        # 使用单个输入文件
        input_file = input_files[0]
        try:
            df = load_data(input_file)
        except Exception as e:
            print(f"加载文件{input_file}时出错: {str(e)}")
            return
        
        # 根据输入文件名生成输出目录
        if args.output is None:
            # 提取输入文件目录
            input_dir = os.path.dirname(input_file)
            # 提取数据文件名（不含扩展名）
            data_basename = os.path.basename(input_file)
            dataset_name = os.path.splitext(data_basename)[0]
            
            # 如果是all_datasets_transients.xlsx这种合并文件，使用专门的输出目录
            if dataset_name == "all_datasets_transients":
                output_dir = "../results/all_datasets_clustering"
            else:
                output_dir = f"../results/{dataset_name}_clustering"
        else:
            output_dir = args.output
    
    print(f"输出目录设置为: {output_dir}")
    
    # 确保结果目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用增强版预处理函数
    try:
        # 添加特征权重设置
        feature_weights = None
        if args.weights:
            feature_weights = {
                'amplitude': 1.5,  # 振幅权重更高
                'duration': 1.5,   # 持续时间权重更高
                'rise_time': 0.6,  # 上升时间权重较低
                'decay_time': 0.6, # 衰减时间权重较低
                'snr': 1.0,        # 信噪比正常权重
                'fwhm': 1.0        # 半高宽正常权重
            }
            # 解析用户指定的权重
            for pair in args.weights.split(','):
                feature, weight = pair.split(':')
                feature_weights[feature] = float(weight)
        
        features_scaled, feature_names, df_clean = enhance_preprocess_data(df, feature_weights=feature_weights)
    except Exception as e:
        print(f"预处理数据时出错: {str(e)}")
        return
    
    # 检查数据是否足够进行聚类
    if len(df_clean) < 10:
        print(f"错误: 有效数据不足(只有{len(df_clean)}行)，无法进行聚类分析")
        return
    
    # 处理聚类数K
    if args.compare:
        # 如果需要比较多个K值
        try:
            k_values = [int(k) for k in args.compare.split(',')]
            best_k = compare_multiple_k(features_scaled, feature_names, df_clean, k_values, input_file, output_dir=output_dir)
            print(f"在比较的K值中，K={best_k}的轮廓系数最高")
            # 使用最佳K值进行后续分析
            optimal_k = best_k
        except Exception as e:
            print(f"比较K值时出错: {str(e)}")
            return
    else:
        # 如果指定了K值，使用指定值
        if args.k:
            optimal_k = args.k
            print(f"使用指定的聚类数: K={optimal_k}")
        else:
            # 自动确定最佳聚类数
            try:
                optimal_k = determine_optimal_k(features_scaled, output_dir=output_dir)
            except Exception as e:
                print(f"确定最佳聚类数时出错: {str(e)}")
                print("使用默认聚类数K=5")
                optimal_k = 5
    
    # K均值聚类
    try:
        kmeans_labels = cluster_kmeans(features_scaled, optimal_k)
    except Exception as e:
        print(f"执行K-means聚类时出错: {str(e)}")
        return
    
    # 计算聚类中心点，用于颜色分配
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(features_scaled)
    cluster_centers = kmeans.cluster_centers_
    
    # 分配基于波形特征的一致颜色
    persistent_color_file = None
    if args.color_mapping:
        # 用户指定的文件
        persistent_color_file = args.color_mapping
    elif args.output:
        # 默认使用输出目录下的颜色映射文件
        persistent_color_file = f"{args.output}/cluster_color_mapping.json"
    else:
        # 如果没有指定输出目录，使用通用位置
        persistent_color_file = "../results/cluster_color_mapping.json"
    
    # 计算颜色映射
    color_indices, _ = assign_wave_colors(
        cluster_centers, 
        feature_names=feature_names,
        persistent_file=persistent_color_file
    )
    
    # 可视化聚类结果，使用一致的颜色
    visualize_clusters_2d(features_scaled, kmeans_labels, feature_names, 
                          method='pca', output_dir=output_dir, color_indices=color_indices)
    visualize_clusters_2d(features_scaled, kmeans_labels, feature_names, 
                          method='t-sne', output_dir=output_dir, color_indices=color_indices)
    
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
    
    # 可视化不同聚类的平均钙瞬变波形，使用一致的颜色
    if not args.skip_waveform:
        visualize_cluster_waveforms(df_clean, kmeans_labels, 
                                   output_dir=output_dir, 
                                   raw_data_path=args.raw_data_path, 
                                   raw_data_dir=args.raw_data_dir,
                                   color_indices=color_indices)
    
    # 将聚类标签添加到Excel
    output_file = f'{output_dir}/transients_clustered_k{optimal_k}.xlsx'
    add_cluster_to_excel(input_file, output_file, kmeans_labels)
    print(f"聚类结果已保存到: {output_file}")
    
    print("聚类分析完成!")

if __name__ == "__main__":
    main()