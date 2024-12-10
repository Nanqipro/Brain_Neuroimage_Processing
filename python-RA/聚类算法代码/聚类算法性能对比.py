# clustering_comparison.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import pairwise_distances, silhouette_score
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler
import warnings
import time
from tqdm import tqdm
import matplotlib

# 忽略警告
warnings.filterwarnings("ignore")

# 设置Matplotlib字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_data(filepath):
    """
    读取神经元数据。

    参数:
        filepath (str): CSV文件路径。

    返回:
        labels (np.array): 神经元标签。
        time (np.array): 时间点。
        neuron_data (np.ndarray): 神经元钙离子浓度数据。
    """
    data = pd.read_excel(r"C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day6\calcium_data.xlsx")

    # 提取标签、时间和神经元数据
    labels = data.iloc[0, 1:].values  # 第一行的标签，除去第一列时间
    time = data.iloc[1:, 0].values  # 第一列的时间，除去第一行标签
    neuron_data = data.iloc[1:, 1:].values.astype(float)  # 神经元钙离子浓度数据

    return labels, time, neuron_data


def normalize_data(neuron_data):
    """
    对数据进行归一化处理（缩放到[0,1]范围）。

    参数:
        neuron_data (np.ndarray): 原始神经元数据。

    返回:
        normalized_data (np.ndarray): 归一化后的数据。
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(neuron_data)
    return normalized_data


def compute_manhattan_distance(data):
    """
    计算曼哈顿距离矩阵。

    参数:
        data (np.ndarray): 神经元数据，形状为（时间步数，神经元数）。

    返回:
        manhattan_dist (np.ndarray): 曼哈顿距离矩阵。
    """
    manhattan_dist = pairwise_distances(data.T, metric='manhattan')
    return manhattan_dist


def compute_hausdorff_distance(data):
    """
    计算Hausdorff距离矩阵。

    参数:
        data (np.ndarray): 神经元数据，形状为（时间步数，神经元数）。

    返回:
        hausdorff_dist (np.ndarray): Hausdorff距离矩阵。
    """
    n_neurons = data.shape[1]
    hausdorff_dist = np.zeros((n_neurons, n_neurons))

    print("开始计算Hausdorff距离矩阵...")
    start_time = time.time()

    for i in tqdm(range(n_neurons), desc="Hausdorff"):
        for j in range(i + 1, n_neurons):
            u = data[:, i].reshape(-1, 1)
            v = data[:, j].reshape(-1, 1)
            d1 = directed_hausdorff(u, v)[0]
            d2 = directed_hausdorff(v, u)[0]
            dist = max(d1, d2)
            hausdorff_dist[i, j] = dist
            hausdorff_dist[j, i] = dist

    end_time = time.time()
    print(f"Hausdorff距离矩阵计算完成，耗时 {end_time - start_time:.2f} 秒")

    return hausdorff_dist


def compute_emd_distance(data):
    """
    计算Earth Mover's Distance (EMD) 距离矩阵。

    参数:
        data (np.ndarray): 神经元数据，形状为（时间步数，神经元数）。

    返回:
        emd_dist (np.ndarray): EMD距离矩阵。
    """
    n_neurons = data.shape[1]
    emd_dist = np.zeros((n_neurons, n_neurons))

    print("开始计算EMD距离矩阵...")
    start_time = time.time()

    # 生成位置向量（假设时间步数为位置）
    positions = np.arange(data.shape[0]).astype(float)

    for i in tqdm(range(n_neurons), desc="EMD"):
        u = data[:, i]
        u_sum = np.sum(u)
        if u_sum == 0:
            u_norm = np.ones_like(u) / len(u)
        else:
            u_norm = u / u_sum

        for j in range(i + 1, n_neurons):
            v = data[:, j]
            v_sum = np.sum(v)
            if v_sum == 0:
                v_norm = np.ones_like(v) / len(v)
            else:
                v_norm = v / v_sum

            # 计算一维的EMD（等同于Wasserstein距离）
            dist = wasserstein_distance(positions, positions, u_weights=u_norm, v_weights=v_norm)
            emd_dist[i, j] = dist
            emd_dist[j, i] = dist

    end_time = time.time()
    print(f"EMD距离矩阵计算完成，耗时 {end_time - start_time:.2f} 秒")

    return emd_dist


def apply_kmeans(data, n_clusters):
    """
    应用KMeans聚类算法。

    参数:
        data (np.ndarray): 神经元数据，形状为（时间步数，神经元数）。
        n_clusters (int): 聚类数量。

    返回:
        labels (np.ndarray): 聚类标签。
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data.T)  # 转置使每列为一个样本
    return labels


def apply_dbscan(distance_matrix, eps=80, min_samples=3):
    """
    应用DBSCAN聚类算法。

    参数:
        distance_matrix (np.ndarray): 预先计算的距离矩阵。
        eps (float): DBSCAN的eps参数。
        min_samples (int): DBSCAN的min_samples参数。

    返回:
        labels (np.ndarray): 聚类标签。
    """
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(distance_matrix)
    return labels


def apply_hierarchical(distance_matrix, n_clusters):
    """
    应用层次聚类算法。

    参数:
        distance_matrix (np.ndarray): 预先计算的距离矩阵。
        n_clusters (int): 聚类数量。

    返回:
        labels (np.ndarray): 聚类标签。
    """
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    labels = hierarchical.fit_predict(distance_matrix)
    return labels


def evaluate_clustering(data, labels, metric='euclidean'):
    """
    评估聚类性能，计算轮廓系数。

    参数:
        data (np.ndarray or np.ndarray): 神经元数据（样本为神经元）或距离矩阵。
        labels (np.ndarray): 聚类标签。
        metric (str): 距离度量方式。

    返回:
        score (float): 轮廓系数。
    """
    unique_labels = set(labels)
    if len(unique_labels) <= 1 or len(unique_labels) == len(labels):
        return -1  # 无法计算轮廓系数
    try:
        if metric == 'precomputed':
            score = silhouette_score(data, labels, metric='precomputed')
        else:
            score = silhouette_score(data, labels, metric=metric)
    except Exception as e:
        print(f"计算轮廓系数时出错: {e}")
        score = -1
    return score


def main():
    # 文件路径（请根据实际情况修改）
    filepath = 'neuron_data.csv'

    # 加载数据
    print("加载数据...")
    labels, time, neuron_data = load_data(filepath)
    print(f"数据加载完成，神经元数量: {neuron_data.shape[1]}, 时间步数: {neuron_data.shape[0]}")

    # 数据归一化
    print("对数据进行归一化处理...")
    normalized_data = normalize_data(neuron_data)

    # 计算距离矩阵
    print("计算曼哈顿距离矩阵...")
    manhattan_dist = compute_manhattan_distance(normalized_data)

    print("计算Hausdorff距离矩阵...")
    hausdorff_dist = compute_hausdorff_distance(normalized_data)

    print("计算EMD距离矩阵...")
    emd_dist = compute_emd_distance(normalized_data)

    distance_metrics = {
        '曼哈顿距离': manhattan_dist,
        'Hausdorff距离': hausdorff_dist,
        'EMD距离': emd_dist
    }

    # 定义聚类算法
    clustering_algorithms = ['KMeans', 'DBSCAN', 'Hierarchical']

    # 预定义的簇数量（根据实际情况调整）
    n_clusters = 5

    # 存储性能结果
    performance = {algo: [] for algo in clustering_algorithms}

    # 遍历每种距离度量
    for metric_name, distance_matrix in distance_metrics.items():
        print(f"\n处理距离度量: {metric_name}")

        # KMeans聚类
        print("应用KMeans聚类...")
        km_labels = apply_kmeans(normalized_data, n_clusters)
        if metric_name == '曼哈顿距离':
            # 传入转置后的数据，以确保样本数量一致
            km_score = evaluate_clustering(normalized_data.T, km_labels, metric='manhattan')
        else:
            # KMeans始终基于欧几里得距离
            km_score = evaluate_clustering(normalized_data.T, km_labels, metric='euclidean')
        performance['KMeans'].append(km_score)
        print(f"KMeans轮廓系数: {km_score:.4f}")

        # DBSCAN聚类
        print("应用DBSCAN聚类...")
        # 可以根据数据特性调整eps和min_samples
        db_labels = apply_dbscan(distance_matrix, eps=80, min_samples=3)
        db_score = evaluate_clustering(distance_matrix, db_labels, metric='precomputed')
        performance['DBSCAN'].append(db_score)
        print(f"DBSCAN轮廓系数: {db_score:.4f}")

        # 层次聚类
        print("应用Hierarchical聚类...")
        hierarchical_labels = apply_hierarchical(distance_matrix, n_clusters)
        hierarchical_score = evaluate_clustering(distance_matrix, hierarchical_labels, metric='precomputed')
        performance['Hierarchical'].append(hierarchical_score)
        print(f"Hierarchical轮廓系数: {hierarchical_score:.4f}")

    # 生成对比图
    print("\n生成聚类性能对比图...")
    x = np.arange(len(distance_metrics))
    width = 0.2  # 条形宽度

    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, algo in enumerate(clustering_algorithms):
        scores = performance[algo]
        ax.bar(x + idx * width - width, scores, width, label=algo)

    ax.set_xlabel('距离度量', fontsize=14)
    ax.set_ylabel('轮廓系数', fontsize=14)
    ax.set_title('不同聚类算法在不同距离度量下的性能对比', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(distance_metrics.keys(), fontsize=12)
    ax.set_ylim(-1, 1)  # 轮廓系数范围通常在-1到1之间
    ax.legend(fontsize=12)

    # 在每个条形上添加数值标签
    for idx, algo in enumerate(clustering_algorithms):
        for i, score in enumerate(performance[algo]):
            if score != -1:
                ax.text(x[i] + idx * width - width, score + 0.02, f"{score:.2f}", ha='center', va='bottom', fontsize=10)
            else:
                ax.text(x[i] + idx * width - width, 0, "N/A", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()
    print("聚类性能对比图生成完毕。")


if __name__ == "__main__":
    main()
