import pandas as pd
import numpy as np
from tqdm import tqdm
from numba import cuda
import math

# 读取单个工作表的数据
file_path = 'calcium_window_data_separate_sheets.xlsx'
sheet_name = 'Window_30_Step_5'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# 设置聚类参数
n_clusters = 5
max_iter = 100

# 初始化存储特征的列表
features = []

# 计算每个窗口的峰值、标准差和平均值
for i, row in df.iterrows():
    window_data = row[2:].values  # 跳过 'Neuron' 和 'Start Time' 列
    peak_value = window_data.max()
    std_dev_value = window_data.std()
    mean_value = window_data.mean()

    features.append({
        'Neuron': row['Neuron'],
        'Start Time': row['Start Time'],
        'Peak': peak_value,
        'StdDev': std_dev_value,
        'Mean': mean_value
    })

# 转换为 DataFrame
features_df = pd.DataFrame(features)

# 将特征数据提取为 numpy 数组，用于 EMD 计算
data = features_df[['Peak', 'StdDev', 'Mean']].values

# 随机初始化质心
np.random.seed(42)
initial_centers = data[np.random.choice(data.shape[0], n_clusters, replace=False)]


# CUDA 函数：用于在 GPU 上计算 Wasserstein 距离
@cuda.jit
def calculate_emd(point, center, result):
    i = cuda.grid(1)
    if i < point.shape[0]:
        dist = 0.0
        for k in range(point.shape[1]):
            dist += abs(point[i, k] - center[i, k])
        result[i] = dist


# 聚类函数：在 GPU 上执行 EMD K-means，并使用进度条
def emd_kmeans_cuda(data, centers, max_iter=100):
    labels = np.zeros(len(data), dtype=np.int32)
    new_centers = np.empty_like(centers)

    # 计算线程和块的大小
    threads_per_block = 1024
    blocks_per_grid = (data.shape[0] + threads_per_block - 1) // threads_per_block

    for iteration in tqdm(range(max_iter), desc="Clustering Progress"):
        # 将数据复制到设备
        data_device = cuda.to_device(data)
        centers_device = cuda.to_device(centers)
        result = cuda.device_array(data.shape[0], dtype=np.float32)

        # 分配样本到最接近的质心
        for i, point in enumerate(data):
            min_dist = math.inf
            min_index = 0
            for j, center in enumerate(centers):
                # 调用 CUDA 核函数
                calculate_emd[blocks_per_grid, threads_per_block](data_device, centers_device, result)
                dist = result.copy_to_host()
                if dist[i] < min_dist:
                    min_dist = dist[i]
                    min_index = j
            labels[i] = min_index

        # 更新质心
        for j in range(n_clusters):
            cluster_points = data[labels == j]
            if len(cluster_points) > 0:
                new_center = np.mean(cluster_points, axis=0)
                new_centers[j] = new_center
            else:
                new_centers[j] = centers[j]

        # 检查收敛
        if np.allclose(new_centers, centers):
            break
        centers = new_centers.copy()

    return labels, centers


# 执行基于 EMD 的 K-means 聚类（CUDA 加速）
labels, centers = emd_kmeans_cuda(data, initial_centers, max_iter=max_iter)

# 将聚类标签添加到原始特征 DataFrame
features_df['Cluster'] = labels

# 将结果保存到新的 Excel 文件中
output_file = 'calcium_window_emd_cluster_single_sheet_cuda_optimized.xlsx'
features_df.to_excel(output_file, sheet_name=sheet_name, index=False)

print(f"聚类分析完成，结果已保存到 {output_file} 文件的 {sheet_name} 工作表中")
