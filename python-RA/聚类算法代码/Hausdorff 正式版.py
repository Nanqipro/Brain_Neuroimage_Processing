import os
import pandas as pd
import numpy as np
import math
from numba import jit

# 设置输入和输出文件路径
file_path = r'C:\Users\PAN\PycharmProjects\Brain_Neuroimage_Processing\python-RA\Day6\calcium_window_data_separate_sheets.xlsx'
output_folder = r'C:\Users\PAN\PycharmProjects\Brain_Neuroimage_Processing\python-RA\标准化代码'
output_file = os.path.join(output_folder, 'calcium_window_result_weighted_output.xlsx')

# 如果输出文件夹不存在，则创建
os.makedirs(output_folder, exist_ok=True)

# 聚类参数
n_clusters = 5
max_iter = 100

# 定义 Hausdorff 距离计算函数，不再创建新的 NumPy 数组
@jit(nopython=True)
def hausdorff_distance(a, b):
    max_dist = 0
    for i in range(len(a)):
        min_dist = math.inf
        for j in range(len(b)):
            dist = 0
            for k in range(len(a[i])):
                dist += (a[i][k] - b[j][k]) ** 2
            min_dist = min(min_dist, math.sqrt(dist))
        max_dist = max(max_dist, min_dist)
    return max_dist

# 使用 JIT 加速的 K-means 聚类主循环
@jit(nopython=True)
def hausdorff_kmeans(data, centers, max_iter=100):
    labels = np.zeros(len(data), dtype=np.int32)
    new_centers = np.empty_like(centers)

    for iteration in range(max_iter):
        # 分配样本到最近的质心
        for i in range(len(data)):
            min_dist = math.inf
            min_index = 0
            for j in range(len(centers)):
                dist = hausdorff_distance(data[i:i + 1], centers[j:j + 1])
                if dist < min_dist:
                    min_dist = dist
                    min_index = j
            labels[i] = min_index

        # 更新质心
        for j in range(n_clusters):
            cluster_points = data[labels == j]
            if len(cluster_points) > 0:
                # 手动计算均值
                mean_center = np.zeros(cluster_points.shape[1])
                for k in range(cluster_points.shape[1]):
                    mean_center[k] = np.sum(cluster_points[:, k]) / cluster_points.shape[0]
                new_centers[j] = mean_center
            else:
                new_centers[j] = centers[j]

        # 检查收敛
        if np.allclose(new_centers, centers, rtol=1e-6):
            break
        centers = new_centers.copy()

    return labels, centers

# 特征权重
weights = {
    'amplitude': 0.40,
    'peak': 0.10,
    'latency': 0.15,
    'frequency': 0.25,
    'decay_time': 0.05,
    'rise_time': 0.05
}

# 读取并处理 Excel 文件中的所有工作表
def process_sheets(input_path, output_path):
    try:
        # 检查文件是否存在
        xls = pd.ExcelFile(input_path)
    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
        return

    with pd.ExcelWriter(output_path) as writer:
        for sheet_name in xls.sheet_names:
            print(f"Processing sheet: {sheet_name}")
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # 确保当前工作表有数据
            if df.empty:
                print(f"Warning: Sheet '{sheet_name}' is empty and will be skipped.")
                continue

            # 提取特征
            features = []
            for i, row in df.iterrows():
                window_data = row[2:].values
                peak_value = window_data.max()
                mean_value = window_data.mean()
                amplitude = peak_value - mean_value

                # 特征计算
                decay_time = np.argmax(window_data <= peak_value / 2) if np.any(window_data <= peak_value / 2) else len(window_data)
                rise_time = np.argmax(window_data >= mean_value)
                latency = decay_time + rise_time
                frequency = len(np.where(window_data > mean_value)[0]) / len(window_data)

                # 应用权重
                weighted_features = [
                    amplitude * weights['amplitude'],
                    peak_value * weights['peak'],
                    decay_time * weights['decay_time'],
                    rise_time * weights['rise_time'],
                    latency * weights['latency'],
                    frequency * weights['frequency']
                ]

                features.append(weighted_features)

            # 准备数据进行聚类
            data = np.array(features)

            # 随机初始化质心
            np.random.seed(42)
            initial_centers = data[np.random.choice(data.shape[0], n_clusters, replace=False)]

            # 执行 Hausdorff K-means 聚类
            labels, centers = hausdorff_kmeans(data, initial_centers, max_iter=max_iter)

            # 将聚类标签和结果添加到 DataFrame
            result_df = pd.DataFrame(data, columns=['Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency'])
            result_df['Cluster'] = labels

            # 保存每个工作表的聚类结果到 Excel 输出文件
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Completed processing for sheet: {sheet_name}")

    print(f"聚类分析完成，结果已保存到 {output_path} 中的各个工作表中")

# 执行处理函数
process_sheets(file_path, output_file)
