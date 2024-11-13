# import pandas as pd
# import numpy as np
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import cv2  # OpenCV 用于高维 EMD 计算
# import math
#
# # 设置输入和输出文件路径
# file_path = r'C:\Users\PAN\PycharmProjects\pythonProject\python-RA\Day6\calcium_window_data_separate_sheets.xlsx'
# output_file = 'calcium_window_emd_cluster_results_all_sheets.xlsx'
#
# # 聚类参数
# n_clusters = 5
# max_iter = 100
#
#
# # EMD 计算函数 (使用 OpenCV 的 cv2.EMD)
# def emd_distance(a, b):
#     a = a.astype(np.float32).reshape(-1, 1)
#     b = b.astype(np.float32).reshape(-1, 1)
#     weights_a = np.ones((a.shape[0], 1), dtype=np.float32) / a.shape[0]
#     weights_b = np.ones((b.shape[0], 1), dtype=np.float32) / b.shape[0]
#     sig_a = np.hstack((weights_a, a))
#     sig_b = np.hstack((weights_b, b))
#     emd, _, _ = cv2.EMD(sig_a, sig_b, cv2.DIST_L2)
#     return emd
#
#
# # 并行计算 EMD 距离矩阵
# def parallel_emd_distances(data_point, centers):
#     distances = [emd_distance(data_point, center) for center in centers]
#     return distances
#
#
# # K-means 聚类函数（基于并行化 EMD）
# def emd_kmeans(data, centers, max_iter=100):
#     labels = np.zeros(len(data), dtype=np.int32)
#     new_centers = np.empty_like(centers)
#
#     for iteration in range(max_iter):
#         # 并行分配样本到最近的质心
#         with ThreadPoolExecutor() as executor:
#             futures = {executor.submit(parallel_emd_distances, data[i], centers): i for i in range(len(data))}
#             for future in as_completed(futures):
#                 i = futures[future]
#                 distances = future.result()
#                 min_index = np.argmin(distances)
#                 labels[i] = min_index
#
#         # 更新质心
#         for j in range(n_clusters):
#             cluster_points = data[labels == j]
#             if len(cluster_points) > 0:
#                 mean_center = np.zeros(cluster_points.shape[1])
#                 for k in range(cluster_points.shape[1]):
#                     mean_center[k] = np.sum(cluster_points[:, k]) / cluster_points.shape[0]
#                 new_centers[j] = mean_center
#             else:
#                 new_centers[j] = centers[j]
#
#         # 检查收敛
#         if np.allclose(new_centers, centers, rtol=1e-6):
#             break
#         centers = new_centers.copy()
#
#     return labels, centers
#
#
# # 读取并处理 Excel 文件中的所有工作表
# with pd.ExcelWriter(output_file) as writer:
#     xls = pd.ExcelFile(file_path)
#     for sheet_name in xls.sheet_names:
#         print(f"Processing sheet: {sheet_name}")
#         df = pd.read_excel(xls, sheet_name=sheet_name)
#
#         # 提取特征
#         features = []
#         for i, row in df.iterrows():
#             window_data = row[2:].values
#             peak_value = window_data.max()
#             mean_value = window_data.mean()
#             amplitude = peak_value - mean_value
#             decay_time = np.argmax(window_data <= peak_value / 2) if np.any(window_data <= peak_value / 2) else len(
#                 window_data)
#             rise_time = np.argmax(window_data >= mean_value)
#             latency = decay_time + rise_time
#             frequency = len(np.where(window_data > mean_value)[0]) / len(window_data)
#             features.append([amplitude, peak_value, decay_time, rise_time, latency, frequency])
#
#         data = np.array(features, dtype=np.float32)
#
#         # 随机初始化质心
#         np.random.seed(42)
#         initial_centers = data[np.random.choice(data.shape[0], n_clusters, replace=False)]
#
#         # 执行基于 EMD 的并行化 K-means 聚类
#         labels, centers = emd_kmeans(data, initial_centers, max_iter=max_iter)
#
#         # 将聚类标签和结果添加到 DataFrame
#         result_df = pd.DataFrame(data, columns=['Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency'])
#         result_df['Cluster'] = labels
#
#         # 保存每个工作表的聚类结果到 Excel 输出文件
#         result_df.to_excel(writer, sheet_name=sheet_name, index=False)
#         print(f"Completed processing for sheet: {sheet_name}")
#
# print(f"聚类分析完成，结果已保存到 {output_file} 中的各个工作表中")


import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2  # OpenCV 用于高维 EMD 计算
import math

# 设置输入和输出文件路径
file_path = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day6\calcium_window_data_separate_sheets.xlsx'
output_file = 'calcium_window_emd_cluster_results_weighted_sheets.xlsx'

# 聚类参数
n_clusters = 5
max_iter = 100

# 特征权重
weights = {
    'amplitude': 0.40,
    'peak': 0.10,
    'latency': 0.15,
    'frequency': 0.25,
    'decay_time': 0.05,
    'rise_time': 0.05
}

# EMD 计算函数 (使用 OpenCV 的 cv2.EMD)
def emd_distance(a, b):
    a = a.astype(np.float32).reshape(-1, 1)
    b = b.astype(np.float32).reshape(-1, 1)
    weights_a = np.ones((a.shape[0], 1), dtype=np.float32) / a.shape[0]
    weights_b = np.ones((b.shape[0], 1), dtype=np.float32) / b.shape[0]
    sig_a = np.hstack((weights_a, a))
    sig_b = np.hstack((weights_b, b))
    emd, _, _ = cv2.EMD(sig_a, sig_b, cv2.DIST_L2)
    return emd

# 并行计算 EMD 距离矩阵
def parallel_emd_distances(data_point, centers):
    distances = [emd_distance(data_point, center) for center in centers]
    return distances

# K-means 聚类函数（基于并行化 EMD）
def emd_kmeans(data, centers, max_iter=100):
    labels = np.zeros(len(data), dtype=np.int32)
    new_centers = np.empty_like(centers)

    for iteration in range(max_iter):
        # 并行分配样本到最近的质心
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(parallel_emd_distances, data[i], centers): i for i in range(len(data))}
            for future in as_completed(futures):
                i = futures[future]
                distances = future.result()
                min_index = np.argmin(distances)
                labels[i] = min_index

        # 更新质心
        for j in range(n_clusters):
            cluster_points = data[labels == j]
            if len(cluster_points) > 0:
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

# 读取并处理 Excel 文件中的所有工作表
with pd.ExcelWriter(output_file) as writer:
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        print(f"Processing sheet: {sheet_name}")
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # 提取加权特征
        features = []
        for i, row in df.iterrows():
            window_data = row[2:].values
            peak_value = window_data.max()
            mean_value = window_data.mean()
            amplitude = (peak_value - mean_value) * weights['amplitude']
            decay_time = (np.argmax(window_data <= peak_value / 2) if np.any(window_data <= peak_value / 2) else len(window_data)) * weights['decay_time']
            rise_time = np.argmax(window_data >= mean_value) * weights['rise_time']
            latency = (decay_time + rise_time) * weights['latency']
            frequency = (len(np.where(window_data > mean_value)[0]) / len(window_data)) * weights['frequency']

            features.append([amplitude, peak_value * weights['peak'], decay_time, rise_time, latency, frequency])

        data = np.array(features, dtype=np.float32)

        # 随机初始化质心
        np.random.seed(42)
        initial_centers = data[np.random.choice(data.shape[0], n_clusters, replace=False)]

        # 执行基于 EMD 的并行化 K-means 聚类
        labels, centers = emd_kmeans(data, initial_centers, max_iter=max_iter)

        # 将聚类标签和结果添加到 DataFrame
        result_df = pd.DataFrame(data, columns=['Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency'])
        result_df['Cluster'] = labels

        # 保存每个工作表的聚类结果到 Excel 输出文件
        result_df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Completed processing for sheet: {sheet_name}")

print(f"聚类分析完成，结果已保存到 {output_file} 中的各个工作表中")
