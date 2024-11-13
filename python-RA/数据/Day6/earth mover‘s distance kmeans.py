import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
import plotly.express as px

# 读取单个工作表的数据
file_path = 'calcium_window_data_separate_sheets.xlsx'
sheet_name = 'Window_30_Step_10'  # 指定要读取的工作表名称
df = pd.read_excel(file_path, sheet_name=sheet_name)

# 设置聚类参数
n_clusters = 5  # 假设分成 5 个类
max_iter = 100  # 最大迭代次数

# 初始化存储特征的列表
features = []

# 计算每个窗口的峰值、标准差和平均值
for i, row in df.iterrows():
    window_data = row[2:].values  # 跳过 'Neuron' 和 'Start Time' 列
    peak_value = window_data.max()  # 峰值
    std_dev_value = window_data.std()  # 标准差
    mean_value = window_data.mean()  # 平均值

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


# 基于 EMD 的 K-means 聚类函数
def emd_kmeans(data, centers, max_iter=100):
    labels = np.zeros(len(data), dtype=int)

    for iteration in range(max_iter):
        # 分配样本到最接近的质心（基于 EMD）
        for i, point in enumerate(data):
            distances = [wasserstein_distance(point, center) for center in centers]
            labels[i] = np.argmin(distances)

        # 更新质心
        new_centers = []
        for j in range(n_clusters):
            cluster_points = data[labels == j]
            if len(cluster_points) > 0:
                new_center = np.mean(cluster_points, axis=0)
                new_centers.append(new_center)
            else:
                new_centers.append(centers[j])

        # 检查收敛
        new_centers = np.array(new_centers)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return labels, centers


# 执行基于 EMD 的 K-means 聚类
labels, centers = emd_kmeans(data, initial_centers, max_iter=max_iter)

# 将聚类标签添加到原始特征 DataFrame
features_df['Cluster'] = labels

# 将结果保存到新的 Excel 文件中
output_file = 'calcium_window_emd_cluster_single_sheet2.xlsx'
features_df.to_excel(output_file, sheet_name=sheet_name, index=False)

print(f"聚类分析完成，结果已保存到 {output_file} 文件的 {sheet_name} 工作表中")
