# 导入必要的库
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 添加以下代码来设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号显示问题

# 读取Excel文件
df = pd.read_excel(r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day6\calcium_data.xlsx', header=0)

# 打印DataFrame的信息，检查数据是否正确读取
print("DataFrame 信息：")
print(df.info())
print(df.head())

# 提取时间点（第一列），跳过第一行（标签行）
time_points = df.iloc[1:, 0].values.astype(float)

# 提取神经元名称（列标签），排除第一列的时间点
neuron_names = df.columns[1:]

# 初始化数据列表
data = []

# 遍历每个神经元，提取并处理数据
for neuron in neuron_names:
    # 提取每个神经元的钙波序列，跳过第一行（标签行）
    signal = df[neuron].iloc[1:]

    # 数据清洗：将非数值转换为NaN，并填充或删除
    signal = pd.to_numeric(signal, errors='coerce')  # 将无法转换为数值的值设为NaN
    signal = signal.fillna(method='ffill')  # 使用前一个有效值填充NaN
    signal = signal.fillna(method='bfill')  # 如果开头有NaN，用后面的有效值填充
    signal = signal.astype(float)  # 转换为浮点型

    # 将signal转换为numpy数组
    signal = signal.values

    # 检查signal的形状
    if signal.ndim != 1:
        print(f'警告: 神经元 {neuron} 的数据不是一维数组，当前形状为 {signal.shape}，将其展平。')
        signal = signal.flatten()

    data.append(signal)

# 确认数据格式
num_segments = len(data)  # 神经元数量
print(f'共有 {num_segments} 个神经元的钙波数据。')

# 检查所有序列的长度是否一致
sequence_lengths = [len(seq) for seq in data]
print("序列长度列表:", sequence_lengths)
if len(set(sequence_lengths)) != 1:
    print("警告：序列长度不一致，可能需要进行截断或填充。")
    # 可选择截断或填充序列，使其长度一致
    min_length = min(sequence_lengths)
    data = [seq[:min_length] for seq in data]
    time_points = time_points[:min_length]


# 使用自定义的标量距离函数
def scalar_distance(x, y):
    return abs(x - y)


# 计算DTW距离矩阵
distance_matrix = np.zeros((num_segments, num_segments))

print('正在计算DTW距离矩阵，请稍候...')

# 使用tqdm添加进度条
for i in tqdm(range(num_segments)):
    for j in range(i + 1, num_segments):
        distance, _ = fastdtw(data[i], data[j], dist=scalar_distance)
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance  # 对称矩阵

print('DTW距离矩阵计算完成。')

# 将距离矩阵转换为压缩形式（用于层次聚类）
condensed_distance_matrix = squareform(distance_matrix)

# 进行层次聚类
linked = linkage(condensed_distance_matrix, method='average')

# 绘制树状图
plt.figure(figsize=(12, 8))
dendrogram(linked,
           labels=neuron_names[:num_segments],  # 可能有部分神经元被跳过，需要截取
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('层次聚类树状图')
plt.xlabel('神经元')
plt.ylabel('距离')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 确定聚类数量并获取聚类标签
# 您可以根据树状图选择合适的聚类数量
num_clusters = 5  # 示例中选择3个聚类
cluster_labels = fcluster(linked, num_clusters, criterion='maxclust')

# 输出每个神经元的聚类标签
print('聚类结果：')
for neuron, label in zip(neuron_names[:num_segments], cluster_labels):
    print(f'神经元 {neuron}: 聚类 {label}')

# 将数据和聚类标签组合
data_with_labels = list(zip(data, cluster_labels, neuron_names[:num_segments]))

# 绘制每个聚类的平均波形
plt.figure(figsize=(12, 8))
for cluster_num in range(1, num_clusters + 1):
    cluster_data = [d for d, label, name in data_with_labels if label == cluster_num]
    if len(cluster_data) > 0:
        mean_waveform = np.mean(cluster_data, axis=0)
        plt.plot(time_points, mean_waveform, label=f'聚类 {cluster_num}')

plt.title('每个聚类的平均钙波波形')
plt.xlabel('时间')
plt.ylabel('钙离子浓度（相对值）')
plt.legend()
plt.tight_layout()
plt.show()

# 使用不同的颜色绘制不同聚类的钙波片段
plt.figure(figsize=(12, 8))
colors = sns.color_palette('hls', num_clusters)

for signal, label, name in data_with_labels:
    plt.plot(time_points, signal, color=colors[label - 1], alpha=0.5)

plt.title('基于DTW距离的钙波片段聚类')
plt.xlabel('时间')
plt.ylabel('钙离子浓度（相对值）')
plt.tight_layout()
plt.show()

# # 导入必要的库
# import numpy as np
# import pandas as pd
# from scipy.spatial.distance import squareform
# from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
# from sklearn.cluster import AgglomerativeClustering
# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm  # 用于显示进度条
#
# # 读取Excel文件
# # 请将'calcium_data.xlsx'替换为您的实际文件路径
# # 设置header=0，表示第一行是列名
# df = pd.read_excel(r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day6\calcium_data.xlsx', header=0)
#
# # 打印DataFrame的信息，检查数据是否正确读取
# print("DataFrame 信息：")
# print(df.info())
# print(df.head())
#
# # 提取时间点（第一列），跳过第一行（标签行）
# time_points = df.iloc[1:, 0].values.astype(float)
#
# # 提取神经元名称（列标签），排除第一列的时间点
# neuron_names = df.columns[1:]
#
# # 初始化数据列表
# data = []
#
# # 遍历每个神经元，提取并处理数据
# for neuron in neuron_names:
#     # 提取每个神经元的钙波序列，跳过第一行（标签行）
#     signal = df[neuron].iloc[1:]
#
#     # 数据清洗：将非数值转换为NaN，并填充或删除
#     signal = pd.to_numeric(signal, errors='coerce')  # 将无法转换为数值的值设为NaN
#     signal = signal.fillna(method='ffill')  # 使用前一个有效值填充NaN
#     signal = signal.fillna(method='bfill')  # 如果开头有NaN，用后面的有效值填充
#     signal = signal.astype(float)  # 转换为浮点型
#
#     # 将signal转换为numpy数组
#     signal = signal.values
#
#     # 检查signal的形状
#     print(f'神经元 {neuron} 的数据形状: {signal.shape}')
#
#     # 如果signal是二维数组，进行展平处理
#     if signal.ndim > 1:
#         print(f'警告: 神经元 {neuron} 的数据是 {signal.ndim} 维的，将其展平为一维数组。')
#         signal = signal.flatten()
#
#     # 再次检查signal的形状
#     if signal.ndim != 1:
#         print(f'错误: 神经元 {neuron} 的数据无法转换为一维数组，当前形状为 {signal.shape}')
#         continue  # 跳过该神经元的数据
#
#     data.append(signal)
#
# # 确认数据格式
# num_segments = len(data)  # 神经元数量
# print(f'共有 {num_segments} 个神经元的钙波数据。')
#
# # 检查所有序列的长度是否一致
# sequence_lengths = [len(seq) for seq in data]
# print("序列长度列表:", sequence_lengths)
# if len(set(sequence_lengths)) != 1:
#     print("警告：序列长度不一致，可能需要进行截断或填充。")
#     # 可选择截断或填充序列，使其长度一致
#     min_length = min(sequence_lengths)
#     data = [seq[:min_length] for seq in data]
#     time_points = time_points[:min_length]
#
# # 计算DTW距离矩阵
# distance_matrix = np.zeros((num_segments, num_segments))
#
# print('正在计算DTW距离矩阵，请稍候...')
#
# # 使用tqdm添加进度条
# for i in tqdm(range(num_segments)):
#     for j in range(i + 1, num_segments):
#         # 确保输入的是一维数组
#         if data[i].ndim != 1 or data[j].ndim != 1:
#             print(f'错误: 数据序列不是一维数组，无法计算DTW距离。序列{i}形状：{data[i].shape}，序列{j}形状：{data[j].shape}')
#             continue
#         distance, _ = fastdtw(data[i], data[j], dist=euclidean)
#         distance_matrix[i, j] = distance
#         distance_matrix[j, i] = distance  # 对称矩阵
#
# print('DTW距离矩阵计算完成。')
#
# # 将距离矩阵转换为压缩形式（用于层次聚类）
# condensed_distance_matrix = squareform(distance_matrix)
#
# # 进行层次聚类
# linked = linkage(condensed_distance_matrix, method='average')
#
# # 绘制树状图
# plt.figure(figsize=(12, 8))
# dendrogram(linked,
#            labels=neuron_names[:num_segments],  # 可能有部分神经元被跳过，需要截取
#            distance_sort='descending',
#            show_leaf_counts=True)
# plt.title('层次聚类树状图')
# plt.xlabel('神经元')
# plt.ylabel('距离')
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()
#
# # 确定聚类数量并获取聚类标签
# # 您可以根据树状图选择合适的聚类数量
# num_clusters = 3  # 示例中选择3个聚类
# cluster_labels = fcluster(linked, num_clusters, criterion='maxclust')
#
# # 输出每个神经元的聚类标签
# print('聚类结果：')
# for neuron, label in zip(neuron_names[:num_segments], cluster_labels):
#     print(f'神经元 {neuron}: 聚类 {label}')
#
# # 将数据和聚类标签组合
# data_with_labels = list(zip(data, cluster_labels, neuron_names[:num_segments]))
#
# # 绘制每个聚类的平均波形
# plt.figure(figsize=(12, 8))
# for cluster_num in range(1, num_clusters + 1):
#     cluster_data = [d for d, label, name in data_with_labels if label == cluster_num]
#     if len(cluster_data) > 0:
#         mean_waveform = np.mean(cluster_data, axis=0)
#         plt.plot(time_points, mean_waveform, label=f'聚类 {cluster_num}')
#
# plt.title('每个聚类的平均钙波波形')
# plt.xlabel('时间')
# plt.ylabel('钙离子浓度（相对值）')
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# # 使用不同的颜色绘制不同聚类的钙波片段
# plt.figure(figsize=(12, 8))
# colors = sns.color_palette('hls', num_clusters)
#
# for signal, label, name in data_with_labels:
#     plt.plot(time_points, signal, color=colors[label - 1], alpha=0.5)
#
# plt.title('基于DTW距离的钙波片段聚类')
# plt.xlabel('时间')
# plt.ylabel('钙离子浓度（相对值）')
# plt.tight_layout()
# plt.show()
