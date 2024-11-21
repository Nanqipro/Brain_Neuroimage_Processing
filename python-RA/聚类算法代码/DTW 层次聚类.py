# # 导入必要的库
# import numpy as np
# import pandas as pd
# from scipy.spatial.distance import squareform
# from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
# from fastdtw import fastdtw
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
#
# # 添加以下代码来设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
# plt.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号显示问题
#
# # 读取Excel文件
# df = pd.read_excel(r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day3\Day3.xlsx', header=0)
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
#     if signal.ndim != 1:
#         print(f'警告: 神经元 {neuron} 的数据不是一维数组，当前形状为 {signal.shape}，将其展平。')
#         signal = signal.flatten()
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
#
# # 使用自定义的标量距离函数
# def scalar_distance(x, y):
#     return abs(x - y)
#
#
# # 计算DTW距离矩阵
# distance_matrix = np.zeros((num_segments, num_segments))
#
# print('正在计算DTW距离矩阵，请稍候...')
#
# # 使用tqdm添加进度条
# for i in tqdm(range(num_segments)):
#     for j in range(i + 1, num_segments):
#         distance, _ = fastdtw(data[i], data[j], dist=scalar_distance)
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
# num_clusters = 6  # 示例中选择3个聚类
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
#

# 导入必要的库
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号显示问题

# 定义数据标签（可编辑）
data_label = 'Day 9'  # 这里可以修改为您需要的日期标签

# 读取Excel文件
df = pd.read_excel(r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day9\calcium_data9.xlsx', header=0)

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

# 绘制层次聚类树状图
plt.figure(figsize=(12, 8))
dendrogram(linked,
           labels=neuron_names[:num_segments],  # 可能有部分神经元被跳过，需要截取
           distance_sort='descending',
           show_leaf_counts=True)
plt.title(f'{data_label} 层次聚类树状图')
plt.xlabel('神经元')
plt.ylabel('距离')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 确定聚类数量并获取聚类标签
# 您可以根据树状图选择合适的聚类数量
num_clusters = 5  # 选择n个聚类
cluster_labels = fcluster(linked, num_clusters, criterion='maxclust')

# 输出每个神经元的聚类标签
print('聚类结果：')
for neuron, label in zip(neuron_names[:num_segments], cluster_labels):
    print(f'神经元 {neuron}: 聚类 {label}')

# 将数据、聚类标签和神经元名称组合
data_with_labels = list(zip(data, cluster_labels, neuron_names[:num_segments]))

# 根据聚类标签对数据进行排序
data_with_labels.sort(key=lambda x: x[1])  # 按照聚类标签排序

# 绘制每个神经元的钙波曲线（排序后，y 轴上等间隔排列）
plt.figure(figsize=(12, 8))
colors = sns.color_palette('hls', num_clusters)

y_offset = 2  # 定义每个曲线的垂直偏移量

for idx, (signal, label, name) in enumerate(data_with_labels):
    offset = idx * y_offset  # 计算当前曲线的垂直偏移
    plt.plot(time_points, signal + offset, color=colors[label - 1], alpha=0.8)
    # 在曲线左侧标注神经元名称
    plt.text(time_points[0], signal[0] + offset, f'{name}', fontsize=8, verticalalignment='bottom')
    # 在曲线右侧标注聚类编号（可选）
    plt.text(time_points[-1], signal[-1] + offset, f'聚类 {label}', fontsize=8, verticalalignment='bottom', horizontalalignment='right')

plt.title(f'{data_label} 基于 DTW 距离的钙波片段聚类')
plt.xlabel('时间')
plt.ylabel('钙离子浓度（相对值）+ 垂直偏移')
plt.yticks([])  # 隐藏 y 轴刻度
plt.tight_layout()
plt.show()

# 绘制每个聚类的平均钙波（在 y 轴上隔开距离）
plt.figure(figsize=(12, 8))

for cluster_num in range(1, num_clusters + 1):
    cluster_data = [d for d, label, name in data_with_labels if label == cluster_num]
    if len(cluster_data) > 0:
        mean_waveform = np.mean(cluster_data, axis=0)
        offset = (cluster_num - 1) * y_offset  # 为每个聚类添加偏移
        plt.plot(time_points, mean_waveform + offset, label=f'聚类 {cluster_num}')
        # 在曲线左侧标注聚类编号
        plt.text(time_points[0], mean_waveform[0] + offset, f'聚类 {cluster_num}', fontsize=10, verticalalignment='bottom')

plt.title(f'{data_label} 每个聚类的平均钙波波形')
plt.xlabel('时间')
plt.ylabel('钙离子浓度（相对值）+ 垂直偏移')
plt.yticks([])  # 隐藏 y 轴刻度
plt.legend()
plt.tight_layout()
plt.show()
