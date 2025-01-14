import pandas as pd
import numpy as np  # 导入 NumPy 以处理无穷大值
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from itertools import combinations

# =============== 1. 参数区（可灵活调整） ===============
NODES_FILE = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\DeepLearning\Day6 训练结果\Nodes.csv'    # <-- 修改为 Nodes.csv 的路径
EDGES_FILE = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\DeepLearning\Day6 训练结果\Edges.csv'    # <-- 修改为 Edges.csv 的路径
LABELS_FILE = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\DeepLearning\Day6 训练结果\Labels.csv'  # <-- 修改为 Labels.csv 的路径
BEHAVIOR_DATA_FILE = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day6\day 6  cell 行为学标记.xlsx'  # <-- 可调整：中文行为数据Excel文件

# 超参数
BATCH_SIZE = 32       # <-- 可调整：批大小
HIDDEN_DIM = 16       # <-- 可调整：GCN隐藏层大小
LEARNING_RATE = 0.01  # <-- 可调整：学习率
EPOCHS = 100          # <-- 可调整：训练轮数

# =============== 2. 数据导入和预处理 ===============
print("==== 加载 CSV 数据 ...")

try:
    nodes_df = pd.read_csv(NODES_FILE)
    edges_df = pd.read_csv(EDGES_FILE)
    labels_df = pd.read_csv(LABELS_FILE)
    behavior_data_df = pd.read_excel(BEHAVIOR_DATA_FILE, sheet_name='CHB')  # 假设行为标签在 'CHB' sheet
    print("Nodes DataFrame head:")
    print(nodes_df.head())
    print("Edges DataFrame head:")
    print(edges_df.head())
    print("Labels DataFrame head:")
    print(labels_df.head())
    print("Behavior Data DataFrame head (含中文行为):")
    print(behavior_data_df.head())
except Exception as e:
    print("加载数据时出错:", e)
    exit(1)

print("==== 检查列名 ...")
print("labels_df 的列名:", labels_df.columns.tolist())
print("behavior_data_df 的列名:", behavior_data_df.columns.tolist())

# 确认 'time' 列存在于 labels_df 中，'ID' 列存在于 behavior_data_df 中
if 'time' not in labels_df.columns:
    print("Error: 'time' 列在 labels_df 中不存在。请检查列名或数据源。")
    exit(1)

if 'ID' not in behavior_data_df.columns:
    print("Error: 'ID' 列在 behavior_data_df 中不存在。请检查列名或数据源。")
    exit(1)

print("==== 合并中文行为标签 ...")

# 统一数据类型：将 'time' 和 'ID' 列转换为整数类型
try:
    # 检查 'time' 和 'ID' 列是否存在缺失值
    print("Missing values in labels_df['time']:", labels_df['time'].isnull().sum())
    print("Missing values in behavior_data_df['ID']:", behavior_data_df['ID'].isnull().sum())

    # 检查 'time' 和 'ID' 列是否存在无穷大值
    print("Number of inf in labels_df['time']:", np.isinf(labels_df['time']).sum())
    print("Number of -inf in labels_df['time']:", np.isneginf(labels_df['time']).sum())
    print("Number of inf in behavior_data_df['ID']:", np.isinf(behavior_data_df['ID']).sum())
    print("Number of -inf in behavior_data_df['ID']:", np.isneginf(behavior_data_df['ID']).sum())

    # 替换 'inf' 和 '-inf' 为 NaN
    labels_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    behavior_data_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 删除 'time' 或 'ID' 列中存在缺失值的行
    labels_df = labels_df.dropna(subset=['time'])
    behavior_data_df = behavior_data_df.dropna(subset=['ID'])

    # 再次检查是否还有缺失值
    print("After replacing inf, missing values in labels_df['time']:", labels_df['time'].isnull().sum())
    print("After replacing inf, missing values in behavior_data_df['ID']:", behavior_data_df['ID'].isnull().sum())

    # 将 'time' 和 'ID' 列转换为整数类型
    labels_df['time'] = pd.to_numeric(labels_df['time'], errors='coerce').astype(int)
    behavior_data_df['ID'] = pd.to_numeric(behavior_data_df['ID'], errors='coerce').astype(int)
except Exception as e:
    print("转换 'time' 或 'ID' 列为整数时出错:", e)
    print("请检查 'labels_df' 和 'behavior_data_df' 中的 'time' 和 'ID' 列是否存在非数值值。")
    exit(1)

# 确保没有缺失值后进行合并
try:
    labels_merged = pd.merge(
        labels_df,
        behavior_data_df[['ID', 'FrameLost']],
        how='left',
        left_on='time',
        right_on='ID'  # <-- 使用 labels_df 的 'time' 与 behavior_data_df 的 'ID' 进行合并
    )
    print("合并后的 DataFrame head:")
    print(labels_merged.head())
except Exception as e:
    print("合并行为标签时出错:", e)
    exit(1)

# 使用 LabelEncoder 将中文行为（即 'FrameLost' 列）转为数字
le = LabelEncoder()
# 注意：如果 'FrameLost' 里有空值，则需要先填充或过滤
labels_merged['FrameLost'] = labels_merged['FrameLost'].fillna('无行为')
labels_merged['behavior_num'] = le.fit_transform(labels_merged['FrameLost'].astype(str))

label_mapping = {label: idx for idx, label in enumerate(le.classes_)}
print("标签映射（中文 -> 数字）：", label_mapping)

# 更新 labels_df
labels_df = labels_merged

# =============== 3. 全局神经元映射 ===============
print("==== 创建全局神经元映射 ...")
all_neurons = sorted(nodes_df['Neuron'].unique())
print(f"总共有 {len(all_neurons)} 个唯一的神经元。")
neuron_mapping = {name: i for i, name in enumerate(all_neurons)}
print("神经元映射示例:", list(neuron_mapping.items())[:5])

# =============== 4. 计算激活阈值 ===============
print("==== 计算每个神经元的激活阈值（平均浓度） ...")

# 识别神经元列（排除 'time'）
neuron_columns = [col for col in nodes_df.columns if col != 'time']
print("神经元列名:")
print(neuron_columns)

# 确保神经元列的数据类型为数值型
print("将神经元列转换为数值型...")
for neuron in neuron_columns:
    try:
        nodes_df[neuron] = pd.to_numeric(nodes_df[neuron], errors='coerce')
    except Exception as e:
        print(f"转换神经元 {neuron} 时出错:", e)
        # 根据需要决定是否退出或继续
        continue

# 计算每个神经元的平均浓度
activation_thresholds = nodes_df[neuron_columns].mean().to_dict()
print("激活阈值（每个神经元的平均浓度）:")
print(activation_thresholds)

# =============== 5. 确定每个时间戳的激活神经元 ===============
print("==== 确定每个时间戳的激活神经元 ...")

# 创建一个字典，键是时间戳 (time)，值是激活的神经元列表
activated_neurons_per_time = {}

for idx, row in nodes_df.iterrows():
    time = row['time']  # 使用 'time' 作为时间戳
    activated_neurons = []
    for neuron, threshold in activation_thresholds.items():
        concentration = row[neuron]
        # 调试信息
        if isinstance(concentration, pd.Series):
            print(f"警告: Time {time}, Neuron {neuron} 的浓度是一个Series: {concentration}")
            continue  # 跳过该神经元
        if pd.notna(concentration) and concentration > threshold:
            activated_neurons.append(neuron)
    activated_neurons_per_time[time] = activated_neurons

# 查看部分时间戳的激活神经元
print("部分时间戳的激活神经元:")
for time, neurons in list(activated_neurons_per_time.items())[:5]:
    print(f"Time {time}: {neurons}")

# =============== 6. 构建Nodes和Edges数据 ===============
print("==== 构建Nodes和Edges数据 ...")
nodes_data = []
edges_set = set()  # 使用集合来存储唯一的边
edges_data = []
labels_data = []

for time, neurons in activated_neurons_per_time.items():
    # 处理Nodes
    for neuron in neurons:
        nodes_data.append({
            'Neuron': neuron,
            'state': 'ON',
            'time': time  # 使用 'time' 作为时间戳
            # 如果需要其他列，可以在这里添加，如 'color', 'group_id' 等
        })

    # 处理Edges，避免重复
    for source, target in combinations(sorted(neurons), 2):
        edge = (source, target)
        if edge not in edges_set:
            edges_set.add(edge)
            edges_data.append({
                'source': source,
                'target': target,
                'time': time  # 使用 'time' 作为时间戳
                # 如果需要其他列，可以在这里添加，如 'color' 等
            })

    # 处理Labels
    # 这里假设行为标签在 labels_df 中已包含 'behavior_num' 列
    label_num = labels_df.loc[labels_df['time'] == time, 'behavior_num'].values
    if label_num.size > 0:
        label_t = label_num[0]
    else:
        label_t = -1  # 未找到对应标签

    labels_data.append({
        'time': time,
        'behavior_num': label_t
    })

# 转换为DataFrame
nodes_df_processed = pd.DataFrame(nodes_data)
edges_df_processed = pd.DataFrame(edges_data)
labels_df_processed = pd.DataFrame(labels_data).drop_duplicates()
print("Nodes DataFrame shape:", nodes_df_processed.shape)
print("Edges DataFrame shape:", edges_df_processed.shape)
print("Labels DataFrame shape:", labels_df_processed.shape)

print("Nodes DataFrame head:")
print(nodes_df_processed.head())
print("Edges DataFrame head:")
print(edges_df_processed.head())
print("Labels DataFrame head:")
print(labels_df_processed.head())

# 继续后续的数据处理和模型训练步骤...
