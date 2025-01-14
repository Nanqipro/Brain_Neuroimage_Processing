import pandas as pd
import numpy as np
import torch
from itertools import combinations
import os

# =============== 1. 参数区（可灵活调整） ===============
RAW_DATA_FILE = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day6\calcium_data.xlsx'  # <-- 可调整：原始钙离子浓度数据的Excel文件
SHEET_NAME_RAW = 'Sheet1'  # <-- 可调整：原始数据所在的sheet名
OUTPUT_DIR = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\DeepLearning\Day6 训练结果'  # <-- 可调整：输出的CSV文件所在的目录

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

NODES_CSV = os.path.join(OUTPUT_DIR, 'Nodes.csv')
EDGES_CSV = os.path.join(OUTPUT_DIR, 'Edges.csv')
LABELS_CSV = os.path.join(OUTPUT_DIR, 'Labels.csv')

# =============== 2. 读取原始数据 ===============
print("==== 读取原始钙离子浓度数据 ...")
try:
    raw_df = pd.read_excel(RAW_DATA_FILE, sheet_name=SHEET_NAME_RAW)
    print("原始数据预览:")
    print(raw_df.head())
    print("原始数据的列名:")
    print(raw_df.columns.tolist())
except Exception as e:
    print("读取Excel文件时出错:", e)
    exit(1)

# =============== 3. 转置数据和处理 'Time' ===============
print("==== 转置数据 ...")
try:
    raw_df = raw_df.transpose()
    print("转置后的数据预览:")
    print(raw_df.head())
except Exception as e:
    print("转置数据时出错:", e)
    exit(1)

# 设置第一行作为列名
try:
    raw_df.columns = raw_df.iloc[0]  # 第一行作为列名
    raw_df = raw_df[1:]  # 删除第一行
    print("设置列名后的数据预览:")
    print(raw_df.head())
    print("列名:")
    print(raw_df.columns.tolist())
except Exception as e:
    print("设置列名时出错:", e)
    exit(1)

# =============== 3.1. 去重列名函数 ===============
def deduplicate_columns(columns):
    """
    确保所有列名唯一，重复的列名会添加后缀 _1, _2, 等。
    """
    counts = {}
    new_columns = []
    for col in columns:
        if col in counts:
            counts[col] += 1
            new_col = f"{col}_{counts[col]}"
        else:
            counts[col] = 0
            new_col = col
        new_columns.append(new_col)
    return new_columns

# 确保列名唯一，防止重复导致的问题
print("检查并处理重复的列名...")
raw_df.columns = deduplicate_columns(raw_df.columns)
print("处理后的列名:")
print(raw_df.columns.tolist())

# 将所有列名转换为字符串，避免使用浮点数作为列名可能引发的问题
raw_df.columns = raw_df.columns.map(str)
print("转换为字符串后的列名:")
print(raw_df.columns.tolist())

# 假设转置后，行索引是时间戳
raw_df.index.name = 'Time'

# 重置索引以将 'Time' 转换为列
raw_df.reset_index(inplace=True)

# 检查重设索引后的数据
print("重设索引后的数据预览:")
print(raw_df.head())
print("列名:")
print(raw_df.columns.tolist())

# 确保 'Time' 列存在
if 'Time' not in raw_df.columns:
    raise KeyError("'Time' column is not found after transposing and resetting the index.")

# =============== 4. 计算激活阈值 ===============
print("==== 计算每个神经元的激活阈值（平均浓度） ...")

# 识别神经元列（排除 'Time'）
neuron_columns = [col for col in raw_df.columns if col != 'Time']

print("神经元列名:")
print(neuron_columns)

# 检查是否有重复的神经元列名（已在deduplicate_columns中处理）
# 此处可以省略重复检查步骤

# 确保神经元列的数据类型为数值型
print("将神经元列转换为数值型...")
for neuron in neuron_columns:
    try:
        raw_df[neuron] = pd.to_numeric(raw_df[neuron], errors='coerce')
    except Exception as e:
        print(f"转换神经元 {neuron} 时出错:", e)
        # 根据需要决定是否退出或继续
        continue

# 计算每个神经元的平均浓度
activation_thresholds = raw_df[neuron_columns].mean().to_dict()

print("激活阈值（每个神经元的平均浓度）:")
print(activation_thresholds)

# =============== 5. 确定每个时间戳的激活神经元 ===============
print("==== 确定每个时间戳的激活神经元 ...")

# 创建一个字典，键是时间戳，值是激活的神经元列表
activated_neurons_per_time = {}

for idx, row in raw_df.iterrows():
    time = row['Time']
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
            'time': time
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
                'time': time
                # 如果需要其他列，可以在这里添加，如 'color' 等
            })

    # 处理Labels
    # 这里假设行为标签在另一个文件中处理，Labels sheet只需包含 'time' 列
    labels_data.append({
        'time': time
        # 'behavior' 列将在训练脚本中通过合并行为标签文件进行填充
    })

# 转换为DataFrame
nodes_df = pd.DataFrame(nodes_data)
edges_df = pd.DataFrame(edges_data)
labels_df = pd.DataFrame(labels_data).drop_duplicates()

print("Nodes DataFrame shape:", nodes_df.shape)
print("Edges DataFrame shape:", edges_df.shape)
print("Labels DataFrame shape:", labels_df.shape)

print("Nodes DataFrame head:")
print(nodes_df.head())
print("Edges DataFrame head:")
print(edges_df.head())
print("Labels DataFrame head:")
print(labels_df.head())

# =============== 7. 保存到CSV文件 ===============
print(f"==== 保存Nodes、Edges和Labels到 {OUTPUT_DIR} 目录下的CSV文件 ...")
try:
    nodes_df.to_csv(NODES_CSV, index=False)
    edges_df.to_csv(EDGES_CSV, index=False)
    labels_df.to_csv(LABELS_CSV, index=False)
    print("==== CSV文件保存完成 ====")
    print(f"Nodes CSV: {NODES_CSV}")
    print(f"Edges CSV: {EDGES_CSV}")
    print(f"Labels CSV: {LABELS_CSV}")
except Exception as e:
    print("保存为CSV文件时出错:", e)
