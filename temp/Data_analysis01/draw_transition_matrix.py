import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.sankey import Sankey

# Step 1: Load the trace and clustering result data
trace_file_path = './data/2979 CSDS Day3.xlsx'
cluster_file_path = './data/kmeans_clustering_results_2979_CSDS_Day3.xlsx'

trace_df = pd.read_excel(trace_file_path)
clustered_df = pd.read_excel(cluster_file_path)

# Step 2: 标注聚类结果中的状态
# 创建一个神经元状态时间序列矩阵，行是时间戳，列是神经元
# 我们假设trace_df中的每个神经元列都与clustered_df中的“Neuron”列一一对应

# 获取每个神经元的聚类状态
neuron_clusters = {neuron: cluster for neuron, cluster in zip(clustered_df['Neuron'], clustered_df['Cluster'])}

# 使用状态标注，创建一个新的DataFrame来记录每个时间点每个神经元的状态
state_matrix = trace_df.copy()
for neuron in trace_df.columns[1:]:  # 跳过时间列
    state_matrix[neuron] = neuron_clusters[neuron]

# Step 3: 构建状态转移矩阵
def build_transition_matrix(state_matrix):
    """
    构建状态转移矩阵，记录每个神经元状态随时间的变化。

    Args:
    - state_matrix (DataFrame): 每个时间点每个神经元的状态矩阵。

    Returns:
    - transition_matrix (ndarray): 转移矩阵，表示状态之间的转换频率。
    """
    unique_states = sorted(state_matrix.iloc[:, 1:].stack().unique())  # 获取唯一状态列表
    state_index = {state: i for i, state in enumerate(unique_states)}  # 为每个状态分配索引

    num_states = len(unique_states)
    transition_matrix = np.zeros((num_states, num_states))

    # 逐个神经元计算状态转换
    for neuron in state_matrix.columns[1:]:
        neuron_states = state_matrix[neuron].values
        for i in range(1, len(neuron_states)):
            prev_state = neuron_states[i - 1]
            curr_state = neuron_states[i]
            transition_matrix[state_index[prev_state], state_index[curr_state]] += 1

    # 转化为频率
    transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)
    return transition_matrix, unique_states

# 计算转移矩阵
transition_matrix, unique_states = build_transition_matrix(state_matrix)

# Step 4: 可视化状态之间的转换频率
# 使用Seaborn绘制热力图
def plot_transition_matrix(transition_matrix, unique_states):
    """
    使用热力图可视化状态转移矩阵。

    Args:
    - transition_matrix (ndarray): 状态转移矩阵。
    - unique_states (list): 状态的列表。
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(transition_matrix, annot=True, cmap='Blues', xticklabels=unique_states, yticklabels=unique_states)
    plt.title('State Transition Matrix')
    plt.xlabel('Next State')
    plt.ylabel('Previous State')
    plt.show()

# 显示状态转换矩阵的热力图
plot_transition_matrix(transition_matrix, unique_states)

# # Step 5: 使用桑基图可视化状态转换（可选）
# def plot_sankey_transition(unique_states, transition_matrix):
#     """
#     使用桑基图展示状态之间的转换。
#
#     Args:
#     - unique_states (list): 状态的列表。
#     - transition_matrix (ndarray): 状态转移矩阵。
#     """
#     flows = []
#     labels = []
#     orientations = []
#     for i in range(len(unique_states)):
#         for j in range(len(unique_states)):
#             if transition_matrix[i, j] > 0:
#                 flows.append(transition_matrix[i, j])
#                 labels.append(f"{unique_states[i]} -> {unique_states[j]}")
#                 orientations.append(1 if i < j else -1)
#
#     # 创建桑基图
#     sankey = Sankey(flows=flows, labels=labels, orientations=orientations)
#     sankey.finish()
#     plt.title('Sankey Diagram of State Transitions')
#     plt.show()
#
# # 绘制桑基图
# plot_sankey_transition(unique_states, transition_matrix)
