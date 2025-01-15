import pandas as pd
import networkx as nx
import os
import logging
import re

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 辅助函数：提取神经元序号
def extract_index(neuron_name):
    match = re.search(r'\d+', neuron_name)
    return int(match.group()) if match else float('inf')


# 1. 读取Excel文件
def read_calcium_data(file_path):
    """
    读取钙离子浓度Excel数据。

    参数:
    - file_path: Excel文件的路径。

    返回:
    - df: 包含钙离子浓度数据的DataFrame。
    """
    try:
        df = pd.read_excel(file_path)
        logging.info(f"成功读取文件: {file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"文件未找到: {file_path}")
        exit(1)
    except Exception as e:
        logging.error(f"读取文件时出错: {e}")
        exit(1)


# 2. 计算每个神经元的平均钙离子浓度
def compute_average(df):
    """
    计算每个神经元的平均钙离子浓度。

    参数:
    - df: 钙离子浓度DataFrame。

    返回:
    - avg_series: 包含每个神经元平均值的Series。
    """
    avg_series = df.mean(axis=0)
    logging.info("计算每个神经元的平均钙离子浓度完成")
    return avg_series


# 3. 根据平均值将神经元状态分类为ON/OFF
def classify_on_off(df, avg_series):
    """
    根据每个神经元的平均值将其状态分类为ON/OFF。

    参数:
    - df: 钙离子浓度DataFrame。
    - avg_series: 每个神经元的平均钙离子浓度。

    返回:
    - on_off_df: 包含ON/OFF状态的DataFrame。
    """
    on_off_df = df > avg_series
    on_off_df = on_off_df.astype(int)  # 1表示ON，0表示OFF
    logging.info("分类ON/OFF状态完成")
    return on_off_df


# 4. 生成拓扑结构并记录边信息
def generate_topologies(on_off_df):
    """
    根据ON/OFF状态生成每个时间戳的拓扑结构，并记录边信息。

    参数:
    - on_off_df: 包含ON/OFF状态的DataFrame。

    返回:
    - edge_records: 包含时间戳、节点1、节点2的列表。
    """
    edge_records = []  # 用于记录所有边的信息
    neurons = on_off_df.columns.tolist()

    # 排序所有神经元以确保序号最小的优先
    sorted_neurons = sorted(neurons, key=extract_index)

    for idx, row in on_off_df.iterrows():
        time_stamp = idx + 1  # 假设时间戳从1开始
        # 获取ON状态的神经元并按序号排序
        on_neurons = row[row == 1].index.tolist()
        on_neurons_sorted = sorted(on_neurons, key=extract_index)

        if not on_neurons_sorted:
            # 没有神经元处于ON状态，跳过
            continue

        # 连接逻辑：第一个神经元不连接任何边，其余每个神经元连接到序号最小的已连接神经元
        connected_neurons = [on_neurons_sorted[0]]  # 第一个神经元作为根节点
        for neuron in on_neurons_sorted[1:]:
            # 连接到已连接神经元中序号最小的那个
            # 由于已排序，最小序号的已连接神经元是connected_neurons[0]
            edge = (connected_neurons[0], neuron)
            edge_sorted = tuple(sorted(edge, key=extract_index))
            edge_records.append({'Time_Stamp': time_stamp, 'Neuron1': edge_sorted[0], 'Neuron2': edge_sorted[1]})
            connected_neurons.append(neuron)

        # 日志每100个时间戳
        if (idx + 1) % 100 == 0:
            logging.info(f"已生成 {idx + 1} 个拓扑结构")

    logging.info("所有拓扑结构生成完成")
    return edge_records


# 5. 生成拓扑连接矩阵并保存到Excel
def save_topology_matrix(edge_records, save_path='topology_matrix.xlsx'):
    """
    根据边记录生成拓扑连接矩阵，并保存到Excel文件。

    参数:
    - edge_records: 包含时间戳、节点1、节点2的列表。
    - save_path: 保存文件的路径。
    """
    try:
        edge_df = pd.DataFrame(edge_records)
        if edge_df.empty:
            logging.warning("边记录为空，未生成拓扑连接矩阵。")
            return

        # 创建一个新的列表示连接名称，按Neuron1_Neuron2排序
        edge_df['Connection'] = edge_df.apply(lambda row: f"{row['Neuron1']}_{row['Neuron2']}", axis=1)

        # 获取所有独特的连接并排序
        unique_connections = edge_df['Connection'].unique()
        unique_connections_sorted = sorted(unique_connections, key=lambda x: (extract_index(x.split('_')[0]), extract_index(x.split('_')[1])))

        # 获取所有时间戳并排序
        time_stamps = sorted(edge_df['Time_Stamp'].unique())

        # 初始化拓扑连接矩阵
        topology_matrix = pd.DataFrame(0, index=time_stamps, columns=unique_connections_sorted)
        topology_matrix.index.name = 'Time_Stamp'

        # 填充连接情况
        for _, row in edge_df.iterrows():
            topology_matrix.at[row['Time_Stamp'], row['Connection']] = 1

        # 重置索引以将Time_Stamp作为第一列
        topology_matrix = topology_matrix.reset_index()

        # 保存到Excel
        topology_matrix.to_excel(save_path, index=False)
        logging.info(f"已保存拓扑连接矩阵到 '{save_path}'")
    except Exception as e:
        logging.error(f"保存拓扑连接矩阵到Excel时出错: {e}")


# 6. 主函数
def main(file_path, should_save_topology_matrix=True,
         topology_matrix_path='topology_matrix.xlsx',
         initial_edges=None):
    """
    主函数执行整个流程。

    参数:
    - file_path: Excel文件路径。
    - should_save_topology_matrix: 是否保存拓扑连接矩阵到Excel。
    - topology_matrix_path: 拓扑连接矩阵Excel文件的保存路径。
    - initial_edges: 可选，初始的边列表，用于定义神经元之间的连接。

    返回:
    - edge_records: 拓扑边列表。
    """
    # 读取数据
    df = read_calcium_data(file_path)

    # 计算平均值
    avg_series = compute_average(df)

    # 分类ON/OFF
    on_off_df = classify_on_off(df, avg_series)

    # 生成拓扑结构并记录边信息
    edge_records = generate_topologies(on_off_df)

    # 保存拓扑连接矩阵到Excel
    if should_save_topology_matrix:
        save_topology_matrix(edge_records, save_path=topology_matrix_path)

    return edge_records


if __name__ == "__main__":
    # 示例用法
    excel_file = r"C:\Users\PAN\PycharmProjects\GitHub\python-RA\DeepLearning\Day6 训练结果\calcium_data.xlsx"  # 替换为你的Excel文件路径

    # 参数说明：
    # should_save_topology_matrix=True: 是否保存拓扑连接矩阵到Excel
    # topology_matrix_path='topology_matrix.xlsx': 拓扑连接矩阵Excel文件的保存路径
    # initial_edges=None: 如果有初始边列表，可以在这里传入

    edge_records = main(
        file_path=r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day6\Day6 calcium_data.xlsx',
        should_save_topology_matrix=True,  # 设置为True以保存拓扑连接矩阵到Excel
        topology_matrix_path=r"C:\Users\PAN\PycharmProjects\GitHub\python-RA\拓扑结构与图像聚类\topology_matrix.xlsx",
        initial_edges=None  # 如果需要，可以传入初始边列表
    )
