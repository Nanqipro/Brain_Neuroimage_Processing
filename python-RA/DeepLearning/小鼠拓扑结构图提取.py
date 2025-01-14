import pandas as pd
import networkx as nx
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
def generate_topologies(on_off_df, initial_edges=None):
    """
    根据ON/OFF状态生成每个时间戳的拓扑结构，并记录边信息。

    参数:
    - on_off_df: 包含ON/OFF状态的DataFrame。
    - initial_edges: 可选，初始的边列表，用于定义神经元之间的连接。

    返回:
    - topologies: 包含每个时间戳拓扑结构的列表（NetworkX图对象）。
    - edge_records: 包含时间戳、节点1、节点2的列表。
    """
    topologies = []
    edge_records = []  # 用于记录所有边的信息
    neurons = on_off_df.columns.tolist()

    for idx, row in on_off_df.iterrows():
        time_stamp = idx + 1  # 假设时间戳从1开始
        # 获取ON状态的神经元
        on_neurons = row[row == 1].index.tolist()

        # 创建一个空图
        G = nx.Graph()

        # 添加ON神经元作为节点
        G.add_nodes_from(on_neurons)

        # 添加边
        if initial_edges:
            # 如果提供了初始边列表，只添加在initial_edges中的边
            edges = [(u, v) for (u, v) in initial_edges if u in on_neurons and v in on_neurons]
        else:
            # 否则，假设所有ON神经元之间是全连接的
            if len(on_neurons) > 1:
                edges = [(on_neurons[i], on_neurons[j])
                         for i in range(len(on_neurons))
                         for j in range(i + 1, len(on_neurons))]
            else:
                edges = []

        G.add_edges_from(edges)

        # 记录边信息
        for edge in edges:
            edge_records.append({'Time_Stamp': time_stamp, 'Neuron1': edge[0], 'Neuron2': edge[1]})

        topologies.append(G)

        # 日志每100个时间戳
        if (idx + 1) % 100 == 0:
            logging.info(f"已生成 {idx + 1} 个拓扑结构")

    logging.info("所有拓扑结构生成完成")
    return topologies, edge_records


# 5. 保存拓扑结构
def save_topologies(topologies, save_format='gexf', save_dir='topologies'):
    """
    保存所有拓扑结构到指定目录。

    参数:
    - topologies: 包含所有NetworkX图对象的列表。
    - save_format: 保存的文件格式，如 'gexf', 'graphml' 等。
    - save_dir: 保存目录名称。
    """
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
            logging.info(f"创建保存目录: {save_dir}")
        except Exception as e:
            logging.error(f"创建目录 {save_dir} 时出错: {e}")
            exit(1)

    for idx, G in enumerate(topologies):
        time_stamp = idx + 1
        file_path = os.path.join(save_dir, f"topology_{time_stamp}.{save_format}")
        try:
            if save_format == 'gexf':
                nx.write_gexf(G, file_path)
            elif save_format == 'graphml':
                nx.write_graphml(G, file_path)
            else:
                logging.error("不支持的保存格式。使用 'gexf' 或 'graphml'。")
                continue
            if (idx + 1) % 100 == 0:
                logging.info(f"已保存 {idx + 1} 个拓扑文件")
        except Exception as e:
            logging.error(f"保存时间戳 {time_stamp} 的拓扑结构时出错: {e}")

    logging.info("所有拓扑结构文件保存完成")


# 6. 保存边列表到表格
def save_edge_list(edge_records, save_path='topology_edges.xlsx'):
    """
    保存所有边信息到一个Excel文件中。

    参数:
    - edge_records: 包含时间戳、节点1、节点2的列表。
    - save_path: 保存文件的路径。
    """
    try:
        edge_df = pd.DataFrame(edge_records)
        edge_df.to_excel(save_path, index=False)
        logging.info(f"已保存所有拓扑边信息到 '{save_path}'")
    except Exception as e:
        logging.error(f"保存边信息到表格时出错: {e}")


# 7. 主函数
def main(file_path, save_topology_files=False,
         topology_format='gexf', topology_dir='topologies',
         save_edge_list_flag=True, edge_list_path='topology_edges.xlsx'):
    """
    主函数执行整个流程。

    参数:
    - file_path: Excel文件路径。
    - save_topology_files: 是否保存拓扑结构文件。
    - topology_format: 拓扑结构文件的格式，如 'gexf', 'graphml'。
    - topology_dir: 拓扑结构文件的保存目录。
    - save_edge_list_flag: 是否保存拓扑边列表表格。
    - edge_list_path: 拓扑边列表表格的保存路径。

    返回:
    - topologies: 拓扑结构列表。
    - on_off_df: ON/OFF状态DataFrame。
    - df: 原始钙离子浓度DataFrame。
    - edge_records: 拓扑边列表。
    """
    # 读取数据
    df = read_calcium_data(file_path)

    # 计算平均值
    avg_series = compute_average(df)

    # 分类ON/OFF
    on_off_df = classify_on_off(df, avg_series)

    # 生成拓扑结构并记录边信息
    topologies, edge_records = generate_topologies(on_off_df)

    # 保存ON/OFF状态
    try:
        on_off_df.to_excel("on_off_states.xlsx", index=False)
        logging.info("已保存神经元的ON/OFF状态到 'on_off_states.xlsx'")
    except Exception as e:
        logging.error(f"保存ON/OFF状态时出错: {e}")

    # 保存原始钙离子浓度数据（可选）
    try:
        df.to_excel("calcium_concentrations.xlsx", index=False)
        logging.info("已保存钙离子浓度数据到 'calcium_concentrations.xlsx'")
    except Exception as e:
        logging.error(f"保存钙离子浓度数据时出错: {e}")

    # 保存拓扑结构文件
    if save_topology_files:
        save_topologies(topologies, save_format=topology_format, save_dir=topology_dir)

    # 保存拓扑边列表到表格
    if save_edge_list_flag:
        save_edge_list(edge_records, save_path=edge_list_path)

    return topologies, on_off_df, df, edge_records


if __name__ == "__main__":
    # 示例用法
    excel_file = r"C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day6\calcium_data.xlsx"  # 替换为你的Excel文件路径

    # 参数说明：
    # save_topology_files=True: 是否保存拓扑结构文件
    # topology_format='gexf': 拓扑结构文件格式，可以是 'gexf' 或 'graphml'
    # topology_dir='topologies': 拓扑结构文件保存目录
    # save_edge_list_flag=True: 是否保存拓扑边列表表格
    # edge_list_path='topology_edges.xlsx': 拓扑边列表表格的保存路径

    topologies, on_off_df, df, edge_records = main(
        file_path=excel_file,
        save_topology_files=True,  # 设置为True以保存拓扑结构文件
        topology_format='gexf',
        topology_dir=r"C:\Users\PAN\PycharmProjects\GitHub\python-RA\DeepLearning\Day6 训练结果\topologies",
        save_edge_list_flag=True,  # 设置为True以保存拓扑边列表表格
        edge_list_path=r"C:\Users\PAN\PycharmProjects\GitHub\python-RA\DeepLearning\Day6 训练结果\topology_edges.xlsx"
    )
