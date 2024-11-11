import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# 从文件中读取Excel的所有工作表
file_path = r'C:\Users\PAN\PycharmProjects\pythonProject\python-RA\标准化代码\calcium_window_hausdorff_cluster_results_all_sheets.xlsx'  # 请替换为你的 Excel 文件路径
sheets = pd.read_excel(file_path, sheet_name=None)  # 读取所有工作表

# 定义颜色列表（红、绿、黄、蓝），规模最大的不显示（用白色）
colors = ['#D62728', '#2CA02C', '#FF7F0E', '#1F77B4', '#FFFFFF']  # 红、绿、黄、蓝、白

# 遍历每个工作表
for sheet_name, df in sheets.items():
    # 提取神经元编号、时间戳和簇类号
    neuron_ids = df.iloc[:, 0]
    timestamps = df.iloc[:, 1]
    clusters = df.iloc[:, 8]

    # 统计每个簇类的数量并排序
    cluster_counts = Counter(clusters)
    sorted_clusters = [cluster for cluster, _ in cluster_counts.most_common()]  # 按数量降序排序
    most_common_cluster = sorted_clusters[0]  # 数量最多的簇类，用白色（不显示）

    # 设置颜色映射：数量最多的簇类用白色，其它按数量排序映射到红、绿、黄、蓝
    color_map = {
        cluster: (colors[i % (len(colors) - 1)] if cluster != most_common_cluster else colors[-1])
        for i, cluster in enumerate(sorted_clusters)
    }

    # 绘制时间活动条形图
    plt.figure(figsize=(15, 8))

    # 绘制条形图
    for neuron_id, time, cluster in zip(neuron_ids, timestamps, clusters):
        if cluster != most_common_cluster:  # 跳过数量最多的簇类
            color = color_map.get(cluster, 'white')  # 获取对应颜色，数量最多的簇类用白色
            plt.hlines(neuron_id, time, time + 50, colors=color, linewidth=2)

    # 设置 y 轴刻度，以 'Neuron_1', 'Neuron_10', ..., 'Neuron_60' 的格式显示
    plt.yticks(
        ticks=[1, 10, 20, 30, 40, 50, 60],
        labels=[f'Neuron_{i}' for i in [1, 10, 20, 30, 40, 50, 60]]
    )

    # 图形设置
    plt.xlabel("Time")
    plt.ylabel("Neuron ID")
    plt.title(f"Time Activity Raster Plot of Neurons by Cluster - {sheet_name} (Most Common Cluster Hidden)")

    # 显示图像而不保存
    plt.show()

print("所有工作表的图像已生成并显示。")



# # 设置文件路径
# file_path_neuron = r'C:\Users\PAN\PycharmProjects\pythonProject\python-RA\标准化代码\calcium_window_hausdorff_cluster_results_all_sheets.xlsx'  # 请替换为你的神经元 Excel 文件路径
# file_path_behavior = r'C:\Users\PAN\PycharmProjects\pythonProject\python-RA\Day6 behavior.xlsx'  # 请替换为你的行为状态 Excel 文件路径

# import matplotlib.pyplot as plt
# import pandas as pd
# from collections import Counter
#
# # 数据文件路径
# file_path_neuron = r'C:\Users\PAN\PycharmProjects\pythonProject\python-RA\标准化代码\calcium_window_hausdorff_cluster_results_all_sheets.xlsx'  # 请替换为你的神经元 Excel 文件路径
# file_path_behavior = r'C:\Users\PAN\PycharmProjects\pythonProject\python-RA\Day6 behavior.xlsx'  # 请替换为你的行为状态 Excel 文件路径
#
# # 从文件中读取神经元数据的所有工作表
# sheets_neuron = pd.read_excel(file_path_neuron, sheet_name=None)  # 读取所有神经元数据工作表
# behavior_data = pd.read_excel(file_path_behavior)  # 读取行为状态数据
#
# # 提取行为状态的时间戳和行为状态
# behavior_timestamps = behavior_data.iloc[:, 0]
# behavior_states = behavior_data.iloc[:, 2]
#
# # 定义颜色列表（红、绿、黄、蓝），规模最大的不显示（用白色）
# colors = ['#D62728', '#2CA02C', '#FF7F0E', '#1F77B4', '#FFFFFF']  # 红、绿、黄、蓝、白
#
# # 遍历每个神经元数据工作表
# for sheet_name, df in sheets_neuron.items():
#     # 提取神经元编号、时间戳和簇类号
#     neuron_ids = df.iloc[:, 0]
#     timestamps = df.iloc[:, 1]
#     clusters = df.iloc[:, 8]
#
#     # 统计每个簇类的数量并排序
#     cluster_counts = Counter(clusters)
#     sorted_clusters = [cluster for cluster, _ in cluster_counts.most_common()]  # 按数量降序排序
#     most_common_cluster = sorted_clusters[0]  # 数量最多的簇类，用白色（不显示）
#
#     # 设置颜色映射：数量最多的簇类用白色，其它按数量排序映射到红、绿、黄、蓝
#     color_map = {
#         cluster: (colors[i % (len(colors) - 1)] if cluster != most_common_cluster else colors[-1])
#         for i, cluster in enumerate(sorted_clusters)
#     }
#
#     # 绘制时间活动条形图
#     plt.figure(figsize=(15, 8))
#
#     # 绘制条形图
#     for neuron_id, time, cluster in zip(neuron_ids, timestamps, clusters):
#         if cluster != most_common_cluster:  # 跳过数量最多的簇类
#             color = color_map.get(cluster, 'white')  # 获取对应颜色，数量最多的簇类用白色
#             plt.hlines(neuron_id, time, time + 50, colors=color, linewidth=2)
#
#     # 在图上标记行为区间，跳过空白行为状态
#     for i in range(len(behavior_timestamps) - 1):
#         start_time = float(behavior_timestamps[i])  # 转换为浮点数
#         end_time = float(behavior_timestamps[i + 1])  # 转换为浮点数
#         behavior = behavior_states[i]
#
#         # 检查是否存在行为状态，跳过空缺值
#         if pd.notna(behavior):
#             # 用虚线标记区间，并在图上方显示行为状态
#             plt.axvline(x=start_time, color='gray', linestyle='--', linewidth=1)
#             plt.text((start_time + end_time) / 2, neuron_ids.max() + 5, str(behavior), ha='center', color='black')
#
#     # 设置 y 轴刻度，以 'Neuron_1', 'Neuron_10', ..., 'Neuron_60' 的格式显示
#     plt.yticks(
#         ticks=[1, 10, 20, 30, 40, 50, 60],
#         labels=[f'Neuron_{i}' for i in [1, 10, 20, 30, 40, 50, 60]]
#     )
#
#     # 图形设置
#     plt.xlabel("Time")
#     plt.ylabel("Neuron ID")
#     plt.title(f"Time Activity Raster Plot of Neurons by Cluster - {sheet_name} (Most Common Cluster Hidden)")
#
#     # 显示图像而不保存
#     plt.show()
#
# print("所有工作表的图像已生成并显示。")



