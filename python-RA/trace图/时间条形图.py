# import matplotlib.pyplot as plt
# import pandas as pd
# from collections import Counter
#
# # 设置字体，确保可以显示中文
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 显示中文
# plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题
#
# # 数据文件路径
# file_path_neuron = r'C:\Users\PAN\PycharmProjects\Brain_Neuroimage_Processing\python-RA\标准化代码\calcium_window_emd_cluster_results_all_sheets.xlsx'  # 请替换为你的神经元 Excel 文件路径
# file_path_behavior = r'C:\Users\PAN\PycharmProjects\Brain_Neuroimage_Processing\python-RA\Day6 behavior.xlsx'  # 请替换为你的行为状态 Excel 文件路径
#
# # 从文件中读取神经元数据的特定工作表
# sheet_name_to_plot = 'Window_30_Step_10'  # 请指定要生成图像的工作表名称
# df = pd.read_excel(file_path_neuron, sheet_name=sheet_name_to_plot)
#
# # 读取行为状态数据，跳过第一行标签行
# behavior_data = pd.read_excel(file_path_behavior, skiprows=1)
#
# # 提取行为状态的时间戳和行为状态
# behavior_timestamps = behavior_data.iloc[:, 0]
# behavior_states = behavior_data.iloc[:, 2]
#
# # 定义颜色列表（红、绿、黄、蓝），规模最大的不显示（用白色）
# colors = ['#D62728', '#2CA02C', '#FF7F0E', '#1F77B4', '#FFFFFF']  # 红、绿、黄、蓝、白
#
# # 提取神经元编号、时间戳和簇类号
# neuron_ids = df.iloc[:, 0].apply(lambda x: int(str(x).split('_')[-1]))  # 提取编号并转换为整数
# timestamps = df.iloc[:, 1]
# clusters = df.iloc[:, 8]
#
# # 统计当前工作表中每个簇类的数量
# cluster_counts = Counter(clusters)
#
# # 找到数量最多的簇类
# sorted_clusters = [cluster for cluster, _ in cluster_counts.most_common()]  # 按数量降序排序
# most_common_cluster = sorted_clusters[0]  # 数量最多的簇类，用白色（不显示）
#
# # 设置颜色映射：数量最多的簇类用白色，其它按数量排序映射到红、绿、黄、蓝
# color_map = {
#     cluster: (colors[i % (len(colors) - 1)] if cluster != most_common_cluster else colors[-1])
#     for i, cluster in enumerate(sorted_clusters)
# }
#
# # 绘制时间活动条形图
# plt.figure(figsize=(25, 8))
#
# # 绘制条形图
# for neuron_id, time, cluster in zip(neuron_ids, timestamps, clusters):
#     if cluster != most_common_cluster:  # 跳过数量最多的簇类
#         color = color_map.get(cluster, 'white')  # 获取对应颜色，数量最多的簇类用白色
#         plt.hlines(neuron_id, time, time + 50, colors=color, linewidth=2)
#
# # 在图上标记行为区间的起始位置，跳过空白行为状态
# for i in range(len(behavior_timestamps) - 1):
#     start_time = float(behavior_timestamps[i])  # 转换为浮点数
#     behavior = behavior_states[i]
#
#     # 检查是否存在行为状态，跳过空缺值
#     if pd.notna(behavior):
#         # 用虚线标记行为起始位置
#         plt.axvline(x=start_time, color='gray', linestyle='--', linewidth=1)
#         # 在图下方竖直显示行为状态
#         plt.text(start_time, -5, str(behavior), ha='center', va='top', rotation=90, color='black')
#
# # 设置 y 轴刻度，以 'Neuron_1', 'Neuron_10', ..., 'Neuron_60' 的格式显示
# plt.yticks(
#     ticks=[1, 10, 20, 30, 40, 50, 60],
#     labels=[f'Neuron_{i}' for i in [1, 10, 20, 30, 40, 50, 60]]
# )
#
# # 图形设置
# plt.xlabel("Time")
# plt.ylabel("Neuron ID")
# plt.title(f"Time Activity Raster Plot of Neurons by Cluster - {sheet_name_to_plot} (Most Common Cluster Hidden)")
#
# # 显示图像而不保存
# plt.show()

import pandas as pd
import plotly.graph_objects as go
from collections import Counter
from numba import jit
import plotly.io as pio

# 加速数据处理函数
@jit(nopython=True)
def process_neuron_ids(neuron_ids_column):
    return [int(str(x).split('_')[-1]) for x in neuron_ids_column]

# 设置数据文件路径
file_path_neuron = r'C:\Users\PAN\PycharmProjects\Brain_Neuroimage_Processing\python-RA\标准化代码\calcium_Hausdorff_test.xlsx'  # 请替换为你的神经元 Excel 文件路径
file_path_behavior = r'C:\Users\PAN\PycharmProjects\Brain_Neuroimage_Processing\python-RA\Day6 behavior.xlsx'  # 请替换为你的行为状态 Excel 文件路径

# 从文件中读取神经元数据的特定工作表
sheet_name_to_plot = 'Sheet1'  # 请指定要生成图像的工作表名称
df = pd.read_excel(file_path_neuron, sheet_name=sheet_name_to_plot)

# 读取行为状态数据，跳过第一行标签行
behavior_data = pd.read_excel(file_path_behavior, skiprows=1)

# 提取行为状态的时间戳和行为状态
behavior_timestamps = behavior_data.iloc[:, 0]
behavior_states = behavior_data.iloc[:, 2]

# 定义颜色列表（红、绿、黄、蓝），最大簇类不显示（用白色）
colors = ['#D62728', '#2CA02C', '#FF7F0E', '#1F77B4', '#FFFFFF']

# 使用 JIT 加速处理神经元编号
neuron_ids = process_neuron_ids(df.iloc[:, 0].values)  # 使用加速函数处理神经元编号
timestamps = df.iloc[:, 1]
clusters = df.iloc[:, 8]

# 统计当前工作表中每个簇类的数量
cluster_counts = Counter(clusters)

# 找到数量最多的簇类
sorted_clusters = [cluster for cluster, _ in cluster_counts.most_common()]  # 按数量降序排序
most_common_cluster = sorted_clusters[0]  # 数量最多的簇类，用白色（不显示）

# 设置颜色映射：数量最多的簇类用白色，其它按数量排序映射到红、绿、黄、蓝
color_map = {
    cluster: (colors[i % (len(colors) - 1)] if cluster != most_common_cluster else colors[-1])
    for i, cluster in enumerate(sorted_clusters)
}

# 创建一个 Plotly 图形对象
fig = go.Figure()

# 绘制每个神经元的条形图
for neuron_id, time, cluster in zip(neuron_ids, timestamps, clusters):
    if cluster != most_common_cluster:  # 跳过数量最多的簇类
        color = color_map.get(cluster, 'white')  # 获取对应颜色
        fig.add_trace(go.Scatter(
            x=[time, time + 50],  # 窗口大小假设为 50
            y=[neuron_id, neuron_id],
            mode='lines',
            line=dict(color=color, width=4),
            showlegend=False,
            visible=True if neuron_id == neuron_ids[0] else "legendonly",  # 默认只显示第一个神经元
            name=f'Neuron_{neuron_id}'
        ))

# 添加行为区间的虚线分割和交错排列的竖向标签
position_switch = True  # 开关，用于交替标签位置
for start_time, behavior in zip(behavior_timestamps[:-1], behavior_states[:-1]):
    start_time = float(start_time)
    if pd.notna(behavior):
        # 交替标签位置：True为上方，False为下方
        label_position = max(neuron_ids) + 10 if position_switch else -5
        # 切换开关，实现上下交替
        position_switch = not position_switch
        # 添加虚线
        fig.add_trace(go.Scatter(
            x=[start_time, start_time],
            y=[0, max(neuron_ids) + 10],
            mode='lines',
            line=dict(color='gray', dash='dash', width=1),
            showlegend=False
        ))
        # 将行为标签竖排显示
        vertical_text = "<br>".join(list(str(behavior)))  # 将字符串分成单个字符，并用换行符拼接
        fig.add_trace(go.Scatter(
            x=[start_time],
            y=[label_position],
            text=[vertical_text],
            mode='text',
            showlegend=False
        ))

# 设置图形布局
fig.update_layout(
    title=f"Time Activity Raster Plot of Neurons by Cluster - {sheet_name_to_plot} (Most Common Cluster Hidden)",
    xaxis_title="Time",
    yaxis_title="Neuron ID",
    yaxis=dict(
        tickmode='array',
        tickvals=[1, 10, 20, 30, 40, 50, 60],
        ticktext=[f'Neuron_{i}' for i in [1, 10, 20, 30, 40, 50, 60]]
    ),
    height=1100, width=1900,  # 调整图像大小
)

# 使用 plotly.io.show 打开默认浏览器显示
pio.show(fig)

# 或将图表保存为 HTML 文件并在浏览器中打开
fig.write_html("interactive_neuron_plot.html")

