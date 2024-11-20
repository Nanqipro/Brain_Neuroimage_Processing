# import pandas as pd
# import plotly.graph_objects as go
# from tqdm import tqdm
# from collections import Counter
#
# # 设置数据文件路径
# file_path_neuron = r"C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day3\calcium_window_result_weighted_output_Day3.xlsx"  # 请替换为你的神经元 Excel 文件路径
# file_path_behavior = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day3\Day3 behavior.xlsx'  # 请替换为你的行为状态 Excel 文件路径
#
# # 从文件中读取神经元数据的特定工作表
# sheet_name_to_plot = 'Window_50_Step_10'  # 请指定要生成图像的工作表名称
# df = pd.read_excel(file_path_neuron, sheet_name=sheet_name_to_plot)
#
# # 读取行为状态数据，跳过第一行标签行
# behavior_data = pd.read_excel(file_path_behavior, skiprows=1)
#
# # 提取行为状态的时间戳和行为状态
# behavior_timestamps = behavior_data.iloc[:, 0]
# behavior_states = behavior_data.iloc[:, 1]
#
# # 定义颜色列表（红、绿、黄、蓝），最大簇类不显示（用白色）
# colors = ['#D62728', '#2CA02C', '#FF7F0E', '#1F77B4', '#FFFFFF']
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
# # 创建一个 Plotly 图形对象
# fig = go.Figure()
#
# # 绘制每个神经元的条形图，使用进度条
# for neuron_id, time, cluster in tqdm(zip(neuron_ids, timestamps, clusters), total=len(neuron_ids), desc="绘制神经元条形图"):
#     if cluster != most_common_cluster:  # 跳过数量最多的簇类
#         color = color_map.get(cluster, 'white')  # 获取对应颜色
#         fig.add_trace(go.Scattergl(
#             x=[time, time + 50],  # 窗口大小假设为 50
#             y=[neuron_id, neuron_id],
#             mode='lines',
#             line=dict(color=color, width=4),
#             showlegend=False
#         ))
#
# # 使用注释绘制行为标签
# annotations = []
# for idx, (start_time, behavior) in enumerate(zip(behavior_timestamps[:-1], behavior_states[:-1])):
#     if pd.notna(behavior):
#         start_time = float(start_time)
#         # 上下位置交替：上侧为神经元最大值上方 +10，下侧为 -10
#         label_position = neuron_ids.max() + 11 if idx % 2 == 0 else -2
#         # 添加虚线
#         fig.add_trace(go.Scattergl(
#             x=[start_time, start_time],
#             y=[0, neuron_ids.max() + 10],
#             mode='lines',
#             line=dict(color='gray', dash='dash', width=1),
#             showlegend=False
#         ))
#         # 添加行为标签注释
#         annotations.append(
#             dict(
#                 x=start_time,
#                 y=label_position,
#                 text=behavior,
#                 showarrow=False,
#                 textangle=270 if idx % 2 == 0 else -90,  # 上下侧旋转角度
#                 font=dict(size=11, color="black"),
#                 xanchor="center",
#                 yanchor="middle"
#             )
#         )
#
# # 将注释添加到图形中
# fig.update_layout(annotations=annotations)
#
# # 设置图形布局
# fig.update_layout(
#     title=f"Time Activity Raster Plot of Neurons by Cluster - {sheet_name_to_plot} (Most Common Cluster Hidden)",
#     xaxis_title="Time",
#     yaxis_title="Neuron ID",
#     yaxis=dict(
#         tickmode='array',
#         tickvals=[1, 10, 20, 30, 40, 50, 60],
#         ticktext=[f'Neuron_{i}' for i in [1, 10, 20, 30, 40, 50, 60]]
#     ),
#     height=1250, width=2050,  # 调整图像大小
#     margin=dict(b=0, t=0)  # 确保上下空间充足
# )
#
# # 显示交互式图像
# fig.write_html('neuron_raster_plot_Day3win50step10.html')  # 保存图形为HTML文件
#
#


import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
from collections import Counter

# 设置数据文件路径
file_path_neuron = r"C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day9\calcium_window_result_weighted_output_Day9.xlsx"  # 请替换为你的神经元 Excel 文件路径
file_path_behavior = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day9\calcium_data9.xlsx'  # 请替换为你的行为状态 Excel 文件路径

# 从文件中读取神经元数据的特定工作表
sheet_name_to_plot = 'Window_50_Step_10'  # 请指定要生成图像的工作表名称
df = pd.read_excel(file_path_neuron, sheet_name=sheet_name_to_plot)

# 读取行为状态数据，跳过第一行标签行
behavior_data = pd.read_excel(file_path_behavior, skiprows=1)

# 提取行为状态的时间戳和行为状态
behavior_timestamps = behavior_data.iloc[:, 0]
behavior_states = behavior_data.iloc[:, 1]

# 定义颜色列表（红、绿、黄、蓝），最大簇类不显示（用白色）
colors = ['#D62728', '#2CA02C', '#FF7F0E', '#1F77B4', '#FFFFFF']

# 提取神经元编号、时间戳和簇类号
neuron_ids = df.iloc[:, 0].apply(lambda x: int(str(x).split('_')[-1]))  # 提取编号并转换为整数
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

# 绘制每个神经元的条形图，使用进度条
for neuron_id, time, cluster in tqdm(zip(neuron_ids, timestamps, clusters), total=len(neuron_ids), desc="绘制神经元条形图"):
    if cluster != most_common_cluster:  # 跳过数量最多的簇类
        color = color_map.get(cluster, 'white')  # 获取对应颜色
        fig.add_trace(go.Scattergl(
            x=[time, time + 50],  # 窗口大小假设为 50
            y=[neuron_id, neuron_id],
            mode='lines',
            line=dict(color=color, width=4),
            showlegend=False
        ))

# 使用注释绘制行为标签
annotations = []
for idx, (start_time, behavior) in enumerate(zip(behavior_timestamps[:-1], behavior_states[:-1])):
    if pd.notna(behavior):
        start_time = float(start_time)
        # 上下位置交替：上侧为神经元最大值上方 +11，下侧为 -2
        label_position = neuron_ids.max() + 11 if idx % 2 == 0 else -2
        # 将行为标签竖向排列：每个字符用 <br> 分隔
        vertical_text = "<br>".join(list(str(behavior)))  # 转换为竖向文本
        # 添加虚线
        fig.add_trace(go.Scattergl(
            x=[start_time, start_time],
            y=[0, neuron_ids.max() + 10],
            mode='lines',
            line=dict(color='gray', dash='dash', width=1),
            showlegend=False
        ))
        # 添加行为标签注释
        annotations.append(
            dict(
                x=start_time,
                y=label_position,
                text=vertical_text,
                showarrow=False,
                font=dict(size=12, color="black"),
                xanchor="center",
                yanchor="middle"
            )
        )

# 将注释添加到图形中
fig.update_layout(annotations=annotations)

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
    height=1250, width=2050,  # 调整图像大小
    margin=dict(b=50, t=50)  # 确保上下空间充足
)

# 显示交互式图像
fig.write_html('neuron_raster_plot_Day9win50step10_vertical.html')  # 保存图形为HTML文件
