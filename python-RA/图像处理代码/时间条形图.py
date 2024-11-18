import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
from collections import Counter

# 设置数据文件路径
file_path_neuron = r"C:\Users\PAN\PycharmProjects\GitHub\python-RA\聚类算法代码\聚类结果\calcium_window_Hausdorff_weighted_output.xlsx"  # 请替换为你的神经元 Excel 文件路径
file_path_behavior = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day6\Day6 behavior.xlsx'  # 请替换为你的行为状态 Excel 文件路径

# 从文件中读取神经元数据的特定工作表
sheet_name_to_plot = 'Window_50_Step_5'  # 请指定要生成图像的工作表名称
df = pd.read_excel(file_path_neuron, sheet_name=sheet_name_to_plot)

# 读取行为状态数据，跳过第一行标签行
behavior_data = pd.read_excel(file_path_behavior, skiprows=1)

# 提取行为状态的时间戳和行为状态
behavior_timestamps = behavior_data.iloc[:, 0]
behavior_states = behavior_data.iloc[:, 2]

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

# 绘制行为区间虚线并显示标签，使用进度条
position_switch = True  # 开关，用于交替标签位置
for start_time, behavior in tqdm(zip(behavior_timestamps[:-1], behavior_states[:-1]), total=len(behavior_timestamps)-1, desc="绘制行为标签"):
    start_time = float(start_time)
    if pd.notna(behavior):
        # 交替标签位置：True为上方，False为下方
        label_position = neuron_ids.max() + 10 if position_switch else -5
        # 切换开关，实现上下交替
        position_switch = not position_switch
        # 添加虚线
        fig.add_trace(go.Scattergl(
            x=[start_time, start_time],
            y=[0, neuron_ids.max() + 10],
            mode='lines',
            line=dict(color='gray', dash='dash', width=1),
            showlegend=False,
            hoverinfo='text',  # 鼠标悬停时显示行为标签
            text=behavior  # 行为标签内容
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
    height=1100, width=2000  # 调整图像大小
)

# 显示交互式图像
fig.write_html('neuron_raster_plot_Day6win50step5.html')  # 保存图形为HTML文件
