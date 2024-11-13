import pandas as pd
import plotly.graph_objects as go
from collections import Counter

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

# 处理神经元编号，将字符串中的编号部分提取为整数
df['Neuron_ID'] = df.iloc[:, 0].apply(lambda x: int(str(x).split('_')[-1]))
df['Time'] = df.iloc[:, 1]
df['Cluster'] = df.iloc[:, 8]

# 找到数量最多的簇类
most_common_cluster = df['Cluster'].mode()[0]

# 创建颜色映射表，将数量最多的簇类映射为白色
colors = ['#D62728', '#2CA02C', '#FF7F0E', '#1F77B4', '#FFFFFF']
unique_clusters = sorted(df['Cluster'].unique())
color_map = {cluster: colors[i % (len(colors) - 1)] for i, cluster in enumerate(unique_clusters)}
color_map[most_common_cluster] = '#FFFFFF'

# 使用 pivot_table 创建矩阵
pivot_df = df.pivot_table(index='Neuron_ID', columns='Time', values='Cluster', fill_value=most_common_cluster)

# 将簇类值映射为对应的颜色
raster_data = pivot_df.replace(color_map).values

# 绘制光栅图
fig = go.Figure(data=go.Heatmap(
    z=raster_data,
    x=pivot_df.columns,
    y=pivot_df.index,
    colorscale=[(i / len(colors), color) for i, color in enumerate(colors)],
    showscale=True,
    colorbar=dict(title="Cluster")
))

# 添加行为区间的虚线分割和交错排列的竖向标签
position_switch = True  # 开关，用于交替标签位置
for start_time, behavior in zip(behavior_timestamps[:-1], behavior_states[:-1]):
    start_time = float(start_time)
    if pd.notna(behavior):
        # 交替标签位置：True为上方，False为下方
        label_position = pivot_df.index.max() + 5 if position_switch else -5
        # 切换开关，实现上下交替
        position_switch = not position_switch
        # 添加竖排的行为标签
        vertical_text = "<br>".join(list(str(behavior)))  # 将字符串分成单个字符并竖排显示
        fig.add_trace(go.Scatter(
            x=[start_time],
            y=[label_position],
            text=[vertical_text],
            mode='text',
            showlegend=False
        ))

# 更新布局设置
fig.update_layout(
    title="Neuron Activity Raster Plot",
    xaxis_title="Time",
    yaxis_title="Neuron ID",
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False)
)

# 显示图像
fig.show()




