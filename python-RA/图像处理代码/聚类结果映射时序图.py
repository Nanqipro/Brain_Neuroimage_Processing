import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm  # 用于显示进度条

# 设置Excel文件路径
file_path = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\聚类算法代码\聚类结果\calcium_window_Hausdorff_weighted_output.xlsx'

# 指定要处理的工作表名称
sheet_name = input("请输入要处理的工作表名称: ")

# 定义关键列名称
neuron_col = 'Neuron'
time_col = 'Start Time'
category_col = 'Cluster'

# 读取指定工作表
data_df = pd.read_excel(file_path, sheet_name=sheet_name)

# 创建颜色映射字典
category_colors = {1: 'red', 2: 'green', 3: 'blue', 4: 'yellow', 5: 'orange', 6: 'gray'}
data_df['color'] = data_df[category_col].map(category_colors)

# 确保数据只包含必要的列
data_df = data_df[[neuron_col, time_col, category_col, 'color']]

# 创建动画图形
fig = go.Figure()

# 获取唯一时间戳
time_values = sorted(data_df[time_col].unique())

# 使用 tqdm 显示进度条
with tqdm(total=len(time_values), desc="Processing frames") as pbar:
    # 为每个时间戳生成帧
    for time in time_values:
        time_data = data_df[data_df[time_col] == time]
        fig.add_trace(go.Scatter(
            x=time_data[neuron_col],
            y=time_data[category_col],
            mode='markers',
            marker=dict(size=10, color=time_data['color'], opacity=0.8),
            name=f'Time {time}'
        ))
        pbar.update(1)  # 更新进度条

# 设置帧参数，使得时间可拖动
frames = [go.Frame(data=[go.Scatter(
                        x=data_df[data_df[time_col] == t][neuron_col],
                        y=data_df[data_df[time_col] == t][category_col],
                        mode='markers',
                        marker=dict(color=data_df[data_df[time_col] == t]['color'], size=10)
                    )],
                   name=f"Time {t}") for t in time_values]

# 将帧添加到图形中
fig.frames = frames

# 添加播放按钮和拖动条
fig.update_layout(
    title=f'Neuron Clusters Over Time - {sheet_name}',
    xaxis_title='Neuron',
    yaxis_title='Cluster',
    updatemenus=[dict(type="buttons",
                      showactive=False,
                      buttons=[dict(label="Play",
                                    method="animate",
                                    args=[None, {"frame": {"duration": 500, "redraw": True},
                                                 "fromcurrent": True}]),
                               dict(label="Pause",
                                    method="animate",
                                    args=[[None], {"frame": {"duration": 0, "redraw": True},
                                                   "mode": "immediate",
                                                   "transition": {"duration": 0}}])
                              ])],
    sliders=[dict(steps=[dict(method='animate',
                              args=[[f'Time {t}'], dict(mode='immediate',
                                                        frame=dict(duration=500, redraw=True),
                                                        transition=dict(duration=0))],
                              label=f'{t}') for t in time_values],
                 active=0,
                 transition=dict(duration=0),
                 x=0.1,  # 可以调整拖动条的位置
                 y=0,
                 currentvalue=dict(font=dict(size=20), prefix='Time:', visible=True),
                 len=0.9)]
)

# 保存图像为HTML文件和静态PNG图片
html_output = "neuron_clusters_animation.html"
png_output = "neuron_clusters_snapshot.png"

fig.write_html(html_output)
fig.write_image(png_output)

print(f"图表已保存为 {html_output} 和 {png_output}")

