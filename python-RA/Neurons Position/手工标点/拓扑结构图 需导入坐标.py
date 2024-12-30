import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import itertools
from tqdm import tqdm  # 用于添加程序运行进度条
import warnings
import matplotlib.cm as cm
import matplotlib

warnings.filterwarnings('ignore')

# 设置字体
import plotly.io as pio

pio.templates.default = ("plotly")

# 1. 数据读取
neuron_data = pd.read_excel(
    r'C:\Users\PAN\Desktop\RA\数据集\NO2979\240924EM2\29792409EM2Trace.xlsx')  # 请修改为您的数据路径

# 获取神经元ID列表
neuron_ids = neuron_data.columns[1:]  # 假设第一列是Time，后面列是神经元ID

# 读取先前标注的坐标数据
positions_data = pd.read_csv(
    r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\Neurons Position\手工标点\clicked_points EM2.csv')

# 确保标点数与神经元数一致
if len(positions_data) != len(neuron_ids):
    raise ValueError("标记点数量与神经元数量不一致，请检查数据。")

# 将relative_x, relative_y转为pos
pos = {}
for nid, (rx, ry) in zip(neuron_ids, positions_data[['relative_x', 'relative_y']].values):
    pos[nid] = (rx, ry)

# 2. 为每个神经元计算平均值作为该神经元自己的阈值
threshold_dict = neuron_data[neuron_ids].mean().to_dict()

# 3. 初始化变量
group_id_counter = itertools.count(1)

# 颜色映射，给每个组分配一个颜色
# color_list = [
#     'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
#     'yellow', 'black', 'white']
# color_cycle = itertools.cycle(color_list)
N = 100  # 假设您最多有100个组
cmap = cm.get_cmap('tab20', N)  # 使用tab20色图生成100个颜色
color_list = [matplotlib.colors.to_hex(cmap(i)) for i in range(N)]
color_cycle = itertools.cycle(color_list)

# 5. 预计算所有帧的数据
frames = []
current_groups = {}  # 组ID: 神经元列表
neuron_to_group = {}  # 神经元ID: 组ID
group_colors = {}

# 用于存储每一帧的节点和边信息
frame_node_x = []
frame_node_y = []
frame_node_text = []
frame_node_color = []
frame_edge_x = []
frame_edge_y = []
frame_edge_color = []
frame_titles = []

for num in tqdm(range(len(neuron_data)), desc="预计算帧数据"):
    t = neuron_data['Time'].iloc[num]
    activity_values = neuron_data[neuron_ids].iloc[num]

    # 对每个神经元使用其本身的平均值作为阈值进行判断
    state = []
    for nid, val in zip(neuron_ids, activity_values):
        if val >= threshold_dict[nid]:
            state.append('ON')
        else:
            state.append('OFF')

    state_df = pd.DataFrame({
        'neuron_id': neuron_ids,
        'activity_value': activity_values.values,
        'state': state
    })

    # 更新神经元所属组
    # 处理未激活的神经元
    inactive_neurons = state_df[state_df['state'] == 'OFF']['neuron_id'].tolist()
    for neuron in inactive_neurons:
        if neuron in neuron_to_group:
            group_id = neuron_to_group[neuron]
            current_groups[group_id].remove(neuron)
            if len(current_groups[group_id]) == 0:
                # 如果组为空，删除该组
                del current_groups[group_id]
                del group_colors[group_id]
            del neuron_to_group[neuron]

    # 处理激活的神经元
    active_neurons = state_df[state_df['state'] == 'ON']['neuron_id'].tolist()
    # 找出未分配组的激活神经元
    ungrouped_active_neurons = [n for n in active_neurons if n not in neuron_to_group]

    if ungrouped_active_neurons:
        # 为这些神经元创建一个新的组
        new_group_id = next(group_id_counter)
        current_groups[new_group_id] = ungrouped_active_neurons
        for neuron in ungrouped_active_neurons:
            neuron_to_group[neuron] = new_group_id
        # 分配一个颜色
        group_colors[new_group_id] = next(color_cycle)

    # 构建当前时间点的图
    G = nx.Graph()
    G.add_nodes_from(neuron_ids)

    # 添加组内的连线
    edges = []
    edge_colors = ['black']
    for group_id, neurons in current_groups.items():
        # 在组内选择一个神经元，连接其他神经元
        representative_node = neurons[0]
        group_edges = [(representative_node, neuron) for neuron in neurons[1:]]
        edges.extend(group_edges)
        edge_colors.extend([group_colors[group_id]] * len(group_edges))

    G.add_edges_from(edges)

    # 为节点添加状态属性和颜色
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
        if node in neuron_to_group:
            group_id = neuron_to_group[node]
            node_color.append(group_colors[group_id])
        else:
            node_color.append('lightgray')

    # 边的坐标
    edge_x = []
    edge_y = []
    for idx, edge in enumerate(edges):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    frame_node_x.append(node_x)
    frame_node_y.append(node_y)
    frame_node_text.append(node_text)
    frame_node_color.append(node_color)
    frame_edge_x.append(edge_x)
    frame_edge_y.append(edge_y)
    frame_edge_color.append(edge_colors)
    frame_titles.append(f"神经元拓扑结构图 - 时间点：{t}")

# # 创建动画
# fig = go.Figure(
#     data=[
#         go.Scatter(
#             x=frame_edge_x[0],
#             y=frame_edge_y[0],
#             mode='lines',
#             line=dict(color='black', width=2),
#             hoverinfo='none'
#         ),
#         go.Scatter(
#             x=frame_node_x[0],
#             y=frame_node_y[0],
#             mode='markers+text',
#             text=frame_node_text[0],
#             textposition='middle center',
#             marker=dict(color=frame_node_color[0], size=15),
#             hoverinfo='text'
#         )
#     ],
#     layout=go.Layout(
#         title=frame_titles[0],
#         showlegend=False,
#         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#         plot_bgcolor='white',
#         sliders=[dict(
#             active=0,
#             steps=[dict(
#                 label=str(i),
#                 method="animate",
#                 args=[[f"frame_{i}"], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}],
#             ) for i in range(len(frame_node_x))]
#         )],
#         updatemenus=[dict(
#             type='buttons',
#             showactive=False,
#             buttons=[dict(
#                 label='Play',
#                 method='animate',
#                 args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)]
#             ), dict(
#                 label='Pause',
#                 method='animate',
#                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]
#             )]
#         )]
#     ),
#     frames=[
#         go.Frame(
#             data=[
#                 go.Scatter(
#                     x=frame_edge_x[k],
#                     y=frame_edge_y[k],
#                     mode='lines',
#                     line=dict(color='black', width=2),
#                     hoverinfo='none'
#                 ),
#                 go.Scatter(
#                     x=frame_node_x[k],
#                     y=frame_node_y[k],
#                     mode='markers+text',
#                     text=frame_node_text[k],
#                     textposition='middle center',
#                     marker=dict(color=frame_node_color[k], size=15),
#                     hoverinfo='text'
#                 )
#             ],
#             name=f"frame_{k}",
#             layout=go.Layout(title=frame_titles[k])
#         )
#         for k in range(len(frame_node_x))
#     ]
# )
#
# # 保存为HTML文件
# fig.write_html('neuron_activity_animation_individual_threshold.html')

# ...前面的代码保持不变...

# 创建动画
fig = go.Figure(
    data=[
        go.Scatter(
            x=frame_edge_x[0],
            y=frame_edge_y[0],
            mode='lines',
            line=dict(color='black', width=2),
            hoverinfo='none'
        ),
        go.Scatter(
            x=frame_node_x[0],
            y=frame_node_y[0],
            mode='markers+text',
            text=frame_node_text[0],
            textposition='middle center',
            marker=dict(color=frame_node_color[0], size=15),
            hoverinfo='text'
        )
    ],
    layout=go.Layout(
        title=frame_titles[0],
        showlegend=False,
        width=1437,    # 限制图的宽度
        height=949,    # 限制图的高度
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y', scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange='reversed'),
        plot_bgcolor='white',
        sliders=[dict(
            active=0,
            steps=[dict(
                label=str(i),
                method="animate",
                args=[[f"frame_{i}"], {"frame": {"duration": 1000, "redraw": True}, "mode": "immediate"}],
            ) for i in range(len(frame_node_x))]
        )],
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(
                label='Play',
                method='animate',
                args=[None, dict(frame=dict(duration=1000, redraw=True), fromcurrent=True)]  # 动画速度变慢
            ), dict(
                label='Pause',
                method='animate',
                args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]
            )]
        )]
    ),
    frames=[
        go.Frame(
            data=[
                go.Scatter(
                    x=frame_edge_x[k],
                    y=frame_edge_y[k],
                    mode='lines',
                    line=dict(color='black', width=2),
                    hoverinfo='none'
                ),
                go.Scatter(
                    x=frame_node_x[k],
                    y=frame_node_y[k],
                    mode='markers+text',
                    text=frame_node_text[k],
                    textposition='middle center',
                    marker=dict(color=frame_node_color[k], size=15),
                    hoverinfo='text'
                )
            ],
            name=f"frame_{k}",
            layout=go.Layout(title=frame_titles[k])
        )
        for k in range(len(frame_node_x))
    ]
)

import base64

# 将本地图片转换为 base64 编码
image_path = r'C:\Users\PAN\Desktop\RA\数据集\NO2979\240924EM2\2979240924EM2.png'  # 替换为你的图片路径
with open(image_path, "rb") as f:
    image_data = f.read()

encoded_image = base64.b64encode(image_data).decode("ascii")
image_source = "data:image/jpg;base64," + encoded_image

# 使用add_layout_image将图片添加为背景
fig.add_layout_image(
    dict(
        source=image_source,
        xref='paper',
        yref='paper',
        x=0.07,
        y=1.03,
        sizex=1,
        sizey=1,
        # xanchor='left',
        # yanchor='top',
        # sizing='stretch',
        opacity=0.5,    # 调整透明度，让前景元素更加醒目
        layer='below'    # 将图片置于图形的下层
    )
)

# 最后再输出HTML
fig.write_html('neuron_activity_animation_individual_threshold4.html')




