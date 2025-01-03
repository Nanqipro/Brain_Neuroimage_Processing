import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import itertools
from tqdm import tqdm  # 用于添加程序运行进度条
import warnings
import matplotlib.cm as cm
import matplotlib
import json

warnings.filterwarnings('ignore')

# 设置字体
import plotly.io as pio

pio.templates.default = "plotly"

# 1. 数据读取
neuron_data = pd.read_excel(
    r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\Neurons Position\240924EM\240924EM trace.xlsx')  # 请修改为您的数据路径

# 获取神经元ID列表
neuron_ids = neuron_data.columns[1:]  # 假设第一列是Time，后面列是神经元ID

# 读取先前标注的坐标数据
positions_data = pd.read_csv(
    r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\Neurons Position\手工标点\clicked_points EM.csv')

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

N = 100  # 假设您最多有100个组
cmap = cm.get_cmap('tab20', N)  # 使用tab20色图生成100个颜色
color_list = [matplotlib.colors.to_hex(cmap(i)) for i in range(N)]
color_cycle = itertools.cycle(color_list)

# 4. 预计算所有帧的数据
# 初始化用于存储每一帧的网络结构
network_structures = []

# 初始化用于存储每一帧的节点和边信息（用于动画）
frame_node_x = []
frame_node_y = []
frame_node_text = []
frame_node_color = []
frame_edge_x = []
frame_edge_y = []
frame_titles = []

current_groups = {}  # 组ID: 神经元列表
neuron_to_group = {}  # 神经元ID: 组ID
group_colors = {}

# 5. 读取行为标签数据（假设有一个对应的CSV文件）
# 如果没有行为标签，请跳过此部分或自行添加
# 这里假设行为标签文件包含两列：Time 和 Behavior
# behavior_data = pd.read_csv(r'path_to_behavior_labels.csv')
# 将行为标签与neuron_data按Time进行匹配
# 在这里我们假设 behavior_data 已经被正确读取和匹配

# 如果没有行为标签文件，可以创建一个示例标签
# 请根据实际情况替换或删除此部分
# 示例：假设每个时间点的行为标签为 "Behavior_{time}"
# behavior_labels = {t: f"Behavior_{t}" for t in neuron_data['Time']}
# 如果您有实际的行为标签，请替换上述代码

# 示例：创建随机行为标签（请根据实际情况替换）
# import random
# possible_behaviors = ['Resting', 'Running', 'Eating', 'Exploring']
# behavior_labels = {t: random.choice(possible_behaviors) for t in neuron_data['Time']}

# 请取消下面的注释并加载您的实际行为标签
# behavior_labels = dict(zip(behavior_data['Time'], behavior_data['Behavior']))

# 如果没有行为标签，使用默认标签
behavior_labels = {t: f"Behavior_{t}" for t in neuron_data['Time']}

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
    edge_colors = []
    for group_id, neurons in current_groups.items():
        if len(neurons) < 2:
            continue  # 需要至少两个神经元才能形成边
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
    node_features = []  # 用于存储节点特征，例如活动值
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
    for edge in edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # 保存当前帧的节点和边信息
    frame_node_x.append(node_x)
    frame_node_y.append(node_y)
    frame_node_text.append(node_text)
    frame_node_color.append(node_color)
    frame_edge_x.append(edge_x)
    frame_edge_y.append(edge_y)
    frame_titles.append(f"神经元拓扑结构图 - 时间点：{t}")

    # 保存网络结构到列表
    network_structure = {
        'time': t,
        'nodes': [{'Neuron': node, 'x': pos[node][0], 'y': pos[node][1],
                   'color': node_color[i], 'state': state_df[state_df['neuron_id'] == node]['state'].values[0],
                   'activity_value': state_df[state_df['neuron_id'] == node]['activity_value'].values[0],
                   'group_id': neuron_to_group.get(node, None)} for i, node in enumerate(G.nodes())],
        'edges': [{'source': edge[0], 'target': edge[1], 'color': edge_colors[i]}
                  for i, edge in enumerate(edges)],
        'behavior': behavior_labels.get(t, "Unknown")  # 添加行为标签
    }
    network_structures.append(network_structure)

# 6. 创建动画
fig = go.Figure(
    data=[
        go.Scatter(
            x=frame_edge_x[0],
            y=frame_edge_y[0],
            mode='lines',
            line=dict(color='black', width=2),
            hoverinfo='none',
            showlegend=False
        ),
        go.Scatter(
            x=frame_node_x[0],
            y=frame_node_y[0],
            mode='markers+text',
            text=frame_node_text[0],
            textposition='middle center',
            marker=dict(color=frame_node_color[0], size=15),
            hoverinfo='text',
            showlegend=False
        )
    ],
    layout=go.Layout(
        title=frame_titles[0],
        showlegend=False,
        width=1437,  # 限制图的宽度
        height=949,  # 限制图的高度
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y', scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange='reversed'),
        plot_bgcolor='white',
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "帧: "},
            pad={"t": 50},
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
                args=[None, dict(frame=dict(duration=1000, redraw=True), fromcurrent=True)]
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
                    hoverinfo='none',
                    showlegend=False
                ),
                go.Scatter(
                    x=frame_node_x[k],
                    y=frame_node_y[k],
                    mode='markers+text',
                    text=frame_node_text[k],
                    textposition='middle center',
                    marker=dict(color=frame_node_color[k], size=15),
                    hoverinfo='text',
                    showlegend=False
                )
            ],
            name=f"frame_{k}",
            layout=go.Layout(title=frame_titles[k])
        )
        for k in range(len(frame_node_x))
    ]
)

# 7. 添加背景图片（可选）
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
        opacity=0.5,  # 调整透明度，让前景元素更加醒目
        layer='below'  # 将图片置于图形的下层
    )
)

# 8. 保存动画为HTML文件
fig.write_html('neuron_activity_animation_individual_threshold.html')
print("动画已保存为 'neuron_activity_animation_individual_threshold.html'")

# 9. 保存网络结构到 Excel
# 为了确保Excel文件的可读性和易用性，我们将所有节点和边信息分别保存到两个单独的工作表中，
# 并在每个表中添加时间戳列以区分不同时间点的数据。
# 同时，我们将行为标签作为图的标签保存到另一个工作表中。

# 创建空的DataFrame列表
nodes_list = []
edges_list = []
labels_list = []

for network in network_structures:
    time = network['time']
    behavior = network['behavior']
    nodes = network['nodes']
    edges = network['edges']

    # 创建节点DataFrame
    nodes_df = pd.DataFrame(nodes)
    nodes_df['time'] = time
    nodes_list.append(nodes_df)

    # 创建边DataFrame
    edges_df = pd.DataFrame(edges)
    edges_df['time'] = time
    edges_list.append(edges_df)

    # 创建标签DataFrame
    labels_df = pd.DataFrame([{'time': time, 'behavior': behavior}])
    labels_list.append(labels_df)

# 合并所有节点、边和标签数据
all_nodes_df = pd.concat(nodes_list, ignore_index=True)
all_edges_df = pd.concat(edges_list, ignore_index=True)
all_labels_df = pd.concat(labels_list, ignore_index=True)

# 将数据保存到Excel
excel_output_path = 'network_structures_for_GNN.xlsx'
with pd.ExcelWriter(excel_output_path) as writer:
    all_nodes_df.to_excel(writer, sheet_name='Nodes', index=False)
    all_edges_df.to_excel(writer, sheet_name='Edges', index=False)
    all_labels_df.to_excel(writer, sheet_name='Labels', index=False)

print(f"网络结构已保存为 '{excel_output_path}'")


