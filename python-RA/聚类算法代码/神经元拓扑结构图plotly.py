import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import itertools
from tqdm import tqdm  # 用于添加程序运行进度条
import warnings

warnings.filterwarnings('ignore')

# 设置字体
import plotly.io as pio

pio.templates.default = ("plotly")

# 1. 数据读取
neuron_data = pd.read_excel(r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day3\Day3.xlsx')  # 请修改为您的数据路径

# 获取神经元ID列表
neuron_ids = neuron_data.columns[1:]

# 2. 设置阈值（为使用不同阈值留好入口）
def calculate_threshold(data, method='mean'):
    if method == 'mean':
        return data.mean().mean()
    elif method == 'median':
        return data.median().median()
    elif method == 'percentile':
        return data.quantile(0.75).mean()
    else:
        # 默认返回均值
        return data.mean().mean()

# 调用阈值计算函数
threshold = calculate_threshold(neuron_data[neuron_ids], method='mean')

# 3. 初始化变量
group_id_counter = itertools.count(1)

# 颜色映射，给每个组分配一个颜色
color_list = [
    'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
    'yellow', 'black', 'white']
color_cycle = itertools.cycle(color_list)

# 4. 创建神经元的环形位置，保持相对位置稳定
def create_circular_layout(nodes):
    theta = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos = {}
    for idx, node in enumerate(nodes):
        pos[node] = (np.cos(theta[idx]), np.sin(theta[idx]))
    return pos

pos = create_circular_layout(neuron_ids)

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

# 遍历每个时间点，预先计算数据
for num in tqdm(range(len(neuron_data)), desc="预计算帧数据"):
    t = neuron_data['Time'].iloc[num]
    activity_values = neuron_data[neuron_ids].iloc[num]
    # 标记神经元状态
    state = np.where(activity_values >= threshold, 'ON', 'OFF')
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
        representative_node = neurons[0]  # 选择组内的第一个节点作为代表
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
            # 神经元属于某个组，使用组的颜色
            group_id = neuron_to_group[node]
            node_color.append(group_colors[group_id])
        else:
            # 神经元未激活，使用灰色
            node_color.append('lightgray')

    # 边的坐标
    edge_x = []
    edge_y = []
    for idx, edge in enumerate(edges):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # 存储每一帧的数据
    frame_node_x.append(node_x)
    frame_node_y.append(node_y)
    frame_node_text.append(node_text)
    frame_node_color.append(node_color)
    frame_edge_x.append(edge_x)
    frame_edge_y.append(edge_y)
    frame_edge_color.append(edge_colors)  # 边的颜色列表
    frame_titles.append(f"神经元拓扑结构图 - 时间点：{t}")

# 6. 创建动画
# 初始化图形
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
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',  # 设置白色背景
        sliders=[dict(
            active=0,
            steps=[dict(
                label=str(i),
                method="animate",
                args=[[f"frame_{i}"], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}],
            ) for i in range(len(frame_node_x))]
        )],
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(
                label='Play',
                method='animate',
                args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)]
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

# 7. 显示或保存动画
# 可以将动画保存为 HTML 文件
fig.write_html('neuron_activity_animation_mean_Day3.html')



