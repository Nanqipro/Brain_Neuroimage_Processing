# # import pandas as pd
# # import numpy as np
# # import networkx as nx
# # import matplotlib.pyplot as plt
# # from matplotlib.animation import FuncAnimation, PillowWriter
# # import itertools
# # import os
# # import warnings
# # warnings.filterwarnings('ignore')
# #
# # # 设置全局字体
# # plt.rcParams['font.family'] = 'SimHei'  # 请确保系统中安装了 SimHei 字体，或者替换为您系统中的其他中文字体
# #
# # # 1. 数据读取
# # neuron_data = pd.read_excel(r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day6\calcium_data.xlsx')
# #
# # # 获取神经元ID列表
# # neuron_ids = neuron_data.columns[1:]
# #
# # # 2. 设置阈值
# # threshold = neuron_data[neuron_ids].mean().mean()
# #
# # # 3. 初始化变量
# # current_groups = {}  # 组ID: 神经元列表
# # group_id_counter = itertools.count(1)
# # neuron_to_group = {}  # 神经元ID: 组ID
# #
# # # 颜色映射，给每个组分配一个颜色
# # group_colors = {}
# #
# # # 定义颜色列表，用于分配给不同的组
# # color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
# # color_cycle = itertools.cycle(color_list)
# #
# # # 4. 创建神经元的初始位置
# # G_all = nx.Graph()
# # G_all.add_nodes_from(neuron_ids)
# # pos = nx.spring_layout(G_all, seed=42)
# #
# # # 设置保存路径
# # save_path = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\图像处理代码\图像结果\神经元拓扑结构图 动态'
# # if not os.path.exists(save_path):
# #     os.makedirs(save_path)
# #
# # # 5. 创建绘图函数
# # fig, ax = plt.subplots(figsize=(12, 8))
# #
# # def update(num):
# #     global current_groups, neuron_to_group, group_id_counter, group_colors
# #     ax.clear()
# #     t = neuron_data['Time'].iloc[num]
# #     activity_values = neuron_data[neuron_ids].iloc[num]
# #     # 标记神经元状态
# #     state = np.where(activity_values >= threshold, 'ON', 'OFF')
# #     state_df = pd.DataFrame({
# #         'neuron_id': neuron_ids,
# #         'activity_value': activity_values.values,
# #         'state': state
# #     })
# #
# #     # 更新神经元所属组
# #     # 处理未激活的神经元
# #     inactive_neurons = state_df[state_df['state'] == 'OFF']['neuron_id'].tolist()
# #     for neuron in inactive_neurons:
# #         if neuron in neuron_to_group:
# #             group_id = neuron_to_group[neuron]
# #             current_groups[group_id].remove(neuron)
# #             if len(current_groups[group_id]) == 0:
# #                 # 如果组为空，删除该组
# #                 del current_groups[group_id]
# #                 del group_colors[group_id]
# #             del neuron_to_group[neuron]
# #
# #     # 处理激活的神经元
# #     active_neurons = state_df[state_df['state'] == 'ON']['neuron_id'].tolist()
# #     # 找出未分配组的激活神经元
# #     ungrouped_active_neurons = [n for n in active_neurons if n not in neuron_to_group]
# #
# #     if ungrouped_active_neurons:
# #         # 为这些神经元创建一个新的组
# #         new_group_id = next(group_id_counter)
# #         current_groups[new_group_id] = ungrouped_active_neurons
# #         for neuron in ungrouped_active_neurons:
# #             neuron_to_group[neuron] = new_group_id
# #         # 分配一个颜色
# #         group_colors[new_group_id] = next(color_cycle)
# #
# #     # 构建当前时间点的图
# #     G = nx.Graph()
# #     G.add_nodes_from(neuron_ids)
# #
# #     # 添加组内的连线
# #     edges = []
# #     edge_colors = []
# #     for group_id, neurons in current_groups.items():
# #         # 在组内的神经元两两相连
# #         group_edges = list(itertools.combinations(neurons, 2))
# #         edges.extend(group_edges)
# #         edge_colors.extend([group_colors[group_id]] * len(group_edges))
# #
# #     G.add_edges_from(edges)
# #
# #     # 为节点添加状态属性
# #     state_dict = state_df.set_index('neuron_id')['state'].to_dict()
# #     nx.set_node_attributes(G, state_dict, 'state')
# #
# #     # 根据状态设置节点颜色
# #     node_color_map = {'ON': 'yellow', 'OFF': 'lightgray'}
# #     node_colors = [node_color_map[G.nodes[node]['state']] for node in G.nodes()]
# #
# #     # 绘制网络图
# #     nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, ax=ax)
# #     nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, ax=ax)
# #
# #     # 绘制节点标签，手动设置字体属性
# #     for node in G.nodes():
# #         x, y = pos[node]
# #         ax.text(x, y, str(node), fontsize=8, ha='center', va='center')
# #
# #     ax.set_title(f'神经元拓扑结构图 - 时间点：{t}')
# #     ax.axis('off')
# #
# # # 6. 创建并保存动画
# # ani = FuncAnimation(fig, update, frames=len(neuron_data), interval=1000, repeat=False)
# #
# # # 保存动画为 GIF 文件
# # ani.save(os.path.join(save_path, 'neuron_activity_animation.gif'), writer=PillowWriter(fps=1))
#
#
# import pandas as pd
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter
# from matplotlib.gridspec import GridSpec
# import itertools
# import os
# from tqdm import tqdm  # 用于添加程序运行进度条
# import warnings
# warnings.filterwarnings('ignore')
#
# # 设置全局字体
# plt.rcParams['font.family'] = 'SimHei'  # 请确保系统中安装了 SimHei 字体，或者替换为您系统中的其他中文字体
#
# # 1. 数据读取
# neuron_data = pd.read_excel(r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day6\calcium_data.xlsx')
#
# # 获取神经元ID列表
# neuron_ids = neuron_data.columns[1:]
#
# # 2. 设置阈值（为使用不同阈值留好入口）
# def calculate_threshold(data, method='mean'):
#     if method == 'mean':
#         return data.mean().mean()
#     elif method == 'median':
#         return data.median().median()
#     elif method == 'percentile':
#         return data.quantile(0.75).mean()
#     else:
#         # 默认返回均值
#         return data.mean().mean()
#
# # 调用阈值计算函数（您可以修改 method 参数为 'mean', 'median', 'percentile' 或其他自定义方法）
# threshold = calculate_threshold(neuron_data[neuron_ids], method='mean')
#
# # 3. 初始化变量
# current_groups = {}  # 组ID: 神经元列表
# group_id_counter = itertools.count(1)
# neuron_to_group = {}  # 神经元ID: 组ID
#
# # 颜色映射，给每个组分配一个颜色
# group_colors = {}
#
# # 定义颜色列表，用于分配给不同的组
# color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
# color_cycle = itertools.cycle(color_list)
#
# # 4. 创建神经元的环形位置，保持相对位置稳定
# def create_circular_layout(nodes):
#     theta = np.linspace(0, 2 * np.pi, len(nodes) + 1)
#     pos = {}
#     for idx, node in enumerate(nodes):
#         pos[node] = (np.cos(theta[idx]), np.sin(theta[idx]))
#     return pos
#
# pos = create_circular_layout(neuron_ids)
#
# # 设置保存路径
# save_path = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\图像处理代码\图像结果\神经元拓扑结构图 动态'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
#
# # 5. 创建绘图函数
# # 添加交互式拖动进度条
# from ipywidgets import widgets
# from IPython.display import display
# from tqdm import tqdm  # 进度条
#
# fig = plt.figure(figsize=(12, 8))
# gs = GridSpec(1, 2, width_ratios=[3, 1])  # 左右布局，右侧用于显示亮起的神经元列表
#
# def update(num):
#     global current_groups, neuron_to_group, group_colors
#     fig.clf()
#     ax = fig.add_subplot(gs[0])
#     ax_right = fig.add_subplot(gs[1])
#
#     t = neuron_data['Time'].iloc[num]
#     activity_values = neuron_data[neuron_ids].iloc[num]
#     # 标记神经元状态
#     state = np.where(activity_values >= threshold, 'ON', 'OFF')
#     state_df = pd.DataFrame({
#         'neuron_id': neuron_ids,
#         'activity_value': activity_values.values,
#         'state': state
#     })
#
#     # 更新神经元所属组
#     # 处理未激活的神经元
#     inactive_neurons = state_df[state_df['state'] == 'OFF']['neuron_id'].tolist()
#     for neuron in inactive_neurons:
#         if neuron in neuron_to_group:
#             group_id = neuron_to_group[neuron]
#             current_groups[group_id].remove(neuron)
#             if len(current_groups[group_id]) == 0:
#                 # 如果组为空，删除该组
#                 del current_groups[group_id]
#                 del group_colors[group_id]
#             del neuron_to_group[neuron]
#
#     # 处理激活的神经元
#     active_neurons = state_df[state_df['state'] == 'ON']['neuron_id'].tolist()
#     # 找出未分配组的激活神经元
#     ungrouped_active_neurons = [n for n in active_neurons if n not in neuron_to_group]
#
#     if ungrouped_active_neurons:
#         # 为这些神经元创建一个新的组
#         new_group_id = next(group_id_counter)
#         current_groups[new_group_id] = ungrouped_active_neurons
#         for neuron in ungrouped_active_neurons:
#             neuron_to_group[neuron] = new_group_id
#         # 分配一个颜色
#         group_colors[new_group_id] = next(color_cycle)
#
#     # 构建当前时间点的图
#     G = nx.Graph()
#     G.add_nodes_from(neuron_ids)
#
#     # 添加组内的连线
#     edges = []
#     edge_colors = []
#     for group_id, neurons in current_groups.items():
#         # 在组内的神经元两两相连
#         group_edges = list(itertools.combinations(neurons, 2))
#         edges.extend(group_edges)
#         edge_colors.extend([group_colors[group_id]] * len(group_edges))
#
#     G.add_edges_from(edges)
#
#     # 为节点添加状态属性和颜色
#     node_colors = []
#     for node in G.nodes():
#         if node in neuron_to_group:
#             # 神经元属于某个组，使用组的颜色
#             group_id = neuron_to_group[node]
#             node_colors.append(group_colors[group_id])
#         else:
#             # 神经元未激活，使用灰色
#             node_colors.append('lightgray')
#
#     # 绘制网络图
#     nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax)
#     nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, ax=ax)
#
#     # 绘制节点标签
#     for node in G.nodes():
#         x, y = pos[node]
#         ax.text(x, y, str(node), fontsize=8, ha='center', va='center')
#
#     ax.set_title(f'神经元拓扑结构图 - 时间点：{t}')
#     ax.axis('off')
#
#     # 绘制右侧亮起的神经元列表
#     ax_right.axis('off')
#     active_info = []
#     for group_id, neurons in current_groups.items():
#         color = group_colors[group_id]
#         for neuron in neurons:
#             active_info.append({'neuron': neuron, 'group': group_id, 'color': color, 'time': t})
#     if active_info:
#         active_df = pd.DataFrame(active_info)
#         ax_right.set_title('亮起的神经元', fontsize=12)
#         # 绘制文本列表
#         for idx, row in active_df.iterrows():
#             ax_right.text(0, 1 - idx * 0.05, f"时间: {row['time']}, 神经元: {row['neuron']}, 组: {row['group']}", color=row['color'], fontsize=8)
#     else:
#         ax_right.set_title('无亮起的神经元', fontsize=12)
#
#     # 保存当前帧的图像
#     fig.savefig(os.path.join(save_path, f'neuron_network_time_{t}.png'))
#
# # 添加程序运行进度条
# progress_bar = tqdm(total=len(neuron_data), desc='生成动画帧')
#
# # 创建动画
# def animate():
#     for num in range(len(neuron_data)):
#         update(num)
#         progress_bar.update(1)
#     progress_bar.close()
#
# animate()
#
# # 保存动画为 GIF 文件
# ani = FuncAnimation(fig, update, frames=len(neuron_data), interval=1000, repeat=False)
# ani.save(os.path.join(save_path, 'neuron_activity_animation.gif'), writer=PillowWriter(fps=1))
#
# # 增加可拖动的进度条用于交互
# from ipywidgets import interact, IntSlider
#
# def interactive_plot(num):
#     update(num)
#     display(fig)
#
# interact_slider = IntSlider(min=0, max=len(neuron_data)-1, step=1, value=0, description='时间点')
# interact(interactive_plot, num=interact_slider)


# import pandas as pd
# import numpy as np
# import networkx as nx
# import plotly.graph_objects as go
# import itertools
# from tqdm import tqdm  # 用于添加程序运行进度条
#
# # 1. 数据读取
# neuron_data = pd.read_excel(r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day6\calcium_data.xlsx')
#
# # 获取神经元ID列表
# neuron_ids = neuron_data.columns[1:]
#
# # 2. 设置阈值（可调整）
# def calculate_threshold(data, method='mean'):
#     if method == 'mean':
#         return data.mean().mean()
#     elif method == 'median':
#         return data.median().median()
#     elif method == 'percentile':
#         return data.quantile(0.75).mean()
#     else:
#         # 默认返回均值
#         return data.mean().mean()
#
# # 调用阈值计算函数（您可以修改 method 参数）
# threshold = calculate_threshold(neuron_data[neuron_ids], method='mean')
#
# # 3. 初始化变量
# current_groups = {}  # 组ID: 神经元列表
# group_id_counter = itertools.count(1)
# neuron_to_group = {}  # 神经元ID: 组ID
#
# # 颜色映射，给每个组分配一个颜色
# group_colors = {}
# color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
# color_cycle = itertools.cycle(color_list)
#
# # 4. 创建神经元的环形位置，保持相对位置稳定
# def create_circular_layout(nodes, scale=1.5):
#     theta = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
#     pos = {}
#     for idx, node in enumerate(nodes):
#         pos[node] = (scale * np.cos(theta[idx]), scale * np.sin(theta[idx]))
#     return pos
#
# pos = create_circular_layout(neuron_ids, scale=2)  # 您可以调整 scale 参数放大环形
#
# # 获取所有节点的x和y坐标，用于设置固定的坐标轴范围
# all_x_positions = [pos[node][0] for node in neuron_ids]
# all_y_positions = [pos[node][1] for node in neuron_ids]
# margin = 1  # 可根据需要调整
# x_range = [min(all_x_positions) - margin, max(all_x_positions) + margin]
# y_range = [min(all_y_positions) - margin, max(all_y_positions) + margin]
#
# # 5. 准备 Plotly 图形数据
# G = nx.Graph()
# G.add_nodes_from(neuron_ids)
#
# # 将节点位置转换为列表
# node_x = []
# node_y = []
# for node in G.nodes():
#     x, y = pos[node]
#     node_x.append(x)
#     node_y.append(y)
#
# # 准备帧数据
# frames = []
#
# # 初始化节点颜色和边数据
# node_colors_list = []
# edge_traces_list = []
#
# # 添加程序运行进度条
# total_steps = len(neuron_data)
# progress_bar = tqdm(total=total_steps, desc='生成动画帧')
#
# # 遍历每个时间点，生成帧
# for num in range(len(neuron_data)):
#     t = neuron_data['Time'].iloc[num]
#     activity_values = neuron_data[neuron_ids].iloc[num]
#
#     # 标记神经元状态
#     state = np.where(activity_values >= threshold, 'ON', 'OFF')
#     state_df = pd.DataFrame({
#         'neuron_id': neuron_ids,
#         'activity_value': activity_values.values,
#         'state': state
#     })
#
#     # 更新神经元所属组
#     # 处理未激活的神经元
#     inactive_neurons = state_df[state_df['state'] == 'OFF']['neuron_id'].tolist()
#     for neuron in inactive_neurons:
#         if neuron in neuron_to_group:
#             group_id = neuron_to_group[neuron]
#             current_groups[group_id].remove(neuron)
#             if len(current_groups[group_id]) == 0:
#                 # 如果组为空，删除该组
#                 del current_groups[group_id]
#                 del group_colors[group_id]
#             del neuron_to_group[neuron]
#
#     # 处理激活的神经元
#     active_neurons = state_df[state_df['state'] == 'ON']['neuron_id'].tolist()
#     # 找出未分配组的激活神经元
#     ungrouped_active_neurons = [n for n in active_neurons if n not in neuron_to_group]
#
#     if ungrouped_active_neurons:
#         # 为这些神经元创建一个新的组
#         new_group_id = next(group_id_counter)
#         current_groups[new_group_id] = ungrouped_active_neurons
#         for neuron in ungrouped_active_neurons:
#             neuron_to_group[neuron] = new_group_id
#         # 分配一个颜色
#         group_colors[new_group_id] = next(color_cycle)
#
#     # 更新节点颜色
#     node_colors = []
#     for node in G.nodes():
#         if node in neuron_to_group:
#             group_id = neuron_to_group[node]
#             node_colors.append(group_colors[group_id])
#         else:
#             node_colors.append('lightgray')
#     node_colors_list.append(node_colors)
#
#     # 更新边数据
#     edge_traces = []
#     for group_id, neurons in current_groups.items():
#         color = group_colors[group_id]
#         group_edges = list(itertools.combinations(neurons, 2))
#         if group_edges:
#             edge_x = []
#             edge_y = []
#             for edge in group_edges:
#                 x0, y0 = pos[edge[0]]
#                 x1, y1 = pos[edge[1]]
#                 edge_x.extend([x0, x1, None])
#                 edge_y.extend([y0, y1, None])
#             edge_trace = go.Scatter(
#                 x=edge_x,
#                 y=edge_y,
#                 mode='lines',
#                 line=dict(color=color, width=2),
#                 hoverinfo='none'
#             )
#             edge_traces.append(edge_trace)
#     edge_traces_list.append(edge_traces)
#
#     # 创建帧
#     frame_data = []
#     # 添加边
#     frame_data.extend(edge_traces)
#     # 添加节点
#     frame_data.append(go.Scatter(
#         x=node_x,
#         y=node_y,
#         mode='markers+text',
#         text=[str(node) for node in G.nodes()],
#         textposition='bottom center',
#         marker=dict(size=20, color=node_colors),
#         hoverinfo='text'
#     ))
#     # 创建帧
#     frames.append(go.Frame(
#         data=frame_data,
#         name=str(num),
#         layout=go.Layout(
#             title=f'神经元拓扑结构图 - 时间点：{t}',
#             xaxis=dict(range=x_range),  # 确保坐标轴范围固定
#             yaxis=dict(range=y_range)
#         )
#     ))
#
#     # 更新进度条
#     progress_bar.update(1)
#
# # 关闭进度条
# progress_bar.close()
#
# # 创建初始图形
# initial_edge_traces = edge_traces_list[0]
# initial_node_colors = node_colors_list[0]
#
# fig = go.Figure(
#     data=initial_edge_traces + [
#         go.Scatter(
#             x=node_x,
#             y=node_y,
#             mode='markers+text',
#             text=[str(node) for node in G.nodes()],
#             textposition='bottom center',
#             marker=dict(size=20, color=initial_node_colors),
#             hoverinfo='text'
#         )
#     ],
#     layout=go.Layout(
#         title=f'神经元拓扑结构图 - 时间点：{neuron_data["Time"].iloc[0]}',
#         xaxis=dict(
#             range=x_range,
#             showgrid=False,
#             zeroline=False,
#             showticklabels=False
#         ),
#         yaxis=dict(
#             range=y_range,
#             showgrid=False,
#             zeroline=False,
#             showticklabels=False
#         ),
#         hovermode='closest',
#         updatemenus=[dict(
#             type='buttons',
#             buttons=[dict(label='播放',
#                           method='animate',
#                           args=[None, {
#                               'frame': {'duration': 500, 'redraw': True},
#                               'fromcurrent': True,
#                               'transition': {'duration': 0}
#                           }])]
#         )],
#         sliders=[dict(
#             active=0,
#             currentvalue={'prefix': '时间点: '},
#             steps=[dict(method='animate',
#                         args=[[str(k)], {
#                             'frame': {'duration': 500, 'redraw': True},
#                             'mode': 'immediate',
#                             'transition': {'duration': 0}
#                         }],
#                         label=str(neuron_data['Time'].iloc[k]))
#                    for k in range(len(frames))]
#         )]
#     ),
#     frames=frames
# )
#
# # 显示图形
# fig.show()



import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import itertools
from tqdm import tqdm  # 用于添加程序运行进度条

# 1. 数据读取
neuron_data = pd.read_excel(r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day6\calcium_data.xlsx')

# 获取神经元ID列表
neuron_ids = neuron_data.columns[1:]

# 2. 设置队值（可调整）
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

# 调用队值计算函数（您可以修改 method 参数）
threshold = calculate_threshold(neuron_data[neuron_ids], method='mean')

# 3. 初始化变量
current_groups = {}  # 组ID: 神经元列表
group_id_counter = itertools.count(1)
neuron_to_group = {}  # 神经元ID: 组ID

# 颜色映射，给每个组分配一个颜色
group_colors = {}
color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
color_cycle = itertools.cycle(color_list)

# 4. 创建神经元的环形位置，保持相对位置稳定
def create_circular_layout(nodes, scale=2):
    theta = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos = {}
    for idx, node in enumerate(nodes):
        pos[node] = (scale * np.cos(theta[idx]), scale * np.sin(theta[idx]))
    return pos

pos = create_circular_layout(neuron_ids, scale=2)  # 您可以调整 scale 参数放大环形

# 获取所有节点的x和y坐标，用于设置固定的坐标轴范围
all_x_positions = [pos[node][0] for node in neuron_ids]
all_y_positions = [pos[node][1] for node in neuron_ids]
margin = 1  # 可根据需要调整
x_range = [min(all_x_positions) - margin, max(all_x_positions) + margin]
y_range = [min(all_y_positions) - margin, max(all_y_positions) + margin]

# 5. 准备 Plotly 图形数据
G = nx.Graph()
G.add_nodes_from(neuron_ids)

# 将节点位置转换为列表
node_x = []
node_y = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

# 准备帧数据
frames = []

# 添加程序运行进度条
total_steps = len(neuron_data)
progress_bar = tqdm(total=total_steps, desc='生成动画帧')

# 遍历每个时间点，生成帧
for num in range(len(neuron_data)):
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

    # 6. 更新节点颜色，确保每个节点都有颜色
    node_colors = ['lightgray'] * len(G.nodes())  # 默认颜色
    for node in neuron_to_group:
        group_id = neuron_to_group[node]
        if group_id in group_colors:
            node_colors[list(G.nodes()).index(node)] = group_colors[group_id]

    # 更新边数据
    edge_traces = []
    for group_id, neurons in current_groups.items():
        color = group_colors[group_id]
        if len(neurons) > 1:
            # 从组内的一个点出发，连接到其他点
            central_neuron = neurons[0]  # 选择第一个神经元作为中心点
            edges = [(central_neuron, neuron) for neuron in neurons[1:]]
            edge_x = []
            edge_y = []
            for edge in edges:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(color=color, width=2),
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)

    # 如果没有边数据，添加一个空的边 trace，防止数据不一致
    if not edge_traces:
        edge_traces.append(go.Scatter(
            x=[],
            y=[],
            mode='lines',
            line=dict(color='black', width=2),
            hoverinfo='none'
        ))

    # 创建帧
    frame_data = edge_traces + [go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=[str(node) for node in G.nodes()],  # 只保留数字作为标签
        textposition='bottom center',
        marker=dict(size=20, color=node_colors),
        hoverinfo='text'
    )]

    frames.append(go.Frame(
        data=frame_data,
        name=str(num)
    ))

    # 更新进度条
    progress_bar.update(1)

# 关闭进度条
progress_bar.close()

# 更新每个帧的标题
for i, frame in enumerate(frames):
    frame.layout = go.Layout(
        title=f'神经元拓扑结构图 - 时间点：{neuron_data["Time"].iloc[i]}'
    )

# 创建初始图形
initial_frame = frames[0]

fig = go.Figure(
    data=initial_frame.data,
    layout=go.Layout(
        title=f'神经元拓扑结构图 - 时间点：{neuron_data["Time"].iloc[0]}',
        xaxis=dict(
            range=x_range,
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=y_range,
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        hovermode='closest',
        updatemenus=[dict(
            type='buttons',
            buttons=[dict(label='播放',
                          method='animate',
                          args=[None, {
                              'frame': {'duration': 500, 'redraw': True},
                              'fromcurrent': True,
                              'transition': {'duration': 0}
                          }])]
        )],
        sliders=[dict(
            active=0,
            currentvalue={'prefix': '时间点: '},
            steps=[dict(method='animate',
                        args=[[str(k)], {
                            'frame': {'duration': 500, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        label=str(neuron_data['Time'].iloc[k]))
                   for k in range(len(frames))]
        )]
    ),
    frames=frames
)

# 显示图形
fig.show()