"""
This module analyzes and visualizes the temporal topology of neuron activity patterns.
It creates an animated visualization showing how neurons form groups based on their activity levels over time.
"""

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import itertools
from tqdm import tqdm
import warnings
import plotly.io as pio

# Suppress warnings
warnings.filterwarnings('ignore')

# Set default plotly template
pio.templates.default = "plotly"

def load_neuron_data(file_path):
    """
    Load neuron data from Excel file and identify neuron columns.
    
    Args:
        file_path (str): Path to the Excel file containing neuron data
        
    Returns:
        tuple: (DataFrame with neuron data, list of neuron column names)
    """
    data = pd.read_excel(file_path)
    neuron_cols = [col for col in data.columns if 'n' in col.lower()]
    
    if not neuron_cols:
        raise ValueError("No neuron columns found in the Excel file!")
    if 'behavior' not in data.columns:
        raise ValueError("Behavior column not found in the Excel file!")
    
    print(f"Found {len(neuron_cols)} neuron columns:", neuron_cols)
    return data, neuron_cols

def calculate_threshold(data, method='mean'):
    """
    Calculate activation threshold for neuron data.
    
    Args:
        data (DataFrame): Neuron activity data
        method (str): Method to calculate threshold ('mean', 'median', or 'percentile')
        
    Returns:
        float: Calculated threshold value
    """
    if method == 'mean':
        return data.mean().mean()
    elif method == 'median':
        return data.median().median()
    elif method == 'percentile':
        return data.quantile(0.75).mean()
    return data.mean().mean()  # Default to mean

def create_circular_layout(nodes):
    """
    Create circular layout positions for nodes.
    
    Args:
        nodes (list): List of node identifiers
        
    Returns:
        dict: Mapping of nodes to their positions
    """
    theta = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    return {node: (np.cos(theta[i]), np.sin(theta[i])) for i, node in enumerate(nodes)}

def process_frame_data(neuron_data, neuron_ids, threshold, pos):
    """
    Process and calculate frame data for the animation.
    
    Args:
        neuron_data (DataFrame): Input neuron activity data
        neuron_ids (list): List of neuron column names
        threshold (float): Activity threshold value
        pos (dict): Node positions in circular layout
        
    Returns:
        tuple: Lists containing frame data for visualization
    """
    frames_data = {
        'node_x': [], 'node_y': [], 'node_text': [], 'node_color': [],
        'edge_x': [], 'edge_y': [], 'edge_color': [], 'titles': [],
        'behaviors': []  # 新增行为标签存储
    }
    
    current_groups = {}  # group_id: list of neurons
    neuron_to_group = {}  # neuron_id: group_id
    group_colors = {}
    group_id_counter = itertools.count(1)
    color_cycle = itertools.cycle([
        'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink',
        'gray', 'olive', 'cyan', 'yellow', 'black', 'white'
    ])

    for num in tqdm(range(len(neuron_data)), desc="Processing frame data"):
        timestamp = neuron_data['stamp'].iloc[num]
        behavior = neuron_data['behavior'].iloc[num]  # 获取行为标签
        activity_values = neuron_data[neuron_ids].iloc[num]
        state = np.where(activity_values >= threshold, 'ON', 'OFF')
        
        # Update neuron groups
        inactive_neurons = [nid for nid, s in zip(neuron_ids, state) if s == 'OFF']
        active_neurons = [nid for nid, s in zip(neuron_ids, state) if s == 'ON']
        
        # Handle inactive neurons
        for neuron in inactive_neurons:
            if neuron in neuron_to_group:
                group_id = neuron_to_group[neuron]
                current_groups[group_id].remove(neuron)
                if not current_groups[group_id]:
                    del current_groups[group_id]
                    del group_colors[group_id]
                del neuron_to_group[neuron]
        
        # Handle active neurons
        ungrouped_active = [n for n in active_neurons if n not in neuron_to_group]
        if ungrouped_active:
            new_group_id = next(group_id_counter)
            current_groups[new_group_id] = ungrouped_active
            for neuron in ungrouped_active:
                neuron_to_group[neuron] = new_group_id
            group_colors[new_group_id] = next(color_cycle)
        
        # Create graph for current frame
        G = nx.Graph()
        G.add_nodes_from(neuron_ids)
        
        # Add edges within groups
        edges = []
        edge_colors = []
        for group_id, neurons in current_groups.items():
            if len(neurons) > 1:
                representative = neurons[0]
                group_edges = [(representative, n) for n in neurons[1:]]
                edges.extend(group_edges)
                edge_colors.extend([group_colors[group_id]] * len(group_edges))
        
        G.add_edges_from(edges)
        
        # Collect frame data
        frame_data = collect_frame_data(G, pos, neuron_to_group, group_colors, edges)
        for key, value in frame_data.items():
            frames_data[key].append(value)
        frames_data['titles'].append(f"Neuron Topology - Time: {timestamp}")
        frames_data['behaviors'].append(behavior)  # 存储行为标签
    
    return frames_data

def collect_frame_data(G, pos, neuron_to_group, group_colors, edges):
    """
    Collect visualization data for a single frame.
    
    Args:
        G (NetworkX Graph): Graph for current frame
        pos (dict): Node positions
        neuron_to_group (dict): Mapping of neurons to their groups
        group_colors (dict): Mapping of group IDs to colors
        edges (list): List of edges in the graph
        
    Returns:
        dict: Frame data for visualization
    """
    node_x, node_y, node_text, node_color = [], [], [], []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
        node_color.append(group_colors.get(neuron_to_group.get(node), 'lightgray'))
    
    edge_x, edge_y = [], []
    for edge in edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    return {
        'node_x': node_x, 'node_y': node_y,
        'node_text': node_text, 'node_color': node_color,
        'edge_x': edge_x, 'edge_y': edge_y
    }

def create_animation(frames_data, output_path):
    """创建并保存动画"""
    fig = go.Figure(
        data=create_frame_traces(frames_data, 0),
        layout=create_layout(frames_data['titles'][0], len(frames_data['node_x']), frames_data['behaviors'])
    )
    
    # 添加动画帧
    fig.frames = [
        go.Frame(
            data=create_frame_traces(frames_data, k),
            name=f"frame_{k}",
            layout=go.Layout(
                title=frames_data['titles'][k],
                annotations=[
                    dict(
                        x=0.02,
                        y=0.98,
                        xref='paper',
                        yref='paper',
                        text=f"Current Behavior: {frames_data['behaviors'][k]}",
                        showarrow=False,
                        font=dict(size=14),
                        xanchor='left',
                        yanchor='top'
                    )
                ]
            )
        )
        for k in range(len(frames_data['node_x']))
    ]
    
    # 修改动画控件配置
    fig.update_layout(
        updatemenus=[
            # 播放/暂停按钮
            {
                "buttons": [
                    {
                        "label": "播放",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ]
                    },
                    {
                        "label": "暂停",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ]
                    }
                ],
                "direction": "left",
                "pad": {"r": 10},
                "showactive": False,
                "type": "buttons",
                "x": 0.9,
                "y": 1.1,
                "xanchor": "right",
                "yanchor": "top"
            }
        ],
        # 添加速度控制滑块和时间轴滑块
        sliders=[
            # 时间轴滑块
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "时间点: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "pad": {"b": 50, "t": 50},
                "len": 1.0,
                "x": 0,
                "y": -0.1,
                "steps": [
                    {
                        "args": [
                            [f"frame_{k}"],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ],
                        "label": f"{k}" if k % 50 == 0 else "",
                        "method": "animate"
                    }
                    for k in range(len(frames_data['node_x']))
                ]
            },
            # 速度控制滑块
            {
                "active": 2,  # 默认选择正常速度
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "播放速度: ",
                    "suffix": "x",
                    "visible": True,
                    "xanchor": "right"
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.3,  # 滑块长度
                "x": 0.6,    # 位置
                "y": 1.1,    # 位置
                "steps": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": int(200/speed), "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ],
                        "label": str(speed),
                        "method": "animate"
                    }
                    for speed in [0.25, 0.5, 1, 2, 4, 8]  # 速度选项
                ]
            }
        ]
    )
    
    fig.write_html(output_path)

def create_layout(initial_title, frame_count, behaviors):
    """
    Create the layout configuration for the animation.
    
    Args:
        initial_title (str): Title for the first frame
        frame_count (int): Total number of frames
        behaviors (list): List of behaviors for each frame
        
    Returns:
        go.Layout: Layout configuration
    """
    # 创建行为区间标记
    behavior_shapes = []  # 用于存储形状（线条）
    behavior_annotations = []  # 用于存储标签
    current_behavior = behaviors[0]
    start_idx = 0
    
    # 计算行为区间
    for i in range(1, len(behaviors)):
        if behaviors[i] != current_behavior:
            # 添加竖线
            behavior_shapes.append({
                'type': 'line',
                'x0': i / len(behaviors),  # 归一化坐标
                'x1': i / len(behaviors),
                'y0': -0.1,
                'y1': -0.15,
                'xref': 'paper',
                'yref': 'paper',
                'line': {'color': 'black', 'width': 1}
            })
            
            # 添加行为标签
            behavior_annotations.append({
                'x': (start_idx + i) / (2 * len(behaviors)),  # 居中显示
                'y': -0.15,
                'xref': 'paper',
                'yref': 'paper',
                'text': current_behavior,
                'showarrow': False,
                'font': {'size': 12}
            })
            
            current_behavior = behaviors[i]
            start_idx = i
    
    # 添加最后一个行为区间的标签
    behavior_annotations.append({
        'x': (start_idx + len(behaviors)) / (2 * len(behaviors)),
        'y': -0.15,
        'xref': 'paper',
        'yref': 'paper',
        'text': current_behavior,
        'showarrow': False,
        'font': {'size': 12}
    })
    
    # 添加底部的水平线
    behavior_shapes.append({
        'type': 'line',
        'x0': 0,
        'x1': 1,
        'y0': -0.1,
        'y1': -0.1,
        'xref': 'paper',
        'yref': 'paper',
        'line': {'color': 'black', 'width': 1}
    })
    
    # 合并所有注释
    all_annotations = [
        # 左上角的当前行为标签
        dict(
            x=0.02,
            y=0.98,
            xref='paper',
            yref='paper',
            text=f"Current Behavior: {behaviors[0]}",
            showarrow=False,
            font=dict(size=14),
            xanchor='left',
            yanchor='top'
        )
    ]
    all_annotations.extend(behavior_annotations)
    
    return go.Layout(
        title=initial_title,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        margin=dict(b=100, t=150),  # 增加顶部边距，为速度控制按钮留出空间
        sliders=[dict(
            active=0,
            y=-0.2,  # 调整滑块位置，为行为标签留出空间
            steps=[dict(
                label=str(i),
                method="animate",
                args=[[f"frame_{i}"], {
                    "frame": {"duration": 500, "redraw": True},
                    "mode": "immediate"
                }]
            ) for i in range(frame_count)]
        )],
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[
                dict(
                    label='Play',
                    method='animate',
                    args=[None, dict(
                        frame=dict(duration=500, redraw=True),
                        fromcurrent=True
                    )]
                ),
                dict(
                    label='Pause',
                    method='animate',
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode='immediate'
                    )]
                )
            ]
        )],
        annotations=all_annotations,
        shapes=behavior_shapes  # 只包含线条形状
    )

def create_frame_traces(frames_data, k):
    """
    创建单个帧的图形数据
    
    Args:
        frames_data (dict): 所有帧的数据
        k (int): 当前帧的索引
        
    Returns:
        list: 包含边和节点的图形数据列表
    """
    return [
        # 边的trace
        go.Scatter(
            x=frames_data['edge_x'][k],
            y=frames_data['edge_y'][k],
            mode='lines',
            line=dict(color='black', width=2),
            hoverinfo='none'
        ),
        # 节点的trace
        go.Scatter(
            x=frames_data['node_x'][k],
            y=frames_data['node_y'][k],
            mode='markers+text',
            text=frames_data['node_text'][k],
            textposition='middle center',
            marker=dict(color=frames_data['node_color'][k], size=15),
            hoverinfo='text'
        )
    ]

def main():
    """Main function to run the neuron topology analysis."""
    # Load data
    data_path = '../datasets/Day6_with_behavior_labels_filled.xlsx'
    neuron_data, neuron_ids = load_neuron_data(data_path)
    
    # Calculate threshold and create layout
    threshold = calculate_threshold(neuron_data[neuron_ids], method='mean')
    pos = create_circular_layout(neuron_ids)
    
    # Process frame data
    frames_data = process_frame_data(neuron_data, neuron_ids, threshold, pos)
    
    # Create and save animation
    output_path = '../graph/Day6_Time_Topology.html'
    create_animation(frames_data, output_path)

if __name__ == "__main__":
    main()