"""
This script analyzes and visualizes the topological structure of neuron activity over time.
It creates an interactive visualization showing how neurons form groups based on their activity patterns.

可自定义的参数说明：
1. 背景图参数：
   - use_background: 是否使用背景图
   - background_opacity: 背景图透明度 (0-1)
2. 节点（神经元）参数：
   - node_size: 节点大小
   - node_text_position: 节点文本位置
3. 边（连接）参数：
   - edge_width: 边的宽度
   - edge_color: 边的颜色
4. 颜色方案：
   - color_scheme: 节点颜色方案 ('tab20', 'Set1', 'Set2' 等)
   - max_groups: 最大组数
5. 动画参数：
   - frame_duration: 每帧持续时间（毫秒）
"""

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import itertools
from tqdm import tqdm
import warnings
import matplotlib.cm as cm
import matplotlib
from typing import Dict, List, Tuple, Iterator, Optional
import plotly.io as pio
import base64
from PIL import Image
import os
from pathlib import Path
import io

# Suppress warnings
warnings.filterwarnings('ignore')

# Set default plotly template
pio.templates.default = "plotly"

class NeuronTopologyAnalyzer:
    """
    A class to analyze and visualize the topological structure of neuron activity.
    支持自定义多个可视化参数，包括背景图、节点、边的显示效果，以及动画参数。
    """
    
    def __init__(self, 
                 neuron_data_path: str, 
                 position_data_path: str, 
                 background_image_path: Optional[str] = None,
                 use_background: bool = True,
                 node_size: int = 15,
                 node_text_position: str = 'middle center',
                 edge_width: int = 2,
                 edge_color: str = 'black',
                 background_opacity: float = 1.0,
                 frame_duration: int = 1000,
                 color_scheme: str = 'tab20',
                 max_groups: int = 100):
        """
        Initialize the analyzer with data paths and visualization parameters.
        
        Args:
            neuron_data_path (str): Path to the Excel file containing neuron activity data
            position_data_path (str): Path to the CSV file containing neuron positions
            background_image_path (Optional[str]): Path to the background image file
            use_background (bool): Whether to use background image in visualization
            node_size (int): Size of the nodes in the visualization
            node_text_position (str): Position of node labels ('middle center', 'top center', etc.)
            edge_width (int): Width of the edges connecting nodes
            edge_color (str): Color of the edges
            background_opacity (float): Opacity of the background image (0-1)
            frame_duration (int): Duration of each frame in the animation (milliseconds)
            color_scheme (str): Color scheme for node groups ('tab20', 'Set1', 'Set2', etc.)
            max_groups (int): Maximum number of groups for color cycling
        """
        # 保存数据路径
        self.neuron_data = pd.read_excel(neuron_data_path)
        
        # 验证behavior列是否存在
        if 'behavior' not in self.neuron_data.columns:
            raise ValueError("Behavior column not found in the Excel file!")
        
        self.positions_data = pd.read_csv(position_data_path)
        self.background_image_path = background_image_path
        
        # 保存可视化参数
        self.use_background = use_background
        self.node_size = node_size
        self.node_text_position = node_text_position
        self.edge_width = edge_width
        self.edge_color = edge_color
        self.background_opacity = background_opacity
        self.frame_duration = frame_duration
        self.color_scheme = color_scheme
        self.max_groups = max_groups
        
        # Initialize neuron IDs and positions
        self.neuron_ids = self._get_neuron_ids()
        self.pos = self._get_positions()
        
        # Validate data
        self._validate_data()
        
        # Calculate thresholds
        self.threshold_dict = self._calculate_thresholds()
        
        # Initialize group tracking
        self.group_id_counter = itertools.count(1)
        self.current_groups: Dict[int, List[str]] = {}  # group_id: neuron_list
        self.neuron_to_group: Dict[str, int] = {}  # neuron_id: group_id
        self.group_colors = {}
        
        # Initialize color scheme
        self.color_cycle = self._initialize_colors()
        
        # Initialize frame storage
        self.frames_data = {
            'node_x': [], 'node_y': [], 'node_text': [],
            'node_color': [], 'edge_x': [], 'edge_y': [],
            'edge_color': [], 'titles': [], 'behaviors': []
        }
        
        # Load background image if specified
        self.background_image = None
        self.background_image_size = None
        if self.use_background and self.background_image_path:
            self._load_background_image()
        
    def _load_background_image(self) -> None:
        """Load and prepare background image for visualization."""
        try:
            print(f"Loading background image from: {self.background_image_path}")
            img = Image.open(self.background_image_path)
            self.background_image_size = img.size
            print(f"Original image size: {self.background_image_size}")
            
            # 修改图像编码方式
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            encoded_image = base64.b64encode(img_byte_arr).decode()
            self.background_image = encoded_image
            print("Successfully encoded background image")
        except Exception as e:
            print(f"Warning: Failed to load background image: {e}")
            self.use_background = False
        
    def _get_neuron_ids(self) -> List[str]:
        """Extract neuron IDs from the data columns."""
        neuron_ids = [col for col in self.neuron_data.columns if 'n' in col.lower()]
        if not neuron_ids:
            raise ValueError("No neuron columns found in the Excel file!")
        print(f"Found {len(neuron_ids)} neuron columns:", neuron_ids)
        return neuron_ids
        
    def _get_positions(self) -> Dict[str, Tuple[float, float]]:
        """Convert position data to dictionary format."""
        pos = {}
        for nid, (rx, ry) in zip(self.neuron_ids, 
                                self.positions_data[['relative_x', 'relative_y']].values):
            pos[nid] = (rx, ry)
        return pos
        
    def _validate_data(self) -> None:
        """Validate that the number of positions matches the number of neurons."""
        if len(self.positions_data) != len(self.neuron_ids):
            raise ValueError("标记点数量与神经元数量不一致，请检查数据。")
            
    def _calculate_thresholds(self) -> Dict[str, float]:
        """Calculate activity thresholds for each neuron."""
        return self.neuron_data[self.neuron_ids].mean().to_dict()
        
    def _initialize_colors(self) -> Iterator[str]:
        """
        Initialize color scheme for neuron groups.
        使用指定的颜色方案和最大组数生成颜色循环器。
        """
        cmap = cm.get_cmap(self.color_scheme, self.max_groups)
        color_list = [matplotlib.colors.to_hex(cmap(i)) for i in range(self.max_groups)]
        return itertools.cycle(color_list)
        
    def _process_frame(self, frame_num: int) -> None:
        """
        Process a single frame of neuron activity data.
        
        Args:
            frame_num (int): The frame number to process
        """
        # Get timestamp and activity values
        t = self.neuron_data['stamp'].iloc[frame_num]
        behavior = self.neuron_data['behavior'].iloc[frame_num]  # 获取行为标签
        activity_values = self.neuron_data[self.neuron_ids].iloc[frame_num]
        
        # Determine neuron states
        state_df = self._get_neuron_states(activity_values)
        
        # Update neuron groups
        self._update_neuron_groups(state_df)
        
        # Create and store frame visualization data
        self._create_frame_visualization(t, behavior)
        
    def _get_neuron_states(self, activity_values: pd.Series) -> pd.DataFrame:
        """
        Determine the state (ON/OFF) of each neuron based on activity values.
        
        Args:
            activity_values (pd.Series): Activity values for each neuron
            
        Returns:
            pd.DataFrame: DataFrame containing neuron states
        """
        state = ['ON' if val >= self.threshold_dict[nid] else 'OFF'
                for nid, val in zip(self.neuron_ids, activity_values)]
                
        return pd.DataFrame({
            'neuron_id': self.neuron_ids,
            'activity_value': activity_values.values,
            'state': state
        })
        
    def _update_neuron_groups(self, state_df: pd.DataFrame) -> None:
        """
        Update neuron group assignments based on their current states.
        
        Args:
            state_df (pd.DataFrame): DataFrame containing neuron states
        """
        # Handle inactive neurons
        inactive_neurons = state_df[state_df['state'] == 'OFF']['neuron_id'].tolist()
        for neuron in inactive_neurons:
            if neuron in self.neuron_to_group:
                group_id = self.neuron_to_group[neuron]
                self.current_groups[group_id].remove(neuron)
                if len(self.current_groups[group_id]) == 0:
                    del self.current_groups[group_id]
                    del self.group_colors[group_id]
                del self.neuron_to_group[neuron]
        
        # Handle active neurons
        active_neurons = state_df[state_df['state'] == 'ON']['neuron_id'].tolist()
        ungrouped_active_neurons = [n for n in active_neurons if n not in self.neuron_to_group]
        
        if ungrouped_active_neurons:
            new_group_id = next(self.group_id_counter)
            self.current_groups[new_group_id] = ungrouped_active_neurons
            for neuron in ungrouped_active_neurons:
                self.neuron_to_group[neuron] = new_group_id
            self.group_colors[new_group_id] = next(self.color_cycle)
            
    def _create_frame_visualization(self, timestamp: float, behavior: str) -> None:
        """
        Create visualization data for the current frame.
        
        Args:
            timestamp (float): Current frame timestamp
            behavior (str): Current behavior label
        """
        G = nx.Graph()
        G.add_nodes_from(self.neuron_ids)
        
        # Create edges within groups
        edges = []
        edge_colors = []
        for group_id, neurons in self.current_groups.items():
            if len(neurons) > 1:
                representative_node = neurons[0]
                group_edges = [(representative_node, n) for n in neurons[1:]]
                edges.extend(group_edges)
                edge_colors.extend([self.group_colors[group_id]] * len(group_edges))
        
        G.add_edges_from(edges)
        
        # Store node information
        node_x, node_y, node_text, node_color = [], [], [], []
        for node in G.nodes():
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
            node_color.append(self.group_colors.get(self.neuron_to_group.get(node), 'lightgray'))
        
        # Store edge information
        edge_x, edge_y = [], []
        for edge in edges:
            x0, y0 = self.pos[edge[0]]
            x1, y1 = self.pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Store frame data
        self.frames_data['node_x'].append(node_x)
        self.frames_data['node_y'].append(node_y)
        self.frames_data['node_text'].append(node_text)
        self.frames_data['node_color'].append(node_color)
        self.frames_data['edge_x'].append(edge_x)
        self.frames_data['edge_y'].append(edge_y)
        self.frames_data['edge_color'].append(edge_colors)
        self.frames_data['titles'].append(f"神经元拓扑结构图 - 时间点：{timestamp}")
        self.frames_data['behaviors'].append(behavior)  # 存储行为标签
        
    def process_all_frames(self) -> None:
        """Process all frames in the neuron data."""
        for frame_num in tqdm(range(len(self.neuron_data)), desc="预计算帧数据"):
            self._process_frame(frame_num)
            
    def create_animation(self, output_path: str) -> None:
        """
        Create and save the interactive animation.
        
        Args:
            output_path (str): Path to save the HTML animation
        """
        # Create base figure
        fig = self._create_base_figure()
        
        # Add frames
        fig.frames = self._create_animation_frames()
        
        # Save animation
        fig.write_html(output_path)
        print(f"Animation saved to {output_path}")
        
    def _create_base_figure(self) -> go.Figure:
        """Create the base figure for the animation."""
        return go.Figure(
            data=[
                # 创建边的散点图
                go.Scatter(
                    x=self.frames_data['edge_x'][0],
                    y=self.frames_data['edge_y'][0],
                    mode='lines',
                    line=dict(
                        color=self.edge_color,  # 使用自定义边颜色
                        width=self.edge_width   # 使用自定义边宽度
                    ),
                    hoverinfo='none'
                ),
                # 创建节点的散点图
                go.Scatter(
                    x=self.frames_data['node_x'][0],
                    y=self.frames_data['node_y'][0],
                    mode='markers+text',
                    text=self.frames_data['node_text'][0],
                    textposition=self.node_text_position,  # 使用自定义文本位置
                    marker=dict(
                        color=self.frames_data['node_color'][0], 
                        size=self.node_size  # 使用自定义节点大小
                    ),
                    hoverinfo='text'
                )
            ],
            layout=self._create_layout()
        )
        
    def _create_layout(self) -> go.Layout:
        """
        Create the layout for the animation.
        包含了背景图、坐标轴、动画控制等设置。
        """
        # 设置基本布局配置
        layout_config = {
            'title': self.frames_data['titles'][0],
            'showlegend': False,
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'margin': dict(l=10, r=10, t=30, b=60),  # 进一步减小左右边距
            'width': 1200,
            'height': 800
        }

        # 如果有背景图，根据背景图设置画布大小和比例
        if self.use_background and self.background_image and self.background_image_size:
            # 获取图像尺寸
            img_width, img_height = self.background_image_size
            
            # 设置画布大小（保持宽度1200，高度根据背景图比例调整）
            layout_config['width'] = 1200
            layout_config['height'] = int(1200 * img_height / img_width) + 60  # 为标题和时间轴留出空间
            
            # 计算坐标轴范围
            layout_config['xaxis'] = dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.05, 1.05],
                domain=[0.05, 0.95],  # 调整domain，让图形在画布中居中
                scaleanchor='y',
                scaleratio=1
            )
            
            layout_config['yaxis'] = dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.05, 1.05],
                domain=[0.05, 0.95],  # 调整domain，让图形在画布中居中
                autorange='reversed'
            )
            
            # 添加背景图
            layout_config['images'] = [{
                'source': f'data:image/png;base64,{self.background_image}',
                'xref': 'paper',
                'yref': 'paper',
                'x': 0,
                'y': 1,
                'sizex': 1,
                'sizey': 1,
                'sizing': 'stretch',
                'opacity': self.background_opacity,
                'layer': 'below'
            }]
        else:
            # 如果没有背景图，使用默认设置
            layout_config.update({
                'xaxis': dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    range=[-0.05, 1.05],
                    domain=[0.05, 0.95],  # 调整domain，让图形在画布中居中
                    scaleanchor='y',
                    scaleratio=1
                ),
                'yaxis': dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    range=[-0.05, 1.05],
                    domain=[0.05, 0.95],  # 调整domain，让图形在画布中居中
                    autorange='reversed'
                )
            })

        # 调整滑块和按钮位置
        layout_config['updatemenus'] = [{
            'type': 'buttons',
            'showactive': False,
            'x': 0.1,
            'y': 0.9,
            'xanchor': 'left',
            'buttons': [
                dict(
                    label='Play',
                    method='animate',
                    args=[None, dict(frame=dict(duration=self.frame_duration, redraw=True),
                                   fromcurrent=True)]
                ),
                dict(
                    label='Pause',
                    method='animate',
                    args=[[None], dict(frame=dict(duration=0, redraw=False),
                                     mode='immediate')]
                )
            ]
        }]

        # 调整滑块位置和宽度
        layout_config['sliders'] = [{
            'active': 0,
            'y': -0.2,  # 与Time_topology.py完全一致
            'xanchor': 'left',
            'x': 0,     # 从画面最左端开始
            'len': 1,   # 占据100%宽度
            'steps': [{
                'label': str(i),
                'method': "animate",
                'args': [[f"frame_{i}"], {
                    "frame": {"duration": self.frame_duration, "redraw": True},
                    "mode": "immediate"
                }]
            } for i in range(len(self.frames_data['node_x']))]
        }]

        # 行为标签和时间轴配置（与Time_topology.py完全一致）
        behavior_shapes = []
        behavior_annotations = []
        current_behavior = self.frames_data['behaviors'][0]
        start_idx = 0
        label_alternate = True  # 用于交替显示标签位置
        
        # 计算行为区间
        for i in range(1, len(self.frames_data['behaviors'])):
            if self.frames_data['behaviors'][i] != current_behavior:
                # 添加竖线
                behavior_shapes.append({
                    'type': 'line',
                    'x0': i/len(self.frames_data['behaviors']),
                    'x1': i/len(self.frames_data['behaviors']),
                    'y0': -0.1,
                    'y1': -0.15 if label_alternate else -0.05,  # 根据交替标志调整竖线长度
                    'xref': 'paper',
                    'yref': 'paper',
                    'line': {'color': 'black', 'width': 1}
                })
                
                # 添加行为标签（交替显示在上下方）
                behavior_annotations.append({
                    'x': (start_idx + i)/(2*len(self.frames_data['behaviors'])),
                    'y': -0.17 if label_alternate else -0.03,  # 交替位置
                    'xref': 'paper',
                    'yref': 'paper',
                    'text': current_behavior,
                    'showarrow': False,
                    'font': {'size': 12},
                    'yanchor': 'top' if label_alternate else 'bottom'  # 调整锚点
                })
                
                current_behavior = self.frames_data['behaviors'][i]
                start_idx = i
                label_alternate = not label_alternate  # 切换交替标志
        
        # 添加最后一个区间的标签
        behavior_annotations.append({
            'x': (start_idx + len(self.frames_data['behaviors']))/(2*len(self.frames_data['behaviors'])),
            'y': -0.17 if label_alternate else -0.03,
            'xref': 'paper',
            'yref': 'paper',
            'text': current_behavior,
            'showarrow': False,
            'font': {'size': 12},
            'yanchor': 'top' if label_alternate else 'bottom'
        })
        
        # 添加底部的水平线（保持原有位置）
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

        # 添加当前行为标签和行为区间标签
        all_annotations = [
            # 左上角的当前行为标签
            dict(
                x=0.02,
                y=0.98,
                xref='paper',
                yref='paper',
                text=f"Current Behavior: {self.frames_data['behaviors'][0]}",
                showarrow=False,
                font=dict(size=14),
                xanchor='left',
                yanchor='top'
            )
        ]
        all_annotations.extend(behavior_annotations)
        layout_config['annotations'] = all_annotations
        layout_config['shapes'] = behavior_shapes
        
        return go.Layout(**layout_config)
        
    def _create_animation_frames(self) -> List[go.Frame]:
        """Create frames for the animation with custom parameters."""
        return [
            go.Frame(
                data=[
                    go.Scatter(
                        x=self.frames_data['edge_x'][k],
                        y=self.frames_data['edge_y'][k],
                        mode='lines',
                        line=dict(color=self.edge_color, width=self.edge_width),
                        hoverinfo='none'
                    ),
                    go.Scatter(
                        x=self.frames_data['node_x'][k],
                        y=self.frames_data['node_y'][k],
                        mode='markers+text',
                        text=self.frames_data['node_text'][k],
                        textposition=self.node_text_position,
                        marker=dict(color=self.frames_data['node_color'][k], size=self.node_size),
                        hoverinfo='text'
                    )
                ],
                name=f"frame_{k}",
                layout=go.Layout(
                    title=self.frames_data['titles'][k],
                    annotations=[
                        dict(
                            x=0.02,
                            y=0.98,
                            xref='paper',
                            yref='paper',
                            text=f"Current Behavior: {self.frames_data['behaviors'][k]}",
                            showarrow=False,
                            font=dict(size=14),
                            xanchor='left',
                            yanchor='top'
                        )
                    ]
                )
            )
            for k in range(len(self.frames_data['node_x']))
        ]

def main():
    """Main function to run the topology analysis."""
    # Define file paths
    base_dir = Path(__file__).parent.parent
    neuron_data_path = base_dir / 'datasets/Day6_with_behavior_labels_filled.xlsx'
    position_data_path = base_dir / 'datasets/Day6_Max_position.csv'
    background_image_path = base_dir / 'datasets/Day6_Max.png'
    output_path = base_dir / 'graph/Day6_pos_topology.html'
    
    # 检查文件是否存在
    print(f"Checking if background image exists: {background_image_path}")
    if not background_image_path.exists():
        print(f"Warning: Background image not found at {background_image_path}")
    else:
        print(f"Background image found at {background_image_path}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建分析器实例（可以自定义参数）
    analyzer = NeuronTopologyAnalyzer(
        str(neuron_data_path),
        str(position_data_path),
        str(background_image_path),
        use_background=False,  # 是否使用背景图
        node_size=15,         # 节点大小
        node_text_position='middle center',  # 节点文本位置
        edge_width=2,         # 边的宽度
        edge_color='black',   # 边的颜色
        background_opacity=0.8,  # 背景图透明度
        frame_duration=1000,  # 帧持续时间（毫秒）
        color_scheme='tab20',  # 颜色方案
        max_groups=100        # 最大组数
    )
    analyzer.process_all_frames()
    analyzer.create_animation(str(output_path))

if __name__ == '__main__':
    main()