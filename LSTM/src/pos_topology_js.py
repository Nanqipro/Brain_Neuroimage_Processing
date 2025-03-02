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
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Iterator, Optional, Set
import plotly.io as pio
import base64
from PIL import Image
import os
from pathlib import Path
import io
import imageio
import json
from analysis_config import AnalysisConfig  # 导入配置类

# Suppress warnings
warnings.filterwarnings('ignore')

# Set default plotly template
pio.templates.default = "plotly"

class NeuronTopologyAnalyzer:
    """
    A class to analyze and visualize the topological structure of neuron activity.
    支持自定义多个可视化参数，包括背景图、节点、边的显示效果，以及动画参数。
    利用network_analysis_results.json中的实际神经元连接数据创建拓扑结构。
    """
    
    def __init__(self, 
                 config: AnalysisConfig,  # 使用配置类替代单独的参数
                 use_background: Optional[bool] = None,
                 node_size: Optional[int] = None,
                 node_text_position: Optional[str] = None,
                 edge_width: Optional[int] = None,
                 edge_color: Optional[str] = None,
                 background_opacity: Optional[float] = None,
                 frame_duration: Optional[int] = None,
                 color_scheme: Optional[str] = None,
                 max_groups: Optional[int] = None,
                 edge_weight_threshold: Optional[float] = None):
        """
        Initialize the analyzer with configuration and optional override parameters.
        
        Args:
            config (AnalysisConfig): Configuration object containing paths and parameters
            use_background (Optional[bool]): Whether to use background image in visualization
            node_size (Optional[int]): Size of the nodes in the visualization
            node_text_position (Optional[str]): Position of node labels ('middle center', 'top center', etc.)
            edge_width (Optional[int]): Width of the edges connecting nodes
            edge_color (Optional[str]): Color of the edges
            background_opacity (Optional[float]): Opacity of the background image (0-1)
            frame_duration (Optional[int]): Duration of each frame in the animation (milliseconds)
            color_scheme (Optional[str]): Color scheme for node groups ('tab20', 'Set1', 'Set2', etc.)
            max_groups (Optional[int]): Maximum number of groups for color cycling
            edge_weight_threshold (Optional[float]): Threshold for edge weight to create a connection
        """
        # 保存配置
        self.config = config
        
        # 保存数据路径
        self.neuron_data = pd.read_excel(config.data_file)
        
        # 验证behavior列是否存在
        if 'behavior' not in self.neuron_data.columns:
            raise ValueError("Behavior column not found in the Excel file!")
        
        self.positions_data = pd.read_csv(config.position_data_file)
        self.background_image_path = config.background_image
        
        # 获取拓扑可视化参数（优先使用传入的参数，如果未传入则使用配置中的默认值）
        topo_params = config.visualization_params['topology']
        self.use_background = use_background if use_background is not None else topo_params['use_background']
        self.node_size = node_size if node_size is not None else topo_params['node_size']
        self.node_text_position = node_text_position if node_text_position is not None else topo_params['node_text_position']
        self.edge_width = edge_width if edge_width is not None else topo_params['edge_width']
        self.edge_color = edge_color if edge_color is not None else topo_params['edge_color']
        self.background_opacity = background_opacity if background_opacity is not None else topo_params['background_opacity']
        self.frame_duration = frame_duration if frame_duration is not None else topo_params['frame_duration']
        self.color_scheme = color_scheme if color_scheme is not None else topo_params['color_scheme']
        self.max_groups = max_groups if max_groups is not None else topo_params['max_groups']
        
        # 边权重阈值 - 只有权重大于此值的边才会被显示
        self.edge_weight_threshold = edge_weight_threshold if edge_weight_threshold is not None else 0.3
        
        # 加载网络分析结果和神经元效应大小数据
        self.network_data = self._load_network_data(config.network_analysis_file)
        self.effect_sizes = self._load_effect_sizes(config.neuron_effect_file)
        
        # 从网络数据中提取边信息
        self.edges = self._extract_edge_data()
        
        # Initialize neuron IDs and positions
        self.neuron_ids = self._get_neuron_ids()
        self.pos = self._get_positions()
        
        # Validate data
        self._validate_data()
        
        # Calculate thresholds
        self.threshold_dict = self._calculate_thresholds()
        
        # 初始化连接图
        self.G = nx.Graph()
        for neuron_id in self.neuron_ids:
            self.G.add_node(neuron_id)
        
        # 添加边（使用网络分析中的边数据）
        for edge in self.edges:
            source, target, weight = edge
            # 如果源和目标都在神经元ID列表中且权重大于阈值，则添加边
            if source in self.neuron_ids and target in self.neuron_ids and weight > self.edge_weight_threshold:
                self.G.add_edge(source, target, weight=weight)
                
        # Initialize group tracking
        self.group_id_counter = itertools.count(1)
        self.current_groups: Dict[int, List[str]] = {}  # group_id: neuron_list
        self.neuron_to_group: Dict[str, int] = {}  # neuron_id: group_id
        self.group_colors = {}
        
        # Initialize color scheme
        self.color_cycle = self._initialize_colors()
        
        # 神经元状态: 1表示活跃, 0表示不活跃
        self.neuron_states = {nid: 0 for nid in self.neuron_ids}
        
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
    
    def _load_network_data(self, file_path: str) -> Dict:
        """加载网络分析结果数据"""
        try:
            if not os.path.exists(file_path):
                print(f"警告: 网络分析文件不存在: {file_path}，将使用默认连接方式")
                return {}
            
            with open(file_path, 'r') as f:
                network_data = json.load(f)
            print(f"成功加载网络分析数据: {file_path}")
            return network_data
        except Exception as e:
            print(f"警告: 加载网络分析数据失败: {e}，将使用默认连接方式")
            return {}
    
    def _load_effect_sizes(self, file_path: str) -> pd.DataFrame:
        """加载神经元效应大小数据"""
        try:
            if not os.path.exists(file_path):
                print(f"警告: 神经元效应大小文件不存在: {file_path}，将使用默认神经元特性")
                return pd.DataFrame()
            
            effect_sizes = pd.read_csv(file_path)
            print(f"成功加载神经元效应大小数据: {file_path}")
            return effect_sizes
        except Exception as e:
            print(f"警告: 加载神经元效应大小数据失败: {e}，将使用默认神经元特性")
            return pd.DataFrame()
    
    def _extract_edge_data(self) -> List[Tuple[str, str, float]]:
        """从网络分析数据中提取边信息"""
        edges = []
        try:
            # 检查网络数据是否包含图的边信息
            if self.network_data and 'topology_metrics' in self.network_data and 'graph' in self.network_data['topology_metrics'] and 'edges' in self.network_data['topology_metrics']['graph']:
                edges_data = self.network_data['topology_metrics']['graph']['edges']
                for edge in edges_data:
                    if len(edge) == 3:  # 确保边数据格式正确
                        source, target, weight = edge
                        edges.append((source, target, float(weight)))
                print(f"从网络分析数据中提取了 {len(edges)} 条边")
        except Exception as e:
            print(f"警告: 提取边数据失败: {e}")
        
        return edges
        
    def _load_background_image(self) -> None:
        """Load and prepare background image for visualization."""
        try:
            print(f"正在加载背景图像: {self.background_image_path}")
            img = Image.open(self.background_image_path)
            self.background_image_size = img.size
            print(f"原始图像尺寸: {self.background_image_size}")
            
            # 修改图像编码方式
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            encoded_image = base64.b64encode(img_byte_arr).decode()
            self.background_image = encoded_image
            print("背景图像编码成功")
        except Exception as e:
            print(f"警告: 加载背景图像失败: {e}")
            self.use_background = False
        
    def _get_neuron_ids(self) -> List[str]:
        """Extract neuron IDs from the data columns."""
        neuron_ids = [col for col in self.neuron_data.columns if 'n' in col.lower()]
        if not neuron_ids:
            raise ValueError("在Excel文件中未找到神经元列!")
        print(f"找到 {len(neuron_ids)} 个神经元列:", neuron_ids)
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
        
        # 更新神经元状态
        for _, row in state_df.iterrows():
            neuron_id = row['neuron_id']
            state = 1 if row['state'] == 'ON' else 0
            self.neuron_states[neuron_id] = state
        
        # Update neuron groups
        self._update_neuron_groups(state_df, behavior)
        
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
        
    def _update_neuron_groups(self, state_df: pd.DataFrame, behavior: str) -> None:
        """
        Update neuron group assignments based on their current states and network structure.
        根据神经元状态和网络结构更新神经元组分配。
        
        Args:
            state_df (pd.DataFrame): DataFrame containing neuron states
            behavior (str): Current behavior label
        """
        # 重置组分配
        self.current_groups = {}
        self.neuron_to_group = {}
        
        # 获取活跃的神经元列表
        active_neurons = state_df[state_df['state'] == 'ON']['neuron_id'].tolist()
        
        if not active_neurons:
            return
        
        # 使用网络结构确定组
        # 获取活跃神经元之间的连接子图
        active_subgraph = self.G.subgraph(active_neurons)
        
        # 获取连通分量（每个连通分量是一个组）
        connected_components = list(nx.connected_components(active_subgraph))
        
        # 为每个连通分量分配组 ID 和颜色
        for component in connected_components:
            if len(component) > 0:  # 确保组不为空
                group_id = next(self.group_id_counter)
                self.current_groups[group_id] = list(component)
                
                # 分配神经元到组
                for neuron in component:
                    self.neuron_to_group[neuron] = group_id
                
                # 分配颜色到组
                self.group_colors[group_id] = next(self.color_cycle)
        
        # 处理孤立的活跃神经元（没有连接到其他活跃神经元）
        isolated_neurons = [n for n in active_neurons if n not in self.neuron_to_group]
        if isolated_neurons:
            for neuron in isolated_neurons:
                group_id = next(self.group_id_counter)
                self.current_groups[group_id] = [neuron]
                self.neuron_to_group[neuron] = group_id
                self.group_colors[group_id] = next(self.color_cycle)
            
    def _create_frame_visualization(self, timestamp: float, behavior: str) -> None:
        """
        Create visualization data for the current frame.
        根据神经元组和实际网络连接创建当前帧的可视化数据。
        
        Args:
            timestamp (float): Current frame timestamp
            behavior (str): Current behavior label
        """
        # 创建用于当前帧可视化的图
        G = nx.Graph()
        G.add_nodes_from(self.neuron_ids)
        
        # 创建边 - 根据网络结构和活跃状态
        edges = []
        edge_colors = []
        
        # 如果有效应大小数据，用它来调整节点大小
        size_multiplier = {}
        if not self.effect_sizes.empty and behavior in self.effect_sizes['Behavior'].values:
            # 获取当前行为的神经元效应大小
            behavior_effects = self.effect_sizes[self.effect_sizes['Behavior'] == behavior].iloc[0]
            
            # 对于每个神经元，获取其效应大小
            for nid in self.neuron_ids:
                # 神经元列名可能是形如"Neuron_X"，而不是"nX"，需要映射
                neuron_col = f"Neuron_{nid[1:]}" if nid[0] == 'n' else nid
                
                # 如果找到对应的效应大小，用它来调整节点大小
                if neuron_col in behavior_effects.index:
                    effect_size = behavior_effects[neuron_col]
                    # 将效应大小标准化为1.0到3.0之间的乘数
                    size_multiplier[nid] = max(1.0, min(3.0, effect_size / 2.0 + 1.0))
                else:
                    size_multiplier[nid] = 1.0
        else:
            # 如果没有效应大小数据，所有神经元使用相同大小
            size_multiplier = {nid: 1.0 for nid in self.neuron_ids}
        
        # 遍历网络中的所有边
        for u, v, data in self.G.edges(data=True):
            # 只有当两个神经元都处于活跃状态时才显示它们之间的边
            if self.neuron_states.get(u, 0) == 1 and self.neuron_states.get(v, 0) == 1:
                edges.append((u, v))
                
                # 获取边的颜色 - 使用两个神经元所属组的颜色的混合
                u_group = self.neuron_to_group.get(u)
                v_group = self.neuron_to_group.get(v)
                
                if u_group is not None and v_group is not None:
                    if u_group == v_group:
                        # 如果两个神经元在同一组，使用该组的颜色
                        edge_colors.append(self.group_colors.get(u_group, 'lightgray'))
                    else:
                        # 如果在不同组，使用黑色（或其他特定颜色）
                        edge_colors.append('gray')
                else:
                    edge_colors.append('lightgray')
        
        G.add_edges_from(edges)
        
        # 存储节点信息
        node_x, node_y, node_text, node_color, node_size_list = [], [], [], [], []
        for node in G.nodes():
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
            
            # 节点颜色 - 根据其组分配
            node_color.append(self.group_colors.get(self.neuron_to_group.get(node), 'lightgray'))
            
            # 节点大小 - 基本大小 * 效应大小乘数 * 活跃状态
            base_size = self.node_size
            is_active = self.neuron_states.get(node, 0)
            effect_mult = size_multiplier.get(node, 1.0)
            
            # 如果神经元处于活跃状态，使用效应大小调整其大小；否则使用较小的固定大小
            node_size_list.append(base_size * effect_mult if is_active else base_size * 0.7)
        
        # 存储边信息
        edge_x, edge_y = [], []
        for edge in edges:
            x0, y0 = self.pos[edge[0]]
            x1, y1 = self.pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # 存储帧数据
        self.frames_data['node_x'].append(node_x)
        self.frames_data['node_y'].append(node_y)
        self.frames_data['node_text'].append(node_text)
        self.frames_data['node_color'].append(node_color)
        self.frames_data['node_size'] = self.frames_data.get('node_size', []) + [node_size_list]  # 添加节点大小列表
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
        print(f"动画已保存至 {output_path}")
        
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
                        size=self.frames_data.get('node_size', [[self.node_size] * len(self.frames_data['node_x'][0])])[0]  # 使用动态节点大小或默认大小
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

        # 修改动画控制配置
        layout_config['updatemenus'] = [
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
        ]

        # 修改滑块配置
        layout_config['sliders'] = [
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
                    for k in range(len(self.frames_data['node_x']))
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
        frames = []
        for k in range(len(self.frames_data['node_x'])):
            # 获取当前帧的边颜色列表
            edge_colors = self.frames_data['edge_color'][k] if 'edge_color' in self.frames_data and k < len(self.frames_data['edge_color']) else self.edge_color
            
            # 获取当前帧的节点大小列表
            node_sizes = self.frames_data.get('node_size', [[self.node_size] * len(self.frames_data['node_x'][k])])[k]
            
            frames.append(go.Frame(
                data=[
                    go.Scatter(
                        x=self.frames_data['edge_x'][k],
                        y=self.frames_data['edge_y'][k],
                        mode='lines',
                        line=dict(
                            color=edge_colors if isinstance(edge_colors, str) else self.edge_color,
                            width=self.edge_width
                        ),
                        hoverinfo='none'
                    ),
                    go.Scatter(
                        x=self.frames_data['node_x'][k],
                        y=self.frames_data['node_y'][k],
                        mode='markers+text',
                        text=self.frames_data['node_text'][k],
                        textposition=self.node_text_position,
                        marker=dict(
                            color=self.frames_data['node_color'][k], 
                            size=node_sizes
                        ),
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
            ))
        return frames

    def create_gif(self, output_path: str, fps: int = 10) -> None:
        """
        Generate GIF animation directly from frame data.
        
        Args:
            output_path (str): Path to save the GIF file
            fps (int): Frames per second, default is 10
        """
        print("正在生成GIF动画...")
        
        # Create temporary directory for frame images
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Set fixed figure size and DPI
        fig = plt.figure(figsize=(12, 8), dpi=100)
        
        # Calculate fixed margins to ensure consistent frame size
        left_margin = 0.1
        right_margin = 0.9
        bottom_margin = 0.1
        top_margin = 0.9
        
        # Generate each frame
        frame_files = []
        for k in tqdm(range(len(self.frames_data['node_x'])), desc="生成GIF帧"):
            plt.clf()  # Clear current figure
            
            # Create main axes with fixed position
            ax_main = plt.axes([left_margin, bottom_margin, right_margin-left_margin, top_margin-bottom_margin])
            
            # Draw edges
            if self.frames_data['edge_x'][k]:
                edge_x = self.frames_data['edge_x'][k]
                edge_y = self.frames_data['edge_y'][k]
                
                # 处理边颜色
                edge_colors = self.frames_data['edge_color'][k] if 'edge_color' in self.frames_data and k < len(self.frames_data['edge_color']) else self.edge_color
                
                # 如果是颜色列表，需要扩展为线段颜色
                if isinstance(edge_colors, list):
                    # 边颜色列表需要与边数量匹配
                    edge_color_expanded = []
                    color_idx = 0
                    for i in range(0, len(edge_x), 3):  # 每条边包含3个点(x0, x1, None)
                        if i < len(edge_x) - 2:  # 确保有完整的边
                            edge_color_expanded.extend([edge_colors[color_idx]] * 3)  # 对每条边应用相同颜色
                            color_idx = (color_idx + 1) % len(edge_colors)  # 循环使用颜色
                    
                    # 绘制每个线段（考虑每段边的颜色）
                    for i in range(0, len(edge_x) - 2, 3):
                        x_segment = edge_x[i:i+3]
                        y_segment = edge_y[i:i+3]
                        color = edge_color_expanded[i] if i < len(edge_color_expanded) else self.edge_color
                        ax_main.plot(x_segment, y_segment, color=color, linewidth=self.edge_width, zorder=1)
                else:
                    # 使用单一颜色绘制所有边
                    ax_main.plot(edge_x, edge_y, color=edge_colors, linewidth=self.edge_width, zorder=1)
            
            # 获取节点大小
            node_sizes = self.frames_data.get('node_size', [[self.node_size * 20] * len(self.frames_data['node_x'][k])])[k]
            # 转换大小为matplotlib格式（乘以比例因子）
            node_sizes = [size * 20 for size in node_sizes]  # 调整大小以匹配HTML版本
            
            # Draw nodes with dynamic sizes
            ax_main.scatter(self.frames_data['node_x'][k], 
                          self.frames_data['node_y'][k],
                          c=self.frames_data['node_color'][k],
                          s=node_sizes,
                          zorder=2)
            
            # Add node labels
            for i, txt in enumerate(self.frames_data['node_text'][k]):
                ax_main.annotate(txt, 
                               (self.frames_data['node_x'][k][i], self.frames_data['node_y'][k][i]),
                               textcoords="offset points",
                               xytext=(0, 0),
                               ha='center',
                               va='center',
                               fontsize=8)
            
            # Set title and behavior label
            # Convert Chinese title to English
            title = self.frames_data['titles'][k].replace("神经元拓扑结构图 - 时间:", "Neuron Topology - Time:")
            ax_main.set_title(title)
            ax_main.text(0.02, 0.98, f"Current Behavior: {self.frames_data['behaviors'][k]}",
                        transform=ax_main.transAxes,
                        fontsize=10,
                        verticalalignment='top')
            
            # Set axes limits
            ax_main.set_xlim(-0.05, 1.05)
            ax_main.set_ylim(1.05, -0.05)  # Reverse Y axis to match HTML version
            ax_main.axis('off')
            
            # Add background if enabled
            if self.use_background and self.background_image:
                ax_main.imshow(Image.open(io.BytesIO(base64.b64decode(self.background_image))),
                             extent=[-0.05, 1.05, -0.05, 1.05],
                             alpha=self.background_opacity,
                             zorder=0)
            
            # Save current frame with fixed size
            frame_path = os.path.join(temp_dir, f"frame_{k:04d}.png")
            plt.savefig(frame_path, bbox_inches=None, pad_inches=0)
            frame_files.append(frame_path)
        
        # Read all frames and create GIF
        frames = [imageio.imread(f) for f in frame_files]
        imageio.mimsave(output_path, frames, fps=fps)
        
        # Clean up temporary files
        for f in frame_files:
            os.remove(f)
        os.rmdir(temp_dir)
        
        plt.close()  # Close figure
        print(f"GIF动画已保存至: {output_path}")

def main():
    """Main function to run the topology analysis."""
    # 创建配置实例
    config = AnalysisConfig()
    
    # 确保输出目录存在
    config.setup_directories()
    
    # 输出文件路径信息
    print(f"神经元数据文件: {config.data_file}")
    print(f"位置数据文件: {config.position_data_file}")
    print(f"背景图像文件: {config.background_image}")
    print(f"网络分析结果文件: {config.network_analysis_file}")
    print(f"神经元效应大小文件: {config.neuron_effect_file}")
    print(f"输出HTML文件: {config.topology_html}")
    print(f"输出GIF文件: {config.topology_gif}")
    
    # 检查文件是否存在
    print(f"检查背景图像是否存在: {config.background_image}")
    if not os.path.exists(config.background_image):
        print(f"警告: 未找到背景图像 {config.background_image}")
    else:
        print(f"成功找到背景图像: {config.background_image}")
    
    print(f"检查网络分析结果文件是否存在: {config.network_analysis_file}")
    if not os.path.exists(config.network_analysis_file):
        print(f"警告: 未找到网络分析结果文件 {config.network_analysis_file}，将使用默认连接方式")
    else:
        print(f"成功找到网络分析结果文件: {config.network_analysis_file}")
    
    print(f"检查神经元效应大小文件是否存在: {config.neuron_effect_file}")
    if not os.path.exists(config.neuron_effect_file):
        print(f"警告: 未找到神经元效应大小文件 {config.neuron_effect_file}，将使用默认神经元特性")
    else:
        print(f"成功找到神经元效应大小文件: {config.neuron_effect_file}")
    
    # 背景图显示开关
    SHOW_BACKGROUND = False  # 设置为 True 显示背景图，False 不显示
    
    # 创建分析器实例
    analyzer = NeuronTopologyAnalyzer(
        config=config,
        use_background=SHOW_BACKGROUND,  # 使用开关变量
        edge_weight_threshold=config.visualization_params['topology']['edge_weight_threshold']
    )
    analyzer.process_all_frames()
    
    # 生成HTML动画
    analyzer.create_animation(config.topology_html)
    
    # 生成GIF动画
    analyzer.create_gif(config.topology_gif, fps=config.visualization_params['topology']['gif_fps'])

if __name__ == '__main__':
    main()