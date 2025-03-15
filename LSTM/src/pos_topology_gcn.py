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
import sys

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
                 edge_weight_threshold: Optional[float] = None,
                 network_file: Optional[str] = None,
                 gcn_topo_file: Optional[str] = None):
        """
        使用配置和可选的覆盖参数初始化分析器。
        
        参数:
            config (AnalysisConfig): 包含路径和参数的配置对象
            use_background (Optional[bool]): 是否在可视化中使用背景图像
            node_size (Optional[int]): 可视化中节点的大小
            node_text_position (Optional[str]): 节点标签的位置（'middle center'、'top center'等）
            edge_width (Optional[int]): 连接节点的边的宽度
            edge_color (Optional[str]): 边的颜色
            background_opacity (Optional[float]): 背景图像的不透明度（0-1）
            frame_duration (Optional[int]): 动画中每帧的持续时间（毫秒）
            color_scheme (Optional[str]): 节点组的配色方案（'tab20'、'Set1'、'Set2'等）
            max_groups (Optional[int]): 颜色循环的最大组数
            edge_weight_threshold (Optional[float]): 创建连接的边权重阈值
            network_file (Optional[str]): 网络分析JSON文件的路径
            gcn_topo_file (Optional[str]): GCN拓扑数据JSON文件的路径
        """
        # 存储配置参数
        self.config = config
        
        # 初始化数据存储
        self.neuron_data = None
        self.position_data = None
        self.network_data = None
        self.gcn_topo_data = None
        self.effect_sizes = None
        self.edges = []
        self.neuron_ids = []
        self.positions = {}
        self.frames = []
        self.current_frame_states = {}
        self.neuron_groups = {}
        self.group_colors = {}
        self.background_image_path = None
        self.background_image_size = None
        
        # 加载网络文件路径
        self.network_file_path = network_file or config.network_analysis_file
        
        # 加载GCN拓扑文件路径
        self.gcn_topo_file_path = gcn_topo_file or os.path.join(config.analysis_dir, 'gnn_results/gcn_topology_data.json')
        
        # 加载数据
        try:
            # 加载神经元数据
            self.neuron_data = pd.read_excel(config.data_file)
            print(f"成功加载神经元数据: {config.data_file}")
            
            # 加载位置数据
            self.position_data = pd.read_csv(config.position_data_file)
            print(f"成功加载位置数据: {config.position_data_file}")
            
            # 加载网络分析结果
            self.network_data = self._load_network_data(self.network_file_path)
            
            # 加载GCN拓扑数据
            self.gcn_topo_data = self._load_gcn_topo_data(self.gcn_topo_file_path)
            print(f"成功加载GCN拓扑数据: {self.gcn_topo_file_path}")
            
            # 加载神经元效应大小数据
            self.effect_sizes = self._load_effect_sizes(config.neuron_effect_file)
            
            # 提取边数据 (优先使用GCN拓扑数据中的边)
            self.edges = self._extract_edge_data()
            
            # 验证behavior列是否存在
            if 'behavior' not in self.neuron_data.columns:
                raise ValueError("Behavior column not found in the Excel file!")
            
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
            
            # 高亮边的颜色
            self.highlight_edge_color = "red"
            
            # 非高亮边的颜色
            self.normal_edge_color = "rgba(180, 180, 180, 0.5)"
            
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
            
            # 添加边（使用GCN拓扑数据中的边数据）
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
            
            # 确定网络数据类型
            self.network_type = "GCN拓扑结构分析"
            print(f"检测到网络数据类型: {self.network_type}")
            
            # 修改颜色设置
            self.active_node_color = 'red'  # 活跃节点的颜色
            self.inactive_node_color = 'lightgray'  # 非活跃节点的颜色
            self.edge_color = 'rgba(180, 180, 180, 0.5)'  # 统一的边颜色
        except Exception as e:
            print(f"初始化失败: {str(e)}")
    
    def _load_gcn_topo_data(self, file_path: str) -> Dict:
        """
        加载GCN拓扑数据
        
        参数:
            file_path: GCN拓扑数据的JSON文件路径
            
        返回:
            拓扑数据字典
        """
        try:
            if file_path and os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"成功加载GCN拓扑数据: {file_path}")
                return data
            else:
                print(f"警告: GCN拓扑数据文件不存在: {file_path}")
                # 返回一个空的拓扑数据结构
                return {
                    'nodes': [],
                    'edges': []
                }
        except Exception as e:
            print(f"加载GCN拓扑数据出错: {str(e)}")
            return {
                'nodes': [],
                'edges': []
            }
    
    def _load_network_data(self, file_path: str) -> Dict:
        """
        加载网络分析结果数据
        
        参数:
            file_path: 网络分析结果的JSON文件路径
            
        返回:
            网络数据字典
        """
        try:
            if file_path and os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"成功加载网络数据: {file_path}")
                return data
            else:
                print(f"警告: 网络数据文件不存在: {file_path}")
                # 返回一个空的网络数据结构
                return {
                    'nodes': [],
                    'links': []
                }
        except Exception as e:
            print(f"加载网络数据出错: {str(e)}")
            return {
                'nodes': [],
                'links': []
            }
    
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
        """从网络分析数据中提取边信息，优先使用GCN拓扑数据"""
        edges = []
        try:
            # 检查是否有GCN拓扑数据
            if self.gcn_topo_data and 'edges' in self.gcn_topo_data:
                edges_data = self.gcn_topo_data['edges']
                for edge in edges_data:
                    if 'source' in edge and 'target' in edge and 'similarity' in edge:
                        # 使用名称而不是索引，将节点ID映射为n加上索引号
                        source_id = f"n{edge['source']+1}" if isinstance(edge['source'], int) else str(edge['source'])
                        target_id = f"n{edge['target']+1}" if isinstance(edge['target'], int) else str(edge['target'])
                        weight = float(edge['similarity'])
                        edges.append((source_id, target_id, weight))
                print(f"从GCN拓扑数据中提取了 {len(edges)} 条边")
                
                # 如果从GCN拓扑数据获取到边，直接返回
                if edges:
                    return edges
            
            # 如果没有GCN拓扑数据或者没有提取到边，尝试使用网络分析数据
            # 检查是否是标准的网络分析结果格式
            if self.network_data and 'topology_metrics' in self.network_data and 'graph' in self.network_data['topology_metrics'] and 'edges' in self.network_data['topology_metrics']['graph']:
                edges_data = self.network_data['topology_metrics']['graph']['edges']
                for edge in edges_data:
                    if len(edge) == 3:  # 确保边数据格式正确
                        source, target, weight = edge
                        edges.append((source, target, float(weight)))
                print(f"从标准网络分析数据中提取了 {len(edges)} 条边")
            
            # 检查是否是特殊格式的网络数据 (mst, threshold, top_edges)
            elif self.network_data and 'links' in self.network_data:
                links_data = self.network_data['links']
                for link in links_data:
                    if 'source' in link and 'target' in link and 'weight' in link:
                        source = link['source']
                        target = link['target']
                        weight = link['weight']
                        edges.append((source, target, float(weight)))
                print(f"从特殊格式网络数据中提取了 {len(edges)} 条边")
            
            else:
                print("警告: 网络数据格式不兼容，无法提取边信息")
        
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
        """Extract neuron IDs from the network data or data columns."""
        neuron_ids = []
        
        # 如果有GCN拓扑数据，首先尝试从中获取神经元ID
        if self.gcn_topo_data and 'nodes' in self.gcn_topo_data:
            # 从GCN拓扑数据中提取节点ID，并转换为n+索引号的格式
            neuron_ids = [f"n{node['id']+1}" for node in self.gcn_topo_data['nodes']]
            print(f"从GCN拓扑数据中获取了 {len(neuron_ids)} 个神经元ID")
            
            # 验证这些ID是否存在于神经元数据中
            valid_ids = [nid for nid in neuron_ids if nid in self.neuron_data.columns]
            if len(valid_ids) == len(neuron_ids):
                return neuron_ids
            else:
                print(f"警告: GCN拓扑数据中的节点ID与神经元数据列不完全匹配，将使用神经元数据中的ID")
                neuron_ids = []
        
        # 尝试从标准网络分析数据中获取神经元ID
        if self.network_data and 'topology_metrics' in self.network_data and 'graph' in self.network_data['topology_metrics'] and 'nodes' in self.network_data['topology_metrics']['graph']:
            neuron_ids = self.network_data['topology_metrics']['graph']['nodes']
            print(f"从标准网络数据中获取了 {len(neuron_ids)} 个神经元ID")
        
        # 检查特殊格式网络数据
        elif self.network_data and 'nodes' in self.network_data:
            neuron_ids = [node['id'] for node in self.network_data['nodes']]
            print(f"从特殊格式网络数据中获取了 {len(neuron_ids)} 个神经元ID")
        
        # 如果网络数据中没有神经元ID，则从神经元数据中提取
        if not neuron_ids:
            neuron_ids = [col for col in self.neuron_data.columns if 'n' in col.lower()]
            print(f"从神经元数据中提取了 {len(neuron_ids)} 个神经元ID")
        
        if not neuron_ids:
            raise ValueError("未能找到任何神经元ID!")
            
        print(f"最终使用 {len(neuron_ids)} 个神经元ID")
        return neuron_ids
        
    def _get_positions(self) -> Dict[str, Tuple[float, float]]:
        """Convert position data to dictionary format."""
        pos = {}
        for nid, (rx, ry) in zip(self.neuron_ids, 
                                self.position_data[['relative_x', 'relative_y']].values):
            pos[nid] = (rx, ry)
        return pos
        
    def _validate_data(self) -> None:
        """Validate that the number of positions matches the number of neurons."""
        if len(self.position_data) != len(self.neuron_ids):
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
        创建当前帧的可视化数据。
        
        Args:
            timestamp (float): 当前时间戳
            behavior (str): 当前行为标签
        """
        # 存储节点信息
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        # 获取活跃的神经元列表
        active_neurons = [nid for nid, state in self.neuron_states.items() if state == 1]
        
        # 为所有神经元创建可视化数据
        for node in self.neuron_ids:
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
            
            # 设置节点颜色和大小
            if node in active_neurons:
                node_color.append(self.active_node_color)
                node_size.append(self.node_size * 1.2)  # 活跃节点略大
            else:
                node_color.append(self.inactive_node_color)
                node_size.append(self.node_size)
        
        # 存储帧数据
        self.frames_data['node_x'].append(node_x)
        self.frames_data['node_y'].append(node_y)
        self.frames_data['node_text'].append(node_text)
        self.frames_data['node_color'].append(node_color)
        self.frames_data['node_size'].append(node_size)
        self.frames_data['titles'].append(f"神经元拓扑结构 ({self.network_type}) - 时间戳: {timestamp}")
        self.frames_data['behaviors'].append(behavior)
        
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
        """
        创建可视化的基础图形。
        返回:
            go.Figure: 包含初始节点和边的图形
        """
        # 创建所有边的跟踪对象（固定位置）
        edge_x = []
        edge_y = []
        
        # 使用所有边创建固定的边线位置
        for edge in self.edges:
            if edge[0] in self.pos and edge[1] in self.pos:  # 确保两端的节点都有位置信息
                x0, y0 = self.pos[edge[0]]
                x1, y1 = self.pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        # 创建边的跟踪对象
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(
                width=self.edge_width,
                color=self.edge_color
            ),
            hoverinfo='none',
            name='神经元连接'
        )
        
        # 创建节点的跟踪对象
        node_x = []
        node_y = []
        node_text = []
        
        for node in self.neuron_ids:
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=self.node_size,
                color=self.inactive_node_color,  # 初始状态所有节点都是非活跃的
                line=dict(width=1, color='white')
            ),
            name='神经元'
        )
        
        # 创建图形并添加基础数据
        layout = self._create_layout()
        fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
        
        return fig
        
    def _create_layout(self) -> go.Layout:
        """创建图形布局"""
        # 检查行为类型是否可用
        behavior_available = 'Behavior' in self.neuron_data.columns
        
        # 计算节点位置的范围
        x_coords = [pos[0] for pos in self.pos.values()]
        y_coords = [pos[1] for pos in self.pos.values()]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # 计算坐标范围和添加边距
        x_range = max_x - min_x
        y_range = max_y - min_y
        padding = 0.2  # 增加20%的边距
        
        x_min = min_x - (x_range * padding)
        x_max = max_x + (x_range * padding)
        y_min = min_y - (y_range * padding)
        y_max = max_y + (y_range * padding)
        
        # 设置图形的宽高比
        width = 1200
        height = int(width * (y_range / x_range))
        
        # 确保最小高度
        height = max(height, 800)
        
        # 如果有行为信息，增加高度
        if behavior_available:
            height += 50
        
        # 创建自定义网格线
        grid_lines = []
        
        # 添加网格线，使用相对间距
        grid_step_x = x_range / 10  # 将范围分成10份
        grid_step_y = y_range / 10
        
        # X方向网格线
        for x in np.arange(x_min, x_max + grid_step_x, grid_step_x):
            grid_lines.append(
                go.layout.Shape(
                    type="line",
                    x0=x, y0=y_min,
                    x1=x, y1=y_max,
                    line=dict(
                        color="rgba(200, 200, 200, 0.2)",
                        width=1,
                        dash="dash"
                    )
                )
            )
        
        # Y方向网格线
        for y in np.arange(y_min, y_max + grid_step_y, grid_step_y):
            grid_lines.append(
                go.layout.Shape(
                    type="line",
                    x0=x_min, y0=y,
                    x1=x_max, y1=y,
                    line=dict(
                        color="rgba(200, 200, 200, 0.2)",
                        width=1,
                        dash="dash"
                    )
                )
            )
        
        # 创建布局
        layout = go.Layout(
            title=dict(
                text=f"神经元拓扑结构 - {self.network_type}",
                font=dict(
                    family="Arial",
                    size=24,
                    color="#333333"
                ),
                x=0.5,
                xanchor="center"
            ),
            width=width,
            height=height,
            showlegend=True,
            legend=dict(
                x=1.05,
                y=0.5,
                xanchor="left",
                yanchor="middle",
                font=dict(
                    family="Arial",
                    size=12,
                    color="#333333"
                ),
                bgcolor="rgba(255, 255, 255, 0.5)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            ),
            xaxis=dict(
                range=[x_min, x_max],
                showgrid=False,
                zeroline=False,
                showticklabels=True,  # 显示坐标轴刻度
                title="X Position",
                tickformat=".2f"  # 保留两位小数
            ),
            yaxis=dict(
                range=[y_min, y_max],
                showgrid=False,
                zeroline=False,
                showticklabels=True,  # 显示坐标轴刻度
                title="Y Position",
                tickformat=".2f",  # 保留两位小数
                scaleanchor="x",
                scaleratio=1
            ),
            margin=dict(
                l=50,  # 增加左边距
                r=50,  # 增加右边距
                t=50,
                b=50   # 增加底部边距
            ),
            plot_bgcolor="rgba(255, 255, 255, 0)",
            paper_bgcolor="rgba(255, 255, 255, 1)",
            hovermode="closest",
            shapes=grid_lines,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="播放",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=self.frame_duration, redraw=True),
                                    fromcurrent=True,
                                    mode="immediate"
                                )
                            ]
                        ),
                        dict(
                            label="暂停",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode="immediate"
                                )
                            ]
                        )
                    ],
                    direction="left",
                    pad=dict(r=10, t=10),
                    x=0.1,
                    y=0,
                    xanchor="right",
                    yanchor="top"
                )
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    currentvalue=dict(
                        font=dict(size=14),
                        prefix="时间点: ",
                        visible=True,
                        xanchor="right"
                    ),
                    pad=dict(t=50, b=10),
                    len=0.9,
                    x=0.1,
                    y=0,
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [f"frame{i}"],
                                dict(
                                    mode="immediate",
                                    frame=dict(duration=self.frame_duration, redraw=True),
                                    transition=dict(duration=300)
                                )
                            ],
                            label=str(i)
                        ) for i in range(len(self.neuron_data))
                    ]
                )
            ]
        )
        
        # 如果使用背景图像
        if self.use_background and self.background_image:
            layout.images = [
                dict(
                    source=f"data:image/png;base64,{self.background_image}",
                    xref="x",
                    yref="y",
                    x=x_min,
                    y=y_max,
                    sizex=x_max - x_min,
                    sizey=y_max - y_min,
                    sizing="stretch",
                    opacity=self.background_opacity,
                    layer="below"
                )
            ]
        
        return layout
        
    def _create_animation_frames(self) -> List[go.Frame]:
        """
        创建动画帧列表。
        
        返回:
            List[go.Frame]: 动画帧列表
        """
        frames = []
        
        # 获取所有边的固定位置
        edge_x = []
        edge_y = []
        for edge in self.edges:
            if edge[0] in self.pos and edge[1] in self.pos:
                x0, y0 = self.pos[edge[0]]
                x1, y1 = self.pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        # 创建固定的边跟踪对象
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(
                width=self.edge_width,
                color=self.edge_color
            ),
            hoverinfo='none',
            name='神经元连接'
        )
        
        for i in range(len(self.frames_data['node_x'])):
            # 获取当前帧的活跃神经元
            active_neurons = [nid for nid, state in self.neuron_states.items() if state == 1]
            
            # 创建节点颜色列表
            node_colors = [
                self.active_node_color if node in active_neurons else self.inactive_node_color
                for node in self.neuron_ids
            ]
            
            # 创建节点大小列表
            node_sizes = [
                self.node_size * 1.2 if node in active_neurons else self.node_size
                for node in self.neuron_ids
            ]
            
            # 创建当前帧的节点可视化
            node_trace = go.Scatter(
                x=self.frames_data['node_x'][i],
                y=self.frames_data['node_y'][i],
                mode='markers+text',
                text=self.frames_data['node_text'][i],
                textposition="top center",
                marker=dict(
                    color=node_colors,
                    size=node_sizes,
                    line=dict(width=1, color='white')
                ),
                name='神经元'
            )
            
            # 创建帧
            frame = go.Frame(
                data=[edge_trace, node_trace],
                name=f"frame{i}",
                layout=go.Layout(title=self.frames_data['titles'][i])
            )
            frames.append(frame)
        
        return frames

def main():
    """Main function to run the topology analysis."""
    # 创建配置实例
    config = AnalysisConfig()
    
    # 确保输出目录存在
    config.setup_directories()
    
    # 设置GCN拓扑数据文件路径
    gcn_topo_file = os.path.join(config.analysis_dir, 'gnn_results/gcn_topology_data.json')
    if not os.path.exists(gcn_topo_file):
        print(f"警告: GCN拓扑数据文件不存在: {gcn_topo_file}")
    else:
        print(f"成功找到GCN拓扑数据文件: {gcn_topo_file}")
    
    # 背景图显示开关
    SHOW_BACKGROUND = False  # 设置为 True 显示背景图，False 不显示
    
    # 线条和节点大小自定义设置
    EDGE_WIDTH = 2      # 线条宽度
    NODE_SIZE = 15      # 节点大小
    
    # 设置输出文件路径
    html_output = os.path.join(config.topology_dir, f'pos_topology_{config.data_identifier}_gcn.html')
    
    print(f"使用GCN拓扑数据: {gcn_topo_file}")
    print(f"输出HTML文件: {html_output}")
    
    # 创建分析器实例
    analyzer = NeuronTopologyAnalyzer(
        config=config,
        use_background=SHOW_BACKGROUND,
        edge_weight_threshold=0.6,  # 使用较高的阈值过滤低相似度连接
        edge_width=EDGE_WIDTH,
        node_size=NODE_SIZE,
        gcn_topo_file=gcn_topo_file  # 添加GCN拓扑数据文件路径
    )
    analyzer.process_all_frames()
    
    # 生成HTML动画
    analyzer.create_animation(html_output)
    
    print(f"完成GCN拓扑结构可视化")

if __name__ == '__main__':
    main()