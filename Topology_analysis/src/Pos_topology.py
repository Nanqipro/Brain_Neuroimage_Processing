"""
This script analyzes and visualizes the topological structure of neuron activity over time.
It creates an interactive visualization showing how neurons form groups based on their activity patterns.
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
from typing import Dict, List, Tuple, Iterator
import plotly.io as pio
import base64
from PIL import Image
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Set default plotly template
pio.templates.default = "plotly"

class NeuronTopologyAnalyzer:
    """
    A class to analyze and visualize the topological structure of neuron activity.
    """
    
    def __init__(self, neuron_data_path: str, position_data_path: str):
        """
        Initialize the analyzer with data paths.
        
        Args:
            neuron_data_path (str): Path to the Excel file containing neuron activity data
            position_data_path (str): Path to the CSV file containing neuron positions
        """
        self.neuron_data = pd.read_excel(neuron_data_path)
        self.positions_data = pd.read_csv(position_data_path)
        
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
            'edge_color': [], 'titles': []
        }
        
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
        """Initialize color scheme for neuron groups."""
        N = 100  # Maximum number of groups
        cmap = cm.get_cmap('tab20', N)
        color_list = [matplotlib.colors.to_hex(cmap(i)) for i in range(N)]
        return itertools.cycle(color_list)
        
    def _process_frame(self, frame_num: int) -> None:
        """
        Process a single frame of neuron activity data.
        
        Args:
            frame_num (int): The frame number to process
        """
        # Get timestamp and activity values
        t = self.neuron_data['stamp'].iloc[frame_num]
        activity_values = self.neuron_data[self.neuron_ids].iloc[frame_num]
        
        # Determine neuron states
        state_df = self._get_neuron_states(activity_values)
        
        # Update neuron groups
        self._update_neuron_groups(state_df)
        
        # Create and store frame visualization data
        self._create_frame_visualization(t)
        
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
            
    def _create_frame_visualization(self, timestamp: float) -> None:
        """
        Create visualization data for the current frame.
        
        Args:
            timestamp (float): Current frame timestamp
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
                go.Scatter(
                    x=self.frames_data['edge_x'][0],
                    y=self.frames_data['edge_y'][0],
                    mode='lines',
                    line=dict(color='black', width=2),
                    hoverinfo='none'
                ),
                go.Scatter(
                    x=self.frames_data['node_x'][0],
                    y=self.frames_data['node_y'][0],
                    mode='markers+text',
                    text=self.frames_data['node_text'][0],
                    textposition='middle center',
                    marker=dict(color=self.frames_data['node_color'][0], size=15),
                    hoverinfo='text'
                )
            ],
            layout=self._create_layout()
        )
        
    def _create_layout(self) -> go.Layout:
        """Create the layout for the animation."""
        # Load and encode background image
        bg_image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', 'Day6_Max.png')
        img = Image.open(bg_image_path)
        img_width, img_height = img.size
        
        # Convert image to base64 string
        import io
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        encoded_image = base64.b64encode(img_byte_arr).decode()
        
        # Calculate the aspect ratio
        aspect_ratio = img_width / img_height
        
        return go.Layout(
            title=dict(
                text=self.frames_data['titles'][0],
                y=0.98  # 调整标题位置
            ),
            showlegend=False,
            width=800,  # 设置固定宽度
            height=int(800/aspect_ratio),  # 根据宽高比计算高度
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                scaleanchor='y', 
                scaleratio=1,
                range=[0, 1],
                domain=[0, 1],
                constrain='domain'  # 确保x轴范围被限制在domain内
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                autorange='reversed',
                range=[0, 1],
                domain=[0, 1],
                constrain='domain',  # 确保y轴范围被限制在domain内
                scaleanchor='x',  # 确保x和y轴的缩放比例相同
                scaleratio=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            images=[dict(
                source='data:image/png;base64,{}'.format(encoded_image),
                xref="paper",
                yref="paper",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                sizing="contain",  # 改为contain以保持图像比例
                opacity=1,
                layer="below"
            )],
            margin=dict(l=0, r=0, t=30, b=0),
            sliders=[dict(
                active=0,
                steps=[dict(
                    label=str(i),
                    method="animate",
                    args=[[f"frame_{i}"], {
                        "frame": {"duration": 1000, "redraw": True},
                        "mode": "immediate"
                    }]
                ) for i in range(len(self.frames_data['node_x']))]
            )],
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[dict(
                    label='Play',
                    method='animate',
                    args=[None, dict(frame=dict(duration=1000, redraw=True),
                                   fromcurrent=True)]
                ), dict(
                    label='Pause',
                    method='animate',
                    args=[[None], dict(frame=dict(duration=0, redraw=False),
                                     mode='immediate')]
                )]
            )]
        )
        
    def _create_animation_frames(self) -> List[go.Frame]:
        """Create frames for the animation."""
        return [
            go.Frame(
                data=[
                    go.Scatter(
                        x=self.frames_data['edge_x'][k],
                        y=self.frames_data['edge_y'][k],
                        mode='lines',
                        line=dict(color='black', width=2),
                        hoverinfo='none'
                    ),
                    go.Scatter(
                        x=self.frames_data['node_x'][k],
                        y=self.frames_data['node_y'][k],
                        mode='markers+text',
                        text=self.frames_data['node_text'][k],
                        textposition='middle center',
                        marker=dict(color=self.frames_data['node_color'][k], size=15),
                        hoverinfo='text'
                    )
                ],
                name=f"frame_{k}",
                layout=go.Layout(title=self.frames_data['titles'][k])
            )
            for k in range(len(self.frames_data['node_x']))
        ]

def main():
    """Main function to run the topology analysis."""
    # Define file paths
    neuron_data_path = '../datasets/Day6_with_behavior_labels_filled.xlsx'
    position_data_path = '../datasets/Day6_Max_position.csv'
    output_path = '../graph/Day6_pos_topology.html'
    
    # Create analyzer and process data
    analyzer = NeuronTopologyAnalyzer(neuron_data_path, position_data_path)
    analyzer.process_all_frames()
    analyzer.create_animation(output_path)

if __name__ == '__main__':
    main()