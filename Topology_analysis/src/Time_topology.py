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
        'edge_x': [], 'edge_y': [], 'edge_color': [], 'titles': []
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
    """
    Create and save the animation visualization.
    
    Args:
        frames_data (dict): Processed frame data for visualization
        output_path (str): Path to save the HTML animation file
    """
    fig = go.Figure(
        data=[
            go.Scatter(
                x=frames_data['edge_x'][0],
                y=frames_data['edge_y'][0],
                mode='lines',
                line=dict(color='black', width=2),
                hoverinfo='none'
            ),
            go.Scatter(
                x=frames_data['node_x'][0],
                y=frames_data['node_y'][0],
                mode='markers+text',
                text=frames_data['node_text'][0],
                textposition='middle center',
                marker=dict(color=frames_data['node_color'][0], size=15),
                hoverinfo='text'
            )
        ],
        layout=create_layout(frames_data['titles'][0], len(frames_data['node_x']))
    )
    
    # Add frames to animation
    fig.frames = [
        go.Frame(
            data=[
                go.Scatter(
                    x=frames_data['edge_x'][k],
                    y=frames_data['edge_y'][k],
                    mode='lines',
                    line=dict(color='black', width=2),
                    hoverinfo='none'
                ),
                go.Scatter(
                    x=frames_data['node_x'][k],
                    y=frames_data['node_y'][k],
                    mode='markers+text',
                    text=frames_data['node_text'][k],
                    textposition='middle center',
                    marker=dict(color=frames_data['node_color'][k], size=15),
                    hoverinfo='text'
                )
            ],
            name=f"frame_{k}",
            layout=go.Layout(title=frames_data['titles'][k])
        )
        for k in range(len(frames_data['node_x']))
    ]
    
    fig.write_html(output_path)

def create_layout(initial_title, frame_count):
    """
    Create the layout configuration for the animation.
    
    Args:
        initial_title (str): Title for the first frame
        frame_count (int): Total number of frames
        
    Returns:
        go.Layout: Layout configuration
    """
    return go.Layout(
        title=initial_title,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        sliders=[dict(
            active=0,
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
        )]
    )

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
    output_path = '../graph/neuron_activity_animation_mean_Day6.html'
    create_animation(frames_data, output_path)

if __name__ == "__main__":
    main()