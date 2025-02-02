#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates a time activity raster plot of neurons by cluster using GMM,
with behavioral annotations overlaid on the plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple
import sys


def load_data(metrics_path: str, behavior_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and validate both neuron metrics and behavioral data from Excel files.
    
    Args:
        metrics_path (str): Path to the Excel file containing neuron metrics data.
        behavior_path (str): Path to the Excel file containing behavioral data.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing metrics and behavioral DataFrames.
        
    Raises:
        ValueError: If required columns are missing from either dataset.
    """
    try:
        # Load metrics data
        metrics_df = pd.read_excel(metrics_path, sheet_name='Windows100_step10')
        metrics_required_columns = ['Neuron', 'Start Time', 'GMM']
        
        if not all(col in metrics_df.columns for col in metrics_required_columns):
            raise ValueError(f"Metrics dataset must contain all required columns: {metrics_required_columns}")
        
        # Load behavioral data
        behavioral_df = pd.read_excel(behavior_path)
        behavior_required_columns = ['behavior', 'stamp']
        
        if not all(col in behavioral_df.columns for col in behavior_required_columns):
            raise ValueError(f"Behavioral dataset must contain all required columns: {behavior_required_columns}")
            
        return metrics_df, behavioral_df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)


def create_neuron_mapping(metrics_df: pd.DataFrame) -> Dict[str, int]:
    """
    Create a mapping of neuron labels to sequential IDs.
    
    Args:
        metrics_df (pd.DataFrame): Input DataFrame containing neuron data.
        
    Returns:
        Dict[str, int]: Mapping of neuron labels to sequential IDs.
    """
    metrics_df['Neuron'] = metrics_df['Neuron'].astype(str)
    unique_neurons = metrics_df['Neuron'].drop_duplicates().tolist()
    return {neuron: i + 1 for i, neuron in enumerate(unique_neurons)}


def plot_activity_raster(metrics_df: pd.DataFrame, behavioral_df: pd.DataFrame,
                        neuron_mapping: Dict[str, int], unique_neurons: List[str]) -> None:
    """
    Generate the time activity raster plot with behavioral annotations.
    
    Args:
        metrics_df (pd.DataFrame): Input DataFrame containing neuron data.
        behavioral_df (pd.DataFrame): Input DataFrame containing behavioral data.
        neuron_mapping (Dict[str, int]): Mapping of neuron labels to sequential IDs.
        unique_neurons (List[str]): List of unique neuron labels.
    """
    plt.figure(figsize=(20, 15))
    colors = plt.cm.Set3.colors

    # Generate legend for clusters
    unique_clusters = sorted(metrics_df['GMM'].unique())
    handles = [
        mpatches.Patch(color=colors[int(cluster) % len(colors)], 
                      label=f"Cluster {int(cluster)}")
        for cluster in unique_clusters
    ]

    # Plot neuron activities
    for neuron in unique_neurons:
        neuron_data = metrics_df[metrics_df['Neuron'] == neuron]
        y_position = neuron_mapping[neuron]

        for _, row in neuron_data.iterrows():
            start_time = row['Start Time']
            end_time = start_time + 5  # Fixed step size of 5
            cluster = int(row['GMM'])
            plt.plot(
                [start_time, end_time],
                [y_position, y_position],
                color=colors[cluster % len(colors)],
                linewidth=5,
            )

    # Add behavioral annotations
    # Keep track of the last behavior we plotted
    last_behavior = None
    
    # Sort behavioral data by stamp
    sorted_behavioral_data = behavioral_df[behavioral_df['behavior'].notna()].sort_values('stamp')
    
    for _, behavioral_row in sorted_behavioral_data.iterrows():
        stamp = behavioral_row['stamp']
        behavior = behavioral_row['behavior']
        
        # Plot the vertical line and label if:
        # 1. This is the first occurrence of any behavior
        # 2. This behavior is different from the last one
        if last_behavior is None or behavior != last_behavior:
            # Plot a black vertical line at the behavioral 'stamp'
            plt.axvline(x=stamp, color='black', linewidth=1, linestyle='--')

            # Add behavioral annotation next to the line
            plt.text(stamp + 0.5, len(unique_neurons) + 1, behavior, 
                    color='black', ha='left', va='bottom', fontsize=12, rotation=90)
            
            # Update the last behavior
            last_behavior = behavior

    # Customize plot appearance
    plt.xlabel('Time (stamp)', fontsize=16, fontweight='bold')
    plt.ylabel('Neuron ID', fontsize=16, fontweight='bold')
    plt.title('Time Activity Raster Plot of Neurons by Cluster (GMM) with Behavioral Labels', 
             fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(range(1, len(unique_neurons) + 1), unique_neurons, 
              fontsize=12, fontweight='bold')
    plt.legend(handles=handles, title="Clusters", fontsize=12, 
              title_fontsize=14, loc='upper right')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adjust plot margins to accommodate behavioral labels
    plt.subplots_adjust(top=0.85)
    plt.tight_layout()


def main():
    """Main function to execute the plotting workflow."""
    # Configuration
    metrics_path = '../datasets/Day3_Neuron_Calcium_Metrics.xlsx'
    behavior_path = '../datasets/Day3_with_behavior_labels_filled.xlsx'
    
    # Load and process data
    metrics_df, behavioral_df = load_data(metrics_path, behavior_path)
    neuron_mapping = create_neuron_mapping(metrics_df)
    unique_neurons = list(neuron_mapping.keys())
    
    # Create and display plot
    plot_activity_raster(metrics_df, behavioral_df, neuron_mapping, unique_neurons)
    plt.show()
    
    # Uncomment below lines to save the plot
    # output_image_path = './data/Time_Activity_Raster_Plot_GMM_Set3_with_behavior.png'
    # plt.savefig(output_image_path)
    # print(f"Image saved to: {output_image_path}")


if __name__ == "__main__":
    main()
